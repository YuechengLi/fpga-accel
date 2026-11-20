//=============================================================================
// frontend.sv -- Stage 7: front-end video preprocessing module.
//
// The whole-model I/O front-end that sits before the unified engine
// (video in -> front-end -> VDMA/DDR -> main accelerator) and, on the
// output side, after it (accelerator -> VDMA/DDR -> front-end -> video out).
// Derived from the frozen reference conv2d_stream (~/workspace/fpga-2d-conv/,
// Verilog, max 9x9, same-size in/out) and from conv2d_dense (Stage 1, SV):
// same streaming structure (BRAM line buffers + alignment delay + register
// window + MAC + shift/saturate), same config bus and stream interface.
//
// What it does, per stream position (one spatial position per cycle, PX=1):
//   * 1 or 3 input channels (runtime config 0x0D), processed in parallel
//     lanes that share one coefficient bank (blur / sharpen / edge /
//     identity are channel-independent filters). The stream group is
//     24 bits (3 x 8-bit); for 1ch only byte 0 is used and the upper
//     output bytes are forced to 0.
//   * Small kernel K=1/3/5 (25 multipliers/lane, 75 total), 16-bit signed
//     coefficients via the config bus (5x5 grid, row stride 5).
//   * Downsampling DS=1/2/4 (config 0x0C): the kernel is the anti-alias
//     filter, and DS folds into the output decimation, so the total
//     decimation is stride * DS per axis. DS=1 is passthrough resolution.
//   * Arbitrary per-side padding (pad_t/pad_b/pad_l/pad_r, each 0..K,
//     zero fill, physically in the stream) and arbitrary per-axis stride
//     (stride_x/stride_y, each 1..K, decimate -- every position is
//     computed, out_valid is asserted only at stride positions).
//   * 32-bit accumulator, shift + saturate output stage, fused ReLU/Leaky
//     in the output stage (config 0x0A/0x0B, same as conv2d_mc /
//     conv2d_dense Stage 4), signed intermediates (config 0x0F, bit0
//     signed in, bit1 signed out; sign-extend MAC, saturate [-128,127]).
//   * No frame buffering: only (K-1)=4 BRAM line buffers + the 5x5
//     register window. Pipeline latency 11 cycles (line-buffer + alignment
//     delay 4, window 1, prod 1, sum2 1, sum4 1, acc 1, shift 1, output 1);
//     the first output additionally waits for the (K-1)-row line-buffer
//     fill (~4 rows, ~26 us at 4K60 -- under 0.2% of a frame).
//
// Output size: out_W = (W + pad_l + pad_r - K)/(Sx*DS) + 1,
//              out_H = (H + pad_t + pad_b - K)/(Sy*DS) + 1.
// The DUT never divides: decimation uses the "next exceeds" comparisons,
// so no divider is synthesized (same as Stage 1).
//
// Config bus (write between frames; single-cycle writes):
//   addr 0x00 : kernel size (1/3/5)         addr 0x01 : output shift (0..15)
//   addr 0x02 : image width  (1..4095)      addr 0x03 : image height (1..4095)
//   addr 0x04 : pad_top    (0..K)           addr 0x05 : pad_bottom (0..K)
//   addr 0x06 : pad_left   (0..K)           addr 0x07 : pad_right  (0..K)
//   addr 0x08 : stride_x   (1..K)           addr 0x09 : stride_y   (1..K)
//   addr 0x0A : fused act (0=none, 1=ReLU, 2=Leaky)
//   addr 0x0B : Leaky slope_fp (Q0.16)
//   addr 0x0C : downsample factor (1/2/4)   addr 0x0D : channels (1 or 3)
//   addr 0x0F : sign (bit0 signed in, bit1 signed out)
//   addr 0x10+i : coeff[i], i = 0..24, row-major, row stride 5.
// Reset defaults: ksize=3, shift=8, W=640, H=480, all pads 0,
//   stride_x=stride_y=1, downsample=1, channels=3,
//   sign 0, fused act 0, slope 0.125,
//   3x3 identity (centre tap 256 at index 1*5+1=6).
// No bias register (preprocessing kernels -- blur/sharpen/edge/identity --
// do not use a bias; the engine's conv2d_mc has one if a bias is needed).
// PX=1 spatially (one position/cycle); the plan's PX=2 is left for a later
// stage, as in Stage 1.
//=============================================================================
`timescale 1ns/1ps

package frontend_pkg;
    localparam int MAX_K     = 5;
    localparam int MAX_COEFF = 25;   // 5x5
    localparam int MAX_CH    = 3;
    localparam int MAX_W_DEFAULT = 3840;

    localparam logic [7:0] CFG_KSIZE  = 8'h00;
    localparam logic [7:0] CFG_SHIFT  = 8'h01;
    localparam logic [7:0] CFG_WIDTH  = 8'h02;
    localparam logic [7:0] CFG_HEIGHT = 8'h03;
    localparam logic [7:0] CFG_PAD_T  = 8'h04;
    localparam logic [7:0] CFG_PAD_B  = 8'h05;
    localparam logic [7:0] CFG_PAD_L  = 8'h06;
    localparam logic [7:0] CFG_PAD_R  = 8'h07;
    localparam logic [7:0] CFG_STRIDE_X = 8'h08;
    localparam logic [7:0] CFG_STRIDE_Y = 8'h09;
    localparam logic [7:0] CFG_ACT     = 8'h0A;  // fused act: 0=none, 1=ReLU, 2=Leaky
    localparam logic [7:0] CFG_ACT_SLOPE = 8'h0B;  // Leaky slope_fp (Q0.16)
    localparam logic [7:0] CFG_DS      = 8'h0C;  // downsample factor (1/2/4)
    localparam logic [7:0] CFG_CH      = 8'h0D;  // channels (1 or 3)
    localparam logic [7:0] CFG_SIGN    = 8'h0F;  // bit0: signed in, bit1: signed out
    localparam logic [7:0] CFG_COEFF_BASE = 8'h10;
    localparam logic [7:0] CFG_COEFF_LAST = 8'h28;  // 0x10 + 24

    typedef logic [7:0]         pixel_t;
    typedef logic signed [15:0] coeff_t;
endpackage : frontend_pkg

//-----------------------------------------------------------------------------
// One row delay in BRAM, 24-bit wide (3 channels together). Read-first:
// returns the group written one full row ago. Pointer/reset in one block,
// memory access in its own reset-free block, or Yosys will not infer a
// block RAM. (Same template as conv2d_dense's line_buffer_bram_dense.)
//-----------------------------------------------------------------------------
module line_buffer_bram_fe #(
    parameter int WIDTH  = 24,
    parameter int DEPTH  = 3872,
    parameter int ADDR_W = 12
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             en,
    input  logic [WIDTH-1:0] din,
    input  logic [ADDR_W-1:0] wrap,
    output logic [WIDTH-1:0] dout
);
    (* ram_style = "block" *) logic [WIDTH-1:0] mem [0:DEPTH-1];
    logic [ADDR_W-1:0] ptr;

    // Wrap test is >= (not ==): if the pointer ever drifts to >= wrap
    // (e.g. pw shrinks across frames while the pointer holds a larger
    // value), it recovers to 0 in one cycle instead of counting up to
    // 2^ADDR_W. Identical to == in normal operation. (Stage 2 bug, same
    // fix: the == form broke case 8/13/14, which run after a larger-pw
    // case; they passed in isolation.)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ptr <= '0;
        else if (en) begin
            if (ptr >= wrap - 1'b1) ptr <= '0;
            else                     ptr <= ptr + 1'b1;
        end
    end
    always_ff @(posedge clk) begin
        if (en) begin
            dout <= mem[ptr];
            mem[ptr] <= din;
        end
    end
endmodule : line_buffer_bram_fe

//-----------------------------------------------------------------------------
// Main module
//-----------------------------------------------------------------------------
module frontend #(
    // Yosys does not accept a package-scoped (::) default here, so the
    // default is a literal; it matches frontend_pkg::MAX_W_DEFAULT.
    parameter int MAX_W = 3840
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (one spatial position per cycle, 3 x 8-bit channels)
    input  logic [23:0] pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (one spatial position per cycle, 3 x 8-bit channels)
    output logic [23:0] pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    // Config-bus addresses (mirrored in frontend_pkg above; kept as
    // module localparams because Yosys does not accept `import pkg::*`).
    localparam logic [7:0] CFG_KSIZE  = 8'h00;
    localparam logic [7:0] CFG_SHIFT  = 8'h01;
    localparam logic [7:0] CFG_WIDTH  = 8'h02;
    localparam logic [7:0] CFG_HEIGHT = 8'h03;
    localparam logic [7:0] CFG_PAD_T  = 8'h04;
    localparam logic [7:0] CFG_PAD_B  = 8'h05;
    localparam logic [7:0] CFG_PAD_L  = 8'h06;
    localparam logic [7:0] CFG_PAD_R  = 8'h07;
    localparam logic [7:0] CFG_STRIDE_X = 8'h08;
    localparam logic [7:0] CFG_STRIDE_Y = 8'h09;
    localparam logic [7:0] CFG_ACT     = 8'h0A;  // fused act: 0=none, 1=ReLU, 2=Leaky
    localparam logic [7:0] CFG_ACT_SLOPE = 8'h0B;  // Leaky slope_fp (Q0.16)
    localparam logic [7:0] CFG_DS      = 8'h0C;  // downsample factor (1/2/4)
    localparam logic [7:0] CFG_CH      = 8'h0D;  // channels (1 or 3)
    localparam logic [7:0] CFG_SIGN    = 8'h0F;  // bit0: signed in, bit1: signed out
    localparam logic [7:0] CFG_COEFF_BASE = 8'h10;
    localparam logic [7:0] CFG_COEFF_LAST = 8'h28;  // 0x10 + 24

    localparam int MAX_COEFF = 25;   // 5x5 (mirrors the package)
    localparam int MAX_CH    = 3;    // 3 parallel lanes
    // Padded row can exceed MAX_W by up to 2*5 (pad_l + pad_r).
    localparam int LB_DEPTH = MAX_W + 32;
    localparam int ADDR_W   = (LB_DEPTH <= 2048) ? 11 :
                              (LB_DEPTH <= 4096) ? 12 : 13;
    localparam int MAX_K_L  = 5;

    typedef logic [23:0] group_t;

    //------------------------- config registers ----------------------------
    logic [3:0]  ksize;
    logic [3:0]  shift_r;
    logic [11:0] width_r, height_r;
    logic [3:0]  pad_t, pad_b, pad_l, pad_r;
    logic [3:0]  stride_x, stride_y;
    logic [2:0]  ds_r;               // downsample factor (1/2/4)
    logic [1:0]  ch_r;               // channels (1 or 3)
    logic        sign_in_r, sign_out_r;
    logic [1:0]  fuse_act_r;         // 0=none, 1=ReLU, 2=Leaky
    logic [15:0] act_slope_r;        // Leaky slope_fp (Q0.16)
    logic signed [15:0] coeff [0:MAX_COEFF-1];
    int ci;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ksize    <= 4'd3;
            shift_r  <= 4'd8;
            width_r  <= 12'd640;
            height_r <= 12'd480;
            pad_t <= 4'd0; pad_b <= 4'd0; pad_l <= 4'd0; pad_r <= 4'd0;
            stride_x <= 4'd1; stride_y <= 4'd1;
            ds_r <= 3'd1;
            ch_r <= 2'd3;
            sign_in_r <= 1'b0; sign_out_r <= 1'b0;
            fuse_act_r <= 2'd0; act_slope_r <= 16'd8192;
            for (ci = 0; ci < MAX_COEFF; ci++) coeff[ci] <= '0;
            // default: 3x3 identity (centre 256, shift 8). The 3x3 kernel
            // occupies the top-left 3x3 of the 5x5 coeff grid, so its
            // centre tap is at index 1*5+1 = 6.
            coeff[1*5+1] <= 16'sd256;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_KSIZE : ksize    <= cfg_data[3:0];
                CFG_SHIFT : shift_r  <= cfg_data[3:0];
                CFG_WIDTH : width_r  <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                CFG_PAD_T : pad_t    <= cfg_data[3:0];
                CFG_PAD_B : pad_b    <= cfg_data[3:0];
                CFG_PAD_L : pad_l    <= cfg_data[3:0];
                CFG_PAD_R : pad_r    <= cfg_data[3:0];
                CFG_STRIDE_X: stride_x <= cfg_data[3:0];
                CFG_STRIDE_Y: stride_y <= cfg_data[3:0];
                CFG_ACT   : fuse_act_r <= cfg_data[1:0];
                CFG_ACT_SLOPE: act_slope_r <= cfg_data;
                CFG_DS    : ds_r     <= cfg_data[2:0];
                CFG_CH    : ch_r     <= cfg_data[1:0];
                CFG_SIGN  : begin
                    sign_in_r <= cfg_data[0];
                    sign_out_r <= cfg_data[1];
                end
                default: begin
                    if (cfg_addr >= CFG_COEFF_BASE && cfg_addr <= CFG_COEFF_LAST)
                        coeff[cfg_addr - CFG_COEFF_BASE] <= cfg_data;
                end
            endcase
        end
    end

    //------------------------- derived geometry ---------------------------
    // Padded image size (pixels). The output size is
    //   out_W = (pw - K)/eff_Sx + 1, out_H = (ph - K)/eff_Sy + 1
    // with eff_Sx = stride_x * DS, eff_Sy = stride_y * DS; the DUT never
    // divides: the decimation uses the "next exceeds" comparisons
    // (nxt_sc + eff_Sx > pw - 1, nxt_sr + eff_Sy > ph - 1), so no divider
    // is synthesized. The testbench computes out_W/out_H itself.
    logic [12:0] pw;      // padded width  = W + pad_l + pad_r
    logic [12:0] ph;      // padded height = H + pad_t + pad_b
    assign pw = {1'b0, width_r} + {9'd0, pad_l} + {9'd0, pad_r};
    assign ph = {1'b0, height_r} + {9'd0, pad_t} + {9'd0, pad_b};

    // Effective decimation per axis (conv stride x downsample factor).
    // stride 1..5, DS 1/2/4 -> eff 1..20.
    logic [5:0] eff_sx, eff_sy;
    assign eff_sx = {2'b0, stride_x} * {3'b0, ds_r};
    assign eff_sy = {2'b0, stride_y} * {3'b0, ds_r};

    //------------------------- enable shift register ----------------------
    // en_d[i] = stream_en delayed by i cycles. The window loads on en_d[4]
    // (4 = line-buffer + alignment delay); the line buffers use en_d[0..3].
    logic [4:0] en_shreg;
    logic [4:0] en_d;

    //------------------------- stream control ------------------------------
    // The stream walks the PADDED image, (strm_r, strm_c) in raster order.
    // A stream position is "real" when it maps inside the input image;
    // otherwise the DUT feeds a zero (padding). The testbench sends only
    // the raw W*H positions; in_ready is high only at real positions.
    logic        frame_active;
    logic [12:0] strm_r, strm_c;
    logic [12:0] next_c;
    logic [12:0] next_r;
    logic is_last_stream;
    logic pos_is_real;
    logic stall;
    logic start_frame;
    logic stream_en;
    logic pipe_empty;
    logic can_start;
    group_t stream_data;

    assign next_c = (strm_c == pw - 13'd1) ? 13'd0 : strm_c + 13'd1;
    assign next_r = (strm_c == pw - 13'd1) ? strm_r + 13'd1 : strm_r;
    assign is_last_stream = (strm_r == ph - 13'd1) && (strm_c == pw - 13'd1);
    assign pos_is_real = (strm_r >= {9'd0, pad_t}) &&
                         (strm_r <  {9'd0, pad_t} + {1'b0, height_r}) &&
                         (strm_c >= {9'd0, pad_l}) &&
                         (strm_c <  {9'd0, pad_l} + {1'b0, width_r});

    // pipeline valid flags; empty when all are 0
    logic win_v, prod_v, sum2_v, sum4_v, acc_v, shift_v;
    assign pipe_empty = !win_v && !prod_v && !sum2_v && !sum4_v &&
                        !acc_v && !shift_v && !out_valid;
    assign can_start  = !frame_active && pipe_empty;
    assign start_frame = can_start && in_valid && in_sof;
    assign stall = out_valid && !out_ready;
    assign stream_en = !stall && (start_frame ||
                         (frame_active && (!pos_is_real || in_valid)));
    assign stream_data = pos_is_real ? pixel_in : '0;
    assign in_ready = !stall && pos_is_real;
    assign busy = !can_start;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_active <= 1'b0;
            strm_r <= 13'd0;
            strm_c <= 13'd0;
        end else if (stream_en) begin
            if (is_last_stream) begin
                frame_active <= 1'b0;
                strm_r <= 13'd0;
                strm_c <= 13'd0;
            end else begin
                frame_active <= 1'b1;
                strm_r <= next_r;
                strm_c <= next_c;
            end
        end
    end

    //------------------------- next-output (decimation) counters ----------
    // The next output (nxt_or, nxt_oc) occurs at stream position
    // (nxt_sr, nxt_sc) = (nxt_or*eff_Sy + K - 1, nxt_oc*eff_Sx + K - 1).
    // Counters advance only when a window load matches (nxt_sr, nxt_sc);
    // the row wraps when the next column would exceed the padded row
    // (nxt_sc + eff_Sx > pw - 1), which is exactly out_w - 1. No divider.
    logic [12:0] nxt_or, nxt_oc, nxt_sr, nxt_sc;
    logic        match_out;
    logic [12:0] mac_r, mac_c;   // aligned stream position (window load)
    assign match_out = (mac_r == nxt_sr) && (mac_c == nxt_sc);
    logic is_eol_c, is_last_c;
    assign is_eol_c  = match_out && ((nxt_sc + {7'd0, eff_sx}) > (pw - 13'd1));
    assign is_last_c = is_eol_c  && ((nxt_sr + {7'd0, eff_sy}) > (ph - 13'd1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= 13'd2; nxt_sc <= 13'd2;  // K-1 for default K=3
        end else if (start_frame) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= {9'd0, ksize} - 13'd1;
            nxt_sc <= {9'd0, ksize} - 13'd1;
        end else if (!stall && en_d[4] && match_out) begin
            if ((nxt_sc + {7'd0, eff_sx}) <= (pw - 13'd1)) begin
                nxt_oc <= nxt_oc + 13'd1;
                nxt_sc <= nxt_sc + {7'd0, eff_sx};
            end else begin
                nxt_oc <= 13'd0;
                nxt_sc <= {9'd0, ksize} - 13'd1;
                nxt_or <= nxt_or + 13'd1;
                nxt_sr <= nxt_sr + {7'd0, eff_sy};
            end
        end
    end

    //------------------------- enable / coordinate delay lines -----------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) en_shreg <= 5'd0;
        else if (!stall) en_shreg <= {en_shreg[3:0], stream_en};
    end
    assign en_d = {en_shreg[3:0], stream_en};

    // coordinate delay line, depth 4: coords of the stream position
    // accepted 4 cycles ago (the aligned position for the window load).
    logic [12:0] cr_d [0:3];
    logic [12:0] cc_d [0:3];
    int di;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (di = 0; di < 4; di++) begin
                cr_d[di] <= 13'd0; cc_d[di] <= 13'd0;
            end
        end else if (!stall) begin
            cr_d[0] <= stream_en ? strm_r : 13'd0;
            cc_d[0] <= stream_en ? strm_c : 13'd0;
            for (di = 1; di < 4; di++) begin
                cr_d[di] <= cr_d[di-1];
                cc_d[di] <= cc_d[di-1];
            end
        end
    end
    assign mac_r = cr_d[3];
    assign mac_c = cc_d[3];

    //------------------------- line buffers (BRAM, 24-bit) ----------------
    // 4 line buffers (K-1 = 4 for K=5), each holding one full padded row
    // of 3-channel groups. The delay per row is (4-k) [line buffer] + k
    // [alignment] = 4 cycles for every row.
    group_t lb_out [0:3];
    group_t lb_in  [0:3];
    assign lb_in[0] = stream_data;
    genvar gi;
    generate
        for (gi = 0; gi < 4; gi++) begin : lb_chain
            if (gi > 0) assign lb_in[gi] = lb_out[gi-1];
            line_buffer_bram_fe #(.WIDTH(24), .DEPTH(LB_DEPTH),
                                  .ADDR_W(ADDR_W)) u_lb (
                .clk(clk), .rst_n(rst_n),
                .en(en_d[gi] && !stall),
                .din(lb_in[gi]),
                .wrap(pw[ADDR_W-1:0]),
                .dout(lb_out[gi])
            );
        end
    endgenerate

    //------------------------- alignment delay lines ---------------------
    // row k (0..4): k=4 is the current stream row, k=4-i is lb_i.
    // Row k is delayed by k cycles so all 5 taps of a column arrive
    // together (total delay per row = (4-k) [line buffer] + k = 4).
    group_t row_in [0:4];
    assign row_in[4] = stream_data;
    generate
        for (gi = 0; gi < 4; gi++) begin : row_assign
            assign row_in[gi] = lb_out[3-gi];
        end
    endgenerate

    group_t aln [0:4][0:3];
    int ai, aj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ai = 0; ai < 5; ai++)
                for (aj = 0; aj < 4; aj++) aln[ai][aj] <= '0;
        end else if (!stall) begin
            for (ai = 0; ai < 5; ai++) begin
                aln[ai][0] <= row_in[ai];
                for (aj = 1; aj < 4; aj++) aln[ai][aj] <= aln[ai][aj-1];
            end
        end
    end
    group_t acol [0:4];
    generate
        for (gi = 0; gi < 5; gi++) begin : acol_assign
            if (gi == 0) assign acol[gi] = row_in[gi];
            else         assign acol[gi] = aln[gi][gi-1];
        end
    endgenerate

    //------------------------- 5 x 5 window (24-bit groups) ---------------
    // Window col wc holds stream group (s_r, s_c + wc - 4) for the row.
    // The window shifts left by one group per cycle and loads the new
    // group into col 4. The bottom-right corner (win[4][4]) is the current
    // stream group. Lane c uses byte c of each group: tap (i,j) for lane c
    // reads win[5-K+i][5-K+j][8*c +: 8] with coeff[i*5+j]. No row/col-valid
    // masks are needed: padding is physically in the stream (zeros), and
    // the window resets to 0, so taps outside the padded image read 0.
    group_t win [0:4][0:4];
    int wi, wj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wi = 0; wi < 5; wi++)
                for (wj = 0; wj < 5; wj++) win[wi][wj] <= '0;
        end else if (!stall && en_d[4]) begin
            for (wi = 0; wi < 5; wi++) begin
                for (wj = 0; wj < 4; wj++) win[wi][wj] <= win[wi][wj+1];
                win[wi][4] <= acol[wi];
            end
        end
    end

    // window coordinates + output flags, captured when the window loads.
    // is_out / out_r / out_c / is_eol / is_last ride the pipeline to the
    // output stage.
    logic        win_is_out;
    logic [12:0] win_or, win_oc;
    logic        win_eol, win_last;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win_v <= 1'b0; win_is_out <= 1'b0;
            win_or <= 13'd0; win_oc <= 13'd0;
            win_eol <= 1'b0; win_last <= 1'b0;
        end else if (!stall) begin
            win_v <= en_d[4];
            win_is_out <= en_d[4] && match_out;
            win_or <= nxt_or;
            win_oc <= nxt_oc;
            win_eol <= en_d[4] && is_eol_c;
            win_last <= en_d[4] && is_last_c;
        end
    end

    //------------------------- shared pipeline control -------------------
    // Valid flags and output coords are SHARED across lanes. They advance
    // in their own always_ff; the lane generate below touches datapath only.
    logic        p_is_out, s2_is_out, s4_is_out, a_is_out, sh_is_out;
    logic [12:0] p_or, p_oc, s2_or, s2_oc, s4_or, s4_oc,
                 a_or, a_oc, sh_or, sh_oc;
    logic        p_eol, p_last, s2_eol, s2_last, s4_eol, s4_last,
                 a_eol, a_last, sh_eol, sh_last;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_v <= 1'b0; p_is_out <= 1'b0; p_or <= 13'd0; p_oc <= 13'd0;
            p_eol <= 1'b0; p_last <= 1'b0;
            sum2_v <= 1'b0; s2_is_out <= 1'b0; s2_or <= 13'd0; s2_oc <= 13'd0;
            s2_eol <= 1'b0; s2_last <= 1'b0;
            sum4_v <= 1'b0; s4_is_out <= 1'b0; s4_or <= 13'd0; s4_oc <= 13'd0;
            s4_eol <= 1'b0; s4_last <= 1'b0;
            acc_v  <= 1'b0; a_is_out <= 1'b0; a_or <= 13'd0; a_oc <= 13'd0;
            a_eol <= 1'b0; a_last <= 1'b0;
            shift_v<= 1'b0; sh_is_out <= 1'b0; sh_or <= 13'd0; sh_oc <= 13'd0;
            sh_eol <= 1'b0; sh_last <= 1'b0;
        end else if (!stall) begin
            prod_v <= win_v;  p_is_out <= win_is_out;
            p_or <= win_or;   p_oc <= win_oc;
            p_eol <= win_eol; p_last <= win_last;
            sum2_v <= prod_v; s2_is_out <= p_is_out;
            s2_or <= p_or;    s2_oc <= p_oc;
            s2_eol <= p_eol;  s2_last <= p_last;
            sum4_v <= sum2_v; s4_is_out <= s2_is_out;
            s4_or <= s2_or;   s4_oc <= s2_oc;
            s4_eol <= s2_eol; s4_last <= s2_last;
            acc_v  <= sum4_v; a_is_out <= s4_is_out;
            a_or <= s4_or;    a_oc <= s4_oc;
            a_eol <= s4_eol;  a_last <= s4_last;
            shift_v<= acc_v;  sh_is_out <= a_is_out;
            sh_or <= a_or;    sh_oc <= a_oc;
            sh_eol <= a_eol;  sh_last <= a_last;
        end
    end

    // Per-lane shift outputs (registered), read by the saturate/output stage.
    logic signed [31:0] shifted [0:MAX_CH-1];

    //------------------------- MAC: 3 parallel lanes ----------------------
    // Each lane (one channel): 25 multipliers + 2-stage adder tree +
    // split shift stage. Tap (i,j) for lane c: window [5-K+i][5-K+j],
    // byte c, coeff[i*5+j]. Max |acc| = 25*255*32768 ~ 2.09e8 < 2^31.
    // Tree (25 products, 25-bit): 25->13->7 (27b) -> 4->2 (29b) -> 1 (32b).
    genvar c;
    generate
        for (c = 0; c < MAX_CH; c++) begin : lane
            logic signed [24:0] prod_comb [0:24];
            logic signed [24:0] prod [0:24];
            logic signed [26:0] s1_comb [0:12];
            logic signed [26:0] sum2_comb [0:6];
            logic signed [26:0] sum2 [0:6];
            logic signed [28:0] s3_comb [0:3];
            logic signed [28:0] sum4_comb [0:1];
            logic signed [28:0] sum4 [0:1];
            logic signed [31:0] acc;
            logic signed [31:0] rnd;
            int li, lj, lk, wr, wc, qi, si;

            // multiply stage (combinational)
            always @* begin
                for (li = 0; li < 5; li++) begin
                    for (lj = 0; lj < 5; lj++) begin
                        lk = li*5 + lj;
                        wr = 5 - ksize + li;
                        wc = 5 - ksize + lj;
                        if ((li < ksize) && (lj < ksize))
                            prod_comb[lk] = (sign_in_r ?
                                $signed(win[wr][wc][8*c +: 8]) :
                                $signed({1'b0, win[wr][wc][8*c +: 8]})) * coeff[lk];
                        else
                            prod_comb[lk] = 25'sd0;
                    end
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (qi = 0; qi < 25; qi++) prod[qi] <= 25'sd0;
                end else if (!stall) begin
                    for (qi = 0; qi < 25; qi++) prod[qi] <= prod_comb[qi];
                end
            end

            // adder tree stage A: 25 -> 13 -> 7 (27-bit)
            always @* begin
                for (si = 0; si < 12; si++) begin
                    s1_comb[si] = prod[2*si] + prod[2*si+1];
                end
                s1_comb[12] = prod[24];
                for (si = 0; si < 6; si++) begin
                    sum2_comb[si] = s1_comb[2*si] + s1_comb[2*si+1];
                end
                sum2_comb[6] = s1_comb[12];
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 7; si++) sum2[si] <= 27'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 7; si++) sum2[si] <= sum2_comb[si];
                end
            end

            // adder tree stage B: 7 -> 4 -> 2 (29-bit)
            always @* begin
                for (si = 0; si < 3; si++) begin
                    s3_comb[si] = sum2[2*si] + sum2[2*si+1];
                end
                s3_comb[3] = sum2[6];
                for (si = 0; si < 2; si++) begin
                    sum4_comb[si] = s3_comb[2*si] + s3_comb[2*si+1];
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 2; si++) sum4[si] <= 29'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 2; si++) sum4[si] <= sum4_comb[si];
                end
            end

            // accumulator: 2 -> 1 (32-bit)
            logic signed [31:0] acc_comb;
            assign acc_comb = sum4[0] + sum4[1];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    acc <= 32'sd0;
                end else if (!stall) begin
                    acc <= acc_comb;
                end
            end

            // output stage 1: shift (barrel shifter, split from saturate)
            always @* begin
                rnd = acc + ((shift_r == 0) ? 32'sd0 : (32'sd1 <<< (shift_r - 1'b1)));
            end
            logic signed [31:0] sh_comb;
            assign sh_comb = rnd >>> shift_r;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    shifted[c] <= 32'sd0;
                end else if (!stall) begin
                    shifted[c] <= sh_comb;
                end
            end
        end
    endgenerate

    //------------------------- output stage 2: fused act + saturate -----
    // Fused ReLU/Leaky (Stage 4 convention): applied to the shifted
    // accumulator value, BEFORE saturation. ReLU is a comparator + mux;
    // Leaky needs one 32x16 multiply per lane (48-bit product, take bits
    // [47:16] = (shifted * slope_fp) >> 16). The separate act module
    // (bypass mode) is reserved for complex activations.
    // Lanes >= channels are forced to 0 (1ch mode: only byte 0 carries data).
    logic signed [31:0] act_val [0:MAX_CH-1];
    logic signed [47:0] leaky_prod [0:MAX_CH-1];
    int fi;
    always @* begin
        for (fi = 0; fi < MAX_CH; fi++) begin
            leaky_prod[fi] = shifted[fi] *
                $signed({{16{act_slope_r[15]}}, act_slope_r});
            if (fuse_act_r == 2'd1)
                act_val[fi] = (shifted[fi] < 32'sd0) ? 32'sd0 : shifted[fi];
            else if (fuse_act_r == 2'd2)
                act_val[fi] = (shifted[fi] < 32'sd0) ?
                    leaky_prod[fi][47:16] : shifted[fi];
            else
                act_val[fi] = shifted[fi];
        end
    end

    logic [7:0] px_sat [0:MAX_CH-1];
    int oi;
    always @* begin
        for (oi = 0; oi < MAX_CH; oi++) begin
            if (sign_out_r) begin
                if (act_val[oi] > 32'sd127)       px_sat[oi] = 8'd127;
                else if (act_val[oi] < -32'sd128) px_sat[oi] = 8'h80;
                else                                px_sat[oi] = act_val[oi][7:0];
            end else begin
                if (act_val[oi] > 32'sd255)      px_sat[oi] = 8'd255;
                else if (act_val[oi] < 32'sd0)   px_sat[oi] = 8'd0;
                else                               px_sat[oi] = act_val[oi][7:0];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= '0;
            out_valid <= 1'b0; out_sof <= 1'b0;
            out_eol <= 1'b0;    out_last <= 1'b0;
        end else if (!stall) begin
            for (oi = 0; oi < MAX_CH; oi++) begin
                pixel_out[8*oi +: 8] <= (oi < ch_r) ? px_sat[oi] : 8'd0;
            end
            out_valid <= shift_v && sh_is_out;
            out_sof  <= shift_v && sh_is_out && (sh_or == 13'd0) && (sh_oc == 13'd0);
            out_eol  <= shift_v && sh_is_out && sh_eol;
            out_last <= shift_v && sh_is_out && sh_last;
        end
    end
endmodule : frontend
