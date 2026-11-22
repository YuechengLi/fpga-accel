//=============================================================================
// conv2d_dense.sv -- Stage 1: dense 2-D convolution, kernel up to 11x11,
// arbitrary per-side padding and per-axis stride (decimate), SystemVerilog.
//
// Streaming, no frame buffer. One unified datapath:
//   * Input  : 8*PX-bit pixel groups, raster order, one group/cycle when
//              in_valid. in_sof marks the first group of a frame.
//              The DUT expands the input to the PADDED image on the fly
//              (zero fill); the testbench sends only the raw W*H image.
//   * Output : 8*PX-bit pixel groups, raster order, out_valid[PX-1:0] is a
//              per-pixel valid mask; out_sof / out_eol / out_last.
//              Output size out_W = (W + pad_l + pad_r - K)/S_x + 1,
//              out_H = (H + pad_t + pad_b - K)/S_y + 1.
//              Stride is DECIMATE: every position is computed, out_valid is
//              asserted only at stride positions.
//   * Kernel : odd size 1/3/5/7/9/11 (max 11x11 = 121 multipliers/lane),
//              16-bit signed coefficients, loaded at run time through the
//              config bus (row-major, stride 11, top-left KxK is the kernel).
//              out = clamp((acc + round) >> shift), 32-bit accumulator.
//   * 10 line buffers (BRAM) + an 11 x (11+PX-1) register window.
//     The window's bottom-right corner is the current stream pixel, so the
//     KxK kernel is the bottom-right KxK sub-window: tap (i,j) reads
//     win[11-K+i][11-K+p+j] and coeff[i*11+j]. No row/col-valid masks are
//     needed: padding is physically in the stream (zeros), and the window
//     resets to 0, so taps outside the padded image read 0.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//   * Pipeline: window -> prod -> sum2 -> sum4 -> sum6 -> acc -> shift ->
//     saturate/output. Latency 24 cycles (PX=1).
//
// Config bus (write between frames; single-cycle writes):
//   addr 0x00 : kernel size (1/3/5/7/9/11)   addr 0x01 : output shift (0..15)
//   addr 0x02 : image width  (1..4095)       addr 0x03 : image height (1..4095)
//   addr 0x04 : pad_top    (0..K)            addr 0x05 : pad_bottom (0..K)
//   addr 0x06 : pad_left   (0..K)            addr 0x07 : pad_right  (0..K)
//   addr 0x08 : stride_x   (1..K)            addr 0x09 : stride_y   (1..K)
//   addr 0x10+i : coeff[i], i = 0..120, row-major, row stride 11.
//   addr 0x0A : fused act (0=none, 1=ReLU, 2=Leaky) -- Stage 4
//   addr 0x0B : Leaky slope_fp (Q0.16) -- Stage 4
//   addr 0x0F : sign (bit0 signed in, bit1 signed out) -- Stage 4
// Reset defaults: ksize=3, shift=8, W=640, H=480, all pads 0,
//   stride_x=stride_y=1, 3x3 identity (centre tap 256 at index 1*11+1=12),
//   sign 0, fused act 0, slope 0.125.
// Stage 4 additions (mirroring conv2d_mc / conv2d_dw): signed
// intermediates (sign-extend MAC when sign_in, saturate to [-128,127]
// when sign_out; zero padding stays correct because padded 0 is 0.0 in
// signed too) and fused ReLU/Leaky in the output stage (applied to the
// shifted accumulator BEFORE saturation; ReLU is a comparator + mux,
// Leaky needs one 32x16 multiply, bits [47:16]). The fused act is what
// lets the unified engine cover the FC/MLP path: FC reuses this 1x1
// datapath, so a ReLU after an MLP fuses here exactly as after a conv.
// The separate act module (bypass mode) is reserved for complex
// activations (sigmoid/tanh/GELU/SiLU).
//
// Stage 1 verifies PX=1. PX is a parameter (window/MAC generalize), but the
// stride-decimation / framing logic is written and verified for PX=1; PX=2
// with arbitrary padding/stride is left for a later stage.
//=============================================================================
`timescale 1ns/1ps

package conv2d_dense_pkg;
    localparam int MAX_K     = 11;
    localparam int MAX_COEFF = 121;   // 11x11
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
    localparam logic [7:0] CFG_SIGN    = 8'h0F;  // bit0: signed in, bit1: signed out
    localparam logic [7:0] CFG_COEFF_BASE = 8'h10;
    localparam logic [7:0] CFG_COEFF_LAST = 8'h88;  // 0x10 + 120

    typedef logic [7:0]         pixel_t;
    typedef logic signed [15:0] coeff_t;
endpackage : conv2d_dense_pkg

//-----------------------------------------------------------------------------
// One row delay in BRAM, 8*PX-bit wide. Read-first: returns the group
// written one full row ago. Pointer/reset in one block, memory access in
// its own reset-free block, or Yosys will not infer a block RAM.
// (Same template as the reference conv2d_stream_px.)
//-----------------------------------------------------------------------------
module line_buffer_bram_dense #(
    parameter int WIDTH  = 8,
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ptr <= '0;
        else if (en) begin
            if (ptr == wrap - 1'b1) ptr <= '0;
            else                     ptr <= ptr + 1'b1;
        end
    end
    always_ff @(posedge clk) begin
        if (en) begin
            dout <= mem[ptr];
            mem[ptr] <= din;
        end
    end
endmodule : line_buffer_bram_dense

//-----------------------------------------------------------------------------
// Main module
//-----------------------------------------------------------------------------
module conv2d_dense #(
    parameter int PX    = 1,
    // Yosys does not accept a package-scoped (::) default here, so the
    // default is a literal; it matches conv2d_dense_pkg::MAX_W_DEFAULT.
    parameter int MAX_W = 3840
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (pixel groups)
    input  logic [8*PX-1:0] pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (pixel groups)
    output logic [8*PX-1:0] pixel_out,
    output logic [PX-1:0]   out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    // Config-bus addresses (mirrored in conv2d_dense_pkg above; kept as
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
    localparam logic [7:0] CFG_SIGN    = 8'h0F;  // bit0: signed in, bit1: signed out
    localparam logic [7:0] CFG_COEFF_BASE = 8'h10;
    localparam logic [7:0] CFG_COEFF_LAST = 8'h88;  // 0x10 + 120

    localparam int LOG2_PX  = (PX <= 1) ? 0 : (PX <= 2) ? 1 : 2;
    localparam int MAX_COEFF = 121;   // 11x11 (mirrors the package)
    // Padded row can exceed MAX_W by up to 2*11 (pad_l + pad_r).
    localparam int LB_DEPTH = MAX_W + 32;
    localparam int ADDR_W   = (LB_DEPTH <= 2048) ? 11 :
                              (LB_DEPTH <= 4096) ? 12 : 13;
    localparam int WIN_W    = 11 + PX - 1;   // window columns
    localparam int MAX_K_L  = 11;

    typedef logic [8*PX-1:0] group_t;

    //------------------------- config registers ----------------------------
    logic [3:0]  ksize;
    logic [3:0]  shift_r;
    logic [11:0] width_r, height_r;
    logic [3:0]  pad_t, pad_b, pad_l, pad_r;
    logic [3:0]  stride_x, stride_y;
    logic        sign_in_r, sign_out_r;  // Stage 4: signed intermediates
    logic [1:0]  fuse_act_r;             // Stage 4: 0=none, 1=ReLU, 2=Leaky
    logic [15:0] act_slope_r;            // Stage 4: Leaky slope_fp (Q0.16)
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
            sign_in_r <= 1'b0; sign_out_r <= 1'b0;
            fuse_act_r <= 2'd0; act_slope_r <= 16'd8192;
            for (ci = 0; ci < MAX_COEFF; ci++) coeff[ci] <= '0;
            // default: 3x3 identity (centre 256, shift 8). The 3x3 kernel
            // occupies the top-left 3x3 of the 11x11 coeff grid, so its
            // centre tap is at index 1*11+1 = 12.
            coeff[1*11+1] <= 16'sd256;
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
    //   out_W = (pw - K)/Sx + 1, out_H = (ph - K)/Sy + 1
    // but the DUT never divides: the decimation uses the "next exceeds"
    // comparisons (nxt_sc + Sx > pw - 1, nxt_sr + Sy > ph - 1), so no
    // divider is synthesized. The testbench computes out_W/out_H itself.
    logic [12:0] pw;      // padded width  = W + pad_l + pad_r
    logic [12:0] ph;      // padded height = H + pad_t + pad_b
    assign pw = {1'b0, width_r} + {9'd0, pad_l} + {9'd0, pad_r};
    assign ph = {1'b0, height_r} + {9'd0, pad_t} + {9'd0, pad_b};

    //------------------------- enable shift register ----------------------
    // en_d[i] = stream_en delayed by i cycles. The window loads on en_d[10]
    // (10 = line-buffer + alignment delay); the line buffers use en_d[0..9].
    logic [10:0] en_shreg;
    logic [10:0] en_d;

    //------------------------- stream control ------------------------------
    // The stream walks the PADDED image, (strm_r, strm_c) in raster order.
    // A stream position is "real" when it maps inside the input image;
    // otherwise the DUT feeds a zero (padding). The testbench sends only
    // the raw W*H pixels; in_ready is high only at real positions.
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
    logic win_v, prod_v, sum2_v, sum4_v, sum6_v, acc_v, shift_v;
    assign pipe_empty = !win_v && !prod_v && !sum2_v && !sum4_v &&
                        !sum6_v && !acc_v && !shift_v && (out_valid == '0);
    assign can_start  = !frame_active && pipe_empty;
    assign start_frame = can_start && in_valid && in_sof;
    assign stall = (out_valid != '0) && !out_ready;
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
    // (nxt_sr, nxt_sc) = (nxt_or*Sy + K - 1, nxt_oc*Sx + K - 1).
    // Counters advance only when a window load matches (nxt_sr, nxt_sc);
    // the row wraps when the next column would exceed the padded row
    // (nxt_sc + Sx > pw - 1), which is exactly out_w - 1. No divider.
    logic [12:0] nxt_or, nxt_oc, nxt_sr, nxt_sc;
    logic        match_out;
    logic [12:0] mac_r, mac_c;   // aligned stream position (window load)
    assign match_out = (mac_r == nxt_sr) && (mac_c == nxt_sc);
    logic is_eol_c, is_last_c;
    assign is_eol_c  = match_out && ((nxt_sc + {9'd0, stride_x}) > (pw - 13'd1));
    assign is_last_c = is_eol_c  && ((nxt_sr + {9'd0, stride_y}) > (ph - 13'd1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= 13'd10; nxt_sc <= 13'd10;  // K-1 for default K=3
        end else if (start_frame) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= {9'd0, ksize} - 13'd1;
            nxt_sc <= {9'd0, ksize} - 13'd1;
        end else if (!stall && en_d[10] && match_out) begin
            if ((nxt_sc + {9'd0, stride_x}) <= (pw - 13'd1)) begin
                nxt_oc <= nxt_oc + 13'd1;
                nxt_sc <= nxt_sc + {9'd0, stride_x};
            end else begin
                nxt_oc <= 13'd0;
                nxt_sc <= {9'd0, ksize} - 13'd1;
                nxt_or <= nxt_or + 13'd1;
                nxt_sr <= nxt_sr + {9'd0, stride_y};
            end
        end
    end

    //------------------------- enable / coordinate delay lines -----------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) en_shreg <= 11'd0;
        else if (!stall) en_shreg <= {en_shreg[9:0], stream_en};
    end
    assign en_d = {en_shreg[9:0], stream_en};

    // coordinate delay line, depth 10: coords of the stream position
    // accepted 10 cycles ago (the aligned position for the window load).
    logic [12:0] cr_d [0:9];
    logic [12:0] cc_d [0:9];
    int di;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (di = 0; di < 10; di++) begin
                cr_d[di] <= 13'd0; cc_d[di] <= 13'd0;
            end
        end else if (!stall) begin
            cr_d[0] <= stream_en ? strm_r : 13'd0;
            cc_d[0] <= stream_en ? strm_c : 13'd0;
            for (di = 1; di < 10; di++) begin
                cr_d[di] <= cr_d[di-1];
                cc_d[di] <= cc_d[di-1];
            end
        end
    end
    assign mac_r = cr_d[9];
    assign mac_c = cc_d[9];

    //------------------------- line buffers (BRAM, 8*PX-bit) --------------
    group_t lb_out [0:9];
    group_t lb_in  [0:9];
    assign lb_in[0] = stream_data;
    genvar gi;
    generate
        for (gi = 0; gi < 10; gi++) begin : lb_chain
            if (gi > 0) assign lb_in[gi] = lb_out[gi-1];
            line_buffer_bram_dense #(.WIDTH(8*PX), .DEPTH(LB_DEPTH),
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
    // row k (0..10): k=10 is the current stream row, k=10-i is lb_i.
    // Row k is delayed by k cycles so all 11 taps of a column arrive
    // together (total delay per row = (10-k) [line buffer] + k = 10).
    group_t row_in [0:10];
    assign row_in[10] = stream_data;
    generate
        for (gi = 0; gi < 10; gi++) begin : row_assign
            assign row_in[gi] = lb_out[9-gi];
        end
    endgenerate

    group_t aln [0:10][0:9];
    int ai, aj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ai = 0; ai < 11; ai++)
                for (aj = 0; aj < 10; aj++) aln[ai][aj] <= '0;
        end else if (!stall) begin
            for (ai = 0; ai < 11; ai++) begin
                aln[ai][0] <= row_in[ai];
                for (aj = 1; aj < 10; aj++) aln[ai][aj] <= aln[ai][aj-1];
            end
        end
    end
    group_t acol [0:10];
    generate
        for (gi = 0; gi < 11; gi++) begin : acol_assign
            if (gi == 0) assign acol[gi] = row_in[gi];
            else         assign acol[gi] = aln[gi][gi-1];
        end
    endgenerate

    //------------------------- 11 x (11+PX-1) window ----------------------
    // Window col wc holds stream pixel (s_r, s_c + wc - 10) for the row.
    // The window shifts left by PX per cycle and loads the new group into
    // cols 10..10+PX-1. The bottom-right corner (win[10][10+PX-1]) is the
    // current stream pixel. Lane p uses sub-window cols p..p+10.
    logic [7:0] win [0:10][0:WIN_W-1];
    int wi, wj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wi = 0; wi < 11; wi++)
                for (wj = 0; wj < WIN_W; wj++) win[wi][wj] <= '0;
        end else if (!stall && en_d[10]) begin
            for (wi = 0; wi < 11; wi++) begin
                for (wj = 0; wj < 10; wj++) win[wi][wj] <= win[wi][wj+PX];
                for (wj = 0; wj < PX; wj++)
                    win[wi][10+wj] <= acol[wi][8*wj +: 8];
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
            win_v <= en_d[10];
            win_is_out <= en_d[10] && match_out;
            win_or <= nxt_or;
            win_oc <= nxt_oc;
            win_eol <= en_d[10] && is_eol_c;
            win_last <= en_d[10] && is_last_c;
        end
    end

    //------------------------- shared pipeline control -------------------
    // Valid flags and output coords are SHARED across lanes. They advance
    // in their own always_ff; the lane generate below touches datapath only.
    logic        p_is_out, s2_is_out, s4_is_out, s6_is_out, a_is_out, sh_is_out;
    logic [12:0] p_or, p_oc, s2_or, s2_oc, s4_or, s4_oc,
                 s6_or, s6_oc, a_or, a_oc, sh_or, sh_oc;
    logic        p_eol, p_last, s2_eol, s2_last, s4_eol, s4_last,
                 s6_eol, s6_last, a_eol, a_last, sh_eol, sh_last;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_v <= 1'b0; p_is_out <= 1'b0; p_or <= 13'd0; p_oc <= 13'd0;
            p_eol <= 1'b0; p_last <= 1'b0;
            sum2_v <= 1'b0; s2_is_out <= 1'b0; s2_or <= 13'd0; s2_oc <= 13'd0;
            s2_eol <= 1'b0; s2_last <= 1'b0;
            sum4_v <= 1'b0; s4_is_out <= 1'b0; s4_or <= 13'd0; s4_oc <= 13'd0;
            s4_eol <= 1'b0; s4_last <= 1'b0;
            sum6_v <= 1'b0; s6_is_out <= 1'b0; s6_or <= 13'd0; s6_oc <= 13'd0;
            s6_eol <= 1'b0; s6_last <= 1'b0;
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
            sum6_v <= sum4_v; s6_is_out <= s4_is_out;
            s6_or <= s4_or;   s6_oc <= s4_oc;
            s6_eol <= s4_eol; s6_last <= s4_last;
            acc_v  <= sum6_v; a_is_out <= s6_is_out;
            a_or <= s6_or;    a_oc <= s6_oc;
            a_eol <= s6_eol;  a_last <= s6_last;
            shift_v<= acc_v;  sh_is_out <= a_is_out;
            sh_or <= a_or;    sh_oc <= a_oc;
            sh_eol <= a_eol;  sh_last <= a_last;
        end
    end

    // Per-lane shift outputs (registered), read by the saturate/output stage.
    logic signed [31:0] shifted [0:PX-1];

    //------------------------- MAC: PX parallel lanes --------------------
    // Each lane: 121 multipliers + 4-stage adder tree + split shift stage.
    // Tap (i,j) for lane p: window [11-K+i][11-K+p+j], coeff[i*11+j].
    // Tree (121 products, 25-bit): 121->61->31 (27b) -> 16->8 (29b) ->
    // 4->2 (31b) -> 1 (32b acc). Max |acc| = 121*255*32768 ~ 1.01e9 < 2^31.
    genvar p;
    generate
        for (p = 0; p < PX; p++) begin : lane
            logic signed [24:0] prod_comb [0:120];
            logic signed [24:0] prod [0:120];
            logic signed [26:0] s1_comb [0:60];
            logic signed [26:0] sum2_comb [0:30];
            logic signed [26:0] sum2 [0:30];
            logic signed [28:0] s3_comb [0:15];
            logic signed [28:0] sum4_comb [0:7];
            logic signed [28:0] sum4 [0:7];
            logic signed [30:0] s5_comb [0:3];
            logic signed [30:0] sum6_comb [0:1];
            logic signed [30:0] sum6 [0:1];
            logic signed [31:0] acc;
            logic signed [31:0] rnd;
            int li, lj, lk, wr, wc, qi, si;

            // multiply stage (combinational)
            always @* begin
                for (li = 0; li < 11; li++) begin
                    for (lj = 0; lj < 11; lj++) begin
                        lk = li*11 + lj;
                        wr = 11 - ksize + li;
                        wc = 11 - ksize + p + lj;
                        if ((li < ksize) && (lj < ksize))
                            prod_comb[lk] = (sign_in_r ?
                                $signed(win[wr][wc]) :
                                $signed({1'b0, win[wr][wc]})) * coeff[lk];
                        else
                            prod_comb[lk] = 25'sd0;
                    end
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (qi = 0; qi < 121; qi++) prod[qi] <= 25'sd0;
                end else if (!stall) begin
                    for (qi = 0; qi < 121; qi++) prod[qi] <= prod_comb[qi];
                end
            end

            // adder tree stage A: 121 -> 61 -> 31 (27-bit)
            always @* begin
                for (si = 0; si < 60; si++) begin
                    s1_comb[si] = prod[2*si] + prod[2*si+1];
                end
                s1_comb[60] = prod[120];
                for (si = 0; si < 30; si++) begin
                    sum2_comb[si] = s1_comb[2*si] + s1_comb[2*si+1];
                end
                sum2_comb[30] = s1_comb[60];
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 31; si++) sum2[si] <= 27'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 31; si++) sum2[si] <= sum2_comb[si];
                end
            end

            // adder tree stage B: 31 -> 16 -> 8 (29-bit)
            always @* begin
                for (si = 0; si < 15; si++) begin
                    s3_comb[si] = sum2[2*si] + sum2[2*si+1];
                end
                s3_comb[15] = sum2[30];
                for (si = 0; si < 8; si++) begin
                    sum4_comb[si] = s3_comb[2*si] + s3_comb[2*si+1];
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 8; si++) sum4[si] <= 29'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 8; si++) sum4[si] <= sum4_comb[si];
                end
            end

            // adder tree stage C: 8 -> 4 -> 2 (31-bit)
            always @* begin
                for (si = 0; si < 4; si++) begin
                    s5_comb[si] = sum4[2*si] + sum4[2*si+1];
                end
                for (si = 0; si < 2; si++) begin
                    sum6_comb[si] = s5_comb[2*si] + s5_comb[2*si+1];
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 2; si++) sum6[si] <= 31'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 2; si++) sum6[si] <= sum6_comb[si];
                end
            end

            // accumulator: 2 -> 1 (32-bit)
            logic signed [31:0] acc_comb;
            assign acc_comb = sum6[0] + sum6[1];
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
                    shifted[p] <= 32'sd0;
                end else if (!stall) begin
                    shifted[p] <= sh_comb;
                end
            end
        end
    endgenerate

    //------------------------- output stage 2: fused act + saturate -----
    // Fused ReLU/Leaky (Stage 4): applied to the shifted accumulator
    // value, BEFORE saturation (mirrors conv2d_mc). ReLU is a comparator
    // + mux; Leaky needs one 32x16 multiply per lane (48-bit product,
    // take bits [47:16] = (shifted * slope_fp) >> 16). The separate act
    // module (bypass mode) is reserved for complex activations.
    logic signed [31:0] act_val [0:PX-1];
    logic signed [47:0] leaky_prod [0:PX-1];
    int fi;
    always @* begin
        for (fi = 0; fi < PX; fi++) begin
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

    logic [7:0] px_sat [0:PX-1];
    int oi;
    always @* begin
        for (oi = 0; oi < PX; oi++) begin
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
            out_valid <= '0; out_sof <= 1'b0;
            out_eol <= 1'b0;    out_last <= 1'b0;
        end else if (!stall) begin
            for (oi = 0; oi < PX; oi++) begin
                out_valid[oi] <= shift_v && sh_is_out;
                pixel_out[8*oi +: 8] <= px_sat[oi];
            end
            out_sof  <= shift_v && sh_is_out && (sh_or == 13'd0) && (sh_oc == 13'd0);
            out_eol  <= shift_v && sh_is_out && sh_eol;
            out_last <= shift_v && sh_is_out && sh_last;
        end
    end
endmodule : conv2d_dense
