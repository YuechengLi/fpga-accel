//=============================================================================
// conv2d_dw.sv -- Stage 2: depthwise (DW) convolution, SystemVerilog.
// Modes (config register 0x0A): DW-SQUARE / DW-STRIP-H / DW-STRIP-V /
// DW-DILATED. Single channel, PX=1. DENSE is NOT re-implemented here;
// Stage 1's conv2d_dense.sv provides it, bit-identical and untouched.
//
// Streaming, no frame buffer. One unified datapath, fully parallel:
//   * Input  : 8-bit pixels, raster order, one pixel/cycle when in_valid.
//              in_sof marks the first pixel of a frame. The DUT expands
//              the input to the PADDED image on the fly (zero fill); the
//              testbench sends only the raw W*H image.
//   * Output : 8-bit pixels, raster order, out_valid is a single bit;
//              out_sof / out_eol / out_last.
//              Output size out_W = (pw - span_x)/Sx + 1,
//              out_H = (ph - span_y)/Sy + 1, where (span_x, span_y) is the
//              effective kernel footprint of the selected mode:
//                DW-SQUARE  : span = K            (both axes)
//                DW-STRIP-H : span_x = L, span_y = 1
//                DW-STRIP-V : span_x = 1, span_y = L
//                DW-DILATED : span = (K-1)*d + 1  (both axes, d = dilation)
//              Stride is DECIMATE: every position is computed, out_valid
//              is asserted only at stride positions.
//   * Kernel : stored kernel K (DW-SQUARE 1/3/5/7/9/11/13/15/17;
//              DW-STRIP 1/3/5/7/9/11/13/17/21/31; DW-DILATED stored
//              3/5/7/9 with dilation d = 1..4, effective span <= 33).
//              16-bit signed coefficients, loaded at run time through the
//              config bus. out = clamp((acc + round) >> shift).
//              289 multipliers when fully parallel (17x17 square); the
//              unified MAC instantiates all 289 and unused slots compute 0,
//              so DSP = 289 for every mode (see README).
//   * 32 line buffers (BRAM) + a 33 x 33 register window. The window's
//     bottom-right corner is the current stream pixel, so each mode's
//     kernel footprint is anchored bottom-right: tap (i,j) reads
//     win[33-span_y+i*dy][33-span_x+j*dx] with (dy,dx) = (1,1) square,
//     (0,1) strip-H, (1,0) strip-V, (d,d) dilated. No row/col-valid
//     masks are needed: padding is physically in the stream (zeros), and
//     the window resets to 0, so taps outside the padded image read 0.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//   * Pipeline: window -> prod -> sA -> sB -> sC -> sD -> acc -> shift ->
//     saturate/output (9 register stages from window load; one more than
//     Stage 1, because the 289-product adder tree needs 5 register
//     stages instead of 4).
//   * Accumulator is 32-bit (same as Stage 1). Worst case |acc| for
//     17x17 DW is 289*255*32768 ~ 2.41e9 > 2^31, so a fully-saturated
//     17x17 kernel at full-scale input can wrap the accumulator; the
//     golden model wraps identically (2's complement), so golden and RTL
//     still agree. Real kernels are normalised and do not hit this.
//
// Config bus (write between frames; single-cycle writes; 9-bit address
// because 289 coefficients do not fit in an 8-bit address space):
//   addr 0x00 : kernel size K / strip length L (1..31)
//   addr 0x01 : output shift (0..15)
//   addr 0x02 : image width  (1..4095)   addr 0x03 : image height (1..4095)
//   addr 0x04 : pad_top    (0..33)       addr 0x05 : pad_bottom (0..33)
//   addr 0x06 : pad_left   (0..33)       addr 0x07 : pad_right  (0..33)
//   addr 0x08 : stride_x   (1..33)       addr 0x09 : stride_y   (1..33)
//   addr 0x0A : mode (0=DW-SQUARE, 1=DW-STRIP-H, 2=DW-STRIP-V,
//                      3=DW-DILATED)
//   addr 0x0B : dilation d (1..4, DW-DILATED only)
//   addr 0x10+i : coeff[i], i = 0..288. Layout depends on the mode:
//     DW-SQUARE : 17x17 grid, coeff[i*17+j] (top-left KxK is the kernel)
//     DW-STRIP-H/V : coeff[j], j = 0..L-1 (1-D, left-to-right / top-to-bottom)
//     DW-DILATED : 9x9 grid, coeff[i*9+j] (top-left KxK is the kernel)
// Reset defaults: mode=0 (DW-SQUARE), ksize=3, dil=1, shift=8, W=640,
//   H=480, all pads 0, stride_x=stride_y=1, 3x3 identity (centre tap 256
//   at index 1*17+1 = 18).
//=============================================================================
`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// One row delay in BRAM, 8-bit wide. Read-first: returns the group written
// one full row ago. Pointer/reset in one block, memory access in its own
// reset-free block, or Yosys will not infer a block RAM.
// (Same template as conv2d_dense's line_buffer_bram_dense.)
//-----------------------------------------------------------------------------
module line_buffer_bram_dw #(
    parameter int WIDTH  = 8,
    parameter int DEPTH  = 3968,
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
    // 2^ADDR_W. Identical to == in normal operation.
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
endmodule : line_buffer_bram_dw

//-----------------------------------------------------------------------------
// Main module (PX=1, single channel)
//-----------------------------------------------------------------------------
module conv2d_dw #(
    parameter int MAX_W = 3840
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus (9-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [8:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (single pixels)
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (single pixels)
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    // Config-bus addresses.
    localparam logic [8:0] CFG_KSIZE  = 9'h000;
    localparam logic [8:0] CFG_SHIFT  = 9'h001;
    localparam logic [8:0] CFG_WIDTH  = 9'h002;
    localparam logic [8:0] CFG_HEIGHT = 9'h003;
    localparam logic [8:0] CFG_PAD_T  = 9'h004;
    localparam logic [8:0] CFG_PAD_B  = 9'h005;
    localparam logic [8:0] CFG_PAD_L  = 9'h006;
    localparam logic [8:0] CFG_PAD_R  = 9'h007;
    localparam logic [8:0] CFG_STRIDE_X = 9'h008;
    localparam logic [8:0] CFG_STRIDE_Y = 9'h009;
    localparam logic [8:0] CFG_MODE   = 9'h00A;
    localparam logic [8:0] CFG_DIL    = 9'h00B;
    localparam logic [8:0] CFG_COEFF_BASE = 9'h010;
    localparam logic [8:0] CFG_COEFF_LAST = 9'h130;  // 0x10 + 288

    // Modes.
    localparam logic [1:0] MODE_SQUARE = 2'd0;
    localparam logic [1:0] MODE_STRIP_H = 2'd1;
    localparam logic [1:0] MODE_STRIP_V = 2'd2;
    localparam logic [1:0] MODE_DILATED = 2'd3;

    localparam int MAX_COEFF = 289;   // 17x17
    localparam int WIN_S     = 33;    // window rows/cols (max effective span)
    localparam int N_LB      = 32;    // line buffers
    // Padded row can exceed MAX_W by up to 2*33 (pad_l + pad_r).
    localparam int LB_DEPTH = MAX_W + 128;
    localparam int ADDR_W   = (LB_DEPTH <= 2048) ? 11 :
                              (LB_DEPTH <= 4096) ? 12 : 13;

    //------------------------- config registers ----------------------------
    logic [1:0]  mode;
    logic [4:0]  ksize;
    logic [2:0]  dil;
    logic [3:0]  shift_r;
    logic [11:0] width_r, height_r;
    logic [5:0]  pad_t, pad_b, pad_l, pad_r;
    logic [5:0]  stride_x, stride_y;
    logic signed [15:0] coeff [0:MAX_COEFF-1];
    int ci;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode     <= MODE_SQUARE;
            ksize    <= 5'd3;
            dil      <= 3'd1;
            shift_r  <= 4'd8;
            width_r  <= 12'd640;
            height_r <= 12'd480;
            pad_t <= 6'd0; pad_b <= 6'd0; pad_l <= 6'd0; pad_r <= 6'd0;
            stride_x <= 6'd1; stride_y <= 6'd1;
            for (ci = 0; ci < MAX_COEFF; ci++) coeff[ci] <= '0;
            // default: 3x3 identity (centre 256, shift 8). The 3x3 kernel
            // occupies the top-left 3x3 of the 17x17 coeff grid, so its
            // centre tap is at index 1*17+1 = 18.
            coeff[1*17+1] <= 16'sd256;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_KSIZE : ksize    <= cfg_data[4:0];
                CFG_SHIFT : shift_r  <= cfg_data[3:0];
                CFG_WIDTH : width_r  <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                CFG_PAD_T : pad_t    <= cfg_data[5:0];
                CFG_PAD_B : pad_b    <= cfg_data[5:0];
                CFG_PAD_L : pad_l    <= cfg_data[5:0];
                CFG_PAD_R : pad_r    <= cfg_data[5:0];
                CFG_STRIDE_X: stride_x <= cfg_data[5:0];
                CFG_STRIDE_Y: stride_y <= cfg_data[5:0];
                CFG_MODE  : mode     <= cfg_data[1:0];
                CFG_DIL   : dil      <= cfg_data[2:0];
                default: begin
                    if (cfg_addr >= CFG_COEFF_BASE && cfg_addr <= CFG_COEFF_LAST)
                        coeff[cfg_addr - CFG_COEFF_BASE] <= cfg_data;
                end
            endcase
        end
    end

    //------------------------- derived geometry ---------------------------
    // Padded image size (pixels). The output size is
    //   out_W = (pw - span_x)/Sx + 1, out_H = (ph - span_y)/Sy + 1
    // but the DUT never divides: the decimation uses the "next exceeds"
    // comparisons, so no divider is synthesized.
    logic [12:0] pw;      // padded width  = W + pad_l + pad_r
    logic [12:0] ph;      // padded height = H + pad_t + pad_b
    assign pw = {1'b0, width_r} + {7'd0, pad_l} + {7'd0, pad_r};
    assign ph = {1'b0, height_r} + {7'd0, pad_t} + {7'd0, pad_b};

    // Effective kernel footprint (span) per mode.
    logic [5:0] span_x, span_y;
    logic [5:0] dil_span;   // (K-1)*d + 1 for DW-DILATED
    assign dil_span = ({3'd0, ksize} - 6'd1) * {3'd0, dil} + 6'd1;
    always @* begin
        case (mode)
            MODE_SQUARE : begin span_x = {1'b0, ksize}; span_y = {1'b0, ksize}; end
            MODE_STRIP_H: begin span_x = {1'b0, ksize}; span_y = 6'd1; end
            MODE_STRIP_V: begin span_x = 6'd1; span_y = {1'b0, ksize}; end
            MODE_DILATED: begin span_x = dil_span; span_y = dil_span; end
            default     : begin span_x = {1'b0, ksize}; span_y = {1'b0, ksize}; end
        endcase
    end

    //------------------------- enable shift register ----------------------
    // en_d[i] = stream_en delayed by i cycles. The window loads on en_d[32]
    // (32 = line-buffer + alignment delay); the line buffers use en_d[0..31].
    logic [32:0] en_shreg;
    logic [32:0] en_d;

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
    logic [7:0] stream_data;

    assign next_c = (strm_c == pw - 13'd1) ? 13'd0 : strm_c + 13'd1;
    assign next_r = (strm_c == pw - 13'd1) ? strm_r + 13'd1 : strm_r;
    assign is_last_stream = (strm_r == ph - 13'd1) && (strm_c == pw - 13'd1);
    assign pos_is_real = (strm_r >= {7'd0, pad_t}) &&
                         (strm_r <  {7'd0, pad_t} + {1'b0, height_r}) &&
                         (strm_c >= {7'd0, pad_l}) &&
                         (strm_c <  {7'd0, pad_l} + {1'b0, width_r});

    // pipeline valid flags; empty when all are 0
    logic win_v, prod_v, sA_v, sB_v, sC_v, sD_v, acc_v, shift_v;
    assign pipe_empty = !win_v && !prod_v && !sA_v && !sB_v && !sC_v &&
                        !sD_v && !acc_v && !shift_v && !out_valid;
    assign can_start  = !frame_active && pipe_empty;
    assign start_frame = can_start && in_valid && in_sof;
    assign stall = out_valid && !out_ready;
    assign stream_en = !stall && (start_frame ||
                         (frame_active && (!pos_is_real || in_valid)));
    assign stream_data = pos_is_real ? pixel_in : 8'd0;
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
    // (nxt_sr, nxt_sc) = (nxt_or*Sy + span_y - 1, nxt_oc*Sx + span_x - 1).
    // Counters advance only when a window load matches (nxt_sr, nxt_sc);
    // the row wraps when the next column would exceed the padded row
    // (nxt_sc + Sx > pw - 1), which is exactly out_w - 1. No divider.
    logic [12:0] nxt_or, nxt_oc, nxt_sr, nxt_sc;
    logic        match_out;
    logic [12:0] mac_r, mac_c;   // aligned stream position (window load)
    assign match_out = (mac_r == nxt_sr) && (mac_c == nxt_sc);
    logic is_eol_c, is_last_c;
    assign is_eol_c  = match_out && ((nxt_sc + {7'd0, stride_x}) > (pw - 13'd1));
    assign is_last_c = is_eol_c  && ((nxt_sr + {7'd0, stride_y}) > (ph - 13'd1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= 13'd2; nxt_sc <= 13'd2;  // span-1 for default K=3
        end else if (start_frame) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= {7'd0, span_y} - 13'd1;
            nxt_sc <= {7'd0, span_x} - 13'd1;
        end else if (!stall && en_d[32] && match_out) begin
            if ((nxt_sc + {7'd0, stride_x}) <= (pw - 13'd1)) begin
                nxt_oc <= nxt_oc + 13'd1;
                nxt_sc <= nxt_sc + {7'd0, stride_x};
            end else begin
                nxt_oc <= 13'd0;
                nxt_sc <= {7'd0, span_x} - 13'd1;
                nxt_or <= nxt_or + 13'd1;
                nxt_sr <= nxt_sr + {7'd0, stride_y};
            end
        end
    end

    //------------------------- enable / coordinate delay lines -----------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) en_shreg <= 33'd0;
        else if (!stall) en_shreg <= {en_shreg[31:0], stream_en};
    end
    assign en_d = {en_shreg[31:0], stream_en};

    // coordinate delay line, depth 32: coords of the stream position
    // accepted 32 cycles ago (the aligned position for the window load).
    logic [12:0] cr_d [0:31];
    logic [12:0] cc_d [0:31];
    int di;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (di = 0; di < 32; di++) begin
                cr_d[di] <= 13'd0; cc_d[di] <= 13'd0;
            end
        end else if (!stall) begin
            cr_d[0] <= stream_en ? strm_r : 13'd0;
            cc_d[0] <= stream_en ? strm_c : 13'd0;
            for (di = 1; di < 32; di++) begin
                cr_d[di] <= cr_d[di-1];
                cc_d[di] <= cc_d[di-1];
            end
        end
    end
    assign mac_r = cr_d[31];
    assign mac_c = cc_d[31];

    //------------------------- line buffers (BRAM, 8-bit) -----------------
    logic [7:0] lb_out [0:31];
    logic [7:0] lb_in  [0:31];
    assign lb_in[0] = stream_data;
    genvar gi;
    generate
        for (gi = 0; gi < 32; gi++) begin : lb_chain
            if (gi > 0) assign lb_in[gi] = lb_out[gi-1];
            line_buffer_bram_dw #(.WIDTH(8), .DEPTH(LB_DEPTH),
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
    // row k (0..32): k=32 is the current stream row, k=32-i is lb_i.
    // Row k is delayed by k cycles so all 33 taps of a column arrive
    // together (total delay per row = (32-k) [line buffer] + k = 32).
    logic [7:0] row_in [0:32];
    assign row_in[32] = stream_data;
    generate
        for (gi = 0; gi < 32; gi++) begin : row_assign
            assign row_in[gi] = lb_out[31-gi];
        end
    endgenerate

    logic [7:0] aln [0:32][0:31];
    int ai, aj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ai = 0; ai < 33; ai++)
                for (aj = 0; aj < 32; aj++) aln[ai][aj] <= 8'd0;
        end else if (!stall) begin
            for (ai = 0; ai < 33; ai++) begin
                aln[ai][0] <= row_in[ai];
                for (aj = 1; aj < 32; aj++) aln[ai][aj] <= aln[ai][aj-1];
            end
        end
    end
    logic [7:0] acol [0:32];
    generate
        for (gi = 0; gi < 33; gi++) begin : acol_assign
            if (gi == 0) assign acol[gi] = row_in[gi];
            else         assign acol[gi] = aln[gi][gi-1];
        end
    endgenerate

    //------------------------- 33 x 33 window -----------------------------
    // Window col wc holds stream pixel (s_r, s_c + wc - 32) for the row.
    // The window shifts left by 1 per cycle and loads the new pixel into
    // col 32. The bottom-right corner (win[32][32]) is the current stream
    // pixel; each mode's kernel footprint is its bottom-right sub-window.
    logic [7:0] win [0:32][0:32];
    int wi, wj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wi = 0; wi < 33; wi++)
                for (wj = 0; wj < 33; wj++) win[wi][wj] <= 8'd0;
        end else if (!stall && en_d[32]) begin
            for (wi = 0; wi < 33; wi++) begin
                for (wj = 0; wj < 32; wj++) win[wi][wj] <= win[wi][wj+1];
                win[wi][32] <= acol[wi];
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
            win_v <= en_d[32];
            win_is_out <= en_d[32] && match_out;
            win_or <= nxt_or;
            win_oc <= nxt_oc;
            win_eol <= en_d[32] && is_eol_c;
            win_last <= en_d[32] && is_last_c;
        end
    end

    //------------------------- shared pipeline control -------------------
    // Valid flags and output coords are SHARED. They advance in their own
    // always_ff; the datapath below touches datapath only.
    logic        p_is_out, sA_is_out, sB_is_out, sC_is_out, sD_is_out,
                 a_is_out, sh_is_out;
    logic [12:0] p_or, p_oc, sA_or, sA_oc, sB_or, sB_oc, sC_or, sC_oc,
                 sD_or, sD_oc, a_or, a_oc, sh_or, sh_oc;
    logic        p_eol, p_last, sA_eol, sA_last, sB_eol, sB_last,
                 sC_eol, sC_last, sD_eol, sD_last, a_eol, a_last,
                 sh_eol, sh_last;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_v <= 1'b0; p_is_out <= 1'b0; p_or <= 13'd0; p_oc <= 13'd0;
            p_eol <= 1'b0; p_last <= 1'b0;
            sA_v <= 1'b0; sA_is_out <= 1'b0; sA_or <= 13'd0; sA_oc <= 13'd0;
            sA_eol <= 1'b0; sA_last <= 1'b0;
            sB_v <= 1'b0; sB_is_out <= 1'b0; sB_or <= 13'd0; sB_oc <= 13'd0;
            sB_eol <= 1'b0; sB_last <= 1'b0;
            sC_v <= 1'b0; sC_is_out <= 1'b0; sC_or <= 13'd0; sC_oc <= 13'd0;
            sC_eol <= 1'b0; sC_last <= 1'b0;
            sD_v <= 1'b0; sD_is_out <= 1'b0; sD_or <= 13'd0; sD_oc <= 13'd0;
            sD_eol <= 1'b0; sD_last <= 1'b0;
            acc_v  <= 1'b0; a_is_out <= 1'b0; a_or <= 13'd0; a_oc <= 13'd0;
            a_eol <= 1'b0; a_last <= 1'b0;
            shift_v<= 1'b0; sh_is_out <= 1'b0; sh_or <= 13'd0; sh_oc <= 13'd0;
            sh_eol <= 1'b0; sh_last <= 1'b0;
        end else if (!stall) begin
            prod_v <= win_v;  p_is_out <= win_is_out;
            p_or <= win_or;   p_oc <= win_oc;
            p_eol <= win_eol; p_last <= win_last;
            sA_v <= prod_v; sA_is_out <= p_is_out;
            sA_or <= p_or;  sA_oc <= p_oc;
            sA_eol <= p_eol; sA_last <= p_last;
            sB_v <= sA_v; sB_is_out <= sA_is_out;
            sB_or <= sA_or; sB_oc <= sA_oc;
            sB_eol <= sA_eol; sB_last <= sA_last;
            sC_v <= sB_v; sC_is_out <= sB_is_out;
            sC_or <= sB_or; sC_oc <= sB_oc;
            sC_eol <= sB_eol; sC_last <= sB_last;
            sD_v <= sC_v; sD_is_out <= sC_is_out;
            sD_or <= sC_or; sD_oc <= sC_oc;
            sD_eol <= sC_eol; sD_last <= sC_last;
            acc_v  <= sD_v; a_is_out <= sD_is_out;
            a_or <= sD_or;  a_oc <= sD_oc;
            a_eol <= sD_eol; a_last <= sD_last;
            shift_v<= acc_v; sh_is_out <= a_is_out;
            sh_or <= a_or;  sh_oc <= a_oc;
            sh_eol <= a_eol; sh_last <= a_last;
        end
    end

    //------------------------- MAC: 289 parallel multipliers -------------
    // Slot m (0..288) of the 17x17 grid, m = i*17+j. Per mode:
    //   DW-SQUARE : active if i<K, j<K; tap win[33-K+i][33-K+j], coeff m
    //   DW-STRIP-H: active if m<L; tap win[32][33-L+m], coeff m
    //   DW-STRIP-V: active if m<L; tap win[33-L+m][32], coeff m
    //   DW-DILATED: active if i<K, j<K; tap win[33-S+i*d][33-S+j*d],
    //               coeff i*9+j (S = (K-1)*d+1)
    // Strips use the linear slot index m (not a single grid row/column),
    // so a length-31 strip gets all 31 taps. Inactive slots compute 0.
    // The adder tree (289 products, 25-bit):
    //   A: 289->145->73 (27b)  B: 73->37->19 (29b)  C: 19->10->5 (31b)
    //   D: 5->3->2 (33b)       E: 2->1 (33b, truncated to the 32-bit acc)
    // Max |acc| = 289*255*32768 ~ 2.41e9; the 32-bit acc wraps (2's
    // complement) at full scale, matching the golden model.
    logic signed [24:0] prod_comb [0:288];
    logic signed [24:0] prod [0:288];
    int m, mi, mj;
    int tr, tc, cidx;
    logic act;
    always @* begin
        for (m = 0; m < 289; m++) begin
            mi = m / 17;
            mj = m % 17;
            act = 1'b0;
            tr = 32; tc = 32; cidx = m;
            case (mode)
                MODE_SQUARE: begin
                    if ((mi < ksize) && (mj < ksize)) begin
                        act = 1'b1;
                        tr = 33 - ksize + mi;
                        tc = 33 - ksize + mj;
                        cidx = m;
                    end
                end
                MODE_STRIP_H: begin
                    if (m < ksize) begin
                        act = 1'b1;
                        tr = 32;
                        tc = 33 - ksize + m;
                        cidx = m;
                    end
                end
                MODE_STRIP_V: begin
                    if (m < ksize) begin
                        act = 1'b1;
                        tr = 33 - ksize + m;
                        tc = 32;
                        cidx = m;
                    end
                end
                MODE_DILATED: begin
                    if ((mi < ksize) && (mj < ksize)) begin
                        act = 1'b1;
                        tr = 33 - span_y + mi*dil;
                        tc = 33 - span_x + mj*dil;
                        cidx = mi*9 + mj;
                    end
                end
                default: begin end
            endcase
            if (act)
                prod_comb[m] = $signed({1'b0, win[tr][tc]}) * coeff[cidx];
            else
                prod_comb[m] = 25'sd0;
        end
    end
    int qi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (qi = 0; qi < 289; qi++) prod[qi] <= 25'sd0;
        end else if (!stall) begin
            for (qi = 0; qi < 289; qi++) prod[qi] <= prod_comb[qi];
        end
    end

    // adder tree stage A: 289 -> 145 -> 73 (27-bit)
    logic signed [26:0] s1_comb [0:144];
    logic signed [26:0] sA_comb [0:72];
    logic signed [26:0] sumA [0:72];
    int si;
    always @* begin
        for (si = 0; si < 144; si++) s1_comb[si] = prod[2*si] + prod[2*si+1];
        s1_comb[144] = prod[288];
        for (si = 0; si < 72; si++) sA_comb[si] = s1_comb[2*si] + s1_comb[2*si+1];
        sA_comb[72] = s1_comb[144];
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (si = 0; si < 73; si++) sumA[si] <= 27'sd0;
        end else if (!stall) begin
            for (si = 0; si < 73; si++) sumA[si] <= sA_comb[si];
        end
    end

    // adder tree stage B: 73 -> 37 -> 19 (29-bit)
    logic signed [28:0] s3_comb [0:36];
    logic signed [28:0] sB_comb [0:18];
    logic signed [28:0] sumB [0:18];
    always @* begin
        for (si = 0; si < 36; si++) s3_comb[si] = sumA[2*si] + sumA[2*si+1];
        s3_comb[36] = sumA[72];
        for (si = 0; si < 18; si++) sB_comb[si] = s3_comb[2*si] + s3_comb[2*si+1];
        sB_comb[18] = s3_comb[36];
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (si = 0; si < 19; si++) sumB[si] <= 29'sd0;
        end else if (!stall) begin
            for (si = 0; si < 19; si++) sumB[si] <= sB_comb[si];
        end
    end

    // adder tree stage C: 19 -> 10 -> 5 (31-bit)
    logic signed [30:0] s5_comb [0:9];
    logic signed [30:0] sC_comb [0:4];
    logic signed [30:0] sumC [0:4];
    always @* begin
        for (si = 0; si < 9; si++) s5_comb[si] = sumB[2*si] + sumB[2*si+1];
        s5_comb[9] = sumB[18];
        for (si = 0; si < 5; si++) sC_comb[si] = s5_comb[2*si] + s5_comb[2*si+1];
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (si = 0; si < 5; si++) sumC[si] <= 31'sd0;
        end else if (!stall) begin
            for (si = 0; si < 5; si++) sumC[si] <= sC_comb[si];
        end
    end

    // adder tree stage D: 5 -> 3 -> 2 (33-bit)
    logic signed [32:0] s7_comb [0:2];
    logic signed [32:0] sD_comb [0:1];
    logic signed [32:0] sumD [0:1];
    always @* begin
        s7_comb[0] = sumC[0] + sumC[1];
        s7_comb[1] = sumC[2] + sumC[3];
        s7_comb[2] = sumC[4];
        sD_comb[0] = s7_comb[0] + s7_comb[1];
        sD_comb[1] = s7_comb[2];
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (si = 0; si < 2; si++) sumD[si] <= 33'sd0;
        end else if (!stall) begin
            for (si = 0; si < 2; si++) sumD[si] <= sD_comb[si];
        end
    end

    // accumulator: 2 -> 1 (33-bit sum, truncated to the 32-bit acc)
    logic signed [32:0] acc33_comb;
    logic signed [31:0] acc;
    assign acc33_comb = sumD[0] + sumD[1];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 32'sd0;
        end else if (!stall) begin
            acc <= acc33_comb[31:0];
        end
    end

    // output stage 1: shift (barrel shifter, split from saturate)
    logic signed [31:0] rnd;
    logic signed [31:0] sh_comb;
    logic signed [31:0] shifted;
    always @* begin
        rnd = acc + ((shift_r == 0) ? 32'sd0 : (32'sd1 <<< (shift_r - 1'b1)));
    end
    assign sh_comb = rnd >>> shift_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shifted <= 32'sd0;
        end else if (!stall) begin
            shifted <= sh_comb;
        end
    end

    //------------------------- output stage 2: saturate + framing -------
    logic [7:0] px_sat;
    always @* begin
        if (shifted > 32'sd255)      px_sat = 8'd255;
        else if (shifted < 32'sd0)   px_sat = 8'd0;
        else                          px_sat = shifted[7:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= 8'd0;
            out_valid <= 1'b0; out_sof <= 1'b0;
            out_eol <= 1'b0;    out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= shift_v && sh_is_out;
            pixel_out <= px_sat;
            out_sof  <= shift_v && sh_is_out && (sh_or == 13'd0) && (sh_oc == 13'd0);
            out_eol  <= shift_v && sh_is_out && sh_eol;
            out_last <= shift_v && sh_is_out && sh_last;
        end
    end
endmodule : conv2d_dw
