//=============================================================================
// conv2d_tappar.sv -- Stage 7b: tap-parallel dense convolution engine.
// (SRCNN L3: 5x5, 32 -> 1.)
//
// The K^2-parallel shape of conv2d_mc is wrong for L3 (a 5x5 kernel with
// 32 input channels is 800 mults per output position; a single K^2 lane
// would need 800 cycles/position). This engine parallelises across
// (tap-row x input channel): one tap-row (5 taps x 32 channels = 160
// mults) per cycle, accumulating over the 5 tap-rows, so one output
// position costs 5 cycles. 160 DSPs.
//
// The input is a stream of 32-byte position vectors (the L2 gather
// output), one vector per 5 cycles, in raster order. The frame streams
// through (K-1) = 4 line buffers + a 5x5 window of 256-bit cells; the
// window advances once per input vector. The compute rhythm is 5 cycles
// per position (tap = cycle mod 5); the adder tree (N=160, 4-cycle
// latency) is fed one tap-row per cycle, and the row-sums are accumulated
// as they arrive. A 4-deep flag pipeline tracks which tap-row each
// tree output belongs to; when the tap-row-4 row-sum arrives the
// accumulator is complete and the position is emitted (shift, fused
// ReLU/Leaky, saturate, 1 byte).
//
// Phasing (same as conv2d_outpar): the window advances on the LAST tap
// cycle (tap==4), when the input vector is captured, so the new window
// is in place for the next group's tap==0 compute. The tap-row-4
// compute on the advance cycle reads the OLD window (pre-edge), which
// is the window for the position being finished. in_ready pulses once
// per 5-cycle group (at tap==4).
//
// Config bus (same conventions as conv2d_mc / conv2d_outpar):
//   0x00 ksize (1/3/5/7/9)   0x01 shift (0..15)
//   0x02 width  (input W)    0x03 height (input H)
//   0x04 pad_top  0x05 pad_bottom  0x06 pad_left  0x07 pad_right
//   0x08 stride_x  0x09 stride_y   0x0A C_in (32)  0x0B C_out (1)
//   0x0D out_w  0x0E out_h  0x0F sign (bit0 in, bit1 out)
//   0x10 fused act (0=none, 1=ReLU, 2=Leaky)  0x11 Leaky slope (Q0.16)
//   0x0C GO (write 1 = start the frame)
//   0x100 + tap*32 + c : coeff for tap (t*K+j), channel c
//     (tap 0..K*K-1, c 0..C_in-1), 16-bit signed.
//   0x420 + o : bias low 16 bits for output channel o (o=0 only)
//   0x440 + o : bias high 16 bits for output channel o
//   The bias bases sit AFTER the coeff range (0x100..0x41F, 800
//   entries); putting them inside it would route the bias writes
//   into the coeff array.
//
// Pipeline latency (first tap-row fed -> emit): ~10 cycles. Throughput
// 1 output position per 5 cycles.
//
// Resource (inferred): 160 DSPs (5x32 mults). The 160-input adder tree
// is the timing-critical path; see the STA logs.
//=============================================================================
`timescale 1ns/1ps

module conv2d_tappar #(
    parameter int P_IN   = 32,       // input channels (vector width)
    parameter int MAX_K  = 5,        // max kernel size
    parameter int C_OUT  = 1,        // output channels (always 1 here)
    parameter int LB_DEPTH = 2048,   // line-buffer depth (max input W)
    parameter int LB_ADDR_W = 11,
    // Position lag: the window's bottom-right lags the advance counter
    // by this many advances. The line buffers + BRAM read + column
    // shift give MAX_K-2 for this datapath (measured: the window holds
    // the columns MAX_K-2 ahead of a MAX_K lag, i.e. the effective
    // latency is MAX_K-2 advances, not MAX_K).
    parameter int POS_LAG = MAX_K - 2
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (P_IN-channel position vector, raster order)
    input  logic [P_IN*8-1:0] in_vec,
    input  logic        in_valid,
    input  logic        in_sof,
    input  logic        in_eol,
    input  logic        in_last,
    output logic        in_ready,
    // stream out (1 byte per position)
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [15:0] CFG_KSIZE   = 16'h0000;
    localparam logic [15:0] CFG_SHIFT   = 16'h0001;
    localparam logic [15:0] CFG_W       = 16'h0002;
    localparam logic [15:0] CFG_H       = 16'h0003;
    localparam logic [15:0] CFG_PAD_T   = 16'h0004;
    localparam logic [15:0] CFG_PAD_B   = 16'h0005;
    localparam logic [15:0] CFG_PAD_L   = 16'h0006;
    localparam logic [15:0] CFG_PAD_R   = 16'h0007;
    localparam logic [15:0] CFG_STR_X   = 16'h0008;
    localparam logic [15:0] CFG_STR_Y   = 16'h0009;
    localparam logic [15:0] CFG_CIN     = 16'h000A;
    localparam logic [15:0] CFG_COUT    = 16'h000B;
    localparam logic [15:0] CFG_GO      = 16'h000C;
    localparam logic [15:0] CFG_OUT_W   = 16'h000D;
    localparam logic [15:0] CFG_OUT_H   = 16'h000E;
    localparam logic [15:0] CFG_SIGN    = 16'h000F;
    localparam logic [15:0] CFG_ACT     = 16'h0010;
    localparam logic [15:0] CFG_ACT_SLOPE = 16'h0011;
    localparam logic [15:0] CFG_COEFF_BASE = 16'h0100;
    // Bias bases sit AFTER the coeff range (0x100..0x41F, 800 entries).
    localparam logic [15:0] CFG_BIAS_LO_BASE = 16'h0420;
    localparam logic [15:0] CFG_BIAS_HI_BASE = 16'h0440;

    // ------------------------- config registers -------------------------
    logic [3:0]  ksize_r;
    logic [3:0]  shift_r;
    logic [15:0] width_r, height_r;
    logic [3:0]  pad_t_r, pad_b_r, pad_l_r, pad_r_r;
    logic [3:0]  stride_x_r, stride_y_r;
    logic [7:0]  cin_r, cout_r;
    logic [15:0] out_w_r, out_h_r;
    logic        sign_in_r, sign_out_r;
    logic [1:0]  fuse_act_r;
    logic signed [15:0] act_slope_r;
    // GO is a one-cycle pulse on the config write (not a level): a
    // level would restart the frame as soon as the state returns to
    // IDLE after the flush.
    logic        go_pulse;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) go_pulse <= 1'b0;
        else go_pulse <= cfg_valid && cfg_we && (cfg_addr == CFG_GO) &&
                         cfg_data[0];
    end
    // coeff[tap][c]: tap = t*K+j (0..24), c = input channel (0..31).
    logic signed [15:0] coeff [0:MAX_K*MAX_K-1][0:P_IN-1];
    logic [15:0] bias_lo [0:C_OUT-1];
    logic [15:0] bias_hi [0:C_OUT-1];

    integer ci, cj;
    integer oi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ksize_r <= 4'd5; shift_r <= 4'd8;
            width_r <= 16'd16; height_r <= 16'd16;
            pad_t_r <= 4'd0; pad_b_r <= 4'd0;
            pad_l_r <= 4'd0; pad_r_r <= 4'd0;
            stride_x_r <= 4'd1; stride_y_r <= 4'd1;
            cin_r <= 8'd32; cout_r <= 8'd1;
            out_w_r <= 16'd12; out_h_r <= 16'd12;
            sign_in_r <= 1'b0; sign_out_r <= 1'b0;
            fuse_act_r <= 2'd0; act_slope_r <= 16'sd8192;
            for (ci = 0; ci < MAX_K*MAX_K; ci++)
                for (cj = 0; cj < P_IN; cj++)
                    coeff[ci][cj] <= 16'sd0;
            for (oi = 0; oi < C_OUT; oi++) begin
                bias_lo[oi] <= 16'd0; bias_hi[oi] <= 16'd0;
            end
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_KSIZE   : ksize_r <= cfg_data[3:0];
                CFG_SHIFT   : shift_r <= cfg_data[3:0];
                CFG_W       : width_r <= cfg_data;
                CFG_H       : height_r <= cfg_data;
                CFG_PAD_T   : pad_t_r <= cfg_data[3:0];
                CFG_PAD_B   : pad_b_r <= cfg_data[3:0];
                CFG_PAD_L   : pad_l_r <= cfg_data[3:0];
                CFG_PAD_R   : pad_r_r <= cfg_data[3:0];
                CFG_STR_X   : stride_x_r <= cfg_data[3:0];
                CFG_STR_Y   : stride_y_r <= cfg_data[3:0];
                CFG_CIN     : cin_r <= cfg_data[7:0];
                CFG_COUT    : cout_r <= cfg_data[7:0];
                CFG_OUT_W   : out_w_r <= cfg_data;
                CFG_OUT_H   : out_h_r <= cfg_data;
                CFG_SIGN    : begin
                    sign_in_r  <= cfg_data[0];
                    sign_out_r <= cfg_data[1];
                end
                CFG_ACT     : fuse_act_r <= cfg_data[1:0];
                CFG_ACT_SLOPE : act_slope_r <= cfg_data;
                default: begin
                    if (cfg_addr >= CFG_COEFF_BASE &&
                        cfg_addr < CFG_COEFF_BASE + 16'd800) begin
                        ci = (cfg_addr - CFG_COEFF_BASE) / P_IN;
                        cj = (cfg_addr - CFG_COEFF_BASE) % P_IN;
                        if (ci < MAX_K*MAX_K)
                            coeff[ci][cj] <= cfg_data;
                    end else if (cfg_addr >= CFG_BIAS_LO_BASE &&
                                 cfg_addr < CFG_BIAS_LO_BASE + 16'd1) begin
                        bias_lo[cfg_addr - CFG_BIAS_LO_BASE] <= cfg_data;
                    end else if (cfg_addr >= CFG_BIAS_HI_BASE &&
                                 cfg_addr < CFG_BIAS_HI_BASE + 16'd1) begin
                        bias_hi[cfg_addr - CFG_BIAS_HI_BASE] <= cfg_data;
                    end
                end
            endcase
        end
    end

    // ------------------------- stream control ---------------------------
    // States: IDLE (waiting for GO), STREAM (accepting input vectors and
    // computing), FLUSH (draining the last positions after the input ends).
    typedef enum logic [1:0] { ST_IDLE, ST_STREAM, ST_FLUSH } state_t;
    state_t state;
    logic stall;
    assign stall = out_valid && !out_ready;

    // Tap counter: 0..4, the tap-row being computed this cycle. Advances
    // every cycle while streaming (and not stalled). The window advances
    // on tap==4 (the last tap cycle of a group), when the input vector
    // is captured.
    logic [2:0] tap;
    // take: the input vector is captured into the window this cycle.
    logic take;
    assign take = (state == ST_STREAM) && (tap == 3'd4) &&
                  in_valid && !stall;
    assign in_ready = (state == ST_STREAM) && (tap == 3'd4) && !stall;
    // win_adv: the window/line-buffer datapath advances on `take`.
    // After the last input vector, the window must keep advancing (flush)
    // so the last output positions load; during the flush it advances
    // once per 5-cycle tap group (at tap==4), the same rhythm as `take`.
    logic win_adv;
    logic stream_done;
    assign win_adv = take || ((state == ST_FLUSH) && (tap == 3'd4));

    // Stream position (input raster): counts input vectors accepted.
    logic [15:0] in_r, in_c;
    // Advance counter (mac_r/mac_c): counts win_adv's in padded
    // coordinates; delayed by MAX_K through pos_r/pos_c to give the
    // window position (wpos_r/wpos_c).
    logic [15:0] mac_r, mac_c;
    // win_valid: the current window corresponds to a real output position.
    logic win_valid;
    // win_r/win_c: the window position captured at the win_adv edge
    // (the pre-edge wpos_r/wpos_c, which is the position the window now
    // holds). win_valid is set from match_out on the same edge, so
    // (win_r, win_c, win_valid) are consistent; the flag pipeline uses
    // these, NOT the live (wpos_r, wpos_c), which has already advanced
    // to the next position by the time the tap group runs.
    logic [15:0] win_r, win_c;
    logic match_out;

    // ------------------------- line buffers + window --------------------
    // 4 line buffers (K-1), each LB_DEPTH x 256-bit. Row 0 = oldest.
    // The wrap is the padded row width: the buffer delays by exactly
    // one padded row (width + pad_l + pad_r), not by DEPTH.
    logic [15:0] pw;
    assign pw = width_r + {12'd0, pad_l_r} + {12'd0, pad_r_r};
    logic [P_IN*8-1:0] lb_out [0:MAX_K-2];
    logic [P_IN*8-1:0] lb_in  [0:MAX_K-2];
    assign lb_in[0] = in_vec;
    genvar gi;
    generate
        for (gi = 0; gi < MAX_K-1; gi++) begin : lb_chain
            if (gi > 0) assign lb_in[gi] = lb_out[gi-1];
            line_buffer_bram_mc #(.WIDTH(P_IN*8), .DEPTH(LB_DEPTH),
                .ADDR_W(LB_ADDR_W)) lb_inst (
                .clk(clk), .rst_n(rst_n),
                .en(win_adv),
                .din(lb_in[gi]),
                .wrap(pw[LB_ADDR_W-1:0]),
                .dout(lb_out[gi])
            );
        end
    endgenerate
    // row_in[0..4]: the 5 window rows. row 0 = oldest line-buffer output,
    // row 4 = current input row.
    logic [P_IN*8-1:0] row_in [0:MAX_K-1];
    genvar gi2;
    generate
        for (gi2 = 0; gi2 < MAX_K; gi2++) begin : row_assign
            if (gi2 < MAX_K-1)
                assign row_in[gi2] = lb_out[MAX_K-2-gi2];
            else
                assign row_in[gi2] = in_vec;
        end
    endgenerate
    // aln: per-row shift registers (K-1 deep) holding the recent columns;
    // acol: the column entering the window this advance.
    logic [P_IN*8-1:0] aln [0:MAX_K-1][0:MAX_K-2];
    logic [P_IN*8-1:0] acol [0:MAX_K-1];
    integer ai, aj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ai = 0; ai < MAX_K; ai++)
                for (aj = 0; aj < MAX_K-1; aj++) aln[ai][aj] <= '0;
        end else if (!stall && win_adv) begin
            for (ai = 0; ai < MAX_K; ai++) begin
                aln[ai][0] <= row_in[ai];
                for (aj = 1; aj < MAX_K-1; aj++) aln[ai][aj] <= aln[ai][aj-1];
            end
        end
    end
    genvar gi3;
    generate
        for (gi3 = 0; gi3 < MAX_K; gi3++) begin : acol_assign
            if (gi3 == 0) assign acol[gi3] = row_in[gi3];
            else           assign acol[gi3] = aln[gi3][gi3-1];
        end
    endgenerate
    // win: the KxK window of 256-bit cells. win[t][j] is tap-row t,
    // tap-column j. Shifts left each advance; the new column is acol.
    logic [P_IN*8-1:0] win [0:MAX_K-1][0:MAX_K-1];
    integer wi, wj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wi = 0; wi < MAX_K; wi++)
                for (wj = 0; wj < MAX_K; wj++) win[wi][wj] <= '0;
        end else if (!stall && win_adv) begin
            for (wi = 0; wi < MAX_K; wi++) begin
                for (wj = 0; wj < MAX_K-1; wj++) win[wi][wj] <= win[wi][wj+1];
                win[wi][MAX_K-1] <= acol[wi];
            end
        end
    end

    // ------------------------- position / decimation --------------------
    // The window position lags the stream by the line-buffer latency
    // (MAX_K advances: MAX_K-1 line buffers + 1 BRAM read), exactly as
    // in conv2d_outpar. mac_r/mac_c counts win_adv's in padded
    // coordinates (it keeps advancing during the flush, which the
    // padded tail needs); pos_r/pos_c delays it by MAX_K advances, and
    // wpos_r/wpos_c is the window's bottom-right corner. Matching on
    // the undelayed mac would fire MAX_K advances too early, before
    // the window is filled (the window would be a diagonal shear).
    // Positions reset while IDLE (so a new frame starts from 0), and
    // (win_r, win_c) capture the window position at the win_adv edge.
    logic [15:0] pos_r [0:MAX_K];
    logic [15:0] pos_c [0:MAX_K];
    logic [15:0] wpos_r, wpos_c;
    assign wpos_r = pos_r[POS_LAG];
    assign wpos_c = pos_c[POS_LAG];
    integer pi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_r <= 16'd0; in_c <= 16'd0;
            mac_r <= 16'd0; mac_c <= 16'd0;
            for (pi = 0; pi <= MAX_K; pi++) begin
                pos_r[pi] <= 16'd0; pos_c[pi] <= 16'd0;
            end
            win_valid <= 1'b0;
            win_r <= 16'd0; win_c <= 16'd0;
        end else if (state == ST_IDLE) begin
            in_r <= 16'd0; in_c <= 16'd0;
            mac_r <= 16'd0; mac_c <= 16'd0;
            for (pi = 0; pi <= MAX_K; pi++) begin
                pos_r[pi] <= 16'd0; pos_c[pi] <= 16'd0;
            end
            win_valid <= 1'b0;
            win_r <= 16'd0; win_c <= 16'd0;
        end else if (!stall && win_adv) begin
            // stream position advances only on a real input take
            if (take) begin
                if (in_c == width_r - 1) begin
                    in_c <= 16'd0;
                    in_r <= in_r + 16'd1;
                end else begin
                    in_c <= in_c + 16'd1;
                end
            end
            // advance counter advances on every win_adv (incl. flush)
            if (mac_c == width_r + pad_l_r + pad_r_r - 1) begin
                mac_c <= 16'd0;
                mac_r <= mac_r + 16'd1;
            end else begin
                mac_c <= mac_c + 16'd1;
            end
            // delay the advance counter by MAX_K: the window loaded
            // on this advance corresponds to the position the stream
            // had MAX_K advances ago.
            pos_r[0] <= mac_r;
            pos_c[0] <= mac_c;
            for (pi = 1; pi <= MAX_K; pi++) begin
                pos_r[pi] <= pos_r[pi-1];
                pos_c[pi] <= pos_c[pi-1];
            end
            win_valid <= match_out;
            win_r <= wpos_r;
            win_c <= wpos_c;
        end
    end
    // match_out: the window position (wpos_r, wpos_c) is the window's
    // BOTTOM-RIGHT corner (the window is loaded as the stream delivers
    // it). wpos is already in PADDED coordinates, so an output
    // position matches when wpos equals the bottom-right of that
    // output's window: (or*sy + K - 1, oc*sx + K - 1). (No pad term:
    // the pad is already inside the padded coordinates.) Comparing wpos
    // to the TOP-LEFT instead would fire K-1 rows/cols too early,
    // before the window is filled.
    assign match_out =
        (wpos_r >= ksize_r - 1) &&
        (wpos_r < ksize_r - 1 + out_h_r * stride_y_r) &&
        (wpos_c >= ksize_r - 1) &&
        (wpos_c < ksize_r - 1 + out_w_r * stride_x_r) &&
        (((wpos_r - ksize_r + 1) % stride_y_r) == 0) &&
        (((wpos_c - ksize_r + 1) % stride_x_r) == 0);

    // ------------------------- state machine ---------------------------
    // The flush runs until every output has been emitted (emit_cnt
    // reaches out_w*out_h), not a fixed advance count: the number of
    // window advances needed after the last input depends on the frame
    // size, and a fixed count is too short for larger frames.
    // FLUSH_MAX is a safety cap so a bug cannot hang the engine.
    logic [15:0] flush_cnt;
    localparam int FLUSH_MAX = 20000;
    // emit_cnt: outputs emitted so far in this frame; the flush ends
    // when it reaches out_w*out_h. (Declared here, before the state
    // machine that reads it; it is written in the accumulate block.)
    logic [15:0] emit_cnt;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            tap <= 3'd0;
            stream_done <= 1'b0;
            flush_cnt <= 16'd0;
        end else if (!stall) begin
            case (state)
                ST_IDLE: begin
                    tap <= 3'd0;
                    flush_cnt <= 16'd0;
                    if (go_pulse) state <= ST_STREAM;
                end
                ST_STREAM: begin
                    if (tap == 3'd4) tap <= 3'd0;
                    else             tap <= tap + 3'd1;
                    // the last input vector ends the stream
                    if (take && in_last) begin
                        state <= ST_FLUSH;
                        stream_done <= 1'b1;
                        flush_cnt <= 16'd0;
                    end
                end
                ST_FLUSH: begin
                    if (tap == 3'd4) tap <= 3'd0;
                    else             tap <= tap + 3'd1;
                    flush_cnt <= flush_cnt + 16'd1;
                    if (emit_cnt >= out_w_r * out_h_r) begin
                        state <= ST_IDLE;
                        stream_done <= 1'b0;
                    end else if (flush_cnt >= FLUSH_MAX) begin
                        state <= ST_IDLE;
                        stream_done <= 1'b0;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
    assign busy = (state != ST_IDLE);

    // ------------------------- tap-row compute --------------------------
    // Products for the current tap-row: 5 taps x 32 channels = 160.
    // prod_comb[k] = win_tap[j][c] * cf_tap[j][c], k=j*P_IN+c.
    //
    // The kernel sits at the BOTTOM-RIGHT of the MAX_K x MAX_K window
    // (the L1 convention): kernel row `tap` reads window row
    // wrow = tap + (MAX_K-K), kernel column j reads window column
    // j + (MAX_K-K). For K=5 the offset is 0; for K=3 it is 2
    // (kernel rows/cols 0..2 read window rows/cols 2..4). The coeff
    // row is `tap` and the coeff column is `pj` (the TB writes the
    // coeff into the MAX_K x MAX_K grid, so coeff[tap*MAX_K+pj] is the
    // kernel (tap, pj) entry for any K).
    //
    // All selections use case statements on plain signals (constant
    // indices). A direct variable-index read win[wr][wc] inside
    // always @* goes stale in Icarus (its sensitivity misses the
    // variable index), which is why the window row is first selected
    // by wrow, then the columns are shifted by ksize into win_tap.
    logic [2:0] wrow;
    assign wrow = tap + (3'd5 - ksize_r[2:0]);
    logic [P_IN*8-1:0] win_tap [0:MAX_K-1];
    logic signed [15:0] cf_tap [0:MAX_K-1][0:P_IN-1];
    integer tj;
    integer tc;
    always @* begin
        // Window row (wrow): the kernel sits at the bottom-right of
        // the window, so kernel row `tap` reads window row wrow.
        // (A two-stage select through an intermediate win_row signal
        // read in the same always @* block misbehaves in Icarus, so
        // win_tap is selected directly here.)
        case (wrow)
            3'd0: for (tj = 0; tj < MAX_K; tj++) win_tap[tj] = win[0][tj];
            3'd1: for (tj = 0; tj < MAX_K; tj++) win_tap[tj] = win[1][tj];
            3'd2: for (tj = 0; tj < MAX_K; tj++) win_tap[tj] = win[2][tj];
            3'd3: for (tj = 0; tj < MAX_K; tj++) win_tap[tj] = win[3][tj];
            default: for (tj = 0; tj < MAX_K; tj++) win_tap[tj] = win[4][tj];
        endcase
        // Coeff row for kernel row `tap`.
        case (tap)
            3'd0: begin
                for (tj = 0; tj < MAX_K; tj++)
                    for (tc = 0; tc < P_IN; tc++)
                        cf_tap[tj][tc] = coeff[0*MAX_K+tj][tc];
            end
            3'd1: begin
                for (tj = 0; tj < MAX_K; tj++)
                    for (tc = 0; tc < P_IN; tc++)
                        cf_tap[tj][tc] = coeff[1*MAX_K+tj][tc];
            end
            3'd2: begin
                for (tj = 0; tj < MAX_K; tj++)
                    for (tc = 0; tc < P_IN; tc++)
                        cf_tap[tj][tc] = coeff[2*MAX_K+tj][tc];
            end
            3'd3: begin
                for (tj = 0; tj < MAX_K; tj++)
                    for (tc = 0; tc < P_IN; tc++)
                        cf_tap[tj][tc] = coeff[3*MAX_K+tj][tc];
            end
            default: begin
                for (tj = 0; tj < MAX_K; tj++)
                    for (tc = 0; tc < P_IN; tc++)
                        cf_tap[tj][tc] = coeff[4*MAX_K+tj][tc];
            end
        endcase
    end
    // prod_comb is combinational; prod registers it (one cycle, the
    // L1 pattern), so the tree output arrives 5 cycles after the tap
    // (1 prod register + 4 tree stages) and the 5-deep flag pipeline
    // (index 4) lines up.
    logic signed [24:0] prod_comb [0:MAX_K*P_IN-1];
    logic signed [24:0] prod [0:MAX_K*P_IN-1];
    // Flat pack for the adder tree: Yosys 0.52 rejects unpacked array
    // ports, so the tree takes in_flat (element i in bits [i*25 +: 25]);
    // the values are unchanged.
    logic signed [MAX_K*P_IN*25-1:0] prod_flat;
    genvar pk;
    generate
        for (pk = 0; pk < MAX_K*P_IN; pk++) begin : pack
            assign prod_flat[pk*25 +: 25] = prod[pk];
        end
    endgenerate
    genvar pj, pc;
    generate
        for (pj = 0; pj < MAX_K; pj++) begin : tap_prod
            for (pc = 0; pc < P_IN; pc++) begin : ch_prod
                // px is 9 bits: an unsigned byte zero-extends to 0..255
                // (9 bits), a signed byte sign-extends to -128..127.
                // Declaring px 8 bits would truncate the zero-extension
                // back to 8 bits, so bytes >= 128 would go negative.
                logic signed [8:0] px;
                logic signed [15:0] cf;
                always @* begin
                    // Out-of-kernel taps/columns (tap/pj >= ksize)
                    // contribute zero (their coeff is unwritten/zero,
                    // but guard explicitly so no X leaks in).
                    if ((tap < ksize_r[2:0]) && (pj < ksize_r)) begin
                        if (sign_in_r)
                            px = $signed(win_tap[pj][pc*8 +: 8]);
                        else
                            px = $signed({1'b0, win_tap[pj][pc*8 +: 8]});
                        cf = cf_tap[pj][pc];
                    end else begin
                        px = 9'sd0;
                        cf = 16'sd0;
                    end
                end
                assign prod_comb[pj*P_IN+pc] = px * cf;
            end
        end
    endgenerate
    integer pr;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (pr = 0; pr < MAX_K*P_IN; pr++) prod[pr] <= 25'sd0;
        end else if (!stall) begin
            for (pr = 0; pr < MAX_K*P_IN; pr++) prod[pr] <= prod_comb[pr];
        end
    end
    // Row-sum via the pipelined adder tree (N=160, latency 4 cycles).
    logic signed [32:0] row_sum;
    adder_tree #(.N(MAX_K*P_IN), .W(25)) tree_inst (
        .clk(clk), .rst_n(rst_n), .stall(stall),
        .in_flat(prod_flat), .out(row_sum)
    );

    // ------------------------- flag pipeline ---------------------------
    // The tree output arrives 5 cycles after the tap (1 prod register
    // + 4 tree stages, registered at edge T+5, valid during T+6); a
    // 5-deep flag pipeline tracks the tap-row and the framing for each
    // tree output. flag[0] is loaded with the current tap/position
    // each cycle; flag[4] is the one whose row-sum is arriving now
    // (same construction as conv2d_outpar / conv1x1_cp, whose emit
    // reads index 4).
    logic [2:0]  flag_tap [0:4];
    logic        flag_v   [0:4];
    logic        flag_sof [0:4];
    logic        flag_eol [0:4];
    logic        flag_last[0:4];
    integer fi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (fi = 0; fi < 5; fi++) begin
                flag_tap[fi] <= 3'd0;
                flag_v[fi] <= 1'b0;
                flag_sof[fi] <= 1'b0;
                flag_eol[fi] <= 1'b0;
                flag_last[fi] <= 1'b0;
            end
        end else if (!stall) begin
            flag_tap[0] <= tap;
            flag_v[0]   <= win_valid;
            // Framing from the captured window position (win_r, win_c),
            // which is the stream position = the window's bottom-right.
            // The output column is (win_c-K+1)/stride_x, so the last
            // column is at win_c = K-1+(out_w-1)*stride_x.
            flag_sof[0] <= win_valid && (win_r == ksize_r - 1) &&
                           (win_c == ksize_r - 1);
            flag_eol[0] <= win_valid &&
                           (win_c == ksize_r - 1 + (out_w_r - 1) * stride_x_r);
            flag_last[0] <= win_valid &&
                            (win_c == ksize_r - 1 + (out_w_r - 1) * stride_x_r) &&
                            (win_r == ksize_r - 1 + (out_h_r - 1) * stride_y_r);
            for (fi = 1; fi < 5; fi++) begin
                flag_tap[fi] <= flag_tap[fi-1];
                flag_v[fi]   <= flag_v[fi-1];
                flag_sof[fi] <= flag_sof[fi-1];
                flag_eol[fi] <= flag_eol[fi-1];
                flag_last[fi] <= flag_last[fi-1];
            end
        end
    end

    // ------------------------- accumulate + emit -----------------------
    // The row-sums arrive one per cycle. acc accumulates them for the
    // current position; when the tap-row-4 row-sum arrives the position
    // is complete and is latched into the emit register.
    logic signed [32:0] acc;
    logic [2:0] acc_cnt;
    logic signed [32:0] emit_val;
    logic signed [32:0] emit_rnd;
    logic signed [32:0] emit_sh;
    logic signed [32:0] emit_act;
    logic signed [48:0] leaky_prod;
    logic signed [32:0] leaky_neg;
    logic [7:0] emit_px;
    logic emit_v;
    logic emit_sof, emit_eol, emit_last;
    // The emit register holds the completed position until out_ready.
    logic        out_v_r;
    logic [7:0]  out_px_r;
    logic        out_sof_r, out_eol_r, out_last_r;

    // bias (single output channel)
    logic signed [32:0] bias_full;
    assign bias_full = $signed({{1{bias_hi[0][15]}}, bias_hi[0], bias_lo[0]});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 33'sd0; acc_cnt <= 3'd0;
            out_v_r <= 1'b0; out_px_r <= 8'd0;
            out_sof_r <= 1'b0; out_eol_r <= 1'b0; out_last_r <= 1'b0;
            emit_cnt <= 16'd0;
        end else if (state == ST_IDLE) begin
            emit_cnt <= 16'd0;
        end else if (!stall) begin
            // A row-sum arrives this cycle (the tree output is valid
            // whenever the flag pipeline says the input was valid).
            if (flag_v[4]) begin
                if (flag_tap[4] == 3'd0) begin
                    // first tap-row of a position: start from the bias
                    acc <= bias_full + row_sum;
                    acc_cnt <= 3'd1;
                end else if (flag_tap[4] == 3'd4) begin
                    // last tap-row: the position is complete
                    acc <= 33'sd0;
                    acc_cnt <= 3'd0;
                end else begin
                    acc <= acc + row_sum;
                    acc_cnt <= acc_cnt + 3'd1;
                end
            end
            // Emit when the tap-row-4 row-sum arrives.
            if (flag_v[4] && (flag_tap[4] == 3'd4)) begin
                out_v_r <= 1'b1;
                out_sof_r <= flag_sof[4];
                out_eol_r <= flag_eol[4];
                out_last_r <= flag_last[4];
                emit_cnt <= emit_cnt + 16'd1;
            end else if (out_v_r && out_ready) begin
                out_v_r <= 1'b0;
            end
        end
    end

    // emit_val is the completed accumulator (acc + last row-sum).
    assign emit_val = acc + row_sum;
    always @* begin
        emit_rnd = emit_val + ((shift_r == 0) ? 33'sd0 :
            (33'sd1 <<< (shift_r - 1'b1)));
    end
    assign emit_sh = emit_rnd >>> shift_r;
    // The slope is an UNSIGNED Q0.16 fraction (0x8000 = 0.5):
    // zero-extend to 17 bits so 32768 stays +32768.
    assign leaky_prod = emit_sh * $signed({1'b0, act_slope_r});
    assign leaky_neg = leaky_prod[48:16];
    always @* begin
        if (fuse_act_r == 2'd1)
            emit_act = (emit_sh < 33'sd0) ? 33'sd0 : emit_sh;
        else if (fuse_act_r == 2'd2)
            emit_act = (emit_sh < 33'sd0) ? leaky_neg : emit_sh;
        else
            emit_act = emit_sh;
    end
    always @* begin
        if (sign_out_r) begin
            if (emit_act > 33'sd127)        emit_px = 8'd127;
            else if (emit_act < -33'sd128)  emit_px = 8'h80;
            else                             emit_px = emit_act[7:0];
        end else begin
            if (emit_act > 33'sd255)        emit_px = 8'd255;
            else if (emit_act < 33'sd0)     emit_px = 8'd0;
            else                             emit_px = emit_act[7:0];
        end
    end
    // The emit value is captured into the output register when the
    // position completes; out_px_r holds it until consumed.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_px_r <= 8'd0;
        end else if (!stall) begin
            if (flag_v[4] && (flag_tap[4] == 3'd4))
                out_px_r <= emit_px;
        end
    end

    assign pixel_out = out_px_r;
    assign out_valid = out_v_r;
    assign out_sof   = out_sof_r;
    assign out_eol   = out_eol_r;
    assign out_last  = out_last_r;
endmodule
