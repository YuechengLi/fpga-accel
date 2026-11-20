//=============================================================================
// conv2d_outpar.sv -- Stage 7b: output-channel-parallel dense convolution.
//
// L1 of SRCNN (9x9, 1 -> 64): one shared KxK window feeds P_OUT parallel
// MAC units (one per output channel), so all 64 output channels for one
// spatial position are produced in ceil(64/P_OUT) cycles instead of 64
// serial passes (the conv2d_mc shape, which is input-channel-parallel
// and leaves 3 of 4 lanes idle for C_in=1, is L1-bound at 2.3 fps).
// Raster streaming (no tiles): the input frame streams in raster order
// through (K-1) line buffers + a KxK window; the window advances once
// per output position (P_OUT groups, one group per cycle), and the
// input stream is throttled to the same rhythm (in_ready pulses once
// per P_OUT cycles during compute, every cycle during the initial
// line-buffer fill -- the fill is short, so the uniform 1-per-P_OUT
// rhythm is used throughout for a simpler control).
//
// Weights are loaded ONCE at startup into registers (64 outputs x KxK
// coefficients + 64 biases); the per-cycle group select is a mux, so
// there is no per-tile/per-pass coefficient reloading and no ping-pong
// bank. Config bus (same conventions as conv2d_mc):
//   0x00 ksize (1/3/5/7/9)   0x01 shift (0..15)
//   0x02 width  (input W)    0x03 height (input H)
//   0x04 pad_top  0x05 pad_bottom  0x06 pad_left  0x07 pad_right
//   0x08 stride_x  0x09 stride_y   0x0A C_in (1)  0x0B C_out (<=64)
//   0x0D out_w  0x0E out_h (host-computed, for emit framing)
//   0x0F sign (bit0 signed in, bit1 signed out)
//   0x10 fused act (0=none, 1=ReLU, 2=Leaky)  0x11 Leaky slope (Q0.16)
//   0x0C GO (write 1 = start the frame)
//   0x100 + o*128 + tap : coeff for output channel o, tap r*MAX_K+c
//     (tap < K*K; stride 128 so o = offset[10:7], tap = offset[6:0])
//   0x2100 + o : bias low 16 bits for output channel o
//   0x2140 + o : bias high 16 bits for output channel o
//   The bias bases sit AFTER the coeff range (0x100..0x20FF); putting
//   them inside it would route the bias writes into the coeff array.
// Output: P_OUT bytes/cycle (one per output channel in the group) +
//   out_valid + framing (sof/eol/last for the tile/frame). The group
//   index rides with the flags; the consumer (L2 gather) reassembles
//   the full 64-channel position vector over ceil(64/P_OUT) cycles.
// Pipeline latency (window load -> emit): 7 cycles. A single stall
//   signal (out_valid && !out_ready) freezes all state.
//=============================================================================
`timescale 1ns/1ps

module conv2d_outpar #(
    parameter int P_OUT    = 16,
    parameter int MAX_K    = 9,
    parameter int LB_DEPTH = 2048,
    parameter int LB_ADDR_W = 11
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (1-channel frame, raster order)
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (P_OUT channels per cycle, group-major within a position)
    output logic [P_OUT*8-1:0] pixel_out,
    output logic        out_valid,
    output logic [3:0]  out_group,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [15:0] CFG_KSIZE   = 16'h0000;
    localparam logic [15:0] CFG_SHIFT   = 16'h0001;
    localparam logic [15:0] CFG_WIDTH   = 16'h0002;
    localparam logic [15:0] CFG_HEIGHT  = 16'h0003;
    localparam logic [15:0] CFG_PAD_T   = 16'h0004;
    localparam logic [15:0] CFG_PAD_B   = 16'h0005;
    localparam logic [15:0] CFG_PAD_L   = 16'h0006;
    localparam logic [15:0] CFG_PAD_R   = 16'h0007;
    localparam logic [15:0] CFG_STRIDE_X = 16'h0008;
    localparam logic [15:0] CFG_STRIDE_Y = 16'h0009;
    localparam logic [15:0] CFG_CIN     = 16'h000A;
    localparam logic [15:0] CFG_COUT    = 16'h000B;
    localparam logic [15:0] CFG_GO      = 16'h000C;
    localparam logic [15:0] CFG_OUT_W   = 16'h000D;
    localparam logic [15:0] CFG_OUT_H   = 16'h000E;
    localparam logic [15:0] CFG_SIGN    = 16'h000F;
    localparam logic [15:0] CFG_ACT     = 16'h0010;
    localparam logic [15:0] CFG_ACT_SLOPE = 16'h0011;
    localparam logic [15:0] CFG_COEFF_BASE = 16'h0100;
    // Bias bases sit AFTER the coeff range (0x100..0x20FF, 8192 entries):
    // putting them at 0x300/0x400 would overlap the coeff addresses and
    // the bias writes would land in the coeff array instead (the bias
    // would silently never be applied).
    localparam logic [15:0] CFG_BIAS_LO_BASE = 16'h2100;
    localparam logic [15:0] CFG_BIAS_HI_BASE = 16'h2140;

    localparam int NK = MAX_K * MAX_K;    // 81
    // NOTE: the group count is RUNTIME (ngroup, declared below, from
    // cout_r / P_OUT) -- the old compile-time NGROUP = 64/P_OUT broke
    // C_out < 64: the group counter wrapped at 3 even for C_out=16,
    // emitting phantom groups 1..3.

    //------------------------- config registers ----------------------------
    logic [3:0]  ksize;
    logic [3:0]  shift_r;
    logic [11:0] width_r, height_r;
    logic [3:0]  pad_t, pad_b, pad_l, pad_r;
    logic [3:0]  stride_x, stride_y;
    logic [6:0]  cin_r, cout_r;
    logic [12:0] out_w_r, out_h_r;
    logic        sign_in_r, sign_out_r;
    logic [1:0]  fuse_act_r;
    logic signed [15:0] act_slope_r;
    logic signed [15:0] coeff [0:63][0:NK-1];
    // Bias as two 16-bit arrays (a bit-select write into a memory word
    // is not reliable in Icarus; the read side concatenates them).
    logic [15:0] bias_lo [0:63];
    logic [15:0] bias_hi [0:63];
    int ci, cj;

    logic go_pulse;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) go_pulse <= 1'b0;
        else go_pulse <= cfg_valid && cfg_we && (cfg_addr == CFG_GO) &&
                         (cfg_data[0] == 1'b1);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ksize    <= 4'd3;
            shift_r  <= 4'd8;
            width_r  <= 12'd16;
            height_r <= 12'd16;
            pad_t <= 4'd0; pad_b <= 4'd0; pad_l <= 4'd0; pad_r <= 4'd0;
            stride_x <= 4'd1; stride_y <= 4'd1;
            cin_r <= 7'd1;
            cout_r <= 7'd1;
            out_w_r <= 13'd16;
            out_h_r <= 13'd16;
            sign_in_r <= 1'b0;
            sign_out_r <= 1'b0;
            fuse_act_r <= 2'd0;
            act_slope_r <= 16'sd8192;
            for (ci = 0; ci < 64; ci++) begin
                for (cj = 0; cj < NK; cj++) coeff[ci][cj] <= '0;
                bias_lo[ci] <= 16'd0;
                bias_hi[ci] <= 16'd0;
            end
            coeff[0][1*MAX_K+1] <= 16'sd256;
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
                CFG_CIN   : cin_r    <= cfg_data[6:0];
                CFG_COUT  : cout_r   <= cfg_data[6:0];
                CFG_OUT_W : out_w_r  <= cfg_data[12:0];
                CFG_OUT_H : out_h_r  <= cfg_data[12:0];
                CFG_SIGN  : begin
                    sign_in_r  <= cfg_data[0];
                    sign_out_r <= cfg_data[1];
                end
                CFG_ACT   : fuse_act_r <= cfg_data[1:0];
                CFG_ACT_SLOPE : act_slope_r <= cfg_data;
                default: begin
                    if (cfg_addr >= CFG_COEFF_BASE &&
                        cfg_addr < CFG_COEFF_BASE + 16'd8192) begin
                        ci = (cfg_addr - CFG_COEFF_BASE) >> 7;
                        cj = (cfg_addr - CFG_COEFF_BASE) & 16'h007F;
                        if (cj < NK)
                            coeff[ci][cj] <= cfg_data;
                    end else if (cfg_addr >= CFG_BIAS_LO_BASE &&
                               cfg_addr < CFG_BIAS_LO_BASE + 16'd64) begin
                        bias_lo[cfg_addr - CFG_BIAS_LO_BASE] <= cfg_data;
                    end else if (cfg_addr >= CFG_BIAS_HI_BASE &&
                               cfg_addr < CFG_BIAS_HI_BASE + 16'd64) begin
                        bias_hi[cfg_addr - CFG_BIAS_HI_BASE] <= cfg_data;
                    end
                end
            endcase
        end
    end

    //------------------------- derived geometry ---------------------------
    logic [12:0] pw, ph;
    logic [12:0] frame_px;
    assign pw = {1'b0, width_r} + {9'd0, pad_l} + {9'd0, pad_r};
    assign ph = {1'b0, height_r} + {9'd0, pad_t} + {9'd0, pad_b};
    assign frame_px = pw * ph;

    // Runtime group count: C_out / P_OUT (C_out is runtime-configured;
    // the pipeline always uses C_out=64, ngroup=4). Clamped to >= 1 so
    // the reset default C_out=1 cannot stall the engine (group ==
    // ngroup-1 would never fire for ngroup=0).
    logic [3:0] ngroup;
    assign ngroup = (cout_r <= P_OUT) ? 4'd1 : (cout_r / P_OUT);

    //------------------------- state machine -------------------------------
    typedef enum logic [1:0] { ST_IDLE, ST_STREAM } state_t;
    state_t state;
    logic stall;
    assign stall = out_valid && !out_ready;
    assign busy = (state == ST_STREAM);

    // Group counter: cycles 0..ngroup-1 per output position; the stream
    // advances (and the window loads) on group==ngroup-1 (the LAST group
    // cycle). This phasing matters: the window loaded at the advance edge
    // is the one the NEXT group-0 cycle's MAC uses, so a window's groups
    // are computed (and emitted) in the natural order 0,1,2,3. Advancing
    // on group==0 instead would load the window at the end of the
    // group-0 cycle, so group 0's MAC would use the previous window and
    // the emission order would come out 1,2,3,0.
    // `take` is an advance cycle on which the source actually presents a
    // pixel; the STREAM position advances only on `take`, so in_valid
    // gaps stall the stream without desynchronising the group rhythm.
    // `win_adv` is the enable for the window datapath (line buffers, aln,
    // window, position shift register, decimation counters, win_valid).
    // While streaming it equals `take`; after the last pixel it keeps
    // pulsing on `advance` for FLUSH_ADV advances, because the window
    // lags the stream by MAX_K positions and the last output positions
    // only load once the window has flushed past the end of the frame.
    // (Without this the window freezes when in_valid drops and the last
    // outputs never appear -- the same role conv2d_mc's enable shift
    // register plays.) During the flush the line buffers shift in the
    // stale pixel_in, which is harmless: the window positions loaded
    // then are the last real outputs, whose data is already in aln.
    // in_ready is the uniform 1-per-4 rhythm (no fill fast-path: a
    // fast-path would accept pixels the line buffers do not consume).
    logic [3:0] group;
    logic advance;
    logic take;
    logic win_adv;
    logic [3:0] flush_adv_cnt;
    logic stream_done;
    localparam int FLUSH_ADV = 12;
    assign advance = (state == ST_STREAM) && !stall &&
                     (group == (ngroup - 1));
    assign take = advance && in_valid;
    // The flush advance must be a PULSE (once per group period, at
    // group==ngroup-1, via `advance`), not a level: without the
    // `advance` gate, win_adv stays high for the whole flush and the
    // window shifts every cycle (4x per group period), desyncing the
    // window from the group counter for the last output row (the
    // group slips by one at the row-7 boundary and outputs go missing).
    assign win_adv = take ||
        (stream_done && advance && (flush_adv_cnt < FLUSH_ADV[3:0]));
    assign in_ready = advance;

    // Pipeline-advance: a new (window, group) pair is computed this
    // cycle -- and only then may an entry enter the prod/flag pipeline.
    // The group advances through a window's groups one per cycle, and
    // the window advances on win_adv; but during an in_valid gap the
    // window does NOT advance, so the group must hold at ngroup-1 (not
    // wrap and recompute the same window's groups) and the pipeline
    // must take a bubble instead of a duplicate entry. (Without this,
    // a gap makes the pipeline re-emit the held window's groups: the
    // debug run showed every case-1 position emitted twice, so the
    // collector finished on 32 distinct positions and rows 4-7 were
    // never produced.) window_live is false once the flush advances
    // are exhausted (the window is past the last output); after that
    // no new entries are created and the state returns to IDLE.
    logic window_live;
    logic pipe_adv;
    assign window_live = !stream_done ||
                         (flush_adv_cnt < FLUSH_ADV[3:0]);
    assign pipe_adv = win_adv ||
                      ((group < (ngroup - 1)) && window_live);

    // Stream position (input frame, padded coordinates).
    logic [12:0] strm_r, strm_c;
    logic [6:0]  flush_cnt;
    logic [12:0] next_c, next_r;
    assign next_c = (strm_c == pw - 13'd1) ? 13'd0 : strm_c + 13'd1;
    assign next_r = (strm_c == pw - 13'd1) ? strm_r + 13'd1 : strm_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            group <= 4'd0;
            strm_r <= 13'd0;
            strm_c <= 13'd0;
            stream_done <= 1'b0;
            flush_cnt <= 7'd0;
            flush_adv_cnt <= 4'd0;
        end else if (!stall) begin
            case (state)
                ST_IDLE: begin
                    if (go_pulse) begin
                        state <= ST_STREAM;
                        group <= 4'd0;
                        strm_r <= 13'd0;
                        strm_c <= 13'd0;
                        stream_done <= 1'b0;
                        flush_cnt <= 7'd0;
                        flush_adv_cnt <= 4'd0;
                    end
                end
                ST_STREAM: begin
                    // Advance the group only when a new (window, group)
                    // is computed (pipe_adv); during an in_valid gap the
                    // group holds at ngroup-1 so the same window's
                    // groups are not recomputed. (win_adv implies
                    // group==ngroup-1, so the wrap only happens when
                    // the window advances.)
                    if (pipe_adv) begin
                        if (group == (ngroup - 1)) group <= 4'd0;
                        else group <= group + 4'd1;
                    end
                    if (take) begin
                        if ((strm_r == ph - 13'd1) &&
                            (strm_c == pw - 13'd1)) begin
                            strm_r <= 13'd0;
                            strm_c <= 13'd0;
                            stream_done <= 1'b1;
                            flush_cnt <= 7'd0;
                            flush_adv_cnt <= 4'd0;
                        end else begin
                            strm_r <= next_r;
                            strm_c <= next_c;
                        end
                    end else if (stream_done) begin
                        flush_cnt <= flush_cnt + 7'd1;
                        if (advance && (flush_adv_cnt < FLUSH_ADV[3:0]))
                            flush_adv_cnt <= flush_adv_cnt + 4'd1;
                        if (flush_cnt >= 7'd64) state <= ST_IDLE;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    //------------------------- line buffers + window -----------------------
    // One input channel: (MAX_K-1) line buffers in a chain, fed by the
    // stream on `advance`. The window is a MAX_K x MAX_K register array
    // loaded from the column shift registers on `advance` and held for
    // the ngroup group cycles. Same structure as conv2d_mc.s lane
    // (line_buffer_bram_mc + aln/acol), but the shift/load is gated by
    // `advance` (once per output position) instead of every cycle.
    logic [7:0] lb_out [0:MAX_K-2];
    logic [7:0] lb_in  [0:MAX_K-2];
    assign lb_in[0] = pixel_in;
    genvar gi;
    generate
        for (gi = 0; gi < MAX_K-1; gi++) begin : lb_chain
            if (gi > 0) assign lb_in[gi] = lb_out[gi-1];
            line_buffer_bram_mc #(.WIDTH(8), .DEPTH(LB_DEPTH),
                                  .ADDR_W(LB_ADDR_W)) u_lb (
                .clk(clk), .rst_n(rst_n),
                .en(win_adv),
                .din(lb_in[gi]),
                .wrap(pw[LB_ADDR_W-1:0]),
                .dout(lb_out[gi])
            );
        end
    endgenerate

    // Column shift registers: row_in[0..MAX_K-1] is the current column
    // for each window row (row 0 = oldest line-buffer output).
    logic [7:0] row_in [0:MAX_K-1];
    assign row_in[MAX_K-1] = pixel_in;
    genvar gi2;
    for (gi2 = 0; gi2 < MAX_K-1; gi2++) begin : row_assign
        assign row_in[gi2] = lb_out[MAX_K-2-gi2];
    end
    logic [7:0] aln [0:MAX_K-1][0:MAX_K-2];
    logic [7:0] acol [0:MAX_K-1];
    int ai, aj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ai = 0; ai < MAX_K; ai++)
                for (aj = 0; aj < MAX_K-1; aj++) aln[ai][aj] <= 8'd0;
        end else if (!stall && win_adv) begin
            for (ai = 0; ai < MAX_K; ai++) begin
                aln[ai][0] <= row_in[ai];
                for (aj = 1; aj < MAX_K-1; aj++) aln[ai][aj] <= aln[ai][aj-1];
            end
        end
    end
    genvar gi3;
    for (gi3 = 0; gi3 < MAX_K; gi3++) begin : acol_assign
        if (gi3 == 0) assign acol[gi3] = row_in[gi3];
        else           assign acol[gi3] = aln[gi3][gi3-1];
    end

    logic [7:0] win [0:MAX_K-1][0:MAX_K-1];
    int wi, wj;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wi = 0; wi < MAX_K; wi++)
                for (wj = 0; wj < MAX_K; wj++) win[wi][wj] <= 8'd0;
        end else if (!stall && win_adv) begin
            for (wi = 0; wi < MAX_K; wi++) begin
                for (wj = 0; wj < MAX_K-1; wj++) win[wi][wj] <= win[wi][wj+1];
                win[wi][MAX_K-1] <= acol[wi];
            end
        end
    end

    // Window position: the window's bottom-right lags the stream by
    // MAX_K-1 advances. (The newest pixel is row_in[MAX_K-1] delayed
    // MAX_K-1 advances through the aln column register; the diagonal
    // acol gives every row the same lag, so the window is an exact
    // patch, not a shear -- the per-stage BRAM read latency is
    // cancelled by the diagonal, not added.) mac is the window
    // position, sampled pre-edge at the load edge, so the shift
    // register holds MAX_K-1 positions and mac = pos[MAX_K-2]:
    // pre-edge mac is then the stream position MAX_K-1 advances ago,
    // which is the window's bottom-right. (An earlier version used
    // depth MAX_K; that made mac two advances too old, so match_out
    // fired two advances late and every output used the window two
    // columns to the right -- a debug window dump showed rows 0..8,
    // cols 2..10 for output (0,0). The L3 uses the same MAX_K-2 lag.)
    // Tracked in the position domain (shifts on `advance`, not per
    // cycle).
    logic [12:0] pos_r [0:MAX_K-2];
    logic [12:0] pos_c [0:MAX_K-2];
    int pi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (pi = 0; pi <= MAX_K-2; pi++) begin
                pos_r[pi] <= 13'd0; pos_c[pi] <= 13'd0;
            end
        end else if (!stall && win_adv) begin
            pos_r[0] <= strm_r;
            pos_c[0] <= strm_c;
            for (pi = 1; pi <= MAX_K-2; pi++) begin
                pos_r[pi] <= pos_r[pi-1];
                pos_c[pi] <= pos_c[pi-1];
            end
        end
    end
    logic [12:0] mac_r, mac_c;
    assign mac_r = pos_r[MAX_K-2];
    assign mac_c = pos_c[MAX_K-2];

    //------------------------- next-output (decimation) counters ----------
    // Same "next exceeds" method as Stage 1 (no divider). The counters
    // advance when a window position is an output; they are evaluated
    // against the window position (mac_r/mac_c) on the advance cycle.
    logic [12:0] nxt_or, nxt_oc, nxt_sr, nxt_sc;
    logic        match_out;
    logic is_eol_c, is_last_c;
    assign match_out = (mac_r == nxt_sr) && (mac_c == nxt_sc);
    assign is_eol_c  = match_out && ((nxt_sc + {9'd0, stride_x}) > (pw - 13'd1));
    assign is_last_c = is_eol_c  && ((nxt_sr + {9'd0, stride_y}) > (ph - 13'd1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= 13'd2; nxt_sc <= 13'd2;
        end else if (state == ST_IDLE) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= {9'd0, ksize} - 13'd1;
            nxt_sc <= {9'd0, ksize} - 13'd1;
        end else if (!stall && win_adv && match_out) begin
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

    // The window is a valid output position when it matches the next
    // output AND the stream has delivered enough rows (the window is
    // fully inside the frame). Captured on the advance cycle and held
    // for the ngroup group cycles.
    logic win_valid_r;
    logic [12:0] win_or_r, win_oc_r;
    logic win_eol_r, win_last_r;
    logic pos_is_real;
    assign pos_is_real = (mac_r >= {9'd0, pad_t}) &&
                         (mac_r <  {9'd0, pad_t} + {1'b0, height_r}) &&
                         (mac_c >= {9'd0, pad_l}) &&
                         (mac_c <  {9'd0, pad_l} + {1'b0, width_r});
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win_valid_r <= 1'b0;
            win_or_r <= 13'd0; win_oc_r <= 13'd0;
            win_eol_r <= 1'b0; win_last_r <= 1'b0;
        end else if (!stall && win_adv) begin
            win_valid_r <= match_out && pos_is_real;
            win_or_r <= nxt_or;
            win_oc_r <= nxt_oc;
            win_eol_r <= is_eol_c;
            win_last_r <= is_last_c;
        end
    end

    //------------------------- per-unit MAC + tree -------------------------
    // Each unit u computes output channel (group*P_OUT + u) for the
    // held window: NK products (registered) -> adder_tree (4 cycles)
    // -> + bias -> shift -> fused act -> saturate -> emit.
    // The group index rides the flag pipeline with the data.
    logic signed [32:0] tree_out [0:P_OUT-1];
    logic [3:0] group_d [0:4];
    logic       prod_v_d [0:4];
    logic [12:0] prod_or_d [0:4];
    logic [12:0] prod_oc_d [0:4];
    logic       prod_eol_d [0:4];
    logic       prod_last_d [0:4];
    int fi;
    // Flag pipeline: prod (stage 0) -> 4 tree-delay stages -> emit.
    // The flags enter at the prod stage (one group per cycle) and exit
    // at the emit stage, aligned with the data by construction: the
    // tree output is registered at edge T+4 (valid during T+5), and the
    // flags sit at index 4 during T+5, so the emit logic and the output
    // register (edge T+5) both see the same group.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (fi = 0; fi < 5; fi++) begin
                group_d[fi] <= 4'd0;
                prod_v_d[fi] <= 1'b0;
                prod_or_d[fi] <= 13'd0;
                prod_oc_d[fi] <= 13'd0;
                prod_eol_d[fi] <= 1'b0;
                prod_last_d[fi] <= 1'b0;
            end
        end else if (!stall) begin
            group_d[0] <= group;
            prod_v_d[0] <= (state == ST_STREAM) && win_valid_r && pipe_adv;
            prod_or_d[0] <= win_or_r;
            prod_oc_d[0] <= win_oc_r;
            prod_eol_d[0] <= win_eol_r;
            prod_last_d[0] <= win_last_r;
            for (fi = 1; fi < 5; fi++) begin
                group_d[fi] <= group_d[fi-1];
                prod_v_d[fi] <= prod_v_d[fi-1];
                prod_or_d[fi] <= prod_or_d[fi-1];
                prod_oc_d[fi] <= prod_oc_d[fi-1];
                prod_eol_d[fi] <= prod_eol_d[fi-1];
                prod_last_d[fi] <= prod_last_d[fi-1];
            end
        end
    end

    genvar u;
    generate
        for (u = 0; u < P_OUT; u++) begin : unit
            logic signed [24:0] prod [0:NK-1];
            logic signed [24:0] prod_comb [0:NK-1];
            // Flat pack for the adder tree: Yosys 0.52 rejects unpacked
            // array ports, so the tree takes in_flat (element i in
            // bits [i*25 +: 25]); the values are unchanged.
            logic signed [NK*25-1:0] prod_flat;
            genvar pk;
            for (pk = 0; pk < NK; pk++) begin : pack
                assign prod_flat[pk*25 +: 25] = prod[pk];
            end
            int li, lj, lk, wr, wc;
            always @* begin
                for (li = 0; li < MAX_K; li++) begin
                    for (lj = 0; lj < MAX_K; lj++) begin
                        lk = li*MAX_K + lj;
                        wr = MAX_K - ksize + li;
                        wc = MAX_K - ksize + lj;
                        if ((li < ksize) && (lj < ksize))
                            if (sign_in_r)
                                prod_comb[lk] = $signed(win[wr][wc]) *
                                    coeff[group*P_OUT+u][lk];
                            else
                                prod_comb[lk] = $signed({1'b0, win[wr][wc]}) *
                                    coeff[group*P_OUT+u][lk];
                        else
                            prod_comb[lk] = 25'sd0;
                    end
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (lk = 0; lk < NK; lk++) prod[lk] <= 25'sd0;
                end else if (!stall) begin
                    for (lk = 0; lk < NK; lk++) prod[lk] <= prod_comb[lk];
                end
            end
            adder_tree #(.N(NK), .W(25)) u_tree (
                .clk(clk), .rst_n(rst_n), .stall(stall),
                .in_flat(prod_flat), .out(tree_out[u]));
        end
    endgenerate

    //------------------------- emit (bias + shift + act + saturate) -------
    logic signed [32:0] emit_val [0:P_OUT-1];
    logic signed [32:0] emit_rnd [0:P_OUT-1];
    logic signed [32:0] emit_sh [0:P_OUT-1];
    logic signed [32:0] emit_act [0:P_OUT-1];
    logic [7:0]  emit_px [0:P_OUT-1];
    genvar e;
    generate
        for (e = 0; e < P_OUT; e++) begin : emit_unit
            // Leaky slope product: 33-bit x 16-bit = 49-bit; the Q16
            // result is bits [48:16].
            logic signed [48:0] leaky_prod;
            logic signed [32:0] leaky_neg;
            assign emit_val[e] = tree_out[e] +
                $signed({{1{bias_hi[group_d[4]*P_OUT+e][15]}},
                         bias_hi[group_d[4]*P_OUT+e],
                         bias_lo[group_d[4]*P_OUT+e]});
            always @* begin
                emit_rnd[e] = emit_val[e] + ((shift_r == 0) ? 33'sd0 :
                    (33'sd1 <<< (shift_r - 1'b1)));
            end
            assign emit_sh[e] = emit_rnd[e] >>> shift_r;
            // The slope is an UNSIGNED Q0.16 fraction (0x8000 = 0.5):
            // zero-extend to 17 bits so 32768 stays +32768. Sign-
            // extending would turn 0.5 into -32768 and flip the sign
            // of the Leaky output.
            assign leaky_prod = emit_sh[e] * $signed({1'b0, act_slope_r});
            assign leaky_neg = leaky_prod[48:16];
            always @* begin
                if (fuse_act_r == 2'd1)
                    emit_act[e] = (emit_sh[e] < 33'sd0) ? 33'sd0 : emit_sh[e];
                else if (fuse_act_r == 2'd2)
                    emit_act[e] = (emit_sh[e] < 33'sd0) ? leaky_neg : emit_sh[e];
                else
                    emit_act[e] = emit_sh[e];
            end
            always @* begin
                if (sign_out_r) begin
                    if (emit_act[e] > 33'sd127)       emit_px[e] = 8'd127;
                    else if (emit_act[e] < -33'sd128) emit_px[e] = 8'h80;
                    else                              emit_px[e] = emit_act[e][7:0];
                end else begin
                    if (emit_act[e] > 33'sd255)      emit_px[e] = 8'd255;
                    else if (emit_act[e] < 33'sd0)   emit_px[e] = 8'd0;
                    else                              emit_px[e] = emit_act[e][7:0];
                end
            end
        end
    endgenerate

    int ok;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= '0;
            out_valid <= 1'b0;
            out_group <= 4'd0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= prod_v_d[4];
            for (ok = 0; ok < P_OUT; ok++)
                pixel_out[ok*8 +: 8] <= emit_px[ok];
            out_group <= group_d[4];
            out_sof <= prod_v_d[4] && (prod_or_d[4] == 13'd0) &&
                       (prod_oc_d[4] == 13'd0) && (group_d[4] == 4'd0);
            out_eol <= prod_v_d[4] && prod_eol_d[4];
            out_last <= prod_v_d[4] && prod_last_d[4] &&
                        (group_d[4] == (ngroup - 1));
        end
    end
endmodule : conv2d_outpar
