//=============================================================================
// conv2d_mc.sv -- Stage 3: multi-channel dense convolution with channel
// tiling, SystemVerilog.
//
// Architecture (tile-major, input-tile reuse):
//   * The host streams C_in frames (channel-major, raster order, W*H each)
//     once into the input tile buffer (C_PAR BRAM banks; bank = channel %
//     C_PAR). The tile is buffered on-chip and reused for every output
//     channel, so the input is NOT re-streamed C_out times.
//   * For each output channel o, the input channels are processed in
//     tiles of C_PAR parallel lanes. Each lane is a Stage-1 dense datapath
//     (10 line buffers + 11x11 window + 121-mult MAC + adder tree).
//     The C_PAR partial sums are added (cross-lane sum) and accumulated
//     into a per-tile 32-bit accumulator. After all C_in tiles, the
//     accumulator holds the full sum.
//   * C_out is serial: the host rewrites the C_PAR x 121 lane coefficients
//     via the config bus for each (o, t) pass (serial kernel swap), then
//     writes GO to start that pass. The DUT tracks (o, t) internally.
//   * Output is emitted during the LAST tile's compute: when a tile
//     partial is written to the accumulator on the last tile, the final
//     sum is captured with its output position and emitted (shift +
//     saturate) one cycle later. There is no separate emit state.
//   * K=1 mode is the 1x1 pointwise convolution; K=1 with W=H=1 is the FC
//     path (input vector as a 1x1 feature map).
//   * One tile per invocation: the tile is at most TILE_W x TILE_H pixels
//     (default 32x32). Larger images are processed tile-by-tile by the
//     host (Stage 5 tile scheduler). W/H in the config are the TILE dims.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//
// Config bus (write between passes; single-cycle writes; 16-bit address):
//   addr 0x00 : kernel size (1/3/5/7/9/11)
//   addr 0x01 : output shift (0..15)
//   addr 0x02 : tile width  (1..TILE_W)    addr 0x03 : tile height (1..TILE_H)
//   addr 0x04 : pad_top    (0..K)          addr 0x05 : pad_bottom (0..K)
//   addr 0x06 : pad_left   (0..K)          addr 0x07 : pad_right  (0..K)
//   addr 0x08 : stride_x   (1..K)          addr 0x09 : stride_y   (1..K)
//   addr 0x0A : C_in (1..C_IN_MAX)         addr 0x0B : C_out (1..C_OUT_MAX)
//   addr 0x0F : sign mode, bit0 = signed input, bit1 = signed output.
//     Signed int8 intermediates (e.g. a conv layer with no activation,
//     such as RetinexNet Decom-Net conv0, whose output is ~50% negative):
//     the input pixel is sign-extended instead of zero-extended, and the
//     output saturates to [-128, 127] instead of [0, 255]. Zero padding
//     stays correct (a padded 0 is 0.0 in signed too). A uint8+offset
//     representation is NOT supported: it is incompatible with zero
//     padding (Stage 6 weights prep: offset scheme gave conv1 max error
//     215, R PSNR 14.65 dB; signed int8 gives R PSNR 28.95 dB).
//   addr 0x0C : GO (write 1 = start the next (o,t) compute pass)
//   addr 0x0D : output width  (host-computed, for emit framing)
//   addr 0x0E : output height (host-computed, for emit framing)
//   addr 0x100 + l*128 + k : coeff for lane l (0..C_PAR-1), index k
//     (0..120, 11x11 grid, top-left KxK is the kernel). The stride is 128
//     (not 121) so lane/index decode is bit slicing, not a divider.
// Reset defaults: ksize=3, shift=8, W=16, H=16, pads 0, stride 1,
//   C_in=1, C_out=1, out_w=16, out_h=16, 3x3 identity on lane 0
//   (centre tap 256 at index 12).
//=============================================================================
`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// One row delay in BRAM, 8-bit wide. Same template as Stages 1-2.
//-----------------------------------------------------------------------------
module line_buffer_bram_mc #(
    parameter int WIDTH  = 8,
    parameter int DEPTH  = 64,
    parameter int ADDR_W = 6
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

    // Simulation: initialise the memory to 0 so the first-row reads (before
    // the buffer has wrapped once) return 0 instead of X. This matches
    // FPGA BRAM power-up behaviour (memory powers up to 0); synthesis
    // ignores the initial block for a block RAM.
    initial begin
        for (int ii = 0; ii < DEPTH; ii++) mem[ii] = '0;
    end

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
endmodule : line_buffer_bram_mc

//-----------------------------------------------------------------------------
// Main module
//-----------------------------------------------------------------------------
module conv2d_mc #(
    parameter int C_PAR     = 4,
    parameter int C_IN_MAX  = 64,
    parameter int C_OUT_MAX = 32,
    parameter int MAX_K     = 11,
    parameter int TILE_W    = 32,
    parameter int TILE_H    = 32
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (input tile: C_in frames, channel-major, raster order)
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (one output frame per output channel)
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
    localparam logic [15:0] CFG_SIGN    = 16'h000F;  // bit0: signed in, bit1: signed out
    localparam logic [15:0] CFG_COEFF_BASE = 16'h0100;
    localparam logic [15:0] CFG_COEFF_LAST = 16'h02FF;  // 0x100 + 4*128 - 1

    localparam int TILE_PIX = TILE_W * TILE_H;   // 1024
    localparam int NTILE_MAX = C_IN_MAX / C_PAR; // 16
    localparam int TB_DEPTH = NTILE_MAX * TILE_PIX;  // 16384
    localparam int TB_ADDR_W = 14;
    localparam int LB_DEPTH = TILE_W + 32;
    localparam int LB_ADDR_W = (LB_DEPTH <= 64) ? 6 : 7;
    localparam int MAX_COEFF = 121;

    //------------------------- config registers ----------------------------
    logic [3:0]  ksize;
    logic [3:0]  shift_r;
    logic [11:0] width_r, height_r;
    logic [3:0]  pad_t, pad_b, pad_l, pad_r;
    logic [3:0]  stride_x, stride_y;
    logic [6:0]  cin_r, cout_r;
    logic [12:0] out_w_r, out_h_r;
    logic        sign_in_r, sign_out_r;
    logic signed [15:0] coeff [0:C_PAR-1][0:MAX_COEFF-1];
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
            for (ci = 0; ci < C_PAR; ci++)
                for (cj = 0; cj < MAX_COEFF; cj++) coeff[ci][cj] <= '0;
            coeff[0][1*11+1] <= 16'sd256;
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
                default: begin
                    if (cfg_addr >= CFG_COEFF_BASE && cfg_addr <= CFG_COEFF_LAST) begin
                        // Stride 128: offset = addr - 0x100 = l*128 + k, so
                        // lane = offset[9:7], index = offset[6:0].
                        // (index 121..127 are unused.)
                        ci = (cfg_addr - CFG_COEFF_BASE) >> 7;
                        cj = (cfg_addr - CFG_COEFF_BASE) & 16'h007F;
                        if (cj < MAX_COEFF)
                            coeff[ci][cj] <= cfg_data;
                    end
                end
            endcase
        end
    end

    //------------------------- derived geometry ---------------------------
    logic [12:0] pw, ph;
    logic [12:0] frame_px;
    logic [6:0]  ntiles;
    assign pw = {1'b0, width_r} + {9'd0, pad_l} + {9'd0, pad_r};
    assign ph = {1'b0, height_r} + {9'd0, pad_t} + {9'd0, pad_b};
    assign frame_px = {1'b0, width_r} * {1'b0, height_r};
    // ntiles = ceil(C_in / C_PAR); C_PAR=4 is a power of two, so this is
    // a shift, not a divider.
    assign ntiles = (cin_r + 7'd3) >> 2;

    //------------------------- state machine -------------------------------
    typedef enum logic [1:0] {
        ST_IDLE, ST_LOAD, ST_WAIT_GO, ST_COMPUTE
    } state_t;
    state_t state;
    logic [6:0] load_ch;
    logic [12:0] load_idx;
    logic [6:0] cur_o;
    logic [6:0] cur_t;
    logic stall;
    assign stall = out_valid && !out_ready;
    // busy is high only while loading or computing; it is low in IDLE and
    // in WAIT_GO, so the host can poll it to know a compute pass finished.
    assign busy = (state == ST_LOAD) || (state == ST_COMPUTE);

    //------------------------- input tile buffer ---------------------------
    // C_PAR banks; bank l holds channels l, l+C_PAR, l+2*C_PAR, ...
    // Write (LOAD): bank = load_ch % C_PAR, addr = (load_ch/C_PAR)*TILE_PIX
    //   + load_idx. Read (COMPUTE): bank l, addr = cur_t*TILE_PIX + pix_idx.
    logic [7:0] tb_pix [0:C_PAR-1];
    logic [TB_ADDR_W-1:0] tb_raddr [0:C_PAR-1];
    logic [TB_ADDR_W-1:0] tb_waddr;
    logic [1:0] tb_wbank;
    logic       tb_we;
    genvar gi;
    generate
        for (gi = 0; gi < C_PAR; gi++) begin : tb_bank
            (* ram_style = "block" *) logic [7:0] tb_mem [0:TB_DEPTH-1];
            // Simulation: initialise to 0 (matches FPGA BRAM power-up).
            // Banks for inactive lanes are never written during LOAD, so
            // without this their read output would be X.
            initial begin
                for (int ii = 0; ii < TB_DEPTH; ii++) tb_mem[ii] = 8'd0;
            end
            always_ff @(posedge clk) begin
                if (tb_we && (tb_wbank == gi[1:0]))
                    tb_mem[tb_waddr] <= pixel_in;
                // The BRAM read must freeze during a stall, like every
                // other pipeline register: tb_raddr is frozen while
                // stalled, but an ungated read still advances tb_pix to
                // the next pixel (the address was already presented),
                // while pos_real_d1 (the valid flag) stays frozen --
                // so lane_pix would deliver the NEXT pixel and the
                // current one would be lost (one-pixel shift).
                if (!stall)
                    tb_pix[gi] <= tb_mem[tb_raddr[gi]];
            end
        end
    endgenerate

    assign tb_we = (state == ST_LOAD) && in_valid && in_ready;
    assign tb_wbank = load_ch[1:0];
    // (load_ch / C_PAR) is load_ch[5:2] (4 bits); load_idx is 10 bits.
    assign tb_waddr = {load_ch[5:2], load_idx[9:0]};
    assign in_ready = (state == ST_LOAD) && !stall;

    //------------------------- compute stream control ----------------------
    // Walks the PADDED image for the current (o,t). fetch_en advances the
    // stream and presents the tile-buffer read address; the BRAM output is
    // valid one cycle later, so the line-buffer enable chain is shifted by
    // one (en_d[1] feeds the first line buffer, window loads on en_d[11]).
    // stream_done stops the walk after the last padded position; the
    // flush counter then waits for the pipeline to drain.
    logic        fetch_en;
    logic [12:0] strm_r, strm_c;
    logic [12:0] next_c, next_r;
    logic        is_last_stream;
    logic        pos_is_real;
    logic [12:0] pix_idx;
    logic        stream_done;
    logic [4:0]  flush_cnt;

    assign next_c = (strm_c == pw - 13'd1) ? 13'd0 : strm_c + 13'd1;
    assign next_r = (strm_c == pw - 13'd1) ? strm_r + 13'd1 : strm_r;
    assign is_last_stream = (strm_r == ph - 13'd1) && (strm_c == pw - 13'd1);
    assign pos_is_real = (strm_r >= {9'd0, pad_t}) &&
                         (strm_r <  {9'd0, pad_t} + {1'b0, height_r}) &&
                         (strm_c >= {9'd0, pad_l}) &&
                         (strm_c <  {9'd0, pad_l} + {1'b0, width_r});
    assign fetch_en = (state == ST_COMPUTE) && !stall && !stream_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            strm_r <= 13'd0;
            strm_c <= 13'd0;
            pix_idx <= 13'd0;
            stream_done <= 1'b0;
            flush_cnt <= 5'd0;
        end else if (state == ST_COMPUTE) begin
            if (!stall) begin
                if (fetch_en) begin
                    if (is_last_stream) begin
                        strm_r <= 13'd0;
                        strm_c <= 13'd0;
                        stream_done <= 1'b1;
                        flush_cnt <= 5'd0;
                    end else begin
                        strm_r <= next_r;
                        strm_c <= next_c;
                        if (pos_is_real) pix_idx <= pix_idx + 13'd1;
                    end
                end else if (stream_done) begin
                    flush_cnt <= flush_cnt + 5'd1;
                end
            end
        end else begin
            strm_r <= 13'd0;
            strm_c <= 13'd0;
            pix_idx <= 13'd0;
            stream_done <= 1'b0;
            flush_cnt <= 5'd0;
        end
    end

    // tile-buffer read address per lane (presented on fetch_en).
    // cur_t is 0..15 (4 bits); pix_idx is 10 bits.
    int rl;
    always @* begin
        for (rl = 0; rl < C_PAR; rl++) begin
            tb_raddr[rl] = {cur_t[3:0], pix_idx[9:0]};
        end
    end

    //------------------------- enable shift register -----------------------
    logic [11:0] en_shreg;
    logic [11:0] en_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) en_shreg <= 12'd0;
        else if (!stall) en_shreg <= {en_shreg[10:0], fetch_en};
    end
    assign en_d = {en_shreg[10:0], fetch_en};

    logic [12:0] cr_d [0:10];
    logic [12:0] cc_d [0:10];
    int di;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (di = 0; di < 11; di++) begin
                cr_d[di] <= 13'd0; cc_d[di] <= 13'd0;
            end
        end else if (!stall) begin
            cr_d[0] <= fetch_en ? strm_r : 13'd0;
            cc_d[0] <= fetch_en ? strm_c : 13'd0;
            for (di = 1; di < 11; di++) begin
                cr_d[di] <= cr_d[di-1];
                cc_d[di] <= cc_d[di-1];
            end
        end
    end
    logic [12:0] mac_r, mac_c;
    assign mac_r = cr_d[10];
    assign mac_c = cc_d[10];

    //------------------------- next-output (decimation) counters ----------
    // Same "next exceeds" method as Stage 1 (no divider). Reset in
    // ST_WAIT_GO (before each compute pass).
    logic [12:0] nxt_or, nxt_oc, nxt_sr, nxt_sc;
    logic        match_out;
    logic is_eol_c, is_last_c;
    assign match_out = (mac_r == nxt_sr) && (mac_c == nxt_sc);
    assign is_eol_c  = match_out && ((nxt_sc + {9'd0, stride_x}) > (pw - 13'd1));
    assign is_last_c = is_eol_c  && ((nxt_sr + {9'd0, stride_y}) > (ph - 13'd1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= 13'd10; nxt_sc <= 13'd10;
        end else if (state == ST_WAIT_GO) begin
            nxt_or <= 13'd0; nxt_oc <= 13'd0;
            nxt_sr <= {9'd0, ksize} - 13'd1;
            nxt_sc <= {9'd0, ksize} - 13'd1;
        end else if (!stall && en_d[11] && match_out) begin
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

    //------------------------- per-lane pixel source -----------------------
    // The BRAM output is valid one cycle after fetch_en; gate it with
    // pos_is_real delayed by one cycle.
    logic pos_real_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pos_real_d1 <= 1'b0;
        else if (!stall) pos_real_d1 <= fetch_en && pos_is_real;
    end
    logic [7:0] lane_pix [0:C_PAR-1];
    genvar pl;
    generate
        for (pl = 0; pl < C_PAR; pl++) begin : lane_pix_assign
            assign lane_pix[pl] = pos_real_d1 ? tb_pix[pl] : 8'd0;
        end
    endgenerate

    //------------------------- shared pipeline flags -----------------------
    // Flags travel with the data through every pipeline stage:
    // win -> prod -> sum2 -> sum4 -> sum6 -> acc -> lane_partial -> tile_partial.
    // (The acc/lp stages were missing, so tp_* used s6_* two stages early.)
    logic win_v, prod_v, sum2_v, sum4_v, sum6_v, acc_v, lp_v;
    logic        win_is_out, p_is_out, s2_is_out, s4_is_out, s6_is_out,
                 acc_is_out, lp_is_out;
    logic [12:0] win_or, win_oc, p_or, p_oc, s2_or, s2_oc,
                 s4_or, s4_oc, s6_or, s6_oc, acc_or, acc_oc, lp_or, lp_oc;
    logic        win_eol, win_last, p_eol, p_last, s2_eol, s2_last,
                 s4_eol, s4_last, s6_eol, s6_last, acc_eol, acc_last,
                 lp_eol, lp_last;
    logic signed [31:0] lane_partial [0:C_PAR-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win_v <= 1'b0; win_is_out <= 1'b0;
            win_or <= 13'd0; win_oc <= 13'd0;
            win_eol <= 1'b0; win_last <= 1'b0;
        end else if (!stall) begin
            win_v <= en_d[11];
            win_is_out <= en_d[11] && match_out;
            win_or <= nxt_or;
            win_oc <= nxt_oc;
            win_eol <= en_d[11] && is_eol_c;
            win_last <= en_d[11] && is_last_c;
        end
    end

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
            acc_v  <= 1'b0; acc_is_out <= 1'b0; acc_or <= 13'd0; acc_oc <= 13'd0;
            acc_eol <= 1'b0; acc_last <= 1'b0;
            lp_v   <= 1'b0; lp_is_out <= 1'b0; lp_or <= 13'd0; lp_oc <= 13'd0;
            lp_eol <= 1'b0; lp_last <= 1'b0;
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
            acc_v  <= sum6_v; acc_is_out <= s6_is_out;
            acc_or <= s6_or;  acc_oc <= s6_oc;
            acc_eol <= s6_eol; acc_last <= s6_last;
            lp_v   <= acc_v;  lp_is_out <= acc_is_out;
            lp_or  <= acc_or; lp_oc <= acc_oc;
            lp_eol <= acc_eol; lp_last <= acc_last;
        end
    end

    //------------------------- per-lane window + MAC -----------------------
    genvar ln;
    generate
        for (ln = 0; ln < C_PAR; ln++) begin : lane
            logic [7:0] lb_out [0:9];
            logic [7:0] lb_in  [0:9];
            assign lb_in[0] = lane_pix[ln];
            genvar gi2;
            for (gi2 = 0; gi2 < 10; gi2++) begin : lb_chain
                if (gi2 > 0) assign lb_in[gi2] = lb_out[gi2-1];
                line_buffer_bram_mc #(.WIDTH(8), .DEPTH(LB_DEPTH),
                                      .ADDR_W(LB_ADDR_W)) u_lb (
                    .clk(clk), .rst_n(rst_n),
                    .en(en_d[gi2+1] && !stall),
                    .din(lb_in[gi2]),
                    .wrap(pw[LB_ADDR_W-1:0]),
                    .dout(lb_out[gi2])
                );
            end

            logic [7:0] row_in [0:10];
            assign row_in[10] = lane_pix[ln];
            genvar gi3;
            for (gi3 = 0; gi3 < 10; gi3++) begin : row_assign
                assign row_in[gi3] = lb_out[9-gi3];
            end
            logic [7:0] aln [0:10][0:9];
            logic [7:0] acol [0:10];
            int ai, aj;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (ai = 0; ai < 11; ai++)
                        for (aj = 0; aj < 10; aj++) aln[ai][aj] <= 8'd0;
                end else if (!stall) begin
                    for (ai = 0; ai < 11; ai++) begin
                        aln[ai][0] <= row_in[ai];
                        for (aj = 1; aj < 10; aj++) aln[ai][aj] <= aln[ai][aj-1];
                    end
                end
            end
            genvar gi4;
            for (gi4 = 0; gi4 < 11; gi4++) begin : acol_assign
                if (gi4 == 0) assign acol[gi4] = row_in[gi4];
                else           assign acol[gi4] = aln[gi4][gi4-1];
            end

            logic [7:0] win [0:10][0:10];
            int wi, wj;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (wi = 0; wi < 11; wi++)
                        for (wj = 0; wj < 11; wj++) win[wi][wj] <= 8'd0;
                end else if (!stall && en_d[11]) begin
                    for (wi = 0; wi < 11; wi++) begin
                        for (wj = 0; wj < 10; wj++) win[wi][wj] <= win[wi][wj+1];
                        win[wi][10] <= acol[wi];
                    end
                end
            end

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
            int li, lj, lk, wr, wc, qi, si;

            always @* begin
                for (li = 0; li < 11; li++) begin
                    for (lj = 0; lj < 11; lj++) begin
                        lk = li*11 + lj;
                        wr = 11 - ksize + li;
                        wc = 11 - ksize + lj;
                        if ((li < ksize) && (lj < ksize))
                            if (sign_in_r)
                                prod_comb[lk] = $signed(win[wr][wc]) * coeff[ln][lk];
                            else
                                prod_comb[lk] = $signed({1'b0, win[wr][wc]}) * coeff[ln][lk];
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
            always @* begin
                for (si = 0; si < 60; si++) s1_comb[si] = prod[2*si] + prod[2*si+1];
                s1_comb[60] = prod[120];
                for (si = 0; si < 30; si++) sum2_comb[si] = s1_comb[2*si] + s1_comb[2*si+1];
                sum2_comb[30] = s1_comb[60];
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 31; si++) sum2[si] <= 27'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 31; si++) sum2[si] <= sum2_comb[si];
                end
            end
            always @* begin
                for (si = 0; si < 15; si++) s3_comb[si] = sum2[2*si] + sum2[2*si+1];
                s3_comb[15] = sum2[30];
                for (si = 0; si < 8; si++) sum4_comb[si] = s3_comb[2*si] + s3_comb[2*si+1];
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 8; si++) sum4[si] <= 29'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 8; si++) sum4[si] <= sum4_comb[si];
                end
            end
            always @* begin
                for (si = 0; si < 4; si++) s5_comb[si] = sum4[2*si] + sum4[2*si+1];
                for (si = 0; si < 2; si++) sum6_comb[si] = s5_comb[2*si] + s5_comb[2*si+1];
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (si = 0; si < 2; si++) sum6[si] <= 31'sd0;
                end else if (!stall) begin
                    for (si = 0; si < 2; si++) sum6[si] <= sum6_comb[si];
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) acc <= 32'sd0;
                else if (!stall) acc <= sum6[0] + sum6[1];
            end
            // Lane partial (registered). Inactive lanes (global channel
            // index >= C_in) contribute 0.
            logic lane_active;
            assign lane_active = ((cur_t * C_PAR + ln) < cin_r);
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) lane_partial[ln] <= 32'sd0;
                else if (!stall) begin
                    lane_partial[ln] <= lane_active ? acc : 32'sd0;
                end
            end
        end
    endgenerate

    //------------------------- cross-lane sum + accumulator ----------------
    // Tile partial = sum of the C_PAR lane partials (one register stage).
    // tp_* ride with it. acc_idx counts output positions in raster order
    // (reset in ST_WAIT_GO, before each compute pass).
    logic signed [31:0] tile_partial_comb;
    logic signed [31:0] tile_partial;
    logic        tp_v;
    logic        tp_is_out;
    logic [12:0] tp_or, tp_oc;
    logic        tp_eol, tp_last;
    assign tile_partial_comb = lane_partial[0] + lane_partial[1] +
                               lane_partial[2] + lane_partial[3];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_partial <= 32'sd0;
            tp_v <= 1'b0;
            tp_is_out <= 1'b0;
            tp_or <= 13'd0; tp_oc <= 13'd0;
            tp_eol <= 1'b0; tp_last <= 1'b0;
        end else if (!stall) begin
            tile_partial <= tile_partial_comb;
            tp_v <= lp_v;
            tp_is_out <= lp_v && lp_is_out;
            tp_or <= lp_or;
            tp_oc <= lp_oc;
            tp_eol <= lp_v && lp_eol;
            tp_last <= lp_v && lp_last;
        end
    end

    logic [12:0] acc_idx;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_idx <= 13'd0;
        else if (state == ST_WAIT_GO) acc_idx <= 13'd0;
        else if (!stall && tp_v && tp_is_out) acc_idx <= acc_idx + 13'd1;
    end

    // Accumulator (register array, combinational read). First tile of an
    // output channel overwrites (stale values ignored); later tiles add.
    logic signed [31:0] acc_mem [0:TILE_PIX-1];
    logic signed [31:0] acc_rd;
    logic signed [31:0] acc_wr_val;
    assign acc_rd = acc_mem[acc_idx];
    assign acc_wr_val = (cur_t == 7'd0) ? tile_partial : acc_rd + tile_partial;
    int mi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (mi = 0; mi < TILE_PIX; mi++) acc_mem[mi] <= 32'sd0;
        end else if (!stall && tp_v && tp_is_out) begin
            acc_mem[acc_idx] <= acc_wr_val;
        end
    end

    //------------------------- emit (last tile only) -----------------------
    // When a tile partial is written on the LAST tile, the final sum is
    // captured with its output position and emitted (shift + saturate)
    // one cycle later. acc_wr_val is the final sum for this pixel.
    logic signed [31:0] emit_val;
    logic [12:0] emit_or, emit_oc;
    logic        emit_eol, emit_last;
    logic        emit_v;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            emit_val <= 32'sd0;
            emit_or <= 13'd0; emit_oc <= 13'd0;
            emit_eol <= 1'b0; emit_last <= 1'b0;
            emit_v <= 1'b0;
        end else if (!stall) begin
            emit_v <= tp_v && tp_is_out && (cur_t == ntiles - 7'd1);
            if (tp_v && tp_is_out && (cur_t == ntiles - 7'd1)) begin
                emit_val <= acc_wr_val;
                emit_or <= tp_or;
                emit_oc <= tp_oc;
                emit_eol <= tp_eol;
                emit_last <= tp_last;
            end
        end
    end

    logic signed [31:0] emit_rnd;
    logic signed [31:0] emit_sh;
    logic [7:0]  emit_px;
    always @* begin
        emit_rnd = emit_val + ((shift_r == 0) ? 32'sd0 :
                               (32'sd1 <<< (shift_r - 1'b1)));
    end
    assign emit_sh = emit_rnd >>> shift_r;
    always @* begin
        if (sign_out_r) begin
            if (emit_sh > 32'sd127)       emit_px = 8'd127;
            else if (emit_sh < -32'sd128) emit_px = 8'h80;  // -128
            else                           emit_px = emit_sh[7:0];
        end else begin
            if (emit_sh > 32'sd255)      emit_px = 8'd255;
            else if (emit_sh < 32'sd0)   emit_px = 8'd0;
            else                          emit_px = emit_sh[7:0];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= emit_v;
            pixel_out <= emit_px;
            out_sof <= emit_v && (emit_or == 13'd0) && (emit_oc == 13'd0);
            out_eol <= emit_v && emit_eol;
            out_last <= emit_v && emit_last;
        end
    end

    //------------------------- main state machine --------------------------
    // Flush: after the last fetch, wait 24 cycles for the pipeline to
    // drain (fetch->window 11 + MAC 6 + cross-lane 1 = 18, plus margin),
    // then move on. The last outputs are emitted during the flush.
    localparam int FLUSH_CYCLES = 24;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            load_ch <= 7'd0;
            load_idx <= 13'd0;
            cur_o <= 7'd0;
            cur_t <= 7'd0;
        end else if (!stall) begin
            case (state)
                ST_IDLE: begin
                    if (in_valid && in_sof) begin
                        state <= ST_LOAD;
                        load_ch <= 7'd0;
                        load_idx <= 13'd0;
                    end
                end
                ST_LOAD: begin
                    if (in_valid) begin
                        if (load_idx == frame_px - 13'd1) begin
                            load_idx <= 13'd0;
                            if (load_ch == cin_r - 7'd1) begin
                                state <= ST_WAIT_GO;
                                cur_o <= 7'd0;
                                cur_t <= 7'd0;
                            end else begin
                                load_ch <= load_ch + 7'd1;
                            end
                        end else begin
                            load_idx <= load_idx + 13'd1;
                        end
                    end
                end
                ST_WAIT_GO: begin
                    if (go_pulse) begin
                        state <= ST_COMPUTE;
                    end
                end
                ST_COMPUTE: begin
                    if (stream_done && (flush_cnt >= FLUSH_CYCLES[4:0])) begin
                        if (cur_t == ntiles - 7'd1) begin
                            if (cur_o == cout_r - 7'd1) begin
                                state <= ST_IDLE;
                            end else begin
                                cur_o <= cur_o + 7'd1;
                                cur_t <= 7'd0;
                                state <= ST_WAIT_GO;
                            end
                        end else begin
                            cur_t <= cur_t + 7'd1;
                            state <= ST_WAIT_GO;
                        end
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule : conv2d_mc
