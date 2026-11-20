//=============================================================================
// layernorm.sv -- Stage 4: LayerNorm (frame mean/variance, normalize +
// scale/shift), SystemVerilog.
//
// Normalizes over one frame (W x H pixels = one "layer" of activations):
//   mean = sum(x) / N,  var = sum(x^2)/N - mean^2,  N = W*H
//   y = (x - mean) * inv_std * gamma + beta,  inv_std = 1/sqrt(var + eps)
// W and H must be powers of two, so N is a power of two and the divisions
// are arithmetic right shifts by log2(N) (no divider is synthesized).
//   * Pass 1: the frame is streamed in, buffered on-chip (up to
//     MAX_PIXELS entries), and sum / sum-of-squares are accumulated.
//   * Compute: mean = sum >> log2(N); var = (sumsq >> log2(N)) -
//     mean*mean, clamped to >= 0; v = var + eps; inv_std is looked up
//     from a 256-entry reciprocal-square-root LUT (Q1.15), indexed by
//     (v >> isqrt_shift), clamped to [0,255]. The LUT is loaded via the
//     config bus (16-bit address, 0x100+i); the compiler owns its
//     contents and the shift (it picks the shift so the index spans the
//     expected var range). The LUT is the approximation: inv_std is
//     quantized to Q1.15 and to the LUT bin width (2^isqrt_shift in v).
//     The golden reports the error vs the true float LayerNorm.
//   * Pass 2: the buffered frame is streamed out, one pixel per cycle:
//       diff = x - mean (signed)
//       norm_q = (diff * inv_std_q) >> 7        (Q8.8)
//       y = (norm_q * gamma_q) >> 23 + beta
//     gamma_q is Q1.15 (config), beta is a signed integer (config).
//     The result saturates to [0,255] (unsigned out) or [-128,127]
//     (signed out, two's-complement byte). Input pixels are 8-bit
//     (unsigned, or signed int8 if sign_in is set).
//   * The input frame is W*H pixels; the output frame is W*H pixels.
//     A single stall signal (out_valid && !out_ready) freezes all state.
//     There is no terminal DONE state: after a frame completes, the FSM
//     returns to ST_LOAD (idle) so the next frame can start without a
//     reset.
//
// Config bus (write between frames; single-cycle writes; 16-bit address):
//   addr 0x01 : image width  (power of 2, 1..4095)
//   addr 0x02 : image height (power of 2, 1..4095)
//   addr 0x03 : sign mode, bit0 = signed input, bit1 = signed output
//   addr 0x04 : isqrt_shift (0..8, LUT index = (var+eps) >> isqrt_shift)
//   addr 0x05 : eps (0..255, added to var before the sqrt)
//   addr 0x06 : gamma_q (16-bit Q1.15, default 32767 = ~1.0)
//   addr 0x07 : beta (signed 16-bit integer, default 0)
//   addr 0x100 + i : inv_std LUT entry i (16-bit Q1.15), i = 0..255
// Reset defaults: W=16, H=16, sign=0, isqrt_shift=6, eps=1,
//   gamma=32767, beta=0, LUT=0.
//=============================================================================
`timescale 1ns/1ps

module layernorm #(
    parameter int MAX_PIXELS = 4096
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [15:0] CFG_WIDTH = 16'h0001;
    localparam logic [15:0] CFG_HEIGHT = 16'h0002;
    localparam logic [15:0] CFG_SIGN  = 16'h0003;
    localparam logic [15:0] CFG_ISQRT_SHIFT = 16'h0004;
    localparam logic [15:0] CFG_EPS   = 16'h0005;
    localparam logic [15:0] CFG_GAMMA = 16'h0006;
    localparam logic [15:0] CFG_BETA  = 16'h0007;
    localparam logic [15:0] CFG_LUT_BASE = 16'h0100;
    localparam logic [15:0] CFG_LUT_LAST = 16'h01FF;

    //------------------------- config registers ----------------------------
    logic [11:0] width_r, height_r;
    logic        sign_in_r, sign_out_r;
    logic [3:0]  isqrt_shift_r;
    logic [7:0]  eps_r;
    logic signed [15:0] gamma_r;
    logic signed [15:0] beta_r;
    logic [15:0] isqrt_lut [0:255];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            width_r <= 12'd16;
            height_r <= 12'd16;
            sign_in_r <= 1'b0;
            sign_out_r <= 1'b0;
            isqrt_shift_r <= 4'd6;
            eps_r <= 8'd1;
            gamma_r <= 16'sd32767;
            beta_r <= 16'sd0;
            for (int i = 0; i < 256; i++) isqrt_lut[i] <= 16'd0;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_WIDTH:  width_r <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                CFG_SIGN: begin
                    sign_in_r <= cfg_data[0];
                    sign_out_r <= cfg_data[1];
                end
                CFG_ISQRT_SHIFT: isqrt_shift_r <= cfg_data[3:0];
                CFG_EPS:    eps_r <= cfg_data[7:0];
                CFG_GAMMA:  gamma_r <= cfg_data;
                CFG_BETA:   beta_r <= cfg_data;
                default: begin
                    if (cfg_addr >= CFG_LUT_BASE && cfg_addr <= CFG_LUT_LAST)
                        isqrt_lut[cfg_addr[7:0]] <= cfg_data;
                end
            endcase
        end
    end

    //------------------------- frame buffer --------------------------------
    logic [7:0] frame_buf [0:MAX_PIXELS-1];
    logic [24:0] frame_px;
    assign frame_px = {1'b0, width_r} * {1'b0, height_r};

    // log2(N) where N = W*H (both powers of two): the index of the
    // highest set bit of W plus that of H. Computed combinationally.
    function automatic [4:0] log2_pow2(input logic [11:0] v);
        casez (v)
            12'b1???????????: log2_pow2 = 5'd11;
            12'b01??????????: log2_pow2 = 5'd10;
            12'b001?????????: log2_pow2 = 5'd9;
            12'b0001????????: log2_pow2 = 5'd8;
            12'b00001???????: log2_pow2 = 5'd7;
            12'b000001??????: log2_pow2 = 5'd6;
            12'b0000001?????: log2_pow2 = 5'd5;
            12'b00000001????: log2_pow2 = 5'd4;
            12'b000000001???: log2_pow2 = 5'd3;
            12'b0000000001??: log2_pow2 = 5'd2;
            12'b00000000001?: log2_pow2 = 5'd1;
            default:          log2_pow2 = 5'd0;
        endcase
    endfunction
    logic [4:0] log_n;
    assign log_n = log2_pow2(width_r) + log2_pow2(height_r);

    //------------------------- FSM -----------------------------------------
    // ST_LOAD: accept input pixels, buffer them, accumulate sum/sumsq.
    // ST_EMIT: stream the buffered frame out, normalizing each pixel.
    // (The mean/var/inv_std are computed combinationally from the
    // accumulators when ST_LOAD finishes; ST_EMIT uses the latched
    // values.) No terminal DONE state: after a frame, return to ST_LOAD.
    localparam logic [1:0] ST_LOAD = 2'd0;
    localparam logic [1:0] ST_EMIT = 2'd1;
    logic [1:0] state;

    logic [24:0] idx;       // input index in ST_LOAD (0..N-1)
    logic [24:0] out_idx;   // output index in ST_EMIT (0..N-1)
    logic [12:0] out_col;
    logic        stall;

    // Accumulators (signed sum, signed sumsq). x is sign- or
    // zero-extended to 16-bit signed; x^2 is up to 65025, times N=4096
    // is ~2.7e8, fits in signed 32-bit. sum and sumsq are signed
    // 32-bit. (sumsq MUST be signed: an unsigned sumsq_acc forces the
    // px_s*px_s product into an unsigned context, so a negative pixel
    // squares to ~4.3e9 instead of its true square.)
    logic signed [31:0] sum_acc;
    logic signed [31:0] sumsq_acc;
    logic signed [15:0] px_s;  // current input pixel, sign/zero extended
    // (always @*, not a continuous assign: Icarus does not reliably
    // re-evaluate a ternary/$signed continuous assign when pixel_in
    // changes, and px_s then lags pixel_in by one cycle.)
    always @* begin
        if (sign_in_r) px_s = $signed(pixel_in);
        else px_s = $signed({1'b0, pixel_in});
    end

    // Mean / var / inv_std, computed combinationally from the
    // accumulators. The *_new variants include the current input pixel
    // (used when latching the normalize parameters on the last pixel).
    logic signed [31:0] mean_c;
    logic signed [31:0] var_c;
    logic [31:0] v_c;
    logic [7:0]  lut_idx;
    logic [15:0] inv_std_c;
    assign mean_c = sum_acc >>> log_n;
    // var = E[x^2] - mean^2; clamp to >= 0 (floor in the shifts can make
    // it -1 for a constant frame).
    assign var_c = $signed({1'b0, (sumsq_acc >> log_n)}) - mean_c * mean_c;
    assign v_c = (var_c < 0 ? 32'd0 : var_c[31:0]) + {24'd0, eps_r};
    assign lut_idx = (v_c >> isqrt_shift_r) > 32'd255 ? 8'd255 :
                     (v_c >> isqrt_shift_r);
    assign inv_std_c = isqrt_lut[lut_idx];

    logic signed [31:0] sum_new;
    logic signed [31:0] sumsq_new;
    logic signed [31:0] mean_new;
    logic signed [31:0] var_new;
    logic [31:0] v_new;
    logic [7:0]  lut_idx_new;
    assign sum_new = sum_acc + $signed({{16{px_s[15]}}, px_s});
    assign sumsq_new = sumsq_acc + $signed(px_s) * $signed(px_s);
    assign mean_new = sum_new >>> log_n;
    assign var_new = $signed({1'b0, (sumsq_new >> log_n)}) - mean_new * mean_new;
    assign v_new = (var_new < 0 ? 32'd0 : var_new[31:0]) + {24'd0, eps_r};
    assign lut_idx_new = (v_new >> isqrt_shift_r) > 32'd255 ? 8'd255 :
                         (v_new >> isqrt_shift_r);

    // Latched normalize parameters for the emit pass.
    logic signed [31:0] mean_r;
    logic signed [15:0] inv_std_r;
    logic signed [15:0] gamma_r_lat;
    logic signed [15:0] beta_r_lat;

    // Per-pixel normalize, PIPELINED into two registered stages (Stage 4
    // STA: the unpipelined two-multiply chain violates 3.333 ns at
    // 4.301 ns arrival, slack -1.009 -- it would cap the system Fmax
    // at ~232 MHz). Stage A (combinational, registered into norm_q_pipe):
    // diff = x - mean, norm_q = (diff * inv_std) >> 7 (Q8.8). Stage B
    // (combinational from norm_q_pipe, registered into pixel_out):
    // y = (norm_q * gamma) >> 23 + beta, saturate. Each stage has one
    // 32x16 multiply (~2.25 ns, like the Leaky path). Throughput stays
    // 1 pixel/cycle; latency is +1 cycle (the first output emerges one
    // cycle after the first pixel is fed). The framing flags travel
    // with norm_q_pipe so they stay aligned with the pipelined output.
    logic [7:0]  buf_px;
    logic signed [15:0] buf_px_s;
    logic signed [31:0] diff_c;
    logic signed [47:0] prod1_c;
    logic signed [31:0] norm_q_c;
    logic signed [31:0] norm_q_pipe;
    logic signed [47:0] prod2_c;
    logic signed [31:0] y_pipe_c;
    logic [7:0]  px_sat_pipe;
    logic pipe_v, pipe_eol, pipe_last, pipe_sof;
    // (always @*, not continuous assigns: Icarus does not reliably
    // re-evaluate a memory read or a ternary/$signed continuous assign.)
    always @* begin
        buf_px = frame_buf[out_idx];
        if (sign_in_r) buf_px_s = $signed(buf_px);
        else buf_px_s = $signed({1'b0, buf_px});
    end
    assign diff_c = $signed({{16{buf_px_s[15]}}, buf_px_s}) - mean_r;
    assign prod1_c = diff_c * $signed({{16{inv_std_r[15]}}, inv_std_r});
    assign norm_q_c = prod1_c >>> 7;
    assign prod2_c = norm_q_pipe * $signed({{16{gamma_r_lat[15]}}, gamma_r_lat});
    assign y_pipe_c = (prod2_c >>> 23) + $signed({{16{beta_r_lat[15]}}, beta_r_lat});
    always @* begin
        if (sign_out_r) begin
            if (y_pipe_c > 32'sd127)       px_sat_pipe = 8'd127;
            else if (y_pipe_c < -32'sd128) px_sat_pipe = 8'h80;
            else                            px_sat_pipe = y_pipe_c[7:0];
        end else begin
            if (y_pipe_c > 32'sd255)      px_sat_pipe = 8'd255;
            else if (y_pipe_c < 32'sd0)   px_sat_pipe = 8'd0;
            else                           px_sat_pipe = y_pipe_c[7:0];
        end
    end

    assign stall = out_valid && !out_ready;
    assign in_ready = (state == ST_LOAD) && !stall;
    assign busy = (state == ST_EMIT);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_LOAD;
            idx <= 25'd0;
            out_idx <= 25'd0;
            out_col <= 13'd0;
            sum_acc <= 32'sd0;
            sumsq_acc <= 32'd0;
            norm_q_pipe <= 32'sd0;
            pipe_v <= 1'b0; pipe_eol <= 1'b0;
            pipe_last <= 1'b0; pipe_sof <= 1'b0;
            mean_r <= 32'sd0;
            inv_std_r <= 16'sd0;
            gamma_r_lat <= 16'sd0;
            beta_r_lat <= 16'sd0;
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= 1'b0;
            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
            case (state)
                ST_LOAD: begin
                    if (in_valid) begin
                        frame_buf[idx] <= pixel_in;
                        sum_acc <= sum_acc + $signed({{16{px_s[15]}}, px_s});
                        sumsq_acc <= sumsq_acc + $signed(px_s) * $signed(px_s);
                        if (idx >= frame_px - 25'd1) begin
                            idx <= 25'd0;
                            out_idx <= 25'd0;
                            out_col <= 13'd0;
                            // Latch the normalize parameters from the
                            // accumulators INCLUDING this (last) pixel.
                            mean_r <= mean_new;
                            inv_std_r <= isqrt_lut[lut_idx_new];
                            gamma_r_lat <= gamma_r;
                            beta_r_lat <= beta_r;
                            state <= ST_EMIT;
                        end else begin
                            idx <= idx + 25'd1;
                        end
                    end
                end
                ST_EMIT: begin
                    // Pipelined emit: each cycle the output is the
                    // PREVIOUS pixel's Stage-B result (from
                    // norm_q_pipe), while Stage A feeds the current
                    // pixel's norm_q into the pipe. out_valid follows
                    // pipe_v (the pre-edge value). When the feed is
                    // done (out_idx >= frame_px) and the pipe is empty
                    // (pipe_v == 0), the frame is fully drained and
                    // we return to ST_LOAD.
                    pixel_out <= px_sat_pipe;
                    out_valid <= pipe_v;
                    out_sof <= pipe_sof;
                    out_eol <= pipe_eol;
                    out_last <= pipe_last;
                    if (out_idx < frame_px) begin
                        // Feed Stage A with the current pixel.
                        norm_q_pipe <= norm_q_c;
                        pipe_v <= 1'b1;
                        pipe_sof <= (out_idx == 25'd0);
                        pipe_eol <= (out_col == {1'b0, width_r} - 13'd1);
                        pipe_last <= (out_idx == frame_px - 25'd1);
                        out_idx <= out_idx + 25'd1;
                        if (out_col == {1'b0, width_r} - 13'd1)
                            out_col <= 13'd0;
                        else
                            out_col <= out_col + 13'd1;
                    end else begin
                        pipe_v <= 1'b0;
                        if (!pipe_v) begin
                            out_idx <= 25'd0;
                            out_col <= 13'd0;
                            sum_acc <= 32'sd0;
                            sumsq_acc <= 32'sd0;
            norm_q_pipe <= 32'sd0;
            pipe_v <= 1'b0; pipe_eol <= 1'b0;
            pipe_last <= 1'b0; pipe_sof <= 1'b0;
                            state <= ST_LOAD;
                        end
                    end
                end
                default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
