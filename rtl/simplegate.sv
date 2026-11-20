//=============================================================================
// simplegate.sv -- Stage 4: SimpleGate elementwise multiply (NAFNet),
// SystemVerilog.
//
// Input is a single stream of 2*W*H pixels: the first W*H pixels are half
// A (channels 0..C-1), the next W*H pixels are half B (channels C..2C-1).
// Output is W*H pixels: out[i] = (A[i] * B[i]) >> shift, saturated.
//   * The first half is buffered on-chip (a memory array of up to
//     MAX_PIXELS entries; the scheduler tiles the frame so the tile fits).
//     When the second half arrives, each B pixel is multiplied by the
//     buffered A pixel at the same index and emitted.
//   * Multiply is 8-bit x 8-bit = 16-bit (signed if sign_in is set, where
//     the 8-bit bytes are two's-complement int8). The product is shifted
//     right by `shift` (arithmetic shift) and saturated to [0,255]
//     (unsigned out) or [-128,127] (signed out, emitted as a
//     two's-complement byte).
//   * One pixel per cycle in steady state (the second half streams
//     through); the first half is load-only (no output). Latency is one
//     cycle from a B pixel to its output.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//
// Config bus (write between frames; single-cycle writes; 8-bit address):
//   addr 0x00 : shift (0..8)
//   addr 0x01 : image width  (1..4095)
//   addr 0x02 : image height (1..4095)
//   addr 0x03 : sign mode, bit0 = signed input, bit1 = signed output
// Reset defaults: shift=8, W=16, H=16, sign=0.
//=============================================================================
`timescale 1ns/1ps

module simplegate #(
    parameter int MAX_PIXELS = 4096
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (2*W*H pixels: half A, then half B)
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (W*H pixels)
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [7:0] CFG_SHIFT = 8'h00;
    localparam logic [7:0] CFG_WIDTH = 8'h01;
    localparam logic [7:0] CFG_HEIGHT = 8'h02;
    localparam logic [7:0] CFG_SIGN  = 8'h03;

    //------------------------- config registers ----------------------------
    logic [3:0]  shift_r;
    logic [11:0] width_r, height_r;
    logic        sign_in_r, sign_out_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_r <= 4'd8;
            width_r <= 12'd16;
            height_r <= 12'd16;
            sign_in_r <= 1'b0;
            sign_out_r <= 1'b0;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_SHIFT:  shift_r <= cfg_data[3:0];
                CFG_WIDTH:  width_r <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                CFG_SIGN: begin
                    sign_in_r <= cfg_data[0];
                    sign_out_r <= cfg_data[1];
                end
                default: ;
            endcase
        end
    end

    //------------------------- half-A buffer -------------------------------
    // Stores the first W*H pixels. Written during the first half, read
    // during the second half (same index). A simple dual-port-ish array:
    // write on in_valid (first half), read combinationally on the B index.
    logic [7:0] a_buf [0:MAX_PIXELS-1];
    logic [24:0] frame_px;
    assign frame_px = {1'b0, width_r} * {1'b0, height_r};

    //------------------------- stream control ------------------------------
    // idx counts input pixels 0..2*W*H-1. First half (idx < frame_px) is
    // A (buffered); second half is B (multiplied with buffered A and
    // emitted). out_idx counts output pixels 0..W*H-1.
    logic [24:0] idx;
    logic [24:0] out_idx;
    logic [12:0] out_col;
    logic        stall;
    logic        in_first_half;
    logic        can_accept;
    logic        can_emit;

    assign stall = out_valid && !out_ready;
    assign in_first_half = (idx < frame_px);
    assign can_accept = in_valid && !stall;
    assign in_ready = !stall;
    // We can emit when a B pixel is being accepted (combinational multiply
    // of the incoming B pixel with the buffered A pixel at the same index).
    assign can_emit = can_accept && !in_first_half;

    assign busy = (idx < (frame_px << 1));

    // Combinational multiply: buffered A at (idx - frame_px) times the
    // incoming B pixel. Signed if sign_in_r. The buffer read is in an
    // always @* block (Icarus rejects a memory read in a continuous
    // assign).
    logic [7:0]  a_val;
    // 32-bit product: unsigned 255*255 = 65025 needs 17 bits signed, so
    // a 16-bit product would wrap (65025 -> -511). The Python golden
    // uses arbitrary precision, so the RTL must too.
    logic signed [31:0] prod;
    logic signed [31:0] prod_sh;
    logic [7:0]  px_sat;
    always @* begin
        // Guard the read index: during the first half (idx < frame_px)
        // the subtraction underflows, so read 0 (unused -- a_val only
        // matters when a B pixel is emitted).
        if (in_first_half)
            a_val = 8'd0;
        else
            a_val = a_buf[idx - frame_px];
    end
    always @* begin
        if (sign_in_r)
            prod = $signed(a_val) * $signed(pixel_in);
        else
            prod = $signed({1'b0, a_val}) * $signed({1'b0, pixel_in});
    end
    // Arithmetic right shift by shift_r (prod is 32-bit signed).
    assign prod_sh = prod >>> shift_r;
    always @* begin
        if (sign_out_r) begin
            if (prod_sh > 32'sd127)       px_sat = 8'd127;
            else if (prod_sh < -32'sd128) px_sat = 8'h80;
            else                           px_sat = prod_sh[7:0];
        end else begin
            if (prod_sh > 32'sd255)      px_sat = 8'd255;
            else if (prod_sh < 32'sd0)   px_sat = 8'd0;
            else                          px_sat = prod_sh[7:0];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idx <= 25'd0;
            out_idx <= 25'd0;
            out_col <= 13'd0;
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
        end else begin
            if (can_accept) begin
                if (in_first_half)
                    a_buf[idx] <= pixel_in;
                idx <= idx + 25'd1;
                if (idx == (frame_px << 1) - 25'd1)
                    idx <= 25'd0;
            end
            if (!stall) begin
                out_valid <= can_emit;
                if (can_emit) begin
                    pixel_out <= px_sat;
                    out_sof <= (out_idx == 25'd0);
                    out_eol <= (out_col == {1'b0, width_r} - 13'd1);
                    out_last <= (out_idx == frame_px - 25'd1);
                    if (out_idx == frame_px - 25'd1) begin
                        out_idx <= 25'd0;
                        out_col <= 13'd0;
                    end else begin
                        out_idx <= out_idx + 25'd1;
                        if (out_col == {1'b0, width_r} - 13'd1)
                            out_col <= 13'd0;
                        else
                            out_col <= out_col + 13'd1;
                    end
                end else begin
                    out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
                end
            end
        end
    end
endmodule
