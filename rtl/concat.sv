//=============================================================================
// concat.sv -- Stage 4: channel concatenation of two streams, SystemVerilog.
//
// Two input streams A and B (same W/H), one output stream.
//   mode 0 (sequential): out = A frame followed by B frame. Channel order
//     is (C,H,W): all of channel 0 (A), then all of channel 1 (B). The
//     consumer (e.g. a conv with C_in tiling) reads the stream as
//     channel-major. Output is 2*W*H pixels, out_eol at the end of each
//     W-pixel row.
//   mode 1 (interleave): out = A and B interleaved per pixel: for each
//     spatial position, A then B. Channel order is (H,W,C). Output is
//     2*W*H pixels, out_eol at the end of each 2W-pixel row.
// The two streams are aligned by small FIFOs (depth 4 each), so the TB may
// send A and B with independent random gaps. A single stall signal
// (out_valid && !out_ready) freezes the output; each input's in_ready is
// high while its FIFO has space. If B arrives early, it waits in its FIFO
// until A is fully output (sequential mode) or until its pair is ready
// (interleave mode); backpressure on B is correct, not a deadlock.
//
// Config bus (write between frames; single-cycle writes; 8-bit address):
//   addr 0x00 : mode (0=sequential, 1=interleave)
//   addr 0x01 : image width  (1..4095)
//   addr 0x02 : image height (1..4095)
// Reset defaults: mode=0 (sequential), W=16, H=16.
//=============================================================================
`timescale 1ns/1ps

module concat #(
    parameter int FIFO_DEPTH = 4
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    // stream A in
    input  logic [7:0]  a_in,
    input  logic        a_valid,
    input  logic        a_sof,
    output logic        a_ready,
    // stream B in
    input  logic [7:0]  b_in,
    input  logic        b_valid,
    input  logic        b_sof,
    output logic        b_ready,
    // stream out
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [7:0] CFG_MODE   = 8'h00;
    localparam logic [7:0] CFG_WIDTH  = 8'h01;
    localparam logic [7:0] CFG_HEIGHT = 8'h02;

    //------------------------- config registers ----------------------------
    logic        mode_r;  // 0 = sequential, 1 = interleave
    logic [11:0] width_r, height_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= 1'b0;
            width_r <= 12'd16;
            height_r <= 12'd16;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_MODE:   mode_r <= cfg_data[0];
                CFG_WIDTH:  width_r <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                default: ;
            endcase
        end
    end

    //------------------------- input FIFOs (depth 4) -----------------------
    // Declared before the output sequencing logic, which reads cnt_a/cnt_b
    // (Icarus requires a declaration before its use).
    logic [7:0] fifo_a [0:FIFO_DEPTH-1];
    logic [7:0] fifo_b [0:FIFO_DEPTH-1];
    logic [2:0] wr_a, rd_a, cnt_a;
    logic [2:0] wr_b, rd_b, cnt_b;
    assign a_ready = (cnt_a < FIFO_DEPTH[2:0]);
    assign b_ready = (cnt_b < FIFO_DEPTH[2:0]);

    //------------------------- output sequencing ---------------------------
    // out_idx counts output pixels 0..2*W*H-1. Sequential: first W*H are
    // A, next W*H are B. Interleave: even idx is A, odd idx is B.
    // A pop happens when we emit an A pixel; a B pop when we emit a B
    // pixel. We can emit only when the needed FIFO is non-empty.
    // out_col tracks the column within the output row (row length W in
    // sequential mode, 2W in interleave mode) for out_eol; no divider
    // is synthesized.
    logic [24:0] out_idx;
    logic [24:0] total_px;
    logic [12:0] out_col;
    logic [12:0] out_row_len;  // W (sequential) or 2W (interleave)
    logic        need_a, need_b, can_emit;
    logic        pop_a, pop_b;
    logic        emit_a;  // 1 = this output pixel comes from A
    logic        stall;

    assign total_px = {1'b0, width_r} * {1'b0, height_r} * 25'd2;
    assign out_row_len = mode_r ? ({1'b0, width_r} << 1) : {1'b0, width_r};
    assign emit_a = mode_r ? !out_idx[0] :
                    (out_idx < ({1'b0, width_r} * {1'b0, height_r}));
    assign need_a = emit_a;
    assign need_b = !emit_a;
    assign stall = out_valid && !out_ready;
    assign can_emit = (out_idx < total_px) && !stall &&
                      ((need_a && (cnt_a != 0)) || (need_b && (cnt_b != 0)));
    assign pop_a = can_emit && need_a;
    assign pop_b = can_emit && need_b;

    assign busy = (out_idx < total_px);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_a <= 3'd0; rd_a <= 3'd0; cnt_a <= 3'd0;
            wr_b <= 3'd0; rd_b <= 3'd0; cnt_b <= 3'd0;
        end else begin
            if (a_valid && a_ready) begin
                fifo_a[wr_a[1:0]] <= a_in;
                wr_a <= wr_a + 3'd1;
            end
            if (b_valid && b_ready) begin
                fifo_b[wr_b[1:0]] <= b_in;
                wr_b <= wr_b + 3'd1;
            end
            if (pop_a) rd_a <= rd_a + 3'd1;
            if (pop_b) rd_b <= rd_b + 3'd1;
            // count updates (push and pop can coincide)
            if ((a_valid && a_ready) && !pop_a)       cnt_a <= cnt_a + 3'd1;
            else if (!(a_valid && a_ready) && pop_a)  cnt_a <= cnt_a - 3'd1;
            if ((b_valid && b_ready) && !pop_b)       cnt_b <= cnt_b + 3'd1;
            else if (!(b_valid && b_ready) && pop_b)  cnt_b <= cnt_b - 3'd1;
        end
    end

    //------------------------- output registers ----------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_idx <= 25'd0;
            out_col <= 13'd0;
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= can_emit;
            if (can_emit) begin
                pixel_out <= emit_a ? fifo_a[rd_a[1:0]] : fifo_b[rd_b[1:0]];
                out_sof <= (out_idx == 25'd0);
                out_eol <= (out_col == out_row_len - 13'd1);
                out_last <= (out_idx == total_px - 25'd1);
                if (out_idx == total_px - 25'd1) begin
                    out_idx <= 25'd0;
                    out_col <= 13'd0;
                end else begin
                    out_idx <= out_idx + 25'd1;
                    if (out_col == out_row_len - 13'd1)
                        out_col <= 13'd0;
                    else
                        out_col <= out_col + 13'd1;
                end
            end else begin
                out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
            end
        end
    end
endmodule
