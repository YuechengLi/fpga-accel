//=============================================================================
// residual.sv -- Stage 3: elementwise residual addition (skip connections),
// SystemVerilog.
//
// Two input streams A and B (same W/H), one output stream.
//   out = clamp(A + B, 0, 255)   (mode 0, add)
//   out = clamp(A - B, 0, 255)   (mode 1, subtract)
//   out = A                      (mode 2, bypass)
// The two streams are aligned by small FIFOs (depth 4 each), so the TB may
// send A and B with independent random gaps. Output framing is generated
// from the W/H config (raster order, out_sof on the first output pixel,
// out_eol at the end of each output row, out_last on the final pixel).
// A single stall signal (out_valid && !out_ready) freezes the output;
// each input's in_ready is high while its FIFO has space.
//
// Config bus (write between frames; single-cycle writes; 8-bit address):
//   addr 0x00 : mode (0=add, 1=subtract, 2=bypass A)
//   addr 0x01 : image width  (1..4095)
//   addr 0x02 : image height (1..4095)
// Reset defaults: mode=0 (add), W=16, H=16.
//=============================================================================
`timescale 1ns/1ps

module residual #(
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
    logic [1:0]  mode_r;
    logic [11:0] width_r, height_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= 2'd0;
            width_r <= 12'd16;
            height_r <= 12'd16;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_MODE:   mode_r <= cfg_data[1:0];
                CFG_WIDTH:  width_r <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                default: ;
            endcase
        end
    end

    //------------------------- input FIFOs (depth 4) -----------------------
    // Circular buffers with count. Push on (valid && ready), pop when both
    // non-empty and the output stage can take a pair.
    logic [7:0] fifo_a [0:FIFO_DEPTH-1];
    logic [7:0] fifo_b [0:FIFO_DEPTH-1];
    logic [2:0] wr_a, rd_a, cnt_a;
    logic [2:0] wr_b, rd_b, cnt_b;
    logic       pop;
    logic       stall;
    assign stall = out_valid && !out_ready;
    assign a_ready = (cnt_a < FIFO_DEPTH[2:0]);
    assign b_ready = (cnt_b < FIFO_DEPTH[2:0]);
    assign pop = !stall && (cnt_a != 0) && (cnt_b != 0);

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
            if (pop) begin
                rd_a <= rd_a + 3'd1;
                rd_b <= rd_b + 3'd1;
            end
            // count update: push and pop in the same cycle cancel out
            case ({(a_valid && a_ready), pop})
                2'b10: cnt_a <= cnt_a + 3'd1;
                2'b01: cnt_a <= cnt_a - 3'd1;
                default: cnt_a <= cnt_a;
            endcase
            case ({(b_valid && b_ready), pop})
                2'b10: cnt_b <= cnt_b + 3'd1;
                2'b01: cnt_b <= cnt_b - 3'd1;
                default: cnt_b <= cnt_b;
            endcase
        end
    end

    //------------------------- output stage --------------------------------
    // Pops a pair, computes the result, registers the output. Row/col
    // counters generate framing from W/H (no divider).
    logic [12:0] out_row, out_col;
    logic [8:0]  sum_ab;
    logic [8:0]  diff_ab;
    logic [7:0]  result;
    assign sum_ab = {1'b0, fifo_a[rd_a[1:0]]} + {1'b0, fifo_b[rd_b[1:0]]};
    assign diff_ab = {1'b0, fifo_a[rd_a[1:0]]} - {1'b0, fifo_b[rd_b[1:0]]};
    always @* begin
        if (mode_r == 2'd0) begin
            result = (sum_ab > 9'd255) ? 8'd255 : sum_ab[7:0];
        end else if (mode_r == 2'd1) begin
            result = diff_ab[8] ? 8'd0 : diff_ab[7:0];
        end else begin
            result = fifo_a[rd_a[1:0]];
        end
    end

    assign busy = (cnt_a != 0) || (cnt_b != 0) || (out_valid == 1'b1) ||
                  (out_row != 13'd0) || (out_col != 13'd0);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_last <= 1'b0;
            out_row <= 13'd0;
            out_col <= 13'd0;
        end else if (!stall) begin
            out_valid <= pop;
            if (pop) begin
                pixel_out <= result;
                out_sof  <= (out_row == 13'd0) && (out_col == 13'd0);
                out_eol  <= (out_col == {1'b0, width_r} - 13'd1);
                out_last <= (out_row == {1'b0, height_r} - 13'd1) &&
                            (out_col == {1'b0, width_r} - 13'd1);
                if (out_col == {1'b0, width_r} - 13'd1) begin
                    out_col <= 13'd0;
                    out_row <= out_row + 13'd1;
                end else begin
                    out_col <= out_col + 13'd1;
                end
                if ((out_row == {1'b0, height_r} - 13'd1) &&
                    (out_col == {1'b0, width_r} - 13'd1))
                    out_row <= 13'd0;
            end
        end
    end
endmodule : residual
