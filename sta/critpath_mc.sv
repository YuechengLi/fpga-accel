// critpath_mc.sv -- the longest combinational stages of the Stage 3
// modules, each as a small registered module (input regs -> combinational
// logic -> output reg), so Yosys+OpenSTA can measure each stage's delay.
// Same method as critpath_dense.sv (Stage 1).
// The designs are fully pipelined: every stage is registered, so
// Fmax = 1 / (longest single-stage delay + setup).
// Stages (new in Stage 3):
//   stage_mc_accum   : cross-lane sum (4 x 32-bit lane partials -> 1 sum)
//                      + accumulator add (acc_rd + sum, first-tile mux)
//   stage_pool_max   : max 2x2 compare tree (4 x 8-bit -> 1 max)
//   stage_pool_gap   : GAP accumulate add (18-bit sum + 8-bit pixel)
//   stage_act_leaky  : Leaky ReLU slope multiply (32-bit x 16-bit -> 48-bit,
//                      output bits [47:16])
//   stage_act_pwl    : PWL interpolation (16-bit dslope x 10-bit frac ->
//                      26-bit product, >> 10, + 16-bit y0)
// (The shift-only and saturate-only stages are identical to Stage 1's
//  stage_shift_only / stage_sat_only and are not re-measured here.)
`timescale 1ns/1ps

// Stage: MC accumulate -- 4 lane partials (32-bit each) summed through a
// 2-level adder tree, then added to the accumulator (or taken directly on
// the first tile). This is the new combinational path in conv2d_mc.
module stage_mc_accum (
    input  logic        clk,
    input  logic [31:0] p0, p1, p2, p3,
    input  logic [31:0] acc_rd,
    input  logic        is_first_tile,
    output logic [31:0] acc_wr
);
    logic [31:0] p0_r, p1_r, p2_r, p3_r, acc_rd_r;
    logic        is_first_tile_r;
    logic [31:0] s0, s1, tile_sum;
    logic [31:0] acc_wr_c;
    always_ff @(posedge clk) begin
        p0_r <= p0; p1_r <= p1; p2_r <= p2; p3_r <= p3;
        acc_rd_r <= acc_rd;
        is_first_tile_r <= is_first_tile;
    end
    always @* begin
        s0 = p0_r + p1_r;
        s1 = p2_r + p3_r;
        tile_sum = s0 + s1;
        acc_wr_c = is_first_tile_r ? tile_sum : (acc_rd_r + tile_sum);
    end
    always_ff @(posedge clk) acc_wr <= acc_wr_c;
endmodule : stage_mc_accum

// Stage: pool max 2x2 -- 4-way max compare tree (two levels of 8-bit
// unsigned compare), as in pool.sv's max mode.
module stage_pool_max (
    input  logic       clk,
    input  logic [7:0] a, b, c, d,
    output logic [7:0] m
);
    logic [7:0] a_r, b_r, c_r, d_r;
    logic [7:0] m01, m23, m_c;
    always_ff @(posedge clk) begin
        a_r <= a; b_r <= b; c_r <= c; d_r <= d;
    end
    always @* begin
        m01 = (a_r >= b_r) ? a_r : b_r;
        m23 = (c_r >= d_r) ? c_r : d_r;
        m_c = (m01 >= m23) ? m01 : m23;
    end
    always_ff @(posedge clk) m <= m_c;
endmodule : stage_pool_max

// Stage: pool GAP accumulate -- 18-bit running sum + 8-bit pixel, as in
// pool.sv's GAP mode.
module stage_pool_gap (
    input  logic        clk,
    input  logic [17:0] sum_in,
    input  logic [7:0]  pix_in,
    output logic [17:0] sum_out
);
    logic [17:0] sum_r;
    logic [7:0]  pix_r;
    always_ff @(posedge clk) begin
        sum_r <= sum_in;
        pix_r <= pix_in;
    end
    always_ff @(posedge clk) sum_out <= sum_r + {10'b0, pix_r};
endmodule : stage_pool_gap

// Stage: act Leaky ReLU slope multiply -- 32-bit signed x 16-bit signed
// -> 48-bit product, output bits [47:16] (i.e. (x * slope_fp) >> 16).
module stage_act_leaky (
    input  logic        clk,
    input  logic [31:0] x_in,
    input  logic [15:0] slope_in,
    output logic [31:0] y_out
);
    logic signed [31:0] x_r;
    logic signed [15:0] slope_r;
    logic signed [47:0] prod;
    always_ff @(posedge clk) begin
        x_r <= x_in;
        slope_r <= slope_in;
    end
    always @* prod = x_r * slope_r;
    always_ff @(posedge clk) y_out <= prod[47:16];
endmodule : stage_act_leaky

// Stage: act PWL interpolation -- 16-bit signed dslope x 11-bit unsigned
// frac (0..1024) -> 27-bit product, >> 10, + 16-bit signed y0, 32-bit
// sum (no 16-bit wrap). As in act.sv (frac widened to 11 bits for the
// x==8.0 top edge, y widened to 32 bits for GELU/SiLU range).
module stage_act_pwl (
    input  logic        clk,
    input  logic [15:0] dslope_in,
    input  logic [10:0] frac_in,
    input  logic [15:0] y0_in,
    output logic [31:0] y_out
);
    logic signed [15:0] dslope_r;
    logic [10:0] frac_r;
    logic signed [15:0] y0_r;
    logic signed [31:0] prod;
    logic signed [31:0] y_c;
    always_ff @(posedge clk) begin
        dslope_r <= dslope_in;
        frac_r   <= frac_in;
        y0_r     <= y0_in;
    end
    always @* begin
        prod = $signed({{16{dslope_r[15]}}, dslope_r}) *
               $signed({21'd0, frac_r});
        y_c  = $signed({{16{y0_r[15]}}, y0_r}) +
               $signed({{16{prod[25]}}, prod[25:10]});
    end
    always_ff @(posedge clk) y_out <= y_c;
endmodule : stage_act_pwl
