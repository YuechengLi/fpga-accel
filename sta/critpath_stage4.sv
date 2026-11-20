// critpath_stage4.sv -- the longest combinational stages of the Stage 4
// modules, each as a small registered module (input regs -> combinational
// logic -> output reg), so Yosys+OpenSTA can measure each stage's delay.
// Same method as critpath_mc.sv (Stage 3). The designs are fully
// pipelined: every stage is registered, so Fmax = 1 / (longest
// single-stage delay + setup).
// Stages (new in Stage 4):
//   stage_ln_norm_a  : LayerNorm normalize stage A -- diff x inv_std
//                      >> 7 -> norm_q (one 32x16 multiply).
//   stage_ln_norm_b  : LayerNorm normalize stage B -- norm_q x gamma
//                      >> 23 + beta (one 32x16 multiply). The
//                      unpipelined two-multiply chain violates 3.333 ns
//                      (4.301 ns arrival, slack -1.009); pipelining
//                      splits it into these two stages.
//   stage_ln_sumsq   : LayerNorm sumsq accumulate -- px (16-bit signed)
//                      x px -> 32-bit, + 32-bit sumsq_acc.
//   stage_act_mlut   : MLUT lookup -- coarse base (16-bit) + fine delta
//                      (16-bit, 8-to-1 mux selected) -> 16-bit sum,
//                      saturating.
//   stage_dense_leaky: dense fused Leaky -- 32-bit x 16-bit -> 48-bit,
//                      output bits [47:16] (same shape as Stage 3's
//                      act_leaky, now in the dense emit path).
//   stage_sg_mult    : SimpleGate -- 8-bit x 8-bit -> 32-bit product,
//                      >> shift, saturate to 8-bit.
//   stage_up_bilin   : bilinear upsample -- (a+b+c+d+2) >> 2 (three
//                      8-bit adds + shift).
// (Concat is a 2:1 mux and the scheduler is control logic; neither has
// a datapath critical path worth a separate stage. The unified engine's
// stream mux is a 2:1 mux on the pixel path, also trivial.)
`timescale 1ns/1ps

// Stage: LayerNorm normalize, PIPELINED (the unpipelined two-multiply
// chain violates 3.333 ns at 4.301 ns arrival). Stage A: diff x inv_std
// >> 7 -> norm_q (registered). Stage B: norm_q x gamma >> 23 + beta
// (registered). Each stage has one 32x16 multiply.
module stage_ln_norm_a (
    input  logic        clk,
    input  logic signed [31:0] diff,
    input  logic signed [15:0] inv_std,
    output logic signed [31:0] norm_q
);
    logic signed [31:0] diff_r;
    logic signed [15:0] inv_std_r;
    logic signed [47:0] prod1_c;
    logic signed [31:0] norm_q_c;
    always_ff @(posedge clk) begin
        diff_r <= diff; inv_std_r <= inv_std;
    end
    assign prod1_c = diff_r * $signed({{16{inv_std_r[15]}}, inv_std_r});
    assign norm_q_c = prod1_c >>> 7;
    always_ff @(posedge clk) norm_q <= norm_q_c;
endmodule : stage_ln_norm_a

module stage_ln_norm_b (
    input  logic        clk,
    input  logic signed [31:0] norm_q,
    input  logic signed [15:0] gamma,
    input  logic signed [15:0] beta,
    output logic signed [31:0] y
);
    logic signed [31:0] norm_q_r;
    logic signed [15:0] gamma_r, beta_r;
    logic signed [47:0] prod2_c;
    logic signed [31:0] y_c;
    always_ff @(posedge clk) begin
        norm_q_r <= norm_q; gamma_r <= gamma; beta_r <= beta;
    end
    assign prod2_c = norm_q_r * $signed({{16{gamma_r[15]}}, gamma_r});
    assign y_c = (prod2_c >>> 23) + $signed({{16{beta_r[15]}}, beta_r});
    always_ff @(posedge clk) y <= y_c;
endmodule : stage_ln_norm_b

// Stage: LayerNorm sumsq accumulate -- px*px + sumsq_acc.
module stage_ln_sumsq (
    input  logic        clk,
    input  logic signed [15:0] px,
    input  logic signed [31:0] sumsq_acc,
    output logic signed [31:0] sumsq_new
);
    logic signed [15:0] px_r;
    logic signed [31:0] sumsq_r;
    logic signed [31:0] sumsq_c;
    always_ff @(posedge clk) begin
        px_r <= px; sumsq_r <= sumsq_acc;
    end
    assign sumsq_c = sumsq_r + $signed(px_r) * $signed(px_r);
    always_ff @(posedge clk) sumsq_new <= sumsq_c;
endmodule : stage_ln_sumsq

// Stage: MLUT lookup -- base + delta (8-to-1 mux), saturating to 16-bit.
module stage_act_mlut (
    input  logic        clk,
    input  logic signed [15:0] base,
    input  logic signed [15:0] d0, d1, d2, d3, d4, d5, d6, d7,
    input  logic [2:0] sel,
    output logic signed [15:0] out
);
    logic signed [15:0] base_r;
    logic signed [15:0] d_r [0:7];
    logic [2:0] sel_r;
    logic signed [15:0] delta_c;
    logic signed [16:0] sum_c;
    logic signed [15:0] out_c;
    always_ff @(posedge clk) begin
        base_r <= base; sel_r <= sel;
        d_r[0] <= d0; d_r[1] <= d1; d_r[2] <= d2; d_r[3] <= d3;
        d_r[4] <= d4; d_r[5] <= d5; d_r[6] <= d6; d_r[7] <= d7;
    end
    always @* begin
        case (sel_r)
            3'd0: delta_c = d_r[0];
            3'd1: delta_c = d_r[1];
            3'd2: delta_c = d_r[2];
            3'd3: delta_c = d_r[3];
            3'd4: delta_c = d_r[4];
            3'd5: delta_c = d_r[5];
            3'd6: delta_c = d_r[6];
            default: delta_c = d_r[7];
        endcase
    end
    assign sum_c = $signed({base_r[15], base_r}) + $signed({delta_c[15], delta_c});
    assign out_c = (sum_c > 17'sd32767) ? 16'sd32767 :
                   (sum_c < -17'sd32768) ? -16'sd32768 : sum_c[15:0];
    always_ff @(posedge clk) out <= out_c;
endmodule : stage_act_mlut

// Stage: dense fused Leaky -- 32-bit x 16-bit -> 48-bit, bits [47:16].
module stage_dense_leaky (
    input  logic        clk,
    input  logic signed [31:0] x,
    input  logic signed [15:0] slope,
    output logic signed [31:0] y
);
    logic signed [31:0] x_r;
    logic signed [15:0] slope_r;
    logic signed [47:0] prod_c;
    logic signed [31:0] y_c;
    always_ff @(posedge clk) begin
        x_r <= x; slope_r <= slope;
    end
    assign prod_c = x_r * $signed({{16{slope_r[15]}}, slope_r});
    assign y_c = prod_c[47:16];
    always_ff @(posedge clk) y <= y_c;
endmodule : stage_dense_leaky

// Stage: SimpleGate -- 8-bit x 8-bit -> 32-bit, >> shift, saturate.
module stage_sg_mult (
    input  logic        clk,
    input  logic [7:0] a, b,
    input  logic [3:0] shift,
    output logic [7:0] y
);
    logic [7:0] a_r, b_r;
    logic [3:0] shift_r;
    logic signed [31:0] prod_c;
    logic signed [31:0] sh_c;
    logic [7:0] y_c;
    always_ff @(posedge clk) begin
        a_r <= a; b_r <= b; shift_r <= shift;
    end
    assign prod_c = $signed({1'b0, a_r}) * $signed({1'b0, b_r});
    assign sh_c = prod_c >>> shift_r;
    assign y_c = (sh_c > 32'sd255) ? 8'd255 :
                 (sh_c < 32'sd0) ? 8'd0 : sh_c[7:0];
    always_ff @(posedge clk) y <= y_c;
endmodule : stage_sg_mult

// Stage: bilinear upsample -- (a+b+c+d+2) >> 2.
module stage_up_bilin (
    input  logic        clk,
    input  logic [7:0] a, b, c, d,
    output logic [7:0] y
);
    logic [7:0] a_r, b_r, c_r, d_r;
    logic [9:0] sum_c;
    logic [7:0] y_c;
    always_ff @(posedge clk) begin
        a_r <= a; b_r <= b; c_r <= c; d_r <= d;
    end
    assign sum_c = {2'd0, a_r} + {2'd0, b_r} + {2'd0, c_r} + {2'd0, d_r} + 10'd2;
    assign y_c = sum_c[9:2];
    always_ff @(posedge clk) y <= y_c;
endmodule : stage_up_bilin
