// critpath_frontend.sv -- the longest combinational stages of frontend
// (Stage 7), each as a small registered module (input regs ->
// combinational logic -> output reg), so Yosys+OpenSTA can measure each
// stage's delay. Same method as critpath_dense.sv (Stage 1) and
// critpath_mc.sv (Stage 3).
// The design is fully pipelined: every stage is registered, so
// Fmax = 1 / (longest single-stage delay + setup).
// Stages:
//   fe_mult      : 8-bit pixel x 16-bit signed coeff (25-bit) -- same as
//                  dense stage_mult (one lane's multiply)
//   fe_add       : adder-tree level-pair, 25 -> 13 -> 7 (27-bit)
//                  (the widest fan-in pair in the 25-product tree)
//   fe_add_wide  : adder-tree level-pair, 7 -> 4 -> 2 (29-bit)
//                  (the widest-operand pair in the tree)
//   fe_shift     : 32-bit barrel shifter + rounding (same as dense)
//   fe_sat       : saturate to [0,255] (same as dense)
//   fe_leaky     : Leaky ReLU slope multiply (32-bit x 16-bit -> 48-bit,
//                  output bits [47:16]) -- same as Stage 3 act_leaky; the
//                  fused Leaky sits in the saturate path, so it is the
//                  candidate bottleneck when Leaky is enabled.
`timescale 1ns/1ps

// Stage: prod -- 8-bit pixel x 16-bit signed coeff (same as RTL)
module fe_mult (
    input  logic        clk,
    input  logic [7:0]  pixel_in,
    input  logic [15:0] coeff_in,
    output logic [24:0] prod
);
    logic [7:0]  pixel_r;
    logic [15:0] coeff_r;
    always_ff @(posedge clk) begin
        pixel_r <= pixel_in;
        coeff_r <= coeff_in;
    end
    always_ff @(posedge clk)
        prod <= $signed({1'b0, pixel_r}) * coeff_r;
endmodule : fe_mult

// Stage: adder-tree level-pair, 25 x 27-bit inputs -> 13 sums (level 1)
// -> 7 sums (level 2), registered. The widest fan-in pair of the tree.
module fe_add (
    input  logic         clk,
    input  logic [26:0]  in0,  in1,  in2,  in3,  in4,  in5,  in6,  in7,
    input  logic [26:0]  in8,  in9,  in10, in11, in12, in13, in14, in15,
    input  logic [26:0]  in16, in17, in18, in19, in20, in21, in22, in23,
    input  logic [26:0]  in24,
    output logic [26:0]  out0, out1, out2, out3, out4, out5, out6
);
    logic [26:0] r [0:24];
    logic [26:0] s1 [0:12];
    logic [26:0] s2 [0:6];
    int i;
    always_ff @(posedge clk) begin
        r[0]<=in0; r[1]<=in1; r[2]<=in2; r[3]<=in3; r[4]<=in4; r[5]<=in5;
        r[6]<=in6; r[7]<=in7; r[8]<=in8; r[9]<=in9; r[10]<=in10;
        r[11]<=in11; r[12]<=in12; r[13]<=in13; r[14]<=in14; r[15]<=in15;
        r[16]<=in16; r[17]<=in17; r[18]<=in18; r[19]<=in19; r[20]<=in20;
        r[21]<=in21; r[22]<=in22; r[23]<=in23; r[24]<=in24;
    end
    always @* begin
        for (i = 0; i < 12; i++) s1[i] = r[2*i] + r[2*i+1];
        s1[12] = r[24];
        for (i = 0; i < 6; i++) s2[i] = s1[2*i] + s1[2*i+1];
        s2[6] = s1[12];
    end
    always_ff @(posedge clk) begin
        out0<=s2[0]; out1<=s2[1]; out2<=s2[2]; out3<=s2[3]; out4<=s2[4];
        out5<=s2[5]; out6<=s2[6];
    end
endmodule : fe_add

// Stage: adder-tree level-pair, 7 x 29-bit inputs -> 4 sums (level 1)
// -> 2 sums (level 2), registered. The widest-operand pair of the tree.
module fe_add_wide (
    input  logic         clk,
    input  logic [28:0]  in0, in1, in2, in3, in4, in5, in6,
    output logic [28:0]  out0, out1
);
    logic [28:0] r [0:6];
    logic [28:0] s1 [0:3];
    logic [28:0] s2 [0:1];
    always_ff @(posedge clk) begin
        r[0]<=in0; r[1]<=in1; r[2]<=in2; r[3]<=in3;
        r[4]<=in4; r[5]<=in5; r[6]<=in6;
    end
    always @* begin
        s1[0] = r[0] + r[1]; s1[1] = r[2] + r[3];
        s1[2] = r[4] + r[5]; s1[3] = r[6];
        s2[0] = s1[0] + s1[1];
        s2[1] = s1[2] + s1[3];
    end
    always_ff @(posedge clk) begin
        out0<=s2[0]; out1<=s2[1];
    end
endmodule : fe_add_wide

// Stage: shift only -- 32-bit barrel shifter + rounding (same as dense)
module fe_shift (
    input  logic        clk,
    input  logic [31:0] acc_in,
    input  logic [3:0]  shift_in,
    output logic [31:0] shifted
);
    logic signed [31:0] acc_r;
    logic [3:0]  shift_r;
    logic signed [31:0] rnd;
    always_ff @(posedge clk) begin
        acc_r   <= acc_in;
        shift_r <= shift_in;
    end
    always @* begin
        rnd = $signed(acc_r) + ((shift_r == 0) ? 32'sd0 : (32'sd1 <<< (shift_r - 1'b1)));
    end
    always_ff @(posedge clk) shifted <= rnd >>> shift_r;
endmodule : fe_shift

// Stage: saturate only -- clamp a 32-bit signed value to [0,255]
module fe_sat (
    input  logic        clk,
    input  logic signed [31:0] shifted_in,
    output logic [7:0]  pixel_out
);
    logic signed [31:0] shifted_r;
    always_ff @(posedge clk) shifted_r <= shifted_in;
    always_ff @(posedge clk) begin
        if (shifted_r > 32'sd255)      pixel_out <= 8'd255;
        else if (shifted_r < 32'sd0)   pixel_out <= 8'd0;
        else                            pixel_out <= shifted_r[7:0];
    end
endmodule : fe_sat

// Stage: Leaky ReLU slope multiply -- 32-bit x 16-bit -> 48-bit,
// output bits [47:16]. Same as Stage 3 stage_act_leaky.
module fe_leaky (
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
endmodule : fe_leaky
