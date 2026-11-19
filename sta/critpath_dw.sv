// critpath_dw.sv -- the longest combinational stages of conv2d_dw
// (Stage 2, DW, 17x17 square / 31 strip / dilated), each as a small
// registered module (input regs -> combinational logic -> output reg), so
// Yosys+OpenSTA can measure each stage's delay. Same method as
// critpath_dense.sv. The design is fully pipelined: every stage is
// registered, so Fmax = 1 / (longest single-stage delay + setup).
// Stages:
//   stage_mult_dw    : 8-bit unsigned pixel x 16-bit signed coeff (25-bit)
//                      (identical to Stage 1's stage_mult)
//   stage_add_dw     : adder-tree level-pair, 289 -> 145 -> 73 (27-bit)
//                      (the widest fan-in pair of the 289-product tree;
//                      packed bus, since 289 ports are unwieldy)
//   stage_add_wide_dw: adder-tree level-pair, 8 -> 4 -> 2 (31-bit)
//                      (the widest-operand pair of the tree)
//   stage_shift_only : 32-bit barrel shifter + rounding (same as Stage 1)
//   stage_sat_only   : saturate to [0,255] (same as Stage 1)
// Note: the adder-tree level-pair delay depends on the operand width and
// on being two adds in series, not on how many parallel adders there are,
// so stage_add_dw (289 inputs) is expected to match Stage 1's stage_add
// (121 inputs, 1.566 ns); it is measured here to confirm.
`timescale 1ns/1ps

// Stage A: prod -- 8-bit unsigned pixel x 16-bit signed coeff (same as RTL)
module stage_mult_dw (
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
endmodule : stage_mult_dw

// Stage B: adder-tree level-pair, 289 x 27-bit inputs -> 145 sums (level 1)
// -> 73 sums (level 2), registered. Packed bus (289*27 = 7803 bits in,
// 73*27 = 1971 bits out).
module stage_add_dw (
    input  logic         clk,
    input  logic [27*289-1:0] in_bus,
    output logic [27*73-1:0]  out_bus
);
    logic [26:0] r [0:288];
    logic [26:0] s1 [0:144];
    logic [26:0] s2 [0:72];
    int i;
    always_ff @(posedge clk) begin
        for (i = 0; i < 289; i++) r[i] <= in_bus[i*27 +: 27];
    end
    always @* begin
        for (i = 0; i < 144; i++) s1[i] = r[2*i] + r[2*i+1];
        s1[144] = r[288];
        for (i = 0; i < 72; i++) s2[i] = s1[2*i] + s1[2*i+1];
        s2[72] = s1[144];
    end
    always_ff @(posedge clk) begin
        for (i = 0; i < 73; i++) out_bus[i*27 +: 27] <= s2[i];
    end
endmodule : stage_add_dw

// Stage C: adder-tree level-pair, 8 x 31-bit inputs -> 4 sums (level 1)
// -> 2 sums (level 2), registered. The widest-operand pair of the tree.
module stage_add_wide_dw (
    input  logic         clk,
    input  logic [30:0]  in0, in1, in2, in3, in4, in5, in6, in7,
    output logic [30:0]  out0, out1
);
    logic [30:0] r [0:7];
    logic [30:0] s1 [0:3];
    logic [30:0] s2 [0:1];
    always_ff @(posedge clk) begin
        r[0]<=in0; r[1]<=in1; r[2]<=in2; r[3]<=in3;
        r[4]<=in4; r[5]<=in5; r[6]<=in6; r[7]<=in7;
    end
    always @* begin
        s1[0] = r[0] + r[1]; s1[1] = r[2] + r[3];
        s1[2] = r[4] + r[5]; s1[3] = r[6] + r[7];
        s2[0] = s1[0] + s1[1];
        s2[1] = s1[2] + s1[3];
    end
    always_ff @(posedge clk) begin
        out0<=s2[0]; out1<=s2[1];
    end
endmodule : stage_add_wide_dw

// Stage: shift only -- 32-bit barrel shifter + rounding (same as Stage 1)
module stage_shift_only_dw (
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
endmodule : stage_shift_only_dw

// Stage: saturate only -- clamp a 32-bit signed value to [0,255]
module stage_sat_only_dw (
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
endmodule : stage_sat_only_dw
