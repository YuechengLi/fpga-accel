// critpath_dense.sv -- the longest combinational stages of conv2d_dense
// (Stage 1, 11x11 dense), each as a small registered module (input regs ->
// combinational logic -> output reg), so Yosys+OpenSTA can measure each
// stage's delay. Same method as the reference critpath_px.sv.
// The design is fully pipelined: every stage is registered, so
// Fmax = 1 / (longest single-stage delay + setup).
// Stages:
//   stage_mult       : 8-bit unsigned pixel x 16-bit signed coeff (25-bit)
//   stage_add        : adder-tree level-pair, 121 -> 61 -> 31 (27-bit)
//                      (the widest fan-in pair in the tree)
//   stage_add_wide   : adder-tree level-pair, 8 -> 4 -> 2 (31-bit)
//                      (the widest-operand pair in the tree)
//   stage_shift_only : 32-bit barrel shifter + rounding (same as reference)
//   stage_sat_only   : saturate to [0,255] (same as reference)
`timescale 1ns/1ps

// Stage A: prod -- 8-bit unsigned pixel x 16-bit signed coeff (same as RTL)
module stage_mult (
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
endmodule : stage_mult

// Stage B: adder-tree level-pair, 121 x 27-bit inputs -> 61 sums (level 1)
// -> 31 sums (level 2), registered. The widest fan-in pair of the tree.
module stage_add (
    input  logic         clk,
    input  logic [26:0]  in0,  in1,  in2,  in3,  in4,  in5,  in6,  in7,
    input  logic [26:0]  in8,  in9,  in10, in11, in12, in13, in14, in15,
    input  logic [26:0]  in16, in17, in18, in19, in20, in21, in22, in23,
    input  logic [26:0]  in24, in25, in26, in27, in28, in29, in30, in31,
    input  logic [26:0]  in32, in33, in34, in35, in36, in37, in38, in39,
    input  logic [26:0]  in40, in41, in42, in43, in44, in45, in46, in47,
    input  logic [26:0]  in48, in49, in50, in51, in52, in53, in54, in55,
    input  logic [26:0]  in56, in57, in58, in59, in60, in61, in62, in63,
    input  logic [26:0]  in64, in65, in66, in67, in68, in69, in70, in71,
    input  logic [26:0]  in72, in73, in74, in75, in76, in77, in78, in79,
    input  logic [26:0]  in80, in81, in82, in83, in84, in85, in86, in87,
    input  logic [26:0]  in88, in89, in90, in91, in92, in93, in94, in95,
    input  logic [26:0]  in96, in97, in98, in99, in100, in101, in102, in103,
    input  logic [26:0]  in104, in105, in106, in107, in108, in109, in110,
    input  logic [26:0]  in111, in112, in113, in114, in115, in116, in117,
    input  logic [26:0]  in118, in119, in120,
    output logic [26:0]  out0, out1, out2, out3, out4, out5, out6, out7,
    output logic [26:0]  out8, out9, out10, out11, out12, out13, out14,
    output logic [26:0]  out15, out16, out17, out18, out19, out20, out21,
    output logic [26:0]  out22, out23, out24, out25, out26, out27, out28,
    output logic [26:0]  out29, out30
);
    logic [26:0] r [0:120];
    logic [26:0] s1 [0:60];
    logic [26:0] s2 [0:30];
    int i;
    always_ff @(posedge clk) begin
        r[0]<=in0; r[1]<=in1; r[2]<=in2; r[3]<=in3; r[4]<=in4; r[5]<=in5;
        r[6]<=in6; r[7]<=in7; r[8]<=in8; r[9]<=in9; r[10]<=in10; r[11]<=in11;
        r[12]<=in12; r[13]<=in13; r[14]<=in14; r[15]<=in15; r[16]<=in16;
        r[17]<=in17; r[18]<=in18; r[19]<=in19; r[20]<=in20; r[21]<=in21;
        r[22]<=in22; r[23]<=in23; r[24]<=in24; r[25]<=in25; r[26]<=in26;
        r[27]<=in27; r[28]<=in28; r[29]<=in29; r[30]<=in30; r[31]<=in31;
        r[32]<=in32; r[33]<=in33; r[34]<=in34; r[35]<=in35; r[36]<=in36;
        r[37]<=in37; r[38]<=in38; r[39]<=in39; r[40]<=in40; r[41]<=in41;
        r[42]<=in42; r[43]<=in43; r[44]<=in44; r[45]<=in45; r[46]<=in46;
        r[47]<=in47; r[48]<=in48; r[49]<=in49; r[50]<=in50; r[51]<=in51;
        r[52]<=in52; r[53]<=in53; r[54]<=in54; r[55]<=in55; r[56]<=in56;
        r[57]<=in57; r[58]<=in58; r[59]<=in59; r[60]<=in60; r[61]<=in61;
        r[62]<=in62; r[63]<=in63; r[64]<=in64; r[65]<=in65; r[66]<=in66;
        r[67]<=in67; r[68]<=in68; r[69]<=in69; r[70]<=in70; r[71]<=in71;
        r[72]<=in72; r[73]<=in73; r[74]<=in74; r[75]<=in75; r[76]<=in76;
        r[77]<=in77; r[78]<=in78; r[79]<=in79; r[80]<=in80; r[81]<=in81;
        r[82]<=in82; r[83]<=in83; r[84]<=in84; r[85]<=in85; r[86]<=in86;
        r[87]<=in87; r[88]<=in88; r[89]<=in89; r[90]<=in90; r[91]<=in91;
        r[92]<=in92; r[93]<=in93; r[94]<=in94; r[95]<=in95; r[96]<=in96;
        r[97]<=in97; r[98]<=in98; r[99]<=in99; r[100]<=in100; r[101]<=in101;
        r[102]<=in102; r[103]<=in103; r[104]<=in104; r[105]<=in105;
        r[106]<=in106; r[107]<=in107; r[108]<=in108; r[109]<=in109;
        r[110]<=in110; r[111]<=in111; r[112]<=in112; r[113]<=in113;
        r[114]<=in114; r[115]<=in115; r[116]<=in116; r[117]<=in117;
        r[118]<=in118; r[119]<=in119; r[120]<=in120;
    end
    always @* begin
        for (i = 0; i < 60; i++) s1[i] = r[2*i] + r[2*i+1];
        s1[60] = r[120];
        for (i = 0; i < 30; i++) s2[i] = s1[2*i] + s1[2*i+1];
        s2[30] = s1[60];
    end
    always_ff @(posedge clk) begin
        out0<=s2[0]; out1<=s2[1]; out2<=s2[2]; out3<=s2[3]; out4<=s2[4];
        out5<=s2[5]; out6<=s2[6]; out7<=s2[7]; out8<=s2[8]; out9<=s2[9];
        out10<=s2[10]; out11<=s2[11]; out12<=s2[12]; out13<=s2[13];
        out14<=s2[14]; out15<=s2[15]; out16<=s2[16]; out17<=s2[17];
        out18<=s2[18]; out19<=s2[19]; out20<=s2[20]; out21<=s2[21];
        out22<=s2[22]; out23<=s2[23]; out24<=s2[24]; out25<=s2[25];
        out26<=s2[26]; out27<=s2[27]; out28<=s2[28]; out29<=s2[29];
        out30<=s2[30];
    end
endmodule : stage_add

// Stage C: adder-tree level-pair, 8 x 31-bit inputs -> 4 sums (level 1)
// -> 2 sums (level 2), registered. The widest-operand pair of the tree.
module stage_add_wide (
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
endmodule : stage_add_wide

// Stage: shift only -- 32-bit barrel shifter + rounding (same as reference)
module stage_shift_only (
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
endmodule : stage_shift_only

// Stage: saturate only -- clamp a 32-bit signed value to [0,255]
module stage_sat_only (
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
endmodule : stage_sat_only
