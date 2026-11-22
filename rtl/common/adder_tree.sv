//=============================================================================
// adder_tree.sv -- Stage 7b: generic pipelined adder tree.
//
// Reduces N signed inputs (width W) to one sum, 2 adder levels per
// pipeline stage (the same shape as conv2d_mc's hand-written tree, which
// closes at ~469 MHz in Nangate45). Always 8 levels / 4 stages for
// N <= 256; levels past the reduction just pass the single value
// through (sign-extended), so the latency is a uniform 4 cycles and the
// output width is a uniform W+8 bits (for W=25: 33 bits; the consumer
// accumulator is sized accordingly -- the true sums here are far below
// 2^31, verified by the goldens and by Stage 6's accumulator bound).
// The stage registers freeze on `stall`, like every other pipeline
// register in this repo (backpressure must not lose a partial sum).
//
// Level counts (N -> after each level), for the sizes used here:
//   N=160: 160->80->40->20->10->5->3->2->1  (L3 tap-parallel MAC)
//   N=81:  81->41->21->11->6->3->2->1->1    (L1 9x9 MAC)
//   N=64:  64->32->16->8->4->2->1->1->1     (L2 1x1 dot product)
//   N=25:  25->13->7->4->2->1->1->1->1      (L3 5x5 lane MAC, unused)
// Widths grow by 1 bit per level (level L output width = W+L).
//
// NOTE (vendor neutrality, 2026-11-20): the input port is a FLAT
// packed vector `in_flat`, not an unpacked array port. Yosys 0.52's
// Verilog frontend rejects unpacked array ports ("syntax error,
// unexpected '['"), so the unpacked form cannot be synthesized by
// Yosys at all -- for any target, ECP5 included. The consumers pack
// their product array into `in_flat` (one assign per element, element
// i in bits [i*W +: W]); the values and the 4-cycle latency are
// unchanged. The flat vector is unpacked here with generate assigns
// (constant indices): a variable-base part-select `in_flat[(2*i)*W
// +: W]` inside the always @* level-1 block has incomplete @*
// sensitivity in Icarus and leaves stale sums (smoke test failed
// with 781 errors); the generate unpack plus the original `in[2*i]`
// references keep the proven level logic untouched. Internal level
// arrays stay unpacked (Yosys turns those into registers/wires);
// only the port was the blocker.
//=============================================================================
`timescale 1ns/1ps

module adder_tree #(
    parameter int N = 81,
    parameter int W = 25
)(
    input  logic clk,
    input  logic rst_n,
    input  logic stall,
    input  logic signed [N*W-1:0] in_flat,
    output logic signed [W+8-1:0] out
);
    // Level counts after each of the 8 levels (ceil divide by 2).
    localparam int C1 = (N + 1) / 2;
    localparam int C2 = (C1 + 1) / 2;
    localparam int C3 = (C2 + 1) / 2;
    localparam int C4 = (C3 + 1) / 2;
    localparam int C5 = (C4 + 1) / 2;
    localparam int C6 = (C5 + 1) / 2;
    localparam int C7 = (C6 + 1) / 2;
    localparam int C8 = (C7 + 1) / 2;

    // Combinational level outputs (level L width = W+L).
    // Unpack in_flat into the internal array (generate assigns, constant
    // indices -- see the header note on @* sensitivity).
    logic signed [W-1:0] in [0:N-1];
    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : unpack
            assign in[gi] = in_flat[gi*W +: W];
        end
    endgenerate
    logic signed [W+1-1:0] l1 [0:C1-1];
    logic signed [W+2-1:0] l2 [0:C2-1];
    logic signed [W+3-1:0] l3 [0:C3-1];
    logic signed [W+4-1:0] l4 [0:C4-1];
    logic signed [W+5-1:0] l5 [0:C5-1];
    logic signed [W+6-1:0] l6 [0:C6-1];
    logic signed [W+7-1:0] l7 [0:C7-1];
    logic signed [W+8-1:0] l8 [0:C8-1];
    // Stage registers (after levels 2, 4, 6, 8).
    logic signed [W+2-1:0] s2 [0:C2-1];
    logic signed [W+4-1:0] s4 [0:C4-1];
    logic signed [W+6-1:0] s6 [0:C6-1];

    // Module-level loop variables (Icarus sensitivity: never declare
    // loop variables inside always blocks).
    int i;

    // Levels 1-2 (stage 0): in -> l1 -> l2 -> s2.
    always @* begin
        for (i = 0; i < N / 2; i++)
            l1[i] = in[2*i] + in[2*i+1];
        if (N % 2 == 1)
            l1[N/2] = in[N-1];
        for (i = 0; i < C1 / 2; i++)
            l2[i] = l1[2*i] + l1[2*i+1];
        if (C1 % 2 == 1)
            l2[C1/2] = l1[C1-1];
    end
    // Levels 3-4 (stage 1): s2 -> l3 -> l4 -> s4.
    always @* begin
        for (i = 0; i < C2 / 2; i++)
            l3[i] = s2[2*i] + s2[2*i+1];
        if (C2 % 2 == 1)
            l3[C2/2] = s2[C2-1];
        for (i = 0; i < C3 / 2; i++)
            l4[i] = l3[2*i] + l3[2*i+1];
        if (C3 % 2 == 1)
            l4[C3/2] = l3[C3-1];
    end
    // Levels 5-6 (stage 2): s4 -> l5 -> l6 -> s6.
    always @* begin
        for (i = 0; i < C4 / 2; i++)
            l5[i] = s4[2*i] + s4[2*i+1];
        if (C4 % 2 == 1)
            l5[C4/2] = s4[C4-1];
        for (i = 0; i < C5 / 2; i++)
            l6[i] = l5[2*i] + l5[2*i+1];
        if (C5 % 2 == 1)
            l6[C5/2] = l5[C5-1];
    end
    // Levels 7-8 (stage 3): s6 -> l7 -> l8 -> out.
    always @* begin
        for (i = 0; i < C6 / 2; i++)
            l7[i] = s6[2*i] + s6[2*i+1];
        if (C6 % 2 == 1)
            l7[C6/2] = s6[C6-1];
        for (i = 0; i < C7 / 2; i++)
            l8[i] = l7[2*i] + l7[2*i+1];
        if (C7 % 2 == 1)
            l8[C7/2] = l7[C7-1];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < C2; i++) s2[i] <= '0;
            for (i = 0; i < C4; i++) s4[i] <= '0;
            for (i = 0; i < C6; i++) s6[i] <= '0;
            out <= '0;
        end else if (!stall) begin
            for (i = 0; i < C2; i++) s2[i] <= l2[i];
            for (i = 0; i < C4; i++) s4[i] <= l4[i];
            for (i = 0; i < C6; i++) s6[i] <= l6[i];
            out <= l8[0];
        end
    end
endmodule : adder_tree
