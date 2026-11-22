// pe_mac_ps_wrap.sv -- post-synthesis wrapper for pe_mac
// (Stage 7c, Part 2, Mac Phase 3).
//
// Same parameter problem as the VM wrappers (see
// tb/adder_tree_ps_wrap.sv): tb_pe_mac.sv instantiates pe_mac
// with #(.TM(16), .TN(8), .TK_MAX(32)), but a Yosys-synthesized
// netlist has no parameters -- Icarus rejects the override.
// This wrapper accepts the same parameters and passes every
// port through to the renamed netlist (pe_mac_net, synthesized
// with TM=16/TN=8/TK_MAX=32, the TB's config -- the ZU19EG
// config, not the ECP5 TN=4 config). It adds no logic. The TB
// is compiled unchanged. A different parameter set fails fast
// (it would silently use the wrong netlist).
//
// Usage (Mac Phase 3, see run_ecp5_mac.sh):
//   yosys -p "read_verilog -sv rtl/pe_mac.sv rtl/adder_tree.sv;
//     chparam -set TM 16 -set TN 8 pe_mac; synth -top pe_mac;
//     write_verilog logs/stage7c/pe_mac_net.v"
//   sed the module name to pe_mac_net, then
//   iverilog -g2012 -o tb_pe.vvp pe_mac_ps_wrap.sv \
//     ../logs/stage7c/pe_mac_net.v <simcells.v> tb_pe_mac.sv
`timescale 1ns/1ps

module pe_mac #(
    parameter int TM     = 16,
    parameter int TN     = 8,
    parameter int TK_MAX = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic        mode_dot,
    input  logic [5:0]  tk,
    input  logic [5:0]  tn,
    input  logic        first_k,
    input  logic        sign_in,
    input  logic signed [TN*32-1:0] bias_flat,
    input  logic [TM*TK_MAX*8-1:0]  a_flat,
    input  logic [TK_MAX*TN*16-1:0] b_flat,
    input  logic start,
    output logic busy,
    output logic done,
    output logic signed [TM*TN*32-1:0] partial_flat
);
    initial begin
        if (TM != 16 || TN != 8 || TK_MAX != 32) begin
            $display("FATAL: pe_mac_ps_wrap: netlist synthesized with TM=16 TN=8 TK_MAX=32, got TM=%0d TN=%0d TK_MAX=%0d", TM, TN, TK_MAX);
            $finish;
        end
    end

    pe_mac_net dut (
        .clk(clk), .rst_n(rst_n),
        .mode_dot(mode_dot), .tk(tk), .tn(tn),
        .first_k(first_k), .sign_in(sign_in),
        .bias_flat(bias_flat),
        .a_flat(a_flat), .b_flat(b_flat),
        .start(start), .busy(busy), .done(done),
        .partial_flat(partial_flat)
    );
endmodule
