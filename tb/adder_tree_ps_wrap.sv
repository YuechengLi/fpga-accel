// adder_tree_ps_wrap.sv -- post-synthesis wrapper for adder_tree
// (Stage 7c, Part 2).
//
// Problem: the smoke TB (tb_adder_tree.sv) instantiates adder_tree
// four times with different parameters (#(.N(160)/81/64/25, .W(25))),
// but a Yosys-synthesized netlist has no parameters -- Icarus
// rejects a parameter override on a parameter-less module
// ("parameter `N` not found"). So the netlist cannot be dropped in
// place of the RTL.
//
// Solution: synthesize adder_tree once per N (four netlists,
// renamed adder_tree_n160 / _n81 / _n64 / _n25 by sed on the
// module declaration) and provide this wrapper, named adder_tree,
// which accepts the same parameters and selects the matching
// netlist with a generate-if. The wrapper adds no logic -- it is a
// pure port passthrough -- and the TB is compiled unchanged
// against it. An N value with no netlist falls through to a
// $display + $finish (fail fast, never a silent wrong netlist).
//
// Usage (see tb/run_postsynth.sh):
//   yosys -p "read_verilog -sv rtl/adder_tree.sv; chparam -set N 160
//     adder_tree; synth -top adder_tree; write_verilog ..."
//   (once per N), sed the module names, then
//   iverilog -g2012 -o tb_tree_ps.vvp tb/adder_tree_ps_wrap.sv \
//     adder_tree_n160.v adder_tree_n81.v adder_tree_n64.v \
//     adder_tree_n25.v /usr/share/yosys/simcells.v tb/tb_adder_tree.sv
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
    generate
        if (N == 160 && W == 25) begin : g160
            adder_tree_n160 dut (
                .clk(clk), .rst_n(rst_n), .stall(stall),
                .in_flat(in_flat), .out(out));
        end else if (N == 81 && W == 25) begin : g81
            adder_tree_n81 dut (
                .clk(clk), .rst_n(rst_n), .stall(stall),
                .in_flat(in_flat), .out(out));
        end else if (N == 64 && W == 25) begin : g64
            adder_tree_n64 dut (
                .clk(clk), .rst_n(rst_n), .stall(stall),
                .in_flat(in_flat), .out(out));
        end else if (N == 25 && W == 25) begin : g25
            adder_tree_n25 dut (
                .clk(clk), .rst_n(rst_n), .stall(stall),
                .in_flat(in_flat), .out(out));
        end else begin : g_bad
            initial begin
                $display("FATAL: adder_tree_ps_wrap: no netlist for N=%0d W=%0d", N, W);
                $finish;
            end
        end
    endgenerate
endmodule
