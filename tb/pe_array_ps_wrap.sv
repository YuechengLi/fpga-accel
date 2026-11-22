// pe_array_ps_wrap.sv -- post-synthesis wrapper for pe_array
// (Stage 7c, Part 2, Mac Phase 3).
//
// Same parameter problem as the VM wrappers (see
// tb/adder_tree_ps_wrap.sv): tb_pe_array.sv instantiates pe_array
// with #(.N_PE(4), .TM(16), .TN(8), .TK_MAX(32)) (N_MERGE takes
// its default 5), but a Yosys-synthesized netlist has no
// parameters -- Icarus rejects the override. This wrapper
// accepts the same parameters and passes every port through to
// the renamed netlist (pe_array_net, synthesized with N_PE=4 /
// TM=16 / TN=8 / TK_MAX=32 / N_MERGE=5, the TB's config). It
// adds no logic. The TB -- and the chain driver
// (sw/vision/run_pe_srcnn.py --netlist) -- are compiled unchanged.
// A different parameter set fails fast (it would silently use
// the wrong netlist).
//
// Usage (Mac Phase 3, see run_ecp5_mac.sh):
//   yosys -p "read_verilog -sv rtl/common/pe_array.sv rtl/common/pe_mac.sv
//     rtl/common/pe_emit.sv rtl/common/adder_tree.sv; chparam -set N_PE 4
//     -set TM 16 -set TN 8 pe_array; synth -top pe_array;
//     write_verilog logs/stage7c/pe_array_net.v"
//   sed the module name to pe_array_net, then
//   python3 sw/vision/run_pe_srcnn.py --netlist logs/stage7c/pe_array_net.v
`timescale 1ns/1ps

module pe_array #(
    parameter int N_PE    = 4,
    parameter int TM      = 16,
    parameter int TN      = 8,
    parameter int TK_MAX  = 32,
    parameter int N_MERGE = 5
)(
    input  logic clk,
    input  logic rst_n,
    input  logic        ts_valid,
    output logic        ts_ready,
    input  logic        ts_mode_dot,
    input  logic [5:0]  ts_tk,
    input  logic [5:0]  ts_tn,
    input  logic        ts_first_k,
    input  logic        ts_sign_in,
    input  logic [3:0]  ts_shift,
    input  logic [1:0]  ts_act,
    input  logic [15:0] ts_slope,
    input  logic        ts_sign_out,
    input  logic signed [TN*32-1:0] ts_bias_flat,
    input  logic [TM*TK_MAX*8-1:0]  ts_a_flat,
    input  logic [TK_MAX*TN*16-1:0] ts_b_flat,
    input  logic [15:0] ts_mtile,
    input  logic [7:0]  ts_nchunk,
    input  logic [5:0]  ts_n_kchunk,
    output logic        out_valid,
    output logic [15:0] out_mtile,
    output logic [7:0]  out_nchunk,
    output logic [TM*TN*8-1:0] out_bytes,
    output logic [31:0] done_count,
    output logic [31:0] out_count
);
    initial begin
        if (N_PE != 4 || TM != 16 || TN != 8 || TK_MAX != 32
                || N_MERGE != 5) begin
            $display("FATAL: pe_array_ps_wrap: netlist synthesized with N_PE=4 TM=16 TN=8 TK_MAX=32 N_MERGE=5, got N_PE=%0d TM=%0d TN=%0d TK_MAX=%0d N_MERGE=%0d", N_PE, TM, TN, TK_MAX, N_MERGE);
            $finish;
        end
    end

    pe_array_net dut (
        .clk(clk), .rst_n(rst_n),
        .ts_valid(ts_valid), .ts_ready(ts_ready),
        .ts_mode_dot(ts_mode_dot), .ts_tk(ts_tk), .ts_tn(ts_tn),
        .ts_first_k(ts_first_k), .ts_sign_in(ts_sign_in),
        .ts_shift(ts_shift), .ts_act(ts_act), .ts_slope(ts_slope),
        .ts_sign_out(ts_sign_out),
        .ts_bias_flat(ts_bias_flat),
        .ts_a_flat(ts_a_flat), .ts_b_flat(ts_b_flat),
        .ts_mtile(ts_mtile), .ts_nchunk(ts_nchunk),
        .ts_n_kchunk(ts_n_kchunk),
        .out_valid(out_valid), .out_mtile(out_mtile),
        .out_nchunk(out_nchunk), .out_bytes(out_bytes),
        .done_count(done_count), .out_count(out_count)
    );
endmodule
