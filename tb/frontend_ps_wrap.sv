// frontend_ps_wrap.sv -- post-synthesis wrapper for frontend
// (Stage 7c, Part 2).
//
// Problem: the golden TB (tb_frontend.sv) instantiates frontend
// with #(.MAX_W(3840)), but a Yosys-synthesized netlist has no
// parameters -- Icarus rejects a parameter override on a
// parameter-less module ("parameter `MAX_W` not found"). So the
// netlist cannot be dropped in place of the RTL.
//
// Solution: synthesize frontend once (default MAX_W=3840, which
// is the value the TB passes), rename the netlist module to
// frontend_net, and provide this wrapper, named frontend, which
// accepts the same parameter and passes every port through to the
// netlist. The wrapper adds no logic. The TB is compiled unchanged
// against it. (MAX_W does not appear in the port list, so the
// wrapper is correct for exactly the MAX_W the netlist was
// synthesized with -- 3840, the TB's value; any other value would
// be silently wrong, so the wrapper fails fast if MAX_W != 3840.)
//
// Usage (see tb/run_postsynth.sh):
//   yosys -p "read_verilog -sv rtl/vision/frontend.sv; synth -top frontend;
//     write_verilog ...", sed the module name to frontend_net, then
//   iverilog -g2012 -o tb_fe_ps.vvp tb/frontend_ps_wrap.sv \
//     frontend_net.v /usr/share/yosys/simcells.v tb/tb_frontend.sv
`timescale 1ns/1ps

module frontend #(
    parameter int MAX_W = 3840
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    input  logic [23:0] pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    output logic [23:0] pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    initial begin
        if (MAX_W != 3840) begin
            $display("FATAL: frontend_ps_wrap: netlist synthesized with MAX_W=3840, got %0d", MAX_W);
            $finish;
        end
    end

    frontend_net dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ready(out_ready), .busy(busy)
    );
endmodule
