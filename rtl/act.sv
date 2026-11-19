//=============================================================================
// act.sv -- Stage 3: activation functions (post-processing chain),
// SystemVerilog.
//
// Modes (config 0x00): 0 = ReLU, 1 = Leaky ReLU / PReLU, 2 = PWL
// (piecewise-linear sigmoid / tanh / GELU / SiLU).
//   * ReLU: out = max(0, x). Zero cost.
//   * Leaky: out = x for x >= 0, else (x * slope_fp) >> 16, where slope_fp
//     is a 16-bit fixed-point slope (slope = slope_fp / 65536), loaded via
//     config 0x01. E.g. slope 0.125 -> slope_fp = 8192. One 32x16 multiply.
//   * PWL: input is a signed 16-bit Q5.11 value (in the low 16 bits of the
//     32-bit input). The function is approximated over [-8, 8] by 32
//     segments of width 0.5. Each segment stores y0 (function value at the
//     segment start, Q1.15) and dslope = y1 - y0 (Q1.15). Output =
//     y0 + ((dslope * dx) >> 10), where dx = x - x0 in Q5.11 units
//     (0..1023). Inputs outside [-8, 8] clamp to the endpoint segments.
//     The table (32 segments x y0/dslope) is loaded via the config bus
//     (0x100 + seg*2 = y0, 0x100 + seg*2+1 = dslope); the host selects the
//     function by loading the corresponding table (sigmoid, tanh, GELU or
//     SiLU). Output is signed 16-bit Q1.15, sign-extended to 32 bits.
//   * One value per cycle, 1-cycle latency (input reg -> output reg).
//     Framing (in_sof/in_eol/in_last) is passed through the pipeline.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//
// Config bus (write between frames; single-cycle writes; 16-bit address):
//   addr 0x00 : mode (0=ReLU, 1=Leaky, 2=PWL)
//   addr 0x01 : slope_fp (16-bit, Leaky only)
//   addr 0x100 + seg*2     : PWL y0[seg]   (signed 16-bit Q1.15)
//   addr 0x100 + seg*2 + 1 : PWL dslope[seg] (signed 16-bit Q1.15)
// Reset defaults: mode=0 (ReLU), slope_fp=8192 (0.125), PWL table = 0.
//=============================================================================
`timescale 1ns/1ps

module act (
    input  logic        clk,
    input  logic        rst_n,
    // config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (signed 32-bit values)
    input  logic signed [31:0] data_in,
    input  logic        in_valid,
    input  logic        in_sof,
    input  logic        in_eol,
    input  logic        in_last,
    output logic        in_ready,
    // stream out (signed 32-bit values)
    output logic signed [31:0] data_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [15:0] CFG_MODE  = 16'h0000;
    localparam logic [15:0] CFG_SLOPE = 16'h0001;
    localparam logic [15:0] CFG_PWL_BASE = 16'h0100;
    localparam logic [15:0] CFG_PWL_LAST = 16'h013F;  // 0x100 + 63

    //------------------------- config registers ----------------------------
    logic [2:0]  mode_r;
    logic signed [15:0] slope_fp_r;
    logic signed [15:0] pwl_y0    [0:31];
    logic signed [15:0] pwl_dslope [0:31];
    int ci;
    logic [5:0] pwl_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= 3'd0;
            slope_fp_r <= 16'sd8192;
            for (ci = 0; ci < 32; ci++) begin
                pwl_y0[ci] <= 16'sd0;
                pwl_dslope[ci] <= 16'sd0;
            end
        end else if (cfg_valid && cfg_we) begin
            if (cfg_addr == CFG_MODE) begin
                mode_r <= cfg_data[2:0];
            end else if (cfg_addr == CFG_SLOPE) begin
                slope_fp_r <= cfg_data;
            end else if (cfg_addr >= CFG_PWL_BASE && cfg_addr <= CFG_PWL_LAST) begin
                pwl_idx = (cfg_addr - CFG_PWL_BASE) >> 1;
                if (cfg_addr[0] == 1'b0)
                    pwl_y0[pwl_idx[4:0]] <= cfg_data;
                else
                    pwl_dslope[pwl_idx[4:0]] <= cfg_data;
            end
        end
    end

    //------------------------- pipeline ------------------------------------
    logic        stall;
    assign stall = out_valid && !out_ready;
    assign in_ready = !stall;
    assign busy = out_valid;

    // Input register + framing.
    logic signed [31:0] x_r;
    logic in_v_r, in_sof_r, in_eol_r, in_last_r;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_r <= 32'sd0;
            in_v_r <= 1'b0; in_sof_r <= 1'b0;
            in_eol_r <= 1'b0; in_last_r <= 1'b0;
        end else if (!stall) begin
            x_r <= data_in;
            in_v_r <= in_valid;
            in_sof_r <= in_sof;
            in_eol_r <= in_eol;
            in_last_r <= in_last;
        end
    end

    //---- ReLU / Leaky (32-bit path) ----
    logic signed [47:0] leaky_prod;
    logic signed [31:0] leaky_neg;
    assign leaky_prod = x_r * $signed({{16{slope_fp_r[15]}}, slope_fp_r});
    // (x * slope_fp) >> 16: take bits [47:16] of the 48-bit product.
    assign leaky_neg = leaky_prod[47:16];

    logic signed [31:0] relu_leaky_out;
    always @* begin
        if (mode_r == 3'd0) begin
            relu_leaky_out = (x_r < 32'sd0) ? 32'sd0 : x_r;
        end else begin
            relu_leaky_out = (x_r < 32'sd0) ? leaky_neg : x_r;
        end
    end

    //---- PWL (16-bit Q5.11 in, Q1.15 out) ----
    // Clamp x to [-8, 8] (Q5.11: -16384 .. 16384), then segment index =
    // (x_clamped + 16384) >> 10 (0..31), dx = (x_clamped + 16384) & 1023.
    logic signed [15:0] pwl_x;
    logic signed [15:0] pwl_xc;
    logic [4:0]  pwl_seg;
    logic [10:0] pwl_dx;  // 0..1024 (1024 only at the x==8.0 top edge)
    logic signed [15:0] pwl_y0v, pwl_dsv;
    logic signed [31:0] pwl_prod;     // dslope(16) * dx(10), in 32 bits
    logic signed [31:0] pwl_y;
    assign pwl_x = x_r[15:0];
    always @* begin
        if (pwl_x < -16'sd16384)      pwl_xc = -16'sd16384;
        else if (pwl_x > 16'sd16384)  pwl_xc = 16'sd16384;
        else                          pwl_xc = pwl_x;
    end
    // x_clamped + 16384 is in [0, 32768]; seg = >> 10 gives 0..32, clamp
    // the top edge (x == 8.0) to segment 31.
    logic [15:0] pwl_off;
    assign pwl_off = $unsigned(pwl_xc) + 16'd16384;
    assign pwl_seg = (pwl_off[15:10] >= 6'd32) ? 5'd31 : pwl_off[14:10];
    // Top edge (x == 8.0): dx = 1024 so y = y0[31] + dslope[31] = f(8),
    // matching the golden clamp. (dx = 1023 would fall short by dslope/1024.)
    assign pwl_dx  = (pwl_off[15:10] >= 6'd32) ? 11'd1024 : {1'b0, pwl_off[9:0]};
    assign pwl_y0v = pwl_y0[pwl_seg];
    assign pwl_dsv = pwl_dslope[pwl_seg];
    assign pwl_prod = $signed({{16{pwl_dsv[15]}}, pwl_dsv}) *
                      $signed({21'd0, pwl_dx});
    // y = y0 + ((dslope * dx) >> 10). The product fits in 26 bits
    // (|dslope| <= 32768, dx <= 1023), so bits [25:10] are the >> 10
    // result. The sum is kept at 32 bits (no 16-bit wrap): GELU/SiLU
    // exceed the Q1.15 range (|f| > 1 for |x| > ~1), and the golden
    // model returns the unwrapped sum.
    assign pwl_y = $signed({{16{pwl_y0v[15]}}, pwl_y0v}) +
                   $signed({{16{pwl_prod[25]}}, pwl_prod[25:10]});

    //---- output select ----
    logic signed [31:0] result;
    // modes 2..5 are all PWL (sigmoid/tanh/GELU/SiLU); the table loaded
    // via config selects which function. Modes 0/1 are ReLU/Leaky.
    always @* begin
        if (mode_r >= 3'd2)
            result = pwl_y;
        else
            result = relu_leaky_out;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 32'sd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_last <= 1'b0;
        end else if (!stall) begin
            data_out <= result;
            out_valid <= in_v_r;
            out_sof <= in_sof_r;
            out_eol <= in_eol_r;
            out_last <= in_last_r;
        end
    end
endmodule : act
