//=============================================================================
// act.sv -- Stage 3: activation functions (post-processing chain),
// SystemVerilog. The activation stays a SEPARATE module (not folded
// into conv): simple activations (ReLU/Leaky) are cheap enough to FUSE
// into the conv output stage (see conv2d_mc, Stage 4), while complex
// activations (sigmoid/tanh/GELU/SiLU) go through this LUT module. The
// BYPASS mode (7) covers the fused and no-activation cases: the layer's
// output passes through this module unchanged when the activation was
// already applied in conv, or when there is no activation at all. The
// compiler (Stage 5) decides per layer: fuse ReLU/Leaky into conv, or
// route the complex activation through this module.
//
// Modes (config 0x00): 0 = ReLU, 1 = Leaky ReLU / PReLU, 2 = PWL
// (piecewise-linear sigmoid / tanh / GELU / SiLU), 6 = MULTI-LEVEL LUT
// (two-level table, compiler-generated, see below), 7 = BYPASS
// (output = input unchanged, framing passes through unchanged).
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
//   * MULTI-LEVEL LUT (mode 6, Stage 4): the 12-bit activation path.
//     A FLAT table cannot guarantee precision for steep functions without
//     becoming huge (a 12-bit input needs a 4096-entry flat table), so
//     the input is split into bit-fields and looked up in two levels:
//     Level 1 (coarse, indexed by the high H1 bits) gives a base value
//     and selects a fine table; Level 2 (fine, indexed by the next H2
//     bits) gives a delta. Output = base + delta (an ADDITION, not a
//     multiply), so the lookup is table reads + one add -- no multiply,
//     and it stays fast. The golden uses H1=8/H2=4 with 8 fine tables:
//     256 (coarse) + 8x16 (fine) = 384 entries, not 4096, with max
//     error vs float ~0.027 (comparable to the PWL's ~0.024); the
//     smaller H1=6/H2=6/4-band split gives max 0.117 (tanh) -- too
//     inaccurate to ship. Three levels are a straightforward extension
//     (the level-2 delta table becomes a two-level lookup); two levels
//     suffice for the four activations here. The hardware is fully
//     configurable: H1/H2 (the bit-split, H1 1..8, H2 1..6, H1+H2 <= 12),
//     the number of fine tables (up to 8), and all table contents are
//     loaded via the config bus. The COMPILER owns the levels, the
//     splits, and the lookup: it groups coarse segments by local slope
//     (steep regions get a fine table matched to their rapid variation,
//     flat regions a fine table matched to theirs -- a non-uniform split),
//     and it verifies the accuracy in the software functional simulator
//     + fixed-point golden (Stage 5 compiler verification flow). The
//     input is a 12-bit offset-binary value in the low 12 bits
//     (in12 = round((x+8)*256), x in [-8, 8)); the output is a 16-bit
//     signed value (Q5.11), sign-extended to 32 bits (same width as the
//     PWL output, wider range: GELU/SiLU reach ~8, which Q1.15 cannot
//     hold).
//     Config: 0x02 = MLUT control (bits [3:0] = H1, bits [7:5] = H2);
//     0x200 + idx = coarse base table (16-bit signed base, idx 0..255);
//     0x300 + idx = coarse select table (bits [2:0] = fine table id,
//     idx 0..255); 0x400 + t*64 + idx = fine table t (16-bit signed
//     delta, t 0..7, idx 0..63). All eight fine tables are read in
//     parallel (indexed by the fine bits) and the coarse-selected one is
//     muxed out, so the lookup fits in one combinational stage (same
//     1-cycle latency as the PWL path).
//   * BYPASS (mode 7, Stage 4): output = input unchanged, framing passes
//     through unchanged. Used when the activation is fused into the conv
//     module (ReLU/Leaky), or when the layer has no activation at all.
//   * One value per cycle, 1-cycle latency (input reg -> output reg).
//     Framing (in_sof/in_eol/in_last) is passed through the pipeline.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//
// Config bus (write between frames; single-cycle writes; 16-bit address):
//   addr 0x00 : mode (0=ReLU, 1=Leaky, 2=PWL, 6=MULTI-LEVEL LUT, 7=BYPASS)
//   addr 0x01 : slope_fp (16-bit, Leaky only)
//   addr 0x02 : MLUT control, bits [3:0] = H1 (coarse bits, 1..8),
//               bits [7:5] = H2 (fine bits, 1..6). H1+H2 <= 12.
//   addr 0x100 + seg*2     : PWL y0[seg]   (signed 16-bit Q1.15)
//   addr 0x100 + seg*2 + 1 : PWL dslope[seg] (signed 16-bit Q1.15)
//   addr 0x200 + idx       : MLUT coarse base[idx] (16-bit signed, idx 0..255)
//   addr 0x300 + idx       : MLUT coarse select[idx] (bits [2:0] =
//                           fine table id, idx 0..255)
//   addr 0x400 + t*64 + idx: MLUT fine table t delta[idx] (16-bit signed,
//                           t 0..7, idx 0..63)
// Reset defaults: mode=0 (ReLU), slope_fp=8192 (0.125), PWL table = 0,
//   H1=8, H2=4, MLUT tables = 0.
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
    localparam logic [15:0] CFG_MLUT_CTRL = 16'h0002;  // [3:0]=H1, [7:4]=H2
    localparam logic [15:0] CFG_PWL_BASE = 16'h0100;
    localparam logic [15:0] CFG_PWL_LAST = 16'h013F;  // 0x100 + 63
    localparam logic [15:0] CFG_MLUT_BASE = 16'h0200;  // coarse base, idx 0..255
    localparam logic [15:0] CFG_MLUT_BASE_LAST = 16'h02FF;
    localparam logic [15:0] CFG_MLUT_SEL = 16'h0300;   // coarse select, idx 0..255
    localparam logic [15:0] CFG_MLUT_SEL_LAST = 16'h03FF;
    localparam logic [15:0] CFG_MLUT_FINE_BASE = 16'h0400;  // fine t*64+idx
    localparam logic [15:0] CFG_MLUT_FINE_LAST = 16'h05FF;  // 0x400 + 511

    //------------------------- config registers ----------------------------
    logic [2:0]  mode_r;
    logic signed [15:0] slope_fp_r;
    logic signed [15:0] pwl_y0    [0:31];
    logic signed [15:0] pwl_dslope [0:31];
    // Multi-level LUT (mode 6). Coarse table up to 256 entries (H1<=8);
    // 8 fine tables x 64 entries (H2<=6); 3-bit fine-table selector.
    logic [3:0]  mlut_h1_r;   // coarse index bits, 1..8
    logic [2:0]  mlut_h2_r;   // fine index bits, 1..6 (H1+H2 <= 12)
    logic signed [15:0] mlut_base [0:255];  // coarse base table
    logic [2:0]  mlut_sel  [0:255];  // coarse select: fine table id 0..7
    logic signed [15:0] mlut_fine [0:7][0:63];  // 8 fine tables x 64 deltas
    int ci, cj;
    logic [5:0] pwl_idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= 3'd0;
            slope_fp_r <= 16'sd8192;
            for (ci = 0; ci < 32; ci++) begin
                pwl_y0[ci] <= 16'sd0;
                pwl_dslope[ci] <= 16'sd0;
            end
            mlut_h1_r <= 4'd8;
            mlut_h2_r <= 3'd4;
            for (ci = 0; ci < 256; ci++) begin
                mlut_base[ci] <= 16'sd0;
                mlut_sel[ci] <= 3'd0;
            end
            for (ci = 0; ci < 8; ci++)
                for (cj = 0; cj < 64; cj++) mlut_fine[ci][cj] <= 16'sd0;
        end else if (cfg_valid && cfg_we) begin
            if (cfg_addr == CFG_MODE) begin
                mode_r <= cfg_data[2:0];
            end else if (cfg_addr == CFG_SLOPE) begin
                slope_fp_r <= cfg_data;
            end else if (cfg_addr == CFG_MLUT_CTRL) begin
                mlut_h1_r <= cfg_data[3:0];
                mlut_h2_r <= cfg_data[7:5];
            end else if (cfg_addr >= CFG_PWL_BASE && cfg_addr <= CFG_PWL_LAST) begin
                pwl_idx = (cfg_addr - CFG_PWL_BASE) >> 1;
                if (cfg_addr[0] == 1'b0)
                    pwl_y0[pwl_idx[4:0]] <= cfg_data;
                else
                    pwl_dslope[pwl_idx[4:0]] <= cfg_data;
            end else if (cfg_addr >= CFG_MLUT_BASE &&
                         cfg_addr <= CFG_MLUT_BASE_LAST) begin
                mlut_base[cfg_addr[7:0]] <= cfg_data;
            end else if (cfg_addr >= CFG_MLUT_SEL &&
                         cfg_addr <= CFG_MLUT_SEL_LAST) begin
                mlut_sel[cfg_addr[7:0]] <= cfg_data[2:0];
            end else if (cfg_addr >= CFG_MLUT_FINE_BASE &&
                         cfg_addr <= CFG_MLUT_FINE_LAST) begin
                // addr 0x400 + t*64 + idx: t = addr[8:6], idx = addr[5:0].
                mlut_fine[cfg_addr[8:6]][cfg_addr[5:0]] <= cfg_data;
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

    //---- MULTI-LEVEL LUT (mode 6) ----
    // Input is a 12-bit signed value in the low 12 bits of x_r (Q4.8,
    // range [-8, 8)). Split: coarse index = top H1 bits, fine index =
    // the next H2 bits. All four fine tables are read in parallel
    // (indexed by the fine bits) and the coarse-selected one is muxed
    // out, so the whole lookup is one combinational stage (same 1-cycle
    // latency as the PWL path). Output = base + delta, saturating to
    // the 16-bit signed range, sign-extended to 32 bits.
    logic [11:0] mlut_in;
    logic [17:0] mlut_in_ext;
    logic [17:0] mlut_fine_shifted, mlut_fine_mask;
    logic [7:0]  mlut_coarse_idx;
    logic [5:0]  mlut_fine_idx;
    logic signed [15:0] mlut_base_v;
    logic [2:0]  mlut_fsel;
    logic signed [15:0] mlut_d0, mlut_d1, mlut_d2, mlut_d3;
    logic signed [15:0] mlut_d4, mlut_d5, mlut_d6, mlut_d7;
    logic signed [15:0] mlut_delta;
    logic signed [16:0] mlut_sum;
    logic signed [15:0] mlut_y;
    assign mlut_in = x_r[11:0];
    assign mlut_in_ext = {6'd0, mlut_in};
    // Variable shifts (H1/H2 are config, constant during a layer):
    // coarse = top H1 bits, fine = the next H2 bits. The 18-bit
    // intermediates avoid the 6-bit-mask wrap bug when H2 = 6:
    // (in >> (12 - H1 - H2)) is in [0, 63] and the mask
    // ((1 << H2) - 1) would be 0 at H2 = 6 in 6-bit arithmetic.
    assign mlut_coarse_idx = mlut_in_ext >> (12 - {2'b0, mlut_h1_r});
    assign mlut_fine_shifted = mlut_in_ext >> (12 - {2'b0, mlut_h1_r}
                                                   - {1'b0, mlut_h2_r});
    assign mlut_fine_mask = (18'd1 << mlut_h2_r) - 18'd1;
    assign mlut_fine_idx = mlut_fine_shifted & mlut_fine_mask;
    assign mlut_base_v = mlut_base[mlut_coarse_idx];
    assign mlut_fsel = mlut_sel[mlut_coarse_idx];
    assign mlut_d0 = mlut_fine[0][mlut_fine_idx];
    assign mlut_d1 = mlut_fine[1][mlut_fine_idx];
    assign mlut_d2 = mlut_fine[2][mlut_fine_idx];
    assign mlut_d3 = mlut_fine[3][mlut_fine_idx];
    assign mlut_d4 = mlut_fine[4][mlut_fine_idx];
    assign mlut_d5 = mlut_fine[5][mlut_fine_idx];
    assign mlut_d6 = mlut_fine[6][mlut_fine_idx];
    assign mlut_d7 = mlut_fine[7][mlut_fine_idx];
    always @* begin
        case (mlut_fsel)
            3'd0: mlut_delta = mlut_d0;
            3'd1: mlut_delta = mlut_d1;
            3'd2: mlut_delta = mlut_d2;
            3'd3: mlut_delta = mlut_d3;
            3'd4: mlut_delta = mlut_d4;
            3'd5: mlut_delta = mlut_d5;
            3'd6: mlut_delta = mlut_d6;
            default: mlut_delta = mlut_d7;
        endcase
    end
    assign mlut_sum = {mlut_base_v[15], mlut_base_v} +
                      {mlut_delta[15], mlut_delta};
    always @* begin
        if (mlut_sum > 17'sd32767)       mlut_y = 16'sd32767;
        else if (mlut_sum < -17'sd32768) mlut_y = -16'sd32768;
        else                             mlut_y = mlut_sum[15:0];
    end

    //---- output select ----
    logic signed [31:0] result;
    // Mode 7 is BYPASS (output = input unchanged; used when the
    // activation is fused into the conv module, or when the layer has
    // no activation at all). Mode 6 is MULTI-LEVEL LUT; modes 2..5 are
    // PWL (sigmoid/tanh/GELU/SiLU); the table loaded via config selects
    // which function. Modes 0/1 are ReLU/Leaky.
    always @* begin
        if (mode_r == 3'd7)
            result = x_r;
        else if (mode_r == 3'd6)
            result = {{16{mlut_y[15]}}, mlut_y};
        else if (mode_r >= 3'd2)
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
