//=============================================================================
// pe_mac.sv -- one generic PE (Stage 7b, PE-array redesign).
//
// A GENERIC MAC array, NOT bound to the KxK window shape. The PE computes
// one GEMM tile-step: TM output positions x TN output channels over TK
// reduction steps (K-dim = C_in x K x K, implicit im2col). The scheduler
// decides the mode and the tile shape per layer/tile; the PE just executes.
//
// The PE is a PURE GEMM engine: it produces the raw 32-bit partial sums
// (accumulator file) for one tile-step. It does NOT chain K-chunks and
// does NOT apply the output stage. K-chunk partials for one
// (M-tile, N-chunk) may run on DIFFERENT PEs (the dispatcher assigns
// tile-steps round-robin); the pe_array's merge unit sums those
// partials and applies the output stage (shift/act/saturate) to the
// FINAL merged sum. Chaining K-chunks inside one PE (a `cont` flag)
// would require all chunks of one (M-tile, N-chunk) to land on the
// same PE, which the round-robin dispatcher does not guarantee --
// that is why the merge lives in the array, not the PE (see
// logs/stage7b/design.md, "Split-K merge").
//
// Parameters (rescale by parameter, not rewrite -- see logs/stage7b/design.md):
//   TM      = output positions per tile-step (16)
//   TN      = output channels per tile-step, outer mode (8)
//   TK_MAX  = max K per tile-step (32; register-file bound A = TM x TK,
//             B = TK x TN, both <= 512 entries)
//
// Modes (selected per tile-step by the scheduler):
//   outer-product (mode_dot = 0): rows = TM positions, cols = TN output
//     channels; one k per cycle; TM x TN products accumulate into the
//     TM x TN accumulator file. TK cycles per tile-step.
//   dot-product (mode_dot = 1): rows = TM positions, cols = 8 taps; one
//     output channel n per cycle; each row's 8 products reduce through a
//     combinational tree (7 adders) into one partial, which accumulates.
//     tn x (TK/8) cycles per tile-step. Used for N=1 layers (e.g. SRCNN
//     L3, 5x5 32->1), where outer-product would leave TN-1 lanes idle.
//     tk must be a multiple of 8 in dot mode (the scheduler guarantees it).
//     tn is the runtime output-channel count (1..TN).
//
// The multiplier grid (TM x TN = 128 products) is SHARED between the two
// modes; only the operand routing and the adder network differ. The adder
// network is implemented as two paths (outer: direct accumulate;
// dot: row tree + accumulate) selected by mode; sharing the adders
// between modes is a LUT optimization, not a correctness requirement.
//
// Operands: the PE takes the A and B tiles as FLAT packed vectors, held
// stable during the tile-step. The double-buffered operand register
// files live in pe_array (the dispatcher loads them); the PE is the
// pure compute datapath. Layout (k-major, contiguous per k):
//   a_flat: A[m][k] at byte (k*TM + m), i.e. bits [(k*TM+m)*8 +: 8].
//     A is 8-bit; sign_in = 1 treats it as signed, 0 as unsigned.
//   b_flat: B[k][n] at int16 (k*TN + n), i.e. bits [(k*TN+n)*16 +: 16].
//     B is always signed 16-bit.
//   bias_flat: bias[n] at int32 n, i.e. bits [n*32 +: 32] (signed).
//     Added once, when first_k = 1 (this tile-step is the first K-chunk
//     of the (M-tile, N-chunk)).
//
// Output: the raw 32-bit partial sums (accumulator file) as
// partial_flat, index (m*TN + n), bits [(m*TN+n)*32 +: 32], valid
// when done = 1 (one cycle). The golden (tb/golden_pe.py) compares
// partial_flat against a Python GEMM reference, max|err| = 0.
//
// Vendor neutrality: inferred multipliers only (the * operator), no
// Xilinx primitives, no BRAM (the operand files are in pe_array).
// All ports are flat packed vectors (Yosys 0.52 rejects unpacked
// array ports); internal arrays are unpacked (Yosys turns them into
// registers/wires). Operand selection uses continuous assigns in
// generate blocks (constant indices); no variable-base part-select
// inside an always @* block (Icarus @* sensitivity).
//=============================================================================

module pe_mac #(
    parameter int TM     = 16,
    parameter int TN     = 8,
    parameter int TK_MAX = 32
)(
    input  logic clk,
    input  logic rst_n,
    // Config (set before start, held during compute).
    input  logic        mode_dot,   // 0 = outer-product, 1 = dot-product
    input  logic [5:0]  tk,         // K per tile-step (1..TK_MAX)
    input  logic [5:0]  tn,         // output channels, dot mode (1..TN)
    input  logic        first_k,    // 1 = init accumulators from bias
    input  logic        sign_in,    // 1 = A values are signed
    input  logic signed [TN*32-1:0] bias_flat,
    // Operands (flat, held stable during compute).
    input  logic [TM*TK_MAX*8-1:0]  a_flat,
    input  logic [TK_MAX*TN*16-1:0] b_flat,
    // Control.
    input  logic start,             // pulse: begin one tile-step
    output logic busy,              // high during compute
    output logic done,              // pulse: tile-step complete
    output logic signed [TM*TN*32-1:0] partial_flat
);

    localparam int N_MULT = TM * TN;   // 128 for the default shape

    //--------------------------------------------------------------------------
    // State machine.
    //--------------------------------------------------------------------------
    typedef enum logic [1:0] { ST_IDLE, ST_RUN, ST_DONE } state_t;
    state_t state;

    // Outer mode: k = 0..tk-1 (one k per cycle).
    // Dot mode: n = 0..tn-1, g = 0..(tk/8)-1 (one (n,g) per cycle).
    logic [5:0] k_cnt;
    logic [5:0] n_cnt;
    logic [5:0] g_cnt;
    logic [5:0] tk_groups;   // tk/8, dot mode

    //--------------------------------------------------------------------------
    // Operand selection (continuous assigns in generate blocks; the
    // part-select base depends on the counter registers, which is fine
    // in a continuous assign -- assigns re-evaluate on any input change
    // and have no sensitivity list).
    //--------------------------------------------------------------------------
    // Outer mode: one k-column of A (TM bytes) and one k-row of B (TN int16).
    logic [7:0]  a_byte_o [0:TM-1];
    logic signed [8:0]  a_val [0:TM-1];
    logic signed [15:0] b_val [0:TN-1];
    // Dot mode: 8 tap-columns of A (TM x 8 bytes) and 8 tap-entries of B
    // (8 int16) for the current output channel n_cnt.
    logic [7:0]  a_byte_t [0:TM-1][0:7];
    logic signed [8:0]  a_tap [0:TM-1][0:7];
    logic signed [15:0] b_tap [0:7];

    genvar ga, gb, gt;
    generate
        for (ga = 0; ga < TM; ga++) begin : gen_a_outer
            assign a_byte_o[ga] = a_flat[(k_cnt*TM + ga)*8 +: 8];
            assign a_val[ga] = sign_in ?
                $signed({a_byte_o[ga][7], a_byte_o[ga]}) :
                $signed({1'b0, a_byte_o[ga]});
        end
        for (gb = 0; gb < TN; gb++) begin : gen_b_outer
            assign b_val[gb] = b_flat[(k_cnt*TN + gb)*16 +: 16];
        end
        for (ga = 0; ga < TM; ga++) begin : gen_a_dot
            for (gt = 0; gt < 8; gt++) begin : gen_a_dot_t
                assign a_byte_t[ga][gt] =
                    a_flat[((8*g_cnt + gt)*TM + ga)*8 +: 8];
                assign a_tap[ga][gt] = sign_in ?
                    $signed({a_byte_t[ga][gt][7], a_byte_t[ga][gt]}) :
                    $signed({1'b0, a_byte_t[ga][gt]});
            end
        end
        for (gt = 0; gt < 8; gt++) begin : gen_b_dot
            assign b_tap[gt] = b_flat[((8*g_cnt + gt)*TN + n_cnt)*16 +: 16];
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Products (combinational, shared multiplier grid).
    // Outer: prod[m*TN+n] = A[m][k] * B[k][n].
    // Dot:   prod[m*TN+t] = A[m][8g+t] * B[8g+t][n] (t = 0..7).
    //--------------------------------------------------------------------------
    logic signed [24:0] prod [0:N_MULT-1];
    genvar gm, gn;
    generate
        for (gm = 0; gm < TM; gm++) begin : gen_prod_m
            for (gn = 0; gn < TN; gn++) begin : gen_prod_n
                assign prod[gm*TN + gn] = mode_dot ?
                    (a_tap[gm][gn] * b_tap[gn]) :
                    (a_val[gm] * b_val[gn]);
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // 8-input reduction tree (dot mode): sums 8 signed 25-bit products
    // per row. 3 adder levels (26 / 27 / 28 bit), result widened to
    // 32-bit signed. Implemented with array-element reads and continuous
    // assigns (the adder_tree pattern, which Icarus handles
    // sign-correctly). A packed-vector + indexed part-select version
    // (tree8 function) is WRONG in Icarus: an indexed part-select of a
    // signed vector loses its signedness, so negative products become
    // large positives and the sum is off by a power of two (verified
    // 2026-11-20: tree8(-5,10,-3,7,0,0,0,0) gave -67108855, not 9).
    // (Placed after prod: Icarus requires declaration before use.)
    //--------------------------------------------------------------------------
    logic signed [25:0] rs_s0 [0:TM-1];
    logic signed [25:0] rs_s1 [0:TM-1];
    logic signed [25:0] rs_s2 [0:TM-1];
    logic signed [25:0] rs_s3 [0:TM-1];
    logic signed [26:0] rs_t0 [0:TM-1];
    logic signed [26:0] rs_t1 [0:TM-1];
    logic signed [27:0] rs_u0 [0:TM-1];
    logic signed [31:0] row_sum [0:TM-1];
    genvar gr;
    generate
        for (gr = 0; gr < TM; gr++) begin : gen_rowsum
            assign rs_s0[gr] = prod[gr*TN+0] + prod[gr*TN+1];
            assign rs_s1[gr] = prod[gr*TN+2] + prod[gr*TN+3];
            assign rs_s2[gr] = prod[gr*TN+4] + prod[gr*TN+5];
            assign rs_s3[gr] = prod[gr*TN+6] + prod[gr*TN+7];
            assign rs_t0[gr] = rs_s0[gr] + rs_s1[gr];
            assign rs_t1[gr] = rs_s2[gr] + rs_s3[gr];
            assign rs_u0[gr] = rs_t0[gr] + rs_t1[gr];
            assign row_sum[gr] = rs_u0[gr];
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Accumulator file (registered, TM x TN x 32-bit).
    //--------------------------------------------------------------------------
    logic signed [31:0] acc [0:N_MULT-1];

    // partial_flat is the accumulator file, packed (valid when done = 1).
    genvar gp;
    generate
        for (gp = 0; gp < N_MULT; gp++) begin : gen_partial
            assign partial_flat[gp*32 +: 32] = acc[gp];
        end
    endgenerate

    //--------------------------------------------------------------------------
    // State machine + accumulator update.
    // On start: init acc from bias (first_k) or 0; counters to 0.
    // Each RUN cycle: one k (outer) or one (n, g) (dot) is accumulated.
    // The last cycle's products are added at the edge that moves to DONE,
    // so partial_flat (combinational from acc) is valid when done = 1.
    //--------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            k_cnt     <= '0;
            n_cnt     <= '0;
            g_cnt     <= '0;
            tk_groups <= '0;
            busy      <= 1'b0;
            done      <= 1'b0;
            for (int i = 0; i < N_MULT; i++) acc[i] <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        tk_groups <= tk >> 3;
                        k_cnt <= '0;
                        n_cnt <= '0;
                        g_cnt <= '0;
                        // Init accumulators: bias (first K-chunk) or 0.
                        for (int m = 0; m < TM; m++)
                            for (int n = 0; n < TN; n++)
                                acc[m*TN + n] <= first_k ?
                                    bias_flat[n*32 +: 32] : 32'sd0;
                        state <= ST_RUN;
                        busy  <= 1'b1;
                    end
                end
                ST_RUN: begin
                    if (!mode_dot) begin
                        // Outer: accumulate one k-column into all acc.
                        for (int m = 0; m < TM; m++)
                            for (int n = 0; n < TN; n++)
                                acc[m*TN + n] <= acc[m*TN + n] + prod[m*TN + n];
                        if (k_cnt == tk - 1) begin
                            state <= ST_DONE;
                            done  <= 1'b1;
                            busy  <= 1'b0;
                        end else begin
                            k_cnt <= k_cnt + 1'b1;
                        end
                    end else begin
                        // Dot: accumulate one row-sum into acc[m][n_cnt].
                        for (int m = 0; m < TM; m++)
                            acc[m*TN + n_cnt] <= acc[m*TN + n_cnt] + row_sum[m];
                        if (g_cnt == tk_groups - 1) begin
                            g_cnt <= '0;
                            if (n_cnt == tn - 1) begin
                                state <= ST_DONE;
                                done  <= 1'b1;
                                busy  <= 1'b0;
                            end else begin
                                n_cnt <= n_cnt + 1'b1;
                            end
                        end else begin
                            g_cnt <= g_cnt + 1'b1;
                        end
                    end
                end
                ST_DONE: begin
                    // A new step may be assigned in the cycle the PE
                    // completes: busy drops with done, so the
                    // dispatcher sees the PE as free and may pulse
                    // start while the PE is still in ST_DONE. Accept
                    // it here (same init as ST_IDLE) instead of
                    // dropping it. ST_DONE used to ignore start, so
                    // a step dispatched in that cycle never ran
                    // (Stage 7d: the YOLO C49 layer lost 4 of 576
                    // steps -- 572 done pulses, data off by up to
                    // 24). The merge already captured the finished
                    // partial in the done cycle, so re-initialising
                    // the accumulators here is safe.
                    busy <= 1'b0;
                    if (start) begin
                        tk_groups <= tk >> 3;
                        k_cnt <= '0;
                        n_cnt <= '0;
                        g_cnt <= '0;
                        for (int m = 0; m < TM; m++)
                            for (int n = 0; n < TN; n++)
                                acc[m*TN + n] <= first_k ?
                                    bias_flat[n*32 +: 32] : 32'sd0;
                        state <= ST_RUN;
                        busy  <= 1'b1;
                    end else begin
                        state <= ST_IDLE;
                    end
                end
                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule : pe_mac
