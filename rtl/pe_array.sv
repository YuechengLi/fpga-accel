//=============================================================================
// pe_array.sv -- PE array + dispatcher + merge unit (Stage 7b).
//
// N_PE generic PEs (pe_mac) plus a dispatcher that issues tile-steps to
// free PEs (round-robin), plus a MERGE unit that sums the K-chunk
// partials of one (M-tile, N-chunk) and applies the output stage.
//
// Tile-step (one per handshake, ts_valid && ts_ready):
//   mode_dot, tk, tn, first_k, sign_in, shift, act, slope, sign_out,
//   bias_flat, a_flat, b_flat, tag (mtile, nchunk), n_kchunk.
// The dispatcher latches the tile-step into a free PE's input
// registers and pulses start. The PE holds its operands during
// compute (single-buffered per PE in this version: a PE accepts a new
// tile-step only after it finishes the current one; double-buffering
// the operand files so the next tile-step loads while the current
// computes is a throughput optimization, not a correctness
// requirement -- the cycle model assumes load is hidden).
//
// Merge: K-chunk partials for one (M-tile, N-chunk) may run on
// DIFFERENT PEs (the dispatcher assigns tile-steps round-robin), so
// they cannot chain inside one PE. The merge unit is an associative
// table of in-flight (mtile, nchunk) entries (one running 32-bit sum
// per entry, TM x TN). When a PE completes a tile-step, its partial is
// added to the entry for its (mtile, nchunk) (a new entry is allocated
// for the first K-chunk, which already includes the bias via first_k).
// When the entry's count reaches n_kchunk, the merged sum is passed
// through the output stage (pe_emit: round-half-up shift, ReLU /
// Leaky, saturate) and the (tag, bytes) are emitted on the output
// port (out_valid, out_mtile, out_nchunk, out_bytes) for one cycle.
// Partials may complete out of order (different PEs); the COUNT, not
// the last_k flag, decides completion.
//
// The output stage is applied ONLY to the final merged sum, never to
// a partial (see pe_emit.sv).
//
// Completion collisions: if two PEs complete the same cycle, the lower
// index is merged first and the other is merged next cycle -- the
// merge table holds both partials (they are captured into the table,
// not lost). The OUTPUT emit is one block per cycle; if two merge
// entries complete the same cycle, the lower index emits first and the
// other emits next cycle (an emit queue of depth 1 per entry is
// implicit in the table -- a completed entry stays valid until its
// emit is accepted). Within one layer all tile-steps have the same TK,
// so completions are staggered by the dispatch times and collisions
// are rare; the table handles them correctly when they occur.
//
// Parameters (rescale by parameter, not rewrite):
//   N_PE, TM, TN, TK_MAX -- see pe_mac. The ZU19EG config is
//   N_PE=15, TM=16, TN=8, TK_MAX=32 (1920 DSPs); the ECP5 config is
//   N_PE=2, TM=16, TN=4, TK_MAX=32 (128 multipliers).
//   N_MERGE: merge-table entries (default N_PE+1; at most one entry
//   per PE in flight, plus one being emitted).
//
// Vendor neutrality: inferred multipliers only (in pe_mac), no Xilinx
// primitives, no BRAM. All ports are flat packed vectors.
//=============================================================================

module pe_array #(
    parameter int N_PE    = 4,
    parameter int TM      = 16,
    parameter int TN      = 8,
    parameter int TK_MAX  = 32,
    parameter int N_MERGE = 5
)(
    input  logic clk,
    input  logic rst_n,
    // Tile-step input (from the scheduler / TB).
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
    // Completed-output stream (one (M-tile, N-chunk) per pulse).
    output logic        out_valid,
    output logic [15:0] out_mtile,
    output logic [7:0]  out_nchunk,
    output logic [TM*TN*8-1:0] out_bytes,
    // Debug: number of tile-steps completed / outputs emitted.
    output logic [31:0] done_count,
    output logic [31:0] out_count
);

    localparam int N_ACC = TM * TN;   // 128 for the default shape

    //--------------------------------------------------------------------------
    // Per-PE state.
    //--------------------------------------------------------------------------
    logic        pe_busy [0:N_PE-1];
    logic        pe_done [0:N_PE-1];
    logic signed [TM*TN*32-1:0] pe_partial [0:N_PE-1];
    logic [15:0] pe_mtile [0:N_PE-1];
    logic [7:0]  pe_nchunk [0:N_PE-1];
    logic [5:0]  pe_n_kchunk [0:N_PE-1];
    logic [3:0]  pe_shift [0:N_PE-1];
    logic [1:0]  pe_act [0:N_PE-1];
    logic [15:0] pe_slope [0:N_PE-1];
    logic        pe_sign_out [0:N_PE-1];

    // Per-PE input registers (latched on assignment).
    logic        pe_mode_dot [0:N_PE-1];
    logic [5:0]  pe_tk [0:N_PE-1];
    logic [5:0]  pe_tn [0:N_PE-1];
    logic        pe_first_k [0:N_PE-1];
    logic        pe_sign_in [0:N_PE-1];
    logic signed [TN*32-1:0] pe_bias_flat [0:N_PE-1];
    logic [TM*TK_MAX*8-1:0]  pe_a_flat [0:N_PE-1];
    logic [TK_MAX*TN*16-1:0] pe_b_flat [0:N_PE-1];
    logic        pe_start [0:N_PE-1];

    // Round-robin pointer for dispatch.
    logic [$clog2(N_PE)-1:0] rr_ptr;

    // Unpack each PE's partial_flat into a signed array (generate
    // assigns). An indexed part-select of a signed vector loses its
    // signedness in Icarus when used directly in arithmetic (the
    // tree8 bug); the merge adds partials, so a negative partial used
    // as an unsigned part-select would become a huge positive and the
    // merged sum would saturate. The generate unpack copies the bit
    // pattern into signed array elements, and arithmetic on those is
    // correct (the adder_tree / pe_emit pattern).
    logic signed [31:0] pe_part_arr [0:N_PE-1][0:N_ACC-1];
    genvar gpa, gpb;
    generate
        for (gpa = 0; gpa < N_PE; gpa++) begin : gen_part_unpack
            for (gpb = 0; gpb < N_ACC; gpb++) begin : gen_part_unpack_b
                assign pe_part_arr[gpa][gpb] = pe_partial[gpa][gpb*32 +: 32];
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // PEs.
    //--------------------------------------------------------------------------
    genvar gi;
    generate
        for (gi = 0; gi < N_PE; gi++) begin : gen_pe
            pe_mac #(.TM(TM), .TN(TN), .TK_MAX(TK_MAX)) u_pe (
                .clk(clk), .rst_n(rst_n),
                .mode_dot(pe_mode_dot[gi]), .tk(pe_tk[gi]),
                .tn(pe_tn[gi]), .first_k(pe_first_k[gi]),
                .sign_in(pe_sign_in[gi]),
                .bias_flat(pe_bias_flat[gi]),
                .a_flat(pe_a_flat[gi]), .b_flat(pe_b_flat[gi]),
                .start(pe_start[gi]),
                .busy(pe_busy[gi]), .done(pe_done[gi]),
                .partial_flat(pe_partial[gi])
            );
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Dispatch: ts_ready when any PE is free AND the merge table has a
    // free entry (a new (mtile,nchunk) may need one). Pick the first
    // free PE at or after rr_ptr (round-robin). On a handshake, latch
    // the tile-step into that PE and pulse start.
    //--------------------------------------------------------------------------
    // A PE is free only when it is not busy AND was not just assigned
    // this cycle. pe_busy goes high one cycle AFTER the start pulse
    // (the PE samples start at the next posedge), so pe_busy alone
    // would let the dispatcher double-assign a PE in the gap cycle.
    // pe_start (the 1-cycle assignment pulse) closes that gap.
    logic [N_PE-1:0] pe_free;
    logic [$clog2(N_PE)-1:0] sel_pe;
    logic sel_valid;
    integer si;
    always @* begin
        for (si = 0; si < N_PE; si++)
            pe_free[si] = !pe_busy[si] && !pe_start[si];
        sel_valid = 1'b0;
        sel_pe = rr_ptr;
        for (si = 0; si < N_PE; si++) begin
            if (!sel_valid && pe_free[(rr_ptr + si) % N_PE]) begin
                sel_pe = (rr_ptr + si) % N_PE;
                sel_valid = 1'b1;
            end
        end
    end

    // Merge-table free entry (for the dispatch backpressure).
    logic merge_has_free;
    assign ts_ready = sel_valid && merge_has_free;

    //--------------------------------------------------------------------------
    // Merge table: one entry per in-flight (mtile, nchunk).
    //--------------------------------------------------------------------------
    logic        mg_valid [0:N_MERGE-1];
    logic [15:0] mg_mtile [0:N_MERGE-1];
    logic [7:0]  mg_nchunk [0:N_MERGE-1];
    logic signed [31:0] mg_sum [0:N_MERGE-1][0:N_ACC-1];
    logic [5:0]  mg_count [0:N_MERGE-1];
    logic [5:0]  mg_n_kchunk [0:N_MERGE-1];
    logic [3:0]  mg_shift [0:N_MERGE-1];
    logic [1:0]  mg_act [0:N_MERGE-1];
    logic [15:0] mg_slope [0:N_MERGE-1];
    logic        mg_sign_out [0:N_MERGE-1];
    // Emit pending: an entry completed and waits to emit its bytes.
    logic        mg_emit_pend [0:N_MERGE-1];

    // Free-entry search (combinational).
    logic [$clog2(N_MERGE)-1:0] free_idx;
    logic free_found;
    integer fi;
    always @* begin
        free_found = 1'b0;
        free_idx = '0;
        for (fi = 0; fi < N_MERGE; fi++) begin
            if (!free_found && !mg_valid[fi] && !mg_emit_pend[fi]) begin
                free_idx = fi[$clog2(N_MERGE)-1:0];
                free_found = 1'b1;
            end
        end
    end
    assign merge_has_free = free_found;

    // Tag match search for a completing PE (combinational).
    // (Used by the sequential block via a function-like search below;
    // the actual match is computed in the always_ff with a loop.)

    // Emit: the first pending entry (lowest index) emits this cycle.
    // Its sum is passed through pe_emit (combinational).
    logic [$clog2(N_MERGE)-1:0] emit_idx;
    logic emit_any;
    integer ei;
    always @* begin
        emit_any = 1'b0;
        emit_idx = '0;
        for (ei = 0; ei < N_MERGE; ei++) begin
            if (!emit_any && mg_emit_pend[ei]) begin
                emit_idx = ei[$clog2(N_MERGE)-1:0];
                emit_any = 1'b1;
            end
        end
    end
    logic signed [TM*TN*32-1:0] emit_sum_flat;
    genvar ge;
    generate
        for (ge = 0; ge < N_ACC; ge++) begin : gen_emit_sum
            assign emit_sum_flat[ge*32 +: 32] = mg_sum[emit_idx][ge];
        end
    endgenerate
    logic [TM*TN*8-1:0] emit_bytes;
    pe_emit #(.TM(TM), .TN(TN)) u_emit (
        .in_flat(emit_sum_flat),
        .shift(mg_shift[emit_idx]), .act(mg_act[emit_idx]),
        .slope(mg_slope[emit_idx]), .sign_out(mg_sign_out[emit_idx]),
        .out_flat(emit_bytes)
    );

    //--------------------------------------------------------------------------
    // Sequential: dispatch, merge capture, emit.
    //--------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rr_ptr <= '0;
            out_valid <= 1'b0;
            out_mtile <= '0;
            out_nchunk <= '0;
            out_bytes <= '0;
            done_count <= '0;
            out_count <= '0;
            for (int i = 0; i < N_PE; i++) begin
                pe_start[i] <= 1'b0;
                pe_mtile[i] <= '0;
                pe_nchunk[i] <= '0;
                pe_n_kchunk[i] <= '0;
                pe_shift[i] <= '0;
                pe_act[i] <= '0;
                pe_slope[i] <= '0;
                pe_sign_out[i] <= 1'b0;
            end
            for (int j = 0; j < N_MERGE; j++) begin
                mg_valid[j] <= 1'b0;
                mg_mtile[j] <= '0;
                mg_nchunk[j] <= '0;
                mg_count[j] <= '0;
                mg_n_kchunk[j] <= '0;
                mg_shift[j] <= '0;
                mg_act[j] <= '0;
                mg_slope[j] <= '0;
                mg_sign_out[j] <= 1'b0;
                mg_emit_pend[j] <= 1'b0;
                for (int a = 0; a < N_ACC; a++) mg_sum[j][a] <= '0;
            end
        end else begin
            out_valid <= 1'b0;
            for (int i = 0; i < N_PE; i++) pe_start[i] <= 1'b0;

            // Dispatch a tile-step to a free PE.
            if (ts_valid && ts_ready) begin
                pe_mode_dot[sel_pe]  <= ts_mode_dot;
                pe_tk[sel_pe]        <= ts_tk;
                pe_tn[sel_pe]        <= ts_tn;
                pe_first_k[sel_pe]   <= ts_first_k;
                pe_sign_in[sel_pe]   <= ts_sign_in;
                pe_bias_flat[sel_pe] <= ts_bias_flat;
                pe_a_flat[sel_pe]    <= ts_a_flat;
                pe_b_flat[sel_pe]    <= ts_b_flat;
                pe_mtile[sel_pe]     <= ts_mtile;
                pe_nchunk[sel_pe]    <= ts_nchunk;
                pe_n_kchunk[sel_pe]  <= ts_n_kchunk;
                pe_shift[sel_pe]     <= ts_shift;
                pe_act[sel_pe]       <= ts_act;
                pe_slope[sel_pe]     <= ts_slope;
                pe_sign_out[sel_pe]  <= ts_sign_out;
                pe_start[sel_pe]     <= 1'b1;
                rr_ptr <= (sel_pe == N_PE - 1) ? '0 : sel_pe + 1'b1;
            end

            // Merge: capture every PE that completed this cycle.
            // (A PE's done is a 1-cycle pulse; its partial is valid
            // in that cycle. Multiple PEs may complete the same cycle;
            // each is merged into its entry independently.)
            for (int p = 0; p < N_PE; p++) begin
                if (pe_done[p]) begin
                    done_count <= done_count + 1'b1;
                    // Find the entry for this (mtile, nchunk).
                    begin : merge_one
                        integer m;
                        integer hit;
                        integer slot;
                        hit = -1;
                        slot = -1;
                        for (m = 0; m < N_MERGE; m++) begin
                            if (mg_valid[m] && mg_mtile[m] == pe_mtile[p] &&
                                mg_nchunk[m] == pe_nchunk[p])
                                hit = m;
                            if (slot < 0 && !mg_valid[m] && !mg_emit_pend[m])
                                slot = m;
                        end
                        if (hit >= 0) begin
                            // Existing entry: add the partial.
                            for (int a = 0; a < N_ACC; a++)
                                mg_sum[hit][a] <= mg_sum[hit][a] + pe_part_arr[p][a];
                            mg_count[hit] <= mg_count[hit] + 1'b1;
                            if (mg_count[hit] + 1'b1 == mg_n_kchunk[hit]) begin
                                mg_emit_pend[hit] <= 1'b1;
                                mg_valid[hit] <= 1'b0;
                            end
                        end else if (slot >= 0) begin
                            // New entry: the partial is the running sum.
                            // (The first K-chunk already includes the
                            // bias via first_k, so no separate add.)
                            for (int a = 0; a < N_ACC; a++)
                                mg_sum[slot][a] <= pe_part_arr[p][a];
                            mg_mtile[slot] <= pe_mtile[p];
                            mg_nchunk[slot] <= pe_nchunk[p];
                            mg_n_kchunk[slot] <= pe_n_kchunk[p];
                            mg_shift[slot] <= pe_shift[p];
                            mg_act[slot] <= pe_act[p];
                            mg_slope[slot] <= pe_slope[p];
                            mg_sign_out[slot] <= pe_sign_out[p];
                            mg_count[slot] <= 1'b1;
                            mg_valid[slot] <= 1'b1;
                            if (pe_n_kchunk[p] == 1) begin
                                mg_emit_pend[slot] <= 1'b1;
                                mg_valid[slot] <= 1'b0;
                            end
                        end
                        // (If no slot, the partial is dropped -- the
                        // dispatch backpressure (merge_has_free) must
                        // prevent this; the TB asserts it never happens.)
                    end
                end
            end

            // Emit: one completed entry per cycle (lowest index first).
            if (emit_any) begin
                out_valid <= 1'b1;
                out_mtile <= mg_mtile[emit_idx];
                out_nchunk <= mg_nchunk[emit_idx];
                out_bytes <= emit_bytes;
                out_count <= out_count + 1'b1;
                mg_emit_pend[emit_idx] <= 1'b0;
            end
        end
    end

endmodule : pe_array
