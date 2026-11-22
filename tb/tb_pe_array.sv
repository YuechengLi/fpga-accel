// tb_pe_array.sv -- testbench for pe_array (Stage 7b).
// Reads one layer's tile-steps (written by sw/common/sched_pe.py
// write_layer_vectors), streams them through the pe_array, and collects
// the per-(M-tile, N-chunk) outputs. Writes the outputs to
// out_pe_array.txt as blocks: "mtile nchunk" then TM*TN bytes (one per
// line, (m, n) order). A Python driver assembles the layer output and
// compares it to the funcsim golden.
//
// The sender and the collector run concurrently (fork/join). The
// sender streams tile-steps as fast as the array accepts them
// (ts_ready backpressure); the collector waits for all expected
// outputs (n_out from the header). Within one layer all tile-steps
// have the same TK, so completions are staggered by the dispatch
// times and do not collide (see pe_array.sv).
`timescale 1ns/1ps

module tb_pe_array;
    parameter int N_PE = 4;
    parameter int TM = 16;
    parameter int TN = 8;
    parameter int TK_MAX = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        ts_valid = 1'b0;
    logic        ts_ready;
    logic        ts_mode_dot = 1'b0;
    logic [5:0]  ts_tk = 6'd0;
    logic [5:0]  ts_tn = 6'd0;
    logic        ts_first_k = 1'b0;
    logic        ts_sign_in = 1'b0;
    logic [3:0]  ts_shift = 4'd0;
    logic [1:0]  ts_act = 2'd0;
    logic [15:0] ts_slope = 16'd0;
    logic        ts_sign_out = 1'b0;
    logic signed [TN*32-1:0] ts_bias_flat = '0;
    logic [TM*TK_MAX*8-1:0]  ts_a_flat = '0;
    logic [TK_MAX*TN*16-1:0] ts_b_flat = '0;
    logic [15:0] ts_mtile = 16'd0;
    logic [7:0]  ts_nchunk = 8'd0;
    logic [5:0]  ts_n_kchunk = 6'd0;
    logic        out_valid;
    logic [15:0] out_mtile;
    logic [7:0]  out_nchunk;
    logic [TM*TN*8-1:0] out_bytes;
    logic [31:0] done_count;

    pe_array #(.N_PE(N_PE), .TM(TM), .TN(TN), .TK_MAX(TK_MAX)) dut (
        .clk(clk), .rst_n(rst_n),
        .ts_valid(ts_valid), .ts_ready(ts_ready),
        .ts_mode_dot(ts_mode_dot), .ts_tk(ts_tk), .ts_tn(ts_tn),
        .ts_first_k(ts_first_k),
        .ts_sign_in(ts_sign_in), .ts_shift(ts_shift), .ts_act(ts_act),
        .ts_slope(ts_slope), .ts_sign_out(ts_sign_out),
        .ts_bias_flat(ts_bias_flat),
        .ts_a_flat(ts_a_flat), .ts_b_flat(ts_b_flat),
        .ts_mtile(ts_mtile), .ts_nchunk(ts_nchunk),
        .ts_n_kchunk(ts_n_kchunk),
        .out_valid(out_valid), .out_mtile(out_mtile),
        .out_nchunk(out_nchunk), .out_bytes(out_bytes),
        .done_count(done_count), .out_count()
    );

    int fvec, fout, code, nsteps, n_out;
    int tm_p, tn_p, tkmax_p, h_out, w_out, c_out;
    int mode_v, tk_v, tn_v, firstk_v, sign_v;
    int shift_v, act_v, slope_v, signout_v, mtile_v, nchunk_v, nkchunk_v;
    int s, kk, m, n, tmp, got_out;

    // Send one tile-step (blocks until the array accepts it).
    // ts_ready is combinational (from pe_free) and changes at posedges
    // (when a PE is assigned), so it must be sampled at negedge
    // (mid-cycle, stable). Sampling it just after a posedge races with
    // the assignment: the step would be presented for an extra cycle
    // and dispatched TWICE (the merge then adds the same partial
    // twice and the sums saturate -- verified 2026-11-20).
    task send_step;
        begin
            @(negedge clk);
            ts_valid = 1'b1;
            while (!ts_ready) @(negedge clk);
            // ts_ready is high now; the next posedge accepts the step.
            @(posedge clk);
            @(negedge clk);
            ts_valid = 1'b0;
        end
    endtask

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_pe_array.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_pe_array.txt"); $finish; end
        fout = $fopen("out_pe_array.txt", "w");
        code = $fscanf(fvec, "%d %d %d %d", nsteps, tm_p, tn_p, tkmax_p);
        code = $fscanf(fvec, "%d %d %d %d", h_out, w_out, c_out, n_out);
        $display("PE ARRAY TB: %0d steps, out %0dx%0dx%0d, expect %0d outputs",
                 nsteps, h_out, w_out, c_out, n_out);

        fork
            begin : sender
                for (s = 0; s < nsteps; s++) begin
                    code = $fscanf(fvec, "%d %d %d %d %d %d %d %d %d %d %d %d",
                                   mode_v, tk_v, tn_v, firstk_v,
                                   sign_v, shift_v, act_v, slope_v,
                                   signout_v, mtile_v, nchunk_v, nkchunk_v);
                    ts_mode_dot = mode_v[0];
                    ts_tk       = tk_v[5:0];
                    ts_tn       = tn_v[5:0];
                    ts_first_k  = firstk_v[0];
                    ts_sign_in  = sign_v[0];
                    ts_shift    = shift_v[3:0];
                    ts_act      = act_v[1:0];
                    ts_slope    = slope_v[15:0];
                    ts_sign_out = signout_v[0];
                    ts_mtile    = mtile_v[15:0];
                    ts_nchunk   = nchunk_v[7:0];
                    ts_n_kchunk = nkchunk_v[5:0];
                    for (n = 0; n < TN; n++) begin
                        code = $fscanf(fvec, "%d", tmp);
                        ts_bias_flat[n*32 +: 32] = tmp;
                    end
                    for (kk = 0; kk < tk_v; kk++)
                        for (m = 0; m < TM; m++) begin
                            code = $fscanf(fvec, "%d", tmp);
                            ts_a_flat[(kk*TM + m)*8 +: 8] = tmp[7:0];
                        end
                    for (kk = 0; kk < tk_v; kk++)
                        for (n = 0; n < TN; n++) begin
                            code = $fscanf(fvec, "%d", tmp);
                            ts_b_flat[(kk*TN + n)*16 +: 16] = tmp[15:0];
                        end
                    send_step;
                end
                $display("SENDER DONE: %0d steps sent", nsteps);
            end
            begin : collector
                got_out = 0;
                while (got_out < n_out) begin
                    @(posedge clk);
                    if (out_valid) begin
                        $fwrite(fout, "%0d %0d\n", out_mtile, out_nchunk);
                        for (m = 0; m < TM; m++)
                            for (n = 0; n < TN; n++)
                                $fwrite(fout, "%0d\n",
                                        out_bytes[(m*TN + n)*8 +: 8]);
                        got_out = got_out + 1;
                    end
                end
                $display("COLLECTOR DONE: %0d outputs, done_count=%0d",
                         got_out, done_count);
            end
        join
        $display("PE ARRAY TB DONE");
        $fclose(fout);
        $fclose(fvec);
        $finish;
    end
endmodule
