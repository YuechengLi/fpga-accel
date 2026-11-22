// tb_e2e.sv -- integrated end-to-end testbench (Stage 7c).
//
// The sender/collector below are the tb_pe_array.sv pattern
// verbatim (same vectors format, same handshake, same output
// file), with the DUT swapped from the bare pe_array to the
// srcnn_pipe_pe wrapper -- the Stage 7b top, which had no
// testbench until now. The vectors file is written by
// sw/sched_pe.py::write_layer_vectors (header: nsteps TM TN
// TK_MAX / h_out w_out c_out n_out; per step: 12 header ints,
// TN 32-bit biases, tk*TM int8 A values, tk*TN int16 B values).
//
// Added coverage over tb_pe_array.sv: the wrapper's config bus
// is exercised -- the TB reads back N_PE / TM / TN (0x00-0x02)
// before the run and done_count (0x04) after it, and fails if
// any readback is wrong. This is the only place the config bus
// is verified end-to-end.
//
// Usage: iverilog -g2012 -o tb_e2e.vvp ../rtl/srcnn_pipe_pe.sv
//   ../rtl/pe_array.sv ../rtl/pe_mac.sv ../rtl/pe_emit.sv tb_e2e.sv
//   && vvp tb_e2e.vvp   (cwd = tb/)
`timescale 1ns/1ps

module tb_e2e;
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

    // Config bus (wrapper only).
    logic        cfg_valid = 1'b0;
    logic        cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'h0;
    logic [15:0] cfg_data = 16'h0;
    logic [15:0] cfg_rdata;
    logic        cfg_rvalid;

    srcnn_pipe_pe #(.N_PE(N_PE), .TM(TM), .TN(TN), .TK_MAX(TK_MAX)) dut (
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
        .done_count(done_count),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .cfg_rdata(cfg_rdata), .cfg_rvalid(cfg_rvalid)
    );

    int fvec, fout, code, nsteps, n_out;
    int tm_p, tn_p, tkmax_p, h_out, w_out, c_out;
    int mode_v, tk_v, tn_v, firstk_v, sign_v;
    int shift_v, act_v, slope_v, signout_v, mtile_v, nchunk_v, nkchunk_v;
    int s, kk, m, n, tmp, got_out, errors, done_read;

    // Config readback: present the request before a clock edge;
    // the wrapper registers rvalid+rdata on that edge, so sample
    // at the following negedge (rvalid high, rdata stable).
    task cfg_read(input [15:0] addr, output [15:0] data);
        begin
            @(negedge clk);
            cfg_valid = 1'b1; cfg_we = 1'b0;
            cfg_addr = addr; cfg_data = 16'h0;
            @(posedge clk);
            @(negedge clk);
            if (!cfg_rvalid) begin
                $display("ERROR: cfg read 0x%0h: rvalid low", addr);
                errors = errors + 1;
            end
            data = cfg_rdata;
            cfg_valid = 1'b0;
        end
    endtask

    // Send one tile-step (blocks until the array accepts it).
    // ts_ready is combinational (from pe_free) and changes at
    // posedges (when a PE is assigned), so it must be sampled at
    // negedge (mid-cycle, stable). Sampling it just after a posedge
    // races with the assignment: the step would be presented for an
    // extra cycle and dispatched TWICE (verified 2026-11-20).
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
        errors = 0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ---- config readback (wrapper-only coverage) ----
        cfg_read(16'h00, tmp);
        if (tmp != N_PE) begin
            $display("ERROR: cfg N_PE readback %0d, expected %0d",
                     tmp, N_PE);
            errors = errors + 1;
        end
        cfg_read(16'h01, tmp);
        if (tmp != TM) begin
            $display("ERROR: cfg TM readback %0d, expected %0d", tmp, TM);
            errors = errors + 1;
        end
        cfg_read(16'h02, tmp);
        if (tmp != TN) begin
            $display("ERROR: cfg TN readback %0d, expected %0d", tmp, TN);
            errors = errors + 1;
        end
        if (errors == 0)
            $display("WRAPPER CONFIG CHECK PASSED (N_PE=%0d TM=%0d TN=%0d)",
                     N_PE, TM, TN);

        fvec = $fopen("vectors_pe_array.txt", "r");
        if (fvec == 0) begin
            $display("FATAL: cannot open vectors_pe_array.txt");
            $finish;
        end
        fout = $fopen("out_pe_array.txt", "w");
        if (fout == 0) begin
            $display("FATAL: cannot open out_pe_array.txt");
            $finish;
        end
        code = $fscanf(fvec, "%d %d %d %d", nsteps, tm_p, tn_p, tkmax_p);
        code = $fscanf(fvec, "%d %d %d %d", h_out, w_out, c_out, n_out);
        $display("E2E TB: %0d steps, out %0dx%0dx%0d, expect %0d outputs",
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

        // ---- done_count readback via the config bus ----
        // done_count counts completed TILE-STEPS (one per
        // dispatched step), not completed outputs: for this layer
        // nsteps steps produce n_out outputs (several K-chunks
        // merge into one output block).
        cfg_read(16'h04, done_read);
        if (done_read != nsteps) begin
            $display("ERROR: cfg done_count readback %0d, expected %0d",
                     done_read, nsteps);
            errors = errors + 1;
        end else begin
            $display("WRAPPER CONFIG CHECK PASSED (done_count=%0d)",
                     done_read);
        end
        if (done_count != nsteps) begin
            $display("ERROR: done_count port %0d, expected %0d",
                     done_count, nsteps);
            errors = errors + 1;
        end
        if (got_out != n_out) begin
            $display("ERROR: collected %0d outputs, expected %0d",
                     got_out, n_out);
            errors = errors + 1;
        end
        if (errors != 0) begin
            $display("E2E FAIL: %0d errors", errors);
        end else begin
            $display("E2E PASS: %0d outputs collected", got_out);
        end
        $fclose(fout);
        $fclose(fvec);
        $finish;
    end

    // Watchdog: 20M cycles is far beyond any layer (DnCNN ~35k
    // steps x ~10 cycles/step); a stall is a hang, not a slow run.
    initial begin
        #200_000_000;
        $display("ERROR: watchdog timeout");
        $finish;
    end

endmodule
