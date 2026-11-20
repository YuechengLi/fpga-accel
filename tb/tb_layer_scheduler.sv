//=============================================================================
// tb_layer_scheduler.sv -- Stage 4 testbench for layer_scheduler.sv +
// unified_engine.sv (integrated).
//
// Reads vectors_sched.txt (golden_sched.py gen): 2 runs x 3 layers.
//   run 0 (on-time): layer 2's weights are loaded into bank A while
//     layer 1 runs (fork/join with the layer-1 frame stream), so the
//     scheduler switches immediately (no ST_WAIT_W stall).
//   run 1 (late): layer 2's weights are loaded only AFTER layer 1
//     finishes, so the scheduler must wait in ST_WAIT_W and still
//     produce correct outputs.
// The TB programs the scheduler (layer descriptors + weights), starts
// it, and on each eng_start pulse streams that layer's input frame
// into the engine. Engine outputs are collected with random
// out_ready backpressure and written to out_sched.txt. No reset
// between the two runs (the scheduler is reset via its own config
// bit1 between runs).
//=============================================================================
`timescale 1ns/1ps

module tb_layer_scheduler;
    logic clk = 0;
    logic rst_n = 0;
    // scheduler config bus (driven by the TB)
    logic scfg_valid = 0, scfg_we = 0;
    logic [15:0] scfg_addr = 0, scfg_data = 0;
    // scheduler <-> engine
    logic eng_cfg_valid, eng_cfg_we;
    logic [15:0] eng_cfg_addr, eng_cfg_data;
    logic eng_busy, eng_start;
    logic sched_done, sched_busy, sched_wfull;
    // engine stream (driven by the TB)
    logic [7:0] pixel_in = 0;
    logic in_valid = 0, in_sof = 0, in_ready;
    logic [7:0] pixel_out;
    logic out_valid, out_sof, out_eol, out_last, out_ready = 0;
    logic eng_busy_out;

    layer_scheduler #(.NLAYER(4), .BANK_COEFF(289)) sched (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(scfg_valid), .cfg_we(scfg_we),
        .cfg_addr(scfg_addr), .cfg_data(scfg_data),
        .eng_cfg_valid(eng_cfg_valid), .eng_cfg_we(eng_cfg_we),
        .eng_cfg_addr(eng_cfg_addr), .eng_cfg_data(eng_cfg_data),
        .eng_busy(eng_busy), .eng_start(eng_start),
        .done(sched_done), .busy(sched_busy), .wfull(sched_wfull)
    );

    unified_engine eng (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(eng_cfg_valid), .cfg_we(eng_cfg_we),
        .cfg_addr(eng_cfg_addr), .cfg_data(eng_cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_ready(out_ready),
        .busy(eng_busy)
    );

    always #5 clk = ~clk;

    integer vf, of, rc, i, j, run, li, late_flag, nruns;
    // Count eng_start pulses (1-cycle) so the TB never misses one.
    // A plain `wait (eng_start === 1'b1)` can miss the pulse if it
    // fires while the TB is between waits (e.g. inside load_bank_n).
    integer start_pulse_cnt;
    initial start_pulse_cnt = 0;
    always @(posedge clk) if (eng_start) start_pulse_cnt = start_pulse_cnt + 1;
    // Wait for the NEXT eng_start pulse after this call.
    task wait_start;
        integer c0;
        begin
            c0 = start_pulse_cnt;
            wait (start_pulse_cnt > c0);
        end
    endtask
    integer tmp;
    integer mode, k, shift, w, h, pt, pb, pl, pr, sx, sy;
    integer sign_in, sign_out, fuse_act, act_slope, d, ncoeff;
    integer coeff [0:2][0:288];
    integer img [0:2][0:4095];
    integer ww [0:2], hh [0:2];
    integer ow [0:2], oh [0:2];
    integer exp_count [0:2];
    // Per-layer ncoeff (read from the vector file).
    integer ncoeff_arr [0:2];
    integer out_count, framing_err;
    integer seed = 4242;
    integer accepted;
    integer out_idx;
    integer start_cycle, l2_start_cycle, l1_end_cycle;
    integer cycle_cnt = 0;

    always @(posedge clk) cycle_cnt <= cycle_cnt + 1;

    task scfg_write(input [15:0] a, input [15:0] d);
        begin
            @(posedge clk); #1;
            scfg_valid = 1; scfg_we = 1; scfg_addr = a; scfg_data = d;
            @(posedge clk); #1;
            scfg_valid = 0; scfg_we = 0;
        end
    endtask

    // Load a layer's coefficients into a bank (0 = A at 0x0200+i,
    // 1 = B at 0x0400+i), using the vector's ncoeff for that layer.
    task load_bank_n(input integer bank, input integer layer);
        integer n;
        begin
            for (n = 0; n < ncoeff_arr[layer]; n = n + 1) begin
                scfg_write(bank == 0 ? 16'h0200 + n[15:0] :
                                        16'h0400 + n[15:0],
                           coeff[layer][n][15:0]);
            end
        end
    endtask

    // Stream one layer's input frame into the engine.
    task stream_frame(input integer layer);
        integer n, npix;
        begin
            npix = ww[layer] * hh[layer];
            for (n = 0; n < npix; n = n + 1) begin
                while (($random(seed) % 100) < 30) begin
                    @(posedge clk); #1; in_valid = 0;
                end
                @(posedge clk); #1;
                pixel_in = img[layer][n][7:0];
                in_valid = 1;
                in_sof = (n == 0);
                accepted = 0;
                while (!accepted) begin
                    @(posedge clk);
                    if (in_ready) accepted = 1;
                    #1;
                end
                in_valid = 0; in_sof = 0;
            end
        end
    endtask

    // Collect outputs in the background.
    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            $fwrite(of, "%0d\n", pixel_out);
            out_idx = out_idx + 1;
            out_count = out_count + 1;
        end
    end

    always @(posedge clk) begin
        out_ready <= ($random(seed) % 100) < 80;
    end

    initial begin
        vf = $fopen("vectors_sched.txt", "r");
        if (vf == 0) begin $display("cannot open vectors_sched.txt"); $finish; end
        of = $fopen("out_sched.txt", "w");
        rc = $fscanf(vf, "%d", nruns);
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (run = 0; run < nruns; run = run + 1) begin
            rc = $fscanf(vf, "%d", late_flag);
            // Read the 3 layers.
            for (li = 0; li < 3; li = li + 1) begin
                rc = $fscanf(vf, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                             mode, k, shift, w, h, pt, pb, pl, pr, sx, sy,
                             sign_in, sign_out, fuse_act, act_slope, d,
                             tmp);
                ncoeff_arr[li] = tmp;
                for (i = 0; i < ncoeff_arr[li]; i = i + 1) begin
                    rc = $fscanf(vf, "%d", tmp);
                    coeff[li][i] = tmp;
                end
                for (i = 0; i < w * h; i = i + 1) begin
                    rc = $fscanf(vf, "%d", tmp);
                    img[li][i] = tmp;
                end
                ww[li] = w; hh[li] = h;
                // Output size.
                if (mode == 1) begin
                    // DW square 5x5: span k.
                    ow[li] = (w + pl + pr - k) / sx + 1;
                    oh[li] = (h + pt + pb - k) / sy + 1;
                end else begin
                    ow[li] = (w + pl + pr - k) / sx + 1;
                    oh[li] = (h + pt + pb - k) / sy + 1;
                end
                exp_count[li] = ow[li] * oh[li];
                // Program this layer's descriptor into the scheduler.
                scfg_write(16'h0002, li[15:0]);
                scfg_write(16'h0003, mode[15:0]);
                scfg_write(16'h0004, k[15:0]);
                scfg_write(16'h0005, shift[15:0]);
                scfg_write(16'h0006, w[15:0]);
                scfg_write(16'h0007, h[15:0]);
                scfg_write(16'h0008, pt[15:0]);
                scfg_write(16'h0009, pb[15:0]);
                scfg_write(16'h000A, pl[15:0]);
                scfg_write(16'h000B, pr[15:0]);
                scfg_write(16'h000C, sx[15:0]);
                scfg_write(16'h000D, sy[15:0]);
                scfg_write(16'h000E, ((sign_out << 1) | sign_in));
                scfg_write(16'h000F, fuse_act[15:0]);
                scfg_write(16'h0010, act_slope[15:0]);
                scfg_write(16'h0011, d[15:0]);
                scfg_write(16'h0012, ncoeff_arr[li][15:0]);
            end
            scfg_write(16'h0001, 16'd3);  // 3 layers
            // Load layer 0 -> bank A, layer 1 -> bank B (before start).
            // (load_bank uses ncoeff_arr; patch ncoeff_of via a local.)
            load_bank_n(0, 0);
            load_bank_n(1, 1);
            out_count = 0; out_idx = 0; framing_err = 0;
            start_cycle = cycle_cnt;
            // Start the sequence.
            scfg_write(16'h0000, 16'h0001);
            // Layer 0: wait for eng_start, stream its frame.
            wait_start;
            @(posedge clk); #1;
            stream_frame(0);
            // Layer 1: wait for eng_start. While layer 1 runs, load
            // layer 2's weights into bank A -- on-time for run 0
            // (fork/join with the frame stream), late for run 1
            // (load only after layer 1's frame is fully streamed AND
            // the engine has gone idle, i.e. after the scheduler has
            // already entered ST_WAIT_W).
            wait_start;
            @(posedge clk); #1;
            if (late_flag == 0) begin
                fork
                    stream_frame(1);
                    load_bank_n(0, 2);
                join
            end else begin
                stream_frame(1);
                // Wait until the scheduler is actually in ST_WAIT_W
                // (state 4, waiting for the next layer's weights)
                // before loading them. (Waiting only for eng_busy==0
                // races the scheduler's ST_WAIT_W entry: the TB can
                // load the weights and the scheduler can pulse
                // eng_start before the TB starts waiting for it.)
                wait (sched.state === 3'd4);
                l1_end_cycle = cycle_cnt;
                load_bank_n(0, 2);
            end
            // Layer 2: wait for eng_start, stream its frame.
            wait_start;
            @(posedge clk); #1;
            l2_start_cycle = cycle_cnt;
            stream_frame(2);
            // Wait for the scheduler to report done.
            begin : wait_done
                integer t;
                t = 0;
                while (!sched_done && t < 500000) begin
                    @(posedge clk); t = t + 1;
                end
                if (!sched_done)
                    $display("ERROR run %0d: scheduler done timeout", run);
            end
            // Wait for all outputs.
            begin : wait_out
                integer t, total;
                total = exp_count[0] + exp_count[1] + exp_count[2];
                t = 0;
                while (out_count < total && t < 500000) begin
                    @(posedge clk); t = t + 1;
                end
                if (out_count < total)
                    $display("ERROR run %0d: timeout, got %0d/%0d outputs",
                             run, out_count, total);
            end
            $display("run %0d (late=%0d): %0d outputs, sched done, layer2 eng_start at cycle %0d (start %0d)",
                     run, late_flag, out_count, l2_start_cycle, start_cycle);
            // Reset the scheduler for the next run (its own config).
            scfg_write(16'h0000, 16'h0002);
            repeat (4) @(posedge clk);
        end
        $display("TB layer_scheduler DONE");
        $fclose(of);
        $finish;
    end
endmodule
