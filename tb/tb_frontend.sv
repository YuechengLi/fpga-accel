// tb_frontend.sv -- SystemVerilog testbench for frontend (Stage 7).
// Reads vectors_frontend.txt (written by golden_frontend.py gen), streams
// each case as 24-bit groups (one spatial position per cycle, 3 x 8-bit
// channels; for 1ch only byte 0 carries data) with random in_valid gaps
// and random out_ready stalls, writes one output value per line to
// out_frontend.txt (position-major, channel-minor), checks out_sof /
// out_eol / out_last framing against the golden output size, and measures
// the first-output latency (cycles from first input accept to first
// output valid).
// Run:
//   iverilog -g2012 -o tb_fe.vvp ../rtl/vision/frontend.sv tb_frontend.sv && vvp tb_fe.vvp
`timescale 1ns/1ps

module tb_frontend;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [7:0]  cfg_addr = 8'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [23:0] pixel_in = '0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [23:0] pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;
    logic        busy;

    frontend #(.MAX_W(3840)) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ready(out_ready_r), .busy(busy)
    );

    // random out_ready: ~80% ready, exercises backpressure
    always_ff @(posedge clk) begin
        if (rst_n)
            out_ready_r <= ((($random & 32'h7fffffff) % 5) != 0);
    end

    // cycle counter (increments on negedge, so it is stable at posedge and
    // free of read/write races with the sender/collector sampling it).
    longint unsigned cycle_cnt = 0;
    always @(negedge clk) begin
        if (rst_n) cycle_cnt = cycle_cnt + 1;
    end
    longint unsigned t_first_in = 0, t_first_out = 0;
    logic first_in_done = 1'b0, first_out_done = 1'b0;

    task cfg_wr(input logic [7:0] a, input logic [15:0] d);
        begin
            @(negedge clk);
            cfg_valid = 1'b1; cfg_we = 1'b1; cfg_addr = a; cfg_data = d;
            @(negedge clk);
            cfg_valid = 1'b0; cfg_we = 1'b0;
        end
    endtask

    // send one 24-bit group, holding it until the DUT accepts it, then a
    // random 0..2 cycle gap. Records the cycle of the first input accept.
    task send_pixel(input logic [23:0] px, input logic sof);
        int gap;
        begin
            @(negedge clk);
            pixel_in = px; in_valid = 1'b1; in_sof = sof;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            if (!first_in_done) begin
                first_in_done = 1'b1;
                t_first_in = cycle_cnt;
            end
            @(negedge clk);
            pixel_in = '0; in_valid = 1'b0; in_sof = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    int fvec, fout, code, ncases, ci;
    int w, h, k, sh, pt, pb, pl, pr, sx, sy, ds, ch, sign, act, slope;
    int sign_out;
    int pw, ph, ow, oh, esx, esy;
    int i, p, c;
    int coeff_v [0:24];
    int img [0:131071];
    int n_out, tb_err;
    int cycles;
    logic [23:0] grp;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_frontend.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_frontend.txt"); $finish; end
        fout = $fopen("out_frontend.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB frontend: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                           w, h, k, sh, pt, pb, pl, pr, sx, sy, ds, ch,
                           sign, act, slope);
            for (i = 0; i < 25; i++)
                code = $fscanf(fvec, "%d", coeff_v[i]);
            for (i = 0; i < w*h*ch; i++)
                code = $fscanf(fvec, "%d", img[i]);

            pw = w + pl + pr;
            ph = h + pt + pb;
            esx = sx * ds;
            esy = sy * ds;
            ow = (pw - k) / esx + 1;
            oh = (ph - k) / esy + 1;
            sign_out = (sign >> 1) & 1;

            cfg_wr(8'h00, k[15:0]);
            cfg_wr(8'h01, sh[15:0]);
            cfg_wr(8'h02, w[15:0]);
            cfg_wr(8'h03, h[15:0]);
            cfg_wr(8'h04, pt[15:0]);
            cfg_wr(8'h05, pb[15:0]);
            cfg_wr(8'h06, pl[15:0]);
            cfg_wr(8'h07, pr[15:0]);
            cfg_wr(8'h08, sx[15:0]);
            cfg_wr(8'h09, sy[15:0]);
            cfg_wr(8'h0A, act[15:0]);
            cfg_wr(8'h0B, slope[15:0]);
            cfg_wr(8'h0C, ds[15:0]);
            cfg_wr(8'h0D, ch[15:0]);
            cfg_wr(8'h0F, sign[15:0]);
            for (i = 0; i < 25; i++)
                cfg_wr(8'h10 + i[7:0], coeff_v[i][15:0]);

            n_out = 0; tb_err = 0; cycles = 0;
            first_in_done = 1'b0; first_out_done = 1'b0;
            t_first_in = 0; t_first_out = 0;
            fork
                begin : sender
                    for (p = 0; p < w*h; p++) begin
                        grp = '0;
                        for (c = 0; c < ch; c++)
                            grp[8*c +: 8] = img[p*ch + c][7:0];
                        send_pixel(grp, (p == 0));
                    end
                end
                begin : collector
                    while (n_out < ow*oh) begin
                        @(posedge clk);
                        cycles++;
                        if (cycles > 3000000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                                     ci, n_out, ow*oh);
                            $finish;
                        end
                        if (out_valid && out_ready_r) begin
                            if (!first_out_done) begin
                                first_out_done = 1'b1;
                                t_first_out = cycle_cnt;
                            end
                            // signed output: write the byte as int8 so the
                            // golden (which expects -128..127) matches
                            for (c = 0; c < ch; c++) begin
                                if (sign_out)
                                    $fwrite(fout, "%0d\n", $signed(pixel_out[8*c +: 8]));
                                else
                                    $fwrite(fout, "%0d\n", pixel_out[8*c +: 8]);
                            end
                            if (n_out == 0 && !out_sof) begin
                                $display("ERROR case %0d: first output missing out_sof", ci);
                                tb_err++;
                            end
                            if ((n_out % ow == ow-1) && !out_eol) begin
                                $display("ERROR case %0d: output %0d missing out_eol", ci, n_out);
                                tb_err++;
                            end
                            if (n_out == ow*oh-1 && !out_last) begin
                                $display("ERROR case %0d: last output missing out_last", ci);
                                tb_err++;
                            end
                            n_out++;
                        end
                    end
                end
            join
            $display("case %0d (%0dx%0d ch=%0d k=%0d shift=%0d pad=(%0d,%0d,%0d,%0d) stride=(%0d,%0d) ds=%0d sign=%0d act=%0d out=%0dx%0d): %0d outputs, framing errors=%0d, first-out latency=%0d cycles",
                     ci, w, h, ch, k, sh, pt, pb, pl, pr, sx, sy, ds, sign, act, ow, oh, n_out, tb_err, t_first_out - t_first_in);
            // Wait for the DUT to finish the frame before rewriting config.
            // The stream continues past the last output to (ph-1, pw-1) --
            // up to ~(eff_sy-1) rows + (eff_sx-1) cols of padding positions
            // -- and a config write that shrinks pw/ph mid-stream corrupts
            // the stream counters (strm_c never wraps, the frame never
            // ends, the next case starts with a stale frame_active and
            // wrong decimation counters). A fixed 20-cycle wait is NOT
            // enough for high stride/DS cases (case 12 needs ~105 cycles).
            // busy = !can_start = frame_active || !pipe_empty, so waiting
            // for !busy waits for the stream to finish AND the pipeline to
            // drain. (Found as cases 8/13/14 failing only when run after
            // earlier cases; each passes in isolation.)
            begin : wait_idle
                int idle_cycles;
                idle_cycles = 0;
                while (busy) begin
                    @(posedge clk);
                    idle_cycles++;
                    if (idle_cycles > 100000) begin
                        $display("ERROR case %0d: timeout waiting for idle (busy stuck)", ci);
                        $finish;
                    end
                end
            end
            repeat (5) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB frontend DONE");
        $finish;
    end
endmodule
