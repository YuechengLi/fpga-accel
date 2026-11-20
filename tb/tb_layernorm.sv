//=============================================================================
// tb_layernorm.sv -- Stage 4 testbench for layernorm.sv.
//
// Reads vectors_layernorm.txt (written by golden_layernorm.py gen):
//   header: W H sign_in sign_out isqrt_shift eps gamma_q beta nvals
//   then nvals input pixels, then 256 LUT lines (one integer each).
// Writes config (W, H, sign, isqrt_shift, eps, gamma, beta, 256 LUT
// entries), sends the frame with random in_valid gaps, collects the
// output with random out_ready backpressure, checks framing (sof/eol/
// last, output count) and writes out_layernorm.txt.
// No reset between cases (the DUT returns to ST_LOAD after a frame).
//=============================================================================
`timescale 1ns/1ps

module tb_layernorm;
    logic clk = 0;
    logic rst_n = 0;
    logic cfg_valid = 0, cfg_we = 0;
    logic [15:0] cfg_addr = 0, cfg_data = 0;
    logic [7:0] pixel_in = 0;
    logic in_valid = 0, in_sof = 0, in_ready;
    logic [7:0] pixel_out;
    logic out_valid, out_sof, out_eol, out_last, out_ready = 0;
    logic busy;

    layernorm #(.MAX_PIXELS(4096)) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_ready(out_ready),
        .busy(busy)
    );

    always #5 clk = ~clk;

    integer vf, of, rc, i, j, ci, ncases;
    integer W, H, sign_in, sign_out, ish, eps, gamma_q, beta, nvals;
    integer px_in [0:4095];
    integer lut [0:255];
    integer out_count, framing_err, exp_count;
    integer seed = 12345;
    integer accepted;

    task cfg_write(input [15:0] a, input [15:0] d);
        begin
            @(posedge clk);
            cfg_valid = 1; cfg_we = 1; cfg_addr = a; cfg_data = d;
            @(posedge clk);
            cfg_valid = 0; cfg_we = 0;
        end
    endtask

    // Collect outputs in the background while a case runs.
    integer out_idx;
    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            if (out_idx == 0 && !out_sof) framing_err = framing_err + 1;
            if (out_eol && ((out_idx % W) != W - 1))
                framing_err = framing_err + 1;
            if (!out_eol && ((out_idx % W) == W - 1))
                framing_err = framing_err + 1;
            if (out_last && (out_idx != exp_count - 1))
                framing_err = framing_err + 1;
            if (!out_last && (out_idx == exp_count - 1))
                framing_err = framing_err + 1;
            $fwrite(of, "%0d\n", pixel_out);
            out_idx = out_idx + 1;
            out_count = out_count + 1;
        end
        if (out_valid && !out_ready && out_sof && out_idx != 0)
            framing_err = framing_err + 1;
    end

    // Random out_ready.
    always @(posedge clk) begin
        out_ready <= ($random(seed) % 100) < 80;
    end

    initial begin
        vf = $fopen("vectors_layernorm.txt", "r");
        if (vf == 0) begin $display("cannot open vectors_layernorm.txt"); $finish; end
        of = $fopen("out_layernorm.txt", "w");
        rc = $fscanf(vf, "%d", ncases);
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (ci = 0; ci < ncases; ci = ci + 1) begin
            rc = $fscanf(vf, "%d %d %d %d %d %d %d %d %d", W, H, sign_in,
                         sign_out, ish, eps, gamma_q, beta, nvals);
            if (rc != 9) begin
                $display("bad header rc=%0d case %0d", rc, ci);
                $finish;
            end
            for (i = 0; i < nvals; i = i + 1) begin
                rc = $fscanf(vf, "%d", px_in[i]);
            end
            for (i = 0; i < 256; i = i + 1) begin
                rc = $fscanf(vf, "%d", lut[i]);
            end
            // Config.
            cfg_write(16'h0001, W[15:0]);
            cfg_write(16'h0002, H[15:0]);
            cfg_write(16'h0003, (sign_out << 1) | sign_in);
            cfg_write(16'h0004, ish[15:0]);
            cfg_write(16'h0005, eps[15:0]);
            cfg_write(16'h0006, gamma_q[15:0]);
            cfg_write(16'h0007, beta[15:0]);
            for (i = 0; i < 256; i = i + 1)
                cfg_write(16'h0100 + i[15:0], lut[i][15:0]);
            // Send the frame.
            exp_count = nvals;
            out_count = 0; out_idx = 0; framing_err = 0;
            for (i = 0; i < nvals; i = i + 1) begin
                // random gap before this pixel
                while (($random(seed) % 100) < 30) begin
                    @(posedge clk); #1; in_valid = 0;
                end
                @(posedge clk); #1;
                pixel_in = px_in[i][7:0];
                in_valid = 1;
                in_sof = (i == 0);
                // Wait for the clock edge at which in_ready is high:
                // the pixel is accepted at that edge, so deassert
                // in_valid immediately. (Do NOT wait for in_ready
                // after acceptance: holding in_valid high lets the
                // DUT re-accept the same pixel as the first pixel of
                // the next frame once it returns to idle.)
                // The #1 after each edge lets the DUT's edge-triggered
                // logic settle before the TB drives, so combinational
                // intermediates (px_s) are stable when sampled.
                // Acceptance is checked immediately after the edge
                // (pre-edge in_ready, no #1): if in_ready was high,
                // the pixel was accepted at that edge.
                accepted = 0;
                while (!accepted) begin
                    @(posedge clk);
                    if (in_ready) accepted = 1;
                    #1;
                end
                in_valid = 0; in_sof = 0;
            end
            // Wait for all outputs.
            begin : wait_out
                integer t;
                t = 0;
                while (out_count < exp_count && t < 200000) begin
                    @(posedge clk); t = t + 1;
                end
                if (out_count < exp_count)
                    $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                             ci, out_count, exp_count);
            end
            $display("case %0d (%0dx%0d): %0d outputs, framing errors=%0d",
                     ci, W, H, out_count, framing_err);
        end
        $display("TB layernorm DONE");
        $fclose(of);
        $finish;
    end
endmodule
