// tb_conv2d_mc.sv -- SystemVerilog testbench for conv2d_mc (Stage 3).
// Reads vectors_mc.txt (written by golden_mc.py), streams each case's
// C_in frames into the tile buffer, then for each output channel writes
// the (o,t) lane coefficients and GO, collecting the emitted output frame.
// Random in_valid gaps + random out_ready stalls. Checks out_sof /
// out_eol / out_last framing against the golden output size.
// Run:
//   iverilog -g2012 -o tb_mc.vvp ../rtl/conv2d_mc.sv tb_conv2d_mc.sv && vvp tb_mc.vvp
`timescale 1ns/1ps

module tb_conv2d_mc;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [7:0]  pixel_in = 8'd0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;
    logic        busy;

    conv2d_mc #(.C_PAR(4), .C_IN_MAX(64), .C_OUT_MAX(32),
                .TILE_W(32), .TILE_H(32)) dut (
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

    task cfg_wr(input logic [15:0] a, input logic [15:0] d);
        begin
            @(negedge clk);
            cfg_valid = 1'b1; cfg_we = 1'b1; cfg_addr = a; cfg_data = d;
            @(negedge clk);
            cfg_valid = 1'b0; cfg_we = 1'b0;
        end
    endtask

    // wait until the DUT is not busy (a compute pass finished), with a
    // timeout. A short fixed delay first, so busy has time to go high.
    task wait_not_busy;
        int t;
        begin
            repeat (10) @(posedge clk);
            t = 0;
            while (busy && t < 200000) begin
                @(posedge clk);
                t++;
            end
            if (busy) begin
                $display("ERROR: timeout waiting for busy==0");
                $finish;
            end
        end
    endtask

    // send one pixel, holding it until the DUT accepts it, then a random
    // 0..2 cycle gap
    task send_pixel(input logic [7:0] px, input logic sof);
        int gap;
        begin
            @(negedge clk);
            pixel_in = px; in_valid = 1'b1; in_sof = sof;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            pixel_in = '0; in_valid = 1'b0; in_sof = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    int fvec, fout, code, ncases, ci;
    int k, sh, w, h, pt, pb, pl, pr, sx, sy, cin, cout, ow, oh;
    int i, j, c, o, t, l, p;
    int tmp;
    int ntiles;
    int coeff_v [0:31][0:63][0:120];  // [o][c][k] (KxK region only, rest 0)
    int img [0:63][0:1023];
    int n_out, tb_err;
    int cycles;
    int sgn_in, sgn_out;
    int fuse_act, act_slope;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_mc.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_mc.txt"); $finish; end
        fout = $fopen("out_mc.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB mc: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                           k, sh, w, h, pt, pb, pl, pr, sx, sy,
                           cin, cout, ow, oh, sgn_in, sgn_out,
                           fuse_act, act_slope);
            for (o = 0; o < cout; o++)
                for (c = 0; c < cin; c++) begin
                    for (i = 0; i < 121; i++) coeff_v[o][c][i] = 0;
                    for (i = 0; i < k*k; i++) begin
                        code = $fscanf(fvec, "%d", tmp);
                        coeff_v[o][c][i] = tmp;
                    end
                end
            for (c = 0; c < cin; c++)
                for (i = 0; i < w*h; i++) begin
                    code = $fscanf(fvec, "%d", tmp);
                    img[c][i] = tmp;
                end

            ntiles = (cin + 3) / 4;

            cfg_wr(16'h0000, k[15:0]);
            cfg_wr(16'h0001, sh[15:0]);
            cfg_wr(16'h0002, w[15:0]);
            cfg_wr(16'h0003, h[15:0]);
            cfg_wr(16'h0004, pt[15:0]);
            cfg_wr(16'h0005, pb[15:0]);
            cfg_wr(16'h0006, pl[15:0]);
            cfg_wr(16'h0007, pr[15:0]);
            cfg_wr(16'h0008, sx[15:0]);
            cfg_wr(16'h0009, sy[15:0]);
            cfg_wr(16'h000A, cin[15:0]);
            cfg_wr(16'h000B, cout[15:0]);
            cfg_wr(16'h000D, ow[15:0]);
            cfg_wr(16'h000E, oh[15:0]);
            cfg_wr(16'h000F, {14'd0, (sgn_out != 0), (sgn_in != 0)});
            cfg_wr(16'h0010, fuse_act[15:0]);
            cfg_wr(16'h0011, act_slope[15:0]);

            // ---- LOAD: stream C_in frames, channel-major ----
            for (c = 0; c < cin; c++) begin
                for (p = 0; p < w*h; p++) begin
                    send_pixel(img[c][p][7:0], (c == 0 && p == 0));
                end
            end

            // ---- COMPUTE + collect, one output channel at a time ----
            for (o = 0; o < cout; o++) begin
                n_out = 0; tb_err = 0; cycles = 0;
                fork
                    begin : driver
                        for (t = 0; t < ntiles; t++) begin
                            // write lane coeffs for (o, t): lane l = channel
                            // c = t*4 + l; only the KxK region (row-major
                            // in the 11x11 grid).
                            for (l = 0; l < 4; l++) begin
                                c = t*4 + l;
                                for (i = 0; i < k; i++) begin
                                    for (j = 0; j < k; j++) begin
                                        if (c < cin)
                                            cfg_wr(16'h0100 + l*128 + i*11 + j,
                                                   coeff_v[o][c][i*k+j][15:0]);
                                        else
                                            cfg_wr(16'h0100 + l*128 + i*11 + j,
                                                   16'd0);
                                    end
                                end
                                // (Coefficients outside the current KxK
                                // region are never read by the DUT, so they
                                // need no zeroing.)
                            end
                            cfg_wr(16'h000C, 16'd1);  // GO
                            wait_not_busy;
                        end
                    end
                    begin : collector
                        while (n_out < ow*oh) begin
                            @(posedge clk);
                            cycles++;
                            if (cycles > 5000000) begin
                                $display("ERROR case %0d ch %0d: timeout, got %0d/%0d outputs",
                                         ci, o, n_out, ow*oh);
                                $finish;
                            end
                            if (out_valid && out_ready_r) begin
                                if (sgn_out)
                                    $fwrite(fout, "%0d\n", $signed(pixel_out));
                                else
                                    $fwrite(fout, "%0d\n", pixel_out);
                                if (n_out == 0 && !out_sof) begin
                                    $display("ERROR case %0d ch %0d: first output missing out_sof", ci, o);
                                    tb_err++;
                                end
                                if ((n_out % ow == ow-1) && !out_eol) begin
                                    $display("ERROR case %0d ch %0d: output %0d missing out_eol", ci, o, n_out);
                                    tb_err++;
                                end
                                if (n_out == ow*oh-1 && !out_last) begin
                                    $display("ERROR case %0d ch %0d: last output missing out_last", ci, o);
                                    tb_err++;
                                end
                                n_out++;
                            end
                        end
                    end
                join
                $display("case %0d ch %0d (K=%0d %0dx%0d C_in=%0d C_out=%0d shift=%0d pad=(%0d,%0d,%0d,%0d) stride=(%0d,%0d) out=%0dx%0d): %0d outputs, framing errors=%0d",
                         ci, o, k, w, h, cin, cout, sh, pt, pb, pl, pr, sx, sy, ow, oh, n_out, tb_err);
                repeat (20) @(posedge clk);
            end
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB mc DONE");
        $finish;
    end
endmodule
