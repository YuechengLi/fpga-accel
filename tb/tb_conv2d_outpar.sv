// tb_conv2d_outpar.sv -- SystemVerilog testbench for conv2d_outpar (Stage 7b,
// SRCNN L1 engine: P_OUT=16 output channels per cycle, NGROUP=C_out/16
// groups per output position).
// Reads vectors_outpar.txt (written by golden_outpar.py gen), streams each
// case as single pixels with random in_valid gaps and random out_ready
// stalls, writes one output channel value per line to out_outpar.txt
// (position raster order, group 0..NGROUP-1, channel 0..15 within a group),
// checks out_group / out_sof / out_eol / out_last framing, and measures
// cycles/frame from GO to the last output (for the cycle-model cross-check).
// Run:
//   iverilog -g2012 -o tb_outpar.vvp ../rtl/conv2d_outpar.sv \
//       ../rtl/adder_tree.sv ../rtl/conv2d_mc.sv tb_conv2d_outpar.sv \
//       && vvp tb_outpar.vvp
`timescale 1ns/1ps

module tb_conv2d_outpar;
    parameter int P_OUT = 16;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [7:0]  pixel_in = '0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [8*P_OUT-1:0] pixel_out;
    logic        out_valid;
    logic [3:0]  out_group;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;
    logic        busy;

    conv2d_outpar #(.P_OUT(P_OUT), .MAX_K(9), .LB_DEPTH(2048)) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid),
        .out_group(out_group),
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
            // One clock per write: assert at negedge, the DUT samples at
            // the following posedge, deassert just after it. (The
            // negedge-to-negedge form costs 2 clocks per write and the
            // 5184 coeff writes would dominate the sim.)
            @(negedge clk);
            cfg_valid = 1'b1; cfg_we = 1'b1; cfg_addr = a; cfg_data = d;
            @(posedge clk);
            #1;
            cfg_valid = 1'b0; cfg_we = 1'b0;
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
    int w, h, k, sh, pt, pb, pl, pr, sx, sy;
    int sign_in, sign_out, act, slope, cout;
    int pw, ph, ow, oh, ngroup;
    int i, p, o, t;
    int coeff_v [0:63][0:80];
    int bias_v [0:63];
    int img [0:131071];
    int n_cyc, n_val, tb_err;
    int go_cycle, last_out_cycle;
    int cyc_cnt = 0;
    int tmp;

    // free-running cycle counter (post-reset) for the GO->last-out measure
    always @(posedge clk) if (rst_n) cyc_cnt <= cyc_cnt + 1;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_outpar.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_outpar.txt"); $finish; end
        fout = $fopen("out_outpar.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB outpar P_OUT=%0d: %0d cases", P_OUT, ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                           w, h, k, sh, pt, pb, pl, pr, sx, sy,
                           sign_in, sign_out, act, slope, cout);
            for (o = 0; o < cout; o++)
                for (t = 0; t < 81; t++) begin
                    code = $fscanf(fvec, "%d", tmp);
                    coeff_v[o][t] = tmp;
                end
            for (o = 0; o < cout; o++) begin
                code = $fscanf(fvec, "%d", tmp);
                bias_v[o] = tmp;
            end
            for (i = 0; i < w*h; i++) begin
                code = $fscanf(fvec, "%d", tmp);
                img[i] = tmp;
            end

            pw = w + pl + pr;
            ph = h + pt + pb;
            ow = (pw - k) / sx + 1;
            oh = (ph - k) / sy + 1;
            ngroup = cout / P_OUT;

            // geometry + mode (same config map as conv2d_mc)
            cfg_wr(16'h00, k[15:0]);
            cfg_wr(16'h01, sh[15:0]);
            cfg_wr(16'h02, w[15:0]);
            cfg_wr(16'h03, h[15:0]);
            cfg_wr(16'h04, pt[15:0]);
            cfg_wr(16'h05, pb[15:0]);
            cfg_wr(16'h06, pl[15:0]);
            cfg_wr(16'h07, pr[15:0]);
            cfg_wr(16'h08, sx[15:0]);
            cfg_wr(16'h09, sy[15:0]);
            cfg_wr(16'h0A, 16'd1);          // C_in = 1
            cfg_wr(16'h0B, cout[15:0]);
            cfg_wr(16'h0D, ow[15:0]);
            cfg_wr(16'h0E, oh[15:0]);
            cfg_wr(16'h0F, {14'd0, sign_out[0], sign_in[0]});
            cfg_wr(16'h10, act[15:0]);
            cfg_wr(16'h11, slope[15:0]);
            // coefficients: addr 0x100 + o*128 + tap (tap = r*9+c)
            for (o = 0; o < cout; o++)
                for (t = 0; t < 81; t++)
                    cfg_wr(16'h0100 + o*128 + t, coeff_v[o][t][15:0]);
            // bias: lo 0x2100+o, hi 0x2140+o (after the coeff range)
            for (o = 0; o < cout; o++) begin
                cfg_wr(16'h2100 + o, bias_v[o][15:0]);
                cfg_wr(16'h2140 + o, bias_v[o][31:16]);
            end

            n_cyc = 0; n_val = 0; tb_err = 0;
            go_cycle = 0; last_out_cycle = 0;
            cfg_wr(16'h000C, 16'd1);        // GO
            go_cycle = cyc_cnt;
            fork
                begin : sender
                    for (p = 0; p < w*h; p++) begin
                        send_pixel(img[p][7:0], (p == 0));
                    end
                end
                begin : collector
                    int ocyc, orow, ocol, grp;
                    while (n_cyc < ow*oh*ngroup) begin
                        @(posedge clk);
                        if (cyc_cnt - go_cycle > 200000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d output cycles",
                                     ci, n_cyc, ow*oh*ngroup);
                            $finish;
                        end
                        if (out_valid && out_ready_r) begin
                            ocyc = n_cyc;
                            orow = ocyc / (ow*ngroup);
                            ocol = (ocyc / ngroup) % ow;
                            grp  = ocyc % ngroup;
                            for (int e = 0; e < P_OUT; e++)
                                $fwrite(fout, "%0d\n", pixel_out[e*8 +: 8]);
                            n_val += P_OUT;
                            if (out_group != grp[3:0]) begin
                                $display("ERROR case %0d: cycle %0d wrong out_group %0d, expected %0d",
                                         ci, ocyc, out_group, grp);
                                tb_err++;
                            end
                            if ((ocyc == 0) != out_sof) begin
                                $display("ERROR case %0d: cycle %0d out_sof=%0d, expected %0d",
                                         ci, ocyc, out_sof, (ocyc == 0));
                                tb_err++;
                            end
                            if ((ocol == ow-1) != out_eol) begin
                                $display("ERROR case %0d: cycle %0d out_eol=%0d, expected %0d (ocol=%0d)",
                                         ci, ocyc, out_eol, (ocol == ow-1), ocol);
                                tb_err++;
                            end
                            if ((ocyc == ow*oh*ngroup-1) != out_last) begin
                                $display("ERROR case %0d: cycle %0d out_last=%0d, expected %0d",
                                         ci, ocyc, out_last, (ocyc == ow*oh*ngroup-1));
                                tb_err++;
                            end
                            n_cyc++;
                            last_out_cycle = cyc_cnt;
                        end
                    end
                end
            join
            $display("case %0d (%0dx%0d k=%0d shift=%0d pad=(%0d,%0d,%0d,%0d) stride=(%0d,%0d) act=%0d sign=(%0d,%0d) out=%0dx%0dx%0d): %0d values, %0d cycles GO->last-out, framing errors=%0d",
                     ci, w, h, k, sh, pt, pb, pl, pr, sx, sy, act, sign_in, sign_out,
                     ow, oh, cout, n_val, last_out_cycle - go_cycle, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB outpar DONE");
        $finish;
    end
endmodule
