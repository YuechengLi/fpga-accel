// tb_conv2d_dense.sv -- SystemVerilog testbench for conv2d_dense (Stage 1).
// Reads vectors_dense.txt (written by golden_dense.py gen), streams each
// case as single pixels with random in_valid gaps and random out_ready
// stalls, writes one output pixel per line to out_dense.txt, checks
// out_sof / out_eol / out_last framing against the golden output size.
// PX=1 (Stage 1 target).
// Run:
//   iverilog -g2012 -o tb_dense.vvp ../rtl/vision/conv2d_dense.sv tb_conv2d_dense.sv && vvp tb_dense.vvp
`timescale 1ns/1ps

module tb_conv2d_dense;
    parameter int PX = 1;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [7:0]  cfg_addr = 8'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [8*PX-1:0] pixel_in = '0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [8*PX-1:0] pixel_out;
    logic [PX-1:0]   out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;
    logic        busy;

    conv2d_dense #(.PX(PX), .MAX_W(3840)) dut (
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

    task cfg_wr(input logic [7:0] a, input logic [15:0] d);
        begin
            @(negedge clk);
            cfg_valid = 1'b1; cfg_we = 1'b1; cfg_addr = a; cfg_data = d;
            @(negedge clk);
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
    int pw, ph, ow, oh;
    int i, p;
    int coeff_v [0:120];
    int img [0:131071];
    int n_out, tb_err;
    int cycles;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_dense.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_dense.txt"); $finish; end
        fout = $fopen("out_dense.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB dense PX=%0d: %0d cases", PX, ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d %d %d %d %d",
                           w, h, k, sh, pt, pb, pl, pr, sx, sy);
            for (i = 0; i < 121; i++)
                code = $fscanf(fvec, "%d", coeff_v[i]);
            for (i = 0; i < w*h; i++)
                code = $fscanf(fvec, "%d", img[i]);

            pw = w + pl + pr;
            ph = h + pt + pb;
            ow = (pw - k) / sx + 1;
            oh = (ph - k) / sy + 1;

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
            for (i = 0; i < 121; i++)
                cfg_wr(8'h10 + i[7:0], coeff_v[i][15:0]);

            n_out = 0; tb_err = 0; cycles = 0;
            fork
                begin : sender
                    for (p = 0; p < w*h; p++) begin
                        send_pixel(img[p][7:0], (p == 0));
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
                        if ((out_valid != '0) && out_ready_r) begin
                            $fwrite(fout, "%0d\n", pixel_out[7:0]);
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
            $display("case %0d (%0dx%0d k=%0d shift=%0d pad=(%0d,%0d,%0d,%0d) stride=(%0d,%0d) out=%0dx%0d): %0d outputs, framing errors=%0d",
                     ci, w, h, k, sh, pt, pb, pl, pr, sx, sy, ow, oh, n_out, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB dense PX=%0d DONE", PX);
        $finish;
    end
endmodule
