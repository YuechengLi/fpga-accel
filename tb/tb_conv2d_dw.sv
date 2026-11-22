// tb_conv2d_dw.sv -- SystemVerilog testbench for conv2d_dw (Stage 2).
// Reads vectors_dw.txt (written by golden_dw.py gen), streams each case
// as single pixels with random in_valid gaps and random out_ready stalls,
// writes one output pixel per line to out_dw.txt, checks out_sof /
// out_eol / out_last framing against the golden output size.
// Run:
//   iverilog -g2012 -o tb_dw.vvp ../rtl/vision/conv2d_dw.sv tb_conv2d_dw.sv && vvp tb_dw.vvp
`timescale 1ns/1ps

module tb_conv2d_dw;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [8:0]  cfg_addr = 9'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [7:0]  pixel_in = 8'd0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;
    logic        busy;

    conv2d_dw #(.MAX_W(3840)) dut (
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

    task cfg_wr(input logic [8:0] a, input logic [15:0] d);
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
    int mode, k, dil, sh, w, h, pt, pb, pl, pr, sx, sy;
    int sign_in, sign_out;
    int spx, spy, pw, ph, ow, oh;
    int i, p;
    int coeff_v [0:288];
    int img [0:131071];
    int n_out, tb_err;
    int cycles;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_dw.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_dw.txt"); $finish; end
        fout = $fopen("out_dw.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB dw: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                           mode, k, dil, sh, w, h, pt, pb, pl, pr, sx, sy,
                           sign_in, sign_out);
            for (i = 0; i < 289; i++)
                code = $fscanf(fvec, "%d", coeff_v[i]);
            for (i = 0; i < w*h; i++)
                code = $fscanf(fvec, "%d", img[i]);

            if (mode == 0) begin spx = k; spy = k; end
            else if (mode == 1) begin spx = k; spy = 1; end
            else if (mode == 2) begin spx = 1; spy = k; end
            else begin spx = (k - 1) * dil + 1; spy = (k - 1) * dil + 1; end
            pw = w + pl + pr;
            ph = h + pt + pb;
            ow = (pw - spx) / sx + 1;
            oh = (ph - spy) / sy + 1;

            cfg_wr(9'h000, k[15:0]);
            cfg_wr(9'h001, sh[15:0]);
            cfg_wr(9'h002, w[15:0]);
            cfg_wr(9'h003, h[15:0]);
            cfg_wr(9'h004, pt[15:0]);
            cfg_wr(9'h005, pb[15:0]);
            cfg_wr(9'h006, pl[15:0]);
            cfg_wr(9'h007, pr[15:0]);
            cfg_wr(9'h008, sx[15:0]);
            cfg_wr(9'h009, sy[15:0]);
            cfg_wr(9'h00A, mode[15:0]);
            cfg_wr(9'h00B, dil[15:0]);
            cfg_wr(9'h00F, (sign_out << 1) | sign_in);
            for (i = 0; i < 289; i++)
                cfg_wr(9'h010 + i[8:0], coeff_v[i][15:0]);

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
                        if (out_valid && out_ready_r) begin
                            $fwrite(fout, "%0d\n", pixel_out);
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
            $display("case %0d (mode=%0d k=%0d d=%0d %0dx%0d shift=%0d pad=(%0d,%0d,%0d,%0d) stride=(%0d,%0d) out=%0dx%0d): %0d outputs, framing errors=%0d",
                     ci, mode, k, dil, w, h, sh, pt, pb, pl, pr, sx, sy, ow, oh, n_out, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB dw DONE");
        $finish;
    end
endmodule
