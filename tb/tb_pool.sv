// tb_pool.sv -- SystemVerilog testbench for pool (Stage 3).
// Reads vectors_pool.txt (written by golden_pool.py), streams each case's
// image with random in_valid gaps, collects the pooled output with random
// out_ready stalls, and checks out_sof / out_eol / out_last framing.
// Run:
//   iverilog -g2012 -o tb_pool.vvp ../rtl/pool.sv tb_pool.sv && vvp tb_pool.vvp
`timescale 1ns/1ps

module tb_pool;
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

    pool dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ready(out_ready_r)
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
    int mode, w, h, gap_shift, ow, oh;
    int i, n_out, tb_err, cycles;
    int tmp;
    int img [0:65535];

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_pool.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_pool.txt"); $finish; end
        fout = $fopen("out_pool.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB pool: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d",
                           mode, w, h, gap_shift, ow, oh);
            for (i = 0; i < w*h; i++) begin
                code = $fscanf(fvec, "%d", tmp);
                img[i] = tmp;
            end

            cfg_wr(16'h0000, mode[15:0]);
            cfg_wr(16'h0001, w[15:0]);
            cfg_wr(16'h0002, h[15:0]);
            cfg_wr(16'h0003, gap_shift[15:0]);

            n_out = 0; tb_err = 0; cycles = 0;
            fork
                begin : sender
                    for (i = 0; i < w*h; i++)
                        send_pixel(img[i][7:0], (i == 0));
                end
                begin : collector
                    while (n_out < ow*oh) begin
                        @(posedge clk);
                        cycles++;
                        if (cycles > 5000000) begin
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
            $display("case %0d (mode=%0d %0dx%0d -> %0dx%0d): %0d outputs, framing errors=%0d",
                     ci, mode, w, h, ow, oh, n_out, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB pool DONE");
        $finish;
    end
endmodule
