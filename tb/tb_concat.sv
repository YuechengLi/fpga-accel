// tb_concat.sv -- SystemVerilog testbench for concat (Stage 4).
// Reads vectors_concat.txt (written by golden_concat.py), streams the
// two feature-map streams A and B concurrently (independent random gaps),
// collects the concatenated output with random out_ready stalls, and
// checks out_sof / out_eol / out_last framing.
// Run:
//   iverilog -g2012 -o tb_concat.vvp ../rtl/vision/concat.sv tb_concat.sv && vvp tb_concat.vvp
`timescale 1ns/1ps

module tb_concat;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [7:0]  a_in = 8'd0, b_in = 8'd0;
    logic        a_valid = 1'b0, a_sof = 1'b0;
    logic        b_valid = 1'b0, b_sof = 1'b0;
    logic        a_ready, b_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;

    concat dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .a_in(a_in), .a_valid(a_valid), .a_sof(a_sof), .a_ready(a_ready),
        .b_in(b_in), .b_valid(b_valid), .b_sof(b_sof), .b_ready(b_ready),
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

    // send one pixel on stream A, random 0..2 cycle gap.
    task send_a(input logic [7:0] px, input logic sof);
        int gap;
        begin
            @(negedge clk);
            while (!a_ready) @(negedge clk);
            a_in = px; a_valid = 1'b1; a_sof = sof;
            @(posedge clk);
            @(negedge clk);
            a_in = '0; a_valid = 1'b0; a_sof = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    // send one pixel on stream B, random 0..2 cycle gap (same handshake as A).
    task send_b(input logic [7:0] px, input logic sof);
        int gap;
        begin
            @(negedge clk);
            while (!b_ready) @(negedge clk);
            b_in = px; b_valid = 1'b1; b_sof = sof;
            @(posedge clk);
            @(negedge clk);
            b_in = '0; b_valid = 1'b0; b_sof = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    int fvec, fout, code, ncases, ci;
    int mode, w, h;
    int i, ia, ib, n_out, tb_err, cycles;
    int tmp;
    int orl;  // output row length (w sequential, 2w interleave)
    int img_a [0:65535];
    int img_b [0:65535];

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_concat.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_concat.txt"); $finish; end
        fout = $fopen("out_concat.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB concat: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d", mode, w, h);
            for (i = 0; i < w*h; i++) begin
                code = $fscanf(fvec, "%d", tmp);
                img_a[i] = tmp;
            end
            for (i = 0; i < w*h; i++) begin
                code = $fscanf(fvec, "%d", tmp);
                img_b[i] = tmp;
            end

            cfg_wr(16'h0000, mode[15:0]);
            cfg_wr(16'h0001, w[15:0]);
            cfg_wr(16'h0002, h[15:0]);
            orl = (mode == 1) ? 2*w : w;

            n_out = 0; tb_err = 0; cycles = 0;
            fork
                begin : sender_a
                    for (ia = 0; ia < w*h; ia++)
                        send_a(img_a[ia][7:0], (ia == 0));
                end
                begin : sender_b
                    for (ib = 0; ib < w*h; ib++)
                        send_b(img_b[ib][7:0], (ib == 0));
                end
                begin : collector
                    while (n_out < 2*w*h) begin
                        @(posedge clk);
                        cycles++;
                        if (cycles > 5000000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                                     ci, n_out, 2*w*h);
                            $finish;
                        end
                        if (out_valid && out_ready_r) begin
                            $fwrite(fout, "%0d\n", pixel_out);
                            if (n_out == 0 && !out_sof) begin
                                $display("ERROR case %0d: first output missing out_sof", ci);
                                tb_err++;
                            end
                            if ((n_out % orl == orl-1) && !out_eol) begin
                                $display("ERROR case %0d: output %0d missing out_eol", ci, n_out);
                                tb_err++;
                            end
                            if ((n_out % orl != orl-1) && out_eol) begin
                                $display("ERROR case %0d: output %0d spurious out_eol", ci, n_out);
                                tb_err++;
                            end
                            if (n_out == 2*w*h-1 && !out_last) begin
                                $display("ERROR case %0d: last output missing out_last", ci);
                                tb_err++;
                            end
                            n_out++;
                        end
                    end
                end
            join
            $display("case %0d (mode=%0d %0dx%0d out=%0d): %0d outputs, framing errors=%0d",
                     ci, mode, w, h, 2*w*h, n_out, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB concat DONE");
        $finish;
    end
endmodule
