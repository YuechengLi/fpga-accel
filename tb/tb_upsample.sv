// tb_upsample.sv -- SystemVerilog testbench for upsample (Stage 4).
// Reads vectors_upsample.txt (written by golden_upsample.py), streams the
// input with random gaps, collects the 2W x 2H output with random
// out_ready stalls, and checks out_sof / out_eol / out_last framing.
// Run:
//   iverilog -g2012 -o tb_up.vvp ../rtl/upsample.sv tb_upsample.sv && vvp tb_up.vvp
`timescale 1ns/1ps

module tb_upsample;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [7:0]  cfg_addr = 8'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [7:0]  pixel_in = 8'd0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;

    upsample dut (
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

    task cfg_wr(input logic [7:0] a, input logic [15:0] d);
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
            while (!in_ready) @(negedge clk);
            pixel_in = px; in_valid = 1'b1; in_sof = sof;
            @(posedge clk);
            @(negedge clk);
            pixel_in = '0; in_valid = 1'b0; in_sof = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    int fvec, fout, code, ncases, ci;
    int mode, w, h;
    int i, p, n_out, tb_err, cycles;
    int tmp;
    int nin, nout;
    int inp [0:131071];

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_upsample.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_upsample.txt"); $finish; end
        fout = $fopen("out_upsample.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB upsample: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d", mode, w, h);
            nin = (mode == 2) ? 4*w*h : w*h;
            nout = 4*w*h;
            for (i = 0; i < nin; i++) begin
                code = $fscanf(fvec, "%d", tmp);
                inp[i] = tmp;
            end

            cfg_wr(8'h00, mode[15:0]);
            cfg_wr(8'h01, w[15:0]);
            cfg_wr(8'h02, h[15:0]);

            n_out = 0; tb_err = 0; cycles = 0;
            fork
                begin : sender
                    for (p = 0; p < nin; p++)
                        send_pixel(inp[p][7:0], (p == 0));
                end
                begin : collector
                    while (n_out < nout) begin
                        @(posedge clk);
                        cycles++;
                        if (cycles > 3000000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                                     ci, n_out, nout);
                            $finish;
                        end
                        if (out_valid && out_ready_r) begin
                            $fwrite(fout, "%0d\n", pixel_out);
                            if (n_out == 0 && !out_sof) begin
                                $display("ERROR case %0d: first output missing out_sof", ci);
                                tb_err++;
                            end
                            if ((n_out % (2*w) == 2*w-1) && !out_eol) begin
                                $display("ERROR case %0d: output %0d missing out_eol", ci, n_out);
                                tb_err++;
                            end
                            if ((n_out % (2*w) != 2*w-1) && out_eol) begin
                                $display("ERROR case %0d: output %0d spurious out_eol", ci, n_out);
                                tb_err++;
                            end
                            if (n_out == nout-1 && !out_last) begin
                                $display("ERROR case %0d: last output missing out_last", ci);
                                tb_err++;
                            end
                            n_out++;
                        end
                    end
                end
            join
            $display("case %0d (mode=%0d %0dx%0d -> %0dx%0d): %0d outputs, framing errors=%0d",
                     ci, mode, w, h, 2*w, 2*h, n_out, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB upsample DONE");
        $finish;
    end
endmodule
