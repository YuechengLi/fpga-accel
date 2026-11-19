// tb_act.sv -- SystemVerilog testbench for act (Stage 3).
// Reads vectors_act.txt (written by golden_act.py). For PWL modes it
// loads the 32-segment table (y0 / dslope) via the config bus, then
// streams the input values with random in_valid gaps and collects the
// outputs with random out_ready stalls. Framing is a passthrough: the
// testbench sets in_sof on the first value, in_eol on every value, and
// in_last on the last value, and checks the same on the output.
// Run:
//   iverilog -g2012 -o tb_act.vvp ../rtl/act.sv tb_act.sv && vvp tb_act.vvp
`timescale 1ns/1ps

module tb_act;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [31:0] pixel_in = 32'd0;
    logic        in_valid = 1'b0, in_sof = 1'b0, in_eol = 1'b0, in_last = 1'b0;
    logic        in_ready;
    logic [31:0] pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;

    act dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .data_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_eol(in_eol), .in_last(in_last), .in_ready(in_ready),
        .data_out(pixel_out), .out_valid(out_valid),
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

    task send_val(input logic [31:0] v, input logic sof, input logic eol,
                  input logic last);
        int gap;
        begin
            @(negedge clk);
            pixel_in = v; in_valid = 1'b1; in_sof = sof;
            in_eol = eol; in_last = last;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            pixel_in = '0; in_valid = 1'b0; in_sof = 1'b0;
            in_eol = 1'b0; in_last = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    int fvec, fout, code, ncases, ci;
    int mode, slope_fp, nvals;
    int i, n_out, tb_err, cycles;
    int tmp;
    int vals [0:1023];
    int y0v, dsv;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_act.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_act.txt"); $finish; end
        fout = $fopen("out_act.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB act: %0d cases", ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d", mode, slope_fp, nvals);
            if (mode >= 2) begin
                // load the 32-segment PWL table
                for (i = 0; i < 32; i++) begin
                    code = $fscanf(fvec, "%d %d", y0v, dsv);
                    cfg_wr(16'h0100 + i*2,     y0v[15:0]);
                    cfg_wr(16'h0100 + i*2 + 1, dsv[15:0]);
                end
            end
            for (i = 0; i < nvals; i++) begin
                code = $fscanf(fvec, "%d", tmp);
                vals[i] = tmp;
            end

            cfg_wr(16'h0000, mode[15:0]);
            cfg_wr(16'h0001, slope_fp[15:0]);

            n_out = 0; tb_err = 0; cycles = 0;
            fork
                begin : sender
                    for (i = 0; i < nvals; i++)
                        send_val(vals[i], (i == 0), 1'b1, (i == nvals-1));
                end
                begin : collector
                    while (n_out < nvals) begin
                        @(posedge clk);
                        cycles++;
                        if (cycles > 5000000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                                     ci, n_out, nvals);
                            $finish;
                        end
                        if (out_valid && out_ready_r) begin
                            $fwrite(fout, "%0d\n", $signed(pixel_out));
                            if (n_out == 0 && !out_sof) begin
                                $display("ERROR case %0d: first output missing out_sof", ci);
                                tb_err++;
                            end
                            if (!out_eol) begin
                                $display("ERROR case %0d: output %0d missing out_eol", ci, n_out);
                                tb_err++;
                            end
                            if (n_out == nvals-1 && !out_last) begin
                                $display("ERROR case %0d: last output missing out_last", ci);
                                tb_err++;
                            end
                            n_out++;
                        end
                    end
                end
            join
            $display("case %0d (mode=%0d): %0d outputs, framing errors=%0d",
                     ci, mode, n_out, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB act DONE");
        $finish;
    end
endmodule
