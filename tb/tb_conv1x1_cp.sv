// tb_conv1x1_cp.sv -- SystemVerilog testbench for conv1x1_cp (Stage 7b,
// SRCNN L2 engine: P_OUT=7 x P_IN=64 per cycle, NGROUP=5 groups per
// input position; the last group has only 4 valid outputs).
// Reads vectors_cp.txt (written by golden_cp.py gen), streams each case's
// input position vectors with random in_valid gaps and random out_ready
// stalls, writes one output channel value per line to out_cp.txt (only
// the valid bytes of each group, per out_mask), checks out_group /
// out_mask / out_sof / out_eol / out_last framing, and measures
// cycles/position from the first accepted input vector to the last
// output group (for the cycle-model cross-check: 5 cycles/position).
// Run:
//   iverilog -g2012 -o tb_cp.vvp ../rtl/vision/conv1x1_cp.sv ../rtl/common/adder_tree.sv \
//       tb_conv1x1_cp.sv && vvp tb_cp.vvp
`timescale 1ns/1ps

module tb_conv1x1_cp;
    parameter int P_OUT = 7;
    parameter int P_IN = 64;
    parameter int C_OUT = 32;
    parameter int NGROUP = 5;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [8*P_IN-1:0] in_vec = '0;
    logic        in_valid = 1'b0, in_sof = 1'b0, in_eol = 1'b0, in_last = 1'b0;
    logic        in_ready;
    logic [8*P_OUT-1:0] pixel_out;
    logic        out_valid;
    logic [P_OUT-1:0] out_mask;
    logic [3:0]  out_group;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;

    conv1x1_cp #(.P_OUT(P_OUT), .P_IN(P_IN), .C_OUT(C_OUT), .NGROUP(NGROUP)) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .in_vec(in_vec), .in_valid(in_valid),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid), .out_mask(out_mask),
        .out_group(out_group),
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
            @(posedge clk);
            #1;
            cfg_valid = 1'b0; cfg_we = 1'b0;
        end
    endtask

    // send one input vector, holding it until the DUT accepts it, then a
    // random 0..2 cycle gap
    task send_vec(input logic [8*P_IN-1:0] v, input logic sof,
                  input logic eol, input logic last);
        int gap;
        begin
            @(negedge clk);
            in_vec = v; in_valid = 1'b1; in_sof = sof; in_eol = eol; in_last = last;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            in_vec = '0; in_valid = 1'b0; in_sof = 1'b0; in_eol = 1'b0; in_last = 1'b0;
            gap = ($random & 32'h7fffffff) % 3;
            repeat (gap) @(negedge clk);
        end
    endtask

    int fvec, fout, code, ncases, ci;
    int shift, sign_in, sign_out, act, slope, npos;
    int o, c, p, e;
    int coeff_v [0:31][0:63];
    int bias_v [0:31];
    int vec_v [0:63];
    int tmp;
    int n_grp, n_val, tb_err;
    int cyc_cnt = 0;
    int first_in_cycle, last_out_cycle;
    logic [8*P_IN-1:0] vec_packed;

    always @(posedge clk) if (rst_n) cyc_cnt <= cyc_cnt + 1;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_cp.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_cp.txt"); $finish; end
        fout = $fopen("out_cp.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        $display("TB cp P_OUT=%0d P_IN=%0d: %0d cases", P_OUT, P_IN, ncases);

        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d",
                           shift, sign_in, sign_out, act, slope, npos);
            for (o = 0; o < C_OUT; o++)
                for (c = 0; c < P_IN; c++) begin
                    code = $fscanf(fvec, "%d", tmp);
                    coeff_v[o][c] = tmp;
                end
            for (o = 0; o < C_OUT; o++) begin
                code = $fscanf(fvec, "%d", tmp);
                bias_v[o] = tmp;
            end

            cfg_wr(16'h0001, shift[15:0]);
            cfg_wr(16'h000A, 16'd64);
            cfg_wr(16'h000B, 16'd32);
            cfg_wr(16'h000F, {14'd0, sign_out[0], sign_in[0]});
            cfg_wr(16'h0010, act[15:0]);
            cfg_wr(16'h0011, slope[15:0]);
            for (o = 0; o < C_OUT; o++)
                for (c = 0; c < P_IN; c++)
                    cfg_wr(16'h0100 + o*64 + c, coeff_v[o][c][15:0]);
            for (o = 0; o < C_OUT; o++) begin
                cfg_wr(16'h0900 + o, bias_v[o][15:0]);
                cfg_wr(16'h0920 + o, bias_v[o][31:16]);
            end

            n_grp = 0; n_val = 0; tb_err = 0;
            first_in_cycle = 0; last_out_cycle = 0;
            fork
                begin : sender
                    for (p = 0; p < npos; p++) begin
                        for (c = 0; c < P_IN; c++) begin
                            code = $fscanf(fvec, "%d", tmp);
                            vec_v[c] = tmp;
                        end
                        if (p == 0) first_in_cycle = cyc_cnt;
                        // eol on positions 3 and 7 (arbitrary row ends),
                        // last on the final position
                        for (c = 0; c < P_IN; c++)
                            vec_packed[c*8 +: 8] = vec_v[c][7:0];
                        send_vec(vec_packed, (p == 0),
                                 ((p == 3) || (p == 7)), (p == npos-1));
                    end
                end
                begin : collector
                    int ocyc, pos, grp;
                    int exp_mask;
                    while (n_grp < npos*NGROUP) begin
                        @(posedge clk);
                        if (cyc_cnt - first_in_cycle > 200000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d output groups",
                                     ci, n_grp, npos*NGROUP);
                            $display("  DEBUG cur_valid=%0d group=%0d buf_valid=%b%b buf_rd=%0d buf_wr=%0d in_ready=%0d out_valid=%0d out_ready=%0d stall=%0d",
                                     dut.cur_valid, dut.group, dut.buf_valid[0], dut.buf_valid[1],
                                     dut.buf_rd, dut.buf_wr, in_ready, out_valid, out_ready_r, dut.stall);
                            $finish;
                        end
                        if (out_valid && out_ready_r) begin
                            ocyc = n_grp;
                            pos = ocyc / NGROUP;
                            grp = ocyc % NGROUP;
                            exp_mask = (grp == NGROUP-1) ? 7'b0001111 : 7'b1111111;
                            for (e = 0; e < P_OUT; e++)
                                if (out_mask[e])
                                    $fwrite(fout, "%0d\n", pixel_out[e*8 +: 8]);
                            n_val += (grp == NGROUP-1) ? 4 : 7;
                            if (out_group != grp[3:0]) begin
                                $display("ERROR case %0d: group %0d wrong out_group %0d, expected %0d",
                                         ci, ocyc, out_group, grp);
                                tb_err++;
                            end
                            if (out_mask != exp_mask[6:0]) begin
                                $display("ERROR case %0d: group %0d wrong out_mask %b, expected %b",
                                         ci, ocyc, out_mask, exp_mask[6:0]);
                                tb_err++;
                            end
                            if ((ocyc == 0) != out_sof) begin
                                $display("ERROR case %0d: group %0d out_sof=%0d, expected %0d",
                                         ci, ocyc, out_sof, (ocyc == 0));
                                tb_err++;
                            end
                            if ((((pos == 3) || (pos == 7))) != out_eol) begin
                                $display("ERROR case %0d: group %0d out_eol=%0d, expected %0d (pos=%0d)",
                                         ci, ocyc, out_eol, ((pos == 3) || (pos == 7)), pos);
                                tb_err++;
                            end
                            if (((pos == npos-1) && (grp == NGROUP-1)) != out_last) begin
                                $display("ERROR case %0d: group %0d out_last=%0d, expected %0d",
                                         ci, ocyc, out_last, ((pos == npos-1) && (grp == NGROUP-1)));
                                tb_err++;
                            end
                            n_grp++;
                            last_out_cycle = cyc_cnt;
                        end
                    end
                end
            join
            $display("case %0d (shift=%0d act=%0d sign=(%0d,%0d) npos=%0d): %0d values, %0d cycles first-in->last-out (%.2f cy/pos), framing errors=%0d",
                     ci, shift, act, sign_in, sign_out, npos, n_val,
                     last_out_cycle - first_in_cycle,
                     (last_out_cycle - first_in_cycle) * 1.0 / npos, tb_err);
            repeat (20) @(posedge clk);
        end
        $fclose(fout);
        $fclose(fvec);
        $display("TB cp DONE");
        $finish;
    end
endmodule
