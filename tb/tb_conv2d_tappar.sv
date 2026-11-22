// tb_conv2d_tappar.sv -- SystemVerilog testbench for conv2d_tappar
// (Stage 7b, SRCNN L3 engine: tap-parallel 5x5, 32 -> 1; one tap-row
// (5 taps x 32 channels = 160 mults) per cycle, 5 cycles per output
// position). Reads vectors_tappar.txt (written by golden_tappar.py gen),
// streams each case's input frame (H*W position vectors, 32 bytes each)
// with random in_valid gaps and random out_ready stalls, writes one
// output byte per line to out_tappar.txt, checks out_sof / out_eol /
// out_last framing, and measures cycles from GO to the last output
// (for the cycle-model cross-check: 5 cycles/output position).
// Run:
//   iverilog -g2012 -o tb_tappar.vvp ../rtl/vision/conv2d_tappar.sv \
//       ../rtl/common/adder_tree.sv ../rtl/vision/conv2d_mc.sv tb_conv2d_tappar.sv \
//       && vvp tb_tappar.vvp
`timescale 1ns/1ps

module tb_conv2d_tappar;
    parameter int P_IN = 32;
    parameter int MAX_K = 5;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        cfg_valid = 1'b0, cfg_we = 1'b0;
    logic [15:0] cfg_addr = 16'd0;
    logic [15:0] cfg_data = 16'd0;
    logic [8*P_IN-1:0] in_vec = '0;
    logic        in_valid = 1'b0, in_sof = 1'b0, in_eol = 1'b0, in_last = 1'b0;
    logic        in_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;

    conv2d_tappar #(.P_IN(P_IN), .MAX_K(MAX_K)) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .in_vec(in_vec), .in_valid(in_valid),
        .in_sof(in_sof), .in_eol(in_eol), .in_last(in_last),
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
            cfg_valid = 1'b1; cfg_we = 1'b1;
            cfg_addr = a; cfg_data = d;
            @(posedge clk);
            #1;
            cfg_valid = 1'b0; cfg_we = 1'b0;
        end
    endtask

    integer fin, fout;
    integer ncases, ci;
    integer K, shift, W, H;
    integer pad_t, pad_b, pad_l, pad_r;
    integer stride_x, stride_y;
    integer sign_in, sign_out, act, slope;
    integer out_w, out_h;
    integer tmp;
    integer coeff_v [0:24][0:31];
    integer bias_v;
    integer frame_v [0:4095][0:31];
    integer npos_in, npos_out;
    integer i, j, c, p, r, cc;
    integer cyc_cnt;
    integer first_in_cycle, last_out_cycle;
    integer framing_err;
    integer out_count;
    integer exp_sof, exp_eol, exp_last;
    integer timeout_cnt;

    // send one input position vector (with random in_valid gaps)
    task send_vec(input integer idx);
        integer rr, cc2;
        begin
            rr = idx / W;
            cc2 = idx % W;
            @(negedge clk);
            for (c = 0; c < P_IN; c++)
                in_vec[c*8 +: 8] = frame_v[idx][c][7:0];
            in_sof = (rr == 0 && cc2 == 0);
            in_eol = (cc2 == W - 1);
            in_last = (idx == npos_in - 1);
            in_valid = 1'b1;
            // random gap before presenting: hold in_valid low 0..2 cycles
            if ((($random & 32'h7fffffff) % 4) == 0) begin
                in_valid = 1'b0;
                @(posedge clk);
                @(negedge clk);
                in_valid = 1'b1;
            end
            // wait for in_ready
            timeout_cnt = 0;
            while (!in_ready) begin
                @(posedge clk);
                @(negedge clk);
                timeout_cnt = timeout_cnt + 1;
                if (timeout_cnt > 1000) begin
                    $display("ERROR: sender stuck waiting for in_ready (vec %0d)", idx);
                    $finish;
                end
            end
            if (idx == 0) first_in_cycle = cyc_cnt;
            @(posedge clk);
            #1;
            in_valid = 1'b0;
            in_sof = 1'b0; in_eol = 1'b0; in_last = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (rst_n) cyc_cnt <= cyc_cnt + 1;
        else cyc_cnt <= 0;
    end

    initial begin
        fin = $fopen("vectors_tappar.txt", "r");
        if (fin == 0) begin $display("ERROR: cannot open vectors_tappar.txt"); $finish; end
        fout = $fopen("out_tappar.txt", "w");
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        if ($fscanf(fin, "%d\n", ncases) != 1) begin
            $display("ERROR: cannot read ncases"); $finish;
        end
        $display("TB tappar P_IN=%0d: %0d cases", P_IN, ncases);
        for (ci = 0; ci < ncases; ci++) begin
            // header: read one value at a time (Icarus $fscanf does
            // not accept a 16-argument call here)
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header K", ci); $finish; end
            K = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header shift", ci); $finish; end
            shift = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header W", ci); $finish; end
            W = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header H", ci); $finish; end
            H = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header pad_t", ci); $finish; end
            pad_t = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header pad_b", ci); $finish; end
            pad_b = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header pad_l", ci); $finish; end
            pad_l = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header pad_r", ci); $finish; end
            pad_r = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header stride_x", ci); $finish; end
            stride_x = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header stride_y", ci); $finish; end
            stride_y = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header sign_in", ci); $finish; end
            sign_in = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header sign_out", ci); $finish; end
            sign_out = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header act", ci); $finish; end
            act = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header slope", ci); $finish; end
            slope = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header out_w", ci); $finish; end
            out_w = tmp;
            if ($fscanf(fin, "%d", tmp) != 1) begin $display("ERROR case %0d: header out_h", ci); $finish; end
            out_h = tmp;
            for (i = 0; i < K*K; i++)
                for (c = 0; c < P_IN; c++) begin
                    if ($fscanf(fin, "%d", tmp) != 1) begin
                        $display("ERROR case %0d: cannot read coeff", ci); $finish;
                    end
                    coeff_v[i][c] = tmp;
                end
            if ($fscanf(fin, "%d", tmp) != 1) begin
                $display("ERROR case %0d: cannot read bias", ci); $finish;
            end
            bias_v = tmp;
            npos_in = W * H;
            npos_out = out_w * out_h;
            for (p = 0; p < npos_in; p++)
                for (c = 0; c < P_IN; c++) begin
                    if ($fscanf(fin, "%d", tmp) != 1) begin
                        $display("ERROR case %0d: cannot read frame", ci); $finish;
                    end
                    frame_v[p][c] = tmp;
                end
            // ---- configure the DUT ----
            cfg_wr(16'h0000, K[15:0]);
            cfg_wr(16'h0001, shift[15:0]);
            cfg_wr(16'h0002, W[15:0]);
            cfg_wr(16'h0003, H[15:0]);
            cfg_wr(16'h0004, pad_t[15:0]);
            cfg_wr(16'h0005, pad_b[15:0]);
            cfg_wr(16'h0006, pad_l[15:0]);
            cfg_wr(16'h0007, pad_r[15:0]);
            cfg_wr(16'h0008, stride_x[15:0]);
            cfg_wr(16'h0009, stride_y[15:0]);
            cfg_wr(16'h000A, 32);
            cfg_wr(16'h000B, 1);
            cfg_wr(16'h000D, out_w[15:0]);
            cfg_wr(16'h000E, out_h[15:0]);
            cfg_wr(16'h000F, (sign_in & 1) | ((sign_out & 1) << 1));
            cfg_wr(16'h0010, act[15:0]);
            cfg_wr(16'h0011, slope[15:0]);
            // Coeffs go into the MAX_K x MAX_K grid (the L1 convention):
            // vector line i is kernel (t=i/K, j=i%K); the RTL reads
            // coeff[tap*MAX_K+pj], so write it at tap*MAX_K+pj.
            // Writing it packed as i=t*K+j would make the RTL read
            // kernel row 1's coeffs as row 0's cols K..MAX_K-1 for K<5.
            for (i = 0; i < K*K; i++)
                for (c = 0; c < P_IN; c++)
                    cfg_wr(16'h0100 + ((i / K) * 5 + (i % K))*32 + c,
                           coeff_v[i][c][15:0]);
            cfg_wr(16'h0420, bias_v[15:0]);
            cfg_wr(16'h0440, bias_v[31:16]);
            // ---- run the frame ----
            cyc_cnt = 0;
            first_in_cycle = -1;
            last_out_cycle = -1;
            framing_err = 0;
            out_count = 0;
            fork
                begin : sender
                    cfg_wr(16'h000C, 16'd1);  // GO
                    for (p = 0; p < npos_in; p++) send_vec(p);
                end
                begin : collector
                    timeout_cnt = 0;
                    while (out_count < npos_out) begin
                        @(posedge clk);
                        if (out_valid && out_ready_r) begin
                            #1;
                            $fwrite(fout, "%0d\n", pixel_out);
                            // framing: out_sof on the first output,
                            // out_eol on the last column, out_last on
                            // the last output
                            exp_sof = (out_count == 0);
                            exp_eol = ((out_count % out_w) == out_w - 1);
                            exp_last = (out_count == npos_out - 1);
                            if (out_sof !== exp_sof[0] ||
                                out_eol !== exp_eol[0] ||
                                out_last !== exp_last[0]) begin
                                framing_err = framing_err + 1;
                                if (framing_err <= 5)
                                    $display("  framing err out %0d: sof=%b eol=%b last=%b (exp %0d %0d %0d)",
                                             out_count, out_sof, out_eol, out_last,
                                             exp_sof, exp_eol, exp_last);
                            end
                            out_count = out_count + 1;
                            if (out_count == npos_out)
                                last_out_cycle = cyc_cnt;
                        end
                        timeout_cnt = timeout_cnt + 1;
                        if (timeout_cnt > 200000) begin
                            $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                                     ci, out_count, npos_out);
                            $display("  dut.state=%0d tap=%0d win_valid=%b busy=%b out_valid=%b out_ready=%b stall=%b",
                                     dut.state, dut.tap, dut.win_valid, dut.busy,
                                     out_valid, out_ready_r, dut.stall);
                            $finish;
                        end
                    end
                end
            join
            $display("case %0d (K=%0d shift=%0d act=%0d sign=(%0d,%0d) %0dx%0d->%0dx%0d): %0d values, %0d cycles GO->last-out (%.2f cy/pos), framing errors=%0d",
                     ci, K, shift, act, sign_in, sign_out, W, H, out_w, out_h,
                     out_count, last_out_cycle - first_in_cycle + 1,
                     (last_out_cycle - first_in_cycle + 1) * 1.0 / npos_out,
                     framing_err);
            // wait for the DUT to go idle before the next case
            timeout_cnt = 0;
            while (dut.busy) begin
                @(posedge clk);
                timeout_cnt = timeout_cnt + 1;
                if (timeout_cnt > 5000) begin
                    $display("ERROR case %0d: DUT never went idle", ci);
                    $finish;
                end
            end
            repeat (5) @(posedge clk);
        end
        $display("TB tappar DONE");
        $fclose(fout);
        $finish;
    end
endmodule
