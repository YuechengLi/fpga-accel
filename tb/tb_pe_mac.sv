// tb_pe_mac.sv -- testbench for pe_mac (Stage 7b).
// Reads vectors_pe.txt (written by golden_pe.py gen), runs one tile-step
// per case in the specified mode, writes the TM*TN partial sums (int32,
// decimal, one per line) to out_pe.txt. golden_pe.py check compares.
// The PE is a pure GEMM engine (partial sums only); the output stage
// and K-chunk merge are verified at the pe_array level.
`timescale 1ns/1ps

module tb_pe_mac;
    parameter int TM = 16;
    parameter int TN = 8;
    parameter int TK_MAX = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        mode_dot = 1'b0;
    logic [5:0]  tk = 6'd0;
    logic [5:0]  tn = 6'd0;
    logic        first_k = 1'b0;
    logic        sign_in = 1'b0;
    logic signed [TN*32-1:0] bias_flat = '0;
    logic [TM*TK_MAX*8-1:0]  a_flat = '0;
    logic [TK_MAX*TN*16-1:0] b_flat = '0;
    logic        start = 1'b0;
    logic        busy;
    logic        done;
    logic signed [TM*TN*32-1:0] partial_flat;

    pe_mac #(.TM(TM), .TN(TN), .TK_MAX(TK_MAX)) dut (
        .clk(clk), .rst_n(rst_n),
        .mode_dot(mode_dot), .tk(tk), .tn(tn), .first_k(first_k),
        .sign_in(sign_in), .bias_flat(bias_flat),
        .a_flat(a_flat), .b_flat(b_flat),
        .start(start), .busy(busy), .done(done),
        .partial_flat(partial_flat)
    );

    int fvec, fout, code, ncases, ci;
    int tm_p, tn_p, tkmax_p, tk_v, tn_v, mode_v, firstk_v, sign_v;
    int k, m, n, tmp, cyc;

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fvec = $fopen("vectors_pe.txt", "r");
        if (fvec == 0) begin $display("FATAL: cannot open vectors_pe.txt"); $finish; end
        fout = $fopen("out_pe.txt", "w");
        code = $fscanf(fvec, "%d", ncases);
        for (ci = 0; ci < ncases; ci++) begin
            code = $fscanf(fvec, "%d %d %d %d %d %d %d %d",
                           tm_p, tn_p, tkmax_p, tk_v, tn_v, mode_v,
                           firstk_v, sign_v);
            mode_dot = mode_v[0];
            first_k  = firstk_v[0];
            sign_in  = sign_v[0];
            tk       = tk_v[5:0];
            tn       = tn_v[5:0];
            // Pack A (k-major: for k, for m) into a_flat.
            for (k = 0; k < tk_v; k++)
                for (m = 0; m < TM; m++) begin
                    code = $fscanf(fvec, "%d", tmp);
                    a_flat[(k*TM + m)*8 +: 8] = tmp[7:0];
                end
            // Pack B (k-major: for k, for n) into b_flat.
            for (k = 0; k < tk_v; k++)
                for (n = 0; n < TN; n++) begin
                    code = $fscanf(fvec, "%d", tmp);
                    b_flat[(k*TN + n)*16 +: 16] = tmp[15:0];
                end
            // Pack bias.
            for (n = 0; n < TN; n++) begin
                code = $fscanf(fvec, "%d", tmp);
                bias_flat[n*32 +: 32] = tmp;
            end
            @(negedge clk);
            start = 1'b1;
            @(posedge clk);
            #1;
            start = 1'b0;
            // Wait for done, with a cycle counter (cross-check vs the
            // cycle model: outer = tk cycles, dot = tn*(tk/8) cycles).
            cyc = 0;
            while (!done) begin
                @(posedge clk);
                #1;
                cyc = cyc + 1;
                if (cyc > 1000) begin
                    $display("ERROR case %0d: timeout waiting for done", ci);
                    $finish;
                end
            end
            $display("case %0d (mode_dot=%0d tk=%0d tn=%0d first_k=%0d sign_in=%0d): done after %0d cycles (start->done)", ci, mode_v, tk_v, tn_v, firstk_v, sign_v, cyc);
            for (m = 0; m < TM; m++)
                for (n = 0; n < TN; n++)
                    $fwrite(fout, "%0d\n",
                            $signed(partial_flat[(m*TN + n)*32 +: 32]));
            repeat (2) @(posedge clk);
        end
        $display("PE TB DONE: %0d cases", ncases);
        $fclose(fout);
        $fclose(fvec);
        $finish;
    end
endmodule
