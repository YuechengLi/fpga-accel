// tb_adder_tree.sv -- smoke test for adder_tree (Stage 7b).
// Feeds random signed values into three instances (N=81, 64, 25),
// checks the registered sum 4 cycles later against a TB-side sum.
`timescale 1ns/1ps

module tb_adder_tree;
    logic clk, rst_n, stall;
    always #5 clk = ~clk;

    logic signed [24:0] in160 [0:159];
    logic signed [24:0] in81 [0:80];
    logic signed [24:0] in64 [0:63];
    logic signed [24:0] in25 [0:24];
    logic signed [32:0] out160;
    logic signed [32:0] out81, out64, out25;

    // Flat packs for the tree's in_flat port (Yosys 0.52 rejects
    // unpacked array ports); element i in bits [i*25 +: 25].
    logic signed [160*25-1:0] in160_flat;
    logic signed [81*25-1:0] in81_flat;
    logic signed [64*25-1:0] in64_flat;
    logic signed [25*25-1:0] in25_flat;
    genvar gi;
    generate
        for (gi = 0; gi < 160; gi++) assign in160_flat[gi*25 +: 25] = in160[gi];
        for (gi = 0; gi < 81; gi++) assign in81_flat[gi*25 +: 25] = in81[gi];
        for (gi = 0; gi < 64; gi++) assign in64_flat[gi*25 +: 25] = in64[gi];
        for (gi = 0; gi < 25; gi++) assign in25_flat[gi*25 +: 25] = in25[gi];
    endgenerate

    adder_tree #(.N(160), .W(25)) u160 (
        .clk(clk), .rst_n(rst_n), .stall(stall), .in_flat(in160_flat), .out(out160));

    adder_tree #(.N(81), .W(25)) u81 (
        .clk(clk), .rst_n(rst_n), .stall(stall), .in_flat(in81_flat), .out(out81));
    adder_tree #(.N(64), .W(25)) u64 (
        .clk(clk), .rst_n(rst_n), .stall(stall), .in_flat(in64_flat), .out(out64));
    adder_tree #(.N(25), .W(25)) u25 (
        .clk(clk), .rst_n(rst_n), .stall(stall), .in_flat(in25_flat), .out(out25));

    int seed, t, i, errors;
    logic signed [32:0] exp160, exp81, exp64, exp25;
    logic signed [32:0] q160 [0:3];
    logic signed [32:0] q81 [0:3];
    logic signed [32:0] q64 [0:3];
    logic signed [32:0] q25 [0:3];

    initial begin
        clk = 0; rst_n = 0; stall = 0; seed = 12345; errors = 0;
        for (i = 0; i < 160; i++) in160[i] = 0;
        for (i = 0; i < 81; i++) in81[i] = 0;
        for (i = 0; i < 64; i++) in64[i] = 0;
        for (i = 0; i < 25; i++) in25[i] = 0;
        for (i = 0; i < 4; i++) begin
            q160[i] = 0; q81[i] = 0; q64[i] = 0; q25[i] = 0;
        end
        repeat (4) @(posedge clk);
        rst_n = 1;
        // 200 random vectors; check the value that entered 4 cycles ago.
        for (t = 0; t < 200; t++) begin
            exp160 = 0; exp81 = 0; exp64 = 0; exp25 = 0;
            for (i = 0; i < 160; i++) begin
                in160[i] = $random(seed) % 200 - 100;
                exp160 = exp160 + in160[i];
            end
            for (i = 0; i < 81; i++) begin
                in81[i] = $random(seed) % 200 - 100;
                exp81 = exp81 + in81[i];
            end
            for (i = 0; i < 64; i++) begin
                in64[i] = $random(seed) % 200 - 100;
                exp64 = exp64 + in64[i];
            end
            for (i = 0; i < 25; i++) begin
                in25[i] = $random(seed) % 200 - 100;
                exp25 = exp25 + in25[i];
            end
            // Shift the expected-value pipeline (4 stages).
            q160[3] = q160[2]; q160[2] = q160[1]; q160[1] = q160[0]; q160[0] = exp160;
            q81[3] = q81[2]; q81[2] = q81[1]; q81[1] = q81[0]; q81[0] = exp81;
            q64[3] = q64[2]; q64[2] = q64[1]; q64[1] = q64[0]; q64[0] = exp64;
            q25[3] = q25[2]; q25[2] = q25[1]; q25[1] = q25[0]; q25[0] = exp25;
            @(posedge clk);
            #1;
            if (t >= 4) begin
                if (out160 !== q160[3]) begin
                    $display("FAIL t=%0d N=160: expected %0d got %0d", t, q160[3], out160);
                    errors = errors + 1;
                end
                if (out81 !== q81[3]) begin
                    $display("FAIL t=%0d N=81: expected %0d got %0d", t, q81[3], out81);
                    errors = errors + 1;
                end
                if (out64 !== q64[3]) begin
                    $display("FAIL t=%0d N=64: expected %0d got %0d", t, q64[3], out64);
                    errors = errors + 1;
                end
                if (out25 !== q25[3]) begin
                    $display("FAIL t=%0d N=25: expected %0d got %0d", t, q25[3], out25);
                    errors = errors + 1;
                end
            end
        end
        if (errors == 0) $display("ADDER TREE SMOKE TEST PASSED (200 vectors x 4 sizes)");
        else $display("ADDER TREE SMOKE TEST FAILED (%0d errors)", errors);
        $finish;
    end
endmodule
