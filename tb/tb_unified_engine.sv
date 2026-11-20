//=============================================================================
// tb_unified_engine.sv -- Stage 4 testbench for unified_engine.sv.
//
// Reads vectors_unified.txt (written by golden_unified.py gen):
//   ncases
//   per case:
//     mode k shift w h pt pb pl pr sx sy sign_in sign_out fuse_act
//     act_slope d
//     ncoeff
//     <coeffs, space-separated>          (skipped when ncoeff == 0)
//     <img, space-separated, w*h values>
// For each case the TB writes the engine mode FIRST (0x0C, so the
// following writes route to the newly-active submodule), then the
// control registers (shared map), then the coefficients (0x100+i)
// -- except when ncoeff == 0, where the coefficients are NOT reloaded
// (tests that the inactive submodule preserved them across the switch).
// It sends the frame with random in_valid gaps, collects the output
// with random out_ready backpressure, checks framing, and writes
// out_unified.txt. No reset between cases (the engine switches mode
// between frames).
//=============================================================================
`timescale 1ns/1ps

module tb_unified_engine;
    logic clk = 0;
    logic rst_n = 0;
    logic cfg_valid = 0, cfg_we = 0;
    logic [15:0] cfg_addr = 0, cfg_data = 0;
    logic [7:0] pixel_in = 0;
    logic in_valid = 0, in_sof = 0, in_ready;
    logic [7:0] pixel_out;
    logic out_valid, out_sof, out_eol, out_last, out_ready = 0;
    logic busy;

    unified_engine dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_ready(out_ready),
        .busy(busy)
    );

    always #5 clk = ~clk;

    integer vf, of, rc, i, ci, ncases;
    integer mode, k, shift, w, h, pt, pb, pl, pr, sx, sy;
    integer sign_in, sign_out, fuse_act, act_slope, d, ncoeff;
    integer coeff [0:288];
    integer px_in [0:4095];
    integer out_count, framing_err, exp_count;
    integer seed = 777;
    integer accepted;
    integer ow, oh;

    task cfg_write(input [15:0] a, input [15:0] d);
        begin
            @(posedge clk); #1;
            cfg_valid = 1; cfg_we = 1; cfg_addr = a; cfg_data = d;
            @(posedge clk); #1;
            cfg_valid = 0; cfg_we = 0;
        end
    endtask

    // Collect outputs in the background while a case runs.
    integer out_idx;
    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            if (out_idx == 0 && !out_sof) framing_err = framing_err + 1;
            if (out_eol && ((out_idx % ow) != ow - 1))
                framing_err = framing_err + 1;
            if (!out_eol && ((out_idx % ow) == ow - 1))
                framing_err = framing_err + 1;
            if (out_last && (out_idx != exp_count - 1))
                framing_err = framing_err + 1;
            if (!out_last && (out_idx == exp_count - 1))
                framing_err = framing_err + 1;
            $fwrite(of, "%0d\n", pixel_out);
            out_idx = out_idx + 1;
            out_count = out_count + 1;
        end
    end

    // Random out_ready.
    always @(posedge clk) begin
        out_ready <= ($random(seed) % 100) < 80;
    end

    initial begin
        vf = $fopen("vectors_unified.txt", "r");
        if (vf == 0) begin $display("cannot open vectors_unified.txt"); $finish; end
        of = $fopen("out_unified.txt", "w");
        rc = $fscanf(vf, "%d", ncases);
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        for (ci = 0; ci < ncases; ci = ci + 1) begin
            rc = $fscanf(vf, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                         mode, k, shift, w, h, pt, pb, pl, pr, sx, sy,
                         sign_in, sign_out, fuse_act, act_slope, d);
            if (rc != 16) begin
                $display("bad header rc=%0d case %0d", rc, ci);
                $finish;
            end
            rc = $fscanf(vf, "%d", ncoeff);
            for (i = 0; i < ncoeff; i = i + 1)
                rc = $fscanf(vf, "%d", coeff[i]);
            for (i = 0; i < w * h; i = i + 1)
                rc = $fscanf(vf, "%d", px_in[i]);
            // Output size (same formula as the goldens).
            ow = (w + pl + pr - k) / sx + 1;
            oh = (h + pt + pb - k) / sy + 1;
            // For DW-dilated the span is (k-1)*d+1, not k; recompute.
            if (mode == 4) begin
                ow = (w + pl + pr - ((k - 1) * d + 1)) / sx + 1;
                oh = (h + pt + pb - ((k - 1) * d + 1)) / sy + 1;
            end
            // For DW-strip the span is k in one axis and 1 in the other.
            if (mode == 2) begin
                ow = (w + pl + pr - k) / sx + 1;
                oh = (h + pt + pb - 1) / sy + 1;
            end
            if (mode == 3) begin
                ow = (w + pl + pr - 1) / sx + 1;
                oh = (h + pt + pb - k) / sy + 1;
            end
            exp_count = ow * oh;
            // Config: mode FIRST (routes the rest to the new submodule).
            cfg_write(16'h000C, mode[15:0]);
            cfg_write(16'h0000, k[15:0]);
            cfg_write(16'h0001, shift[15:0]);
            cfg_write(16'h0002, w[15:0]);
            cfg_write(16'h0003, h[15:0]);
            cfg_write(16'h0004, pt[15:0]);
            cfg_write(16'h0005, pb[15:0]);
            cfg_write(16'h0006, pl[15:0]);
            cfg_write(16'h0007, pr[15:0]);
            cfg_write(16'h0008, sx[15:0]);
            cfg_write(16'h0009, sy[15:0]);
            // 0x0B: dense/FC = act slope, DW = dilation.
            if (mode == 0 || mode == 5)
                cfg_write(16'h000B, act_slope[15:0]);
            else
                cfg_write(16'h000B, d[15:0]);
            // 0x0A: dense/FC = fused act. (DW mode is set by the mode
            // write; do not write 0x0A by hand in a DW mode.)
            if (mode == 0 || mode == 5)
                cfg_write(16'h000A, fuse_act[15:0]);
            cfg_write(16'h000F, ((sign_out << 1) | sign_in));
            // Coefficients (skipped when ncoeff == 0: preservation test).
            for (i = 0; i < ncoeff; i = i + 1)
                cfg_write(16'h0100 + i[15:0], coeff[i][15:0]);
            // Send the frame.
            out_count = 0; out_idx = 0; framing_err = 0;
            for (i = 0; i < w * h; i = i + 1) begin
                while (($random(seed) % 100) < 30) begin
                    @(posedge clk); #1; in_valid = 0;
                end
                @(posedge clk); #1;
                pixel_in = px_in[i][7:0];
                in_valid = 1;
                in_sof = (i == 0);
                accepted = 0;
                while (!accepted) begin
                    @(posedge clk);
                    if (in_ready) accepted = 1;
                    #1;
                end
                in_valid = 0; in_sof = 0;
            end
            // Wait for all outputs.
            begin : wait_out
                integer t;
                t = 0;
                while (out_count < exp_count && t < 200000) begin
                    @(posedge clk); t = t + 1;
                end
                if (out_count < exp_count)
                    $display("ERROR case %0d: timeout, got %0d/%0d outputs",
                             ci, out_count, exp_count);
            end
            $display("case %0d (mode=%0d %0dx%0d -> %0dx%0d): %0d outputs, framing errors=%0d",
                     ci, mode, w, h, ow, oh, out_count, framing_err);
        end
        $display("TB unified_engine DONE");
        $fclose(of);
        $finish;
    end
endmodule
