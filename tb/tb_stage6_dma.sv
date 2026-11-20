// tb_stage6_dma.sv -- Stage 6 DMA -> conv2d_mc integration test.
//
// The Stage 5 DMA (weight_dma.sv) was verified standalone (8/8 runs
// PASS) against its own documented addressing, and the Stage 3
// conv2d_mc was verified with coefficients written directly via the
// config bus at the 11x11 grid positions (i*11+j). This test is the
// FIRST check that the two agree: it loads a REAL model slice
// (SRCNN layer 1, 9x9, output channel 5, bias included) through the
// AXI-MM weight DMA into the conv2d_mc config bus, streams the
// input, and checks the output against the Stage 5 functional
// simulator (conv2d_tiled for that single output channel).
//
// The DMA's documented addressing is "lane l at dst + l*stride, tap
// k at +k" with taps = K*K (contiguous within a lane). The MC's
// coefficient memory is indexed by the 11x11 grid (i*11+j). If those
// two do not agree for K < 11, this test FAILS -- and that is a real
// Stage 5 emitter/DMA bug, not a testbench bug (the direct-config
// path already proves the MC is correct when the coefficients are at
// the grid positions).
//
// Vector file: tb/vectors_stage6_dma.txt (sw/gen_stage6_dma.py).
//   # K=.. shift=.. w=.. h=.. pt=.. pb=.. pl=.. pr=.. sx=.. sy=..
//   #   cin=.. cout=.. ow=.. oh=.. sgn_in=.. sgn_out=..
//   #   fuse_act=.. act_slope=.. bias=..
//   # ncoeff=..
//   C <idx> <hex>          (coefficients, KxK row-major, (o=ch,c=0))
//   # nin=..
//   I <val>                (input pixels, raster order)
//   # nout=..
//   E <val>                (expected output pixels, raster order)
// Run:
//   iverilog -g2012 -o tb_s6dma.vvp ../rtl/weight_dma.sv \
//       ../rtl/conv2d_mc.sv tb_stage6_dma.sv && vvp tb_s6dma.vvp
`default_nettype none
`timescale 1ns/1ps

module tb_stage6_dma;
    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk; // 100 MHz

    // ---------- AXI-Lite slave (DMA programming) ----------
    logic        s_axil_awvalid = 0;
    logic        s_axil_awready;
    logic [5:0]  s_axil_awaddr = 0;
    logic        s_axil_wvalid = 0;
    logic        s_axil_wready;
    logic [31:0] s_axil_wdata = 0;
    logic        s_axil_bvalid;
    logic        s_axil_bready = 0;
    logic [1:0]  s_axil_bresp;
    logic        s_axil_arvalid = 0;
    logic        s_axil_arready;
    logic [5:0]  s_axil_araddr = 0;
    logic        s_axil_rvalid;
    logic        s_axil_rready = 0;
    logic [31:0] s_axil_rdata;
    logic [1:0]  s_axil_rresp;

    // ---------- AXI-MM master (DMA -> DRAM) ----------
    logic [31:0] m_axi_araddr;
    logic [7:0]  m_axi_arlen;
    logic [2:0]  m_axi_arsize;
    logic [1:0]  m_axi_arburst;
    logic        m_axi_arvalid;
    logic        m_axi_arready;
    logic [63:0] m_axi_rdata;
    logic        m_axi_rlast;
    logic        m_axi_rvalid;
    logic        m_axi_rready;

    // ---------- DMA config-bus output ----------
    logic        dma_cfg_wr_en;
    logic [15:0] dma_cfg_wr_addr;
    logic [15:0] dma_cfg_wr_data;
    logic        dma_done, dma_busy;

    // ---------- MC config bus (muxed: DMA while busy, else TB) ----------
    logic        tb_cfg_valid = 1'b0, tb_cfg_we = 1'b0;
    logic [15:0] tb_cfg_addr = 16'd0;
    logic [15:0] tb_cfg_data = 16'd0;
    logic        mc_cfg_valid, mc_cfg_we;
    logic [15:0] mc_cfg_addr, mc_cfg_data;

    assign mc_cfg_valid = dma_busy ? dma_cfg_wr_en : tb_cfg_valid;
    assign mc_cfg_we    = dma_busy ? 1'b1            : tb_cfg_we;
    assign mc_cfg_addr  = dma_busy ? dma_cfg_wr_addr : tb_cfg_addr;
    assign mc_cfg_data  = dma_busy ? dma_cfg_wr_data : tb_cfg_data;

    // ---------- MC stream ----------
    logic [7:0]  pixel_in = 8'd0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;
    logic        mc_busy;

    weight_dma dma (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_awaddr(s_axil_awaddr),
        .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
        .s_axil_wdata(s_axil_wdata),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
        .s_axil_bresp(s_axil_bresp),
        .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_araddr(s_axil_araddr),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .cfg_wr_en(dma_cfg_wr_en), .cfg_wr_addr(dma_cfg_wr_addr),
        .cfg_wr_data(dma_cfg_wr_data), .cfg_wr_full(1'b0),
        .done(dma_done), .busy(dma_busy)
    );

    conv2d_mc #(.C_PAR(4), .C_IN_MAX(64), .C_OUT_MAX(64),
                .TILE_W(32), .TILE_H(32)) mc (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(mc_cfg_valid), .cfg_we(mc_cfg_we),
        .cfg_addr(mc_cfg_addr), .cfg_data(mc_cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ready(out_ready_r), .busy(mc_busy)
    );

    // ---------- AXI-MM slave (ideal DRAM, from tb_weight_dma.sv) ----------
    logic [63:0] dram [0:1023];
    logic [31:0] slave_araddr_q;
    logic [7:0]  slave_beats_q;
    logic [7:0]  slave_beat_idx_q;
    logic [2:0]  slave_lat_q;
    logic        slave_active_q;

    assign m_axi_arready = 1'b1;
    assign m_axi_rvalid  = slave_active_q && (slave_lat_q == 3'd0)
                           && (slave_beats_q != 8'd0);
    assign m_axi_rlast   = (slave_beats_q == 8'd1);
    assign m_axi_rdata   = dram[(slave_araddr_q[12:3]
                                 + slave_beat_idx_q) % 1024];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            slave_araddr_q   <= 32'd0;
            slave_beats_q    <= 8'd0;
            slave_beat_idx_q <= 8'd0;
            slave_lat_q      <= 3'd0;
            slave_active_q   <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                slave_araddr_q   <= m_axi_araddr;
                slave_beats_q    <= m_axi_arlen + 8'd1;
                slave_beat_idx_q <= 8'd0;
                slave_lat_q      <= 3'd4;
                slave_active_q   <= 1'b1;
            end else if (slave_active_q) begin
                if (slave_lat_q != 3'd0)
                    slave_lat_q <= slave_lat_q - 3'd1;
                else if (m_axi_rvalid && m_axi_rready) begin
                    slave_beat_idx_q <= slave_beat_idx_q + 8'd1;
                    slave_beats_q    <= slave_beats_q - 8'd1;
                    if (m_axi_rlast)
                        slave_active_q <= 1'b0;
                end
            end
        end
    end

    // ---------- TB tasks ----------
    task axil_write(input [5:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            s_axil_awaddr  <= addr;
            s_axil_wdata   <= data;
            s_axil_awvalid <= 1'b1;
            s_axil_wvalid  <= 1'b1;
            @(posedge clk);
            s_axil_awvalid <= 1'b0;
            s_axil_wvalid  <= 1'b0;
            s_axil_bready  <= 1'b1;
            @(posedge clk);
            s_axil_bready  <= 1'b0;
        end
    endtask

    task cfg_wr(input logic [15:0] a, input logic [15:0] d);
        begin
            @(negedge clk);
            tb_cfg_valid = 1'b1; tb_cfg_we = 1'b1;
            tb_cfg_addr = a; tb_cfg_data = d;
            @(negedge clk);
            tb_cfg_valid = 1'b0; tb_cfg_we = 1'b0;
        end
    endtask

    task wait_not_busy;
        int t;
        begin
            repeat (10) @(posedge clk);
            t = 0;
            while (mc_busy && t < 200000) begin
                @(posedge clk);
                t++;
            end
            if (mc_busy) begin
                $display("ERROR: timeout waiting for mc busy==0");
                $finish;
            end
        end
    endtask

    task send_pixel(input logic [7:0] px, input logic sof);
        begin
            @(negedge clk);
            pixel_in = px; in_valid = 1'b1; in_sof = sof;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            pixel_in = '0; in_valid = 1'b0; in_sof = 1'b0;
        end
    endtask

    int fvec, code;
    int k, sh, w, h, pt, pb, pl, pr, sx, sy, cin, cout, ow, oh;
    int sgn_in, sgn_out, fuse_act, act_slope, bias_v;
    int ncoeff, nin, nout, taps;
    int i, p, idx, val;
    int img [0:1023];
    int exp_out [0:1023];
    int n_out, errors, cycles;
    int wait_i, start_cycle, done_cycle, cycle_cnt = 0;

    always @(posedge clk) begin
        if (rst_n)
            cycle_cnt <= cycle_cnt + 1;
    end

    initial begin
        fvec = $fopen("vectors_stage6_dma.txt", "r");
        if (fvec == 0) begin
            $display("FATAL: cannot open vectors_stage6_dma.txt");
            $finish;
        end
        code = $fscanf(fvec, "# K=%d shift=%d w=%d h=%d pt=%d pb=%d pl=%d pr=%d sx=%d sy=%d cin=%d cout=%d ow=%d oh=%d sgn_in=%d sgn_out=%d fuse_act=%d act_slope=%d bias=%d taps=%d\n",
                       k, sh, w, h, pt, pb, pl, pr, sx, sy, cin, cout,
                       ow, oh, sgn_in, sgn_out, fuse_act, act_slope,
                       bias_v, taps);
        code = $fscanf(fvec, "# ncoeff=%d\n", ncoeff);
        // Coefficients -> DRAM (4 per 64-bit beat, little-endian).
        for (i = 0; i < 1024; i++) dram[i] = 64'd0;
        for (i = 0; i < ncoeff; i++) begin
            code = $fscanf(fvec, "C %d %x\n", idx, val);
            dram[idx / 4][(idx % 4) * 16 +: 16] = val[15:0];
        end
        code = $fscanf(fvec, "# nin=%d\n", nin);
        for (i = 0; i < nin; i++) begin
            code = $fscanf(fvec, "I %d\n", val);
            img[i] = val;
        end
        code = $fscanf(fvec, "# nout=%d\n", nout);
        for (i = 0; i < nout; i++) begin
            code = $fscanf(fvec, "E %d\n", val);
            exp_out[i] = val;
        end
        $fclose(fvec);
        $display("vectors: K=%0d shift=%0d %0dx%0d cin=%0d cout=%0d out=%0dx%0d bias=%0d ncoeff=%0d taps=%0d nin=%0d nout=%0d",
                 k, sh, w, h, cin, cout, ow, oh, bias_v, ncoeff, taps, nin, nout);

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ---- program the MC control registers (direct config) ----
        cfg_wr(16'h0000, k[15:0]);
        cfg_wr(16'h0001, sh[15:0]);
        cfg_wr(16'h0002, w[15:0]);
        cfg_wr(16'h0003, h[15:0]);
        cfg_wr(16'h0004, pt[15:0]);
        cfg_wr(16'h0005, pb[15:0]);
        cfg_wr(16'h0006, pl[15:0]);
        cfg_wr(16'h0007, pr[15:0]);
        cfg_wr(16'h0008, sx[15:0]);
        cfg_wr(16'h0009, sy[15:0]);
        cfg_wr(16'h000A, cin[15:0]);
        cfg_wr(16'h000B, cout[15:0]);
        cfg_wr(16'h000D, ow[15:0]);
        cfg_wr(16'h000E, oh[15:0]);
        cfg_wr(16'h000F, {14'd0, (sgn_out != 0), (sgn_in != 0)});
        cfg_wr(16'h0010, fuse_act[15:0]);
        cfg_wr(16'h0011, act_slope[15:0]);
        cfg_wr(16'h0012, bias_v[15:0]);
        cfg_wr(16'h0013, bias_v[31:16]);

        // ---- DMA the coefficients into the MC (dst 0x100, taps as
        // in the vector, stride 128) -- the Stage 5 emitter's
        // programming (taps=121 after the Stage 6 grid fix). ----
        axil_write(6'h00, 32'd0);        // src (DRAM offset 0)
        axil_write(6'h04, ncoeff);       // count
        axil_write(6'h08, 32'h100);      // dst (MC coeff base)
        axil_write(6'h0c, taps);         // taps
        axil_write(6'h10, 128);          // stride
        start_cycle = cycle_cnt;
        axil_write(6'h14, 32'd1);        // ctrl: start
        done_cycle = 0;
        for (wait_i = 0; wait_i < 100000; wait_i++) begin
            @(posedge clk);
            if (dma_done) begin
                done_cycle = cycle_cnt;
                wait_i = 100000;
            end
        end
        if (done_cycle == 0) begin
            $display("FATAL: timeout waiting for DMA done");
            $finish;
        end
        $display("DMA done: %0d cycles for %0d coeffs",
                 done_cycle - start_cycle, ncoeff);
        repeat (4) @(posedge clk);

        // ---- stream the input tile ----
        for (p = 0; p < w * h; p++)
            send_pixel(img[p][7:0], (p == 0));

        // ---- GO + collect ----
        n_out = 0; errors = 0; cycles = 0;
        fork
            begin : driver
                cfg_wr(16'h000C, 16'd1);  // GO
                wait_not_busy;
            end
            begin : collector
                while (n_out < nout) begin
                    @(posedge clk);
                    cycles++;
                    if (cycles > 5000000) begin
                        $display("ERROR: timeout, got %0d/%0d outputs",
                                 n_out, nout);
                        $finish;
                    end
                    if (out_valid && out_ready_r) begin
                        if (pixel_out !== exp_out[n_out][7:0]) begin
                            if (errors < 10)
                                $display("MISMATCH out[%0d]: got %0d exp %0d",
                                         n_out, pixel_out, exp_out[n_out]);
                            errors++;
                        end
                        if (n_out == 0 && !out_sof) begin
                            $display("ERROR: first output missing out_sof");
                            errors++;
                        end
                        if ((n_out % ow == ow-1) && !out_eol) begin
                            $display("ERROR: output %0d missing out_eol", n_out);
                            errors++;
                        end
                        if (n_out == nout-1 && !out_last) begin
                            $display("ERROR: last output missing out_last");
                            errors++;
                        end
                        n_out++;
                    end
                end
            end
        join

        if (errors == 0)
            $display("DMA INTEGRATION CHECK PASSED (%0d outputs, exact)", n_out);
        else
            $display("DMA INTEGRATION CHECK FAILED (%0d errors / %0d outputs)",
                     errors, n_out);
        $finish;
    end
endmodule
`default_nettype wire
