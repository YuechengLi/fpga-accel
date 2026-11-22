// tb_stage6_sched.sv -- Stage 6 scheduler-path check (DMA -> scheduler
// bank -> unified engine, dense mode).
//
// A REAL model slice: SRCNN layer 1 (9x9, 1 -> 64, ReLU), output
// channel 5, on the 24x24 Stage 6 test input. The coefficients are
// loaded through the AXI-MM weight DMA into the scheduler's
// ping-pong bank A (the full 121-entry grid), then the scheduler
// streams them to the unified engine (dense mode) and runs the layer.
//
// The dense path has NO bias register (a documented follow-up), so
// the reference is the functional simulator with the bias FORCED to
// 0 for this channel (stated in the vector file).
//
// This is the first check of the scheduler path with DMA-loaded
// weights (Stage 4 verified the scheduler with directly-written
// banks; Stage 5 verified the DMA standalone). The MC path (all
// Stage 6 model layers) is verified separately by tb_stage6_mc.sv
// and tb_stage6_dma.sv.
//
// Vector file: tb/vectors_stage6_sched.txt (sw/vision/gen_stage6_sched.py).
// Run:
//   iverilog -g2012 -o tb_s6sched.vvp ../rtl/common/weight_dma.sv \
//       ../rtl/common/layer_scheduler.sv ../rtl/common/unified_engine.sv \
//       ../rtl/vision/conv2d_dense.sv ../rtl/vision/conv2d_dw.sv tb_stage6_sched.sv \
//       && vvp tb_s6sched.vvp
`default_nettype none
`timescale 1ns/1ps

module tb_stage6_sched;
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

    // ---------- scheduler config bus (muxed: DMA while busy, else TB) ----------
    logic        tb_cfg_valid = 1'b0, tb_cfg_we = 1'b0;
    logic [15:0] tb_cfg_addr = 16'd0;
    logic [15:0] tb_cfg_data = 16'd0;
    logic        sc_cfg_valid, sc_cfg_we;
    logic [15:0] sc_cfg_addr, sc_cfg_data;
    logic        sched_wfull;

    assign sc_cfg_valid = dma_busy ? dma_cfg_wr_en : tb_cfg_valid;
    assign sc_cfg_we    = dma_busy ? 1'b1            : tb_cfg_we;
    assign sc_cfg_addr  = dma_busy ? dma_cfg_wr_addr : tb_cfg_addr;
    assign sc_cfg_data  = dma_busy ? dma_cfg_wr_data : tb_cfg_data;

    // ---------- scheduler <-> engine ----------
    logic        eng_cfg_valid, eng_cfg_we;
    logic [15:0] eng_cfg_addr, eng_cfg_data;
    logic        eng_busy, eng_start;
    logic        sched_done, sched_busy;

    // ---------- engine stream ----------
    logic [7:0]  pixel_in = 8'd0;
    logic        in_valid = 1'b0, in_sof = 1'b0;
    logic        in_ready;
    logic [7:0]  pixel_out;
    logic        out_valid;
    logic        out_sof, out_eol, out_last;
    logic        out_ready_r = 1'b1;

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
        .cfg_wr_data(dma_cfg_wr_data), .cfg_wr_full(sched_wfull),
        .done(dma_done), .busy(dma_busy)
    );

    layer_scheduler #(.NLAYER(4), .BANK_COEFF(289)) sched (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(sc_cfg_valid), .cfg_we(sc_cfg_we),
        .cfg_addr(sc_cfg_addr), .cfg_data(sc_cfg_data),
        .eng_cfg_valid(eng_cfg_valid), .eng_cfg_we(eng_cfg_we),
        .eng_cfg_addr(eng_cfg_addr), .eng_cfg_data(eng_cfg_data),
        .eng_busy(eng_busy), .eng_start(eng_start),
        .done(sched_done), .busy(sched_busy), .wfull(sched_wfull)
    );

    unified_engine eng (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(eng_cfg_valid), .cfg_we(eng_cfg_we),
        .cfg_addr(eng_cfg_addr), .cfg_data(eng_cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid), .out_sof(out_sof),
        .out_eol(out_eol), .out_last(out_last), .out_ready(out_ready_r),
        .busy(eng_busy)
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

    task scfg_write(input [15:0] a, input [15:0] d);
        begin
            @(posedge clk); #1;
            tb_cfg_valid = 1'b1; tb_cfg_we = 1'b1;
            tb_cfg_addr = a; tb_cfg_data = d;
            @(posedge clk); #1;
            tb_cfg_valid = 1'b0; tb_cfg_we = 1'b0;
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
    int k, sh, w, h, pt, pb, pl, pr, sx, sy, ow, oh;
    int sgn_in, sgn_out, fuse_act, act_slope, bias_v;
    int ncoeff, nin, nout;
    int i, p, idx, val;
    int img [0:1023];
    int exp_out [0:1023];
    int n_out, errors, cycles;
    int wait_i;
    integer start_pulse_cnt = 0;
    always @(posedge clk) if (eng_start) start_pulse_cnt++;

    initial begin
        fvec = $fopen("vectors_stage6_sched.txt", "r");
        if (fvec == 0) begin
            $display("FATAL: cannot open vectors_stage6_sched.txt");
            $finish;
        end
        code = $fscanf(fvec, "# K=%d shift=%d w=%d h=%d pt=%d pb=%d pl=%d pr=%d sx=%d sy=%d ow=%d oh=%d sgn_in=%d sgn_out=%d fuse_act=%d act_slope=%d bias=%d\n",
                       k, sh, w, h, pt, pb, pl, pr, sx, sy, ow, oh,
                       sgn_in, sgn_out, fuse_act, act_slope, bias_v);
        code = $fscanf(fvec, "# ncoeff=%d\n", ncoeff);
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
        $display("vectors: K=%0d shift=%0d %0dx%0d out=%0dx%0d bias=%0d (forced 0) ncoeff=%0d nin=%0d nout=%0d",
                 k, sh, w, h, ow, oh, bias_v, ncoeff, nin, nout);
        if (bias_v != 0) begin
            $display("FATAL: scheduler-path reference requires bias=0");
            $finish;
        end

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ---- program the scheduler layer descriptor (layer 0) ----
        scfg_write(16'h0002, 16'd0);       // layer select 0
        scfg_write(16'h0003, 16'd0);       // mode 0 = dense
        scfg_write(16'h0004, k[15:0]);     // K
        scfg_write(16'h0005, sh[15:0]);    // shift
        scfg_write(16'h0006, w[15:0]);     // W
        scfg_write(16'h0007, h[15:0]);     // H
        scfg_write(16'h0008, pt[15:0]);
        scfg_write(16'h0009, pb[15:0]);
        scfg_write(16'h000A, pl[15:0]);
        scfg_write(16'h000B, pr[15:0]);
        scfg_write(16'h000C, sx[15:0]);
        scfg_write(16'h000D, sy[15:0]);
        scfg_write(16'h000E, {14'd0, (sgn_out != 0), (sgn_in != 0)});
        scfg_write(16'h000F, fuse_act[15:0]);
        scfg_write(16'h0010, act_slope[15:0]);
        scfg_write(16'h0011, 16'd1);       // dilation 1
        scfg_write(16'h0012, ncoeff[15:0]);
        scfg_write(16'h0001, 16'd1);       // 1 layer

        // ---- DMA the grid into bank A (dst 0x0200, taps 121) ----
        axil_write(6'h00, 32'd0);          // src
        axil_write(6'h04, ncoeff);         // count
        axil_write(6'h08, 32'h200);        // dst (bank A)
        axil_write(6'h0c, ncoeff);         // taps (121 = grid)
        axil_write(6'h10, 1);              // stride (contiguous bank)
        axil_write(6'h14, 32'd1);          // ctrl: start
        for (wait_i = 0; wait_i < 100000; wait_i++) begin
            @(posedge clk);
            if (dma_done) wait_i = 100000;
        end
        if (!dma_done) begin
            $display("FATAL: timeout waiting for DMA done");
            $finish;
        end
        $display("DMA done: bank A loaded (%0d coeffs)", ncoeff);
        repeat (4) @(posedge clk);

        // ---- start the scheduler; stream the input on eng_start ----
        n_out = 0; errors = 0; cycles = 0;
        fork
            begin : driver
                scfg_write(16'h0000, 16'd1);  // ctrl: start
                // Wait for the eng_start pulse, then stream the frame.
                wait (start_pulse_cnt > 0);
                for (p = 0; p < w * h; p++)
                    send_pixel(img[p][7:0], (p == 0));
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
            $display("SCHEDULER-PATH CHECK PASSED (%0d outputs, exact, bias=0)", n_out);
        else
            $display("SCHEDULER-PATH CHECK FAILED (%0d errors / %0d outputs)",
                     errors, n_out);
        $finish;
    end
endmodule
`default_nettype wire
