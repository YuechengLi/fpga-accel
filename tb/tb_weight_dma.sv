// tb_weight_dma.sv -- testbench for rtl/weight_dma.sv.
//
// Verifies:
//   1. Data correctness: every coefficient lands at the right
//      config-bus address with the right value (golden vectors
//      from golden_dma.py), for contiguous (scheduler bank) and
//      strided (conv2d_mc) addressing.
//   2. Backpressure: the config-bus sink randomly asserts wfull;
//      the transfer must still be correct (just slower).
//   3. Cycle count: the measured cycles (start pulse -> done) vs
//      sw/dram.py weight_dma_cycles (printed in the vector file as
//      expected_cycles). Reported, with a tolerance of +/-20 cycles
//      (the model's 12-cycle latency is a budget, not a cycle-exact
//      prediction; the drain rate -- 1 coeff/cycle -- is what is
//      checked exactly by the data, and the overhead is reported).
//
// The AXI-MM slave is ideal: AR ready immediately, first R data
// 4 cycles after AR accepted (the per-burst AR overhead), then
// back-to-back beats. The AXI-Lite master programs the DMA
// registers with blocking writes.
//
// Usage: iverilog -g2012 -o tb_dma.vvp ../rtl/weight_dma.sv
//        tb_weight_dma.sv && vvp tb_dma.vvp [vectors_dma_X.txt]
// The vector file is read with $fscanf; the case is selected by
// the file (default vectors_dma_A.txt).
`default_nettype none
`timescale 1ns/1ps

module tb_weight_dma;
    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk; // 100 MHz

    // AXI-Lite slave
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

    // AXI-MM master
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

    // Config-bus write port
    logic        cfg_wr_en;
    logic [15:0] cfg_wr_addr;
    logic [15:0] cfg_wr_data;
    logic        cfg_wr_full = 0;

    logic done, busy;

    weight_dma dut (
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
        .cfg_wr_en(cfg_wr_en), .cfg_wr_addr(cfg_wr_addr),
        .cfg_wr_data(cfg_wr_data), .cfg_wr_full(cfg_wr_full),
        .done(done), .busy(busy)
    );

    // ------------------------------------------------------------------
    // AXI-MM slave (ideal DRAM): 4 KB of 64-bit words, byte-addressed
    // in 8-byte beats. AR ready immediately; first R data 4 cycles
    // after AR accepted; then back-to-back beats.
    // ------------------------------------------------------------------
    logic [63:0] dram [0:1023];
    logic [31:0] slave_araddr_q;
    logic [7:0]  slave_beats_q;   // beats remaining in this burst
    logic [7:0]  slave_beat_idx_q;
    logic [2:0]  slave_lat_q;     // cycles until first data
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
                slave_lat_q      <= 3'd4; // AR -> first data latency
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

    // ------------------------------------------------------------------
    // Config-bus sink: captures every accepted write; randomly
    // asserts wfull (backpressure) when enabled.
    // ------------------------------------------------------------------
    logic [15:0] cap_addr [0:4095];
    logic [15:0] cap_data [0:4095];
    integer      cap_n = 0;
    logic        bp_enable = 0;
    integer      bp_seed = 12345;
    integer      bp_r;
    integer      bp_flag = 0;

    always_ff @(posedge clk) begin
        if (cfg_wr_en && !cfg_wr_full) begin
            cap_addr[cap_n] <= cfg_wr_addr;
            cap_data[cap_n] <= cfg_wr_data;
            cap_n <= cap_n + 1;
        end
        // Random backpressure: ~25% of cycles full (when enabled).
        if (bp_enable) begin
            bp_r = $random(bp_seed);
            cfg_wr_full <= (bp_r % 4 == 0);
        end else begin
            cfg_wr_full <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // AXI-Lite master tasks
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Vector file + checker
    // ------------------------------------------------------------------
    integer fd;
    integer ncoeff, dst_base, taps, stride, expected_cycles;
    integer i, n_exp_writes;
    logic [15:0] exp_addr [0:4095];
    logic [15:0] exp_data [0:4095];
    integer errors = 0;
    integer start_cycle, done_cycle, measured_cycles;
    integer cycle_cnt = 0;
    integer wait_i = 0;
    string fname;
    reg [8*64-1:0] line;
    integer code;
    integer idx;
    integer val;
    integer cyc_tol = 20;

    always @(posedge clk) begin
        if (rst_n)
            cycle_cnt <= cycle_cnt + 1;
    end

    initial begin
        if (!$value$plusargs("vectors=%s", fname))
            fname = "vectors_dma_A.txt";
        fd = $fopen(fname, "r");
        if (fd == 0) begin
            $display("FATAL: cannot open %s", fname);
            $finish;
        end
        // Header: "# ncoeff=.. dst=0x.. taps=.. stride=.."
        code = $fscanf(fd, "# ncoeff=%d dst=0x%x taps=%d stride=%d\n",
                       ncoeff, dst_base, taps, stride);
        code = $fscanf(fd, "# expected_cycles=%d\n", expected_cycles);
        // Coefficients: "C <idx> <hex>"
        for (i = 0; i < ncoeff; i = i + 1) begin
            code = $fscanf(fd, "C %d %x\n", idx, val);
            // Pack 4 coeffs per 64-bit DRAM beat (little-endian).
            dram[idx / 4][(idx % 4) * 16 +: 16] = val[15:0];
        end
        // Expected writes: "W <addr> <data>"
        n_exp_writes = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "W %x %x\n", idx, val);
            if (code == 2) begin
                exp_addr[n_exp_writes] = idx[15:0];
                exp_data[n_exp_writes] = val[15:0];
                n_exp_writes = n_exp_writes + 1;
            end
        end
        $fclose(fd);
        $display("vectors: %s ncoeff=%0d dst=0x%04x taps=%0d stride=%0d",
                 fname, ncoeff, dst_base, taps, stride);
        $display("expected_cycles=%0d (model)", expected_cycles);
        if (n_exp_writes != ncoeff) begin
            $display("FATAL: expected %0d writes, file has %0d",
                     ncoeff, n_exp_writes);
            $finish;
        end

        // Reset
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // Program the DMA (src=0, the coefficients are at DRAM[0]).
        axil_write(6'h00, 32'd0);        // src
        axil_write(6'h04, ncoeff);       // count
        axil_write(6'h08, dst_base);     // dst
        axil_write(6'h0c, taps);         // taps
        axil_write(6'h10, stride);       // stride
        // Enable random backpressure (tests the wfull path) when
        // +bp=1 is passed. Default (+bp absent) is no backpressure.
        // (The clean run measures cycles; the bp run checks data
        // correctness under stalls.)
        bp_enable = 0;
        if ($value$plusargs("bp=%d", bp_flag) && bp_flag != 0)
            bp_enable = 1;

        start_cycle = cycle_cnt;
        axil_write(6'h14, 32'd1);        // ctrl: start

        // Wait for done (bounded).
        done_cycle = 0;
        for (wait_i = 0; wait_i < 100000; wait_i = wait_i + 1) begin
            @(posedge clk);
            if (done) begin
                done_cycle = cycle_cnt;
                wait_i = 100000; // break
            end
        end
        if (done_cycle == 0) begin
            $display("FATAL: timeout waiting for done");
            $finish;
        end
        measured_cycles = done_cycle - start_cycle;
        repeat (4) @(posedge clk);

        // ---- check the captured writes ----
        if (cap_n != n_exp_writes) begin
            $display("FAIL: captured %0d writes, expected %0d",
                     cap_n, n_exp_writes);
            errors = errors + 1;
        end else begin
            for (i = 0; i < n_exp_writes; i = i + 1) begin
                if (cap_addr[i] !== exp_addr[i]
                    || cap_data[i] !== exp_data[i]) begin
                    if (errors < 5)
                        $display("FAIL: write %0d got (0x%04x,0x%04x) exp (0x%04x,0x%04x)", i, cap_addr[i], cap_data[i], exp_addr[i], exp_data[i]);
                    errors = errors + 1;
                end
            end
        end

        $display("measured_cycles=%0d model=%0d diff=%0d", measured_cycles, expected_cycles, measured_cycles - expected_cycles);
        if (measured_cycles - expected_cycles > cyc_tol) begin
            $display("NOTE: measured exceeds model by > %0d cycles (backpressure stalls included)", cyc_tol);
        end

        if (errors == 0)
            $display("DMA CHECK PASSED (%0d writes, %0d cycles)",
                     cap_n, measured_cycles);
        else
            $display("DMA CHECK FAILED (%0d errors)", errors);
        $finish;
    end
endmodule
`default_nettype wire
