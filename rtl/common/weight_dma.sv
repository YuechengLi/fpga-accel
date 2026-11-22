// weight_dma.sv -- Stage 5 AXI-MM weight-streaming DMA.
//
// Streams int16 coefficients from DRAM (AXI-MM, 64-bit, up to 16-beat
// INCR bursts) into the ping-pong weight banks (scheduler, contiguous
// addressing) or directly into the engine's coefficient registers
// (conv2d_mc, strided addressing: lane l at dst + l*stride, tap k at
// +k). Stage 4 loaded weights via the config bus (2 cycles/coeff,
// PS-written); this DMA is autonomous (the PS only programs the
// registers) and drain-limited at 1 coeff/cycle (the config-bus
// write rate into the bank).
//
// Cycle model (must match sw/common/dram.py weight_dma_cycles):
//   cycles = ncoeff + n_bursts * DMA_AR_OVERHEAD + DMA_LATENCY
// where n_bursts = ceil(ncoeff / 64) (4 coeffs/beat * 16 beats),
// DMA_AR_OVERHEAD = 4 cycles/burst, DMA_LATENCY = 12 cycles.
// The model assumes the AXI slave is fast enough that the drain
// (1 coeff/cycle) is the bottleneck; the TB's AXI slave is ideal
// (AR ready immediately, first data 4 cycles after AR) so the
// measured count matches the model.
//
// Register map (AXI-Lite slave, 6-bit addr, 32-bit data):
//   0x00 src    DRAM byte offset (source)
//   0x01 count  number of int16 coefficients
//   0x02 dst    config-bus base address (destination)
//   0x03 taps   taps per lane (121 = the full 11x11 grid, for the
//                 dense engines; 0 = contiguous (stride 1))
//   0x04 stride address stride between lanes (128 for conv2d_mc)
//   0x05 ctrl   bit0 = start (self-clearing)
//   0x06 status bit0 = done (sticky, cleared by a new start),
//               bit1 = busy
//
// The config-bus write port (cfg_wr_*) drives the destination
// module's config bus. cfg_wr_full is backpressure: when asserted
// the DMA stalls (does not advance the coefficient pointer).
// The DMA asserts done when the last coefficient has been written
// (and accepted, i.e. not stalled by wfull).
//
// This is RTL (synthesizable SystemVerilog). It is verified by
// tb/tb_weight_dma.sv against the Python cycle model and a
// golden data/address check. Board-level AXI integration (the
// SmartConnect, the DRAM controller, the PS driver handshake) is
// Stages 8-11.
`default_nettype none

module weight_dma #(
    parameter int AXI_ADDR_W  = 32,
    parameter int AXI_DATA_W  = 64,
    parameter int BURST_BEATS = 16,  // max beats per burst (AXI)
    parameter int FIFO_DEPTH  = 2    // beats (64-bit), circular buffer
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // AXI-Lite slave (programming)
    input  logic                    s_axil_awvalid,
    output logic                    s_axil_awready,
    input  logic [5:0]              s_axil_awaddr,
    input  logic                    s_axil_wvalid,
    output logic                    s_axil_wready,
    input  logic [31:0]             s_axil_wdata,
    output logic                    s_axil_bvalid,
    input  logic                    s_axil_bready,
    output logic [1:0]              s_axil_bresp,
    input  logic                    s_axil_arvalid,
    output logic                    s_axil_arready,
    input  logic [5:0]              s_axil_araddr,
    output logic                    s_axil_rvalid,
    input  logic                    s_axil_rready,
    output logic [31:0]             s_axil_rdata,
    output logic [1:0]              s_axil_rresp,

    // AXI-MM master (read-only, 64-bit)
    output logic [AXI_ADDR_W-1:0]   m_axi_araddr,
    output logic [7:0]              m_axi_arlen,
    output logic [2:0]              m_axi_arsize,
    output logic [1:0]              m_axi_arburst,
    output logic                    m_axi_arvalid,
    input  logic                    m_axi_arready,
    input  logic [AXI_DATA_W-1:0]   m_axi_rdata,
    input  logic                    m_axi_rlast,
    input  logic                    m_axi_rvalid,
    output logic                    m_axi_rready,

    // Config-bus write port (to the engine / scheduler bank)
    output logic                    cfg_wr_en,
    output logic [15:0]             cfg_wr_addr,
    output logic [15:0]             cfg_wr_data,
    input  logic                    cfg_wr_full,

    output logic                    done,
    output logic                    busy
);

    // ------------------------------------------------------------------
    // Registers (programmed via AXI-Lite)
    // ------------------------------------------------------------------
    logic [31:0] reg_src;
    logic [15:0] reg_count;
    logic [15:0] reg_dst;
    logic [15:0] reg_taps;
    logic [15:0] reg_stride;
    logic        reg_done;
    logic        start_pulse;

    // AXI-Lite write/read channels (single-beat, always ready).
    assign s_axil_awready = 1'b1;
    assign s_axil_wready  = 1'b1;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_rresp   = 2'b00;
    assign s_axil_arready = 1'b1;

    logic        bvalid_q;
    logic        rvalid_q;
    logic [31:0] rdata_q;
    assign s_axil_bvalid = bvalid_q;
    assign s_axil_rvalid = rvalid_q;
    assign s_axil_rdata  = rdata_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_src     <= 32'd0;
            reg_count   <= 16'd0;
            reg_dst     <= 16'd0;
            reg_taps    <= 16'd0;
            reg_stride  <= 16'd0;
            reg_done    <= 1'b0;
            bvalid_q    <= 1'b0;
            rvalid_q    <= 1'b0;
            rdata_q     <= 32'd0;
            start_pulse <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (s_axil_awvalid && s_axil_wvalid) begin
                case (s_axil_awaddr[5:2])
                    4'h0: reg_src    <= s_axil_wdata;
                    4'h1: reg_count  <= s_axil_wdata[15:0];
                    4'h2: reg_dst    <= s_axil_wdata[15:0];
                    4'h3: reg_taps   <= s_axil_wdata[15:0];
                    4'h4: reg_stride <= s_axil_wdata[15:0];
                    4'h5: if (s_axil_wdata[0]) begin
                              start_pulse <= 1'b1;
                              reg_done    <= 1'b0;
                          end
                    default: begin end
                endcase
                bvalid_q <= 1'b1;
            end else if (bvalid_q && s_axil_bready) begin
                bvalid_q <= 1'b0;
            end
            if (s_axil_arvalid) begin
                case (s_axil_araddr[5:2])
                    4'h0: rdata_q <= reg_src;
                    4'h1: rdata_q <= {16'd0, reg_count};
                    4'h2: rdata_q <= {16'd0, reg_dst};
                    4'h3: rdata_q <= {16'd0, reg_taps};
                    4'h4: rdata_q <= {16'd0, reg_stride};
                    4'h5: rdata_q <= 32'd0;
                    4'h6: rdata_q <= {30'd0, busy, reg_done};
                    default: rdata_q <= 32'd0;
                endcase
                rvalid_q <= 1'b1;
            end else if (rvalid_q && s_axil_rready) begin
                rvalid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // State + counters (declared before the FIFO, which uses them).
    // Main state machine: IDLE -> AR -> RUN (read + drain) -> DONE.
    // In RUN the AXI read fills the FIFO while the writer drains it;
    // a new AR is issued whenever the current burst is fully
    // received and more coefficients remain.
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {
        ST_IDLE, ST_AR, ST_RUN, ST_DONE
    } state_t;
    state_t state_q, state_d;

    logic [31:0] ar_addr_q;       // next DRAM byte address to read
    logic [15:0] coeff_to_read_q; // coefficients not yet read (AR'd)
    logic [15:0] coeff_written_q; // coefficients written (accepted)
    logic [15:0] coeff_idx_q;     // index of the next coeff to write
    logic [1:0]  lane_pos_q;      // position in the head beat (0..3)
    logic [15:0] burst_left_q;    // beats remaining in the current burst
    logic        burst_active_q;  // an AR has been issued, beats pending

    // Coefficient addressing: lane = idx / taps, tap = idx % taps
    // (taps=0 means contiguous: lane 0, tap = idx).
    logic [15:0] cur_lane, cur_tap;
    logic [15:0] cur_dst_addr;
    always_comb begin
        if (reg_taps == 16'd0) begin
            cur_lane = 16'd0;
            cur_tap  = coeff_idx_q;
        end else begin
            cur_lane = coeff_idx_q / reg_taps;
            cur_tap  = coeff_idx_q % reg_taps;
        end
        cur_dst_addr = reg_dst + cur_lane * reg_stride + cur_tap;
    end

    // Beats needed for the remaining coefficients (ceil(/4)),
    // capped at BURST_BEATS.
    logic [15:0] beats_needed;
    always_comb begin
        beats_needed = (coeff_to_read_q + 16'd3) >> 2;
        if (beats_needed > BURST_BEATS[15:0])
            beats_needed = BURST_BEATS[15:0];
    end

    // The writer pops a beat when it has consumed all 4 coefficients
    // (or the last coefficient of the transfer).
    logic beat_done;
    assign beat_done = (lane_pos_q == 2'd3)
                       || (coeff_idx_q + 16'd1 >= reg_count);

    // ------------------------------------------------------------------
    // FIFO (circular buffer, FIFO_DEPTH beats of 64 bits).
    // ------------------------------------------------------------------
    logic [63:0] fifo_mem [0:FIFO_DEPTH-1];
    logic        fifo_wr_ptr;   // 0/1 (FIFO_DEPTH=2)
    logic        fifo_rd_ptr;
    logic [1:0]  fifo_cnt;      // 0..FIFO_DEPTH
    logic        fifo_push, fifo_pop;

    assign fifo_push = m_axi_rvalid && m_axi_rready;
    assign fifo_pop  = (state_q == ST_RUN) && (fifo_cnt != 2'd0)
                       && !cfg_wr_full && beat_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_wr_ptr <= 1'b0;
            fifo_rd_ptr <= 1'b0;
            fifo_cnt    <= 2'd0;
            fifo_mem[0] <= 64'd0;
            fifo_mem[1] <= 64'd0;
        end else begin
            if (fifo_push)
                fifo_mem[fifo_wr_ptr] <= m_axi_rdata;
            if (fifo_push)
                fifo_wr_ptr <= ~fifo_wr_ptr;
            if (fifo_pop)
                fifo_rd_ptr <= ~fifo_rd_ptr;
            if (fifo_push && !fifo_pop)
                fifo_cnt <= fifo_cnt + 2'd1;
            else if (!fifo_push && fifo_pop)
                fifo_cnt <= fifo_cnt - 2'd1;
        end
    end

    logic [63:0] fifo_head_data;
    assign fifo_head_data = fifo_mem[fifo_rd_ptr];

    // The 4 coefficients in the head beat (little-endian int16).
    logic [15:0] beat_coeff [0:3];
    assign beat_coeff[0] = fifo_head_data[15:0];
    assign beat_coeff[1] = fifo_head_data[31:16];
    assign beat_coeff[2] = fifo_head_data[47:32];
    assign beat_coeff[3] = fifo_head_data[63:48];

    // Config-bus write: one coefficient per cycle when the FIFO has
    // data and the destination is not full.
    assign cfg_wr_en   = (state_q == ST_RUN) && (fifo_cnt != 2'd0)
                         && !cfg_wr_full;
    assign cfg_wr_addr = cur_dst_addr;
    assign cfg_wr_data = beat_coeff[lane_pos_q];

    // AXI read: accept data when the FIFO has room (in RUN).
    assign m_axi_rready = (state_q == ST_RUN)
                          && (fifo_cnt < FIFO_DEPTH[1:0]);
    assign m_axi_araddr  = ar_addr_q;
    assign m_axi_arlen   = beats_needed[7:0] - 8'd1;
    assign m_axi_arsize  = 3'b011; // 8 bytes
    assign m_axi_arburst = 2'b01;  // INCR
    assign m_axi_arvalid = (state_q == ST_AR);

    assign done = reg_done;
    assign busy = (state_q != ST_IDLE);

    always_comb begin
        state_d = state_q;
        case (state_q)
            ST_IDLE: begin
                if (start_pulse) begin
                    if (reg_count == 16'd0)
                        state_d = ST_DONE;
                    else
                        state_d = ST_AR;
                end
            end
            ST_AR: begin
                if (m_axi_arvalid && m_axi_arready)
                    state_d = ST_RUN;
            end
            ST_RUN: begin
                if (coeff_written_q == reg_count)
                    state_d = ST_DONE;
                else if (!burst_active_q && (coeff_to_read_q != 16'd0)
                         && (fifo_cnt == 2'd0))
                    state_d = ST_AR; // next burst (FIFO drained)
            end
            ST_DONE: begin
                state_d = ST_IDLE;
            end
            default: state_d = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q         <= ST_IDLE;
            ar_addr_q       <= 32'd0;
            coeff_to_read_q <= 16'd0;
            coeff_written_q <= 16'd0;
            coeff_idx_q     <= 16'd0;
            lane_pos_q      <= 2'd0;
            burst_left_q    <= 16'd0;
            burst_active_q  <= 1'b0;
        end else begin
            state_q <= state_d;
            case (state_q)
                ST_IDLE: begin
                    if (start_pulse) begin
                        ar_addr_q       <= reg_src;
                        coeff_to_read_q <= reg_count;
                        coeff_written_q <= 16'd0;
                        coeff_idx_q     <= 16'd0;
                        lane_pos_q      <= 2'd0;
                        burst_left_q    <= 16'd0;
                        burst_active_q  <= 1'b0;
                    end
                end
                ST_AR: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        burst_left_q   <= beats_needed;
                        burst_active_q <= 1'b1;
                        ar_addr_q      <= ar_addr_q
                                          + {16'd0, beats_needed} * 32'd8;
                        if (coeff_to_read_q > beats_needed * 16'd4)
                            coeff_to_read_q <= coeff_to_read_q
                                               - beats_needed * 16'd4;
                        else
                            coeff_to_read_q <= 16'd0;
                    end
                end
                ST_RUN: begin
                    if (fifo_push) begin
                        if (burst_left_q != 16'd0)
                            burst_left_q <= burst_left_q - 16'd1;
                        if (burst_left_q == 16'd1)
                            burst_active_q <= 1'b0;
                    end
                    if (cfg_wr_en) begin
                        coeff_written_q <= coeff_written_q + 16'd1;
                        coeff_idx_q     <= coeff_idx_q + 16'd1;
                        if (lane_pos_q == 2'd3)
                            lane_pos_q <= 2'd0;
                        else
                            lane_pos_q <= lane_pos_q + 2'd1;
                    end
                    if (state_d == ST_DONE)
                        reg_done <= 1'b1;
                end
                ST_DONE: begin
                    reg_done <= 1'b1;
                end
                default: begin end
            endcase
        end
    end

endmodule
`default_nettype wire
