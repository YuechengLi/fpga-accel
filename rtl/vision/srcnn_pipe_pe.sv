//=============================================================================
// srcnn_pipe_pe.sv -- top wrapper for the PE-array SRCNN pipeline (Stage 7b).
//
// Thin top level around pe_array. The PE-array pipeline streams
// TILE-STEPS, not pixels: the compiler scheduler (sw/common/sched_pe.py)
// does the im2col gather in software and emits the tile-step program
// (mode/TM/TN/TK, A/B tiles, tags); this wrapper feeds that program to
// the array and presents the completed (M-tile, N-chunk) output blocks.
// There is no pixel-level AXI-Stream datapath inside the array -- the
// input image reaches the scheduler via DMA (the driver), and the
// driver assembles the output blocks into the final image. Pixel
// streaming I/O and DMA integration belong to the board stages (8-11).
//
// Ports:
//   - Tile-step program input (ts_*): streamed tile-steps from the
//     compiler/driver (same handshake as pe_array).
//   - Output blocks (out_*): completed (M-tile, N-chunk) byte blocks.
//   - Config bus (AXI-Lite-like, 16-bit data): a small register file
//     for the device/layer config and status. Addresses:
//       0x00: N_PE (read-only, the array's parameter)
//       0x01: TM   (read-only)
//       0x02: TN   (read-only)
//       0x03: status: bit0 = any PE busy, bit1 = out_valid (sticky)
//       0x04: done_count (low 16 bits)
//     The tile-step program itself is streamed (not config-written):
//     a program is 737 tile-steps for SRCNN 24x24, far too large for
//     a config bus; in the system it lives in DRAM and is DMA'd to
//     the array's program port.
//   - clk / rst_n.
//
// The wrapper is verified by compilation and by the pe_array chain
// driver (sw/vision/run_pe_srcnn.py), which drives the tile-step program
// through this wrapper's program port. There is no separate wrapper
// TB: the wrapper adds no datapath logic beyond the config registers
// (the array is the datapath), so a separate TB would only re-test
// the array.
//=============================================================================

module srcnn_pipe_pe #(
    parameter int N_PE    = 4,
    parameter int TM      = 16,
    parameter int TN      = 8,
    parameter int TK_MAX  = 32
)(
    input  logic clk,
    input  logic rst_n,
    // Tile-step program input (from the compiler/driver).
    input  logic        ts_valid,
    output logic        ts_ready,
    input  logic        ts_mode_dot,
    input  logic [5:0]  ts_tk,
    input  logic [5:0]  ts_tn,
    input  logic        ts_first_k,
    input  logic        ts_sign_in,
    input  logic [3:0]  ts_shift,
    input  logic [1:0]  ts_act,
    input  logic [15:0] ts_slope,
    input  logic        ts_sign_out,
    input  logic signed [TN*32-1:0] ts_bias_flat,
    input  logic [TM*TK_MAX*8-1:0]  ts_a_flat,
    input  logic [TK_MAX*TN*16-1:0] ts_b_flat,
    input  logic [15:0] ts_mtile,
    input  logic [7:0]  ts_nchunk,
    input  logic [7:0]  ts_n_kchunk,
    // Completed-output blocks.
    output logic        out_valid,
    output logic [15:0] out_mtile,
    output logic [7:0]  out_nchunk,
    output logic [TM*TN*8-1:0] out_bytes,
    output logic [31:0] done_count,
    // Config bus (AXI-Lite-like).
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    output logic [15:0] cfg_rdata,
    output logic        cfg_rvalid
);

    //--------------------------------------------------------------------------
    // PE array (the datapath).
    //--------------------------------------------------------------------------
    logic any_busy;
    pe_array #(.N_PE(N_PE), .TM(TM), .TN(TN), .TK_MAX(TK_MAX)) u_arr (
        .clk(clk), .rst_n(rst_n),
        .ts_valid(ts_valid), .ts_ready(ts_ready),
        .ts_mode_dot(ts_mode_dot), .ts_tk(ts_tk), .ts_tn(ts_tn),
        .ts_first_k(ts_first_k),
        .ts_sign_in(ts_sign_in), .ts_shift(ts_shift), .ts_act(ts_act),
        .ts_slope(ts_slope), .ts_sign_out(ts_sign_out),
        .ts_bias_flat(ts_bias_flat),
        .ts_a_flat(ts_a_flat), .ts_b_flat(ts_b_flat),
        .ts_mtile(ts_mtile), .ts_nchunk(ts_nchunk),
        .ts_n_kchunk(ts_n_kchunk),
        .out_valid(out_valid), .out_mtile(out_mtile),
        .out_nchunk(out_nchunk), .out_bytes(out_bytes),
        .done_count(done_count), .out_count()
    );

    //--------------------------------------------------------------------------
    // Config registers (read-only device info + status). Writes are
    // ignored (the device config is a parameter, set at elaboration);
    // reads return the addressed register. cfg_rvalid pulses one cycle
    // after a read request.
    //--------------------------------------------------------------------------
    logic out_valid_sticky;

    // any_busy: OR of the array's PE busy flags (via the array's
    // done_count vs dispatched count is not visible here, so busy is
    // approximated as "a tile-step was accepted and its output has not
    // yet been produced" -- for status only, not for control).
    // (The array does not export a busy aggregate; use ts_ready as the
    // inverse indicator: not ready means all PEs are busy.)
    assign any_busy = !ts_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_rdata <= '0;
            cfg_rvalid <= 1'b0;
            out_valid_sticky <= 1'b0;
        end else begin
            cfg_rvalid <= 1'b0;
            if (out_valid) out_valid_sticky <= 1'b1;
            if (cfg_valid && !cfg_we) begin
                cfg_rvalid <= 1'b1;
                case (cfg_addr)
                    16'h00: cfg_rdata <= N_PE[15:0];
                    16'h01: cfg_rdata <= TM[15:0];
                    16'h02: cfg_rdata <= TN[15:0];
                    16'h03: cfg_rdata <= {14'd0, out_valid_sticky, any_busy};
                    16'h04: cfg_rdata <= done_count[15:0];
                    default: cfg_rdata <= 16'h0000;
                endcase
            end
            // A read of the status register clears the sticky out_valid.
            if (cfg_valid && !cfg_we && cfg_addr == 16'h03)
                out_valid_sticky <= 1'b0;
        end
    end

endmodule : srcnn_pipe_pe
