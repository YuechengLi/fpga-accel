//=============================================================================
// srcnn_pipe.sv -- Stage 7b: SRCNN 3-stage streaming pipeline wrapper.
//   L1 conv2d_outpar (9x9, 1 -> 64, P_OUT=16, 4 cy/pos, 1296 DSPs)
//     -> gather (4 x 16-byte groups -> one 64-byte vector)
//   L2 conv1x1_cp   (1x1, 64 -> 32, P_OUT=7, 5 cy/pos, 448 DSPs)
//     -> gather (5 x 7-byte groups -> one 32-byte vector)
//   L3 conv2d_tappar (5x5, 32 -> 1, tap-parallel, 5 cy/pos, 160 DSPs)
//
// Total 1904 DSPs (96.7% of the 1968-DSP XCZU19EG). The pipeline
// bottleneck is 5 cycles per output position (L2/L3); at 400 MHz that
// is 39.0 fps at 1080p (1920x1080 output, 10,248,320 cycles/frame).
//
// All intermediates are on-chip (gather registers + the engines' line
// buffers); weights are preloaded once at startup into registers, so
// there is no per-frame weight traffic and no DRAM traffic in steady
// state. The bicubic upsample (640x360 -> 1920x1080) feeding L1 is done
// by the host (PIL, as in the Stage 5/6 demos), ~16 MACs/output px
// (~33M MACs/frame, 0.2% of the 16.45 GMACs/frame); a future RTL
// bicubic unit would cost ~16 DSPs and does not change the fps.
//
// Backpressure: each stage's out_ready is driven by the downstream
// gather/engine, so a slower stage stalls the faster one (valid/ready
// throughout). The gathers are 2-entry skid buffers so the L1 (4
// cy/pos) to L2 (5 cy/pos) rate mismatch is absorbed without dropping
// vectors.
//
// Config: the wrapper exposes a single config bus; the caller writes
// each engine's registers (L1 at its addresses, L2 at its, L3 at its)
// and then writes GO to all three (the wrapper fans the GO out). The
// geometry (W/H) is the upsampled input frame (1920x1080); L1's out_w
// and out_h are the valid-conv output size (1912x1072 for 9x9 valid),
// which is L2's input size; L3's out_w/out_h are the final output
// (1908x1068 for 5x5 valid on top of L2's output). The caller is
// responsible for consistent geometry; see the cycle model
// (sw/vision/stage7b_model.py) for the frame sizes.
//
// Cycle counter: cycles/frame (GO to last output) is exposed on
// cyc_frame for the cycle-model cross-check.
//=============================================================================
`timescale 1ns/1ps

module srcnn_pipe #(
    parameter int L1_P_OUT = 16,
    parameter int L2_P_OUT = 7,
    parameter int L2_P_IN  = 64,
    parameter int L3_P_IN  = 32
)(
    input  logic clk,
    input  logic rst_n,
    // config bus (fanned out to all three engines; the caller uses
    // each engine's own address map)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (1-channel upsampled frame, raster order)
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (1 byte per output position)
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy,
    output logic [31:0] cyc_frame
);
    // Config: the wrapper exposes a single config bus; the top two
    // address bits select the engine (all three engines share the same
    // low-14-bit address map, so the bus must be routed, not fanned
    // out -- fanning a coeff write out would load it into all three):
    //   L1: 0x0000-0x3FFF (cfg_addr[15:14]==00), L2: 0x4000-0x7FFF (01),
    //   L3: 0x8000-0xBFFF (10). The engine sees cfg_addr[13:0].
    // The caller writes each engine's registers (L1 at 0x0xxx, L2 at
    // 0x4xxx, L3 at 0x8xxx) and then GO to L1 (0x000C) and L3 (0x800C);
    // L2 has no GO (it runs when vectors arrive).
    logic cfg_l1, cfg_l2, cfg_l3;
    logic [15:0] cfg_addr_lo;
    assign cfg_l1 = cfg_valid && (cfg_addr[15:14] == 2'b00);
    assign cfg_l2 = cfg_valid && (cfg_addr[15:14] == 2'b01);
    assign cfg_l3 = cfg_valid && (cfg_addr[15:14] == 2'b10);
    assign cfg_addr_lo = {2'b00, cfg_addr[13:0]};

    // ------------------------- L1 -> L2 gather -------------------------
    // L1 emits 16-byte groups (group 0..3) per position; the gather
    // accumulates 4 groups into one 64-byte vector for L2. 2-entry skid
    // buffer: while one vector is being consumed by L2, the next is
    // being gathered.
    logic [8*L2_P_IN-1:0] g1_vec [0:1];
    logic g1_sof [0:1], g1_eol [0:1], g1_last [0:1];
    logic g1_valid [0:1];
    logic g1_wr, g1_rd;
    logic [1:0] g1_cnt;             // groups gathered for the current vector
    logic l1_out_valid, l1_out_sof, l1_out_eol, l1_out_last;
    logic [8*L1_P_OUT-1:0] l1_pixel_out;
    logic [3:0] l1_out_group;
    logic l1_out_ready;
    logic l1_busy;
    logic [8*L2_P_IN-1:0] g1_out_vec;
    logic g1_out_sof, g1_out_eol, g1_out_last;
    logic g1_out_valid;
    logic l2_in_ready;

    // The gather accepts an L1 group when the current skid slot is free
    // (not holding a complete vector waiting for L2).
    assign l1_out_ready = !g1_valid[g1_wr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g1_valid[0] <= 1'b0; g1_valid[1] <= 1'b0;
            g1_wr <= 1'b0; g1_rd <= 1'b0;
            g1_cnt <= 2'd0;
        end else begin
            // L1 group arrives: place it at the right byte offset.
            // Group g1_cnt (0..3) occupies bytes [g1_cnt*16, +15].
            if (l1_out_valid && l1_out_ready) begin
                for (int b = 0; b < L1_P_OUT; b++)
                    g1_vec[g1_wr][(g1_cnt*16+b)*8 +: 8] <=
                        l1_pixel_out[b*8 +: 8];
                if (l1_out_group == 4'd0) begin
                    g1_sof[g1_wr] <= l1_out_sof;
                end
                if (l1_out_group == 4'd3) begin
                    g1_eol[g1_wr] <= l1_out_eol;
                    g1_last[g1_wr] <= l1_out_last;
                end
                if (g1_cnt == 2'd3) begin
                    g1_valid[g1_wr] <= 1'b1;
                    g1_wr <= ~g1_wr;
                    g1_cnt <= 2'd0;
                end else begin
                    g1_cnt <= g1_cnt + 2'd1;
                end
            end
            // L2 consumes a vector.
            if (g1_out_valid && l2_in_ready) begin
                g1_valid[g1_rd] <= 1'b0;
                g1_rd <= ~g1_rd;
            end
        end
    end
    assign g1_out_vec   = g1_vec[g1_rd];
    assign g1_out_sof   = g1_sof[g1_rd];
    assign g1_out_eol   = g1_eol[g1_rd];
    assign g1_out_last  = g1_last[g1_rd];
    assign g1_out_valid = g1_valid[g1_rd];

    // ------------------------- L2 -> L3 gather -------------------------
    // L2 emits 7-byte groups (group 0..4, last group 4 valid); the
    // gather accumulates them into one 32-byte vector for L3.
    logic [8*L3_P_IN-1:0] g2_vec [0:1];
    logic g2_sof [0:1], g2_eol [0:1], g2_last [0:1];
    logic g2_valid [0:1];
    logic g2_wr, g2_rd;
    logic [2:0] g2_cnt;             // groups gathered for the current vector
    logic [8*L2_P_OUT-1:0] l2_pixel_out;
    logic l2_out_valid, l2_out_sof, l2_out_eol, l2_out_last;
    logic [L2_P_OUT-1:0] l2_out_mask;
    logic [3:0] l2_out_group;
    logic l2_out_ready;
    logic [8*L3_P_IN-1:0] g2_out_vec;
    logic g2_out_sof, g2_out_eol, g2_out_last;
    logic g2_out_valid;
    logic l3_in_ready;

    assign l2_out_ready = !g2_valid[g2_wr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g2_valid[0] <= 1'b0; g2_valid[1] <= 1'b0;
            g2_wr <= 1'b0; g2_rd <= 1'b0;
            g2_cnt <= 3'd0;
        end else begin
            if (l2_out_valid && l2_out_ready) begin
                // place the valid bytes of this group (mask marks them)
                for (int b = 0; b < L2_P_OUT; b++)
                    if (l2_out_mask[b])
                        g2_vec[g2_wr][(g2_cnt*7+b)*8 +: 8] <=
                            l2_pixel_out[b*8 +: 8];
                if (l2_out_group == 4'd0)
                    g2_sof[g2_wr] <= l2_out_sof;
                if (l2_out_group == 4'd4) begin
                    g2_eol[g2_wr] <= l2_out_eol;
                    g2_last[g2_wr] <= l2_out_last;
                end
                if (g2_cnt == 3'd4) begin
                    g2_valid[g2_wr] <= 1'b1;
                    g2_wr <= ~g2_wr;
                    g2_cnt <= 3'd0;
                end else begin
                    g2_cnt <= g2_cnt + 3'd1;
                end
            end
            if (g2_out_valid && l3_in_ready) begin
                g2_valid[g2_rd] <= 1'b0;
                g2_rd <= ~g2_rd;
            end
        end
    end
    assign g2_out_vec   = g2_vec[g2_rd];
    assign g2_out_sof   = g2_sof[g2_rd];
    assign g2_out_eol   = g2_eol[g2_rd];
    assign g2_out_last  = g2_last[g2_rd];
    assign g2_out_valid = g2_valid[g2_rd];

    // ------------------------- engines ---------------------------------
    logic l3_busy;
    logic [31:0] cyc_cnt;
    logic cyc_run;

    conv2d_outpar #(.P_OUT(L1_P_OUT)) l1_inst (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_l1), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr_lo), .cfg_data(cfg_data),
        .pixel_in(pixel_in), .in_valid(in_valid), .in_sof(in_sof),
        .in_ready(in_ready),
        .pixel_out(l1_pixel_out), .out_valid(l1_out_valid),
        .out_group(l1_out_group),
        .out_sof(l1_out_sof), .out_eol(l1_out_eol), .out_last(l1_out_last),
        .out_ready(l1_out_ready), .busy(l1_busy)
    );

    conv1x1_cp #(.P_OUT(L2_P_OUT), .P_IN(L2_P_IN)) l2_inst (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_l2), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr_lo), .cfg_data(cfg_data),
        .in_vec(g1_out_vec), .in_valid(g1_out_valid),
        .in_sof(g1_out_sof), .in_eol(g1_out_eol), .in_last(g1_out_last),
        .in_ready(l2_in_ready),
        .pixel_out(l2_pixel_out), .out_valid(l2_out_valid),
        .out_mask(l2_out_mask), .out_group(l2_out_group),
        .out_sof(l2_out_sof), .out_eol(l2_out_eol), .out_last(l2_out_last),
        .out_ready(l2_out_ready)
    );

    conv2d_tappar #(.P_IN(L3_P_IN)) l3_inst (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_l3), .cfg_we(cfg_we),
        .cfg_addr(cfg_addr_lo), .cfg_data(cfg_data),
        .in_vec(g2_out_vec), .in_valid(g2_out_valid),
        .in_sof(g2_out_sof), .in_eol(g2_out_eol), .in_last(g2_out_last),
        .in_ready(l3_in_ready),
        .pixel_out(pixel_out), .out_valid(out_valid),
        .out_sof(out_sof), .out_eol(out_eol), .out_last(out_last),
        .out_ready(out_ready), .busy(l3_busy)
    );

    assign busy = l1_busy || l3_busy;

    // Cycle counter: runs from the first input accepted to the last
    // output, for the cycle-model cross-check.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cyc_cnt <= 32'd0; cyc_run <= 1'b0; cyc_frame <= 32'd0;
        end else begin
            if (in_valid && in_ready && !cyc_run) begin
                cyc_run <= 1'b1;
                cyc_cnt <= 32'd1;
            end else if (cyc_run) begin
                cyc_cnt <= cyc_cnt + 32'd1;
                if (out_valid && out_ready && out_last) begin
                    cyc_frame <= cyc_cnt + 32'd1;
                    cyc_run <= 1'b0;
                end
            end
        end
    end
endmodule
