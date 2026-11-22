//=============================================================================
// conv1x1_cp.sv -- Stage 7b: channel-parallel 1x1 convolution engine
// (SRCNN L2: 64 input channels -> 32 output channels).
//
// The K^2-parallel shape of conv2d_mc / conv2d_outpar is wrong for 1x1
// (1 mult/lane would need ~494 lanes for L2). This engine parallelises
// across channels instead: P_OUT output channels x P_IN input channels
// per cycle. Default P_OUT=7, P_IN=64 -> 448 multipliers, 5 cycles per
// input position (output groups 0..4; the last group has only 4 valid
// outputs, 32 = 4*7 + 4).
//
// Interface (streaming, no frame state -- the engine processes one input
// position vector at a time as it arrives):
//   in_vec    512-bit input position vector (64 bytes, channel-major)
//   in_valid  / in_ready  (AXI-Stream-like; a 2-entry skid buffer
//             decouples the input rate from the 5-cycle compute rate)
//   in_sof / in_eol / in_last  per-position framing, carried with the
//             vector and re-emitted on the output groups (same
//             convention as conv2d_outpar: out_sof on group 0 of the
//             first position, out_eol on every group of a last-column
//             position, out_last on the last group of the last position)
//   pixel_out 56-bit output group (7 bytes), out_mask marks the valid
//             bytes (7'b1111111 for groups 0..3, 7'b0001111 for group 4)
//   out_valid / out_ready  (backpressure freezes the whole pipeline)
//   out_group  the output group index (0..4)
//
// Config bus (same map as conv2d_mc for the shared registers):
//   0x01 shift, 0x0A C_in (64), 0x0B C_out (32), 0x0F sign (bit0 in,
//   bit1 out), 0x10 act (0 none, 1 ReLU, 2 Leaky), 0x11 slope (Q16).
//   Coefficients: 0x100 + o*64 + c  (o = output channel 0..31,
//   c = input channel 0..63), 16-bit signed.
//   Bias: lo 0x900+o, hi 0x920+o (32-bit signed). The bias bases sit
//   AFTER the coeff range (0x100..0x8FF); putting them inside it
//   would route the bias writes into the coeff array.
//   There is no GO: coefficients are loaded once at startup and the
//   engine runs whenever input vectors arrive.
//
// Datapath (per output group, one group per cycle):
//   7 dot-product units, each 64 mults (25-bit products) -> adder_tree
//   (N=64, 4 cycles) -> + bias -> round-half-up shift -> fused
//   ReLU/Leaky -> saturate -> emit. Pipeline latency from the group
//   register to the output register is 5 cycles; the group / framing
//   flags ride a 5-deep flag pipeline aligned with the data (same
//   construction as conv2d_outpar).
//
// Resource (inferred): 448 DSPs (7x64 mults). The adder trees are the
// timing-critical path (64-input tree); see the STA logs.
//=============================================================================
`timescale 1ns/1ps

module conv1x1_cp #(
    parameter int P_OUT = 7,        // output channels per cycle
    parameter int P_IN  = 64,       // input channels (vector width)
    parameter int C_OUT = 32,       // total output channels
    parameter int NGROUP = 5        // ceil(C_OUT / P_OUT); last group partial
)(
    input  logic clk,
    input  logic rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // input position vector (channel-major, 64 bytes)
    input  logic [8*P_IN-1:0] in_vec,
    input  logic        in_valid,
    input  logic        in_sof,
    input  logic        in_eol,
    input  logic        in_last,
    output logic        in_ready,
    // output group (P_OUT bytes per cycle)
    output logic [8*P_OUT-1:0] pixel_out,
    output logic        out_valid,
    output logic [P_OUT-1:0] out_mask,
    output logic [3:0]  out_group,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready
);
    localparam int NK = P_IN;       // products per dot product

    //------------------------- config --------------------------------------
    localparam logic [15:0] CFG_SHIFT     = 16'h0001;
    localparam logic [15:0] CFG_CIN       = 16'h000A;
    localparam logic [15:0] CFG_COUT      = 16'h000B;
    localparam logic [15:0] CFG_SIGN      = 16'h000F;
    localparam logic [15:0] CFG_ACT       = 16'h0010;
    localparam logic [15:0] CFG_ACT_SLOPE = 16'h0011;
    localparam logic [15:0] CFG_COEFF_BASE = 16'h0100;
    // Bias bases sit AFTER the coeff range (0x100..0x8FF, 2048 entries):
    // putting them at 0x300/0x400 would overlap the coeff addresses and
    // the bias writes would land in the coeff array instead (the bias
    // would silently never be applied).
    localparam logic [15:0] CFG_BIAS_LO_BASE = 16'h0900;
    localparam logic [15:0] CFG_BIAS_HI_BASE = 16'h0920;

    logic [3:0]  shift_r;
    logic [6:0]  cin_r, cout_r;
    logic        sign_in_r, sign_out_r;
    logic [1:0]  fuse_act_r;
    logic signed [15:0] act_slope_r;
    // Coeff/bias arrays are sized to NGROUP*P_OUT (35) rather than C_OUT
    // (32): the last group (group 4) reads channels 28..34, and the
    // three phantom channels 32..34 must read as zero (never written)
    // instead of indexing out of bounds. Their outputs are masked off
    // by out_mask.
    logic signed [15:0] coeff [0:NGROUP*P_OUT-1][0:P_IN-1];
    logic [15:0] bias_lo [0:NGROUP*P_OUT-1];
    logic [15:0] bias_hi [0:NGROUP*P_OUT-1];
    int ci, cj;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_r  <= 4'd8;
            cin_r <= 7'd64;
            cout_r <= 7'd32;
            sign_in_r <= 1'b0;
            sign_out_r <= 1'b0;
            fuse_act_r <= 2'd0;
            act_slope_r <= 16'sd8192;
            for (ci = 0; ci < NGROUP*P_OUT; ci++) begin
                for (cj = 0; cj < P_IN; cj++) coeff[ci][cj] <= '0;
                bias_lo[ci] <= 16'd0;
                bias_hi[ci] <= 16'd0;
            end
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_SHIFT : shift_r  <= cfg_data[3:0];
                CFG_CIN   : cin_r    <= cfg_data[6:0];
                CFG_COUT  : cout_r   <= cfg_data[6:0];
                CFG_SIGN  : begin
                    sign_in_r  <= cfg_data[0];
                    sign_out_r <= cfg_data[1];
                end
                CFG_ACT   : fuse_act_r <= cfg_data[1:0];
                CFG_ACT_SLOPE : act_slope_r <= cfg_data;
                default: begin
                    if (cfg_addr >= CFG_COEFF_BASE &&
                        cfg_addr < CFG_COEFF_BASE + 16'd2048) begin
                        ci = (cfg_addr - CFG_COEFF_BASE) >> 6;
                        cj = (cfg_addr - CFG_COEFF_BASE) & 16'h003F;
                        coeff[ci][cj] <= cfg_data;
                    end else if (cfg_addr >= CFG_BIAS_LO_BASE &&
                               cfg_addr < CFG_BIAS_LO_BASE + 16'd32) begin
                        bias_lo[cfg_addr - CFG_BIAS_LO_BASE] <= cfg_data;
                    end else if (cfg_addr >= CFG_BIAS_HI_BASE &&
                               cfg_addr < CFG_BIAS_HI_BASE + 16'd32) begin
                        bias_hi[cfg_addr - CFG_BIAS_HI_BASE] <= cfg_data;
                    end
                end
            endcase
        end
    end

    //------------------------- input skid buffer ----------------------------
    // 2-entry buffer; in_ready is high when a slot is free. The compute
    // side loads the oldest valid entry into cur_* at the group wrap.
    logic [8*P_IN-1:0] buf_vec [0:1];
    logic buf_sof [0:1];
    logic buf_eol [0:1];
    logic buf_last [0:1];
    logic buf_valid [0:1];
    logic buf_wr;                   // write pointer (0/1)
    logic buf_rd;                   // read pointer (0/1)
    logic stall;
    assign stall = out_valid && !out_ready;

    assign in_ready = !(buf_valid[0] && buf_valid[1]) && !stall;
    // in_ready MUST include !stall: when the pipeline is frozen by
    // backpressure (out_valid && !out_ready) the always_ff below does
    // not execute, so a vector the sender thinks was accepted would
    // never be captured. (L1's in_ready already gates on !stall; this
    // one did not, and the lost vectors stalled the engine.)

    // Current position being computed (held for NGROUP cycles).
    logic [8*P_IN-1:0] cur_vec;
    logic cur_sof, cur_eol, cur_last;
    logic cur_valid;
    logic [3:0] group;              // 0..NGROUP-1

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_valid[0] <= 1'b0; buf_valid[1] <= 1'b0;
            buf_wr <= 1'b0; buf_rd <= 1'b0;
            cur_valid <= 1'b0;
            cur_vec <= '0;
            cur_sof <= 1'b0; cur_eol <= 1'b0; cur_last <= 1'b0;
            group <= 4'd0;
        end else if (!stall) begin
            // Compute side FIRST, input side SECOND: a write and a read
            // can target the SAME slot in one cycle (when the buffer
            // holds one entry and the pointers coincide). The read
            // consumes the OLD data and the write puts NEW data in, so
            // the slot must stay valid -- therefore the write's
            // buf_valid assignment (later in this block) wins over the
            // read's clear. (Read priority loses the newly written
            // vector and starves the engine.) The read's RHS values are
            // all pre-edge, so it still gets the old entry.
            if (!cur_valid) begin
                if (buf_valid[buf_rd]) begin
                    cur_vec <= buf_vec[buf_rd];
                    cur_sof <= buf_sof[buf_rd];
                    cur_eol <= buf_eol[buf_rd];
                    cur_last <= buf_last[buf_rd];
                    cur_valid <= 1'b1;
                    buf_valid[buf_rd] <= 1'b0;
                    buf_rd <= ~buf_rd;
                end
                group <= 4'd0;
            end else if (group == (NGROUP - 1)) begin
                if (buf_valid[buf_rd]) begin
                    cur_vec <= buf_vec[buf_rd];
                    cur_sof <= buf_sof[buf_rd];
                    cur_eol <= buf_eol[buf_rd];
                    cur_last <= buf_last[buf_rd];
                    cur_valid <= 1'b1;
                    buf_valid[buf_rd] <= 1'b0;
                    buf_rd <= ~buf_rd;
                end else begin
                    // Wrap with no vector buffered: go idle instead of
                    // re-computing the stale vector (that duplicate
                    // would emit 5 extra groups and pollute the next
                    // frame/case).
                    cur_valid <= 1'b0;
                end
                group <= 4'd0;
            end else begin
                group <= group + 4'd1;
            end
            // Input side: accept a vector into the free slot.
            if (in_valid && in_ready) begin
                buf_vec[buf_wr] <= in_vec;
                buf_sof[buf_wr] <= in_sof;
                buf_eol[buf_wr] <= in_eol;
                buf_last[buf_wr] <= in_last;
                buf_valid[buf_wr] <= 1'b1;
                buf_wr <= ~buf_wr;
            end
        end
    end

    //------------------------- per-unit dot product + tree ------------------
    // Each unit u computes output channel (group*P_OUT + u) for the held
    // vector: NK products (registered) -> adder_tree (4 cycles) -> emit.
    // Units past C_out in the last group still compute (their coeffs are
    // the row for a non-existent channel, read as zero because the config
    // never writes them) but their outputs are masked off.
    logic signed [32:0] tree_out [0:P_OUT-1];
    logic [3:0] group_d [0:4];
    logic       prod_v_d [0:4];
    logic       prod_sof_d [0:4];
    logic       prod_eol_d [0:4];
    logic       prod_last_d [0:4];
    logic [3:0] grp_last_d [0:4];
    int fi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (fi = 0; fi < 5; fi++) begin
                group_d[fi] <= 4'd0;
                prod_v_d[fi] <= 1'b0;
                prod_sof_d[fi] <= 1'b0;
                prod_eol_d[fi] <= 1'b0;
                prod_last_d[fi] <= 1'b0;
                grp_last_d[fi] <= 4'd0;
            end
        end else if (!stall) begin
            group_d[0] <= group;
            prod_v_d[0] <= cur_valid;
            prod_sof_d[0] <= cur_sof;
            prod_eol_d[0] <= cur_eol;
            prod_last_d[0] <= cur_last;
            grp_last_d[0] <= (NGROUP - 1);
            for (fi = 1; fi < 5; fi++) begin
                group_d[fi] <= group_d[fi-1];
                prod_v_d[fi] <= prod_v_d[fi-1];
                prod_sof_d[fi] <= prod_sof_d[fi-1];
                prod_eol_d[fi] <= prod_eol_d[fi-1];
                prod_last_d[fi] <= prod_last_d[fi-1];
                grp_last_d[fi] <= grp_last_d[fi-1];
            end
        end
    end

    genvar u;
    generate
        for (u = 0; u < P_OUT; u++) begin : unit
            logic signed [24:0] prod [0:NK-1];
            logic signed [24:0] prod_comb [0:NK-1];
            // Flat pack for the adder tree: Yosys 0.52 rejects unpacked
            // array ports, so the tree takes in_flat (element i in
            // bits [i*25 +: 25]); the values are unchanged.
            logic signed [NK*25-1:0] prod_flat;
            genvar pk;
            for (pk = 0; pk < NK; pk++) begin : pack
                assign prod_flat[pk*25 +: 25] = prod[pk];
            end
            int c;
            always @* begin
                for (c = 0; c < NK; c++) begin
                    if (sign_in_r)
                        prod_comb[c] = $signed(cur_vec[c*8 +: 8]) *
                            coeff[group*P_OUT+u][c];
                    else
                        prod_comb[c] = $signed({1'b0, cur_vec[c*8 +: 8]}) *
                            coeff[group*P_OUT+u][c];
                end
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (c = 0; c < NK; c++) prod[c] <= 25'sd0;
                end else if (!stall) begin
                    for (c = 0; c < NK; c++) prod[c] <= prod_comb[c];
                end
            end
            adder_tree #(.N(NK), .W(25)) u_tree (
                .clk(clk), .rst_n(rst_n), .stall(stall),
                .in_flat(prod_flat), .out(tree_out[u]));
        end
    endgenerate

    //------------------------- emit (bias + shift + act + saturate) -------
    logic signed [32:0] emit_val [0:P_OUT-1];
    logic signed [32:0] emit_rnd [0:P_OUT-1];
    logic signed [32:0] emit_sh [0:P_OUT-1];
    logic signed [32:0] emit_act [0:P_OUT-1];
    logic [7:0]  emit_px [0:P_OUT-1];
    genvar e;
    generate
        for (e = 0; e < P_OUT; e++) begin : emit_unit
            logic signed [48:0] leaky_prod;
            logic signed [32:0] leaky_neg;
            assign emit_val[e] = tree_out[e] +
                $signed({{1{bias_hi[group_d[4]*P_OUT+e][15]}},
                         bias_hi[group_d[4]*P_OUT+e],
                         bias_lo[group_d[4]*P_OUT+e]});
            always @* begin
                emit_rnd[e] = emit_val[e] + ((shift_r == 0) ? 33'sd0 :
                    (33'sd1 <<< (shift_r - 1'b1)));
            end
            assign emit_sh[e] = emit_rnd[e] >>> shift_r;
            // The slope is an UNSIGNED Q0.16 fraction (0x8000 = 0.5):
            // zero-extend to 17 bits so 32768 stays +32768. Sign-
            // extending would turn 0.5 into -32768 and flip the sign
            // of the Leaky output.
            assign leaky_prod = emit_sh[e] * $signed({1'b0, act_slope_r});
            assign leaky_neg = leaky_prod[48:16];
            always @* begin
                if (fuse_act_r == 2'd1)
                    emit_act[e] = (emit_sh[e] < 33'sd0) ? 33'sd0 : emit_sh[e];
                else if (fuse_act_r == 2'd2)
                    emit_act[e] = (emit_sh[e] < 33'sd0) ? leaky_neg : emit_sh[e];
                else
                    emit_act[e] = emit_sh[e];
            end
            always @* begin
                if (sign_out_r) begin
                    if (emit_act[e] > 33'sd127)       emit_px[e] = 8'd127;
                    else if (emit_act[e] < -33'sd128) emit_px[e] = 8'h80;
                    else                              emit_px[e] = emit_act[e][7:0];
                end else begin
                    if (emit_act[e] > 33'sd255)      emit_px[e] = 8'd255;
                    else if (emit_act[e] < 33'sd0)   emit_px[e] = 8'd0;
                    else                              emit_px[e] = emit_act[e][7:0];
                end
            end
        end
    endgenerate

    // Valid mask: the last group has only C_OUT - (NGROUP-1)*P_OUT valid
    // outputs (4 for the default 32 = 4*7 + 4).
    localparam int LAST_VALID = C_OUT - (NGROUP-1)*P_OUT;
    logic [P_OUT-1:0] mask_comb;
    int mi;
    always @* begin
        if (group_d[4] == (NGROUP - 1)) begin
            mask_comb = '0;
            for (mi = 0; mi < LAST_VALID; mi++) mask_comb[mi] = 1'b1;
        end else begin
            mask_comb = '1;
        end
    end

    int ok;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= '0;
            out_valid <= 1'b0;
            out_mask <= '0;
            out_group <= 4'd0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= prod_v_d[4];
            for (ok = 0; ok < P_OUT; ok++)
                pixel_out[ok*8 +: 8] <= emit_px[ok];
            out_mask <= mask_comb;
            out_group <= group_d[4];
            out_sof <= prod_v_d[4] && prod_sof_d[4] && (group_d[4] == 4'd0);
            out_eol <= prod_v_d[4] && prod_eol_d[4];
            out_last <= prod_v_d[4] && prod_last_d[4] &&
                        (group_d[4] == grp_last_d[4]);
        end
    end
endmodule : conv1x1_cp
