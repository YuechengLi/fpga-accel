//=============================================================================
// layer_scheduler.sv -- Stage 4: layer scheduler with ping-pong weights.
//
// Steps through a layer configuration sequence: for each layer it
// programs the layer's config (mode, K, pad, stride, shift, sign,
// fused act), triggers execution, and waits for completion. Weight
// double-buffering (ping-pong): two weight register banks (A and B).
// Layer L's weights live in bank (L % 2): while bank A drives layer 0,
// bank B is loaded with layer 1's weights, so the load time is hidden;
// while bank B drives layer 1, bank A is reloaded with layer 2's
// weights, and so on. The scheduler never touches a bank while it is
// driving the engine.
//   * Weights stream in through the scheduler's own config bus in
//     this stage (the DRAM/AXI-MM streaming that fills the banks is
//     Stage 5); the boundary is explicit: the scheduler only knows
//     "write coefficient i of bank X", not where the coefficients
//     come from. The TB (or the Stage 5 DMA) writes them; the
//     scheduler counts them per bank.
//   * Ping-pong timing: when the next layer's weights are fully
//     loaded before the current layer finishes, the switch is
//     immediate (no stall -- the next layer starts on the next cycle
//     after the current one completes). When they arrive late, the
//     scheduler waits in ST_WAIT_W (it does NOT start the next layer
//     with a half-loaded bank, and it does not corrupt the active
//     bank). The TB verifies both cases.
//   * Program: up to NLAYER layer descriptors, each a set of fields
//     written through the config bus (one write per field, addressed
//     by (layer, field)). The scheduler sequences layers 0..N-1 in
//     order and asserts `done` when the last layer completes. The
//     engine is the unified_engine (single-channel); the scheduler
//     drives its config bus as a master and watches its `busy` to
//     know when a layer's frame is complete.
//   * A bank is "ready" for layer L when its loaded count >=
//     prog_ncoeff[L]. The count for a bank resets when the scheduler
//     switches AWAY from it (so the TB can load the next layer that
//     reuses that bank from a clean count).
//
// Scheduler config bus (16-bit address, write before / during a run):
//   0x0000 : control: bit0 = start (pulse), bit1 = reset (clears the
//            program counter / state back to idle)
//   0x0001 : number of layers N (1..NLAYER)
//   0x0002 : layer select (which layer the field writes below target)
//   0x0003 : field: mode (0=dense, 1=DW-sq, 2=DW-SH, 3=DW-SV,
//            4=DW-dil, 5=FC)
//   0x0004 : field: K
//   0x0005 : field: shift
//   0x0006 : field: W
//   0x0007 : field: H
//   0x0008 : field: pad_t, 0x0009 : field: pad_b,
//            0x000A : field: pad_l, 0x000B : field: pad_r
//   0x000C : field: stride_x, 0x000D : field: stride_y
//   0x000E : field: sign (bit0 in, bit1 out)
//   0x000F : field: fused act (0=none, 1=ReLU, 2=Leaky)
//   0x0010 : field: act slope (Q0.16)
//   0x0011 : field: dilation (DW-dilated, 1..4)
//   0x0012 : field: ncoeff (number of coefficients for this layer)
//   0x0200+i : write coefficient i (16-bit) into bank A (layer even)
//   0x0400+i : write coefficient i (16-bit) into bank B (layer odd)
//     (Bank B base is 0x0400, not 0x0300: each bank holds 289 coeffs,
//     so bank A spans 0x0200..0x0320 and a 0x0300 base would overlap
//     it in 0x0300..0x0320 -- the else-if would misroute bank-B writes
//     with index < 33 into bank_a[256+i], leaving bank_b[0..32] X.)
//            A write to a bank that is currently driving the engine
//            is ignored and `wfull` pulses (the writer must retry);
//            this is the backpressure on the weight stream.
// Outputs:
//   eng_cfg_* : config bus master to the unified_engine.
//   eng_start : pulse when a layer's config + weights are programmed
//               (the top level uses it to begin streaming the layer's
//               input frame into the engine).
//   done      : high when the whole sequence has completed.
//   busy      : high while any layer is running or being programmed.
//   wfull     : high for one cycle when a coefficient write was
//               ignored (bank busy).
//=============================================================================
`timescale 1ns/1ps

module layer_scheduler #(
    parameter int NLAYER = 4,
    parameter int BANK_COEFF = 289
)(
    input  logic        clk,
    input  logic        rst_n,
    // scheduler config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // engine config bus master (to unified_engine)
    output logic        eng_cfg_valid,
    output logic        eng_cfg_we,
    output logic [15:0] eng_cfg_addr,
    output logic [15:0] eng_cfg_data,
    // engine status / trigger
    input  logic        eng_busy,
    output logic        eng_start,
    // sequence status
    output logic        done,
    output logic        busy,
    output logic        wfull
);
    //------------------------- layer program -------------------------------
    logic [2:0]  prog_mode [0:NLAYER-1];
    logic [3:0]  prog_k [0:NLAYER-1];
    logic [3:0]  prog_shift [0:NLAYER-1];
    logic [11:0] prog_w [0:NLAYER-1];
    logic [11:0] prog_h [0:NLAYER-1];
    logic [3:0]  prog_pt [0:NLAYER-1];
    logic [3:0]  prog_pb [0:NLAYER-1];
    logic [3:0]  prog_pl [0:NLAYER-1];
    logic [3:0]  prog_pr [0:NLAYER-1];
    logic [3:0]  prog_sx [0:NLAYER-1];
    logic [3:0]  prog_sy [0:NLAYER-1];
    logic [1:0]  prog_sign [0:NLAYER-1];
    logic [1:0]  prog_act [0:NLAYER-1];
    logic [15:0] prog_slope [0:NLAYER-1];
    logic [3:0]  prog_dil [0:NLAYER-1];
    logic [8:0]  prog_ncoeff [0:NLAYER-1];

    logic [3:0] nlayer_r;
    logic [3:0] layer_sel_r;

    //------------------------- FSM state (declared before use) -----------
    localparam logic [2:0] ST_IDLE = 3'd0;
    localparam logic [2:0] ST_PROG = 3'd1;
    localparam logic [2:0] ST_COEFF = 3'd2;
    localparam logic [2:0] ST_RUN = 3'd3;
    localparam logic [2:0] ST_WAIT_W = 3'd4;
    localparam logic [2:0] ST_DONE = 3'd5;
    logic [2:0] state;
    logic [3:0] layer_idx;
    logic [3:0] prog_step;
    logic [8:0] stream_idx;
    // Set when the engine's busy has been seen high for the current
    // layer (so ST_RUN does not falsely complete before the frame
    // starts streaming -- busy is low while the TB is still reacting
    // to eng_start).
    logic seen_busy_r;

    //------------------------- ping-pong weight banks ----------------------
    logic [15:0] bank_a [0:BANK_COEFF-1];
    logic [15:0] bank_b [0:BANK_COEFF-1];
    logic [8:0]  bank_a_cnt, bank_b_cnt;  // coefficients loaded per bank
    // Layer L uses bank (L[0]==0 ? A : B). A bank is ready for layer L
    // when its count >= prog_ncoeff[L].
    logic bank_a_ready, bank_b_ready;
    logic [3:0] next_layer;
    assign next_layer = layer_idx + 4'd1;
    // Ready for the NEXT layer (the bank it will use). Used at the
    // ST_RUN transition, where layer_idx is still the layer just
    // finished, so next_layer is the upcoming layer.
    logic next_ready;
    always @* begin
        if (next_layer[0] == 1'b0)
            next_ready = (bank_a_cnt >= prog_ncoeff[next_layer]);
        else
            next_ready = (bank_b_cnt >= prog_ncoeff[next_layer]);
    end
    // Ready for the CURRENT layer (the bank it will use). Used in
    // ST_WAIT_W, where layer_idx has already advanced to the layer
    // being waited for. (next_ready would point one layer too far
    // here: layer_idx+1, whose prog_ncoeff is 0 when unprogrammed,
    // so it is trivially true and ST_WAIT_W exits immediately --
    // the scheduler would stream a half-loaded bank and pulse
    // eng_start while the TB is still loading weights.)
    logic cur_ready;
    always @* begin
        if (layer_idx[0] == 1'b0)
            cur_ready = (bank_a_cnt >= prog_ncoeff[layer_idx]);
        else
            cur_ready = (bank_b_cnt >= prog_ncoeff[layer_idx]);
    end

    //------------------------- FSM -----------------------------------------
    // ST_IDLE: wait for start. ST_PROG: program the engine control
    //   registers for the current layer (one per cycle, steps 0..13).
    // ST_COEFF: stream the current layer's bank to the engine
    //   (0x100+i, one per cycle). ST_RUN: wait for eng_busy to fall.
    // ST_WAIT_W: the next layer's bank is not ready yet; wait.
    // ST_DONE: sequence complete.

    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);

    // The bank driving the current layer.
    logic cur_bank;  // 0 = A, 1 = B
    assign cur_bank = layer_idx[0];

    // Single sequential block (no multiple drivers).
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nlayer_r <= 4'd0;
            layer_sel_r <= 4'd0;
            bank_a_cnt <= 9'd0;
            bank_b_cnt <= 9'd0;
            state <= ST_IDLE;
            layer_idx <= 4'd0;
            prog_step <= 4'd0;
            stream_idx <= 9'd0;
            seen_busy_r <= 1'b0;
            eng_cfg_valid <= 1'b0; eng_cfg_we <= 1'b0;
            eng_cfg_addr <= 16'd0; eng_cfg_data <= 16'd0;
            eng_start <= 1'b0;
            wfull <= 1'b0;
            for (int i = 0; i < NLAYER; i++) begin
                prog_mode[i] <= 3'd0; prog_k[i] <= 4'd3;
                prog_shift[i] <= 4'd8; prog_w[i] <= 12'd16;
                prog_h[i] <= 12'd16; prog_pt[i] <= 4'd0;
                prog_pb[i] <= 4'd0; prog_pl[i] <= 4'd0;
                prog_pr[i] <= 4'd0; prog_sx[i] <= 4'd1;
                prog_sy[i] <= 4'd1; prog_sign[i] <= 2'd0;
                prog_act[i] <= 2'd0; prog_slope[i] <= 16'd8192;
                prog_dil[i] <= 4'd1; prog_ncoeff[i] <= 9'd0;
            end
        end else begin
            eng_cfg_valid <= 1'b0;
            eng_start <= 1'b0;
            wfull <= 1'b0;

            //---- config writes (program fields + weight banks) ----
            // Weight writes are accepted only when the target bank is
            // NOT currently driving the engine (i.e. it is not the
            // current layer's bank while that layer is running or
            // being programmed / streamed). Otherwise wfull pulses.
            if (cfg_valid && cfg_we) begin
                if (cfg_addr == 16'h0000) begin
                    if (cfg_data[1]) begin
                        layer_idx <= 4'd0;
                        state <= ST_IDLE;
                        bank_a_cnt <= 9'd0;
                        bank_b_cnt <= 9'd0;
                    end
                end else if (cfg_addr == 16'h0001) begin
                    nlayer_r <= cfg_data[3:0];
                end else if (cfg_addr == 16'h0002) begin
                    layer_sel_r <= cfg_data[3:0];
                end else if (cfg_addr >= 16'h0003 && cfg_addr <= 16'h0012) begin
                    case (cfg_addr)
                        16'h0003: prog_mode[layer_sel_r] <= cfg_data[2:0];
                        16'h0004: prog_k[layer_sel_r] <= cfg_data[3:0];
                        16'h0005: prog_shift[layer_sel_r] <= cfg_data[3:0];
                        16'h0006: prog_w[layer_sel_r] <= cfg_data[11:0];
                        16'h0007: prog_h[layer_sel_r] <= cfg_data[11:0];
                        16'h0008: prog_pt[layer_sel_r] <= cfg_data[3:0];
                        16'h0009: prog_pb[layer_sel_r] <= cfg_data[3:0];
                        16'h000A: prog_pl[layer_sel_r] <= cfg_data[3:0];
                        16'h000B: prog_pr[layer_sel_r] <= cfg_data[3:0];
                        16'h000C: prog_sx[layer_sel_r] <= cfg_data[3:0];
                        16'h000D: prog_sy[layer_sel_r] <= cfg_data[3:0];
                        16'h000E: prog_sign[layer_sel_r] <= cfg_data[1:0];
                        16'h000F: prog_act[layer_sel_r] <= cfg_data[1:0];
                        16'h0010: prog_slope[layer_sel_r] <= cfg_data;
                        16'h0011: prog_dil[layer_sel_r] <= cfg_data[3:0];
                        16'h0012: prog_ncoeff[layer_sel_r] <= cfg_data[8:0];
                        default: begin end
                    endcase
                end else if (cfg_addr >= 16'h0200 &&
                           cfg_addr < 16'h0200 + BANK_COEFF) begin
                    // Bank A write (layers with even index).
                    if ((state == ST_PROG || state == ST_COEFF ||
                         state == ST_RUN) && cur_bank == 1'b0) begin
                        wfull <= 1'b1;
                    end else begin
                        // Bank index = address - base. (Using cfg_addr[8:0]
                        // directly is wrong for bank B, whose base
                        // 0x0400 is not a multiple of 512.)
                        bank_a[cfg_addr - 16'h0200] <= cfg_data;
                        // Count = highest index written + 1 (robust
                        // to a single lost increment if a write lands
                        // in the same cycle the FSM resets the count).
                        if ((cfg_addr - 16'h0200) >= {7'd0, bank_a_cnt})
                            bank_a_cnt <= (cfg_addr - 16'h0200) + 9'd1;
                    end
                end else if (cfg_addr >= 16'h0400 &&
                           cfg_addr < 16'h0400 + BANK_COEFF) begin
                    // Bank B write (layers with odd index).
                    if ((state == ST_PROG || state == ST_COEFF ||
                         state == ST_RUN) && cur_bank == 1'b1) begin
                        wfull <= 1'b1;
                    end else begin
                        bank_b[cfg_addr - 16'h0400] <= cfg_data;
                        if ((cfg_addr - 16'h0400) >= {7'd0, bank_b_cnt})
                            bank_b_cnt <= (cfg_addr - 16'h0400) + 9'd1;
                    end
                end
            end

            //---- FSM ----
            // A config reset write (0x0000 bit1) takes priority over
            // the FSM: without this guard the FSM's own state assignment
            // (e.g. ST_DONE <= ST_DONE) overwrites the config section's
            // state <= ST_IDLE, because the FSM section runs later in
            // this same always_ff block -- so the scheduler could never
            // leave ST_DONE for a second run.
            if (cfg_valid && cfg_we && cfg_addr == 16'h0000 &&
                cfg_data[1]) begin
                state <= ST_IDLE;
                prog_step <= 4'd0;
                stream_idx <= 9'd0;
                seen_busy_r <= 1'b0;
            end else begin
            case (state)
                ST_IDLE: begin
                    if (cfg_valid && cfg_we && cfg_addr == 16'h0000 &&
                        cfg_data[0]) begin
                        layer_idx <= 4'd0;
                        prog_step <= 4'd0;
                        state <= ST_PROG;
                    end
                end
                ST_PROG: begin
                    eng_cfg_valid <= 1'b1;
                    eng_cfg_we <= 1'b1;
                    case (prog_step)
                        4'd0: begin
                            eng_cfg_addr <= 16'h000C;
                            eng_cfg_data <= {13'd0, prog_mode[layer_idx]};
                        end
                        4'd1: begin
                            eng_cfg_addr <= 16'h0000;
                            eng_cfg_data <= {12'd0, prog_k[layer_idx]};
                        end
                        4'd2: begin
                            eng_cfg_addr <= 16'h0001;
                            eng_cfg_data <= {12'd0, prog_shift[layer_idx]};
                        end
                        4'd3: begin
                            eng_cfg_addr <= 16'h0002;
                            eng_cfg_data <= {4'd0, prog_w[layer_idx]};
                        end
                        4'd4: begin
                            eng_cfg_addr <= 16'h0003;
                            eng_cfg_data <= {4'd0, prog_h[layer_idx]};
                        end
                        4'd5: begin
                            eng_cfg_addr <= 16'h0004;
                            eng_cfg_data <= {12'd0, prog_pt[layer_idx]};
                        end
                        4'd6: begin
                            eng_cfg_addr <= 16'h0005;
                            eng_cfg_data <= {12'd0, prog_pb[layer_idx]};
                        end
                        4'd7: begin
                            eng_cfg_addr <= 16'h0006;
                            eng_cfg_data <= {12'd0, prog_pl[layer_idx]};
                        end
                        4'd8: begin
                            eng_cfg_addr <= 16'h0007;
                            eng_cfg_data <= {12'd0, prog_pr[layer_idx]};
                        end
                        4'd9: begin
                            eng_cfg_addr <= 16'h0008;
                            eng_cfg_data <= {12'd0, prog_sx[layer_idx]};
                        end
                        4'd10: begin
                            eng_cfg_addr <= 16'h0009;
                            eng_cfg_data <= {12'd0, prog_sy[layer_idx]};
                        end
                        4'd11: begin
                            eng_cfg_addr <= 16'h000B;
                            if (prog_mode[layer_idx] == 3'd0 ||
                                prog_mode[layer_idx] == 3'd5)
                                eng_cfg_data <= prog_slope[layer_idx];
                            else
                                eng_cfg_data <= {12'd0, prog_dil[layer_idx]};
                        end
                        4'd12: begin
                            eng_cfg_addr <= 16'h000A;
                            eng_cfg_data <= {14'd0, prog_act[layer_idx]};
                        end
                        4'd13: begin
                            eng_cfg_addr <= 16'h000F;
                            eng_cfg_data <= {14'd0, prog_sign[layer_idx]};
                        end
                        default: eng_cfg_valid <= 1'b0;
                    endcase
                    if (prog_step >= 4'd13) begin
                        prog_step <= 4'd0;
                        stream_idx <= 9'd0;
                        state <= ST_COEFF;
                    end else begin
                        prog_step <= prog_step + 4'd1;
                    end
                end
                ST_COEFF: begin
                    eng_cfg_valid <= 1'b1;
                    eng_cfg_we <= 1'b1;
                    eng_cfg_addr <= 16'h0100 + stream_idx;
                    eng_cfg_data <= (cur_bank == 1'b0) ?
                        bank_a[stream_idx] : bank_b[stream_idx];
                    if (stream_idx >= prog_ncoeff[layer_idx] - 9'd1) begin
                        stream_idx <= 9'd0;
                        eng_start <= 1'b1;
                        state <= ST_RUN;
                    end else begin
                        stream_idx <= stream_idx + 9'd1;
                    end
                end
                ST_RUN: begin
                    // Wait for the frame to actually start (busy high)
                    // before waiting for it to finish (busy low); the
                    // TB needs a few cycles after eng_start to begin
                    // streaming, during which busy is still low.
                    if (!seen_busy_r) begin
                        if (eng_busy) seen_busy_r <= 1'b1;
                    end else if (!eng_busy) begin
                        seen_busy_r <= 1'b0;
                        if (layer_idx >= nlayer_r - 4'd1) begin
                            state <= ST_DONE;
                        end else begin
                            // Switch to the next layer (which flips
                            // the bank). Reset the count of the bank
                            // just finished, so it can be reloaded
                            // for the layer after next.
                            if (cur_bank == 1'b0)
                                bank_a_cnt <= 9'd0;
                            else
                                bank_b_cnt <= 9'd0;
                            layer_idx <= layer_idx + 4'd1;
                            prog_step <= 4'd0;
                            if (next_ready)
                                state <= ST_PROG;
                            else
                                state <= ST_WAIT_W;
                        end
                    end
                end
                ST_WAIT_W: begin
                    if (cur_ready) begin
                        prog_step <= 4'd0;
                        state <= ST_PROG;
                    end
                end
                ST_DONE: begin
                    state <= ST_DONE;
                end
                default: state <= ST_IDLE;
            endcase
            end
        end
    end
endmodule
