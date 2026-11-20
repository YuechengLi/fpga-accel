//=============================================================================
// unified_engine.sv -- Stage 4: unified reconfigurable convolution engine.
//
// One engine, five modes, selected by a single config write (the "mode"):
//   MODE_DENSE   (0) : dense convolution, K=1/3/5/7/9/11 (conv2d_dense)
//   MODE_DW_SQ   (1) : depthwise square, stored K=1..17 (conv2d_dw)
//   MODE_DW_SH   (2) : depthwise strip horizontal, L=1..31 (conv2d_dw)
//   MODE_DW_SV   (3) : depthwise strip vertical, L=1..31 (conv2d_dw)
//   MODE_DW_DIL  (4) : depthwise dilated, stored K=3/5/7/9, d=1..4
//   MODE_FC      (5) : fully-connected / 1x1, K=1, W=H=1 (conv2d_dense)
// FC reuses the 1x1 dense datapath (the same 121-mult MAC array as the
// dense mode); there is no separate FC datapath. The fused ReLU/Leaky
// in the conv output stage therefore covers the FC/MLP path exactly as
// it covers the conv path (verified by MC case 14: FC + fused ReLU,
// and by the engine's own mode-switch case 2: FC + fused ReLU).
//
// Structure: the engine WRAPS the Stage 1 dense module (conv2d_dense,
// now with the Stage 4 signed + fused-act additions) and the Stage 2
// DW module (conv2d_dw, now with the Stage 4 signed addition) -- those
// stay the reference implementations and are not modified in their
// datapath. It does NOT duplicate five datapaths: the dense
// submodule's one 121-mult MAC serves dense and FC/1x1, and the DW
// submodule's one 289-mult MAC serves the three DW modes
// (square/strip/dilated) via its own mode select. The engine provides
// the unified stream/config interface and the mode mux.
//   * Mode switch: a single config write to CFG_MODE (0x0C) selects
//     the active submodule and, for a DW mode, programs the DW
//     submodule's own mode register in the same write. The switch
//     takes effect on the next clock (1 cycle) and MUST be done
//     between frames (no frame may be in flight); the stream mux is
//     combinational, so there is no added pipeline latency for the
//     pixel stream. Switching back to a previous mode does NOT require
//     reloading its coefficients: each submodule keeps its own
//     coefficient/config registers, so the inactive submodule's state
//     (coefficients, K, pad, stride, shift, sign, fused act) is
//     preserved across a switch. Only the active submodule's line
//     buffers / pipeline hold in-flight frame state, and a switch is
//     only legal between frames, so no pixel state is lost.
//   * Config: the engine's config map IS the submodules' shared map
//     (see below), so forwarding is a pure router: the write is
//     truncated to the active submodule's bus width and forwarded.
//     Coefficients (0x10+i) go to the ACTIVE submodule only, so the
//     user sets the mode first and then loads that mode's
//     coefficients. The inactive submodule's coefficients are
//     untouched. Address 0x0A means "fused act" in the dense
//     submodule and "DW mode" in the DW submodule; address 0x0B means
//     "act slope" in dense and "dilation" in DW. The engine's mode
//     write programs the DW mode automatically, so the user never
//     needs to write 0x0A by hand for a DW switch.
//   * Stream: one input stream, one output stream. in_ready is the
//     active submodule's in_ready; the output is the active
//     submodule's output. The inactive submodule's in_valid is held
//     low so it accepts nothing.
//   * The engine is single-channel (PX=1), like its two submodules.
//     Multi-channel tiling is conv2d_mc (Stage 3); the layer
//     scheduler (Stage 4) drives the engine per tile / per channel.
//
// Config bus (16-bit address, write between frames; shared submodule
// map, forwarded to the active submodule):
//   0x0000 K (dense K, or DW stored K)
//   0x0001 shift (0..15)
//   0x0002 W, 0x0003 H
//   0x0004 pad_t, 0x0005 pad_b, 0x0006 pad_l, 0x0007 pad_r
//   0x0008 stride_x, 0x0009 stride_y
//   0x000A fused act (dense: 0=none, 1=ReLU, 2=Leaky) / DW mode
//          (DW: 0=square, 1=strip-H, 2=strip-V, 3=dilated; set
//          automatically by the engine mode write)
//   0x000B act slope (dense, Q0.16) / dilation (DW, 1..4)
//   0x000C ENGINE mode (0=dense, 1=DW-sq, 2=DW-strip-H, 3=DW-strip-V,
//          4=DW-dilated, 5=FC). Consumed by the engine, not forwarded
//          as a control write (but it programs the DW mode, above).
//   0x000F sign (bit0 signed in, bit1 signed out)
//   0x0100+i coefficients, i=0..288 (active submodule only; dense
//          uses 0..120, DW uses 0..288)
// Reset defaults: mode=dense, K=3, shift=8, W=640, H=480, pads 0,
//   strides 1, sign 0, fused act 0, slope 0.125 (from the submodules).
//=============================================================================
`timescale 1ns/1ps

module unified_engine (
    input  logic        clk,
    input  logic        rst_n,
    // config bus (16-bit address)
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [15:0] cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [2:0] MODE_DENSE  = 3'd0;
    localparam logic [2:0] MODE_DW_SQ  = 3'd1;
    localparam logic [2:0] MODE_DW_SH  = 3'd2;
    localparam logic [2:0] MODE_DW_SV  = 3'd3;
    localparam logic [2:0] MODE_DW_DIL = 3'd4;
    localparam logic [2:0] MODE_FC     = 3'd5;

    // The DW submodule's own mode select (conv2d_dw config 0x0A):
    // 0=square, 1=strip-H, 2=strip-V, 3=dilated.
    localparam logic [1:0] DW_SQ = 2'd0;
    localparam logic [1:0] DW_SH = 2'd1;
    localparam logic [1:0] DW_SV = 2'd2;
    localparam logic [1:0] DW_DIL = 2'd3;

    localparam logic [15:0] CFG_MODE = 16'h000C;

    logic [2:0] mode_r;
    logic dense_active;
    assign dense_active = (mode_r == MODE_DENSE) || (mode_r == MODE_FC);

    //------------------------- engine mode register ------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            mode_r <= MODE_DENSE;
        else if (cfg_valid && cfg_we && cfg_addr == CFG_MODE)
            mode_r <= cfg_data[2:0];
    end

    // DW submodule mode for the NEW mode being written (not the old
    // mode_r: the mode write must program the DW mode in the same
    // cycle it switches the engine).
    logic [1:0] dw_mode_new;
    always @* begin
        case (cfg_data[2:0])
            MODE_DW_SQ:  dw_mode_new = DW_SQ;
            MODE_DW_SH:  dw_mode_new = DW_SH;
            MODE_DW_SV:  dw_mode_new = DW_SV;
            MODE_DW_DIL: dw_mode_new = DW_DIL;
            default:     dw_mode_new = DW_SQ;
        endcase
    end

    //------------------------- config forwarding ---------------------------
    // Pure router: a write is forwarded to the ACTIVE submodule with
    // the address truncated to that submodule's bus width. The mode
    // write (0x0C) is consumed by the engine; it is NOT forwarded as
    // a control write, but when the new mode is a DW mode it also
    // programs the DW submodule's mode register (0x0A) in the same
    // write, so one write both switches the engine and configures DW.
    logic dense_cfg_valid, dense_cfg_we;
    logic [7:0] dense_cfg_addr;
    logic [15:0] dense_cfg_data;
    logic dw_cfg_valid, dw_cfg_we;
    logic [8:0] dw_cfg_addr;
    logic [15:0] dw_cfg_data;

    always @* begin
        dense_cfg_valid = 1'b0;
        dense_cfg_we = 1'b0;
        dense_cfg_addr = 8'd0;
        dense_cfg_data = 16'd0;
        dw_cfg_valid = 1'b0;
        dw_cfg_we = 1'b0;
        dw_cfg_addr = 9'd0;
        dw_cfg_data = 16'd0;
        if (cfg_valid && cfg_we) begin
            if (cfg_addr == CFG_MODE) begin
                if (cfg_data[2:0] == MODE_DW_SQ ||
                    cfg_data[2:0] == MODE_DW_SH ||
                    cfg_data[2:0] == MODE_DW_SV ||
                    cfg_data[2:0] == MODE_DW_DIL) begin
                    dw_cfg_valid = 1'b1;
                    dw_cfg_we = 1'b1;
                    dw_cfg_addr = 9'h00A;
                    dw_cfg_data = {14'd0, dw_mode_new};
                end
            end else if (cfg_addr >= 16'h0100) begin
                // Coefficient write: engine 0x100+i -> submodule
                // 0x10+i (a raw truncation would map coeff 0 to K).
                // The subtraction is in 16-bit; i (0..288) fits in
                // 9 bits, so the truncation to the submodule width
                // is safe.
                if (dense_active) begin
                    dense_cfg_valid = 1'b1;
                    dense_cfg_we = 1'b1;
                    dense_cfg_addr = 8'h10 + (cfg_addr - 16'h0100);
                    dense_cfg_data = cfg_data;
                end else begin
                    dw_cfg_valid = 1'b1;
                    dw_cfg_we = 1'b1;
                    dw_cfg_addr = 9'h010 + (cfg_addr - 16'h0100);
                    dw_cfg_data = cfg_data;
                end
            end else if (dense_active) begin
                dense_cfg_valid = 1'b1;
                dense_cfg_we = 1'b1;
                dense_cfg_addr = cfg_addr[7:0];
                dense_cfg_data = cfg_data;
            end else begin
                dw_cfg_valid = 1'b1;
                dw_cfg_we = 1'b1;
                dw_cfg_addr = cfg_addr[8:0];
                dw_cfg_data = cfg_data;
            end
        end
    end

    //------------------------- submodules ----------------------------------
    logic [7:0] dense_px_out, dw_px_out;
    logic dense_out_valid, dw_out_valid;
    logic dense_out_sof, dw_out_sof;
    logic dense_out_eol, dw_out_eol;
    logic dense_out_last, dw_out_last;
    logic dense_in_ready, dw_in_ready;
    logic dense_busy, dw_busy;

    conv2d_dense dense_u (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(dense_cfg_valid), .cfg_we(dense_cfg_we),
        .cfg_addr(dense_cfg_addr), .cfg_data(dense_cfg_data),
        .pixel_in(pixel_in),
        .in_valid(in_valid && dense_active),
        .in_sof(in_sof),
        .in_ready(dense_in_ready),
        .pixel_out(dense_px_out), .out_valid(dense_out_valid),
        .out_sof(dense_out_sof), .out_eol(dense_out_eol),
        .out_last(dense_out_last), .out_ready(out_ready),
        .busy(dense_busy)
    );

    conv2d_dw dw_u (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(dw_cfg_valid), .cfg_we(dw_cfg_we),
        .cfg_addr(dw_cfg_addr), .cfg_data(dw_cfg_data),
        .pixel_in(pixel_in),
        .in_valid(in_valid && !dense_active),
        .in_sof(in_sof),
        .in_ready(dw_in_ready),
        .pixel_out(dw_px_out), .out_valid(dw_out_valid),
        .out_sof(dw_out_sof), .out_eol(dw_out_eol),
        .out_last(dw_out_last), .out_ready(out_ready),
        .busy(dw_busy)
    );

    //------------------------- stream mux ----------------------------------
    assign in_ready = dense_active ? dense_in_ready : dw_in_ready;
    assign pixel_out = dense_active ? dense_px_out : dw_px_out;
    assign out_valid = dense_active ? dense_out_valid : dw_out_valid;
    assign out_sof = dense_active ? dense_out_sof : dw_out_sof;
    assign out_eol = dense_active ? dense_out_eol : dw_out_eol;
    assign out_last = dense_active ? dense_out_last : dw_out_last;
    assign busy = dense_active ? dense_busy : dw_busy;
endmodule
