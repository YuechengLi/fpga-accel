//=============================================================================
// upsample.sv -- Stage 4: upsampling (scale factor 2), SystemVerilog.
//
// Modes (config 0x00): 0 = nearest-neighbor, 1 = bilinear, 2 = pixel
// shuffle (depth-to-space, r=2). All modes share one structure: load an
// input row into buffer(s), then emit two output rows from the buffers.
// Input is paused (in_ready low) while emitting.
//   * Nearest (mode 0): input W x H -> output 2W x 2H. Each input pixel
//     is replicated 2x2. One input row is buffered; the two output rows
//     are emitted from it, each pixel twice.
//   * Bilinear (mode 1): input W x H -> output 2W x 2H.
//       out[2r,2c]     = in[r,c]
//       out[2r,2c+1]   = (in[r,c] + in[r,c+1]) / 2
//       out[2r+1,2c]   = (in[r,c] + in[r+1,c]) / 2
//       out[2r+1,2c+1] = (in[r,c]+in[r,c+1]+in[r+1,c]+in[r+1,c+1]) / 4
//     Edges replicate (in[r,W] = in[r,W-1], in[H,c] = in[H-1,c]).
//     Division is round-half-up: (a+b+1)>>1 and (a+b+c+d+2)>>2. Two
//     input rows are buffered in ping-pong buffers (buf0/buf1); a `cur`
//     bit selects which holds the current row, so no row copy is needed
//     -- the roles swap each row. For the last input row, the "next"
//     row is the current row itself (replicate).
//   * Pixel-shuffle (mode 2): input is W x H positions x 4 channels,
//     arriving in (H,W,C) order (4 consecutive pixels per position).
//     Output is 2W x 2H, 1 channel:
//       out[2r,2c]=ch0, out[2r,2c+1]=ch1, out[2r+1,2c]=ch2,
//       out[2r+1,2c+1]=ch3.
//     The 4-channel groups of one input row are loaded into two buffers
//     (even_buf: ch0/ch1, odd_buf: ch2/ch3, each 2W pixels); the two
//     output rows are then emitted from them.
//   * The line buffers are register arrays sized by MAX_W (default 64);
//     the scheduler tiles the frame so the tile width fits. (A BRAM
//     line buffer is the production choice for large W; the register
//     array keeps Stage 4 verification simple and exact.)
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//
// Config bus (write between frames; single-cycle writes; 8-bit address):
//   addr 0x00 : mode (0=nearest, 1=bilinear, 2=pixel-shuffle)
//   addr 0x01 : image width  (1..MAX_W)
//   addr 0x02 : image height (1..4095)
// Reset defaults: mode=0 (nearest), W=16, H=16.
//=============================================================================
`timescale 1ns/1ps

module upsample #(
    parameter int MAX_W = 64
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
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
    localparam logic [7:0] CFG_MODE   = 8'h00;
    localparam logic [7:0] CFG_WIDTH  = 8'h01;
    localparam logic [7:0] CFG_HEIGHT = 8'h02;

    localparam logic [1:0] MODE_NEAREST = 2'd0;
    localparam logic [1:0] MODE_BILINEAR = 2'd1;
    localparam logic [1:0] MODE_SHUFFLE = 2'd2;

    //------------------------- config registers ----------------------------
    logic [1:0]  mode_r;
    logic [11:0] width_r, height_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= MODE_NEAREST;
            width_r <= 12'd16;
            height_r <= 12'd16;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_MODE:   mode_r <= cfg_data[1:0];
                CFG_WIDTH:  width_r <= cfg_data[11:0];
                CFG_HEIGHT: height_r <= cfg_data[11:0];
                default: ;
            endcase
        end
    end

    //------------------------- buffers -------------------------------------
    // buf0/buf1: ping-pong input rows (bilinear). Nearest uses buf0 only.
    // even_buf/odd_buf: pixel-shuffle output rows (2W pixels each).
    logic [7:0] buf0 [0:MAX_W-1];
    logic [7:0] buf1 [0:MAX_W-1];
    logic [7:0] even_buf [0:2*MAX_W-1];
    logic [7:0] odd_buf [0:2*MAX_W-1];

    //------------------------- FSM -----------------------------------------
    // ST_LOAD: accept input pixels into the buffer(s) for the current row.
    // ST_EMIT: emit the two output rows from the buffers (input paused).
    // There is no terminal DONE state: when a frame completes, the FSM
    // returns to ST_LOAD (idle) so the next frame can start without a
    // reset (the TB does not reset between cases).
    // For bilinear, ST_LOAD loads the NEXT row into the non-current
    // buffer (except the first row, which loads into buf0 as the current
    // row, and the last row, which needs no load -- the next row is the
    // current row itself).
    localparam logic [1:0] ST_LOAD = 2'd0;
    localparam logic [1:0] ST_EMIT = 2'd1;
    logic [1:0] state;

    logic [12:0] in_row;    // input row being processed (0..H-1)
    logic [12:0] in_col;    // input col being loaded (0..W-1, or 0..4W-1 shuffle)
    logic [12:0] out_row;   // output row being emitted (0..2H-1)
    logic [12:0] out_col;   // output col being emitted (0..2W-1)
    logic        cur;       // bilinear: 0 = buf0 is current row, 1 = buf1
    logic        load_phase; // bilinear: 0 = load current/next row,
                            // 1 = load the second row (row 0 only)
    logic        stall;
    logic        last_in_row;

    assign stall = out_valid && !out_ready;
    assign last_in_row = (in_row >= {1'b0, height_r} - 13'd1);
    assign in_ready = (state == ST_LOAD) && !stall;
    assign busy = (state == ST_EMIT);

    //------------------------- output value --------------------------------
    // Combinational output pixel. Nearest: buf0[out_col>>1]. Bilinear:
    // from the current row (buf[cur]) and the next row (buf[!cur], or
    // the current row itself on the last input row). Pixel-shuffle:
    // even output row from even_buf, odd output row from odd_buf.
    logic [7:0]  out_px;
    logic [12:0] bc;  // bilinear input col = out_col >> 1
    logic [7:0]  a_c, a_c1, b_c, b_c1;
    logic [15:0] bil_sum;
    assign bc = out_col >> 1;
    // Select current / next row pixels by the cur bit.
    assign a_c  = cur ? buf1[bc] : buf0[bc];
    assign a_c1 = (bc >= {1'b0, width_r} - 13'd1) ?
                  (cur ? buf1[bc] : buf0[bc]) :
                  (cur ? buf1[bc + 13'd1] : buf0[bc + 13'd1]);
    assign b_c  = last_in_row ? a_c :
                  (cur ? buf0[bc] : buf1[bc]);
    assign b_c1 = last_in_row ? a_c1 :
                  (bc >= {1'b0, width_r} - 13'd1) ?
                  (cur ? buf0[bc] : buf1[bc]) :
                  (cur ? buf0[bc + 13'd1] : buf1[bc + 13'd1]);
    always @* begin
        if (mode_r == MODE_NEAREST) begin
            out_px = buf0[out_col >> 1];
        end else if (mode_r == MODE_BILINEAR) begin
            if (!out_row[0] && !out_col[0]) begin
                out_px = a_c;
            end else if (!out_row[0] && out_col[0]) begin
                bil_sum = {8'd0, a_c} + {8'd0, a_c1} + 16'd1;
                out_px = bil_sum[8:1];
            end else if (out_row[0] && !out_col[0]) begin
                bil_sum = {8'd0, a_c} + {8'd0, b_c} + 16'd1;
                out_px = bil_sum[8:1];
            end else begin
                bil_sum = {8'd0, a_c} + {8'd0, a_c1} + {8'd0, b_c} +
                          {8'd0, b_c1} + 16'd2;
                out_px = bil_sum[9:2];
            end
        end else begin
            out_px = out_row[0] ? odd_buf[out_col] : even_buf[out_col];
        end
    end

    //------------------------- main FSM ------------------------------------
    // Row-load counts: nearest/bilinear load W pixels per input row;
    // pixel-shuffle loads 4W pixels (4 channels per position).
    logic [12:0] load_total;
    assign load_total = (mode_r == MODE_SHUFFLE) ?
                        ({1'b0, width_r} << 2) : {1'b0, width_r};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_LOAD;
            in_row <= 13'd0; in_col <= 13'd0;
            out_row <= 13'd0; out_col <= 13'd0;
            cur <= 1'b0;
            load_phase <= 1'b0;
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
        end else if (!stall) begin
            out_valid <= 1'b0;
            out_sof <= 1'b0; out_eol <= 1'b0; out_last <= 1'b0;
            case (state)
                ST_LOAD: begin
                    if (in_valid) begin
                        if (mode_r == MODE_SHUFFLE) begin
                            // 4 channels per position: c = in_col>>2,
                            // ch = in_col[1:0]. ch0/ch1 -> even_buf[2c],
                            // even_buf[2c+1]; ch2/ch3 -> odd_buf[2c],
                            // odd_buf[2c+1]. (Using in_col>>1 as the
                            // base is wrong: it is 2c+1 for ch2/ch3.)
                            if (in_col[1:0] == 2'd0)
                                even_buf[(in_col >> 2) << 1] <= pixel_in;
                            else if (in_col[1:0] == 2'd1)
                                even_buf[((in_col >> 2) << 1) + 13'd1] <= pixel_in;
                            else if (in_col[1:0] == 2'd2)
                                odd_buf[(in_col >> 2) << 1] <= pixel_in;
                            else
                                odd_buf[((in_col >> 2) << 1) + 13'd1] <= pixel_in;
                        end else if (mode_r == MODE_BILINEAR) begin
                            // Row 0, phase 0 -> buf0 (cur=0). Row 0,
                            // phase 1 -> buf1. Later rows -> the
                            // non-current buffer.
                            if (in_row == 13'd0 && !load_phase)
                                buf0[in_col] <= pixel_in;
                            else if (in_row == 13'd0 && load_phase)
                                buf1[in_col] <= pixel_in;
                            else if (cur)
                                buf0[in_col] <= pixel_in;
                            else
                                buf1[in_col] <= pixel_in;
                        end else begin
                            buf0[in_col] <= pixel_in;
                        end
                        if (in_col >= load_total - 13'd1) begin
                            in_col <= 13'd0;
                            out_col <= 13'd0;
                            if (mode_r == MODE_BILINEAR && in_row == 13'd0 &&
                                !load_phase && !last_in_row) begin
                                // Row 0 loaded; load row 1 before
                                // emitting (the odd output row needs it).
                                load_phase <= 1'b1;
                                state <= ST_LOAD;
                            end else begin
                                load_phase <= 1'b0;
                                state <= ST_EMIT;
                            end
                        end else begin
                            in_col <= in_col + 13'd1;
                        end
                    end
                end
                ST_EMIT: begin
                    pixel_out <= out_px;
                    out_valid <= 1'b1;
                    out_sof <= (out_row == 13'd0) && (out_col == 13'd0);
                    out_eol <= (out_col == ({1'b0, width_r} << 1) - 13'd1);
                    out_last <= (out_row == ({1'b0, height_r} << 1) - 13'd1) &&
                                (out_col == ({1'b0, width_r} << 1) - 13'd1);
                    if (out_col >= ({1'b0, width_r} << 1) - 13'd1) begin
                        out_col <= 13'd0;
                        out_row <= out_row + 13'd1;
                        if (!out_row[0]) begin
                            // Just finished the even output row; emit
                            // the odd output row from the same buffers.
                            state <= ST_EMIT;
                        end else begin
                            // Just finished the odd output row; advance
                            // to the next input row, or return to idle
                            // (ST_LOAD) when the frame is complete.
                            // Bilinear: the next row was already loaded
                            // as the "next" row, so if it is the last
                            // row, emit directly (no load; the odd
                            // output row replicates the current row).
                            if (last_in_row) begin
                                in_row <= 13'd0;
                                out_row <= 13'd0;
                                cur <= 1'b0;
                                load_phase <= 1'b0;
                                state <= ST_LOAD;
                            end else if (mode_r == MODE_BILINEAR &&
                                         in_row >= {1'b0, height_r} - 13'd2) begin
                                in_row <= in_row + 13'd1;
                                cur <= ~cur;
                                state <= ST_EMIT;
                            end else begin
                                in_row <= in_row + 13'd1;
                                if (mode_r == MODE_BILINEAR)
                                    cur <= ~cur;
                                state <= ST_LOAD;
                            end
                        end
                    end else begin
                        out_col <= out_col + 13'd1;
                    end
                end
                default: state <= ST_LOAD;
            endcase
        end
    end
endmodule
