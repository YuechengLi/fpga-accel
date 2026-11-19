//=============================================================================
// pool.sv -- Stage 3: streaming 2x2 max/average pooling + global average
// pooling (GAP), SystemVerilog.
//
// Modes (config 0x00): 0 = MAX 2x2, 1 = AVG 2x2, 2 = GAP.
//   * MAX/AVG 2x2: input is zero-padded on the fly to even W/H, then pooled
//     over 2x2 blocks. Output size out_W = ceil(W/2), out_H = ceil(H/2).
//     AVG divides by 4 (sum >> 2, zeros in the padded region included).
//     One line buffer (BRAM) + a 2x2 window. The line-buffer output lags
//     the stream by one column, so the current row is delayed by one cycle
//     to align: at the delayed cycle the four pixels of the block ending
//     at (r,c) are above_left (r-1,c-1), above (r-1,c), left (r,c-1),
//     cur (r,c). Output is produced when the delayed position has r and
//     c both odd (bottom-right of a 2x2 block).
//   * GAP: accumulate the sum of all REAL input pixels (32-bit), then emit
//     a single value (sum >> gap_shift, saturated to 8-bit) at end of
//     frame. gap_shift is a config (host sets log2(W*H) for power-of-2
//     sizes). The first pixel is added on start_frame; gap_done is set
//     when the padded stream ends, even if the last padded position is
//     padding.
//   * A single stall signal (out_valid && !out_ready) freezes all state.
//   * Streaming, no frame buffer (one line buffer for 2x2 modes only).
//
// Config bus (write between frames; single-cycle writes; 8-bit address):
//   addr 0x00 : mode (0=MAX 2x2, 1=AVG 2x2, 2=GAP)
//   addr 0x01 : image width  (1..4095)
//   addr 0x02 : image height (1..4095)
//   addr 0x03 : gap_shift (0..15, GAP only)
// Reset defaults: mode=0 (MAX), W=16, H=16, gap_shift=8.
//=============================================================================
`timescale 1ns/1ps

module pool #(
    parameter int MAX_W = 4096
)(
    input  logic        clk,
    input  logic        rst_n,
    // config bus
    input  logic        cfg_valid,
    input  logic        cfg_we,
    input  logic [7:0]  cfg_addr,
    input  logic [15:0] cfg_data,
    // stream in (single pixels)
    input  logic [7:0]  pixel_in,
    input  logic        in_valid,
    input  logic        in_sof,
    output logic        in_ready,
    // stream out (single pixels)
    output logic [7:0]  pixel_out,
    output logic        out_valid,
    output logic        out_sof,
    output logic        out_eol,
    output logic        out_last,
    input  logic        out_ready,
    output logic        busy
);
    localparam logic [7:0] CFG_MODE  = 8'h00;
    localparam logic [7:0] CFG_WIDTH = 8'h01;
    localparam logic [7:0] CFG_HEIGHT = 8'h02;
    localparam logic [7:0] CFG_GAP_SHIFT = 8'h03;

    localparam int LB_DEPTH = MAX_W + 32;
    localparam int ADDR_W = (LB_DEPTH <= 2048) ? 11 :
                            (LB_DEPTH <= 4096) ? 12 : 13;

    //------------------------- config registers ----------------------------
    logic [1:0]  mode_r;
    logic [11:0] width_r, height_r;
    logic [3:0]  gap_shift_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r <= 2'd0;
            width_r <= 12'd16;
            height_r <= 12'd16;
            gap_shift_r <= 4'd8;
        end else if (cfg_valid && cfg_we) begin
            unique case (cfg_addr)
                CFG_MODE:      mode_r <= cfg_data[1:0];
                CFG_WIDTH:     width_r <= cfg_data[11:0];
                CFG_HEIGHT:    height_r <= cfg_data[11:0];
                CFG_GAP_SHIFT: gap_shift_r <= cfg_data[3:0];
                default: ;
            endcase
        end
    end

    //------------------------- derived geometry ---------------------------
    // Padded (even) dimensions. Output size for 2x2 modes.
    logic [12:0] pw;      // padded width  = W rounded up to even
    logic [12:0] ph;      // padded height = H rounded up to even
    logic [12:0] out_w, out_h;
    assign pw = {1'b0, width_r} + {12'd0, width_r[0]};
    assign ph = {1'b0, height_r} + {12'd0, height_r[0]};
    assign out_w = pw >> 1;
    assign out_h = ph >> 1;

    //------------------------- stream control ------------------------------
    // Walk the PADDED image in raster order. A position is "real" when it
    // maps inside the input image; otherwise feed zero (padding).
    // The padded image starts at (0,0), which is always real.
    logic        frame_active;
    logic [12:0] strm_r, strm_c;
    logic [12:0] next_c, next_r;
    logic        is_last_stream;
    logic        pos_is_real;
    logic        stall;
    logic        start_frame;
    logic        stream_en;
    logic        pipe_empty;
    logic        can_start;
    logic [7:0]  stream_data;

    assign next_c = (strm_c == pw - 13'd1) ? 13'd0 : strm_c + 13'd1;
    assign next_r = (strm_c == pw - 13'd1) ? strm_r + 13'd1 : strm_r;
    assign is_last_stream = (strm_r == ph - 13'd1) && (strm_c == pw - 13'd1);
    assign pos_is_real = (strm_r < {1'b0, height_r}) &&
                         (strm_c < {1'b0, width_r});

    // pipeline valid: window stage for 2x2 modes; gap_done for GAP.
    logic win_v;
    logic gap_done;
    assign pipe_empty = !frame_active && !win_v && !gap_done &&
                        (out_valid == 1'b0);
    assign can_start  = !frame_active && pipe_empty;
    assign start_frame = can_start && in_valid && in_sof;
    assign stall = out_valid && !out_ready;
    assign stream_en = !stall && (start_frame ||
                         (frame_active && (!pos_is_real || in_valid)));
    assign stream_data = pos_is_real ? pixel_in : 8'd0;
    // in_ready does NOT require frame_active: on the start_frame cycle
    // frame_active is still 0, and the first pixel must be accepted then
    // (Stage 1 convention).
    assign in_ready = !stall && pos_is_real;
    assign busy = !can_start;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_active <= 1'b0;
            strm_r <= 13'd0;
            strm_c <= 13'd0;
        end else if (stream_en) begin
            if (is_last_stream) begin
                frame_active <= 1'b0;
                strm_r <= 13'd0;
                strm_c <= 13'd0;
            end else begin
                frame_active <= 1'b1;
                strm_r <= next_r;
                strm_c <= next_c;
            end
        end
    end

    //------------------------- line buffer (previous row) ------------------
    // Read-first BRAM, one row delay. Same template as Stages 1-2.
    (* ram_style = "block" *) logic [7:0] lb_mem [0:LB_DEPTH-1];
    logic [ADDR_W-1:0] lb_ptr;
    logic [7:0] lb_dout;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) lb_ptr <= '0;
        else if (stream_en) begin
            if (lb_ptr >= pw[ADDR_W-1:0] - 1'b1) lb_ptr <= '0;
            else lb_ptr <= lb_ptr + 1'b1;
        end
    end
    always_ff @(posedge clk) begin
        if (stream_en) begin
            lb_dout <= lb_mem[lb_ptr];
            lb_mem[lb_ptr] <= stream_data;
        end
    end

    //------------------------- 1-cycle alignment delay ---------------------
    // The line-buffer output lags the stream by one column, so the current
    // row is delayed by one cycle. At the delayed cycle (dly_v=1) the four
    // pixels of the block ending at (dly_r, dly_c) are:
    //   a = above_left_d1 = (r-1,c-1), b = lb_dout = (r-1,c),
    //   c = cur_left_d1 = (r,c-1),     d = cur_d1 = (r,c).
    logic [7:0] cur_left, cur_d1, cur_left_d1, above_left_d1;
    logic [12:0] dly_r, dly_c;
    logic        dly_v;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cur_left <= 8'd0; cur_d1 <= 8'd0; cur_left_d1 <= 8'd0;
            above_left_d1 <= 8'd0;
            dly_r <= 13'd0; dly_c <= 13'd0; dly_v <= 1'b0;
        end else if (!stall) begin
            if (stream_en) begin
                cur_left <= stream_data;
                cur_d1 <= stream_data;
                cur_left_d1 <= cur_left;
                above_left_d1 <= lb_dout;
                dly_r <= strm_r;
                dly_c <= strm_c;
                dly_v <= 1'b1;
            end else begin
                dly_v <= 1'b0;
            end
        end
    end

    // window valid + output coords + four pixels ride one stage to output.
    logic        win_is_out;
    logic [12:0] win_or, win_oc;
    logic [7:0]  win_a, win_b, win_c, win_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            win_v <= 1'b0; win_is_out <= 1'b0;
            win_or <= 13'd0; win_oc <= 13'd0;
            win_a <= 8'd0; win_b <= 8'd0; win_c <= 8'd0; win_d <= 8'd0;
        end else if (!stall) begin
            win_v <= dly_v;
            win_is_out <= dly_v && dly_r[0] && dly_c[0];
            win_or <= dly_r >> 1;
            win_oc <= dly_c >> 1;
            win_a <= above_left_d1;
            win_b <= lb_dout;
            win_c <= cur_left_d1;
            win_d <= cur_d1;
        end
    end

    //------------------------- 2x2 pool result -----------------------------
    logic [9:0] sum4;
    logic [7:0] max4;
    assign sum4 = {2'b00, win_a} + {2'b00, win_b} +
                  {2'b00, win_c} + {2'b00, win_d};
    always @* begin
        max4 = win_a;
        if (win_b > max4) max4 = win_b;
        if (win_c > max4) max4 = win_c;
        if (win_d > max4) max4 = win_d;
    end
    logic [7:0] pool_result;
    assign pool_result = (mode_r == 2'd0) ? max4 : sum4[9:2];

    //------------------------- GAP accumulator -----------------------------
    // Sum all REAL input pixels (padding excluded). The first pixel is
    // added on start_frame (position (0,0) is always real). gap_done is
    // set when the padded stream ends, even if the last padded position
    // is padding.
    logic [31:0] gap_sum;
    logic [31:0] gap_shifted;
    logic [7:0]  gap_result;
    assign gap_shifted = gap_sum >> gap_shift_r;
    always @* begin
        if (gap_shifted > 32'd255) gap_result = 8'd255;
        else                       gap_result = gap_shifted[7:0];
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gap_sum <= 32'd0;
            gap_done <= 1'b0;
        end else if (!stall) begin
            if (start_frame) begin
                gap_sum <= {24'd0, pixel_in};
                gap_done <= 1'b0;
            end else if (stream_en) begin
                if (pos_is_real) gap_sum <= gap_sum + {24'd0, pixel_in};
                if (is_last_stream) gap_done <= 1'b1;
            end else if (gap_done) begin
                gap_done <= 1'b0;
            end
        end
    end

    //------------------------- output stage --------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out <= 8'd0;
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_last <= 1'b0;
        end else if (!stall) begin
            if (mode_r == 2'd2) begin
                out_valid <= gap_done;
                pixel_out <= gap_result;
                out_sof <= gap_done;
                out_eol <= gap_done;
                out_last <= gap_done;
            end else begin
                out_valid <= win_v && win_is_out;
                pixel_out <= pool_result;
                out_sof <= win_v && win_is_out &&
                           (win_or == 13'd0) && (win_oc == 13'd0);
                out_eol <= win_v && win_is_out &&
                           (win_oc == out_w - 13'd1);
                out_last <= win_v && win_is_out &&
                            (win_or == out_h - 13'd1) &&
                            (win_oc == out_w - 13'd1);
            end
        end
    end
endmodule : pool
