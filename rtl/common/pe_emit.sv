//=============================================================================
// pe_emit.sv -- output stage for the PE array (Stage 7b).
//
// Combinational: round-half-up shift, then ReLU / Leaky, then saturate
// to a byte. Applied by the pe_array's merge unit to the FINAL merged
// sum of one (M-tile, N-chunk) -- after all its K-chunk partials are
// summed -- never to a partial. Mirrors conv2d_mc's emit stage (same
// quantization: the RTL shift is round-half-up; the slope is a Q0.16
// UNSIGNED fraction, so it is zero-extended to 17 bits -- a signed
// 16-bit 0x8000 would overflow to -32768 and flip the Leaky sign,
// the bug fixed in the L1/L2 engines).
//
// Ports are flat packed vectors (Yosys 0.52 rejects unpacked array
// ports). in_flat: TM*TN int32, index (m*TN+n). out_flat: TM*TN
// bytes, index (m*TN+n).
//=============================================================================

module pe_emit #(
    parameter int TM = 16,
    parameter int TN = 8
)(
    input  logic signed [TM*TN*32-1:0] in_flat,
    input  logic [3:0]  shift,      // output shift (0..15)
    input  logic [1:0]  act,        // 0 = none, 1 = ReLU, 2 = Leaky
    input  logic [15:0] slope,      // Leaky slope (Q0.16 unsigned)
    input  logic        sign_out,   // 1 = signed output bytes
    output logic [TM*TN*8-1:0] out_flat
);

    localparam int N = TM * TN;

    // Unpack in_flat into a signed array (generate assigns, constant
    // indices). An indexed part-select of a signed vector loses its
    // signedness in Icarus when used directly in arithmetic (the
    // tree8 bug: a negative value becomes a large positive and the
    // sum saturates). Assigning the part-select to a signed variable
    // copies the bit pattern, and arithmetic on the signed array
    // element is then correct (the adder_tree pattern).
    logic signed [31:0] acc_arr [0:N-1];
    genvar gu;
    generate
        for (gu = 0; gu < N; gu++) begin : gen_unpack
            assign acc_arr[gu] = in_flat[gu*32 +: 32];
        end
    endgenerate

    logic signed [31:0] emit_sh [0:N-1];
    logic signed [31:0] emit_act [0:N-1];
    logic signed [48:0] leaky_prod [0:N-1];

    genvar ge;
    generate
        for (ge = 0; ge < N; ge++) begin : gen_emit
            // Round-half-up shift: (acc + 2^(shift-1)) >> shift.
            assign emit_sh[ge] = (shift == 0) ? acc_arr[ge] :
                ((acc_arr[ge] + (32'sd1 <<< (shift - 1))) >>> shift);
            // Activation.
            assign leaky_prod[ge] = emit_sh[ge] * $signed({1'b0, slope});
            assign emit_act[ge] = (act == 2'd1) ? ((emit_sh[ge] < 0) ? 32'sd0 : emit_sh[ge]) :
                                  (act == 2'd2) ? ((emit_sh[ge] < 0) ? leaky_prod[ge][47:16] : emit_sh[ge]) :
                                                   emit_sh[ge];
            // Saturate to a byte (unsigned 0..255, or signed -128..127).
            assign out_flat[ge*8 +: 8] = sign_out ?
                ((emit_act[ge] > 127) ? 8'sd127 : ((emit_act[ge] < -128) ? -8'sd128 : emit_act[ge][7:0])) :
                ((emit_act[ge] > 255) ? 8'd255 : ((emit_act[ge] < 0) ? 8'd0 : emit_act[ge][7:0]));
        end
    endgenerate

endmodule : pe_emit
