#!/usr/bin/env python3
"""gen_stage6_dma.py -- generate the vector file for the Stage 6
DMA -> conv2d_mc integration test (tb_stage6_dma.sv).

Uses a REAL model slice: SRCNN layer 1 (9x9, 1 -> 64, ReLU), the
output channel with the largest activation range (ch 5, [0,132]), on
the 24x24 Stage 6 test input. The coefficients are
loaded through the AXI-MM weight DMA (from the TB's DRAM model) into
the conv2d_mc config bus; the control registers and the per-channel
bias are written directly (as the driver would). The expected output
is the Stage 5 functional simulator's conv2d_tiled for that single
output channel (which must equal the full layer's channel 0).

Vector format (tb/vectors_stage6_dma.txt):
  # K=.. shift=.. w=.. h=.. pt=.. pb=.. pl=.. pr=.. sx=.. sy=..
  #   cin=.. cout=.. ow=.. oh=.. sgn_in=.. sgn_out=..
  #   fuse_act=.. act_slope=.. bias=..
  # ncoeff=..
  C <idx> <hex>          (coefficients, KxK row-major for (o=0,c=0))
  # nin=..
  I <val>                (input pixels, one per line, raster order)
  # nout=..
  E <val>                (expected output pixels, one per line)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "sw", "vision"))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "sw", "vision"))

import stage6
import models_torch
from funcsim import conv2d_tiled


def main():
    inp_u8, inp_f = stage6.build_srcnn_input(24)
    program, per_layer, _out, _f = stage6.compile_and_sim(
        "srcnn", models_torch.SRCNN(), inp_f, inp_u8, (24, 24, 1))
    op = stage6.conv_op(program, "conv2d")
    W = program["_weights_arrays"]["conv2d"]
    W_q, b_q = W["W_q"], W["b_q"]
    K = op["K"]
    # Output channel with the largest activation range (channel 0 is
    # [0,1] -- too weak to catch a wrong coefficient; channel 5 is
    # [0,132]).
    ch = int(np.argmax(per_layer["conv2d"].max(axis=(0, 1))))
    # Slope as a Q0.16 integer (same conversion as stage6.mc_case);
    # the op stores the Leaky slope as a float (0.01 default).
    slope_q16 = int(round(op.get("act_slope", 0.01) * 65536.0))
    exp = conv2d_tiled(inp_u8, W_q[:, :, :, ch:ch + 1], b_q[ch:ch + 1],
                       op["shift"], pad=tuple(op["pad"]),
                       pad_mode=op.get("pad_mode", "zero"),
                       signed_out=bool((op.get("sign", 0) >> 1) & 1),
                       fused_act=op.get("fused_act"),
                       act_slope=op.get("act_slope", 8192),
                       C_in_tile=op.get("C_in_tile", 4))
    assert exp.shape == (16, 16, 1), exp.shape
    # Cross-check against the full layer's channel ch.
    full = per_layer["conv2d"][:, :, ch:ch + 1]
    assert np.array_equal(exp, full), "single-channel slice != full layer ch"
    coeffs = W_q[:, :, 0, ch].reshape(-1)  # KxK row-major
    assert coeffs.size == K * K
    # The full 121-entry grid (Stage 6 fix): the kernel at the
    # top-left KxK of the 11x11 grid (index r*11+c), zeros elsewhere.
    # The DMA writes tap k at dst + l*128 + k, so the blob must
    # carry the grid, not the contiguous K*K taps.
    grid = np.zeros(121, dtype=np.int16)
    for r in range(K):
        grid[r * 11:(r * 11) + K] = coeffs[r * K:(r * K) + K]
    bias = int(b_q[ch])
    fuse_num = stage6.FUSED_ACT_CODE.get(op.get("fused_act") or "none", 0)
    out_path = os.path.join("..", "..", "tb", "vectors_stage6_dma.txt")
    with open(out_path, "w") as f:
        f.write(f"# K={K} shift={op['shift']} w=24 h=24 "
                f"pt={op['pad'][0]} pb={op['pad'][1]} "
                f"pl={op['pad'][2]} pr={op['pad'][3]} "
                f"sx={op['stride'][0]} sy={op['stride'][1]} "
                f"cin=1 cout=1 ow=16 oh=16 "
                f"sgn_in={op.get('sign', 0) & 1} "
                f"sgn_out={(op.get('sign', 0) >> 1) & 1} "
                f"fuse_act={fuse_num} "
                f"act_slope={slope_q16} "
                f"bias={bias} taps=121\n")
        f.write(f"# ncoeff=121\n")
        for i, c in enumerate(grid):
            f.write(f"C {i} {int(c) & 0xffff:04x}\n")
        f.write(f"# nin={inp_u8.size}\n")
        for v in inp_u8[:, :, 0].reshape(-1):
            f.write(f"I {int(v)}\n")
        f.write(f"# nout={exp.size}\n")
        for v in exp[:, :, 0].reshape(-1):
            f.write(f"E {int(v)}\n")
    print(f"wrote {out_path}: K={K} ch={ch} ncoeff=121 (grid) bias={bias} "
          f"nin={inp_u8.size} nout={exp.size} "
          f"exp range [{exp.min()},{exp.max()}]")


if __name__ == "__main__":
    main()
