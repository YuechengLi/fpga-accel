#!/usr/bin/env python3
"""gen_stage6_sched.py -- generate the vector file for the Stage 6
scheduler-path check (tb_stage6_sched.sv).

Uses a REAL model slice: SRCNN layer 1 (9x9, 1 -> 64, ReLU), the
output channel with the largest activation range (ch 5, [0,132]), on
the 24x24 Stage 6 test input. The coefficients are loaded through
the AXI-MM weight DMA into the scheduler's ping-pong bank A, then
the scheduler streams them to the unified engine (dense mode).

The dense path has NO bias register (a documented follow-up), so
the reference is the functional simulator with the bias FORCED to 0
for this channel. The vector file states this explicitly.

Vector format (tb/vectors_stage6_sched.txt):
  # K=.. shift=.. w=.. h=.. pt=.. pb=.. pl=.. pr=.. sx=.. sy=..
  #   ow=.. oh=.. sgn_in=.. sgn_out=.. fuse_act=.. act_slope=..
  #   bias=0 (forced -- dense path has no bias register)
  # ncoeff=121
  C <idx> <hex>          (coefficients, full 121-entry grid)
  # nin=..
  I <val>                (input pixels, raster order)
  # nout=..
  E <val>                (expected output pixels, bias=0 reference)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sw"))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sw"))

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
    ch = int(np.argmax(per_layer["conv2d"].max(axis=(0, 1))))
    # Reference with the bias FORCED to 0 (dense path has no bias
    # register -- a documented Stage 6 follow-up, not a model change).
    b0 = np.zeros(1, dtype=np.int64)
    exp = conv2d_tiled(inp_u8, W_q[:, :, :, ch:ch + 1], b0,
                       op["shift"], pad=tuple(op["pad"]),
                       pad_mode=op.get("pad_mode", "zero"),
                       signed_out=bool((op.get("sign", 0) >> 1) & 1),
                       fused_act=op.get("fused_act"),
                       act_slope=op.get("act_slope", 8192),
                       C_in_tile=op.get("C_in_tile", 4))
    assert exp.shape == (16, 16, 1), exp.shape
    coeffs = W_q[:, :, 0, ch].reshape(-1)  # KxK row-major
    assert coeffs.size == K * K
    grid = np.zeros(121, dtype=np.int16)
    for r in range(K):
        grid[r * 11:(r * 11) + K] = coeffs[r * K:(r * K) + K]
    slope_q16 = int(round(op.get("act_slope", 0.01) * 65536.0))
    fuse_num = stage6.FUSED_ACT_CODE.get(op.get("fused_act") or "none", 0)
    out_path = os.path.join("..", "tb", "vectors_stage6_sched.txt")
    with open(out_path, "w") as f:
        f.write(f"# K={K} shift={op['shift']} w=24 h=24 "
                f"pt={op['pad'][0]} pb={op['pad'][1]} "
                f"pl={op['pad'][2]} pr={op['pad'][3]} "
                f"sx={op['stride'][0]} sy={op['stride'][1]} "
                f"ow=16 oh=16 "
                f"sgn_in={op.get('sign', 0) & 1} "
                f"sgn_out={(op.get('sign', 0) >> 1) & 1} "
                f"fuse_act={fuse_num} "
                f"act_slope={slope_q16} bias=0\n")
        f.write(f"# ncoeff=121\n")
        for i, c in enumerate(grid):
            f.write(f"C {i} {int(c) & 0xffff:04x}\n")
        f.write(f"# nin={inp_u8.size}\n")
        for v in inp_u8[:, :, 0].reshape(-1):
            f.write(f"I {int(v)}\n")
        f.write(f"# nout={exp.size}\n")
        for v in exp[:, :, 0].reshape(-1):
            f.write(f"E {int(v)}\n")
    print(f"wrote {out_path}: K={K} ch={ch} ncoeff=121 (grid) "
          f"bias=0 (forced) nin={inp_u8.size} nout={exp.size} "
          f"exp range [{exp.min()},{exp.max()}]")


if __name__ == "__main__":
    main()
