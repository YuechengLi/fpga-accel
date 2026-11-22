#!/usr/bin/env python3
"""Stage 7b acceptance test: compile SRCNN in streaming mode and verify
it against the iterative program and the float reference.

  torch model (models_torch.SRCNN, real pretrained weights)
    -> capture_torch (torch.export FX graph)
    -> compile_model (streaming=True) -> fused streaming program
    -> compile_model (streaming=False) -> iterative program
    -> fixed_forward on both -> must match EXACTLY (max|err| = 0)
    -> PSNR of the streaming output vs the float reference must
       reproduce the Stage 6 number (end-to-end 43.51 dB on the
       24x24 test input).

Writes the streaming program (program.json, program.txt, weights.bin)
to the outdir (default out/srcnn_streaming) -- the same files the
pipeline vector generator (sw/gen_stage7b_pipe.py) and the C driver
host test consume.

Usage: python3 demo_srcnn_streaming.py [--outdir DIR]
Exit code 0 iff all checks pass.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
import models_torch
from ir import capture_torch
from compile import compile_model, infer_shapes
from funcsim import fixed_forward, float_forward
from passes.fusion import fuse
from models.vision.common import psnr, max_abs_err
from emit import write_program
from stage6 import build_srcnn_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/srcnn_streaming")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    outdir = os.path.join(root, args.outdir) \
        if not os.path.isabs(args.outdir) else args.outdir

    inp_u8, inp_f = build_srcnn_input(24)
    print(f"input {inp_u8.shape}, range [{inp_u8.min()},{inp_u8.max()}]")

    model = models_torch.SRCNN()
    x_t = torch.from_numpy(
        np.transpose(inp_f, (2, 0, 1))[None, :, :, :].astype(np.float32))
    g, _ = capture_torch(model, x_t, "srcnn")
    prog_s, instrs_s, blob_s, rep_s = compile_model(
        g, inp_f, (24, 24, 1), model_name="srcnn", streaming=True)
    g2, _ = capture_torch(model, x_t, "srcnn")
    prog_i, instrs_i, blob_i, rep_i = compile_model(
        g2, inp_f, (24, 24, 1), model_name="srcnn", streaming=False)

    eng = prog_s["streaming"]["engines"]
    print("streaming engines:")
    for e in eng:
        print(f"  {e['layer']}: {e['engine']} K={e['K']} "
              f"{e['C_in']}->{e['C_out']} shift={e['shift']} "
              f"DMA {e['ncoeff']} @ {e['dma_dst']:#06x}")
    print(f"streaming: {len(instrs_s)} instrs, {len(blob_s)} B blob; "
          f"iterative: {len(instrs_i)} instrs, {len(blob_i)} B blob")

    _, out_s = fixed_forward(prog_s, inp_u8)
    _, out_i = fixed_forward(prog_i, inp_u8)
    e_si = int(np.abs(out_s.astype(np.int32)
                      - out_i.astype(np.int32)).max())
    print(f"streaming vs iterative fixed_forward: max|err| = {e_si}")
    ok = e_si == 0

    # Float reference (the fused graph, as in stage6.compile_and_sim).
    fused, _g, _r = fuse(g)
    infer_shapes(fused, (24, 24, 1))
    _vals, float_out = float_forward(fused, inp_f)
    float_ref255 = np.clip(float_out * 255.0, 0, 255)[:, :, 0]
    p = psnr(out_s[:, :, 0], float_ref255)
    e_fl = max_abs_err(out_s[:, :, 0], float_ref255)
    print(f"streaming vs float: PSNR = {p:.2f} dB "
          f"(Stage 6: 43.51 dB), max|err| = {e_fl:.2f}")
    ok = ok and p >= 43.0

    jp, tp, bp = write_program(prog_s, instrs_s, blob_s, outdir)
    print(f"wrote {jp}\n      {tp}\n      {bp}")
    print("STREAMING DEMO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
