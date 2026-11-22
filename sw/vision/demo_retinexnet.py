#!/usr/bin/env python3
"""Stage 5 demo: compile RetinexNet Decom-Net end-to-end through the
compiler -> functional simulator -> compare vs the float reference.

This is the demo that exercises the compiler's activation path:
  * conv0 has NO activation and a SIGNED output (~50% negative) --
    the compiler must carry a signed intermediate (signed int8, not
    a ReLU clip and not a uint8+offset, which breaks under zero
    padding -- see models/README.md).
  * the final sigmoid goes through the SEPARATE LUT act module
    (multi-level LUT, compiler-generated table) -- this validates
    the MLUT table generation end-to-end, not just on a grid.
  * replicate padding (folded into the convs by the fusion pass).

Comparison: the functional simulator (replicate padding, MLUT
sigmoid, signed intermediates) vs the float reference (replicate
padding). Expected (Stage 6 prep, fixed-point replicate): R PSNR
39.71 dB, L PSNR 45.11 dB. The RTL uses ZERO padding, so the
replicate-vs-zero border difference is reported separately (it is a
Stage 6 item: Stage 6 compares RTL (zero pad) against a zero-pad
float reference and reports the border separately).

Usage: python3 demo_retinexnet.py [--outdir DIR]
Exit code 0 iff the R/L PSNRs meet the budget (>= 35 dB R, >= 40 dB L).
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
from compile import compile_model
from funcsim import fixed_forward, float_forward
from models.vision.common import psnr, max_abs_err, make_test_rgb
from emit import write_program


def build_retinexnet_input():
    """Low-light RGB 32x32 -> x4 = [max, R, G, B] (the Decom-Net
    input). Returns (uint8 (32,32,4), float (32,32,4) in [0,1])."""
    rgb = make_test_rgb(32, 32, seed=7)
    low = np.clip(rgb.astype(np.float64) * 0.25, 0, 255).astype(np.uint8)
    mx = low.max(axis=2, keepdims=True)
    x4_u8 = np.concatenate([mx, low], axis=2)
    return x4_u8, (x4_u8.astype(np.float64) / 255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/retinexnet")
    args = ap.parse_args()
    inp_u8, inp_f = build_retinexnet_input()
    print(f"RetinexNet input {inp_u8.shape}, range [{inp_u8.min()},"
          f"{inp_u8.max()}]")

    model = models_torch.RetinexNetDecom()
    x_t = torch.from_numpy(
        np.transpose(inp_f, (2, 0, 1))[None, :, :, :].astype(np.float32))
    graph, _state = capture_torch(model, x_t, "retinexnet")
    print(f"captured: {len(graph.nodes)} IR nodes")

    program, instrs, blob, report = compile_model(
        graph, inp_f, (32, 32, 4), model_name="retinexnet")
    print("fusion (first 5):")
    for r in report["passes"]["fusion"][:5]:
        print(f"  * {r}")
    print(f"  ... ({len(report['passes']['fusion'])} fusion decisions)")
    print("quantize:")
    for r in report["passes"]["quantize"]:
        print(f"  * {r}")
    print("activation tables:")
    for r in report["passes"]["actapprox"]:
        print(f"  * {r}")
    print("tiling (first 3):")
    for r in report["passes"]["tiling"][:3]:
        print(f"  * {r}")
    print(f"emitted: {report['instr_count']} instrs, "
          f"{report['weight_blob_bytes']} B weight blob")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          args.outdir)
    jp, tp, bp = write_program(program, instrs, blob, outdir)
    print(f"wrote {jp}\n      {tp}\n      {bp}")

    # Float reference (replicate padding) via the fused graph.
    from passes.fusion import fuse
    fused, _g, _r = fuse(graph)
    from compile import infer_shapes
    infer_shapes(fused, (32, 32, 4))
    _vals, float_out = float_forward(fused, inp_f)
    # The fused graph's output is the sigmoid (R 3ch + L 1ch).
    R_f = float_out[:, :, 0:3]
    L_f = float_out[:, :, 3:4]

    # Functional simulator.
    sim_program = {
        "ops": program["ops"],
        "weights": program["_weights_arrays"],
        "act_tables": program["_act_tables_full"],
        "outputs": program["outputs"],
    }
    per_layer, out = fixed_forward(sim_program, inp_u8)
    R_s = out[:, :, 0:3].astype(np.float64)
    L_s = out[:, :, 3:4].astype(np.float64)
    R_f255 = np.clip(R_f * 255.0, 0, 255)
    L_f255 = np.clip(L_f * 255.0, 0, 255)
    pR = psnr(R_s, R_f255)
    pL = psnr(L_s, L_f255)
    print(f"END-TO-END (functional simulator vs float, replicate pad):")
    print(f"  R: PSNR={pR:.2f} dB max|err|={max_abs_err(R_s, R_f255):.2f} "
          f"(Stage 6 prep fixed-point: 39.71 dB)")
    print(f"  L: PSNR={pL:.2f} dB max|err|={max_abs_err(L_s, L_f255):.2f} "
          f"(Stage 6 prep fixed-point: 45.11 dB)")
    # Per-layer: conv0 signed output range (the signed-intermediate check).
    conv0 = per_layer["conv2d"]
    print(f"  conv0 (signed intermediate): range [{conv0.min()},"
          f"{conv0.max()}] frac negative "
          f"{float((conv0 < 0).mean()) * 100:.1f}%")
    ok = pR >= 35.0 and pL >= 40.0
    print("STAGE 5 RETINEXNET DEMO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
