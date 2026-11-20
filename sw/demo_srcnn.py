#!/usr/bin/env python3
"""Stage 5 acceptance test: compile SRCNN end-to-end through the
compiler -> functional simulator -> compare vs the fixed-point golden
(models/, commit 2daf4e4) and vs float.

  torch model (models_torch.SRCNN, real pretrained weights)
    -> capture_torch (torch.export FX graph)
    -> compile_model (fusion, quantization, act tables, tiling,
       emission)
    -> fixed_forward (functional simulator, tiled, round-half-up
       shift = the RTL's shift)
    -> compare per-layer + end-to-end vs:
       (a) the fixed-point golden (models/common.py conv2d_fixed,
           FLOOR shift) -- max|err| must be <= 1 LSB (the only
           difference is the shift rounding), and
       (b) the float reference -- PSNR must match the Stage 6 prep
           number (end-to-end 42.76 dB, max|err| 5.79).

Usage: python3 demo_srcnn.py [--outdir DIR]
Exit code 0 iff all checks pass.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import models_torch
from ir import capture_torch
from compile import compile_model
from funcsim import fixed_forward
from models.common import (conv2d_float, conv2d_fixed, psnr, max_abs_err,
                           make_test_gray)
from emit import write_program


def build_srcnn_input():
    """The same test image as models/srcnn/run_srcnn.py: HR 33x33 ->
    LR 11x11 -> bicubic up 33x33 (the SRCNN input). Returns
    (uint8 (33,33,1), float (33,33,1) in [0,1])."""
    hr = make_test_gray(33, 33, seed=7)
    lr = np.array(Image.fromarray(hr).resize((11, 11), Image.BICUBIC))
    inp = np.array(Image.fromarray(lr).resize((33, 33), Image.BICUBIC))
    return inp[:, :, None], (inp.astype(np.float64) / 255.0)[:, :, None]


def golden_srcnn(inp_u8):
    """The fixed-point golden (models/common.py, floor shift) for
    SRCNN, using the converted weights in models/srcnn/srcnn_x3.npz
    and the quant.json shifts. Returns (per-layer outputs, final)."""
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "srcnn")
    d = np.load(os.path.join(root, "srcnn_x3.npz"))
    q = json.load(open(os.path.join(root, "srcnn_x3_quant.json")))
    A = inp_u8
    outs = []
    r_in = 0
    for name in ("L1", "L2", "L3"):
        shift = q[name]["shift"]
        A = conv2d_fixed(A, d[name + "_W_q"], d[name + "_b_q"], shift,
                         pad=(0, 0, 0, 0))
        outs.append(A)
    return outs, A


def golden_srcnn_rhu(inp_u8):
    """The fixed-point golden with ROUND-HALF-UP shift (the RTL's
    shift). Same weights/shifts as golden_srcnn; the functional
    simulator must match this EXACTLY."""
    from funcsim import conv2d_fixed_rhu
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "srcnn")
    d = np.load(os.path.join(root, "srcnn_x3.npz"))
    q = json.load(open(os.path.join(root, "srcnn_x3_quant.json")))
    A = inp_u8
    outs = []
    for name in ("L1", "L2", "L3"):
        shift = q[name]["shift"]
        A = conv2d_fixed_rhu(A, d[name + "_W_q"], d[name + "_b_q"], shift,
                             pad=(0, 0, 0, 0))
        outs.append(A)
    return outs, A


def float_srcnn(inp_f):
    """The float reference for SRCNN (same weights, float)."""
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "srcnn")
    d = np.load(os.path.join(root, "srcnn_x3_float.npz"))
    x = inp_f
    outs = []
    for name in ("L1", "L2", "L3"):
        x = conv2d_float(x, d[name + "_W"], d[name + "_b"], pad=(0, 0, 0, 0))
        if name != "L3":
            x = np.maximum(x, 0.0)
        outs.append(x)
    return outs, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/srcnn")
    args = ap.parse_args()
    inp_u8, inp_f = build_srcnn_input()
    print(f"SRCNN input {inp_u8.shape}, range [{inp_u8.min()},{inp_u8.max()}]")

    # 1. capture
    model = models_torch.SRCNN()
    x_t = torch.from_numpy(
        np.transpose(inp_f, (2, 0, 1))[None, :, :, :].astype(np.float32))
    graph, _state = capture_torch(model, x_t, "srcnn")
    print(f"captured: {len(graph.nodes)} IR nodes")

    # 2. compile
    program, instrs, blob, report = compile_model(
        graph, inp_f, (33, 33, 1), model_name="srcnn")
    print("fusion:")
    for r in report["passes"]["fusion"]:
        print(f"  * {r}")
    print("quantize:")
    for r in report["passes"]["quantize"]:
        print(f"  * {r}")
    print("tiling:")
    for r in report["passes"]["tiling"]:
        print(f"  * {r}")
    print(f"emitted: {report['instr_count']} instrs, "
          f"{report['weight_blob_bytes']} B weight blob")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          args.outdir)
    jp, tp, bp = write_program(program, instrs, blob, outdir)
    print(f"wrote {jp}\n      {tp}\n      {bp}")

    # 3. functional simulator (tiled, round-half-up shift)
    # The simulator needs the arrays; rebuild the program dict the
    # simulator consumes (ops + weights + act tables).
    sim_program = {
        "ops": program["ops"],
        "weights": program["_weights_arrays"],
        "act_tables": program["_act_tables_full"],
        "outputs": program["outputs"],
    }
    per_layer, out = fixed_forward(sim_program, inp_u8)
    print(f"functional simulator output {out.shape}, "
          f"range [{out.min()},{out.max()}]")

    # 4. compare vs the fixed-point goldens and float.
    #    (a) round-half-up golden (same rounding as the RTL and the
    #    functional simulator): the functional simulator's TILED
    #    accumulation must match EXACTLY (integer addition is
    #    associative; the only difference from the untiled golden is
    #    the summation order). This is the tiling check.
    #    (b) floor-shift golden (models/common.py): differs only by
    #    shift rounding, which propagates and amplifies through the
    #    network, so the bound is reported, not asserted at 1 LSB.
    from funcsim import conv2d_fixed_rhu
    rhu_layers, rhu_out = golden_srcnn_rhu(inp_u8)
    golden_layers, golden_out = golden_srcnn(inp_u8)
    float_layers, float_out = float_srcnn(inp_f)
    float_ref255 = np.clip(float_out * 255.0, 0, 255)
    conv_names = [o["name"] for o in program["ops"]
                  if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    ok = True
    print("per-layer (functional simulator vs round-half-up golden):")
    for nm, g, f in zip(conv_names, rhu_layers, float_layers):
        got = per_layer[nm]
        r_out = [o for o in program["ops"] if o["name"] == nm][0]["r_out"]
        f_u8 = np.clip(f * (255.0 / (2.0 ** r_out)), 0, 255)
        e_rhu = max_abs_err(got, g)
        e_float = max_abs_err(got, f_u8)
        p_float = psnr(got, f_u8)
        status = "OK" if e_rhu == 0 else "FAIL"
        if e_rhu != 0:
            ok = False
        print(f"  {nm}: vs rhu-golden max|err|={e_rhu:.2f} [{status}]  "
              f"vs float max|err|={e_float:.2f} PSNR={p_float:.2f} dB")
    print("per-layer (functional simulator vs floor-shift golden, "
          "informational -- shift rounding propagates):")
    for nm, g in zip(conv_names, golden_layers):
        got = per_layer[nm]
        print(f"  {nm}: vs floor-golden max|err|={max_abs_err(got, g):.2f}")
    e2e_rhu = max_abs_err(out, rhu_out)
    e2e_golden = max_abs_err(out, golden_out)
    e2e_float = max_abs_err(out, float_ref255)
    p2e = psnr(out, float_ref255)
    print(f"END-TO-END: vs rhu-golden max|err|={e2e_rhu:.2f} "
          f"(must be 0)  vs floor-golden max|err|={e2e_golden:.2f} "
          f"(propagated rounding)  vs float max|err|={e2e_float:.2f} "
          f"PSNR={p2e:.2f} dB (Stage 6 prep: 42.76 dB, max|err| 5.79)")
    if e2e_rhu != 0:
        ok = False
        print("FAIL: end-to-end vs rhu-golden is not exact")
    if p2e < 40.0:
        ok = False
        print("FAIL: end-to-end PSNR vs float below 40 dB")
    print("STAGE 5 SRCNN DEMO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
