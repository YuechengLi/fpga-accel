#!/usr/bin/env python3
"""Stage 6 supplement: per-layer RTL-vs-float error (pure Python).

The RTL output equals the functional simulator output exactly for
every conv layer (max|err| = 0, verified by the RTL sims), and within
1 LSB for the PWL sigmoid. So the per-layer RTL-vs-float error is
identical to the per-layer functional-simulator-vs-float error, which
is computed here in pure Python (no RTL sim, no re-run).

For each conv layer: quantize the float per-layer output to the
layer's output scale (uint8, or signed int8 when the layer is signed)
and compare against the functional simulator's per-layer output
(max|err| and PSNR). For RetinexNet the float reference uses ZERO
padding (the same as the RTL) so the numbers are the quantization
error only; the replicate-vs-zero border difference is reported
separately by the main Stage 6 run (border_analysis).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import stage6
import models_torch
from funcsim import fixed_forward, float_forward
from passes.fusion import fuse
from compile import infer_shapes


def quantize_float(vals, r_out, signed_out):
    """Quantize a float array to the layer's output scale."""
    scale = 255.0 / (2.0 ** r_out)
    v = np.round(np.asarray(vals, dtype=np.float64) * scale)
    if signed_out:
        return np.clip(v, -128, 127).astype(np.int64)
    return np.clip(v, 0, 255).astype(np.uint8)


def float_per_layer(model_name, torch_model, inp_f, input_shape, zero_pad=False):
    """Float forward; returns (float per-layer vals dict, final out).
    If zero_pad, force pad_mode='zero' on all conv nodes."""
    x_t = torch.from_numpy(
        np.transpose(inp_f, (2, 0, 1))[None, :, :, :].astype(np.float32))
    graph, _state = stage6.capture_torch(torch_model, x_t, model_name)
    fused, _g, _r = fuse(graph)
    infer_shapes(fused, input_shape)
    if zero_pad:
        for n in fused.nodes:
            if n.op in ("conv2d", "dw_conv2d", "linear"):
                n.kwargs["pad_mode"] = "zero"
    vals, out = float_forward(fused, inp_f)
    return vals, out


def analyze(model_name, torch_model, inp_f, inp_u8, input_shape, zero_pad=False):
    program, per_layer, _o, _f = stage6.compile_and_sim(
        model_name, torch_model, inp_f, inp_u8, input_shape)
    if zero_pad:
        # Use the zero-pad functional simulator as the reference (the
        # RTL uses zero padding; the replicate-pad funcsim would differ
        # at the borders and pollute the quantization comparison).
        zp = stage6.zero_pad_program(program)
        sim_zp = {"ops": zp["ops"], "weights": zp["_weights_arrays"],
                  "act_tables": zp["_act_tables_full"],
                  "outputs": zp["outputs"]}
        per_layer, _out_zp = fixed_forward(sim_zp, inp_u8)
    fvals, _fout = float_per_layer(model_name, torch_model, inp_f,
                                   input_shape, zero_pad=zero_pad)
    print(f"== {model_name} (float ref: {'zero-pad' if zero_pad else 'model pad'}) ==")
    res = {}
    for op in program["ops"]:
        if op["op"] not in ("conv2d", "dw_conv2d", "linear"):
            continue
        nm = op["name"]
        r_out = int(op.get("r_out", 0))
        signed_out = bool(op.get("signed_out", False))
        f_u8 = quantize_float(fvals[nm], r_out, signed_out)
        s_u8 = per_layer[nm]
        # The funcsim output may be a subset of channels in the RTL
        # run, but per_layer here is the full funcsim output.
        e = stage6.max_abs_err(s_u8.astype(np.float64), f_u8.astype(np.float64))
        p = stage6.psnr(s_u8.astype(np.float64), f_u8.astype(np.float64))
        print(f"  {nm}: vs float max|err|={e:.2f} PSNR={p:.2f} dB "
              f"(r_out={r_out} signed_out={signed_out})")
        res[nm] = dict(max_err_float=float(e), psnr_float=float(p),
                       r_out=r_out, signed_out=signed_out)
    # Act node (RetinexNet sigmoid): funcsim vs float.
    for op in program["ops"]:
        if op["op"] != "act":
            continue
        nm = op["name"]
        r_out = int(op.get("r_out", 0))
        signed_out = bool(op.get("signed_out", False))
        f_u8 = quantize_float(fvals[nm], r_out, signed_out)
        s_u8 = per_layer[nm]
        e = stage6.max_abs_err(s_u8.astype(np.float64), f_u8.astype(np.float64))
        p = stage6.psnr(s_u8.astype(np.float64), f_u8.astype(np.float64))
        print(f"  {nm} (act): vs float max|err|={e:.2f} PSNR={p:.2f} dB")
        res[nm] = dict(max_err_float=float(e), psnr_float=float(p),
                       r_out=r_out, signed_out=signed_out)
    return res


def main():
    all_res = {}
    # SRCNN (zero pad, same as RTL).
    inp_u8, inp_f = stage6.build_srcnn_input(24)
    all_res["srcnn"] = analyze("srcnn", models_torch.SRCNN(), inp_f,
                               inp_u8, (24, 24, 1), zero_pad=False)
    # DnCNN (zero pad, same as RTL). Full 17 layers.
    inp_u8, _clean, inp_f = stage6.build_dncnn_input(16)
    all_res["dncnn"] = analyze("dncnn", models_torch.DnCNNFragment(n_layers=17),
                               inp_f, inp_u8, (16, 16, 1), zero_pad=False)
    # RetinexNet (zero-pad float reference, apples-to-apples).
    inp_u8, inp_f = stage6.build_retinexnet_input(16)
    all_res["retinexnet"] = analyze("retinexnet",
                                    models_torch.RetinexNetDecom(), inp_f,
                                    inp_u8, (16, 16, 4), zero_pad=True)
    import json
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "logs", "stage6", "per_layer_float.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
