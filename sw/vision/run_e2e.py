#!/usr/bin/env python3
"""Integrated end-to-end driver (Stage 7c).

One system flow per model: a real test image (the Stage 6 input)
streams through the frontend RTL (3x3 Gaussian blur, same size),
the frontend's RTL output feeds the PE array driven by the
compiled tile-step program (sw/common/emit.py::emit_pe_program), one wave
per layer with the RTL output feeding the next layer (the Stage 6
pattern), and the final image is checked against the COMPOSED
golden (frontend golden -> per-layer funcsim golden) and the float
reference (float model on the blurred input).

Two vvp passes per model (not one TB): the tile-steps for layer
N+1 depend on the frontend's RTL output, which is only known
after phase 1 runs, and the scheduler does im2col in software
by design, so the RTL cannot generate its own tile-steps. The
integration is real at every stage -- a frontend bug changes
the array's input and fails the composed golden; a scheduler
bug changes the tile-steps and fails the per-layer golden.
Python only does the compiler's job (scheduling) and the
golden check, never the datapath.

Tails (model-specific, RTL, as in Stage 6):
  - RetinexNet: act.sv PWL sigmoid after the recon layer
    (stage6.run_act_sigmoid, budget max|err| <= 1 LSB).
  - DnCNN: residual.sv mode 3 (noisy - residual) for the
    denoised output (stage6.run_residual_dncnn).

Padding: the scheduler assumes a valid conv, so a padded
layer's input is pre-padded in the driver (zero for DnCNN,
replicate for RetinexNet) and the golden is computed on the
same padded input. The driver cross-checks the padded golden
against fixed_forward's per-layer golden for every layer; a
mismatch is a driver bug (fail fast).

Usage: python3 sw/vision/run_e2e.py [--model srcnn|dncnn|retinexnet|all]
                              [--outdir logs/stage7c]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "common"))
import numpy as np

import emit
from funcsim import conv2d_fixed_rhu
from models.vision.common import max_abs_err, pad2d, psnr
from sched_pe import assemble_output, write_layer_vectors
from stage6 import (build_dncnn_input, build_retinexnet_input,
                    build_srcnn_input, compile_and_sim,
                    run_act_sigmoid, run_residual_dncnn)
import models_torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TB = os.path.join(REPO, "tb")
sys.path.insert(0, TB)
import golden_frontend

# 3x3 Gaussian [1 2 1; 2 4 2; 1 2 1]/16 as int16 coeffs (x16) in the
# frontend's 5x5 grid (row stride 5); shift=8 divides by 256.
FE_COEFF = [16, 32, 16, 0, 0,
            32, 64, 32, 0, 0,
            16, 32, 16, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]


def layers_from_program(program):
    """Compiled program conv ops -> sched_pe layer dicts.

    act maps relu->1 / leaky->2 / none->0 (pe_emit codes);
    mode_dot=1 iff C_out==1 (SRCNN L3, DnCNN L17); pad/pad_mode
    are carried for the pre-padding (emit_pe_program applies
    them). sign_in/out come straight from the program ops.
    """
    layers = []
    weights = program["_weights_arrays"]
    for o in program["ops"]:
        if o["op"] not in ("conv2d", "dw_conv2d", "linear"):
            continue
        w = weights[o["name"]]
        act = {"relu": 1, "leaky": 2}.get(o.get("fused_act") or "none", 0)
        slope = (int(round(o.get("act_slope", 0) * 65536))
                 if act == 2 else 0)
        layers.append(dict(
            name=o["name"], K=o["K"], C_out=o["C_out"],
            mode_dot=1 if o["C_out"] == 1 else 0,
            tn=1 if o["C_out"] == 1 else 8,
            sign_in=1 if o.get("signed_in") else 0,
            sign_out=1 if o.get("signed_out") else 0,
            shift=o["shift"], act=act, slope=slope,
            W_q=w["W_q"], b_q=w["b_q"],
            pad=tuple(o.get("pad", (0, 0, 0, 0))),
            pad_mode=o.get("pad_mode", "zero")))
    return layers


def as_signed(arr):
    """Reinterpret uint8 bytes as signed int8 (two's complement)."""
    a = np.asarray(arr).astype(np.int32)
    return np.where(a > 127, a - 256, a)


def run_frontend(img_u8, outdir, tag):
    """Phase 1: stream the image through the frontend RTL
    (tb_frontend.sv, reused unchanged) and check its output
    against the Stage 7 golden. Returns (blurred uint8, max|err|).
    """
    H, W, ch = img_u8.shape
    cs = dict(w=W, h=H, k=3, shift=8, pt=1, pb=1, pl=1, pr=1,
              sx=1, sy=1, ds=1, ch=ch, sign=0, act=0, slope=0,
              coeff=FE_COEFF,
              img=[int(v) for v in img_u8.reshape(-1)])
    with open(os.path.join(TB, "vectors_frontend.txt"), "w") as f:
        f.write("1\n")
        f.write(f"{W} {H} 3 8 1 1 1 1 1 1 1 {ch} 0 0 0\n")
        f.write(" ".join(str(v) for v in FE_COEFF) + "\n")
        f.write(" ".join(str(v) for v in cs["img"]) + "\n")
    t0 = time.time()
    comp = subprocess.run(
        ["iverilog", "-g2012", "-o", "tb_fe_e2e.vvp",
         "../rtl/vision/frontend.sv", "tb_frontend.sv"],
        cwd=TB, capture_output=True, text=True)
    if comp.returncode != 0:
        raise RuntimeError("frontend iverilog failed:\n"
                           + comp.stdout + comp.stderr)
    run = subprocess.run(["vvp", "tb_fe_e2e.vvp"], cwd=TB,
                         capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    if run.returncode != 0 or "ERROR" in run.stdout:
        raise RuntimeError(f"frontend vvp failed rc={run.returncode}:\n"
                           + run.stdout[-2000:] + run.stderr[-2000:])
    vals = [int(x) for x in
            open(os.path.join(TB, "out_frontend.txt")).read().split()]
    if len(vals) != H * W * ch:
        raise RuntimeError(f"frontend output short: {len(vals)} vs "
                           f"{H * W * ch}")
    exp = golden_frontend.golden(cs)
    err = max((abs(a - b) for a, b in zip(vals, exp)), default=0)
    out = np.array(vals, dtype=np.uint8).reshape((H, W, ch))
    print(f"  [{tag}] frontend: {H}x{W}x{ch} blur, "
          f"max|err|={err} vs golden ({dt:.1f}s)", flush=True)
    return out, err


def read_pe_outputs(path):
    """Read out_pe_array.txt -> list of (mtile, nchunk, bytes[128])."""
    toks = open(path).read().split()
    out = []
    i = 0
    while i < len(toks):
        mtile = int(toks[i])
        nchunk = int(toks[i + 1])
        i += 2
        byts = [int(toks[i + j]) for j in range(128)]
        i += 128
        out.append((mtile, nchunk, byts))
    return out


def compile_e2e_tb():
    """Compile tb_e2e.sv + the srcnn_pipe_pe wrapper once (reused
    for every layer wave of a model)."""
    comp = subprocess.run(
        ["iverilog", "-g2012", "-o", "tb_e2e.vvp",
         "../rtl/vision/srcnn_pipe_pe.sv", "../rtl/common/pe_array.sv",
         "../rtl/common/pe_mac.sv", "../rtl/common/pe_emit.sv", "tb_e2e.sv"],
        cwd=TB, capture_output=True, text=True)
    if comp.returncode != 0:
        raise RuntimeError("e2e iverilog failed:\n"
                           + comp.stdout + comp.stderr)


def run_pe_layer(steps, shape, outdir, tag):
    """Phase 2, one wave: run one layer's tile-steps through the
    wrapper TB (tb_e2e.sv: srcnn_pipe_pe) and assemble the layer
    output. Returns (rtl_out uint8, wall_seconds)."""
    write_layer_vectors(steps, shape,
                        os.path.join(TB, "vectors_pe_array.txt"))
    t0 = time.time()
    run = subprocess.run(["vvp", "tb_e2e.vvp"], cwd=TB,
                         capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if run.returncode != 0 or "ERROR" in run.stdout:
        raise RuntimeError(f"e2e vvp failed rc={run.returncode} [{tag}]:\n"
                           + run.stdout[-2000:] + run.stderr[-2000:])
    if "WRAPPER CONFIG CHECK PASSED" not in run.stdout:
        raise RuntimeError(f"wrapper config check did not pass [{tag}]:\n"
                           + run.stdout[-2000:])
    outs = read_pe_outputs(os.path.join(TB, "out_pe_array.txt"))
    return assemble_output(outs, shape), dt


def load_ckpt(ckpt_dir):
    """Load a per-layer checkpoint (see run_chain). Returns
    (done, records) where done maps layer name -> saved RTL output
    (uint8) and records maps layer name -> the per-layer result
    record. A layer whose .npy fails to load (killed mid-write)
    truncates the checkpoint at that layer -- layers after it
    depend on it, so they are dropped too."""
    ckpt_path = os.path.join(ckpt_dir, "ckpt.json")
    done, records = {}, {}
    if not os.path.exists(ckpt_path):
        return done, records
    try:
        with open(ckpt_path) as f:
            ck = json.load(f)
    except (ValueError, OSError):
        return done, records
    for lname in ck.get("order", []):
        rec = ck["layers"].get(lname)
        if rec is None:
            break
        path = os.path.join(ckpt_dir, f"layer_{lname}.npy")
        try:
            arr = np.load(path)
        except (OSError, ValueError):
            break
        if list(arr.shape) != rec["shape"] or arr.dtype != np.uint8:
            break
        done[lname] = arr
        records[lname] = rec
    return done, records


def save_ckpt(ckpt_dir, lname, rtl_out, record, order):
    """Save one layer's RTL output + record, atomically (tmp file +
    os.replace, so a kill mid-write never leaves a half-written
    checkpoint behind)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = os.path.join(ckpt_dir, f"layer_{lname}.npy.tmp")
    # np.save appends ".npy" to a plain filename, so save through
    # a file object to control the exact tmp name.
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(rtl_out, dtype=np.uint8))
    os.replace(tmp, os.path.join(ckpt_dir, f"layer_{lname}.npy"))
    ckpt_path = os.path.join(ckpt_dir, "ckpt.json")
    ck = {"order": order, "layers": {}}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                ck = json.load(f)
        except (ValueError, OSError):
            ck = {"order": order, "layers": {}}
    ck["layers"][lname] = record
    if lname not in ck["order"]:
        ck["order"].append(lname)
    tmpj = ckpt_path + ".tmp"
    with open(tmpj, "w") as f:
        json.dump(ck, f, indent=2)
    os.replace(tmpj, ckpt_path)


def run_chain(model_name, torch_model, inp_u8, inp_f, shape, outdir,
              resume=False):
    """Full integrated flow for one model. Returns (ok, results).

    resume: continue from a per-layer checkpoint under
    <outdir>/ckpt_<model>/ (one .npy + ckpt.json per completed
    layer, written only after that layer passes max|err|==0).
    The VM is replaced every few tens of minutes and a kill
    mid-run loses the in-memory chain; the checkpoint makes a
    re-launch continue from the last passed layer instead of
    restarting the model. A run without --resume ignores any
    checkpoint. The frontend phase is fast (under 1 s) and always
    re-runs; the compile is re-run too (the program is
    deterministic).
    """
    print(f"=== {model_name}: integrated end-to-end ===", flush=True)
    results = {}
    # Phase 1: frontend. RetinexNet's input is [max,R,G,B]; the
    # frontend is 1/3ch, so it runs on the RGB channels and the max
    # channel is recomputed from the blurred RGB (the same input
    # construction Stage 6 uses, applied to the preprocessed image).
    if model_name == "retinexnet":
        rgb = inp_u8[:, :, 1:4]
        blurred_rgb, fe_err = run_frontend(rgb, outdir, "frontend")
        mx = blurred_rgb.max(axis=2, keepdims=True)
        blurred = np.concatenate([mx, blurred_rgb], axis=2)
    else:
        blurred, fe_err = run_frontend(inp_u8, outdir, "frontend")
    ok = (fe_err == 0)
    results["frontend"] = dict(max_err=int(fe_err),
                               shape=list(blurred.shape))
    if not ok:
        print(f"  frontend FAILED (max|err|={fe_err})")
        return False, results
    # Compile on the blurred input: program + per-layer funcsim
    # goldens + float reference, all for the true (blurred) input.
    blurred_f = blurred.astype(np.float64) / 255.0
    program, per_layer, _out, float_out = compile_and_sim(
        model_name, torch_model, blurred_f, blurred, shape)
    layers = layers_from_program(program)
    pe_program, steps_by_layer = emit.emit_pe_program(
        layers, blurred, "zu19eg")
    n_total = sum(L["n_steps"] for L in pe_program["layers"])
    print(f"  compiled program: {len(layers)} layers, "
          f"{n_total} tile-steps (streaming_pe, zu19eg)", flush=True)
    # Phase 2: wave-by-wave through the wrapper TB (compile once).
    compile_e2e_tb()
    ckpt_dir = os.path.join(outdir, f"ckpt_{model_name}")
    done, ckpt_records = (load_ckpt(ckpt_dir) if resume
                          else ({}, {}))
    if done:
        print(f"  resuming from checkpoint: {len(done)} layer(s) "
              f"already passed ({', '.join(done)})", flush=True)
    results.update(ckpt_records)
    cur = blurred
    rtl_map = {}
    for (lname, steps, lshape) in steps_by_layer:
        L = next(x for x in layers if x["name"] == lname)
        op = next(o for o in program["ops"] if o["name"] == lname)
        if lname in done:
            # Already passed in an earlier (killed) run: reuse the
            # saved RTL output, no vvp. The next layer's input is
            # reconstructed exactly as the live path does it.
            rtl_out = done[lname]
            print(f"  {lname}: resumed from checkpoint "
                  f"(max|err|=0, {len(steps)} steps skipped)",
                  flush=True)
            rtl_map[lname] = rtl_out
            cur = as_signed(rtl_out).astype(np.int64) \
                if L["sign_out"] else rtl_out
            continue
        padded = (pad2d(cur, L["pad"], L["pad_mode"])
                  if any(L["pad"]) else cur)
        # Golden on the padded input; cross-check it against
        # fixed_forward's per-layer golden (padding equivalence).
        golden = conv2d_fixed_rhu(
            padded, L["W_q"], L["b_q"], L["shift"], pad=(0, 0, 0, 0),
            signed_out=bool(L["sign_out"]))
        ref = per_layer[lname]
        cross = int(max_abs_err(
            golden.astype(np.int64), ref.astype(np.int64)))
        if cross != 0:
            print(f"  {lname}: GOLDEN MISMATCH (padded golden vs "
                  f"fixed_forward max|err|={cross}) -- driver bug")
            return False, results
        rtl_out, dt = run_pe_layer(steps, lshape, outdir, lname)
        if L["sign_out"]:
            cmp_rtl = as_signed(rtl_out)
        else:
            cmp_rtl = rtl_out.astype(np.int32)
        err = int(max_abs_err(cmp_rtl, golden.astype(np.int64)))
        status = "PASS" if err == 0 else "FAIL"
        print(f"  {lname}: {len(steps)} steps, RTL vs funcsim "
              f"max|err|={err} [{status}] ({dt:.1f}s)", flush=True)
        record = dict(max_err_funcsim=err, steps=len(steps),
                      shape=list(lshape), wall_s=round(dt, 1),
                      sign_out=int(L["sign_out"]))
        results[lname] = record
        if err != 0:
            return False, results
        rtl_map[lname] = rtl_out
        # Checkpoint now: a later VM replacement must not lose
        # this passed layer.
        save_ckpt(ckpt_dir, lname, rtl_out, record,
                  [n for (n, _s, _sh) in steps_by_layer])
        # Next layer's input: signed layers feed int8 values.
        cur = as_signed(rtl_out).astype(np.int64) if L["sign_out"] \
            else rtl_out
    final = rtl_map[layers[-1]["name"]]
    # Tails + final check.
    if model_name == "retinexnet":
        act_op = next(o for o in program["ops"] if o["op"] == "act")
        table = program["_act_tables_full"][act_op["name"]]
        ref_sig = per_layer[act_op["name"]]
        ok_sig, sig_rtl, sig_err = run_act_sigmoid(
            as_signed(final).astype(np.int64), act_op, table,
            ref_sig, outdir)
        print(f"  sigmoid: RTL vs funcsim max|err|={sig_err} "
              f"[{'PASS' if ok_sig else 'FAIL'}] (PWL budget <= 1)",
              flush=True)
        results["sigmoid"] = dict(max_err=int(sig_err))
        if not ok_sig:
            return False, results
        final = sig_rtl
        r_rtl, l_rtl = final[:, :, 0:3], final[:, :, 3]
        r_fl = np.clip(float_out[:, :, 0:3] * 255.0, 0, 255)
        l_fl = np.clip(float_out[:, :, 3] * 255.0, 0, 255)
        p_r, p_l = psnr(r_rtl, r_fl), psnr(l_rtl, l_fl)
        print(f"  FINAL vs float: R PSNR={p_r:.2f} dB, "
              f"L PSNR={p_l:.2f} dB", flush=True)
        results["final"] = dict(psnr_R=float(p_r), psnr_L=float(p_l))
    elif model_name == "dncnn":
        noisy = blurred[:, :, 0]
        l17 = as_signed(rtl_map[layers[-1]["name"]][:, :, 0])
        denoised = run_residual_dncnn(noisy, l17, outdir)
        l17_ref = per_layer[layers[-1]["name"]][:, :, 0]
        den_ref = np.clip(noisy.astype(np.int64)
                          - l17_ref.astype(np.int64), 0, 255)
        e_den = int(max_abs_err(denoised.astype(np.int64),
                                den_ref.astype(np.int64)))
        den_fl = np.clip(noisy.astype(np.float64)
                         - float_out[:, :, 0] * 255.0, 0, 255)
        p_den = psnr(denoised, den_fl)
        print(f"  denoised: vs funcsim max|err|={e_den} "
              f"[{'PASS' if e_den == 0 else 'FAIL'}], "
              f"vs float PSNR={p_den:.2f} dB", flush=True)
        results["denoised"] = dict(max_err_funcsim=e_den,
                                   psnr_float=float(p_den))
        if e_den != 0:
            return False, results
        final = denoised
    else:  # srcnn
        ref255 = np.clip(float_out * 255.0, 0, 255)[:, :, 0]
        p = psnr(final[:, :, 0], ref255)
        print(f"  FINAL vs float: PSNR={p:.2f} dB", flush=True)
        results["final"] = dict(psnr_float=float(p))
    print(f"E2E {model_name.upper()}: ALL PASS")
    return True, results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["srcnn", "dncnn", "retinexnet", "all"])
    ap.add_argument("--outdir", default="logs/stage7c")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the per-layer checkpoint "
                    "(logs/stage7c/ckpt_<model>/) instead of "
                    "restarting the model; no-op without one")
    args = ap.parse_args()
    outdir = os.path.join(REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    want = ["srcnn", "dncnn", "retinexnet"] \
        if args.model == "all" else [args.model]
    specs = {
        "srcnn": (models_torch.SRCNN(), build_srcnn_input(24),
                  (24, 24, 1)),
        "dncnn": (models_torch.DnCNNFragment(n_layers=17),
                  # build_dncnn_input returns (noisy_u8, clean_u8,
                  # noisy_f) -- the float input is the third element.
                  (lambda t: (t[0], t[2]))(build_dncnn_input(16)),
                  (16, 16, 1)),
        "retinexnet": (models_torch.RetinexNetDecom(),
                       build_retinexnet_input(16), (16, 16, 4)),
    }
    all_ok = True
    summary = {}
    for name in want:
        model, (inp_u8, inp_f), shape = specs[name]
        ok, results = run_chain(name, model, inp_u8, inp_f, shape,
                                outdir, resume=args.resume)
        summary[name] = results
        with open(os.path.join(outdir, f"e2e_{name}.json"), "w") as f:
            json.dump(results, f, indent=2)
        all_ok = all_ok and ok
        if not ok:
            print(f"E2E {name.upper()}: FAIL")
            break
    print("E2E ALL PASS" if all_ok else "E2E FAILURES (see above)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
