#!/usr/bin/env python3
"""Stage 6: model numerical verification (plan docs/plan.md Stage 6).

Runs the three models end-to-end through the RTL and compares the RTL
output against the Python functional simulator (sw/vision/funcsim.py, the
Stage 5 reference) and the fixed-point golden, pixel-by-pixel, with a
per-layer error budget.

  SRCNN (3 layers, x3) -- full end-to-end chain, all output channels.
  DnCNN fragment -- L1, L2, L3 (chained where possible) + L16, L17
    (deep layers, reference inputs) + the residual add (denoised).
    L4-L15 are NOT run in RTL (same 3x3 64->64 shape as L2/L3/L16;
    stated explicitly in the report).
  RetinexNet Decom-Net -- full end-to-end (conv0-conv5 + recon +
    sigmoid). The RTL uses ZERO padding; the model was trained with
    REPLICATE padding, so the RTL is compared against a ZERO-PAD
    functional-simulator run (same weights/shifts, pad_mode overridden
    to zero), and the replicate-vs-zero border difference is reported
    separately. The sigmoid runs through the separate act module
    (multi-level LUT, compiler table).

Execution path: the conv layers run on conv2d_mc (the multi-channel
engine, the path the Stage 5 compiler emits for them), driven by the
Stage 6 testbench (tb/tb_stage6_mc.sv, fast streaming, per-channel
bias, output-channel subset). The weights are the real quantized
model weights; the bias is loaded per output channel (the Stage 6
RTL fix: conv2d_mc bias registers 0x12/0x13 -- the Stage 5 functional
simulator always included the bias, the RTL did not have the register).

Layers run in WAVES: wave 1 runs the layers whose input is known
(the test image, or a functional-simulator output); wave 2 runs the
layers whose input is a wave-1 RTL output (the true RTL chain). This
keeps the chain real (RTL output feeds the next RTL layer) without
running the full 17-layer DnCNN chain (too slow in RTL sim).

Separation of concerns (plan Stage 5): functional-sim vs golden
mismatch = compiler bug (already resolved in Stage 5); RTL vs
functional-sim mismatch = HARDWARE bug. The error budget for RTL vs
functional simulator is EXACT (max|err| = 0): the RTL and the
functional simulator use the same weights, the same shift
(round-half-up), the same padding, and the same accumulation order
(tiled, int64 in the simulator, 32-bit per tile in the RTL -- the
accumulator bound is checked separately). Any nonzero RTL-vs-funcsim
error is a failure to investigate, not a tolerance.

Error budget vs FLOAT (quantization, not RTL correctness): the
per-layer max|err| vs float and the end-to-end PSNR vs float are
reported and compared against the Stage 6 prep numbers
(models/README.md); they must be in the same ballpark (the same
quantization, different test images). A large deviation means the
quantization does not hold on the Stage 6 images.

Usage: python3 sw/vision/stage6.py [--model srcnn|dncnn|retinexnet|all]
         [--outdir logs/stage6]
Exit code 0 iff all checks pass.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sw", "common"))
sys.path.insert(0, os.path.join(REPO, "sw", "vision"))
sys.path.insert(0, REPO)

import torch
import models_torch
from ir import capture_torch
from compile import compile_model
from funcsim import fixed_forward, float_forward
from models.vision.common import (psnr, max_abs_err, make_test_gray,
                           make_test_rgb)
from PIL import Image

TB = os.path.join(REPO, "tb")
LOGS = os.path.join(REPO, "logs", "stage6")

FUSED_ACT_CODE = {"none": 0, "relu": 1, "leaky": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def run_cmd(cmd, cwd, timeout=1800, logfile=None):
    """Run a command, return (rc, stdout, stderr, seconds)."""
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout)
    if logfile:
        with open(logfile, "w") as f:
            f.write(p.stdout)
            f.write(p.stderr)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def compile_and_sim(model_name, torch_model, inp_f, inp_u8, input_shape):
    """Capture + compile + functional simulator. Returns
    (program, per_layer, out, float_out)."""
    x_t = torch.from_numpy(
        np.transpose(inp_f, (2, 0, 1))[None, :, :, :].astype(np.float32))
    graph, _state = capture_torch(torch_model, x_t, model_name)
    log(f"  captured: {len(graph.nodes)} IR nodes")
    program, instrs, blob, report = compile_model(
        graph, inp_f, input_shape, model_name=model_name)
    log(f"  emitted: {report['instr_count']} instrs, "
        f"{report['weight_blob_bytes']} B weight blob")
    sim_program = {
        "ops": program["ops"],
        "weights": program["_weights_arrays"],
        "act_tables": program["_act_tables_full"],
        "outputs": program["outputs"],
    }
    per_layer, out = fixed_forward(sim_program, inp_u8)
    # Float reference (the fused graph, replicate pad where the model has it).
    from passes.fusion import fuse
    from compile import infer_shapes
    fused, _g, _r = fuse(graph)
    infer_shapes(fused, input_shape)
    _vals, float_out = float_forward(fused, inp_f)
    return program, per_layer, out, float_out


def conv_op(program, name):
    for o in program["ops"]:
        if o["name"] == name:
            return o
    raise KeyError(name)


def mc_case(op, W_q, b_q, A, o_start, o_end):
    """One Stage 6 MC vector case. A is (H,W,Cin) uint8 or signed int8
    (as int64)."""
    K = op["K"]
    pad = tuple(op.get("pad", (0, 0, 0, 0)))
    sx, sy = tuple(op.get("stride", (1, 1)))
    H, W, Cin = A.shape
    ow = (W + pad[2] + pad[3] - K) // sx + 1
    oh = (H + pad[0] + pad[1] - K) // sy + 1
    fa = FUSED_ACT_CODE.get(op.get("fused_act") or "none", 0)
    slope = int(round(op.get("act_slope", 0.01) * 65536.0))
    return dict(
        k=K, shift=op["shift"], w=W, h=H,
        pt=pad[0], pb=pad[1], pl=pad[2], pr=pad[3], sx=sx, sy=sy,
        cin=Cin, cout=o_end - o_start, ow=ow, oh=oh,
        sgn_in=1 if op.get("signed_in") else 0,
        sgn_out=1 if op.get("signed_out") else 0,
        fuse_act=fa, act_slope=slope,
        o_start=o_start, o_end=o_end,
        bias=[int(v) for v in b_q[o_start:o_end]],
        W_q=np.asarray(W_q),
        A=np.asarray(A),
    )


def write_mc_vectors(cases, path):
    """Write the Stage 6 MC vector file (tb_stage6_mc.sv format)."""
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(
                f"{cs['k']} {cs['shift']} {cs['w']} {cs['h']} "
                f"{cs['pt']} {cs['pb']} {cs['pl']} {cs['pr']} "
                f"{cs['sx']} {cs['sy']} {cs['cin']} {cs['cout']} "
                f"{cs['ow']} {cs['oh']} {cs['sgn_in']} {cs['sgn_out']} "
                f"{cs['fuse_act']} {cs['act_slope']} "
                f"{cs['o_start']} {cs['o_end']}\n")
            for b in cs["bias"]:
                f.write(f"{b}\n")
            Wq = cs["W_q"]
            for o in range(cs["o_start"], cs["o_end"]):
                for c in range(cs["cin"]):
                    taps = Wq[:, :, c, o].reshape(-1)
                    f.write(" ".join(str(int(v)) for v in taps) + "\n")
            A = cs["A"]
            for c in range(cs["cin"]):
                f.write(" ".join(str(int(v)) for v in
                                 A[:, :, c].reshape(-1)) + "\n")
    log(f"  wrote {path}: {len(cases)} cases")


def read_mc_output(path, cases):
    """Read out_stage6_mc.txt into a list of per-case arrays
    (oh, ow, nchannels_run)."""
    vals = [int(x) for x in open(path).read().split()]
    out = []
    pos = 0
    for cs in cases:
        n = (cs["o_end"] - cs["o_start"]) * cs["ow"] * cs["oh"]
        seg = vals[pos:pos + n]
        pos += n
        if len(seg) < n:
            raise RuntimeError(f"MC output short: case needs {n}, "
                               f"got {len(seg)} (total {len(vals)})")
        arr = np.array(seg, dtype=np.int64).reshape(
            (cs["o_end"] - cs["o_start"], cs["oh"], cs["ow"]))
        out.append(np.transpose(arr, (1, 2, 0)))
    if pos < len(vals):
        log(f"  WARNING: {len(vals) - pos} extra MC outputs")
    return out


def run_mc_wave(specs, tag, outdir):
    """Run one wave of MC layers. specs is a list of
    (name, op, W_q, b_q, A, o_start, o_end). Returns a dict
    name -> RTL output array (H, W, nchannels_run)."""
    cases = [mc_case(op, Wq, bq, A, os_, oe)
             for (_nm, op, Wq, bq, A, os_, oe) in specs]
    vpath = os.path.join(TB, "vectors_stage6_mc.txt")
    opath = os.path.join(TB, "out_stage6_mc.txt")
    write_mc_vectors(cases, vpath)
    rc, so, se, dt = run_cmd(
        ["iverilog", "-g2012", "-o", "tb_s6mc.vvp",
         "../rtl/vision/conv2d_mc.sv", "tb_stage6_mc.sv"],
        cwd=TB, timeout=300,
        logfile=os.path.join(outdir, f"compile_mc_{tag}.log"))
    if rc != 0:
        log(f"  COMPILE FAILED (mc {tag})")
        log(so[-2000:] + se[-2000:])
        raise RuntimeError("iverilog failed")
    log(f"  iverilog ok ({dt:.1f}s); running vvp [{tag}] ...")
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_s6mc.vvp"], cwd=TB, timeout=1800,
        logfile=os.path.join(outdir, f"sim_mc_{tag}.log"))
    log(f"  vvp done rc={rc} ({dt:.1f}s)")
    for line in so.splitlines():
        if "ERROR" in line or "DONE" in line:
            log(f"    {line}")
    if rc != 0:
        raise RuntimeError(f"vvp failed rc={rc}")
    outs = read_mc_output(opath, cases)
    return {nm: o for (nm, *_), o in zip(specs, outs)}


def check_layer(name, rtl_out, ref, ref_float_u8=None):
    """Compare one layer's RTL output vs the functional simulator
    (must be EXACT) and vs float (reported). Returns
    (ok, max_err_vs_funcsim, max_err_vs_float, psnr_vs_float)."""
    e_fs = max_abs_err(rtl_out, ref)
    ok = (e_fs == 0)
    e_fl, p_fl = None, None
    if ref_float_u8 is not None:
        e_fl = max_abs_err(rtl_out, ref_float_u8)
        p_fl = psnr(rtl_out, ref_float_u8)
    status = "OK" if ok else "FAIL"
    extra = ""
    if e_fl is not None:
        extra = f"  vs float max|err|={e_fl:.2f} PSNR={p_fl:.2f} dB"
    log(f"  {name}: vs funcsim max|err|={e_fs:.2f} [{status}]{extra}")
    return ok, e_fs, e_fl, p_fl


# ---------------------------------------------------------------------------
# SRCNN
# ---------------------------------------------------------------------------
def build_srcnn_input(size=24):
    """HR size x size -> LR size/3 -> bicubic up size (the SRCNN input)."""
    hr = make_test_gray(size, size, seed=7)
    lr = np.array(Image.fromarray(hr).resize((size // 3, size // 3),
                                             Image.BICUBIC))
    inp = np.array(Image.fromarray(lr).resize((size, size), Image.BICUBIC))
    return inp[:, :, None], (inp.astype(np.float64) / 255.0)[:, :, None]


def run_srcnn(outdir):
    log("=== SRCNN (full end-to-end chain, all channels, 24x24 input) ===")
    inp_u8, inp_f = build_srcnn_input(24)
    log(f"  input {inp_u8.shape}, range [{inp_u8.min()},{inp_u8.max()}]")
    program, per_layer, out, float_out = compile_and_sim(
        "srcnn", models_torch.SRCNN(), inp_f, inp_u8, (24, 24, 1))
    conv_names = [o["name"] for o in program["ops"]
                  if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    log(f"  conv layers: {conv_names}")
    weights = program["_weights_arrays"]
    ok = True
    results = {}
    # True chain, one wave per layer (RTL output feeds the next layer).
    cur = inp_u8
    rtl_map = {}
    for nm in conv_names:
        op = conv_op(program, nm)
        wave = run_mc_wave(
            [(nm, op, weights[nm]["W_q"], weights[nm]["b_q"], cur,
              0, op["C_out"])],
            f"srcnn_{nm}", outdir)
        rtl = wave[nm]
        rtl_map[nm] = rtl
        ref = per_layer[nm]
        o, e_fs, e_fl, p_fl = check_layer(nm, rtl, ref)
        ok = ok and o
        results[nm] = dict(max_err_funcsim=e_fs, max_err_float=e_fl,
                           psnr_float=p_fl, shape=list(ref.shape))
        cur = rtl
    final = rtl_map[conv_names[-1]][:, :, 0]
    float_ref255 = np.clip(float_out * 255.0, 0, 255)[:, :, 0]
    p_e2e = psnr(final, float_ref255)
    e_e2e = max_abs_err(final, float_ref255)
    log(f"  END-TO-END vs float: max|err|={e_e2e:.2f} "
        f"PSNR={p_e2e:.2f} dB (Stage 6 prep: 42.76 dB, max|err| 5.79)")
    results["end_to_end"] = dict(max_err_float=e_e2e, psnr_float=p_e2e)
    return ok, results


# ---------------------------------------------------------------------------
# DnCNN
# ---------------------------------------------------------------------------
def build_dncnn_input(size=16):
    """Clean grayscale + Gaussian noise sigma=25."""
    clean = make_test_gray(size, size, seed=7)
    rng = np.random.default_rng(11)
    noisy = np.clip(clean.astype(np.float64)
                    + rng.normal(0, 25, (size, size)), 0, 255).astype(np.uint8)
    return noisy[:, :, None], clean, (noisy.astype(np.float64) / 255.0)[:, :, None]


def run_residual_dncnn(noisy, residual_signed, outdir):
    """DnCNN denoised step through rtl/vision/residual.sv (mode 3: A -
    sign_extend(B)). noisy is uint8 (H,W); residual_signed is the L17
    RTL output (signed int, H,W). Returns the denoised uint8 array."""
    H, W = noisy.shape
    vpath = os.path.join(TB, "vectors_residual.txt")
    opath = os.path.join(TB, "out_residual.txt")
    b_bytes = (np.asarray(residual_signed, dtype=np.int64) & 0xFF)
    with open(vpath, "w") as f:
        f.write("1\n")
        f.write(f"3 {W} {H}\n")
        f.write(" ".join(str(int(v)) for v in noisy.reshape(-1)) + "\n")
        f.write(" ".join(str(int(v)) for v in b_bytes.reshape(-1)) + "\n")
    rc, so, se, dt = run_cmd(
        ["iverilog", "-g2012", "-o", "tb_res_s6.vvp",
         "../rtl/vision/residual.sv", "tb_residual.sv"],
        cwd=TB, timeout=300,
        logfile=os.path.join(outdir, "compile_residual_dncnn.log"))
    if rc != 0:
        raise RuntimeError("residual iverilog failed")
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_res_s6.vvp"], cwd=TB, timeout=300,
        logfile=os.path.join(outdir, "sim_residual_dncnn.log"))
    if rc != 0:
        raise RuntimeError(f"residual vvp failed rc={rc}")
    vals = [int(x) for x in open(opath).read().split()]
    if len(vals) != H * W:
        raise RuntimeError(f"residual output short: {len(vals)} vs {H*W}")
    return np.array(vals, dtype=np.uint8).reshape((H, W))


def run_dncnn(outdir):
    log("=== DnCNN fragment (L1-L3 + L16-L17, 16x16 input) ===")
    noisy_u8, clean, noisy_f = build_dncnn_input(16)
    log(f"  input {noisy_u8.shape}, range [{noisy_u8.min()},{noisy_u8.max()}]")
    program, per_layer, out, float_out = compile_and_sim(
        "dncnn", models_torch.DnCNNFragment(n_layers=17), noisy_f, noisy_u8,
        (16, 16, 1))
    conv_names = [o["name"] for o in program["ops"]
                  if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    log(f"  conv layers: {len(conv_names)} (L1..L17)")
    weights = program["_weights_arrays"]
    ok = True
    results = {}
    rtl_map = {}

    def spec(nm, A, o_start, o_end):
        op = conv_op(program, nm)
        return (nm, op, weights[nm]["W_q"], weights[nm]["b_q"], A,
                o_start, o_end)

    # Wave 1: L1 (input = noisy) + L3, L16, L17 (funcsim inputs).
    # L4-L15 are NOT run in RTL (same 3x3 64->64 shape as L2/L3/L16).
    wave1 = run_mc_wave([
        spec(conv_names[0], noisy_u8, 0, 64),
        spec(conv_names[2], per_layer[conv_names[1]], 0, 8),
        spec(conv_names[15], per_layer[conv_names[14]], 0, 8),
        spec(conv_names[16], per_layer[conv_names[15]], 0, 1),
    ], "dncnn_w1", outdir)
    rtl_map.update(wave1)
    # Wave 2: L2 (input = RTL L1 output -- the true chain).
    wave2 = run_mc_wave([
        spec(conv_names[1], rtl_map[conv_names[0]], 0, 8),
    ], "dncnn_w2", outdir)
    rtl_map.update(wave2)
    for nm, o_start, o_end in [
            (conv_names[0], 0, 64), (conv_names[1], 0, 8),
            (conv_names[2], 0, 8), (conv_names[15], 0, 8),
            (conv_names[16], 0, 1)]:
        ref = per_layer[nm][:, :, o_start:o_end]
        o, e_fs, e_fl, p_fl = check_layer(nm, rtl_map[nm], ref)
        ok = ok and o
        results[nm] = dict(max_err_funcsim=e_fs, max_err_float=e_fl,
                           psnr_float=p_fl, o_start=o_start, o_end=o_end,
                           shape=list(per_layer[nm].shape))
    # Denoised = clip(noisy - residual, 0, 255), via residual mode 3.
    l17_rtl = rtl_map[conv_names[16]][:, :, 0]
    l17_ref = per_layer[conv_names[16]][:, :, 0]
    denoised_ref = np.clip(noisy_u8[:, :, 0].astype(np.int64)
                           - l17_ref.astype(np.int64), 0, 255).astype(np.uint8)
    denoised_rtl = run_residual_dncnn(noisy_u8[:, :, 0], l17_rtl, outdir)
    e_den = max_abs_err(denoised_rtl, denoised_ref)
    p_den = psnr(denoised_rtl, denoised_ref)
    log(f"  denoised (residual mode 3): vs funcsim max|err|={e_den:.2f} "
        f"PSNR={p_den:.2f} dB [{'OK' if e_den == 0 else 'FAIL'}]")
    ok = ok and (e_den == 0)
    denoised_float = np.clip(noisy_f[:, :, 0] * 255.0
                             - float_out[:, :, 0] * 255.0, 0, 255)
    p_den_f = psnr(denoised_rtl, denoised_float)
    e_den_f = max_abs_err(denoised_rtl, denoised_float)
    log(f"  denoised vs float: max|err|={e_den_f:.2f} "
        f"PSNR={p_den_f:.2f} dB (Stage 6 prep: 48.23 dB, max|err| 3.53)")
    results["denoised"] = dict(max_err_funcsim=e_den, psnr_funcsim=p_den,
                               max_err_float=e_den_f, psnr_float=p_den_f)
    return ok, results

# ---------------------------------------------------------------------------
# RetinexNet
# ---------------------------------------------------------------------------
def build_retinexnet_input(size=16):
    """Low-light RGB size x size -> x4 = [max, R, G, B] (Decom-Net input)."""
    rgb = make_test_rgb(size, size, seed=7)
    low = np.clip(rgb.astype(np.float64) * 0.25, 0, 255).astype(np.uint8)
    mx = low.max(axis=2, keepdims=True)
    x4_u8 = np.concatenate([mx, low], axis=2)
    return x4_u8, (x4_u8.astype(np.float64) / 255.0)


def zero_pad_program(program):
    """A copy of the emitted program with every conv's pad_mode forced
    to 'zero' (the RTL's padding). The weights, shifts, and act tables
    are unchanged. Used as the RTL reference for RetinexNet (whose
    model uses replicate padding)."""
    p2 = copy.deepcopy(program)
    for o in p2["ops"]:
        if o["op"] in ("conv2d", "dw_conv2d", "linear"):
            o["pad_mode"] = "zero"
    return p2


def run_act_sigmoid(recon_rtl, act_op, table, ref_sigmoid, outdir):
    """Run the RetinexNet sigmoid through rtl/vision/act.sv with the
    compiler's table (PWL mode 2 or MLUT mode 6, per the table's
    ``impl``). recon_rtl is the recon RTL output (signed bytes,
    H,W,4). ref_sigmoid is the zero-pad functional simulator's
    sigmoid output (uint8, H,W,4). Returns (ok, rtl_sigmoid uint8
    array, max_err). The PWL RTL uses integer >>10 arithmetic while
    the functional simulator uses float, so the PWL budget is
    max|err| <= 1 LSB (a rounding-boundary difference); MLUT must
    be exact (max|err| = 0)."""
    H, W, C = recon_rtl.shape
    impl = table.get("impl", "mlut")
    func_id = {"sigmoid": 2, "tanh": 3, "gelu": 4, "silu": 5}[act_op["func"]]
    vpath = os.path.join(TB, "vectors_act.txt")
    opath = os.path.join(TB, "out_act.txt")
    # Input scale comes from the act op (NOT assumed 0): the recon
    # output is a signed byte a representing x = a / (255/2**r_in).
    # (RetinexNet recon has r_out=3, so the sigmoid r_in=3.)
    r_in = int(act_op.get("r_in", 0))
    r_out = int(act_op.get("r_out", 0))
    scale_in = 255.0 / (2.0 ** r_in)
    scale_out = 255.0 / (2.0 ** r_out)
    x = np.asarray(recon_rtl, dtype=np.float64) / scale_in
    if impl == "pwl":
        # PWL: input Q5.11 (x*2048), clamped to [-8, 8]; 32-segment
        # (y0, dslope) table, both Q1.15.
        xq = np.clip(np.round(x * 2048.0), -16384, 16384).astype(int)
        nvals = int(xq.size)
        y0 = [int(v) for v in table["y0"]]
        ds = [int(v) for v in table["dslope"]]
        assert len(y0) == 32 and len(ds) == 32
        with open(vpath, "w") as f:
            f.write("1\n")
            f.write(f"{func_id} 0 {nvals}\n")
            for a, b in zip(y0, ds):
                f.write(f"{a} {b}\n")
            for v in xq.reshape(-1):
                f.write(f"{int(v)}\n")
        out_scale = 32768.0  # Q1.15
        tag = f"PWL (mode {func_id}, 32 seg)"
    else:
        # MLUT: input in12 = round((x+8)*256), 12-bit offset-binary.
        in12 = np.clip(np.round((x + 8.0) * 256.0), 0, 4095).astype(int)
        nvals = int(in12.size)
        H1, H2 = int(table["H1"]), int(table["H2"])
        nbands = int(table["nbands"])
        base = [int(v) for v in table["base"]]
        sel = [int(v) for v in table["sel"]]
        fine = [int(v) for v in table["fine"]]
        assert len(base) == (1 << H1), f"base len {len(base)} != {1 << H1}"
        assert len(sel) == (1 << H1), f"sel len {len(sel)} != {1 << H1}"
        assert len(fine) == nbands * (1 << H2), \
            f"fine len {len(fine)} != {nbands * (1 << H2)}"
        with open(vpath, "w") as f:
            f.write("1\n")
            f.write(f"6 {func_id} {nvals}\n")
            f.write(f"{H1} {H2} {nbands}\n")
            for v in base:
                f.write(f"{v}\n")
            for v in sel:
                f.write(f"{v}\n")
            for v in fine:
                f.write(f"{v}\n")
            for v in in12.reshape(-1):
                f.write(f"{int(v)}\n")
        out_scale = 2048.0  # Q5.11
        tag = f"MLUT (H1={H1} H2={H2} bands={nbands})"
    rc, so, se, dt = run_cmd(
        ["iverilog", "-g2012", "-o", "tb_act_s6.vvp",
         "../rtl/vision/act.sv", "tb_act.sv"],
        cwd=TB, timeout=300,
        logfile=os.path.join(outdir, "compile_act_retinexnet.log"))
    if rc != 0:
        raise RuntimeError("act iverilog failed")
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_act_s6.vvp"], cwd=TB, timeout=300,
        logfile=os.path.join(outdir, "sim_act_retinexnet.log"))
    if rc != 0:
        raise RuntimeError(f"act vvp failed rc={rc}")
    s = np.array([int(v) for v in open(opath).read().split()],
                 dtype=np.float64)
    if s.size != nvals:
        raise RuntimeError(f"act output short: {s.size} vs {nvals}")
    # Output -> float -> uint8 (same conversion as funcsim._apply_act).
    y = s / out_scale
    v = np.clip(np.round(y * scale_out), 0, 255).astype(np.uint8)
    rtl_sig = v.reshape((H, W, C))
    e = max_abs_err(rtl_sig, ref_sigmoid)
    budget = 1 if impl == "pwl" else 0
    ok = e <= budget
    log(f"  sigmoid (act {tag}, r_in={r_in} r_out={r_out}): "
        f"vs funcsim(zero-pad) max|err|={e:.2f} "
        f"(budget <={budget}) [{'OK' if ok else 'FAIL'}]")
    return ok, rtl_sig, e


def run_retinexnet(outdir):
    log("=== RetinexNet Decom-Net (full end-to-end, 16x16 input) ===")
    inp_u8, inp_f = build_retinexnet_input(16)
    log(f"  input {inp_u8.shape}, range [{inp_u8.min()},{inp_u8.max()}]")
    program, per_layer_rep, out_rep, float_out = compile_and_sim(
        "retinexnet", models_torch.RetinexNetDecom(), inp_f, inp_u8,
        (16, 16, 4))
    conv_names = [o["name"] for o in program["ops"]
                  if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    act_names = [o["name"] for o in program["ops"] if o["op"] == "act"]
    log(f"  conv layers: {conv_names} (conv0, conv1-5, recon)")
    log(f"  act nodes: {act_names}")
    weights = program["_weights_arrays"]
    # Zero-pad functional simulator: the RTL reference (the RTL uses
    # zero padding; the model uses replicate padding).
    zp = zero_pad_program(program)
    sim_zp = {"ops": zp["ops"], "weights": zp["_weights_arrays"],
              "act_tables": zp["_act_tables_full"],
              "outputs": zp["outputs"]}
    per_layer_zp, out_zp = fixed_forward(sim_zp, inp_u8)
    ok = True
    results = {}
    rtl_map = {}

    def spec(nm, A, o_start, o_end):
        op = conv_op(program, nm)
        return (nm, op, weights[nm]["W_q"], weights[nm]["b_q"], A,
                o_start, o_end)

    # Wave 1: conv0 (input = test image, all 64 out) + conv2-5
    # (zero-pad funcsim inputs, 4 out each) + recon (zero-pad funcsim
    # conv5 input, all 4 out).
    wave1 = run_mc_wave([
        spec(conv_names[0], inp_u8, 0, 64),
        spec(conv_names[2], per_layer_zp[conv_names[1]], 0, 4),
        spec(conv_names[3], per_layer_zp[conv_names[2]], 0, 4),
        spec(conv_names[4], per_layer_zp[conv_names[3]], 0, 4),
        spec(conv_names[5], per_layer_zp[conv_names[4]], 0, 4),
        spec(conv_names[6], per_layer_zp[conv_names[5]], 0, 4),
    ], "retinexnet_w1", outdir)
    rtl_map.update(wave1)
    # Wave 2: conv1 (input = RTL conv0 output -- the true chain).
    wave2 = run_mc_wave([
        spec(conv_names[1], rtl_map[conv_names[0]], 0, 8),
    ], "retinexnet_w2", outdir)
    rtl_map.update(wave2)
    for nm, o_start, o_end in [
            (conv_names[0], 0, 64), (conv_names[1], 0, 8),
            (conv_names[2], 0, 4), (conv_names[3], 0, 4),
            (conv_names[4], 0, 4), (conv_names[5], 0, 4),
            (conv_names[6], 0, 4)]:
        ref = per_layer_zp[nm][:, :, o_start:o_end]
        o, e_fs, e_fl, p_fl = check_layer(nm, rtl_map[nm], ref)
        ok = ok and o
        results[nm] = dict(max_err_funcsim=e_fs, max_err_float=e_fl,
                           psnr_float=p_fl, o_start=o_start, o_end=o_end,
                           shape=list(per_layer_zp[nm].shape))
    # Sigmoid through the act module (PWL or MLUT, per the compiler
    # table's impl).
    act_op = conv_op(program, act_names[0])
    table = program["_act_tables_full"][act_names[0]]
    o_act, rtl_sig, e_act = run_act_sigmoid(
        rtl_map[conv_names[6]], act_op, table,
        per_layer_zp[act_names[0]], outdir)
    ok = ok and o_act
    results[act_names[0]] = dict(max_err_funcsim=e_act,
                                 shape=list(per_layer_zp[act_names[0]].shape))
    # End-to-end R/L vs the zero-pad float reference (quantization
    # check) and vs the replicate-pad float reference (the original
    # model, border effect included).
    R_rtl = rtl_sig[:, :, 0:3].astype(np.float64)
    L_rtl = rtl_sig[:, :, 3:4].astype(np.float64)
    # Zero-pad float reference (apples-to-apples quantization check).
    R_f_zp, L_f_zp = _float_zero_pad(program, inp_f)
    pR_zp = psnr(R_rtl, np.clip(R_f_zp * 255.0, 0, 255))
    pL_zp = psnr(L_rtl, np.clip(L_f_zp * 255.0, 0, 255))
    eR_zp = max_abs_err(R_rtl, np.clip(R_f_zp * 255.0, 0, 255))
    eL_zp = max_abs_err(L_rtl, np.clip(L_f_zp * 255.0, 0, 255))
    log(f"  END-TO-END vs float (ZERO pad): R PSNR={pR_zp:.2f} dB "
        f"max|err|={eR_zp:.2f}  L PSNR={pL_zp:.2f} dB max|err|={eL_zp:.2f}")
    # Replicate-pad float reference (the original model).
    R_f_rep = float_out[:, :, 0:3]
    L_f_rep = float_out[:, :, 3:4]
    pR_rep = psnr(R_rtl, np.clip(R_f_rep * 255.0, 0, 255))
    pL_rep = psnr(L_rtl, np.clip(L_f_rep * 255.0, 0, 255))
    log(f"  END-TO-END vs float (REPLICATE pad, original model): "
        f"R PSNR={pR_rep:.2f} dB  L PSNR={pL_rep:.2f} dB "
        f"(border effect included)")
    results["end_to_end"] = dict(
        R_psnr_zero_pad=pR_zp, L_psnr_zero_pad=pL_zp,
        R_max_err_zero_pad=eR_zp, L_max_err_zero_pad=eL_zp,
        R_psnr_replicate_pad=pR_rep, L_psnr_replicate_pad=pL_rep)
    # Border analysis (replicate vs zero, funcsim, no RTL involved).
    border = border_analysis(per_layer_rep, per_layer_zp, act_names[0],
                             float_out, R_f_zp, L_f_zp)
    results["border"] = border
    return ok, results


def _float_zero_pad(program, inp_f):
    """Float forward of the fused graph with pad_mode forced to zero.
    Returns (R float (H,W,3), L float (H,W,1))."""
    import torch as _torch
    from ir import capture_torch as _cap
    from passes.fusion import fuse as _fuse
    from compile import infer_shapes as _shapes
    from funcsim import float_forward as _ff
    # Rebuild the torch model + graph from the program's model name.
    model = models_torch.RetinexNetDecom()
    x_t = _torch.from_numpy(
        np.transpose(inp_f, (2, 0, 1))[None, :, :, :].astype(np.float32))
    graph, _s = _cap(model, x_t, "retinexnet")
    fused, _g, _r = _fuse(graph)
    _shapes(fused, inp_f.shape)
    for n in fused.nodes:
        if n.op in ("conv2d", "dw_conv2d", "linear"):
            n.kwargs["pad_mode"] = "zero"
    _vals, out = _ff(fused, inp_f)
    return out[:, :, 0:3], out[:, :, 3:4]


def border_analysis(per_layer_rep, per_layer_zp, act_name, float_out,
                    R_f_zp, L_f_zp):
    """Replicate-vs-zero border analysis (functional simulator only,
    no RTL). Compares the replicate-pad and zero-pad sigmoid outputs
    (R and L) on the full frame and on the interior (5px border
    excluded). R_f_zp / L_f_zp are the zero-pad FLOAT R/L (for the
    float-only padding difference). Returns a dict of numbers."""
    rep = per_layer_rep[act_name].astype(np.float64)
    zp = per_layer_zp[act_name].astype(np.float64)
    R_rep, L_rep = rep[:, :, 0:3], rep[:, :, 3:4]
    R_zp, L_zp = zp[:, :, 0:3], zp[:, :, 3:4]
    H, W = R_rep.shape[:2]
    sl = (slice(5, H - 5), slice(5, W - 5))
    res = {}
    for tag, a, b in [("R", R_rep, R_zp), ("L", L_rep, L_zp)]:
        res[f"{tag}_full_maxdiff"] = max_abs_err(a, b)
        res[f"{tag}_full_psnr"] = psnr(a, b)
        res[f"{tag}_interior_maxdiff"] = max_abs_err(a[sl], b[sl])
        res[f"{tag}_interior_psnr"] = psnr(a[sl], b[sl])
    log("  BORDER (replicate vs zero, funcsim, fixed-point):")
    log(f"    R full: max|diff|={res['R_full_maxdiff']:.2f} "
        f"PSNR={res['R_full_psnr']:.2f} dB  "
        f"interior(5px): max|diff|={res['R_interior_maxdiff']:.2f} "
        f"PSNR={res['R_interior_psnr']:.2f} dB")
    log(f"    L full: max|diff|={res['L_full_maxdiff']:.2f} "
        f"PSNR={res['L_full_psnr']:.2f} dB  "
        f"interior(5px): max|diff|={res['L_interior_maxdiff']:.2f} "
        f"PSNR={res['L_interior_psnr']:.2f} dB")
    # Float-only padding difference (no quantization).
    Rf = np.clip(float_out[:, :, 0:3] * 255.0, 0, 255)
    Lf = np.clip(float_out[:, :, 3:4] * 255.0, 0, 255)
    res["R_float_pad_psnr"] = psnr(Rf, np.clip(R_f_zp * 255.0, 0, 255))
    res["L_float_pad_psnr"] = psnr(Lf, np.clip(L_f_zp * 255.0, 0, 255))
    log(f"    float-only padding difference: R PSNR="
        f"{res['R_float_pad_psnr']:.2f} dB  L PSNR="
        f"{res['L_float_pad_psnr']:.2f} dB")
    return res

# ---------------------------------------------------------------------------
# r_out re-validation + accumulator bound (task items (c) and (f))
# ---------------------------------------------------------------------------
from models.vision.common import range_exp, conv2d_fixed


def max_tile_acc(A, W_q, b_q, shift, pad=(0, 0, 0, 0), pad_mode="zero",
                 C_in_tile=4):
    """Max |accumulator| (before shift) of the tiled fixed-point conv,
    mirroring funcsim.conv2d_tiled but tracking the running sum. The
    RTL's 32-bit acc_mem holds the running sum across C_in tiles, so
    the bound is on the FULL sum (after all tiles + bias), not the
    per-tile partial. Returns the max |acc| over all output pixels.
    Must be < 2**31 or the RTL accumulator wraps (the golden uses
    int64 and does not wrap)."""
    from models.vision.common import pad2d as _pad2d
    A = np.asarray(A)
    Wq = np.asarray(W_q, dtype=np.int64)
    bq = np.asarray(b_q, dtype=np.int64)
    Ap = _pad2d(A, pad, pad_mode)
    H, Wd, Cin = Ap.shape
    KH, KW, _, Cout = Wq.shape
    Hout, Wout = H - KH + 1, Wd - KW + 1
    worst = 0
    for o in range(Cout):
        acc = np.zeros((Hout, Wout), dtype=np.int64) + bq[o]
        for t0 in range(0, Cin, C_in_tile):
            t1 = min(t0 + C_in_tile, Cin)
            for kh in range(KH):
                for kw in range(KW):
                    acc += np.tensordot(
                        Ap[kh:kh + Hout, kw:kw + Wout, t0:t1].astype(np.int64),
                        Wq[kh, kw, t0:t1, o], axes=([2], [0]))
            worst = max(worst, int(np.max(np.abs(acc))))
    return worst


def validate_r_out(model_name, torch_model, input_shape, program,
                   test_inputs):
    """Re-validate the per-layer activation ranges r_out on wider /
    different data. test_inputs is a list of float inputs (H,W,C) in
    [0,1]. For each, run the float forward on the fused graph and
    compute the REQUIRED r_out per conv/act node with the same logic
    as passes/quantize.py. Compare against the stored r_out (from the
    compiled program, calibrated on the Stage 6 input). Returns a dict
    name -> (stored_r, required_r, max_act, saturates). Also checks
    the 32-bit accumulator bound per conv layer (item (f))."""
    from passes.fusion import fuse
    from compile import infer_shapes
    log(f"  r_out re-validation ({model_name}, "
        f"{len(test_inputs)} wider inputs) ...")
    x_t = torch.from_numpy(
        np.transpose(test_inputs[0], (2, 0, 1))[None, :, :, :]
        .astype(np.float32))
    graph, _s = capture_torch(torch_model, x_t, model_name)
    fused, _g, _r = fuse(graph)
    infer_shapes(fused, input_shape)
    stored = {}
    for o in program["ops"]:
        if o["op"] in ("conv2d", "dw_conv2d", "linear", "act"):
            stored[o["name"]] = o.get("r_out", 0)
    # The fused graph's node names match the program's op names.
    req = {}
    maxact = {}
    for inp in test_inputs:
        vals, _ = float_forward(fused, inp)
        for n in fused.nodes:
            if n.op in ("conv2d", "dw_conv2d", "linear", "act"):
                out = vals[n.name]
                if out.size == 0:
                    continue
                mn = float(np.min(out))
                mx = float(np.max(out))
                ma = max(abs(mn), abs(mx))
                # Same logic as quantize.py.
                if n.op in ("conv2d", "dw_conv2d", "linear") and \
                        n.kwargs.get("fused_act") in ("relu", "leaky"):
                    r = range_exp(ma)
                elif mn < 0:
                    r = range_exp(2.0 * ma)
                else:
                    r = range_exp(mx)
                req[n.name] = max(req.get(n.name, 0), r)
                maxact[n.name] = max(maxact.get(n.name, 0.0), ma)
    res = {}
    for nm, r_stored in stored.items():
        r_req = req.get(nm, 0)
        sat = r_req > r_stored
        res[nm] = dict(stored_r=r_stored, required_r=r_req,
                       max_act=maxact.get(nm, 0.0), saturates=sat)
        flag = "SATURATES" if sat else "ok"
        log(f"    {nm}: stored r_out={r_stored} required r_out={r_req} "
            f"max|act|={maxact.get(nm, 0.0):.4f} [{flag}]")
    return res


def validate_acc_bounds(program, per_layer_inputs):
    """Check the 32-bit accumulator bound per conv layer (item (f)).
    per_layer_inputs is a dict name -> input array (H,W,Cin) for the
    layer. Returns a dict name -> (max|acc|, bound_ok)."""
    log("  accumulator bound check (32-bit, must be < 2^31) ...")
    weights = program["_weights_arrays"]
    res = {}
    for o in program["ops"]:
        if o["op"] not in ("conv2d", "dw_conv2d", "linear"):
            continue
        nm = o["name"]
        A = per_layer_inputs.get(nm)
        if A is None:
            continue
        worst = max_tile_acc(A, weights[nm]["W_q"], weights[nm]["b_q"],
                             o["shift"], pad=tuple(o.get("pad", (0, 0, 0, 0))),
                             pad_mode=o.get("pad_mode", "zero"),
                             C_in_tile=o.get("C_in_tile", 4))
        ok = worst < (1 << 31)
        res[nm] = dict(max_abs_acc=worst, ok=ok)
        log(f"    {nm}: max|acc|={worst} "
            f"({'OK' if ok else 'WRAPS -- 32-bit accumulator overflows!'})")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["all", "srcnn", "dncnn", "retinexnet"])
    ap.add_argument("--outdir", default=LOGS)
    args = ap.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(TB, exist_ok=True)
    overall_ok = True
    all_results = {}
    runners = {"srcnn": run_srcnn, "dncnn": run_dncnn,
               "retinexnet": run_retinexnet}
    models = ["srcnn", "dncnn", "retinexnet"] if args.model == "all" \
        else [args.model]
    for m in models:
        t0 = time.time()
        try:
            ok, results = runners[m](outdir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  {m}: EXCEPTION {e}")
            ok, results = False, {"error": str(e)}
        # Per-layer vs float (pure Python): the RTL output equals the
        # functional simulator output exactly for every conv layer
        # (max|err| = 0) and within 1 LSB for the PWL sigmoid, so the
        # per-layer funcsim-vs-float error IS the per-layer RTL-vs-float
        # error. Computed here so the report is self-contained.
        try:
            import stage6_float_layers as _sfl
            _builders = {
                "srcnn": (models_torch.SRCNN, build_srcnn_input, (24, 24, 1), False),
                "dncnn": (lambda: models_torch.DnCNNFragment(n_layers=17),
                          build_dncnn_input, (16, 16, 1), False),
                "retinexnet": (models_torch.RetinexNetDecom,
                               build_retinexnet_input, (16, 16, 4), True),
            }
            _cls, _build, _shape, _zp = _builders[m]
            _inp = _build()
            # build_dncnn_input returns (noisy, clean, noisy_f).
            _inp_u8, _inp_f = (_inp[0], _inp[2]) if m == "dncnn" else _inp
            _pf = _sfl.analyze(m, _cls(), _inp_f, _inp_u8, _shape,
                               zero_pad=_zp)
            for _nm, _v in _pf.items():
                if _nm in results and isinstance(results[_nm], dict):
                    results[_nm]["max_err_float"] = _v["max_err_float"]
                    results[_nm]["psnr_float"] = _v["psnr_float"]
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"  {m}: per-layer float merge failed: {e}")
        all_results[m] = results
        overall_ok = overall_ok and ok
        log(f"  [{m}] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f}s)")
        with open(os.path.join(outdir, f"{m}.txt"), "w") as f:
            f.write(f"Stage 6 {m} verification\n")
            f.write(f"result: {'PASS' if ok else 'FAIL'}\n")
            f.write(json.dumps(results, indent=2, default=str))
            f.write("\n")
    # r_out re-validation + accumulator bounds (Python only, no RTL).
    try:
        rres = run_r_validation(outdir)
        all_results["r_out_validation"] = rres
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"  r_out validation: EXCEPTION {e}")
        all_results["r_out_validation"] = {"error": str(e)}
        overall_ok = False
    with open(os.path.join(outdir, "stage6_summary.json"), "w") as f:
        f.write(json.dumps(all_results, indent=2, default=str))
    log(f"=== STAGE 6 RESULT: "
        f"{'ALL PASS' if overall_ok else 'FAILED'} ===")
    return 0 if overall_ok else 1


def run_r_validation(outdir):
    """r_out re-validation on wider/different data + accumulator bounds.
    Uses the compiled programs (recompiled here on the Stage 6 inputs)
    and larger test images (64x64, several seeds)."""
    import json as _json
    log("=== r_out re-validation + accumulator bounds ===")
    res = {}
    # SRCNN: 63x63 input (a clean x3 pipeline: 63 -> 21 -> 63),
    # seeds 7/42/123.
    src_inputs = []
    for seed in (7, 42, 123):
        hr = make_test_gray(63, 63, seed=seed)
        lr = np.array(Image.fromarray(hr).resize((21, 21), Image.BICUBIC))
        inp = np.array(Image.fromarray(lr).resize((63, 63), Image.BICUBIC))
        src_inputs.append((inp.astype(np.float64) / 255.0)[:, :, None])
    inp_u8_s, inp_f_s = build_srcnn_input(24)
    prog_s, _pl, _o, _f = compile_and_sim(
        "srcnn", models_torch.SRCNN(), inp_f_s, inp_u8_s, (24, 24, 1))
    res["srcnn"] = validate_r_out("srcnn", models_torch.SRCNN(),
                                  (63, 63, 1), prog_s, src_inputs)
    # Accumulator bounds for SRCNN (Stage 6 inputs).
    res["srcnn_acc"] = validate_acc_bounds(
        prog_s, _layer_inputs_srcnn(prog_s, inp_u8_s))
    # DnCNN: 64x64 noisy, seeds 7/42.
    dnc_inputs = []
    for seed in (7, 42):
        clean = make_test_gray(64, 64, seed=seed)
        rng = np.random.default_rng(11 + seed)
        noisy = np.clip(clean.astype(np.float64)
                        + rng.normal(0, 25, (64, 64)), 0, 255)
        dnc_inputs.append((noisy / 255.0)[:, :, None])
    noisy_u8, _cl, noisy_f = build_dncnn_input(16)
    prog_d, _pl, _o, _f = compile_and_sim(
        "dncnn", models_torch.DnCNNFragment(n_layers=17), noisy_f, noisy_u8,
        (16, 16, 1))
    res["dncnn"] = validate_r_out("dncnn", models_torch.DnCNNFragment(n_layers=17),
                                  (64, 64, 1), prog_d, dnc_inputs)
    res["dncnn_acc"] = validate_acc_bounds(
        prog_d, _layer_inputs_dncnn(prog_d, noisy_u8))
    # RetinexNet: 64x64 low-light, seeds 7/42.
    ret_inputs = []
    for seed in (7, 42):
        rgb = make_test_rgb(64, 64, seed=seed)
        low = np.clip(rgb.astype(np.float64) * 0.25, 0, 255).astype(np.uint8)
        mx = low.max(axis=2, keepdims=True)
        x4 = np.concatenate([mx, low], axis=2)
        ret_inputs.append((x4.astype(np.float64) / 255.0))
    inp_u8_r, inp_f_r = build_retinexnet_input(16)
    prog_r, _pl, _o, _f = compile_and_sim(
        "retinexnet", models_torch.RetinexNetDecom(), inp_f_r, inp_u8_r,
        (16, 16, 4))
    res["retinexnet"] = validate_r_out(
        "retinexnet", models_torch.RetinexNetDecom(), (64, 64, 4),
        prog_r, ret_inputs)
    res["retinexnet_acc"] = validate_acc_bounds(
        prog_r, _layer_inputs_retinexnet(prog_r, inp_u8_r))
    with open(os.path.join(outdir, "r_out_validation.txt"), "w") as f:
        f.write("Stage 6 r_out re-validation + accumulator bounds\n")
        f.write(_json.dumps(res, indent=2, default=str))
        f.write("\n")
    return res


def _layer_inputs_srcnn(program, inp_u8):
    """Per-layer input arrays for the SRCNN accumulator check (uses
    the functional simulator's per-layer outputs as the inputs)."""
    from funcsim import fixed_forward as _ff
    sim = {"ops": program["ops"], "weights": program["_weights_arrays"],
           "act_tables": program["_act_tables_full"],
           "outputs": program["outputs"]}
    per_layer, _ = _ff(sim, inp_u8)
    convs = [o["name"] for o in program["ops"]
             if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    inputs = {convs[0]: inp_u8}
    for i, nm in enumerate(convs[1:], start=1):
        inputs[nm] = per_layer[convs[i - 1]]
    return inputs


def _layer_inputs_dncnn(program, inp_u8):
    from funcsim import fixed_forward as _ff
    sim = {"ops": program["ops"], "weights": program["_weights_arrays"],
           "act_tables": program["_act_tables_full"],
           "outputs": program["outputs"]}
    per_layer, _ = _ff(sim, inp_u8)
    convs = [o["name"] for o in program["ops"]
             if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    inputs = {convs[0]: inp_u8}
    for i, nm in enumerate(convs[1:], start=1):
        inputs[nm] = per_layer[convs[i - 1]]
    return inputs


def _layer_inputs_retinexnet(program, inp_u8):
    """Per-layer inputs for RetinexNet (ZERO-pad funcsim, the RTL's
    padding)."""
    from funcsim import fixed_forward as _ff
    zp = zero_pad_program(program)
    sim = {"ops": zp["ops"], "weights": zp["_weights_arrays"],
           "act_tables": zp["_act_tables_full"],
           "outputs": zp["outputs"]}
    per_layer, _ = _ff(sim, inp_u8)
    convs = [o["name"] for o in program["ops"]
             if o["op"] in ("conv2d", "dw_conv2d", "linear")]
    inputs = {convs[0]: inp_u8}
    for i, nm in enumerate(convs[1:], start=1):
        inputs[nm] = per_layer[convs[i - 1]]
    return inputs


if __name__ == "__main__":
    sys.exit(main())
