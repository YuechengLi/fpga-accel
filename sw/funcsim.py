#!/usr/bin/env python3
"""Stage 5: functional simulator + float forward.

The functional simulator executes the compiler-emitted program in
fixed point and compares output + per-layer intermediates against
the fixed-point golden (models/common.py) and PyTorch float. This is
the compiler's software verification: a functional-simulator-vs-
golden mismatch is a COMPILER bug; an RTL-vs-functional-simulator
mismatch (Stage 6) is a HARDWARE bug. The simulator runs BEFORE any
RTL sim (plan Stage 5 verification flow).

Two execution modes:
  * ``float_forward`` -- interpret the fused IR graph in float
    (numpy, models/common.py conv2d_float). Used by the quantization
    pass to calibrate per-layer activation ranges, and as the float
    reference for the demo.
  * ``fixed_forward`` -- execute the emitted program (the compiler's
    op list) in fixed point, TILED the way the hardware does it:
    for each conv, loop over output channels o and C_in tiles t,
    accumulate the partial sums into a 32-bit (here int64) accumulator
    in tile order, then apply the shift + saturate + fused act once.
    The shift is ROUND-HALF-UP (matching the RTL); the Stage 6 golden
    (models/common.py conv2d_fixed) uses FLOOR shift, so the two
    differ by <= 1 LSB -- expected, not a bug, and the acceptance
    test allows max|err| <= 1 vs the floor-shift golden.

The ``act`` op (complex activation) is executed with the COMPILER's
generated table (MLUT or PWL), so the simulator validates the
compiler's table generation, not just the function.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import conv2d_float, conv2d_fixed, pad2d, psnr, max_abs_err
from passes.actapprox import FUNCS as ACT_FUNCS, pwl_eval
from passes.mlut import mlut_eval


# ---------------------------------------------------------------------------
# Float forward (IR interpreter, float).
# ---------------------------------------------------------------------------
def float_forward(graph, x):
    """Run the fused IR graph in float. ``x`` is (H, W, C) float.
    Returns (outputs dict name->array, final output)."""
    vals = {}
    for n in graph.nodes:
        if n.op == "input":
            vals[n.name] = np.asarray(x, dtype=np.float64)
        elif n.op in ("conv2d", "dw_conv2d", "linear"):
            a = vals[n.args[0]]
            W = n.params["weight"]
            b = n.params.get("bias", np.zeros(W.shape[3]))
            pad = n.kwargs.get("pad", (0, 0, 0, 0))
            pad_mode = n.kwargs.get("pad_mode", "zero")
            out = conv2d_float(a, W, b, pad=pad, pad_mode=pad_mode)
            fa = n.kwargs.get("fused_act")
            if fa == "relu":
                out = np.maximum(out, 0.0)
            elif fa == "leaky":
                s = n.kwargs.get("act_slope", 0.01)
                out = np.where(out >= 0, out, out * s)
            vals[n.name] = out
        elif n.op == "relu":
            vals[n.name] = np.maximum(vals[n.args[0]], 0.0)
        elif n.op == "leaky_relu":
            a = vals[n.args[0]]
            s = n.kwargs.get("negative_slope", 0.01)
            vals[n.name] = np.where(a >= 0, a, a * s)
        elif n.op in ("sigmoid", "tanh", "gelu", "silu"):
            vals[n.name] = ACT_FUNCS[n.op](vals[n.args[0]])
        elif n.op == "act":
            vals[n.name] = ACT_FUNCS[n.kwargs["func"]](vals[n.args[0]])
        elif n.op == "residual":
            a, b = vals[n.args[0]], vals[n.args[1]]
            vals[n.name] = a + b if n.kwargs.get("mode", "add") == "add" else a - b
        elif n.op in ("add", "sub"):
            a, b = vals[n.args[0]], vals[n.args[1]]
            vals[n.name] = a + b if n.op == "add" else a - b
        elif n.op in ("mul", "simplegate"):
            vals[n.name] = vals[n.args[0]] * vals[n.args[1]]
        elif n.op == "max_pool2d":
            vals[n.name] = _pool(vals[n.args[0]], n.kwargs, "max")
        elif n.op == "avg_pool2d":
            vals[n.name] = _pool(vals[n.args[0]], n.kwargs, "avg")
        elif n.op == "gap":
            a = vals[n.args[0]]
            vals[n.name] = a.mean(axis=(0, 1), keepdims=True)
        elif n.op == "pad":
            a = vals[n.args[0]]
            vals[n.name] = pad2d(a, n.kwargs["pad"],
                                 mode=n.kwargs.get("mode", "zero"))
        elif n.op == "upsample":
            vals[n.name] = _upsample(vals[n.args[0]], n.kwargs)
        elif n.op == "concat":
            vals[n.name] = np.concatenate(
                [vals[a] for a in n.args], axis=2)
        elif n.op == "layer_norm":
            vals[n.name] = _layernorm(vals[n.args[0]])
        elif n.op == "bypass":
            vals[n.name] = vals[n.args[0]]
        else:
            raise RuntimeError(f"float_forward: unsupported op {n.op} "
                               f"(node {n.name})")
    out = vals[graph.outputs[0]] if graph.outputs else None
    return vals, out


def _pool(a, kwargs, mode):
    kh, kw = kwargs.get("K", (2, 2))
    sh, sw = kwargs.get("stride", (kh, kw))
    H, W, C = a.shape
    ho = (H - kh) // sh + 1
    wo = (W - kw) // sw + 1
    out = np.zeros((ho, wo, C))
    for i in range(ho):
        for j in range(wo):
            blk = a[i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            out[i, j, :] = blk.max(axis=(0, 1)) if mode == "max" \
                else blk.mean(axis=(0, 1))
    return out


def _upsample(a, kwargs):
    scale = kwargs.get("scale")
    if scale is None:
        scale = 2
    if isinstance(scale, (list, tuple)):
        sh, sw = int(scale[0]), int(scale[1])
    else:
        sh = sw = int(scale)
    if kwargs.get("mode", "nearest") == "nearest":
        return np.repeat(np.repeat(a, sh, axis=0), sw, axis=1)
    # bilinear (align_corners=False, the torch default)
    H, W, C = a.shape
    ho, wo = H * sh, W * sw
    ys = (np.arange(ho) + 0.5) / sh - 0.5
    xs = (np.arange(wo) + 0.5) / sw - 0.5
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _layernorm(a):
    mu = a.mean(axis=2, keepdims=True)
    var = a.var(axis=2, keepdims=True)
    return (a - mu) / np.sqrt(var + 1e-5)


# ---------------------------------------------------------------------------
# Fixed-point tiled conv (mirrors the RTL accumulation order).
# ---------------------------------------------------------------------------
def _shift_rhu(acc, shift):
    """Arithmetic right shift with ROUND-HALF-UP (the RTL's shift).
    acc is int64; shift in [0,15]."""
    if shift == 0:
        return acc
    return (acc + (1 << (shift - 1))) >> shift


def conv2d_tiled(A, W_q, b_q, shift, pad=(0, 0, 0, 0), pad_mode="zero",
                 signed_out=False, fused_act=None, act_slope=0.01,
                 C_in_tile=4, shift_mode="rhu"):
    """Fixed-point conv, tiled the way conv2d_mc does it.

    A: (H,W,Cin) uint8 (or int8 if signed in). W_q: (KH,KW,Cin,Cout)
    int16. b_q: (Cout,) int32. Accumulates per (o, t) pass into an
    int64 accumulator (the RTL uses a 32-bit accumulator per tile;
    the golden uses int64 -- the tiled order is what is checked
    here, not the accumulator width). After all C_in tiles, applies
    the shift (round-half-up by default, matching the RTL) +
    saturate + fused act.
    """
    A = np.asarray(A)
    Wq = np.asarray(W_q, dtype=np.int64)
    bq = np.asarray(b_q, dtype=np.int64)
    Ap = pad2d(A, pad, pad_mode)
    H, Wd, Cin = Ap.shape
    KH, KW, _, Cout = Wq.shape
    Hout, Wout = H - KH + 1, Wd - KW + 1
    out = np.zeros((Hout, Wout, Cout), dtype=np.int64)
    for o in range(Cout):
        acc = np.zeros((Hout, Wout), dtype=np.int64) + bq[o]
        for t0 in range(0, Cin, C_in_tile):
            t1 = min(t0 + C_in_tile, Cin)
            for kh in range(KH):
                for kw in range(KW):
                    acc += np.tensordot(
                        Ap[kh:kh + Hout, kw:kw + Wout, t0:t1].astype(np.int64),
                        Wq[kh, kw, t0:t1, o], axes=([2], [0]))
        if shift_mode == "rhu":
            v = _shift_rhu(acc, shift)
        else:  # floor (the models/common.py golden)
            v = np.floor_divide(acc, 2 ** shift)
        if fused_act == "relu":
            v = np.maximum(v, 0)
        elif fused_act == "leaky":
            # Leaky: x if x>=0 else x*slope (slope Q0.16)
            s = int(round(act_slope * 65536.0))
            v = np.where(v >= 0, v, (v * s) >> 16)
        if signed_out:
            out[:, :, o] = np.clip(v, -128, 127)
        else:
            out[:, :, o] = np.clip(v, 0, 255)
    return out.astype(np.int64 if signed_out else np.uint8)


def conv2d_fixed_rhu(A, W_q, b_q, shift, pad=(0, 0, 0, 0),
                     pad_mode="zero", signed_out=False):
    """conv2d_fixed (models/common.py) but with round-half-up shift
    (the RTL's shift) instead of floor shift."""
    return conv2d_tiled(A, W_q, b_q, shift, pad=pad, pad_mode=pad_mode,
                        signed_out=signed_out,
                        C_in_tile=max(1, np.asarray(A).shape[2]),
                        shift_mode="rhu")


# ---------------------------------------------------------------------------
# Program executor (the functional simulator).
# ---------------------------------------------------------------------------
def fixed_forward(program, x):
    """Execute an emitted program (see emit.py) in fixed point.

    ``program`` is the dict from emit.emit_program (ops + weights +
    act tables). ``x`` is the input image as uint8 (H, W, C).
    Returns (per_layer dict name->array, final output).
    """
    ops = program["ops"]
    # Weights: the emitted program (emit.py) stores a JSON-safe
    # ``weights`` entry (W_q_shape, no arrays -- the arrays live in
    # weights.bin). compile_model attaches the runtime arrays as
    # ``_weights_arrays``. Prefer the runtime arrays; fall back to
    # ``weights`` (which has the arrays when the caller built the
    # program by hand, as the demo's sim_program does).
    weights = program.get("_weights_arrays") or program.get("weights", {})
    # Act tables: same split as weights -- the emitted ``act_tables``
    # is a JSON-safe summary; the runtime tables live in
    # ``_act_tables_full`` (attached by compile_model).
    acts = (program.get("_act_tables_full")
            or program.get("act_tables", {}))
    vals = {}
    per_layer = {}
    for op in ops:
        name = op["name"]
        kind = op["op"]
        if kind == "input":
            vals[name] = np.asarray(x)
            per_layer[name] = vals[name]
        elif kind in ("conv2d", "dw_conv2d", "linear"):
            a = vals[op["args"][0]]
            w = weights[name]
            out = conv2d_tiled(
                a, w["W_q"], w["b_q"], op["shift"],
                pad=tuple(op.get("pad", (0, 0, 0, 0))),
                pad_mode=op.get("pad_mode", "zero"),
                signed_out=op.get("signed_out", False),
                fused_act=op.get("fused_act"),
                act_slope=op.get("act_slope", 0.01),
                C_in_tile=op.get("C_in_tile", 4),
                shift_mode="rhu")
            vals[name] = out
            per_layer[name] = out
        elif kind == "act":
            a = vals[op["args"][0]]
            vals[name] = _apply_act(a, op, acts.get(name, {}))
            per_layer[name] = vals[name]
        elif kind == "residual":
            a, b = vals[op["args"][0]], vals[op["args"][1]]
            v = (a.astype(np.int64) + b.astype(np.int64)
                 if op.get("mode", "add") == "add"
                 else a.astype(np.int64) - b.astype(np.int64))
            vals[name] = np.clip(v, 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
        elif kind == "max_pool2d":
            vals[name] = _pool(vals[op["args"][0]], op, "max").astype(np.uint8)
            per_layer[name] = vals[name]
        elif kind == "avg_pool2d":
            vals[name] = np.round(
                _pool(vals[op["args"][0]].astype(np.float64), op, "avg")
            ).astype(np.uint8)
            per_layer[name] = vals[name]
        elif kind == "gap":
            a = vals[op["args"][0]].astype(np.float64)
            vals[name] = np.round(a.mean(axis=(0, 1), keepdims=True)).astype(np.uint8)
            per_layer[name] = vals[name]
        elif kind == "upsample":
            vals[name] = _upsample(vals[op["args"][0]], op).astype(np.uint8)
            per_layer[name] = vals[name]
        elif kind == "concat":
            vals[name] = np.concatenate(
                [vals[a] for a in op["args"]], axis=2)
            per_layer[name] = vals[name]
        elif kind == "layer_norm":
            vals[name] = _layernorm_fixed(vals[op["args"][0]])
            per_layer[name] = vals[name]
        elif kind == "bypass":
            vals[name] = vals[op["args"][0]]
            per_layer[name] = vals[name]
        else:
            raise RuntimeError(f"fixed_forward: unsupported op {kind} "
                               f"(node {name})")
    out = vals[program["outputs"][0]] if program.get("outputs") else None
    return per_layer, out


def _apply_act(a, op, table):
    """Apply a compiler-generated act table (MLUT or PWL) to a
    fixed-point array. The act input is the conv output: for a
    signed intermediate it is int8 in [-128,127] (representing
    float * 255/2**r); the act module takes a 12-bit offset-binary
    (MLUT) or Q5.11 (PWL) value in [-8, 8). The conversion is:
    x_float = a / (255 / 2**r_in) for unsigned, or a / (255/2**r_in)
    for signed (a is already the signed value). The act output is
    Q5.11 (MLUT) or Q1.15 (PWL), converted back to the layer's
    output scale (uint8 = float * 255/2**r_out).
    """
    impl = table.get("impl", "mlut")
    r_in = op.get("r_in", 0)
    r_out = op.get("r_out", 0)
    scale_in = 255.0 / (2.0 ** r_in)
    x = np.asarray(a, dtype=np.float64) / scale_in
    if impl == "mlut":
        H1 = table["H1"]
        H2 = table["H2"]
        base = table["base"]
        sel = table["sel"]
        fine = table["fine"]
        # Vectorized MLUT eval over the array.
        in12 = np.clip(np.round((x + 8.0) * 256.0), 0, 4095).astype(int)
        coarse = in12 >> (12 - H1)
        fine_idx = (in12 >> (12 - H1 - H2)) & ((1 << H2) - 1)
        b = np.array(base, dtype=np.int64)[coarse]
        t = np.array(sel, dtype=int)[coarse]
        fine_arr = np.array(fine, dtype=np.int64)
        d = fine_arr[t, fine_idx]
        s = np.clip(b + d, -32768, 32767)
        y = s / 2048.0
    else:  # pwl
        y = pwl_eval(op.get("func", "sigmoid"), x,
                     table["y0"], table["dslope"])
    scale_out = 255.0 / (2.0 ** r_out)
    v = np.round(y * scale_out)
    if op.get("signed_out"):
        # Signed intermediate: signed int8, same scale 255/2**r_out
        # (the quantize pass chose r_out so |v| <= 127).
        return np.clip(v, -128, 127).astype(np.int64)
    return np.clip(v, 0, 255).astype(np.uint8)


def _layernorm_fixed(a):
    """Fixed-point LayerNorm (Stage 4 layernorm.sv behaviour, in
    float for the simulator; the exact fixed-point error is a Stage 4
    result, max 1.36 LSB)."""
    a = np.asarray(a, dtype=np.float64)
    mu = a.mean(axis=2, keepdims=True)
    var = a.var(axis=2, keepdims=True)
    y = (a - mu) / np.sqrt(var + 1e-5)
    return np.clip(np.round(y * 32.0 + 128.0), 0, 255).astype(np.uint8)
