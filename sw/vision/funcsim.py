#!/usr/bin/env python3
"""Stage 5: functional simulator + float forward.

The functional simulator executes the compiler-emitted program in
fixed point and compares output + per-layer intermediates against
the fixed-point golden (models/vision/common.py) and PyTorch float. This is
the compiler's software verification: a functional-simulator-vs-
golden mismatch is a COMPILER bug; an RTL-vs-functional-simulator
mismatch (Stage 6) is a HARDWARE bug. The simulator runs BEFORE any
RTL sim (plan Stage 5 verification flow).

Two execution modes:
  * ``float_forward`` -- interpret the fused IR graph in float
    (numpy, models/vision/common.py conv2d_float). Used by the quantization
    pass to calibrate per-layer activation ranges, and as the float
    reference for the demo.
  * ``fixed_forward`` -- execute the emitted program (the compiler's
    op list) in fixed point, TILED the way the hardware does it:
    for each conv, loop over output channels o and C_in tiles t,
    accumulate the partial sums into a 32-bit (here int64) accumulator
    in tile order, then apply the shift + saturate + fused act once.
    The shift is ROUND-HALF-UP (matching the RTL); the Stage 6 golden
    (models/vision/common.py conv2d_fixed) uses FLOOR shift, so the two
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from models.vision.common import conv2d_float, conv2d_fixed, pad2d, psnr, max_abs_err
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
            # Stride is a decimation (the accelerator's convention):
            # the conv runs at stride 1 and the output is subsampled.
            # float_forward must honor it (the 7c models are all
            # stride 1, so this never fired before Stage 7d).
            sx, sy = n.kwargs.get("stride", (1, 1))
            if sx != 1 or sy != 1:
                out = out[::sy, ::sx, :]
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
        elif n.op == "gmax":
            # Global max pool (ChannelAttention max branch, DNANet).
            a = vals[n.args[0]]
            vals[n.name] = a.max(axis=(0, 1), keepdims=True)
        elif n.op == "cmean":
            # Channel mean (SpatialAttention, DNANet): (H,W,C)->(H,W,1).
            a = vals[n.args[0]]
            vals[n.name] = a.mean(axis=2, keepdims=True)
        elif n.op == "cmax":
            # Channel max (SpatialAttention, DNANet): (H,W,C)->(H,W,1).
            a = vals[n.args[0]]
            vals[n.name] = a.max(axis=2, keepdims=True)
        elif n.op == "pad":
            a = vals[n.args[0]]
            vals[n.name] = pad2d(a, n.kwargs["pad"],
                                 mode=n.kwargs.get("mode", "zero"))
        elif n.op == "upsample":
            vals[n.name] = _upsample(vals[n.args[0]], n.kwargs)
        elif n.op == "downsample":
            # Bilinear 0.5x, align_corners=True (DNANet `down`). The
            # RTL has no downsample, so the driver executes this node
            # on the CPU with the same helper (see fixed_forward).
            vals[n.name] = _downsample_ac(vals[n.args[0]])
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


def _downsample_ac(a):
    """Bilinear 0.5x downsample, align_corners=True (torch's
    nn.Upsample(scale_factor=0.5, mode='bilinear',
    align_corners=True), as used by DNANet's `down`). Output size
    floor(H/2) x floor(W/2); the source index for output i is
    i * (H_in - 1) / (H_out - 1) (corner-aligned). Used by both
    float_forward and fixed_forward (the latter dequantizes,
    applies this, and re-quantizes) and by the 7d driver's CPU
    execution of the `downsample` node, so all three agree."""
    a = np.asarray(a, dtype=np.float64)
    H, W, C = a.shape
    ho, wo = H // 2, W // 2
    if ho < 1 or wo < 1:
        raise RuntimeError(f"_downsample_ac: input {H}x{W} too small")
    ys = np.arange(ho) * (H - 1) / max(ho - 1, 1)
    xs = np.arange(wo) * (W - 1) / max(wo - 1, 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _pool(a, kwargs, mode):
    kh, kw = kwargs.get("K", (2, 2))
    sh, sw = kwargs.get("stride", (kh, kw))
    # Optional zero padding (the YOLO SPPF 5x5 stride-1 pool pads by 2
    # to keep the size; the 2x2 pools have no pad). Added in Stage 7d;
    # the default (no pad) is unchanged.
    pad = kwargs.get("pad", (0, 0, 0, 0))
    if any(pad):
        a = pad2d(a, tuple(pad), "zero")
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
    if kwargs.get("mode") == "bilinear_rtl":
        # The upsample RTL's bilinear (Stage 4 golden_upsample.py):
        # out[2r,2c]=in[r,c], out[2r,2c+1]=(in[r,c]+in[r,c+1])/2,
        # out[2r+1,2c]=(in[r,c]+in[r+1,c])/2,
        # out[2r+1,2c+1]=(in[r,c]+in[r,c+1]+in[r+1,c]+in[r+1,c+1])/4,
        # edges replicate. For fixed-point (int) inputs the
        # divisions are round-half-up, matching the RTL; for float
        # inputs (float_forward calibration) they are exact halves.
        # This matches NEITHER torch convention (align_corners
        # True/False); it is the hardware's formula, so the 7d
        # golden uses it (the torch float reference keeps torch's
        # convention and the difference is reported separately).
        # Scale 2 only.
        assert sh == 2 and sw == 2, "bilinear_rtl is scale-2 only"
        a = np.asarray(a)
        is_int = np.issubdtype(a.dtype, np.integer)
        a = a.astype(np.int64) if is_int else a.astype(np.float64)
        H, W, C = a.shape
        out = np.zeros((2 * H, 2 * W, C), dtype=a.dtype)
        out[0::2, 0::2, :] = a
        ar = np.concatenate([a[:, 1:, :], a[:, -1:, :]], axis=1)
        ab = np.concatenate([a[1:, :, :], a[-1:, :, :]], axis=0)
        abr = np.concatenate([ab[:, 1:, :], ab[:, -1:, :]], axis=1)
        if is_int:
            out[0::2, 1::2, :] = (a + ar + 1) >> 1
            out[1::2, 0::2, :] = (a + ab + 1) >> 1
            out[1::2, 1::2, :] = (a + ar + ab + abr + 2) >> 2
        else:
            out[0::2, 1::2, :] = (a + ar) / 2.0
            out[1::2, 0::2, :] = (a + ab) / 2.0
            out[1::2, 1::2, :] = (a + ar + ab + abr) / 4.0
        return out
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
        else:  # floor (the models/vision/common.py golden)
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
    """conv2d_fixed (models/vision/common.py) but with round-half-up shift
    (the RTL's shift) instead of floor shift."""
    return conv2d_tiled(A, W_q, b_q, shift, pad=pad, pad_mode=pad_mode,
                        signed_out=signed_out,
                        C_in_tile=max(1, np.asarray(A).shape[2]),
                        shift_mode="rhu")


# ---------------------------------------------------------------------------
# Program executor (the functional simulator).
# ---------------------------------------------------------------------------
def rescale_bytes(a, r_from, r_to, signed=False):
    """Rescale a fixed-point array from scale 255/2**r_from to
    255/2**r_to. ``a`` is an int64/uint8 array; the returned array
    is int64 (signed) or uint8 (unsigned), clipped to the target
    range. Left shift (r_from > r_to) multiplies and may clip;
    right shift (r_from < r_to) uses round-half-up, matching the
    RTL shift convention. Used by the 7d capture/driver when a
    consumer's r_in differs from its producer's r_out (DNANet
    branching); a no-op when r_from == r_to."""
    a = np.asarray(a, dtype=np.int64)
    if r_from == r_to:
        v = a
    elif r_from > r_to:
        v = a << (r_from - r_to)
    else:
        v = (a + (1 << (r_to - r_from - 1))) >> (r_to - r_from)
    if signed:
        return np.clip(v, -128, 127).astype(np.int64)
    return np.clip(v, 0, 255).astype(np.uint8)


def rescale_input(a, groups, r_to):
    """Rescale a (H,W,C) fixed-point array to scale r_to, per
    channel group. ``groups`` is a list of (c0, c1, r_from, signed)
    covering the channels (as produced by a concat or a single
    tensor). Returns an array in the target scale (uint8 unless
    any group is signed, in which case int64)."""
    a = np.asarray(a, dtype=np.int64)
    out = np.zeros_like(a)
    any_signed = False
    for c0, c1, r_from, signed in groups:
        out[:, :, c0:c1] = rescale_bytes(a[:, :, c0:c1], r_from, r_to,
                                        signed=signed)
        any_signed = any_signed or signed
    if any_signed:
        return out.astype(np.int64)
    return out.astype(np.uint8)


def fixed_forward(program, x):
    """Execute an emitted program (see emit.py) in fixed point.

    ``program`` is the dict from emit.emit_program (ops + weights +
    act tables). ``x`` is the input image as uint8 (H, W, C).
    Returns (per_layer dict name->array, final output).

    Scale tracking (Stage 7d): ``groups`` records each node's
    output scale per channel group, [(c0, c1, r, signed), ...].
    A consumer whose ``r_in`` differs from a producer's group
    scale rescales that group (see rescale_input) before the op
    runs -- this is how DNANet's branching (a concat feeding a
    conv, or a residual add of two differently-scaled tensors)
    is handled. Programs without ``r_in``/``r_out`` on their ops
    (the 7c programs) skip the rescale entirely, so this is
    backward compatible.
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
    groups = {}

    def _rescaled(op, arg):
        """The arg tensor, rescaled to the consumer op's r_in.
        No-op when the op carries no r_in or the scales match."""
        a = vals[arg]
        r_to = op.get("r_in")
        g = groups.get(arg)
        if r_to is None or g is None:
            return a
        if len(g) == 1 and g[0][2] == r_to:
            return a
        return rescale_input(a, g, int(r_to))

    def _set_groups(name, r_out, signed, C):
        groups[name] = [(0, int(C), int(r_out), bool(signed))]

    for op in ops:
        name = op["name"]
        kind = op["op"]
        if kind == "input":
            vals[name] = np.asarray(x)
            per_layer[name] = vals[name]
            _set_groups(name, 0, False, vals[name].shape[2])
        elif kind in ("conv2d", "dw_conv2d", "linear"):
            a = _rescaled(op, op["args"][0])
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
            # Stride is implemented as DECIMATE (the accelerator's
            # convention, and the registration's): the PE runs the
            # conv at stride 1 on the pre-padded input and the
            # driver takes every s-th output. YOLO's 7 stride-2
            # convs need this; the 7c models are all stride 1.
            sx, sy = op.get("stride", (1, 1))
            if sx != 1 or sy != 1:
                out = out[::sy, ::sx, :]
            vals[name] = out
            per_layer[name] = out
            _set_groups(name, op.get("r_out", 0),
                        op.get("signed_out", False), out.shape[2])
        elif kind == "act":
            a = _rescaled(op, op["args"][0])
            vals[name] = _apply_act(a, op, acts.get(name, {}))
            per_layer[name] = vals[name]
            _set_groups(name, op.get("r_out", 0),
                        op.get("signed_out", False), vals[name].shape[2])
        elif kind == "relu":
            # Standalone ReLU (fixed_forward had no relu case before
            # Stage 7d; DNANet's post-CBAM ReLU and the C2f residual
            # path need it). Input may be signed int64.
            a = _rescaled(op, op["args"][0])
            vals[name] = np.maximum(a.astype(np.int64), 0).astype(np.uint8)
            per_layer[name] = vals[name]
            _set_groups(name, op.get("r_out", 0), False,
                        vals[name].shape[2])
        elif kind == "simplegate":
            # Elementwise multiply (Stage 4 simplegate.sv): the
            # exact integer arithmetic of the RTL -- (A*B)>>shift
            # with a Python floor shift, saturating to the output
            # range. sign_in reinterprets uint8 bytes >= 128 as
            # negative (a no-op when the operands already carry
            # signed int64 values). Used by the SiLU decomposition
            # (Stage 7d) and by DNANet attention. The shift absorbs
            # the two input scales, so the inputs are NOT rescaled
            # here; the capture sets the shift from the measured
            # scales.
            A = vals[op["args"][0]].astype(np.int64)
            B = vals[op["args"][1]].astype(np.int64)
            shift = int(op.get("shift", 8))
            sign_in = bool(op.get("sign_in", False))
            sign_out = bool(op.get("sign_out", False))
            if sign_in:
                A = np.where(A >= 128, A - 256, A)
                B = np.where(B >= 128, B - 256, B)
            v = (A * B) >> shift
            if sign_out:
                vals[name] = np.clip(v, -128, 127).astype(np.int64)
            else:
                vals[name] = np.clip(v, 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
            _set_groups(name, op.get("r_out", 0), sign_out,
                        vals[name].shape[2])
        elif kind == "residual":
            a = _rescaled(op, op["args"][0])
            b = _rescaled(op, op["args"][1])
            v = (a.astype(np.int64) + b.astype(np.int64)
                 if op.get("mode", "add") == "add"
                 else a.astype(np.int64) - b.astype(np.int64))
            if op.get("signed_out"):
                vals[name] = np.clip(v, -128, 127).astype(np.int64)
            else:
                vals[name] = np.clip(v, 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
            _set_groups(name, op.get("r_out", op.get("r_in", 0)),
                        op.get("signed_out", False), vals[name].shape[2])
        elif kind == "max_pool2d":
            vals[name] = _pool(vals[op["args"][0]], op, "max").astype(np.uint8)
            per_layer[name] = vals[name]
            # uint8 output: force unsigned even if the producer is
            # signed (else a consumer's rescale would clip 128..255
            # to 127, and a checkpoint resume would reinterpret
            # them as negative).
            groups[name] = [(c0, c1, r, False) for (c0, c1, r, _s)
                            in groups[op["args"][0]]]
        elif kind == "avg_pool2d":
            vals[name] = np.round(
                _pool(vals[op["args"][0]].astype(np.float64), op, "avg")
            ).astype(np.uint8)
            per_layer[name] = vals[name]
            groups[name] = [(c0, c1, r, False) for (c0, c1, r, _s)
                            in groups[op["args"][0]]]
        elif kind == "gap":
            a = vals[op["args"][0]].astype(np.float64)
            vals[name] = np.round(a.mean(axis=(0, 1), keepdims=True)).astype(np.uint8)
            per_layer[name] = vals[name]
            groups[name] = [(c0, c1, r, False) for (c0, c1, r, _s)
                            in groups[op["args"][0]]]
        elif kind == "gmax":
            # Global max pool (ChannelAttention max branch, DNANet).
            a = vals[op["args"][0]]
            vals[name] = np.clip(
                a.max(axis=(0, 1), keepdims=True), 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
            groups[name] = [(c0, c1, r, False) for (c0, c1, r, _s)
                            in groups[op["args"][0]]]
        elif kind == "cmean":
            # Channel mean (SpatialAttention, DNANet): (H,W,C)->(H,W,1).
            a = vals[op["args"][0]].astype(np.float64)
            vals[name] = np.clip(
                np.round(a.mean(axis=2, keepdims=True)), 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
            g = groups[op["args"][0]]
            _set_groups(name, g[0][2] if g else 0, False, 1)
        elif kind == "cmax":
            # Channel max (SpatialAttention, DNANet): (H,W,C)->(H,W,1).
            a = vals[op["args"][0]]
            vals[name] = np.clip(
                a.max(axis=2, keepdims=True), 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
            g = groups[op["args"][0]]
            _set_groups(name, g[0][2] if g else 0, False, 1)
        elif kind == "upsample":
            vals[name] = _upsample(vals[op["args"][0]], op).astype(np.uint8)
            per_layer[name] = vals[name]
            # uint8 output: force unsigned (see max_pool2d).
            groups[name] = [(c0, c1, r, False) for (c0, c1, r, _s)
                            in groups[op["args"][0]]]
        elif kind == "downsample":
            # Bilinear 0.5x, align_corners=True (DNANet `down`).
            # No RTL: the driver executes this node on the CPU with
            # the same computation (dequantize -> float bilinear ->
            # re-quantize to the node's r_out), so the golden and
            # the driver agree by construction.
            a = vals[op["args"][0]]
            g = groups[op["args"][0]]
            # Dequantize to float. If the input has mixed per-channel
            # scales (a concat), rescale to one common scale first.
            if len(g) == 1:
                r_in = g[0][2]
                af = a.astype(np.float64) / (255.0 / (2.0 ** r_in))
            else:
                r_in = max(gr[2] for gr in g)
                au = rescale_input(a, g, r_in)
                af = au.astype(np.float64) / (255.0 / (2.0 ** r_in))
            f = _downsample_ac(af)
            r_out = int(op.get("r_out", 0))
            v = np.round(f * (255.0 / (2.0 ** r_out)))
            if op.get("signed_out"):
                vals[name] = np.clip(v, -128, 127).astype(np.int64)
            else:
                vals[name] = np.clip(v, 0, 255).astype(np.uint8)
            per_layer[name] = vals[name]
            _set_groups(name, r_out, op.get("signed_out", False),
                        vals[name].shape[2])
        elif kind == "concat":
            vals[name] = np.concatenate(
                [vals[a] for a in op["args"]], axis=2)
            per_layer[name] = vals[name]
            g = []
            c0 = 0
            for a in op["args"]:
                for (s0, s1, r, sg) in groups[a]:
                    g.append((c0 + s0, c0 + s1, r, sg))
                c0 += vals[a].shape[2]
            groups[name] = g
        elif kind == "layer_norm":
            vals[name] = _layernorm_fixed(vals[op["args"][0]])
            per_layer[name] = vals[name]
            _set_groups(name, 0, False, vals[name].shape[2])
        elif kind == "bypass":
            vals[name] = vals[op["args"][0]]
            per_layer[name] = vals[name]
            groups[name] = groups[op["args"][0]]
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
