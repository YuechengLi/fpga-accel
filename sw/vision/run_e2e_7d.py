#!/usr/bin/env python3
"""Stage 7d integrated end-to-end driver (YOLOv8n + DNANet).

Graph-generic driver for the capture programs in
logs/stage7d/program_{yolo,dnanet}.{json,npz}: real image in ->
output out, node by node, vs the COMPOSED funcsim golden
(fixed_forward on the program).

Op execution (see logs/stage7d/design.md for the mapping):
  RTL: conv2d (PE array, tb_e2e.sv), act/sigmoid (act.sv),
       simplegate (simplegate.sv), max_pool2d 2x2 (pool.sv mode 0),
       upsample (upsample.sv modes 0/1), concat (concat.sv).
  CPU (driver, same funcsim code as the golden): residual add
       (the RTL residual mode 0 is an unsigned add with no input
       rescaling; 49/52 DNANet residuals have a signed input and
       the inputs almost always carry different scales),
       relu (pass-through), gap, gmax, cmean, cmax, downsample,
       SPPF max_pool2d 5x5.

Per-node checkpoints under <outdir>/ckpt_<model>/ (one .npy +
ckpt.json per passed node, written only after that node passes
its budget); --resume continues from the last passed node.
The VM is replaced every few tens of minutes, so the YOLO
(~1.8 h) and DNANet (~6.8 h) runs must survive a kill.

Budgets: max|err| = 0 for every RTL node except act (PWL),
which is <= 1 LSB (the PWL RTL uses integer >>10 arithmetic
while the funcsim uses float, as in Stage 6).

Final checks:
  YOLO: the 6 Detect outputs -> CPU DFL + NMS decode ->
       detections vs the torch float detections on the same
       128x128 input (detection match, not mAP).
  DNANet: the 4 deep-supervision logits -> sigmoid -> mask vs
       the torch float mask on the same 64x64 crop (IoU).

Usage:
  python3 sw/vision/run_e2e_7d.py --model yolo [--resume]
  python3 sw/vision/run_e2e_7d.py --model dnanet [--resume]
  python3 sw/vision/run_e2e_7d.py --model yolo --smoke
      (first 3 convs + the first wave of each other op, no full run)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TB = os.path.join(REPO, "tb")
LOGD = os.path.join(REPO, "logs", "stage7d")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "sw", "common"))
sys.path.insert(0, os.path.join(REPO, "sw", "common", "passes"))

from capture_7d import load_program  # noqa: E402
from funcsim import (fixed_forward, max_abs_err, rescale_bytes,  # noqa: E402
                     rescale_input, _pool, _upsample,
                     _downsample_ac,
                     conv2d_fixed_rhu)  # noqa: E402
from sched_pe import gen_layer_steps, write_layer_vectors, assemble_output  # noqa: E402
from devices import DEVICES  # noqa: E402
from run_e2e import as_signed, load_ckpt, save_ckpt, compile_e2e_tb, run_pe_layer  # noqa: E402

DEV = DEVICES["zu19eg"]
CPU_OPS = ("residual", "relu", "gap", "gmax", "cmean", "cmax",
           "downsample")
# max_pool2d with K != (2,2) is CPU (SPPF 5x5); 2x2 is RTL.


def yolo_input(size=128):
    """YOLO e2e input: bus.jpg resized to (size,size), uint8."""
    from PIL import Image
    im = np.array(Image.open(os.path.join(
        REPO, "models", "vision", "yolov8n", "bus.jpg")).convert("RGB"))
    im = np.array(Image.fromarray(im).resize((size, size),
                                             Image.BILINEAR))
    return im.astype(np.uint8)


def dnanet_input():
    """DNANet e2e input: 64x64 target-centred crop of target1.png,
    uint8."""
    from PIL import Image
    im = np.array(Image.open(os.path.join(
        REPO, "models", "vision", "dnanet", "target1.png")
    ).convert("RGB"))
    crop = im[192:256, 128:192, :]
    assert crop.shape == (64, 64, 3), crop.shape
    return crop.astype(np.uint8)


def run_cmd(cmd, cwd, timeout, logfile=None):
    """Run a command; return (rc, stdout, stderr, seconds)."""
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout)
    dt = time.time() - t0
    if logfile:
        with open(logfile, "w") as f:
            f.write(f"$ {' '.join(cmd)}\nrc={r.returncode} "
                    f"({dt:.1f}s)\n--- stdout ---\n{r.stdout}"
                    f"\n--- stderr ---\n{r.stderr}\n")
    return r.returncode, r.stdout, r.stderr, dt


def pad_input(a, pad, pad_mode="zero"):
    """Zero/replicate-pad a (H,W,C) array. pad = (pt,pb,pl,pr)."""
    pt, pb, pl, pr = (int(v) for v in pad)
    if pt == pb == pl == pr == 0:
        return a
    if pad_mode == "replicate":
        return np.pad(a, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
    return np.pad(a, ((pt, pb), (pl, pr), (0, 0)), mode="constant")


def to_bytes(a):
    """(H,W,C) int64/uint8 -> uint8 bytes (signed values as
    two's complement)."""
    return (np.asarray(a, dtype=np.int64) & 0xFF).astype(np.uint8)


def bytes_to_values(a_u8, groups):
    """uint8 bytes -> values (int64 for signed channel groups,
    uint8 for unsigned), using the node's channel-group scales.
    The inverse of to_bytes. ``vals`` in the driver stores
    VALUES (like fixed_forward's per_layer), so the rescale /
    mean / max in the CPU nodes see the signed values, not the
    two's-complement bytes."""
    a = np.asarray(a_u8, dtype=np.int64)
    if not any(s for (_, _, _, s) in groups):
        return a.astype(np.uint8)
    out = a.copy()
    for c0, c1, _r, signed in groups:
        if signed:
            out[:, :, c0:c1] = np.where(
                a[:, :, c0:c1] >= 128, a[:, :, c0:c1] - 256,
                a[:, :, c0:c1])
    return out.astype(np.int64)


def run_pe_conv(op, in_arr, weights, outdir, tag):
    """One conv through the PE array (tb_e2e.sv wrapper).

    in_arr: the conv's input (H,W,C_in) as int64 (signed values
    already reinterpreted) or uint8. Returns (rtl_out uint8,
    wall_s, n_steps). Handles stride-2 (decimate after the
    stride-1 PE run) and the fc1/fc2 1x1-on-1x1 trick (pad the
    (1,1,C) input to (4,4,C), run, take [0,0,:]).
    """
    W_q = weights[op["name"]]["W_q"]
    b_q = weights[op["name"]]["b_q"]
    K = int(op["K"])
    C_out = int(op["C_out"])
    stride = op.get("stride", [1, 1])
    sx = int(stride[0])
    sy = int(stride[1])
    # Dot mode (C_out == 1) is only correct when tk is a
    # multiple of 8: pe_mac's dot reduction processes
    # tk_groups = tk >> 3 groups of 8 taps, so a tk that is not
    # a multiple of 8 drops (tk % 8) taps per K-chunk. The
    # scheduler (sched_pe) only guards TN < 8, not this, so
    # force outer mode here when tk % 8 != 0. sa.conv1 (7x7,
    # C_in=2, Kdim=98, tk=14) hit exactly this: max|err|=45 in
    # dot mode. Outer mode is TN-generic and handles C_out=1
    # (zero-padded last N-chunk, assemble_output ignores
    # co >= C_out).
    from sched_pe import largest_divisor
    _kdim = int(in_arr.shape[2]) * K * K
    _tk = largest_divisor(_kdim, 32)
    use_dot = (C_out == 1 and _tk % 8 == 0)
    layer = dict(
        name=op["name"], K=K, C_out=C_out,
        mode_dot=1 if use_dot else 0,
        tn=1 if use_dot else 8,
        sign_in=1 if op.get("signed_in") else 0,
        sign_out=1 if op.get("signed_out") else 0,
        shift=int(op["shift"]), act=0, slope=0)
    # The fused ReLU (DNANet conv1) is applied in pe_emit.
    if op.get("fused_act") == "relu":
        layer["act"] = 1
    pad = tuple(int(v) for v in op.get("pad", (0, 0, 0, 0)))
    pad_mode = op.get("pad_mode", "zero")
    H, W, C_in = in_arr.shape
    fc = (H == 1 and W == 1 and K == 1)
    if fc:
        # 1x1 conv on a 1x1 map: pad to (4,4,C) with zeros (the
        # M-tile is 4x4), run, take [0,0,:].
        padded = np.zeros((4, 4, C_in), dtype=np.int64)
        padded[0, 0, :] = np.asarray(in_arr, dtype=np.int64)[0, 0, :]
    else:
        padded = pad_input(np.asarray(in_arr, dtype=np.int64),
                           pad, pad_mode)
    # The scheduler reads the input as uint8 bytes; a signed
    # input is passed as two's-complement bytes (sign_in=1
    # makes the PE reinterpret them).
    inp_u8 = to_bytes(padded)
    steps, shape = gen_layer_steps(layer, inp_u8, W_q, b_q, DEV)
    rtl_out, dt = run_pe_layer(steps, shape, outdir, tag)
    if fc:
        rtl_out = rtl_out[0:1, 0:1, :]
    elif sx != 1 or sy != 1:
        rtl_out = rtl_out[::sy, ::sx, :]
    return rtl_out, dt, len(steps)


def compile_wave_tbs(outdir):
    """Compile the wave TBs once (act, simplegate, pool, upsample,
    concat). The e2e PE wrapper is compiled by run_e2e's
    compile_e2e_tb (called separately)."""
    tbs = [
        ("tb_act_7d.vvp", ["../rtl/vision/act.sv", "tb_act.sv"]),
        ("tb_sg_7d.vvp",
         ["../rtl/vision/simplegate.sv", "tb_simplegate.sv"]),
        ("tb_pool_7d.vvp", ["../rtl/vision/pool.sv", "tb_pool.sv"]),
        ("tb_up_7d.vvp",
         ["../rtl/vision/upsample.sv", "tb_upsample.sv"]),
        ("tb_cat_7d.vvp",
         ["../rtl/vision/concat.sv", "tb_concat.sv"]),
    ]
    for vvp, srcs in tbs:
        rc, so, se, dt = run_cmd(
            ["iverilog", "-g2012", "-o", vvp] + srcs, cwd=TB,
            timeout=300,
            logfile=os.path.join(outdir, f"compile_{vvp}.log"))
        if rc != 0:
            raise RuntimeError(f"{vvp} iverilog failed:\n{so}{se}")


def _read_out(path, n):
    vals = [int(x) for x in open(path).read().split()]
    if len(vals) != n:
        raise RuntimeError(f"{path}: output short {len(vals)} vs {n}")
    return np.array(vals, dtype=np.uint8)


def run_act_wave(op, in_arr, table, outdir, tag):
    """One act (sigmoid) through act.sv. in_arr is (H,W,C) int64
    (signed values already reinterpreted) or uint8. Returns
    (rtl_out uint8 (H,W,C), wall_s). PWL budget <= 1 LSB, MLUT
    exact -- enforced by the caller.
    The act TB's input array is capped at 8192 values per case
    (tb_act.sv: vals[0:8191]), so a large input is split into
    one TB case per <=8192-value chunk (the act is elementwise,
    so chunking is exact); the table is re-loaded per case."""
    H, W, C = in_arr.shape
    func = op.get("func", "sigmoid")
    func_id = {"sigmoid": 2, "tanh": 3, "gelu": 4, "silu": 5}[func]
    impl = table.get("impl", "mlut")
    r_in = int(op.get("r_in", 0))
    r_out = int(op.get("r_out", 0))
    scale_in = 255.0 / (2.0 ** r_in)
    scale_out = 255.0 / (2.0 ** r_out)
    x = np.asarray(in_arr, dtype=np.float64) / scale_in
    nvals = int(x.size)
    vpath = os.path.join(TB, "vectors_act.txt")
    opath = os.path.join(TB, "out_act.txt")
    CHUNK = 8192
    chunks = [x.reshape(-1)[i:i + CHUNK]
              for i in range(0, nvals, CHUNK)]
    with open(vpath, "w") as f:
        f.write(f"{len(chunks)}\n")
        for ch in chunks:
            if impl == "pwl":
                xq = np.clip(np.round(ch * 2048.0), -16384, 16384
                             ).astype(int)
                y0 = [int(v) for v in table["y0"]]
                ds = [int(v) for v in table["dslope"]]
                assert len(y0) == 32 and len(ds) == 32
                f.write(f"{func_id} 0 {len(ch)}\n")
                for a, b in zip(y0, ds):
                    f.write(f"{a} {b}\n")
                for v in xq:
                    f.write(f"{int(v)}\n")
                out_scale = 32768.0  # Q1.15
            else:
                in12 = np.clip(np.round((ch + 8.0) * 256.0), 0, 4095
                               ).astype(int)
                H1, H2 = int(table["H1"]), int(table["H2"])
                nbands = int(table["nbands"])
                base = [int(v) for v in table["base"]]
                sel = [int(v) for v in table["sel"]]
                fine = [int(v) for v in table["fine"]]
                f.write(f"6 {func_id} {len(ch)}\n")
                f.write(f"{H1} {H2} {nbands}\n")
                for v in base:
                    f.write(f"{v}\n")
                for v in sel:
                    f.write(f"{v}\n")
                for v in fine:
                    f.write(f"{v}\n")
                for v in in12:
                    f.write(f"{int(v)}\n")
                out_scale = 2048.0  # Q5.11
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_act_7d.vvp"], cwd=TB, timeout=600,
        logfile=os.path.join(outdir, f"sim_act_{tag}.log"))
    if rc != 0 or "ERROR" in so:
        raise RuntimeError(f"act vvp failed rc={rc} [{tag}]:\n"
                           + so[-2000:] + se[-2000:])
    s = np.array([int(v) for v in open(opath).read().split()],
                 dtype=np.float64)
    if s.size != nvals:
        raise RuntimeError(f"act output short: {s.size} vs {nvals}")
    y = s / out_scale
    v = np.clip(np.round(y * scale_out), 0, 255).astype(np.uint8)
    return v.reshape((H, W, C)), dt


def run_simplegate_wave(op, a_arr, b_arr, outdir, tag):
    """One simplegate through simplegate.sv. a_arr/b_arr are
    (H,W,C) int64/uint8 (one case per channel plane). Returns
    (rtl_out uint8 (H,W,C), wall_s). A (1,1,C) operand (the
    DNANet attention map) is broadcast to the other operand's
    (H,W,C) -- the attention map is constant over the plane."""
    H, W, C = a_arr.shape
    if b_arr.shape != a_arr.shape:
        # Broadcast a smaller operand to the other's shape:
        #  - (1,1,C) attention map over the (H,W,C) plane
        #    (DNANet ca_mul), or
        #  - (H,W,1) attention map over the (H,W,C) channels
        #    (DNANet sa_mul).
        if a_arr.shape[0] == 1 and a_arr.shape[1] == 1:
            a_arr = np.broadcast_to(a_arr, b_arr.shape).copy()
            H, W, C = a_arr.shape
        elif b_arr.shape[0] == 1 and b_arr.shape[1] == 1:
            b_arr = np.broadcast_to(b_arr, a_arr.shape).copy()
        elif a_arr.shape[2] == 1 and a_arr.shape[:2] == b_arr.shape[:2]:
            a_arr = np.broadcast_to(a_arr, b_arr.shape).copy()
            H, W, C = a_arr.shape
        elif b_arr.shape[2] == 1 and b_arr.shape[:2] == a_arr.shape[:2]:
            b_arr = np.broadcast_to(b_arr, a_arr.shape).copy()
        else:
            raise ValueError(
                f"simplegate {tag}: shape mismatch "
                f"{a_arr.shape} vs {b_arr.shape}")
    shift = int(op.get("shift", 8))
    sign_in = 1 if op.get("sign_in") else 0
    sign_out = 1 if op.get("sign_out") else 0
    a_b = to_bytes(a_arr)
    b_b = to_bytes(b_arr)
    vpath = os.path.join(TB, "vectors_simplegate.txt")
    opath = os.path.join(TB, "out_simplegate.txt")
    with open(vpath, "w") as f:
        f.write(f"{C}\n")
        for c in range(C):
            f.write(f"{shift} {W} {H} {sign_in} {sign_out}\n")
            f.write(" ".join(str(int(v)) for v in
                             a_b[:, :, c].reshape(-1)) + " ")
            f.write(" ".join(str(int(v)) for v in
                             b_b[:, :, c].reshape(-1)) + "\n")
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_sg_7d.vvp"], cwd=TB, timeout=600,
        logfile=os.path.join(outdir, f"sim_sg_{tag}.log"))
    if rc != 0 or "ERROR" in so:
        raise RuntimeError(f"simplegate vvp failed rc={rc} [{tag}]:\n"
                           + so[-2000:] + se[-2000:])
    vals = _read_out(opath, H * W * C)
    # The TB emits one case per channel plane, so the output is
    # in (C,H,W) case order -- transpose to (H,W,C).
    return vals.reshape((C, H, W)).transpose(1, 2, 0), dt


def run_pool_wave(op, in_arr, outdir, tag):
    """One max_pool2d 2x2 through pool.sv mode 0. in_arr is
    (H,W,C) uint8. Returns (rtl_out uint8 (H/2,W/2,C), wall_s)."""
    H, W, C = in_arr.shape
    assert H % 2 == 0 and W % 2 == 0, "pool input must be even"
    a_b = to_bytes(in_arr)
    vpath = os.path.join(TB, "vectors_pool.txt")
    opath = os.path.join(TB, "out_pool.txt")
    with open(vpath, "w") as f:
        f.write(f"{C}\n")
        for c in range(C):
            f.write(f"0 {W} {H} 0 {W // 2} {H // 2}\n")
            f.write(" ".join(str(int(v)) for v in
                             a_b[:, :, c].reshape(-1)) + "\n")
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_pool_7d.vvp"], cwd=TB, timeout=600,
        logfile=os.path.join(outdir, f"sim_pool_{tag}.log"))
    if rc != 0 or "ERROR" in so:
        raise RuntimeError(f"pool vvp failed rc={rc} [{tag}]:\n"
                           + so[-2000:] + se[-2000:])
    vals = _read_out(opath, (H // 2) * (W // 2) * C)
    # (C,H/2,W/2) case order -> (H/2,W/2,C).
    return vals.reshape((C, H // 2, W // 2)).transpose(1, 2, 0), dt


def run_upsample_wave(op, in_arr, outdir, tag):
    """One upsample 2x through upsample.sv. mode 0 (nearest) or
    1 (bilinear, RTL formula). in_arr is (H,W,C) uint8. Returns
    (rtl_out uint8 (2H,2W,C), wall_s). The program stores the
    mode as a string ('nearest' / 'bilinear_rtl'); the TB wants
    the RTL mode number."""
    H, W, C = in_arr.shape
    m = op.get("mode", 1)
    if isinstance(m, str):
        mode = {"nearest": 0, "bilinear_rtl": 1, "bilinear": 1}[m]
    else:
        mode = int(m)
    a_b = to_bytes(in_arr)
    vpath = os.path.join(TB, "vectors_upsample.txt")
    opath = os.path.join(TB, "out_upsample.txt")
    with open(vpath, "w") as f:
        f.write(f"{C}\n")
        for c in range(C):
            f.write(f"{mode} {W} {H}\n")
            f.write(" ".join(str(int(v)) for v in
                             a_b[:, :, c].reshape(-1)) + "\n")
    rc, so, se, dt = run_cmd(
        ["vvp", "tb_up_7d.vvp"], cwd=TB, timeout=600,
        logfile=os.path.join(outdir, f"sim_up_{tag}.log"))
    if rc != 0 or "ERROR" in so:
        raise RuntimeError(f"upsample vvp failed rc={rc} [{tag}]:\n"
                           + so[-2000:] + se[-2000:])
    vals = _read_out(opath, (2 * H) * (2 * W) * C)
    # (C,2H,2W) case order -> (2H,2W,C).
    return vals.reshape((C, 2 * H, 2 * W)).transpose(1, 2, 0), dt


def run_concat_wave(op, inputs, outdir, tag):
    """One concat through concat.sv (mode 0, sequential). inputs
    is a list of (H,W,Ci) uint8 arrays, all the same H/W.
    Chained pairwise: the output of one pair feeds the next.
    The concat module is channel-major: one TB case per PAIR of
    consecutive output planes (A = output plane 2i, B = output
    plane 2i+1); mode 0 emits the A frame then the B frame.
    Returns (rtl_out uint8 (H,W,sum Ci), wall_s)."""
    H, W = inputs[0].shape[:2]
    vpath = os.path.join(TB, "vectors_concat.txt")
    opath = os.path.join(TB, "out_concat.txt")
    cur = to_bytes(inputs[0])
    total_dt = 0.0
    for nxt in inputs[1:]:
        nxt_b = to_bytes(nxt)
        assert nxt_b.shape[:2] == (H, W), "concat H/W mismatch"
        ca, cb = cur.shape[2], nxt_b.shape[2]
        # The concatenated channel list, in order.
        planes = [cur[:, :, i] for i in range(ca)] + \
                 [nxt_b[:, :, i] for i in range(cb)]
        n_out = ca + cb
        npair = (n_out + 1) // 2
        with open(vpath, "w") as f:
            f.write(f"{npair}\n")
            for i in range(npair):
                a = planes[2 * i]
                b = planes[2 * i + 1] if 2 * i + 1 < n_out else a
                f.write(f"0 {W} {H}\n")
                f.write(" ".join(str(int(v)) for v in
                                 a.reshape(-1)) + "\n")
                f.write(" ".join(str(int(v)) for v in
                                 b.reshape(-1)) + "\n")
        rc, so, se, dt = run_cmd(
            ["vvp", "tb_cat_7d.vvp"], cwd=TB, timeout=600,
            logfile=os.path.join(outdir, f"sim_cat_{tag}.log"))
        total_dt += dt
        if rc != 0 or "ERROR" in so:
            raise RuntimeError(f"concat vvp failed rc={rc} [{tag}]:\n"
                               + so[-2000:] + se[-2000:])
        vals = _read_out(opath, npair * 2 * H * W)
        out = np.zeros((H, W, n_out), dtype=np.uint8)
        per = 2 * H * W
        for i in range(npair):
            fr = vals[i * per:(i + 1) * per]
            out[:, :, 2 * i] = fr[:H * W].reshape((H, W))
            if 2 * i + 1 < n_out:
                out[:, :, 2 * i + 1] = fr[H * W:].reshape((H, W))
        cur = out
    return cur, total_dt


def track_groups(program, per_layer, in_shape):
    """Per-node channel-group scales [(c0,c1,r,signed), ...],
    mirroring fixed_forward's _set_groups. Needed by the driver
    to rescale a conv's input to the conv's r_in (a concat input
    carries mixed per-channel scales; 10 YOLO + 70 DNANet convs
    need the rescale)."""
    ops = {o["name"]: o for o in program["ops"]}
    groups = {"input": [(0, in_shape[2], 0, False)]}
    for o in program["ops"]:
        name = o["name"]
        if o["op"] == "conv2d":
            groups[name] = [(0, int(o["C_out"]),
                             int(o.get("r_out", 0)),
                             bool(o.get("signed_out", False)))]
        elif o["op"] == "act":
            groups[name] = [(0, per_layer[name].shape[2],
                             int(o.get("r_out", 0)), False)]
        elif o["op"] == "simplegate":
            groups[name] = [(0, per_layer[name].shape[2],
                             int(o.get("r_out", 0)),
                             bool(o.get("sign_out", False)))]
        elif o["op"] == "residual":
            groups[name] = [(0, per_layer[name].shape[2],
                             int(o.get("r_out", 0)),
                             bool(o.get("signed_out", False)))]
        elif o["op"] in ("relu", "max_pool2d", "avg_pool2d", "gap",
                         "gmax", "upsample"):
            # uint8 output (fixed_forward stores these as uint8):
            # force unsigned even if the producer is signed, else
            # a consumer's rescale would clip 128..255 to 127 and
            # a checkpoint resume would reinterpret them as
            # negative. (DNANet gap/gmax: producer conv2 is signed.)
            groups[name] = [(c0, c1, r, False) for (c0, c1, r, _s)
                            in groups[o["args"][0]]]
        elif o["op"] == "downsample":
            # fixed_forward sets the downsample groups from the
            # op's own r_out / signed_out, not the producer's.
            groups[name] = [(0, per_layer[name].shape[2],
                             int(o.get("r_out", 0)),
                             bool(o.get("signed_out", False)))]
        elif o["op"] in ("cmean", "cmax"):
            g = groups[o["args"][0]]
            groups[name] = [(0, 1, g[0][2], False)]
        elif o["op"] == "concat":
            gg = []
            off = 0
            for a in o["args"]:
                for (c0, c1, r, s) in groups[a]:
                    gg.append((off + c0, off + c1, r, s))
                off += per_layer[a].shape[2]
            groups[name] = gg
    return groups


def cpu_node(op, vals, groups):
    """Execute a CPU node in Python with the same funcsim code as
    the golden (residual, relu, gap, gmax, cmean, cmax,
    downsample, SPPF max_pool2d). ``vals`` maps node name ->
    the node's input array (the RTL/CPU output of the producer).
    Returns the node's output array."""
    kind = op["op"]
    if kind == "residual":
        a = rescale_input(vals[op["args"][0]],
                          groups[op["args"][0]],
                          int(op.get("r_in", 0)))
        b = rescale_input(vals[op["args"][1]],
                          groups[op["args"][1]],
                          int(op.get("r_in", 0)))
        v = (a.astype(np.int64) + b.astype(np.int64)
             if op.get("mode", "add") == "add"
             else a.astype(np.int64) - b.astype(np.int64))
        if op.get("signed_out"):
            return np.clip(v, -128, 127).astype(np.int64)
        return np.clip(v, 0, 255).astype(np.uint8)
    if kind == "relu":
        return np.maximum(vals[op["args"][0]], 0)
    if kind == "gap":
        a = vals[op["args"][0]].astype(np.float64)
        return np.round(a.mean(axis=(0, 1), keepdims=True)).astype(
            np.uint8)
    if kind == "gmax":
        a = vals[op["args"][0]]
        return np.clip(a.max(axis=(0, 1), keepdims=True), 0, 255
                       ).astype(np.uint8)
    if kind == "cmean":
        a = vals[op["args"][0]].astype(np.float64)
        return np.clip(np.round(a.mean(axis=2, keepdims=True)), 0, 255
                       ).astype(np.uint8)
    if kind == "cmax":
        a = vals[op["args"][0]]
        return np.clip(a.max(axis=2, keepdims=True), 0, 255
                       ).astype(np.uint8)
    if kind == "downsample":
        # Bilinear 0.5x, align_corners=True (DNANet `down`).
        # Dequantize -> _downsample_ac -> re-quantize, exactly as
        # fixed_forward's downsample case. (Do NOT use
        # _upsample(scale=0.5): it does int(0.5)==0 and returns
        # an empty array.)
        a = vals[op["args"][0]]
        g = groups[op["args"][0]]
        if len(g) == 1:
            r_in = g[0][2]
            af = a.astype(np.float64) / (255.0 / (2.0 ** r_in))
        else:
            r_in = max(gr[2] for gr in g)
            au = rescale_input(a, g, r_in)
            af = au.astype(np.float64) / (255.0 / (2.0 ** r_in))
        f = _downsample_ac(af)
        r_out = int(op.get("r_out", r_in))
        v = np.round(f * (255.0 / (2.0 ** r_out)))
        if op.get("signed_out"):
            return np.clip(v, -128, 127).astype(np.int64)
        return np.clip(v, 0, 255).astype(np.uint8)
    if kind == "max_pool2d":
        # SPPF 5x5 stride-1 pad-2 (the RTL pool is 2x2-only).
        a = vals[op["args"][0]]
        kw = {"K": op.get("K", [2, 2]), "stride": op.get("stride",
                                                         [2, 2]),
              "pad": op.get("pad", [0, 0, 0, 0])}
        return _pool(a, kw, "max").astype(np.uint8)
    raise ValueError(f"cpu_node: unhandled op {kind}")


def run_chain(model, program, per_layer, fin_u8, outdir, resume=False,
              smoke=False):
    """Node-by-node integrated run. Returns (ok, results).

    fin_u8: the raw e2e input (uint8). per_layer: the COMPOSED
    funcsim golden (fixed_forward on the program). Each RTL node's
    output is checked against its golden; a CPU node's output is
    computed with the same funcsim code as the golden (so it
    matches by construction, and the check catches wiring bugs).
    """
    ops = program["ops"]
    weights = program["_weights_arrays"]
    acts = program.get("_act_tables_full") or program.get(
        "act_tables", {})
    groups = track_groups(program, per_layer, fin_u8.shape)
    ckpt_dir = os.path.join(outdir, f"ckpt_{model}")
    done, ckpt_records = (load_ckpt(ckpt_dir) if resume
                          else ({}, {}))
    if done:
        print(f"  resuming from checkpoint: {len(done)} node(s) "
              f"already passed", flush=True)
    results = dict(ckpt_records)
    vals = {"input": fin_u8}
    for name in done:
        vals[name] = bytes_to_values(done[name], groups[name])
    # The PE wrapper + wave TBs are compiled once per run.
    compile_e2e_tb()
    compile_wave_tbs(outdir)
    order = [o["name"] for o in ops if o["op"] != "input"]
    if smoke:
        # A fixed short prefix (the first 12 nodes in program
        # order, so every kept node's inputs are kept too).
        # Covers the conv + act + simplegate waves and the
        # rescale + checkpointing; the full run covers the rest
        # (concat / residual / pool / upsample).
        order = order[:12]
        print(f"  smoke: first {len(order)} nodes", flush=True)
    n_done = 0
    for o in ops:
        name = o["name"]
        if o["op"] == "input":
            continue
        if smoke and name not in order:
            continue
        if name in done:
            print(f"  {name}: resumed from checkpoint", flush=True)
            n_done += 1
            continue
        kind = o["op"]
        golden = per_layer[name]
        t0 = time.time()
        if kind == "conv2d":
            # Rescale the input to the conv's r_in (a concat
            # input carries mixed per-channel scales), then the
            # PE wave. The rescaled input has no negative values
            # (verified), so sign_in = 0.
            src = o["args"][0]
            a_rs = rescale_input(vals[src], groups[src],
                                 int(o.get("r_in", 0)))
            assert a_rs.min() >= 0, \
                f"{name}: rescaled conv input has negatives"
            rtl_out, dt, nsteps = run_pe_conv(o, a_rs, weights,
                                              outdir, name)
            # Independent golden: conv2d_fixed_rhu on the padded
            # input (the same computation as fixed_forward's conv
            # case, via a different path), cross-checked against
            # fixed_forward's per-node golden.
            pad = tuple(int(v) for v in o.get("pad", (0, 0, 0, 0)))
            H, W, _ = a_rs.shape
            if H == 1 and W == 1 and int(o["K"]) == 1:
                padded = np.zeros((4, 4, a_rs.shape[2]),
                                  dtype=np.int64)
                padded[0, 0, :] = a_rs[0, 0, :]
                ind = conv2d_fixed_rhu(
                    padded, weights[name]["W_q"],
                    weights[name]["b_q"], int(o["shift"]),
                    pad=(0, 0, 0, 0),
                    signed_out=bool(o.get("signed_out", False)))
                ind = ind[0:1, 0:1, :]
            else:
                padded = pad_input(a_rs, pad, o.get("pad_mode",
                                                    "zero"))
                ind = conv2d_fixed_rhu(
                    padded, weights[name]["W_q"],
                    weights[name]["b_q"], int(o["shift"]),
                    pad=(0, 0, 0, 0),
                    signed_out=bool(o.get("signed_out", False)))
                sx, sy = (int(v) for v in o.get("stride", [1, 1]))
                if sx != 1 or sy != 1:
                    ind = ind[::sy, ::sx, :]
            if o.get("fused_act") == "relu":
                ind = np.maximum(ind, 0)
            cross = int(max_abs_err(ind.astype(np.int64),
                                    golden.astype(np.int64)))
            if cross != 0:
                print(f"  {name}: GOLDEN MISMATCH (independent "
                      f"golden vs fixed_forward max|err|={cross})")
                return False, results
            cmp_rtl = (as_signed(rtl_out) if o.get("signed_out")
                       else rtl_out.astype(np.int32))
            err = int(max_abs_err(cmp_rtl, golden.astype(np.int64)))
            budget = 0
            tag = f"{nsteps} steps"
        elif kind == "act":
            # vals[src] already holds the VALUES (int64 for a
            # signed producer), so the act wave converts them
            # directly with the node's r_in.
            table = acts[name]
            rtl_out, dt = run_act_wave(o, vals[o["args"][0]], table,
                                       outdir, name)
            err = int(max_abs_err(rtl_out.astype(np.int64),
                                  golden.astype(np.int64)))
            budget = 1 if table.get("impl") == "pwl" else 0
            tag = f"{table.get('impl', '?')} (budget <={budget})"
        elif kind == "simplegate":
            a_in = vals[o["args"][0]]
            b_in = vals[o["args"][1]]
            rtl_out, dt = run_simplegate_wave(o, a_in, b_in, outdir,
                                              name)
            cmp_rtl = (as_signed(rtl_out) if o.get("sign_out")
                       else rtl_out.astype(np.int32))
            err = int(max_abs_err(cmp_rtl, golden.astype(np.int64)))
            budget = 0
            tag = ""
        elif kind == "max_pool2d":
            if list(o.get("K", [2, 2])) != [2, 2]:
                # SPPF 5x5: CPU (the RTL pool is 2x2-only).
                rtl_out = cpu_node(o, vals, groups)
                dt = time.time() - t0
                err = int(max_abs_err(
                    rtl_out.astype(np.int64),
                    golden.astype(np.int64)))
                budget = 0
                tag = "CPU (SPPF 5x5)"
            else:
                rtl_out, dt = run_pool_wave(o, vals[o["args"][0]],
                                            outdir, name)
                err = int(max_abs_err(
                    rtl_out.astype(np.int64),
                    golden.astype(np.int64)))
                budget = 0
                tag = "pool RTL mode 0"
        elif kind == "upsample":
            # The RTL upsample averages bytes as UNSIGNED; a signed
            # input would be averaged wrong (up16_a4_0 failed with
            # max|err|=128: the final-aggregation adapter convs are
            # signed, 138/256 values negative). Route signed-input
            # upsamples to the CPU with the same funcsim code as
            # the golden (matches by construction, as for the other
            # CPU ops). Unsigned-input upsamples stay on the RTL.
            src = o["args"][0]
            if any(s for (_, _, _, s) in groups[src]):
                rtl_out = _upsample(vals[src], o).astype(np.uint8)
                dt = time.time() - t0
                tag = "CPU (signed input)"
            else:
                rtl_out, dt = run_upsample_wave(o, vals[src], outdir,
                                                name)
                tag = f"mode {o.get('mode', 1)}"
            err = int(max_abs_err(rtl_out.astype(np.int64),
                                  golden.astype(np.int64)))
            budget = 0
        elif kind == "concat":
            ins = [vals[a] for a in o["args"]]
            rtl_out, dt = run_concat_wave(o, ins, outdir, name)
            err = int(max_abs_err(rtl_out.astype(np.int64),
                                  golden.astype(np.int64)))
            budget = 0
            tag = f"{len(ins)} inputs"
        elif kind in CPU_OPS:
            rtl_out = cpu_node(o, vals, groups)
            dt = time.time() - t0
            err = int(max_abs_err(rtl_out.astype(np.int64),
                                  golden.astype(np.int64)))
            budget = 0
            tag = "CPU"
        else:
            raise ValueError(f"run_chain: unhandled op {kind} "
                             f"({name})")
        status = "PASS" if err <= budget else "FAIL"
        print(f"  {name} [{kind}]: RTL vs funcsim max|err|={err} "
              f"(budget <={budget}) [{status}] {tag} ({dt:.1f}s)",
              flush=True)
        record = dict(max_err_funcsim=err, budget=budget,
                      shape=list(golden.shape), wall_s=round(dt, 1),
                      kind=kind, tag=tag)
        results[name] = record
        if err > budget:
            return False, results
        # Store the node's output as VALUES (int64 for a signed
        # output, uint8 for unsigned), like fixed_forward's
        # per_layer. The wave executors convert to bytes when
        # they write the vectors; the checkpoint stores bytes.
        if kind in ("conv2d", "simplegate") and (
                o.get("signed_out") or o.get("sign_out")):
            vals[name] = as_signed(rtl_out).astype(np.int64)
        else:
            vals[name] = np.asarray(rtl_out)
        save_ckpt(ckpt_dir, name, to_bytes(vals[name]),
                  record, order)
        n_done += 1
    print(f"  chain: {n_done} node(s) passed", flush=True)
    return True, results


# ---------------------------------------------------------------------------
# Final checks.
# ---------------------------------------------------------------------------
def dfl_decode(box_maps, cls_maps, strides=(8, 16, 32), reg_max=16):
    """Decode the 6 Detect outputs (float, dequantized) to the
    (336, 84) prediction tensor [x1,y1,x2,y2,80 cls scores].
    The Detect returns xywh; convert to xyxy for NMS."""
    preds = []
    for box, cls, s in zip(box_maps, cls_maps, strides):
        H, W, _ = box.shape
        b = box.reshape(H * W, 4, reg_max)
        b = b - np.max(b, axis=2, keepdims=True)
        e = np.exp(b)
        p = e / e.sum(axis=2, keepdims=True)
        d = (p * np.arange(reg_max)).sum(axis=2)  # (l,t,r,b)
        cy, cx = np.mgrid[0:H, 0:W]
        ax = (cx.reshape(-1) + 0.5) * s
        ay = (cy.reshape(-1) + 0.5) * s
        cxp = ax + (d[:, 2] - d[:, 0]) / 2 * s
        cyp = ay + (d[:, 3] - d[:, 1]) / 2 * s
        w = (d[:, 0] + d[:, 2]) * s
        h = (d[:, 1] + d[:, 3]) * s
        x1 = cxp - w / 2
        y1 = cyp - h / 2
        x2 = cxp + w / 2
        y2 = cyp + h / 2
        cs = 1.0 / (1.0 + np.exp(-cls.reshape(H * W, -1)))
        preds.append(np.concatenate(
            [x1[:, None], y1[:, None], x2[:, None], y2[:, None], cs],
            axis=1))
    return np.concatenate(preds, axis=0)


def nms_np(pred, conf=0.25, iou=0.45):
    """Per-class NMS on the (N,84) prediction tensor. Returns
    (M,6) [x1,y1,x2,y2,score,class]."""
    out = []
    for c in range(80):
        sc = pred[:, 4 + c]
        idx = np.where(sc > conf)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-sc[idx])]
        boxes = pred[order, :4]
        keep = []
        while len(order):
            keep.append(order[0])
            if len(order) == 1:
                break
            x1 = np.maximum(boxes[0, 0], boxes[1:, 0])
            y1 = np.maximum(boxes[0, 1], boxes[1:, 1])
            x2 = np.minimum(boxes[0, 2], boxes[1:, 2])
            y2 = np.minimum(boxes[0, 3], boxes[1:, 3])
            iw = np.maximum(0, x2 - x1)
            ih = np.maximum(0, y2 - y1)
            inter = iw * ih
            a1 = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3]
                                                - boxes[0, 1])
            a2 = (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3]
                                                  - boxes[1:, 1])
            ov = inter / (a1 + a2 - inter)
            m = np.where(ov <= iou)[0] + 1
            order = order[m]
            boxes = boxes[m]
        for k in keep:
            out.append([*pred[k, :4], pred[k, 4 + c], c])
    return np.array(out) if out else np.zeros((0, 6))


def dequant(arr, r_out, signed=False):
    """Fixed-point bytes/values -> float (scale 255/2**r_out)."""
    a = np.asarray(arr, dtype=np.float64)
    if signed:
        a = np.where(a >= 128, a - 256, a) if a.dtype == np.uint8 \
            else a
    return a / (255.0 / (2.0 ** r_out))


def final_yolo_pred(program, vals):
    """The DFL-decoded prediction tensor (N,84) [x1,y1,x2,y2,
    80 cls scores] from the 6 Detect outputs in `vals`
    (fixed-point, dequantized). No NMS -- the raw decode, so a
    unit test can compare it against the torch Detect output."""
    ops = {o["name"]: o for o in program["ops"]}
    box_maps, cls_maps = [], []
    for j in range(3):
        b = vals[f"det_box{j}"]
        c = vals[f"det_cls{j}"]
        box_maps.append(dequant(b, int(ops[f"det_box{j}"]["r_out"]),
                                bool(ops[f"det_box{j}"]
                                     .get("signed_out"))))
        cls_maps.append(dequant(c, int(ops[f"det_cls{j}"]["r_out"]),
                                bool(ops[f"det_cls{j}"]
                                     .get("signed_out"))))
    return dfl_decode(box_maps, cls_maps)


def final_dnanet_masks(program, vals):
    """The 4 deep-supervision masks (list of 4 boolean (H,W)
    arrays) from the fixed-point logits in `vals`."""
    ops = {o["name"]: o for o in program["ops"]}
    logits = []
    for i in range(1, 5):
        name = f"final{i}"
        logits.append(dequant(vals[name], int(ops[name]["r_out"]),
                              bool(ops[name].get("signed_out"))))
    sig = [1.0 / (1.0 + np.exp(-z[:, :, 0])) for z in logits]
    return [s > 0.5 for s in sig]


def final_yolo(program, vals, fin_u8):
    """YOLO final check: the 6 Detect outputs (RTL) -> dequantize
    -> DFL + NMS -> detections, vs the torch float detections on
    the same 128x128 input. Returns (ok, results)."""
    pred = final_yolo_pred(program, vals)
    det = nms_np(pred)
    # Torch float reference on the same input.
    import os as _os
    _os.environ.setdefault("YOLO_VERBOSE", "false")
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    m = YOLO(os.path.join(REPO, "models", "vision", "yolov8n",
                          "raw", "yolov8n.pt"))
    model = m.model.eval()
    x = torch.from_numpy(fin_u8.transpose(2, 0, 1)).float(
    ).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    ref = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45,
                              max_det=300)[0].numpy()
    print(f"  final YOLO: RTL detections {len(det)}, "
          f"torch float detections {len(ref)}", flush=True)
    print(f"    RTL: {np.round(det, 1).tolist()}", flush=True)
    print(f"    torch: {np.round(ref, 1).tolist()}", flush=True)
    ok = len(det) == len(ref)
    if ok and len(det):
        # Same classes, and each RTL box within 2 px of the
        # matching torch box (the quantization moves the boxes
        # by < 1 px at this resolution; 2 px is the bar).
        det_s = det[np.argsort(det[:, 5] * 1000 + det[:, 4])]
        ref_s = ref[np.argsort(ref[:, 5] * 1000 + ref[:, 4])]
        if not np.array_equal(det_s[:, 5], ref_s[:, 5]):
            ok = False
        else:
            box_err = np.abs(det_s[:, :4] - ref_s[:, :4]).max()
            print(f"    box max|diff| vs torch = {box_err:.2f} px",
                  flush=True)
            ok = box_err <= 2.0
    return ok, dict(n_rtl=int(len(det)), n_torch=int(len(ref)),
                    rtl=det.tolist(), torch=ref.tolist())


def final_dnanet(program, vals, fin_u8):
    """DNANet final check: the 4 deep-supervision logits (RTL)
    -> sigmoid -> mask, vs the torch float mask on the same
    64x64 crop (IoU). Returns (ok, results)."""
    masks = final_dnanet_masks(program, vals)
    # The aggregate mask (mean of the 4 sigmoids > 0.5) is
    # recomputed here for the IoU report.
    ops = {o["name"]: o for o in program["ops"]}
    logits = []
    for i in range(1, 5):
        name = f"final{i}"
        logits.append(dequant(vals[name], int(ops[name]["r_out"]),
                              bool(ops[name].get("signed_out"))))
    sig = [1.0 / (1.0 + np.exp(-z[:, :, 0])) for z in logits]
    agg = (np.mean(sig, axis=0) > 0.5)
    # Torch float reference on the same crop.
    import torch
    sys.path.insert(0, os.path.join(REPO, "models", "vision",
                                    "dnanet"))
    from model_DNANet import DNANet, Res_CBAM_block
    d = np.load(os.path.join(REPO, "models", "vision", "dnanet",
                             "dnanet_float.npz"))
    model = DNANet(num_classes=1, input_channels=3,
                   block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                   nb_filter=[16, 32, 64, 128, 256],
                   deep_supervision=True)
    convs = [(p, m) for p, m in model.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    sd = {}
    for i, (path, c) in enumerate(convs):
        sd[path + ".weight"] = torch.from_numpy(
            np.transpose(d[f"L{i}_W"].astype(np.float32),
                         (3, 2, 0, 1)))
        if c.bias is not None:
            sd[path + ".bias"] = torch.from_numpy(
                d[f"L{i}_b"].astype(np.float32))
        if bool(d[f"L{i}_has_bn"][0]):
            if path.endswith(".conv1"):
                bpath = path[:-len(".conv1")] + ".bn1"
            elif path.endswith(".conv2"):
                bpath = path[:-len(".conv2")] + ".bn2"
            else:
                bpath = path[:-len(".shortcut.0")] + ".shortcut.1"
            for suf, key in [("weight", "bn_w"), ("bias", "bn_b"),
                             ("running_mean", "bn_mean"),
                             ("running_var", "bn_var")]:
                sd[bpath + "." + suf] = torch.from_numpy(
                    d[f"L{i}_{key}"].astype(np.float32))
    model.load_state_dict(sd, strict=False)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    xn = (fin_u8.astype(np.float64) / 255.0 - mean) / std
    x = torch.from_numpy(xn.astype(np.float32)).permute(2, 0, 1
                                                         ).unsqueeze(0)
    with torch.no_grad():
        outs = model(x)
    ref_masks = [(1.0 / (1.0 + np.exp(-o[0, 0].numpy()))) > 0.5
                 for o in outs]
    ref_agg = (np.mean([1.0 / (1.0 + np.exp(-o[0, 0].numpy()))
                        for o in outs], axis=0) > 0.5)

    def iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union else 1.0
    ious = [iou(masks[i], ref_masks[i]) for i in range(4)]
    agg_iou = iou(agg, ref_agg)
    print(f"  final DNANet: mask IoU per final = "
          f"{[round(v, 4) for v in ious]}, aggregate IoU = "
          f"{agg_iou:.4f} (RTL px per final = "
          f"{[int(m.sum()) for m in masks]}, torch = "
          f"{[int(m.sum()) for m in ref_masks]})", flush=True)
    ok = agg_iou >= 0.9 and min(ious) >= 0.9
    return ok, dict(ious=[float(v) for v in ious],
                    agg_iou=float(agg_iou),
                    n_rtl=[int(m.sum()) for m in masks],
                    n_torch=[int(m.sum()) for m in ref_masks])


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["yolo", "dnanet"])
    ap.add_argument("--outdir", default="logs/stage7d")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    outdir = os.path.join(REPO, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    model = args.model
    print(f"=== {model}: integrated end-to-end (Stage 7d) ===",
          flush=True)
    fin_u8 = yolo_input() if model == "yolo" else dnanet_input()
    print(f"  input {fin_u8.shape} (raw, no frontend blur -- "
          f"the 7d models take the raw image)", flush=True)
    program = load_program(model, args.outdir)
    print(f"  program: {len(program['ops'])} ops, "
          f"{sum(1 for o in program['ops'] if o['op'] == 'conv2d')} "
          f"convs", flush=True)
    t0 = time.time()
    per_layer, _out = fixed_forward(program, fin_u8)
    print(f"  COMPOSED funcsim golden: {len(per_layer)} nodes "
          f"({time.time() - t0:.1f}s)", flush=True)
    ok, results = run_chain(model, program, per_layer, fin_u8,
                            outdir, resume=args.resume,
                            smoke=args.smoke)
    if not ok:
        print(f"  CHAIN FAILED", flush=True)
        with open(os.path.join(outdir, f"e2e_{model}.json"),
                  "w") as f:
            json.dump(results, f, indent=2)
        return 1
    if args.smoke:
        print(f"  SMOKE PASSED ({len(results)} nodes)", flush=True)
        return 0
    # Final check (the chain's vals are rebuilt from the
    # checkpoints + the golden for the CPU nodes -- re-run the
    # chain's vals from the saved checkpoints).
    vals = {"input": fin_u8}
    ckpt_dir = os.path.join(outdir, f"ckpt_{model}")
    done, _rec = load_ckpt(ckpt_dir)
    groups = track_groups(program, per_layer, fin_u8.shape)
    for name in done:
        vals[name] = bytes_to_values(done[name], groups[name])
    if model == "yolo":
        fok, fres = final_yolo(program, vals, fin_u8)
    else:
        fok, fres = final_dnanet(program, vals, fin_u8)
    results["final"] = fres
    with open(os.path.join(outdir, f"e2e_{model}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  === {model}: {'ALL PASS' if fok else 'FINAL FAILED'} "
          f"({time.time() - t0:.0f}s) ===", flush=True)
    return 0 if fok else 1


if __name__ == "__main__":
    raise SystemExit(main())
