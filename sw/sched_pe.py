#!/usr/bin/env python3
"""Scheduler for the PE array (Stage 7b) -- also the compiler scheduling
pass (sw/): given a layer, its input feature map, its weights, and a
device config, it decides the mode, TM/TN/TK, frame-tile size, and
K-split, and emits the tile-step program (one tile-step per
(M-tile, N-chunk, K-chunk)) that the pe_array dispatcher executes.

GEMM lowering (implicit im2col): M = output positions x output
channels, K-dim = C_in x K x K. The K-dim order is
k = (ci*K + kr)*K + kc (ci-major, then kr, then kc); this matches the
K^2 engines' tap order (tap = r*K+c for C_in=1) and the weight array
layout W_q[kr][kc][ci][co].

Tile-step = one (M-tile, N-chunk, K-chunk):
  - M-tile: 4x4 output positions (TM=16), raster order m = dr*4+dc.
  - N-chunk: TN output channels (outer) or tn channels (dot; tn=1 for
    an N=1 layer such as SRCNN L3).
  - K-chunk: TK reduction steps (TK = largest divisor of the K-dim
    that is <= 32, the register-file bound). Chunks chain inside the
    PE (cont=1); the first chunk has first_k=1 (bias), the last has
    last_k=1 (emit bytes).
  - A tile: A[m][kk] = inp[or+kr][oc+kc][ci] (im2col gather).
  - B tile: B[kk][n] = W_q[kr][kc][ci][co] (co = nchunk*TN + n).

The scheduler is device-driven (sw/devices.py): it reads the device
config (N_PE, TM, TN, TK_MAX) and emits the tile-steps for that
device. The same scheduler serves the ZU19EG (1080p30) and the ECP5
(small resolutions); only the device config changes.

Usage (as a module):
  steps, shape = gen_layer_steps(layer, inp, W_q, b_q, dev)
  write_layer_vectors(steps, path)
  out = assemble_output(outputs, shape)   # outputs = [(mtile, nchunk, bytes)]

Usage (driver): python3 sw/run_pe_srcnn.py  (runs the 3 SRCNN layers
through the pe_array TB, one wave per layer, RTL output feeding the
next layer -- the Stage 6 pattern).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from devices import get as get_device


def largest_divisor(kdim, bound=32):
    """Largest divisor of kdim that is <= bound (register-file bound)."""
    best = 1
    for d in range(1, min(bound, kdim) + 1):
        if kdim % d == 0:
            best = d
    return best


def k_to_krkc_ci(k, K, C_in):
    """K-dim index -> (kr, kc, ci), k = (ci*K + kr)*K + kc."""
    ci = k // (K * K)
    rem = k % (K * K)
    kr = rem // K
    kc = rem % K
    return kr, kc, ci


def gen_layer_steps(layer, inp, W_q, b_q, dev):
    """Generate the tile-step list for one conv layer.

    layer: dict(K, C_out, mode_dot, sign_in, sign_out, shift, act,
                slope).  inp: HxWxC_in uint8.  W_q: (K,K,C_in,C_out)
    int16.  b_q: (C_out,) int32.  dev: device config dict.
    Returns (steps, (H_out, W_out, C_out)).
    """
    K = layer["K"]
    H, W, C_in = inp.shape
    C_out = layer["C_out"]
    H_out = H - K + 1
    W_out = W - K + 1
    TM = dev["tm"]
    TN = dev["tn"]
    TK_MAX = 32
    # M-tile is 4x4 output positions (TM=16); require divisibility.
    assert TM == 16, "M-tile is 4x4 (TM=16)"
    assert H_out % 4 == 0 and W_out % 4 == 0, \
        f"output {H_out}x{W_out} not divisible by 4 (M-tile)"
    Kdim = C_in * K * K
    tk = largest_divisor(Kdim, TK_MAX)
    n_kchunk = Kdim // tk
    mode_dot = layer["mode_dot"]
    # Dot mode is only correct for TN=8: the PE's dot reduction tree
    # sums 8 products per row (pe_mac.sv gen_rowsum reads
    # prod[gr*TN+0..7]) and each (n,g) step processes 8 taps, so with
    # TN<8 it mixes adjacent rows, drops half the K-dim, and reads
    # prod out of bounds for the last row. Force outer mode there;
    # outer mode is TN-generic (the TM x TN grid works for any TN).
    # Found 2026-11-21: ecp5u85 (TN=4) emitted L3 as dot before this
    # guard; zu19eg (TN=8) is unaffected.
    if mode_dot and TN < 8:
        mode_dot = 0
    # N-chunks: dot -> C_out/tn (exact; tn=1 for an N=1 layer).
    # Outer -> ceil(C_out/TN); the last chunk is zero-padded for
    # C_out < TN (e.g. L3 C_out=1 on the ECP5 TN=4 config) -- the B
    # and bias loops below already guard co < C_out, and
    # assemble_output ignores co >= C_out.
    tn = layer.get("tn", TN) if mode_dot else TN
    if mode_dot:
        assert C_out % tn == 0, f"C_out {C_out} not divisible by tn {tn}"
        n_nchunk = C_out // tn
    else:
        n_nchunk = -(-C_out // tn)
    n_mtile_r = H_out // 4
    n_mtile_c = W_out // 4

    steps = []
    for mtr in range(n_mtile_r):
        for mtc in range(n_mtile_c):
            mtile = mtr * n_mtile_c + mtc
            for nchunk in range(n_nchunk):
                # Bias for this N-chunk (TN entries; dot mode uses n=0).
                co0 = nchunk * tn
                bias = [int(b_q[co0 + n]) if co0 + n < C_out else 0
                        for n in range(TN)]
                for kchunk in range(n_kchunk):
                    k0 = kchunk * tk
                    # A tile: TM x tk (m = dr*4+dc, kk = k-k0).
                    A = [[0] * tk for _ in range(TM)]
                    for dr in range(4):
                        for dc in range(4):
                            m = dr * 4 + dc
                            orow = mtr * 4 + dr
                            ocol = mtc * 4 + dc
                            for kk in range(tk):
                                k = k0 + kk
                                kr, kc, ci = k_to_krkc_ci(k, K, C_in)
                                A[m][kk] = int(inp[orow + kr, ocol + kc, ci])
                    # B tile: tk x TN (kk, n). co = nchunk*tn + n.
                    B = [[0] * TN for _ in range(tk)]
                    for kk in range(tk):
                        k = k0 + kk
                        kr, kc, ci = k_to_krkc_ci(k, K, C_in)
                        for n in range(TN):
                            co = co0 + n
                            if co < C_out:
                                B[kk][n] = int(W_q[kr, kc, ci, co])
                    steps.append(dict(
                        mode_dot=mode_dot, tk=tk, tn=tn,
                        first_k=1 if kchunk == 0 else 0,
                        sign_in=layer["sign_in"],
                        shift=layer["shift"], act=layer["act"],
                        slope=layer["slope"],
                        sign_out=layer["sign_out"],
                        bias=bias, A=A, B=B,
                        mtile=mtile, nchunk=nchunk,
                        kchunk=kchunk, n_kchunk=n_kchunk,
                    ))
    return steps, (H_out, W_out, C_out)


def write_layer_vectors(steps, shape, path):
    """Write one layer's tile-steps to a vectors file for tb_pe_array.

    Format:
      line 1: nsteps TM TN TK_MAX
      line 2: H_out W_out C_out n_out (n_out = # (mtile, nchunk) outputs)
      per step: mode_dot tk tn first_k sign_in shift act slope sign_out
                mtile nchunk n_kchunk
                bias (TN int32, one line)
                A (tk lines, TM values each, k-major)
                B (tk lines, TN values each, k-major)
    """
    TM = 16
    TN = 8
    H_out, W_out, C_out = shape
    # n_out = number of distinct (mtile, nchunk) outputs. (Do NOT use
    # C_out // TN: for dot mode (L3) the N-chunk count is C_out // tn
    # with tn=1, so C_out // TN would give 0 and the TB would expect
    # no outputs.)
    n_out = len(set((s["mtile"], s["nchunk"]) for s in steps))
    with open(path, "w") as f:
        f.write(f"{len(steps)} {TM} {TN} 32\n")
        f.write(f"{H_out} {W_out} {C_out} {n_out}\n")
        for s in steps:
            f.write(f"{s['mode_dot']} {s['tk']} {s['tn']} "
                    f"{s['first_k']} {s['sign_in']} {s['shift']} "
                    f"{s['act']} {s['slope']} {s['sign_out']} "
                    f"{s['mtile']} {s['nchunk']} {s['n_kchunk']}\n")
            f.write(" ".join(str(v) for v in s["bias"]) + "\n")
            for kk in range(s["tk"]):
                f.write(" ".join(str(s["A"][m][kk]) for m in range(TM)) + "\n")
            for kk in range(s["tk"]):
                f.write(" ".join(str(s["B"][kk][n]) for n in range(TN)) + "\n")
    return len(steps)


def assemble_output(outputs, shape, TN=8):
    """Assemble per-(mtile, nchunk) output byte blocks into HxWxC_out.

    outputs: list of (mtile, nchunk, bytes) where bytes is TM*TN in
    (m, n) order (m = dr*4+dc, n = output-channel offset within the
    chunk). shape: (H_out, W_out, C_out). Returns a uint8 array.
    """
    H_out, W_out, C_out = shape
    out = np.zeros((H_out, W_out, C_out), dtype=np.uint8)
    n_mtile_c = W_out // 4
    for (mtile, nchunk, byts) in outputs:
        mtr = mtile // n_mtile_c
        mtc = mtile % n_mtile_c
        for m in range(16):
            dr, dc = m // 4, m % 4
            orow, ocol = mtr * 4 + dr, mtc * 4 + dc
            for n in range(TN):
                co = nchunk * TN + n
                if co < C_out:
                    out[orow, ocol, co] = byts[m * TN + n]
    return out


def load_srcnn():
    """Load SRCNN weights + shifts. Returns (layers, quant)."""
    base = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(base, "..", "models", "srcnn", "srcnn_x3.npz"))
    with open(os.path.join(base, "..", "models", "srcnn",
                           "srcnn_x3_quant.json")) as f:
        q = json.load(f)
    layers = [
        dict(name="L1", K=9, C_out=64, mode_dot=0, sign_in=0,
             sign_out=0, shift=q["L1"]["shift"], act=1, slope=0,
             W_q=d["L1_W_q"], b_q=d["L1_b_q"]),
        dict(name="L2", K=1, C_out=32, mode_dot=0, sign_in=0,
             sign_out=0, shift=q["L2"]["shift"], act=1, slope=0,
             W_q=d["L2_W_q"], b_q=d["L2_b_q"]),
        dict(name="L3", K=5, C_out=1, mode_dot=1, tn=1, sign_in=0,
             sign_out=0, shift=q["L3"]["shift"], act=1, slope=0,
             W_q=d["L3_W_q"], b_q=d["L3_b_q"]),
    ]
    return layers, q


if __name__ == "__main__":
    # Smoke test: generate steps per layer (dummy inputs, right shapes)
    # and report counts.
    from stage6 import build_srcnn_input
    layers, _ = load_srcnn()
    dev = get_device("zu19eg")
    inp_u8, _ = build_srcnn_input(24)
    dummy = {
        "L1": inp_u8,
        "L2": np.zeros((16, 16, 64), dtype=np.uint8),
        "L3": np.zeros((16, 16, 32), dtype=np.uint8),
    }
    for L in layers:
        steps, shape = gen_layer_steps(L, dummy[L["name"]],
                                       L["W_q"], L["b_q"], dev)
        print(f"{L['name']}: {len(steps)} steps, out {shape}, "
              f"tk={steps[0]['tk']}, mode_dot={L['mode_dot']}")
