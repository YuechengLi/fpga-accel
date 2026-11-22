#!/usr/bin/env python3
"""Golden model + vector generation + result check for conv2d_tappar
(Stage 7b, SRCNN L3 engine: tap-parallel 5x5, 32 -> 1; one tap-row
(5 taps x 32 channels = 160 mults) per cycle, accumulating over the
5 tap-rows, so one output position costs 5 cycles).

Math (identical to the RTL):
  For each output position (or, oc) (valid conv with zero-fill padding):
    acc = sum over t=0..K-1, j=0..K-1, c=0..31 of
          frame[or*sy + t - pad_t][oc*sx + j - pad_l][c] * coeff[t][j][c]
          + bias
    (out-of-bounds frame positions contribute 0)
    rnd = acc + (1 << (shift-1)) if shift > 0 else acc  (round-half-up)
    v = rnd >> shift                                     (arithmetic)
    act: 0 none, 1 ReLU (v<0 -> 0), 2 Leaky (v<0 -> (v*slope)>>16, Q16)
    saturate: unsigned -> [0,255]; signed -> [-128,127].
  Frame bytes are uint8; with sign_in=1 they are int8.
  Output order: raster over output positions (or, oc).

Vector file format (vectors_tappar.txt):
  line 1: ncases
  per case: K shift W H pad_t pad_b pad_l pad_r stride_x stride_y
            sign_in sign_out act slope out_w out_h
            K*K lines, each 32 coeff values (tap t*K+j, c=0..31)
            1 line with the single bias value (int32)
            H*W lines, each 32 input-frame bytes (raster order)
Output file (out_tappar.txt): one output byte per line, raster order.

Usage:
  python3 golden_tappar.py gen     # write vectors_tappar.txt
  python3 golden_tappar.py check   # compare out_tappar.txt vs golden
"""
import os
import sys

import numpy as np

VEC = "vectors_tappar.txt"
OUT = "out_tappar.txt"
P_IN = 32


def rand_frame(W, H, seed):
    rng = np.random.default_rng(seed)
    fr = rng.integers(0, 256, size=(H, W, P_IN), dtype=np.int64)
    # a bright block so the kernel sees a strong edge
    fr[0:2, 0:2, 4:8] = 255
    return [[int(v) for v in fr[r, c]] for r in range(H) for c in range(W)]


def make_cases():
    cases = []

    def add(K, shift, W, H, coeff, bias, frame, pad=(0, 0, 0, 0),
            stride=(1, 1), act=0, slope=8192, sign_in=0, sign_out=0):
        pad_t, pad_b, pad_l, pad_r = pad
        sx, sy = stride
        out_w = (W + pad_l + pad_r - K) // sx + 1
        out_h = (H + pad_t + pad_b - K) // sy + 1
        cases.append(dict(K=K, shift=shift, W=W, H=H, coeff=coeff,
                          bias=bias, frame=frame, pad_t=pad_t,
                          pad_b=pad_b, pad_l=pad_l, pad_r=pad_r,
                          stride_x=sx, stride_y=sy, act=act,
                          slope=slope, sign_in=sign_in,
                          sign_out=sign_out, out_w=out_w, out_h=out_h))

    # 0: K=5, 8x8, shift 15, ReLU, random coeffs/bias (SRCNN-L3-like)
    rng = np.random.default_rng(41)
    coeff = [[[int(v) for v in rng.integers(-512, 512, size=P_IN)]
              for _ in range(5)] for _ in range(5)]
    bias = int(rng.integers(-200000, 200000))
    add(5, 15, 8, 8, coeff, bias, rand_frame(8, 8, 42), act=1)
    # 1: SRCNN L3 real weights (models/vision/srcnn/srcnn_x3.npz), shift 13, ReLU.
    #    L3_W_q is (5,5,32,1); L3_b_q is (1,).
    npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "models", "vision", "srcnn", "srcnn_x3.npz"))
    wq = npz["L3_W_q"][:, :, :, 0]           # (5, 5, 32)
    bq = npz["L3_b_q"]                       # (1,)
    coeff = [[[int(wq[t, j, c]) for c in range(P_IN)]
              for j in range(5)] for t in range(5)]
    add(5, 13, 8, 8, coeff, int(bq[0]), rand_frame(8, 8, 43), act=1)
    # 2: K=3, 10x10, pad 1, stride 1, shift 8, no act, unsigned.
    rng = np.random.default_rng(44)
    coeff = [[[int(v) for v in rng.integers(-512, 512, size=P_IN)]
              for _ in range(3)] for _ in range(3)]
    bias = int(rng.integers(-200000, 200000))
    add(3, 8, 10, 10, coeff, bias, rand_frame(10, 10, 45),
        pad=(1, 1, 1, 1))
    # 3: K=5, 8x8, shift 8, Leaky slope 0.5, signed in/out.
    rng = np.random.default_rng(46)
    coeff = [[[int(v) for v in rng.integers(-512, 512, size=P_IN)]
              for _ in range(5)] for _ in range(5)]
    bias = int(rng.integers(-200000, 200000))
    add(5, 8, 8, 8, coeff, bias, rand_frame(8, 8, 47), act=2,
        slope=32768, sign_in=1, sign_out=1)
    return cases


def golden(cs):
    K = cs["K"]
    W, H = cs["W"], cs["H"]
    pad_t, pad_l = cs["pad_t"], cs["pad_l"]
    sx, sy = cs["stride_x"], cs["stride_y"]
    shift = cs["shift"]
    out = []
    frame = cs["frame"]
    for or_ in range(cs["out_h"]):
        for oc in range(cs["out_w"]):
            acc = 0
            for t in range(K):
                for j in range(K):
                    r = or_ * sy + t - pad_t
                    c = oc * sx + j - pad_l
                    if 0 <= r < H and 0 <= c < W:
                        px = frame[r * W + c]
                        for ch in range(P_IN):
                            v = px[ch]
                            if cs["sign_in"] and v >= 128:
                                v -= 256
                            acc += v * cs["coeff"][t][j][ch]
            acc += cs["bias"]
            rnd = acc + (1 << (shift - 1)) if shift > 0 else acc
            v = rnd >> shift
            if cs["act"] == 1 and v < 0:
                v = 0
            elif cs["act"] == 2 and v < 0:
                v = (v * cs["slope"]) >> 16
            if cs["sign_out"]:
                v = max(-128, min(127, v))
            else:
                v = max(0, min(255, v))
            out.append(v)
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['K']} {cs['shift']} {cs['W']} {cs['H']} "
                    f"{cs['pad_t']} {cs['pad_b']} {cs['pad_l']} {cs['pad_r']} "
                    f"{cs['stride_x']} {cs['stride_y']} {cs['sign_in']} "
                    f"{cs['sign_out']} {cs['act']} {cs['slope']} "
                    f"{cs['out_w']} {cs['out_h']}\n")
            for t in range(cs["K"]):
                for j in range(cs["K"]):
                    f.write(" ".join(str(v) for v in cs["coeff"][t][j]) + "\n")
            f.write(f"{cs['bias']}\n")
            for vec in cs["frame"]:
                f.write(" ".join(str(v) for v in vec) + "\n")
    print(f"wrote {VEC}: {len(cases)} cases")


def check():
    cases = make_cases()
    got = [int(x) for x in open(OUT).read().split()]
    pos, nerr = 0, 0
    for idx, cs in enumerate(cases):
        exp = golden(cs)
        n = len(exp)
        seg = got[pos:pos + n]
        pos += n
        if len(seg) < n:
            print(f"case {idx}: MISSING outputs: got {len(seg)}, expected {n}")
            nerr += n - len(seg)
            continue
        bad = []
        for i, (e, g) in enumerate(zip(exp, seg)):
            # The TB writes each output byte as an unsigned decimal
            # (0..255). For a signed output, convert back to int8.
            if cs["sign_out"] and g >= 128:
                g -= 256
            if e != g:
                bad.append((i, e, g))
        if bad:
            nerr += len(bad)
            print(f"case {idx} (K={cs['K']} shift={cs['shift']} "
                  f"act={cs['act']} sign=({cs['sign_in']},{cs['sign_out']}) "
                  f"{cs['W']}x{cs['H']}->{cs['out_w']}x{cs['out_h']}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                p, rem = divmod(i, cs["out_w"])
                print(f"  out {i} (r={p}, c={rem}): expected {e}, got {g}")
        else:
            print(f"case {idx} (K={cs['K']} shift={cs['shift']} "
                  f"act={cs['act']} sign=({cs['sign_in']},{cs['sign_out']}) "
                  f"{cs['W']}x{cs['H']}->{cs['out_w']}x{cs['out_h']}): "
                  f"PASS ({n} values)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    elif len(got) < pos:
        print(f"WARNING: {pos - len(got)} outputs missing from {OUT} "
              f"(already counted)")
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: golden_tappar.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_tappar.py [gen|check]")
        sys.exit(2)
