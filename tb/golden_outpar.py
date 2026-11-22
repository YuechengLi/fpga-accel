#!/usr/bin/env python3
"""Golden model + vector generation + result check for conv2d_outpar (Stage 7b,
SRCNN L1 engine: one input channel, P_OUT=16 output channels per cycle,
NGROUP = C_out/P_OUT groups per output position).

Math (identical to the RTL):
  padded image P_W = W + pad_l + pad_r, P_H = H + pad_t + pad_b (zero fill).
  out_W = (P_W - K)//S_x + 1, out_H = (P_H - K)//S_y + 1.
  For each output channel o and position (or, oc):
    acc = sum over the KxK kernel of
          padded[or*S_y + i][oc*S_x + j] * coeff[o][i*MAX_K + j]
    acc += bias[o]
    rnd = acc + (1 << (shift-1)) if shift > 0 else acc   (round-half-up)
    v = rnd >> shift                                     (arithmetic)
    act: 0 none, 1 ReLU (v<0 -> 0), 2 Leaky (v<0 -> (v*slope)>>16, Q16)
    saturate: unsigned -> [0,255]; signed -> [-128,127].
  Coefficients are stored in a 9x9 grid (row-major, stride 9); the KxK
  kernel is the top-left KxK of that grid (same convention as the RTL,
  which indexes coeff[li*MAX_K+lj] and win[MAX_K-K+li][MAX_K-K+lj]).
  Input pixels are uint8; with sign_in=1 they are interpreted as int8.

Vector file format (vectors_outpar.txt):
  line 1: ncases
  per case: W H K shift pad_t pad_b pad_l pad_r stride_x stride_y
            sign_in sign_out act slope cout
            cout lines, each 81 coeff values (9x9 grid, row-major)
            1 line of cout bias values (int32)
            W*H image pixels (raster order)
Output file (out_outpar.txt): one output channel value per line, in the
order the engine emits them: output position raster order, and within a
position group 0..NGROUP-1, channel 0..15 inside a group. I.e. the flat
sequence is (or, oc, o) with o running 0..cout-1.

Usage:
  python3 golden_outpar.py gen     # write vectors_outpar.txt
  python3 golden_outpar.py check   # compare out_outpar.txt vs golden
"""
import json
import os
import sys

import numpy as np

VEC = "vectors_outpar.txt"
OUT = "out_outpar.txt"
MAX_K = 9
P_OUT = 16


def img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.integers(0, 256, size=w * h, dtype=np.int64)
    # a bright block, so the kernel sees a strong edge
    for r in range(h):
        for c in range(w):
            if 4 <= r < 8 and 5 <= c < 9:
                p[r * w + c] = 255
    return [int(v) for v in p]


def grid(kern, k):
    """Place a KxK kernel (row-major, stride k) into the 9x9 grid
    (top-left aligned, matching the RTL: coeff index li*MAX_K+lj, window
    pixel at (MAX_K-K+li, MAX_K-K+lj))."""
    g = [0] * (MAX_K * MAX_K)
    for i in range(k):
        for j in range(k):
            g[i * MAX_K + j] = kern[i * k + j]
    return g


def rand_kernels(cout, k, seed, lo=-256, hi=256):
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(cout):
        kern = [int(v) for v in rng.integers(lo, hi, size=k * k)]
        ks.append(grid(kern, k))
    return ks


def make_cases():
    cases = []

    def add(w, h, k, shift, coeff, bias, act=0, slope=8192, sign_in=0,
            sign_out=0, pt=0, pb=0, pl=0, pr=0, sx=1, sy=1, im=None):
        cases.append(dict(w=w, h=h, k=k, shift=shift, coeff=coeff,
                          bias=bias, act=act, slope=slope,
                          sign_in=sign_in, sign_out=sign_out,
                          pt=pt, pb=pb, pl=pl, pr=pr, sx=sx, sy=sy,
                          img=im if im is not None else img(w, h)))

    # 0: K=9, 16x16, shift 15, 64 random kernels + bias, ReLU (SRCNN-like)
    add(16, 16, 9, 15, rand_kernels(64, 9, 11),
        [int(v) for v in np.random.default_rng(12).integers(
            -200000, 200000, size=64)], act=1)
    # 1: K=9, 16x16, shift 8, box kernel (all 1s), no bias, no act.
    #    16 channels (NGROUP=1): the 64-channel path is covered by
    #    cases 0 and 4; the shared-kernel cases use 16 to keep the
    #    5184-write config load from dominating the sim.
    add(16, 16, 9, 8, [grid([1] * 81, 9)] * 16, [0] * 16)
    # 2: K=5, 20x12, shift 8, Gaussian, symmetric pad 1, stride 1, ReLU
    g5 = [1, 4, 6, 4, 1,
          4, 16, 24, 16, 4,
          6, 24, 36, 24, 6,
          4, 16, 24, 16, 4,
          1, 4, 6, 4, 1]
    add(20, 12, 5, 8, [grid(g5, 5)] * 16, [0] * 16, act=1,
        pt=1, pb=1, pl=1, pr=1)
    # 3: K=3, 16x16, shift 8, Sobel X, pad 1, stride 2, Leaky (slope 0.5)
    add(16, 16, 3, 8, [grid([-1, 0, 1, -2, 0, 2, -1, 0, 1], 3)] * 16,
        [0] * 16, act=2, slope=32768, pt=1, pb=1, pl=1, pr=1, sx=2, sy=2)
    # 4: SRCNN L1 real weights (models/vision/srcnn/srcnn_x3.npz), 16x16, shift 15,
    #    ReLU, unsigned in/out. L1_W_q is (9,9,1,64); tap = r*9+c.
    npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "models", "vision", "srcnn", "srcnn_x3.npz"))
    wq = npz["L1_W_q"][:, :, 0, :]          # (9,9,64)
    bq = npz["L1_b_q"]                       # (64,)
    coeff = []
    for o in range(64):
        coeff.append([int(wq[i, j, o]) for i in range(9) for j in range(9)])
    add(16, 16, 9, 15, coeff, [int(v) for v in bq], act=1)
    # 5: K=9, 16x16, shift 8, random kernels, signed in/out, no act
    #    (16 channels; the 64-channel path is covered by cases 0 and 4)
    add(16, 16, 9, 8, rand_kernels(16, 9, 21),
        [int(v) for v in np.random.default_rng(22).integers(
            -100000, 100000, size=16)], sign_in=1, sign_out=1)
    return cases


def out_size(cs):
    pw = cs["w"] + cs["pl"] + cs["pr"]
    ph = cs["h"] + cs["pt"] + cs["pb"]
    ow = (pw - cs["k"]) // cs["sx"] + 1
    oh = (ph - cs["k"]) // cs["sy"] + 1
    return ow, oh


def golden(cs):
    w, h, k, shift = cs["w"], cs["h"], cs["k"], cs["shift"]
    coeff, bias = cs["coeff"], cs["bias"]
    im = cs["img"]
    pt, pl, sx, sy = cs["pt"], cs["pl"], cs["sx"], cs["sy"]
    ow, oh = out_size(cs)
    cout = len(coeff)
    out = []
    for orow in range(oh):
        for ocol in range(ow):
            for o in range(cout):
                acc = 0
                for i in range(k):
                    for j in range(k):
                        rr = orow * sy + i - pt
                        cc = ocol * sx + j - pl
                        if 0 <= rr < h and 0 <= cc < w:
                            px = im[rr * w + cc]
                            if cs["sign_in"] and px >= 128:
                                px -= 256
                            acc += px * coeff[o][i * MAX_K + j]
                acc += bias[o]
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
            f.write(f"{cs['w']} {cs['h']} {cs['k']} {cs['shift']} "
                    f"{cs['pt']} {cs['pb']} {cs['pl']} {cs['pr']} "
                    f"{cs['sx']} {cs['sy']} {cs['sign_in']} {cs['sign_out']} "
                    f"{cs['act']} {cs['slope']} {len(cs['coeff'])}\n")
            for o in range(len(cs["coeff"])):
                f.write(" ".join(str(v) for v in cs["coeff"][o]) + "\n")
            f.write(" ".join(str(v) for v in cs["bias"]) + "\n")
            f.write(" ".join(str(v) for v in cs["img"]) + "\n")
    print(f"wrote {VEC}: {len(cases)} cases")


def check():
    cases = make_cases()
    got = [int(x) for x in open(OUT).read().split()]
    pos, nerr = 0, 0
    for idx, cs in enumerate(cases):
        exp = golden(cs)
        ow, oh = out_size(cs)
        n = len(exp)
        seg = got[pos:pos + n]
        pos += n
        if len(seg) < n:
            print(f"case {idx}: MISSING outputs: got {len(seg)}, expected {n}")
            nerr += n - len(seg)
            continue
        bad = [(i, e, g) for i, (e, g) in enumerate(zip(exp, seg)) if e != g]
        if bad:
            nerr += len(bad)
            print(f"case {idx} ({cs['w']}x{cs['h']} k={cs['k']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) act={cs['act']} "
                  f"sign=({cs['sign_in']},{cs['sign_out']}) "
                  f"out={ow}x{oh}x{len(cs['coeff'])}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                orow, rem = divmod(i, ow * len(cs["coeff"]))
                ocol, o = divmod(rem, len(cs["coeff"]))
                print(f"  out {i} (r={orow}, c={ocol}, ch={o}): "
                      f"expected {e}, got {g}")
        else:
            print(f"case {idx} ({cs['w']}x{cs['h']} k={cs['k']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) act={cs['act']} "
                  f"sign=({cs['sign_in']},{cs['sign_out']}) "
                  f"out={ow}x{oh}x{len(cs['coeff'])}): PASS ({n} values)")
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
        print("usage: golden_outpar.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_outpar.py [gen|check]")
        sys.exit(2)
