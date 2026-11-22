#!/usr/bin/env python3
"""Golden model + vector generation + result check for conv1x1_cp (Stage 7b,
SRCNN L2 engine: P_OUT=7 output channels x P_IN=64 input channels per
cycle, NGROUP=5 groups per input position; the last group has only 4
valid outputs, 32 = 4*7 + 4).

Math (identical to the RTL):
  For each input position (a 64-byte vector) and each output channel o:
    acc = sum over c=0..63 of in_vec[c] * coeff[o][c] + bias[o]
    rnd = acc + (1 << (shift-1)) if shift > 0 else acc   (round-half-up)
    v = rnd >> shift                                     (arithmetic)
    act: 0 none, 1 ReLU (v<0 -> 0), 2 Leaky (v<0 -> (v*slope)>>16, Q16)
    saturate: unsigned -> [0,255]; signed -> [-128,127].
  Input vector bytes are uint8; with sign_in=1 they are int8.
  Output order: position-major, group 0..4, channel 0..6 within a group
  (the last group's channels 4..6 are masked off by the RTL and are not
  part of the golden output).

Vector file format (vectors_cp.txt):
  line 1: ncases
  per case: shift sign_in sign_out act slope npos
            32 lines, each 64 coeff values (o=0..31, c=0..63)
            1 line of 32 bias values (int32)
            npos lines, each 64 input-vector bytes
Output file (out_cp.txt): one output channel value per line, in the order
the engine emits them: position, group, channel (only the valid channels
of the last group are written by the TB; the golden omits them too).

Usage:
  python3 golden_cp.py gen     # write vectors_cp.txt
  python3 golden_cp.py check   # compare out_cp.txt vs golden
"""
import os
import sys

import numpy as np

VEC = "vectors_cp.txt"
OUT = "out_cp.txt"
P_IN = 64
C_OUT = 32
P_OUT = 7
NGROUP = 5


def rand_vecs(npos, seed):
    rng = np.random.default_rng(seed)
    vecs = rng.integers(0, 256, size=(npos, P_IN), dtype=np.int64)
    # a bright block in the first vector, so the kernel sees a strong edge
    vecs[0, 4:8] = 255
    return [[int(v) for v in row] for row in vecs]


def make_cases():
    cases = []

    def add(shift, coeff, bias, vecs, act=0, slope=8192, sign_in=0,
            sign_out=0):
        cases.append(dict(shift=shift, coeff=coeff, bias=bias, vecs=vecs,
                          act=act, slope=slope, sign_in=sign_in,
                          sign_out=sign_out))

    # 0: random coeffs/bias, shift 15, ReLU, unsigned (SRCNN-L2-like)
    rng = np.random.default_rng(31)
    coeff = [[int(v) for v in rng.integers(-512, 512, size=P_IN)]
             for _ in range(C_OUT)]
    bias = [int(v) for v in rng.integers(-200000, 200000, size=C_OUT)]
    add(15, coeff, bias, rand_vecs(8, 32), act=1)
    # 1: SRCNN L2 real weights (models/vision/srcnn/srcnn_x3.npz), shift 15, ReLU.
    #    L2_W_q is (1,1,64,32); L2_b_q is (32,).
    npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "models", "vision", "srcnn", "srcnn_x3.npz"))
    wq = npz["L2_W_q"][0, 0, :, :]          # (64, 32)
    bq = npz["L2_b_q"]                       # (32,)
    coeff = [[int(wq[c, o]) for c in range(P_IN)] for o in range(C_OUT)]
    add(15, coeff, [int(v) for v in bq], rand_vecs(8, 33), act=1)
    # 2: diagonal identity (coeff[o][o]=256), shift 8, no act, unsigned.
    #    Output o of a position is the input channel o (clamped).
    coeff = [[256 if c == o else 0 for c in range(P_IN)]
             for o in range(C_OUT)]
    add(8, coeff, [0] * C_OUT, rand_vecs(8, 34))
    # 3: random coeffs/bias, shift 8, Leaky (slope 0.5), signed in/out.
    rng = np.random.default_rng(35)
    coeff = [[int(v) for v in rng.integers(-512, 512, size=P_IN)]
             for _ in range(C_OUT)]
    bias = [int(v) for v in rng.integers(-100000, 100000, size=C_OUT)]
    add(8, coeff, bias, rand_vecs(8, 36), act=2, slope=32768,
        sign_in=1, sign_out=1)
    return cases


def golden(cs):
    shift = cs["shift"]
    coeff, bias, vecs = cs["coeff"], cs["bias"], cs["vecs"]
    out = []
    for vec in vecs:
        for g in range(NGROUP):
            for u in range(P_OUT):
                o = g * P_OUT + u
                if o >= C_OUT:
                    continue
                acc = 0
                for c in range(P_IN):
                    px = vec[c]
                    if cs["sign_in"] and px >= 128:
                        px -= 256
                    acc += px * coeff[o][c]
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
            f.write(f"{cs['shift']} {cs['sign_in']} {cs['sign_out']} "
                    f"{cs['act']} {cs['slope']} {len(cs['vecs'])}\n")
            for o in range(C_OUT):
                f.write(" ".join(str(v) for v in cs["coeff"][o]) + "\n")
            f.write(" ".join(str(v) for v in cs["bias"]) + "\n")
            for vec in cs["vecs"]:
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
            # (0..255). For a signed output, convert back to int8
            # (-128..127) before comparing.
            if cs['sign_out'] and g >= 128:
                g -= 256
            if e != g:
                bad.append((i, e, g))
        if bad:
            nerr += len(bad)
            print(f"case {idx} (shift={cs['shift']} act={cs['act']} "
                  f"sign=({cs['sign_in']},{cs['sign_out']}) "
                  f"npos={len(cs['vecs'])}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                p, rem = divmod(i, C_OUT)
                print(f"  out {i} (pos={p}, ch={rem}): expected {e}, got {g}")
        else:
            print(f"case {idx} (shift={cs['shift']} act={cs['act']} "
                  f"sign=({cs['sign_in']},{cs['sign_out']}) "
                  f"npos={len(cs['vecs'])}): PASS ({n} values)")
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
        print("usage: golden_cp.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_cp.py [gen|check]")
        sys.exit(2)
