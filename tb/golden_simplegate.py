#!/usr/bin/env python3
"""Golden model + vector generation + result check for simplegate (Stage 4).

SimpleGate (NAFNet): input is a single stream of 2*W*H pixels (first W*H
= half A, next W*H = half B). Output is W*H pixels:
  out[i] = (A[i] * B[i]) >> shift, saturated to [0,255] (unsigned out)
  or [-128,127] (signed out, emitted as a two's-complement byte).
If sign_in is set, the 8-bit bytes are two's-complement int8.

Vector file format (vectors_simplegate.txt):
  line 1: ncases
  per case: shift W H sign_in sign_out
            2*W*H input pixels (raster order: half A, then half B)
Output file (out_simplegate.txt): one output pixel per line, raster order.

Usage:
  python3 golden_simplegate.py gen     # write vectors_simplegate.txt
  python3 golden_simplegate.py check   # compare out_simplegate.txt vs golden
"""
import sys

VEC = "vectors_simplegate.txt"
OUT = "out_simplegate.txt"


def img(w, h, seed):
    return [(r * 37 + c * 17 + seed * 11 + 7) % 256
            for r in range(h) for c in range(w)]


def make_cases():
    cases = []

    def add(shift, w, h, sign_in=0, sign_out=0):
        a = img(w, h, 1)
        b = img(w, h, 3)
        cases.append(dict(shift=shift, w=w, h=h,
                          sign_in=sign_in, sign_out=sign_out,
                          inp=a + b, a=a, b=b))

    # 0: shift 8, 16x16, unsigned
    add(8, 16, 16)
    # 1: shift 0 (full product, saturates), 16x16, unsigned
    add(0, 16, 16)
    # 2: shift 4, odd 15x13, unsigned
    add(4, 15, 13)
    # 3: shift 8, small 4x4, unsigned
    add(8, 4, 4)
    # 4: shift 8, signed in + signed out, 16x16 (~50% negative bytes)
    add(8, 16, 16, sign_in=1, sign_out=1)
    # 5: shift 4, signed in + signed out, odd 15x13
    add(4, 15, 13, sign_in=1, sign_out=1)
    return cases


def golden(cs):
    out = []
    for av, bv in zip(cs["a"], cs["b"]):
        if cs["sign_in"]:
            if av >= 128:
                av -= 256
            if bv >= 128:
                bv -= 256
        v = (av * bv) >> cs["shift"]
        if cs["sign_out"]:
            v = max(-128, min(127, v)) & 0xFF
        else:
            v = max(0, min(255, v))
        out.append(v)
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['shift']} {cs['w']} {cs['h']} "
                    f"{cs['sign_in']} {cs['sign_out']}\n")
            f.write(" ".join(str(v) for v in cs["inp"]) + "\n")
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
        bad = [(i, e, g) for i, (e, g) in enumerate(zip(exp, seg)) if e != g]
        if bad:
            nerr += len(bad)
            print(f"case {idx} (shift={cs['shift']} {cs['w']}x{cs['h']} "
                  f"sign=({cs['sign_in']},{cs['sign_out']})): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                print(f"  pixel {i}: expected {e}, got {g}")
        else:
            print(f"case {idx} (shift={cs['shift']} {cs['w']}x{cs['h']} "
                  f"sign=({cs['sign_in']},{cs['sign_out']})): "
                  f"PASS ({n} pixels)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    elif len(got) < pos:
        print(f"WARNING: {pos - len(got)} outputs missing from {OUT} (already counted)")
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: golden_simplegate.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_simplegate.py [gen|check]")
        sys.exit(2)
