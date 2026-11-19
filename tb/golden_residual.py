#!/usr/bin/env python3
"""Golden model + vector generation + result check for residual (Stage 3).

Modes (match rtl/residual.sv):
  0 = add (a + b, saturating to [0, 255])
  1 = sub (a - b, saturating to [0, 255])
  2 = bypass (a)

Vector file (vectors_residual.txt):
  line 1: ncases
  per case: mode W H
            then W*H pixels for stream A (one line)
            then W*H pixels for stream B (one line)
Output (out_residual.txt): one output pixel per line, raster order.
Usage: golden_residual.py [gen|check]
"""
import sys

VEC = "vectors_residual.txt"
OUT = "out_residual.txt"


def img(w, h, seed):
    p = []
    for r in range(h):
        for c in range(w):
            v = (r * 37 + c * 17 + seed * 11 + 7) % 256
            if 3 <= r < 6 and 4 <= c < 7:
                v = 200
            p.append(v)
    return p


def make_cases():
    cases = []

    def add(mode, w, h, seed_a, seed_b):
        cases.append(dict(mode=mode, w=w, h=h,
                          a=img(w, h, seed_a), b=img(w, h, seed_b)))

    add(0, 16, 16, 1, 2)     # add, even
    add(0, 15, 13, 3, 4)     # add, odd
    add(0, 16, 16, 5, 5)     # add, identical streams (a+b can saturate)
    add(1, 16, 16, 6, 7)     # sub, even
    add(1, 15, 13, 8, 9)     # sub, odd (a-b can go negative -> saturate)
    add(2, 16, 16, 10, 11)   # bypass A
    add(0, 1, 1, 12, 13)     # add, single pixel
    return cases


def golden(cs):
    mode = cs["mode"]
    out = []
    for a, b in zip(cs["a"], cs["b"]):
        if mode == 0:
            out.append(max(0, min(255, a + b)))
        elif mode == 1:
            out.append(max(0, min(255, a - b)))
        else:
            out.append(a)
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['mode']} {cs['w']} {cs['h']}\n")
            f.write(" ".join(str(v) for v in cs["a"]) + "\n")
            f.write(" ".join(str(v) for v in cs["b"]) + "\n")
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
        bad = sum(1 for e, g in zip(exp, seg) if e != g)
        if bad:
            nerr += bad
            for i, (e, g) in enumerate(zip(exp, seg)):
                if e != g and bad <= 3:
                    print(f"case {idx} pixel {i}: expected {e}, got {g}")
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']}): "
                  f"{bad}/{n} mismatches")
        else:
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']}): "
                  f"PASS ({n} pixels)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gen", "check"):
        print("usage: golden_residual.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    else:
        sys.exit(check())
