#!/usr/bin/env python3
"""Golden model + vector generation + result check for concat (Stage 4).

Two input streams A and B (same W/H), one output stream.
  mode 0 (sequential): out = A frame followed by B frame ((C,H,W) order).
    Output is 2*W*H pixels, out_eol at the end of each W-pixel row.
  mode 1 (interleave): out = A and B interleaved per pixel (H,W,C order).
    Output is 2*W*H pixels, out_eol at the end of each 2W-pixel row.

Vector file format (vectors_concat.txt):
  line 1: ncases
  per case: mode W H
            W*H pixels of A (raster order)
            W*H pixels of B (raster order)
Output file (out_concat.txt): one output pixel per line, raster order.

Usage:
  python3 golden_concat.py gen     # write vectors_concat.txt
  python3 golden_concat.py check   # compare out_concat.txt vs golden
"""
import sys

VEC = "vectors_concat.txt"
OUT = "out_concat.txt"


def img_a(w, h):
    return [(r * 37 + c * 17 + 11) % 256 for r in range(h) for c in range(w)]


def img_b(w, h):
    return [(r * 53 + c * 29 + 7) % 256 for r in range(h) for c in range(w)]


def make_cases():
    cases = []

    def add(mode, w, h):
        cases.append(dict(mode=mode, w=w, h=h,
                          a=img_a(w, h), b=img_b(w, h)))

    # 0: sequential, 16x16
    add(0, 16, 16)
    # 1: sequential, odd 15x13
    add(0, 15, 13)
    # 2: sequential, small 4x4
    add(0, 4, 4)
    # 3: interleave, 16x16
    add(1, 16, 16)
    # 4: interleave, odd 15x13
    add(1, 15, 13)
    # 5: interleave, small 4x4
    add(1, 4, 4)
    return cases


def golden(cs):
    if cs["mode"] == 0:
        return cs["a"] + cs["b"]
    out = []
    for av, bv in zip(cs["a"], cs["b"]):
        out.append(av)
        out.append(bv)
    return out


def out_row_len(cs):
    return cs["w"] * 2 if cs["mode"] == 1 else cs["w"]


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
        bad = [(i, e, g) for i, (e, g) in enumerate(zip(exp, seg)) if e != g]
        if bad:
            nerr += len(bad)
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                print(f"  pixel {i}: expected {e}, got {g}")
        else:
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']}): "
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
        print("usage: golden_concat.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_concat.py [gen|check]")
        sys.exit(2)
