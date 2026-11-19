#!/usr/bin/env python3
"""Golden model + vector generation + result check for pool (Stage 3).

Modes (match rtl/pool.sv):
  0 = MAX 2x2 (stride 2, zero-pad to even W/H on the fly;
      output ceil(W/2) x ceil(H/2))
  1 = AVG 2x2 (sum of the 4 window positions, padding = 0, then >> 2;
      output ceil(W/2) x ceil(H/2))
  2 = GAP (sum of all real pixels >> gap_shift, single output)

Vector file (vectors_pool.txt):
  line 1: ncases
  per case: mode W H gap_shift out_w out_h
            then W*H image pixels (one line)
Output (out_pool.txt): one output pixel per line, raster order.
Usage: golden_pool.py [gen|check]
"""
import sys

VEC = "vectors_pool.txt"
OUT = "out_pool.txt"


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

    def add(mode, w, h, seed, gap_shift=0):
        if mode == 2:
            ow, oh = 1, 1
        else:
            ow, oh = (w + 1) // 2, (h + 1) // 2
        cases.append(dict(mode=mode, w=w, h=h, gap_shift=gap_shift,
                          ow=ow, oh=oh, img=img(w, h, seed)))

    add(0, 16, 16, 1)          # MAX 2x2, even
    add(0, 15, 13, 2)          # MAX 2x2, odd (zero-pad path)
    add(0, 1, 1, 3)            # MAX 2x2, 1x1 (all padding except (0,0))
    add(1, 16, 16, 4)          # AVG 2x2, even
    add(1, 15, 13, 5)          # AVG 2x2, odd
    add(2, 16, 16, 6, 8)       # GAP, 16x16 = 256 pixels, gap_shift 8 = mean
    add(2, 15, 13, 7, 7)       # GAP, 15x13 = 195 pixels, gap_shift 7
    add(2, 1, 1, 8, 0)         # GAP, single pixel, gap_shift 0
    return cases


def golden(cs):
    mode, w, h = cs["mode"], cs["w"], cs["h"]
    im = cs["img"]
    if mode == 2:
        return [sum(im) >> cs["gap_shift"]]
    out = []
    for r in range(cs["oh"]):
        for c in range(cs["ow"]):
            vals = []
            for dr in (0, 1):
                for dc in (0, 1):
                    rr, cc = 2 * r + dr, 2 * c + dc
                    vals.append(im[rr * w + cc] if (rr < h and cc < w) else 0)
            if mode == 0:
                out.append(max(vals))
            else:
                out.append(sum(vals) >> 2)
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['mode']} {cs['w']} {cs['h']} {cs['gap_shift']} "
                    f"{cs['ow']} {cs['oh']}\n")
            f.write(" ".join(str(v) for v in cs["img"]) + "\n")
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
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']} "
                  f"out={cs['ow']}x{cs['oh']}): {bad}/{n} mismatches")
        else:
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']} "
                  f"out={cs['ow']}x{cs['oh']}): PASS ({n} pixels)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gen", "check"):
        print("usage: golden_pool.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    else:
        sys.exit(check())
