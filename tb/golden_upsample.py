#!/usr/bin/env python3
"""Golden model + vector generation + result check for upsample (Stage 4).

Scale factor 2, three modes:
  mode 0 (nearest): out[2r+i, 2c+j] = in[r,c] (replicate 2x2).
  mode 1 (bilinear):
      out[2r,2c]     = in[r,c]
      out[2r,2c+1]   = (in[r,c] + in[r,c+1]) / 2
      out[2r+1,2c]   = (in[r,c] + in[r+1,c]) / 2
      out[2r+1,2c+1] = (in[r,c]+in[r,c+1]+in[r+1,c]+in[r+1,c+1]) / 4
    Edges replicate (in[r,W] = in[r,W-1], in[H,c] = in[H-1,c]).
    Division is round-half-up: (a+b+1)>>1 and (a+b+c+d+2)>>2.
  mode 2 (pixel-shuffle, r=2): input is W x H positions x 4 channels in
    (H,W,C) order (4 consecutive pixels per position). Output is 2W x 2H:
      out[2r,2c]=ch0, out[2r,2c+1]=ch1, out[2r+1,2c]=ch2, out[2r+1,2c+1]=ch3.

Vector file format (vectors_upsample.txt):
  line 1: ncases
  per case: mode W H
            input pixels (raster order; for pixel-shuffle, 4*W*H pixels
            in (H,W,C) order)
Output file (out_upsample.txt): one output pixel per line, raster order
(2W x 2H).

Usage:
  python3 golden_upsample.py gen     # write vectors_upsample.txt
  python3 golden_upsample.py check   # compare out_upsample.txt vs golden
"""
import sys

VEC = "vectors_upsample.txt"
OUT = "out_upsample.txt"


def img(w, h, seed=1):
    return [(r * 37 + c * 17 + seed * 11 + 7) % 256
            for r in range(h) for c in range(w)]


def img_shuffle(w, h):
    # 4 channels per position, (H,W,C) order.
    p = []
    for r in range(h):
        for c in range(w):
            for ch in range(4):
                p.append((r * 37 + c * 17 + ch * 53 + 7) % 256)
    return p


def make_cases():
    cases = []

    def add(mode, w, h):
        if mode == 2:
            inp = img_shuffle(w, h)
        else:
            inp = img(w, h)
        cases.append(dict(mode=mode, w=w, h=h, inp=inp))

    # 0: nearest, 16x16
    add(0, 16, 16)
    # 1: nearest, odd 15x13
    add(0, 15, 13)
    # 2: nearest, small 4x4
    add(0, 4, 4)
    # 3: bilinear, 16x16
    add(1, 16, 16)
    # 4: bilinear, odd 15x13
    add(1, 15, 13)
    # 5: bilinear, small 4x4
    add(1, 4, 4)
    # 6: bilinear, 1x1 (single pixel, all edges replicate)
    add(1, 1, 1)
    # 7: pixel-shuffle, 16x16
    add(2, 16, 16)
    # 8: pixel-shuffle, odd 15x13
    add(2, 15, 13)
    # 9: pixel-shuffle, small 4x4
    add(2, 4, 4)
    return cases


def golden(cs):
    w, h, mode = cs["w"], cs["h"], cs["mode"]
    inp = cs["inp"]
    out = []
    if mode == 0:
        for r in range(h):
            for i in range(2):
                for c in range(w):
                    for j in range(2):
                        out.append(inp[r * w + c])
    elif mode == 1:
        def px(r, c):
            r = min(r, h - 1)
            c = min(c, w - 1)
            return inp[r * w + c]
        # Row-major: even output row 2r, then odd output row 2r+1.
        for r in range(2 * h):
            for c in range(2 * w):
                ir, ic = r // 2, c // 2
                if r % 2 == 0 and c % 2 == 0:
                    out.append(px(ir, ic))
                elif r % 2 == 0:
                    out.append((px(ir, ic) + px(ir, ic + 1) + 1) >> 1)
                elif c % 2 == 0:
                    out.append((px(ir, ic) + px(ir + 1, ic) + 1) >> 1)
                else:
                    out.append((px(ir, ic) + px(ir, ic + 1) +
                                px(ir + 1, ic) + px(ir + 1, ic + 1) + 2) >> 2)
    else:
        # pixel-shuffle: input is (H,W,C), 4 channels per position.
        # Row-major: even output row 2r, then odd row 2r+1.
        for r in range(2 * h):
            for c in range(2 * w):
                ir, ic = r // 2, c // 2
                ch = (r % 2) * 2 + (c % 2)
                out.append(inp[(ir * w + ic) * 4 + ch])
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['mode']} {cs['w']} {cs['h']}\n")
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
            print(f"case {idx} (mode={cs['mode']} {cs['w']}x{cs['h']}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                print(f"  pixel {i} (r={i // (2 * cs['w'])}, c={i % (2 * cs['w'])}): "
                      f"expected {e}, got {g}")
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
        print("usage: golden_upsample.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_upsample.py [gen|check]")
        sys.exit(2)
