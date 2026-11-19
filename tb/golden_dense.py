#!/usr/bin/env python3
"""Golden model + vector generation + result check for conv2d_dense (Stage 1).

Math (identical to the RTL):
  padded image P_W = W + pad_l + pad_r, P_H = H + pad_t + pad_b (zero fill).
  out_W = (P_W - K)//S_x + 1, out_H = (P_H - K)//S_y + 1.
  out[or][oc] = clamp((acc + round) >> shift), acc = sum over the KxK kernel
  of padded[or*S_y + i][oc*S_x + j] * kern[i][j].
  Coefficients are stored in an 11x11 grid (row-major, stride 11); the KxK
  kernel is the top-left KxK of that grid.

Vector file format (vectors_dense.txt):
  line 1: ncases
  per case: W H K shift pad_t pad_b pad_l pad_r stride_x stride_y
            121 coeff values (11x11 grid, row-major)
            W*H image pixels (raster order)
Output file (out_dense.txt): one output pixel per line, raster order.

Usage:
  python3 golden_dense.py gen     # write vectors_dense.txt
  python3 golden_dense.py check   # compare out_dense.txt vs golden
"""
import sys

VEC = "vectors_dense.txt"
OUT = "out_dense.txt"
MAX_K = 11


def img(w, h):
    p = []
    for r in range(h):
        for c in range(w):
            v = (r * 37 + c * 17 + 11) % 256
            if 4 <= r < 8 and 5 <= c < 9:
                v = 255
            p.append(v)
    return p


def grid(kern, k):
    """Place a KxK kernel (row-major, stride k) into the 11x11 grid."""
    g = [0] * (MAX_K * MAX_K)
    for i in range(k):
        for j in range(k):
            g[i * MAX_K + j] = kern[i * k + j]
    return g


def make_cases():
    cases = []

    def add(w, h, k, shift, kern, pt=0, pb=0, pl=0, pr=0, sx=1, sy=1):
        cases.append(dict(w=w, h=h, k=k, shift=shift, coeff=grid(kern, k),
                          img=img(w, h), pt=pt, pb=pb, pl=pl, pr=pr,
                          sx=sx, sy=sy))

    # 0: K=3 Sobel X, no padding, stride 1
    add(16, 16, 3, 0, [-1, 0, 1, -2, 0, 2, -1, 0, 1])
    # 1: K=3 box blur, symmetric padding 1, stride 1
    add(16, 16, 3, 3, [1] * 9, pt=1, pb=1, pl=1, pr=1)
    # 2: K=5 Gaussian, asymmetric padding, stride 1, non-square
    g5 = [1, 4, 6, 4, 1,
          4, 16, 24, 16, 4,
          6, 24, 36, 24, 6,
          4, 16, 24, 16, 4,
          1, 4, 6, 4, 1]
    add(12, 20, 5, 8, g5, pt=0, pb=2, pl=1, pr=3)
    # 3: K=11 box, no padding, stride 1 (max kernel)
    add(16, 16, 11, 6, [1] * 121)
    # 4: K=1 identity, no padding, stride 1
    add(16, 16, 1, 8, [256])
    # 5: K=3 identity, symmetric padding 1, stride 2
    add(16, 16, 3, 8, [0, 0, 0, 0, 256, 0, 0, 0, 0],
        pt=1, pb=1, pl=1, pr=1, sx=2, sy=2)
    # 6: K=3 box, symmetric padding 1, asymmetric stride 2x3
    add(16, 16, 3, 3, [1] * 9, pt=1, pb=1, pl=1, pr=1, sx=2, sy=3)
    # 7: K=7 sharpen, asymmetric padding, stride 3, odd h
    s7 = [0] * 49
    s7[3 * 7 + 3] = 512
    for d in (-7, -1, 1, 7):
        s7[3 * 7 + 3 + d] = -64
    add(16, 12, 7, 8, s7, pt=2, pb=1, pl=0, pr=2, sx=3, sy=3)
    # 8: K=5, padding > K/2 (pad 3 each side, K=5, h=2), stride 1, odd W/H
    add(15, 11, 5, 8, g5, pt=3, pb=3, pl=3, pr=3)
    # 9: K=9 box, symmetric padding 4, stride 1, larger image
    add(64, 48, 9, 6, [1] * 81, pt=4, pb=4, pl=4, pr=4)
    # 10: K=11, asymmetric padding, stride 2
    c11 = [0] * 121
    c11[5 * 11 + 5] = 512
    for d in (-11, -1, 1, 11):
        c11[5 * 11 + 5 + d] = -32
    add(32, 24, 11, 8, c11, pt=1, pb=3, pl=2, pr=0, sx=2, sy=2)
    # 11: K=3, large padding (3 each side, K=3, h=1), stride 1
    add(20, 18, 3, 3, [1] * 9, pt=3, pb=3, pl=3, pr=3)
    # 12: K=7, stride 1, odd width and height
    add(15, 13, 7, 8, s7, pt=3, pb=3, pl=3, pr=3)
    # 13: K=5, asymmetric stride 3x2, padding 2
    add(24, 20, 5, 8, g5, pt=2, pb=2, pl=2, pr=2, sx=3, sy=2)
    return cases


def out_size(cs):
    pw = cs["w"] + cs["pl"] + cs["pr"]
    ph = cs["h"] + cs["pt"] + cs["pb"]
    ow = (pw - cs["k"]) // cs["sx"] + 1
    oh = (ph - cs["k"]) // cs["sy"] + 1
    return ow, oh


def golden(cs):
    w, h, k, shift = cs["w"], cs["h"], cs["k"], cs["shift"]
    coeff = cs["coeff"]
    im = cs["img"]
    pt, pl, sx, sy = cs["pt"], cs["pl"], cs["sx"], cs["sy"]
    ow, oh = out_size(cs)
    out = []
    for orow in range(oh):
        for ocol in range(ow):
            acc = 0
            for i in range(k):
                for j in range(k):
                    rr = orow * sy + i - pt
                    cc = ocol * sx + j - pl
                    if 0 <= rr < h and 0 <= cc < w:
                        acc += im[rr * w + cc] * coeff[i * MAX_K + j]
            rnd = acc + (1 << (shift - 1)) if shift > 0 else acc
            v = rnd >> shift
            out.append(max(0, min(255, v)))
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['w']} {cs['h']} {cs['k']} {cs['shift']} "
                    f"{cs['pt']} {cs['pb']} {cs['pl']} {cs['pr']} "
                    f"{cs['sx']} {cs['sy']}\n")
            f.write(" ".join(str(v) for v in cs["coeff"]) + "\n")
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
                  f"stride=({cs['sx']},{cs['sy']}) out={ow}x{oh}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                print(f"  pixel {i} (r={i // ow}, c={i % ow}): expected {e}, got {g}")
        else:
            print(f"case {idx} ({cs['w']}x{cs['h']} k={cs['k']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) out={ow}x{oh}): PASS ({n} pixels)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    elif len(got) < pos:
        print(f"WARNING: {pos - len(got)} outputs missing from {OUT} (already counted)")
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: golden_dense.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_dense.py [gen|check]")
        sys.exit(2)
