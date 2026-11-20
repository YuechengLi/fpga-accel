#!/usr/bin/env python3
"""Golden model + vector generation + result check for conv2d_dw (Stage 2).

Math (identical to the RTL):
  padded image P_W = W + pad_l + pad_r, P_H = H + pad_t + pad_b (zero fill).
  Effective footprint (span_x, span_y) per mode:
    DW-SQUARE  (mode 0): span = K            (both axes)
    DW-STRIP-H (mode 1): span_x = L, span_y = 1
    DW-STRIP-V (mode 2): span_x = 1, span_y = L
    DW-DILATED (mode 3): span = (K-1)*d + 1  (both axes)
  out_W = (P_W - span_x)//S_x + 1, out_H = (P_H - span_y)//S_y + 1.
  out[or][oc] = clamp((acc32 + round) >> shift), where acc is the sum over
  the mode's taps of padded[or*S_y + dr][oc*S_x + dc] * kern, and acc32 is
  acc wrapped to signed 32-bit (2's complement), matching the RTL's
  32-bit accumulator. Tap offsets (dr, dc) within the footprint:
    square : (i, j), i,j in [0,K)
    strip-H: (0, j), j in [0,L)
    strip-V: (i, 0), i in [0,L)
    dilated: (i*d, j*d), i,j in [0,K)
  Coefficients are stored in one 289-entry array. Layout by mode:
    square : coeff[i*17+j] (17x17 grid, top-left KxK is the kernel)
    strip  : coeff[j] (1-D)
    dilated: coeff[i*9+j] (9x9 grid, top-left KxK is the kernel)

Vector file format (vectors_dw.txt):
  line 1: ncases
  per case: mode ksize dil shift W H pad_t pad_b pad_l pad_r stride_x stride_y
            289 coeff values
            W*H image pixels (raster order)
Output file (out_dw.txt): one output pixel per line, raster order.

Usage:
  python3 golden_dw.py gen     # write vectors_dw.txt
  python3 golden_dw.py check   # compare out_dw.txt vs golden
"""
import sys

VEC = "vectors_dw.txt"
OUT = "out_dw.txt"
NCOEFF = 289

MODE_SQUARE = 0
MODE_STRIP_H = 1
MODE_STRIP_V = 2
MODE_DILATED = 3


def img(w, h):
    p = []
    for r in range(h):
        for c in range(w):
            v = (r * 37 + c * 17 + 11) % 256
            if 4 <= r < 8 and 5 <= c < 9:
                v = 255
            p.append(v)
    return p


def coeff_array(mode, k, kern):
    """Place a kernel into the 289-entry coeff array (mode-dependent layout)."""
    g = [0] * NCOEFF
    if mode == MODE_SQUARE:
        for i in range(k):
            for j in range(k):
                g[i * 17 + j] = kern[i * k + j]
    elif mode in (MODE_STRIP_H, MODE_STRIP_V):
        for j in range(k):
            g[j] = kern[j]
    elif mode == MODE_DILATED:
        for i in range(k):
            for j in range(k):
                g[i * 9 + j] = kern[i * k + j]
    return g


def span(mode, k, d):
    if mode == MODE_SQUARE:
        return k, k
    if mode == MODE_STRIP_H:
        return k, 1
    if mode == MODE_STRIP_V:
        return 1, k
    return (k - 1) * d + 1, (k - 1) * d + 1


def make_cases():
    cases = []

    def add(mode, k, d, shift, kern, w, h, pt=0, pb=0, pl=0, pr=0, sx=1, sy=1,
            sign_in=0, sign_out=0, img_fn=None):
        cases.append(dict(mode=mode, k=k, d=d, shift=shift,
                          coeff=coeff_array(mode, k, kern),
                          img=img_fn(w, h) if img_fn else img(w, h), w=w, h=h,
                          pt=pt, pb=pb, pl=pl, pr=pr, sx=sx, sy=sy,
                          sign_in=sign_in, sign_out=sign_out))

    def img_signed(w, h):
        # Signed feature map: ~50% negative (two's-complement bytes).
        p = []
        for r in range(h):
            for c in range(w):
                v = (r * 53 + c * 29 + 7) % 256
                p.append(v)
        return p

    # ---- DW-SQUARE ----
    # 0: K=17 box, no pad, stride 1 (max square kernel)
    add(MODE_SQUARE, 17, 1, 6, [1] * 289, 24, 24)
    # 1: K=11 sharpen, symmetric pad 5, stride 1
    c11 = [0] * 121
    c11[5 * 11 + 5] = 512
    for dd in (-11, -1, 1, 11):
        c11[5 * 11 + 5 + dd] = -32
    add(MODE_SQUARE, 11, 1, 8, c11, 20, 20, pt=5, pb=5, pl=5, pr=5)
    # 2: K=7 sharpen, symmetric pad 3, stride 1, odd W/H
    s7 = [0] * 49
    s7[3 * 7 + 3] = 512
    for dd in (-7, -1, 1, 7):
        s7[3 * 7 + 3 + dd] = -64
    add(MODE_SQUARE, 7, 1, 8, s7, 15, 13, pt=3, pb=3, pl=3, pr=3)
    # 3: K=3 Sobel X, pad 1, stride 1
    add(MODE_SQUARE, 3, 1, 0, [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        16, 16, pt=1, pb=1, pl=1, pr=1)
    # 4: K=5 Gaussian, pad 2, stride 2
    g5 = [1, 4, 6, 4, 1,
          4, 16, 24, 16, 4,
          6, 24, 36, 24, 6,
          4, 16, 24, 16, 4,
          1, 4, 6, 4, 1]
    add(MODE_SQUARE, 5, 1, 8, g5, 16, 16, pt=2, pb=2, pl=2, pr=2, sx=2, sy=2)
    # 5: K=13 box, pad 6, stride 1
    add(MODE_SQUARE, 13, 1, 8, [1] * 169, 20, 16, pt=6, pb=6, pl=6, pr=6)
    # 6: K=15 box, pad 7, stride 1
    add(MODE_SQUARE, 15, 1, 8, [1] * 225, 16, 16, pt=7, pb=7, pl=7, pr=7)

    # ---- DW-STRIP-H ----
    # 7: L=31 box, pad 15 left/right, stride 1 (max strip)
    add(MODE_STRIP_H, 31, 1, 6, [1] * 31, 16, 16, pl=15, pr=15)
    # 8: L=21 box, pad 10, stride 1
    add(MODE_STRIP_H, 21, 1, 8, [1] * 21, 20, 16, pl=10, pr=10)
    # 9: L=11 box, pad 5, stride 2
    add(MODE_STRIP_H, 11, 1, 8, [1] * 11, 16, 16, pl=5, pr=5, sx=2, sy=2)
    # 10: L=7, pad 3, stride 1, odd W/H
    add(MODE_STRIP_H, 7, 1, 3, [1, 2, 3, 4, 3, 2, 1], 15, 13,
        pl=3, pr=3)

    # ---- DW-STRIP-V ----
    # 11: L=31 box, pad 15 top/bottom, stride 1 (max strip)
    add(MODE_STRIP_V, 31, 1, 6, [1] * 31, 16, 16, pt=15, pb=15)
    # 12: L=21 box, pad 10, stride 1
    add(MODE_STRIP_V, 21, 1, 8, [1] * 21, 16, 20, pt=10, pb=10)
    # 13: L=11 box, pad 5, stride 2
    add(MODE_STRIP_V, 11, 1, 8, [1] * 11, 16, 16, pt=5, pb=5, sx=2, sy=2)
    # 14: L=7, pad 3, stride 1, odd W/H
    add(MODE_STRIP_V, 7, 1, 3, [1, 2, 3, 4, 3, 2, 1], 13, 15,
        pt=3, pb=3)

    # ---- DW-DILATED ----
    # 15: K=3 d=1 Sobel, pad 1, stride 1 (d=1 is dense 3x3)
    add(MODE_DILATED, 3, 1, 0, [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        16, 16, pt=1, pb=1, pl=1, pr=1)
    # 16: K=3 d=2 box, pad 2, stride 1 (span 5)
    add(MODE_DILATED, 3, 2, 3, [1] * 9, 16, 16, pt=2, pb=2, pl=2, pr=2)
    # 17: K=3 d=4 box, pad 4, stride 1 (span 9)
    add(MODE_DILATED, 3, 4, 3, [1] * 9, 16, 16, pt=4, pb=4, pl=4, pr=4)
    # 18: K=5 d=2 Gaussian, pad 4, stride 1 (span 9)
    add(MODE_DILATED, 5, 2, 8, g5, 20, 20, pt=4, pb=4, pl=4, pr=4)
    # 19: K=9 d=4 box, pad 16, stride 1 (span 33, max)
    add(MODE_DILATED, 9, 4, 6, [1] * 81, 16, 16,
        pt=16, pb=16, pl=16, pr=16)
    # 20: K=5 d=4 Gaussian, pad 8, stride 2 (span 17)
    add(MODE_DILATED, 5, 4, 8, g5, 20, 20, pt=8, pb=8, pl=8, pr=8,
        sx=2, sy=2)
    # 21: K=7 d=2 sharpen, pad 6, stride 1, odd W/H (span 13)
    add(MODE_DILATED, 7, 2, 8, s7, 15, 13, pt=6, pb=6, pl=6, pr=6)
    # 22: K=9 d=2 box, pad 8, stride 1 (span 17)
    add(MODE_DILATED, 9, 2, 6, [1] * 81, 16, 16, pt=8, pb=8, pl=8, pr=8)

    # ---- SIGNED (Stage 4) ----
    # 23: K=3 Sobel X on a signed feature map (~50% negative bytes),
    #     pad 1, stride 1, signed in + signed out. A 3x3 Sobel on signed
    #     data exercises sign-extension in the MAC and the [-128,127]
    #     saturation; unsigned handling would read negatives as 128..255.
    add(MODE_SQUARE, 3, 1, 0, [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        16, 16, pt=1, pb=1, pl=1, pr=1, sign_in=1, sign_out=1,
        img_fn=img_signed)
    # 24: K=5 Gaussian on signed data, pad 2, stride 1, signed in/out.
    add(MODE_SQUARE, 5, 1, 8, g5, 16, 16, pt=2, pb=2, pl=2, pr=2,
        sign_in=1, sign_out=1, img_fn=img_signed)
    # 25: L=7 strip-H on signed data, pad 3, stride 1, signed in/out.
    add(MODE_STRIP_H, 7, 1, 3, [1, 2, 3, 4, 3, 2, 1], 15, 13,
        pl=3, pr=3, sign_in=1, sign_out=1, img_fn=img_signed)
    return cases


def out_size(cs):
    spx, spy = span(cs["mode"], cs["k"], cs["d"])
    pw = cs["w"] + cs["pl"] + cs["pr"]
    ph = cs["h"] + cs["pt"] + cs["pb"]
    ow = (pw - spx) // cs["sx"] + 1
    oh = (ph - spy) // cs["sy"] + 1
    return ow, oh


def wrap32(v):
    v = v % (1 << 32)
    return v - (1 << 32) if v >= (1 << 31) else v


def taps(cs):
    """Yield (dr, dc, coeff_value) for every tap of this case's kernel."""
    mode, k, d = cs["mode"], cs["k"], cs["d"]
    coeff = cs["coeff"]
    if mode == MODE_SQUARE:
        for i in range(k):
            for j in range(k):
                yield i, j, coeff[i * 17 + j]
    elif mode == MODE_STRIP_H:
        for j in range(k):
            yield 0, j, coeff[j]
    elif mode == MODE_STRIP_V:
        for i in range(k):
            yield i, 0, coeff[i]
    else:  # dilated
        for i in range(k):
            for j in range(k):
                yield i * d, j * d, coeff[i * 9 + j]


def golden(cs):
    w, h, shift = cs["w"], cs["h"], cs["shift"]
    im = cs["img"]
    pt, pl, sx, sy = cs["pt"], cs["pl"], cs["sx"], cs["sy"]
    sign_in = cs.get("sign_in", 0)
    sign_out = cs.get("sign_out", 0)
    ow, oh = out_size(cs)
    tl = list(taps(cs))
    out = []
    for orow in range(oh):
        for ocol in range(ow):
            acc = 0
            for dr, dc, cv in tl:
                rr = orow * sy + dr - pt
                cc = ocol * sx + dc - pl
                if 0 <= rr < h and 0 <= cc < w:
                    pv = im[rr * w + cc]
                    if sign_in and pv >= 128:
                        pv -= 256
                    acc += pv * cv
            acc = wrap32(acc)
            rnd = acc + (1 << (shift - 1)) if shift > 0 else acc
            rnd = wrap32(rnd)
            v = rnd >> shift
            if sign_out:
                # Saturate to [-128, 127], emitted as a two's-complement
                # byte (the TB prints pixel_out as an unsigned byte).
                v = max(-128, min(127, v)) & 0xFF
            else:
                v = max(0, min(255, v))
            out.append(v)
    return out


def mode_name(m):
    return ["DW-SQUARE", "DW-STRIP-H", "DW-STRIP-V", "DW-DILATED"][m]


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['mode']} {cs['k']} {cs['d']} {cs['shift']} "
                    f"{cs['w']} {cs['h']} "
                    f"{cs['pt']} {cs['pb']} {cs['pl']} {cs['pr']} "
                    f"{cs['sx']} {cs['sy']} "
                    f"{cs.get('sign_in', 0)} {cs.get('sign_out', 0)}\n")
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
            print(f"case {idx} ({mode_name(cs['mode'])} k={cs['k']} d={cs['d']} "
                  f"{cs['w']}x{cs['h']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) "
                  f"sign=({cs.get('sign_in', 0)},{cs.get('sign_out', 0)}) "
                  f"out={ow}x{oh}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for i, e, g in bad[:5]:
                print(f"  pixel {i} (r={i // ow}, c={i % ow}): expected {e}, got {g}")
        else:
            print(f"case {idx} ({mode_name(cs['mode'])} k={cs['k']} d={cs['d']} "
                  f"{cs['w']}x{cs['h']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) "
                  f"sign=({cs.get('sign_in', 0)},{cs.get('sign_out', 0)}) "
                  f"out={ow}x{oh}): PASS ({n} pixels)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    elif len(got) < pos:
        print(f"WARNING: {pos - len(got)} outputs missing from {OUT} (already counted)")
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: golden_dw.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_dw.py [gen|check]")
        sys.exit(2)
