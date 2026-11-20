#!/usr/bin/env python3
"""Golden model + vector generation + result check for frontend (Stage 7).

Math (identical to the RTL):
  padded image P_W = W + pad_l + pad_r, P_H = H + pad_t + pad_b (zero fill).
  eff_Sx = stride_x * DS, eff_Sy = stride_y * DS (DS = downsample 1/2/4).
  out_W = (P_W - K)//eff_Sx + 1, out_H = (P_H - K)//eff_Sy + 1.
  Per output position (or, oc) and channel c:
    acc = sum over the KxK kernel of
          padded[or*eff_Sy + i][oc*eff_Sx + j][c] * kern[i][j],
          where the input pixel is sign-extended (int8) when sign_in is set.
    rnd = acc + (1 << (shift-1)) if shift > 0 else acc; v = rnd >> shift.
    fused act (after shift, before saturate): 0=none, 1=ReLU (max(0,v)),
    2=Leaky (v<0 -> (v * slope_signed) >> 16, slope is Q0.16).
    saturate: sign_out -> [-128, 127]; else [0, 255].
  Coefficients are stored in a 5x5 grid (row-major, stride 5); the KxK
  kernel is the top-left KxK of that grid. The same kernel is applied
  independently to every channel (shared coefficient bank).

Vector file format (vectors_frontend.txt):
  line 1: ncases
  per case: W H K shift pad_t pad_b pad_l pad_r stride_x stride_y ds
            channels sign fused_act slope
            25 coeff values (5x5 grid, row-major)
            W*H*channels image values (raster order, channel-minor:
            at each position the channels in order ch0, ch1, ch2)
Output file (out_frontend.txt): one output value per line, raster order,
channel-minor (same order as the golden's flat output list).

Usage:
  python3 golden_frontend.py gen     # write vectors_frontend.txt
  python3 golden_frontend.py check   # compare out_frontend.txt vs golden
"""
import sys

VEC = "vectors_frontend.txt"
OUT = "out_frontend.txt"
MAX_K = 5


def s16(v):
    """Interpret a 16-bit value as signed."""
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def s8(v):
    """Interpret an 8-bit value as signed."""
    v &= 0xFF
    return v - 0x100 if v & 0x80 else v


def img(w, h, ch):
    """Per-channel test image. Channels differ so a channel mix-up is caught."""
    p = []
    for r in range(h):
        for c in range(w):
            for k in range(ch):
                if k == 0:
                    v = (r * 37 + c * 17 + 11) % 256
                elif k == 1:
                    v = (r * 23 + c * 41 + 97) % 256
                else:
                    v = (r * 53 + c * 29 + 199) % 256
                if 4 <= r < 8 and 5 <= c < 9:
                    v = 255
                p.append(v)
    return p


def grid(kern, k):
    """Place a KxK kernel (row-major, stride k) into the 5x5 grid."""
    g = [0] * (MAX_K * MAX_K)
    for i in range(k):
        for j in range(k):
            g[i * MAX_K + j] = kern[i * k + j]
    return g


def make_cases():
    cases = []

    def add(w, h, k, shift, kern, ch=3, ds=1, pt=0, pb=0, pl=0, pr=0,
            sx=1, sy=1, sign=0, act=0, slope=16384):
        cases.append(dict(w=w, h=h, k=k, shift=shift, coeff=grid(kern, k),
                          img=img(w, h, ch), ch=ch, ds=ds,
                          pt=pt, pb=pb, pl=pl, pr=pr, sx=sx, sy=sy,
                          sign=sign, act=act, slope=slope))

    g5 = [1, 4, 6, 4, 1,
          4, 16, 24, 16, 4,
          6, 24, 36, 24, 6,
          4, 16, 24, 16, 4,
          1, 4, 6, 4, 1]
    sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    sharpen3 = [0, -1, 0, -1, 5, -1, 0, -1, 0]
    lap5 = [0, 0, -1, 0, 0,
            0, -1, -2, -1, 0,
            -1, -2, 16, -2, -1,
            0, -1, -2, -1, 0,
            0, 0, -1, 0, 0]

    # 0: 1ch K=3 identity, no pad, stride 1, DS=1
    add(16, 16, 3, 8, [0, 0, 0, 0, 256, 0, 0, 0, 0], ch=1)
    # 1: 3ch K=3 box blur, symmetric pad 1, stride 1, DS=1
    add(16, 16, 3, 3, [1] * 9, ch=3, pt=1, pb=1, pl=1, pr=1)
    # 2: 3ch K=5 Gaussian, asymmetric pad, stride 1, DS=1
    add(12, 20, 5, 8, g5, ch=3, pt=0, pb=2, pl=1, pr=3)
    # 3: 1ch K=1 identity, DS=2 (downsample on)
    add(16, 16, 1, 8, [256], ch=1, ds=2)
    # 4: 3ch K=3 blur, symmetric pad 1, DS=2
    add(16, 16, 3, 3, [1] * 9, ch=3, ds=2, pt=1, pb=1, pl=1, pr=1)
    # 5: 3ch K=3 sharpen, pad 1, DS=2, stride 2 (eff 4x4)
    add(16, 16, 3, 8, sharpen3, ch=3, ds=2, pt=1, pb=1, pl=1, pr=1,
        sx=2, sy=2)
    # 6: 1ch K=3 Sobel X, no pad, DS=1, fused ReLU (negatives clip to 0)
    add(16, 16, 3, 0, sobel_x, ch=1, act=1)
    # 7: 3ch K=5 Gaussian, pad 2, DS=4 (max downsample)
    add(16, 16, 5, 8, g5, ch=3, ds=4, pt=2, pb=2, pl=2, pr=2)
    # 8: 1ch K=5, asymmetric pad, stride 2, DS=1, odd W/H
    add(15, 11, 5, 8, g5, ch=1, pt=3, pb=1, pl=2, pr=0, sx=2, sy=1)
    # 9: 3ch K=3 Sobel, signed in/out (sign=3), pad 1, DS=1
    #    (signed intermediates: negative pixels in, negative out)
    add(16, 16, 3, 0, sobel_x, ch=3, pt=1, pb=1, pl=1, pr=1, sign=3)
    # 10: 3ch K=1 per-channel gain (x2, shift 1), DS=2, shift 0
    add(16, 16, 1, 0, [512], ch=3, ds=2)
    # 11: 1ch K=3 Sobel, fused Leaky (slope 0.25), pad 1, DS=1
    add(16, 16, 3, 0, sobel_x, ch=1, pt=1, pb=1, pl=1, pr=1,
        act=2, slope=16384)
    # 12: 3ch K=3 blur, pad 1, stride 2, DS=2, larger image
    add(32, 24, 3, 3, [1] * 9, ch=3, pt=1, pb=1, pl=1, pr=1,
        sx=2, sy=2, ds=2)
    # 13: 1ch K=5 Laplacian edge, pad 2, DS=1, shift 0, fused ReLU
    add(16, 16, 5, 0, lap5, ch=1, pt=2, pb=2, pl=2, pr=2, act=1)
    # 14: 3ch K=3 blur, DS=4, stride 1, no pad, larger image
    add(32, 24, 3, 3, [1] * 9, ch=3, ds=4)
    # 15: 1ch K=3 identity, DS=1, stride 3, pad 1 (stride test)
    add(16, 16, 3, 8, [0, 0, 0, 0, 256, 0, 0, 0, 0], ch=1,
        pt=1, pb=1, pl=1, pr=1, sx=3, sy=3)
    return cases


def out_size(cs):
    pw = cs["w"] + cs["pl"] + cs["pr"]
    ph = cs["h"] + cs["pt"] + cs["pb"]
    esx = cs["sx"] * cs["ds"]
    esy = cs["sy"] * cs["ds"]
    ow = (pw - cs["k"]) // esx + 1
    oh = (ph - cs["k"]) // esy + 1
    return ow, oh


def golden(cs):
    w, h, k, shift = cs["w"], cs["h"], cs["k"], cs["shift"]
    coeff = cs["coeff"]
    im = cs["img"]
    ch = cs["ch"]
    pt, pl = cs["pt"], cs["pl"]
    esx = cs["sx"] * cs["ds"]
    esy = cs["sy"] * cs["ds"]
    sign_in = cs["sign"] & 1
    sign_out = (cs["sign"] >> 1) & 1
    ow, oh = out_size(cs)
    out = []
    for orow in range(oh):
        for ocol in range(ow):
            for c in range(ch):
                acc = 0
                for i in range(k):
                    for j in range(k):
                        rr = orow * esy + i - pt
                        cc = ocol * esx + j - pl
                        if 0 <= rr < h and 0 <= cc < w:
                            px = im[(rr * w + cc) * ch + c]
                            if sign_in:
                                px = s8(px)
                            acc += px * coeff[i * MAX_K + j]
                rnd = acc + (1 << (shift - 1)) if shift > 0 else acc
                v = rnd >> shift
                if cs["act"] == 1:  # ReLU
                    v = max(0, v)
                elif cs["act"] == 2:  # Leaky
                    if v < 0:
                        v = (v * s16(cs["slope"])) >> 16
                if sign_out:
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
                    f"{cs['sx']} {cs['sy']} {cs['ds']} {cs['ch']} "
                    f"{cs['sign']} {cs['act']} {cs['slope']}\n")
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
        maxerr = max((abs(e - g) for e, g in zip(exp, seg)), default=0)
        if bad:
            nerr += len(bad)
            print(f"case {idx} ({cs['w']}x{cs['h']} ch={cs['ch']} k={cs['k']} "
                  f"shift={cs['shift']} pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) ds={cs['ds']} sign={cs['sign']} "
                  f"act={cs['act']} out={ow}x{oh}): "
                  f"{len(bad)}/{n} mismatches, max|err|={maxerr}, first 5:")
            for i, e, g in bad[:5]:
                print(f"  value {i} (r={i // (ow * cs['ch'])}, "
                      f"c={(i // cs['ch']) % ow}, ch={i % cs['ch']}): "
                      f"expected {e}, got {g}")
        else:
            print(f"case {idx} ({cs['w']}x{cs['h']} ch={cs['ch']} k={cs['k']} "
                  f"shift={cs['shift']} pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) ds={cs['ds']} sign={cs['sign']} "
                  f"act={cs['act']} out={ow}x{oh}): PASS ({n} values, max|err|=0)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    elif len(got) < pos:
        print(f"WARNING: {pos - len(got)} outputs missing from {OUT} (already counted)")
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: golden_frontend.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_frontend.py [gen|check]")
        sys.exit(2)
