#!/usr/bin/env python3
"""Golden model + vector generation + result check for conv2d_mc (Stage 3).

Math (identical to the RTL):
  Input: C_in frames (channel-major), each W*H, raster order.
  For each output channel o and output pixel (or, oc):
    acc = sum over c in [0, C_in), (i, j) in KxK of
          padded_c[or*S_y + i - pad_t][oc*S_x + j - pad_l] * kern[o][c][i][j]
    where padded_c is channel c zero-padded. acc is wrapped to signed
    32-bit (2's complement), matching the RTL's 32-bit accumulator.
    out = clamp((acc + round) >> shift), round = 1 << (shift-1) if shift>0.
  The RTL tiles C_in by C_PAR=4 and accumulates across tiles; the golden
  sums over all C_in directly (tiling is an implementation detail).
  K=1 is the 1x1 pointwise convolution; K=1 with W=H=1 is the FC path.

Vector file format (vectors_mc.txt):
  line 1: ncases
  per case: ksize shift W H pad_t pad_b pad_l pad_r stride_x stride_y
            C_in C_out out_w out_h sgn_in sgn_out
            (sgn_in=1: signed-int8 input pixels; sgn_out=1: signed-int8
            output, clamped to [-128, 127] instead of [0, 255])
            then C_out*C_in*K*K coefficients, grouped by (o, c), row-major
            within each KxK kernel (only the KxK region is stored)
            then C_in*W*H image pixels, channel-major, raster order
Output file (out_mc.txt): one output pixel per line, raster order,
  output-channel-major (all of frame 0, then frame 1, ...).

Usage:
  python3 golden_mc.py gen     # write vectors_mc.txt
  python3 golden_mc.py check   # compare out_mc.txt vs golden
"""
import sys

VEC = "vectors_mc.txt"
OUT = "out_mc.txt"
C_PAR = 4


def img(w, h, seed):
    p = []
    for r in range(h):
        for c in range(w):
            v = (r * 37 + c * 17 + seed * 11 + 7) % 256
            if 3 <= r < 6 and 4 <= c < 7:
                v = 200
            p.append(v)
    return p


def simg(w, h, seed):
    """Signed-int8 image: values in [-128, 127], ~50% negative.
    Models a signed intermediate activation (e.g. RetinexNet conv0
    output, range about [-0.33, +0.31] scaled by 255)."""
    p = []
    for r in range(h):
        for c in range(w):
            v = (r * 37 + c * 17 + seed * 11 + 7) % 256 - 128
            if 3 <= r < 6 and 4 <= c < 7:
                v = 100
            p.append(v)
    return p


def kern(o, c, k, kind):
    """Deterministic KxK kernel for (output channel o, input channel c)."""
    g = []
    for i in range(k):
        for j in range(k):
            if kind == "box":
                g.append(1)
            elif kind == "ident":
                g.append(256 if (i == k // 2 and j == k // 2) else 0)
            elif kind == "rand":
                g.append(((o * 7 + c * 13 + i * 5 + j * 3 + 2) % 5) - 2)
            elif kind == "edge":
                # Sobel-like for 3x3; box for other sizes
                if k == 3:
                    g = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
                    return g
                g.append(1)
            else:
                g.append(1)
    return g


def make_cases():
    cases = []

    def add(k, shift, w, h, cin, cout, kind="rand",
            pt=0, pb=0, pl=0, pr=0, sx=1, sy=1, sgn_in=0, sgn_out=0,
            fuse_act=0, act_slope=8192):
        pw = w + pl + pr
        ph = h + pt + pb
        ow = (pw - k) // sx + 1
        oh = (ph - k) // sy + 1
        kernels = {}
        for o in range(cout):
            for c in range(cin):
                kernels[(o, c)] = kern(o, c, k, kind)
        if sgn_in:
            images = [simg(w, h, 1 + c) for c in range(cin)]
        else:
            images = [img(w, h, 1 + c) for c in range(cin)]
        cases.append(dict(k=k, shift=shift, w=w, h=h, cin=cin, cout=cout,
                          kind=kind, pt=pt, pb=pb, pl=pl, pr=pr,
                          sx=sx, sy=sy, ow=ow, oh=oh, sgn_in=sgn_in,
                          sgn_out=sgn_out, kernels=kernels, images=images,
                          fuse_act=fuse_act, act_slope=act_slope))

    # 0: C_in=3, C_out=2, K=3, box-ish random, no pad, stride 1
    add(3, 8, 16, 16, 3, 2, "rand")
    # 1: C_in=8, C_out=2, K=3, random, pad 1, stride 1
    add(3, 8, 16, 16, 8, 2, "rand", pt=1, pb=1, pl=1, pr=1)
    # 2: C_in=16, C_out=8, K=3, random, pad 1, stride 1
    add(3, 8, 16, 16, 16, 8, "rand", pt=1, pb=1, pl=1, pr=1)
    # 3: C_in=64, C_out=2, K=3, random, no pad, stride 1 (max C_in)
    add(3, 8, 16, 16, 64, 2, "rand")
    # 4: C_in=8, C_out=4, K=5, random, pad 2, stride 1
    add(5, 8, 16, 16, 8, 4, "rand", pt=2, pb=2, pl=2, pr=2)
    # 5: C_in=16, C_out=16, K=1 (1x1 pointwise), random, no pad
    add(1, 8, 16, 16, 16, 16, "rand")
    # 6: C_in=64, C_out=32, K=1 (1x1 pointwise, max C_out), random
    add(1, 8, 16, 16, 64, 32, "rand")
    # 7: FC: C_in=8, C_out=4, K=1, W=H=1 (input vector as 1x1 feature map)
    add(1, 8, 1, 1, 8, 4, "rand")
    # 8: C_in=3, C_out=2, K=3, random, pad 1, stride 2x2, odd W/H
    add(3, 8, 15, 13, 3, 2, "rand", pt=1, pb=1, pl=1, pr=1, sx=2, sy=2)
    # 9: C_in=8, C_out=4, K=3, identity-ish, shift 0, no pad
    add(3, 0, 16, 16, 8, 4, "ident")
    # 10: signed in + signed out, K=3, C_in=4, C_out=2, pad 1
    # (RetinexNet Decom-Net conv0: no activation, ~50% negative output)
    add(3, 8, 16, 16, 4, 2, "rand", pt=1, pb=1, pl=1, pr=1,
        sgn_in=1, sgn_out=1)
    # 11: signed in, unsigned out, K=3, C_in=8, C_out=2, pad 1
    # (next layer: signed intermediate into a ReLU / unsigned output)
    add(3, 8, 16, 16, 8, 2, "rand", pt=1, pb=1, pl=1, pr=1,
        sgn_in=1, sgn_out=0)
    # 12: FUSED ReLU (Stage 4), signed in + signed out, K=3, C_in=4,
    # C_out=2, pad 1. Without the fuse, negatives would be kept as
    # [-128,127]; with fused ReLU they are clamped to 0. The act
    # module is bypassed for this layer (mode 7).
    add(3, 8, 16, 16, 4, 2, "rand", pt=1, pb=1, pl=1, pr=1,
        sgn_in=1, sgn_out=1, fuse_act=1)
    # 13: FUSED Leaky (Stage 4), slope 0.125, signed in + signed out,
    # K=3, C_in=4, C_out=2, pad 1. Negatives are scaled by 0.125.
    add(3, 8, 16, 16, 4, 2, "rand", pt=1, pb=1, pl=1, pr=1,
        sgn_in=1, sgn_out=1, fuse_act=2, act_slope=8192)
    # 14: FUSED ReLU after an FC/MLP layer (Stage 4 steer). FC reuses
    # the 1x1 datapath (K=1, W=H=1, C_in=8, C_out=4), so the fused
    # ReLU in the conv output stage covers the MLP path exactly as it
    # covers the conv path -- no separate FC activation is needed.
    # Signed in/out so the pre-ReLU values can be negative (with
    # unsigned output the [0,255] saturation would already clip them
    # and the fuse would be a no-op). The act module is bypassed.
    add(1, 8, 1, 1, 8, 4, "rand",
        sgn_in=1, sgn_out=1, fuse_act=1)
    return cases


def wrap32(v):
    v = v % (1 << 32)
    return v - (1 << 32) if v >= (1 << 31) else v


def golden(cs):
    """Return list of C_out frames, each a flat list of out pixels."""
    k, shift = cs["k"], cs["shift"]
    w, h, cin, cout = cs["w"], cs["h"], cs["cin"], cs["cout"]
    pt, pl, sx, sy = cs["pt"], cs["pl"], cs["sx"], cs["sy"]
    ow, oh = cs["ow"], cs["oh"]
    sgn_out = cs.get("sgn_out", 0)
    fuse_act = cs.get("fuse_act", 0)
    act_slope = cs.get("act_slope", 8192)
    frames = []
    for o in range(cout):
        out = []
        for orow in range(oh):
            for ocol in range(ow):
                acc = 0
                for c in range(cin):
                    kern = cs["kernels"][(o, c)]
                    im = cs["images"][c]
                    for i in range(k):
                        for j in range(k):
                            rr = orow * sy + i - pt
                            cc = ocol * sx + j - pl
                            if 0 <= rr < h and 0 <= cc < w:
                                acc += im[rr * w + cc] * kern[i * k + j]
                acc = wrap32(acc)
                rnd = acc + (1 << (shift - 1)) if shift > 0 else acc
                rnd = wrap32(rnd)
                v = rnd >> shift
                # Fused activation (Stage 4): applied to the shifted
                # value, BEFORE saturation (mirrors the RTL emit stage).
                if fuse_act == 1:
                    v = max(0, v)
                elif fuse_act == 2:
                    v = v if v >= 0 else (v * act_slope) >> 16
                if sgn_out:
                    out.append(max(-128, min(127, v)))
                else:
                    out.append(max(0, min(255, v)))
        frames.append(out)
    return frames


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['k']} {cs['shift']} {cs['w']} {cs['h']} "
                    f"{cs['pt']} {cs['pb']} {cs['pl']} {cs['pr']} "
                    f"{cs['sx']} {cs['sy']} {cs['cin']} {cs['cout']} "
                    f"{cs['ow']} {cs['oh']} "
                    f"{cs.get('sgn_in', 0)} {cs.get('sgn_out', 0)} "
                    f"{cs.get('fuse_act', 0)} {cs.get('act_slope', 8192)}\n")
            # coefficients: (o, c) order, KxK row-major each
            for o in range(cs["cout"]):
                for c in range(cs["cin"]):
                    f.write(" ".join(str(v) for v in cs["kernels"][(o, c)]) + "\n")
            # images: channel-major
            for c in range(cs["cin"]):
                f.write(" ".join(str(v) for v in cs["images"][c]) + "\n")
    print(f"wrote {VEC}: {len(cases)} cases")


def check():
    cases = make_cases()
    got = [int(x) for x in open(OUT).read().split()]
    pos, nerr = 0, 0
    for idx, cs in enumerate(cases):
        frames = golden(cs)
        n = sum(len(fr) for fr in frames)
        seg = got[pos:pos + n]
        pos += n
        if len(seg) < n:
            print(f"case {idx}: MISSING outputs: got {len(seg)}, expected {n}")
            nerr += n - len(seg)
            continue
        bad = 0
        p = 0
        for o, fr in enumerate(frames):
            for i, (e, g) in enumerate(zip(fr, seg[p:p + len(fr)])):
                if e != g:
                    bad += 1
                    if bad <= 3:
                        print(f"case {idx} ch{o} pixel {i} "
                              f"(r={i // cs['ow']}, c={i % cs['ow']}): "
                              f"expected {e}, got {g}")
            p += len(fr)
        tag = (f"sgn_in={cs.get('sgn_in', 0)} "
               f"sgn_out={cs.get('sgn_out', 0)} ") if cs.get('sgn_in', 0) or cs.get('sgn_out', 0) else ""
        if cs.get('fuse_act', 0):
            tag += f"fuse_act={cs['fuse_act']} slope={cs.get('act_slope', 8192)} "
        if bad:
            nerr += bad
            print(f"case {idx} (K={cs['k']} {cs['w']}x{cs['h']} "
                  f"C_in={cs['cin']} C_out={cs['cout']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) out={cs['ow']}x{cs['oh']} {tag}): "
                  f"{bad}/{n} mismatches")
        else:
            print(f"case {idx} (K={cs['k']} {cs['w']}x{cs['h']} "
                  f"C_in={cs['cin']} C_out={cs['cout']} shift={cs['shift']} "
                  f"pad=({cs['pt']},{cs['pb']},{cs['pl']},{cs['pr']}) "
                  f"stride=({cs['sx']},{cs['sy']}) out={cs['ow']}x{cs['oh']} {tag}): "
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
        print("usage: golden_mc.py [gen|check]")
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        gen()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_mc.py [gen|check]")
        sys.exit(2)
