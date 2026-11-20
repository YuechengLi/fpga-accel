#!/usr/bin/env python3
"""Golden model + vector generator / checker for unified_engine.sv (Stage 4).

Verifies the one-click mode switch: consecutive frames run in different
modes (dense -> DW -> FC -> DW-strip -> dense) with no reset between
frames, and the engine must switch configuration correctly (including
dense -> DW -> 1x1). Coefficients are per-mode (each submodule keeps
its own), so switching back to a previous mode needs no reload -- the
last case re-runs dense with the FIRST case's coefficients and must
match without reloading them.

Dense golden mirrors conv2d_dense (now with Stage 4 sign + fused act):
  acc = sum over KxK taps (sign-extended pixels if sign_in), wrap32,
  rnd = acc + (1 << (shift-1)) if shift > 0, v = rnd >> shift
  (Python >> is a floor shift, matching RTL >>>),
  fused act (before saturation): ReLU max(0,v); Leaky v<0 ? (v*slope)>>16 : v,
  saturate to [0,255] (unsigned out) or [-128,127] (signed out, as a byte).
DW golden is imported from golden_dw (same math as conv2d_dw, incl. sign).
The FC mode is dense with K=1, W=H=1 (the 1x1 datapath); the fused ReLU
there covers the MLP path (MC case 14 is the multi-channel version).

Vector format (vectors_unified.txt):
  ncases
  per case:
    mode k shift w h pt pb pl pr sx sy sign_in sign_out fuse_act act_slope d
    <coeffs: dense 121 values (11x11 grid); DW 289 values (DW layout)>
    <img: w*h byte values>
Output file: out_unified.txt (one byte per line, in order).
"""
import sys
sys.path.insert(0, ".")
import golden_dw as dw

VEC = "vectors_unified.txt"
OUT = "out_unified.txt"

MODE_DENSE = 0
MODE_DW_SQ = 1
MODE_DW_SH = 2
MODE_DW_SV = 3
MODE_DW_DIL = 4
MODE_FC = 5

MAX_K = 11

def wrap32(v):
    v = v % (1 << 32)
    return v - (1 << 32) if v >= (1 << 31) else v

def dense_golden(w, h, k, shift, coeff121, img,
                 pt=0, pb=0, pl=0, pr=0, sx=1, sy=1,
                 sign_in=0, sign_out=0, fuse_act=0, act_slope=8192):
    pw = w + pl + pr
    ph = h + pt + pb
    ow = (pw - k) // sx + 1
    oh = (ph - k) // sy + 1
    out = []
    for orow in range(oh):
        for ocol in range(ow):
            acc = 0
            for i in range(k):
                for j in range(k):
                    rr = orow * sy + i - pt
                    cc = ocol * sx + j - pl
                    if 0 <= rr < h and 0 <= cc < w:
                        pv = img[rr * w + cc]
                        if sign_in and pv >= 128:
                            pv -= 256
                        acc += pv * coeff121[i * MAX_K + j]
            acc = wrap32(acc)
            rnd = acc + (1 << (shift - 1)) if shift > 0 else acc
            rnd = wrap32(rnd)
            v = rnd >> shift
            if fuse_act == 1:
                v = max(0, v)
            elif fuse_act == 2:
                v = v if v >= 0 else (v * act_slope) >> 16
            if sign_out:
                v = max(-128, min(127, v)) & 0xFF
            else:
                v = max(0, min(255, v))
            out.append(v)
    return out

def img_dense(w, h, seed=1):
    p = []
    for r in range(h):
        for c in range(w):
            p.append((r * 37 + c * 17 + seed * 11) % 256)
    return p

def img_signed(w, h):
    p = []
    for r in range(h):
        for c in range(w):
            p.append((r * 53 + c * 29 + 7) % 256)
    return p

def grid(kern, k):
    g = [0] * 121
    for i in range(k):
        for j in range(k):
            g[i * MAX_K + j] = kern[i * k + j]
    return g

CASES = []
def add(mode, **kw):
    CASES.append(dict(mode=mode, **kw))

def build_cases():
    global CASES
    CASES = []
    g5 = [1, 4, 6, 4, 1,
          4, 16, 24, 16, 4,
          6, 24, 36, 24, 6,
          4, 16, 24, 16, 4,
          1, 4, 6, 4, 1]
    # 0: dense 3x3 Sobel X, pad 1, unsigned.
    add(MODE_DENSE, k=3, shift=0, w=16, h=16,
        coeff=grid([-1, 0, 1, -2, 0, 2, -1, 0, 1], 3),
        img=img_dense(16, 16, 1),
        pt=1, pb=1, pl=1, pr=1)
    # 1: DW square 5x5 Gaussian, pad 2 (dense -> DW switch).
    add(MODE_DW_SQ, k=5, d=1, shift=8, w=16, h=16,
        coeff=dw.coeff_array(dw.MODE_SQUARE, 5, g5),
        img=img_dense(16, 16, 2),
        pt=2, pb=2, pl=2, pr=2)
    # 2: dense 3x3 Sobel X again, SAME coefficients as case 0, NO
    #    reload (tests dense preservation across the DW switch: the
    #    DW case uses the other submodule, so it cannot have touched
    #    the dense coefficients). Different image.
    add(MODE_DENSE, k=3, shift=0, w=16, h=16,
        coeff=None,  # no reload: reuse the dense submodule's coeffs
        img=img_dense(16, 16, 5),
        pt=1, pb=1, pl=1, pr=1)
    # 3: DW square 5x5 Gaussian again, SAME coefficients as case 1,
    #    NO reload (tests DW preservation across the dense switch:
    #    case 2 used the dense submodule only). Different image.
    add(MODE_DW_SQ, k=5, d=1, shift=8, w=16, h=16,
        coeff=None,  # no reload: reuse the DW submodule's coeffs
        img=img_dense(16, 16, 6),
        pt=2, pb=2, pl=2, pr=2)
    # 4: FC 1x1, W=H=1, signed in/out, fused ReLU (DW -> FC switch;
    #    the MLP path through the fused act). Single-channel FC here
    #    (the engine is PX=1); the multi-channel FC+ReLU is MC
    #    case 14. Input byte 200 = signed -56; coeff 256, shift 8:
    #    pre-ReLU -56, ReLU clips to 0 (without the fuse the signed
    #    output would be -56, emitted as byte 200).
    #    NOTE: FC shares the dense submodule/MAC, so loading the FC
    #    coefficients overwrites the dense coefficients from case 0
    #    (expected layer-switch behaviour: each layer loads its own
    #    weights; the two submodules' coefficients are preserved
    #    across switches, not the modes that share one submodule).
    fc_coeff = [0] * 121
    fc_coeff[0] = 256  # K=1 uses coeff[0] (tap (0,0) of the 1x1 grid)
    add(MODE_FC, k=1, shift=8, w=1, h=1,
        coeff=fc_coeff,
        img=[200],
        sign_in=1, sign_out=1, fuse_act=1)
    # 5: DW strip-H L=7, pad 3 (FC -> DW switch).
    add(MODE_DW_SH, k=7, d=1, shift=3, w=15, h=13,
        coeff=dw.coeff_array(dw.MODE_STRIP_H, 7, [1, 2, 3, 4, 3, 2, 1]),
        img=img_dense(15, 13, 3),
        pl=3, pr=3)
    # 6: DW dilated K=3 d=2, pad 2 (DW-strip -> DW-dilated switch).
    add(MODE_DW_DIL, k=3, d=2, shift=3, w=16, h=16,
        coeff=dw.coeff_array(dw.MODE_DILATED, 3, [1] * 9),
        img=img_dense(16, 16, 7),
        pt=2, pb=2, pl=2, pr=2)

def expected(cs):
    m = cs["mode"]
    if m in (MODE_DENSE, MODE_FC):
        coeff = cs["coeff"]
        if coeff is None:
            # case 4: reuse case 0's coefficients (no reload)
            coeff = CASES[0]["coeff"]
        return dense_golden(cs["w"], cs["h"], cs["k"], cs["shift"],
                            coeff, cs["img"],
                            cs.get("pt", 0), cs.get("pb", 0),
                            cs.get("pl", 0), cs.get("pr", 0),
                            cs.get("sx", 1), cs.get("sy", 1),
                            cs.get("sign_in", 0), cs.get("sign_out", 0),
                            cs.get("fuse_act", 0), cs.get("act_slope", 8192))
    else:
        dwmode = {MODE_DW_SQ: dw.MODE_SQUARE, MODE_DW_SH: dw.MODE_STRIP_H,
                  MODE_DW_SV: dw.MODE_STRIP_V, MODE_DW_DIL: dw.MODE_DILATED}[m]
        coeff = cs["coeff"]
        if coeff is None:
            # case 4: reuse case 1's coefficients (no reload)
            coeff = CASES[1]["coeff"]
        dcs = dict(mode=dwmode, k=cs["k"], d=cs.get("d", 1),
                   shift=cs["shift"], coeff=coeff, img=cs["img"],
                   w=cs["w"], h=cs["h"],
                   pt=cs.get("pt", 0), pb=cs.get("pb", 0),
                   pl=cs.get("pl", 0), pr=cs.get("pr", 0),
                   sx=cs.get("sx", 1), sy=cs.get("sy", 1),
                   sign_in=cs.get("sign_in", 0),
                   sign_out=cs.get("sign_out", 0))
        return dw.golden(dcs)

def gen():
    build_cases()
    with open(VEC, "w") as f:
        f.write("%d\n" % len(CASES))
        for cs in CASES:
            m = cs["mode"]
            ncoeff = 0 if cs.get("coeff") is None else len(cs["coeff"])
            f.write("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n" %
                    (m, cs["k"], cs["shift"], cs["w"], cs["h"],
                     cs.get("pt", 0), cs.get("pb", 0),
                     cs.get("pl", 0), cs.get("pr", 0),
                     cs.get("sx", 1), cs.get("sy", 1),
                     cs.get("sign_in", 0), cs.get("sign_out", 0),
                     cs.get("fuse_act", 0), cs.get("act_slope", 8192),
                     cs.get("d", 1)))
            f.write("%d\n" % ncoeff)
            if ncoeff:
                f.write(" ".join(str(v) for v in cs["coeff"]) + "\n")
            f.write(" ".join(str(v) for v in cs["img"]) + "\n")
    print("wrote %d cases to %s" % (len(CASES), VEC))

def check():
    build_cases()
    got = [int(x) for x in open(OUT).read().split()]
    pos = 0
    errs = 0
    names = ["dense", "DW-square", "DW-strip-H", "DW-strip-V",
             "DW-dilated", "FC"]
    for ci, cs in enumerate(CASES):
        exp = expected(cs)
        n = len(exp)
        have = got[pos:pos + n]
        pos += n
        if len(have) < n:
            print("case %d (%s): MISSING outputs: got %d, expected %d" %
                  (ci, names[cs["mode"]], len(have), n))
            errs += n - len(have)
            continue
        bad = 0
        for i, (a, b) in enumerate(zip(have, exp)):
            if a != b:
                bad += 1
                if bad <= 3:
                    print("case %d px %d: got %d expected %d" % (ci, i, a, b))
        print("case %d (%s %dx%d): %s (%d pixels)" %
              (ci, names[cs["mode"]], cs["w"], cs["h"],
               "PASS" if bad == 0 else "FAIL (%d bad)" % bad, n))
        errs += bad
    if pos < len(got):
        print("WARNING: %d extra outputs" % (len(got) - pos))
    if errs == 0:
        print("GOLDEN CHECK PASSED")
    else:
        print("GOLDEN CHECK FAILED (%d errors)" % errs)
    return 1 if errs else 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        gen()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_unified.py gen|check")
