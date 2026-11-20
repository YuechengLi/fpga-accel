#!/usr/bin/env python3
"""Golden model + vector generator / checker for layernorm.sv (Stage 4).

Fixed-point spec (mirrors the RTL exactly):
  mean = sum(x) >> log2(N)                     (arithmetic shift, floor)
  var  = (sumsq >> log2(N)) - mean*mean, clamp >= 0
  v    = var + eps
  inv_std_q = LUT[min((v >> isqrt_shift), 255)] (Q1.15, 16-bit)
  LUT[i] = round(1/sqrt((i << isqrt_shift) + (1 << (isqrt_shift-1))) * 32768)
           clamped to [0, 32767]. (Bin centre of the LUT bin.)
  per pixel: diff = x - mean
             norm_q = (diff * inv_std_q) >> 7   (Q8.8, arithmetic shift)
             y = (norm_q * gamma_q) >> 23 + beta (arithmetic shift)
             saturate to [0,255] (unsigned) or [-128,127] (signed).
Python >> on negatives is a floor (arithmetic) shift, matching RTL >>>.
The error report compares the fixed-point output against the true float
LayerNorm (mean_f, var_f, inv_std_f = 1/sqrt(var_f+eps)).

Vector format (one case per line, fields separated by spaces):
  W H sign isqrt_shift eps gamma_q beta  nvals  in0 in1 ...  [256 LUT lines]
  The LUT (256 lines of one integer each) follows the header line.
  Input pixels are byte values (0..255); for signed input, byte>=128
  means a negative int8.
Output file: out_layernorm.txt (one byte per line, in order).
"""
import math, random, sys

VEC = "vectors_layernorm.txt"
OUT = "out_layernorm.txt"

def isqrt_lut(isqrt_shift):
    lut = []
    for i in range(256):
        vc = (i << isqrt_shift) + (1 << (isqrt_shift - 1))
        v = 1.0 / math.sqrt(vc)
        q = int(round(v * 32768.0))
        lut.append(max(0, min(32767, q)))
    return lut

def layernorm_golden(px, W, H, sign_in, sign_out, isqrt_shift, eps,
                     gamma_q, beta, lut):
    N = W * H
    log_n = (N).bit_length() - 1
    xs = [p - 256 if (sign_in and p >= 128) else p for p in px]
    s = sum(xs)
    ss = sum(x * x for x in xs)
    mean = s >> log_n
    var = (ss >> log_n) - mean * mean
    if var < 0:
        var = 0
    v = var + eps
    idx = min((v >> isqrt_shift), 255)
    inv_std = lut[idx]
    out = []
    for x in xs:
        diff = x - mean
        norm_q = (diff * inv_std) >> 7
        y = ((norm_q * gamma_q) >> 23) + beta
        if sign_out:
            y = max(-128, min(127, y))
            out.append(y & 0xFF)
        else:
            y = max(0, min(255, y))
            out.append(y)
    return out, mean, var, inv_std

def layernorm_float(px, W, H, sign_in, sign_out, eps, gamma_q, beta):
    xs = [p - 256 if (sign_in and p >= 128) else p for p in px]
    N = len(xs)
    mf = sum(xs) / N
    vf = sum((x - mf) ** 2 for x in xs) / N
    inv = 1.0 / math.sqrt(vf + eps)
    gf = gamma_q / 32768.0
    out = []
    for x in xs:
        y = (x - mf) * inv * gf + beta
        if sign_out:
            y = max(-128, min(127, y))
        else:
            y = max(0, min(255, y))
        out.append(y)
    return out

CASES = []
def add(W, H, sign_in, sign_out, isqrt_shift, eps, gamma_q, beta, img):
    CASES.append((W, H, sign_in, sign_out, isqrt_shift, eps, gamma_q,
                  beta, list(img)))

def build_cases():
    global CASES
    CASES = []
    random.seed(42)
    # 0: 8x8 unsigned, gamma~1, beta 0
    add(8, 8, 0, 0, 6, 1, 32767, 0,
        [random.randint(0, 255) for _ in range(64)])
    # 1: 16x16 unsigned, gamma 0.5, beta 10
    add(16, 16, 0, 0, 6, 1, 16384, 10,
        [random.randint(0, 255) for _ in range(256)])
    # 2: 8x8 signed in/out, gamma~1, beta 0 (values centred near 0)
    add(8, 8, 1, 1, 6, 1, 32767, 0,
        [random.randint(96, 160) for _ in range(64)])
    # 3: 4x4 unsigned, constant frame (var=0) -- inv_std clamps, out=beta
    add(4, 4, 0, 0, 6, 1, 32767, 5, [100] * 16)
    # 4: 16x16 unsigned, gamma 2.0 (Q1.15 wraps: use 32767*2 not allowed;
    #     gamma is Q1.15 so max ~1.0; use gamma=0.25, beta=-5)
    add(16, 16, 0, 0, 6, 1, 8192, -5,
        [random.randint(0, 255) for _ in range(256)])
    # 5: 8x16 unsigned, low contrast (small var)
    add(8, 16, 0, 0, 6, 1, 32767, 0,
        [128 + random.randint(-3, 3) for _ in range(128)])

def gen():
    build_cases()
    with open(VEC, "w") as f:
        f.write("%d\n" % len(CASES))
        for (W, H, si, so, ish, eps, gq, beta, img) in CASES:
            lut = isqrt_lut(ish)
            f.write("%d %d %d %d %d %d %d %d %d" %
                    (W, H, si, so, ish, eps, gq, beta, len(img)))
            for p in img:
                f.write(" %d" % p)
            f.write("\n")
            for e in lut:
                f.write("%d\n" % e)
    print("wrote %d cases to %s" % (len(CASES), VEC))

def check():
    build_cases()
    got = [int(x) for x in open(OUT).read().split()]
    pos = 0
    errs = 0
    for ci, (W, H, si, so, ish, eps, gq, beta, img) in enumerate(CASES):
        lut = isqrt_lut(ish)
        exp, mean, var, inv_std = layernorm_golden(
            img, W, H, si, so, ish, eps, gq, beta, lut)
        n = len(exp)
        have = got[pos:pos + n]
        pos += n
        if len(have) < n:
            print("case %d: MISSING outputs: got %d, expected %d" %
                  (ci, len(have), n))
            errs += n - len(have)
            continue
        bad = 0
        for i, (a, b) in enumerate(zip(have, exp)):
            if a != b:
                bad += 1
                if bad <= 3:
                    print("case %d px %d: got %d expected %d" %
                          (ci, i, a, b))
        # error vs float
        fl = layernorm_float(img, W, H, si, so, eps, gq, beta)
        maxe = 0.0
        for a, b in zip(exp, fl):
            # compare fixed-point byte to float value (as signed if signed)
            av = a - 256 if (so and a >= 128) else a
            maxe = max(maxe, abs(av - b))
        print("case %d (%dx%d sign=%d/%d gamma=%.3f beta=%d): %s "
              "(%d pixels) mean=%d var=%d inv_std_q=%d "
              "max|fixed-float|=%.2f" %
              (ci, W, H, si, so, gq / 32768.0, beta,
               "PASS" if bad == 0 else "FAIL (%d bad)" % bad,
               n, mean, var, inv_std, maxe))
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
        print("usage: golden_layernorm.py gen|check")
