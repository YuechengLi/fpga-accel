#!/usr/bin/env python3
"""Golden model + vector generation + result check for act (Stage 3).

Modes (match rtl/act.sv):
  0 = ReLU:      max(0, x)                     (signed 32-bit in/out)
  1 = Leaky:     x if x>=0 else (x * slope_fp) >> 16
                 slope_fp is a 16-bit fixed-point slope (Q0.16)
  2 = PWL sigmoid, 3 = PWL tanh, 4 = PWL GELU, 5 = PWL SiLU
                 PWL table: 32 segments over [-8, 8], width 0.5.
                 Input is the signed 16-bit Q5.11 value in the low 16
                 bits of the 32-bit input; output is a signed 16-bit
                 Q1.15 value sign-extended to 32 bits.
                 y0 = f(x0) in Q1.15; dslope = f(x1) - f(x0) in Q1.15;
                 y = y0 + ((dslope * frac) >> 10), frac in [0, 1024).
                 Outside [-8, 8] the output clamps to f(-8) / f(8).

The vector file (vectors_act.txt) carries the PWL table explicitly, so
the testbench loads exactly what the golden used (the table is a
config-loaded coefficient, not a hardcoded constant):
  line 1: ncases
  per case: mode slope_fp nvals
            then (for PWL modes) 32 lines: y0 dslope   (Q1.15, signed)
            then nvals input values (signed 32-bit), one per line
Output (out_act.txt): one output value per line (signed 32-bit).
Usage: golden_act.py [gen|check]
"""
import math
import sys

VEC = "vectors_act.txt"
OUT = "out_act.txt"
NSEG = 32
X0 = -8.0
WID = 0.5


def f_sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def f_tanh(x):
    return math.tanh(x)


def f_gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def f_silu(x):
    return x / (1.0 + math.exp(-x))


FUNCS = {2: f_sigmoid, 3: f_tanh, 4: f_gelu, 5: f_silu}
FNAMES = {2: "sigmoid", 3: "tanh", 4: "GELU", 5: "SiLU"}


def q15(v):
    """Round to Q1.15 (range [-1, 1) representable; clamp to [-32768, 32767])."""
    return max(-32768, min(32767, int(round(v * 32768.0))))


def pwl_table(mode):
    """Return (y0_list, dslope_list) as Q1.15 integers, 32 segments."""
    fn = FUNCS[mode]
    y0, ds = [], []
    for s in range(NSEG):
        xa = X0 + s * WID
        xb = xa + WID
        ya, yb = fn(xa), fn(xb)
        y0.append(q15(ya))
        ds.append(q15(yb - ya))
    return y0, ds


def pwl_golden(mode, x_q511):
    """Golden PWL evaluation, mirroring the RTL exactly."""
    y0, ds = pwl_table(mode)
    x = x_q511  # signed Q5.11, integer (x * 2048)
    if x <= -16384:      # x <= -8.0
        return y0[0]
    if x >= 16384:       # x >= 8.0: clamp to f(8) = y0[31] + ds[31]
        return y0[31] + ds[31]
    seg = (x + 16384) >> 10
    if seg > 31:
        seg = 31
    frac = (x + 16384) & 1023
    return y0[seg] + ((ds[seg] * frac) >> 10)


def make_cases():
    cases = []

    def add(mode, slope_fp, vals):
        cs = dict(mode=mode, slope_fp=slope_fp, vals=vals)
        if mode >= 2:
            cs["y0"], cs["ds"] = pwl_table(mode)
        cases.append(cs)

    # ReLU: signed 32-bit inputs incl. negatives and large positives
    add(0, 0, [-1000000, -255, -1, 0, 1, 127, 255, 1000, 2**31 - 1,
               -2**31, 123456, -654321])
    # Leaky, slope 0.1 (Q0.16 = 6554)
    add(1, 6554, [-100000, -1000, -255, -10, -1, 0, 1, 255, 1000, 100000,
                  -2**31 + 1, 2**31 - 1])
    # Leaky, slope 0.01 (Q0.16 = 655)
    add(1, 655, [-100000, -1000, -255, -1, 0, 1, 255, 100000])
    # PWL sigmoid: Q5.11 inputs from -10 to +10 in 0.25 steps + edges
    q = []
    for i in range(-40, 41):
        q.append(i * 512)          # 0.25 steps in Q5.11
    q += [-20480, 20480, -16384, 16383, 16384, 0]  # out-of-range + edges
    add(2, 0, q)
    # PWL tanh: same grid
    add(3, 0, q)
    # PWL GELU: same grid
    add(4, 0, q)
    # PWL SiLU: same grid
    add(5, 0, q)
    return cases


def golden(cs):
    mode = cs["mode"]
    out = []
    for v in cs["vals"]:
        if mode == 0:
            out.append(max(0, v))
        elif mode == 1:
            out.append(v if v >= 0 else (v * cs["slope_fp"]) >> 16)
        else:
            # PWL: input is the low 16 bits (Q5.11); sign-extend them
            xv = v & 0xFFFF
            if xv >= 0x8000:
                xv -= 0x10000
            out.append(pwl_golden(mode, xv))
    return out


def gen():
    cases = make_cases()
    with open(VEC, "w") as f:
        f.write(f"{len(cases)}\n")
        for cs in cases:
            f.write(f"{cs['mode']} {cs['slope_fp']} {len(cs['vals'])}\n")
            if cs["mode"] >= 2:
                for s in range(NSEG):
                    f.write(f"{cs['y0'][s]} {cs['ds'][s]}\n")
            for v in cs["vals"]:
                f.write(f"{v}\n")
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
        bad = 0
        for i, (e, g) in enumerate(zip(exp, seg)):
            if e != g:
                bad += 1
                if bad <= 5:
                    print(f"case {idx} val {i} (in={cs['vals'][i]}): "
                          f"expected {e}, got {g}")
        name = FNAMES.get(cs["mode"], "ReLU" if cs["mode"] == 0 else "Leaky")
        if bad:
            nerr += bad
            print(f"case {idx} ({name}): {bad}/{n} mismatches")
        else:
            print(f"case {idx} ({name}): PASS ({n} values)")
    if len(got) > pos:
        print(f"WARNING: {len(got) - pos} extra outputs in {OUT}")
        nerr += len(got) - pos
    print("GOLDEN CHECK " + ("PASSED" if nerr == 0 else f"FAILED ({nerr} errors)"))
    return 0 if nerr == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gen", "check"):
        print("usage: golden_act.py [gen|check]")
        sys.exit(2)
    if sys.argv[1] == "gen":
        gen()
    else:
        sys.exit(check())
