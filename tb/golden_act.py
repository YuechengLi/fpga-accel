#!/usr/bin/env python3
"""Golden model + vector generation + result check for act (Stage 3).

Modes (match rtl/vision/act.sv):
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
  6 = MULTI-LEVEL LUT (Stage 4): two-level table, compiler-generated.
                 Input is a 12-bit offset-binary value in the low 12
                 bits (in12 = round((x+8)*256), x in [-8, 8)). Coarse:
                 top H1 bits -> base + fine-table select. Fine: next H2
                 bits -> delta. Output = base + delta (saturating),
                 Q5.11, sign-extended to 32 bits. Coarse segments are
                 grouped into 4 bands by local slope; each band gets a
                 fine table (non-uniform split). See mlut_table().

The vector file (vectors_act.txt) carries the PWL table explicitly, so
the testbench loads exactly what the golden used (the table is a
config-loaded coefficient, not a hardcoded constant):
  line 1: ncases
  per case (modes 0,1): mode slope_fp nvals, then nvals input values
  per case (PWL modes 2..5): mode slope_fp nvals,
            then 32 lines: y0 dslope   (Q1.15, signed),
            then nvals input values (signed 32-bit), one per line
  per case (MLUT mode 6): 6 func nvals, then "H1 H2 nbands",
            then 2^H1 lines: coarse base (Q5.11, signed),
            then 2^H1 lines: coarse select (fine table id 0..nbands-1),
            then nbands * 2^H2 lines: fine tables (t=0..nbands-1,
                idx=0..2^H2-1, Q5.11),
            then nvals input values (12-bit offset-binary), one per line
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


def q511(v):
    """Round to Q5.11 (range [-16, 16), clamp to [-32768, 32767])."""
    return max(-32768, min(32767, int(round(v * 2048.0))))


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


# ---------------------------------------------------------------------------
# Multi-level LUT (mode 6, Stage 4). Compiler-style table generation.
# Input is a 12-bit offset-binary value: in12 = round((x + 8) * 256),
# x in [-8, 8), so in12 in [0, 4095]. Coarse: top H1 bits -> 2^H1
# segments, each (16 / 2^H1) wide in x. Fine: next H2 bits -> 2^H2
# sub-steps per segment. Output is Q5.11 (16-bit signed), same width as
# the PWL output but a wider range (GELU/SiLU reach ~8, which Q1.15
# cannot hold).
# Non-uniform split: coarse segments are grouped into 4 bands by local
# slope (|slope| sorted, contiguous bands); each band gets one fine
# table, generated from the band's representative segment (the segment
# whose slope is closest to the band mean). Steep bands get a fine table
# matched to their rapid variation, flat bands one matched to theirs.
# The RTL computes base + delta from these tables; this golden mirrors
# that computation exactly (so the check is exact), and the approximation
# error vs the true float function is reported separately (see
# mlut_error_report).
# ---------------------------------------------------------------------------
MLUT_H1 = 8
MLUT_H2 = 4
MLUT_NBANDS = 8


def mlut_table(func_mode, H1=MLUT_H1, H2=MLUT_H2, nbands=MLUT_NBANDS):
    """Generate a two-level MLUT for the given function (func_mode 2..5).
    Returns (H1, H2, coarse_base, coarse_sel, fine_tables).
    coarse_base: 2^H1 entries, Q5.11 base = f(segment start).
    coarse_sel:  2^H1 entries, fine table id (0..nbands-1) per segment.
    fine_tables: nbands tables x 2^H2 entries, Q5.11 MEAN delta across
                 the band's segments (least-squares fit; the max error
                 is then about half the spread of deltas in the band).
    Accuracy budget: the split is chosen so the MLUT's max error vs
    float is comparable to the PWL mode's (~0.024 for tanh, 32 seg).
    H1=8/H2=4/8 bands gives max ~0.027 (tanh 0.019, GELU 0.027, SiLU
    0.023, sigmoid 0.003); H1=6/H2=6/4 bands gives max 0.117 (tanh) —
    too inaccurate to ship. Table: 256 coarse + 8 x 16 fine = 384
    entries, still 10x smaller than a flat 4096 table.
    """
    fn = FUNCS[func_mode]
    NC = 1 << H1
    NF = 1 << H2
    seg_w = 16.0 / NC
    base = []
    slope = []
    for s in range(NC):
        x0 = -8.0 + s * seg_w
        x1 = x0 + seg_w
        base.append(q511(fn(x0)))
        slope.append((fn(x1) - fn(x0)) / seg_w)
    # Group segments into nbands bands by |slope| (sorted, contiguous).
    order = sorted(range(NC), key=lambda s: abs(slope[s]))
    band_of = [0] * NC
    for b in range(nbands):
        lo = b * (NC // nbands)
        hi = (b + 1) * (NC // nbands) if b < nbands - 1 else NC
        for s in order[lo:hi]:
            band_of[s] = b
    fine_tables = []
    for b in range(nbands):
        members = [s for s in range(NC) if band_of[s] == b]
        if not members:
            fine_tables.append([0] * NF)
            continue
        ft = []
        for j in range(NF):
            # Mean delta across the band's segments.
            d = sum(fn(-8.0 + s * seg_w + (j / NF) * seg_w)
                    - fn(-8.0 + s * seg_w) for s in members) / len(members)
            ft.append(q511(d))
        fine_tables.append(ft)
    return H1, H2, base, band_of, fine_tables


def mlut_golden(func_mode, in12, H1, H2, base, sel, fine_tables):
    """Mirror the RTL MLUT computation exactly (base + delta, saturating)."""
    coarse_idx = in12 >> (12 - H1)
    fine_idx = (in12 >> (12 - H1 - H2)) & ((1 << H2) - 1)
    b = base[coarse_idx]
    t = sel[coarse_idx]
    d = fine_tables[t][fine_idx]
    s = b + d
    if s > 32767:
        s = 32767
    if s < -32768:
        s = -32768
    return s


def mlut_error_report(func_mode, H1, H2, base, sel, fine_tables):
    """Max / RMS error of the MLUT approximation vs the true float
    function, over all 4096 possible 12-bit inputs. Returns (max_abs_q511,
    rms_q511, max_abs_float)."""
    fn = FUNCS[func_mode]
    errs = []
    for in12 in range(4096):
        x = -8.0 + in12 / 256.0
        got_q = mlut_golden(func_mode, in12, H1, H2, base, sel, fine_tables)
        got = got_q / 2048.0
        errs.append(got - fn(x))
    mx = max(abs(e) for e in errs)
    rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
    return mx * 2048.0, rms * 2048.0, mx


def make_cases():
    cases = []

    def add(mode, slope_fp, vals):
        cs = dict(mode=mode, slope_fp=slope_fp, vals=vals)
        if 2 <= mode <= 5:
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
    # Multi-level LUT (mode 6): one case per function, all 4096 possible
    # 12-bit inputs (offset-binary, in12 = round((x+8)*256)). The table
    # is compiler-generated (slope-grouped fine tables, H1=8, H2=4,
    # 8 bands).
    for func_mode in (2, 3, 4, 5):
        H1, H2, base, sel, fine = mlut_table(func_mode)
        cs = dict(mode=6, func=func_mode, slope_fp=0,
                  vals=list(range(4096)), H1=H1, H2=H2,
                  mlut_base=base, mlut_sel=sel, mlut_fine=fine)
        cases.append(cs)
    # BYPASS (mode 7, Stage 4): output = input exactly, including
    # negatives and extremes (used when the activation is fused into
    # conv, or when the layer has no activation at all).
    add(7, 0, [-1000000, -255, -1, 0, 1, 127, 255, 1000, 2**31 - 1,
               -2**31, 123456, -654321])
    return cases


def golden(cs):
    mode = cs["mode"]
    out = []
    for v in cs["vals"]:
        if mode == 0:
            out.append(max(0, v))
        elif mode == 1:
            out.append(v if v >= 0 else (v * cs["slope_fp"]) >> 16)
        elif mode == 6:
            # Multi-level LUT: input is the 12-bit offset-binary value.
            out.append(mlut_golden(cs["func"], v, cs["H1"], cs["H2"],
                                   cs["mlut_base"], cs["mlut_sel"],
                                   cs["mlut_fine"]))
        elif mode == 7:
            # BYPASS: output = input unchanged.
            out.append(v)
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
            if cs["mode"] == 6:
                f.write(f"6 {cs['func']} {len(cs['vals'])}\n")
                f.write(f"{cs['H1']} {cs['H2']} {MLUT_NBANDS}\n")
                for v in cs["mlut_base"]:
                    f.write(f"{v}\n")
                for v in cs["mlut_sel"]:
                    f.write(f"{v}\n")
                for t in range(MLUT_NBANDS):
                    for v in cs["mlut_fine"][t]:
                        f.write(f"{v}\n")
            else:
                f.write(f"{cs['mode']} {cs['slope_fp']} {len(cs['vals'])}\n")
                if 2 <= cs["mode"] <= 5:
                    for s in range(NSEG):
                        f.write(f"{cs['y0'][s]} {cs['ds'][s]}\n")
            for v in cs["vals"]:
                f.write(f"{v}\n")
    print(f"wrote {VEC}: {len(cases)} cases")
    # Report the MLUT approximation error vs the true float function.
    for cs in cases:
        if cs["mode"] == 6:
            mx_q, rms_q, mx_f = mlut_error_report(
                cs["func"], cs["H1"], cs["H2"], cs["mlut_base"],
                cs["mlut_sel"], cs["mlut_fine"])
            print(f"MLUT {FNAMES[cs['func']]}: approx error vs float: "
                  f"max {mx_q:.1f} Q5.11 LSBs ({mx_f:.4f}), "
                  f"RMS {rms_q:.1f} LSBs")


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
        if cs["mode"] == 6:
            name = "MLUT-" + FNAMES[cs["func"]]
        if cs["mode"] == 7:
            name = "BYPASS"
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
