#!/usr/bin/env python3
"""Stage 5 compiler: multi-level LUT table generation.

The compiler decides, per activation, the number of levels, the
bit-splits and the table contents, driven by the function's curvature
and the accuracy budget. The hardware (act.sv mode 6) implements TWO
levels with configurable splits: level 1 (coarse, high H1 bits) gives
a base value and selects a fine table; level 2 (fine, next H2 bits)
gives a delta; output = base + delta (an ADD, no multiply). The
table generation below reproduces the Stage 4 golden method exactly
(tb/golden_act.py mlut_table / mlut_golden), so the compiler's tables
are the same ones Stage 4 verified in RTL:

  * coarse base[s] = Q5.11 f(segment start), 2^H1 segments over [-8, 8)
  * coarse segments are grouped into ``nbands`` bands by |slope|
    (sorted, contiguous) -- a NON-UNIFORM split: steep regions get a
    fine table matched to their rapid variation, flat regions one
    matched to theirs
  * fine table[b][j] = Q5.11 MEAN delta across the band's segments
    (least-squares fit; the max error is then about half the spread
    of deltas in the band)

The compiler SEARCHES (H1, H2, nbands) for the smallest total table
(2^H1 coarse + nbands * 2^H2 fine entries) whose max error vs float
meets the budget. The Stage 4 golden split (H1=8, H2=4, 8 bands,
384 entries) is the reference point: max error sigmoid 0.0027,
tanh 0.0186, GELU 0.0269, SiLU 0.0234.

Levels: the hardware implements 2 levels (mode 6). A 3rd/4th level
would need the hardware extension noted in act.sv (the level-2
delta table becomes a two-level lookup); the table generator is
structured so the level count is a parameter, but 3/4-level tables
are NOT emitted for the current act.sv -- they are recorded as a
follow-up, not silently treated as executable. The compiler therefore
decides the level count as 2 (with the split chosen per function)
until the hardware extension lands.
"""
import numpy as np
from passes.actapprox import FUNCS


def q511(v):
    """Round to Q5.11 (range [-16, 16), clamp to [-32768, 32767])."""
    return max(-32768, min(32767, int(round(float(v) * 2048.0))))


def mlut_table(func_name, H1=8, H2=4, nbands=8):
    """Generate a two-level MLUT for ``func_name`` (Stage 4 method).
    Returns (H1, H2, coarse_base, coarse_sel, fine_tables):
      coarse_base: 2^H1 Q5.11 ints (f at the segment start)
      coarse_sel:  2^H1 fine-table ids (0..nbands-1) per segment
      fine_tables: nbands x 2^H2 Q5.11 ints (mean delta per band)
    """
    fn = FUNCS[func_name]
    NC = 1 << H1
    NF = 1 << H2
    seg_w = 16.0 / NC
    base = []
    slope = []
    for s in range(NC):
        x0 = -8.0 + s * seg_w
        x1 = x0 + seg_w
        base.append(q511(fn(x0)))
        slope.append((float(fn(x1)) - float(fn(x0))) / seg_w)
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
            d = sum(float(fn(-8.0 + s * seg_w + (j / NF) * seg_w))
                    - float(fn(-8.0 + s * seg_w)) for s in members) \
                / len(members)
            ft.append(q511(d))
        fine_tables.append(ft)
    return H1, H2, base, band_of, fine_tables


def mlut_eval(in12, H1, H2, base, sel, fine_tables):
    """Mirror the RTL MLUT computation exactly (base + delta,
    saturating). in12 is a 12-bit offset-binary input
    (in12 = round((x+8)*256), x in [-8, 8)). Returns a Q5.11 int."""
    coarse_idx = in12 >> (12 - H1)
    fine_idx = (in12 >> (12 - H1 - H2)) & ((1 << H2) - 1)
    b = base[coarse_idx]
    t = sel[coarse_idx]
    d = fine_tables[t][fine_idx]
    s = b + d
    return max(-32768, min(32767, s))


def mlut_eval_float(x, H1, H2, base, sel, fine_tables):
    """MLUT output as a float (Q5.11 / 2048) for one input x."""
    in12 = int(round((float(x) + 8.0) * 256.0))
    in12 = max(0, min(4095, in12))
    return mlut_eval(in12, H1, H2, base, sel, fine_tables) / 2048.0


def mlut_error(func_name, H1, H2, base, sel, fine_tables):
    """Max / RMS error of the MLUT vs the true float function over
    all 4096 possible 12-bit inputs. Returns (max_abs, rms)."""
    fn = FUNCS[func_name]
    errs = []
    for in12 in range(4096):
        x = -8.0 + in12 / 256.0
        got = mlut_eval(in12, H1, H2, base, sel, fine_tables) / 2048.0
        errs.append(got - float(fn(x)))
    mx = max(abs(e) for e in errs)
    rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
    return mx, rms


def mlut_search(func_name, err_budget=0.03):
    """Search (H1, H2, nbands) for the SMALLEST total table
    (2^H1 + nbands * 2^H2 entries) whose max error meets the budget.

    Returns (H1, H2, nbands, base, sel, fine, max_err, total_entries).
    Raises RuntimeError if no split in the search space meets the
    budget (the caller must not silently ship an over-budget table).
    """
    candidates = []
    for H1 in (8, 7, 6, 5, 4):
        for H2 in (6, 5, 4, 3, 2):
            if H1 + H2 > 12:
                continue
            for nbands in (8, 4, 2, 1):
                total = (1 << H1) + nbands * (1 << H2)
                candidates.append((total, H1, H2, nbands))
    # Smallest table first; prefer more bands (finer non-uniformity)
    # at the same size.
    candidates.sort(key=lambda c: (c[0], -c[3]))
    for total, H1, H2, nbands in candidates:
        h1, h2, base, sel, fine = mlut_table(func_name, H1, H2, nbands)
        mx, _rms = mlut_error(func_name, h1, h2, base, sel, fine)
        if mx <= err_budget:
            return h1, h2, nbands, base, sel, fine, mx, total
    raise RuntimeError(
        f"mlut_search: no (H1,H2,nbands) in the search space meets "
        f"err_budget={err_budget} for {func_name} (smallest table "
        f"tried: {candidates[0][0]} entries); increase the budget or "
        f"extend the hardware (3-level LUT)")


def mlut_config_writes(func_name, H1, H2, base, sel, fine_tables):
    """Emit the act.sv config-bus writes that load this MLUT table.
    Returns a list of (addr, data) 16-bit pairs:
      0x02 = MLUT control (H1 in bits[3:0], H2 in bits[7:4])
      0x200+idx = coarse base, 0x300+idx = coarse select,
      0x400+t*64+idx = fine table t.
    """
    writes = [(0x02, (H1 & 0xF) | ((H2 & 0xF) << 4))]
    for i, v in enumerate(base):
        writes.append((0x200 + i, v & 0xFFFF))
    for i, v in enumerate(sel):
        writes.append((0x300 + i, v & 0x7))
    for t, ft in enumerate(fine_tables):
        for j, v in enumerate(ft):
            writes.append((0x400 + t * 64 + j, v & 0xFFFF))
    return writes
