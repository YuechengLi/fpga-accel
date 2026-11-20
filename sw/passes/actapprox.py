#!/usr/bin/env python3
"""Stage 5 compiler: activation approximation pass.

The compiler decides, per activation, how to approximate it in hardware:

  * PWL (act.sv mode 2): 32 UNIFORM segments over [-8, 8]. The segment
    boundaries are fixed by the hardware (the segment index is the
    input's high bits), so the compiler cannot choose non-uniform PWL
    boundaries. What the compiler does: compute y0 / dslope (Q1.15)
    per segment, verify the max error on a dense grid, and report
    whether the 32-segment PWL meets the accuracy budget. If it does
    not, the compiler selects the multi-level LUT instead (passes/mlut).
  * Cheap substitutes: hard-sigmoid, hard-tanh, GELU->SiLU, GELU->ReLU.
    A substitute changes the model math, so it is only accepted when
    the END-TO-END error stays in budget (validated by the functional
    simulator, not just by the point error). The pass provides the
    substitute and its point error; acceptance is decided end-to-end.
  * Range clipping: sigmoid/tanh saturate (|x| > 8 for sigmoid,
    |x| > 5 for tanh the output is within 1e-3 of the asymptote), so
    the table only needs to be accurate inside the calibrated input
    range; outside it the output is the saturated value. The pass
    records the calibrated range (measured by the quantization pass
    on the calibration input) and the saturated values.

Accuracy constraint: the approximation error must be within the
per-layer error budget, verified by the functional simulator +
fixed-point golden (Stage 6). The pass never silently accepts an
approximation that exceeds the budget.
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Activation functions (float, numpy).
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def tanh(x):
    return np.tanh(np.asarray(x, dtype=np.float64))


def gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * sigmoid(x)


def hard_sigmoid(x):
    """Hard sigmoid: clip((x+3)/6, 0, 1)."""
    return np.clip((np.asarray(x, dtype=np.float64) + 3.0) / 6.0, 0.0, 1.0)


def hard_tanh(x):
    """Hard tanh: clip(x, -1, 1)."""
    return np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)


FUNCS = {"sigmoid": sigmoid, "tanh": tanh, "gelu": gelu, "silu": silu,
         "hard_sigmoid": hard_sigmoid, "hard_tanh": hard_tanh}

# PWL hardware constants (act.sv mode 2): 32 uniform segments over
# [-8, 8]; y0 and dslope are signed 16-bit Q1.15.
PWL_SEGMENTS = 32
PWL_XMIN, PWL_XMAX = -8.0, 8.0


def q115(v):
    """Quantize a float to signed 16-bit Q1.15 (saturating)."""
    q = int(round(float(v) * 32768.0))
    return max(-32768, min(32767, q))


def pwl_table(func_name, err_budget=0.03):
    """Generate the 32-segment PWL table for ``func_name``.

    Returns (y0, dslope, max_err, meets_budget) where y0/dslope are
    lists of 32 Q1.15 ints (act.sv config 0x100+seg*2 = y0,
    0x100+seg*2+1 = dslope), max_err is the max |error| vs the float
    function on a 4096-point grid over [-8, 8], and meets_budget is
    whether max_err <= err_budget.
    """
    fn = FUNCS[func_name]
    seg_w = (PWL_XMAX - PWL_XMIN) / PWL_SEGMENTS
    y0, dslope = [], []
    for s in range(PWL_SEGMENTS):
        x0 = PWL_XMIN + s * seg_w
        x1 = x0 + seg_w
        y0.append(q115(fn(x0)))
        # dslope is the slope in Q1.15 units per unit dx, where dx is
        # in Q5.11 (0..1023 per segment): y = y0 + (dslope*dx)>>10,
        # so dslope = (f(x1)-f(x0)) / seg_w * (seg_w in Q5.11 units
        # is 1024) -> dslope_q115 = (f(x1)-f(x0)) * 1024 / seg_w / 32768
        # in Q1.15. Equivalently dslope = (f(x1)-f(x0)) / seg_w,
        # quantized to Q1.15 (the RTL multiplies by dx (0..1023) and
        # shifts right 10, i.e. dx/1024 * dslope = slope * dx_q511).
        dslope.append(q115(float(fn(x1)) - float(fn(x0))))
    # Verify on a dense grid (4096 points, matching the Stage 4 golden).
    xs = np.linspace(PWL_XMIN, PWL_XMAX, 4096, endpoint=False)
    got = pwl_eval(func_name, xs, y0, dslope)
    max_err = float(np.max(np.abs(got - fn(xs))))
    return y0, dslope, max_err, max_err <= err_budget


def pwl_eval(func_name, xs, y0, dslope):
    """Evaluate the PWL table (mirror the RTL: y0 + (dslope*dx)>>10,
    Q1.15 arithmetic, no 16-bit wrap on the sum)."""
    xs = np.asarray(xs, dtype=np.float64)
    xc = np.clip(xs, PWL_XMIN, PWL_XMAX)
    off = xc - PWL_XMIN  # 0..16
    seg = np.minimum((off / 0.5).astype(int), PWL_SEGMENTS - 1)
    dx = ((off - seg * 0.5) / 0.5 * 1024.0)
    dx = np.minimum(dx, 1024.0)
    y0f = np.array(y0, dtype=np.float64)[seg] / 32768.0
    dsf = np.array(dslope, dtype=np.float64)[seg] / 32768.0
    # RTL: y = y0 + ((dslope * dx) >> 10); dslope is Q1.15, dx is
    # 0..1024, so (dslope*dx)>>10 = dslope * dx/1024 in float.
    return y0f + dsf * dx / 1024.0


def substitute(func_name, sub_name):
    """Return (callable, max_point_err) for a cheap substitute of
    ``func_name``. The substitute is a DIFFERENT function; acceptance
    is decided end-to-end by the functional simulator, not here.
    """
    fn = FUNCS[func_name]
    sub = FUNCS[sub_name]
    xs = np.linspace(-8.0, 8.0, 4096)
    err = float(np.max(np.abs(sub(xs) - fn(xs))))
    return sub, err


# Range clipping: the input range beyond which the function is within
# 1e-3 of its asymptote (saturated). Outside the clipped range the
# output is the saturated value and needs no table accuracy.
SAT_RANGE = {"sigmoid": (-8.0, 8.0, 0.0, 1.0),
             "tanh": (-5.0, 5.0, -1.0, 1.0),
             "gelu": (-8.0, 8.0, None, None),   # no saturation
             "silu": (-8.0, 8.0, None, None)}


def clip_range(func_name, calib_min, calib_max):
    """Clip the calibrated input range to the function's saturation
    range. Returns (xmin, xmax, lo_sat, hi_sat) where lo_sat/hi_sat
    are the saturated outputs outside [xmin, xmax] (None if the
    function does not saturate)."""
    lo, hi, lo_sat, hi_sat = SAT_RANGE[func_name]
    xmin = max(calib_min, lo)
    xmax = min(calib_max, hi)
    return xmin, xmax, lo_sat, hi_sat


def choose_act_impl(func_name, err_budget=0.03):
    """Choose the hardware implementation for one activation.

    Returns a dict: {"impl": "pwl"|"mlut", "table": ..., "max_err": ...,
    "meets_budget": bool, "reason": ...}. PWL is tried first; if the
    32-segment PWL does not meet the budget, the MLUT is selected (its
    table is generated by passes/mlut.py and attached by the caller).
    Note: the PWL output is Q1.15 (range [-1, 1)), so it can only
    represent functions whose range fits in [-1, 1) -- sigmoid and
    tanh. GELU/SiLU reach ~8 and MUST use the MLUT (Q5.11 output).
    """
    fn = FUNCS[func_name]
    xs = np.linspace(-8.0, 8.0, 4096)
    if float(np.max(np.abs(fn(xs)))) >= 1.0:
        return {"impl": "mlut", "y0": None, "dslope": None,
                "max_err": None, "meets_budget": False,
                "reason": f"{func_name} range exceeds Q1.15 [-1,1); "
                          f"PWL cannot represent it, MLUT (Q5.11) required"}
    y0, dslope, max_err, ok = pwl_table(func_name, err_budget)
    if ok:
        return {"impl": "pwl", "y0": y0, "dslope": dslope,
                "max_err": max_err, "meets_budget": True,
                "reason": "32-seg PWL meets the budget"}
    return {"impl": "mlut", "y0": None, "dslope": None,
            "max_err": max_err, "meets_budget": False,
            "reason": f"32-seg PWL max_err={max_err:.4f} exceeds "
                      f"budget {err_budget}"}
