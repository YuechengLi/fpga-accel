#!/usr/bin/env python3
"""Stage 5: roofline / partitioning analysis.

Compute roofline (plan §3.1): 1968 DSP x 300 MHz = 590.4 GMAC/s peak;
actual ~70% (utilization + pipeline overhead) = 413 GMAC/s.
DRAM roofline: PL DDR4 actual ~27 GB/s (theoretical 38.4).

A single conv2d_mc engine (C_PAR=4, 484 DSP) peaks at
484 x 300 MHz = 145.2 GMAC/s. The device fits floor(1968/484) = 4
such engines (1936 DSP, 98%) -- 580.8 GMAC/s peak. The roofline is
therefore drawn for the full device (590) and for one engine (145);
a model that needs more than 145 GMAC/s needs more than one engine
(streaming) or more time (iterative).

For each model the analysis computes MACs/frame at 1080p/720p/480p,
the achievable fps at peak / actual / single-engine, and the
arithmetic intensity (MACs per fused DRAM byte) -- the roofline
point. The numbers reproduce plan Table 7 from the layer shapes
(SRCNN 1080p 143 GMACs, DnCNN 1080p 1154 GMACs).
"""
import math

DSP_TOTAL = 1968
F_CLK_MHZ = 300.0
PEAK_GMACS = DSP_TOTAL * F_CLK_MHZ / 1000.0      # 590.4
ACTUAL_GMACS = PEAK_GMACS * 0.70                  # 413.3
ENGINE_DSP = 484                                  # conv2d_mc C_PAR=4
ENGINE_PEAK_GMACS = ENGINE_DSP * F_CLK_MHZ / 1000.0  # 145.2
ENGINES_FIT = DSP_TOTAL // ENGINE_DSP             # 4
PL_DDR4_ACTUAL_GBS = 27.0

# VCU128 (VU37P, docs/vcu128-bringup.md): 9,024 DSP x 300 MHz =
# 2707.2 GMAC/s peak; HBM2 460 GB/s (8 GB, 32x 256-bit AXI3 ports).
# The HBM roofline is the decode ceiling: decode is bandwidth-bound
# (one weight byte per MAC per cycle), so the ridge point is far
# lower than the DDR ridge and most layers are compute-bound on
# this device instead.
VU37P_DSP = 9024
VU37P_PEAK_GMACS = VU37P_DSP * F_CLK_MHZ / 1000.0  # 2707.2
HBM_ACTUAL_GBS = 460.0


def model_macs(layers):
    """Total MACs/frame for a model (list of LayerShape-like dicts
    with H_out, W_out, C_in, C_out, K)."""
    return sum(L["H_out"] * L["W_out"] * L["C_out"] * L["C_in"]
               * L["K"] * L["K"] for L in layers)


def fps_at(macs_per_frame, gmacs):
    """Achievable fps at a given throughput (GMAC/s)."""
    return gmacs * 1e9 / macs_per_frame


def roofline_point(layers, fused_dram_bytes_per_frame,
                   mem_gbs=PL_DDR4_ACTUAL_GBS, peak_gmacs=PEAK_GMACS):
    """Arithmetic intensity (MACs per fused DRAM byte) and the
    compute-vs-memory bound. Returns (intensity, ridge, bound) where
    bound is 'compute' if the ridge point is below the intensity.
    Defaults are the ZU19EG (PL DDR4 27 GB/s, 590.4 GMAC/s peak);
    pass mem_gbs=HBM_ACTUAL_GBS and peak_gmacs=VU37P_PEAK_GMACS for
    the VCU128. Existing callers with no extra arguments are
    unaffected."""
    macs = model_macs(layers)
    intensity = macs / max(1, fused_dram_bytes_per_frame)
    # Ridge point: where compute roof meets DRAM roof (MACs/byte).
    ridge = peak_gmacs * 1e9 / (mem_gbs * 1e9)
    bound = "compute" if intensity >= ridge else "memory"
    return intensity, ridge, bound


def analyze_model(name, layers_fn, resolutions=((1920, 1080),
                                               (1280, 720),
                                               (720, 480))):
    """Analyze one model at several resolutions. ``layers_fn(h, w)``
    returns the layer list (with H_out/W_out/C_out/C_in/K) for an
    input frame h*w. Returns a list of per-resolution dicts."""
    rows = []
    for (w, h) in resolutions:
        layers = layers_fn(h, w)
        macs = model_macs(layers)
        rows.append({
            "model": name, "W": w, "H": h,
            "macs_G": macs / 1e9,
            "fps_peak": fps_at(macs, PEAK_GMACS),
            "fps_actual": fps_at(macs, ACTUAL_GMACS),
            "fps_1engine": fps_at(macs, ENGINE_PEAK_GMACS),
            "fps_4engine": fps_at(macs, ENGINE_PEAK_GMACS * ENGINES_FIT),
        })
    return rows


def srcnn_layers(h, w):
    """SRCNN 9-1-5 at input h*w (valid conv, no padding)."""
    return [
        {"K": 9, "C_in": 1, "C_out": 64,
         "H_out": h - 8, "W_out": w - 8},
        {"K": 1, "C_in": 64, "C_out": 32,
         "H_out": h - 8, "W_out": w - 8},
        {"K": 5, "C_in": 32, "C_out": 1,
         "H_out": h - 12, "W_out": w - 12},
    ]


def dncnn_layers(h, w):
    """DnCNN 17x 3x3, 64ch, pad 1 (same resolution throughout)."""
    layers = [{"K": 3, "C_in": 1, "C_out": 64,
               "H_out": h, "W_out": w}]
    layers += [{"K": 3, "C_in": 64, "C_out": 64,
                "H_out": h, "W_out": w} for _ in range(15)]
    layers.append({"K": 3, "C_in": 64, "C_out": 1,
                   "H_out": h, "W_out": w})
    return layers


def retinexnet_layers(h, w):
    """RetinexNet Decom-Net: 9x9 4->64, 5x 3x3 64->64, 3x3 64->4
    (replicate pad, same resolution)."""
    layers = [{"K": 9, "C_in": 4, "C_out": 64,
               "H_out": h, "W_out": w}]
    layers += [{"K": 3, "C_in": 64, "C_out": 64,
                "H_out": h, "W_out": w} for _ in range(5)]
    layers.append({"K": 3, "C_in": 64, "C_out": 4,
                   "H_out": h, "W_out": w})
    return layers


def partition_advice(name, layers, h, w, fps_target=30.0):
    """Streaming vs iterative vs hybrid (plan §3.6) for one model at
    one resolution and a target fps. Returns a dict."""
    macs = model_macs(layers)
    need_gmacs = macs * fps_target / 1e9
    engines_needed = math.ceil(need_gmacs / ENGINE_PEAK_GMACS)
    if len(layers) <= 4 and engines_needed <= ENGINES_FIT:
        mode = "streaming (one engine per layer, layers pipelined)"
    elif engines_needed <= 1:
        mode = "iterative (one engine, time-multiplexed)"
    else:
        mode = ("hybrid (group layers into fused groups, stream "
                "within a group, iterate between groups)")
    return {"model": name, "H": h, "W": w, "fps_target": fps_target,
            "need_GMACs": need_gmacs, "engines_needed": engines_needed,
            "mode": mode}
