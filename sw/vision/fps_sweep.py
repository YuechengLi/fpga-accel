#!/usr/bin/env python3
"""fps sweep: model throughput vs resolution, 640x480 .. 4K.

Uses the Stage 5 roofline method (sw/common/roofline.py), validated against
RTL at small sizes: MACs/frame from the layer shapes, fps from the
device roofline (1968 DSP x 300 MHz = 590.4 GMAC/s peak, 413.3 actual
at 70% utilization; single conv2d_mc engine 145.2 GMAC/s, 4 engines
fit = 580.8 GMAC/s).

Resolution axis = the frame resolution the model operates at:
- SRCNN (x3 SR): OUTPUT (HR) resolution; the LR input is 1/3 per side.
- DnCNN / RetinexNet: input = output resolution (1:1).

RTL cycle simulation at these sizes is infeasible (Stage 6's 24x24
SRCNN run is already minutes; 640x480 is ~500x more pixels), so fps
here is the analytic roofline projection, not a cycle-sim or board
measurement. The cycle model itself was validated against RTL
(weight DMA clean cycles within +/-7 of the Python model; Stage 6
RTL vs functional simulator exact).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "common"))
from roofline import (analyze_model, srcnn_layers, dncnn_layers,
                      retinexnet_layers, ACTUAL_GMACS)

RESOLUTIONS = [(640, 480), (854, 480), (1280, 720),
               (1920, 1080), (2560, 1440), (3840, 2160)]

MODELS = [
    ("SRCNN", srcnn_layers),
    ("DnCNN", dncnn_layers),
    ("RetinexNet", retinexnet_layers),
]

# 1080p reference values from Stage 5 (sw/README.md); the sweep must
# reproduce these, otherwise the method changed.
REF_1080P = {"SRCNN": 16.5, "DnCNN": 1149.0, "RetinexNet": 430.0}


def max_pixels_for_fps(model_fn, fps_target=30.0):
    """Largest frame (pixels, 16:9) reaching fps_target at actual
    throughput. MACs scale linearly with pixels, so solve directly."""
    macs_per_px = None
    # MACs per output pixel from the 1080p layer shapes.
    layers = model_fn(1080, 1920)
    macs = sum(L["H_out"] * L["W_out"] * L["C_out"] * L["C_in"]
               * L["K"] * L["K"] for L in layers)
    macs_per_px = macs / (1920 * 1080)
    px = ACTUAL_GMACS * 1e9 / fps_target / macs_per_px
    return px


def main():
    print("fps sweep (roofline model, 300 MHz, full device 1968 DSP)")
    print("fps_peak = 590.4 GMAC/s, fps_actual = 413.3 GMAC/s (70%),")
    print("fps_1eng = single conv2d_mc engine, 145.2 GMAC/s")
    print()
    ok = True
    for name, fn in MODELS:
        rows = analyze_model(name, fn, resolutions=RESOLUTIONS)
        ref = None
        for r in rows:
            if (r["W"], r["H"]) == (1920, 1080):
                ref = r
        assert ref is not None
        err = abs(ref["macs_G"] - REF_1080P[name]) / REF_1080P[name]
        status = "OK" if err < 0.01 else "MISMATCH"
        if err >= 0.01:
            ok = False
        print(f"== {name}: 1080p check {ref['macs_G']:.1f} GMACs "
              f"(ref {REF_1080P[name]:.1f}) [{status}]")
        print(f"{'res':>12} {'GMACs':>8} {'fps_peak':>9} "
              f"{'fps_actual':>10} {'fps_1eng':>9}")
        for r in rows:
            print(f"{r['W']}x{r['H']:>6} {r['macs_G']:8.1f} "
                  f"{r['fps_peak']:9.2f} {r['fps_actual']:10.2f} "
                  f"{r['fps_1engine']:9.2f}")
        px = max_pixels_for_fps(fn, 30.0)
        # 16:9 dimensions for that pixel count.
        w16 = int((px * 16 / 9) ** 0.5)
        h16 = int(w16 * 9 / 16)
        print(f"  max 16:9 frame at 30 fps (actual): ~{w16}x{h16} "
              f"({px / 1e6:.2f} Mpx)")
        print()
    print("SWEEP CHECK:", "ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
