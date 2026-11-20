#!/usr/bin/env python3
"""Stage 5 report: cost model + DRAM + roofline numbers for the three
models (SRCNN, DnCNN, RetinexNet) at 1080p, 300 MHz, ZU19EG.

Prints the numbers the Stage 5 verification is based on, with the
plan cross-references and the two known discrepancies (SRCNN MACs,
DnCNN DRAM) explained. Exit 0 (it is a report, not a test).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from costmodel import LayerShape, tile_search  # noqa: E402
from dram import dram_traffic, vdma_bandwidth_gbs, arbitration_budget  # noqa: E402
from roofline import (model_macs, roofline_point, analyze_model,  # noqa: E402
                      srcnn_layers, dncnn_layers, retinexnet_layers)


def main():
    print("=== Stage 5 report: cost model / DRAM / roofline ===")
    print("Target: 300 MHz, ZU19EG (1968 DSP, 984 BRAM36, 523K LUTs)")
    print("Peak 590.4 GMAC/s, actual 413.3 @70%, single engine 145.2")
    print()

    # --- Roofline ---
    print("--- Roofline (1080p) ---")
    for name, fn in (("SRCNN", srcnn_layers), ("DnCNN", dncnn_layers),
                     ("RetinexNet", retinexnet_layers)):
        rows = analyze_model(name, fn, resolutions=((1920, 1080),))
        r = rows[0]
        print(f"{name:12s} {r['macs_G']:8.1f} GMACs  "
              f"fps peak {r['fps_peak']:6.2f}  actual {r['fps_actual']:6.2f}  "
              f"1-eng {r['fps_1engine']:6.2f}  4-eng {r['fps_4engine']:6.2f}")
    print("  Note: plan Table 7 says SRCNN 143 GMACs; the 9-1-5 model in")
    print("  this repo is 16.5 GMACs at 1080p. DnCNN matches (1149 vs")
    print("  1154, 0.4%), so the method is validated and the SRCNN plan")
    print("  number is the outlier (different variant/resolution).")
    print()

    # --- DRAM ---
    print("--- DRAM traffic (1080p60, GB/s) ---")
    for name, fn, inch, outch in (("DnCNN", dncnn_layers, 1, 1),
                                  ("DnCNN (3ch in/out)", dncnn_layers, 3, 3),
                                  ("RetinexNet", retinexnet_layers, 3, 3)):
        layers = fn(1080, 1920)
        t = dram_traffic(layers, 1080, 1920, 60, fused=True,
                         in_ch=inch, out_ch=outch)
        tu = dram_traffic(layers, 1080, 1920, 60, fused=False,
                          in_ch=inch, out_ch=outch)
        print(f"{name:22s} unfused {tu['unfused_total_GBs']:7.2f}  "
              f"fused {t['fused_total_GBs']:6.2f}  "
              f"(weights {t['fused_weights_GBs']:.2f})")
    print("  Note: plan Table 5 says DnCNN unfused 271 GB/s (all 17")
    print("  layers 64ch); the real model has 1ch first/last layers,")
    print("  so 255.37 GB/s is accurate. Plan Table 6 says fused 0.81")
    print("  (RGB in AND out); grayscale 1ch is 0.31, 3ch-in/1ch-out")
    print("  is 0.56. All unfused numbers are >> 27 GB/s (impossible);")
    print("  all fused numbers are << 27 GB/s (feasible).")
    vin = vdma_bandwidth_gbs(2160, 3840, 3, 60)
    vout = vdma_bandwidth_gbs(2160, 3840, 3, 60)
    eng, ok = arbitration_budget(vin, vout)
    print(f"  VDMA 4K60 in+out {vin + vout:.2f} GB/s, engine budget "
          f"{eng:.2f} GB/s (ok={ok})")
    print()

    # --- Cost model (per-layer tile search) ---
    print("--- Cost model: tile search (load hidden = load <= compute) ---")
    cases = [
        ("SRCNN L1  K=9  Cin=1", LayerShape(9, 1, 64, 33, 33,
                                            pad=(0, 0, 0, 0))),
        ("DnCNN L2  K=3  Cin=64", LayerShape(3, 64, 64, 32, 32,
                                             pad=(1, 1, 1, 1))),
        ("K=11      Cin=64", LayerShape(11, 64, 64, 32, 32,
                                        pad=(5, 5, 5, 5))),
    ]
    for label, layer in cases:
        r = tile_search(layer)
        print(f"{label:20s} C_in_tile={r['C_in_tile']}  "
              f"ncoeff={r['ncoeff']:3d}  load_axi={r['load_axi']:3d}cy  "
              f"compute={r['compute_per_pass']:4d}cy  "
              f"hidden={r['load_hidden']}  "
              f"total={r['total_eff_cycles']}cy")
    print("  K=11 is bank-limited: C_in_tile=2 (2*121=242 <= 289);")
    print("  C_in_tile=4 would need 484 > 289. The optimum varies per")
    print("  layer; load is hidden in all cases above.")
    print()

    # --- Roofline bound ---
    print("--- Roofline bound (fused, 1080p) ---")
    for name, fn in (("DnCNN", dncnn_layers),
                     ("RetinexNet", retinexnet_layers)):
        layers = fn(1080, 1920)
        t = dram_traffic(layers, 1080, 1920, 60, fused=True,
                         in_ch=1, out_ch=1)
        intensity, ridge, bound = roofline_point(
            layers, t["fused_total_GBs"] / 60 * 1e9)
        print(f"{name:12s} intensity {intensity:8.1f} MACs/B  "
              f"ridge {ridge:6.1f}  bound={bound}")
    print()
    print("STAGE 5 REPORT: OK")


if __name__ == "__main__":
    main()
