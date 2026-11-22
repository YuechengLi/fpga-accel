#!/usr/bin/env python3
"""Stage 7b (revised): generic PE-array cycle model, device-driven.

The PE size is DERIVED from the requirements, not assumed (256
mults/PE was only an assumed number). Derivation:

  Required useful throughput: SRCNN 1080p = 16.4534 GMACs/frame
  (L1 10.6255 + L2 4.1977 + L3 1.6302). At 30 fps that is
  493.6 GMAC/s effective, i.e. useful MACs/cycle of:
      300 MHz: 1645 MACs/cycle (83.6% of the 1968-DSP device)
      360 MHz: 1371 MACs/cycle (69.7%)
      400 MHz: 1234 MACs/cycle (62.7%)
  At a realistic 96% array utilization the DSPs needed are
  1714 @300 MHz (87% of device), 1428 @360, 1285 @400.

  PE size vs count: total multipliers = N_PE x mults-per-PE must
  cover the requirement with margin for tile-quantization loss (a
  PE's multipliers are wasted when the tile they map to is smaller
  than the PE). The tile shape must divide each layer's dimensions:
  L1 reduction 5184 (1x64x81), L2 reduction 2048 (64x32x1), L3
  reduction 800 (32x1x25); M = output positions x output channels.
  A PE that is too large wastes multipliers on small tiles (a 256-mult
  PE with TN=16 wastes half its lanes on a C_out=8 layer); a PE that
  is too small costs more control per MAC (more PEs to dispatch, more
  buffer slices). The scheduler picks TK as the largest divisor of
  the K-dim that fits the operand register files (A = TM x TK <= 512,
  B = TK x TN <= 512), so there is no reduction tail for any SRCNN
  layer. The dispatch unit is the (M-tile, N-chunk) task -- the K-loop
  runs inside the PE -- so the issue rate stays well under 1/cycle
  for every size (dispatching every fine tile-step would exceed
  1/cycle for the small PEs; see the dispatch rows below).

Devices come from sw/common/devices.py (XCZU19EG-2, Lattice LFE5U-85F, and
XCVU37P-2FSVH2892E on the VCU128). The ZU19EG is evaluated at
64/128/256 mults per PE; the Lattice part at 640x480 and 320x240
(1080p is NOT its target). The VCU128 uses the SAME 15x128 PE
config as the ZU19EG (decode is bandwidth-bound: HBM at 460 GB/s
delivers ~1,533 weight bytes/cycle at 300 MHz, so ~1,500 MACs/cycle
saturate it and 1,920 MACs already exceed that -- the PE count is
NOT scaled up for the VU37P); its SRCNN fps at 1080p therefore
equals the ZU19EG's, at 21.3% of the VU37P's 9,024 DSPs.

Usage: python3 stage7b_model_pe.py [--device NAME]
Exit code 0 iff the ZU19EG chosen configuration AND the VCU128
1080p configuration both predict >= 30 fps at 300 MHz.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "common"))
from devices import DEVICES

# SRCNN layer shapes: (name, K, C_in, C_out, mode).
#   mode outer: one k/cycle, tile-step = TK cycles.
#   mode dot:   TN taps/cycle, tile-step = TK / TN cycles. Dot
#     requires TN=8 in the RTL (pe_mac.sv 8-tap tree); TN<8 falls
#     back to outer (see model() and sw/common/sched_pe.py).
LAYERS = [
    ("L1 9x9 1->64", 9, 1, 64, "outer"),
    ("L2 1x1 64->32", 1, 64, 32, "outer"),
    ("L3 5x5 32->1", 5, 32, 1, "dot"),
]
A_FILE_MAX = 512
B_FILE_MAX = 512
SWITCH_CYCLES = 150      # per frame tile (3 layer switches x 50)
BUBBLE = 0.01            # dispatch bubbles
GMACS_PER_FRAME_1080P = 16.4534
HBM_GBS = 460.0          # VCU128 HBM2 (sw/common/devices.py vu37p, UG949)


def decode_roofline(mem_gbs, bytes_per_token):
    """Memory-bound decode roofline: tok/s = bandwidth / bytes/token.

    Decode reads each weight once per token and does ~1 MAC per
    weight byte, so the memory bandwidth is the ceiling regardless
    of PE count. 7B INT8: 460 GB/s / 7 GB = 65.7 tok/s. 13B INT4:
    6.5 GB -> 70.8 tok/s (fits one board); 13B INT8: 13 GB ->
    35.4 tok/s (needs 2 boards, 16 GB HBM).
    """
    return mem_gbs * 1e9 / bytes_per_token


def hbm_bytes_per_cycle(mem_gbs, fclk_mhz):
    """Weight bytes/cycle the memory delivers at a clock: HBM at
    460 GB/s is ~1,533 B/cycle at 300 MHz."""
    return mem_gbs * 1e9 / (fclk_mhz * 1e6)


def required_throughput():
    """Print the derivation: useful MACs/cycle per clock."""
    print("Required useful throughput: SRCNN 1080p30")
    print(f"  {GMACS_PER_FRAME_1080P} GMACs/frame x 30 fps = "
          f"{GMACS_PER_FRAME_1080P * 30:.1f} GMAC/s effective")
    for fclk in (300.0, 360.0, 400.0):
        macs = GMACS_PER_FRAME_1080P * 30e9 / (fclk * 1e6)
        mfu = 100.0 * macs / 1968.0
        dsp96 = macs / 0.96
        print(f"  @ {fclk:.0f} MHz: {macs:.0f} useful MACs/cycle "
              f"({mfu:.1f}% device MFU); DSPs needed at 96% util: "
              f"{dsp96:.0f} ({100.0 * dsp96 / 1968:.0f}% of device)")


def choose_tk(kdim, tn):
    """Largest divisor of kdim that fits the operand register files
    (A = 16 x TK <= 512, B = TK x TN <= 512). Returns (TK, exact)."""
    cap = min(32, A_FILE_MAX // 16, B_FILE_MAX // tn)
    for tk in range(cap, 0, -1):
        if kdim % tk == 0:
            return tk, True
    return 1, False


def layer_cycles(name, K, cin, cout, mode, fw, fh, n_pe, tm, tn):
    """Cycles for one layer (ideal, 100% util) + dispatch counts."""
    kdim = cin * K * K
    if name.startswith("L1") or name.startswith("L2"):
        hout, wout = fh - 8, fw - 8
    else:
        hout, wout = fh - 12, fw - 12
    # M-tiles: 4x4 output blocks (TM=16 positions).
    assert hout % 4 == 0 and wout % 4 == 0, f"{name}: M tail"
    m_tiles = (hout // 4) * (wout // 4)
    n_chunks = (cout + tn - 1) // tn if mode == "outer" else 1
    n_tail = cout % tn if mode == "outer" else 0
    tk, exact = choose_tk(kdim, tn)
    k_chunks = (kdim + tk - 1) // tk
    cyc_step = tk if mode == "outer" else tk // tn
    total = m_tiles * n_chunks * k_chunks * cyc_step
    per_pe = total / n_pe
    # Dispatch: one (M-tile, N-chunk) task per issue (K-loop in PE).
    issues = m_tiles * n_chunks
    # Fine tile-step dispatch, for comparison (would be the issue
    # count if every tile-step were dispatched separately).
    fine_issues = m_tiles * n_chunks * k_chunks
    return {"m_tiles": m_tiles, "n_chunks": n_chunks, "n_tail": n_tail,
            "tk": tk, "tk_exact": exact, "k_chunks": k_chunks,
            "cyc_step": cyc_step, "total": total, "per_pe": per_pe,
            "issues": issues, "fine_issues": fine_issues,
            "macs": (hout * wout * cout) * kdim}


def model(dev, fw, fh, pe_mults=None, n_pe=None):
    """Full-frame model for one device + resolution (+ PE size
    override). Returns a dict with per-layer rows and the frame."""
    n_pe = n_pe if n_pe is not None else dev["n_pe"]
    pe_mults = pe_mults if pe_mults is not None else dev["pe_mults"]
    tm, tn = dev["tm"], pe_mults // dev["tm"]
    assert tm * tn == pe_mults, "TM x TN must equal PE multipliers"
    dsps = n_pe * pe_mults
    per_layer = []
    ideal = 0.0
    issues = 0
    fine_issues = 0
    for (name, K, cin, cout, mode) in LAYERS:
        # Dot mode requires TN=8 in the RTL (pe_mac.sv 8-tap tree);
        # with TN<8 the scheduler forces outer mode (sw/common/sched_pe.py),
        # so the model must use the same mode or it overstates fps.
        # Found 2026-11-21: ecp5u85 (TN=4) modeled L3 as dot before
        # this guard; zu19eg (TN=8) is unaffected.
        if mode == "dot" and tn < 8:
            mode = "outer"
        r = layer_cycles(name, K, cin, cout, mode, fw, fh, n_pe, tm, tn)
        r["name"] = name
        per_layer.append(r)
        ideal += r["per_pe"]
        issues += r["issues"]
        fine_issues += r["fine_issues"]
    # Frame tiles: the scheduler searches the candidate sizes and
    # picks the one minimizing (ideal x edge + tiles x switch).
    real = (fw - 8) * (fh - 8)
    best = None
    for ft in dev["frame_tiles"]:
        tiles = math.ceil((fw - 8) / ft) * math.ceil((fh - 8) / ft)
        edge = (tiles * ft * ft) / real
        frame_cy = (ideal * edge + tiles * SWITCH_CYCLES) * (1.0 + BUBBLE)
        if best is None or frame_cy < best["frame_cy"]:
            best = {"frame_tile": ft, "tiles": tiles, "edge": edge,
                    "frame_cy": frame_cy}
    macs = sum(r["macs"] for r in per_layer)
    return {"n_pe": n_pe, "pe_mults": pe_mults, "tm": tm, "tn": tn,
            "dsps": dsps, "per_layer": per_layer, "ideal": ideal,
            "issues": issues, "fine_issues": fine_issues,
            "frame_tile": best["frame_tile"], "tiles": best["tiles"],
            "edge": best["edge"], "frame_cy": best["frame_cy"],
            "macs": macs, "fw": fw, "fh": fh}


def report(dev_name, dev, fw, fh, pe_mults=None, n_pe=None, clocks=None):
    """Print one device/resolution (and PE-size) model result."""
    m = model(dev, fw, fh, pe_mults, n_pe)
    clocks = clocks if clocks is not None else dev["clocks_mhz"]
    tag = (f"{dev_name} {fw}x{fh} "
           f"({m['n_pe']} PEs x {m['pe_mults']} = {m['dsps']} DSPs, "
           f"{100.0 * m['dsps'] / dev['dsp']:.1f}% of {dev['dsp']})")
    print(f"\n{tag}")
    for r in m["per_layer"]:
        ms = r["per_pe"] / (dev["clock_mhz"] * 1e6) * 1000.0
        print(f"  {r['name']}: M-tiles={r['m_tiles']} "
              f"N-chunks={r['n_chunks']} TK={r['tk']}"
              f"{'' if r['tk_exact'] else ' (TAIL)'} "
              f"K-chunks={r['k_chunks']} {r['cyc_step']}cy/step -> "
              f"{r['per_pe']:,.0f}cy ({ms:.2f} ms)")
    iss_rate = m["issues"] / m["ideal"]
    fine_rate = m["fine_issues"] / m["ideal"]
    print(f"  frame tile {m['frame_tile']} ({m['tiles']} tiles, "
          f"edge x{m['edge']:.3f}); ideal {m['ideal']:,.0f}cy, "
          f"frame {m['frame_cy']:,.0f}cy")
    print(f"  dispatch: {m['issues']:,} (M-tile,N-chunk) tasks "
          f"= {iss_rate:.2f}/cycle (fine tile-step dispatch would be "
          f"{fine_rate:.2f}/cycle)")
    out = {}
    for fclk in clocks:
        fps = fclk * 1e6 / m["frame_cy"]
        mfu = (m["macs"] * fps) / (m["dsps"] * fclk * 1e6)
        mark = "MEETS 30fps" if fps >= 30.0 else ""
        print(f"  @ {fclk:.0f} MHz: {fps:.1f} fps {mark}"
              f"(MFU {100.0 * mfu:.1f}%)")
        out[fclk] = fps
    return m, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="device key from sw/common/devices.py (default: all)")
    args = ap.parse_args()
    required_throughput()
    # --- PE-size evaluation on the ZU19EG at 1080p ---
    zu = DEVICES["zu19eg"]
    print("\nPE-size evaluation (XCZU19EG-2, 1080p, 300 MHz baseline):")
    print("  Tile-quantization loss on SRCNN is 0% for all three sizes:")
    print("  N divides exactly (64/32 by TN 4/8/16), TK divides every")
    print("  K-dim (27|81, 32|64, 32|800), M tiles 4x4 exactly.")
    print("  Robustness to small-N layers (not SRCNN, sensitivity):")
    for tn, lbl in ((4, "64-mult (TN=4)"), (8, "128-mult (TN=8)"),
                    (16, "256-mult (TN=16)")):
        w8 = 0.0 if 8 % tn == 0 else 100.0 * (1 - 8 / tn)
        w4 = 0.0 if 4 % tn == 0 else 100.0 * (1 - 4 / tn)
        print(f"    {lbl}: C_out=8 wastes {w8:.0f}%, C_out=4 wastes {w4:.0f}%")
    size_results = {}
    for pe_mults, n_pe in ((64, 30), (128, 15), (256, 7)):
        m, out = report("zu19eg", zu, 1920, 1080,
                        pe_mults=pe_mults, n_pe=n_pe,
                        clocks=(300.0, 360.0, 400.0))
        size_results[pe_mults] = out[300.0]
    print("\nCHOICE: 128 mults/PE (15 PEs, 1920 DSPs) -- the only size "
          f"meeting 30 fps at 300 MHz with margin ({size_results[128]:.1f}); "
          "64 falls to "
          f"{size_results[64]:.1f} fps because TN=4 forces L3 to outer "
          "mode (32cy/step vs 4cy/step in dot), and 256 meets 30 only "
          f"barely ({size_results[256]:.1f} fps) while wasting half its "
          "lanes on C_out=8. 128 also halves the control vs 64 (15 PEs "
          "vs 30 buffer slices / scoreboard entries).")
    # --- Per-device targets ---
    ok = size_results[128] >= 30.0
    vu_fps = None
    for name in DEVICES:
        if args.device and name != args.device:
            continue
        dev = DEVICES[name]
        for (fw, fh) in dev["targets"]:
            m, out = report(name, dev, fw, fh)
            print(f"  verification: {dev['verified']} "
                  f"({'RTL golden + sim' if dev['verified'] == 'rtl' else 'cycle model + synth_ecp5 fit check' if dev['verified'] == 'model+synth' else 'cycle model + scheduler tests'})")
            if name == "vu37p" and (fw, fh) == (1920, 1080):
                vu_fps = out[300.0]
    # --- VCU128 decode roofline (bandwidth-bound, not compute-bound) ---
    vu = DEVICES["vu37p"]
    bpc = hbm_bytes_per_cycle(vu["mem_gbs"], 300.0)
    print("\nVCU128 decode roofline (HBM-bound, not compute-bound):")
    print(f"  HBM {vu['mem_gbs']:.0f} GB/s @ 300 MHz = {bpc:.0f} weight "
          f"bytes/cycle; one MAC consumes one weight byte/cycle, so "
          f"~{bpc:.0f} MACs/cycle saturate HBM")
    print(f"  the 15x128 config has {vu['n_pe'] * vu['pe_mults']} "
          f"MACs/cycle -- already above the saturation point, so the "
          f"PE count is NOT scaled up for the VU37P")
    for label, gbytes, boards in (("7B INT8", 7.0, 1),
                                  ("13B INT4", 6.5, 1),
                                  ("13B INT8", 13.0, 2)):
        print(f"  {label}: "
              f"{decode_roofline(vu['mem_gbs'], gbytes * 1e9):.1f} tok/s "
              f"roofline ({'fits one board' if boards == 1 else 'needs 2 boards, 16 GB HBM'})")
    if vu_fps is not None:
        ok = ok and vu_fps >= 30.0
        print(f"  vu37p 1080p @300 MHz: {vu_fps:.1f} fps "
              f"({'MEETS 30fps' if vu_fps >= 30.0 else 'BELOW 30fps'})")
    print("\nMODEL CHECK:",
          "ALL OK" if ok else "FAILED (300MHz < 30fps)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
