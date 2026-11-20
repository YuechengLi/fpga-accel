#!/usr/bin/env python3
"""Stage 7b: SRCNN 1080p30 streaming-pipeline cycle model.

Architecture (raster streaming, NO tiles -- revised from the tile
version after the design review: tile buffers for the 64-channel L1
output would cost 256 KB per tile and the conv2d_mc LOAD streams at
1 px/cycle, which would dominate L3; the raster pipeline needs only
line buffers + small gather registers):
  L1 9x9 1->64, P_OUT=16 output units x 81 mults = 1296 DSPs.
      4 cycles/output position (64 outputs / 16 per cycle). The input
      streams at 1 px per 4 cycles (the window advances once per
      position); line buffers 8 rows x 1920.
  L2 1x1 64->32, P_OUT=7 output units x 64 mults = 448 DSPs.
      5 cycles/output position (32 outputs / 7 per cycle; last group
      4 valid). Input is the 64-byte position vector gathered from
      L1's 4-cycle output groups.
  L3 5x5 32->1, tap-parallel: 5 tap-rows x 32 channels = 160 mults.
      5 cycles/output position (25 taps / 5 per cycle). Line buffers
      4 rows x 1920 x 32 ch. Input is the 32-byte position vector
      gathered from L2's 5-cycle output groups.
  Total DSPs = 1296 + 448 + 160 = 1904 (96.7% of the 1968-DSP device).
  (A naive one-engine-per-layer with the existing conv2d_mc C_PAR=4
  is L1-bound at 13.5 fps and is NOT this design. L3 does NOT reuse
  conv2d_mc: its channel-major LOAD streams at 1 px/cycle, which
  would dominate the 5x5 compute; the dedicated tap-parallel engine
  streams position vectors instead. conv2d_mc is left untouched.)

Pipeline: the three engines run concurrently on the raster stream,
decoupled by small gather registers + valid/ready backpressure (the
faster stage stalls when the slower stage's input register is full).
Frame time = max(layer total cycles) + fill. No DRAM traffic in
steady state (all intermediates on-chip in line buffers + gather
registers; weights preloaded once at startup); no backpressure
stalls in steady state beyond the rate matching the RTL sim
measures.

The bicubic upsample (640x360 -> 1920x1080) feeding L1 is NOT in
the RTL pipeline for Stage 7b: it is done by the host (PIL, as in
the Stage 5/6 demos) and costs ~16 MACs/output px (~33M MACs/frame,
0.2% of the 16.45 GMACs). A future RTL bicubic unit would cost
~16 DSPs and does not change the fps conclusion.
"""
import math
import sys

DSP_TOTAL = 1968
FRAME_W, FRAME_H = 1920, 1080   # SRCNN input (upsampled HR) resolution
# Output positions per layer (valid conv): L1/L2 1912x1072, L3 1908x1068.
POS_L1 = (FRAME_W - 8) * (FRAME_H - 8)
POS_L3 = (FRAME_W - 12) * (FRAME_H - 12)

# (name, cycles/position, DSPs, fill cycles)
ENGINES = [
    # L1: input stream 1920x1080 at 1 px/4 cycles = 8,294,400cy;
    # compute 4 cy/pos x 2,049,664 = 8,198,656cy; fill 8 rows at
    # full stream rate = 8*1920cy. Time = max(stream, compute) + fill.
    ("L1 9x9 1->64", 4, 16 * 81, 8 * FRAME_W),
    # L2: 5 cy/pos; fill negligible (starts on first vector).
    ("L2 1x1 64->32", 5, 7 * 64, 0),
    # L3: 5 cy/pos; fill 4 rows at 1 pos/5 cycles = 4*1920*5cy.
    ("L3 5x5 32->1", 5, 5 * 32, 4 * FRAME_W * 5),
]


def main():
    print("Stage 7b cycle model: SRCNN raster streaming pipeline")
    print(f"frame {FRAME_W}x{FRAME_H} in, L1/L2 out "
          f"{FRAME_W - 8}x{FRAME_H - 8} ({POS_L1} pos), "
          f"L3 out {FRAME_W - 12}x{FRAME_H - 12} ({POS_L3} pos)")
    dsp_sum = 0
    totals = {}
    for (name, cy_pos, dsp, fill) in ENGINES:
        pos = POS_L3 if name.startswith("L3") else POS_L1
        if name.startswith("L1"):
            # Input stream (full frame, 1 px/4 cycles) vs compute.
            stream = FRAME_W * FRAME_H * 4
            compute = cy_pos * pos
            total = max(stream, compute) + fill
            print(f"  {name}: P_OUT=16 DSPs={dsp} stream={stream}cy "
                  f"compute={compute}cy fill={fill}cy total={total}cy")
        else:
            total = cy_pos * pos + fill
            print(f"  {name}: DSPs={dsp} {cy_pos}cy/pos x {pos} pos "
                  f"+ fill {fill}cy = {total}cy")
        totals[name] = total
        dsp_sum += dsp
    print(f"  total DSPs = {dsp_sum} / {DSP_TOTAL} "
          f"({100.0 * dsp_sum / DSP_TOTAL:.1f}%)")
    slowest_name = max(totals, key=totals.get)
    frame_cycles = totals[slowest_name]
    print(f"  slowest stage: {slowest_name} {frame_cycles}cy")
    ok = True
    for fclk in (300.0, 360.0, 400.0):
        fps = fclk * 1e6 / frame_cycles
        tag = "MEETS 30fps" if fps >= 30.0 else "BELOW 30fps"
        if fclk == 400.0 and fps < 30.0:
            ok = False
        print(f"  @ {fclk:.0f} MHz: {fps:.1f} fps [{tag}]")
    min_clk = frame_cycles * 30.0 / 1e6
    print(f"  minimum clock for 30 fps: {min_clk:.0f} MHz")
    print("MODEL CHECK:", "ALL OK" if ok else "FAILED (400MHz < 30fps)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
