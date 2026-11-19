# Stage 1 — Dense 11x11 convolution + arbitrary padding/stride

SystemVerilog streaming dense convolution (`rtl/conv2d_dense.sv`), PX=1.
Kernel 1/3/5/7/9/11 (121 multipliers/lane), arbitrary per-side padding
(pad_t/pad_b/pad_l/pad_r, each 0..K, zero fill), arbitrary per-axis stride
(stride_x/stride_y, each 1..K, decimate), 32-bit accumulator, shift +
saturation to 8-bit. Config bus, frame-between writes, no re-synthesis.
Reset defaults: ksize=3, shift=8, W=640, H=480, pads 0, stride 1,
3x3 identity (centre tap 256 at coeff index 12).

## Files

| File | What |
|---|---|
| `rtl/conv2d_dense.sv` | DUT + `conv2d_dense_pkg` + BRAM line buffer |
| `tb/golden_dense.py` | Python golden model, vector gen, result check |
| `tb/tb_conv2d_dense.sv` | SystemVerilog testbench (random gaps + backpressure) |
| `tb/run_stage1.sh` | Golden run script (logs to `logs/stage1/`) |
| `sta/critpath_dense.sv` | Per-stage STA modules (mult/add/add_wide/shift/sat) |
| `sta/run_sta_dense.sh` | Per-stage STA run script (logs to `logs/stage1/`) |

## Measured results (freshly run, traceable to `logs/stage1/`)

### Golden (pixel-exact, framing errors 0) — 14/14 PASS

| Case | Config | Output | Result |
|---|---|---|---|
| 0 | 16x16, K=3, shift 0, no pad, stride 1 | 14x14 | PASS (196 px) |
| 1 | 16x16, K=3, shift 3, pad 1, stride 1 | 16x16 | PASS (256 px) |
| 2 | 12x20, K=5, shift 8, pad (0,2,1,3), stride 1 | 12x18 | PASS (216 px) |
| 3 | 16x16, K=11, shift 6, no pad, stride 1 | 6x6 | PASS (36 px) |
| 4 | 16x16, K=1, shift 8, no pad, stride 1 | 16x16 | PASS (256 px) |
| 5 | 16x16, K=3, shift 8, pad 1, stride 2x2 | 8x8 | PASS (64 px) |
| 6 | 16x16, K=3, shift 3, pad 1, stride 2x3 | 8x6 | PASS (48 px) |
| 7 | 16x12, K=7, shift 8, pad (2,1,0,2), stride 3x3 | 4x3 | PASS (12 px) |
| 8 | 15x11, K=5, shift 8, pad 3, stride 1 | 17x13 | PASS (221 px) |
| 9 | 64x48, K=9, shift 6, pad 4, stride 1 | 64x48 | PASS (3072 px) |
| 10 | 32x24, K=11, shift 8, pad (1,3,2,0), stride 2x2 | 12x9 | PASS (108 px) |
| 11 | 20x18, K=3, shift 3, pad 3, stride 1 | 24x22 | PASS (528 px) |
| 12 | 15x13, K=7, shift 8, pad 3, stride 1 | 15x13 | PASS (195 px) |
| 13 | 24x20, K=5, shift 8, pad 2, stride 3x2 | 8x10 | PASS (80 px) |

Coverage: kernels 1/3/5/7/9/11; padding zero / symmetric / asymmetric /
pad > K/2; stride 1 / 2 / 3 / asymmetric; odd width/height; random
in_valid gaps and random out_ready backpressure on every case.
Simulation wall time ~19 s (vvp).

### Per-stage STA (Nangate45, 3.333 ns / 300 MHz, OpenSTA 2.2.0, tns 0)

| Stage | Arrival (ns) | Slack (ns) |
|---|---|---|
| mult (8x16) | 1.351 | +1.948 |
| add (121->61->31, 27-bit) | 1.566 | +1.733 |
| add_wide (8->4->2, 31-bit) | 1.772 | +1.527 |
| shift (32-bit barrel) | 2.093 | +1.199 |
| saturate | 0.329 | +2.959 |

Critical path is the 32-bit barrel shifter (2.093 ns), same as the
reference. Fmax = 1 / (2.093 + 0.041 setup) = **469 MHz**, meeting the
Stage 1 acceptance (Fmax >= 450 MHz, ASIC model).

### Resources (PX=1)

| Resource | Count | Method |
|---|---|---|
| DSP (121 x 8x16 mult, 1 DSP48 each) | 121 | exact (RTL structure) |
| BRAM36 (10 line buffers, ~31 kbit each) | 10 | exact (RTL structure) |
| LUT | ~5,300 | inferred (see note) |
| FF | ~8,750 | counted (RTL structure) |

LUT inference: per-lane datapath (adder tree ~3300 for 121 products +
tap select ~1200 + shift/sat ~230) + fixed control incl. decimation
counters ~600. Full-module Yosys synth was not run (times out on this VM,
as in the reference; the `proc` pass alone ran >5 min without finishing);
LUT is an inferred value, not measured. FF counted from RTL structure
(config 1992 incl. 121x16 coeff regs + datapath pipeline 4220 +
window 968 + alignment 880 + coord delay 260 + control ~430).
On ZU19EG (522K LUT / 1968 DSP / 984 BRAM36): DSP 6%, BRAM 1%.
On Z020 (220 DSP): 121 DSP = 55%, fits.

### Throughput

PX=1 at 300 MHz = 300 Mpx/s input rate; output rate is decimated by
stride (1/(Sx*Sy) of the input rate). 4K60 needs PX=2 (594 MHz pixel
clock); PX=1 covers 1080p60/1080p120/1440p60/4K30 at stride 1.

## Notes

- No row/col-valid masks: padding is physically in the stream (zeros),
  and the window resets to 0, so taps outside the padded image read 0.
- Stride is decimate (compute all, output only at stride positions);
  the next-output counters use "next exceeds" comparisons, so no divider
  is synthesized.
- Pipeline latency 24 cycles (window -> prod -> sum2 -> sum4 -> sum6 ->
  acc -> shift -> saturate/output).
- Stage 1 verifies PX=1. PX is a parameter (window/MAC generalize), but
  the stride-decimation / framing logic is written and verified for PX=1;
  PX=2 with arbitrary padding/stride is left for a later stage.
