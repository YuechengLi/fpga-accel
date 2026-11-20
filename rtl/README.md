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

---

# Stage 2 — Depthwise (DW) convolution: 17x17 square + 31 strips + dilated

SystemVerilog streaming DW convolution (`rtl/conv2d_dw.sv`), PX=1,
single channel. Modes (config 0x0A): DW-SQUARE (stored kernel
1/3/5/7/9/11/13/15/17), DW-STRIP-H (31x1, length 1/3/5/7/9/11/13/17/21/31),
DW-STRIP-V (1x31, same lengths), DW-DILATED (stored kernel 3/5/7/9,
dilation d = 1..4, effective span (K-1)*d+1 <= 33). DENSE is NOT
re-implemented here; Stage 1's `conv2d_dense.sv` provides it, bit-identical
and untouched. Fully parallel unified MAC: 289 multipliers (17x17 grid);
strips use the linear slot index so a length-31 strip gets all 31 taps.
32 line buffers (BRAM) + a 33x33 register window; the window's bottom-right
corner is the current stream pixel and each mode's footprint is anchored
bottom-right. Same streaming interface, config-bus style, shift+saturate,
32-bit accumulator, and stall behavior as Stage 1. Padding is physically
in the stream (as in Stage 1), no row/col-valid masks. Stride is decimate.
Config bus uses a 9-bit address (289 coefficients do not fit in 8 bits):
control 0x00-0x0B (mode at 0x0A, dilation at 0x0B), coeff at 0x10+i,
i = 0..288 (square: 17x17 grid; strips: 1-D; dilated: 9x9 grid).
Reset defaults: mode=0 (DW-SQUARE), ksize=3, dil=1, shift=8, W=640,
H=480, pads 0, stride 1, 3x3 identity (centre tap 256 at coeff index 18).

## Files

| File | What |
|---|---|
| `rtl/conv2d_dw.sv` | DUT + BRAM line buffer (33x33 window, 289-mult MAC) |
| `tb/golden_dw.py` | Python golden model, vector gen, result check |
| `tb/tb_conv2d_dw.sv` | SystemVerilog testbench (random gaps + backpressure) |
| `tb/run_stage2.sh` | Golden run script (logs to `logs/stage2/`) |
| `sta/critpath_dw.sv` | Per-stage STA modules (mult/add_dw/add_wide/shift/sat) |
| `sta/run_sta_dw.sh` | Per-stage STA run script (logs to `logs/stage2/`) |

## Measured results (freshly run, traceable to `logs/stage2/`)

### Golden (pixel-exact, framing errors 0) — 23/23 PASS

| Case | Config | Output | Result |
|---|---|---|---|
| 0 | DW-SQUARE 17x17, shift 6, no pad, stride 1 | 8x8 | PASS (64 px) |
| 1 | DW-SQUARE 11x11, shift 8, pad 5, stride 1 | 20x20 | PASS (400 px) |
| 2 | DW-SQUARE 7x7, shift 8, pad 3, stride 1, odd W/H | 15x13 | PASS (195 px) |
| 3 | DW-SQUARE 3x3, shift 0, pad 1, stride 1 | 16x16 | PASS (256 px) |
| 4 | DW-SQUARE 5x5, shift 8, pad 2, stride 2x2 | 8x8 | PASS (64 px) |
| 5 | DW-SQUARE 13x13, shift 8, pad 6, stride 1 | 20x16 | PASS (320 px) |
| 6 | DW-SQUARE 15x15, shift 8, pad 7, stride 1 | 16x16 | PASS (256 px) |
| 7 | DW-STRIP-H 31x1, shift 6, pad (0,0,15,15), stride 1 | 16x16 | PASS (256 px) |
| 8 | DW-STRIP-H 21x1, shift 8, pad 10, stride 1 | 20x16 | PASS (320 px) |
| 9 | DW-STRIP-H 11x1, shift 8, pad 5, stride 2x2 | 8x8 | PASS (64 px) |
| 10 | DW-STRIP-H 7x1, shift 3, pad 3, stride 1, odd W/H | 15x13 | PASS (195 px) |
| 11 | DW-STRIP-V 1x31, shift 6, pad (15,15,0,0), stride 1 | 16x16 | PASS (256 px) |
| 12 | DW-STRIP-V 1x21, shift 8, pad 10, stride 1 | 16x20 | PASS (320 px) |
| 13 | DW-STRIP-V 1x11, shift 8, pad 5, stride 2x2 | 8x8 | PASS (64 px) |
| 14 | DW-STRIP-V 1x7, shift 3, pad 3, stride 1, odd W/H | 13x15 | PASS (195 px) |
| 15 | DW-DILATED 3x3 d=1, shift 0, pad 1, stride 1 | 16x16 | PASS (256 px) |
| 16 | DW-DILATED 3x3 d=2, shift 3, pad 2, stride 1 | 16x16 | PASS (256 px) |
| 17 | DW-DILATED 3x3 d=4, shift 3, pad 4, stride 1 | 16x16 | PASS (256 px) |
| 18 | DW-DILATED 5x5 d=2, shift 8, pad 4, stride 1 | 20x20 | PASS (400 px) |
| 19 | DW-DILATED 9x9 d=4, shift 6, pad 16, stride 1 (span 33) | 16x16 | PASS (256 px) |
| 20 | DW-DILATED 5x5 d=4, shift 8, pad 8, stride 2x2 (span 17) | 10x10 | PASS (100 px) |
| 21 | DW-DILATED 7x7 d=2, shift 8, pad 6, stride 1, odd W/H (span 13) | 15x13 | PASS (195 px) |
| 22 | DW-DILATED 9x9 d=2, shift 6, pad 8, stride 1 (span 17) | 16x16 | PASS (256 px) |

Coverage: square 3/5/7/11/13/15/17; strips 7/11/21/31 (H and V);
dilated stored 3/5/7/9 with d = 1/2/4 (effective span 3/5/9/13/17/33);
padding zero / symmetric / asymmetric / pad up to 16; stride 1 / 2 /
asymmetric; odd width/height; random in_valid gaps and random out_ready
backpressure on every case; all 23 cases run back-to-back (no reset
between cases). Simulation wall time ~234 s (vvp).

### Per-stage STA (Nangate45, 3.333 ns / 300 MHz, OpenSTA 2.2.0, tns 0)

| Stage | Arrival (ns) | Slack (ns) |
|---|---|---|
| mult_dw (8x16) | 1.351 | +1.948 |
| add_dw (289->145->73, 27-bit) | 1.566 | +1.733 |
| add_wide_dw (8->4->2, 31-bit) | 1.826 | +1.474 |
| shift_only_dw (32-bit barrel) | 2.093 | +1.199 |
| sat_only_dw | 0.328 | +2.960 |

Critical path is the 32-bit barrel shifter (2.093 ns), same as Stage 1.
Fmax = 1 / (2.093 + 0.041 setup) = **469 MHz**, meeting the Stage 2
acceptance (golden all pass; DW 17x17 DSP 289xPX). The 289-product adder
tree measures the same 1.566 ns as Stage 1's 121-product tree: a
level-pair's delay depends on operand width and on being two adds in
series, not on how many parallel adders there are.

### Resources (PX=1)

| Resource | Count | Method |
|---|---|---|
| DSP (289 x 8x16 mult, 1 DSP48 each) | 289 | exact (RTL structure) |
| BRAM36 (32 line buffers, ~31 kbit each) | 32 | exact (RTL structure) |
| LUT | ~12,500 | inferred (see note) |
| FF | ~33,200 | counted (RTL structure) |

LUT inference: adder tree ~7,900 (Stage 1's ~3,300 for 121 products,
scaled by 289/121) + tap select ~3,500 (mode-dependent window mux, 289
slots, 33x33 window) + shift/sat ~230 + fixed control incl. decimation
counters and span/mode logic ~800. Full-module Yosys synth was not run
(times out on this VM, as in Stage 1); LUT is an inferred value, not
measured. FF counted from RTL structure (config 4,698 incl. 289x16 coeff
regs + window 8,712 + alignment 8,448 + coordinate delay 832 + datapath
pipeline 10,032 + pipeline control ~300 + stream control ~150 + output
12). On ZU19EG (522K LUT / 1968 DSP / 984 BRAM36): DSP 15%, BRAM 3%,
LUT ~2%, FF ~3%. On Z020 (220 DSP): 289 DSP does NOT fit; the DW module
targets the ZU19EG. Strips and dilated modes use a subset of the 289
multipliers (the rest compute 0), so DSP is 289 in every mode.

### Throughput

PX=1 at 300 MHz = 300 Mpx/s input rate; output rate is decimated by
stride (1/(Sx*Sy) of the input rate). Single channel at this stage;
multi-channel DW (Stage 3) reuses this datapath per channel tile.

## Notes

- Two bugs found and fixed during verification (both in the RTL, not in
  the testbench):
  1. Strip tap coverage: the first MAC used a 17x17 slot grid indexed by
     (i,j), so a length-31 strip only got 17 taps (one grid row/column).
     Fixed by using the linear slot index m (0..288) for strips, so a
     length-31 strip gets all 31 taps.
  2. Line-buffer pointer drift across frames: the pointer wraps on
     `ptr == wrap-1`; when `pw` shrinks across frames (and stalls shift
     the pointer), `ptr` can end up >= wrap and then counts up to 2^12
     before wrapping, so the line buffer outputs X for a whole frame
     (found as case 11, strip-V 31, all-X only when run after earlier
     cases; it passed in isolation). Fixed by wrapping on
     `ptr >= wrap-1` (recovers in 1 cycle; identical in normal
     operation). The testbench runs all 23 cases back-to-back with no
     reset between cases, so this is verified, not masked.
- Accumulator is 32-bit (same as Stage 1). Worst case |acc| for 17x17 DW
  is 289*255*32768 ~ 2.41e9 > 2^31, so a fully-saturated 17x17 kernel at
  full-scale input can wrap the accumulator; the golden model wraps
  identically (2's complement), so golden and RTL still agree. Real
  kernels are normalised and do not hit this.
- Pipeline: window -> prod -> sA -> sB -> sC -> sD -> acc -> shift ->
  saturate/output (9 register stages from window load; one more than
  Stage 1, because the 289-product adder tree needs 5 register stages
  instead of 4). Enable delay to window load is 32 cycles (33-row
  window) vs 10 in Stage 1.
- Config bus uses a 9-bit address (289 coefficients do not fit in an
  8-bit address space starting at 0x10); control registers keep the
  Stage 1 layout (0x00-0x09) plus mode (0x0A) and dilation (0x0B).

---

# Stage 3 — Multi-channel tiling + 1x1 + residual + pooling + activation

SystemVerilog streaming multi-channel convolution (`rtl/conv2d_mc.sv`),
plus pooling (`rtl/pool.sv`), elementwise residual (`rtl/residual.sv`),
and activations (`rtl/act.sv`). Same AXI4-Stream-style pixel interface,
config bus, stall behavior, and framing conventions as Stages 1-2.

## conv2d_mc — multi-channel tiling (C_PAR=4)

C_in is tiled by C_PAR=4 parallel dense lanes and accumulated across
tiles (one output = sum over all input channels); C_out is serial via
host kernel-swap (input tile buffered on-chip and reused across all
output channels, so the input is streamed once, not C_out times). The
datapath is NOT sized for 256-512 parallel channels: total mults per
output pixel = C_in x K^2, processed C_PAR at a time.

- Tile size 32x32 (one tile per invocation; larger images are tiled by
  the host in Stage 5). Input tile buffer: C_PAR BRAM banks, bank =
  channel % C_PAR, bank depth 16384 = 16 tiles x 1024 px.
- Dense K=1/3/5/7/9/11 (121 mults/lane), arbitrary per-side padding
  0..K, arbitrary per-axis stride 1..K (decimate), 32-bit accumulator.
  K=1 is the 1x1 pointwise convolution; K=1 with W=H=1 is the FC path
  (input vector as a 1x1 feature map).
- C_IN_MAX=64, C_OUT_MAX=32 (runtime-configurable via config bus;
  hardware sized for the max, config selects actual).
- Per-tile 32-bit accumulator is a register array (`acc_mem`, 1024x32,
  combinational read); first tile overwrites, later tiles add. Stage 5
  will move it to BRAM with pipelined read-modify-write for large tiles.
- Output is emitted during the LAST tile's compute (final sum captured
  with its output position, emitted shift+saturate one cycle later);
  no separate EMIT state. Flush 24 cycles after the last fetch
  (fetch->window 11 + MAC 6 + cross-lane 1 = 18, plus margin).
- Config: 0x00 ksize, 0x01 shift, 0x02/0x03 tile W/H, 0x04-0x07 pads,
  0x08/0x09 stride, 0x0A C_in, 0x0B C_out, 0x0C GO, 0x0D out_w, 0x0E
  out_h, 0x0F sign mode, 0x100+l*128+k coeffs (stride 128, so
  lane/index decode is bit-slicing, not a divider).
- `busy` is high only during LOAD and COMPUTE (low in IDLE and
  WAIT_GO), so the testbench can poll it to know a compute pass ended.

### Signed intermediate activations (config 0x0F)

Bit0 = signed input, bit1 = signed output, reset default 0 (unsigned,
preserves existing behavior). When sign_in is set, the pixel is
sign-extended (`$signed(win)`) instead of zero-extended; when sign_out
is set, the output saturates to [-128, 127] (int8 two's complement in
the same 8-bit port) instead of [0, 255]. Zero padding stays correct
(a padded 0 is 0.0 in signed too; the window mask forces out-of-bounds
products to 0 either way).

This is required by the Stage 6 weights prep: RetinexNet Decom-Net
conv0 has no activation and its output is ~50% negative (range about
[-0.33, +0.31]). A ReLU after conv0 destroys the signal, and uint8+offset
encoding is incompatible with zero padding (zero-padded pixels carry no
offset while the bias correction assumes all pixels do; verified on
RetinexNet: conv1 max error 215, end-to-end R PSNR 14.65 dB). The
correct scheme is signed int8 for signed intermediates, which works
under zero padding. The Stage 2 DW module (`conv2d_dw.sv`) is
unsigned-only and would need the same signed treatment if used on
signed feature maps (Stage 4 follow-up).

### Files

| File | What |
|---|---|
| `rtl/conv2d_mc.sv` | Multi-channel tiling datapath + tile buffer + BRAM line buffers |
| `rtl/pool.sv` | Max/avg 2x2 pooling + global average pooling (GAP) |
| `rtl/residual.sv` | Two-stream elementwise add/sub/bypass with saturation |
| `rtl/act.sv` | ReLU, Leaky/PReLU, PWL sigmoid/tanh/GELU/SiLU |
| `tb/golden_mc.py` | Python golden model, vector gen, result check (12 cases) |
| `tb/tb_conv2d_mc.sv` | SystemVerilog testbench (random gaps + backpressure) |
| `tb/golden_pool.py` / `tb_pool.sv` | Pool golden + testbench (8 cases) |
| `tb/golden_residual.py` / `tb_residual.sv` | Residual golden + testbench (7 cases) |
| `tb/golden_act.py` / `tb_act.sv` | Activation golden + testbench (7 cases) |
| `tb/run_stage3.sh` | Golden run script (logs to `logs/stage3/`), incl. Stage 1+2 regressions |
| `sta/critpath_mc.sv` | Per-stage STA modules (mc_accum / pool_max / pool_gap / act_leaky / act_pwl) |
| `sta/run_sta_mc.sh` | Per-stage STA run script (logs to `logs/stage3/`) |

## Measured results (freshly run, traceable to `logs/stage3/`)

### Golden MC (pixel-exact, framing errors 0) — 12/12 PASS

| Case | Config | Output | Result |
|---|---|---|---|
| 0 | K=3 16x16 C_in=3 C_out=2 shift 8, no pad, stride 1 | 14x14 | PASS (392 px) |
| 1 | K=3 16x16 C_in=8 C_out=2 shift 8, pad 1, stride 1 | 16x16 | PASS (512 px) |
| 2 | K=3 16x16 C_in=16 C_out=8 shift 8, pad 1, stride 1 | 16x16 | PASS (2048 px) |
| 3 | K=3 16x16 C_in=64 C_out=2 shift 8, no pad, stride 1 | 14x14 | PASS (392 px) |
| 4 | K=5 16x16 C_in=8 C_out=4 shift 8, pad 2, stride 1 | 16x16 | PASS (1024 px) |
| 5 | K=1 16x16 C_in=16 C_out=16 shift 8, no pad | 16x16 | PASS (4096 px) |
| 6 | K=1 16x16 C_in=64 C_out=32 shift 8, no pad | 16x16 | PASS (8192 px) |
| 7 | K=1 1x1 C_in=8 C_out=4 shift 8 (FC) | 1x1 | PASS (4 px) |
| 8 | K=3 15x13 C_in=3 C_out=2 shift 8, pad 1, stride 2x2 | 8x7 | PASS (112 px) |
| 9 | K=3 16x16 C_in=8 C_out=4 shift 0, no pad | 14x14 | PASS (784 px) |
| 10 | K=3 16x16 C_in=4 C_out=2 shift 8, pad 1, signed in+out | 16x16 | PASS (512 px) |
| 11 | K=3 16x16 C_in=8 C_out=2 shift 8, pad 1, signed in, unsigned out | 16x16 | PASS (512 px) |

Total 18,580 output pixels, all pixel-exact. Coverage: C_in 3/8/16/64
(tiled, C_PAR=4), C_out 2/8/16/32, K=1/3/5, 1x1 pointwise C_in=64
C_out=32, FC (W=H=1), stride 2x2, shift 0, signed intermediates
(cases 10-11: ~48% negative input pixels; case 10 has negative
outputs, so the signed saturation path is exercised). Random in_valid
gaps and random out_ready backpressure on every case. Simulation wall
time ~15 min (vvp; case 6, C_in=64 C_out=32, dominates).

### Golden pool (pixel-exact, framing errors 0) — 8/8 PASS

| Case | Config | Output | Result |
|---|---|---|---|
| 0 | max 2x2, 16x16 | 8x8 | PASS (64 px) |
| 1 | max 2x2, 15x13 (odd) | 8x7 | PASS (56 px) |
| 2 | max 2x2, 1x1 | 1x1 | PASS (1 px) |
| 3 | avg 2x2, 16x16 | 8x8 | PASS (64 px) |
| 4 | avg 2x2, 15x13 (odd) | 8x7 | PASS (56 px) |
| 5 | GAP, 16x16 | 1x1 | PASS (1 px) |
| 6 | GAP, 15x13 (odd) | 1x1 | PASS (1 px) |
| 7 | GAP, 1x1 | 1x1 | PASS (1 px) |

Max/avg 2x2 zero-pads to even W/H on the fly; AVG = sum>>2; GAP sums
real pixels only (>> gap_shift config, single output).

### Golden residual (pixel-exact, framing errors 0) — 7/7 PASS

| Case | Config | Output | Result |
|---|---|---|---|
| 0 | add, 16x16 | 16x16 | PASS (256 px) |
| 1 | add, 15x13 (odd) | 15x13 | PASS (195 px) |
| 2 | add, 16x16 (saturation/overflow) | 16x16 | PASS (256 px) |
| 3 | sub, 16x16 | 16x16 | PASS (256 px) |
| 4 | sub, 15x13 (odd) | 15x13 | PASS (195 px) |
| 5 | bypass, 16x16 | 16x16 | PASS (256 px) |
| 6 | add, 1x1 | 1x1 | PASS (1 px) |

Two-stream elementwise add/sub/bypass with saturation; FIFO depth 4
per stream aligns independent random gaps.

### Golden act (pixel-exact, framing errors 0) — 7/7 PASS

| Case | Config | Result |
|---|---|---|
| 0 | ReLU (incl. negative saturation) | PASS (12 values) |
| 1 | Leaky ReLU | PASS (12 values) |
| 2 | Leaky ReLU | PASS (8 values) |
| 3 | PWL sigmoid (32 segments over [-8,8]) | PASS (87 values) |
| 4 | PWL tanh | PASS (87 values) |
| 5 | PWL GELU | PASS (87 values) |
| 6 | PWL SiLU | PASS (87 values) |

ReLU and Leaky/PReLU are fully verified. PWL sigmoid/tanh/GELU/SiLU
are a single configurable module with a coefficient table loaded via
config bus (32 segments over [-8,8], width 0.5; y0 + dslope per
segment, Q1.15); all four functions verified pixel-exact on 87 sample
points each.

### Regressions (run by `tb/run_stage3.sh`)

Stage 1: 14/14 PASS. Stage 2: 23/23 PASS. No regression.

### Per-stage STA (Nangate45, 3.333 ns / 300 MHz, OpenSTA 2.2.0, tns 0)

| Stage | Arrival (ns) | Slack (ns) |
|---|---|---|
| mc_accum (4x32-bit cross-lane sum + acc add) | 1.797 | +1.495 |
| pool_max (2x2 max compare tree) | 0.787 | +2.506 |
| pool_gap (18-bit accumulate add) | 0.658 | +2.634 |
| act_leaky (32x16-bit slope multiply) | 2.248 | +1.046 |
| act_pwl (16x11-bit interpolation + 32-bit add) | 1.656 | +1.634 |

All stages MET at 300 MHz. The new bottleneck is the Leaky ReLU
32x16-bit slope multiply (2.248 ns): Fmax = 1 / (2.248 + 0.041 setup)
= **437 MHz**. The conv datapath itself is still limited by the
Stage 1 barrel shifter at 469 MHz, so the Stage 3 system Fmax (same
clock domain) is ~437 MHz, well above the 300 MHz operating point.
(The shift-only and saturate-only stages are identical to Stage 1's
and are not re-measured here.)

### Resources

| Module | DSP | BRAM36 | LUT | FF | Method |
|---|---|---|---|---|---|
| conv2d_mc (C_PAR=4, K<=11) | 484 | 36 | ~19,000 | ~65,800 | DSP exact; BRAM measured; LUT inferred; FF counted |
| pool | 0 | ~1 | ~500 | ~500 | inferred |
| residual | 0 | 0 | ~300 | ~300 | inferred |
| act | 2 | 0 | ~800 | ~400 | inferred |

conv2d_mc detail: DSP = C_PAR x 121 = 484 (each an 8/9-bit x 16-bit
multiply, 1 DSP48 each), exact from RTL structure. BRAM36 = 36,
measured: tile buffer 4 banks x 16384x8 = 4 x 4 RAMB36 = 16
(Yosys `synth_xilinx` on a standalone 16384x8 bank: 4 RAMB36E1);
line buffers 40 x 64x8 = 40 x 1 RAMB18E1 = 20 BRAM36 (Yosys
`synth_xilinx` on the standalone line-buffer module: 1 RAMB18E1).
LUT inferred: per-lane adder tree 4x~3,300 (Stage 1's 121-product
tree) + tap select 4x~1,200 + shift/sat ~230 + cross-lane sum ~200 +
control ~600. FF counted from RTL structure, dominated by the
accumulator register array (1024x32 = 32,768) and coefficient regs
(4x121x16 = 7,744); the rest is the datapath pipeline (~24,000) and
control (~1,300). Full-module Yosys synth was not run (times out on
this VM, as in Stages 1-2); LUT is an inferred value, not measured.
On ZU19EG (522K LUT / 1968 DSP / 984 BRAM36): conv2d_mc DSP 25%,
BRAM 4%, LUT ~4%, FF ~6%. On Z020 (220 DSP): 484 DSP does NOT fit;
conv2d_mc targets the ZU19EG (the C_PAR=4 / K=11 max configuration).
A K=3-only build would need 4x9 = 36 DSPs.

## Notes

- Bug found and fixed during verification (in `rtl/conv2d_mc.sv`,
  the tile-buffer BRAM read): the read `tb_pix <= tb_mem[tb_raddr]`
  was NOT gated by `stall`, while its valid flag `pos_real_d1` was.
  During an output stall, `tb_raddr` is frozen, but the ungated read
  still advances `tb_pix` to the next pixel (the address was already
  presented), so `lane_pix` delivered the NEXT pixel and the current
  one was lost -- a one-pixel shift in the data relative to the
  coordinates. It reproduced with backpressure only (68/4096
  mismatches on the case-5 isolation, 16/512 on a 1-tile K=1 case),
  not with input gaps alone (0/4096) and not without backpressure
  (0/4096). The symptom was intermittent: a few pixels per frame got
  the next pixel's value (e.g. ch1 (1,11) got (1,12)'s acc, -8 instead
  of 487). Fixed by gating the read with `!stall`, like every other
  pipeline register. After the fix the full suite passes (MC 12/12,
  pool 8/8, residual 7/7, act 7/7, Stage 1 14/14, Stage 2 23/23).
  Every other state element in the module was already stall-gated;
  `tb_pix` was the only ungated one.
- Signed-mode rounding: the RTL uses round-half-up shift (add
  2^(shift-1), then arithmetic shift), matching Stages 1-2. The
  Stage 6 golden (`models/common.py`) uses floor shift (no rounding);
  the <=1 LSB difference is expected and must be noted in the Stage 6
  comparison, not "fixed" by changing the RTL.
- The DW module (`conv2d_dw.sv`, Stage 2) is unsigned-only. If it is
  used on signed feature maps, it needs the same signed treatment as
  conv2d_mc (Stage 4 follow-up).
- Stage 4/5 design inputs (recorded for the next stages, not
  implemented here): (1) In time-multiplexed mode the weight registers
  need ping-pong (double) buffering -- while the current tile computes,
  the next tile's weights are loaded into the other buffer, so the
  switch is immediate and the load time is hidden; cost is 2x weight
  storage. Coefficients currently load via the config bus (AXI-Lite),
  which is too slow for frequent time-multiplexed switching; weights
  should stream from DRAM via AXI-MM into the ping-pong buffer
  (Stage 5). Pre-compilation should search the optimal
  time-multiplexed tile granularity per layer (cost model: compute
  time, load time, overlap) under DSP/BRAM/DRAM-bandwidth constraints,
  so load time <= compute time (fully hidden) or the two are balanced;
  the optimum varies per layer shape (K/C_in/C_out/H/W), not one
  global value. (2) A separate compiler module is needed, based on
  the PyTorch compile graph (torch.compile / TorchDynamo FX graph;
  torch.export for AOT): capture graph -> fusion (conv+relu etc.,
  Stage 5 layer fusion) -> tiling (the per-layer granularity search
  above) -> scheduling (tile order, ping-pong weight buffer
  management, DRAM) -> output a program (layer descriptors) for the
  accelerator.

## Stage 4: remaining operators + unified engine + layer scheduler

### New operators

**LayerNorm (`rtl/layernorm.sv`).** Normalizes one frame (W x H pixels =
one "layer" of activations): mean = sum(x)/N, var = sum(x^2)/N - mean^2,
y = (x - mean) * inv_std * gamma + beta, inv_std = 1/sqrt(var + eps).
W and H must be powers of two so N is a power of two and the divisions
are arithmetic right shifts by log2(N) (no divider synthesized). Two
passes: pass 1 streams the frame in, buffers it on-chip (up to 4096
entries, 1 BRAM36), and accumulates sum (signed 32-bit) and sumsq
(signed 32-bit); pass 2 streams it out, one pixel per cycle,
normalizing each. The reciprocal square root comes from a 256-entry
LUT (Q1.15) loaded via the config bus, indexed by (var + eps) >>
isqrt_shift, clamped to [0,255]; the compiler owns the LUT contents
and the shift (it picks the shift so the index spans the expected var
range). Approximation: inv_std is quantized to Q1.15 and to the LUT bin
width (2^isqrt_shift in var); the golden reports the error vs the true
float LayerNorm (max |fixed - float| <= 1.36 LSBs across the 6 cases,
dominated by the LUT quantization, not by the fixed-point arithmetic).
The normalize path is PIPELINED into two registered stages (Stage A:
diff x inv_std >> 7 -> norm_q; Stage B: norm_q x gamma >> 23 + beta),
because the unpipelined two-multiply chain violates 3.333 ns (see STA
below). Throughput stays 1 pixel/cycle; latency is +1 cycle. Config
(16-bit address): 0x01 W, 0x02 H, 0x03 sign, 0x04 isqrt_shift, 0x05 eps,
0x06 gamma_q (Q1.15), 0x07 beta, 0x100+i inv_std LUT entry i (0..255).
Golden 6/6 PASS, framing 0.

**Upsample (`rtl/upsample.sv`).** Scale factor 2, three modes, one
shared structure: load an input row into buffer(s), emit two output
rows; input paused (in_ready low) while emitting. Line buffers are
register arrays sized by MAX_W (default 64); the scheduler tiles the
frame so the tile width fits (a BRAM line buffer is the production
choice for large W). Mode 0 nearest: each pixel replicated 2x2. Mode 1
bilinear: out[2r,2c]=in[r,c], out[2r,2c+1]=(in[r,c]+in[r,c+1])/2,
out[2r+1,2c]=(in[r,c]+in[r+1,c])/2,
out[2r+1,2c+1]=(in[r,c]+in[r,c+1]+in[r+1,c]+in[r+1,c+1])/4, edges
replicate, divisions round-half-up ((a+b+1)>>1, (a+b+c+d+2)>>2); two
input rows are ping-pong buffered (a `cur` bit selects which holds the
current row, no row copy). Mode 2 pixel-shuffle (depth-to-space, r=2):
input is W x H positions x 4 channels in (H,W,C) order; output 2W x
2H, 1 channel (out[2r,2c]=ch0, out[2r,2c+1]=ch1, out[2r+1,2c]=ch2,
out[2r+1,2c+1]=ch3). Config: 0x00 mode, 0x01 W, 0x02 H. Golden 10/10
PASS, framing 0.

**Concat (`rtl/concat.sv`).** Two input streams A and B (same W/H), one
output. Mode 0 sequential: out = A frame followed by B frame, channel
order (C,H,W). Mode 1 interleave: out = A and B interleaved per pixel
(H,W,C order). Small FIFOs (depth 4 each) align A/B (independent
random gaps); a single stall freezes output; each input's in_ready is
high while its FIFO has space. If B arrives early it waits in its FIFO
until A is fully output (sequential) or its pair is ready (interleave);
backpressure on B is correct, not a deadlock. Config: 0x00 mode, 0x01
W, 0x02 H. Golden 6/6 PASS, framing 0.

**SimpleGate (`rtl/simplegate.sv`).** Elementwise multiply (NAFNet).
Input is a single stream of 2*W*H pixels: first W*H = half A, next
W*H = half B. Output W*H pixels: out[i] = (A[i] * B[i]) >> shift,
saturated. First half buffered on-chip (up to 4096 entries; the
scheduler tiles the frame so the tile fits); when the second half
arrives, each B pixel is multiplied by buffered A at the same index
and emitted. One pixel/cycle in steady state (second half streams
through); first half is load-only; 1-cycle latency from a B pixel to
its output. Config: 0x00 shift (0..8), 0x01 W, 0x02 H, 0x03 sign.
Golden 6/6 PASS, framing 0.

### Signed DW

`rtl/conv2d_dw.sv` was unsigned-only (Stage 2). Stage 4 adds the same
signed intermediate support Stage 3 added to conv2d_mc: config 0x0F
(bit0 signed in, bit1 signed out), sign-extend MAC, saturate to
[-128,127] when signed out (emitted as a two's-complement byte).
Zero padding stays correct (padded 0 is 0.0 in signed too). Golden
26/26 PASS (23 original unsigned + 3 signed: K=3 Sobel, K=5 Gaussian,
L=7 strip-H, all signed in/out, ~50% negative bytes), framing 0.
Note: the RTL uses round-half-up shift; the Stage 6 golden uses floor
shift -- the <=1 LSB difference is expected and must be noted in the
Stage 6 comparison, not "fixed" in the RTL.

### Multi-level LUT activation (act.sv mode 6) -- supersedes direct LUT

Stage 3 act.sv uses a PWL unit (16-32 segment table + 1 interpolation
multiply). A flat 256-entry direct LUT was considered and then
corrected: a flat table cannot guarantee precision for steep functions
without becoming huge (a 12-bit input would need 4096 entries for full
precision). The activation LUT is therefore MULTI-LEVEL, not flat.
Input is split into bit-fields: Level 1 (coarse, high bits) gives a
base value and/or selects a fine table; Level 2 (fine, low bits) gives
the delta; output = base + delta (an addition, NOT a multiply). Three
levels are possible if needed. A 12-bit input needs 64+64 entries, not
4096; the lookup is table reads + one add, no multiply, so it stays
fast. Hardware is fully configurable: the number of levels, the
bit-split (how many bits per level), and all table contents are loaded
via the config bus. The COMPILER decides the levels/splits/lookup: the
number of levels, the bit-splits (uniform or non-uniform -- steep
regions of the function get finer resolution, flat regions coarser),
and the table contents, driven by the function's curvature and the
accuracy budget; the compiler verifies accuracy (software functional
simulator + fixed-point golden, per the Stage 5 compiler verification
flow). Implemented as mode 6 in act.sv: input is a 12-bit offset-binary
value in data_in[11:0], Q4.8 ([-8, 8)); config 0x02 MLUT control (H1
coarse bits 1..8, H2 fine bits 1..6, H1+H2 <= 12); coarse base table
0x200+idx, coarse select table 0x300+idx (fine table id 0..7), fine
tables 0x400+t*64+idx. Chosen golden split: H1=8, H2=4, 8 bands,
mean-based fine tables (least-squares fit); max error vs float:
sigmoid 0.0027, tanh 0.0186, GELU 0.0269, SiLU 0.0234 (comparable to
PWL's ~0.024). Total table 256 coarse + 8x16 fine = 384 entries, ~10x
smaller than a flat 4096 table. Golden: 4 MLUT cases (sigmoid/tanh/
GELU/SiLU, 4096 values each), all PASS. The direct-LUT vs PWL
trade-off is superseded by this multi-level correction; the PWL mode
remains for functions where the compiler prefers interpolation.

### Bypass mode + fused ReLU/Leaky (act.sv mode 7, conv output stage)

act.sv stays a separate module and has a BYPASS mode (mode 7, config):
output = input unchanged, framing passes through unchanged. Bypass is
used when the activation is fused into the conv module, or when the
layer has no activation at all. Fused ReLU (and fused Leaky ReLU) is
in the conv datapath (conv2d_mc and conv2d_dense and therefore the
unified engine): after accumulator + shift, optionally apply ReLU
(max(0,x)) / Leaky (shift-add) directly in the conv output stage,
controlled by config (0x10 fused act, 0x11 slope). ReLU is trivial
(comparator + mux) and fusing it saves a pipeline stage and a module
traversal; the separate LUT module is reserved for complex activations
(sigmoid/tanh/GELU/SiLU). The compiler (Stage 5) decides per layer:
fuse ReLU/Leaky into conv, or route complex activations through the
separate LUT act module; the act module's bypass covers the fused and
no-activation cases. Golden: act bypass case (12 values, PASS); MC
fused ReLU (case 12) and fused Leaky (case 13) PASS.
**FC/MLP path:** after an MLP (fully-connected) layer there may also
be a ReLU. FC reuses the 1x1 conv datapath (and the unified engine's
1x1/FC mode), so the fused ReLU in the conv output stage already
covers FC/MLP -- confirmed, not assumed. Golden: MC case 14 (FC,
K=1, W=H=1, C_in=8, C_out=4, signed in/out, fused ReLU, act bypassed)
PASS; unified-engine case 4 (FC 1x1, signed, fused ReLU, byte 200 =
-56 clipped to 0) PASS. Only complex activations after FC/MLP (e.g.
GELU in a transformer MLP) go through the separate LUT act module.
conv2d_dense also gained the Stage 4 signed + fused-act additions
(config 0x0F sign, 0x0A fused act, 0x0B slope) so the unified engine's
dense path has them; Stage 1 regression still 14/14 PASS (defaults
off, backward compatible).

### Unified reconfigurable engine (`rtl/unified_engine.sv`)

One engine, five modes, one-click switch via config 0x0C (mode):
0=dense (conv2d_dense, K=1/3/5/7/9/11), 1=DW-square, 2=DW-strip-H,
3=DW-strip-V, 4=DW-dilated (conv2d_dw), 5=FC/1x1 (conv2d_dense, K=1).
It wraps the Stage 1 dense and Stage 2 DW modules (reference
implementations, datapath unmodified) and does NOT duplicate five
datapaths: the dense submodule's one 121-mult MAC serves dense and
FC/1x1, and the DW submodule's one 289-mult MAC serves the three DW
modes via its own mode select. The engine provides the unified
stream/config interface and the mode mux. Its config map IS the
submodules' shared map (0x00 K, 0x01 shift, 0x02 W, 0x03 H, 0x04-0x07
pads, 0x08-0x09 strides, 0x0A fused act [dense] / DW mode [DW],
0x0B act slope [dense] / dilation [DW], 0x0F sign, 0x100+i
coefficients), so forwarding is a pure router (truncate the address
to the active submodule's bus width; coefficients translate
0x100+i -> 0x10+i). The mode write (0x0C) is consumed by the engine
and, for a DW mode, programs the DW submodule's mode register in the
same write.
**Mode-switch latency and state:** the switch takes effect on the
next clock (1 cycle) and MUST be done between frames (no frame may be
in flight); the stream mux is combinational, so there is no added
pipeline latency for the pixel stream. Switching back to a previous
mode does NOT require reloading its coefficients IF the intervening
modes used the other submodule: each submodule keeps its own
coefficient/config registers, so the inactive submodule's state
(coefficients, K, pad, stride, shift, sign, fused act) is preserved
across a switch. Dense and FC share one submodule/MAC, and the three
DW modes share the other, so coefficients are NOT preserved across
modes that share a submodule (each layer loads its own weights -- the
expected layer-switch behaviour). Verified by the golden: cases 2 and
3 re-run dense / DW-square with NO coefficient reload and PASS.
Golden 7/7 PASS (dense -> DW-sq -> dense -> DW-sq -> FC+ReLU ->
DW-strip-H -> DW-dilated, no reset between frames), framing 0.

### Layer scheduler (`rtl/layer_scheduler.sv`)

Steps through a layer configuration sequence: for each layer it
programs the layer's config (mode, K, pad, stride, shift, sign,
fused act), triggers execution, and waits for completion. Weight
double-buffering (ping-pong): two weight register banks (A and B,
289 coefficients each, 16-bit). Layer L's weights live in bank
(L % 2): while bank A drives layer 0, bank B is loaded with layer 1's
weights, so the load time is hidden; while bank B drives layer 1,
bank A is reloaded with layer 2's weights, and so on. The scheduler
never touches a bank while it is driving the engine. Weights stream
in through the scheduler's own config bus in this stage (0x0200+i
bank A, 0x0400+i bank B); the DRAM/AXI-MM streaming that fills the
banks is Stage 5 -- the boundary is explicit (the scheduler only
knows "write coefficient i of bank X", not where the coefficients
come from). A write to a bank that is currently driving the engine is
ignored and `wfull` pulses (backpressure on the weight stream).
**Ping-pong timing:** when the next layer's weights are fully loaded
before the current layer finishes, the switch is immediate (no stall
-- the next layer starts on the next cycle after the current one
completes). When they arrive late, the scheduler waits in ST_WAIT_W
(it does NOT start the next layer with a half-loaded bank, and it
does not corrupt the active bank). The TB verifies both: run 0
(on-time, layer 2's weights loaded into bank A while layer 1 runs
via fork/join) and run 1 (late, layer 2's weights loaded only after
layer 1 finishes, so the scheduler waits in ST_WAIT_W). Both runs
produce correct outputs. A bank is "ready" for layer L when its
loaded count >= prog_ncoeff[L]; the count resets when the scheduler
switches away from that bank. The scheduler drives the engine's
config bus as a master (14 control writes per layer, then streams the
active bank's coefficients to 0x100+i, one per cycle) and watches the
engine's `busy` (with a seen-busy flag, so it does not falsely
complete before the frame starts streaming). Program: up to 4 layer
descriptors (fields written via config, one write per field,
addressed by (layer, field)). Golden: 2 runs x 3 layers (dense 3x3
Sobel, DW 5x5 Gaussian, dense 3x3 Sobel). **Verified: both runs
768 outputs, all 6 layers PASS, framing 0.** Ping-pong timing
(measured): run 0 (on-time) layer-2 eng_start at cycle 3626 (start
928); run 1 (late) layer-2 eng_start at cycle 8705 (start 5612) --
the scheduler waited ~395 cycles in ST_WAIT_W for the late weights
(121 coefficients x 2 cycles each, plus entry/exit overhead) and
did not start layer 2 with a half-loaded bank; when the weights
arrived on time (run 0) there was no stall. **Bugs found and fixed:**
(1) ST_WAIT_W checked `next_ready` (readiness of layer_idx+1),
but layer_idx has already advanced to the layer being waited for,
so it checked the layer AFTER that -- whose prog_ncoeff is 0 when
unprogrammed, hence trivially true. The scheduler left ST_WAIT_W
immediately, streamed a half-loaded bank, and pulsed eng_start
while the TB was still loading weights (the TB then deadlocked
waiting for a 7th pulse). Fixed with a separate `cur_ready`
(readiness of the current layer_idx), used in ST_WAIT_W; `next_ready`
is still used at the ST_RUN transition, where layer_idx is still
the finishing layer and next_layer is correct. (2) Bank address
overlap: bank A (base 0x0200, 289 entries) spans 0x0200..0x0320,
so a bank B base of 0x0300 overlapped it in 0x0300..0x0320; the
else-if checked bank A first, so bank-B writes with index < 33
were misrouted into bank_a[256+i] (bankA_cnt reached 289 instead
of 121) and bank_b[0..32] stayed X -- the DW layer then output X
for all 256 pixels in both runs. Fixed by moving bank B's base to
0x0400 (bank A 0x0200..0x0320, bank B 0x0400..0x0520, no overlap).
(3) TB: a plain `wait (eng_start === 1'b1)` can miss the 1-cycle
pulse if it fires while the TB is between waits; the TB now counts
eng_start pulses and waits for the count to increase (`wait_start`
task), which cannot miss a pulse.

### STA (Nangate45 @ 3.333 ns, OpenSTA 2.2.0, logs/stage4/)

| stage | arrival (ns) | slack (ns) | tns | Fmax (MHz) |
|---|---|---|---|---|
| ln_norm_a (diff x inv_std >> 7) | 2.137 | +1.162 | 0 | ~468 |
| ln_norm_b (norm_q x gamma >> 23 + beta) | 2.451 | +0.841 | 0 | ~408 |
| ln_sumsq (px*px + sumsq) | 2.006 | +1.293 | 0 | ~498 |
| act_mlut (base + delta, 8:1 mux) | 1.273 | +2.017 | 0 | ~786 |
| dense_leaky (32x16, bits [47:16]) | 2.248 | +1.046 | 0 | ~445 |
| sg_mult (8x8 >> shift, saturate) | 1.369 | +1.918 | 0 | ~730 |
| up_bilin ((a+b+c+d+2) >> 2) | 0.687 | +2.604 | 0 | ~1456 |
All MET, tns 0. The NEW bottleneck is ln_norm_b (+0.841), so the
system Fmax is now ~408 MHz (was ~437 MHz with act_leaky as the
bottleneck in Stage 3). Still well above the 300 MHz target. The
unpipelined LayerNorm normalize (two 32x16 multiplies in series in
one cycle) violated at 4.301 ns arrival, slack -1.009, tns -14.250 --
it would have capped the system Fmax at ~232 MHz; pipelining it into
two stages (ln_norm_a / ln_norm_b) fixed the violation at the cost of
+1 cycle of LayerNorm latency (throughput unchanged at 1 pixel/cycle).

### Resources (inferred/estimated; full-module Yosys synth times out
on this VM, as in Stages 1-3; DSP/BRAM exact where counted from RTL
structure, LUT/FF inferred)

- layernorm: BRAM36 1 (frame buffer 4096x8), DSP ~4 (2 normalize
  32x16 + var 32x32 + sumsq 16x16), LUT ~2,000, FF ~1,000.
- upsample: BRAM 0 (register line buffers 2x64x8), DSP 0, LUT ~500,
  FF ~1,000.
- concat: BRAM 0 (2x depth-4 FIFOs in registers), DSP 0, LUT ~200,
  FF ~100.
- simplegate: BRAM36 1 (buffer 4096x8), DSP 1 (8x8 multiply), LUT
  ~300, FF ~100.
- unified engine: DSP 410 (dense 121 + DW 289), BRAM36 42 (dense 10
  + DW 32), LUT ~18,000 (dense ~5,300 + DW ~12,500 + mux), FF
  ~42,000. ZU19EG: DSP 21%, BRAM 4%.
- layer scheduler: BRAM 0 (weight banks 2x289x16 in registers),
  DSP 0, LUT ~2,000, FF ~10,000 (weight banks dominate: 578x16).
- conv2d_dense (with Stage 4 sign + fused act): DSP 121 (unchanged;
  the fused Leaky adds 1 multiply per lane in the emit path, 1 DSP
  when used), BRAM 10, LUT ~5,300.

### Bugs found and fixed (Stage 4)

- LayerNorm `px_s` race (TB/DUT): a combinational signal computed
  from `pixel_in` via a ternary/`$signed` continuous `assign` (and
  later an `always @*` block) lagged `pixel_in` by one cycle in
  Icarus -- the trace showed px=57 but px_s=0 at the sampling edge,
  corrupting the sum/mean/var (case 0 mean 110 vs 114) and leaking the
  previous frame's last pixel into the next frame (case 3 mean 101 vs
  100). Root cause: the TB drives `pixel_in` in the active region at
  the same clock edge the DUT samples, so the combinational `px_s`
  is stale when sampled. Fixed in the TB, not the RTL: a `#1` settle
  delay after each `@(posedge clk)` before driving `pixel_in` /
  `in_valid` (acceptance is still checked immediately after the edge,
  pre-edge `in_ready`, so the check is not disturbed). After the fix
  px_s matches px at every index and the mean is correct.
- LayerNorm sumsq unsigned context: `sumsq_acc` was `logic [31:0]`
  (unsigned), so `px_s * px_s` was evaluated in an unsigned context
  and a negative pixel (e.g. -108, as unsigned 65428) squared to
  ~4.28e9 instead of 11664; sumsq_new was 3,754,469,735 instead of
  ~829,760, var was ~58.6M instead of 12,709, and the isqrt LUT index
  clamped to 255 (inv_std 256 instead of 291). Fixed by declaring
  sumsq_acc/sumsq_new signed and using `$signed(px_s) * $signed(px_s)`.
- LayerNorm TB send-loop re-accept: the original pattern waited for
  `in_ready` AFTER the accepting edge and held `in_valid` high, so
  the DUT re-accepted the same pixel as the first pixel of the next
  frame. Fixed by deasserting `in_valid` immediately on acceptance.
- Upsample terminal DONE state: the FSM had a terminal `ST_DONE`;
  after case 0 the module stayed there, `in_ready` never went high
  for the next frame, and the TB sender blocked forever (TBs in this
  repo do NOT reset the DUT between cases). Fixed by returning to
  `ST_LOAD` (idle) on frame completion.
- Upsample bilinear last-row stall: after emitting for row 14, row 15
  was already buffered (loaded as "next"), but the FSM went to
  `ST_LOAD` and waited for a nonexistent row 16. Fixed by going
  directly to `ST_EMIT` when advancing to the last row (the odd
  output row replicates the current row).
- Upsample pixel-shuffle index: using `in_col>>1` as the base gave
  2c+1 for ch2/ch3 (should be 2c); `odd_buf[0]` was never written (X
  in output). Fixed with base = `(in_col>>2)<<1`.
- SimpleGate 16-bit product overflow: 255x255 = 65025 > 32767 wraps
  to -511 in 16-bit. Product made 32-bit signed.
- SimpleGate `buf` is a reserved Verilog keyword (gate primitive);
  renamed to `a_buf`. Also, Icarus rejects a memory-array read in a
  continuous `assign` -- the read is in an `always @*` block.
- Concat declaration-before-use: Icarus requires a declaration before
  its use in the file (`cnt_a`/`cnt_b` were used before the FIFO
  declarations); moved the FIFO declarations before the sequencing
  logic.
- LayerNorm normalize STA violation (see STA above): the unpipelined
  two-multiply chain violated 3.333 ns. Fixed by pipelining the
  normalize into two registered stages (ln_norm_a / ln_norm_b),
  +1 cycle latency, throughput unchanged. Re-verified 6/6 PASS.
- Unified engine coefficient preservation (test design, not RTL):
  dense and FC share one submodule/MAC and the three DW modes share
  the other, so coefficients are NOT preserved across modes that
  share a submodule. The golden's preservation cases therefore
  re-run dense / DW-square with no reload only when the intervening
  case used the OTHER submodule (cases 2 and 3); an earlier version
  of the test re-ran dense after an FC case and failed because the FC
  coefficients had (correctly) overwritten the dense ones.
- Scheduler ST_RUN false completion: `ST_RUN` waited for `!eng_busy`,
  but `eng_busy` is low while the TB is still reacting to `eng_start`
  (before the first pixel is accepted), so the scheduler could
  falsely complete a layer before its frame started. Fixed with a
  `seen_busy_r` flag (wait for busy high first, then for busy low).

### Notes

- Round-half-up vs floor shift: the RTL uses round-half-up shift
  (add 2^(shift-1), then arithmetic shift) in conv2d_dense, conv2d_dw,
  conv2d_mc, and layernorm. The Stage 6 golden uses floor shift. The
  <=1 LSB difference is expected and must be noted in the Stage 6
  comparison, not "fixed" in the RTL.
- The direct-LUT vs PWL trade-off (Stage 4 steer 1) is superseded by
  the multi-level LUT correction (steer 2): a flat 256-entry table
  cannot guarantee precision for steep functions. See "Multi-level LUT"
  above.
- `tb/run_stage4.sh` runs all Stage 4 modules + Stage 1 / pool /
  residual regressions. The MC sim (~19 min), DW sim (~8 min) and
  scheduler sim (~10 min) are long; each uses timeout 1800.
