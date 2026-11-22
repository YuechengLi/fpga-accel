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

---

# Stage 5 — AXI-MM weight-streaming DMA (`rtl/weight_dma.sv`)

Stage 4 loaded weights via the config bus (2 cycles/coeff,
PS-written), which is too slow for frequent time-multiplexed
switching. Stage 5 adds the AXI-MM weight-streaming path: the DMA
streams int16 coefficients from DRAM into the ping-pong weight banks
(scheduler, contiguous addressing) or directly into the engine's
coefficient registers (conv2d_mc, strided addressing), autonomously
(the PS only programs the registers; the C driver in `sw/driver/`
does that).

## Interface

- AXI-MM read master, 64-bit, up to 16-beat INCR bursts, 2-beat
  FIFO (circular buffer) decoupling the read from the writer.
- AXI-Lite slave for programming (6-bit addr, 32-bit data):
  0x00 src (DRAM byte offset), 0x01 count (coefficients),
  0x02 dst (config-bus base), 0x03 taps (K*K per lane; 0 =
  contiguous), 0x04 stride (address stride between lanes),
  0x05 ctrl (bit0 start, self-clearing), 0x06 status (bit0 done,
  bit1 busy).
- Config-bus write port (`cfg_wr_en/addr/data`, `cfg_wr_full`
  backpressure): one coefficient per cycle when the FIFO has data
  and the destination is not full. The DMA stalls (does not advance
  the coefficient pointer) while `cfg_wr_full` is asserted.
- Addressing: coefficient i goes to lane `i/taps`, tap `i%taps`,
  address `dst + lane*stride + tap` (taps=0: lane 0, tap = i,
  contiguous into a scheduler bank). For conv2d_mc the stride is
  128 (lane l at 0x100+l*128+k); for a scheduler bank the stride is
  1.

## Cycle model (must match `sw/dram.py weight_dma_cycles`)

`cycles = ncoeff + n_bursts*4 + 12`, where n_bursts =
ceil(ncoeff/64) (4 coeffs/beat x 16 beats), 4 = AR overhead per
burst, 12 = initial latency. The drain (1 coeff/cycle) is the
bottleneck; the 64-bit AXI bus is faster than the drain. The TB's
AXI slave is ideal (AR ready immediately, first R data 4 cycles
after AR accepted, then back-to-back beats), so the measured count
matches the model.

## Measured results (freshly run, traceable to `logs/stage5/`)

`tb/run_stage5_rtl.sh` runs 4 cases x clean + wfull backpressure =
8 runs, all data correct (every coefficient at the right address
with the right value, golden from `tb/golden_dma.py`).

| Case | ncoeff | dst | taps | stride | Model (cy) | Measured clean (cy) | Measured wfull (cy) |
|---|---|---|---|---|---|---|---|
| A | 81 | 0x0200 (bank A) | 0 (contig) | 1 | 101 | 99 (-2) | 129 |
| B | 36 | 0x0100 (engine) | 9 | 128 | 52 | 47 (-5) | — |
| C | 289 | 0x0100 (engine) | 289 | 128 | 321 | 328 (+7) | 427 |
| D | 200 | 0x0400 (bank B) | 0 (contig) | 1 | 228 | 232 (+4) | 298 |

Clean cycle counts are within +/-7 cycles of the Python model (the
small offset is the AXI-Lite start-write + AR handshake latency,
which the model's 12-cycle budget covers). Under random wfull
backpressure (~25% of cycles) the transfer is slower but the data
is still correct -- the DMA stalls rather than dropping a
coefficient. `STAGE 5 RTL RESULT: ALL PASS`.

## Notes

- The DMA is RTL. Board-level AXI integration (SmartConnect, the
  DRAM controller, the PS driver handshake, VDMA frame streaming)
  is Stages 8-11 (Pittsburgh). The Python model (`sw/dram.py`) and
  the C driver (`sw/driver/`) are the Stage 5 software side.
- Icarus notes: `fork/join_none/join_any` and `disable fork` are
  NOT supported (replaced with a bounded `for` wait loop); adjacent
  string literals are NOT concatenated in `$display` (single-line
  strings only); declarations must precede their use in the file
  (the state/beat signals are declared before the FIFO that uses
  them); an enum ternary needs an explicit if/else (Icarus requires
  a cast for the ternary form).
- A 4-wide bank write would match the AXI bus rate but needs a
  scheduler hardware change (follow-up, recorded in `sw/README.md`).

# Stage 6 — Model numerical verification (SRCNN / DnCNN / RetinexNet)

## Method

Each model is compiled by the Stage 5 compiler (`sw/`), and each conv
layer is driven through `rtl/conv2d_mc.sv` (C_PAR=4, per-output-channel
bias, signed intermediates) with weights loaded via the AXI-MM weight
DMA (`rtl/weight_dma.sv`). Layers run in waves: wave 1 = layers whose
input is known (the test image or a functional-simulator output), wave
2 = layers fed by a wave-1 RTL output, so the chain is real. The DnCNN
denoised step goes through `rtl/residual.sv` mode 3 (sub with B
sign-extended). The RetinexNet sigmoid goes through `rtl/act.sv` PWL
mode 2 with the compiler's 32-segment table. Driver: `sw/stage6.py`
(plus `sw/stage6_float_layers.py` for the per-layer vs-float numbers);
testbench `tb/tb_stage6_mc.sv` (no random in_valid gaps / out_ready
stalls -- backpressure was verified in Stage 3, Stage 6 is numerical
verification, so speed matters; framing is still checked);
orchestrator `tb/run_stage6.sh` (models + Stage 5 acceptance +
Stage 1-4 regressions).

Error budget: RTL vs functional simulator must be EXACT (max|err| = 0)
for every conv layer -- the functional simulator is the reference, so
an RTL-vs-simulator mismatch is a hardware bug (or a testbench-driving
bug), never a simulator bug unless proven with a minimal case. The PWL
sigmoid is allowed max|err| <= 1 LSB (the RTL uses integer >>10
arithmetic, the simulator float -- a rounding-boundary difference).
Per-layer vs float is the quantization error (RTL == simulator exactly,
so simulator-vs-float = RTL-vs-float). RetinexNet is compared against
a ZERO-pad float reference for the quantization check (the RTL uses
zero padding, the model was trained with replicate padding); the
replicate-vs-zero border difference is reported separately, never mixed
into the quantization number.

## Measured results (freshly run, traceable to `logs/stage6/`)

`tb/run_stage6.sh`: **STAGE 6 RESULT: ALL PASS**. Stage 5 acceptance
re-run: ALL PASS. Stage 1-4 RTL regressions re-run: ALL PASS (2439s).

### SRCNN (full chain, 24x24 input, all channels, 649.9s)

| Layer | vs funcsim max\|err\| | vs float max\|err\| | vs float PSNR |
|---|---|---|---|
| L1 conv2d (9x9, 1->64) | 0.00 | 1 | 71.58 dB |
| L2 conv2d_1 (1x1, 64->32) | 0.00 | 2 | 55.48 dB |
| L3 conv2d_2 (5x5, 32->1) | 0.00 | 6 | 43.38 dB |
| End-to-end | -- | 5.53 | 43.51 dB (prep 42.76 dB) |

### DnCNN fragment (16x16 input, 755.3s)

L1 (full 64 out), L2 (8 out), L3 (8 out), L16 (8 out), L17 (1 out),
all vs funcsim max|err| = 0.00. Denoised (residual mode 3) vs funcsim
max|err| = 0.00 (PSNR inf); vs float max|err| = 1.37, PSNR 53.23 dB
(prep 48.23 dB). L4-L15 NOT run in RTL -- same 3x3 64->64 shape as
L2/L3/L16, verified by sampling; output channels are computed serially
and independently (identical datapath, only weights differ), so
sampling verifies the layer shape. Per-layer vs float for all 17
layers (functional simulator): 1-8 LSB, PSNR 49.84-77.49 dB
(worst L15, 8 LSB / 49.84 dB).

### RetinexNet Decom-Net (full chain, 16x16 input, 1029.0s)

conv0 (full 64 out), conv1 (8 out, fed by RTL conv0), conv2-5 (4 out
each), recon (full 4 out), all vs funcsim max|err| = 0.00. Sigmoid
(PWL mode 2, 32 segments, r_in=3 r_out=0) vs funcsim max|err| = 1.00
(budget <= 1). Per-layer vs float (zero-pad): conv0 1 (73.11 dB),
conv1 1 (61.30 dB), conv2 2 (60.51 dB), conv3 2 (58.08 dB), conv4 4
(55.24 dB), conv5 17 (48.02 dB), recon 5 (50.81 dB), sigmoid 8
(45.93 dB). End-to-end vs float (ZERO pad): R PSNR 45.37 dB
(max|err| 8.18), L PSNR 49.99 dB (max|err| 2.57). Vs float (REPLICATE
pad, original model): R 26.72 dB, L 36.71 dB -- the drop is the border
effect, not quantization.

Border analysis (replicate vs zero, funcsim fixed-point): R full
max|diff| = 55, PSNR 26.78 dB; R interior (5px excluded) max|diff| =
3, PSNR 45.91 dB; L full max|diff| = 21, PSNR 36.90 dB; L interior
max|diff| = 1, PSNR 55.91 dB. Float-only padding difference: R
27.12 dB, L 37.47 dB. The border dominates the full-frame number; the
interior is clean. This matches the Stage 6 prep finding (R max|diff|
65.98, PSNR 28.95 dB).

### r_out re-validation (wider/natural data) + accumulator bound

The stored r_out (calibrated on small synthetic test images) is too
small for 4 layers -- activations exceed 2^r_out and clip to 255. This
is a calibration finding, NOT an RTL bug: the RTL correctly implements
whatever r_out it is given. Corrected r_out: DnCNN L3 stored 1 ->
required 2 (max|act| 2.18), DnCNN L7 stored 1 -> required 2
(max|act| 2.28), DnCNN L15 stored 0 -> required 1 (max|act| 1.09),
RetinexNet conv4 stored 0 -> required 1 (max|act| 1.12). SRCNN: no
saturation on wider data. Saturation is mild (max exceeds the range by
< 20% in all cases). The quantize pass should calibrate on
wider/more representative data (follow-up).

Accumulator bound: the 32-bit accumulator stays in range for all
layers in all three models (max 20,688,812, DnCNN L1, well below
2^31). The bound is on the FULL sum after all C_in tiles + bias, not
the per-tile partial, because the RTL's 32-bit acc_mem holds the
running sum and modular addition means intermediate wraps do not
corrupt the final sum as long as the final sum is within [-2^31, 2^31).

## Bugs found and fixed during Stage 6

- **conv2d_mc had no bias register (real gap).** The functional
  simulator always added the per-output-channel bias (acc = ... +
  b_q[o]), and `models/common.py` conv2d_fixed also adds b_q, but
  `rtl/conv2d_mc.sv` had no bias register and `sw/emit.py` emitted no
  bias writes. The bias is NOT negligible: DnCNN L2 b_q range up to
  ~+/-10.6M (shift 15 -> up to ~+/-325 LSBs contribution); SRCNN L3
  b_q = 105236 (shift 13 -> ~12.8 LSBs). Without the fix every layer
  would fail by the bias offset. Fix: config 0x12 (bias lo) / 0x13
  (bias hi), signed 32-bit bias_r, reset 0; acc_wr_val = (cur_t==0) ?
  (tile_partial + bias_r) : (acc_rd + tile_partial) -- bias added once
  per output pixel, on the first C_in tile. `sw/emit.py` `_emit_mc_layer`
  now emits two W instructions per output channel (lo then hi) before
  the first tile's DMA for that channel. Default bias 0, so the Stage 3
  TB (which never writes 0x12/0x13) is unaffected. Verified with a bias
  sanity test (one small MC case, 64 outputs, 0 mismatches) and by the
  full Stage 6 runs.
- **residual.sv treated both streams as UNSIGNED (real gap for
  DnCNN).** Mode 1 sub computed A - B with B unsigned. DnCNN's final
  step is denoised = clip(noisy - residual, 0, 255) where the residual
  (L17 output) is signed (~50% negative). A negative residual byte
  (e.g. -5 = 0xFB) would be treated as 251. Fix: mode 3 added (sub with
  B sign-extended: out = clamp(A - sign_extend(B), 0, 255), via a 10-bit
  signed difference). Modes 0-2 unchanged. `tb/golden_residual.py`
  mode 3 added; residual regression 8/8 PASS (including the new mode-3
  case).
- **DMA/emitter coefficient placement (Stage 5 bug, caught by the
  Stage 6 DMA integration check).** The Stage 5 emitter stored only
  the K^2 contiguous taps in the weight blob and programmed the DMA
  with taps=K^2. But `conv2d_mc`'s (and `conv2d_dense`'s) coefficient
  memory is indexed by the 11x11 grid (k = i*11+j); the MAC reads
  coeff[ln][i*11+j]. So DMA-loaded coefficients landed at the wrong
  addresses for K < 11 (DMA integration check, SRCNN L1 ch5, K=9:
  245/256 outputs wrong, systematically low -- most taps read as 0).
  The Stage 5 standalone DMA test never caught this because it verified
  the DMA against its own documented addressing, not against what the
  MC needs; the Stage 4 scheduler TB already used the 121 grid, so the
  scheduler path was correct. Fix (no RTL change): `sw/emit.py` now
  stores the FULL 121-entry grid per lane in the blob, and the DMA is
  programmed with taps=121 (MC path ncoeff = (t1-t0)*121; scheduler path
  ncoeff=121, taps=121). Blob sizes grew (SRCNN 16064 B -> 518848 B;
  DnCNN ~14.9 MB; RetinexNet 5080064 B) -- loaded once per model run,
  DRAM is plentiful, load still hidden. Cost model updated accordingly
  (ncoeff per pass = C_in_tile x 121; per-pass budget C_PAR x 128 = 512;
  K=11 is NOT bank-limited, C_in_tile=4, 484 <= 512). DMA integration
  check after the fix: PASSED (256 outputs, exact); DMA done in 139
  cycles for 121 coeffs. Scheduler-path check: PASSED (256 outputs,
  exact, bias=0).
- **Driver 32-bit DMA truncation (exposed by the grid fix).** The C
  driver's dma_write interface and the mock stored DMA registers as
  uint16_t, truncating the source byte offset to 16 bits. The old blob
  (16064 B) fit in 64 KB so it was harmless; the new grid blob (SRCNN
  518 KB) exceeds 64 KB. Fix: widened the DMA register interface to
  32-bit (`sw/driver/driver.h`, `driver.c`, `mock_mmio.h`,
  `mock_mmio.c`). The RTL's reg_src is 32-bit, so the hardware was fine.
- **Sigmoid act table is PWL, not MLUT.** The RetinexNet sigmoid's
  compiler table has impl="pwl" (32-segment PWL, Q1.15 y0/dslope), not
  the MLUT (H1/H2/base/sel/fine) that `run_act_sigmoid` first assumed
  (KeyError 'H1'). Fixed by dispatching on the table's `impl` in
  `sw/stage6.py`: PWL uses `act.sv` mode 2 with the 32-segment table
  (input Q5.11, output Q1.15), MLUT uses mode 6. The input scale comes
  from the act op's r_in (RetinexNet recon r_out=3, so the sigmoid
  r_in=3 -- x = a / 31.875, not a / 255).

## Notes

- The dense path (`conv2d_dense` / unified engine) has NO bias
  register (conv2d_dense uses 8-bit config addresses; coeff base is
  0x10 so 0x12/0x13 would collide with coefficients; the engine uses
  0x000C for mode select). No Stage 6 model layer is single-channel, so
  no model layer uses the scheduler path; the scheduler-path check uses
  a real single-channel slice (SRCNN L1 output channel 5) with bias
  forced to 0 in the reference. The missing dense-path bias is a
  documented follow-up, not a Stage 6 blocker. The MC path (where all
  model layers run) has full bias support.
- The RTL uses round-half-up shift; the Stage 6 weights-prep golden
  uses floor shift -- the <=1 LSB difference is expected and is noted
  here, NOT "fixed" in the RTL.
- 3/4-level MLUT tables are NOT emitted for the current act.sv (2
  levels only); a 4-wide bank write would match the AXI bus rate but
  needs a scheduler hardware change (follow-ups, recorded in
  `sw/README.md`).
- Toolchain note (2026-11-20): the VM was replaced and /usr/bin
  iverilog + yosys vanished. Reinstalled from direct Ubuntu-pool .deb
  downloads (iverilog 13.0-2, yosys 0.52-2 + yosys-abc, libjs-sphinxdoc,
  python3-click), never `apt-get update` (hangs). Cached .debs live in
  `~/tools-dl/`. The run scripts already check for iverilog and point
  at `~/tools-dl/iverilog_13.0-2_amd64.deb` when it is missing.

# Stage 7 — Front-end video preprocessing module (`rtl/frontend.sv`)

The whole-model I/O front-end: it sits at the video input (HDMI/SDI IN)
and output, ahead of the unified engine, and does downsampling + small-kernel
(<=5x5) filtering on the pixel stream. Derived from the `conv2d_dense.sv`
structure (line-buffer chain, alignment delay, window, MAC, shift+saturate,
decimation counters, stall freeze) + the frozen reference `conv2d_stream`
(`~/workspace/fpga-2d-conv/`, untouched).

## Design

- 3 parallel lanes, one per input channel, sharing one 25-coefficient bank
  (5x5 grid, row stride 5). Kernel K = 1/3/5. Each lane computes its own
  25-tap dot product; the 3 lanes run concurrently, so one output pixel
  (up to 3 channels) is produced per cycle (PX=1 spatially, as in Stage 1;
  the plan's PX=2 is a later-stage option).
- Line buffers: 4 x 24-bit BRAM (one 24-bit group = 3 channels x 8 bits per
  pixel), 3872 entries each (max padded width 4096 - 224). Depth is set by
  the padded width, not the channel count, so the buffer count is constant
  (4) for 1ch and 3ch.
- Window: 5x5 groups of 24 bits; lane c reads byte c of each group
  (`win[wr][wc][8*c +: 8]`). A 1ch input is packed into byte 0 of the
  group; lanes 1-2 are idle (their products are zero).
- Adder tree per lane: 25 products -> 13 -> 7 (27 bits) -> 4 -> 2 (29 bits)
  -> 1 (32-bit accumulator). Max |acc| ~ 2.09e8 < 2^31 (worst case K=5,
  25 taps x 255 x 32767).
- Downsampling DS = 1/2/4 folds into the decimation counters: effective
  stride = stride x DS (1..20), implemented as "next exceeds" on the
  output coordinates (no divider). out_W = (pw - K)/eff_Sx + 1.
- Pipeline latency 11 cycles by construction (line buffer + align 4,
  window 1, products 1, sum2 1, sum4 1, accumulator 1, shift 1, output 1).
  The first output additionally waits the (K-1)-row line-buffer fill.
- Fused ReLU / Leaky ReLU in the output stage (same as Stage 4:
  config 0x0A mode, 0x0B slope Q0.16), applied after shift+saturate.
  Signed input/output via config 0x0F (same convention as Stage 3/4).
- No bias register (deliberate: preprocessing kernels do not use bias).
- Config map: 0x00 ksize (1/3/5), 0x01 shift (0..15), 0x02 W, 0x03 H,
  0x04-0x07 pads, 0x08/0x09 stride, 0x0A fused act (0 none, 1 ReLU,
  2 Leaky), 0x0B slope, 0x0C downsample (1/2/4), 0x0D channels (1/3),
  0x0F sign, 0x10+i coefficients i=0..24 (5x5 grid, row stride 5).
  Reset defaults: ksize 3, shift 8, 640x480, pads 0, stride 1, DS 1,
  ch 3, sign 0, fused act 0, slope 0.125, 3x3 identity (centre 256).

## Verification (golden, 16 cases, `tb/golden_frontend.py` + `tb/tb_frontend.sv`)

All 16 cases PASS pixel-exact (max|err| = 0), framing errors 0, with
random in_valid gaps + ~80% out_ready backpressure. Cases: 0 identity
1ch K3; 1 blur 3ch K3 pad1; 2 Gaussian 3ch K5 asymmetric pad; 3 K1 DS=2
1ch; 4 blur 3ch K3 DS=2; 5 sharpen 3ch K3 DS=2 stride 2 (eff 4); 6 Sobel
1ch K3 + fused ReLU; 7 Gaussian 3ch K5 pad2 DS=4; 8 asymmetric pad +
stride (2,1) 1ch K5; 9 Sobel 3ch K3 signed intermediates; 10 gain x2 3ch
K1 DS=2; 11 Sobel 1ch K3 + fused Leaky (slope 0.25); 12 blur 32x24
stride 2 DS=2; 13 Laplacian 1ch K5 pad2 + ReLU; 14 blur 3ch K3 DS=4
32x24; 15 identity 1ch K3 stride 3 pad1. First-out latency 11-211 cycles
(11 for K1 DS=2, 211 for 32x24 K3 DS=4 -- the (K-1)-row line-buffer fill
dominates, as expected).

## STA (Nangate45 @ 3.333 ns, OpenSTA 2.2.0, `sta/run_sta_frontend.sh`)

All 6 critical-path stages MET, tns 0.000:

| stage | arrival (ns) | slack (ns) |
|---|---|---|
| fe_mult (8x16) | 1.535 | +1.756 |
| fe_add (25->13->7, 27b) | 1.566 | +1.733 |
| fe_add_wide (7->4->2, 29b) | 1.681 | +1.618 |
| fe_shift | 2.055 | +1.237 |
| fe_sat | 0.329 | +2.959 |
| fe_leaky (32x16) | 2.248 | +1.046 |

Bottleneck = fe_leaky (2.248 ns) -> system Fmax ~437 MHz
(1 / (2.248 + 0.041 setup)). Without Leaky the bottleneck is fe_shift
(2.055 ns) -> 477 MHz. Both well above the 300 MHz target; the Leaky
arrival (2.248 ns) is identical to Stage 3's act_leaky, as expected (same
32x16 multiply).

## Resources (measured, full-module Yosys synth_xilinx, `logs/stage7/`)

The frontend is small enough to synth as a whole (unlike the
multiplier-heavy Stage 1/2 modules, which timed out), so these are
measured, not inferred:

- DSP48E1: 81 (75 MAC = 3 lanes x 25, + 6 for the fused Leaky 32x16
  multiplies, each split into 2 DSP48E1).
- BRAM36E1: 12 (4 line buffers x 3 each; each 3872x24-bit buffer is
  92,928 bits = 2.52 BRAM36, rounded to 3).
- LUT: ~10,002 LCs (Yosys estimated LCs, whole design).
- FF: ~4,560 (FDCE 4494 + FDPE 18 in the frontend module, + 48 in the
  4 line-buffer instances).

For scale: the frontend uses ~4% of the XCZU19EG DSP (81/1968) and ~1%
of its BRAM36 (12/984). The LUT count (~10k) is higher than Stage 1
dense (~5.3k) because the adder tree and window logic are replicated
across 3 lanes; it is still ~2% of the 523k LUTs.

## Bug found and fixed during Stage 7 (back-to-back case transition)

Symptom: the first full golden run failed on cases 8, 13, 14 (case 8 all
outputs 0; case 13 framing errors 18 + latency underflow; case 14
timeout, 0/48 outputs) while cases 0-7, 9-12 passed, and case 13 passed
in isolation. Root cause: the testbench rewrote the config (W/H/pads ->
pw/ph) while the previous frame's stream was still draining. The stream
continues past the last output to (ph-1, pw-1) -- up to ~(eff_sy-1) rows
+ (eff_sx-1) cols of padding positions (case 12 needs ~105 cycles after
its last output) -- and a config write that shrinks pw/ph mid-stream
corrupts the stream counters: strm_c is already past the new pw-1, so it
never wraps, is_last_stream never fires, frame_active stays 1, and the
next case starts with a stale frame and wrong decimation counters. A
2-case debug run (cases 12->13) confirmed it directly: at case 13's
pre-send, frame_active=1, strm=(25,100), can_start=0, all pipeline
valid flags stuck at 1. Fix (testbench only, no RTL change): after
collecting a case's outputs, wait for `!busy` (busy = !can_start =
frame_active || !pipe_empty) before writing the next case's config, with
a 100k-cycle timeout that fails the test if busy is ever stuck. A fixed
20-cycle drain wait is NOT enough for high stride/DS cases. Lesson for
later stages: the config bus must only be written between frames, and
`busy` is the signal that says when that is -- do not rely on a fixed
cycle count. (The Stage 1-4 testbenches use the same 20-cycle pattern
and were only safe because their cases' post-output drain was short or
their pw/ph transitions were benign; the frontend's high stride/DS cases
exposed the fragility.)

Proactive fix (RTL, no behaviour change in normal operation): the line
buffer pointer wrap uses `>=` (`if (ptr >= wrap - 1)`) instead of `==`,
matching the Stage 2 fix for pointer drift when the padded width shrinks
across frames. Applied to `line_buffer_bram_fe` in `rtl/frontend.sv`.

## Regressions re-run with Stage 7

Stage 5 acceptance ALL PASS (`sw/run_stage5.sh`, `logs/stage7/regress_stage5.log`).
Stage 1-4 RTL regressions ALL PASS (`sw/run_regressions.sh`,
`logs/stage7/regress_stage1_4_full.log`, total wall 2425 s): Stage 1 PASS,
Stage 2 PASS, Stage 3 ALL PASS, Stage 4 ALL PASS (wall 1140 s).
(The first full `run_stage7.sh` attempt was interrupted mid-regression by
a VM restart -- /tmp was wiped and iverilog/yosys vanished again; the
tools were reinstalled from `~/tools-dl/` and the regressions re-run to
completion. The frontend golden + STA + Stage 5 results above are from
the uninterrupted runs and were re-confirmed.)

## Stage 7b: generic PE array (tensor-core-like) -- SRCNN 1080p30

The K^2-parallel engine (conv2d_mc and the Stage 7b outpar/cp/tappar
engines) binds 121 multipliers to the 11x11 window; only K^2 of them
do useful work. Measured MFU, single engine, 300 MHz, 1080p: SRCNN
0.66% of device peak, DnCNN 1.80%. Per-K structural cap: K=1 0.2%,
K=3 1.8%, K=5 5.1%, K=9 16.5%, K=11 24.6% of device peak. The K^2
shape IS the MFU ceiling; per-layer sizing of the same shape cannot
fix it. The K^2 engines (rtl/conv2d_outpar.sv, rtl/conv1x1_cp.sv,
rtl/conv2d_tappar.sv, rtl/srcnn_pipe.sv) are kept as reference /
golden-comparison baselines; they are not the Stage 7b deliverable.
The raster cycle model (sw/stage7b_model.py) is likewise superseded
by sw/stage7b_model_pe.py. The design, sizing derivation, and device
configs are in logs/stage7b/design.md.

### Design

One PE = a GENERIC MAC array (128 multipliers, 16x8 grid), NOT bound
to the KxK window. Two configurations of the same multiplier grid:
outer-product (TM=16 positions x TN=8 channels, one k per cycle) and
dot-product (TM=16 positions x 8 taps, one n per cycle, each row's 8
products reduce through a tree). Convolution is lowered to GEMM
(implicit im2col): M = output positions x output channels, K-dim =
C_in x K x K. The scheduler tiles M and splits K; small-K layers
parallelize by tiling M, large-K layers split the reduction across
PEs and merge partials. The PE is a pure GEMM engine (raw 32-bit
partial sums); the array's merge unit sums the K-chunk partials per
(M-tile, N-chunk) and applies the output stage (pe_emit) to the
final merged sum only.

PE size is DERIVED, not assumed (256 was an assumption, not a spec):
SRCNN 1080p30 needs 1645 useful MACs/cycle at 300 MHz (83.6% MFU),
1371 at 360 MHz, 1234 at 400 MHz. Evaluated 64 / 128 / 256 mults
per PE with the cycle model: 64 and 128 both give 33.7 fps at
300 MHz (1920 DSPs, 96.3% MFU, 0% tile-quantization loss on SRCNN);
256 gives 31.5 fps (1792 DSPs, marginal at 300 MHz). CHOICE: 128
mults/PE, 15 PEs, 1920 DSPs (97.6%) -- same throughput as 64 with
half the control overhead, and more robust than 256 to small-N
layers (0% waste on C_out=8 vs 50%).

Configurable for different FPGAs (parameters, not constants): the
RTL takes N_PE / TM / TN / TK_MAX; the scheduler takes a device
config (sw/devices.py). XCZU19EG-2 (iW-RainboW-G35D): 15x128, 300 MHz,
1080p30 target. LFE5U-85F (Lattice ECP5, low-power): 2x64, 100 MHz,
small resolutions (640x480 -> 4.0 fps, 320x240 -> 16.5 fps), the
FULL SRCNN pipeline at small scale (same compiler flow, same
precision -- the parameterization scales the parallelism, never the
feature set). XCVU37P-2FSVH2892E (VCU128, Virtex UltraScale+ HBM):
15x128, 300 MHz, 1080p30 target -- the SAME PE config as the ZU19EG,
NOT scaled up: decode is bandwidth-bound (HBM 460 GB/s delivers
~1,533 weight bytes/cycle at 300 MHz, so ~1,500 MACs/cycle saturate
it and 1,920 MACs already exceed that), so the 1,920 DSPs are 21.3%
of the VU37P's 9,024 DSPs; memory is HBM2 (8 GB) instead of PL DDR4
and the host link is PCIe Gen4 x8 with no ARM PS (host via XDMA/QDMA
+ XRT -- see docs/vcu128-bringup.md). The RTL is vendor-neutral (no Xilinx primitives;
BRAM/multipliers inferred); see the synth_ecp5 checks below.

### Files

- rtl/pe_mac.sv -- one PE (TM/TN/TK_MAX parameters, outer + dot modes).
- rtl/pe_emit.sv -- output stage (round-half-up shift, ReLU/Leaky, saturate).
- rtl/pe_array.sv -- N_PE PEs + dispatcher (round-robin) + merge unit
  (associative table, count-based completion) + output emit.
- rtl/srcnn_pipe_pe.sv -- top wrapper (config bus + tile-step program port).
- sw/sched_pe.py -- scheduling pass (tile-step generation per layer/device).
- sw/devices.py -- device configs (zu19eg, ecp5u85, vu37p).
- sw/emit.py emit_pe_program -- compiler streaming emission (PE assignments).
- tb/golden_pe.py + tb/tb_pe_mac.sv -- PE golden + TB.
- tb/tb_pe_array.sv + sw/run_pe_srcnn.py -- SRCNN chain driver (one wave
  per layer, RTL output feeding the next layer).
- tb/run_stage7b.sh -- Stage 7b verification runner (fail-fast).

### Measured results (freshly run, traceable to `logs/stage7b/`)

#### PE golden (pixel-exact) -- 11/11 PASS, 1408 values

`tb/golden_pe.py check` (run by `tb/run_stage7b.sh`): outer mode
(TK=27/32/16/11/1/8, first_k 0/1, sign_in 0/1) and dot mode
(TK=32/16/8, tn=8 and tn=1 for the SRCNN L3 shape), including a case
with the same A/B in dot mode as in outer mode (the two modes agree
exactly). Cycle counts match the model (outer = tk cycles,
dot = tn*(tk/8) cycles; the L3-like dot case with tn=1, tk=32 takes
4 cycles). Dot mode requires TN=8 in the RTL: the reduction tree
sums 8 products per row and each (n,g) step processes 8 taps, so
at TN<8 the scheduler forces outer mode (sw/sched_pe.py guard,
sw/tests/test_sched_pe.py) -- the TN=4 path was never RTL-simmed
in dot mode and is wrong there (half the K-dim dropped, adjacent
rows mixed, last row reading prod out of bounds).

#### PE-array SRCNN chain (24x24, one wave per layer) -- ALL PASS

`sw/run_pe_srcnn.py` (log `logs/stage7b/pe_srcnn_chain.log`): L1
(384 steps, 128 outputs), L2 (128 steps, 64 outputs), L3 (225 steps,
9 outputs) vs funcsim max|err| = 0 for every layer; final vs funcsim
max|err| = 0, vs float PSNR = 43.38 dB (expect ~43.5 dB; Stage 6
reported 43.51 dB for the same chain through conv2d_mc).

#### Cycle model (device-driven) -- MODEL CHECK ALL OK

`sw/stage7b_model_pe.py` (log `logs/stage7b/model_pe.log`): ZU19EG
1080p: 15x128 -> 33.7 fps @300 MHz, 40.5 @360, 45.0 @400 (MFU 96.3%);
7x256 -> 31.5 @300; 30x64 -> 26.0 @300 (TN=4 forces L3 to outer
mode; dot requires TN=8 in the RTL). ECP5 640x480 -> 4.0 fps,
320x240 -> 16.5 fps @100 MHz (MFU ~74.4%). Per-layer (128 config,
300 MHz): L1 18.45 ms, L2 7.29 ms, L3 2.83 ms; ideal frame 8.57M
cycles, frame 8.90M cycles (frame tile 64, edge x1.019).
VCU128 (VU37P, log `logs/stage7b/model_vu37p.log`): same 15x128 PE
config, so the same 33.7 fps @300 MHz, 40.5 @360, 45.0 @400 (MFU
96.3%) at 21.3% of the VU37P's 9,024 DSPs. Decode roofline
(HBM-bound, not compute-bound): HBM 460 GB/s = ~1,533 weight
bytes/cycle at 300 MHz vs 1,920 MACs/cycle, so the PE count is NOT
scaled up; 7B INT8 65.7 tok/s, 13B INT4 70.8 tok/s (both fit one
board), 13B INT8 35.4 tok/s (needs 2 boards). Covered by
`sw/tests/test_vcu128.py` (9 tests) in `tb/run_stage7b.sh` §7.
NOT covered: VU37P synthesis (OOMs on the 6 GB VM -- desktop/Mac
only), the XDMA/XRT host driver port (docs/vcu128-bringup.md), and
any RTL change (none -- the PE config is identical to the ZU19EG's).

#### Clock decision (STA, Nangate45, OpenSTA 2.2.0)

The PE's critical paths were STA'd with the accepted per-stage
method (a full-module Yosys synth of the 128-multiplier PE times out
on this VM; the paths are timed as standalone probes, /tmp/pe_crit.sv).
Outer (mult + accumulate): 300 MHz +1.393, 360 MHz +0.838,
400 MHz +0.560 -- all MET. Dot (mult + 3-level tree + accumulate,
the critical path): 300 MHz +0.788, 360 MHz +0.233,
400 MHz -0.045 VIOLATED (Fmax ~393 MHz). DECISION: 360 MHz is the
operating clock (40.5 fps predicted); 300 MHz (33.7 fps) is the
fallback; 400 MHz does not close for the dot path. Both 300 and
360 MHz meet the 30 fps goal. Logs:
logs/stage7b/sta_pe_crit_{outer,dot}_{3.333,2.778,2.500}.txt
(tns 0.000 except dot@400 MHz, tns -0.045).

#### synth_ecp5 (vendor neutrality) -- PASS

- adder_tree (N=81, W=25): PASS, 4509 cells (CCU2C 402, LUT4 2909,
  TRELLIS_FF 828), 0 problems. Log
  logs/stage7b/synth_ecp5_adder_tree.log.
- line_buffer_bram_mc (256-bit x 2048-deep, the L1/L3 size): PASS,
  memory maps to 29 DP16KD (ECP5 EBR), 70 cells total. Log
  logs/stage7b/synth_ecp5_linebuf.log. (The original form, with an
  async reset on the BRAM output register, made Yosys replace the
  memory with registers -- 2344 cells, 0 BRAM; fixed by
  `initial dout = '0;` with a plain clocked memory process.)
- Multiplier/RAM probe (/tmp/probe_ecp5.v): the 256-byte RAM maps to
  1 DP16KD. The 64x 8x16 multipliers in COMBINATIONAL form map to
  LUTs (4175 LUT4), not to ECP5 DSP blocks -- Yosys does not infer
  DSP for combinational multipliers in this flow; the PE's
  multipliers are registered (prod registers), which is the form DSP
  inference expects. Log logs/stage7b/synth_ecp5_probe.log.

#### K^2 reference engines (re-verified after the adder_tree port)

- L2 (conv1x1_cp): 4/4 PASS, 1024 values, pixel-exact.
- L3 (conv2d_tappar): K=5 cases (0, 1, 3) PASS, pixel-exact
  (including SRCNN L3 real weights); K=3 (case 2) has 18/100
  mismatches (known column-alignment limitation, documented below).
- L1 (conv2d_outpar): K=9 cases (0, 1) PASS, 5120 values,
  pixel-exact (with the V1 line-buffer change); case 2 (K=5) times
  out (known limitation, documented below).

### Bugs found and fixed during Stage 7b

1. adder_tree flat-port + Icarus sensitivity: Yosys 0.52 rejects
   unpacked array ports, so the port became a flat packed vector
   `in_flat`. A variable-base part-select inside the `always @*`
   level-1 block has incomplete @* sensitivity in Icarus and leaves
   stale sums (smoke test failed with 781 errors); fixed by
   unpacking `in_flat` with generate assigns (constant indices) and
   keeping the original `in[2*i]` references. The smoke test and
   synth_ecp5 both pass after the fix.
2. L1 position lag (pos shift-register depth MAX_K -> MAX_K-2): the
   window's bottom-right lags the stream by MAX_K-1 advances, and
   `mac` is sampled pre-edge, so the shift-register depth must be
   MAX_K-2 (the L3 already used this lag). The old depth made every
   output use the window two columns to the right.
3. L1 pipeline duplication during in_valid gaps: the prod registers
   and flag pipeline shifted every cycle, but a new window arrives
   only on `win_adv`; during a gap the group counter re-cycled the
   held window's groups and re-emitted them. Fixed with `pipe_adv`
   (a new (window, group) pair enters the pipeline only on a window
   advance or a within-window group step).
4. Leaky slope sign (L1/L2): the slope is a Q0.16 UNSIGNED fraction,
   so 0x8000 (= 0.5) must be zero-extended to 17 bits; a signed
   16-bit 0x8000 overflows to -32768 and flips the Leaky sign.
   (The MC has the same latent bug for slope >= 0.5; SRCNN uses
   ReLU, so it is out of scope here.)
5. Line-buffer BRAM inference (shared module): the async reset on
   the BRAM output register blocks Yosys BRAM inference; fixed with
   `initial dout = '0;` (see the synth_ecp5 section). Covered by
   the Stage 1-4 regressions.
6. Icarus part-select signedness (PE): an indexed part-select of a
   SIGNED vector loses its signedness when used directly in
   arithmetic -- a negative value becomes a large positive. This
   broke the dot-mode tree (tree8 function), the pe_emit shift, and
   the pe_array merge add (negative partials became huge positives
   and the merged sums saturated to 255). Fixed in all three places
   by unpacking into a signed array with generate assigns (the
   adder_tree pattern).
7. K-chunk merge placement: chaining K-chunks inside one PE (a
   `cont` flag) requires all chunks of one (M-tile, N-chunk) to land
   on the same PE, but the dispatcher assigns round-robin, so the
   chain breaks. Fixed by moving the merge into the array (an
   associative table, count-based completion) and making the PE a
   pure GEMM engine.
8. TB handshake race (tb_pe_array): sampling `ts_ready` just after a
   posedge races with the assignment (the step would be presented
   for an extra cycle and dispatched twice). Fixed by sampling
   `ts_ready` at negedge (mid-cycle, stable).

### Known limitations (documented, not hidden)

- L1 (conv2d_outpar) case 2 (K=5, pad 1, 20x12): times out (the
  K<MAX_K window/position alignment for K < MAX_K). The L1's SRCNN
  use is K=9, which passes (cases 0, 1). The K=5 case is a
  generality case, not required by SRCNN.
- L3 (conv2d_tappar) case 2 (K=3, pad 1, 10x10): 18/100 mismatches
  (column-alignment limitation for K < MAX_K). The L3's SRCNN use
  is K=5, which passes (cases 0, 1, 3, including real weights).
- The pe_array's merge table holds one entry per in-flight
  (M-tile, N-chunk); if two PEs completed the same cycle for the
  same tag, the second update would be based on the stale sum.
  Within one layer all tile-steps have the same latency and are
  dispatched >= 1 cycle apart, so same-cycle same-tag completions
  cannot occur; the TB's done_count/out_count would reveal it if
  they did.
- The PE's full-module Yosys synth times out on this VM (128
  multipliers); the STA is on the PE's critical paths (per-stage
  method), not a full-module STA. The ECP5 fit check for the
  parameterized PE array (2 PEs x 64 mults) is pending the RTL
  (the adder_tree and line-buffer checks pass; the array-level
  synth_ecp5 fit is the remaining vendor-neutrality check).

### Regressions re-run with Stage 7b

Stage 5 acceptance ALL PASS (`sw/run_stage5.sh`,
`logs/stage7b/regress_stage5.log`). Stage 1-4 RTL regressions ALL
PASS (`sw/run_regressions.sh`, `logs/stage7b/regress_stage1_4.log`,
total wall 2780 s): Stage 1 PASS, Stage 2 PASS, Stage 3 ALL PASS,
Stage 4 ALL PASS (wall 1216 s). This covers the shared-module
changes (adder_tree flat port, line_buffer_bram_mc dout reset ->
initial) -- the conv2d_mc / line-buffer modules used by Stages 1-3
are re-verified by these regressions.
