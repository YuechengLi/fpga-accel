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
