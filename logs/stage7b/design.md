# Stage 7b design (revised): generic PE array, tensor-core-like

## Why the K^2 design was abandoned

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
by sw/stage7b_model_pe.py.

## Sizing derivation (PE size is derived, not assumed)

The "256 multipliers per PE" figure was an assumption, not a spec.
The PE size is derived from (a) the target model, (b) the real-time
goal, (c) the FPGA resources.

### Required useful throughput

SRCNN 1080p = 16.4534 GMACs/frame. At 30 fps the effective
throughput must be 493.6 GMAC/s:

| clock | useful MACs/cycle | device MFU | DSPs at 96% util |
|-------|-------------------|------------|------------------|
| 300 MHz | 1645 | 83.6% | 1714 (87% of device) |
| 360 MHz | 1371 | 69.7% | 1428 (73% of device) |
| 400 MHz | 1234 | 62.7% | 1285 (65% of device) |

At 300 MHz only a near-full-device configuration qualifies with
margin: 7 PEs x 256 = 1792 DSPs gives 1792 x 0.96 = 1720 effective
MACs/cycle, just barely above the 1714 required -- essentially no
margin. A 1920-DSP configuration (97.6% of the device) gives 1843
effective MACs/cycle, a 7% margin over the requirement.

### PE size vs count

Total multipliers = N_PE x mults-per-PE must cover the requirement
with margin for tile-quantization loss (a PE's multipliers are
wasted when the tile they map to is smaller than the PE). The tile
shape must divide each layer's dimensions: L1 reduction 5184
(1x64x81), L2 reduction 2048 (64x32x1), L3 reduction 800 (32x1x25),
and the M dimension (output pixels x output channels). A PE that is
too large wastes multipliers on small tiles; a PE that is too small
costs more control/merge overhead per MAC.

TK rule (scheduler): pick the largest divisor of the K-dim that is
<= 32 (register-file bound: A = TM x TK <= 512, B = TK x TN <= 512).
L1 (K-dim 81): TK=27; L2 (K-dim 64): TK=32; L3 dot (K-dim 800):
TK=32; K=3 (576): TK=32; K=11 (121): TK=11. TK divides the K-dim,
so there is no reduction tail. The 4x4 M-tile (TM=16) divides every
SRCNN output exactly (1912/4=478, 1072/4=268, 1908/4=477,
1068/4=267), so there is no M tail either. Tile-quantization loss
for SRCNN is therefore ~0% for all three candidate sizes; the
differentiator is total DSPs + control overhead + robustness to
small-N layers.

| PE size | PEs | DSPs | % device | @300 MHz | MFU | coarse-task dispatch |
|---------|-----|------|----------|----------|-----|----------------------|
| 64 mults (TM16 x TN4) | 30 | 1920 | 97.6% | 26.0 fps | 74.4% | 0.29/cycle |
| 128 mults (TM16 x TN8) | 15 | 1920 | 97.6% | 33.7 fps | 96.3% | 0.19/cycle |
| 256 mults (TM16 x TN16) | 7 | 1792 | 91.1% | 31.5 fps | 96.4% | 0.10/cycle |

A 32-mult PE was considered and rejected: 60 PEs, coarse-task
issue rate ~0.74/cycle, more control overhead per MAC than the gain
is worth.

Robustness to small-N layers (not SRCNN; sensitivity analysis):
a hypothetical C_out=8 layer wastes 0% on a 64-mult PE (TN=4),
0% on a 128-mult PE (TN=8), 50% on a 256-mult PE (TN=16);
C_out=4 wastes 0% / 50% / 75% respectively. The 256-mult PE is the
most fragile to small output-channel counts.

Dispatch granularity matters for small PEs. The dispatch unit is
the (M-tile, N-chunk) task with the K-loop INSIDE the PE (not the
fine tile-step): coarse-task issue rates are 0.10-0.37/cycle for
all sizes, so a single-issue dispatcher is not a bottleneck. Fine
tile-step dispatch WOULD bottleneck: 1.33/cycle for the 64-mult
config, 0.85/cycle for 128, 0.57/cycle for 256 -- over 1/cycle for
the 64-mult case. That coarse-task decision is what makes small
PEs viable.

### Choice: 128 multipliers per PE, 15 PEs, 1920 DSPs

The 64-mult config falls below 30 fps at 300 MHz (26.0 fps,
MFU 74.4%): TN=4 forces L3 to outer mode, since dot mode requires
TN=8 in the RTL (the PE's reduction tree sums 8 products per row
and each (n,g) step processes 8 taps); the scheduler enforces this
(sw/sched_pe.py guard, sw/tests/test_sched_pe.py). 128 is chosen
because it meets 30 fps at 300 MHz with margin (33.7 fps) at half
the control overhead of 64 (15 dispatch slots and 15
input-buffer slices instead of 30), and it is more robust than 256
to small-N layers (0% waste on C_out=8 vs 50%).
The 256-mult config is 7% slower (31.5 fps) and marginal at
300 MHz (1720 effective vs 1714 required). Spare DSPs: 48 (bicubic
~16 + margin; the merge unit uses LUT adders). 360 MHz is the
recommended operating point for margin (40.5 fps) and is the STA
target, with 300 MHz (33.7 fps) as the fallback that still meets
the goal.

## PE microarchitecture

One PE = a GENERIC MAC array, 128 multipliers (16x8 grid), NOT bound
to the KxK window. The multiplier grid + adder network have two
configurations, selected per tile-step by the scheduler:

- outer-product (tensor-core form): rows = TM=16 output positions,
  cols = TN=8 output channels; one k per cycle; 128 products
  accumulate into 128 accumulators (TM x TN).
- dot-product: rows = TM=16 output positions, cols = 8 taps; one n
  per cycle; each row's 8 products reduce through a tree into one
  accumulator (for N=1 layers, where outer-product would leave 7 of
  8 lanes idle). Dot mode requires TN=8 in the RTL; at TN<8 the
  scheduler forces outer mode (see the guard in sw/sched_pe.py).

The per-row adder network is configurable, not duplicated: 8 adders
per row act as 8 independent accumulates (outer) or a 7-adder tree +
accumulate (dot). Accumulator file: 128 x 32-bit. Operand register
files (refilled once per tile-step, double-buffered): A = TM x TK
input values (max 512 x 8-bit), B = TK x TN weights (max 512 x
16-bit). TK is chosen per layer (see below); the files bound TK.

## GEMM lowering and tiling

Convolution -> GEMM (implicit im2col): M = output positions x output
channels, K-dim = C_in x K x K. The scheduler tiles M and splits K.
Tile hierarchy: frame tile 64x64 outputs (the three layers run in
sequence on one frame tile, so all intermediates stay on-chip) ->
M-tile 4x4 output positions (TM=16) -> tile-step (TM x TN x TK).
The 4x4 M-tile divides every SRCNN output exactly (1912/4=478,
1072/4=268, 1908/4=477, 1068/4=267), so there is no M tail.

Per-layer mapping (scheduler's choice; TK divides the K-dim, so
there is no reduction tail either):

| layer | K-dim | mode | TM | TN | TK | K-chunks | N-chunks |
|-------|-------|------|----|----|----|----------|----------|
| L1 9x9 1->64 | 81 | outer | 16 | 8 | 27 | 3 | 8 |
| L2 1x1 64->32 | 64 | outer | 16 | 8 | 16 | 4 | 4 |
| L3 5x5 32->1 | 800 | dot | 16 | 1 | 16 | 50 | 1 |
| (K=3, e.g. DnCNN) | 576 | outer | 16 | 8 | 32 | 18 | 8 |
| (K=11 dense) | 121 | outer | 16 | 8 | 11 | 11 | 8 |

Small-K layers (1x1, 3x3) parallelize by TILING M: the 2M output
positions give 128K M-tiles, far more than the PE count -- this is
what fixes the 1x1 layer that took 84% of the frame at 4 MACs/cycle
in the K^2 engine. Large-K layers use a TK that divides the K-dim
(27 for 81, 11 for 121). If a K-dim has no divisor <= 32 (e.g. prime
101), the scheduler falls back to split-K or accepts a small tail.

## Split-K merge

Used when M-tiles are too few to fill the PEs (FC layers, tiny
feature maps), not for SRCNN (M is plentiful). The scheduler splits
the K-dim across S PEs; each computes a partial sum for the same
(M-tile, N-chunk) over its K-slice. Partials (32-bit, TM x TN per
PE) go to a partial-sum buffer; a merge unit (128 adders, S-way
tree, pipelined) sums the S partials when all are present and writes
the final to the output buffer. Merge latency is hidden by
pipelining (tile t's merge overlaps tile t+1's compute). The merge
unit is built and tested with a synthetic split-K test; SRCNN does
not exercise it.

## Dispatcher and compiler scheduler

The compiler scheduling pass (sw/, new) decides per layer the mode,
TM/TN/TK, frame-tile size, K-split factor, and PE assignment, and
emits them with the program (this replaces the per-layer C_PAR
sizing). The hardware dispatcher is a small FSM + scoreboard: it
walks the frame tiles and, per tile, issues tile-steps to free PEs
(round-robin over M-tiles), tracks completion, and issues the merge
for split-K tiles. Weights are preloaded once at startup into a
shared weight SRAM and broadcast to the PEs (all PEs share weights
for SRCNN; split-K PEs get their K-slice). Inputs are staged per PE
(private input buffer slices, filled by the VDMA as the frame
streams in); each PE reads 16 B/cycle from its own slice (dot mode
needs a 128 B/cycle wide refill from the private buffer -- a
1024-bit read, banked 16x8 B). The streaming-mode compiler emission
is reworked to emit PE assignments (mode/TM/TN/TK, tile schedule)
instead of per-engine config; its structure -- one weight-load
phase, one STREAM per frame, no per-layer GO/WAIT -- is retained.

## Device configs and configurability

The same IP adapts to different-size devices by parameters, not by
rewriting RTL. The compiler/scheduler takes a DEVICE CONFIG
(DSP budget, clock, BRAM budget) as an input and produces the
tiling + PE assignment for that device (sw/devices.py).

### XCZU19EG-2 (iW-RainboW-G35D) -- the 1080p30 target

- Part: XCZU19EG-2, 1968 DSP, 523K LUTs, 984 BRAM36.
- Config: 15 PEs x 128 mults = 1920 DSPs (97.6%), TM=16, TN=8,
  frame tiles searched over {8, 16, 32, 64} (best 64 for 1080p).
- Target: 1920x1080. Verification: RTL (golden + sim).

### LFE5U-85F (Lattice ECP5, low-power) -- the small device

- Part: LFE5U-85F (largest ECP5, still low-power). Budget from the
  Lattice ECP5 Family Data Sheet (FPGA-DS-02012-3.2, Aug 2023) and
  the ECP5 product selection guide: 84K LUTs, 208 sysMEM blocks
  (18 Kbits each) = 3,744 Kbits EBR, 669 Kbits distributed RAM, 156
  18x18 multipliers (each sysDSP slice supports half a 36x36, two
  18x18, or four 9x9 multipliers; the 156 figure is the selection
  guide's 18x18 count and is used conservatively).
- Config: 2 PEs x 64 mults = 128 multipliers (82% of 156), TM=16,
  TN=4, clock 100 MHz (conservative planning assumption for a
  first-pass ECP5 design; sysDSP blocks run faster, and the firm
  timing number needs a nextpnr run, out of scope here -- fps
  scales linearly with the clock). 2 PEs, not 1, so the multi-PE
  dispatch and merge are exercised. TN=4 keeps small-N layers
  efficient.
- Functionally complete at small scale: the Lattice configuration
  runs the FULL IP -- the whole SRCNN pipeline (all 3 layers), the
  same compiler flow, the same quantization/precision -- just with
  fewer/smaller PEs and lower fps/resolution. The parameterization
  scales the parallelism, never the feature set. No layer, operator,
  or compiler path is dropped.
- Targets (small-resolution images, NOT 1080p; the ZU19EG owns
  1080p30):
  - 640x480: 5.1 fps at 100 MHz, MFU 95.9% (frame tile 32, 300
    tiles, edge x1.030). Total 2.3924 GMACs/frame (L1 1.5464 +
    L2 0.6109 + L3 0.2351).
  - 320x240: 21.1 fps at 100 MHz, MFU 95.4% (frame tile 8, 1131
    tiles, edge x1.000). Total 0.5796 GMACs/frame.
  - Frame-tile edge waste is larger at small resolutions (64x64
    tiles: 640x480 edge x1.098, 320x240 edge x1.132), so the model
    searches frame tiles {8, 16, 32, 64} per (device, resolution)
    and picks the best, trading edge waste against the 150-cycle
    per-tile layer-switch overhead.
- Verification: cycle model + synth_ecp5 fit check (model+synth).
  The golden/RTL verification runs on the ZU19EG-sized
  configuration; the Lattice configuration is verified by the cycle
  model and the synth_ecp5 fit check, NOT by RTL simulation.

### RTL parameters (rescale by parameter, not rewrite)

- `pe_array`: N_PE (number of PEs), PE_MULTS (multipliers per PE),
  TM/TN (tile shape per PE), TK bound (register-file depth).
- The scheduler reads the device config (DSP budget, clock, BRAM
  budget) and emits mode/TM/TN/TK, frame-tile size, K-split factor,
  and PE assignment for that device. The same compiler flow serves
  both devices; only the device config changes.

## Vendor neutrality

The RTL must be vendor-neutral: no Xilinx-only primitives or IP;
BRAM and multipliers must be inferred. Audit of the existing RTL
(2026-11-20): a grep for `RAMB[0-9]|DSP48|BUFG|IBUF|OBUF|FDRE|
FDSE|LUT6|UNISIM|XPM_|xpm_|RAM64|RAM32` across rtl/*.sv returns
nothing -- no Xilinx primitives. `(* ram_style = "block" *)` is
present on the line-buffer memories (rtl/conv2d_mc.sv); it is a
Xilinx-style attribute that Yosys ignores (inference depends on the
code pattern, not the attribute), and the synth check confirms
whether the memories map to ECP5 EBR.

- adder_tree (rtl/adder_tree.sv): the input port is a flat packed
  vector `in_flat`, because Yosys 0.52 rejects unpacked array
  ports. The flat vector is unpacked with generate assigns
  (constant indices); a variable-base part-select inside the
  always @* level-1 block has incomplete @* sensitivity in Icarus
  and leaves stale sums (smoke test failed with 781 errors before
  the generate-unpack fix). synth_ecp5 (N=81, W=25): PASS,
  4509 cells (CCU2C 402, LUT4 2909, TRELLIS_FF 828), 0 problems;
  log logs/stage7b/synth_ecp5_adder_tree.log. Smoke test:
  PASSED (200 random vectors x N=160/81/64/25).
- Line-buffer BRAM inference (rtl/conv2d_mc.sv, line_buffer_bram_mc):
  FIXED and verified. The original form (async reset on the BRAM
  output register `dout` in the same always_ff as the mem access)
  made Yosys replace the memory with registers (256-bit x 2048-deep
  -> 2344 cells, 0 BRAM, and the full synth never finished). The fix
  is `initial dout = '0;` (power-up 0, same as the reset gave) with
  the mem process a plain `always_ff @(posedge clk)` -- no async
  reset on the BRAM output register. synth_ecp5 (256-bit x
  2048-deep, the L1/L3 size): PASS, memory maps to 29 DP16KD (ECP5
  EBR), 70 cells total; log logs/stage7b/synth_ecp5_linebuf.log.
  Simulation is unchanged (L3 K=5 cases still PASS; the initial
  block gives the same power-up 0 the reset gave, so the line-buffer
  chain cannot poison with X). This is a shared-module change
  (conv2d_mc Stages 1-3), so the Stage 1-4 regressions must cover
  it before commit.
- Multiplier/RAM probe (/tmp/probe_ecp5.v, not design RTL): the
  256-byte RAM maps to 1 DP16KD (EBR) -- PASS. The 64x 8x16 signed
  multipliers in COMBINATIONAL form (always_comb, no registers)
  map to LUTs (4175 LUT4), not to ECP5 DSP blocks (no TRELLIS_DSP)
  -- Yosys does not infer DSP for combinational multipliers in
  this flow. The PE datapath registers its products (like the
  L1/L2/L3 engines), which is the form DSP inference expects; the
  PE-array synth_ecp5 check will confirm whether the registered
  multipliers map to DSP, and that check is the gate for the PE
  RTL (see the RTL plan).

## Cycle-model prediction (sw/stage7b_model_pe.py)

Per-layer time at 300 MHz, 15x128 config: L1 18.45 ms, L2 7.29 ms,
L3 2.83 ms; ideal frame 8.57M cycles; with frame-tile edge waste
(1.9%), layer-switch overhead (150 cycles x 510 tiles, 0.9%) and
dispatch bubbles (1%): 8.90M cycles = 29.66 ms = 33.7 fps, MFU
96.3% (vs the K^2 engine's measured 0.66%). Operating point: 300
MHz MEETS the goal (33.7 fps); 360 MHz is recommended for margin
(40.5 fps) and is the STA target, with 300 MHz as the fallback.
The model is cross-checked against the RTL sim on a 24x24 frame
(measured cycles/frame vs modeled, then extrapolated).
Log: logs/stage7b/model_pe.log (MODEL CHECK ALL OK).

## Clock decision (STA, Nangate45, OpenSTA 2.2.0)

The PE's critical paths were STA'd with the accepted per-stage
method (a full-module Yosys synth of the 128-multiplier PE times out
on this VM, so the paths are timed as standalone probes,
/tmp/pe_crit.sv -- not design RTL). Two paths:
- outer (mult + accumulate): 300 MHz +1.393, 360 MHz +0.838,
  400 MHz +0.560 -- all MET.
- dot (mult + 3-level tree + accumulate, the PE's critical path):
  300 MHz +0.788, 360 MHz +0.233, 400 MHz -0.045 VIOLATED.
The dot-mode tree is the bottleneck (Fmax ~393 MHz). DECISION:
360 MHz is the operating clock (closes with +0.233 ns on the
critical path; 40.5 fps predicted); 300 MHz (33.7 fps) is the
fallback; 400 MHz does NOT close for the dot path and is not used.
Both 300 and 360 MHz meet the 30 fps goal. Logs:
logs/stage7b/sta_pe_crit_{outer,dot}_{3.333,2.778,2.500}.txt
(tns 0.000 except dot@400 MHz, tns -0.045).

## Merge unit (why the K-chunk merge lives in the array, not the PE)

The first PE version chained K-chunks inside one PE (a `cont` flag:
the accumulator was kept across tile-steps). That requires all
K-chunks of one (M-tile, N-chunk) to land on the SAME PE, but the
dispatcher assigns tile-steps round-robin to free PEs, so the
chunks of one output land on different PEs and the chain breaks
(verified 2026-11-20: the SRCNN chain gave saturated outputs --
the merge added the same partial twice / the chain was broken).
The fix is the design.md merge unit: the PE is a PURE GEMM engine
(it outputs the raw 32-bit partial per tile-step, with the bias
added once on first_k); the pe_array's merge unit is an associative
table of in-flight (mtile, nchunk) entries that sums the K-chunk
partials (the COUNT, not the last_k flag, decides completion, since
partials may complete out of order) and applies the output stage
(pe_emit: round-half-up shift, ReLU/Leaky, saturate) to the FINAL
merged sum only. The PE's `cont`/`last_k`/output-stage were removed
when the merge unit was added; the PE golden (tb/golden_pe.py) tests
partial sums only, and the output stage + merge are verified at the
pe_array level (sw/run_pe_srcnn.py). The split-K merge for
parallel K-splits (FC layers, tiny feature maps) uses the same
merge table; SRCNN does not exercise split-K (its K-chunks are
sequential tile-steps merged by count).

## Icarus signedness pitfalls (found during Stage 7b)

Two Icarus quirks, both fixed the same way (the adder_tree pattern:
unpack a flat vector into a signed array with generate assigns, and
do arithmetic on the signed array elements, never on a part-select
directly):
- A variable-base part-select inside an `always @*` block has
  incomplete sensitivity and leaves stale sums (adder_tree: the
  flat-port version failed the smoke test with 781 errors; the
  generate-unpack version passes).
- An indexed part-select (`v[base +: W]`) of a SIGNED vector loses
  its signedness when used directly in arithmetic: a negative value
  becomes a large positive. This broke the dot-mode tree (tree8
  function: tree8(-5,10,-3,7,0,0,0,0) gave -67108855, not 9),
  the pe_emit shift, and the pe_array merge add (negative partials
  became huge positives and the merged sums saturated to 255).
  The fix in all three places is the generate-unpack into a signed
  array. A plain assignment of a part-select to a signed variable
  (bit-pattern copy) is safe; using the part-select directly in an
  addition / shift / multiply is not.

## RTL plan (status 2026-11-20)

1. `rtl/pe_mac.sv` -- one PE (parameterized multipliers/modes),
   standalone golden (random GEMM tiles + SRCNN layer tiles). DONE
   and verified: 11 cases, 1408 values, outer + dot, pixel-exact;
   cycle counts match the model (outer = tk, dot = tn*(tk/8)).
   The PE is a pure GEMM engine (partial sums); it has no cont /
   last_k / output stage (those moved to the array's merge unit).
2. `rtl/pe_emit.sv` -- output stage (round-half-up shift, ReLU /
   Leaky, saturate), applied by the merge unit to the final merged
   sum. DONE; verified via the pe_array chain.
3. `rtl/pe_array.sv` -- N_PE PEs + dispatcher (round-robin, with the
   pe_start gap fix) + merge unit (associative table, count-based
   completion) + output emit. DONE and verified: SRCNN 24x24 chain
   (L1/L2/L3, one wave per layer, RTL output feeding the next layer)
   vs funcsim max|err| = 0, PSNR vs float 43.38 dB.
4. `rtl/srcnn_pipe_pe.sv` -- top wrapper (config bus + program port).
   DONE (compiles); verified via the pe_array chain driver, which
   drives the tile-step program through the wrapper's program port.
   There is no separate wrapper TB (the wrapper adds no datapath
   logic beyond the config registers).
5. Compiler: scheduler pass (sw/sched_pe.py) + streaming emission
   (sw/emit.py emit_pe_program, wrapping the scheduler) + demo +
   vector generator + TB. DONE; the scheduler emits the tile-step
   program, the driver (sw/run_pe_srcnn.py) runs it through the
   array, and the PE streaming tests (sw/tests/test_streaming_pe.py)
   pass. The K^2 streaming emission (emit_program(streaming=True))
   is kept as reference for the superseded rtl/srcnn_pipe.sv.
6. STA: PE critical paths at 300/360/400 MHz (see "Clock decision"
   above); 360 MHz is the operating clock. Regressions; run script;
   docs; commit.

## Per-device verification status

| device | config | golden/RTL sim | cycle model | synth_ecp5 fit |
|--------|--------|----------------|-------------|----------------|
| XCZU19EG-2 | 15x128, 300 MHz | yes (RTL-verified) | yes (33.7 fps @1080p) | n/a (Xilinx target) |
| LFE5U-85F | 2x64, 100 MHz | no (model+synth only) | yes (5.1 fps @640x480, 21.1 fps @320x240) | in progress (PE-array synth check pending the RTL) |
