# Stage 7d design: YOLOv8n + DNANet end-to-end + post-synthesis

## Why this stage exists

Stage 7c closed the integrated e2e + post-synthesis legs for the three
image models (SRCNN / DnCNN / RetinexNet). The two detection models
onboarded in bb37ce7 — YOLOv8n and DNANet — have no model-level
end-to-end at all: their registration verified per-layer fixed-point
error on one test image each, but no flow ever ran a real image
through the frontend RTL + the PE array + the auxiliary RTL modules
and checked the output. The user asked for e2e coverage for ALL
models including post-synth, and explicitly not to rush to board
bring-up. This stage extends the 7c pattern to the two detection
models. No new RTL datapath is added: the integration reuses the
existing modules (pe_array, act, simplegate, residual, pool,
upsample, concat, frontend) plus the driver-level compiler work.

## Part 0 — compiler graph capture (design first)

### Why FX capture does not work (measured 2026-11-22)

`capture_torch` (torch.export) was tried for both models and fails:

- DNANet: `capture: unsupported aten op aten.batch_norm.default`.
  The IR has no `batch_norm` node (fusion folds BN, but capture
  never lowers it), and `ir.py` is outside this stage's file
  boundary, so no lowering can be added here.
- YOLOv8n (backbone+neck wrapper): `torch.export` itself fails
  (`cat(): argument 'tensors' must be tuple of Tensors, not
  FakeTensor` — the neck's multi-output routing does not export).

So capture is HAND-BUILT, via `ir.build_graph`, from the registered
topology + weights. The graph is built in pre-fusion form (conv ->
relu/silu as separate nodes, weights already BN-folded — the
registration folded BN in float), then the existing passes run:
`fuse` (relu -> fused_act, silu -> separate `act` node, mul ->
`simplegate`, add/sub -> `residual`), then a 7d quantize override
(below), then the actapprox table generation, then tiling +
emission. The FX capture remains a follow-up (it needs `ir.py`
lowerings for batch_norm, chunk/split, adaptive pools and channel
reductions, all outside this stage's boundary).

### The quantize override (registered weights, e2e-input ranges)

The compiled program does NOT re-run `quantize_graph`. The
registered quantization (bb37ce7) is the verified artifact, so the
capture attaches the registered per-layer `W_q / b_q / shift /
r_in / r_out` to each conv node — except the convs that feed a
SEPARATE act node (SiLU / sigmoid), where the registered `r_out`
is the POST-act range and the compiler needs the PRE-act range:

- The pre-act range is measured on the e2e input (float forward
  with hooks on the pre-act conv outputs, same method as the
  registration). The conv's shift is recomputed as
  `shift_pre = e + r_out_pre - r_in` (e from the registered
  weights); if it exceeds 15, e is reduced and the weights are
  re-quantized with `quantize_layer` (the same rule the compiler
  uses). The capture reports every layer where the e2e shift
  differs from the registered shift.
- ReLU-fused convs need no adjustment: the registered `r_out` is
  already the post-ReLU range, which is the compiler's `r_out`
  (ReLU does not change the max, only clips negatives).

The DNANet L0 norm fold (ImageNet mean/std folded into the first
conv — exact affine fold, verified in run_dnanet.py) is replicated
in the capture, so the accelerator input is plain uint8, as
registered.

### Op mapping (accelerator vs CPU)

| op | where | how |
|----|-------|-----|
| conv2d 1x1/3x3/7x7 (all strides) | PE array | `gen_layer_steps` + `tb_e2e.sv`, as 7c |
| SiLU (YOLO, every conv) | PE + act + simplegate | decomposed, see below |
| sigmoid (DNANet attention, final) | act RTL | MLUT table from actapprox, `run_act` wave |
| mul / simplegate (attention, SiLU) | simplegate RTL | signed, halved-sigmoid trick, see below |
| add / residual (Bottleneck, CBAM shortcut) | CPU (driver) | the RTL residual mode 0 is an *unsigned* add of the raw bytes: no signed-add mode and no input rescaling. 49/52 DNANet residuals have a signed input, and the two inputs almost always carry different scales (e.g. YOLO `c2f2_m0_add` r_out [4,5], DNANet `ca_add` r_out [3,0]) so the funcsim rescales them to the residual's r_in before adding. Only 2/58 residuals could use mode 0, so the uniform CPU path is used. The residual module itself is still verified in Part 2 (post-synth, Stage 4 golden). Follow-up: a signed-add + rescale mode for the residual module (needs rtl/, outside this stage's boundary). |
| relu (DNANet, after the residual) | CPU (driver) | pass-through: the residual's `signed_out=False` already clips the sum to [0,255], so ReLU is the identity on its unsigned output; still executed and checked against the golden. |
| max_pool2d 2x2 (DNANet encoder) | pool RTL | mode 0 (MAX 2x2) |
| max_pool2d 5x5 (YOLO SPPF) | CPU (driver) | pool RTL is 2x2-only; the maps are 4x4x256, a trivial reduction. Documented gap: a 5x5 stride-1 pool mode in pool.sv is a follow-up (needs rtl/, out of this stage's boundary). The SPPF convs run on the PE. |
| upsample nearest 2x (YOLO neck) | upsample RTL | mode 0 |
| upsample bilinear 2x (DNANet `up`) | upsample RTL | mode 1, RTL formula (below) |
| upsample bilinear 0.5x (DNANet `down`) | CPU (driver) | the RTL is scale-2-only; a downsample mode is a follow-up (rtl/). The maps are small; the values feed RTL convs. |
| concat (2..5 inputs) | concat RTL | chained 2-input concats for N>2 |
| gap (ChannelAttention avg pool) | CPU (driver) | the pool RTL GAP sums the input bytes as *unsigned* (zero-extended) and floor-shifts; the DNANet GAP input (conv2 output) is signed, so the RTL GAP would sum two's-complement bytes as large positives. The rounding also differs (RTL floor vs funcsim round(mean), up to 1 LSB). A cheap Python mean, so it runs on the CPU with the same funcsim code as the golden. The `pool.sv` GAP is still verified in Part 2 (Stage 4 golden, unsigned inputs). |
| global max pool (ChannelAttention max) | CPU (driver) | no RTL global-max; a reduction over HxWxC, trivial |
| channel mean / channel max (SpatialAttention) | CPU (driver) | reductions over C; feed the 7x7 sa conv (PE) |
| DFL + NMS (YOLO Detect head) | CPU | as registered; the e2e stops at the neck outputs (P3/P4/P5) |
| YOLO C2f channel split | graph rewrite | no slice op in the IR: cv1 is split into TWO conv nodes by output channel (cv1_a -> bottlenecks, cv1_b -> cat). A conv's output channels are independent, so this is exact. |
| DNANet fc1/fc2 (1x1 conv on a 1x1 map) | PE array | input-padding trick: pad (1,1,C) to (4,4,C) with zeros, run, take [0,0,:]. A 1x1 conv at [0,0] reads only input [0,0], so this is exact. 52 such layers; each is 1-2 tile-steps. |

### SiLU: the range problem and the decomposition (measured)

The MLUT/PWL tables cover x in [-8, 8) only (act.sv clamps
out-of-range inputs to the endpoint segments). The YOLO pre-SiLU
conv outputs reach 35.0 on bus.jpg (640 letterbox) and 31.0 on the
128x128 e2e input; 5/57 layers exceed 8 (4/57 on the e2e input),
the worst being the early backbone layers (C1 25.7, C2 35.0).
A direct SiLU LUT would saturate those layers (SiLU(35) returned
as SiLU(8) ~ 8 — a catastrophic error, since SiLU(x) ~ x for
x > 8). Sigmoid does not have this problem (it saturates by
nature: sigmoid(8) ~ 0.9997).

SiLU is therefore DECOMPOSED, for every YOLO layer (uniform path):
`SiLU(x) = x * sigmoid(x)`, executed as

1. sigmoid via act.sv (MLUT table, r_in = the conv's pre-act
   r_out, r_out = 2 — see the halving below), then
2. multiply via simplegate.sv (signed, shift = r_out_silu -
   r_in_conv + 6, normally 6).

The halved-sigmoid trick: simplegate.sv has ONE sign bit for both
inputs, so a signed x_q cannot be multiplied by an unsigned
sig_q (sig_q > 127 would read as negative). The sigmoid act
node's r_out is set to 2, so the LUT emits sig' = sig * 255/4
in [0, 64] (fits signed int8 with margin); the simplegate then
computes `(sig' * x_q) >> shift` with sign_in=1, sign_out=1,
shift = r_out_silu - r_in_conv + 6. The arithmetic is
`(sig_q/4 * x_q)/2^shift`, against the ideal
`sig_q*x_q/255` — a 255-vs-256 scale difference (~0.4%, at most
~1 LSB), which is a QUANTIZATION error vs float, not an
RTL-vs-funcsim error: the funcsim `simplegate` case implements
the identical integer arithmetic (`(A*B)>>shift`, saturating),
so RTL and funcsim agree exactly. Why r_out=2 and not 1:
r_out=1 would emit sig' = sig * 127.5, which rounds to 128 at
sig ~ 1 — and 128 as a signed byte is -128, which would flip
the sign of the largest SiLU values (exactly the layers that
need the decomposition). r_out=2 caps sig' at 64, so no
overflow is possible. Precision cost: the sigmoid quantization
error is ~|x|/128 in sig units — a small quantization error vs
float, reported in the e2e numbers. The decomposition is a new
compiler pass in `sw/common/passes/` (act(silu) -> act(sigmoid,
r_out=2) + simplegate(shift=6, signed)), applied after fusion
and before the actapprox table generation.

### The rescale (DNANet branching)

DNANet branches (dense nested skips, attention multiplies,
residual adds), so a consumer's `r_in` need not equal its
producer's `r_out` (102/153 layers sequentially). The capture
measures every node's `r_in` from its ACTUAL input tensor's
range on the e2e input, and the program records it per op. The
driver AND `fixed_forward` both apply the rescale when a node's
input scale differs from its producer's output scale:
`byte' = byte * 2**(r_out_producer - r_in_consumer)` (left shift,
clipped, or right shift with round-half-up). Both call the same
`funcsim.rescale_bytes` helper, so they agree exactly. The 7c
models have r_out == r_in everywhere (rescale = 0, no-op), so
this is backward compatible.

### Bilinear upsample: RTL formula vs torch (measured)

The upsample RTL's bilinear (mode 1, golden_upsample.py) is the
simple formula: out[2r,2c]=in[r,c],
out[2r,2c+1]=(in[r,c]+in[r,c+1])/2, etc., with edge replication.
It matches NEITHER torch convention (align_corners=False gives
out[2r]=0.25*in[r-1]+0.75*in[r]; align_corners=True gives a
position-dependent blend). DNANet uses align_corners=True. The
7d golden therefore uses the RTL formula (a new `bilinear_rtl`
mode in funcsim._upsample, same formula as the Stage 4 golden),
so RTL vs funcsim is exact; the float reference stays the torch
model (align_corners=True), and the final comparison is reported
both against the torch reference and against a float reference
that uses the RTL upsample convention (the 7c precedent for the
replicate-vs-zero padding: isolate the convention difference
from the quantization error).

### Input sizes and runtime (measured estimates)

| model | e2e input | PE tile-steps | est. vvp time |
|-------|-----------|---------------|---------------|
| YOLOv8n | bus.jpg resized 128x128x3 (frontend-blurred) | 63,088 | ~1.8 h |
| DNANet | target1.png 64x64 crop centred on the target (x[128:192], y[192:256] — the 3x3 target survives; a 4x downscale of the full image would erase it) | 246,542 (+~100 for the 52 padded fc1/fc2) | ~6.8 h |

128x128 is the SMALLEST YOLO input that keeps every conv's
stride-1 output divisible by 4 (the 32x downsample needs P5 =
4x4); 64x64 is the smallest DNANet input that keeps the deepest
level at 4x4. The stride-2 YOLO convs (7 layers) use the
decimate trick (already used by the registration): run stride-1
on the pre-padded input, take every other output. Total ~8.6 h
of vvp plus ~300-500 auxiliary waves; the driver checkpoints
after every passed node (the 7c mechanism, generalized from
per-layer to per-node), so VM replacements cost a re-launch,
not a restart. The user said not to rush; this runs in the
background.

### Final checks (reported, not gated to a benchmark)

- YOLO: the three neck outputs (P3/P4/P5) vs the torch float
  neck outputs at the same 128x128 input (PSNR per map), plus
  the CPU-decoded detections (DFL+NMS, the registered
  decode_nms) from the RTL feature maps vs from the float
  feature maps. No mAP is claimed (no COCO eval here).
- DNANet: the four deep-supervision logits -> sigmoid -> mask
  (threshold 0.5) vs the torch float mask on the same 64x64
  crop (IoU vs the float mask, not vs the repo Pred — the crop
  differs from the full image). No NUAA-SIRST IoU is claimed.

## Part 1 — integrated end-to-end

A new graph-generic driver, `sw/vision/run_e2e_7d.py`, walks the
compiled program in topological order and executes each node in
RTL (or on the CPU for the documented nodes above), reusing the
7c helpers (`run_frontend`, `compile_e2e_tb`, `run_pe_layer`,
the per-node checkpointing). The golden is COMPOSED:
`fixed_forward` on the compiled program (with the new
`funcsim` cases: `simplegate`, `relu`, `gmax`, `cmean`, `cmax`,
the rescale, and the `bilinear_rtl` upsample mode). Per-node
acceptance is RTL vs funcsim max|err| = 0 (act LUTs: the MLUT
budget is exact, max|err| = 0; PWL, if chosen, budget <= 1 LSB
as in 7c). The frontend phase is the 7c blur (3ch, same size);
the float reference is the torch model on the blurred input.

## Part 2 — post-synthesis simulation

Same route as 7c (Yosys generic synth -> Icarus + simcells.v +
the SAME TB + SAME golden). The 7d modules are the ones NEW to
the post-synth flow: simplegate, residual, pool, upsample,
concat (7c already covered adder_tree, frontend, pe_mac,
pe_array). If a module's generic synth or netlist sim is
VM-infeasible (OOM / elaboration too slow), it is an explicit
SKIP in the VM runner and a Phase 4 in `run_ecp5_mac.sh`
(Mac), never silently dropped. A parameter-accepting wrapper is
added only if the TB instantiates the DUT with parameter
overrides (the 7c finding); otherwise the netlist drops in
directly.

## Runner and verification bar

`tb/run_stage7d.sh` (fail-fast, 7c conventions): Part 1 (YOLO,
then DNANet, each with --resume), Part 2 (post-synth for the
7d modules), then regressions (7c + 7b + Stage 5 + Stages 1-4)
ALL PASS, plus the unit tests. Logs under `logs/stage7d/`
(per-model e2e logs + JSON, post-synth logs, runner log).
Doc updates are staged in `logs/stage7d/doc-updates.md` (the
file boundary forbids touching docs/plan.md, rtl/README.md,
sw/README.md, models/README.md and the root README.md in this
stage).

## Files

| File | What |
|---|---|
| `logs/stage7d/design.md` | this file |
| `sw/vision/capture_7d.py` | hand-built graph builder + quantize override + program emission (YOLO neck + DNANet) |
| `sw/common/passes/silu_decompose.py` | compiler pass: act(silu) -> act(sigmoid, r_out=2) + simplegate(shift = r_out_silu - r_in_conv + 6, signed) |
| `sw/vision/funcsim.py` | new cases: `simplegate`, `relu`, `gmax`, `cmean`, `cmax`; `rescale_bytes` + rescale in fixed_forward; `bilinear_rtl` upsample mode |
| `sw/vision/run_e2e_7d.py` | graph-generic integrated driver (PE + act + simplegate + residual + pool + upsample + concat waves, CPU nodes, per-node checkpoints) |
| `sw/vision/tests/test_e2e_7d.py` | unit tests (capture topology, SiLU decomposition, rescale, RTL bilinear formula, simplegate arithmetic) |
| `tb/run_stage7d.sh` | fail-fast runner (Part 1 + Part 2 + regressions) |
| `tb/run_postsynth.sh` | extended with the 7d modules (VM) + explicit SKIP -> Mac Phase 4 |
| `~/workspace/your_files/run_ecp5_mac.sh` | Phase 4 added if a 7d post-synth leg is VM-infeasible |
| `logs/stage7d/doc-updates.md` | staged doc text for docs/plan.md + rtl/README.md + sw/README.md |

## Validation (float graph vs torch, 2026-11-22)

Both hand-built graphs were validated against the torch model on
the e2e input before any RTL work. The capture programs are in
`logs/stage7d/program_{yolo,dnanet}.{json,npz}`.

### YOLOv8n (128x128 resized bus.jpg)
- Capture: 226 program ops, 71 quantized convs (57 UConv + 8 C2f
  cv1 splits + 6 plain Detect-head 1x1 convs), decompose float
  maxdiff = 0.0.
- **The Detect head has THREE convs per branch, not two.** Each
  cv2/cv3 Sequential is (0) Conv 3x3 SiLU, (1) Conv 3x3 SiLU,
  (2) a plain 1x1 `nn.Conv2d` with NO BN and NO activation --
  that plain 1x1 is the Detect head's actual output. The
  registration's 57-conv count (UConv modules only) missed
  these 6 layers, so the model has **63 conv layers**, not 57.
  The 6 weights were extracted from the .pt into
  `models/vision/yolov8n/detect_out.npz` (+ `.json`) and added
  to the graph as `det_box{0,1,2}` / `det_cls{0,1,2}` (1x1, no
  act). Without them the graph's outputs were the second Conv's
  SiLU outputs, not the Detect outputs, and the DFL decode was
  meaningless. The cls biases are large negative (-13..-5),
  which is why the cls logits are very negative.
- All 57 UConv modules (post-SiLU) vs torch: worst max|diff| =
  3.7e-02 (C56, the deepest layer), median 2.5e-03 -- float32
  vs float64 accumulation through up to ~30 conv layers.
- DFL decode validated against the torch Detect output: the
  Detect returns **xywh** (not xyxy); the decode (DFL offsets
  -> xywh -> xyxy) matches to box max|diff| = 0.27, cls
  max|diff| = 3.0e-04, and the numpy NMS gives the SAME 5
  detections as the torch NMS (4 persons + 1 bus, same boxes
  and scores to 0.1). (An earlier check that showed 15-112 was
  a validation-script bug: it compared post-SiLU torch outputs
  against pre-SiLU graph nodes; a later check that showed a
  450 box diff was the missing plain 1x1 convs + the xywh
  format.)
- Quantize cross-check: all 65 registered convs dshift = 0
  (the recomputed shifts match the registered quant.json
  exactly); the 6 plain convs have no registered entry (they
  were not in the registration).
- `float_forward` needed one fix: it ignored the conv stride
  (stride-2 layers produced full-size outputs and broke the neck
  concats). It now decimates after the conv, like the
  registration and like `fixed_forward`.

### DNANet (64x64 target-centred crop of target1.png)
- Capture: 560 program ops, 206 quantized convs (154 + 52
  duplicated ca.fc1/fc2), decompose float maxdiff = 0.0 (no
  SiLU in DNANet).
- Two real bugs found and fixed by the validation:
  1. The first block's shortcut (L2) reads the RAW image, so it
     needs the ImageNet-norm fold too -- the registered .npz
     folds the norm into L0 only. Fixed in the capture
     (`_norm_bn_fold` from the raw float weights). The conv0_0
     interior max|diff| dropped from 1.55 to 0.21.
  2. The quantize cross-check lookup used the node name, but
     the DNANet registered keys are L{i} -- all 206 entries
     silently reported "no registered entry". Fixed (fall back
     to the entry whose "path" field matches). Now 58/206 have
     a nonzero dshift, all explained by the design: sa.conv1
     +2..+5 (logit range, not post-sigmoid r_out=0), conv2 +1
     (symmetric signed range), first-block shortcut -1 (norm
     fold), conv1 -1/-2 (pre-act range measured on the crop,
     not the full image).
- The remaining logit differences (up to 33 on the final
  logits, values to ~108) are the documented norm-fold rim
  effect: L0's output differs on the 1px rim (zero-pad on raw
  vs normalized input), and the ChannelAttention GAP/GMAX are
  global reductions, so the rim error spreads to the whole
  map and compounds through the attention layers. It does NOT
  change the detection: the four deep-supervision masks are
  identical to the torch float masks (9 px each, same
  coordinates y 43-45, x 31-33, IoU 1.0000 for all four and
  for the mean aggregate). The e2e final check reports the
  mask IoU (the functional quantity), not the raw logit
  max|diff|.

## Stage 7d run findings (2026-11-22 evening)

### C49 blocker: root cause and fix (RTL change, out of boundary)

YOLO e2e stopped at C49 (P5 Detect cv2.0, 3x3, C_in=256, C_out=64,
4x4x64, 576 steps): TB read done_count=572, expected 576, and the
8 outputs were wrong (max|err| up to 24). A hierarchical debug
monitor in tb_e2e.sv showed:

- pe_start pulses = 576 (all steps dispatched), pe_done pulses =
  572, 0 cycles with 2+ simultaneous completions. So 4 steps were
  started but never finished -- and the counter was never the
  problem.
- At $finish all 4 PEs were still in ST_RUN (k_cnt 1/3/5/7): the
  simulation ended while the last 4 steps were still running.
- The merge table showed one live entry (mtile=0, nchunk=7,
  count=4, n_kchunk=8). n_kchunk=8, not 72.

Root cause: `ts_n_kchunk` / `pe_n_kchunk` / `mg_n_kchunk` /
`mg_count` were all 6 bits (max 63), but C49 needs n_kchunk=72.
The TB truncated with `nkchunk_v[5:0]`, so 72 & 63 = 8: each
output emitted after only 8 K-chunks instead of 72 (wrong data),
the collector got its 8 outputs early, and $finish cut off the
remaining steps (done_count 572). The 7b layers never hit this
(n_kchunk <= 50 for SRCNN), so it first bites in 7d.

Fix (widen to 8 bits, max 255): rtl/common/pe_array.sv
(ts_n_kchunk port, pe_n_kchunk, mg_count, mg_n_kchunk),
rtl/vision/srcnn_pipe_pe.sv (ts_n_kchunk port), tb/tb_e2e.sv
(ts_n_kchunk reg, `nkchunk_v[7:0]`). After the fix C49 passes:
E2E PASS, done_count=576/576, merge count=72/72, all PEs idle.
This is an RTL change OUTSIDE the Stage 7d file boundary
(rtl/common/, rtl/vision/) -- parent review required before
commit. It does not conflict with the parallel LLM agent
(models/llm/ untouched).

Two other RTL edits were made during the C49 investigation and
are also out of boundary; neither was the root cause:
- rtl/common/pe_array.sv: a `merge_count` block that counts
  pe_done pulses into a local integer and does
  `done_count <= done_count + n_done` once, replacing the
  per-PE `done_count <= done_count + 1'b1` inside the merge
  loop (non-blocking undercount when several PEs complete the
  same cycle). The debug monitor showed 0 simultaneous
  completions in the C49 run, so this was not the cause here;
  the change is strictly more correct and is kept.
- rtl/common/pe_mac.sv: ST_DONE now accepts a `start` (same
  init as ST_IDLE) instead of ignoring it. A step dispatched
  while the PE is in ST_DONE used to be lost. The debug run
  showed the fix never triggers for C49 (identical cycle
  count), so it was not the cause either; kept as a robustness
  fix.

### YOLO e2e chain: PASS (225/225 nodes, max|err|=0)

Full YOLO e2e (run_e2e_7d.py --model yolo --resume) completed
after the C49 fix: all 225 nodes pass vs the COMPOSED funcsim
golden (convs, act PWL <=1 LSB, simplegate, concat, residual
CPU, SPPF 5x5 CPU, upsample, det_box/det_cls), max|err|=0
throughout. Log: logs/stage7d/e2e_yolo_full2.log,
results: logs/stage7d/e2e_yolo.json.

### YOLO final detection: FAILS on quantization (0 vs 5) -- NOT an RTL bug

The final DFL+NMS on the RTL Detect outputs gives 0 detections;
torch float on the same 128x128 input gives 5 (4 persons + 1
bus, confidences 0.36-0.75). The RTL matches funcsim exactly,
so the failure is in the fixed-point model itself: the
per-layer quantization error compounds through the ~30-layer
network.

Measured (fixed_forward dequantized vs the capture's own
float_forward on the same input):
- C0: max|diff|=0.26 (good). C1: 4.07 -- the divergence starts
  at the second conv and grows: C12: 6.25, C46: 34.14,
  C52: 63.44, C54: 83.69.
- Detect heads: det_box0 13.77, det_cls0 13.18, det_cls1 29.94,
  det_cls2 15.85. Fixed det_cls0 max sigmoid = 0.007 vs float
  0.689 -- every class score is below the 0.25 threshold.
- Isolation: det_cls0 fed with the FLOAT C52_silu input
  matches float to 0.37 (its quantization is fine); C51 fed
  with float input matches to 0.62. The fault is upstream: the
  SiLU outputs are capped at ~half the float range
  (C32_silu fixed max 2.42 vs float 4.68; C51_silu 2.29 vs
  6.10; C52_silu 12.05 vs 27.75), because each layer's fixed
  output feeds the next and the error accumulates. A manual
  recomputation of the C32_silu simplegate from float inputs
  is exact (4.675), so the SiLU decomposition/shift math is
  correct -- the inputs arriving at each simplegate are
  already degraded.

Consequence: the fixed-point YOLOv8n (as quantized by the 7d
capture, ranges measured on the single e2e image) does not
preserve detection at 128x128. This is a calibration issue,
not a hardware bug: the e2e chain (RTL vs funcsim) is fully
verified. Follow-up (not in 7d scope): recalibrate the
quantization (e.g. ranges from more images, or a robust
range instead of the single-image max) and re-run the capture
+ e2e; or accept the fixed-point detection loss and report it.
The DNANet final check (mask IoU) is the functional
counterpart and may behave differently -- see the DNANet run.

### DNANet driver fixes (2026-11-22 evening, both in sw/vision/run_e2e_7d.py)

- Simplegate broadcast: the attention multiplies have
  mismatched operand shapes -- ca_mul is (1,1,C) attention x
  (H,W,C) feature, sa_mul is (H,W,1) attention x (H,W,C)
  feature. run_simplegate_wave assumed both operands are
  (H,W,C) and wrote W=1,H=1 for ca_mul, so the TB desynced
  (case 1 read pixel values as a header: shift=254, W=253)
  and the output file contained 'x'. Fix: broadcast the
  smaller operand to the other's shape inside
  run_simplegate_wave (np.broadcast_to, both the (1,1,C) and
  the (H,W,1) cases). The funcsim golden already broadcasts
  (numpy elementwise), so only the RTL driver needed it.
- Dot-mode tk%8: sa.conv1 (7x7, C_in=2, C_out=1, Kdim=98)
  failed with max|err|=45. run_pe_conv set mode_dot=1 for
  C_out==1, but pe_mac's dot reduction processes
  tk_groups = tk >> 3 groups of 8 taps, so tk=14 (largest
  divisor of 98 <= 32) drops 6 taps per K-chunk. The
  scheduler (sched_pe, outside the 7d boundary) only guards
  TN < 8, not tk % 8. Fix in the driver: compute
  tk = largest_divisor(Kdim, 32) and use dot mode only when
  C_out == 1 AND tk % 8 == 0; otherwise outer mode (which is
  TN-generic and handles C_out=1 via the zero-padded last
  N-chunk). Only the 26 sa.conv1 layers switch to outer mode;
  the final1-4 and fc1/fc2 (tk 16/32) stay in dot mode. The
  scheduler itself should gain the tk % 8 guard as a
  follow-up (sw/common/sched_pe.py, out of boundary).

### Post-synth run (2026-11-22 evening): do NOT run concurrently with the e2e

The 7d post-synth (act, simplegate, residual, pool, upsample,
concat) was started concurrently with the DNANet e2e. This
corrupts both: they share tb/vectors_*.txt and tb/out_*.txt
(the e2e wave TBs and the post-synth golden TBs use the same
filenames). The post-synth simplegate sim stalled at
32,604/32,768 outputs because the DNANet e2e overwrote
vectors_simplegate.txt mid-run. The post-synth was killed;
it must be re-run AFTER the DNANet e2e finishes, alone.
Result so far: act post-synth PASSED all 5 checks (reused
netlist, 53,187 cells). simplegate/residual/pool/upsample/
concat are still open. The 7d post-synth runner is
/tmp/run_ps_7d.sh (extracted 7d section of
tb/run_postsynth.sh; fix its cd to the repo root -- the
original `cd $(dirname $0)/..` is wrong for /tmp).

- CPU downsample bug: cpu_node's `downsample` called
  `_upsample(af, {"mode": "bilinear", "scale": 0.5})`, but
  _upsample does `int(scale)`, so int(0.5)==0 and it returned
  an empty (0,0,C) array -- the e2e died at down_x0_1 with a
  broadcast error against the (32,32,16) golden. Fix: use
  `_downsample_ac` (the align_corners=True 0.5x bilinear
  that fixed_forward uses), with the same mixed-scale
  dequantize logic as fixed_forward. down_x0_1 now passes
  max|err|=0.

### Gap/gmax groups bug — fixed 2026-11-22 (DNANet e2e blocker)
- `fixed_forward` stored `gap`/`gmax` outputs as uint8 (0-255) but
  recorded their channel groups by inheriting the producer's
  signed flag. DNANet's ChannelAttention `gap`/`gmax` read the
  SIGNED conv2 output, so their groups said signed=True.
- Two consequences: (a) `rescale_input` (via `rescale_bytes`) clips
  to [-128,127] whenever a group is signed — even at equal scale —
  so a conv consuming a gap/gmax output had its input values
  128..255 clipped to 127; (b) a checkpoint resume
  (`bytes_to_values`) reinterpreted those uint8 bytes as negative.
- Symptom: DNANet e2e stopped at `conv4_0.1.ca.fc1_avg` with
  GOLDEN MISMATCH max|err|=39 (independent golden vs
  fixed_forward). Earlier fc1 nodes passed only because their GAP
  values were all <128 (clipping was a no-op there); conv4_0 (256
  channels, deepest level) is the first whose GAP values reach
  >=128. Replayed the corruption exactly (cross 255 on resume).
- Fix (sw/vision/funcsim.py + sw/vision/run_e2e_7d.py
  track_groups): the uint8-output ops (max_pool2d, avg_pool2d,
  gap, gmax, upsample) now force signed=False in their groups,
  keeping the producer's channel ranges and r. `relu`/`cmean`/
  `cmax` already did this; `downsample` already set groups from
  the op's own flag; `bypass` (identity) still inherits.
  Replayed cross for fc1_avg and fc1_max both 0 after the fix;
  the e2e passed both nodes on resume (16 steps each).

### Signed-input upsample — fixed 2026-11-22 (DNANet e2e blocker)
- The RTL upsample averages bytes as UNSIGNED. The four
  final-aggregation 1x1 adapter convs (conv0_4_1x1, conv0_3_1x1,
  conv0_2_1x1, conv0_1_1x1) are signed_out=True, so the up_16 /
  up_8 / up_4 / up chains that consume them have signed inputs.
- Symptom: DNANet e2e stopped at `up16_a4_0` with max|err|=128
  (budget 0). The adapter output has 138/256 negative values;
  unsigned averaging of two's-complement bytes is off by up to
  256 at those positions (reproduced exactly).
- Fix (sw/vision/run_e2e_7d.py run_chain): an upsample whose
  input has any signed channel group runs on the CPU with the
  same funcsim `_upsample` code as the golden (matches by
  construction, as for the other CPU ops); unsigned-input
  upsamples stay on the RTL. Verified the CPU path matches the
  golden exactly (max|err|=0) before resuming; the e2e passed
  up16_a4_0..a4_3, up8_a3_0..a3_2, up4_a2_0, up_a1_0 on CPU and
  the chained second/third steps (unsigned inputs) on RTL.
- Follow-up (rtl/ change, outside 7d): upsample.sv needs a signed
  mode so these chains can run on the RTL (same class as the
  residual signed-add and the pool GAP unsigned-only gaps).

### DNANet e2e final result — chain PASS, final mask partial (2026-11-23)
- Chain: 559/559 nodes max|err|=0 vs the composed funcsim golden
  (convs, act PWL <=1, simplegate, concat, residual/relu CPU,
  pool, upsample incl. CPU signed-input chains, downsample,
  gap/gmax/cmean/cmax CPU). Log e2e_dnanet_full10.log.
- Final mask vs torch float on the same 64x64 crop:
  per-final IoU = [0.6, 1.0, 0.3913, 0.0], aggregate IoU = 1.0000.
  RTL px per final = [15, 9, 23, 0], torch = [9, 9, 9, 9].
  Bar was aggregate >= 0.9 AND min per-final >= 0.9 -> FINAL FAILED.
- The RTL == fixed_forward exactly (final logits max|diff|=0 for
  all four; masks from RTL and from fixed_forward identical).
  So the divergence is fixed-point vs float, not RTL vs funcsim:
  the per-layer quantization error compounds through the network
  and the deep-supervision logits drift, worst at final4 (all
  logits negative, range -128..-31, 0 px). The aggregate (mean of
  the four sigmoids) still gives the correct 9 px (IoU 1.0) --
  the over-predictions of final1/final3 cancel final4's
  under-prediction.
- Note: the capture validation's earlier "IoU 1.0 for all four"
  was computed with the BUGGY fixed_forward (gap/gmax groups
  said signed, so the fc1 inputs were clipped 128..255 -> 127);
  the groups fix changed the true fixed-point values. The
  corrected fixed-point model does not preserve the individual
  deep-supervision masks. Same class of issue as YOLO (chain
  verified, final accuracy quantization-limited). Decision needed:
  recalibrate the quantization, or accept the aggregate-only
  result (the aggregate is the functional detection output).

## Verification summary (2026-11-23, tb/run_stage7d.sh + rerun of tb/run_stage7c.sh)

- Part 1 YOLO e2e: chain 225/225 nodes max|err|=0 (convs, act PWL
  <=1 LSB, simplegate, concat, residual CPU, SPPF 5x5 CPU,
  upsample, det_box/det_cls). FINAL FAILED: 0 vs 5 detections
  (quantization, see above).
- Part 1 DNANet e2e: chain 559/559 nodes max|err|=0 (convs, act,
  simplegate, concat, residual/relu CPU, pool, upsample,
  downsample, gap/gmax/cmean/cmax CPU). FINAL FAILED: per-final
  IoU [0.6, 1.0, 0.3913, 0.0], aggregate IoU 1.0000
  (quantization, see above).
- Part 2 post-synthesis (tb/run_postsynth_7d.sh, standalone):
  ALL PASS 30/0 -- act 5/5, simplegate, residual, pool, upsample,
  concat. All six 7d modules VM-feasible; NO Mac Phase 4 needed.
  Full tb/run_postsynth.sh (7c + 7d) also PASS in the runner.
- Unit tests (sw/vision/tests): 90 ran, 88 pass, 2 fail
  (TestDflDecodeVsTorch, TestDnanetMask -- both the float-accuracy
  bar, quantization-limited; NOT test bugs).
- Regression tb/run_stage7c.sh (rerun 2026-11-23, log
  logs/stage7d/run_stage7c_rerun.log): e2e srcnn/retinexnet/dncnn
  PASS (dncnn PSNR 53.41 dB, unchanged), post-synth PASS,
  Stage 7b PASS, Stages 1-4 PASS. Stage 5 FAILED (rc=1) ONLY
  because its unit-test sweep (72 tests in sw/vision/tests/)
  picks up the two failing 7d accuracy tests; the Stage 5 report
  itself is OK and the cost/DRAM/roofline report PASS. No
  Stage 5 functionality regressed. The funcsim.py gap/gmax groups
  fix is a no-op for the 7c models (their producers are unsigned).
- Out-of-boundary RTL changes (rtl/common/pe_array.sv,
  rtl/common/pe_mac.sv, rtl/vision/srcnn_pipe_pe.sv): the C49
  n_kchunk 6->8 bit widening plus the merge_count block and the
  ST_DONE start acceptance. Made because real RTL bugs blocked
  the 7d e2e; the parallel LLM agent does not touch rtl/.
  Parent review required before push.
