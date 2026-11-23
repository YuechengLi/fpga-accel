# Stage 7d doc updates (for files outside the 7d boundary)

These are the doc changes the 7d work implies, recorded here because
the target files (models/README.md, docs/plan.md, root README.md,
rtl/README.md, sw/README.md) are outside this stage's file boundary
(another agent is registering the LLM models in parallel). Apply
them to the real files when the boundary lifts.

## 1. YOLOv8n has 63 conv layers, not 57 (models/README.md)

The registration's 57-conv count counts UConv modules only. Each
Detect cv2/cv3 Sequential is THREE convs: (0) Conv 3x3 SiLU, (1)
Conv 3x3 SiLU, (2) a plain 1x1 `nn.Conv2d` with NO BN and NO
activation -- that plain 1x1 is the Detect head's actual output.
The 6 plain 1x1 weights live in
models/vision/yolov8n/detect_out.npz (+ detect_out.json), added by
the 7d capture; the registered yolov8n.npz / yolov8n_quant.json
still lack them. Whether to fold them into the registration is
undecided. The 7d e2e (226 ops, 71 convs) includes them as
det_box{0,1,2} / det_cls{0,1,2} and the DFL decode is validated
against the torch Detect output (same 5 detections).

## 2. Residual module needs a signed-add + rescale mode (rtl/README.md)

The 7d driver runs ALL residual adds on the CPU because
residual.sv mode 0 is an *unsigned* add of the raw bytes (no
signed-add mode, no input rescaling), while 49/52 DNANet
residuals have a signed input and the two inputs almost always
carry different scales. Only 2 of the 58 residuals could use
mode 0. Follow-up (needs an rtl/ change, outside 7d): a mode 4
that does a signed add with per-input rescale, so the
DNANet/YOLO residual adds can run on the RTL instead of the CPU.
The residual module itself is verified in Part 2 (post-synth,
Stage 4 golden).

## 3. GAP runs on the CPU in the 7d driver (sw/README.md)

pool.sv mode 2 (GAP) sums the input bytes as *unsigned*
(zero-extended) and floor-shifts; the DNANet ChannelAttention
GAP input (conv2 output) is signed, so the RTL GAP would sum
two's-complement bytes as large positives. The rounding also
differs (RTL floor vs funcsim round(mean), up to 1 LSB). The 7d
driver runs the GAP on the CPU with the same funcsim code as the
golden. The pool.sv GAP is still verified in Part 2 (Stage 4
golden, unsigned inputs).

## 4. No frontend blur in the 7d e2e (docs/plan.md, sw/README.md)

The frontend is the 7c enhancement models' preprocessing; YOLO /
DNANet were trained on raw images (a blur would only degrade the
detection / segmentation). The 7d capture calibrates on the raw
input and the driver feeds the same raw input to the PE chain.

## 5. Upsample module needs a signed mode (rtl/README.md)

upsample.sv averages bytes as *unsigned*. The four DNANet
final-aggregation 1x1 adapter convs are signed_out=True, so the
up_16 / up_8 / up_4 / up chains that consume them have signed
inputs; the RTL average of two's-complement bytes is wrong
(up16_a4_0 failed with max|err|=128; the adapter output has
138/256 negative values). The 7d driver routes signed-input
upsamples to the CPU with the same funcsim code as the golden.
Follow-up (needs an rtl/ change, outside 7d): a signed mode for
upsample.sv (bilinear averaging on signed bytes), so these
chains can run on the RTL. The upsample module itself is
verified in Part 2 (Stage 4 golden, unsigned inputs).

## 6. Gap/gmax groups must be unsigned (sw/README.md, funcsim)

fixed_forward stored `gap`/`gmax` (and max/avg pool, upsample)
outputs as uint8 but recorded their channel groups by inheriting
the producer's signed flag. DNANet's gap/gmax read the SIGNED
conv2 output, so their groups said signed=True; a consumer's
rescale then clipped values 128..255 to 127, and a checkpoint
resume reinterpreted those bytes as negative (DNANet e2e
stopped at conv4_0.1.ca.fc1_avg, GOLDEN MISMATCH 39). Fixed in
7d (funcsim.py + run_e2e_7d.py track_groups): the uint8-output
ops now force signed=False in their groups. The scheduler
(sw/common/sched_pe.py) still lacks the tk % 8 == 0 guard for
dot mode (sa.conv1 finding); the 7d driver works around it by
forcing outer mode when tk % 8 != 0.

## 7. Fixed-point accuracy vs float is quantization-limited (docs/plan.md)

YOLOv8n e2e chain verified (225/225 nodes max|err|=0) but the
final detection gives 0 vs 5 torch detections (fixed det_cls0
max sigmoid 0.007 vs float 0.689; per-layer error compounds
from C1's 4.07 to 13-83 at the Detect heads). DNANet e2e chain
verified (559/559 nodes max|err|=0) but the final mask gives
per-final IoU [0.6, 1.0, 0.39, 0.0] vs torch (aggregate IoU
1.0). Both are calibration issues, not RTL bugs. Follow-up:
recalibrate the quantization (ranges from more images, or a
robust range instead of the single-image max) and re-run the
capture + e2e; or accept the fixed-point detection loss and
report it. The two heavy unit tests (TestDflDecodeVsTorch,
TestDnanetMask) encode the float-accuracy bar and currently
fail on this; the other 88 unit tests pass.

## 8. Stage 5 regression FAIL is the two 7d accuracy tests (tb/run_stage5.sh)

The Stage 5 runner sweeps the whole sw/vision/tests/ directory
(72 tests), which now includes test_e2e_7d.py. The two failing
7d accuracy tests (TestDflDecodeVsTorch, TestDnanetMask) make
Stage 5 report FAILED even though the Stage 5 report itself is
OK and the cost/DRAM/roofline report PASS. No Stage 5
functionality regressed. If the quantization is recalibrated and
the two tests pass, Stage 5 passes again. Alternatively, move
the accuracy-bar tests to a separate directory so the Stage 5
sweep does not pick them up -- a parent decision, not a 7d fix.
