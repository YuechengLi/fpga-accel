# Stage 6 prep — pretrained weights for numerical verification

Pretrained weights for the seven models used in model numerical
verification (plan `docs/plan.md` §4, Stage 6 and later): SRCNN, DnCNN,
RetinexNet Decom-Net (Stage 6), plus YOLOv8n and DNANet (onboarded
2026-11-21 for the detection / IRSTD target applications), plus
Qwen3-VL-4B-Instruct and Qwen2.5-Omni-3B (fetched 2026-11-21 for the
VCU128 LLM decode track — raw weights gitignored, see `models/llm/README.md`). Each model has
a float reference and a fixed-point (quantized) version in Python (numpy);
Stage 6 runs the same quantized weights in RTL and compares pixel-by-pixel
against this golden.

The five vision models total ~111 MB under `models/vision/` (YOLOv8n ~37 MB,
DNANet ~64 MB, raw checkpoints included). The vision raw downloads are
committed as-is; the largest single file is `vision/dnanet/dnanet.npz` (28 MB),
under GitHub's 100 MB limit. The two LLM models live under
`models/llm/` — raw weights 19.44 GiB, gitignored, not committed
(see `models/llm/README.md` for the manifest and SHA256 hashes).

## Quantization scheme (uniform across all three models)

Implemented in `common.py` (`quantize_layer`, `conv2d_float`,
`conv2d_fixed`). The RTL datapath is: 16-bit signed coefficients, 8-bit
pixels/activations, 32-bit accumulator, one runtime right shift per layer.

- **Pixels / unsigned activations**: uint8, value = float * (255 / 2**r),
  where `r` is the layer's activation range exponent (output covers
  [0, 2**r]). `r = 0` if the layer's max float activation is <= 1, else
  `ceil(log2(max_act))`. `r` is measured on the test image used below;
  Stage 6 must re-validate the range on wider data (if an activation
  exceeds 2**r it saturates at 255).
- **Coefficients**: int16, per-layer power-of-two scale.
  `e = floor(log2(32767 / max|W|))` clamped to [0,15];
  `W_q = clip(round(W * 2**e), -32768, 32767)`; weight scale = 2**(-e).
- **Bias**: `b_q = round(b * (255 / 2**r_in) * 2**e)`, int32, added to the
  accumulator.
- **RTL runtime shift for the layer = e + r_out - r_in.** It must stay in
  [0,15] (the RTL shift field is 4 bits). When `e + r_out - r_in` would
  exceed 15, `e` is reduced (slightly lower weight precision) so the shift
  stays legal. This happened for SRCNN L1 (e 15 -> 13, shift 15).
- **Accumulation** is int64 in this golden (no wrap). The RTL tiles input
  channels so the per-tile 32-bit accumulator stays in range; see the
  per-model notes for the worst-case bound.
- **After the shift, saturate to [0,255].** ReLU is free: the [0,255]
  saturation already clips negatives to 0. Hidden layers therefore need no
  separate ReLU in the RTL.
- **Signed intermediates** (a layer with NO activation, e.g. RetinexNet
  conv0, DnCNN's final residual, RetinexNet's recon logit): stored as
  SIGNED int8 / int32 after the shift (no [0,255] clip). Zero padding
  inserts 0, which correctly represents 0.0 for a signed activation.
  **Do NOT use a uint8+offset representation for a signed intermediate:**
  it is incompatible with zero padding (zero-padded pixels carry no offset
  while the bias correction assumes all pixels do). Verified on RetinexNet:
  the offset scheme gave conv1 max error 215 and end-to-end R PSNR 14.65 dB
  under zero padding, while signed int8 gives R PSNR 28.95 dB (equal to the
  pure padding-mode difference, i.e. no extra quantization error). Signed
  int8 therefore requires the RTL conv datapath to accept signed
  intermediate activations -- a Stage 3 design input, not just a golden
  detail.

Per-layer scale/shift are recorded in `<model>_quant.json` alongside the
converted `.npz` (keys: `shift`, `weight_exp_e`, `r_in`, `r_out`, `scale`,
`max_abs_W`, `max_act`, `W_q_range`, `b_q_range`).

Converted `.npz` layout (per layer `L`): `L_W` (float32),
`L_W_q` (int16), `L_b` (float32), `L_b_q` (int32). Shapes are
`(H, W, C_in, C_out)` for weights, `(C_out,)` for biases.

Test images are deterministic synthetic images from `common.py`
(`make_test_gray` / `make_test_rgb`, seed 7: smooth gradient + bright disk
+ dark bar + fine texture). They are small (32x32 / 33x33) so the golden
runs in seconds. Stage 6 should also run a natural-image crop; the
quantization numbers here are for these test images.

## 1. SRCNN — Dong et al. 2014, 9-1-5, x3 upscaling

- **Source URL**: the original author's x3 weights, recovered verbatim from
  the FFmpeg GSoC patch email that embeds `srcnn_weights.h`:
  https://www.mail-archive.com/ffmpeg-devel@ffmpeg.org/msg67144.html
  (raw HTML saved as `srcnn/raw/msg67144.html`).
  Why this source: the CUHK official page
  (mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) has no download links; the
  FFmpeg filter was a GSoC patch, never merged into FFmpeg master (the file
  is 404 in every n4.0-n4.3 tag); the widely used yjn870/SRCNN-pytorch
  checkpoint is 9-5-5, not the paper's 9-1-5, and is the repo author's own
  retraining, so it was NOT used.
- **Files**: `srcnn/raw/msg67144.html` (source);
  `srcnn/srcnn_x3_float.npz` (extracted float32);
  `srcnn/srcnn_x3.npz` (float32 + int16 quantized);
  `srcnn/srcnn_x3_quant.json` (per-layer scale/shift);
  `srcnn/extract_srcnn.py`, `srcnn/run_srcnn.py`.
- **Layers** (valid conv, no padding; Y channel only):
  L1 9x9, 1->64, W range [-0.917,+0.908], b 64;
  L2 1x1, 64->32, W range [-0.921,+0.953], b 32;
  L3 5x5, 32->1, W range [-0.072,+0.118], b 1 (+0.0504).
  Verified counts: W1 5184, W2 2048, W3 800, b1 64, b2 32, b3 1.
- **Forward**: HR Y 33x33 -> bicubic down x3 -> LR 11x11 -> bicubic up x3
  -> input 33x33 -> L1 9x9+ReLU -> L2 1x1+ReLU -> L3 5x5 -> Y out 21x21.
  (PIL BICUBIC for the resize; the conv part is what the RTL verifies.)
- **Scale/shift**: L1 shift 15 (e 13, r 0->2), L2 shift 15 (e 15, r 2->2),
  L3 shift 13 (e 15, r 2->0). All shifts in [0,15].
- **Float vs fixed-point** (test image above; float*255 scale):
  L1 max|err| 1.03, PSNR 56.56 dB; L2 max|err| 3.36, PSNR 46.95 dB;
  L3 / end-to-end max|err| 5.79, PSNR 42.76 dB.
  Float reference vs HR center crop: PSNR 22.87 dB (bicubic input 22.95 dB
  -- this tiny synthetic image is not a good SR benchmark; the number that
  matters for Stage 6 is fixed-vs-float, 42.76 dB). Output finite, range
  [34,242].
- **Notes**: intermediate activations exceed 1.0 (L1 max 2.08, L2 max 2.62),
  which is why `r_out = 2` for L1/L2 (output covers [0,4]). A naive
  uint8 = float*255 scheme would clip them and destroy the result.

## 2. DnCNN — Zhang et al. 2017, Gaussian denoising, sigma=25

- **Source URL**: official cszn/DnCNN release, file
  `model/specifics/sigma=25.mat`, downloaded from
  https://raw.githubusercontent.com/cszn/DnCNN/master/model/specifics/sigma=25.mat
  (2,295,807 bytes, MATLAB v7.3 / HDF5; read with h5py -- scipy cannot
  read v7.3). Saved as `dncnn/raw/sigma25.mat`.
- **Files**: `dncnn/raw/sigma25.mat` (source);
  `dncnn/dncnn_sigma25_float.npz` (extracted float32);
  `dncnn/dncnn_sigma25.npz` (float32 + int16 quantized, all 17 layers);
  `dncnn/dncnn_sigma25_quant.json`;
  `dncnn/extract_dncnn.py`, `dncnn/run_dncnn.py`.
- **Layers** (17 conv, all 3x3, zero pad 1 = same size; residual learning):
  L1 3x3, 1->64; L2-L16 3x3, 64->64 (x15); L17 3x3, 64->1.
  **The released .mat contains NO BatchNorm layers** (verified by parsing
  all 33 layer types: 17 conv + 16 relu). This differs from the paper's
  description (which uses BN); the released sigma=25 model is conv+ReLU
  only, so the golden is conv+ReLU and there is no BN folding to do.
  MATLAB HDF5 dims are reversed (h5py shape is (Cout,Cin,H,W)); the
  extractor transposes to (H,W,Cin,Cout). Last layer is 3-D (64,3,3) ->
  (3,3,64,1).
- **Forward**: noisy = clean + Gaussian noise sigma=25 (0..255 scale),
  clipped [0,1]; L1-L16 conv 3x3+ReLU; L17 conv 3x3 (no activation) ->
  residual (signed); denoised = clip(noisy - residual, 0, 255).
  Stage 6 uses a fragment (first few layers); the full 17-layer weights
  are provided so any fragment can be taken.
- **Scale/shift**: all shifts in [0,15] (mostly 15; L11 and L15 are 14).
  `r_out` per layer: [0,1,1,1,1,1,1,2,2,2,1,1,1,1,0,0,0].
  L17 output is SIGNED (residual, r_out 0, no [0,255] clip).
- **Float vs fixed-point** (32x32 test image, sigma=25 noise):
  L1 max|err| 1.45, PSNR 55.53 dB; L2 max|err| 3.90, PSNR 53.60 dB;
  L16 max|err| 3.69, PSNR 52.61 dB;
  L17 (signed residual) max|err| 3.08, PSNR 48.22 dB;
  end-to-end denoised max|err| 3.53, PSNR (fixed vs float) 48.23 dB.
  Denoised PSNR vs clean: fixed 23.15 dB, float 23.20 dB (the absolute
  number is low because the 32x32 synthetic image is mostly high-frequency
  texture; the fixed-vs-float gap is what Stage 6 checks).
- **Notes**: 32-bit accumulator headroom for the RTL: worst case per output
  pixel is 3*3*64 = 576 products; with int16 weights the per-tile
  accumulator must be checked per tile size in Stage 3 (the golden uses
  int64 and does not wrap).

## 3. RetinexNet Decom-Net — Chen et al. 2018, low-light decomposition

- **Source URL**: PyTorch port aasharma90/RetinexNet_PyTorch, checkpoint
  `ckpts/Decom/9200.tar`, downloaded from
  https://raw.githubusercontent.com/aasharma90/RetinexNet_PyTorch/master/ckpts/Decom/9200.tar
  (833,535 bytes). Saved as `retinexnet/raw/decom_9200.tar`.
  **Fallback (stated clearly)**: the official TF implementation
  weichen582/RetinexNet has NO checkpoint download link in its README;
  the checkpoint lives on Google Drive / Baidu, which is not directly
  downloadable on this VM. The PyTorch port is the reputable fallback.
  It is a raw pickle (protocol 2, old torch 1.1.0 format, NOT a zip),
  loaded with torch 2.15.0+cpu. The port's `model.py` was read in full
  and the forward below matches it. Only Decom-Net is used (not
  Relight-Net).
- **Files**: `retinexnet/raw/decom_9200.tar` (source);
  `retinexnet/retinexnet_decom_float.npz` (extracted float32);
  `retinexnet/retinexnet_decom.npz` (float32 + int16 quantized);
  `retinexnet/retinexnet_decom_quant.json`;
  `retinexnet/extract_retinexnet.py`, `retinexnet/run_retinexnet.py`.
- **Layers** (PyTorch (Cout,Cin,H,W) transposed to (H,W,Cin,Cout)):
  conv0 9x9, 4->64 (no activation); conv1-conv5 3x3, 64->64 (x5, ReLU);
  recon 3x3, 64->4 (no activation) -> sigmoid -> R (3ch), L (1ch).
- **Forward**: RGB -> max channel -> concat [max, R, G, B] (4ch) ->
  conv0 9x9 -> 5x (conv 3x3 + ReLU) -> recon 3x3 -> sigmoid.
  **Padding is REPLICATE (edge replication), NOT zero-fill.** This differs
  from the accelerator's zero-fill padding; the border effect is
  quantified below and must be handled in Stage 6.
- **Scale/shift**: conv0-conv5 shift 15, recon shift 13. All in [0,15].
  `r_out`: [0,0,0,0,1,2,0]. conv0 output is SIGNED int8 (see below);
  recon output is SIGNED (logit, r_out 0, no clip), then float sigmoid.
- **Float vs fixed-point** (32x32 RGB test image, scaled x0.25 to low
  light; replicate padding, i.e. the original model's padding):
  conv0 (signed int8) max|err| 1.01, PSNR 52.90 dB;
  conv1 max|err| 2.34, PSNR 56.28 dB; conv5 max|err| 9.47, PSNR 45.66 dB;
  recon (signed logit) max|err| 35.06, PSNR 26.83 dB;
  end-to-end R max|err| 8.09, PSNR 39.71 dB;
  end-to-end L max|err| 2.24, PSNR 45.11 dB.
  Float sanity: R range [0.14,0.94] mean 0.69, L range [0.04,0.32] mean
  0.17; R*L reconstructs the low-light input at PSNR 46.06 dB.
- **Signed intermediate (design finding)**: conv0 has NO activation and its
  output is signed (~50% negative, range [-0.33,+0.31]). Inserting a ReLU
  after conv0 destroys the signal (float-with-ReLU vs float-without:
  R PSNR 18.44 dB). The correct representation is SIGNED int8
  (`s0 = clip(round(x*255), -128, 127)`), which the golden uses. This
  requires the RTL conv datapath to accept signed intermediate activations.
  A uint8+offset alternative was tried and REJECTED: it works under
  replicate padding but fails under zero padding (see below).
- **Border effect — zero padding vs replicate padding** (fixed-point,
  same weights): R max|diff| 65.98, PSNR 28.95 dB; L max|diff| 21.56,
  PSNR 41.98 dB. Interior (5px border excluded): R max|diff| 3.79,
  L max|diff| 1.50. The float-only padding difference is R PSNR 29.34 dB,
  so the fixed-point border gap is essentially the pure padding-mode
  difference, not extra quantization error. **Stage 6 implication:** the
  RTL uses zero padding, the reference uses replicate. Compare the RTL
  against a ZERO-pad float reference (apples-to-apples quantization check),
  and report the replicate-vs-zero border difference separately -- or
  exclude a ~5px border. Do not expect the border pixels to match the
  replicate-pad reference.
- **Notes**: recon logit range is [-3.08,+2.79] (float); in the 255 scale
  the signed recon output spans [-771,+710]. The sigmoid after recon is a
  PWL unit in the accelerator (Stage 3); this golden applies the float
  sigmoid, so the recon-logit error (PSNR 26.83 dB) is the number to watch
  before the sigmoid, and R/L PSNR (39.71 / 45.11 dB) after it.

## 4. YOLOv8n — Ultralytics YOLOv8 nano, object detection (COCO)

Full notes in `yolov8n/README.md`. Official ultralytics v8.3.0 checkpoint
(6,549,796 bytes, SHA256 f59b3d83…); 57 Conv modules, all Conv2d + BN +
SiLU (39x 3x3, 18x 1x1). Float reference on bus.jpg: 5 detections after
CPU decode + NMS (4 persons + 1 bus). BN folding verified exact (1e-6).
Per-layer fixed-vs-float on representative convs: 27–55 dB — worse than
the image models, because activation ranges reach r_out=7 and BN folding
amplifies some weights ~35x. Shifts 12–15, all legal. DFL decode + NMS
stay on CPU; the 57 convs are accelerator candidates. No mAP claimed
(COCO eval not done); calibration on wider data still open.

## 5. DNANet — Dense Nested Attention Network, infrared small target detection

Full notes in `dnanet/README.md`. DCANet Model Zoo NUAA-SIRST checkpoint
(19,033,849 bytes, SHA256 fcc9b52c…; epoch 1499, recorded train mean IoU
0.8489); official model definition vendored (`model_DNANet.py`, MIT).
154 Conv2d (52x 3x3, 76x 1x1, 26x 7x7 spatial attention), 68 BN.
Float reference on the repo demo image target1.png: 9 predicted target
pixels, IoU 0.556 vs the repo's own Pred image (different checkpoint, so
an exact match is not expected; the target location matches 5/5).
ImageNet input normalization folded into the first conv (exact in the
interior, 1px border rim differs — same class of border effect as
RetinexNet). BN folding verified exact (2.5e-06). Per-layer fixed-vs-float:
55–57 dB on the main convs; attention/final logits 0.6–0.85 logit units
max error. Shifts 7–15, all legal. No NUAA-SIRST IoU claimed (no GT mask
for the demo image); calibration on real SIRST data still open.

## How to use (Stage 6)

- Re-run a model: `cd models/vision/<name> && python3 run_<name>.py`. It prints
  per-layer and end-to-end float-vs-fixed error and rewrites the converted
  `.npz` + `*_quant.json`. Requires numpy, scipy (DnCNN .mat), torch
  (RetinexNet .tar), PIL (SRCNN resize). YOLOv8n additionally needs the
  `ultralytics` package; DNANet needs torch + PIL (the model definition
  is vendored in `vision/dnanet/model_DNANet.py`).
- Re-extract from the raw download: `python3 extract_<name>.py` (writes the
  `*_float.npz`).
- RTL config per layer: load `L_W_q` (int16) into the coefficient memory
  (config bus), `L_b_q` (int32) as the bias, and set the runtime shift to
  `shift` from `*_quant.json`. The activation range exponent `r_out` tells
  the next layer's `r_in`; the first layer's `r_in` is 0 (input pixels are
  uint8 = float*255).
- The golden (`common.conv2d_fixed`) is the reference the RTL must match:
  same weights, same shift, same padding mode. For RetinexNet, use zero
  padding in BOTH the RTL and the golden when checking quantization, and
  handle the border as described above.

## Files

| path | contents |
|---|---|
| `common.py` | quantization + numpy conv golden, test-image generators |
| `srcnn/raw/msg67144.html` | SRCNN x3 weights source (FFmpeg patch email) |
| `srcnn/srcnn_x3_float.npz` | SRCNN extracted float32 weights |
| `srcnn/srcnn_x3.npz` | SRCNN float32 + int16 quantized + bias |
| `srcnn/srcnn_x3_quant.json` | SRCNN per-layer scale/shift |
| `srcnn/extract_srcnn.py`, `srcnn/run_srcnn.py` | SRCNN extract / verify scripts |
| `dncnn/raw/sigma25.mat` | DnCNN sigma=25 official weights (2.3 MB) |
| `dncnn/dncnn_sigma25_float.npz` | DnCNN extracted float32 weights |
| `dncnn/dncnn_sigma25.npz` | DnCNN float32 + int16 quantized (17 layers) |
| `dncnn/dncnn_sigma25_quant.json` | DnCNN per-layer scale/shift |
| `dncnn/extract_dncnn.py`, `dncnn/run_dncnn.py` | DnCNN extract / verify scripts |
| `retinexnet/raw/decom_9200.tar` | RetinexNet Decom checkpoint (PyTorch port) |
| `retinexnet/retinexnet_decom_float.npz` | RetinexNet extracted float32 weights |
| `retinexnet/retinexnet_decom.npz` | RetinexNet float32 + int16 quantized |
| `retinexnet/retinexnet_decom_quant.json` | RetinexNet per-layer scale/shift |
| `retinexnet/extract_retinexnet.py`, `retinexnet/run_retinexnet.py` | RetinexNet extract / verify scripts |
| `yolov8n/raw/yolov8n.pt` | YOLOv8n official checkpoint (6.5 MB) |
| `yolov8n/yolov8n_float.npz` | YOLOv8n extracted float32 weights + BN params |
| `yolov8n/yolov8n.npz` | YOLOv8n BN-folded float32 + int16 quantized (57 convs) |
| `yolov8n/yolov8n_quant.json` | YOLOv8n per-layer scale/shift |
| `yolov8n/bus.jpg` | YOLOv8n test image (standard ultralytics demo) |
| `yolov8n/extract_yolov8n.py`, `yolov8n/run_yolov8n.py` | YOLOv8n extract / verify scripts |
| `yolov8n/README.md` | YOLOv8n notes (detections, BN fold, calibration caveats) |
| `dnanet/raw/dnanet_nuaa-sirst.pth.tar` | DNANet NUAA-SIRST checkpoint (19 MB) |
| `dnanet/dnanet_float.npz` | DNANet extracted float32 weights + BN params |
| `dnanet/dnanet.npz` | DNANet BN-folded float32 + int16 quantized (154 convs) |
| `dnanet/dnanet_quant.json` | DNANet per-layer scale/shift |
| `dnanet/target1.png`, `dnanet/target1_Pred.png` | DNANet demo image + repo's prediction |
| `dnanet/model_DNANet.py`, `dnanet/LICENSE.upstream` | Vendored official model definition (MIT) |
| `dnanet/extract_dnanet.py`, `dnanet/run_dnanet.py` | DNANet extract / verify scripts |
| `dnanet/README.md` | DNANet notes (mask check, norm fold, calibration caveats) |
