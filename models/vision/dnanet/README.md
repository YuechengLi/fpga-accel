# 5. DNANet — Dense Nested Attention Network, infrared small target detection

- **Source URL**: DCANet Model Zoo (github.com/Tianzishu/DCANet), DNANet row,
  NUAA-SIRST column, Google Drive file
  `12XKXFEu8JUv0hMCP1hoRXmYTuJp_k1LP` (19,033,849 bytes, SHA256
  `fcc9b52c1701ed3623ce0b88e4adc7197c77b44dccdeb6be4348c512b7da9f35`).
  Saved as `dnanet/raw/dnanet_nuaa-sirst.pth.tar`.
  Model definition: official repo YeRen123455/Infrared-Small-Target-Detection,
  `model/model_DNANet.py` (vendored here as `model_DNANet.py`, MIT —
  see `LICENSE.upstream`), instantiated with the repo test defaults:
  Res_CBAM_block, nb_filter [16,32,64,128,256] (channel_size 'three'),
  num_blocks [2,2,2,2] (backbone 'resnet_18'), deep_supervision=True,
  input_channels=3. Paper: Li et al., IEEE TIP 2023 (arXiv:2106.00487).
- **Checkpoint**: epoch 1499, recorded train loss 0.1739, mean IoU 0.8489
  (training log, NUAA-SIRST). The state dict (570 entries) loads strictly
  into the official model (no missing/unexpected). The checkpoint is a
  torch 1.x save, so `torch.load` uses weights_only=True with the legacy
  numpy globals allowlisted (`numpy.core.multiarray.scalar`, `numpy.dtype`
  and the numpy 2.x dtype classes); no full-pickle load is used.
- **Files**: `dnanet/raw/dnanet_nuaa-sirst.pth.tar` (source);
  `dnanet/dnanet_float.npz` (extracted float32, BN params kept separate);
  `dnanet/dnanet.npz` (BN-folded float32 + int16 quantized);
  `dnanet/dnanet_quant.json` (per-layer scale/shift);
  `dnanet/extract_dnanet.py`, `dnanet/run_dnanet.py`,
  `dnanet/model_DNANet.py` (vendored definition), `dnanet/LICENSE.upstream`;
  `dnanet/target1.png` (real IR demo image from the official repo,
  256x256) + `dnanet/target1_Pred.png` (the repo's own prediction for it).
- **Model**: 154 Conv2d: 52x 3x3, 76x 1x1, 26x 7x7 (spatial attention);
  68 BatchNorm2d; conv params 4,683,476. Dense nested U-Net with CBAM
  attention (channel attention: 1x1 convs + sigmoid; spatial attention:
  7x7 conv + sigmoid), bilinear upsample, maxpool, concat, elementwise
  multiply, residual adds. Deep supervision: 4 logits (final1..4);
  output4 is the mask. All convs are stride 1.
- **Forward** (float reference, `run_dnanet.py`): target1.png ->
  ImageNet-normalized (mean [.485,.456,.406], std [.229,.224,.225], like
  test.py) -> 4 deep-supervision outputs, all (1,1,256,256). Final logit
  range [-107.58, +49.36], prob mean 0.0001; 9 predicted target pixels
  (>0.5) in a 3x3 blob at x[159,161], y[235,237]. IoU vs the repo's own
  target1_Pred.png = 0.556 (inter 5, union 9, gt 5 px): the target
  location matches (5/5 ground-truth pixels covered, 4 extra neighbours).
  The repo Pred image comes from the repo's own bundled pretrain tar, not
  this DCANet-zoo checkpoint, so an exact match is not expected.
- **Normalization + BN folding** (both in float, before quantizing):
  the ImageNet input normalization is folded into the FIRST conv (exact
  affine fold: W/std per input channel, bias corrected by
  sum(W*mean/std)); BN is folded into every conv that has one
  (eval-mode running stats). Verified: BN fold alone max|diff| 2.5e-06;
  norm+BN fold max|diff| 2.49e-06 in the interior (1px border excluded).
  The 1px border rim differs by up to 15.2: zero-padding is applied to
  the RAW input in the folded path but to the NORMALIZED input in the
  reference — the same class of border effect as RetinexNet's
  replicate-vs-zero padding (documented in the main README). The
  accelerator takes plain uint8 input (r_in=0 for L0), so the folded
  path is the correct one for the hardware; compare the RTL against the
  folded (raw-input) reference, or exclude the rim.
- **Activation ranges** (measured on target1.png): r_out distribution
  {0: 56, 1: 1, 2: 1, 3: 11, 4: 39, 5: 35, 6: 6, 7: 3, 8: 2}; shifts 7–15,
  all in [0,15]. Because the network branches (dense nested skips,
  attention multiplies, residual adds), r_out of one conv need not equal
  r_in of its consumer — 102/153 layers differ — so the compiler must
  insert a rescale (shift) where they differ; that is recorded in the
  quant JSON, not hidden. final1..4 are signed logits (r_out=0, no clip).
- **Per-layer fixed-vs-float** (pre-activation folded conv output):
  L0 55.05 dB, L7 57.21 dB, L23 55.54 dB; attention logits: L5
  (sa.conv1, 7x7) max|err| 0.845 logit units, final4 max|err| 0.625 logit
  units (PSNR is not reported for logits — it is meaningless off the
  [0,255] scale). The attention path consumes post-BN tensors, which are
  signed (input range [-0.987, 10.001] for sa.conv1): the check uses the
  signed-int8 input representation, and the remaining logit error is the
  genuine quantization error of that representation (its [-128,127] range
  at the 255/2^r scale clips positives above ~8 for r_in=4). A scale
  decision for signed attention tensors belongs to the compiler work,
  not this onboarding. The mask decision here is robust (target logits
  ~+49, background ~-107, far from the ~0.6-unit error scale), but
  borderline pixels on other images could flip — calibration on real
  SIRST data is still open.
- **Calibration is open**: ranges are measured on ONE demo image. No
  NUAA-SIRST IoU is claimed here (no ground-truth mask ships with the
  demo image; the checkpoint's recorded train mean IoU is 0.8489).
  Per-layer ranges and the mask threshold must be re-validated on real
  SIRST data before RTL use. Metric is IoU, not PSNR.
- **Accelerator split**: the 154 convs + maxpool + bilinear upsample +
  concat + elementwise multiply are accelerator candidates (all in the op
  set; sigmoid goes through the LUT activation module). The attention
  sigmoid + multiply and the final sigmoid/threshold stay as LUT +
  elementwise ops; no separate CPU post-processing is needed beyond the
  threshold. BN is folded (no runtime BN). All convs are stride 1.
- **How to use**: `cd models/vision/dnanet && python3 run_dnanet.py` (rebuilds
  the state dict from the float npz and verifies it, runs the float
  reference, re-quantizes, rewrites `dnanet.npz` +
  `dnanet_quant.json`). Re-extract: `python3 extract_dnanet.py`
  (rewrites `dnanet_float.npz`).
