# 4. YOLOv8n — Ultralytics YOLOv8 nano, object detection (COCO)

- **Source URL**: official ultralytics release,
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
  (6,549,796 bytes, SHA256
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`).
  Saved as `yolov8n/raw/yolov8n.pt`.
- **Files**: `yolov8n/raw/yolov8n.pt` (source);
  `yolov8n/yolov8n_float.npz` (extracted float32, BN params kept separate);
  `yolov8n/yolov8n.npz` (BN-folded float32 + int16 quantized);
  `yolov8n/yolov8n_quant.json` (per-layer scale/shift);
  `yolov8n/extract_yolov8n.py`, `yolov8n/run_yolov8n.py`;
  `yolov8n/bus.jpg` (standard ultralytics test image, 810x1080).
- **Model** (ultralytics 8.3.0, COCO 80 classes, 640 px, mAPval 37.3,
  3,151,904 params, 8.7 GFLOPs): 57 Conv modules, ALL Conv2d + BatchNorm2d
  + SiLU; 39x 3x3, 18x 1x1, no depthwise. Conv params 3,114,672.
  Backbone (CSPDarknet) + PAN neck + Detect head. The Detect head's
  cv2/cv3 convs (12 of the 57) are included in the extraction; the DFL
  decode (fixed 0..15 weights, not learned) and NMS are NOT accelerator
  ops — they stay on CPU, as does the final class-score thresholding.
- **Forward** (float reference, `run_yolov8n.py`): bus.jpg -> letterbox 640
  -> raw DetectionModel forward (1,84,8400) -> CPU decode + NMS
  (conf>0.25, IoU 0.7). Detections reproduced exactly in both the
  standalone verification and `run_yolov8n.py`:
  person 0.896, person 0.883, person 0.873, bus 0.836, person 0.430.
  (5 detections; the 0.430 person is a partial figure at the left edge.)
- **BatchNorm folding** (eval-mode running stats, folded in float before
  quantizing): verified exact — torch BN(conv(x)) vs numpy folded conv
  max|diff| 1.7e-06 – 4.5e-06 on the worst-error layers (C16/C24/C40).
  Folded weights reach max|W| 17.5 (raw max 0.54): BN amplifies some
  channels ~35x where the running variance is tiny. That amplification
  is real (it is what the network computes), not an extraction error.
- **Activation ranges** (measured on bus.jpg via hooks on every Conv
  module): r_out up to 7 (C1, C56 — activations up to ~128). The
  single-exponent uint8 scheme is coarse at these ranges, so the
  per-layer fixed-vs-float errors are materially worse than the three
  image models: representative layers C0 54.74 dB, C8 38.55 dB, C16
  30.55 dB, C24 27.20 dB, C40 28.91 dB, C48 42.90 dB, C52 31.33 dB,
  C56 42.90 dB (pre-activation conv output). Causes: (1) large activation
  ranges (r_out up to 7, i.e. 0.5 per LSB); (2) BN-folded weight dynamic
  range (max|W| up to 17.5); (3) the SiLU negative tail (min -0.278)
  clipped to 0 by the next layer's uint8 input. All shifts are in
  [0,15] (12–15). No weight clipping occurs (W_q within int16).
- **Calibration is open**: ranges are measured on ONE image (bus.jpg).
  Per-layer ranges must be re-validated on wider data before RTL use,
  and the coarse-uint8 error above says a wider calibration — and
  possibly per-channel scales or a larger activation width — is needed
  before this model can run end-to-end on the accelerator. Detection
  quality is scored in mAP, not PSNR: no mAP is claimed here (that needs
  a COCO eval, not done); the numbers above are per-layer conv errors
  and the bus.jpg detections.
- **Accelerator split**: the 57 convs + SPPF maxpool + concat + upsample
  are accelerator candidates (all in the op set; SiLU goes through the
  LUT activation module). DFL decode + NMS + score thresholding stay on
  CPU. The first conv takes plain uint8 input (r_in=0); no input
  normalization to fold.
- **Environment note**: `run_yolov8n.py` needs the `ultralytics` package
  (`pip install ultralytics`). On this VM the installed torchvision is
  skewed vs torch 2.15.0+cpu (`torchvision::nms` op missing), so
  `ultralytics.predict()` fails at warmup; the script therefore uses the
  raw DetectionModel forward + the CPU decode/NMS above — which is
  exactly the accelerator split anyway. `extract_yolov8n.py` loads the
  .pt through the ultralytics loader (torch.load with weights_only=False;
  the file is the official release, hash-verified — loading it is covered
  by the user's explicit approval for this checkpoint).
- **How to use**: `cd models/vision/yolov8n && python3 run_yolov8n.py` (runs the
  float reference, prints detections, re-quantizes, rewrites
  `yolov8n.npz` + `yolov8n_quant.json`). Re-extract:
  `python3 extract_yolov8n.py` (rewrites `yolov8n_float.npz`).
