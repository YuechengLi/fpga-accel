#!/usr/bin/env python3
"""YOLOv8n (ultralytics 8.3.0) — float reference + fixed-point quantization.
Weights: yolov8n_float.npz (extract_yolov8n.py; 57 Conv modules, all
Conv2d + BatchNorm2d + SiLU). Forward: bus.jpg -> letterbox 640 -> backbone
+ neck + Detect head (raw (1,84,8400) output) -> CPU decode + NMS.
The Detect head's DFL decode + NMS stay on CPU (accelerator split); the
57 convs (3x3/1x1, SiLU, SPPF maxpool, concat, upsample) are the
accelerator candidates.

Quantization (common.py scheme): BN folded into the conv weights in float
(eval-mode running stats); per-layer r_in/r_out measured on bus.jpg via
hooks on every Conv module; quantize_layer gives int16 W_q / int32 b_q /
shift (all shifts must stay in [0,15]). SiLU outputs have a small negative
tail (min ~-0.28) which the uint8 input representation of the NEXT layer
clips to 0 — recorded as min_act per layer; it is part of the quantization
error, not a bug. Detection is scored in mAP, not PSNR — no mAP is claimed
here; the numbers below are the float reference detections on bus.jpg and
per-layer fixed-vs-float conv errors.

Re-run: cd models/yolov8n && python3 run_yolov8n.py (needs ultralytics,
torch, PIL). Writes yolov8n.npz + yolov8n_quant.json."""
import json, os, sys
import numpy as np
import torch
from PIL import Image
os.environ['YOLO_VERBOSE'] = 'false'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import quantize_layer, range_exp, conv2d_float, conv2d_fixed, psnr, max_abs_err
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv as UConv

# --- letterbox + CPU decode/NMS (same as the standalone verification) ---
def letterbox(im0, size=640, pad_value=114):
    h0, w0 = im0.shape[:2]
    r = min(size / h0, size / w0)
    h1, w1 = round(h0 * r), round(w0 * r)
    im = np.array(Image.fromarray(im0).resize((w1, h1), Image.BILINEAR))
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    ph, pw = (size - h1) // 2, (size - w1) // 2
    out[ph:ph + h1, pw:pw + w1] = im
    return out, r, pw, ph

def nms(boxes, scores, thr=0.7):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter)
        order = order[1:][iou <= thr]
    return keep

def decode_nms(out, names, im0_shape, conf_thr=0.25, iou_thr=0.7):
    pred = out[0].float().numpy()
    boxes = pred[:4].T
    scores = pred[4:].T
    cls = scores.argmax(1); conf = scores.max(1)
    keep = conf > conf_thr
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    r = min(640 / im0_shape[0], 640 / im0_shape[1])
    pad_w = (640 - im0_shape[1] * r) / 2
    pad_h = (640 - im0_shape[0] * r) / 2
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_w) / r
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_h) / r
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - pad_w) / r
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - pad_h) / r
    final = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for j in nms(xyxy[idx], conf[idx], iou_thr):
            i = idx[j]
            final.append((names[int(c)], float(conf[i]),
                          [round(float(v), 1) for v in xyxy[i]]))
    return sorted(final, key=lambda t: -t[1])

# --- load float weights + model ---
d = np.load('yolov8n_float.npz')
paths = list(d['paths'])
NL = len(paths)
m = YOLO('raw/yolov8n.pt')
model = m.model.eval()
uconvs = [(p, mod) for p, mod in model.named_modules() if isinstance(mod, UConv)]
assert len(uconvs) == NL == 57
assert [p for p, _ in uconvs] == paths, 'module order differs from extraction'

REP = [0, 8, 16, 24, 40, 48, 52, 56]  # representative layers for the fixed check
act = {}   # idx -> dict(max_in, min_in, max_out, min_out)
stored = {}  # idx -> (input float (H,W,C), output float (H,W,C)) for REP layers

def hook(idx):
    def fn(mod, inp, out):
        a_in = inp[0].detach().float()
        a_out = out.detach().float()
        act[idx] = {'max_in': float(a_in.abs().max()), 'min_in': float(a_in.min()),
                    'max_out': float(a_out.abs().max()), 'min_out': float(a_out.min()),
                    'shape_in': tuple(a_in.shape), 'shape_out': tuple(a_out.shape)}
        if idx in REP:
            stored[idx] = (a_in[0].permute(1, 2, 0).numpy().astype(np.float64),
                           a_out[0].permute(1, 2, 0).numpy().astype(np.float64))
    return fn

handles = [mod.register_forward_hook(hook(i)) for i, (_, mod) in enumerate(uconvs)]

# --- float reference forward on bus.jpg ---
im0 = np.array(Image.open('bus.jpg').convert('RGB'))
im, _, _, _ = letterbox(im0, 640)
x = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0
with torch.no_grad():
    out = model(x)
if isinstance(out, (tuple, list)):
    out = out[0]
for h in handles:
    h.remove()
dets = decode_nms(out, m.names, im0.shape)
print(f'float reference detections on bus.jpg (conf>0.25, iou 0.7): {len(dets)}')
for name, cf, bb in dets:
    print(f'  {name:12s} conf={cf:.3f} xyxy={bb}')
assert len(act) == NL

# --- BN fold + quantize per layer (ranges measured above) ---
def fold_bn(W, b, bn_w, bn_b, bn_mean, bn_var, eps):
    scale = bn_w / np.sqrt(bn_var + eps)          # (Cout,)
    Wf = W * scale[None, None, None, :]
    bf = bn_b + (b - bn_mean) * scale
    return Wf.astype(np.float64), bf.astype(np.float64)

quant, npz = {}, {}
for i in range(NL):
    W = d[f'C{i}_W'].astype(np.float64)
    b = d[f'C{i}_b'].astype(np.float64)
    Wf, bf = fold_bn(W, b, d[f'C{i}_bn_w'].astype(np.float64),
                     d[f'C{i}_bn_b'].astype(np.float64),
                     d[f'C{i}_bn_mean'].astype(np.float64),
                     d[f'C{i}_bn_var'].astype(np.float64),
                     float(d[f'C{i}_bn_eps'][0]))
    r_in = range_exp(act[i]['max_in'])
    r_out = range_exp(act[i]['max_out'])
    W_q, b_q, shift, scale = quantize_layer(Wf, bf, r_in=r_in, r_out=r_out)
    assert 0 <= shift <= 15, (i, shift)
    k = tuple(uconvs[i][1].conv.kernel_size)
    s = tuple(uconvs[i][1].conv.stride)
    quant[f'C{i}'] = {'path': paths[i], 'shape': list(W.shape), 'kernel': list(k),
                      'stride': list(s), 'shift': int(shift),
                      'weight_exp_e': int(shift - r_out + r_in),
                      'r_in': int(r_in), 'r_out': int(r_out), 'scale': float(scale),
                      'max_abs_W': float(np.abs(Wf).max()),
                      'max_act': float(act[i]['max_out']),
                      'min_act': float(act[i]['min_out']),
                      'W_q_range': [int(W_q.min()), int(W_q.max())],
                      'b_q_range': [int(b_q.min()), int(b_q.max())]}
    npz[f'C{i}_W'] = Wf.astype(np.float32)
    npz[f'C{i}_W_q'] = W_q
    npz[f'C{i}_b'] = bf.astype(np.float32)
    npz[f'C{i}_b_q'] = b_q
    if i in REP:
        quant[f'C{i}']['_Wf'] = Wf
        quant[f'C{i}']['_bf'] = bf

print('shifts:', {k: v['shift'] for k, v in quant.items()})
print('r_out per layer:', [quant[f'C{i}']['r_out'] for i in range(NL)])
print('min_act (SiLU negative tail) per layer, worst: %.3f' %
      min(quant[f'C{i}']['min_act'] for i in range(NL)))

# --- per-layer fixed-vs-float check (pre-activation conv output) ---
for i in REP:
    q = quant[f'C{i}']
    x_f, _ = stored[i]
    k = q['kernel'][0]
    pad = (k // 2, k // 2, k // 2, k // 2)
    s = q['stride'][0]
    Wf, bf = q['_Wf'], q['_bf']
    f_out = conv2d_float(x_f, Wf, bf, pad=pad)
    if s > 1:
        f_out = f_out[::s, ::s, :]
    r_in, r_out = q['r_in'], q['r_out']
    A = np.clip(np.round(x_f * (255.0 / (2.0 ** r_in))), 0, 255).astype(np.uint8)
    W_q = npz[f'C{i}_W_q']; b_q = npz[f'C{i}_b_q']
    A_out = conv2d_fixed(A, W_q, b_q, q['shift'], pad=pad)
    if s > 1:
        A_out = A_out[::s, ::s, :]
    f_u8 = np.clip(f_out * (255.0 / (2.0 ** r_out)), 0, 255)
    print(f"  fixed C{i:2d} {q['path']:24s} k={k} s={s} shift={q['shift']} r {r_in}->{r_out} "
          f"vs float: max|err|={max_abs_err(A_out, f_u8):.2f}  PSNR={psnr(A_out, f_u8):.2f} dB")

npz['paths'] = np.array(paths)
np.savez('yolov8n.npz', **npz)
js = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
      for k, v in quant.items()}
json.dump(js, open('yolov8n_quant.json', 'w'), indent=2)
print('saved yolov8n.npz + yolov8n_quant.json')
