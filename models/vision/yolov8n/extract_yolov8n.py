#!/usr/bin/env python3
"""Extract YOLOv8n weights from the official ultralytics checkpoint.
Source: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
(6,549,796 bytes, SHA256 f59b3d83...; COCO, 80 classes, 640 px, mAPval 37.3,
3,151,904 params, 8.7 GFLOPs; saved as raw/yolov8n.pt).

The .pt stores the full DetectionModel object, so it is loaded through the
ultralytics loader (which uses torch.load with weights_only=False; the file
is the official release, hash-verified — loading it is covered by the user's
explicit approval to load this checkpoint). 57 Conv modules, all Conv2d +
BatchNorm2d + SiLU; 39x 3x3, 18x 1x1, no depthwise.

Saves yolov8n_float.npz: per Conv module (model order) C{i}_W (H,W,Cin,Cout
float32), C{i}_b (Cout,), C{i}_bn_w / C{i}_bn_b / C{i}_bn_mean / C{i}_bn_var /
C{i}_bn_eps (BatchNorm params), C{i}_path (module path). BN is NOT folded
here — run_yolov8n.py folds it (eval-mode running stats) before quantizing,
and measures activation ranges on bus.jpg first."""
import json, numpy as np, torch

SRC = 'raw/yolov8n.pt'
import os
os.environ['YOLO_VERBOSE'] = 'false'
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv as UConv

m = YOLO(SRC)
model = m.model.eval()
print('task:', m.task, '| classes:', len(m.names))

uconvs = [(path, mod) for path, mod in model.named_modules()
          if isinstance(mod, UConv)]
print(f'Conv modules: {len(uconvs)}')
assert len(uconvs) == 57, len(uconvs)
k3 = sum(1 for _, c in uconvs if tuple(c.conv.kernel_size) == (3, 3))
k1 = sum(1 for _, c in uconvs if tuple(c.conv.kernel_size) == (1, 1))
assert (k3, k1) == (39, 18), (k3, k1)
assert all(c.bn is not None and type(c.act).__name__ == 'SiLU' for _, c in uconvs)
print(f'  3x3: {k3}, 1x1: {k1}; all have BN + SiLU')

npz = {}
paths = []
for i, (path, c) in enumerate(uconvs):
    W = c.conv.weight.detach().numpy().astype(np.float32)  # (Cout,Cin,H,W)
    W = np.transpose(W, (2, 3, 1, 0))                     # -> (H,W,Cin,Cout)
    b = c.conv.bias.detach().numpy().astype(np.float32) if c.conv.bias is not None \
        else np.zeros(W.shape[3], dtype=np.float32)
    bn = c.bn
    npz[f'C{i}_W'] = W
    npz[f'C{i}_b'] = b
    npz[f'C{i}_bn_w'] = bn.weight.detach().numpy().astype(np.float32)
    npz[f'C{i}_bn_b'] = bn.bias.detach().numpy().astype(np.float32)
    npz[f'C{i}_bn_mean'] = bn.running_mean.detach().numpy().astype(np.float32)
    npz[f'C{i}_bn_var'] = bn.running_var.detach().numpy().astype(np.float32)
    npz[f'C{i}_bn_eps'] = np.array([bn.eps], dtype=np.float32)
    paths.append(path)
    if i < 3 or i >= len(uconvs) - 3:
        print(f"  C{i:2d} {path:28s} W {W.shape}  b {b.shape}  "
              f"W range [{W.min():+.4f},{W.max():+.4f}]  "
              f"stride {tuple(c.conv.stride)}")
npz['paths'] = np.array(paths)
np.savez('yolov8n_float.npz', **npz)
print('saved yolov8n_float.npz')
print('class names:', json.dumps(list(m.names.values())))
print('total conv params:', sum(int(c.conv.weight.numel()) for _, c in uconvs))
