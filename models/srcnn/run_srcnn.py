#!/usr/bin/env python3
"""SRCNN (Dong et al. 2014, 9-1-5, x3) — float reference + fixed-point
quantization check. Weights: original author x3 (via FFmpeg patch email).
Pipeline: HR Y (33x33) -> bicubic down x3 -> LR (11x11) -> bicubic up x3
-> input (33x33) -> L1 9x9+ReLU -> L2 1x1+ReLU -> L3 5x5 -> Y out (21x21).
All convs valid (no padding), matching the original SRCNN.
Quantization: per-layer power-of-two weight scale + per-layer activation
range exponent r (output uint8 = float * 255/2**r); RTL shift = e + r_out - r_in.
Compares fixed-point against the float reference per layer and end-to-end."""
import json, numpy as np
from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (quantize_layer, range_exp, conv2d_float, conv2d_fixed,
                    psnr, max_abs_err, make_test_gray)

d = np.load('srcnn_x3_float.npz')
LAYERS = [('L1', d['L1_W'], d['L1_b']), ('L2', d['L2_W'], d['L2_b']),
          ('L3', d['L3_W'], d['L3_b'])]

# --- test image: HR 33x33 -> LR 11x11 -> bicubic-up 33x33 (SRCNN input)
hr = make_test_gray(33, 33, seed=7)
lr = np.array(Image.fromarray(hr).resize((11, 11), Image.BICUBIC))
inp = np.array(Image.fromarray(lr).resize((33, 33), Image.BICUBIC))
print(f"HR {hr.shape} -> LR {lr.shape} -> SRCNN input {inp.shape}, "
      f"input range [{inp.min()},{inp.max()}]")

# --- float reference (float in [0,1]); record per-layer max activation
x = (inp.astype(np.float64) / 255.0)[:, :, None]
float_outs = []
max_act = []
for name, W, b in LAYERS:
    x = conv2d_float(x, W, b, pad=(0, 0, 0, 0))
    if name != 'L3':
        x = np.maximum(x, 0.0)
    float_outs.append(x)
    max_act.append(float(x.max()))
    print(f"  float {name}: out {x.shape} range [{x.min():+.4f},{x.max():+.4f}]")
ref = float_outs[-1][:, :, 0]
ref255 = np.clip(ref * 255.0, 0, 255)
ctr = hr[6:27, 6:27].astype(np.float64)
print(f"  float ref vs HR-center PSNR: {psnr(ref255, ctr):.2f} dB "
      f"(bicubic input vs HR-center: {psnr(inp[6:27,6:27].astype(float), ctr):.2f} dB)")

# --- per-layer activation range exponents
r_out = [range_exp(m) for m in max_act]
print("  activation range exponents r_out:", r_out,
      " (output uint8 = float * 255/2**r)")

# --- quantize + fixed-point
quant = {}
A = inp[:, :, None]  # uint8, (H,W,1); r_in = 0
fixed_outs = []
r_in = 0
for (name, W, b), xf, ro in zip(LAYERS, float_outs, r_out):
    W_q, b_q, shift, scale = quantize_layer(W, b, r_in=r_in, r_out=ro)
    quant[name] = {'shift': shift, 'weight_exp_e': shift - ro + r_in,
                   'r_in': r_in, 'r_out': ro, 'scale': scale,
                   'max_abs_W': float(np.abs(W).max()),
                   'max_act': float(xf.max()),
                   'W_q_range': [int(W_q.min()), int(W_q.max())],
                   'b_q_range': [int(b_q.min()), int(b_q.max())]}
    A = conv2d_fixed(A, W_q, b_q, shift, pad=(0, 0, 0, 0))
    fixed_outs.append(A)
    # per-layer error vs float, both in uint8 scale (float * 255/2**ro)
    f_u8 = np.clip(xf * (255.0 / (2.0 ** ro)), 0, 255)
    print(f"  fixed {name}: out {A.shape} range [{A.min()},{A.max()}] r_out={ro} shift={shift}  "
          f"vs float: max|err|={max_abs_err(A, f_u8):.2f}  PSNR={psnr(A, f_u8):.2f} dB")
    quant[name].update({'W_q': W_q, 'b_q': b_q, 'W_f32': W.astype(np.float32),
                        'b_f32': b.astype(np.float32)})
    r_in = ro

fx = fixed_outs[-1][:, :, 0]
print(f"END-TO-END: max|err|={max_abs_err(fx, ref255):.2f}  "
      f"PSNR(fixed vs float ref)={psnr(fx, ref255):.2f} dB  "
      f"PSNR(fixed vs HR-center)={psnr(fx, ctr):.2f} dB")
print(f"  fixed output finite: {np.isfinite(fx).all()}, range [{fx.min()},{fx.max()}]")

# --- save converted npz + quant json
npz = {}
for name in ['L1', 'L2', 'L3']:
    q = quant[name]
    npz[name + '_W'] = q['W_f32']
    npz[name + '_W_q'] = q['W_q']
    npz[name + '_b'] = q['b_f32']
    npz[name + '_b_q'] = q['b_q']
np.savez('srcnn_x3.npz', **npz)
js = {k: {kk: vv for kk, vv in v.items() if kk not in ('W_q', 'b_q', 'W_f32', 'b_f32')}
      for k, v in quant.items()}
json.dump(js, open('srcnn_x3_quant.json', 'w'), indent=2)
print("saved srcnn_x3.npz + srcnn_x3_quant.json")
print("quant:", json.dumps(js, indent=2))
