#!/usr/bin/env python3
"""DnCNN (Zhang et al. 2017, sigma=25) — float reference + fixed-point
quantization check. Weights: official cszn/DnCNN model/specifics/sigma=25.mat
(17 conv layers, 3x3, 64ch; NO batch-norm layers in this release — conv+ReLU).
Forward: noisy -> L1 3x3+ReLU -> L2..L16 3x3+ReLU -> L17 3x3 (no act)
-> residual; denoised = noisy - residual. All convs zero-pad 1 (same size).
L17 output is SIGNED (residual, r_out=0, no [0,255] clip); denoised image is
clip(noisy_u8 - residual, 0, 255).
Compares fixed-point against the float reference per layer and end-to-end."""
import json, numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (quantize_layer, range_exp, conv2d_float, conv2d_fixed,
                    psnr, max_abs_err, make_test_gray)

d = np.load('dncnn_sigma25_float.npz')
NL = 17
LAYERS = [(f'L{i+1}', d[f'L{i+1}_W'], d[f'L{i+1}_b']) for i in range(NL)]

# --- test image: clean 32x32 + Gaussian noise sigma=25 (0..255 scale)
rng = np.random.default_rng(11)
clean = make_test_gray(32, 32, seed=7)
noise = rng.normal(0.0, 25.0, clean.shape)
noisy_f = np.clip(clean.astype(np.float64) + noise, 0, 255) / 255.0
noisy_u8 = np.clip(np.round(noisy_f * 255.0), 0, 255).astype(np.uint8)
print(f"clean range [{clean.min()},{clean.max()}]  noisy PSNR vs clean: "
      f"{psnr(noisy_u8, clean):.2f} dB")

# --- float reference (float in [0,1])
x = noisy_f[:, :, None]
float_outs = []
max_act = []
for name, W, b in LAYERS:
    x = conv2d_float(x, W, b, pad=(1, 1, 1, 1))
    if name != 'L17':
        x = np.maximum(x, 0.0)
    float_outs.append(x)
    max_act.append(float(x.max()))
    if name in ('L1', 'L2', 'L17'):
        print(f"  float {name}: out {x.shape} range [{x.min():+.4f},{x.max():+.4f}]")
resid_f = float_outs[-1][:, :, 0]
den_f = np.clip(noisy_f - resid_f, 0, 1)
print(f"  float residual range [{resid_f.min():+.4f},{resid_f.max():+.4f}]  "
      f"denoised PSNR vs clean: {psnr(den_f*255, clean):.2f} dB")

# --- per-layer activation range exponents (L1..L16 are ReLU outputs >= 0)
r_out = [range_exp(m) for m in max_act]
print("  activation range exponents r_out (L1..L17):", r_out)

# --- quantize + fixed-point
quant = {}
A = noisy_u8[:, :, None]
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
    signed = (name == 'L17')
    A = conv2d_fixed(A, W_q, b_q, shift, pad=(1, 1, 1, 1), signed_out=signed)
    fixed_outs.append(A)
    if signed:
        # residual: fixed signed int (residual*255) vs float residual*255
        # (squeeze channel dim to avoid numpy broadcast shape mismatch)
        f_res = xf[:, :, 0] * 255.0
        Aq = A[:, :, 0]
        print(f"  fixed {name} (signed residual): range [{Aq.min()},{Aq.max()}] shift={shift}  "
              f"vs float: max|err|={max_abs_err(Aq, f_res):.2f}  PSNR={psnr(Aq, f_res):.2f} dB")
    else:
        f_u8 = np.clip(xf * (255.0 / (2.0 ** ro)), 0, 255)
        if name in ('L1', 'L2', 'L16'):
            print(f"  fixed {name}: out {A.shape} range [{A.min()},{A.max()}] r_out={ro} shift={shift}  "
                  f"vs float: max|err|={max_abs_err(A, f_u8):.2f}  PSNR={psnr(A, f_u8):.2f} dB")
    quant[name].update({'W_q': W_q, 'b_q': b_q, 'W_f32': W.astype(np.float32),
                        'b_f32': b.astype(np.float32)})
    r_in = ro

resid_q = fixed_outs[-1][:, :, 0]  # signed, residual*255
den_q = np.clip(noisy_u8.astype(np.int64) - resid_q, 0, 255).astype(np.uint8)
print(f"END-TO-END (denoised): max|err| vs float denoised = "
      f"{max_abs_err(den_q, np.clip(den_f*255,0,255)):.2f}  "
      f"PSNR(fixed vs float denoised)={psnr(den_q, np.clip(den_f*255,0,255)):.2f} dB  "
      f"PSNR(fixed denoised vs clean)={psnr(den_q, clean):.2f} dB")
print(f"  denoised finite: {np.isfinite(den_q).all()}, range [{den_q.min()},{den_q.max()}]")

# --- save converted npz + quant json (full 17 layers; Stage 6 uses a fragment)
npz = {}
for i in range(NL):
    name = f'L{i+1}'
    q = quant[name]
    npz[name + '_W'] = q['W_f32']
    npz[name + '_W_q'] = q['W_q']
    npz[name + '_b'] = q['b_f32']
    npz[name + '_b_q'] = q['b_q']
np.savez('dncnn_sigma25.npz', **npz)
js = {k: {kk: vv for kk, vv in v.items() if kk not in ('W_q', 'b_q', 'W_f32', 'b_f32')}
      for k, v in quant.items()}
json.dump(js, open('dncnn_sigma25_quant.json', 'w'), indent=2)
print("saved dncnn_sigma25.npz + dncnn_sigma25_quant.json")
print("shifts:", {k: v['shift'] for k, v in js.items()})
