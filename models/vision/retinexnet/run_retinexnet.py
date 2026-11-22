#!/usr/bin/env python3
"""RetinexNet Decom-Net (Chen et al. 2018) — float reference + fixed-point
quantization check. Weights: PyTorch port aasharma90/RetinexNet_PyTorch,
checkpoint ckpts/Decom/9200.tar (official TF checkpoint is on Google Drive/
Baidu, not in the repo; port fallback, see models/README.md).
Forward: RGB -> max channel -> concat [max,R,G,B] (4ch) -> conv0 9x9 (NO act)
-> 5x (conv 3x3 + ReLU) -> recon 3x3 -> sigmoid -> R (3ch), L (1ch).
Padding is REPLICATE (edge) in the original; the accelerator uses zero-fill,
so the zero-vs-replicate border effect is quantified separately.
conv0 has NO activation (signed output). The fixed-point golden inserts a
ReLU after conv0 (clip negatives to 0) so the activation stays uint8; the
error this introduces is measured below.
Compares fixed-point against the float reference per layer and end-to-end."""
import json, numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (quantize_layer, range_exp, conv2d_float, conv2d_fixed,
                    psnr, max_abs_err, make_test_rgb)

d = np.load('retinexnet_decom_float.npz')
LAYERS = [('conv0', d['L1_W'], d['L1_b'])] + \
         [(f'conv{i}', d[f'L{i+1}_W'], d[f'L{i+1}_b']) for i in range(1, 6)] + \
         [('recon', d['L7_W'], d['L7_b'])]
KS = {'conv0': 9, 'recon': 3}
for i in range(1, 6):
    KS[f'conv{i}'] = 3
PADS = {'conv0': 4, 'recon': 1}
for i in range(1, 6):
    PADS[f'conv{i}'] = 1

# --- test image: RGB 32x32, scaled to low light (x0.25)
rgb = make_test_rgb(32, 32, seed=7)
low = np.clip(rgb.astype(np.float64) * 0.25, 0, 255).astype(np.uint8)
print(f"RGB range [{rgb.min()},{rgb.max()}] -> low-light range [{low.min()},{low.max()}]")

# --- float reference (replicate padding, float in [0,1])
def fwd(x4, pad_mode):
    """x4: (H,W,4) float [0,1]. Returns (R, L, layer outputs, conv0 pre-ReLU)."""
    outs = []
    x = x4
    for name, W, b in LAYERS:
        p = PADS[name]
        x = conv2d_float(x, W, b, pad=(p, p, p, p), pad_mode=pad_mode)
        if name == 'conv0':
            conv0_raw = x.copy()  # no activation after conv0
            # (no ReLU in the original; kept as-is for the float reference)
        elif name != 'recon':
            x = np.maximum(x, 0.0)
        outs.append(x)
    R = 1.0 / (1.0 + np.exp(-x[:, :, 0:3]))
    L = 1.0 / (1.0 + np.exp(-x[:, :, 3:4]))
    return R, L, outs, conv0_raw

mx = low.max(axis=2, keepdims=True).astype(np.float64) / 255.0
x4 = np.concatenate([mx, low.astype(np.float64) / 255.0], axis=2)  # [max,R,G,B]
R_f, L_f, float_outs, conv0_raw = fwd(x4, 'replicate')
print(f"  float conv0 (no act): range [{conv0_raw.min():+.4f},{conv0_raw.max():+.4f}]  "
      f"frac negative: {(conv0_raw < 0).mean()*100:.2f}%  min {conv0_raw.min():+.4f}")
print(f"  float recon (logit): range [{float_outs[-1].min():+.4f},{float_outs[-1].max():+.4f}]")
print(f"  float R range [{R_f.min():.4f},{R_f.max():.4f}] mean {R_f.mean():.4f} | "
      f"L range [{L_f.min():.4f},{L_f.max():.4f}] mean {L_f.mean():.4f}")
# sanity: R * L should approximate the input (low-light) image
recon_img = R_f * L_f[:, :, 0:1]
print(f"  sanity R*L vs low-light input: PSNR={psnr(np.clip(recon_img*255,0,255), low):.2f} dB "
      f"(input is dark; R*L should reconstruct it)")

# --- per-layer activation range exponents.
# conv0 output is signed; the fixed-point golden clips negatives (ReLU) after
# conv0, so its r_out is measured on the clipped (ReLU'd) conv0 output.
max_act = []
for (name, W, b), xf in zip(LAYERS, float_outs):
    if name == 'conv0':
        max_act.append(float(np.maximum(xf, 0.0).max()))
    elif name == 'recon':
        # recon logit is signed; use its positive max for the scale reference
        # (the signed output is stored unclipped, scale 255 = r_out 0)
        max_act.append(float(xf.max()))
    else:
        max_act.append(float(xf.max()))
r_out = [range_exp(m) for m in max_act]
# recon is signed output, r_out forced to 0 (output = logit*255, signed)
r_out[-1] = 0
print("  activation range exponents r_out:", r_out,
      "(conv0 is signed int8, no ReLU — see below)")

# --- quantize + fixed-point (replicate padding)
quant = {}
mx_u8 = low.max(axis=2, keepdims=True)
A = np.concatenate([mx_u8, low], axis=2).astype(np.uint8)  # [max,R,G,B], uint8
fixed_outs = []
r_in = 0
# SIGNED-INTERMEDIATE SCHEME: conv0 has no activation, so its output is
# signed (~50% negative, range [-0.33,+0.31]). It is stored as SIGNED int8:
#   s0 = clip(round(x*255), -128, 127)   (x directly, no offset)
# conv1 takes int8 input (signed x signed multiply). Zero padding inserts 0,
# which correctly represents x=0 for a signed activation.
# (An earlier uint8+offset scheme, u=(x+0.5)*255, worked under replicate
# padding but FAILED under zero padding: zero-padded pixels carry no offset
# while the bias correction assumes all pixels do, so conv1 erred by up to
# 215 at the border. Signed int8 is the correct, padding-robust scheme.
# It requires the RTL conv datapath to accept signed intermediate activations
# -- see models/README.md.)
for (name, W, b), xf, ro in zip(LAYERS, float_outs, r_out):
    W_q, b_q, shift, scale = quantize_layer(W, b, r_in=r_in, r_out=ro)
    quant[name] = {'shift': shift, 'weight_exp_e': shift - ro + r_in,
                   'r_in': r_in, 'r_out': ro, 'scale': scale,
                   'max_abs_W': float(np.abs(W).max()),
                   'max_act': float(xf.max()),
                   'W_q_range': [int(W_q.min()), int(W_q.max())],
                   'b_q_range': [int(b_q.min()), int(b_q.max())],
                   'activation': ('int8-signed' if name == 'conv0' else
                                  ('int32-signed' if name == 'recon' else 'uint8'))}
    p = PADS[name]
    signed = (name in ('conv0', 'recon'))
    A = conv2d_fixed(A, W_q, b_q, shift, pad=(p, p, p, p),
                     pad_mode='replicate', signed_out=signed)
    fixed_outs.append(A)
    if name == 'conv0':
        # conv0 output is signed int8 (clip to [-128,127] after the shift)
        A = np.clip(A, -128, 127)
        f_s = np.clip(xf * 255.0, -128, 127)
        print(f"  fixed conv0 (signed int8): out {A.shape} range [{A.min()},{A.max()}] r_out={ro} shift={shift}  "
              f"vs float: max|err|={max_abs_err(A, f_s):.2f}  PSNR={psnr(A, f_s):.2f} dB")
    elif signed:
        f_logit = xf * 255.0
        Aq = A
        print(f"  fixed recon (signed logit): range [{Aq.min()},{Aq.max()}] shift={shift}  "
              f"vs float logit*255: max|err|={max_abs_err(Aq, f_logit):.2f}  PSNR={psnr(Aq, f_logit):.2f} dB")
    else:
        f_u8 = np.clip(xf * (255.0 / (2.0 ** ro)), 0, 255)
        if name in ('conv1', 'conv5'):
            print(f"  fixed {name}: out {A.shape} range [{A.min()},{A.max()}] r_out={ro} shift={shift}  "
                  f"vs float: max|err|={max_abs_err(A, f_u8):.2f}  PSNR={psnr(A, f_u8):.2f} dB")
    quant[name].update({'W_q': W_q, 'b_q': b_q, 'W_f32': W.astype(np.float32),
                        'b_f32': b.astype(np.float32)})
    r_in = ro

# --- end-to-end: R/L via sigmoid on the fixed recon logit
logit_q = fixed_outs[-1].astype(np.float64) / 255.0
R_q = 1.0 / (1.0 + np.exp(-logit_q[:, :, 0:3]))
L_q = 1.0 / (1.0 + np.exp(-logit_q[:, :, 3:4]))
print(f"END-TO-END R: max|err|={max_abs_err(R_q*255, R_f*255):.2f}  "
      f"PSNR(fixed vs float R)={psnr(R_q*255, R_f*255):.2f} dB")
print(f"END-TO-END L: max|err|={max_abs_err(L_q*255, L_f*255):.2f}  "
      f"PSNR(fixed vs float L)={psnr(L_q*255, L_f*255):.2f} dB")
print(f"  fixed R range [{(R_q*255).min():.1f},{(R_q*255).max():.1f}] mean {(R_q*255).mean():.1f} | "
      f"L range [{(L_q*255).min():.1f},{(L_q*255).max():.1f}] mean {(L_q*255).mean():.1f}")

# --- border effect: zero padding vs replicate padding (fixed-point, same weights)
A0 = np.concatenate([mx_u8, low], axis=2).astype(np.uint8)
r_in = 0
for (name, W, b), ro in zip(LAYERS, r_out):
    q = quant[name]
    p = PADS[name]
    signed = (name in ('conv0', 'recon'))
    A0 = conv2d_fixed(A0, q['W_q'], q['b_q'], q['shift'], pad=(p, p, p, p),
                      pad_mode='zero', signed_out=signed)
    if name == 'conv0':
        A0 = np.clip(A0, -128, 127)
    r_in = ro
logit_z = A0.astype(np.float64) / 255.0
R_z = 1.0 / (1.0 + np.exp(-logit_z[:, :, 0:3]))
L_z = 1.0 / (1.0 + np.exp(-logit_z[:, :, 3:4]))
print(f"BORDER EFFECT (zero-pad vs replicate-pad, fixed-point): "
      f"R max|diff|={max_abs_err(R_z*255, R_q*255):.2f} PSNR={psnr(R_z*255, R_q*255):.2f} dB; "
      f"L max|diff|={max_abs_err(L_z*255, L_q*255):.2f} PSNR={psnr(L_z*255, L_q*255):.2f} dB")
# how far in does the border effect reach? (conv0 is 9x9 -> 4 px; recon 3x3 -> +1)
inner = slice(5, 27)
print(f"  interior (5px border excluded) R max|diff|={max_abs_err(R_z[inner,inner]*255, R_q[inner,inner]*255):.2f}, "
      f"L max|diff|={max_abs_err(L_z[inner,inner]*255, L_q[inner,inner]*255):.2f}")

# --- save converted npz + quant json
npz = {}
for name, _, _ in LAYERS:
    q = quant[name]
    npz[name + '_W'] = q['W_f32']
    npz[name + '_W_q'] = q['W_q']
    npz[name + '_b'] = q['b_f32']
    npz[name + '_b_q'] = q['b_q']
np.savez('retinexnet_decom.npz', **npz)
js = {k: {kk: vv for kk, vv in v.items() if kk not in ('W_q', 'b_q', 'W_f32', 'b_f32')}
      for k, v in quant.items()}
json.dump(js, open('retinexnet_decom_quant.json', 'w'), indent=2)
print("saved retinexnet_decom.npz + retinexnet_decom_quant.json")
print("shifts:", {k: v['shift'] for k, v in js.items()})
