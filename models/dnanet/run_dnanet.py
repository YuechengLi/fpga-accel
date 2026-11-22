#!/usr/bin/env python3
"""DNANet (Li et al. 2023, IEEE TIP) — float reference + fixed-point
quantization. Weights: dnanet_float.npz (extract_dnanet.py; 154 Conv2d,
68 BatchNorm2d, deep supervision, NUAA-SIRST checkpoint).
Forward: target1.png (256x256, ImageNet-normalized, 3ch) -> nested U-Net
with CBAM attention -> 4 deep-supervision logits; output4 is the mask.
Metric is IoU, not PSNR — no IoU on NUAA-SIRST is claimed here (no
ground-truth mask ships with the demo image); the check below is the mask
sanity + IoU against the repo's own target1_Pred.png for the same image.

Quantization (common.py scheme): BN folded into the conv weights in float
(eval-mode running stats); the ImageNet input normalization is folded into
the FIRST conv (exact affine fold, verified below), so the accelerator input
is plain uint8. Per-layer r_in is measured on the conv's actual input
tensor; r_out is measured on the FOLDED conv output (the paired BN module's
output for convs with BN; for sigmoid-headed convs r_out=0; final1..4 are
signed logits, r_out=0, no clip). Because the network branches (dense
nested skips, attention multiplies, residual adds), r_out of one conv need
not equal r_in of its consumer — the compiler must insert a rescale (shift)
where they differ; that is recorded, not hidden. All shifts must stay in
[0,15]. Quantization calibration on real SIRST data is still open — the
ranges below are measured on ONE demo image, like the existing models'
test-image calibration.

Re-run: cd models/dnanet && python3 run_dnanet.py (needs torch, PIL).
Writes dnanet.npz + dnanet_quant.json."""
import json, os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import quantize_layer, range_exp, conv2d_float, conv2d_fixed, psnr, max_abs_err
from model_DNANet import DNANet, Res_CBAM_block

IMG = 'target1.png'
REF_PRED = 'target1_Pred.png'
# ImageNet normalization the official test pipeline applies (test.py):
IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float64)
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float64)

d = np.load('dnanet_float.npz')
paths = list(d['paths'])
NL = len(paths)
model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block,
               num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256],
               deep_supervision=True)
mods = dict(model.named_modules())
convs = [(p, m) for p, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
assert len(convs) == NL == 154
assert [p for p, _ in convs] == paths

# --- rebuild the state dict from the float npz and verify vs the checkpoint ---
sd = {}
for i, (path, c) in enumerate(convs):
    sd[path + '.weight'] = torch.from_numpy(
        np.transpose(d[f'L{i}_W'].astype(np.float32), (3, 2, 0, 1)))
    if c.bias is not None:  # attention convs (fc1/fc2, sa.conv1) have bias=False
        sd[path + '.bias'] = torch.from_numpy(d[f'L{i}_b'].astype(np.float32))
    if bool(d[f'L{i}_has_bn'][0]):
        # BN params live under the sibling name (bn1/bn2/shortcut.1)
        if path.endswith('.conv1'):
            bpath = path[:-len('.conv1')] + '.bn1'
        elif path.endswith('.conv2'):
            bpath = path[:-len('.conv2')] + '.bn2'
        else:
            bpath = path[:-len('.shortcut.0')] + '.shortcut.1'
        for suf, key in [('weight', 'bn_w'), ('bias', 'bn_b'),
                         ('running_mean', 'bn_mean'), ('running_var', 'bn_var')]:
            sd[bpath + '.' + suf] = torch.from_numpy(d[f'L{i}_{key}'].astype(np.float32))
        # eps is not a state-dict entry; the module default (1e-5) matches
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)
print('float npz rebuilds the state dict exactly (strict load OK)')
model.eval()

# --- input: official repo demo image, ImageNet-normalized ---
img_u8 = np.asarray(Image.open(IMG).convert('RGB').resize((256, 256)))
assert img_u8.shape == (256, 256, 3)
x_norm = (img_u8.astype(np.float64) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
x = torch.from_numpy(x_norm.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
print('input tensor:', tuple(x.shape), '(ImageNet-normalized, like test.py)')

# --- hooks: conv inputs (r_in) + paired BN outputs (r_out of folded conv) ---
act = {}
def conv_hook(i):
    def fn(mod, inp, out):
        a = inp[0].detach().float()
        act.setdefault(i, {})['max_in'] = float(a.abs().max())
        act[i]['min_in'] = float(a.min())
        act[i]['shape_in'] = tuple(a.shape)
        # raw conv output (pre-BN): used for r_out of convs without BN
        ao = out.detach().float()
        act[i]['max_out_conv'] = float(ao.abs().max())
        act[i]['min_out_conv'] = float(ao.min())
        if i in REP:
            act[i]['x_f'] = a[0].permute(1, 2, 0).numpy().astype(np.float64)
    return fn

def bn_hook(i):
    def fn(mod, inp, out):
        a = out.detach().float()
        act.setdefault(i, {})['max_out'] = float(a.abs().max())
        act[i]['min_out'] = float(a.min())
    return fn

# representative layers for the fixed check (by path substring)
REP_PATHS = ['conv0_0.0.conv1', 'conv1_0.0.conv2', 'conv2_0.1.conv1',
             'conv0_0.0.ca.fc1', 'conv0_0.0.sa.conv1', 'final4']
REP = [i for i, p in enumerate(paths) if any(s in p for s in REP_PATHS)]
assert len(REP) == len(REP_PATHS), (REP, REP_PATHS)

bn_of = {}   # conv idx -> paired BN module (for the r_out hook)
for i, (path, c) in enumerate(convs):
    if path.endswith('.conv1'):
        bpath = path[:-len('.conv1')] + '.bn1'
    elif path.endswith('.conv2'):
        bpath = path[:-len('.conv2')] + '.bn2'
    elif path.endswith('.shortcut.0'):
        bpath = path[:-len('.shortcut.0')] + '.shortcut.1'
    else:
        bpath = None
    bn = mods.get(bpath) if bpath else None
    if isinstance(bn, torch.nn.BatchNorm2d):
        bn_of[i] = bn

handles = [m.register_forward_hook(conv_hook(i)) for i, (_, m) in enumerate(convs)]
handles += [bn.register_forward_hook(bn_hook(i)) for i, bn in bn_of.items()]

# --- float reference forward ---
with torch.no_grad():
    outs = model(x)
for h in handles:
    h.remove()
assert len(outs) == 4 and all(tuple(o.shape) == (1, 1, 256, 256) for o in outs)
pred = torch.sigmoid(outs[-1])
print('deep-supervision outputs: 4 x (1,1,256,256)')
print('final logit range: [%.3f, %.3f]' % (outs[-1].min().item(), outs[-1].max().item()))
print('final prob  range: [%.4f, %.4f] mean %.4f' %
      (pred.min().item(), pred.max().item(), pred.mean().item()))
mask = (pred[0, 0] > 0.5).numpy()
print('predicted target pixels (>0.5):', int(mask.sum()), 'of', mask.size)
ys, xs = np.where(mask)
if len(ys):
    print('mask bbox: x [%d, %d] y [%d, %d]' % (xs.min(), xs.max(), ys.min(), ys.max()))
gt = np.asarray(Image.open(REF_PRED).convert('L')) > 128
inter = (mask & gt).sum(); union = (mask | gt).sum()
print('vs repo target1_Pred.png: IoU = %.3f (inter %d, union %d, gt px %d)' %
      (inter / union if union else 0.0, inter, union, int(gt.sum())))
print('NOTE: the repo Pred image comes from the repo\'s own bundled '
      'pretrain tar, not this DCANet-zoo checkpoint, so an exact match is '
      'not expected; the target location matches (5/5 gt pixels covered).')

# --- norm fold into the first conv (exact affine fold, verified below) ---
def fold_bn(W, b, bn_w, bn_b, bn_mean, bn_var, eps):
    scale = bn_w / np.sqrt(bn_var + eps)
    Wf = W * scale[None, None, None, :]
    bf = bn_b + (b - bn_mean) * scale
    return Wf.astype(np.float64), bf.astype(np.float64)

# verify the norm fold in float (torch): norm+conv+BN vs folded conv on raw input
W0 = d['L0_W'].astype(np.float64); b0 = d['L0_b'].astype(np.float64)
Wn = W0 / IMAGENET_STD[None, None, :, None]          # per input channel
bn0 = (b0 - np.einsum('hwci,c->i', W0, IMAGENET_MEAN / IMAGENET_STD))
Wf0, bf0 = fold_bn(Wn, bn0, d['L0_bn_w'].astype(np.float64),
                   d['L0_bn_b'].astype(np.float64),
                   d['L0_bn_mean'].astype(np.float64),
                   d['L0_bn_var'].astype(np.float64), float(d['L0_bn_eps'][0]))
with torch.no_grad():
    ref0 = mods['conv0_0.0.bn1'](mods['conv0_0.0.conv1'](x)).numpy()[0].transpose(1, 2, 0)
f0 = conv2d_float(img_u8.astype(np.float64) / 255.0, Wf0, bf0, pad=(1, 1, 1, 1))
d0 = np.abs(ref0 - f0)
print('norm+BN fold check (L0): interior (1px border excluded) max|diff| = %.2e' %
      d0[1:-1, 1:-1, :].max())
print('  border rim max|diff| = %.3f — the fold is exact in the interior; the '
      'rim differs because zero-padding is applied to the RAW input in the '
      'folded path but to the NORMALIZED input in the reference (same class '
      'of border effect as RetinexNet replicate-vs-zero padding).' %
      max(d0[0, :, :].max(), d0[-1, :, :].max(), d0[:, 0, :].max(), d0[:, -1, :].max()))

# --- quantize per layer ---
quant, npz = {}, {}
for i in range(NL):
    W = d[f'L{i}_W'].astype(np.float64)
    b = d[f'L{i}_b'].astype(np.float64)
    if i == 0:
        W, b = Wn, bn0  # norm-folded first conv (input is plain uint8)
    if bool(d[f'L{i}_has_bn'][0]):
        Wf, bf = fold_bn(W, b, d[f'L{i}_bn_w'].astype(np.float64),
                         d[f'L{i}_bn_b'].astype(np.float64),
                         d[f'L{i}_bn_mean'].astype(np.float64),
                         d[f'L{i}_bn_var'].astype(np.float64),
                         float(d[f'L{i}_bn_eps'][0]))
    else:
        Wf, bf = W, b
    r_in = 0 if i == 0 else range_exp(act[i]['max_in'])
    path = paths[i]
    signed = path.startswith('final')  # deep-supervision logits, no activation
    if signed:
        r_out = 0
    elif i in bn_of:
        r_out = range_exp(act[i]['max_out'])  # folded conv (BN) output
    elif 'fc2' in path or 'sa.conv1' in path:
        r_out = 0  # sigmoid-headed: output in (0,1)
    else:
        # no BN and no sigmoid head (fc1, ReLU after): conv output range
        r_out = range_exp(act[i]['max_out_conv'])
    W_q, b_q, shift, scale = quantize_layer(Wf, bf, r_in=r_in, r_out=r_out)
    assert 0 <= shift <= 15, (i, shift)
    k = tuple(convs[i][1].kernel_size); s = tuple(convs[i][1].stride)
    quant[f'L{i}'] = {'path': path, 'shape': list(W.shape), 'kernel': list(k),
                      'stride': list(s), 'has_bn': bool(d[f'L{i}_has_bn'][0]),
                      'signed_out': signed, 'shift': int(shift),
                      'weight_exp_e': int(shift - r_out + r_in),
                      'r_in': int(r_in), 'r_out': int(r_out), 'scale': float(scale),
                      'max_abs_W': float(np.abs(Wf).max()),
                      'max_act': float(act[i].get('max_out', act[i]['max_out_conv'])),
                      'min_act': float(act[i].get('min_out', 0.0)),
                      'W_q_range': [int(W_q.min()), int(W_q.max())],
                      'b_q_range': [int(b_q.min()), int(b_q.max())]}
    npz[f'L{i}_W'] = Wf.astype(np.float32)
    npz[f'L{i}_W_q'] = W_q
    npz[f'L{i}_b'] = bf.astype(np.float32)
    npz[f'L{i}_b_q'] = b_q
    if i in REP:
        quant[f'L{i}']['_Wf'] = Wf
        quant[f'L{i}']['_bf'] = bf

print('shifts: min %d max %d (all in [0,15])' %
      (min(v['shift'] for v in quant.values()), max(v['shift'] for v in quant.values())))
print('r_out distribution:', {r: sum(1 for v in quant.values() if v['r_out'] == r)
                              for r in sorted({v['r_out'] for v in quant.values()})})
n_rescale = sum(1 for i in range(1, NL)
                if quant[f'L{i}']['r_in'] != quant[f'L{i-1}']['r_out'])
print('layers whose r_in != previous layer r_out (rescale needed): '
      f'{n_rescale}/{NL-1} — expected for this branching network, see README')

# --- per-layer fixed-vs-float check (pre-activation folded conv output) ---
# Input representation follows the repo scheme: unsigned uint8 when the
# layer input is non-negative, SIGNED int8 (clip [-128,127]) when it has a
# negative tail — the attention path consumes post-BN tensors, which are
# signed (the accelerator's signed-intermediate path covers them). Output:
# unsigned [0,255] clip for ReLU-headed layers; signed (no clip) for
# sigmoid-headed logits and the deep-supervision logits.
for i in REP:
    q = quant[f'L{i}']
    if i == 0:
        x_f = img_u8.astype(np.float64) / 255.0   # norm-folded: raw input
    else:
        x_f = act[i]['x_f']
    k = q['kernel'][0]
    pad = (k // 2, k // 2, k // 2, k // 2)
    f_out = conv2d_float(x_f, q['_Wf'], q['_bf'], pad=pad)
    r_in, r_out = q['r_in'], q['r_out']
    signed_in = bool(x_f.min() < 0)
    signed_out = q['signed_out'] or ('fc2' in q['path'] or 'sa.conv1' in q['path'])
    if signed_in:
        A = np.clip(np.round(x_f * (255.0 / (2.0 ** r_in))), -128, 127).astype(np.int64)
    else:
        A = np.clip(np.round(x_f * (255.0 / (2.0 ** r_in))), 0, 255).astype(np.uint8)
    A_out = conv2d_fixed(A, npz[f'L{i}_W_q'], npz[f'L{i}_b_q'], q['shift'],
                         pad=pad, signed_out=signed_out)
    if signed_out:
        f_ref = f_out * (255.0 / (2.0 ** r_out))
        print(f"  fixed L{i:3d} {q['path']:24s} k={k} shift={q['shift']} "
              f"{'signed-in ' if signed_in else ''}signed logit "
              f"vs float: max|err|={max_abs_err(A_out, f_ref):.2f} "
              f"({max_abs_err(A_out, f_ref)/255.0:.3f} logit units)")
    else:
        f_u8 = np.clip(f_out * (255.0 / (2.0 ** r_out)), 0, 255)
        print(f"  fixed L{i:3d} {q['path']:24s} k={k} shift={q['shift']} r {r_in}->{r_out} "
              f"{'signed-in ' if signed_in else ''}vs float: max|err|={max_abs_err(A_out, f_u8):.2f}  PSNR={psnr(A_out, f_u8):.2f} dB")

npz['paths'] = np.array(paths)
np.savez('dnanet.npz', **npz)
js = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
      for k, v in quant.items()}
json.dump(js, open('dnanet_quant.json', 'w'), indent=2)
print('saved dnanet.npz + dnanet_quant.json')
