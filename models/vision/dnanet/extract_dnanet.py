#!/usr/bin/env python3
"""Extract DNANet weights from the DCANet Model Zoo NUAA-SIRST checkpoint.
Source: DCANet Model Zoo (github.com/Tianzishu/DCANet), DNANet row,
NUAA-SIRST column, Google Drive file 12XKXFEu8JUv0hMCP1hoRXmYTuJp_k1LP
(19,033,849 bytes, SHA256 fcc9b52c...; saved as raw/dnanet_nuaa-sirst.pth.tar).
Checkpoint: epoch 1499, recorded train loss 0.1739, mean IoU 0.8489.

The checkpoint is a torch 1.x save, so weights_only=True needs the legacy
numpy globals allowlisted (numpy.core.multiarray.scalar, numpy.dtype and
the numpy 2.x dtype classes); no full-pickle load is used. The model
definition is the official repo's model_DNANet.py (vendored here, MIT —
see LICENSE.upstream), instantiated with the repo test defaults:
Res_CBAM_block, nb_filter [16,32,64,128,256] (channel_size 'three'),
num_blocks [2,2,2,2] (backbone 'resnet_18'), deep_supervision=True,
input_channels=3. The state dict loads strictly (no missing/unexpected).

Saves dnanet_float.npz: per Conv2d module (model order) L{i}_W (H,W,Cin,Cout
float32), L{i}_b (Cout,), L{i}_bn_w / L{i}_bn_b / L{i}_bn_mean / L{i}_bn_var /
L{i}_bn_eps (BatchNorm params, or absent when the conv has no BN),
L{i}_path, L{i}_pad, L{i}_stride, L{i}_has_bn. BN is NOT folded here —
run_dnanet.py folds it (eval-mode running stats) before quantizing, and
measures activation ranges on target1.png first. 154 convs: 52x 3x3,
76x 1x1, 26x 7x7 (spatial attention); 68 BatchNorm2d."""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.abspath('.'))
from model_DNANet import DNANet, Res_CBAM_block

CKPT = 'raw/dnanet_nuaa-sirst.pth.tar'

torch.serialization.add_safe_globals([
    (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
    np.dtype,
] + [getattr(np.dtypes, n) for n in [
    'Float64DType', 'Float32DType', 'Float16DType', 'Int64DType', 'Int32DType',
    'Int16DType', 'Int8DType', 'UInt64DType', 'UInt32DType', 'UInt16DType',
    'UInt8DType', 'BoolDType', 'Complex128DType', 'Complex64DType']])
ck = torch.load(CKPT, map_location='cpu', weights_only=True)
print('checkpoint keys:', list(ck.keys()), '| epoch:', ck.get('epoch'),
      '| loss:', ck.get('loss'), '| mean_IOU:', ck.get('mean_IOU'))
sd = ck['state_dict']
print('state_dict entries:', len(sd))

model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block,
               num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256],
               deep_supervision=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)
print('state dict loads strictly into the official model')
model.eval()

mods = dict(model.named_modules())
convs = [(p, m) for p, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
print(f'Conv2d modules: {len(convs)}')
assert len(convs) == 154, len(convs)
k3 = sum(1 for _, m in convs if tuple(m.kernel_size) == (3, 3))
k1 = sum(1 for _, m in convs if tuple(m.kernel_size) == (1, 1))
k7 = sum(1 for _, m in convs if tuple(m.kernel_size) == (7, 7))
assert (k3, k1, k7) == (52, 76, 26), (k3, k1, k7)
print(f'  3x3: {k3}, 1x1: {k1}, 7x7: {k7} (spatial attention)')

def bn_sibling(path):
    """BatchNorm2d paired with this conv inside the same block, or None.
    conv1->bn1, conv2->bn2, shortcut.0->shortcut.1 (Res_CBAM_block)."""
    if path.endswith('.conv1'):
        cand = path[:-len('.conv1')] + '.bn1'
    elif path.endswith('.conv2'):
        cand = path[:-len('.conv2')] + '.bn2'
    elif path.endswith('.shortcut.0'):
        cand = path[:-len('.shortcut.0')] + '.shortcut.1'
    else:
        return None
    m = mods.get(cand)
    return m if isinstance(m, torch.nn.BatchNorm2d) else None

npz, paths, n_bn = {}, [], 0
for i, (path, c) in enumerate(convs):
    W = c.weight.detach().numpy().astype(np.float32)      # (Cout,Cin,H,W)
    W = np.transpose(W, (2, 3, 1, 0))                     # -> (H,W,Cin,Cout)
    b = c.bias.detach().numpy().astype(np.float32) if c.bias is not None \
        else np.zeros(W.shape[3], dtype=np.float32)
    npz[f'L{i}_W'] = W
    npz[f'L{i}_b'] = b
    npz[f'L{i}_path'] = np.array([path])
    npz[f'L{i}_pad'] = np.array(list(c.padding), dtype=np.int64)
    npz[f'L{i}_stride'] = np.array(list(c.stride), dtype=np.int64)
    bn = bn_sibling(path)
    npz[f'L{i}_has_bn'] = np.array([bn is not None])
    if bn is not None:
        n_bn += 1
        npz[f'L{i}_bn_w'] = bn.weight.detach().numpy().astype(np.float32)
        npz[f'L{i}_bn_b'] = bn.bias.detach().numpy().astype(np.float32)
        npz[f'L{i}_bn_mean'] = bn.running_mean.detach().numpy().astype(np.float32)
        npz[f'L{i}_bn_var'] = bn.running_var.detach().numpy().astype(np.float32)
        npz[f'L{i}_bn_eps'] = np.array([bn.eps], dtype=np.float32)
    paths.append(path)
    if i < 2 or i >= len(convs) - 2:
        print(f"  L{i:3d} {path:28s} W {W.shape}  b {b.shape}  "
              f"W range [{W.min():+.4f},{W.max():+.4f}]  BN={bn is not None}")
print(f'BatchNorm2d paired: {n_bn}')
assert n_bn == 68, n_bn
npz['paths'] = np.array(paths)
np.savez('dnanet_float.npz', **npz)
print('saved dnanet_float.npz')
print('total conv params:', sum(int(c.weight.numel()) for _, c in convs))
