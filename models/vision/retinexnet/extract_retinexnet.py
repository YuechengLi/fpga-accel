#!/usr/bin/env python3
"""Extract RetinexNet DecomNet weights from the PyTorch-port checkpoint
(aasharma90/RetinexNet_PyTorch, ckpts/Decom/9200.tar, torch.save state_dict).
PyTorch Conv2d weight is (Cout, Cin, H, W); transpose to (H, W, Cin, Cout).
DecomNet forward (model.py): input RGB -> [max,R,G,B] (4ch) -> conv0 9x9 (no act)
-> 5x (conv 3x3 + ReLU) -> recon 3x3 -> sigmoid -> R(3ch), L(1ch).
Padding mode is 'replicate' (NOT zero-fill) in the port."""
import torch, numpy as np

sd = torch.load('raw/decom_9200.tar', map_location='cpu', weights_only=False)
# Ordered layer list
order = [
    ('L1', 'net1_conv0'),
    ('L2', 'net1_convs.0'),
    ('L3', 'net1_convs.2'),
    ('L4', 'net1_convs.4'),
    ('L5', 'net1_convs.6'),
    ('L6', 'net1_convs.8'),
    ('L7', 'net1_recon'),
]
out = {}
for name, key in order:
    W = sd[key + '.weight'].numpy().astype(np.float32)  # (Cout, Cin, H, W)
    b = sd[key + '.bias'].numpy().astype(np.float32)
    W = np.transpose(W, (2, 3, 1, 0))  # (H, W, Cin, Cout)
    out[name + '_W'] = W
    out[name + '_b'] = b
    print(f"  {name}: W {W.shape} range [{W.min():+.4f},{W.max():+.4f}] |W|mean {np.abs(W).mean():.4f}  "
          f"b {b.shape} range [{b.min():+.4f},{b.max():+.4f}]")

assert out['L1_W'].shape == (9, 9, 4, 64)
assert all(out[f'L{i}_W'].shape == (3, 3, 64, 64) for i in range(2, 7))
assert out['L7_W'].shape == (3, 3, 64, 4)
print("SHAPES OK: conv0 9x9 4->64, 5x conv 3x3 64->64, recon 3x3 64->4; input [max,R,G,B], replicate padding, sigmoid split R(3)/L(1)")
np.savez('retinexnet_decom_float.npz', **out)
print("saved retinexnet_decom_float.npz")
