#!/usr/bin/env python3
"""Extract DnCNN sigma=25 weights from the official MatConvNet .mat (v7.3/HDF5).
MATLAB stores dims reversed in HDF5; h5py shape (Cout, Cin, H, W) -> transpose to (H, W, Cin, Cout).
Saves float32 weights + biases in a clean dict and prints a summary."""
import h5py, numpy as np, json, os

SRC = 'raw/sigma25.mat'
f = h5py.File(SRC, 'r')
layers = f['net']['layers']
n = layers.shape[0]

convs = []  # list of dicts {W:(H,W,Cin,Cout), b:(Cout,)}
for i in range(n):
    grp = f[layers[i,0]]
    t = ''.join(chr(c) for c in grp['type'][()].flatten())
    if t != 'conv':
        continue
    w = grp['weights']
    Wref = f[w[0,0]]  # (Cout, Cin, H, W) as stored
    bref = f[w[1,0]]  # (1, Cout)
    W = np.array(Wref, dtype=np.float32)
    b = np.array(bref, dtype=np.float32).reshape(-1)
    if W.ndim == 4:
        # stored (Cout, Cin, H, W) -> (H, W, Cin, Cout)
        W = np.transpose(W, (2, 3, 1, 0))
    elif W.ndim == 3:
        # last layer stored (Cin, H, W) -> (H, W, Cin, 1)
        W = np.transpose(W, (2, 1, 0)).reshape(W.shape[2], W.shape[1], W.shape[0], 1)
    else:
        raise ValueError(f"unexpected weight ndim {W.ndim}")
    convs.append({'W': W, 'b': b})

print(f"conv layers: {len(convs)}")
for i, c in enumerate(convs):
    W, b = c['W'], c['b']
    print(f"  L{i+1:2d}: W {W.shape}  b {b.shape}  W range [{W.min():+.4f},{W.max():+.4f}]  "
          f"|W|mean {np.abs(W).mean():.4f}  b range [{b.min():+.4f},{b.max():+.4f}]")

# sanity: expected DnCNN-S shapes
assert len(convs) == 17, len(convs)
assert convs[0]['W'].shape == (3,3,1,64)
assert all(convs[i]['W'].shape == (3,3,64,64) for i in range(1,16))
assert convs[16]['W'].shape == (3,3,64,1)
assert convs[0]['b'].shape == (64,) and convs[16]['b'].shape == (1,)
print("SHAPES OK: 17 conv layers, 1->64, 15x 64->64, 64->1, all 3x3; no BN layers in this .mat")

np.savez('dncnn_sigma25_float.npz',
         **{f'L{i+1}_W': c['W'] for i, c in enumerate(convs)},
         **{f'L{i+1}_b': c['b'] for i, c in enumerate(convs)})
print("saved dncnn_sigma25_float.npz")
