#!/usr/bin/env python3
"""Torch model definitions for the three Stage 6 models, built from the
float weights in models/vision/ (commit 2daf4e4). Used by the compiler capture
frontend (torch.export) -- the graph structure is what the compiler
consumes; the weights are carried in the exported state_dict.

Weight layout on disk (models/vision/common.py): (KH, KW, C_in, C_out).
Torch needs (C_out, C_in, KH, KW) -- transposed here.
"""
import os
import numpy as np
import torch.nn as nn


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _load_npz(rel):
    return np.load(os.path.join(_repo_root(), "models", "vision", rel))


def _to_torch_weight(W_hwco):
    """(KH, KW, Cin, Cout) -> torch (Cout, Cin, KH, KW)."""
    return np.transpose(np.asarray(W_hwco, dtype=np.float32), (3, 2, 0, 1))


class SRCNN(nn.Module):
    """SRCNN 9-1-5 x3 (Dong et al. 2014): L1 9x9 1->64 + ReLU,
    L2 1x1 64->32 + ReLU, L3 5x5 32->1. Valid conv (no padding)."""

    def __init__(self):
        super().__init__()
        import torch
        import torch.nn as nn
        d = _load_npz("srcnn/srcnn_x3_float.npz")
        self.torch = torch
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, 5, bias=True),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(torch.from_numpy(_to_torch_weight(d["L1_W"])))
            self.net[0].bias.copy_(torch.from_numpy(d["L1_b"].astype(np.float32)))
            self.net[2].weight.copy_(torch.from_numpy(_to_torch_weight(d["L2_W"])))
            self.net[2].bias.copy_(torch.from_numpy(d["L2_b"].astype(np.float32)))
            self.net[4].weight.copy_(torch.from_numpy(_to_torch_weight(d["L3_W"])))
            self.net[4].bias.copy_(torch.from_numpy(d["L3_b"].astype(np.float32)))

    def forward(self, x):
        return self.net(x)


class DnCNNFragment(nn.Module):
    """DnCNN sigma=25, first ``n_layers`` convs (3x3, 64ch, ReLU after
    each except the last). The released .mat has NO BatchNorm, so the
    graph is conv+ReLU only. ``n_layers`` <= 17."""

    def __init__(self, n_layers=3):
        super().__init__()
        import torch
        import torch.nn as nn
        assert 1 <= n_layers <= 17
        d = _load_npz("dncnn/dncnn_sigma25_float.npz")
        self.torch = torch
        layers = []
        for i in range(1, n_layers + 1):
            cin = 1 if i == 1 else 64
            cout = 1 if i == 17 else 64
            conv = nn.Conv2d(cin, cout, 3, padding=1, bias=True)
            with torch.no_grad():
                conv.weight.copy_(torch.from_numpy(
                    _to_torch_weight(d[f"L{i}_W"])))
                conv.bias.copy_(torch.from_numpy(
                    d[f"L{i}_b"].astype(np.float32)))
            layers.append(conv)
            if i < 17:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RetinexNetDecom(nn.Module):
    """RetinexNet Decom-Net: conv0 9x9 4->64 (no activation, signed
    output), conv1-5 3x3 64->64 + ReLU, recon 3x3 64->4 -> sigmoid
    (R, L). The golden uses REPLICATE padding; the torch model pads
    explicitly with ReplicationPad2d so the capture sees the pad as a
    graph node the fusion pass folds into the conv's pad_mode."""

    def __init__(self):
        super().__init__()
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        d = _load_npz("retinexnet/retinexnet_decom_float.npz")
        self.torch = torch
        self.F = F
        self.conv0 = nn.Conv2d(4, 64, 9, bias=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(64, 64, 3, bias=True) for _ in range(5)])
        self.recon = nn.Conv2d(64, 4, 3, bias=True)
        with torch.no_grad():
            # float npz keys: L1=conv0 (9x9 4->64), L2..L6=conv1..5,
            # L7=recon (3x3 64->4). (The quant json uses conv0..recon.)
            self.conv0.weight.copy_(torch.from_numpy(
                _to_torch_weight(d["L1_W"])))
            self.conv0.bias.copy_(torch.from_numpy(
                d["L1_b"].astype(np.float32)))
            for i, c in enumerate(self.convs, start=1):
                c.weight.copy_(torch.from_numpy(
                    _to_torch_weight(d[f"L{i + 1}_W"])))
                c.bias.copy_(torch.from_numpy(
                    d[f"L{i + 1}_b"].astype(np.float32)))
            self.recon.weight.copy_(torch.from_numpy(
                _to_torch_weight(d["L7_W"])))
            self.recon.bias.copy_(torch.from_numpy(
                d["L7_b"].astype(np.float32)))

    def forward(self, x):
        F = self.F
        # 9x9 conv needs pad 4; 3x3 convs need pad 1 (replicate).
        x = self.conv0(F.pad(x, (4, 4, 4, 4), mode="replicate"))
        for c in self.convs:
            x = F.relu(c(F.pad(x, (1, 1, 1, 1), mode="replicate")))
        x = self.recon(F.pad(x, (1, 1, 1, 1), mode="replicate"))
        return self.torch.sigmoid(x)


def make_model(name, **kw):
    if name == "srcnn":
        return SRCNN()
    if name == "dncnn":
        return DnCNNFragment(**kw)
    if name == "retinexnet":
        return RetinexNetDecom()
    raise ValueError(f"unknown model {name}")
