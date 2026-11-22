#!/usr/bin/env python3
"""Unit tests: fusion pass."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "common"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from ir import build_graph
from passes.fusion import fuse


def _conv(name, K=3, C_in=1, C_out=8, pad=(1, 1)):
    return {"op": "conv2d", "name": name,
            "kwargs": {"K": K, "C_in": C_in, "C_out": C_out, "pad": pad},
            "params": {"weight": np.zeros((K, K, C_in, C_out), dtype=np.float32),
                       "bias": np.zeros(C_out, dtype=np.float32)}}


class TestFusion(unittest.TestCase):
    def test_conv_relu_fused(self):
        g = build_graph("t", [_conv("c1"), {"op": "relu", "name": "r1"},
                              _conv("c2", K=1, C_in=8, C_out=1),
                              {"op": "relu", "name": "r2"}])
        fg, groups, report = fuse(g)
        ops = [n.op for n in fg.nodes]
        self.assertEqual(ops, ["conv2d", "conv2d"])
        self.assertEqual(fg["c1"].kwargs["fused_act"], "relu")
        self.assertEqual(fg["c2"].kwargs["fused_act"], "relu")
        self.assertTrue(any("fuse relu" in r for r in report))

    def test_pad_folded_into_conv(self):
        g = build_graph("t", [
            {"op": "input", "name": "inp"},
            {"op": "pad", "name": "p1",
             "kwargs": {"pad": (4, 4, 4, 4), "mode": "replicate"}},
            _conv("c1", K=9, pad=(0, 0)), {"op": "relu", "name": "r1"}])
        fg, _, _ = fuse(g)
        self.assertEqual([n.op for n in fg.nodes], ["input", "conv2d"])
        self.assertEqual(fg["c1"].kwargs["pad"], (4, 4, 4, 4))
        self.assertEqual(fg["c1"].kwargs.get("pad_mode"), "replicate")
        self.assertEqual(fg["c1"].kwargs["fused_act"], "relu")

    def test_complex_act_separate_node(self):
        g = build_graph("t", [_conv("c1"), {"op": "sigmoid", "name": "s1"}])
        fg, _, _ = fuse(g)
        self.assertEqual([n.op for n in fg.nodes], ["conv2d", "act"])
        self.assertEqual(fg["s1"].kwargs["func"], "sigmoid")
        self.assertEqual(fg["s1"].args, ["c1"])
        self.assertIsNone(fg["c1"].kwargs.get("fused_act"))

    def test_residual_cuts_group(self):
        g = build_graph("t", [_conv("c1"), {"op": "relu", "name": "r1"},
                              {"op": "add", "name": "a1", "args": ["c1", "c1"]},
                              _conv("c2", K=1, C_in=8, C_out=1)])
        fg, groups, _ = fuse(g)
        self.assertEqual([n.op for n in fg.nodes],
                         ["conv2d", "residual", "conv2d"])
        self.assertEqual(fg["a1"].kwargs["mode"], "add")
        # The residual is its own fusion group (it needs both inputs
        # on-chip together); the convs are separate groups.
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[1], ["a1"])

    def test_bn_folded(self):
        # Standard pattern: conv -> BN -> relu. BN folds into the
        # PRECEDING conv (per-output-channel scaling), and the ReLU
        # fuses into the conv output stage. Verified numerically
        # against the manual BN formula (max|diff| ~5e-8, float32).
        from funcsim import float_forward
        from models.vision.common import conv2d_float
        rng = np.random.default_rng(5)
        W = rng.normal(0, 0.5, size=(3, 3, 2, 4)).astype(np.float32)
        b = rng.normal(0, 0.5, size=4).astype(np.float32)
        gamma = rng.normal(1, 0.2, size=4).astype(np.float32)
        beta = rng.normal(0, 0.2, size=4).astype(np.float32)
        mean = rng.normal(0, 0.5, size=4).astype(np.float32)
        var = np.abs(rng.normal(1, 0.3, size=4)).astype(np.float32)
        g = build_graph("t", [
            {"op": "input", "name": "inp"},
            {"op": "conv2d", "name": "c1",
             "kwargs": {"K": 3, "C_in": 2, "C_out": 4, "pad": (1, 1, 1, 1)},
             "params": {"weight": W, "bias": b}},
            {"op": "batch_norm", "name": "b1", "kwargs": {"eps": 1e-5},
             "params": {"weight": gamma, "bias": beta,
                        "running_mean": mean, "running_var": var}},
            {"op": "relu", "name": "r1"},
        ])
        fg, _, report = fuse(g)
        self.assertEqual([n.op for n in fg.nodes], ["input", "conv2d"])
        self.assertEqual(fg["c1"].kwargs["fused_act"], "relu")
        self.assertTrue(any("fold batch_norm" in r for r in report))
        # Numerical check: fused conv == conv -> BN -> ReLU.
        x = rng.random((8, 8, 2))
        conv_out = conv2d_float(x, W, b, pad=(1, 1, 1, 1))
        bn_out = gamma * (conv_out - mean) / np.sqrt(var + 1e-5) + beta
        ref = np.maximum(bn_out, 0)
        _, out = float_forward(fg, x)
        self.assertLess(float(np.max(np.abs(out - ref))), 1e-6)


if __name__ == "__main__":
    unittest.main()
