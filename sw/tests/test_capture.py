#!/usr/bin/env python3
"""Unit tests: capture (torch.export frontend) + hand-built IR."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch
import models_torch
from ir import capture_torch, build_graph, Graph, Node


class TestCaptureTorch(unittest.TestCase):
    def test_srcnn(self):
        m = models_torch.SRCNN()
        g, state = capture_torch(m, torch.randn(1, 1, 33, 33), "srcnn")
        ops = [n.op for n in g.nodes]
        self.assertEqual(ops, ["input", "conv2d", "relu", "conv2d",
                               "relu", "conv2d"])
        convs = [n for n in g.nodes if n.op == "conv2d"]
        self.assertEqual([(c.kwargs["K"], c.kwargs["C_in"],
                           c.kwargs["C_out"]) for c in convs],
                         [(9, 1, 64), (1, 64, 32), (5, 32, 1)])
        # weights are the real pretrained weights (spot-check L1)
        w = convs[0].params["weight"]
        self.assertEqual(w.shape, (9, 9, 1, 64))
        self.assertAlmostEqual(float(np.abs(w).max()), 0.9168, places=3)

    def test_dncnn_fragment(self):
        m = models_torch.DnCNNFragment(2)
        g, _ = capture_torch(m, torch.randn(1, 1, 16, 16), "dncnn2")
        ops = [n.op for n in g.nodes]
        self.assertEqual(ops, ["input", "conv2d", "relu", "conv2d", "relu"])
        convs = [n for n in g.nodes if n.op == "conv2d"]
        self.assertEqual(convs[0].kwargs["pad"], (1, 1))
        self.assertEqual(convs[1].kwargs["C_in"], 64)

    def test_retinexnet(self):
        m = models_torch.RetinexNetDecom()
        g, _ = capture_torch(m, torch.randn(1, 4, 16, 16), "retinexnet")
        ops = [n.op for n in g.nodes]
        self.assertEqual(ops.count("pad"), 7)
        self.assertEqual(ops.count("conv2d"), 7)
        self.assertIn("sigmoid", ops)
        pads = [n for n in g.nodes if n.op == "pad"]
        self.assertEqual(pads[0].kwargs["pad"], (4, 4, 4, 4))
        self.assertEqual(pads[0].kwargs["mode"], "replicate")

    def test_unsupported_op_raises(self):
        # A model using an op with no lowering must fail loudly,
        # not silently fall back to a hand-built graph.
        class Bad(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.softmax(x, dim=1)
        with self.assertRaises(RuntimeError):
            capture_torch(Bad(), torch.randn(1, 4), "bad")


class TestBuildGraph(unittest.TestCase):
    def test_linear_chain(self):
        g = build_graph("t", [
            {"op": "conv2d", "name": "c1",
             "kwargs": {"K": 3, "C_in": 1, "C_out": 8},
             "params": {"weight": np.zeros((3, 3, 1, 8), dtype=np.float32),
                        "bias": np.zeros(8, dtype=np.float32)}},
            {"op": "relu", "name": "r1"},
            {"op": "conv2d", "name": "c2",
             "kwargs": {"K": 1, "C_in": 8, "C_out": 1},
             "params": {"weight": np.zeros((1, 1, 8, 1), dtype=np.float32),
                        "bias": np.zeros(1, dtype=np.float32)}},
        ])
        self.assertEqual([n.op for n in g.nodes],
                         ["conv2d", "relu", "conv2d"])
        self.assertEqual(g["r1"].args, ["c1"])
        self.assertEqual(g.outputs, ["c2"])


if __name__ == "__main__":
    unittest.main()
