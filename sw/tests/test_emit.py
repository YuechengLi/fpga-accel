#!/usr/bin/env python3
"""Unit tests: emission + quantization pass."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch
import models_torch
from ir import capture_torch
from compile import compile_model
from emit import emit_program, write_program
from passes.quantize import quantize_graph
from passes.fusion import fuse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestEmit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = models_torch.SRCNN()
        g, _ = capture_torch(model, torch.randn(1, 1, 33, 33), "srcnn")
        x = np.random.default_rng(3).random((33, 33, 1))
        cls.program, cls.instrs, cls.blob, cls.report = compile_model(
            g, x, (33, 33, 1), model_name="srcnn")

    def test_program_structure(self):
        ops = self.program["ops"]
        self.assertEqual(len([o for o in ops if o["op"] == "conv2d"]), 3)
        self.assertIn("model", self.program)
        self.assertEqual(self.program["model"], "srcnn")
        self.assertGreater(self.report["instr_count"], 0)
        self.assertEqual(self.report["weight_blob_bytes"], len(self.blob))

    def test_blob_size_matches_weights(self):
        # SRCNN weights, one full 121-entry 11x11 grid per (o, c)
        # lane (Stage 6 fix: the MC's coefficient memory is indexed
        # by the grid, so the blob carries the grid, not just the
        # K*K taps). Each (o, c) pair is exactly one lane, so the
        # blob is sum(C_in * C_out * 121) int16 coeffs.
        ncoeff = 0
        for o in self.program["ops"]:
            if o["op"] in ("conv2d", "dw_conv2d", "linear"):
                ncoeff += o["C_in"] * o["C_out"] * 121
        self.assertEqual(len(self.blob), ncoeff * 2)

    def test_instr_stream_opcodes(self):
        # The instruction stream uses W (config write), DMA (weight
        # load), GO (start engine), WAIT, STREAM (pixel stream).
        # GO is emitted once per (o, t) tile pass, not once per layer
        # (SRCNN: L1 64 + L2 512 + L3 8 = 584 MC passes, plus the
        # scheduler-path passes). The invariant is: every pass has
        # exactly one DMA, one GO and one WAIT.
        joined = "\n".join(self.instrs)
        for op in ("W ", "DMA", "GO", "WAIT", "STREAM"):
            self.assertIn(op, joined, op)
        n_dma = sum(1 for l in self.instrs if l.startswith("DMA"))
        n_go = sum(1 for l in self.instrs if l == "GO")
        n_wait = sum(1 for l in self.instrs if l == "WAIT")
        self.assertEqual(n_go, n_dma)
        self.assertEqual(n_wait, n_dma)
        self.assertGreater(n_go, 3)

    def test_write_program_files(self):
        with tempfile.TemporaryDirectory() as d:
            jp, tp, bp = write_program(self.program, self.instrs,
                                       self.blob, d)
            self.assertTrue(os.path.exists(jp))
            self.assertTrue(os.path.exists(tp))
            self.assertTrue(os.path.exists(bp))
            self.assertEqual(os.path.getsize(bp), len(self.blob))
            pub = json.load(open(jp))
            self.assertEqual(pub["model"], "srcnn")
            # program.json must not contain numpy arrays (they are in
            # weights.bin).
            self.assertNotIn("_weights_arrays", pub)


class TestQuantize(unittest.TestCase):
    def test_srcnn_shifts_in_range(self):
        model = models_torch.SRCNN()
        g, _ = capture_torch(model, torch.randn(1, 1, 33, 33), "srcnn")
        fg, _, _ = fuse(g)
        x = np.random.default_rng(4).random((33, 33, 1))
        qg, report = quantize_graph(fg, x)
        # Each conv gets shift/r_in/r_out/e, shift in [0,15].
        for n in qg.nodes:
            if n.op == "conv2d":
                self.assertIn("shift", n.kwargs)
                self.assertIn("r_in", n.kwargs)
                self.assertIn("r_out", n.kwargs)
                self.assertGreaterEqual(n.kwargs["shift"], 0)
                self.assertLessEqual(n.kwargs["shift"], 15)
                # Quantized weights are int16.
                self.assertEqual(n.params["W_q"].dtype, np.int16)
        # The report lists per-layer shift/r_in/r_out/e.
        self.assertEqual(len(report), 3)

    def test_signed_intermediate(self):
        # A conv with no fused act and a negative float output gets
        # signed_out=True (RetinexNet conv0 is the real case: ~50%
        # negative). Here a hand-built conv with a negative bias.
        from ir import build_graph
        W = np.ones((1, 1, 1, 1), dtype=np.float32)
        g = build_graph("s", [
            {"op": "input", "name": "inp"},
            {"op": "conv2d", "name": "c1",
             "kwargs": {"K": 1, "C_in": 1, "C_out": 1, "pad": (0, 0, 0, 0)},
             "params": {"weight": W,
                        "bias": np.array([-5.0], dtype=np.float32)}},
        ])
        fg, _, _ = fuse(g)
        x = np.zeros((4, 4, 1))
        qg, _ = quantize_graph(fg, x)
        n = qg["c1"]
        self.assertTrue(n.kwargs.get("signed_out", False))


if __name__ == "__main__":
    unittest.main()
