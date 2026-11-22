#!/usr/bin/env python3
"""Unit tests: Stage 7b streaming emission (sw/common/emit.py).

The streaming mode emits ONE fused program for the whole SRCNN chain
for the 3-stage pipeline (rtl/vision/srcnn_pipe.sv): weights preloaded once,
one STREAM per frame, no per-layer GO/WAIT. The op list (what the
functional simulator executes) is identical to the iterative program;
only the weight blob layout and the instruction stream differ.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "common"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import torch
import models_torch
from ir import capture_torch
from compile import compile_model
from emit import _assign_streaming_engines
from funcsim import fixed_forward
from demo_srcnn import build_srcnn_input


def _mk_layer(name, K, cin, cout, H, W, H_out, W_out, shift=15):
    """Fake (op dict, node kwargs) for _assign_streaming_engines."""
    op = {"name": name, "K": K, "C_in": cin, "C_out": cout,
          "pad": [0, 0, 0, 0], "stride": [1, 1], "shift": shift,
          "sign": 0, "fused_act": "relu", "act_slope": 0.01,
          "bias": [0] * cout}
    kw = {"H": H, "W": W, "H_out": H_out, "W_out": W_out}
    return (op, kw)


class TestStreamingEmit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = models_torch.SRCNN()
        g, _ = capture_torch(model, torch.randn(1, 1, 33, 33), "srcnn")
        x = np.random.default_rng(3).random((33, 33, 1))
        cls.iter_program, cls.iter_instrs, cls.iter_blob, _ = \
            compile_model(g, x, (33, 33, 1), model_name="srcnn")
        g2, _ = capture_torch(model, torch.randn(1, 1, 33, 33), "srcnn")
        cls.program, cls.instrs, cls.blob, cls.report = compile_model(
            g2, x, (33, 33, 1), model_name="srcnn", streaming=True)
        cls.inp_u8, _ = build_srcnn_input()

    def test_engine_assignment(self):
        # Layer -> engine assignment: L1 outpar, L2 1x1, L3 tappar,
        # with the pipe bases, DMA plan and weight offsets.
        eng = self.program["streaming"]["engines"]
        self.assertEqual(len(eng), 3)
        self.assertEqual([e["engine"] for e in eng],
                         ["conv2d_outpar", "conv1x1_cp", "conv2d_tappar"])
        self.assertEqual([e["pipe_base"] for e in eng],
                         [0x0000, 0x4000, 0x8000])
        self.assertEqual([(e["K"], e["C_in"], e["C_out"]) for e in eng],
                         [(9, 1, 64), (1, 64, 32), (5, 32, 1)])
        self.assertEqual([(e["dma_dst"], e["dma_taps"], e["dma_stride"])
                          for e in eng],
                         [(0x0100, 81, 128), (0x4100, 64, 64),
                          (0x8100, 800, 1)])
        self.assertEqual([e["weight_offset"] for e in eng],
                         [0, 5184, 7232])
        self.assertEqual([e["ncoeff"] for e in eng], [5184, 2048, 800])
        # Geometry chains: 33 -> 25 -> 25 -> 21.
        self.assertEqual([(e["W"], e["H"], e["W_out"], e["H_out"])
                          for e in eng],
                         [(33, 33, 25, 25), (25, 25, 25, 25),
                          (25, 25, 21, 21)])

    def test_program_structure(self):
        # One weight-load phase (3 DMAs), then GO, then one STREAM.
        # No per-layer GO/WAIT serialization.
        kinds = [l.split()[0] for l in self.instrs
                 if l and not l.startswith("#")]
        self.assertEqual(kinds.count("DMA"), 3)
        self.assertEqual(kinds.count("GO"), 0)
        self.assertEqual(kinds.count("WAIT"), 0)
        self.assertEqual(kinds.count("STREAM"), 1)
        # GO is a plain config write to L1 0x000C and L3 0x800C (L2
        # has no GO); both come after the last DMA and before STREAM.
        go_idx = [i for i, l in enumerate(self.instrs)
                  if l.startswith("W 0x000c") or l.startswith("W 0x800c")]
        self.assertEqual(len(go_idx), 2)
        dma_idx = [i for i, l in enumerate(self.instrs)
                   if l.startswith("DMA")]
        stream_idx = [i for i, l in enumerate(self.instrs)
                      if l.startswith("STREAM")]
        self.assertEqual(len(stream_idx), 1)
        self.assertTrue(max(dma_idx) < min(go_idx) < stream_idx[0])
        # Blob: 5184 + 2048 + 800 int16 coefficients, nothing else.
        self.assertEqual(len(self.blob), 8032 * 2)
        self.assertEqual(self.program["instr_count"], len(self.instrs))
        self.assertTrue(self.program["streaming"]["enabled"])

    def test_blob_matches_weights(self):
        # The streaming blob carries the same weights as the iterative
        # program, in each engine's coefficient order.
        arr = np.frombuffer(self.blob, dtype="<i2")
        w = self.program["_weights_arrays"]
        names = [o["name"] for o in self.program["ops"]
                 if o["op"] == "conv2d"]
        l1, l2, l3 = (w[n]["W_q"] for n in names)
        g1 = arr[0:5184].reshape(64, 81)
        for o in range(64):
            for r in range(9):
                self.assertTrue(
                    np.array_equal(g1[o, r * 9:r * 9 + 9], l1[r, :, 0, o]),
                    f"L1 o={o} r={r}")
        g2 = arr[5184:7232].reshape(32, 64)
        for o in range(32):
            self.assertTrue(np.array_equal(g2[o, :], l2[0, 0, :, o]),
                            f"L2 o={o}")
        g3 = arr[7232:8032].reshape(25, 32)
        for t in range(5):
            for j in range(5):
                self.assertTrue(
                    np.array_equal(g3[t * 5 + j, :], l3[t, j, :, 0]),
                    f"L3 t={t} j={j}")

    def test_funcsim_matches_iterative(self):
        # The streaming program executes identically to the iterative
        # program (same ops, same weights): max|err| = 0.
        _, out_s = fixed_forward(self.program, self.inp_u8)
        _, out_i = fixed_forward(self.iter_program, self.inp_u8)
        self.assertEqual(out_s.shape, out_i.shape)
        self.assertEqual(int(np.abs(out_s.astype(np.int32)
                                    - out_i.astype(np.int32)).max()), 0)

    def test_assignment_rejects(self):
        # Wrong layer count.
        with self.assertRaises(ValueError):
            _assign_streaming_engines([
                _mk_layer("a", 9, 1, 64, 33, 33, 25, 25),
                _mk_layer("b", 1, 64, 32, 25, 25, 25, 25)])
        # L2 must be exactly K=1, C_in=64, C_out=32 (compile-time).
        with self.assertRaises(ValueError):
            _assign_streaming_engines([
                _mk_layer("a", 9, 1, 64, 33, 33, 25, 25),
                _mk_layer("b", 1, 64, 16, 25, 25, 25, 25),
                _mk_layer("c", 5, 32, 1, 25, 25, 21, 21)])
        # L3 K=3 is excluded (known RTL column-alignment limitation).
        with self.assertRaises(ValueError):
            _assign_streaming_engines([
                _mk_layer("a", 9, 1, 64, 33, 33, 25, 25),
                _mk_layer("b", 1, 64, 32, 25, 25, 25, 25),
                _mk_layer("c", 3, 32, 1, 25, 25, 23, 23)])
        # Broken chaining (L1 out != L2 in).
        with self.assertRaises(ValueError):
            _assign_streaming_engines([
                _mk_layer("a", 9, 1, 64, 33, 33, 25, 25),
                _mk_layer("b", 1, 64, 32, 24, 24, 24, 24),
                _mk_layer("c", 5, 32, 1, 24, 24, 20, 20)])
        # L1 C_out not a multiple of 16.
        with self.assertRaises(ValueError):
            _assign_streaming_engines([
                _mk_layer("a", 9, 1, 17, 33, 33, 25, 25),
                _mk_layer("b", 1, 64, 32, 25, 25, 25, 25),
                _mk_layer("c", 5, 32, 1, 25, 25, 21, 21)])

    def test_iterative_unchanged(self):
        # Streaming is additive: the iterative program has no
        # streaming key and its blob is the 121-entry-grid layout.
        self.assertNotIn("streaming", self.iter_program)
        ncoeff = 0
        for o in self.iter_program["ops"]:
            if o["op"] in ("conv2d", "dw_conv2d", "linear"):
                ncoeff += o["C_in"] * o["C_out"] * 121
        self.assertEqual(len(self.iter_blob), ncoeff * 2)


if __name__ == "__main__":
    unittest.main()
