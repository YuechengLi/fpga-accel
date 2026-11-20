#!/usr/bin/env python3
"""Unit tests: functional simulator (the Stage 6 reference)."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import torch
import models_torch
from ir import capture_torch, build_graph
from compile import compile_model
from funcsim import (conv2d_tiled, conv2d_fixed_rhu, fixed_forward,
                     float_forward)
from passes.mlut import mlut_search, mlut_eval_float
from models.common import (conv2d_float, psnr, max_abs_err,
                           make_test_gray)
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # repo root


def _srcnn_input():
    """The demo's test image: HR 33x33 -> LR 11x11 -> bicubic 33x33."""
    hr = make_test_gray(33, 33, seed=7)
    lr = np.array(Image.fromarray(hr).resize((11, 11), Image.BICUBIC))
    inp = np.array(Image.fromarray(lr).resize((33, 33), Image.BICUBIC))
    return inp[:, :, None], (inp.astype(np.float64) / 255.0)[:, :, None]


def _golden_srcnn_rhu(inp_u8):
    """Fixed-point golden with ROUND-HALF-UP shift (the RTL's shift),
    from models/srcnn/srcnn_x3.npz + quant.json. The functional
    simulator must match this EXACTLY (tiling does not change the
    integer sum)."""
    root = os.path.join(ROOT, "models", "srcnn")
    d = np.load(os.path.join(root, "srcnn_x3.npz"))
    q = json.load(open(os.path.join(root, "srcnn_x3_quant.json")))
    A = inp_u8
    outs = []
    for name in ("L1", "L2", "L3"):
        A = conv2d_fixed_rhu(A, d[name + "_W_q"], d[name + "_b_q"],
                             q[name]["shift"], pad=(0, 0, 0, 0))
        outs.append(A)
    return outs, A


class TestTiling(unittest.TestCase):
    def test_tiled_equals_untiled_rhu(self):
        rng = np.random.default_rng(0)
        A = rng.integers(0, 256, size=(8, 8, 8), dtype=np.uint8)
        W = rng.integers(-200, 200, size=(3, 3, 8, 4)).astype(np.int16)
        b = rng.integers(-1000, 1000, size=(4,)).astype(np.int32)
        got = conv2d_tiled(A, W, b, shift=6, pad=(1, 1, 1, 1),
                           C_in_tile=2, shift_mode="rhu")
        ref = conv2d_fixed_rhu(A, W, b, shift=6, pad=(1, 1, 1, 1))
        self.assertEqual(max_abs_err(got, ref), 0)

    def test_tiled_signed(self):
        rng = np.random.default_rng(1)
        A = rng.integers(-128, 128, size=(6, 6, 4)).astype(np.int64)
        W = rng.integers(-200, 200, size=(3, 3, 4, 2)).astype(np.int16)
        b = np.zeros(2, dtype=np.int32)
        got = conv2d_tiled(A, W, b, shift=6, pad=(1, 1, 1, 1),
                           signed_out=True, C_in_tile=2, shift_mode="rhu")
        ref = conv2d_fixed_rhu(A, W, b, shift=6, pad=(1, 1, 1, 1),
                               signed_out=True)
        self.assertEqual(max_abs_err(got, ref), 0)


class TestSRCNNEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inp_u8, inp_f = _srcnn_input()
        cls.inp_u8, cls.inp_f = inp_u8, inp_f
        model = models_torch.SRCNN()
        g, _ = capture_torch(model, torch.randn(1, 1, 33, 33), "srcnn")
        # The fused graph (pads normalized to 4-tuples, ReLUs fused)
        # is what float_forward runs on; the raw capture's pads are
        # 2-tuples and would fail pad2d.
        from passes.fusion import fuse
        fg, _, _ = fuse(g)
        cls.graph = fg
        cls.program, cls.instrs, cls.blob, cls.report = compile_model(
            g, inp_f, (33, 33, 1), model_name="srcnn")
        cls.per_layer, cls.final = fixed_forward(cls.program, inp_u8)
        cls.golden_outs, cls.golden_final = _golden_srcnn_rhu(inp_u8)

    def test_per_layer_exact_vs_rhu_golden(self):
        # The functional simulator (tiled, rhu shift) must match the
        # untiled rhu golden EXACTLY for all 3 layers.
        names = [op["name"] for op in self.program["ops"]
                 if op["op"] in ("conv2d", "dw_conv2d", "linear")]
        self.assertEqual(len(names), 3)
        for name, ref in zip(names, self.golden_outs):
            self.assertEqual(max_abs_err(self.per_layer[name], ref), 0,
                             name)

    def test_psnr_vs_float(self):
        # End-to-end PSNR vs the FLOAT model output (21x21 -- the
        # valid convs shrink 33x33 to 21x21), not vs the HR image.
        # Must match the Stage 6 prep number (42.76 dB) within the
        # shift-rounding tolerance.
        _, float_out = float_forward(self.graph, self.inp_f)
        p = psnr(self.final[:, :, 0].astype(np.float64),
                 np.clip(float_out[:, :, 0] * 255.0, 0, 255))
        self.assertGreater(p, 40.0)
        self.assertAlmostEqual(p, 42.55, delta=0.5)

    def test_instr_stream_present(self):
        self.assertGreater(len(self.instrs), 0)
        self.assertGreater(len(self.blob), 0)


class TestMLUTEndToEnd(unittest.TestCase):
    def test_gelu_uses_mlut_in_program(self):
        # GELU cannot use PWL (Q1.15 range), so the compiled program
        # must carry an MLUT table for it, and the functional
        # simulator must apply that table.
        rng = np.random.default_rng(2)
        W = rng.integers(-100, 100, size=(3, 3, 1, 1)).astype(np.float32)
        g = build_graph("gelu_t", [
            {"op": "input", "name": "inp"},
            {"op": "conv2d", "name": "c1",
             "kwargs": {"K": 3, "C_in": 1, "C_out": 1, "pad": (1, 1, 1, 1)},
             "params": {"weight": W,
                        "bias": np.zeros(1, dtype=np.float32)}},
            {"op": "gelu", "name": "a1"},
        ])
        x = rng.integers(0, 256, size=(8, 8, 1)).astype(np.uint8)
        xf = (x.astype(np.float64) / 255.0)
        program, instrs, blob, report = compile_model(
            g, xf, (8, 8, 1), model_name="gelu_t")
        # The act table for a1 must be an MLUT.
        tab = program["_act_tables_full"]["a1"]
        self.assertEqual(tab["impl"], "mlut")
        self.assertLess(tab["max_err"], 0.03)
        # Run the functional simulator and compare the act output
        # against a direct MLUT evaluation of the conv output.
        per_layer, final = fixed_forward(program, x)
        conv_out = per_layer["c1"]
        # The act input scale: r_in of the act op.
        act_op = [op for op in program["ops"] if op["name"] == "a1"][0]
        r_in = act_op.get("r_in", 0)
        r_out = act_op.get("r_out", 0)
        scale_in = 255.0 / (2.0 ** r_in)
        xf_in = conv_out.astype(np.float64) / scale_in
        h1, h2, base, sel, fine = (tab["H1"], tab["H2"], tab["base"],
                                   tab["sel"], tab["fine"])
        ref = np.array([[mlut_eval_float(v, h1, h2, base, sel, fine)
                         for v in row] for row in xf_in])
        # ref is (H, W); the act output is (H, W, 1) -- restore the
        # channel dim. The clip matches _apply_act: signed int8
        # [-128, 127] when signed_out, else uint8 [0, 255].
        scale_out = 255.0 / (2.0 ** r_out)
        if act_op.get("signed_out"):
            ref_q = np.clip(np.round(ref * scale_out),
                            -128, 127).astype(np.int64)[..., None]
        else:
            ref_q = np.clip(np.round(ref * scale_out),
                            0, 255).astype(np.uint8)[..., None]
        self.assertEqual(max_abs_err(per_layer["a1"], ref_q), 0)


if __name__ == "__main__":
    unittest.main()
