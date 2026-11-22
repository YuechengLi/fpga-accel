"""Unit tests for the integrated end-to-end driver (Stage 7c).

Covers, without any vvp run:
  - layers_from_program: the compiled-program -> scheduler layer
    adapter (act codes, dot mode for C_out==1, pad carried).
  - emit_pe_program with padded layers: a padded model's program
    has the right shapes and step counts, and its steps match a
    direct gen_layer_steps call on the pre-padded inputs.
  - Padding equivalence: the padded golden (conv2d_fixed_rhu on a
    pre-padded input) equals fixed_forward's per-layer golden --
    the cross-check the driver runs for every layer.
  - The frontend blur golden: a constant image blurs to itself
    (up to rounding) and keeps its shape (property of the 3x3
    Gaussian + replicate pad).
"""
import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "tb"))

import numpy as np

import emit
import golden_frontend
from funcsim import conv2d_fixed_rhu
from models.common import max_abs_err, pad2d
from run_e2e import FE_COEFF, layers_from_program
from sched_pe import gen_layer_steps
from stage6 import (build_dncnn_input, build_srcnn_input,
                    compile_and_sim)
import models_torch
from devices import get as get_device


class TestLayersFromProgram(unittest.TestCase):
    def test_srcnn_adapter(self):
        model = models_torch.SRCNN()
        inp_u8, _ = build_srcnn_input(24)
        program, per_layer, _out, _f = compile_and_sim(
            "srcnn", model, inp_u8.astype(np.float64) / 255.0,
            inp_u8, (24, 24, 1))
        layers = layers_from_program(program)
        self.assertEqual(len(layers), 3)
        names = [L["name"] for L in layers]
        self.assertEqual(names, ["conv2d", "conv2d_1", "conv2d_2"])
        L1, L2, L3 = layers
        self.assertEqual((L1["K"], L1["C_out"]), (9, 64))
        self.assertEqual((L2["K"], L2["C_out"]), (1, 32))
        self.assertEqual((L3["K"], L3["C_out"]), (5, 1))
        # act: relu->1, none->0.
        self.assertEqual((L1["act"], L2["act"], L3["act"]),
                         (1, 1, 0))
        # Dot mode only for the C_out==1 layer, tn=1 there.
        self.assertEqual((L1["mode_dot"], L2["mode_dot"],
                          L3["mode_dot"]), (0, 0, 1))
        self.assertEqual((L1["tn"], L3["tn"]), (8, 1))
        # No padding in SRCNN.
        for L in layers:
            self.assertEqual(L["pad"], (0, 0, 0, 0))
            self.assertEqual(L["pad_mode"], "zero")
        print("test_srcnn_adapter: PASS")

    def test_dncnn_adapter_carries_pad(self):
        model = models_torch.DnCNNFragment(n_layers=17)
        noisy_u8, _clean, noisy_f = build_dncnn_input(16)
        program, per_layer, _out, _f = compile_and_sim(
            "dncnn", model, noisy_f, noisy_u8, (16, 16, 1))
        layers = layers_from_program(program)
        self.assertEqual(len(layers), 17)
        for L in layers:
            self.assertEqual(L["pad"], (1, 1, 1, 1))
            self.assertEqual(L["pad_mode"], "zero")
        # L17 is the dot layer.
        self.assertEqual(layers[-1]["mode_dot"], 1)
        self.assertEqual(layers[-1]["tn"], 1)
        self.assertEqual(layers[-1]["act"], 0)
        self.assertEqual([L["act"] for L in layers[:-1]],
                         [1] * 16)
        print("test_dncnn_adapter_carries_pad: PASS")


class TestEmitPaddedProgram(unittest.TestCase):
    def test_dncnn_program_shapes_and_steps(self):
        model = models_torch.DnCNNFragment(n_layers=17)
        noisy_u8, _clean, noisy_f = build_dncnn_input(16)
        program, per_layer, _out, _f = compile_and_sim(
            "dncnn", model, noisy_f, noisy_u8, (16, 16, 1))
        layers = layers_from_program(program)
        pe_program, steps_by_layer = emit.emit_pe_program(
            layers, noisy_u8, "zu19eg")
        self.assertEqual(len(pe_program["layers"]), 17)
        # Same-size layers: every output stays 16x16.
        for pl in pe_program["layers"]:
            self.assertEqual(pl["out_shape"][:2], [16, 16])
        # Step counts: L1 128, L2-L16 2304 each, L17 288.
        counts = [pl["n_steps"] for pl in pe_program["layers"]]
        self.assertEqual(counts[0], 128)
        self.assertEqual(counts[1:-1], [2304] * 15)
        self.assertEqual(counts[-1], 288)
        self.assertEqual(sum(counts), 128 + 15 * 2304 + 288)
        # L17 dot, the rest outer.
        modes = [pl["mode_dot"] for pl in pe_program["layers"]]
        self.assertEqual(modes, [0] * 16 + [1])
        print("test_dncnn_program_shapes_and_steps: PASS")

    def test_dncnn_steps_match_padded_scheduler(self):
        model = models_torch.DnCNNFragment(n_layers=17)
        noisy_u8, _clean, noisy_f = build_dncnn_input(16)
        program, per_layer, _out, _f = compile_and_sim(
            "dncnn", model, noisy_f, noisy_u8, (16, 16, 1))
        layers = layers_from_program(program)
        _pe, steps_by_layer = emit.emit_pe_program(
            layers, noisy_u8, "zu19eg")
        dev = get_device("zu19eg")
        # Recompute L2 steps directly on the pre-padded input and
        # compare the first step field-by-field.
        cur = per_layer[layers[0]["name"]]  # L1 golden output
        L2 = layers[1]
        padded = pad2d(cur, L2["pad"], L2["pad_mode"])
        steps, _shape = gen_layer_steps(L2, padded, L2["W_q"],
                                        L2["b_q"], dev)
        self.assertEqual(len(steps), len(steps_by_layer[1][1]))
        a, b = steps[0], steps_by_layer[1][1][0]
        for k in ("mode_dot", "tk", "tn", "first_k", "sign_in",
                  "shift", "act", "slope", "sign_out", "mtile",
                  "nchunk", "n_kchunk"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a["A"], b["A"])
        self.assertEqual(a["B"], b["B"])
        self.assertEqual(a["bias"], b["bias"])
        print("test_dncnn_steps_match_padded_scheduler: PASS")

    def test_srcnn_emission_unchanged(self):
        # The pad extension must not change SRCNN emission at all.
        model = models_torch.SRCNN()
        inp_u8, _ = build_srcnn_input(24)
        program, per_layer, _out, _f = compile_and_sim(
            "srcnn", model, inp_u8.astype(np.float64) / 255.0,
            inp_u8, (24, 24, 1))
        layers = layers_from_program(program)
        pe_program, _steps = emit.emit_pe_program(
            layers, inp_u8, "zu19eg")
        counts = [pl["n_steps"] for pl in pe_program["layers"]]
        self.assertEqual(counts, [384, 128, 225])
        shapes = [pl["out_shape"] for pl in pe_program["layers"]]
        self.assertEqual(shapes,
                         [[16, 16, 64], [16, 16, 32], [12, 12, 1]])
        print("test_srcnn_emission_unchanged: PASS")


class TestPadGoldenEquivalence(unittest.TestCase):
    def test_dncnn_padded_golden_matches_fixed_forward(self):
        # The driver's per-layer cross-check, on one layer, in
        # Python: conv2d_fixed_rhu on the pre-padded input must
        # equal fixed_forward's per-layer golden exactly.
        model = models_torch.DnCNNFragment(n_layers=17)
        noisy_u8, _clean, noisy_f = build_dncnn_input(16)
        program, per_layer, _out, _f = compile_and_sim(
            "dncnn", model, noisy_f, noisy_u8, (16, 16, 1))
        layers = layers_from_program(program)
        L2 = layers[1]
        cur = per_layer[layers[0]["name"]]
        padded = pad2d(cur, L2["pad"], L2["pad_mode"])
        golden = conv2d_fixed_rhu(
            padded, L2["W_q"], L2["b_q"], L2["shift"],
            pad=(0, 0, 0, 0), signed_out=False)
        err = int(max_abs_err(golden.astype(np.int64),
                              per_layer[L2["name"]].astype(np.int64)))
        self.assertEqual(err, 0)
        print("test_dncnn_padded_golden_matches_fixed_forward: PASS")


class TestFrontendBlurGolden(unittest.TestCase):
    def test_constant_image_blur_zero_pads_borders(self):
        # The frontend golden zero-pads: out-of-range kernel taps
        # are skipped (they contribute 0), so a constant image
        # darkens at the border. Kernel sums: corner 144/256,
        # edge 192/256, interior 256/256 -- for value 137 that is
        # 77 / 103 / 137 (round-half-up, shift 8).
        img = np.full((6, 6, 1), 137, dtype=np.uint8)
        cs = dict(w=6, h=6, k=3, shift=8, pt=1, pb=1, pl=1, pr=1,
                  sx=1, sy=1, ds=1, ch=1, sign=0, act=0, slope=0,
                  coeff=FE_COEFF,
                  img=[int(v) for v in img.reshape(-1)])
        out = golden_frontend.golden(cs)
        self.assertEqual(len(out), 36)
        g = np.array(out).reshape(6, 6)
        self.assertTrue((g[0, 0] == 77 and g[0, 5] == 77
                         and g[5, 0] == 77 and g[5, 5] == 77),
                        g[0])
        self.assertTrue((g[0, 2] == 103 and g[2, 0] == 103),
                        (g[0, 2], g[2, 0]))
        self.assertTrue((g[2, 2] == 137 and g[3, 3] == 137),
                        (g[2, 2], g[3, 3]))
        print("test_constant_image_blur_zero_pads_borders: PASS")

    def test_blur_keeps_shape_and_bounds(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        cs = dict(w=8, h=8, k=3, shift=8, pt=1, pb=1, pl=1, pr=1,
                  sx=1, sy=1, ds=1, ch=3, sign=0, act=0, slope=0,
                  coeff=FE_COEFF,
                  img=[int(v) for v in img.reshape(-1)])
        out = golden_frontend.golden(cs)
        self.assertEqual(len(out), 8 * 8 * 3)
        self.assertTrue(all(0 <= v <= 255 for v in out))
        print("test_blur_keeps_shape_and_bounds: PASS")


class TestCheckpoint(unittest.TestCase):
    def test_ckpt_roundtrip(self):
        import tempfile
        from run_e2e import load_ckpt, save_ckpt
        d = tempfile.mkdtemp()
        a = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
        b = np.full((4, 4, 1), 7, dtype=np.uint8)
        order = ["conv2d", "conv2d_1"]
        save_ckpt(d, "conv2d", a,
                  dict(max_err_funcsim=0, steps=10, shape=[4, 4, 1],
                       wall_s=1.0, sign_out=0), order)
        save_ckpt(d, "conv2d_1", b,
                  dict(max_err_funcsim=0, steps=20, shape=[4, 4, 1],
                       wall_s=2.0, sign_out=1), order)
        done, records = load_ckpt(d)
        self.assertEqual(list(done), ["conv2d", "conv2d_1"])
        self.assertTrue(np.array_equal(done["conv2d"], a))
        self.assertTrue(np.array_equal(done["conv2d_1"], b))
        self.assertEqual(records["conv2d_1"]["steps"], 20)
        self.assertEqual(records["conv2d_1"]["sign_out"], 1)
        print("test_ckpt_roundtrip: PASS")

    def test_ckpt_truncated_at_missing_layer(self):
        # ckpt.json lists a layer whose .npy is missing (killed
        # mid-write): the checkpoint truncates there, later layers
        # are dropped too.
        import tempfile
        from run_e2e import load_ckpt, save_ckpt
        d = tempfile.mkdtemp()
        a = np.zeros((2, 2, 1), dtype=np.uint8)
        order = ["conv2d", "conv2d_1", "conv2d_2"]
        save_ckpt(d, "conv2d", a,
                  dict(max_err_funcsim=0, steps=1, shape=[2, 2, 1],
                       wall_s=0.1, sign_out=0), order)
        save_ckpt(d, "conv2d_1", a,
                  dict(max_err_funcsim=0, steps=1, shape=[2, 2, 1],
                       wall_s=0.1, sign_out=0), order)
        # Fake a third layer in the json without its .npy.
        ck = json.load(open(os.path.join(d, "ckpt.json")))
        ck["layers"]["conv2d_2"] = dict(
            max_err_funcsim=0, steps=1, shape=[2, 2, 1], wall_s=0.1,
            sign_out=0)
        ck["order"].append("conv2d_2")
        json.dump(ck, open(os.path.join(d, "ckpt.json"), "w"))
        done, _records = load_ckpt(d)
        self.assertEqual(list(done), ["conv2d", "conv2d_1"])
        print("test_ckpt_truncated_at_missing_layer: PASS")

    def test_ckpt_corrupt_json_is_empty(self):
        import tempfile
        from run_e2e import load_ckpt
        d = tempfile.mkdtemp()
        open(os.path.join(d, "ckpt.json"), "w").write("{not json")
        done, records = load_ckpt(d)
        self.assertEqual(done, {})
        self.assertEqual(records, {})
        print("test_ckpt_corrupt_json_is_empty: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
