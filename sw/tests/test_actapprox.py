#!/usr/bin/env python3
"""Unit tests: activation approximation + multi-level LUT."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from passes.actapprox import (pwl_table, pwl_eval, clip_range,
                              hard_sigmoid, choose_act_impl, FUNCS)
from passes.mlut import (mlut_table, mlut_search, mlut_error,
                         mlut_config_writes)


class TestPWL(unittest.TestCase):
    def test_sigmoid_tanh_meet_budget(self):
        # 32-seg PWL over [-8,8] meets the 0.03 budget for both.
        for fn, budget in (("sigmoid", 0.03), ("tanh", 0.03)):
            y0, ds, max_err, ok = pwl_table(fn, budget)
            self.assertTrue(ok, fn)
            self.assertLess(max_err, budget, fn)
            self.assertEqual(len(y0), 32)
            self.assertEqual(len(ds), 32)
            # Independent re-evaluation on a dense grid agrees.
            xs = np.linspace(-8, 8, 2001)
            got = pwl_eval(fn, xs, y0, ds)
            self.assertLess(np.max(np.abs(got - FUNCS[fn](xs))), budget, fn)

    def test_gelu_silu_cannot_use_pwl(self):
        # PWL output is Q1.15 (range [-1,1)); GELU/SiLU reach ~8,
        # so they MUST go to the MLUT (Q5.11). choose_act_impl must
        # never select PWL for them.
        for fn in ("gelu", "silu"):
            res = choose_act_impl(fn, 0.03)
            self.assertEqual(res["impl"], "mlut", fn)
            self.assertIn("Q1.15", res["reason"], fn)

    def test_choose_pwl_for_sigmoid_tanh(self):
        for fn in ("sigmoid", "tanh"):
            res = choose_act_impl(fn, 0.03)
            self.assertEqual(res["impl"], "pwl", fn)
            self.assertTrue(res["meets_budget"], fn)

    def test_clip_range(self):
        # sigmoid saturates outside [-8,8]; tanh outside [-5,5].
        xmin, xmax, lo_sat, hi_sat = clip_range("sigmoid", -20.0, 20.0)
        self.assertEqual((xmin, xmax), (-8.0, 8.0))
        self.assertEqual((lo_sat, hi_sat), (0.0, 1.0))
        xmin, xmax, lo_sat, hi_sat = clip_range("tanh", -20.0, 20.0)
        self.assertEqual((xmin, xmax), (-5.0, 5.0))
        self.assertEqual((lo_sat, hi_sat), (-1.0, 1.0))
        # gelu does not saturate: range unchanged, no sat values.
        xmin, xmax, lo_sat, hi_sat = clip_range("gelu", -20.0, 20.0)
        self.assertEqual((xmin, xmax), (-8.0, 8.0))
        self.assertIsNone(lo_sat)
        self.assertIsNone(hi_sat)

    def test_hard_sigmoid_matches(self):
        xs = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
        got = hard_sigmoid(xs)
        self.assertTrue(np.allclose(got, [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0]))


class TestMLUT(unittest.TestCase):
    def test_reproduces_stage4_numbers(self):
        # The Stage 4 golden split (H1=8,H2=4,8 bands,384 entries)
        # must reproduce the exact Stage 4 max errors.
        expected = {"sigmoid": 0.0027, "tanh": 0.0186,
                    "gelu": 0.0269, "silu": 0.0234}
        for fn, exp in expected.items():
            h1, h2, base, sel, fine = mlut_table(fn, 8, 4, 8)
            mx, _rms = mlut_error(fn, h1, h2, base, sel, fine)
            self.assertAlmostEqual(mx, exp, places=3, msg=fn)
            self.assertEqual(len(base), 256)
            self.assertEqual(len(fine), 8)
            self.assertEqual(len(fine[0]), 16)
            self.assertEqual(256 + 8 * 16, 384)

    def test_search_meets_budget(self):
        for fn in ("sigmoid", "tanh", "gelu", "silu"):
            h1, h2, nb, base, sel, fine, mx, total = mlut_search(fn, 0.03)
            self.assertLess(mx, 0.03, fn)
            self.assertLess(total, 4096, fn)
            self.assertEqual(total, (1 << h1) + nb * (1 << h2), fn)

    def test_search_smallest_table(self):
        # sigmoid needs far fewer entries than tanh at the same budget.
        *_, total_s = mlut_search("sigmoid", 0.03)
        *_, total_t = mlut_search("tanh", 0.03)
        self.assertLess(total_s, total_t)

    def test_config_emission(self):
        h1, h2, base, sel, fine = mlut_table("sigmoid", 8, 4, 8)
        writes = mlut_config_writes("sigmoid", h1, h2, base, sel, fine)
        # 1 control + 256 coarse base + 256 coarse sel + 8*16 fine
        self.assertEqual(len(writes), 1 + 256 + 256 + 128)
        # control word encodes H1/H2
        self.assertEqual(writes[0], (0x02, 8 | (4 << 4)))
        self.assertEqual(writes[1][0], 0x200)   # first coarse base
        self.assertEqual(writes[257][0], 0x300)  # first coarse sel
        self.assertEqual(writes[513][0], 0x400)  # first fine entry


if __name__ == "__main__":
    unittest.main()
