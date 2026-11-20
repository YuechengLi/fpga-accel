#!/usr/bin/env python3
"""Unit tests: tile-size cost model + DRAM model."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from costmodel import (LayerShape, tile_search, weight_load_cycles_axi,
                       weight_load_cycles_cfg)
from dram import (weight_dma_cycles, weight_cfg_cycles, dram_traffic,
                  vdma_bandwidth_gbs, arbitration_budget)
from roofline import (model_macs, roofline_point, analyze_model,
                      srcnn_layers, dncnn_layers, retinexnet_layers)


class TestCostModel(unittest.TestCase):
    def test_srcnn_l1(self):
        # SRCNN L1: K=9, Cin=1, 33x33 -> 25x25 out (valid, no pad).
        L = LayerShape(9, 1, 64, 33, 33, pad=(0, 0, 0, 0))
        r = tile_search(L)
        self.assertEqual(r["C_in_tile"], 1)
        self.assertEqual(r["ncoeff"], 81)
        self.assertTrue(r["load_hidden"])  # load 101cy < compute 649cy

    def test_dncnn_l2(self):
        # DnCNN L2: K=3, Cin=64, 32x32 (same-pad) -> 32x32.
        L = LayerShape(3, 64, 64, 32, 32, pad=(1, 1, 1, 1))
        r = tile_search(L)
        self.assertEqual(r["C_in_tile"], 4)
        self.assertEqual(r["ncoeff"], 36)
        self.assertTrue(r["load_hidden"])

    def test_k11_bank_limited(self):
        # K=11, Cin=64: C_in_tile=4 would need 4*121=484 coeffs >
        # the 289-coeff bank, so the tile is bank-limited to 2.
        L = LayerShape(11, 64, 64, 32, 32, pad=(5, 5, 5, 5))
        r = tile_search(L)
        self.assertEqual(r["C_in_tile"], 2)
        self.assertEqual(r["ncoeff"], 242)
        self.assertLessEqual(r["ncoeff"], 289)
        self.assertTrue(r["load_hidden"])

    def test_axi_faster_than_cfg(self):
        # AXI-MM (drain-limited, 1 coeff/cycle + burst overhead) is
        # ~2x faster than the Stage 4 config bus (2 cycles/coeff).
        for n in (36, 81, 242, 289):
            self.assertLess(weight_load_cycles_axi(n),
                            weight_load_cycles_cfg(n), f"ncoeff={n}")
        self.assertEqual(weight_load_cycles_cfg(100), 200)
        self.assertEqual(weight_dma_cycles(0), 0)

    def test_optimum_varies_per_layer(self):
        # The optimum tile is per-layer, not one global value.
        r1 = tile_search(LayerShape(9, 1, 64, 33, 33, pad=(0, 0, 0, 0)))
        r2 = tile_search(LayerShape(3, 64, 64, 32, 32, pad=(1, 1, 1, 1)))
        r3 = tile_search(LayerShape(11, 64, 64, 32, 32, pad=(5, 5, 5, 5)))
        tiles = {(r["C_in_tile"], r["H_tile"], r["W_tile"])
                 for r in (r1, r2, r3)}
        self.assertEqual(len(tiles), 3)


class TestDRAM(unittest.TestCase):
    def test_dncnn_unfused_matches_plan(self):
        # Plan Table 5 says 271 GB/s (all 17 layers 64ch in+out);
        # the real model has 1ch first/last layers, so the accurate
        # number is ~255 GB/s. Both are >> 27 GB/s (impossible).
        layers = dncnn_layers(1080, 1920)
        t = dram_traffic(layers, 1080, 1920, 60, fused=False)
        self.assertAlmostEqual(t["unfused_total_GBs"], 255.37, places=1)
        self.assertGreater(t["unfused_total_GBs"], 27.0)

    def test_dncnn_fused_matches_plan(self):
        # Plan Table 6 says 0.81 GB/s (RGB in AND out); the repo's
        # DnCNN is grayscale (1ch in/out), so the accurate number is
        # ~0.56 GB/s with in_ch=3/out_ch=1, ~0.31 with 1ch/1ch.
        layers = dncnn_layers(1080, 1920)
        t = dram_traffic(layers, 1080, 1920, 60, fused=True,
                         in_ch=3, out_ch=1)
        self.assertAlmostEqual(t["fused_total_GBs"], 0.56, places=1)
        t1 = dram_traffic(layers, 1080, 1920, 60, fused=True,
                          in_ch=1, out_ch=1)
        self.assertAlmostEqual(t1["fused_total_GBs"], 0.31, places=1)

    def test_vdma_budget(self):
        # 4K60 VDMA in+out = 2.99 GB/s, leaving ~23 GB/s for the engine.
        vin = vdma_bandwidth_gbs(2160, 3840, 3, 60)
        vout = vdma_bandwidth_gbs(2160, 3840, 3, 60)
        self.assertAlmostEqual(vin + vout, 2.99, places=1)
        eng, ok = arbitration_budget(vin, vout)
        self.assertTrue(ok)
        self.assertGreater(eng, 20.0)


class TestRoofline(unittest.TestCase):
    def test_dncnn_macs_match_plan(self):
        # Plan Table 7 says 1154 GMACs; we compute 1149 (0.4% --
        # a counting convention, method validated).
        layers = dncnn_layers(1080, 1920)
        macs = model_macs(layers)
        self.assertAlmostEqual(macs / 1e9, 1149, delta=5)

    def test_srcnn_macs_note_plan_discrepancy(self):
        # Plan Table 7 says 143 GMACs, but the 9-1-5 model in this
        # repo is 16.5 GMACs at 1080p -- the plan number appears to
        # assume a different SRCNN variant/resolution. The DnCNN
        # number matches, so the method is validated and the SRCNN
        # plan number is the outlier (recorded in sw/README.md).
        layers = srcnn_layers(1080, 1920)
        macs = model_macs(layers)
        self.assertAlmostEqual(macs / 1e9, 16.5, delta=0.5)
        self.assertNotAlmostEqual(macs / 1e9, 143, delta=10)

    def test_roofline_point_compute_bound(self):
        # Fused models are compute-bound (high arithmetic intensity).
        layers = dncnn_layers(1080, 1920)
        t = dram_traffic(layers, 1080, 1920, 60, fused=True,
                         in_ch=1, out_ch=1)
        intensity, ridge, bound = roofline_point(
            layers, t["fused_total_GBs"] / 60 * 1e9)
        self.assertEqual(bound, "compute")
        self.assertGreater(intensity, ridge)

    def test_analyze_model_fps(self):
        rows = analyze_model("DnCNN", dncnn_layers,
                             resolutions=((1920, 1080),))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertAlmostEqual(r["macs_G"], 1149, delta=5)
        # 1 engine = 145 GMAC/s -> ~0.13 fps; 4 engines -> ~0.51 fps.
        self.assertAlmostEqual(r["fps_1engine"], 0.13, places=1)
        self.assertAlmostEqual(r["fps_4engine"], 0.51, places=1)


if __name__ == "__main__":
    unittest.main()
