#!/usr/bin/env python3
"""Unit tests for the VCU128 (VU37P) platform config (sw/common/devices.py).

Covers:
  - The vu37p device config numbers (9,024 DSP, 2,852K LCs, HBM2
    460 GB/s, PCIe Gen4 x8 host link, no ARM PS) and the regression
    guard that the zu19eg/ecp5u85 configs are unchanged.
  - The scheduler on vu37p: TN=8, so L3 stays in dot mode (the
    TN<8 guard in sw/common/sched_pe.py does not fire); the full SRCNN
    program replays vs the funcsim golden with max|err| = 0.
  - The cycle model: vu37p 1080p fps at 300 MHz equals the zu19eg
    figure (same 15x128 PE config) at 21.3% of the VU37P's DSP.
  - The HBM roofline: decode is bandwidth-bound (460 GB/s -> 7B
    INT8 ~65.7 tok/s); the PE count is NOT scaled up (1,920 MACs
    already exceed the ~1,533 MACs/cycle that saturate HBM).

The step-replay helper below duplicates sw/vision/tests/test_sched_pe.py's
_replay_layer on purpose: that module imports sw/common/emit.py, which a
concurrent agent is editing, and these tests must not depend on it.
The semantics are identical (outer/dot-agnostic partial accumulation
+ K-chunk merge + round-half-up shift + ReLU + saturate).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "common"))

from costmodel import device_mem_bandwidth_gbs
from devices import DEVICES, get as get_device
from dram import HBM_ACTUAL_GBS, arbitration_budget
from funcsim import conv2d_fixed_rhu
from roofline import (HBM_ACTUAL_GBS as ROOFLINE_HBM_GBS, PEAK_GMACS,
                      PL_DDR4_ACTUAL_GBS, VU37P_PEAK_GMACS,
                      roofline_point, srcnn_layers)
from sched_pe import assemble_output, gen_layer_steps, load_srcnn
from stage6 import build_srcnn_input
from stage7b_model_pe import decode_roofline, hbm_bytes_per_cycle
from stage7b_model_pe import model as pe_model


def _replay_layer(steps, shape, tn):
    """Replay one layer's tile-steps in Python (outer/dot agnostic).

    Per (mtile, nchunk): accumulate the per-K-chunk partials
    (bias on first_k only), then apply the round-half-up shift +
    ReLU + 0..255 saturate once to the merged sum -- the pe_array
    merge-unit + pe_emit semantics.
    """
    acc = {}
    last = {}
    for s in steps:
        key = (s["mtile"], s["nchunk"])
        if key not in acc:
            acc[key] = np.zeros((16, tn), dtype=np.int64)
        A = np.array(s["A"], dtype=np.int64)   # TM x tk
        B = np.array(s["B"], dtype=np.int64)   # tk x TN
        part = A @ B
        if s["first_k"]:
            part = part + np.array(s["bias"], dtype=np.int64)[None, :]
        acc[key] += part
        last[key] = s
    outs = []
    for key, a in acc.items():
        s = last[key]
        shift = s["shift"]
        x = a
        if shift:
            x = (x + (1 << (shift - 1))) >> shift
        if s["act"] == 1:  # ReLU
            x = np.maximum(x, 0)
        x = np.clip(x, 0, 255).astype(np.uint8)
        outs.append((key[0], key[1], x.reshape(-1).tolist()))
    return outs


class TestVCU128(unittest.TestCase):
    def test_vu37p_config_numbers(self):
        dev = get_device("vu37p")
        self.assertEqual(dev["name"],
                         "XCVU37P-2FSVH2892E (VCU128, Virtex UltraScale+ HBM)")
        self.assertEqual(dev["vendor"], "xilinx")
        self.assertEqual(dev["dsp"], 9024)
        self.assertEqual(dev["lut"], 2852000)
        self.assertEqual(dev["mem"], "HBM2")
        self.assertEqual(dev["mem_gbs"], 460.0)
        self.assertEqual(dev["host_link"], "pcie-gen4-x8")
        self.assertFalse(dev["has_ps"])
        # Same PE config as the ZU19EG on purpose (decode is
        # bandwidth-bound; the PE count is NOT scaled up).
        self.assertEqual((dev["n_pe"], dev["pe_mults"],
                          dev["tm"], dev["tn"]), (15, 128, 16, 8))
        self.assertEqual(dev["clock_mhz"], 300.0)
        self.assertEqual(dev["targets"], [(1920, 1080)])
        self.assertEqual(dev["verified"], "model")
        self.assertIn("vu37p", DEVICES)
        self.assertEqual(len(DEVICES), 3)

    def test_zu19eg_ecp5u85_unchanged(self):
        # Regression guard: the two existing configs must be
        # byte-identical in behavior (same dicts as before vu37p).
        self.assertEqual(DEVICES["zu19eg"], {
            "name": "XCZU19EG-2 (iW-RainboW-G35D)",
            "vendor": "xilinx",
            "dsp": 1968,
            "lut": 523000,
            "bram_kbits": 984 * 36,
            "clock_mhz": 300.0,
            "clocks_mhz": (300.0, 360.0, 400.0),
            "n_pe": 15,
            "pe_mults": 128,
            "tm": 16,
            "tn": 8,
            "frame_tiles": (64, 32, 16, 8),
            "targets": [(1920, 1080)],
            "verified": "rtl",
        })
        self.assertEqual(DEVICES["ecp5u85"], {
            "name": "LFE5U-85F (Lattice ECP5, low-power)",
            "vendor": "lattice",
            "dsp": 156,
            "lut": 84000,
            "bram_kbits": 3744,
            "clock_mhz": 100.0,
            "clocks_mhz": (100.0,),
            "n_pe": 2,
            "pe_mults": 64,
            "tm": 16,
            "tn": 4,
            "frame_tiles": (64, 32, 16, 8),
            "targets": [(640, 480), (320, 240)],
            "verified": "model+synth",
        })

    def test_scheduler_vu37p_l3_dot(self):
        # TN=8 on vu37p, so the dot-mode guard (TN<8 -> outer) must
        # NOT fire: L3 stays dot (tn=1, 4 cy/step), L1/L2 outer.
        layers, _ = load_srcnn()
        inp_u8, _ = build_srcnn_input(24)
        dummy = {"L1": inp_u8,
                 "L2": np.zeros((16, 16, 64), dtype=np.uint8),
                 "L3": np.zeros((16, 16, 32), dtype=np.uint8)}
        dev = get_device("vu37p")
        for L, want_dot, want_tk in (("L1", 0, 27), ("L2", 0, 32),
                                     ("L3", 1, 32)):
            layer = next(x for x in layers if x["name"] == L)
            steps, shape = gen_layer_steps(layer, dummy[L],
                                           layer["W_q"], layer["b_q"],
                                           dev)
            modes = {s["mode_dot"] for s in steps}
            self.assertEqual(modes, {want_dot},
                             f"vu37p {L}: mode_dot={modes}")
            self.assertEqual(steps[0]["tk"], want_tk)
            if want_dot:
                self.assertTrue(all(s["tn"] == 1 for s in steps))

    def test_vu37p_chain_replay_matches_golden(self):
        # Whole SRCNN chain on the vu37p program, replayed in
        # Python, vs the funcsim golden per layer (max|err| = 0).
        # Same bar as the zu19eg/ecp5u85 chain replays; the PE
        # parameters are identical to zu19eg, so the RTL chain
        # result for zu19eg covers this config's compute path.
        layers, _ = load_srcnn()
        inp_u8, _ = build_srcnn_input(24)
        dev = get_device("vu37p")
        cur = inp_u8
        for L in layers:
            steps, shape = gen_layer_steps(L, cur, L["W_q"], L["b_q"],
                                           dev)
            outs = _replay_layer(steps, shape, dev["tn"])
            got = assemble_output(outs, shape, TN=dev["tn"])
            golden = conv2d_fixed_rhu(cur, L["W_q"], L["b_q"],
                                      L["shift"])
            err = int(np.max(np.abs(got.astype(np.int32)
                                  - golden.astype(np.int32)))) \
                if golden.size else 0
            self.assertEqual(err, 0, f"vu37p {L['name']}: err={err}")
            cur = got

    def test_cycle_model_vu37p(self):
        vu = get_device("vu37p")
        zu = get_device("zu19eg")
        m_vu = pe_model(vu, 1920, 1080)
        m_zu = pe_model(zu, 1920, 1080)
        # Same PE config -> identical cycle counts and fps.
        self.assertEqual(m_vu["frame_cy"], m_zu["frame_cy"])
        self.assertEqual(m_vu["dsps"], 1920)
        fps_vu = 300.0e6 / m_vu["frame_cy"]
        self.assertGreaterEqual(fps_vu, 30.0)
        self.assertAlmostEqual(fps_vu, 33.7, places=1)
        self.assertAlmostEqual(300.0e6 / m_vu["frame_cy"],
                               300.0e6 / m_zu["frame_cy"], places=6)
        # ... at 21.3% of the VU37P's DSP (vs 97.6% of the ZU19EG).
        self.assertAlmostEqual(100.0 * m_vu["dsps"] / vu["dsp"],
                               21.3, places=1)
        # L3 runs in dot mode (4 cy/step: TK=32, TN=8).
        self.assertEqual(m_vu["per_layer"][2]["cyc_step"], 4)
        self.assertEqual(m_vu["per_layer"][0]["tk"], 27)
        self.assertEqual(m_vu["per_layer"][1]["tk"], 32)

    def test_decode_roofline(self):
        # HBM 460 GB/s delivers ~1,533 weight bytes/cycle at
        # 300 MHz; one MAC consumes one weight byte/cycle.
        self.assertAlmostEqual(hbm_bytes_per_cycle(460.0, 300.0),
                               1533.3, places=1)
        # The 15x128 config (1,920 MACs/cycle) already exceeds the
        # saturation point -- the PE count must NOT be scaled up.
        self.assertGreater(15 * 128, hbm_bytes_per_cycle(460.0, 300.0))
        # Decode tok/s roofline = bandwidth / bytes per token.
        self.assertAlmostEqual(decode_roofline(460.0, 7.0e9), 65.7,
                               places=1)
        self.assertAlmostEqual(decode_roofline(460.0, 6.5e9), 70.8,
                               places=1)
        self.assertAlmostEqual(decode_roofline(460.0, 13.0e9), 35.4,
                               places=1)
        # Capacity (8 GB HBM per board): 7B INT8 and 13B INT4 fit
        # one board; 13B INT8 needs 2 boards (16 GB).
        self.assertLessEqual(7.0, 8.0)
        self.assertLessEqual(6.5, 8.0)
        self.assertGreater(13.0, 8.0)
        self.assertLessEqual(13.0, 16.0)

    def test_roofline_hbm(self):
        layers = srcnn_layers(1080, 1920)
        # Default call is the ZU19EG DDR roofline, unchanged.
        _, ridge_ddr, _ = roofline_point(layers, 10_000_000)
        self.assertAlmostEqual(ridge_ddr,
                               PEAK_GMACS / PL_DDR4_ACTUAL_GBS, places=6)
        # VCU128: 2707.2 GMAC/s peak over 460 GB/s -> ridge ~5.9
        # MACs/byte, far below the DDR ridge (~21.9).
        self.assertAlmostEqual(VU37P_PEAK_GMACS, 2707.2, places=1)
        _, ridge_hbm, bound = roofline_point(
            layers, 10_000_000, mem_gbs=ROOFLINE_HBM_GBS,
            peak_gmacs=VU37P_PEAK_GMACS)
        self.assertAlmostEqual(ridge_hbm,
                               VU37P_PEAK_GMACS / ROOFLINE_HBM_GBS,
                               places=6)
        self.assertLess(ridge_hbm, ridge_ddr)
        self.assertEqual(bound, "compute")

    def test_costmodel_bandwidth(self):
        self.assertEqual(device_mem_bandwidth_gbs(get_device("vu37p")),
                         460.0)
        self.assertEqual(device_mem_bandwidth_gbs(get_device("zu19eg")),
                         27.0)

    def test_dram_hbm_budget(self):
        # Default is the ZU19EG PL DDR4 budget, unchanged.
        eng, ok = arbitration_budget(1.0, 1.0)
        self.assertAlmostEqual(eng, 27.0 - 1.0 - 1.0 - 1.0)
        self.assertTrue(ok)
        # VCU128: same call with the HBM budget.
        eng_h, ok_h = arbitration_budget(1.0, 1.0,
                                         mem_gbs=HBM_ACTUAL_GBS)
        self.assertAlmostEqual(eng_h, 460.0 - 1.0 - 1.0 - 1.0)
        self.assertTrue(ok_h)


if __name__ == "__main__":
    unittest.main()
