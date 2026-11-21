#!/usr/bin/env python3
"""Unit tests for the PE-array scheduler (Stage 7b, sw/sched_pe.py).

Covers the dot-mode TN guard found 2026-11-21: the PE's dot
reduction tree sums 8 products per row (pe_mac.sv gen_rowsum reads
prod[gr*TN+0..7]) and each (n,g) step processes 8 taps, so dot mode
is only correct for TN=8. With TN<8 the scheduler must force outer
mode; otherwise L3 (the N=1 layer) drops half its K-dim, mixes
adjacent rows, and reads prod out of bounds for the last row.

Tests:
  - L3 is outer on the ECP5 (TN=4) config and dot on the ZU19EG
    (TN=8) config, in both the scheduler steps and the emitted
    program (sw/emit.py must report the effective mode).
  - Outer mode uses ceil-division for N-chunks, so C_out < TN
    (L3 C_out=1 on TN=4) and C_out % TN != 0 layers schedule
    without asserting, with the tail chunk zero-padded.
  - The whole ECP5 SRCNN program, replayed step-by-step in Python
    (outer-mode partials + K-chunk merge + round-half-up shift +
    ReLU, the pe_array + pe_emit semantics), matches the functional
    simulator golden (conv2d_fixed_rhu) with max|err| = 0 per layer.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import emit
from devices import get as get_device
from funcsim import conv2d_fixed_rhu
from sched_pe import assemble_output, gen_layer_steps, load_srcnn
from stage6 import build_srcnn_input


def _replay_layer(steps, shape, tn):
    """Replay one layer's tile-steps in Python (outer/dot agnostic).

    Per (mtile, nchunk): accumulate the per-K-chunk partials
    (bias on first_k only), then apply the round-half-up shift +
    ReLU + 0..255 saturate once to the merged sum -- the pe_array
    merge-unit + pe_emit semantics. Returns (mtile, nchunk, bytes)
    outputs in (m, n) order for assemble_output.
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


class TestSchedPE(unittest.TestCase):
    def test_l3_outer_on_tn4_dot_on_tn8(self):
        layers, _ = load_srcnn()
        inp_u8, _ = build_srcnn_input(24)
        dummy = {"L1": inp_u8,
                 "L2": np.zeros((16, 16, 64), dtype=np.uint8),
                 "L3": np.zeros((16, 16, 32), dtype=np.uint8)}
        for dev_name, want_dot in (("ecp5u85", 0), ("zu19eg", 1)):
            dev = get_device(dev_name)
            steps, _ = gen_layer_steps(layers[2], dummy["L3"],
                                       layers[2]["W_q"], layers[2]["b_q"],
                                       dev)
            modes = {s["mode_dot"] for s in steps}
            self.assertEqual(modes, {want_dot},
                             f"{dev_name}: L3 steps mode_dot={modes}")
            self.assertTrue(all(s["tn"] == (1 if want_dot else dev["tn"])
                                for s in steps))
        # The emitted program reports the effective mode, not the
        # layer's request.
        for dev_name, want_dot in (("ecp5u85", 0), ("zu19eg", 1)):
            program, _ = emit.emit_pe_program(layers, inp_u8, dev_name)
            self.assertEqual(program["layers"][2]["mode_dot"], want_dot,
                             f"{dev_name}: program L3 mode_dot")
            self.assertEqual(program["layers"][0]["mode_dot"], 0)
            self.assertEqual(program["layers"][1]["mode_dot"], 0)

    def test_outer_ceil_nchunk(self):
        # Synthetic outer-mode layers on the TN=4 device: C_out < TN
        # and C_out % TN != 0 must schedule (ceil N-chunks) instead
        # of asserting, with the tail chunk zero-padded.
        dev = get_device("ecp5u85")
        rng = np.random.default_rng(0)
        for c_out, want_chunks in ((1, 1), (6, 2)):
            W_q = rng.integers(-100, 100, size=(1, 1, 1, c_out),
                               dtype=np.int16)
            b_q = np.zeros((c_out,), dtype=np.int32)
            layer = dict(name="SYN", K=1, C_out=c_out, mode_dot=0,
                         sign_in=0, sign_out=0, shift=0, act=1, slope=0,
                         W_q=W_q, b_q=b_q)
            inp = rng.integers(0, 256, size=(8, 8, 1), dtype=np.uint8)
            steps, shape = gen_layer_steps(layer, inp, W_q, b_q, dev)
            self.assertEqual(shape, (8, 8, c_out))
            nchunk = {s["nchunk"] for s in steps}
            self.assertEqual(nchunk, set(range(want_chunks)))
            # Tail padding: B columns and bias beyond C_out are zero.
            for s in steps:
                co0 = s["nchunk"] * dev["tn"]
                for n in range(dev["tn"]):
                    if co0 + n >= c_out:
                        self.assertTrue(
                            all(row[n] == 0 for row in s["B"]),
                            "B tail not zero")
                        self.assertEqual(s["bias"][n], 0, "bias tail")
            # Replay matches the functional golden (padding is exact).
            outs = _replay_layer(steps, shape, dev["tn"])
            got = assemble_output(outs, shape, TN=dev["tn"])
            golden = conv2d_fixed_rhu(inp, W_q, b_q, 0)
            self.assertEqual(
                int(np.max(np.abs(got.astype(np.int32)
                                  - golden.astype(np.int32)))), 0)

    def test_ecp5_chain_replay_matches_golden(self):
        # Whole SRCNN chain on the ECP5 (TN=4) program, replayed in
        # Python, vs the funcsim golden per layer (max|err| = 0).
        # This is the functional check of the forced-outer L3: before
        # the guard the scheduler emitted L3 as dot, which the replay
        # (and the RTL) computes wrongly at TN=4.
        layers, _ = load_srcnn()
        inp_u8, _ = build_srcnn_input(24)
        dev = get_device("ecp5u85")
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
            self.assertEqual(err, 0, f"ecp5u85 {L['name']}: err={err}")
            cur = got


if __name__ == "__main__":
    unittest.main()
