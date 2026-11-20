"""Unit tests for the PE-array streaming emission (Stage 7b).

emit_pe_program (sw/emit.py) wraps the scheduling pass (sw/sched_pe.py)
and emits the tile-step program for the PE array. This test verifies:
  - the program structure (mode, device, 3 layers, step counts),
  - the tile-steps match the scheduler directly,
  - the emission is a tile-step program (no per-layer GO/WAIT
    instruction stream -- the PE array is programmed by tile-steps).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import emit
from sched_pe import gen_layer_steps, load_srcnn
from stage6 import build_srcnn_input
from devices import get as get_device


def test_pe_program_structure():
    layers, _ = load_srcnn()
    inp_u8, _ = build_srcnn_input(24)
    program, steps_by_layer = emit.emit_pe_program(layers, inp_u8, "zu19eg")
    assert program["mode"] == "streaming_pe"
    assert program["device"] == "zu19eg"
    assert len(program["layers"]) == 3
    names = [L["name"] for L in program["layers"]]
    assert names == ["L1", "L2", "L3"], names
    # Step counts match the scheduler's per-layer counts.
    assert program["layers"][0]["n_steps"] == 384
    assert program["layers"][1]["n_steps"] == 128
    assert program["layers"][2]["n_steps"] == 225
    # Output shapes.
    assert program["layers"][0]["out_shape"] == [16, 16, 64]
    assert program["layers"][1]["out_shape"] == [16, 16, 32]
    assert program["layers"][2]["out_shape"] == [12, 12, 1]
    # L3 is dot mode (N=1), L1/L2 are outer.
    assert program["layers"][0]["mode_dot"] == 0
    assert program["layers"][1]["mode_dot"] == 0
    assert program["layers"][2]["mode_dot"] == 1
    print("test_pe_program_structure: PASS")


def test_pe_steps_match_scheduler():
    layers, _ = load_srcnn()
    inp_u8, _ = build_srcnn_input(24)
    program, steps_by_layer = emit.emit_pe_program(layers, inp_u8, "zu19eg")
    dev = get_device("zu19eg")
    # Recompute L1 steps directly and compare the first step's fields.
    steps, shape = gen_layer_steps(layers[0], inp_u8, layers[0]["W_q"],
                                   layers[0]["b_q"], dev)
    assert len(steps) == len(steps_by_layer[0][1])
    a, b = steps[0], steps_by_layer[0][1][0]
    for k in ("mode_dot", "tk", "tn", "first_k", "sign_in", "shift",
              "act", "slope", "sign_out", "mtile", "nchunk", "n_kchunk"):
        assert a[k] == b[k], (k, a[k], b[k])
    assert a["A"] == b["A"] and a["B"] == b["B"] and a["bias"] == b["bias"]
    print("test_pe_steps_match_scheduler: PASS")


def test_pe_program_is_tile_steps_not_instructions():
    # The PE program has no instruction stream / GO / WAIT: it is a
    # list of tile-steps (the array is programmed by tile-steps).
    layers, _ = load_srcnn()
    inp_u8, _ = build_srcnn_input(24)
    program, steps_by_layer = emit.emit_pe_program(layers, inp_u8, "zu19eg")
    assert "instrs" not in program and "blob" not in program
    for (name, steps, shape) in steps_by_layer:
        for s in steps:
            assert "mode_dot" in s and "tk" in s and "A" in s and "B" in s
            assert "GO" not in str(s) and "WAIT" not in str(s)
    print("test_pe_program_is_tile_steps_not_instructions: PASS")


if __name__ == "__main__":
    test_pe_program_structure()
    test_pe_steps_match_scheduler()
    test_pe_program_is_tile_steps_not_instructions()
    print("ALL PE STREAMING EMISSION TESTS PASSED")
