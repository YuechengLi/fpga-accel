#!/usr/bin/env python3
"""Driver: run SRCNN (24x24 input) through the PE array, one wave per
layer (the Stage 6 pattern). For each layer:
  1. Generate tile-steps from the current input (RTL output of the
     previous layer; the original image for L1) via sw/common/sched_pe.py.
  2. Write tb/vectors_pe_array.txt and run tb_pe_array (iverilog+vvp).
  3. Read tb/out_pe_array.txt, assemble the layer output, compare to
     the funcsim golden (conv2d_fixed_rhu) -- max|err| must be 0.
  4. Feed the RTL output as the next layer's input.
At the end, compare the final L3 output to the golden final output
(max|err| = 0) and compute PSNR vs the float reference (~43.5 dB).

Usage: python3 sw/vision/run_pe_srcnn.py [--netlist <pe_array netlist>]
  --netlist: post-synthesis mode (Stage 7c, Mac Phase 3) -- compile
    the synthesized pe_array netlist (tb/pe_array_ps_wrap.sv +
    the netlist + Yosys simcells.v) instead of the RTL, and run
    the same chain + golden check against it. Without it, the
    RTL is used (the Stage 7b path, unchanged).
  --simcells: path to Yosys simcells.v (netlist mode only);
    auto-detected when omitted.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "common"))
import numpy as np

from funcsim import conv2d_fixed_rhu
from sched_pe import (assemble_output, gen_layer_steps, load_srcnn,
                      write_layer_vectors)
from stage6 import build_srcnn_input

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TB = os.path.join(REPO, "tb")

# Post-synthesis mode (set by main() from --netlist/--simcells).
NETLIST = None
SIMCELLS = None


def find_simcells():
    """Locate Yosys simcells.v (VM: /usr/share/yosys; Mac Homebrew:
    /opt/homebrew or /usr/local). Returns None if not found."""
    for p in (SIMCELLS,
              "/usr/share/yosys/simcells.v",
              "/opt/homebrew/share/yosys/simcells.v",
              "/usr/local/share/yosys/simcells.v"):
        if p and os.path.exists(p):
            return p
    return None


def run_tb():
    """Compile + run tb_pe_array (in tb/). Returns (rc, log).
    Netlist mode compiles the parameter-accepting wrapper +
    the synthesized netlist + simcells.v instead of the RTL;
    the TB and the golden check are unchanged."""
    if NETLIST:
        simcells = find_simcells()
        if simcells is None:
            return 1, ("NETLIST MODE: simcells.v not found "
                       "(pass --simcells, or brew install yosys)")
        netlist = NETLIST if os.path.isabs(NETLIST) \
            else os.path.join(REPO, NETLIST)
        if not os.path.exists(netlist):
            return 1, f"NETLIST MODE: netlist not found: {netlist}"
        files = ["pe_array_ps_wrap.sv", netlist, simcells,
                 "tb_pe_array.sv"]
    else:
        files = ["../rtl/common/pe_array.sv", "../rtl/common/pe_mac.sv",
                 "../rtl/common/pe_emit.sv", "tb_pe_array.sv"]
    comp = subprocess.run(
        ["iverilog", "-g2012", "-o", "tb_pe_array.vvp"] + files,
        cwd=TB, capture_output=True, text=True)
    if comp.returncode != 0:
        return comp.returncode, "COMPILE FAILED:\n" + comp.stdout + comp.stderr
    run = subprocess.run(
        ["vvp", "tb_pe_array.vvp"],
        cwd=TB, capture_output=True, text=True, timeout=1200)
    return run.returncode, run.stdout + run.stderr


def read_outputs(path):
    """Read out_pe_array.txt -> list of (mtile, nchunk, bytes[128])."""
    toks = open(path).read().split()
    out = []
    i = 0
    while i < len(toks):
        mtile = int(toks[i]); nchunk = int(toks[i + 1]); i += 2
        byts = [int(toks[i + j]) for j in range(128)]
        i += 128
        out.append((mtile, nchunk, byts))
    return out


def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    global NETLIST, SIMCELLS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlist", default=None,
                    help="synthesized pe_array netlist (repo-relative "
                    "or absolute); post-synthesis mode, see module "
                    "docstring")
    ap.add_argument("--simcells", default=None,
                    help="path to Yosys simcells.v (netlist mode only)")
    args = ap.parse_args()
    NETLIST = args.netlist
    SIMCELLS = args.simcells
    if NETLIST:
        print(f"NETLIST MODE: pe_array netlist {NETLIST}")
    from devices import get as get_device
    dev = get_device("zu19eg")
    layers, _ = load_srcnn()
    inp_u8, inp_f = build_srcnn_input(24)
    # Float reference (for the final PSNR).
    # Golden per-layer outputs via the functional simulator.
    cur = inp_u8
    cur_f = inp_f
    golden = {}
    for L in layers:
        g = conv2d_fixed_rhu(cur, L["W_q"], L["b_q"], L["shift"],
                             pad=(0, 0, 0, 0))
        # conv2d_fixed_rhu returns the activated uint8 output.
        golden[L["name"]] = g
        cur = g
    final_golden = golden["L3"]
    # Float reference: run the float SRCNN (models/common golden).
    from models.vision.common import conv2d_float
    import json
    d = np.load(os.path.join(REPO, "models", "vision", "srcnn", "srcnn_x3.npz"))
    with open(os.path.join(REPO, "models", "vision", "srcnn",
                           "srcnn_x3_quant.json")) as f:
        q = json.load(f)
    # Float forward (ReLU after each conv, same as the fixed golden).
    xf = inp_f
    for L in layers:
        W = d[L["name"] + "_W"]
        b = d[L["name"] + "_b"]
        xf = conv2d_float(xf, W, b)
        xf = np.maximum(xf, 0)
    float_out = np.clip(np.round(xf * 255.0), 0, 255).astype(np.uint8)

    cur = inp_u8
    ok_all = True
    for L in layers:
        steps, shape = gen_layer_steps(L, cur, L["W_q"], L["b_q"], dev)
        write_layer_vectors(steps, shape,
                            os.path.join(TB, "vectors_pe_array.txt"))
        print(f"{L['name']}: {len(steps)} steps, out {shape}", flush=True)
        rc, log = run_tb()
        print(log.strip(), flush=True)
        if rc != 0:
            print(f"{L['name']}: TB FAILED (rc={rc})")
            return 1
        outs = read_outputs(os.path.join(TB, "out_pe_array.txt"))
        rtl_out = assemble_output(outs, shape)
        g = golden[L["name"]]
        err = int(np.max(np.abs(rtl_out.astype(np.int32)
                                - g.astype(np.int32)))) if g.size else 0
        status = "PASS" if err == 0 else "FAIL"
        print(f"  {L['name']}: RTL vs funcsim max|err|={err} [{status}] "
              f"({len(outs)} outputs)", flush=True)
        if err != 0:
            ok_all = False
            # Show the first few mismatches.
            idx = np.argwhere(rtl_out != g)
            for (r, c, ch) in idx[:5]:
                print(f"    ({r},{c},{ch}): rtl={rtl_out[r,c,ch]} "
                      f"golden={g[r,c,ch]}")
            return 1
        cur = rtl_out
    final_err = int(np.max(np.abs(cur.astype(np.int32)
                                  - final_golden.astype(np.int32))))
    p = psnr(cur, float_out)
    print(f"FINAL: vs funcsim max|err|={final_err}, "
          f"vs float PSNR={p:.2f} dB (expect ~43.5 dB)")
    if final_err == 0 and ok_all:
        print("PE ARRAY SRCNN CHAIN: ALL PASS")
        return 0
    print("PE ARRAY SRCNN CHAIN: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
