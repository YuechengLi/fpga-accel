#!/usr/bin/env python3
"""Stage 5 compiler: emission.

Emits the compiled program in three artifacts:

  * program.json -- the full program (op list, per-op quantized
    weights as arrays, act tables, scheduler descriptors). This is
    what the functional simulator (sw/funcsim.py) executes.
  * program.txt -- the instruction stream for the PS driver (C).
    A simple text format, one instruction per line:
      W <addr> <data>              config-bus write (16-bit addr/data)
      DMA <dram_off> <ncoeff> <dst_base> <taps> <stride>
                                   program the weight DMA (rtl/
                                   weight_dma.sv) and wait for done;
                                   dst_base is the config-bus base
                                   (0x0200 bank A / 0x0400 bank B for
                                   the scheduler, 0x0100 for the
                                   engine); taps = taps per lane
                                   (K*K), stride = address stride
                                   between lanes (128 for conv2d_mc,
                                   1 for a contiguous bank).
      GO                         write GO (conv2d_mc 0x0C) / start
                                   the scheduler (0x0000 bit0)
      WAIT                       poll done/busy until idle
      STREAM <bytes>             marker: the driver streams the input
                                   frame (VDMA, Stages 8-11)
  * weights.bin -- the int16 weight blob (little-endian), in
    HARDWARE order: for a multi-channel conv, for each output
    channel o, for each C_in tile t, for each lane l, for each tap k
    (row-major in the 11x11 grid, 0..120): the coefficient. The
    driver writes them to the engine at 0x100 + l*128 + k for that
    (o, t) pass. For a single-channel (scheduler) layer the blob is
    the full 121-entry grid, written contiguously to the bank
    (0x0200+i / 0x0400+i) and streamed by the scheduler to the
    engine at 0x100+i. (Stage 6 fix: the blob used to carry only
    the K*K taps, but both engines index their coefficient memory
    by the 11x11 grid, so the contiguous K*K placement put the
    coefficients at the wrong addresses for K < 11.)

Two execution paths (the Stage 4/5 boundary):
  * multi-channel conv (C_in > 1 or C_out > 1): conv2d_mc, driven
    directly by the driver (config + per-(o,t) coefficient load +
    GO). The weight DMA feeds the engine's coefficient registers
    directly (strided addressing, lane_stride=128).
  * single-channel conv (C_in == 1, C_out == 1): the layer scheduler
    + unified engine. The driver programs the scheduler descriptors
    (0x0000-0x0012), the DMA fills the ping-pong bank, and the
    scheduler sequences the layer. The bank alternates per layer
    (layer L uses bank L%2).
"""
import json
import math
import os
import struct

import numpy as np

# conv2d_mc config addresses (rtl/conv2d_mc.sv)
MC_K = 0x00
MC_SHIFT = 0x01
MC_W = 0x02
MC_H = 0x03
MC_PT = 0x04
MC_PB = 0x05
MC_PL = 0x06
MC_PR = 0x07
MC_SX = 0x08
MC_SY = 0x09
MC_CIN = 0x0A
MC_COUT = 0x0B
MC_GO = 0x0C
MC_OUTW = 0x0D
MC_OUTH = 0x0E
MC_SIGN = 0x0F
MC_FUSED_ACT = 0x10
MC_SLOPE = 0x11
MC_BIAS_LO = 0x12
MC_BIAS_HI = 0x13
MC_COEFF_BASE = 0x100
MC_LANE_STRIDE = 128

# layer_scheduler config addresses (rtl/layer_scheduler.sv)
SCHED_CTRL = 0x0000
SCHED_NLAYER = 0x0001
SCHED_LSEL = 0x0002
SCHED_MODE = 0x0003
SCHED_K = 0x0004
SCHED_SHIFT = 0x0005
SCHED_W = 0x0006
SCHED_H = 0x0007
SCHED_PT = 0x0008
SCHED_PB = 0x0009
SCHED_PL = 0x000A
SCHED_PR = 0x000B
SCHED_SX = 0x000C
SCHED_SY = 0x000D
SCHED_SIGN = 0x000E
SCHED_ACT = 0x000F
SCHED_SLOPE = 0x0010
SCHED_DIL = 0x0011
SCHED_NCOEFF = 0x0012
SCHED_BANK_A = 0x0200
SCHED_BANK_B = 0x0400

# unified_engine modes (rtl/unified_engine.sv)
MODE_DENSE = 0
MODE_DW_SQ = 1
MODE_DW_SH = 2
MODE_DW_SV = 3
MODE_DW_DIL = 4
MODE_FC = 5

FUSED_ACT_CODE = {"none": 0, "relu": 1, "leaky": 2}


def emit_program(graph, tile_results, act_tables, model_name="model",
                 input_shape=None):
    """Emit the program for a fused + quantized graph.

    ``tile_results``: dict node name -> tile_search result (from
    costmodel.tile_search). ``act_tables``: dict node name -> table
    (from passes/mlut.py or passes/actapprox.py). Returns
    (program_dict, instr_lines, weight_blob_bytes).
    """
    ops = []
    weights = {}
    instrs = []
    blob = bytearray()
    layout = {}
    signed_of = {}
    layer_index = 0  # scheduler layer index (single-channel path)
    for n in graph.nodes:
        if n.op == "input":
            ops.append({"name": n.name, "op": "input",
                        "args": [],
                        "shape": list(input_shape) if input_shape else None})
            signed_of[n.name] = False
            continue
        if n.op in ("conv2d", "dw_conv2d", "linear"):
            src = n.args[0]
            signed_in = signed_of.get(src, False)
            signed_out = bool(n.kwargs.get("signed_out", False))
            sign = (1 if signed_in else 0) | (2 if signed_out else 0)
            K = n.kwargs["K"]
            cin = n.kwargs["C_in"]
            cout = n.kwargs["C_out"]
            pad = tuple(n.kwargs.get("pad", (0, 0, 0, 0)))
            stride = tuple(n.kwargs.get("stride", (1, 1)))
            fused_act = n.kwargs.get("fused_act")
            tile = tile_results.get(n.name, {})
            cin_tile = tile.get("C_in_tile", min(4, cin))
            h_tile = tile.get("H_tile", 32)
            w_tile = tile.get("W_tile", 32)
            W_q = n.params["W_q"]
            b_q = n.params["b_q"]
            # Weight blob, hardware order: (o, t, l, k).
            off = len(blob) // 2
            ncoeff_total = 0
            for o in range(cout):
                for t0 in range(0, cin, cin_tile):
                    t1 = min(t0 + cin_tile, cin)
                    for l in range(t1 - t0):
                        c = t0 + l
                        taps = W_q[:, :, c, o].reshape(-1)
                        # taps are K*K row-major; the engine's 11x11
                        # grid uses the top-left KxK, index = r*11+c.
                        grid = np.zeros(121, dtype=np.int16)
                        for r in range(K):
                            grid[r * 11:(r * 11) + K] = \
                                taps[r * K:(r * K) + K]
                        # Store the FULL 121-entry grid per lane
                        # (Stage 6 fix). The MC's coefficient memory
                        # is indexed by the 11x11 grid (r*11+c), NOT
                        # by the contiguous KxK row-major index: the
                        # DMA writes tap k at dst + l*128 + k, so
                        # the blob must carry the grid (zeros outside
                        # the KxK kernel) or the MAC reads the wrong
                        # coefficients. Storing only the K*K taps
                        # (the previous code) put them at 0..K*K-1
                        # instead of the grid positions and failed
                        # the Stage 6 DMA integration check
                        # (245/256 outputs wrong, SRCNN L1 ch5).
                        blob += grid.astype("<i2").tobytes()
                        ncoeff_total += 121
            weights[n.name] = {"W_q": W_q, "b_q": b_q}
            layout[n.name] = {"offset": off, "ncoeff": ncoeff_total,
                              "order": "(o, t, l, k) -- k is the 11x11 "
                              "grid index (0..120), 121/lane"}
            op = {"name": n.name, "op": n.op, "args": [src],
                  "K": K, "C_in": cin, "C_out": cout,
                  "pad": list(pad), "pad_mode": n.kwargs.get("pad_mode", "zero"),
                  "stride": list(stride), "shift": n.kwargs["shift"],
                  "r_in": n.kwargs.get("r_in", 0),
                  "r_out": n.kwargs.get("r_out", 0),
                  "signed_in": signed_in, "signed_out": signed_out,
                  "sign": sign, "fused_act": fused_act,
                  "act_slope": n.kwargs.get("act_slope", 0.01),
                  "C_in_tile": cin_tile, "H_tile": h_tile,
                  "W_tile": w_tile,
                  "weight_offset": off, "ncoeff": ncoeff_total,
                  "bias": [int(v) for v in b_q]}
            ops.append(op)
            signed_of[n.name] = signed_out
            # ---- instruction stream ----
            if cin == 1 and cout == 1 and n.op != "linear":
                _emit_sched_layer(instrs, op, layer_index, off)
                layer_index += 1
            else:
                _emit_mc_layer(instrs, op, off)
            continue
        if n.op == "act":
            src = n.args[0]
            tab = act_tables.get(n.name, {})
            op = {"name": n.name, "op": "act", "args": [src],
                  "func": n.kwargs.get("func"),
                  "impl": tab.get("impl", "mlut"),
                  "r_in": n.kwargs.get("r_in", 0),
                  "r_out": n.kwargs.get("r_out", 0),
                  "signed_out": bool(n.kwargs.get("signed_out", False))}
            ops.append(op)
            signed_of[n.name] = op["signed_out"]
            # Act config writes (MLUT tables or PWL table) go through
            # the act module's config bus (the driver writes them
            # before the layer that uses the act).
            for (a, d) in tab.get("config_writes", []):
                instrs.append(f"W {a:#06x} {d & 0xFFFF:#06x}")
            instrs.append(f"W 0x0000 {6 if op['impl'] == 'mlut' else 2:#06x}  # act mode")
            continue
        # Pass-through / elementwise / pool / upsample / etc.
        op = {"name": n.name, "op": n.op, "args": list(n.args)}
        op.update({k: v for k, v in n.kwargs.items()
                   if k in ("mode", "K", "stride", "scale", "dim")})
        if n.op in ("residual", "add", "sub"):
            op["mode"] = n.kwargs.get("mode", "add")
        ops.append(op)
        src = n.args[0] if n.args else None
        signed_of[n.name] = signed_of.get(src, False) if src else False
    program = {
        "model": model_name,
        "version": 1,
        "ops": ops,
        "weights": {k: {"W_q_shape": list(v["W_q"].shape),
                        "b_q": [int(x) for x in v["b_q"]]}
                    for k, v in weights.items()},
        "weight_layout": layout,
        "act_tables": {k: _table_summary(v)
                       for k, v in act_tables.items()},
        "outputs": list(graph.outputs),
        "instr_count": len(instrs),
    }
    # The functional simulator needs the actual arrays, not just the
    # summary -- attach them under a separate key that the JSON writer
    # strips.
    program["_weights_arrays"] = weights
    program["_act_tables_full"] = act_tables
    return program, instrs, bytes(blob)


def _table_summary(tab):
    if not tab:
        return {}
    if tab.get("impl") == "mlut":
        return {"impl": "mlut", "H1": tab["H1"], "H2": tab["H2"],
                "nbands": tab["nbands"], "max_err": tab["max_err"],
                "entries": (1 << tab["H1"])
                + tab["nbands"] * (1 << tab["H2"])}
    return {"impl": "pwl", "max_err": tab.get("max_err")}


def _emit_mc_layer(instrs, op, blob_off):
    """Instruction stream for one multi-channel conv (conv2d_mc)."""
    K = op["K"]
    cin = op["C_in"]
    cout = op["C_out"]
    cin_tile = op["C_in_tile"]
    pad = op["pad"]
    sx, sy = op["stride"]
    instrs.append(f"# layer {op['name']}: K={K} Cin={cin} Cout={cout} "
                  f"(conv2d_mc, driver-driven)")
    instrs.append(f"W {MC_K:#06x} {K:#06x}")
    instrs.append(f"W {MC_SHIFT:#06x} {op['shift']:#06x}")
    instrs.append(f"W {MC_W:#06x} {op['W_tile']:#06x}")
    instrs.append(f"W {MC_H:#06x} {op['H_tile']:#06x}")
    instrs.append(f"W {MC_PT:#06x} {pad[0]:#06x}")
    instrs.append(f"W {MC_PB:#06x} {pad[1]:#06x}")
    instrs.append(f"W {MC_PL:#06x} {pad[2]:#06x}")
    instrs.append(f"W {MC_PR:#06x} {pad[3]:#06x}")
    instrs.append(f"W {MC_SX:#06x} {sx:#06x}")
    instrs.append(f"W {MC_SY:#06x} {sy:#06x}")
    instrs.append(f"W {MC_CIN:#06x} {cin:#06x}")
    instrs.append(f"W {MC_COUT:#06x} {cout:#06x}")
    instrs.append(f"W {MC_SIGN:#06x} {op['sign']:#06x}")
    fa = FUSED_ACT_CODE.get(op.get("fused_act") or "none", 0)
    instrs.append(f"W {MC_FUSED_ACT:#06x} {fa:#06x}")
    if op.get("fused_act") == "leaky":
        slope = int(round(op.get("act_slope", 0.01) * 65536.0))
        instrs.append(f"W {MC_SLOPE:#06x} {slope:#06x}")
    # Per (o, t) pass: DMA the coefficients into the engine, then GO.
    # The blob offset for pass (o, t) is computed by the driver from
    # the layout (o, t, l, k) order; here we emit one DMA per pass.
    # The per-output-channel bias b_q (int32) is written once per
    # output channel, before the first tile's DMA (Stage 6: the RTL
    # adds it on the first tile only, so one write per channel).
    idx = 0
    for o in range(cout):
        for t0 in range(0, cin, cin_tile):
            t1 = min(t0 + cin_tile, cin)
            # One full 121-entry grid per lane (Stage 6 fix -- see the
            # blob construction above). taps=121, not K*K.
            ncoeff = (t1 - t0) * 121
            if t0 == 0:
                b = int(op["bias"][o])
                instrs.append(f"W {MC_BIAS_LO:#06x} {b & 0xFFFF:#06x}"
                              f"  # bias o={o} (lo)")
                instrs.append(f"W {MC_BIAS_HI:#06x} {(b >> 16) & 0xFFFF:#06x}"
                              f"  # bias o={o} (hi)")
            dram_off = (blob_off + idx) * 2  # byte offset in weights.bin
            instrs.append(
                f"DMA {dram_off} {ncoeff} {MC_COEFF_BASE:#06x} {121} "
                f"{MC_LANE_STRIDE}  # pass o={o} cin_tile={t0}..{t1 - 1}")
            idx += ncoeff
            instrs.append("GO")
            instrs.append("WAIT")
    instrs.append(f"STREAM  # input tile stream for {op['name']}")


def _emit_sched_layer(instrs, op, layer_index, blob_off):
    """Instruction stream for one single-channel layer (scheduler +
    unified engine)."""
    K = op["K"]
    pad = op["pad"]
    sx, sy = op["stride"]
    bank = layer_index % 2
    bank_base = SCHED_BANK_A if bank == 0 else SCHED_BANK_B
    # One full 121-entry grid (Stage 6 fix -- the dense engine's
    # coefficient memory is indexed by the 11x11 grid, r*11+c, not
    # by the contiguous K*K taps; the Stage 4 scheduler TB already
    # uses ncoeff=121). The bank holds 289 entries, so 121 fits.
    ncoeff = 121
    mode = MODE_FC if K == 1 else MODE_DENSE
    instrs.append(f"# layer {op['name']}: K={K} single-channel "
                  f"(scheduler layer {layer_index}, bank "
                  f"{'A' if bank == 0 else 'B'})")
    instrs.append(f"W {SCHED_LSEL:#06x} {layer_index:#06x}")
    instrs.append(f"W {SCHED_MODE:#06x} {mode:#06x}")
    instrs.append(f"W {SCHED_K:#06x} {K:#06x}")
    instrs.append(f"W {SCHED_SHIFT:#06x} {op['shift']:#06x}")
    instrs.append(f"W {SCHED_W:#06x} {op['W_tile']:#06x}")
    instrs.append(f"W {SCHED_H:#06x} {op['H_tile']:#06x}")
    instrs.append(f"W {SCHED_PT:#06x} {pad[0]:#06x}")
    instrs.append(f"W {SCHED_PB:#06x} {pad[1]:#06x}")
    instrs.append(f"W {SCHED_PL:#06x} {pad[2]:#06x}")
    instrs.append(f"W {SCHED_PR:#06x} {pad[3]:#06x}")
    instrs.append(f"W {SCHED_SX:#06x} {sx:#06x}")
    instrs.append(f"W {SCHED_SY:#06x} {sy:#06x}")
    instrs.append(f"W {SCHED_SIGN:#06x} {op['sign']:#06x}")
    fa = FUSED_ACT_CODE.get(op.get("fused_act") or "none", 0)
    instrs.append(f"W {SCHED_ACT:#06x} {fa:#06x}")
    instrs.append(f"W {SCHED_NCOEFF:#06x} {ncoeff:#06x}")
    dram_off = blob_off * 2
    instrs.append(f"DMA {dram_off} {ncoeff} {bank_base:#06x} {121} 1"
                  f"  # fill bank {'A' if bank == 0 else 'B'}")
    # The scheduler is started once, after all layer descriptors and
    # the first bank are programmed (the driver issues the start
    # after the loop; here we record it per layer for clarity).
    instrs.append(f"# (scheduler start issued by the driver after "
                  f"all {layer_index + 1} layers are programmed)")


def write_program(program, instrs, blob, outdir):
    """Write program.json, program.txt, weights.bin to outdir.
    Returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    # JSON: strip the numpy arrays (they are in weights.bin / the
    # functional simulator's in-memory program).
    pub = {k: v for k, v in program.items()
           if not k.startswith("_")}
    # Make the weights entry JSON-serializable (it already is).
    jp = os.path.join(outdir, "program.json")
    with open(jp, "w") as f:
        json.dump(pub, f, indent=2)
    tp = os.path.join(outdir, "program.txt")
    with open(tp, "w") as f:
        f.write("# FPGA accelerator program (Stage 5 compiler)\n")
        f.write(f"# model: {program['model']}\n")
        for line in instrs:
            f.write(line + "\n")
    bp = os.path.join(outdir, "weights.bin")
    with open(bp, "wb") as f:
        f.write(blob)
    return jp, tp, bp
