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
                 input_shape=None, streaming=False):
    """Emit the program for a fused + quantized graph.

    ``tile_results``: dict node name -> tile_search result (from
    costmodel.tile_search). ``act_tables``: dict node name -> table
    (from passes/mlut.py or passes/actapprox.py). Returns
    (program_dict, instr_lines, weight_blob_bytes).

    ``streaming`` (Stage 7b): emit ONE fused program for the whole
    chain for the 3-stage streaming pipeline (rtl/srcnn_pipe.sv)
    instead of the iterative per-layer DMA -> GO -> WAIT program.
    Weights for all layers are preloaded once at startup, the frame
    streams through L1->L2->L3 concurrently, and there is no
    per-layer GO/WAIT serialization. The op list (what the functional
    simulator executes) is identical in both modes; only the weight
    blob layout and the instruction stream differ. Raises ValueError
    if the chain does not fit the pipeline (see
    _assign_streaming_engines).
    """
    ops = []
    weights = {}
    instrs = []
    blob = bytearray()
    layout = {}
    signed_of = {}
    layer_index = 0  # scheduler layer index (single-channel path)
    stream_layers = []  # (op dict, geometry) per conv layer, streaming only
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
            if not streaming:
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
            if streaming:
                # The streaming branch (after the loop) emits the
                # fused program; the geometry comes from infer_shapes
                # (H/W/H_out/W_out on the node kwargs).
                stream_layers.append((op, n.kwargs))
                continue
            if cin == 1 and cout == 1 and n.op != "linear":
                _emit_sched_layer(instrs, op, layer_index, off)
                layer_index += 1
            else:
                _emit_mc_layer(instrs, op, off)
            continue
        if n.op == "act":
            if streaming:
                raise ValueError(
                    "streaming emission: standalone act node "
                    f"{n.name} is not supported by the pipeline "
                    "(only fused activations inside the conv layers)")
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
    streaming_info = None
    if streaming:
        # Fused streaming program: validate the chain, assign the
        # layers to the three pipeline engines, build the per-engine
        # weight blob and the single weight-load + STREAM instruction
        # stream. The op list above is unchanged (the functional
        # simulator executes the same ops in both modes).
        descs = _assign_streaming_engines(stream_layers)
        blob_bytes, layout = _streaming_blob(descs, weights)
        blob = bytearray(blob_bytes)
        instrs = _emit_streaming_instrs(descs)
        # The op dicts record the streaming weight offsets (the
        # iterative layout's offsets are meaningless here).
        for d in descs:
            for op in ops:
                if op.get("name") == d["layer"]:
                    op["weight_offset"] = d["weight_offset"]
                    op["ncoeff"] = d["ncoeff"]
        streaming_info = {
            "enabled": True,
            "engines": [{k: d[k] for k in
                         ("layer", "engine", "pipe_base", "K", "C_in",
                          "C_out", "W", "H", "W_out", "H_out", "shift",
                          "sign", "fused_act", "weight_offset", "ncoeff",
                          "dma_dst", "dma_taps", "dma_stride")}
                        for d in descs],
            "weight_blob_bytes": len(blob),
            "frames": "one STREAM per frame; engines run concurrently",
        }
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
    if streaming_info is not None:
        program["streaming"] = streaming_info
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


# ---------------------------------------------------------------------------
# Stage 7b streaming pipeline (rtl/srcnn_pipe.sv).
#
# The pipeline runs three sized engines concurrently (L1 conv2d_outpar,
# L2 conv1x1_cp, L3 conv2d_tappar) on one frame stream. The streaming
# program is ONE fused program for the whole chain: weights for all
# three layers are preloaded once at startup (DMA into each engine's
# coefficient memory), the geometry + bias are configured per engine,
# GO is written to L1 and L3 (L2 has no GO -- it runs when vectors
# arrive), and a single STREAM marker flows the input frame through.
# There is NO per-layer GO/WAIT serialization: the iterative program's
# per-(o,t) DMA -> GO -> WAIT sequence would cap throughput at
# ~13.5 fps and waste the hardware.
#
# Config routing (rtl/srcnn_pipe.sv): the top two address bits select
# the engine (L1 0x0000-0x3FFF, L2 0x4000-0x7FFF, L3 0x8000-0xBFFF);
# the engine sees cfg_addr[13:0]. The driver writes the routed
# addresses verbatim (W) and the weight DMA writes to the routed
# coefficient bases the same way -- no driver change is needed.
# ---------------------------------------------------------------------------
PIPE_L1_BASE = 0x0000
PIPE_L2_BASE = 0x4000
PIPE_L3_BASE = 0x8000

# Engine-local (low-14-bit) addresses. Geometry registers share the
# conv2d_mc map (0x00 K, 0x01 shift, 0x02 W, 0x03 H, 0x04-0x07 pads,
# 0x08/0x09 strides, 0x0A C_in, 0x0B C_out, 0x0C GO, 0x0D/0x0E out
# W/H, 0x0F sign, 0x10 fused act, 0x11 slope).
PIPE_COEFF_BASE = 0x100
# L1 (conv2d_outpar): coeff 0x100 + o*128 + tap (tap = r*9+c, 0..80,
# the 9x9 grid, kernel at the top-left); bias lo 0x2100+o, hi 0x2140+o
# (after the 0x100..0x20FF coeff range -- 0x300/0x400 would overlap).
PIPE_L1_BIAS_LO = 0x2100
PIPE_L1_BIAS_HI = 0x2140
PIPE_L1_TAPS = 81
PIPE_L1_STRIDE = 128
# L2 (conv1x1_cp): coeff 0x100 + o*64 + c; bias lo 0x900+o, hi 0x920+o.
PIPE_L2_BIAS_LO = 0x0900
PIPE_L2_BIAS_HI = 0x0920
PIPE_L2_TAPS = 64
PIPE_L2_STRIDE = 64
# L3 (conv2d_tappar): coeff 0x100 + tap*32 + c (tap = t*5+j, 0..24),
# contiguous 800 entries; bias lo 0x420, hi 0x440.
PIPE_L3_BIAS_LO = 0x0420
PIPE_L3_BIAS_HI = 0x0440
PIPE_L3_TAPS = 800
PIPE_L3_STRIDE = 1


def _assign_streaming_engines(stream_layers):
    """Assign the chain's conv layers to the pipeline engines.

    ``stream_layers``: list of (op dict, node kwargs) in chain order.
    Returns the engine descriptors (with the routed config addresses
    and the DMA plan, but without weight_offset/ncoeff -- those are
    filled in by _streaming_blob). Raises ValueError if the chain
    does not fit the 3-stage pipeline.

    Fit rules (from the RTL's verified shapes):
      * exactly 3 conv layers, chained (L1 out == L2 in, L2 out == L3
        in, channels included);
      * L1 -> conv2d_outpar: K 3..9, C_in == 1, C_out a multiple of 16
        in 16..64 (runtime ngroup = C_out/16);
      * L2 -> conv1x1_cp: K == 1, C_in == 64, C_out == 32 (the
        engine's NGROUP=5 and last-group mask are compile-time
        parameters for exactly this shape);
      * L3 -> conv2d_tappar: K == 5, C_in == 32, C_out == 1 (K=3 has
        a known column-alignment limitation in the RTL, so it is
        excluded here even though the engine accepts K=3);
      * all layers valid conv (pad 0, stride 1) -- the pipeline was
        verified for the SRCNN valid-conv chain; padded/strided
        shapes need their own golden coverage first.
    """
    if len(stream_layers) != 3:
        raise ValueError(
            "streaming emission: the pipeline has exactly 3 engines, "
            f"but the chain has {len(stream_layers)} conv layers")
    descs = []
    for (op, kw) in stream_layers:
        for key in ("H", "W", "H_out", "W_out"):
            if key not in kw:
                raise ValueError(
                    "streaming emission: layer "
                    f"{op['name']} has no {key} -- run compile_model / "
                    "infer_shapes before emitting")
        descs.append({
            "layer": op["name"], "K": op["K"], "C_in": op["C_in"],
            "C_out": op["C_out"], "pad": tuple(op["pad"]),
            "stride": tuple(op["stride"]), "shift": op["shift"],
            "sign": op["sign"], "fused_act": op.get("fused_act"),
            "act_slope": op.get("act_slope", 0.01),
            "bias": [int(v) for v in op["bias"]],
            "H": kw["H"], "W": kw["W"],
            "H_out": kw["H_out"], "W_out": kw["W_out"],
        })
    l1, l2, l3 = descs
    # L1 -> conv2d_outpar
    if not (3 <= l1["K"] <= 9):
        raise ValueError(
            f"streaming emission: L1 K={l1['K']} (conv2d_outpar needs "
            "3..9)")
    if l1["C_in"] != 1:
        raise ValueError(
            "streaming emission: L1 C_in must be 1 "
            f"(conv2d_outpar is single-channel in, got {l1['C_in']})")
    if l1["C_out"] % 16 or not (16 <= l1["C_out"] <= 64):
        raise ValueError(
            "streaming emission: L1 C_out must be a multiple of 16 in "
            f"16..64 (got {l1['C_out']})")
    # L2 -> conv1x1_cp (compile-time shape)
    if l2["K"] != 1 or l2["C_in"] != 64 or l2["C_out"] != 32:
        raise ValueError(
            "streaming emission: L2 must be K=1, C_in=64, C_out=32 "
            f"(conv1x1_cp parameters; got K={l2['K']} "
            f"C_in={l2['C_in']} C_out={l2['C_out']})")
    # L3 -> conv2d_tappar (K=5 verified; K=3 has a known limitation)
    if l3["K"] != 5 or l3["C_in"] != 32 or l3["C_out"] != 1:
        raise ValueError(
            "streaming emission: L3 must be K=5, C_in=32, C_out=1 "
            f"(got K={l3['K']} C_in={l3['C_in']} C_out={l3['C_out']})")
    for d in descs:
        if d["pad"] != (0, 0, 0, 0) or d["stride"] != (1, 1):
            raise ValueError(
                "streaming emission: layer "
                f"{d['layer']} must be valid conv (pad 0, stride 1); "
                f"got pad={d['pad']} stride={d['stride']}")
        if d.get("fused_act") not in (None, "none", "relu", "leaky"):
            raise ValueError(
                "streaming emission: layer "
                f"{d['layer']} fused_act={d.get('fused_act')} "
                "(pipeline engines implement relu/leaky only)")
    # Chaining (space + channels).
    if (l1["H_out"], l1["W_out"]) != (l2["H"], l2["W"]):
        raise ValueError(
            "streaming emission: L1 out "
            f"{(l1['H_out'], l1['W_out'])} != L2 in {(l2['H'], l2['W'])}")
    if (l2["H_out"], l2["W_out"]) != (l3["H"], l3["W"]):
        raise ValueError(
            "streaming emission: L2 out "
            f"{(l2['H_out'], l2['W_out'])} != L3 in {(l3['H'], l3['W'])}")
    if l1["C_out"] != l2["C_in"] or l2["C_out"] != l3["C_in"]:
        raise ValueError(
            "streaming emission: channel chain "
            f"{l1['C_out']}->{l2['C_in']}->{l2['C_out']}->{l3['C_in']} "
            "is broken")
    l1.update({"engine": "conv2d_outpar", "pipe_base": PIPE_L1_BASE,
               "bias_lo": PIPE_L1_BASE + PIPE_L1_BIAS_LO,
               "bias_hi": PIPE_L1_BASE + PIPE_L1_BIAS_HI,
               "dma_dst": PIPE_L1_BASE + PIPE_COEFF_BASE,
               "dma_taps": PIPE_L1_TAPS, "dma_stride": PIPE_L1_STRIDE})
    l2.update({"engine": "conv1x1_cp", "pipe_base": PIPE_L2_BASE,
               "bias_lo": PIPE_L2_BASE + PIPE_L2_BIAS_LO,
               "bias_hi": PIPE_L2_BASE + PIPE_L2_BIAS_HI,
               "dma_dst": PIPE_L2_BASE + PIPE_COEFF_BASE,
               "dma_taps": PIPE_L2_TAPS, "dma_stride": PIPE_L2_STRIDE})
    l3.update({"engine": "conv2d_tappar", "pipe_base": PIPE_L3_BASE,
               "bias_lo": PIPE_L3_BASE + PIPE_L3_BIAS_LO,
               "bias_hi": PIPE_L3_BASE + PIPE_L3_BIAS_HI,
               "dma_dst": PIPE_L3_BASE + PIPE_COEFF_BASE,
               "dma_taps": PIPE_L3_TAPS, "dma_stride": PIPE_L3_STRIDE})
    return descs


def _streaming_blob(descs, weights):
    """Build the streaming weight blob (int16 LE) in engine order.

    L1: C_out x 81 (the 9x9 grid, kernel at the top-left, tap = r*9+c);
    L2: C_out x 64 (o, c); L3: 25 x 32 (tap, c), tap = t*5+j, the full
    5x5 grid (25 taps even for K < 5 -- the RTL indexes tap*32+c on
    the 5x5 grid). Returns (blob_bytes, layout); also fills
    weight_offset/ncoeff (in coefficients) into each descriptor.
    """
    blob = bytearray()
    layout = {}
    for d in descs:
        name = d["layer"]
        W_q = weights[name]["W_q"]
        eng = d["engine"]
        off = len(blob) // 2
        if eng == "conv2d_outpar":
            K, cout = d["K"], d["C_out"]
            grid = np.zeros((cout, 81), dtype=np.int16)
            for o in range(cout):
                for r in range(K):
                    grid[o, r * 9:r * 9 + K] = W_q[r, :, 0, o]
            blob += grid.astype("<i2").tobytes()
            ncoeff = cout * 81
            order = "(o, tap) -- tap = r*9+c on the 9x9 grid, " \
                "81/lane, lane stride 128"
        elif eng == "conv1x1_cp":
            cout, cin = d["C_out"], d["C_in"]
            grid = np.zeros((cout, cin), dtype=np.int16)
            for o in range(cout):
                grid[o, :] = W_q[0, 0, :, o]
            blob += grid.astype("<i2").tobytes()
            ncoeff = cout * cin
            order = "(o, c) -- 64/lane, lane stride 64"
        else:  # conv2d_tappar
            K, cin = d["K"], d["C_in"]
            grid = np.zeros((25, cin), dtype=np.int16)
            for t in range(K):
                for j in range(K):
                    grid[t * 5 + j, :] = W_q[t, j, :, 0]
            blob += grid.astype("<i2").tobytes()
            ncoeff = 25 * cin
            order = "(tap, c) -- tap = t*5+j on the 5x5 grid, " \
                "contiguous 800"
        layout[name] = {"offset": off, "ncoeff": ncoeff, "order": order}
        d["weight_offset"] = off
        d["ncoeff"] = ncoeff
    return bytes(blob), layout


def _emit_streaming_instrs(descs):
    """Instruction stream for the fused streaming program.

    Per engine, in chain order: geometry + bias config writes, then
    one DMA loading that engine's coefficients. Then GO (as plain W
    writes to L1 0x000C and L3 0x800C; L2 has no GO) and a single
    STREAM marker. No bare GO, no WAIT -- the frame streams through
    all three engines concurrently.
    """
    instrs = []
    instrs.append("# streaming pipeline program (Stage 7b): one fused "
                  "program for the whole SRCNN chain")
    instrs.append("# weights preloaded once at startup; one STREAM per "
                  "frame; no per-layer GO/WAIT")
    for d in descs:
        base = d["pipe_base"]
        nm = d["layer"]
        instrs.append(f"# layer {nm} ({d['engine']}, pipe "
                      f"{base:#06x}): K={d['K']} Cin={d['C_in']} "
                      f"Cout={d['C_out']}")
        if d["engine"] == "conv1x1_cp":
            # No geometry: it processes vectors as they arrive.
            instrs.append(f"W {base + 0x01:#06x} {d['shift']:#06x}")
            instrs.append(f"W {base + 0x0A:#06x} {d['C_in']:#06x}")
            instrs.append(f"W {base + 0x0B:#06x} {d['C_out']:#06x}")
        else:
            instrs.append(f"W {base + 0x00:#06x} {d['K']:#06x}")
            instrs.append(f"W {base + 0x01:#06x} {d['shift']:#06x}")
            instrs.append(f"W {base + 0x02:#06x} {d['W']:#06x}")
            instrs.append(f"W {base + 0x03:#06x} {d['H']:#06x}")
            for a in (0x04, 0x05, 0x06, 0x07):
                instrs.append(f"W {base + a:#06x} 0x0000")
            instrs.append(f"W {base + 0x08:#06x} 0x0001")
            instrs.append(f"W {base + 0x09:#06x} 0x0001")
            instrs.append(f"W {base + 0x0A:#06x} {d['C_in']:#06x}")
            instrs.append(f"W {base + 0x0B:#06x} {d['C_out']:#06x}")
            instrs.append(f"W {base + 0x0D:#06x} {d['W_out']:#06x}")
            instrs.append(f"W {base + 0x0E:#06x} {d['H_out']:#06x}")
        instrs.append(f"W {base + 0x0F:#06x} {d['sign']:#06x}")
        fa = FUSED_ACT_CODE.get(d.get("fused_act") or "none", 0)
        instrs.append(f"W {base + 0x10:#06x} {fa:#06x}")
        if d.get("fused_act") == "leaky":
            slope = int(round(d.get("act_slope", 0.01) * 65536.0))
            instrs.append(f"W {base + 0x11:#06x} {slope:#06x}")
        # Bias (int32 per output channel, lo/hi halves).
        for o in range(d["C_out"]):
            b = int(d["bias"][o])
            instrs.append(f"W {d['bias_lo'] + o:#06x} {b & 0xFFFF:#06x}"
                          f"  # bias o={o} (lo)")
            instrs.append(f"W {d['bias_hi'] + o:#06x} "
                          f"{(b >> 16) & 0xFFFF:#06x}  # bias o={o} (hi)")
        # One DMA loads this engine's coefficients.
        dram_off = d["weight_offset"] * 2  # byte offset in weights.bin
        instrs.append(
            f"DMA {dram_off} {d['ncoeff']} {d['dma_dst']:#06x} "
            f"{d['dma_taps']} {d['dma_stride']}  # {nm} coeffs")
    # Start: GO to L1 and L3 as plain config writes (L2 has no GO).
    instrs.append("W 0x000c 0x0001  # GO L1 (conv2d_outpar)")
    instrs.append("W 0x800c 0x0001  # GO L3 (conv2d_tappar)")
    instrs.append("STREAM  # input frame (VDMA, Stages 8-11)")
    return instrs


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
        if program.get("streaming", {}).get("enabled"):
            f.write("# mode: streaming (Stage 7b pipeline: weights "
                    "preloaded once, one STREAM per frame)\n")
        for line in instrs:
            f.write(line + "\n")
    bp = os.path.join(outdir, "weights.bin")
    with open(bp, "wb") as f:
        f.write(blob)
    return jp, tp, bp


# ---------------------------------------------------------------------------
# Stage 7b (PE array): streaming emission as PE assignments.
#
# emit_pe_program is the compiler's streaming emission for the generic
# PE array (rtl/pe_array.sv). It wraps the scheduling pass
# (sw/sched_pe.py): given the model's layers and the input feature map,
# it emits the tile-step program -- one tile-step per (M-tile, N-chunk,
# K-chunk) -- that the pe_array dispatcher executes. This REPLACES the
# K^2 streaming emission above (emit_program(streaming=True), which
# targets the superseded rtl/srcnn_pipe.sv) for the PE array; the K^2
# streaming code is kept as reference and is not deleted. The
# tile-step program has no per-layer GO/WAIT serialization (one
# weight-load phase, then the frame streams as tile-steps); the
# scheduler decides the mode/TM/TN/TK and the PE assignment per layer.
# ---------------------------------------------------------------------------
def emit_pe_program(layers, inp, device_name="zu19eg"):
    """Emit the PE-array tile-step program for a model.

    ``layers``: list of layer dicts (K, C_out, mode_dot, sign_in,
    sign_out, shift, act, slope, W_q, b_q) as returned by
    sched_pe.load_srcnn(). ``inp``: the model's input feature map
    (HxWxC_in uint8). ``device_name``: a device in sw/devices.py.
    Returns (program_dict, steps_by_layer) where program_dict is
    {"mode": "streaming_pe", "device": device_name, "layers": [...]}
    and steps_by_layer is a list of (layer_name, steps, shape) per
    layer. The steps are consumed by the pe_array TB/driver
    (sw/run_pe_srcnn.py); they are NOT an instruction stream for the
    C driver (the PE array is programmed by tile-steps, not by
    per-layer GO/WAIT).
    """
    import sched_pe
    from devices import get as get_device
    dev = get_device(device_name)
    program = {"mode": "streaming_pe", "device": device_name, "layers": []}
    steps_by_layer = []
    cur = inp
    for L in layers:
        steps, shape = sched_pe.gen_layer_steps(L, cur, L["W_q"],
                                                L["b_q"], dev)
        program["layers"].append({
            "name": L["name"], "K": L["K"], "C_out": L["C_out"],
            "mode_dot": L["mode_dot"], "shift": L["shift"],
            "n_steps": len(steps), "out_shape": list(shape),
        })
        steps_by_layer.append((L["name"], steps, shape))
        # The next layer's input is this layer's golden output (the
        # driver replaces it with the RTL output wave-by-wave; here
        # the golden is used only to size the next layer's steps).
        from funcsim import conv2d_fixed_rhu
        cur = conv2d_fixed_rhu(cur, L["W_q"], L["b_q"], L["shift"])
    return program, steps_by_layer
