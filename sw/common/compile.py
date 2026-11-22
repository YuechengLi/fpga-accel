#!/usr/bin/env python3
"""Stage 5 compiler: the pipeline.

  capture (torch.export FX graph, or a hand-built IR graph)
    -> fusion (conv+ReLU fused, BN folded, pad folded, complex act
       routed to the separate LUT module, fusion groups)
    -> quantization (per-layer activation ranges from the calibration
       input, int16 weights, RTL shift)
    -> activation approximation (per act node: PWL or multi-level LUT
       table, chosen by the accuracy budget)
    -> tiling (per-layer tile-size search with the cost model)
    -> scheduling (tile order + ping-pong bank assignment -- the
       emitter assigns bank L%2 per layer and the DMA fills the next
       bank while the current layer computes)
    -> emission (program.json + program.txt + weights.bin)

``compile_model`` is the single entry point. It returns the program
and a report; it raises (never silently skips) if any pass cannot
meet its budget or a node cannot be lowered.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from passes.fusion import fuse
from passes.quantize import quantize_graph
from passes.actapprox import choose_act_impl, pwl_table
from passes.mlut import mlut_search, mlut_config_writes
from costmodel import LayerShape, tile_search, layer_report
from emit import emit_program


def infer_shapes(graph, input_shape):
    """Propagate (H, W, C) shapes through the fused graph. Stores
    H_out/W_out/C_out in each node's kwargs (used by the tile search
    and the fusion-group tile-buffer check)."""
    shapes = {}
    for n in graph.nodes:
        if n.op == "input":
            shapes[n.name] = tuple(input_shape)
            n.kwargs["H_out"], n.kwargs["W_out"], n.kwargs["C_out"] = \
                input_shape
            continue
        if n.op in ("conv2d", "dw_conv2d", "linear"):
            h, w, c = shapes[n.args[0]]
            K = n.kwargs["K"]
            pt, pb, pl, pr = n.kwargs.get("pad", (0, 0, 0, 0))
            sx, sy = n.kwargs.get("stride", (1, 1))
            dil = n.kwargs.get("dilation", (1, 1))
            keff = (K - 1) * dil[0] + 1
            ho = (h + pt + pb - keff) // sx + 1
            wo = (w + pl + pr - keff) // sy + 1
            co = n.kwargs["C_out"]
            shapes[n.name] = (ho, wo, co)
            n.kwargs["H_out"], n.kwargs["W_out"] = ho, wo
            n.kwargs["H"], n.kwargs["W"] = h, w
            continue
        if n.op in ("max_pool2d", "avg_pool2d"):
            h, w, c = shapes[n.args[0]]
            kh, kw = n.kwargs.get("K", (2, 2))
            sh, sw = n.kwargs.get("stride", (kh, kw))
            shapes[n.name] = ((h - kh) // sh + 1, (w - kw) // sw + 1, c)
            n.kwargs["H_out"], n.kwargs["W_out"], n.kwargs["C_out"] = \
                shapes[n.name]
            continue
        if n.op == "gap":
            h, w, c = shapes[n.args[0]]
            shapes[n.name] = (1, 1, c)
            continue
        if n.op == "upsample":
            h, w, c = shapes[n.args[0]]
            s = n.kwargs.get("scale", 2)
            s = s if isinstance(s, (list, tuple)) else (s, s)
            shapes[n.name] = (h * int(s[0]), w * int(s[1]), c)
            continue
        if n.op in ("residual", "add", "sub", "mul", "simplegate",
                    "relu", "leaky_relu", "act", "sigmoid", "tanh",
                    "gelu", "silu", "bypass", "pad", "layer_norm"):
            shapes[n.name] = shapes[n.args[0]]
            continue
        if n.op == "concat":
            h, w, _ = shapes[n.args[0]]
            c = sum(shapes[a][2] for a in n.args)
            shapes[n.name] = (h, w, c)
            continue
        raise RuntimeError(f"infer_shapes: unsupported op {n.op} "
                           f"(node {n.name})")
    return shapes


def compile_model(graph, calib_input, input_shape, model_name="model",
                  err_budget=0.03, streaming=False):
    """Run the full compiler pipeline. ``calib_input`` is the
    calibration image as float (H, W, C) in [0, 1].
    Returns (program, instrs, blob, report_dict).

    ``streaming`` (Stage 7b): emit the fused streaming program for
    the 3-stage pipeline (rtl/vision/srcnn_pipe.sv) instead of the iterative
    per-layer program. See emit.emit_program.
    """
    report = {"model": model_name, "passes": {}}
    # 1. fusion
    fused, groups, freport = fuse(graph)
    report["passes"]["fusion"] = freport
    report["fusion_groups"] = groups
    # 2. shapes (needed by tiling + the fusion tile-buffer check)
    infer_shapes(fused, input_shape)
    # 3. quantization (calibrates on the calibration input)
    fused, qreport = quantize_graph(fused, calib_input)
    report["passes"]["quantize"] = qreport
    # 4. activation approximation (per act node)
    act_tables = {}
    areport = []
    for n in fused.nodes:
        if n.op != "act":
            continue
        func = n.kwargs["func"]
        choice = choose_act_impl(func, err_budget)
        if choice["impl"] == "pwl":
            y0, ds, mx, ok = pwl_table(func, err_budget)
            cw = []
            for s in range(32):
                cw.append((0x100 + s * 2, y0[s] & 0xFFFF))
                cw.append((0x100 + s * 2 + 1, ds[s] & 0xFFFF))
            act_tables[n.name] = {"impl": "pwl", "y0": y0, "dslope": ds,
                                  "max_err": mx, "config_writes": cw}
            areport.append(f"{n.name} ({func}): PWL 32-seg "
                           f"max_err={mx:.4f}")
        else:
            h1, h2, nb, base, sel, fine, mx, total = mlut_search(
                func, err_budget)
            cw = mlut_config_writes(func, h1, h2, base, sel, fine)
            act_tables[n.name] = {"impl": "mlut", "H1": h1, "H2": h2,
                                  "nbands": nb, "base": base, "sel": sel,
                                  "fine": fine, "max_err": mx,
                                  "entries": total,
                                  "config_writes": cw}
            areport.append(f"{n.name} ({func}): MLUT H1={h1} H2={h2} "
                           f"nbands={nb} entries={total} "
                           f"max_err={mx:.4f}")
    report["passes"]["actapprox"] = areport
    # 5. tiling (per-layer tile-size search)
    tile_results = {}
    treport = []
    for n in fused.nodes:
        if n.op not in ("conv2d", "dw_conv2d", "linear"):
            continue
        mode = ("dw" if n.op == "dw_conv2d"
                else "fc" if n.op == "linear" else "dense")
        layer = LayerShape(
            n.kwargs["K"], n.kwargs["C_in"], n.kwargs["C_out"],
            n.kwargs["H"], n.kwargs["W"],
            stride=tuple(n.kwargs.get("stride", (1, 1))),
            pad=tuple(n.kwargs.get("pad", (0, 0, 0, 0))),
            mode=mode)
        res = tile_search(layer)
        tile_results[n.name] = res
        treport.append(layer_report(layer, res))
    report["passes"]["tiling"] = treport
    # 6. emission
    program, instrs, blob = emit_program(
        fused, tile_results, act_tables, model_name=model_name,
        input_shape=input_shape, streaming=streaming)
    report["instr_count"] = len(instrs)
    report["weight_blob_bytes"] = len(blob)
    report["streaming"] = bool(streaming)
    # Attach the runtime structures the functional simulator needs.
    program["_weights_arrays"] = {
        k: {"W_q": v["W_q"], "b_q": v["b_q"]}
        for k, v in ((n.name, n.params) for n in fused.nodes
                     if n.op in ("conv2d", "dw_conv2d", "linear"))}
    program["_act_tables_full"] = act_tables
    # The functional simulator consumes the program's op list; make
    # sure the op list carries the fused-graph ops (emit already did).
    return program, instrs, blob, report
