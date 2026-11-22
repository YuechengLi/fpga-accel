#!/usr/bin/env python3
"""Stage 5 compiler: quantization pass.

For each conv node, run the float forward on the calibration input,
measure the per-layer max activation, and quantize the weights with
the SAME scheme as models/vision/common.py (the Stage 6 golden):

  * coefficients: int16, per-layer power-of-two scale
    e = floor(log2(32767/max|W|)), W_q = round(W * 2**e)
  * bias: b_q = round(b * (255/2**r_in) * 2**e), int32
  * RTL shift = e + r_out - r_in, kept in [0, 15] (reduce e if needed)
  * r_out = range_exp(max activation) of the layer's float output
    (after the fused act, if any); r_in is the producer's r_out
    (0 for the model input, which is uint8 = float * 255).

For an ``act`` node (complex activation): r_in is the producer
conv's r_out; r_out is set from the act's float output range
(measured in the same float forward). Unsigned output (sigmoid,
[0,1]): r_out = range_exp(max), signed_out=False. Signed output
(tanh/GELU/SiLU, min < 0): the value is a signed int8 with the same
scale 255/2**r_out, so r_out = range_exp(2 * max|output|) to keep
|value| <= 127.

The pass attaches to each conv node: kwargs shift/r_in/r_out/e and
params W_q/b_q. The quantized weights are what the emitter puts in
the weight blob and what the functional simulator executes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vision.common import quantize_layer, range_exp
from funcsim import float_forward


def quantize_graph(graph, calib_input):
    """Quantize all conv/linear nodes in the fused graph.

    ``calib_input`` is the calibration image as float (H, W, C) in
    [0, 1] (the float model's input scale). Returns (graph, report)
    where report lists the per-layer shift/r_in/r_out/e.
    """
    vals, _ = float_forward(graph, calib_input)
    report = []
    # r_out of each node (for the next layer's r_in). The model input
    # is uint8 = float * 255, so r = 0.
    r_out_of = {}
    for n in graph.nodes:
        if n.op == "input":
            r_out_of[n.name] = 0
    for n in graph.nodes:
        if n.op in ("conv2d", "dw_conv2d", "linear"):
            # r_in = the producer's r_out
            src = n.args[0]
            r_in = r_out_of.get(src, 0)
            # max activation of this layer's float output (after the
            # fused act, which float_forward already applied)
            out = vals[n.name]
            max_act = float(np.max(np.abs(out))) if out.size else 0.0
            # For an unsigned output the range is [0, max]; for a
            # signed output (no fused act and min < 0) the range is
            # symmetric. The fused ReLU makes the output unsigned.
            if n.kwargs.get("fused_act") in ("relu", "leaky"):
                r_out = range_exp(max_act)
                signed_out = False
            else:
                mn = float(np.min(out)) if out.size else 0.0
                mx = float(np.max(out)) if out.size else 0.0
                if mn < 0:
                    # signed intermediate: keep |value| <= 127
                    r_out = range_exp(2.0 * max(abs(mn), abs(mx)))
                    signed_out = True
                else:
                    r_out = range_exp(mx)
                    signed_out = False
            W = n.params["weight"]
            b = n.params.get("bias", np.zeros(W.shape[3]))
            W_q, b_q, shift, scale = quantize_layer(W, b, r_in=r_in,
                                                     r_out=r_out)
            n.params["W_q"] = W_q
            n.params["b_q"] = b_q
            n.kwargs["shift"] = shift
            n.kwargs["r_in"] = r_in
            n.kwargs["r_out"] = r_out
            n.kwargs["weight_exp_e"] = shift - r_out + r_in
            n.kwargs["signed_out"] = signed_out
            r_out_of[n.name] = r_out
            report.append(f"{n.name}: shift={shift} r_in={r_in} "
                          f"r_out={r_out} e={shift - r_out + r_in} "
                          f"signed_out={signed_out} "
                          f"max_act={max_act:.4f}")
        elif n.op == "act":
            src = n.args[0]
            r_in = r_out_of.get(src, 0)
            out = vals[n.name]
            mn = float(np.min(out)) if out.size else 0.0
            mx = float(np.max(out)) if out.size else 0.0
            if mn < 0:
                r_out = range_exp(2.0 * max(abs(mn), abs(mx)))
                signed_out = True
            else:
                r_out = range_exp(mx)
                signed_out = False
            n.kwargs["r_in"] = r_in
            n.kwargs["r_out"] = r_out
            n.kwargs["signed_out"] = signed_out
            r_out_of[n.name] = r_out
            report.append(f"{n.name} (act {n.kwargs.get('func')}): "
                          f"r_in={r_in} r_out={r_out} "
                          f"signed_out={signed_out} range=[{mn:.4f},"
                          f"{mx:.4f}]")
        elif n.op in ("residual", "add", "sub", "max_pool2d",
                      "avg_pool2d", "gap", "upsample", "concat",
                      "layer_norm", "bypass", "pad"):
            # These pass through the producer's scale.
            src = n.args[0] if n.args else None
            r_out_of[n.name] = r_out_of.get(src, 0)
    return graph, report
