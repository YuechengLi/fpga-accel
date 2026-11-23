#!/usr/bin/env python3
"""Stage 7d compiler pass: SiLU decomposition.

Rewrites every ``act`` node with func == "silu" into

    act(sigmoid, r_out=2)  ->  simplegate(shift, sign_in=1, sign_out=0)

i.e. SiLU(x) = x * sigmoid(x), with the sigmoid through the LUT
module (act.sv) and the multiply through simplegate.sv.

Why (design.md): the MLUT/PWL tables cover x in [-8, 8) only, and
the YOLO pre-SiLU conv outputs reach 35 (5/57 layers exceed 8).
A direct SiLU LUT would saturate those layers (SiLU(35) returned
as SiLU(8) ~ 8). Sigmoid saturates by nature, so it is safe in
the table; the multiply restores the linear tail.

The halved-sigmoid trick: simplegate.sv has ONE sign bit for both
inputs, so a signed x_q cannot be multiplied by an unsigned
sig_q (sig_q > 127 would read as negative). The sigmoid act
node's r_out is therefore 2, so the LUT emits sig' = sig * 255/4
in [0, 64] (fits signed int8 with margin); the simplegate then
computes (sig' * x_q) >> shift with sign_in=1. shift =
r_out_silu - r_in_conv + 6, where r_out_silu is the act node's
r_out (the post-SiLU range) and r_in_conv is the producer conv's
r_out (the pre-act range). The arithmetic is
(sig_q/4 * x_q)/2**shift against the ideal sig_q*x_q/255 -- a
255-vs-256 scale difference (~0.4%, at most ~1 LSB), a
quantization error vs float, not an RTL-vs-funcsim error (the
funcsim simplegate case implements the identical integer
arithmetic). r_out=2 (not 1): r_out=1 would emit sig' = sig *
127.5, which rounds to 128 at sig ~ 1 -- and 128 as a signed
byte is -128, which would flip the sign of the largest SiLU
values (exactly the layers that need the decomposition).

Requires the graph to be fused and quantized (the act node
carries r_in/r_out and its producer conv carries r_out). The
simplegate reuses the act node's name, so consumers are
unchanged; the sigmoid node takes a fresh name
(<act>_sig). The pass runs after fusion and before the
actapprox table generation (so the sigmoid tables are generated
for the decomposed act nodes, not for SiLU).
"""
from ir import Graph, Node


def decompose_silu(graph):
    """Rewrite act(silu) nodes. Returns (new_graph, report).

    Raises RuntimeError if an act(silu) node lacks the
    quantization (r_in/r_out) the shift needs, or if the shift
    falls outside [0, 15] (simplegate.sv's shift range).
    """
    nodes = []
    report = []
    for n in graph.nodes:
        if n.op == "act" and n.kwargs.get("func") == "silu":
            conv_name = n.args[0]
            conv = graph[conv_name]
            r_in_conv = conv.kwargs.get("r_out")
            r_out_silu = n.kwargs.get("r_out")
            if r_in_conv is None or r_out_silu is None:
                raise RuntimeError(
                    f"decompose_silu: {n.name} (or its producer "
                    f"{conv_name}) is not quantized (r_in/r_out "
                    f"missing) -- run the quantize override first")
            r_in_conv = int(r_in_conv)
            r_out_silu = int(r_out_silu)
            shift = r_out_silu - r_in_conv + 6
            if not (0 <= shift <= 15):
                raise RuntimeError(
                    f"decompose_silu: {n.name} shift={shift} out of "
                    f"[0, 15] (r_out_silu={r_out_silu}, "
                    f"r_in_conv={r_in_conv})")
            sig = Node("act", f"{n.name}_sig", args=[conv_name],
                       kwargs={"func": "sigmoid",
                               "r_in": r_in_conv, "r_out": 2,
                               "signed_out": False})
            sg = Node("simplegate", n.name, args=[sig.name, conv_name],
                      kwargs={"shift": shift, "sign_in": True,
                              "sign_out": False,
                              "r_in": r_in_conv, "r_out": r_out_silu,
                              "signed_out": False})
            nodes.append(sig)
            nodes.append(sg)
            report.append(f"{n.name}: silu -> sigmoid(r_out=2) + "
                          f"simplegate(shift={shift}, sign_in=1) "
                          f"[r_in_conv={r_in_conv} r_out={r_out_silu}]")
        else:
            nodes.append(n)
    g = Graph(graph.name + "_silu")
    g.inputs = list(graph.inputs)
    for n in nodes:
        g.add(n)
    g.outputs = list(graph.outputs)
    return g, report
