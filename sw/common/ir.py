#!/usr/bin/env python3
"""Stage 5 compiler IR + capture frontend.

The compiler IR is a minimal FX-graph-like representation: an ordered list
of Nodes (one op each) with named inputs/outputs. It is deliberately small
-- it is NOT torch.fx itself -- so the passes, the cost model and the
functional simulator can run without a torch dependency (unit tests use the
hand-built builder). When torch IS available, ``capture_torch`` lowers a
real ``torch.export`` (AOT, TorchDynamo) FX graph into this IR, so the
compiler consumes the PyTorch compile graph as the plan requires.

Node ops (all the accelerator supports, see docs/plan.md Stage 5):
  conv2d, dw_conv2d (depthwise, incl. strip/dilated via kwargs),
  linear (FC / 1x1), relu, leaky_relu, sigmoid, tanh, gelu, silu,
  hard_sigmoid, hard_tanh (cheap substitutes, see passes/actapprox.py),
  add, sub, mul (residual / elementwise / SimpleGate),
  max_pool2d, avg_pool2d, gap (global average pooling),
  upsample (nearest/bilinear/pixel-shuffle via kwargs),
  concat, layer_norm, act (a complex activation routed through the
  separate LUT act module -- the fusion pass rewrites sigmoid/tanh/
  gelu/silu into ``act`` with a generated table), bypass (no-op).

Weight layout (matches models/vision/common.py, the Stage 6 golden):
  conv weight: (KH, KW, C_in, C_out) float32; bias: (C_out,) float32.
  torch's (C_out, C_in, KH, KW) is transposed on capture.
"""
import numpy as np


class Node:
    """One IR node. ``args`` are node names (data edges); ``kwargs`` are
    static attributes (K, stride, pad, ...); ``params`` holds tensors
    (weight/bias) as numpy arrays for conv/linear/batch_norm nodes."""

    def __init__(self, op, name, args=(), kwargs=None, params=None):
        self.op = op
        self.name = name
        self.args = list(args)
        self.kwargs = dict(kwargs or {})
        self.params = dict(params or {})

    def __repr__(self):
        return f"Node({self.op} {self.name} args={self.args} kw={self.kwargs})"


class Graph:
    """Ordered list of nodes; ``inputs``/``outputs`` are node names."""

    def __init__(self, name="graph"):
        self.name = name
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self._by_name = {}

    def add(self, node):
        assert node.name not in self._by_name, f"duplicate node {node.name}"
        self.nodes.append(node)
        self._by_name[node.name] = node
        return node

    def __getitem__(self, name):
        return self._by_name[name]

    def __contains__(self, name):
        return name in self._by_name

    def producers(self, name):
        """Nodes that consume ``name``."""
        return [n for n in self.nodes if name in n.args]

    def find(self, op):
        return [n for n in self.nodes if n.op == op]

    def summary(self):
        lines = [f"graph {self.name}: {len(self.nodes)} nodes"]
        for n in self.nodes:
            lines.append(f"  {n.op:12s} {n.name:20s} args={n.args} {n.kwargs}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hand-built builder (no torch needed; used by unit tests and by the
# model definitions in models_torch.py when torch is unavailable).
# ---------------------------------------------------------------------------
def build_graph(name, layers):
    """Build a Graph from a list of layer dicts.

    Each dict: {"op": ..., "name": ..., "args": [...], "kwargs": {...},
    "params": {...}}. ``args`` default to the previous node's name
    (a linear chain) when omitted.
    """
    g = Graph(name)
    prev = None
    for i, layer in enumerate(layers):
        op = layer["op"]
        nm = layer.get("name", f"{op}_{i}")
        args = layer.get("args")
        if args is None:
            args = [prev] if prev is not None else []
        node = Node(op, nm, args=args, kwargs=layer.get("kwargs"),
                    params=layer.get("params"))
        g.add(node)
        if prev is None:
            g.inputs = [nm] if op != "input" else []
        prev = nm
    if prev is not None:
        g.outputs = [prev]
    return g


def input_node(name="input"):
    return Node("input", name)


# ---------------------------------------------------------------------------
# Torch frontend: torch.export (AOT) -> IR.
# ---------------------------------------------------------------------------
_TORCH_OPS = {
    # aten target (str) -> IR op
    # torch 2.15 export decomposes nn.Conv2d to aten.conv2d (not
    # aten.convolution); both are accepted.
    "convolution": "conv2d",
    "conv2d": "conv2d",
    "pad": "pad",
    "relu": "relu",
    "leaky_relu": "leaky_relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "gelu": "gelu",
    "silu": "silu",
    "hardsigmoid": "hard_sigmoid",
    "hardtanh": "hard_tanh",
    "add.Tensor": "add",
    "sub.Tensor": "sub",
    "mul.Tensor": "mul",
    "max_pool2d": "max_pool2d",
    "avg_pool2d": "avg_pool2d",
    "mean": "gap",
    "linear": "linear",
    "upsample_nearest2d": "upsample",
    "upsample_bilinear2d": "upsample",
    "cat": "concat",
    "layer_norm": "layer_norm",
}


def capture_torch(model, example_input, name="torch_model"):
    """Capture a torch.nn.Module with torch.export (AOT, TorchDynamo
    FX graph) and lower it to the compiler IR.

    torch 2.15 lifts parameters to placeholder nodes (e.g.
    ``p_net_0_weight``) whose names are keys of ``ep.state_dict``;
    the model input is the placeholder that is NOT a parameter.
    ``aten.conv2d`` args are (input, weight, bias, stride, padding,
    dilation, groups) positionally, with defaults. ``aten.pad`` is
    F.pad (replicate/constant) -- lowered to a ``pad`` IR node that
    the fusion pass folds into the following conv's pad_mode.

    Returns (graph, state_dict). Raises RuntimeError with a clear
    message if the model cannot be exported -- the caller must not
    silently fall back to a hand-built graph for a model the user
    asked to capture.
    """
    try:
        import torch
        import torch.fx as fx
    except ImportError as e:
        raise RuntimeError("torch is not installed; cannot capture a "
                           "torch model (use build_graph for a hand-built "
                           "IR instead)") from e
    try:
        ep = torch.export.export(model, (example_input,))
    except Exception as e:
        raise RuntimeError(f"torch.export failed for {name}: {e}") from e

    state = {k: v.detach().cpu().numpy() for k, v in ep.state_dict.items()}
    # Lifted parameters: placeholder name -> state_dict key
    # (e.g. p_net_0_weight -> net.0.weight).
    ph_to_state = dict(ep.graph_signature.inputs_to_parameters)
    g = Graph(name)
    fx_to_ir = {}

    def _is_param(node):
        return isinstance(node, fx.Node) and node.op == "placeholder" \
            and node.name in ph_to_state

    def _arg_name(arg):
        """IR name for a data-edge FX node arg."""
        if isinstance(arg, fx.Node):
            if _is_param(arg):
                raise RuntimeError("capture: parameter node used as a "
                                   "data edge (node "
                                   f"{arg.name}); this should have been "
                                   "resolved as a weight/bias")
            return fx_to_ir.get(arg.name, arg.name)
        return arg

    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in ph_to_state:
                continue  # lifted parameter, resolved by consumers
            ir_name = f"input_{node.name}"
            fx_to_ir[node.name] = ir_name
            g.add(Node("input", ir_name))
            g.inputs.append(ir_name)
        elif node.op == "get_attr":
            continue  # resolved inline by consumers
        elif node.op == "output":
            out = node.args[0]
            if isinstance(out, (list, tuple)):
                g.outputs = [fx_to_ir.get(a.name, a.name)
                             for a in out if hasattr(a, "name")]
            elif hasattr(out, "name"):
                g.outputs = [fx_to_ir.get(out.name, out.name)]
        elif node.op == "call_function":
            target = str(node.target)
            short = target
            for pfx in ("torch.ops.aten.", "aten."):
                if short.startswith(pfx):
                    short = short[len(pfx):]
                    break
            short = short.replace(".default", "")
            if short not in _TORCH_OPS:
                raise RuntimeError(
                    f"capture: unsupported aten op {target} "
                    f"(node {node.name}); add a lowering in ir.py "
                    f"or rewrite the model to use a supported op")
            ir_op = _TORCH_OPS[short]
            ir_name = node.name
            fx_to_ir[node.name] = ir_name
            _lower_aten(g, ir_name, ir_op, node, state, _arg_name,
                        ph_to_state)
        else:
            raise RuntimeError(f"capture: unsupported FX node op "
                               f"{node.op} (node {node.name})")
    return g, state


def _param_array(arg, state, what, ph_to_state=None):
    """Resolve a weight/bias FX arg (a lifted-parameter placeholder
    node, a get_attr node, or a raw tensor) to a numpy array."""
    import torch.fx as fx
    if isinstance(arg, fx.Node):
        if arg.op == "placeholder" and ph_to_state is not None \
                and arg.name in ph_to_state:
            return state[ph_to_state[arg.name]]
        if arg.op == "placeholder" and arg.name in state:
            return state[arg.name]
        if arg.op == "get_attr" and arg.target in state:
            return state[arg.target]
        raise RuntimeError(f"capture: cannot resolve {what} "
                           f"(node {arg.name}, op {arg.op})")
    if hasattr(arg, "detach"):
        return arg.detach().cpu().numpy()
    return None


def _lower_aten(g, ir_name, ir_op, node, state, _arg_name,
                ph_to_state=None):
    """Lower one aten call to an IR node."""
    raw = list(node.args)
    if ir_op == "conv2d":
        # aten.conv2d(input, weight, bias, stride=1, padding=0,
        #   dilation=1, groups=1) -- positionally, defaults omitted.
        w = _param_array(raw[1], state, f"conv {ir_name} weight",
                         ph_to_state)
        b = None
        if len(raw) > 2 and raw[2] is not None:
            b = _param_array(raw[2], state, f"conv {ir_name} bias",
                             ph_to_state)
        stride = _pair(raw[3]) if len(raw) > 3 else (1, 1)
        pad = _pair(raw[4]) if len(raw) > 4 else (0, 0)
        dil = _pair(raw[5]) if len(raw) > 5 else (1, 1)
        groups = int(raw[6]) if len(raw) > 6 else 1
        # torch (Cout, Cin, KH, KW) -> golden (KH, KW, Cin, Cout)
        w_g = np.transpose(np.asarray(w, dtype=np.float32), (2, 3, 1, 0))
        p = {"weight": w_g}
        p["bias"] = (np.asarray(b, dtype=np.float32).reshape(-1)
                     if b is not None
                     else np.zeros(w_g.shape[3], dtype=np.float32))
        kh, kw = w_g.shape[0], w_g.shape[1]
        cin, cout = w_g.shape[2], w_g.shape[3]
        if groups != 1:
            if groups == cin == cout and cin > 1:
                ir_op = "dw_conv2d"
            else:
                raise RuntimeError(
                    f"capture: grouped conv (groups={groups}) is not "
                    f"supported by the accelerator")
        kwarg = {"K": kh, "C_in": cin, "C_out": cout,
                 "stride": stride, "pad": pad, "dilation": dil}
        g.add(Node(ir_op, ir_name, args=[_arg_name(raw[0])],
                   kwargs=kwarg, params=p))
        return
    if ir_op == "linear":
        w = _param_array(raw[1], state, f"linear {ir_name} weight",
                         ph_to_state)
        b = None
        if len(raw) > 2 and raw[2] is not None:
            b = _param_array(raw[2], state, f"linear {ir_name} bias",
                             ph_to_state)
        # torch linear weight (Cout, Cin); golden layout (1,1,Cin,Cout)
        w_g = np.transpose(np.asarray(w, dtype=np.float32), (1, 0))
        w_g = w_g.reshape(1, 1, w_g.shape[0], w_g.shape[1])
        p = {"weight": w_g}
        p["bias"] = (np.asarray(b, dtype=np.float32).reshape(-1)
                     if b is not None
                     else np.zeros(w_g.shape[3], dtype=np.float32))
        g.add(Node("linear", ir_name, args=[_arg_name(raw[0])],
                   kwargs={"C_in": w_g.shape[2], "C_out": w_g.shape[3]},
                   params=p))
        return
    if ir_op == "pad":
        # aten.pad(input, [left, right, top, bottom], mode, value?)
        # golden pad order is (top, bottom, left, right).
        pl = list(raw[1])
        mode = raw[2] if len(raw) > 2 else "constant"
        if mode not in ("replicate", "constant"):
            raise RuntimeError(f"capture: pad mode {mode} is not "
                               f"supported (use replicate or constant)")
        pad = (int(pl[2]), int(pl[3]), int(pl[0]), int(pl[1]))
        g.add(Node("pad", ir_name, args=[_arg_name(raw[0])],
                   kwargs={"pad": pad, "mode": mode}))
        return
    if ir_op in ("relu", "sigmoid", "tanh", "gelu", "silu",
                 "hard_sigmoid", "hard_tanh"):
        g.add(Node(ir_op, ir_name, args=[_arg_name(raw[0])]))
        return
    if ir_op == "leaky_relu":
        slope = 0.01
        if len(raw) > 1 and isinstance(raw[1], (int, float)):
            slope = float(raw[1])
        g.add(Node(ir_op, ir_name, args=[_arg_name(raw[0])],
                   kwargs={"negative_slope": slope}))
        return
    if ir_op in ("add", "sub", "mul"):
        g.add(Node(ir_op, ir_name,
                   args=[_arg_name(a) for a in raw
                         if isinstance(a, str) or hasattr(a, "name")]))
        return
    if ir_op in ("max_pool2d", "avg_pool2d"):
        k = _pair(raw[1]) if len(raw) > 1 else (2, 2)
        s = _pair(raw[2]) if len(raw) > 2 else k
        g.add(Node(ir_op, ir_name, args=[_arg_name(raw[0])],
                   kwargs={"K": k, "stride": s}))
        return
    if ir_op == "gap":
        g.add(Node("gap", ir_name, args=[_arg_name(raw[0])]))
        return
    if ir_op == "upsample":
        scale = None
        if len(raw) > 2 and raw[2] is not None:
            scale = raw[2]
        elif "scale_factor" in (node.kwargs or {}):
            scale = node.kwargs["scale_factor"]
        mode = ("nearest" if "nearest" in str(node.target)
                else "bilinear")
        g.add(Node("upsample", ir_name, args=[_arg_name(raw[0])],
                   kwargs={"scale": scale, "mode": mode}))
        return
    if ir_op == "concat":
        g.add(Node("concat", ir_name,
                   args=[_arg_name(a) for a in raw
                         if isinstance(a, str) or hasattr(a, "name")],
                   kwargs={"dim": (node.kwargs or {}).get("dim", 1)}))
        return
    if ir_op == "layer_norm":
        g.add(Node("layer_norm", ir_name, args=[_arg_name(raw[0])]))
        return
    raise RuntimeError(f"capture: no lowering for IR op {ir_op}")


def _pair(v):
    """Normalize an int or (h, w) to a (h, w) tuple."""
    if isinstance(v, (list, tuple)):
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))
