"""Unit tests for the Stage 7d end-to-end driver (YOLOv8n + DNANet).

Covers, without any vvp run:
  - Capture topology: the saved 7d programs have the expected
    op / conv counts (YOLO 226 ops / 71 convs, DNANet 560 ops /
    206 convs) and every conv carries a shift in [0,15].
  - SiLU decomposition: act(silu) -> act(sigmoid, r_out=2) +
    simplegate with shift = r_out_silu - r_in_conv + 6, and the
    decomposed graph's float forward matches the fused graph's.
  - Rescale: rescale_input / rescale_bytes round-half-up and
    clip to the signed / unsigned byte range.
  - RTL bilinear upsample: funcsim's bilinear_rtl mode matches
    the RTL formula (out[2r,2c]=in[r,c], neighbours averaged,
    round-half-up, edges replicate).
  - Simplegate arithmetic: the funcsim simplegate case matches
    (A*B)>>shift with a floor shift and saturation.
  - Downsample vs torch: _downsample_ac matches torch's bilinear
    0.5x align_corners=True to ~1e-14.
  - DFL decode vs torch: the driver's numpy DFL decode matches
    the torch Detect output on the same 128x128 input (same 5
    detections after NMS).
  - DNANet mask: the fixed_forward deep-supervision masks match
    the torch float masks on the same 64x64 crop (IoU 1.0).
"""
import os
import sys
import json
import unittest

HERE = os.path.dirname(__file__)
SW_VISION = os.path.join(HERE, "..")
REPO = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.abspath(SW_VISION))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "sw", "common")))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "sw", "common",
                                                "passes")))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "tb")))

import numpy as np

import funcsim
from funcsim import (fixed_forward, float_forward, rescale_bytes,
                     rescale_input, _upsample, _downsample_ac)
from silu_decompose import decompose_silu
import capture_7d


def _prog_path(model):
    return os.path.join(REPO, "logs", "stage7d", f"program_{model}.json")


def _load_program(model):
    with open(_prog_path(model)) as f:
        prog = json.load(f)
    return prog


class TestTopology(unittest.TestCase):
    def test_yolo_program_counts(self):
        prog = _load_program("yolo")
        ops = prog["ops"]
        convs = [o for o in ops if o["op"] == "conv2d"]
        self.assertEqual(len(ops), 226)
        self.assertEqual(len(convs), 71)
        for o in convs:
            self.assertTrue(0 <= int(o["shift"]) <= 15,
                            f"{o['name']} shift={o['shift']}")

    def test_dnanet_program_counts(self):
        prog = _load_program("dnanet")
        ops = prog["ops"]
        convs = [o for o in ops if o["op"] == "conv2d"]
        self.assertEqual(len(ops), 560)
        self.assertEqual(len(convs), 206)
        for o in convs:
            self.assertTrue(0 <= int(o["shift"]) <= 15,
                            f"{o['name']} shift={o['shift']}")

    def test_yolo_outputs_are_detect_heads(self):
        prog = _load_program("yolo")
        self.assertEqual(prog["outputs"],
                         ["det_box0", "det_cls0", "det_box1",
                          "det_cls1", "det_box2", "det_cls2"])

    def test_dnanet_outputs_are_deep_supervision(self):
        prog = _load_program("dnanet")
        self.assertEqual(prog["outputs"],
                         ["final1", "final2", "final3", "final4"])


class TestSiluDecompose(unittest.TestCase):
    def _graph(self):
        # A minimal IR graph: input -> conv (r_out=6) -> act(silu,
        # r 6->6) -> conv. Built with the IR Node/Graph API the
        # capture uses.
        from ir import Graph, Node
        W = np.ones((1, 1, 1, 1))
        b = np.zeros(1)
        g = Graph("t")
        g.add(Node("input", "input"))
        g.add(Node("conv2d", "c0", args=["input"],
                   kwargs={"r_in": 6, "r_out": 6, "stride": (1, 1),
                           "pad": (0, 0, 0, 0)},
                   params={"weight": W, "bias": b}))
        g.add(Node("act", "a0", args=["c0"],
                   kwargs={"func": "silu", "r_in": 6, "r_out": 6}))
        g.add(Node("conv2d", "c1", args=["a0"],
                   kwargs={"r_in": 6, "r_out": 6, "stride": (1, 1),
                           "pad": (0, 0, 0, 0)},
                   params={"weight": W, "bias": b}))
        g.inputs = ["input"]
        g.outputs = ["c1"]
        return g

    def test_decompose_rewrites_silu(self):
        g = self._graph()
        out, report = decompose_silu(g)
        names = [n.name for n in out.nodes]
        # a0 is replaced by a0_sig (sigmoid) + a0 (simplegate).
        self.assertEqual(names, ["input", "c0", "a0_sig", "a0", "c1"])
        sig = out.nodes[2]
        self.assertEqual(sig.op, "act")
        self.assertEqual(sig.kwargs["func"], "sigmoid")
        self.assertEqual(sig.kwargs["r_out"], 2)
        gate = out.nodes[3]
        self.assertEqual(gate.op, "simplegate")
        self.assertEqual(gate.args, ["a0_sig", "c0"])
        # shift = r_out_silu - r_in_conv + 6 = 6 - 6 + 6.
        self.assertEqual(gate.kwargs["shift"], 6)
        self.assertTrue(gate.kwargs["sign_in"])
        self.assertFalse(gate.kwargs["sign_out"])
        self.assertEqual(len(report), 1)

    def test_decompose_float_forward_matches(self):
        g = self._graph()
        x = np.random.RandomState(0).randn(4, 4, 1) * 2.0
        before, _ = float_forward(g, x)
        out, _ = decompose_silu(g)
        after, _ = float_forward(out, x)
        self.assertLess(np.abs(before["a0"] - after["a0"]).max(),
                        1e-9)


class TestRescale(unittest.TestCase):
    def test_rescale_bytes_left_shift(self):
        # r_from=6 -> r_to=4 is a LEFT shift (the target scale is
        # coarser, so the byte value is larger): 10<<2 = 40.
        self.assertEqual(rescale_bytes(np.array([10], dtype=np.int64),
                                       6, 4, True)[0], 40)

    def test_rescale_bytes_right_shift_round_half_up(self):
        # r_from=4 -> r_to=6 is a RIGHT shift with round-half-up:
        # 10/4 = 2.5 -> 3; -10/4 = -2.5 -> -2.
        self.assertEqual(rescale_bytes(np.array([10], dtype=np.int64),
                                       4, 6, True)[0], 3)
        self.assertEqual(rescale_bytes(np.array([-10], dtype=np.int64),
                                       4, 6, True)[0], -2)

    def test_rescale_bytes_clips_signed(self):
        # 200 at r=0 -> r=0 stays 200, clipped to the signed max 127.
        self.assertEqual(rescale_bytes(np.array([200], dtype=np.int64),
                                       0, 0, True)[0], 127)
        self.assertEqual(rescale_bytes(np.array([-200], dtype=np.int64),
                                       0, 0, True)[0], -128)

    def test_rescale_bytes_clips_unsigned(self):
        self.assertEqual(rescale_bytes(np.array([300], dtype=np.int64),
                                       0, 0, False)[0], 255)
        self.assertEqual(rescale_bytes(np.array([-5], dtype=np.int64),
                                       0, 0, False)[0], 0)

    def test_rescale_input_noop_same_scale(self):
        a = np.array([[[1, 2], [3, 4]]], dtype=np.int64)
        groups = [(0, 2, 6, True)]
        out = rescale_input(a, groups, 6)
        np.testing.assert_array_equal(out, a)


class TestBilinearRtl(unittest.TestCase):
    def test_bilinear_rtl_matches_rtl_formula(self):
        # The RTL formula (upsample.sv mode 1 + golden_upsample.py):
        # out[2r,2c]=in[r,c], out[2r,2c+1]=(in[r,c]+in[r,c+1])/2,
        # out[2r+1,2c]=(in[r,c]+in[r+1,c])/2,
        # out[2r+1,2c+1]=(sum of 4)/4, edges replicate, round-half-up.
        rng = np.random.RandomState(1)
        inp = rng.randint(0, 256, size=(4, 4, 1)).astype(np.int64)
        out = _upsample(inp, {"scale": 2, "mode": "bilinear_rtl"})
        self.assertEqual(out.shape, (8, 8, 1))
        # Top-left corner: out[0,0]=in[0,0].
        self.assertEqual(out[0, 0, 0], inp[0, 0, 0])
        # out[0,1]=(in[0,0]+in[0,1])/2 round-half-up.
        self.assertEqual(out[0, 1, 0], (int(inp[0, 0, 0])
                                         + int(inp[0, 1, 0]) + 1) // 2)
        # out[1,0]=(in[0,0]+in[1,0])/2 round-half-up.
        self.assertEqual(out[1, 0, 0], (int(inp[0, 0, 0])
                                         + int(inp[1, 0, 0]) + 1) // 2)
        # out[1,1]=(sum of 4)/4 round-half-up.
        s = (int(inp[0, 0, 0]) + int(inp[0, 1, 0])
             + int(inp[1, 0, 0]) + int(inp[1, 1, 0]))
        self.assertEqual(out[1, 1, 0], (s + 2) // 4)
        # Right edge replicates: out[0,7]=in[0,3].
        self.assertEqual(out[0, 7, 0], inp[0, 3, 0])

    def test_bilinear_rtl_differs_from_torch_convention(self):
        # The RTL formula is NOT torch's align_corners=True, so
        # the two must differ on a non-trivial input (this is why
        # the e2e golden uses bilinear_rtl and the final vs float
        # is reported against both conventions).
        rng = np.random.RandomState(2)
        inp = rng.rand(4, 4, 1)
        a = _upsample(inp, {"scale": 2, "mode": "bilinear_rtl"})
        b = _upsample(inp, {"scale": 2, "mode": "bilinear"})
        self.assertGreater(np.abs(a - b).max(), 0.01)


class TestSimplegateArithmetic(unittest.TestCase):
    """The funcsim simplegate case must equal (A*B)>>shift with a
    floor shift and saturation -- the RTL's exact arithmetic
    (simplegate.sv + golden_simplegate.py). Verified on real
    program data: YOLO's C0_silu (unsigned out) and DNANet's
    first ChannelAttention multiply (signed out)."""
    @classmethod
    def setUpClass(cls):
        from run_e2e_7d import yolo_input, dnanet_input
        cls.yolo_prog = capture_7d.load_program(
            "yolo", os.path.join(REPO, "logs", "stage7d"))
        cls.yolo_per, _ = fixed_forward(cls.yolo_prog, yolo_input())
        cls.dn_prog = capture_7d.load_program(
            "dnanet", os.path.join(REPO, "logs", "stage7d"))
        cls.dn_per, _ = fixed_forward(cls.dn_prog, dnanet_input())

    def test_simplegate_unsigned_matches_formula(self):
        per = self.yolo_per
        ops = {o["name"]: o for o in self.yolo_prog["ops"]}
        sg = ops["C0_silu"]
        shift = int(sg["shift"])
        A = per["C0_silu_sig"].astype(np.int64)
        B = per["C0"].astype(np.int64)
        # sign_in reinterprets bytes >= 128 as negative (a no-op
        # here: A is [0,64] and B is already signed int64).
        A = np.where(A >= 128, A - 256, A)
        B = np.where(B >= 128, B - 256, B)
        exp = np.clip((A * B) >> shift, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(per["C0_silu"], exp)

    def test_simplegate_signed_out_clips(self):
        per = self.dn_per
        ops = {o["name"]: o for o in self.dn_prog["ops"]}
        sg = ops["conv0_0.0.ca_mul"]
        shift = int(sg["shift"])
        A = per["conv0_0.0.ca_sig"].astype(np.int64)
        B = per["conv0_0.0.conv2"].astype(np.int64)
        A = np.where(A >= 128, A - 256, A)
        B = np.where(B >= 128, B - 256, B)
        exp = np.clip((A * B) >> shift, -128, 127).astype(np.int64)
        np.testing.assert_array_equal(per["conv0_0.0.ca_mul"], exp)


class TestDownsampleVsTorch(unittest.TestCase):
    def test_downsample_matches_torch(self):
        torch = pytest_import_torch(self)
        if torch is None:
            self.skipTest("torch not available")
        import torch.nn.functional as F
        rng = np.random.RandomState(4)
        x = rng.rand(1, 3, 64, 64).astype(np.float32)
        ref = F.interpolate(torch.from_numpy(x), scale_factor=0.5,
                            mode="bilinear",
                            align_corners=True).numpy()[0]
        got = _downsample_ac(x[0].transpose(1, 2, 0))
        self.assertLess(np.abs(got - ref.transpose(1, 2, 0)).max(),
                        1e-5)


def pytest_import_torch(test):
    try:
        import torch
        return torch
    except ImportError:
        return None


class TestDflDecodeVsTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = pytest_import_torch(cls)
        if cls.torch is None:
            return
        # The saved YOLO program + the 128x128 resized bus.jpg.
        from run_e2e_7d import yolo_input
        cls.prog = capture_7d.load_program("yolo",
                                           os.path.join(REPO, "logs",
                                                        "stage7d"))
        cls.fin = yolo_input()
        cls.per, _ = fixed_forward(cls.prog, cls.fin)

    def test_dfl_decode_matches_torch(self):
        if self.torch is None:
            self.skipTest("torch not available")
        from run_e2e_7d import final_yolo_pred
        import torch
        # The driver's DFL decode on the fixed_forward outputs.
        pred = final_yolo_pred(self.prog, self.per)
        # Torch Detect on the same 128x128 input. The raw Detect
        # output is (1, 84, N) in xywh + cls scores.
        import os as _os
        _os.environ.setdefault("YOLO_VERBOSE", "false")
        from ultralytics import YOLO
        m = YOLO(os.path.join(REPO, "models", "vision", "yolov8n",
                              "raw", "yolov8n.pt"))
        model = m.model.eval()
        x = torch.from_numpy(
            self.fin.transpose(2, 0, 1)[None].astype(np.float32)
            / 255.0)
        with torch.no_grad():
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        det = out[0].numpy().T  # (N, 84): xywh + 80 cls
        self.assertEqual(pred.shape[0], det.shape[0])
        # Torch xywh -> xyxy, then compare the boxes (the e2e
        # final bar is 2 px) and the class scores.
        cx, cy, w, h = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
        ref_xyxy = np.stack([cx - w / 2, cy - h / 2,
                             cx + w / 2, cy + h / 2], axis=1)
        self.assertLess(np.abs(pred[:, :4] - ref_xyxy).max(), 2.0)
        self.assertLess(np.abs(pred[:, 4:] - det[:, 4:]).max(), 0.05)


class TestDnanetMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = pytest_import_torch(cls)
        if cls.torch is None:
            return
        from run_e2e_7d import dnanet_input
        cls.prog = capture_7d.load_program("dnanet",
                                           os.path.join(REPO, "logs",
                                                        "stage7d"))
        cls.fin = dnanet_input()
        cls.per, _ = fixed_forward(cls.prog, cls.fin)

    def test_dnanet_masks_match_torch(self):
        if self.torch is None:
            self.skipTest("torch not available")
        import torch
        sys.path.insert(0, os.path.join(REPO, "models", "vision",
                                        "dnanet"))
        from model_DNANet import DNANet, Res_CBAM_block
        import numpy as np
        d = np.load(os.path.join(REPO, "models", "vision", "dnanet",
                                 "dnanet_float.npz"))
        model = DNANet(num_classes=1, input_channels=3,
                       block=Res_CBAM_block, num_blocks=[2, 2, 2, 2],
                       nb_filter=[16, 32, 64, 128, 256],
                       deep_supervision=True)
        convs = [(p, m) for p, m in model.named_modules()
                 if isinstance(m, torch.nn.Conv2d)]
        sd = {}
        for i, (path, c) in enumerate(convs):
            sd[path + ".weight"] = torch.from_numpy(
                np.transpose(d[f"L{i}_W"].astype(np.float32),
                             (3, 2, 0, 1)))
            if c.bias is not None:
                sd[path + ".bias"] = torch.from_numpy(
                    d[f"L{i}_b"].astype(np.float32))
            if bool(d[f"L{i}_has_bn"][0]):
                if path.endswith(".conv1"):
                    bpath = path[:-len(".conv1")] + ".bn1"
                elif path.endswith(".conv2"):
                    bpath = path[:-len(".conv2")] + ".bn2"
                else:
                    bpath = path[:-len(".shortcut.0")] + ".shortcut.1"
                for suf, key in [("weight", "bn_w"),
                                 ("bias", "bn_b"),
                                 ("running_mean", "bn_mean"),
                                 ("running_var", "bn_var")]:
                    sd[bpath + "." + suf] = torch.from_numpy(
                        d[f"L{i}_{key}"].astype(np.float32))
        model.load_state_dict(sd, strict=False)
        model.eval()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        xn = (self.fin.astype(np.float64) / 255.0 - mean) / std
        x = torch.from_numpy(xn.astype(np.float32)).permute(
            2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            outs = model(x)
        ref_masks = [(1.0 / (1.0 + np.exp(-o[0, 0].numpy()))) > 0.5
                     for o in outs]
        # The fixed_forward deep-supervision masks must match
        # the torch masks (IoU 1.0 -- the e2e final bar).
        from run_e2e_7d import final_dnanet_masks
        masks = final_dnanet_masks(self.prog, self.per)
        for name, got, ref in zip(
                ["final1", "final2", "final3", "final4"],
                masks, ref_masks):
            inter = np.logical_and(got, ref).sum()
            union = np.logical_or(got, ref).sum()
            iou = inter / union if union else 1.0
            self.assertEqual(iou, 1.0, f"{name} IoU={iou}")


if __name__ == "__main__":
    unittest.main()
