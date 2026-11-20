#!/usr/bin/env python3
"""Golden model + vector generator / checker for layer_scheduler.sv (Stage 4).

The scheduler sequences 3 layers through the unified_engine:
  layer 0: dense 3x3 Sobel X, 16x16, pad 1, shift 0, unsigned
           (weights in bank A, 121 coeffs)
  layer 1: DW square 5x5 Gaussian, 16x16, pad 2, shift 8, unsigned
           (weights in bank B, 289 coeffs)
  layer 2: dense 3x3 Sobel X, 16x16, pad 1, shift 0, unsigned
           (different image; weights reloaded into bank A while
           layer 1 runs -- the ping-pong overlap)
The TB runs the sequence twice:
  run 0 (on-time): layer 2's weights are loaded into bank A while
       layer 1 runs, so the scheduler switches immediately (no stall).
  run 1 (late): layer 2's weights are loaded only AFTER layer 1
       finishes, so the scheduler must wait in ST_WAIT_W and still
       produce the correct outputs (it must not start layer 2 with a
       half-loaded bank).
Dense golden mirrors conv2d_dense; DW golden is imported from golden_dw.
Vector format (vectors_sched.txt):
  run_count (2)
  per run: late_flag (0=on-time, 1=late)
  per layer (3 per run):
    mode k shift w h pt pb pl pr sx sy sign_in sign_out fuse_act
    act_slope d ncoeff
    <coeffs, space-separated, ncoeff values>
    <img, space-separated, w*h values>
Output file: out_sched.txt (one byte per line, in order, run 0 then run 1).
"""
import sys
sys.path.insert(0, ".")
import golden_dw as dw
from golden_unified import dense_golden, grid, img_dense

VEC = "vectors_sched.txt"
OUT = "out_sched.txt"

MODE_DENSE = 0
MODE_DW_SQ = 1

g5 = [1, 4, 6, 4, 1,
      4, 16, 24, 16, 4,
      6, 24, 36, 24, 6,
      4, 16, 24, 16, 4,
      1, 4, 6, 4, 1]

def build_layers():
    layers = []
    layers.append(dict(
        mode=MODE_DENSE, k=3, shift=0, w=16, h=16,
        coeff=grid([-1, 0, 1, -2, 0, 2, -1, 0, 1], 3),
        img=img_dense(16, 16, 1),
        pt=1, pb=1, pl=1, pr=1))
    layers.append(dict(
        mode=MODE_DW_SQ, k=5, d=1, shift=8, w=16, h=16,
        coeff=dw.coeff_array(dw.MODE_SQUARE, 5, g5),
        img=img_dense(16, 16, 2),
        pt=2, pb=2, pl=2, pr=2))
    layers.append(dict(
        mode=MODE_DENSE, k=3, shift=0, w=16, h=16,
        coeff=grid([-1, 0, 1, -2, 0, 2, -1, 0, 1], 3),
        img=img_dense(16, 16, 3),
        pt=1, pb=1, pl=1, pr=1))
    return layers

def expected(layer):
    if layer["mode"] == MODE_DENSE:
        return dense_golden(layer["w"], layer["h"], layer["k"],
                            layer["shift"], layer["coeff"], layer["img"],
                            layer.get("pt", 0), layer.get("pb", 0),
                            layer.get("pl", 0), layer.get("pr", 0))
    dcs = dict(mode=dw.MODE_SQUARE, k=layer["k"], d=layer.get("d", 1),
               shift=layer["shift"], coeff=layer["coeff"],
               img=layer["img"], w=layer["w"], h=layer["h"],
               pt=layer.get("pt", 0), pb=layer.get("pb", 0),
               pl=layer.get("pl", 0), pr=layer.get("pr", 0),
               sx=1, sy=1, sign_in=0, sign_out=0)
    return dw.golden(dcs)

def gen():
    layers = build_layers()
    with open(VEC, "w") as f:
        f.write("2\n")
        for late in (0, 1):
            f.write("%d\n" % late)
            for layer in layers:
                f.write("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n" %
                        (layer["mode"], layer["k"], layer["shift"],
                         layer["w"], layer["h"],
                         layer.get("pt", 0), layer.get("pb", 0),
                         layer.get("pl", 0), layer.get("pr", 0),
                         1, 1, 0, 0, 0, 8192, layer.get("d", 1),
                         len(layer["coeff"])))
                f.write(" ".join(str(v) for v in layer["coeff"]) + "\n")
                f.write(" ".join(str(v) for v in layer["img"]) + "\n")
    print("wrote %s (2 runs x 3 layers)" % VEC)

def check():
    layers = build_layers()
    got = [int(x) for x in open(OUT).read().split()]
    pos = 0
    errs = 0
    for run in (0, 1):
        for li, layer in enumerate(layers):
            exp = expected(layer)
            n = len(exp)
            have = got[pos:pos + n]
            pos += n
            if len(have) < n:
                print("run %d layer %d: MISSING outputs: got %d, expected %d" %
                      (run, li, len(have), n))
                errs += n - len(have)
                continue
            bad = 0
            for i, (a, b) in enumerate(zip(have, exp)):
                if a != b:
                    bad += 1
                    if bad <= 3:
                        print("run %d layer %d px %d: got %d expected %d" %
                              (run, li, i, a, b))
            print("run %d layer %d (%s): %s (%d pixels)" %
                  (run, li, "dense" if layer["mode"] == MODE_DENSE else "DW-sq",
                   "PASS" if bad == 0 else "FAIL (%d bad)" % bad, n))
            errs += bad
    if pos < len(got):
        print("WARNING: %d extra outputs" % (len(got) - pos))
    if errs == 0:
        print("GOLDEN CHECK PASSED")
    else:
        print("GOLDEN CHECK FAILED (%d errors)" % errs)
    return 1 if errs else 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        gen()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(check())
    else:
        print("usage: golden_sched.py gen|check")
