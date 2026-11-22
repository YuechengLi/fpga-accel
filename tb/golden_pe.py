#!/usr/bin/env python3
"""Golden for pe_mac (Stage 7b PE array).

The PE computes one GEMM tile-step: partial[m][n] =
    (bias[n] if first_k else 0) + sum_k A[m][k] * B[k][n].
Outer mode and dot mode must produce the SAME partial sums (they only
schedule the reduction differently); the golden is mode-independent.

The PE is a PURE GEMM engine: it outputs the raw 32-bit partial sums.
It does NOT chain K-chunks and does NOT apply the output stage --
K-chunk partials for one (M-tile, N-chunk) may run on different PEs,
and the pe_array's merge unit sums them and applies the output stage
(pe_emit) to the final merged sum. The output stage is verified at the
pe_array level (sw/vision/run_pe_srcnn.py), not here.

Layout (matches rtl/common/pe_mac.sv, k-major, contiguous per k):
  a_flat: A[m][k] at byte (k*TM + m).
  b_flat: B[k][n] at int16 (k*TN + n).
  bias_flat: bias[n] at int32 n.
  partial_flat: partial[m][n] at int32 (m*TN + n).

A is 8-bit: sign_in = 1 -> signed (-128..127), 0 -> unsigned (0..255).
B is signed 16-bit. Bias is signed 32-bit. Partial is signed 32-bit.

Usage:
  python3 golden_pe.py gen     # write vectors_pe.txt
  python3 golden_pe.py check   # compare tb/out_pe.txt against the golden
"""
import random
import sys

TM = 16
TN = 8
TK_MAX = 32


def make_cases():
    rnd = random.Random(20261120)
    cases = []
    # (mode_dot, tk, tn, first_k, sign_in)
    specs = [
        (0, 27, TN, 1, 0),   # outer, K-dim 81 (L1-like), unsigned A
        (0, 32, TN, 1, 1),   # outer, K-dim 64 (L2-like), signed A
        (0, 16, TN, 0, 1),   # outer, TK=16, no bias
        (0, 11, TN, 1, 0),   # outer, K-dim 121 (K=11 dense)
        (0, 1, TN, 1, 0),    # outer, TK=1 (K=1)
        (1, 32, TN, 1, 1),   # dot, TK=32 (L3-like), signed A
        (1, 16, TN, 0, 0),   # dot, TK=16, no bias
        (1, 8, TN, 1, 1),    # dot, TK=8 (one tap group)
        (0, 8, TN, 1, 1),    # outer, TK=8
        (1, 32, 1, 1, 1),    # dot, TK=32, tn=1 (SRCNN L3: N=1, 4 cycles)
    ]
    for (mode_dot, tk, tn, first_k, sign_in) in specs:
        if sign_in:
            A = [[rnd.randint(-128, 127) for _ in range(TM)] for _ in range(tk)]
        else:
            A = [[rnd.randint(0, 255) for _ in range(TM)] for _ in range(tk)]
        B = [[rnd.randint(-32768, 32767) for _ in range(TN)] for _ in range(tk)]
        bias = [rnd.randint(-100000, 100000) for _ in range(TN)]
        cases.append(dict(mode_dot=mode_dot, tk=tk, tn=tn,
                          first_k=first_k, sign_in=sign_in, A=A, B=B,
                          bias=bias))
    # Case 10: same A/B/bias as case 1, but dot mode -- outer and dot
    # must agree exactly.
    c1 = cases[1]
    cases.append(dict(mode_dot=1, tk=c1["tk"], tn=TN,
                      first_k=c1["first_k"], sign_in=c1["sign_in"],
                      A=c1["A"], B=c1["B"], bias=c1["bias"]))
    return cases


def golden(case):
    """partial[m][n] = (bias[n] if first_k else 0) + sum_k A[m][k]*B[k][n].

    In dot mode the PE only accumulates the first tn output channels
    (n < tn); the remaining columns keep their init value (bias if
    first_k, else 0).
    """
    tk = case["tk"]
    tn = case["tn"]
    out = []
    for m in range(TM):
        for n in range(TN):
            s = case["bias"][n] if case["first_k"] else 0
            if case["mode_dot"] == 0 or n < tn:
                for k in range(tk):
                    s += case["A"][k][m] * case["B"][k][n]
            out.append(s)
    return out


def write_vectors(path="vectors_pe.txt"):
    cases = make_cases()
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{TM} {TN} {TK_MAX} {c['tk']} {c['tn']} "
                    f"{c['mode_dot']} {c['first_k']} {c['sign_in']}\n")
            for k in range(c["tk"]):
                f.write(" ".join(str(v) for v in c["A"][k]) + "\n")
            for k in range(c["tk"]):
                f.write(" ".join(str(v) for v in c["B"][k]) + "\n")
            f.write(" ".join(str(v) for v in c["bias"]) + "\n")
    print(f"wrote {path}: {len(cases)} cases")


def check(path="out_pe.txt"):
    cases = make_cases()
    try:
        got = [int(x) for x in open(path).read().split()]
    except FileNotFoundError:
        print(f"ERROR: {path} not found")
        return 1
    errors = 0
    idx = 0
    for ci, c in enumerate(cases):
        exp = golden(c)
        n = TM * TN
        vals = got[idx:idx + n]
        idx += n
        if len(vals) < n:
            print(f"case {ci}: MISSING outputs, got {len(vals)}/{n}")
            errors += 1
            continue
        bad = [(i, e, g) for i, (e, g) in enumerate(zip(exp, vals)) if e != g]
        if bad:
            print(f"case {ci} (mode_dot={c['mode_dot']} tk={c['tk']} "
                  f"tn={c['tn']} first_k={c['first_k']} sign_in={c['sign_in']}): "
                  f"{len(bad)}/{n} mismatches, first 5:")
            for (i, e, g) in bad[:5]:
                print(f"  out {i} (m={i // TN}, n={i % TN}): "
                      f"expected {e}, got {g}")
            errors += len(bad)
        else:
            print(f"case {ci} (mode_dot={c['mode_dot']} tk={c['tk']} "
                  f"tn={c['tn']} first_k={c['first_k']} sign_in={c['sign_in']}): "
                  f"PASS ({n} values)")
    if idx < len(got):
        print(f"WARNING: {len(got) - idx} extra values in {path}")
    if errors == 0:
        print("GOLDEN CHECK PASSED")
        return 0
    print(f"GOLDEN CHECK FAILED ({errors} errors)")
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        write_vectors()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(check())
    else:
        write_vectors()
        sys.exit(check())
