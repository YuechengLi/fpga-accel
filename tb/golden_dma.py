#!/usr/bin/env python3
"""Golden for the weight DMA (tb/tb_weight_dma.sv).

Generates the expected config-bus writes (addr, data) for a DMA
transfer and the expected cycle count (sw/dram.py weight_dma_cycles).

The coefficient at index i is coeff[i] = (i * 37 + 11) & 0xFFFF
(a deterministic pseudo-random pattern, not a trivial increment).
The destination address for coefficient i is:
  lane = i // taps, tap = i % taps  (taps=0: lane 0, tap = i)
  addr = dst_base + lane * stride + tap
The expected cycle count is weight_dma_cycles(ncoeff) from sw/dram.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sw"))
from dram import weight_dma_cycles  # noqa: E402


def coeff(i):
    return (i * 37 + 11) & 0xFFFF


def expected_writes(ncoeff, dst_base, taps, stride):
    """List of (addr, data) config-bus writes, in order."""
    out = []
    for i in range(ncoeff):
        if taps == 0:
            lane, tap = 0, i
        else:
            lane, tap = divmod(i, taps)
        out.append((dst_base + lane * stride + tap, coeff(i)))
    return out


def gen(path, ncoeff, dst_base, taps, stride):
    """Write the vector file: first the coefficients (for the AXI
    slave memory), then the expected config writes."""
    with open(path, "w") as f:
        f.write(f"# ncoeff={ncoeff} dst=0x{dst_base:04x} "
                f"taps={taps} stride={stride}\n")
        f.write(f"# expected_cycles={weight_dma_cycles(ncoeff)}\n")
        for i in range(ncoeff):
            f.write(f"C {i} {coeff(i):04x}\n")
        for addr, data in expected_writes(ncoeff, dst_base, taps, stride):
            f.write(f"W {addr:04x} {data:04x}\n")


if __name__ == "__main__":
    # Cases: (ncoeff, dst_base, taps, stride)
    #   A: contiguous (scheduler bank A), 81 coeffs (SRCNN L1, K=9)
    #   B: strided (conv2d_mc, K=3, lane_stride=128), 36 coeffs
    #   C: strided, 289 coeffs (full DW bank, K=17 -> taps=289)
    #   D: multi-burst, 200 coeffs (4 bursts of 64 + remainder)
    cases = [
        ("A", 81, 0x0200, 0, 1),
        ("B", 36, 0x0100, 9, 128),
        ("C", 289, 0x0100, 289, 128),
        ("D", 200, 0x0400, 0, 1),
    ]
    outdir = os.path.dirname(os.path.abspath(__file__))
    for name, ncoeff, dst, taps, stride in cases:
        gen(os.path.join(outdir, f"vectors_dma_{name}.txt"),
            ncoeff, dst, taps, stride)
        print(f"case {name}: ncoeff={ncoeff} dst=0x{dst:04x} taps={taps} "
              f"stride={stride} expected_cycles="
              f"{weight_dma_cycles(ncoeff)}")
