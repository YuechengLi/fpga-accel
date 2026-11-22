#!/usr/bin/env python3
"""Stage 5 compiler: per-layer tile-size cost model.

The tile scheduler searches, per layer, the tile granularity that
minimizes total time under the hardware constraints. The optimum
varies per layer (K / C_in / C_out / H / W), it is NOT one global
value. The cost model is:

  compute (per (o, t) pass, conv2d_mc, C_PAR parallel lanes):
      cycles_pass = H_out_tile * W_out_tile + pipe_latency
      (1 output pixel/cycle in steady state; the K^2 MACs are
      parallel in the datapath, so K cancels out of the cycle count)
  weight load (per pass, ncoeff = C_in_tile * 121):
      The MC loads one full 11x11 grid (121 entries) per lane, not
      just the K*K taps (Stage 6 fix -- the MC's coefficient memory
      is indexed by the grid, r*11+c). AXI-MM: cycles = ncoeff +
      n_bursts*4 + 12 (sw/dram.py weight_dma_cycles). The transfer
      is DRAIN-limited at 1 coeff/cycle (the config-bus write rate
      into the bank); the 64-bit AXI bus (4 coeffs/beat) is faster
      than the drain. The DMA is ~2x the config bus in cycles and
      autonomous (no PS per-coefficient work).
      config bus (Stage 4, for comparison): 2 cycles/coeff
  input stream (per spatial tile, once, reused for all C_out):
      cycles = H_tile * W_tile * C_in  (1 byte/cycle pixel stream)
  ping-pong overlap: while pass (o,t) computes, pass (o,t+1)'s
      weights load into the other bank. The load is FULLY HIDDEN iff
      load_cycles <= cycles_pass; otherwise each pass stalls by
      (load - compute) cycles. Total:
        load <= compute: total = load_first + n_passes * compute
        load >  compute: total = n_passes * load + compute_last

Constraints:
  * weight bank (MC): ncoeff = C_in_tile * 121 <= C_PAR * 128 = 512
    (each lane holds one 121-entry grid in its 128-entry memory).
    C_in_tile <= C_PAR (= 4, the lane count) is the binding limit;
    the 289-entry scheduler bank does NOT limit the MC path.
  * C_in_tile <= C_PAR (= 4, the lane count)
  * tile buffer: H_tile * W_tile * C_in <= 256 KB (plan §2)
  * spatial tile: H_tile, W_tile <= 64 (line-buffer / tile-buffer
    geometry); H_out_tile * W_out_tile is the compute per pass.
"""
import math

C_PAR = 4
MC_GRID = 121            # entries per lane (11x11 grid, Stage 6 fix)
MC_LANE_MEM = 128        # coefficient-memory entries per MC lane
BANK_COEFF = C_PAR * MC_LANE_MEM  # 512, MC per-pass coefficient budget
TILE_BUFFER_BUDGET = 64 * 64 * 64  # bytes
PIPE_LATENCY_DENSE = 24   # cycles (Stage 1)
PIPE_LATENCY_DW = 24
CFG_BUS_CYCLES_PER_COEFF = 2  # Stage 4 measured (121 coeffs ~2x)


class LayerShape:
    """A conv layer's shape (all in pixels/channels)."""

    def __init__(self, K, C_in, C_out, H, W, stride=(1, 1), pad=(0, 0, 0, 0),
                 dilation=1, mode="dense"):
        self.K = K
        self.C_in = C_in
        self.C_out = C_out
        self.H = H
        self.W = W
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.mode = mode  # dense | dw | fc

    @property
    def H_out(self):
        pt, pb, pl, pr = self.pad
        sx, sy = self.stride
        return (self.H + pt + pb - self.K) // sx + 1

    @property
    def W_out(self):
        pt, pb, pl, pr = self.pad
        sx, sy = self.stride
        return (self.W + pl + pr - self.K) // sy + 1

    @property
    def macs(self):
        """Total MACs for the layer (one MAC = one multiply)."""
        return self.H_out * self.W_out * self.C_out * self.C_in \
            * self.K * self.K

    def __repr__(self):
        return (f"Layer(K={self.K} Cin={self.C_in} Cout={self.C_out} "
                f"{self.H}x{self.W} -> {self.H_out}x{self.W_out} "
                f"{self.mode})")


def weight_load_cycles_axi(ncoeff):
    """AXI-MM weight-load cycles for ncoeff 16-bit coefficients
    (drain-limited, sw/dram.py weight_dma_cycles)."""
    from dram import weight_dma_cycles
    return weight_dma_cycles(ncoeff)


def weight_load_cycles_cfg(ncoeff):
    """Config-bus (AXI-Lite) weight-load cycles (Stage 4)."""
    return CFG_BUS_CYCLES_PER_COEFF * ncoeff


def _divisors(n, cap):
    """Divisors of n up to cap (tile sizes that divide the dimension)."""
    out = [d for d in range(1, min(n, cap) + 1) if n % d == 0]
    if not out:
        out = [1]
    return out


def tile_search(layer, f_clk_mhz=300.0, verbose=False):
    """Search the tile granularity for one layer.

    Returns a dict with the best tile (C_in_tile, H_tile, W_tile),
    cycles, load/compute per pass, whether the load is hidden, and
    the full candidate table (for the report).
    """
    K = layer.K
    # C_in_tile is bounded by the lane count AND the weight bank.
    # The MC loads one 121-entry grid per lane (Stage 6 fix), so the
    # per-pass coefficient count is C_in_tile * 121, bounded by the
    # MC's per-lane memory (C_PAR * 128 = 512). This no longer limits
    # C_in_tile below C_PAR for any K (4*121=484 <= 512).
    cin_max = min(C_PAR, BANK_COEFF // MC_GRID, layer.C_in)
    cin_tiles = [c for c in range(1, cin_max + 1)
                 if layer.C_in % c == 0 or c == cin_max]
    if not cin_tiles:
        cin_tiles = [1]
    # Spatial tile: divide the output; bounded by the tile buffer
    # (H_tile*W_tile*C_in <= 256 KB) and by 64 (tile geometry).
    h_cands = _divisors(layer.H_out, 64)
    w_cands = _divisors(layer.W_out, 64)
    # Prefer the largest spatial tiles that fit (fewer tiles = less
    # reload overhead); also try smaller ones (they change the
    # load-vs-compute balance).
    h_cands = sorted(set(h_cands + [min(layer.H_out, 64),
                                    min(layer.H_out, 32),
                                    min(layer.H_out, 16),
                                    min(layer.H_out, 8)]))
    w_cands = sorted(set(w_cands + [min(layer.W_out, 64),
                                    min(layer.W_out, 32),
                                    min(layer.W_out, 16),
                                    min(layer.W_out, 8)]))
    best = None
    table = []
    for cin_t in cin_tiles:
        ncoeff = cin_t * MC_GRID
        load_axi = weight_load_cycles_axi(ncoeff)
        load_cfg = weight_load_cycles_cfg(ncoeff)
        n_cin_passes = math.ceil(layer.C_in / cin_t)
        for ht in h_cands:
            for wt in w_cands:
                if ht * wt * layer.C_in > TILE_BUFFER_BUDGET:
                    continue
                # output tile (stride 1 here; the general case divides
                # the input tile by the stride)
                sx, sy = layer.stride
                hot = max(1, ht // sx)
                wot = max(1, wt // sy)
                compute = hot * wot + PIPE_LATENCY_DENSE
                n_spatial = (math.ceil(layer.H_out / hot)
                             * math.ceil(layer.W_out / wot))
                n_passes = n_cin_passes * layer.C_out * n_spatial
                # ping-pong total
                if load_axi <= compute:
                    total = load_axi + n_passes * compute
                else:
                    total = n_passes * load_axi + compute
                hidden = load_axi <= compute
                # input stream (once per spatial tile, amortized)
                in_stream = ht * wt * layer.C_in * n_spatial
                total_eff = max(total, in_stream)
                row = {"C_in_tile": cin_t, "H_tile": ht, "W_tile": wt,
                       "ncoeff": ncoeff, "load_axi": load_axi,
                       "load_cfg": load_cfg, "compute_per_pass": compute,
                       "n_passes": n_passes, "total_cycles": total,
                       "total_eff_cycles": total_eff,
                       "load_hidden": hidden,
                       "in_stream_cycles": in_stream}
                table.append(row)
                key = (total_eff, 0 if hidden else 1)
                if best is None or key < (best["total_eff_cycles"],
                                          0 if best["load_hidden"] else 1):
                    best = row
    if best is None:
        raise RuntimeError(f"tile_search: no feasible tile for {layer}")
    best = dict(best)
    best["time_us"] = best["total_eff_cycles"] / (f_clk_mhz * 1e6) * 1e6
    best["table"] = table
    return best


def device_mem_bandwidth_gbs(dev):
    """Memory bandwidth (GB/s) for a device config, for the roofline.

    VCU128 (vu37p) has HBM2 at 460 GB/s (sw/devices.py ``mem_gbs``);
    every other device defaults to the ZU19EG PL DDR4 actual figure
    (~27 GB/s, sw/dram.py). The ECP5 has no external DRAM (on-chip
    EBR only) and does not use this path. tile_search above does not
    call this: the MC engine is drain-limited (1 coeff/cycle), not
    DRAM-limited, so the bandwidth only enters the roofline, never
    the per-layer tile choice.
    """
    from dram import PL_DDR4_ACTUAL_GBS
    return float(dev.get("mem_gbs", PL_DDR4_ACTUAL_GBS))


def layer_report(layer, result):
    """One-line summary of the tile search for a layer."""
    b = result
    return (f"{layer}: C_in_tile={b['C_in_tile']} "
            f"HxW tile={b['H_tile']}x{b['W_tile']} "
            f"ncoeff/pass={b['ncoeff']} load={b['load_axi']}cy "
            f"compute/pass={b['compute_per_pass']}cy "
            f"load_hidden={b['load_hidden']} "
            f"total={b['total_eff_cycles']}cy "
            f"({b['time_us']:.1f} us @300MHz)")
