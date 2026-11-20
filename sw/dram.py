#!/usr/bin/env python3
"""Stage 5: shared DRAM model + AXI-MM weight-streaming model.

Two models in one module:

1. AXI-MM weight DMA transfer model (``weight_dma_cycles``). This is
   the cycle model for rtl/weight_dma.sv and MUST match the RTL
   testbench measurement. The DMA reads 16-bit coefficients from DRAM
   over a 64-bit AXI read channel (4 coeffs/beat, up to 16 beats per
   burst = 64 coeffs/burst) and writes them into a weight bank over
   the config bus. The config bus accepts ONE coefficient per cycle,
   so the transfer is DRAIN-limited at 1 coeff/cycle (the AXI bus is
   faster than the drain; a 2-beat FIFO absorbs bursts). Therefore:
       cycles = ncoeff (drain) + n_bursts * AR_OVERHEAD + LATENCY
   with AR_OVERHEAD = 4 cycles per burst (AR handshake) and
   LATENCY = 12 cycles (first AR to first data). The config bus
   (Stage 4, PS-written) needs 2 cycles/coeff, so the DMA is ~2x
   faster in cycles AND autonomous (no PS involvement per
   coefficient). A 4-wide bank write would match the AXI bus rate
   (4 coeffs/cycle) but needs a scheduler hardware change; it is
   recorded as a follow-up, not claimed here.

2. Shared DRAM traffic model (``dram_traffic``). PL DDR4
   dual-channel DDR4-2400: theoretical 38.4 GB/s, actual ~27 GB/s
   (plan §3). SmartConnect arbitration, priority VDMA > engine > PS.
   The model computes, for a model at a resolution/fps:
     * unfused: every layer reads its input feature map from DRAM
       and writes its output back (2 * H*W*C bytes per layer per
       frame, 1 byte/px/channel).
     * fused: only the model's input and output touch DRAM, plus the
       weights (read once per frame, or once per tile if the tile
       reuses weights -- the fused model streams weights once per
       frame for the iterative schedule).
   The numbers reproduce plan Tables 4-6 (DnCNN 1080p60 unfused
   271 GB/s, fused 0.81 GB/s) from the layer shapes, not from
   hard-coded constants.
"""
import math

# --- AXI-MM weight DMA (matches rtl/weight_dma.sv) -------------------
AXI_DATA_BYTES = 8        # 64-bit AXI data bus
COEFF_BYTES = 2           # 16-bit coefficient
COEFFS_PER_BEAT = AXI_DATA_BYTES // COEFF_BYTES   # 4
BURST_BEATS = 16          # max beats per burst
COEFFS_PER_BURST = COEFFS_PER_BEAT * BURST_BEATS  # 64
DMA_AR_OVERHEAD = 4       # cycles per burst (AR handshake)
DMA_LATENCY = 12          # cycles, first AR to first data
CFG_BUS_CYCLES_PER_COEFF = 2  # Stage 4 measured (PS-written config bus)


def weight_dma_cycles(ncoeff, burst_beats=BURST_BEATS):
    """AXI-MM weight-transfer cycles for ncoeff coefficients.

    Drain-limited at 1 coeff/cycle (the config-bus write rate into
    the bank); plus one AR overhead per burst and the initial latency.
    """
    if ncoeff <= 0:
        return 0
    coeffs_per_burst = COEFFS_PER_BEAT * burst_beats
    n_bursts = math.ceil(ncoeff / coeffs_per_burst)
    return ncoeff + n_bursts * DMA_AR_OVERHEAD + DMA_LATENCY


def weight_cfg_cycles(ncoeff):
    """Config-bus (PS-written, Stage 4) cycles for ncoeff coeffs."""
    return CFG_BUS_CYCLES_PER_COEFF * ncoeff


# --- Shared DRAM traffic ----------------------------------------------
PL_DDR4_ACTUAL_GBS = 27.0   # plan §3 (theoretical 38.4)
PL_DDR4_THEOR_GBS = 38.4
PS_DDR4_ACTUAL_GBS = 13.0


def vdma_bandwidth_gbs(h, w, channels, fps):
    """One video stream (in or out) at h*w, channels, fps, 1 B/px/ch."""
    return h * w * channels * fps / 1e9


def layer_fm_bytes(h, w, c):
    """One feature map, bytes (1 byte/px/channel)."""
    return h * w * c


def dram_traffic(layers, h, w, fps, fused=True, in_ch=3, out_ch=3):
    """DRAM traffic for a model (list of LayerShape-like dicts with
    H_out, W_out, C_out, K, C_in) at resolution h*w, fps.

    Returns a dict with per-component GB/s and the total, plus the
    unfused total for comparison. ``layers`` carry the OUTPUT shape
    of each layer (H_out, W_out, C_out); the input shape of layer i
    is the output shape of layer i-1 (or the model input for i=0).
    Weights are counted once per frame (iterative schedule, weights
    streamed from DRAM once per layer per frame).
    """
    # Feature-map traffic
    fm_unfused = 0.0  # bytes/frame, read+write per layer
    prev_h, prev_w, prev_c = h, w, in_ch
    for L in layers:
        ho, wo, co = L["H_out"], L["W_out"], L["C_out"]
        # read input (prev) + write output (this layer)
        fm_unfused += layer_fm_bytes(prev_h, prev_w, prev_c)
        fm_unfused += layer_fm_bytes(ho, wo, co)
        prev_h, prev_w, prev_c = ho, wo, co
    # Weights: sum over layers of C_in*C_out*K^2 coeffs * 2 bytes
    w_bytes = sum(L["C_in"] * L["C_out"] * L["K"] * L["K"] * 2
                  for L in layers)
    # Fused: only model input + model output + weights touch DRAM
    fm_fused = (layer_fm_bytes(h, w, in_ch)
                + layer_fm_bytes(layers[-1]["H_out"],
                                 layers[-1]["W_out"],
                                 layers[-1]["C_out"]))
    return {
        "unfused_fm_GBs": fm_unfused * fps / 1e9,
        "unfused_weights_GBs": w_bytes * fps / 1e9,
        "unfused_total_GBs": (fm_unfused + w_bytes) * fps / 1e9,
        "fused_in_GBs": layer_fm_bytes(h, w, in_ch) * fps / 1e9,
        "fused_out_GBs": layer_fm_bytes(layers[-1]["H_out"],
                                        layers[-1]["W_out"],
                                        layers[-1]["C_out"]) * fps / 1e9,
        "fused_weights_GBs": w_bytes * fps / 1e9,
        "fused_total_GBs": (fm_fused + w_bytes) * fps / 1e9,
        "weights_bytes_per_frame": w_bytes,
    }


def arbitration_budget(vdma_in_gbs, vdma_out_gbs, ps_reserved_gbs=1.0):
    """Bandwidth left for the engine after VDMA (highest priority)
    and a PS reserve (lowest priority). Returns (engine_GBs, ok)."""
    engine = PL_DDR4_ACTUAL_GBS - vdma_in_gbs - vdma_out_gbs \
        - ps_reserved_gbs
    return engine, engine > 0


def cut_point_budget(vdma_in_gbs, vdma_out_gbs, ps_reserved_gbs=1.0):
    """How many fusion cut points (each a full feature-map read+write)
    the remaining bandwidth affords, per plan §3.7. Returns the
    remaining GB/s for cut points."""
    engine, _ = arbitration_budget(vdma_in_gbs, vdma_out_gbs,
                                   ps_reserved_gbs)
    return engine
