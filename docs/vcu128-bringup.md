# VCU128 Bring-up Checklist

**Purpose:** bring the VCU128 boards (VU37P, 8 GB HBM) up in the DIY desktop
and run HBM decode experiments. **Status: checklist only.** Execute after the
2026-11-25 return to Pittsburgh, once the desktop is built. No hardware
action is due before then.

**Inventory (settled):** 2 production VCU128 boards (possibly 3),
part EK-U1-VCU128-G, device XCVU37P-2FSVH2892E. 8 GB HBM2, 460 GB/s.
PCIe Gen3 x16 / Gen4 x8. 4x QSFP28. 9.5 in long, 7.53 in tall. Discontinued;
2022 price ~$10,794. Production boards, not ES1.

**Desktop (settled):** ASRock X870E Taichi, Ryzen 7 9700X, RTX 5060 Ti 16 GB
in slot 1, slot 2 free (x8/x8 bifurcation), 1000 W PSU, Fractal Pop XL Air.
One free slot only — boards 2 and 3 have no slot (see Phase 5).

**Role in the heterogeneous split:** G35D runs small perception models;
VCU128 runs HBM decode / big-model offload; 5060 Ti runs prefill and
training; CPU does routing/scheduling.

**Why:** multi-board distributed LLM was stuck on per-board DDR bandwidth
(G35D PL DDR4 ~38 GB/s, 7B INT8 ~4–5 tok/s per board). VCU128 HBM gives
7B INT8 decode ~60+ tok/s in theory (460 GB/s / 7 GB per token), same order
as the 5060 Ti. VU37P is 2.5x the logic (2,852K vs 1,143K cells) and 4.6x
the DSP (9,024 vs 1,968) of the ZU19EG.

**Capacity math (INT8 unless noted):** 7B = 7 GB, fits one board (tight:
~1 GB left for KV cache). 13B INT4 = 6.5 GB, fits. 13B INT8 = 13 GB, does
not fit one board. 2 boards = 16 GB HBM, fits 13B INT8. 3 boards = 24 GB,
13B INT8 + KV headroom, or 30B INT4 (15 GB) + KV.

---

## Phase 0 — Before touching hardware

- [ ] **Decide the desktop OS. This gates everything.** XRT 2023.1 supports
      RHEL/CentOS 7, RHEL 8/9, Ubuntu 18.04/20.04/22.04 LTS (x86_64) only.
      Fedora is NOT on the supported list. If the desktop is Fedora, either
      install Ubuntu 22.04 LTS (recommended) or build XRT from source.
      Turn off kernel auto-upgrade once XRT is installed (XRT is not tested
      against live-patched kernels).
- [ ] **Pick the tool version.** Vivado/Vitis 2023.1 or newer. Confirm the
      VCU128 board files and the XDMA shell (xilinx_vcu128) are available
      for that version before installing.
- [ ] **Downloads (do now, they are large):** XRT .deb/.rpm for the chosen
      OS; VCU128 deployment platform package (from the VCU128 product page,
      Getting Started tab); UG1302 (VCU128 user guide); UG949 (designing
      with HBM devices).
- [ ] **Order for Phase 5 (not needed for Phase 1–4):** 2x QSFP28 DAC cables
      (board-to-board daisy chain).
- [ ] **Check PSU cables:** one free 8-pin + one free 6-pin PCIe power
      connector. VCU128 max power ~225 W (slot 75 W + aux). 1000 W PSU has
      headroom (9700X ~120 W + 5060 Ti ~180 W + VCU128 ~225 W), but the
      connectors must physically exist.
- [ ] **Check case clearance:** VCU128 is full-height (7.53 in tall) and
      9.5 in long. Pop XL Air fits it; confirm no drive cage blocks slot 2.

## Phase 1 — Physical install + shell bring-up

- [ ] Power off, seat board 1 in slot 2, connect 8-pin + 6-pin aux power.
- [ ] Boot. `lspci -d 10ee:` must show the card (two functions: management
      + user). If nothing appears, check slot bifurcation is x8/x8 in BIOS.
- [ ] Install XRT (`sudo apt install ./xrt_*.deb`), then the deployment
      platform package.
- [ ] Flash the shell: `sudo /opt/xilinx/xrt/bin/xbmgmt program --base
      --device <management BDF>` (XRT 2021.1+ syntax). Then **warm reboot**.
- [ ] `xbutil examine` (or `xbutil validate`): shell loaded, card healthy,
      temperature sane, no errors.
- [ ] **Pass criterion:** `xbutil validate` reports the card ready; the
      management + user PFs are both visible.

## Phase 2 — DMA sanity (before any user logic)

- [ ] Run the XRT hello-world / bandwidth test (`xbutil validate --run
      bandwidth` if available, else the `host_xrt` DMA example).
- [ ] Confirm the PCIe link: Gen4 x8 expected in slot 2 (~16 GB/s
      theoretical, ~13–14 GB/s measured). Gen3 x16 would also be acceptable
      but slot 2 is x8, so Gen4 x8 is the target.
- [ ] **Pass criterion:** host-to-card DMA works both directions at
      ~10+ GB/s. If DMA fails, the shell or the link is wrong — do not
      proceed to user logic.

## Phase 3 — PE array port (ZU19EG → VU37P) + HBM adapter

- [ ] **New device config** in `sw/common/devices.py`: `vu37p` — 9,024 DSP,
      2,852K logic cells, HBM instead of DDR. Re-derive the PE sizing from
      the decode goal (see note below); do not copy the ZU19EG config.
- [ ] **Retarget synthesis:** `synth_xilinx` for part XCVU37P-2FSVH2892E.
      Expect longer runtimes than the ZU19EG (bigger device); run on the
      desktop, not the 6 GB VM.
- [ ] **HBM adapter layer (the new work).** The HBM IP exposes 32 AXI3
      ports, 256-bit each; any port can reach any HBM address through the
      built-in switch (UG949). Each pseudo-channel is 256 MB. Start with
      one port, then interleave across ports.
      - `weight_dma` (AXI-MM) must target HBM addresses, not DDR.
      - AXI3 vs AXI4: the HBM ports are AXI3, the DMA is AXI4 — put an
        AXI protocol converter between them, or configure the DMA for
        AXI3.
      - HBM ports live in SLR0; watch SLR-crossing timing for the PE array.
- [ ] **Host control replaces the Zynq PS flow (biggest engineering
      delta).** VCU128 has no ARM PS. The C driver (`sw/common/driver`) currently
      talks to the PS via AXI-Lite; on VCU128 the PE array's AXI-Lite
      config port maps to a PCIe BAR through the XDMA shell, and host code
      uses XRT (`xrt::ip` register read/write) instead of `/dev/mem`-style
      access. Port the driver to XRT host code; keep the register map
      identical so `sw/common/emit.py` programs are unchanged.
- [ ] **First run:** SRCNN chain on HBM, output vs `funcsim` max|err| = 0
      (same bar as Stage 6/7b). Then a weights-load + HBM readback test.
- [ ] **Pass criterion:** SRCNN chain bit-exact through the HBM path;
      weights load from host to HBM and read back clean.

**Sizing note for decode:** decode is bandwidth-bound, not compute-bound.
HBM at 460 GB/s delivers ~1,533 weight bytes per cycle at 300 MHz, so
~1,500 MACs/cycle saturates it. The existing 15x128 PE config (1,920 MACs)
already roughly saturates HBM — do not over-provision PEs on the VU37P.
Extra DSP is for wider prefill/GEMM work, not for decode.

## Phase 4 — 7B INT8 decode benchmark

- [ ] Load 7B INT8 weights (7 GB) into HBM; KV cache in the remaining ~1 GB.
- [ ] Run the decode loop (GEMV-style, one token per pass), measure tok/s.
- [ ] Same prompt on the 5060 Ti for comparison.
- [ ] **Target:** ~60+ tok/s theoretical (460 / 7); report measured vs
      theoretical and where the gap is (HBM efficiency, PE issue rate,
      host overhead).
- [ ] **Pass criterion:** measured decode within a factor of 2 of the
      HBM roofline, and a clean comparison number vs the 5060 Ti.

## Phase 5 — Multi-board via QSFP28 (only after Phase 4)

- [ ] Connect board 2 to board 1 with a QSFP28 DAC cable (daisy chain).
      Board 2 gets power from its own PSU rail or a second PSU — it has no
      PCIe slot, so it needs external power and its own JTAG/UART for
      bring-up. Same for board 3.
- [ ] Board-to-board link: Aurora 64B/66B or 100G CMAC over the QSFP28
      GTY transceivers. The XDMA shell does not provide this — it is user
      logic. Start with a loopback/ping test, then a bandwidth test.
- [ ] Pipeline split: layers sharded across boards (board 1 = layers 0–N,
      board 2 = layers N+1…); host talks to board 1 only, board 1 forwards
      activations down the chain.
- [ ] 2 boards = 16 GB HBM → 13B INT8. 3 boards = 24 GB → 13B INT8 + KV
      headroom, or 30B INT4.
- [ ] **Pass criterion:** 13B INT8 decode runs across 2 boards with the
      inter-board link not the bottleneck (link BW >> activation size per
      token).

## Gotchas (read before starting)

- **No ARM PS.** Host communication is XDMA/QDMA + XRT, not Zynq
  PS + AXI. The driver port is the largest single work item.
- **OS gate.** Fedora is not XRT-supported; decide the OS in Phase 0.
- **Slot 2 is x8.** Gen4 x8 is fine for this work (HBM is on-card; the
  host link only carries weights in and tokens out).
- **HBM is AXI3, 32 ports, 256-bit.** Adapter layer required; do not
  assume the DDR AXI4 path drops in unchanged.
- **SLR0 holds the HBM interfaces.** PE placement must respect SLR
  boundaries for timing closure at 300 MHz.
- **Board 2/3 have no host.** External power + JTAG/UART for bring-up;
  host sees only board 1.
- **Vivado runtime on VU37P is long.** Use the desktop (or the P1 after
  12/5), never the 6 GB cloud VM, for VU37P synthesis.
