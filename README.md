# fpga-accel

FPGA accelerator for multi-function image processing models — image enhancement / restoration / super-resolution, object detection (YOLO), infrared small target detection (IRSTD), and other CNN models: a unified reconfigurable compute engine that runs complete models.

- **Target boards**: iW-RainboW-G35D (XCZU19EG-2, Zynq UltraScale+) for vision models; VCU128 (XCVU37P-2FSVH2892E, Virtex UltraScale+ HBM) for large-model decode — see below
- **Implementation language**: SystemVerilog (all new modules in SV; the original Verilog streaming convolution is frozen and kept only as a reference baseline)
- **Flow**: Xilinx full command line (vivado batch / Tcl scripts, no GUI)
- **Interfaces**: AXI4-Lite (control/config), AXI4-Stream (pixel stream), AXI4-MM (DDR transfers)

## Implementation Plan

The complete implementation plan is in [docs/plan.md](docs/plan.md) (the Markdown version of the v3.2 plan, a 12-page A4 PDF):

- System block diagram (accelerator at the centre) → §1 Research results (core conclusions → dense kernel ceiling → large-kernel implementation → channel counts → operators beyond convolution → per-application panorama, Tables RD/RL/R7/R8/RP) → §2 Design requirements (architecture + operator set, each traceable to §1) → §3 Bandwidth analysis and model partitioning → §4 Implementation phases → §5 Resource estimates and risks → §6 Input and output expectations → §7 Methodology and toolchain → Appendix: decision basis.

Key design decisions: dense max 11×11, DW 17×17 square kernels / 31×1 and 1×31 strips, dilated stored kernels 3–9, d≤4, effective ≤35; channel tiling (C_in tiled and accumulated, C_out tiled and replicated); layer fusion (unfused DnCNN at 1080p60 needs 271 GB/s, fused 0.81 GB/s); front-end preprocessing module (downsampling, 1/3 channels, kernels ≤5×5).

## Directory Structure

```
fpga-accel/
├── README.md
├── LICENSE
├── Makefile            # One-click reproduction: make sim / synth / impl / bitstream / program
├── docs/
│   ├── plan.md         # Complete implementation plan (v3.2)
│   ├── system_diagram.png
│   └── vcu128-bringup.md  # VCU128 bring-up checklist (shell, XRT, HBM, PE port)
├── rtl/                # SystemVerilog source (parameterized)
├── tb/                 # SystemVerilog testbenches + Python goldens + fail-fast run_stage*.sh
├── sw/                 # Compiler (Python: capture / fusion / tiling / scheduling / emission, cost model, funcsim) + C driver (sw/driver/) + unit tests (sw/tests/)
├── models/             # Model weights + quantization (vision: srcnn/dncnn/retinexnet/yolov8n/dnanet; llm: Qwen3-VL-4B / Qwen2.5-Omni-3B, pending fetch)
├── sta/                # STA probes (tracked critpath_*.sv; generated .tcl/.ys/netlists untracked)
├── scripts/            # (empty, reserved for Tcl flow scripts)
├── constraints/        # (empty, reserved for XDC)
└── logs/               # Logs and reports from each run
```

Generated outputs (untracked, regenerable, not in the tree above): `out/`, `tb/*.vvp`, `tb/out_*.txt`, `tb/vectors_*.txt`, `sta/*.tcl`, `sta/*_net.v`.

## Current Status

- **Plan v3.2 complete** (2026-11-19): system block diagram + research results driving the design requirements.
- **Phase 0 baseline done**: SV parameterized PX=1/2/4, dense 9×9, golden + STA + resources verified.
- **Stage 1 done** (2026-11-19): dense 11x11 + arbitrary padding/stride, PX=1, SystemVerilog (`rtl/conv2d_dense.sv`). Golden 14/14 PASS pixel-exact, framing errors 0; per-stage STA (Nangate45, 300 MHz) Fmax 469 MHz (barrel shifter critical, 2.093 ns); DSP 121, BRAM36 10, LUT ~5,300 (inferred), FF ~8,750. Details + logs in `rtl/README.md` and `logs/stage1/`.
- **Phases 1–7 (on the VM)**: in progress — module design + individual verification + model numerical verification (Stage 1 done, Stages 2–7 to go).
- **Phases 8–11 (Pittsburgh, after 11/26)**: board-level synthesis / integration / on-board testing.
- **VCU128 (VU37P) platform support** (2026-11-21): XCVU37P-2FSVH2892E added as a third device config (`sw/devices.py` vu37p) — 9,024 DSP, 2,852K logic cells, 8 GB HBM2 at 460 GB/s, PCIe Gen4 x8 host link, no ARM PS (host via XDMA/QDMA + XRT). Same 15×128 PE config as the ZU19EG (decode is bandwidth-bound, so the PE count is not scaled up): SRCNN 1080p 33.7 fps @300 MHz at 21.3% of device DSP. Decode roofline: 7B INT8 65.7 tok/s, 13B INT4 70.8 tok/s (both fit one board), 13B INT8 35.4 tok/s (needs 2 boards). Dual-card targets: Qwen3-VL-4B + Qwen2.5-Omni-3B, TP=2 over QSFP28. Not covered yet: VU37P synthesis (desktop/Mac only) and the XDMA/XRT host driver port — see [docs/vcu128-bringup.md](docs/vcu128-bringup.md). Details in `rtl/README.md` and `logs/stage7b/model_vu37p.log`.

## How to Use / Reproduce

> The flow scripts are still being added; the target usage is below.

```sh
git clone <remote-url>
cd fpga-accel
make sim        # Simulation + golden comparison (this VM: Icarus / Yosys / OpenSTA)
make synth      # Synthesis (Vivado machine)
make impl       # Implementation (place and route, timing report)
make bitstream  # Generate bitstream
make program    # Program + ILA capture
make all        # Synthesis → implementation → bitstream
```

On any machine with the same Vivado version, `git clone` then `make all` reproduces synthesis → implementation → bitstream; the Tcl scripts check the Vivado version at the start and record it in the logs.

## License

MIT License, see [LICENSE](LICENSE).
