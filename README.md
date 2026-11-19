# fpga-accel

FPGA accelerator for multi-function image processing models (image enhancement / restoration / super-resolution): a unified reconfigurable compute engine that runs complete models.

- **Target board**: iW-RainboW-G35D (XCZU19EG-2, Zynq UltraScale+)
- **Implementation language**: SystemVerilog (all new modules in SV; the original Verilog streaming convolution is frozen and kept only as a reference baseline)
- **Flow**: Xilinx full command line (vivado batch / Tcl scripts, no GUI)
- **Interfaces**: AXI4-Lite (control/config), AXI4-Stream (pixel stream), AXI4-MM (DDR transfers)

## Implementation Plan

The complete implementation plan is in [docs/plan.md](docs/plan.md) (the Markdown version of the v3 plan, a 12-page A4 PDF):

- System block diagram (accelerator at the centre) → §1 Research results (denoising / super-resolution / low-light enhancement / general restoration / YOLO / channel counts / operators beyond convolution, Tables R1–R8) → §2 Design requirements (architecture + operator set, each traceable to §1) → §3 Bandwidth analysis and model partitioning → §4 Implementation phases → §5 Resource estimates and risks → §6 Input and output expectations → §7 Methodology and toolchain → Appendix: decision basis.

Key design decisions: dense max 11×11, DW 17×17 square kernels / 31×1 and 1×31 strips, dilated stored kernels 3–9, d≤4, effective ≤35; channel tiling (C_in tiled and accumulated, C_out tiled and replicated); layer fusion (unfused DnCNN at 1080p60 needs 271 GB/s, fused 0.81 GB/s); front-end preprocessing module (downsampling, 1/3 channels, kernels ≤5×5).

## Directory Structure

```
fpga-accel/
├── README.md
├── LICENSE
├── Makefile            # One-click reproduction: make sim / synth / impl / bitstream / program
├── docs/
│   ├── plan.md         # Complete implementation plan (v3)
│   └── system_diagram.png
├── rtl/                # SystemVerilog source (parameterized)
├── tb/                 # Testbenches + golden (Python)
├── scripts/            # Tcl scripts (synth / impl / bitstream / program / debug / common)
├── constraints/        # XDC (clocks, pins, timing constraints)
├── sw/                 # PS drivers + layer-scheduling library (C)
└── logs/               # Logs and reports from each run
```

## Current Status

- **Plan v3 complete** (2026-11-19): system block diagram + research results driving the design requirements.
- **Phase 0 baseline done**: SV parameterized PX=1/2/4, dense 9×9, golden + STA + resources verified.
- **Phases 1–7 (on the VM)**: in progress — module design + individual verification + model numerical verification.
- **Phases 8–11 (Pittsburgh, after 11/26)**: board-level synthesis / integration / on-board testing.

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
