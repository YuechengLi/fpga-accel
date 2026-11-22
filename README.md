# fpga-accel

FPGA accelerator for vision models and large-model decode — one repo, two tracks, one shared compute core.

- **Vision track**: image enhancement / restoration / super-resolution, object detection (YOLO), infrared small target detection (IRSTD) — CNNs on the iW-RainboW-G35D (XCZU19EG-2, Zynq UltraScale+).
- **LLM track**: multimodal large-model decode on the VCU128 (XCVU37P-2FSVH2892E, Virtex UltraScale+ HBM) — Qwen3-VL-4B and Qwen2.5-Omni-3B, tensor-parallel across two cards over QSFP28.
- **Shared core**: a generic PE array (tensor-core-like GEMM engine). Both tracks lower to GEMM; the memory system and data path differ (see Architecture).
- **Implementation language**: SystemVerilog (all new modules in SV; the original Verilog streaming convolution is frozen and kept only as a reference baseline)
- **Flow**: Xilinx full command line (vivado batch / Tcl scripts, no GUI)
- **Interfaces**: AXI4-Lite (control/config), AXI4-Stream (pixel stream), AXI4-MM (DDR transfers)

## Architecture: one core, two data paths

The compute core is shared; the memory subsystem and data path are designed per track:

- **Vision / CNN**: weights are small (tens of KB), loaded once and reused across the whole frame. Weight-stationary dataflow, line buffers + pixel-stream frontend. The bottleneck is MAC utilization on small kernels.
- **LLM decode**: weights are large (4.4 GB) and streamed from HBM once per token, zero reuse within a token. Weight-streaming dataflow, plus an attention unit and KV-cache management. The bottleneck is HBM bandwidth. Prefill is compute-bound, like CNN.
- The scheduler switches modes: weight-stationary (prefill / CNN) vs weight-streaming (decode). Quantization and verification differ per track too: CNNs are checked by PSNR vs float, LLMs by logits and end-to-end text comparison.

## Boards and heterogeneous split

- **G35D (ZU19EG)**: vision perception — small models, full pipeline on chip.
- **VCU128 (VU37P, 8 GB HBM2 at 460 GB/s)**: large-model decode. Same 15×128 PE config as the ZU19EG (decode is bandwidth-bound: HBM delivers ~1,533 weight bytes/cycle at 300 MHz, so the PE count is not scaled up). SRCNN 1080p 33.7 fps @300 MHz at 21.3% of device DSP. Decode roofline: 7B INT8 65.7 tok/s, 13B INT4 70.8 tok/s (both fit one board), 13B INT8 35.4 tok/s (needs 2 boards).
- **Dual-card**: two VCU128s daisy-chained over QSFP28, TP=2. Qwen3-VL-4B ~209 tok/s and Qwen2.5-Omni-3B ~300 tok/s roofline; the all-reduce is ~30 µs/token, not the bottleneck. Pipeline parallelism does not speed up single-stream decode, only batched throughput.
- **Rest of the system**: RTX 5060 Ti for prefill + training, CPU for routing/scheduling. Host link is PCIe Gen4 x8 (XDMA/QDMA + XRT); bring-up checklist in [docs/vcu128-bringup.md](docs/vcu128-bringup.md).

## Implementation Plan

The complete implementation plan is in [docs/plan.md](docs/plan.md) (the Markdown version of the v3.2 plan, a 12-page A4 PDF):

- System block diagram (accelerator at the centre) → §1 Research results (core conclusions → dense kernel ceiling → large-kernel implementation → channel counts → operators beyond convolution → per-application panorama, Tables RD/RL/R7/R8/RP) → §2 Design requirements (architecture + operator set, each traceable to §1) → §3 Bandwidth analysis and model partitioning → §4 Implementation phases → §5 Resource estimates and risks → §6 Input and output expectations → §7 Methodology and toolchain → Appendix: decision basis.

Key design decisions: dense max 11×11, DW 17×17 square kernels / 31×1 and 1×31 strips, dilated stored kernels 3–9, d≤4, effective ≤35; channel tiling (C_in tiled and accumulated, C_out tiled and replicated); layer fusion (unfused DnCNN at 1080p60 needs 271 GB/s, fused 0.81 GB/s); front-end preprocessing module (downsampling, 1/3 channels, kernels ≤5×5); generic PE array, tensor-core-like, PE size derived from model + real-time goal + FPGA resources (128 mults/PE × 15 PEs for the ZU19EG).

## Models

**Vision** (weights + int16 quantization in `models/vision/`): SRCNN, DnCNN, RetinexNet, YOLOv8n, DNANet.

**LLM** (`models/llm/`, fetched 2026-11-21): Qwen3-VL-4B-Instruct (36 layers, hidden 2560, 4.4B, Apache 2.0) and Qwen2.5-Omni-3B (Thinker-Talker, research-only license; BF16 checkpoint 11.17 GiB, ~6B params incl. Talker + encoders). **Weights are NOT checked in**: raw weights live in `models/llm/*/raw/` (gitignored, 19.44 GiB total); the repo holds only the download scripts, per-file SHA256, and READMEs.

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
│   ├── common/         # Shared core: PE array, AXI, DMA, scheduler (both tracks' regressions)
│   ├── vision/         # CNN datapath + operators
│   └── llm/            # (empty stub — attention, KV, RoPE, future work)
├── tb/                 # SystemVerilog testbenches + Python goldens + fail-fast run_stage*.sh
├── sw/                 # Python compiler + C driver; runners (run_stage5.sh, run_regressions.sh) stay here
│   ├── common/         # Compiler core: capture / fusion / tiling / scheduling / emission, cost model, passes/, driver/
│   ├── vision/         # funcsim, demos, Stage 6/7b models, e2e drivers, tests/
│   └── llm/            # (empty stub — decode model, KV, attention scheduling, future work)
├── models/             # Model weights + quantization
│   ├── vision/         # srcnn/dncnn/retinexnet/yolov8n/dnanet
│   └── llm/            # Qwen3-VL-4B / Qwen2.5-Omni-3B, fetched 2026-11-21, weights gitignored
├── sta/                # STA probes (tracked critpath_*.sv; generated .tcl/.ys/netlists untracked)
├── scripts/            # (empty, reserved for Tcl flow scripts)
├── constraints/        # (empty, reserved for XDC)
└── logs/               # Logs and reports from each run
```

Generated outputs (untracked, regenerable, not in the tree above): `out/`, `tb/*.vvp`, `tb/out_*.txt`, `tb/vectors_*.txt`, `sta/*.tcl`, `sta/*_net.v`.

> **Restructured 2026-11-21 night** (after Stage 7c passed): `rtl/`, `sw/`, `models/` split into `common/` / `vision/` / `llm/` as shown above. One repo, separate folders — the split is at the memory subsystem and data path only.

## Current Status

- **Stages 1–7 done** (2026-11-19/20): dense / DW / multi-channel / operators / compiler / model numerical verification / frontend. Golden pixel-exact, STA all MET.
- **Stage 7b done** (2026-11-20): generic PE array, 128 mults/PE × 15 PEs = 1920 DSPs (97.6%); SRCNN 1080p 33.7 fps @300 MHz, MFU 96.3%; STA dot-path Fmax ~393 MHz. Details + logs in `rtl/README.md` and `logs/stage7b/`.
- **Stage 7c done** (2026-11-21 night): integrated end-to-end simulation (frontend → PE array → driver, real image in → image out) + post-synthesis simulation (Yosys netlist re-run against golden), ALL PASS (7 PASS / 0 FAIL).
- **VCU128 platform support done** (2026-11-21): `vu37p` device config + HBM roofline + 9 new tests, `tb/run_stage7b.sh` §7. Details in `rtl/README.md` and `logs/stage7b/model_vu37p.log`.
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
