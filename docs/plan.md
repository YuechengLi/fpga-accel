# Multi-Function Image Processing Model FPGA Accelerator — Complete Implementation Plan

> Target board: iW-RainboW-G35D (XCZU19EG-2) · Implementation language: SystemVerilog · Flow: Xilinx full command line
>
> This document is the Markdown version of plan v3.2 (same content as the 12-page A4 PDF). The PDF is a generated artifact and is not kept in version control.

## System Block Diagram

![Hardware/software system block diagram](system_diagram.png)

Diagram notes:

- **The main accelerator sits at the centre of the system and is its only compute core** — centred, largest, bold border in the diagram: the unified reconfigurable compute engine, Dense / DW / Dilated / 1×1 / FC.
- **The video stream enters from the left**: video input (HDMI / SDI IN) → front-end preprocessing (downsampling, 1/3ch, ≤5×5, input/output) → main accelerator → front-end → video output (HDMI OUT).
- **Feature maps and weights** are moved by the VDMA (video frame mover) between the PL DDR4 (2×4GB) and the accelerator; the PS DDR4 (4GB) connects to the PS.
- **The PS (Cortex-A53)**: Linux · drivers · user library / layer scheduling, issuing configuration, layer scheduling and interrupts over the AXI interconnect (SmartConnect).
- Legend: solid arrows are the pixel stream; green dashed arrows are feature maps / weights; red dotted arrows are the control flow (configuration / layer scheduling / interrupts).
- The front-end preprocessing module is the input/output interface of the whole model.

## §1 Research Results

The research covers five areas — image denoising, super-resolution, low-light enhancement, general image restoration, and infrared small target detection — plus object detection (YOLO) and large-kernel CNN literature for calibration. Key conclusions come first (each annotated with the §2 requirement it drives), then the evidence is laid out by design question, and a panorama table closes the chapter.

### Key Conclusions

- Dense ceiling 9×9: the largest standard dense kernel is 9×9, appearing only in the first layers of SRCNN, SRResNet and RetinexNet; everything else is ≤7×7, and the vast majority is 3×3. → §2.2 Dense max 11×11 (literature 9×9 + one step of headroom)
- Large kernels are all DW / strip / dilated: largest square is 17×17 DW (PLKSR/EchoSR), largest strip is 31 (LKMN/LKFMixer), dilated effective ≤35 (MAN, stored kernel 9×9, d=4); the 63×63 DW (OKNet) appears only at the bottleneck and is an outlier. → §2.2 DW 17×17 / strip 31 / dilated
- Channels are mainstream 64ch, bottleneck 256–512ch: denoising, classic SR and lightweight SR are all 64ch; Transformer SR is 180ch, EDSR is 256ch; U-shaped restoration bottlenecks are 256–512ch; lightweight models and IRSTD are narrower and lighter (36–64ch, LCRNet 16→32→64→64). → §2.1 Channel tiling
- Many layers, receptive field from depth: DnCNN has 17 layers, EDSR 32 blocks, RCAN 200 RCABs; denoising is a 3×3 world, and receptive field comes from depth and downsampling, not kernel size. → §2.1 Unified engine / module reuse
- The 2024–26 accuracy gains are not in kernel size: RFONet (2026) reaches SOTA with only 3×3 (72.14 IoU, IRSTD-1k); LCRNet's 11×11 equivalent is 1.73 IoU worse than 7×7; the gains are in loss, priors, dynamic kernels, wavelets and resolution preservation. → §2.2 Explicit boundaries
- Large-kernel DW touches only a few channels: PLKSR 16/64ch, ESC 16ch, LKMN/LKFMixer the first 1/4 of channels. → §2.1 DW folding / time-sharing

### 1.1 What Is the Dense Kernel Ceiling?

Conclusion: the standard dense ceiling is 9×9, appearing only in three first layers; everything else is ≤7×7, and the vast majority is 3×3.

- Denoising is all 3×3: DnCNN (17 layers, 64ch), FFDNet (64/96ch), RIDNet (64ch, EAM uses dilated convolution but the stored kernel is still 3×3), NAFNet (3×3 DW, base 32 → bottleneck 512), Restormer (3×3 DW, base 48 → 384).
- Classic SR CNNs are all 3×3 except 9×9/5×5: VDSR (20 layers, 64ch), EDSR (16–32 blocks, 64–256ch), RCAN (64ch); low-light and restoration are also all 3×3: Zero-DCE (7 layers, 32ch), Zero-DCE++ (3×3 DW, 32ch), EnlightenGAN (32→512), MPRNet (40/80/96ch).
- Detection is the same: YOLO max is 7×7, IRSTD max dense is 7×7; MSHNet and RFONet are both plain 3×3 (RFONet reaches SOTA with only 3×3).

**Table RD Dense kernel ceiling: exceptions above 3×3**

| Paper (year) | Max dense | Type | Notes |
|---|---|---|---|
| SRCNN (2014) | 9×9 | dense | Largest dense in SR; 9-1-5 |
| SRResNet (2017) | 9×9 | dense | 9×9 input/output; all 16 residual blocks 3×3 |
| RetinexNet (2018) | 9×9 | dense | Decom-Net first layer 9×9 (4ch→64); rest 3×3 |
| FSRCNN (2016) | 5×5 | dense | 5×5 + thin layers; transposed-conv upsampling at the end |
| HINet (2021) | 4×4 | dense | Downsampling conv only; 3×3 first/last |
| RDIAN (2023) | 7×7 | dense | IRSTD max dense; parallel 1/3/5/7 |
| SDS-Net (2025) | 7×7 | dense | IRSTD; also a 21 strip (see 1.2) |
| YOLOv12 | 7×7 | DW | Position Perceiver; ablating to 9×9 made it slower |
| YOLOv1 | 7×7 | dense | First layer 7×7 stride 2; other YOLO versions use only 3×3 and 1×1 |

The only dense kernel above 9×9 is ESC's 13×13, but it is a 16-channel shared kernel that decomposes into 1×1 + 13×13 DW (see 1.2), so it does not count as a standard dense kernel. → §2.2 Dense max 11×11

### 1.2 How Are Large Kernels Implemented?

Conclusion: large kernels are all DW / strip / dilated / decomposed — four classes; there is no basis for dense 27×27; effective receptive field is bought with strip length and dilation rate, not with a dense window.

**Table RL How large kernels are implemented**

| Paper (year) | Max kernel (stored) | Effective | Type | Channels / Notes |
|---|---|---|---|---|
| PLKSR (2024) | 17×17 | 17 | DW, partial (16/64ch) | 64ch |
| EchoSR (2026) | 17×17 | 17 | DW | — |
| OKNet (2024) | 63×63 | 63 | DW (+strip), bottleneck only | Outlier |
| UIR-PolyKernel (2024) | 31×31 | 31 | DW (+strip), bottleneck at 1/4 resolution; tried 63, chose 31 | — |
| LKMN (2025) | 31 (strip) | 31 | DW strip 31×1+1×31, partial channels | 36ch / 64ch |
| LKFMixer (2025) | 31 (strip) | 31 | DW strip, partial (α=0.25); performance drops above 31 | 36/64ch family |
| SDS-Net (2025) | 21 (strip) | 21 | Strip 1×21+21×1; also 7×7 dense | 32→64→128→128 |
| ESC (2025) | 13×13 | 13 | Dense 16→16 shared, 16ch only; decomposes into 1×1 + 13×13 DW | 16ch |
| SeemoRe (2024) | 11×11 | 11 | DW strip | 36ch / 48ch |
| ALKDNet (2024) | 9 (strip) | 9 | Strip 9×1+1×9; no gain at 11 | 64ch |
| LaKDNet (2023) | 9×9 | 9 | DW (+1×1); defocus 9×9 / motion 7×7 | 32/48ch → bottleneck 256/384ch |
| MAN (2024) | 9×9 (stored) | 35 | DW + DW dilated (d=4) | — |
| LKD-Net (2023) | 21 (effective) | 21 | DW decomposition (small DW + DW dilated) | — |
| RDIAN (2023) | 3×3 (stored) | 17 | Dense + dilated (d=1,2,4,8) | 16→32ch |
| LCRNet (2024) | 7×7 equivalent | 7 | Decomposed DW + dilated DW + 1×1; 11×11 worse (-1.73 IoU) | 16→32→64→64 |
| PConv (2025) | 1×3/3×1 | 25 | Asymmetric pinwheel; effective receptive field | YOLOv8n-p2 host |

Transformer SR (SwinIR/HAT/SRFormer/DAT) convolutions do not exceed 3×3 and buy long range with attention windows 8→16→24; in IRSTD, SWAN and SCTransNet max DW is only 5×5, SeRankDet's 3×3 dilated is effective 9/layer, and VAN has a 7×7 stored kernel with effective 21.

→ §2.2 DW 17×17 / strip 31 / dilated effective ≤35; the 63×63 DW appears only at the bottleneck and is not in the spec.

### 1.3 Channel Counts

Conclusion: mainstream is 64ch; 64 lanes cover the vast majority of dense layers, 128 is comfortable; 256–512ch occur only at the bottleneck (low resolution), where they are tiled.

**Table R7 Channel counts**

| Category | Typical feature channels | Notes |
|---|---|---|
| Input / output | RGB 3ch (grayscale 1ch) | A few 4ch inputs (RetinexNet, EnlightenGAN); SR pixel-shuffle head outputs 3·s²ch |
| Denoising (DnCNN/FFDNet/RIDNet) | 64 (FFDNet RGB: 96) | — |
| Low-light (Zero-DCE/RetinexNet) | 32–64 | — |
| Classic SR (SRCNN/VDSR/SRResNet/RCAN) | 64 | Except EDSR |
| EDSR (final model) | 256 | Widest dense body (baseline 64) |
| Lightweight SR (SeemoRe/LKMN/PLKSR) | 36–64 | Large kernels touch only 16ch or the first 1/4 of channels |
| Transformer SR (SwinIR/HAT) | 180 (HAT-S 144) | Needs tiling to 192 |
| U-shaped restoration base | 32–48 | NAFNet 32; Restormer/LaKDNet 48 |
| U-shaped restoration bottleneck | 256–512 | NAFNet 512; Restormer 384; EnlightenGAN 512 |
| Infrared small target detection (IRSTD) | 16→32→64→64 (LCRNet); 32→64→128→128 (SDS-Net) | Narrower and lighter in recent years; heavy SeRankDet 64→…→1024 |

Large-kernel DW touches only a few channels (PLKSR 16/64ch; ESC 16ch; LKMN/LKFMixer the first 1/4 of channels). → §2.1 Channel tiling

### 1.4 Operators Beyond Convolution

Conclusion: operators beyond convolution are all small units (comparators / adder trees / PWL LUTs / rearrangement) and do not occupy the MAC array; 1×1 and FC reuse the dense datapath.

**Table R8 Operators beyond convolution**

| Operator | Used in which models |
|---|---|
| 1×1 convolution | SRCNN (9-1-5); NAFNet/Restormer (1×1 → 3×3 DW); RCAN channel attention (1×1) |
| pixel-shuffle | SRResNet, SwinIR, EDSR (SR upsampling ×4) |
| Residual addition | VDSR (global residual); SRResNet/EDSR/RCAN (residual blocks); DnCNN (residual learning) |
| pooling / downsampling | Stride-2 convolution, pixel-unshuffle, max-pool (downsampling); GAP (channel attention) |
| Activation | ReLU (general); GELU (Transformer SR); tanh (Zero-DCE last layer); sigmoid (gating/attention) |
| Normalization | BatchNorm (CNN); LayerNorm (Restormer/NAFNet) |
| Elementwise | SimpleGate (NAFNet, channel halving); gating/attention scaling |
| Upsampling | pixel-shuffle; transposed convolution (FSRCNN, HINet); nearest / bilinear |

→ §2.2 Operator set (Table OP).

### 1.5 Application Overview

**Table RP Application overview**

| Application | Max dense | Max DW / strip | Max dilated effective | Channels | One-line takeaway |
|---|---|---|---|---|---|
| Image denoising | 3×3 | 3×3 DW | — | 64ch | A 3×3 world; receptive field from depth (17–20 layers) |
| Super-resolution | 9×9 | 17×17 DW / 31 strip | 35 | 64ch (EDSR 256; lightweight 36–64) | Large kernels are DW / strip / dilated; ESC 13×13 only on 16ch shared |
| Low-light enhancement | 9×9 (first layer only) | 3×3 DW | — | 32–64ch | Mostly 3×3; only RetinexNet's first layer is 9×9 |
| General image restoration | 4×4 (downsampling only) | 63×63 DW (bottleneck only) | 21 | 32–48ch → bottleneck 256–512ch | 63 is an outlier; the rest are ≤21 effective |
| Object detection (YOLO) | 7×7 (first layer) | 7×7 DW | — | — | 9×9 already covers all of YOLO |
| Infrared small target detection (IRSTD) | 7×7 | 21 strip | 17 | Lightweight 16→64ch | Gains are in loss / priors / wavelets, not kernel size |

Across all six applications, max dense is ≤9×9, max DW/strip is ≤31 (63 only as a bottleneck outlier), dilated effective is ≤35, and channels are mainstream 64ch. → All of §2.

## §2 Design Requirements

Goal: one FPGA accelerator that runs complete image-processing models — enhancement / restoration / super-resolution, object detection (YOLO), infrared small target detection (IRSTD), and other CNN models (pixel pipeline at 4K60 real time; complete multi-channel models at the operating points §3's roofline gives; verified on board). Every requirement in this section traces to a research result in §1 (the source is noted in parentheses).

### 2.1 Architecture Requirements

- **Unified reconfigurable compute engine**: dense / DW / dilated / 1×1 / FC — five modes sharing one MAC array (PX=1/2/4, compile-time parameterized), time-multiplexed across layers. Basis: models have many layers (DnCNN 17 layers, EDSR 32 blocks, RCAN 200 RCABs, §1.1); building each layer separately would need 4896 DSPs, over the ZU19EG limit of 1968 (§5).
- **Module reuse (iterative processing) as the default**: a single engine runs the layers in turn; shallow models (≤3–4 layers, e.g., SRCNN) may use a small streaming pipeline. Basis: many layers + DSP limit (same as above).
- **Memory hierarchy and layer fusion**: registers / line buffers → tile buffer (64×64×64ch = 256 KB) → weight buffer (double-buffered) → PL DDR4 → PS DDR4; consecutive layers fuse into one group, the tile stays on chip across layers, and intermediate results do not go to DRAM. Basis: without fusion, DnCNN at 1080p60 needs 271 GB/s (§3), over the PL DDR4 theoretical peak of 38.4 GB/s.
- **Shared DRAM access architecture**: AXI SmartConnect arbitration; priority VDMA (real-time video) > compute engine (throughput) > PS control (best effort); address partitioning; async FIFOs; bandwidth allocation. Basis: the compute engine, VDMA and PS all access DRAM; real-time video must not underflow (§3).
- **AXI interfaces**: AXI4-Lite (control/config registers), AXI4-Stream (pixel stream), AXI4-MM (DDR transfers). Basis: the universal Xilinx IP standard, convenient for system integration.
- **Front-end video preprocessing module**: derived from the frozen streaming convolution (SV), downsampling, channels configurable to 1/3, kernels ≤5×5; it is the input/output interface of the whole model; lightweight, minimal latency. Basis: inputs/outputs are almost all RGB 3ch or grayscale 1ch (§1.3); the video stream needs downsampling and small-kernel filtering at the interface.
- **Adjustable spatial resolution + inverse principle**: W/H runtime-configurable; large kernels appear only at low resolution, and large-kernel configurations use a folded / time-shared datapath. Basis: OKNet's 63×63 appears only at the bottleneck, UIR-PolyKernel's 31 at 1/4 resolution, and LKMN/LKFMixer large kernels touch only partial channels (§1.2).
- **Channel tiling**: C_in tiled and accumulated, C_out tiled and replicated; 64 lanes cover the vast majority of dense layers, 128 is comfortable; 256–512ch occur only at the bottleneck (low resolution), where they are tiled. Basis: channel counts (§1.3).
- **IRSTD receptive field already covered**: the existing operator set (dense 11×11, DW 17×17 / 31 strip, dilated effective ≤35) already exceeds all verified IRSTD requirements (max dense 7×7, max strip 21, dilated effective 17, §1.1/§1.2); larger kernels can even hurt (LCRNet's 11×11 equivalent is 1.73 IoU worse than 7×7).

**Table 1 Memory hierarchy**

| Level | Capacity | Holds | Bandwidth scale |
|---|---|---|---|
| Registers / line buffers | BRAM, K rows | Sliding window (K rows × W) | On-chip, line rate |
| Tile buffer | BRAM, 256 KB–1 MB | Input tile (reused across C_out) + output tile | On-chip, line rate |
| Weight buffer | BRAM, tens–hundreds of KB | Current-layer weights (double-buffered) | On-chip, line rate |
| PL DDR4 | 2×4GB | Feature maps, weights, frame buffers | Theoretical 38.4 GB/s, actual ~27 GB/s (estimate) |
| PS DDR4 | 4GB | Model files, frame buffers, OS | Theoretical 19.2 GB/s, actual ~13 GB/s (estimate) |

**Table 2 PL DDR4 partitions (planned values)**

| Region | Size | Notes |
|---|---|---|
| Input frame buffer (ping-pong) | 2 frames | One 4K RGB frame is 24.9 MB |
| Output frame buffer (ping-pong) | 2 frames | One 4K RGB frame is 24.9 MB |
| Intermediate feature-map spill | As needed | Cut-point spill (1080p/64ch: 133 MB per point) |
| Weight area | As needed | Full-model weights (DnCNN 1.1 MB; EDSR-256 75.5 MB) |
| PS shared area | Reserved | Control and status |

### AXI Interfaces (Unified Standard)

- **AXI4-Lite (control/config)**: configuration registers (kernel size, stride, padding, channel counts, tile size, start/status, interrupts); 32-bit data width; clock is the PS–PL AXI control clock (usually 100 MHz, set in XDC/Tcl).
- **AXI4-Stream (pixel data flow)**: pixel stream (front-end ↔ compute engine ↔ VDMA/output); data width 8×PX bits (16 bits at PX=2, 32 bits at PX=4), TUSER (sof) and TLAST (eol/last) sideband; clock is the processing clock, 300 MHz; valid/ready backpressure.
- **AXI4 memory-mapped (DDR transfers)**: feature maps and weights moved between PL DDR4 and PS DDR4 (engine tile read/write, VDMA frame transfers, PS model loading); data width 64/128 bits (per MIG/HP port configuration); clock is the memory/AXI clock domain (via MIG, set in XDC/Tcl); burst transfers.

AXI is the universal standard for Xilinx IP cores; there is no more general replacement bus. The Versal NoC still uses AXI ports and exists only on Versal; this board (Zynq UltraScale+) uses AXI Interconnect/SmartConnect.

### Front-End Video Preprocessing Module

It sits at the video input (HDMI/SDI IN → front-end preprocessing → VDMA → PL DDR → main accelerator) and at the output (main accelerator → PL DDR → VDMA → front-end → HDMI OUT), and is the input/output interface of the whole model; it processes inline on the pixel stream, with no DRAM round trip.

Data flow: the input video stream (RGB 3ch or grayscale 1ch) passes through front-end downsampling and small-kernel (≤5×5) filtering, then into the main accelerator; the main accelerator's output passes through front-end post-processing and out to HDMI. The front-end connects to the main accelerator via VDMA/PL DDR (frame level); internally the front-end is streaming (row level) and introduces no frame buffering.

Latency budget: front-end pipeline latency ≤ tens of cycles (same order as the original core's 23 cycles); line-buffer wait is about (K−1) rows (~4 rows for a 5×5 kernel), about 26 µs at 4K60 — under 0.2% of a frame, not a bottleneck.

Source and spec: derived from the frozen original streaming convolution (Verilog 1px/2px, `~/workspace/fpga-2d-conv/`); the original RTL is frozen and never modified; the preprocessing module is an independent module derived from it later, written in SystemVerilog. Spec: downsampling (2×/4×, stride-convolution downsampling, configurable); input/output channels configurable to 1 (grayscale) or 3 (RGB); kernel size max 5×5 (smaller than the main accelerator, lightweight); PX=1/2 parameterized; 8-bit pixels, 16-bit coefficients; shift + saturation output (same structure as the existing design). It uses only 25 multipliers/lane (5×5) and 4 line buffers (K−1=4); DSP is 25×PX (50 at PX=2, ~3%).

### 2.2 Operator Set Requirements

**Table OP Operator set requirements**

| Operator | Spec | Models / papers that need it | Implementation / resources |
|---|---|---|---|
| Dense convolution | Max 11×11 | SRCNN/SRResNet/RetinexNet first layers use 9×9 dense (literature dense ceiling, §1.1); one step of headroom to 11 | 121 multipliers/lane; 32-bit accumulator |
| DW square kernel | 17×17 | PLKSR/EchoSR use 17×17 DW (largest square stored kernel in SR, §1.2) | 289 multipliers/lane; touches only partial channels |
| DW strip | 31×1 / 1×31 | LKMN/LKFMixer use 31 strips (largest kernel in SR, §1.2) | 31 multipliers/direction |
| Dilated DW | Stored kernel 3–9, d≤4, effective ≤35 | MAN/VAN/LKA (effective 35/21, §1.2) | Configurable tap spacing; does not enlarge the window |
| 1×1 pointwise | 1×1 | SRCNN 1×1; NAFNet/Restormer 1×1; RCAN channel attention 1×1 (§1.4) | Reuses the dense datapath (K=1) |
| FC | Fully connected | Detection/classification heads (YOLO, §1.5) and general fully connected layers | Reuses the 1×1 MAC array (H=W=1) |
| Pooling | max/avg 2×2, GAP | Downsampling (stride-2 convolution, pixel-unshuffle, max-pool, §1.4); GAP for channel attention | Comparators / adder tree; no DSP |
| Activation | ReLU/Leaky/GELU/sigmoid/tanh | ReLU general; GELU (Transformer SR, §1.4); tanh (Zero-DCE last layer, §1.4); sigmoid (gating/attention) | ReLU zero cost; PWL approximation (sigmoid/tanh/GELU) |
| Normalization | BatchNorm/LayerNorm | BatchNorm (CNN, folded into convolution coefficients); LayerNorm (Restormer/NAFNet, §1.4) | BatchNorm needs no hardware; LayerNorm has a dedicated unit |
| Elementwise | Residual add / multiply / concat | Residual add (VDSR/SRResNet/EDSR/RCAN/DnCNN, §1.4); concat (U-Net skip connections: NAFNet/Restormer/MPRNet, §1.4); multiply (SimpleGate, NAFNet, §1.4) | Adder / small multiplier array; concat is pure data movement |
| Upsampling | pixel-shuffle / nearest / bilinear / transposed convolution | pixel-shuffle (SRResNet/SwinIR/EDSR, §1.4); transposed convolution (FSRCNN, HINet, §1.4) | Channel-space rearrangement; transposed convolution reuses the convolution array |

### Out of Scope (Explicit Boundaries)

- Transformer attention (SwinIR/HAT attention windows) is not done in the PL; only convolution-class operators.
- Training and quantization-aware training; inference only, with coefficients loaded after offline quantization.
- Dense 27×27 (729 multipliers/output/channel; no literature use, not done).
- Floating point; fixed point throughout.
- No general-purpose NPU / programmable instruction set; a dedicated reconfigurable accelerator + runtime-configurable parameters + a layer-table-driven configuration sequence.
- No new modules in Verilog: all implementation in SystemVerilog; the original Verilog streaming convolution is frozen and kept only as a reference baseline, never modified; the preprocessing module is also written in SV when implemented.
- The 2024–26 IRSTD gain points (loss design, target/background priors, dynamic kernels and attention, Haar wavelet/frequency domain, ERF ordering, non-convolution context) are outside the convolution-engine spec; resolution preservation is a system-level requirement: front-end preprocessing and layer partitioning must not aggressively downsample small targets (generally no more than 1/16, never 1/32), keeping high-resolution shallow branches.

## §3 Bandwidth Analysis and Model Partitioning

### 3.1 Roofline

- **Compute roofline**: 1968 DSP × 300 MHz = 590 GMAC/s peak; actual ~70%, i.e., 413 GMAC/s (estimate, including utilization and pipeline overhead).
- **DRAM roofline**: PL DDR4 dual-channel DDR4-2400, theoretical 2×19.2 = 38.4 GB/s, actual ~27 GB/s (estimate, board manual governs); PS DDR4 single-channel, theoretical 19.2 GB/s, actual ~13 GB/s (estimate).
- **Conclusion**: compute is the bottleneck (multi-channel models); DRAM bandwidth is ample after fusion.

### 3.2 Cost of Not Fusing

**Table 4 Per-layer DRAM round-trip bandwidth (read + write)**

| Scenario (per layer, read + write) | Per-layer bandwidth |
|---|---|
| 4K60 / 64ch | 63.7 GB/s |
| 1080p60 / 64ch | 15.9 GB/s |
| 1080p30 / 64ch | 8.0 GB/s |
| 720p30 / 64ch | 3.5 GB/s |
| 960×540 (×4 SR input) / 256ch / 60 fps | 15.9 GB/s |
| 480p30 / 64ch | 1.2 GB/s |

**Table 5 DnCNN 17 layers, unfused total**

| DnCNN 17 layers, unfused total | DRAM bandwidth |
|---|---|
| 1080p60 | 271 GB/s |
| 1080p30 | 135 GB/s |
| 720p30 | 60 GB/s |
| 480p30 | 20 GB/s |

Conclusion: without fusion, 1080p60 needs 271 GB/s — about 7× the PL DDR4 theoretical peak of 38.4 GB/s, impossible; fusion is mandatory. At 480p30 the unfused total of 20 GB/s is already within actual bandwidth, so low resolutions can fuse less.

### 3.3 Bandwidth After Fusion

**Table 6 Fused DRAM traffic (GB/s)**

| Scenario (fused) | Input | Output | Weights | Total |
|---|---|---|---|---|
| DnCNN 1080p60 | 0.37 | 0.37 | 0.07 | 0.81 GB/s |
| DnCNN 4K60 | 1.49 | 1.49 | 0.07 | 3.05 GB/s |
| EDSR-256 ×4 SR (input 960×540, output 4K), 60 fps | 0.09 | 1.49 | 4.5 | 6.2 GB/s |

Conclusion: fusion cuts DRAM traffic by ~300× (271 → 0.81 GB/s at 1080p60); the large-model weight stream (EDSR-256: 75.5 MB per frame, 4.5 GB/s at 60 fps) is the largest DRAM traffic after fusion; the fused total is far below the ~27 GB/s actual bandwidth.

### 3.4 Model Compute and Achievable Operating Points

**Table 7 Model compute and frame rate (peak compute 590 GMAC/s; in parentheses, 70% actual compute, 413 GMAC/s)**

| Model | 1080p | 720p | 480p |
|---|---|---|---|
| SRCNN (3 layers) | 143 GMACs, 4.1 fps (2.9) | 63.7 GMACs, 9.3 fps (6.5) | 21.2 GMACs, 27.8 fps (19.5) |
| DnCNN (17 layers, 64ch) | 1154 GMACs, 0.51 fps (0.36) | 513 GMACs, 1.15 fps (0.81) | 171 GMACs, 3.45 fps (2.4) |
| EDSR-base (16 blocks, 64ch) | 2450 GMACs, 0.24 fps (0.17) | 1089 GMACs, 0.54 fps (0.38) | 363 GMACs, 1.63 fps (1.1) |
| EDSR-base ×4 SR (body at 960×540) | 612 GMACs, 0.96 fps (0.67) | — | — |
| EDSR-256 ×4 SR (body at 960×540) | 19569 GMACs, 0.03 fps | — | — |
| Single layer 64ch 3×3 (reference) | 76.4 GMACs, 7.7 fps (5.4) | 34.0 GMACs, 17.4 fps (12.2) | 11.3 GMACs, 52 fps (36) |

Conclusion: the pixel pipeline (single/few channels) reaches 4K60; shallow models (SRCNN) reach near 30 fps at 480p (peak ~28 fps, actual ~20 fps); deep models (DnCNN) run at ~3.5 fps at 480p and ~0.5 fps at 1080p — offline rates; EDSR-256 is not real-time. Complete multi-channel models cannot reach real time in HD — this is a hard compute-roofline limit, not an implementation problem.

### 3.5 Partitioning Method (Four-Step Decision Flow)

1. Compute the model's total MACs/frame and target fps → required MAC throughput → required DSPs; if over budget, reduce resolution/frame rate or simplify the model (fewer channels/layers).
2. Set cut points by the fusion rules: consecutive layers that fit in the on-chip tile buffer fuse into one group; cut at resolution changes (downsample/upsample boundaries) and skip connections (U-Net).
3. Check each group's on-chip buffers (tile buffer + line buffer + weight buffer ≤ BRAM budget) and DRAM total (≤ 27 GB/s actual bandwidth).
4. Choose streaming / iterative / hybrid (see 3.6).

### 3.6 Streaming vs Iterative vs Hybrid

- **Streaming (layer pipeline, multiple engines)**: shallow models (≤3–4 layers, e.g., SRCNN) or a model's stem/tail; one engine per layer, layers connected directly by small FIFOs/line buffers with no DRAM in between; high throughput, low latency; resources = sum of the engines; only feasible when the engine count is small.
- **Iterative (single engine, time-multiplexed)**: deep models (DnCNN 17 layers, EDSR 32+ blocks); one engine runs the layers in turn; minimal resources (1 set), throughput = single-engine throughput / layer count; the default approach.
- **Hybrid**: group by cut points; streaming within a group (2–4 layers fused into a small pipeline, the tile staying on chip across layers), iterative between groups (the engine switches between groups, inter-group intermediates landing in DRAM); the cut-point count is bounded by resolution and frame rate (see 3.7).

### 3.7 Hard Constraint on Cut-Point Count

- Per-cut-point bandwidth (write + read): 1080p60/64ch, 16.0 GB/s per point; 720p30/64ch, 3.5 GB/s per point; 480p30/64ch, 1.2 GB/s per point.
- PL DDR4 actual is ~27 GB/s; after subtracting VDMA (4K60 input+output ~3.0 GB/s) and weights (negligible for small models), ~24 GB/s remains for cut points.
- Allowed cut points: 1080p60/64ch, at most 1 (16 GB/s); 2 (32 GB/s) would exceed the budget; 720p30/64ch, ~6; 480p30/64ch, ~20 (enough to cut all 17 layers).

Conclusion: at high resolution / high frame rate, cut little (deep fusion); at low resolution, cut more; cut freedom grows as resolution drops.

### 3.8 Partitioning Examples

- SRCNN (3 layers): streaming, 3 small engines pipelined, all on chip, zero DRAM intermediate traffic; ~4 fps at 1080p, ~9 fps at 720p, ~28 fps at 480p (peak).
- DnCNN (17 layers, 64ch): iterative, single engine; at most 1 cut point at 1080p60 (near-full fusion), fully cuttable at 480p30; ~3.5 fps at 480p, ~0.5 fps at 1080p (peak).
- EDSR-base ×4 SR: body at 960×540, iterative; ~1 fps to 4K (peak).

## §4 Implementation Phases and Milestones

Two phases, split by the return to Pittsburgh on 11/26. Phase 0 baseline is done: SV parameterized PX=1/2/4, dense 9×9, golden + STA + resources verified; deliverables are the existing code and two 4-page reports.

### Phase 1 — On the VM: Module Design + Individual Verification + Model Numerical Verification

**Table 8 Phase 1 (stages 1–7, on the VM)**

| Stage | Content, verification and acceptance |
|---|---|
| Stage 1 Dense to 11×11 + arbitrary padding/stride (VM) | K to 11, pad/stride independent per side, output-size formula, decimate. Verification: golden extended (11×11, asymmetric padding, stride 2/3), STA, resources. Acceptance: golden all pass; Fmax ≥ 450 MHz (ASIC model, estimate); DSP 121×PX (exact count). |
| Stage 2 DW + strip + dilated (VM) | DW datapath (17×17 square kernel, 31×1/1×31 strips, dilated stored kernels 3–9/d≤4). Verification: golden (each DW kernel, dilated effective span), STA, resources. Acceptance: golden all pass; DW 17×17 DSP 289×PX (estimate). |
| Stage 3 Multi-channel tiling + 1×1 + residual + pooling + activation (VM) | C_in tiled accumulation, C_out tiling (serial kernel swap first, P_out parallel optional), input tile buffer, 1×1 pointwise, residual addition, max/avg-pool, ReLU/Leaky ReLU (post-processing chain). Verification: multi-channel golden (3/64ch), pooling/activation golden, end-to-end small model (SRCNN 3 layers or a DnCNN fragment). Acceptance: multi-channel output pixel-exact, frame rate on target. |
| Stage 4 Other operators + unified engine + layer scheduler (VM) | FC (reusing 1×1), LayerNorm, elementwise multiply, sigmoid/tanh/GELU (PWL LUT), upsampling (nearest/bilinear), unified engine refactor (five-mode one-click switch), layer scheduler (configuration sequence, weight double-buffering). Activation module structure: the LUT activation stays a separate module (rtl/act.sv) with a BYPASS mode (output = input, framing unchanged); ReLU and Leaky ReLU are FUSED into the conv datapath output stage (after accumulator + shift, config-controlled) so they do not need a separate act pass — the separate LUT module is reserved for complex activations (sigmoid/tanh/GELU/SiLU); the fused ReLU also covers the FC/MLP path (FC reuses the 1×1 datapath, so a ReLU after an MLP/FC layer is fused the same way; only complex activations after FC/MLP, e.g. GELU in a transformer MLP, go through the LUT act module); the compiler decides per layer (fuse ReLU/Leaky into conv, or route complex activations through the LUT act module; bypass covers fused and no-activation cases). Activation multi-level LUT mode: a configurable multi-level lookup table for activations (level-1 coarse table + level-2 fine table, combined by addition, no multiply; levels, bit-splits, and table contents all loaded via config bus; the compiler decides the number of levels, the bit-splits — uniform or non-uniform, steep regions finer — and the lookup) alongside the Stage 3 PWL mode, selectable via config; the compiler chooses the mode per activation. A flat table cannot guarantee precision for steep functions without becoming huge, so the LUT is multi-level by design. Verification: per-operator golden, layer-switch golden (two consecutive layers switch configuration correctly), weight double-buffering timing. Acceptance: all operator goldens pass; layer switching correct. |
| Stage 5 Layer fusion + tile scheduling + shared DRAM architecture + roofline/partitioning + compiler (VM) | Fusion groups and tile scheduler, shared DRAM architecture (arbitration, priority, async FIFOs, address map, bandwidth allocation), roofline model, SRCNN/DnCNN cut points. Compiler module (Python, sw/): based on PyTorch compile graph — torch.compile/TorchDynamo captures the model as an FX graph (torch.export for AOT); a custom backend lowers FX nodes to accelerator ops; the compiler does fusion → activation approximation (PWL table generation per activation, segment count chosen by the accuracy budget and the calibrated input distribution; optional cheap substitutes such as hard-sigmoid/hard-tanh or GELU→SiLU where the accuracy impact is within budget and validated end-to-end) → per-layer tile-size search (cost model: compute time, load time, overlap; ping-pong weight-buffer management, load time ≤ compute time) → scheduling → emits a program (layer descriptors + weight blob + instruction stream) executed by the PS driver (C). Without it every model needs a hand-written configuration sequence. Verification: multi-layer fusion golden (one tile through multiple layers, on-chip intermediates matching golden), DRAM traffic counting (AXI transactions tallied in simulation, checked against the §3 budget), cut-point verification, compiler end-to-end (compile SRCNN/DnCNN fragment/RetinexNet Decom-Net, emitted program runs the golden correctly). Compiler software verification (no RTL needed, before RTL sim): per-pass unit tests (capture, fusion, tiling, scheduling, emission); a Python functional simulator that executes the compiler-emitted program and compares output and per-layer intermediates against the fixed-point golden (and PyTorch float within budget) — compiler bugs are caught in Python, not in RTL sim; only after the software simulator matches does the program go to RTL sim (Stage 6). This separates concerns: software-simulator-vs-golden mismatch = compiler bug, RTL-vs-software-simulator mismatch = hardware bug. Acceptance: fused output pixel-exact; DRAM traffic within budget; partitioning table complete; compiled programs produce pixel-exact output vs golden; compiler passes its software verification suite before any RTL sim. Measured (Stage 5, 2026-11-20, `sw/run_stage5.sh` ALL PASS): 43 per-pass unit tests pass; SRCNN demo functional simulator vs round-half-up golden EXACT (max\|err\|=0) for all 3 layers, end-to-end PSNR vs float 42.55 dB (Stage 6 prep 42.76 dB); RetinexNet demo R 46.50 dB / L 52.68 dB vs float (prep fixed-point 39.71/45.11 dB), conv0 signed intermediate 49.2% negative preserved; C driver host test passes (634 config writes, 592 DMA, 592 GO for SRCNN); RTL weight DMA (`rtl/weight_dma.sv`, AXI-MM 64-bit, up to 16-beat bursts, 2-beat FIFO, strided/contiguous addressing, wfull backpressure) verified in 8 runs (4 cases x clean + backpressure), all data correct, clean cycles within +/-7 of the Python model (A 99 vs 101, B 47 vs 52, C 328 vs 321, D 232 vs 228). Cost model: load hidden in all cases (SRCNN L1 101cy < 649cy, DnCNN L2 52cy < 1048cy, K=11 bank-limited C_in_tile=2, 270cy < 1048cy). DRAM: DnCNN 1080p60 unfused 255.37 GB/s (plan 271 assumes all 17 layers 64ch; actual model has 1ch first/last), fused 0.56 GB/s with 3ch-in/1ch-out (plan 0.81 assumes RGB in+out; grayscale 1ch is 0.31). Roofline: DnCNN 1149 GMACs (plan 1154, 0.4%), fps peak 0.51 / actual 0.36 / 1-engine 0.13 / 4-engine 0.51; RetinexNet 430 GMACs, fps peak 1.37. Known discrepancy: SRCNN 1080p is 16.5 GMACs by computation vs plan Table 7's 143 GMACs -- the plan number appears to assume a different SRCNN variant/resolution; the DnCNN number matches, so the method is validated and the SRCNN plan number is the outlier. Compiler fixes made during Stage 5 verification: the BN fold now folds into the PRECEDING conv (per-output-channel scaling; the old code folded into a following conv, which was wrong and never fired for the standard conv->BN->relu pattern -- verified numerically exact, max\|diff\| ~5e-8 vs the manual BN formula); `fixed_forward` now accepts the program from `compile_model` directly (preferring `_weights_arrays` / `_act_tables_full` over the JSON-safe shapes/summaries). |
| Stage 6 Model numerical verification (VM) | End-to-end small models (SRCNN 3 layers, DnCNN fragment, RetinexNet Decom-Net) run in RTL simulation, RTL output compared pixel-by-pixel against the Python model golden, with a per-layer error budget. Acceptance: model numerical verification passed (pixel-exact or error within budget). |
| Stage 7 Front-end video preprocessing module (VM) | Derived from the frozen original streaming convolution (original RTL untouched): downsampling 2×/4×, channels configurable to 1/3, kernels ≤5×5, PX=1/2, SystemVerilog implementation. Verification: golden (downsampling, 1/3ch, 5×5), STA, latency (line-buffer wait ~4 rows). Acceptance: golden all pass; pipeline latency ≤ tens of cycles; original RTL unmodified. |

### Phase 1 Completion Criteria (What Counts as "Done")

All modules designed and individually verified (golden PASS, STA, resources), at least one end-to-end model numerically verified; code, testbenches, golden models, run scripts, logs and reports all in git and tagged.

### Handoff Items (Taken Back to Pittsburgh)

- RTL source (parameterized SystemVerilog) + testbenches + Python golden models + run scripts;
- Verification logs (golden PASS, STA reports, resource estimates);
- Model numerical verification evidence (RTL vs Python pixel-by-pixel comparison logs, per-layer errors);
- Interface/configuration spec (register map, AXI interfaces, timing);
- Tcl scripts and XDC (written, to run on a Vivado machine);
- Layer-fusion and partitioning tables (SRCNN/DnCNN cut points, fusion groups, tile sizes, DRAM traffic budgets);
- Compiler (Python, sw/) + program format spec (layer descriptors + weight blob + instruction stream) + PS driver (C);
- Shared DRAM architecture document (arbitration, priority, address map, bandwidth allocation);
- Known risks and assumptions list (ASIC-model Fmax vs real timing, LUT as inferred value, Vivado version, DDR4 bandwidth as estimate);
- Front-end video preprocessing module design document (spec: downsampling, 1/3 channels, ≤5×5 kernels, SystemVerilog implementation, interfaces and latency budget with the main accelerator).

### Phase 2 — Pittsburgh (After 11/26): Board-Level Synthesis / Integration / On-Board Testing

**Table 9 Phase 2 (stages 8–11, Pittsburgh)**

| Stage | Content and acceptance |
|---|---|
| Stage 8 Board-level synthesis and implementation (Vivado) | Synthesis, place and route, timing closure, bitstream generation (full command-line Tcl). Acceptance: timing closed, resources within budget. |
| Stage 9 System integration (AXI/VDMA/DDR/PS drivers + shared DRAM architecture on board) | AXI-Lite configuration, AXI-Stream, VDMA frame transfers, PL DDR feature maps, PS drivers and layer-scheduling library, shared DRAM arbitration on board, measured DRAM traffic; on-board loop (HDMI IN → processing → HDMI OUT). Acceptance: 1080p60 real time on board; DRAM traffic matches the budget. |
| Stage 10 On-board verification and model run-through | Complete models on board (denoising, SR, RetinexNet), 4K60 (PX=2) or 1080p60, measuring frame rate/power and timing closure. Acceptance: model output bit-identical to golden (fixed point), frame rate on target, timing closed. |
| Stage 11 Optimization and wrap-up | Bottleneck optimization (barrel-shifter re-pipelining, Fmax improvement), power, documentation and report updates. |

## §5 Resource Estimates and Risks

Resources (ZU19EG: 522K LUT / 1968 DSP / 984 BRAM36 / 1045K FF; estimates unless noted otherwise):

**Table 10 Resource estimates**

| Item | DSP | Other resources and notes |
|---|---|---|
| Dense 11×11 | 121×PX (exact count: PX=1/2/4 → 121/242/484) | BRAM 10; LUT ~1.5× the 9×9 figure (estimate) |
| DW 17×17 | 289×PX (estimate, fully parallel) | In practice touches only partial channels (16ch or 1/4), or folds to reuse the dense datapath (0 extra DSP, multiple passes) |
| Strip 31×1 | 31 multipliers/lane | Negligible |
| Multi-channel | PX × C_par × K² (parameterized) | Choose the tile by budget; BRAM grows with the tile buffer, estimated in the tens |
| Tile buffer (64×64×64ch) | 0 | 256 KB, ~57 BRAM36 (capacity estimate) |
| Pooling (max/avg/GAP) | 0–1 | Tens–hundreds of LUTs (estimate) |
| Activation (ReLU/Leaky/PWL) | 0–1/lane (PWL interpolation) | 1 BRAM (PWL LUT) + hundreds of LUTs (estimate) |
| LayerNorm | A few (estimate) | Hundreds of LUTs (estimate) |
| Layer scheduler + DRAM arbitration | 0 | Hundreds–thousands of LUTs + async FIFOs (several BRAMs, estimate) |

Reference configurations (estimates): PX=2, K=3 mainstream layer, C_par=16 → DSP 288 (15%); K=11 setting, C_par reduced to 4 → DSP 968 (49%). Whole-machine typical occupancy (estimate): DSP 40–50%, LUT 6–10%, BRAM 8–15% (tile and weight buffers push BRAM above the original estimate), all subject to Vivado synthesis.

Reuse comparison: building each layer as a separate module (17-layer DnCNN, DSP 4896, over the 1968 limit, infeasible); one unified engine (DSP 288, 15%) + dedicated small units (pooling/activation/normalization, DSP 0–a few, LUT hundreds–thousands) runs all 17 layers, time-multiplexed.

DRAM bandwidth demand: fused DnCNN 1080p60 ~0.81 GB/s, 4K60 ~3.05 GB/s; EDSR-256 ×4 SR ~6.2 GB/s; all far below the PL DDR4 actual ~27 GB/s (estimate). Cut-point budget: see §3.7.

### Risks and Countermeasures

| Risk | Description and countermeasure |
|---|---|
| Insufficient compute (deep models cannot reach HD real time) | Quantified in §3: DnCNN at 1080p reaches only 0.51 fps (peak). Countermeasure: lower resolution/frame rate, lightweight models (36ch), offline processing; do not use HD real time as the acceptance standard for deep models. |
| DDR bandwidth | Large multi-channel feature maps round-tripping through DDR may bottleneck. Countermeasure: layer fusion + tile-buffer reuse (input tile reused across C_out), cut-point budget (§3.7). |
| Large-channel bottleneck layers (256–512ch) | Time grows linearly with channel count. Countermeasure: tiling + low pixel rate at low resolution, still real time; reduce PX or raise the clock if needed. |
| Timing closure | The barrel shifter is the bottleneck (ASIC model 2.09 ns); Vivado Fmax to be measured; if it does not reach 300 MHz, the countermeasure is re-pipelining the shifter (already verified approach, ASIC model reaches 600 MHz+). |
| Board-level timing unknown before returning to Pittsburgh | Vivado Fmax not yet measured; if VM-phase verification is incomplete, board work slips. Countermeasure: the Phase 1 completion criteria and handoff list exist to prevent this. |
| Vivado is not on this machine | Synthesis/implementation can only run on a Vivado machine. Countermeasure: full command-line Tcl scripts (§7), versioned, reproducible on any same-version Vivado machine. |
| Coefficient loading | Large models have many coefficients (C_in×C_out×K²); EDSR-256 weights are 75.5 MB, reloading every frame needs 4.5 GB/s. Countermeasure: coefficients stored in PL DDR, loaded on chip as needed; layered loading; weight double-buffering overlapped with compute. |
| Fixed-point precision | Quantization error accumulates. Countermeasure: per-layer golden comparison (Python float vs fixed point), error budget. |
| Layer-scheduling overhead | Inter-layer switching (weight loading + register configuration) may slow frame time. Countermeasure: weight double-buffering overlapped with compute; switching overhead ~7 µs/layer, negligible against millisecond-scale layer compute (already quantified in §2). |

## §6 Input and Output Expectations

Input (full-time-equivalent estimates; actual calendar time depends on the pace of work):

| Phase | Input |
|---|---|
| VM phase (stages 1–7) | ~11–14 weeks (full-time equivalent) |
| Pittsburgh phase (stages 8–11) | ~5–6 weeks (full-time equivalent) |
| Total | ~16–20 weeks |

Output:

- An FPGA accelerator that runs complete image-processing models — enhancement / restoration / super-resolution, object detection (YOLO), infrared small target detection (IRSTD), and other CNN models (pixel pipeline 4K60, complete models at the §3 operating points), implemented in SystemVerilog.
- Complete verification: golden (function), STA (timing), on-board measurement (frame rate / power / timing closure), measured DRAM traffic (checked against the §3 budget).
- A reproducible full command-line flow (§7) and updated project reports.

## §7 Methodology and Toolchain: Xilinx Full Command Line (No GUI)

Principle: synthesis, implementation, bitstream generation, and programming/debugging all run via vivado batch / Tcl scripts, no GUI; any step can be replayed with a single command.

### Directory and Script Structure

| Directory / file | Description |
|---|---|
| `fpga-accel/` | (This repository root) |
| `rtl/` | SystemVerilog source (parameterized) |
| `tb/` | Testbenches + golden (Python) |
| `scripts/` | Tcl scripts |
| `scripts/synth.tcl` | Synthesis (read_verilog -sv, synth_design, write checkpoint + utilization report) |
| `scripts/impl.tcl` | Implementation (place_design, route_design, timing report) |
| `scripts/bitstream.tcl` | Bitstream generation |
| `scripts/program.tcl` | Programming (connect_hw_server, program_hw_device) |
| `scripts/debug.tcl` | ILA capture (create_hw_probe, run_hw_ila, export waveforms) |
| `scripts/common.tcl` | Shared parameters (device, clocks, file list, version) |
| `constraints/` | XDC (clocks, pins, timing constraints) |
| `sw/` | PS drivers + layer-scheduling library (C) |
| `Makefile` | One-click reproduction: make sim / synth / impl / bitstream / program |
| `logs/` | Logs and reports from each run (synthesis utilization, timing, power) |

Version control: RTL + Tcl + XDC + software all in git; bitstreams and synthesis products are not in version control (.gitignore) and are reproduced by scripts; each milestone is tagged; Tcl scripts check the Vivado version at the start and record it in the logs.

Reproduction: on any machine with the same Vivado version, `git clone` then `make all` reproduces synthesis → implementation → bitstream; simulation and STA are reproduced on this VM with Icarus/Yosys/OpenSTA (existing flow); on-board verification uses `make program` + ILA Tcl capture, with logs archived.

Toolchain: simulation/STA on this VM (Icarus 13.0, Yosys 0.52, OpenSTA 2.0.17, Python 3, Nangate45); synthesis/implementation/on-board on a Vivado machine (version locked, checked by Tcl).

## Appendix: Decision Basis (Literature)

Dense max 9×9 (SRCNN, SRResNet, RetinexNet first layers); largest square kernel in SR is 17×17 DW (PLKSR); strip 31 (LKMN/LKFMixer); largest restoration DW is 63×63, bottleneck only (OKNet); dilated stored kernels ≤9, d≤4, effective ≤35 (MAN/VAN). Infrared small target detection (IRSTD): max dense 7×7, max strip 21, dilated effective 17; the 2024–26 gains are in loss / priors / dynamic kernels / wavelets / resolution preservation, not kernel size.

Channel counts: RGB 3ch; mainstream feature width 64ch; lightweight 36–64ch; Transformer SR 180ch; EDSR 256ch; U-shaped restoration bottleneck 256–512ch; 64 lanes cover the vast majority of dense layers, 128 is comfortable.

Compute and bandwidth basis: ZU19EG 1968 DSP × 300 MHz = 590 GMAC/s peak; PL DDR4 dual-channel DDR4-2400, theoretical 38.4 GB/s, actual ~27 GB/s (estimate); feature-map sizes and frame rates are in the §3 tables; bandwidth figures are computed from feature-map size, channel count and frame rate.

**Table 13 Decision comparison**

| Decision | Value | Literature basis |
|---|---|---|
| Dense max kernel | Literature 9×9; design extends to 11×11 | SRCNN, SRResNet, RetinexNet first layers |
| DW max square kernel | 17×17 | PLKSR (SR) |
| DW strip | 31 | LKMN / LKFMixer |
| Largest restoration DW | 63×63, bottleneck only | OKNet |
| IRSTD max kernel | dense 7×7 / strip 21 / dilated effective 17 | SDS-Net, LCRNet, RDIAN |
| Dilated | Stored kernels ≤9, d≤4, effective ≤35 | MAN / VAN |
| Channel width | Mainstream 64ch; bottleneck 256–512ch | Lightweight 36–64ch; Transformer SR 180ch; EDSR 256ch |
| Module reuse | Unified engine time-multiplexed; no per-layer separate modules | Building 17-layer DnCNN separately needs 4896 DSP, over the 1968 limit (§2) |
| Layer fusion | Intermediate results stay on chip; DRAM only for input/output/weights | Unfused 1080p60 needs 271 GB/s, over the DDR4 peak (§3) |
| Front-end preprocessing | Downsampling + 1/3ch + kernels ≤5×5; lightweight, latency ≤ tens of cycles | Based on the frozen original streaming convolution (9/9 golden, 23-cycle latency) |
