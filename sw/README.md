# Stage 5 Compiler (sw/)

The Stage 5 compiler (Python) lowers a PyTorch model to a program
the accelerator executes. It is verified in Python **before** any RTL
simulation (Stage 6): the functional simulator is the reference Stage
6 compares the RTL against.

## Pipeline

```
torch model (models_torch.py, real pretrained weights)
  -> capture_torch (ir.py, torch.export AOT FX graph)
  -> fuse (passes/fusion.py)
  -> infer_shapes (compile.py)
  -> quantize_graph (passes/quantize.py)
  -> activation approximation (passes/actapprox.py + passes/mlut.py)
  -> tile_search (costmodel.py)
  -> emit_program (emit.py)
  -> program.json + program.txt + weights.bin
  -> fixed_forward (funcsim.py, the functional simulator)
```

Entry point: `compile_model(graph, calib_input, input_shape,
model_name)` in `compile.py`. It returns `(program, instrs, blob,
report)`.

## Modules

| File | What it does |
|---|---|
| `ir.py` | Minimal FX-graph-like IR (Node/Graph) + `capture_torch` (torch.export AOT) + hand-built `build_graph`. torch 2.15 export decomposes nn.Conv2d to `aten.conv2d` (args: input, weight, bias, stride, padding, dilation, groups, positional); parameters are lifted to placeholder nodes whose names map to state_dict keys via `ep.graph_signature.inputs_to_parameters`; F.pad replicate becomes `aten.pad`. |
| `models_torch.py` | torch nn.Module definitions for SRCNN, DnCNN fragment, RetinexNet Decom-Net, loading the float weights from `models/`. RetinexNet float npz keys are L1..L7 (L1=conv0 9x9, L2-L6=conv1-5, L7=recon); the quant.json uses conv0..recon. |
| `passes/fusion.py` | Fusion: pad->conv fold, BN fold (into the **preceding** conv, per-output-channel scaling -- the standard transformation), conv+ReLU/Leaky fuse (fused_act in the conv output stage), complex act -> separate `act` node, elementwise add/sub -> `residual`, mul -> `simplegate`. Fusion groups cut at residual/simplegate/upsample/concat/layer_norm or tile-buffer overflow (256 KB). Returns `(graph, groups, report)`. |
| `passes/quantize.py` | Quantization: float forward on the calibration input -> per-layer r_out via range_exp -> `quantize_layer` (models/common.py) -> W_q/b_q/shift/e attached to each node. Signed intermediates: a conv with no fused act and min<0 gets signed_out=True with r_out = range_exp(2*max\|output\|) so \|value\| <= 127. RTL shift = e + r_out - r_in, kept in [0,15]. |
| `passes/actapprox.py` | PWL table generation (32 uniform segments over [-8,8], y0/dslope Q1.15, dslope = the segment *difference*, not the slope), cheap substitutes (hard-sigmoid/hard-tanh, GELU->SiLU/ReLU), range clipping. `choose_act_impl` picks PWL or MLUT. PWL output is Q1.15 (range [-1,1)), so it can only represent sigmoid/tanh; GELU/SiLU reach ~8 and MUST use the MLUT (Q5.11). Default error budget 0.03 (0.01 is too tight for tanh/GELU/SiLU with a 2-level LUT). |
| `passes/mlut.py` | Multi-level LUT generation, reproducing the Stage 4 golden method exactly (H1/H2/nbands, mean-based fine tables, Q5.11) + a search over (H1,H2,nbands) for the smallest table meeting the budget + config-write emission. Verified: the Stage 4 golden split (H1=8,H2=4,8 bands, 384 entries) reproduces the exact Stage 4 numbers (sigmoid 0.0027, tanh 0.0186, GELU 0.0269, SiLU 0.0234). Search (budget 0.03): sigmoid 80 entries, tanh 288, GELU 264, SiLU 264. The hardware implements 2 levels (act.sv mode 6); 3/4-level tables are NOT emitted for the current hardware (a follow-up, not silently treated as executable). |
| `costmodel.py` | Tile-size search (`tile_search`, `LayerShape`). Compute per (o,t) pass = H_out_tile*W_out_tile + 24 cycles latency (1 px/cycle, K^2 cancels). Weight load is DRAIN-limited at 1 coeff/cycle (the config-bus write rate into the bank; the 64-bit AXI bus is faster than the drain) -- AXI-MM load = ncoeff + n_bursts*4 + 12 cycles, vs 2 cycles/coeff for the Stage 4 config bus (~2x faster in cycles, plus autonomous). A 4-wide bank write would match the AXI bus rate but needs a scheduler hardware change (follow-up). Constraints: C_in_tile <= min(C_PAR=4, 289/K^2, C_in) -- the 289-coeff bank is the binding constraint for large K; tile buffer <= 256 KB. Results: SRCNN L1 (K=9,Cin=1): C_in_tile=1, load 101cy, compute 649cy, hidden; DnCNN L2 (K=3,Cin=64): C_in_tile=4, load 52cy, compute 1048cy, hidden; K=11 (Cin=64): C_in_tile=2 (bank-limited: 2*121=242 <= 289), load 270cy, compute 1048cy, hidden. Load is hidden in all cases; the optimum varies per layer. |
| `dram.py` | DRAM model + AXI-MM weight DMA cycle model (`weight_dma_cycles`, must match rtl/weight_dma.sv). `dram_traffic` reproduces the plan's numbers from layer shapes (not hard-coded): DnCNN 1080p60 unfused 255.37 GB/s (plan Table 5 says 271 -- the plan assumes all 17 layers are 64ch in+out; the actual model has 1ch first/last layers, so 255 is accurate); fused 0.56 GB/s with in_ch=3/out_ch=1 (plan Table 6 says 0.81, which assumes RGB 3ch in AND out; the repo's DnCNN is grayscale 1ch -> 0.31 GB/s with in_ch=1). Weights 0.07 GB/s. VDMA 4K60 in+out = 2.99 GB/s, leaving ~23 GB/s for the engine. |
| `roofline.py` | Roofline analysis. Peak 590.4 GMAC/s, actual 413.3, single engine (484 DSP) 145.2 GMAC/s, 4 engines fit (1936 DSP, 98%). DnCNN 1080p 1149 GMACs (plan Table 7 says 1154 -- 0.4% difference, a counting convention; method validated), fps peak 0.51, actual 0.36, 1-engine 0.13, 4-engine 0.51. RetinexNet 1080p 430 GMACs, fps peak 1.37. SRCNN 1080p = 16.5 GMACs by computation vs plan Table 7's 143 GMACs -- a real discrepancy: the plan's 143 appears to assume a different SRCNN variant/resolution (the 9-1-5 model in this repo is 16.5 GMACs at 1080p); the DnCNN number matches, so the method is validated and the SRCNN plan number is the outlier. |
| `funcsim.py` | Functional simulator + float forward. `conv2d_tiled` mirrors the RTL accumulation order (per (o,t) pass, accumulate into int64, then round-half-up shift + saturate + fused act). `fixed_forward` executes the emitted program (conv/act/residual/pool/upsample/concat/layer_norm/bypass). Act nodes use the compiler's generated tables (MLUT/PWL), so the simulator validates the compiler's table generation. `fixed_forward` accepts the program from `compile_model` directly: it prefers `program["_weights_arrays"]` / `program["_act_tables_full"]` (the runtime arrays) over the JSON-safe `program["weights"]` / `program["act_tables"]` (which hold only shapes/summaries). |
| `emit.py` | Emission: program.json + program.txt (instruction stream: W/DMA/GO/WAIT/STREAM) + weights.bin (int16 LE, hardware order (o,t,l,k) for multi-channel conv2d_mc; contiguous tap row-major for single-channel scheduler layers). Two paths: multi-channel conv -> conv2d_mc driven directly by the driver (DMA with lane_stride=128, taps=K^2); single-channel conv -> scheduler + unified engine (bank L%2, DMA to 0x0200/0x0400, contiguous). |
| `compile.py` | The pipeline (capture -> fusion -> shapes -> quantization -> act tables -> tiling -> emission) plus `infer_shapes`. |
| `driver/` | Minimal C driver skeleton (driver.h/c, mock_mmio.h/c, test_driver.c, Makefile). Reads program.txt + weights.bin, issues the config sequence (W/DMA/GO/WAIT/STREAM) via an MMIO abstraction. The host test (gcc) verifies the driver issues the expected write sequence for the SRCNN program (634 config writes, 592 DMA, 592 GO) and that the coefficient memory holds the right values. Full PS integration (VDMA frame streaming, interrupts) is Stages 8-11. Note: the MC and scheduler register maps overlap (both use 0x0000-0x0012 as offsets within their own module), so the driver tracks the execution path from the `# layer ... (scheduler ...)` / `(conv2d_mc ...)` comment in program.txt, not from the config address. |
| `stage5_report.py` | Prints the cost-model / DRAM / roofline numbers (see below). |
| `run_stage5.sh` | Master runner (unit tests + SRCNN demo + RetinexNet demo + driver test + RTL DMA + report). Reports PASS/FAIL. |
| `run_regressions.sh` | Re-runs Stage 1-4 regressions. Fails if any stage fails (checks the script exit code AND the pass marker in the check log AND the .vvp artifact). See the header for the false-pass fixes (2026-11-20). |

## Verification flow (before any RTL sim)

1. **Per-pass unit tests** (`sw/tests/`, stdlib unittest, 43 tests):
   `python3 -m unittest discover -s sw/tests`. Covers capture
   (torch.export), fusion, actapprox/PWL, MLUT (incl. the Stage 4
   number reproduction), quantize, costmodel, DRAM, roofline, emit,
   funcsim (incl. a forced-MLUT GELU end-to-end check).
2. **Functional simulator** (`funcsim.fixed_forward`): executes the
   compiler-emitted program and compares output + per-layer
   intermediates against the fixed-point golden (and PyTorch float
   within budget). Compiler bugs are caught here, in Python.
3. **RTL sim (Stage 6)**: only after the functional simulator
   matches. Separation: functional-sim vs golden mismatch =
   compiler bug; RTL vs functional-sim mismatch = hardware bug.

## Acceptance (SRCNN demo)

`python3 sw/demo_srcnn.py` (Stage 5 acceptance test): compile SRCNN
end-to-end through the compiler -> functional simulator -> compare
vs the fixed-point golden. Result: functional simulator vs
round-half-up golden EXACT (max|err|=0) for all 3 layers (tiling
does not change the integer sum). End-to-end PSNR vs float 42.55 dB
(Stage 6 prep: 42.76 dB, max|err| 6.14 vs 5.79; the small difference
is the round-half-up vs floor shift). Vs the floor-shift golden the
errors are 1/5/10 LSBs per layer (informational -- shift rounding
propagates and amplifies through the 64-channel 1x1 layer; this is
expected, not a bug).

`python3 sw/demo_retinexnet.py`: R PSNR 46.50 dB (Stage 6 prep
fixed-point: 39.71 dB), L PSNR 52.68 dB (prep: 45.11 dB) vs the float
reference (replicate padding). conv0 signed intermediate: range
[-83,78], 49.2% negative -- the signed path preserves the signal.

## RTL vs modeled (Stage 5)

- **RTL**: `rtl/weight_dma.sv` (AXI-MM weight-streaming DMA, 64-bit,
  up to 16-beat bursts, 2-beat FIFO, strided/contiguous addressing,
  wfull backpressure) + `tb/tb_weight_dma.sv` + `tb/run_stage5_rtl.sh`.
  Verified: 4 cases x clean + backpressure = 8 runs, all data
  correct; clean cycle counts within +/-7 cycles of
  `sw/dram.py weight_dma_cycles` (A 99 vs 101, B 47 vs 52, C 328 vs
  321, D 232 vs 228). The DMA is RTL; board-level AXI integration
  (SmartConnect, DRAM controller, PS handshake) is Stages 8-11.
- **Modeled (Python)**: DRAM traffic (`dram.py`), roofline
  (`roofline.py`), tile cost model (`costmodel.py`), the functional
  simulator (`funcsim.py`). These are quantified with numbers, not
  hand-waving (see `stage5_report.py`).
- **C driver**: `sw/driver/` is a skeleton (host-tested with a mock
  MMIO). Full PS integration is Stages 8-11.

## Known discrepancies (recorded, not hidden)

- SRCNN 1080p MACs: computed 16.5 GMACs vs plan Table 7's 143
  GMACs. The DnCNN number matches (1149 vs 1154), so the method is
  validated; the plan's SRCNN number is the outlier.
- DnCNN DRAM: unfused 255.37 GB/s (plan 271, all-64ch assumption)
  and fused 0.56 GB/s with 3ch-in/1ch-out (plan 0.81, RGB in+out).
- A 4-wide bank write would match the AXI bus rate but needs a
  scheduler hardware change (follow-up).
- 3/4-level MLUT tables are not emitted for the current act.sv
  (2 levels only); recorded as a follow-up.

## Running

```bash
bash sw/run_stage5.sh        # everything (unit tests + demos + driver + DMA RTL + report)
python3 -m unittest discover -s sw/tests   # unit tests only
python3 sw/demo_srcnn.py --outdir sw/out/srcnn
python3 sw/demo_retinexnet.py --outdir sw/out/retinexnet
make -C sw/driver test       # C driver host test
bash tb/run_stage5_rtl.sh    # RTL weight DMA
bash sw/run_regressions.sh   # Stage 1-4 regressions
```
