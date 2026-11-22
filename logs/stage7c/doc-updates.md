# Stage 7c doc updates — APPLIED 2026-11-22

The VCU128 platform work landed as commit 96f73bd (pushed,
origin/main = 96f73bd) before the Stage 7c runner finished, so
the text below was applied directly to `docs/plan.md` (Stage 7c
row after the Stage 7b row) and `rtl/README.md` (Stage 7c
section at the end of the file) in the Stage 7c commit. This
file is kept as the record of what was added. `sw/README.md`
already carries the Stage 7c section (it was untouched by the
VCU128 agent).

## 1. docs/plan.md — new table row after the Stage 7b row

| Stage 7c Integrated end-to-end + post-synthesis simulation (VM) | Two verification legs that were missing. (1) Integrated end-to-end: a real test image flows through the Stage 7 frontend RTL (3x3 Gaussian blur) into the Stage 7b PE array driven by the compiled tile-step program, one wave per layer, RTL output feeding the next layer; the final image is checked against the COMPOSED golden (frontend golden -> per-layer funcsim golden, max\|err\| = 0) and against float (PSNR). SRCNN 24x24, RetinexNet 16x16x4, DnCNN 16x16 all 17 layers full 64ch (Stage 6 test inputs). The srcnn_pipe_pe wrapper gets its first testbench (tb/tb_e2e.sv). Two vvp passes per model, not one TB: the tile-steps for layer N+1 depend on the frontend's RTL output, and the scheduler does im2col in software by design. Padding is reproduced by pre-padding the layer input in the driver (zero for DnCNN, replicate for RetinexNet) + valid conv in the scheduler; emit_pe_program is extended (pad / pad_mode / signed_out, backward-compatible) and the driver cross-checks the padded golden against fixed_forward's per-layer golden for every layer. (2) Post-synthesis: Yosys generic synth -> write_verilog netlist -> Icarus + simcells.v + the unchanged golden TB (generic synth, not synth_ecp5 -- ECP5 netlists have TRELLIS/LUT4 cells simcells.v does not model; SDF back-annotated timing sim out of scope, STA covers timing). A synthesized netlist has no parameters, so the TBs' parameter overrides cannot be applied to it (Icarus: "parameter `N` not found", verified by probe); the fix is a parameter-accepting wrapper (pure port passthrough) named like the RTL module around the renamed netlist(s). VM scope: adder_tree (4 netlists) + frontend synth; pe_mac / pe_array and the frontend SIM run on the Mac (24 GB, run_ecp5_mac.sh Phase 3). Full 1080p RTL sim is infeasible in Icarus -- the 1080p claim stays with the cycle model. YOLOv8n + DNANet weights are being onboarded in parallel (separate work); no model-level end-to-end exists for them yet. Verification: tb/run_stage7c.sh (fail-fast; e2e x3 with --resume, post-synth, then Stage 7b + Stage 5 + Stage 1-4 regressions) prints STAGE 7c RESULT: ALL PASS. Measured (Stage 7c, 2026-11-22): SRCNN -- frontend max\|err\| = 0, L1/L2/L3 max\|err\| = 0 (384/128/225 steps), final PSNR vs float 42.75 dB (blurred input; Stage 6's 43.51 dB was unblurred -- not comparable); RetinexNet -- frontend max\|err\| = 0, all 7 layers max\|err\| = 0 (13,344 steps), sigmoid max\|err\| = 1 (PWL budget), final R PSNR 48.18 dB, L PSNR 54.25 dB; DnCNN -- frontend max\|err\| = 0, all 17 layers max\|err\| = 0 (34,976 steps), denoised max\|err\| = 0, PSNR vs float 53.41 dB (L4-L15 RTL-run here for the first time -- the Stage 6 gap is CLOSED); post-synth -- adder_tree 4 netlists (N=160/81/64/25; 25,379 / 12,896 / 10,026 / 3,955 cells) smoke test PASSED (200 vectors x 4 sizes), frontend generic synth PASSED (193,923 cells). Known limitations: the frontend post-synth SIM does not run on this VM -- the line buffer synthesizes to 92,952 FFs + 92,904 muxes in generic synth (the generic `memory` pass cannot map this multi-port line buffer to $mem_v2; vendor synths map it to BRAM, 12x BRAM36) and Icarus elaboration does not complete in 40+ min (2.3 GB RSS, memory pressure) -- it is Mac Phase 3; pe_mac / pe_array post-synth are Mac Phase 3 (netlists OOM on this VM); the VM was replaced three times during Stage 7c, so run_e2e.py checkpoints after every passed layer (--resume) and run_postsynth.sh reuses netlists when the RTL is unchanged. |

## 2. rtl/README.md — new section at the end of the file

## Stage 7c: integrated end-to-end + post-synthesis simulation

Two verification legs that were missing: one system sim where a
real image flows in and the result is checked against the
expected data flow, and a functional check of the
Yosys-synthesized netlist (STA today only checks timing, never
functional correctness of the netlist). The design, per-model
numbers, and evidence are in logs/stage7c/design.md.

### Integrated end-to-end (sw/run_e2e.py, tb/tb_e2e.sv)

A real test image streams through the frontend RTL (3x3 Gaussian
blur, same size), the frontend's RTL output feeds the PE array
driven by the compiled tile-step program, one wave per layer with
the RTL output feeding the next layer, and the final image is
checked against the COMPOSED golden (frontend golden ->
per-layer funcsim golden) and the float reference (float model
on the blurred input). Two vvp passes per model, not one TB:
the tile-steps for layer N+1 depend on the frontend's RTL
output, and the scheduler does im2col in software by design.
The srcnn_pipe_pe wrapper finally has a testbench (tb/tb_e2e.sv:
the tb_pe_array sender/collector pattern with the wrapper DUT,
plus a config-bus readback check for N_PE/TM/TN and
done_count -- done_count counts completed tile-steps, not
outputs). Tails are RTL as in Stage 6: RetinexNet act.sv PWL
sigmoid (budget <= 1 LSB), DnCNN residual.sv mode 3. Padding is
reproduced by pre-padding the layer input in the driver and
valid conv in the scheduler; the driver cross-checks the padded
golden against fixed_forward's per-layer golden for every layer
(mismatch = driver bug, fail fast).

### Post-synthesis simulation (tb/run_postsynth.sh)

Yosys generic `synth -top` -> write_verilog netlist -> Icarus +
simcells.v + the unchanged golden TB. Generic synth, not
synth_ecp5 (ECP5 netlists have TRELLIS/LUT4 cells simcells.v
does not model). A synthesized netlist has no parameters, so the
TBs' parameter overrides cannot be applied to it directly
(Icarus: "parameter `N` not found", verified by probe); the fix
is a parameter-accepting wrapper (pure port passthrough) named
like the RTL module around the renamed netlist(s):
tb/adder_tree_ps_wrap.sv (4 netlists, generate-if),
tb/frontend_ps_wrap.sv, tb/pe_mac_ps_wrap.sv,
tb/pe_array_ps_wrap.sv. Netlists are reused across runs when
the RTL is unchanged (staleness-checked), so a VM replacement
does not force a re-synth. sw/run_pe_srcnn.py gained
--netlist/--simcells for the pe_array chain against a netlist
(the default RTL path is unchanged).

### Measured results (freshly run, traceable to `logs/stage7c/`)

`sw/run_e2e.py --model srcnn` (log `logs/stage7c/e2e_srcnn.log`,
`e2e_srcnn.json`): frontend 24x24 blur max|err| = 0; L1/L2/L3
max|err| = 0 (384/128/225 steps); final PSNR vs float 42.75 dB
(blurred input; Stage 6's 43.51 dB was unblurred -- not
comparable).

`sw/run_e2e.py --model retinexnet` (log
`logs/stage7c/e2e_retinexnet.log`): frontend 16x16x3 blur
max|err| = 0; all 7 layers max|err| = 0 (13,344 steps);
sigmoid RTL vs funcsim max|err| = 1 (within the PWL budget);
final R PSNR 48.18 dB, L PSNR 54.25 dB.

`sw/run_e2e.py --model dncnn` (log `logs/stage7c/e2e_dncnn.log`):
frontend 16x16x1 blur max|err| = 0; all 17 layers max|err| = 0
(34,976 steps); denoised vs funcsim max|err| = 0, vs float
PSNR 53.41 dB. L4-L15 are RTL-run here for the first time
(Stage 6 sampled them by shape) -- the Stage 6 gap is closed.

`tb/run_postsynth.sh` (log `logs/stage7c/run_postsynth.log`):
POSTSYNTH RESULT: ALL PASS (13 PASS, 0 FAIL). adder_tree: 4
generic synth PASS (N=160: 25,379 cells; N=81: 12,896; N=64:
10,026; N=25: 3,955), netlists renamed, post-synth compile
PASS, smoke PASS (200 vectors x 4 sizes) -- the netlists are
functionally identical to the RTL. frontend: generic synth
PASS (193,923 cells); the SIM is Mac Phase 3 (see below).
pe_mac / pe_array: SKIP -- Mac Phase 3.

### Known limitations (Stage 7c)

- The frontend post-synth SIM does not run on this VM: the
  generic netlist is 193,923 cells because the line buffer
  synthesizes to 92,952 FFs + 92,904 muxes (the generic
  `memory` pass cannot map this multi-port line buffer to
  $mem_v2; the vendor synths do map it to BRAM, 12x BRAM36
  in synth_xilinx), and Icarus elaboration does not complete
  on this VM (40+ min, 2.3 GB RSS, memory pressure -- killed
  2026-11-22). It runs in Mac Phase 3
  (~/workspace/your_files/run_ecp5_mac.sh), which also does
  pe_mac and pe_array (netlists OOM on this VM). Mac Phase 3
  is pending user execution.
- Full 1080p RTL sim is infeasible in Icarus; the 1080p claim
  stays with the cycle model.
- YOLOv8n + DNANet weights are being onboarded in parallel
  (separate work); no model-level end-to-end exists for them
  yet.
- The VM was replaced three times during Stage 7c. run_e2e.py
  checkpoints after every passed layer (--resume) and
  run_postsynth.sh reuses netlists when the RTL is unchanged,
  so a replacement costs a re-launch, not a full re-run.

### Regressions re-run with Stage 7c

Stage 7b runner ALL PASS (`tb/run_stage7b.sh`,
`logs/stage7c/regress_stage7b.log`). Stage 5 acceptance ALL
PASS (`sw/run_stage5.sh`, `logs/stage7c/regress_stage5.log`).
Stage 1-4 RTL regressions ALL PASS (`sw/run_regressions.sh`,
`logs/stage7c/regress_stage1_4.log`). (Run by
`tb/run_stage7c.sh`; the log names are filled in from the run.)
