# Stage 7c design: integrated end-to-end + post-synthesis simulation

## Why this stage exists

Stages 1-7b verified the pieces separately: the compiler + functional
simulator (Stage 5), per-module RTL goldens (Stages 1-4, 7), the model
chains through one engine at a time (Stage 6: conv2d_mc; Stage 7b:
pe_array), the C driver host test (Stage 5), and timing (STA on
synthesized netlists). Two legs were missing:

1. **Integrated end-to-end simulation**: no single flow ever fed a real
   image through the frontend RTL, then the PE array driven by the
   compiled program, and checked the result against the expected data
   flow. The frontend, the PE array, and the driver each had their own
   testbench; nothing chained them.
2. **Post-synthesis simulation**: Yosys synth + STA check timing and
   area, but nothing ever re-ran a golden check against a synthesized
   netlist. A synthesis bug (a mis-inferred memory, a dropped
   assignment, a wrong cell mapping) would pass STA silently.

Stage 7c builds both legs, in that order. No new RTL datapath is
added: the integration is at the driver level (Python orchestrates
RTL phases) plus one thin wrapper testbench. The C driver
(sw/driver) is NOT in the loop yet: it targets the K^2 program
format (W/DMA/GO/WAIT/STREAM); the PE array is programmed by
tile-steps, and no C driver exists for the tile-step program.
Driving the array from the compiled program in Python
(sw/emit.py::emit_pe_program) is the Stage 7c scope; a tile-step
C driver belongs to Stages 8-11 (board integration).

## Part 1 — integrated end-to-end simulation

### Data flow (one driver, `sw/run_e2e.py`, per model)

```
real test image (Stage 6 input builder)
  -> frontend RTL (3x3 Gaussian blur, same size)      [phase 1, vvp]
  -> frontend RTL output (the model's true input)
  -> emit_pe_program: compiled tile-step program      [Python compiler]
  -> pe_array RTL, one wave per layer, RTL output
     feeding the next layer (Stage 6 pattern)         [phase 2, vvp]
  -> [RetinexNet only] act.sv PWL sigmoid            [phase 3, vvp]
  -> [DnCNN only] residual.sv mode 3 (denoised)      [phase 3, vvp]
  -> final image, checked vs the composed golden
```

The golden is COMPOSED: the frontend golden
(`tb/golden_frontend.py::golden`, the Stage 7 golden) feeds the
per-layer funcsim golden (`sw/funcsim.py::conv2d_fixed_rhu`, the
Stage 6 golden). The float reference is the float model run on the
BLURRED input (same composition in float). The acceptance is the
same as Stages 6/7b: per-layer RTL vs funcsim max|err| = 0
(RetinexNet sigmoid: PWL budget max|err| <= 1, as in Stage 6),
final PSNR vs float reported.

The frontend blur changes the model input, so the e2e PSNR numbers
are NOT comparable to Stage 6's (which used the unblurred input).
They are reported on their own terms; the blur is a real
preprocessing step, not a test artifact.

### Why two vvp passes (not one TB)

The tile-steps for layer N+1 depend on the frontend's RTL output,
which is only known after phase 1 runs; the scheduler does im2col
in software, by design, so the RTL cannot generate its own
tile-steps. The driver therefore runs phase 1 (frontend), reads
the RTL output frame, emits the program, and runs phase 2
(PE array). The integration is real at every stage: a frontend bug
changes the array's input and fails the composed golden; a
scheduler bug changes the tile-steps and fails the per-layer
golden. Python only does the compiler's job (scheduling) and the
golden check, never the datapath.

### The frontend phase (reuses the Stage 7 TB, no new RTL)

`tb/tb_frontend.sv` is reused unchanged: the driver writes a
1-case `vectors_frontend.txt` (the test image + blur config) and
reads `out_frontend.txt`. The vectors file is generated, not
tracked, so overwriting it is harmless (Stage 7 regenerates it).
Blur config (all models): K=3, shift=8, pad=1 all sides (same
size), stride 1, DS=1, sign=0, act=0; 3x3 Gaussian
[1 2 1; 2 4 2; 1 2 1]/16 as int16 coeffs (x16) in the 5x5 grid,
row stride 5. Channels: 1 for SRCNN/DnCNN, 3 for RetinexNet RGB
(see below).

### The PE phase (new TB around the wrapper)

`tb/tb_e2e.sv` instantiates `rtl/srcnn_pipe_pe.sv` (the top
wrapper), not `pe_array` directly: the wrapper is the integrated
top, and it has no testbench today. The sender/collector are the
proven `tb_pe_array.sv` pattern (same vectors format, same
`vectors_pe_array.txt` / `out_pe_array.txt` contract), with the
DUT swapped to the wrapper and a config-bus readback check added
(read 0x00/0x01/0x02 -> N_PE/TM/TN, read 0x04 done_count after
the run; done_count counts completed TILE-STEPS, not outputs --
several K-chunks merge into one output block, so the check is
done_count == nsteps). The wrapper is a wire passthrough for tile-steps plus
config registers, so this TB verifies its port mapping and its
registers; the datapath is the array's. `tb_pe_array.sv` itself
is untouched (a Stage 7b artifact; the Stage 7b chain keeps
using it).

### The compiled program drives the array

The driver uses `sw/emit.py::emit_pe_program(layers, inp,
device)` — the compiled tile-step program — not direct
`gen_layer_steps` calls. `emit_pe_program` emits all layers'
steps upfront from the golden intermediates; the driver executes
them wave by wave and verifies each layer's RTL output against
its golden before proceeding (fail fast). This is sound because
a layer whose RTL output matches its golden exactly produces
bit-identical inputs for the next layer, so the upfront steps
are exactly the steps the RTL-fed flow needs. Layer dicts come
from a new adapter, `layers_from_program(program)`, which reads
the compiled program's conv ops (K, C_out, pad, pad_mode,
shift, fused_act, signed_in/out, W_q, b_q): act maps
relu->1 / leaky->2 / none->0, mode_dot=1 iff C_out==1
(SRCNN L3, DnCNN L17; the scheduler's TN<8 guard still
applies). Padding is reproduced by PRE-PADDING the layer's
input in the driver (zero for DnCNN, replicate for RetinexNet)
and running a valid conv in the scheduler — the scheduler has
no pad support (its A-tile gather assumes a valid conv). The
driver cross-checks the padded golden against
`fixed_forward`'s per-layer golden for every layer; a mismatch
is a driver bug, fail fast.

### Models and sizes (Stage 6 test inputs, small crops)

| model | input | frontend | PE chain | tail | note |
|-------|-------|----------|----------|------|------|
| SRCNN | 24x24x1 (bicubic upsampled test gray) | 1ch blur, same size | L1 9x9 1->64, L2 1x1 64->32, L3 5x5 32->1 (dot); all valid conv, outputs 16/16/12 (all %4==0) | — | the reference flow |
| DnCNN | 16x16x1 (clean + Gaussian noise sigma=25) | 1ch blur, same size | all 17 layers, full 64ch, pad=1 zero (pre-padded); outputs stay 16x16 | residual.sv mode 3 -> denoised | CLOSES the Stage 6 gap: L4-L15 are RTL-run here for the first time (same 3x3 64->64 shape, now actually run, not sampled) |
| RetinexNet | 16x16x4 ([max,R,G,B] of low-light RGB) | 3ch blur on RGB; max channel recomputed from the blurred RGB (the input construction, applied to the preprocessed image) | conv0 9x9 4->64, pad=4 replicate (16->16); conv1-5 + recon 3x3 with replicate pre-pad (16->16); signed conv0/recon (sign_in/out from the program ops) | act.sv PWL sigmoid (r_in=3, budget <=1 LSB, as Stage 6) | R/L split is the sigmoid output's ch0-2 / ch3, as in Stage 6 |

Step counts (zu19eg, TM=16/TN=8): SRCNN 737 steps (L1 384,
L2 128, L3 225); DnCNN ~35k steps (L1 128, L2-L16 2304 each,
L17 288); RetinexNet 13,344 steps (conv0 1536, conv1-5 2304 each,
recon 288). DnCNN is the long pole (~47x SRCNN);
the driver prints per-layer wall times.

The scheduler requires every layer output divisible by 4
(M-tile 4x4). All three models satisfy this at the sizes
above (SRCNN valid conv 16/16/12; DnCNN/RetinexNet same-size
via pre-padding). A full 1080p RTL sim is infeasible in
Icarus and is NOT attempted; the 1080p claim stays with the
cycle model (Stage 7b), as before.

### Signed intermediates (RetinexNet)

conv0 and recon are signed_out (the program ops carry
`signed_in`/`signed_out`). The RTL output bytes are int8
two's-complement; the driver reinterprets them as signed
(byte>127 -> byte-256) for the golden comparison and for the
next layer's A tiles (written as negative ints; the TB packs
`tmp[7:0]`, so the bits are identical either way). The PE
golden already covers sign_in=1 (outer K-dim 64, dot signed).
The golden for a signed layer is `conv2d_fixed_rhu(...,
signed_out=True)`, which returns int64.

## Part 2 — post-synthesis simulation

### Route: Icarus + simcells, same TB, same golden

Yosys generic `synth -top <mod>` -> `write_verilog` netlist ->
Icarus with `/usr/share/yosys/simcells.v` + the SAME testbench
and the SAME golden check. Yosys 0.52's own `sim` command exists
but needs a separate stimulus harness; Icarus + simcells reuses
the existing TBs unchanged (only the DUT source changes from
RTL to netlist), so a pass means the synthesized netlist is
functionally identical to the RTL on the golden vectors. The
netlist is from the GENERIC synth, not synth_ecp5: the ECP5
netlist contains TRELLIS/LUT4 cells that simcells.v does not
model, while the generic netlist contains $_AND_/$_DFF_P_/
$mem_v2 cells that it does. The ECP5 synth already proves the
vendor mapping (Stage 7b); the generic netlist proves functional
equivalence. SDF back-annotated timing sim is out of scope (no
SDF flow in the repo; STA covers timing).

### The parameter problem (found 2026-11-21, probe in /tmp/psprobe)

Both golden TBs instantiate their DUT with parameter overrides
(`adder_tree #(.N(160)/81/64/25)`, `frontend #(.MAX_W(3840))`),
but a synthesized netlist has no parameters: Icarus rejects the
override with "parameter `N` not found" (verified with a minimal
probe: synth a parameterized module, rename nothing, compile the
TB against the netlist -- elaboration fails). The netlist
therefore cannot be dropped in place of the RTL. The fix is a
parameter-accepting wrapper (pure port passthrough, no logic)
named like the RTL module, around the renamed netlist(s):
`tb/adder_tree_ps_wrap.sv` (one netlist per N, selected by
generate-if; an N with no netlist fails fast) and
`tb/frontend_ps_wrap.sv` (one netlist, default MAX_W=3840, the
TB's value; fails fast if MAX_W != 3840). The TBs are compiled
unchanged; only the DUT implementation changes from RTL to
netlist. The same wrapper pattern applies to the Mac Phase 3
modules (pe_mac / pe_array TBs also use parameter overrides --
see the Phase 3 notes in run_ecp5_mac.sh).

### Per-module plan

| module | synth on VM? | post-synth sim | TB / golden reused |
|--------|--------------|----------------|-------------------|
| adder_tree | yes (fast) | VM, this stage | tb_adder_tree.sv smoke test (200 vectors x 4 sizes), via tb/adder_tree_ps_wrap.sv (4 netlists, generate-if) |
| frontend | yes (full-module synth_xilinx completed in Stage 7; generic synth to be tried here) | VM, this stage | tb_frontend.sv + golden_frontend.py check (16 cases), via tb/frontend_ps_wrap.sv |
| pe_mac | NO (OOMs on this VM, ~6 GB sandbox; Mac peak 5.28 GB) | Mac, via run_ecp5_mac.sh Phase 3 (new) | tb_pe_mac.sv + golden_pe.py (11 cases), via tb/pe_mac_ps_wrap.sv (TM=16/TN=8 netlist) |
| pe_array | NO (needs the Mac, 24 GB) | Mac, via run_ecp5_mac.sh Phase 3 (new) | tb_pe_array.sv + the SRCNN chain driver (sw/run_pe_srcnn.py --netlist), via tb/pe_array_ps_wrap.sv (N_PE=4 netlist) |
| conv2d_mc / conv2d_dw / conv2d_dense / unified_engine | NO (full-module synth times out on the VM, as in Stages 1-4) | out of scope (see below) | — |

The VM runner reports the Mac items as explicit SKIP lines
("SKIP on VM, Mac Phase 3"), not as passes and not as
failures — the Stage 7b precedent (the ECP5 array synth was
"pending" on the VM and done on the Mac). The Mac run is
pending user execution, as run #3 was.

### Out of scope (recorded, not hidden)

- K^2 engines (conv2d_mc, conv2d_dw, conv2d_dense,
  unified_engine) post-synth sim: full-module synth times out
  on the VM; they are superseded reference baselines (the PE
  array is the deliverable), so their netlist sim is not
  attempted here. A Mac generic-synth attempt is possible
  later, not in this stage.
- YOLOv8n / DNANet: the weights are being onboarded in
  parallel (separate work): YOLOv8n (ultralytics yolov8n.pt)
  + DNANet (NUAA-SIRST checkpoint) downloaded + hash-verified
  to ~/workspace/model-onboarding/ on 2026-11-21, but loading
  the checkpoints + quantizing + registering under models/ is
  blocked pending execution-risk approval (torch.load can run
  embedded code). No model-level end-to-end exists for them in
  this stage; the scope stays SRCNN/DnCNN/RetinexNet unless
  the first three finish with time to spare.
- Full-resolution (1080p) RTL sim: infeasible in Icarus.
- SDF timing sim: no SDF flow; STA covers timing.
- The C driver for the tile-step program: does not exist
  (sw/driver targets the K^2 program format); Stages 8-11.

## Runner and verification bar

`tb/run_stage7c.sh` (fail-fast, same conventions as
run_stage7b.sh: `set -o pipefail`, real exit codes captured
into rc, explicit PASS/FAIL per check, overall verdict):
Part 1 (SRCNN, RetinexNet, DnCNN via `sw/run_e2e.py`, one
check per model), Part 2 (adder_tree + frontend post-synth
sim on the VM; pe_mac/pe_array explicit SKIP with the Mac
pointer), then regressions (`tb/run_stage7b.sh` +
`sw/run_stage5.sh` + `sw/run_regressions.sh`). The runner
must print `STAGE 7c RESULT: ALL PASS`. Logs under
`logs/stage7c/` (per-model e2e logs, post-synth synth/sim
logs, runner log). Docs updated in the same commit:
`rtl/README.md` + `sw/README.md` + `docs/plan.md` (Stage 7c
row) + this file.

### VM replacements and per-layer checkpoints (added 2026-11-21)

The VM is replaced every few tens of minutes (two replacements
during this stage: 01:17 UTC and ~01:40 UTC on 2026-11-22,
each killing the running model and wiping /usr/bin Icarus +
Yosys -- reinstall from ~/tools-dl/*.deb, never apt-get
update). A single long model run therefore cannot be assumed
to finish: RetinexNet (~25 min) was killed twice, at 3/7 and
4/7 layers, both times with every completed layer at
max|err|=0. The driver checkpoints after every passed layer
(`sw/run_e2e.py --resume`): `<outdir>/ckpt_<model>/` holds
one `layer_<name>.npy` (the RTL output bytes) + `ckpt.json`
per completed layer, written atomically (tmp file +
os.replace, so a kill mid-write never leaves a half-written
checkpoint; a layer whose .npy fails to load truncates the
checkpoint there, since later layers depend on it). A
re-launch with --resume re-runs the fast frontend phase and
the compile, then skips the checkpointed layers' vvp and
continues from the last passed layer. A run without --resume
ignores any checkpoint (and overwrites it as layers re-pass).
The Stage 7c runner invokes every model with --resume, so a
killed runner can be re-invoked and continues instead of
restarting. The same hazard applies to the regression scripts
(run_stage5.sh / run_regressions.sh / run_stage7b.sh, ~40 min
combined) -- they have no resume; if a replacement kills
them, re-run them.

## Files

| File | What |
|---|---|
| `logs/stage7c/design.md` | this file |
| `sw/run_e2e.py` | integrated driver (frontend phase + PE chain + act/residual tails) |
| `sw/tests/test_e2e.py` | unit tests for the Python side (layer adapter, padding equivalence, frontend golden reuse) |
| `tb/tb_e2e.sv` | wrapper TB (srcnn_pipe_pe + config readback, tb_pe_array pattern) |
| `tb/run_stage7c.sh` | fail-fast runner (Part 1 + Part 2 + regressions) |
| `tb/run_postsynth.sh` | Part 2 runner (VM: adder_tree 4 netlists + frontend netlist; Mac items explicit SKIP) |
| `tb/adder_tree_ps_wrap.sv` | post-synth wrapper: parameter-accepting adder_tree shell over 4 renamed netlists |
| `tb/frontend_ps_wrap.sv` | post-synth wrapper: parameter-accepting frontend shell over the renamed netlist |
| `tb/pe_mac_ps_wrap.sv` | post-synth wrapper for the Mac Phase 3 pe_mac check (TM=16/TN=8 netlist) |
| `tb/pe_array_ps_wrap.sv` | post-synth wrapper for the Mac Phase 3 pe_array check (N_PE=4 netlist) |
| `sw/run_pe_srcnn.py` | --netlist/--simcells options added (Stage 7c): compile the pe_array netlist + simcells instead of the RTL; default path unchanged |
| `~/workspace/your_files/run_ecp5_mac.sh` | Phase 3 added: Mac post-synth sim for pe_mac + frontend + pe_array (generic synth netlist + simcells) |
| `logs/stage7c/doc-updates.md` | the Stage 7c text for docs/plan.md + rtl/README.md, staged here because the VCU128 agent holds uncommitted changes in both files (2026-11-22); apply after that work lands |

## Results (2026-11-21/22, all measured, logs under logs/stage7c/)

### Part 1 — integrated end-to-end

| Model | Frontend (RTL blur) | Per-layer RTL vs funcsim | Final vs float |
|---|---|---|---|
| SRCNN 24x24 | max\|err\| = 0 | L1/L2/L3 max\|err\| = 0 (384/128/225 steps, 737 total) | PSNR 42.75 dB (blurred input; Stage 6's 43.51 dB was unblurred -- not comparable) |
| RetinexNet 16x16x4 | max\|err\| = 0 | all 7 layers max\|err\| = 0 (13,344 steps); sigmoid max\|err\| = 1 (PWL budget <= 1) | R PSNR 48.18 dB, L PSNR 54.25 dB |
| DnCNN 16x16, 17 layers full 64ch | max\|err\| = 0 | every layer max\|err\| = 0 (34,976 steps) | denoised max\|err\| = 0, PSNR 53.41 dB |

DnCNN L4-L15 are RTL-run here for the first time (Stage 6
sampled them by shape) -- the Stage 6 gap is CLOSED. Per-layer
vvp wall times ~0.09-0.13 s/step; DnCNN total run ~60-70 min.
Logs: e2e_srcnn.log + e2e_srcnn.json, e2e_retinexnet.log,
e2e_dncnn.log. The VM was replaced three times during Stage 7c
(2026-11-22 01:17, ~01:40, ~03:17 UTC); RetinexNet was killed
twice mid-run (at 3/7 and 4/7 layers, all completed layers
max|err| = 0) before the checkpoint mechanism existed, and
finished on the third run. Checkpoints for retinexnet (7
layers) and dncnn (17 layers) live in ckpt_retinexnet/ and
ckpt_dncnn/.

### Part 2 — post-synthesis simulation

Route: Yosys generic `synth -top <mod>` -> write_verilog
netlist -> Icarus + simcells.v + the SAME TB + SAME golden
check. Generic synth, not synth_ecp5 (ECP5 netlists have
TRELLIS/LUT4 cells simcells.v does not model). SDF
back-annotated timing sim is out of scope (no SDF flow; STA
covers timing).

| Module | Netlist | Result |
|---|---|---|
| adder_tree N=160/81/64/25 | generic synth PASS (25,379 / 12,896 / 10,026 / 3,955 cells) | smoke PASS (200 vectors x 4 sizes) -- netlists functionally identical to RTL |
| frontend (MAX_W=3840) | generic synth PASS (193,923 cells) | SIM is Mac Phase 3 (see limitation below) |
| pe_mac (TM=16/TN=8) | Mac Phase 3 (netlist OOMs on this VM) | pending user execution |
| pe_array (N_PE=4) | Mac Phase 3 (netlist OOMs on this VM) | pending user execution |

`tb/run_postsynth.sh` final log: POSTSYNTH RESULT: ALL PASS
(13 PASS, 0 FAIL). A synthesized netlist has no parameters,
so the TBs' parameter overrides cannot be applied to it
(Icarus: "parameter `N` not found", verified with a minimal
probe); the fix is a parameter-accepting wrapper (pure port
passthrough) named like the RTL module around the renamed
netlist(s). Netlists are reused across runs when the RTL is
unchanged (staleness-checked), so a VM replacement does not
force a re-synth.

### Limitations not closed

- The frontend post-synth SIM does not run on this VM. The
  generic netlist is 193,923 cells because the line buffer
  synthesizes to 92,952 FFs + 92,904 muxes (the generic
  `memory` pass cannot map this multi-port line buffer to
  $mem_v2; the vendor synths do map it to BRAM, 12x BRAM36 in
  synth_xilinx), and Icarus elaboration does not complete on
  this VM (40+ min, 2.3 GB RSS, memory pressure -- killed
  2026-11-22). The VM runner reports it as an explicit SKIP
  and the sim runs in Mac Phase 3 (run_ecp5_mac.sh), which
  also does pe_mac and pe_array. Mac Phase 3 is pending user
  execution (the user runs the script and sends back
  logs/stage7c/mac_ps_*.log + logs/stage7c/pe_array_ps_chain.log).
- Full 1080p RTL sim is infeasible in Icarus; the 1080p claim
  stays with the cycle model.
- YOLOv8n + DNANet weights are being onboarded in parallel
  (separate work); no model-level end-to-end exists for them
  yet.
- VCU128 platform support is separate work (another agent):
  it touched docs/plan.md + rtl/README.md (uncommitted at the
  time Stage 7c finished), so the Stage 7c doc text for those
  two files is staged in logs/stage7c/doc-updates.md instead of
  being merged directly. It also modified tb/run_stage7b.sh
  (added a §7 VCU128 cycle-model check) and the regression
  logs, and was running the full regression suite when Stage
  7c finished its own parts -- the Stage 7c runner's regression
  section is therefore run after the VCU128 agent's run
  finishes (the two cannot share tb/*.vvp and the log files),
  and its results are recorded in the runner log.
  VCU128 verification/simulation updates are NOT Stage
  7c scope (ZU19EG only).
