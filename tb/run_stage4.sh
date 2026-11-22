#!/bin/bash
# run_stage4.sh -- Stage 4 verification: remaining operators (LayerNorm,
# upsample, concat, SimpleGate) + signed DW + multi-level LUT / bypass /
# fused ReLU in act + unified reconfigurable engine (mode switch) +
# layer scheduler (3-layer sequence + ping-pong weights).
#   1. Generate all vector files (golden_*.py gen)
#   2. Compile + simulate each testbench (iverilog 13.0)
#   3. Check simulation output against the golden models (golden_*.py check)
#   4. Re-run Stage 1, Stage 2 and Stage 3 regressions
# Each module runs independently; a failure in one does not skip the others.
# The MC sim is the slowest (~19 min for 15 cases); the DW sim (~8 min
# for 26 cases) and the scheduler sim (~10 min for 2 runs x 3 layers)
# are also long. Use timeout 1800 per module, as in run_stage3.sh.
# Run from the tb/ directory:
#   bash run_stage4.sh
# Logs are written to ../logs/stage4/ and also echoed here.
set -u
set -o pipefail
# Tool resolution: iverilog must be available. A VM reset can remove
# /usr/bin tools, and a nohup/background shell may have a different PATH.
# Prefer PATH, fall back to the absolute path, fail fast if missing.
IVERILOG="$(command -v iverilog || true)"
[ -n "$IVERILOG" ] || IVERILOG=/usr/bin/iverilog
if [ ! -x "$IVERILOG" ]; then
    echo "ERROR: iverilog not found (checked PATH and /usr/bin/iverilog)." >&2
    echo "Reinstall from ~/tools-dl/iverilog_13.0-2_amd64.deb" >&2
    exit 1
fi
cd "$(dirname "$0")"

LOGDIR=../logs/stage4
mkdir -p "$LOGDIR"
FAIL=0

run_mod() {
    # $1 = name, $2 = golden script, $3.. = rtl+tb files (last is the TB)
    local name="$1" golden="$2"; shift 2
    local vvp="tb_${name}.vvp"
    echo "=== Stage 4: $name ==="
    if ! "$IVERILOG" -g2012 -o "$vvp" "$@" > "$LOGDIR/compile_$name.log" 2>&1; then
        echo "COMPILE FAILED ($name)"; cat "$LOGDIR/compile_$name.log"; FAIL=1; return
    fi
    timeout 1800 vvp "$vvp" > "$LOGDIR/sim_$name.log" 2>&1
    local rc=$?
    tail -n 12 "$LOGDIR/sim_$name.log"
    if [ $rc -ne 0 ]; then
        echo "SIM FAILED ($name) rc=$rc"; FAIL=1; return
    fi
    python3 "$golden" check > "$LOGDIR/check_$name.log" 2>&1
    cat "$LOGDIR/check_$name.log"
    grep -q "GOLDEN CHECK PASSED" "$LOGDIR/check_$name.log" || FAIL=1
}

# ---------------- vector generation ----------------
# NOTE (fixed 2026-11-20): with pipefail a failed gen makes the pipeline
# fail, and `|| FAIL=1` records it (previously tee's rc=0 masked it).
echo "=== Stage 4: generating vectors ==="
python3 golden_layernorm.py gen   | tee "$LOGDIR/vectors_layernorm.log" || FAIL=1
python3 golden_upsample.py gen    | tee "$LOGDIR/vectors_upsample.log"  || FAIL=1
python3 golden_concat.py gen      | tee "$LOGDIR/vectors_concat.log"    || FAIL=1
python3 golden_simplegate.py gen  | tee "$LOGDIR/vectors_simplegate.log"|| FAIL=1
python3 golden_dw.py gen          | tee "$LOGDIR/vectors_dw.log"        || FAIL=1
python3 golden_mc.py gen          | tee "$LOGDIR/vectors_mc.log"        || FAIL=1
python3 golden_act.py gen         | tee "$LOGDIR/vectors_act.log"       || FAIL=1
python3 golden_unified.py gen     | tee "$LOGDIR/vectors_unified.log"   || FAIL=1
python3 golden_sched.py gen       | tee "$LOGDIR/vectors_sched.log"     || FAIL=1

# ---------------- the Stage 4 modules ----------------
run_mod layernorm golden_layernorm.py ../rtl/vision/layernorm.sv tb_layernorm.sv
run_mod upsample golden_upsample.py ../rtl/vision/upsample.sv tb_upsample.sv
run_mod concat golden_concat.py ../rtl/vision/concat.sv tb_concat.sv
run_mod simplegate golden_simplegate.py ../rtl/vision/simplegate.sv tb_simplegate.sv
run_mod dw golden_dw.py ../rtl/vision/conv2d_dw.sv tb_conv2d_dw.sv
run_mod mc golden_mc.py ../rtl/vision/conv2d_mc.sv tb_conv2d_mc.sv
run_mod act golden_act.py ../rtl/vision/act.sv tb_act.sv
run_mod unified golden_unified.py ../rtl/common/unified_engine.sv ../rtl/vision/conv2d_dense.sv ../rtl/vision/conv2d_dw.sv tb_unified_engine.sv
run_mod sched golden_sched.py ../rtl/common/layer_scheduler.sv ../rtl/common/unified_engine.sv ../rtl/vision/conv2d_dense.sv ../rtl/vision/conv2d_dw.sv tb_layer_scheduler.sv

# ---------------- regressions ----------------
echo "=== Stage 1 regression (14 cases, dense now with sign + fused act) ==="
bash run_stage1.sh > "$LOGDIR/regress_stage1.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage1.log"
grep -q "GOLDEN CHECK PASSED" "$LOGDIR/regress_stage1.log" || FAIL=1

echo "=== Stage 3 regressions (pool 8, residual 7; mc + act already run above) ==="
python3 golden_pool.py gen | tee "$LOGDIR/vectors_pool.log" || FAIL=1
python3 golden_residual.py gen | tee "$LOGDIR/vectors_residual.log" || FAIL=1
run_mod pool golden_pool.py ../rtl/vision/pool.sv tb_pool.sv
run_mod residual golden_residual.py ../rtl/vision/residual.sv tb_residual.sv

echo "=== STAGE 4 RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILED) ==="
exit $FAIL
