#!/bin/bash
# run_stage3.sh -- Stage 3 verification: multi-channel tiling + 1x1 +
# residual + pooling + activation.
#   1. Generate all vector files (golden_*.py gen)
#   2. Compile + simulate each testbench (iverilog 13.0)
#   3. Check simulation output against the golden models (golden_*.py check)
#   4. Re-run Stage 1 and Stage 2 regressions
# Each module runs independently; a failure in one does not skip the others.
# Run from the tb/ directory:
#   bash run_stage3.sh
# Logs are written to ../logs/stage3/ and also echoed here.
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

LOGDIR=../logs/stage3
mkdir -p "$LOGDIR"
FAIL=0

run_mod() {
    # $1 = name, $2 = rtl file, $3 = tb file, $4 = vvp, $5 = golden script
    local name="$1" rtl="$2" tb="$3" vvp="$4" golden="$5"
    echo "=== Stage 3: $name ==="
    if ! "$IVERILOG" -g2012 -o "$vvp" "$rtl" "$tb" > "$LOGDIR/compile_$name.log" 2>&1; then
        echo "COMPILE FAILED ($name)"; cat "$LOGDIR/compile_$name.log"; FAIL=1; return
    fi
    timeout 1800 vvp "$vvp" > "$LOGDIR/sim_$name.log" 2>&1
    local rc=$?
    cat "$LOGDIR/sim_$name.log"
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
echo "=== Stage 3: generating vectors ==="
python3 golden_mc.py gen       | tee "$LOGDIR/vectors_mc.log"       || FAIL=1
python3 golden_pool.py gen     | tee "$LOGDIR/vectors_pool.log"     || FAIL=1
python3 golden_residual.py gen | tee "$LOGDIR/vectors_residual.log" || FAIL=1
python3 golden_act.py gen      | tee "$LOGDIR/vectors_act.log"      || FAIL=1

# ---------------- the four Stage 3 modules ----------------
run_mod mc       ../rtl/conv2d_mc.sv tb_conv2d_mc.sv tb_mc.vvp  golden_mc.py
run_mod pool     ../rtl/pool.sv      tb_pool.sv       tb_pool.vvp golden_pool.py
run_mod residual ../rtl/residual.sv  tb_residual.sv   tb_res.vvp  golden_residual.py
run_mod act      ../rtl/act.sv       tb_act.sv        tb_act.vvp  golden_act.py

# ---------------- regressions ----------------
echo "=== Stage 1 regression (14 cases) ==="
bash run_stage1.sh > "$LOGDIR/regress_stage1.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage1.log"
grep -q "GOLDEN CHECK PASSED" "$LOGDIR/regress_stage1.log" || FAIL=1

echo "=== Stage 2 regression (23 cases) ==="
bash run_stage2.sh > "$LOGDIR/regress_stage2.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage2.log"
grep -q "GOLDEN CHECK PASSED" "$LOGDIR/regress_stage2.log" || FAIL=1

echo "=== STAGE 3 RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILED) ==="
exit $FAIL
