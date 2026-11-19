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
cd "$(dirname "$0")"

LOGDIR=../logs/stage3
mkdir -p "$LOGDIR"
FAIL=0

run_mod() {
    # $1 = name, $2 = rtl file, $3 = tb file, $4 = vvp, $5 = golden script
    local name="$1" rtl="$2" tb="$3" vvp="$4" golden="$5"
    echo "=== Stage 3: $name ==="
    if ! iverilog -g2012 -o "$vvp" "$rtl" "$tb" > "$LOGDIR/compile_$name.log" 2>&1; then
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
echo "=== Stage 3: generating vectors ==="
python3 golden_mc.py gen       | tee "$LOGDIR/vectors_mc.log"
python3 golden_pool.py gen     | tee "$LOGDIR/vectors_pool.log"
python3 golden_residual.py gen | tee "$LOGDIR/vectors_residual.log"
python3 golden_act.py gen      | tee "$LOGDIR/vectors_act.log"

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
