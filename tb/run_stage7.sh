#!/bin/bash
# run_stage7.sh -- Stage 7 verification: front-end preprocessing module.
#   1. Generate vectors (golden_frontend.py gen)
#   2. Compile + simulate the frontend testbench (iverilog 13.0)
#   3. Check simulation output against the golden model (golden_frontend.py check)
#   4. Re-run Stage 5 acceptance and the Stage 1-4 regressions
# Run from the tb/ directory:
#   bash run_stage7.sh
# The frontend sim is fast (small images, one position/cycle); the Stage
# 1-4 regressions are the long pole (~40 min). Logs go to ../logs/stage7/.
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

LOGDIR=../logs/stage7
mkdir -p "$LOGDIR"
FAIL=0

echo "=== Stage 7: frontend golden ==="
python3 golden_frontend.py gen | tee "$LOGDIR/gen.log" || FAIL=1
if ! "$IVERILOG" -g2012 -o tb_fe.vvp ../rtl/frontend.sv tb_frontend.sv \
    2>&1 | tee "$LOGDIR/compile.log"; then
    echo "[compile] FAILED (iverilog)" >&2
    exit 1
fi
echo "[compile] rc=0"
t0=$(date +%s)
timeout 900 vvp tb_fe.vvp > "$LOGDIR/vvp.log" 2>&1
rc=$?
t1=$(date +%s)
echo "[vvp] rc=$rc wall=$((t1-t0))s"
cat "$LOGDIR/vvp.log"
if [ $rc -ne 0 ]; then
    echo "SIM FAILED (frontend) rc=$rc"; FAIL=1
fi
python3 golden_frontend.py check | tee "$LOGDIR/check.log"
grep -q "GOLDEN CHECK PASSED" "$LOGDIR/check.log" || FAIL=1

echo "=== Stage 5 acceptance (re-run) ==="
bash ../sw/run_stage5.sh > "$LOGDIR/regress_stage5.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage5.log"
grep -q "STAGE 5 RESULT: ALL PASS" "$LOGDIR/regress_stage5.log" || FAIL=1

echo "=== Stage 1-4 regressions (re-run) ==="
bash ../sw/run_regressions.sh > "$LOGDIR/regress_stage1_4.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage1_4.log"
grep -q "ALL REGRESSIONS PASSED" "$LOGDIR/regress_stage1_4.log" || FAIL=1

echo "=== STAGE 7 RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILED) ==="
exit $FAIL
