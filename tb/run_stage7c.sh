#!/bin/bash
# tb/run_stage7c.sh -- Stage 7c verification runner (fail-fast).
#
# Part 1: integrated end-to-end (sw/run_e2e.py, one model at a time:
# SRCNN, RetinexNet, DnCNN). Part 2: post-synthesis simulation
# (tb/run_postsynth.sh). Then regressions (tb/run_stage7b.sh +
# sw/run_stage5.sh + sw/run_regressions.sh).
#
# Every model is invoked with --resume: the VM is replaced every
# few tens of minutes and a kill mid-run loses the in-memory
# chain, but the driver checkpoints after every passed layer, so
# re-invoking this runner continues from the last passed layer
# instead of restarting (see logs/stage7c/design.md). The
# regression scripts have no resume; if a replacement kills them,
# re-run this runner (the e2e models will skip their checkpointed
# layers and the regressions restart).
#
# Usage: bash tb/run_stage7c.sh   (from the repo root)
# Prints STAGE 7c RESULT: ALL PASS on success; exit 0 / 1.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

LOGDIR=logs/stage7c
mkdir -p "$LOGDIR"
PASS=0
FAIL=0
pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "=== Part 1: integrated end-to-end ==="
for MODEL in srcnn retinexnet dncnn; do
    UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
    python3 sw/run_e2e.py --model "$MODEL" --resume --outdir "$LOGDIR" \
        > "$LOGDIR/e2e_${MODEL}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && grep -q "E2E ${UPPER}: ALL PASS" "$LOGDIR/e2e_${MODEL}.log"; then
        pass "e2e $MODEL (integrated, frontend -> PE chain)"
        grep -E "frontend:|FINAL|denoised:|sigmoid:" "$LOGDIR/e2e_${MODEL}.log" | sed 's/^/    /'
    else
        fail "e2e $MODEL (rc=$rc)"
        tail -8 "$LOGDIR/e2e_${MODEL}.log"
    fi
done

echo "=== Part 2: post-synthesis simulation ==="
bash tb/run_postsynth.sh > "$LOGDIR/run_postsynth.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "POSTSYNTH RESULT: ALL PASS" "$LOGDIR/run_postsynth.log"; then
    pass "post-synth (adder_tree + frontend netlists)"
else
    fail "post-synth (rc=$rc)"
    tail -8 "$LOGDIR/run_postsynth.log"
fi

echo "=== Regressions ==="
bash tb/run_stage7b.sh > "$LOGDIR/regress_stage7b.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "STAGE 7b RESULT: ALL PASS" "$LOGDIR/regress_stage7b.log"; then
    pass "regression Stage 7b"
else
    fail "regression Stage 7b (rc=$rc)"
    tail -5 "$LOGDIR/regress_stage7b.log"
fi
bash sw/run_stage5.sh > "$LOGDIR/regress_stage5.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "STAGE 5 RESULT: ALL PASS" "$LOGDIR/regress_stage5.log"; then
    pass "regression Stage 5"
else
    fail "regression Stage 5 (rc=$rc)"
    tail -5 "$LOGDIR/regress_stage5.log"
fi
bash sw/run_regressions.sh > "$LOGDIR/regress_stage1_4.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "ALL REGRESSIONS PASSED" "$LOGDIR/regress_stage1_4.log"; then
    pass "regression Stages 1-4"
else
    fail "regression Stages 1-4 (rc=$rc)"
    tail -5 "$LOGDIR/regress_stage1_4.log"
fi

echo
echo "=== STAGE 7c RESULT: $([ "$FAIL" -eq 0 ] && echo "ALL PASS" || echo "FAILURES (see above)") ==="
echo "PASS=$PASS FAIL=$FAIL"
exit "$FAIL"
