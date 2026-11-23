#!/bin/bash
# tb/run_stage7d.sh -- Stage 7d verification runner (fail-fast).
#
# Part 1: integrated end-to-end for the two detection models
# (sw/vision/run_e2e_7d.py): YOLOv8n (128x128 bus.jpg) then DNANet
# (64x64 target-centered crop of target1.png). Real image in ->
# output out, vs the COMPOSED funcsim golden (max|err|=0 budget,
# act PWL <= 1 LSB) + accuracy vs the torch float reference on the
# same input (YOLO: same 5 detections, boxes within 2 px; DNANet:
# deep-supervision masks IoU 1.0). No frontend blur -- the 7d
# models take the raw image (see logs/stage7d/design.md).
# Part 2: post-synthesis simulation (tb/run_postsynth.sh, which
# now also covers the six 7d modules: act, simplegate, residual,
# pool, upsample, concat).
# Then the unit tests and the regressions (Stage 7c, which itself
# runs 7b + Stage 5 + Stages 1-4).
#
# Every model is invoked with --resume: the VM is replaced every
# few tens of minutes and a kill mid-run loses the in-memory
# chain, but the driver checkpoints after every passed node, so
# re-invoking this runner continues from the last passed node
# instead of restarting. The regression scripts have no resume;
# if a replacement kills them, re-run this runner (the e2e models
# will skip their checkpointed nodes and the regressions restart).
#
# Usage: bash tb/run_stage7d.sh   (from the repo root)
# Prints STAGE 7d RESULT: ALL PASS on success; exit 0 / 1.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

LOGDIR=logs/stage7d
mkdir -p "$LOGDIR"
PASS=0
FAIL=0
pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "=== Part 1: integrated end-to-end (YOLOv8n + DNANet) ==="
for MODEL in yolo dnanet; do
    UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
    python3 sw/vision/run_e2e_7d.py --model "$MODEL" --resume --outdir "$LOGDIR" \
        > "$LOGDIR/e2e_${MODEL}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && grep -q "=== ${UPPER}: ALL PASS" "$LOGDIR/e2e_${MODEL}.log"; then
        pass "e2e $MODEL (integrated, raw image -> PE chain)"
        grep -E "final|detections|IoU" "$LOGDIR/e2e_${MODEL}.log" | sed 's/^/    /'
    else
        fail "e2e $MODEL (rc=$rc)"
        tail -8 "$LOGDIR/e2e_${MODEL}.log"
    fi
done

echo "=== Part 2: post-synthesis simulation ==="
bash tb/run_postsynth.sh > "$LOGDIR/run_postsynth.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "POSTSYNTH RESULT: ALL PASS" "$LOGDIR/run_postsynth.log"; then
    pass "post-synth (7c modules + 7d modules)"
else
    fail "post-synth (rc=$rc)"
    tail -8 "$LOGDIR/run_postsynth.log"
fi

echo "=== Unit tests (sw/vision/tests) ==="
python3 -m unittest discover -s sw/vision/tests > "$LOGDIR/unittest.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "^OK" "$LOGDIR/unittest.log"; then
    pass "unit tests"
    tail -3 "$LOGDIR/unittest.log" | sed 's/^/    /'
else
    fail "unit tests (rc=$rc)"
    tail -8 "$LOGDIR/unittest.log"
fi

echo "=== Regression: Stage 7c (incl. 7b + Stage 5 + Stages 1-4) ==="
bash tb/run_stage7c.sh > "$LOGDIR/regress_stage7c.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && grep -q "STAGE 7c RESULT: ALL PASS" "$LOGDIR/regress_stage7c.log"; then
    pass "regression Stage 7c"
else
    fail "regression Stage 7c (rc=$rc)"
    tail -5 "$LOGDIR/regress_stage7c.log"
fi

echo
echo "=== STAGE 7d RESULT: $([ "$FAIL" -eq 0 ] && echo "ALL PASS" || echo "FAILURES (see above)") ==="
echo "PASS=$PASS FAIL=$FAIL"
exit "$FAIL"
