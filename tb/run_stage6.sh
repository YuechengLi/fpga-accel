#!/bin/bash
# run_stage6.sh -- Stage 6 model numerical verification (plan Stage 6).
#   Runs SRCNN (full end-to-end), the DnCNN fragment, and RetinexNet
#   Decom-Net through the RTL (conv2d_mc + act + residual) and compares
#   the RTL output against the Stage 5 functional simulator, pixel by
#   pixel (RTL vs funcsim must be EXACT), plus per-layer error vs float.
#   Also re-validates the activation ranges r_out on wider data and
#   checks the 32-bit accumulator bound (Python only).
#   Then re-runs the Stage 5 acceptance and the Stage 1-4 regressions.
# Run from the tb/ directory:
#   bash run_stage6.sh [--model srcnn|dncnn|retinexnet|all]
# The model sims are slow (the MC sim is the long pole, ~15-30 min per
# model); the whole script takes ~1-2 h. Logs go to ../logs/stage6/.
set -u
set -o pipefail
cd "$(dirname "$0")"

IVERILOG="$(command -v iverilog || true)"
[ -n "$IVERILOG" ] || IVERILOG=/usr/bin/iverilog
if [ ! -x "$IVERILOG" ]; then
    echo "ERROR: iverilog not found (checked PATH and /usr/bin/iverilog)." >&2
    exit 1
fi

LOGDIR=../logs/stage6
mkdir -p "$LOGDIR"
FAIL=0

MODEL_ARG="${1:-all}"

echo "=== Stage 6: model numerical verification (model=$MODEL_ARG) ==="
python3 ../sw/vision/stage6.py --model "$MODEL_ARG" --outdir "$LOGDIR" \
    2>&1 | tee "$LOGDIR/stage6_run.log"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
    echo "STAGE 6 MODELS FAILED (rc=$rc)"
    FAIL=1
fi

echo "=== Stage 5 acceptance (re-run) ==="
bash ../sw/run_stage5.sh > "$LOGDIR/regress_stage5.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage5.log"
grep -q "STAGE 5 RESULT: ALL PASS" "$LOGDIR/regress_stage5.log" || FAIL=1

echo "=== Stage 1-4 regressions (re-run) ==="
bash ../sw/run_regressions.sh > "$LOGDIR/regress_stage1_4.log" 2>&1
tail -n 5 "$LOGDIR/regress_stage1_4.log"
grep -q "ALL REGRESSIONS PASSED" "$LOGDIR/regress_stage1_4.log" || FAIL=1

echo "=== STAGE 6 RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILED) ==="
exit $FAIL
