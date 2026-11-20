#!/bin/bash
# sw/run_stage5.sh -- Stage 5 master runner.
# Runs, in order, and fails (non-zero exit) if ANY step fails:
#   1. Per-pass unit tests (sw/tests/, stdlib unittest)
#   2. SRCNN demo (compile -> functional simulator -> rhu golden)
#   3. RetinexNet demo (compile -> functional simulator -> float)
#   4. C driver host test (sw/driver/, gcc)
#   5. RTL weight-DMA verification (tb/run_stage5_rtl.sh)
#   6. Cost-model / DRAM / roofline report (sw/stage5_report.py)
# Usage: bash sw/run_stage5.sh   (from the repo root, or anywhere)
# Logs go to logs/stage5/.
set -u
cd "$(dirname "$0")/.."

LOGDIR=logs/stage5
mkdir -p "$LOGDIR"
exec > >(tee "$LOGDIR/run_stage5.log") 2>&1

FAIL=0
step() {
    # $1 = label, $2.. = command
    local label="$1"; shift
    echo "=== Stage 5: $label ==="
    if "$@"; then
        echo "[stage5] $label: PASS"
    else
        echo "[stage5] $label: FAILED" >&2
        FAIL=1
    fi
}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

step "unit tests (43)" \
    python3 -m unittest discover -s sw/tests

step "SRCNN demo" \
    python3 sw/demo_srcnn.py --outdir out/srcnn

step "RetinexNet demo" \
    python3 sw/demo_retinexnet.py --outdir out/retinexnet

step "C driver host test" \
    make -C sw/driver test

step "RTL weight DMA" \
    bash tb/run_stage5_rtl.sh

step "cost / DRAM / roofline report" \
    python3 sw/stage5_report.py

echo "=== STAGE 5 RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILED) ==="
exit $FAIL
