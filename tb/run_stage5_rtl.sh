#!/bin/bash
# tb/run_stage5_rtl.sh -- Stage 5 RTL verification: the AXI-MM weight DMA.
#   1. Generate the DMA vectors (golden_dma.py).
#   2. Compile the DMA + testbench with Icarus Verilog.
#   3. Run each case twice: clean (no backpressure, cycle count vs
#      the Python model) and with random wfull backpressure (data
#      correctness under stalls).
# Logs go to ../logs/stage5/.
set -u
cd "$(dirname "$0")"
LOGDIR=../logs/stage5
mkdir -p "$LOGDIR"

# Tool resolution (see tb/run_stage1.sh).
IVERILOG="$(command -v iverilog || true)"
[ -n "$IVERILOG" ] || IVERILOG=/usr/bin/iverilog
if [ ! -x "$IVERILOG" ]; then
    echo "ERROR: iverilog not found (checked PATH and /usr/bin/iverilog)." >&2
    exit 1
fi

python3 golden_dma.py | tee "$LOGDIR/vectors_dma.log"

if ! "$IVERILOG" -g2012 -o tb_dma.vvp ../rtl/weight_dma.sv tb_weight_dma.sv \
    > "$LOGDIR/compile_dma.log" 2>&1; then
    echo "COMPILE FAILED (weight_dma)"; cat "$LOGDIR/compile_dma.log"
    exit 1
fi
echo "[compile] rc=0"

FAIL=0
for case in A B C D; do
    echo "=== DMA case $case (clean, cycle measurement) ==="
    vvp tb_dma.vvp +vectors=vectors_dma_${case}.txt +bp=0 \
        2>&1 | tee "$LOGDIR/dma_${case}_clean.log"
    grep -q "DMA CHECK PASSED" "$LOGDIR/dma_${case}_clean.log" || FAIL=1
    echo "=== DMA case $case (wfull backpressure) ==="
    vvp tb_dma.vvp +vectors=vectors_dma_${case}.txt +bp=1 \
        2>&1 | tee "$LOGDIR/dma_${case}_bp.log"
    grep -q "DMA CHECK PASSED" "$LOGDIR/dma_${case}_bp.log" || FAIL=1
done

echo "=== STAGE 5 RTL RESULT: $([ $FAIL -eq 0 ] && echo ALL PASS || echo FAILED) ==="
exit $FAIL
