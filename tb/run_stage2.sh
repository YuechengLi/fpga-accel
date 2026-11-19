#!/bin/bash
# run_stage2.sh -- Stage 2 golden simulation for conv2d_dw (PX=1).
# Generates vectors, compiles the SV RTL + testbench with Icarus Verilog,
# runs the simulation (random in_valid gaps + random out_ready stalls),
# and checks the output pixel-exact against the Python golden model.
# Logs go to ../logs/stage2/.
set -e
cd "$(dirname "$0")"
LOGDIR=../logs/stage2
mkdir -p "$LOGDIR"

python3 golden_dw.py gen | tee "$LOGDIR/gen.log"
iverilog -g2012 -o tb_dw.vvp ../rtl/conv2d_dw.sv tb_conv2d_dw.sv \
    2>&1 | tee "$LOGDIR/compile.log"
echo "[compile] rc=$?"
t0=$(date +%s)
timeout 900 vvp tb_dw.vvp > "$LOGDIR/vvp.log" 2>&1
rc=$?
t1=$(date +%s)
echo "[vvp] rc=$rc wall=$((t1-t0))s"
cat "$LOGDIR/vvp.log"
python3 golden_dw.py check | tee "$LOGDIR/check.log"
