#!/bin/bash
# run_stage1.sh -- Stage 1 golden simulation for conv2d_dense (PX=1).
# Generates vectors, compiles the SV RTL + testbench with Icarus Verilog,
# runs the simulation (random in_valid gaps + random out_ready stalls),
# and checks the output pixel-exact against the Python golden model.
# Logs go to ../logs/stage1/.
set -e
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
LOGDIR=../logs/stage1
mkdir -p "$LOGDIR"

python3 golden_dense.py gen | tee "$LOGDIR/gen.log"
# NOTE (fixed 2026-11-20): the old form
#   iverilog ... 2>&1 | tee compile.log; echo "[compile] rc=$?"
# reported tee's exit code (0), not iverilog's -- a failed compile
# looked like rc=0. With pipefail the pipeline's status is iverilog's,
# and the if/! form reports it explicitly.
if ! "$IVERILOG" -g2012 -o tb_dense.vvp ../rtl/vision/conv2d_dense.sv tb_conv2d_dense.sv \
    2>&1 | tee "$LOGDIR/compile.log"; then
    echo "[compile] FAILED (iverilog)" >&2
    exit 1
fi
echo "[compile] rc=0"
t0=$(date +%s)
timeout 900 vvp tb_dense.vvp > "$LOGDIR/vvp.log" 2>&1
rc=$?
t1=$(date +%s)
echo "[vvp] rc=$rc wall=$((t1-t0))s"
cat "$LOGDIR/vvp.log"
python3 golden_dense.py check | tee "$LOGDIR/check.log"
