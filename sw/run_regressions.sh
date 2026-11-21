#!/bin/bash
# sw/run_regressions.sh -- re-run Stage 1-4 regressions.
#
# The wrapper FAILS (non-zero exit) if ANY stage fails. A stage passes
# only when BOTH hold:
#   (a) its run script exits 0, and
#   (b) its pass marker is present in the check log AND the .vvp was built.
# It prints "ALL REGRESSIONS PASSED" only when every stage passes;
# a wrapper that only echoes "DONE" is a false pass.
#
# Usage: bash sw/run_regressions.sh   (or nohup sw/run_regressions.sh &)
# Log: logs/stage5/regressions.log
set -u
# Portable: run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

# Explicit PATH: a nohup/background shell may not inherit the login PATH.
# Keep Homebrew (macOS) first when present.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

# Fail fast if a tool is missing (a VM reset can remove /usr/bin tools).
for tool in iverilog yosys python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool not found on PATH=$PATH" >&2
        echo "Reinstall iverilog/yosys from the cached .debs in ~/tools-dl/" >&2
        exit 1
    fi
done

LOG=logs/stage5/regressions.log
mkdir -p logs/stage5
# Tee all output to the log (and keep stdout live).
exec > >(tee "$LOG") 2>&1

OVERALL=0
start_all=$(date +%s)

# run_stage <n> <script> <check-log> <marker> <vvp>
# The stage scripts are invoked via `bash` (robust to a lost +x bit).
run_stage() {
    local n="$1" script="$2" checklog="$3" marker="$4" vvp="$5"
    local t0 t1 rc
    echo "=== Stage $n ($script) ==="
    t0=$(date +%s)
    bash "$script"
    rc=$?
    t1=$(date +%s)
    echo "[stage $n] script exit=$rc wall=$((t1 - t0))s"
    if [ "$rc" -ne 0 ]; then
        echo "[stage $n] FAILED (script exit $rc)" >&2
        OVERALL=1
        return
    fi
    if [ -n "$vvp" ] && [ ! -s "$vvp" ]; then
        echo "[stage $n] FAILED (.vvp $vvp missing/empty -- did it compile?)" >&2
        OVERALL=1
        return
    fi
    if [ -n "$checklog" ] && ! grep -q "$marker" "$checklog"; then
        echo "[stage $n] FAILED (marker '$marker' not in $checklog)" >&2
        OVERALL=1
        return
    fi
    echo "[stage $n] PASS"
}

# Stage 1/2: single-module runs; the marker lives in check.log and the
# .vvp is the compile artifact.
run_stage 1 tb/run_stage1.sh logs/stage1/check.log "GOLDEN CHECK PASSED" tb/tb_dense.vvp
run_stage 2 tb/run_stage2.sh logs/stage2/check.log "GOLDEN CHECK PASSED" tb/tb_dw.vvp

# Stage 3/4: multi-module runs. Their run scripts exit $FAIL (0 only on
# ALL PASS) and each module's check log must say GOLDEN CHECK PASSED.
# Verify every module check log independently here.
run_stage 3 tb/run_stage3.sh "" "" ""
if [ "$OVERALL" -eq 0 ]; then
    for f in logs/stage3/check_*.log; do
        grep -q "GOLDEN CHECK PASSED" "$f" || {
            echo "[stage 3] FAILED (marker missing in $f)" >&2; OVERALL=1; }
    done
    grep -q "GOLDEN CHECK PASSED" logs/stage3/regress_stage1.log || {
        echo "[stage 3] FAILED (stage-1 regression marker missing)" >&2; OVERALL=1; }
    grep -q "GOLDEN CHECK PASSED" logs/stage3/regress_stage2.log || {
        echo "[stage 3] FAILED (stage-2 regression marker missing)" >&2; OVERALL=1; }
    [ "$OVERALL" -eq 0 ] && echo "[stage 3] PASS (all module check logs)"
fi

run_stage 4 tb/run_stage4.sh "" "" ""
if [ "$OVERALL" -eq 0 ]; then
    for f in logs/stage4/check_*.log; do
        grep -q "GOLDEN CHECK PASSED" "$f" || {
            echo "[stage 4] FAILED (marker missing in $f)" >&2; OVERALL=1; }
    done
    grep -q "GOLDEN CHECK PASSED" logs/stage4/regress_stage1.log || {
        echo "[stage 4] FAILED (stage-1 regression marker missing)" >&2; OVERALL=1; }
    [ "$OVERALL" -eq 0 ] && echo "[stage 4] PASS (all module check logs)"
fi

end_all=$(date +%s)
echo "=== regressions total wall=$((end_all - start_all))s ==="
if [ "$OVERALL" -eq 0 ]; then
    echo "=== ALL REGRESSIONS PASSED ==="
    exit 0
else
    echo "=== REGRESSIONS FAILED (see above) ===" >&2
    exit 1
fi
