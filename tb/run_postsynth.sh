#!/bin/bash
# tb/run_postsynth.sh -- post-synthesis simulation (Stage 7c, Part 2).
#
# STA checks timing only, never functional correctness of the
# netlist. This runner takes the Yosys-synthesized netlist and
# re-runs the SAME golden checks against it:
#   Yosys generic `synth -top <mod>` -> write_verilog netlist ->
#   Icarus + Yosys simcells.v + the unchanged golden TB.
#
# Generic synth, not synth_ecp5: an ECP5 netlist has TRELLIS/LUT4
# cells simcells.v does not model; a generic netlist has $_AND_ /
# $_DFF_P_ / $mem_v2, which it does. SDF back-annotated timing sim
# is out of scope (no SDF flow; STA covers timing).
#
# Parameter problem (found 2026-11-21): both golden TBs instantiate
# their DUT with parameter overrides (adder_tree #(.N(160)/81/64/25),
# frontend #(.MAX_W(3840))), but a synthesized netlist has no
# parameters -- Icarus rejects the override ("parameter `N` not
# found", verified with a probe). The netlist therefore cannot be
# dropped in place of the RTL. Fix: a parameter-accepting wrapper
# (pure port passthrough, no logic) named like the RTL module,
# around the renamed netlist(s):
#   - adder_tree: one netlist per N (chparam + synth), selected by
#     generate-if in tb/adder_tree_ps_wrap.sv.
#   - frontend: one netlist (default MAX_W=3840, the TB's value),
#     tb/frontend_ps_wrap.sv (fails fast if MAX_W != 3840).
# The TBs themselves are compiled unchanged.
#
# VM scope: adder_tree + frontend (both synth whole on this VM).
# pe_mac alone OOMs on this VM (~6 GB sandbox limit) and pe_array
# needs the Mac (24 GB) -- those run as Phase 3 in
# ~/workspace/your_files/run_ecp5_mac.sh and are reported here as
# explicit SKIP lines, not passes/failures (Stage 7b precedent).
#
# Usage: bash tb/run_postsynth.sh   (from the repo root)
# Prints POSTSYNTH RESULT: ALL PASS on success; exit 0 / 1.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

LOGDIR=logs/stage7c
mkdir -p "$LOGDIR"
PASS=0
FAIL=0
pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "=== 1. adder_tree post-synth (4 netlists, N=160/81/64/25) ==="
for N in 160 81 64 25; do
    NET="$LOGDIR/adder_tree_n${N}.v"
    # Reuse a netlist from an earlier (killed) run when it exists
    # and the RTL is unchanged -- the VM is replaced every few
    # tens of minutes, and re-synthesizing is pure re-work.
    if [ -f "$NET" ] && [ "$NET" -nt rtl/common/adder_tree.sv ] \
            && grep -q "^module adder_tree_n${N}(" "$NET"; then
        echo "[SKIP] adder_tree synth N=$N (netlist exists, RTL unchanged)"
        pass "adder_tree synth N=$N (reused netlist)"
        pass "adder_tree netlist rename N=$N (reused netlist)"
        continue
    fi
    timeout 600 yosys -p "read_verilog -sv rtl/common/adder_tree.sv; chparam -set N $N adder_tree; synth -top adder_tree; stat; write_verilog $LOGDIR/adder_tree_n${N}.v" \
        > "$LOGDIR/synth_adder_tree_n${N}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then pass "adder_tree synth N=$N"; else fail "adder_tree synth N=$N (rc=$rc)"; fi
    sed -i "s/^module adder_tree(/module adder_tree_n${N}(/" "$LOGDIR/adder_tree_n${N}.v"
    if grep -q "^module adder_tree_n${N}(" "$LOGDIR/adder_tree_n${N}.v"; then
        pass "adder_tree netlist rename N=$N"
    else
        fail "adder_tree netlist rename N=$N"
    fi
    grep "Number of cells" "$LOGDIR/synth_adder_tree_n${N}.log" | sed "s/^/    N=$N: /"
done
iverilog -g2012 -o /tmp/tree_ps.vvp tb/adder_tree_ps_wrap.sv \
    "$LOGDIR/adder_tree_n160.v" "$LOGDIR/adder_tree_n81.v" \
    "$LOGDIR/adder_tree_n64.v" "$LOGDIR/adder_tree_n25.v" \
    /usr/share/yosys/simcells.v tb/tb_adder_tree.sv \
    > "$LOGDIR/compile_tree_ps.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then pass "adder_tree post-synth compile"; else fail "adder_tree post-synth compile (rc=$rc)"; fi
timeout 600 vvp /tmp/tree_ps.vvp > "$LOGDIR/vvp_tree_ps.log" 2>&1
rc=$?
if grep -q "ADDER TREE SMOKE TEST PASSED" "$LOGDIR/vvp_tree_ps.log" && [ "$rc" -eq 0 ]; then
    pass "adder_tree post-synth smoke (200 vectors x 4 sizes)"
else
    fail "adder_tree post-synth smoke (rc=$rc)"
    tail -5 "$LOGDIR/vvp_tree_ps.log"
fi

echo "=== 2. frontend post-synth (generic netlist, MAX_W=3840) ==="
if [ -f "$LOGDIR/frontend_net.v" ] \
        && [ "$LOGDIR/frontend_net.v" -nt rtl/vision/frontend.sv ] \
        && grep -q "^module frontend_net(" "$LOGDIR/frontend_net.v"; then
    echo "[SKIP] frontend synth (netlist exists, RTL unchanged)"
    pass "frontend synth (generic) (reused netlist)"
    pass "frontend netlist rename (reused netlist)"
else
    timeout 1800 yosys -p "read_verilog -sv rtl/vision/frontend.sv; synth -top frontend; stat; write_verilog $LOGDIR/frontend_net.v" \
        > "$LOGDIR/synth_frontend.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then pass "frontend synth (generic)"; else fail "frontend synth (rc=$rc)"; fi
    sed -i 's/^module frontend(/module frontend_net(/' "$LOGDIR/frontend_net.v"
    if grep -q "^module frontend_net(" "$LOGDIR/frontend_net.v"; then
        pass "frontend netlist rename"
    else
        fail "frontend netlist rename"
    fi
    grep "Number of cells" "$LOGDIR/synth_frontend.log" | sed 's/^/    /'
fi
(cd tb && python3 golden_frontend.py gen > "../$LOGDIR/gen_frontend_ps.log" 2>&1)
rc=$?
if [ "$rc" -eq 0 ]; then pass "frontend vectors gen (16 cases)"; else fail "frontend vectors gen (rc=$rc)"; fi
# Frontend SIM is Mac Phase 3, not run here: the generic netlist is
# 193,923 cells because the line buffer synthesizes to 92,952 FFs +
# 92,904 muxes (the generic `memory` pass cannot map this multi-port
# line buffer to $mem_v2; the vendor synths do map it to BRAM -- 12x
# BRAM36 in synth_xilinx), and Icarus elaboration of that netlist
# does not complete on this VM (40+ min, 2.3 GB RSS, memory pressure
# -- killed 2026-11-22). The netlist itself is valid (synth PASS
# above); the sim runs on the Mac (24 GB) via run_ecp5_mac.sh
# Phase 3. Reported as SKIP here, not a pass or a failure.
echo "[SKIP] frontend post-synth sim -- Mac Phase 3 (Icarus elaboration does not complete on this VM; see design.md)"
echo "[SKIP] frontend post-synth golden -- Mac Phase 3"

echo "=== 3. pe_mac / pe_array post-synth (Mac Phase 3) ==="
echo "[SKIP] pe_mac post-synth -- Mac Phase 3 (netlist OOMs on this VM)"
echo "[SKIP] pe_array post-synth -- Mac Phase 3 (netlist OOMs on this VM)"

echo
echo "=== POSTSYNTH RESULT: $([ "$FAIL" -eq 0 ] && echo "ALL PASS" || echo "FAILURES (see above)") ==="
echo "PASS=$PASS FAIL=$FAIL"
exit "$FAIL"
