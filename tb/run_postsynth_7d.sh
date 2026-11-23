#!/bin/bash
# tb/run_postsynth_7d.sh -- Stage 7d post-synthesis simulation ONLY.
#
# The 7d e2e drives six RTL modules besides the PE array: act,
# simplegate, residual, pool, upsample and concat. All six TBs
# instantiate the DUT with DEFAULT parameters, so the
# generic-synth netlist (default params baked in) drops in place
# of the RTL. Yosys generic `synth -top <mod>` -> write_verilog
# netlist -> Icarus + Yosys simcells.v + the unchanged golden TB.
#
# This is the 7d section of tb/run_postsynth.sh extracted as a
# standalone runner, so Part 2 can run without re-running the
# slow 7c legs (adder_tree / frontend / pe_mac / pe_array).
# It must NOT run concurrently with the e2e: both share
# tb/vectors_*.txt and tb/out_*.txt.
#
# Usage: bash tb/run_postsynth_7d.sh   (from the repo root)
# Prints POSTSYNTH 7d RESULT: ALL PASS on success; exit 0 / 1.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

LOGDIR7D=logs/stage7d
mkdir -p "$LOGDIR7D"
PASS=0
FAIL=0
pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

run_ps_mod() {
    # $1 = name, $2 = rtl file, $3 = tb file, $4 = golden script,
    # $5 = vvp name. Netlist + logs live in logs/stage7d.
    local name="$1" rtl="$2" tb="$3" golden="$4" vvp="$5"
    local net="$LOGDIR7D/${name}_net.v"
    echo "=== 7d post-synth: $name ==="
    if [ -f "$net" ] && [ "$net" -nt "$rtl" ] \
            && grep -q "^module ${name}(" "$net"; then
        echo "[SKIP] $name synth (netlist exists, RTL unchanged)"
        pass "$name synth (generic) (reused netlist)"
        pass "$name netlist module name (reused netlist)"
    else
        timeout 1800 yosys -p "read_verilog -sv $rtl; synth -top $name; stat; write_verilog $net" \
            > "$LOGDIR7D/synth_${name}.log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then pass "$name synth (generic)"; else fail "$name synth (rc=$rc)"; fi
        if grep -q "^module ${name}(" "$net"; then
            pass "$name netlist module name"
        else
            fail "$name netlist module name"
        fi
        grep "Number of cells" "$LOGDIR7D/synth_${name}.log" | sed 's/^/    /'
    fi
    (cd tb && python3 "$golden" gen > "../$LOGDIR7D/vectors_${name}_ps.log" 2>&1)
    rc=$?
    if [ "$rc" -eq 0 ]; then pass "$name vectors gen"; else fail "$name vectors gen (rc=$rc)"; fi
    # The TB reads vectors_<name>.txt from its own (tb/) directory
    # and writes out_<name>.txt there; the golden check reads it
    # from tb/ too. Compile the netlist (not the RTL) + simcells
    # + the unchanged golden TB.
    iverilog -g2012 -o "/tmp/${vvp}" "$net" \
        /usr/share/yosys/simcells.v "tb/$tb" \
        > "$LOGDIR7D/compile_${name}_ps.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then pass "$name post-synth compile"; else fail "$name post-synth compile (rc=$rc)"; fi
    (cd tb && timeout 1800 vvp "/tmp/${vvp}" > "../$LOGDIR7D/vvp_${name}_ps.log" 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then fail "$name post-synth sim (rc=$rc)"; fi
    (cd tb && python3 "$golden" check > "../$LOGDIR7D/check_${name}_ps.log" 2>&1)
    rc=$?
    if grep -q "GOLDEN CHECK PASSED" "$LOGDIR7D/check_${name}_ps.log" && [ "$rc" -eq 0 ]; then
        pass "$name post-synth golden"
    else
        fail "$name post-synth golden (rc=$rc)"
        tail -5 "$LOGDIR7D/check_${name}_ps.log"
    fi
}
run_ps_mod act       rtl/vision/act.sv        tb_act.sv        golden_act.py        act_ps.vvp
run_ps_mod simplegate rtl/vision/simplegate.sv tb_simplegate.sv golden_simplegate.py sg_ps.vvp
run_ps_mod residual  rtl/vision/residual.sv   tb_residual.sv   golden_residual.py   res_ps.vvp
run_ps_mod pool      rtl/vision/pool.sv       tb_pool.sv       golden_pool.py       pool_ps.vvp
run_ps_mod upsample  rtl/vision/upsample.sv   tb_upsample.sv   golden_upsample.py   up_ps.vvp
run_ps_mod concat    rtl/vision/concat.sv     tb_concat.sv     golden_concat.py     concat_ps.vvp

echo
echo "=== POSTSYNTH 7d RESULT: $([ "$FAIL" -eq 0 ] && echo "ALL PASS" || echo "FAILURES (see above)") ==="
echo "PASS=$PASS FAIL=$FAIL"
exit "$FAIL"
