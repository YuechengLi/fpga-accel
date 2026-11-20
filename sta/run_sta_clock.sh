#!/bin/bash
# run_sta_clock.sh -- Stage 7b clock decision: STA the existing critical
# stages (the shapes the new SRCNN pipeline reuses: mult, adder tree,
# shift, saturate, MC accumulate) at 300 / 360 / 400 MHz and report
# which clocks close. Method: Yosys synth + OpenSTA, Nangate45, same
# per-stage pattern as run_sta_dense.sh / run_sta_mc.sh.
# Usage: ./run_sta_clock.sh
# Logs: ../logs/stage7b/sta_clock_<stage>_<period>ns.txt
set -o pipefail
cd "$(dirname "$0")"
LIB=/home/hatch/workspace/fpga-timing-demo/Nangate45_typ.lib
STA_BIN=/home/hatch/workspace/opensta-2.2.0/app/sta
LOGDIR=../logs/stage7b
mkdir -p "$LOGDIR"
FAIL=0

run_one() {
  local src="$1" stage="$2" period="$3" tag="$4"
  cat > "synth_clock_${tag}.ys" <<YS
read_verilog -sv $src
hierarchy -top $stage
synth -top $stage
dfflibmap -liberty $LIB
abc -liberty $LIB
write_verilog -noattr clock_${tag}_net.v
stat
YS
  if ! yosys -q "synth_clock_${tag}.ys" > "$LOGDIR/synth_clock_${tag}.log" 2>&1; then
    echo "FAIL yosys $tag"; FAIL=1; return
  fi
  cat > "sta_clock_${tag}.tcl" <<TCL
read_liberty $LIB
read_verilog clock_${tag}_net.v
link_design $stage
create_clock -period $period -name clk [get_ports clk]
report_checks -path_delay max -fields {slew cap input_pins} -format full_clock_expanded -digits 3 > $LOGDIR/sta_clock_${tag}.txt
report_tns -digits 3 >> $LOGDIR/sta_clock_${tag}.txt
puts "=== $stage STA done ==="
TCL
  if ! "$STA_BIN" -no_init -exit "sta_clock_${tag}.tcl" > /dev/null 2>&1; then
    echo "FAIL sta $tag"; FAIL=1; return
  fi
  local line
  line=$(grep -E "slack \(MET|slack \(VIOLATED" "$LOGDIR/sta_clock_${tag}.txt" | head -n 1)
  echo "  $stage @ ${period}ns: $line"
  if echo "$line" | grep -q VIOLATED; then FAIL=1; fi
}

echo "=== Clock decision STA (Nangate45) ==="
for spec in "3.333 300" "2.778 360" "2.500 400"; do
  set -- $spec; period="$1"; mhz="$2"
  echo "--- period ${period} ns (${mhz} MHz) ---"
  run_one critpath_dense.sv stage_mult "$period" "mult_${period}"
  run_one critpath_dense.sv stage_add "$period" "add_${period}"
  run_one critpath_dense.sv stage_add_wide "$period" "addwide_${period}"
  run_one critpath_dense.sv stage_shift_only "$period" "shift_${period}"
  run_one critpath_dense.sv stage_sat_only "$period" "sat_${period}"
  run_one critpath_mc.sv stage_mc_accum "$period" "mcaccum_${period}"
  run_one critpath_mc.sv stage_act_leaky "$period" "leaky_${period}"
done
if [ "$FAIL" -eq 0 ]; then
  echo "[STA clock decision: ALL PERIODS MET]"
else
  echo "[STA clock decision: SOME VIOLATIONS - see above]"
fi
exit $FAIL
