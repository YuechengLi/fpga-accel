#!/bin/bash
# run_sta_pe.sh -- Stage 7b PE STA: synth pe_mac (128 multipliers +
# accumulator file + dot-mode tree) and STA it at 300 / 360 / 400 MHz
# (Nangate45, OpenSTA 2.2.0), same per-stage pattern as run_sta_clock.sh.
# The PE is multiplier-heavy, so the Yosys synth may take minutes;
# run in the background. Logs: ../logs/stage7b/sta_pe_<period>ns.txt.
# Usage: ./run_sta_pe.sh
set -o pipefail
cd "$(dirname "$0")"
LIB=/home/hatch/workspace/fpga-timing-demo/Nangate45_typ.lib
STA_BIN=/home/hatch/workspace/opensta-2.2.0/app/sta
LOGDIR=../logs/stage7b
mkdir -p "$LOGDIR"
FAIL=0

for spec in "3.333 300" "2.778 360" "2.500 400"; do
  set -- $spec
  period="$1"; mhz="$2"; tag="pe_${period}"
  cat > "synth_${tag}.ys" <<YS
read_verilog -sv ../rtl/pe_mac.sv
hierarchy -top pe_mac
synth -top pe_mac
dfflibmap -liberty $LIB
abc -liberty $LIB
write_verilog -noattr ${tag}_net.v
stat
YS
  echo "=== pe_mac @ ${mhz} MHz (${period} ns): synth ==="
  if ! yosys -q "synth_${tag}.ys" > "$LOGDIR/synth_${tag}.log" 2>&1; then
    echo "FAIL yosys $tag"; FAIL=1; continue
  fi
  cat > "sta_${tag}.tcl" <<TCL
read_liberty $LIB
read_verilog ${tag}_net.v
link_design pe_mac
create_clock -period $period -name clk [get_ports clk]
report_checks -path_delay max -fields {slew cap input_pins} -format full_clock_expanded -digits 3 > $LOGDIR/sta_${tag}.txt
report_tns -digits 3 >> $LOGDIR/sta_${tag}.txt
puts "=== pe_mac STA done ==="
TCL
  if ! "$STA_BIN" -no_init -exit "sta_${tag}.tcl" > /dev/null 2>&1; then
    echo "FAIL sta $tag"; FAIL=1; continue
  fi
  line=$(grep -E "slack \\(MET|slack \\(VIOLATED" "$LOGDIR/sta_${tag}.txt" | head -n 1)
  echo "  pe_mac @ ${period}ns: $line"
  if echo "$line" | grep -q VIOLATED; then FAIL=1; fi
done

if [ "$FAIL" = 0 ]; then echo "PE STA: ALL MET"; else echo "PE STA: FAILURES (see logs)"; fi
exit "$FAIL"
