#!/bin/bash
# run_sta_frontend.sh -- per-stage STA for frontend (Stage 7).
# Method: Yosys synth + OpenSTA, Nangate45, 3.333 ns (300 MHz), same as
# run_sta_dense.sh (Stage 1) / run_sta_stage4.sh (Stage 4). The six stages
# are one lane's stages. Logs go to ../logs/stage7/.
# Usage: ./run_sta_frontend.sh
set -e
cd "$(dirname "$0")"
LIB=/home/hatch/workspace/fpga-timing-demo/Nangate45_typ.lib
STA=/home/hatch/workspace/opensta-2.2.0/app/sta
LOGDIR=../logs/stage7
mkdir -p "$LOGDIR"

for stage in fe_mult fe_add fe_add_wide fe_shift fe_sat fe_leaky; do
  tag="fe_${stage}"
  cat > "synth_${tag}.ys" <<YS
read_verilog -sv critpath_frontend.sv
hierarchy -top $stage
synth -top $stage
dfflibmap -liberty $LIB
abc -liberty $LIB
write_verilog -noattr ${stage}_${tag}_net.v
stat
YS
  yosys -q "synth_${tag}.ys" > "$LOGDIR/synth_${tag}.log" 2>&1
  cat > "sta_${tag}.tcl" <<TCL
read_liberty $LIB
read_verilog ${stage}_${tag}_net.v
link_design $stage
create_clock -period 3.333 -name clk [get_ports clk]
report_checks -path_delay max -fields {slew cap input_pins} -format full_clock_expanded -digits 3 > $LOGDIR/sta_${tag}_3p333ns.txt
report_tns -digits 3 >> $LOGDIR/sta_${tag}_3p333ns.txt
puts "=== $stage STA done ==="
TCL
  "$STA" -no_init -exit "sta_${tag}.tcl" > /dev/null 2>&1
  echo "=== frontend $stage ==="
  grep -E "data arrival time|slack \\(MET|slack \\(VIOLATED|tns " "$LOGDIR/sta_${tag}_3p333ns.txt" | head -n 4
done
echo "[STA frontend done]"
