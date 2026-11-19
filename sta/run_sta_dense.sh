#!/bin/bash
# run_sta_dense.sh -- per-stage STA for conv2d_dense (Stage 1, PX=1).
# Method: Yosys synth + OpenSTA, Nangate45, 3.333 ns (300 MHz), same as the
# reference run_sta.sh. The five stages are one lane's stages.
# Logs go to ../logs/stage1/.
# Usage: ./run_sta_dense.sh
set -e
cd "$(dirname "$0")"
LIB=/home/hatch/workspace/fpga-timing-demo/Nangate45_typ.lib
LOGDIR=../logs/stage1
mkdir -p "$LOGDIR"

for stage in stage_mult stage_add stage_add_wide stage_shift_only stage_sat_only; do
  tag="dense_px1_${stage}"
  cat > "synth_${tag}.ys" <<YS
read_verilog -sv critpath_dense.sv
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
  sta -no_init -exit "sta_${tag}.tcl" > /dev/null 2>&1
  echo "=== dense PX=1 $stage ==="
  grep -E "data arrival time|slack \\(MET|slack \\(VIOLATED|tns " "$LOGDIR/sta_${tag}_3p333ns.txt" | head -n 4
done
echo "[STA dense PX=1 done]"
