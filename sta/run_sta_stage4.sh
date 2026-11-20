#!/bin/bash
# run_sta_stage4.sh -- per-stage STA for the Stage 4 modules.
# Method: Yosys synth + OpenSTA, Nangate45, 3.333 ns (300 MHz), same as
# run_sta_mc.sh (Stage 3). Logs go to ../logs/stage4/.
# Usage: ./run_sta_stage4.sh
set -e
cd "$(dirname "$0")"
LIB=/home/hatch/workspace/fpga-timing-demo/Nangate45_typ.lib
STA=/home/hatch/workspace/opensta-2.2.0/app/sta
LOGDIR=../logs/stage4
mkdir -p "$LOGDIR"

for stage in stage_ln_norm_a stage_ln_norm_b stage_ln_sumsq stage_act_mlut stage_dense_leaky stage_sg_mult stage_up_bilin; do
  tag="s4_${stage}"
  cat > "synth_${tag}.ys" <<YS
read_verilog -sv critpath_stage4.sv
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
  echo "=== Stage 4 $stage ==="
  grep -E "data arrival time|slack \(MET|slack \(VIOLATED|tns " "$LOGDIR/sta_${tag}_3p333ns.txt" | head -n 4
done
echo "[STA Stage 4 done]"
