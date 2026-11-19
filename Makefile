# fpga-accel — Makefile（占位版）
#
# 目标用法（一键复现）：
#   make sim        仿真 + golden 比对（本 VM：Icarus / Yosys / OpenSTA）
#   make synth      综合（Vivado 机器，scripts/synth.tcl）
#   make impl       实现：布局布线 + 时序报告（scripts/impl.tcl）
#   make bitstream  生成 bitstream（scripts/bitstream.tcl）
#   make program    烧录 + ILA 抓取（scripts/program.tcl、scripts/debug.tcl）
#   make all        综合 → 实现 → bitstream
#   make clean      清理生成物（不删源码与日志）
#
# 注意：各 target 的具体命令待阶段 1–7 推进时填入；当前为占位，
# 仅说明接口，不执行任何综合/烧录。

.PHONY: sim synth impl bitstream program all clean

sim:
	@echo "TODO: 仿真 + golden 比对（Icarus/Yosys/OpenSTA），待填入"

synth:
	@echo "TODO: vivado -mode batch -source scripts/synth.tcl，待填入"

impl:
	@echo "TODO: vivado -mode batch -source scripts/impl.tcl，待填入"

bitstream:
	@echo "TODO: vivado -mode batch -source scripts/bitstream.tcl，待填入"

program:
	@echo "TODO: vivado -mode batch -source scripts/program.tcl，待填入"

all: synth impl bitstream

clean:
	@echo "TODO: 清理生成物（bitstream/综合产物），待填入"
