# fpga-accel — Makefile (placeholder version)
#
# Target usage (one-click reproduction):
#   make sim        Simulation + golden comparison (this VM: Icarus / Yosys / OpenSTA)
#   make synth      Synthesis (Vivado machine, scripts/synth.tcl)
#   make impl       Implementation: place and route + timing report (scripts/impl.tcl)
#   make bitstream  Generate bitstream (scripts/bitstream.tcl)
#   make program    Program + ILA capture (scripts/program.tcl, scripts/debug.tcl)
#   make all        Synthesis → implementation → bitstream
#   make clean      Clean generated products (does not delete source or logs)
#
# Note: the concrete commands for each target will be filled in as
# stages 1–7 progress; this is a placeholder that only documents
# the interface and runs no synthesis/programming.

.PHONY: sim synth impl bitstream program all clean

sim:
	@echo "TODO: simulation + golden comparison (Icarus/Yosys/OpenSTA), to be filled in"

synth:
	@echo "TODO: vivado -mode batch -source scripts/synth.tcl, to be filled in"

impl:
	@echo "TODO: vivado -mode batch -source scripts/impl.tcl, to be filled in"

bitstream:
	@echo "TODO: vivado -mode batch -source scripts/bitstream.tcl, to be filled in"

program:
	@echo "TODO: vivado -mode batch -source scripts/program.tcl, to be filled in"

all: synth impl bitstream

clean:
	@echo "TODO: clean generated products (bitstream/synthesis products), to be filled in"
