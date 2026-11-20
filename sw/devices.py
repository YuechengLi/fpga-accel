#!/usr/bin/env python3
"""Stage 7b: device configurations for the generic PE array.

The same IP (PE array + compiler/scheduler) adapts to different-size
FPGAs through a device config: the scheduler takes the DSP budget,
clock and BRAM budget as inputs and produces the tiling + PE
assignment for that device. The RTL is parameterized (number of PEs
and multipliers/tile-shape per PE); changing parameters -- not
rewriting RTL -- rescales the design. The parameterization scales
the parallelism, never the feature set: every configuration runs
the FULL SRCNN pipeline (all 3 layers), the same compiler flow and
the same quantization/precision.

Devices (budgets from the vendor data, not guesses):
  * XCZU19EG-2 (iW-RainboW-G35D): Xilinx Zynq UltraScale+ MPSoC,
    1,968 DSP slices, 984 BRAM36 (4.4 MB), 523K LUTs. Target:
    SRCNN 1080p30. Source: Xilinx DS891 / the Stage 1 board record.
  * LFE5U-85F (Lattice ECP5, low-power): 84K LUTs, 208 sysMEM EBR
    blocks (18 Kbits each = 3,744 Kbits), 669 Kbits distributed RAM,
    156 18x18 multipliers (each sysDSP slice also does 2x 18x18 or
    4x 9x9). Target: SMALL-resolution images (640x480, 320x240),
    full pipeline at a few to ~20 fps. Source: Lattice ECP5 Family
    Data Sheet FPGA-DS-02012-3.2 (Aug 2023), ECP5 selection guide
    (latticesemi.com ECP5 product page). The 100 MHz clock is a
    conservative planning assumption for a first-pass ECP5 design
    (the sysDSP blocks run faster; the firm timing number needs a
    nextpnr run, which is out of scope here -- fps scales linearly
    with the clock, so the sensitivity is visible in the model).

PE sizing (derived in logs/stage7b/design.md, not assumed):
  * ZU19EG: 15 PEs x 128 mults = 1920 DSPs (97.6%). 128 is the sweet
    spot: same DSP total as 30x64 (1920) with half the control
    overhead, and TN=8 divides every SRCNN N (64, 32) exactly while
    staying robust to small-N layers (a 256-mult PE with TN=16 wastes
    half its lanes on C_out=8).
  * LFE5U-85F: 2 PEs x 64 mults = 128 multipliers (82% of 156). 64
    is the right size here: 2 PEs (not 1) so the multi-PE dispatch
    and merge are exercised, and TN=4 keeps small-N layers efficient.
"""
DEVICES = {
    "zu19eg": {
        "name": "XCZU19EG-2 (iW-RainboW-G35D)",
        "vendor": "xilinx",
        "dsp": 1968,
        "lut": 523000,
        "bram_kbits": 984 * 36,
        "clock_mhz": 300.0,
        "clocks_mhz": (300.0, 360.0, 400.0),
        "n_pe": 15,
        "pe_mults": 128,
        "tm": 16,
        "tn": 8,
        "frame_tiles": (64, 32, 16, 8),
        "targets": [(1920, 1080)],
        "verified": "rtl",  # golden + RTL sim on this configuration
    },
    "ecp5u85": {
        "name": "LFE5U-85F (Lattice ECP5, low-power)",
        "vendor": "lattice",
        "dsp": 156,
        "lut": 84000,
        "bram_kbits": 3744,
        "clock_mhz": 100.0,
        "clocks_mhz": (100.0,),
        "n_pe": 2,
        "pe_mults": 64,
        "tm": 16,
        "tn": 4,
        "frame_tiles": (64, 32, 16, 8),
        "targets": [(640, 480), (320, 240)],
        "verified": "model+synth",  # cycle model + synth_ecp5 fit check
    },
}


def get(name):
    """Return the device config dict for `name` (KeyError if unknown)."""
    return DEVICES[name]


def list_devices():
    """Device names in table order."""
    return list(DEVICES)
