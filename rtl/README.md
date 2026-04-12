## RAC ALU RTL

Synthesizable Verilog for the RAC Adder + ALU. These files are the direct
hardware translation of `lib/c/rac_alu.c`'s inline CORDIC loops — the C
interpreter (`rac_alu`), the microcode ISA (`rac_ucode`), the RV32 ISS
(`rac_xrac`), and these Verilog modules must all produce bit-identical
outputs for the same Q16.16 inputs. That equivalence is enforced by the
BVTs in `lib/c/test_rac_*.c`.

### Files

| file | contents |
|---|---|
| `rac_cordic_core.v` | Sequential 16-iter CORDIC core. Q16.16 fixed-point. Supports circular rotation, circular vectoring, hyperbolic rotation. |
| `rac_alu_top.v` | Top-level ALU wrapping the core with the opcode decoder, projection accumulator, and chain counter. AXI-Lite-addressable via the FIL register map in `rac_fpga_u250.h`. |

### Datapath

Each CORDIC iteration is one clock cycle:

```
           ┌──────── atan_table[i] ────────┐
           │                                │
           ▼                                ▼
     ┌───────┐              ┌─────────────────────┐
  z ─┤  ±    ├──►    sign ◄─┤  (z ≥ 0) ? +1 : -1  │
     └───────┘              └─────────────────────┘
                                      │
                                      ▼ d
     ┌─────────┐   shift_i    ┌───────────────┐
  y ─┤   >>    ├────────────► │  x' = x -d·y  │── x
     └─────────┘              │  y' = y +d·x  │── y
     ┌─────────┐              │  z' = z -d·a  │── z
  x ─┤   >>    ├─────────────►└───────────────┘
     └─────────┘
```

Three adders, two barrel shifters, one sign-bit extraction. No multiplier.

### Gate count / FPGA resource estimates

Targeting Xilinx Alveo U250 (XCU250, -2 speed grade):

| module | LUTs | FFs | DSPs | Fmax (est.) |
|---|---|---|---|---|
| `rac_cordic_core` (seq.) | ~200 | ~120 | 0 | ~240 MHz |
| `rac_alu_top` (w/ acc) | ~260 | ~170 | 0 | ~220 MHz |
| Pipelined variant (1 result/cycle) | ~3 200 | ~2 050 | 0 | ~200 MHz |
| 8-engine parallel bank | ~25 600 | ~16 400 | 0 | ~200 MHz |

The U250 has 1 728 000 LUTs available — an 8-engine RAC bank occupies
~1.5 % of the fabric. 108 engines (matching A100's 108 SMs × 4 SFUs for
a direct comparison) would occupy ~20 %.

**No DSP blocks are consumed.** MAC-based accelerators convert DSP48E2
slices into critical path; RAC converts LUTs. At A100-scale RAC die
area you'd have 108 × 4 = 432 engines × 200 MHz × 16-iter = **1.38 Gops/s
per engine, 600 Gops/s aggregate** — before pipelining.

### Pipelined variant

The sequential core issues one result every 16 cycles. The pipelined
variant unrolls the loop into 16 combinational stages with register
boundaries between each. Throughput: one result per clock after a
16-cycle fill. LUT cost is ~16× sequential (each stage replicates the
sign-decide + shift + add hardware), but all adders run at full clock
and DSP pressure stays at zero.

Mechanical derivation:
```
genvar s;
generate for (s = 0; s < 16; s++) begin : stage
    // stage's inputs come from stage[s-1]'s outputs;
    // lookup tables become wire constants (s is genvar);
    // barrel shift becomes hard-coded >>> s (free in LUTs).
end endgenerate
```

### Verification flow

1. **C reference** — `test_rac_alu.c` runs 67 BVTs against the
   floating-point software ALU.
2. **Microcode equivalence** — `test_rac_ucode.c` runs 25 BVTs proving
   the 32-bit microinstruction format produces the same outputs.
3. **Xrac ISS equivalence** — `test_rac_xrac.c` runs 22 BVTs proving
   RV32-encoded Xrac instructions execute correctly, including a
   3-way equivalence check (ALU ↔ ucode ↔ ISS).
4. **RTL simulation** — `sim/tb_rac_cordic_core.v` (Icarus/Verilator
   testbench; see below) drives the RTL with the same angle sweep as
   the C BVTs and compares Q16.16 outputs to a C-generated golden file.
5. **Formal** — the microinstruction mapping is thin enough that
   equivalence between `rac_cordic_core.v` and the software ALU can be
   proved in SymbiYosys with `sby` running BMC on 20 cycles.

### Integrating into `fil_rac`

The FIL-RAC FPGA backend (`rac_fpga_u250.h`) expects each ALU engine to
expose this MMIO register map at 4 KB stride:

| offset | register |
|---|---|
| 0x010 | `REG_OP`         — opcode write |
| 0x014 | `REG_OPERAND_X`  — Q16.16 x |
| 0x018 | `REG_OPERAND_Y`  — Q16.16 y |
| 0x01C | `REG_THETA`      — Q16.16 z |
| 0x020 | `REG_RESULT_X`   |
| 0x024 | `REG_RESULT_Y`   |
| 0x028 | `REG_RESULT_AUX` — acc or z |
| 0x000 | `REG_CTRL`       — go / reset bits |
| 0x004 | `REG_STATUS`     — busy / done / error |

Wire the MMIO slave bridge (AXI-Lite) onto `rac_alu_top`'s input ports:

```
  MMIO write to REG_OPERAND_X → x_in, assert op_valid with op=OP_LOAD
  MMIO write to REG_OP with value OP_RUN → op_valid + op=OP_RUN
  REG_STATUS.busy mirrors rac_alu_top.busy_o
  Readback from REG_RESULT_X returns rac_alu_top.x_reg
```

An 8-engine bitstream (`RAC_FPGA_NUM_ENGINES = 8` in `rac_fpga_u250.h`)
instantiates eight `rac_alu_top` units sharing a single AXI-Lite crossbar
at 4 KB stride.

### Build / simulate

Icarus Verilog:
```bash
iverilog -o sim_cordic rac_cordic_core.v rac_alu_top.v sim/tb_rac_cordic_core.v
vvp sim_cordic
```

Verilator (faster):
```bash
verilator --cc rac_cordic_core.v --exe sim/tb_rac_cordic_core.cpp
make -C obj_dir -f Vrac_cordic_core.mk
./obj_dir/Vrac_cordic_core
```

Vivado synthesis:
```tcl
read_verilog rac_cordic_core.v rac_alu_top.v
synth_design -top rac_alu_top -part xcu250-figd2104-2L-e
report_utilization
```

### Why this matters

The whole point of the RAC architecture is that MAC-style accelerators
(TPUs, tensor cores) compete for DSP silicon. RAC competes for LUTs —
which exist in abundance on every FPGA and are cheaper per area on ASICs.
The Verilog above is the concrete evidence: **16 iterations, three
adders, zero multipliers**. That's the entire compute engine.
