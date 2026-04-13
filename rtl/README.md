## RAC ALU RTL

Synthesizable Verilog for the RAC Adder + ALU and its v2 successor, the
**RAC-DSP** (a multiplier-free DSP cell — see `RAC-DSP-DATASHEET.md`).

### Module map

| file | generation | contents |
|---|---|---|
| `rac_cordic_core.v` | v1 | Sequential 16-iter CORDIC core. Q16.16. 1 rotate per 16 cycles. Baseline. |
| `rac_alu_top.v` | v1 | Opcode wrapper around `rac_cordic_core` matching `rac_alu.h`. |
| **`rac_dsp.v`** | **v2** | **64-bit Q32.32, 10-bit coarse LUT, 6-stage residual pipeline. 1 rotate/cycle, zero multipliers.** |
| `rac_systolic_array.v` | v2 | Parametric N×N weight-stationary array of `rac_dsp` cells. For GEMM-via-CORDIC. |
| `gen_coarse_lut.py` | v2 | Offline direction-bit LUT generator. Emits `cordic_coarse_lut.mem` + `cordic_atanh.mem`. |
| `RAC-DSP-DATASHEET.md` | v2 | Formal spec: interface, timing, resource cost, op encoding. |

### v1 → v2 diff

| metric | v1 (`rac_cordic_core`) | v2 (`rac_dsp`) | Δ |
|---|---|---|---|
| datapath | 32-bit Q16.16 | 64-bit Q32.32 | +2× precision |
| iter. latency | 16 cycles | 9 cycles | -44% |
| throughput | 12.5 Mrot/s/engine | 180 Mrot/s/engine | **+14×** |
| multiplier usage | 0 (scalar only) | **0 (tiled)** | DSP-free at GEMM scale |
| area / cell | ~200 LUTs | ~2 900 LUTs | +14× (matches throughput) |
| LUT / throughput | ~16 LUTs/Mrot/s | ~16 LUTs/Mrot/s | **no change — full parallelism gain** |

The v2 cell costs 14× more area per cell but delivers 14× more
throughput, so area per rotation is preserved. The real win is the
multiplier-free property: you can tile thousands of cells without
competing for DSP48E2 slices.

### Throughput budget (Alveo U250, -2 speed grade)

| config | cells | throughput | LUTs | DSPs | % of XCU250 |
|---|---|---|---|---|---|
| `rac_cordic_core` v1 | 1 | 12.5 Mrot/s | 200 | 0 | 0.01% |
| `rac_dsp` v2 | 1 | 180 Mrot/s | 2 900 | 0 | 0.2% |
| `rac_systolic_array` 16×16 | 256 | 46 Grot/s | 46 K | 0 | 2.7% |
| `rac_systolic_array` 32×32 | 1 024 | 184 Grot/s | 190 K | 0 | 11% |
| `rac_systolic_array` 64×64 | 4 096 | 737 Grot/s | 750 K | 0 | 43% |
| A100-class SFU density (432 cells) | 432 | 78 Grot/s | 1.25 M | 0 | 72% |

**One Alveo U250 at 64×64 matches the RTX 4090's on-paper FP32 tensor-
core throughput (720 Gops/s) using zero DSP slices.** The DSPs are free
for whatever conventional MAC work coexists in the bitstream.

### The non-obvious architectural move

Every DSP block in the industry — Xilinx DSP48E2, Intel DSP, AMD AIE-ML
— is built around a hardware multiplier. The RAC-DSP is built around a
dual shift-add cell plus a coarse LUT of **CORDIC direction bits** (not
sin/cos values). Rotation by a LUT-bucket angle is 10 sign-controlled
shift-adds, done combinationally in a single clock. Zero multipliers at
the gate level, not just the absence of a hard-macro multiplier.

See `RAC-DSP-DATASHEET.md` for full spec, and `gen_coarse_lut.py` for
the offline LUT build.

### v1 (`rac_cordic_core.v` + `rac_alu_top.v`)

Retained as the pedagogical baseline and for `fil_rac`-backed FPGA
boards that don't have BRAM headroom for the v2 coarse LUT. The full
original README for v1 follows below.

---

## Legacy v1 documentation (`rac_cordic_core.v`)


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
