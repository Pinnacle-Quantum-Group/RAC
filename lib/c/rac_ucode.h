/*
 * rac_ucode.h — RAC Microcode Instruction Set and RISC-V Xrac Sketch
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * The RAC ALU opcode set in rac_alu.h is already micro-instruction-like:
 * eight opcodes, fixed-width, no control flow. This file formalises that
 * into a 32-bit microcode ISA and provides:
 *
 *   1. An instruction encoding / decoding scheme.
 *   2. A pure-software microsequencer (rac_ucore_execute) that fetches
 *      32-bit words from a program ROM and drives the ALU state machine.
 *   3. Pre-built ROMs for rotate / polar / project / inner (the "boot
 *      tape" for each RAC primitive).
 *   4. A disassembler (rac_ucode_disasm) for programmer sanity.
 *   5. A layer-2 sketch showing how these micro-ops map onto RISC-V
 *      custom-0 opcode space as an Xrac extension.
 *
 * Why this is useful:
 *   - The ROM for a 16-iter CORDIC rotate is literally 19 instructions.
 *     You can read the entire CORDIC algorithm as a flat ROM table.
 *   - The C interpreter IS the ISA simulator: if it matches rac_alu,
 *     the microcode is correct.
 *   - The same ROM can be lifted into synthesizable Verilog as a ROM
 *     module driving the ALU datapath. The "hardware" and the "software"
 *     agree by construction.
 *   - For a RISC-V implementation, encoding the eight micro-ops into
 *     custom-0 instruction bits makes the ALU a drop-in FP extension.
 *
 * Instruction format (32 bits, little-endian fields):
 *
 *     31                    24 23              16 15                 0
 *   ┌─────┬─────────────────┬─────────────────┬─────────────────────┐
 *   │ op  │      mode       │      imm8       │        imm16        │
 *   │ 4b  │       4b        │       8b        │         16b         │
 *   └─────┴─────────────────┴─────────────────┴─────────────────────┘
 *
 *   op    — RAC_UC_* opcode
 *   mode  — RAC_ALU_MODE_* (for SETMODE) or dir for ACCUM, unused else
 *   imm8  — shift amount / iter index / float-immediate encoding
 *   imm16 — direction / table index / small constant
 */

#ifndef RAC_UCODE_H
#define RAC_UCODE_H

#include "rac_alu.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t rac_uinst;

/* ── Encoding helpers ────────────────────────────────────────────────────── */

#define RAC_UC_MAKE(op, mode, imm8, imm16)          \
    ( ((uint32_t)((op)    & 0xFu)   << 28) |        \
      ((uint32_t)((mode)  & 0xFu)   << 24) |        \
      ((uint32_t)((imm8)  & 0xFFu)  << 16) |        \
      ((uint32_t)((imm16) & 0xFFFFu) <<  0) )

#define RAC_UC_OP(w)    (((uint32_t)(w) >> 28) & 0xFu)
#define RAC_UC_MODE(w)  (((uint32_t)(w) >> 24) & 0xFu)
#define RAC_UC_IMM8(w)  (((uint32_t)(w) >> 16) & 0xFFu)
#define RAC_UC_IMM16(w) (((uint32_t)(w) >>  0) & 0xFFFFu)

/* ── Opcodes ─────────────────────────────────────────────────────────────── */

typedef enum {
    RAC_UC_NOP      = 0x0,   /* no-op                                        */
    RAC_UC_CLR_ACC  = 0x1,   /* acc ← 0                                      */
    RAC_UC_SETMODE  = 0x2,   /* set mode (imm8), dir (imm16)                 */
    RAC_UC_MICRO    = 0x3,   /* one CORDIC iteration                         */
    RAC_UC_ACCUM    = 0x4,   /* acc += x · scale  (scale = 1.0)              */
    RAC_UC_COMP     = 0x5,   /* apply K^-chain gain compensation             */
    RAC_UC_SIGN     = 0x6,   /* d ← sign(z) or sign(-y) per dir              */
    RAC_UC_RET      = 0xE,   /* return cleanly                               */
    RAC_UC_HALT     = 0xF,   /* halt with error (undefined opcode fallback)  */
} rac_ucode_op;

/* ── Microsequencer ─────────────────────────────────────────────────────── */

/*
 * Execute a microcode program against an ALU state. The ALU must be
 * pre-loaded by the caller via rac_alu_load before entry (LOAD is
 * deliberately not an opcode — operand delivery is host business).
 *
 * Returns:
 *    pc reached (>=0) on clean RET or end-of-program
 *    -1 on HALT / undefined opcode
 */
int rac_ucore_execute(rac_alu_state *s,
                      const rac_uinst *prog,
                      int prog_len);

/* Count executed instructions (for ISS-style cycle counting). */
int rac_ucore_execute_counted(rac_alu_state *s,
                              const rac_uinst *prog,
                              int prog_len,
                              uint64_t *cycles_out);

/* ── Prebuilt ROMs ──────────────────────────────────────────────────────── */

/*
 * Each ROM assumes the caller has loaded x,y,z via rac_alu_load before
 * execution. The ROMs emit their result in s->x, s->y, s->z (and s->acc
 * for project/inner).
 *
 * rac_ucode_rom_rotate:   circular rotation, 16-iter.   19 instructions.
 * rac_ucode_rom_polar:    circular vectoring, 16-iter.  19 instructions.
 * rac_ucode_rom_project:  rotate + ACCUM.               20 instructions.
 * rac_ucode_rom_exp_core: hyperbolic rotation, 16-iter. 19 instructions.
 */
extern const rac_uinst rac_ucode_rom_rotate   [19];
extern const rac_uinst rac_ucode_rom_polar    [19];
extern const rac_uinst rac_ucode_rom_project  [20];
extern const rac_uinst rac_ucode_rom_exp_core [19];

/* Length of each ROM (in instructions). */
int rac_ucode_rom_len(const rac_uinst *rom);

/* ── Disassembler ───────────────────────────────────────────────────────── */

/* Disassemble one instruction into `buf` (caller-provided, >=64 bytes).
 * Returns the number of characters written. */
int rac_ucode_disasm(rac_uinst w, char *buf, int buflen);

/* Disassemble a full program to stdout with PC annotations. */
void rac_ucode_dump(const rac_uinst *prog, int prog_len);

/* ── Layer 2: RISC-V Xrac custom-0 encoding sketch ──────────────────────── */
/*
 * RISC-V reserves opcode[6:0] = 0b0001011 (0x0B) for custom-0. Xrac maps
 * the RAC micro-ops into R-type instructions at that opcode:
 *
 *     31    25 24   20 19   15 14  12 11    7 6        0
 *   ┌────────┬───────┬───────┬──────┬───────┬──────────┐
 *   │ funct7 │  rs2  │  rs1  │fnct3 │  rd   │ 0b0001011│
 *   └────────┴───────┴───────┴──────┴───────┴──────────┘
 *
 *     funct3   meaning
 *     ─────── ─────────────────────────────────────────────
 *     0x0     rac.micro  rd, rs1, rs2
 *             rs1 = {x,y} packed FP, rs2 = {z, mode, iter},
 *             rd  = {x', y'} post-micro
 *     0x1     rac.sign   rd, rs1
 *             rd = copysign(1.0, rs1) — the direction bit
 *     0x2     rac.accum  rd, rs1, rs2
 *             rd = rs1 + x_lane(rs2) · 1.0  (acc FMA)
 *     0x3     rac.comp   rd, rs1, imm5 (encoded in funct7)
 *             rd = rs1 · K^-imm5
 *     0x4     rac.polar  rd, rs1
 *             rs1 = {x,y}; rd = {|v|, atan2(y,x)} in one issue
 *     0x5     rac.rot    rd, rs1, rs2
 *             rs1 = {x,y}; rs2 = theta; rd = rotated vector
 *             (multi-cycle, 16 micro-ops fused)
 *
 * The funct7 field selects mode/iter for rac.micro (so the single
 * R-type instruction covers all 32 circular/hyperbolic iteration
 * variants). The same microcode ROMs above would be embedded in the
 * Xrac sequencer FSM — the C interpreter serves as a bit-for-bit
 * reference for Spike/Sail integration.
 */

/* These helpers are provided for an eventual Xrac encoder. They are
 * currently unused by the software path. */
#define RAC_XRAC_OPCODE 0x0B

static inline uint32_t rac_xrac_encode_R(uint32_t funct7, uint32_t rs2,
                                         uint32_t rs1,    uint32_t funct3,
                                         uint32_t rd) {
    return ((funct7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) |
           ((rs1    & 0x1F) << 15) | ((funct3 & 0x7) << 12) |
           ((rd     & 0x1F) <<  7) | RAC_XRAC_OPCODE;
}

#ifdef __cplusplus
}
#endif

#endif /* RAC_UCODE_H */
