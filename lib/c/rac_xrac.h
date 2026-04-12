/*
 * rac_xrac.h — RISC-V Xrac Custom-0 Extension: Encoding + ISS
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * Xrac is a custom-0 (opcode 0b0001011 = 0x0B) R-type ISA extension that
 * exposes the RAC ALU micro-ops as individual RV32 instructions. The
 * rac_ucode microsequencer is re-cast here as an RV32-native fetch/
 * decode/execute loop reading 32-bit instruction words from a program
 * memory — the same arrangement a real RV32I+Xrac core uses.
 *
 * This header + rac_xrac.c contain:
 *   - A complete Xrac encoder (rac_xrac_encode_*)
 *   - A decoder + ISS (rac_xrac_step / rac_xrac_run)
 *   - A translator that lifts rac_ucode ROMs into RV32-encoded programs
 *     (rac_xrac_translate_rom) so the same CORDIC tapes drive the ISS
 *   - Tracing hooks for cycle-accurate bring-up
 *
 * RV32 R-type layout (for reference):
 *
 *     31    25 24   20 19   15 14  12 11    7 6        0
 *   ┌────────┬───────┬───────┬──────┬───────┬──────────┐
 *   │ funct7 │  rs2  │  rs1  │funct3│  rd   │ 0b0001011│
 *   └────────┴───────┴───────┴──────┴───────┴──────────┘
 *
 * Xrac funct3 map (closes over the nine rac_ucode opcodes):
 *
 *     funct3  mnemonic         funct7 / rs2 / rs1 usage
 *     ──────  ──────────────   ────────────────────────────────────────
 *     0x0     rac.setmode      funct7[3:0]=mode, rs2=dir
 *     0x1     rac.micro        funct7[4:0] = iter index (i for 2^-i)
 *     0x2     rac.accum        —
 *     0x3     rac.comp         —
 *     0x4     rac.clr_acc      —
 *     0x5     rac.sign         —
 *     0x6     rac.ret          (system-like; terminates ROM)
 *     0x7     rac.halt         — (illegal / error)
 *
 * rd, rs1, rs2 are RV32 integer regs for operand delivery where needed.
 * For the CORDIC ROMs we operate on architectural ALU state held inside
 * rac_xrac_cpu.alu — the same state machine rac_ucore_execute drives.
 *
 * We also decode a tiny RV32I subset so you can write real programs:
 *   ADDI, LUI, JAL, BEQ, EBREAK
 * Enough to run a hand-assembled program that calls Xrac instructions
 * in a loop. The decoder no-ops unrecognised instructions rather than
 * trapping, which keeps the demo self-contained.
 */

#ifndef RAC_XRAC_H
#define RAC_XRAC_H

#include "rac_alu.h"
#include "rac_ucode.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opcode constants ────────────────────────────────────────────────────── */

#define RAC_XRAC_OPCODE_CUSTOM0  0x0B   /* 0b0001011 */
#define RAC_XRAC_OPCODE_ADDI     0x13   /* 0b0010011 I-type */
#define RAC_XRAC_OPCODE_LUI      0x37   /* 0b0110111 U-type */
#define RAC_XRAC_OPCODE_JAL      0x6F   /* 0b1101111 J-type */
#define RAC_XRAC_OPCODE_BRANCH   0x63   /* 0b1100011 B-type (BEQ/BNE/...) */
#define RAC_XRAC_OPCODE_SYSTEM   0x73   /* 0b1110011 I-type (ECALL/EBREAK) */

/* Xrac funct3 codes — same ordering as rac_ucode opcodes so the two
 * encoders can share a mapping. */
typedef enum {
    RAC_XRAC_F3_SETMODE  = 0x0,
    RAC_XRAC_F3_MICRO    = 0x1,
    RAC_XRAC_F3_ACCUM    = 0x2,
    RAC_XRAC_F3_COMP     = 0x3,
    RAC_XRAC_F3_CLR_ACC  = 0x4,
    RAC_XRAC_F3_SIGN     = 0x5,
    RAC_XRAC_F3_RET      = 0x6,
    RAC_XRAC_F3_HALT     = 0x7,
} rac_xrac_f3;

/* ── Encoder ─────────────────────────────────────────────────────────────── */

/* R-type word builder (re-exports the helper in rac_ucode.h). */
uint32_t rac_xrac_encode(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                         rac_xrac_f3 funct3, uint32_t rd);

/* Convenience encoders for each Xrac op. */
uint32_t rac_xrac_enc_setmode (rac_alu_mode mode, rac_alu_direction dir);
uint32_t rac_xrac_enc_micro   (uint32_t iter);    /* iter stored in funct7 */
uint32_t rac_xrac_enc_accum   (void);
uint32_t rac_xrac_enc_comp    (void);
uint32_t rac_xrac_enc_clr_acc (void);
uint32_t rac_xrac_enc_sign    (void);
uint32_t rac_xrac_enc_ret     (void);

/* RV32I subset. */
uint32_t rac_xrac_enc_ebreak  (void);
uint32_t rac_xrac_enc_addi    (uint32_t rd, uint32_t rs1, int32_t imm12);
uint32_t rac_xrac_enc_lui     (uint32_t rd, uint32_t imm20);
uint32_t rac_xrac_enc_jal     (uint32_t rd, int32_t imm21);
uint32_t rac_xrac_enc_beq     (uint32_t rs1, uint32_t rs2, int32_t imm13);

/* Translate a rac_ucode ROM into RV32+Xrac machine code.
 * Output buffer must have room for (rom_len * 4) bytes (one word per
 * microinstruction). Returns the number of words written. */
int rac_xrac_translate_rom(const rac_uinst *rom, int rom_len,
                           uint32_t *out_words, int out_capacity);

/* ── CPU state + ISS ─────────────────────────────────────────────────────── */

typedef struct {
    rac_alu_state  alu;            /* ALU = floating-point co-processor */
    uint32_t       xreg[32];       /* RV32I integer register file      */
    uint32_t       pc;             /* byte address of next instruction */
    const uint32_t *imem;          /* instruction memory               */
    size_t         imem_words;     /* size of imem in 32-bit words     */
    uint64_t       cycles;         /* cycles retired                   */
    int            halted;         /* set on EBREAK or RET             */
    int            trace;          /* non-zero → print every instr     */
} rac_xrac_cpu;

/* Initialize: zero regs, set pc=0, attach imem. */
void rac_xrac_init(rac_xrac_cpu *cpu,
                   const uint32_t *imem, size_t imem_words);

/* Single step: fetch, decode, execute one instruction.
 * Returns 0 on success, -1 on fault, +1 on halt. */
int  rac_xrac_step(rac_xrac_cpu *cpu);

/* Run until halt or max_cycles reached. Returns retired cycle count. */
uint64_t rac_xrac_run(rac_xrac_cpu *cpu, uint64_t max_cycles);

/* Disassemble one instruction word. Returns chars written to buf. */
int  rac_xrac_disasm(uint32_t word, char *buf, int buflen);

#ifdef __cplusplus
}
#endif

#endif /* RAC_XRAC_H */
