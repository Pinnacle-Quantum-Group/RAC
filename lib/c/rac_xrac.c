/*
 * rac_xrac.c — Xrac encoder + ISS implementation
 * Pinnacle Quantum Group — April 2026
 */

#include "rac_xrac.h"
#include <stdio.h>
#include <string.h>

/* ── Encoder ─────────────────────────────────────────────────────────────── */

uint32_t rac_xrac_encode(uint32_t funct7, uint32_t rs2, uint32_t rs1,
                         rac_xrac_f3 funct3, uint32_t rd) {
    return ((funct7 & 0x7Fu)   << 25) |
           ((rs2    & 0x1Fu)   << 20) |
           ((rs1    & 0x1Fu)   << 15) |
           ((((uint32_t)funct3) & 0x7u) << 12) |
           ((rd     & 0x1Fu)   <<  7) |
           RAC_XRAC_OPCODE_CUSTOM0;
}

uint32_t rac_xrac_enc_setmode(rac_alu_mode mode, rac_alu_direction dir) {
    return rac_xrac_encode((uint32_t)mode & 0xF, (uint32_t)dir, 0,
                           RAC_XRAC_F3_SETMODE, 0);
}
uint32_t rac_xrac_enc_micro(uint32_t iter) {
    /* iter goes into funct7 (lowest 5 bits — 0..31). */
    return rac_xrac_encode(iter & 0x1F, 0, 0, RAC_XRAC_F3_MICRO, 0);
}
uint32_t rac_xrac_enc_accum   (void) { return rac_xrac_encode(0,0,0, RAC_XRAC_F3_ACCUM,   0); }
uint32_t rac_xrac_enc_comp    (void) { return rac_xrac_encode(0,0,0, RAC_XRAC_F3_COMP,    0); }
uint32_t rac_xrac_enc_clr_acc (void) { return rac_xrac_encode(0,0,0, RAC_XRAC_F3_CLR_ACC, 0); }
uint32_t rac_xrac_enc_sign    (void) { return rac_xrac_encode(0,0,0, RAC_XRAC_F3_SIGN,    0); }
uint32_t rac_xrac_enc_ret     (void) { return rac_xrac_encode(0,0,0, RAC_XRAC_F3_RET,     0); }

/* RV32I subset encoders. */
uint32_t rac_xrac_enc_ebreak(void) {
    /* EBREAK: imm=1 in I-type SYSTEM. */
    return (1u << 20) | (0u << 15) | (0u << 12) | (0u << 7) | RAC_XRAC_OPCODE_SYSTEM;
}
uint32_t rac_xrac_enc_addi(uint32_t rd, uint32_t rs1, int32_t imm12) {
    uint32_t imm = ((uint32_t)imm12) & 0xFFFu;
    return (imm << 20) | ((rs1 & 0x1F) << 15) | (0u << 12) |
           ((rd & 0x1F) << 7) | RAC_XRAC_OPCODE_ADDI;
}
uint32_t rac_xrac_enc_lui(uint32_t rd, uint32_t imm20) {
    return ((imm20 & 0xFFFFFu) << 12) | ((rd & 0x1F) << 7) | RAC_XRAC_OPCODE_LUI;
}
uint32_t rac_xrac_enc_jal(uint32_t rd, int32_t imm21) {
    /* J-type packing: imm[20|10:1|11|19:12] */
    uint32_t u = (uint32_t)imm21;
    uint32_t w = 0;
    w |= ((u >> 20) & 0x1u) << 31;
    w |= ((u >>  1) & 0x3FFu) << 21;
    w |= ((u >> 11) & 0x1u) << 20;
    w |= ((u >> 12) & 0xFFu) << 12;
    w |= (rd & 0x1F) << 7;
    w |= RAC_XRAC_OPCODE_JAL;
    return w;
}
uint32_t rac_xrac_enc_beq(uint32_t rs1, uint32_t rs2, int32_t imm13) {
    uint32_t u = (uint32_t)imm13;
    uint32_t w = 0;
    w |= ((u >> 12) & 0x1u) << 31;
    w |= ((u >>  5) & 0x3Fu) << 25;
    w |= (rs2 & 0x1F) << 20;
    w |= (rs1 & 0x1F) << 15;
    w |= 0u << 12;   /* funct3=0 for BEQ */
    w |= ((u >> 1)  & 0xFu) << 8;
    w |= ((u >> 11) & 0x1u) << 7;
    w |= RAC_XRAC_OPCODE_BRANCH;
    return w;
}

/* ── Translator: rac_uinst → RV32+Xrac instruction ──────────────────────── */

static uint32_t _xlate_one(rac_uinst w) {
    rac_ucode_op op = (rac_ucode_op)RAC_UC_OP(w);
    switch (op) {
        case RAC_UC_CLR_ACC:
            return rac_xrac_enc_clr_acc();
        case RAC_UC_SETMODE:
            return rac_xrac_enc_setmode(
                       (rac_alu_mode)RAC_UC_IMM8(w),
                       (rac_alu_direction)RAC_UC_IMM16(w));
        case RAC_UC_MICRO:
            return rac_xrac_enc_micro(RAC_UC_IMM8(w));
        case RAC_UC_ACCUM:
            return rac_xrac_enc_accum();
        case RAC_UC_COMP:
            return rac_xrac_enc_comp();
        case RAC_UC_SIGN:
            return rac_xrac_enc_sign();
        case RAC_UC_RET:
            return rac_xrac_enc_ret();
        case RAC_UC_NOP:
        default:
            return rac_xrac_enc_addi(0, 0, 0);   /* canonical NOP */
    }
}

int rac_xrac_translate_rom(const rac_uinst *rom, int rom_len,
                           uint32_t *out, int out_capacity) {
    if (rom_len > out_capacity) rom_len = out_capacity;
    for (int i = 0; i < rom_len; i++) out[i] = _xlate_one(rom[i]);
    return rom_len;
}

/* ── CPU state + ISS ─────────────────────────────────────────────────────── */

void rac_xrac_init(rac_xrac_cpu *cpu,
                   const uint32_t *imem, size_t imem_words) {
    memset(cpu, 0, sizeof(*cpu));
    rac_alu_reset(&cpu->alu);
    cpu->imem       = imem;
    cpu->imem_words = imem_words;
}

/* Extract I-type immediate (sign-extended). */
static inline int32_t _imm_I(uint32_t w) {
    return ((int32_t)w) >> 20;
}

/* Extract B-type immediate (sign-extended). */
static inline int32_t _imm_B(uint32_t w) {
    int32_t imm = 0;
    imm |= ((w >> 31) & 0x1) << 12;
    imm |= ((w >>  7) & 0x1) << 11;
    imm |= ((w >> 25) & 0x3F) << 5;
    imm |= ((w >>  8) & 0xF) << 1;
    /* sign-extend from bit 12 */
    if (imm & 0x1000) imm |= 0xFFFFE000;
    return imm;
}

/* Extract J-type immediate (sign-extended). */
static inline int32_t _imm_J(uint32_t w) {
    int32_t imm = 0;
    imm |= ((w >> 31) & 0x1) << 20;
    imm |= ((w >> 12) & 0xFF) << 12;
    imm |= ((w >> 20) & 0x1) << 11;
    imm |= ((w >> 21) & 0x3FF) << 1;
    if (imm & 0x100000) imm |= 0xFFE00000;
    return imm;
}

int rac_xrac_step(rac_xrac_cpu *cpu) {
    if (cpu->halted) return 1;
    size_t word_idx = cpu->pc / 4;
    if (word_idx >= cpu->imem_words) {
        cpu->halted = 1;
        return -1;
    }
    uint32_t w = cpu->imem[word_idx];
    uint32_t opcode = w & 0x7F;
    uint32_t rd   = (w >>  7) & 0x1F;
    uint32_t f3   = (w >> 12) & 0x7;
    uint32_t rs1  = (w >> 15) & 0x1F;
    uint32_t rs2  = (w >> 20) & 0x1F;
    uint32_t f7   = (w >> 25) & 0x7F;

    int branched = 0;
    cpu->cycles++;

    if (cpu->trace) {
        char buf[96];
        rac_xrac_disasm(w, buf, sizeof(buf));
        printf("[%04X] %08X  %s\n", cpu->pc, w, buf);
    }

    switch (opcode) {
        case RAC_XRAC_OPCODE_CUSTOM0: {
            switch ((rac_xrac_f3)f3) {
                case RAC_XRAC_F3_SETMODE:
                    rac_alu_set_mode(&cpu->alu,
                                     (rac_alu_mode)(f7 & 0xF),
                                     (rac_alu_direction)(rs2 & 0x1));
                    break;
                case RAC_XRAC_F3_MICRO:
                    if (rac_alu_micro_step(&cpu->alu) != 0) {
                        cpu->halted = 1; return -1;
                    }
                    break;
                case RAC_XRAC_F3_ACCUM:
                    rac_alu_accum(&cpu->alu, 1.0f);
                    break;
                case RAC_XRAC_F3_COMP:
                    rac_alu_compensate(&cpu->alu);
                    break;
                case RAC_XRAC_F3_CLR_ACC:
                    rac_alu_clear_acc(&cpu->alu);
                    break;
                case RAC_XRAC_F3_SIGN:
                    (void)rac_alu_sign_decide(&cpu->alu);
                    break;
                case RAC_XRAC_F3_RET:
                    cpu->halted = 1;
                    return 1;
                case RAC_XRAC_F3_HALT:
                default:
                    cpu->halted = 1; return -1;
            }
            break;
        }

        case RAC_XRAC_OPCODE_ADDI: {
            int32_t imm = _imm_I(w);
            if (rd != 0) cpu->xreg[rd] = cpu->xreg[rs1] + (uint32_t)imm;
            break;
        }

        case RAC_XRAC_OPCODE_LUI: {
            uint32_t imm = (w & 0xFFFFF000u);
            if (rd != 0) cpu->xreg[rd] = imm;
            break;
        }

        case RAC_XRAC_OPCODE_JAL: {
            int32_t imm = _imm_J(w);
            if (rd != 0) cpu->xreg[rd] = cpu->pc + 4;
            cpu->pc = cpu->pc + (uint32_t)imm;
            branched = 1;
            break;
        }

        case RAC_XRAC_OPCODE_BRANCH: {
            /* Only BEQ (funct3=0) implemented — enough for our demos. */
            int32_t imm = _imm_B(w);
            if (f3 == 0 && cpu->xreg[rs1] == cpu->xreg[rs2]) {
                cpu->pc = cpu->pc + (uint32_t)imm;
                branched = 1;
            }
            break;
        }

        case RAC_XRAC_OPCODE_SYSTEM: {
            /* EBREAK encoded as imm=1 */
            if ((w >> 20) == 1) {
                cpu->halted = 1;
                return 1;
            }
            break;
        }

        default:
            /* Unrecognised — NOP (keeps the demo robust). */
            break;
    }

    if (!branched) cpu->pc += 4;
    return 0;
}

uint64_t rac_xrac_run(rac_xrac_cpu *cpu, uint64_t max_cycles) {
    uint64_t start = cpu->cycles;
    while (!cpu->halted && (cpu->cycles - start) < max_cycles) {
        int rc = rac_xrac_step(cpu);
        if (rc != 0) break;
    }
    return cpu->cycles - start;
}

/* ── Disassembler ───────────────────────────────────────────────────────── */

static const char *_xrac_f3_name(uint32_t f3) {
    switch (f3) {
        case RAC_XRAC_F3_SETMODE:  return "rac.setmode";
        case RAC_XRAC_F3_MICRO:    return "rac.micro";
        case RAC_XRAC_F3_ACCUM:    return "rac.accum";
        case RAC_XRAC_F3_COMP:     return "rac.comp";
        case RAC_XRAC_F3_CLR_ACC:  return "rac.clr_acc";
        case RAC_XRAC_F3_SIGN:     return "rac.sign";
        case RAC_XRAC_F3_RET:      return "rac.ret";
        case RAC_XRAC_F3_HALT:     return "rac.halt";
        default:                   return "rac.???";
    }
}

int rac_xrac_disasm(uint32_t w, char *buf, int buflen) {
    uint32_t opcode = w & 0x7F;
    uint32_t rd   = (w >>  7) & 0x1F;
    uint32_t f3   = (w >> 12) & 0x7;
    uint32_t rs1  = (w >> 15) & 0x1F;
    uint32_t rs2  = (w >> 20) & 0x1F;
    uint32_t f7   = (w >> 25) & 0x7F;

    switch (opcode) {
        case RAC_XRAC_OPCODE_CUSTOM0:
            switch ((rac_xrac_f3)f3) {
                case RAC_XRAC_F3_SETMODE:
                    return snprintf(buf, buflen, "%-12s %s, %s",
                        _xrac_f3_name(f3),
                        rac_alu_mode_name((rac_alu_mode)(f7 & 0xF)),
                        (rs2 & 1) ? "vec" : "rot");
                case RAC_XRAC_F3_MICRO:
                    return snprintf(buf, buflen, "%-12s i=%u",
                                    _xrac_f3_name(f3), f7 & 0x1F);
                default:
                    return snprintf(buf, buflen, "%-12s", _xrac_f3_name(f3));
            }
        case RAC_XRAC_OPCODE_ADDI:
            return snprintf(buf, buflen, "addi         x%u, x%u, %d",
                            rd, rs1, _imm_I(w));
        case RAC_XRAC_OPCODE_LUI:
            return snprintf(buf, buflen, "lui          x%u, 0x%X",
                            rd, (w >> 12) & 0xFFFFF);
        case RAC_XRAC_OPCODE_JAL:
            return snprintf(buf, buflen, "jal          x%u, %d",
                            rd, _imm_J(w));
        case RAC_XRAC_OPCODE_BRANCH:
            return snprintf(buf, buflen, "beq          x%u, x%u, %d",
                            rs1, rs2, _imm_B(w));
        case RAC_XRAC_OPCODE_SYSTEM:
            return snprintf(buf, buflen, "ebreak");
        default:
            return snprintf(buf, buflen, "?? 0x%08X", w);
    }
}
