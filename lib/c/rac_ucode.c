/*
 * rac_ucode.c — RAC Microcode Interpreter + ROMs
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * See rac_ucode.h. The interpreter below is the executable reference
 * for the Xrac micro-op set: whatever bit pattern it accepts is what
 * a hardware Xrac core must also accept. Matching outputs against
 * rac_alu (the non-microcoded path) proves semantic equivalence.
 */

#include "rac_ucode.h"
#include <stdio.h>
#include <string.h>

/* ── Interpreter ────────────────────────────────────────────────────────── */

int rac_ucore_execute(rac_alu_state *s,
                      const rac_uinst *prog,
                      int prog_len) {
    uint64_t dummy;
    return rac_ucore_execute_counted(s, prog, prog_len, &dummy);
}

int rac_ucore_execute_counted(rac_alu_state *s,
                              const rac_uinst *prog,
                              int prog_len,
                              uint64_t *cycles_out) {
    uint64_t cyc = 0;
    for (int pc = 0; pc < prog_len; pc++) {
        rac_uinst w = prog[pc];
        rac_ucode_op op = (rac_ucode_op)RAC_UC_OP(w);
        cyc++;
        switch (op) {
            case RAC_UC_NOP:
                break;
            case RAC_UC_CLR_ACC:
                rac_alu_clear_acc(s);
                break;
            case RAC_UC_SETMODE:
                rac_alu_set_mode(s,
                    (rac_alu_mode)RAC_UC_IMM8(w),
                    (rac_alu_direction)RAC_UC_IMM16(w));
                break;
            case RAC_UC_MICRO:
                if (rac_alu_micro_step(s) != 0) {
                    *cycles_out = cyc;
                    return -1;
                }
                break;
            case RAC_UC_ACCUM:
                /* imm8 encodes whether to use scale 1.0 (0) or the
                 * polar magnitude held in IMM16-indexed temp. For the
                 * prebuilt ROMs we always use 1.0. */
                rac_alu_accum(s, 1.0f);
                break;
            case RAC_UC_COMP:
                rac_alu_compensate(s);
                break;
            case RAC_UC_SIGN:
                rac_alu_sign_decide(s);
                break;
            case RAC_UC_RET:
                *cycles_out = cyc;
                return pc;
            case RAC_UC_HALT:
            default:
                *cycles_out = cyc;
                return -1;
        }
    }
    *cycles_out = cyc;
    return prog_len;
}

/* ── ROMs ───────────────────────────────────────────────────────────────── */
/*
 * Each ROM is exactly 19 or 20 instructions for 16 CORDIC iterations
 * plus setup/teardown. "Set mode" is one word; each MICRO is one word;
 * ACCUM/RET close the program.
 */

/* Setup word used by every circular ROM. */
#define UC_SET_CIRC_ROT   RAC_UC_MAKE(RAC_UC_SETMODE, 0, \
                           RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION)
#define UC_SET_CIRC_VEC   RAC_UC_MAKE(RAC_UC_SETMODE, 0, \
                           RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_VECTORING)
#define UC_SET_HYP_ROT    RAC_UC_MAKE(RAC_UC_SETMODE, 0, \
                           RAC_ALU_MODE_HYPERBOLIC, RAC_ALU_DIR_ROTATION)
#define UC_MICRO_I(i)     RAC_UC_MAKE(RAC_UC_MICRO, 0, (i), 0)
#define UC_CLR_ACC        RAC_UC_MAKE(RAC_UC_CLR_ACC, 0, 0, 0)
#define UC_ACCUM          RAC_UC_MAKE(RAC_UC_ACCUM,   0, 0, 0)
#define UC_RET            RAC_UC_MAKE(RAC_UC_RET,     0, 0, 0)

/* CORDIC rotate: set mode → 16 × MICRO → RET.  (Plus a leading CLR_ACC
 * so the program leaves the ALU in a clean state.) */
const rac_uinst rac_ucode_rom_rotate[19] = {
    UC_CLR_ACC,         /*  0: acc = 0                          */
    UC_SET_CIRC_ROT,    /*  1: mode=circular, dir=rotation      */
    UC_MICRO_I( 0),     /*  2: iter 0  — d·(y>>0),  atan(1)     */
    UC_MICRO_I( 1),     /*  3: iter 1  — d·(y>>1),  atan(1/2)   */
    UC_MICRO_I( 2),     /*  4: iter 2  — d·(y>>2),  atan(1/4)   */
    UC_MICRO_I( 3),     /*  5                                   */
    UC_MICRO_I( 4),     /*  6                                   */
    UC_MICRO_I( 5),     /*  7                                   */
    UC_MICRO_I( 6),     /*  8                                   */
    UC_MICRO_I( 7),     /*  9                                   */
    UC_MICRO_I( 8),     /* 10                                   */
    UC_MICRO_I( 9),     /* 11                                   */
    UC_MICRO_I(10),     /* 12                                   */
    UC_MICRO_I(11),     /* 13                                   */
    UC_MICRO_I(12),     /* 14                                   */
    UC_MICRO_I(13),     /* 15                                   */
    UC_MICRO_I(14),     /* 16                                   */
    UC_MICRO_I(15),     /* 17: iter 15 — d·(y>>15), atan(2^-15) */
    UC_RET,             /* 18: return — x,y in regs, magnitude |v|
                                                  (K-compensated when
                                                   caller pre-scales by
                                                   K_INV, as rac_alu_rotate
                                                   does)                   */
};

/* CORDIC vectoring (polar): same CORDIC iteration sequence, but the
 * direction bit comes from sign(-y) instead of sign(z). The mode word
 * encodes that. */
const rac_uinst rac_ucode_rom_polar[19] = {
    UC_CLR_ACC,
    UC_SET_CIRC_VEC,
    UC_MICRO_I( 0), UC_MICRO_I( 1), UC_MICRO_I( 2), UC_MICRO_I( 3),
    UC_MICRO_I( 4), UC_MICRO_I( 5), UC_MICRO_I( 6), UC_MICRO_I( 7),
    UC_MICRO_I( 8), UC_MICRO_I( 9), UC_MICRO_I(10), UC_MICRO_I(11),
    UC_MICRO_I(12), UC_MICRO_I(13), UC_MICRO_I(14), UC_MICRO_I(15),
    UC_RET,
};

/* Project: rotate, then dump x into the accumulator. This is the canonical
 * MAC-equivalent — a full CORDIC path ending in a single ACCUM. */
const rac_uinst rac_ucode_rom_project[20] = {
    UC_CLR_ACC,
    UC_SET_CIRC_ROT,
    UC_MICRO_I( 0), UC_MICRO_I( 1), UC_MICRO_I( 2), UC_MICRO_I( 3),
    UC_MICRO_I( 4), UC_MICRO_I( 5), UC_MICRO_I( 6), UC_MICRO_I( 7),
    UC_MICRO_I( 8), UC_MICRO_I( 9), UC_MICRO_I(10), UC_MICRO_I(11),
    UC_MICRO_I(12), UC_MICRO_I(13), UC_MICRO_I(14), UC_MICRO_I(15),
    UC_ACCUM,
    UC_RET,
};

/* Hyperbolic rotation core — used for rac_exp / rac_tanh. */
const rac_uinst rac_ucode_rom_exp_core[19] = {
    UC_CLR_ACC,
    UC_SET_HYP_ROT,
    UC_MICRO_I( 0), UC_MICRO_I( 1), UC_MICRO_I( 2), UC_MICRO_I( 3),
    UC_MICRO_I( 4), UC_MICRO_I( 5), UC_MICRO_I( 6), UC_MICRO_I( 7),
    UC_MICRO_I( 8), UC_MICRO_I( 9), UC_MICRO_I(10), UC_MICRO_I(11),
    UC_MICRO_I(12), UC_MICRO_I(13), UC_MICRO_I(14), UC_MICRO_I(15),
    UC_RET,
};

int rac_ucode_rom_len(const rac_uinst *rom) {
    if (rom == rac_ucode_rom_rotate)   return 19;
    if (rom == rac_ucode_rom_polar)    return 19;
    if (rom == rac_ucode_rom_project)  return 20;
    if (rom == rac_ucode_rom_exp_core) return 19;
    return 0;
}

/* ── Disassembler ───────────────────────────────────────────────────────── */

static const char *_op_name(rac_ucode_op op) {
    switch (op) {
        case RAC_UC_NOP:      return "nop";
        case RAC_UC_CLR_ACC:  return "clr_acc";
        case RAC_UC_SETMODE:  return "setmode";
        case RAC_UC_MICRO:    return "micro";
        case RAC_UC_ACCUM:    return "accum";
        case RAC_UC_COMP:     return "comp";
        case RAC_UC_SIGN:     return "sign";
        case RAC_UC_RET:      return "ret";
        case RAC_UC_HALT:     return "halt";
        default:              return "???";
    }
}

int rac_ucode_disasm(rac_uinst w, char *buf, int buflen) {
    rac_ucode_op op = (rac_ucode_op)RAC_UC_OP(w);
    const char *name = _op_name(op);

    switch (op) {
        case RAC_UC_SETMODE: {
            const char *m = rac_alu_mode_name((rac_alu_mode)RAC_UC_IMM8(w));
            const char *d = (RAC_UC_IMM16(w) == RAC_ALU_DIR_ROTATION)
                            ? "rot" : "vec";
            return snprintf(buf, buflen, "%-8s %s, %s", name, m, d);
        }
        case RAC_UC_MICRO:
            return snprintf(buf, buflen, "%-8s i=%u", name, RAC_UC_IMM8(w));
        default:
            return snprintf(buf, buflen, "%-8s", name);
    }
}

void rac_ucode_dump(const rac_uinst *prog, int prog_len) {
    char line[96];
    printf("  PC | word        | instruction\n");
    printf(" ────┼─────────────┼──────────────────────\n");
    for (int pc = 0; pc < prog_len; pc++) {
        rac_ucode_disasm(prog[pc], line, sizeof(line));
        printf(" %4d | 0x%08X  | %s\n", pc, prog[pc], line);
    }
}
