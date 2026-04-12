/*
 * test_rac_xrac.c — RV32 + Xrac ISS BVT
 * Pinnacle Quantum Group — April 2026
 *
 * Three-way equivalence proof:
 *
 *   1. rac_alu_rotate                       (direct ALU path)
 *   2. rac_ucore_execute(rac_ucode_rom_*)   (microcode interpreter)
 *   3. rac_xrac_run(translated_rom)         (RV32+Xrac ISS)
 *
 * Whatever angle you feed in, all three paths must produce the same
 * (x, y) to within CORDIC rounding. If that holds, the RV32+Xrac
 * encoding is semantically complete.
 */

#include "rac_xrac.h"
#include "rac_ucode.h"
#include "rac_alu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int passed = 0, failed = 0;
#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)
#define HEADER(s) printf("\n== %s ==\n", s)

/* Run the rotate ROM through the Xrac ISS and return (x, y). */
static rac_vec2 run_rotate_xrac(rac_vec2 v, float theta) {
    /* Host-side quadrant fold + K_INV pre-scale (what rac_alu_rotate does
     * before entering the CORDIC loop — this is the "driver" the Xrac
     * program expects). */
    float t = theta;
    while (t >  3.14159265f) t -= 2.0f * 3.14159265f;
    while (t < -3.14159265f) t += 2.0f * 3.14159265f;
    if (t > 0.5f * 3.14159265f)       { v.x = -v.x; v.y = -v.y; t -= 3.14159265f; }
    else if (t < -0.5f * 3.14159265f) { v.x = -v.x; v.y = -v.y; t += 3.14159265f; }

    /* Translate the rotate ROM into RV32+Xrac machine code and append
     * an EBREAK so the ISS halts cleanly. */
    uint32_t imem[32];
    int rom_len = 19;
    int nwords  = rac_xrac_translate_rom(rac_ucode_rom_rotate, rom_len,
                                         imem, 31);
    imem[nwords] = rac_xrac_enc_ebreak();
    nwords++;

    rac_xrac_cpu cpu;
    rac_xrac_init(&cpu, imem, (size_t)nwords);
    /* Seed ALU registers directly — in a real chip this would be a
     * pair of custom loads, or the scalar RV32I code would have written
     * FP regs and an Xrac LD_XY instruction would move them into the
     * ALU. We're demonstrating the CORDIC pipeline, not the glue. */
    rac_alu_load(&cpu.alu, v.x * RAC_ALU_K_INV, v.y * RAC_ALU_K_INV, t);

    (void)rac_xrac_run(&cpu, 200);
    return (rac_vec2){cpu.alu.x, cpu.alu.y};
}

int main(void) {
    printf("RAC Xrac ISS BVT — Pinnacle Quantum Group\n");

    /* ── 1. Encoder round-trip ──────────────────────────────────────── */
    HEADER("1. Encoder / decoder round-trip");
    uint32_t w;
    char     buf[96];

    w = rac_xrac_enc_setmode(RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_xrac_disasm(w, buf, sizeof(buf));
    CHECK("setmode opcode = CUSTOM-0", (w & 0x7F) == RAC_XRAC_OPCODE_CUSTOM0);
    CHECK("setmode disasm mentions CIRCULAR", strstr(buf, "CIRCULAR") != NULL);

    w = rac_xrac_enc_micro(5);
    rac_xrac_disasm(w, buf, sizeof(buf));
    CHECK("micro opcode = CUSTOM-0", (w & 0x7F) == RAC_XRAC_OPCODE_CUSTOM0);
    CHECK("micro disasm mentions i=5", strstr(buf, "i=5") != NULL);

    w = rac_xrac_enc_ebreak();
    CHECK("ebreak opcode = SYSTEM",  (w & 0x7F) == RAC_XRAC_OPCODE_SYSTEM);
    CHECK("ebreak imm==1",           (w >> 20)  == 1);

    w = rac_xrac_enc_addi(1, 0, 42);
    CHECK("addi opcode = OP-IMM",    (w & 0x7F) == RAC_XRAC_OPCODE_ADDI);
    CHECK("addi rd=1",               ((w >> 7) & 0x1F) == 1);
    CHECK("addi imm=42",             (w >> 20) == 42);

    w = rac_xrac_enc_jal(1, 8);
    CHECK("jal opcode = JAL",        (w & 0x7F) == RAC_XRAC_OPCODE_JAL);

    /* ── 2. Translate rotate ROM ────────────────────────────────────── */
    HEADER("2. ROM → Xrac translation");
    uint32_t mem[32];
    int nw = rac_xrac_translate_rom(rac_ucode_rom_rotate, 19, mem, 32);
    CHECK("translate count = 19", nw == 19);
    CHECK("mem[0] is clr_acc",
          ((mem[0] & 0x7F) == RAC_XRAC_OPCODE_CUSTOM0) &&
          (((mem[0] >> 12) & 7) == RAC_XRAC_F3_CLR_ACC));
    CHECK("mem[1] is setmode",
          (((mem[1] >> 12) & 7) == RAC_XRAC_F3_SETMODE));
    CHECK("mem[2] is micro i=0",
          (((mem[2] >> 12) & 7) == RAC_XRAC_F3_MICRO) &&
          ((mem[2] >> 25) & 0x1F) == 0);
    CHECK("mem[17] is micro i=15",
          (((mem[17] >> 12) & 7) == RAC_XRAC_F3_MICRO) &&
          ((mem[17] >> 25) & 0x1F) == 15);
    CHECK("mem[18] is ret",
          (((mem[18] >> 12) & 7) == RAC_XRAC_F3_RET));

    /* ── 3. Three-way equivalence ───────────────────────────────────── */
    HEADER("3. Three-way equivalence (ALU ↔ ucode ↔ Xrac ISS)");
    int mismatches = 0;
    for (float th = -1.4f; th <= 1.4f; th += 0.2f) {
        rac_vec2 v_in = {1.0f, 0.0f};
        rac_vec2 r_alu = rac_alu_rotate(v_in, th);

        /* Microcode path */
        {
            rac_alu_state s;
            rac_alu_reset(&s);
            float t = th;
            rac_vec2 vv = v_in;
            /* same host-side fold as rac_alu_rotate */
            while (t >  3.14159265f) t -= 2.0f * 3.14159265f;
            while (t < -3.14159265f) t += 2.0f * 3.14159265f;
            if (t > 0.5f * 3.14159265f)       { vv.x = -vv.x; vv.y = -vv.y; t -= 3.14159265f; }
            else if (t < -0.5f * 3.14159265f) { vv.x = -vv.x; vv.y = -vv.y; t += 3.14159265f; }
            rac_alu_load(&s, vv.x * RAC_ALU_K_INV, vv.y * RAC_ALU_K_INV, t);
            (void)rac_ucore_execute(&s, rac_ucode_rom_rotate, 19);
            rac_vec2 r_uc = {s.x, s.y};

            if (fabsf(r_alu.x - r_uc.x) > 1e-5f || fabsf(r_alu.y - r_uc.y) > 1e-5f) {
                printf("  [MISMATCH UCODE] θ=%.2f alu=(%f,%f) ucode=(%f,%f)\n",
                       th, r_alu.x, r_alu.y, r_uc.x, r_uc.y);
                mismatches++;
            }
        }
        /* Xrac ISS path */
        {
            rac_vec2 r_xrac = run_rotate_xrac(v_in, th);
            if (fabsf(r_alu.x - r_xrac.x) > 1e-5f || fabsf(r_alu.y - r_xrac.y) > 1e-5f) {
                printf("  [MISMATCH XRAC] θ=%.2f alu=(%f,%f) xrac=(%f,%f)\n",
                       th, r_alu.x, r_alu.y, r_xrac.x, r_xrac.y);
                mismatches++;
            }
        }
    }
    CHECK("three-way rotate equivalence across 15 angles", mismatches == 0);

    /* ── 4. Pure RV32I control flow — BEQ loop that counts to 16 ───── */
    HEADER("4. RV32I control flow: BEQ loop → 16 MICRO iterations");
    {
        /*
         * Hand-assembled program:
         *
         *   addi x1, x0, 16        ; target count
         *   addi x2, x0, 0         ; counter = 0
         *   rac.clr_acc
         *   rac.setmode CIRC, ROT
         * loop:
         *   rac.micro i=0          ; iter counter kept inside ALU, so
         *                           the encoding's "iter" field isn't
         *                           strictly required — ALU auto-advances.
         *                           We emit i=0 each time to keep the
         *                           program compact.
         *   addi x2, x2, 1
         *   beq  x1, x2, done      ; if counter == 16, exit loop
         *   jal  x0, loop          ; else jump back
         * done:
         *   rac.ret
         *   ebreak
         *
         * Goal: verify the RV32I subset is correctly decoded. The ALU
         * should end in iter=16.
         */
        uint32_t prog[32];
        int i = 0;
        prog[i++] = rac_xrac_enc_addi(1, 0, 16);    /* x1 = 16 */
        prog[i++] = rac_xrac_enc_addi(2, 0, 0);     /* x2 = 0  */
        prog[i++] = rac_xrac_enc_clr_acc();
        prog[i++] = rac_xrac_enc_setmode(RAC_ALU_MODE_CIRCULAR,
                                         RAC_ALU_DIR_ROTATION);
        int loop_pc = i;
        prog[i++] = rac_xrac_enc_micro(0);
        prog[i++] = rac_xrac_enc_addi(2, 2, 1);     /* x2++ */
        /* beq x1, x2, +8  (skip the JAL if equal → fall through to done) */
        prog[i++] = rac_xrac_enc_beq(1, 2, 8);
        /* jal x0, -offset_to_loop */
        int jal_pc = i;
        int jal_offset = (loop_pc - jal_pc) * 4;
        prog[i++] = rac_xrac_enc_jal(0, jal_offset);
        prog[i++] = rac_xrac_enc_ret();
        prog[i++] = rac_xrac_enc_ebreak();

        rac_xrac_cpu cpu;
        rac_xrac_init(&cpu, prog, (size_t)i);
        rac_alu_load(&cpu.alu, 1.0f * RAC_ALU_K_INV,
                              0.0f * RAC_ALU_K_INV, 0.3f);
        cpu.alu.mode = RAC_ALU_MODE_CIRCULAR;
        cpu.alu.dir  = RAC_ALU_DIR_ROTATION;

        uint64_t cyc = rac_xrac_run(&cpu, 10000);
        printf("  retired %llu cycles, x1=%u x2=%u iter=%d\n",
               (unsigned long long)cyc, cpu.xreg[1], cpu.xreg[2], cpu.alu.iter);
        CHECK("x1 == 16",       cpu.xreg[1] == 16);
        CHECK("x2 == 16",       cpu.xreg[2] == 16);
        CHECK("iter == 16",     cpu.alu.iter == RAC_ALU_ITERS);
        CHECK("halted cleanly", cpu.halted == 1);
    }

    /* ── 5. Trace one short program ─────────────────────────────────── */
    HEADER("5. Traced run (visual inspection)");
    {
        uint32_t prog[8];
        int n = 0;
        prog[n++] = rac_xrac_enc_clr_acc();
        prog[n++] = rac_xrac_enc_setmode(RAC_ALU_MODE_CIRCULAR,
                                         RAC_ALU_DIR_ROTATION);
        prog[n++] = rac_xrac_enc_micro(0);
        prog[n++] = rac_xrac_enc_micro(1);
        prog[n++] = rac_xrac_enc_ret();

        rac_xrac_cpu cpu;
        rac_xrac_init(&cpu, prog, (size_t)n);
        rac_alu_load(&cpu.alu, 1.0f, 0.0f, 0.5f);
        cpu.trace = 1;
        (void)rac_xrac_run(&cpu, 32);
        cpu.trace = 0;
        CHECK("trace ran to halt", cpu.halted);
    }

    printf("\npassed=%d failed=%d\n", passed, failed);
    printf("%s\n", failed == 0 ? "ALL XRAC ISS BVT PASSED"
                               : "XRAC ISS FAILURES");
    return failed == 0 ? 0 : 1;
}
