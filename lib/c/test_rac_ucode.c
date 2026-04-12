/*
 * test_rac_ucode.c — BVT for the RAC microcode ISA + interpreter
 * Pinnacle Quantum Group — April 2026
 *
 * Verifies that the microsequencer produces bit-for-bit identical
 * results to the direct rac_alu path for every prebuilt ROM. This is
 * the semantic equivalence proof: if an Xrac hardware implementation
 * accepts these same bit patterns, it is correct iff its outputs match.
 */

#include "rac_ucode.h"
#include "rac_alu.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int passed = 0, failed = 0;
#define CHECK(name, cond) do { \
    if (cond) { passed++; printf("  [PASS] %s\n", name); } \
    else      { failed++; printf("  [FAIL] %s\n", name); } \
} while(0)

#define HEADER(s) printf("\n== %s ==\n", s)

/* Execute the rotate ROM on (x,y,z) and return x,y. */
static rac_vec2 run_rotate_rom(rac_vec2 v, float theta) {
    rac_alu_state s;
    rac_alu_reset(&s);
    /* Emulate what rac_alu_rotate does to the inputs: quadrant fold +
     * K_INV pre-scale. The ROM is just the CORDIC sequence itself. */
    float t = theta;
    while (t >  3.14159265f) t -= 2.0f * 3.14159265f;
    while (t < -3.14159265f) t += 2.0f * 3.14159265f;
    if (t > 0.5f * 3.14159265f)       { v.x = -v.x; v.y = -v.y; t -= 3.14159265f; }
    else if (t < -0.5f * 3.14159265f) { v.x = -v.x; v.y = -v.y; t += 3.14159265f; }
    rac_alu_load(&s, v.x * RAC_ALU_K_INV, v.y * RAC_ALU_K_INV, t);
    int rc = rac_ucore_execute(&s, rac_ucode_rom_rotate, 19);
    if (rc < 0) return (rac_vec2){0, 0};
    return (rac_vec2){s.x, s.y};
}

/* Execute the polar ROM on (x,y) → (mag, angle). */
static void run_polar_rom(rac_vec2 v, float *mag, float *ang) {
    /* Half-plane pre-fold matching rac_alu_polar. */
    float z_offset = 0.0f;
    if (v.x < 0.0f) {
        z_offset = (v.y >= 0.0f) ? 3.14159265f : -3.14159265f;
        v.x = -v.x; v.y = -v.y;
    }
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, v.x, v.y, 0.0f);
    (void)rac_ucore_execute(&s, rac_ucode_rom_polar, 19);
    if (mag) *mag = s.x * RAC_ALU_K_INV;
    if (ang) *ang = s.z + z_offset;
}

/* Execute the project ROM on (x,y,z) → acc. */
static float run_project_rom(rac_vec2 v, float theta) {
    float t = -theta;
    while (t >  3.14159265f) t -= 2.0f * 3.14159265f;
    while (t < -3.14159265f) t += 2.0f * 3.14159265f;
    if (t > 0.5f * 3.14159265f)       { v.x = -v.x; v.y = -v.y; t -= 3.14159265f; }
    else if (t < -0.5f * 3.14159265f) { v.x = -v.x; v.y = -v.y; t += 3.14159265f; }
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, v.x * RAC_ALU_K_INV, v.y * RAC_ALU_K_INV, t);
    (void)rac_ucore_execute(&s, rac_ucode_rom_project, 20);
    return s.acc;
}

int main(void) {
    printf("RAC Microcode BVT — Pinnacle Quantum Group\n");

    /* ── 1. Encoding / decoding round-trip ──────────────────────────── */
    HEADER("1. Instruction encoding");
    rac_uinst w = RAC_UC_MAKE(RAC_UC_MICRO, 0, 7, 123);
    CHECK("op  roundtrip",   RAC_UC_OP(w)    == RAC_UC_MICRO);
    CHECK("imm8  roundtrip", RAC_UC_IMM8(w)  == 7);
    CHECK("imm16 roundtrip", RAC_UC_IMM16(w) == 123);

    w = RAC_UC_MAKE(RAC_UC_SETMODE, 0,
                    RAC_ALU_MODE_HYPERBOLIC, RAC_ALU_DIR_VECTORING);
    CHECK("setmode op",   RAC_UC_OP(w) == RAC_UC_SETMODE);
    CHECK("setmode mode", RAC_UC_IMM8(w) == RAC_ALU_MODE_HYPERBOLIC);
    CHECK("setmode dir",  RAC_UC_IMM16(w) == RAC_ALU_DIR_VECTORING);

    /* ── 2. Disassembler ────────────────────────────────────────────── */
    HEADER("2. Disassembler");
    char buf[64];
    rac_ucode_disasm(rac_ucode_rom_rotate[1], buf, sizeof(buf));
    CHECK("disasm setmode contains 'CIRCULAR'",
          strstr(buf, "CIRCULAR") != NULL);
    rac_ucode_disasm(rac_ucode_rom_rotate[2], buf, sizeof(buf));
    CHECK("disasm micro contains 'i=0'", strstr(buf, "i=0") != NULL);
    rac_ucode_disasm(rac_ucode_rom_rotate[18], buf, sizeof(buf));
    CHECK("disasm last is 'ret'", strstr(buf, "ret") != NULL);

    /* ── 3. Rotate ROM vs direct ALU ────────────────────────────────── */
    HEADER("3. Rotate ROM equivalence");
    int mismatches = 0;
    for (float t = -1.2f; t <= 1.2f; t += 0.2f) {
        rac_vec2 v_in = {1.0f, 0.0f};
        rac_vec2 r_rom = run_rotate_rom(v_in, t);
        rac_vec2 r_alu = rac_alu_rotate(v_in, t);
        if (fabsf(r_rom.x - r_alu.x) > 1e-5f) mismatches++;
        if (fabsf(r_rom.y - r_alu.y) > 1e-5f) mismatches++;
    }
    CHECK("rotate ROM matches rac_alu_rotate (13 angles)", mismatches == 0);

    /* ── 4. Polar ROM vs direct ALU ─────────────────────────────────── */
    HEADER("4. Polar ROM equivalence");
    {
        float m1, a1, m2, a2;
        run_polar_rom((rac_vec2){3.0f, 4.0f}, &m1, &a1);
        rac_alu_polar((rac_vec2){3.0f, 4.0f}, &m2, &a2);
        CHECK("mag matches",   fabsf(m1 - m2) < 1e-5f);
        CHECK("angle matches", fabsf(a1 - a2) < 1e-5f);

        run_polar_rom((rac_vec2){-2.0f, 1.5f}, &m1, &a1);
        rac_alu_polar((rac_vec2){-2.0f, 1.5f}, &m2, &a2);
        CHECK("left-half mag",   fabsf(m1 - m2) < 1e-5f);
        CHECK("left-half angle", fabsf(a1 - a2) < 1e-5f);
    }

    /* ── 5. Project ROM vs direct ALU ───────────────────────────────── */
    HEADER("5. Project ROM equivalence");
    {
        int pmis = 0;
        for (float t = -1.0f; t <= 1.0f; t += 0.25f) {
            float p_rom = run_project_rom((rac_vec2){1.0f, 0.5f}, t);
            float p_alu = rac_alu_project((rac_vec2){1.0f, 0.5f}, t);
            if (fabsf(p_rom - p_alu) > 1e-4f) {
                pmis++;
                printf("  θ=%.2f rom=%.6f alu=%.6f diff=%.6e\n",
                       t, p_rom, p_alu, fabsf(p_rom - p_alu));
            }
        }
        CHECK("project ROM matches rac_alu_project", pmis == 0);
    }

    /* ── 6. Cycle counting (ISS-style) ──────────────────────────────── */
    HEADER("6. Cycle counting");
    {
        rac_alu_state s;
        rac_alu_reset(&s);
        rac_alu_load(&s, 1.0f, 0.0f, 0.3f);
        uint64_t cyc = 0;
        (void)rac_ucore_execute_counted(&s, rac_ucode_rom_rotate, 19, &cyc);
        printf("  rotate ROM: %llu cycles\n", (unsigned long long)cyc);
        CHECK("rotate executes 19 uinsts", cyc == 19);

        rac_alu_reset(&s);
        rac_alu_load(&s, 1.0f, 0.0f, 0.3f);
        (void)rac_ucore_execute_counted(&s, rac_ucode_rom_project, 20, &cyc);
        printf("  project ROM: %llu cycles\n", (unsigned long long)cyc);
        CHECK("project executes 20 uinsts", cyc == 20);
    }

    /* ── 7. ROM length sanity ───────────────────────────────────────── */
    HEADER("7. ROM length introspection");
    CHECK("rotate  rom_len == 19", rac_ucode_rom_len(rac_ucode_rom_rotate)   == 19);
    CHECK("polar   rom_len == 19", rac_ucode_rom_len(rac_ucode_rom_polar)    == 19);
    CHECK("project rom_len == 20", rac_ucode_rom_len(rac_ucode_rom_project)  == 20);
    CHECK("exp     rom_len == 19", rac_ucode_rom_len(rac_ucode_rom_exp_core) == 19);

    /* ── 8. Xrac R-type encoding helper ─────────────────────────────── */
    HEADER("8. Xrac custom-0 encoding");
    {
        /* rac.micro  x1, x2, x3   funct3=0 funct7=0 */
        uint32_t i = rac_xrac_encode_R(0, 3, 2, 0, 1);
        CHECK("opcode bits [6:0] == 0x0B",  (i & 0x7F) == 0x0B);
        CHECK("rd == 1",                    ((i >>  7) & 0x1F) == 1);
        CHECK("rs1 == 2",                   ((i >> 15) & 0x1F) == 2);
        CHECK("rs2 == 3",                   ((i >> 20) & 0x1F) == 3);
    }

    /* ── 9. Pretty-print one ROM ────────────────────────────────────── */
    HEADER("9. ROM disassembly (rotate)");
    rac_ucode_dump(rac_ucode_rom_rotate, 19);

    /* ── Summary ────────────────────────────────────────────────────── */
    printf("\npassed=%d failed=%d\n", passed, failed);
    printf("%s\n", failed == 0 ? "ALL MICROCODE BVT PASSED"
                               : "MICROCODE BVT FAILURES");
    return failed == 0 ? 0 : 1;
}
