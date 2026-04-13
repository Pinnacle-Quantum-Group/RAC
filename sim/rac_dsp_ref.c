/*
 * rac_dsp_ref.c — bit-accurate C reference model for rac_dsp.v
 * Pinnacle Quantum Group — April 2026
 *
 * Mirrors the RTL's Q32.32 arithmetic exactly: same coarse-LUT read,
 * same 10-stage combinational shift-add chain, same 6 residual CORDIC
 * iterations, same rounding (truncation toward −∞ via arithmetic shift).
 *
 * Consumed by cosim.py as an alternative golden for debugging — it lets
 * you diff (RTL output) vs (bit-accurate reference) to isolate whether
 * a mismatch is an RTL bug or an algorithmic one.
 *
 * Build:
 *     cc -O2 -Wall -Wextra -o rac_dsp_ref rac_dsp_ref.c
 * Run:
 *     ./rac_dsp_ref test_vectors.hex cordic_coarse_lut.mem cordic_atanh.mem \
 *                   > ref_outputs.hex
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WIDTH          64
#define LUT_BITS       10
#define RESIDUAL       9          /* iters in residual stage */
#define RESIDUAL_START 8          /* first residual shift (overlaps coarse by 2) */
#define LUT_SIZE       (1 << LUT_BITS)

/*
 * Why RESIDUAL_START = 8 (overlap coarse by 2) instead of LUT_BITS = 10:
 *
 * Worst-case residual input |z_res| ≤ bucket_half_width + coarse_leftover
 *                                   = π/(2·LUT_SIZE) + atan(2^-(LUT_BITS-1))
 *                                   ≈ 1.53e-3 + 1.95e-3 = 3.49e-3
 *
 * CORDIC convergence requires the residual's Σatan(2^-i) for i ≥ START
 * to exceed this bound. Starting at shift 8:
 *   Σ atan(2^-8..2^-15) ≈ 7.7e-3   — plenty of margin.
 * Starting at shift 10 (no overlap):
 *   Σ atan(2^-10..2^-15) ≈ 1.95e-3 — FAILS to converge.
 *
 * The overlap "costs" 2 extra pipeline stages (RESIDUAL went 6 → 8) but
 * buys guaranteed convergence for any input in the folded range.
 */

typedef int64_t  q_t;
typedef uint64_t qu_t;

/* Arithmetic right shift of signed 64-bit (C's >> on signed is
 * implementation-defined before C99; all modern compilers sign-extend
 * but we're explicit). */
static inline q_t asr(q_t v, int n) {
    if (v >= 0) return (q_t)((qu_t)v >> n);
    return ~(((qu_t)~v) >> n);
}

/*
 * z is now SIGNED Q0.63 fraction-of-π:
 *   z_signed / 2^63 = θ / π
 * Full signed range [-2^63, +2^63) ↔ [-π, +π).
 * Quadrant fold: flip iff z[63] XOR z[62] == 1 (angle in Q2 or Q3);
 * fold operation is XOR with 2^63 (toggle MSB, shifts θ by ±π).
 *
 * x/y remain Q32.32 signed (Cartesian coordinates, not angles).
 */
#define SIGN_BIT_64  ((q_t)0x8000000000000000LL)

/* Quadrant fold (Q0.63 fraction-of-π):
 *   Condition: z[63] XOR z[62] == 1 (angle in quadrant 2 or 3)
 *   Action:    z XOR 2^63 (toggle MSB → shifts θ by ±π, auto-wraps)
 *              Flip (x, y) sign simultaneously. */
static void quadrant_fold(q_t *x, q_t *y, q_t *z) {
    uint64_t uz = (uint64_t)*z;
    int msb    = (int)((uz >> 63) & 1u);
    int next   = (int)((uz >> 62) & 1u);
    if (msb ^ next) {
        *z = (q_t)(uz ^ (uint64_t)SIGN_BIT_64);
        *x = -*x;
        *y = -*y;
    }
}

/* LUT index (pure bit slice, no libm, no multiply):
 *   z_folded is Q0.63 fraction-of-π, in (-2^62, 2^62].
 *   Shift left 1 bit → z_scaled spans (-2^63, 2^63], top LUT_BITS
 *   give a signed [-LUT_SIZE/2, LUT_SIZE/2-1]. Bias by LUT_SIZE/2
 *   for unsigned [0, LUT_SIZE). */
static int lut_idx_of(q_t z_folded) {
    q_t z_scaled = z_folded << 1;
    /* Extract top LUT_BITS as signed int: sign-extending arithmetic shift. */
    int idx_signed = (int)(asr(z_scaled, WIDTH - LUT_BITS));
    int idx = (idx_signed + LUT_SIZE / 2) & (LUT_SIZE - 1);
    return idx;
}

/* Load a hex-format LUT from $readmemh-style file.
 * Lines starting with "//" are comments; otherwise each non-empty line
 * is parsed as a hex integer. */
static int load_hex_lut(const char *path, uint64_t *out, int expected) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return -1; }
    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || (*p == '/' && p[1] == '/')) continue;
        uint64_t v = 0;
        if (sscanf(p, "%" SCNx64, &v) != 1) continue;
        if (n < expected) out[n++] = v;
    }
    fclose(f);
    return n;
}

/* Globals loaded at startup */
static uint64_t coarse_lut[LUT_SIZE];
#define ATAN_ROM_SIZE (RESIDUAL_START + RESIDUAL)
static q_t      atan_rom [ATAN_ROM_SIZE];    /* Q0.63 fraction-of-π */
static q_t      atanh_tab[RESIDUAL];         /* Q0.63 fraction-of-π */

/* One rac_dsp CGLUT evaluation at full precision.
 *
 * Flow:
 *   1. Quadrant fold: reduce z to (-π/2, π/2], flipping (x,y) if needed
 *   2. Coarse stage: look up LUT_BITS direction bits for the bucket
 *      containing z, apply them as shift-adds to (x,y), track the
 *      cumulative rotation angle z_applied
 *   3. Residual: dynamic CORDIC at shifts RESIDUAL_START..RESIDUAL_START+RESIDUAL-1
 *      driving (z_input_folded - z_applied) toward zero
 */
static void dsp_eval(q_t x_in, q_t y_in, q_t z_in, int op,
                     q_t *x_out, q_t *y_out, q_t *z_out) {
    q_t x = x_in, y = y_in, z = z_in;
    quadrant_fold(&x, &y, &z);

    int hyperbolic = (op == 3) ? 1 : 0;
    int vectoring  = (op == 2) ? 1 : 0;

    /* ── Coarse stage ──
     * Look up direction bits for the bucket containing z. Apply them
     * to (x,y) while tracking z_applied = sum of ±atan(2^-i) over the
     * LUT_BITS directions. Skip for vectoring mode (which computes
     * direction from y sign dynamically, not from a precomputed LUT). */
    q_t z_applied = 0;
    if (!vectoring) {
        int idx = lut_idx_of(z);
        uint64_t d_vec = coarse_lut[idx];
        for (int i = 0; i < LUT_BITS; i++) {
            int d_pos = ((d_vec >> i) & 1u) == 1u;
            q_t y_sh  = asr(y, i);
            q_t x_sh  = asr(x, i);
            q_t atan_i = atan_rom[i];       /* Q0.63 fraction-of-π */
            if (d_pos) {
                q_t xn = x - y_sh;
                q_t yn = y + x_sh;
                x = xn; y = yn;
                z_applied += atan_i;
            } else {
                q_t xn = x + y_sh;
                q_t yn = y - x_sh;
                x = xn; y = yn;
                z_applied -= atan_i;
            }
        }
        z = z - z_applied;   /* residual = input - what coarse actually did */
    }

    /* ── Residual stage ──
     * RESIDUAL iterations at shifts starting at RESIDUAL_START (may
     * overlap coarse range by a few iters for convergence margin). */
    for (int i = 0; i < RESIDUAL; i++) {
        int phys = RESIDUAL_START + i;
        int d_positive = vectoring ? (y < 0) : (z >= 0);
        q_t x_sh = asr(x, phys);
        q_t y_sh = asr(y, phys);
        q_t atan_const =
            hyperbolic ? atanh_tab[i] : atan_rom[phys];

        q_t x_next, y_next, z_next;
        if (hyperbolic) {
            x_next = d_positive ? x + y_sh : x - y_sh;
        } else {
            x_next = d_positive ? x - y_sh : x + y_sh;
        }
        y_next = d_positive ? y + x_sh : y - x_sh;
        z_next = d_positive ? z - atan_const : z + atan_const;
        x = x_next; y = y_next; z = z_next;
    }

    *x_out = x; *y_out = y; *z_out = z;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "usage: %s test_vectors.hex coarse_lut.mem atan.mem atanh.mem\n",
            argv[0]);
        return 1;
    }
    if (load_hex_lut(argv[2], coarse_lut, LUT_SIZE) < 0) return 2;
    uint64_t atan_u[ATAN_ROM_SIZE];
    if (load_hex_lut(argv[3], atan_u, ATAN_ROM_SIZE) < 0) return 3;
    for (int i = 0; i < ATAN_ROM_SIZE; i++) atan_rom[i] = (q_t)atan_u[i];
    uint64_t atanh_u[RESIDUAL];
    if (load_hex_lut(argv[4], atanh_u, RESIDUAL) < 0) return 4;
    for (int i = 0; i < RESIDUAL; i++) atanh_tab[i] = (q_t)atanh_u[i];

    FILE *vf = fopen(argv[1], "r");
    if (!vf) { fprintf(stderr, "open %s: %s\n", argv[1], strerror(errno)); return 4; }

    printf("// rac_dsp C reference outputs\n");
    printf("// format: x_q3232 y_q3232 z_q3232\n");

    char line[512];
    while (fgets(line, sizeof(line), vf)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || (*p == '/' && p[1] == '/')) continue;
        unsigned op;
        uint64_t xh, yh, zh;
        if (sscanf(p, "%x %" SCNx64 " %" SCNx64 " %" SCNx64,
                   &op, &xh, &yh, &zh) != 4) continue;
        q_t xo, yo, zo;
        dsp_eval((q_t)xh, (q_t)yh, (q_t)zh, (int)(op & 7), &xo, &yo, &zo);
        printf("%016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
               (uint64_t)xo, (uint64_t)yo, (uint64_t)zo);
    }
    fclose(vf);
    return 0;
}
