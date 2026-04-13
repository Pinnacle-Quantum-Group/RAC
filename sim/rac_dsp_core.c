/*
 * rac_dsp_core.c — shared RAC-DSP bit-exact math engine
 *
 * Implementation lives here; rac_dsp_ref.c and rac_systolic_ref.c both
 * link against this for the actual CORDIC evaluation. Moving the code
 * here means any bug fix is picked up by both reference binaries.
 *
 * Function names are rac_*-prefixed (not static) so they're visible to
 * the linker. Globals (rac_coarse_lut, rac_atan_rom, rac_atanh_tab)
 * are defined here and extern'd in rac_dsp_core.h.
 */

#define _USE_MATH_DEFINES
#include "rac_dsp_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>

/* ── Globals ───────────────────────────────────────────────────────── */

uint64_t rac_coarse_lut[RAC_DSP_LUT_SIZE];
rac_q_t  rac_atan_rom [RAC_DSP_ATAN_ROM_SIZE];
rac_q_t  rac_atanh_tab[RAC_DSP_RESIDUAL];

/* ── Arithmetic right shift of signed 64-bit ──────────────────────── */

static inline rac_q_t asr(rac_q_t v, int n) {
    if (v >= 0) return (rac_q_t)((uint64_t)v >> n);
    return ~(((uint64_t)~v) >> n);
}

/* ── Q0.63 quadrant fold ──────────────────────────────────────────── */

#define SIGN_BIT_64  ((rac_q_t)0x8000000000000000LL)

static void quadrant_fold(rac_q_t *x, rac_q_t *y, rac_q_t *z) {
    uint64_t uz = (uint64_t)*z;
    int msb    = (int)((uz >> 63) & 1u);
    int next   = (int)((uz >> 62) & 1u);
    if (msb ^ next) {
        *z = (rac_q_t)(uz ^ (uint64_t)SIGN_BIT_64);
        *x = -*x;
        *y = -*y;
    }
}

/* ── LUT index ────────────────────────────────────────────────────── */

static int lut_idx_of(rac_q_t z_folded) {
    rac_q_t z_scaled = z_folded << 1;
    int idx_signed = (int)(asr(z_scaled, RAC_DSP_WIDTH - RAC_DSP_LUT_BITS));
    int idx = (idx_signed + RAC_DSP_LUT_SIZE / 2) & (RAC_DSP_LUT_SIZE - 1);
    return idx;
}

/* ── $readmemh loader ─────────────────────────────────────────────── */

int rac_load_hex_lut(const char *path, uint64_t *out, int expected) {
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

int rac_load_all_roms(const char *coarse_lut_path,
                      const char *atan_path,
                      const char *atanh_path) {
    if (rac_load_hex_lut(coarse_lut_path, rac_coarse_lut,
                         RAC_DSP_LUT_SIZE) < 0) return 1;
    uint64_t atan_u[RAC_DSP_ATAN_ROM_SIZE];
    if (rac_load_hex_lut(atan_path, atan_u, RAC_DSP_ATAN_ROM_SIZE) < 0) return 2;
    for (int i = 0; i < RAC_DSP_ATAN_ROM_SIZE; i++)
        rac_atan_rom[i] = (rac_q_t)atan_u[i];
    uint64_t atanh_u[RAC_DSP_RESIDUAL];
    if (rac_load_hex_lut(atanh_path, atanh_u, RAC_DSP_RESIDUAL) < 0) return 3;
    for (int i = 0; i < RAC_DSP_RESIDUAL; i++)
        rac_atanh_tab[i] = (rac_q_t)atanh_u[i];
    return 0;
}

/* ── One rac_dsp CGLUT evaluation at full precision ──────────────── */

void rac_dsp_eval(rac_q_t x_in, rac_q_t y_in, rac_q_t z_in, int op,
                  rac_q_t *x_out, rac_q_t *y_out, rac_q_t *z_out) {
    rac_q_t x = x_in, y = y_in, z = z_in;
    quadrant_fold(&x, &y, &z);

    int hyperbolic = (op == 3) ? 1 : 0;
    int vectoring  = (op == 2) ? 1 : 0;

    /* Coarse stage: LUT-driven direction bits, z_applied tracking. */
    rac_q_t z_applied = 0;
    if (!vectoring) {
        int idx = lut_idx_of(z);
        uint64_t d_vec = rac_coarse_lut[idx];
        for (int i = 0; i < RAC_DSP_LUT_BITS; i++) {
            int d_pos = ((d_vec >> i) & 1u) == 1u;
            rac_q_t y_sh  = asr(y, i);
            rac_q_t x_sh  = asr(x, i);
            rac_q_t atan_i = rac_atan_rom[i];
            if (d_pos) {
                rac_q_t xn = x - y_sh;
                rac_q_t yn = y + x_sh;
                x = xn; y = yn;
                z_applied += atan_i;
            } else {
                rac_q_t xn = x + y_sh;
                rac_q_t yn = y - x_sh;
                x = xn; y = yn;
                z_applied -= atan_i;
            }
        }
        z = z - z_applied;
    }

    /* Residual: RESIDUAL iters at shifts starting at RESIDUAL_START. */
    for (int i = 0; i < RAC_DSP_RESIDUAL; i++) {
        int phys = RAC_DSP_RESIDUAL_START + i;
        int d_positive = vectoring ? (y < 0) : (z >= 0);
        rac_q_t x_sh = asr(x, phys);
        rac_q_t y_sh = asr(y, phys);
        rac_q_t atan_const =
            hyperbolic ? rac_atanh_tab[i] : rac_atan_rom[phys];

        rac_q_t x_next, y_next, z_next;
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

/* ── Batch helpers for efficient benchmarking ─────────────────────── */

void rac_dsp_project_batch(int n,
                           const rac_q_t *xs, const rac_q_t *zs,
                           rac_q_t       *x_out) {
    rac_q_t xo, yo, zo;
    for (int i = 0; i < n; i++) {
        rac_dsp_eval(xs[i], 0, zs[i], /*project*/ 1, &xo, &yo, &zo);
        x_out[i] = xo;
    }
}

rac_q_t rac_dsp_project_sum(int n,
                            const rac_q_t *xs, const rac_q_t *zs) {
    rac_q_t sum = 0;
    rac_q_t xo, yo, zo;
    for (int i = 0; i < n; i++) {
        rac_dsp_eval(xs[i], 0, zs[i], /*project*/ 1, &xo, &yo, &zo);
        sum += xo;
    }
    return sum;
}
