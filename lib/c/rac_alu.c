/*
 * rac_alu.c — RAC Adder + ALU implementation
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * See rac_alu.h for the opcode model. This file implements the ALU
 * state machine and re-expresses the RAC primitives strictly through
 * ALU opcodes (so the opcode set is demonstrably complete).
 *
 * Build: cc -O3 -march=native rac_alu.c rac_cpu.c -lm -c
 */

/* posix_memalign needs _POSIX_C_SOURCE >= 200112L. Define before any
 * system header so the prototype is visible under -std=c99. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif
#include "rac_alu.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define RAC_ALU_HAVE_AVX2 1
#else
  #define RAC_ALU_HAVE_AVX2 0
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define RAC_ALU_FORCE_INLINE static inline __attribute__((always_inline))
#else
  #define RAC_ALU_FORCE_INLINE static inline
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

/* ── CORDIC lookup tables (copied here so this TU is self-contained) ─────── */

static const float _alu_atan_table[RAC_ALU_ITERS] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f
};

static const float _alu_atanh_table[RAC_ALU_ITERS] = {
    0.54930614f, 0.25541281f, 0.12565721f, 0.06258157f,
    0.03126017f, 0.01562627f, 0.00781265f, 0.00390626f,
    0.00195313f, 0.00097656f, 0.00048828f, 0.00024414f,
    0.00012207f, 0.00006104f, 0.00003052f, 0.00001526f
};

/* Hyperbolic CORDIC iteration indices with repeats at 4 and 13
 * (standard Walther sequence). Values here are the physical iteration
 * index i used to form 2^-i, starting at 1. */
static const int _alu_hyp_iter_map[RAC_ALU_ITERS] = {
    1,2,3,4,4,5,6,7,8,9,10,11,12,13,13,14
};

/* Precomputed 2^-i tables keyed by iteration counter. Circular CORDIC
 * uses i directly; hyperbolic uses iter_map[i]. Dodges the _alu_shift
 * loop on the hot path. */
static const float _alu_circ_scale[RAC_ALU_ITERS] = {
    1.0f, 0.5f, 0.25f, 0.125f,
    0.0625f, 0.03125f, 0.015625f, 0.0078125f,
    0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f,
    0.000244140625f, 0.0001220703125f, 0.00006103515625f, 0.000030517578125f
};

static const float _alu_hyp_scale[RAC_ALU_ITERS] = {
    0.5f,         0.25f,        0.125f,        0.0625f,
    0.0625f,      0.03125f,     0.015625f,     0.0078125f,
    0.00390625f,  0.001953125f, 0.0009765625f, 0.00048828125f,
    0.000244140625f, 0.0001220703125f, 0.0001220703125f, 0.00006103515625f
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */

#ifndef RAC_ALU_PI
#define RAC_ALU_PI 3.14159265358979f
#endif

static inline float _alu_shift(float v, int i) {
    /* 2^-i applied as a multiply; in hardware this is a barrel shift. */
    float s = 1.0f;
    for (int k = 0; k < i; k++) s *= 0.5f;
    return v * s;
}

/* Quadrant folding: reduce theta to (-π/2, π/2] for circular CORDIC.
 * CORDIC circular rotation converges only for |theta| ≲ π/2 (sum of
 * atan(2^-i) entries). For any theta outside that range we fold by
 * negating the input vector (a rotation by π costs nothing) and
 * shifting theta by ±π. Hardware cost: a pair of sign inverters.
 * Returns folded (v, theta) pair via pointers. */
static inline void _alu_quadrant_fold(rac_vec2 *v, float *theta) {
    float t = *theta;
    /* normalize to (-π, π] */
    while (t >  RAC_ALU_PI) t -= 2.0f * RAC_ALU_PI;
    while (t < -RAC_ALU_PI) t += 2.0f * RAC_ALU_PI;
    if (t > 0.5f * RAC_ALU_PI) {
        v->x = -v->x; v->y = -v->y;
        t -= RAC_ALU_PI;
    } else if (t < -0.5f * RAC_ALU_PI) {
        v->x = -v->x; v->y = -v->y;
        t += RAC_ALU_PI;
    }
    *theta = t;
}

static inline float _alu_sign_z(float z) { return (z >= 0.0f) ?  1.0f : -1.0f; }
static inline float _alu_sign_y(float y) { return (y <  0.0f) ?  1.0f : -1.0f; }

static inline float _alu_powf(float base, int n) {
    float r = 1.0f;
    for (int i = 0; i < n; i++) r *= base;
    return r;
}

/* Map iteration counter → physical shift index and atan[h] table lookup.
 * For hyperbolic mode we honour the standard Walther repeat at 4 and 13. */
static inline int _alu_phys_index(const rac_alu_state *s) {
    if (s->mode == RAC_ALU_MODE_HYPERBOLIC) {
        return _alu_hyp_iter_map[s->iter];
    }
    return s->iter + 1;   /* circular/linear use 2^-i with i starting at 0 */
}

static inline float _alu_table_value(const rac_alu_state *s) {
    int idx = s->iter;
    switch (s->mode) {
        case RAC_ALU_MODE_HYPERBOLIC:
            return _alu_atanh_table[_alu_hyp_iter_map[idx] - 1];
        case RAC_ALU_MODE_LINEAR:
            /* linear CORDIC: z residual is 2^-i (directly) */
            return _alu_circ_scale[idx] * 0.5f;   /* z residual = 2^-(i+1) */
        case RAC_ALU_MODE_CIRCULAR:
        default:
            return _alu_atan_table[idx];
    }
}

static inline float _alu_scale_value(const rac_alu_state *s) {
    if (s->mode == RAC_ALU_MODE_HYPERBOLIC) return _alu_hyp_scale[s->iter];
    if (s->mode == RAC_ALU_MODE_CIRCULAR)   return _alu_circ_scale[s->iter];
    return _alu_circ_scale[s->iter];  /* linear: same 2^-i */
}

/* ── Low-level ALU ops ───────────────────────────────────────────────────── */

void rac_alu_reset(rac_alu_state *s) {
    s->x = 0.0f; s->y = 0.0f; s->z = 0.0f;
    s->acc = 0.0f;
    s->iter = 0;
    s->chain = 0;
    s->mode = RAC_ALU_MODE_CIRCULAR;
    s->dir  = RAC_ALU_DIR_ROTATION;
    s->d = 0.0f;
}

void rac_alu_load(rac_alu_state *s, float x, float y, float z) {
    s->x = x; s->y = y; s->z = z;
    s->iter = 0;
}

void rac_alu_clear_acc(rac_alu_state *s) {
    s->acc = 0.0f;
}

void rac_alu_set_mode(rac_alu_state *s,
                      rac_alu_mode mode,
                      rac_alu_direction dir) {
    s->mode = mode;
    s->dir  = dir;
    s->iter = 0;
}

float rac_alu_sign_decide(rac_alu_state *s) {
    s->d = (s->dir == RAC_ALU_DIR_ROTATION) ? _alu_sign_z(s->z)
                                            : _alu_sign_y(s->y);
    return s->d;
}

int rac_alu_micro_step(rac_alu_state *s) {
    if (s->iter >= RAC_ALU_ITERS) return -1;

    (void)rac_alu_sign_decide(s);
    const float d     = s->d;
    const float scale = _alu_scale_value(s);
    const float ys    = s->y * scale;
    const float xs    = s->x * scale;
    const float tbl   = _alu_table_value(s);

    float xn, yn, zn;
    switch (s->mode) {
        case RAC_ALU_MODE_HYPERBOLIC:
            /* hyperbolic: x' = x + d*y*2^-i,  y' = y + d*x*2^-i */
            xn = s->x + d * ys;
            yn = s->y + d * xs;
            zn = s->z - d * tbl;
            break;
        case RAC_ALU_MODE_LINEAR:
            /* linear: y' = y + d*x*2^-i, x unchanged, z' = z - d*2^-i */
            xn = s->x;
            yn = s->y + d * xs;
            zn = s->z - d * tbl;
            break;
        case RAC_ALU_MODE_CIRCULAR:
        default:
            /* circular: x' = x - d*y*2^-i, y' = y + d*x*2^-i */
            xn = s->x - d * ys;
            yn = s->y + d * xs;
            zn = s->z - d * tbl;
            break;
    }
    s->x = xn; s->y = yn; s->z = zn;
    s->iter++;
    s->chain++;
    return 0;
}

/* Branchless sign: d = copysign(1, v). Compiles to a single andnps+orps pair
 * on x86 (sign-bit extraction + OR with 1.0). */
RAC_ALU_FORCE_INLINE float _alu_dir_rot(float z) { return copysignf(1.0f, z); }
RAC_ALU_FORCE_INLINE float _alu_dir_vec(float y) { return copysignf(1.0f, -y); }

/* Specialized straight-line loops for each mode. The mode switch inside
 * rac_alu_micro_step is hoisted out of the loop — the compiler then fully
 * unrolls the CORDIC sequence since the table reads become constant loads.
 * d is computed branchlessly via copysign to avoid per-iter conditional
 * jumps; `d * scale` folds into one FMA per coordinate. */
RAC_ALU_FORCE_INLINE void _alu_run_circ_rot(rac_alu_state *s, int iters) {
    float x = s->x, y = s->y, z = s->z;
    for (int i = 0; i < iters; i++) {
        float d     = _alu_dir_rot(z);
        float ds    = d * _alu_circ_scale[i];
        float xn    = x - ds * y;          /* fnmadd */
        float yn    = y + ds * x;          /* fmadd  */
        z -= d * _alu_atan_table[i];       /* fnmadd */
        x = xn; y = yn;
    }
    s->x = x; s->y = y; s->z = z;
    s->d = _alu_dir_rot(z);
    s->iter  = iters;
    s->chain += iters;
}

RAC_ALU_FORCE_INLINE void _alu_run_circ_vec(rac_alu_state *s, int iters) {
    float x = s->x, y = s->y, z = s->z;
    for (int i = 0; i < iters; i++) {
        float d     = _alu_dir_vec(y);
        float ds    = d * _alu_circ_scale[i];
        float xn    = x - ds * y;
        float yn    = y + ds * x;
        z -= d * _alu_atan_table[i];
        x = xn; y = yn;
    }
    s->x = x; s->y = y; s->z = z;
    s->d = _alu_dir_vec(y);
    s->iter  = iters;
    s->chain += iters;
}

RAC_ALU_FORCE_INLINE void _alu_run_hyp_rot(rac_alu_state *s, int iters) {
    float x = s->x, y = s->y, z = s->z;
    for (int i = 0; i < iters; i++) {
        float d     = _alu_dir_rot(z);
        float ds    = d * _alu_hyp_scale[i];
        float xn    = x + ds * y;          /* hyperbolic: + not - */
        float yn    = y + ds * x;
        z -= d * _alu_atanh_table[_alu_hyp_iter_map[i] - 1];
        x = xn; y = yn;
    }
    s->x = x; s->y = y; s->z = z;
    s->d = _alu_dir_rot(z);
    s->iter  = iters;
    s->chain += iters;
}

int rac_alu_run(rac_alu_state *s, int iters) {
    if (iters > RAC_ALU_ITERS) iters = RAC_ALU_ITERS;
    s->iter = 0;

    if (s->mode == RAC_ALU_MODE_CIRCULAR && s->dir == RAC_ALU_DIR_ROTATION) {
        _alu_run_circ_rot(s, iters);
    } else if (s->mode == RAC_ALU_MODE_CIRCULAR && s->dir == RAC_ALU_DIR_VECTORING) {
        _alu_run_circ_vec(s, iters);
    } else if (s->mode == RAC_ALU_MODE_HYPERBOLIC && s->dir == RAC_ALU_DIR_ROTATION) {
        _alu_run_hyp_rot(s, iters);
    } else {
        /* Linear mode and hyperbolic-vectoring: fall through to the generic
         * micro-step path (used rarely, not perf-critical). */
        for (int i = 0; i < iters; i++) {
            if (rac_alu_micro_step(s) != 0) return -1;
        }
    }
    return 0;
}

void rac_alu_accum(rac_alu_state *s, float scale) {
    /* acc <- acc + x * scale.  This is the only scalar multiplier the
     * ALU needs — and even that folds out to a shift when scale is a
     * power of two or pre-baked gain constant. */
    s->acc += s->x * scale;
}

void rac_alu_compensate(rac_alu_state *s) {
    /* Circular CORDIC gain K ≈ 1.647 inflates magnitude per call; compensate
     * by K^-1 ≈ 0.607 (RAC_ALU_K_INV). Hyperbolic CORDIC gain K_hyp ≈ 0.828
     * deflates magnitude per call; compensate by 1/K_hyp ≈ 1.208. The macro
     * Use RAC_ALU_K_HYP_RECIP (= 1/K_HYP ≈ 1.207) to un-gain hyperbolic output. */
    float c;
    if (s->mode == RAC_ALU_MODE_HYPERBOLIC) {
        c = _alu_powf(RAC_ALU_K_HYP_RECIP, s->chain);
    } else {
        c = _alu_powf(RAC_ALU_K_INV, s->chain);
    }
    s->x *= c;
    s->y *= c;
    s->chain = 0;
}

int rac_alu_dispatch(rac_alu_state *s, rac_alu_opcode op,
                     float arg0, float arg1, float arg2) {
    switch (op) {
        case RAC_ALU_OP_LOAD:
            rac_alu_load(s, arg0, arg1, arg2);
            return 0;
        case RAC_ALU_OP_CLEAR_ACC:
            rac_alu_clear_acc(s);
            return 0;
        case RAC_ALU_OP_SET_MODE:
            rac_alu_set_mode(s,
                             (rac_alu_mode)(int)arg0,
                             (rac_alu_direction)(int)arg1);
            return 0;
        case RAC_ALU_OP_MICRO:
            return rac_alu_micro_step(s);
        case RAC_ALU_OP_RUN:
            return rac_alu_run(s, (int)arg0);
        case RAC_ALU_OP_ACCUM:
            rac_alu_accum(s, arg0);
            return 0;
        case RAC_ALU_OP_COMPENSATE:
            rac_alu_compensate(s);
            return 0;
        case RAC_ALU_OP_SIGN:
            (void)rac_alu_sign_decide(s);
            return 0;
        default:
            return -1;
    }
}

/* ── High-level primitives, routed through the ALU ───────────────────────── */
/*
 * Each of these is written only in terms of the ALU opcodes / helpers.
 * The pattern "reset → load → set_mode → run → read" is the RAC
 * equivalent of a MAC "load → multiply → add → store" sequence.
 */

rac_vec2 rac_alu_rotate(rac_vec2 v, float theta) {
    rac_alu_state s;
    rac_alu_reset(&s);
    _alu_quadrant_fold(&v, &theta);
    /* pre-scale input by K_INV so output magnitude == input magnitude */
    rac_alu_load(&s, v.x * RAC_ALU_K_INV, v.y * RAC_ALU_K_INV, theta);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    return (rac_vec2){s.x, s.y};
}

rac_vec2 rac_alu_rotate_raw(rac_vec2 v, float theta) {
    rac_alu_state s;
    rac_alu_reset(&s);
    _alu_quadrant_fold(&v, &theta);
    rac_alu_load(&s, v.x, v.y, theta);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    return (rac_vec2){s.x, s.y};
}

float rac_alu_project(rac_vec2 v, float theta) {
    /* project(v, θ) = v · (cos θ, sin θ) = v.x cos θ + v.y sin θ.
     * This equals the x-component of v rotated by -θ (the dot product
     * rotates the vector back onto the projection axis). */
    rac_alu_state s;
    rac_alu_reset(&s);
    float t = -theta;
    _alu_quadrant_fold(&v, &t);
    rac_alu_load(&s, v.x * RAC_ALU_K_INV, v.y * RAC_ALU_K_INV, t);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    rac_alu_accum(&s, 1.0f);   /* acc = x  (the "adder" step) */
    return s.acc;
}

/* Fast-path polar: branchless half-plane fold + inlined vectoring CORDIC.
 * This is the version called from all hot loops (inner, outer, matmul).
 * Branchless fold via XOR with sign-bit mask avoids the polar-branch that
 * otherwise blocks aggressive inlining into the surrounding loop. */
RAC_ALU_FORCE_INLINE void _alu_polar_fast(rac_vec2 v, float *mag, float *angle) {
    /* Build a 0x80000000-or-0 sign mask from (v.x < 0). */
    union { float f; unsigned u; } vx = { v.x };
    unsigned flip_mask = (vx.u & 0x80000000u);   /* sign bit of vx */
    /* Flip vx, vy iff vx is negative. */
    union { float f; unsigned u; } nvx = { v.x }, nvy = { v.y };
    nvx.u ^= flip_mask;
    nvy.u ^= flip_mask;
    /* z_offset = +π if original vy ≥ 0, else −π — only applied when flipping.
     * copysignf(π, vy_original) picks +π/-π, then masked by flip. */
    float offset_raw = copysignf(RAC_ALU_PI, v.y);
    float offset     = (flip_mask != 0u) ? offset_raw : 0.0f;

    /* Inlined circular vectoring CORDIC — no extern call. */
    float x = nvx.f, y = nvy.f, z = 0.0f;
    for (int i = 0; i < RAC_ALU_ITERS; i++) {
        float d     = _alu_dir_vec(y);
        float ds    = d * _alu_circ_scale[i];
        float xn    = x - ds * y;
        float yn    = y + ds * x;
        z -= d * _alu_atan_table[i];
        x = xn; y = yn;
    }
    if (mag)   *mag   = x * RAC_ALU_K_INV;
    if (angle) *angle = z + offset;
}

void rac_alu_polar(rac_vec2 v, float *mag, float *angle) {
    _alu_polar_fast(v, mag, angle);
}

float rac_alu_norm(rac_vec2 v) {
    float mag, ang;
    rac_alu_polar(v, &mag, &ang);
    (void)ang;
    return mag;
}

rac_vec2 rac_alu_normalize(rac_vec2 v) {
    float mag, ang;
    rac_alu_polar(v, &mag, &ang);
    (void)mag;
    /* rotate unit vector (1,0) by ang — pure ALU path */
    rac_vec2 base = {1.0f, 0.0f};
    _alu_quadrant_fold(&base, &ang);
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, base.x * RAC_ALU_K_INV, base.y * RAC_ALU_K_INV, ang);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    return (rac_vec2){s.x, s.y};
}

float rac_alu_dot(rac_vec2 a, rac_vec2 b) {
    float ma, aa, mb, ab;
    rac_alu_polar(a, &ma, &aa);
    rac_alu_polar(b, &mb, &ab);
    /* a·b = |a||b| cos(aa - ab). Evaluate cos via a rotation whose
     * x-component is cos of the angle: rotate (1,0) by (aa - ab), read x,
     * scale by |a||b|. Fold into convergence range first. */
    rac_vec2 base = {1.0f, 0.0f};
    float dtheta = aa - ab;
    _alu_quadrant_fold(&base, &dtheta);
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, base.x * RAC_ALU_K_INV, base.y * RAC_ALU_K_INV, dtheta);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    rac_alu_accum(&s, ma * mb);
    return s.acc;
}

/* Inlined rotation CORDIC kernel used by the hot loops. Pre-scales by
 * K_INV so output magnitude equals input magnitude. Caller is responsible
 * for quadrant folding the angle. */
RAC_ALU_FORCE_INLINE float _alu_project_fast(rac_vec2 v, float theta) {
    /* Quadrant fold: if |θ| > π/2, flip v and shift θ by ±π. Branchless
     * via a sign-mask XOR on x, y. */
    float t = theta;
    while (t >  RAC_ALU_PI) t -= 2.0f * RAC_ALU_PI;
    while (t < -RAC_ALU_PI) t += 2.0f * RAC_ALU_PI;
    if (t > 0.5f * RAC_ALU_PI)        { v.x = -v.x; v.y = -v.y; t -= RAC_ALU_PI; }
    else if (t < -0.5f * RAC_ALU_PI)  { v.x = -v.x; v.y = -v.y; t += RAC_ALU_PI; }

    float x = v.x * RAC_ALU_K_INV, y = v.y * RAC_ALU_K_INV, z = t;
    for (int i = 0; i < RAC_ALU_ITERS; i++) {
        float d     = _alu_dir_rot(z);
        float ds    = d * _alu_circ_scale[i];
        float xn    = x - ds * y;
        float yn    = y + ds * x;
        z -= d * _alu_atan_table[i];
        x = xn; y = yn;
    }
    return x;
}

float rac_alu_inner(const rac_vec2 *a, const rac_vec2 *b, int n) {
    /* Two-phase structure mirrors the hardware pipeline:
     *   1. Vectoring pass on b[i]  →  (|b[i]|, angle(b[i]))
     *   2. Rotation pass on a[i] by -angle(b[i])  →  a[i] · b_hat[i]
     * Contribution: project · |b[i]| summed into the accumulator.
     *
     * Phase fusion (combining the two CORDICs into one element loop) is
     * ~2x faster than separate phases for small n where the temp array
     * would overflow L1. For large n where precomputing pays off, see
     * rac_alu_inner_batch. */
    float total = 0.0f;
    #pragma omp parallel for reduction(+:total) schedule(static) if(n >= 1024)
    for (int i = 0; i < n; i++) {
        float mb, ab;
        _alu_polar_fast(b[i], &mb, &ab);
        total += _alu_project_fast(a[i], -ab) * mb;
    }
    return total;
}

void rac_alu_outer(const rac_vec2 *a, const rac_vec2 *b,
                   float *C, int m, int n) {
    /* Two-phase: precompute polar(b[j]) into temporary arrays so each
     * b-element pays for a vectoring pass ONCE instead of m times. For
     * m=1024, n=1024 this cuts polar work by 1024×.
     *
     * Phase 2 parallelizes over rows; each row writes a disjoint slice
     * of C (no reduction / false-sharing concerns). */
    float *mb = (float *)malloc((size_t)n * sizeof(float));
    float *ab = (float *)malloc((size_t)n * sizeof(float));
    if (!mb || !ab) { free(mb); free(ab); return; }

    #pragma omp parallel for schedule(static) if(n >= 256)
    for (int j = 0; j < n; j++) {
        _alu_polar_fast(b[j], &mb[j], &ab[j]);
    }

    #pragma omp parallel for schedule(static) if(m >= 64)
    for (int i = 0; i < m; i++) {
        const rac_vec2 av = a[i];
        float *Ci = C + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            Ci[j] = _alu_project_fast(av, -ab[j]) * mb[j];
        }
    }

    free(mb); free(ab);
}

float rac_alu_exp(float x) {
    /* Hyperbolic CORDIC converges only for |z| ≲ 1.12. For arbitrary x
     * we reduce the argument:
     *    k  = round(x / ln2)
     *    r  = x - k * ln2           ∈ [-ln2/2, ln2/2] ⊂ [-0.347, 0.347]
     *    e^x = 2^k · e^r
     * 2^k is a direct float32 exponent manipulation (no multiply),
     * and e^r stays well inside the CORDIC convergence domain. */
    const float LN2     = 0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    int k  = (int)(x * INV_LN2 + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)k * LN2;

    /* e^r = cosh(r) + sinh(r) via hyperbolic CORDIC with x0=1, y0=0, z=r.
     * After the sequence, x' = K_HYP · cosh(r), y' = K_HYP · sinh(r),
     * where K_HYP ≈ 0.82816. Multiply by K_HYP_RECIP (1/K_HYP ≈ 1.207)
     * to recover cosh/sinh. */
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, 1.0f, 0.0f, r);
    rac_alu_set_mode(&s, RAC_ALU_MODE_HYPERBOLIC, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    float er = (s.x + s.y) * RAC_ALU_K_HYP_RECIP;

    /* Apply 2^k via direct float32 exponent bit manipulation.
     * ldexpf would work, but the inline form avoids a libc call. */
    union { float f; int i; } u;
    int   bias_k = k + 127;
    if (bias_k <   0) return 0.0f;                          /* underflow */
    if (bias_k > 254) return er * INFINITY;                 /* overflow  */
    u.i = bias_k << 23;
    return er * u.f;
}

float rac_alu_tanh(float x) {
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, 1.0f, 0.0f, x);
    rac_alu_set_mode(&s, RAC_ALU_MODE_HYPERBOLIC, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    /* tanh = sinh/cosh = y/x (K_hyp cancels). */
    if (s.x == 0.0f) return 0.0f;
    return s.y / s.x;
}

/* ── AVX2 batch path ═════════════════════════════════════════════════════ */
/*
 * 8-wide parallel CORDIC. Each CORDIC iteration becomes 3 FMAs on YMM
 * registers (24 FLOPs) operating on 8 independent rotations at once.
 * The atan/scale constants are broadcast once per iteration from the
 * lookup tables, so all 8 lanes share the same control sequence — this
 * is exactly how a hardware batch-CORDIC engine would be wired.
 *
 * Sign decision uses the sign-bit extraction trick:
 *   d = copysign(1, z) = (z & 0x80000000) | 1.0f
 * No branches, no blends — pure bitwise ops.
 */

#if RAC_ALU_HAVE_AVX2

RAC_ALU_FORCE_INLINE __m256 _alu_sign_avx2(__m256 v) {
    /* d = copysign(1.0, v) = (v & sign_mask) | 1.0 */
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 one       = _mm256_set1_ps( 1.0f);
    __m256 sign      = _mm256_and_ps(v, sign_mask);
    return _mm256_or_ps(sign, one);
}

/* Core 8-wide circular-rotation CORDIC. Operates on SoA x/y/z YMMs in
 * place. Caller is responsible for K_INV pre-scaling and quadrant fold. */
RAC_ALU_FORCE_INLINE void _alu_avx2_cordic_rot(__m256 *x, __m256 *y, __m256 *z) {
    __m256 xr = *x, yr = *y, zr = *z;
    for (int i = 0; i < RAC_ALU_ITERS; i++) {
        __m256 d     = _alu_sign_avx2(zr);
        __m256 scale = _mm256_set1_ps(_alu_circ_scale[i]);
        __m256 atan  = _mm256_set1_ps(_alu_atan_table[i]);
        __m256 ds    = _mm256_mul_ps(d, scale);
        /* xn = x - ds * y,  yn = y + ds * x,  zn = z - d * atan */
        __m256 xn    = _mm256_fnmadd_ps(ds, yr, xr);
        __m256 yn    = _mm256_fmadd_ps (ds, xr, yr);
        zr           = _mm256_fnmadd_ps(d, atan, zr);
        xr = xn; yr = yn;
    }
    *x = xr; *y = yr; *z = zr;
}

/* Core 8-wide circular-vectoring CORDIC. Drives y→0; z ends up at atan2(y0,x0). */
RAC_ALU_FORCE_INLINE void _alu_avx2_cordic_vec(__m256 *x, __m256 *y, __m256 *z) {
    __m256 xr = *x, yr = *y, zr = *z;
    for (int i = 0; i < RAC_ALU_ITERS; i++) {
        /* Direction opposite sign of y: d = copysign(1, -y) = -copysign(1,y) */
        __m256 neg_y = _mm256_sub_ps(_mm256_setzero_ps(), yr);
        __m256 d     = _alu_sign_avx2(neg_y);
        __m256 scale = _mm256_set1_ps(_alu_circ_scale[i]);
        __m256 atan  = _mm256_set1_ps(_alu_atan_table[i]);
        __m256 ds    = _mm256_mul_ps(d, scale);
        __m256 xn    = _mm256_fnmadd_ps(ds, yr, xr);
        __m256 yn    = _mm256_fmadd_ps (ds, xr, yr);
        zr           = _mm256_fnmadd_ps(d, atan, zr);
        xr = xn; yr = yn;
    }
    *x = xr; *y = yr; *z = zr;
}

/* Load 8 rac_vec2 from AoS into SoA x, y YMMs.
 * Input layout (16 floats):
 *   [v0.x v0.y v1.x v1.y v2.x v2.y v3.x v3.y | v4.x v4.y v5.x v5.y v6.x v6.y v7.x v7.y]
 * Desired:
 *   xs = [v0.x v1.x v2.x v3.x v4.x v5.x v6.x v7.x]
 *   ys = [v0.y v1.y v2.y v3.y v4.y v5.y v6.y v7.y]
 */
RAC_ALU_FORCE_INLINE void _alu_load_aos8(const rac_vec2 *v, __m256 *xs, __m256 *ys) {
    __m256 a = _mm256_loadu_ps((const float *)(v + 0));   /* v0..v3 */
    __m256 b = _mm256_loadu_ps((const float *)(v + 4));   /* v4..v7 */
    /* Deinterleave via unpacklo/hi of 128-bit halves, then permute. */
    __m256 ab_lo = _mm256_permute2f128_ps(a, b, 0x20);    /* v0..v1 v4..v5 */
    __m256 ab_hi = _mm256_permute2f128_ps(a, b, 0x31);    /* v2..v3 v6..v7 */
    *xs = _mm256_shuffle_ps(ab_lo, ab_hi, 0x88);          /* even lanes (x) */
    *ys = _mm256_shuffle_ps(ab_lo, ab_hi, 0xDD);          /* odd  lanes (y) */
}

/* Store SoA xs, ys back to 8 AoS rac_vec2. */
RAC_ALU_FORCE_INLINE void _alu_store_aos8(rac_vec2 *out, __m256 xs, __m256 ys) {
    __m256 lo = _mm256_unpacklo_ps(xs, ys);   /* x0 y0 x1 y1 x4 y4 x5 y5 */
    __m256 hi = _mm256_unpackhi_ps(xs, ys);   /* x2 y2 x3 y3 x6 y6 x7 y7 */
    __m256 p0 = _mm256_permute2f128_ps(lo, hi, 0x20);   /* v0..v3 */
    __m256 p1 = _mm256_permute2f128_ps(lo, hi, 0x31);   /* v4..v7 */
    _mm256_storeu_ps((float *)(out + 0), p0);
    _mm256_storeu_ps((float *)(out + 4), p1);
}

int rac_alu_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 1;  /* compiled in, assume runtime support */
#endif
}

/* Single 8-element AVX2 rotate block: load, fold, run CORDIC, store. */
RAC_ALU_FORCE_INLINE void _alu_rotate_block8(const rac_vec2 *v,
                                             const float *theta,
                                             rac_vec2 *out) {
    const __m256 kinv      = _mm256_set1_ps(RAC_ALU_K_INV);
    const __m256 pi        = _mm256_set1_ps(RAC_ALU_PI);
    const __m256 half_pi   = _mm256_set1_ps(0.5f * RAC_ALU_PI);
    const __m256 two_pi    = _mm256_set1_ps(2.0f * RAC_ALU_PI);
    const __m256 neg_half  = _mm256_set1_ps(-0.5f * RAC_ALU_PI);
    const __m256 neg_pi    = _mm256_set1_ps(-RAC_ALU_PI);
    const __m256 sign_bit  = _mm256_set1_ps(-0.0f);

    __m256 xs, ys;
    _alu_load_aos8(v, &xs, &ys);
    __m256 zs = _mm256_loadu_ps(theta);

    /* Reduce to (-π, π] */
    __m256 gt_pi  = _mm256_cmp_ps(zs,  pi,     _CMP_GT_OQ);
    __m256 lt_npi = _mm256_cmp_ps(zs,  neg_pi, _CMP_LT_OQ);
    zs = _mm256_blendv_ps(zs, _mm256_sub_ps(zs, two_pi), gt_pi);
    zs = _mm256_blendv_ps(zs, _mm256_add_ps(zs, two_pi), lt_npi);
    /* Fold to (-π/2, π/2] */
    __m256 gt_half = _mm256_cmp_ps(zs, half_pi,  _CMP_GT_OQ);
    __m256 lt_nh   = _mm256_cmp_ps(zs, neg_half, _CMP_LT_OQ);
    __m256 flip    = _mm256_or_ps(gt_half, lt_nh);
    __m256 sign    = _mm256_and_ps(flip, sign_bit);
    xs = _mm256_xor_ps(xs, sign);
    ys = _mm256_xor_ps(ys, sign);
    zs = _mm256_blendv_ps(zs, _mm256_sub_ps(zs, pi), gt_half);
    zs = _mm256_blendv_ps(zs, _mm256_add_ps(zs, pi), lt_nh);

    xs = _mm256_mul_ps(xs, kinv);
    ys = _mm256_mul_ps(ys, kinv);
    _alu_avx2_cordic_rot(&xs, &ys, &zs);
    _alu_store_aos8(out, xs, ys);
}

static void _alu_rotate_batch_avx2(const rac_vec2 *v, const float *theta,
                                   rac_vec2 *out, int n) {
    const int nblocks = n / 8;
    #pragma omp parallel for schedule(static) if(nblocks >= 64)
    for (int b = 0; b < nblocks; b++) {
        const int i = b * 8;
        _alu_rotate_block8(v + i, theta + i, out + i);
    }
    /* Scalar tail */
    for (int i = nblocks * 8; i < n; i++) {
        out[i] = rac_alu_rotate(v[i], theta[i]);
    }
}

/* 8-element AVX2 polar block: produces {|b|, angle(b)} for 8 b-vectors
 * with half-plane pre-fold. Used in phase 1 of the batch inner/outer. */
RAC_ALU_FORCE_INLINE void _alu_polar_block8(const rac_vec2 *b,
                                            __m256 *mag, __m256 *ang) {
    const __m256 kinv     = _mm256_set1_ps(RAC_ALU_K_INV);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 pi_pos   = _mm256_set1_ps( RAC_ALU_PI);
    const __m256 pi_neg   = _mm256_set1_ps(-RAC_ALU_PI);
    const __m256 zero     = _mm256_setzero_ps();

    __m256 bx, by;
    _alu_load_aos8(b, &bx, &by);

    __m256 bx_neg = _mm256_cmp_ps(bx, zero, _CMP_LT_OQ);
    __m256 by_neg = _mm256_cmp_ps(by, zero, _CMP_LT_OQ);
    __m256 flip   = _mm256_and_ps(bx_neg, sign_bit);
    __m256 bx_f   = _mm256_xor_ps(bx, flip);
    __m256 by_f   = _mm256_xor_ps(by, flip);
    __m256 offs   = _mm256_blendv_ps(pi_pos, pi_neg, by_neg);
    offs          = _mm256_and_ps(offs, bx_neg);

    __m256 vx = bx_f, vy = by_f, vz = zero;
    _alu_avx2_cordic_vec(&vx, &vy, &vz);
    *mag = _mm256_mul_ps(vx, kinv);
    *ang = _mm256_add_ps(vz, offs);
}

/* 8-element AVX2 project block: rotate a by theta (already quadrant-folded
 * caller-side is OK; we fold again here for safety) and return x-lane. */
RAC_ALU_FORCE_INLINE __m256 _alu_project_block8(const rac_vec2 *a,
                                                __m256 theta) {
    const __m256 kinv      = _mm256_set1_ps(RAC_ALU_K_INV);
    const __m256 pi        = _mm256_set1_ps(RAC_ALU_PI);
    const __m256 half_pi   = _mm256_set1_ps(0.5f * RAC_ALU_PI);
    const __m256 two_pi    = _mm256_set1_ps(2.0f * RAC_ALU_PI);
    const __m256 neg_half  = _mm256_set1_ps(-0.5f * RAC_ALU_PI);
    const __m256 neg_pi    = _mm256_set1_ps(-RAC_ALU_PI);
    const __m256 sign_bit  = _mm256_set1_ps(-0.0f);

    __m256 ax, ay;
    _alu_load_aos8(a, &ax, &ay);

    /* Reduce & fold theta. */
    __m256 gt_pi  = _mm256_cmp_ps(theta,  pi,     _CMP_GT_OQ);
    __m256 lt_npi = _mm256_cmp_ps(theta,  neg_pi, _CMP_LT_OQ);
    theta = _mm256_blendv_ps(theta, _mm256_sub_ps(theta, two_pi), gt_pi);
    theta = _mm256_blendv_ps(theta, _mm256_add_ps(theta, two_pi), lt_npi);
    __m256 gt_half = _mm256_cmp_ps(theta, half_pi,  _CMP_GT_OQ);
    __m256 lt_nh   = _mm256_cmp_ps(theta, neg_half, _CMP_LT_OQ);
    __m256 flip    = _mm256_or_ps(gt_half, lt_nh);
    __m256 sgn     = _mm256_and_ps(flip, sign_bit);
    ax = _mm256_xor_ps(ax, sgn);
    ay = _mm256_xor_ps(ay, sgn);
    theta = _mm256_blendv_ps(theta, _mm256_sub_ps(theta, pi), gt_half);
    theta = _mm256_blendv_ps(theta, _mm256_add_ps(theta, pi), lt_nh);

    ax = _mm256_mul_ps(ax, kinv);
    ay = _mm256_mul_ps(ay, kinv);
    _alu_avx2_cordic_rot(&ax, &ay, &theta);
    return ax;   /* x-component = projection */
}

RAC_ALU_FORCE_INLINE float _alu_hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    float r; _mm_store_ss(&r, s1); return r;
}

/* AVX2 + OpenMP inner product. Phase-fused: each thread does full
 * polar+project+accum on its block and contributes a scalar reduction.
 * For small n we keep it single-threaded to avoid fork overhead. */
static float _alu_inner_batch_avx2(const rac_vec2 *a, const rac_vec2 *b, int n) {
    const int nblocks = n / 8;
    float total = 0.0f;

    #pragma omp parallel for reduction(+:total) schedule(static) \
            if(nblocks >= 128)
    for (int blk = 0; blk < nblocks; blk++) {
        const int i = blk * 8;
        __m256 mb, ab;
        _alu_polar_block8(b + i, &mb, &ab);
        __m256 neg_ab = _mm256_sub_ps(_mm256_setzero_ps(), ab);
        __m256 proj   = _alu_project_block8(a + i, neg_ab);
        __m256 part   = _mm256_mul_ps(proj, mb);
        total        += _alu_hsum_ps(part);
    }
    /* Scalar tail */
    for (int i = nblocks * 8; i < n; i++) {
        float mb, ab;
        _alu_polar_fast(b[i], &mb, &ab);
        total += _alu_project_fast(a[i], -ab) * mb;
    }
    return total;
}

/* AVX2 outer product: phase 1 precomputes |b|,angle(b) for all j using
 * the 8-wide polar block; phase 2 walks rows of C, dispatching each row
 * to 8-wide project blocks against the angle table. Phase 2 is
 * embarrassingly parallel across rows. */
static void _alu_outer_batch_avx2(const rac_vec2 *a, const rac_vec2 *b,
                                  float *C, int m, int n) {
    /* Aligned temp tables. posix_memalign is C99-friendly; aligned_alloc
     * would require C11 which this tree doesn't use (CMAKE_C_STANDARD=99). */
    float *mb = NULL, *ab = NULL;
    size_t nb = ((size_t)n * sizeof(float) + 31u) & ~31u;
    if (posix_memalign((void **)&mb, 32, nb) != 0) { return; }
    if (posix_memalign((void **)&ab, 32, nb) != 0) { free(mb); return; }

    /* Phase 1: polar(b) once per j. */
    const int n8 = n / 8;
    #pragma omp parallel for schedule(static) if(n8 >= 64)
    for (int blk = 0; blk < n8; blk++) {
        __m256 mm, aa;
        _alu_polar_block8(b + blk * 8, &mm, &aa);
        _mm256_storeu_ps(mb + blk * 8, mm);
        _mm256_storeu_ps(ab + blk * 8, aa);
    }
    for (int j = n8 * 8; j < n; j++) _alu_polar_fast(b[j], &mb[j], &ab[j]);

    /* Phase 2: for each row i, project a[i] onto all b-directions.
     * We broadcast a[i] into an 8-lane SoA vector and reuse it across
     * the row, picking up a block of 8 angles each step. */
    #pragma omp parallel for schedule(static) if(m >= 32)
    for (int i = 0; i < m; i++) {
        const rac_vec2 av = a[i];
        /* Broadcast a[i] into 8-lane xs, ys. */
        const __m256 ax_bc = _mm256_set1_ps(av.x);
        const __m256 ay_bc = _mm256_set1_ps(av.y);
        float *Ci = C + (size_t)i * (size_t)n;
        int j = 0;
        const __m256 kinv      = _mm256_set1_ps(RAC_ALU_K_INV);
        const __m256 pi        = _mm256_set1_ps(RAC_ALU_PI);
        const __m256 half_pi   = _mm256_set1_ps(0.5f * RAC_ALU_PI);
        const __m256 two_pi    = _mm256_set1_ps(2.0f * RAC_ALU_PI);
        const __m256 neg_half  = _mm256_set1_ps(-0.5f * RAC_ALU_PI);
        const __m256 neg_pi    = _mm256_set1_ps(-RAC_ALU_PI);
        const __m256 sign_bit  = _mm256_set1_ps(-0.0f);
        for (; j + 8 <= n; j += 8) {
            __m256 theta = _mm256_sub_ps(_mm256_setzero_ps(),
                                         _mm256_loadu_ps(ab + j));
            __m256 ax = ax_bc, ay = ay_bc;
            /* Quadrant fold (same as _alu_project_block8 but inlined) */
            __m256 gt_pi  = _mm256_cmp_ps(theta,  pi,     _CMP_GT_OQ);
            __m256 lt_npi = _mm256_cmp_ps(theta,  neg_pi, _CMP_LT_OQ);
            theta = _mm256_blendv_ps(theta, _mm256_sub_ps(theta, two_pi), gt_pi);
            theta = _mm256_blendv_ps(theta, _mm256_add_ps(theta, two_pi), lt_npi);
            __m256 gt_half = _mm256_cmp_ps(theta, half_pi,  _CMP_GT_OQ);
            __m256 lt_nh   = _mm256_cmp_ps(theta, neg_half, _CMP_LT_OQ);
            __m256 flip    = _mm256_or_ps(gt_half, lt_nh);
            __m256 sgn     = _mm256_and_ps(flip, sign_bit);
            ax = _mm256_xor_ps(ax, sgn);
            ay = _mm256_xor_ps(ay, sgn);
            theta = _mm256_blendv_ps(theta, _mm256_sub_ps(theta, pi), gt_half);
            theta = _mm256_blendv_ps(theta, _mm256_add_ps(theta, pi), lt_nh);
            ax = _mm256_mul_ps(ax, kinv);
            ay = _mm256_mul_ps(ay, kinv);
            _alu_avx2_cordic_rot(&ax, &ay, &theta);
            __m256 cij = _mm256_mul_ps(ax, _mm256_loadu_ps(mb + j));
            _mm256_storeu_ps(Ci + j, cij);
        }
        /* Tail */
        for (; j < n; j++) {
            Ci[j] = _alu_project_fast(av, -ab[j]) * mb[j];
        }
    }

    free(mb); free(ab);
}

/* SoA batch rotate — same CORDIC kernel as AoS version but operates on
 * separate x/y arrays. Saves the AoS→SoA transpose (~4 perm/shuffle per
 * 8-element block) plus improves cache utilization when x,y arrays are
 * aligned and contiguous. */
static void _alu_rotate_batch_soa_avx2(const float *vx, const float *vy,
                                       const float *theta,
                                       float *out_x, float *out_y, int n) {
    const __m256 kinv      = _mm256_set1_ps(RAC_ALU_K_INV);
    const __m256 pi        = _mm256_set1_ps(RAC_ALU_PI);
    const __m256 half_pi   = _mm256_set1_ps(0.5f * RAC_ALU_PI);
    const __m256 two_pi    = _mm256_set1_ps(2.0f * RAC_ALU_PI);
    const __m256 neg_half  = _mm256_set1_ps(-0.5f * RAC_ALU_PI);
    const __m256 neg_pi    = _mm256_set1_ps(-RAC_ALU_PI);
    const __m256 sign_bit  = _mm256_set1_ps(-0.0f);

    const int nblocks = n / 8;
    #pragma omp parallel for schedule(static) if(nblocks >= 64)
    for (int blk = 0; blk < nblocks; blk++) {
        const int i = blk * 8;
        __m256 xs = _mm256_loadu_ps(vx + i);
        __m256 ys = _mm256_loadu_ps(vy + i);
        __m256 zs = _mm256_loadu_ps(theta + i);

        __m256 gt_pi  = _mm256_cmp_ps(zs,  pi,     _CMP_GT_OQ);
        __m256 lt_npi = _mm256_cmp_ps(zs,  neg_pi, _CMP_LT_OQ);
        zs = _mm256_blendv_ps(zs, _mm256_sub_ps(zs, two_pi), gt_pi);
        zs = _mm256_blendv_ps(zs, _mm256_add_ps(zs, two_pi), lt_npi);
        __m256 gt_half = _mm256_cmp_ps(zs, half_pi,  _CMP_GT_OQ);
        __m256 lt_nh   = _mm256_cmp_ps(zs, neg_half, _CMP_LT_OQ);
        __m256 flip    = _mm256_or_ps(gt_half, lt_nh);
        __m256 sgn     = _mm256_and_ps(flip, sign_bit);
        xs = _mm256_xor_ps(xs, sgn);
        ys = _mm256_xor_ps(ys, sgn);
        zs = _mm256_blendv_ps(zs, _mm256_sub_ps(zs, pi), gt_half);
        zs = _mm256_blendv_ps(zs, _mm256_add_ps(zs, pi), lt_nh);

        xs = _mm256_mul_ps(xs, kinv);
        ys = _mm256_mul_ps(ys, kinv);
        _alu_avx2_cordic_rot(&xs, &ys, &zs);
        _mm256_storeu_ps(out_x + i, xs);
        _mm256_storeu_ps(out_y + i, ys);
    }
    /* Scalar tail */
    for (int i = nblocks * 8; i < n; i++) {
        rac_vec2 r = rac_alu_rotate((rac_vec2){vx[i], vy[i]}, theta[i]);
        out_x[i] = r.x; out_y[i] = r.y;
    }
}

#else  /* !RAC_ALU_HAVE_AVX2 */

int rac_alu_has_avx2(void) { return 0; }

#endif

int rac_alu_omp_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void rac_alu_rotate_batch(const rac_vec2 *v, const float *theta,
                          rac_vec2 *out, int n) {
#if RAC_ALU_HAVE_AVX2
    if (rac_alu_has_avx2()) {
        _alu_rotate_batch_avx2(v, theta, out, n);
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(n >= 1024)
    for (int i = 0; i < n; i++) out[i] = rac_alu_rotate(v[i], theta[i]);
}

float rac_alu_inner_batch(const rac_vec2 *a, const rac_vec2 *b, int n) {
#if RAC_ALU_HAVE_AVX2
    if (rac_alu_has_avx2()) return _alu_inner_batch_avx2(a, b, n);
#endif
    return rac_alu_inner(a, b, n);
}

void rac_alu_outer_batch(const rac_vec2 *a, const rac_vec2 *b,
                         float *C, int m, int n) {
#if RAC_ALU_HAVE_AVX2
    if (rac_alu_has_avx2()) {
        _alu_outer_batch_avx2(a, b, C, m, n);
        return;
    }
#endif
    rac_alu_outer(a, b, C, m, n);
}

void rac_alu_rotate_batch_soa(const float *vx, const float *vy,
                              const float *theta,
                              float *out_x, float *out_y, int n) {
#if RAC_ALU_HAVE_AVX2
    if (rac_alu_has_avx2()) {
        _alu_rotate_batch_soa_avx2(vx, vy, theta, out_x, out_y, n);
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(n >= 1024)
    for (int i = 0; i < n; i++) {
        rac_vec2 r = rac_alu_rotate((rac_vec2){vx[i], vy[i]}, theta[i]);
        out_x[i] = r.x; out_y[i] = r.y;
    }
}

/* ── Introspection ───────────────────────────────────────────────────────── */

const char *rac_alu_op_name(rac_alu_opcode op) {
    switch (op) {
        case RAC_ALU_OP_LOAD:       return "LOAD";
        case RAC_ALU_OP_CLEAR_ACC:  return "CLEAR_ACC";
        case RAC_ALU_OP_SET_MODE:   return "SET_MODE";
        case RAC_ALU_OP_MICRO:      return "MICRO";
        case RAC_ALU_OP_RUN:        return "RUN";
        case RAC_ALU_OP_ACCUM:      return "ACCUM";
        case RAC_ALU_OP_COMPENSATE: return "COMPENSATE";
        case RAC_ALU_OP_SIGN:       return "SIGN";
        default:                    return "UNKNOWN";
    }
}

const char *rac_alu_mode_name(rac_alu_mode mode) {
    switch (mode) {
        case RAC_ALU_MODE_CIRCULAR:   return "CIRCULAR";
        case RAC_ALU_MODE_HYPERBOLIC: return "HYPERBOLIC";
        case RAC_ALU_MODE_LINEAR:     return "LINEAR";
        default:                      return "UNKNOWN";
    }
}
