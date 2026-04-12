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

#include "rac_alu.h"
#include <math.h>
#include <stddef.h>

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
            return _alu_shift(1.0f, idx);
        case RAC_ALU_MODE_CIRCULAR:
        default:
            return _alu_atan_table[idx];
    }
}

static inline float _alu_scale_value(const rac_alu_state *s) {
    if (s->mode == RAC_ALU_MODE_HYPERBOLIC) {
        return _alu_shift(1.0f, _alu_hyp_iter_map[s->iter]);
    }
    return _alu_shift(1.0f, s->iter);
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

int rac_alu_run(rac_alu_state *s, int iters) {
    if (iters > RAC_ALU_ITERS) iters = RAC_ALU_ITERS;
    s->iter = 0;
    for (int i = 0; i < iters; i++) {
        if (rac_alu_micro_step(s) != 0) return -1;
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
     * RAC_ALU_K_HYP_INV stores K_hyp itself, so we divide rather than multiply. */
    float c;
    if (s->mode == RAC_ALU_MODE_HYPERBOLIC) {
        c = _alu_powf(1.0f / RAC_ALU_K_HYP_INV, s->chain);
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

void rac_alu_polar(rac_vec2 v, float *mag, float *angle) {
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, v.x, v.y, 0.0f);
    rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_VECTORING);
    rac_alu_run(&s, RAC_ALU_ITERS);
    /* Vectoring drives y→0, x = K * sqrt(x0^2+y0^2), z = atan2(y0,x0). */
    if (mag)   *mag   = s.x * RAC_ALU_K_INV;
    if (angle) *angle = s.z;
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

float rac_alu_inner(const rac_vec2 *a, const rac_vec2 *b, int n) {
    /* The canonical use of the projection accumulator: run the ALU once
     * per element, feeding x into acc via RAC_ALU_OP_ACCUM.
     *
     * Per element:  a[i] · b[i]
     *            = (|a[i]| cos(aa))(|b[i]| cos(ab)) + ... sin terms
     *            = rac_project(a[i], ab) * |b[i]|
     *            = rotate(a[i], -ab).x * |b[i]|
     */
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_clear_acc(&s);

    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        float mb, ab;
        rac_alu_polar(b[i], &mb, &ab);   /* vectoring pass */

        /* rotation pass by -ab (project onto direction of b[i]) */
        rac_vec2 av = a[i];
        float theta = -ab;
        _alu_quadrant_fold(&av, &theta);
        rac_alu_load(&s, av.x * RAC_ALU_K_INV, av.y * RAC_ALU_K_INV, theta);
        rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
        rac_alu_run(&s, RAC_ALU_ITERS);
        rac_alu_accum(&s, mb);           /* acc += x * |b[i]| */
        total = s.acc;
    }
    return total;
}

void rac_alu_outer(const rac_vec2 *a, const rac_vec2 *b,
                   float *C, int m, int n) {
    rac_alu_state s;
    rac_alu_reset(&s);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float mb, ab;
            rac_alu_polar(b[j], &mb, &ab);
            rac_vec2 av = a[i];
            float theta = -ab;
            _alu_quadrant_fold(&av, &theta);
            rac_alu_load(&s, av.x * RAC_ALU_K_INV, av.y * RAC_ALU_K_INV, theta);
            rac_alu_set_mode(&s, RAC_ALU_MODE_CIRCULAR, RAC_ALU_DIR_ROTATION);
            rac_alu_run(&s, RAC_ALU_ITERS);
            C[i * n + j] = s.x * mb;
        }
    }
}

float rac_alu_exp(float x) {
    /* e^x = cosh(x) + sinh(x), computed via hyperbolic CORDIC with
     *   x0 = 1, y0 = 0, z0 = x
     * After the sequence, x' = K_hyp · cosh(x), y' = K_hyp · sinh(x),
     * where K_hyp ≈ 0.82816 is the hyperbolic CORDIC gain (< 1, unlike
     * circular CORDIC whose gain is > 1). So we DIVIDE by K_hyp to
     * recover cosh/sinh. The macro RAC_ALU_K_HYP_INV holds K_hyp itself
     * (the name is inherited from rac_cpu.h for consistency).          */
    rac_alu_state s;
    rac_alu_reset(&s);
    rac_alu_load(&s, 1.0f, 0.0f, x);
    rac_alu_set_mode(&s, RAC_ALU_MODE_HYPERBOLIC, RAC_ALU_DIR_ROTATION);
    rac_alu_run(&s, RAC_ALU_ITERS);
    const float inv_K_hyp = 1.0f / RAC_ALU_K_HYP_INV;
    float cosh_x = s.x * inv_K_hyp;
    float sinh_x = s.y * inv_K_hyp;
    return cosh_x + sinh_x;
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
