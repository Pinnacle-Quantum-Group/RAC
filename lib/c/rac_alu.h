/*
 * rac_alu.h — RAC Adder + ALU (unified)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — April 2026
 *
 * A purposeful combination of an adder and an ALU, tailored specifically
 * for the Rotation-Accumulate primitive. Where a MAC ALU is built around
 * a multiplier and a single adder, the RAC ALU is built around a *dual*
 * shift-add/subtract cell (the CORDIC micro-step) and an integrated
 * projection accumulator. There is no multiplier.
 *
 * The ALU has five opcode classes, and every one of the 17 RAC primitives
 * (rac.h) reduces to a sequence of these:
 *
 *   RAC_ALU_MICRO        one CORDIC micro-rotation (x,y,z dual-adder)
 *   RAC_ALU_ACCUM        pull x-register into scalar accumulator
 *   RAC_ALU_COMPENSATE   apply K^-chain gain compensation (post-chain)
 *   RAC_ALU_SIGN         direction decision d = ±1 from z or y sign
 *   RAC_ALU_LOAD / CLEAR state register load / accumulator clear
 *
 * The micro-step cell implements, in a single cycle-equivalent operation:
 *
 *     x' = x  -  d · (y >> i)        circular mode
 *     y' = y  +  d · (x >> i)
 *     z' = z  -  d · atan(2^-i)
 *
 *     x' = x  +  d · (y >> i)        hyperbolic mode
 *     y' = y  +  d · (x >> i)
 *     z' = z  -  d · atanh(2^-i)
 *
 *     x' = x                         linear mode (multiply by z, debug only)
 *     y' = y  +  d · (x >> i)
 *     z' = z  -  d · 2^-i
 *
 * The sign d is chosen from z (rotation mode) or y (vectoring mode).
 * All three coordinates update through the same adder pair — rotation,
 * vectoring, circular, hyperbolic, and linear CORDIC all share hardware.
 *
 * This header exposes both the low-level ALU state machine (for anyone
 * modelling the hardware) and high-level wrappers that re-express the
 * core RAC primitives strictly through ALU opcodes, proving the
 * ALU is functionally complete for RAC.
 */

#ifndef RAC_ALU_H
#define RAC_ALU_H

#include "rac_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants (match rac_cpu.h) ─────────────────────────────────────────── */

#define RAC_ALU_ITERS       RAC_ITERS
#define RAC_ALU_K           RAC_K
#define RAC_ALU_K_INV       RAC_K_INV
#define RAC_ALU_K_HYP_INV   RAC_K_HYP_INV

/* ── Mode / direction / opcode enums ─────────────────────────────────────── */

typedef enum {
    RAC_ALU_MODE_CIRCULAR   = 0,   /* atan table, scale halves each step       */
    RAC_ALU_MODE_HYPERBOLIC = 1,   /* atanh table, repeat iters 4 and 13       */
    RAC_ALU_MODE_LINEAR     = 2,   /* scale = 2^-i, used for multiply/divide   */
} rac_alu_mode;

typedef enum {
    RAC_ALU_DIR_ROTATION  = 0,     /* drive z to zero; d = sign(z)             */
    RAC_ALU_DIR_VECTORING = 1,     /* drive y to zero; d = -sign(y)            */
} rac_alu_direction;

typedef enum {
    RAC_ALU_OP_LOAD       = 0,
    RAC_ALU_OP_CLEAR_ACC  = 1,
    RAC_ALU_OP_SET_MODE   = 2,
    RAC_ALU_OP_MICRO      = 3,     /* one CORDIC iteration                     */
    RAC_ALU_OP_RUN        = 4,     /* RAC_ALU_ITERS micro-steps                */
    RAC_ALU_OP_ACCUM      = 5,     /* acc += x * scale (scale given externally)*/
    RAC_ALU_OP_COMPENSATE = 6,     /* x,y *= K^-chain (or K_hyp^-chain)        */
    RAC_ALU_OP_SIGN       = 7,     /* compute direction d for next micro step  */
} rac_alu_opcode;

/* ── ALU state ───────────────────────────────────────────────────────────── */

typedef struct {
    /* Data registers — rotated/accumulated in-place */
    float x;
    float y;
    float z;

    /* Scalar projection accumulator — the "adder" half of RAC.
     * rac_inner / rac_dot / rac_matmul fold their partial products here. */
    float acc;

    /* Iteration counter, 0 .. RAC_ALU_ITERS-1 */
    int   iter;

    /* Chain length: how many raw rotations have been applied since the
     * last compensate. Used by RAC_ALU_OP_COMPENSATE. */
    int   chain;

    /* Mode + direction control */
    rac_alu_mode       mode;
    rac_alu_direction  dir;

    /* Last direction decision d ∈ {-1, +1} */
    float d;
} rac_alu_state;

/* ── Low-level ALU ops (opcode dispatch) ─────────────────────────────────── */

/* Reset all registers to zero, mode circular-rotation. */
void  rac_alu_reset(rac_alu_state *s);

/* Load x/y/z registers. Does not touch accumulator or chain. */
void  rac_alu_load(rac_alu_state *s, float x, float y, float z);

/* Clear the scalar accumulator. */
void  rac_alu_clear_acc(rac_alu_state *s);

/* Set mode and direction. Resets iter to 0. */
void  rac_alu_set_mode(rac_alu_state *s,
                       rac_alu_mode mode,
                       rac_alu_direction dir);

/* Decide the direction d for the current state.
 * Rotation:  d = (z >= 0) ?  +1 : -1
 * Vectoring: d = (y <  0) ?  +1 : -1
 * Result also stored in s->d and returned. */
float rac_alu_sign_decide(rac_alu_state *s);

/* Execute ONE CORDIC micro-step using current mode/direction.
 * Internally: calls rac_alu_sign_decide, then performs dual shift-add/sub.
 * Advances s->iter and increments s->chain. Returns 0 on success, -1 if
 * the ALU has already completed RAC_ALU_ITERS iterations. */
int   rac_alu_micro_step(rac_alu_state *s);

/* Run a full CORDIC sequence of `iters` steps (≤ RAC_ALU_ITERS). */
int   rac_alu_run(rac_alu_state *s, int iters);

/* Accumulate the current x-register (scaled) into the scalar accumulator.
 * This is the "adder" half of RAC: the only place partial results get
 * summed. Used by inner, dot, matmul, DCT, outer. */
void  rac_alu_accum(rac_alu_state *s, float scale);

/* Apply K^-chain gain compensation to x,y (circular K, else K_hyp).
 * Clears the chain counter. */
void  rac_alu_compensate(rac_alu_state *s);

/* Generic opcode dispatcher — a single entry point for hardware modeling.
 *
 * opcode          param     behaviour
 * ─────────────── ───────── ───────────────────────────────────────────────
 * OP_LOAD         arg0=x, arg1=y, arg2=z
 * OP_CLEAR_ACC    —
 * OP_SET_MODE     arg0 = rac_alu_mode cast to float
 *                 arg1 = rac_alu_direction cast to float
 * OP_MICRO        —         (ignores args)
 * OP_RUN          arg0 = iters
 * OP_ACCUM        arg0 = scale
 * OP_COMPENSATE   —
 * OP_SIGN         —         (result in s->d)
 *
 * Returns 0 on success, -1 on opcode/arg error. */
int   rac_alu_dispatch(rac_alu_state *s, rac_alu_opcode op,
                       float arg0, float arg1, float arg2);

/* ── High-level: RAC primitives expressed strictly through the ALU ───────── */
/*
 * These are reference re-implementations of the core RAC primitives using
 * ONLY the ALU opcodes above. They prove the ALU is functionally complete
 * for the RAC primitive set, and they are bit-for-bit equivalent (within
 * CORDIC precision) to the rac_cpu.c versions.
 *
 * Naming: rac_alu_<primitive> to avoid symbol collisions with rac_cpu.c.
 */

rac_vec2 rac_alu_rotate(rac_vec2 v, float theta);
rac_vec2 rac_alu_rotate_raw(rac_vec2 v, float theta);
float    rac_alu_project(rac_vec2 v, float theta);
void     rac_alu_polar(rac_vec2 v, float *mag, float *angle);
float    rac_alu_norm(rac_vec2 v);
rac_vec2 rac_alu_normalize(rac_vec2 v);
float    rac_alu_dot(rac_vec2 a, rac_vec2 b);
float    rac_alu_inner(const rac_vec2 *a, const rac_vec2 *b, int n);
void     rac_alu_outer(const rac_vec2 *a, const rac_vec2 *b,
                       float *C, int m, int n);

/* Hyperbolic CORDIC: reference via the same ALU. Accuracy is lower than
 * libm — provided for architectural completeness / hardware modeling.
 *
 * rac_alu_exp handles arbitrary float32 input via argument reduction
 *   exp(x) = 2^k · exp(r)  where r ∈ [-ln(2)/2, ln(2)/2]
 * keeping the hyperbolic CORDIC inside its convergence domain (|r| < 1.12). */
float    rac_alu_exp(float x);
float    rac_alu_tanh(float x);

/* ── AVX2 batch path (optional — compiled only if -mavx2 -mfma) ──────────── */
/*
 * These operate on AoS arrays of rac_vec2. When compiled with AVX2+FMA they
 * process 8 independent CORDICs per iteration in parallel, sharing the
 * same atan/scale table broadcasts. On a host without AVX2 they fall back
 * to the scalar straight-line path transparently.
 *
 * rac_alu_has_avx2() returns 1 if the AVX2 fast path was compiled in AND
 * the host CPU supports AVX2+FMA at runtime.
 */
int      rac_alu_has_avx2(void);
int      rac_alu_omp_threads(void);   /* 1 if OpenMP not enabled */
void     rac_alu_rotate_batch(const rac_vec2 *v, const float *theta,
                              rac_vec2 *out, int n);
float    rac_alu_inner_batch(const rac_vec2 *a, const rac_vec2 *b, int n);
void     rac_alu_outer_batch(const rac_vec2 *a, const rac_vec2 *b,
                             float *C, int m, int n);

/* SoA batch rotate — skips the AoS→SoA transpose. Preferred for callers
 * that can keep x and y in separate aligned arrays. */
void     rac_alu_rotate_batch_soa(const float *vx, const float *vy,
                                  const float *theta,
                                  float *out_x, float *out_y, int n);

/* ── Inspection / introspection ──────────────────────────────────────────── */

/* Return a short human-readable description of the ALU opcode (static). */
const char *rac_alu_op_name(rac_alu_opcode op);
const char *rac_alu_mode_name(rac_alu_mode mode);

#ifdef __cplusplus
}
#endif

#endif /* RAC_ALU_H */
