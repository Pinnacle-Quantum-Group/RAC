/*
 * rac_dsp_core.h — shared math engine for the RAC-DSP C reference.
 *
 * rac_dsp_ref.c and rac_systolic_ref.c both need the same Q0.63
 * CORDIC evaluation. Put it behind a thin API here so they share one
 * implementation; any bug fix lands in rac_dsp_core.c once and the
 * systolic reference stays in lockstep with the single-DSP reference.
 */

#ifndef RAC_DSP_CORE_H
#define RAC_DSP_CORE_H

#include <stdint.h>

#define RAC_DSP_WIDTH           64
#define RAC_DSP_LUT_BITS        10
#define RAC_DSP_RESIDUAL        9
#define RAC_DSP_RESIDUAL_START  8
#define RAC_DSP_LUT_SIZE        (1 << RAC_DSP_LUT_BITS)
#define RAC_DSP_ATAN_ROM_SIZE   (RAC_DSP_RESIDUAL_START + RAC_DSP_RESIDUAL)

typedef int64_t rac_q_t;

/* Globals — populated by load_*_roms(). Declared extern; defined in
 * rac_dsp_core.c. Callers who want to replace them can assign directly. */
extern uint64_t rac_coarse_lut[RAC_DSP_LUT_SIZE];
extern rac_q_t  rac_atan_rom [RAC_DSP_ATAN_ROM_SIZE];
extern rac_q_t  rac_atanh_tab[RAC_DSP_RESIDUAL];

/* Load one $readmemh-style hex file into a uint64_t array.
 * Returns the number of lines successfully parsed (or -1 on error). */
int  rac_load_hex_lut(const char *path, uint64_t *out, int expected);

/* One-shot helper: loads coarse_lut.mem + atan.mem + atanh.mem into the
 * three globals. Returns 0 on success, non-zero on any load error. */
int  rac_load_all_roms(const char *coarse_lut_path,
                      const char *atan_path,
                      const char *atanh_path);

/* Evaluate one rac_dsp op. op ∈ {0=rotate, 1=project, 2=vectoring,
 * 3=hyperbolic, ...} — mirrors the RTL's op_in encoding. */
void rac_dsp_eval(rac_q_t x_in, rac_q_t y_in, rac_q_t z_in, int op,
                  rac_q_t *x_out, rac_q_t *y_out, rac_q_t *z_out);

/* Batch helper: run N project-mode evaluations in a tight C loop.
 * Inputs are parallel arrays of length `n`; outputs go into x_out[].
 * Avoids Python ctypes overhead when timing from bench_three_paths.py. */
void rac_dsp_project_batch(int n,
                           const rac_q_t *xs, const rac_q_t *zs,
                           rac_q_t       *x_out);

/* Accumulated-sum variant: returns the column sum of N projections as
 * a single scalar (y = Σ rac_dsp_project(xs[i], 0, zs[i]).x_out). This
 * is exactly what each column of the systolic array computes — one
 * call to this gives a full systolic-column-GEMM-equivalent unit. */
rac_q_t rac_dsp_project_sum(int n,
                            const rac_q_t *xs, const rac_q_t *zs);

#endif  /* RAC_DSP_CORE_H */
