/*
 * rac_systolic_ref.c — software mirror of rac_systolic_array.v GEMM.
 *
 * Dataflow (weight-stationary, N×N PE grid):
 *   - PE[r][c] holds a pre-programmed angle (from gemm_weights.hex)
 *   - Activation x[r] enters row r on the left edge
 *   - Each PE computes x_proj = project(x[r], 0, pe_angle[r][c])
 *                             = x[r] * cos(pe_angle[r][c]) * K_CORDIC
 *   - Column c output: y[c] = Σ_r x_proj[r][c]
 *
 * Caller pre-scales x by K_INV so post-CORDIC y[c] = Σ x[r] · W[r,c].
 *
 * This C reference uses rac_dsp_eval() from rac_dsp_core.c — identical
 * bit-for-bit to what rac_dsp.v does. Any divergence between the RTL
 * testbench output and this reference's output is an RTL bug, not an
 * algorithmic one.
 *
 * Build:  cc -O2 -std=c99 -Wall -Wextra -o rac_systolic_ref \
 *             rac_systolic_ref.c rac_dsp_core.c -lm
 * Run:    ./rac_systolic_ref N \
 *             gemm_weights.hex gemm_input.hex \
 *             cordic_coarse_lut.mem cordic_atan.mem cordic_atanh.mem \
 *             > ref_gemm_outputs.hex
 */

#include "rac_dsp_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

static int load_q_array(const char *path, rac_q_t *out, int expected) {
    uint64_t *buf = calloc(expected, sizeof(uint64_t));
    if (!buf) return -1;
    int n = rac_load_hex_lut(path, buf, expected);
    for (int i = 0; i < expected; i++) out[i] = (rac_q_t)buf[i];
    free(buf);
    return n;
}

static void systolic_simulate(int N, const rac_q_t *W_angles,
                              const rac_q_t *x_in, rac_q_t *y_out) {
    for (int c = 0; c < N; c++) {
        rac_q_t sum = 0;
        for (int r = 0; r < N; r++) {
            rac_q_t xo, yo, zo;
            rac_q_t angle = W_angles[r * N + c];
            rac_dsp_eval(x_in[r], 0, angle, /*op=project*/ 1,
                         &xo, &yo, &zo);
            sum += xo;
        }
        y_out[c] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
            "usage: %s N weights.hex input.hex "
            "coarse_lut.mem atan.mem atanh.mem [K]\n"
            "  If K > 1 and input.hex has K·N entries, runs matrix-matrix\n"
            "  Y = W·X_batch; else single matrix-vector y = W·x.\n",
            argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int K = (argc >= 8) ? atoi(argv[7]) : 1;
    if (N < 1 || N > 256) { fprintf(stderr, "bad N=%d\n", N); return 2; }
    if (K < 1 || K > 256) { fprintf(stderr, "bad K=%d\n", K); return 2; }

    int rc = rac_load_all_roms(argv[4], argv[5], argv[6]);
    if (rc != 0) return rc + 2;

    rac_q_t *W = calloc((size_t)N * N,     sizeof(rac_q_t));
    rac_q_t *X = calloc((size_t)K * N,     sizeof(rac_q_t));
    rac_q_t *Y = calloc((size_t)K * N,     sizeof(rac_q_t));
    if (!W || !X || !Y) { fprintf(stderr, "oom\n"); return 6; }

    if (load_q_array(argv[2], W, N * N) < 0) {
        fprintf(stderr, "weights load failed\n"); return 7;
    }
    if (load_q_array(argv[3], X, K * N) < 0) {
        fprintf(stderr, "input load failed\n"); return 8;
    }

    for (int k = 0; k < K; k++) {
        systolic_simulate(N, W, &X[k * N], &Y[k * N]);
    }

    printf("// rac_systolic_ref outputs — N=%d K=%d\n", N, K);
    printf("// format: Y[k][c] in Q32.32 signed (K blocks of N)\n");
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < N; c++) {
            printf("%016" PRIx64 "\n", (uint64_t)Y[k * N + c]);
        }
    }

    free(W); free(X); free(Y);
    return 0;
}
