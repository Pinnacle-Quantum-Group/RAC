/*
 * rac_mpi.c — RAC Distributed SGEMM via MPI + OpenMP
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Distributes matmul across MPI ranks by row-partitioning A.
 * Each rank computes its slice of C = A_slice @ B using OpenMP + AVX2.
 * Results gathered to rank 0.
 *
 * Build:
 *   mpicc -O3 -mavx2 -mfma -fopenmp \
 *     rac_mpi.c rac_cpu.c rac_avx2.c -lm -o rac_mpi
 *
 * Run:
 *   mpirun -np 4 ./rac_mpi 1024 1024 1024
 */

#include "rac_cpu.h"
#include "rac_avx2.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Distributed SGEMM ──────────────────────────────────────────────────── */

/*
 * rac_distributed_sgemm:
 *   C[M,N] = A[M,K] @ B[K,N]
 *
 * Strategy:
 *   1. Broadcast B to all ranks (B is needed in full by every rank).
 *   2. Scatter rows of A across ranks (each gets M/nproc rows).
 *   3. Each rank computes C_local = A_local @ B using AVX2 + OpenMP.
 *   4. Gather C_local rows back to rank 0.
 *
 * Memory: each rank stores A_local[M_local, K] + B[K, N] + C_local[M_local, N].
 * For M=4096, K=4096, N=4096, float32:
 *   With 4 ranks: ~192MB per rank instead of 192MB total.
 */
rac_status rac_distributed_sgemm(
    const float *A,     /* [M, K] on rank 0, NULL on others */
    const float *B_in,  /* [K, N] on rank 0, NULL on others */
    float *C,           /* [M, N] on rank 0, NULL on others */
    int M, int N, int K,
    MPI_Comm comm)
{
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    /* ── Compute per-rank row counts ── */
    int base_rows = M / nproc;
    int remainder = M % nproc;
    int M_local = base_rows + (rank < remainder ? 1 : 0);

    int *sendcounts_a = NULL, *displs_a = NULL;
    int *sendcounts_c = NULL, *displs_c = NULL;

    if (rank == 0) {
        sendcounts_a = (int*)malloc(nproc * sizeof(int));
        displs_a     = (int*)malloc(nproc * sizeof(int));
        sendcounts_c = (int*)malloc(nproc * sizeof(int));
        displs_c     = (int*)malloc(nproc * sizeof(int));

        int offset_a = 0, offset_c = 0;
        for (int r = 0; r < nproc; r++) {
            int rows_r = base_rows + (r < remainder ? 1 : 0);
            sendcounts_a[r] = rows_r * K;
            displs_a[r] = offset_a;
            sendcounts_c[r] = rows_r * N;
            displs_c[r] = offset_c;
            offset_a += rows_r * K;
            offset_c += rows_r * N;
        }
    }

    /* ── Allocate local buffers ── */
    float *A_local = (float*)malloc((size_t)M_local * K * sizeof(float));
    float *B_full  = (float*)malloc((size_t)K * N * sizeof(float));
    float *C_local = (float*)malloc((size_t)M_local * N * sizeof(float));

    if (!A_local || !B_full || !C_local) {
        free(A_local); free(B_full); free(C_local);
        return RAC_ERR_ALLOC;
    }

    /* ── Scatter A rows ── */
    MPI_Scatterv(A, sendcounts_a, displs_a, MPI_FLOAT,
                 A_local, M_local * K, MPI_FLOAT,
                 0, comm);

    /* ── Broadcast B ── */
    if (rank == 0)
        memcpy(B_full, B_in, (size_t)K * N * sizeof(float));
    MPI_Bcast(B_full, K * N, MPI_FLOAT, 0, comm);

    /* ── Local compute: C_local = A_local @ B_full ── */
    rac_config cfg = rac_default_config();

    if (rac_has_avx2()) {
        rac_sgemm_avx2(A_local, B_full, C_local,
                        M_local, N, K, 1.0f, 0.0f, &cfg);
    } else {
        rac_sgemm(A_local, B_full, C_local,
                   M_local, N, K, 1.0f, 0.0f, &cfg);
    }

    /* ── Gather C rows ── */
    MPI_Gatherv(C_local, M_local * N, MPI_FLOAT,
                C, sendcounts_c, displs_c, MPI_FLOAT,
                0, comm);

    /* ── Cleanup ── */
    free(A_local);
    free(B_full);
    free(C_local);
    if (rank == 0) {
        free(sendcounts_a); free(displs_a);
        free(sendcounts_c); free(displs_c);
    }

    return RAC_OK;
}

/* ── Main: standalone benchmark ─────────────────────────────────────────── */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : M;
    int K = (argc > 3) ? atoi(argv[3]) : M;

    float *A = NULL, *B = NULL, *C = NULL;

    if (rank == 0) {
        A = (float*)malloc((size_t)M * K * sizeof(float));
        B = (float*)malloc((size_t)K * N * sizeof(float));
        C = (float*)malloc((size_t)M * N * sizeof(float));

        srand(42);
        for (int i = 0; i < M * K; i++) A[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        for (int i = 0; i < K * N; i++) B[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;

        printf("RAC MPI SGEMM — %d ranks, %dx%d @ %dx%d\n", nproc, M, K, K, N);
        printf("AVX2: %s\n", rac_has_avx2() ? "yes" : "no");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int iters = 5;
    for (int iter = 0; iter < iters; iter++) {
        rac_distributed_sgemm(A, B, C, M, N, K, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = (MPI_Wtime() - t0) / iters;

    if (rank == 0) {
        double ops = 2.0 * M * N * K;
        double gflops = ops / (elapsed * 1e9);
        printf("Time: %.3f ms  GFLOPS: %.2f  (%.4f TOPS)\n",
               elapsed * 1000, gflops, gflops / 1000);
        free(A); free(B); free(C);
    }

    MPI_Finalize();
    return 0;
}
