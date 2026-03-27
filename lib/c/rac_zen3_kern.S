/*
 * rac_zen3_kern.S — RAC AVX2/FMA Assembly Micro-Kernel for AMD Zen3
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * Hand-tuned 6x16 micro-kernel: 6 rows × 16 columns (2 × YMM)
 * 12 accumulator YMMs + 2 B-load YMMs + software prefetch
 *
 * Calling convention (System V AMD64):
 *   rdi = packed_a  [MR×kc] contiguous
 *   rsi = packed_b  [kc×NR] contiguous
 *   rdx = C pointer [MR rows, ldc stride]
 *   rcx = ldc (C leading dimension, in floats)
 *   r8  = kc (K-dimension of this tile)
 *
 * Register allocation:
 *   ymm0-ymm11:  6×2 accumulators (rows 0-5, lo/hi)
 *   ymm12-ymm13: B loads (lo 8, hi 8)
 *   ymm14:       A broadcast
 *   ymm15:       alpha (loaded at end)
 *
 * Build:
 *   as -o rac_zen3_kern.o rac_zen3_kern.S
 *   # or: cc -c rac_zen3_kern.S -o rac_zen3_kern.o
 */

.text
.globl rac_micro_kernel_6x16_asm
.type  rac_micro_kernel_6x16_asm, @function
.align 32

/*
 * void rac_micro_kernel_6x16_asm(
 *     const float *packed_a,   // rdi  [kc × MR=6]
 *     const float *packed_b,   // rsi  [kc × NR=16]
 *     float *C,                // rdx
 *     int ldc,                 // ecx  (in floats)
 *     int kc                   // r8d
 * );
 */
rac_micro_kernel_6x16_asm:
    /* Convert ldc from floats to bytes */
    movslq  %ecx, %rcx
    shlq    $2, %rcx            /* rcx = ldc * 4 (byte stride) */

    /* Zero accumulators */
    vxorps  %ymm0,  %ymm0,  %ymm0
    vxorps  %ymm1,  %ymm1,  %ymm1
    vxorps  %ymm2,  %ymm2,  %ymm2
    vxorps  %ymm3,  %ymm3,  %ymm3
    vxorps  %ymm4,  %ymm4,  %ymm4
    vxorps  %ymm5,  %ymm5,  %ymm5
    vxorps  %ymm6,  %ymm6,  %ymm6
    vxorps  %ymm7,  %ymm7,  %ymm7
    vxorps  %ymm8,  %ymm8,  %ymm8
    vxorps  %ymm9,  %ymm9,  %ymm9
    vxorps  %ymm10, %ymm10, %ymm10
    vxorps  %ymm11, %ymm11, %ymm11

    /* kc loop counter */
    movl    %r8d, %eax          /* eax = kc */
    testl   %eax, %eax
    jle     .L_store

    /* Prefetch first cache lines of A and B */
    prefetcht0  (%rdi)
    prefetcht0  (%rsi)
    prefetcht0  64(%rsi)

.align 16
.L_kloop:
    /* ── Load B[k, 0..15]: 16 floats = 2 YMM ── */
    vmovups (%rsi), %ymm12          /* B[k, 0..7]  */
    vmovups 32(%rsi), %ymm13        /* B[k, 8..15] */

    /* Prefetch next B row (64 bytes ahead = next k+1 row) */
    prefetcht0  128(%rsi)

    /* ── Row 0: broadcast A[k,0], FMA ── */
    vbroadcastss (%rdi), %ymm14
    vfmadd231ps  %ymm14, %ymm12, %ymm0     /* acc[0][lo] += A[k,0] * B[k,0..7] */
    vfmadd231ps  %ymm14, %ymm13, %ymm1     /* acc[0][hi] += A[k,0] * B[k,8..15] */

    /* ── Row 1 ── */
    vbroadcastss 4(%rdi), %ymm14
    vfmadd231ps  %ymm14, %ymm12, %ymm2
    vfmadd231ps  %ymm14, %ymm13, %ymm3

    /* ── Row 2 ── */
    vbroadcastss 8(%rdi), %ymm14
    vfmadd231ps  %ymm14, %ymm12, %ymm4
    vfmadd231ps  %ymm14, %ymm13, %ymm5

    /* ── Row 3 ── */
    vbroadcastss 12(%rdi), %ymm14
    vfmadd231ps  %ymm14, %ymm12, %ymm6
    vfmadd231ps  %ymm14, %ymm13, %ymm7

    /* ── Row 4 ── */
    vbroadcastss 16(%rdi), %ymm14
    vfmadd231ps  %ymm14, %ymm12, %ymm8
    vfmadd231ps  %ymm14, %ymm13, %ymm9

    /* ── Row 5 ── */
    vbroadcastss 20(%rdi), %ymm14
    vfmadd231ps  %ymm14, %ymm12, %ymm10
    vfmadd231ps  %ymm14, %ymm13, %ymm11

    /* Advance pointers: A += MR*4 = 24 bytes, B += NR*4 = 64 bytes */
    addq    $24, %rdi           /* packed_a += 6 floats */
    addq    $64, %rsi           /* packed_b += 16 floats */

    /* Prefetch next A */
    prefetcht0  (%rdi)

    decl    %eax
    jnz     .L_kloop

.L_store:
    /* ── Store: C[r, :] += acc[r][:] ── */
    /* Row 0 */
    vaddps  (%rdx), %ymm0, %ymm0
    vaddps  32(%rdx), %ymm1, %ymm1
    vmovups %ymm0, (%rdx)
    vmovups %ymm1, 32(%rdx)

    /* Row 1 */
    leaq    (%rdx, %rcx), %rdx         /* C += ldc bytes */
    vaddps  (%rdx), %ymm2, %ymm2
    vaddps  32(%rdx), %ymm3, %ymm3
    vmovups %ymm2, (%rdx)
    vmovups %ymm3, 32(%rdx)

    /* Row 2 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm4, %ymm4
    vaddps  32(%rdx), %ymm5, %ymm5
    vmovups %ymm4, (%rdx)
    vmovups %ymm5, 32(%rdx)

    /* Row 3 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm6, %ymm6
    vaddps  32(%rdx), %ymm7, %ymm7
    vmovups %ymm6, (%rdx)
    vmovups %ymm7, 32(%rdx)

    /* Row 4 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm8, %ymm8
    vaddps  32(%rdx), %ymm9, %ymm9
    vmovups %ymm8, (%rdx)
    vmovups %ymm9, 32(%rdx)

    /* Row 5 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm10, %ymm10
    vaddps  32(%rdx), %ymm11, %ymm11
    vmovups %ymm10, (%rdx)
    vmovups %ymm11, 32(%rdx)

    vzeroupper
    ret

.size rac_micro_kernel_6x16_asm, .-rac_micro_kernel_6x16_asm

/* Mark stack as non-executable */
.section .note.GNU-stack, "", @progbits
