/*
 * rac_avx512_kern.S — RAC AVX-512 Assembly Micro-Kernel
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * 6x32 micro-kernel: 6 rows × 32 columns (2 × ZMM = 32 floats)
 * 12 accumulator ZMMs + 2 B-load ZMMs + 1 A broadcast = 15/32 ZMM used
 * Double the throughput of the AVX2 6x16 kernel per K step.
 *
 * Calling convention (System V AMD64):
 *   rdi = packed_a  [MR×kc] contiguous, MR=6
 *   rsi = packed_b  [kc×NR] contiguous, NR=32
 *   rdx = C pointer
 *   rcx = ldc (in floats)
 *   r8  = kc
 */

.text
.globl rac_micro_kernel_6x32_avx512
.type  rac_micro_kernel_6x32_avx512, @function
.align 64

rac_micro_kernel_6x32_avx512:
    movslq  %ecx, %rcx
    shlq    $2, %rcx            /* ldc in bytes */

    /* Zero 12 accumulator ZMMs */
    vpxord  %zmm0,  %zmm0,  %zmm0
    vpxord  %zmm1,  %zmm1,  %zmm1
    vpxord  %zmm2,  %zmm2,  %zmm2
    vpxord  %zmm3,  %zmm3,  %zmm3
    vpxord  %zmm4,  %zmm4,  %zmm4
    vpxord  %zmm5,  %zmm5,  %zmm5
    vpxord  %zmm6,  %zmm6,  %zmm6
    vpxord  %zmm7,  %zmm7,  %zmm7
    vpxord  %zmm8,  %zmm8,  %zmm8
    vpxord  %zmm9,  %zmm9,  %zmm9
    vpxord  %zmm10, %zmm10, %zmm10
    vpxord  %zmm11, %zmm11, %zmm11

    movl    %r8d, %eax
    testl   %eax, %eax
    jle     .L512_store

    prefetcht0  (%rdi)
    prefetcht0  (%rsi)
    prefetcht0  64(%rsi)

.align 16
.L512_kloop:
    /* Load B[k, 0..31]: 32 floats = 2 ZMM */
    vmovups (%rsi), %zmm12             /* B[k, 0..15]  */
    vmovups 64(%rsi), %zmm13           /* B[k, 16..31] */

    prefetcht0  256(%rsi)

    /* Row 0 */
    vbroadcastss (%rdi), %zmm14
    vfmadd231ps  %zmm14, %zmm12, %zmm0
    vfmadd231ps  %zmm14, %zmm13, %zmm1

    /* Row 1 */
    vbroadcastss 4(%rdi), %zmm14
    vfmadd231ps  %zmm14, %zmm12, %zmm2
    vfmadd231ps  %zmm14, %zmm13, %zmm3

    /* Row 2 */
    vbroadcastss 8(%rdi), %zmm14
    vfmadd231ps  %zmm14, %zmm12, %zmm4
    vfmadd231ps  %zmm14, %zmm13, %zmm5

    /* Row 3 */
    vbroadcastss 12(%rdi), %zmm14
    vfmadd231ps  %zmm14, %zmm12, %zmm6
    vfmadd231ps  %zmm14, %zmm13, %zmm7

    /* Row 4 */
    vbroadcastss 16(%rdi), %zmm14
    vfmadd231ps  %zmm14, %zmm12, %zmm8
    vfmadd231ps  %zmm14, %zmm13, %zmm9

    /* Row 5 */
    vbroadcastss 20(%rdi), %zmm14
    vfmadd231ps  %zmm14, %zmm12, %zmm10
    vfmadd231ps  %zmm14, %zmm13, %zmm11

    addq    $24, %rdi           /* A += MR=6 floats */
    addq    $128, %rsi          /* B += NR=32 floats */

    prefetcht0  (%rdi)

    decl    %eax
    jnz     .L512_kloop

.L512_store:
    /* Row 0 */
    vaddps  (%rdx), %zmm0, %zmm0
    vaddps  64(%rdx), %zmm1, %zmm1
    vmovups %zmm0, (%rdx)
    vmovups %zmm1, 64(%rdx)

    /* Row 1 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %zmm2, %zmm2
    vaddps  64(%rdx), %zmm3, %zmm3
    vmovups %zmm2, (%rdx)
    vmovups %zmm3, 64(%rdx)

    /* Row 2 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %zmm4, %zmm4
    vaddps  64(%rdx), %zmm5, %zmm5
    vmovups %zmm4, (%rdx)
    vmovups %zmm5, 64(%rdx)

    /* Row 3 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %zmm6, %zmm6
    vaddps  64(%rdx), %zmm7, %zmm7
    vmovups %zmm6, (%rdx)
    vmovups %zmm7, 64(%rdx)

    /* Row 4 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %zmm8, %zmm8
    vaddps  64(%rdx), %zmm9, %zmm9
    vmovups %zmm8, (%rdx)
    vmovups %zmm9, 64(%rdx)

    /* Row 5 */
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %zmm10, %zmm10
    vaddps  64(%rdx), %zmm11, %zmm11
    vmovups %zmm10, (%rdx)
    vmovups %zmm11, 64(%rdx)

    vzeroupper
    ret

.size rac_micro_kernel_6x32_avx512, .-rac_micro_kernel_6x32_avx512

.section .note.GNU-stack, "", @progbits
