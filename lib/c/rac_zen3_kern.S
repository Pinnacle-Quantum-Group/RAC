/*
 * rac_zen3_kern.S — RAC AVX2/FMA Assembly Micro-Kernel (Zen3-tuned)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * 6x16 micro-kernel with interleaved B-hi load.
 * Key optimization: B[k] hi loaded BETWEEN row0-lo FMA and row0-hi FMA,
 * hiding the load latency behind the first FMA's execution.
 *
 * System V AMD64: rdi=packed_a, rsi=packed_b, rdx=C, ecx=ldc, r8d=kc
 */

.text
.globl rac_micro_kernel_6x16_asm
.type  rac_micro_kernel_6x16_asm, @function
.align 64

rac_micro_kernel_6x16_asm:
    movslq  %ecx, %rcx
    shlq    $2, %rcx

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

    movl    %r8d, %eax
    testl   %eax, %eax
    jle     .L_store

.align 32
.L_kloop:
    /*
     * Interleaved load: B-lo first, start row0 FMA on lo half,
     * load B-hi while row0-lo FMA executes (4-cycle latency on Zen3),
     * then row0-hi FMA uses the now-ready B-hi.
     *
     * This overlaps the B-hi load with the B-lo FMA execution.
     */

    /* Load B[k] lo half */
    vmovups     (%rsi), %ymm12

    /* Row 0: broadcast + FMA on lo half — B-hi loads in shadow */
    vbroadcastss (%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0      /* row0 lo — starts 4-cycle FMA */
    vmovups     32(%rsi), %ymm13             /* B[k] hi — loads during FMA latency */
    vfmadd231ps %ymm14, %ymm13, %ymm1      /* row0 hi — ymm13 ready by now */

    /* Row 1 */
    vbroadcastss 4(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3

    /* Row 2 + prefetch B ahead */
    vbroadcastss 8(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    prefetcht0  256(%rsi)
    vfmadd231ps %ymm14, %ymm13, %ymm5

    /* Row 3 */
    vbroadcastss 12(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7

    /* Row 4 + prefetch A ahead */
    vbroadcastss 16(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    prefetcht0  96(%rdi)
    vfmadd231ps %ymm14, %ymm13, %ymm9

    /* Row 5 */
    vbroadcastss 20(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11

    addq    $24, %rdi
    addq    $64, %rsi
    decl    %eax
    jnz     .L_kloop

.L_store:
    vaddps  (%rdx), %ymm0, %ymm0
    vaddps  32(%rdx), %ymm1, %ymm1
    vmovups %ymm0, (%rdx)
    vmovups %ymm1, 32(%rdx)
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm2, %ymm2
    vaddps  32(%rdx), %ymm3, %ymm3
    vmovups %ymm2, (%rdx)
    vmovups %ymm3, 32(%rdx)
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm4, %ymm4
    vaddps  32(%rdx), %ymm5, %ymm5
    vmovups %ymm4, (%rdx)
    vmovups %ymm5, 32(%rdx)
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm6, %ymm6
    vaddps  32(%rdx), %ymm7, %ymm7
    vmovups %ymm6, (%rdx)
    vmovups %ymm7, 32(%rdx)
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm8, %ymm8
    vaddps  32(%rdx), %ymm9, %ymm9
    vmovups %ymm8, (%rdx)
    vmovups %ymm9, 32(%rdx)
    leaq    (%rdx, %rcx), %rdx
    vaddps  (%rdx), %ymm10, %ymm10
    vaddps  32(%rdx), %ymm11, %ymm11
    vmovups %ymm10, (%rdx)
    vmovups %ymm11, 32(%rdx)

    vzeroupper
    ret

.size rac_micro_kernel_6x16_asm, .-rac_micro_kernel_6x16_asm
.section .note.GNU-stack, "", @progbits
