/*
 * rac_zen3_kern.S — RAC AVX2/FMA Assembly Micro-Kernel (Zen3-tuned)
 * Pinnacle Quantum Group — Michael A. Doran Jr. — March 2026
 *
 * 6x16 micro-kernel with 4x K-unroll for Zen3's dual FMA pipes.
 * 12 accumulator YMMs, software prefetch tuned for Zen3 L1/L2 latency.
 *
 * System V AMD64 ABI:
 *   rdi = packed_a  [kc × MR=6], stride 24 bytes per K
 *   rsi = packed_b  [kc × NR=16], stride 64 bytes per K
 *   rdx = C pointer
 *   ecx = ldc (floats)
 *   r8d = kc
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

    /* Split into groups of 4 + remainder */
    movl    %eax, %r9d
    shrl    $2, %r9d            /* r9 = kc / 4 */
    andl    $3, %eax            /* eax = kc % 4 */
    testl   %r9d, %r9d
    jle     .L_tail

    prefetcht0  (%rsi)
    prefetcht0  64(%rsi)
    prefetcht0  128(%rsi)
    prefetcht0  192(%rsi)

/*
 * Unrolled 4x K-loop body.
 * Each iteration: 6 broadcasts + 2 loads + 12 FMAs = 20 instructions
 * 4 iterations = 80 instructions before branch.
 * Zen3 OOO window is 256 ops — easily overlaps all 4 iterations.
 */
.align 32
.L_kloop4:
    /* ── K+0 ── */
    vmovups     (%rsi), %ymm12
    vmovups     32(%rsi), %ymm13
    vbroadcastss (%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0
    vfmadd231ps %ymm14, %ymm13, %ymm1
    vbroadcastss 4(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3
    vbroadcastss 8(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    vfmadd231ps %ymm14, %ymm13, %ymm5
    vbroadcastss 12(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7
    vbroadcastss 16(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    vfmadd231ps %ymm14, %ymm13, %ymm9
    vbroadcastss 20(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11

    /* ── K+1 ── */
    vmovups     64(%rsi), %ymm12
    vmovups     96(%rsi), %ymm13
    vbroadcastss 24(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0
    vfmadd231ps %ymm14, %ymm13, %ymm1
    vbroadcastss 28(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3
    vbroadcastss 32(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    vfmadd231ps %ymm14, %ymm13, %ymm5
    vbroadcastss 36(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7
    vbroadcastss 40(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    vfmadd231ps %ymm14, %ymm13, %ymm9
    vbroadcastss 44(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11

    /* ── K+2 ── */
    vmovups     128(%rsi), %ymm12
    vmovups     160(%rsi), %ymm13
    prefetcht0  512(%rsi)           /* prefetch B +8 iterations ahead */
    vbroadcastss 48(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0
    vfmadd231ps %ymm14, %ymm13, %ymm1
    vbroadcastss 52(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3
    vbroadcastss 56(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    vfmadd231ps %ymm14, %ymm13, %ymm5
    vbroadcastss 60(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7
    vbroadcastss 64(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    vfmadd231ps %ymm14, %ymm13, %ymm9
    vbroadcastss 68(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11

    /* ── K+3 ── */
    vmovups     192(%rsi), %ymm12
    vmovups     224(%rsi), %ymm13
    prefetcht0  96(%rdi)            /* prefetch A +4 iterations ahead */
    vbroadcastss 72(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0
    vfmadd231ps %ymm14, %ymm13, %ymm1
    vbroadcastss 76(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3
    vbroadcastss 80(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    vfmadd231ps %ymm14, %ymm13, %ymm5
    vbroadcastss 84(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7
    vbroadcastss 88(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    vfmadd231ps %ymm14, %ymm13, %ymm9
    vbroadcastss 92(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11

    /* Advance: A += 4*24=96 bytes, B += 4*64=256 bytes */
    addq    $96, %rdi
    addq    $256, %rsi

    decl    %r9d
    jnz     .L_kloop4

    /* ── Tail: remaining 0-3 K iterations ── */
.L_tail:
    testl   %eax, %eax
    jle     .L_store

.align 16
.L_kloop1:
    vmovups     (%rsi), %ymm12
    vmovups     32(%rsi), %ymm13
    vbroadcastss (%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0
    vfmadd231ps %ymm14, %ymm13, %ymm1
    vbroadcastss 4(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3
    vbroadcastss 8(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    vfmadd231ps %ymm14, %ymm13, %ymm5
    vbroadcastss 12(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7
    vbroadcastss 16(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    vfmadd231ps %ymm14, %ymm13, %ymm9
    vbroadcastss 20(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11
    addq    $24, %rdi
    addq    $64, %rsi
    decl    %eax
    jnz     .L_kloop1

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
