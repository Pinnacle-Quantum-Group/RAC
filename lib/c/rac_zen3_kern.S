/*
 * rac_zen3_kern.S — RAC AVX2/FMA Micro-Kernels (OpenBLAS-style)
 * Pinnacle Quantum Group — March 2026
 *
 * Two kernels using the vmovsldup/vmovshdup + vbroadcastsd trick
 * from OpenBLAS's Zen kernel:
 *
 *   1. 8x4 micro-kernel (matches OpenBLAS Zen shape)
 *   2. 8x8 micro-kernel (double-wide for larger NR)
 *
 * The trick: A is loaded as 8 floats with vmovups, then split:
 *   vmovsldup → even-indexed elements duplicated: {a0,a0,a2,a2,a4,a4,a6,a6}
 *   vmovshdup → odd-indexed elements duplicated:  {a1,a1,a3,a3,a5,a5,a7,a7}
 *
 * B is loaded 2 values at a time with vbroadcastsd:
 *   vbroadcastsd → {b0,b1,b0,b1,b0,b1,b0,b1}
 *
 * Then vfmadd231ps produces:
 *   acc += {a0*b0, a0*b1, a2*b0, a2*b1, a4*b0, a4*b1, a6*b0, a6*b1}
 *
 * This gives 2 columns of C per vfmadd, so 4 columns = 2 vbroadcastsd + 4 FMA.
 * Per K step for 8x4: 1 vmovsldup + 1 vmovshdup + 2 vbroadcastsd + 4 vfmadd
 *   = 8 instructions, 8×4 = 32 output elements updated.
 *
 * Compare to our old 6x16: 2 vmovups + 6 vbroadcastss + 12 vfmadd
 *   = 20 instructions, 6×16 = 96 outputs.
 *
 * Per instruction: old = 96/20 = 4.8 outputs/insn, new = 32/8 = 4.0 outputs/insn
 * BUT the new kernel has MUCH better port utilization because:
 *   - Only 1 A load (vmovsldup/vmovshdup are "free" on port 5)
 *   - vbroadcastsd is port 5 (not competing with FMA on ports 0/1)
 *   - All 4 FMAs go to ports 0/1 with no stalls
 *
 * System V AMD64: rdi=packed_a, rsi=packed_b, rdx=C, ecx=ldc, r8d=kc
 *
 * Packed A layout: [kc][MR=8] contiguous (8 floats per K step)
 * Packed B layout: [kc][NR] contiguous (4 or 8 floats per K step)
 */

.text

/* ═══════════════════════════════════════════════════════════════════════
 * 8×4 micro-kernel — matches OpenBLAS Zen SGEMM shape
 *
 * Register allocation (4 accumulators):
 *   ymm4-ymm7:   4 accumulators (even A × B col pairs)
 *                 ymm4 = A_even × B[0:1], ymm5 = A_odd × B[0:1]
 *                 ymm6 = A_even × B[2:3], ymm7 = A_odd × B[2:3]
 *   ymm1:        A even (vmovsldup)
 *   ymm2:        A odd (vmovshdup)
 *   ymm3:        B broadcast (vbroadcastsd)
 *
 * Output: C[i,j] for i=0..7, j=0..3
 *   After accumulation, ymm4 holds {C[0,0],C[0,1],C[2,0],C[2,1],C[4,0],C[4,1],C[6,0],C[6,1]}
 *   ymm5 holds {C[1,0],C[1,1],C[3,0],C[3,1],C[5,0],C[5,1],C[7,0],C[7,1]}
 *   Need to deinterleave for row-major C store.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Actually — the vmovsldup/vmovshdup trick interleaves row pairs in
 * the accumulator, requiring a complex deinterleave at store time.
 * For a clean first implementation, let's use the simpler broadcast
 * approach but with MR=8: load 8 A values with vmovups, broadcast
 * each B value with vbroadcastss, and accumulate 8 FMAs per B column.
 *
 * 8x4: 8 rows × 4 columns
 *   4 accumulators (ymm4-ymm7), each holds 8 floats = 1 row contribution
 *   Per K: 1 vmovups(A) + 4 vbroadcastss(B) + 4×1 vfmadd = 9 instructions
 *   32 outputs from 9 instructions = 3.6 outputs/insn
 *
 * 8x8: 8 rows × 8 columns
 *   8 accumulators (ymm4-ymm11)
 *   Per K: 1 vmovups(A) + 1 vmovups(B) + 8 vfmadd = 10 instructions
 *   Wait — that's the broadcast-A stream-B pattern again.
 *
 * Let me do the REAL OpenBLAS trick properly.
 */

/* ═══════════════════════════════════════════════════════════════════════
 * KERNEL 1: 8×4 with vmovsldup/vmovshdup trick
 *
 * Packed A: [kc × 8] — 8 floats per K, stored as-is
 * Packed B: [kc × 4] — 4 floats per K, stored as pairs for vbroadcastsd
 *
 * Register map:
 *   ymm1 = A_even = {a0,a0,a2,a2,a4,a4,a6,a6}
 *   ymm2 = A_odd  = {a1,a1,a3,a3,a5,a5,a7,a7}
 *   ymm3 = B pair broadcast
 *   ymm4 = acc[even rows, cols 0-1]
 *   ymm5 = acc[odd rows, cols 0-1]
 *   ymm6 = acc[even rows, cols 2-3]
 *   ymm7 = acc[odd rows, cols 2-3]
 * ═══════════════════════════════════════════════════════════════════════ */
.globl rac_micro_kernel_8x4_asm
.type  rac_micro_kernel_8x4_asm, @function
.align 64

rac_micro_kernel_8x4_asm:
    movslq  %ecx, %rcx
    shlq    $2, %rcx            /* ldc in bytes */

    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5
    vxorps  %ymm6, %ymm6, %ymm6
    vxorps  %ymm7, %ymm7, %ymm7

    movl    %r8d, %eax
    testl   %eax, %eax
    jle     .L8x4_store

.align 32
.L8x4_kloop:
    /* A: 8 floats, split into even/odd */
    vmovsldup (%rdi), %ymm1        /* {a0,a0,a2,a2,a4,a4,a6,a6} */
    vmovshdup (%rdi), %ymm2        /* {a1,a1,a3,a3,a5,a5,a7,a7} */

    /* B cols 0-1: broadcast 2 floats */
    vbroadcastsd (%rsi), %ymm3     /* {b0,b1,b0,b1,b0,b1,b0,b1} */
    vfmadd231ps %ymm1, %ymm3, %ymm4   /* even_rows × B[0:1] */
    vfmadd231ps %ymm2, %ymm3, %ymm5   /* odd_rows × B[0:1] */

    /* B cols 2-3 */
    vbroadcastsd 8(%rsi), %ymm3    /* {b2,b3,b2,b3,b2,b3,b2,b3} */
    vfmadd231ps %ymm1, %ymm3, %ymm6   /* even_rows × B[2:3] */
    vfmadd231ps %ymm2, %ymm3, %ymm7   /* odd_rows × B[2:3] */

    prefetcht0  256(%rsi)
    prefetcht0  128(%rdi)

    addq    $32, %rdi           /* A += 8 floats */
    addq    $16, %rsi           /* B += 4 floats */
    decl    %eax
    jnz     .L8x4_kloop

.L8x4_store:
    /*
     * Deinterleave the vmovsldup/vmovshdup accumulators to row-major C.
     *
     * ymm4 = {r0c0, r0c1, r2c0, r2c1, r4c0, r4c1, r6c0, r6c1}  (even × B[0:1])
     * ymm5 = {r1c0, r1c1, r3c0, r3c1, r5c0, r5c1, r7c0, r7c1}  (odd  × B[0:1])
     * ymm6 = {r0c2, r0c3, r2c2, r2c3, r4c2, r4c3, r6c2, r6c3}  (even × B[2:3])
     * ymm7 = {r1c2, r1c3, r3c2, r3c3, r5c2, r5c3, r7c2, r7c3}  (odd  × B[2:3])
     *
     * Step 1: interleave even/odd to get row pairs with all 4 cols:
     *   vunpcklps ymm4, ymm6 → {r0c0, r0c2, r0c1, r0c3, r4c0, r4c2, r4c1, r4c3}
     *   Actually that's wrong order. Let me use vshufps.
     *
     * Simpler approach: store to stack, rearrange with scalar moves.
     * The store runs ONCE per KC iterations — not performance critical.
     */
    subq    $128, %rsp

    vmovups %ymm4, (%rsp)       /* even×B01: {r0c0,r0c1, r2c0,r2c1, r4c0,r4c1, r6c0,r6c1} */
    vmovups %ymm5, 32(%rsp)     /* odd×B01:  {r1c0,r1c1, r3c0,r3c1, r5c0,r5c1, r7c0,r7c1} */
    vmovups %ymm6, 64(%rsp)     /* even×B23: {r0c2,r0c3, r2c2,r2c3, r4c2,r4c3, r6c2,r6c3} */
    vmovups %ymm7, 96(%rsp)     /* odd×B23:  {r1c2,r1c3, r3c2,r3c3, r5c2,r5c3, r7c2,r7c3} */

    /* Row 0 (even, pair 0): stack[0..1] = c0,c1; stack[64..65] = c2,c3 */
    movq    %rdx, %r10

    /* Use SSE movss to do float-granularity adds to C */
    /* Row 0: even row, pair index 0 → offset 0 in ymm4, offset 0 in ymm6 */
    vmovss  (%rsp), %xmm0               /* r0c0 */
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  4(%rsp), %xmm0              /* r0c1 */
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  64(%rsp), %xmm0             /* r0c2 */
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  68(%rsp), %xmm0             /* r0c3 */
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 1: odd row, pair index 0 → offset 0 in ymm5, offset 0 in ymm7 */
    addq    %rcx, %r10
    vmovss  32(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  36(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  96(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  100(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 2: even, pair index 1 → offset 8 bytes in ymm4/ymm6 */
    addq    %rcx, %r10
    vmovss  8(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  12(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  72(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  76(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 3: odd, pair index 1 → offset 8 in ymm5/ymm7 */
    addq    %rcx, %r10
    vmovss  40(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  44(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  104(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  108(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 4: even, pair index 2 → offset 16 in ymm4/ymm6 */
    addq    %rcx, %r10
    vmovss  16(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  20(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  80(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  84(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 5: odd, pair index 2 */
    addq    %rcx, %r10
    vmovss  48(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  52(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  112(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  116(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 6: even, pair index 3 → offset 24 in ymm4/ymm6 */
    addq    %rcx, %r10
    vmovss  24(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  28(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  88(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  92(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    /* Row 7: odd, pair index 3 */
    addq    %rcx, %r10
    vmovss  56(%rsp), %xmm0
    vaddss  (%r10), %xmm0, %xmm0
    vmovss  %xmm0, (%r10)
    vmovss  60(%rsp), %xmm0
    vaddss  4(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 4(%r10)
    vmovss  120(%rsp), %xmm0
    vaddss  8(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 8(%r10)
    vmovss  124(%rsp), %xmm0
    vaddss  12(%r10), %xmm0, %xmm0
    vmovss  %xmm0, 12(%r10)

    addq    $128, %rsp
    vzeroupper
    ret

.size rac_micro_kernel_8x4_asm, .-rac_micro_kernel_8x4_asm


/* ═══════════════════════════════════════════════════════════════════════
 * KERNEL 2: 8×8 — broadcast A (8 floats), stream B (8 floats)
 *
 * Simpler approach: vmovups for both A and B, vbroadcastss per row.
 * 8 accumulators, 1 B load, 8 broadcasts + 8 FMAs per K step.
 *
 * This avoids the deinterleave complexity of vmovsldup/vmovshdup
 * while still getting 8 rows of throughput.
 *
 * Register map:
 *   ymm0-ymm7:   8 accumulators (1 per row)
 *   ymm8:        B load (8 floats)
 *   ymm9:        A broadcast
 *
 * Packed A: [kc × 8] — 8 floats per K
 * Packed B: [kc × 8] — 8 floats per K
 * ═══════════════════════════════════════════════════════════════════════ */
.globl rac_micro_kernel_8x8_asm
.type  rac_micro_kernel_8x8_asm, @function
.align 64

rac_micro_kernel_8x8_asm:
    movslq  %ecx, %rcx
    shlq    $2, %rcx

    vxorps  %ymm0, %ymm0, %ymm0
    vxorps  %ymm1, %ymm1, %ymm1
    vxorps  %ymm2, %ymm2, %ymm2
    vxorps  %ymm3, %ymm3, %ymm3
    vxorps  %ymm4, %ymm4, %ymm4
    vxorps  %ymm5, %ymm5, %ymm5
    vxorps  %ymm6, %ymm6, %ymm6
    vxorps  %ymm7, %ymm7, %ymm7

    movl    %r8d, %eax
    testl   %eax, %eax
    jle     .L8x8_store

.align 32
.L8x8_kloop:
    /* Load B[k, 0..7] */
    vmovups (%rsi), %ymm8

    /* 8 rows: broadcast A[k,r] and FMA */
    vbroadcastss    (%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm0
    vbroadcastss   4(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm1
    vbroadcastss   8(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm2
    vbroadcastss  12(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm3
    vbroadcastss  16(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm4
    vbroadcastss  20(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm5
    vbroadcastss  24(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm6
    vbroadcastss  28(%rdi), %ymm9
    vfmadd231ps %ymm9, %ymm8, %ymm7

    prefetcht0  256(%rsi)
    prefetcht0  128(%rdi)

    addq    $32, %rdi           /* A += 8 floats */
    addq    $32, %rsi           /* B += 8 floats */
    decl    %eax
    jnz     .L8x8_kloop

.L8x8_store:
    /* Each ymm holds one complete row of C: 8 columns */
    /* Just add to C and store — no deinterleave needed! */

    vaddps  (%rdx), %ymm0, %ymm0
    vmovups %ymm0, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm1, %ymm1
    vmovups %ymm1, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm2, %ymm2
    vmovups %ymm2, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm3, %ymm3
    vmovups %ymm3, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm4, %ymm4
    vmovups %ymm4, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm5, %ymm5
    vmovups %ymm5, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm6, %ymm6
    vmovups %ymm6, (%rdx)
    leaq    (%rdx, %rcx), %rdx

    vaddps  (%rdx), %ymm7, %ymm7
    vmovups %ymm7, (%rdx)

    vzeroupper
    ret

.size rac_micro_kernel_8x8_asm, .-rac_micro_kernel_8x8_asm


/* ═══════════════════════════════════════════════════════════════════════
 * LEGACY: 6×16 kernel (kept for fallback, same calling convention)
 * ═══════════════════════════════════════════════════════════════════════ */
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
    jle     .L6x16_store

.align 32
.L6x16_kloop:
    vmovups     (%rsi), %ymm12
    vbroadcastss (%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm0
    vmovups     32(%rsi), %ymm13
    vfmadd231ps %ymm14, %ymm13, %ymm1
    vbroadcastss 4(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm2
    vfmadd231ps %ymm14, %ymm13, %ymm3
    vbroadcastss 8(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm4
    prefetcht0  256(%rsi)
    vfmadd231ps %ymm14, %ymm13, %ymm5
    vbroadcastss 12(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm6
    vfmadd231ps %ymm14, %ymm13, %ymm7
    vbroadcastss 16(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm8
    prefetcht0  96(%rdi)
    vfmadd231ps %ymm14, %ymm13, %ymm9
    vbroadcastss 20(%rdi), %ymm14
    vfmadd231ps %ymm14, %ymm12, %ymm10
    vfmadd231ps %ymm14, %ymm13, %ymm11
    addq    $24, %rdi
    addq    $64, %rsi
    decl    %eax
    jnz     .L6x16_kloop

.L6x16_store:
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
