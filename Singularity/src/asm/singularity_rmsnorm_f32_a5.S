/*
 * Copyright (C) 2026 Kival
 * This file is part of the Arkantos AI Stack Suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

.syntax unified
.arch armv7-a
.fpu neon-vfpv4
.arm
.text

/* ----------------------------------------------------------------------------
 * void singularity_rmsnorm_f32_a5(
 *     const float* x,      // r0: input vector (n floats)
 *     const float* weight, // r1: scale weight vector (n floats)
 *     float* out,          // r2: output vector (n floats, may alias x)
 *     int n,               // r3: vector dimension
 *     float eps            // s0: epsilon (e.g. 1e-5f)
 * )
 * ---------------------------------------------------------------------------- */
.global singularity_rmsnorm_f32_a5
.type singularity_rmsnorm_f32_a5, %function
.p2align 5
singularity_rmsnorm_f32_a5:
    cmp         r3, #0
    bxle        lr

    push        {r4, lr}

    /* Save eps (s0) in s15 (d7[1], caller-saved scratch) */
    vmov.f32    s15, s0

    /* Initialize sum-of-squares accumulator s8 = 0.0f */
    vmov.i32    d4, #0                  /* s8 = 0.0f, s9 = 0.0f */

    mov         r12, r0                 /* r12 = cursor over x */
    bic         r4, r3, #7              /* r4  = n & ~7 */
    cmp         r4, #0
    beq         .Lrms_sum_tail_init
    add         r4, r0, r4, lsl #2      /* r4  = end pointer for 8x loop */
    pld         [r12, #128]

.p2align 4
.Lrms_sum_loop8:
    vldr        s0, [r12, #0]
    vfma.f32    s8, s0, s0
    vldr        s1, [r12, #4]
    vfma.f32    s8, s1, s1
    vldr        s2, [r12, #8]
    vfma.f32    s8, s2, s2
    vldr        s3, [r12, #12]
    vfma.f32    s8, s3, s3
    vldr        s0, [r12, #16]
    vfma.f32    s8, s0, s0
    vldr        s1, [r12, #20]
    vfma.f32    s8, s1, s1
    vldr        s2, [r12, #24]
    vfma.f32    s8, s2, s2
    vldr        s3, [r12, #28]
    add         r12, r12, #32
    vfma.f32    s8, s3, s3
    cmp         r4, r12
    bne         .Lrms_sum_loop8

.Lrms_sum_tail_init:
    add         r4, r0, r3, lsl #2      /* r4 = exact end pointer of x */
    cmp         r4, r12
    beq         .Lrms_reduce_sum
.Lrms_sum_scalar:
    vldmia      r12!, {s0}
    vfma.f32    s8, s0, s0
    cmp         r4, r12
    bne         .Lrms_sum_scalar

.Lrms_reduce_sum:
    /* Fast IEEE-754 exact reciprocal for power-of-two n: 1.0f/n = (96 + clz(n)) << 23 */
    sub         r12, r3, #1
    tst         r3, r12
    bne         .Lrms_non_pow2_n
    clz         r12, r3
    add         r12, r12, #96
    lsl         r12, r12, #23
    vmov        s4, r12
    b           .Lrms_rsqrt

.Lrms_non_pow2_n:
    /* General n fallback: compute 1.0f / n via NEON vrecpe + 2x vrecps */
    vmov        s2, r3
    vcvt.f32.s32 s2, s2
    vrecpe.f32  d2, d1                  /* s4 = ~1.0f / n */
    vrecps.f32  d3, d1, d2
    vmul.f32    s4, s4, s6
    vrecps.f32  d3, d1, d2
    vmul.f32    s4, s4, s6              /* s4 = 1.0f / n */

.Lrms_rsqrt:
    /* s15 = eps + sum_sq * (1.0f / n) */
    vmla.f32    s15, s8, s4
    vmov.f32    s12, s15                /* d6[0] = s12 = mean_sq + eps */

    /* Compute inv_rms = 1.0f / sqrt(s12) via NEON vrsqrte + Newton-Raphson vrsqrts */
    vrsqrte.f32 d7, d6                  /* s14 = ~1.0f / sqrt(s12) */
    vmul.f32    s8, s14, s12
    vrsqrts.f32 d4, d4, d7
    vmul.f32    s14, s14, s8            /* s14 = inv_rms (< 1.5e-5 rel error) */

    bic         r12, r3, #7             /* r12 = n & ~7 */
    cmp         r12, #0
    beq         .Lrms_scale_tail_init
    add         r12, r0, r12, lsl #2    /* r12 = end pointer for 8x scale loop */
    pld         [r0, #128]
    pld         [r1, #128]

.p2align 4
.Lrms_scale_loop8:
    vldr        s0, [r0, #0]
    vldr        s4, [r1, #0]
    vmul.f32    s0, s0, s14
    vmul.f32    s0, s0, s4
    vstr        s0, [r2, #0]

    vldr        s1, [r0, #4]
    vldr        s5, [r1, #4]
    vmul.f32    s1, s1, s14
    vmul.f32    s1, s1, s5
    vstr        s1, [r2, #4]

    vldr        s2, [r0, #8]
    vldr        s6, [r1, #8]
    vmul.f32    s2, s2, s14
    vmul.f32    s2, s2, s6
    vstr        s2, [r2, #8]

    vldr        s3, [r0, #12]
    vldr        s7, [r1, #12]
    vmul.f32    s3, s3, s14
    vmul.f32    s3, s3, s7
    vstr        s3, [r2, #12]

    vldr        s0, [r0, #16]
    vldr        s4, [r1, #16]
    vmul.f32    s0, s0, s14
    vmul.f32    s0, s0, s4
    vstr        s0, [r2, #16]

    vldr        s1, [r0, #20]
    vldr        s5, [r1, #20]
    vmul.f32    s1, s1, s14
    vmul.f32    s1, s1, s5
    vstr        s1, [r2, #20]

    vldr        s2, [r0, #24]
    vldr        s6, [r1, #24]
    vmul.f32    s2, s2, s14
    vmul.f32    s2, s2, s6
    vstr        s2, [r2, #24]

    vldr        s3, [r0, #28]
    vldr        s7, [r1, #28]
    add         r0, r0, #32
    add         r1, r1, #32
    vmul.f32    s3, s3, s14
    vmul.f32    s3, s3, s7
    vstr        s3, [r2, #28]
    add         r2, r2, #32

    cmp         r12, r0
    bne         .Lrms_scale_loop8

.Lrms_scale_tail_init:
    cmp         r4, r0
    beq         .Lrms_done
.Lrms_scale_scalar:
    vldmia      r0!, {s0}
    vldmia      r1!, {s4}
    vmul.f32    s0, s0, s14
    vmul.f32    s0, s0, s4
    vstmia      r2!, {s0}
    cmp         r4, r0
    bne         .Lrms_scale_scalar

.Lrms_done:
    pop         {r4, pc}
.size singularity_rmsnorm_f32_a5, .-singularity_rmsnorm_f32_a5

.section .note.GNU-stack,"",%progbits
