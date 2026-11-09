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
 * void singularity_softmax_f32_a5(
 *     const float* x,      // r0: input logits (n floats)
 *     float* out,          // r1: output probabilities (n floats, may alias x)
 *     int n                // r2: number of elements
 * )
 * ---------------------------------------------------------------------------- */
.global singularity_softmax_f32_a5
.type singularity_softmax_f32_a5, %function
.p2align 5
singularity_softmax_f32_a5:
    push    {r4, r5, r6, lr}
    vpush   {d8-d15}

    cmp     r2, #0
    ble     .Lsm_done

    /* ==================== PASS 1: Find max(x[0..n-1]) ==================== */
    ldr         r4, .Lsm_neg_inf
    vdup.32     q8, r4                  /* q8 = [-inf, -inf, -inf, -inf] */

    mov         r4, r2                  /* r4 = count */
    mov         r5, r0                  /* r5 = x cursor */

    cmp         r4, #8
    blt         .Lsm_max_tail4

.p2align 4
.Lsm_max_loop8:
    pld         [r5, #128]
    vld1.32     {d0, d1, d2, d3}, [r5]! /* Load 8 floats (q0, q1) */
    sub         r4, r4, #8
    cmp         r4, #8
    vmax.f32    q0, q0, q1              /* q0, q1 loaded 2 insns ago */
    vmax.f32    q8, q8, q0
    bge         .Lsm_max_loop8

.Lsm_max_tail4:
    cmp         r4, #4
    blt         .Lsm_max_tail1
    vld1.32     {d0, d1}, [r5]!
    sub         r4, r4, #4
    cmp         r4, #0
    vmax.f32    q8, q8, q0

.Lsm_max_tail1:
    vpmax.f32   d16, d16, d17
    vpmax.f32   d16, d16, d16           /* d16 = [max_val, max_val] */

    cmp         r4, #0
    ble         .Lsm_pass2_init
.Lsm_max_scalar:
    vld1.32     {d0[0]}, [r5]!
    subs        r4, r4, #1
    vdup.32     d0, d0[0]
    vmax.f32    d16, d16, d0
    bgt         .Lsm_max_scalar

.Lsm_pass2_init:
    /* Broadcast max_val into q4 (d8, d9) */
    vdup.32     q4, d16[0]

    /* Load polynomial & range-reduction constants:
     *   q4 (d8, d9):   max_val
     *   q5 (d10, d11): clamp_lo = -87.3f
     *   q6 (d12, d13): log2(e)  = 1.44269504f
     *   q7 (d14, d15): ln2      = 0.69314718f
     *   q9 (d18, d19): c3 = 1/6
     *   q10 (d20, d21): 0.5f
     *   q11 (d22, d23): 127 (int32)
     *   q12 (d24, d25): c4 = 1/24
     *   q13 (d26, d27): -0.5f
     *   q14 (d28, d29): 1.0f
     *   q15 (d30, d31): 0 (int32)
     */
    ldr         r4, .Lsm_clamp_lo
    vdup.32     q5, r4
    ldr         r4, .Lsm_log2e
    vdup.32     q6, r4
    adr         r4, .Lsm_coeffs
    vld1.32     {d14, d15}, [r4]        /* d14=[ln2_hi, ln2_lo], d15=[c5, c4] */
    ldr         r4, .Lsm_c3
    vdup.32     q9, r4                  /* q9  = c3 */
    vmov.f32    q10, #0.5               /* q10 = 0.5f */
    vmov.i32    q11, #127               /* q11 = 127 */
    vdup.32     q12, d15[1]             /* q12 = c4 */
    vneg.f32    q13, q10                /* q13 = -0.5f */
    ldr         r4, .Lsm_ln2_full
    vdup.32     q7, r4                  /* q7  = ln2 = 0.69314718f */
    vmov.f32    q14, #1.0               /* q14 = 1.0f */
    vmov.i32    q15, #0                 /* q15 = 0 */

    /* Zero sum accumulator q8 (float32x4) */
    vmov.i32    q8, #0

    mov         r4, r2                  /* r4 = count */
    mov         r5, r0                  /* r5 = x cursor */
    mov         r6, r1                  /* r6 = out cursor */

    cmp         r4, #4
    blt         .Lsm_exp_tail

.p2align 4
.Lsm_exp_loop4:
    pld         [r5, #128]
    vldmia      r5!, {d0, d1}           /* 64-bit burst-load 4 floats into q0 */

    /* z = max(x - max_val, -87.3f) <= 0.0f */
    vsub.f32    q0, q0, q4
    vmax.f32    q0, q0, q5

    /* Since z <= 0, k = round(z * log2(e)) = int32(z * log2(e) - 0.5f) */
    vmov        q1, q13                 /* q1 = -0.5f */
    vfma.f32    q1, q0, q6              /* q1 = z * log2(e) - 0.5f */
    vcvt.s32.f32 q2, q1                 /* q2 = integer k */
    vcvt.f32.s32 q1, q2                 /* q1 = float k */

    /* Fused range reduction: r = z - k * ln2 */
    vfms.f32    q0, q1, q7

    /* Degree-4 Horner polynomial on r (q0) using fused vfma.f32 */
    vmov        q1, q9                  /* q1 = c3 */
    vfma.f32    q1, q0, q12             /* q1 = c3 + r * c4 */
    vmov        q3, q10                 /* q3 = 0.5f */
    vfma.f32    q3, q0, q1              /* q3 = 0.5 + r * q1 */
    vmov        q1, q14                 /* q1 = 1.0f */
    vfma.f32    q1, q0, q3              /* q1 = 1.0 + r * q3 */
    vmov        q3, q14                 /* q3 = 1.0f */
    vfma.f32    q3, q0, q1              /* q3 = p(r) */

    /* Reconstruct 2^k * p(r) */
    vadd.s32    q2, q2, q11             /* k + 127 */
    vmax.s32    q2, q2, q15             /* clamp >= 0 */
    vshl.i32    q2, q2, #23             /* 2^k in IEEE-754 bits */
    vmul.f32    q3, q3, q2              /* q3 = exp(z) */

    vadd.f32    q8, q8, q3              /* Accumulate sum += exp(z) */
    vstmia      r6!, {d6, d7}           /* 64-bit burst-store 4 exp values */
    subs        r4, r4, #4
    cmp         r4, #4
    bge         .Lsm_exp_loop4

.Lsm_exp_tail:
    cmp         r4, #0
    ble         .Lsm_pass3_norm

    /* Handle 1..3 remaining elements using the same NEON polynomial */
    vmov        q0, q5                  /* Default unused lanes to -87.3f */
    vld1.32     {d0[0]}, [r5]!
    cmp         r4, #2
    blt         .Lsm_tail_eval
    vld1.32     {d0[1]}, [r5]!
    beq         .Lsm_tail_eval
    vld1.32     {d1[0]}, [r5]!

.Lsm_tail_eval:
    vsub.f32    q0, q0, q4
    vmax.f32    q0, q0, q5

    vmov        q1, q13
    vfma.f32    q1, q0, q6
    vcvt.s32.f32 q2, q1
    vcvt.f32.s32 q1, q2

    vfms.f32    q0, q1, q7

    vmov        q1, q9
    vfma.f32    q1, q0, q12
    vmov        q3, q10
    vfma.f32    q3, q0, q1
    vmov        q1, q14
    vfma.f32    q1, q0, q3
    vmov        q3, q14
    vfma.f32    q3, q0, q1

    vadd.s32    q2, q2, q11
    vmax.s32    q2, q2, q15
    vshl.i32    q2, q2, #23
    vmul.f32    q3, q3, q2              /* q3 (s12, s13, s14, s15) = exp(z) */

    /* Store and accumulate only the valid 1..3 lanes into q8 (via d0/d1 -> q8) */
    vmov.i32    q0, #0
    vst1.32     {d6[0]}, [r6]!
    vmov.f32    s0, s12
    cmp         r4, #2
    blt         .Lsm_tail_add
    vst1.32     {d6[1]}, [r6]!
    vmov.f32    s1, s13
    beq         .Lsm_tail_add
    vst1.32     {d7[0]}, [r6]!
    vmov.f32    s2, s14
.Lsm_tail_add:
    vadd.f32    q8, q8, q0

.Lsm_pass3_norm:
    /* Horizontal sum of q8 (d16, d17) -> d16 */
    vadd.f32    d16, d16, d17
    vpadd.f32   d16, d16, d16           /* d16 = [sum, sum] */

    /* Compute inv_sum = 1.0f / sum via vrecpe.f32 + 2x Newton-Raphson vrecps.f32 */
    vrecpe.f32  d8, d16
    vrecps.f32  d9, d16, d8
    vmul.f32    d8, d8, d9
    vrecps.f32  d9, d16, d8
    vmul.f32    d8, d8, d9              /* d8 (s16, s17) = [inv_sum, inv_sum] */

    /* Normalize out[0..n-1] *= inv_sum */
    mov         r4, r2
    mov         r6, r1
    cmp         r4, #8
    blt         .Lsm_norm_tail4

.p2align 4
.Lsm_norm_loop8:
    pld         [r6, #128]
    vld1.32     {d0, d1, d2, d3}, [r6]
    sub         r4, r4, #8
    cmp         r4, #8
    vmul.f32    q0, q0, d8[0]           /* q0 loaded 2 insns ago */
    vmul.f32    q1, q1, d8[0]
    vst1.32     {d0, d1, d2, d3}, [r6]!
    bge         .Lsm_norm_loop8

.Lsm_norm_tail4:
    cmp         r4, #4
    blt         .Lsm_norm_tail1
    vld1.32     {d0, d1}, [r6]
    sub         r4, r4, #4
    cmp         r4, #0
    vmul.f32    q0, q0, d8[0]
    vst1.32     {d0, d1}, [r6]!

.Lsm_norm_tail1:
    cmp         r4, #0
    ble         .Lsm_done
.Lsm_norm_scalar:
    vld1.32     {d0[0]}, [r6]
    subs        r4, r4, #1
    vmul.f32    s0, s0, s16             /* s16 = d8[0] = inv_sum */
    vst1.32     {d0[0]}, [r6]!
    bgt         .Lsm_norm_scalar

.Lsm_done:
    vpop    {d8-d15}
    pop     {r4, r5, r6, pc}

.p2align 3
.Lsm_coeffs:
    .word   0x3f317200                  /* ln2_hi = 0.693145751953125f */
    .word   0x35bfbe8e                  /* ln2_lo = 1.428606765330187e-6f */
    .word   0x3c088889                  /* c5 = 1/120 = 0.0083333338f */
    .word   0x3d2aaaab                  /* c4 = 1/24  = 0.0416666679f */
.Lsm_c3:
    .word   0x3e2aaaab                  /* c3 = 1/6   = 0.1666666716f */
.Lsm_ln2_full:
    .word   0x3f317218                  /* ln2 = 0.69314718056f */
.Lsm_log2e:
    .word   0x3fb8aa3b                  /* log2(e) = 1.4426950409f */
.Lsm_clamp_lo:
    .word   0xc2ae999a                  /* -87.3f */
.Lsm_clamp_hi:
    .word   0x42b00000                  /* +88.0f */
.Lsm_neg_inf:
    .word   0xff800000                  /* -Infinity */
.size singularity_softmax_f32_a5, .-singularity_softmax_f32_a5

/* ----------------------------------------------------------------------------
 * void singularity_silu_mul_f32_a5(
 *     float* hb,           // r0: gate activation (in/out, n floats)
 *     const float* hb2,    // r1: up projection (n floats)
 *     int n                // r2: dimension
 * )
 * Computes hb[i] = (hb[i] / (1.0f + expf(-hb[i]))) * hb2[i] in pure NEON.
 * ---------------------------------------------------------------------------- */
.global singularity_silu_mul_f32_a5
.type singularity_silu_mul_f32_a5, %function
.p2align 5
singularity_silu_mul_f32_a5:
    push    {r4, lr}
    vpush   {d8-d15}

    cmp     r2, #4
    blt     .Lsilu_done

    ldr         r4, .Lsm_clamp_lo
    vdup.32     q4, r4                  /* q4 = -87.3f */
    ldr         r4, .Lsm_clamp_hi
    vdup.32     q5, r4                  /* q5 = +88.0f */
    ldr         r4, .Lsm_log2e
    vdup.32     q6, r4                  /* q6 = log2(e) */
    adr         r4, .Lsm_coeffs
    vld1.32     {d14, d15}, [r4]        /* d14=[ln2_hi, ln2_lo], d15=[c5, c4] */
    ldr         r4, .Lsm_c3
    vdup.32     d18, r4                 /* d18 = [c3, c3] */
    vmov.i32    d19, #0x80000000        /* d19 = sign bit mask */
    vmov.f32    q10, #0.5               /* q10 = 0.5f */
    vmov.i32    q11, #127               /* q11 = 127 */

.p2align 4
.Lsilu_loop4:
    pld         [r0, #128]
    pld         [r1, #128]
    vld1.32     {d16, d17}, [r0]        /* q8 = hb[0..3] */
    vld1.32     {d24, d25}, [r1]!       /* q12 = hb2[0..3] */
    sub         r2, r2, #4
    vmov.i32    q15, #0

    /* z = clamp(-hb, -87.3f, +88.0f) */
    vneg.f32    q0, q8
    vmax.f32    q0, q0, q4
    vmin.f32    q0, q0, q5

    /* k = round(z * log2(e)) */
    vmul.f32    q1, q0, q6
    vand.u32    d4, d2, d19
    vand.u32    d5, d3, d19
    vorr.u32    q2, q2, q10
    vadd.f32    q1, q1, q2
    vcvt.s32.f32 q2, q1
    vcvt.f32.s32 q1, q2

    /* Cody-Waite reduction */
    vmls.f32    q0, q1, d14[0]
    vmls.f32    q0, q1, d14[1]

    /* Degree-5 Horner polynomial */
    vdup.32     q3, d15[1]
    vmla.f32    q3, q0, d15[0]
    vdup.32     q1, d18[0]
    vmla.f32    q1, q0, q3
    vmov.f32    q3, #0.5
    vmla.f32    q3, q0, q1
    vmov.f32    q1, #1.0
    vmla.f32    q1, q0, q3
    vmov.f32    q3, #1.0
    vmla.f32    q3, q0, q1

    vadd.s32    q2, q2, q11
    vmax.s32    q2, q2, q15
    vshl.i32    q2, q2, #23
    vmul.f32    q3, q3, q2              /* q3 = exp(-hb) */

    /* denom = 1.0f + exp(-hb) */
    vmov.f32    q1, #1.0
    vadd.f32    q3, q3, q1

    /* Reciprocal 1.0f / denom via vrecpe.f32 + 2x vrecps.f32 */
    vrecpe.f32  q1, q3
    vrecps.f32  q2, q3, q1
    vmul.f32    q1, q1, q2
    vrecps.f32  q2, q3, q1
    vmul.f32    q1, q1, q2              /* q1 = sigmoid(hb) */

    /* hb = (hb * sigmoid(hb)) * hb2 */
    vmul.f32    q8, q8, q1
    cmp         r2, #4
    vmul.f32    q8, q8, q12
    vst1.32     {d16, d17}, [r0]!
    bge         .Lsilu_loop4

.Lsilu_done:
    vpop    {d8-d15}
    pop     {r4, pc}
.size singularity_silu_mul_f32_a5, .-singularity_silu_mul_f32_a5

.section .note.GNU-stack,"",%progbits
