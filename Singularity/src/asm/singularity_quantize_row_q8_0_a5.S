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
 * void singularity_quantize_row_q8_0_a5(
 *     const float* x,      // r0: input fp32 array (k elements)
 *     void* vy_q8_0,       // r1: output block_q8_0 array (k/32 blocks)
 *     int k                // r2: number of elements (multiple of 32)
 * )
 * ---------------------------------------------------------------------------- */
.global singularity_quantize_row_q8_0_a5
.type singularity_quantize_row_q8_0_a5, %function
.p2align 5
singularity_quantize_row_q8_0_a5:
    push    {r4, lr}
    vpush   {d8-d15}

    /* nb = k >> 5 */
    lsrs    r3, r2, #5
    ble     .Lquant_done

    /* Load constants into q5 (0x80000000 sign mask), q6 (1/127.0f, 127.0f), q7 (0.5f) */
    vmov.i32    q5, #0x80000000
    vmov.f32    q7, #0.5
    ldr         r4, .Lconst_inv127
    vdup.32     d12, r4                 /* d12 = 1.0f / 127.0f */
    ldr         r4, .Lconst_127
    vdup.32     d13, r4                 /* d13 = 127.0f */

.p2align 4
.Lquant_loop:
    /* Prefetch next block's 128 bytes (4 x 32B cache lines) */
    pld     [r0, #128]
    pld     [r0, #160]
    pld     [r0, #192]
    pld     [r0, #224]

    /* Load all 32 floats into q0..q3 and q8..q11 before using any loaded reg */
    vld1.32     {d0, d1, d2, d3}, [r0]!     /* q0, q1 = x[0..7] */
    vld1.32     {d4, d5, d6, d7}, [r0]!     /* q2, q3 = x[8..15] */
    vld1.32     {d16, d17, d18, d19}, [r0]! /* q8, q9 = x[16..23] */
    vld1.32     {d20, d21, d22, d23}, [r0]! /* q10, q11 = x[24..31] */

    /* Compute absolute values and reduce max across 32 floats (q0,q1 loaded 3 insns ago) */
    vabs.f32    q12, q0
    vabs.f32    q13, q1
    vabs.f32    q14, q2
    vabs.f32    q15, q3
    vmax.f32    q12, q12, q13
    vmax.f32    q14, q14, q15

    vabs.f32    q13, q8
    vabs.f32    q15, q9
    vmax.f32    q12, q12, q14
    vmax.f32    q13, q13, q15

    vabs.f32    q14, q10
    vabs.f32    q15, q11
    vmax.f32    q12, q12, q13
    vmax.f32    q14, q14, q15
    vmax.f32    q12, q12, q14               /* q12 (d24, d25) = 4-lane amax */

    vpmax.f32   d24, d24, d25
    vpmax.f32   d24, d24, d24               /* d24 = [amax, amax] */

    /* Compute scale d = amax * (1.0f / 127.0f) and store as fp16 */
    vmul.f32    d26, d24, d12               /* d26 (in q13) = d */
    vcgt.f32    d30, d24, #0                /* d30 = non-zero mask (amax > 0) */
    vcvt.f16.f32 d28, q13                   /* d28[0] = fp16(d) */
    vst1.16     {d28[0]}, [r1]!             /* Store y[b].d (2 bytes) */

    /* Fast reciprocal: 1.0f / amax via vrecpe.f32 + 2x Newton-Raphson vrecps.f32 */
    vrecpe.f32  d8, d24
    vrecps.f32  d9, d24, d8
    vmul.f32    d8, d8, d9
    vrecps.f32  d9, d24, d8
    vmul.f32    d8, d8, d9                  /* d8 = 1.0f / amax (24-bit accurate) */
    vmul.f32    d8, d8, d13                 /* d8 = 127.0f / amax */
    vand        d8, d8, d30                 /* Zero out id if amax == 0.0f */

    /* Quantize first 16 elements (q0..q3) */
    vmul.f32    q0, q0, d8[0]
    vmul.f32    q1, q1, d8[0]
    vmul.f32    q2, q2, d8[0]
    vmul.f32    q3, q3, d8[0]

    vand.u32    q12, q0, q5
    vand.u32    q13, q1, q5
    vand.u32    q14, q2, q5
    vand.u32    q15, q3, q5

    vorr.u32    q12, q12, q7
    vorr.u32    q13, q13, q7
    vorr.u32    q14, q14, q7
    vorr.u32    q15, q15, q7

    vadd.f32    q0, q0, q12
    vadd.f32    q1, q1, q13
    vadd.f32    q2, q2, q14
    vadd.f32    q3, q3, q15

    vcvt.s32.f32 q0, q0
    vcvt.s32.f32 q1, q1
    vcvt.s32.f32 q2, q2
    vcvt.s32.f32 q3, q3

    vqmovn.s32  d0, q0
    vqmovn.s32  d1, q1
    vqmovn.s32  d2, q2
    vqmovn.s32  d3, q3

    vqmovn.s16  d0, q0
    vqmovn.s16  d1, q1
    vst1.8      {d0, d1}, [r1]!             /* Store qs[0..15] */

    /* Quantize second 16 elements (q8..q11) */
    vmul.f32    q8, q8, d8[0]
    vmul.f32    q9, q9, d8[0]
    vmul.f32    q10, q10, d8[0]
    vmul.f32    q11, q11, d8[0]

    vand.u32    q12, q8, q5
    vand.u32    q13, q9, q5
    vand.u32    q14, q10, q5
    vand.u32    q15, q11, q5

    vorr.u32    q12, q12, q7
    vorr.u32    q13, q13, q7
    vorr.u32    q14, q14, q7
    vorr.u32    q15, q15, q7

    vadd.f32    q8, q8, q12
    vadd.f32    q9, q9, q13
    vadd.f32    q10, q10, q14
    vadd.f32    q11, q11, q15

    vcvt.s32.f32 q8, q8
    vcvt.s32.f32 q9, q9
    vcvt.s32.f32 q10, q10
    vcvt.s32.f32 q11, q11

    vqmovn.s32  d16, q8
    vqmovn.s32  d17, q9
    vqmovn.s32  d18, q10
    vqmovn.s32  d19, q11

    vqmovn.s16  d16, q8
    vqmovn.s16  d17, q9
    subs        r3, r3, #1
    vst1.8      {d16, d17}, [r1]!           /* Store qs[16..31] */
    bgt         .Lquant_loop

.Lquant_done:
    vpop    {d8-d15}
    pop     {r4, pc}

.p2align 2
.Lconst_inv127:
    .word   0x3c010204                      /* 1.0f / 127.0f = 0.0078740157f */
.Lconst_127:
    .word   0x42fe0000                      /* 127.0f */
.size singularity_quantize_row_q8_0_a5, .-singularity_quantize_row_q8_0_a5

/* ----------------------------------------------------------------------------
 * void singularity_f32_to_f16_row_a5(const float* src, uint16_t* dst, int n)
 *   Converts n fp32 elements to IEEE-754 fp16 using hardware VFPv4 vcvt.f16.f32.
 * ---------------------------------------------------------------------------- */
.global singularity_f32_to_f16_row_a5
.type singularity_f32_to_f16_row_a5, %function
.p2align 5
singularity_f32_to_f16_row_a5:
    cmp     r2, #8
    blt     .Lf32_f16_tail4

.p2align 4
.Lf32_f16_loop8:
    pld         [r0, #128]
    vld1.32     {d0, d1, d2, d3}, [r0]!     /* Load 8 floats (q0, q1) */
    sub         r2, r2, #8
    cmp         r2, #8
    vcvt.f16.f32 d4, q0                     /* Convert 4 floats -> 4 fp16 */
    vcvt.f16.f32 d5, q1                     /* Convert 4 floats -> 4 fp16 */
    vst1.16     {d4, d5}, [r1]!             /* Store 8 fp16 values */
    bge         .Lf32_f16_loop8

.Lf32_f16_tail4:
    cmp     r2, #4
    blt     .Lf32_f16_tail1
    vld1.32     {d0, d1}, [r0]!
    sub         r2, r2, #4
    cmp         r2, #0
    vcvt.f16.f32 d4, q0
    vst1.16     {d4}, [r1]!

.Lf32_f16_tail1:
    cmp     r2, #0
    bxle    lr
.Lf32_f16_scalar:
    vld1.32     {d0[0]}, [r0]!
    subs        r2, r2, #1
    vcvt.f16.f32 d2, q0
    vst1.16     {d2[0]}, [r1]!
    bgt         .Lf32_f16_scalar
    bx      lr
.size singularity_f32_to_f16_row_a5, .-singularity_f32_to_f16_row_a5

/* ----------------------------------------------------------------------------
 * void singularity_f16_to_f32_row_a5(const uint16_t* src, float* dst, int n)
 *   Converts n IEEE-754 fp16 elements to fp32 using hardware VFPv4 vcvt.f32.f16.
 * ---------------------------------------------------------------------------- */
.global singularity_f16_to_f32_row_a5
.type singularity_f16_to_f32_row_a5, %function
.p2align 5
singularity_f16_to_f32_row_a5:
    cmp     r2, #8
    blt     .Lf16_f32_tail4

.p2align 4
.Lf16_f32_loop8:
    pld         [r0, #128]
    vld1.16     {d0, d1}, [r0]!             /* Load 8 fp16 values */
    sub         r2, r2, #8
    cmp         r2, #8
    vcvt.f32.f16 q1, d0                     /* Convert first 4 fp16 -> fp32 */
    vcvt.f32.f16 q2, d1                     /* Convert next 4 fp16 -> fp32 */
    vst1.32     {d2, d3, d4, d5}, [r1]!     /* Store 8 fp32 values */
    bge         .Lf16_f32_loop8

.Lf16_f32_tail4:
    cmp     r2, #4
    blt     .Lf16_f32_tail1
    vld1.16     {d0}, [r0]!
    sub         r2, r2, #4
    cmp         r2, #0
    vcvt.f32.f16 q1, d0
    vst1.32     {d2, d3}, [r1]!

.Lf16_f32_tail1:
    cmp     r2, #0
    bxle    lr
.Lf16_f32_scalar:
    vld1.16     {d0[0]}, [r0]!
    subs        r2, r2, #1
    vcvt.f32.f16 q1, d0
    vst1.32     {d2[0]}, [r1]!
    bgt         .Lf16_f32_scalar
    bx      lr
.size singularity_f16_to_f32_row_a5, .-singularity_f16_to_f32_row_a5

.section .note.GNU-stack,"",%progbits
