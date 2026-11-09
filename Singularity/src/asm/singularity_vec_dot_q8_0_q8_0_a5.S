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
 * void singularity_vec_dot_q8_0_q8_0_a5(
 *     int n,               // r0: number of elements (multiple of 32)
 *     float* out_sum,      // r1: pointer to output float
 *     const void* vx_q8_0, // r2: pointer to block_q8_0 array
 *     const void* vy_q8_0  // r3: pointer to block_q8_0 array
 * )
 * ---------------------------------------------------------------------------- */
.global singularity_vec_dot_q8_0_q8_0_a5
.type singularity_vec_dot_q8_0_q8_0_a5, %function
.p2align 5
singularity_vec_dot_q8_0_q8_0_a5:
    /* Preserve AAPCS-VFP callee-saved GP (even count = 8B aligned) and NEON */
    push    {r4, r5, r6, lr}
    vpush   {d8-d15}

    /* Zero the 4x unrolled float32x4 accumulators q8, q9, q10, q11 */
    vmov.i32    q8, #0
    vmov.i32    q9, #0
    vmov.i32    q10, #0
    vmov.i32    q11, #0

    /* Compute number of 32-element blocks: nb = n >> 5 */
    lsrs    r4, r0, #5
    ble     .Lq8q8_reduce

    cmp     r4, #4
    blt     .Lq8q8_tail_check

.p2align 4
.Lq8q8_loop4:
    /* Prefetch upcoming 32-byte cache lines (136B per 4 blocks each) */
    pld     [r2, #128]
    pld     [r3, #128]

    /* ==================== BLOCK 0 (accumulates into q5 -> q8) ==================== */
    vld1.16     {d4[0]}, [r2]!          /* Load x[0].d (fp16) */
    vld1.16     {d5[0]}, [r3]!          /* Load y[0].d (fp16) */
    vld1.8      {d0, d1}, [r2]!         /* Load x[0].qs[0..15] */
    vld1.8      {d6, d7}, [r3]!         /* Load y[0].qs[0..15] */
    vld1.8      {d2, d3}, [r2]!         /* Load x[0].qs[16..31] */
    vld1.8      {d8, d9}, [r3]!         /* Load y[0].qs[16..31] */

    pld         [r2, #160]
    pld         [r3, #160]
    vmov.i32    q5, #0                  /* 3 instructions after d6,d7 load */

    vmull.s8    q12, d0, d6             /* prod 0..7 (d0,d6 loaded 4-5 insns ago) */
    vmlal.s8    q12, d1, d7             /* + prod 8..15 */
    vmull.s8    q13, d2, d8             /* prod 16..23 (d2,d8 loaded 5-6 insns ago) */
    vmlal.s8    q13, d3, d9             /* + prod 24..31 */
    vpadal.s16  q5, q12
    vpadal.s16  q5, q13

    /* ==================== BLOCK 1 (accumulates into q15 -> q9) =================== */
    vld1.16     {d4[1]}, [r2]!          /* Load x[1].d (fp16) */
    vld1.16     {d5[1]}, [r3]!          /* Load y[1].d (fp16) */
    vld1.8      {d0, d1}, [r2]!         /* Load x[1].qs[0..15] */
    vld1.8      {d6, d7}, [r3]!         /* Load y[1].qs[0..15] */
    vld1.8      {d2, d3}, [r2]!         /* Load x[1].qs[16..31] */
    vld1.8      {d8, d9}, [r3]!         /* Load y[1].qs[16..31] */

    nop                                 /* Maintain >= 3 insns load-to-use separation */
    vmov.i32    q15, #0                 /* Zero Block 1 int32 accumulator q15 */

    vmull.s8    q12, d0, d6
    vmlal.s8    q12, d1, d7
    vmull.s8    q13, d2, d8
    vmlal.s8    q13, d3, d9
    vpadal.s16  q15, q12
    vpadal.s16  q15, q13

    /* ==================== BLOCK 2 (accumulates into q14 -> q10) ================== */
    vld1.16     {d4[2]}, [r2]!          /* Load x[2].d (fp16) */
    vld1.16     {d5[2]}, [r3]!          /* Load y[2].d (fp16) */
    vld1.8      {d0, d1}, [r2]!         /* Load x[2].qs[0..15] */
    vld1.8      {d6, d7}, [r3]!         /* Load y[2].qs[0..15] */
    vld1.8      {d2, d3}, [r2]!         /* Load x[2].qs[16..31] */
    vld1.8      {d8, d9}, [r3]!         /* Load y[2].qs[16..31] */

    nop                                 /* Maintain >= 3 insns load-to-use separation */
    vmov.i32    q14, #0                 /* Zero Block 2 int32 accumulator q14 */

    vmull.s8    q12, d0, d6
    vmlal.s8    q12, d1, d7
    vmull.s8    q13, d2, d8
    vmlal.s8    q13, d3, d9
    vpadal.s16  q14, q12
    vpadal.s16  q14, q13

    /* ==================== BLOCK 3 (accumulates into q0 -> q11) =================== */
    vld1.16     {d4[3]}, [r2]!          /* Load x[3].d (fp16) */
    vld1.16     {d5[3]}, [r3]!          /* Load y[3].d (fp16) */
    vld1.8      {d0, d1}, [r2]!         /* Load x[3].qs[0..15] */
    vld1.8      {d6, d7}, [r3]!         /* Load y[3].qs[0..15] */
    vld1.8      {d2, d3}, [r2]!         /* Load x[3].qs[16..31] */
    vld1.8      {d8, d9}, [r3]!         /* Load y[3].qs[16..31] */

    nop                                 /* Maintain >= 3 insns load-to-use separation */
    sub         r4, r4, #4              /* Decrement block counter early */

    vmull.s8    q12, d0, d6
    vmlal.s8    q12, d1, d7
    vmull.s8    q13, d2, d8
    vmlal.s8    q13, d3, d9

    /* Now q0 is free; use q0 for Block 3 int32 accumulation */
    vmov.i32    q0, #0
    vcvt.f32.f16 q6, d4                 /* Convert all 4 x_d fp16 scales -> fp32 */
    vcvt.f32.f16 q7, d5                 /* Convert all 4 y_d fp16 scales -> fp32 */
    vpadal.s16  q0, q12
    vpadal.s16  q0, q13

    /* Exact int32 horizontal reduction of all 4 blocks into q5 = [s0, s1, s2, s3] */
    vpadd.s32   d10, d10, d11           /* Block 0 (q5) */
    vpadd.s32   d11, d30, d31           /* Block 1 (q15) */
    vpadd.s32   d28, d28, d29           /* Block 2 (q14) */
    vpadd.s32   d29, d0, d1             /* Block 3 (q0) */
    vpadd.s32   d10, d10, d11           /* d10 = [sumi_0, sumi_1] */
    vpadd.s32   d11, d28, d29           /* d11 = [sumi_2, sumi_3] */

    vmul.f32    q6, q6, q7              /* q6 (d12, d13) = [scale0, scale1, scale2, scale3] */
    vcvt.f32.s32 q5, q5                 /* q5 = float([sumi_0, sumi_1, sumi_2, sumi_3]) */
    vmul.f32    q5, q5, q6              /* q5 = exact block float products */

    /* Accumulate across q8, q9 */
    vadd.f32    d16, d16, d10
    vadd.f32    d18, d18, d11

    cmp         r4, #4
    bge         .Lq8q8_loop4

.Lq8q8_tail_check:
    cmp         r4, #0
    ble         .Lq8q8_reduce

.p2align 4
.Lq8q8_tail1:
    vld1.16     {d4[0]}, [r2]!
    vld1.16     {d5[0]}, [r3]!
    vld1.8      {d0, d1}, [r2]!
    vld1.8      {d6, d7}, [r3]!
    vld1.8      {d2, d3}, [r2]!
    vld1.8      {d8, d9}, [r3]!

    vmov.i32    q5, #0
    subs        r4, r4, #1

    vmull.s8    q12, d0, d6
    vmlal.s8    q12, d1, d7
    vmull.s8    q13, d2, d8
    vmlal.s8    q13, d3, d9

    vcvt.f32.f16 q6, d4
    vcvt.f32.f16 q7, d5
    vpadal.s16  q5, q12
    vpadal.s16  q5, q13
    vpadd.s32   d10, d10, d11
    vpadd.s32   d10, d10, d10
    vmul.f32    d12, d12, d14
    vcvt.f32.s32 d10, d10
    vmul.f32    s20, s20, s24
    vadd.f32    s16, s16, s20
    bgt         .Lq8q8_tail1

.Lq8q8_reduce:
    /* Combine accumulators in exact left-to-right IEEE-754 order ((b0+b1)+b2)+b3 */
    vadd.f32    q8, q8, q10
    vadd.f32    q9, q9, q11
    vadd.f32    d16, d16, d17
    vmov        d2, d18                 /* s4 = b2, s5 = b3 */
    vpadd.f32   d0, d16, d16            /* s0 = b0 + b1 */
    vadd.f32    s0, s0, s4              /* s0 = (b0 + b1) + b2 */
    vadd.f32    s0, s0, s5              /* s0 = ((b0 + b1) + b2) + b3 */

    /* Store scalar float result to *out_sum if non-null */
    cmp         r1, #0
    it          ne
    vstrne.32   s0, [r1]

    vpop    {d8-d15}
    pop     {r4, r5, r6, pc}
.size singularity_vec_dot_q8_0_q8_0_a5, .-singularity_vec_dot_q8_0_q8_0_a5

.section .note.GNU-stack,"",%progbits
