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

.macro ROPE_BURST_4PAIRS base_reg
    /* Pairs 0 & 1: load x0..x3, rotate with (s4=c0,s5=s0) and (s6=c1,s7=s1) */
    vldmia      \base_reg, {s0-s3}
    vmul.f32    s12, s0, s4
    vmul.f32    s13, s1, s4
    vmul.f32    s14, s2, s6
    vmul.f32    s15, s3, s6
    vfms.f32    s12, s1, s5
    vfma.f32    s13, s0, s5
    vfms.f32    s14, s3, s7
    vfma.f32    s15, s2, s7
    vstmia      \base_reg!, {s12-s15}

    /* Pairs 2 & 3: load x4..x7, rotate with (s8=c2,s9=s2) and (s10=c3,s11=s3) */
    vldmia      \base_reg, {s0-s3}
    vmul.f32    s12, s0, s8
    vmul.f32    s13, s1, s8
    vmul.f32    s14, s2, s10
    vmul.f32    s15, s3, s10
    vfms.f32    s12, s1, s9
    vfma.f32    s13, s0, s9
    vfms.f32    s14, s3, s11
    vfma.f32    s15, s2, s11
    vstmia      \base_reg, {s12-s15}
.endm

/* ----------------------------------------------------------------------------
 * void singularity_rope_f32_a5(
 *     float* x,                   // r0: in/out vector [n_heads * head_dim]
 *     const float* cos_sin_table, // r1: interleaved [cos0, sin0, cos1, sin1...]
 *     int head_dim,               // r2: dimension per head (even)
 *     int n_heads                 // r3: number of heads to rotate
 * )
 * ---------------------------------------------------------------------------- */
.global singularity_rope_f32_a5
.type singularity_rope_f32_a5, %function
.p2align 5
singularity_rope_f32_a5:
    cmp         r3, #0
    bxle        lr
    cmp         r2, #2
    bxlt        lr

    push        {r4, r5}                /* 2 regs = 8 bytes (8B aligned), zero vpush */

    lsl         r12, r2, #2             /* r12 = head_dim * sizeof(float) byte stride */
    sub         r12, r12, #16           /* Pre-adjust stride for vstmia ! (+16) */
    pld         [r1, #128]

    cmp         r2, #8
    blt         .Lrope_tail2

.p2align 4
.Lrope_dim8_loop:
    vldmia      r1!, {s4-s11}           /* Hoist 4 (cos, sin) pairs into caller-saved s4..s11 */
    mov         r4, r0                  /* r4 = &x[0 * head_dim + d] */
    mov         r5, r3                  /* r5 = n_heads counter */

    cmp         r5, #4
    blt         .Lrope_head8_single

.p2align 4
.Lrope_head8_loop4:
    ROPE_BURST_4PAIRS r4
    add         r4, r4, r12
    ROPE_BURST_4PAIRS r4
    add         r4, r4, r12
    ROPE_BURST_4PAIRS r4
    add         r4, r4, r12
    ROPE_BURST_4PAIRS r4
    add         r4, r4, r12
    subs        r5, r5, #4
    cmp         r5, #4
    bge         .Lrope_head8_loop4

.Lrope_head8_single:
    cmp         r5, #1
    blt         .Lrope_dim8_next
.Lrope_head8_tail1:
    ROPE_BURST_4PAIRS r4
    add         r4, r4, r12
    subs        r5, r5, #1
    bgt         .Lrope_head8_tail1

.Lrope_dim8_next:
    add         r0, r0, #32
    subs        r2, r2, #8
    cmp         r2, #8
    bge         .Lrope_dim8_loop

.Lrope_tail2:
    cmp         r2, #2
    blt         .Lrope_done
    add         r12, r12, #16           /* Restore full head_dim * 4 stride */
.Lrope_dim2_loop:
    vldmia      r1!, {s4-s5}            /* c0, s0 */
    mov         r4, r0
    mov         r5, r3
.Lrope_head2_loop:
    vldmia      r4, {s0-s1}
    vmul.f32    s2, s0, s4
    vmul.f32    s3, s1, s4
    vfms.f32    s2, s1, s5
    vfma.f32    s3, s0, s5
    vstmia      r4, {s2-s3}
    add         r4, r4, r12
    subs        r5, r5, #1
    bgt         .Lrope_head2_loop
    add         r0, r0, #8
    subs        r2, r2, #2
    cmp         r2, #2
    bge         .Lrope_dim2_loop

.Lrope_done:
    pop         {r4, r5}
    bx          lr
.size singularity_rope_f32_a5, .-singularity_rope_f32_a5

.section .note.GNU-stack,"",%progbits
