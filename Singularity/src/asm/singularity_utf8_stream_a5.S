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

.macro UTF8_DFA_WORD reg, skip_label
    ands        r12, \reg, r3
    orrseq      r12, r12, r4
    beq         \skip_label

    uxtb        r12, \reg
    ldrb        r12, [r6, r12]
    add         r4, r4, r12
    ldrb        r4, [r7, r4]

    ubfx        r12, \reg, #8, #8
    ldrb        r12, [r6, r12]
    add         r4, r4, r12
    ldrb        r4, [r7, r4]

    ubfx        r12, \reg, #16, #8
    ldrb        r12, [r6, r12]
    add         r4, r4, r12
    ldrb        r4, [r7, r4]

    lsr         r12, \reg, #24
    ldrb        r12, [r6, r12]
    add         r4, r4, r12
    ldrb        r4, [r7, r4]
\skip_label:
.endm

/* ----------------------------------------------------------------------------
 * int singularity_utf8_stream_a5(
 *     const uint8_t* buf,        // r0: input UTF-8 byte stream
 *     size_t len,                // r1: length in bytes
 *     uint32_t* out_script_mask  // r2: optional pointer to output script mask
 * )
 * Returns: 1 if valid UTF-8, 0 if invalid UTF-8.
 * ---------------------------------------------------------------------------- */
.global singularity_utf8_stream_a5
.type singularity_utf8_stream_a5, %function
.p2align 5
singularity_utf8_stream_a5:
    push        {r4-r11, lr}
    sub         sp, sp, #20             /* 9 regs (36B) + 20B = 56B (8-byte aligned) */

    /* Zero NEON script mask accumulator q8 and scalar ASCII flag r11 */
    vmov.i8     q8, #0
    mov         r11, #0                 /* r11 = fast-path ASCII script mask */
    mov         r4, #0                  /* r4  = UTF-8 DFA state (0 = ACCEPT) */

    cmp         r1, #0
    beq         .Lutf8_finish

    /* Load vtbl.8 nibble lookup tables into {d24, d25} (hi) and {d26, d27} (lo) */
    adr         r5, .Lmilda_vtbl_tables
    vld1.8      {d24, d25, d26, d27}, [r5]
    vmov.i8     q14, #0x0f              /* q14 = low-nibble mask 0x0F */

    /* Load pointer to branchless RFC 3629 DFA table in r6 and 0x80808080 in r3 */
    adr         r6, .Lutf8_dfa_table
    ldr         r3, .Lconst_80808080
    add         r7, r6, #256            /* r7 = state transition sub-table */

    cmp         r1, #16
    blt         .Lutf8_tail_check

.p2align 4
.Lutf8_simd_loop16:
    pld         [r0, #128]
    ldr         r5, [r0, #0]
    ldr         r8, [r0, #4]
    ldr         r9, [r0, #8]
    ldr         r10, [r0, #12]

    /* Check if all 16 bytes are pure ASCII (< 0x80) and DFA state == 0 */
    orr         r12, r5, r8
    orr         r12, r12, r9
    orr         r12, r12, r10
    ands        r12, r12, r3
    orrseq      r12, r12, r4
    bne         .Lutf8_non_ascii_16

    /* Pure 16-byte ASCII fast-path */
    orr         r11, r11, #0x01
    add         r0, r0, #16
    sub         r1, r1, #16
    cmp         r1, #16
    bge         .Lutf8_simd_loop16
    b           .Lutf8_tail_check

.p2align 4
.Lutf8_non_ascii_16:
    /* Load 16 bytes into NEON q0 and run branchless vtbl.8 script router */
    vld1.8      {d0, d1}, [r0]!
    sub         r1, r1, #16
    vshr.u8     q1, q0, #4
    vand.u8     q2, q0, q14

    vtbl.8      d18, {d24, d25}, d2     /* hi_mask for bytes 0..7 */
    vtbl.8      d19, {d24, d25}, d3     /* hi_mask for bytes 8..15 */
    vtbl.8      d20, {d26, d27}, d4     /* lo_mask for bytes 0..7 */
    vtbl.8      d21, {d26, d27}, d5     /* lo_mask for bytes 8..15 */
    vand.u8     q9, q9, q10             /* q9 = exact 5-language script flags */
    vorr.u8     q8, q8, q9              /* Accumulate script flags in q8 */

    /* Validate the 4 32-bit words already in r5, r8, r9, r10 (skipping ASCII words) */
    UTF8_DFA_WORD r5,  .Lskip_w0
    UTF8_DFA_WORD r8,  .Lskip_w1
    UTF8_DFA_WORD r9,  .Lskip_w2
    UTF8_DFA_WORD r10, .Lskip_w3

    cmp         r4, #12                 /* State 12 == REJECT */
    beq         .Lutf8_invalid
    cmp         r1, #16
    bge         .Lutf8_simd_loop16

.Lutf8_tail_check:
    cmp         r1, #0
    beq         .Lutf8_finish

    /* Pad 16-byte stack scratch buffer with 0x80 (neutral continuation byte for vtbl.8) */
    vmov.i8     q0, #0x80
    mov         r8, sp
    vst1.8      {d0, d1}, [r8]

.Lutf8_tail_loop:
    ldrb        r5, [r0], #1
    strb        r5, [r8], #1
    ldrb        r12, [r6, r5]
    subs        r1, r1, #1
    add         r4, r4, r12
    ldrb        r4, [r7, r4]
    bgt         .Lutf8_tail_loop

    /* Single 16-byte NEON vtbl.8 pass for all tail bytes */
    vld1.8      {d0, d1}, [sp]
    vshr.u8     q1, q0, #4
    vand.u8     q2, q0, q14
    vtbl.8      d18, {d24, d25}, d2
    vtbl.8      d19, {d24, d25}, d3
    vtbl.8      d20, {d26, d27}, d4
    vtbl.8      d21, {d26, d27}, d5
    vand.u8     q9, q9, q10
    vorr.u8     q8, q8, q9

.Lutf8_finish:
    /* Valid iff final DFA state r4 == 0 (UTF8_ACCEPT) */
    cmp         r4, #0
    bne         .Lutf8_invalid

    /* Reduce all 16 lanes of q8 (d16, d17) via bitwise OR */
    vorr.u8     d16, d16, d17
    vext.8      d17, d16, d16, #4
    vorr.u8     d16, d16, d17
    vext.8      d17, d16, d16, #2
    vorr.u8     d16, d16, d17
    vext.8      d17, d16, d16, #1
    vorr.u8     d16, d16, d17
    vmov.u8     r8, d16[0]
    orr         r8, r8, r11

    cmp         r2, #0
    it          ne
    strne       r8, [r2]

    mov         r0, #1
    add         sp, sp, #20
    pop         {r4-r11, pc}

.Lutf8_invalid:
    cmp         r2, #0
    mov         r8, #0
    it          ne
    strne       r8, [r2]

    mov         r0, #0
    add         sp, sp, #20
    pop         {r4-r11, pc}

/* ----------------------------------------------------------------------------
 * NEON vtbl.8 Nibble Classification Lookup Tables (32 bytes)
 * ---------------------------------------------------------------------------- */
.p2align 2
.Lconst_80808080:
    .word   0x80808080

.p2align 4
.Lmilda_vtbl_tables:
    /* tbl_hi (16 bytes, indexed by byte >> 4):
     *   0..7:  0x01 (EN / ASCII)
     *   8..11: 0x00 (UTF-8 continuation byte 0x80..0xBF)
     *   12:    0x06 (0xC_ -> ES 0x02 | EO 0x04)
     *   13:    0x08 (0xD_ -> RU 0x08)
     *   14:    0x10 (0xE_ -> ZH 0x10)
     *   15:    0x00 (0xF_)
     */
    .byte   0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01
    .byte   0x00, 0x00, 0x00, 0x00, 0x06, 0x08, 0x10, 0x00

    /* tbl_lo (16 bytes, indexed by byte & 0x0F):
     *   0..1:   0x09 (EN 0x01 | RU 0x08)
     *   2:      0x0B (EN 0x01 | ES 0x02 | RU 0x08)
     *   3:      0x1B (EN 0x01 | ES 0x02 | RU 0x08 | ZH 0x10)
     *   4..5:   0x15 (EN 0x01 | EO 0x04 | ZH 0x10)
     *   6..9:   0x11 (EN 0x01 | ZH 0x10)
     *   10..14: 0x01 (EN 0x01)
     *   15:     0x11 (EN 0x01 | ZH 0x10)
     */
    .byte   0x09, 0x09, 0x0b, 0x1b, 0x15, 0x15, 0x11, 0x11
    .byte   0x11, 0x11, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11

/* ----------------------------------------------------------------------------
 * RFC 3629 Branchless UTF-8 DFA Table (Hoehrmann 256B character classes + 108B states)
 *   State 0  = ACCEPT
 *   State 12 = REJECT
 * ---------------------------------------------------------------------------- */
.p2align 4
.Lutf8_dfa_table:
    /* Byte -> Character Class (0..255) */
    .byte    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    .byte    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    .byte    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    .byte    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    .byte    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9
    .byte    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
    .byte    8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
    .byte   10,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3, 11,6,6,6,5,8,8,8,8,8,8,8,8,8,8,8

    /* (state + class) -> Next State (108 bytes) */
    .byte    0,12,24,36,60,96,84,12,12,12,48,72, 12,12,12,12,12,12,12,12,12,12,12,12
    .byte   12, 0,12,12,12,12,12, 0,12, 0,12,12, 12,24,12,12,12,12,12,24,12,24,12,12
    .byte   12,12,12,12,12,12,12,24,12,12,12,12, 12,24,12,12,12,12,12,12,12,24,12,12
    .byte   12,12,12,12,12,12,12,36,12,36,12,12, 12,36,12,12,12,12,12,36,12,36,12,12
    .byte   12,36,12,12,12,12,12,12,12,12,12,12
.size singularity_utf8_stream_a5, .-singularity_utf8_stream_a5

.section .note.GNU-stack,"",%progbits
