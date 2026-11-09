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
 * void* singularity_sys_mmap2(
 *     void* addr,          // r0
 *     size_t length,       // r1
 *     int prot,            // r2
 *     int flags,           // r3
 *     int fd,              // [sp, #0] -> r4
 *     uint32_t pgoffset    // [sp, #4] -> r5
 * )
 * Linux ARMv7 EABI __NR_mmap2 = 192
 * ---------------------------------------------------------------------------- */
.global singularity_sys_mmap2
.type singularity_sys_mmap2, %function
.p2align 4
singularity_sys_mmap2:
    push    {r4, r5, r7, lr}
    ldr     r4, [sp, #16]
    ldr     r5, [sp, #20]
    mov     r7, #192
    svc     #0
    cmn     r0, #4096
    it      hi
    movhi   r0, #-1
    pop     {r4, r5, r7, pc}
.size singularity_sys_mmap2, .-singularity_sys_mmap2

/* ----------------------------------------------------------------------------
 * int singularity_sys_munmap(void* addr, size_t length)
 * Linux ARMv7 EABI __NR_munmap = 91
 * ---------------------------------------------------------------------------- */
.global singularity_sys_munmap
.type singularity_sys_munmap, %function
.p2align 4
singularity_sys_munmap:
    push    {r7, lr}
    mov     r7, #91
    svc     #0
    pop     {r7, pc}
.size singularity_sys_munmap, .-singularity_sys_munmap

/* ----------------------------------------------------------------------------
 * int singularity_sys_madvise(void* addr, size_t length, int advice)
 * Linux ARMv7 EABI __NR_madvise = 220
 * ---------------------------------------------------------------------------- */
.global singularity_sys_madvise
.type singularity_sys_madvise, %function
.p2align 4
singularity_sys_madvise:
    push    {r7, lr}
    mov     r7, #220
    svc     #0
    pop     {r7, pc}
.size singularity_sys_madvise, .-singularity_sys_madvise

/* ----------------------------------------------------------------------------
 * int singularity_sys_open(const char* pathname, int flags, int mode)
 * Linux ARMv7 EABI __NR_open = 5
 * ---------------------------------------------------------------------------- */
.global singularity_sys_open
.type singularity_sys_open, %function
.p2align 4
singularity_sys_open:
    push    {r7, lr}
    mov     r7, #5
    svc     #0
    pop     {r7, pc}
.size singularity_sys_open, .-singularity_sys_open

/* ----------------------------------------------------------------------------
 * int32_t singularity_sys_read(int fd, void* buf, size_t count)
 * Linux ARMv7 EABI __NR_read = 3
 * ---------------------------------------------------------------------------- */
.global singularity_sys_read
.type singularity_sys_read, %function
.p2align 4
singularity_sys_read:
    push    {r7, lr}
    mov     r7, #3
    svc     #0
    pop     {r7, pc}
.size singularity_sys_read, .-singularity_sys_read

/* ----------------------------------------------------------------------------
 * int32_t singularity_sys_write(int fd, const void* buf, size_t count)
 * Linux ARMv7 EABI __NR_write = 4
 * ---------------------------------------------------------------------------- */
.global singularity_sys_write
.type singularity_sys_write, %function
.p2align 4
singularity_sys_write:
    push    {r7, lr}
    mov     r7, #4
    svc     #0
    pop     {r7, pc}
.size singularity_sys_write, .-singularity_sys_write

/* ----------------------------------------------------------------------------
 * int singularity_sys_close(int fd)
 * Linux ARMv7 EABI __NR_close = 6
 * ---------------------------------------------------------------------------- */
.global singularity_sys_close
.type singularity_sys_close, %function
.p2align 4
singularity_sys_close:
    push    {r7, lr}
    mov     r7, #6
    svc     #0
    pop     {r7, pc}
.size singularity_sys_close, .-singularity_sys_close

/* ----------------------------------------------------------------------------
 * int singularity_sys_fsync(int fd)
 * Linux ARMv7 EABI __NR_fsync = 118
 * ---------------------------------------------------------------------------- */
.global singularity_sys_fsync
.type singularity_sys_fsync, %function
.p2align 4
singularity_sys_fsync:
    push    {r7, lr}
    mov     r7, #118
    svc     #0
    pop     {r7, pc}
.size singularity_sys_fsync, .-singularity_sys_fsync

/* ----------------------------------------------------------------------------
 * int32_t singularity_sys_lseek(int fd, int32_t offset, int whence)
 * Linux ARMv7 EABI __NR_lseek = 19
 * ---------------------------------------------------------------------------- */
.global singularity_sys_lseek
.type singularity_sys_lseek, %function
.p2align 4
singularity_sys_lseek:
    push    {r7, lr}
    mov     r7, #19
    svc     #0
    pop     {r7, pc}
.size singularity_sys_lseek, .-singularity_sys_lseek

/* ----------------------------------------------------------------------------
 * int singularity_sys_clock_gettime(int clk_id, singularity::timespec32* tp)
 * Linux ARMv7 EABI __NR_clock_gettime = 263 (32-bit timespec for Linux 3.4/4.x)
 * ---------------------------------------------------------------------------- */
.global singularity_sys_clock_gettime
.type singularity_sys_clock_gettime, %function
.p2align 4
singularity_sys_clock_gettime:
    push    {r7, lr}
    movw    r7, #263
    svc     #0
    pop     {r7, pc}
.size singularity_sys_clock_gettime, .-singularity_sys_clock_gettime

/* ----------------------------------------------------------------------------
 * int singularity_sys_sched_normal(void)
 *   Resets scheduling policy to SCHED_NORMAL (0) via __NR_sched_setscheduler
 *   (syscall 156) so single-core Cortex-A5 kernel threads (kswapd0, mmcqd/0,
 *   and WCN1314 Wi-Fi VOS_MC_Thread) are never starved by SCHED_FIFO (chrt -f).
 * ---------------------------------------------------------------------------- */
.global singularity_sys_sched_normal
.type singularity_sys_sched_normal, %function
.p2align 4
singularity_sys_sched_normal:
    push    {r7, lr}
    sub     sp, sp, #8
    mov     r0, #0                      /* mask = 0 so 0666 session files work across sudo/user */
    mov     r7, #60                     /* __NR_umask = 60 */
    svc     #0
    mov     r0, #0                      /* pid = 0 (current process) */
    mov     r1, #0                      /* policy = SCHED_NORMAL (0) */
    str     r0, [sp, #0]                /* sched_param.sched_priority = 0 */
    mov     r2, sp
    mov     r7, #156                    /* __NR_sched_setscheduler = 156 */
    svc     #0
    add     sp, sp, #8
    pop     {r7, pc}
.size singularity_sys_sched_normal, .-singularity_sys_sched_normal

/* ----------------------------------------------------------------------------
 * void* singularity_memcpy_a5(void* dst, const void* src, size_t n)
 *   32-byte cache-line burst copy using vldmia/vstmia across d0-d7 with PLD
 *   prefetching when word-aligned, with safe unaligned NEON vld1.8 fallback.
 * ---------------------------------------------------------------------------- */
.global singularity_memcpy_a5
.type singularity_memcpy_a5, %function
.p2align 5
singularity_memcpy_a5:
    mov     r3, r0                      /* r3 = write cursor; preserve r0 return */
    cmp     r2, #32
    blt     .Lmc_tail16

    /* Check if both dst and src are 4-byte aligned for vldmia/vstmia burst */
    orr     r12, r3, r1
    tst     r12, #3
    bne     .Lmc_unaligned32

    cmp     r2, #64
    blt     .Lmc_burst32

.p2align 4
.Lmc_burst64:
    pld         [r1, #128]
    pld         [r1, #160]
    vldmia      r1!, {d0-d3}            /* Burst load 32B cache line 0 */
    vldmia      r1!, {d4-d7}            /* Burst load 32B cache line 1 */
    sub         r2, r2, #64
    cmp         r2, #64
    vstmia      r3!, {d0-d3}            /* Burst store 32B cache line 0 */
    vstmia      r3!, {d4-d7}            /* Burst store 32B cache line 1 */
    bge         .Lmc_burst64

.Lmc_burst32:
    cmp         r2, #32
    blt         .Lmc_tail16
    vldmia      r1!, {d0-d3}
    sub         r2, r2, #32
    cmp         r2, #16
    vstmia      r3!, {d0-d3}
    b           .Lmc_tail16

.p2align 4
.Lmc_unaligned32:
    pld         [r1, #128]
    vld1.8      {d0, d1, d2, d3}, [r1]!
    sub         r2, r2, #32
    cmp         r2, #32
    vst1.8      {d0, d1, d2, d3}, [r3]!
    bge         .Lmc_unaligned32

.Lmc_tail16:
    cmp         r2, #16
    blt         .Lmc_tail4
    vld1.8      {d0, d1}, [r1]!
    sub         r2, r2, #16
    cmp         r2, #4
    vst1.8      {d0, d1}, [r3]!

.Lmc_tail4:
    cmp         r2, #0
    bxle        lr
.Lmc_byte_loop:
    ldrb        r12, [r1], #1
    subs        r2, r2, #1
    strb        r12, [r3], #1
    bgt         .Lmc_byte_loop
    bx          lr
.size singularity_memcpy_a5, .-singularity_memcpy_a5

.section .note.GNU-stack,"",%progbits
