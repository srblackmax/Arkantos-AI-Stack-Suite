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

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace singularity {

constexpr int QK4_0 = 32;
constexpr int QK8_0 = 32;

// GGUF Q4_0 block: 2-byte fp16 scale + 16 bytes of 4-bit nibbles (32 weights)
// Low nibbles: qs[0..15] & 0x0F -> weights 0..15 (offset -8)
// High nibbles: (qs[0..15] >> 4) -> weights 16..31 (offset -8)
struct __attribute__((packed)) block_q4_0 {
    uint16_t d;       // IEEE-754 binary16 scale (converted in .S via vcvt.f32.f16)
    uint8_t  qs[16];  // 32 nibbles
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// GGUF Q8_0 block: 2-byte fp16 scale + 32 x int8_t quantized values
struct __attribute__((packed)) block_q8_0 {
    uint16_t d;       // IEEE-754 binary16 scale (converted in .S via vcvt.f32.f16)
    int8_t   qs[32];  // 32 signed 8-bit values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// Linux 4.x 32-bit timespec for __NR_clock_gettime (syscall 263)
struct timespec32 {
    int32_t tv_sec;
    int32_t tv_nsec;
};

// Milda 5-Language Script Bitmask Flags returned by singularity_utf8_stream_a5
enum MildaScriptMask : uint32_t {
    SCRIPT_MASK_EN = (1u << 0), // English / Basic Latin (ASCII 0x00..0x7F)
    SCRIPT_MASK_ES = (1u << 1), // Spanish (Latin-1 Supplement: ¡, ¿, á, é, í, ó, ú, ñ, ü)
    SCRIPT_MASK_EO = (1u << 2), // Esperanto (Latin Extended-A: ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)
    SCRIPT_MASK_RU = (1u << 3), // Russian (Cyrillic: А..я, Ё, ё)
    SCRIPT_MASK_ZH = (1u << 4), // Simplified Mandarin Chinese (CJK Unified Ideographs)
};

enum MildaLanguage : uint32_t {
    LANG_ENGLISH   = 0,
    LANG_SPANISH   = 1,
    LANG_ESPERANTO = 2,
    LANG_RUSSIAN   = 3,
    LANG_MANDARIN  = 4,
};

} // namespace singularity

extern "C" {

// ----------------------------------------------------------------------------
// 1. Quantized Dot Product Kernels (Cortex-A5 In-Order Scheduled NEON)
// ----------------------------------------------------------------------------
void singularity_vec_dot_q4_0_q8_0_a5(
    int n,
    float* out_sum,
    const void* vx_q4_0,
    const void* vy_q8_0
);

void singularity_vec_dot_q8_0_q8_0_a5(
    int n,
    float* out_sum,
    const void* vx_q8_0,
    const void* vy_q8_0
);

// ----------------------------------------------------------------------------
// 2. Activation Quantization Kernel (fp32 -> block_q8_0, zero vdiv.f32)
// ----------------------------------------------------------------------------
void singularity_quantize_row_q8_0_a5(
    const float* x,
    void* vy_q8_0,
    int k
);

// ----------------------------------------------------------------------------
// 3. Normalization & Activation Kernels (vrsqrte/vrsqrts & polynomial exp)
// ----------------------------------------------------------------------------
void singularity_rmsnorm_f32_a5(
    const float* x,
    const float* weight,
    float* out,
    int n,
    float eps
);

void singularity_softmax_f32_a5(
    const float* x,
    float* out,
    int n
);

void singularity_silu_mul_f32_a5(
    float* hb,
    const float* hb2,
    int n
);

// ----------------------------------------------------------------------------
// 4. Rotary Position Embedding (RoPE) Kernel (de-interleaved pairs)
// ----------------------------------------------------------------------------
void singularity_rope_f32_a5(
    float* x,
    const float* cos_sin_table,
    int head_dim,
    int n_heads
);

// ----------------------------------------------------------------------------
// 5. Branchless NEON vtbl.8 UTF-8 Validator & 5-Language Script Router
// ----------------------------------------------------------------------------
int singularity_utf8_stream_a5(
    const uint8_t* buf,
    size_t len,
    uint32_t* out_script_mask
);

// ----------------------------------------------------------------------------
// 6. Hardware VFPv4 fp16 <-> fp32 Vector Conversions (for KV Cache)
// ----------------------------------------------------------------------------
void singularity_f32_to_f16_row_a5(
    const float* src,
    uint16_t* dst,
    int n
);

void singularity_f16_to_f32_row_a5(
    const uint16_t* src,
    float* dst,
    int n
);

// ----------------------------------------------------------------------------
// 7. Direct Linux 4.x ARMv7 EABI Syscalls & Burst Memcpy
// ----------------------------------------------------------------------------
void* singularity_sys_mmap2(
    void* addr,
    size_t length,
    int prot,
    int flags,
    int fd,
    uint32_t pgoffset
);

int singularity_sys_munmap(void* addr, size_t length);
int singularity_sys_madvise(void* addr, size_t length, int advice);
int singularity_sys_open(const char* pathname, int flags, int mode);
int32_t singularity_sys_read(int fd, void* buf, size_t count);
int32_t singularity_sys_write(int fd, const void* buf, size_t count);
int singularity_sys_close(int fd);
int singularity_sys_fsync(int fd);
int32_t singularity_sys_lseek(int fd, int32_t offset, int whence);
int singularity_sys_clock_gettime(int clk_id, singularity::timespec32* tp);
int singularity_sys_sched_normal(void);

void* singularity_memcpy_a5(void* dst, const void* src, size_t n);

} // extern "C"
