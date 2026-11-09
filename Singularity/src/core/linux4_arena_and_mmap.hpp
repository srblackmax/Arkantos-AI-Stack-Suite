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
#include "../asm/singularity_asm_kernels.h"

namespace singularity {

// Linux 3.4 / 4.x ARM EABI constants
constexpr int LINUX4_PROT_READ       = 0x1;
constexpr int LINUX4_PROT_WRITE      = 0x2;
constexpr int LINUX4_MAP_SHARED      = 0x01;
constexpr int LINUX4_MAP_PRIVATE     = 0x02;
constexpr int LINUX4_MAP_ANONYMOUS   = 0x20;

constexpr int LINUX4_MADV_NORMAL     = 0;
constexpr int LINUX4_MADV_RANDOM     = 1;
constexpr int LINUX4_MADV_SEQUENTIAL = 2;
constexpr int LINUX4_MADV_WILLNEED   = 3;
constexpr int LINUX4_MADV_DONTNEED   = 4;

constexpr int LINUX4_O_RDONLY        = 0;
constexpr int LINUX4_O_WRONLY        = 1;
constexpr int LINUX4_O_RDWR          = 2;
constexpr int LINUX4_O_CREAT         = 0100;
constexpr int LINUX4_O_TRUNC         = 01000;

constexpr int LINUX4_SEEK_SET        = 0;
constexpr int LINUX4_SEEK_CUR        = 1;
constexpr int LINUX4_SEEK_END        = 2;

constexpr int LINUX4_CLOCK_MONOTONIC = 1;

// Hard Memory Budgets (512 MB Physical RAM Ceiling on LG-E612 "Billy")
constexpr size_t MAX_SCRATCH_ARENA_BYTES = 16u * 1024u * 1024u; // <= 16 MB
constexpr size_t MAX_KV_CACHE_BYTES      = 35u * 1024u * 1024u; // <= 35 MB
constexpr uint32_t OOM_GUARD_FLOOR_KB    = 48u * 1024u;         // 48 MB floor

// ----------------------------------------------------------------------------
// 32-Byte Cache-Line Aligned Contiguous Arena (allocated once via sys_mmap2)
// ----------------------------------------------------------------------------
struct Linux4Arena {
    uint8_t* base;
    size_t   capacity;
    size_t   used;

    constexpr Linux4Arena() : base(nullptr), capacity(0), used(0) {}

    bool init(size_t bytes, size_t max_budget_bytes);
    void release();
    void reset() { used = 0; }

    void* alloc_aligned32(size_t bytes);

    template <typename T>
    T* alloc_array(size_t count) {
        return static_cast<T*>(alloc_aligned32(count * sizeof(T)));
    }
};

// ----------------------------------------------------------------------------
// Memory-Mapped GGUF File with Sliding-Window madvise Layer Streaming
// ----------------------------------------------------------------------------
struct Linux4MmapFile {
    uint8_t* data;
    size_t   size;
    int      fd;

    constexpr Linux4MmapFile() : data(nullptr), size(0), fd(-1) {}

    bool open_and_map(const char* path);
    void close_and_unmap();

    int advise_willneed(size_t offset, size_t length) const;
    int advise_dontneed(size_t offset, size_t length) const;
};

// ----------------------------------------------------------------------------
// Raw /proc/meminfo Parser & 48 MB OOM Guard Floor
// ----------------------------------------------------------------------------
struct Linux4MemoryGuard {
    uint32_t last_available_kb;
    uint32_t last_mem_free_kb;
    uint32_t evictions_triggered;

    constexpr Linux4MemoryGuard()
        : last_available_kb(0), last_mem_free_kb(0), evictions_triggered(0) {}

    // Parses /proc/meminfo using raw sys_open / sys_read (zero std::ifstream/sscanf)
    uint32_t query_available_ram_kb();

    // Checks if available RAM < 48 MB (or force_stream == true) and evicts
    // completed layer pages via MADV_DONTNEED while prefetching next layer.
    bool enforce_oom_floor(
        const Linux4MmapFile* mapped_model,
        size_t evict_offset,
        size_t evict_length,
        size_t prefetch_offset,
        size_t prefetch_length,
        bool force_stream = false
    );
};

// ----------------------------------------------------------------------------
// Zero-Bloat, Division-Free Console & String Helpers
// ----------------------------------------------------------------------------
size_t str_len(const char* s);
bool str_eq(const char* a, const char* b);
bool str_starts_with(const char* s, const char* prefix);

// Magic-reciprocal base-10 division: q = (n * 0xCCCCCCCDULL) >> 35 (zero udiv!)
inline void divmod10_u32(uint32_t n, uint32_t& q, uint32_t& r) {
    q = static_cast<uint32_t>((static_cast<uint64_t>(n) * 0xCCCCCCCDULL) >> 35);
    r = n - (q * 10u);
}

void write_stdout(const char* buf, size_t len);
int32_t read_stdin_line(char* buf, size_t max_len);
void print_str(const char* s);
void print_u32(uint32_t v);
void print_i32(int32_t v);
void print_f32(float v, int frac_digits = 4);
void print_hex32(uint32_t v);

uint64_t monotonic_time_ns();

} // namespace singularity
