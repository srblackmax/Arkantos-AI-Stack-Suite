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

#include "linux4_arena_and_mmap.hpp"

namespace singularity {

bool Linux4Arena::init(size_t bytes, size_t max_budget_bytes) {
    if (bytes == 0 || bytes > max_budget_bytes) {
        return false;
    }
    // Round up to 4096-byte page size using bitmask (zero integer division)
    const size_t page_aligned = (bytes + 4095u) & ~size_t(4095u);
    void* ptr = singularity_sys_mmap2(
        nullptr,
        page_aligned,
        LINUX4_PROT_READ | LINUX4_PROT_WRITE,
        LINUX4_MAP_PRIVATE | LINUX4_MAP_ANONYMOUS,
        -1,
        0
    );
    if (ptr == reinterpret_cast<void*>(-1) || ptr == nullptr) {
        base = nullptr;
        capacity = 0;
        used = 0;
        return false;
    }
    base = static_cast<uint8_t*>(ptr);
    capacity = page_aligned;
    used = 0;
    return true;
}

void Linux4Arena::release() {
    if (base != nullptr && capacity > 0) {
        singularity_sys_munmap(base, capacity);
    }
    base = nullptr;
    capacity = 0;
    used = 0;
}

void* Linux4Arena::alloc_aligned32(size_t bytes) {
    if (base == nullptr) {
        return nullptr;
    }
    const size_t aligned_used = (used + 31u) & ~size_t(31u);
    const size_t next_used = aligned_used + ((bytes + 31u) & ~size_t(31u));
    if (next_used > capacity) {
        return nullptr;
    }
    void* ptr = base + aligned_used;
    used = next_used;
    return ptr;
}

bool Linux4MmapFile::open_and_map(const char* path) {
    close_and_unmap();
    if (path == nullptr) {
        return false;
    }
    const int file_fd = singularity_sys_open(path, LINUX4_O_RDONLY, 0);
    if (file_fd < 0) {
        return false;
    }
    const int32_t file_len = singularity_sys_lseek(file_fd, 0, LINUX4_SEEK_END);
    if (file_len <= 0) {
        singularity_sys_close(file_fd);
        return false;
    }
    singularity_sys_lseek(file_fd, 0, LINUX4_SEEK_SET);

    void* mapped = singularity_sys_mmap2(
        nullptr,
        static_cast<size_t>(file_len),
        LINUX4_PROT_READ,
        LINUX4_MAP_PRIVATE,
        file_fd,
        0
    );
    if (mapped == reinterpret_cast<void*>(-1) || mapped == nullptr) {
        singularity_sys_close(file_fd);
        return false;
    }
    data = static_cast<uint8_t*>(mapped);
    size = static_cast<size_t>(file_len);
    fd = file_fd;
    return true;
}

void Linux4MmapFile::close_and_unmap() {
    if (data != nullptr && size > 0) {
        singularity_sys_munmap(data, size);
    }
    if (fd >= 0) {
        singularity_sys_close(fd);
    }
    data = nullptr;
    size = 0;
    fd = -1;
}

int Linux4MmapFile::advise_willneed(size_t offset, size_t length) const {
    if (data == nullptr || offset >= size || length == 0) {
        return -1;
    }
    if (offset + length > size) {
        length = size - offset;
    }
    const size_t page_off = offset & ~size_t(4095u);
    const size_t page_len = length + (offset - page_off);
    return singularity_sys_madvise(data + page_off, page_len, LINUX4_MADV_WILLNEED);
}

int Linux4MmapFile::advise_dontneed(size_t offset, size_t length) const {
    if (data == nullptr || offset >= size || length == 0) {
        return -1;
    }
    if (offset + length > size) {
        length = size - offset;
    }
    const size_t page_off = offset & ~size_t(4095u);
    const size_t page_len = length + (offset - page_off);
    return singularity_sys_madvise(data + page_off, page_len, LINUX4_MADV_DONTNEED);
}

static uint32_t parse_kb_after_colon(const char* p, const char* end) {
    while (p < end && *p != ':') {
        ++p;
    }
    if (p < end && *p == ':') {
        ++p;
    }
    while (p < end && (*p == ' ' || *p == '\t')) {
        ++p;
    }
    uint32_t val = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        val = (val * 10u) + static_cast<uint32_t>(*p - '0');
        ++p;
    }
    return val;
}

static bool line_starts_with(const char* line, const char* end, const char* key) {
    while (*key != '\0') {
        if (line >= end || *line != *key) {
            return false;
        }
        ++line;
        ++key;
    }
    return true;
}

uint32_t Linux4MemoryGuard::query_available_ram_kb() {
    char buf[1024];
    const int fd = singularity_sys_open("/proc/meminfo", LINUX4_O_RDONLY, 0);
    if (fd < 0) {
        // Fallback if /proc is unmounted
        last_available_kb = 128u * 1024u;
        return last_available_kb;
    }
    const int32_t nread = singularity_sys_read(fd, buf, sizeof(buf) - 1);
    singularity_sys_close(fd);
    if (nread <= 0) {
        last_available_kb = 128u * 1024u;
        return last_available_kb;
    }
    buf[nread] = '\0';
    const char* cur = buf;
    const char* end = buf + nread;

    uint32_t mem_available = 0;
    uint32_t mem_free      = 0;
    uint32_t buffers       = 0;
    uint32_t cached        = 0;

    while (cur < end) {
        if (line_starts_with(cur, end, "MemAvailable:")) {
            mem_available = parse_kb_after_colon(cur, end);
        } else if (line_starts_with(cur, end, "MemFree:")) {
            mem_free = parse_kb_after_colon(cur, end);
        } else if (line_starts_with(cur, end, "Buffers:")) {
            buffers = parse_kb_after_colon(cur, end);
        } else if (line_starts_with(cur, end, "Cached:")) {
            cached = parse_kb_after_colon(cur, end);
        }
        while (cur < end && *cur != '\n') {
            ++cur;
        }
        if (cur < end && *cur == '\n') {
            ++cur;
        }
    }

    last_mem_free_kb = mem_free;
    if (mem_available > 0) {
        last_available_kb = mem_available;
    } else {
        // Linux 3.4 fallback (MemFree + Buffers + Cached)
        last_available_kb = mem_free + buffers + cached;
    }
    return last_available_kb;
}

bool Linux4MemoryGuard::enforce_oom_floor(
    const Linux4MmapFile* mapped_model,
    size_t evict_offset,
    size_t evict_length,
    size_t prefetch_offset,
    size_t prefetch_length,
    bool force_stream
) {
    const uint32_t avail_kb = query_available_ram_kb();
    const bool low_mem =
        (avail_kb < OOM_GUARD_FLOOR_KB) ||
        (last_mem_free_kb > 0 && last_mem_free_kb < OOM_GUARD_FLOOR_KB) ||
        force_stream;
    if (mapped_model != nullptr && mapped_model->data != nullptr) {
        if (low_mem && evict_length > 0) {
            mapped_model->advise_dontneed(evict_offset, evict_length);
            ++evictions_triggered;
        }
        if (low_mem && prefetch_length > 0) {
            mapped_model->advise_willneed(prefetch_offset, prefetch_length);
        }
    }
    return low_mem;
}

size_t str_len(const char* s) {
    if (s == nullptr) {
        return 0;
    }
    size_t n = 0;
    while (s[n] != '\0') {
        ++n;
    }
    return n;
}

bool str_eq(const char* a, const char* b) {
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    while (*a != '\0' && *b != '\0') {
        if (*a != *b) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

bool str_starts_with(const char* s, const char* prefix) {
    if (s == nullptr || prefix == nullptr) {
        return false;
    }
    while (*prefix != '\0') {
        if (*s != *prefix) {
            return false;
        }
        ++s;
        ++prefix;
    }
    return true;
}

void write_stdout(const char* buf, size_t len) {
    if (buf != nullptr && len > 0) {
        singularity_sys_write(1, buf, len);
    }
}

int32_t read_stdin_line(char* buf, size_t max_len) {
    if (buf == nullptr || max_len < 2) {
        return -1;
    }
    size_t pos = 0;
    while (pos + 1 < max_len) {
        char ch = 0;
        const int32_t n = singularity_sys_read(0, &ch, 1);
        if (n <= 0) {
            if (pos == 0) {
                return -1;
            }
            break;
        }
        if (ch == '\r') {
            continue;
        }
        if (ch == '\n') {
            break;
        }
        buf[pos++] = ch;
    }
    buf[pos] = '\0';
    return static_cast<int32_t>(pos);
}

void print_str(const char* s) {
    write_stdout(s, str_len(s));
}

void print_u32(uint32_t v) {
    char buf[16];
    int pos = 15;
    if (v == 0) {
        buf[pos--] = '0';
    } else {
        while (v > 0) {
            uint32_t q = 0, r = 0;
            divmod10_u32(v, q, r);
            buf[pos--] = static_cast<char>('0' + r);
            v = q;
        }
    }
    write_stdout(buf + pos + 1, static_cast<size_t>(15 - pos));
}

void print_i32(int32_t v) {
    if (v < 0) {
        write_stdout("-", 1);
        const uint32_t uv = static_cast<uint32_t>(-(v + 1)) + 1u;
        print_u32(uv);
    } else {
        print_u32(static_cast<uint32_t>(v));
    }
}

void print_f32(float v, int frac_digits) {
    if (v < 0.0f) {
        write_stdout("-", 1);
        v = -v;
    }
    const uint32_t int_part = static_cast<uint32_t>(v);
    print_u32(int_part);
    if (frac_digits > 0) {
        write_stdout(".", 1);
        float rem = v - static_cast<float>(int_part);
        for (int i = 0; i < frac_digits; ++i) {
            rem *= 10.0f;
            uint32_t d = static_cast<uint32_t>(rem);
            if (d > 9u) {
                d = 9u;
            }
            const char c = static_cast<char>('0' + d);
            write_stdout(&c, 1);
            rem -= static_cast<float>(d);
        }
    }
}

void print_hex32(uint32_t v) {
    char buf[10];
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 7; i >= 0; --i) {
        const uint32_t nib = (v >> (i * 4)) & 0xFu;
        buf[9 - i] = static_cast<char>(nib < 10 ? ('0' + nib) : ('a' + (nib - 10)));
    }
    write_stdout(buf, 10);
}

uint64_t monotonic_time_ns() {
    timespec32 ts{0, 0};
    singularity_sys_clock_gettime(LINUX4_CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(static_cast<uint32_t>(ts.tv_sec)) * 1000000000ULL +
           static_cast<uint64_t>(static_cast<uint32_t>(ts.tv_nsec));
}

} // namespace singularity
