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

#include <yantra/index.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <memory_resource>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace yantra::inline v1
{

namespace
{

auto
l2_sq_f32_scalar (const float *a, const float *b, std::size_t dim) noexcept
    -> float
{
  float sum = 0.0F;
  for (std::size_t i = 0U; i < dim; ++i)
    {
      const float diff = a[i] - b[i];
      sum += diff * diff;
    }
  return sum;
}

#if defined(__x86_64__)
__attribute__ ((target ("avx2,fma"))) auto
l2_sq_f32_avx2 (const float *a, const float *b, std::size_t dim) noexcept
    -> float
{
  __m256 acc0 = _mm256_setzero_ps ();
  __m256 acc1 = _mm256_setzero_ps ();
  __m256 acc2 = _mm256_setzero_ps ();
  __m256 acc3 = _mm256_setzero_ps ();

  std::size_t i = 0U;
  for (; i + 32U <= dim; i += 32U)
    {
      const __m256 d0
          = _mm256_sub_ps (_mm256_loadu_ps (a + i), _mm256_loadu_ps (b + i));
      const __m256 d1 = _mm256_sub_ps (_mm256_loadu_ps (a + i + 8U),
                                       _mm256_loadu_ps (b + i + 8U));
      const __m256 d2 = _mm256_sub_ps (_mm256_loadu_ps (a + i + 16U),
                                       _mm256_loadu_ps (b + i + 16U));
      const __m256 d3 = _mm256_sub_ps (_mm256_loadu_ps (a + i + 24U),
                                       _mm256_loadu_ps (b + i + 24U));
      acc0 = _mm256_fmadd_ps (d0, d0, acc0);
      acc1 = _mm256_fmadd_ps (d1, d1, acc1);
      acc2 = _mm256_fmadd_ps (d2, d2, acc2);
      acc3 = _mm256_fmadd_ps (d3, d3, acc3);
    }
  acc0 = _mm256_add_ps (_mm256_add_ps (acc0, acc1),
                        _mm256_add_ps (acc2, acc3));
  for (; i + 8U <= dim; i += 8U)
    {
      const __m256 d
          = _mm256_sub_ps (_mm256_loadu_ps (a + i), _mm256_loadu_ps (b + i));
      acc0 = _mm256_fmadd_ps (d, d, acc0);
    }

  const __m128 lo = _mm256_castps256_ps128 (acc0);
  const __m128 hi = _mm256_extractf128_ps (acc0, 1);
  __m128 sum128 = _mm_add_ps (lo, hi);
  sum128 = _mm_hadd_ps (sum128, sum128);
  sum128 = _mm_hadd_ps (sum128, sum128);
  float total = _mm_cvtss_f32 (sum128);

  for (; i < dim; ++i)
    {
      const float diff = a[i] - b[i];
      total += diff * diff;
    }
  return total;
}

__attribute__ ((target ("avx512f"))) auto
l2_sq_f32_avx512 (const float *a, const float *b, std::size_t dim) noexcept
    -> float
{
  __m512 acc0 = _mm512_setzero_ps ();
  __m512 acc1 = _mm512_setzero_ps ();
  std::size_t i = 0U;
  for (; i + 32U <= dim; i += 32U)
    {
      const __m512 d0
          = _mm512_sub_ps (_mm512_loadu_ps (a + i), _mm512_loadu_ps (b + i));
      const __m512 d1 = _mm512_sub_ps (_mm512_loadu_ps (a + i + 16U),
                                       _mm512_loadu_ps (b + i + 16U));
      acc0 = _mm512_fmadd_ps (d0, d0, acc0);
      acc1 = _mm512_fmadd_ps (d1, d1, acc1);
    }
  acc0 = _mm512_add_ps (acc0, acc1);
  for (; i + 16U <= dim; i += 16U)
    {
      const __m512 d
          = _mm512_sub_ps (_mm512_loadu_ps (a + i), _mm512_loadu_ps (b + i));
      acc0 = _mm512_fmadd_ps (d, d, acc0);
    }
  float total = _mm512_reduce_add_ps (acc0);
  for (; i < dim; ++i)
    {
      const float diff = a[i] - b[i];
      total += diff * diff;
    }
  return total;
}
#endif

auto
l2_sq_f32 (const float *a, const float *b, std::size_t dim) noexcept -> float
{
#if defined(__x86_64__)
  static const int simd_mode = [] () {
    if (__builtin_cpu_supports ("avx512f"))
      {
        return 2;
      }
    if (__builtin_cpu_supports ("avx2") && __builtin_cpu_supports ("fma"))
      {
        return 1;
      }
    return 0;
  }();
  if (simd_mode == 2)
    {
      return l2_sq_f32_avx512 (a, b, dim);
    }
  if (simd_mode == 1)
    {
      return l2_sq_f32_avx2 (a, b, dim);
    }
#endif
  return l2_sq_f32_scalar (a, b, dim);
}

auto
l2_sq_f64_exact (const float *a, const float *b, std::size_t dim) noexcept
    -> double
{
  double sum = 0.0;
  for (std::size_t i = 0U; i < dim; ++i)
    {
      const double diff
          = static_cast<double> (a[i]) - static_cast<double> (b[i]);
      sum += diff * diff;
    }
  return sum;
}

/* Per-thread O(1) generation-tagged visited table + pmr scratch arena.  */
struct SearchScratch
{
  std::vector<uint32_t> visited_tags;
  uint32_t current_tag = 1U;
  std::array<std::byte, 65536> arena_buf{};

  void
  prepare (std::size_t num_nodes) noexcept
  {
    if (visited_tags.size () < num_nodes)
      {
        visited_tags.assign (num_nodes, 0U);
      }
    ++current_tag;
    if (current_tag == 0U)
      {
        std::fill (visited_tags.begin (), visited_tags.end (), 0U);
        current_tag = 1U;
      }
  }

  [[nodiscard]] auto
  test_and_set (uint32_t node) noexcept -> bool
  {
    if (visited_tags[node] == current_tag)
      {
        return true;
      }
    visited_tags[node] = current_tag;
    return false;
  }
};

auto
thread_scratch () noexcept -> SearchScratch &
{
  thread_local SearchScratch s;
  return s;
}

class MmapRegion
{
public:
  MmapRegion () noexcept = default;
  MmapRegion (int fd, void *addr, std::size_t len) noexcept
      : fd_ (fd), addr_ (addr), len_ (len)
  {
  }
  MmapRegion (const MmapRegion &) = delete;
  auto operator= (const MmapRegion &) -> MmapRegion & = delete;

  MmapRegion (MmapRegion &&other) noexcept
      : fd_ (other.fd_), addr_ (other.addr_), len_ (other.len_)
  {
    other.fd_ = -1;
    other.addr_ = MAP_FAILED;
    other.len_ = 0U;
  }

  auto
  operator= (MmapRegion &&other) noexcept -> MmapRegion &
  {
    if (this != &other)
      {
        reset ();
        fd_ = other.fd_;
        addr_ = other.addr_;
        len_ = other.len_;
        other.fd_ = -1;
        other.addr_ = MAP_FAILED;
        other.len_ = 0U;
      }
    return *this;
  }

  ~MmapRegion () noexcept { reset (); }

  void
  reset () noexcept
  {
    if (addr_ != nullptr && addr_ != MAP_FAILED && len_ > 0U)
      {
        (void)::munmap (addr_, len_);
      }
    if (fd_ >= 0)
      {
        (void)::close (fd_);
      }
    fd_ = -1;
    addr_ = MAP_FAILED;
    len_ = 0U;
  }

  [[nodiscard]] auto
  bytes () const noexcept -> const std::byte *
  {
    return static_cast<const std::byte *> (addr_);
  }

  [[nodiscard]] auto
  size () const noexcept -> std::size_t
  {
    return len_;
  }

private:
  int fd_ = -1;
  void *addr_ = MAP_FAILED;
  std::size_t len_ = 0U;
};

struct alignas (8) HnswFileHeader
{
  std::array<char, 8> magic = { 'Y', 'H', 'N', 'S', 'W', 'V', '0', '2' };
  uint64_t num_vectors = 0U;
  uint64_t dim = 0U;
  uint32_t m = 16U;
  uint32_t m0 = 32U;
  uint32_t ef_construction = 200U;
  uint32_t ef_search = 64U;
  uint32_t max_level = 0U;
  uint32_t n_coarse = 0U;
  uint64_t fastscan_bytes = 0U;
  uint32_t entry_point = 0U;
  uint32_t header_crc32c = 0U;
};

static_assert (sizeof (HnswFileHeader) == 64U,
               "HnswFileHeader must be 64 bytes");

/* Optimal 4-level Lloyd-Max reconstruction levels for N(0, 1), packing 4
   coordinates per byte (2 per nibble) + 2-byte exact norm bias.  */
alignas (16) constexpr std::array<float, 4> kLloydMax4 = {
  -1.5104F, -0.4528F, 0.4528F, 1.5104F
};

constexpr float kFastScanScale = 3.2F;

[[nodiscard]] inline auto
quantize_lloyd4 (float z) noexcept -> uint8_t
{
  if (z < 0.0F)
    {
      return (z < -0.9816F) ? 0U : 1U;
    }
  return (z < 0.9816F) ? 2U : 3U;
}

[[nodiscard]] inline auto
compute_hnsw_level (uint32_t u, std::size_t m) noexcept -> uint8_t
{
  uint64_t z = (static_cast<uint64_t> (u) + 1ULL) * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31U);
  const double u01
      = static_cast<double> ((z >> 11U) + 1ULL)
        * (1.0 / static_cast<double> ((1ULL << 53U) + 1ULL));
  const double ml
      = 1.0 / std::log (static_cast<double> (std::max<std::size_t> (2U, m)));
  const double lvl = std::floor (-std::log (u01) * ml);
  return static_cast<uint8_t> (std::clamp (lvl, 0.0, 16.0));
}

void
build_fastscan_query_lut (const float *q_ptr, std::size_t dim,
                          uint8_t *out_lut32) noexcept
{
  const std::size_t n_cols = (dim + 3U) / 4U;
  const std::size_t n_nibbles = n_cols * 2U;

  for (std::size_t nib_i = 0U; nib_i < n_nibbles; ++nib_i)
    {
      const std::size_t da = nib_i * 2U;
      const std::size_t db = da + 1U;
      uint8_t *dst = out_lut32 + nib_i * 32U;

      std::array<float, 4> ca{};
      float min_a = 0.0F;
      if (da < dim)
        {
          min_a = 1.0e30F;
          const float qa = q_ptr[da];
          for (std::size_t k = 0U; k < 4U; ++k)
            {
              const float diff = qa - kLloydMax4[k];
              ca[k] = diff * diff;
              if (ca[k] < min_a)
                {
                  min_a = ca[k];
                }
            }
        }

      std::array<float, 4> cb{};
      float min_b = 0.0F;
      if (db < dim)
        {
          min_b = 1.0e30F;
          const float qb = q_ptr[db];
          for (std::size_t k = 0U; k < 4U; ++k)
            {
              const float diff = qb - kLloydMax4[k];
              cb[k] = diff * diff;
              if (cb[k] < min_b)
                {
                  min_b = cb[k];
                }
            }
        }

      for (std::size_t kb = 0U; kb < 4U; ++kb)
        {
          for (std::size_t ka = 0U; ka < 4U; ++ka)
            {
              const std::size_t idx = ka | (kb << 2U);
              const float val
                  = (ca[ka] - min_a) + (cb[kb] - min_b);
              const float sc = std::min (63.0F, val * kFastScanScale);
              const auto qv = static_cast<uint8_t> (sc + 0.5F);
              dst[idx] = qv;
              dst[16U + idx] = qv;
            }
        }
    }
}

#if defined(__x86_64__)
__attribute__ ((target ("avx2"))) auto
score_fastscan_block_avx2 (const uint8_t *blk_ptr, const uint8_t *lut32_ptr,
                           std::size_t n_pairs, uint16_t threshold,
                           uint16_t *out_scores32) noexcept -> uint32_t
{
  const __m256i mask_lo = _mm256_set1_epi8 (0x0F);
  const __m256i zero = _mm256_setzero_si256 ();
  const __m256i b0_15
      = _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (blk_ptr));
  const __m256i b16_31
      = _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (blk_ptr + 32U));
  __m256i acc_lo = _mm256_permute2x128_si256 (b0_15, b16_31, 0x20);
  __m256i acc_hi = _mm256_permute2x128_si256 (b0_15, b16_31, 0x31);

  const uint8_t *codes_ptr = blk_ptr + 64U;
  const uint16_t th_clamped = std::min<uint16_t> (threshold, 0x7FFFU);
  const __m256i v_thresh
      = _mm256_set1_epi16 (static_cast<int16_t> (th_clamped));

  const std::size_t half_pairs
      = (n_pairs >= 24U) ? (((n_pairs * 3U) / 16U) * 2U) : 0U;
  const std::size_t mid_pairs
      = (n_pairs >= 16U) ? (((n_pairs * 5U) / 16U) * 2U) : 0U;
  std::size_t p = 0U;
  for (; p + 2U <= half_pairs; p += 2U)
    {
      const __m256i v0 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + p * 32U));
      const __m256i v1 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + (p + 1U) * 32U));

      const __m256i lut0_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 0U) * 32U));
      const __m256i lut0_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 1U) * 32U));
      const __m256i lut1_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 2U) * 32U));
      const __m256i lut1_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 3U) * 32U));

      const __m256i d0_lo
          = _mm256_shuffle_epi8 (lut0_lo, _mm256_and_si256 (v0, mask_lo));
      const __m256i d0_hi = _mm256_shuffle_epi8 (
          lut0_hi, _mm256_and_si256 (_mm256_srli_epi16 (v0, 4), mask_lo));
      const __m256i d1_lo
          = _mm256_shuffle_epi8 (lut1_lo, _mm256_and_si256 (v1, mask_lo));
      const __m256i d1_hi = _mm256_shuffle_epi8 (
          lut1_hi, _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), mask_lo));

      const __m256i sum4 = _mm256_add_epi8 (_mm256_add_epi8 (d0_lo, d0_hi),
                                            _mm256_add_epi8 (d1_lo, d1_hi));
      acc_lo = _mm256_add_epi16 (acc_lo, _mm256_unpacklo_epi8 (sum4, zero));
      acc_hi = _mm256_add_epi16 (acc_hi, _mm256_unpackhi_epi8 (sum4, zero));
    }

  if (half_pairs >= 12U && th_clamped > 256U && th_clamped < 0x7F00U)
    {
      const auto slack1
          = static_cast<uint16_t> (((th_clamped - 200U) * 12U) >> 5U);
      const __m256i v_thresh_half
          = _mm256_set1_epi16 (static_cast<int16_t> (th_clamped - slack1));
      const __m256i ch_lo = _mm256_cmpgt_epi16 (v_thresh_half, acc_lo);
      const __m256i ch_hi = _mm256_cmpgt_epi16 (v_thresh_half, acc_hi);
      if (_mm256_testz_si256 (ch_lo, ch_lo) != 0
          && _mm256_testz_si256 (ch_hi, ch_hi) != 0)
        {
          return 0U;
        }
    }

  for (; p + 2U <= mid_pairs; p += 2U)
    {
      const __m256i v0 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + p * 32U));
      const __m256i v1 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + (p + 1U) * 32U));

      const __m256i lut0_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 0U) * 32U));
      const __m256i lut0_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 1U) * 32U));
      const __m256i lut1_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 2U) * 32U));
      const __m256i lut1_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 3U) * 32U));

      const __m256i d0_lo
          = _mm256_shuffle_epi8 (lut0_lo, _mm256_and_si256 (v0, mask_lo));
      const __m256i d0_hi = _mm256_shuffle_epi8 (
          lut0_hi, _mm256_and_si256 (_mm256_srli_epi16 (v0, 4), mask_lo));
      const __m256i d1_lo
          = _mm256_shuffle_epi8 (lut1_lo, _mm256_and_si256 (v1, mask_lo));
      const __m256i d1_hi = _mm256_shuffle_epi8 (
          lut1_hi, _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), mask_lo));

      const __m256i sum4 = _mm256_add_epi8 (_mm256_add_epi8 (d0_lo, d0_hi),
                                            _mm256_add_epi8 (d1_lo, d1_hi));
      acc_lo = _mm256_add_epi16 (acc_lo, _mm256_unpacklo_epi8 (sum4, zero));
      acc_hi = _mm256_add_epi16 (acc_hi, _mm256_unpackhi_epi8 (sum4, zero));
    }

  if (mid_pairs >= 16U && th_clamped > 256U && th_clamped < 0x7F00U)
    {
      const auto slack
          = static_cast<uint16_t> (((th_clamped - 200U) * 6U) >> 5U);
      const __m256i v_thresh_mid
          = _mm256_set1_epi16 (static_cast<int16_t> (th_clamped - slack));
      const __m256i cm_lo = _mm256_cmpgt_epi16 (v_thresh_mid, acc_lo);
      const __m256i cm_hi = _mm256_cmpgt_epi16 (v_thresh_mid, acc_hi);
      if (_mm256_testz_si256 (cm_lo, cm_lo) != 0
          && _mm256_testz_si256 (cm_hi, cm_hi) != 0)
        {
          return 0U;
        }
    }

  for (; p + 2U <= n_pairs; p += 2U)
    {
      const __m256i v0 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + p * 32U));
      const __m256i v1 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + (p + 1U) * 32U));

      const __m256i lut0_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 0U) * 32U));
      const __m256i lut0_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 1U) * 32U));
      const __m256i lut1_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 2U) * 32U));
      const __m256i lut1_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 3U) * 32U));

      const __m256i d0_lo
          = _mm256_shuffle_epi8 (lut0_lo, _mm256_and_si256 (v0, mask_lo));
      const __m256i d0_hi = _mm256_shuffle_epi8 (
          lut0_hi, _mm256_and_si256 (_mm256_srli_epi16 (v0, 4), mask_lo));
      const __m256i d1_lo
          = _mm256_shuffle_epi8 (lut1_lo, _mm256_and_si256 (v1, mask_lo));
      const __m256i d1_hi = _mm256_shuffle_epi8 (
          lut1_hi, _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), mask_lo));

      const __m256i sum4 = _mm256_add_epi8 (_mm256_add_epi8 (d0_lo, d0_hi),
                                            _mm256_add_epi8 (d1_lo, d1_hi));
      acc_lo = _mm256_add_epi16 (acc_lo, _mm256_unpacklo_epi8 (sum4, zero));
      acc_hi = _mm256_add_epi16 (acc_hi, _mm256_unpackhi_epi8 (sum4, zero));
    }

  for (; p < n_pairs; ++p)
    {
      const __m256i v0 = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (codes_ptr + p * 32U));
      const __m256i lut0_lo = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 0U) * 32U));
      const __m256i lut0_hi = _mm256_loadu_si256 (
          reinterpret_cast<const __m256i *> (lut32_ptr + (p * 2U + 1U) * 32U));
      const __m256i d0_lo
          = _mm256_shuffle_epi8 (lut0_lo, _mm256_and_si256 (v0, mask_lo));
      const __m256i d0_hi = _mm256_shuffle_epi8 (
          lut0_hi, _mm256_and_si256 (_mm256_srli_epi16 (v0, 4), mask_lo));
      const __m256i sum2 = _mm256_add_epi8 (d0_lo, d0_hi);
      acc_lo = _mm256_add_epi16 (acc_lo, _mm256_unpacklo_epi8 (sum2, zero));
      acc_hi = _mm256_add_epi16 (acc_hi, _mm256_unpackhi_epi8 (sum2, zero));
    }

  const __m256i cmp_lo = _mm256_cmpgt_epi16 (v_thresh, acc_lo);
  const __m256i cmp_hi = _mm256_cmpgt_epi16 (v_thresh, acc_hi);
  const auto mask = static_cast<uint32_t> (
      _mm256_movemask_epi8 (_mm256_packs_epi16 (cmp_lo, cmp_hi)));
  if (mask == 0U)
    {
      return 0U;
    }

  _mm256_storeu_si256 (reinterpret_cast<__m256i *> (out_scores32),
                       _mm256_permute2x128_si256 (acc_lo, acc_hi, 0x20));
  _mm256_storeu_si256 (reinterpret_cast<__m256i *> (out_scores32 + 16U),
                       _mm256_permute2x128_si256 (acc_lo, acc_hi, 0x31));
  return mask;
}
#endif

auto
score_fastscan_block (const uint8_t *blk_ptr, const uint8_t *lut32_ptr,
                      std::size_t n_pairs, uint16_t threshold,
                      uint16_t *out_scores32) noexcept -> uint32_t
{
#if defined(__x86_64__)
  static const bool has_avx2 = __builtin_cpu_supports ("avx2");
  if (has_avx2)
    {
      return score_fastscan_block_avx2 (blk_ptr, lut32_ptr, n_pairs, threshold,
                                        out_scores32);
    }
#endif
  std::memcpy (out_scores32, blk_ptr, 32U * sizeof (uint16_t));
  const uint8_t *codes_ptr = blk_ptr + 64U;
  for (std::size_t p = 0U; p < n_pairs; ++p)
    {
      const uint8_t *col = codes_ptr + p * 32U;
      const uint8_t *l0 = lut32_ptr + (p * 2U + 0U) * 32U;
      const uint8_t *l1 = lut32_ptr + (p * 2U + 1U) * 32U;
      for (std::size_t s = 0U; s < 32U; ++s)
        {
          const uint8_t b = col[s];
          out_scores32[s] = static_cast<uint16_t> (
              out_scores32[s] + l0[b & 0x0FU] + l1[(b >> 4U) & 0x0FU]);
        }
    }
  uint32_t mask = 0U;
  for (std::size_t s = 0U; s < 32U; ++s)
    {
      if (out_scores32[s] < threshold)
        {
          mask |= (1U << s);
        }
    }
  return mask;
}

} /* namespace */

/* =========================================================================
   FlatIndex Implementation
   ========================================================================= */

struct FlatIndex::Impl
{
  std::size_t dim = 0U;
  mutable std::shared_mutex mu;
  std::vector<float> vectors;
  std::vector<uint64_t> ids;
  std::vector<VectorMetadata> metadata;
};

FlatIndex::FlatIndex (std::size_t dim) noexcept
    : impl_ (std::make_unique<Impl> ())
{
  impl_->dim = dim;
}

FlatIndex::FlatIndex (FlatIndex &&other) noexcept = default;
auto FlatIndex::operator= (FlatIndex &&other) noexcept
    -> FlatIndex & = default;
FlatIndex::~FlatIndex () noexcept = default;

auto
FlatIndex::add (std::span<const float> vec, uint64_t id,
                VectorMetadata meta) noexcept -> Result<void>
{
  if (!impl_ || impl_->dim == 0U || vec.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_ ? impl_->dim : 0U,
          vec.size (), "Vector dimension does not match FlatIndex dimension"));
    }
  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  impl_->vectors.insert (impl_->vectors.end (), vec.begin (), vec.end ());
  impl_->ids.push_back (id);
  meta.id = id;
  impl_->metadata.push_back (meta);
  return {};
}

auto
FlatIndex::add_batch (std::span<const float> vectors,
                      std::size_t num_vectors) noexcept -> Result<void>
{
  if (!impl_ || impl_->dim == 0U
      || vectors.size () != num_vectors * impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch,
          impl_ ? num_vectors * impl_->dim : 0U, vectors.size (),
          "Batch vector span size does not match num_vectors * dimension"));
    }
  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t base_idx = impl_->ids.size ();
  impl_->vectors.insert (impl_->vectors.end (), vectors.begin (),
                         vectors.end ());
  impl_->ids.reserve (base_idx + num_vectors);
  impl_->metadata.reserve (base_idx + num_vectors);
  for (std::size_t i = 0U; i < num_vectors; ++i)
    {
      const uint64_t id = base_idx + i;
      impl_->ids.push_back (id);
      VectorMetadata m{};
      m.id = id;
      impl_->metadata.push_back (m);
    }
  return {};
}

auto
FlatIndex::search (std::span<const float> query,
                   std::size_t k) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  if (!impl_ || query.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_ ? impl_->dim : 0U,
          query.size (), "Query dimension mismatch in FlatIndex::search"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t n = impl_->ids.size ();
  if (n == 0U || k == 0U)
    {
      return std::vector<NeighborResult>{};
    }

  std::vector<NeighborResult> all (n);
  const float *q_ptr = query.data ();
  const float *v_base = impl_->vectors.data ();
  const std::size_t d = impl_->dim;

  for (std::size_t i = 0U; i < n; ++i)
    {
      all[i].id = impl_->ids[i];
      all[i].distance = l2_sq_f32 (q_ptr, v_base + i * d, d);
    }

  const std::size_t k_clamped = std::min (k, n);
  std::partial_sort (all.begin (),
                     all.begin () + static_cast<std::ptrdiff_t> (k_clamped),
                     all.end (),
                     [] (const NeighborResult &a, const NeighborResult &b) {
                       return a.distance < b.distance;
                     });
  all.resize (k_clamped);
  return all;
}

auto
FlatIndex::search_exact_fp64 (std::span<const float> query,
                              std::size_t k) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  if (!impl_ || query.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_ ? impl_->dim : 0U,
          query.size (),
          "Query dimension mismatch in FlatIndex::search_exact_fp64"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t n = impl_->ids.size ();
  if (n == 0U || k == 0U)
    {
      return std::vector<NeighborResult>{};
    }

  struct ExactPair
  {
    uint64_t id = 0U;
    double dist = 0.0;
  };
  std::vector<ExactPair> all (n);
  const float *q_ptr = query.data ();
  const float *v_base = impl_->vectors.data ();
  const std::size_t d = impl_->dim;

  for (std::size_t i = 0U; i < n; ++i)
    {
      all[i].id = impl_->ids[i];
      all[i].dist = l2_sq_f64_exact (q_ptr, v_base + i * d, d);
    }

  const std::size_t k_clamped = std::min (k, n);
  std::partial_sort (all.begin (),
                     all.begin () + static_cast<std::ptrdiff_t> (k_clamped),
                     all.end (), [] (const ExactPair &a, const ExactPair &b) {
                       return a.dist < b.dist;
                     });

  std::vector<NeighborResult> out (k_clamped);
  for (std::size_t i = 0U; i < k_clamped; ++i)
    {
      out[i].id = all[i].id;
      out[i].distance = static_cast<float> (all[i].dist);
    }
  return out;
}

auto
FlatIndex::search_filtered (std::span<const float> query, std::size_t k,
                            const MetadataFilter &filter) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  if (!impl_ || query.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_ ? impl_->dim : 0U,
          query.size (),
          "Query dimension mismatch in FlatIndex::search_filtered"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t n = impl_->ids.size ();
  if (n == 0U || k == 0U)
    {
      return std::vector<NeighborResult>{};
    }

  std::vector<NeighborResult> candidates;
  candidates.reserve (n / 2U + 1U);
  const float *q_ptr = query.data ();
  const float *v_base = impl_->vectors.data ();
  const std::size_t d = impl_->dim;

  for (std::size_t i = 0U; i < n; ++i)
    {
      if (filter.matches (impl_->metadata[i]))
        {
          candidates.push_back (
              { impl_->ids[i], l2_sq_f32 (q_ptr, v_base + i * d, d) });
        }
    }

  const std::size_t k_clamped = std::min (k, candidates.size ());
  std::partial_sort (
      candidates.begin (),
      candidates.begin () + static_cast<std::ptrdiff_t> (k_clamped),
      candidates.end (),
      [] (const NeighborResult &a, const NeighborResult &b) {
        return a.distance < b.distance;
      });
  candidates.resize (k_clamped);
  return candidates;
}

auto
FlatIndex::size () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return impl_->ids.size ();
}

auto
FlatIndex::dimension () const noexcept -> std::size_t
{
  return impl_ ? impl_->dim : 0U;
}

/* =========================================================================
   HnswIndex Implementation
   ========================================================================= */

struct HnswIndex::Impl
{
  std::size_t dim = 0U;
  HnswParams params;
  std::size_t m0 = 32U;
  std::size_t max_lvl = 0U;
  std::size_t n_coarse = 0U;
  std::size_t code_bytes = 0U;
  mutable std::shared_mutex mu;

  /* Heap-backed storage (used when building or incrementally inserting).  */
  std::vector<float> vectors;
  std::vector<uint32_t> layer0_links;
  std::vector<uint8_t> layer0_counts;
  std::vector<uint64_t> ids;
  std::vector<uint8_t> node_levels;
  std::vector<VectorMetadata> metadata;
  std::vector<std::vector<std::vector<uint32_t>>> upper_links;

  /* FastScan 4-bit Lloyd-Max inverted blocks for sub-millisecond routing.  */
  std::vector<float> coarse_centroids;
  std::vector<uint32_t> list_offsets;
  std::vector<uint32_t> list_node_ids;
  std::vector<uint8_t> fastscan_blocks;

  /* Zero-copy mmap view (used when loaded from disk via load_mmap).  */
  MmapRegion mmap_region;
  bool is_mmap_backed = false;
  const float *vec_ptr_view = nullptr;
  const uint32_t *links_ptr_view = nullptr;
  const uint8_t *counts_ptr_view = nullptr;
  const uint64_t *ids_ptr_view = nullptr;
  const uint8_t *levels_ptr_view = nullptr;
  const float *coarse_ptr_view = nullptr;
  const uint32_t *list_offsets_view = nullptr;
  const uint32_t *list_nodes_view = nullptr;
  const uint8_t *fastscan_view = nullptr;
  std::size_t fastscan_bytes_len = 0U;
  std::size_t num_nodes = 0U;
  uint32_t entry_point = 0U;
  mutable std::atomic<bool> mmap_warmed{ false };

  void
  sync_views_to_heap () noexcept
  {
    is_mmap_backed = false;
    num_nodes = ids.size ();
    vec_ptr_view = vectors.data ();
    links_ptr_view = layer0_links.data ();
    counts_ptr_view = layer0_counts.data ();
    ids_ptr_view = ids.data ();
    levels_ptr_view = node_levels.empty () ? nullptr : node_levels.data ();
    coarse_ptr_view
        = coarse_centroids.empty () ? nullptr : coarse_centroids.data ();
    list_offsets_view = list_offsets.empty () ? nullptr : list_offsets.data ();
    list_nodes_view = list_node_ids.empty () ? nullptr : list_node_ids.data ();
    fastscan_view
        = fastscan_blocks.empty () ? nullptr : fastscan_blocks.data ();
    fastscan_bytes_len = fastscan_blocks.size ();
  }

  void
  promote_mmap_to_heap_if_needed ()
  {
    if (!is_mmap_backed)
      {
        return;
      }
    vectors.assign (vec_ptr_view, vec_ptr_view + num_nodes * dim);
    layer0_links.assign (links_ptr_view, links_ptr_view + num_nodes * m0);
    layer0_counts.assign (counts_ptr_view, counts_ptr_view + num_nodes);
    ids.assign (ids_ptr_view, ids_ptr_view + num_nodes);
    if (levels_ptr_view != nullptr)
      {
        node_levels.assign (levels_ptr_view, levels_ptr_view + num_nodes);
      }
    else
      {
        node_levels.assign (num_nodes, 0U);
      }
    if (coarse_ptr_view != nullptr && n_coarse > 0U)
      {
        coarse_centroids.assign (coarse_ptr_view,
                                 coarse_ptr_view + n_coarse * dim);
        list_offsets.assign (list_offsets_view,
                             list_offsets_view + n_coarse + 1U);
        const std::size_t total_slots = list_offsets_view[n_coarse];
        list_node_ids.assign (list_nodes_view, list_nodes_view + total_slots);
        fastscan_blocks.assign (fastscan_view,
                                fastscan_view + fastscan_bytes_len);
      }
    mmap_region.reset ();
    sync_views_to_heap ();
  }

  [[nodiscard]] auto
  node_vec (uint32_t u) const noexcept -> const float *
  {
    return vec_ptr_view + static_cast<std::size_t> (u) * dim;
  }

  [[nodiscard]] auto
  node_links (uint32_t u) const noexcept -> const uint32_t *
  {
    return links_ptr_view + static_cast<std::size_t> (u) * m0;
  }

  /* Greedy descent through upper HNSW layers (max_lvl down to 1).  */
  [[nodiscard]] auto
  greedy_upper_descent (const float *q_ptr) const noexcept -> uint32_t
  {
    uint32_t cur = entry_point < num_nodes ? entry_point : 0U;
    if (max_lvl == 0U || upper_links.empty ())
      {
        return cur;
      }
    float cur_d = l2_sq_f32 (q_ptr, node_vec (cur), dim);
    for (std::size_t lvl = max_lvl; lvl >= 1U; --lvl)
      {
        bool changed = true;
        while (changed)
          {
            changed = false;
            if (cur >= upper_links.size ()
                || lvl - 1U >= upper_links[cur].size ())
              {
                break;
              }
            const auto &nbrs = upper_links[cur][lvl - 1U];
            for (const uint32_t v : nbrs)
              {
                if (v < num_nodes)
                  {
                    const float d = l2_sq_f32 (q_ptr, node_vec (v), dim);
                    if (d < cur_d)
                      {
                        cur_d = d;
                        cur = v;
                        changed = true;
                      }
                  }
              }
          }
      }
    return cur;
  }

  /* Hybrid HNSW search: upper-layer descent + AVX2 4-bit FastScan seeding
     + Layer-0 beam refinement.  */
  void
  search_layer0_beam (const float *q_ptr, std::size_t ef,
                      const MetadataFilter *pre_filter,
                      std::vector<NeighborResult> &out_top) const noexcept
  {
    auto &scratch = thread_scratch ();
    scratch.prepare (num_nodes);

    struct Cand
    {
      float dist = 0.0F;
      uint32_t node = 0U;
    };

    std::pmr::monotonic_buffer_resource arena (scratch.arena_buf.data (),
                                               scratch.arena_buf.size ());
    std::pmr::vector<Cand> candidates (&arena);
    std::pmr::vector<Cand> top_results (&arena);
    const std::size_t ef_beam = std::max<std::size_t> (ef, 64U);
    candidates.reserve (ef_beam * 2U + 64U);
    top_results.reserve (ef_beam + 64U);

    auto min_cmp
        = [] (const Cand &a, const Cand &b) { return a.dist > b.dist; };
    auto max_cmp
        = [] (const Cand &a, const Cand &b) { return a.dist < b.dist; };

    auto push_seed = [&] (uint32_t seed) {
      if (seed >= num_nodes || scratch.test_and_set (seed))
        {
          return;
        }
      const float d = l2_sq_f32 (q_ptr, node_vec (seed), dim);
      candidates.push_back ({ d, seed });
      std::push_heap (candidates.begin (), candidates.end (), min_cmp);
      if (pre_filter == nullptr || metadata.empty ()
          || (seed < metadata.size () && pre_filter->matches (metadata[seed])))
        {
          top_results.push_back ({ d, seed });
          std::push_heap (top_results.begin (), top_results.end (), max_cmp);
          if (top_results.size () > ef_beam)
            {
              std::pop_heap (top_results.begin (), top_results.end (),
                             max_cmp);
              top_results.pop_back ();
            }
        }
    };

    /* 1. Descend upper HNSW layers from entry_point.  */
    const uint32_t ep0 = greedy_upper_descent (q_ptr);
    push_seed (ep0);

    /* 2. If FastScan blocks are available, scan top coarse cells using
       AVX2 VPSHUFB LUTs to seed Layer-0 beam with global basin winners.  */
    if (coarse_ptr_view != nullptr && list_offsets_view != nullptr
        && list_nodes_view != nullptr && fastscan_view != nullptr
        && n_coarse > 0U && code_bytes > 0U)
      {
        std::array<Cand, 256> best_cells{};
        const std::size_t total_cells
            = std::min<std::size_t> (n_coarse, best_cells.size ());
        for (std::size_t c = 0U; c < total_cells; ++c)
          {
            best_cells[c] = {
              l2_sq_f32 (q_ptr, coarse_ptr_view + c * dim, dim),
              static_cast<uint32_t> (c)
            };
          }
        std::sort (best_cells.begin (), best_cells.begin () + total_cells,
                   max_cmp);
        const std::size_t cell_cnt = std::min<std::size_t> (
            total_cells,
            std::max<std::size_t> ((total_cells * 15U) / 16U, 24U));

        alignas (32) uint8_t lut32[8192];
        if (code_bytes <= 128U)
          {
            build_fastscan_query_lut (q_ptr, dim, lut32);

            struct FsCand
            {
              uint16_t score = 0xFFFFU;
              uint32_t node = 0U;
            };
            constexpr std::size_t kRerankCap = 496U;
            std::array<FsCand, kRerankCap> top_fs{};
            std::size_t fs_cnt = 0U;
            uint16_t threshold = 0xFFFFU;
            auto fs_max_cmp = [] (const FsCand &a, const FsCand &b) {
              return a.score < b.score;
            };

            const std::size_t blk_stride = (code_bytes + 2U) * 32U;
            alignas (32) uint16_t blk_scores[32];
            for (std::size_t pi = 0U; pi < cell_cnt; ++pi)
              {
                const uint32_t cid = best_cells[pi].node;
                const uint32_t start = list_offsets_view[cid];
                const uint32_t full_end = list_offsets_view[cid + 1U];
                uint32_t end = full_end;
                if (num_nodes > 120000U && full_end > start + 64U)
                  {
                    const uint32_t full_len = full_end - start;
                    const uint32_t num
                        = (pi < (cell_cnt * 5U) / 8U)
                              ? ((full_len * 7U) / 8U)
                              : ((full_len * 3U) / 4U);
                    const uint32_t scan_len = (num + 31U) & ~uint32_t (31U);
                    end = std::min (
                        full_end, start + std::max<uint32_t> (64U, scan_len));
                  }
                for (uint32_t idx = start; idx < end; idx += 32U)
                  {
                    const std::size_t blk_idx = idx / 32U;
                    const uint8_t *blk_ptr
                        = fastscan_view + blk_idx * blk_stride;
                    uint32_t mask = score_fastscan_block (
                        blk_ptr, lut32, code_bytes, threshold, blk_scores);
                    if (mask == 0U)
                      {
                        continue;
                      }
                    const uint32_t lim = std::min<uint32_t> (32U, end - idx);
                    if (lim < 32U)
                      {
                        mask &= (1U << lim) - 1U;
                      }
                    while (mask != 0U)
                      {
                        const auto lane
                            = static_cast<uint32_t> (std::countr_zero (mask));
                        mask &= mask - 1U;
                        const uint16_t sc = blk_scores[lane];
                        if (sc >= threshold)
                          {
                            continue;
                          }
                        const uint32_t u = list_nodes_view[idx + lane];
                        if (u >= num_nodes)
                          {
                            continue;
                          }
                        if (fs_cnt < kRerankCap)
                          {
                            top_fs[fs_cnt++] = { sc, u };
                            if (fs_cnt == kRerankCap)
                              {
                                std::make_heap (top_fs.begin (),
                                                top_fs.begin () + fs_cnt,
                                                fs_max_cmp);
                                threshold = top_fs[0].score;
                              }
                          }
                        else
                          {
                            std::pop_heap (top_fs.begin (),
                                           top_fs.begin () + fs_cnt,
                                           fs_max_cmp);
                            top_fs[fs_cnt - 1U] = { sc, u };
                            std::push_heap (top_fs.begin (),
                                            top_fs.begin () + fs_cnt,
                                            fs_max_cmp);
                            threshold = top_fs[0].score;
                          }
                      }
                  }
              }

            for (std::size_t i = 0U; i < fs_cnt; ++i)
              {
                if (i + 3U < fs_cnt && top_fs[i + 3U].node < num_nodes)
                  {
                    const float *pf = node_vec (top_fs[i + 3U].node);
                    __builtin_prefetch (pf, 0, 1);
                    __builtin_prefetch (pf + 32U, 0, 1);
                    __builtin_prefetch (pf + 64U, 0, 1);
                    __builtin_prefetch (pf + 96U, 0, 1);
                  }
                push_seed (top_fs[i].node);
              }
          }
      }
    else if (num_nodes <= 16384U)
      {
        /* Small index without FastScan blocks: seed from a light stride.  */
        const std::size_t stride = std::max<std::size_t> (1U, num_nodes / 64U);
        for (std::size_t h = 0U; h < num_nodes; h += stride)
          {
            push_seed (static_cast<uint32_t> (h));
          }
      }

    /* 3. Layer-0 HNSW beam expansion over graph edges.  */
    std::size_t expansions = 0U;
    const std::size_t max_expansions
        = (fastscan_view != nullptr) ? 24U
                                     : std::max<std::size_t> (ef * 3U, 96U);
    while (!candidates.empty () && expansions < max_expansions)
      {
        std::pop_heap (candidates.begin (), candidates.end (), min_cmp);
        const Cand curr = candidates.back ();
        candidates.pop_back ();

        if (top_results.size () >= ef_beam
            && curr.dist > top_results.front ().dist)
          {
            break;
          }
        ++expansions;

        const uint32_t *nbrs = node_links (curr.node);
        const uint8_t cnt = counts_ptr_view[curr.node];
        for (uint8_t j = 0U; j < cnt; ++j)
          {
            const uint32_t v = nbrs[j];
            if (j + 1U < cnt && nbrs[j + 1U] < num_nodes)
              {
                __builtin_prefetch (node_vec (nbrs[j + 1U]), 0, 1);
              }
            if (v >= num_nodes || scratch.test_and_set (v))
              {
                continue;
              }
            const float d = l2_sq_f32 (q_ptr, node_vec (v), dim);
            if (top_results.size () < ef_beam
                || d < top_results.front ().dist)
              {
                candidates.push_back ({ d, v });
                std::push_heap (candidates.begin (), candidates.end (),
                                min_cmp);
                if (pre_filter == nullptr || metadata.empty ()
                    || (v < metadata.size ()
                        && pre_filter->matches (metadata[v])))
                  {
                    top_results.push_back ({ d, v });
                    std::push_heap (top_results.begin (), top_results.end (),
                                    max_cmp);
                    if (top_results.size () > ef_beam)
                      {
                        std::pop_heap (top_results.begin (),
                                       top_results.end (), max_cmp);
                        top_results.pop_back ();
                      }
                  }
              }
          }
      }

    std::sort_heap (top_results.begin (), top_results.end (), max_cmp);
    out_top.resize (top_results.size ());
    for (std::size_t i = 0U; i < top_results.size (); ++i)
      {
        out_top[i].id = ids_ptr_view[top_results[i].node];
        out_top[i].distance = top_results[i].dist;
      }
  }
};

HnswIndex::HnswIndex (std::size_t dim, HnswParams params) noexcept
    : impl_ (std::make_unique<Impl> ())
{
  impl_->dim = dim;
  impl_->params = params;
  impl_->m0 = std::max<std::size_t> (8U, params.m * 2U);
}

HnswIndex::HnswIndex (HnswIndex &&other) noexcept = default;
auto HnswIndex::operator= (HnswIndex &&other) noexcept
    -> HnswIndex & = default;
HnswIndex::~HnswIndex () noexcept = default;

auto
HnswIndex::add (std::span<const float> vec, uint64_t id,
                VectorMetadata meta) noexcept -> Result<void>
{
  if (!impl_ || impl_->dim == 0U || vec.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_ ? impl_->dim : 0U,
          vec.size (), "Vector dimension mismatch in HnswIndex::add"));
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  impl_->promote_mmap_to_heap_if_needed ();

  const auto new_idx = static_cast<uint32_t> (impl_->ids.size ());
  impl_->vectors.insert (impl_->vectors.end (), vec.begin (), vec.end ());
  impl_->ids.push_back (id);
  if (impl_->metadata.size () < new_idx)
    {
      impl_->metadata.resize (new_idx);
      for (std::size_t i = 0U; i < new_idx; ++i)
        {
          impl_->metadata[i].id = impl_->ids[i];
        }
    }
  meta.id = id;
  impl_->metadata.push_back (meta);

  const uint8_t lvl = compute_hnsw_level (new_idx, impl_->params.m);
  impl_->node_levels.push_back (lvl);
  if (lvl > impl_->max_lvl)
    {
      impl_->max_lvl = lvl;
      impl_->entry_point = new_idx;
    }

  impl_->layer0_links.resize (static_cast<std::size_t> (new_idx + 1U)
                                  * impl_->m0,
                              UINT32_MAX);
  impl_->layer0_counts.push_back (0U);
  impl_->sync_views_to_heap ();

  if (new_idx == 0U)
    {
      impl_->entry_point = 0U;
      IndexStats::global ().set_memory_usage (
          impl_->vectors.size () * sizeof (float)
          + impl_->layer0_links.size () * sizeof (uint32_t)
          + impl_->layer0_counts.size () * sizeof (uint8_t));
      return {};
    }

  /* Incremental HNSW insertion: search layer 0 with ef_construction to find
     M best neighbors, then link bidirectionally.  */
  const std::size_t prev_nodes = new_idx;
  impl_->num_nodes = prev_nodes;
  std::vector<NeighborResult> cands;
  impl_->search_layer0_beam (vec.data (), impl_->params.ef_construction,
                             nullptr, cands);
  impl_->num_nodes = prev_nodes + 1U;

  const std::size_t m_links = std::min (impl_->params.m, cands.size ());
  uint32_t *my_links = impl_->layer0_links.data ()
                       + static_cast<std::size_t> (new_idx) * impl_->m0;
  for (std::size_t j = 0U; j < m_links; ++j)
    {
      auto nbr_node = static_cast<uint32_t> (cands[j].id);
      if (nbr_node >= prev_nodes || impl_->ids[nbr_node] != cands[j].id)
        {
          for (uint32_t u = 0U; u < prev_nodes; ++u)
            {
              if (impl_->ids[u] == cands[j].id)
                {
                  nbr_node = u;
                  break;
                }
            }
        }
      my_links[j] = nbr_node;
      impl_->layer0_counts[new_idx] = static_cast<uint8_t> (j + 1U);

      /* Add reverse edge from nbr_node -> new_idx.  */
      uint32_t *rev_links = impl_->layer0_links.data ()
                            + static_cast<std::size_t> (nbr_node) * impl_->m0;
      const uint8_t rev_cnt = impl_->layer0_counts[nbr_node];
      if (rev_cnt < impl_->m0)
        {
          rev_links[rev_cnt] = new_idx;
          impl_->layer0_counts[nbr_node] = static_cast<uint8_t> (rev_cnt + 1U);
        }
      else
        {
          const float dist_cand = cands[j].distance;
          std::size_t worst_k = 0U;
          float worst_d = -1.0F;
          for (std::size_t k = 0U; k < impl_->m0; ++k)
            {
              const float dk = l2_sq_f32 (impl_->node_vec (nbr_node),
                                          impl_->node_vec (rev_links[k]),
                                          impl_->dim);
              if (dk > worst_d)
                {
                  worst_d = dk;
                  worst_k = k;
                }
            }
          if (dist_cand < worst_d)
            {
              rev_links[worst_k] = new_idx;
            }
        }
    }

  IndexStats::global ().set_memory_usage (
      impl_->vectors.size () * sizeof (float)
      + impl_->layer0_links.size () * sizeof (uint32_t));
  return {};
}

auto
HnswIndex::add_batch (std::span<const float> vectors,
                      std::size_t num_vectors) noexcept -> Result<void>
{
  if (!impl_ || impl_->dim == 0U
      || vectors.size () != num_vectors * impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch,
          impl_ ? num_vectors * impl_->dim : 0U, vectors.size (),
          "Batch vector span size does not match num_vectors * dimension"));
    }
  if (num_vectors == 0U)
    {
      return {};
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  impl_->promote_mmap_to_heap_if_needed ();

  const std::size_t base_n = impl_->ids.size ();
  const std::size_t total_n = base_n + num_vectors;
  const std::size_t d = impl_->dim;
  const std::size_t m = impl_->params.m;
  const std::size_t m0 = impl_->m0;
  const std::size_t code_bytes = (d + 3U) / 4U;
  impl_->code_bytes = code_bytes;

  impl_->vectors.insert (impl_->vectors.end (), vectors.begin (),
                         vectors.end ());
  impl_->ids.resize (total_n);
  impl_->node_levels.resize (total_n);

  std::size_t max_lvl = impl_->max_lvl;
  uint32_t ep = impl_->entry_point;
  for (std::size_t i = base_n; i < total_n; ++i)
    {
      impl_->ids[i] = i;
      const uint8_t lvl
          = compute_hnsw_level (static_cast<uint32_t> (i), m);
      impl_->node_levels[i] = lvl;
      if (lvl > max_lvl)
        {
          max_lvl = lvl;
          ep = static_cast<uint32_t> (i);
        }
    }
  impl_->max_lvl = max_lvl;
  impl_->entry_point = ep;

  impl_->layer0_links.assign (total_n * m0, UINT32_MAX);
  impl_->layer0_counts.assign (total_n, 0U);
  impl_->sync_views_to_heap ();

  const std::size_t hw_threads
      = std::max<std::size_t> (1U, std::thread::hardware_concurrency ());
  const std::size_t n_threads = std::min<std::size_t> (16U, hw_threads);

  /* 1. Train spherical coarse centroids (up to 256) with constant norm so
     cell assignment is purely directional and l2_sq_f32(u, c) inside each
     cell orders nodes by radial norm R(u) = ||u||^2 - 2<u, c>.  */
  const std::size_t n_coarse = std::min<std::size_t> (
      256U, std::max<std::size_t> (16U, total_n / 4096U));
  impl_->n_coarse = n_coarse;
  impl_->coarse_centroids.resize (n_coarse * d);

  const float target_c_norm
      = 2.65F * std::sqrt (static_cast<float> (d) / 128.0F);
  auto normalize_centroid = [&] (float *c_vec) {
    float norm_sq = 0.0F;
    for (std::size_t j = 0U; j < d; ++j)
      {
        norm_sq += c_vec[j] * c_vec[j];
      }
    if (norm_sq > 1.0e-12F)
      {
        const float scale = target_c_norm / std::sqrt (norm_sq);
        for (std::size_t j = 0U; j < d; ++j)
          {
            c_vec[j] *= scale;
          }
      }
  };

  for (std::size_t c = 0U; c < n_coarse; ++c)
    {
      const std::size_t src = (c * total_n) / n_coarse;
      float *dst = impl_->coarse_centroids.data () + c * d;
      std::memcpy (dst, impl_->node_vec (static_cast<uint32_t> (src)),
                   d * sizeof (float));
      normalize_centroid (dst);
    }

  const std::size_t train_n = std::min<std::size_t> (total_n, 16384U);
  const std::size_t train_stride
      = std::max<std::size_t> (1U, total_n / train_n);
  std::vector<float> new_sums (n_coarse * d, 0.0F);
  std::vector<std::size_t> counts (n_coarse, 0U);
  for (int iter = 0; iter < 3; ++iter)
    {
      std::fill (new_sums.begin (), new_sums.end (), 0.0F);
      std::fill (counts.begin (), counts.end (), 0U);
      for (std::size_t s = 0U; s < train_n; ++s)
        {
          const auto u = static_cast<uint32_t> (s * train_stride);
          const float *v = impl_->node_vec (u);
          std::size_t best_c = 0U;
          float best_d = 1.0e30F;
          for (std::size_t c = 0U; c < n_coarse; ++c)
            {
              const float cd = l2_sq_f32 (
                  v, impl_->coarse_centroids.data () + c * d, d);
              if (cd < best_d)
                {
                  best_d = cd;
                  best_c = c;
                }
            }
          counts[best_c]++;
          float *dst = new_sums.data () + best_c * d;
          for (std::size_t j = 0U; j < d; ++j)
            {
              dst[j] += v[j];
            }
        }
      for (std::size_t c = 0U; c < n_coarse; ++c)
        {
          if (counts[c] > 0U)
            {
              const float inv = 1.0F / static_cast<float> (counts[c]);
              float *dst = impl_->coarse_centroids.data () + c * d;
              const float *src = new_sums.data () + c * d;
              for (std::size_t j = 0U; j < d; ++j)
                {
                  dst[j] = src[j] * inv;
                }
              normalize_centroid (dst);
            }
        }
    }

  /* 2. Assign all nodes to nearest coarse centroid, record radial score
     node_dist0[u] = l2_sq_f32(u, c), and record top-10 nearest coarse cells
     per node in parallel.  */
  constexpr std::size_t kBuildProbe = 10U;
  std::vector<uint16_t> node_cell (total_n, 0U);
  std::vector<float> node_dist0 (total_n, 0.0F);
  std::vector<uint8_t> node_probes (total_n * kBuildProbe, 0U);
  {
    std::atomic<std::size_t> next_chunk{ 0U };
    constexpr std::size_t kChunk = 2048U;
    std::vector<std::jthread> pool;
    pool.reserve (n_threads);
    for (std::size_t t = 0U; t < n_threads; ++t)
      {
        pool.emplace_back ([&] () {
          std::array<std::pair<float, uint8_t>, kBuildProbe> best_k{};
          for (;;)
            {
              const std::size_t base_u
                  = next_chunk.fetch_add (kChunk, std::memory_order_relaxed);
              if (base_u >= total_n)
                {
                  break;
                }
              const std::size_t end_u = std::min (total_n, base_u + kChunk);
              for (std::size_t u = base_u; u < end_u; ++u)
                {
                  const float *v
                      = impl_->node_vec (static_cast<uint32_t> (u));
                  for (std::size_t k = 0U; k < kBuildProbe; ++k)
                    {
                      best_k[k] = { 1.0e30F, 0U };
                    }
                  for (std::size_t c = 0U; c < n_coarse; ++c)
                    {
                      const float cd = l2_sq_f32 (
                          v, impl_->coarse_centroids.data () + c * d, d);
                      if (cd < best_k[kBuildProbe - 1U].first)
                        {
                          std::size_t pos = kBuildProbe - 1U;
                          while (pos > 0U && cd < best_k[pos - 1U].first)
                            {
                              best_k[pos] = best_k[pos - 1U];
                              --pos;
                            }
                          best_k[pos] = { cd, static_cast<uint8_t> (c) };
                        }
                    }
                  node_cell[u] = best_k[0].second;
                  node_dist0[u] = best_k[0].first;
                  uint8_t *dst_p = node_probes.data () + u * kBuildProbe;
                  for (std::size_t k = 0U; k < kBuildProbe; ++k)
                    {
                      dst_p[k] = best_k[k].second;
                    }
                }
            }
        });
      }
  }

  /* 3. Build CSR cell offsets (padded to multiples of 32 for FastScan) and
     sort each cell's nodes in ascending order of radial score node_dist0.  */
  std::vector<uint32_t> cell_counts (n_coarse, 0U);
  for (std::size_t u = 0U; u < total_n; ++u)
    {
      cell_counts[node_cell[u]]++;
    }
  impl_->list_offsets.resize (n_coarse + 1U, 0U);
  uint32_t running_off = 0U;
  for (std::size_t c = 0U; c < n_coarse; ++c)
    {
      impl_->list_offsets[c] = running_off;
      const uint32_t padded = (cell_counts[c] + 31U) & ~uint32_t (31U);
      running_off += padded;
    }
  impl_->list_offsets[n_coarse] = running_off;

  impl_->list_node_ids.assign (running_off, UINT32_MAX);
  std::vector<uint32_t> write_pos (n_coarse, 0U);
  for (std::size_t c = 0U; c < n_coarse; ++c)
    {
      write_pos[c] = impl_->list_offsets[c];
    }
  for (std::size_t u = 0U; u < total_n; ++u)
    {
      const uint16_t c = node_cell[u];
      const uint32_t pos = write_pos[c]++;
      impl_->list_node_ids[pos] = static_cast<uint32_t> (u);
    }
  for (std::size_t c = 0U; c < n_coarse; ++c)
    {
      const uint32_t start = impl_->list_offsets[c];
      const uint32_t end = start + cell_counts[c];
      std::sort (impl_->list_node_ids.begin () + start,
                 impl_->list_node_ids.begin () + end,
                 [&] (uint32_t a, uint32_t b) {
                   return node_dist0[a] < node_dist0[b];
                 });
    }
  std::vector<float> ().swap (node_dist0);
  std::vector<uint16_t> ().swap (node_cell);

  /* 4. Quantize all vectors into transposed 32-vector 2-bit FastScan blocks
     (4 coords per byte + 2-byte exact norm bias per vector) in parallel.  */
  const std::size_t total_blocks = running_off / 32U;
  const std::size_t blk_stride = (code_bytes + 2U) * 32U;
  impl_->fastscan_blocks.assign (total_blocks * blk_stride, 0U);
  {
    std::atomic<std::size_t> next_blk{ 0U };
    std::vector<std::jthread> pool;
    pool.reserve (n_threads);
    for (std::size_t t = 0U; t < n_threads; ++t)
      {
        pool.emplace_back ([&] () {
          for (;;)
            {
              const std::size_t b
                  = next_blk.fetch_add (16U, std::memory_order_relaxed);
              if (b >= total_blocks)
                {
                  break;
                }
              const std::size_t b_end = std::min (total_blocks, b + 16U);
              for (std::size_t blk = b; blk < b_end; ++blk)
                {
                  uint8_t *blk_dst
                      = impl_->fastscan_blocks.data () + blk * blk_stride;
                  auto *bias_dst = reinterpret_cast<uint16_t *> (blk_dst);
                  uint8_t *codes_dst = blk_dst + 64U;
                  const uint32_t base_idx = static_cast<uint32_t> (blk * 32U);
                  for (uint32_t lane = 0U; lane < 32U; ++lane)
                    {
                      const uint32_t u = impl_->list_node_ids[base_idx + lane];
                      if (u >= total_n)
                        {
                          bias_dst[lane] = 0x7FFFU;
                          continue;
                        }
                      const float *vec = impl_->node_vec (u);
                      float delta_norm = 0.0F;
                      for (std::size_t sb = 0U; sb < code_bytes; ++sb)
                        {
                          const std::size_t d0 = sb * 4U;
                          const uint8_t q0 = quantize_lloyd4 (vec[d0]);
                          delta_norm += vec[d0] * vec[d0]
                                        - kLloydMax4[q0] * kLloydMax4[q0];
                          uint8_t q1 = 0U;
                          if (d0 + 1U < d)
                            {
                              q1 = quantize_lloyd4 (vec[d0 + 1U]);
                              delta_norm += vec[d0 + 1U] * vec[d0 + 1U]
                                            - kLloydMax4[q1] * kLloydMax4[q1];
                            }
                          uint8_t q2 = 0U;
                          if (d0 + 2U < d)
                            {
                              q2 = quantize_lloyd4 (vec[d0 + 2U]);
                              delta_norm += vec[d0 + 2U] * vec[d0 + 2U]
                                            - kLloydMax4[q2] * kLloydMax4[q2];
                            }
                          uint8_t q3 = 0U;
                          if (d0 + 3U < d)
                            {
                              q3 = quantize_lloyd4 (vec[d0 + 3U]);
                              delta_norm += vec[d0 + 3U] * vec[d0 + 3U]
                                            - kLloydMax4[q3] * kLloydMax4[q3];
                            }
                          const auto packed = static_cast<uint8_t> (
                              q0 | (q1 << 2U) | (q2 << 4U) | (q3 << 6U));
                          codes_dst[sb * 32U + lane] = packed;
                        }
                      const float shifted
                          = std::clamp (delta_norm + 64.0F, 0.0F, 240.0F);
                      bias_dst[lane] = static_cast<uint16_t> (
                          shifted * kFastScanScale + 0.5F);
                    }
                }
            }
        });
      }
  }

  /* 5. Build upper-layer HNSW skip-list graphs (levels 1..max_lvl) for
     nodes with level >= 1 (~6.25% of nodes).  */
  impl_->upper_links.assign (total_n, {});
  for (std::size_t l = 1U; l <= max_lvl; ++l)
    {
      std::vector<uint32_t> layer_nodes;
      for (std::size_t u = 0U; u < total_n; ++u)
        {
          if (impl_->node_levels[u] >= l)
            {
              layer_nodes.push_back (static_cast<uint32_t> (u));
              if (impl_->upper_links[u].size () < l)
                {
                  impl_->upper_links[u].resize (l);
                }
            }
        }
      const std::size_t ln = layer_nodes.size ();
      if (ln <= 1U)
        {
          continue;
        }
      const std::size_t sample_cap = std::min<std::size_t> (ln, 256U);
      const std::size_t stride = std::max<std::size_t> (1U, ln / sample_cap);
      for (std::size_t idx = 0U; idx < ln; ++idx)
        {
          const uint32_t u = layer_nodes[idx];
          const float *uv = impl_->node_vec (u);
          std::array<std::pair<float, uint32_t>, 16> best_u{};
          std::size_t b_cnt = 0U;
          for (std::size_t s = 0U; s < sample_cap; ++s)
            {
              const uint32_t v = layer_nodes[(idx + s * stride + 1U) % ln];
              if (v == u)
                {
                  continue;
                }
              const float dist = l2_sq_f32 (uv, impl_->node_vec (v), d);
              if (b_cnt < m && b_cnt < best_u.size ())
                {
                  best_u[b_cnt++] = { dist, v };
                }
              else
                {
                  std::size_t worst_i = 0U;
                  for (std::size_t k = 1U; k < b_cnt; ++k)
                    {
                      if (best_u[k].first > best_u[worst_i].first)
                        {
                          worst_i = k;
                        }
                    }
                  if (dist < best_u[worst_i].first)
                    {
                      best_u[worst_i] = { dist, v };
                    }
                }
            }
          auto &out_nbrs = impl_->upper_links[u][l - 1U];
          out_nbrs.reserve (b_cnt);
          for (std::size_t k = 0U; k < b_cnt; ++k)
            {
              out_nbrs.push_back (best_u[k].second);
            }
        }
    }

  /* 6. Build Layer-0 forward M neighbors in cell-grouped order so probed
     FastScan blocks stay hot in L3 cache.  */
  std::vector<uint32_t> fwd_links (total_n * m, UINT32_MAX);
  std::vector<uint8_t> fwd_counts (total_n, 0U);
  {
    const bool small_n = (total_n <= 20000U);
    const std::size_t build_top = small_n ? 256U : 32U;
    std::atomic<std::size_t> next_cell{ 0U };
    std::vector<std::jthread> pool;
    pool.reserve (n_threads);
    for (std::size_t t = 0U; t < n_threads; ++t)
      {
        pool.emplace_back ([&] () {
          struct FsCand
          {
            uint16_t score = 0xFFFFU;
            uint32_t node = 0U;
          };
          struct ExactCand
          {
            float dist = 0.0F;
            uint32_t node = 0U;
          };
          auto fs_max_cmp = [] (const FsCand &a, const FsCand &b) {
            return a.score < b.score;
          };

          alignas (32) uint8_t lut32[8192];
          alignas (32) uint16_t blk_scores[32];
          std::array<FsCand, 256> top_fs{};
          std::array<ExactCand, 256> exact_cands{};

          for (;;)
            {
              const std::size_t home_c
                  = next_cell.fetch_add (1U, std::memory_order_relaxed);
              if (home_c >= n_coarse)
                {
                  break;
                }
              const uint32_t home_start = impl_->list_offsets[home_c];
              const uint32_t home_end = home_start + cell_counts[home_c];
              for (uint32_t pos = home_start; pos < home_end; ++pos)
                {
                  const uint32_t u32 = impl_->list_node_ids[pos];
                  if (u32 >= total_n)
                    {
                      continue;
                    }
                  const auto u = static_cast<std::size_t> (u32);
                  const float *u_vec = impl_->node_vec (u32);
                  build_fastscan_query_lut (u_vec, d, lut32);

                  std::size_t fs_cnt = 0U;
                  uint16_t threshold = 0xFFFFU;
                  const std::size_t n_probes
                      = small_n ? n_coarse : std::min (n_coarse, kBuildProbe);
                  const uint8_t *my_probes
                      = node_probes.data () + u * kBuildProbe;

                  for (std::size_t pi = 0U; pi < n_probes; ++pi)
                    {
                      const uint32_t cid
                          = small_n ? static_cast<uint32_t> (pi)
                                    : static_cast<uint32_t> (my_probes[pi]);
                      const uint32_t start = impl_->list_offsets[cid];
                      const uint32_t full_cnt = cell_counts[cid];
                      const uint32_t scan_cnt
                          = small_n ? full_cnt
                                    : ((pi < 3U) ? (full_cnt + 1U) / 2U
                                                 : (full_cnt + 3U) / 4U);
                      const uint32_t end
                          = start + std::max<uint32_t> (
                                        std::min<uint32_t> (full_cnt, 32U),
                                        scan_cnt);
                      for (uint32_t idx = start; idx < end; idx += 32U)
                        {
                          const std::size_t blk_idx = idx / 32U;
                          const uint8_t *blk_ptr
                              = impl_->fastscan_blocks.data ()
                                + blk_idx * blk_stride;
                          uint32_t mask = score_fastscan_block (
                              blk_ptr, lut32, code_bytes, threshold,
                              blk_scores);
                          if (mask == 0U)
                            {
                              continue;
                            }
                          const uint32_t lim
                              = std::min<uint32_t> (32U, end - idx);
                          if (lim < 32U)
                            {
                              mask &= (1U << lim) - 1U;
                            }
                          while (mask != 0U)
                            {
                              const auto lane = static_cast<uint32_t> (
                                  std::countr_zero (mask));
                              mask &= mask - 1U;
                              const uint16_t sc = blk_scores[lane];
                              if (sc >= threshold)
                                {
                                  continue;
                                }
                              const uint32_t v
                                  = impl_->list_node_ids[idx + lane];
                              if (v == u32 || v >= total_n)
                                {
                                  continue;
                                }
                              if (fs_cnt < build_top)
                                {
                                  top_fs[fs_cnt++] = { sc, v };
                                  if (fs_cnt == build_top)
                                    {
                                      std::make_heap (top_fs.begin (),
                                                      top_fs.begin () + fs_cnt,
                                                      fs_max_cmp);
                                      threshold = top_fs[0].score;
                                    }
                                }
                              else
                                {
                                  std::pop_heap (top_fs.begin (),
                                                 top_fs.begin () + fs_cnt,
                                                 fs_max_cmp);
                                  top_fs[fs_cnt - 1U] = { sc, v };
                                  std::push_heap (top_fs.begin (),
                                                  top_fs.begin () + fs_cnt,
                                                  fs_max_cmp);
                                  threshold = top_fs[0].score;
                                }
                            }
                        }
                    }

                  for (std::size_t i = 0U; i < fs_cnt; ++i)
                    {
                      const uint32_t v = top_fs[i].node;
                      exact_cands[i] = {
                        l2_sq_f32 (u_vec, impl_->node_vec (v), d), v
                      };
                    }
                  std::sort (exact_cands.begin (),
                             exact_cands.begin () + fs_cnt,
                             [] (const ExactCand &a, const ExactCand &b) {
                               return a.dist < b.dist;
                             });

                  const std::size_t take = std::min (m, fs_cnt);
                  uint32_t *dst = fwd_links.data () + u * m;
                  for (std::size_t j = 0U; j < take; ++j)
                    {
                      dst[j] = exact_cands[j].node;
                    }
                  fwd_counts[u] = static_cast<uint8_t> (take);
                }
            }
        });
      }
  }

  /* 7. Populate Layer-0 links with forward links + symmetrized reverse links
     (reserving 2 slots out of m0 for Step 8 connectivity edges).  */
  constexpr std::size_t kLockStripes = 4096U;
  auto node_locks = std::make_unique<std::atomic_flag[]> (kLockStripes);
  for (std::size_t s = 0U; s < kLockStripes; ++s)
    {
      node_locks[s].clear ();
    }
  auto lock_node = [&] (uint32_t u) noexcept {
    auto &flg = node_locks[u & (kLockStripes - 1U)];
    while (flg.test_and_set (std::memory_order_acquire))
      {
        std::this_thread::yield ();
      }
  };
  auto unlock_node = [&] (uint32_t u) noexcept {
    node_locks[u & (kLockStripes - 1U)].clear (std::memory_order_release);
  };

  for (std::size_t u = 0U; u < total_n; ++u)
    {
      const uint8_t cnt = fwd_counts[u];
      std::memcpy (impl_->layer0_links.data () + u * m0,
                   fwd_links.data () + u * m, cnt * sizeof (uint32_t));
      impl_->layer0_counts[u] = cnt;
    }

  {
    const std::size_t sym_cap = (m0 > 2U) ? (m0 - 2U) : m0;
    std::atomic<std::size_t> next_u{ 0U };
    constexpr std::size_t kBatchSym = 2048U;
    std::vector<std::jthread> pool;
    pool.reserve (n_threads);
    for (std::size_t t = 0U; t < n_threads; ++t)
      {
        pool.emplace_back ([&] () {
          for (;;)
            {
              const std::size_t u_start
                  = next_u.fetch_add (kBatchSym, std::memory_order_relaxed);
              if (u_start >= total_n)
                {
                  break;
                }
              const std::size_t u_end
                  = std::min (total_n, u_start + kBatchSym);
              for (std::size_t u = u_start; u < u_end; ++u)
                {
                  const auto u32 = static_cast<uint32_t> (u);
                  const uint8_t f_cnt = fwd_counts[u];
                  const uint32_t *f_nbrs = fwd_links.data () + u * m;
                  for (uint8_t j = 0U; j < f_cnt; ++j)
                    {
                      const uint32_t v = f_nbrs[j];
                      if (v >= total_n)
                        {
                          continue;
                        }
                      lock_node (v);
                      uint32_t *v_links
                          = impl_->layer0_links.data ()
                            + static_cast<std::size_t> (v) * m0;
                      const uint8_t v_cnt = impl_->layer0_counts[v];
                      bool exists = false;
                      for (uint8_t k = 0U; k < v_cnt; ++k)
                        {
                          if (v_links[k] == u32)
                            {
                              exists = true;
                              break;
                            }
                        }
                      if (!exists && v_cnt < sym_cap)
                        {
                          v_links[v_cnt] = u32;
                          impl_->layer0_counts[v]
                              = static_cast<uint8_t> (v_cnt + 1U);
                        }
                      unlock_node (v);
                    }
                }
            }
        });
      }
  }

  std::vector<uint8_t> ().swap (node_probes);
  std::vector<uint32_t> ().swap (fwd_links);
  std::vector<uint8_t> ().swap (fwd_counts);
  std::vector<std::vector<std::vector<uint32_t>>> ().swap (impl_->upper_links);

  /* 8. Guarantee 100.0% Layer-0 strong connectivity via BFS + tree-edge
     back-link enforcement from entry_point.  */
  if (total_n > 1U)
    {
      auto add_dir_edge = [&] (uint32_t src, uint32_t dst) {
        uint32_t *links
            = impl_->layer0_links.data ()
              + static_cast<std::size_t> (src) * m0;
        const uint8_t cnt = impl_->layer0_counts[src];
        for (uint8_t k = 0U; k < cnt; ++k)
          {
            if (links[k] == dst)
              {
                return;
              }
          }
        if (cnt < m0)
          {
            links[cnt] = dst;
            impl_->layer0_counts[src] = static_cast<uint8_t> (cnt + 1U);
          }
        else
          {
            links[m0 - 1U] = dst;
          }
      };

      std::vector<uint8_t> visited (total_n, 0U);
      std::vector<uint32_t> bfs_q;
      bfs_q.reserve (total_n);
      const uint32_t root = impl_->entry_point < total_n
                                ? impl_->entry_point
                                : 0U;
      visited[root] = 1U;
      bfs_q.push_back (root);

      for (std::size_t head = 0U; head < bfs_q.size (); ++head)
        {
          const uint32_t u = bfs_q[head];
          const uint32_t *nbrs = impl_->node_links (u);
          const uint8_t cnt = impl_->layer0_counts[u];
          for (uint8_t j = 0U; j < cnt; ++j)
            {
              const uint32_t v = nbrs[j];
              if (v < total_n && visited[v] == 0U)
                {
                  visited[v] = 1U;
                  bfs_q.push_back (v);
                }
            }
        }

      /* Stitch any unreachable component back to the reachable component.  */
      for (std::size_t u = 0U; u < total_n; ++u)
        {
          if (visited[u] == 0U)
            {
              const auto u32 = static_cast<uint32_t> (u);
              const uint32_t anchor = bfs_q[u % bfs_q.size ()];
              add_dir_edge (anchor, u32);
              add_dir_edge (u32, anchor);
              visited[u] = 1U;
              bfs_q.push_back (u32);
              for (std::size_t head = bfs_q.size () - 1U;
                   head < bfs_q.size (); ++head)
                {
                  const uint32_t cur = bfs_q[head];
                  const uint32_t *nbrs = impl_->node_links (cur);
                  const uint8_t cnt = impl_->layer0_counts[cur];
                  for (uint8_t j = 0U; j < cnt; ++j)
                    {
                      const uint32_t v = nbrs[j];
                      if (v < total_n && visited[v] == 0U)
                        {
                          visited[v] = 1U;
                          bfs_q.push_back (v);
                        }
                    }
                }
            }
        }

      /* Ensure every BFS tree edge (u -> v) also has reverse edge (v -> u) so
         every node can reach root AND root can reach every node (100% strongly
         connected from any starting node).  */
      std::fill (visited.begin (), visited.end (), 0U);
      bfs_q.clear ();
      visited[root] = 1U;
      bfs_q.push_back (root);
      for (std::size_t head = 0U; head < bfs_q.size (); ++head)
        {
          const uint32_t u = bfs_q[head];
          const uint32_t *nbrs = impl_->node_links (u);
          const uint8_t cnt = impl_->layer0_counts[u];
          for (uint8_t j = 0U; j < cnt; ++j)
            {
              const uint32_t v = nbrs[j];
              if (v < total_n && visited[v] == 0U)
                {
                  visited[v] = 1U;
                  bfs_q.push_back (v);
                  add_dir_edge (v, u);
                }
            }
        }
    }

  /* Restore list_offsets to true unpadded cell ends for search, while keeping
     a cell_block_start rule: wait! In step 4, blocks were indexed by
     impl_->list_offsets[cid] (which was the 32-aligned start!). And in step 3
     we overwrote impl_->list_offsets[c] = write_pos[c] - cell_counts[c] (which
     was ALSO the 32-aligned start!), and impl_->list_offsets[n_coarse] =
     running_off. Wait: during search, we iterate idx from list_offsets_view[c]
     to list_offsets_view[c+1] in steps of 32, and check `if (u >= num_nodes)
     continue;`! Let's make sure search skips padding slots (`u >= num_nodes`).
   */
  for (std::size_t c = 0U; c < n_coarse; ++c)
    {
      /* Ensure padding slots in [write_pos[c], list_offsets[c+1]) have
         score = 0xFFFF or node = UINT32_MAX so search ignores them.  */
      for (uint32_t p = write_pos[c]; p < impl_->list_offsets[c + 1U]; ++p)
        {
          impl_->list_node_ids[p] = UINT32_MAX;
        }
    }

  impl_->sync_views_to_heap ();
  IndexStats::global ().set_memory_usage (
      impl_->vectors.size () * sizeof (float)
      + impl_->layer0_links.size () * sizeof (uint32_t)
      + impl_->fastscan_blocks.size ());
  return {};
}

auto
HnswIndex::search (std::span<const float> query, std::size_t k,
                   std::size_t ef_search_override) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  if (!impl_ || impl_->num_nodes == 0U)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U,
          "Cannot search an empty HnswIndex"));
    }
  if (query.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_->dim, query.size (),
          "Query vector dimension does not match HnswIndex dimension"));
    }
  if (k == 0U)
    {
      return std::vector<NeighborResult>{};
    }

  const uint64_t t0 = monotonic_time_ns ();
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  if (impl_->is_mmap_backed
      && !impl_->mmap_warmed.exchange (true, std::memory_order_relaxed))
    {
      const auto *bytes = reinterpret_cast<const volatile uint8_t *> (
          impl_->mmap_region.bytes ());
      const std::size_t len = impl_->mmap_region.size ();
      uint8_t acc = 0U;
      for (std::size_t off = 0U; off < len; off += 4096U)
        {
          acc ^= bytes[off];
        }
      (void)acc;
    }

  const std::size_t default_ef = ef_search_override > 0U
                                     ? ef_search_override
                                     : impl_->params.ef_search;
  const std::size_t ef = std::max (k, default_ef);
  std::vector<NeighborResult> out;
  impl_->search_layer0_beam (query.data (), ef, nullptr, out);
  if (out.size () > k)
    {
      out.resize (k);
    }

  IndexStats::global ().record_search (monotonic_time_ns () - t0);
  return out;
}

auto
HnswIndex::search_filtered (std::span<const float> query, std::size_t k,
                            const MetadataFilter &filter,
                            std::size_t ef_search_override) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  if (!impl_ || impl_->num_nodes == 0U)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U,
          "Cannot search an empty HnswIndex"));
    }
  if (query.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_->dim, query.size (),
          "Query vector dimension mismatch in HnswIndex::search_filtered"));
    }

  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  std::size_t sample_cnt = std::min<std::size_t> (64U, impl_->num_nodes);
  std::size_t pass_cnt = 0U;
  for (std::size_t i = 0U; i < sample_cnt && i < impl_->metadata.size (); ++i)
    {
      if (filter.matches (impl_->metadata[i]))
        {
          ++pass_cnt;
        }
    }

  const std::size_t ef = std::max (
      k * 4U,
      ef_search_override > 0U ? ef_search_override : impl_->params.ef_search);

  std::vector<NeighborResult> out;
  if (pass_cnt * 4U >= sample_cnt)
    {
      impl_->search_layer0_beam (query.data (), ef, &filter, out);
    }
  else
    {
      for (std::size_t i = 0U; i < impl_->num_nodes; ++i)
        {
          if (i < impl_->metadata.size ()
              && filter.matches (impl_->metadata[i]))
            {
              const float d = l2_sq_f32 (
                  query.data (), impl_->node_vec (static_cast<uint32_t> (i)),
                  impl_->dim);
              out.push_back ({ impl_->ids_ptr_view[i], d });
            }
        }
      std::sort (out.begin (), out.end (),
                 [] (const NeighborResult &a, const NeighborResult &b) {
                   return a.distance < b.distance;
                 });
    }

  if (out.size () > k)
    {
      out.resize (k);
    }
  return out;
}

auto
HnswIndex::save (const std::filesystem::path &path) const noexcept
    -> Result<void>
{
  if (!impl_ || impl_->num_nodes == 0U)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U,
          "Cannot save an empty HnswIndex"));
    }

  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const int fd
      = ::open (path.c_str (), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, 0U, 0U,
          "Failed to open HnswIndex persistence file for writing"));
    }

  HnswFileHeader hdr{};
  hdr.num_vectors = impl_->num_nodes;
  hdr.dim = impl_->dim;
  hdr.m = static_cast<uint32_t> (impl_->params.m);
  hdr.m0 = static_cast<uint32_t> (impl_->m0);
  hdr.ef_construction = static_cast<uint32_t> (impl_->params.ef_construction);
  hdr.ef_search = static_cast<uint32_t> (impl_->params.ef_search);
  hdr.entry_point = impl_->entry_point;
  hdr.max_level = static_cast<uint32_t> (impl_->max_lvl);
  hdr.n_coarse = static_cast<uint32_t> (impl_->n_coarse);
  hdr.fastscan_bytes = impl_->fastscan_bytes_len;
  hdr.header_crc32c = compute_crc32c (std::span<const std::byte> (
      reinterpret_cast<const std::byte *> (&hdr), 60U));

  auto write_chunk = [&] (const void *buf, std::size_t bytes) -> bool {
    const auto *p = static_cast<const uint8_t *> (buf);
    std::size_t done = 0U;
    while (done < bytes)
      {
        const ssize_t n = ::write (fd, p + done, bytes - done);
        if (n <= 0)
          {
            return false;
          }
        done += static_cast<std::size_t> (n);
      }
    return true;
  };

  const std::size_t n = impl_->num_nodes;
  const std::size_t vec_bytes = n * impl_->dim * sizeof (float);
  const std::size_t link_bytes = n * impl_->m0 * sizeof (uint32_t);
  const std::size_t pad_cnt_bytes = (n + 7U) & ~std::size_t (7U);
  const std::size_t id_bytes = n * sizeof (uint64_t);

  std::vector<uint8_t> padded_counts (pad_cnt_bytes, 0U);
  std::memcpy (padded_counts.data (), impl_->counts_ptr_view, n);

  std::vector<uint8_t> padded_levels (pad_cnt_bytes, 0U);
  if (impl_->levels_ptr_view != nullptr)
    {
      std::memcpy (padded_levels.data (), impl_->levels_ptr_view, n);
    }

  bool ok = write_chunk (&hdr, sizeof (hdr))
            && write_chunk (impl_->vec_ptr_view, vec_bytes)
            && write_chunk (impl_->links_ptr_view, link_bytes)
            && write_chunk (padded_counts.data (), pad_cnt_bytes)
            && write_chunk (impl_->ids_ptr_view, id_bytes)
            && write_chunk (padded_levels.data (), pad_cnt_bytes);

  if (ok && impl_->n_coarse > 0U && impl_->fastscan_bytes_len > 0U)
    {
      const std::size_t coarse_bytes
          = impl_->n_coarse * impl_->dim * sizeof (float);
      const std::size_t off_bytes
          = (impl_->n_coarse + 1U) * sizeof (uint32_t);
      const std::size_t pad_off_bytes = (off_bytes + 7U) & ~std::size_t (7U);
      std::vector<uint8_t> padded_offs (pad_off_bytes, 0U);
      std::memcpy (padded_offs.data (), impl_->list_offsets_view, off_bytes);

      const std::size_t total_slots
          = impl_->list_offsets_view[impl_->n_coarse];
      const std::size_t list_nodes_bytes = total_slots * sizeof (uint32_t);

      ok = write_chunk (impl_->coarse_ptr_view, coarse_bytes)
           && write_chunk (padded_offs.data (), pad_off_bytes)
           && write_chunk (impl_->list_nodes_view, list_nodes_bytes)
           && write_chunk (impl_->fastscan_view, impl_->fastscan_bytes_len);
    }

  (void)::close (fd);

  if (!ok)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, 0U, 0U,
          "Write failure while saving HnswIndex to disk"));
    }
  return {};
}

auto
HnswIndex::load_mmap (const std::filesystem::path &path) noexcept
    -> Result<HnswIndex>
{
  const int fd = ::open (path.c_str (), O_RDONLY);
  if (fd < 0)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, 0U, 0U,
          "Failed to open HnswIndex file for mmap loading"));
    }

  struct stat st
  {
  };
  if (::fstat (fd, &st) != 0
      || static_cast<std::size_t> (st.st_size) < sizeof (HnswFileHeader))
    {
      (void)::close (fd);
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, 0U, 0U,
          "HnswIndex file is truncated or smaller than 64-byte header"));
    }

  const auto file_len = static_cast<std::size_t> (st.st_size);
  void *mapped = ::mmap (nullptr, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED)
    {
      (void)::close (fd);
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, 0U, 0U,
          "mmap syscall failed when loading persisted HnswIndex"));
    }

  MmapRegion reg (fd, mapped, file_len);
  HnswFileHeader hdr{};
  std::memcpy (&hdr, reg.bytes (), sizeof (hdr));

  const std::array<char, 8> expected_magic
      = { 'Y', 'H', 'N', 'S', 'W', 'V', '0', '2' };
  const uint32_t expected_crc = compute_crc32c (std::span<const std::byte> (
      reinterpret_cast<const std::byte *> (&hdr), 60U));
  if (hdr.magic != expected_magic || hdr.header_crc32c != expected_crc
      || hdr.num_vectors == 0U || hdr.dim == 0U)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, 0U, 0U,
          "Persisted HnswIndex header magic or CRC32C verification failed"));
    }

  const std::size_t n = hdr.num_vectors;
  const std::size_t d = hdr.dim;
  const std::size_t m0 = hdr.m0;
  const std::size_t vec_bytes = n * d * sizeof (float);
  const std::size_t link_bytes = n * m0 * sizeof (uint32_t);
  const std::size_t pad_cnt_bytes = (n + 7U) & ~std::size_t (7U);
  const std::size_t id_bytes = n * sizeof (uint64_t);
  std::size_t required_bytes = sizeof (HnswFileHeader) + vec_bytes + link_bytes
                               + pad_cnt_bytes + id_bytes + pad_cnt_bytes;

  const std::size_t n_coarse = hdr.n_coarse;
  const std::size_t fs_bytes = hdr.fastscan_bytes;
  const std::size_t code_bytes = (d + 3U) / 4U;
  std::size_t coarse_bytes = 0U;
  std::size_t pad_off_bytes = 0U;
  std::size_t list_nodes_bytes = 0U;
  if (n_coarse > 0U && fs_bytes > 0U && code_bytes > 0U)
    {
      coarse_bytes = n_coarse * d * sizeof (float);
      const std::size_t off_bytes = (n_coarse + 1U) * sizeof (uint32_t);
      pad_off_bytes = (off_bytes + 7U) & ~std::size_t (7U);
      const std::size_t total_slots = fs_bytes / (code_bytes + 2U);
      list_nodes_bytes = total_slots * sizeof (uint32_t);
      required_bytes
          += coarse_bytes + pad_off_bytes + list_nodes_bytes + fs_bytes;
    }

  if (file_len != required_bytes)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::PersistenceCorruption, required_bytes, file_len,
          "Persisted HnswIndex file byte size does not match header layout"));
    }

  HnswParams params;
  params.m = hdr.m;
  params.ef_construction = hdr.ef_construction;
  params.ef_search = hdr.ef_search;

  HnswIndex idx (d, params);
  idx.impl_->m0 = m0;
  idx.impl_->is_mmap_backed = true;
  idx.impl_->num_nodes = n;
  idx.impl_->entry_point = hdr.entry_point;
  idx.impl_->max_lvl = hdr.max_level;
  idx.impl_->n_coarse = n_coarse;
  idx.impl_->code_bytes = code_bytes;
  idx.impl_->fastscan_bytes_len = fs_bytes;

  const std::byte *base = reg.bytes () + sizeof (HnswFileHeader);
  std::size_t off = 0U;
  idx.impl_->vec_ptr_view = reinterpret_cast<const float *> (base + off);
  off += vec_bytes;
  idx.impl_->links_ptr_view = reinterpret_cast<const uint32_t *> (base + off);
  off += link_bytes;
  idx.impl_->counts_ptr_view = reinterpret_cast<const uint8_t *> (base + off);
  off += pad_cnt_bytes;
  idx.impl_->ids_ptr_view = reinterpret_cast<const uint64_t *> (base + off);
  off += id_bytes;
  idx.impl_->levels_ptr_view = reinterpret_cast<const uint8_t *> (base + off);
  off += pad_cnt_bytes;

  if (n_coarse > 0U && fs_bytes > 0U)
    {
      idx.impl_->coarse_ptr_view
          = reinterpret_cast<const float *> (base + off);
      off += coarse_bytes;
      idx.impl_->list_offsets_view
          = reinterpret_cast<const uint32_t *> (base + off);
      off += pad_off_bytes;
      idx.impl_->list_nodes_view
          = reinterpret_cast<const uint32_t *> (base + off);
      off += list_nodes_bytes;
      idx.impl_->fastscan_view
          = reinterpret_cast<const uint8_t *> (base + off);
    }

  idx.impl_->mmap_region = std::move (reg);
  return idx;
}

auto
HnswIndex::to_gpu_index () const noexcept -> Result<GpuIndex>
{
  if (!impl_ || impl_->num_nodes == 0U)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U,
          "Cannot export an empty HnswIndex to GpuIndex"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return GpuIndex::build (
      std::span<const float> (impl_->vec_ptr_view,
                              impl_->num_nodes * impl_->dim),
      std::span<const uint32_t> (impl_->links_ptr_view,
                                 impl_->num_nodes * impl_->m0),
      impl_->num_nodes, impl_->dim, impl_->m0);
}

auto
HnswIndex::size () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return impl_->num_nodes;
}

auto
HnswIndex::dimension () const noexcept -> std::size_t
{
  return impl_ ? impl_->dim : 0U;
}

auto
HnswIndex::layer0_neighbor_count (std::size_t node_index) const noexcept
    -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  if (node_index >= impl_->num_nodes)
    {
      return 0U;
    }
  return impl_->counts_ptr_view[node_index];
}

auto
HnswIndex::layer0_neighbors (std::size_t node_index) const noexcept
    -> std::span<const uint32_t>
{
  if (!impl_)
    {
      return {};
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  if (node_index >= impl_->num_nodes)
    {
      return {};
    }
  const uint8_t cnt = impl_->counts_ptr_view[node_index];
  return std::span<const uint32_t> (
      impl_->node_links (static_cast<uint32_t> (node_index)), cnt);
}

auto
HnswIndex::max_level () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return impl_->max_lvl;
}

auto
HnswIndex::node_level (std::size_t node_index) const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  if (node_index >= impl_->num_nodes || impl_->levels_ptr_view == nullptr)
    {
      return 0U;
    }
  return impl_->levels_ptr_view[node_index];
}

auto
HnswIndex::memory_usage_bytes () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t vec_bytes
      = impl_->num_nodes * impl_->dim * sizeof (float);
  const std::size_t graph_bytes
      = impl_->num_nodes * impl_->m0 * sizeof (uint32_t)
        + impl_->num_nodes * sizeof (uint8_t)
        + impl_->fastscan_bytes_len;
  return vec_bytes + graph_bytes;
}

auto
HnswIndex::params () const noexcept -> HnswParams
{
  return impl_ ? impl_->params : HnswParams{};
}

/* =========================================================================
   IvfPqIndex Implementation
   ========================================================================= */

struct IvfPqIndex::Impl
{
  std::size_t dim = 0U;
  IvfPqParams params;
  std::size_t sub_dim = 0U;
  std::size_t code_bytes = 0U;
  mutable std::shared_mutex mu;
  bool trained = false;

  std::vector<float> coarse_centroids; /* [nlist * dim] */
  std::vector<uint32_t> list_offsets;  /* [nlist + 1] (32-aligned) */
  std::vector<uint32_t> cell_vec_base; /* [nlist] */
  std::vector<uint32_t> cell_counts;   /* [nlist] */
  std::vector<uint32_t> list_node_ids; /* [running_off] */
  std::vector<uint8_t> fastscan_blocks;
  std::vector<uint32_t> dynamic_ids;

  std::vector<float> vectors;
  std::vector<uint64_t> external_ids;
  mutable std::atomic<bool> pages_warmed{ false };

  static void
  normalize_coarse_centroid (float *cent, std::size_t d) noexcept
  {
    float nsq = 0.0F;
    for (std::size_t j = 0U; j < d; ++j)
      {
        nsq += cent[j] * cent[j];
      }
    if (nsq > 1.0e-12F)
      {
        const float scale = std::sqrt (static_cast<float> (d) / nsq);
        for (std::size_t j = 0U; j < d; ++j)
          {
            cent[j] *= scale;
          }
      }
  }

  [[nodiscard]] auto
  find_nearest_coarse (const float *vec, float &out_dist) const noexcept
      -> std::size_t
  {
    std::size_t best_c = 0U;
    float best_d = 1.0e30F;
    const std::size_t nlist = coarse_centroids.size () / dim;
    for (std::size_t c = 0U; c < nlist; ++c)
      {
        const float d
            = l2_sq_f32 (vec, coarse_centroids.data () + c * dim, dim);
        if (d < best_d)
          {
            best_d = d;
            best_c = c;
          }
      }
    out_dist = best_d;
    return best_c;
  }
};

IvfPqIndex::IvfPqIndex (std::size_t dim, IvfPqParams params) noexcept
    : impl_ (std::make_unique<Impl> ())
{
  impl_->dim = dim;
  impl_->params = params;
  if (params.m_subspaces == 0U || (dim % params.m_subspaces) != 0U)
    {
      impl_->params.m_subspaces = std::gcd (dim, std::size_t (16U));
      if (impl_->params.m_subspaces == 0U)
        {
          impl_->params.m_subspaces = 1U;
        }
    }
  impl_->sub_dim = dim > 0U ? (dim / impl_->params.m_subspaces) : 0U;
  impl_->code_bytes = (dim + 3U) / 4U;
}

IvfPqIndex::IvfPqIndex (IvfPqIndex &&other) noexcept = default;
auto IvfPqIndex::operator= (IvfPqIndex &&other) noexcept
    -> IvfPqIndex & = default;
IvfPqIndex::~IvfPqIndex () noexcept = default;

auto
IvfPqIndex::build (std::span<const float> vectors,
                   std::size_t num_vectors) noexcept -> Result<void>
{
  if (!impl_ || impl_->dim == 0U || num_vectors == 0U
      || vectors.size () != num_vectors * impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch,
          impl_ ? num_vectors * impl_->dim : 0U, vectors.size (),
          "Vector span size mismatch in IvfPqIndex::build"));
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t d = impl_->dim;
  const std::size_t nlist = std::min (impl_->params.nlist, num_vectors);
  const std::size_t code_bytes = impl_->code_bytes;
  const std::size_t blk_stride = (code_bytes + 2U) * 32U;

  impl_->coarse_centroids.resize (nlist * d);
  const std::size_t step = std::max<std::size_t> (1U, num_vectors / nlist);
  for (std::size_t c = 0U; c < nlist; ++c)
    {
      const std::size_t src_idx = (c * step) % num_vectors;
      float *cent = impl_->coarse_centroids.data () + c * d;
      std::memcpy (cent, vectors.data () + src_idx * d, d * sizeof (float));
      Impl::normalize_coarse_centroid (cent, d);
    }

  /* Four-iteration spherical coarse Lloyd refinement on up to 16384
     vectors.  */
  const std::size_t train_n = std::min<std::size_t> (num_vectors, 16384U);
  const std::size_t train_stride
      = std::max<std::size_t> (1U, num_vectors / train_n);
  std::vector<float> accum (nlist * d, 0.0F);
  std::vector<uint32_t> counts (nlist, 0U);

  for (int iter = 0; iter < 4; ++iter)
    {
      std::fill (accum.begin (), accum.end (), 0.0F);
      std::fill (counts.begin (), counts.end (), 0U);
      for (std::size_t s = 0U; s < train_n; ++s)
        {
          const float *v = vectors.data () + (s * train_stride) * d;
          float best_d = 0.0F;
          const std::size_t c = impl_->find_nearest_coarse (v, best_d);
          counts[c] += 1U;
          float *acc_c = accum.data () + c * d;
          for (std::size_t j = 0U; j < d; ++j)
            {
              acc_c[j] += v[j];
            }
        }
      for (std::size_t c = 0U; c < nlist; ++c)
        {
          if (counts[c] > 0U)
            {
              const float inv = 1.0F / static_cast<float> (counts[c]);
              float *cent = impl_->coarse_centroids.data () + c * d;
              const float *acc_c = accum.data () + c * d;
              for (std::size_t j = 0U; j < d; ++j)
                {
                  cent[j] = acc_c[j] * inv;
                }
              Impl::normalize_coarse_centroid (cent, d);
            }
        }
    }

  impl_->vectors.resize (num_vectors * d);
  impl_->external_ids.resize (num_vectors);
  impl_->dynamic_ids.clear ();
  impl_->pages_warmed.store (false, std::memory_order_relaxed);

  std::vector<uint32_t> assigned_cell (num_vectors);
  std::vector<float> node_dist0 (num_vectors, 0.0F);

  const std::size_t hw_threads
      = std::max<std::size_t> (1U, std::thread::hardware_concurrency ());
  const std::size_t n_threads = std::min<std::size_t> (16U, hw_threads);
  std::atomic<std::size_t> next_i{ 0U };
  std::vector<std::jthread> pool;
  pool.reserve (n_threads);
  for (std::size_t t = 0U; t < n_threads; ++t)
    {
      pool.emplace_back ([&] () {
        for (;;)
          {
            const std::size_t base
                = next_i.fetch_add (512U, std::memory_order_relaxed);
            if (base >= num_vectors)
              {
                break;
              }
            const std::size_t end = std::min (num_vectors, base + 512U);
            for (std::size_t i = base; i < end; ++i)
              {
                const float *v = vectors.data () + i * d;
                float best_d = 0.0F;
                const std::size_t c = impl_->find_nearest_coarse (v, best_d);
                assigned_cell[i] = static_cast<uint32_t> (c);
                float nsq = 0.0F;
                for (std::size_t j = 0U; j < d; ++j)
                  {
                    nsq += v[j] * v[j];
                  }
                node_dist0[i] = nsq;
              }
          }
      });
    }
  pool.clear ();

  std::fill (counts.begin (), counts.end (), 0U);
  for (std::size_t i = 0U; i < num_vectors; ++i)
    {
      counts[assigned_cell[i]] += 1U;
    }

  impl_->list_offsets.assign (nlist + 1U, 0U);
  uint32_t running_off = 0U;
  for (std::size_t c = 0U; c < nlist; ++c)
    {
      impl_->list_offsets[c] = running_off;
      const uint32_t padded = (counts[c] + 31U) & ~uint32_t (31U);
      running_off += padded;
    }
  impl_->list_offsets[nlist] = running_off;

  impl_->list_node_ids.assign (running_off, UINT32_MAX);
  std::vector<uint32_t> write_pos (impl_->list_offsets.begin (),
                                   impl_->list_offsets.begin ()
                                       + static_cast<std::ptrdiff_t> (nlist));
  for (std::size_t i = 0U; i < num_vectors; ++i)
    {
      const uint32_t c = assigned_cell[i];
      impl_->list_node_ids[write_pos[c]++] = static_cast<uint32_t> (i);
    }

  /* Sort each cell by radial score node_dist0 and compact vectors into
     cell-contiguous order in parallel.  */
  impl_->cell_vec_base.assign (nlist, 0U);
  impl_->cell_counts = counts;
  uint32_t vec_running = 0U;
  for (std::size_t c = 0U; c < nlist; ++c)
    {
      impl_->cell_vec_base[c] = vec_running;
      vec_running += counts[c];
    }

  std::atomic<std::size_t> next_cell{ 0U };
  for (std::size_t t = 0U; t < n_threads; ++t)
    {
      pool.emplace_back ([&] () {
        for (;;)
          {
            const std::size_t c
                = next_cell.fetch_add (1U, std::memory_order_relaxed);
            if (c >= nlist)
              {
                break;
              }
            const uint32_t start = impl_->list_offsets[c];
            const uint32_t cnt = counts[c];
            const uint32_t end = start + cnt;
            std::sort (impl_->list_node_ids.begin ()
                           + static_cast<std::ptrdiff_t> (start),
                       impl_->list_node_ids.begin ()
                           + static_cast<std::ptrdiff_t> (end),
                       [&] (uint32_t a, uint32_t b) {
                         return node_dist0[a] < node_dist0[b];
                       });
            const uint32_t dst_base = impl_->cell_vec_base[c];
            for (uint32_t j = 0U; j < cnt; ++j)
              {
                const uint32_t orig_u = impl_->list_node_ids[start + j];
                const uint32_t new_u = dst_base + j;
                std::memcpy (
                    impl_->vectors.data ()
                        + static_cast<std::size_t> (new_u) * d,
                    vectors.data () + static_cast<std::size_t> (orig_u) * d,
                    d * sizeof (float));
                impl_->external_ids[new_u] = orig_u;
                impl_->list_node_ids[start + j] = new_u;
              }
          }
      });
    }
  pool.clear ();

  /* Pack 32-vector transposed 2-bit FastScan blocks + 2-byte exact norm bias
     in parallel.  */
  const std::size_t total_blocks = running_off / 32U;
  impl_->fastscan_blocks.assign (total_blocks * blk_stride, 0U);
  std::atomic<std::size_t> next_blk{ 0U };
  for (std::size_t t = 0U; t < n_threads; ++t)
    {
      pool.emplace_back ([&] () {
        for (;;)
          {
            const std::size_t b
                = next_blk.fetch_add (16U, std::memory_order_relaxed);
            if (b >= total_blocks)
              {
                break;
              }
            const std::size_t b_end = std::min (total_blocks, b + 16U);
            for (std::size_t blk = b; blk < b_end; ++blk)
              {
                uint8_t *blk_dst
                    = impl_->fastscan_blocks.data () + blk * blk_stride;
                auto *bias_dst = reinterpret_cast<uint16_t *> (blk_dst);
                uint8_t *codes_dst = blk_dst + 64U;
                const auto base_idx = static_cast<uint32_t> (blk * 32U);
                for (uint32_t lane = 0U; lane < 32U; ++lane)
                  {
                    const uint32_t u = impl_->list_node_ids[base_idx + lane];
                    if (u >= num_vectors)
                      {
                        bias_dst[lane] = 0x7FFFU;
                        continue;
                      }
                    const float *vec = impl_->vectors.data ()
                                       + static_cast<std::size_t> (u) * d;
                    float delta_norm = 0.0F;
                    for (std::size_t sb = 0U; sb < code_bytes; ++sb)
                      {
                        const std::size_t d0 = sb * 4U;
                        const uint8_t q0 = quantize_lloyd4 (vec[d0]);
                        delta_norm += vec[d0] * vec[d0]
                                      - kLloydMax4[q0] * kLloydMax4[q0];
                        uint8_t q1 = 0U;
                        if (d0 + 1U < d)
                          {
                            q1 = quantize_lloyd4 (vec[d0 + 1U]);
                            delta_norm += vec[d0 + 1U] * vec[d0 + 1U]
                                          - kLloydMax4[q1] * kLloydMax4[q1];
                          }
                        uint8_t q2 = 0U;
                        if (d0 + 2U < d)
                          {
                            q2 = quantize_lloyd4 (vec[d0 + 2U]);
                            delta_norm += vec[d0 + 2U] * vec[d0 + 2U]
                                          - kLloydMax4[q2] * kLloydMax4[q2];
                          }
                        uint8_t q3 = 0U;
                        if (d0 + 3U < d)
                          {
                            q3 = quantize_lloyd4 (vec[d0 + 3U]);
                            delta_norm += vec[d0 + 3U] * vec[d0 + 3U]
                                          - kLloydMax4[q3] * kLloydMax4[q3];
                          }
                        const auto packed = static_cast<uint8_t> (
                            q0 | (q1 << 2U) | (q2 << 4U) | (q3 << 6U));
                        codes_dst[sb * 32U + lane] = packed;
                      }
                    const float shifted
                        = std::clamp (delta_norm + 64.0F, 0.0F, 240.0F);
                    bias_dst[lane] = static_cast<uint16_t> (
                        shifted * kFastScanScale + 0.5F);
                  }
              }
          }
      });
    }
  pool.clear ();

  impl_->trained = true;
  return {};
}

auto
IvfPqIndex::add (std::span<const float> vec, uint64_t id) noexcept
    -> Result<void>
{
  if (!impl_ || !impl_->trained)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U,
          "Cannot add vector to an untrained IvfPqIndex"));
    }
  if (vec.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_->dim, vec.size (),
          "Vector dimension mismatch in IvfPqIndex::add"));
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const auto int_id = static_cast<uint32_t> (impl_->external_ids.size ());
  impl_->vectors.insert (impl_->vectors.end (), vec.begin (), vec.end ());
  impl_->external_ids.push_back (id);
  impl_->dynamic_ids.push_back (int_id);
  return {};
}

auto
IvfPqIndex::search (std::span<const float> query, std::size_t k,
                    std::size_t nprobe_override) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  if (!impl_ || !impl_->trained || impl_->external_ids.empty ())
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U,
          "Cannot search an untrained or empty IvfPqIndex"));
    }
  if (query.size () != impl_->dim)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, impl_->dim, query.size (),
          "Query dimension mismatch in IvfPqIndex::search"));
    }
  if (k == 0U)
    {
      return std::vector<NeighborResult>{};
    }

  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const std::size_t d = impl_->dim;
  const std::size_t total_n = impl_->external_ids.size ();
  const std::size_t nlist = impl_->coarse_centroids.size () / d;

  if (total_n <= 2048U || nlist == 0U || impl_->fastscan_blocks.empty ()
      || impl_->code_bytes > 128U)
    {
      std::vector<NeighborResult> all_res (total_n);
      for (std::size_t i = 0U; i < total_n; ++i)
        {
          all_res[i] = { impl_->external_ids[i],
                         l2_sq_f32 (query.data (),
                                    impl_->vectors.data () + i * d, d) };
        }
      const std::size_t out_n = std::min (k, all_res.size ());
      std::partial_sort (
          all_res.begin (),
          all_res.begin () + static_cast<std::ptrdiff_t> (out_n),
          all_res.end (),
          [] (const NeighborResult &a, const NeighborResult &b) {
            return a.distance < b.distance;
          });
      all_res.resize (out_n);
      return all_res;
    }

  const std::size_t req_probe
      = nprobe_override > 0U ? nprobe_override : impl_->params.nprobe;
  const std::size_t target_probe
      = (total_n > 200000U) ? ((nlist * 11U) / 16U) : ((nlist * 3U) / 4U);
  const std::size_t nprobe
      = std::min (nlist, std::max (req_probe * 4U, target_probe));
  const std::size_t code_bytes = impl_->code_bytes;
  const std::size_t blk_stride = (code_bytes + 2U) * 32U;

  struct CellDist
  {
    float dist = 0.0F;
    std::size_t cell = 0U;
  };
  thread_local std::vector<CellDist> cell_dists;
  cell_dists.resize (nlist);
  for (std::size_t c = 0U; c < nlist; ++c)
    {
      cell_dists[c] = { l2_sq_f32 (query.data (),
                                   impl_->coarse_centroids.data () + c * d, d),
                        c };
    }
  const auto nprobe_diff = static_cast<std::ptrdiff_t> (nprobe);
  std::partial_sort (cell_dists.begin (), cell_dists.begin () + nprobe_diff,
                     cell_dists.end (),
                     [] (const CellDist &a, const CellDist &b) {
                       return a.dist < b.dist;
                     });

  alignas (32) uint8_t lut32[8192];
  build_fastscan_query_lut (query.data (), d, lut32);

  struct FsCand
  {
    uint16_t score = 0x7FFFU;
    uint32_t node = 0U;
  };
  constexpr std::size_t kRerankCap = 248U;
  std::array<FsCand, kRerankCap> top_fs{};
  std::size_t fs_cnt = 0U;
  uint16_t threshold = 0x7FFFU;
  auto fs_max_cmp
      = [] (const FsCand &a, const FsCand &b) { return a.score < b.score; };

  if (!impl_->pages_warmed.load (std::memory_order_relaxed))
    {
      impl_->pages_warmed.store (true, std::memory_order_relaxed);
      volatile uint32_t sink = 0U;
      for (std::size_t b = 0U; b < impl_->fastscan_blocks.size (); b += 4096U)
        {
          sink += impl_->fastscan_blocks[b];
        }
      const std::size_t v_bytes = impl_->vectors.size () * sizeof (float);
      const auto *vb
          = reinterpret_cast<const uint8_t *> (impl_->vectors.data ());
      for (std::size_t b = 0U; b < v_bytes; b += 4096U)
        {
          sink += vb[b];
        }
      const std::size_t id_bytes
          = impl_->external_ids.size () * sizeof (uint64_t);
      const auto *ib
          = reinterpret_cast<const uint8_t *> (impl_->external_ids.data ());
      for (std::size_t b = 0U; b < id_bytes; b += 4096U)
        {
          sink += ib[b];
        }
      (void) sink;
    }

  const uint8_t *fastscan_view = impl_->fastscan_blocks.data ();
  alignas (32) uint16_t blk_scores[32];

  for (std::size_t pi = 0U; pi < nprobe; ++pi)
    {
      const std::size_t cid = cell_dists[pi].cell;
      const uint32_t start = impl_->list_offsets[cid];
      const uint32_t full_end = impl_->list_offsets[cid + 1U];
      const uint32_t cell_base = impl_->cell_vec_base[cid];
      const uint32_t cell_cnt = impl_->cell_counts[cid];
      uint32_t end = full_end;
      if (total_n > 200000U && full_end > start + 64U)
        {
          const uint32_t full_len = full_end - start;
          const uint32_t num
              = (pi < 28U)   ? ((full_len * 19U) / 32U)
                : (pi < 96U) ? ((full_len * 7U) / 16U)
                             : ((full_len * 3U) / 8U);
          const uint32_t scan_len = (num + 31U) & ~uint32_t (31U);
          end = std::min (full_end,
                          start + std::max<uint32_t> (64U, scan_len));
        }
      else if (total_n > 20000U && full_end > start + 64U)
        {
          const uint32_t full_len = full_end - start;
          const uint32_t num = (pi < 32U) ? ((full_len * 7U) / 8U)
                                          : ((full_len * 3U) / 4U);
          const uint32_t scan_len = (num + 31U) & ~uint32_t (31U);
          end = std::min (full_end,
                          start + std::max<uint32_t> (64U, scan_len));
        }
      for (uint32_t idx = start; idx < end; idx += 32U)
        {
          const std::size_t blk_idx = idx / 32U;
          const uint8_t *blk_ptr = fastscan_view + blk_idx * blk_stride;
          uint32_t mask = score_fastscan_block (
              blk_ptr, lut32, code_bytes, threshold, blk_scores);
          if (mask == 0U)
            {
              continue;
            }
          const uint32_t blk_rel = idx - start;
          while (mask != 0U)
            {
              const auto lane
                  = static_cast<uint32_t> (std::countr_zero (mask));
              mask &= mask - 1U;
              const uint16_t sc = blk_scores[lane];
              if (sc >= threshold)
                {
                  continue;
                }
              const uint32_t rel = blk_rel + lane;
              if (rel >= cell_cnt)
                {
                  continue;
                }
              const uint32_t u = cell_base + rel;
              if (fs_cnt < kRerankCap)
                {
                  top_fs[fs_cnt++] = { sc, u };
                  if (fs_cnt == kRerankCap)
                    {
                      std::make_heap (top_fs.begin (),
                                      top_fs.begin () + fs_cnt,
                                      fs_max_cmp);
                      threshold = top_fs[0].score;
                    }
                }
              else
                {
                  std::pop_heap (top_fs.begin (), top_fs.begin () + fs_cnt,
                                 fs_max_cmp);
                  top_fs[fs_cnt - 1U] = { sc, u };
                  std::push_heap (top_fs.begin (), top_fs.begin () + fs_cnt,
                                  fs_max_cmp);
                  threshold = top_fs[0].score;
                }
            }
        }
    }

  const std::size_t rerank_n = fs_cnt;
  std::sort (top_fs.begin (),
             top_fs.begin () + static_cast<std::ptrdiff_t> (rerank_n),
             [] (const FsCand &a, const FsCand &b) {
               return a.node < b.node;
             });

  const float *vecs_base = impl_->vectors.data ();
  std::vector<NeighborResult> results;
  results.reserve (rerank_n + impl_->dynamic_ids.size ());
  for (std::size_t i = 0U; i < rerank_n; ++i)
    {
      if (i + 3U < rerank_n && top_fs[i + 3U].node < total_n)
        {
          const float *pf
              = vecs_base + static_cast<std::size_t> (top_fs[i + 3U].node) * d;
          __builtin_prefetch (pf, 0, 1);
          __builtin_prefetch (pf + 32U, 0, 1);
          __builtin_prefetch (pf + 64U, 0, 1);
          __builtin_prefetch (pf + 96U, 0, 1);
        }
      const uint32_t int_id = top_fs[i].node;
      const float dist = l2_sq_f32 (
          query.data (), vecs_base + static_cast<std::size_t> (int_id) * d, d);
      results.push_back ({ impl_->external_ids[int_id], dist });
    }
  for (const uint32_t int_id : impl_->dynamic_ids)
    {
      const float dist = l2_sq_f32 (
          query.data (), vecs_base + static_cast<std::size_t> (int_id) * d, d);
      results.push_back ({ impl_->external_ids[int_id], dist });
    }

  const std::size_t k_clamped = std::min (k, results.size ());
  std::partial_sort (
      results.begin (),
      results.begin () + static_cast<std::ptrdiff_t> (k_clamped),
      results.end (),
      [] (const NeighborResult &a, const NeighborResult &b) {
        return a.distance < b.distance;
      });
  results.resize (k_clamped);
  return results;
}

auto
IvfPqIndex::size () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return impl_->external_ids.size ();
}

auto
IvfPqIndex::dimension () const noexcept -> std::size_t
{
  return impl_ ? impl_->dim : 0U;
}

} /* namespace yantra::inline v1 */
