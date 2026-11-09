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

#include <obsidian/version.h>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/memory.hpp>
#include <obsidian/simd.hpp>

#include <cmath>
#include <sstream>

#if OBSIDIAN_HAS_AVX2 || OBSIDIAN_HAS_SSE42
#  include <immintrin.h>
#endif

namespace obsidian
{

const char *
version_string (void) noexcept
{
  return OBSIDIAN_VERSION_STRING;
}

int
version_major (void) noexcept
{
  return OBSIDIAN_VERSION_MAJOR;
}

int
version_minor (void) noexcept
{
  return OBSIDIAN_VERSION_MINOR;
}

int
version_patch (void) noexcept
{
  return OBSIDIAN_VERSION_PATCH;
}

const char *
error_code_name (Error_Code code) noexcept
{
  switch (code)
    {
    case Error_Code::Ok:
      return "Ok";
    case Error_Code::Invalid_Argument:
      return "Invalid_Argument";
    case Error_Code::Out_Of_Range:
      return "Out_Of_Range";
    case Error_Code::Shape_Mismatch:
      return "Shape_Mismatch";
    case Error_Code::Dtype_Mismatch:
      return "Dtype_Mismatch";
    case Error_Code::Device_Mismatch:
      return "Device_Mismatch";
    case Error_Code::IO_Failure:
      return "IO_Failure";
    case Error_Code::Corrupted_Format:
      return "Corrupted_Format";
    case Error_Code::Invalid_UTF8:
      return "Invalid_UTF8";
    case Error_Code::Vocab_Not_Found:
      return "Vocab_Not_Found";
    case Error_Code::Not_Initialized:
      return "Not_Initialized";
    case Error_Code::Convergence_Failure:
      return "Convergence_Failure";
    case Error_Code::Backend_Failure:
      return "Backend_Failure";
    }
  return "Unknown";
}

std::string
Error::format (void) const
{
  std::ostringstream oss;
  oss << "obsidian: ";
  if (file_ != nullptr && file_[0] != '\0')
    {
      oss << file_ << ":" << line_ << ": ";
    }
  oss << "[" << error_code_name (code_) << "] " << message_;
  return oss.str ();
}

Arena &
thread_local_arena (void) noexcept
{
  thread_local Arena tls_arena;
  return tls_arena;
}

namespace simd
{

namespace
{

constexpr bool
is_ascii_space (unsigned char c) noexcept
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r'
         || c == '\v' || c == '\f';
}

constexpr bool
is_ascii_punct (unsigned char c) noexcept
{
  return (c >= 33 && c <= 47) || (c >= 58 && c <= 64)
         || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
}

#if OBSIDIAN_HAS_AVX2
OBSIDIAN_ALWAYS_INLINE uint32_t
avx2_whitespace_mask (__m256i chunk) noexcept
{
  const __m256i v_space = _mm256_set1_epi8 (' ');
  const __m256i v_tab = _mm256_set1_epi8 ('\t');
  const __m256i v_nl = _mm256_set1_epi8 ('\n');
  const __m256i v_cr = _mm256_set1_epi8 ('\r');

  __m256i m1 = _mm256_cmpeq_epi8 (chunk, v_space);
  __m256i m2 = _mm256_cmpeq_epi8 (chunk, v_tab);
  __m256i m3 = _mm256_cmpeq_epi8 (chunk, v_nl);
  __m256i m4 = _mm256_cmpeq_epi8 (chunk, v_cr);

  __m256i combined = _mm256_or_si256 (_mm256_or_si256 (m1, m2),
                                      _mm256_or_si256 (m3, m4));
  return static_cast<uint32_t> (_mm256_movemask_epi8 (combined));
}

OBSIDIAN_ALWAYS_INLINE float
hsum256_ps (__m256 v) noexcept
{
  __m128 low = _mm256_castps256_ps128 (v);
  __m128 high = _mm256_extractf128_ps (v, 1);
  __m128 sum128 = _mm_add_ps (low, high);
  __m128 shuf = _mm_movehdup_ps (sum128);
  __m128 sums = _mm_add_ps (sum128, shuf);
  shuf = _mm_movehl_ps (shuf, sums);
  sums = _mm_add_ss (sums, shuf);
  return _mm_cvtss_f32 (sums);
}
#endif

} /* namespace */

const char *
active_isa_name (void) noexcept
{
#if OBSIDIAN_HAS_AVX2
  return "AVX2";
#elif OBSIDIAN_HAS_SSE42
  return "SSE4.2";
#else
  return "Scalar";
#endif
}

size_t
find_first_whitespace (std::string_view text) noexcept
{
  const char *data = text.data ();
  const size_t n = text.size ();
  size_t i = 0;

#if OBSIDIAN_HAS_AVX2
  while (i + 32 <= n)
    {
      __m256i chunk = _mm256_loadu_si256 (
        reinterpret_cast<const __m256i *> (data + i));
      uint32_t mask = avx2_whitespace_mask (chunk);
      if (mask != 0)
        {
          size_t offset = static_cast<size_t> (__builtin_ctz (mask));
          return i + offset;
        }
      i += 32;
    }
#endif

  while (i < n)
    {
      if (is_ascii_space (static_cast<unsigned char> (data[i])))
        {
          return i;
        }
      ++i;
    }
  return std::string_view::npos;
}

size_t
find_first_non_whitespace (std::string_view text) noexcept
{
  const char *data = text.data ();
  const size_t n = text.size ();
  size_t i = 0;

#if OBSIDIAN_HAS_AVX2
  while (i + 32 <= n)
    {
      __m256i chunk = _mm256_loadu_si256 (
        reinterpret_cast<const __m256i *> (data + i));
      uint32_t ws_mask = avx2_whitespace_mask (chunk);
      uint32_t non_ws = ~ws_mask;
      if (non_ws != 0)
        {
          size_t cand = i + static_cast<size_t> (__builtin_ctz (non_ws));
          while (cand < i + 32
                 && is_ascii_space (static_cast<unsigned char> (data[cand])))
            {
              ++cand;
            }
          if (cand < i + 32)
            {
              return cand;
            }
        }
      i += 32;
    }
#endif

  while (i < n)
    {
      if (!is_ascii_space (static_cast<unsigned char> (data[i])))
        {
          return i;
        }
      ++i;
    }
  return std::string_view::npos;
}

void
scan_pretoken_spans (std::string_view text,
                     Pmr_Vector<std::string_view> &out_spans,
                     bool split_punctuation)
{
  size_t pos = 0;
  const size_t n = text.size ();

  while (pos < n)
    {
      size_t non_ws = find_first_non_whitespace (text.substr (pos));
      if (non_ws == std::string_view::npos)
        {
          break;
        }
      pos += non_ws;

      size_t ws_rel = find_first_whitespace (text.substr (pos));
      size_t word_end = (ws_rel == std::string_view::npos) ? n : pos + ws_rel;
      std::string_view chunk = text.substr (pos, word_end - pos);

      if (!split_punctuation)
        {
          out_spans.push_back (chunk);
        }
      else
        {
          size_t run_start = 0;
          for (size_t j = 0; j < chunk.size (); ++j)
            {
              unsigned char ch = static_cast<unsigned char> (chunk[j]);
              if (is_ascii_punct (ch))
                {
                  if (j > run_start)
                    {
                      out_spans.push_back (
                        chunk.substr (run_start, j - run_start));
                    }
                  out_spans.push_back (chunk.substr (j, 1));
                  run_start = j + 1;
                }
            }
          if (run_start < chunk.size ())
            {
              out_spans.push_back (
                chunk.substr (run_start, chunk.size () - run_start));
            }
        }
      pos = word_end;
    }
}

Result<size_t>
validate_and_count_utf8 (std::string_view text) noexcept
{
  const unsigned char *bytes =
    reinterpret_cast<const unsigned char *> (text.data ());
  const size_t n = text.size ();
  size_t i = 0;
  size_t codepoints = 0;

  while (i < n)
    {
      unsigned char b0 = bytes[i];
      if (b0 < 0x80)
        {
          ++i;
          ++codepoints;
        }
      else if ((b0 & 0xE0) == 0xC0)
        {
          if (i + 1 >= n || (bytes[i + 1] & 0xC0) != 0x80 || b0 < 0xC2)
            {
              return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                     "invalid 2-byte UTF-8 sequence");
            }
          i += 2;
          ++codepoints;
        }
      else if ((b0 & 0xF0) == 0xE0)
        {
          if (i + 2 >= n || (bytes[i + 1] & 0xC0) != 0x80
              || (bytes[i + 2] & 0xC0) != 0x80)
            {
              return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                     "invalid 3-byte UTF-8 sequence");
            }
          unsigned char b1 = bytes[i + 1];
          if ((b0 == 0xE0 && b1 < 0xA0)
              || (b0 == 0xED && b1 >= 0xA0))
            {
              return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                     "overlong or surrogate UTF-8 sequence");
            }
          i += 3;
          ++codepoints;
        }
      else if ((b0 & 0xF8) == 0xF0)
        {
          if (i + 3 >= n || (bytes[i + 1] & 0xC0) != 0x80
              || (bytes[i + 2] & 0xC0) != 0x80
              || (bytes[i + 3] & 0xC0) != 0x80)
            {
              return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                     "invalid 4-byte UTF-8 sequence");
            }
          unsigned char b1 = bytes[i + 1];
          if ((b0 == 0xF0 && b1 < 0x90) || b0 > 0xF4
              || (b0 == 0xF4 && b1 >= 0x90))
            {
              return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                     "out-of-range 4-byte UTF-8 codepoint");
            }
          i += 4;
          ++codepoints;
        }
      else
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "unexpected UTF-8 continuation or lead byte");
        }
    }
  return codepoints;
}

void
ascii_to_lower (std::string_view src, char *dst) noexcept
{
  const size_t n = src.size ();
  const char *in = src.data ();
  size_t i = 0;

#if OBSIDIAN_HAS_AVX2
  const __m256i v_a_minus_1 = _mm256_set1_epi8 ('A' - 1);
  const __m256i v_z_plus_1 = _mm256_set1_epi8 ('Z' + 1);
  const __m256i v_diff = _mm256_set1_epi8 (32);

  while (i + 32 <= n)
    {
      __m256i chunk = _mm256_loadu_si256 (
        reinterpret_cast<const __m256i *> (in + i));
      __m256i gt_a = _mm256_cmpgt_epi8 (chunk, v_a_minus_1);
      __m256i lt_z = _mm256_cmpgt_epi8 (v_z_plus_1, chunk);
      __m256i is_upper = _mm256_and_si256 (gt_a, lt_z);
      __m256i mask = _mm256_and_si256 (is_upper, v_diff);
      __m256i lowered = _mm256_or_si256 (chunk, mask);
      _mm256_storeu_si256 (reinterpret_cast<__m256i *> (dst + i), lowered);
      i += 32;
    }
#endif

  while (i < n)
    {
      char c = in[i];
      if (c >= 'A' && c <= 'Z')
        {
          c = static_cast<char> (c + 32);
        }
      dst[i] = c;
      ++i;
    }
}

float
dot_f32 (std::span<const float> a, std::span<const float> b) noexcept
{
  OBSIDIAN_ASSERT (a.size () == b.size ());
  const size_t n = a.size ();
  const float *pa = a.data ();
  const float *pb = b.data ();
  size_t i = 0;
  float total = 0.0f;

#if OBSIDIAN_HAS_AVX2
  __m256 acc0 = _mm256_setzero_ps ();
  __m256 acc1 = _mm256_setzero_ps ();
  while (i + 16 <= n)
    {
      __m256 va0 = _mm256_loadu_ps (pa + i);
      __m256 vb0 = _mm256_loadu_ps (pb + i);
      __m256 va1 = _mm256_loadu_ps (pa + i + 8);
      __m256 vb1 = _mm256_loadu_ps (pb + i + 8);
      acc0 = _mm256_add_ps (acc0, _mm256_mul_ps (va0, vb0));
      acc1 = _mm256_add_ps (acc1, _mm256_mul_ps (va1, vb1));
      i += 16;
    }
  while (i + 8 <= n)
    {
      __m256 va = _mm256_loadu_ps (pa + i);
      __m256 vb = _mm256_loadu_ps (pb + i);
      acc0 = _mm256_add_ps (acc0, _mm256_mul_ps (va, vb));
      i += 8;
    }
  total = hsum256_ps (_mm256_add_ps (acc0, acc1));
#endif

  while (i < n)
    {
      total += pa[i] * pb[i];
      ++i;
    }
  return total;
}

float
norm_sq_f32 (std::span<const float> x) noexcept
{
  return dot_f32 (x, x);
}

float
norm_f32 (std::span<const float> x) noexcept
{
  return std::sqrt (norm_sq_f32 (x));
}

float
cosine_similarity_f32 (std::span<const float> a,
                       std::span<const float> b,
                       float eps) noexcept
{
  OBSIDIAN_ASSERT (a.size () == b.size ());
  const size_t n = a.size ();
  const float *pa = a.data ();
  const float *pb = b.data ();
  size_t i = 0;

  float dot = 0.0f;
  float na = 0.0f;
  float nb = 0.0f;

#if OBSIDIAN_HAS_AVX2
  __m256 v_dot = _mm256_setzero_ps ();
  __m256 v_na = _mm256_setzero_ps ();
  __m256 v_nb = _mm256_setzero_ps ();
  while (i + 8 <= n)
    {
      __m256 va = _mm256_loadu_ps (pa + i);
      __m256 vb = _mm256_loadu_ps (pb + i);
      v_dot = _mm256_add_ps (v_dot, _mm256_mul_ps (va, vb));
      v_na = _mm256_add_ps (v_na, _mm256_mul_ps (va, va));
      v_nb = _mm256_add_ps (v_nb, _mm256_mul_ps (vb, vb));
      i += 8;
    }
  dot = hsum256_ps (v_dot);
  na = hsum256_ps (v_na);
  nb = hsum256_ps (v_nb);
#endif

  while (i < n)
    {
      dot += pa[i] * pb[i];
      na += pa[i] * pa[i];
      nb += pb[i] * pb[i];
      ++i;
    }

  float denom = std::sqrt (na * nb);
  if (denom < eps)
    {
      return 0.0f;
    }
  return dot / denom;
}

void
axpy_f32 (float alpha, std::span<const float> x,
          std::span<float> y) noexcept
{
  OBSIDIAN_ASSERT (x.size () == y.size ());
  const size_t n = x.size ();
  const float *px = x.data ();
  float *py = y.data ();
  size_t i = 0;

#if OBSIDIAN_HAS_AVX2
  __m256 v_alpha = _mm256_set1_ps (alpha);
  while (i + 8 <= n)
    {
      __m256 vx = _mm256_loadu_ps (px + i);
      __m256 vy = _mm256_loadu_ps (py + i);
      vy = _mm256_add_ps (vy, _mm256_mul_ps (v_alpha, vx));
      _mm256_storeu_ps (py + i, vy);
      i += 8;
    }
#endif

  while (i < n)
    {
      py[i] += alpha * px[i];
      ++i;
    }
}

void
normalize_inplace_f32 (std::span<float> x, float eps) noexcept
{
  float nrm = norm_f32 (x);
  if (nrm < eps)
    {
      return;
    }
  float inv = 1.0f / nrm;
  float *px = x.data ();
  const size_t n = x.size ();
  size_t i = 0;

#if OBSIDIAN_HAS_AVX2
  __m256 v_inv = _mm256_set1_ps (inv);
  while (i + 8 <= n)
    {
      __m256 vx = _mm256_loadu_ps (px + i);
      vx = _mm256_mul_ps (vx, v_inv);
      _mm256_storeu_ps (px + i, vx);
      i += 8;
    }
#endif

  while (i < n)
    {
      px[i] *= inv;
      ++i;
    }
}

} /* namespace simd */
} /* namespace obsidian */
