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

#include <surya/imgproc/morphology.hpp>

#include <algorithm>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace surya::inline v1::imgproc
{

namespace
{

std::vector<std::pair<ptrdiff_t, ptrdiff_t>>
build_offsets (size_t kwidth, size_t kheight, Structuring_Element shape)
{
  std::vector<std::pair<ptrdiff_t, ptrdiff_t>> offsets;
  const ptrdiff_t rx = static_cast<ptrdiff_t> (kwidth / 2U);
  const ptrdiff_t ry = static_cast<ptrdiff_t> (kheight / 2U);
  const double inv_rx2 = rx > 0 ? (1.0 / static_cast<double> (rx * rx)) : 0.0;
  const double inv_ry2 = ry > 0 ? (1.0 / static_cast<double> (ry * ry)) : 0.0;

  for (ptrdiff_t dy = -ry; dy <= ry; ++dy)
    {
      for (ptrdiff_t dx = -rx; dx <= rx; ++dx)
        {
          bool keep = true;
          if (shape == Structuring_Element::Cross)
            {
              keep = (dx == 0 || dy == 0);
            }
          else if (shape == Structuring_Element::Ellipse)
            {
              const double d =
                static_cast<double> (dx * dx) * inv_rx2
                + static_cast<double> (dy * dy) * inv_ry2;
              keep = (d <= 1.0001);
            }
          if (keep)
            {
              offsets.emplace_back (dy, dx);
            }
        }
    }
  return offsets;
}

core::Result<core::Image_U8C1>
morph_basic (core::Const_View_U8C1 src,
             size_t kwidth,
             size_t kheight,
             Structuring_Element shape,
             bool is_dilate)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "morphology: input image is empty");
    }
  if (kwidth == 0U || (kwidth % 2U) == 0U || kheight == 0U
      || (kheight % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "morphology: kernel dimensions must be positive odd integers");
    }

  const size_t w = src.width ();
  const size_t h = src.height ();

  /* Fast separable path for rectangular structuring elements.  */
  if (shape == Structuring_Element::Rect)
    {
      auto tmp_res = core::Image_U8C1::create (w, h, false);
      auto dst_res = core::Image_U8C1::create (w, h, false);
      if (!tmp_res.has_value () || !dst_res.has_value ())
        {
          return core::make_error (core::Error_Code::Allocation_Failed,
                                   "morphology: allocation failed");
        }
      core::Image_U8C1 tmp = std::move (*tmp_res);
      core::Image_U8C1 dst = std::move (*dst_res);

      const ptrdiff_t rx = static_cast<ptrdiff_t> (kwidth / 2U);
      const ptrdiff_t ry = static_cast<ptrdiff_t> (kheight / 2U);
      const ptrdiff_t pw = static_cast<ptrdiff_t> (w);
      const ptrdiff_t ph = static_cast<ptrdiff_t> (h);

      /* Horizontal pass.  */
      for (size_t y = 0U; y < h; ++y)
        {
          const uint8_t *s_row = src.row_ptr (y);
          uint8_t *t_row = tmp.row_ptr (y);
          for (ptrdiff_t x = 0; x < pw; ++x)
            {
              const size_t x0 =
                static_cast<size_t> (std::max<ptrdiff_t> (0, x - rx));
              const size_t x1 =
                static_cast<size_t> (std::min<ptrdiff_t> (pw - 1, x + rx));
              uint8_t best = s_row[x0];
              if (is_dilate)
                {
                  for (size_t k = x0 + 1U; k <= x1; ++k)
                    {
                      best = std::max (best, s_row[k]);
                    }
                }
              else
                {
                  for (size_t k = x0 + 1U; k <= x1; ++k)
                    {
                      best = std::min (best, s_row[k]);
                    }
                }
              t_row[static_cast<size_t> (x)] = best;
            }
        }

      /* Vertical pass with AVX2 vectorization across scanlines.  */
      for (ptrdiff_t y = 0; y < ph; ++y)
        {
          const size_t y0 =
            static_cast<size_t> (std::max<ptrdiff_t> (0, y - ry));
          const size_t y1 =
            static_cast<size_t> (std::min<ptrdiff_t> (ph - 1, y + ry));
          uint8_t *d_row = dst.row_ptr (static_cast<size_t> (y));

          size_t x = 0U;
#if defined(__AVX2__)
          for (; x + 32U <= w; x += 32U)
            {
              __m256i acc = _mm256_loadu_si256 (
                reinterpret_cast<const __m256i *> (tmp.row_ptr (y0) + x));
              for (size_t ky = y0 + 1U; ky <= y1; ++ky)
                {
                  const __m256i v = _mm256_loadu_si256 (
                    reinterpret_cast<const __m256i *> (tmp.row_ptr (ky) + x));
                  acc = is_dilate ? _mm256_max_epu8 (acc, v)
                                  : _mm256_min_epu8 (acc, v);
                }
              _mm256_storeu_si256 (reinterpret_cast<__m256i *> (d_row + x),
                                   acc);
            }
#endif
          for (; x < w; ++x)
            {
              uint8_t best = tmp.row_ptr (y0)[x];
              for (size_t ky = y0 + 1U; ky <= y1; ++ky)
                {
                  const uint8_t v = tmp.row_ptr (ky)[x];
                  best = is_dilate ? std::max (best, v) : std::min (best, v);
                }
              d_row[x] = best;
            }
        }
      return dst;
    }

  /* General structuring element path (Cross / Ellipse).  */
  auto dst_res = core::Image_U8C1::create (w, h, false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  const auto offsets = build_offsets (kwidth, kheight, shape);
  const ptrdiff_t pw = static_cast<ptrdiff_t> (w);
  const ptrdiff_t ph = static_cast<ptrdiff_t> (h);

  for (ptrdiff_t y = 0; y < ph; ++y)
    {
      uint8_t *d_row = dst.row_ptr (static_cast<size_t> (y));
      for (ptrdiff_t x = 0; x < pw; ++x)
        {
          uint8_t best = is_dilate ? 0U : 255U;
          for (const auto &[dy, dx] : offsets)
            {
              const size_t sy = static_cast<size_t> (
                std::clamp<ptrdiff_t> (y + dy, 0, ph - 1));
              const size_t sx = static_cast<size_t> (
                std::clamp<ptrdiff_t> (x + dx, 0, pw - 1));
              const uint8_t v = src.at (sy, sx);
              best = is_dilate ? std::max (best, v) : std::min (best, v);
            }
          d_row[static_cast<size_t> (x)] = best;
        }
    }
  return dst;
}

} /* namespace */

core::Result<core::Image_U8C1>
erode (core::Const_View_U8C1 src,
       size_t kwidth,
       size_t kheight,
       Structuring_Element shape)
{
  return morph_basic (src, kwidth, kheight, shape, false);
}

core::Result<core::Image_U8C1>
dilate (core::Const_View_U8C1 src,
        size_t kwidth,
        size_t kheight,
        Structuring_Element shape)
{
  return morph_basic (src, kwidth, kheight, shape, true);
}

core::Result<core::Image_U8C1>
morphology_ex (core::Const_View_U8C1 src,
               Morph_Op op,
               size_t kwidth,
               size_t kheight,
               Structuring_Element shape)
{
  switch (op)
    {
    case Morph_Op::Erode:
      return erode (src, kwidth, kheight, shape);
    case Morph_Op::Dilate:
      return dilate (src, kwidth, kheight, shape);
    case Morph_Op::Open:
      {
        auto er = erode (src, kwidth, kheight, shape);
        if (!er.has_value ())
          {
            return er;
          }
        return dilate (er->const_view (), kwidth, kheight, shape);
      }
    case Morph_Op::Close:
      {
        auto dl = dilate (src, kwidth, kheight, shape);
        if (!dl.has_value ())
          {
            return dl;
          }
        return erode (dl->const_view (), kwidth, kheight, shape);
      }
    case Morph_Op::Gradient:
      {
        auto dl = dilate (src, kwidth, kheight, shape);
        auto er = erode (src, kwidth, kheight, shape);
        if (!dl.has_value () || !er.has_value ())
          {
            return core::make_error (core::Error_Code::Allocation_Failed,
                                     "morphology_ex: gradient failed");
          }
        core::Image_U8C1 out = std::move (*dl);
        for (size_t y = 0U; y < out.height (); ++y)
          {
            core::simd_abs_diff_u8 (out.row_ptr (y), out.row_ptr (y),
                                    er->row_ptr (y), out.width ());
          }
        return out;
      }
    case Morph_Op::Top_Hat:
      {
        auto opened =
          morphology_ex (src, Morph_Op::Open, kwidth, kheight, shape);
        if (!opened.has_value ())
          {
            return opened;
          }
        core::Image_U8C1 out = std::move (*opened);
        for (size_t y = 0U; y < out.height (); ++y)
          {
            const uint8_t *s = src.row_ptr (y);
            uint8_t *d = out.row_ptr (y);
            for (size_t x = 0U; x < out.width (); ++x)
              {
                d[x] = s[x] > d[x] ? static_cast<uint8_t> (s[x] - d[x]) : 0U;
              }
          }
        return out;
      }
    case Morph_Op::Black_Hat:
      {
        auto closed =
          morphology_ex (src, Morph_Op::Close, kwidth, kheight, shape);
        if (!closed.has_value ())
          {
            return closed;
          }
        core::Image_U8C1 out = std::move (*closed);
        for (size_t y = 0U; y < out.height (); ++y)
          {
            const uint8_t *s = src.row_ptr (y);
            uint8_t *d = out.row_ptr (y);
            for (size_t x = 0U; x < out.width (); ++x)
              {
                d[x] = d[x] > s[x] ? static_cast<uint8_t> (d[x] - s[x]) : 0U;
              }
          }
        return out;
      }
    }
  return core::make_error (core::Error_Code::Invalid_Argument,
                           "morphology_ex: unknown Morph_Op");
}

} /* namespace surya::inline v1::imgproc */
