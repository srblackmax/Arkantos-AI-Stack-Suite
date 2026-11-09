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

#include <surya/imgproc/analysis.hpp>
#include <surya/imgproc/filter.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace surya::inline v1::imgproc
{

core::Result<core::Image_U8C1>
threshold (core::Const_View_U8C1 src,
           uint8_t thresh,
           uint8_t max_val,
           Threshold_Mode mode)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "threshold: input image is empty");
    }
  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  std::array<uint8_t, 256U> lut {};
  for (size_t i = 0U; i < 256U; ++i)
    {
      const uint8_t v = static_cast<uint8_t> (i);
      switch (mode)
        {
        case Threshold_Mode::Binary:
          lut[i] = v > thresh ? max_val : 0U;
          break;
        case Threshold_Mode::Binary_Inv:
          lut[i] = v > thresh ? 0U : max_val;
          break;
        case Threshold_Mode::Truncate:
          lut[i] = v > thresh ? thresh : v;
          break;
        case Threshold_Mode::To_Zero:
          lut[i] = v > thresh ? v : 0U;
          break;
        case Threshold_Mode::To_Zero_Inv:
          lut[i] = v > thresh ? 0U : v;
          break;
        }
    }

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          d[x] = lut[s[x]];
        }
    }
  return dst;
}

core::Result<std::pair<uint8_t, core::Image_U8C1>>
otsu_threshold (core::Const_View_U8C1 src,
                uint8_t max_val,
                Threshold_Mode mode)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "otsu_threshold: input image is empty");
    }

  std::array<uint64_t, 256U> hist {};
  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          ++hist[s[x]];
        }
    }

  const double total =
    static_cast<double> (src.width () * src.height ());
  double sum_all = 0.0;
  for (size_t i = 0U; i < 256U; ++i)
    {
      sum_all += static_cast<double> (i) * static_cast<double> (hist[i]);
    }

  double sum_bg = 0.0;
  double w_bg = 0.0;
  double max_var = -1.0;
  uint8_t best_t = 0U;

  for (size_t t = 0U; t < 256U; ++t)
    {
      w_bg += static_cast<double> (hist[t]);
      if (w_bg <= 0.0)
        {
          continue;
        }
      const double w_fg = total - w_bg;
      if (w_fg <= 0.0)
        {
          break;
        }
      sum_bg += static_cast<double> (t) * static_cast<double> (hist[t]);
      const double mean_bg = sum_bg / w_bg;
      const double mean_fg = (sum_all - sum_bg) / w_fg;
      const double diff = mean_bg - mean_fg;
      const double between_var = w_bg * w_fg * diff * diff;
      if (between_var > max_var)
        {
          max_var = between_var;
          best_t = static_cast<uint8_t> (t);
        }
    }

  auto img_res = threshold (src, best_t, max_val, mode);
  if (!img_res.has_value ())
    {
      return std::unexpected (img_res.error ());
    }
  return std::make_pair (best_t, std::move (*img_res));
}

core::Result<core::Image_U8C1>
adaptive_threshold (core::Const_View_U8C1 src,
                    uint8_t max_val,
                    size_t block_size,
                    float c_offset,
                    Threshold_Mode mode)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "adaptive_threshold: input image is empty");
    }
  if (block_size < 3U || (block_size % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "adaptive_threshold: block_size must be an odd integer >= 3");
    }
  if (mode != Threshold_Mode::Binary && mode != Threshold_Mode::Binary_Inv)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "adaptive_threshold: mode must be Binary or Binary_Inv");
    }

  auto sat_res = integral_image (src);
  if (!sat_res.has_value ())
    {
      return std::unexpected (sat_res.error ());
    }
  const core::Image_I32C1 &sat = *sat_res;
  auto sat_v = sat.const_view ();

  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  const ptrdiff_t r = static_cast<ptrdiff_t> (block_size / 2U);
  const ptrdiff_t w = static_cast<ptrdiff_t> (src.width ());
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());

  for (ptrdiff_t y = 0; y < h; ++y)
    {
      const size_t y0 = static_cast<size_t> (std::max<ptrdiff_t> (0, y - r));
      const size_t y1 =
        static_cast<size_t> (std::min<ptrdiff_t> (h, y + r + 1));
      const uint8_t *s_row = src.row_ptr (static_cast<size_t> (y));
      uint8_t *d_row = dst.row_ptr (static_cast<size_t> (y));

      for (ptrdiff_t x = 0; x < w; ++x)
        {
          const size_t x0 =
            static_cast<size_t> (std::max<ptrdiff_t> (0, x - r));
          const size_t x1 =
            static_cast<size_t> (std::min<ptrdiff_t> (w, x + r + 1));
          const float area =
            static_cast<float> ((y1 - y0) * (x1 - x0));
          const float sum =
            static_cast<float> (integral_box_sum (sat_v, x0, y0, x1, y1));
          const float local_thresh = (sum / area) - c_offset;
          const bool above =
            static_cast<float> (s_row[static_cast<size_t> (x)]) > local_thresh;
          if (mode == Threshold_Mode::Binary)
            {
              d_row[static_cast<size_t> (x)] = above ? max_val : 0U;
            }
          else
            {
              d_row[static_cast<size_t> (x)] = above ? 0U : max_val;
            }
        }
    }
  return dst;
}

core::Result<core::Image_U8C1>
canny (core::Const_View_U8C1 src,
       float low_threshold,
       float high_threshold,
       size_t gaussian_ksize,
       float gaussian_sigma)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "canny: input image is empty");
    }
  if (low_threshold < 0.0F || high_threshold < low_threshold)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "canny: require 0 <= low_threshold <= high_threshold");
    }

  auto blurred_res = gaussian_blur (src, gaussian_ksize, gaussian_sigma);
  if (!blurred_res.has_value ())
    {
      return std::unexpected (blurred_res.error ());
    }
  const core::Image_U8C1 &smoothed = *blurred_res;

  auto gx_res = sobel_3x3 (smoothed.const_view (), 1, 0);
  auto gy_res = sobel_3x3 (smoothed.const_view (), 0, 1);
  if (!gx_res.has_value () || !gy_res.has_value ())
    {
      return core::make_error (core::Error_Code::Allocation_Failed,
                               "canny: failed to compute Sobel gradients");
    }
  const core::Image_F32C1 &gx = *gx_res;
  const core::Image_F32C1 &gy = *gy_res;

  const size_t w = src.width ();
  const size_t h = src.height ();
  auto mag_res = core::Image_F32C1::create (w, h, true);
  auto dir_res = core::Image_U8C1::create (w, h, true);
  auto edge_res = core::Image_U8C1::create (w, h, true);
  if (!mag_res.has_value () || !dir_res.has_value () || !edge_res.has_value ())
    {
      return core::make_error (core::Error_Code::Allocation_Failed,
                               "canny: buffer allocation failed");
    }
  core::Image_F32C1 mag = std::move (*mag_res);
  core::Image_U8C1 dir = std::move (*dir_res);
  core::Image_U8C1 edges = std::move (*edge_res);

  /* tan(22.5 deg) = 0.41421356, tan(67.5 deg) = 2.41421356.  */
  constexpr float kTan225 = 0.41421356F;
  constexpr float kTan675 = 2.41421356F;

  for (size_t y = 0U; y < h; ++y)
    {
      const float *rx = gx.row_ptr (y);
      const float *ry = gy.row_ptr (y);
      float *rm = mag.row_ptr (y);
      uint8_t *rd = dir.row_ptr (y);

      for (size_t x = 0U; x < w; ++x)
        {
          const float vx = rx[x];
          const float vy = ry[x];
          rm[x] = std::hypot (vx, vy);

          const float ax = std::abs (vx);
          const float ay = std::abs (vy);
          if (ay <= ax * kTan225)
            {
              rd[x] = 0U; /* 0 degrees (horizontal gradient) */
            }
          else if (ay >= ax * kTan675)
            {
              rd[x] = 2U; /* 90 degrees (vertical gradient) */
            }
          else
            {
              rd[x] = (vx * vy > 0.0F) ? 1U : 3U; /* 45 or 135 degrees */
            }
        }
    }

  /* Non-maximum suppression and double thresholding:
     255 = strong edge, 128 = weak candidate, 0 = suppressed.  */
  std::vector<std::pair<size_t, size_t>> strong_stack;
  if (w > 2U && h > 2U)
    {
      for (size_t y = 1U; y + 1U < h; ++y)
        {
          const float *m_prev = mag.row_ptr (y - 1U);
          const float *m_curr = mag.row_ptr (y);
          const float *m_next = mag.row_ptr (y + 1U);
          const uint8_t *d_row = dir.row_ptr (y);
          uint8_t *e_row = edges.row_ptr (y);

          for (size_t x = 1U; x + 1U < w; ++x)
            {
              const float m = m_curr[x];
              if (m < low_threshold)
                {
                  continue;
                }
              float n1 = 0.0F;
              float n2 = 0.0F;
              switch (d_row[x])
                {
                case 0U:
                  n1 = m_curr[x - 1U];
                  n2 = m_curr[x + 1U];
                  break;
                case 1U:
                  n1 = m_prev[x - 1U];
                  n2 = m_next[x + 1U];
                  break;
                case 2U:
                  n1 = m_prev[x];
                  n2 = m_next[x];
                  break;
                default:
                  n1 = m_prev[x + 1U];
                  n2 = m_next[x - 1U];
                  break;
                }
              if (m >= n1 && m >= n2)
                {
                  if (m >= high_threshold)
                    {
                      e_row[x] = 255U;
                      strong_stack.emplace_back (y, x);
                    }
                  else
                    {
                      e_row[x] = 128U;
                    }
                }
            }
        }
    }

  /* Hysteresis edge tracking via DFS from strong edges.  */
  while (!strong_stack.empty ())
    {
      const auto [cy, cx] = strong_stack.back ();
      strong_stack.pop_back ();

      for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dx = -1; dx <= 1; ++dx)
            {
              if (dy == 0 && dx == 0)
                {
                  continue;
                }
              const size_t ny =
                static_cast<size_t> (static_cast<ptrdiff_t> (cy) + dy);
              const size_t nx =
                static_cast<size_t> (static_cast<ptrdiff_t> (cx) + dx);
              if (ny < h && nx < w && edges.at (ny, nx) == 128U)
                {
                  edges.at (ny, nx) = 255U;
                  strong_stack.emplace_back (ny, nx);
                }
            }
        }
    }

  /* Clear remaining unpromoted weak edges.  */
  for (size_t y = 0U; y < h; ++y)
    {
      uint8_t *e_row = edges.row_ptr (y);
      for (size_t x = 0U; x < w; ++x)
        {
          if (e_row[x] != 255U)
            {
              e_row[x] = 0U;
            }
        }
    }

  return edges;
}

namespace
{

int32_t
uf_find (std::vector<int32_t> &parent, int32_t x) noexcept
{
  int32_t root = x;
  while (parent[static_cast<size_t> (root)] != root)
    {
      root = parent[static_cast<size_t> (root)];
    }
  while (x != root)
    {
      const int32_t nxt = parent[static_cast<size_t> (x)];
      parent[static_cast<size_t> (x)] = root;
      x = nxt;
    }
  return root;
}

void
uf_union (std::vector<int32_t> &parent, int32_t a, int32_t b) noexcept
{
  const int32_t ra = uf_find (parent, a);
  const int32_t rb = uf_find (parent, b);
  if (ra != rb)
    {
      if (ra < rb)
        {
          parent[static_cast<size_t> (rb)] = ra;
        }
      else
        {
          parent[static_cast<size_t> (ra)] = rb;
        }
    }
}

} /* namespace */

core::Result<Connected_Components_Result>
connected_components (core::Const_View_U8C1 binary_src, int connectivity)
{
  if (binary_src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "connected_components: input image is empty");
    }
  if (connectivity != 4 && connectivity != 8)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "connected_components: connectivity must be 4 or 8");
    }

  const size_t w = binary_src.width ();
  const size_t h = binary_src.height ();
  auto lbl_res = core::Image_I32C1::create (w, h, true);
  if (!lbl_res.has_value ())
    {
      return std::unexpected (lbl_res.error ());
    }
  core::Image_I32C1 labels = std::move (*lbl_res);

  std::vector<int32_t> parent (1U, 0);
  int32_t next_label = 1;

  /* Pass 1: provisional labeling and union-find equivalence recording.  */
  for (size_t y = 0U; y < h; ++y)
    {
      const uint8_t *s_row = binary_src.row_ptr (y);
      int32_t *l_row = labels.row_ptr (y);
      const int32_t *l_prev = y > 0U ? labels.row_ptr (y - 1U) : nullptr;

      for (size_t x = 0U; x < w; ++x)
        {
          if (s_row[x] == 0U)
            {
              continue;
            }

          int32_t neighbors[4];
          size_t n_count = 0U;
          if (x > 0U && l_row[x - 1U] > 0)
            {
              neighbors[n_count++] = l_row[x - 1U];
            }
          if (l_prev != nullptr)
            {
              if (l_prev[x] > 0)
                {
                  neighbors[n_count++] = l_prev[x];
                }
              if (connectivity == 8)
                {
                  if (x > 0U && l_prev[x - 1U] > 0)
                    {
                      neighbors[n_count++] = l_prev[x - 1U];
                    }
                  if (x + 1U < w && l_prev[x + 1U] > 0)
                    {
                      neighbors[n_count++] = l_prev[x + 1U];
                    }
                }
            }

          if (n_count == 0U)
            {
              const int32_t id = next_label++;
              parent.push_back (id);
              l_row[x] = id;
            }
          else
            {
              int32_t min_lbl = neighbors[0];
              for (size_t i = 1U; i < n_count; ++i)
                {
                  min_lbl = std::min (min_lbl, neighbors[i]);
                }
              l_row[x] = min_lbl;
              for (size_t i = 0U; i < n_count; ++i)
                {
                  uf_union (parent, min_lbl, neighbors[i]);
                }
            }
        }
    }

  /* Flatten union-find trees and assign compact sequential labels 1..K.  */
  std::vector<int32_t> remap (parent.size (), 0);
  int32_t compact_count = 0;
  for (size_t i = 1U; i < parent.size (); ++i)
    {
      const int32_t root = uf_find (parent, static_cast<int32_t> (i));
      if (remap[static_cast<size_t> (root)] == 0)
        {
          remap[static_cast<size_t> (root)] = ++compact_count;
        }
      remap[i] = remap[static_cast<size_t> (root)];
    }

  struct Accum
  {
    size_t area = 0U;
    size_t min_x = ~static_cast<size_t> (0U);
    size_t min_y = ~static_cast<size_t> (0U);
    size_t max_x = 0U;
    size_t max_y = 0U;
    double sum_x = 0.0;
    double sum_y = 0.0;
  };
  std::vector<Accum> acc (static_cast<size_t> (compact_count) + 1U);

  /* Pass 2: relabel image and accumulate bounding boxes & centroids.  */
  for (size_t y = 0U; y < h; ++y)
    {
      int32_t *l_row = labels.row_ptr (y);
      for (size_t x = 0U; x < w; ++x)
        {
          const int32_t raw = l_row[x];
          if (raw > 0)
            {
              const int32_t cid = remap[static_cast<size_t> (raw)];
              l_row[x] = cid;
              Accum &a = acc[static_cast<size_t> (cid)];
              ++a.area;
              a.min_x = std::min (a.min_x, x);
              a.min_y = std::min (a.min_y, y);
              a.max_x = std::max (a.max_x, x);
              a.max_y = std::max (a.max_y, y);
              a.sum_x += static_cast<double> (x);
              a.sum_y += static_cast<double> (y);
            }
        }
    }

  Connected_Components_Result out;
  out.labels = std::move (labels);
  out.components.reserve (static_cast<size_t> (compact_count));
  for (int32_t cid = 1; cid <= compact_count; ++cid)
    {
      const Accum &a = acc[static_cast<size_t> (cid)];
      Component_Stats st;
      st.label = cid;
      st.area = a.area;
      st.bbox = core::Rectu (a.min_x, a.min_y, a.max_x - a.min_x + 1U,
                             a.max_y - a.min_y + 1U);
      const double inv_a =
        a.area > 0U ? (1.0 / static_cast<double> (a.area)) : 0.0;
      st.centroid = core::Point2f (static_cast<float> (a.sum_x * inv_a),
                                   static_cast<float> (a.sum_y * inv_a));
      out.components.push_back (st);
    }

  return out;
}

} /* namespace surya::inline v1::imgproc */
