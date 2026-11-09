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

#include <surya/classical/classical.hpp>

#include <bellows/bellows.hpp>
#include <hyper_math/scientific/linalg.hpp>
#include <hyper_math/tensor/nd_tensor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

namespace surya::inline v1::classical
{

namespace
{

/* 16-pixel Bresenham circle offsets (radius 3) for FAST-9 detector.  */
constexpr int k_fast_dx[16] = {
  0,  1,  2,  3,  3,  3,  2,  1,
  0, -1, -2, -3, -3, -3, -2, -1
};
constexpr int k_fast_dy[16] = {
 -3, -3, -2, -1,  0,  1,  2,  3,
  3,  3,  2,  1,  0, -1, -2, -3
};

float
fast9_corner_score (const uint8_t *center_ptr,
                    ptrdiff_t row_stride,
                    int threshold) noexcept
{
  const int c = static_cast<int> (*center_ptr);
  const int hi = c + threshold;
  const int lo = c - threshold;

  int ring[16];
  for (size_t k = 0U; k < 16U; ++k)
    {
      ring[k] = static_cast<int> (
        center_ptr[k_fast_dy[k] * row_stride + k_fast_dx[k]]);
    }

  /* Quick cardinal compass rejection test (pixels 0, 4, 8, 12).  */
  int count_hi = (ring[0] > hi) + (ring[4] > hi)
                 + (ring[8] > hi) + (ring[12] > hi);
  int count_lo = (ring[0] < lo) + (ring[4] < lo)
                 + (ring[8] < lo) + (ring[12] < lo);
  if (count_hi < 2 && count_lo < 2)
    {
      return 0.0F;
    }

  /* Check for 9 contiguous brighter or darker pixels around the 16-circle.  */
  int run_hi = 0;
  int run_lo = 0;
  int max_hi = 0;
  int max_lo = 0;
  int diff_sum = 0;

  for (size_t i = 0U; i < 31U; ++i)
    {
      const int v = ring[i & 15U];
      if (v > hi)
        {
          ++run_hi;
          run_lo = 0;
          max_hi = std::max (max_hi, run_hi);
        }
      else if (v < lo)
        {
          ++run_lo;
          run_hi = 0;
          max_lo = std::max (max_lo, run_lo);
        }
      else
        {
          run_hi = 0;
          run_lo = 0;
        }
      if (i < 16U)
        {
          diff_sum += std::abs (v - c);
        }
    }

  if (max_hi >= 9 || max_lo >= 9)
    {
      return static_cast<float> (diff_sum);
    }
  return 0.0F;
}

float
bilinear_sample_u8 (core::Const_View_U8C1 img, float x, float y) noexcept
{
  const float cx = std::clamp (
    x, 0.0F, static_cast<float> (img.width () - 1U));
  const float cy = std::clamp (
    y, 0.0F, static_cast<float> (img.height () - 1U));
  const size_t x0 = static_cast<size_t> (std::floor (cx));
  const size_t y0 = static_cast<size_t> (std::floor (cy));
  const size_t x1 = std::min (x0 + 1U, img.width () - 1U);
  const size_t y1 = std::min (y0 + 1U, img.height () - 1U);
  const float fx = cx - static_cast<float> (x0);
  const float fy = cy - static_cast<float> (y0);

  const float v00 = static_cast<float> (img.row_ptr (y0)[x0]);
  const float v01 = static_cast<float> (img.row_ptr (y0)[x1]);
  const float v10 = static_cast<float> (img.row_ptr (y1)[x0]);
  const float v11 = static_cast<float> (img.row_ptr (y1)[x1]);

  const float top = v00 + fx * (v01 - v00);
  const float bot = v10 + fx * (v11 - v10);
  return top + fy * (bot - top);
}

/* Solve 4-point or N-point DLT homography via Hyper Math eigh (A^T A).  */
std::array<double, 9U>
solve_dlt_homography (std::span<const core::Point2f> src_pts,
                      std::span<const core::Point2f> dst_pts,
                      std::span<const size_t> indices)
{
  const size_t m = indices.size ();
  /* 1. Compute isotropic Hartley normalization for src and dst points.  */
  double src_cx = 0.0;
  double src_cy = 0.0;
  double dst_cx = 0.0;
  double dst_cy = 0.0;
  for (size_t idx : indices)
    {
      src_cx += static_cast<double> (src_pts[idx].x);
      src_cy += static_cast<double> (src_pts[idx].y);
      dst_cx += static_cast<double> (dst_pts[idx].x);
      dst_cy += static_cast<double> (dst_pts[idx].y);
    }
  const double inv_m = 1.0 / static_cast<double> (m);
  src_cx *= inv_m;
  src_cy *= inv_m;
  dst_cx *= inv_m;
  dst_cy *= inv_m;

  double src_dist = 0.0;
  double dst_dist = 0.0;
  for (size_t idx : indices)
    {
      const double sx = static_cast<double> (src_pts[idx].x) - src_cx;
      const double sy = static_cast<double> (src_pts[idx].y) - src_cy;
      const double dx = static_cast<double> (dst_pts[idx].x) - dst_cx;
      const double dy = static_cast<double> (dst_pts[idx].y) - dst_cy;
      src_dist += std::sqrt (sx * sx + sy * sy);
      dst_dist += std::sqrt (dx * dx + dy * dy);
    }
  const double s_src =
    (src_dist > 1e-9)
      ? (std::numbers::sqrt2 * static_cast<double> (m) / src_dist)
      : 1.0;
  const double s_dst =
    (dst_dist > 1e-9)
      ? (std::numbers::sqrt2 * static_cast<double> (m) / dst_dist)
      : 1.0;

  /* 2. Accumulate normal matrix A^T * A in R^{9x9}.  */
  hyper_math::tensor::nd_tensor<double> ata (
    hyper_math::tensor::shape_type {9U, 9U}, 0.0);

  for (size_t idx : indices)
    {
      const double x =
        (static_cast<double> (src_pts[idx].x) - src_cx) * s_src;
      const double y =
        (static_cast<double> (src_pts[idx].y) - src_cy) * s_src;
      const double u =
        (static_cast<double> (dst_pts[idx].x) - dst_cx) * s_dst;
      const double v =
        (static_cast<double> (dst_pts[idx].y) - dst_cy) * s_dst;

      const double r1[9] = {-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u};
      const double r2[9] = {0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v};

      for (size_t i = 0U; i < 9U; ++i)
        {
          for (size_t j = 0U; j < 9U; ++j)
            {
              ata (i, j) += r1[i] * r1[j] + r2[i] * r2[j];
            }
        }
    }

  /* 3. Extract eigenvector of smallest eigenvalue via Hyper Math eigh!  */
  auto eig = hyper_math::scientific::linalg::eigh (ata, 120U, 1e-13);
  double h_norm[9];
  for (size_t i = 0U; i < 9U; ++i)
    {
      h_norm[i] = eig.eigenvectors (i, 0U);
    }

  /* 4. Denormalize H = T_dst^{-1} * H_norm * T_src.  */
  const double t_src[9] = {
    s_src, 0.0,   -s_src * src_cx,
    0.0,   s_src, -s_src * src_cy,
    0.0,   0.0,    1.0
  };
  const double inv_s_dst = 1.0 / s_dst;
  const double t_dst_inv[9] = {
    inv_s_dst, 0.0,       dst_cx,
    0.0,       inv_s_dst, dst_cy,
    0.0,       0.0,       1.0
  };

  double tmp[9] = {0.0};
  for (size_t r = 0U; r < 3U; ++r)
    {
      for (size_t c = 0U; c < 3U; ++c)
        {
          tmp[r * 3U + c] = h_norm[r * 3U + 0U] * t_src[0U * 3U + c]
                            + h_norm[r * 3U + 1U] * t_src[1U * 3U + c]
                            + h_norm[r * 3U + 2U] * t_src[2U * 3U + c];
        }
    }

  std::array<double, 9U> h_out = {0.0};
  for (size_t r = 0U; r < 3U; ++r)
    {
      for (size_t c = 0U; c < 3U; ++c)
        {
          h_out[r * 3U + c] = t_dst_inv[r * 3U + 0U] * tmp[0U * 3U + c]
                              + t_dst_inv[r * 3U + 1U] * tmp[1U * 3U + c]
                              + t_dst_inv[r * 3U + 2U] * tmp[2U * 3U + c];
        }
    }

  if (std::abs (h_out[8]) > 1e-12)
    {
      const double scale = 1.0 / h_out[8];
      for (double &v : h_out)
        {
          v *= scale;
        }
    }
  return h_out;
}

} /* namespace */

core::Result<std::vector<Keypoint>>
detect_fast9_keypoints (core::Const_View_U8C1 src,
                        uint8_t threshold,
                        bool nonmax_suppression,
                        size_t max_keypoints)
{
  if (src.width () < 7U || src.height () < 7U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "detect_fast9_keypoints requires >= 7x7 image");
    }

  const size_t w = src.width ();
  const size_t h = src.height ();
  const ptrdiff_t stride =
    static_cast<ptrdiff_t> (src.row_stride_bytes ());
  const int thresh = static_cast<int> (threshold);

  core::Image_F32C1 score_map (w, h, core::Scalar<float, 1U> (0.0F));
  for (size_t y = 3U; y + 3U < h; ++y)
    {
      const uint8_t *srow = src.row_ptr (y);
      float *drow = score_map.row_ptr (y);
      for (size_t x = 3U; x + 3U < w; ++x)
        {
          drow[x] = fast9_corner_score (srow + x, stride, thresh);
        }
    }

  std::vector<Keypoint> kps;
  for (size_t y = 3U; y + 3U < h; ++y)
    {
      const float *crow = score_map.row_ptr (y);
      const float *prow = score_map.row_ptr (y - 1U);
      const float *nrow = score_map.row_ptr (y + 1U);
      for (size_t x = 3U; x + 3U < w; ++x)
        {
          const float s = crow[x];
          if (s <= 0.0F)
            {
              continue;
            }
          if (nonmax_suppression)
            {
              if (s < crow[x - 1U] || s <= crow[x + 1U]
                  || s < prow[x - 1U] || s < prow[x] || s < prow[x + 1U]
                  || s <= nrow[x - 1U] || s <= nrow[x] || s <= nrow[x + 1U])
                {
                  continue;
                }
            }
          kps.push_back (Keypoint {
            static_cast<float> (x), static_cast<float> (y), s, 0.0F});
        }
    }

  std::sort (kps.begin (), kps.end (),
             [] (const Keypoint &a, const Keypoint &b)
             {
               return a.response > b.response;
             });

  if (kps.size () > max_keypoints)
    {
      kps.resize (max_keypoints);
    }
  return kps;
}

core::Result<std::vector<Orb_Feature>>
extract_orb_features (core::Const_View_U8C1 src,
                      uint8_t fast_threshold,
                      size_t max_features)
{
  if (src.width () < 32U || src.height () < 32U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "extract_orb_features requires >= 32x32 image");
    }

  auto kps_res = detect_fast9_keypoints (
    src, fast_threshold, true, max_features * 2U);
  if (!kps_res)
    {
      return std::unexpected (kps_res.error ());
    }

  const size_t w = src.width ();
  const size_t h = src.height ();
  constexpr int patch_rad = 15;

  std::vector<Orb_Feature> features;
  features.reserve (std::min (kps_res->size (), max_features));

  for (const auto &kp : *kps_res)
    {
      const int cx = static_cast<int> (kp.x);
      const int cy = static_cast<int> (kp.y);
      if (cx < patch_rad || cy < patch_rad
          || cx + patch_rad >= static_cast<int> (w)
          || cy + patch_rad >= static_cast<int> (h))
        {
          continue;
        }

      /* 1. Compute intensity centroid moments m10 and m01 for orientation.  */
      double m10 = 0.0;
      double m01 = 0.0;
      for (int dy = -patch_rad; dy <= patch_rad; ++dy)
        {
          const uint8_t *row =
            src.row_ptr (static_cast<size_t> (cy + dy));
          for (int dx = -patch_rad; dx <= patch_rad; ++dx)
            {
              if (dx * dx + dy * dy <= patch_rad * patch_rad)
                {
                  const double val = static_cast<double> (
                    row[static_cast<size_t> (cx + dx)]);
                  m10 += static_cast<double> (dx) * val;
                  m01 += static_cast<double> (dy) * val;
                }
            }
        }

      const float angle = static_cast<float> (std::atan2 (m01, m10));
      const float cos_a = std::cos (angle);
      const float sin_a = std::sin (angle);

      /* 2. Compute 256-bit steered BRIEF descriptor (4 x 64-bit words).  */
      Orb_Descriptor desc;
      for (size_t bit = 0U; bit < 256U; ++bit)
        {
          /* Deterministic Golden-ratio Halton-like pair inside [-12, 12].  */
          const int u1 = static_cast<int> ((bit * 73U + 19U) % 25U) - 12;
          const int v1 = static_cast<int> ((bit * 131U + 7U) % 25U) - 12;
          const int u2 = static_cast<int> ((bit * 197U + 53U) % 25U) - 12;
          const int v2 = static_cast<int> ((bit * 251U + 11U) % 25U) - 12;

          const int rx1 = std::clamp (
            cx + static_cast<int> (std::lround (
              static_cast<float> (u1) * cos_a
              - static_cast<float> (v1) * sin_a)),
            0, static_cast<int> (w - 1U));
          const int ry1 = std::clamp (
            cy + static_cast<int> (std::lround (
              static_cast<float> (u1) * sin_a
              + static_cast<float> (v1) * cos_a)),
            0, static_cast<int> (h - 1U));
          const int rx2 = std::clamp (
            cx + static_cast<int> (std::lround (
              static_cast<float> (u2) * cos_a
              - static_cast<float> (v2) * sin_a)),
            0, static_cast<int> (w - 1U));
          const int ry2 = std::clamp (
            cy + static_cast<int> (std::lround (
              static_cast<float> (u2) * sin_a
              + static_cast<float> (v2) * cos_a)),
            0, static_cast<int> (h - 1U));

          const uint8_t p1 = src.row_ptr (
            static_cast<size_t> (ry1))[static_cast<size_t> (rx1)];
          const uint8_t p2 = src.row_ptr (
            static_cast<size_t> (ry2))[static_cast<size_t> (rx2)];
          if (p1 < p2)
            {
              desc.words[bit >> 6U] |= (uint64_t {1ULL} << (bit & 63U));
            }
        }

      Keypoint oriented_kp = kp;
      oriented_kp.angle_rad = angle;
      features.push_back (Orb_Feature {oriented_kp, desc});
      if (features.size () >= max_features)
        {
          break;
        }
    }

  return features;
}

std::vector<Descriptor_Match>
match_descriptors_hamming (std::span<const Orb_Feature> query,
                           std::span<const Orb_Feature> train,
                           float lowe_ratio,
                           uint32_t max_distance) noexcept
{
  std::vector<Descriptor_Match> matches;
  if (query.empty () || train.empty ())
    {
      return matches;
    }

  for (size_t q = 0U; q < query.size (); ++q)
    {
      uint32_t best_d1 = 257U;
      uint32_t best_d2 = 257U;
      size_t best_t = 0U;

      for (size_t t = 0U; t < train.size (); ++t)
        {
          const uint32_t d = query[q].desc.hamming_distance (train[t].desc);
          if (d < best_d1)
            {
              best_d2 = best_d1;
              best_d1 = d;
              best_t = t;
            }
          else if (d < best_d2)
            {
              best_d2 = d;
            }
        }

      if (best_d1 <= max_distance)
        {
          if (train.size () == 1U
              || static_cast<float> (best_d1)
                   <= lowe_ratio * static_cast<float> (best_d2))
            {
              matches.push_back (Descriptor_Match {q, best_t, best_d1});
            }
        }
    }

  return matches;
}

size_t
HOG_Extractor::descriptor_length (size_t width, size_t height) const noexcept
{
  if (cell_size_ == 0U || block_cells_ == 0U || num_bins_ == 0U)
    {
      return 0U;
    }
  const size_t cells_x = width / cell_size_;
  const size_t cells_y = height / cell_size_;
  if (cells_x < block_cells_ || cells_y < block_cells_)
    {
      return 0U;
    }
  const size_t blocks_x = cells_x - block_cells_ + 1U;
  const size_t blocks_y = cells_y - block_cells_ + 1U;
  return blocks_x * blocks_y * block_cells_ * block_cells_ * num_bins_;
}

core::Result<std::vector<double>>
HOG_Extractor::compute (core::Const_View_U8C1 src) const
{
  const size_t desc_len = descriptor_length (src.width (), src.height ());
  if (desc_len == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "HOG_Extractor image smaller than one block");
    }

  const size_t cells_x = src.width () / cell_size_;
  const size_t cells_y = src.height () / cell_size_;
  std::vector<double> cell_hists (cells_y * cells_x * num_bins_, 0.0);

  const double bin_width =
    std::numbers::pi / static_cast<double> (num_bins_);

  for (size_t cy = 0U; cy < cells_y; ++cy)
    {
      for (size_t cx = 0U; cx < cells_x; ++cx)
        {
          double *hist =
            cell_hists.data () + (cy * cells_x + cx) * num_bins_;
          for (size_t dy = 0U; dy < cell_size_; ++dy)
            {
              const size_t y = cy * cell_size_ + dy;
              const size_t ym = (y > 0U) ? (y - 1U) : 0U;
              const size_t yp =
                (y + 1U < src.height ()) ? (y + 1U) : (src.height () - 1U);
              const uint8_t *row_c = src.row_ptr (y);
              const uint8_t *row_m = src.row_ptr (ym);
              const uint8_t *row_p = src.row_ptr (yp);

              for (size_t dx = 0U; dx < cell_size_; ++dx)
                {
                  const size_t x = cx * cell_size_ + dx;
                  const size_t xm = (x > 0U) ? (x - 1U) : 0U;
                  const size_t xp =
                    (x + 1U < src.width ()) ? (x + 1U) : (src.width () - 1U);

                  const double gx = static_cast<double> (row_c[xp])
                                    - static_cast<double> (row_c[xm]);
                  const double gy = static_cast<double> (row_p[x])
                                    - static_cast<double> (row_m[x]);
                  const double mag = std::sqrt (gx * gx + gy * gy);
                  double ang = std::atan2 (gy, gx);
                  if (ang < 0.0)
                    {
                      ang += std::numbers::pi;
                    }
                  if (ang >= std::numbers::pi)
                    {
                      ang -= std::numbers::pi;
                    }

                  const double b_f = ang / bin_width;
                  const size_t b0 =
                    static_cast<size_t> (b_f) % num_bins_;
                  const size_t b1 = (b0 + 1U) % num_bins_;
                  const double frac = b_f - std::floor (b_f);

                  hist[b0] += mag * (1.0 - frac);
                  hist[b1] += mag * frac;
                }
            }
        }
    }

  /* Block L2-Hys normalization across (block_cells_ x block_cells_) cells.  */
  const size_t blocks_x = cells_x - block_cells_ + 1U;
  const size_t blocks_y = cells_y - block_cells_ + 1U;
  const size_t block_dim = block_cells_ * block_cells_ * num_bins_;

  std::vector<double> out;
  out.reserve (desc_len);
  std::vector<double> blk (block_dim, 0.0);

  for (size_t by = 0U; by < blocks_y; ++by)
    {
      for (size_t bx = 0U; bx < blocks_x; ++bx)
        {
          size_t k = 0U;
          double sq_sum = 0.0;
          for (size_t dy = 0U; dy < block_cells_; ++dy)
            {
              for (size_t dx = 0U; dx < block_cells_; ++dx)
                {
                  const double *ch =
                    cell_hists.data ()
                    + ((by + dy) * cells_x + (bx + dx)) * num_bins_;
                  for (size_t b = 0U; b < num_bins_; ++b)
                    {
                      blk[k] = ch[b];
                      sq_sum += ch[b] * ch[b];
                      ++k;
                    }
                }
            }

          const double inv_norm = 1.0 / std::sqrt (sq_sum + 1e-6);
          double clipped_sq = 0.0;
          for (size_t i = 0U; i < block_dim; ++i)
            {
              blk[i] = std::min (0.2, blk[i] * inv_norm);
              clipped_sq += blk[i] * blk[i];
            }
          const double inv_norm2 = 1.0 / std::sqrt (clipped_sq + 1e-6);
          for (size_t i = 0U; i < block_dim; ++i)
            {
              out.push_back (blk[i] * inv_norm2);
            }
        }
    }

  return out;
}

core::Result<Homography_Result>
estimate_homography_ransac (std::span<const core::Point2f> src_pts,
                            std::span<const core::Point2f> dst_pts,
                            float reproj_threshold,
                            size_t max_iterations,
                            uint64_t seed)
{
  const size_t n = src_pts.size ();
  if (n < 4U || dst_pts.size () != n)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "estimate_homography_ransac requires >= 4 pts");
    }

  std::mt19937_64 rng (seed);
  std::uniform_int_distribution<size_t> dist (0U, n - 1U);
  const float thresh_sq = reproj_threshold * reproj_threshold;

  Homography_Result best_res;
  best_res.inlier_mask.assign (n, 0U);

  for (size_t iter = 0U; iter < max_iterations; ++iter)
    {
      size_t sample[4];
      for (size_t k = 0U; k < 4U; ++k)
        {
          bool unique = false;
          while (!unique)
            {
              sample[k] = dist (rng);
              unique = true;
              for (size_t j = 0U; j < k; ++j)
                {
                  if (sample[j] == sample[k])
                    {
                      unique = false;
                      break;
                    }
                }
            }
        }

      Homography_Result cand;
      cand.h_matrix = solve_dlt_homography (src_pts, dst_pts, sample);
      cand.inlier_mask.assign (n, 0U);

      for (size_t i = 0U; i < n; ++i)
        {
          const core::Point2f proj = cand.project_point (src_pts[i]);
          const float dx = proj.x - dst_pts[i].x;
          const float dy = proj.y - dst_pts[i].y;
          if (dx * dx + dy * dy <= thresh_sq)
            {
              cand.inlier_mask[i] = 1U;
              ++cand.num_inliers;
            }
        }

      if (cand.num_inliers > best_res.num_inliers)
        {
          best_res = std::move (cand);
          if (best_res.num_inliers == n)
            {
              break;
            }
        }
    }

  /* Refine H over all inliers using normalized DLT.  */
  if (best_res.num_inliers >= 4U)
    {
      std::vector<size_t> inlier_indices;
      inlier_indices.reserve (best_res.num_inliers);
      for (size_t i = 0U; i < n; ++i)
        {
          if (best_res.inlier_mask[i] != 0U)
            {
              inlier_indices.push_back (i);
            }
        }
      best_res.h_matrix =
        solve_dlt_homography (src_pts, dst_pts, inlier_indices);
    }

  return best_res;
}

core::Result<Template_Match_Result>
match_template (core::Const_View_U8C1 image,
                core::Const_View_U8C1 templ,
                Template_Match_Mode mode)
{
  if (image.empty () || templ.empty ()
      || templ.width () > image.width ()
      || templ.height () > image.height ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "match_template invalid template dimensions");
    }

  const size_t out_w = image.width () - templ.width () + 1U;
  const size_t out_h = image.height () - templ.height () + 1U;
  const size_t tw = templ.width ();
  const size_t th = templ.height ();

  double t_mean = 0.0;
  double t_norm_sq = 0.0;
  if (mode == Template_Match_Mode::NCC)
    {
      for (size_t y = 0U; y < th; ++y)
        {
          const uint8_t *trow = templ.row_ptr (y);
          for (size_t x = 0U; x < tw; ++x)
            {
              t_mean += static_cast<double> (trow[x]);
            }
        }
      t_mean /= static_cast<double> (tw * th);
      for (size_t y = 0U; y < th; ++y)
        {
          const uint8_t *trow = templ.row_ptr (y);
          for (size_t x = 0U; x < tw; ++x)
            {
              const double d = static_cast<double> (trow[x]) - t_mean;
              t_norm_sq += d * d;
            }
        }
    }

  Template_Match_Result res;
  res.response_map =
    core::Image_F32C1 (out_w, out_h, core::Scalar<float, 1U> (0.0F));
  res.best_score = (mode == Template_Match_Mode::NCC)
                     ? -std::numeric_limits<float>::infinity ()
                     : std::numeric_limits<float>::infinity ();

  for (size_t oy = 0U; oy < out_h; ++oy)
    {
      float *rrow = res.response_map.row_ptr (oy);
      for (size_t ox = 0U; ox < out_w; ++ox)
        {
          float score = 0.0F;
          if (mode == Template_Match_Mode::SAD)
            {
              uint64_t sad = 0ULL;
              for (size_t y = 0U; y < th; ++y)
                {
                  const uint8_t *irow = image.row_ptr (oy + y) + ox;
                  const uint8_t *trow = templ.row_ptr (y);
                  for (size_t x = 0U; x < tw; ++x)
                    {
                      sad += static_cast<uint64_t> (std::abs (
                        static_cast<int> (irow[x])
                        - static_cast<int> (trow[x])));
                    }
                }
              score = static_cast<float> (sad);
              if (score < res.best_score)
                {
                  res.best_score = score;
                  res.best_x = ox;
                  res.best_y = oy;
                }
            }
          else if (mode == Template_Match_Mode::SSD)
            {
              double ssd = 0.0;
              for (size_t y = 0U; y < th; ++y)
                {
                  const uint8_t *irow = image.row_ptr (oy + y) + ox;
                  const uint8_t *trow = templ.row_ptr (y);
                  for (size_t x = 0U; x < tw; ++x)
                    {
                      const double d = static_cast<double> (irow[x])
                                       - static_cast<double> (trow[x]);
                      ssd += d * d;
                    }
                }
              score = static_cast<float> (ssd);
              if (score < res.best_score)
                {
                  res.best_score = score;
                  res.best_x = ox;
                  res.best_y = oy;
                }
            }
          else
            {
              double p_mean = 0.0;
              for (size_t y = 0U; y < th; ++y)
                {
                  const uint8_t *irow = image.row_ptr (oy + y) + ox;
                  for (size_t x = 0U; x < tw; ++x)
                    {
                      p_mean += static_cast<double> (irow[x]);
                    }
                }
              p_mean /= static_cast<double> (tw * th);

              double num = 0.0;
              double p_norm_sq = 0.0;
              for (size_t y = 0U; y < th; ++y)
                {
                  const uint8_t *irow = image.row_ptr (oy + y) + ox;
                  const uint8_t *trow = templ.row_ptr (y);
                  for (size_t x = 0U; x < tw; ++x)
                    {
                      const double di =
                        static_cast<double> (irow[x]) - p_mean;
                      const double dt =
                        static_cast<double> (trow[x]) - t_mean;
                      num += di * dt;
                      p_norm_sq += di * di;
                    }
                }
              const double denom = std::sqrt (p_norm_sq * t_norm_sq);
              score = (denom > 1e-8)
                        ? static_cast<float> (num / denom)
                        : 0.0F;
              if (score > res.best_score)
                {
                  res.best_score = score;
                  res.best_x = ox;
                  res.best_y = oy;
                }
            }
          rrow[ox] = score;
        }
    }

  return res;
}

core::Result<Flow_Track_Result>
calc_optical_flow_lk (core::Const_View_U8C1 prev_img,
                      core::Const_View_U8C1 next_img,
                      std::span<const core::Point2f> prev_pts,
                      size_t win_radius,
                      size_t max_iterations,
                      float epsilon)
{
  if (prev_img.empty () || next_img.empty ()
      || prev_img.width () != next_img.width ()
      || prev_img.height () != next_img.height ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "calc_optical_flow_lk dimension mismatch");
    }

  const size_t n = prev_pts.size ();
  Flow_Track_Result out;
  out.next_points.resize (n);
  out.status.assign (n, 0U);
  out.error.assign (n, 0.0F);

  const int rad = static_cast<int> (win_radius);
  const float w_f = static_cast<float> (prev_img.width ());
  const float h_f = static_cast<float> (prev_img.height ());

  for (size_t i = 0U; i < n; ++i)
    {
      const float px = prev_pts[i].x;
      const float py = prev_pts[i].y;
      if (px < static_cast<float> (rad + 1)
          || py < static_cast<float> (rad + 1)
          || px + static_cast<float> (rad + 1) >= w_f
          || py + static_cast<float> (rad + 1) >= h_f)
        {
          out.next_points[i] = prev_pts[i];
          continue;
        }

      /* Precompute spatial structure tensor G = [gxx, gxy; gxy, gyy].  */
      double gxx = 0.0;
      double gxy = 0.0;
      double gyy = 0.0;
      for (int dy = -rad; dy <= rad; ++dy)
        {
          for (int dx = -rad; dx <= rad; ++dx)
            {
              const float sx = px + static_cast<float> (dx);
              const float sy = py + static_cast<float> (dy);
              const double ix = 0.5 * static_cast<double> (
                bilinear_sample_u8 (prev_img, sx + 1.0F, sy)
                - bilinear_sample_u8 (prev_img, sx - 1.0F, sy));
              const double iy = 0.5 * static_cast<double> (
                bilinear_sample_u8 (prev_img, sx, sy + 1.0F)
                - bilinear_sample_u8 (prev_img, sx, sy - 1.0F));
              gxx += ix * ix;
              gxy += ix * iy;
              gyy += iy * iy;
            }
        }

      const double det = gxx * gyy - gxy * gxy;
      if (det < 1e-4)
        {
          out.next_points[i] = prev_pts[i];
          continue;
        }
      const double inv_det = 1.0 / det;

      float qx = px;
      float qy = py;
      for (size_t iter = 0U; iter < max_iterations; ++iter)
        {
          double bx = 0.0;
          double by = 0.0;
          for (int dy = -rad; dy <= rad; ++dy)
            {
              for (int dx = -rad; dx <= rad; ++dx)
                {
                  const float sx = px + static_cast<float> (dx);
                  const float sy = py + static_cast<float> (dy);
                  const float tx = qx + static_cast<float> (dx);
                  const float ty = qy + static_cast<float> (dy);

                  const double it = static_cast<double> (
                    bilinear_sample_u8 (next_img, tx, ty)
                    - bilinear_sample_u8 (prev_img, sx, sy));
                  const double ix = 0.5 * static_cast<double> (
                    bilinear_sample_u8 (prev_img, sx + 1.0F, sy)
                    - bilinear_sample_u8 (prev_img, sx - 1.0F, sy));
                  const double iy = 0.5 * static_cast<double> (
                    bilinear_sample_u8 (prev_img, sx, sy + 1.0F)
                    - bilinear_sample_u8 (prev_img, sx, sy - 1.0F));
                  bx -= ix * it;
                  by -= iy * it;
                }
            }

          const float vx =
            static_cast<float> ((gyy * bx - gxy * by) * inv_det);
          const float vy =
            static_cast<float> ((-gxy * bx + gxx * by) * inv_det);
          qx += vx;
          qy += vy;
          if (vx * vx + vy * vy < epsilon * epsilon)
            {
              break;
            }
        }

      out.next_points[i] = core::Point2f (qx, qy);
      out.status[i] = 1U;
    }

  return out;
}

/* 5. Classical ML Vision Pipelines delegating to Bellows.  */
struct HOG_SVM_Classifier::Impl
{
  HOG_Extractor hog;
  bellows::max_margin_classifier svm;

  Impl (HOG_Extractor h, double pen, double lr, size_t epochs)
    : hog (h), svm (pen, lr, epochs)
  {
  }
};

HOG_SVM_Classifier::HOG_SVM_Classifier (HOG_Extractor hog,
                                        double penalty,
                                        double learning_rate,
                                        size_t max_epochs)
  : impl_ (std::make_shared<Impl> (hog, penalty, learning_rate, max_epochs))
{
}

HOG_SVM_Classifier::~HOG_SVM_Classifier (void) noexcept = default;

core::Result<void>
HOG_SVM_Classifier::fit (std::span<const core::Const_View_U8C1> patches,
                         std::span<const int> labels)
{
  if (patches.empty () || patches.size () != labels.size ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "HOG_SVM_Classifier::fit invalid input count");
    }

  auto first_hog = impl_->hog.compute (patches[0]);
  if (!first_hog)
    {
      return std::unexpected (first_hog.error ());
    }

  const size_t n = patches.size ();
  const size_t d = first_hog->size ();
  bellows::data_table table (n, d, 0.0);
  std::vector<double> targets (n, 0.0);

  for (size_t r = 0U; r < n; ++r)
    {
      auto feat = (r == 0U) ? first_hog : impl_->hog.compute (patches[r]);
      if (!feat || feat->size () != d)
        {
          return core::make_error (core::Error_Code::Dimension_Mismatch,
                                   "HOG_SVM_Classifier patch size mismatch");
        }
      for (size_t c = 0U; c < d; ++c)
        {
          table.at (r, c) = (*feat)[c];
        }
      targets[r] = static_cast<double> (labels[r]);
    }

  impl_->svm.calibrate (table, targets);
  return {};
}

core::Result<int>
HOG_SVM_Classifier::predict (core::Const_View_U8C1 patch) const
{
  if (!impl_->svm.is_calibrated ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "HOG_SVM_Classifier is not calibrated");
    }
  auto feat = impl_->hog.compute (patch);
  if (!feat)
    {
      return std::unexpected (feat.error ());
    }
  bellows::data_table row_tab (1U, feat->size (), 0.0);
  for (size_t c = 0U; c < feat->size (); ++c)
    {
      row_tab.at (0U, c) = (*feat)[c];
    }
  auto preds = impl_->svm.evaluate (row_tab);
  return static_cast<int> (std::lround (preds[0]));
}

core::Result<std::vector<double>>
HOG_SVM_Classifier::predict_probabilities (core::Const_View_U8C1 patch) const
{
  if (!impl_->svm.is_calibrated ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "HOG_SVM_Classifier is not calibrated");
    }
  auto feat = impl_->hog.compute (patch);
  if (!feat)
    {
      return std::unexpected (feat.error ());
    }
  return impl_->svm.evaluate_beliefs_one (*feat);
}

struct Visual_Vocabulary_BoVW::Impl
{
  size_t num_clusters = 8U;
  bellows::centroid_partitioner kmeans;

  Impl (size_t k, size_t max_iters, uint64_t seed)
    : num_clusters (k), kmeans (k, max_iters, seed)
  {
  }
};

Visual_Vocabulary_BoVW::Visual_Vocabulary_BoVW (size_t num_clusters,
                                                size_t max_iterations,
                                                uint64_t seed)
  : impl_ (std::make_shared<Impl> (num_clusters, max_iterations, seed))
{
}

Visual_Vocabulary_BoVW::~Visual_Vocabulary_BoVW (void) noexcept = default;

core::Result<void>
Visual_Vocabulary_BoVW::fit (const core::SuryaTensor &descriptors)
{
  if (descriptors.ndim () != 2U
      || descriptors.size (0) < impl_->num_clusters
      || descriptors.size (1) == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "Visual_Vocabulary_BoVW insufficient rows");
    }

  const size_t rows = descriptors.size (0);
  const size_t cols = descriptors.size (1);
  bellows::data_table tab (rows, cols, 0.0);
  for (size_t r = 0U; r < rows; ++r)
    {
      for (size_t c = 0U; c < cols; ++c)
        {
          tab.at (r, c) = static_cast<double> (descriptors.at_2d (r, c));
        }
    }
  impl_->kmeans.calibrate (tab);
  return {};
}

core::Result<std::vector<double>>
Visual_Vocabulary_BoVW::encode_histogram (
  const core::SuryaTensor &image_descriptors) const
{
  if (!impl_->kmeans.is_calibrated ()
      || image_descriptors.ndim () != 2U
      || image_descriptors.size (0) == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "Visual_Vocabulary_BoVW not calibrated");
    }

  const size_t rows = image_descriptors.size (0);
  const size_t cols = image_descriptors.size (1);
  bellows::data_table tab (rows, cols, 0.0);
  for (size_t r = 0U; r < rows; ++r)
    {
      for (size_t c = 0U; c < cols; ++c)
        {
          tab.at (r, c) =
            static_cast<double> (image_descriptors.at_2d (r, c));
        }
    }

  auto assignments = impl_->kmeans.evaluate (tab);
  std::vector<double> hist (impl_->num_clusters, 0.0);
  for (double idx_d : assignments)
    {
      const size_t c = std::min (
        static_cast<size_t> (std::max (0L, std::lround (idx_d))),
        impl_->num_clusters - 1U);
      hist[c] += 1.0;
    }

  const double inv_total = 1.0 / static_cast<double> (rows);
  for (double &v : hist)
    {
      v *= inv_total;
    }
  return hist;
}

struct Patch_Forest_Classifier::Impl
{
  HOG_Extractor hog;
  bellows::subspace_tree_ensemble forest;

  Impl (HOG_Extractor h, size_t trees, size_t depth, uint64_t seed)
    : hog (h),
      forest (trees, depth, bellows::proximity_mode::classification, seed)
  {
  }
};

Patch_Forest_Classifier::Patch_Forest_Classifier (HOG_Extractor hog,
                                                  size_t tree_count,
                                                  size_t max_depth,
                                                  uint64_t seed)
  : impl_ (std::make_shared<Impl> (hog, tree_count, max_depth, seed))
{
}

Patch_Forest_Classifier::~Patch_Forest_Classifier (void) noexcept = default;

core::Result<void>
Patch_Forest_Classifier::fit (std::span<const core::Const_View_U8C1> patches,
                              std::span<const int> labels)
{
  if (patches.empty () || patches.size () != labels.size ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "Patch_Forest_Classifier::fit invalid input");
    }

  auto first_hog = impl_->hog.compute (patches[0]);
  if (!first_hog)
    {
      return std::unexpected (first_hog.error ());
    }

  const size_t n = patches.size ();
  const size_t d = first_hog->size ();
  bellows::data_table table (n, d, 0.0);
  std::vector<double> targets (n, 0.0);

  for (size_t r = 0U; r < n; ++r)
    {
      auto feat = (r == 0U) ? first_hog : impl_->hog.compute (patches[r]);
      if (!feat || feat->size () != d)
        {
          return core::make_error (core::Error_Code::Dimension_Mismatch,
                                   "Patch_Forest_Classifier size mismatch");
        }
      for (size_t c = 0U; c < d; ++c)
        {
          table.at (r, c) = (*feat)[c];
        }
      targets[r] = static_cast<double> (labels[r]);
    }

  impl_->forest.calibrate (table, targets);
  return {};
}

core::Result<int>
Patch_Forest_Classifier::predict (core::Const_View_U8C1 patch) const
{
  if (!impl_->forest.is_calibrated ())
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "Patch_Forest_Classifier not calibrated");
    }
  auto feat = impl_->hog.compute (patch);
  if (!feat)
    {
      return std::unexpected (feat.error ());
    }
  bellows::data_table row_tab (1U, feat->size (), 0.0);
  for (size_t c = 0U; c < feat->size (); ++c)
    {
      row_tab.at (0U, c) = (*feat)[c];
    }
  auto preds = impl_->forest.evaluate (row_tab);
  return static_cast<int> (std::lround (preds[0]));
}

} /* namespace surya::inline v1::classical */
