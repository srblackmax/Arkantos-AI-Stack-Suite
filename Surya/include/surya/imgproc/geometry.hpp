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

#ifndef SURYA_IMGPROC_GEOMETRY_HPP
#define SURYA_IMGPROC_GEOMETRY_HPP

#include <array>
#include <cmath>
#include <surya/core.hpp>

namespace surya::inline v1::imgproc
{

/**
 * @brief Resampling interpolation mode for resize and geometric warps.
 */
enum class Interpolation_Mode : uint8_t
{
  Nearest = 0,
  Bilinear = 1,
  Bicubic = 2,
  Area = 3
};

/**
 * @brief 3x3 projective homography matrix in row-major order.
 *
 * Also represents 2D affine transforms when the bottom row is [0, 0, 1].
 */
struct SURYA_API Homography_3x3
{
  std::array<double, 9U> m {1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0};

  constexpr Homography_3x3 (void) noexcept = default;

  constexpr Homography_3x3 (double m00, double m01, double m02,
                            double m10, double m11, double m12,
                            double m20 = 0.0, double m21 = 0.0,
                            double m22 = 1.0) noexcept
    : m {m00, m01, m02, m10, m11, m12, m20, m21, m22}
  {
  }

  static constexpr Homography_3x3
  identity (void) noexcept
  {
    return Homography_3x3 ();
  }

  static constexpr Homography_3x3
  translation (double tx, double ty) noexcept
  {
    return Homography_3x3 (1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0);
  }

  static constexpr Homography_3x3
  scaling (double sx, double sy) noexcept
  {
    return Homography_3x3 (sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0);
  }

  static Homography_3x3
  rotation_scale_around (core::Point2f center,
                         double angle_deg,
                         double scale = 1.0) noexcept;

  /**
   * @brief Compose two 3x3 homographies: (this) * other.
   * @pre None.
   * @complexity O(1) time (27 FMAs).
   */
  constexpr Homography_3x3
  operator* (const Homography_3x3 &rhs) const noexcept
  {
    Homography_3x3 out {};
    for (size_t r = 0U; r < 3U; ++r)
      {
        for (size_t c = 0U; c < 3U; ++c)
          {
            out.m[r * 3U + c] =
              m[r * 3U + 0U] * rhs.m[0U * 3U + c]
              + m[r * 3U + 1U] * rhs.m[1U * 3U + c]
              + m[r * 3U + 2U] * rhs.m[2U * 3U + c];
          }
      }
    return out;
  }

  /**
   * @brief Compute the analytical 3x3 matrix inverse if non-singular.
   */
  core::Result<Homography_3x3>
  inverse (void) const noexcept;

  /**
   * @brief Project a 2D point PT through this homography.
   */
  constexpr core::Point2f
  transform_point (core::Point2f pt) const noexcept
  {
    const double x = static_cast<double> (pt.x);
    const double y = static_cast<double> (pt.y);
    const double w = m[6] * x + m[7] * y + m[8];
    const double inv_w = std::abs (w) > 1e-12 ? (1.0 / w) : 1.0;
    const double nx = (m[0] * x + m[1] * y + m[2]) * inv_w;
    const double ny = (m[3] * x + m[4] * y + m[5]) * inv_w;
    return core::Point2f (static_cast<float> (nx), static_cast<float> (ny));
  }
};

/**
 * @brief Resize a 1-channel uint8 image to DST_WIDTH x DST_HEIGHT using
 *        precomputed horizontal and vertical interpolation coefficient tables.
 */
SURYA_API core::Result<core::Image_U8C1>
resize (core::Const_View_U8C1 src,
        size_t dst_width,
        size_t dst_height,
        Interpolation_Mode mode = Interpolation_Mode::Bilinear);

/**
 * @brief Resize a 3-channel RGB uint8 image to DST_WIDTH x DST_HEIGHT.
 */
SURYA_API core::Result<core::Image_U8C3>
resize (core::Const_View_U8C3 src,
        size_t dst_width,
        size_t dst_height,
        Interpolation_Mode mode = Interpolation_Mode::Bilinear);

/**
 * @brief Resize a 1-channel float32 image to DST_WIDTH x DST_HEIGHT.
 */
SURYA_API core::Result<core::Image_F32C1>
resize (core::Const_View_F32C1 src,
        size_t dst_width,
        size_t dst_height,
        Interpolation_Mode mode = Interpolation_Mode::Bilinear);

/**
 * @brief Warp a 1-channel uint8 image by a forward 2x3 affine matrix (stored
 *        in the upper 2x3 of FORWARD_AFFINE) using inverse bilinear mapping.
 */
SURYA_API core::Result<core::Image_U8C1>
warp_affine (
  core::Const_View_U8C1 src,
  const Homography_3x3 &forward_affine,
  size_t dst_width,
  size_t dst_height,
  Interpolation_Mode mode = Interpolation_Mode::Bilinear,
  core::Border_Mode border = core::Border_Mode::Constant,
  uint8_t border_value = 0U);

/**
 * @brief Warp a 3-channel uint8 image by a forward 2x3 affine transform.
 */
SURYA_API core::Result<core::Image_U8C3>
warp_affine (
  core::Const_View_U8C3 src,
  const Homography_3x3 &forward_affine,
  size_t dst_width,
  size_t dst_height,
  Interpolation_Mode mode = Interpolation_Mode::Bilinear,
  core::Border_Mode border = core::Border_Mode::Constant,
  core::Scalar<uint8_t, 3U> border_value =
    core::Scalar<uint8_t, 3U> (0U, 0U, 0U));

/**
 * @brief Warp a 1-channel uint8 image by a general 3x3 forward homography.
 */
SURYA_API core::Result<core::Image_U8C1>
warp_perspective (
  core::Const_View_U8C1 src,
  const Homography_3x3 &forward_homography,
  size_t dst_width,
  size_t dst_height,
  Interpolation_Mode mode = Interpolation_Mode::Bilinear,
  core::Border_Mode border = core::Border_Mode::Constant,
  uint8_t border_value = 0U);

/**
 * @brief Warp a 3-channel uint8 image by a general 3x3 forward homography.
 */
SURYA_API core::Result<core::Image_U8C3>
warp_perspective (
  core::Const_View_U8C3 src,
  const Homography_3x3 &forward_homography,
  size_t dst_width,
  size_t dst_height,
  Interpolation_Mode mode = Interpolation_Mode::Bilinear,
  core::Border_Mode border = core::Border_Mode::Constant,
  core::Scalar<uint8_t, 3U> border_value =
    core::Scalar<uint8_t, 3U> (0U, 0U, 0U));

} /* namespace surya::inline v1::imgproc */

#endif /* SURYA_IMGPROC_GEOMETRY_HPP */
