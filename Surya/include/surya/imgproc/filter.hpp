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

#ifndef SURYA_IMGPROC_FILTER_HPP
#define SURYA_IMGPROC_FILTER_HPP

#include <span>
#include <vector>
#include <surya/core.hpp>

namespace surya::inline v1::imgproc
{

/**
 * @brief Generate a normalized 1D Gaussian kernel of odd length KSIZE and
 *        standard deviation SIGMA.
 * @pre KSIZE >= 1 and (KSIZE % 2) == 1.
 * @complexity O(KSIZE) time and space.
 */
SURYA_API core::Result<std::vector<float>>
gaussian_kernel_1d (size_t ksize, float sigma);

/**
 * @brief Apply a 2D separable filter (KX horizontally, KY vertically) to a
 *        1-channel uint8 image using AVX2 FMA vertical vectorization.
 */
SURYA_API core::Result<core::Image_U8C1>
separable_filter_2d (
  core::Const_View_U8C1 src,
  std::span<const float> kx,
  std::span<const float> ky,
  core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply a 2D separable filter to a 3-channel RGB uint8 image.
 */
SURYA_API core::Result<core::Image_U8C3>
separable_filter_2d (
  core::Const_View_U8C3 src,
  std::span<const float> kx,
  std::span<const float> ky,
  core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply a 2D separable filter to a 1-channel float32 image.
 */
SURYA_API core::Result<core::Image_F32C1>
separable_filter_2d (
  core::Const_View_F32C1 src,
  std::span<const float> kx,
  std::span<const float> ky,
  core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply a separable Gaussian blur to a 1-channel uint8 image.
 */
SURYA_API core::Result<core::Image_U8C1>
gaussian_blur (core::Const_View_U8C1 src,
               size_t ksize,
               float sigma,
               core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply a separable Gaussian blur to a 3-channel uint8 image.
 */
SURYA_API core::Result<core::Image_U8C3>
gaussian_blur (core::Const_View_U8C3 src,
               size_t ksize,
               float sigma,
               core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply a separable Gaussian blur to a 1-channel float32 image.
 */
SURYA_API core::Result<core::Image_F32C1>
gaussian_blur (core::Const_View_F32C1 src,
               size_t ksize,
               float sigma,
               core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply an O(1)-per-pixel sliding-window box filter of size
 *        KWIDTH x KHEIGHT to a 1-channel uint8 image.
 */
SURYA_API core::Result<core::Image_U8C1>
box_filter (core::Const_View_U8C1 src, size_t kwidth, size_t kheight);

/**
 * @brief Compute 3x3 Sobel spatial derivatives (DX_ORDER, DY_ORDER in {0, 1})
 *        producing a float32 response map.
 */
SURYA_API core::Result<core::Image_F32C1>
sobel_3x3 (core::Const_View_U8C1 src,
           int dx_order,
           int dy_order,
           core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Compute 3x3 Scharr rotational-symmetry derivatives ([3, 10, 3]
 *        smoothing weight).
 */
SURYA_API core::Result<core::Image_F32C1>
scharr_3x3 (core::Const_View_U8C1 src,
            int dx_order,
            int dy_order,
            core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Compute 4-connected 3x3 discrete Laplacian response.
 */
SURYA_API core::Result<core::Image_F32C1>
laplacian_3x3 (core::Const_View_U8C1 src,
               core::Border_Mode border = core::Border_Mode::Reflect_101);

/**
 * @brief Apply a branchless 9-element sorting-network 3x3 median filter.
 * @pre !src.empty().
 * @complexity O(W * H) time (19 compare-and-swaps per pixel).
 */
SURYA_API core::Result<core::Image_U8C1>
median_filter_3x3 (core::Const_View_U8C1 src);

/**
 * @brief Apply an edge-preserving bilateral filter using a precomputed
 *        256-entry intensity range LUT and spatial Gaussian weights.
 */
SURYA_API core::Result<core::Image_U8C1>
bilateral_filter (core::Const_View_U8C1 src,
                  size_t diameter,
                  float sigma_color,
                  float sigma_space);

/**
 * @brief Compute the summed-area table (integral image) of size
 *        (W + 1) x (H + 1) where I(y, x) is the sum of pixels in [0, y) x
 *        [0, x).
 * @pre !src.empty().
 * @complexity O(W * H) time and space.
 */
SURYA_API core::Result<core::Image_I32C1>
integral_image (core::Const_View_U8C1 src);

/**
 * @brief Query the exact sum of pixels inside rectangle [X0, X1) x [Y0, Y1)
 *        in O(1) time from an (W + 1) x (H + 1) integral image.
 */
SURYA_INLINE int32_t
integral_box_sum (core::Image_View<const int32_t, 1U> sat,
                  size_t x0,
                  size_t y0,
                  size_t x1,
                  size_t y1) noexcept
{
  return sat.at (y1, x1) - sat.at (y0, x1) - sat.at (y1, x0) + sat.at (y0, x0);
}

} /* namespace surya::inline v1::imgproc */

#endif /* SURYA_IMGPROC_FILTER_HPP */
