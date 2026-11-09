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

#ifndef SURYA_IMGPROC_COLOR_HPP
#define SURYA_IMGPROC_COLOR_HPP

#include <surya/core.hpp>

namespace surya::inline v1::imgproc
{

/**
 * @brief Convert 3-channel RGB uint8 image to 1-channel Grayscale using
 *        fixed-point Q15 ITU-R BT.601 weights (0.299 R + 0.587 G + 0.114 B).
 * @pre !src.empty().
 * @complexity O(W * H) time and space.
 */
SURYA_API core::Result<core::Image_U8C1>
rgb_to_gray (core::Const_View_U8C3 src);

/**
 * @brief Convert 3-channel BGR uint8 image to 1-channel Grayscale using
 *        fixed-point Q15 ITU-R BT.601 weights.
 * @pre !src.empty().
 * @complexity O(W * H) time and space.
 */
SURYA_API core::Result<core::Image_U8C1>
bgr_to_gray (core::Const_View_U8C3 src);

/**
 * @brief Broadcast 1-channel Grayscale uint8 image to 3-channel RGB.
 * @pre !src.empty().
 * @complexity O(W * H) time and space.
 */
SURYA_API core::Result<core::Image_U8C3>
gray_to_rgb (core::Const_View_U8C1 src);

/**
 * @brief Swap R and B channels between 3-channel RGB and BGR images.
 * @pre !src.empty().
 * @complexity O(W * H) time and space.
 */
SURYA_API core::Result<core::Image_U8C3>
rgb_to_bgr (core::Const_View_U8C3 src);

/**
 * @brief Expand 3-channel RGB uint8 image to 4-channel RGBA with constant
 *        ALPHA channel.
 */
SURYA_API core::Result<core::Image_U8C4>
rgb_to_rgba (core::Const_View_U8C3 src, uint8_t alpha = 255U);

/**
 * @brief Strip alpha channel from 4-channel RGBA uint8 image to 3-channel RGB.
 */
SURYA_API core::Result<core::Image_U8C3>
rgba_to_rgb (core::Const_View_U8C4 src);

/**
 * @brief Convert 3-channel RGB uint8 image to HSV (H in [0, 180], S in
 *        [0, 255], V in [0, 255]).
 */
SURYA_API core::Result<core::Image_U8C3>
rgb_to_hsv (core::Const_View_U8C3 src);

/**
 * @brief Convert 3-channel HSV uint8 image (H in [0, 180], S, V in [0, 255])
 *        back to RGB uint8.
 */
SURYA_API core::Result<core::Image_U8C3>
hsv_to_rgb (core::Const_View_U8C3 src);

/**
 * @brief Convert 3-channel RGB uint8 image to ITU-R BT.601 full-range YCbCr.
 */
SURYA_API core::Result<core::Image_U8C3>
rgb_to_ycbcr (core::Const_View_U8C3 src);

/**
 * @brief Convert 3-channel YCbCr uint8 image back to RGB uint8.
 */
SURYA_API core::Result<core::Image_U8C3>
ycbcr_to_rgb (core::Const_View_U8C3 src);

/**
 * @brief Convert 3-channel RGB uint8 image to CIE L*a*b* float32 (D65 white
 *        point, L in [0, 100], a and b in [-128, 127]).
 */
SURYA_API core::Result<core::Image_F32C3>
rgb_to_lab (core::Const_View_U8C3 src);

/**
 * @brief De-interleave an HWC float32 image into a planar CHW float32 image.
 */
SURYA_API core::Result<core::Image_F32C3_Planar>
interleaved_to_planar (core::Const_View_F32C3 src);

/**
 * @brief Interleave a planar CHW float32 image into an HWC float32 image.
 */
SURYA_API core::Result<core::Image_F32C3>
planar_to_interleaved (
  core::Image_View<const float, 3U, core::Channel_Layout::Planar> src);

} /* namespace surya::inline v1::imgproc */

#endif /* SURYA_IMGPROC_COLOR_HPP */
