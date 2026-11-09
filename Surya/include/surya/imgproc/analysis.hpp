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

#ifndef SURYA_IMGPROC_ANALYSIS_HPP
#define SURYA_IMGPROC_ANALYSIS_HPP

#include <cstdint>
#include <utility>
#include <vector>
#include <surya/core.hpp>

namespace surya::inline v1::imgproc
{

/**
 * @brief Pixel thresholding rule.
 */
enum class Threshold_Mode : uint8_t
{
  Binary = 0,
  Binary_Inv = 1,
  Truncate = 2,
  To_Zero = 3,
  To_Zero_Inv = 4
};

/**
 * @brief Apply fixed-level thresholding to a 1-channel uint8 image.
 */
SURYA_API core::Result<core::Image_U8C1>
threshold (core::Const_View_U8C1 src,
           uint8_t thresh,
           uint8_t max_val = 255U,
           Threshold_Mode mode = Threshold_Mode::Binary);

/**
 * @brief Compute the optimal Otsu between-class variance threshold and return
 *        both the threshold value and the thresholded binary image.
 */
SURYA_API core::Result<std::pair<uint8_t, core::Image_U8C1>>
otsu_threshold (core::Const_View_U8C1 src,
                uint8_t max_val = 255U,
                Threshold_Mode mode = Threshold_Mode::Binary);

/**
 * @brief Apply O(1)-per-pixel integral-image local mean adaptive thresholding.
 * @pre BLOCK_SIZE >= 3 and (BLOCK_SIZE % 2) == 1.
 * @complexity O(W * H) time and space.
 */
SURYA_API core::Result<core::Image_U8C1>
adaptive_threshold (core::Const_View_U8C1 src,
                    uint8_t max_val,
                    size_t block_size,
                    float c_offset,
                    Threshold_Mode mode = Threshold_Mode::Binary);

/**
 * @brief Canny multi-stage edge detector (Gaussian smoothing, Sobel gradient
 *        magnitude & 4-sector orientation, non-maximum suppression, and
 *        hysteresis edge tracking).
 */
SURYA_API core::Result<core::Image_U8C1>
canny (core::Const_View_U8C1 src,
       float low_threshold,
       float high_threshold,
       size_t gaussian_ksize = 5U,
       float gaussian_sigma = 1.0F);

/**
 * @brief Geometric summary statistics for a single connected foreground
 *        component.
 */
struct Component_Stats
{
  int32_t label = 0;
  size_t area = 0U;
  core::Rectu bbox {};
  core::Point2f centroid {};
};

/**
 * @brief Output of connected-components labeling (label map + stats).
 */
struct Connected_Components_Result
{
  core::Image_I32C1 labels;
  std::vector<Component_Stats> components;
};

/**
 * @brief Perform 4-connected or 8-connected union-find component labeling and
 *        simultaneously extract bounding boxes, pixel areas, and centroids.
 * @pre CONNECTIVITY == 4 or CONNECTIVITY == 8.
 * @complexity O(W * H * alpha(W * H)) time and O(W * H) space.
 */
SURYA_API core::Result<Connected_Components_Result>
connected_components (core::Const_View_U8C1 binary_src,
                      int connectivity = 8);

} /* namespace surya::inline v1::imgproc */

#endif /* SURYA_IMGPROC_ANALYSIS_HPP */
