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

#ifndef SURYA_NN_TENSOR_BRIDGE_HPP
#define SURYA_NN_TENSOR_BRIDGE_HPP

#include <array>
#include <span>
#include <surya/core.hpp>

namespace surya::inline v1::nn
{

/**
 * @brief Per-channel affine normalization parameters for vision models:
 *        out[c] = (pixel[c] * scale - mean[c]) / std[c].
 * @note Thread-safety: Value type.
 */
struct Normalize_Params
{
  float scale = 1.0F / 255.0F;
  std::array<float, 3U> mean {0.0F, 0.0F, 0.0F};
  std::array<float, 3U> std {1.0F, 1.0F, 1.0F};

  static constexpr Normalize_Params
  unit_scale (void) noexcept
  {
    return Normalize_Params {};
  }

  static constexpr Normalize_Params
  imagenet (void) noexcept
  {
    Normalize_Params p;
    p.scale = 1.0F / 255.0F;
    p.mean = {0.485F, 0.456F, 0.406F};
    p.std = {0.229F, 0.224F, 0.225F};
    return p;
  }

  static constexpr Normalize_Params
  zero_centered (void) noexcept
  {
    Normalize_Params p;
    p.scale = 1.0F / 255.0F;
    p.mean = {0.5F, 0.5F, 0.5F};
    p.std = {0.5F, 0.5F, 0.5F};
    return p;
  }
};

/**
 * @brief Convert an HWC uint8 RGB/BGR image into a [1, 3, H, W] float32
 *        SuryaTensor in a single fused de-interleave + normalize pass.
 * @pre !src.empty() && norm.std[c] != 0.
 * @post Returns [1, 3, H, W] float32 SuryaTensor.
 * @returns std::expected<core::SuryaTensor, SuryaError>.
 * @note Thread-safety: Safe to call concurrently on distinct views.
 */
SURYA_API core::Result<core::SuryaTensor>
image_to_tensor_nchw (
  core::Const_View_U8C3 src,
  const Normalize_Params &norm = Normalize_Params::unit_scale (),
  bool swap_rb = false);

/**
 * @brief Convert a 1-channel uint8 image into a [1, 1, H, W] float32
 *        SuryaTensor with affine scaling (px * scale - mean) / std_dev.
 * @pre !src.empty() && std_dev != 0.
 * @post Returns [1, 1, H, W] float32 SuryaTensor.
 * @returns std::expected<core::SuryaTensor, SuryaError>.
 * @note Thread-safety: Safe to call concurrently on distinct views.
 */
SURYA_API core::Result<core::SuryaTensor>
image_to_tensor_nchw (core::Const_View_U8C1 src,
                      float scale = 1.0F / 255.0F,
                      float mean = 0.0F,
                      float std_dev = 1.0F);

/**
 * @brief Convert a batch of equal-sized HWC uint8 images into a contiguous
 *        [B, 3, H, W] float32 SuryaTensor in parallel.
 * @pre !batch.empty() && all images share identical non-zero dimensions.
 * @post Returns [B, 3, H, W] float32 SuryaTensor.
 * @returns std::expected<core::SuryaTensor, SuryaError>.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API core::Result<core::SuryaTensor>
batch_images_to_tensor_nchw (
  std::span<const core::Const_View_U8C3> batch,
  const Normalize_Params &norm = Normalize_Params::unit_scale (),
  bool swap_rb = false);

/**
 * @brief Convert slice BATCH_IDX of an NCHW ([B, 3, H, W] or [3, H, W])
 *        SuryaTensor back into a Surya Image_F32C3 (HWC).
 * @pre !tensor.empty().
 * @post Returns interleaved HWC Image_F32C3.
 * @returns std::expected<core::Image_F32C3, SuryaError>.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API core::Result<core::Image_F32C3>
tensor_nchw_to_image_f32 (const core::SuryaTensor &tensor,
                          size_t batch_idx = 0U);

/**
 * @brief Convert slice BATCH_IDX of an NCHW ([B, 3, H, W] or [3, H, W])
 *        SuryaTensor in [0, 1] back into a Surya Image_U8C3 (HWC).
 * @pre !tensor.empty().
 * @post Returns interleaved HWC Image_U8C3.
 * @returns std::expected<core::Image_U8C3, SuryaError>.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API core::Result<core::Image_U8C3>
tensor_nchw_to_image_u8 (const core::SuryaTensor &tensor,
                         size_t batch_idx = 0U,
                         float mul_scale = 255.0F);

} /* namespace surya::inline v1::nn */

#endif /* SURYA_NN_TENSOR_BRIDGE_HPP */
