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

#ifndef SURYA_NN_RUNTIME_HPP
#define SURYA_NN_RUNTIME_HPP

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <surya/core.hpp>
#include <surya/nn/tensor_bridge.hpp>
#include <surya/nn/vision_blocks.hpp>

namespace surya::inline v1::nn
{

/**
 * @brief Production vision model inference and serialization runtime built on
 *        Basalt (graph execution) and Obsidian (zero-copy mmap SafeTensors).
 * @note Thread-safety: infer_* methods are safe for concurrent calls once the
 *       model is loaded.
 */
class SURYA_API Vision_Runtime
{
public:
  Vision_Runtime (
    std::shared_ptr<Vision_Module> model,
    Normalize_Params norm = Normalize_Params::unit_scale (),
    bool swap_rb = false);

  std::shared_ptr<Vision_Module>
  model (void) const noexcept
  {
    return model_;
  }

  /**
   * @brief Save all parameters and buffers of the wrapped vision model to a
   *        Hugging Face .safetensors file via Obsidian.
   * @pre model() != nullptr.
   * @post Writes .safetensors file at PATH.
   * @returns std::expected<void, SuryaError>.
   * @note Thread-safety: Safe to call concurrently with read-only inference.
   */
  core::Result<void>
  save_safetensors (
    const std::string &path,
    const std::map<std::string, std::string> &metadata = {}) const;

  /**
   * @brief Memory-map and load all model parameters and buffers from a
   *        .safetensors file via Obsidian.
   * @pre model() != nullptr.
   * @post Model parameters are populated from PATH and set to eval() mode.
   * @returns std::expected<void, SuryaError>.
   * @note Thread-safety: Mutates model weights; synchronize with callers.
   */
  core::Result<void>
  load_safetensors (const std::string &path, bool strict = true);

  /**
   * @brief Execute a no-grad forward pass on an NCHW SuryaTensor.
   * @pre model() != nullptr && !input_nchw.empty().
   * @post Returns output SuryaTensor.
   * @returns std::expected<core::SuryaTensor, SuryaError>.
   * @note Thread-safety: Safe for concurrent calls in eval mode.
   */
  core::Result<core::SuryaTensor>
  infer_tensor (const core::SuryaTensor &input_nchw) const;

  /**
   * @brief Convert an HWC uint8 image to NCHW and run inference in one call.
   * @pre !image.empty().
   * @post Returns output SuryaTensor.
   * @returns std::expected<core::SuryaTensor, SuryaError>.
   * @note Thread-safety: Safe for concurrent calls.
   */
  core::Result<core::SuryaTensor>
  infer_image (core::Const_View_U8C3 image) const;

  /**
   * @brief Convert a batch of equal-sized HWC uint8 images to NCHW in parallel
   *        and run batched inference.
   * @pre !batch.empty().
   * @post Returns batched output SuryaTensor.
   * @returns std::expected<core::SuryaTensor, SuryaError>.
   * @note Thread-safety: Safe for concurrent calls.
   */
  core::Result<core::SuryaTensor>
  infer_batch (std::span<const core::Const_View_U8C3> batch) const;

  /**
   * @brief Inspect tensor names present in a .safetensors checkpoint on disk
   *        using Obsidian's zero-copy mmap SafeTensors_File.
   * @pre PATH refers to a valid .safetensors file.
   * @post Returns vector of tensor key names.
   * @returns std::expected<std::vector<std::string>, SuryaError>.
   * @note Thread-safety: Safe to call concurrently.
   */
  static core::Result<std::vector<std::string>>
  inspect_checkpoint_tensors (const std::string &path);

private:
  std::shared_ptr<Vision_Module> model_;
  Normalize_Params norm_;
  bool swap_rb_ = false;
};

} /* namespace surya::inline v1::nn */

#endif /* SURYA_NN_RUNTIME_HPP */
