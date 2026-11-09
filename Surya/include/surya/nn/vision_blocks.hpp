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

#ifndef SURYA_NN_VISION_BLOCKS_HPP
#define SURYA_NN_VISION_BLOCKS_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <surya/core.hpp>

namespace surya::inline v1::nn
{

/**
 * @brief Pointwise activation function used in vision convolutional blocks.
 */
enum class Activation_Kind : uint8_t
{
  Identity = 0,
  ReLU = 1,
  SiLU = 2,
  GELU = 3
};

/**
 * @brief Abstract ABI-stable base class for Surya neural network modules.
 *        Encapsulates Basalt differentiable graph modules behind an opaque
 *        handle so public headers remain 100% standard-library only.
 * @note Thread-safety: Concurrent forward() calls are safe in eval() mode.
 */
class SURYA_API Vision_Module
{
public:
  virtual ~Vision_Module (void) noexcept = default;

  virtual core::SuryaTensor
  forward (const core::SuryaTensor &input) = 0;

  virtual void
  eval (void) noexcept = 0;

  virtual void
  fuse_for_inference (void) = 0;

  virtual void *
  native_module_ptr (void) noexcept = 0;

  virtual const void *
  native_module_ptr (void) const noexcept = 0;
};

/**
 * @brief Fundamental Conv2d + BatchNorm2d + Activation vision building block.
 *        Supports analytical Conv+BN folding via fuse_for_inference().
 */
class SURYA_API Conv_BN_Act : public Vision_Module
{
public:
  Conv_BN_Act (size_t in_channels,
               size_t out_channels,
               size_t kernel_size,
               size_t stride = 1U,
               size_t padding = 0U,
               Activation_Kind act = Activation_Kind::SiLU,
               double bn_eps = 1e-5);

  ~Conv_BN_Act (void) noexcept override;

  core::SuryaTensor
  forward (const core::SuryaTensor &input) override;

  void
  eval (void) noexcept override;

  void
  fuse_for_inference (void) override;

  bool
  is_fused (void) const noexcept;

  /**
   * @brief Randomize BatchNorm running statistics and affine weights for
   *        testing analytical Conv+BN folding equivalence.
   */
  void
  randomize_bn_parameters (void);

  void *
  native_module_ptr (void) noexcept override;

  const void *
  native_module_ptr (void) const noexcept override;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

/**
 * @brief Standard 2-conv residual bottleneck block with optional shortcut.
 */
class SURYA_API Residual_Bottleneck : public Vision_Module
{
public:
  Residual_Bottleneck (size_t in_channels,
                       size_t out_channels,
                       bool shortcut = true,
                       double expansion = 0.5,
                       Activation_Kind act = Activation_Kind::SiLU);

  ~Residual_Bottleneck (void) noexcept override;

  core::SuryaTensor
  forward (const core::SuryaTensor &input) override;

  void
  eval (void) noexcept override;

  void
  fuse_for_inference (void) override;

  void *
  native_module_ptr (void) noexcept override;

  const void *
  native_module_ptr (void) const noexcept override;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

/**
 * @brief Cross-Stage Partial (C2f) feature aggregation block.
 */
class SURYA_API CSP_C2f_Block : public Vision_Module
{
public:
  CSP_C2f_Block (size_t in_channels,
                 size_t out_channels,
                 size_t num_bottlenecks = 1U,
                 bool shortcut = true,
                 Activation_Kind act = Activation_Kind::SiLU);

  ~CSP_C2f_Block (void) noexcept override;

  core::SuryaTensor
  forward (const core::SuryaTensor &input) override;

  void
  eval (void) noexcept override;

  void
  fuse_for_inference (void) override;

  void *
  native_module_ptr (void) noexcept override;

  const void *
  native_module_ptr (void) const noexcept override;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

/**
 * @brief Spatial Pyramid Pooling - Fast (SPPF) multi-scale receptive field
 *        aggregation block.
 */
class SURYA_API SPPF_Block : public Vision_Module
{
public:
  SPPF_Block (size_t in_channels,
              size_t out_channels,
              size_t pool_kernel = 5U);

  ~SPPF_Block (void) noexcept override;

  core::SuryaTensor
  forward (const core::SuryaTensor &input) override;

  void
  eval (void) noexcept override;

  void
  fuse_for_inference (void) override;

  void *
  native_module_ptr (void) noexcept override;

  const void *
  native_module_ptr (void) const noexcept override;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

/**
 * @brief Complete end-to-end convolutional vision classifier built from
 *        Conv_BN_Act, CSP_C2f_Block, SPPF_Block, Adaptive_Avg_Pool2d, and
 *        Linear layers.
 */
class SURYA_API Vision_Classifier : public Vision_Module
{
public:
  Vision_Classifier (size_t in_channels,
                     size_t base_channels,
                     size_t num_classes);

  ~Vision_Classifier (void) noexcept override;

  core::SuryaTensor
  forward (const core::SuryaTensor &input) override;

  void
  eval (void) noexcept override;

  void
  fuse_for_inference (void) override;

  /**
   * @brief Zero the linear classification head weights (for checkpoint reload
   *        verification).
   */
  void
  zero_head_weights (void);

  void *
  native_module_ptr (void) noexcept override;

  const void *
  native_module_ptr (void) const noexcept override;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} /* namespace surya::inline v1::nn */

#endif /* SURYA_NN_VISION_BLOCKS_HPP */
