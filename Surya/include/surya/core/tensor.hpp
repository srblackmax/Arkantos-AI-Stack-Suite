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

#ifndef SURYA_CORE_TENSOR_HPP
#define SURYA_CORE_TENSOR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include <surya/core/config.hpp>
#include <surya/core/memory.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Standard-library-only N-dimensional float32 tensor container backed
 *        by 64-byte aligned pooled memory.  Used as the public ABI tensor
 *        representation across surya::nn, surya::classical, and surya::gpu
 *        without exposing third-party or CUDA headers.
 * @note Thread-safety: Distinct instances may be accessed concurrently;
 *       const methods are safe for concurrent reads.
 */
class SURYA_API SuryaTensor
{
public:
  SuryaTensor (void) noexcept = default;

  /**
   * @brief Allocate a contiguous float32 tensor of shape DIMS.
   * @pre None.
   * @post numel() equals product of DIMS; data() is 64-byte aligned.
   * @note Thread-safety: Lock-free pool acquisition.
   */
  explicit SuryaTensor (std::vector<size_t> dims, float init_val = 0.0F)
    : shape_ (std::move (dims))
  {
    size_t count = shape_.empty () ? 0U : 1U;
    for (size_t d : shape_)
      {
        count *= d;
      }
    numel_ = count;
    if (numel_ > 0U)
      {
        alloc_ = Aligned_Buffer_Pool::global_pool ().acquire (
          numel_ * sizeof (float));
        float *p = data ();
        if (p != nullptr)
          {
            std::fill (p, p + numel_, init_val);
          }
      }
  }

  ~SuryaTensor (void) noexcept
  {
    if (alloc_.ptr != nullptr)
      {
        Aligned_Buffer_Pool::global_pool ().release (alloc_);
        alloc_ = {};
      }
  }

  SuryaTensor (const SuryaTensor &other)
    : shape_ (other.shape_), numel_ (other.numel_)
  {
    if (numel_ > 0U && other.data () != nullptr)
      {
        alloc_ = Aligned_Buffer_Pool::global_pool ().acquire (
          numel_ * sizeof (float));
        if (alloc_.ptr != nullptr)
          {
            std::copy (other.data (), other.data () + numel_, data ());
          }
      }
  }

  SuryaTensor &
  operator= (const SuryaTensor &other)
  {
    if (this != &other)
      {
        SuryaTensor tmp (other);
        *this = std::move (tmp);
      }
    return *this;
  }

  SuryaTensor (SuryaTensor &&other) noexcept
    : shape_ (std::move (other.shape_)),
      numel_ (other.numel_),
      alloc_ (other.alloc_)
  {
    other.numel_ = 0U;
    other.alloc_ = {};
  }

  SuryaTensor &
  operator= (SuryaTensor &&other) noexcept
  {
    if (this != &other)
      {
        if (alloc_.ptr != nullptr)
          {
            Aligned_Buffer_Pool::global_pool ().release (alloc_);
          }
        shape_ = std::move (other.shape_);
        numel_ = other.numel_;
        alloc_ = other.alloc_;
        other.numel_ = 0U;
        other.alloc_ = {};
      }
    return *this;
  }

  void
  fill_ (float val) noexcept
  {
    float *p = data ();
    if (p != nullptr)
      {
        std::fill (p, p + numel_, val);
      }
  }

  size_t
  ndim (void) const noexcept
  {
    return shape_.size ();
  }

  size_t
  size (size_t dim) const noexcept
  {
    return dim < shape_.size () ? shape_[dim] : 0U;
  }

  size_t
  numel (void) const noexcept
  {
    return numel_;
  }

  bool
  empty (void) const noexcept
  {
    return numel_ == 0U || alloc_.ptr == nullptr;
  }

  const std::vector<size_t> &
  shape (void) const noexcept
  {
    return shape_;
  }

  float *
  data (void) noexcept
  {
    return static_cast<float *> (alloc_.ptr);
  }

  const float *
  data (void) const noexcept
  {
    return static_cast<const float *> (alloc_.ptr);
  }

  std::span<float>
  span (void) noexcept
  {
    return {data (), numel_};
  }

  std::span<const float>
  span (void) const noexcept
  {
    return {data (), numel_};
  }

  float &
  at_2d (size_t r, size_t c) noexcept
  {
    const size_t cols = shape_.size () >= 2U ? shape_[1] : 1U;
    return data ()[r * cols + c];
  }

  float
  at_2d (size_t r, size_t c) const noexcept
  {
    const size_t cols = shape_.size () >= 2U ? shape_[1] : 1U;
    return data ()[r * cols + c];
  }

  /**
   * @brief Check elementwise approximate equality within RTOL and ATOL.
   * @pre None.
   * @post Returns true iff shapes match and |a - b| <= atol + rtol * |b|.
   * @returns True if all elements are within tolerance.
   * @note Thread-safety: Const read-only operation.
   */
  bool
  allclose (const SuryaTensor &other,
            float rtol = 1e-4F,
            float atol = 1e-5F) const noexcept
  {
    if (shape_ != other.shape_ || numel_ != other.numel_)
      {
        return false;
      }
    const float *pa = data ();
    const float *pb = other.data ();
    for (size_t i = 0U; i < numel_; ++i)
      {
        const float diff = std::fabs (pa[i] - pb[i]);
        if (diff > atol + rtol * std::fabs (pb[i]))
          {
            return false;
          }
      }
    return true;
  }

private:
  std::vector<size_t> shape_;
  size_t numel_ = 0U;
  Aligned_Buffer_Pool::Allocation alloc_ {};
};

} /* namespace surya::inline v1::core */

namespace surya::inline v1
{
using SuryaTensor = core::SuryaTensor;
} /* namespace surya::inline v1 */

#endif /* SURYA_CORE_TENSOR_HPP */
