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

#include <surya/core/image.hpp>

namespace surya::inline v1::core
{

Dyn_Image::Dyn_Image (size_t width_px,
                      size_t height_px,
                      size_t num_channels,
                      Pixel_Depth depth,
                      Channel_Layout layout,
                      bool zero_init) noexcept
{
  if (width_px == 0U || height_px == 0U || num_channels == 0U
      || num_channels > 4U)
    {
      return;
    }

  const size_t elem_bytes = pixel_depth_bytes (depth);
  size_t total_bytes = 0U;
  if (layout == Channel_Layout::Interleaved)
    {
      const size_t raw_row = width_px * num_channels * elem_bytes;
      row_stride_bytes_ =
        compute_aligned_pitch (raw_row, SURYA_CACHELINE_BYTES);
      plane_stride_bytes_ = height_px * row_stride_bytes_;
      total_bytes = plane_stride_bytes_;
    }
  else
    {
      const size_t raw_row = width_px * elem_bytes;
      row_stride_bytes_ =
        compute_aligned_pitch (raw_row, SURYA_CACHELINE_BYTES);
      plane_stride_bytes_ = height_px * row_stride_bytes_;
      total_bytes = num_channels * plane_stride_bytes_;
    }

  alloc_ = Aligned_Buffer_Pool::global_pool ().acquire (total_bytes);
  if (alloc_.ptr == nullptr)
    {
      row_stride_bytes_ = 0U;
      plane_stride_bytes_ = 0U;
      return;
    }

  width_ = width_px;
  height_ = height_px;
  channels_ = num_channels;
  depth_ = depth;
  layout_ = layout;
  if (zero_init)
    {
      std::memset (alloc_.ptr, 0, total_bytes);
    }
}

Dyn_Image::~Dyn_Image (void) noexcept
{
  if (alloc_.ptr != nullptr)
    {
      Aligned_Buffer_Pool::global_pool ().release (alloc_);
      alloc_ = {};
    }
}

Dyn_Image::Dyn_Image (Dyn_Image &&other) noexcept
  : alloc_ (other.alloc_),
    width_ (other.width_),
    height_ (other.height_),
    channels_ (other.channels_),
    depth_ (other.depth_),
    layout_ (other.layout_),
    row_stride_bytes_ (other.row_stride_bytes_),
    plane_stride_bytes_ (other.plane_stride_bytes_)
{
  other.alloc_ = {};
  other.width_ = 0U;
  other.height_ = 0U;
  other.channels_ = 0U;
  other.row_stride_bytes_ = 0U;
  other.plane_stride_bytes_ = 0U;
}

Dyn_Image &
Dyn_Image::operator= (Dyn_Image &&other) noexcept
{
  if (this != &other)
    {
      if (alloc_.ptr != nullptr)
        {
          Aligned_Buffer_Pool::global_pool ().release (alloc_);
        }
      alloc_ = other.alloc_;
      width_ = other.width_;
      height_ = other.height_;
      channels_ = other.channels_;
      depth_ = other.depth_;
      layout_ = other.layout_;
      row_stride_bytes_ = other.row_stride_bytes_;
      plane_stride_bytes_ = other.plane_stride_bytes_;

      other.alloc_ = {};
      other.width_ = 0U;
      other.height_ = 0U;
      other.channels_ = 0U;
      other.row_stride_bytes_ = 0U;
      other.plane_stride_bytes_ = 0U;
    }
  return *this;
}

Result<Dyn_Image>
Dyn_Image::create (size_t width_px,
                   size_t height_px,
                   size_t num_channels,
                   Pixel_Depth depth,
                   Channel_Layout layout,
                   bool zero_init)
{
  if (width_px == 0U || height_px == 0U || num_channels == 0U
      || num_channels > 4U)
    {
      return make_error (Error_Code::Invalid_Argument,
                         "Invalid Dyn_Image dimensions or channel count");
    }
  Dyn_Image img (width_px, height_px, num_channels, depth, layout, zero_init);
  if (img.empty ())
    {
      return make_error (Error_Code::Allocation_Failed,
                         "Failed to allocate Dyn_Image storage");
    }
  return img;
}

Dyn_Image
Dyn_Image::clone (void) const noexcept
{
  if (empty ())
    {
      return Dyn_Image ();
    }
  Dyn_Image copy (width_, height_, channels_, depth_, layout_, false);
  if (!copy.empty ())
    {
      const size_t total_bytes =
        (layout_ == Channel_Layout::Interleaved)
          ? (height_ * row_stride_bytes_)
          : (channels_ * plane_stride_bytes_);
      std::memcpy (copy.raw_data (), raw_data (), total_bytes);
    }
  return copy;
}

} /* namespace surya::inline v1::core */
