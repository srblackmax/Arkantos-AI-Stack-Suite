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

#ifndef SURYA_CORE_IMAGE_HPP
#define SURYA_CORE_IMAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <surya/core/config.hpp>
#include <surya/core/error.hpp>
#include <surya/core/memory.hpp>
#include <surya/core/types.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Lightweight, non-owning 2D/3D strided image view with compile-time
 *        scalar type T, channel count C, and layout LAYOUT.
 *
 * Passing Image_View by value is O(1) (40 bytes) with zero atomic refcounting.
 */
template <pixel_scalar T,
          size_t C = 1U,
          Channel_Layout Layout = Channel_Layout::Interleaved>
class Image_View
{
  static_assert (C >= 1U && C <= 4U,
                 "Image_View channel count C must be in [1, 4].");

public:
  using value_type = T;
  using bare_type = std::remove_cv_t<T>;
  using byte_ptr =
    std::conditional_t<std::is_const_v<T>, const uint8_t *, uint8_t *>;

  static constexpr size_t static_channels = C;
  static constexpr Channel_Layout static_layout = Layout;

  constexpr Image_View (void) noexcept = default;

  constexpr Image_View (T *data_ptr,
                        size_t width_px,
                        size_t height_px,
                        size_t row_stride,
                        size_t plane_stride = 0U) noexcept
    : data_ (data_ptr),
      width_ (width_px),
      height_ (height_px),
      row_stride_bytes_ (row_stride),
      plane_stride_bytes_ (
        plane_stride > 0U ? plane_stride : (height_px * row_stride))
  {
  }

  /* Allow implicit conversion from mutable Image_View to const Image_View.  */
  template <typename U = T,
            typename = std::enable_if_t<std::is_const_v<U>>>
  constexpr Image_View (
    const Image_View<std::remove_const_t<U>, C, Layout> &other) noexcept
    : data_ (other.data ()),
      width_ (other.width ()),
      height_ (other.height ()),
      row_stride_bytes_ (other.row_stride_bytes ()),
      plane_stride_bytes_ (other.plane_stride_bytes ())
  {
  }

  constexpr bool
  empty (void) const noexcept
  {
    return data_ == nullptr || width_ == 0U || height_ == 0U;
  }

  constexpr size_t
  width (void) const noexcept
  {
    return width_;
  }

  constexpr size_t
  height (void) const noexcept
  {
    return height_;
  }

  constexpr Size2u
  size (void) const noexcept
  {
    return Size2u (width_, height_);
  }

  static constexpr size_t
  channels (void) noexcept
  {
    return C;
  }

  static constexpr Channel_Layout
  layout (void) noexcept
  {
    return Layout;
  }

  constexpr size_t
  row_stride_bytes (void) const noexcept
  {
    return row_stride_bytes_;
  }

  constexpr size_t
  plane_stride_bytes (void) const noexcept
  {
    return plane_stride_bytes_;
  }

  constexpr T *
  data (void) const noexcept
  {
    return data_;
  }

  /**
   * @brief Return true if every row starts on an ALIGNMENT-byte boundary.
   */
  bool
  is_row_aligned (size_t alignment = SURYA_CACHELINE_BYTES) const noexcept
  {
    if (empty ())
      {
        return false;
      }
    const uintptr_t addr = reinterpret_cast<uintptr_t> (data_);
    return (addr % alignment == 0U) && (row_stride_bytes_ % alignment == 0U);
  }

  /**
   * @brief Return true if rows are tightly packed without row padding.
   */
  constexpr bool
  is_contiguous (void) const noexcept
  {
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        return row_stride_bytes_ == width_ * C * sizeof (bare_type);
      }
    else
      {
        return row_stride_bytes_ == width_ * sizeof (bare_type)
               && plane_stride_bytes_ == height_ * row_stride_bytes_;
      }
  }

  /**
   * @brief Return pointer to the start of scanline Y (for Interleaved layout
   *        or channel 0 of Planar layout).
   * @pre Y < height().
   * @complexity O(1) time.
   */
  SURYA_INLINE T *
  row_ptr (size_t y) const noexcept
  {
    SURYA_ASSERT (y < height_);
    byte_ptr base = reinterpret_cast<byte_ptr> (data_);
    return reinterpret_cast<T *> (base + y * row_stride_bytes_);
  }

  /**
   * @brief Return pointer to scanline Y of channel CH in Planar layout.
   * @pre Y < height() and CH < C.
   * @complexity O(1) time.
   */
  SURYA_INLINE T *
  plane_row_ptr (size_t ch, size_t y) const noexcept
  {
    SURYA_ASSERT (ch < C && y < height_);
    byte_ptr base = reinterpret_cast<byte_ptr> (data_);
    return reinterpret_cast<T *> (
      base + ch * plane_stride_bytes_ + y * row_stride_bytes_);
  }

  /**
   * @brief Return pointer to the start of channel plane CH in Planar layout.
   * @pre CH < C.
   * @complexity O(1) time.
   */
  SURYA_INLINE T *
  plane_ptr (size_t ch) const noexcept
  {
    SURYA_ASSERT (ch < C);
    byte_ptr base = reinterpret_cast<byte_ptr> (data_);
    return reinterpret_cast<T *> (base + ch * plane_stride_bytes_);
  }

  /**
   * @brief Access element at row Y, column X, channel CH.
   * @pre Y < height(), X < width(), CH < C.
   * @complexity O(1) time.
   */
  SURYA_INLINE T &
  at (size_t y, size_t x, size_t ch = 0U) const noexcept
  {
    SURYA_ASSERT (y < height_ && x < width_ && ch < C);
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        return row_ptr (y)[x * C + ch];
      }
    else
      {
        return plane_row_ptr (ch, y)[x];
      }
  }

  /**
   * @brief Extract a zero-copy rectangular region-of-interest (ROI) subview.
   * @pre ROI lies within [0, width()] x [0, height()].
   * @complexity O(1) time and space.
   */
  Result<Image_View>
  subview (Rectu roi) const noexcept
  {
    if (empty ())
      {
        return make_error (Error_Code::Empty_Input,
                           "Cannot take subview of an empty Image_View");
      }
    if (roi.empty () || roi.x + roi.width > width_
        || roi.y + roi.height > height_)
      {
        return make_error (Error_Code::Out_Of_Bounds,
                           "Requested ROI exceeds image dimensions");
      }

    byte_ptr base = reinterpret_cast<byte_ptr> (data_);
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        T *sub_ptr = reinterpret_cast<T *> (
          base + roi.y * row_stride_bytes_ + roi.x * C * sizeof (bare_type));
        return Image_View (sub_ptr, roi.width, roi.height, row_stride_bytes_,
                           plane_stride_bytes_);
      }
    else
      {
        T *sub_ptr = reinterpret_cast<T *> (
          base + roi.y * row_stride_bytes_ + roi.x * sizeof (bare_type));
        return Image_View (sub_ptr, roi.width, roi.height, row_stride_bytes_,
                           plane_stride_bytes_);
      }
  }

private:
  T *data_ = nullptr;
  size_t width_ = 0U;
  size_t height_ = 0U;
  size_t row_stride_bytes_ = 0U;
  size_t plane_stride_bytes_ = 0U;
};

/**
 * @brief Exclusively owning 2D/3D image container with guaranteed 64-byte
 *        cache-line row-pitch alignment and pooled buffer recycling.
 *
 * Move-only by default to prevent accidental deep copies or hidden aliasing;
 * call .clone() for an explicit deep copy or .view() for a zero-cost view.
 */
template <pixel_scalar T,
          size_t C = 1U,
          Channel_Layout Layout = Channel_Layout::Interleaved>
class Image
{
  static_assert (!std::is_const_v<T>,
                 "Owning Image element type T must not be const.");
  static_assert (C >= 1U && C <= 4U,
                 "Image channel count C must be in [1, 4].");

public:
  using value_type = T;
  using view_type = Image_View<T, C, Layout>;
  using const_view_type = Image_View<const T, C, Layout>;

  static constexpr size_t static_channels = C;
  static constexpr Channel_Layout static_layout = Layout;

  Image (void) noexcept = default;

  /**
   * @brief Allocate a 64-byte row-aligned image of dimensions WIDTH x HEIGHT.
   */
  Image (size_t width_px, size_t height_px, bool zero_init = false) noexcept
  {
    allocate_storage (width_px, height_px, zero_init);
  }

  /**
   * @brief Allocate an image and fill every pixel with FILL_VAL.
   */
  Image (size_t width_px, size_t height_px, Scalar<T, C> fill_val) noexcept
  {
    if (allocate_storage (width_px, height_px, false))
      {
        fill (fill_val);
      }
  }

  ~Image (void) noexcept
  {
    release_storage ();
  }

  Image (const Image &) = delete;
  Image &operator= (const Image &) = delete;

  Image (Image &&other) noexcept
    : alloc_ (other.alloc_),
      width_ (other.width_),
      height_ (other.height_),
      row_stride_bytes_ (other.row_stride_bytes_),
      plane_stride_bytes_ (other.plane_stride_bytes_)
  {
    other.alloc_ = {};
    other.width_ = 0U;
    other.height_ = 0U;
    other.row_stride_bytes_ = 0U;
    other.plane_stride_bytes_ = 0U;
  }

  Image &
  operator= (Image &&other) noexcept
  {
    if (this != &other)
      {
        release_storage ();
        alloc_ = other.alloc_;
        width_ = other.width_;
        height_ = other.height_;
        row_stride_bytes_ = other.row_stride_bytes_;
        plane_stride_bytes_ = other.plane_stride_bytes_;
        other.alloc_ = {};
        other.width_ = 0U;
        other.height_ = 0U;
        other.row_stride_bytes_ = 0U;
        other.plane_stride_bytes_ = 0U;
      }
    return *this;
  }

  /**
   * @brief Fallible factory method returning Result<Image>.
   */
  static Result<Image>
  create (size_t width_px, size_t height_px, bool zero_init = false)
  {
    if (width_px == 0U || height_px == 0U)
      {
        return make_error (Error_Code::Invalid_Argument,
                           "Image::create dimensions must be non-zero");
      }
    Image img (width_px, height_px, zero_init);
    if (img.empty ())
      {
        return make_error (Error_Code::Allocation_Failed,
                           "Failed to allocate 64-byte aligned image buffer");
      }
    return img;
  }

  /**
   * @brief Perform an explicit deep copy of this image.
   * @pre None.
   * @complexity O(height * row_stride_bytes) time.
   */
  Image
  clone (void) const noexcept
  {
    if (empty ())
      {
        return Image ();
      }
    Image copy (width_, height_, false);
    if (!copy.empty ())
      {
        const size_t total_bytes =
          (Layout == Channel_Layout::Interleaved)
            ? (height_ * row_stride_bytes_)
            : (C * plane_stride_bytes_);
        std::memcpy (copy.data (), data (), total_bytes);
      }
    return copy;
  }

  /**
   * @brief Create an owning Image by copying from an arbitrary Image_View
   *        (including unaligned or ROI subviews).
   */
  static Image
  from_view (const_view_type src) noexcept
  {
    if (src.empty ())
      {
        return Image ();
      }
    Image dst (src.width (), src.height (), false);
    if (dst.empty ())
      {
        return dst;
      }
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        const size_t row_bytes = src.width () * C * sizeof (T);
        for (size_t y = 0U; y < src.height (); ++y)
          {
            std::memcpy (dst.row_ptr (y), src.row_ptr (y), row_bytes);
          }
      }
    else
      {
        const size_t row_bytes = src.width () * sizeof (T);
        for (size_t ch = 0U; ch < C; ++ch)
          {
            for (size_t y = 0U; y < src.height (); ++y)
              {
                std::memcpy (dst.plane_row_ptr (ch, y),
                             src.plane_row_ptr (ch, y), row_bytes);
              }
          }
      }
    return dst;
  }

  bool
  empty (void) const noexcept
  {
    return alloc_.ptr == nullptr || width_ == 0U || height_ == 0U;
  }

  size_t
  width (void) const noexcept
  {
    return width_;
  }

  size_t
  height (void) const noexcept
  {
    return height_;
  }

  Size2u
  size (void) const noexcept
  {
    return Size2u (width_, height_);
  }

  static constexpr size_t
  channels (void) noexcept
  {
    return C;
  }

  static constexpr Channel_Layout
  layout (void) noexcept
  {
    return Layout;
  }

  size_t
  row_stride_bytes (void) const noexcept
  {
    return row_stride_bytes_;
  }

  size_t
  plane_stride_bytes (void) const noexcept
  {
    return plane_stride_bytes_;
  }

  T *
  data (void) noexcept
  {
    return static_cast<T *> (alloc_.ptr);
  }

  const T *
  data (void) const noexcept
  {
    return static_cast<const T *> (alloc_.ptr);
  }

  bool
  is_row_aligned (size_t alignment = SURYA_CACHELINE_BYTES) const noexcept
  {
    return const_view ().is_row_aligned (alignment);
  }

  SURYA_INLINE T *
  row_ptr (size_t y) noexcept
  {
    return view ().row_ptr (y);
  }

  SURYA_INLINE const T *
  row_ptr (size_t y) const noexcept
  {
    return const_view ().row_ptr (y);
  }

  SURYA_INLINE T *
  plane_row_ptr (size_t ch, size_t y) noexcept
  {
    return view ().plane_row_ptr (ch, y);
  }

  SURYA_INLINE const T *
  plane_row_ptr (size_t ch, size_t y) const noexcept
  {
    return const_view ().plane_row_ptr (ch, y);
  }

  SURYA_INLINE T *
  plane_ptr (size_t ch) noexcept
  {
    return view ().plane_ptr (ch);
  }

  SURYA_INLINE const T *
  plane_ptr (size_t ch) const noexcept
  {
    return const_view ().plane_ptr (ch);
  }

  SURYA_INLINE T &
  at (size_t y, size_t x, size_t ch = 0U) noexcept
  {
    return view ().at (y, x, ch);
  }

  SURYA_INLINE const T &
  at (size_t y, size_t x, size_t ch = 0U) const noexcept
  {
    return const_view ().at (y, x, ch);
  }

  view_type
  view (void) noexcept
  {
    return view_type (data (), width_, height_, row_stride_bytes_,
                      plane_stride_bytes_);
  }

  const_view_type
  view (void) const noexcept
  {
    return const_view ();
  }

  const_view_type
  const_view (void) const noexcept
  {
    return const_view_type (data (), width_, height_, row_stride_bytes_,
                            plane_stride_bytes_);
  }

  operator view_type (void) noexcept
  {
    return view ();
  }

  operator const_view_type (void) const noexcept
  {
    return const_view ();
  }

  Result<view_type>
  subview (Rectu roi) noexcept
  {
    return view ().subview (roi);
  }

  Result<const_view_type>
  subview (Rectu roi) const noexcept
  {
    return const_view ().subview (roi);
  }

  /**
   * @brief Zero-initialize the entire allocated image buffer (including row
   *        padding bytes for deterministic SIMD tail loads).
   */
  void
  zero (void) noexcept
  {
    if (empty ())
      {
        return;
      }
    const size_t total_bytes =
      (Layout == Channel_Layout::Interleaved)
        ? (height_ * row_stride_bytes_)
        : (C * plane_stride_bytes_);
    std::memset (alloc_.ptr, 0, total_bytes);
  }

  /**
   * @brief Fill all channels of all pixels with a uniform scalar V.
   */
  void
  fill (T v) noexcept
  {
    fill (Scalar<T, C> (v));
  }

  /**
   * @brief Fill each pixel with multi-channel tuple COLOR.
   */
  void
  fill (Scalar<T, C> color) noexcept
  {
    if (empty ())
      {
        return;
      }
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        for (size_t y = 0U; y < height_; ++y)
          {
            T *row = row_ptr (y);
            for (size_t x = 0U; x < width_; ++x)
              {
                for (size_t ch = 0U; ch < C; ++ch)
                  {
                    row[x * C + ch] = color[ch];
                  }
              }
          }
      }
    else
      {
        for (size_t ch = 0U; ch < C; ++ch)
          {
            const T val = color[ch];
            for (size_t y = 0U; y < height_; ++y)
              {
                T *row = plane_row_ptr (ch, y);
                std::fill (row, row + width_, val);
              }
          }
      }
  }

private:
  bool
  allocate_storage (size_t width_px, size_t height_px, bool zero_init) noexcept
  {
    if (width_px == 0U || height_px == 0U)
      {
        return false;
      }
    size_t total_bytes = 0U;
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        const size_t raw_row = width_px * C * sizeof (T);
        row_stride_bytes_ =
          compute_aligned_pitch (raw_row, SURYA_CACHELINE_BYTES);
        plane_stride_bytes_ = height_px * row_stride_bytes_;
        total_bytes = plane_stride_bytes_;
      }
    else
      {
        const size_t raw_row = width_px * sizeof (T);
        row_stride_bytes_ =
          compute_aligned_pitch (raw_row, SURYA_CACHELINE_BYTES);
        plane_stride_bytes_ = height_px * row_stride_bytes_;
        total_bytes = C * plane_stride_bytes_;
      }

    alloc_ = Aligned_Buffer_Pool::global_pool ().acquire (total_bytes);
    if (alloc_.ptr == nullptr)
      {
        row_stride_bytes_ = 0U;
        plane_stride_bytes_ = 0U;
        return false;
      }
    width_ = width_px;
    height_ = height_px;
    if (zero_init)
      {
        std::memset (alloc_.ptr, 0, total_bytes);
      }
    return true;
  }

  void
  release_storage (void) noexcept
  {
    if (alloc_.ptr != nullptr)
      {
        Aligned_Buffer_Pool::global_pool ().release (alloc_);
        alloc_ = {};
      }
  }

  Aligned_Buffer_Pool::Allocation alloc_ {};
  size_t width_ = 0U;
  size_t height_ = 0U;
  size_t row_stride_bytes_ = 0U;
  size_t plane_stride_bytes_ = 0U;
};

/* Standard typed aliases for common computer vision formats.  */
using Image_U8C1 = Image<uint8_t, 1U, Channel_Layout::Interleaved>;
using Image_U8C2 = Image<uint8_t, 2U, Channel_Layout::Interleaved>;
using Image_U8C3 = Image<uint8_t, 3U, Channel_Layout::Interleaved>;
using Image_U8C4 = Image<uint8_t, 4U, Channel_Layout::Interleaved>;

using Image_U16C1 = Image<uint16_t, 1U, Channel_Layout::Interleaved>;
using Image_U16C3 = Image<uint16_t, 3U, Channel_Layout::Interleaved>;

using Image_I16C1 = Image<int16_t, 1U, Channel_Layout::Interleaved>;
using Image_I32C1 = Image<int32_t, 1U, Channel_Layout::Interleaved>;

using Image_F32C1 = Image<float, 1U, Channel_Layout::Interleaved>;
using Image_F32C2 = Image<float, 2U, Channel_Layout::Interleaved>;
using Image_F32C3 = Image<float, 3U, Channel_Layout::Interleaved>;
using Image_F32C4 = Image<float, 4U, Channel_Layout::Interleaved>;

using Image_F32C3_Planar = Image<float, 3U, Channel_Layout::Planar>;

using View_U8C1 = Image_View<uint8_t, 1U, Channel_Layout::Interleaved>;
using View_U8C3 = Image_View<uint8_t, 3U, Channel_Layout::Interleaved>;
using View_U8C4 = Image_View<uint8_t, 4U, Channel_Layout::Interleaved>;
using View_F32C1 = Image_View<float, 1U, Channel_Layout::Interleaved>;
using View_F32C3 = Image_View<float, 3U, Channel_Layout::Interleaved>;

using Const_View_U8C1 =
  Image_View<const uint8_t, 1U, Channel_Layout::Interleaved>;
using Const_View_U8C3 =
  Image_View<const uint8_t, 3U, Channel_Layout::Interleaved>;
using Const_View_U8C4 =
  Image_View<const uint8_t, 4U, Channel_Layout::Interleaved>;
using Const_View_F32C1 =
  Image_View<const float, 1U, Channel_Layout::Interleaved>;
using Const_View_F32C3 =
  Image_View<const float, 3U, Channel_Layout::Interleaved>;

/**
 * @brief Runtime-typed owning image container for file I/O boundaries where
 *        pixel depth and channel count are discovered from codec headers.
 *
 * Converts to compile-time typed Image_View<T, C, Layout> in O(1) time via
 * try_view<T, C, Layout>().
 */
class SURYA_API Dyn_Image
{
public:
  Dyn_Image (void) noexcept = default;

  Dyn_Image (size_t width_px,
             size_t height_px,
             size_t num_channels,
             Pixel_Depth depth = Pixel_Depth::U8,
             Channel_Layout layout = Channel_Layout::Interleaved,
             bool zero_init = false) noexcept;

  ~Dyn_Image (void) noexcept;

  Dyn_Image (const Dyn_Image &) = delete;
  Dyn_Image &operator= (const Dyn_Image &) = delete;

  Dyn_Image (Dyn_Image &&other) noexcept;
  Dyn_Image &operator= (Dyn_Image &&other) noexcept;

  static Result<Dyn_Image>
  create (size_t width_px,
          size_t height_px,
          size_t num_channels,
          Pixel_Depth depth = Pixel_Depth::U8,
          Channel_Layout layout = Channel_Layout::Interleaved,
          bool zero_init = false);

  Dyn_Image
  clone (void) const noexcept;

  bool
  empty (void) const noexcept
  {
    return alloc_.ptr == nullptr || width_ == 0U || height_ == 0U;
  }

  size_t
  width (void) const noexcept
  {
    return width_;
  }

  size_t
  height (void) const noexcept
  {
    return height_;
  }

  size_t
  channels (void) const noexcept
  {
    return channels_;
  }

  Pixel_Depth
  depth (void) const noexcept
  {
    return depth_;
  }

  Channel_Layout
  layout (void) const noexcept
  {
    return layout_;
  }

  size_t
  row_stride_bytes (void) const noexcept
  {
    return row_stride_bytes_;
  }

  size_t
  plane_stride_bytes (void) const noexcept
  {
    return plane_stride_bytes_;
  }

  void *
  raw_data (void) noexcept
  {
    return alloc_.ptr;
  }

  const void *
  raw_data (void) const noexcept
  {
    return alloc_.ptr;
  }

  uint8_t *
  raw_row_ptr (size_t y) noexcept
  {
    SURYA_ASSERT (y < height_);
    return static_cast<uint8_t *> (alloc_.ptr) + y * row_stride_bytes_;
  }

  const uint8_t *
  raw_row_ptr (size_t y) const noexcept
  {
    SURYA_ASSERT (y < height_);
    return static_cast<const uint8_t *> (alloc_.ptr) + y * row_stride_bytes_;
  }

  /**
   * @brief Obtain a strongly-typed mutable Image_View if runtime metadata
   *        matches <T, C, Layout>.
   * @pre None.
   * @complexity O(1) time and space.
   */
  template <pixel_scalar T,
            size_t C = 1U,
            Channel_Layout Layout = Channel_Layout::Interleaved>
  Result<Image_View<T, C, Layout>>
  try_view (void) noexcept
  {
    using CleanT = std::remove_cv_t<T>;
    if (empty ())
      {
        return make_error (Error_Code::Empty_Input, "Dyn_Image is empty");
      }
    if (depth_ != Pixel_Traits<CleanT>::depth_enum () || channels_ != C
        || layout_ != Layout)
      {
        return make_error (
          Error_Code::Format_Mismatch,
          "Dyn_Image runtime format does not match requested Image_View");
      }
    return Image_View<T, C, Layout> (
      static_cast<T *> (alloc_.ptr), width_, height_, row_stride_bytes_,
      plane_stride_bytes_);
  }

  /**
   * @brief Obtain a strongly-typed const Image_View if runtime metadata
   *        matches <T, C, Layout>.
   */
  template <pixel_scalar T,
            size_t C = 1U,
            Channel_Layout Layout = Channel_Layout::Interleaved>
  Result<Image_View<const std::remove_cv_t<T>, C, Layout>>
  try_const_view (void) const noexcept
  {
    using CleanT = std::remove_cv_t<T>;
    if (empty ())
      {
        return make_error (Error_Code::Empty_Input, "Dyn_Image is empty");
      }
    if (depth_ != Pixel_Traits<CleanT>::depth_enum () || channels_ != C
        || layout_ != Layout)
      {
        return make_error (
          Error_Code::Format_Mismatch,
          "Dyn_Image runtime format does not match requested Image_View");
      }
    return Image_View<const CleanT, C, Layout> (
      static_cast<const CleanT *> (alloc_.ptr), width_, height_,
      row_stride_bytes_, plane_stride_bytes_);
  }

  /**
   * @brief Copy or convert a strongly-typed Image into a Dyn_Image.
   */
  template <pixel_scalar T, size_t C, Channel_Layout Layout>
  static Dyn_Image
  from_view (Image_View<const T, C, Layout> src) noexcept
  {
    if (src.empty ())
      {
        return Dyn_Image ();
      }
    Dyn_Image out (src.width (), src.height (), C,
                   Pixel_Traits<T>::depth_enum (), Layout, false);
    if (out.empty ())
      {
        return out;
      }
    auto v_res = out.try_view<T, C, Layout> ();
    if (!v_res.has_value ())
      {
        return Dyn_Image ();
      }
    auto dst_v = *v_res;
    if constexpr (Layout == Channel_Layout::Interleaved)
      {
        const size_t row_bytes = src.width () * C * sizeof (T);
        for (size_t y = 0U; y < src.height (); ++y)
          {
            std::memcpy (dst_v.row_ptr (y), src.row_ptr (y), row_bytes);
          }
      }
    else
      {
        const size_t row_bytes = src.width () * sizeof (T);
        for (size_t ch = 0U; ch < C; ++ch)
          {
            for (size_t y = 0U; y < src.height (); ++y)
              {
                std::memcpy (dst_v.plane_row_ptr (ch, y),
                             src.plane_row_ptr (ch, y), row_bytes);
              }
          }
      }
    return out;
  }

private:
  Aligned_Buffer_Pool::Allocation alloc_ {};
  size_t width_ = 0U;
  size_t height_ = 0U;
  size_t channels_ = 0U;
  Pixel_Depth depth_ = Pixel_Depth::U8;
  Channel_Layout layout_ = Channel_Layout::Interleaved;
  size_t row_stride_bytes_ = 0U;
  size_t plane_stride_bytes_ = 0U;
};

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_IMAGE_HPP */
