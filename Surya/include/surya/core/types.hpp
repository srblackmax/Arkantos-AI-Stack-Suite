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

#ifndef SURYA_CORE_TYPES_HPP
#define SURYA_CORE_TYPES_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <surya/core/config.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Memory arrangement of multi-channel image elements.
 *
 * Interleaved (HWC): channels of pixel (y, x) are contiguous in memory.
 * Planar (CHW): each channel c is stored as a contiguous 2D plane.
 */
enum class Channel_Layout : uint8_t
{
  Interleaved = 0,
  Planar = 1
};

/**
 * @brief Boundary extrapolation rules for spatial neighborhood operations.
 */
enum class Border_Mode : uint8_t
{
  Constant = 0,
  Clamp = 1,
  Reflect_101 = 2,
  Wrap = 3
};

/**
 * @brief Runtime scalar depth enumeration for dynamic image containers.
 */
enum class Pixel_Depth : uint8_t
{
  U8 = 0,
  U16 = 1,
  I16 = 2,
  I32 = 3,
  F32 = 4,
  F64 = 5
};

/**
 * @brief Return byte width of a Pixel_Depth element.
 */
constexpr size_t
pixel_depth_bytes (Pixel_Depth depth) noexcept
{
  switch (depth)
    {
    case Pixel_Depth::U8:
      return 1U;
    case Pixel_Depth::U16:
    case Pixel_Depth::I16:
      return 2U;
    case Pixel_Depth::I32:
    case Pixel_Depth::F32:
      return 4U;
    case Pixel_Depth::F64:
      return 8U;
    }
  return 1U;
}

/**
 * @brief Concept constraining valid scalar pixel element types.
 */
template <typename T>
concept pixel_scalar =
  std::is_same_v<std::remove_cv_t<T>, uint8_t>
  || std::is_same_v<std::remove_cv_t<T>, uint16_t>
  || std::is_same_v<std::remove_cv_t<T>, int16_t>
  || std::is_same_v<std::remove_cv_t<T>, int32_t>
  || std::is_same_v<std::remove_cv_t<T>, float>
  || std::is_same_v<std::remove_cv_t<T>, double>;

/**
 * @brief Compile-time traits for scalar pixel types.
 */
template <pixel_scalar T>
struct Pixel_Traits
{
  using value_type = std::remove_cv_t<T>;
  using work_type =
    std::conditional_t<std::is_floating_point_v<value_type>, double, int64_t>;

  static constexpr bool is_integer = std::is_integral_v<value_type>;
  static constexpr bool is_float = std::is_floating_point_v<value_type>;

  static constexpr value_type
  zero (void) noexcept
  {
    return static_cast<value_type> (0);
  }

  static constexpr value_type
  max_unit (void) noexcept
  {
    if constexpr (std::is_floating_point_v<value_type>)
      {
        return static_cast<value_type> (1.0);
      }
    else
      {
        return std::numeric_limits<value_type>::max ();
      }
  }

  static constexpr Pixel_Depth
  depth_enum (void) noexcept
  {
    if constexpr (std::is_same_v<value_type, uint8_t>)
      return Pixel_Depth::U8;
    else if constexpr (std::is_same_v<value_type, uint16_t>)
      return Pixel_Depth::U16;
    else if constexpr (std::is_same_v<value_type, int16_t>)
      return Pixel_Depth::I16;
    else if constexpr (std::is_same_v<value_type, int32_t>)
      return Pixel_Depth::I32;
    else if constexpr (std::is_same_v<value_type, float>)
      return Pixel_Depth::F32;
    else
      return Pixel_Depth::F64;
  }
};

/**
 * @brief Saturating numeric cast with nearest-integer rounding for float->int.
 * @pre None.
 * @complexity O(1) time and space.
 */
template <pixel_scalar DstT, typename SrcT>
constexpr DstT
saturate_cast (SrcT val) noexcept
{
  using CleanDst = std::remove_cv_t<DstT>;
  if constexpr (std::is_same_v<CleanDst, std::remove_cv_t<SrcT>>)
    {
      return val;
    }
  else if constexpr (std::is_floating_point_v<CleanDst>)
    {
      return static_cast<CleanDst> (val);
    }
  else if constexpr (std::is_floating_point_v<SrcT>)
    {
      const double lo =
        static_cast<double> (std::numeric_limits<CleanDst>::min ());
      const double hi =
        static_cast<double> (std::numeric_limits<CleanDst>::max ());
      const double rounded = std::nearbyint (static_cast<double> (val));
      const double clamped = std::clamp (rounded, lo, hi);
      return static_cast<CleanDst> (clamped);
    }
  else
    {
      const int64_t v = static_cast<int64_t> (val);
      const int64_t lo =
        static_cast<int64_t> (std::numeric_limits<CleanDst>::min ());
      const int64_t hi =
        static_cast<int64_t> (std::numeric_limits<CleanDst>::max ());
      return static_cast<CleanDst> (std::clamp (v, lo, hi));
    }
}

/**
 * @brief Map an out-of-range coordinate IDX into [0, LEN - 1] according to
 *        MODE, or return -1 when MODE is Border_Mode::Constant and IDX is
 *        outside [0, LEN - 1].
 * @pre LEN > 0.
 * @complexity O(1) time and space.
 */
constexpr ptrdiff_t
border_interpolate (ptrdiff_t idx, ptrdiff_t len, Border_Mode mode) noexcept
{
  if (idx >= 0 && idx < len)
    {
      return idx;
    }
  if (len <= 0)
    {
      return -1;
    }

  switch (mode)
    {
    case Border_Mode::Constant:
      return -1;
    case Border_Mode::Clamp:
      return idx < 0 ? 0 : (len - 1);
    case Border_Mode::Reflect_101:
      {
        if (len == 1)
          {
            return 0;
          }
        const ptrdiff_t period = 2 * (len - 1);
        ptrdiff_t m = idx % period;
        if (m < 0)
          {
            m += period;
          }
        return m >= len ? (period - m) : m;
      }
    case Border_Mode::Wrap:
      {
        ptrdiff_t m = idx % len;
        if (m < 0)
          {
            m += len;
          }
        return m;
      }
    }
  return -1;
}

/**
 * @brief 2D point coordinate in image space (x = column, y = row).
 */
template <typename T = float>
struct Point2
{
  T x {};
  T y {};

  constexpr Point2 (void) noexcept = default;

  constexpr Point2 (T x_val, T y_val) noexcept : x (x_val), y (y_val)
  {
  }

  constexpr bool
  operator== (const Point2 &other) const noexcept = default;
};

using Point2i = Point2<int32_t>;
using Point2f = Point2<float>;
using Point2d = Point2<double>;

/**
 * @brief 2D extent descriptor (width, height).
 */
template <typename T = size_t>
struct Size2
{
  T width {};
  T height {};

  constexpr Size2 (void) noexcept = default;

  constexpr Size2 (T w, T h) noexcept : width (w), height (h)
  {
  }

  constexpr T
  area (void) const noexcept
  {
    return width * height;
  }

  constexpr bool
  empty (void) const noexcept
  {
    return width <= static_cast<T> (0) || height <= static_cast<T> (0);
  }

  constexpr bool
  operator== (const Size2 &other) const noexcept = default;
};

using Size2u = Size2<size_t>;
using Size2i = Size2<int32_t>;
using Size2f = Size2<float>;

/**
 * @brief Axis-aligned 2D rectangle defined by top-left (x, y) and
 *        dimensions (width, height).
 */
template <typename T = float>
struct Rect
{
  T x {};
  T y {};
  T width {};
  T height {};

  constexpr Rect (void) noexcept = default;

  constexpr Rect (T x_val, T y_val, T w_val, T h_val) noexcept
    : x (x_val), y (y_val), width (w_val), height (h_val)
  {
  }

  constexpr T
  x2 (void) const noexcept
  {
    return x + width;
  }

  constexpr T
  y2 (void) const noexcept
  {
    return y + height;
  }

  constexpr Point2<T>
  center (void) const noexcept
  {
    return Point2<T> (x + width / static_cast<T> (2),
                      y + height / static_cast<T> (2));
  }

  constexpr T
  area (void) const noexcept
  {
    if (width <= static_cast<T> (0) || height <= static_cast<T> (0))
      {
        return static_cast<T> (0);
      }
    return width * height;
  }

  constexpr bool
  empty (void) const noexcept
  {
    return width <= static_cast<T> (0) || height <= static_cast<T> (0);
  }

  constexpr bool
  contains (Point2<T> pt) const noexcept
  {
    return pt.x >= x && pt.x < x2 () && pt.y >= y && pt.y < y2 ();
  }

  constexpr Rect
  intersect (const Rect &other) const noexcept
  {
    const T ix1 = std::max (x, other.x);
    const T iy1 = std::max (y, other.y);
    const T ix2 = std::min (x2 (), other.x2 ());
    const T iy2 = std::min (y2 (), other.y2 ());
    if (ix2 <= ix1 || iy2 <= iy1)
      {
        return Rect ();
      }
    return Rect (ix1, iy1, ix2 - ix1, iy2 - iy1);
  }

  constexpr double
  iou (const Rect &other) const noexcept
  {
    const Rect inter = intersect (other);
    const double inter_area = static_cast<double> (inter.area ());
    if (inter_area <= 0.0)
      {
        return 0.0;
      }
    const double union_area =
      static_cast<double> (area ()) + static_cast<double> (other.area ())
      - inter_area;
    return union_area > 0.0 ? (inter_area / union_area) : 0.0;
  }

  constexpr bool
  operator== (const Rect &other) const noexcept = default;
};

using Rectu = Rect<size_t>;
using Recti = Rect<int32_t>;
using Rectf = Rect<float>;
using Rectd = Rect<double>;

/**
 * @brief Oriented 2D bounding box defined by center, size, and angle in
 *        radians.
 */
struct Rotated_Rect
{
  Point2f center {};
  Size2f size {};
  float angle_rad = 0.0F;

  constexpr Rotated_Rect (void) noexcept = default;

  constexpr Rotated_Rect (Point2f c, Size2f s, float a_rad) noexcept
    : center (c), size (s), angle_rad (a_rad)
  {
  }
};

/**
 * @brief Fixed-channel pixel tuple for border values and fill colors.
 */
template <pixel_scalar T, size_t C = 4>
struct Scalar
{
  std::array<T, C> v {};

  constexpr Scalar (void) noexcept = default;

  constexpr explicit Scalar (T c0) noexcept
  {
    v.fill (c0);
  }

  constexpr Scalar (T c0, T c1, T c2, T c3 = Pixel_Traits<T>::zero ()) noexcept
  {
    if constexpr (C >= 1)
      v[0] = c0;
    if constexpr (C >= 2)
      v[1] = c1;
    if constexpr (C >= 3)
      v[2] = c2;
    if constexpr (C >= 4)
      v[3] = c3;
  }

  constexpr T &
  operator[] (size_t idx) noexcept
  {
    return v[idx];
  }

  constexpr const T &
  operator[] (size_t idx) const noexcept
  {
    return v[idx];
  }
};

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_TYPES_HPP */
