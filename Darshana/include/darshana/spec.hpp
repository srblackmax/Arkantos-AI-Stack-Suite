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

#ifndef DARSHANA_SPEC_HPP
#define DARSHANA_SPEC_HPP

#include <darshana/error.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace darshana::inline v1
{

/**
 * @brief 8-bit per channel RGBA color representation.
 * @pre None.
 * @post Stores red, green, blue, and alpha channels in [0, 255].
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct ColorRgba
{
  uint8_t r = 31U;
  uint8_t g = 119U;
  uint8_t b = 180U;
  uint8_t a = 255U;

  /**
   * @brief Creates a ColorRgba from a 24-bit 0xRRGGBB hex integer and alpha.
   * @param rgb_hex 24-bit RGB hexadecimal color (e.g. 0x1F77B4).
   * @param alpha Alpha channel value in [0, 255].
   * @pre None.
   * @post Returns initialized ColorRgba.
   * @returns Constructed ColorRgba value.
   * @note Thread-safety: Pure constexpr function.
   */
  [[nodiscard]] static constexpr auto
  from_hex (uint32_t rgb_hex, uint8_t alpha = 255U) noexcept -> ColorRgba
  {
    return ColorRgba{
      static_cast<uint8_t> ((rgb_hex >> 16U) & 0xFFU),
      static_cast<uint8_t> ((rgb_hex >> 8U) & 0xFFU),
      static_cast<uint8_t> (rgb_hex & 0xFFU),
      alpha,
    };
  }

  /**
   * @brief Packs the RGBA channels into a 32-bit ABGR word for GPU buffers.
   * @pre None.
   * @post Returns packed 32-bit color.
   * @returns uint32_t packed word.
   * @note Thread-safety: Pure constexpr function.
   */
  [[nodiscard]] constexpr auto
  pack_abgr () const noexcept -> uint32_t
  {
    return (static_cast<uint32_t> (a) << 24U)
           | (static_cast<uint32_t> (b) << 16U)
           | (static_cast<uint32_t> (g) << 8U) | static_cast<uint32_t> (r);
  }

  auto operator== (const ColorRgba &) const noexcept -> bool = default;
};

/**
 * @brief Visual mark primitives in the Darshana Grammar of Graphics.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class MarkType : uint8_t
{
  Scatter = 0,
  Line = 1,
  Bar = 2,
  Area = 3,
  Heatmap = 4,
  Box = 5,
  Violin = 6,
  Histogram = 7,
  Kde = 8,
  Ecdf = 9,
  RegressionLine = 10,
  ConfidenceBand = 11,
  ErrorBar = 12
};

/**
 * @brief Coordinate scale transformation types.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class ScaleType : uint8_t
{
  Linear = 0,
  Log = 1,
  Categorical = 2
};

/**
 * @brief GPU/CPU signed-distance field marker shapes for scatter plots.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class MarkerShape : uint8_t
{
  Circle = 0,
  Square = 1,
  Triangle = 2,
  Diamond = 3,
  Cross = 4
};

/**
 * @brief Parametric stroke dash patterns for line and overlay marks.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class DashPattern : uint8_t
{
  Solid = 0,
  Dashed = 1,
  Dotted = 2,
  DashDot = 3
};

/**
 * @brief Defines domain-to-range coordinate mapping for a plot axis.
 * @pre For Linear/Log scales, domain_min != domain_max; for Log scales,
 *      domain_min > 0 and domain_max > 0.
 * @post Maps domain values monotonically into [range_min, range_max].
 * @note Thread-safety: Const methods are safe for concurrent reads.
 */
struct ScaleSpec
{
  ScaleType type = ScaleType::Linear;
  double domain_min = 0.0;
  double domain_max = 1.0;
  double range_min = 0.0;
  double range_max = 1.0;
  bool auto_domain = true;
  bool clamp = true;
  std::vector<std::string> categories;

  /**
   * @brief Maps a continuous domain value to the target range.
   * @param value Input data coordinate.
   * @pre Scale domain is valid (and value > 0 for ScaleType::Log).
   * @post Returned value lies within [min(range), max(range)] when clamped.
   * @returns Result<double> containing mapped range coordinate.
   * @note Thread-safety: Pure const calculation; thread-safe.
   */
  [[nodiscard]] auto map (double value) const noexcept -> Result<double>;

  /**
   * @brief Maps a discrete category label to a normalized range coordinate.
   * @param category Category string present in categories.
   * @pre type == ScaleType::Categorical and !categories.empty().
   * @post Returned value lies within [range_min, range_max].
   * @returns Result<double> containing mapped coordinate or DataError.
   * @note Thread-safety: Pure const calculation; thread-safe.
   */
  [[nodiscard]] auto
  map_category (std::string_view category) const noexcept -> Result<double>;

  auto operator== (const ScaleSpec &) const noexcept -> bool = default;
};

/**
 * @brief Aesthetic styling parameters for a visual mark layer.
 * @pre stroke_width >= 0, marker_size > 0, opacity in [0, 1].
 * @post Controls SDF marker/stroke appearance on GPU and CPU backends.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct AestheticSpec
{
  ColorRgba stroke_color{ 31U, 119U, 180U, 255U };
  ColorRgba fill_color{ 31U, 119U, 180U, 190U };
  float stroke_width = 1.5F;
  float marker_size = 4.0F;
  MarkerShape marker_shape = MarkerShape::Circle;
  DashPattern dash_pattern = DashPattern::Solid;
  float opacity = 1.0F;
  float sdf_aa_width = 1.25F;

  auto operator== (const AestheticSpec &) const noexcept -> bool = default;
};

/**
 * @brief Maps dataset columns to visual channels of a layer.
 * @pre None.
 * @post Specifies column names bound to x, y, y2, color, and size.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct EncodingSpec
{
  std::string x_field = "x";
  std::string y_field = "y";
  std::string y2_field;
  std::string color_field;
  std::string size_field;
  std::string label;

  auto operator== (const EncodingSpec &) const noexcept -> bool = default;
};

/**
 * @brief Describes a single visual mark layer within a plot Spec.
 * @pre None.
 * @post Encapsulates mark type, channel encodings, aesthetics, and optional
 *       inline summary/statistical data.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct LayerSpec
{
  MarkType mark = MarkType::Scatter;
  EncodingSpec encoding;
  AestheticSpec aesthetics;
  std::vector<float> inline_x;
  std::vector<float> inline_y;
  std::vector<float> inline_y2;
  std::vector<float> inline_q1;
  std::vector<float> inline_q3;
  std::size_t matrix_rows = 0U;
  std::size_t matrix_cols = 0U;
  std::vector<float> matrix_values;

  auto operator== (const LayerSpec &) const noexcept -> bool = default;
};

/**
 * @brief Describes interactive overlay elements rendered on a separate layer.
 * @pre Coordinates are normalized in [0.0, 1.0] relative to the plot canvas.
 * @post Updating OverlaySpec never invalidates or redraws the static data
 *       layer.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct OverlaySpec
{
  bool crosshair_enabled = false;
  float crosshair_x = 0.5F;
  float crosshair_y = 0.5F;
  bool tooltip_enabled = false;
  float tooltip_x = 0.5F;
  float tooltip_y = 0.5F;
  std::string tooltip_text;
  bool brush_enabled = false;
  float brush_x0 = 0.0F;
  float brush_y0 = 0.0F;
  float brush_x1 = 0.0F;
  float brush_y1 = 0.0F;

  auto operator== (const OverlaySpec &) const noexcept -> bool = default;
};

/**
 * @brief Renderer-agnostic declarative plot specification (Grammar of
 * Graphics).
 * @pre width > 0, height > 0, dpi > 0, and !layers.empty().
 * @post Can be serialized to/from JSON and rendered by Vulkan GPU or CPU SVG
 *       backends.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct Spec
{
  std::string title;
  std::string subtitle;
  std::string x_label = "x";
  std::string y_label = "y";
  uint32_t width = 800U;
  uint32_t height = 600U;
  float dpi = 96.0F;
  ColorRgba background{ 255U, 255U, 255U, 255U };
  bool show_grid = true;
  bool show_legend = true;
  ScaleSpec x_scale;
  ScaleSpec y_scale;
  std::vector<LayerSpec> layers;
  OverlaySpec overlay;

  /**
   * @brief Validates structural and numerical invariants of the Spec.
   * @pre None.
   * @post State is unchanged.
   * @returns Result<void> if valid, or DataError describing the violation.
   * @note Thread-safety: Pure const validation; thread-safe.
   */
  [[nodiscard]] auto validate () const noexcept -> Result<void>;

  /**
   * @brief Serializes this Spec into a deterministic JSON string.
   * @pre Spec must be valid.
   * @post State is unchanged.
   * @returns Result<std::string> containing JSON document or DataError.
   * @note Thread-safety: Pure const serialization; thread-safe.
   */
  [[nodiscard]] auto to_json () const noexcept -> Result<std::string>;

  /**
   * @brief Deserializes a Spec from a JSON string representation.
   * @param json UTF-8 JSON document produced by to_json() or user config.
   * @pre !json.empty().
   * @post Returned Spec satisfies validate() and equals original Spec.
   * @returns Result<Spec> or DataError with DataErrorCode::JsonParseError.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  from_json (std::string_view json) noexcept -> Result<Spec>;

  auto operator== (const Spec &) const noexcept -> bool = default;
};

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_SPEC_HPP */
