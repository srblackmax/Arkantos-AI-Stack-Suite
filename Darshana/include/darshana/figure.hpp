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

#ifndef DARSHANA_FIGURE_HPP
#define DARSHANA_FIGURE_HPP

#include <darshana/error.hpp>
#include <darshana/export.hpp>
#include <darshana/gpu.hpp>
#include <darshana/memory.hpp>
#include <darshana/observability.hpp>
#include <darshana/spec.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace darshana::inline v1
{

/**
 * @brief Selects the rendering backend for a Figure instance.
 * @pre None.
 * @post Auto selects VulkanGpu when available and falls back to CpuVector.
 * @note Thread-safety: Immutable enum type.
 */
enum class BackendKind : uint8_t
{
  Auto = 0,
  VulkanGpu = 1,
  CpuVector = 2
};

/**
 * @brief Tightly packed 8-bit RGBA framebuffer image for readback and golden
 * image verification.
 * @pre rgba.size() == width * height * 4.
 * @post Supports deterministic pixel-by-pixel comparison with tolerance.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct ImageBuffer
{
  uint32_t width = 0U;
  uint32_t height = 0U;
  std::vector<uint8_t> rgba;

  /**
   * @brief Computes the fraction of pixels matching reference within
   * tolerance.
   * @param reference Golden reference ImageBuffer of identical dimensions.
   * @param tolerance Maximum per-channel absolute difference allowed.
   * @pre width == reference.width and height == reference.height.
   * @post State is unchanged.
   * @returns Result<double> match ratio in [0.0, 1.0] or DataError.
   * @note Thread-safety: Pure const comparison; thread-safe.
   */
  [[nodiscard]] auto pixel_match_ratio (const ImageBuffer &reference,
                                        uint8_t tolerance
                                        = 2U) const noexcept -> Result<double>;

  /**
   * @brief Computes the maximum per-channel difference against reference.
   * @param reference Golden reference ImageBuffer of identical dimensions.
   * @pre width == reference.width and height == reference.height.
   * @post State is unchanged.
   * @returns Result<uint8_t> maximum channel delta or DataError.
   * @note Thread-safety: Pure const comparison; thread-safe.
   */
  [[nodiscard]] auto max_channel_diff (
      const ImageBuffer &reference) const noexcept -> Result<uint8_t>;

  /**
   * @brief Computes a 64-bit FNV-1a checksum of the RGBA pixel buffer.
   * @pre None.
   * @post State is unchanged.
   * @returns 64-bit FNV-1a hash value.
   * @note Thread-safety: Pure const calculation; thread-safe.
   */
  [[nodiscard]] auto fnv1a_hash () const noexcept -> uint64_t;
};

/**
 * @brief Re-entrant, layer-composited figure owning its plot Spec, data,
 * scratch arena, and GPU/CPU rendering resources.
 * @pre Created via Figure::create() or Figure::create_with_device().
 * @post Interactive overlay updates never trigger redraws of the static base
 *       data layer.
 * @note Thread-safety: Distinct Figure instances can be rendered concurrently
 *       from multiple threads without shared mutable state.
 */
class Figure
{
public:
  Figure () noexcept;
  Figure (const Figure &) = delete;
  auto operator= (const Figure &) -> Figure & = delete;

  /**
   * @brief Transfers ownership of a Figure instance.
   * @param other Source Figure to move from.
   * @pre None.
   * @post other is in an empty moved-from state.
   */
  Figure (Figure &&other) noexcept;

  /**
   * @brief Move-assigns ownership of a Figure instance.
   * @param other Source Figure to move from.
   * @pre None.
   * @post Previous resources are released and other is empty.
   * @returns Reference to *this.
   */
  auto operator= (Figure &&other) noexcept -> Figure &;

  /**
   * @brief Destroys the Figure and releases all owned GPU/CPU resources.
   */
  ~Figure () noexcept;

  /**
   * @brief Creates a Figure from a declarative Spec and backend preference.
   * @param spec Declarative plot specification.
   * @param backend Preferred rendering backend (Auto, VulkanGpu, CpuVector).
   * @param max_points Maximum point capacity to pre-allocate for
   * streaming/GPU.
   * @pre spec.validate() succeeds and max_points > 0.
   * @post If Vulkan initialization fails, gracefully falls back to CpuVector
   *       with a logged warning.
   * @returns Result<Figure> or DataError if spec is invalid.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] static auto
  create (const Spec &spec, BackendKind backend = BackendKind::Auto,
          std::size_t max_points = 65536U) noexcept -> Result<Figure>;

  /**
   * @brief Creates a Figure bound to an explicit GpuDevice instance.
   * @param spec Declarative plot specification.
   * @param device GpuDevice providing the memory pool and Vulkan context.
   * @param max_points Maximum point capacity to pre-allocate from device pool.
   * @pre spec.validate() succeeds and max_points > 0.
   * @post If device is invalid, gracefully falls back to CpuVector with a
   *       logged warning.
   * @returns Result<Figure> or DataError if spec is invalid.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] static auto
  create_with_device (const Spec &spec, GpuDevice &device,
                      std::size_t max_points
                      = 65536U) noexcept -> Result<Figure>;

  /**
   * @brief Replaces the plot Spec and marks the static base layer dirty.
   * @param spec New validated plot specification.
   * @pre spec.validate() succeeds.
   * @post Next render_frame() will redraw the static base layer.
   * @returns Result<void> or DataError if spec is invalid.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto set_spec (const Spec &spec) noexcept -> Result<void>;

  /**
   * @brief Retrieves the current declarative plot Spec.
   * @pre Figure is valid.
   * @post State is unchanged.
   * @returns Const reference to the owned Spec.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto spec () const noexcept -> const Spec &;

  /**
   * @brief Binds a static SnapshotData dataset and schedules async GPU upload.
   * @param data SnapshotData containing plot points.
   * @pre data.point_count() > 0 and data.point_count() <= max_points.
   * @post Copies points into pre-allocated GPU buffer without allocating new
   *       VRAM and marks base layer dirty.
   * @returns Result<void> or DataError/GpuError.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto
  bind_snapshot (const SnapshotData &data) noexcept -> Result<void>;

  /**
   * @brief Binds raw x and y coordinate spans as a snapshot dataset.
   * @param x Horizontal coordinate span.
   * @param y Vertical coordinate span.
   * @pre !x.empty(), x.size() == y.size(), and x.size() <= max_points.
   * @post Uploads coordinates into pre-allocated GPU buffer and marks base
   *       layer dirty.
   * @returns Result<void> or DataError/GpuError.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto
  bind_xy (std::span<const float> x,
           std::span<const float> y) noexcept -> Result<void>;

  /**
   * @brief Appends a single streaming data point in O(1) to the ring buffer.
   * @param x Horizontal coordinate.
   * @param y Vertical coordinate.
   * @pre Figure is valid.
   * @post Appends (x, y) to the O(1) ring buffer with zero heap allocation.
   * @returns Result<void> or DataError.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto append_stream (float x, float y) noexcept -> Result<void>;

  /**
   * @brief Appends a batch of streaming data points to the O(1) ring buffer.
   * @param x Horizontal coordinate span.
   * @param y Vertical coordinate span.
   * @pre x.size() == y.size().
   * @post Appends batch to ring buffer without heap allocation.
   * @returns Result<void> or DataError.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto
  append_stream_batch (std::span<const float> x,
                       std::span<const float> y) noexcept -> Result<void>;

  /**
   * @brief Updates interactive overlays (crosshair, tooltip, brush) WITHOUT
   * invalidating the static base data layer.
   * @param overlay New OverlaySpec state.
   * @pre Coordinates in overlay lie in [0.0, 1.0].
   * @post Marks ONLY the overlay layer dirty; base_layer_dirty remains false!
   * @returns Result<void> or DataError if coordinates are NaN/infinite.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto
  update_overlay (const OverlaySpec &overlay) noexcept -> Result<void>;

  /**
   * @brief Updates the axis viewport domain for interactive pan/zoom.
   * @param x_min New minimum X domain coordinate.
   * @param x_max New maximum X domain coordinate.
   * @param y_min New minimum Y domain coordinate.
   * @param y_max New maximum Y domain coordinate.
   * @pre x_min < x_max and y_min < y_max.
   * @post Updates scale domains and marks base layer dirty for re-projection.
   * @returns Result<void> or DataError on invalid bounds.
   * @note Thread-safety: Mutates this Figure instance.
   */
  [[nodiscard]] auto set_viewport (double x_min, double x_max, double y_min,
                                   double y_max) noexcept -> Result<void>;

  /**
   * @brief Renders a frame using the layered GPU or CPU pipeline.
   * @param force_base_redraw If true, forces base layer redraw even if cached
   *        (used by raw GPU throughput benchmarks).
   * @pre Figure is valid.
   * @post If base layer is clean and only overlay changed, composites overlay
   *       onto cached base layer without redrawing static data.
   * @returns Result<FrameStatsSnapshot> or GpuError/FrameworkError.
   * @note Thread-safety: Safe to call concurrently on distinct Figures.
   */
  [[nodiscard]] auto render_frame (bool force_base_redraw = false) noexcept
      -> Result<FrameStatsSnapshot>;

  /**
   * @brief Reads back the composited RGBA framebuffer into host memory.
   * @pre At least one frame has been rendered (or renders on demand).
   * @post Returns ImageBuffer of dimensions spec().width x spec().height.
   * @returns Result<ImageBuffer> or GpuError/FrameworkError.
   * @note Thread-safety: Mutates readback staging buffer of this Figure.
   */
  [[nodiscard]] auto readback_image () noexcept -> Result<ImageBuffer>;

  /**
   * @brief Asynchronously exports the Figure to SVG or PDF without blocking.
   * @param file_path Destination path.
   * @param format ExportFormat (Svg or Pdf).
   * @param callback Optional completion callback.
   * @pre !file_path.empty().
   * @post Spawns background export job and returns immediately.
   * @returns std::future<Result<ExportResult>> resolving upon completion.
   * @note Thread-safety: Non-blocking; safe to invoke during rendering.
   */
  [[nodiscard]] auto
  export_async (std::string_view file_path,
                ExportFormat format = ExportFormat::Svg,
                ExportCallback callback
                = nullptr) const noexcept -> std::future<Result<ExportResult>>;

  /**
   * @brief Queries the active rendering backend for this Figure.
   * @pre None.
   * @post State is unchanged.
   * @returns BackendKind::VulkanGpu or BackendKind::CpuVector.
   */
  [[nodiscard]] auto active_backend () const noexcept -> BackendKind;

  /**
   * @brief Accesses the per-figure FrameStats telemetry accumulator.
   * @pre Figure is valid.
   * @post State is unchanged.
   * @returns Const reference to this Figure's FrameStats.
   */
  [[nodiscard]] auto stats () const noexcept -> const FrameStats &;

  /**
   * @brief Queries how many times the static base data layer was redrawn.
   * @pre None.
   * @post State is unchanged.
   * @returns Base layer redraw count.
   */
  [[nodiscard]] auto base_redraw_count () const noexcept -> uint64_t;

  /**
   * @brief Queries how many times the interactive overlay layer was
   * composited.
   * @pre None.
   * @post State is unchanged.
   * @returns Overlay layer redraw count.
   */
  [[nodiscard]] auto overlay_redraw_count () const noexcept -> uint64_t;

private:
  struct Impl;
  explicit Figure (std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_FIGURE_HPP */
