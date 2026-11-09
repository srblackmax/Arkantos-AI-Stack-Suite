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

#ifndef DARSHANA_GPU_INTERNAL_HPP
#define DARSHANA_GPU_INTERNAL_HPP

#include <darshana/error.hpp>
#include <darshana/gpu.hpp>

#include <cstdint>
#include <span>

namespace darshana::inline v1::detail
{

/**
 * @brief Parameters for a Vulkan compute base-layer draw pass.
 */
struct GpuBasePassParams
{
  uint32_t width = 800U;
  uint32_t height = 600U;
  uint32_t point_count = 0U;
  bool is_line_plot = false;
  float scale_x = 1.0F;
  float offset_x = 0.0F;
  float scale_y = 1.0F;
  float offset_y = 0.0F;
  uint32_t bg_color = 0xFFFFFFFFU;
  uint32_t grid_color = 0xFFE6E6E6U;
  uint32_t mark_color = 0xFFB4771FU;
};

/**
 * @brief Parameters for a Vulkan compute interactive overlay pass.
 */
struct GpuOverlayPassParams
{
  uint32_t width = 800U;
  uint32_t height = 600U;
  uint32_t cross_x = 0xFFFFFFFFU;
  uint32_t cross_y = 0xFFFFFFFFU;
  uint32_t brush_x0 = 0U;
  uint32_t brush_y0 = 0U;
  uint32_t brush_x1 = 0U;
  uint32_t brush_y1 = 0U;
};

/* Acquires or returns the process-wide shared default GpuDevice.  */
[[nodiscard]] auto acquire_shared_gpu_device () noexcept -> Result<GpuDevice>;

/* Executes the static base-layer compute pass (clear/grid + scatter/line).  */
[[nodiscard]] auto gpu_execute_base_pass (
    GpuDevice &device, const GpuBuffer &points_buf, GpuBuffer &base_pixels_buf,
    const GpuBasePassParams &params) noexcept -> Result<void>;

/* Executes the overlay compositing compute pass onto out_pixels_buf.  */
[[nodiscard]] auto gpu_execute_overlay_pass (
    GpuDevice &device, const GpuBuffer &base_pixels_buf,
    GpuBuffer &out_pixels_buf,
    const GpuOverlayPassParams &params) noexcept -> Result<void>;

/* Reads back RGBA pixels from a device GpuBuffer into host memory.  */
[[nodiscard]] auto
gpu_readback_pixels (GpuDevice &device, const GpuBuffer &pixels_buf,
                     std::span<uint8_t> dst_rgba) noexcept -> Result<void>;

} /* namespace darshana::inline v1::detail */

#endif /* DARSHANA_GPU_INTERNAL_HPP */
