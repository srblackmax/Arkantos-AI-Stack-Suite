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

#ifndef DARSHANA_GPU_HPP
#define DARSHANA_GPU_HPP

#include <darshana/error.hpp>
#include <darshana/memory.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace darshana::inline v1
{

/**
 * @brief Configuration options for Vulkan 1.2+ GpuDevice initialization.
 * @pre device_pool_bytes > 0 and staging_pool_bytes > 0.
 * @post Controls pre-allocated GPU memory pool sizes and device selection.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct GpuDeviceConfig
{
  std::size_t device_pool_bytes = 256ULL * 1024ULL * 1024ULL;
  std::size_t staging_pool_bytes = 128ULL * 1024ULL * 1024ULL;
  bool prefer_discrete_gpu = true;
  bool force_device_failure = false;
};

/**
 * @brief Opaque RAII handle wrapping a Vulkan 1.2+ logical device, compute
 * pipelines, timeline semaphores, and pre-allocated GPU memory pools.
 * @pre Vulkan runtime must be present when calling GpuDevice::create().
 * @post Destructor deterministically releases all owned Vulkan resources.
 * @note Thread-safety: Move-only handle; pool sub-allocation and queue
 *       submissions are internally synchronized for multi-threaded Figure
 *       rendering.
 */
class GpuDevice
{
public:
  struct Impl;

  GpuDevice () noexcept;
  GpuDevice (const GpuDevice &) = delete;
  auto operator= (const GpuDevice &) -> GpuDevice & = delete;

  /**
   * @brief Transfers ownership of a GpuDevice handle.
   * @param other Source GpuDevice to move from.
   * @pre None.
   * @post other.is_valid() is false.
   */
  GpuDevice (GpuDevice &&other) noexcept;

  /**
   * @brief Move-assigns ownership of a GpuDevice handle.
   * @param other Source GpuDevice to move from.
   * @pre None.
   * @post Previous resources are released and other.is_valid() is false.
   * @returns Reference to *this.
   */
  auto operator= (GpuDevice &&other) noexcept -> GpuDevice &;

  /**
   * @brief Destroys the underlying Vulkan device handle if owned.
   */
  ~GpuDevice () noexcept;

  /**
   * @brief Initializes a Vulkan 1.2+ device and pre-allocated memory pools.
   * @param config Pool capacities and device selection preferences.
   * @pre config.device_pool_bytes > 0 and config.staging_pool_bytes > 0.
   * @post Returned GpuDevice owns pre-allocated device-local and staging
   *       memory slabs.
   * @returns Result<GpuDevice> or GpuError on initialization failure.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] static auto create (const GpuDeviceConfig &config
                                    = {}) noexcept -> Result<GpuDevice>;

  /**
   * @brief Checks whether this GpuDevice owns an active Vulkan context.
   * @pre None.
   * @post State is unchanged.
   * @returns True if valid and not in simulated device-loss state.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto is_valid () const noexcept -> bool;

  /**
   * @brief Retrieves the physical GPU device name (e.g. "NVIDIA GeForce RTX
   * 3070").
   * @pre is_valid() is true.
   * @post State is unchanged.
   * @returns String view of physical device name.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto device_name () const noexcept -> std::string_view;

  /**
   * @brief Queries the bytes currently sub-allocated from the GPU pool.
   * @pre None.
   * @post State is unchanged.
   * @returns Active bytes sub-allocated across live GpuBuffers.
   * @note Thread-safety: Lock-free atomic load.
   */
  [[nodiscard]] auto allocated_bytes () const noexcept -> std::size_t;

  /**
   * @brief Queries the peak bytes sub-allocated from the GPU pool.
   * @pre None.
   * @post State is unchanged.
   * @returns High-water mark of GPU pool sub-allocations in bytes.
   * @note Thread-safety: Lock-free atomic load.
   */
  [[nodiscard]] auto peak_allocated_bytes () const noexcept -> std::size_t;

  /**
   * @brief Queries the total pre-allocated device-local pool capacity.
   * @pre None.
   * @post State is unchanged.
   * @returns Pool capacity in bytes.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto pool_capacity_bytes () const noexcept -> std::size_t;

  /**
   * @brief Waits for all in-flight GPU command buffers to complete.
   * @pre is_valid() is true.
   * @post All submitted GPU work has retired.
   * @returns Result<void> or GpuError on device loss.
   * @note Thread-safety: Synchronized across threads.
   */
  [[nodiscard]] auto wait_idle () const noexcept -> Result<void>;

  /**
   * @brief Injects a simulated VK_ERROR_DEVICE_LOST fault for testing error
   * paths and fallback behavior.
   * @pre None.
   * @post Subsequent GPU operations on this handle return
   *       GpuErrorCode::DeviceLost.
   * @note Thread-safety: Lock-free atomic flag store.
   */
  void simulate_device_loss () noexcept;

  /**
   * @brief Accesses the opaque internal implementation pointer.
   * @pre None.
   * @post State is unchanged.
   * @returns Opaque pointer to internal Vulkan context or nullptr.
   */
  [[nodiscard]] auto native_handle () const noexcept -> void *;

private:
  explicit GpuDevice (std::shared_ptr<Impl> impl) noexcept;

  std::shared_ptr<Impl> impl_;
};

/**
 * @brief Opaque RAII handle managing a pre-allocated GPU buffer sub-allocated
 * from a GpuDevice pool.
 * @pre GpuDevice must be valid when calling GpuBuffer::allocate().
 * @post Destructor returns the slab back to the GpuDevice memory pool in O(1)
 *       without calling vkFreeMemory in the frame loop.
 * @note Thread-safety: Move-only handle owned by a single Figure.
 */
class GpuBuffer
{
public:
  struct Impl;

  GpuBuffer () noexcept;
  GpuBuffer (const GpuBuffer &) = delete;
  auto operator= (const GpuBuffer &) -> GpuBuffer & = delete;

  /**
   * @brief Transfers ownership of a GpuBuffer handle.
   * @param other Source GpuBuffer to move from.
   * @pre None.
   * @post other.capacity_bytes() == 0.
   */
  GpuBuffer (GpuBuffer &&other) noexcept;

  /**
   * @brief Move-assigns ownership of a GpuBuffer handle.
   * @param other Source GpuBuffer to move from.
   * @pre None.
   * @post Previous slab is returned to pool and other is empty.
   * @returns Reference to *this.
   */
  auto operator= (GpuBuffer &&other) noexcept -> GpuBuffer &;

  /**
   * @brief Returns the sub-allocated slab to the parent GpuDevice memory pool.
   */
  ~GpuBuffer () noexcept;

  /**
   * @brief Sub-allocates a GpuBuffer from the GpuDevice's pre-allocated pool.
   * @param device Parent GpuDevice owning the memory pool.
   * @param capacity_bytes Maximum byte capacity required for this buffer.
   * @pre device.is_valid() and capacity_bytes > 0.
   * @post Returned GpuBuffer owns a sub-allocated slab of capacity_bytes.
   * @returns Result<GpuBuffer> or GpuErrorCode::OutOfMemory if pool exhausted.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] static auto
  allocate (GpuDevice &device,
            std::size_t capacity_bytes) noexcept -> Result<GpuBuffer>;

  /**
   * @brief Asynchronously uploads 2D points from host to device via staging
   * buffer and timeline semaphore.
   * @param points Contiguous host span of DataPoint2D elements.
   * @param offset_points Destination point offset inside this GpuBuffer.
   * @pre (offset_points + points.size()) * sizeof(DataPoint2D) <=
   *      capacity_bytes().
   * @post Enqueues asynchronous DMA transfer and signals timeline value.
   * @returns Result<uint64_t> timeline semaphore counter or GpuError.
   * @note Thread-safety: Safe for concurrent uploads on distinct GpuBuffers.
   */
  [[nodiscard]] auto upload_points_async (std::span<const DataPoint2D> points,
                                          std::size_t offset_points
                                          = 0U) noexcept -> Result<uint64_t>;

  /**
   * @brief Waits on the Vulkan timeline semaphore for an async upload to
   * finish.
   * @param timeline_value Timeline counter returned by upload_points_async().
   * @pre Buffer is valid.
   * @post Host is guaranteed that the DMA transfer has completed on the GPU.
   * @returns Result<void> or GpuError on device loss/timeout.
   * @note Thread-safety: Lock-free wait via vkWaitSemaphores.
   */
  [[nodiscard]] auto
  wait_timeline (uint64_t timeline_value) const noexcept -> Result<void>;

  /**
   * @brief Queries the pre-allocated capacity of this buffer in bytes.
   * @pre None.
   * @post State is unchanged.
   * @returns Buffer capacity in bytes.
   */
  [[nodiscard]] auto capacity_bytes () const noexcept -> std::size_t;

  /**
   * @brief Queries the byte offset of this buffer within the device pool.
   * @pre None.
   * @post State is unchanged.
   * @returns Offset in bytes.
   */
  [[nodiscard]] auto pool_offset () const noexcept -> std::size_t;

  /**
   * @brief Accesses the opaque internal buffer handle.
   * @pre None.
   * @post State is unchanged.
   * @returns Opaque pointer to internal buffer representation or nullptr.
   */
  [[nodiscard]] auto native_handle () const noexcept -> void *;

private:
  explicit GpuBuffer (std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Checks whether a Vulkan 1.2+ capable GPU is available at runtime.
 * @pre None.
 * @post State is unchanged.
 * @returns True if Vulkan is enabled at build time and a GPU is accessible.
 * @note Thread-safety: Safe for concurrent calls across threads.
 */
[[nodiscard]] auto is_vulkan_available () noexcept -> bool;

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_GPU_HPP */
