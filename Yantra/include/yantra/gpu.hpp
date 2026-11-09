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

#ifndef YANTRA_GPU_HPP
#define YANTRA_GPU_HPP

#include <yantra/error.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace yantra::inline v1
{

/**
 * @brief Checks whether a usable CUDA GPU device is present at runtime.
 * @pre None.
 * @post State is unchanged.
 * @returns True if CUDA support was compiled and a GPU is accessible.
 * @note Thread-safety: Safe for concurrent calls across threads.
 */
[[nodiscard]] auto is_cuda_available () noexcept -> bool;

/**
 * @brief Opaque RAII handle wrapping a non-blocking CUDA execution stream.
 * @pre CUDA runtime must be available when calling GpuStream::create().
 * @post Destructor releases the underlying CUDA stream automatically.
 * @note Thread-safety: Move-only handle; synchronization is thread-safe.
 */
class GpuStream
{
public:
  GpuStream () noexcept = default;
  GpuStream (const GpuStream &) = delete;
  auto operator= (const GpuStream &) -> GpuStream & = delete;

  /**
   * @brief Transfers ownership of a GpuStream handle.
   * @param other Source GpuStream to move from.
   * @post other.native_handle() is nullptr.
   */
  GpuStream (GpuStream &&other) noexcept;

  /**
   * @brief Move-assigns ownership of a GpuStream handle.
   * @param other Source GpuStream to move from.
   * @returns Reference to *this.
   */
  auto operator= (GpuStream &&other) noexcept -> GpuStream &;

  /**
   * @brief Destroys the underlying CUDA stream if owned.
   */
  ~GpuStream () noexcept;

  /**
   * @brief Allocates a non-blocking CUDA stream and configures the device
   * async memory pool (cudaMallocAsync).
   * @returns Result<GpuStream> or IndexErrorCode::GpuOutOfMemory on failure.
   */
  [[nodiscard]] static auto create () noexcept -> Result<GpuStream>;

  /**
   * @brief Waits for all queued operations on this stream to complete.
   * @returns Result<void> or IndexError on failure.
   */
  [[nodiscard]] auto synchronize () const noexcept -> Result<void>;

  /**
   * @brief Accesses the opaque native stream pointer.
   * @returns Opaque pointer to cudaStream_t or nullptr.
   */
  [[nodiscard]] auto native_handle () const noexcept -> void *;

private:
  explicit GpuStream (void *handle) noexcept;

  void *native_handle_ = nullptr;
};

/**
 * @brief Opaque RAII handle managing a device buffer allocated via
 * cudaMallocAsync on a stream-ordered memory pool.
 * @pre Never calls synchronous cudaMalloc on a per-query path.
 * @post Destructor frees the device memory asynchronously via cudaFreeAsync.
 * @note Thread-safety: Exclusive ownership; move-only.
 */
class GpuBuffer
{
public:
  GpuBuffer () noexcept = default;
  GpuBuffer (const GpuBuffer &) = delete;
  auto operator= (const GpuBuffer &) -> GpuBuffer & = delete;

  GpuBuffer (GpuBuffer &&other) noexcept;
  auto operator= (GpuBuffer &&other) noexcept -> GpuBuffer &;
  ~GpuBuffer () noexcept;

  /**
   * @brief Allocates device memory asynchronously from the stream pool.
   * @param byte_size Number of bytes to allocate (> 0).
   * @param stream Target GpuStream for stream-ordered pool allocation.
   * @returns Result<GpuBuffer> or IndexErrorCode::GpuOutOfMemory.
   */
  [[nodiscard]] static auto
  allocate_async (std::size_t byte_size,
                  const GpuStream &stream) noexcept -> Result<GpuBuffer>;

  /**
   * @brief Enqueues an asynchronous host-to-device copy on stream.
   * @param src Host byte span matching size_bytes().
   * @param stream Target GpuStream.
   * @returns Result<void> or IndexError.
   */
  [[nodiscard]] auto
  copy_from_host_async (std::span<const std::byte> src,
                        const GpuStream &stream) noexcept -> Result<void>;

  /**
   * @brief Enqueues an asynchronous device-to-host copy on stream.
   * @param dst Host byte span matching size_bytes().
   * @param stream Target GpuStream.
   * @returns Result<void> or IndexError.
   */
  [[nodiscard]] auto
  copy_to_host_async (std::span<std::byte> dst,
                      const GpuStream &stream) const noexcept -> Result<void>;

  /**
   * @brief Queries the allocated size in bytes.
   */
  [[nodiscard]] auto size_bytes () const noexcept -> std::size_t;

  /**
   * @brief Accesses the opaque device pointer.
   */
  [[nodiscard]] auto device_ptr () const noexcept -> void *;

private:
  GpuBuffer (void *ptr, std::size_t bytes, void *stream_handle) noexcept;

  void *device_ptr_ = nullptr;
  std::size_t size_bytes_ = 0U;
  void *stream_handle_ = nullptr;
};

/**
 * @brief Represents a single nearest-neighbor result from GpuIndex or CPU
 * vector indices.
 * @note Thread-safety: Value type.
 */
struct NeighborResult
{
  uint64_t id = 0U;
  float distance = 0.0F;
};

/**
 * @brief Opaque RAII handle for GPU-accelerated CAGRA-style graph & batch
 * vector search.
 * @pre CUDA device must be available when calling GpuIndex::build().
 * @post All GPU buffers are managed via cudaMallocAsync pool allocator; zero
 * per-query cudaMalloc calls occur during search().
 * @note Thread-safety: Concurrent search() calls are safe.
 */
class GpuIndex
{
public:
  GpuIndex () noexcept;
  GpuIndex (const GpuIndex &) = delete;
  auto operator= (const GpuIndex &) -> GpuIndex & = delete;
  GpuIndex (GpuIndex &&other) noexcept;
  auto operator= (GpuIndex &&other) noexcept -> GpuIndex &;
  ~GpuIndex () noexcept;

  /**
   * @brief Builds a GPU CAGRA-style fixed-degree graph index from host fp32
   * vectors and their level-0 navigable graph neighbors.
   * @param vectors Contiguous row-major fp32 vectors [num_vectors * dim].
   * @param graph_neighbors Fixed-degree neighbor list [num_vectors * degree].
   * @param num_vectors Number of vectors in the dataset.
   * @param dim Vector dimensionality.
   * @param degree Fixed out-degree per node in the navigable graph.
   * @returns Result<GpuIndex> or IndexErrorCode::GpuOutOfMemory.
   */
  [[nodiscard]] static auto
  build (std::span<const float> vectors,
         std::span<const uint32_t> graph_neighbors, std::size_t num_vectors,
         std::size_t dim, std::size_t degree) noexcept -> Result<GpuIndex>;

  /**
   * @brief Executes GPU-accelerated approximate k-NN search for a single
   * query or batch of queries using pre-allocated stream pool workspace.
   * @param query Query vector of length dim().
   * @param k Number of nearest neighbors to retrieve.
   * @returns Result<std::vector<NeighborResult>> sorted by ascending L2
   * distance.
   */
  [[nodiscard]] auto
  search (std::span<const float> query,
          std::size_t k) const noexcept -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Executes high-throughput batched GPU search over num_queries
   * queries simultaneously on the RTX 3070.
   * @param queries Contiguous [num_queries * dim] query vectors.
   * @param num_queries Number of queries in the batch.
   * @param k Number of top neighbors per query.
   * @returns Result<std::vector<NeighborResult>> of size num_queries * k.
   */
  [[nodiscard]] auto search_batch (std::span<const float> queries,
                                   std::size_t num_queries,
                                   std::size_t k) const noexcept
      -> Result<std::vector<NeighborResult>>;

  /**
   * @brief Queries the number of indexed vectors.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Queries the vector dimensionality.
   */
  [[nodiscard]] auto dimension () const noexcept -> std::size_t;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} /* namespace yantra::inline v1 */

#endif /* YANTRA_GPU_HPP */
