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

#ifndef SUTRA_GPU_HPP
#define SUTRA_GPU_HPP

#include <sutra/error.hpp>

#include <cstddef>
#include <span>

namespace sutra::inline v1
{

/**
 * @brief Opaque RAII handle wrapping a non-blocking CUDA execution stream.
 * @pre CUDA runtime must be available when calling Stream::create().
 * @post Destructor releases the underlying CUDA stream automatically.
 * @note Thread-safety: Move-only handle; synchronization is thread-safe.
 */
class Stream
{
public:
  Stream () noexcept = default;
  Stream (const Stream &) = delete;
  auto operator= (const Stream &) -> Stream & = delete;

  /**
   * @brief Transfers ownership of a Stream handle.
   * @param other Source Stream to move from.
   * @pre None.
   * @post other.native_handle() is nullptr.
   */
  Stream (Stream &&other) noexcept;

  /**
   * @brief Move-assigns ownership of a Stream handle.
   * @param other Source Stream to move from.
   * @pre None.
   * @post Previous stream is destroyed and other.native_handle() is nullptr.
   * @returns Reference to *this.
   */
  auto operator= (Stream &&other) noexcept -> Stream &;

  /**
   * @brief Destroys the underlying CUDA stream if owned.
   */
  ~Stream () noexcept;

  /**
   * @brief Allocates a non-blocking CUDA stream on the active device.
   * @pre Active GPU device must be accessible.
   * @post Returned Stream owns a valid native stream handle.
   * @returns Result containing Stream or FrameworkErrorCode::CudaError.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] static auto create () noexcept -> Result<Stream>;

  /**
   * @brief Waits for all queued operations on this stream to complete.
   * @pre Stream must be valid.
   * @post All prior device work on this stream has completed.
   * @returns Result<void> or FrameworkErrorCode::CudaError on failure.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] auto synchronize () const noexcept -> Result<void>;

  /**
   * @brief Accesses the opaque native stream pointer.
   * @pre None.
   * @post State is unchanged.
   * @returns Opaque pointer to cudaStream_t or nullptr.
   */
  [[nodiscard]] auto native_handle () const noexcept -> void *;

private:
  explicit Stream (void *handle) noexcept;

  void *native_handle_ = nullptr;
};

/**
 * @brief Opaque RAII handle managing a contiguous fp32 buffer in GPU VRAM.
 * @pre CUDA runtime must be available when calling Tensor::allocate().
 * @post Destructor frees the device memory buffer automatically.
 * @note Thread-safety: Exclusive ownership; move-only.
 */
class Tensor
{
public:
  Tensor () noexcept = default;
  Tensor (const Tensor &) = delete;
  auto operator= (const Tensor &) -> Tensor & = delete;

  /**
   * @brief Transfers ownership of device memory from other.
   * @param other Source Tensor to move from.
   * @pre None.
   * @post other.device_ptr() is nullptr and other.element_count() is 0.
   */
  Tensor (Tensor &&other) noexcept;

  /**
   * @brief Move-assigns ownership of device memory from other.
   * @param other Source Tensor to move from.
   * @pre None.
   * @post Any previous buffer is freed and other is empty.
   * @returns Reference to *this.
   */
  auto operator= (Tensor &&other) noexcept -> Tensor &;

  /**
   * @brief Frees the device memory buffer if owned.
   */
  ~Tensor () noexcept;

  /**
   * @brief Allocates a contiguous fp32 tensor in GPU VRAM.
   * @param element_count Number of 32-bit floats to allocate (> 0).
   * @pre element_count > 0.
   * @post Returned Tensor owns element_count * sizeof(float) VRAM bytes.
   * @returns Result<Tensor> or FrameworkErrorCode::CudaError.
   * @note Thread-safety: Safe for concurrent calls across threads.
   */
  [[nodiscard]] static auto
  allocate (std::size_t element_count) noexcept -> Result<Tensor>;

  /**
   * @brief Asynchronously copies fp32 data from host memory to device VRAM.
   * @param src Host span of floats matching element_count().
   * @param stream Target CUDA Stream for asynchronous transfer.
   * @pre src.size() == element_count() and stream is valid.
   * @post Host data is enqueued for H2D transfer on stream.
   * @returns Result<void> or FrameworkErrorCode::CudaError.
   */
  [[nodiscard]] auto
  copy_from_host (std::span<const float> src,
                  const Stream &stream) noexcept -> Result<void>;

  /**
   * @brief Asynchronously copies fp32 data from device VRAM to host memory.
   * @param dst Host span of floats matching element_count().
   * @param stream Target CUDA Stream for asynchronous transfer.
   * @pre dst.size() == element_count() and stream is valid.
   * @post Device data is enqueued for D2H transfer on stream.
   * @returns Result<void> or FrameworkErrorCode::CudaError.
   */
  [[nodiscard]] auto
  copy_to_host (std::span<float> dst,
                const Stream &stream) const noexcept -> Result<void>;

  /**
   * @brief Queries the number of fp32 elements in this tensor.
   * @pre None.
   * @post State is unchanged.
   * @returns Number of float elements allocated.
   */
  [[nodiscard]] auto element_count () const noexcept -> std::size_t;

  /**
   * @brief Accesses the opaque device buffer pointer.
   * @pre None.
   * @post State is unchanged.
   * @returns Opaque pointer to device memory or nullptr.
   */
  [[nodiscard]] auto device_ptr () const noexcept -> void *;

private:
  Tensor (void *ptr, std::size_t count) noexcept;

  void *device_ptr_ = nullptr;
  std::size_t element_count_ = 0U;
};

/**
 * @brief Checks whether a usable CUDA GPU device is present at runtime.
 * @pre None.
 * @post State is unchanged.
 * @returns True if at least one CUDA-capable GPU is accessible.
 * @note Thread-safety: Safe for concurrent calls across threads.
 */
[[nodiscard]] auto is_cuda_available () noexcept -> bool;

/**
 * @brief Computes batch fp32 dot products between a matrix and query on GPU.
 * @param matrix Device Tensor of shape [num_rows, dim].
 * @param query Device Tensor of shape [dim].
 * @param out_scores Device Tensor of shape [num_rows] receiving scores.
 * @param num_rows Number of embedding rows in matrix.
 * @param dim Dimension of each embedding vector.
 * @param stream CUDA Stream on which to launch the kernel.
 * @pre matrix.element_count() == num_rows * dim, query.element_count() == dim,
 *      and out_scores.element_count() == num_rows.
 * @post out_scores contains dot products queued on stream.
 * @returns Result<void> or FrameworkErrorCode::CudaError.
 * @note Thread-safety: Safe for concurrent calls on distinct output tensors.
 */
[[nodiscard]] auto
gpu_batch_dot_f32 (const Tensor &matrix, const Tensor &query,
                   Tensor &out_scores, std::size_t num_rows, std::size_t dim,
                   const Stream &stream) noexcept -> Result<void>;

} /* namespace sutra::inline v1 */

#endif /* SUTRA_GPU_HPP */
