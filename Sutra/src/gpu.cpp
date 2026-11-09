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

#include <sutra/gpu.hpp>
#include <sutra/observability.hpp>

#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
#include "gpu/kernels.hpp"
#include <cuda_runtime.h>
#endif

#include <string>

#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define SUTRA_CUDA_CHECK(expr)                                                \
  if (const cudaError_t cuda_err_ = (expr); cuda_err_ != cudaSuccess)         \
    {                                                                         \
      std::string msg_ = "CUDA runtime operation failed: ";                   \
      msg_.append (cudaGetErrorString (cuda_err_));                           \
      Logger::log (LogLevel::Error, "gpu", msg_);                             \
      return std::unexpected (AgentError::framework (                         \
          FrameworkErrorCode::CudaError, static_cast<int> (cuda_err_),        \
          std::move (msg_)));                                                 \
    }
#endif

namespace sutra::inline v1
{

/* Constructs a Stream taking ownership of an opaque cudaStream_t handle.  */
Stream::Stream (void *handle) noexcept : native_handle_ (handle) {}

/* Move-constructs a Stream handle.  */
Stream::Stream (Stream &&other) noexcept
    : native_handle_ (other.native_handle_)
{
  other.native_handle_ = nullptr;
}

/* Move-assigns a Stream handle, destroying any existing stream.  */
auto
Stream::operator= (Stream &&other) noexcept -> Stream &
{
  if (this != &other)
    {
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
      if (native_handle_ != nullptr)
        {
          (void)cudaStreamDestroy (static_cast<cudaStream_t> (native_handle_));
        }
#endif
      native_handle_ = other.native_handle_;
      other.native_handle_ = nullptr;
    }
  return *this;
}

/* Destroys the owned CUDA stream.  */
Stream::~Stream () noexcept
{
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  if (native_handle_ != nullptr)
    {
      (void)cudaStreamDestroy (static_cast<cudaStream_t> (native_handle_));
      native_handle_ = nullptr;
    }
#endif
}

/* Creates a non-blocking CUDA stream via SUTRA_CUDA_CHECK.  */
auto
Stream::create () noexcept -> Result<Stream>
{
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  cudaStream_t raw_stream = nullptr;
  SUTRA_CUDA_CHECK (
      cudaStreamCreateWithFlags (&raw_stream, cudaStreamNonBlocking));
  return Stream (static_cast<void *> (raw_stream));
#else
  return std::unexpected (AgentError::framework (
      FrameworkErrorCode::CudaError, -1,
      "CUDA support was disabled at compile time; rebuild with "
      "SUTRA_ENABLE_CUDA=ON"));
#endif
}

/* Synchronizes the CUDA stream via SUTRA_CUDA_CHECK.  */
auto
Stream::synchronize () const noexcept -> Result<void>
{
  if (native_handle_ == nullptr)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Cannot synchronize an uninitialized CUDA Stream handle"));
    }
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  SUTRA_CUDA_CHECK (
      cudaStreamSynchronize (static_cast<cudaStream_t> (native_handle_)));
  return {};
#else
  return std::unexpected (
      AgentError::framework (FrameworkErrorCode::CudaError, -1,
                             "CUDA support was disabled at compile time"));
#endif
}

/* Returns the opaque stream pointer.  */
auto
Stream::native_handle () const noexcept -> void *
{
  return native_handle_;
}

/* Constructs a Tensor taking ownership of device memory.  */
Tensor::Tensor (void *ptr, std::size_t count) noexcept
    : device_ptr_ (ptr), element_count_ (count)
{
}

/* Move-constructs a Tensor handle.  */
Tensor::Tensor (Tensor &&other) noexcept
    : device_ptr_ (other.device_ptr_), element_count_ (other.element_count_)
{
  other.device_ptr_ = nullptr;
  other.element_count_ = 0U;
}

/* Move-assigns a Tensor handle, freeing any previously held device memory.  */
auto
Tensor::operator= (Tensor &&other) noexcept -> Tensor &
{
  if (this != &other)
    {
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
      if (device_ptr_ != nullptr)
        {
          (void)cudaFree (device_ptr_);
        }
#endif
      device_ptr_ = other.device_ptr_;
      element_count_ = other.element_count_;
      other.device_ptr_ = nullptr;
      other.element_count_ = 0U;
    }
  return *this;
}

/* Frees the owned device buffer.  */
Tensor::~Tensor () noexcept
{
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  if (device_ptr_ != nullptr)
    {
      (void)cudaFree (device_ptr_);
      device_ptr_ = nullptr;
      element_count_ = 0U;
    }
#endif
}

/* Allocates a device fp32 buffer via SUTRA_CUDA_CHECK.  */
auto
Tensor::allocate (std::size_t element_count) noexcept -> Result<Tensor>
{
  if (element_count == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Tensor::allocate requires a positive element count"));
    }
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  void *raw_ptr = nullptr;
  SUTRA_CUDA_CHECK (cudaMalloc (&raw_ptr, element_count * sizeof (float)));
  return Tensor (raw_ptr, element_count);
#else
  return std::unexpected (
      AgentError::framework (FrameworkErrorCode::CudaError, -1,
                             "CUDA support was disabled at compile time"));
#endif
}

/* Copies host float span to device buffer asynchronously on stream.  */
auto
Tensor::copy_from_host (std::span<const float> src,
                        const Stream &stream) noexcept -> Result<void>
{
  if (device_ptr_ == nullptr || src.size () != element_count_)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Host source span size does not match Tensor element count"));
    }
  if (stream.native_handle () == nullptr)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Stream handle is uninitialized in Tensor::copy_from_host"));
    }
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  SUTRA_CUDA_CHECK (
      cudaMemcpyAsync (device_ptr_, src.data (),
                       element_count_ * sizeof (float), cudaMemcpyHostToDevice,
                       static_cast<cudaStream_t> (stream.native_handle ())));
  return {};
#else
  return std::unexpected (
      AgentError::framework (FrameworkErrorCode::CudaError, -1,
                             "CUDA support was disabled at compile time"));
#endif
}

/* Copies device buffer to host float span asynchronously on stream.  */
auto
Tensor::copy_to_host (std::span<float> dst,
                      const Stream &stream) const noexcept -> Result<void>
{
  if (device_ptr_ == nullptr || dst.size () != element_count_)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Host destination span size does not match Tensor element count"));
    }
  if (stream.native_handle () == nullptr)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Stream handle is uninitialized in Tensor::copy_to_host"));
    }
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  SUTRA_CUDA_CHECK (
      cudaMemcpyAsync (dst.data (), device_ptr_,
                       element_count_ * sizeof (float), cudaMemcpyDeviceToHost,
                       static_cast<cudaStream_t> (stream.native_handle ())));
  return {};
#else
  return std::unexpected (
      AgentError::framework (FrameworkErrorCode::CudaError, -1,
                             "CUDA support was disabled at compile time"));
#endif
}

/* Returns the number of allocated float elements.  */
auto
Tensor::element_count () const noexcept -> std::size_t
{
  return element_count_;
}

/* Returns the opaque device memory pointer.  */
auto
Tensor::device_ptr () const noexcept -> void *
{
  return device_ptr_;
}

/* Queries whether at least one CUDA device is online.  */
auto
is_cuda_available () noexcept -> bool
{
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount (&device_count);
  return (err == cudaSuccess) && (device_count > 0);
#else
  return false;
#endif
}

/* Dispatches the GPU batch dot-product kernel via SUTRA_CUDA_CHECK.  */
auto
gpu_batch_dot_f32 (const Tensor &matrix, const Tensor &query,
                   Tensor &out_scores, std::size_t num_rows, std::size_t dim,
                   const Stream &stream) noexcept -> Result<void>
{
  if (num_rows == 0U || dim == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "gpu_batch_dot_f32 requires non-zero num_rows and dim"));
    }
  if (matrix.element_count () != num_rows * dim
      || query.element_count () != dim
      || out_scores.element_count () != num_rows)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Tensor dimensions do not match num_rows and dim in "
          "gpu_batch_dot_f32"));
    }
  if (stream.native_handle () == nullptr)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Stream handle is uninitialized in gpu_batch_dot_f32"));
    }
#if defined(SUTRA_HAS_CUDA) && SUTRA_HAS_CUDA
  const int launch_rc = detail::launch_batch_dot_f32_kernel (
      static_cast<const float *> (matrix.device_ptr ()),
      static_cast<const float *> (query.device_ptr ()),
      static_cast<float *> (out_scores.device_ptr ()), num_rows, dim,
      stream.native_handle ());
  SUTRA_CUDA_CHECK (static_cast<cudaError_t> (launch_rc));
  return {};
#else
  return std::unexpected (
      AgentError::framework (FrameworkErrorCode::CudaError, -1,
                             "CUDA support was disabled at compile time"));
#endif
}

} /* namespace sutra::inline v1 */
