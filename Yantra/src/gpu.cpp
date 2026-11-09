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

#include <yantra/gpu.hpp>

#include <algorithm>
#include <mutex>
#include <string>
#include <utility>

#if YANTRA_HAS_CUDA
#include <cuda_runtime.h>

#define YANTRA_CUDA_CHECK(expr)                                               \
  do                                                                          \
    {                                                                         \
      const cudaError_t err__ = (expr);                                       \
      if (err__ != cudaSuccess)                                               \
        {                                                                     \
          return std::unexpected (Error::index (                              \
              IndexErrorCode::GpuOutOfMemory, 0U, 0U,                         \
              std::string ("CUDA runtime call failed: ")                      \
                  + cudaGetErrorString (err__)));                             \
        }                                                                     \
    }                                                                         \
  while (false)

namespace yantra::inline v1::gpu_detail
{
auto launch_batch_l2_f32 (const float *d_vectors, const float *d_queries,
                          float *d_out_dists, std::size_t num_vectors,
                          std::size_t num_queries, std::size_t dim,
                          cudaStream_t stream) noexcept -> cudaError_t;

auto launch_cagra_search (const float *d_vectors, const uint32_t *d_neighbors,
                          const float *d_queries, uint64_t *d_out_ids,
                          float *d_out_dists, std::size_t num_vectors,
                          std::size_t num_queries, std::size_t dim,
                          std::size_t degree, std::size_t k_out,
                          cudaStream_t stream) noexcept -> cudaError_t;
} /* namespace yantra::inline v1::gpu_detail */
#endif

namespace yantra::inline v1
{

auto
is_cuda_available () noexcept -> bool
{
#if YANTRA_HAS_CUDA
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount (&count);
  return (err == cudaSuccess && count > 0);
#else
  return false;
#endif
}

GpuStream::GpuStream (void *handle) noexcept : native_handle_ (handle) {}

GpuStream::GpuStream (GpuStream &&other) noexcept
    : native_handle_ (other.native_handle_)
{
  other.native_handle_ = nullptr;
}

auto
GpuStream::operator= (GpuStream &&other) noexcept -> GpuStream &
{
  if (this != &other)
    {
#if YANTRA_HAS_CUDA
      if (native_handle_ != nullptr)
        {
          (void)cudaStreamDestroy (
              static_cast<cudaStream_t> (native_handle_));
        }
#endif
      native_handle_ = other.native_handle_;
      other.native_handle_ = nullptr;
    }
  return *this;
}

GpuStream::~GpuStream () noexcept
{
#if YANTRA_HAS_CUDA
  if (native_handle_ != nullptr)
    {
      (void)cudaStreamDestroy (static_cast<cudaStream_t> (native_handle_));
      native_handle_ = nullptr;
    }
#endif
}

auto
GpuStream::create () noexcept -> Result<GpuStream>
{
#if YANTRA_HAS_CUDA
  if (!is_cuda_available ())
    {
      return std::unexpected (Error::index (
          IndexErrorCode::GpuOutOfMemory, 0U, 0U,
          "No CUDA-capable GPU device is available on this host"));
    }
  cudaMemPool_t mempool = nullptr;
  if (cudaDeviceGetDefaultMemPool (&mempool, 0) == cudaSuccess)
    {
      uint64_t threshold = UINT64_MAX;
      (void)cudaMemPoolSetAttribute (mempool, cudaMemPoolAttrReleaseThreshold,
                                     &threshold);
    }
  cudaStream_t stream = nullptr;
  YANTRA_CUDA_CHECK (
      cudaStreamCreateWithFlags (&stream, cudaStreamNonBlocking));
  return GpuStream (static_cast<void *> (stream));
#else
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuStream::synchronize () const noexcept -> Result<void>
{
#if YANTRA_HAS_CUDA
  if (native_handle_ == nullptr)
    {
      return std::unexpected (
          Error::index (IndexErrorCode::InvalidParameter, 0U, 0U,
                        "Cannot synchronize an uninitialized GpuStream"));
    }
  YANTRA_CUDA_CHECK (
      cudaStreamSynchronize (static_cast<cudaStream_t> (native_handle_)));
  return {};
#else
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuStream::native_handle () const noexcept -> void *
{
  return native_handle_;
}

GpuBuffer::GpuBuffer (void *ptr, std::size_t bytes,
                      void *stream_handle) noexcept
    : device_ptr_ (ptr), size_bytes_ (bytes), stream_handle_ (stream_handle)
{
}

GpuBuffer::GpuBuffer (GpuBuffer &&other) noexcept
    : device_ptr_ (other.device_ptr_), size_bytes_ (other.size_bytes_),
      stream_handle_ (other.stream_handle_)
{
  other.device_ptr_ = nullptr;
  other.size_bytes_ = 0U;
  other.stream_handle_ = nullptr;
}

auto
GpuBuffer::operator= (GpuBuffer &&other) noexcept -> GpuBuffer &
{
  if (this != &other)
    {
#if YANTRA_HAS_CUDA
      if (device_ptr_ != nullptr)
        {
          (void)cudaFreeAsync (device_ptr_,
                               static_cast<cudaStream_t> (stream_handle_));
        }
#endif
      device_ptr_ = other.device_ptr_;
      size_bytes_ = other.size_bytes_;
      stream_handle_ = other.stream_handle_;
      other.device_ptr_ = nullptr;
      other.size_bytes_ = 0U;
      other.stream_handle_ = nullptr;
    }
  return *this;
}

GpuBuffer::~GpuBuffer () noexcept
{
#if YANTRA_HAS_CUDA
  if (device_ptr_ != nullptr)
    {
      (void)cudaFreeAsync (device_ptr_,
                           static_cast<cudaStream_t> (stream_handle_));
      device_ptr_ = nullptr;
      size_bytes_ = 0U;
    }
#endif
}

auto
GpuBuffer::allocate_async (std::size_t byte_size,
                           const GpuStream &stream) noexcept
    -> Result<GpuBuffer>
{
#if YANTRA_HAS_CUDA
  if (byte_size == 0U || stream.native_handle () == nullptr)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::InvalidParameter, 0U, 0U,
          "GpuBuffer allocation requires non-zero size and valid GpuStream"));
    }
  void *ptr = nullptr;
  auto cu_stream = static_cast<cudaStream_t> (stream.native_handle ());
  YANTRA_CUDA_CHECK (cudaMallocAsync (&ptr, byte_size, cu_stream));
  return GpuBuffer (ptr, byte_size, stream.native_handle ());
#else
  (void)byte_size;
  (void)stream;
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuBuffer::copy_from_host_async (std::span<const std::byte> src,
                                 const GpuStream &stream) noexcept
    -> Result<void>
{
#if YANTRA_HAS_CUDA
  if (src.size () > size_bytes_ || device_ptr_ == nullptr)
    {
      return std::unexpected (
          Error::index (IndexErrorCode::InvalidParameter, size_bytes_,
                        src.size (), "Host source exceeds GpuBuffer size"));
    }
  YANTRA_CUDA_CHECK (cudaMemcpyAsync (
      device_ptr_, src.data (), src.size (), cudaMemcpyHostToDevice,
      static_cast<cudaStream_t> (stream.native_handle ())));
  return {};
#else
  (void)src;
  (void)stream;
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuBuffer::copy_to_host_async (std::span<std::byte> dst,
                               const GpuStream &stream) const noexcept
    -> Result<void>
{
#if YANTRA_HAS_CUDA
  if (dst.size () > size_bytes_ || device_ptr_ == nullptr)
    {
      return std::unexpected (
          Error::index (IndexErrorCode::InvalidParameter, size_bytes_,
                        dst.size (), "Host destination exceeds GpuBuffer"));
    }
  YANTRA_CUDA_CHECK (cudaMemcpyAsync (
      dst.data (), device_ptr_, dst.size (), cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t> (stream.native_handle ())));
  return {};
#else
  (void)dst;
  (void)stream;
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuBuffer::size_bytes () const noexcept -> std::size_t
{
  return size_bytes_;
}

auto
GpuBuffer::device_ptr () const noexcept -> void *
{
  return device_ptr_;
}

struct GpuIndex::Impl
{
  std::size_t num_vectors = 0U;
  std::size_t dim = 0U;
  std::size_t degree = 0U;
  std::size_t max_batch = 128U;
  std::size_t max_k = 32U;
  GpuStream stream;
  GpuBuffer d_vectors;
  GpuBuffer d_neighbors;
  GpuBuffer d_queries;
  GpuBuffer d_out_ids;
  GpuBuffer d_out_dists;
  GpuBuffer d_all_dists;
  mutable std::mutex query_mu;
};

GpuIndex::GpuIndex () noexcept = default;
GpuIndex::GpuIndex (GpuIndex &&other) noexcept = default;
auto GpuIndex::operator= (GpuIndex &&other) noexcept -> GpuIndex & = default;
GpuIndex::~GpuIndex () noexcept = default;

auto
GpuIndex::build (std::span<const float> vectors,
                 std::span<const uint32_t> graph_neighbors,
                 std::size_t num_vectors, std::size_t dim,
                 std::size_t degree) noexcept -> Result<GpuIndex>
{
#if YANTRA_HAS_CUDA
  if (num_vectors == 0U || dim == 0U || degree == 0U)
    {
      return std::unexpected (
          Error::index (IndexErrorCode::InvalidParameter, dim, 0U,
                        "GpuIndex::build requires non-zero dimensions"));
    }
  if (vectors.size () != num_vectors * dim
      || graph_neighbors.size () != num_vectors * degree)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::DimensionMismatch, num_vectors * dim,
          vectors.size (), "GpuIndex input span size mismatch"));
    }

  auto st_res = GpuStream::create ();
  if (!st_res.has_value ())
    {
      return std::unexpected (st_res.error ());
    }

  auto impl = std::make_unique<Impl> ();
  impl->num_vectors = num_vectors;
  impl->dim = dim;
  impl->degree = degree;
  impl->stream = std::move (*st_res);

  auto vec_buf = GpuBuffer::allocate_async (
      vectors.size_bytes (), impl->stream);
  if (!vec_buf.has_value ())
    {
      return std::unexpected (vec_buf.error ());
    }
  auto nbr_buf = GpuBuffer::allocate_async (
      graph_neighbors.size_bytes (), impl->stream);
  if (!nbr_buf.has_value ())
    {
      return std::unexpected (nbr_buf.error ());
    }
  auto q_buf = GpuBuffer::allocate_async (
      impl->max_batch * dim * sizeof (float), impl->stream);
  if (!q_buf.has_value ())
    {
      return std::unexpected (q_buf.error ());
    }
  auto ids_buf = GpuBuffer::allocate_async (
      impl->max_batch * impl->max_k * sizeof (uint64_t), impl->stream);
  if (!ids_buf.has_value ())
    {
      return std::unexpected (ids_buf.error ());
    }
  auto dist_buf = GpuBuffer::allocate_async (
      impl->max_batch * impl->max_k * sizeof (float), impl->stream);
  if (!dist_buf.has_value ())
    {
      return std::unexpected (dist_buf.error ());
    }
  auto all_dist_buf = GpuBuffer::allocate_async (
      impl->max_batch * num_vectors * sizeof (float), impl->stream);
  if (!all_dist_buf.has_value ())
    {
      return std::unexpected (all_dist_buf.error ());
    }

  impl->d_vectors = std::move (*vec_buf);
  impl->d_neighbors = std::move (*nbr_buf);
  impl->d_queries = std::move (*q_buf);
  impl->d_out_ids = std::move (*ids_buf);
  impl->d_out_dists = std::move (*dist_buf);
  impl->d_all_dists = std::move (*all_dist_buf);

  auto cp1 = impl->d_vectors.copy_from_host_async (
      std::as_bytes (vectors), impl->stream);
  if (!cp1.has_value ())
    {
      return std::unexpected (cp1.error ());
    }
  auto cp2 = impl->d_neighbors.copy_from_host_async (
      std::as_bytes (graph_neighbors), impl->stream);
  if (!cp2.has_value ())
    {
      return std::unexpected (cp2.error ());
    }
  auto sync_res = impl->stream.synchronize ();
  if (!sync_res.has_value ())
    {
      return std::unexpected (sync_res.error ());
    }

  GpuIndex idx;
  idx.impl_ = std::move (impl);
  return idx;
#else
  (void)vectors;
  (void)graph_neighbors;
  (void)num_vectors;
  (void)dim;
  (void)degree;
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuIndex::search (std::span<const float> query,
                  std::size_t k) const noexcept
    -> Result<std::vector<NeighborResult>>
{
  return search_batch (query, 1U, k);
}

auto
GpuIndex::search_batch (std::span<const float> queries,
                        std::size_t num_queries,
                        std::size_t k) const noexcept
    -> Result<std::vector<NeighborResult>>
{
#if YANTRA_HAS_CUDA
  if (!impl_ || impl_->num_vectors == 0U)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::IndexNotBuilt, 0U, 0U, "GpuIndex is not built"));
    }
  if (queries.size () != num_queries * impl_->dim)
    {
      return std::unexpected (
          Error::index (IndexErrorCode::DimensionMismatch,
                        num_queries * impl_->dim, queries.size (),
                        "Query span dimension mismatch in GpuIndex"));
    }
  if (num_queries == 0U || k == 0U || num_queries > impl_->max_batch
      || k > impl_->max_k)
    {
      return std::unexpected (Error::index (
          IndexErrorCode::InvalidParameter, impl_->max_batch, num_queries,
          "Batch size or k exceeds GpuIndex pre-allocated pool workspace"));
    }

  std::lock_guard<std::mutex> lock (impl_->query_mu);
  auto cu_stream = static_cast<cudaStream_t> (impl_->stream.native_handle ());

  YANTRA_CUDA_CHECK (cudaMemcpyAsync (
      impl_->d_queries.device_ptr (), queries.data (), queries.size_bytes (),
      cudaMemcpyHostToDevice, cu_stream));

  const std::size_t k_clamped = std::min (k, impl_->num_vectors);
  YANTRA_CUDA_CHECK (gpu_detail::launch_cagra_search (
      static_cast<const float *> (impl_->d_vectors.device_ptr ()),
      static_cast<const uint32_t *> (impl_->d_neighbors.device_ptr ()),
      static_cast<const float *> (impl_->d_queries.device_ptr ()),
      static_cast<uint64_t *> (impl_->d_out_ids.device_ptr ()),
      static_cast<float *> (impl_->d_out_dists.device_ptr ()),
      impl_->num_vectors, num_queries, impl_->dim, impl_->degree, k_clamped,
      cu_stream));

  std::vector<uint64_t> host_ids (num_queries * k_clamped);
  std::vector<float> host_dists (num_queries * k_clamped);

  YANTRA_CUDA_CHECK (cudaMemcpyAsync (
      host_ids.data (), impl_->d_out_ids.device_ptr (),
      host_ids.size () * sizeof (uint64_t), cudaMemcpyDeviceToHost,
      cu_stream));
  YANTRA_CUDA_CHECK (cudaMemcpyAsync (
      host_dists.data (), impl_->d_out_dists.device_ptr (),
      host_dists.size () * sizeof (float), cudaMemcpyDeviceToHost, cu_stream));
  YANTRA_CUDA_CHECK (cudaStreamSynchronize (cu_stream));

  std::vector<NeighborResult> out (num_queries * k_clamped);
  for (std::size_t i = 0U; i < out.size (); ++i)
    {
      out[i].id = host_ids[i];
      out[i].distance = host_dists[i];
    }
  return out;
#else
  (void)queries;
  (void)num_queries;
  (void)k;
  return std::unexpected (Error::index (
      IndexErrorCode::GpuOutOfMemory, 0U, 0U,
      "Yantra was compiled without CUDA support in this build preset"));
#endif
}

auto
GpuIndex::size () const noexcept -> std::size_t
{
  return impl_ ? impl_->num_vectors : 0U;
}

auto
GpuIndex::dimension () const noexcept -> std::size_t
{
  return impl_ ? impl_->dim : 0U;
}

} /* namespace yantra::inline v1 */
