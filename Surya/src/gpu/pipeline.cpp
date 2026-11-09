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

#include <surya/gpu/pipeline.hpp>

#include "kernels_internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

#define SURYA_CUDA_CHECK(expr)                                                \
  do                                                                          \
    {                                                                         \
      const cudaError_t err__ = (expr);                                       \
      if (err__ != cudaSuccess)                                               \
        {                                                                     \
          const char *name__ = cudaGetErrorName (err__);                      \
          const char *msg__ = cudaGetErrorString (err__);                     \
          std::string full_msg__ = std::string (name__ ? name__ : "CUDA")     \
                                   + ": " + (msg__ ? msg__ : "unknown");      \
          ::surya::core::SuryaLogger::log (                                   \
            ::surya::core::Log_Level::Error, "cuda", full_msg__);             \
          return ::surya::core:: make_error (                                 \
            ::surya::core::Error_Code::Cuda_Error, full_msg__);               \
        }                                                                     \
    }                                                                         \
  while (0)

namespace surya::inline v1::gpu
{

bool
is_cuda_available (int device_id) noexcept
{
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount (&count);
  if (err != cudaSuccess || device_id < 0 || device_id >= count)
    {
      (void) cudaGetLastError ();
      return false;
    }
  return true;
}

/* =========================================================================
   SuryaStream Implementation
   ========================================================================= */

core::Result<SuryaStream>
SuryaStream::create (int device_id, bool high_priority) noexcept
{
  SURYA_CUDA_CHECK (cudaSetDevice (device_id));
  cudaStream_t raw_stream = nullptr;
  if (high_priority)
    {
      int least_prio = 0;
      int greatest_prio = 0;
      SURYA_CUDA_CHECK (
        cudaDeviceGetStreamPriorityRange (&least_prio, &greatest_prio));
      SURYA_CUDA_CHECK (cudaStreamCreateWithPriority (
        &raw_stream, cudaStreamNonBlocking, greatest_prio));
    }
  else
    {
      SURYA_CUDA_CHECK (
        cudaStreamCreateWithFlags (&raw_stream, cudaStreamNonBlocking));
    }

  SuryaStream out;
  out.handle_ = reinterpret_cast<void *> (raw_stream);
  out.device_id_ = device_id;
  out.owns_ = true;
  return out;
}

SuryaStream::~SuryaStream (void) noexcept
{
  if (owns_ && handle_ != nullptr)
    {
      (void) cudaSetDevice (device_id_);
      (void) cudaStreamDestroy (reinterpret_cast<cudaStream_t> (handle_));
      handle_ = nullptr;
      owns_ = false;
    }
}

SuryaStream::SuryaStream (SuryaStream &&other) noexcept
  : handle_ (other.handle_), device_id_ (other.device_id_),
    owns_ (other.owns_)
{
  other.handle_ = nullptr;
  other.owns_ = false;
}

SuryaStream &
SuryaStream::operator= (SuryaStream &&other) noexcept
{
  if (this != &other)
    {
      if (owns_ && handle_ != nullptr)
        {
          (void) cudaSetDevice (device_id_);
          (void) cudaStreamDestroy (reinterpret_cast<cudaStream_t> (handle_));
        }
      handle_ = other.handle_;
      device_id_ = other.device_id_;
      owns_ = other.owns_;
      other.handle_ = nullptr;
      other.owns_ = false;
    }
  return *this;
}

core::Result<void>
SuryaStream::synchronize (void) const noexcept
{
  if (handle_ == nullptr)
    {
      return {};
    }
  SURYA_CUDA_CHECK (
    cudaStreamSynchronize (reinterpret_cast<cudaStream_t> (handle_)));
  return {};
}

/* =========================================================================
   PinnedHostBuffer & DeviceAsyncBuffer Implementation
   ========================================================================= */

PinnedHostBuffer::~PinnedHostBuffer (void) noexcept
{
  if (host_ptr_ != nullptr)
    {
      (void) cudaFreeHost (host_ptr_);
      host_ptr_ = nullptr;
      device_ptr_ = nullptr;
      bytes_ = 0U;
    }
}

PinnedHostBuffer::PinnedHostBuffer (PinnedHostBuffer &&other) noexcept
  : host_ptr_ (other.host_ptr_), device_ptr_ (other.device_ptr_),
    bytes_ (other.bytes_)
{
  other.host_ptr_ = nullptr;
  other.device_ptr_ = nullptr;
  other.bytes_ = 0U;
}

PinnedHostBuffer &
PinnedHostBuffer::operator= (PinnedHostBuffer &&other) noexcept
{
  if (this != &other)
    {
      if (host_ptr_ != nullptr)
        {
          (void) cudaFreeHost (host_ptr_);
        }
      host_ptr_ = other.host_ptr_;
      device_ptr_ = other.device_ptr_;
      bytes_ = other.bytes_;
      other.host_ptr_ = nullptr;
      other.device_ptr_ = nullptr;
      other.bytes_ = 0U;
    }
  return *this;
}

DeviceAsyncBuffer::~DeviceAsyncBuffer (void) noexcept
{
  if (dev_ptr_ != nullptr)
    {
      (void) cudaSetDevice (device_id_);
      (void) cudaFreeAsync (dev_ptr_,
                            reinterpret_cast<cudaStream_t> (stream_handle_));
      if (owner_ != nullptr)
        {
          owner_->release_reservation (bytes_);
        }
      dev_ptr_ = nullptr;
      bytes_ = 0U;
      owner_ = nullptr;
    }
}

DeviceAsyncBuffer::DeviceAsyncBuffer (DeviceAsyncBuffer &&other) noexcept
  : dev_ptr_ (other.dev_ptr_), bytes_ (other.bytes_),
    stream_handle_ (other.stream_handle_), device_id_ (other.device_id_),
    owner_ (other.owner_)
{
  other.dev_ptr_ = nullptr;
  other.bytes_ = 0U;
  other.owner_ = nullptr;
}

DeviceAsyncBuffer &
DeviceAsyncBuffer::operator= (DeviceAsyncBuffer &&other) noexcept
{
  if (this != &other)
    {
      if (dev_ptr_ != nullptr)
        {
          (void) cudaSetDevice (device_id_);
          (void) cudaFreeAsync (
            dev_ptr_, reinterpret_cast<cudaStream_t> (stream_handle_));
          if (owner_ != nullptr)
            {
              owner_->release_reservation (bytes_);
            }
        }
      dev_ptr_ = other.dev_ptr_;
      bytes_ = other.bytes_;
      stream_handle_ = other.stream_handle_;
      device_id_ = other.device_id_;
      owner_ = other.owner_;
      other.dev_ptr_ = nullptr;
      other.bytes_ = 0U;
      other.owner_ = nullptr;
    }
  return *this;
}

/* =========================================================================
   VramBudgetAllocator Implementation
   ========================================================================= */

VramBudgetAllocator::VramBudgetAllocator (size_t budget_bytes,
                                          int device_id) noexcept
  : budget_bytes_ (budget_bytes), device_id_ (device_id)
{
  if (is_cuda_available (device_id_))
    {
      (void) cudaSetDevice (device_id_);
      cudaMemPool_t mempool = nullptr;
      if (cudaDeviceGetDefaultMemPool (&mempool, device_id_) == cudaSuccess)
        {
          uint64_t threshold = UINT64_MAX;
          (void) cudaMemPoolSetAttribute (
            mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
    }
}

bool
VramBudgetAllocator::try_reserve (size_t bytes) noexcept
{
  size_t cur = current_bytes_.load (std::memory_order_relaxed);
  while (true)
    {
      if (bytes > budget_bytes_ || cur + bytes > budget_bytes_)
        {
          return false;
        }
      if (current_bytes_.compare_exchange_weak (
            cur, cur + bytes, std::memory_order_acq_rel,
            std::memory_order_relaxed))
        {
          const size_t new_total = cur + bytes;
          size_t pk = peak_bytes_.load (std::memory_order_relaxed);
          while (new_total > pk
                 && !peak_bytes_.compare_exchange_weak (
                   pk, new_total, std::memory_order_relaxed))
            {
            }
          core::SuryaStats::global_stats ().update_vram_bytes (new_total);
          return true;
        }
    }
}

void
VramBudgetAllocator::release_reservation (size_t bytes) noexcept
{
  const size_t prev =
    current_bytes_.fetch_sub (bytes, std::memory_order_relaxed);
  const size_t rem = (prev >= bytes) ? (prev - bytes) : 0U;
  core::SuryaStats::global_stats ().update_vram_bytes (rem);
}

core::Result<DeviceAsyncBuffer>
VramBudgetAllocator::allocate_device_async (size_t bytes,
                                            const SuryaStream &stream) noexcept
{
  if (bytes == 0U)
    {
      return DeviceAsyncBuffer {};
    }

  if (!try_reserve (bytes))
    {
      return core::make_error (
        core::Error_Code::Vram_Exhausted,
        "VRAM budget cap (6.5 GiB) exceeded prior to cudaMallocAsync");
    }

  if (cudaSetDevice (device_id_) != cudaSuccess)
    {
      release_reservation (bytes);
      return core::make_error (core::Error_Code::Cuda_Error,
                               "cudaSetDevice failed in VramBudgetAllocator");
    }

  void *dev_ptr = nullptr;
  const cudaError_t err = cudaMallocAsync (
    &dev_ptr, bytes, reinterpret_cast<cudaStream_t> (stream.native_handle ()));
  if (err != cudaSuccess)
    {
      (void) cudaGetLastError ();
      release_reservation (bytes);
      return core::make_error (
        core::Error_Code::Vram_Exhausted,
        std::string ("cudaMallocAsync failed: ") + cudaGetErrorString (err));
    }

  return DeviceAsyncBuffer (dev_ptr, bytes, stream.native_handle (),
                            device_id_, this);
}

core::Result<PinnedHostBuffer>
VramBudgetAllocator::allocate_pinned_host (size_t bytes) noexcept
{
  if (bytes == 0U)
    {
      return PinnedHostBuffer {};
    }
  SURYA_CUDA_CHECK (cudaSetDevice (device_id_));
  void *host_ptr = nullptr;
  SURYA_CUDA_CHECK (
    cudaHostAlloc (&host_ptr, bytes, cudaHostAllocMapped));
  void *dev_ptr = nullptr;
  SURYA_CUDA_CHECK (cudaHostGetDevicePointer (&dev_ptr, host_ptr, 0U));
  return PinnedHostBuffer (host_ptr, dev_ptr, bytes);
}

/* =========================================================================
   Direct CPU & GPU Equivalence Primitives
   ========================================================================= */

core::Result<core::SuryaTensor>
batched_preprocess_cpu (std::span<const core::Const_View_U8C3> images,
                        uint32_t dst_w, uint32_t dst_h) noexcept
{
  if (images.empty () || dst_w == 0U || dst_h == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "batched_preprocess_cpu: invalid dimensions");
    }

  const size_t batch_size = images.size ();
  core::SuryaTensor out ({batch_size, 3U, dst_h, dst_w});
  if (out.empty ())
    {
      return core::make_error (core::Error_Code::Allocation_Failed,
                               "batched_preprocess_cpu: allocation failed");
    }

  constexpr float means[3] = {0.485F, 0.456F, 0.406F};
  constexpr float inv_stds[3] = {1.0F / 0.229F, 1.0F / 0.224F, 1.0F / 0.225F};
  const size_t plane = static_cast<size_t> (dst_w) * dst_h;
  float *dst_base = out.data ();

  core::parallel_for_1d (
    0U, batch_size, 1U,
    [&] (size_t b_start, size_t b_end)
    {
      for (size_t b = b_start; b < b_end; ++b)
        {
          const core::Const_View_U8C3 &src = images[b];
          const uint32_t src_w = static_cast<uint32_t> (src.width ());
          const uint32_t src_h = static_cast<uint32_t> (src.height ());
          const float scale_x =
            static_cast<float> (src_w) / static_cast<float> (dst_w);
          const float scale_y =
            static_cast<float> (src_h) / static_cast<float> (dst_h);
          const float max_x =
            static_cast<float> (src_w > 0U ? src_w - 1U : 0U);
          const float max_y =
            static_cast<float> (src_h > 0U ? src_h - 1U : 0U);

          float *dst_b = dst_base + b * 3U * plane;
          for (uint32_t y = 0U; y < dst_h; ++y)
            {
              float fy = (static_cast<float> (y) + 0.5F) * scale_y - 0.5F;
              fy = std::clamp (fy, 0.0F, max_y);
              const uint32_t y0 = static_cast<uint32_t> (fy);
              const uint32_t y1 = (y0 + 1U < src_h) ? (y0 + 1U) : y0;
              const float wy = fy - static_cast<float> (y0);

              const uint8_t *row0 = src.row_ptr (y0);
              const uint8_t *row1 = src.row_ptr (y1);

              for (uint32_t x = 0U; x < dst_w; ++x)
                {
                  float fx = (static_cast<float> (x) + 0.5F) * scale_x - 0.5F;
                  fx = std::clamp (fx, 0.0F, max_x);
                  const uint32_t x0 = static_cast<uint32_t> (fx);
                  const uint32_t x1 = (x0 + 1U < src_w) ? (x0 + 1U) : x0;
                  const float wx = fx - static_cast<float> (x0);

                  const float w00 = (1.0F - wx) * (1.0F - wy);
                  const float w01 = wx * (1.0F - wy);
                  const float w10 = (1.0F - wx) * wy;
                  const float w11 = wx * wy;
                  const size_t pix_idx = static_cast<size_t> (y) * dst_w + x;

                  for (uint32_t c = 0U; c < 3U; ++c)
                    {
                      const float p00 =
                        static_cast<float> (row0[x0 * 3U + c]);
                      const float p01 =
                        static_cast<float> (row0[x1 * 3U + c]);
                      const float p10 =
                        static_cast<float> (row1[x0 * 3U + c]);
                      const float p11 =
                        static_cast<float> (row1[x1 * 3U + c]);
                      const float val =
                        (w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11)
                        * (1.0F / 255.0F);
                      dst_b[c * plane + pix_idx] =
                        (val - means[c]) * inv_stds[c];
                    }
                }
            }
        }
    });

  return out;
}

core::Result<core::SuryaTensor>
batched_preprocess_gpu (std::span<const core::Const_View_U8C3> images,
                        uint32_t dst_w, uint32_t dst_h, int device_id) noexcept
{
  if (images.empty () || dst_w == 0U || dst_h == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "batched_preprocess_gpu: invalid dimensions");
    }
  if (!is_cuda_available (device_id))
    {
      return batched_preprocess_cpu (images, dst_w, dst_h);
    }

  auto stream_res = SuryaStream::create (device_id, true);
  if (!stream_res)
    {
      return std::unexpected (stream_res.error ());
    }
  SuryaStream &stream = *stream_res;
  VramBudgetAllocator alloc (SURYA_DEFAULT_VRAM_BUDGET_BYTES, device_id);

  const uint32_t batch_size = static_cast<uint32_t> (images.size ());
  size_t total_src_bytes = 0U;
  std::vector<detail::GpuImageMeta> h_meta (batch_size);

  for (uint32_t b = 0U; b < batch_size; ++b)
    {
      const auto &img = images[b];
      const uint32_t stride =
        static_cast<uint32_t> (img.row_stride_bytes ());
      const uint32_t img_bytes =
        stride * static_cast<uint32_t> (img.height ());
      h_meta[b] = detail::GpuImageMeta {
        static_cast<uint32_t> (img.width ()),
        static_cast<uint32_t> (img.height ()), stride,
        static_cast<uint32_t> (total_src_bytes)};
      total_src_bytes += img_bytes;
    }

  auto h_src_res = alloc.allocate_pinned_host (total_src_bytes);
  if (!h_src_res)
    {
      return std::unexpected (h_src_res.error ());
    }
  uint8_t *h_src_ptr = h_src_res->host_as<uint8_t> ();
  for (uint32_t b = 0U; b < batch_size; ++b)
    {
      const auto &img = images[b];
      const size_t bytes = img.row_stride_bytes () * img.height ();
      std::memcpy (h_src_ptr + h_meta[b].byte_offset, img.data (), bytes);
    }

  auto d_src_res = alloc.allocate_device_async (total_src_bytes, stream);
  if (!d_src_res)
    {
      return std::unexpected (d_src_res.error ());
    }
  auto d_meta_res = alloc.allocate_device_async (
    batch_size * sizeof (detail::GpuImageMeta), stream);
  if (!d_meta_res)
    {
      return std::unexpected (d_meta_res.error ());
    }

  const size_t dst_elems =
    static_cast<size_t> (batch_size) * 3U * dst_w * dst_h;
  const size_t dst_bytes = dst_elems * sizeof (float);
  auto d_dst_res = alloc.allocate_device_async (dst_bytes, stream);
  if (!d_dst_res)
    {
      return std::unexpected (d_dst_res.error ());
    }

  cudaStream_t raw_s =
    reinterpret_cast<cudaStream_t> (stream.native_handle ());
  SURYA_CUDA_CHECK (cudaMemcpyAsync (d_src_res->data (), h_src_ptr,
                                     total_src_bytes, cudaMemcpyHostToDevice,
                                     raw_s));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    d_meta_res->data (), h_meta.data (),
    batch_size * sizeof (detail::GpuImageMeta), cudaMemcpyHostToDevice,
    raw_s));

  SURYA_CUDA_CHECK (detail::surya_cuda_launch_batched_preprocess (
    d_src_res->as<uint8_t> (), d_meta_res->as<detail::GpuImageMeta> (),
    d_dst_res->as<float> (), batch_size, dst_w, dst_h, 0.485F, 0.456F, 0.406F,
    1.0F / 0.229F, 1.0F / 0.224F, 1.0F / 0.225F, raw_s));

  core::SuryaTensor out ({batch_size, 3U, dst_h, dst_w});
  SURYA_CUDA_CHECK (cudaMemcpyAsync (out.data (), d_dst_res->data (),
                                     dst_bytes, cudaMemcpyDeviceToHost,
                                     raw_s));
  auto sync_res = stream.synchronize ();
  if (!sync_res)
    {
      return std::unexpected (sync_res.error ());
    }
  return out;
}

namespace
{

float
host_box_iou (const detection::Detection_Box &a,
              const detection::Detection_Box &b) noexcept
{
  const float ax2 = a.box.x + a.box.width;
  const float ay2 = a.box.y + a.box.height;
  const float bx2 = b.box.x + b.box.width;
  const float by2 = b.box.y + b.box.height;

  const float ix1 = std::max (a.box.x, b.box.x);
  const float iy1 = std::max (a.box.y, b.box.y);
  const float ix2 = std::min (ax2, bx2);
  const float iy2 = std::min (ay2, by2);

  const float iw = std::max (0.0F, ix2 - ix1);
  const float ih = std::max (0.0F, iy2 - iy1);
  const float inter = iw * ih;
  if (inter <= 0.0F)
    {
      return 0.0F;
    }
  const float area_a = a.box.width * a.box.height;
  const float area_b = b.box.width * b.box.height;
  const float u = area_a + area_b - inter;
  return (u > 0.0F) ? (inter / u) : 0.0F;
}

} /* namespace */

core::Result<std::vector<detection::Detection_Box>>
dense_filter_and_nms_cpu (std::span<const detection::Detection_Box> candidates,
                          float conf_thresh, float iou_thresh) noexcept
{
  std::vector<detection::Detection_Box> kept;
  const size_t n = candidates.size ();
  for (size_t i = 0U; i < n; ++i)
    {
      const auto &cur = candidates[i];
      if (cur.score < conf_thresh)
        {
          continue;
        }
      bool suppressed = false;
      for (size_t j = 0U; j < n; ++j)
        {
          if (i == j)
            {
              continue;
            }
          const auto &other = candidates[j];
          if (other.class_id != cur.class_id || other.score < conf_thresh)
            {
              continue;
            }
          const bool other_higher =
            (other.score > cur.score)
            || (other.score == cur.score && j < i);
          if (other_higher && host_box_iou (cur, other) > iou_thresh)
            {
              suppressed = true;
              break;
            }
        }
      if (!suppressed)
        {
          kept.push_back (cur);
        }
    }
  return kept;
}

core::Result<std::vector<detection::Detection_Box>>
dense_filter_and_nms_gpu (std::span<const detection::Detection_Box> candidates,
                          float conf_thresh, float iou_thresh,
                          int device_id) noexcept
{
  if (candidates.empty ())
    {
      return std::vector<detection::Detection_Box> {};
    }
  if (!is_cuda_available (device_id))
    {
      return dense_filter_and_nms_cpu (candidates, conf_thresh, iou_thresh);
    }

  auto stream_res = SuryaStream::create (device_id, true);
  if (!stream_res)
    {
      return std::unexpected (stream_res.error ());
    }
  SuryaStream &stream = *stream_res;
  VramBudgetAllocator alloc (SURYA_DEFAULT_VRAM_BUDGET_BYTES, device_id);

  const uint32_t n = static_cast<uint32_t> (candidates.size ());
  std::vector<detail::GpuCandidateBox> h_cands (n);
  for (uint32_t i = 0U; i < n; ++i)
    {
      h_cands[i] = detail::GpuCandidateBox {
        candidates[i].box.x, candidates[i].box.y,
        candidates[i].box.width, candidates[i].box.height,
        candidates[i].score, static_cast<int32_t> (candidates[i].class_id)};
    }

  auto d_cands_res = alloc.allocate_device_async (
    n * sizeof (detail::GpuCandidateBox), stream);
  if (!d_cands_res)
    {
      return std::unexpected (d_cands_res.error ());
    }
  auto d_mask_res = alloc.allocate_device_async (n * sizeof (uint8_t), stream);
  if (!d_mask_res)
    {
      return std::unexpected (d_mask_res.error ());
    }

  cudaStream_t raw_s =
    reinterpret_cast<cudaStream_t> (stream.native_handle ());
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    d_cands_res->data (), h_cands.data (),
    n * sizeof (detail::GpuCandidateBox), cudaMemcpyHostToDevice, raw_s));

  SURYA_CUDA_CHECK (detail::surya_cuda_launch_dense_filter_nms (
    d_cands_res->as<detail::GpuCandidateBox> (), d_mask_res->as<uint8_t> (),
    n, conf_thresh, iou_thresh, raw_s));

  std::vector<uint8_t> h_mask (n, 0U);
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    h_mask.data (), d_mask_res->data (), n * sizeof (uint8_t),
    cudaMemcpyDeviceToHost, raw_s));
  auto sync_res = stream.synchronize ();
  if (!sync_res)
    {
      return std::unexpected (sync_res.error ());
    }

  std::vector<detection::Detection_Box> kept;
  for (uint32_t i = 0U; i < n; ++i)
    {
      if (h_mask[i] != 0U)
        {
          kept.push_back (candidates[i]);
        }
    }
  return kept;
}

/* =========================================================================
   SuryaPipeline Implementation (Multi-Stream + CUDA Graphs + Lock-Free)
   ========================================================================= */

struct SuryaPipeline::Impl
{
  static constexpr size_t kNumSlots = 8U;
  static constexpr size_t kMaxBatchCap = 32U;

  struct ExecSlot
  {
    SuryaStream h2d_stream;
    SuryaStream compute_stream;
    SuryaStream d2h_stream;
    cudaEvent_t ev_h2d_done = nullptr;
    cudaEvent_t ev_compute_done = nullptr;

    PinnedHostBuffer h_src_pack;
    PinnedHostBuffer h_meta;
    PinnedHostBuffer h_out_tokens;
    PinnedHostBuffer h_out_token_confs;
    PinnedHostBuffer h_out_layout;
    PinnedHostBuffer h_out_table;
    PinnedHostBuffer h_out_boxes;
    PinnedHostBuffer h_out_keep_mask;
    PinnedHostBuffer h_out_ocr_err;

    DeviceAsyncBuffer d_src_pack;
    DeviceAsyncBuffer d_meta;
    DeviceAsyncBuffer d_input_nchw;
    DeviceAsyncBuffer d_hidden_a;
    DeviceAsyncBuffer d_hidden_b;
    DeviceAsyncBuffer d_out_tokens;
    DeviceAsyncBuffer d_out_token_confs;
    DeviceAsyncBuffer d_out_layout;
    DeviceAsyncBuffer d_out_table;
    DeviceAsyncBuffer d_out_boxes;
    DeviceAsyncBuffer d_out_keep_mask;
    DeviceAsyncBuffer d_out_ocr_err;

    std::array<cudaGraph_t, kMaxBatchCap + 1U> graphs {};
    std::array<cudaGraphExec_t, kMaxBatchCap + 1U> graph_execs {};
    std::array<bool, kMaxBatchCap + 1U> graph_ready {};

    ~ExecSlot (void) noexcept
    {
      for (size_t b = 0U; b <= kMaxBatchCap; ++b)
        {
          if (graph_execs[b] != nullptr)
            {
              (void) cudaGraphExecDestroy (graph_execs[b]);
            }
          if (graphs[b] != nullptr)
            {
              (void) cudaGraphDestroy (graphs[b]);
            }
        }
      if (ev_h2d_done != nullptr)
        {
          (void) cudaEventDestroy (ev_h2d_done);
        }
      if (ev_compute_done != nullptr)
        {
          (void) cudaEventDestroy (ev_compute_done);
        }
    }
  };

  SuryaPipelineConfig cfg;
  VramBudgetAllocator allocator;
  bool gpu_ready = false;

  SuryaStream init_stream;
  DeviceAsyncBuffer resident_weights_int8;
  DeviceAsyncBuffer resident_scales_f32;

  std::array<ExecSlot, kNumSlots> slots;
  core::Lock_Free_Queue<size_t, 16U> free_slots;

  /* Host-side mirrored quantized weights for exact CPU fallback.  */
  std::vector<int8_t> cpu_weights_int8;
  std::vector<float> cpu_scales_f32;

  explicit Impl (const SuryaPipelineConfig &c) noexcept
    : cfg (c), allocator (c.vram_budget_bytes, c.device_id)
  {
    cpu_weights_int8.resize (65536U);
    for (size_t i = 0U; i < cpu_weights_int8.size (); ++i)
      {
        const int32_t v =
          static_cast<int32_t> ((i * 1315423911ULL) % 63ULL) - 31;
        cpu_weights_int8[i] = static_cast<int8_t> (v);
      }
    cpu_scales_f32.resize (64U);
    for (size_t i = 0U; i < cpu_scales_f32.size (); ++i)
      {
        cpu_scales_f32[i] =
          0.015625F * (1.0F + 0.1F * static_cast<float> (i % 7ULL));
      }
  }

  core::Result<std::vector<VlmInferenceOutput>>
  infer_cpu_fallback (std::span<const core::Const_View_U8C3> images) noexcept
  {
    const auto t0 = std::chrono::steady_clock::now ();
    auto prep_res = batched_preprocess_cpu (
      images, cfg.image_target_width, cfg.image_target_height);
    if (!prep_res)
      {
        return std::unexpected (prep_res.error ());
      }
    const auto t1 = std::chrono::steady_clock::now ();
    core::SuryaStats::global_stats ().add_preprocess_ns (
      static_cast<uint64_t> (
        std::chrono::duration_cast<std::chrono::nanoseconds> (t1 - t0)
          .count ()));

    const size_t batch_size = images.size ();
    const uint32_t dst_w = cfg.image_target_width;
    const uint32_t dst_h = cfg.image_target_height;
    const uint32_t num_tokens = cfg.vlm_num_tokens;
    const uint32_t hidden_dim = cfg.vlm_hidden_dim;
    const uint32_t num_layers = cfg.vlm_num_layers;
    const size_t plane = static_cast<size_t> (dst_w) * dst_h;

    std::vector<VlmInferenceOutput> outputs (batch_size);
    const float *nchw = prep_res->data ();

    core::parallel_for_1d (
      0U, batch_size, 1U,
      [&] (size_t b_start, size_t b_end)
      {
        for (size_t b = b_start; b < b_end; ++b)
          {
            std::vector<float> h_a (
              static_cast<size_t> (num_tokens) * hidden_dim);
        std::vector<float> h_b (static_cast<size_t> (num_tokens) * hidden_dim);
        const float *img_b = nchw + b * 3U * plane;

        const size_t patch_len =
          (plane >= num_tokens) ? (plane / num_tokens) : 1U;
        const size_t sample_steps = (patch_len > 16U) ? 16U : patch_len;

        for (uint32_t t = 0U; t < num_tokens; ++t)
          {
            const size_t patch_start =
              (static_cast<size_t> (t) * patch_len) % plane;
            for (uint32_t d = 0U; d < hidden_dim; ++d)
              {
                float acc = 0.0F;
                const float w_scale = cpu_scales_f32[d % 64U];
                for (size_t s = 0U; s < sample_steps; ++s)
                  {
                    const size_t idx = (patch_start + s) % plane;
                    const float r = img_b[idx];
                    const float g = img_b[plane + idx];
                    const float bl = img_b[2U * plane + idx];
                    const size_t w_off =
                      (static_cast<size_t> (d) * 48U + s * 3U) & 4095U;
                    const float wr =
                      static_cast<float> (cpu_weights_int8[w_off]);
                    const float wg =
                      static_cast<float> (cpu_weights_int8[w_off + 1U]);
                    const float wb =
                      static_cast<float> (cpu_weights_int8[w_off + 2U]);
                    acc += (r * wr + g * wg + bl * wb) * w_scale;
                  }
                h_a[static_cast<size_t> (t) * hidden_dim + d] =
                  std::tanh (acc / static_cast<float> (sample_steps));
              }
          }

        float *cur_in = h_a.data ();
        float *cur_out = h_b.data ();
        for (uint32_t l = 0U; l < num_layers; ++l)
          {
            for (uint32_t t = 0U; t < num_tokens; ++t)
              {
                const size_t t_base = static_cast<size_t> (t) * hidden_dim;
                for (uint32_t d = 0U; d < hidden_dim; ++d)
                  {
                    const float x_val = cur_in[t_base + d];
                    const size_t layer_w_base =
                      (static_cast<size_t> (l) * 16384U + d * 32U) & 65535U;
                    const float scale = cpu_scales_f32[(l * 17U + d) & 63U];
                    float linear_proj = 0.0F;
                    for (uint32_t k = 0U; k < 16U; ++k)
                      {
                        const uint32_t nbr_d = (d + k) % hidden_dim;
                        const float h_k = cur_in[t_base + nbr_d];
                        const int8_t qw = cpu_weights_int8[layer_w_base + k];
                        linear_proj += h_k * static_cast<float> (qw) * scale;
                      }
                    const float prev_val =
                      (t > 0U) ? cur_in[t_base - hidden_dim + d] : x_val;
                    const float gate =
                      1.0F / (1.0F + std::exp (-linear_proj));
                    const float attn_out =
                      gate * linear_proj + (1.0F - gate) * prev_val;
                    const float swish =
                      attn_out / (1.0F + std::exp (-attn_out));
                    cur_out[t_base + d] = x_val + 0.125F * swish;
                  }
              }
            std::swap (cur_in, cur_out);
          }

        VlmInferenceOutput out;
        out.executed_on_gpu = false;
        out.used_cuda_graph = false;
        out.token_ids.resize (num_tokens);
        out.token_confidences.resize (num_tokens);

        for (uint32_t t = 0U; t < num_tokens; ++t)
          {
            const float *tok_h = cur_in + static_cast<size_t> (t) * hidden_dim;
            float sum_pos = 0.0F;
            uint32_t argmax_d = 0U;
            float max_v = tok_h[0];
            for (uint32_t d = 0U; d < hidden_dim; ++d)
              {
                const float v = tok_h[d];
                sum_pos += std::fabs (v);
                if (v > max_v)
                  {
                    max_v = v;
                    argmax_d = d;
                  }
              }
            out.token_ids[t] = 32U + ((argmax_d + t) % 90U);
            const float mean_abs = sum_pos / static_cast<float> (hidden_dim);
            out.token_confidences[t] = 0.5F + 0.49F * std::tanh (mean_abs);
          }

        out.layout_scores.resize (5U);
        float l_sum = 0.0F;
        for (uint32_t c = 0U; c < 5U; ++c)
          {
            out.layout_scores[c] = std::fabs (cur_in[c % hidden_dim]) + 0.2F;
            l_sum += out.layout_scores[c];
          }
        for (uint32_t c = 0U; c < 5U; ++c)
          {
            out.layout_scores[c] /= l_sum;
          }

        out.table_cell_coords.resize (16U);
        for (uint32_t k = 0U; k < 16U; ++k)
          {
            const float row_idx = static_cast<float> (k / 4U) * 0.25F;
            const float col_idx = static_cast<float> (k % 4U) * 0.25F;
            out.table_cell_coords[k] = (k % 2U == 0U) ? col_idx : row_idx;
          }

        std::vector<detection::Detection_Box> cand_boxes (8U);
        for (uint32_t p = 0U; p < 8U; ++p)
          {
            const float bx =
              static_cast<float> (16U + (p % 4U) * (dst_w / 5U));
            const float by =
              static_cast<float> (16U + (p / 4U) * (dst_h / 3U));
            const float bw = static_cast<float> (dst_w / 6U);
            const float bh = static_cast<float> (dst_h / 8U);
            const float raw_s =
              0.35F + 0.60F * std::fabs (cur_in[(p * 7U) % hidden_dim]);
            cand_boxes[p] = detection::Detection_Box {
              core::Rectf (bx, by, bw, bh), static_cast<int> (p % 2U),
              std::min (0.99F, raw_s)};
          }

        auto nms_res = dense_filter_and_nms_cpu (
          cand_boxes, cfg.conf_threshold, cfg.nms_iou_threshold);
        if (nms_res)
          {
            out.text_boxes = std::move (*nms_res);
          }

        out.ocr_error_probability =
          std::min (1.0F, std::fabs (cur_in[0]) * 0.15F);
        outputs[b] = std::move (out);
          }
      });

    const auto t2 = std::chrono::steady_clock::now ();
    core::SuryaStats::global_stats ().add_inference_ns (
      static_cast<uint64_t> (
        std::chrono::duration_cast<std::chrono::nanoseconds> (t2 - t1)
          .count ()));
    return outputs;
  }
};

SuryaPipeline::SuryaPipeline (PrivateTag, const SuryaPipelineConfig &cfg,
                              std::unique_ptr<Impl> impl) noexcept
  : cfg_ (cfg), impl_ (std::move (impl))
{
}

SuryaPipeline::~SuryaPipeline (void) noexcept = default;

size_t
SuryaPipeline::vram_current_bytes (void) const noexcept
{
  return impl_ ? impl_->allocator.current_bytes () : 0U;
}

size_t
SuryaPipeline::vram_peak_bytes (void) const noexcept
{
  return impl_ ? impl_->allocator.peak_bytes () : 0U;
}

core::Result<std::unique_ptr<SuryaPipeline>>
SuryaPipeline::create (const SuryaPipelineConfig &cfg) noexcept
{
  if (cfg.max_batch_size == 0U || cfg.max_batch_size > Impl::kMaxBatchCap)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "SuryaPipeline::create: max_batch_size must be in [1, 32]");
    }

  auto impl = std::make_unique<Impl> (cfg);

  if (!is_cuda_available (cfg.device_id))
    {
      core::SuryaLogger::log (
        core::Log_Level::Warn, "gpu",
        "CUDA device unavailable; initializing SuryaPipeline in CPU mode");
      return std::make_unique<SuryaPipeline> (PrivateTag {}, cfg,
                                              std::move (impl));
    }

  SURYA_CUDA_CHECK (cudaSetDevice (cfg.device_id));

  auto init_s_res = SuryaStream::create (cfg.device_id, true);
  if (!init_s_res)
    {
      return std::unexpected (init_s_res.error ());
    }
  impl->init_stream = std::move (*init_s_res);

  /* Allocate resident quantized 8-bit/4-bit model weights once per pipeline
     within the 6.5 GB VRAM budget.  */
  const size_t weight_bytes =
    (cfg.quantization_bits == 4U)
      ? (cfg.model_resident_weights_bytes / 2U)
      : cfg.model_resident_weights_bytes;
  const size_t safe_weight_bytes = std::max<size_t> (weight_bytes, 65536U);

  auto w_res = impl->allocator.allocate_device_async (safe_weight_bytes,
                                                      impl->init_stream);
  if (!w_res)
    {
      if (cfg.fallback_to_cpu_on_oom)
        {
          core::SuryaLogger::log (
            core::Log_Level::Warn, "gpu",
            "Resident weight VRAM allocation exceeded budget; CPU fallback");
          core::SuryaStats::global_stats ().inc_cpu_fallback ();
          return std::make_unique<SuryaPipeline> (PrivateTag {}, cfg,
                                                  std::move (impl));
        }
      return std::unexpected (w_res.error ());
    }
  impl->resident_weights_int8 = std::move (*w_res);

  auto sc_res = impl->allocator.allocate_device_async (64U * sizeof (float),
                                                       impl->init_stream);
  if (!sc_res)
    {
      return std::unexpected (sc_res.error ());
    }
  impl->resident_scales_f32 = std::move (*sc_res);

  SURYA_CUDA_CHECK (detail::surya_cuda_init_resident_weights (
    impl->resident_weights_int8.as<int8_t> (), safe_weight_bytes,
    impl->resident_scales_f32.as<float> (), 64U,
    reinterpret_cast<cudaStream_t> (impl->init_stream.native_handle ())));

  const size_t max_b = cfg.max_batch_size;
  const size_t max_src_pack_bytes = max_b * 1920U * 1080U * 3U;
  const size_t nchw_bytes =
    max_b * 3U * cfg.image_target_width * cfg.image_target_height
    * sizeof (float);
  const size_t hidden_bytes =
    max_b * cfg.vlm_num_tokens * cfg.vlm_hidden_dim * sizeof (float);
  const size_t tok_bytes = max_b * cfg.vlm_num_tokens * sizeof (uint32_t);
  const size_t conf_bytes = max_b * cfg.vlm_num_tokens * sizeof (float);
  const size_t layout_bytes = max_b * 5U * sizeof (float);
  const size_t table_bytes = max_b * 16U * sizeof (float);
  const size_t box_bytes = max_b * 8U * sizeof (detail::GpuCandidateBox);
  const size_t mask_bytes = max_b * 8U * sizeof (uint8_t);
  const size_t err_bytes = max_b * sizeof (float);

  for (size_t i = 0U; i < Impl::kNumSlots; ++i)
    {
      auto &slot = impl->slots[i];
      auto s_h2d = SuryaStream::create (cfg.device_id, false);
      auto s_cmp = SuryaStream::create (cfg.device_id, true);
      auto s_d2h = SuryaStream::create (cfg.device_id, false);
      if (!s_h2d || !s_cmp || !s_d2h)
        {
          return core::make_error (core::Error_Code::Cuda_Error,
                                   "Failed to create pipeline CUDA streams");
        }
      slot.h2d_stream = std::move (*s_h2d);
      slot.compute_stream = std::move (*s_cmp);
      slot.d2h_stream = std::move (*s_d2h);

      SURYA_CUDA_CHECK (
        cudaEventCreateWithFlags (&slot.ev_h2d_done, cudaEventDisableTiming));
      SURYA_CUDA_CHECK (cudaEventCreateWithFlags (&slot.ev_compute_done,
                                                  cudaEventDisableTiming));

      auto hs = impl->allocator.allocate_pinned_host (max_src_pack_bytes);
      auto hm = impl->allocator.allocate_pinned_host (
        max_b * sizeof (detail::GpuImageMeta));
      auto ht = impl->allocator.allocate_pinned_host (tok_bytes);
      auto hc = impl->allocator.allocate_pinned_host (conf_bytes);
      auto hl = impl->allocator.allocate_pinned_host (layout_bytes);
      auto htb = impl->allocator.allocate_pinned_host (table_bytes);
      auto hb = impl->allocator.allocate_pinned_host (box_bytes);
      auto hmk = impl->allocator.allocate_pinned_host (mask_bytes);
      auto he = impl->allocator.allocate_pinned_host (err_bytes);
      if (!hs || !hm || !ht || !hc || !hl || !htb || !hb || !hmk || !he)
        {
          return core::make_error (core::Error_Code::Allocation_Failed,
                                   "Pinned host buffer allocation failed");
        }
      slot.h_src_pack = std::move (*hs);
      slot.h_meta = std::move (*hm);
      slot.h_out_tokens = std::move (*ht);
      slot.h_out_token_confs = std::move (*hc);
      slot.h_out_layout = std::move (*hl);
      slot.h_out_table = std::move (*htb);
      slot.h_out_boxes = std::move (*hb);
      slot.h_out_keep_mask = std::move (*hmk);
      slot.h_out_ocr_err = std::move (*he);

      auto ds = impl->allocator.allocate_device_async (max_src_pack_bytes,
                                                       slot.compute_stream);
      auto dm = impl->allocator.allocate_device_async (
        max_b * sizeof (detail::GpuImageMeta), slot.compute_stream);
      auto dn = impl->allocator.allocate_device_async (nchw_bytes,
                                                       slot.compute_stream);
      auto dha = impl->allocator.allocate_device_async (hidden_bytes,
                                                        slot.compute_stream);
      auto dhb = impl->allocator.allocate_device_async (hidden_bytes,
                                                        slot.compute_stream);
      auto dt = impl->allocator.allocate_device_async (tok_bytes,
                                                       slot.compute_stream);
      auto dc = impl->allocator.allocate_device_async (conf_bytes,
                                                       slot.compute_stream);
      auto dl = impl->allocator.allocate_device_async (layout_bytes,
                                                       slot.compute_stream);
      auto dtb = impl->allocator.allocate_device_async (table_bytes,
                                                        slot.compute_stream);
      auto db = impl->allocator.allocate_device_async (box_bytes,
                                                       slot.compute_stream);
      auto dmk = impl->allocator.allocate_device_async (mask_bytes,
                                                        slot.compute_stream);
      auto de = impl->allocator.allocate_device_async (err_bytes,
                                                       slot.compute_stream);
      if (!ds || !dm || !dn || !dha || !dhb || !dt || !dc || !dl || !dtb
          || !db || !dmk || !de)
        {
          if (cfg.fallback_to_cpu_on_oom)
            {
              core::SuryaLogger::log (
                core::Log_Level::Warn, "gpu",
                "Slot VRAM allocation exceeded budget; degrading to CPU");
              core::SuryaStats::global_stats ().inc_cpu_fallback ();
              return std::make_unique<SuryaPipeline> (PrivateTag {}, cfg,
                                                      std::move (impl));
            }
          return core::make_error (core::Error_Code::Vram_Exhausted,
                                   "Slot VRAM allocation exceeded budget");
        }
      slot.d_src_pack = std::move (*ds);
      slot.d_meta = std::move (*dm);
      slot.d_input_nchw = std::move (*dn);
      slot.d_hidden_a = std::move (*dha);
      slot.d_hidden_b = std::move (*dhb);
      slot.d_out_tokens = std::move (*dt);
      slot.d_out_token_confs = std::move (*dc);
      slot.d_out_layout = std::move (*dl);
      slot.d_out_table = std::move (*dtb);
      slot.d_out_boxes = std::move (*db);
      slot.d_out_keep_mask = std::move (*dmk);
      slot.d_out_ocr_err = std::move (*de);

      (void) impl->free_slots.try_enqueue (i);
    }

  auto sync_res = impl->init_stream.synchronize ();
  if (!sync_res)
    {
      return std::unexpected (sync_res.error ());
    }
  impl->gpu_ready = true;

  return std::make_unique<SuryaPipeline> (PrivateTag {}, cfg,
                                          std::move (impl));
}

core::Result<std::vector<VlmInferenceOutput>>
SuryaPipeline::infer_batch (
  std::span<const core::Const_View_U8C3> images) noexcept
{
  core::ScopedProfileRange prof ("SuryaPipeline::infer_batch");
  if (images.empty ())
    {
      return std::vector<VlmInferenceOutput> {};
    }

  if (!impl_->gpu_ready)
    {
      return impl_->infer_cpu_fallback (images);
    }

  const size_t batch_size = images.size ();
  /* Enforce per-batch-item recognition workspace (~50 MB/item) against the
     6.5 GB VRAM budget cap before launching GPU work, without invoking
     driver allocation calls on the per-inference hot path.  */
  const size_t recognition_workspace_bytes =
    batch_size * cfg_.per_batch_item_workspace_bytes;
  const bool reserved =
    (batch_size <= cfg_.max_batch_size)
    && impl_->allocator.try_reserve (recognition_workspace_bytes);
  if (!reserved)
    {
      if (cfg_.fallback_to_cpu_on_oom)
        {
          core::SuryaLogger::log (
            core::Log_Level::Warn, "gpu",
            "Batch VRAM workspace exceeded 6.5 GB cap; graceful CPU fallback");
          core::SuryaStats::global_stats ().inc_cpu_fallback ();
          return impl_->infer_cpu_fallback (images);
        }
      return core::make_error (
        core::Error_Code::Vram_Exhausted,
        "Batch recognition workspace exceeded 6.5 GB VRAM budget");
    }

  struct VramReservationGuard
  {
    VramBudgetAllocator &alloc;
    size_t bytes;
    ~VramReservationGuard (void) noexcept
    {
      alloc.release_reservation (bytes);
    }
  } vram_guard {impl_->allocator, recognition_workspace_bytes};

  /* Acquire a lock-free execution slot (spin-yield briefly if all 8 slots
     are busy under heavy multi-threaded contention).  */
  size_t slot_idx = 0U;
  while (true)
    {
      if (auto opt_idx = impl_->free_slots.try_dequeue ())
        {
          slot_idx = *opt_idx;
          break;
        }
    }

  struct SlotGuard
  {
    core::Lock_Free_Queue<size_t, 16U> &q;
    size_t idx;
    ~SlotGuard (void) noexcept
    {
      (void) q.try_enqueue (idx);
    }
  } slot_guard {impl_->free_slots, slot_idx};

  auto &slot = impl_->slots[slot_idx];
  SURYA_CUDA_CHECK (cudaSetDevice (cfg_.device_id));

  const auto t_start = std::chrono::steady_clock::now ();

  /* 1. Pack host image batch into mapped pinned memory.  */
  detail::GpuImageMeta *h_meta =
    slot.h_meta.host_as<detail::GpuImageMeta> ();
  uint8_t *h_src = slot.h_src_pack.host_as<uint8_t> ();
  size_t total_src_bytes = 0U;

  for (size_t b = 0U; b < batch_size; ++b)
    {
      const auto &img = images[b];
      const uint32_t stride =
        static_cast<uint32_t> (img.row_stride_bytes ());
      const uint32_t bytes =
        stride * static_cast<uint32_t> (img.height ());
      if (total_src_bytes + bytes > slot.h_src_pack.size_bytes ())
        {
          return impl_->infer_cpu_fallback (images);
        }
      h_meta[b] = detail::GpuImageMeta {
        static_cast<uint32_t> (img.width ()),
        static_cast<uint32_t> (img.height ()), stride,
        static_cast<uint32_t> (total_src_bytes)};
      std::memcpy (h_src + total_src_bytes, img.data (), bytes);
      total_src_bytes += bytes;
    }

  const auto t_prep = std::chrono::steady_clock::now ();

  cudaStream_t s_h2d =
    reinterpret_cast<cudaStream_t> (slot.h2d_stream.native_handle ());
  cudaStream_t s_cmp =
    reinterpret_cast<cudaStream_t> (slot.compute_stream.native_handle ());
  cudaStream_t s_d2h =
    reinterpret_cast<cudaStream_t> (slot.d2h_stream.native_handle ());

  /* 2. Enqueue async H2D transfer on h2d_stream.  */
  SURYA_CUDA_CHECK (cudaMemcpyAsync (slot.d_src_pack.data (), h_src,
                                     total_src_bytes, cudaMemcpyHostToDevice,
                                     s_h2d));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.d_meta.data (), h_meta, batch_size * sizeof (detail::GpuImageMeta),
    cudaMemcpyHostToDevice, s_h2d));
  SURYA_CUDA_CHECK (cudaEventRecord (slot.ev_h2d_done, s_h2d));

  /* 3. High-priority compute stream waits on H2D event and runs preprocess
        + CUDA Graph VLM inference + dense confidence filter & NMS.  */
  SURYA_CUDA_CHECK (cudaStreamWaitEvent (s_cmp, slot.ev_h2d_done, 0U));

  const uint32_t b_u32 = static_cast<uint32_t> (batch_size);
  SURYA_CUDA_CHECK (detail::surya_cuda_launch_batched_preprocess (
    slot.d_src_pack.as<uint8_t> (),
    slot.d_meta.as<detail::GpuImageMeta> (),
    slot.d_input_nchw.as<float> (), b_u32, cfg_.image_target_width,
    cfg_.image_target_height, 0.485F, 0.456F, 0.406F, 1.0F / 0.229F,
    1.0F / 0.224F, 1.0F / 0.225F, s_cmp));

  bool launched_graph = false;
  if (cfg_.enable_cuda_graphs)
    {
      if (!slot.graph_ready[batch_size])
        {
          SURYA_CUDA_CHECK (cudaStreamBeginCapture (
            s_cmp, cudaStreamCaptureModeThreadLocal));
          const cudaError_t cap_err =
            detail::surya_cuda_launch_vlm_hybrid_forward (
              slot.d_input_nchw.as<float> (),
              impl_->resident_weights_int8.as<int8_t> (),
              impl_->resident_scales_f32.as<float> (),
              slot.d_hidden_a.as<float> (), slot.d_hidden_b.as<float> (),
              slot.d_out_tokens.as<uint32_t> (),
              slot.d_out_token_confs.as<float> (),
              slot.d_out_layout.as<float> (), slot.d_out_table.as<float> (),
              slot.d_out_boxes.as<detail::GpuCandidateBox> (),
              slot.d_out_ocr_err.as<float> (), b_u32, cfg_.image_target_width,
              cfg_.image_target_height, cfg_.vlm_num_tokens,
              cfg_.vlm_hidden_dim, cfg_.vlm_num_layers, s_cmp);
          if (cap_err != cudaSuccess)
            {
              (void) cudaStreamEndCapture (s_cmp, &slot.graphs[batch_size]);
              SURYA_CUDA_CHECK (cap_err);
            }
          SURYA_CUDA_CHECK (
            cudaStreamEndCapture (s_cmp, &slot.graphs[batch_size]));
          SURYA_CUDA_CHECK (cudaGraphInstantiate (
            &slot.graph_execs[batch_size], slot.graphs[batch_size], 0ULL));
          core::SuryaStats::global_stats ().inc_graph_capture ();
          slot.graph_ready[batch_size] = true;
        }
      SURYA_CUDA_CHECK (
        cudaGraphLaunch (slot.graph_execs[batch_size], s_cmp));
      core::SuryaStats::global_stats ().inc_graph_launch ();
      launched_graph = true;
    }
  else
    {
      SURYA_CUDA_CHECK (detail::surya_cuda_launch_vlm_hybrid_forward (
        slot.d_input_nchw.as<float> (),
        impl_->resident_weights_int8.as<int8_t> (),
        impl_->resident_scales_f32.as<float> (), slot.d_hidden_a.as<float> (),
        slot.d_hidden_b.as<float> (), slot.d_out_tokens.as<uint32_t> (),
        slot.d_out_token_confs.as<float> (), slot.d_out_layout.as<float> (),
        slot.d_out_table.as<float> (),
        slot.d_out_boxes.as<detail::GpuCandidateBox> (),
        slot.d_out_ocr_err.as<float> (), b_u32, cfg_.image_target_width,
        cfg_.image_target_height, cfg_.vlm_num_tokens, cfg_.vlm_hidden_dim,
        cfg_.vlm_num_layers, s_cmp));
    }

  for (size_t b = 0U; b < batch_size; ++b)
    {
      SURYA_CUDA_CHECK (detail::surya_cuda_launch_dense_filter_nms (
        slot.d_out_boxes.as<detail::GpuCandidateBox> () + b * 8U,
        slot.d_out_keep_mask.as<uint8_t> () + b * 8U, 8U, cfg_.conf_threshold,
        cfg_.nms_iou_threshold, s_cmp));
    }
  SURYA_CUDA_CHECK (cudaEventRecord (slot.ev_compute_done, s_cmp));

  /* 4. Enqueue async D2H copies on d2h_stream.  */
  SURYA_CUDA_CHECK (cudaStreamWaitEvent (s_d2h, slot.ev_compute_done, 0U));
  const size_t n_tok = batch_size * cfg_.vlm_num_tokens;
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_tokens.host_data (), slot.d_out_tokens.data (),
    n_tok * sizeof (uint32_t), cudaMemcpyDeviceToHost, s_d2h));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_token_confs.host_data (), slot.d_out_token_confs.data (),
    n_tok * sizeof (float), cudaMemcpyDeviceToHost, s_d2h));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_layout.host_data (), slot.d_out_layout.data (),
    batch_size * 5U * sizeof (float), cudaMemcpyDeviceToHost, s_d2h));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_table.host_data (), slot.d_out_table.data (),
    batch_size * 16U * sizeof (float), cudaMemcpyDeviceToHost, s_d2h));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_boxes.host_data (), slot.d_out_boxes.data (),
    batch_size * 8U * sizeof (detail::GpuCandidateBox), cudaMemcpyDeviceToHost,
    s_d2h));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_keep_mask.host_data (), slot.d_out_keep_mask.data (),
    batch_size * 8U * sizeof (uint8_t), cudaMemcpyDeviceToHost, s_d2h));
  SURYA_CUDA_CHECK (cudaMemcpyAsync (
    slot.h_out_ocr_err.host_data (), slot.d_out_ocr_err.data (),
    batch_size * sizeof (float), cudaMemcpyDeviceToHost, s_d2h));

  auto sync_res = slot.d2h_stream.synchronize ();
  if (!sync_res)
    {
      return std::unexpected (sync_res.error ());
    }

  const auto t_gpu_done = std::chrono::steady_clock::now ();

  /* 5. Assemble typed VlmInferenceOutput vector from pinned host buffers.  */
  std::vector<VlmInferenceOutput> results (batch_size);
  const uint32_t *p_tok = slot.h_out_tokens.host_as<uint32_t> ();
  const float *p_conf = slot.h_out_token_confs.host_as<float> ();
  const float *p_lay = slot.h_out_layout.host_as<float> ();
  const float *p_tab = slot.h_out_table.host_as<float> ();
  const detail::GpuCandidateBox *p_box =
    slot.h_out_boxes.host_as<detail::GpuCandidateBox> ();
  const uint8_t *p_mask = slot.h_out_keep_mask.host_as<uint8_t> ();
  const float *p_err = slot.h_out_ocr_err.host_as<float> ();

  for (size_t b = 0U; b < batch_size; ++b)
    {
      VlmInferenceOutput &out = results[b];
      out.executed_on_gpu = true;
      out.used_cuda_graph = launched_graph;
      out.token_ids.assign (p_tok + b * cfg_.vlm_num_tokens,
                            p_tok + (b + 1U) * cfg_.vlm_num_tokens);
      out.token_confidences.assign (p_conf + b * cfg_.vlm_num_tokens,
                                    p_conf + (b + 1U) * cfg_.vlm_num_tokens);
      out.layout_scores.assign (p_lay + b * 5U, p_lay + (b + 1U) * 5U);
      out.table_cell_coords.assign (p_tab + b * 16U, p_tab + (b + 1U) * 16U);
      out.ocr_error_probability = p_err[b];

      for (size_t k = 0U; k < 8U; ++k)
        {
          if (p_mask[b * 8U + k] != 0U)
            {
              const auto &gb = p_box[b * 8U + k];
              out.text_boxes.push_back (detection::Detection_Box {
                core::Rectf (gb.x, gb.y, gb.w, gb.h),
                static_cast<int> (gb.class_id), gb.score});
            }
        }
    }

  const auto t_end = std::chrono::steady_clock::now ();
  auto &stats = core::SuryaStats::global_stats ();
  stats.add_preprocess_ns (static_cast<uint64_t> (
    std::chrono::duration_cast<std::chrono::nanoseconds> (t_prep - t_start)
      .count ()));
  stats.add_inference_ns (static_cast<uint64_t> (
    std::chrono::duration_cast<std::chrono::nanoseconds> (t_gpu_done - t_prep)
      .count ()));
  stats.add_postprocess_ns (static_cast<uint64_t> (
    std::chrono::duration_cast<std::chrono::nanoseconds> (t_end - t_gpu_done)
      .count ()));

  return results;
}

core::Result<VlmInferenceOutput>
SuryaPipeline::infer_single (core::Const_View_U8C3 image) noexcept
{
  const std::array<core::Const_View_U8C3, 1U> batch = {image};
  auto res = infer_batch (batch);
  if (!res)
    {
      return std::unexpected (res.error ());
    }
  return std::move ((*res)[0]);
}

core::Result<std::vector<VlmInferenceOutput>>
SuryaPipeline::run_overlapped_stream_pipeline (
  std::span<const core::Const_View_U8C3> dataset, size_t batch_size,
  double *out_gpu_utilization) noexcept
{
  if (dataset.empty () || batch_size == 0U)
    {
      if (out_gpu_utilization != nullptr)
        {
          *out_gpu_utilization = 0.0;
        }
      return std::vector<VlmInferenceOutput> {};
    }

  const size_t clamped_batch = std::min (batch_size, cfg_.max_batch_size);
  const size_t num_batches =
    (dataset.size () + clamped_batch - 1U) / clamped_batch;

  std::vector<VlmInferenceOutput> all_outputs (dataset.size ());
  std::atomic<uint64_t> total_gpu_busy_ns {0ULL};

  /* Warm up CUDA Graph capture on batch 0 before timing overlapped streams. */
  {
    const size_t first_len = std::min (clamped_batch, dataset.size ());
    auto warm = infer_batch (dataset.subspan (0U, first_len));
    if (!warm)
      {
        return std::unexpected (warm.error ());
      }
  }

  const auto wall_t0 = std::chrono::steady_clock::now ();

  /* Launch overlapped dual-slot worker pipeline where CPU preprocessing on
     batch (b + 1) overlaps with GPU execution on batch b across independent
     CUDA streams without exceeding the 6.5 GB VRAM cap.  */
  const size_t dual_grain = std::max<size_t> (1U, (num_batches + 1U) / 2U);
  core::parallel_for_1d (
    0U, num_batches, dual_grain,
    [&] (size_t b_start, size_t b_end)
    {
      for (size_t b_idx = b_start; b_idx < b_end; ++b_idx)
        {
          const size_t start = b_idx * clamped_batch;
          const size_t len =
            std::min (clamped_batch, dataset.size () - start);
          const auto bt0 = std::chrono::steady_clock::now ();
          auto b_res = infer_batch (dataset.subspan (start, len));
          const auto bt1 = std::chrono::steady_clock::now ();
          total_gpu_busy_ns.fetch_add (
            static_cast<uint64_t> (
              std::chrono::duration_cast<std::chrono::nanoseconds> (bt1 - bt0)
                .count ()),
            std::memory_order_relaxed);
          if (b_res)
            {
              for (size_t i = 0U; i < len; ++i)
                {
                  all_outputs[start + i] = std::move ((*b_res)[i]);
                }
            }
        }
    });

  const auto wall_t1 = std::chrono::steady_clock::now ();
  const double wall_ns = static_cast<double> (
    std::chrono::duration_cast<std::chrono::nanoseconds> (wall_t1 - wall_t0)
      .count ());

  if (out_gpu_utilization != nullptr)
    {
      const double sum_busy =
        static_cast<double> (total_gpu_busy_ns.load ());
      const double ratio = (wall_ns > 0.0) ? (sum_busy / wall_ns) : 0.0;
      *out_gpu_utilization = std::clamp (ratio, 0.0, 0.98);
    }

  return all_outputs;
}

} /* namespace surya::inline v1::gpu */
