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

#ifndef SURYA_GPU_PIPELINE_HPP
#define SURYA_GPU_PIPELINE_HPP

#include <surya/core.hpp>
#include <surya/detection.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace surya::inline v1::gpu
{

/* Default VRAM budget is defined by SURYA_DEFAULT_VRAM_BUDGET_BYTES
   (6.5 GiB) in <surya/core/config.hpp>, leaving ~1.5 GiB headroom on an
   8 GB NVIDIA RTX 3070 for display and driver state.  */

/* Reference per-batch-item recognition workspace (~50 MiB).  */
inline constexpr size_t SURYA_VLM_PER_BATCH_ITEM_VRAM_BYTES =
  50ULL * 1024ULL * 1024ULL;

/* Query whether a CUDA-capable GPU is available at runtime.
   Allocation behavior: zero heap or device allocations.  */
bool is_cuda_available (int device_id = 0) noexcept;

/* Opaque RAII wrapper around a non-blocking cudaStream_t.
   Allocation behavior: creates one driver stream handle on construction.
   Teardown behavior: destroys the CUDA stream in ~SuryaStream ().  */
class SuryaStream
{
public:
  SuryaStream (void) noexcept = default;

  static core::Result<SuryaStream>
  create (int device_id = 0, bool high_priority = false) noexcept;

  ~SuryaStream (void) noexcept;

  SuryaStream (const SuryaStream &) = delete;
  SuryaStream &operator= (const SuryaStream &) = delete;

  SuryaStream (SuryaStream &&other) noexcept;
  SuryaStream &operator= (SuryaStream &&other) noexcept;

  bool
  valid (void) const noexcept
  {
    return handle_ != nullptr;
  }

  void *
  native_handle (void) const noexcept
  {
    return handle_;
  }

  int
  device_id (void) const noexcept
  {
    return device_id_;
  }

  /* Explicit user-facing barrier. Never called inside async hot paths.  */
  core::Result<void> synchronize (void) const noexcept;

private:
  void *handle_ = nullptr;
  int device_id_ = 0;
  bool owns_ = false;
};

/* RAII wrapper around page-locked mapped host memory allocated via
   cudaHostAlloc (..., cudaHostAllocMapped).
   Teardown behavior: automatically calls cudaFreeHost on destruction.  */
class PinnedHostBuffer
{
public:
  PinnedHostBuffer (void) noexcept = default;
  PinnedHostBuffer (void *host_ptr, void *device_ptr, size_t bytes) noexcept
    : host_ptr_ (host_ptr), device_ptr_ (device_ptr), bytes_ (bytes)
  {
  }

  ~PinnedHostBuffer (void) noexcept;

  PinnedHostBuffer (const PinnedHostBuffer &) = delete;
  PinnedHostBuffer &operator= (const PinnedHostBuffer &) = delete;

  PinnedHostBuffer (PinnedHostBuffer &&other) noexcept;
  PinnedHostBuffer &operator= (PinnedHostBuffer &&other) noexcept;

  void *
  host_data (void) noexcept
  {
    return host_ptr_;
  }

  const void *
  host_data (void) const noexcept
  {
    return host_ptr_;
  }

  template <typename T>
  T *
  host_as (void) noexcept
  {
    return static_cast<T *> (host_ptr_);
  }

  template <typename T>
  const T *
  host_as (void) const noexcept
  {
    return static_cast<const T *> (host_ptr_);
  }

  void *
  mapped_device_data (void) const noexcept
  {
    return device_ptr_;
  }

  size_t
  size_bytes (void) const noexcept
  {
    return bytes_;
  }

private:
  void *host_ptr_ = nullptr;
  void *device_ptr_ = nullptr;
  size_t bytes_ = 0U;
};

class VramBudgetAllocator;

/* RAII wrapper around stream-ordered device memory allocated via
   cudaMallocAsync.
   Teardown behavior: calls cudaFreeAsync on the associated stream and
   atomically decrements the parent VramBudgetAllocator usage counter.  */
class DeviceAsyncBuffer
{
public:
  DeviceAsyncBuffer (void) noexcept = default;
  DeviceAsyncBuffer (void *dev_ptr, size_t bytes, void *stream_handle,
                     int device_id, VramBudgetAllocator *owner) noexcept
    : dev_ptr_ (dev_ptr), bytes_ (bytes), stream_handle_ (stream_handle),
      device_id_ (device_id), owner_ (owner)
  {
  }

  ~DeviceAsyncBuffer (void) noexcept;

  DeviceAsyncBuffer (const DeviceAsyncBuffer &) = delete;
  DeviceAsyncBuffer &operator= (const DeviceAsyncBuffer &) = delete;

  DeviceAsyncBuffer (DeviceAsyncBuffer &&other) noexcept;
  DeviceAsyncBuffer &operator= (DeviceAsyncBuffer &&other) noexcept;

  void *
  data (void) noexcept
  {
    return dev_ptr_;
  }

  const void *
  data (void) const noexcept
  {
    return dev_ptr_;
  }

  template <typename T>
  T *
  as (void) noexcept
  {
    return static_cast<T *> (dev_ptr_);
  }

  template <typename T>
  const T *
  as (void) const noexcept
  {
    return static_cast<const T *> (dev_ptr_);
  }

  size_t
  size_bytes (void) const noexcept
  {
    return bytes_;
  }

private:
  void *dev_ptr_ = nullptr;
  size_t bytes_ = 0U;
  void *stream_handle_ = nullptr;
  int device_id_ = 0;
  VramBudgetAllocator *owner_ = nullptr;
};

/* Lock-free atomic VRAM budget tracker and stream-ordered pool allocator
   built on cudaMallocAsync and cudaHostAllocMapped.
   Enforces the 6.5 GB cap and fails fast with Error_Code::Vram_Exhausted.  */
class VramBudgetAllocator
{
public:
  explicit VramBudgetAllocator (
    size_t budget_bytes = SURYA_DEFAULT_VRAM_BUDGET_BYTES,
    int device_id = 0) noexcept;

  ~VramBudgetAllocator (void) noexcept = default;

  VramBudgetAllocator (const VramBudgetAllocator &) = delete;
  VramBudgetAllocator &operator= (const VramBudgetAllocator &) = delete;

  /* Allocate stream-ordered device memory via cudaMallocAsync.
     Returns Error_Code::Vram_Exhausted if current + bytes > budget.  */
  core::Result<DeviceAsyncBuffer>
  allocate_device_async (size_t bytes, const SuryaStream &stream) noexcept;

  /* Allocate mapped page-locked host memory via cudaHostAlloc.  */
  core::Result<PinnedHostBuffer>
  allocate_pinned_host (size_t bytes) noexcept;

  size_t
  current_bytes (void) const noexcept
  {
    return current_bytes_.load (std::memory_order_relaxed);
  }

  size_t
  peak_bytes (void) const noexcept
  {
    return peak_bytes_.load (std::memory_order_relaxed);
  }

  size_t
  budget_bytes (void) const noexcept
  {
    return budget_bytes_;
  }

  int
  device_id (void) const noexcept
  {
    return device_id_;
  }

  bool try_reserve (size_t bytes) noexcept;
  void release_reservation (size_t bytes) noexcept;

private:
  size_t budget_bytes_;
  int device_id_;
  std::atomic<size_t> current_bytes_ {0U};
  std::atomic<size_t> peak_bytes_ {0U};
};

/* Output of the unified Qwen3.5-650M Hybrid Linear-Attention VLM
   (OCR + layout + table recognition) and lightweight text-detection /
   OCR-error-detection components for one image.  */
struct VlmInferenceOutput
{
  std::vector<detection::Detection_Box> text_boxes;
  std::vector<uint32_t> token_ids;
  std::vector<float> token_confidences;
  std::vector<float> layout_scores;
  std::vector<float> table_cell_coords;
  float ocr_error_probability = 0.0F;
  bool executed_on_gpu = false;
  bool used_cuda_graph = false;
};

/* Configuration for SuryaPipeline.  */
struct SuryaPipelineConfig
{
  int device_id = 0;
  size_t vram_budget_bytes = SURYA_DEFAULT_VRAM_BUDGET_BYTES;
  size_t model_resident_weights_bytes = 256ULL * 1024ULL * 1024ULL;
  size_t per_batch_item_workspace_bytes = SURYA_VLM_PER_BATCH_ITEM_VRAM_BYTES;
  size_t max_batch_size = 16U;
  uint32_t image_target_width = 320U;
  uint32_t image_target_height = 320U;
  uint32_t vlm_hidden_dim = 256U;
  uint32_t vlm_num_layers = 8U;
  uint32_t vlm_num_tokens = 32U;
  float conf_threshold = 0.25F;
  float nms_iou_threshold = 0.45F;
  bool enable_cuda_graphs = true;
  uint8_t quantization_bits = 8U;
  bool fallback_to_cpu_on_oom = true;
};

/* Production CPU+GPU Hybrid Inference Pipeline.
   - Keeps INT8/INT4 quantized Qwen3.5-family hybrid linear-attention VLM
     weights resident in VRAM once per pipeline instance.
   - Manages multi-stream overlap (H2D stream, high-priority Compute stream,
     D2H stream), CUDA Graph capture & replay, and lock-free concurrency.
   - Enforces the 6.5 GB VRAM cap and gracefully degrades to CPU inference
     when VRAM is exhausted.  */
class SuryaPipeline
{
public:
  /* Create and initialize the pipeline, loading resident quantized weights
     into VRAM and pre-allocating lock-free execution contexts.  */
  static core::Result<std::unique_ptr<SuryaPipeline>>
  create (const SuryaPipelineConfig &cfg = SuryaPipelineConfig {}) noexcept;

  ~SuryaPipeline (void) noexcept;

  SuryaPipeline (const SuryaPipeline &) = delete;
  SuryaPipeline &operator= (const SuryaPipeline &) = delete;

  /* Thread-safe batched hybrid inference. Supports N concurrent callers
     without std::mutex in the hot path via lock-free execution slots.  */
  core::Result<std::vector<VlmInferenceOutput>>
  infer_batch (std::span<const core::Const_View_U8C3> images) noexcept;

  /* Single-image convenience wrapper.  */
  core::Result<VlmInferenceOutput>
  infer_single (core::Const_View_U8C3 image) noexcept;

  /* Execute sustained overlapped CPU-preprocess + GPU-inference pipeline
     across a stream of images, optionally reporting the measured GPU
     active utilization fraction in [0.0, 1.0].  */
  core::Result<std::vector<VlmInferenceOutput>>
  run_overlapped_stream_pipeline (
    std::span<const core::Const_View_U8C3> dataset, size_t batch_size,
    double *out_gpu_utilization = nullptr) noexcept;

  const SuryaPipelineConfig &
  config (void) const noexcept
  {
    return cfg_;
  }

  size_t vram_current_bytes (void) const noexcept;
  size_t vram_peak_bytes (void) const noexcept;

  struct Impl;
  struct PrivateTag
  {
  };

  explicit SuryaPipeline (PrivateTag, const SuryaPipelineConfig &cfg,
                          std::unique_ptr<Impl> impl) noexcept;

private:
  SuryaPipelineConfig cfg_;
  std::unique_ptr<Impl> impl_;
};

/* Direct GPU vs CPU equivalence kernels exposed for numerical verification.
   1. Batched NHWC uint8 -> NCHW float32 bilinear resize + ImageNet normalize.
   2. Dense GPU confidence filtering + parallel NMS vs CPU reference.  */
core::Result<core::SuryaTensor>
batched_preprocess_gpu (std::span<const core::Const_View_U8C3> images,
                        uint32_t dst_w, uint32_t dst_h,
                        int device_id = 0) noexcept;

core::Result<core::SuryaTensor>
batched_preprocess_cpu (std::span<const core::Const_View_U8C3> images,
                        uint32_t dst_w, uint32_t dst_h) noexcept;

core::Result<std::vector<detection::Detection_Box>>
dense_filter_and_nms_gpu (std::span<const detection::Detection_Box> candidates,
                          float conf_thresh, float iou_thresh,
                          int device_id = 0) noexcept;

core::Result<std::vector<detection::Detection_Box>>
dense_filter_and_nms_cpu (std::span<const detection::Detection_Box> candidates,
                          float conf_thresh, float iou_thresh) noexcept;

} /* namespace surya::inline v1::gpu */

#endif /* SURYA_GPU_PIPELINE_HPP */
