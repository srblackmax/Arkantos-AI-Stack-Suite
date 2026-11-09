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

#ifndef BASALT_INTERNAL_BELLOWS_BRIDGE_HPP
#define BASALT_INTERNAL_BELLOWS_BRIDGE_HPP

/* Clearly-marked thin compatibility and extension adapter layer over Bellows
   (<bellows/bellows.hpp>) and its transitive Hyper Math engine.

   Bellows provides:
     - hyper_math::slab_buffer<T> (64-byte aligned refcounted CPU buffers)
     - hyper_math::tensor::lattice_descriptor and nd_tensor<T>
     - hyper_math::random_engine, thread_pool, parallel_for, matmul, linalg
     - bellows::data_table, preprocessors, metrics, and artifact framing

   This adapter extends Bellows with:
     1. Runtime-loaded CUDA driver, cuBLAS, and NVRTC JIT kernel dispatch
        (via dlopen of libcuda.so.1, libcublas.so.12, and libnvrtc.so.12)
        with a caching Unified Memory allocator and deterministic virtual
        device fallback.
     2. Size-bucketed caching slab allocator for Bellows_Storage on CPU and
        CUDA to eliminate per-tensor heap/driver allocation overhead.  */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <bellows/version.hpp>
#include <bellows/io/artifact.hpp>
#include <hyper_math/core/memory.hpp>
#include <hyper_math/core/parallel.hpp>
#include <hyper_math/tensor/lattice.hpp>
#include <hyper_math/tensor/nd_tensor.hpp>
#include <hyper_math/tensor/ufunc.hpp>
#include <hyper_math/tensor/reductions.hpp>
#include <hyper_math/tensor/manipulation.hpp>
#include <basalt/config.hpp>
#include <basalt/device.hpp>
#include <basalt/dtype.hpp>
#include <basalt/error.hpp>

namespace basalt::bellows_compat
{

/* Unary GPU kernel opcodes compiled via NVRTC.  */
enum class Cuda_Unary_Op : int
{
  None = -1,
  Neg = 0,
  Relu = 1,
  Sigmoid = 2,
  Tanh = 3,
  Silu = 4,
  Gelu = 5,
  Add_Scalar = 6,
  Mul_Scalar = 7,
  Exp = 8,
  Log = 9,
  Sqrt = 10,
  Square = 11,
  Abs = 12
};

/* Binary GPU kernel opcodes compiled via NVRTC.  */
enum class Cuda_Binary_Op : int
{
  None = -1,
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
  Relu_Backward = 4
};

/* Runtime-loaded CUDA driver, cuBLAS handle, NVRTC JIT module, and caching
   device memory manager.  */
class Cuda_Runtime_Driver
{
public:
  static Cuda_Runtime_Driver &
  instance (void) noexcept;

  bool
  has_hardware (void) const noexcept
  {
    return has_hw_driver_;
  }

  bool
  is_hw_device (size_t device_index) const noexcept
  {
    return has_hw_driver_ && device_index < hw_device_count_;
  }

  bool
  has_cublas (size_t device_index) const noexcept
  {
    return is_hw_device (device_index) && cublas_context_ != nullptr;
  }

  bool
  has_kernels (size_t device_index) const noexcept
  {
    return is_hw_device (device_index) && has_jit_kernels_;
  }

  size_t
  device_count (void) const noexcept;

  void
  set_virtual_device_count (size_t count) noexcept;

  void *
  allocate (size_t device_index, size_t byte_count);

  void
  deallocate (size_t device_index, void *ptr) noexcept;

  void
  memcpy_htod (size_t device_index, void *dst_dev,
               const void *src_host, size_t byte_count);

  void
  memcpy_dtoh (size_t device_index, void *dst_host,
               const void *src_dev, size_t byte_count);

  void
  memcpy_dtod (size_t dst_device, void *dst_dev,
               size_t src_device, const void *src_dev,
               size_t byte_count);

  void
  synchronize (size_t device_index);

  /* Fast atomic check: synchronizes physical GPU stream only if a GPU
     kernel was launched since the last host synchronization.  */
  void
  sync_if_busy (void) const noexcept
  {
    if (gpu_busy_.load (std::memory_order_acquire))
      {
        sync_hw_internal ();
      }
  }

  /* Execute row-major FP32 GEMM C = alpha * op(A) * op(B) + beta * C on
     physical GPU via cuBLAS.
     A has logical shape [m, k] (or [k, m] if trans_a),
     B has logical shape [k, n] (or [n, k] if trans_b),
     C has shape [m, n].  */
  bool
  cublas_sgemm (size_t device_index, bool trans_a, bool trans_b,
                size_t m, size_t n, size_t k, float alpha,
                const float *a_ptr, size_t lda,
                const float *b_ptr, size_t ldb, float beta,
                float *c_ptr, size_t ldc);

  /* Execute batched row-major FP32 GEMM via cublasSgemmStridedBatched.  */
  bool
  cublas_sgemm_batched (size_t device_index, bool trans_a, bool trans_b,
                        size_t m, size_t n, size_t k, float alpha,
                        const float *a_ptr, size_t lda, long long stride_a,
                        const float *b_ptr, size_t ldb, long long stride_b,
                        float beta, float *c_ptr, size_t ldc,
                        long long stride_c, size_t batch_count);

  /* Execute row-major FP16 or BF16 Tensor Core GEMM via cublasGemmEx with
     FP32 accumulation.  is_bf16 selects BF16 (true) vs FP16 (false).  */
  bool
  cublas_gemm_half (size_t device_index, bool is_bf16,
                    bool trans_a, bool trans_b,
                    size_t m, size_t n, size_t k, float alpha,
                    const void *a_ptr, size_t lda,
                    const void *b_ptr, size_t ldb, float beta,
                    void *c_ptr, size_t ldc);

  bool
  cublas_gemm_half_batched (size_t device_index, bool is_bf16,
                            bool trans_a, bool trans_b,
                            size_t m, size_t n, size_t k, float alpha,
                            const void *a_ptr, size_t lda, long long stride_a,
                            const void *b_ptr, size_t ldb, long long stride_b,
                            float beta, void *c_ptr, size_t ldc,
                            long long stride_c, size_t batch_count);

  /* Launch NVRTC-compiled FP32 GPU kernels on physical GPU.  */
  bool
  launch_fill_f32 (size_t device_index, float *dst, float val, size_t n);

  bool
  launch_unary_f32 (size_t device_index, const float *src, float *dst,
                    float param, Cuda_Unary_Op op, size_t n);

  bool
  launch_binary_f32 (size_t device_index, const float *a, const float *b,
                     float *c, Cuda_Binary_Op op, size_t n);

  bool
  launch_row_bcast_f32 (size_t device_index, const float *a, const float *b,
                        float *c, Cuda_Binary_Op op, size_t total,
                        size_t cols);

  bool
  launch_softmax_fwd_f32 (size_t device_index, const float *x, float *y,
                          size_t rows, size_t cols);

  bool
  launch_softmax_bwd_f32 (size_t device_index, const float *g, const float *y,
                          float *gx, size_t rows, size_t cols);

  bool
  launch_layernorm_fwd_f32 (size_t device_index, const float *x,
                            const float *w, const float *b, float *y,
                            size_t rows, size_t cols, float eps);

  bool
  launch_sum_last_f32 (size_t device_index, const float *x, float *out,
                       size_t outer_ext, size_t reduce_ext);

  bool
  launch_sgd_step_f32 (size_t device_index, float *p, const float *g,
                       float *buf, float lr, float wd, float mom, float damp,
                       bool nesterov, bool has_mom, bool init_buf, size_t n);

  bool
  launch_adam_step_f32 (size_t device_index, float *p, const float *g,
                        float *m, float *v, float lr, float b1, float b2,
                        float eps, float wd, float bc1, float bc2,
                        bool adamw, size_t n);

  bool
  launch_rmsnorm_fwd_f32 (size_t device_index, const float *x, const float *w,
                          float *y, size_t rows, size_t cols, float eps);

  bool
  launch_im2col_3d_f32 (size_t device_index, const float *in_n, float *col,
                        size_t c_per_grp, size_t in_d, size_t in_h,
                        size_t in_w, size_t kd, size_t kh, size_t kw,
                        size_t out_d, size_t out_h, size_t out_w,
                        size_t sd, size_t sh, size_t sw,
                        size_t pd, size_t ph, size_t pw,
                        size_t dd, size_t dh, size_t dw);

  bool
  launch_col2im_3d_f32 (size_t device_index, const float *dcol, float *gx_n,
                        size_t c_per_grp, size_t in_d, size_t in_h,
                        size_t in_w, size_t kd, size_t kh, size_t kw,
                        size_t out_d, size_t out_h, size_t out_w,
                        size_t sd, size_t sh, size_t sw,
                        size_t pd, size_t ph, size_t pw,
                        size_t dd, size_t dh, size_t dw);

private:
  Cuda_Runtime_Driver (void) noexcept;
  ~Cuda_Runtime_Driver (void) noexcept;

  void
  sync_hw_internal (void) const noexcept;

  void
  compile_jit_kernels (void) noexcept;

  void *libcuda_handle_;
  void *libcublas_handle_;
  void *libnvrtc_handle_;
  void *primary_ctx_;
  void *cublas_context_;
  void *jit_module_;
  void *fn_fill_f32_;
  void *fn_unary_f32_;
  void *fn_binary_f32_;
  void *fn_row_bcast_f32_;
  void *fn_softmax_fwd_f32_;
  void *fn_softmax_bwd_f32_;
  void *fn_layernorm_fwd_f32_;
  void *fn_sum_last_f32_;
  void *fn_sgd_step_f32_;
  void *fn_adam_step_f32_;
  void *fn_rmsnorm_fwd_f32_;
  void *fn_im2col_3d_f32_;
  void *fn_col2im_3d_f32_;
  bool has_hw_driver_;
  bool has_jit_kernels_;
  size_t hw_device_count_;
  std::atomic<size_t> virtual_device_count_;
  mutable std::atomic<bool> gpu_busy_;
  mutable std::mutex mutex_;

  struct Alloc_Info
  {
    size_t device_index;
    size_t bucket_bytes;
    bool is_managed_hw;
  };

  std::unordered_map<void *, Alloc_Info> active_allocs_;
  std::unordered_map<size_t, std::vector<void *>> hw_pool_;
  std::unordered_map<size_t, std::vector<void *>> host_pool_;
};

/* Reference-counted multi-dtype buffer backed by a pooled
   hyper_math::slab_buffer on CPU and Cuda_Runtime_Driver on CUDA devices.  */
class Bellows_Storage
{
public:
  Bellows_Storage (size_t byte_size, Device device);
  ~Bellows_Storage (void) noexcept;

  Bellows_Storage (const Bellows_Storage &) = delete;
  Bellows_Storage &operator= (const Bellows_Storage &) = delete;

  void *
  data (void) noexcept
  {
    return data_ptr_;
  }

  const void *
  data (void) const noexcept
  {
    return data_ptr_;
  }

  size_t
  byte_size (void) const noexcept
  {
    return byte_size_;
  }

  Device
  device (void) const noexcept
  {
    return device_;
  }

  void
  copy_to_host (void *dst_host, size_t byte_offset, size_t count) const;

  void
  copy_from_host (const void *src_host, size_t byte_offset, size_t count);

private:
  size_t byte_size_;
  size_t bucket_size_;
  Device device_;
  hyper_math::slab_buffer<uint8_t> cpu_slab_;
  void *data_ptr_;
};

} /* namespace basalt::bellows_compat */

#endif /* BASALT_INTERNAL_BELLOWS_BRIDGE_HPP */
