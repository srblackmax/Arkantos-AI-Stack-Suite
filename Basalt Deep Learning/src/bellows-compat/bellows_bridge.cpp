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

#include "config.h"
#include "bellows-compat/bellows_bridge.hpp"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define BASALT_INTERNAL_SANITIZER_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define BASALT_INTERNAL_SANITIZER_ACTIVE 1
#endif
#endif

namespace basalt::bellows_compat
{

namespace
{

using Cu_Init_Fn = int (*) (unsigned int);
using Cu_Device_Get_Count_Fn = int (*) (int *);
using Cu_Device_Get_Fn = int (*) (int *, int);
using Cu_Primary_Ctx_Retain_Fn = int (*) (void **, int);
using Cu_Primary_Ctx_Release_Fn = int (*) (int);
using Cu_Ctx_Set_Current_Fn = int (*) (void *);
using Cu_Ctx_Synchronize_Fn = int (*) (void);
using Cu_Mem_Alloc_Managed_Fn = int (*) (unsigned long long *, size_t,
                                         unsigned int);
using Cu_Mem_Free_Fn = int (*) (unsigned long long);
using Cu_Memset_D8_Fn = int (*) (unsigned long long, unsigned char, size_t);
using Cu_Module_Load_Data_Fn = int (*) (void **, const void *);
using Cu_Module_Unload_Fn = int (*) (void *);
using Cu_Module_Get_Function_Fn = int (*) (void **, void *, const char *);
using Cu_Launch_Kernel_Fn = int (*) (void *, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, void *,
                                     void **, void **);

using Cublas_Create_Fn = int (*) (void **);
using Cublas_Destroy_Fn = int (*) (void *);
using Cublas_Sgemm_Fn = int (*) (void *, int, int, int, int, int,
                                 const float *, const float *, int,
                                 const float *, int, const float *,
                                 float *, int);
using Cublas_Sgemm_Batched_Fn = int (*) (void *, int, int, int, int, int,
                                         const float *, const float *, int,
                                         long long, const float *, int,
                                         long long, const float *, float *,
                                         int, long long, int);
using Cublas_Gemm_Ex_Fn = int (*) (void *, int, int, int, int, int,
                                   const void *, const void *, int, int,
                                   const void *, int, int, const void *,
                                   void *, int, int, int, int);
using Cublas_Gemm_Batched_Ex_Fn = int (*) (void *, int, int, int, int, int,
                                           const void *, const void *, int,
                                           int, long long, const void *, int,
                                           int, long long, const void *,
                                           void *, int, int, long long, int,
                                           int, int);

using Nvrtc_Create_Prog_Fn = int (*) (void **, const char *, const char *,
                                      int, const char *const *,
                                      const char *const *);
using Nvrtc_Compile_Prog_Fn = int (*) (void *, int, const char *const *);
using Nvrtc_Get_Ptx_Size_Fn = int (*) (void *, size_t *);
using Nvrtc_Get_Ptx_Fn = int (*) (void *, char *);
using Nvrtc_Destroy_Prog_Fn = int (*) (void **);

struct Driver_Syms
{
  Cu_Ctx_Set_Current_Fn ctx_set_current = nullptr;
  Cu_Ctx_Synchronize_Fn ctx_sync = nullptr;
  Cu_Mem_Alloc_Managed_Fn mem_alloc_managed = nullptr;
  Cu_Mem_Free_Fn mem_free = nullptr;
  Cu_Memset_D8_Fn memset_d8 = nullptr;
  Cu_Primary_Ctx_Release_Fn primary_ctx_release = nullptr;
  Cu_Module_Load_Data_Fn module_load_data = nullptr;
  Cu_Module_Unload_Fn module_unload = nullptr;
  Cu_Module_Get_Function_Fn module_get_func = nullptr;
  Cu_Launch_Kernel_Fn launch_kernel = nullptr;
  Cublas_Create_Fn cublas_create = nullptr;
  Cublas_Destroy_Fn cublas_destroy = nullptr;
  Cublas_Sgemm_Fn cublas_sgemm = nullptr;
  Cublas_Sgemm_Batched_Fn cublas_sgemm_batched = nullptr;
  Cublas_Gemm_Ex_Fn cublas_gemm_ex = nullptr;
  Cublas_Gemm_Batched_Ex_Fn cublas_gemm_batched_ex = nullptr;
};

Driver_Syms g_syms;

size_t
round_up_bucket (size_t bytes) noexcept
{
  size_t b = 256;
  while (b < bytes && b < (size_t (1) << 30))
    {
      b <<= 1;
    }
  return (b >= bytes) ? b : ((bytes + 63U) & ~size_t (63U));
}

/* Process-wide CPU slab cache keyed by bucket byte size.  */
class Cpu_Slab_Pool
{
public:
  static Cpu_Slab_Pool &
  instance (void) noexcept
  {
    static Cpu_Slab_Pool inst;
    return inst;
  }

  hyper_math::slab_buffer<uint8_t>
  acquire (size_t bucket_bytes)
  {
    if (bucket_bytes <= (size_t (64) << 20))
      {
        std::lock_guard<std::mutex> lock (mutex_);
        auto it = buckets_.find (bucket_bytes);
        if (it != buckets_.end () && !it->second.empty ())
          {
            hyper_math::slab_buffer<uint8_t> buf
              = std::move (it->second.back ());
            it->second.pop_back ();
            return buf;
          }
      }
    return hyper_math::slab_buffer<uint8_t> (bucket_bytes);
  }

  void
  release (size_t bucket_bytes,
           hyper_math::slab_buffer<uint8_t> &&buf) noexcept
  {
    if (bucket_bytes <= (size_t (64) << 20) && buf.use_count () == 1)
      {
        std::lock_guard<std::mutex> lock (mutex_);
        auto &vec = buckets_[bucket_bytes];
        if (vec.size () < 64)
          {
            vec.push_back (std::move (buf));
          }
      }
  }

private:
  std::mutex mutex_;
  std::unordered_map<size_t, std::vector<hyper_math::slab_buffer<uint8_t>>>
    buckets_;
};

const char *const k_jit_cuda_source = R"CUDA(
extern "C" {

__global__ void
basalt_fill_f32 (float *dst, float val, size_t n)
{
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = val;
}

__global__ void
basalt_unary_f32 (const float *src, float *dst, float param, int op, size_t n)
{
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    {
      float x = src[i];
      float y = x;
      if (op == 0)
        y = -x;
      else if (op == 1)
        y = x > 0.0f ? x : 0.0f;
      else if (op == 2)
        y = 1.0f / (1.0f + expf (-x));
      else if (op == 3)
        y = tanhf (x);
      else if (op == 4)
        y = x / (1.0f + expf (-x));
      else if (op == 5)
        {
          float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
          y = 0.5f * x * (1.0f + tanhf (u));
        }
      else if (op == 6)
        y = x + param;
      else if (op == 7)
        y = x * param;
      else if (op == 8)
        y = expf (x);
      else if (op == 9)
        y = logf (x);
      else if (op == 10)
        y = sqrtf (x);
      else if (op == 11)
        y = x * x;
      else if (op == 12)
        y = fabsf (x);
      dst[i] = y;
    }
}

__global__ void
basalt_binary_f32 (const float *a, const float *b, float *c, int op, size_t n)
{
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    {
      float x = a[i];
      float y = b[i];
      if (op == 0)
        c[i] = x + y;
      else if (op == 1)
        c[i] = x - y;
      else if (op == 2)
        c[i] = x * y;
      else if (op == 3)
        c[i] = x / y;
      else if (op == 4)
        c[i] = y > 0.0f ? x : 0.0f;
    }
}

__global__ void
basalt_row_bcast_f32 (const float *a, const float *b, float *c, int op,
                      size_t total, size_t cols)
{
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total)
    {
      float x = a[i];
      float y = b[i % cols];
      if (op == 0)
        c[i] = x + y;
      else if (op == 1)
        c[i] = x - y;
      else if (op == 2)
        c[i] = x * y;
      else if (op == 3)
        c[i] = x / y;
    }
}

__global__ void
basalt_softmax_fwd_f32 (const float *x, float *y, size_t rows, size_t cols)
{
  __shared__ float sdata[256];
  size_t r = blockIdx.x;
  if (r >= rows || cols == 0)
    return;
  unsigned int tid = threadIdx.x;
  const float *xr = x + r * cols;
  float *yr = y + r * cols;
  float local_max = -1e38f;
  for (size_t c = tid; c < cols; c += blockDim.x)
    {
      float v = xr[c];
      if (v > local_max)
        local_max = v;
    }
  sdata[tid] = local_max;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s && sdata[tid + s] > sdata[tid])
        sdata[tid] = sdata[tid + s];
      __syncthreads ();
    }
  float max_v = sdata[0];
  float local_sum = 0.0f;
  for (size_t c = tid; c < cols; c += blockDim.x)
    {
      float e = expf (xr[c] - max_v);
      yr[c] = e;
      local_sum += e;
    }
  sdata[tid] = local_sum;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads ();
    }
  float inv_sum = 1.0f / sdata[0];
  for (size_t c = tid; c < cols; c += blockDim.x)
    yr[c] *= inv_sum;
}

__global__ void
basalt_softmax_bwd_f32 (const float *g, const float *y, float *gx,
                        size_t rows, size_t cols)
{
  __shared__ float sdata[256];
  size_t r = blockIdx.x;
  if (r >= rows || cols == 0)
    return;
  unsigned int tid = threadIdx.x;
  const float *gr = g + r * cols;
  const float *yr = y + r * cols;
  float *gxr = gx + r * cols;
  float local_dot = 0.0f;
  for (size_t c = tid; c < cols; c += blockDim.x)
    local_dot += gr[c] * yr[c];
  sdata[tid] = local_dot;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads ();
    }
  float dot_gy = sdata[0];
  for (size_t c = tid; c < cols; c += blockDim.x)
    gxr[c] = yr[c] * (gr[c] - dot_gy);
}

__global__ void
basalt_layernorm_fwd_f32 (const float *x, const float *w, const float *b,
                          float *y, size_t rows, size_t cols, float eps)
{
  __shared__ float sdata[256];
  size_t r = blockIdx.x;
  if (r >= rows || cols == 0)
    return;
  unsigned int tid = threadIdx.x;
  const float *xr = x + r * cols;
  float *yr = y + r * cols;
  float local_sum = 0.0f;
  for (size_t c = tid; c < cols; c += blockDim.x)
    local_sum += xr[c];
  sdata[tid] = local_sum;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads ();
    }
  float mean = sdata[0] / (float) cols;
  float local_var = 0.0f;
  for (size_t c = tid; c < cols; c += blockDim.x)
    {
      float d = xr[c] - mean;
      local_var += d * d;
    }
  sdata[tid] = local_var;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads ();
    }
  float inv_std = rsqrtf (sdata[0] / (float) cols + eps);
  for (size_t c = tid; c < cols; c += blockDim.x)
    {
      float val = (xr[c] - mean) * inv_std;
      if (w != 0)
        val *= w[c];
      if (b != 0)
        val += b[c];
      yr[c] = val;
    }
}

__global__ void
basalt_sum_last_f32 (const float *x, float *out, size_t outer_ext,
                     size_t reduce_ext)
{
  __shared__ float sdata[256];
  size_t o = blockIdx.x;
  if (o >= outer_ext)
    return;
  unsigned int tid = threadIdx.x;
  const float *row = x + o * reduce_ext;
  float local_sum = 0.0f;
  for (size_t r = tid; r < reduce_ext; r += blockDim.x)
    local_sum += row[r];
  sdata[tid] = local_sum;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads ();
    }
  if (tid == 0)
    out[o] = sdata[0];
}

__global__ void
basalt_sgd_step_f32 (float *p, const float *g, float *buf, float lr,
                     float wd, float mom, float damp, int nesterov,
                     int has_mom, int init_buf, size_t n)
{
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    {
      float pi = p[i];
      float dp = g[i] + wd * pi;
      if (has_mom != 0)
        {
          float bi = (init_buf != 0)
                     ? dp
                     : (mom * buf[i] + (1.0f - damp) * dp);
          buf[i] = bi;
          dp = (nesterov != 0) ? (dp + mom * bi) : bi;
        }
      p[i] = pi - lr * dp;
    }
}

__global__ void
basalt_adam_step_f32 (float *p, const float *g, float *m, float *v,
                      float lr, float b1, float b2, float eps, float wd,
                      float bc1, float bc2, int adamw, size_t n)
{
  size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    {
      float pi = p[i];
      float gi = g[i];
      if (adamw != 0 && wd != 0.0f)
        pi -= lr * wd * pi;
      else if (wd != 0.0f)
        gi += wd * pi;
      float mi = b1 * m[i] + (1.0f - b1) * gi;
      float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
      m[i] = mi;
      v[i] = vi;
      float m_hat = mi / bc1;
      float v_hat = vi / bc2;
      p[i] = pi - lr * m_hat / (sqrtf (v_hat) + eps);
    }
}

__global__ void
basalt_rmsnorm_fwd_f32 (const float *x, const float *w, float *y,
                        size_t rows, size_t cols, float eps)
{
  __shared__ float sdata[256];
  size_t r = blockIdx.x;
  if (r >= rows)
    return;
  unsigned int tid = threadIdx.x;
  const float *xr = x + r * cols;
  float *yr = y + r * cols;
  float sq = 0.0f;
  for (size_t c = tid; c < cols; c += blockDim.x)
    {
      float v = xr[c];
      sq += v * v;
    }
  sdata[tid] = sq;
  __syncthreads ();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads ();
    }
  float inv_rms = rsqrtf (sdata[0] / (float) cols + eps);
  for (size_t c = tid; c < cols; c += blockDim.x)
    {
      float v = xr[c] * inv_rms;
      if (w != (const float *) 0)
        v *= w[c];
      yr[c] = v;
    }
}

__global__ void
basalt_im2col_3d_f32 (const float *in_n, float *col,
                      size_t c_per_grp, size_t in_d, size_t in_h, size_t in_w,
                      size_t kd, size_t kh, size_t kw,
                      size_t out_d, size_t out_h, size_t out_w,
                      size_t sd, size_t sh, size_t sw,
                      size_t pd, size_t ph, size_t pw,
                      size_t dd, size_t dh, size_t dw)
{
  size_t spatial_out = out_d * out_h * out_w;
  size_t col_rows = c_per_grp * kd * kh * kw;
  size_t total = col_rows * spatial_out;
  size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  size_t col_c = idx % spatial_out;
  size_t col_r = idx / spatial_out;
  size_t ow = col_c % out_w;
  size_t tmp_c = col_c / out_w;
  size_t oh = tmp_c % out_h;
  size_t od = tmp_c / out_h;
  size_t fw = col_r % kw;
  size_t tmp_r = col_r / kw;
  size_t fh = tmp_r % kh;
  tmp_r /= kh;
  size_t fd = tmp_r % kd;
  size_t ci = tmp_r / kd;
  long long id = (long long) (od * sd + fd * dd) - (long long) pd;
  long long ih = (long long) (oh * sh + fh * dh) - (long long) ph;
  long long iw = (long long) (ow * sw + fw * dw) - (long long) pw;
  if (id >= 0 && id < (long long) in_d
      && ih >= 0 && ih < (long long) in_h
      && iw >= 0 && iw < (long long) in_w)
    {
      size_t src_idx = (((ci * in_d + (size_t) id) * in_h
                         + (size_t) ih) * in_w + (size_t) iw);
      col[idx] = in_n[src_idx];
    }
  else
    {
      col[idx] = 0.0f;
    }
}

__global__ void
basalt_col2im_3d_f32 (const float *dcol, float *gx_n,
                      size_t c_per_grp, size_t in_d, size_t in_h, size_t in_w,
                      size_t kd, size_t kh, size_t kw,
                      size_t out_d, size_t out_h, size_t out_w,
                      size_t sd, size_t sh, size_t sw,
                      size_t pd, size_t ph, size_t pw,
                      size_t dd, size_t dh, size_t dw)
{
  size_t total_in = c_per_grp * in_d * in_h * in_w;
  size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_in)
    return;
  size_t iw = idx % in_w;
  size_t tmp = idx / in_w;
  size_t ih = tmp % in_h;
  tmp /= in_h;
  size_t id = tmp % in_d;
  size_t ci = tmp / in_d;
  size_t spatial_out = out_d * out_h * out_w;
  float acc = 0.0f;
  for (size_t fd = 0; fd < kd; ++fd)
    {
      long long d_nom = (long long) id + (long long) pd
                        - (long long) (fd * dd);
      if (d_nom < 0 || (d_nom % (long long) sd) != 0)
        continue;
      size_t od = (size_t) (d_nom / (long long) sd);
      if (od >= out_d)
        continue;
      for (size_t fh = 0; fh < kh; ++fh)
        {
          long long h_nom = (long long) ih + (long long) ph
                            - (long long) (fh * dh);
          if (h_nom < 0 || (h_nom % (long long) sh) != 0)
            continue;
          size_t oh = (size_t) (h_nom / (long long) sh);
          if (oh >= out_h)
            continue;
          for (size_t fw = 0; fw < kw; ++fw)
            {
              long long w_nom = (long long) iw + (long long) pw
                                - (long long) (fw * dw);
              if (w_nom < 0 || (w_nom % (long long) sw) != 0)
                continue;
              size_t ow = (size_t) (w_nom / (long long) sw);
              if (ow >= out_w)
                continue;
              size_t col_r = ((ci * kd + fd) * kh + fh) * kw + fw;
              size_t col_c = (od * out_h + oh) * out_w + ow;
              acc += dcol[col_r * spatial_out + col_c];
            }
        }
    }
  gx_n[idx] += acc;
}

}
)CUDA";

} /* namespace */

Cuda_Runtime_Driver &
Cuda_Runtime_Driver::instance (void) noexcept
{
  static Cuda_Runtime_Driver inst;
  return inst;
}

Cuda_Runtime_Driver::Cuda_Runtime_Driver (void) noexcept
  : libcuda_handle_ (nullptr),
    libcublas_handle_ (nullptr),
    libnvrtc_handle_ (nullptr),
    primary_ctx_ (nullptr),
    cublas_context_ (nullptr),
    jit_module_ (nullptr),
    fn_fill_f32_ (nullptr),
    fn_unary_f32_ (nullptr),
    fn_binary_f32_ (nullptr),
    fn_row_bcast_f32_ (nullptr),
    fn_softmax_fwd_f32_ (nullptr),
    fn_softmax_bwd_f32_ (nullptr),
    fn_layernorm_fwd_f32_ (nullptr),
    fn_sum_last_f32_ (nullptr),
    fn_sgd_step_f32_ (nullptr),
    fn_adam_step_f32_ (nullptr),
    fn_rmsnorm_fwd_f32_ (nullptr),
    fn_im2col_3d_f32_ (nullptr),
    fn_col2im_3d_f32_ (nullptr),
    has_hw_driver_ (false),
    has_jit_kernels_ (false),
    hw_device_count_ (0),
    virtual_device_count_ (2),
    gpu_busy_ (false)
{
#if defined(BASALT_ENABLE_CUDA) && !defined(BASALT_INTERNAL_SANITIZER_ACTIVE)
  libcuda_handle_ = dlopen ("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (libcuda_handle_ != nullptr)
    {
      auto cu_init = reinterpret_cast<Cu_Init_Fn> (
        dlsym (libcuda_handle_, "cuInit"));
      auto cu_dev_count = reinterpret_cast<Cu_Device_Get_Count_Fn> (
        dlsym (libcuda_handle_, "cuDeviceGetCount"));
      auto cu_dev_get = reinterpret_cast<Cu_Device_Get_Fn> (
        dlsym (libcuda_handle_, "cuDeviceGet"));
      auto cu_ctx_retain = reinterpret_cast<Cu_Primary_Ctx_Retain_Fn> (
        dlsym (libcuda_handle_, "cuDevicePrimaryCtxRetain"));

      g_syms.ctx_set_current = reinterpret_cast<Cu_Ctx_Set_Current_Fn> (
        dlsym (libcuda_handle_, "cuCtxSetCurrent"));
      g_syms.ctx_sync = reinterpret_cast<Cu_Ctx_Synchronize_Fn> (
        dlsym (libcuda_handle_, "cuCtxSynchronize"));
      g_syms.mem_alloc_managed = reinterpret_cast<Cu_Mem_Alloc_Managed_Fn> (
        dlsym (libcuda_handle_, "cuMemAllocManaged"));
      g_syms.mem_free = reinterpret_cast<Cu_Mem_Free_Fn> (
        dlsym (libcuda_handle_, "cuMemFree_v2"));
      g_syms.memset_d8 = reinterpret_cast<Cu_Memset_D8_Fn> (
        dlsym (libcuda_handle_, "cuMemsetD8_v2"));
      g_syms.primary_ctx_release
        = reinterpret_cast<Cu_Primary_Ctx_Release_Fn> (
          dlsym (libcuda_handle_, "cuDevicePrimaryCtxRelease_v2"));
      g_syms.module_load_data = reinterpret_cast<Cu_Module_Load_Data_Fn> (
        dlsym (libcuda_handle_, "cuModuleLoadData"));
      g_syms.module_unload = reinterpret_cast<Cu_Module_Unload_Fn> (
        dlsym (libcuda_handle_, "cuModuleUnload"));
      g_syms.module_get_func = reinterpret_cast<Cu_Module_Get_Function_Fn> (
        dlsym (libcuda_handle_, "cuModuleGetFunction"));
      g_syms.launch_kernel = reinterpret_cast<Cu_Launch_Kernel_Fn> (
        dlsym (libcuda_handle_, "cuLaunchKernel"));

      if (cu_init != nullptr && cu_dev_count != nullptr
          && cu_dev_get != nullptr && cu_ctx_retain != nullptr
          && g_syms.ctx_set_current != nullptr && g_syms.ctx_sync != nullptr
          && g_syms.mem_alloc_managed != nullptr && g_syms.mem_free != nullptr
          && cu_init (0U) == 0)
        {
          int count = 0;
          int dev = 0;
          if (cu_dev_count (&count) == 0 && count > 0
              && cu_dev_get (&dev, 0) == 0
              && cu_ctx_retain (&primary_ctx_, dev) == 0
              && g_syms.ctx_set_current (primary_ctx_) == 0)
            {
              has_hw_driver_ = true;
              hw_device_count_ = static_cast<size_t> (count);

              libcublas_handle_ = dlopen ("libcublas.so.12",
                                          RTLD_LAZY | RTLD_LOCAL);
              if (libcublas_handle_ == nullptr)
                {
                  libcublas_handle_ = dlopen ("libcublas.so.11",
                                              RTLD_LAZY | RTLD_LOCAL);
                }
              if (libcublas_handle_ != nullptr)
                {
                  g_syms.cublas_create = reinterpret_cast<Cublas_Create_Fn> (
                    dlsym (libcublas_handle_, "cublasCreate_v2"));
                  g_syms.cublas_destroy = reinterpret_cast<Cublas_Destroy_Fn> (
                    dlsym (libcublas_handle_, "cublasDestroy_v2"));
                  g_syms.cublas_sgemm = reinterpret_cast<Cublas_Sgemm_Fn> (
                    dlsym (libcublas_handle_, "cublasSgemm_v2"));
                  g_syms.cublas_sgemm_batched
                    = reinterpret_cast<Cublas_Sgemm_Batched_Fn> (
                      dlsym (libcublas_handle_, "cublasSgemmStridedBatched"));
                  g_syms.cublas_gemm_ex
                    = reinterpret_cast<Cublas_Gemm_Ex_Fn> (
                      dlsym (libcublas_handle_, "cublasGemmEx"));
                  g_syms.cublas_gemm_batched_ex
                    = reinterpret_cast<Cublas_Gemm_Batched_Ex_Fn> (
                      dlsym (libcublas_handle_, "cublasGemmStridedBatchedEx"));
                  if (g_syms.cublas_create != nullptr
                      && g_syms.cublas_sgemm != nullptr)
                    {
                      if (g_syms.cublas_create (&cublas_context_) != 0)
                        {
                          cublas_context_ = nullptr;
                        }
                    }
                }
              compile_jit_kernels ();
            }
        }
    }
#endif
}

void
Cuda_Runtime_Driver::compile_jit_kernels (void) noexcept
{
  if (!has_hw_driver_ || g_syms.module_load_data == nullptr
      || g_syms.module_get_func == nullptr || g_syms.launch_kernel == nullptr)
    {
      return;
    }
  libnvrtc_handle_ = dlopen ("libnvrtc.so.12", RTLD_LAZY | RTLD_LOCAL);
  if (libnvrtc_handle_ == nullptr)
    {
      libnvrtc_handle_ = dlopen ("libnvrtc.so.11", RTLD_LAZY | RTLD_LOCAL);
    }
  if (libnvrtc_handle_ == nullptr)
    {
      return;
    }

  auto create_prog = reinterpret_cast<Nvrtc_Create_Prog_Fn> (
    dlsym (libnvrtc_handle_, "nvrtcCreateProgram"));
  auto compile_prog = reinterpret_cast<Nvrtc_Compile_Prog_Fn> (
    dlsym (libnvrtc_handle_, "nvrtcCompileProgram"));
  auto get_ptx_size = reinterpret_cast<Nvrtc_Get_Ptx_Size_Fn> (
    dlsym (libnvrtc_handle_, "nvrtcGetPTXSize"));
  auto get_ptx = reinterpret_cast<Nvrtc_Get_Ptx_Fn> (
    dlsym (libnvrtc_handle_, "nvrtcGetPTX"));
  auto destroy_prog = reinterpret_cast<Nvrtc_Destroy_Prog_Fn> (
    dlsym (libnvrtc_handle_, "nvrtcDestroyProgram"));

  if (create_prog == nullptr || compile_prog == nullptr
      || get_ptx_size == nullptr || get_ptx == nullptr
      || destroy_prog == nullptr)
    {
      return;
    }

  void *prog = nullptr;
  if (create_prog (&prog, k_jit_cuda_source, "basalt_kernels.cu",
                   0, nullptr, nullptr) != 0)
    {
      return;
    }

  const char *opts[] = {"--use_fast_math"};
  if (compile_prog (prog, 1, opts) == 0)
    {
      size_t ptx_sz = 0;
      if (get_ptx_size (prog, &ptx_sz) == 0 && ptx_sz > 0)
        {
          std::vector<char> ptx (ptx_sz);
          if (get_ptx (prog, ptx.data ()) == 0
              && g_syms.module_load_data (&jit_module_, ptx.data ()) == 0)
            {
              g_syms.module_get_func (&fn_fill_f32_, jit_module_,
                                      "basalt_fill_f32");
              g_syms.module_get_func (&fn_unary_f32_, jit_module_,
                                      "basalt_unary_f32");
              g_syms.module_get_func (&fn_binary_f32_, jit_module_,
                                      "basalt_binary_f32");
              g_syms.module_get_func (&fn_row_bcast_f32_, jit_module_,
                                      "basalt_row_bcast_f32");
              g_syms.module_get_func (&fn_softmax_fwd_f32_, jit_module_,
                                      "basalt_softmax_fwd_f32");
              g_syms.module_get_func (&fn_softmax_bwd_f32_, jit_module_,
                                      "basalt_softmax_bwd_f32");
              g_syms.module_get_func (&fn_layernorm_fwd_f32_, jit_module_,
                                      "basalt_layernorm_fwd_f32");
              g_syms.module_get_func (&fn_sum_last_f32_, jit_module_,
                                      "basalt_sum_last_f32");
              g_syms.module_get_func (&fn_sgd_step_f32_, jit_module_,
                                      "basalt_sgd_step_f32");
              g_syms.module_get_func (&fn_adam_step_f32_, jit_module_,
                                      "basalt_adam_step_f32");
              g_syms.module_get_func (&fn_rmsnorm_fwd_f32_, jit_module_,
                                      "basalt_rmsnorm_fwd_f32");
              g_syms.module_get_func (&fn_im2col_3d_f32_, jit_module_,
                                      "basalt_im2col_3d_f32");
              g_syms.module_get_func (&fn_col2im_3d_f32_, jit_module_,
                                      "basalt_col2im_3d_f32");
              if (fn_fill_f32_ != nullptr && fn_unary_f32_ != nullptr
                  && fn_binary_f32_ != nullptr
                  && fn_row_bcast_f32_ != nullptr)
                {
                  has_jit_kernels_ = true;
                }
            }
        }
    }
  destroy_prog (&prog);
}

Cuda_Runtime_Driver::~Cuda_Runtime_Driver (void) noexcept
{
  sync_if_busy ();
  if (has_hw_driver_ && g_syms.ctx_set_current != nullptr)
    {
      g_syms.ctx_set_current (primary_ctx_);
    }
  for (auto &kv : hw_pool_)
    {
      for (void *p : kv.second)
        {
          if (g_syms.mem_free != nullptr)
            {
              g_syms.mem_free (reinterpret_cast<unsigned long long> (p));
            }
        }
    }
  for (auto &kv : host_pool_)
    {
      for (void *p : kv.second)
        {
          std::free (p);
        }
    }
  if (jit_module_ != nullptr && g_syms.module_unload != nullptr)
    {
      g_syms.module_unload (jit_module_);
    }
  if (cublas_context_ != nullptr && g_syms.cublas_destroy != nullptr)
    {
      g_syms.cublas_destroy (cublas_context_);
    }
  if (primary_ctx_ != nullptr && g_syms.primary_ctx_release != nullptr)
    {
      g_syms.primary_ctx_release (0);
    }
  if (libnvrtc_handle_ != nullptr)
    {
      dlclose (libnvrtc_handle_);
    }
  if (libcublas_handle_ != nullptr)
    {
      dlclose (libcublas_handle_);
    }
  if (libcuda_handle_ != nullptr)
    {
      dlclose (libcuda_handle_);
    }
}

void
Cuda_Runtime_Driver::sync_hw_internal (void) const noexcept
{
  std::lock_guard<std::mutex> lock (mutex_);
  if (gpu_busy_.load (std::memory_order_relaxed))
    {
      if (has_hw_driver_ && g_syms.ctx_set_current != nullptr
          && g_syms.ctx_sync != nullptr)
        {
          g_syms.ctx_set_current (primary_ctx_);
          g_syms.ctx_sync ();
        }
      gpu_busy_.store (false, std::memory_order_release);
    }
}

size_t
Cuda_Runtime_Driver::device_count (void) const noexcept
{
  size_t vcount = virtual_device_count_.load (std::memory_order_relaxed);
  if (vcount > 0)
    {
      return vcount;
    }
  return has_hw_driver_ ? hw_device_count_ : 0U;
}

void
Cuda_Runtime_Driver::set_virtual_device_count (size_t count) noexcept
{
  virtual_device_count_.store (count, std::memory_order_relaxed);
}

void *
Cuda_Runtime_Driver::allocate (size_t device_index, size_t byte_count)
{
  BASALT_ASSERT_DEVICE (device_index < device_count (),
                        "CUDA device index out of range");
  size_t bucket = round_up_bucket (byte_count == 0 ? 64U : byte_count);
  void *ptr = nullptr;
  bool is_hw = is_hw_device (device_index);

  {
    std::lock_guard<std::mutex> lock (mutex_);
    if (is_hw)
      {
        auto it = hw_pool_.find (bucket);
        if (it != hw_pool_.end () && !it->second.empty ())
          {
            ptr = it->second.back ();
            it->second.pop_back ();
          }
        else
          {
            g_syms.ctx_set_current (primary_ctx_);
            unsigned long long dptr = 0;
            if (g_syms.mem_alloc_managed (&dptr, bucket, 0x1U) == 0)
              {
                ptr = reinterpret_cast<void *> (dptr);
              }
            else
              {
                is_hw = false;
              }
          }
        if (is_hw && ptr != nullptr)
          {
            if (g_syms.memset_d8 != nullptr && byte_count > 0)
              {
                g_syms.ctx_set_current (primary_ctx_);
                g_syms.memset_d8 (
                  reinterpret_cast<unsigned long long> (ptr), 0, byte_count);
                gpu_busy_.store (true, std::memory_order_release);
              }
            active_allocs_[ptr] = {device_index, bucket, true};
            return ptr;
          }
      }

    auto it = host_pool_.find (bucket);
    if (it != host_pool_.end () && !it->second.empty ())
      {
        ptr = it->second.back ();
        it->second.pop_back ();
      }
    else
      {
        ptr = std::aligned_alloc (64, bucket);
      }
    if (ptr == nullptr)
      {
        BASALT_THROW (Device_Error, "CUDA device memory allocation failed");
      }
    active_allocs_[ptr] = {device_index, bucket, false};
  }
  if (byte_count > 0)
    {
      std::memset (ptr, 0, byte_count);
    }
  return ptr;
}

void
Cuda_Runtime_Driver::deallocate (size_t /*device_index*/, void *ptr) noexcept
{
  if (ptr == nullptr)
    {
      return;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  auto it = active_allocs_.find (ptr);
  if (it == active_allocs_.end ())
    {
      return;
    }
  Alloc_Info info = it->second;
  active_allocs_.erase (it);

  if (info.is_managed_hw)
    {
      auto &vec = hw_pool_[info.bucket_bytes];
      if (vec.size () < 128)
        {
          vec.push_back (ptr);
          return;
        }
      if (g_syms.ctx_set_current != nullptr && g_syms.mem_free != nullptr)
        {
          g_syms.ctx_set_current (primary_ctx_);
          if (gpu_busy_.load (std::memory_order_relaxed)
              && g_syms.ctx_sync != nullptr)
            {
              g_syms.ctx_sync ();
              gpu_busy_.store (false, std::memory_order_release);
            }
          g_syms.mem_free (reinterpret_cast<unsigned long long> (ptr));
        }
    }
  else
    {
      auto &vec = host_pool_[info.bucket_bytes];
      if (vec.size () < 128)
        {
          vec.push_back (ptr);
          return;
        }
      std::free (ptr);
    }
}

void
Cuda_Runtime_Driver::memcpy_htod (size_t device_index, void *dst_dev,
                                  const void *src_host, size_t byte_count)
{
  BASALT_ASSERT_DEVICE (device_index < device_count (),
                        "CUDA device index out of range in memcpy_htod");
  if (byte_count > 0)
    {
      sync_if_busy ();
      std::memcpy (dst_dev, src_host, byte_count);
    }
}

void
Cuda_Runtime_Driver::memcpy_dtoh (size_t device_index, void *dst_host,
                                  const void *src_dev, size_t byte_count)
{
  BASALT_ASSERT_DEVICE (device_index < device_count (),
                        "CUDA device index out of range in memcpy_dtoh");
  if (byte_count > 0)
    {
      sync_if_busy ();
      std::memcpy (dst_host, src_dev, byte_count);
    }
}

void
Cuda_Runtime_Driver::memcpy_dtod (size_t dst_device, void *dst_dev,
                                  size_t src_device, const void *src_dev,
                                  size_t byte_count)
{
  BASALT_ASSERT_DEVICE (dst_device < device_count ()
                        && src_device < device_count (),
                        "CUDA device index out of range in memcpy_dtod");
  if (byte_count > 0 && dst_dev != src_dev)
    {
      sync_if_busy ();
      std::memcpy (dst_dev, src_dev, byte_count);
    }
}

void
Cuda_Runtime_Driver::synchronize (size_t device_index)
{
  BASALT_ASSERT_DEVICE (device_index < device_count (),
                        "CUDA device index out of range in synchronize");
  sync_if_busy ();
}

bool
Cuda_Runtime_Driver::cublas_sgemm (
  size_t device_index, bool trans_a, bool trans_b, size_t m, size_t n,
  size_t k, float alpha, const float *a_ptr, size_t lda,
  const float *b_ptr, size_t ldb, float beta, float *c_ptr, size_t ldc)
{
  if (!has_cublas (device_index))
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  int op_b = trans_b ? 1 : 0;
  int op_a = trans_a ? 1 : 0;
  int status = g_syms.cublas_sgemm (
    cublas_context_, op_b, op_a,
    static_cast<int> (n), static_cast<int> (m), static_cast<int> (k),
    &alpha, b_ptr, static_cast<int> (ldb),
    a_ptr, static_cast<int> (lda),
    &beta, c_ptr, static_cast<int> (ldc));
  if (status == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::cublas_sgemm_batched (
  size_t device_index, bool trans_a, bool trans_b, size_t m, size_t n,
  size_t k, float alpha, const float *a_ptr, size_t lda, long long stride_a,
  const float *b_ptr, size_t ldb, long long stride_b, float beta,
  float *c_ptr, size_t ldc, long long stride_c, size_t batch_count)
{
  if (!has_cublas (device_index) || g_syms.cublas_sgemm_batched == nullptr)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  int op_b = trans_b ? 1 : 0;
  int op_a = trans_a ? 1 : 0;
  int status = g_syms.cublas_sgemm_batched (
    cublas_context_, op_b, op_a,
    static_cast<int> (n), static_cast<int> (m), static_cast<int> (k),
    &alpha, b_ptr, static_cast<int> (ldb), stride_b,
    a_ptr, static_cast<int> (lda), stride_a,
    &beta, c_ptr, static_cast<int> (ldc), stride_c,
    static_cast<int> (batch_count));
  if (status == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::cublas_gemm_half (
  size_t device_index, bool is_bf16, bool trans_a, bool trans_b,
  size_t m, size_t n, size_t k, float alpha, const void *a_ptr, size_t lda,
  const void *b_ptr, size_t ldb, float beta, void *c_ptr, size_t ldc)
{
  if (!has_cublas (device_index) || g_syms.cublas_gemm_ex == nullptr)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  int op_b = trans_b ? 1 : 0;
  int op_a = trans_a ? 1 : 0;
  /* CUDA_R_16F = 2, CUDA_R_16BF = 14, CUBLAS_COMPUTE_32F = 68,
     CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99.  */
  int dtype_code = is_bf16 ? 14 : 2;
  int compute_type = 68;
  int algo = 99;
  int status = g_syms.cublas_gemm_ex (
    cublas_context_, op_b, op_a,
    static_cast<int> (n), static_cast<int> (m), static_cast<int> (k),
    &alpha, b_ptr, dtype_code, static_cast<int> (ldb),
    a_ptr, dtype_code, static_cast<int> (lda),
    &beta, c_ptr, dtype_code, static_cast<int> (ldc),
    compute_type, algo);
  if (status == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::cublas_gemm_half_batched (
  size_t device_index, bool is_bf16, bool trans_a, bool trans_b,
  size_t m, size_t n, size_t k, float alpha, const void *a_ptr, size_t lda,
  long long stride_a, const void *b_ptr, size_t ldb, long long stride_b,
  float beta, void *c_ptr, size_t ldc, long long stride_c, size_t batch_count)
{
  if (!has_cublas (device_index) || g_syms.cublas_gemm_batched_ex == nullptr)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  int op_b = trans_b ? 1 : 0;
  int op_a = trans_a ? 1 : 0;
  int dtype_code = is_bf16 ? 14 : 2;
  int compute_type = 68;
  int algo = 99;
  int status = g_syms.cublas_gemm_batched_ex (
    cublas_context_, op_b, op_a,
    static_cast<int> (n), static_cast<int> (m), static_cast<int> (k),
    &alpha, b_ptr, dtype_code, static_cast<int> (ldb), stride_b,
    a_ptr, dtype_code, static_cast<int> (lda), stride_a,
    &beta, c_ptr, dtype_code, static_cast<int> (ldc), stride_c,
    static_cast<int> (batch_count), compute_type, algo);
  if (status == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_fill_f32 (size_t device_index, float *dst,
                                      float val, size_t n)
{
  if (!has_kernels (device_index) || n == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (
    (n + threads - 1) / threads);
  void *args[] = {&dst, &val, &n};
  if (g_syms.launch_kernel (fn_fill_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_unary_f32 (size_t device_index, const float *src,
                                       float *dst, float param,
                                       Cuda_Unary_Op op, size_t n)
{
  if (!has_kernels (device_index) || n == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (
    (n + threads - 1) / threads);
  int op_code = static_cast<int> (op);
  void *args[] = {&src, &dst, &param, &op_code, &n};
  if (g_syms.launch_kernel (fn_unary_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_binary_f32 (size_t device_index, const float *a,
                                        const float *b, float *c,
                                        Cuda_Binary_Op op, size_t n)
{
  if (!has_kernels (device_index) || n == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (
    (n + threads - 1) / threads);
  int op_code = static_cast<int> (op);
  void *args[] = {&a, &b, &c, &op_code, &n};
  if (g_syms.launch_kernel (fn_binary_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_row_bcast_f32 (size_t device_index,
                                           const float *a, const float *b,
                                           float *c, Cuda_Binary_Op op,
                                           size_t total, size_t cols)
{
  if (!has_kernels (device_index) || total == 0 || cols == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (
    (total + threads - 1) / threads);
  int op_code = static_cast<int> (op);
  void *args[] = {&a, &b, &c, &op_code, &total, &cols};
  if (g_syms.launch_kernel (fn_row_bcast_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_softmax_fwd_f32 (size_t device_index,
                                             const float *x, float *y,
                                             size_t rows, size_t cols)
{
  if (!has_kernels (device_index) || fn_softmax_fwd_f32_ == nullptr
      || rows == 0 || cols == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (rows);
  void *args[] = {&x, &y, &rows, &cols};
  if (g_syms.launch_kernel (fn_softmax_fwd_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_softmax_bwd_f32 (size_t device_index,
                                             const float *g, const float *y,
                                             float *gx, size_t rows,
                                             size_t cols)
{
  if (!has_kernels (device_index) || fn_softmax_bwd_f32_ == nullptr
      || rows == 0 || cols == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (rows);
  void *args[] = {&g, &y, &gx, &rows, &cols};
  if (g_syms.launch_kernel (fn_softmax_bwd_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_layernorm_fwd_f32 (size_t device_index,
                                               const float *x, const float *w,
                                               const float *b, float *y,
                                               size_t rows, size_t cols,
                                               float eps)
{
  if (!has_kernels (device_index) || fn_layernorm_fwd_f32_ == nullptr
      || rows == 0 || cols == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (rows);
  void *args[] = {&x, &w, &b, &y, &rows, &cols, &eps};
  if (g_syms.launch_kernel (fn_layernorm_fwd_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_sum_last_f32 (size_t device_index, const float *x,
                                          float *out, size_t outer_ext,
                                          size_t reduce_ext)
{
  if (!has_kernels (device_index) || fn_sum_last_f32_ == nullptr
      || outer_ext == 0 || reduce_ext == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (outer_ext);
  void *args[] = {&x, &out, &outer_ext, &reduce_ext};
  if (g_syms.launch_kernel (fn_sum_last_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_sgd_step_f32 (
  size_t device_index, float *p, const float *g, float *buf, float lr,
  float wd, float mom, float damp, bool nesterov, bool has_mom,
  bool init_buf, size_t n)
{
  if (!has_kernels (device_index) || fn_sgd_step_f32_ == nullptr || n == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks
    = static_cast<unsigned int> ((n + threads - 1) / threads);
  int nest_i = nesterov ? 1 : 0;
  int mom_i = has_mom ? 1 : 0;
  int init_i = init_buf ? 1 : 0;
  void *args[] = {&p, &g, &buf, &lr, &wd, &mom, &damp, &nest_i, &mom_i,
                  &init_i, &n};
  if (g_syms.launch_kernel (fn_sgd_step_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_adam_step_f32 (
  size_t device_index, float *p, const float *g, float *m, float *v,
  float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
  bool adamw, size_t n)
{
  if (!has_kernels (device_index) || fn_adam_step_f32_ == nullptr || n == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks
    = static_cast<unsigned int> ((n + threads - 1) / threads);
  int adamw_i = adamw ? 1 : 0;
  void *args[] = {&p, &g, &m, &v, &lr, &b1, &b2, &eps, &wd, &bc1, &bc2,
                  &adamw_i, &n};
  if (g_syms.launch_kernel (fn_adam_step_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_rmsnorm_fwd_f32 (
  size_t device_index, const float *x, const float *w, float *y,
  size_t rows, size_t cols, float eps)
{
  if (!has_kernels (device_index) || fn_rmsnorm_fwd_f32_ == nullptr
      || rows == 0 || cols == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks = static_cast<unsigned int> (rows);
  void *args[] = {&x, &w, &y, &rows, &cols, &eps};
  if (g_syms.launch_kernel (fn_rmsnorm_fwd_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_im2col_3d_f32 (
  size_t device_index, const float *in_n, float *col,
  size_t c_per_grp, size_t in_d, size_t in_h, size_t in_w,
  size_t kd, size_t kh, size_t kw, size_t out_d, size_t out_h, size_t out_w,
  size_t sd, size_t sh, size_t sw, size_t pd, size_t ph, size_t pw,
  size_t dd, size_t dh, size_t dw)
{
  size_t total = c_per_grp * kd * kh * kw * out_d * out_h * out_w;
  if (!has_kernels (device_index) || fn_im2col_3d_f32_ == nullptr
      || total == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks
    = static_cast<unsigned int> ((total + threads - 1) / threads);
  void *args[] = {&in_n, &col, &c_per_grp, &in_d, &in_h, &in_w,
                  &kd, &kh, &kw, &out_d, &out_h, &out_w,
                  &sd, &sh, &sw, &pd, &ph, &pw, &dd, &dh, &dw};
  if (g_syms.launch_kernel (fn_im2col_3d_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

bool
Cuda_Runtime_Driver::launch_col2im_3d_f32 (
  size_t device_index, const float *dcol, float *gx_n,
  size_t c_per_grp, size_t in_d, size_t in_h, size_t in_w,
  size_t kd, size_t kh, size_t kw, size_t out_d, size_t out_h, size_t out_w,
  size_t sd, size_t sh, size_t sw, size_t pd, size_t ph, size_t pw,
  size_t dd, size_t dh, size_t dw)
{
  size_t total = c_per_grp * in_d * in_h * in_w;
  if (!has_kernels (device_index) || fn_col2im_3d_f32_ == nullptr
      || total == 0)
    {
      return false;
    }
  std::lock_guard<std::mutex> lock (mutex_);
  g_syms.ctx_set_current (primary_ctx_);
  unsigned int threads = 256U;
  unsigned int blocks
    = static_cast<unsigned int> ((total + threads - 1) / threads);
  void *args[] = {&dcol, &gx_n, &c_per_grp, &in_d, &in_h, &in_w,
                  &kd, &kh, &kw, &out_d, &out_h, &out_w,
                  &sd, &sh, &sw, &pd, &ph, &pw, &dd, &dh, &dw};
  if (g_syms.launch_kernel (fn_col2im_3d_f32_, blocks, 1, 1, threads, 1, 1,
                            0, nullptr, args, nullptr) == 0)
    {
      gpu_busy_.store (true, std::memory_order_release);
      return true;
    }
  return false;
}

Bellows_Storage::Bellows_Storage (size_t byte_size, Device device)
  : byte_size_ (byte_size),
    bucket_size_ (round_up_bucket (byte_size == 0 ? 64U : byte_size)),
    device_ (device),
    cpu_slab_ (),
    data_ptr_ (nullptr)
{
  size_t alloc_size = (byte_size == 0) ? 1 : byte_size;
  if (device_.is_cpu ())
    {
      cpu_slab_ = Cpu_Slab_Pool::instance ().acquire (bucket_size_);
      std::memset (cpu_slab_.data (), 0, alloc_size);
      data_ptr_ = cpu_slab_.data ();
    }
  else
    {
      data_ptr_ = Cuda_Runtime_Driver::instance ().allocate (
        device_.index (), alloc_size);
    }
}

Bellows_Storage::~Bellows_Storage (void) noexcept
{
  if (device_.is_cpu ())
    {
      if (data_ptr_ != nullptr)
        {
          Cpu_Slab_Pool::instance ().release (bucket_size_,
                                              std::move (cpu_slab_));
          data_ptr_ = nullptr;
        }
    }
  else if (device_.is_cuda () && data_ptr_ != nullptr)
    {
      Cuda_Runtime_Driver::instance ().deallocate (device_.index (),
                                                   data_ptr_);
      data_ptr_ = nullptr;
    }
}

void
Bellows_Storage::copy_to_host (void *dst_host, size_t byte_offset,
                               size_t count) const
{
  if (count == 0)
    {
      return;
    }
  BASALT_ASSERT_ARG (byte_offset + count <= byte_size_,
                     "copy_to_host out of storage bounds");
  const uint8_t *src = static_cast<const uint8_t *> (data_ptr_) + byte_offset;
  if (device_.is_cpu ())
    {
      std::memcpy (dst_host, src, count);
    }
  else
    {
      Cuda_Runtime_Driver::instance ().memcpy_dtoh (
        device_.index (), dst_host, src, count);
    }
}

void
Bellows_Storage::copy_from_host (const void *src_host, size_t byte_offset,
                                 size_t count)
{
  if (count == 0)
    {
      return;
    }
  BASALT_ASSERT_ARG (byte_offset + count <= byte_size_,
                     "copy_from_host out of storage bounds");
  uint8_t *dst = static_cast<uint8_t *> (data_ptr_) + byte_offset;
  if (device_.is_cpu ())
    {
      std::memcpy (dst, src_host, count);
    }
  else
    {
      Cuda_Runtime_Driver::instance ().memcpy_htod (
        device_.index (), dst, src_host, count);
    }
}

} /* namespace basalt::bellows_compat */
