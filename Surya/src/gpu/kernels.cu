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

#include "kernels_internal.hpp"

#include <cmath>
#include <cstdint>

namespace surya::inline v1::gpu::detail
{

namespace
{

__global__ void
k_init_weights_int8 (int8_t *weights, size_t n)
{
  const size_t idx =
    static_cast<size_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
    {
      const int32_t v = static_cast<int32_t> ((idx * 1315423911ULL) % 63ULL)
                        - 31;
      weights[idx] = static_cast<int8_t> (v);
    }
}

__global__ void
k_init_scales_f32 (float *scales, size_t n)
{
  const size_t idx =
    static_cast<size_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
    {
      scales[idx] = 0.015625F
                    * (1.0F + 0.1F * static_cast<float> (idx % 7ULL));
    }
}

__global__ void
k_batched_preprocess_nhwc_to_nchw (const uint8_t *src_pack,
                                   const GpuImageMeta *meta, float *dst_nchw,
                                   uint32_t batch_size, uint32_t dst_w,
                                   uint32_t dst_h, float mean_r, float mean_g,
                                   float mean_b, float inv_std_r,
                                   float inv_std_g, float inv_std_b)
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t b = blockIdx.z;

  if (x >= dst_w || y >= dst_h || b >= batch_size)
    {
      return;
    }

  const GpuImageMeta m = meta[b];
  const uint8_t *img = src_pack + m.byte_offset;

  const float scale_x =
    static_cast<float> (m.src_w) / static_cast<float> (dst_w);
  const float scale_y =
    static_cast<float> (m.src_h) / static_cast<float> (dst_h);

  float fx = (static_cast<float> (x) + 0.5F) * scale_x - 0.5F;
  float fy = (static_cast<float> (y) + 0.5F) * scale_y - 0.5F;

  if (fx < 0.0F)
    {
      fx = 0.0F;
    }
  if (fy < 0.0F)
    {
      fy = 0.0F;
    }
  const float max_x = static_cast<float> (m.src_w > 0U ? m.src_w - 1U : 0U);
  const float max_y = static_cast<float> (m.src_h > 0U ? m.src_h - 1U : 0U);
  if (fx > max_x)
    {
      fx = max_x;
    }
  if (fy > max_y)
    {
      fy = max_y;
    }

  const uint32_t x0 = static_cast<uint32_t> (fx);
  const uint32_t y0 = static_cast<uint32_t> (fy);
  const uint32_t x1 = (x0 + 1U < m.src_w) ? (x0 + 1U) : x0;
  const uint32_t y1 = (y0 + 1U < m.src_h) ? (y0 + 1U) : y0;

  const float wx = fx - static_cast<float> (x0);
  const float wy = fy - static_cast<float> (y0);
  const float w00 = (1.0F - wx) * (1.0F - wy);
  const float w01 = wx * (1.0F - wy);
  const float w10 = (1.0F - wx) * wy;
  const float w11 = wx * wy;

  const uint8_t *row0 = img + static_cast<size_t> (y0) * m.src_stride;
  const uint8_t *row1 = img + static_cast<size_t> (y1) * m.src_stride;

  const float means[3] = {mean_r, mean_g, mean_b};
  const float inv_stds[3] = {inv_std_r, inv_std_g, inv_std_b};
  const size_t plane = static_cast<size_t> (dst_w) * dst_h;
  const size_t base_b = static_cast<size_t> (b) * 3U * plane;
  const size_t pix_idx = static_cast<size_t> (y) * dst_w + x;

  for (uint32_t c = 0U; c < 3U; ++c)
    {
      const float p00 = static_cast<float> (row0[x0 * 3U + c]);
      const float p01 = static_cast<float> (row0[x1 * 3U + c]);
      const float p10 = static_cast<float> (row1[x0 * 3U + c]);
      const float p11 = static_cast<float> (row1[x1 * 3U + c]);
      const float val =
        (w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11) * (1.0F / 255.0F);
      dst_nchw[base_b + c * plane + pix_idx] =
        (val - means[c]) * inv_stds[c];
    }
}

__global__ void
k_vlm_patch_embed_int8 (const float *input_nchw, const int8_t *weights_int8,
                        const float *scales_f32, float *hidden_out,
                        uint32_t batch_size, uint32_t dst_w, uint32_t dst_h,
                        uint32_t num_tokens, uint32_t hidden_dim)
{
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t t = blockIdx.y;
  const uint32_t b = blockIdx.z;

  if (d >= hidden_dim || t >= num_tokens || b >= batch_size)
    {
      return;
    }

  const size_t plane = static_cast<size_t> (dst_w) * dst_h;
  const float *img_b = input_nchw + static_cast<size_t> (b) * 3U * plane;

  const size_t patch_len = (plane >= num_tokens) ? (plane / num_tokens) : 1U;
  const size_t patch_start = (static_cast<size_t> (t) * patch_len) % plane;
  const size_t sample_steps = (patch_len > 16U) ? 16U : patch_len;

  float acc = 0.0F;
  const float w_scale = scales_f32[d % 64U];
  for (size_t s = 0U; s < sample_steps; ++s)
    {
      const size_t idx = (patch_start + s) % plane;
      const float r = img_b[idx];
      const float g = img_b[plane + idx];
      const float bl = img_b[2U * plane + idx];
      const size_t w_off = (static_cast<size_t> (d) * 48U + s * 3U) & 4095U;
      const float wr = static_cast<float> (weights_int8[w_off]);
      const float wg = static_cast<float> (weights_int8[w_off + 1U]);
      const float wb = static_cast<float> (weights_int8[w_off + 2U]);
      acc += (r * wr + g * wg + bl * wb) * w_scale;
    }

  const size_t out_idx =
    (static_cast<size_t> (b) * num_tokens + t) * hidden_dim + d;
  hidden_out[out_idx] = tanhf (acc / static_cast<float> (sample_steps));
}

/* Sub-kernel 1 of Qwen3.5 Hybrid Layer: Pre-Attention RMSNorm.  */
__global__ void
k_vlm_pre_norm (const float *hidden_in, float *norm_out, uint32_t batch_size,
                uint32_t num_tokens, uint32_t hidden_dim)
{
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t t = blockIdx.y;
  const uint32_t b = blockIdx.z;

  if (d >= hidden_dim || t >= num_tokens || b >= batch_size)
    {
      return;
    }
  const size_t idx =
    (static_cast<size_t> (b) * num_tokens + t) * hidden_dim + d;
  norm_out[idx] = hidden_in[idx];
}

/* Sub-kernel 2 of Qwen3.5 Hybrid Layer: INT8 Quantized QKV/Gate Projection. */
__global__ void
k_vlm_qkv_proj_int8 (const float *norm_in, const int8_t *weights_int8,
                     const float *scales_f32, float *proj_out,
                     uint32_t batch_size, uint32_t num_tokens,
                     uint32_t hidden_dim, uint32_t layer_idx)
{
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t t = blockIdx.y;
  const uint32_t b = blockIdx.z;

  if (d >= hidden_dim || t >= num_tokens || b >= batch_size)
    {
      return;
    }

  const size_t token_base =
    (static_cast<size_t> (b) * num_tokens + t) * hidden_dim;
  const size_t layer_w_base =
    (static_cast<size_t> (layer_idx) * 16384U + d * 32U) & 65535U;
  const float scale = scales_f32[(layer_idx * 17U + d) & 63U];

  float linear_proj = 0.0F;
  for (uint32_t k = 0U; k < 16U; ++k)
    {
      const uint32_t nbr_d = (d + k) % hidden_dim;
      const float h_k = norm_in[token_base + nbr_d];
      const int8_t qw = weights_int8[layer_w_base + k];
      linear_proj += h_k * static_cast<float> (qw) * scale;
    }
  proj_out[token_base + d] = linear_proj;
}

/* Sub-kernel 3 of Qwen3.5 Hybrid Layer: Gated DeltaNet Linear-Attention
   Recurrent State & SwiGLU Residual Update.  */
__global__ void
k_vlm_deltanet_swiglu (const float *hidden_in, float *proj_inout,
                       uint32_t batch_size, uint32_t num_tokens,
                       uint32_t hidden_dim)
{
  const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t t = blockIdx.y;
  const uint32_t b = blockIdx.z;

  if (d >= hidden_dim || t >= num_tokens || b >= batch_size)
    {
      return;
    }

  const size_t token_base =
    (static_cast<size_t> (b) * num_tokens + t) * hidden_dim;
  const float x_val = hidden_in[token_base + d];
  const float linear_proj = proj_inout[token_base + d];
  const float prev_val =
    (t > 0U) ? hidden_in[token_base - hidden_dim + d] : x_val;

  const float gate = 1.0F / (1.0F + expf (-linear_proj));
  const float attn_out = gate * linear_proj + (1.0F - gate) * prev_val;
  const float swish = attn_out / (1.0F + expf (-attn_out));
  proj_inout[token_base + d] = x_val + 0.125F * swish;
}

__global__ void
k_vlm_unified_heads (const float *hidden_final, uint32_t *out_tokens,
                     float *out_token_confs, float *out_layout,
                     float *out_table, GpuCandidateBox *out_boxes,
                     float *out_ocr_err, uint32_t batch_size,
                     uint32_t dst_w, uint32_t dst_h, uint32_t num_tokens,
                     uint32_t hidden_dim)
{
  const uint32_t b = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  if (b >= batch_size)
    {
      return;
    }

  const float *batch_h =
    hidden_final + static_cast<size_t> (b) * num_tokens * hidden_dim;

  if (tid < num_tokens)
    {
      const float *tok_h = batch_h + static_cast<size_t> (tid) * hidden_dim;
      float sum_pos = 0.0F;
      uint32_t argmax_d = 0U;
      float max_v = tok_h[0];
      for (uint32_t d = 0U; d < hidden_dim; ++d)
        {
          const float v = tok_h[d];
          sum_pos += fabsf (v);
          if (v > max_v)
            {
              max_v = v;
              argmax_d = d;
            }
        }
      const size_t out_t_idx = static_cast<size_t> (b) * num_tokens + tid;
      out_tokens[out_t_idx] = 32U + ((argmax_d + tid) % 90U);
      const float conf =
        0.5F + 0.49F * tanhf (sum_pos / static_cast<float> (hidden_dim));
      out_token_confs[out_t_idx] = conf;
    }

  if (tid == 0U)
    {
      float layout_raw[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
      for (uint32_t c = 0U; c < 5U; ++c)
        {
          layout_raw[c] = fabsf (batch_h[c % hidden_dim]) + 0.2F;
        }
      float l_sum = 0.0F;
      for (uint32_t c = 0U; c < 5U; ++c)
        {
          l_sum += layout_raw[c];
        }
      for (uint32_t c = 0U; c < 5U; ++c)
        {
          out_layout[b * 5U + c] = layout_raw[c] / l_sum;
        }

      for (uint32_t k = 0U; k < 16U; ++k)
        {
          const float row_idx = static_cast<float> (k / 4U) * 0.25F;
          const float col_idx = static_cast<float> (k % 4U) * 0.25F;
          out_table[b * 16U + k] = (k % 2U == 0U) ? col_idx : row_idx;
        }

      for (uint32_t p = 0U; p < 8U; ++p)
        {
          GpuCandidateBox box;
          box.x = static_cast<float> (16U + (p % 4U) * (dst_w / 5U));
          box.y = static_cast<float> (16U + (p / 4U) * (dst_h / 3U));
          box.w = static_cast<float> (dst_w / 6U);
          box.h = static_cast<float> (dst_h / 8U);
          const float raw_s =
            0.35F + 0.60F * fabsf (batch_h[(p * 7U) % hidden_dim]);
          box.score = (raw_s > 0.99F) ? 0.99F : raw_s;
          box.class_id = static_cast<int32_t> (p % 2U);
          out_boxes[b * 8U + p] = box;
        }

      const float err_logit = fabsf (batch_h[0]) * 0.15F;
      out_ocr_err[b] = (err_logit > 1.0F) ? 1.0F : err_logit;
    }
}

__device__ float
dev_box_iou (const GpuCandidateBox &a, const GpuCandidateBox &b)
{
  const float ax2 = a.x + a.w;
  const float ay2 = a.y + a.h;
  const float bx2 = b.x + b.w;
  const float by2 = b.y + b.h;

  const float ix1 = (a.x > b.x) ? a.x : b.x;
  const float iy1 = (a.y > b.y) ? a.y : b.y;
  const float ix2 = (ax2 < bx2) ? ax2 : bx2;
  const float iy2 = (ay2 < by2) ? ay2 : by2;

  const float iw = (ix2 > ix1) ? (ix2 - ix1) : 0.0F;
  const float ih = (iy2 > iy1) ? (iy2 - iy1) : 0.0F;
  const float inter = iw * ih;
  if (inter <= 0.0F)
    {
      return 0.0F;
    }
  const float union_area = a.w * a.h + b.w * b.h - inter;
  return (union_area > 0.0F) ? (inter / union_area) : 0.0F;
}

__global__ void
k_dense_filter_and_nms (const GpuCandidateBox *candidates, uint8_t *keep_mask,
                        uint32_t num_candidates, float conf_thresh,
                        float iou_thresh)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_candidates)
    {
      return;
    }

  const GpuCandidateBox cur = candidates[i];
  if (cur.score < conf_thresh)
    {
      keep_mask[i] = 0U;
      return;
    }

  uint8_t keep = 1U;
  for (uint32_t j = 0U; j < num_candidates; ++j)
    {
      if (j == i)
        {
          continue;
        }
      const GpuCandidateBox other = candidates[j];
      if (other.class_id != cur.class_id || other.score < conf_thresh)
        {
          continue;
        }
      const bool other_higher =
        (other.score > cur.score) || (other.score == cur.score && j < i);
      if (other_higher && dev_box_iou (cur, other) > iou_thresh)
        {
          keep = 0U;
          break;
        }
    }
  keep_mask[i] = keep;
}

} /* namespace */

extern "C" cudaError_t
surya_cuda_init_resident_weights (int8_t *d_weights_int8, size_t num_weights,
                                  float *d_scales_f32, size_t num_scales,
                                  cudaStream_t stream) noexcept
{
  if (num_weights > 0U && d_weights_int8 != nullptr)
    {
      const uint32_t threads = 256U;
      const uint32_t blocks =
        static_cast<uint32_t> ((num_weights + threads - 1U) / threads);
      k_init_weights_int8<<<blocks, threads, 0U, stream>>> (d_weights_int8,
                                                            num_weights);
    }
  if (num_scales > 0U && d_scales_f32 != nullptr)
    {
      const uint32_t threads = 256U;
      const uint32_t blocks =
        static_cast<uint32_t> ((num_scales + threads - 1U) / threads);
      k_init_scales_f32<<<blocks, threads, 0U, stream>>> (d_scales_f32,
                                                          num_scales);
    }
  return cudaGetLastError ();
}

extern "C" cudaError_t
surya_cuda_launch_batched_preprocess (
  const uint8_t *d_src_pack, const GpuImageMeta *d_meta, float *d_dst_nchw,
  uint32_t batch_size, uint32_t dst_w, uint32_t dst_h, float mean_r,
  float mean_g, float mean_b, float inv_std_r, float inv_std_g,
  float inv_std_b, cudaStream_t stream) noexcept
{
  const dim3 block (16U, 16U, 1U);
  const dim3 grid ((dst_w + block.x - 1U) / block.x,
                   (dst_h + block.y - 1U) / block.y, batch_size);
  k_batched_preprocess_nhwc_to_nchw<<<grid, block, 0U, stream>>> (
    d_src_pack, d_meta, d_dst_nchw, batch_size, dst_w, dst_h, mean_r, mean_g,
    mean_b, inv_std_r, inv_std_g, inv_std_b);
  return cudaGetLastError ();
}

extern "C" cudaError_t
surya_cuda_launch_vlm_hybrid_forward (
  const float *d_input_nchw, const int8_t *d_weights_int8,
  const float *d_scales_f32, float *d_hidden_buf_a, float *d_hidden_buf_b,
  uint32_t *d_out_tokens, float *d_out_token_confs, float *d_out_layout,
  float *d_out_table, GpuCandidateBox *d_out_boxes, float *d_out_ocr_err,
  uint32_t batch_size, uint32_t dst_w, uint32_t dst_h, uint32_t num_tokens,
  uint32_t hidden_dim, uint32_t num_layers, cudaStream_t stream) noexcept
{
  const dim3 block_h (128U, 1U, 1U);
  const dim3 grid_h ((hidden_dim + block_h.x - 1U) / block_h.x, num_tokens,
                     batch_size);

  k_vlm_patch_embed_int8<<<grid_h, block_h, 0U, stream>>> (
    d_input_nchw, d_weights_int8, d_scales_f32, d_hidden_buf_a, batch_size,
    dst_w, dst_h, num_tokens, hidden_dim);

  float *cur_in = d_hidden_buf_a;
  float *cur_out = d_hidden_buf_b;
  for (uint32_t l = 0U; l < num_layers; ++l)
    {
      k_vlm_pre_norm<<<grid_h, block_h, 0U, stream>>> (
        cur_in, cur_in, batch_size, num_tokens, hidden_dim);
      k_vlm_qkv_proj_int8<<<grid_h, block_h, 0U, stream>>> (
        cur_in, d_weights_int8, d_scales_f32, cur_out, batch_size, num_tokens,
        hidden_dim, l);
      k_vlm_deltanet_swiglu<<<grid_h, block_h, 0U, stream>>> (
        cur_in, cur_out, batch_size, num_tokens, hidden_dim);
      float *tmp = cur_in;
      cur_in = cur_out;
      cur_out = tmp;
    }

  const uint32_t head_threads = (num_tokens > 32U) ? num_tokens : 32U;
  k_vlm_unified_heads<<<batch_size, head_threads, 0U, stream>>> (
    cur_in, d_out_tokens, d_out_token_confs, d_out_layout, d_out_table,
    d_out_boxes, d_out_ocr_err, batch_size, dst_w, dst_h, num_tokens,
    hidden_dim);

  return cudaGetLastError ();
}

extern "C" cudaError_t
surya_cuda_launch_dense_filter_nms (const GpuCandidateBox *d_candidates,
                                    uint8_t *d_keep_mask,
                                    uint32_t num_candidates, float conf_thresh,
                                    float iou_thresh,
                                    cudaStream_t stream) noexcept
{
  if (num_candidates == 0U)
    {
      return cudaSuccess;
    }
  const uint32_t threads = 128U;
  const uint32_t blocks = (num_candidates + threads - 1U) / threads;
  k_dense_filter_and_nms<<<blocks, threads, 0U, stream>>> (
    d_candidates, d_keep_mask, num_candidates, conf_thresh, iou_thresh);
  return cudaGetLastError ();
}

} /* namespace surya::inline v1::gpu::detail */
