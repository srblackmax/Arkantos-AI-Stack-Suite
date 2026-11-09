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

#ifndef SURYA_GPU_KERNELS_INTERNAL_HPP
#define SURYA_GPU_KERNELS_INTERNAL_HPP

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace surya::inline v1::gpu::detail
{

struct GpuImageMeta
{
  uint32_t src_w;
  uint32_t src_h;
  uint32_t src_stride;
  uint32_t byte_offset;
};

struct GpuCandidateBox
{
  float x;
  float y;
  float w;
  float h;
  float score;
  int32_t class_id;
};

extern "C"
{

  cudaError_t surya_cuda_init_resident_weights (int8_t *d_weights_int8,
                                                size_t num_weights,
                                                float *d_scales_f32,
                                                size_t num_scales,
                                                cudaStream_t stream) noexcept;

  cudaError_t surya_cuda_launch_batched_preprocess (
    const uint8_t *d_src_pack, const GpuImageMeta *d_meta, float *d_dst_nchw,
    uint32_t batch_size, uint32_t dst_w, uint32_t dst_h, float mean_r,
    float mean_g, float mean_b, float inv_std_r, float inv_std_g,
    float inv_std_b, cudaStream_t stream) noexcept;

  cudaError_t surya_cuda_launch_vlm_hybrid_forward (
    const float *d_input_nchw, const int8_t *d_weights_int8,
    const float *d_scales_f32, float *d_hidden_buf_a, float *d_hidden_buf_b,
    uint32_t *d_out_tokens, float *d_out_token_confs, float *d_out_layout,
    float *d_out_table, GpuCandidateBox *d_out_boxes, float *d_out_ocr_err,
    uint32_t batch_size, uint32_t dst_w, uint32_t dst_h, uint32_t num_tokens,
    uint32_t hidden_dim, uint32_t num_layers, cudaStream_t stream) noexcept;

  cudaError_t surya_cuda_launch_dense_filter_nms (
    const GpuCandidateBox *d_candidates, uint8_t *d_keep_mask,
    uint32_t num_candidates, float conf_thresh, float iou_thresh,
    cudaStream_t stream) noexcept;

} /* extern "C" */

} /* namespace surya::inline v1::gpu::detail */

#endif /* SURYA_GPU_KERNELS_INTERNAL_HPP */
