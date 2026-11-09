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

#ifndef BASALT_FUNCTIONAL_HPP
#define BASALT_FUNCTIONAL_HPP

/* Stateless differentiable neural network operations (linear projections,
   1D/2D/3D convolutions, spatial pooling, batch/layer/group/RMS
   normalization, embeddings, dropout, and scaled dot-product attention).  */

#include <cstddef>
#include <basalt/config.hpp>
#include <basalt/ops.hpp>
#include <basalt/tensor.hpp>

namespace basalt::nn::functional
{

/* Affine linear projection: y = x * weight^T + bias.  */
BASALT_API Tensor
linear (const Tensor &input, const Tensor &weight,
        const Tensor &bias = Tensor ());

/* 1-D, 2-D, and 3-D convolutions with stride, zero-padding, dilation, and
   grouped channel support.  */
BASALT_API Tensor
conv1d (const Tensor &input, const Tensor &weight,
        const Tensor &bias = Tensor (),
        size_t stride = 1, size_t padding = 0,
        size_t dilation = 1, size_t groups = 1);

BASALT_API Tensor
conv2d (const Tensor &input, const Tensor &weight,
        const Tensor &bias = Tensor (),
        size_t stride_h = 1, size_t stride_w = 1,
        size_t pad_h = 0, size_t pad_w = 0,
        size_t dilation_h = 1, size_t dilation_w = 1,
        size_t groups = 1);

BASALT_API Tensor
conv3d (const Tensor &input, const Tensor &weight,
        const Tensor &bias = Tensor (),
        size_t stride_d = 1, size_t stride_h = 1, size_t stride_w = 1,
        size_t pad_d = 0, size_t pad_h = 0, size_t pad_w = 0,
        size_t dilation_d = 1, size_t dilation_h = 1, size_t dilation_w = 1,
        size_t groups = 1);

/* 2-D transposed convolution (deconvolution) for upsampling decoders and
   generative models.  */
BASALT_API Tensor
conv_transpose2d (const Tensor &input, const Tensor &weight,
                  const Tensor &bias = Tensor (),
                  size_t stride_h = 1, size_t stride_w = 1,
                  size_t pad_h = 0, size_t pad_w = 0,
                  size_t out_pad_h = 0, size_t out_pad_w = 0,
                  size_t dilation_h = 1, size_t dilation_w = 1,
                  size_t groups = 1);

/* Max and average spatial pooling over 1-D, 2-D, and 3-D feature maps.  */
BASALT_API Tensor
max_pool1d (const Tensor &input, size_t kernel_size,
            size_t stride = 0, size_t padding = 0);

BASALT_API Tensor
max_pool2d (const Tensor &input, size_t kernel_h, size_t kernel_w,
            size_t stride_h = 0, size_t stride_w = 0,
            size_t pad_h = 0, size_t pad_w = 0);

BASALT_API Tensor
max_pool3d (const Tensor &input,
            size_t kernel_d, size_t kernel_h, size_t kernel_w,
            size_t stride_d = 0, size_t stride_h = 0, size_t stride_w = 0,
            size_t pad_d = 0, size_t pad_h = 0, size_t pad_w = 0);

BASALT_API Tensor
avg_pool1d (const Tensor &input, size_t kernel_size,
            size_t stride = 0, size_t padding = 0);

BASALT_API Tensor
avg_pool2d (const Tensor &input, size_t kernel_h, size_t kernel_w,
            size_t stride_h = 0, size_t stride_w = 0,
            size_t pad_h = 0, size_t pad_w = 0);

BASALT_API Tensor
avg_pool3d (const Tensor &input,
            size_t kernel_d, size_t kernel_h, size_t kernel_w,
            size_t stride_d = 0, size_t stride_h = 0, size_t stride_w = 0,
            size_t pad_d = 0, size_t pad_h = 0, size_t pad_w = 0);

BASALT_API Tensor
adaptive_avg_pool2d (const Tensor &input, size_t output_h, size_t output_w);

/* Normalization operations: Batch, Layer, Group, and RMS normalization.  */
BASALT_API Tensor
batch_norm (const Tensor &input,
            Tensor &running_mean,
            Tensor &running_var,
            const Tensor &weight = Tensor (),
            const Tensor &bias = Tensor (),
            bool training = false,
            double momentum = 0.1,
            double eps = 1e-5);

BASALT_API Tensor
layer_norm (const Tensor &input,
            const Shape_Vector &normalized_shape,
            const Tensor &weight = Tensor (),
            const Tensor &bias = Tensor (),
            double eps = 1e-5);

BASALT_API Tensor
group_norm (const Tensor &input,
            size_t num_groups,
            const Tensor &weight = Tensor (),
            const Tensor &bias = Tensor (),
            double eps = 1e-5);

BASALT_API Tensor
rms_norm (const Tensor &input,
          const Shape_Vector &normalized_shape,
          const Tensor &weight = Tensor (),
          double eps = 1e-6);

/* Discrete token embedding lookup.  */
BASALT_API Tensor
embedding (const Tensor &indices, const Tensor &weight,
           ptrdiff_t padding_idx = -1);

/* Inverted dropout during training; identity pass-through during eval.  */
BASALT_API Tensor
dropout (const Tensor &input, double p = 0.5, bool training = true);

/* Scaled dot-product attention with optional mask, causal triangular mask,
   and attention-weight dropout.  */
BASALT_API Tensor
scaled_dot_product_attention (const Tensor &query,
                              const Tensor &key,
                              const Tensor &value,
                              const Tensor &attn_mask = Tensor (),
                              double dropout_p = 0.0,
                              bool is_causal = false,
                              bool training = false);

/* Re-export activations inside basalt::nn::functional.  */
using basalt::ops::elu;
using basalt::ops::gelu;
using basalt::ops::leaky_relu;
using basalt::ops::log_softmax;
using basalt::ops::mish;
using basalt::ops::relu;
using basalt::ops::sigmoid;
using basalt::ops::silu;
using basalt::ops::softmax;
using basalt::ops::softplus;
using basalt::ops::tanh;

/* Apply Rotary Position Embeddings (RoPE) to a query or key tensor of shape
   [..., seq_len, head_dim] using precomputed cos/sin tables of shape
   [seq_len, head_dim / 2].  */
BASALT_API Tensor
apply_rotary_pos_emb (const Tensor &x,
                      const Tensor &cos_cached,
                      const Tensor &sin_cached);

/* Differentiable cosine similarity between X1 and X2 along DIM.  */
BASALT_API Tensor
cosine_similarity (const Tensor &x1, const Tensor &x2,
                   ptrdiff_t dim = 1, double eps = 1e-8);

} /* namespace basalt::nn::functional */

namespace basalt
{
namespace F = nn::functional;
} /* namespace basalt */

#endif /* BASALT_FUNCTIONAL_HPP */
