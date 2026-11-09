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

#include <obsidian/tensor_bridge.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace obsidian
{

Result<Tensor_Block>
Tensor_Block::zeros (const Shape_Vector &shape, Tensor_Options opts)
{
  if (shape.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "Tensor_Block::zeros requires non-empty shape");
    }
  for (size_t d : shape)
    {
      if (d == 0)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                                 "Tensor_Block::zeros dimension cannot be 0");
        }
    }
  try
    {
      return Tensor_Block (basalt::zeros (shape, opts));
    }
  catch (const std::exception &e)
    {
      return OBSIDIAN_ERROR (Error_Code::Backend_Failure, e.what ());
    }
}

Result<Tensor_Block>
Tensor_Block::from_span_f32 (std::span<const float> data,
                             const Shape_Vector &shape,
                             bool requires_grad)
{
  if (shape.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "from_span_f32 requires non-empty shape");
    }
  size_t expected = 1;
  for (size_t d : shape)
    {
      expected *= d;
    }
  if (expected != data.size ())
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "from_span_f32 span size does not match shape");
    }
  try
    {
      Tensor_Options opts = Tensor_Options ()
        .with_dtype (Dtype::Float32)
        .with_requires_grad (requires_grad);
      Tensor t = basalt::from_blob (data.data (), shape, opts);
      return Tensor_Block (std::move (t));
    }
  catch (const std::exception &e)
    {
      return OBSIDIAN_ERROR (Error_Code::Backend_Failure, e.what ());
    }
}

Result<Tensor_Block>
Tensor_Block::from_token_ids (std::span<const int64_t> ids,
                              size_t batch_size,
                              size_t seq_len)
{
  if (batch_size == 0 || seq_len == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "batch_size and seq_len must be > 0");
    }
  if (ids.size () != batch_size * seq_len)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "token ID span size mismatch");
    }
  try
    {
      Tensor_Options opts = Tensor_Options ().with_dtype (Dtype::Int64);
      Tensor t = basalt::from_blob (ids.data (), {batch_size, seq_len}, opts);
      return Tensor_Block (std::move (t));
    }
  catch (const std::exception &e)
    {
      return OBSIDIAN_ERROR (Error_Code::Backend_Failure, e.what ());
    }
}

Result<void>
Tensor_Block::expect_rank (size_t expected_ndim) const
{
  if (!tensor_.defined ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "tensor is undefined");
    }
  if (tensor_.ndim () != expected_ndim)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "tensor rank mismatch");
    }
  return {};
}

Result<void>
Tensor_Block::expect_shape (const Shape_Vector &expected_shape) const
{
  if (!tensor_.defined ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "tensor is undefined");
    }
  if (tensor_.shape () != expected_shape)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "tensor shape mismatch");
    }
  return {};
}

Result<void>
Tensor_Block::expect_dtype (Dtype expected_dtype) const
{
  if (!tensor_.defined ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "tensor is undefined");
    }
  if (tensor_.dtype () != expected_dtype)
    {
      return OBSIDIAN_ERROR (Error_Code::Dtype_Mismatch,
                             "tensor dtype mismatch");
    }
  return {};
}

Result<std::span<const float>>
Tensor_Block::span_f32 (void) const
{
  if (!tensor_.defined ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "tensor is undefined");
    }
  if (tensor_.dtype () != Dtype::Float32)
    {
      return OBSIDIAN_ERROR (Error_Code::Dtype_Mismatch,
                             "span_f32 requires Float32 tensor");
    }
  if (!tensor_.device ().is_cpu ())
    {
      return OBSIDIAN_ERROR (Error_Code::Device_Mismatch,
                             "span_f32 requires CPU tensor");
    }
  if (!tensor_.is_contiguous ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "span_f32 requires contiguous tensor");
    }
  return std::span<const float> (tensor_.data_ptr<float> (),
                                 tensor_.numel ());
}

Result<std::span<float>>
Tensor_Block::span_f32_mut (void)
{
  if (!tensor_.defined ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "tensor is undefined");
    }
  if (tensor_.dtype () != Dtype::Float32)
    {
      return OBSIDIAN_ERROR (Error_Code::Dtype_Mismatch,
                             "span_f32_mut requires Float32 tensor");
    }
  if (!tensor_.device ().is_cpu ())
    {
      return OBSIDIAN_ERROR (Error_Code::Device_Mismatch,
                             "span_f32_mut requires CPU tensor");
    }
  if (!tensor_.is_contiguous ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "span_f32_mut requires contiguous tensor");
    }
  return std::span<float> (tensor_.data_ptr<float> (), tensor_.numel ());
}

Result<Sequence_Batch>
Sequence_Batch::from_sequences (std::span<const std::vector<int64_t>> seqs,
                                int64_t pad_id,
                                size_t max_len,
                                bool left_pad)
{
  if (seqs.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "Sequence_Batch::from_sequences empty batch");
    }

  size_t longest = 0;
  for (const auto &s : seqs)
    {
      longest = std::max (longest, s.size ());
    }
  if (longest == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "all sequences in batch have length 0");
    }

  size_t target_len = (max_len > 0) ? std::min (longest, max_len) : longest;
  const size_t bsz = seqs.size ();

  Sequence_Batch batch;
  batch.input_ids = basalt::full ({bsz, target_len},
                                  static_cast<double> (pad_id),
                                  Tensor_Options ().with_dtype (Dtype::Int64));
  batch.attention_mask = basalt::zeros (
    {bsz, target_len}, Tensor_Options ().with_dtype (Dtype::Float32));
  batch.token_type_ids = basalt::zeros (
    {bsz, target_len}, Tensor_Options ().with_dtype (Dtype::Int64));
  batch.lengths.resize (bsz, 0);

  int64_t *ids_ptr = batch.input_ids.data_ptr<int64_t> ();
  float *mask_ptr = batch.attention_mask.data_ptr<float> ();

  for (size_t b = 0; b < bsz; ++b)
    {
      const auto &seq = seqs[b];
      size_t len = std::min (seq.size (), target_len);
      batch.lengths[b] = len;
      size_t start_col = left_pad ? (target_len - len) : 0;
      for (size_t j = 0; j < len; ++j)
        {
          ids_ptr[b * target_len + start_col + j] = seq[j];
          mask_ptr[b * target_len + start_col + j] = 1.0f;
        }
    }

  return batch;
}

namespace tensor_ops
{

Result<Tensor>
build_causal_mask (size_t seq_len, Device dev)
{
  if (seq_len == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "build_causal_mask requires seq_len > 0");
    }
  Tensor mask = basalt::zeros ({1, 1, seq_len, seq_len},
                               Tensor_Options ().with_dtype (Dtype::Float32));
  float *ptr = mask.data_ptr<float> ();
  for (size_t i = 0; i < seq_len; ++i)
    {
      for (size_t j = i + 1; j < seq_len; ++j)
        {
          ptr[i * seq_len + j] = -1e9f;
        }
    }
  if (!dev.is_cpu ())
    {
      mask = mask.to (dev);
    }
  return mask;
}

Result<Tensor>
build_padding_mask (const Tensor &attention_mask)
{
  if (!attention_mask.defined () || attention_mask.ndim () != 2)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "build_padding_mask expects [B, S] tensor");
    }
  size_t bsz = attention_mask.size (0);
  size_t seq_len = attention_mask.size (1);

  /* Compute (1.0 - mask) * (-1e9) with shape [B, 1, 1, S].  */
  Tensor fmask = attention_mask.to (Dtype::Float32);
  Tensor inv = basalt::ops::rsub_scalar (1.0, fmask);
  Tensor scaled = basalt::ops::mul_scalar (inv, -1e9);
  return scaled.reshape ({bsz, 1, 1, seq_len});
}

Result<Tensor>
build_alibi_bias (size_t num_heads, size_t seq_len, Device dev)
{
  if (num_heads == 0 || seq_len == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "build_alibi_bias requires positive dims");
    }
  Tensor bias = basalt::zeros ({1, num_heads, seq_len, seq_len},
                               Tensor_Options ().with_dtype (Dtype::Float32));
  float *ptr = bias.data_ptr<float> ();

  for (size_t h = 0; h < num_heads; ++h)
    {
      double exponent = -8.0 * static_cast<double> (h + 1)
                        / static_cast<double> (num_heads);
      float slope = static_cast<float> (std::pow (2.0, exponent));
      float *head_ptr = ptr + h * seq_len * seq_len;
      for (size_t i = 0; i < seq_len; ++i)
        {
          for (size_t j = 0; j < seq_len; ++j)
            {
              if (j <= i)
                {
                  float dist = static_cast<float> (i) - static_cast<float> (j);
                  head_ptr[i * seq_len + j] = -slope * dist;
                }
              else
                {
                  head_ptr[i * seq_len + j] = -1e9f;
                }
            }
        }
    }
  if (!dev.is_cpu ())
    {
      bias = bias.to (dev);
    }
  return bias;
}

Result<Tensor>
build_sinusoidal_positions (size_t max_len, size_t dim, Device dev)
{
  if (max_len == 0 || dim == 0 || (dim % 2) != 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "sinusoidal positions require even dim > 0");
    }
  Tensor pe = basalt::zeros ({max_len, dim},
                             Tensor_Options ().with_dtype (Dtype::Float32));
  float *ptr = pe.data_ptr<float> ();
  const double log_10000 = std::log (10000.0);

  for (size_t pos = 0; pos < max_len; ++pos)
    {
      for (size_t i = 0; i < dim; i += 2)
        {
          double scale = -log_10000 / static_cast<double> (dim);
          double div_term = std::exp (static_cast<double> (i) * scale);
          double angle = static_cast<double> (pos) * div_term;
          ptr[pos * dim + i] = static_cast<float> (std::sin (angle));
          ptr[pos * dim + i + 1] = static_cast<float> (std::cos (angle));
        }
    }
  if (!dev.is_cpu ())
    {
      pe = pe.to (dev);
    }
  return pe;
}

Result<Tensor>
masked_mean_pooling (const Tensor &hidden_states,
                     const Tensor &attention_mask)
{
  if (!hidden_states.defined () || hidden_states.ndim () != 3)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "masked_mean_pooling expects [B, S, D] states");
    }
  if (!attention_mask.defined () || attention_mask.ndim () != 2
      || attention_mask.size (0) != hidden_states.size (0)
      || attention_mask.size (1) != hidden_states.size (1))
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "masked_mean_pooling mask shape mismatch");
    }

  size_t bsz = hidden_states.size (0);
  size_t seq_len = hidden_states.size (1);
  size_t dim = hidden_states.size (2);

  Tensor mask_3d = attention_mask.to (hidden_states.dtype ())
                     .reshape ({bsz, seq_len, 1})
                     .expand ({bsz, seq_len, dim});
  Tensor masked = basalt::ops::mul (hidden_states, mask_3d);
  Tensor summed = basalt::ops::sum (masked, 1, false);

  Tensor counts = basalt::ops::sum (
    attention_mask.to (hidden_states.dtype ()), 1, true);
  Tensor safe_counts = basalt::ops::clamp (counts, 1.0, 1e12)
                         .expand ({bsz, dim});
  return basalt::ops::div (summed, safe_counts);
}

Result<Tensor>
last_token_pooling (const Tensor &hidden_states,
                    const Tensor &attention_mask)
{
  if (!hidden_states.defined () || hidden_states.ndim () != 3)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "last_token_pooling expects [B, S, D] states");
    }
  if (!attention_mask.defined () || attention_mask.ndim () != 2
      || attention_mask.size (0) != hidden_states.size (0)
      || attention_mask.size (1) != hidden_states.size (1))
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "last_token_pooling mask shape mismatch");
    }

  size_t bsz = hidden_states.size (0);
  size_t seq_len = hidden_states.size (1);
  std::vector<Tensor> rows;
  rows.reserve (bsz);

  for (size_t b = 0; b < bsz; ++b)
    {
      size_t last_idx = 0;
      for (size_t s = 0; s < seq_len; ++s)
        {
          double v = attention_mask.read_scalar_at (b * seq_len + s);
          if (v > 0.5)
            {
              last_idx = s;
            }
        }
      Tensor seq_slice = hidden_states.select (0, static_cast<ptrdiff_t> (b));
      Tensor tok_slice = seq_slice.select (
        0, static_cast<ptrdiff_t> (last_idx));
      rows.push_back (tok_slice);
    }

  return basalt::ops::stack (rows, 0);
}

Result<Tensor>
sequence_cross_entropy_loss (const Tensor &logits,
                             std::span<const int64_t> target_ids,
                             int64_t ignore_index,
                             double label_smoothing)
{
  if (!logits.defined () || (logits.ndim () != 2 && logits.ndim () != 3))
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "logits must be rank 2 [N, V] or rank 3 [B,S,V]");
    }
  if (label_smoothing < 0.0 || label_smoothing >= 1.0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "label_smoothing must be in [0.0, 1.0)");
    }

  size_t vocab_size = logits.size (-1);
  size_t num_tokens = logits.numel () / vocab_size;
  if (target_ids.size () != num_tokens)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "target_ids size must match token count");
    }

  Tensor flat_logits = (logits.ndim () == 3)
                         ? logits.reshape ({num_tokens, vocab_size})
                         : logits;
  Tensor log_probs = basalt::ops::log_softmax (flat_logits, -1);

  /* Build differentiable target weight matrix [num_tokens, vocab_size].  */
  Tensor weights = basalt::zeros (
    {num_tokens, vocab_size},
    Tensor_Options ().with_dtype (logits.dtype ())
                     .with_device (logits.device ()));

  size_t valid_count = 0;
  double smooth_val = label_smoothing / static_cast<double> (vocab_size);
  double on_val = (1.0 - label_smoothing) + smooth_val;

  for (size_t i = 0; i < num_tokens; ++i)
    {
      int64_t tgt = target_ids[i];
      if (tgt == ignore_index)
        {
          continue;
        }
      if (tgt < 0 || static_cast<size_t> (tgt) >= vocab_size)
        {
          return OBSIDIAN_ERROR (Error_Code::Out_Of_Range,
                                 "target token ID out of vocabulary range");
        }
      ++valid_count;
      if (label_smoothing > 0.0)
        {
          for (size_t v = 0; v < vocab_size; ++v)
            {
              weights.write_scalar_at (i * vocab_size + v, smooth_val);
            }
        }
      weights.write_scalar_at (
        i * vocab_size + static_cast<size_t> (tgt), on_val);
    }

  if (valid_count == 0)
    {
      return basalt::ops::mul_scalar (basalt::ops::sum (log_probs), 0.0);
    }

  Tensor weighted = basalt::ops::mul (log_probs, weights);
  Tensor total_nll = basalt::ops::neg (basalt::ops::sum (weighted));
  return basalt::ops::div_scalar (total_nll,
                                  static_cast<double> (valid_count));
}

} /* namespace tensor_ops */

} /* namespace obsidian */
