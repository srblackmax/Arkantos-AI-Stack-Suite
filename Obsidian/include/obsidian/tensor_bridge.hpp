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

#ifndef OBSIDIAN_TENSOR_BRIDGE_HPP
#define OBSIDIAN_TENSOR_BRIDGE_HPP

/**
 * @file tensor_bridge.hpp
 * @brief Basalt Deep Learning tensor wrapper, sequence batching, and NLP ops.
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <vector>
#include <basalt/basalt.hpp>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>

namespace obsidian
{

using basalt::Device;
using basalt::Device_Type;
using basalt::Dtype;
using basalt::Enable_Grad_Guard;
using basalt::Grad_Mode;
using basalt::No_Grad_Guard;
using basalt::Shape_Vector;
using basalt::Stride_Vector;
using basalt::Tensor;
using basalt::Tensor_Options;

/**
 * @brief Concept satisfied by ranges of integral token identifiers.
 */
template <typename R>
concept Token_ID_Range = std::ranges::input_range<R>
  && std::convertible_to<std::ranges::range_reference_t<R>, int64_t>;

/**
 * @brief Validated wrapper around basalt::Tensor providing Result-based
 *        precondition checks and zero-copy host span access.
 */
class OBSIDIAN_API Tensor_Block
{
public:
  /**
   * @brief Construct an empty undefined Tensor_Block.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Tensor_Block (void) noexcept = default;

  /**
   * @brief Wrap an existing basalt::Tensor handle (O(1) shallow copy).
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  explicit Tensor_Block (Tensor t) noexcept
    : tensor_ (std::move (t))
  {
  }

  /**
   * @brief Create a zero-initialized Tensor_Block of the given shape.
   * @pre Every dimension in SHAPE must be > 0.
   * @complexity O(numel) time and space.
   */
  static Result<Tensor_Block>
  zeros (const Shape_Vector &shape,
         Tensor_Options opts = Tensor_Options ());

  /**
   * @brief Create a Tensor_Block copied from a contiguous float32 span.
   * @pre DATA.size() must equal the product of SHAPE dimensions.
   * @complexity O(numel) time and space.
   */
  static Result<Tensor_Block>
  from_span_f32 (std::span<const float> data,
                 const Shape_Vector &shape,
                 bool requires_grad = false);

  /**
   * @brief Create a 1-D or 2-D Int64 Tensor_Block from token IDs.
   * @pre IDS.size() == batch_size * seq_len.
   * @complexity O(N) time and space in number of tokens N.
   */
  static Result<Tensor_Block>
  from_token_ids (std::span<const int64_t> ids,
                  size_t batch_size,
                  size_t seq_len);

  /**
   * @brief Return a reference to the underlying basalt::Tensor.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Tensor &
  raw (void) noexcept
  {
    return tensor_;
  }

  /**
   * @brief Return a const reference to the underlying basalt::Tensor.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const Tensor &
  raw (void) const noexcept
  {
    return tensor_;
  }

  /**
   * @brief Implicit conversion to const basalt::Tensor&.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  operator const Tensor & (void) const noexcept
  {
    return tensor_;
  }

  /**
   * @brief Return true if the underlying basalt::Tensor is defined.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  bool
  defined (void) const noexcept
  {
    return tensor_.defined ();
  }

  /**
   * @brief Verify that the tensor is defined and has the expected rank.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Result<void>
  expect_rank (size_t expected_ndim) const;

  /**
   * @brief Verify that the tensor is defined and matches EXPECTED_SHAPE.
   * @pre None.
   * @complexity O(ndim) time and O(1) space.
   */
  Result<void>
  expect_shape (const Shape_Vector &expected_shape) const;

  /**
   * @brief Verify that the tensor has the expected scalar Dtype.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Result<void>
  expect_dtype (Dtype expected_dtype) const;

  /**
   * @brief Return a zero-copy const float span over contiguous CPU storage.
   * @pre Tensor must be defined, contiguous, on CPU, and Dtype::Float32.
   * @complexity O(1) time and O(1) space.
   */
  Result<std::span<const float>>
  span_f32 (void) const;

  /**
   * @brief Return a zero-copy mutable float span over contiguous CPU storage.
   * @pre Tensor must be defined, contiguous, on CPU, and Dtype::Float32.
   * @complexity O(1) time and O(1) space.
   */
  Result<std::span<float>>
  span_f32_mut (void);

private:
  Tensor tensor_;
};

/**
 * @brief Padded batch of tokenized sequences ready for Transformer execution.
 */
struct OBSIDIAN_API Sequence_Batch
{
  Tensor input_ids;
  Tensor attention_mask;
  Tensor token_type_ids;
  std::vector<size_t> lengths;

  /**
   * @brief Return the batch dimension B.
   * @pre input_ids is defined with rank 2.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  batch_size (void) const noexcept
  {
    return input_ids.defined () ? input_ids.size (0) : 0;
  }

  /**
   * @brief Return the padded sequence length S.
   * @pre input_ids is defined with rank 2.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  seq_len (void) const noexcept
  {
    return input_ids.defined () ? input_ids.size (1) : 0;
  }

  /**
   * @brief Collate variable-length token ID sequences into a padded batch.
   * @pre SEQS must not be empty; at least one sequence must be non-empty.
   * @complexity O(B * S_max) time and space.
   */
  static Result<Sequence_Batch>
  from_sequences (std::span<const std::vector<int64_t>> seqs,
                  int64_t pad_id = 0,
                  size_t max_len = 0,
                  bool left_pad = false);
};

namespace tensor_ops
{

/**
 * @brief Build an upper-triangular causal additive attention mask.
 *
 * Returns a Float32 tensor of shape [1, 1, seq_len, seq_len] with 0.0 on and
 * below the main diagonal and -1e9 strictly above the main diagonal.
 *
 * @pre SEQ_LEN > 0.
 * @complexity O(seq_len^2) time and space.
 */
OBSIDIAN_API Result<Tensor>
build_causal_mask (size_t seq_len, Device dev = Device::cpu ());

/**
 * @brief Convert a [batch, seq_len] binary mask (1=valid, 0=pad) into an
 *        additive [batch, 1, 1, seq_len] attention bias (0.0=valid, -1e9=pad).
 * @pre ATTENTION_MASK has shape [B, S].
 * @complexity O(B * S) time and space.
 */
OBSIDIAN_API Result<Tensor>
build_padding_mask (const Tensor &attention_mask);

/**
 * @brief Build an ALiBi (Attention with Linear Biases) relative bias tensor
 *        of shape [1, num_heads, seq_len, seq_len].
 * @pre NUM_HEADS > 0 and SEQ_LEN > 0.
 * @complexity O(num_heads * seq_len^2) time and space.
 */
OBSIDIAN_API Result<Tensor>
build_alibi_bias (size_t num_heads, size_t seq_len,
                  Device dev = Device::cpu ());

/**
 * @brief Build a sinusoidal positional encoding table of shape [max_len, dim].
 * @pre MAX_LEN > 0 and DIM > 0 (DIM must be even).
 * @complexity O(max_len * dim) time and space.
 */
OBSIDIAN_API Result<Tensor>
build_sinusoidal_positions (size_t max_len, size_t dim,
                            Device dev = Device::cpu ());

/**
 * @brief Differentiable attention-mask-weighted mean pooling over sequence
 *        hidden states of shape [batch, seq_len, hidden_dim].
 * @pre HIDDEN_STATES is [B, S, D] and ATTENTION_MASK is [B, S].
 * @complexity O(B * S * D) time and O(B * D) output space.
 */
OBSIDIAN_API Result<Tensor>
masked_mean_pooling (const Tensor &hidden_states,
                     const Tensor &attention_mask);

/**
 * @brief Extract the hidden state of the last valid (non-padded) token for
 *        each sequence in the batch, returning shape [batch, hidden_dim].
 * @pre HIDDEN_STATES is [B, S, D] and ATTENTION_MASK is [B, S].
 * @complexity O(B * S + B * D) time and O(B * D) output space.
 */
OBSIDIAN_API Result<Tensor>
last_token_pooling (const Tensor &hidden_states,
                    const Tensor &attention_mask);

/**
 * @brief Differentiable sequence cross-entropy loss with ignore_index and
 *        optional label smoothing over [batch, seq_len, vocab] or [N, vocab].
 * @pre LOGITS last dim is vocab_size; TARGET_IDS has N = B * S elements.
 * @complexity O(N * V) time and space via Basalt log_softmax.
 */
OBSIDIAN_API Result<Tensor>
sequence_cross_entropy_loss (const Tensor &logits,
                             std::span<const int64_t> target_ids,
                             int64_t ignore_index = -100,
                             double label_smoothing = 0.0);

} /* namespace tensor_ops */

} /* namespace obsidian */

#endif /* OBSIDIAN_TENSOR_BRIDGE_HPP */
