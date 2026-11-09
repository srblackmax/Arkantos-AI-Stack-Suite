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

#ifndef OBSIDIAN_MODELS_HPP
#define OBSIDIAN_MODELS_HPP

/**
 * @file models.hpp
 * @brief End-to-end Transformer NLP models (Bidirectional Encoder, Sequence &
 *        Token Classifiers, Causal Language Model with KV-cached generation,
 *        and Sequence-to-Sequence Encoder-Decoder).
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <basalt/basalt.hpp>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/tensor_bridge.hpp>
#include <obsidian/transformer.hpp>

namespace obsidian
{

/**
 * @brief Decoding hyperparameters for autoregressive and beam-search text
 *        generation.
 */
struct OBSIDIAN_API Generation_Config
{
  size_t max_new_tokens = 16;
  float temperature = 1.0f;
  size_t top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  size_t num_beams = 1;
  bool do_sample = false;
  int64_t eos_token_id = -1;
  uint64_t seed = 42;
};

/**
 * @brief Bidirectional Transformer Encoder (BERT / RoBERTa architecture).
 */
class OBSIDIAN_API Encoder_Model : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a bidirectional Encoder_Model.
   * @pre VOCAB_SIZE > 0 and CFG.d_model % CFG.num_heads == 0.
   * @complexity O(vocab_size * d_model + num_layers * d_model^2) space.
   */
  Encoder_Model (size_t vocab_size,
                 const Transformer_Config &cfg,
                 size_t num_token_types = 2,
                 ptrdiff_t pad_id = 0,
                 Tensor_Options options = Tensor_Options ());

  /**
   * @brief Compute contextual representations [B, S, d_model] from token IDs.
   * @pre INPUT_IDS has shape [B, S] with Dtype::Int64.
   * @complexity O(num_layers * B * S * (d_model^2 + S * d_model)).
   */
  Tensor
  forward (const Tensor &input_ids) override;

  /**
   * @brief Compute contextual representations over a padded Sequence_Batch.
   * @pre BATCH.input_ids is defined with shape [B, S].
   * @complexity O(num_layers * B * S * (d_model^2 + S * d_model)).
   */
  Tensor
  forward_batch (const Sequence_Batch &batch);

  /**
   * @brief Run inference over BATCH under No_Grad_Guard and eval() mode
   *        without allocating autograd graph nodes.
   * @pre BATCH.input_ids is defined with shape [B, S].
   * @complexity O(num_layers * B * S * (d_model^2 + S * d_model)).
   */
  Tensor
  infer_batch (const Sequence_Batch &batch);

  std::shared_ptr<basalt::nn::Embedding> word_embeddings;
  std::shared_ptr<basalt::nn::Embedding> position_embeddings;
  std::shared_ptr<basalt::nn::Embedding> token_type_embeddings;
  std::shared_ptr<basalt::nn::Dropout> embed_dropout;
  std::shared_ptr<Obsidian_Transformer_Stack> encoder;

private:
  size_t vocab_size_;
  Transformer_Config cfg_;
};

/**
 * @brief Transformer Sequence Classification model (e.g. sentiment, NLI).
 */
class OBSIDIAN_API Sequence_Classifier : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a Sequence_Classifier with NUM_CLASSES output logits.
   * @pre VOCAB_SIZE > 0 and NUM_CLASSES >= 2.
   * @complexity O(vocab_size * d_model + num_layers * d_model^2) space.
   */
  Sequence_Classifier (size_t vocab_size,
                       size_t num_classes,
                       const Transformer_Config &cfg,
                       bool use_mean_pooling = true,
                       Tensor_Options options = Tensor_Options ());

  /**
   * @brief Compute [B, num_classes] classification logits from INPUT_IDS.
   * @pre INPUT_IDS has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2).
   */
  Tensor
  forward (const Tensor &input_ids) override;

  /**
   * @brief Compute [B, num_classes] logits over a padded Sequence_Batch.
   * @pre BATCH.input_ids has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2).
   */
  Result<Tensor>
  forward_batch (const Sequence_Batch &batch);

  /**
   * @brief Run sequence classification inference under No_Grad_Guard and
   *        eval() mode.
   * @pre BATCH.input_ids has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2).
   */
  Result<Tensor>
  infer_batch (const Sequence_Batch &batch);

  std::shared_ptr<Encoder_Model> backbone;
  std::shared_ptr<basalt::nn::Linear> classifier;

private:
  bool use_mean_pooling_;
};

/**
 * @brief Transformer Token Classification model (e.g. NER, POS tagging).
 */
class OBSIDIAN_API Token_Classifier : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a Token_Classifier with NUM_TAGS per-token logits.
   * @pre VOCAB_SIZE > 0 and NUM_TAGS >= 2.
   * @complexity O(vocab_size * d_model + num_layers * d_model^2) space.
   */
  Token_Classifier (size_t vocab_size,
                    size_t num_tags,
                    const Transformer_Config &cfg,
                    Tensor_Options options = Tensor_Options ());

  /**
   * @brief Compute [B, S, num_tags] token logits from INPUT_IDS.
   * @pre INPUT_IDS has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2).
   */
  Tensor
  forward (const Tensor &input_ids) override;

  /**
   * @brief Compute [B, S, num_tags] logits over a padded Sequence_Batch.
   * @pre BATCH.input_ids has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2).
   */
  Tensor
  forward_batch (const Sequence_Batch &batch);

  /**
   * @brief Run token classification inference under No_Grad_Guard and eval()
   *        mode without retaining autograd state.
   * @pre BATCH.input_ids has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2).
   */
  Tensor
  infer_batch (const Sequence_Batch &batch);

  std::shared_ptr<Encoder_Model> backbone;
  std::shared_ptr<basalt::nn::Linear> classifier;
};

/**
 * @brief Autoregressive Causal Language Model (GPT / LLaMA architecture)
 *        with KV-cached incremental decoding and sampling/beam search.
 */
class OBSIDIAN_API Causal_LM : public basalt::nn::Module
{
public:
  /**
   * @brief Construct an autoregressive Causal_LM.
   * @pre VOCAB_SIZE > 0 and CFG.d_model % CFG.num_heads == 0.
   * @complexity O(vocab_size * d_model + num_layers * d_model^2) space.
   */
  Causal_LM (size_t vocab_size,
             Transformer_Config cfg,
             Tensor_Options options = Tensor_Options ());

  /**
   * @brief Compute [B, S, vocab_size] next-token logits from INPUT_IDS.
   * @pre INPUT_IDS has shape [B, S] with Dtype::Int64.
   * @complexity O(num_layers * B * S * d_model^2 + B * S * d_model * V).
   */
  Tensor
  forward (const Tensor &input_ids) override;

  /**
   * @brief Forward pass with optional attention mask and per-layer KV_Cache.
   * @pre INPUT_IDS has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2 + B * S * d_model * V).
   */
  Tensor
  forward_ids (const Tensor &input_ids,
               const Tensor &attention_mask = Tensor (),
               std::vector<basalt::nn::KV_Cache> *kv_caches = nullptr);

  /**
   * @brief Generate continuation token IDs from PROMPT_IDS using KV-cached
   *        greedy, temperature/top-k/top-p sampling, or beam search.
   * @pre PROMPT_IDS must not be empty; all IDs < vocab_size().
   * @complexity O((|PROMPT| + max_new_tokens) * num_layers * d_model^2).
   */
  Result<std::vector<int64_t>>
  generate (std::span<const int64_t> prompt_ids,
            const Generation_Config &gen_cfg = Generation_Config ());

  /**
   * @brief Generate continuations for a batch of variable-length token
   *        prompts.
   * @pre PROMPTS must not be empty and each prompt must be non-empty.
   * @complexity O(B * (S_prompt + max_new_tokens) * num_layers * d_model^2).
   */
  Result<std::vector<std::vector<int64_t>>>
  generate_batch (const std::vector<std::vector<int64_t>> &prompts,
                  const Generation_Config &gen_cfg = Generation_Config ());

  /**
   * @brief Run next-token logit inference under No_Grad_Guard and eval() mode.
   * @pre INPUT_IDS has shape [B, S].
   * @complexity O(num_layers * B * S * d_model^2 + B * S * d_model * V).
   */
  Tensor
  infer (const Tensor &input_ids);

  /**
   * @brief Return the vocabulary size V.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_size_;
  }

  std::shared_ptr<basalt::nn::Embedding> tok_embeddings;
  std::shared_ptr<basalt::nn::Embedding> pos_embeddings;
  std::shared_ptr<Obsidian_Transformer_Stack> decoder;
  std::shared_ptr<basalt::nn::Linear> lm_head;

private:
  size_t vocab_size_;
  Transformer_Config cfg_;
};

/**
 * @brief Sequence-to-Sequence Encoder-Decoder Transformer (T5 / BART style).
 */
class OBSIDIAN_API Seq2Seq_Model : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a Seq2Seq_Model with encoder and cross-attending decoder.
   * @pre VOCAB_SIZE > 0 and CFG.d_model % CFG.num_heads == 0.
   * @complexity O(vocab_size * d_model + 2 * num_layers * d_model^2) space.
   */
  Seq2Seq_Model (size_t vocab_size,
                 const Transformer_Config &cfg,
                 Tensor_Options options = Tensor_Options ());

  /**
   * @brief Compute decoder logits [B, S_tgt, vocab_size] given SRC_IDS and
   *        TGT_IDS.
   * @pre SRC_IDS is [B, S_src] and TGT_IDS is [B, S_tgt].
   * @complexity O(num_layers * B * (S_src + S_tgt) * d_model^2).
   */
  Tensor
  forward_seq2seq (const Tensor &src_ids,
                   const Tensor &tgt_ids,
                   const Tensor &src_mask = Tensor (),
                   const Tensor &tgt_mask = Tensor ());

  /**
   * @brief Run encoder-decoder inference under No_Grad_Guard and eval() mode.
   * @pre SRC_IDS is [B, S_src] and TGT_IDS is [B, S_tgt].
   * @complexity O(num_layers * B * (S_src + S_tgt) * d_model^2).
   */
  Tensor
  infer_seq2seq (const Tensor &src_ids,
                 const Tensor &tgt_ids,
                 const Tensor &src_mask = Tensor (),
                 const Tensor &tgt_mask = Tensor ());

  std::shared_ptr<basalt::nn::Embedding> shared_embeddings;
  std::shared_ptr<Obsidian_Transformer_Stack> encoder;
  std::shared_ptr<Obsidian_Transformer_Stack> decoder;
  std::shared_ptr<basalt::nn::Linear> lm_head;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_MODELS_HPP */
