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

#ifndef OBSIDIAN_TRANSFORMER_HPP
#define OBSIDIAN_TRANSFORMER_HPP

/**
 * @file transformer.hpp
 * @brief Cache-optimized Transformer blocks built on Basalt: Fused-QKV
 *        Multi-Head / Grouped-Query Attention, RoPE, ALiBi, SwiGLU/GeGLU
 *        feed-forward layers, and Encoder/Decoder stacks.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <basalt/basalt.hpp>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/tensor_bridge.hpp>

namespace obsidian
{

/**
 * @brief Positional encoding strategy for Transformer attention blocks.
 */
enum class Position_Encoding_Kind : uint8_t
{
  None = 0,
  Learned_Absolute = 1,
  Sinusoidal = 2,
  Rotary_RoPE = 3,
  ALiBi = 4
};

/**
 * @brief Layer normalization strategy for Transformer residual blocks.
 */
enum class Norm_Kind : uint8_t
{
  Layer_Norm = 0,
  RMS_Norm = 1
};

/**
 * @brief Feed-forward network activation and gating architecture.
 */
enum class FFN_Activation_Kind : uint8_t
{
  GELU = 0,
  ReLU = 1,
  SwiGLU = 2,
  GeGLU = 3
};

/**
 * @brief Hyperparameter configuration for a Transformer block or stack.
 */
struct OBSIDIAN_API Transformer_Config
{
  size_t d_model = 64;
  size_t num_heads = 4;
  size_t num_kv_heads = 4;
  size_t d_ff = 256;
  size_t num_layers = 2;
  size_t max_seq_len = 512;
  double dropout_p = 0.0;
  double norm_eps = 1e-5;
  double rope_base = 10000.0;
  bool pre_norm = true;
  bool use_bias = true;
  bool is_causal = false;
  bool has_cross_attention = false;
  Position_Encoding_Kind pos_encoding = Position_Encoding_Kind::Rotary_RoPE;
  Norm_Kind norm_kind = Norm_Kind::RMS_Norm;
  FFN_Activation_Kind ffn_activation = FFN_Activation_Kind::SwiGLU;
};

/**
 * @brief Fused-QKV Multi-Head / Grouped-Query / Multi-Query Attention module
 *        built on basalt::nn::Module.
 *
 * Uses a single fused linear projection for (Q, K, V) in self-attention to
 * maximize L1/L2 cache locality and minimize GEMM launch overhead.
 */
class OBSIDIAN_API Grouped_Query_Attention : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a Grouped_Query_Attention module.
   * @pre D_MODEL % NUM_HEADS == 0 and NUM_HEADS % NUM_KV_HEADS == 0.
   * @complexity O(d_model^2) parameter allocation time and space.
   */
  Grouped_Query_Attention (
    size_t d_model,
    size_t num_heads,
    size_t num_kv_heads = 0,
    double dropout_p = 0.0,
    bool use_bias = true,
    bool use_rope = false,
    size_t max_seq_len = 2048,
    double rope_base = 10000.0,
    Tensor_Options options = Tensor_Options ());

  /**
   * @brief Compute self-attention over INPUT of shape [B, S, d_model].
   * @pre INPUT has shape [B, S, d_model].
   * @complexity O(B * S * d_model^2 + B * num_heads * S^2 * head_dim).
   */
  Tensor
  forward (const Tensor &input) override;

  /**
   * @brief Compute self-attention with optional additive mask, causal flag,
   *        and incremental KV_Cache.
   * @pre INPUT has shape [B, S, d_model].
   * @complexity O(B * S * d_model^2 + B * H * S * S_kv * head_dim).
   */
  Tensor
  forward_self (const Tensor &input,
                const Tensor &attn_mask = Tensor (),
                bool is_causal = false,
                basalt::nn::KV_Cache *kv_cache = nullptr);

  /**
   * @brief Compute cross-attention from QUERY [B, S_q, D] to MEMORY
   *        [B, S_k, D].
   * @pre QUERY and MEMORY have last dimension d_model.
   * @complexity O(B * (S_q + S_k) * d_model^2 + B * H * S_q * S_k * head_dim).
   */
  Tensor
  forward_cross (const Tensor &query,
                 const Tensor &memory,
                 const Tensor &memory_mask = Tensor ());

  std::shared_ptr<basalt::nn::Linear> qkv_proj;
  std::shared_ptr<basalt::nn::Linear> q_proj;
  std::shared_ptr<basalt::nn::Linear> kv_proj;
  std::shared_ptr<basalt::nn::Linear> out_proj;
  std::shared_ptr<basalt::nn::Rotary_Embedding> rope;

private:
  Tensor
  repeat_kv_heads (const Tensor &kv) const;

  size_t d_model_;
  size_t num_heads_;
  size_t num_kv_heads_;
  size_t head_dim_;
  size_t kv_dim_;
  double dropout_p_;
  bool use_rope_;
};

/**
 * @brief Cache-optimized Feed-Forward Network supporting standard GELU/ReLU
 *        and fused-projection SwiGLU/GeGLU gated architectures.
 */
class OBSIDIAN_API Feed_Forward_Block : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a Feed_Forward_Block.
   * @pre D_MODEL > 0 and D_FF > 0.
   * @complexity O(d_model * d_ff) parameter allocation time and space.
   */
  Feed_Forward_Block (
    size_t d_model,
    size_t d_ff,
    FFN_Activation_Kind act_kind = FFN_Activation_Kind::SwiGLU,
    double dropout_p = 0.0,
    bool use_bias = true,
    Tensor_Options options = Tensor_Options ());

  /**
   * @brief Apply the feed-forward transformation to INPUT [B, S, d_model].
   * @pre INPUT last dimension equals d_model.
   * @complexity O(B * S * d_model * d_ff) time.
   */
  Tensor
  forward (const Tensor &input) override;

  std::shared_ptr<basalt::nn::Linear> up_proj;
  std::shared_ptr<basalt::nn::Linear> down_proj;
  std::shared_ptr<basalt::nn::Dropout> dropout;

private:
  size_t d_model_;
  size_t d_ff_;
  FFN_Activation_Kind act_kind_;
  bool is_gated_;
};

/**
 * @brief Configurable Pre-Norm / Post-Norm Transformer Layer supporting
 *        self-attention, optional cross-attention, and feed-forward blocks.
 */
class OBSIDIAN_API Obsidian_Transformer_Layer : public basalt::nn::Module
{
public:
  /**
   * @brief Construct an Obsidian_Transformer_Layer from CFG.
   * @pre CFG.d_model % CFG.num_heads == 0.
   * @complexity O(d_model^2 + d_model * d_ff) space.
   */
  explicit Obsidian_Transformer_Layer (
    const Transformer_Config &cfg,
    Tensor_Options options = Tensor_Options ());

  /**
   * @brief Forward pass for encoder or causal decoder self-attention.
   * @pre X has shape [B, S, d_model].
   * @complexity O(B * S * d_model^2 + B * H * S^2 * head_dim).
   */
  Tensor
  forward (const Tensor &x) override;

  /**
   * @brief Extended forward pass with attention mask, optional encoder memory,
   *        and incremental KV_Cache.
   * @pre X has shape [B, S, d_model].
   * @complexity O(B * S * (d_model^2 + d_model * d_ff) + B * H * S^2 * D_h).
   */
  Tensor
  forward_ext (const Tensor &x,
               const Tensor &self_mask = Tensor (),
               const Tensor &encoder_memory = Tensor (),
               const Tensor &cross_mask = Tensor (),
               basalt::nn::KV_Cache *kv_cache = nullptr);

  std::shared_ptr<Grouped_Query_Attention> self_attn;
  std::shared_ptr<Grouped_Query_Attention> cross_attn;
  std::shared_ptr<Feed_Forward_Block> ffn;
  std::shared_ptr<basalt::nn::Module> norm1;
  std::shared_ptr<basalt::nn::Module> norm2;
  std::shared_ptr<basalt::nn::Module> norm_cross;
  std::shared_ptr<basalt::nn::Dropout> resid_dropout;

private:
  Transformer_Config cfg_;
};

/**
 * @brief Multi-layer stack of Obsidian_Transformer_Layer modules with
 *        optional ALiBi bias generation and final normalization.
 */
class OBSIDIAN_API Obsidian_Transformer_Stack : public basalt::nn::Module
{
public:
  /**
   * @brief Construct a stack of CFG.num_layers Transformer blocks.
   * @pre CFG.num_layers >= 1 and CFG.d_model % CFG.num_heads == 0.
   * @complexity O(num_layers * (d_model^2 + d_model * d_ff)) space.
   */
  explicit Obsidian_Transformer_Stack (
    const Transformer_Config &cfg,
    Tensor_Options options = Tensor_Options ());

  /**
   * @brief Run input embeddings [B, S, d_model] through all layers.
   * @pre HIDDEN_STATES has shape [B, S, d_model].
   * @complexity O(num_layers * B * S * (d_model^2 + d_model * d_ff)).
   */
  Tensor
  forward (const Tensor &hidden_states) override;

  /**
   * @brief Extended stack forward pass supporting padding masks,
   *        cross-attention memory, and per-layer KV_Cache vector.
   * @pre HIDDEN_STATES has shape [B, S, d_model].
   * @complexity O(num_layers * B * S * (d_model^2 + d_model * d_ff)).
   */
  Tensor
  forward_ext (const Tensor &hidden_states,
               const Tensor &attention_mask = Tensor (),
               const Tensor &encoder_memory = Tensor (),
               const Tensor &cross_mask = Tensor (),
               std::vector<basalt::nn::KV_Cache> *kv_caches = nullptr);

  /**
   * @brief Return the configuration of this stack.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const Transformer_Config &
  config (void) const noexcept
  {
    return cfg_;
  }

  std::vector<std::shared_ptr<Obsidian_Transformer_Layer>> layers;
  std::shared_ptr<basalt::nn::Module> final_norm;

private:
  Transformer_Config cfg_;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_TRANSFORMER_HPP */
