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

#include <obsidian/transformer.hpp>

#include <algorithm>

namespace obsidian
{

namespace
{

std::shared_ptr<basalt::nn::Module>
make_norm_layer (Norm_Kind kind, size_t dim, double eps,
                 Tensor_Options options)
{
  if (kind == Norm_Kind::RMS_Norm)
    {
      return std::make_shared<basalt::nn::RMS_Norm> (
        dim, eps, true, options);
    }
  return std::make_shared<basalt::nn::Layer_Norm> (
    dim, eps, true, options);
}

} /* namespace */

/* -------------------------------------------------------------------------
   Grouped_Query_Attention implementation
   ------------------------------------------------------------------------- */

Grouped_Query_Attention::Grouped_Query_Attention (
  size_t d_model,
  size_t num_heads,
  size_t num_kv_heads,
  double dropout_p,
  bool use_bias,
  bool use_rope,
  size_t max_seq_len,
  double rope_base,
  Tensor_Options options)
  : d_model_ (d_model),
    num_heads_ (num_heads),
    num_kv_heads_ (num_kv_heads == 0 ? num_heads : num_kv_heads),
    head_dim_ (d_model / std::max (num_heads, size_t (1))),
    kv_dim_ (num_kv_heads_ * head_dim_),
    dropout_p_ (dropout_p),
    use_rope_ (use_rope)
{
  BASALT_ASSERT_ARG (d_model_ > 0 && num_heads_ > 0
                     && (d_model_ % num_heads_) == 0,
                     "d_model must be divisible by num_heads");
  BASALT_ASSERT_ARG (num_kv_heads_ > 0
                     && (num_heads_ % num_kv_heads_) == 0,
                     "num_heads must be divisible by num_kv_heads");

  /* Fused QKV projection for self-attention cache locality.  */
  size_t fused_out = d_model_ + 2 * kv_dim_;
  qkv_proj = register_module (
    "qkv_proj",
    std::make_shared<basalt::nn::Linear> (
      d_model_, fused_out, use_bias, options));

  /* Separate Q and fused KV projections for cross-attention.  */
  q_proj = register_module (
    "q_proj",
    std::make_shared<basalt::nn::Linear> (
      d_model_, d_model_, use_bias, options));
  kv_proj = register_module (
    "kv_proj",
    std::make_shared<basalt::nn::Linear> (
      d_model_, 2 * kv_dim_, use_bias, options));

  out_proj = register_module (
    "out_proj",
    std::make_shared<basalt::nn::Linear> (
      d_model_, d_model_, use_bias, options));

  if (use_rope_)
    {
      rope = register_module (
        "rope",
        std::make_shared<basalt::nn::Rotary_Embedding> (
          head_dim_, max_seq_len, rope_base, options));
    }
}

Tensor
Grouped_Query_Attention::repeat_kv_heads (const Tensor &kv) const
{
  /* KV has shape [B, num_kv_heads, S, head_dim].  */
  if (num_kv_heads_ == num_heads_)
    {
      return kv;
    }
  size_t repeats = num_heads_ / num_kv_heads_;
  size_t B = kv.size (0);
  size_t S = kv.size (2);

  std::vector<Tensor> expanded_heads;
  expanded_heads.reserve (num_heads_);
  for (size_t h = 0; h < num_kv_heads_; ++h)
    {
      Tensor single_h = kv.narrow (1, static_cast<ptrdiff_t> (h), 1);
      for (size_t r = 0; r < repeats; ++r)
        {
          expanded_heads.push_back (single_h);
        }
    }
  return basalt::ops::cat (expanded_heads, 1)
           .reshape ({B, num_heads_, S, head_dim_});
}

Tensor
Grouped_Query_Attention::forward (const Tensor &input)
{
  return forward_self (input);
}

Tensor
Grouped_Query_Attention::forward_self (const Tensor &input,
                                       const Tensor &attn_mask,
                                       bool is_causal,
                                       basalt::nn::KV_Cache *kv_cache)
{
  BASALT_ASSERT_SHAPE (input.defined () && input.ndim () == 3
                       && input.size (2) == d_model_,
                       "Grouped_Query_Attention expects [B, S, d_model]");

  const size_t B = input.size (0);
  const size_t S = input.size (1);

  /* Single fused GEMM producing [B, S, d_model + 2 * kv_dim].  */
  Tensor qkv = qkv_proj->forward (input);

  Tensor q_raw = qkv.narrow (2, 0, d_model_).contiguous ();
  Tensor k_raw = qkv.narrow (
    2, static_cast<ptrdiff_t> (d_model_), kv_dim_).contiguous ();
  Tensor v_raw = qkv.narrow (
    2, static_cast<ptrdiff_t> (d_model_ + kv_dim_), kv_dim_).contiguous ();

  /* Reshape to [B, H, S, D_h].  */
  Tensor q = q_raw.reshape ({B, S, num_heads_, head_dim_})
               .transpose (1, 2)
               .contiguous ();
  Tensor k = k_raw.reshape ({B, S, num_kv_heads_, head_dim_})
               .transpose (1, 2)
               .contiguous ();
  Tensor v = v_raw.reshape ({B, S, num_kv_heads_, head_dim_})
               .transpose (1, 2)
               .contiguous ();

  if (use_rope_ && rope != nullptr)
    {
      q = rope->forward (q);
      k = rope->forward (k);
    }

  Tensor k_full = repeat_kv_heads (k);
  Tensor v_full = repeat_kv_heads (v);

  if (kv_cache != nullptr)
    {
      auto [k_cached, v_cached] = kv_cache->update (k_full, v_full);
      k_full = k_cached;
      v_full = v_cached;
    }

  bool causal_flag = is_causal && (kv_cache == nullptr || S > 1);
  Tensor attn_out = basalt::nn::functional::scaled_dot_product_attention (
    q, k_full, v_full, attn_mask, dropout_p_, causal_flag, is_training ());

  Tensor merged = attn_out.transpose (1, 2)
                    .contiguous ()
                    .reshape ({B, S, d_model_});
  return out_proj->forward (merged);
}

Tensor
Grouped_Query_Attention::forward_cross (const Tensor &query,
                                        const Tensor &memory,
                                        const Tensor &memory_mask)
{
  BASALT_ASSERT_SHAPE (query.defined () && query.ndim () == 3
                       && memory.defined () && memory.ndim () == 3,
                       "forward_cross expects 3D query and memory");

  const size_t B = query.size (0);
  const size_t S_q = query.size (1);
  const size_t S_k = memory.size (1);

  Tensor q_raw = q_proj->forward (query);
  Tensor kv_raw = kv_proj->forward (memory);

  Tensor k_raw = kv_raw.narrow (2, 0, kv_dim_).contiguous ();
  Tensor v_raw = kv_raw.narrow (
    2, static_cast<ptrdiff_t> (kv_dim_), kv_dim_).contiguous ();

  Tensor q = q_raw.reshape ({B, S_q, num_heads_, head_dim_})
               .transpose (1, 2)
               .contiguous ();
  Tensor k = k_raw.reshape ({B, S_k, num_kv_heads_, head_dim_})
               .transpose (1, 2)
               .contiguous ();
  Tensor v = v_raw.reshape ({B, S_k, num_kv_heads_, head_dim_})
               .transpose (1, 2)
               .contiguous ();

  Tensor k_full = repeat_kv_heads (k);
  Tensor v_full = repeat_kv_heads (v);

  Tensor attn_out = basalt::nn::functional::scaled_dot_product_attention (
    q, k_full, v_full, memory_mask, dropout_p_, false, is_training ());

  Tensor merged = attn_out.transpose (1, 2)
                    .contiguous ()
                    .reshape ({B, S_q, d_model_});
  return out_proj->forward (merged);
}

/* -------------------------------------------------------------------------
   Feed_Forward_Block implementation
   ------------------------------------------------------------------------- */

Feed_Forward_Block::Feed_Forward_Block (
  size_t d_model,
  size_t d_ff,
  FFN_Activation_Kind act_kind,
  double dropout_p,
  bool use_bias,
  Tensor_Options options)
  : d_model_ (d_model),
    d_ff_ (d_ff),
    act_kind_ (act_kind),
    is_gated_ (act_kind == FFN_Activation_Kind::SwiGLU
               || act_kind == FFN_Activation_Kind::GeGLU)
{
  size_t up_out = is_gated_ ? (2 * d_ff_) : d_ff_;
  up_proj = register_module (
    "up_proj",
    std::make_shared<basalt::nn::Linear> (
      d_model_, up_out, use_bias, options));
  down_proj = register_module (
    "down_proj",
    std::make_shared<basalt::nn::Linear> (
      d_ff_, d_model_, use_bias, options));
  dropout = register_module (
    "dropout",
    std::make_shared<basalt::nn::Dropout> (dropout_p));
}

Tensor
Feed_Forward_Block::forward (const Tensor &input)
{
  Tensor projected = up_proj->forward (input);
  Tensor hidden;

  if (is_gated_)
    {
      Tensor gate = projected.narrow (2, 0, d_ff_).contiguous ();
      Tensor up = projected.narrow (
        2, static_cast<ptrdiff_t> (d_ff_), d_ff_).contiguous ();
      Tensor act_gate = (act_kind_ == FFN_Activation_Kind::SwiGLU)
                          ? basalt::ops::silu (gate)
                          : basalt::ops::gelu (gate);
      hidden = basalt::ops::mul (act_gate, up);
    }
  else
    {
      hidden = (act_kind_ == FFN_Activation_Kind::ReLU)
                 ? basalt::ops::relu (projected)
                 : basalt::ops::gelu (projected);
    }

  hidden = dropout->forward (hidden);
  return down_proj->forward (hidden);
}

/* -------------------------------------------------------------------------
   Obsidian_Transformer_Layer implementation
   ------------------------------------------------------------------------- */

Obsidian_Transformer_Layer::Obsidian_Transformer_Layer (
  const Transformer_Config &cfg,
  Tensor_Options options)
  : cfg_ (cfg)
{
  bool use_rope = (cfg.pos_encoding == Position_Encoding_Kind::Rotary_RoPE);
  self_attn = register_module (
    "self_attn",
    std::make_shared<Grouped_Query_Attention> (
      cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout_p,
      cfg.use_bias, use_rope, cfg.max_seq_len, cfg.rope_base, options));

  if (cfg.has_cross_attention)
    {
      cross_attn = register_module (
        "cross_attn",
        std::make_shared<Grouped_Query_Attention> (
          cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout_p,
          cfg.use_bias, false, cfg.max_seq_len, cfg.rope_base, options));
      norm_cross = register_module (
        "norm_cross",
        make_norm_layer (cfg.norm_kind, cfg.d_model, cfg.norm_eps, options));
    }

  ffn = register_module (
    "ffn",
    std::make_shared<Feed_Forward_Block> (
      cfg.d_model, cfg.d_ff, cfg.ffn_activation, cfg.dropout_p,
      cfg.use_bias, options));

  norm1 = register_module (
    "norm1",
    make_norm_layer (cfg.norm_kind, cfg.d_model, cfg.norm_eps, options));
  norm2 = register_module (
    "norm2",
    make_norm_layer (cfg.norm_kind, cfg.d_model, cfg.norm_eps, options));
  resid_dropout = register_module (
    "resid_dropout",
    std::make_shared<basalt::nn::Dropout> (cfg.dropout_p));
}

Tensor
Obsidian_Transformer_Layer::forward (const Tensor &x)
{
  return forward_ext (x);
}

Tensor
Obsidian_Transformer_Layer::forward_ext (const Tensor &x,
                                         const Tensor &self_mask,
                                         const Tensor &encoder_memory,
                                         const Tensor &cross_mask,
                                         basalt::nn::KV_Cache *kv_cache)
{
  Tensor h = x;

  if (cfg_.pre_norm)
    {
      Tensor normed = norm1->forward (h);
      Tensor attn_out = self_attn->forward_self (
        normed, self_mask, cfg_.is_causal, kv_cache);
      h = h + resid_dropout->forward (attn_out);

      if (cfg_.has_cross_attention && cross_attn != nullptr
          && encoder_memory.defined ())
        {
          Tensor c_norm = norm_cross->forward (h);
          Tensor c_out = cross_attn->forward_cross (
            c_norm, encoder_memory, cross_mask);
          h = h + resid_dropout->forward (c_out);
        }

      Tensor ffn_in = norm2->forward (h);
      h = h + resid_dropout->forward (ffn->forward (ffn_in));
    }
  else
    {
      Tensor attn_out = self_attn->forward_self (
        h, self_mask, cfg_.is_causal, kv_cache);
      h = norm1->forward (h + resid_dropout->forward (attn_out));

      if (cfg_.has_cross_attention && cross_attn != nullptr
          && encoder_memory.defined ())
        {
          Tensor c_out = cross_attn->forward_cross (
            h, encoder_memory, cross_mask);
          h = norm_cross->forward (h + resid_dropout->forward (c_out));
        }

      h = norm2->forward (h + resid_dropout->forward (ffn->forward (h)));
    }

  return h;
}

/* -------------------------------------------------------------------------
   Obsidian_Transformer_Stack implementation
   ------------------------------------------------------------------------- */

Obsidian_Transformer_Stack::Obsidian_Transformer_Stack (
  const Transformer_Config &cfg,
  Tensor_Options options)
  : cfg_ (cfg)
{
  layers.reserve (cfg.num_layers);
  for (size_t i = 0; i < cfg.num_layers; ++i)
    {
      auto layer = std::make_shared<Obsidian_Transformer_Layer> (
        cfg, options);
      register_module ("layer_" + std::to_string (i), layer);
      layers.push_back (std::move (layer));
    }
  final_norm = register_module (
    "final_norm",
    make_norm_layer (cfg.norm_kind, cfg.d_model, cfg.norm_eps, options));
}

Tensor
Obsidian_Transformer_Stack::forward (const Tensor &hidden_states)
{
  return forward_ext (hidden_states);
}

Tensor
Obsidian_Transformer_Stack::forward_ext (
  const Tensor &hidden_states,
  const Tensor &attention_mask,
  const Tensor &encoder_memory,
  const Tensor &cross_mask,
  std::vector<basalt::nn::KV_Cache> *kv_caches)
{
  Tensor h = hidden_states;
  const size_t S = h.size (1);

  Tensor effective_mask = attention_mask;
  if (effective_mask.defined () && effective_mask.ndim () == 2)
    {
      auto pm = tensor_ops::build_padding_mask (effective_mask);
      if (pm.has_value ())
        {
          effective_mask = *pm;
        }
    }

  if (cfg_.pos_encoding == Position_Encoding_Kind::ALiBi
      && (kv_caches == nullptr || S > 1))
    {
      auto alibi_res = tensor_ops::build_alibi_bias (
        cfg_.num_heads, S, h.device ());
      if (alibi_res.has_value ())
        {
          effective_mask = effective_mask.defined ()
                             ? (effective_mask + *alibi_res)
                             : *alibi_res;
        }
    }

  for (size_t i = 0; i < layers.size (); ++i)
    {
      basalt::nn::KV_Cache *layer_cache = nullptr;
      if (kv_caches != nullptr && i < kv_caches->size ())
        {
          layer_cache = &(*kv_caches)[i];
        }
      h = layers[i]->forward_ext (
        h, effective_mask, encoder_memory, cross_mask, layer_cache);
    }

  return final_norm->forward (h);
}

} /* namespace obsidian */
