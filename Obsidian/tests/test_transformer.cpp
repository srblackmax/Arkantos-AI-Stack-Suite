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
#include "test_harness.hpp"

using namespace obsidian;

OBSIDIAN_TEST (grouped_query_attention_and_kv_cache)
{
  /* 4 query heads, 2 KV heads (GQA), d_model = 16 -> head_dim = 4.  */
  Grouped_Query_Attention gqa (16, 4, 2, 0.0, true, true, 64);

  Tensor x = basalt::randn (
    {2, 5, 16},
    Tensor_Options ().with_dtype (Dtype::Float32).with_requires_grad (true));
  Tensor out = gqa.forward_self (x, Tensor (), true);
  OBSIDIAN_CHECK_EQ (out.shape (), (Shape_Vector {2, 5, 16}));

  /* Verify backward pass flows cleanly through fused QKV and RoPE.  */
  Tensor loss = basalt::ops::sum (out);
  loss.backward ();
  OBSIDIAN_CHECK (x.grad ().defined ());
  OBSIDIAN_CHECK (gqa.qkv_proj->weight.grad ().defined ());

  /* Verify incremental KV_Cache decoding.  */
  No_Grad_Guard no_grad;
  gqa.eval ();
  basalt::nn::KV_Cache cache (1, 4, 16, 4);
  Tensor step1 = basalt::randn ({1, 1, 16});
  Tensor step2 = basalt::randn ({1, 1, 16});
  Tensor o1 = gqa.forward_self (step1, Tensor (), true, &cache);
  Tensor o2 = gqa.forward_self (step2, Tensor (), true, &cache);
  OBSIDIAN_CHECK_EQ (o1.shape (), (Shape_Vector {1, 1, 16}));
  OBSIDIAN_CHECK_EQ (o2.shape (), (Shape_Vector {1, 1, 16}));
  OBSIDIAN_CHECK_EQ (cache.seq_len (), 2u);
}

OBSIDIAN_TEST (swiglu_and_geglu_feed_forward_blocks)
{
  Feed_Forward_Block swiglu (16, 32, FFN_Activation_Kind::SwiGLU);
  Feed_Forward_Block geglu (16, 32, FFN_Activation_Kind::GeGLU);

  Tensor x = basalt::randn (
    {2, 4, 16},
    Tensor_Options ().with_dtype (Dtype::Float32).with_requires_grad (true));
  Tensor y1 = swiglu.forward (x);
  Tensor y2 = geglu.forward (x);
  OBSIDIAN_CHECK_EQ (y1.shape (), (Shape_Vector {2, 4, 16}));
  OBSIDIAN_CHECK_EQ (y2.shape (), (Shape_Vector {2, 4, 16}));

  Tensor loss = basalt::ops::sum (y1 + y2);
  loss.backward ();
  OBSIDIAN_CHECK (x.grad ().defined ());
}

OBSIDIAN_TEST (transformer_stack_rope_and_alibi)
{
  Transformer_Config cfg;
  cfg.d_model = 16;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.d_ff = 32;
  cfg.num_layers = 2;
  cfg.is_causal = true;
  cfg.pos_encoding = Position_Encoding_Kind::ALiBi;
  cfg.norm_kind = Norm_Kind::RMS_Norm;
  cfg.ffn_activation = FFN_Activation_Kind::SwiGLU;

  Obsidian_Transformer_Stack stack (cfg);
  Tensor h = basalt::randn (
    {2, 6, 16},
    Tensor_Options ().with_dtype (Dtype::Float32).with_requires_grad (true));
  Tensor out = stack.forward (h);
  OBSIDIAN_CHECK_EQ (out.shape (), (Shape_Vector {2, 6, 16}));

  Tensor loss = basalt::ops::mean (out);
  loss.backward ();
  OBSIDIAN_CHECK (h.grad ().defined ());
  OBSIDIAN_CHECK (stack.num_parameters () > 0u);
}

int
main (void)
{
  return obsidian::testing::run_all_tests ("Phase 6: Transformer Blocks");
}
