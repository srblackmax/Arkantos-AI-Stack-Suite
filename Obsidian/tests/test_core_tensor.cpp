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

#include <obsidian/obsidian.hpp>
#include "test_harness.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace obsidian;

OBSIDIAN_TEST (version_and_error_handling)
{
  OBSIDIAN_CHECK (std::strlen (version_string ()) > 0);
  OBSIDIAN_CHECK_EQ (version_major (), 0);
  OBSIDIAN_CHECK_EQ (version_minor (), 1);

  Result<int> ok_res = 42;
  OBSIDIAN_CHECK_OK (ok_res);
  OBSIDIAN_CHECK_EQ (*ok_res, 42);

  Result<int> err_res = OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                                        "bad parameter");
  OBSIDIAN_CHECK (!err_res.has_value ());
  OBSIDIAN_CHECK_EQ (err_res.error ().code (), Error_Code::Invalid_Argument);
  OBSIDIAN_CHECK (err_res.error ().format ().find ("Invalid_Argument")
                  != std::string::npos);
}

OBSIDIAN_TEST (pmr_arena_allocation)
{
  Arena arena;
  Pmr_Vector<int> vec (arena.resource ());
  for (int i = 0; i < 512; ++i)
    {
      vec.push_back (i * 2);
    }
  OBSIDIAN_CHECK_EQ (vec.size (), 512u);
  OBSIDIAN_CHECK_EQ (vec[100], 200);

  auto span = arena.allocate_span<float> (64);
  OBSIDIAN_CHECK_EQ (span.size (), 64u);
  span[0] = 3.25f;
  OBSIDIAN_CHECK_NEAR (span[0], 3.25f, 1e-6);
  arena.reset ();
}

OBSIDIAN_TEST (simd_string_scanning_and_utf8)
{
  std::string sample =
    "   Obsidian, the C++23 NLP library!  Fast-path\tscanning...\nEnd.";
  Arena &arena = thread_local_arena ();
  arena.reset ();

  Pmr_Vector<std::string_view> tokens (arena.resource ());
  simd::scan_pretoken_spans (sample, tokens, true);

  OBSIDIAN_CHECK (tokens.size () >= 12u);
  OBSIDIAN_CHECK_EQ (tokens[0], "Obsidian");
  OBSIDIAN_CHECK_EQ (tokens[1], ",");
  OBSIDIAN_CHECK_EQ (tokens[2], "the");

  auto cp_res = simd::validate_and_count_utf8 ("Hello \xCE\xB1\xCE\xB2!");
  OBSIDIAN_CHECK_OK (cp_res);
  OBSIDIAN_CHECK_EQ (*cp_res, 9u);

  const char bad_utf8[] = {'a', static_cast<char> (0xFF), 'b'};
  auto bad_res = simd::validate_and_count_utf8 (
    std::string_view (bad_utf8, 3));
  OBSIDIAN_CHECK (!bad_res.has_value ());
  OBSIDIAN_CHECK_EQ (bad_res.error ().code (), Error_Code::Invalid_UTF8);

  std::string upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789HelloWorld";
  std::string lower (upper.size (), '\0');
  simd::ascii_to_lower (upper, lower.data ());
  OBSIDIAN_CHECK_EQ (lower, "abcdefghijklmnopqrstuvwxyz0123456789helloworld");
}

OBSIDIAN_TEST (simd_vector_math)
{
  std::vector<float> a (37, 0.0f);
  std::vector<float> b (37, 0.0f);
  for (size_t i = 0; i < a.size (); ++i)
    {
      a[i] = static_cast<float> (i + 1) * 0.1f;
      b[i] = static_cast<float> (i + 1) * 0.2f;
    }

  float dot = simd::dot_f32 (a, b);
  float expected_dot = 0.0f;
  for (size_t i = 0; i < a.size (); ++i)
    {
      expected_dot += a[i] * b[i];
    }
  OBSIDIAN_CHECK_NEAR (dot, expected_dot, 1e-4f);

  float cos_sim = simd::cosine_similarity_f32 (a, b);
  OBSIDIAN_CHECK_NEAR (cos_sim, 1.0f, 1e-5f);

  simd::axpy_f32 (2.0f, a, b);
  OBSIDIAN_CHECK_NEAR (b[0], 0.4f, 1e-5f);

  simd::normalize_inplace_f32 (a);
  OBSIDIAN_CHECK_NEAR (simd::norm_f32 (a), 1.0f, 1e-5f);
}

OBSIDIAN_TEST (basalt_tensor_block_and_sequence_batch)
{
  std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto blk_res = Tensor_Block::from_span_f32 (vals, {2, 3}, true);
  OBSIDIAN_CHECK_OK (blk_res);

  Tensor_Block &blk = *blk_res;
  OBSIDIAN_CHECK_OK (blk.expect_rank (2));
  OBSIDIAN_CHECK_OK (blk.expect_shape ({2, 3}));
  OBSIDIAN_CHECK_OK (blk.expect_dtype (Dtype::Float32));

  auto sp_res = blk.span_f32 ();
  OBSIDIAN_CHECK_OK (sp_res);
  OBSIDIAN_CHECK_NEAR ((*sp_res)[4], 5.0f, 1e-6f);

  /* Verify Basalt autograd flows through Tensor_Block.  */
  Tensor doubled = blk.raw () * 3.0;
  Tensor loss = basalt::ops::sum (doubled);
  loss.backward ();
  OBSIDIAN_CHECK (blk.raw ().grad ().defined ());
  OBSIDIAN_CHECK_NEAR (blk.raw ().grad ().read_scalar_at (0), 3.0, 1e-5);

  /* Sequence_Batch collation and padding.  */
  std::vector<std::vector<int64_t>> seqs = {
    {10, 20, 30},
    {40, 50}
  };
  auto batch_res = Sequence_Batch::from_sequences (seqs, 0);
  OBSIDIAN_CHECK_OK (batch_res);
  OBSIDIAN_CHECK_EQ (batch_res->batch_size (), 2u);
  OBSIDIAN_CHECK_EQ (batch_res->seq_len (), 3u);
  OBSIDIAN_CHECK_EQ (batch_res->input_ids.read_scalar_at (5), 0.0);
  OBSIDIAN_CHECK_NEAR (batch_res->attention_mask.read_scalar_at (4),
                       1.0, 1e-6);
  OBSIDIAN_CHECK_NEAR (batch_res->attention_mask.read_scalar_at (5),
                       0.0, 1e-6);
}

OBSIDIAN_TEST (nlp_tensor_ops_and_sequence_loss_autograd)
{
  auto causal_res = tensor_ops::build_causal_mask (4);
  OBSIDIAN_CHECK_OK (causal_res);
  OBSIDIAN_CHECK_NEAR (causal_res->read_scalar_at (0), 0.0, 1e-5);
  OBSIDIAN_CHECK (causal_res->read_scalar_at (1) < -1e8);

  auto alibi_res = tensor_ops::build_alibi_bias (4, 4);
  OBSIDIAN_CHECK_OK (alibi_res);
  OBSIDIAN_CHECK_EQ (alibi_res->shape (), (Shape_Vector {1, 4, 4, 4}));

  auto sin_res = tensor_ops::build_sinusoidal_positions (8, 16);
  OBSIDIAN_CHECK_OK (sin_res);
  OBSIDIAN_CHECK_EQ (sin_res->shape (), (Shape_Vector {8, 16}));

  /* Test masked_mean_pooling and last_token_pooling.  */
  std::vector<std::vector<int64_t>> seqs = {{1, 2, 3}, {4, 5}};
  auto batch = Sequence_Batch::from_sequences (seqs, 0).value ();
  Tensor hidden = basalt::ones (
    {2, 3, 4},
    Tensor_Options ().with_dtype (Dtype::Float32).with_requires_grad (true));

  auto pooled_res = tensor_ops::masked_mean_pooling (
    hidden, batch.attention_mask);
  OBSIDIAN_CHECK_OK (pooled_res);
  OBSIDIAN_CHECK_EQ (pooled_res->shape (), (Shape_Vector {2, 4}));
  OBSIDIAN_CHECK_NEAR (pooled_res->read_scalar_at (0), 1.0, 1e-5);

  auto last_res = tensor_ops::last_token_pooling (
    hidden, batch.attention_mask);
  OBSIDIAN_CHECK_OK (last_res);
  OBSIDIAN_CHECK_EQ (last_res->shape (), (Shape_Vector {2, 4}));

  /* Test sequence_cross_entropy_loss with ignore_index and autograd.  */
  Tensor logits = basalt::zeros (
    {1, 3, 5},
    Tensor_Options ().with_dtype (Dtype::Float32).with_requires_grad (true));
  std::vector<int64_t> targets = {1, 3, -100};
  auto ce_res = tensor_ops::sequence_cross_entropy_loss (
    logits, targets, -100, 0.0);
  OBSIDIAN_CHECK_OK (ce_res);
  /* Uniform logits over 5 classes -> loss = log(5) ~= 1.6094379.  */
  OBSIDIAN_CHECK_NEAR (ce_res->item_double (), std::log (5.0), 1e-4);

  ce_res->backward ();
  OBSIDIAN_CHECK (logits.grad ().defined ());
  /* Ignored token at index 2 must receive zero gradient.  */
  for (size_t v = 0; v < 5; ++v)
    {
      OBSIDIAN_CHECK_NEAR (logits.grad ().read_scalar_at (2 * 5 + v),
                           0.0, 1e-6);
    }
}

int
main (void)
{
  return obsidian::testing::run_all_tests ("Phase 1: Core & Basalt Bridge");
}
