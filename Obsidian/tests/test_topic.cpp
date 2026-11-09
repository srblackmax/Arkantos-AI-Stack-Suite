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

#include <obsidian/topic.hpp>
#include <obsidian/simd.hpp>
#include "test_harness.hpp"

#include <string_view>
#include <vector>

using namespace obsidian;

OBSIDIAN_TEST (lda_collapsed_gibbs_topic_modeling)
{
  std::vector<std::string_view> corpus = {
    "tensor autograd gradient neural layer transformer attention",
    "transformer attention head embedding token neural network",
    "autograd backward gradient tensor matrix linear layer",
    "planet orbit star galaxy telescope astronomy cosmos",
    "galaxy star nebula astronomy telescope orbit solar",
    "cosmos planet telescope star astronomy orbit universe"
  };

  LDA_Model lda (2, 0.1f, 0.05f, 80, 42);
  OBSIDIAN_CHECK_OK (lda.fit (corpus));
  OBSIDIAN_CHECK_EQ (lda.num_topics (), 2u);
  OBSIDIAN_CHECK_EQ (lda.num_documents (), 6u);

  auto d0_dist = lda.document_topic_distribution (0).value ();
  auto d3_dist = lda.document_topic_distribution (3).value ();
  OBSIDIAN_CHECK_NEAR (d0_dist[0] + d0_dist[1], 1.0f, 1e-5f);
  OBSIDIAN_CHECK_NEAR (d3_dist[0] + d3_dist[1], 1.0f, 1e-5f);

  /* Doc 0 (deep learning) and Doc 3 (astronomy) should peak on opposite
     topics.  */
  size_t d0_top = (d0_dist[0] > d0_dist[1]) ? 0 : 1;
  size_t d3_top = (d3_dist[0] > d3_dist[1]) ? 0 : 1;
  OBSIDIAN_CHECK (d0_top != d3_top);

  auto top_w = lda.topic_words (0, 5);
  OBSIDIAN_CHECK_OK (top_w);
  OBSIDIAN_CHECK_EQ (top_w->size (), 5u);

  auto inf_res = lda.infer_document ("star galaxy telescope orbit", 30);
  OBSIDIAN_CHECK_OK (inf_res);
  size_t inf_top = ((*inf_res)[0] > (*inf_res)[1]) ? 0 : 1;
  OBSIDIAN_CHECK_EQ (inf_top, d3_top);

  auto tw_tensor = lda.topic_word_tensor ();
  OBSIDIAN_CHECK_OK (tw_tensor);
  OBSIDIAN_CHECK_EQ (tw_tensor->size (0), 2u);
  OBSIDIAN_CHECK_EQ (tw_tensor->size (1), lda.vocab_size ());
}

OBSIDIAN_TEST (lsa_truncated_svd_via_basalt_tensors)
{
  std::vector<std::string_view> corpus = {
    "compiler optimizer register instruction assembly cpu",
    "cpu cache register pipeline instruction compiler",
    "syntax grammar parser tree token semantics",
    "parser token grammar syntax ast lexical"
  };

  LSA_Model lsa (2, 35);
  auto coords_res = lsa.fit_transform (corpus);
  OBSIDIAN_CHECK_OK (coords_res);
  OBSIDIAN_CHECK_EQ (coords_res->shape (), (Shape_Vector {4, 2}));
  OBSIDIAN_CHECK_EQ (lsa.singular_values ().shape (), (Shape_Vector {2}));
  OBSIDIAN_CHECK (lsa.singular_values ().read_scalar_at (0) > 0.5);

  /* Project a new document into the 2-D latent semantic space.  */
  std::vector<std::string_view> query_docs = {
    "cpu instruction register optimizer"
  };
  auto proj_res = lsa.transform (query_docs);
  OBSIDIAN_CHECK_OK (proj_res);
  OBSIDIAN_CHECK_EQ (proj_res->shape (), (Shape_Vector {1, 2}));
}

int
main (void)
{
  return obsidian::testing::run_all_tests (
    "Phase 5: Topic Modeling (LDA/LSA)");
}
