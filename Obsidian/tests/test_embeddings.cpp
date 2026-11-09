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

#include <obsidian/embeddings.hpp>
#include <obsidian/simd.hpp>
#include "test_harness.hpp"

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using namespace obsidian;

OBSIDIAN_TEST (mmap_embeddings_zero_copy_and_analogy)
{
  std::vector<std::string> words = {
    "king", "queen", "man", "woman", "apple"
  };
  /* 4-D embeddings where king - man + woman = queen.  */
  std::vector<float> matrix = {
    1.0f,  1.0f, 0.0f, 0.0f, /* king  */
    1.0f, -1.0f, 0.0f, 0.0f, /* queen */
    0.0f,  1.0f, 0.0f, 0.0f, /* man   */
    0.0f, -1.0f, 0.0f, 0.0f, /* woman */
    0.0f,  0.0f, 1.0f, 1.0f  /* apple */
  };

  auto mem_res = Mmap_Embeddings::from_memory (words, matrix, 4);
  OBSIDIAN_CHECK_OK (mem_res);

  const std::string obvec_path = "/tmp/obsidian_test_emb.obvec";
  OBSIDIAN_CHECK_OK (mem_res->save_obvec_mmap (obvec_path));

  auto mmap_res = Mmap_Embeddings::open_mmap (obvec_path);
  std::remove (obvec_path.c_str ());
  OBSIDIAN_CHECK_OK (mmap_res);

  const Mmap_Embeddings &store = *mmap_res;
  OBSIDIAN_CHECK (store.is_memory_mapped ());
  OBSIDIAN_CHECK_EQ (store.vocab_size (), 5u);
  OBSIDIAN_CHECK_EQ (store.dim (), 4u);

  auto king_sp = store.lookup ("king");
  OBSIDIAN_CHECK_OK (king_sp);
  OBSIDIAN_CHECK_NEAR ((*king_sp)[0], 1.0f, 1e-6f);
  OBSIDIAN_CHECK_NEAR ((*king_sp)[1], 1.0f, 1e-6f);

  auto oov_sp = store.lookup ("missing_word");
  OBSIDIAN_CHECK (!oov_sp.has_value ());
  OBSIDIAN_CHECK_EQ (oov_sp.error ().code (), Error_Code::Vocab_Not_Found);

  /* Solve analogy: man is to king as woman is to ? -> queen.  */
  auto ana_res = store.analogy ("man", "king", "woman", 1);
  OBSIDIAN_CHECK_OK (ana_res);
  OBSIDIAN_CHECK_EQ (ana_res->size (), 1u);
  OBSIDIAN_CHECK_EQ ((*ana_res)[0].word, "queen");

  /* Convert to Basalt Tensor and embed sequence.  */
  auto tensor_res = store.to_basalt_tensor ();
  OBSIDIAN_CHECK_OK (tensor_res);
  OBSIDIAN_CHECK_EQ (tensor_res->shape (), (Shape_Vector {5, 4}));

  std::vector<std::string_view> seq = {"king", "apple"};
  auto seq_tensor = store.embed_sequence (seq);
  OBSIDIAN_CHECK_OK (seq_tensor);
  OBSIDIAN_CHECK_EQ (seq_tensor->shape (), (Shape_Vector {2, 4}));
}

OBSIDIAN_TEST (glove_and_word2vec_format_io)
{
  std::vector<std::string> words = {"cat", "dog", "bird"};
  std::vector<float> matrix = {
    0.8f, 0.6f, 0.0f,
    0.6f, 0.8f, 0.0f,
    0.0f, 0.0f, 1.0f
  };

  auto store = Mmap_Embeddings::from_memory (words, matrix, 3).value ();

  const std::string glove_path = "/tmp/obsidian_test_glove.txt";
  OBSIDIAN_CHECK_OK (store.save_glove_text (glove_path));
  auto glove_loaded = Mmap_Embeddings::load_glove_text (glove_path);
  std::remove (glove_path.c_str ());
  OBSIDIAN_CHECK_OK (glove_loaded);
  OBSIDIAN_CHECK_EQ (glove_loaded->vocab_size (), 3u);
  auto sim_res = glove_loaded->similarity ("cat", "dog");
  OBSIDIAN_CHECK_OK (sim_res);
  OBSIDIAN_CHECK_NEAR (*sim_res, 0.96f, 1e-4f);

  const std::string w2v_path = "/tmp/obsidian_test_w2v.bin";
  OBSIDIAN_CHECK_OK (store.save_word2vec_binary (w2v_path));
  auto w2v_loaded = Mmap_Embeddings::load_word2vec_binary (w2v_path);
  std::remove (w2v_path.c_str ());
  OBSIDIAN_CHECK_OK (w2v_loaded);
  OBSIDIAN_CHECK_EQ (w2v_loaded->vocab_size (), 3u);
  auto nn_res = w2v_loaded->most_similar ("cat", 2);
  OBSIDIAN_CHECK_OK (nn_res);
  OBSIDIAN_CHECK_EQ ((*nn_res)[0].word, "dog");
}

OBSIDIAN_TEST (word2vec_skipgram_cbow_and_fasttext_subwords)
{
  std::vector<std::string_view> corpus = {
    "neural networks learn deep representations from text",
    "transformers learn contextual representations from text",
    "gradient descent optimizes neural networks and transformers"
  };

  Word2Vec_Config cfg;
  cfg.dim = 16;
  cfg.epochs = 8;
  cfg.skip_gram = true;
  auto sg_res = Word2Vec_Trainer::train (corpus, cfg);
  OBSIDIAN_CHECK_OK (sg_res);
  OBSIDIAN_CHECK (sg_res->vocab_size () >= 8u);

  cfg.skip_gram = false;
  auto cbow_res = Word2Vec_Trainer::train (corpus, cfg);
  OBSIDIAN_CHECK_OK (cbow_res);
  OBSIDIAN_CHECK_EQ (cbow_res->dim (), 16u);

  /* FastText subword embeddings for OOV morphological similarity.  */
  FastText_Embeddings ft (16, 1024, 3, 5);
  OBSIDIAN_CHECK_OK (ft.train (corpus, 5, 0.05f));

  std::vector<float> v_trans (16, 0.0f);
  std::vector<float> v_trans_oov (16, 0.0f);
  ft.compute_word_vector ("transformers", v_trans);
  ft.compute_word_vector ("transformer_models", v_trans_oov);

  float oov_sim = simd::cosine_similarity_f32 (v_trans, v_trans_oov);
  /* Overlapping 3..5-grams ("tra", "rans", "form", etc.) yield positive
     cosine similarity even for an unseen compound word.  */
  OBSIDIAN_CHECK (oov_sim > 0.35f);
}

int
main (void)
{
  return obsidian::testing::run_all_tests ("Phase 3: Mmap Word Embeddings");
}
