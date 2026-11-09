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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace obsidian;

OBSIDIAN_TEST (encoder_sequence_and_token_classifiers)
{
  Transformer_Config cfg;
  cfg.d_model = 16;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.d_ff = 32;
  cfg.num_layers = 2;
  cfg.pos_encoding = Position_Encoding_Kind::Learned_Absolute;

  Sequence_Classifier seq_cls (64, 3, cfg, true);
  Token_Classifier tok_cls (64, 5, cfg);

  std::vector<std::vector<int64_t>> seqs = {
    {2, 10, 15, 20, 3},
    {2, 12, 18, 3}
  };
  auto batch = Sequence_Batch::from_sequences (seqs, 0).value ();

  auto seq_logits = seq_cls.forward_batch (batch);
  OBSIDIAN_CHECK_OK (seq_logits);
  OBSIDIAN_CHECK_EQ (seq_logits->shape (), (Shape_Vector {2, 3}));

  Tensor tok_logits = tok_cls.forward_batch (batch);
  OBSIDIAN_CHECK_EQ (tok_logits.shape (), (Shape_Vector {2, 5, 5}));
  basalt::ops::sum (tok_logits).backward ();
  OBSIDIAN_CHECK (tok_cls.classifier->weight.grad ().defined ());

  /* Verify backward pass and optimizer step on Sequence_Classifier.  */
  std::vector<int64_t> labels = {1, 2};
  auto loss_res = tensor_ops::sequence_cross_entropy_loss (
    *seq_logits, labels, -100, 0.0);
  OBSIDIAN_CHECK_OK (loss_res);
  loss_res->backward ();
  OBSIDIAN_CHECK (seq_cls.classifier->weight.grad ().defined ());
}

OBSIDIAN_TEST (causal_lm_generation_and_seq2seq)
{
  Transformer_Config cfg;
  cfg.d_model = 16;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.d_ff = 32;
  cfg.num_layers = 2;
  cfg.pos_encoding = Position_Encoding_Kind::Rotary_RoPE;

  Causal_LM lm (32, cfg);
  std::vector<int64_t> prompt = {2, 5, 8};

  /* 1. Greedy decoding with KV cache.  */
  Generation_Config greedy_cfg;
  greedy_cfg.max_new_tokens = 4;
  greedy_cfg.do_sample = false;
  auto greedy_out = lm.generate (prompt, greedy_cfg);
  OBSIDIAN_CHECK_OK (greedy_out);
  OBSIDIAN_CHECK_EQ (greedy_out->size (), 7u);

  /* 2. Top-K + Top-P (Nucleus) + Repetition Penalty sampling.  */
  Generation_Config sample_cfg;
  sample_cfg.max_new_tokens = 4;
  sample_cfg.do_sample = true;
  sample_cfg.temperature = 0.8f;
  sample_cfg.top_k = 8;
  sample_cfg.top_p = 0.9f;
  sample_cfg.repetition_penalty = 1.2f;
  auto sample_out = lm.generate (prompt, sample_cfg);
  OBSIDIAN_CHECK_OK (sample_out);
  OBSIDIAN_CHECK_EQ (sample_out->size (), 7u);

  /* 3. Beam search decoding.  */
  Generation_Config beam_cfg;
  beam_cfg.max_new_tokens = 3;
  beam_cfg.num_beams = 2;
  auto beam_out = lm.generate (prompt, beam_cfg);
  OBSIDIAN_CHECK_OK (beam_out);
  OBSIDIAN_CHECK_EQ (beam_out->size (), 6u);

  /* 4. Seq2Seq Encoder-Decoder model.  */
  Seq2Seq_Model s2s (32, cfg);
  int64_t src_arr[4] = {2, 6, 7, 3};
  int64_t tgt_arr[3] = {2, 9, 10};
  Tensor src_t = Tensor_Block::from_token_ids (src_arr, 1, 4)->raw ();
  Tensor tgt_t = Tensor_Block::from_token_ids (tgt_arr, 1, 3)->raw ();
  Tensor s2s_logits = s2s.forward_seq2seq (src_t, tgt_t);
  OBSIDIAN_CHECK_EQ (s2s_logits.shape (), (Shape_Vector {1, 3, 32}));
  basalt::ops::sum (s2s_logits).backward ();
  OBSIDIAN_CHECK (s2s.lm_head->weight.grad ().defined ());
}

OBSIDIAN_TEST (cpp23_pipe_composition_and_end_to_end_pipelines)
{
  /* Test C++23 pipe syntax (val | stage1 | stage2).  */
  Porter_Stemmer stemmer;
  auto tokenize_stage = make_stage ([] (std::string_view s)
    {
      Arena &a = thread_local_arena ();
      a.reset ();
      Pmr_Vector<std::string_view> spans (a.resource ());
      simd::scan_pretoken_spans (s, spans, true);
      return std::vector<std::string_view> (spans.begin (), spans.end ());
    });

  auto stem_stage = make_stage ([&] (const std::vector<std::string_view> &toks)
    {
      std::vector<std::string> out;
      out.reserve (toks.size ());
      for (auto t : toks)
        {
          out.push_back (stemmer.stem (t));
        }
      return out;
    });

  auto stemmed = std::string_view ("running cats walked")
                 | tokenize_stage
                 | stem_stage;
  OBSIDIAN_CHECK_EQ (stemmed.size (), 3u);
  OBSIDIAN_CHECK_EQ (stemmed[0], "run");
  OBSIDIAN_CHECK_EQ (stemmed[1], "cat");
  OBSIDIAN_CHECK_EQ (stemmed[2], "walk");

  /* Test Linguistic_Pipeline annotation.  */
  Linguistic_Pipeline ling;
  ling.add_entity_gazetteer ({"Obsidian"}, "SOFTWARE");
  auto doc_res = ling.annotate (
    "Dr. Turing tested Obsidian. The children were running.");
  OBSIDIAN_CHECK_OK (doc_res);
  OBSIDIAN_CHECK_EQ (doc_res->sentences.size (), 2u);
  OBSIDIAN_CHECK (!doc_res->entities.empty ());
  bool found_software = false;
  for (const auto &ent : doc_res->entities)
    {
      if (ent.label == "SOFTWARE" && ent.text == "Obsidian")
        {
          found_software = true;
        }
    }
  OBSIDIAN_CHECK (found_software);

  /* Test Text_Classification_Pipeline and Text_Generation_Pipeline.  */
  std::vector<std::string_view> corpus = {
    "deep learning transformers",
    "natural language processing"
  };
  auto bpe = std::make_shared<BPE_Tokenizer> ();
  OBSIDIAN_CHECK_OK (bpe->train_from_corpus (corpus, 40, 1));

  Transformer_Config cfg;
  cfg.d_model = 16;
  cfg.num_heads = 4;
  cfg.d_ff = 32;
  cfg.num_layers = 1;

  auto cls_model = std::make_shared<Sequence_Classifier> (
    bpe->vocab_size (), 2, cfg);
  Text_Classification_Pipeline cls_pipe (
    bpe, cls_model, {"NEGATIVE", "POSITIVE"});
  auto pred_res = cls_pipe.predict ("deep learning");
  OBSIDIAN_CHECK_OK (pred_res);
  OBSIDIAN_CHECK (pred_res->probability >= 0.0f
                  && pred_res->probability <= 1.0f);

  auto gen_model = std::make_shared<Causal_LM> (bpe->vocab_size (), cfg);
  Text_Generation_Pipeline gen_pipe (bpe, gen_model);
  Generation_Config gcfg;
  gcfg.max_new_tokens = 3;
  auto gen_text = gen_pipe.generate_text ("deep", gcfg);
  OBSIDIAN_CHECK_OK (gen_text);
  OBSIDIAN_CHECK (!gen_text->empty ());
}

int
main (void)
{
  return obsidian::testing::run_all_tests (
    "Phase 7: End-to-End Models & C++23 Pipelines");
}
