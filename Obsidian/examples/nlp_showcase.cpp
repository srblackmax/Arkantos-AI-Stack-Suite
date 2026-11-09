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

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace obsidian;

int
main (void)
{
  std::cout << "=== Obsidian C++23 Deep Learning & NLP Showcase ===\n";
  std::cout << "Version: " << version_string ()
            << " | SIMD ISA: " << simd::active_isa_name () << "\n\n";

  /* 1. Train a high-throughput BPE tokenizer on a sample corpus.  */
  std::vector<std::string_view> corpus = {
    "transformers process token sequences with multi-head attention",
    "obsidian integrates basalt tensors with simd tokenization",
    "memory mapped word embeddings provide zero copy vector lookups"
  };

  auto bpe = std::make_shared<BPE_Tokenizer> ();
  auto tr_ok = bpe->train_from_corpus (corpus, 64, 1);
  if (!tr_ok.has_value ())
    {
      std::cerr << tr_ok.error ().format () << "\n";
      return 1;
    }

  Arena arena;
  auto ids = bpe->encode_into_arena (
    "obsidian integrates transformers", arena, true, true);
  std::cout << "[1] BPE Encoded (" << ids.size () << " tokens) -> Decoded: \""
            << bpe->decode (ids, true).value () << "\"\n";

  /* 2. Train Word2Vec Skip-Gram embeddings and run similarity query.  */
  Word2Vec_Config wcfg;
  wcfg.dim = 16;
  wcfg.epochs = 10;
  auto emb = Word2Vec_Trainer::train (corpus, wcfg).value ();
  auto sim = emb.similarity ("transformers", "attention").value ();
  std::cout << "[2] Word2Vec embeddings (" << emb.vocab_size ()
            << " words, dim=" << emb.dim ()
            << ") cos(transformers, attention) = " << sim << "\n";

  /* 3. Classical NLP Pipeline (Segmentation, Stemming, POS, Lemmas, NER).  */
  Linguistic_Pipeline ling;
  ling.add_entity_gazetteer ({"Obsidian"}, "LIBRARY");
  ling.add_entity_gazetteer ({"Basalt"}, "BACKEND");
  auto doc = ling.annotate (
    "Dr. Smith built Obsidian on Basalt. The models are running.").value ();
  std::cout << "[3] Classical Annotation: " << doc.sentences.size ()
            << " sentences, " << doc.tokens.size () << " tokens, "
            << doc.entities.size () << " named entities.\n";

  /* 4. Basalt-backed Causal Transformer LM training step & generation.  */
  Transformer_Config tcfg;
  tcfg.d_model = 16;
  tcfg.num_heads = 4;
  tcfg.num_kv_heads = 2;
  tcfg.d_ff = 32;
  tcfg.num_layers = 2;
  tcfg.pos_encoding = Position_Encoding_Kind::Rotary_RoPE;

  auto lm = std::make_shared<Causal_LM> (bpe->vocab_size (), tcfg);
  basalt::optim::AdamW optimizer (lm->parameters (), 1e-2);

  auto batch = bpe->encode_batch (corpus, true, true).value ();
  Tensor logits = lm->forward_ids (batch.input_ids);
  std::vector<int64_t> targets = batch.input_ids.to_vector_int64 ();
  Tensor loss = tensor_ops::sequence_cross_entropy_loss (
    logits, targets, 0).value ();
  optimizer.zero_grad ();
  loss.backward ();
  optimizer.step ();

  Text_Generation_Pipeline gen_pipe (bpe, lm);
  Generation_Config gcfg;
  gcfg.max_new_tokens = 4;
  std::string generated = gen_pipe.generate_text ("obsidian", gcfg).value ();
  std::cout << "[4] Transformer Causal LM Loss = " << loss.item_double ()
            << " | Generated: \"" << generated << "\"\n";

  return 0;
}
