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

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using namespace obsidian;

OBSIDIAN_TEST (unicode_nfkc_normalization_and_byte_level_codec)
{
  /* Fullwidth ASCII, ligatures, curly quotes, em-dash, and ellipsis.  */
  std::string raw_utf8 = "“Ｏｂｓｉｄｉａｎ—"
                         "ﬁne\xE2\x80\x8B ﬂow…”";
  auto norm_res = unicode::normalize_nfkc_text (raw_utf8, false);
  OBSIDIAN_CHECK_OK (norm_res);
  OBSIDIAN_CHECK_EQ (*norm_res, "\"Obsidian-fine flow...\"");

  /* Accent stripping on Latin-1 Supplement characters.  */
  auto stripped = unicode::normalize_nfkc_text ("café résumé naïve", true);
  OBSIDIAN_CHECK_OK (stripped);
  OBSIDIAN_CHECK_EQ (*stripped, "cafe resume naive");

  /* Bijective 256-byte GPT-2 / LLaMA byte-to-unicode round-trip.  */
  std::string all_bytes (256, '\0');
  for (size_t i = 0; i < 256; ++i)
    {
      all_bytes[i] = static_cast<char> (static_cast<uint8_t> (i));
    }
  std::string encoded = unicode::byte_to_unicode_encode (all_bytes);
  OBSIDIAN_CHECK (!encoded.empty ());
  auto decoded = unicode::byte_to_unicode_decode (encoded);
  OBSIDIAN_CHECK_OK (decoded);
  OBSIDIAN_CHECK_EQ (*decoded, all_bytes);
}

OBSIDIAN_TEST (byte_level_bpe_hf_json_and_parallel_batch_encoding)
{
  std::vector<std::string_view> corpus = {
    "obsidian transformers accelerate deep learning",
    "obsidian tokenizers support byte level unicode"
  };

  BPE_Tokenizer bpe;
  bpe.set_byte_level (true);
  OBSIDIAN_CHECK (bpe.is_byte_level ());
  OBSIDIAN_CHECK_OK (bpe.train_from_corpus (corpus, 540, 1));

  /* Unseen emojis and multilingual UTF-8 must never emit <unk>.  */
  std::string multilingual = "obsidian 🚀 café 深層学習";
  auto ids_res = bpe.encode (multilingual, true, true);
  OBSIDIAN_CHECK_OK (ids_res);
  for (int64_t id : *ids_res)
    {
      OBSIDIAN_CHECK (id != bpe.special_tokens ().unk_id);
    }

  auto dec_res = bpe.decode (*ids_res, true);
  OBSIDIAN_CHECK_OK (dec_res);
  OBSIDIAN_CHECK_EQ (*dec_res, multilingual);

  /* Hugging Face tokenizer.json export and reload.  */
  const std::string json_path = "/tmp/obsidian_test_hf_tokenizer.json";
  OBSIDIAN_CHECK_OK (bpe.save_hf_tokenizer_json (json_path));

  auto loaded_bpe = BPE_Tokenizer::load_hf_tokenizer_json (json_path);
  OBSIDIAN_CHECK_OK (loaded_bpe);
  OBSIDIAN_CHECK (loaded_bpe->is_byte_level ());
  OBSIDIAN_CHECK_EQ (loaded_bpe->vocab_size (), bpe.vocab_size ());

  auto reloaded_ids = loaded_bpe->encode (multilingual, true, true);
  OBSIDIAN_CHECK_OK (reloaded_ids);
  OBSIDIAN_CHECK_EQ (*reloaded_ids, *ids_res);
  std::remove (json_path.c_str ());

  /* OpenMP multi-threaded parallel batch encoding vs sequential batch.  */
  std::vector<std::string_view> batch_texts;
  batch_texts.reserve (32);
  for (size_t i = 0; i < 16; ++i)
    {
      batch_texts.push_back ("obsidian transformers accelerate deep learning");
      batch_texts.push_back ("byte level unicode 🚀 café");
    }

  auto seq_batch = bpe.encode_batch (batch_texts, true, true);
  auto par_batch = bpe.encode_batch_parallel (batch_texts, true, true, 0,
                                              false, 4);
  OBSIDIAN_CHECK_OK (seq_batch);
  OBSIDIAN_CHECK_OK (par_batch);
  OBSIDIAN_CHECK_EQ (seq_batch->input_ids.shape (),
                     par_batch->input_ids.shape ());

  size_t total = seq_batch->input_ids.numel ();
  for (size_t i = 0; i < total; ++i)
    {
      OBSIDIAN_CHECK_EQ (seq_batch->input_ids.read_scalar_at (i),
                         par_batch->input_ids.read_scalar_at (i));
    }
}

OBSIDIAN_TEST (safetensors_mmap_io_and_module_checkpointing)
{
  const std::string st_path = "/tmp/obsidian_test_weights.safetensors";

  basalt::nn::State_Dict tensors;
  tensors["embed.weight"] = basalt::randn (
    {8, 4}, Tensor_Options ().with_dtype (Dtype::Float32));
  tensors["token_counts"] = basalt::ones (
    {8}, Tensor_Options ().with_dtype (Dtype::Int64));

  std::map<std::string, std::string> meta = {
    {"format", "pt"},
    {"framework", "obsidian-cpp23"}
  };

  OBSIDIAN_CHECK_OK (obsidian::save_safetensors (st_path, tensors, meta));

  auto st_file = SafeTensors_File::open (st_path);
  OBSIDIAN_CHECK_OK (st_file);
  OBSIDIAN_CHECK_EQ (st_file->num_tensors (), 2u);
  OBSIDIAN_CHECK (st_file->contains ("embed.weight"));
  OBSIDIAN_CHECK (st_file->contains ("token_counts"));
  OBSIDIAN_CHECK_EQ (st_file->metadata ().at ("framework"), "obsidian-cpp23");

  auto raw_bytes = st_file->get_raw_bytes ("embed.weight");
  OBSIDIAN_CHECK_OK (raw_bytes);
  OBSIDIAN_CHECK_EQ (raw_bytes->size (), 8u * 4u * sizeof (float));

  auto loaded_emb = st_file->load_tensor ("embed.weight");
  OBSIDIAN_CHECK_OK (loaded_emb);
  OBSIDIAN_CHECK_EQ (loaded_emb->shape (), (Shape_Vector {8, 4}));
  for (size_t i = 0; i < 32; ++i)
    {
      OBSIDIAN_CHECK_NEAR (
        loaded_emb->read_scalar_at (i),
        tensors["embed.weight"].read_scalar_at (i), 1e-6);
    }
  std::remove (st_path.c_str ());

  /* Full Causal_LM checkpoint save, weight perturbation, and reload.  */
  const std::string model_path = "/tmp/obsidian_test_lm.safetensors";
  Transformer_Config cfg;
  cfg.d_model = 16;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.d_ff = 32;
  cfg.num_layers = 2;

  Causal_LM lm1 (32, cfg);
  int64_t ids_arr[3] = {2, 5, 9};
  Tensor input_ids = Tensor_Block::from_token_ids (ids_arr, 1, 3)->raw ();
  Tensor orig_logits = lm1.infer (input_ids);

  OBSIDIAN_CHECK_OK (save_module_safetensors (lm1, model_path));

  Causal_LM lm2 (32, cfg);
  OBSIDIAN_CHECK_OK (load_module_safetensors (lm2, model_path, true));
  Tensor reloaded_logits = lm2.infer (input_ids);

  OBSIDIAN_CHECK_EQ (orig_logits.shape (), reloaded_logits.shape ());
  for (size_t i = 0; i < orig_logits.numel (); ++i)
    {
      OBSIDIAN_CHECK_NEAR (orig_logits.read_scalar_at (i),
                           reloaded_logits.read_scalar_at (i), 1e-5);
    }
  std::remove (model_path.c_str ());
}

OBSIDIAN_TEST (no_grad_infer_wrappers_and_batched_generation)
{
  Transformer_Config cfg;
  cfg.d_model = 16;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.d_ff = 32;
  cfg.num_layers = 1;

  std::vector<std::vector<int64_t>> seqs = {
    {2, 4, 6, 3},
    {2, 7, 3}
  };
  auto batch = Sequence_Batch::from_sequences (seqs, 0).value ();

  /* Call infer_batch without No_Grad_Guard or backward(): must not leak!  */
  Encoder_Model enc (32, cfg);
  Tensor enc_out = enc.infer_batch (batch);
  OBSIDIAN_CHECK_EQ (enc_out.shape (), (Shape_Vector {2, 4, 16}));
  OBSIDIAN_CHECK (enc.is_training ());

  Sequence_Classifier seq_cls (32, 3, cfg);
  auto cls_out = seq_cls.infer_batch (batch);
  OBSIDIAN_CHECK_OK (cls_out);
  OBSIDIAN_CHECK_EQ (cls_out->shape (), (Shape_Vector {2, 3}));

  Token_Classifier tok_cls (32, 5, cfg);
  Tensor tok_out = tok_cls.infer_batch (batch);
  OBSIDIAN_CHECK_EQ (tok_out.shape (), (Shape_Vector {2, 4, 5}));

  Seq2Seq_Model s2s (32, cfg);
  Tensor s2s_out = s2s.infer_seq2seq (batch.input_ids, batch.input_ids);
  OBSIDIAN_CHECK_EQ (s2s_out.shape (), (Shape_Vector {2, 4, 32}));

  Causal_LM lm (32, cfg);
  Generation_Config gcfg;
  gcfg.max_new_tokens = 3;
  auto batch_gen = lm.generate_batch (seqs, gcfg);
  OBSIDIAN_CHECK_OK (batch_gen);
  OBSIDIAN_CHECK_EQ (batch_gen->size (), 2u);
  OBSIDIAN_CHECK_EQ ((*batch_gen)[0].size (), 7u);
  OBSIDIAN_CHECK_EQ ((*batch_gen)[1].size (), 6u);
}

int
main (void)
{
  return obsidian::testing::run_all_tests (
    "Phase 8: Production Hardening & SafeTensors/HF Interop");
}
