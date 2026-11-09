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

#include <obsidian/tokenizer.hpp>
#include "test_harness.hpp"

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using namespace obsidian;

static_assert (Tokenizer_Engine<BPE_Tokenizer>);
static_assert (Tokenizer_Engine<WordPiece_Tokenizer>);
static_assert (Tokenizer_Engine<Unigram_Tokenizer>);

OBSIDIAN_TEST (vocabulary_open_addressing)
{
  Vocabulary vocab;
  int32_t id_a = vocab.add_token ("alpha");
  int32_t id_b = vocab.add_token ("beta");
  int32_t id_a_dup = vocab.add_token ("alpha");

  OBSIDIAN_CHECK_EQ (id_a, 0);
  OBSIDIAN_CHECK_EQ (id_b, 1);
  OBSIDIAN_CHECK_EQ (id_a_dup, 0);
  OBSIDIAN_CHECK_EQ (vocab.token_to_id ("beta"), 1);
  OBSIDIAN_CHECK_EQ (vocab.token_to_id ("nonexistent"), -1);

  for (int i = 0; i < 200; ++i)
    {
      std::string tok = "tok_" + std::to_string (i);
      vocab.add_token (tok);
    }
  OBSIDIAN_CHECK_EQ (vocab.size (), 202u);
  OBSIDIAN_CHECK_EQ (vocab.id_to_token (0).value (), "alpha");
  OBSIDIAN_CHECK_EQ (vocab.token_to_id ("tok_199"), 201);
}

OBSIDIAN_TEST (bpe_tokenizer_training_encode_decode_and_io)
{
  std::vector<std::string_view> corpus = {
    "low lower lowest low lower",
    "new newer newest new newer",
    "wider widest wide low new"
  };

  BPE_Tokenizer bpe;
  auto train_res = bpe.train_from_corpus (corpus, 48, 2);
  OBSIDIAN_CHECK_OK (train_res);
  OBSIDIAN_CHECK (bpe.vocab_size () > 15u);

  /* Zero-allocation hot path using an explicit stack-backed Arena.  */
  Arena arena;
  auto span_ids = bpe.encode_into_arena ("low lower newest", arena,
                                         true, true);
  OBSIDIAN_CHECK (span_ids.size () >= 5u);
  OBSIDIAN_CHECK_EQ (span_ids.front (), bpe.special_tokens ().bos_id);
  OBSIDIAN_CHECK_EQ (span_ids.back (), bpe.special_tokens ().eos_id);

  auto dec_res = bpe.decode (span_ids, true);
  OBSIDIAN_CHECK_OK (dec_res);
  OBSIDIAN_CHECK_EQ (*dec_res, "low lower newest");

  /* Batch encoding into a Basalt Sequence_Batch.  */
  std::vector<std::string_view> batch_texts = {
    "low newest",
    "lower wider lowest"
  };
  auto batch_res = bpe.encode_batch (batch_texts, true, true);
  OBSIDIAN_CHECK_OK (batch_res);
  OBSIDIAN_CHECK_EQ (batch_res->batch_size (), 2u);
  OBSIDIAN_CHECK (batch_res->seq_len () >= 4u);

  /* Save and reload verification.  */
  const std::string tmp_path = "/tmp/obsidian_test_bpe.tok";
  OBSIDIAN_CHECK_OK (bpe.save (tmp_path));
  auto loaded_res = BPE_Tokenizer::load (tmp_path);
  std::remove (tmp_path.c_str ());
  OBSIDIAN_CHECK_OK (loaded_res);

  auto enc1 = bpe.encode ("lowest newer").value ();
  auto enc2 = loaded_res->encode ("lowest newer").value ();
  OBSIDIAN_CHECK_EQ (enc1.size (), enc2.size ());
  for (size_t i = 0; i < enc1.size (); ++i)
    {
      OBSIDIAN_CHECK_EQ (enc1[i], enc2[i]);
    }
}

OBSIDIAN_TEST (wordpiece_tokenizer_subwords)
{
  WordPiece_Tokenizer wp;
  int32_t id_un = wp.add_token ("un");
  int32_t id_aff = wp.add_token ("##aff");
  int32_t id_able = wp.add_token ("##able");
  int32_t id_play = wp.add_token ("play");
  int32_t id_ing = wp.add_token ("##ing");

  auto enc_res = wp.encode ("unaffable playing unknownword", true);
  OBSIDIAN_CHECK_OK (enc_res);
  const auto &ids = *enc_res;
  OBSIDIAN_CHECK_EQ (ids.size (), 8u);
  OBSIDIAN_CHECK_EQ (ids[1], id_un);
  OBSIDIAN_CHECK_EQ (ids[2], id_aff);
  OBSIDIAN_CHECK_EQ (ids[3], id_able);
  OBSIDIAN_CHECK_EQ (ids[4], id_play);
  OBSIDIAN_CHECK_EQ (ids[5], id_ing);
  OBSIDIAN_CHECK_EQ (ids[6], 1); /* unk_id */

  std::vector<int64_t> known_ids = {id_un, id_aff, id_able, id_play, id_ing};
  auto dec_res = wp.decode (known_ids, true);
  OBSIDIAN_CHECK_OK (dec_res);
  OBSIDIAN_CHECK_EQ (*dec_res, "unaffable playing");
}

OBSIDIAN_TEST (unigram_tokenizer_viterbi_segmentation)
{
  Unigram_Tokenizer uni;
  uni.add_piece ("h", -3.0f);
  uni.add_piece ("e", -3.0f);
  uni.add_piece ("l", -3.0f);
  uni.add_piece ("o", -3.0f);
  int32_t id_hel = uni.add_piece ("hel", -1.2f);
  int32_t id_lo = uni.add_piece ("lo", -1.1f);
  uni.add_piece ("he", -2.5f);
  uni.add_piece ("llo", -2.5f);

  /* "hel" + "lo" has score -2.3, beating "he" + "llo" (-5.0).  */
  auto enc_res = uni.encode ("hello");
  OBSIDIAN_CHECK_OK (enc_res);
  OBSIDIAN_CHECK_EQ (enc_res->size (), 2u);
  OBSIDIAN_CHECK_EQ ((*enc_res)[0], id_hel);
  OBSIDIAN_CHECK_EQ ((*enc_res)[1], id_lo);

  auto dec_res = uni.decode (*enc_res);
  OBSIDIAN_CHECK_OK (dec_res);
  OBSIDIAN_CHECK_EQ (*dec_res, "hello");
}

int
main (void)
{
  return obsidian::testing::run_all_tests ("Phase 2: SIMD Tokenizer Suite");
}
