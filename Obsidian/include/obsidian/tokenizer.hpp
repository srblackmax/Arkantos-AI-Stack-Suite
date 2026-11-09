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

#ifndef OBSIDIAN_TOKENIZER_HPP
#define OBSIDIAN_TOKENIZER_HPP

/**
 * @file tokenizer.hpp
 * @brief High-throughput SIMD-accelerated BPE, WordPiece, and Unigram
 *        tokenizers with zero-allocation PMR arena hot paths.
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/memory.hpp>
#include <obsidian/tensor_bridge.hpp>

namespace obsidian
{

/**
 * @brief Special token configuration and reserved IDs.
 */
struct OBSIDIAN_API Special_Tokens
{
  std::string pad_token = "<pad>";
  std::string unk_token = "<unk>";
  std::string bos_token = "<bos>";
  std::string eos_token = "<eos>";
  std::string cls_token = "<cls>";
  std::string sep_token = "<sep>";
  std::string mask_token = "<mask>";
  std::string end_of_word_suffix = "</w>";

  int32_t pad_id = 0;
  int32_t unk_id = 1;
  int32_t bos_id = 2;
  int32_t eos_id = 3;
  int32_t cls_id = 4;
  int32_t sep_id = 5;
  int32_t mask_id = 6;
};

/**
 * @brief Cache-friendly, open-addressed vocabulary table mapping token strings
 *        to 32-bit integer IDs in O(1) time without heap allocation on lookup.
 */
class OBSIDIAN_API Vocabulary
{
public:
  /**
   * @brief Construct an empty Vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Vocabulary (void) = default;

  /**
   * @brief Insert a token string and return its assigned or existing ID.
   * @pre TOKEN must not be empty.
   * @complexity Amortized O(|TOKEN|) time and space.
   */
  int32_t
  add_token (std::string_view token);

  /**
   * @brief Look up the ID of TOKEN in O(1) time without allocating memory.
   * @pre None.
   * @complexity O(|TOKEN|) expected time and O(1) space.
   */
  int32_t
  token_to_id (std::string_view token) const noexcept;

  /**
   * @brief Look up the string view for ID in O(1) time.
   * @pre 0 <= ID < size().
   * @complexity O(1) time and O(1) space.
   */
  Result<std::string_view>
  id_to_token (int32_t id) const noexcept;

  /**
   * @brief Return true if TOKEN is present in the vocabulary.
   * @pre None.
   * @complexity O(|TOKEN|) expected time and O(1) space.
   */
  bool
  contains (std::string_view token) const noexcept
  {
    return token_to_id (token) >= 0;
  }

  /**
   * @brief Return the total number of tokens in the vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  size (void) const noexcept
  {
    return entries_.size ();
  }

  /**
   * @brief Clear all tokens from the vocabulary.
   * @pre None.
   * @complexity O(V) time.
   */
  void
  clear (void) noexcept;

private:
  struct Token_Entry
  {
    size_t offset = 0;
    size_t length = 0;
    uint64_t hash = 0;
  };

  struct Slot
  {
    uint64_t hash = 0;
    int32_t id = -1;
  };

  void
  rebuild_index (size_t new_capacity);

  std::string_view
  entry_view (int32_t id) const noexcept
  {
    const auto &e = entries_[static_cast<size_t> (id)];
    return std::string_view (char_pool_.data () + e.offset, e.length);
  }

  std::vector<char> char_pool_;
  std::vector<Token_Entry> entries_;
  std::vector<Slot> table_;
};

/**
 * @brief High-throughput Byte-Pair Encoding (BPE) tokenizer using SIMD
 *        pre-tokenization and stack/arena doubly-linked symbol merging.
 */
class OBSIDIAN_API BPE_Tokenizer
{
public:
  /**
   * @brief Construct an empty BPE_Tokenizer initialized with special tokens.
   * @pre None.
   * @complexity O(1) time and space.
   */
  explicit BPE_Tokenizer (Special_Tokens specials = Special_Tokens ());

  /**
   * @brief Train BPE vocabulary and merge rules directly from a text corpus.
   * @pre CORPUS must not be empty; TARGET_VOCAB_SIZE >= 16.
   * @complexity O(N_merges * U_unique_words * L_word) time.
   */
  Result<void>
  train_from_corpus (std::span<const std::string_view> corpus,
                     size_t target_vocab_size,
                     size_t min_frequency = 2);

  /**
   * @brief Register an explicit BPE merge rule (LEFT, RIGHT) -> LEFT+RIGHT.
   * @pre LEFT and RIGHT must be non-empty.
   * @complexity Amortized O(|LEFT| + |RIGHT|) time.
   */
  Result<int32_t>
  add_merge (std::string_view left, std::string_view right);

  /**
   * @brief Encode TEXT into an Arena-backed span with zero heap allocations
   *        in the hot path.
   * @pre ARENA must have sufficient capacity or an upstream allocator.
   * @complexity O(N * log_merges) time and O(N) arena space.
   */
  std::span<const int64_t>
  encode_into_arena (std::string_view text,
                     Arena &arena,
                     bool add_bos = false,
                     bool add_eos = false) const;

  /**
   * @brief Encode TEXT into a std::vector<int64_t> of token IDs.
   * @pre TEXT must be valid UTF-8.
   * @complexity O(N) time and space.
   */
  Result<std::vector<int64_t>>
  encode (std::string_view text,
          bool add_bos = false,
          bool add_eos = false) const;

  /**
   * @brief Enable or disable GPT-2 / LLaMA 256-byte-to-Unicode Byte-Level BPE.
   *
   * When enabled, all 256 single-byte codepoints (and their end-of-word
   * variants) are registered in the vocabulary so arbitrary UTF-8 or binary
   * sequences never produce <unk>.
   *
   * @pre None.
   * @complexity O(1) time (256 byte tokens).
   */
  void
  set_byte_level (bool enabled);

  /**
   * @brief Return true if Byte-Level BPE mode is active.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  bool
  is_byte_level (void) const noexcept
  {
    return byte_level_;
  }

  /**
   * @brief Encode a batch of texts into a padded Basalt Sequence_Batch.
   * @pre TEXTS must not be empty.
   * @complexity O(B * S_max) time and space.
   */
  Result<Sequence_Batch>
  encode_batch (std::span<const std::string_view> texts,
                bool add_bos = false,
                bool add_eos = false,
                size_t max_len = 0,
                bool left_pad = false) const;

  /**
   * @brief Encode a batch of texts in parallel across OpenMP worker threads
   *        using per-thread monotonic arenas.
   * @pre TEXTS must not be empty.
   * @complexity O((B / P) * S_max) time across P threads.
   */
  Result<Sequence_Batch>
  encode_batch_parallel (std::span<const std::string_view> texts,
                         bool add_bos = false,
                         bool add_eos = false,
                         size_t max_len = 0,
                         bool left_pad = false,
                         size_t num_threads = 0) const;

  /**
   * @brief Decode a sequence of token IDs back into a UTF-8 string.
   * @pre All IDs must be within vocabulary bounds.
   * @complexity O(N) time and space in output byte length.
   */
  Result<std::string>
  decode (std::span<const int64_t> ids,
          bool skip_special_tokens = true) const;

  /**
   * @brief Save vocabulary and merge table to a text file at PATH.
   * @pre PATH must be a writable filesystem path.
   * @complexity O(V + M) time in vocabulary size V and merge count M.
   */
  Result<void>
  save (const std::string &path) const;

  /**
   * @brief Load vocabulary and merge table from a text file at PATH.
   * @pre PATH must refer to a valid saved BPE tokenizer file.
   * @complexity O(V + M) time and space.
   */
  static Result<BPE_Tokenizer>
  load (const std::string &path);

  /**
   * @brief Export vocabulary and merges in Hugging Face tokenizer.json format.
   * @pre PATH must be a writable filesystem path.
   * @complexity O(V + M) time and space.
   */
  Result<void>
  save_hf_tokenizer_json (const std::string &path) const;

  /**
   * @brief Load a BPE_Tokenizer from a Hugging Face tokenizer.json file.
   * @pre PATH must refer to a valid Hugging Face BPE tokenizer.json file.
   * @complexity O(V + M) time and space.
   */
  static Result<BPE_Tokenizer>
  load_hf_tokenizer_json (const std::string &path);

  /**
   * @brief Return the underlying Vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const Vocabulary &
  vocab (void) const noexcept
  {
    return vocab_;
  }

  /**
   * @brief Return a mutable reference to the underlying Vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Vocabulary &
  vocab_mut (void) noexcept
  {
    return vocab_;
  }

  /**
   * @brief Return the number of tokens in the vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_.size ();
  }

  /**
   * @brief Return the special tokens configuration.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const Special_Tokens &
  special_tokens (void) const noexcept
  {
    return specials_;
  }

private:
  struct Merge_Entry
  {
    uint64_t pair_key = UINT64_MAX;
    uint32_t rank = UINT32_MAX;
    int32_t merged_id = -1;
  };

  struct Merge_Rule
  {
    int32_t left_id = -1;
    int32_t right_id = -1;
    int32_t merged_id = -1;
    uint32_t rank = 0;
  };

  static constexpr uint64_t
  pack_pair (int32_t left, int32_t right) noexcept
  {
    return (static_cast<uint64_t> (static_cast<uint32_t> (left)) << 32)
           | static_cast<uint64_t> (static_cast<uint32_t> (right));
  }

  const Merge_Entry *
  find_merge (int32_t left, int32_t right) const noexcept;

  void
  rebuild_merge_table (size_t new_capacity);

  void
  encode_word_bpe (std::string_view word,
                   Arena &arena,
                   Pmr_Vector<int64_t> &out_ids) const;

  Special_Tokens specials_;
  Vocabulary vocab_;
  std::vector<Merge_Rule> merges_;
  std::vector<Merge_Entry> merge_table_;
  bool byte_level_ = false;
};

/**
 * @brief Greedy longest-prefix WordPiece tokenizer (BERT-compatible).
 */
class OBSIDIAN_API WordPiece_Tokenizer
{
public:
  /**
   * @brief Construct a WordPiece_Tokenizer with optional continuation prefix.
   * @pre None.
   * @complexity O(1) time and space.
   */
  explicit WordPiece_Tokenizer (
    Special_Tokens specials = Special_Tokens (),
    std::string continuing_subword_prefix = "##",
    size_t max_input_chars_per_word = 100);

  /**
   * @brief Add a token or continuation subword to the WordPiece vocabulary.
   * @pre TOKEN must not be empty.
   * @complexity Amortized O(|TOKEN|) time.
   */
  int32_t
  add_token (std::string_view token);

  /**
   * @brief Encode TEXT into token IDs using greedy longest-prefix matching.
   * @pre TEXT must be valid UTF-8.
   * @complexity O(N * L_max) worst-case time, O(N) typical time.
   */
  Result<std::vector<int64_t>>
  encode (std::string_view text,
          bool add_cls_sep = false) const;

  /**
   * @brief Decode WordPiece token IDs back into a reconstructed string.
   * @pre IDs must be valid vocabulary indices.
   * @complexity O(N) time and space.
   */
  Result<std::string>
  decode (std::span<const int64_t> ids,
          bool skip_special_tokens = true) const;

  /**
   * @brief Return the vocabulary size.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_.size ();
  }

  /**
   * @brief Return the underlying Vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const Vocabulary &
  vocab (void) const noexcept
  {
    return vocab_;
  }

private:
  Special_Tokens specials_;
  std::string prefix_;
  size_t max_chars_;
  Vocabulary vocab_;
};

/**
 * @brief Viterbi minimum-cost lattice Unigram subword tokenizer
 *        (SentencePiece-compatible).
 */
class OBSIDIAN_API Unigram_Tokenizer
{
public:
  /**
   * @brief Construct a Unigram_Tokenizer initialized with special tokens.
   * @pre None.
   * @complexity O(1) time and space.
   */
  explicit Unigram_Tokenizer (Special_Tokens specials = Special_Tokens ());

  /**
   * @brief Add a subword piece with its log-probability score.
   * @pre PIECE must not be empty; LOG_PROB <= 0.0f.
   * @complexity Amortized O(|PIECE|) time.
   */
  int32_t
  add_piece (std::string_view piece, float log_prob);

  /**
   * @brief Segment TEXT into maximum-likelihood subword IDs via Viterbi DP.
   * @pre TEXT must be valid UTF-8.
   * @complexity O(N * K_max_piece_len) time and O(N) arena space.
   */
  Result<std::vector<int64_t>>
  encode (std::string_view text,
          bool add_bos = false,
          bool add_eos = false) const;

  /**
   * @brief Decode Unigram token IDs back into text.
   * @pre IDs must be valid vocabulary indices.
   * @complexity O(N) time and space.
   */
  Result<std::string>
  decode (std::span<const int64_t> ids,
          bool skip_special_tokens = true) const;

  /**
   * @brief Return the number of subword pieces in the model.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_.size ();
  }

private:
  Special_Tokens specials_;
  Vocabulary vocab_;
  std::vector<float> scores_;
  size_t max_piece_len_ = 1;
};

/**
 * @brief Concept satisfied by any Obsidian tokenizer engine.
 */
template <typename T>
concept Tokenizer_Engine = requires (const T &tok, std::string_view text,
                                     std::span<const int64_t> ids)
{
  { tok.encode (text) } -> std::same_as<Result<std::vector<int64_t>>>;
  { tok.decode (ids) } -> std::same_as<Result<std::string>>;
  { tok.vocab_size () } -> std::convertible_to<size_t>;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_TOKENIZER_HPP */
