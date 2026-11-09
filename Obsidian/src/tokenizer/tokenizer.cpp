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
#include <obsidian/simd.hpp>
#include <obsidian/unicode.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace obsidian
{

namespace
{

constexpr uint64_t
hash_bytes (std::string_view sv) noexcept
{
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : sv)
    {
      h ^= static_cast<uint64_t> (c);
      h *= 1099511628211ULL;
    }
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (h == 0) ? 1ULL : h;
}

constexpr uint64_t
hash_u64 (uint64_t k) noexcept
{
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

size_t
utf8_char_len (unsigned char lead) noexcept
{
  if (lead < 0x80)
    {
      return 1;
    }
  if ((lead & 0xE0) == 0xC0)
    {
      return 2;
    }
  if ((lead & 0xF0) == 0xE0)
    {
      return 3;
    }
  if ((lead & 0xF8) == 0xF0)
    {
      return 4;
    }
  return 1;
}

bool
is_special_id (int64_t id, const Special_Tokens &sp) noexcept
{
  return id == sp.pad_id || id == sp.unk_id || id == sp.bos_id
         || id == sp.eos_id || id == sp.cls_id || id == sp.sep_id
         || id == sp.mask_id;
}

} /* namespace */

/* -------------------------------------------------------------------------
   Vocabulary implementation
   ------------------------------------------------------------------------- */

int32_t
Vocabulary::add_token (std::string_view token)
{
  int32_t existing = token_to_id (token);
  if (existing >= 0)
    {
      return existing;
    }

  if (table_.empty () || (entries_.size () + 1) * 2 >= table_.size ())
    {
      size_t next_cap = table_.empty () ? 64 : table_.size () * 2;
      rebuild_index (next_cap);
    }

  /* Append characters into char_pool_.  */
  size_t offset = char_pool_.size ();
  char_pool_.insert (char_pool_.end (), token.begin (), token.end ());
  uint64_t h = hash_bytes (token);
  int32_t new_id = static_cast<int32_t> (entries_.size ());
  entries_.push_back (Token_Entry {offset, token.size (), h});

  size_t mask = table_.size () - 1;
  size_t slot_idx = static_cast<size_t> (h) & mask;
  while (table_[slot_idx].id >= 0)
    {
      slot_idx = (slot_idx + 1) & mask;
    }
  table_[slot_idx].hash = h;
  table_[slot_idx].id = new_id;
  return new_id;
}

int32_t
Vocabulary::token_to_id (std::string_view token) const noexcept
{
  if (table_.empty ())
    {
      return -1;
    }
  uint64_t h = hash_bytes (token);
  size_t mask = table_.size () - 1;
  size_t slot_idx = static_cast<size_t> (h) & mask;

  while (table_[slot_idx].id >= 0)
    {
      if (table_[slot_idx].hash == h)
        {
          int32_t cand_id = table_[slot_idx].id;
          if (entry_view (cand_id) == token)
            {
              return cand_id;
            }
        }
      slot_idx = (slot_idx + 1) & mask;
    }
  return -1;
}

Result<std::string_view>
Vocabulary::id_to_token (int32_t id) const noexcept
{
  if (id < 0 || static_cast<size_t> (id) >= entries_.size ())
    {
      return OBSIDIAN_ERROR (Error_Code::Out_Of_Range,
                             "token ID out of vocabulary bounds");
    }
  return entry_view (id);
}

void
Vocabulary::clear (void) noexcept
{
  char_pool_.clear ();
  entries_.clear ();
  table_.clear ();
}

void
Vocabulary::rebuild_index (size_t new_capacity)
{
  table_.assign (new_capacity, Slot {});
  size_t mask = new_capacity - 1;
  for (size_t i = 0; i < entries_.size (); ++i)
    {
      uint64_t h = entries_[i].hash;
      size_t slot_idx = static_cast<size_t> (h) & mask;
      while (table_[slot_idx].id >= 0)
        {
          slot_idx = (slot_idx + 1) & mask;
        }
      table_[slot_idx].hash = h;
      table_[slot_idx].id = static_cast<int32_t> (i);
    }
}

/* -------------------------------------------------------------------------
   BPE_Tokenizer implementation
   ------------------------------------------------------------------------- */

BPE_Tokenizer::BPE_Tokenizer (Special_Tokens specials)
  : specials_ (std::move (specials))
{
  specials_.pad_id = vocab_.add_token (specials_.pad_token);
  specials_.unk_id = vocab_.add_token (specials_.unk_token);
  specials_.bos_id = vocab_.add_token (specials_.bos_token);
  specials_.eos_id = vocab_.add_token (specials_.eos_token);
  specials_.cls_id = vocab_.add_token (specials_.cls_token);
  specials_.sep_id = vocab_.add_token (specials_.sep_token);
  specials_.mask_id = vocab_.add_token (specials_.mask_token);
}

void
BPE_Tokenizer::set_byte_level (bool enabled)
{
  byte_level_ = enabled;
  if (byte_level_)
    {
      for (uint32_t b = 0; b < 256; ++b)
        {
          char raw_c = static_cast<char> (static_cast<uint8_t> (b));
          std::string b_str = unicode::byte_to_unicode_encode (
            std::string_view (&raw_c, 1));
          vocab_.add_token (b_str);
          if (!specials_.end_of_word_suffix.empty ())
            {
              vocab_.add_token (b_str + specials_.end_of_word_suffix);
            }
        }
    }
}

const BPE_Tokenizer::Merge_Entry *
BPE_Tokenizer::find_merge (int32_t left, int32_t right) const noexcept
{
  if (merge_table_.empty () || left < 0 || right < 0)
    {
      return nullptr;
    }
  uint64_t key = pack_pair (left, right);
  size_t mask = merge_table_.size () - 1;
  size_t idx = static_cast<size_t> (hash_u64 (key)) & mask;

  while (merge_table_[idx].pair_key != UINT64_MAX)
    {
      if (merge_table_[idx].pair_key == key)
        {
          return &merge_table_[idx];
        }
      idx = (idx + 1) & mask;
    }
  return nullptr;
}

void
BPE_Tokenizer::rebuild_merge_table (size_t new_capacity)
{
  merge_table_.assign (new_capacity, Merge_Entry {});
  size_t mask = new_capacity - 1;
  for (const auto &rule : merges_)
    {
      uint64_t key = pack_pair (rule.left_id, rule.right_id);
      size_t idx = static_cast<size_t> (hash_u64 (key)) & mask;
      while (merge_table_[idx].pair_key != UINT64_MAX)
        {
          idx = (idx + 1) & mask;
        }
      merge_table_[idx].pair_key = key;
      merge_table_[idx].rank = rule.rank;
      merge_table_[idx].merged_id = rule.merged_id;
    }
}

Result<int32_t>
BPE_Tokenizer::add_merge (std::string_view left, std::string_view right)
{
  if (left.empty () || right.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "BPE merge tokens must not be empty");
    }

  int32_t left_id = vocab_.add_token (left);
  int32_t right_id = vocab_.add_token (right);

  if (const Merge_Entry *existing = find_merge (left_id, right_id))
    {
      return existing->merged_id;
    }

  std::string combined;
  combined.reserve (left.size () + right.size ());
  combined.append (left);
  combined.append (right);
  int32_t merged_id = vocab_.add_token (combined);

  uint32_t rank = static_cast<uint32_t> (merges_.size ());
  merges_.push_back (Merge_Rule {left_id, right_id, merged_id, rank});

  if (merge_table_.empty ()
      || (merges_.size () + 1) * 2 >= merge_table_.size ())
    {
      size_t next_cap = merge_table_.empty () ? 64 : merge_table_.size () * 2;
      rebuild_merge_table (next_cap);
    }
  else
    {
      uint64_t key = pack_pair (left_id, right_id);
      size_t mask = merge_table_.size () - 1;
      size_t idx = static_cast<size_t> (hash_u64 (key)) & mask;
      while (merge_table_[idx].pair_key != UINT64_MAX)
        {
          idx = (idx + 1) & mask;
        }
      merge_table_[idx].pair_key = key;
      merge_table_[idx].rank = rank;
      merge_table_[idx].merged_id = merged_id;
    }

  return merged_id;
}

Result<void>
BPE_Tokenizer::train_from_corpus (std::span<const std::string_view> corpus,
                                  size_t target_vocab_size,
                                  size_t min_frequency)
{
  if (corpus.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "train_from_corpus requires non-empty corpus");
    }
  if (target_vocab_size < 8)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "target_vocab_size must be >= 8");
    }

  /* Count word frequencies across the corpus using SIMD pre-tokenization.  */
  std::unordered_map<std::string, size_t> word_counts;
  Arena scratch;
  for (std::string_view doc : corpus)
    {
      scratch.reset ();
      Pmr_Vector<std::string_view> spans (scratch.resource ());
      simd::scan_pretoken_spans (doc, spans, true);
      for (std::string_view w : spans)
        {
          std::string key = byte_level_
                              ? unicode::byte_to_unicode_encode (w)
                              : std::string (w);
          ++word_counts[std::move (key)];
        }
    }

  struct Word_Seq
  {
    std::vector<int32_t> symbols;
    size_t count = 0;
  };

  std::vector<Word_Seq> words;
  words.reserve (word_counts.size ());

  for (const auto &[word_str, count] : word_counts)
    {
      Word_Seq ws;
      ws.count = count;
      size_t pos = 0;
      const size_t n = word_str.size ();
      while (pos < n)
        {
          size_t clen = std::min (
            utf8_char_len (static_cast<unsigned char> (word_str[pos])),
            n - pos);
          std::string piece (word_str.data () + pos, clen);
          if (pos + clen == n)
            {
              piece.append (specials_.end_of_word_suffix);
            }
          ws.symbols.push_back (vocab_.add_token (piece));
          pos += clen;
        }
      if (!ws.symbols.empty ())
        {
          words.push_back (std::move (ws));
        }
    }

  /* Iteratively merge the most frequent adjacent symbol pair.  */
  while (vocab_.size () < target_vocab_size)
    {
      std::unordered_map<uint64_t, size_t> pair_counts;
      for (const auto &ws : words)
        {
          if (ws.symbols.size () < 2)
            {
              continue;
            }
          for (size_t i = 0; i + 1 < ws.symbols.size (); ++i)
            {
              uint64_t key = pack_pair (ws.symbols[i], ws.symbols[i + 1]);
              pair_counts[key] += ws.count;
            }
        }

      uint64_t best_key = UINT64_MAX;
      size_t best_freq = 0;
      for (const auto &[key, freq] : pair_counts)
        {
          if (freq >= min_frequency
              && (freq > best_freq
                  || (freq == best_freq && key < best_key)))
            {
              best_freq = freq;
              best_key = key;
            }
        }

      if (best_key == UINT64_MAX)
        {
          break;
        }

      int32_t left_id = static_cast<int32_t> (best_key >> 32);
      int32_t right_id = static_cast<int32_t> (best_key & 0xFFFFFFFFULL);
      std::string left_s (vocab_.id_to_token (left_id).value ());
      std::string right_s (vocab_.id_to_token (right_id).value ());

      auto merged_res = add_merge (left_s, right_s);
      if (!merged_res.has_value ())
        {
          return std::unexpected (merged_res.error ());
        }
      int32_t merged_id = *merged_res;

      for (auto &ws : words)
        {
          if (ws.symbols.size () < 2)
            {
              continue;
            }
          std::vector<int32_t> updated;
          updated.reserve (ws.symbols.size ());
          size_t i = 0;
          while (i < ws.symbols.size ())
            {
              if (i + 1 < ws.symbols.size ()
                  && ws.symbols[i] == left_id
                  && ws.symbols[i + 1] == right_id)
                {
                  updated.push_back (merged_id);
                  i += 2;
                }
              else
                {
                  updated.push_back (ws.symbols[i]);
                  ++i;
                }
            }
          ws.symbols = std::move (updated);
        }
    }

  return {};
}

void
BPE_Tokenizer::encode_word_bpe (std::string_view word,
                                Arena &arena,
                                Pmr_Vector<int64_t> &out_ids) const
{
  if (word.empty ())
    {
      return;
    }

  std::string byte_mapped;
  if (byte_level_)
    {
      byte_mapped = unicode::byte_to_unicode_encode (word);
      word = std::string_view (byte_mapped);
    }

  /* Fast path: check if whole word (or word + </w>) is a single token.  */
  int32_t direct_id = vocab_.token_to_id (word);
  if (direct_id >= 0)
    {
      out_ids.push_back (direct_id);
      return;
    }

  std::array<char, 256> stack_eow_buf;
  const size_t eow_len = specials_.end_of_word_suffix.size ();
  if (word.size () + eow_len <= stack_eow_buf.size ())
    {
      std::copy (word.begin (), word.end (), stack_eow_buf.begin ());
      std::copy (specials_.end_of_word_suffix.begin (),
                 specials_.end_of_word_suffix.end (),
                 stack_eow_buf.begin () + word.size ());
      std::string_view word_eow (stack_eow_buf.data (),
                                 word.size () + eow_len);
      int32_t full_id = vocab_.token_to_id (word_eow);
      if (full_id >= 0)
        {
          out_ids.push_back (full_id);
          return;
        }
    }

  struct Symbol
  {
    int32_t id = -1;
    int32_t prev = -1;
    int32_t next = -1;
  };

  auto syms = arena.allocate_span<Symbol> (word.size ());
  size_t sym_count = 0;
  size_t pos = 0;
  const size_t n = word.size ();

  while (pos < n)
    {
      size_t clen = std::min (
        utf8_char_len (static_cast<unsigned char> (word[pos])), n - pos);
      int32_t sym_id = -1;
      if (pos + clen == n && clen + eow_len <= stack_eow_buf.size ())
        {
          std::copy (word.data () + pos, word.data () + pos + clen,
                     stack_eow_buf.begin ());
          std::copy (specials_.end_of_word_suffix.begin (),
                     specials_.end_of_word_suffix.end (),
                     stack_eow_buf.begin () + clen);
          std::string_view piece_eow (stack_eow_buf.data (), clen + eow_len);
          sym_id = vocab_.token_to_id (piece_eow);
        }
      if (sym_id < 0)
        {
          sym_id = vocab_.token_to_id (word.substr (pos, clen));
        }
      if (sym_id < 0)
        {
          sym_id = specials_.unk_id;
        }

      int32_t idx = static_cast<int32_t> (sym_count);
      syms[sym_count].id = sym_id;
      syms[sym_count].prev = idx - 1;
      syms[sym_count].next = idx + 1;
      ++sym_count;
      pos += clen;
    }

  if (sym_count == 0)
    {
      return;
    }
  syms[sym_count - 1].next = -1;

  /* Iteratively merge the lowest-rank adjacent pair in place.  */
  while (sym_count > 1 && !merge_table_.empty ())
    {
      uint32_t best_rank = UINT32_MAX;
      int32_t best_pos = -1;
      int32_t best_merged_id = -1;

      for (int32_t cur = 0; cur != -1;
           cur = syms[static_cast<size_t> (cur)].next)
        {
          int32_t nxt = syms[static_cast<size_t> (cur)].next;
          if (nxt == -1)
            {
              break;
            }
          const Merge_Entry *me = find_merge (
            syms[static_cast<size_t> (cur)].id,
            syms[static_cast<size_t> (nxt)].id);
          if (me != nullptr && me->rank < best_rank)
            {
              best_rank = me->rank;
              best_pos = cur;
              best_merged_id = me->merged_id;
            }
        }

      if (best_pos == -1)
        {
          break;
        }

      size_t u_pos = static_cast<size_t> (best_pos);
      size_t u_nxt = static_cast<size_t> (syms[u_pos].next);
      syms[u_pos].id = best_merged_id;
      int32_t after = syms[u_nxt].next;
      syms[u_pos].next = after;
      if (after != -1)
        {
          syms[static_cast<size_t> (after)].prev = best_pos;
        }
    }

  for (int32_t cur = 0; cur != -1;
       cur = syms[static_cast<size_t> (cur)].next)
    {
      out_ids.push_back (syms[static_cast<size_t> (cur)].id);
    }
}

std::span<const int64_t>
BPE_Tokenizer::encode_into_arena (std::string_view text,
                                  Arena &arena,
                                  bool add_bos,
                                  bool add_eos) const
{
  Pmr_Vector<std::string_view> pre_tokens (arena.resource ());
  simd::scan_pretoken_spans (text, pre_tokens, true);

  Pmr_Vector<int64_t> out_ids (arena.resource ());
  out_ids.reserve (pre_tokens.size () * 2 + 2);

  if (add_bos)
    {
      out_ids.push_back (specials_.bos_id);
    }

  for (std::string_view word : pre_tokens)
    {
      encode_word_bpe (word, arena, out_ids);
    }

  if (add_eos)
    {
      out_ids.push_back (specials_.eos_id);
    }

  auto result_span = arena.allocate_span<int64_t> (out_ids.size ());
  std::copy (out_ids.begin (), out_ids.end (), result_span.begin ());
  return result_span;
}

Result<std::vector<int64_t>>
BPE_Tokenizer::encode (std::string_view text,
                       bool add_bos,
                       bool add_eos) const
{
  if (!byte_level_)
    {
      auto utf8_ok = simd::validate_and_count_utf8 (text);
      if (!utf8_ok.has_value ())
        {
          return std::unexpected (utf8_ok.error ());
        }
    }

  Arena &arena = thread_local_arena ();
  arena.reset ();
  auto sp = encode_into_arena (text, arena, add_bos, add_eos);
  return std::vector<int64_t> (sp.begin (), sp.end ());
}

Result<Sequence_Batch>
BPE_Tokenizer::encode_batch (std::span<const std::string_view> texts,
                             bool add_bos,
                             bool add_eos,
                             size_t max_len,
                             bool left_pad) const
{
  if (texts.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "encode_batch requires at least one text");
    }

  std::vector<std::vector<int64_t>> encoded_seqs;
  encoded_seqs.reserve (texts.size ());

  for (std::string_view t : texts)
    {
      auto enc = encode (t, add_bos, add_eos);
      if (!enc.has_value ())
        {
          return std::unexpected (enc.error ());
        }
      if (enc->empty ())
        {
          enc->push_back (specials_.pad_id);
        }
      encoded_seqs.push_back (std::move (*enc));
    }

  return Sequence_Batch::from_sequences (
    encoded_seqs, specials_.pad_id, max_len, left_pad);
}

Result<Sequence_Batch>
BPE_Tokenizer::encode_batch_parallel (std::span<const std::string_view> texts,
                                      bool add_bos,
                                      bool add_eos,
                                      size_t max_len,
                                      bool left_pad,
                                      size_t num_threads) const
{
  if (texts.empty ())
    {
      return OBSIDIAN_ERROR (
        Error_Code::Invalid_Argument,
        "encode_batch_parallel requires at least one text");
    }

  const size_t n = texts.size ();
  std::vector<std::vector<int64_t>> encoded_seqs (n);
  std::mutex err_mtx;
  bool has_err = false;
  Error first_err (Error_Code::Ok, "");

#ifdef _OPENMP
  int threads = (num_threads > 0)
                  ? static_cast<int> (num_threads)
                  : omp_get_max_threads ();
#pragma omp parallel for schedule(dynamic) num_threads(threads)
#endif
  for (size_t i = 0; i < n; ++i)
    {
      if (!byte_level_)
        {
          auto utf8_ok = simd::validate_and_count_utf8 (texts[i]);
          if (!utf8_ok.has_value ())
            {
              std::lock_guard<std::mutex> lock (err_mtx);
              if (!has_err)
                {
                  has_err = true;
                  first_err = utf8_ok.error ();
                }
              continue;
            }
        }
      Arena &arena = thread_local_arena ();
      arena.reset ();
      auto sp = encode_into_arena (texts[i], arena, add_bos, add_eos);
      if (sp.empty ())
        {
          encoded_seqs[i].push_back (specials_.pad_id);
        }
      else
        {
          encoded_seqs[i].assign (sp.begin (), sp.end ());
        }
    }

  if (has_err)
    {
      return std::unexpected (first_err);
    }

  return Sequence_Batch::from_sequences (
    encoded_seqs, specials_.pad_id, max_len, left_pad);
}

Result<std::string>
BPE_Tokenizer::decode (std::span<const int64_t> ids,
                       bool skip_special_tokens) const
{
  std::string out;
  const std::string &eow = specials_.end_of_word_suffix;
  bool need_space = false;

  for (int64_t id64 : ids)
    {
      if (skip_special_tokens && is_special_id (id64, specials_))
        {
          continue;
        }
      auto tok_res = vocab_.id_to_token (static_cast<int32_t> (id64));
      if (!tok_res.has_value ())
        {
          return std::unexpected (tok_res.error ());
        }
      std::string_view tok = *tok_res;
      if (need_space)
        {
          out.push_back (' ');
          need_space = false;
        }
      std::string_view piece = tok;
      if (!eow.empty () && tok.size () >= eow.size ()
          && tok.substr (tok.size () - eow.size ()) == eow)
        {
          piece = tok.substr (0, tok.size () - eow.size ());
          need_space = true;
        }
      if (byte_level_ && !is_special_id (id64, specials_))
        {
          auto dec_res = unicode::byte_to_unicode_decode (piece);
          if (!dec_res.has_value ())
            {
              return std::unexpected (dec_res.error ());
            }
          out.append (*dec_res);
        }
      else
        {
          out.append (piece);
        }
    }
  return out;
}

Result<void>
BPE_Tokenizer::save (const std::string &path) const
{
  std::ofstream ofs (path);
  if (!ofs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open tokenizer file for writing");
    }

  ofs << "OBSIDIAN_BPE_V1\n";
  ofs << vocab_.size () << " " << merges_.size () << "\n";
  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      std::string_view tok = vocab_.id_to_token (
        static_cast<int32_t> (i)).value ();
      ofs << tok << "\n";
    }
  for (const auto &m : merges_)
    {
      std::string_view l = vocab_.id_to_token (m.left_id).value ();
      std::string_view r = vocab_.id_to_token (m.right_id).value ();
      ofs << l << " " << r << "\n";
    }
  return {};
}

Result<BPE_Tokenizer>
BPE_Tokenizer::load (const std::string &path)
{
  std::ifstream ifs (path);
  if (!ifs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open tokenizer file for reading");
    }

  std::string header;
  if (!std::getline (ifs, header) || header != "OBSIDIAN_BPE_V1")
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "invalid BPE tokenizer file header");
    }

  size_t num_vocab = 0;
  size_t num_merges = 0;
  if (!(ifs >> num_vocab >> num_merges))
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "failed to read BPE counts");
    }
  std::string dummy;
  std::getline (ifs, dummy);

  BPE_Tokenizer tok;
  tok.vocab_.clear ();
  for (size_t i = 0; i < num_vocab; ++i)
    {
      std::string line;
      if (!std::getline (ifs, line))
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "unexpected EOF reading BPE vocabulary");
        }
      tok.vocab_.add_token (line);
    }

  for (size_t i = 0; i < num_merges; ++i)
    {
      std::string left;
      std::string right;
      if (!(ifs >> left >> right))
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "unexpected EOF reading BPE merges");
        }
      auto res = tok.add_merge (left, right);
      if (!res.has_value ())
        {
          return std::unexpected (res.error ());
        }
    }

  return tok;
}

namespace
{

std::string
json_escape_tok (std::string_view s)
{
  std::string out;
  out.reserve (s.size () + 4);
  for (char c : s)
    {
      switch (c)
        {
        case '"':
          out.append ("\\\"");
          break;
        case '\\':
          out.append ("\\\\");
          break;
        case '\n':
          out.append ("\\n");
          break;
        case '\r':
          out.append ("\\r");
          break;
        case '\t':
          out.append ("\\t");
          break;
        default:
          out.push_back (c);
          break;
        }
    }
  return out;
}

class HF_Json_Scanner
{
public:
  explicit HF_Json_Scanner (std::string_view src) noexcept
    : src_ (src),
      pos_ (0)
  {
  }

  void
  skip_ws (void) noexcept
  {
    while (pos_ < src_.size ())
      {
        char c = src_[pos_];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
          {
            ++pos_;
          }
        else
          {
            break;
          }
      }
  }

  bool
  consume (char expected) noexcept
  {
    skip_ws ();
    if (pos_ < src_.size () && src_[pos_] == expected)
      {
        ++pos_;
        return true;
      }
    return false;
  }

  char
  peek (void) noexcept
  {
    skip_ws ();
    return (pos_ < src_.size ()) ? src_[pos_] : '\0';
  }

  Result<std::string>
  parse_string (void)
  {
    if (!consume ('"'))
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected '\"' in tokenizer.json");
      }
    std::string out;
    while (pos_ < src_.size ())
      {
        char c = src_[pos_++];
        if (c == '"')
          {
            return out;
          }
        if (c == '\\')
          {
            if (pos_ >= src_.size ())
              {
                return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                       "unterminated escape in JSON");
              }
            char esc = src_[pos_++];
            switch (esc)
              {
              case '"':
              case '\\':
              case '/':
                out.push_back (esc);
                break;
              case 'n':
                out.push_back ('\n');
                break;
              case 'r':
                out.push_back ('\r');
                break;
              case 't':
                out.push_back ('\t');
                break;
              default:
                out.push_back (esc);
                break;
              }
          }
        else
          {
            out.push_back (c);
          }
      }
    return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                           "unterminated string in tokenizer.json");
  }

  Result<int32_t>
  parse_int (void)
  {
    skip_ws ();
    bool neg = false;
    if (pos_ < src_.size () && src_[pos_] == '-')
      {
        neg = true;
        ++pos_;
      }
    if (pos_ >= src_.size () || src_[pos_] < '0' || src_[pos_] > '9')
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected integer in tokenizer.json");
      }
    int64_t val = 0;
    while (pos_ < src_.size () && src_[pos_] >= '0' && src_[pos_] <= '9')
      {
        val = val * 10 + (src_[pos_] - '0');
        ++pos_;
      }
    return static_cast<int32_t> (neg ? -val : val);
  }

  Result<bool>
  parse_bool (void)
  {
    skip_ws ();
    if (src_.substr (pos_, 4) == "true")
      {
        pos_ += 4;
        return true;
      }
    if (src_.substr (pos_, 5) == "false")
      {
        pos_ += 5;
        return false;
      }
    return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                           "expected boolean in tokenizer.json");
  }

  Result<void>
  skip_value (void)
  {
    char c = peek ();
    if (c == '"')
      {
        auto s = parse_string ();
        if (!s.has_value ())
          {
            return std::unexpected (s.error ());
          }
        return {};
      }
    if (c == '{')
      {
        consume ('{');
        if (consume ('}'))
          {
            return {};
          }
        while (true)
          {
            auto k = parse_string ();
            if (!k.has_value () || !consume (':'))
              {
                return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                       "malformed object in tokenizer.json");
              }
            auto sv = skip_value ();
            if (!sv.has_value ())
              {
                return sv;
              }
            if (consume (','))
              {
                continue;
              }
            if (consume ('}'))
              {
                return {};
              }
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "unclosed object in tokenizer.json");
          }
      }
    if (c == '[')
      {
        consume ('[');
        if (consume (']'))
          {
            return {};
          }
        while (true)
          {
            auto sv = skip_value ();
            if (!sv.has_value ())
              {
                return sv;
              }
            if (consume (','))
              {
                continue;
              }
            if (consume (']'))
              {
                return {};
              }
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "unclosed array in tokenizer.json");
          }
      }
    /* Primitive literal (number, true, false, null).  */
    skip_ws ();
    while (pos_ < src_.size ())
      {
        char ch = src_[pos_];
        if (ch == ',' || ch == '}' || ch == ']' || ch == ' '
            || ch == '\t' || ch == '\n' || ch == '\r')
          {
            break;
          }
        ++pos_;
      }
    return {};
  }

private:
  std::string_view src_;
  size_t pos_;
};

} /* namespace */

Result<void>
BPE_Tokenizer::save_hf_tokenizer_json (const std::string &path) const
{
  std::ofstream ofs (path);
  if (!ofs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open tokenizer.json for writing");
    }

  ofs << "{\n  \"version\": \"1.0\",\n  \"model\": {\n";
  ofs << "    \"type\": \"BPE\",\n";
  ofs << "    \"byte_level\": " << (byte_level_ ? "true" : "false") << ",\n";
  ofs << "    \"end_of_word_suffix\": \""
      << json_escape_tok (specials_.end_of_word_suffix) << "\",\n";
  ofs << "    \"vocab\": {\n";

  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      std::string_view tok = vocab_.id_to_token (
        static_cast<int32_t> (i)).value ();
      ofs << "      \"" << json_escape_tok (tok) << "\": " << i;
      if (i + 1 < vocab_.size ())
        {
          ofs << ",";
        }
      ofs << "\n";
    }
  ofs << "    },\n    \"merges\": [\n";
  for (size_t i = 0; i < merges_.size (); ++i)
    {
      std::string_view l = vocab_.id_to_token (merges_[i].left_id).value ();
      std::string_view r = vocab_.id_to_token (merges_[i].right_id).value ();
      ofs << "      \"" << json_escape_tok (l) << " "
          << json_escape_tok (r) << "\"";
      if (i + 1 < merges_.size ())
        {
          ofs << ",";
        }
      ofs << "\n";
    }
  ofs << "    ]\n  }\n}\n";

  if (!ofs.good ())
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "error writing tokenizer.json");
    }
  return {};
}

Result<BPE_Tokenizer>
BPE_Tokenizer::load_hf_tokenizer_json (const std::string &path)
{
  std::ifstream ifs (path);
  if (!ifs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open tokenizer.json for reading");
    }
  std::ostringstream ss;
  ss << ifs.rdbuf ();
  std::string content = ss.str ();

  HF_Json_Scanner sc (content);
  if (!sc.consume ('{'))
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "tokenizer.json must begin with '{'");
    }

  BPE_Tokenizer tok;
  std::vector<std::pair<int32_t, std::string>> id_tokens;
  std::vector<std::pair<std::string, std::string>> merge_pairs;
  bool found_model = false;
  bool byte_level = false;

  if (!sc.consume ('}'))
    {
      while (true)
        {
          auto root_key = sc.parse_string ();
          if (!root_key.has_value () || !sc.consume (':'))
            {
              return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                     "invalid key in tokenizer.json root");
            }
          if (*root_key == "model")
            {
              found_model = true;
              if (!sc.consume ('{'))
                {
                  return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                         "model field must be a JSON object");
                }
              if (!sc.consume ('}'))
                {
                  while (true)
                    {
                      auto mkey = sc.parse_string ();
                      if (!mkey.has_value () || !sc.consume (':'))
                        {
                          return OBSIDIAN_ERROR (
                            Error_Code::Corrupted_Format,
                            "invalid field inside model object");
                        }
                      if (*mkey == "byte_level")
                        {
                          auto bl = sc.parse_bool ();
                          if (!bl.has_value ())
                            {
                              return std::unexpected (bl.error ());
                            }
                          byte_level = *bl;
                        }
                      else if (*mkey == "end_of_word_suffix")
                        {
                          if (sc.peek () == '"')
                            {
                              auto eow = sc.parse_string ();
                              if (!eow.has_value ())
                                {
                                  return std::unexpected (eow.error ());
                                }
                              tok.specials_.end_of_word_suffix = *eow;
                            }
                          else
                            {
                              auto sv = sc.skip_value ();
                              if (!sv.has_value ())
                                {
                                  return std::unexpected (sv.error ());
                                }
                            }
                        }
                      else if (*mkey == "vocab")
                        {
                          if (!sc.consume ('{'))
                            {
                              return OBSIDIAN_ERROR (
                                Error_Code::Corrupted_Format,
                                "vocab field must be an object");
                            }
                          if (!sc.consume ('}'))
                            {
                              while (true)
                                {
                                  auto tstr = sc.parse_string ();
                                  if (!tstr.has_value () || !sc.consume (':'))
                                    {
                                      return OBSIDIAN_ERROR (
                                        Error_Code::Corrupted_Format,
                                        "invalid vocab entry");
                                    }
                                  auto tid = sc.parse_int ();
                                  if (!tid.has_value ())
                                    {
                                      return std::unexpected (tid.error ());
                                    }
                                  id_tokens.emplace_back (*tid, *tstr);
                                  if (sc.consume (','))
                                    {
                                      continue;
                                    }
                                  if (sc.consume ('}'))
                                    {
                                      break;
                                    }
                                  return OBSIDIAN_ERROR (
                                    Error_Code::Corrupted_Format,
                                    "unclosed vocab object");
                                }
                            }
                        }
                      else if (*mkey == "merges")
                        {
                          if (!sc.consume ('['))
                            {
                              return OBSIDIAN_ERROR (
                                Error_Code::Corrupted_Format,
                                "merges field must be an array");
                            }
                          if (!sc.consume (']'))
                            {
                              while (true)
                                {
                                  if (sc.peek () == '[')
                                    {
                                      /* HF v2 pair array ["left", "right"]. */
                                      sc.consume ('[');
                                      auto l = sc.parse_string ();
                                      sc.consume (',');
                                      auto r = sc.parse_string ();
                                      sc.consume (']');
                                      if (!l.has_value () || !r.has_value ())
                                        {
                                          return OBSIDIAN_ERROR (
                                            Error_Code::Corrupted_Format,
                                            "invalid merge pair array");
                                        }
                                      merge_pairs.emplace_back (*l, *r);
                                    }
                                  else
                                    {
                                      auto mline = sc.parse_string ();
                                      if (!mline.has_value ())
                                        {
                                          return std::unexpected (
                                            mline.error ());
                                        }
                                      size_t sp = mline->find (' ');
                                      if (sp == std::string::npos)
                                        {
                                          return OBSIDIAN_ERROR (
                                            Error_Code::Corrupted_Format,
                                            "merge string missing space");
                                        }
                                      merge_pairs.emplace_back (
                                        mline->substr (0, sp),
                                        mline->substr (sp + 1));
                                    }
                                  if (sc.consume (','))
                                    {
                                      continue;
                                    }
                                  if (sc.consume (']'))
                                    {
                                      break;
                                    }
                                  return OBSIDIAN_ERROR (
                                    Error_Code::Corrupted_Format,
                                    "unclosed merges array");
                                }
                            }
                        }
                      else
                        {
                          auto sv = sc.skip_value ();
                          if (!sv.has_value ())
                            {
                              return std::unexpected (sv.error ());
                            }
                        }

                      if (sc.consume (','))
                        {
                          continue;
                        }
                      if (sc.consume ('}'))
                        {
                          break;
                        }
                      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                             "unclosed model object");
                    }
                }
            }
          else
            {
              auto sv = sc.skip_value ();
              if (!sv.has_value ())
                {
                  return std::unexpected (sv.error ());
                }
            }

          if (sc.consume (','))
            {
              continue;
            }
          if (sc.consume ('}'))
            {
              break;
            }
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "unclosed root object in tokenizer.json");
        }
    }

  if (!found_model || id_tokens.empty ())
    {
      return OBSIDIAN_ERROR (
        Error_Code::Corrupted_Format,
        "tokenizer.json missing model or non-empty vocab");
    }

  std::sort (id_tokens.begin (), id_tokens.end (),
             [] (const auto &a, const auto &b)
             {
               return a.first < b.first;
             });

  tok.vocab_.clear ();
  for (const auto &kv : id_tokens)
    {
      tok.vocab_.add_token (kv.second);
    }
  tok.byte_level_ = byte_level;

  for (const auto &mp : merge_pairs)
    {
      auto r = tok.add_merge (mp.first, mp.second);
      if (!r.has_value ())
        {
          return std::unexpected (r.error ());
        }
    }

  return tok;
}

/* -------------------------------------------------------------------------
   WordPiece_Tokenizer implementation
   ------------------------------------------------------------------------- */

WordPiece_Tokenizer::WordPiece_Tokenizer (
  Special_Tokens specials,
  std::string continuing_subword_prefix,
  size_t max_input_chars_per_word)
  : specials_ (std::move (specials)),
    prefix_ (std::move (continuing_subword_prefix)),
    max_chars_ (max_input_chars_per_word)
{
  specials_.pad_id = vocab_.add_token (specials_.pad_token);
  specials_.unk_id = vocab_.add_token (specials_.unk_token);
  specials_.bos_id = vocab_.add_token (specials_.bos_token);
  specials_.eos_id = vocab_.add_token (specials_.eos_token);
  specials_.cls_id = vocab_.add_token (specials_.cls_token);
  specials_.sep_id = vocab_.add_token (specials_.sep_token);
  specials_.mask_id = vocab_.add_token (specials_.mask_token);
}

int32_t
WordPiece_Tokenizer::add_token (std::string_view token)
{
  return vocab_.add_token (token);
}

Result<std::vector<int64_t>>
WordPiece_Tokenizer::encode (std::string_view text, bool add_cls_sep) const
{
  auto utf8_ok = simd::validate_and_count_utf8 (text);
  if (!utf8_ok.has_value ())
    {
      return std::unexpected (utf8_ok.error ());
    }

  Arena &arena = thread_local_arena ();
  arena.reset ();
  Pmr_Vector<std::string_view> words (arena.resource ());
  simd::scan_pretoken_spans (text, words, true);

  std::vector<int64_t> out;
  out.reserve (words.size () * 2 + 2);
  if (add_cls_sep)
    {
      out.push_back (specials_.cls_id);
    }

  std::array<char, 256> sub_buf;
  for (std::string_view word : words)
    {
      if (word.size () > max_chars_)
        {
          out.push_back (specials_.unk_id);
          continue;
        }

      bool is_bad = false;
      size_t start = 0;
      Pmr_Vector<int64_t> sub_tokens (arena.resource ());

      while (start < word.size ())
        {
          size_t end = word.size ();
          int32_t cur_id = -1;
          while (start < end)
            {
              std::string_view slice = word.substr (start, end - start);
              if (start > 0)
                {
                  size_t total_len = prefix_.size () + slice.size ();
                  if (total_len <= sub_buf.size ())
                    {
                      std::copy (prefix_.begin (), prefix_.end (),
                                 sub_buf.begin ());
                      std::copy (slice.begin (), slice.end (),
                                 sub_buf.begin () + prefix_.size ());
                      cur_id = vocab_.token_to_id (
                        std::string_view (sub_buf.data (), total_len));
                    }
                }
              else
                {
                  cur_id = vocab_.token_to_id (slice);
                }

              if (cur_id >= 0)
                {
                  break;
                }
              --end;
            }

          if (cur_id < 0)
            {
              is_bad = true;
              break;
            }
          sub_tokens.push_back (cur_id);
          start = end;
        }

      if (is_bad)
        {
          out.push_back (specials_.unk_id);
        }
      else
        {
          out.insert (out.end (), sub_tokens.begin (), sub_tokens.end ());
        }
    }

  if (add_cls_sep)
    {
      out.push_back (specials_.sep_id);
    }
  return out;
}

Result<std::string>
WordPiece_Tokenizer::decode (std::span<const int64_t> ids,
                             bool skip_special_tokens) const
{
  std::string out;
  for (int64_t id64 : ids)
    {
      if (skip_special_tokens && is_special_id (id64, specials_))
        {
          continue;
        }
      auto tok_res = vocab_.id_to_token (static_cast<int32_t> (id64));
      if (!tok_res.has_value ())
        {
          return std::unexpected (tok_res.error ());
        }
      std::string_view tok = *tok_res;
      if (tok.size () > prefix_.size ()
          && tok.substr (0, prefix_.size ()) == prefix_)
        {
          out.append (tok.substr (prefix_.size ()));
        }
      else
        {
          if (!out.empty ())
            {
              out.push_back (' ');
            }
          out.append (tok);
        }
    }
  return out;
}

/* -------------------------------------------------------------------------
   Unigram_Tokenizer implementation
   ------------------------------------------------------------------------- */

Unigram_Tokenizer::Unigram_Tokenizer (Special_Tokens specials)
  : specials_ (std::move (specials))
{
  add_piece (specials_.pad_token, 0.0f);
  add_piece (specials_.unk_token, -20.0f);
  add_piece (specials_.bos_token, 0.0f);
  add_piece (specials_.eos_token, 0.0f);
  add_piece (specials_.cls_token, 0.0f);
  add_piece (specials_.sep_token, 0.0f);
  add_piece (specials_.mask_token, 0.0f);
}

int32_t
Unigram_Tokenizer::add_piece (std::string_view piece, float log_prob)
{
  int32_t id = vocab_.add_token (piece);
  if (static_cast<size_t> (id) >= scores_.size ())
    {
      scores_.resize (static_cast<size_t> (id) + 1, -20.0f);
    }
  scores_[static_cast<size_t> (id)] = log_prob;
  max_piece_len_ = std::max (max_piece_len_, piece.size ());
  return id;
}

Result<std::vector<int64_t>>
Unigram_Tokenizer::encode (std::string_view text,
                           bool add_bos,
                           bool add_eos) const
{
  auto utf8_ok = simd::validate_and_count_utf8 (text);
  if (!utf8_ok.has_value ())
    {
      return std::unexpected (utf8_ok.error ());
    }

  Arena &arena = thread_local_arena ();
  arena.reset ();
  Pmr_Vector<std::string_view> words (arena.resource ());
  simd::scan_pretoken_spans (text, words, true);

  std::vector<int64_t> out;
  if (add_bos)
    {
      out.push_back (specials_.bos_id);
    }

  struct Lattice_Node
  {
    float best_score = -1e30f;
    int32_t prev_pos = -1;
    int32_t token_id = -1;
  };

  for (std::string_view word : words)
    {
      const size_t n = word.size ();
      if (n == 0)
        {
          continue;
        }
      auto dp = arena.allocate_span<Lattice_Node> (n + 1);
      for (size_t i = 0; i <= n; ++i)
        {
          dp[i] = Lattice_Node {-1e30f, -1, -1};
        }
      dp[0].best_score = 0.0f;

      for (size_t i = 0; i < n; ++i)
        {
          if (dp[i].best_score <= -1e29f)
            {
              continue;
            }
          size_t max_l = std::min (max_piece_len_, n - i);
          bool matched_any = false;
          for (size_t l = 1; l <= max_l; ++l)
            {
              std::string_view sub = word.substr (i, l);
              int32_t id = vocab_.token_to_id (sub);
              if (id >= 0)
                {
                  matched_any = true;
                  float cand = dp[i].best_score
                               + scores_[static_cast<size_t> (id)];
                  if (cand > dp[i + l].best_score)
                    {
                      dp[i + l].best_score = cand;
                      dp[i + l].prev_pos = static_cast<int32_t> (i);
                      dp[i + l].token_id = id;
                    }
                }
            }
          if (!matched_any)
            {
              size_t clen = std::min (
                utf8_char_len (static_cast<unsigned char> (word[i])), n - i);
              float cand = dp[i].best_score - 20.0f;
              if (cand > dp[i + clen].best_score)
                {
                  dp[i + clen].best_score = cand;
                  dp[i + clen].prev_pos = static_cast<int32_t> (i);
                  dp[i + clen].token_id = specials_.unk_id;
                }
            }
        }

      Pmr_Vector<int64_t> word_ids (arena.resource ());
      int32_t cur = static_cast<int32_t> (n);
      while (cur > 0 && dp[static_cast<size_t> (cur)].prev_pos >= 0)
        {
          word_ids.push_back (dp[static_cast<size_t> (cur)].token_id);
          cur = dp[static_cast<size_t> (cur)].prev_pos;
        }
      for (auto it = word_ids.rbegin (); it != word_ids.rend (); ++it)
        {
          out.push_back (*it);
        }
    }

  if (add_eos)
    {
      out.push_back (specials_.eos_id);
    }
  return out;
}

Result<std::string>
Unigram_Tokenizer::decode (std::span<const int64_t> ids,
                           bool skip_special_tokens) const
{
  std::string out;
  for (int64_t id64 : ids)
    {
      if (skip_special_tokens && is_special_id (id64, specials_))
        {
          continue;
        }
      auto tok_res = vocab_.id_to_token (static_cast<int32_t> (id64));
      if (!tok_res.has_value ())
        {
          return std::unexpected (tok_res.error ());
        }
      out.append (*tok_res);
    }
  return out;
}

} /* namespace obsidian */
