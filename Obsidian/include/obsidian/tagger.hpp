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

#ifndef OBSIDIAN_TAGGER_HPP
#define OBSIDIAN_TAGGER_HPP

/**
 * @file tagger.hpp
 * @brief Log-space Viterbi decoder, HMM & Averaged Perceptron POS taggers,
 *        and BIO/BIOES Named Entity Recognition span extractor.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/memory.hpp>

namespace obsidian
{

/**
 * @brief Token paired with its Part-of-Speech or BIO sequence tag.
 */
struct OBSIDIAN_API Tagged_Token
{
  std::string token;
  std::string tag;
};

/**
 * @brief Contiguous named entity span extracted from a token sequence.
 *
 * The token index interval [start_token, end_token) is half-open.
 */
struct OBSIDIAN_API Entity_Span
{
  size_t start_token = 0;
  size_t end_token = 0;
  std::string label;
  std::string text;

  bool
  operator== (const Entity_Span &other) const noexcept
  {
    return start_token == other.start_token
           && end_token == other.end_token
           && label == other.label
           && text == other.text;
  }
};

namespace detail
{

template <typename PairLike>
std::string_view
extract_token_view (const PairLike &item) noexcept
{
  if constexpr (requires { item.token; })
    {
      return item.token;
    }
  else
    {
      return item.first;
    }
}

template <typename PairLike>
std::string_view
extract_tag_view (const PairLike &item) noexcept
{
  if constexpr (requires { item.tag; })
    {
      return item.tag;
    }
  else
    {
      return item.second;
    }
}

inline std::string
ascii_lower (std::string_view s)
{
  std::string out (s);
  for (char &ch : out)
    {
      unsigned char u = static_cast<unsigned char> (ch);
      if (u >= 'A' && u <= 'Z')
        {
          ch = static_cast<char> (u + ('a' - 'A'));
        }
    }
  return out;
}

inline bool
is_capitalized (std::string_view s) noexcept
{
  if (s.empty ())
    {
      return false;
    }
  unsigned char first = static_cast<unsigned char> (s[0]);
  return first >= 'A' && first <= 'Z';
}

inline bool
is_all_upper (std::string_view s) noexcept
{
  if (s.empty ())
    {
      return false;
    }
  bool has_upper = false;
  for (char ch : s)
    {
      unsigned char u = static_cast<unsigned char> (ch);
      if (u >= 'a' && u <= 'z')
        {
          return false;
        }
      if (u >= 'A' && u <= 'Z')
        {
          has_upper = true;
        }
    }
  return has_upper;
}

inline bool
has_digit_char (std::string_view s) noexcept
{
  for (char ch : s)
    {
      unsigned char u = static_cast<unsigned char> (ch);
      if (u >= '0' && u <= '9')
        {
          return true;
        }
    }
  return false;
}

inline bool
is_numeric_token (std::string_view s) noexcept
{
  if (s.empty ())
    {
      return false;
    }
  bool seen_digit = false;
  for (char ch : s)
    {
      unsigned char u = static_cast<unsigned char> (ch);
      if (u >= '0' && u <= '9')
        {
          seen_digit = true;
        }
      else if (ch != '.' && ch != ',' && ch != '-' && ch != ':')
        {
          return false;
        }
    }
  return seen_digit;
}

inline bool
has_hyphen_char (std::string_view s) noexcept
{
  return s.find ('-') != std::string_view::npos;
}

inline std::string
word_shape_signature (std::string_view s)
{
  std::string shape;
  shape.reserve (std::min<size_t> (s.size (), 8));
  char last = '\0';
  for (char ch : s)
    {
      unsigned char u = static_cast<unsigned char> (ch);
      char mapped = ch;
      if (u >= 'A' && u <= 'Z')
        {
          mapped = 'X';
        }
      else if (u >= 'a' && u <= 'z')
        {
          mapped = 'x';
        }
      else if (u >= '0' && u <= '9')
        {
          mapped = 'd';
        }
      if (mapped != last || shape.size () < 2)
        {
          shape.push_back (mapped);
          last = mapped;
        }
      if (shape.size () >= 8)
        {
          break;
        }
    }
  return shape;
}

} /* namespace detail */

/**
 * @brief Decode the most probable state sequence in log-space via Viterbi.
 *
 * Uses a contiguous flat backpointer trellis allocated in ARENA to avoid heap
 * allocations in the decoding hot path.
 *
 * @pre NUM_STATES > 0 when SEQ_LEN > 0.
 * @complexity O(SEQ_LEN * NUM_STATES^2) time and O(SEQ_LEN * NUM_STATES)
 *             arena space.
 */
template <typename Init_Fn, typename Trans_Fn, typename Emit_Fn>
  requires (std::invocable<Init_Fn, size_t>
            && std::invocable<Trans_Fn, size_t, size_t>
            && std::invocable<Emit_Fn, size_t, size_t>)
Result<std::vector<size_t>>
viterbi_decode (size_t seq_len, size_t num_states,
                Init_Fn &&init_score,
                Trans_Fn &&trans_score,
                Emit_Fn &&emit_score,
                Arena &arena)
{
  if (seq_len == 0)
    {
      return std::vector<size_t>{};
    }
  if (num_states == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "viterbi_decode: num_states must be > 0");
    }

  constexpr float k_neg_inf = -1.0e30f;

  std::span<float> dp_prev = arena.allocate_span<float> (num_states);
  std::span<float> dp_curr = arena.allocate_span<float> (num_states);
  std::span<uint32_t> backptr =
    arena.allocate_span<uint32_t> (seq_len * num_states);

  for (size_t s = 0; s < num_states; ++s)
    {
      dp_prev[s] = static_cast<float> (init_score (s))
                   + static_cast<float> (emit_score (0, s));
      backptr[s] = 0;
    }

  for (size_t t = 1; t < seq_len; ++t)
    {
      uint32_t *bp_row = backptr.data () + t * num_states;
      for (size_t curr = 0; curr < num_states; ++curr)
        {
          float best_val = k_neg_inf;
          uint32_t best_prev = 0;
          float emit = static_cast<float> (emit_score (t, curr));

          for (size_t prev = 0; prev < num_states; ++prev)
            {
              float cand = dp_prev[prev]
                           + static_cast<float> (trans_score (prev, curr));
              if (cand > best_val)
                {
                  best_val = cand;
                  best_prev = static_cast<uint32_t> (prev);
                }
            }
          dp_curr[curr] = best_val + emit;
          bp_row[curr] = best_prev;
        }
      std::swap (dp_prev, dp_curr);
    }

  size_t best_final = 0;
  float best_final_score = dp_prev[0];
  for (size_t s = 1; s < num_states; ++s)
    {
      if (dp_prev[s] > best_final_score)
        {
          best_final_score = dp_prev[s];
          best_final = s;
        }
    }

  std::vector<size_t> path (seq_len);
  path[seq_len - 1] = best_final;
  for (size_t t = seq_len - 1; t > 0; --t)
    {
      path[t - 1] = backptr[t * num_states + path[t]];
    }
  return path;
}

/**
 * @brief Overload of viterbi_decode with an internal stack-backed Arena.
 * @pre NUM_STATES > 0 when SEQ_LEN > 0.
 * @complexity O(SEQ_LEN * NUM_STATES^2) time and O(SEQ_LEN * NUM_STATES)
 *             space.
 */
template <typename Init_Fn, typename Trans_Fn, typename Emit_Fn>
  requires (std::invocable<Init_Fn, size_t>
            && std::invocable<Trans_Fn, size_t, size_t>
            && std::invocable<Emit_Fn, size_t, size_t>)
Result<std::vector<size_t>>
viterbi_decode (size_t seq_len, size_t num_states,
                Init_Fn &&init_score,
                Trans_Fn &&trans_score,
                Emit_Fn &&emit_score)
{
  Arena local_arena;
  return viterbi_decode (seq_len, num_states,
                         std::forward<Init_Fn> (init_score),
                         std::forward<Trans_Fn> (trans_score),
                         std::forward<Emit_Fn> (emit_score),
                         local_arena);
}

/**
 * @brief Overload of viterbi_decode with uniform initial state scores.
 * @pre NUM_STATES > 0 when SEQ_LEN > 0.
 * @complexity O(SEQ_LEN * NUM_STATES^2) time and O(SEQ_LEN * NUM_STATES)
 *             arena space.
 */
template <typename Trans_Fn, typename Emit_Fn>
  requires (std::invocable<Trans_Fn, size_t, size_t>
            && std::invocable<Emit_Fn, size_t, size_t>)
Result<std::vector<size_t>>
viterbi_decode (size_t seq_len, size_t num_states,
                Trans_Fn &&trans_score,
                Emit_Fn &&emit_score,
                Arena &arena)
{
  auto zero_init = [] (size_t) noexcept -> float { return 0.0f; };
  return viterbi_decode (seq_len, num_states, zero_init,
                         std::forward<Trans_Fn> (trans_score),
                         std::forward<Emit_Fn> (emit_score),
                         arena);
}

/**
 * @brief Overload of viterbi_decode with uniform initial state scores and
 *        internal stack-backed Arena.
 * @pre NUM_STATES > 0 when SEQ_LEN > 0.
 * @complexity O(SEQ_LEN * NUM_STATES^2) time and O(SEQ_LEN * NUM_STATES)
 *             space.
 */
template <typename Trans_Fn, typename Emit_Fn>
  requires (std::invocable<Trans_Fn, size_t, size_t>
            && std::invocable<Emit_Fn, size_t, size_t>)
Result<std::vector<size_t>>
viterbi_decode (size_t seq_len, size_t num_states,
                Trans_Fn &&trans_score,
                Emit_Fn &&emit_score)
{
  Arena local_arena;
  auto zero_init = [] (size_t) noexcept -> float { return 0.0f; };
  return viterbi_decode (seq_len, num_states, zero_init,
                         std::forward<Trans_Fn> (trans_score),
                         std::forward<Emit_Fn> (emit_score),
                         local_arena);
}

/**
 * @brief First-order Hidden Markov Model Part-of-Speech tagger.
 *
 * Trains log-probabilities with add-k smoothing and decodes tag sequences via
 * viterbi_decode, using orthographic suffix, capitalization, numeric, and
 * hyphenation backoff distributions for out-of-vocabulary tokens.
 */
class OBSIDIAN_API HMM_POS_Tagger
{
public:
  /**
   * @brief Default-construct an untrained HMM_POS_Tagger.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  HMM_POS_Tagger (void) = default;

  /**
   * @brief Train HMM transition, emission, and OOV suffix backoff parameters.
   * @pre SENTENCES must be non-empty and SMOOTHING_K > 0.
   * @complexity O(W + S^2 + V * S) time where W is total corpus tokens, S is
   *             number of POS tags, and V is vocabulary size.
   */
  template <typename SentenceRange>
  Result<void>
  train (const SentenceRange &sentences, float smoothing_k = 0.01f)
  {
    if (smoothing_k <= 0.0f)
      {
        return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                               "HMM_POS_Tagger::train: smoothing_k <= 0");
      }

    tags_.clear ();
    tag_to_id_.clear ();
    word_to_id_.clear ();
    log_init_.clear ();
    log_trans_.clear ();
    log_emit_.clear ();
    log_oov_base_.clear ();
    suffix_log_prob_.clear ();

    size_t non_empty_sentences = 0;
    for (const auto &sent : sentences)
      {
        if (sent.empty ())
          {
            continue;
          }
        ++non_empty_sentences;
        for (const auto &pair : sent)
          {
            std::string_view w = detail::extract_token_view (pair);
            std::string_view t = detail::extract_tag_view (pair);
            if (w.empty () || t.empty ())
              {
                continue;
              }
            std::string lw = detail::ascii_lower (w);
            if (word_to_id_.find (lw) == word_to_id_.end ())
              {
                size_t wid = word_to_id_.size ();
                word_to_id_.emplace (std::move (lw), wid);
              }
            if (tag_to_id_.find (std::string (t)) == tag_to_id_.end ())
              {
                size_t tid = tags_.size ();
                tags_.emplace_back (t);
                tag_to_id_.emplace (tags_.back (), tid);
              }
          }
      }

    if (non_empty_sentences == 0 || tags_.empty () || word_to_id_.empty ())
      {
        return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                               "HMM_POS_Tagger::train: empty training corpus");
      }

    const size_t num_s = tags_.size ();
    const size_t num_v = word_to_id_.size ();
    smoothing_k_ = smoothing_k;

    std::vector<float> init_counts (num_s, 0.0f);
    std::vector<float> tag_counts (num_s, 0.0f);
    std::vector<float> trans_counts (num_s * num_s, 0.0f);
    std::vector<float> trans_row_totals (num_s, 0.0f);
    std::vector<float> emit_counts (num_s * num_v, 0.0f);

    std::vector<float> cap_counts (num_s, 0.0f);
    std::vector<float> num_counts (num_s, 0.0f);
    std::vector<float> hyp_counts (num_s, 0.0f);
    std::unordered_map<std::string, std::vector<float>> suf_counts;

    for (const auto &sent : sentences)
      {
        if (sent.empty ())
          {
            continue;
          }
        size_t prev_tid = num_s;
        size_t pos_idx = 0;
        for (const auto &pair : sent)
          {
            std::string_view w = detail::extract_token_view (pair);
            std::string_view t = detail::extract_tag_view (pair);
            if (w.empty () || t.empty ())
              {
                continue;
              }
            std::string lw = detail::ascii_lower (w);
            size_t wid = word_to_id_.at (lw);
            size_t tid = tag_to_id_.at (std::string (t));

            if (pos_idx == 0)
              {
                init_counts[tid] += 1.0f;
              }
            else if (prev_tid < num_s)
              {
                trans_counts[prev_tid * num_s + tid] += 1.0f;
                trans_row_totals[prev_tid] += 1.0f;
              }

            tag_counts[tid] += 1.0f;
            emit_counts[tid * num_v + wid] += 1.0f;

            if (detail::is_capitalized (w))
              {
                cap_counts[tid] += 1.0f;
              }
            if (detail::has_digit_char (w))
              {
                num_counts[tid] += 1.0f;
              }
            if (detail::has_hyphen_char (w))
              {
                hyp_counts[tid] += 1.0f;
              }

            for (size_t slen = 1; slen <= 3; ++slen)
              {
                if (lw.size () > slen)
                  {
                    std::string suf = lw.substr (lw.size () - slen);
                    auto &vec = suf_counts[suf];
                    if (vec.empty ())
                      {
                        vec.assign (num_s, 0.0f);
                      }
                    vec[tid] += 1.0f;
                  }
              }

            prev_tid = tid;
            ++pos_idx;
          }
      }

    log_init_.resize (num_s);
    float init_denom = static_cast<float> (non_empty_sentences)
                       + smoothing_k * static_cast<float> (num_s);
    for (size_t s = 0; s < num_s; ++s)
      {
        log_init_[s] =
          std::log ((init_counts[s] + smoothing_k) / init_denom);
      }

    log_trans_.resize (num_s * num_s);
    for (size_t prev = 0; prev < num_s; ++prev)
      {
        float denom = trans_row_totals[prev]
                      + smoothing_k * static_cast<float> (num_s);
        for (size_t curr = 0; curr < num_s; ++curr)
          {
            float c = trans_counts[prev * num_s + curr];
            log_trans_[prev * num_s + curr] =
              std::log ((c + smoothing_k) / denom);
          }
      }

    log_emit_.resize (num_s * num_v);
    log_oov_base_.resize (num_s);
    log_cap_bonus_.resize (num_s);
    log_num_bonus_.resize (num_s);
    log_hyp_bonus_.resize (num_s);

    for (size_t s = 0; s < num_s; ++s)
      {
        float denom = tag_counts[s]
                      + smoothing_k * static_cast<float> (num_v + 1);
        for (size_t v = 0; v < num_v; ++v)
          {
            float c = emit_counts[s * num_v + v];
            log_emit_[s * num_v + v] = std::log ((c + smoothing_k) / denom);
          }
        log_oov_base_[s] = std::log (smoothing_k / denom);

        float tag_total = tag_counts[s] + 2.0f * smoothing_k;
        log_cap_bonus_[s] =
          std::log ((cap_counts[s] + smoothing_k) / tag_total);
        log_num_bonus_[s] =
          std::log ((num_counts[s] + smoothing_k) / tag_total);
        log_hyp_bonus_[s] =
          std::log ((hyp_counts[s] + smoothing_k) / tag_total);
      }

    for (auto &[suf, vec] : suf_counts)
      {
        std::vector<float> lprobs (num_s);
        for (size_t s = 0; s < num_s; ++s)
          {
            float denom = tag_counts[s] + smoothing_k * 10.0f;
            lprobs[s] = std::log ((vec[s] + smoothing_k) / denom);
          }
        suffix_log_prob_.emplace (suf, std::move (lprobs));
      }

    return {};
  }

  /**
   * @brief Decode the optimal POS tag sequence for TOKENS via Viterbi.
   * @pre The tagger must have been trained via train().
   * @complexity O(T * S^2) time and O(T * S) space for T tokens and S tags.
   */
  Result<std::vector<std::string>>
  tag (std::span<const std::string_view> tokens) const
  {
    if (tags_.empty ())
      {
        return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                               "HMM_POS_Tagger::tag: model is not trained");
      }
    if (tokens.empty ())
      {
        return std::vector<std::string>{};
      }

    const size_t seq_len = tokens.size ();
    const size_t num_s = tags_.size ();
    const size_t num_v = word_to_id_.size ();

    std::vector<int64_t> token_wids (seq_len, -1);
    std::vector<std::string> lower_tokens (seq_len);
    for (size_t t = 0; t < seq_len; ++t)
      {
        lower_tokens[t] = detail::ascii_lower (tokens[t]);
        auto it = word_to_id_.find (lower_tokens[t]);
        if (it != word_to_id_.end ())
          {
            token_wids[t] = static_cast<int64_t> (it->second);
          }
      }

    auto init_fn = [this] (size_t s) noexcept -> float
      {
        return log_init_[s];
      };
    auto trans_fn = [this, num_s] (size_t prev, size_t curr) noexcept -> float
      {
        return log_trans_[prev * num_s + curr];
      };
    auto emit_fn = [this, num_v, &tokens, &lower_tokens, &token_wids] (
                     size_t t, size_t s) -> float
      {
        int64_t wid = token_wids[t];
        if (wid >= 0)
          {
            return log_emit_[s * num_v + static_cast<size_t> (wid)];
          }
        return compute_oov_emission (tokens[t], lower_tokens[t], s);
      };

    Arena local_arena;
    auto decoded = viterbi_decode (seq_len, num_s, init_fn, trans_fn, emit_fn,
                                   local_arena);
    if (!decoded.has_value ())
      {
        return std::unexpected<Error> (decoded.error ());
      }

    std::vector<std::string> out;
    out.reserve (seq_len);
    for (size_t sid : decoded.value ())
      {
        out.push_back (tags_[sid]);
      }
    return out;
  }

  /**
   * @brief Overload of tag() accepting a span of std::string tokens.
   * @pre The tagger must have been trained via train().
   * @complexity O(T * S^2) time and O(T * S) space.
   */
  Result<std::vector<std::string>>
  tag (std::span<const std::string> tokens) const
  {
    std::vector<std::string_view> views;
    views.reserve (tokens.size ());
    for (const auto &tok : tokens)
      {
        views.emplace_back (tok);
      }
    return tag (std::span<const std::string_view> (views));
  }

  /**
   * @brief Return the number of distinct POS tags learned during training.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  num_states (void) const noexcept
  {
    return tags_.size ();
  }

  /**
   * @brief Return the number of unique lowercase words in the vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return word_to_id_.size ();
  }

private:
  std::vector<std::string> tags_;
  std::unordered_map<std::string, size_t> tag_to_id_;
  std::unordered_map<std::string, size_t> word_to_id_;
  std::vector<float> log_init_;
  std::vector<float> log_trans_;
  std::vector<float> log_emit_;
  std::vector<float> log_oov_base_;
  std::vector<float> log_cap_bonus_;
  std::vector<float> log_num_bonus_;
  std::vector<float> log_hyp_bonus_;
  std::unordered_map<std::string, std::vector<float>> suffix_log_prob_;
  float smoothing_k_ = 0.01f;

  float
  compute_oov_emission (std::string_view raw_tok,
                        std::string_view lower_tok,
                        size_t state_id) const
  {
    float score = log_oov_base_[state_id];
    if (detail::is_capitalized (raw_tok))
      {
        score += 0.5f * log_cap_bonus_[state_id];
      }
    if (detail::has_digit_char (raw_tok))
      {
        score += 0.75f * log_num_bonus_[state_id];
      }
    if (detail::has_hyphen_char (raw_tok))
      {
        score += 0.5f * log_hyp_bonus_[state_id];
      }

    for (size_t slen = 3; slen >= 1; --slen)
      {
        if (lower_tok.size () > slen)
          {
            std::string suf (lower_tok.substr (lower_tok.size () - slen));
            auto it = suffix_log_prob_.find (suf);
            if (it != suffix_log_prob_.end ())
              {
                score += 0.8f * it->second[state_id];
                break;
              }
          }
      }
    return score;
  }
};

/**
 * @brief Alias for HMM_POS_Tagger matching ARCHITECTURE.md / TASKS.md naming.
 */
using HMM_Tagger = HMM_POS_Tagger;

/**
 * @brief Averaged Structured Perceptron POS tagger (Collins 2002 / Honnibal).
 *
 * Extracts rich orthographic prefix, suffix, word shape, lexical window, and
 * previous-tag history features, performing online passive/perceptron updates
 * with cumulative parameter averaging for strong generalization.
 */
class OBSIDIAN_API Perceptron_POS_Tagger
{
public:
  /**
   * @brief Default-construct an untrained Perceptron_POS_Tagger.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Perceptron_POS_Tagger (void) = default;

  /**
   * @brief Train the averaged perceptron POS tagger over SENTENCES.
   * @pre SENTENCES must be non-empty, EPOCHS >= 1, and LEARNING_RATE > 0.
   * @complexity O(EPOCHS * W * F * C) time where W is corpus token count,
   *             F is active features per token (~18), and C is tag count.
   */
  template <typename SentenceRange>
  Result<void>
  train (const SentenceRange &sentences,
         size_t epochs = 5,
         float learning_rate = 1.0f)
  {
    if (epochs == 0 || learning_rate <= 0.0f)
      {
        return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                               "Perceptron_POS_Tagger::train: invalid args");
      }

    classes_.clear ();
    class_to_id_.clear ();
    weights_.clear ();
    totals_.clear ();
    timestamps_.clear ();
    step_ = 0;

    for (const auto &sent : sentences)
      {
        for (const auto &pair : sent)
          {
            std::string_view t = detail::extract_tag_view (pair);
            if (!t.empty ()
                && class_to_id_.find (std::string (t)) == class_to_id_.end ())
              {
                size_t cid = classes_.size ();
                classes_.emplace_back (t);
                class_to_id_.emplace (classes_.back (), cid);
              }
          }
      }

    if (classes_.empty ())
      {
        return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                               "Perceptron_POS_Tagger::train: empty corpus");
      }

    for (size_t ep = 0; ep < epochs; ++ep)
      {
        for (const auto &sent : sentences)
          {
            if (sent.empty ())
              {
                continue;
              }
            std::vector<std::string_view> tok_views;
            std::vector<std::string_view> tag_views;
            tok_views.reserve (sent.size ());
            tag_views.reserve (sent.size ());
            for (const auto &pair : sent)
              {
                std::string_view w = detail::extract_token_view (pair);
                std::string_view t = detail::extract_tag_view (pair);
                if (!w.empty () && !t.empty ())
                  {
                    tok_views.push_back (w);
                    tag_views.push_back (t);
                  }
              }
            if (!tok_views.empty ())
              {
                auto res = update_sentence (tok_views, tag_views,
                                            learning_rate);
                if (!res.has_value ())
                  {
                    return std::unexpected<Error> (res.error ());
                  }
              }
          }
      }

    finalize_averages ();
    return {};
  }

  /**
   * @brief Perform an online perceptron update on a single tagged sentence.
   * @pre TOKENS.size() == GOLD_TAGS.size().
   * @complexity O(T * F * C) time in sentence length T and class count C.
   */
  Result<size_t>
  update_sentence (std::span<const std::string_view> tokens,
                   std::span<const std::string_view> gold_tags,
                   float learning_rate = 1.0f)
  {
    if (tokens.size () != gold_tags.size ())
      {
        return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                               "Perceptron_POS_Tagger: token/tag mismatch");
      }
    if (tokens.empty ())
      {
        return 0;
      }

    for (std::string_view gt : gold_tags)
      {
        if (class_to_id_.find (std::string (gt)) == class_to_id_.end ())
          {
            size_t new_id = classes_.size ();
            classes_.emplace_back (gt);
            class_to_id_.emplace (classes_.back (), new_id);
            for (auto &[feat, wvec] : weights_)
              {
                wvec.resize (classes_.size (), 0.0f);
                totals_[feat].resize (classes_.size (), 0.0f);
                timestamps_[feat].resize (classes_.size (), step_);
              }
          }
      }

    std::string prev_tag = "-START-";
    std::string prev2_tag = "-START2-";
    size_t num_correct = 0;

    std::vector<std::string> features;
    features.reserve (24);

    for (size_t i = 0; i < tokens.size (); ++i)
      {
        ++step_;
        extract_features (tokens, i, prev_tag, prev2_tag, features);
        size_t guess_id = predict_class (features);
        size_t gold_id = class_to_id_.at (std::string (gold_tags[i]));

        if (guess_id != gold_id)
          {
            for (const std::string &f : features)
              {
                adjust_feature (f, gold_id, +learning_rate);
                adjust_feature (f, guess_id, -learning_rate);
              }
          }
        else
          {
            ++num_correct;
          }

        prev2_tag = prev_tag;
        prev_tag = classes_[guess_id];
      }

    return num_correct;
  }

  /**
   * @brief Finalize Collins-style averaged parameter weights after training.
   * @pre step_ > 0.
   * @complexity O(|F| * C) time over all active features |F| and classes C.
   */
  void
  finalize_averages (void)
  {
    if (step_ == 0)
      {
        return;
      }
    const float inv_step = 1.0f / static_cast<float> (step_);
    const size_t num_c = classes_.size ();

    for (auto &[feat, wvec] : weights_)
      {
        auto &tot = totals_[feat];
        auto &ts = timestamps_[feat];
        wvec.resize (num_c, 0.0f);
        tot.resize (num_c, 0.0f);
        ts.resize (num_c, step_);

        for (size_t c = 0; c < num_c; ++c)
          {
            size_t elapsed = step_ - ts[c];
            tot[c] += static_cast<float> (elapsed) * wvec[c];
            ts[c] = step_;
            wvec[c] = tot[c] * inv_step;
          }
      }
  }

  /**
   * @brief Predict the POS tag sequence for TOKENS.
   * @pre The tagger must have been trained via train() or update_sentence().
   * @complexity O(T * F * C) time and O(T) space for T tokens.
   */
  Result<std::vector<std::string>>
  tag (std::span<const std::string_view> tokens) const
  {
    if (classes_.empty ())
      {
        return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                               "Perceptron_POS_Tagger::tag: untrained model");
      }
    if (tokens.empty ())
      {
        return std::vector<std::string>{};
      }

    std::vector<std::string> out;
    out.reserve (tokens.size ());

    std::string prev_tag = "-START-";
    std::string prev2_tag = "-START2-";
    std::vector<std::string> features;
    features.reserve (24);

    for (size_t i = 0; i < tokens.size (); ++i)
      {
        extract_features (tokens, i, prev_tag, prev2_tag, features);
        size_t best_cid = predict_class (features);
        out.push_back (classes_[best_cid]);
        prev2_tag = prev_tag;
        prev_tag = out.back ();
      }
    return out;
  }

  /**
   * @brief Overload of tag() accepting a span of std::string tokens.
   * @pre The tagger must have been trained via train() or update_sentence().
   * @complexity O(T * F * C) time and O(T) space.
   */
  Result<std::vector<std::string>>
  tag (std::span<const std::string> tokens) const
  {
    std::vector<std::string_view> views;
    views.reserve (tokens.size ());
    for (const auto &tok : tokens)
      {
        views.emplace_back (tok);
      }
    return tag (std::span<const std::string_view> (views));
  }

  /**
   * @brief Return the number of target POS classes.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  num_classes (void) const noexcept
  {
    return classes_.size ();
  }

private:
  std::vector<std::string> classes_;
  std::unordered_map<std::string, size_t> class_to_id_;
  std::unordered_map<std::string, std::vector<float>> weights_;
  std::unordered_map<std::string, std::vector<float>> totals_;
  std::unordered_map<std::string, std::vector<size_t>> timestamps_;
  size_t step_ = 0;

  void
  adjust_feature (const std::string &feat, size_t class_id, float delta)
  {
    const size_t num_c = classes_.size ();
    auto &wvec = weights_[feat];
    if (wvec.size () < num_c)
      {
        wvec.resize (num_c, 0.0f);
        totals_[feat].resize (num_c, 0.0f);
        timestamps_[feat].resize (num_c, step_);
      }
    auto &tot = totals_[feat];
    auto &ts = timestamps_[feat];
    size_t elapsed = step_ - ts[class_id];
    tot[class_id] += static_cast<float> (elapsed) * wvec[class_id];
    wvec[class_id] += delta;
    ts[class_id] = step_;
  }

  size_t
  predict_class (const std::vector<std::string> &features) const
  {
    const size_t num_c = classes_.size ();
    std::vector<float> scores (num_c, 0.0f);
    for (const std::string &f : features)
      {
        auto it = weights_.find (f);
        if (it == weights_.end ())
          {
            continue;
          }
        const auto &wvec = it->second;
        const size_t limit = std::min (num_c, wvec.size ());
        for (size_t c = 0; c < limit; ++c)
          {
            scores[c] += wvec[c];
          }
      }
    size_t best_c = 0;
    float best_score = scores[0];
    for (size_t c = 1; c < num_c; ++c)
      {
        if (scores[c] > best_score)
          {
            best_score = scores[c];
            best_c = c;
          }
      }
    return best_c;
  }

  static void
  extract_features (std::span<const std::string_view> tokens,
                    size_t idx,
                    std::string_view prev_tag,
                    std::string_view prev2_tag,
                    std::vector<std::string> &out)
  {
    out.clear ();
    std::string_view raw = tokens[idx];
    std::string w0 = detail::ascii_lower (raw);
    std::string w_prev =
      (idx >= 1) ? detail::ascii_lower (tokens[idx - 1]) : "-BOS-";
    std::string w_prev2 =
      (idx >= 2) ? detail::ascii_lower (tokens[idx - 2]) : "-BOS2-";
    std::string w_next =
      (idx + 1 < tokens.size ()) ? detail::ascii_lower (tokens[idx + 1])
                                 : "-EOS-";
    std::string w_next2 =
      (idx + 2 < tokens.size ()) ? detail::ascii_lower (tokens[idx + 2])
                                 : "-EOS2-";

    out.emplace_back ("bias");
    out.push_back ("w0=" + w0);
    out.push_back ("w-1=" + w_prev);
    out.push_back ("w-2=" + w_prev2);
    out.push_back ("w+1=" + w_next);
    out.push_back ("w+2=" + w_next2);
    out.push_back ("t-1=" + std::string (prev_tag));
    out.push_back ("t-2=" + std::string (prev2_tag));
    out.push_back ("t-2,t-1=" + std::string (prev2_tag) + "/"
                   + std::string (prev_tag));
    out.push_back ("t-1,w0=" + std::string (prev_tag) + "/" + w0);

    for (size_t k = 1; k <= 3; ++k)
      {
        if (w0.size () >= k)
          {
            out.push_back ("pref" + std::to_string (k) + "="
                           + w0.substr (0, k));
            out.push_back ("suf" + std::to_string (k) + "="
                           + w0.substr (w0.size () - k));
          }
      }

    if (w_prev.size () >= 3)
      {
        out.push_back ("suf-1=" + w_prev.substr (w_prev.size () - 3));
      }
    if (w_next.size () >= 3)
      {
        out.push_back ("suf+1=" + w_next.substr (w_next.size () - 3));
      }

    out.push_back ("shape=" + detail::word_shape_signature (raw));
    if (detail::is_capitalized (raw))
      {
        out.emplace_back ("is_cap=1");
        if (idx > 0)
          {
            out.emplace_back ("mid_cap=1");
          }
      }
    if (detail::is_all_upper (raw))
      {
        out.emplace_back ("is_upper=1");
      }
    if (detail::has_digit_char (raw))
      {
        out.emplace_back ("has_digit=1");
      }
    if (detail::is_numeric_token (raw))
      {
        out.emplace_back ("is_num=1");
      }
    if (detail::has_hyphen_char (raw))
      {
        out.emplace_back ("has_hyp=1");
      }
  }
};

/**
 * @brief Alias for Perceptron_POS_Tagger matching ARCHITECTURE.md naming.
 */
using Perceptron_Tagger = Perceptron_POS_Tagger;

/**
 * @brief BIO / BIOES sequence chunker and Gazetteer + Viterbi NER tagger.
 *
 * Combines constrained Viterbi BIO sequence decoding, multi-token gazetteer
 * phrase matching, and orthographic capitalization heuristics to extract
 * structured Entity_Span objects.
 */
class OBSIDIAN_API NER_Span_Extractor
{
public:
  /**
   * @brief Default-construct an NER_Span_Extractor with default entity labels.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  NER_Span_Extractor (void)
  {
    ensure_bio_state ("O");
    ensure_entity_label ("PER");
    ensure_entity_label ("ORG");
    ensure_entity_label ("LOC");
    ensure_entity_label ("MISC");
  }

  /**
   * @brief Decode a BIO or BIOES tag sequence into contiguous Entity_Span
   *        records.
   * @pre TOKENS.size() == BIO_TAGS.size().
   * @complexity O(T) time and O(E) space for T tokens and E extracted spans.
   */
  static Result<std::vector<Entity_Span>>
  decode_bio_spans (std::span<const std::string_view> tokens,
                    std::span<const std::string_view> bio_tags)
  {
    if (tokens.size () != bio_tags.size ())
      {
        return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                               "decode_bio_spans: length mismatch");
      }

    std::vector<Entity_Span> spans;
    bool active = false;
    size_t cur_start = 0;
    std::string cur_label;

    auto flush_current = [&] (size_t end_idx) {
      if (!active)
        {
          return;
        }
      std::string joined;
      for (size_t k = cur_start; k < end_idx; ++k)
        {
          if (k > cur_start)
            {
              joined.push_back (' ');
            }
          joined.append (tokens[k]);
        }
      spans.push_back (Entity_Span{cur_start, end_idx, cur_label,
                                   std::move (joined)});
      active = false;
      cur_label.clear ();
    };

    for (size_t i = 0; i < tokens.size (); ++i)
      {
        std::string_view tag = bio_tags[i];
        if (tag.empty () || tag == "O")
          {
            flush_current (i);
            continue;
          }

        char prefix = tag[0];
        std::string_view label = tag;
        if (tag.size () > 2 && tag[1] == '-')
          {
            label = tag.substr (2);
          }
        else
          {
            prefix = 'B';
          }

        if (prefix == 'S' || prefix == 'U')
          {
            flush_current (i);
            cur_start = i;
            cur_label = std::string (label);
            active = true;
            flush_current (i + 1);
          }
        else if (prefix == 'B')
          {
            flush_current (i);
            cur_start = i;
            cur_label = std::string (label);
            active = true;
          }
        else if (prefix == 'I')
          {
            if (active && cur_label == label)
              {
                /* Continue the current span.  */
              }
            else
              {
                flush_current (i);
                cur_start = i;
                cur_label = std::string (label);
                active = true;
              }
          }
        else if (prefix == 'E' || prefix == 'L')
          {
            if (!active || cur_label != label)
              {
                flush_current (i);
                cur_start = i;
                cur_label = std::string (label);
                active = true;
              }
            flush_current (i + 1);
          }
        else
          {
            flush_current (i);
            cur_start = i;
            cur_label = std::string (label);
            active = true;
          }
      }

    flush_current (tokens.size ());
    return spans;
  }

  /**
   * @brief Overload of decode_bio_spans accepting std::string spans.
   * @pre TOKENS.size() == BIO_TAGS.size().
   * @complexity O(T) time and O(E) space.
   */
  static Result<std::vector<Entity_Span>>
  decode_bio_spans (std::span<const std::string> tokens,
                    std::span<const std::string> bio_tags)
  {
    std::vector<std::string_view> tok_views;
    std::vector<std::string_view> tag_views;
    tok_views.reserve (tokens.size ());
    tag_views.reserve (bio_tags.size ());
    for (const auto &w : tokens)
      {
        tok_views.emplace_back (w);
      }
    for (const auto &t : bio_tags)
      {
        tag_views.emplace_back (t);
      }
    return decode_bio_spans (tok_views, tag_views);
  }

  /**
   * @brief Encode Entity_Span intervals into BIO or BIOES tag strings.
   * @pre Every span in SPANS must satisfy start_token < end_token <=
   *      NUM_TOKENS.
   * @complexity O(NUM_TOKENS + SPANS.size()) time and O(NUM_TOKENS) space.
   */
  static Result<std::vector<std::string>>
  encode_bio_tags (size_t num_tokens,
                   std::span<const Entity_Span> spans,
                   bool use_bioes = false)
  {
    std::vector<std::string> tags (num_tokens, "O");
    for (const Entity_Span &sp : spans)
      {
        if (sp.start_token >= sp.end_token || sp.end_token > num_tokens
            || sp.label.empty ())
          {
            return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                                   "encode_bio_tags: invalid Entity_Span");
          }
        size_t len = sp.end_token - sp.start_token;
        if (use_bioes && len == 1)
          {
            tags[sp.start_token] = "S-" + sp.label;
          }
        else
          {
            tags[sp.start_token] = "B-" + sp.label;
            for (size_t i = sp.start_token + 1; i + 1 < sp.end_token; ++i)
              {
                tags[i] = "I-" + sp.label;
              }
            if (len > 1)
              {
                tags[sp.end_token - 1] =
                  (use_bioes ? "E-" : "I-") + sp.label;
              }
          }
      }
    return tags;
  }

  /**
   * @brief Register a single-token or multi-token gazetteer entry for LABEL.
   * @pre PHRASE and LABEL must be non-empty.
   * @complexity O(K) time where K is the number of tokens in PHRASE.
   */
  void
  add_gazetteer_entry (std::string_view phrase, std::string_view label)
  {
    if (phrase.empty () || label.empty ())
      {
        return;
      }
    ensure_entity_label (label);

    std::vector<std::string> words;
    size_t i = 0;
    while (i < phrase.size ())
      {
        while (i < phrase.size () && phrase[i] == ' ')
          {
            ++i;
          }
        size_t start = i;
        while (i < phrase.size () && phrase[i] != ' ')
          {
            ++i;
          }
        if (i > start)
          {
            words.push_back (
              detail::ascii_lower (phrase.substr (start, i - start)));
          }
      }
    if (words.empty ())
      {
        return;
      }
    gazetteer_.push_back (Gazetteer_Entry{std::move (words),
                                          std::string (label)});
  }

  /**
   * @brief Register a multi-token gazetteer entry from a list of tokens.
   * @pre TOKENS and LABEL must be non-empty.
   * @complexity O(K) time where K is the number of tokens.
   */
  void
  add_gazetteer_entry (std::initializer_list<std::string_view> tokens,
                       std::string_view label)
  {
    std::string joined;
    for (std::string_view t : tokens)
      {
        if (!joined.empty ())
          {
            joined.push_back (' ');
          }
        joined.append (t);
      }
    add_gazetteer_entry (joined, label);
  }

  /**
   * @brief Train the underlying BIO Viterbi sequence model on BIO-tagged data.
   * @pre SENTENCES must be non-empty and SMOOTHING_K > 0.
   * @complexity O(W + S^2) time where W is corpus tokens and S is BIO states.
   */
  template <typename SentenceRange>
  Result<void>
  train (const SentenceRange &sentences, float smoothing_k = 0.01f)
  {
    for (const auto &sent : sentences)
      {
        for (const auto &pair : sent)
          {
            std::string_view t = detail::extract_tag_view (pair);
            if (t.size () > 2 && t[1] == '-')
              {
                ensure_entity_label (t.substr (2));
              }
            else if (!t.empty ())
              {
                ensure_bio_state (t);
              }
          }
      }

    auto res = hmm_.train (sentences, smoothing_k);
    if (!res.has_value ())
      {
        return res;
      }
    trained_ = true;
    return {};
  }

  /**
   * @brief Predict BIO tags for TOKENS using Viterbi, gazetteer, and rules.
   * @pre None (can operate with gazetteer/rules alone or trained HMM + rules).
   * @complexity O(T * S^2 + G * T) time for T tokens, S states, G entries.
   */
  Result<std::vector<std::string>>
  tag_bio (std::span<const std::string_view> tokens) const
  {
    if (tokens.empty ())
      {
        return std::vector<std::string>{};
      }

    const size_t seq_len = tokens.size ();
    std::vector<std::string> bio_tags (seq_len, "O");

    if (trained_)
      {
        auto hmm_res = hmm_.tag (tokens);
        if (hmm_res.has_value ())
          {
            bio_tags = std::move (hmm_res.value ());
          }
      }
    else
      {
        /* Decode with rule-based Viterbi potentials over BIO states.  */
        const size_t num_s = states_.size ();
        auto init_fn = [this] (size_t s) noexcept -> float
          {
            std::string_view st = states_[s];
            if (st.size () > 2 && st[0] == 'I' && st[1] == '-')
              {
                return -100.0f;
              }
            return (st == "O") ? 0.0f : -0.5f;
          };
        auto trans_fn = [this] (size_t prev, size_t curr) noexcept -> float
          {
            std::string_view sp = states_[prev];
            std::string_view sc = states_[curr];
            if (sc.size () > 2 && sc[0] == 'I' && sc[1] == '-')
              {
                std::string_view l_curr = sc.substr (2);
                if (sp.size () > 2 && sp[1] == '-'
                    && (sp[0] == 'B' || sp[0] == 'I')
                    && sp.substr (2) == l_curr)
                  {
                    return -0.2f;
                  }
                return -100.0f;
              }
            return (sc == "O") ? -0.1f : -1.0f;
          };
        auto emit_fn = [this, &tokens] (size_t t, size_t s) -> float
          {
            return rule_emission_score (tokens, t, states_[s]);
          };

        Arena local_arena;
        auto dec = viterbi_decode (seq_len, num_s, init_fn, trans_fn, emit_fn,
                                   local_arena);
        if (dec.has_value ())
          {
            for (size_t t = 0; t < seq_len; ++t)
              {
                bio_tags[t] = states_[dec.value ()[t]];
              }
          }
      }

    /* Overlay longest-match gazetteer phrases onto the BIO sequence.  */
    apply_gazetteer_matches (tokens, bio_tags);
    repair_bio_transitions (bio_tags);
    return bio_tags;
  }

  /**
   * @brief Overload of tag_bio accepting a span of std::string tokens.
   * @pre None.
   * @complexity O(T * S^2 + G * T) time.
   */
  Result<std::vector<std::string>>
  tag_bio (std::span<const std::string> tokens) const
  {
    std::vector<std::string_view> views;
    views.reserve (tokens.size ());
    for (const auto &w : tokens)
      {
        views.emplace_back (w);
      }
    return tag_bio (std::span<const std::string_view> (views));
  }

  /**
   * @brief Extract structured Entity_Span objects from TOKENS.
   * @pre None.
   * @complexity O(T * S^2 + G * T) time and O(T) space.
   */
  Result<std::vector<Entity_Span>>
  extract_spans (std::span<const std::string_view> tokens) const
  {
    auto bio_res = tag_bio (tokens);
    if (!bio_res.has_value ())
      {
        return std::unexpected<Error> (bio_res.error ());
      }
    std::vector<std::string_view> tag_views;
    tag_views.reserve (bio_res.value ().size ());
    for (const std::string &bt : bio_res.value ())
      {
        tag_views.emplace_back (bt);
      }
    return decode_bio_spans (tokens, tag_views);
  }

  /**
   * @brief Overload of extract_spans accepting a span of std::string tokens.
   * @pre None.
   * @complexity O(T * S^2 + G * T) time and O(T) space.
   */
  Result<std::vector<Entity_Span>>
  extract_spans (std::span<const std::string> tokens) const
  {
    std::vector<std::string_view> views;
    views.reserve (tokens.size ());
    for (const auto &w : tokens)
      {
        views.emplace_back (w);
      }
    return extract_spans (std::span<const std::string_view> (views));
  }

  /**
   * @brief Convenience alias for extract_spans(tokens).
   * @pre None.
   * @complexity O(T * S^2 + G * T) time and O(T) space.
   */
  Result<std::vector<Entity_Span>>
  extract (std::span<const std::string_view> tokens) const
  {
    return extract_spans (tokens);
  }

  /**
   * @brief Convenience alias for extract_spans(tokens) on std::string spans.
   * @pre None.
   * @complexity O(T * S^2 + G * T) time and O(T) space.
   */
  Result<std::vector<Entity_Span>>
  extract (std::span<const std::string> tokens) const
  {
    return extract_spans (tokens);
  }

private:
  struct Gazetteer_Entry
  {
    std::vector<std::string> lower_tokens;
    std::string label;
  };

  std::vector<std::string> states_;
  std::unordered_map<std::string, size_t> state_to_id_;
  std::vector<Gazetteer_Entry> gazetteer_;
  HMM_POS_Tagger hmm_;
  bool trained_ = false;

  void
  ensure_bio_state (std::string_view st)
  {
    std::string s (st);
    if (state_to_id_.find (s) == state_to_id_.end ())
      {
        size_t id = states_.size ();
        states_.push_back (s);
        state_to_id_.emplace (std::move (s), id);
      }
  }

  void
  ensure_entity_label (std::string_view label)
  {
    std::string b = "B-" + std::string (label);
    std::string i = "I-" + std::string (label);
    ensure_bio_state (b);
    ensure_bio_state (i);
  }

  float
  rule_emission_score (std::span<const std::string_view> tokens,
                       size_t t,
                       std::string_view state) const
  {
    std::string_view tok = tokens[t];
    bool cap = detail::is_capitalized (tok);
    bool all_up = detail::is_all_upper (tok) && tok.size () >= 2;
    std::string low = detail::ascii_lower (tok);

    bool is_stop = (low == "the" || low == "a" || low == "an" || low == "in"
                    || low == "on" || low == "at" || low == "of"
                    || low == "for" || low == "to" || low == "and"
                    || low == "or" || low == "is" || low == "was"
                    || low == "were" || low == "are");

    if (state == "O")
      {
        if (!cap || is_stop || (t == 0 && !all_up))
          {
            return 2.0f;
          }
        return -0.5f;
      }

    if (!cap || is_stop)
      {
        return -5.0f;
      }

    std::string_view label = (state.size () > 2) ? state.substr (2) : state;
    if (all_up && label == "ORG")
      {
        return 2.5f;
      }
    if (t > 0)
      {
        std::string prev_low = detail::ascii_lower (tokens[t - 1]);
        if ((prev_low == "in" || prev_low == "at" || prev_low == "from")
            && label == "LOC")
          {
            return 2.5f;
          }
        if ((prev_low == "mr." || prev_low == "ms." || prev_low == "dr."
             || prev_low == "prof.")
            && label == "PER")
          {
            return 3.0f;
          }
      }
    if (t + 1 < tokens.size ())
      {
        std::string next_low = detail::ascii_lower (tokens[t + 1]);
        if ((next_low == "inc." || next_low == "corp." || next_low == "ltd."
             || next_low == "corp" || next_low == "inc"
             || next_low == "university")
            && label == "ORG")
          {
            return 2.8f;
          }
      }
    return 0.5f;
  }

  void
  apply_gazetteer_matches (std::span<const std::string_view> tokens,
                           std::vector<std::string> &bio_tags) const
  {
    if (gazetteer_.empty () || tokens.empty ())
      {
        return;
      }

    std::vector<std::string> lower_toks (tokens.size ());
    for (size_t i = 0; i < tokens.size (); ++i)
      {
        lower_toks[i] = detail::ascii_lower (tokens[i]);
      }

    size_t i = 0;
    while (i < tokens.size ())
      {
        size_t best_len = 0;
        std::string_view best_label;

        for (const Gazetteer_Entry &entry : gazetteer_)
          {
            size_t glen = entry.lower_tokens.size ();
            if (glen <= best_len || i + glen > tokens.size ())
              {
                continue;
              }
            bool match = true;
            for (size_t k = 0; k < glen; ++k)
              {
                if (lower_toks[i + k] != entry.lower_tokens[k])
                  {
                    match = false;
                    break;
                  }
              }
            if (match)
              {
                best_len = glen;
                best_label = entry.label;
              }
          }

        if (best_len > 0)
          {
            bio_tags[i] = "B-" + std::string (best_label);
            for (size_t k = 1; k < best_len; ++k)
              {
                bio_tags[i + k] = "I-" + std::string (best_label);
              }
            i += best_len;
          }
        else
          {
            ++i;
          }
      }
  }

  static void
  repair_bio_transitions (std::vector<std::string> &bio_tags)
  {
    std::string prev_label;
    for (std::string &tag : bio_tags)
      {
        if (tag.size () > 2 && tag[1] == '-')
          {
            std::string label = tag.substr (2);
            if (tag[0] == 'I' && prev_label != label)
              {
                tag[0] = 'B';
              }
            prev_label = std::move (label);
          }
        else
          {
            prev_label.clear ();
          }
      }
  }
};

/**
 * @brief Aliases for NER_Span_Extractor matching ARCHITECTURE.md / TASKS.md.
 */
using NER_Tagger = NER_Span_Extractor;
using NER_Recognizer = NER_Span_Extractor;

} /* namespace obsidian */

#endif /* OBSIDIAN_TAGGER_HPP */
