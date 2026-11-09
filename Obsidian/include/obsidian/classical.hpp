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

#ifndef OBSIDIAN_CLASSICAL_HPP
#define OBSIDIAN_CLASSICAL_HPP

/**
 * @file classical.hpp
 * @brief Classical NLP algorithms: edit distance templates, sentence boundary
 *        detection, N-gram language models (Kneser-Ney/Laplace), TF-IDF, and
 *        Okapi BM25 ranking.
 */

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/memory.hpp>
#include <obsidian/tensor_bridge.hpp>
#include <obsidian/tokenizer.hpp>

namespace obsidian
{

/**
 * @brief Compute the Levenshtein edit distance between two character or token
 *        sequences using O(min(M, N)) arena space.
 * @pre None.
 * @complexity O(M * N) time and O(min(M, N)) space.
 */
template <typename Elem = char>
constexpr size_t
levenshtein_distance (std::span<const Elem> a,
                      std::span<const Elem> b)
{
  if (a.size () > b.size ())
    {
      std::swap (a, b);
    }
  const size_t m = a.size ();
  const size_t n = b.size ();
  if (m == 0)
    {
      return n;
    }

  std::vector<size_t> prev (m + 1);
  std::vector<size_t> curr (m + 1);
  for (size_t i = 0; i <= m; ++i)
    {
      prev[i] = i;
    }

  for (size_t j = 1; j <= n; ++j)
    {
      curr[0] = j;
      for (size_t i = 1; i <= m; ++i)
        {
          size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
          curr[i] = std::min ({prev[i] + 1,
                               curr[i - 1] + 1,
                               prev[i - 1] + cost});
        }
      std::swap (prev, curr);
    }
  return prev[m];
}

/**
 * @brief Convenience overload computing Levenshtein distance on string views.
 * @pre None.
 * @complexity O(|A| * |B|) time and O(min(|A|, |B|)) space.
 */
inline size_t
levenshtein_distance (std::string_view a, std::string_view b)
{
  return levenshtein_distance<char> (
    std::span<const char> (a.data (), a.size ()),
    std::span<const char> (b.data (), b.size ()));
}

/**
 * @brief Compute the Optimal String Alignment / Damerau-Levenshtein distance
 *        accounting for adjacent character transpositions.
 * @pre None.
 * @complexity O(|A| * |B|) time and O(|B|) space.
 */
OBSIDIAN_API size_t
damerau_levenshtein_distance (std::string_view a, std::string_view b);

/**
 * @brief Orthographic and abbreviation-aware Sentence Boundary Detector
 *        (Punkt-inspired).
 */
class OBSIDIAN_API Sentence_Segmenter
{
public:
  /**
   * @brief Construct a Sentence_Segmenter with common English abbreviations.
   * @pre None.
   * @complexity O(1) time and space.
   */
  Sentence_Segmenter (void);

  /**
   * @brief Register a custom abbreviation (without trailing period).
   * @pre ABBR must not be empty.
   * @complexity O(|ABBR|) time.
   */
  void
  add_abbreviation (std::string_view abbr);

  /**
   * @brief Segment TEXT into zero-copy sentence string_view spans.
   * @pre TEXT must remain alive while returned string_views are accessed.
   * @complexity O(N) time in character length N.
   */
  std::vector<std::string_view>
  segment (std::string_view text) const;

private:
  Vocabulary abbrevs_;
};

/**
 * @brief Smoothing policy for N-gram language models.
 */
enum class Smoothing_Method : uint8_t
{
  MLE = 0,
  Laplace = 1,
  Kneser_Ney = 2
};

/**
 * @brief Statistical N-gram language model supporting MLE, Laplace (add-k),
 *        and Interpolated Kneser-Ney smoothing.
 */
class OBSIDIAN_API NGram_Model
{
public:
  /**
   * @brief Construct an N-gram language model of order N.
   * @pre 1 <= ORDER <= 5.
   * @complexity O(1) time and space.
   */
  explicit NGram_Model (
    size_t order = 3,
    Smoothing_Method smoothing = Smoothing_Method::Kneser_Ney,
    double discount_or_alpha = 0.75);

  /**
   * @brief Fit N-gram counts and continuation statistics on a text corpus.
   * @pre CORPUS must not be empty.
   * @complexity O(total_tokens * order) time and space.
   */
  Result<void>
  fit (std::span<const std::string_view> corpus);

  /**
   * @brief Compute conditional probability P(word | context).
   * @pre Model must be fitted via fit().
   * @complexity O(order) hash lookups and O(1) space.
   */
  double
  probability (std::span<const std::string_view> context,
               std::string_view word) const;

  /**
   * @brief Compute log-probability log(P(sentence)) of a document.
   * @pre Model must be fitted.
   * @complexity O(N_tokens * order) time.
   */
  double
  sentence_log_prob (std::string_view text) const;

  /**
   * @brief Compute corpus perplexity exp(-1/N * sum log P(w_i | context)).
   * @pre CORPUS must not be empty and model must be fitted.
   * @complexity O(N_tokens * order) time.
   */
  Result<double>
  perplexity (std::span<const std::string_view> corpus) const;

  /**
   * @brief Return the N-gram order.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  order (void) const noexcept
  {
    return order_;
  }

  /**
   * @brief Return the vocabulary size including <bos>, <eos>, and <unk>.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_.size ();
  }

private:
  double
  prob_recursive (std::span<const int32_t> ctx, int32_t word_id) const;

  size_t order_;
  Smoothing_Method smoothing_;
  double param_;
  Vocabulary vocab_;
  int32_t bos_id_ = 0;
  int32_t eos_id_ = 1;
  int32_t unk_id_ = 2;
  size_t total_unigrams_ = 0;
  size_t total_unique_bigrams_ = 0;
  std::vector<size_t> unigram_counts_;
  std::vector<size_t> continuation_counts_;
  /* Maps history key string -> (word_id -> count).  */
  std::unordered_map<std::string,
                     std::unordered_map<int32_t, size_t>> ngram_counts_;
  std::unordered_map<std::string, size_t> context_totals_;
};

/**
 * @brief Term Frequency - Inverse Document Frequency (TF-IDF) vectorizer
 *        producing L2-normalized Basalt Tensor representations.
 */
class OBSIDIAN_API TFIDF_Vectorizer
{
public:
  /**
   * @brief Construct a TFIDF_Vectorizer with sublinear TF and smooth IDF.
   * @pre None.
   * @complexity O(1) time and space.
   */
  explicit TFIDF_Vectorizer (bool sublinear_tf = false,
                             bool smooth_idf = true);

  /**
   * @brief Learn vocabulary and inverse document frequencies from CORPUS.
   * @pre CORPUS must not be empty.
   * @complexity O(total_tokens) time and O(V) space.
   */
  Result<void>
  fit (std::span<const std::string_view> corpus);

  /**
   * @brief Transform a single document into an L2-normalized [V] float vector.
   * @pre Vectorizer must be fitted.
   * @complexity O(N_doc + V) time and O(V) space.
   */
  Result<std::vector<float>>
  transform_doc (std::string_view doc) const;

  /**
   * @brief Transform a batch of documents into a [num_docs, V] Basalt Tensor.
   * @pre Vectorizer must be fitted and DOCS must not be empty.
   * @complexity O(D * V) time and space.
   */
  Result<Tensor>
  transform (std::span<const std::string_view> docs) const;

  /**
   * @brief Fit on CORPUS and return the [num_docs, V] TF-IDF Basalt Tensor.
   * @pre CORPUS must not be empty.
   * @complexity O(total_tokens + D * V) time and space.
   */
  Result<Tensor>
  fit_transform (std::span<const std::string_view> corpus);

  /**
   * @brief Return the learned vocabulary.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const Vocabulary &
  vocab (void) const noexcept
  {
    return vocab_;
  }

private:
  bool sublinear_tf_;
  bool smooth_idf_;
  Vocabulary vocab_;
  std::vector<float> idf_;
};

/**
 * @brief Document search hit returned by BM25_Index.
 */
struct OBSIDIAN_API Search_Hit
{
  size_t doc_id = 0;
  float score = 0.0f;
};

/**
 * @brief Okapi BM25 inverted index for high-speed lexical information
 *        retrieval.
 */
class OBSIDIAN_API BM25_Index
{
public:
  /**
   * @brief Construct a BM25_Index with saturation k1 and length norm b.
   * @pre K1 > 0.0f and 0.0f <= B <= 1.0f.
   * @complexity O(1) time and space.
   */
  explicit BM25_Index (float k1 = 1.5f, float b = 0.75f);

  /**
   * @brief Index a collection of documents.
   * @pre DOCS must not be empty.
   * @complexity O(total_tokens) time and space.
   */
  Result<void>
  index_documents (std::span<const std::string_view> docs);

  /**
   * @brief Rank indexed documents against QUERY and return the top TOP_K hits.
   * @pre Index must be populated via index_documents().
   * @complexity O(|QUERY| * postings_len + N_docs * log(TOP_K)) time.
   */
  Result<std::vector<Search_Hit>>
  search (std::string_view query, size_t top_k = 5) const;

  /**
   * @brief Return the number of indexed documents.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  num_documents (void) const noexcept
  {
    return doc_lengths_.size ();
  }

private:
  struct Posting
  {
    uint32_t doc_id = 0;
    uint32_t term_freq = 0;
  };

  float k1_;
  float b_;
  float avg_doc_len_ = 0.0f;
  Vocabulary vocab_;
  std::vector<uint32_t> doc_lengths_;
  std::vector<float> idf_;
  std::vector<std::vector<Posting>> postings_;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_CLASSICAL_HPP */
