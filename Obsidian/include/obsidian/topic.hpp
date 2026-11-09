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

#ifndef OBSIDIAN_TOPIC_HPP
#define OBSIDIAN_TOPIC_HPP

/**
 * @file topic.hpp
 * @brief Topic modeling and semantic space discovery: Latent Dirichlet
 *        Allocation (Collapsed Gibbs Sampling) and Latent Semantic Analysis
 *        (Truncated SVD via Basalt Tensors).
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <obsidian/classical.hpp>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/tensor_bridge.hpp>
#include <obsidian/tokenizer.hpp>

namespace obsidian
{

/**
 * @brief Latent Dirichlet Allocation (LDA) topic model trained via Collapsed
 *        Gibbs Sampling.
 */
class OBSIDIAN_API LDA_Model
{
public:
  /**
   * @brief Construct an LDA_Model with Dirichlet priors alpha and beta.
   * @pre NUM_TOPICS >= 2, ALPHA > 0.0f, and BETA > 0.0f.
   * @complexity O(1) time and space.
   */
  explicit LDA_Model (size_t num_topics = 4,
                      float alpha = 0.1f,
                      float beta = 0.01f,
                      size_t iterations = 50,
                      uint64_t seed = 42);

  /**
   * @brief Fit the LDA topic model on CORPUS using collapsed Gibbs sampling.
   * @pre CORPUS must not be empty.
   * @complexity O(iterations * total_tokens * num_topics) time.
   */
  Result<void>
  fit (std::span<const std::string_view> corpus);

  /**
   * @brief Return the TOP_K highest-probability words for TOPIC_ID.
   * @pre Model must be fitted and TOPIC_ID < num_topics().
   * @complexity O(V * log(TOP_K)) time.
   */
  Result<std::vector<std::pair<std::string, float>>>
  topic_words (size_t topic_id, size_t top_k = 10) const;

  /**
   * @brief Return the posterior topic distribution theta_d for DOC_ID.
   * @pre Model must be fitted and DOC_ID < num_documents().
   * @complexity O(num_topics) time and space.
   */
  Result<std::vector<float>>
  document_topic_distribution (size_t doc_id) const;

  /**
   * @brief Infer the topic distribution for an unseen document DOC.
   * @pre Model must be fitted.
   * @complexity O(infer_iters * N_doc_tokens * num_topics) time.
   */
  Result<std::vector<float>>
  infer_document (std::string_view doc,
                  size_t infer_iters = 25) const;

  /**
   * @brief Export the [num_topics, vocab_size] topic-word probability matrix
   *        as a Basalt Tensor.
   * @pre Model must be fitted.
   * @complexity O(num_topics * vocab_size) time and space.
   */
  Result<Tensor>
  topic_word_tensor (void) const;

  /**
   * @brief Return the number of latent topics K.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  num_topics (void) const noexcept
  {
    return num_topics_;
  }

  /**
   * @brief Return the number of training documents.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  num_documents (void) const noexcept
  {
    return docs_.size ();
  }

  /**
   * @brief Return the vocabulary size V.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_.size ();
  }

private:
  size_t num_topics_;
  float alpha_;
  float beta_;
  size_t iterations_;
  uint64_t seed_;
  Vocabulary vocab_;
  std::vector<std::vector<int32_t>> docs_;
  std::vector<std::vector<size_t>> z_;
  std::vector<size_t> doc_topic_counts_;
  std::vector<size_t> topic_word_counts_;
  std::vector<size_t> topic_totals_;
};

/**
 * @brief Latent Semantic Analysis (LSA / Truncated SVD) powered by Basalt
 *        Tensor matrix multiplications.
 */
class OBSIDIAN_API LSA_Model
{
public:
  /**
   * @brief Construct an LSA_Model extracting N_COMPONENTS singular vectors.
   * @pre N_COMPONENTS >= 1.
   * @complexity O(1) time and space.
   */
  explicit LSA_Model (size_t n_components = 4,
                      size_t power_iterations = 30);

  /**
   * @brief Fit TF-IDF and Truncated SVD on CORPUS, returning the
   *        [num_docs, n_components] latent document coordinates Tensor.
   * @pre CORPUS must not be empty.
   * @complexity O(n_components * power_iterations * D * V) via Basalt matmul.
   */
  Result<Tensor>
  fit_transform (std::span<const std::string_view> corpus);

  /**
   * @brief Project DOCS into the learned [num_docs, n_components] LSA space.
   * @pre Model must be fitted and DOCS must not be empty.
   * @complexity O(D * V * n_components) via Basalt matmul.
   */
  Result<Tensor>
  transform (std::span<const std::string_view> docs) const;

  /**
   * @brief Return the learned [n_components, vocab_size] singular components.
   * @pre Model must be fitted.
   * @complexity O(1) time and O(1) space.
   */
  const Tensor &
  components (void) const noexcept
  {
    return components_;
  }

  /**
   * @brief Return the singular values tensor [n_components].
   * @pre Model must be fitted.
   * @complexity O(1) time and O(1) space.
   */
  const Tensor &
  singular_values (void) const noexcept
  {
    return singular_values_;
  }

private:
  size_t n_components_;
  size_t power_iters_;
  TFIDF_Vectorizer tfidf_;
  Tensor components_;
  Tensor singular_values_;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_TOPIC_HPP */
