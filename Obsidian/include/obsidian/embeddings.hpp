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

#ifndef OBSIDIAN_EMBEDDINGS_HPP
#define OBSIDIAN_EMBEDDINGS_HPP

/**
 * @file embeddings.hpp
 * @brief Memory-mapped read-only word vector store (GloVe/word2vec/obvec),
 *        SIMD nearest-neighbor search, Word2Vec trainer, and FastText hashing.
 */

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
#include <obsidian/tokenizer.hpp>

namespace obsidian
{

/**
 * @brief Word neighbor entry returned by nearest-neighbor and analogy queries.
 */
struct OBSIDIAN_API Word_Neighbor
{
  std::string word;
  int32_t id = -1;
  float similarity = 0.0f;
};

/**
 * @brief Read-only word vector store supporting POSIX mmap zero-copy access
 *        as well as GloVe text and Word2Vec text/binary formats.
 */
class OBSIDIAN_API Mmap_Embeddings
{
public:
  /**
   * @brief Construct an empty uninitialized Mmap_Embeddings store.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Mmap_Embeddings (void) noexcept = default;

  /**
   * @brief Unmap any active memory-mapped file region upon destruction.
   * @pre None.
   * @complexity O(1) time.
   */
  ~Mmap_Embeddings (void) noexcept;

  Mmap_Embeddings (const Mmap_Embeddings &) = delete;
  Mmap_Embeddings &operator= (const Mmap_Embeddings &) = delete;

  Mmap_Embeddings (Mmap_Embeddings &&other) noexcept;
  Mmap_Embeddings &operator= (Mmap_Embeddings &&other) noexcept;

  /**
   * @brief Build an in-memory embedding store from words and a flat matrix.
   * @pre MATRIX.size() == WORDS.size() * DIM and DIM > 0.
   * @complexity O(V * D) time and space.
   */
  static Result<Mmap_Embeddings>
  from_memory (std::span<const std::string> words,
               std::span<const float> matrix,
               size_t dim);

  /**
   * @brief Memory-map a native 64-byte-aligned .obvec binary file via mmap(2).
   *
   * Zero-copy: the float vectors remain in the OS page cache and are never
   * copied to the heap.
   *
   * @pre PATH must refer to a valid .obvec file.
   * @complexity O(V) index attachment time and O(1) resident heap for vectors.
   */
  static Result<Mmap_Embeddings>
  open_mmap (const std::string &path);

  /**
   * @brief Load embeddings from a GloVe text file ("word f1 f2 ... fD").
   * @pre PATH must be a readable GloVe-formatted text file.
   * @complexity O(V * D) parsing time and space.
   */
  static Result<Mmap_Embeddings>
  load_glove_text (const std::string &path);

  /**
   * @brief Load embeddings from a Word2Vec text file ("V D\nword f1 ...").
   * @pre PATH must be a readable Word2Vec text file.
   * @complexity O(V * D) parsing time and space.
   */
  static Result<Mmap_Embeddings>
  load_word2vec_text (const std::string &path);

  /**
   * @brief Load embeddings from a Google Word2Vec binary file.
   * @pre PATH must be a readable Word2Vec binary file.
   * @complexity O(V * D) time and space.
   */
  static Result<Mmap_Embeddings>
  load_word2vec_binary (const std::string &path);

  /**
   * @brief Save this embedding store to a 64-byte-aligned .obvec file for
   *        subsequent zero-copy open_mmap() loading.
   * @pre Store must be initialized (vocab_size() > 0).
   * @complexity O(V * D) sequential disk write time.
   */
  Result<void>
  save_obvec_mmap (const std::string &path) const;

  /**
   * @brief Save this embedding store in GloVe text format.
   * @pre Store must be initialized.
   * @complexity O(V * D) time.
   */
  Result<void>
  save_glove_text (const std::string &path) const;

  /**
   * @brief Save this embedding store in Word2Vec binary format.
   * @pre Store must be initialized.
   * @complexity O(V * D) time.
   */
  Result<void>
  save_word2vec_binary (const std::string &path) const;

  /**
   * @brief Look up the embedding vector of WORD in O(1) time with zero copy.
   * @pre Store must be initialized.
   * @complexity O(|WORD|) hash time and O(1) zero-copy span return.
   */
  Result<std::span<const float>>
  lookup (std::string_view word) const noexcept;

  /**
   * @brief Look up the embedding vector by integer ID in O(1) time.
   * @pre 0 <= ID < vocab_size().
   * @complexity O(1) time and O(1) space.
   */
  Result<std::span<const float>>
  lookup_by_id (int32_t id) const noexcept;

  /**
   * @brief Return the integer ID of WORD or -1 if out-of-vocabulary.
   * @pre None.
   * @complexity O(|WORD|) expected time and O(1) space.
   */
  int32_t
  word_to_id (std::string_view word) const noexcept
  {
    return vocab_.token_to_id (word);
  }

  /**
   * @brief Return the word string for ID.
   * @pre 0 <= ID < vocab_size().
   * @complexity O(1) time and O(1) space.
   */
  Result<std::string_view>
  id_to_word (int32_t id) const noexcept
  {
    return vocab_.id_to_token (id);
  }

  /**
   * @brief Compute SIMD cosine similarity between W1 and W2.
   * @pre Both W1 and W2 must exist in the vocabulary.
   * @complexity O(D / 8) AVX2 operations and O(1) space.
   */
  Result<float>
  similarity (std::string_view w1, std::string_view w2) const noexcept;

  /**
   * @brief Find the TOP_K nearest neighbors of WORD by cosine similarity.
   * @pre WORD must exist in the vocabulary; TOP_K > 0.
   * @complexity O(V * D / 8 + V * log(TOP_K)) time.
   */
  Result<std::vector<Word_Neighbor>>
  most_similar (std::string_view word, size_t top_k = 5) const;

  /**
   * @brief Find the TOP_K words closest to an arbitrary query vector.
   * @pre QUERY.size() == dim().
   * @complexity O(V * D / 8 + V * log(TOP_K)) time.
   */
  Result<std::vector<Word_Neighbor>>
  most_similar_to_vector (
    std::span<const float> query,
    size_t top_k = 5,
    std::span<const std::string_view> exclude = {}) const;

  /**
   * @brief Solve the analogy A is to A_STAR as B is to ? (v_a* - v_a + v_b).
   * @pre A, A_STAR, and B must exist in the vocabulary.
   * @complexity O(V * D / 8) time and O(D) space.
   */
  Result<std::vector<Word_Neighbor>>
  analogy (std::string_view a,
           std::string_view a_star,
           std::string_view b,
           size_t top_k = 1) const;

  /**
   * @brief Convert the entire embedding table into a Basalt Tensor [V, D].
   * @pre Store must be initialized.
   * @complexity O(V * D) time.
   */
  Result<Tensor>
  to_basalt_tensor (bool requires_grad = false) const;

  /**
   * @brief Look up a sequence of words and return a [seq_len, dim] Tensor.
   * @pre WORDS must not be empty.
   * @complexity O(seq_len * dim) time and space.
   */
  Result<Tensor>
  embed_sequence (std::span<const std::string_view> words) const;

  /**
   * @brief Return the number of words in the store.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  vocab_size (void) const noexcept
  {
    return vocab_.size ();
  }

  /**
   * @brief Return the dimensionality D of each word vector.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  dim (void) const noexcept
  {
    return dim_;
  }

  /**
   * @brief Return true if backed by an OS mmap(2) file mapping.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  bool
  is_memory_mapped (void) const noexcept
  {
    return mmap_addr_ != nullptr;
  }

private:
  const float *
  raw_matrix_ptr (void) const noexcept
  {
    return (mmap_floats_ != nullptr) ? mmap_floats_ : owned_floats_.data ();
  }

  Vocabulary vocab_;
  size_t dim_ = 0;
  std::vector<float> owned_floats_;
  void *mmap_addr_ = nullptr;
  size_t mmap_size_ = 0;
  const float *mmap_floats_ = nullptr;
};

/**
 * @brief Training configuration for Word2Vec (Skip-Gram / CBOW).
 */
struct OBSIDIAN_API Word2Vec_Config
{
  size_t dim = 32;
  size_t window_size = 2;
  size_t negative_samples = 4;
  size_t min_count = 1;
  size_t epochs = 10;
  float learning_rate = 0.05f;
  bool skip_gram = true;
  uint64_t seed = 42;
};

/**
 * @brief Native C++23 Word2Vec trainer supporting Skip-Gram and CBOW with
 *        negative sampling and SIMD gradient updates.
 */
class OBSIDIAN_API Word2Vec_Trainer
{
public:
  /**
   * @brief Train a Word2Vec embedding store from a corpus of documents.
   * @pre CORPUS must not be empty; CFG.dim > 0.
   * @complexity O(epochs * N_tokens * window_size * negative_samples * dim).
   */
  static Result<Mmap_Embeddings>
  train (std::span<const std::string_view> corpus,
         const Word2Vec_Config &cfg = Word2Vec_Config ());
};

/**
 * @brief FastText-style character n-gram subword embedding model capable of
 *        synthesizing embeddings for out-of-vocabulary (OOV) words.
 */
class OBSIDIAN_API FastText_Embeddings
{
public:
  /**
   * @brief Construct a FastText subword model with bucket hashing.
   * @pre DIM > 0, NUM_BUCKETS > 0, and 1 <= MIN_N <= MAX_N.
   * @complexity O(num_buckets * dim) initialization time and space.
   */
  FastText_Embeddings (size_t dim = 32,
                       size_t num_buckets = 4096,
                       size_t min_n = 3,
                       size_t max_n = 6,
                       uint64_t seed = 1337);

  /**
   * @brief Train subword + word vectors on a corpus using Skip-Gram negative
   *        sampling.
   * @pre CORPUS must not be empty.
   * @complexity O(epochs * N_tokens * ngrams * dim) time.
   */
  Result<void>
  train (std::span<const std::string_view> corpus,
         size_t epochs = 10,
         float lr = 0.05f);

  /**
   * @brief Compute the normalized embedding vector for any word (known or OOV)
   *        by averaging its character n-gram bucket vectors.
   * @pre OUT_VEC.size() == dim().
   * @complexity O(|WORD| * (max_n - min_n + 1) * dim / 8) SIMD time.
   */
  void
  compute_word_vector (std::string_view word,
                       std::span<float> out_vec) const noexcept;

  /**
   * @brief Return the embedding dimension D.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  dim (void) const noexcept
  {
    return dim_;
  }

  /**
   * @brief Compute deterministic FNV-1a bucket index for a character n-gram.
   * @pre None.
   * @complexity O(|NGRAM|) time and O(1) space.
   */
  size_t
  ngram_bucket (std::string_view ngram) const noexcept;

private:
  size_t dim_;
  size_t num_buckets_;
  size_t min_n_;
  size_t max_n_;
  Vocabulary known_words_;
  std::vector<float> word_vectors_;
  std::vector<float> bucket_vectors_;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_EMBEDDINGS_HPP */
