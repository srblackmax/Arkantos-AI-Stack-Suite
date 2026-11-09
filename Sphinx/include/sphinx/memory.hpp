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

#ifndef SPHINX_MEMORY_HPP
#define SPHINX_MEMORY_HPP

/**
 * @file memory.hpp
 * @brief Cross-session memory subsystem: CPU EmbeddingClient, Yantra
 *        WAL + HNSW MemoryStore, decay/supersession policy, and memory
 *        extraction/retrieval helpers.
 * @note Exposes zero third-party LLM backend symbols in public headers.
 */

#include <sutra/sutra.hpp>
#include <yantra/yantra.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sphinx::inline v1
{

using MemoryId = uint64_t;

inline constexpr float kDefaultMemoryThreshold = 0.50F;
inline constexpr double kEpisodicHalfLifeDays = 30.0;
inline constexpr std::size_t kDefaultRetrieveTopK = 3U;

/**
 * @brief Memory tier classification for cross-session persistence.
 */
enum class MemoryType : uint8_t
{
  EPISODIC = 0,
  SEMANTIC = 1,
  CORRECTION = 2,
  ANY = 255
};

/**
 * @brief Converts a MemoryType enum to its uppercase label.
 */
[[nodiscard]] constexpr auto
memory_type_to_string (MemoryType type) noexcept -> std::string_view
{
  switch (type)
    {
    case MemoryType::EPISODIC:
      return "EPISODIC";
    case MemoryType::SEMANTIC:
      return "SEMANTIC";
    case MemoryType::CORRECTION:
      return "CORRECTION";
    case MemoryType::ANY:
      return "ANY";
    }
  return "SEMANTIC";
}

/**
 * @brief Single persisted memory record indexed in MemoryStore.
 */
struct MemoryEntry
{
  MemoryId id = 0U;
  std::string text;
  std::vector<float> embedding;
  MemoryType type = MemoryType::SEMANTIC;
  uint64_t timestamp_ns = 0U;
  std::optional<MemoryId> superseded_by;
  float similarity = 0.0F;
  float effective_score = 0.0F;
};

/**
 * @brief Returns default path for Qwen3-Embedding-0.6B-Q8_0.gguf.
 */
[[nodiscard]] inline auto
default_embedding_model_path () -> std::filesystem::path
{
  const char *home = std::getenv ("HOME");
  const std::string base = (home != nullptr && home[0] != '\0') ? home : ".";
  return std::filesystem::path (base)
         / "models/Qwen3-Embedding-0.6B-Q8_0.gguf";
}

/**
 * @brief Configuration for EmbeddingClient.
 */
struct EmbeddingClientConfig
{
  std::filesystem::path model_path = default_embedding_model_path ();
  int32_t n_threads = 8;
  uint32_t n_ctx = 128U;
};

/**
 * @brief CPU-only embedding client running Qwen3-Embedding-0.6B-GGUF on a
 *        dedicated background worker thread behind an opaque handle.
 */
class EmbeddingClient
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  EmbeddingClient (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates an EmbeddingClient. If the model file is missing, prints a
   *        warning to stderr and returns a disabled client (graceful
   *        degradation).
   */
  [[nodiscard]] static auto
  create (const EmbeddingClientConfig &config
          = EmbeddingClientConfig{}) noexcept
      -> sutra::Result<std::unique_ptr<EmbeddingClient>>;

  EmbeddingClient (const EmbeddingClient &) = delete;
  auto operator= (const EmbeddingClient &) -> EmbeddingClient & = delete;
  EmbeddingClient (EmbeddingClient &&) noexcept;
  auto operator= (EmbeddingClient &&) noexcept -> EmbeddingClient &;
  ~EmbeddingClient () noexcept;

  /**
   * @brief Computes a unit-normalized embedding vector on the dedicated CPU
   *        worker thread.
   */
  [[nodiscard]] auto embed (std::string_view text) noexcept
      -> sutra::Result<std::vector<float>>;

  /**
   * @brief Queries the embedding vector dimension read from model metadata.
   */
  [[nodiscard]] auto dimension () const noexcept -> std::size_t;

  /**
   * @brief Returns true if the embedding model was loaded and is active.
   */
  [[nodiscard]] auto is_enabled () const noexcept -> bool;

  /**
   * @brief Computes cosine similarity between two unit vectors.
   */
  [[nodiscard]] static auto
  cosine_similarity (std::span<const float> a,
                     std::span<const float> b) noexcept -> float;

private:
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Durable cross-session memory store backed by Yantra WalEngine
 *        (sphinx_memory.wal) and Yantra HnswIndex (sphinx_memory_hnsw).
 */
class MemoryStore
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  MemoryStore (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Opens or initializes the MemoryStore in memory_dir, replaying
   *        sphinx_memory.wal and loading or rebuilding sphinx_memory_hnsw.
   */
  [[nodiscard]] static auto
  open (const std::filesystem::path &memory_dir,
        std::size_t dimension = 1024U) noexcept
      -> sutra::Result<std::unique_ptr<MemoryStore>>;

  MemoryStore (const MemoryStore &) = delete;
  auto operator= (const MemoryStore &) -> MemoryStore & = delete;
  MemoryStore (MemoryStore &&) noexcept;
  auto operator= (MemoryStore &&) noexcept -> MemoryStore &;
  ~MemoryStore () noexcept;

  /**
   * @brief Appends a memory entry to the WAL and inserts its embedding into
   *        the HNSW index.
   * @param text Memory statement text.
   * @param embedding Unit-normalized embedding vector.
   * @param type EPISODIC, SEMANTIC, or CORRECTION.
   * @param timestamp_ns Optional explicit timestamp in nanoseconds since epoch
   *        (0 uses current system clock time).
   */
  [[nodiscard]] auto
  store (std::string_view text, std::span<const float> embedding,
         MemoryType type, uint64_t timestamp_ns = 0U) noexcept
      -> sutra::Result<MemoryId>;

  /**
   * @brief Retrieves the top_k non-superseded memories sorted by descending
   *        effective score (applying 30-day exponential decay to EPISODIC
   *        entries).
   */
  [[nodiscard]] auto
  retrieve (std::span<const float> query_embedding,
            std::size_t top_k = kDefaultRetrieveTopK,
            MemoryType filter = MemoryType::ANY,
            uint64_t now_timestamp_ns = 0U) const noexcept
      -> sutra::Result<std::vector<MemoryEntry>>;

  /**
   * @brief Marks memory id as superseded by memory by. The superseded entry
   *        remains in the WAL for audit but is excluded from retrieve().
   */
  [[nodiscard]] auto supersede (MemoryId id, MemoryId by) noexcept
      -> sutra::Result<void>;

  /**
   * @brief Flushes and closes the WAL and saves the HNSW index to disk.
   */
  [[nodiscard]] auto close () noexcept -> sutra::Result<void>;

  /**
   * @brief Completely erases all in-memory entries (SEMANTIC, EPISODIC, and
   *        CORRECTION), resets the HNSW index, and truncates/recreates the
   *        underlying WAL and payload files on disk.
   */
  [[nodiscard]] auto clear () noexcept -> sutra::Result<void>;

  /**
   * @brief Removes all persistent memory files (sphinx_memory.wal,
   *        sphinx_memory.dat, sphinx_memory_hnsw) inside memory_dir on disk.
   */
  [[nodiscard]] static auto
  wipe_directory (const std::filesystem::path &memory_dir) noexcept
      -> sutra::Result<void>;

  /**
   * @brief Returns total number of stored entries (including superseded).
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Returns all entries (including superseded) for audit inspection.
   */
  [[nodiscard]] auto all_entries () const -> std::vector<MemoryEntry>;

private:
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Detects whether user_text is an explicit fact storage request
 *        ("remember that ...", "remember: ...", "note that ...",
 *        "keep in mind that ...") and extracts normalized fact strings.
 */
[[nodiscard]] auto
extract_explicit_semantic_facts (std::string_view user_text)
    -> std::vector<std::string>;

/**
 * @brief Detects whether user_text is an explicit correction
 *        ("no, that's wrong", "no, it's X, not Y", "actually, ...",
 *        "I said X, not Y") and returns the normalized correction statement.
 */
[[nodiscard]] auto
extract_explicit_correction (std::string_view user_text)
    -> std::optional<std::string>;

/**
 * @brief Returns true if user_text is a trivial greeting or farewell that
 *        should not trigger an episodic session summary on its own.
 */
[[nodiscard]] auto
is_trivial_greeting_turn (std::string_view user_text) noexcept -> bool;

/**
 * @brief Returns true if user_text is an interactive command or natural
 *        language instruction to clear/erase/reset persistent memory.
 */
[[nodiscard]] auto
is_clear_memory_command (std::string_view user_text) noexcept -> bool;

/**
 * @brief Formats retrieved memories (with effective_score > threshold) into
 *        the system prompt injection block. Returns empty string if none
 *        exceed threshold.
 */
[[nodiscard]] auto
format_memories_for_prompt (std::span<const MemoryEntry> memories,
                            float threshold = kDefaultMemoryThreshold,
                            uint64_t now_timestamp_ns = 0U) -> std::string;

} /* namespace sphinx::inline v1 */

#endif /* SPHINX_MEMORY_HPP */
