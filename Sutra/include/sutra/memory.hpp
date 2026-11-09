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

#ifndef SUTRA_MEMORY_HPP
#define SUTRA_MEMORY_HPP

#include <sutra/error.hpp>

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sutra::inline v1
{

inline constexpr std::size_t kDefaultArenaCapacityBytes = 65536U;
inline constexpr std::size_t kDefaultShortTermCapacity = 64U;
inline constexpr std::size_t kDefaultSemanticDimension = 128U;
inline constexpr std::size_t kDefaultSemanticTopK = 5U;

/**
 * @brief Identifies the speaker role of a conversation message.
 * @note Thread-safety: Immutable enum type.
 */
enum class MessageRole : uint8_t
{
  System = 0,
  User = 1,
  Assistant = 2,
  Tool = 3
};

/**
 * @brief Represents a single conversation turn in short-term memory.
 * @note Thread-safety: Value type; safe for concurrent reads when const.
 */
struct Message
{
  MessageRole role = MessageRole::User;
  std::string content;
  std::string name;
  std::string tool_call_id;
  uint64_t sequence_id = 0U;
  uint64_t timestamp_ns = 0U;
};

/**
 * @brief Non-owning zero-copy view of ring-buffer messages for ToolContext.
 * @pre Underlying ShortTermMemory must not be mutated while view is read.
 * @post Zero heap allocations or message copies occur when creating this view.
 * @note Thread-safety: Immutable view of two contiguous ring segments.
 */
struct ConversationSpanView
{
  std::span<const Message> first_segment;
  std::span<const Message> second_segment;
  std::size_t total_turns = 0U;

  /**
   * @brief Returns the number of visible messages across both ring segments.
   * @pre None.
   * @post State is unchanged.
   * @returns Sum of first_segment.size() and second_segment.size().
   */
  [[nodiscard]] constexpr auto
  size () const noexcept -> std::size_t
  {
    return first_segment.size () + second_segment.size ();
  }

  /**
   * @brief Accesses the i-th logical message in chronological order.
   * @param index Logical index in [0, size()).
   * @pre index < size().
   * @post State is unchanged.
   * @returns Const reference to the Message at logical index.
   */
  [[nodiscard]] constexpr auto
  at (std::size_t index) const noexcept -> const Message &
  {
    if (index < first_segment.size ())
      {
        return first_segment[index];
      }
    return second_segment[index - first_segment.size ()];
  }
};

/**
 * @brief Per-inference PMR monotonic scratch arena with RAII reset semantics.
 * @pre capacity_bytes > 0.
 * @post Releases all arena memory deterministically on reset() or destruction.
 * @note Thread-safety: Intended for single-inference-turn exclusive use.
 */
class InferenceArena
{
public:
  /**
   * @brief Constructs a monotonic arena with a pre-allocated backing buffer.
   * @param capacity_bytes Initial backing buffer size in bytes.
   * @pre capacity_bytes > 0.
   * @post Backing buffer is allocated once via RAII std::vector<std::byte>.
   */
  explicit InferenceArena (std::size_t capacity_bytes
                           = kDefaultArenaCapacityBytes);

  InferenceArena (const InferenceArena &) = delete;
  auto operator= (const InferenceArena &) -> InferenceArena & = delete;
  InferenceArena (InferenceArena &&) = delete;
  auto operator= (InferenceArena &&) -> InferenceArena & = delete;
  ~InferenceArena () noexcept;

  /**
   * @brief Returns the std::pmr::memory_resource pointer for scratch use.
   * @pre None.
   * @post Returned pointer remains valid for the lifetime of InferenceArena.
   * @returns Non-null pointer to monotonic_buffer_resource.
   */
  [[nodiscard]] auto resource () noexcept -> std::pmr::memory_resource *;

  /**
   * @brief Reclaims all scratch allocations in O(1) time.
   * @pre No live PMR containers using resource() may be accessed after reset.
   * @post Monotonic pointer rewinds to start of backing buffer.
   */
  void reset () noexcept;

  /**
   * @brief Queries the pre-allocated backing capacity in bytes.
   * @pre None.
   * @post State is unchanged.
   * @returns Backing buffer size in bytes.
   */
  [[nodiscard]] auto capacity_bytes () const noexcept -> std::size_t;

private:
  std::vector<std::byte> buffer_;
  std::pmr::monotonic_buffer_resource pool_;
};

/**
 * @brief Tier 1 Memory: Fixed-capacity ring buffer for conversation history.
 * @pre capacity > 0.
 * @post Never reallocates the ring buffer after construction.
 * @note Thread-safety: Thread-safe via internal std::shared_mutex.
 */
class ShortTermMemory
{
public:
  /**
   * @brief Creates a fixed-capacity short-term memory ring buffer.
   * @param capacity Maximum number of messages retained in the ring (> 0).
   * @pre capacity > 0.
   * @post Ring buffer storage is pre-allocated to capacity.
   * @returns Result<ShortTermMemory> or
   * FrameworkErrorCode::InvalidConfiguration.
   */
  [[nodiscard]] static auto
  create (std::size_t capacity
          = kDefaultShortTermCapacity) noexcept -> Result<ShortTermMemory>;

  ShortTermMemory () noexcept = default;
  ShortTermMemory (const ShortTermMemory &other);
  auto operator= (const ShortTermMemory &other) -> ShortTermMemory &;
  ShortTermMemory (ShortTermMemory &&other) noexcept;
  auto operator= (ShortTermMemory &&other) noexcept -> ShortTermMemory &;
  ~ShortTermMemory () noexcept = default;

  /**
   * @brief Appends a message to the ring buffer, overwriting oldest if full.
   * @param msg Message to store.
   * @pre capacity() > 0.
   * @post Message is assigned a monotonic sequence_id and stored in O(1).
   * @returns Result<uint64_t> holding the assigned sequence_id.
   * @note Thread-safety: Acquires exclusive write lock on shared_mutex.
   */
  [[nodiscard]] auto append (Message msg) noexcept -> Result<uint64_t>;

  /**
   * @brief Returns a zero-copy two-segment span view of retained messages.
   * @pre Caller must ensure no concurrent append occurs while reading spans,
   *      or hold external read synchronization.
   * @post Zero message copies or allocations are performed.
   * @returns ConversationSpanView referencing contiguous ring slices.
   */
  [[nodiscard]] auto borrow_view () const noexcept -> ConversationSpanView;

  /**
   * @brief Collects messages with sequence_id > since_seq for checkpointing.
   * @param since_seq Last checkpointed sequence number.
   * @pre None.
   * @post State is unchanged.
   * @returns Vector of newly appended messages since since_seq.
   * @note Thread-safety: Acquires shared read lock.
   */
  [[nodiscard]] auto
  delta_since (uint64_t since_seq) const -> std::vector<Message>;

  /**
   * @brief Returns the number of messages currently in the ring buffer.
   * @pre None.
   * @post State is unchanged.
   * @returns Count in [0, capacity()].
   * @note Thread-safety: Acquires shared read lock.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Returns the fixed ring capacity.
   * @pre None.
   * @post State is unchanged.
   * @returns Maximum number of messages stored.
   */
  [[nodiscard]] auto capacity () const noexcept -> std::size_t;

  /**
   * @brief Returns the total number of messages ever appended.
   * @pre None.
   * @post State is unchanged.
   * @returns Monotonic count of appended messages.
   */
  [[nodiscard]] auto total_appended () const noexcept -> uint64_t;

  /**
   * @brief Clears all messages without releasing ring capacity.
   * @pre None.
   * @post size() == 0 and capacity() is unchanged.
   */
  void clear () noexcept;

private:
  explicit ShortTermMemory (std::size_t capacity);
  ShortTermMemory (const ShortTermMemory &other,
                   const std::shared_lock<std::shared_mutex> &lock);
  ShortTermMemory (ShortTermMemory &&other,
                   const std::unique_lock<std::shared_mutex> &lock) noexcept;

  mutable std::shared_mutex mtx_;
  std::vector<Message> ring_;
  std::size_t capacity_ = 0U;
  std::size_t head_ = 0U;
  std::size_t count_ = 0U;
  uint64_t next_seq_ = 1U;
};

/**
 * @brief Tier 2 Memory: Persistent cross-session key-value & episodic store.
 * @pre Keys must be non-empty.
 * @post Completely isolated from ShortTermMemory and SemanticMemory.
 * @note Thread-safety: Thread-safe via internal std::shared_mutex.
 */
class LongTermMemory
{
public:
  LongTermMemory () = default;
  LongTermMemory (const LongTermMemory &other);
  auto operator= (const LongTermMemory &other) -> LongTermMemory &;
  LongTermMemory (LongTermMemory &&other) noexcept;
  auto operator= (LongTermMemory &&other) noexcept -> LongTermMemory &;
  ~LongTermMemory () noexcept = default;

  /**
   * @brief Inserts or updates a persistent key-value fact.
   * @param key Non-empty key identifier.
   * @param value Persistent fact string payload.
   * @pre !key.empty().
   * @post Entry is stored in memory.
   * @returns Result<void> or FrameworkErrorCode::InvalidConfiguration.
   * @note Thread-safety: Acquires exclusive write lock.
   */
  [[nodiscard]] auto put (std::string key,
                          std::string value) noexcept -> Result<void>;

  /**
   * @brief Retrieves a persistent fact by key.
   * @param key Key identifier to look up.
   * @pre None.
   * @post State is unchanged.
   * @returns Result<std::string> or FrameworkErrorCode::InvalidConfiguration.
   * @note Thread-safety: Acquires shared read lock.
   */
  [[nodiscard]] auto
  get (std::string_view key) const noexcept -> Result<std::string>;

  /**
   * @brief Removes a key from the store if present.
   * @param key Key identifier to remove.
   * @pre None.
   * @post Key is no longer present.
   * @returns True if an entry was removed, false otherwise.
   * @note Thread-safety: Acquires exclusive write lock.
   */
  auto erase (std::string_view key) noexcept -> bool;

  /**
   * @brief Returns the number of stored entries.
   * @pre None.
   * @post State is unchanged.
   * @returns Entry count.
   * @note Thread-safety: Acquires shared read lock.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Serializes the long-term store to a binary file on disk.
   * @param filepath Destination file path.
   * @pre !filepath.empty().
   * @post File contains binary representation with integrity header.
   * @returns Result<void> or FrameworkError on I/O failure.
   * @note Thread-safety: Acquires shared read lock during snapshot write.
   */
  [[nodiscard]] auto
  save_to_file (std::string_view filepath) const noexcept -> Result<void>;

  /**
   * @brief Loads entries from a binary long-term memory file on disk.
   * @param filepath Source file path.
   * @pre File must have been written by save_to_file().
   * @post Replaces store contents with deserialized entries.
   * @returns Result<void> or FrameworkError on corruption or I/O failure.
   * @note Thread-safety: Acquires exclusive write lock.
   */
  [[nodiscard]] auto
  load_from_file (std::string_view filepath) noexcept -> Result<void>;

  /**
   * @brief Clears all entries from the store.
   * @pre None.
   * @post size() == 0.
   */
  void clear () noexcept;

private:
  LongTermMemory (const LongTermMemory &other,
                  const std::shared_lock<std::shared_mutex> &lock);
  LongTermMemory (LongTermMemory &&other,
                  const std::unique_lock<std::shared_mutex> &lock) noexcept;

  mutable std::shared_mutex mtx_;
  std::unordered_map<std::string, std::string> store_;
};

/**
 * @brief Hardware SIMD instruction set levels detected at runtime.
 * @note Thread-safety: Immutable enum type.
 */
enum class SimdLevel : uint8_t
{
  Scalar = 0,
  SSE4_2 = 1,
  AVX2 = 2,
  AVX512 = 3
};

/**
 * @brief Detects the highest SIMD ISA supported by the host CPU.
 * @pre None.
 * @post State is unchanged.
 * @returns SimdLevel enumerator.
 * @note Thread-safety: Safe for concurrent calls.
 */
[[nodiscard]] auto detect_simd_level () noexcept -> SimdLevel;

/**
 * @brief Computes fp32 dot product using scalar double-precision accumulation.
 * @param a First float vector span.
 * @param b Second float vector span.
 * @pre a.size() == b.size().
 * @post State is unchanged.
 * @returns Dot product of a and b.
 * @note Thread-safety: Pure function; thread-safe.
 */
[[nodiscard]] auto
simd_dot_f32_scalar (std::span<const float> a,
                     std::span<const float> b) noexcept -> float;

/**
 * @brief Computes fp32 dot product using runtime-dispatched AVX-512/AVX2 SIMD.
 * @param a First float vector span.
 * @param b Second float vector span.
 * @pre a.size() == b.size().
 * @post State is unchanged.
 * @returns Dot product of a and b.
 * @note Thread-safety: Pure function; thread-safe.
 */
[[nodiscard]] auto simd_dot_f32 (std::span<const float> a,
                                 std::span<const float> b) noexcept -> float;

/**
 * @brief Represents a retrieved document from SemanticMemory vector search.
 * @note Thread-safety: Value type.
 */
struct SemanticMatch
{
  std::string id;
  std::string text;
  float score = 0.0F;
};

/**
 * @brief Tier 3 Memory: SIMD & GPU-accelerated vector similarity index.
 * @pre dimension > 0.
 * @post Stores contiguous embedding rows for cache-aligned SIMD/GPU scan.
 * @note Thread-safety: Thread-safe via internal std::shared_mutex.
 */
class SemanticMemory
{
public:
  /**
   * @brief Creates a SemanticMemory index for embeddings of fixed dimension.
   * @param dimension Embedding vector length (> 0).
   * @pre dimension > 0.
   * @post Index is ready to accept documents of length dimension.
   * @returns Result<SemanticMemory> or
   * FrameworkErrorCode::InvalidConfiguration.
   */
  [[nodiscard]] static auto
  create (std::size_t dimension
          = kDefaultSemanticDimension) noexcept -> Result<SemanticMemory>;

  SemanticMemory () noexcept = default;
  SemanticMemory (const SemanticMemory &other);
  auto operator= (const SemanticMemory &other) -> SemanticMemory &;
  SemanticMemory (SemanticMemory &&other) noexcept;
  auto operator= (SemanticMemory &&other) noexcept -> SemanticMemory &;
  ~SemanticMemory () noexcept = default;

  /**
   * @brief Indexes a document and its embedding vector.
   * @param id Unique document identifier.
   * @param text Document text payload.
   * @param embedding Float span of length dimension().
   * @pre !id.empty() and embedding.size() == dimension().
   * @post Document is appended to the contiguous vector index.
   * @returns Result<void> or FrameworkErrorCode::InvalidConfiguration.
   * @note Thread-safety: Acquires exclusive write lock.
   */
  [[nodiscard]] auto
  add_document (std::string id, std::string text,
                std::span<const float> embedding) noexcept -> Result<void>;

  /**
   * @brief Searches for the top_k highest dot-product documents.
   * @param query Query embedding of length dimension().
   * @param top_k Maximum number of results to return (> 0).
   * @param prefer_gpu If true and CUDA is available, offloads to GPU Tensor.
   * @pre query.size() == dimension() and top_k > 0.
   * @post State is unchanged.
   * @returns Result<std::vector<SemanticMatch>> sorted descending by score.
   * @note Thread-safety: Acquires shared read lock.
   */
  [[nodiscard]] auto search (std::span<const float> query,
                             std::size_t top_k = kDefaultSemanticTopK,
                             bool prefer_gpu = false) const noexcept
      -> Result<std::vector<SemanticMatch>>;

  /**
   * @brief Returns the number of indexed documents.
   * @pre None.
   * @post State is unchanged.
   * @returns Document count.
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Returns the configured embedding vector dimension.
   * @pre None.
   * @post State is unchanged.
   * @returns Embedding dimension.
   */
  [[nodiscard]] auto dimension () const noexcept -> std::size_t;

  /**
   * @brief Clears all indexed documents and embeddings.
   * @pre None.
   * @post size() == 0.
   */
  void clear () noexcept;

private:
  struct DocumentMeta
  {
    std::string id;
    std::string text;
  };

  explicit SemanticMemory (std::size_t dimension) noexcept;
  SemanticMemory (const SemanticMemory &other,
                  const std::shared_lock<std::shared_mutex> &lock);
  SemanticMemory (SemanticMemory &&other,
                  const std::unique_lock<std::shared_mutex> &lock) noexcept;

  mutable std::shared_mutex mtx_;
  std::size_t dimension_ = kDefaultSemanticDimension;
  std::vector<DocumentMeta> docs_;
  std::vector<float> flat_embeddings_;
};

/**
 * @brief Unified Three-Tier Memory system composing replaceable tiers.
 * @pre All three tiers must be valid.
 * @post Operations on any tier are strictly isolated from the other tiers.
 * @note Thread-safety: Delegates to each tier's synchronization.
 */
class MemorySystem
{
public:
  /**
   * @brief Creates a Three-Tier MemorySystem with given capacities.
   * @param short_term_capacity Ring capacity for Tier 1 ShortTermMemory.
   * @param semantic_dimension Vector dimension for Tier 3 SemanticMemory.
   * @pre short_term_capacity > 0 and semantic_dimension > 0.
   * @post All three memory tiers are initialized and isolated.
   * @returns Result<MemorySystem> or AgentError on invalid configuration.
   */
  [[nodiscard]] static auto
  create (std::size_t short_term_capacity = kDefaultShortTermCapacity,
          std::size_t semantic_dimension
          = kDefaultSemanticDimension) noexcept -> Result<MemorySystem>;

  MemorySystem () = default;

  /**
   * @brief Accesses Tier 1 ShortTermMemory.
   */
  [[nodiscard]] auto short_term () noexcept -> ShortTermMemory &;
  [[nodiscard]] auto short_term () const noexcept -> const ShortTermMemory &;

  /**
   * @brief Accesses Tier 2 LongTermMemory.
   */
  [[nodiscard]] auto long_term () noexcept -> LongTermMemory &;
  [[nodiscard]] auto long_term () const noexcept -> const LongTermMemory &;

  /**
   * @brief Accesses Tier 3 SemanticMemory.
   */
  [[nodiscard]] auto semantic () noexcept -> SemanticMemory &;
  [[nodiscard]] auto semantic () const noexcept -> const SemanticMemory &;

private:
  MemorySystem (ShortTermMemory st, LongTermMemory lt,
                SemanticMemory sem) noexcept;

  ShortTermMemory short_term_;
  LongTermMemory long_term_;
  SemanticMemory semantic_;
};

} /* namespace sutra::inline v1 */

#endif /* SUTRA_MEMORY_HPP */
