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

#ifndef DARSHANA_MEMORY_HPP
#define DARSHANA_MEMORY_HPP

#include <darshana/error.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace darshana::inline v1
{

/**
 * @brief Represents a tightly packed 2D floating-point coordinate pair.
 * @pre None.
 * @post Trivially copyable 8-byte vertex layout matching GPU std430 Point2D.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct DataPoint2D
{
  float x = 0.0F;
  float y = 0.0F;
};

/**
 * @brief Per-frame monotonic PMR scratch arena for tessellation, layout, and
 * text shaping intermediates.
 * @pre capacity_bytes > 0.
 * @post Reclaims all per-frame scratch allocations in O(1) on reset().
 * @note Thread-safety: Owned per-Figure; not intended for unsynchronized
 *       multi-threaded allocation on the same arena instance.
 */
class ScratchArena
{
public:
  /**
   * @brief Constructs a ScratchArena with a pre-allocated backing buffer.
   * @param capacity_bytes Size of the pre-allocated arena slab in bytes.
   * @pre capacity_bytes > 0.
   * @post Backing buffer is allocated once at construction.
   */
  explicit ScratchArena (std::size_t capacity_bytes = 262144U) noexcept;

  ScratchArena (const ScratchArena &) = delete;
  auto operator= (const ScratchArena &) -> ScratchArena & = delete;
  ScratchArena (ScratchArena &&other) noexcept;
  auto operator= (ScratchArena &&other) noexcept -> ScratchArena &;
  ~ScratchArena () noexcept;

  /**
   * @brief Allocates aligned scratch memory from the monotonic arena.
   * @param bytes Number of bytes to allocate.
   * @param alignment Byte alignment (power of two).
   * @pre bytes > 0.
   * @post Increases used_bytes() and updates peak_bytes().
   * @returns Result<void *> pointing to aligned scratch memory.
   * @note Thread-safety: Single-owner per figure.
   */
  [[nodiscard]] auto
  allocate (std::size_t bytes,
            std::size_t alignment
            = alignof (std::max_align_t)) noexcept -> Result<void *>;

  /**
   * @brief Returns a pointer to the underlying std::pmr::memory_resource.
   * @pre None.
   * @post State is unchanged.
   * @returns Non-null pointer to std::pmr::memory_resource.
   * @note Thread-safety: Single-owner per figure.
   */
  [[nodiscard]] auto resource () noexcept -> std::pmr::memory_resource *;

  /**
   * @brief Resets the arena cursor to zero in O(1) without freeing the slab.
   * @pre None.
   * @post used_bytes() == 0 and capacity_bytes() is unchanged.
   * @note Thread-safety: Single-owner per figure.
   */
  void reset () noexcept;

  /**
   * @brief Queries the pre-allocated slab capacity in bytes.
   * @pre None.
   * @post State is unchanged.
   * @returns Capacity of the backing buffer in bytes.
   */
  [[nodiscard]] auto capacity_bytes () const noexcept -> std::size_t;

  /**
   * @brief Queries the bytes allocated in the current frame epoch.
   * @pre None.
   * @post State is unchanged.
   * @returns Currently used bytes in the arena.
   */
  [[nodiscard]] auto used_bytes () const noexcept -> std::size_t;

  /**
   * @brief Queries the high-water mark of arena usage across all frames.
   * @pre None.
   * @post State is unchanged.
   * @returns Maximum used_bytes() observed since construction.
   */
  [[nodiscard]] auto peak_bytes () const noexcept -> std::size_t;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Pre-allocated O(1) ring buffer for 60+ Hz streaming data updates.
 * @pre Capacity > 0.
 * @post Appends execute in O(1) with zero dynamic memory allocations.
 * @note Thread-safety: Safe for single producer / single consumer or external
 *       Figure synchronization.
 */
class StreamingBuffer
{
public:
  StreamingBuffer () noexcept = default;
  StreamingBuffer (const StreamingBuffer &) = default;
  auto operator= (const StreamingBuffer &) -> StreamingBuffer & = default;
  StreamingBuffer (StreamingBuffer &&) noexcept = default;
  auto operator= (StreamingBuffer &&) noexcept -> StreamingBuffer & = default;
  ~StreamingBuffer () noexcept = default;

  /**
   * @brief Creates a pre-allocated StreamingBuffer with fixed point capacity.
   * @param capacity Maximum number of 2D points retained in the ring buffer.
   * @pre capacity > 0.
   * @post Ring buffer storage is pre-allocated; subsequent push calls never
   *       allocate heap memory.
   * @returns Result<StreamingBuffer> or DataError on invalid capacity.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  create (std::size_t capacity) noexcept -> Result<StreamingBuffer>;

  /**
   * @brief Appends a single (x, y) sample in O(1) without heap allocation.
   * @param x Horizontal coordinate value.
   * @param y Vertical coordinate value.
   * @pre Buffer was initialized via create().
   * @post Ring head advances by 1 and revision increments.
   * @returns Result<void> or DataError if uninitialized.
   * @note Thread-safety: Mutates ring state in O(1).
   */
  [[nodiscard]] auto push (float x, float y) noexcept -> Result<void>;

  /**
   * @brief Appends a batch of (x, y) samples in O(batch_size) without
   * allocation.
   * @param xs Span of horizontal coordinates.
   * @param ys Span of vertical coordinates.
   * @pre xs.size() == ys.size() and buffer was initialized via create().
   * @post Ring advances by xs.size() and revision increments.
   * @returns Result<void> or DataError on size mismatch.
   * @note Thread-safety: Mutates ring state without heap allocation.
   */
  [[nodiscard]] auto
  push_batch (std::span<const float> xs,
              std::span<const float> ys) noexcept -> Result<void>;

  /**
   * @brief Copies active points in chronological order into a caller buffer.
   * @param out Destination span of DataPoint2D with size >= size().
   * @pre out.size() >= size().
   * @post First size() elements of out contain ordered points.
   * @returns Number of points copied.
   * @note Thread-safety: Safe for concurrent reads when not mutated.
   */
  [[nodiscard]] auto
  copy_ordered (std::span<DataPoint2D> out) const noexcept -> std::size_t;

  /**
   * @brief Direct contiguous view of the underlying ring storage array.
   * @pre None.
   * @post State is unchanged.
   * @returns Span of all active storage slots (up to size()).
   * @note Thread-safety: Safe for concurrent reads when not mutated.
   */
  [[nodiscard]] auto
  raw_storage () const noexcept -> std::span<const DataPoint2D>;

  /**
   * @brief Queries the current number of valid points in the ring buffer.
   * @pre None.
   * @post State is unchanged.
   * @returns Active point count in [0, capacity()].
   */
  [[nodiscard]] auto size () const noexcept -> std::size_t;

  /**
   * @brief Queries the fixed maximum point capacity of the ring buffer.
   * @pre None.
   * @post State is unchanged.
   * @returns Pre-allocated point capacity.
   */
  [[nodiscard]] auto capacity () const noexcept -> std::size_t;

  /**
   * @brief Queries the monotonic mutation revision counter.
   * @pre None.
   * @post State is unchanged.
   * @returns Total number of push operations performed.
   */
  [[nodiscard]] auto revision () const noexcept -> uint64_t;

  /**
   * @brief Clears all samples in O(1) while retaining pre-allocated memory.
   * @pre None.
   * @post size() == 0 and capacity() is unchanged.
   */
  void clear () noexcept;

private:
  std::vector<DataPoint2D> buffer_;
  std::size_t capacity_ = 0U;
  std::size_t head_ = 0U;
  std::size_t count_ = 0U;
  uint64_t revision_ = 0U;
};

/**
 * @brief Immutable/snapshot columnar dataset optimized for bulk GPU upload and
 * vector export.
 * @pre Columns must contain finite floating-point values.
 * @post Provides contiguous spans and packed 2D point buffers.
 * @note Thread-safety: Safe for concurrent reads across threads.
 */
class SnapshotData
{
public:
  SnapshotData () noexcept = default;
  SnapshotData (const SnapshotData &) = default;
  auto operator= (const SnapshotData &) -> SnapshotData & = default;
  SnapshotData (SnapshotData &&) noexcept = default;
  auto operator= (SnapshotData &&) noexcept -> SnapshotData & = default;
  ~SnapshotData () noexcept = default;

  /**
   * @brief Constructs a SnapshotData instance from x and y coordinate spans.
   * @param x Horizontal coordinate span.
   * @param y Vertical coordinate span.
   * @pre !x.empty() and x.size() == y.size().
   * @post Stores packed 2D points and x/y spans with non-zero revision.
   * @returns Result<SnapshotData> or DataError on empty or mismatched spans.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  from_xy (std::span<const float> x,
           std::span<const float> y) noexcept -> Result<SnapshotData>;

  /**
   * @brief Sets or replaces a named numeric column in the snapshot.
   * @param name Column identifier (e.g. "x", "y", "color", "size").
   * @param values Contiguous span of float values.
   * @pre !name.empty() and !values.empty().
   * @post Column is stored and revision() is incremented.
   * @returns Result<void> or DataError on invalid input.
   * @note Thread-safety: Mutates instance; synchronize external writers.
   */
  [[nodiscard]] auto
  set_column (std::string_view name,
              std::span<const float> values) noexcept -> Result<void>;

  /**
   * @brief Retrieves a named column span if present.
   * @param name Column identifier.
   * @pre None.
   * @post State is unchanged.
   * @returns Result<std::span<const float>> or DataError if column is absent.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto column (std::string_view name) const noexcept
      -> Result<std::span<const float>>;

  /**
   * @brief Retrieves packed 2D points for primary (x, y) channels.
   * @pre Snapshot was created via from_xy() or has "x" and "y" columns.
   * @post State is unchanged.
   * @returns Span of contiguous DataPoint2D elements.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto points () const noexcept -> std::span<const DataPoint2D>;

  /**
   * @brief Queries the number of data points in the primary dataset.
   * @pre None.
   * @post State is unchanged.
   * @returns Number of points.
   */
  [[nodiscard]] auto point_count () const noexcept -> std::size_t;

  /**
   * @brief Queries the monotonic data revision identifier.
   * @pre None.
   * @post State is unchanged.
   * @returns Revision number incremented on each modification.
   */
  [[nodiscard]] auto revision () const noexcept -> uint64_t;

private:
  struct ColumnEntry
  {
    std::string name;
    std::vector<float> values;
  };

  std::vector<ColumnEntry> columns_;
  std::vector<DataPoint2D> packed_points_;
  uint64_t revision_ = 1U;
};

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_MEMORY_HPP */
