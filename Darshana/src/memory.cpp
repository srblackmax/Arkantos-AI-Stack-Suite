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

#include <darshana/memory.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace darshana::inline v1
{

struct ScratchArena::Impl
{
  std::vector<std::byte> slab;
  std::pmr::monotonic_buffer_resource mbr;
  std::size_t capacity = 0U;
  std::size_t used = 0U;
  std::size_t peak = 0U;

  explicit Impl (std::size_t cap)
      : slab (cap > 0U ? cap : 4096U),
        mbr (slab.data (), slab.size (), std::pmr::null_memory_resource ()),
        capacity (slab.size ())
  {
  }
};

/* Constructs a ScratchArena with a pre-allocated backing buffer.  */
ScratchArena::ScratchArena (std::size_t capacity_bytes) noexcept
{
  try
    {
      impl_ = std::make_unique<Impl> (capacity_bytes);
    }
  catch (...)
    {
      impl_.reset ();
    }
}

ScratchArena::ScratchArena (ScratchArena &&other) noexcept = default;

auto ScratchArena::operator= (ScratchArena &&other) noexcept
    -> ScratchArena & = default;

ScratchArena::~ScratchArena () noexcept = default;

/* Allocates aligned scratch memory from the monotonic arena.  */
auto
ScratchArena::allocate (std::size_t bytes,
                        std::size_t alignment) noexcept -> Result<void *>
{
  if (!impl_ || bytes == 0U)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0,
          "ScratchArena allocation requested with zero bytes or invalid "
          "arena"));
    }
  const std::size_t mask = alignment > 0U ? (alignment - 1U) : 0U;
  const std::size_t aligned_used = (impl_->used + mask) & ~mask;
  if (aligned_used + bytes > impl_->capacity)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "ScratchArena pre-allocated slab capacity exhausted"));
    }
  try
    {
      void *ptr = impl_->mbr.allocate (bytes, alignment);
      impl_->used = aligned_used + bytes;
      impl_->peak = std::max (impl_->peak, impl_->used);
      return ptr;
    }
  catch (...)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "ScratchArena std::pmr::monotonic_buffer_resource exhausted"));
    }
}

/* Returns a pointer to the underlying std::pmr::memory_resource.  */
auto
ScratchArena::resource () noexcept -> std::pmr::memory_resource *
{
  return impl_ ? &impl_->mbr : std::pmr::null_memory_resource ();
}

/* Resets the arena cursor to zero in O(1) without freeing the slab.  */
void
ScratchArena::reset () noexcept
{
  if (impl_)
    {
      impl_->mbr.release ();
      impl_->used = 0U;
    }
}

/* Queries the pre-allocated slab capacity in bytes.  */
auto
ScratchArena::capacity_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->capacity : 0U;
}

/* Queries the bytes allocated in the current frame epoch.  */
auto
ScratchArena::used_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->used : 0U;
}

/* Queries the high-water mark of arena usage across all frames.  */
auto
ScratchArena::peak_bytes () const noexcept -> std::size_t
{
  return impl_ ? impl_->peak : 0U;
}

/* Creates a pre-allocated StreamingBuffer with fixed point capacity.  */
auto
StreamingBuffer::create (std::size_t capacity) noexcept
    -> Result<StreamingBuffer>
{
  if (capacity == 0U)
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "capacity",
                       "StreamingBuffer capacity must be greater than zero"));
    }
  try
    {
      StreamingBuffer buf;
      buf.buffer_.resize (capacity);
      buf.capacity_ = capacity;
      buf.head_ = 0U;
      buf.count_ = 0U;
      buf.revision_ = 1U;
      return buf;
    }
  catch (...)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Failed to allocate StreamingBuffer backing storage"));
    }
}

/* Appends a single (x, y) sample in O(1) without heap allocation.  */
auto
StreamingBuffer::push (float x, float y) noexcept -> Result<void>
{
  if (capacity_ == 0U || buffer_.empty ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "StreamingBuffer",
                       "Cannot push to an uninitialized StreamingBuffer"));
    }
  if (!std::isfinite (x) || !std::isfinite (y))
    {
      return std::unexpected (Error::data (
          DataErrorCode::UnsupportedDataType, "xy",
          "StreamingBuffer requires finite floating-point coordinates"));
    }
  buffer_[head_] = DataPoint2D{ x, y };
  head_ = (head_ + 1U) % capacity_;
  if (count_ < capacity_)
    {
      ++count_;
    }
  ++revision_;
  return {};
}

/* Appends a batch of (x, y) samples in O(batch_size) without allocation.  */
auto
StreamingBuffer::push_batch (std::span<const float> xs,
                             std::span<const float> ys) noexcept
    -> Result<void>
{
  if (capacity_ == 0U || buffer_.empty ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidSpec, "StreamingBuffer",
          "Cannot push batch to an uninitialized StreamingBuffer"));
    }
  if (xs.size () != ys.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "xs/ys",
          "StreamingBuffer push_batch requires equal x and y lengths"));
    }
  if (xs.empty ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::EmptyDataset, "xs",
                       "StreamingBuffer push_batch received an empty span"));
    }
  for (std::size_t i = 0U; i < xs.size (); ++i)
    {
      if (!std::isfinite (xs[i]) || !std::isfinite (ys[i]))
        {
          return std::unexpected (Error::data (
              DataErrorCode::UnsupportedDataType, "xy",
              "StreamingBuffer batch contains non-finite coordinates"));
        }
      buffer_[head_] = DataPoint2D{ xs[i], ys[i] };
      head_ = (head_ + 1U) % capacity_;
      if (count_ < capacity_)
        {
          ++count_;
        }
    }
  ++revision_;
  return {};
}

/* Copies active points in chronological order into a caller buffer.  */
auto
StreamingBuffer::copy_ordered (std::span<DataPoint2D> out) const noexcept
    -> std::size_t
{
  const std::size_t n = std::min (count_, out.size ());
  if (n == 0U)
    {
      return 0U;
    }
  if (count_ < capacity_)
    {
      std::memcpy (out.data (), buffer_.data (), n * sizeof (DataPoint2D));
      return n;
    }
  const std::size_t start = head_;
  const std::size_t first_part = std::min (n, capacity_ - start);
  std::memcpy (out.data (), buffer_.data () + start,
               first_part * sizeof (DataPoint2D));
  if (n > first_part)
    {
      std::memcpy (out.data () + first_part, buffer_.data (),
                   (n - first_part) * sizeof (DataPoint2D));
    }
  return n;
}

/* Direct contiguous view of the underlying ring storage array.  */
auto
StreamingBuffer::raw_storage () const noexcept -> std::span<const DataPoint2D>
{
  return std::span<const DataPoint2D> (buffer_.data (), count_);
}

/* Queries current number of valid points in the ring buffer.  */
auto
StreamingBuffer::size () const noexcept -> std::size_t
{
  return count_;
}

/* Queries fixed maximum point capacity of the ring buffer.  */
auto
StreamingBuffer::capacity () const noexcept -> std::size_t
{
  return capacity_;
}

/* Queries monotonic mutation revision counter.  */
auto
StreamingBuffer::revision () const noexcept -> uint64_t
{
  return revision_;
}

/* Clears all samples in O(1) while retaining pre-allocated memory.  */
void
StreamingBuffer::clear () noexcept
{
  head_ = 0U;
  count_ = 0U;
  ++revision_;
}

/* Constructs a SnapshotData instance from x and y coordinate spans.  */
auto
SnapshotData::from_xy (std::span<const float> x,
                       std::span<const float> y) noexcept
    -> Result<SnapshotData>
{
  if (x.empty () || y.empty ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::EmptyDataset, "xy",
          "SnapshotData::from_xy requires non-empty coordinate spans"));
    }
  if (x.size () != y.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "xy",
          "SnapshotData::from_xy requires x and y spans of equal length"));
    }
  try
    {
      SnapshotData snap;
      const std::size_t n = x.size ();
      snap.packed_points_.resize (n);
      DataPoint2D *dst = snap.packed_points_.data ();
      const float *px = x.data ();
      const float *py = y.data ();
      for (std::size_t i = 0U; i < n; ++i)
        {
          dst[i].x = px[i];
          dst[i].y = py[i];
        }
      snap.revision_ = 1U;
      return snap;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Failed to allocate SnapshotData point buffer"));
    }
}

/* Sets or replaces a named numeric column in the snapshot.  */
auto
SnapshotData::set_column (std::string_view name,
                          std::span<const float> values) noexcept
    -> Result<void>
{
  if (name.empty ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "name",
                       "SnapshotData column name must not be empty"));
    }
  if (values.empty ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::EmptyDataset, std::string (name),
                       "SnapshotData column values must not be empty"));
    }
  try
    {
      for (auto &col : columns_)
        {
          if (col.name == name)
            {
              col.values.assign (values.begin (), values.end ());
              ++revision_;
              return {};
            }
        }
      columns_.push_back (ColumnEntry{
          std::string (name),
          std::vector<float> (values.begin (), values.end ()),
      });
      auto x_col = column ("x");
      auto y_col = column ("y");
      if (x_col.has_value () && y_col.has_value ()
          && x_col->size () == y_col->size ())
        {
          const std::size_t n = x_col->size ();
          packed_points_.resize (n);
          for (std::size_t i = 0U; i < n; ++i)
            {
              packed_points_[i] = DataPoint2D{ (*x_col)[i], (*y_col)[i] };
            }
        }
      ++revision_;
      return {};
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Failed to allocate SnapshotData column vector"));
    }
}

/* Retrieves a named column span if present.  */
auto
SnapshotData::column (std::string_view name) const noexcept
    -> Result<std::span<const float>>
{
  for (const auto &col : columns_)
    {
      if (col.name == name)
        {
          return std::span<const float> (col.values.data (),
                                         col.values.size ());
        }
    }
  return std::unexpected (
      Error::data (DataErrorCode::InvalidSpec, std::string (name),
                   "Requested column does not exist in SnapshotData"));
}

/* Retrieves packed 2D points for primary (x, y) channels.  */
auto
SnapshotData::points () const noexcept -> std::span<const DataPoint2D>
{
  return std::span<const DataPoint2D> (packed_points_.data (),
                                       packed_points_.size ());
}

/* Queries the number of data points in the primary dataset.  */
auto
SnapshotData::point_count () const noexcept -> std::size_t
{
  return packed_points_.size ();
}

/* Queries the monotonic data revision identifier.  */
auto
SnapshotData::revision () const noexcept -> uint64_t
{
  return revision_;
}

} /* namespace darshana::inline v1 */
