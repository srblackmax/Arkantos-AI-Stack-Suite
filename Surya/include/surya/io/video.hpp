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

#ifndef SURYA_IO_VIDEO_HPP
#define SURYA_IO_VIDEO_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <surya/core.hpp>

namespace surya::inline v1::io
{

/**
 * @brief Metadata for a decoded video frame.
 */
struct Frame_Metadata
{
  size_t frame_index = 0U;
  double timestamp_sec = 0.0;
};

/**
 * @brief Abstract base interface for sequential and real-time video frame
 *        ingestion pipelines.
 */
class SURYA_API Frame_Stream
{
public:
  virtual ~Frame_Stream (void) noexcept = default;

  /**
   * @brief Fetch the next frame into OUT_FRAME and its metadata into OUT_META.
   *
   * Returns true if a frame was produced, false on end-of-stream, or Error
   * on I/O failure.
   */
  virtual core::Result<bool>
  next_frame (core::Dyn_Image &out_frame, Frame_Metadata &out_meta) = 0;

  virtual void
  reset (void) noexcept = 0;

  virtual double
  fps (void) const noexcept = 0;
};

/**
 * @brief Video stream backed by an ordered sequence of image file paths.
 */
class SURYA_API Image_Sequence_Stream : public Frame_Stream
{
public:
  explicit Image_Sequence_Stream (std::vector<std::string> frame_paths,
                                  double frame_rate = 30.0);

  static core::Result<Image_Sequence_Stream>
  from_directory (const std::string &dir_path, double frame_rate = 30.0);

  core::Result<bool>
  next_frame (core::Dyn_Image &out_frame, Frame_Metadata &out_meta) override;

  void
  reset (void) noexcept override
  {
    cursor_ = 0U;
  }

  double
  fps (void) const noexcept override
  {
    return fps_;
  }

  size_t
  total_frames (void) const noexcept
  {
    return paths_.size ();
  }

private:
  std::vector<std::string> paths_;
  double fps_ = 30.0;
  size_t cursor_ = 0U;
};

/**
 * @brief Low-latency in-memory ring stream of pre-loaded or camera-pushed
 *        frames.
 */
class SURYA_API Memory_Ring_Stream : public Frame_Stream
{
public:
  explicit Memory_Ring_Stream (size_t capacity = 32U,
                               double frame_rate = 60.0);

  void
  push_frame (core::Dyn_Image frame);

  core::Result<bool>
  next_frame (core::Dyn_Image &out_frame, Frame_Metadata &out_meta) override;

  void
  reset (void) noexcept override
  {
    read_idx_ = 0U;
    emitted_ = 0U;
  }

  double
  fps (void) const noexcept override
  {
    return fps_;
  }

  size_t
  buffered_count (void) const noexcept
  {
    return count_;
  }

private:
  std::vector<core::Dyn_Image> ring_;
  size_t capacity_ = 0U;
  size_t write_idx_ = 0U;
  size_t read_idx_ = 0U;
  size_t count_ = 0U;
  size_t emitted_ = 0U;
  double fps_ = 60.0;
};

/**
 * @brief Deterministic procedural video stream rendering moving geometric
 *        targets for reproducible optical-flow and multi-object tracker tests.
 */
class SURYA_API Synthetic_Video_Stream : public Frame_Stream
{
public:
  Synthetic_Video_Stream (size_t width,
                          size_t height,
                          size_t total_frames,
                          double frame_rate = 30.0) noexcept;

  core::Result<bool>
  next_frame (core::Dyn_Image &out_frame, Frame_Metadata &out_meta) override;

  void
  reset (void) noexcept override
  {
    cursor_ = 0U;
  }

  double
  fps (void) const noexcept override
  {
    return fps_;
  }

private:
  size_t width_ = 0U;
  size_t height_ = 0U;
  size_t total_frames_ = 0U;
  size_t cursor_ = 0U;
  double fps_ = 30.0;
};

} /* namespace surya::inline v1::io */

#endif /* SURYA_IO_VIDEO_HPP */
