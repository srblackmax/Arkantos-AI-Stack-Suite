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

#include <surya/io/video.hpp>
#include <surya/io/codecs.hpp>

#include <algorithm>
#include <filesystem>
#include <utility>

namespace fs = std::filesystem;

namespace surya::inline v1::io
{

Image_Sequence_Stream::Image_Sequence_Stream (
  std::vector<std::string> frame_paths, double frame_rate)
  : paths_ (std::move (frame_paths)),
    fps_ (frame_rate > 0.0 ? frame_rate : 30.0)
{
}

core::Result<Image_Sequence_Stream>
Image_Sequence_Stream::from_directory (const std::string &dir_path,
                                       double frame_rate)
{
  std::error_code ec;
  if (!fs::exists (dir_path, ec) || !fs::is_directory (dir_path, ec))
    {
      return core::make_error (
        core::Error_Code::Io_Error,
        "Image_Sequence_Stream: directory does not exist: " + dir_path);
    }

  std::vector<std::string> files;
  for (const auto &entry : fs::directory_iterator (dir_path, ec))
    {
      if (entry.is_regular_file ())
        {
          const std::string p = entry.path ().string ();
          if (format_from_extension (p) != Image_Format::Auto)
            {
              files.push_back (p);
            }
        }
    }
  std::sort (files.begin (), files.end ());
  return Image_Sequence_Stream (std::move (files), frame_rate);
}

core::Result<bool>
Image_Sequence_Stream::next_frame (core::Dyn_Image &out_frame,
                                   Frame_Metadata &out_meta)
{
  if (cursor_ >= paths_.size ())
    {
      return false;
    }
  auto img_res = read_image (paths_[cursor_]);
  if (!img_res.has_value ())
    {
      return std::unexpected (img_res.error ());
    }
  out_frame = std::move (*img_res);
  out_meta.frame_index = cursor_;
  out_meta.timestamp_sec = static_cast<double> (cursor_) / fps_;
  ++cursor_;
  return true;
}

Memory_Ring_Stream::Memory_Ring_Stream (size_t capacity, double frame_rate)
  : capacity_ (capacity > 0U ? capacity : 1U),
    fps_ (frame_rate > 0.0 ? frame_rate : 60.0)
{
  ring_.resize (capacity_);
}

void
Memory_Ring_Stream::push_frame (core::Dyn_Image frame)
{
  ring_[write_idx_] = std::move (frame);
  write_idx_ = (write_idx_ + 1U) % capacity_;
  if (count_ < capacity_)
    {
      ++count_;
    }
  else
    {
      read_idx_ = (read_idx_ + 1U) % capacity_;
    }
}

core::Result<bool>
Memory_Ring_Stream::next_frame (core::Dyn_Image &out_frame,
                                Frame_Metadata &out_meta)
{
  if (count_ == 0U)
    {
      return false;
    }
  out_frame = ring_[read_idx_].clone ();
  read_idx_ = (read_idx_ + 1U) % capacity_;
  --count_;
  out_meta.frame_index = emitted_;
  out_meta.timestamp_sec = static_cast<double> (emitted_) / fps_;
  ++emitted_;
  return true;
}

Synthetic_Video_Stream::Synthetic_Video_Stream (size_t width,
                                                size_t height,
                                                size_t total_frames,
                                                double frame_rate) noexcept
  : width_ (width),
    height_ (height),
    total_frames_ (total_frames),
    fps_ (frame_rate > 0.0 ? frame_rate : 30.0)
{
}

core::Result<bool>
Synthetic_Video_Stream::next_frame (core::Dyn_Image &out_frame,
                                    Frame_Metadata &out_meta)
{
  if (cursor_ >= total_frames_)
    {
      return false;
    }
  if (out_frame.empty () || out_frame.width () != width_
      || out_frame.height () != height_ || out_frame.channels () != 3U)
    {
      auto res = core::Dyn_Image::create (width_, height_, 3U);
      if (!res.has_value ())
        {
          return std::unexpected (res.error ());
        }
      out_frame = std::move (*res);
    }

  for (size_t y = 0U; y < height_; ++y)
    {
      uint8_t *row = out_frame.raw_row_ptr (y);
      std::memset (row, 16, width_ * 3U);
    }

  /* Render two moving target boxes across the frame.  */
  const size_t box_w = std::max<size_t> (8U, width_ / 10U);
  const size_t box_h = std::max<size_t> (8U, height_ / 10U);
  const size_t x1 =
    (cursor_ * 4U + 10U) % std::max<size_t> (1U, width_ - box_w);
  const size_t y1 =
    (cursor_ * 2U + 10U) % std::max<size_t> (1U, height_ - box_h);

  for (size_t y = y1; y < std::min (y1 + box_h, height_); ++y)
    {
      uint8_t *row = out_frame.raw_row_ptr (y);
      for (size_t x = x1; x < std::min (x1 + box_w, width_); ++x)
        {
          row[x * 3U + 0U] = 240U;
          row[x * 3U + 1U] = 80U;
          row[x * 3U + 2U] = 40U;
        }
    }

  out_meta.frame_index = cursor_;
  out_meta.timestamp_sec = static_cast<double> (cursor_) / fps_;
  ++cursor_;
  return true;
}

} /* namespace surya::inline v1::io */
