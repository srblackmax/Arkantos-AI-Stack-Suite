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

#include <surya/io.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>

using namespace surya;
using namespace surya::core;
using namespace surya::io;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_io.cpp:" << line << "]: " << expr << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_memory_codecs (void)
{
  /* Odd width (29x19) tests 64-byte row-stride handling across all codecs.  */
  Image_U8C3 orig (29U, 19U, false);
  for (size_t y = 0U; y < orig.height (); ++y)
    {
      for (size_t x = 0U; x < orig.width (); ++x)
        {
          orig.at (y, x, 0U) =
            static_cast<uint8_t> ((x * 7U + y * 3U) & 0xFFU);
          orig.at (y, x, 1U) =
            static_cast<uint8_t> ((x * 5U + y * 9U) & 0xFFU);
          orig.at (y, x, 2U) =
            static_cast<uint8_t> ((x * 11U + y * 2U) & 0xFFU);
        }
    }
  Dyn_Image dyn_orig = Dyn_Image::from_view (orig.const_view ());

  /* Lossless codecs: PNM, BMP, QOI, PNG.  */
  const Image_Format lossless_fmts[] = {Image_Format::PNM, Image_Format::BMP,
                                        Image_Format::QOI, Image_Format::PNG};
  for (Image_Format fmt : lossless_fmts)
    {
      auto enc = encode_image_to_memory (dyn_orig, fmt);
      CHECK (enc.has_value ());
      CHECK (detect_image_format (*enc) == fmt);

      auto dec = decode_image_from_memory (*enc);
      CHECK (dec.has_value ());
      CHECK (dec->width () == 29U);
      CHECK (dec->height () == 19U);
      CHECK (dec->channels () == 3U);

      auto v = dec->try_const_view<uint8_t, 3U> ();
      CHECK (v.has_value ());
      CHECK (v->is_row_aligned (64U));
      CHECK (v->at (10U, 15U, 0U) == orig.at (10U, 15U, 0U));
      CHECK (v->at (10U, 15U, 1U) == orig.at (10U, 15U, 1U));
      CHECK (v->at (10U, 15U, 2U) == orig.at (10U, 15U, 2U));
    }

  /* Lossy/Near-lossless codecs: JPEG (quality 98) and WebP (quality 98).  */
  Image_U8C3 smooth (32U, 32U, Scalar<uint8_t, 3U> (80U, 150U, 200U));
  Dyn_Image dyn_smooth = Dyn_Image::from_view (smooth.const_view ());

  const Image_Format lossy_fmts[] = {Image_Format::JPEG, Image_Format::WebP};
  for (Image_Format fmt : lossy_fmts)
    {
      auto enc = encode_image_to_memory (dyn_smooth, fmt, 98);
      CHECK (enc.has_value ());
      CHECK (detect_image_format (*enc) == fmt);

      auto dec = decode_image_from_memory (*enc);
      CHECK (dec.has_value ());
      auto v = dec->try_const_view<uint8_t, 3U> ();
      CHECK (v.has_value ());
      CHECK (std::abs (static_cast<int> (v->at (8U, 8U, 0U)) - 80) <= 4);
      CHECK (std::abs (static_cast<int> (v->at (8U, 8U, 1U)) - 150) <= 4);
      CHECK (std::abs (static_cast<int> (v->at (8U, 8U, 2U)) - 200) <= 4);
    }
}

void
test_disk_io_and_tiff (void)
{
  Image_U8C3 img (24U, 16U, Scalar<uint8_t, 3U> (42U, 128U, 210U));
  Dyn_Image dyn = Dyn_Image::from_view (img.const_view ());

  const std::string tiff_path = "/tmp/surya_test_image.tiff";
  const std::string png_path = "/tmp/surya_test_image.png";

  auto w_tif = write_image (tiff_path, dyn);
  CHECK (w_tif.has_value ());
  auto r_tif = read_image_rgb (tiff_path);
  CHECK (r_tif.has_value ());
  CHECK (r_tif->at (5U, 5U, 0U) == 42U);
  CHECK (r_tif->at (5U, 5U, 1U) == 128U);
  CHECK (r_tif->at (5U, 5U, 2U) == 210U);

  auto w_png = write_image (png_path, dyn);
  CHECK (w_png.has_value ());
  auto r_png = read_image_rgb (png_path);
  CHECK (r_png.has_value ());
  CHECK (r_png->at (5U, 5U, 0U) == 42U);

  std::filesystem::remove (tiff_path);
  std::filesystem::remove (png_path);
}

void
test_video_streams (void)
{
  Synthetic_Video_Stream synth (160U, 120U, 5U, 25.0);
  Dyn_Image frame;
  Frame_Metadata meta;
  size_t count = 0U;
  for (;;)
    {
      auto ok = synth.next_frame (frame, meta);
      CHECK (ok.has_value ());
      if (!*ok)
        {
          break;
        }
      CHECK (frame.width () == 160U);
      CHECK (frame.height () == 120U);
      CHECK (meta.frame_index == count);
      ++count;
    }
  CHECK (count == 5U);

  Memory_Ring_Stream ring (4U, 60.0);
  for (size_t i = 0U; i < 6U; ++i)
    {
      Dyn_Image f (32U, 32U, 1U);
      f.raw_row_ptr (0U)[0] = static_cast<uint8_t> (i + 10U);
      ring.push_frame (std::move (f));
    }
  CHECK (ring.buffered_count () == 4U);
  auto rf = ring.next_frame (frame, meta);
  CHECK (rf.has_value () && *rf);
  /* Oldest 2 frames (10, 11) were overwritten by capacity-4 ring.  */
  CHECK (frame.raw_row_ptr (0U)[0] == 12U);
}

} /* namespace */

int
main (void)
{
  test_memory_codecs ();
  test_disk_io_and_tiff ();
  test_video_streams ();

  if (g_failures > 0)
    {
      std::cerr << "test_io FAILED with " << g_failures << " error(s).\n";
      return 1;
    }
  std::cout << "test_io PASSED.\n";
  return 0;
}
