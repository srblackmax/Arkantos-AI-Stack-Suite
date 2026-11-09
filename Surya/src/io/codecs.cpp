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

#include <surya/io/codecs.hpp>
#include <surya/imgproc/color.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <jpeglib.h>
#include <png.h>
#include <tiffio.h>
#include <webp/decode.h>
#include <webp/encode.h>

namespace surya::inline v1::io
{

Image_Format
detect_image_format (std::span<const uint8_t> bytes) noexcept
{
  if (bytes.size () < 4U)
    {
      return Image_Format::Auto;
    }
  if (bytes[0] == 0xFFU && bytes[1] == 0xD8U && bytes[2] == 0xFFU)
    {
      return Image_Format::JPEG;
    }
  if (bytes.size () >= 8U && bytes[0] == 0x89U && bytes[1] == 'P'
      && bytes[2] == 'N' && bytes[3] == 'G' && bytes[4] == 0x0DU
      && bytes[5] == 0x0AU && bytes[6] == 0x1AU && bytes[7] == 0x0AU)
    {
      return Image_Format::PNG;
    }
  if (bytes.size () >= 12U && bytes[0] == 'R' && bytes[1] == 'I'
      && bytes[2] == 'F' && bytes[3] == 'F' && bytes[8] == 'W'
      && bytes[9] == 'E' && bytes[10] == 'B' && bytes[11] == 'P')
    {
      return Image_Format::WebP;
    }
  if ((bytes[0] == 'I' && bytes[1] == 'I' && bytes[2] == 42U
       && bytes[3] == 0U)
      || (bytes[0] == 'M' && bytes[1] == 'M' && bytes[2] == 0U
          && bytes[3] == 42U))
    {
      return Image_Format::TIFF;
    }
  if (bytes[0] == 'q' && bytes[1] == 'o' && bytes[2] == 'i'
      && bytes[3] == 'f')
    {
      return Image_Format::QOI;
    }
  if (bytes[0] == 'B' && bytes[1] == 'M')
    {
      return Image_Format::BMP;
    }
  if (bytes[0] == 'P' && (bytes[1] == '5' || bytes[1] == '6'))
    {
      return Image_Format::PNM;
    }
  return Image_Format::Auto;
}

Image_Format
format_from_extension (std::string_view path) noexcept
{
  const size_t dot = path.find_last_of ('.');
  if (dot == std::string_view::npos)
    {
      return Image_Format::Auto;
    }
  std::string ext (path.substr (dot + 1U));
  for (char &c : ext)
    {
      c = static_cast<char> (std::tolower (static_cast<unsigned char> (c)));
    }
  if (ext == "png")
    return Image_Format::PNG;
  if (ext == "jpg" || ext == "jpeg")
    return Image_Format::JPEG;
  if (ext == "webp")
    return Image_Format::WebP;
  if (ext == "tif" || ext == "tiff")
    return Image_Format::TIFF;
  if (ext == "bmp")
    return Image_Format::BMP;
  if (ext == "qoi")
    return Image_Format::QOI;
  if (ext == "pgm" || ext == "ppm" || ext == "pnm")
    return Image_Format::PNM;
  return Image_Format::Auto;
}

namespace
{

constexpr size_t kMaxDimension = 32768U;

/* =========================================================================
   1. Clean-Room PNM (PGM P5 / PPM P6) Codec
   ========================================================================= */

bool
parse_pnm_uint (std::span<const uint8_t> bytes,
                size_t &pos,
                size_t &out_val) noexcept
{
  const size_t n = bytes.size ();
  while (pos < n)
    {
      if (bytes[pos] == '#')
        {
          while (pos < n && bytes[pos] != '\n')
            {
              ++pos;
            }
        }
      else if (std::isspace (static_cast<unsigned char> (bytes[pos])))
        {
          ++pos;
        }
      else
        {
          break;
        }
    }
  if (pos >= n || !std::isdigit (static_cast<unsigned char> (bytes[pos])))
    {
      return false;
    }
  size_t val = 0U;
  while (pos < n && std::isdigit (static_cast<unsigned char> (bytes[pos])))
    {
      val = val * 10U + static_cast<size_t> (bytes[pos] - '0');
      if (val > 100000000U)
        {
          return false;
        }
      ++pos;
    }
  out_val = val;
  return true;
}

core::Result<core::Dyn_Image>
decode_pnm (std::span<const uint8_t> bytes)
{
  if (bytes.size () < 7U || bytes[0] != 'P'
      || (bytes[1] != '5' && bytes[1] != '6'))
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_pnm: invalid PNM magic header");
    }
  const size_t channels = (bytes[1] == '5') ? 1U : 3U;
  size_t pos = 2U;
  size_t width = 0U;
  size_t height = 0U;
  size_t maxval = 0U;

  if (!parse_pnm_uint (bytes, pos, width)
      || !parse_pnm_uint (bytes, pos, height)
      || !parse_pnm_uint (bytes, pos, maxval))
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_pnm: malformed header tokens");
    }
  if (width == 0U || height == 0U || width > kMaxDimension
      || height > kMaxDimension || maxval != 255U || pos >= bytes.size ()
      || !std::isspace (static_cast<unsigned char> (bytes[pos])))
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_pnm: unsupported PNM dimensions");
    }
  ++pos; /* Single whitespace byte after maxval.  */

  const size_t row_bytes = width * channels;
  const size_t payload_bytes = row_bytes * height;
  if (bytes.size () - pos < payload_bytes)
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_pnm: truncated PNM pixel payload");
    }

  auto img_res = core::Dyn_Image::create (width, height, channels);
  if (!img_res.has_value ())
    {
      return img_res;
    }
  core::Dyn_Image img = std::move (*img_res);
  const uint8_t *src_ptr = bytes.data () + pos;
  for (size_t y = 0U; y < height; ++y)
    {
      std::memcpy (img.raw_row_ptr (y), src_ptr + y * row_bytes, row_bytes);
    }
  return img;
}

core::Result<std::vector<uint8_t>>
encode_pnm (const core::Dyn_Image &img)
{
  if (img.empty () || img.depth () != core::Pixel_Depth::U8
      || (img.channels () != 1U && img.channels () != 3U))
    {
      return core::make_error (
        core::Error_Code::Format_Mismatch,
        "encode_pnm: requires U8C1 (PGM) or U8C3 (PPM) image");
    }
  const char magic = (img.channels () == 1U) ? '5' : '6';
  std::string hdr = std::string ("P") + magic + "\n"
                    + std::to_string (img.width ()) + " "
                    + std::to_string (img.height ()) + "\n255\n";
  const size_t row_bytes = img.width () * img.channels ();
  std::vector<uint8_t> out (hdr.size () + row_bytes * img.height ());
  std::memcpy (out.data (), hdr.data (), hdr.size ());
  uint8_t *dst = out.data () + hdr.size ();
  for (size_t y = 0U; y < img.height (); ++y)
    {
      std::memcpy (dst + y * row_bytes, img.raw_row_ptr (y), row_bytes);
    }
  return out;
}

/* =========================================================================
   2. Clean-Room Uncompressed 24-bit / 32-bit BMP Codec
   ========================================================================= */

uint16_t
read_le16 (const uint8_t *p) noexcept
{
  return static_cast<uint16_t> (static_cast<uint16_t> (p[0])
                                | (static_cast<uint16_t> (p[1]) << 8U));
}

uint32_t
read_le32 (const uint8_t *p) noexcept
{
  return static_cast<uint32_t> (p[0]) | (static_cast<uint32_t> (p[1]) << 8U)
         | (static_cast<uint32_t> (p[2]) << 16U)
         | (static_cast<uint32_t> (p[3]) << 24U);
}

void
write_le16 (uint8_t *p, uint16_t v) noexcept
{
  p[0] = static_cast<uint8_t> (v & 0xFFU);
  p[1] = static_cast<uint8_t> ((v >> 8U) & 0xFFU);
}

void
write_le32 (uint8_t *p, uint32_t v) noexcept
{
  p[0] = static_cast<uint8_t> (v & 0xFFU);
  p[1] = static_cast<uint8_t> ((v >> 8U) & 0xFFU);
  p[2] = static_cast<uint8_t> ((v >> 16U) & 0xFFU);
  p[3] = static_cast<uint8_t> ((v >> 24U) & 0xFFU);
}

core::Result<core::Dyn_Image>
decode_bmp (std::span<const uint8_t> bytes)
{
  if (bytes.size () < 54U || bytes[0] != 'B' || bytes[1] != 'M')
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_bmp: truncated or invalid BMP header");
    }
  const uint32_t data_offset = read_le32 (bytes.data () + 10U);
  const int32_t w_signed =
    static_cast<int32_t> (read_le32 (bytes.data () + 18U));
  const int32_t h_signed =
    static_cast<int32_t> (read_le32 (bytes.data () + 22U));
  const uint16_t bpp = read_le16 (bytes.data () + 28U);
  const uint32_t compression = read_le32 (bytes.data () + 30U);

  if (w_signed <= 0 || h_signed == 0 || compression != 0U
      || (bpp != 24U && bpp != 32U))
    {
      return core::make_error (
        core::Error_Code::Codec_Error,
        "decode_bmp: only uncompressed 24/32-bit BMP supported");
    }

  const bool top_down = (h_signed < 0);
  const size_t width = static_cast<size_t> (w_signed);
  const size_t height =
    static_cast<size_t> (top_down ? -h_signed : h_signed);
  if (width > kMaxDimension || height > kMaxDimension)
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_bmp: BMP dimensions exceed limit");
    }

  const size_t channels = (bpp == 32U) ? 4U : 3U;
  const size_t bmp_row_stride = ((width * channels + 3U) / 4U) * 4U;
  const size_t total_payload = bmp_row_stride * height;
  if (data_offset > bytes.size ()
      || bytes.size () - data_offset < total_payload)
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_bmp: truncated BMP pixel array");
    }

  auto img_res = core::Dyn_Image::create (width, height, channels);
  if (!img_res.has_value ())
    {
      return img_res;
    }
  core::Dyn_Image img = std::move (*img_res);
  const uint8_t *base = bytes.data () + data_offset;

  for (size_t y = 0U; y < height; ++y)
    {
      const size_t src_y = top_down ? y : (height - 1U - y);
      const uint8_t *s_row = base + src_y * bmp_row_stride;
      uint8_t *d_row = img.raw_row_ptr (y);
      for (size_t x = 0U; x < width; ++x)
        {
          /* Convert BGR(A) -> RGB(A).  */
          d_row[x * channels + 0U] = s_row[x * channels + 2U];
          d_row[x * channels + 1U] = s_row[x * channels + 1U];
          d_row[x * channels + 2U] = s_row[x * channels + 0U];
          if (channels == 4U)
            {
              d_row[x * channels + 3U] = s_row[x * channels + 3U];
            }
        }
    }
  return img;
}

core::Result<std::vector<uint8_t>>
encode_bmp (const core::Dyn_Image &img)
{
  if (img.empty () || img.depth () != core::Pixel_Depth::U8
      || (img.channels () != 3U && img.channels () != 4U))
    {
      return core::make_error (core::Error_Code::Format_Mismatch,
                               "encode_bmp: requires U8C3 or U8C4 image");
    }
  const size_t w = img.width ();
  const size_t h = img.height ();
  const size_t c = img.channels ();
  const size_t bmp_row_stride = ((w * c + 3U) / 4U) * 4U;
  const size_t payload_bytes = bmp_row_stride * h;
  std::vector<uint8_t> out (54U + payload_bytes, 0U);

  out[0] = 'B';
  out[1] = 'M';
  write_le32 (out.data () + 2U, static_cast<uint32_t> (out.size ()));
  write_le32 (out.data () + 10U, 54U);
  write_le32 (out.data () + 14U, 40U);
  write_le32 (out.data () + 18U, static_cast<uint32_t> (w));
  write_le32 (out.data () + 22U, static_cast<uint32_t> (h));
  write_le16 (out.data () + 26U, 1U);
  write_le16 (out.data () + 28U, static_cast<uint16_t> (c * 8U));
  write_le32 (out.data () + 34U, static_cast<uint32_t> (payload_bytes));

  uint8_t *dst_base = out.data () + 54U;
  for (size_t y = 0U; y < h; ++y)
    {
      const uint8_t *s_row = img.raw_row_ptr (h - 1U - y);
      uint8_t *d_row = dst_base + y * bmp_row_stride;
      for (size_t x = 0U; x < w; ++x)
        {
          d_row[x * c + 0U] = s_row[x * c + 2U];
          d_row[x * c + 1U] = s_row[x * c + 1U];
          d_row[x * c + 2U] = s_row[x * c + 0U];
          if (c == 4U)
            {
              d_row[x * c + 3U] = s_row[x * c + 3U];
            }
        }
    }
  return out;
}

/* =========================================================================
   3. Clean-Room QOI (Quite OK Image Format) Codec
   ========================================================================= */

uint32_t
read_be32 (const uint8_t *p) noexcept
{
  return (static_cast<uint32_t> (p[0]) << 24U)
         | (static_cast<uint32_t> (p[1]) << 16U)
         | (static_cast<uint32_t> (p[2]) << 8U)
         | static_cast<uint32_t> (p[3]);
}

void
write_be32 (uint8_t *p, uint32_t v) noexcept
{
  p[0] = static_cast<uint8_t> ((v >> 24U) & 0xFFU);
  p[1] = static_cast<uint8_t> ((v >> 16U) & 0xFFU);
  p[2] = static_cast<uint8_t> ((v >> 8U) & 0xFFU);
  p[3] = static_cast<uint8_t> (v & 0xFFU);
}

struct Qoi_Rgba
{
  uint8_t r = 0U;
  uint8_t g = 0U;
  uint8_t b = 0U;
  uint8_t a = 255U;

  constexpr bool
  operator== (const Qoi_Rgba &o) const noexcept = default;
};

constexpr size_t
qoi_hash (Qoi_Rgba px) noexcept
{
  return (static_cast<size_t> (px.r) * 3U + static_cast<size_t> (px.g) * 5U
          + static_cast<size_t> (px.b) * 7U
          + static_cast<size_t> (px.a) * 11U)
         & 63U;
}

core::Result<core::Dyn_Image>
decode_qoi (std::span<const uint8_t> bytes)
{
  if (bytes.size () < 22U || bytes[0] != 'q' || bytes[1] != 'o'
      || bytes[2] != 'i' || bytes[3] != 'f')
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_qoi: invalid QOI header");
    }
  const uint32_t width = read_be32 (bytes.data () + 4U);
  const uint32_t height = read_be32 (bytes.data () + 8U);
  const uint8_t channels = bytes[12];
  if (width == 0U || height == 0U || width > kMaxDimension
      || height > kMaxDimension || (channels != 3U && channels != 4U))
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_qoi: invalid dimensions or channels");
    }

  auto img_res = core::Dyn_Image::create (width, height, channels);
  if (!img_res.has_value ())
    {
      return img_res;
    }
  core::Dyn_Image img = std::move (*img_res);

  std::array<Qoi_Rgba, 64U> index {};
  for (auto &e : index)
    {
      e.a = 0U;
    }
  Qoi_Rgba px {0U, 0U, 0U, 255U};
  size_t run = 0U;
  size_t pos = 14U;
  const size_t limit = bytes.size () - 8U;

  for (size_t y = 0U; y < height; ++y)
    {
      uint8_t *d_row = img.raw_row_ptr (y);
      for (size_t x = 0U; x < width; ++x)
        {
          if (run > 0U)
            {
              --run;
            }
          else if (pos < limit)
            {
              const uint8_t b1 = bytes[pos++];
              if (b1 == 0xFEU) /* QOI_OP_RGB */
                {
                  if (pos + 3U > limit)
                    {
                      return core::make_error (core::Error_Code::Codec_Error,
                                               "decode_qoi: truncated OP_RGB");
                    }
                  px.r = bytes[pos++];
                  px.g = bytes[pos++];
                  px.b = bytes[pos++];
                }
              else if (b1 == 0xFFU) /* QOI_OP_RGBA */
                {
                  if (pos + 4U > limit)
                    {
                      return core::make_error (
                        core::Error_Code::Codec_Error,
                        "decode_qoi: truncated OP_RGBA");
                    }
                  px.r = bytes[pos++];
                  px.g = bytes[pos++];
                  px.b = bytes[pos++];
                  px.a = bytes[pos++];
                }
              else if ((b1 & 0xC0U) == 0x00U) /* QOI_OP_INDEX */
                {
                  px = index[b1 & 63U];
                }
              else if ((b1 & 0xC0U) == 0x40U) /* QOI_OP_DIFF */
                {
                  px.r = static_cast<uint8_t> (
                    px.r + ((b1 >> 4U) & 0x03U) - 2);
                  px.g = static_cast<uint8_t> (
                    px.g + ((b1 >> 2U) & 0x03U) - 2);
                  px.b = static_cast<uint8_t> (px.b + (b1 & 0x03U) - 2);
                }
              else if ((b1 & 0xC0U) == 0x80U) /* QOI_OP_LUMA */
                {
                  if (pos >= limit)
                    {
                      return core::make_error (
                        core::Error_Code::Codec_Error,
                        "decode_qoi: truncated OP_LUMA");
                    }
                  const uint8_t b2 = bytes[pos++];
                  const int vg = static_cast<int> (b1 & 0x3FU) - 32;
                  px.r = static_cast<uint8_t> (
                    px.r + vg - 8 + ((b2 >> 4U) & 0x0FU));
                  px.g = static_cast<uint8_t> (px.g + vg);
                  px.b = static_cast<uint8_t> (
                    px.b + vg - 8 + (b2 & 0x0FU));
                }
              else if ((b1 & 0xC0U) == 0xC0U) /* QOI_OP_RUN */
                {
                  run = static_cast<size_t> (b1 & 0x3FU);
                }
              index[qoi_hash (px)] = px;
            }
          d_row[x * channels + 0U] = px.r;
          d_row[x * channels + 1U] = px.g;
          d_row[x * channels + 2U] = px.b;
          if (channels == 4U)
            {
              d_row[x * channels + 3U] = px.a;
            }
        }
    }
  return img;
}

core::Result<std::vector<uint8_t>>
encode_qoi (const core::Dyn_Image &img)
{
  if (img.empty () || img.depth () != core::Pixel_Depth::U8
      || (img.channels () != 3U && img.channels () != 4U))
    {
      return core::make_error (core::Error_Code::Format_Mismatch,
                               "encode_qoi: requires U8C3 or U8C4 image");
    }

  const size_t w = img.width ();
  const size_t h = img.height ();
  const size_t c = img.channels ();
  std::vector<uint8_t> out;
  out.reserve (14U + w * h * (c + 1U) + 8U);

  out.resize (14U);
  out[0] = 'q';
  out[1] = 'o';
  out[2] = 'i';
  out[3] = 'f';
  write_be32 (out.data () + 4U, static_cast<uint32_t> (w));
  write_be32 (out.data () + 8U, static_cast<uint32_t> (h));
  out[12] = static_cast<uint8_t> (c);
  out[13] = 0U;

  std::array<Qoi_Rgba, 64U> index {};
  for (auto &e : index)
    {
      e.a = 0U;
    }
  Qoi_Rgba prev {0U, 0U, 0U, 255U};
  size_t run = 0U;

  for (size_t y = 0U; y < h; ++y)
    {
      const uint8_t *s_row = img.raw_row_ptr (y);
      for (size_t x = 0U; x < w; ++x)
        {
          const Qoi_Rgba px {s_row[x * c + 0U], s_row[x * c + 1U],
                             s_row[x * c + 2U],
                             c == 4U ? s_row[x * c + 3U] : uint8_t (255U)};
          if (px == prev)
            {
              ++run;
              if (run == 62U || (y + 1U == h && x + 1U == w))
                {
                  out.push_back (
                    static_cast<uint8_t> (0xC0U | (run - 1U)));
                  run = 0U;
                }
            }
          else
            {
              if (run > 0U)
                {
                  out.push_back (
                    static_cast<uint8_t> (0xC0U | (run - 1U)));
                  run = 0U;
                }
              const size_t h_idx = qoi_hash (px);
              if (index[h_idx] == px)
                {
                  out.push_back (static_cast<uint8_t> (h_idx));
                }
              else
                {
                  index[h_idx] = px;
                  if (px.a == prev.a)
                    {
                      out.push_back (0xFEU);
                      out.push_back (px.r);
                      out.push_back (px.g);
                      out.push_back (px.b);
                    }
                  else
                    {
                      out.push_back (0xFFU);
                      out.push_back (px.r);
                      out.push_back (px.g);
                      out.push_back (px.b);
                      out.push_back (px.a);
                    }
                }
              prev = px;
            }
        }
    }

  for (size_t i = 0U; i < 7U; ++i)
    {
      out.push_back (0U);
    }
  out.push_back (1U);
  return out;
}

/* =========================================================================
   4. Scanline-Direct libjpeg-turbo Codec
   ========================================================================= */

struct Surya_Jpeg_Err
{
  jpeg_error_mgr pub;
  std::jmp_buf setjmp_buffer;
};

void
surya_jpeg_error_exit (j_common_ptr cinfo) noexcept
{
  auto *err = reinterpret_cast<Surya_Jpeg_Err *> (cinfo->err);
  std::longjmp (err->setjmp_buffer, 1);
}

core::Result<core::Dyn_Image>
decode_jpeg (std::span<const uint8_t> bytes)
{
  jpeg_decompress_struct cinfo {};
  Surya_Jpeg_Err jerr {};
  cinfo.err = jpeg_std_error (&jerr.pub);
  jerr.pub.error_exit = surya_jpeg_error_exit;

  if (setjmp (jerr.setjmp_buffer))
    {
      jpeg_destroy_decompress (&cinfo);
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_jpeg: libjpeg decompression failure");
    }

  jpeg_create_decompress (&cinfo);
  jpeg_mem_src (&cinfo, bytes.data (),
                static_cast<unsigned long> (bytes.size ()));
  if (jpeg_read_header (&cinfo, TRUE) != JPEG_HEADER_OK)
    {
      jpeg_destroy_decompress (&cinfo);
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_jpeg: invalid JPEG header");
    }

  if (cinfo.num_components == 1)
    {
      cinfo.out_color_space = JCS_GRAYSCALE;
    }
  else
    {
      cinfo.out_color_space = JCS_RGB;
    }

  jpeg_start_decompress (&cinfo);
  const size_t w = cinfo.output_width;
  const size_t h = cinfo.output_height;
  const size_t c = static_cast<size_t> (cinfo.output_components);
  if (w == 0U || h == 0U || w > kMaxDimension || h > kMaxDimension
      || (c != 1U && c != 3U))
    {
      jpeg_destroy_decompress (&cinfo);
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_jpeg: unsupported output dimensions");
    }

  auto img_res = core::Dyn_Image::create (w, h, c);
  if (!img_res.has_value ())
    {
      jpeg_destroy_decompress (&cinfo);
      return img_res;
    }
  core::Dyn_Image img = std::move (*img_res);

  while (cinfo.output_scanline < cinfo.output_height)
    {
      JSAMPROW row_ptr = img.raw_row_ptr (cinfo.output_scanline);
      jpeg_read_scanlines (&cinfo, &row_ptr, 1U);
    }

  jpeg_finish_decompress (&cinfo);
  jpeg_destroy_decompress (&cinfo);
  return img;
}

core::Result<std::vector<uint8_t>>
encode_jpeg (const core::Dyn_Image &img, int quality)
{
  if (img.empty () || img.depth () != core::Pixel_Depth::U8
      || (img.channels () != 1U && img.channels () != 3U))
    {
      return core::make_error (core::Error_Code::Format_Mismatch,
                               "encode_jpeg: requires U8C1 or U8C3 image");
    }

  jpeg_compress_struct cinfo {};
  Surya_Jpeg_Err jerr {};
  cinfo.err = jpeg_std_error (&jerr.pub);
  jerr.pub.error_exit = surya_jpeg_error_exit;

  unsigned char *mem_buf = nullptr;
  unsigned long mem_size = 0UL;

  if (setjmp (jerr.setjmp_buffer))
    {
      jpeg_destroy_compress (&cinfo);
      if (mem_buf != nullptr)
        {
          std::free (mem_buf);
        }
      return core::make_error (core::Error_Code::Codec_Error,
                               "encode_jpeg: libjpeg compression failure");
    }

  jpeg_create_compress (&cinfo);
  jpeg_mem_dest (&cinfo, &mem_buf, &mem_size);

  cinfo.image_width = static_cast<JDIMENSION> (img.width ());
  cinfo.image_height = static_cast<JDIMENSION> (img.height ());
  cinfo.input_components = static_cast<int> (img.channels ());
  cinfo.in_color_space =
    (img.channels () == 1U) ? JCS_GRAYSCALE : JCS_RGB;

  jpeg_set_defaults (&cinfo);
  jpeg_set_quality (&cinfo, std::clamp (quality, 1, 100), TRUE);
  jpeg_start_compress (&cinfo, TRUE);

  while (cinfo.next_scanline < cinfo.image_height)
    {
      JSAMPROW row_ptr =
        const_cast<JSAMPROW> (img.raw_row_ptr (cinfo.next_scanline));
      jpeg_write_scanlines (&cinfo, &row_ptr, 1U);
    }

  jpeg_finish_compress (&cinfo);
  jpeg_destroy_compress (&cinfo);

  std::vector<uint8_t> out (mem_buf, mem_buf + mem_size);
  std::free (mem_buf);
  return out;
}

/* =========================================================================
   5. Scanline-Direct libpng Codec
   ========================================================================= */

core::Result<core::Dyn_Image>
decode_png (std::span<const uint8_t> bytes)
{
  png_image image {};
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory (&image, bytes.data (), bytes.size ()))
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_png: invalid PNG header");
    }

  if (image.width == 0U || image.height == 0U || image.width > kMaxDimension
      || image.height > kMaxDimension)
    {
      png_image_free (&image);
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_png: invalid PNG dimensions");
    }

  size_t channels = 3U;
  if ((image.format & PNG_FORMAT_FLAG_ALPHA) != 0U)
    {
      image.format = PNG_FORMAT_RGBA;
      channels = 4U;
    }
  else if ((image.format & PNG_FORMAT_FLAG_COLOR) != 0U)
    {
      image.format = PNG_FORMAT_RGB;
      channels = 3U;
    }
  else
    {
      image.format = PNG_FORMAT_GRAY;
      channels = 1U;
    }

  auto img_res = core::Dyn_Image::create (image.width, image.height, channels);
  if (!img_res.has_value ())
    {
      png_image_free (&image);
      return img_res;
    }
  core::Dyn_Image dst = std::move (*img_res);

  /* Decode directly into dst's 64-byte row-aligned buffer using stride.  */
  const png_int_32 row_stride =
    static_cast<png_int_32> (dst.row_stride_bytes ());
  if (!png_image_finish_read (&image, nullptr, dst.raw_data (), row_stride,
                              nullptr))
    {
      png_image_free (&image);
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_png: PNG payload decode failed");
    }
  png_image_free (&image);
  return dst;
}

core::Result<std::vector<uint8_t>>
encode_png (const core::Dyn_Image &img)
{
  if (img.empty () || img.depth () != core::Pixel_Depth::U8
      || (img.channels () != 1U && img.channels () != 3U
          && img.channels () != 4U))
    {
      return core::make_error (
        core::Error_Code::Format_Mismatch,
        "encode_png: requires U8C1, U8C3, or U8C4 image");
    }

  png_image image {};
  image.version = PNG_IMAGE_VERSION;
  image.width = static_cast<png_uint_32> (img.width ());
  image.height = static_cast<png_uint_32> (img.height ());
  if (img.channels () == 1U)
    image.format = PNG_FORMAT_GRAY;
  else if (img.channels () == 3U)
    image.format = PNG_FORMAT_RGB;
  else
    image.format = PNG_FORMAT_RGBA;

  const png_int_32 stride =
    static_cast<png_int_32> (img.row_stride_bytes ());
  png_alloc_size_t bytes_needed = 0U;
  if (!png_image_write_to_memory (&image, nullptr, &bytes_needed, 0,
                                  img.raw_data (), stride, nullptr)
      || bytes_needed == 0U)
    {
      png_image_free (&image);
      return core::make_error (core::Error_Code::Codec_Error,
                               "encode_png: failed to query PNG size");
    }

  std::vector<uint8_t> out (bytes_needed);
  if (!png_image_write_to_memory (&image, out.data (), &bytes_needed, 0,
                                  img.raw_data (), stride, nullptr))
    {
      png_image_free (&image);
      return core::make_error (core::Error_Code::Codec_Error,
                               "encode_png: failed to write PNG to memory");
    }
  out.resize (bytes_needed);
  png_image_free (&image);
  return out;
}

/* =========================================================================
   6. Direct-Stride libwebp Codec
   ========================================================================= */

core::Result<core::Dyn_Image>
decode_webp (std::span<const uint8_t> bytes)
{
  WebPBitstreamFeatures features {};
  const VP8StatusCode st =
    WebPGetFeatures (bytes.data (), bytes.size (), &features);
  if (st != VP8_STATUS_OK
      || features.width <= 0 || features.height <= 0
      || static_cast<size_t> (features.width) > kMaxDimension
      || static_cast<size_t> (features.height) > kMaxDimension)
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_webp: invalid WebP bitstream");
    }

  const size_t w = static_cast<size_t> (features.width);
  const size_t h = static_cast<size_t> (features.height);
  const size_t c = features.has_alpha ? 4U : 3U;

  auto img_res = core::Dyn_Image::create (w, h, c);
  if (!img_res.has_value ())
    {
      return img_res;
    }
  core::Dyn_Image dst = std::move (*img_res);

  const size_t total_buf = h * dst.row_stride_bytes ();
  const int stride = static_cast<int> (dst.row_stride_bytes ());
  uint8_t *ok = nullptr;
  if (c == 4U)
    {
      ok = WebPDecodeRGBAInto (bytes.data (), bytes.size (),
                               static_cast<uint8_t *> (dst.raw_data ()),
                               total_buf, stride);
    }
  else
    {
      ok = WebPDecodeRGBInto (bytes.data (), bytes.size (),
                              static_cast<uint8_t *> (dst.raw_data ()),
                              total_buf, stride);
    }
  if (ok == nullptr)
    {
      return core::make_error (core::Error_Code::Codec_Error,
                               "decode_webp: WebP payload decode failed");
    }
  return dst;
}

core::Result<std::vector<uint8_t>>
encode_webp (const core::Dyn_Image &img, int quality)
{
  if (img.empty () || img.depth () != core::Pixel_Depth::U8
      || (img.channels () != 3U && img.channels () != 4U))
    {
      return core::make_error (core::Error_Code::Format_Mismatch,
                               "encode_webp: requires U8C3 or U8C4 image");
    }

  uint8_t *webp_ptr = nullptr;
  const float q = static_cast<float> (std::clamp (quality, 1, 100));
  const int w = static_cast<int> (img.width ());
  const int h = static_cast<int> (img.height ());
  const int stride = static_cast<int> (img.row_stride_bytes ());

  size_t encoded_size = 0U;
  if (img.channels () == 3U)
    {
      encoded_size = WebPEncodeRGB (
        static_cast<const uint8_t *> (img.raw_data ()), w, h, stride, q,
        &webp_ptr);
    }
  else
    {
      encoded_size = WebPEncodeRGBA (
        static_cast<const uint8_t *> (img.raw_data ()), w, h, stride, q,
        &webp_ptr);
    }

  if (encoded_size == 0U || webp_ptr == nullptr)
    {
      if (webp_ptr != nullptr)
        {
          WebPFree (webp_ptr);
        }
      return core::make_error (core::Error_Code::Codec_Error,
                               "encode_webp: WebPEncode failed");
    }

  std::vector<uint8_t> out (webp_ptr, webp_ptr + encoded_size);
  WebPFree (webp_ptr);
  return out;
}

} /* namespace */

core::Result<core::Dyn_Image>
decode_image_from_memory (std::span<const uint8_t> bytes)
{
  const Image_Format fmt = detect_image_format (bytes);
  switch (fmt)
    {
    case Image_Format::PNM:
      return decode_pnm (bytes);
    case Image_Format::BMP:
      return decode_bmp (bytes);
    case Image_Format::QOI:
      return decode_qoi (bytes);
    case Image_Format::JPEG:
      return decode_jpeg (bytes);
    case Image_Format::PNG:
      return decode_png (bytes);
    case Image_Format::WebP:
      return decode_webp (bytes);
    default:
      return core::make_error (
        core::Error_Code::Codec_Error,
        "decode_image_from_memory: unrecognized or unsupported header");
    }
}

core::Result<std::vector<uint8_t>>
encode_image_to_memory (const core::Dyn_Image &img,
                        Image_Format fmt,
                        int quality)
{
  switch (fmt)
    {
    case Image_Format::PNM:
      return encode_pnm (img);
    case Image_Format::BMP:
      return encode_bmp (img);
    case Image_Format::QOI:
      return encode_qoi (img);
    case Image_Format::JPEG:
      return encode_jpeg (img, quality);
    case Image_Format::PNG:
      return encode_png (img);
    case Image_Format::WebP:
      return encode_webp (img, quality);
    default:
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "encode_image_to_memory: unsupported target Image_Format");
    }
}

core::Result<core::Dyn_Image>
read_image (const std::string &path)
{
  if (format_from_extension (path) == Image_Format::TIFF)
    {
      TIFFSetErrorHandler (nullptr);
      TIFFSetWarningHandler (nullptr);
      TIFF *tif = TIFFOpen (path.c_str (), "r");
      if (tif == nullptr)
        {
          return core::make_error (core::Error_Code::Io_Error,
                                   "read_image: cannot open TIFF file");
        }
      uint32_t w = 0U;
      uint32_t h = 0U;
      TIFFGetField (tif, TIFFTAG_IMAGEWIDTH, &w);
      TIFFGetField (tif, TIFFTAG_IMAGELENGTH, &h);
      if (w == 0U || h == 0U || w > kMaxDimension || h > kMaxDimension)
        {
          TIFFClose (tif);
          return core::make_error (core::Error_Code::Codec_Error,
                                   "read_image: invalid TIFF dimensions");
        }
      std::vector<uint32_t> raster (static_cast<size_t> (w) * h);
      if (!TIFFReadRGBAImageOriented (tif, w, h, raster.data (),
                                      ORIENTATION_TOPLEFT, 0))
        {
          TIFFClose (tif);
          return core::make_error (core::Error_Code::Codec_Error,
                                   "read_image: failed to read TIFF raster");
        }
      TIFFClose (tif);
      auto img_res = core::Dyn_Image::create (w, h, 3U);
      if (!img_res.has_value ())
        {
          return img_res;
        }
      core::Dyn_Image dst = std::move (*img_res);
      for (size_t y = 0U; y < h; ++y)
        {
          uint8_t *d_row = dst.raw_row_ptr (y);
          const uint32_t *s_row = raster.data () + y * w;
          for (size_t x = 0U; x < w; ++x)
            {
              const uint32_t abgr = s_row[x];
              d_row[x * 3U + 0U] = static_cast<uint8_t> (TIFFGetR (abgr));
              d_row[x * 3U + 1U] = static_cast<uint8_t> (TIFFGetG (abgr));
              d_row[x * 3U + 2U] = static_cast<uint8_t> (TIFFGetB (abgr));
            }
        }
      return dst;
    }

  const int fd = ::open (path.c_str (), O_RDONLY);
  if (fd < 0)
    {
      return core::make_error (core::Error_Code::Io_Error,
                               "read_image: failed to open file " + path);
    }

  struct stat st {};
  if (::fstat (fd, &st) != 0 || st.st_size <= 0)
    {
      ::close (fd);
      return core::make_error (core::Error_Code::Io_Error,
                               "read_image: empty or invalid file " + path);
    }

  const size_t file_size = static_cast<size_t> (st.st_size);
  void *mapped =
    ::mmap (nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close (fd);
  if (mapped == MAP_FAILED)
    {
      return core::make_error (core::Error_Code::Io_Error,
                               "read_image: mmap failed for " + path);
    }

  const std::span<const uint8_t> view (
    static_cast<const uint8_t *> (mapped), file_size);
  auto res = decode_image_from_memory (view);
  ::munmap (mapped, file_size);
  return res;
}

core::Result<void>
write_image (const std::string &path,
             const core::Dyn_Image &img,
             Image_Format fmt,
             int quality)
{
  const Image_Format target_fmt =
    (fmt == Image_Format::Auto) ? format_from_extension (path) : fmt;
  if (target_fmt == Image_Format::Auto)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "write_image: cannot infer Image_Format from path " + path);
    }

  if (target_fmt == Image_Format::TIFF)
    {
      if (img.empty () || img.depth () != core::Pixel_Depth::U8
          || (img.channels () != 1U && img.channels () != 3U))
        {
          return core::make_error (
            core::Error_Code::Format_Mismatch,
            "write_image (TIFF): requires U8C1 or U8C3");
        }
      TIFFSetErrorHandler (nullptr);
      TIFFSetWarningHandler (nullptr);
      TIFF *tif = TIFFOpen (path.c_str (), "w");
      if (tif == nullptr)
        {
          return core::make_error (core::Error_Code::Io_Error,
                                   "write_image: cannot create TIFF file");
        }
      TIFFSetField (tif, TIFFTAG_IMAGEWIDTH,
                    static_cast<uint32_t> (img.width ()));
      TIFFSetField (tif, TIFFTAG_IMAGELENGTH,
                    static_cast<uint32_t> (img.height ()));
      TIFFSetField (tif, TIFFTAG_SAMPLESPERPIXEL,
                    static_cast<uint16_t> (img.channels ()));
      TIFFSetField (tif, TIFFTAG_BITSPERSAMPLE, uint16_t (8U));
      TIFFSetField (tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
      TIFFSetField (tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
      TIFFSetField (tif, TIFFTAG_PHOTOMETRIC,
                    img.channels () == 1U ? PHOTOMETRIC_MINISBLACK
                                          : PHOTOMETRIC_RGB);
      for (size_t y = 0U; y < img.height (); ++y)
        {
          uint8_t *row = const_cast<uint8_t *> (img.raw_row_ptr (y));
          TIFFWriteScanline (tif, row, static_cast<uint32_t> (y), 0);
        }
      TIFFClose (tif);
      return {};
    }

  auto bytes_res = encode_image_to_memory (img, target_fmt, quality);
  if (!bytes_res.has_value ())
    {
      return std::unexpected (bytes_res.error ());
    }
  std::ofstream out (path, std::ios::binary | std::ios::trunc);
  if (!out.is_open ())
    {
      return core::make_error (core::Error_Code::Io_Error,
                               "write_image: cannot open " + path);
    }
  out.write (reinterpret_cast<const char *> (bytes_res->data ()),
             static_cast<std::streamsize> (bytes_res->size ()));
  if (!out.good ())
    {
      return core::make_error (core::Error_Code::Io_Error,
                               "write_image: disk write failure on " + path);
    }
  return {};
}

core::Result<core::Image_U8C3>
read_image_rgb (const std::string &path)
{
  auto dyn_res = read_image (path);
  if (!dyn_res.has_value ())
    {
      return std::unexpected (dyn_res.error ());
    }
  const core::Dyn_Image &dyn = *dyn_res;
  if (dyn.channels () == 3U)
    {
      auto v = dyn.try_const_view<uint8_t, 3U> ();
      if (!v.has_value ())
        {
          return std::unexpected (v.error ());
        }
      return core::Image_U8C3::from_view (*v);
    }
  if (dyn.channels () == 4U)
    {
      auto v = dyn.try_const_view<uint8_t, 4U> ();
      if (!v.has_value ())
        {
          return std::unexpected (v.error ());
        }
      return imgproc::rgba_to_rgb (*v);
    }
  if (dyn.channels () == 1U)
    {
      auto v = dyn.try_const_view<uint8_t, 1U> ();
      if (!v.has_value ())
        {
          return std::unexpected (v.error ());
        }
      return imgproc::gray_to_rgb (*v);
    }
  return core::make_error (core::Error_Code::Format_Mismatch,
                           "read_image_rgb: unsupported channel count");
}

core::Result<core::Image_U8C1>
read_image_gray (const std::string &path)
{
  auto dyn_res = read_image (path);
  if (!dyn_res.has_value ())
    {
      return std::unexpected (dyn_res.error ());
    }
  const core::Dyn_Image &dyn = *dyn_res;
  if (dyn.channels () == 1U)
    {
      auto v = dyn.try_const_view<uint8_t, 1U> ();
      if (!v.has_value ())
        {
          return std::unexpected (v.error ());
        }
      return core::Image_U8C1::from_view (*v);
    }
  if (dyn.channels () == 3U)
    {
      auto v = dyn.try_const_view<uint8_t, 3U> ();
      if (!v.has_value ())
        {
          return std::unexpected (v.error ());
        }
      return imgproc::rgb_to_gray (*v);
    }
  if (dyn.channels () == 4U)
    {
      auto v = dyn.try_const_view<uint8_t, 4U> ();
      if (!v.has_value ())
        {
          return std::unexpected (v.error ());
        }
      auto rgb = imgproc::rgba_to_rgb (*v);
      if (!rgb.has_value ())
        {
          return std::unexpected (rgb.error ());
        }
      return imgproc::rgb_to_gray (rgb->const_view ());
    }
  return core::make_error (core::Error_Code::Format_Mismatch,
                           "read_image_gray: unsupported channel count");
}

} /* namespace surya::inline v1::io */
