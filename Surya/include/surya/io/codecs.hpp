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

#ifndef SURYA_IO_CODECS_HPP
#define SURYA_IO_CODECS_HPP

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <surya/core.hpp>

namespace surya::inline v1::io
{

/**
 * @brief Supported image container and compression formats.
 */
enum class Image_Format : uint8_t
{
  Auto = 0,
  PNM = 1,
  BMP = 2,
  QOI = 3,
  JPEG = 4,
  PNG = 5,
  WebP = 6,
  TIFF = 7
};

/**
 * @brief Inspect magic header bytes to identify the container format.
 * @pre None.
 * @complexity O(1) time and space.
 */
SURYA_API Image_Format
detect_image_format (std::span<const uint8_t> bytes) noexcept;

/**
 * @brief Infer Image_Format from a filename extension (.png, .jpg, .webp,
 *        .tiff, .bmp, .ppm, .pgm, .qoi).
 */
SURYA_API Image_Format
format_from_extension (std::string_view path) noexcept;

/**
 * @brief Decode an image from an in-memory byte span directly into a 64-byte
 *        row-aligned Dyn_Image.
 *
 * Supports PNM (P5/P6), uncompressed 24/32-bit BMP, QOI, JPEG (libjpeg-turbo),
 * PNG (libpng), WebP (libwebp), and TIFF (libtiff).
 */
SURYA_API core::Result<core::Dyn_Image>
decode_image_from_memory (std::span<const uint8_t> bytes);

/**
 * @brief Encode a Dyn_Image (U8C1, U8C3, or U8C4) into an in-memory byte
 *        buffer in format FMT.
 * @pre FMT != Image_Format::Auto.
 */
SURYA_API core::Result<std::vector<uint8_t>>
encode_image_to_memory (const core::Dyn_Image &img,
                        Image_Format fmt,
                        int quality = 90);

/**
 * @brief Read an image file from disk using zero-copy POSIX mmap(2) into
 *        decode_image_from_memory.
 */
SURYA_API core::Result<core::Dyn_Image>
read_image (const std::string &path);

/**
 * @brief Write a Dyn_Image to disk at PATH (format inferred from extension or
 *        explicitly specified via FMT).
 */
SURYA_API core::Result<void>
write_image (const std::string &path,
             const core::Dyn_Image &img,
             Image_Format fmt = Image_Format::Auto,
             int quality = 90);

/**
 * @brief Convenience helper to read a 3-channel RGB uint8 image from PATH.
 */
SURYA_API core::Result<core::Image_U8C3>
read_image_rgb (const std::string &path);

/**
 * @brief Convenience helper to read a 1-channel Grayscale uint8 image.
 */
SURYA_API core::Result<core::Image_U8C1>
read_image_gray (const std::string &path);

} /* namespace surya::inline v1::io */

#endif /* SURYA_IO_CODECS_HPP */
