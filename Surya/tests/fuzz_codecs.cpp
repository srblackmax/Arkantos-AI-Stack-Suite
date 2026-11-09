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

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace surya;
using namespace surya::core;
using namespace surya::io;

int
main (void)
{
  /* Seed corpus with valid encoded PNM, BMP, QOI, PNG, JPEG, and WebP buffers
     and mutate/truncate them across 600 iterations under ASan/UBSan.  */
  Image_U8C3 seed_img (16U, 16U, Scalar<uint8_t, 3U> (50U, 120U, 200U));
  Dyn_Image dyn = Dyn_Image::from_view (seed_img.const_view ());

  const Image_Format fmts[] = {Image_Format::PNM, Image_Format::BMP,
                               Image_Format::QOI, Image_Format::PNG,
                               Image_Format::JPEG, Image_Format::WebP};
  std::vector<std::vector<uint8_t>> corpus;
  for (Image_Format f : fmts)
    {
      auto enc = encode_image_to_memory (dyn, f, 85);
      if (enc.has_value ())
        {
          corpus.push_back (std::move (*enc));
        }
    }

  std::mt19937_64 rng (0x5355525941ULL);
  for (size_t iter = 0U; iter < 600U; ++iter)
    {
      std::vector<uint8_t> buf = corpus[iter % corpus.size ()];
      const size_t action = rng () % 3U;
      if (action == 0U && !buf.empty ())
        {
          /* Truncate buffer at random offset.  */
          buf.resize (rng () % buf.size ());
        }
      else if (action == 1U && !buf.empty ())
        {
          /* Bit-flip 1..4 bytes.  */
          const size_t flips = 1U + (rng () % 4U);
          for (size_t k = 0U; k < flips; ++k)
            {
              buf[rng () % buf.size ()] ^=
                static_cast<uint8_t> (1U << (rng () % 8U));
            }
        }
      else
        {
          /* Pure random noise with magic prefix.  */
          const size_t len = rng () % 256U;
          buf.resize (len);
          for (uint8_t &b : buf)
            {
              b = static_cast<uint8_t> (rng () & 0xFFU);
            }
        }

      auto res = decode_image_from_memory (buf);
      (void) res;
    }

  std::cout << "fuzz_codecs PASSED (600 mutated streams verified).\n";
  return 0;
}
