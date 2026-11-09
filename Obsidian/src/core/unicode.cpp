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

#include <obsidian/unicode.hpp>

#include <array>

namespace obsidian::unicode
{

namespace
{

struct Byte_Unicode_Tables
{
  std::array<uint32_t, 256> byte_to_cp {};
  std::array<int16_t, 512> cp_to_byte {};

  constexpr Byte_Unicode_Tables (void) noexcept
  {
    cp_to_byte.fill (-1);
    uint32_t extra = 256;
    for (uint32_t b = 0; b < 256; ++b)
      {
        bool direct = (b >= 0x21 && b <= 0x7E)
                      || (b >= 0xA1 && b <= 0xAC)
                      || (b >= 0xAE && b <= 0xFF);
        uint32_t cp = direct ? b : extra++;
        byte_to_cp[b] = cp;
        if (cp < cp_to_byte.size ())
          {
            cp_to_byte[cp] = static_cast<int16_t> (b);
          }
      }
  }
};

constexpr Byte_Unicode_Tables k_gpt2_tables {};

/* Strip accent from common precomposed Latin-1 Supplement (U+00C0..U+00FF)
   and Latin Extended-A characters.  Returns 0 if no ASCII base applies.  */
char
latin_accent_to_ascii (uint32_t cp) noexcept
{
  if (cp >= 0x00C0 && cp <= 0x00C5)
    return 'A';
  if (cp == 0x00C7)
    return 'C';
  if (cp >= 0x00C8 && cp <= 0x00CB)
    return 'E';
  if (cp >= 0x00CC && cp <= 0x00CF)
    return 'I';
  if (cp == 0x00D1)
    return 'N';
  if ((cp >= 0x00D2 && cp <= 0x00D6) || cp == 0x00D8)
    return 'O';
  if (cp >= 0x00D9 && cp <= 0x00DC)
    return 'U';
  if (cp == 0x00DD)
    return 'Y';
  if (cp >= 0x00E0 && cp <= 0x00E5)
    return 'a';
  if (cp == 0x00E7)
    return 'c';
  if (cp >= 0x00E8 && cp <= 0x00EB)
    return 'e';
  if (cp >= 0x00EC && cp <= 0x00EF)
    return 'i';
  if (cp == 0x00F1)
    return 'n';
  if ((cp >= 0x00F2 && cp <= 0x00F6) || cp == 0x00F8)
    return 'o';
  if (cp >= 0x00F9 && cp <= 0x00FC)
    return 'u';
  if (cp == 0x00FD || cp == 0x00FF)
    return 'y';
  return 0;
}

} /* namespace */

Result<uint32_t>
decode_utf8_codepoint (std::string_view text, size_t &offset) noexcept
{
  if (offset >= text.size ())
    {
      return OBSIDIAN_ERROR (Error_Code::Out_Of_Range,
                             "UTF-8 decode offset past end of string");
    }

  const auto *bytes = reinterpret_cast<const uint8_t *> (text.data ());
  const size_t n = text.size ();
  uint8_t b0 = bytes[offset];

  if (b0 < 0x80)
    {
      offset += 1;
      return static_cast<uint32_t> (b0);
    }
  if ((b0 & 0xE0) == 0xC0)
    {
      if (offset + 1 >= n)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "truncated 2-byte UTF-8 sequence");
        }
      uint8_t b1 = bytes[offset + 1];
      if ((b1 & 0xC0) != 0x80)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "invalid UTF-8 continuation byte");
        }
      uint32_t cp = ((static_cast<uint32_t> (b0 & 0x1F) << 6)
                     | static_cast<uint32_t> (b1 & 0x3F));
      if (cp < 0x80)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "overlong 2-byte UTF-8 sequence");
        }
      offset += 2;
      return cp;
    }
  if ((b0 & 0xF0) == 0xE0)
    {
      if (offset + 2 >= n)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "truncated 3-byte UTF-8 sequence");
        }
      uint8_t b1 = bytes[offset + 1];
      uint8_t b2 = bytes[offset + 2];
      if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "invalid UTF-8 continuation byte");
        }
      uint32_t cp = ((static_cast<uint32_t> (b0 & 0x0F) << 12)
                     | (static_cast<uint32_t> (b1 & 0x3F) << 6)
                     | static_cast<uint32_t> (b2 & 0x3F));
      if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "overlong or surrogate 3-byte UTF-8");
        }
      offset += 3;
      return cp;
    }
  if ((b0 & 0xF8) == 0xF0)
    {
      if (offset + 3 >= n)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "truncated 4-byte UTF-8 sequence");
        }
      uint8_t b1 = bytes[offset + 1];
      uint8_t b2 = bytes[offset + 2];
      uint8_t b3 = bytes[offset + 3];
      if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "invalid UTF-8 continuation byte");
        }
      uint32_t cp = ((static_cast<uint32_t> (b0 & 0x07) << 18)
                     | (static_cast<uint32_t> (b1 & 0x3F) << 12)
                     | (static_cast<uint32_t> (b2 & 0x3F) << 6)
                     | static_cast<uint32_t> (b3 & 0x3F));
      if (cp < 0x10000 || cp > 0x10FFFF)
        {
          return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                                 "invalid 4-byte UTF-8 scalar value");
        }
      offset += 4;
      return cp;
    }

  return OBSIDIAN_ERROR (Error_Code::Invalid_UTF8,
                         "invalid UTF-8 leading byte");
}

void
append_utf8_codepoint (uint32_t cp, std::string &out)
{
  if (cp < 0x80)
    {
      out.push_back (static_cast<char> (cp));
    }
  else if (cp < 0x800)
    {
      out.push_back (static_cast<char> (0xC0 | (cp >> 6)));
      out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
    }
  else if (cp < 0x10000)
    {
      out.push_back (static_cast<char> (0xE0 | (cp >> 12)));
      out.push_back (static_cast<char> (0x80 | ((cp >> 6) & 0x3F)));
      out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
    }
  else
    {
      out.push_back (static_cast<char> (0xF0 | (cp >> 18)));
      out.push_back (static_cast<char> (0x80 | ((cp >> 12) & 0x3F)));
      out.push_back (static_cast<char> (0x80 | ((cp >> 6) & 0x3F)));
      out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
    }
}

bool
is_unicode_whitespace (uint32_t cp) noexcept
{
  if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'
      || cp == '\v' || cp == '\f')
    {
      return true;
    }
  if (cp == 0x0085 || cp == 0x00A0 || cp == 0x1680)
    {
      return true;
    }
  if (cp >= 0x2000 && cp <= 0x200A)
    {
      return true;
    }
  return cp == 0x2028 || cp == 0x2029 || cp == 0x202F
         || cp == 0x205F || cp == 0x3000;
}

bool
is_unicode_punctuation (uint32_t cp) noexcept
{
  if ((cp >= 0x21 && cp <= 0x2F)
      || (cp >= 0x3A && cp <= 0x40)
      || (cp >= 0x5B && cp <= 0x60)
      || (cp >= 0x7B && cp <= 0x7E))
    {
      return true;
    }
  if (cp >= 0x00A1 && cp <= 0x00BF && cp != 0x00A0 && cp != 0x00AD)
    {
      return true;
    }
  if ((cp >= 0x2010 && cp <= 0x2027)
      || (cp >= 0x2030 && cp <= 0x205E)
      || (cp >= 0x3001 && cp <= 0x303F)
      || (cp >= 0xFE30 && cp <= 0xFE6F))
    {
      return true;
    }
  return false;
}

bool
is_ignorable_control (uint32_t cp) noexcept
{
  if (cp == 0x0000 || cp == 0x00AD || cp == 0xFEFF || cp == 0xFFFD)
    {
      return cp != 0xFFFD;
    }
  if (cp < 0x0020 && cp != '\t' && cp != '\n' && cp != '\r')
    {
      return true;
    }
  if (cp >= 0x007F && cp <= 0x009F && cp != 0x0085)
    {
      return true;
    }
  if ((cp >= 0x200B && cp <= 0x200F)
      || (cp >= 0x202A && cp <= 0x202E)
      || (cp >= 0x2060 && cp <= 0x206F))
    {
      return true;
    }
  return false;
}

Result<std::string>
normalize_nfkc_text (std::string_view text, bool strip_accents)
{
  std::string out;
  out.reserve (text.size ());

  size_t offset = 0;
  while (offset < text.size ())
    {
      auto cp_res = decode_utf8_codepoint (text, offset);
      if (!cp_res.has_value ())
        {
          return std::unexpected (cp_res.error ());
        }
      uint32_t cp = *cp_res;

      if (is_ignorable_control (cp))
        {
          continue;
        }
      if (strip_accents && cp >= 0x0300 && cp <= 0x036F)
        {
          /* Skip combining diacritical marks.  */
          continue;
        }
      if (is_unicode_whitespace (cp))
        {
          if (cp == '\n' || cp == '\r' || cp == '\t')
            {
              out.push_back (static_cast<char> (cp));
            }
          else
            {
              out.push_back (' ');
            }
          continue;
        }

      /* Fullwidth ASCII variants (U+FF01..U+FF5E) -> ASCII (0x21..0x7E).  */
      if (cp >= 0xFF01 && cp <= 0xFF5E)
        {
          out.push_back (static_cast<char> (cp - 0xFEE0));
          continue;
        }

      /* Typographic single quotes -> ASCII apostrophe.  */
      if (cp >= 0x2018 && cp <= 0x201B)
        {
          out.push_back ('\'');
          continue;
        }
      /* Typographic double quotes and guillemets -> ASCII quote.  */
      if ((cp >= 0x201C && cp <= 0x201F) || cp == 0x00AB || cp == 0x00BB)
        {
          out.push_back ('"');
          continue;
        }
      /* Unicode hyphens and figure/en/em dashes -> ASCII hyphen-minus.  */
      if (cp >= 0x2010 && cp <= 0x2015)
        {
          out.push_back ('-');
          continue;
        }
      /* Horizontal ellipsis -> three periods.  */
      if (cp == 0x2026)
        {
          out.append ("...");
          continue;
        }

      /* Compatibility Latin ligatures.  */
      if (cp == 0xFB00)
        {
          out.append ("ff");
          continue;
        }
      if (cp == 0xFB01)
        {
          out.append ("fi");
          continue;
        }
      if (cp == 0xFB02)
        {
          out.append ("fl");
          continue;
        }
      if (cp == 0xFB03)
        {
          out.append ("ffi");
          continue;
        }
      if (cp == 0xFB04)
        {
          out.append ("ffl");
          continue;
        }
      if (cp == 0x00C6)
        {
          out.append ("AE");
          continue;
        }
      if (cp == 0x00E6)
        {
          out.append ("ae");
          continue;
        }
      if (cp == 0x0152)
        {
          out.append ("OE");
          continue;
        }
      if (cp == 0x0153)
        {
          out.append ("oe");
          continue;
        }
      if (cp == 0x00DF)
        {
          out.append ("ss");
          continue;
        }

      if (strip_accents)
        {
          char base = latin_accent_to_ascii (cp);
          if (base != 0)
            {
              out.push_back (base);
              continue;
            }
        }

      append_utf8_codepoint (cp, out);
    }

  return out;
}

std::string
byte_to_unicode_encode (std::string_view raw_bytes)
{
  std::string out;
  out.reserve (raw_bytes.size () * 2);
  for (unsigned char b : raw_bytes)
    {
      uint32_t cp = k_gpt2_tables.byte_to_cp[b];
      append_utf8_codepoint (cp, out);
    }
  return out;
}

Result<std::string>
byte_to_unicode_decode (std::string_view encoded_utf8)
{
  std::string out;
  out.reserve (encoded_utf8.size ());

  size_t offset = 0;
  while (offset < encoded_utf8.size ())
    {
      auto cp_res = decode_utf8_codepoint (encoded_utf8, offset);
      if (!cp_res.has_value ())
        {
          return std::unexpected (cp_res.error ());
        }
      uint32_t cp = *cp_res;
      if (cp >= k_gpt2_tables.cp_to_byte.size ()
          || k_gpt2_tables.cp_to_byte[cp] < 0)
        {
          return OBSIDIAN_ERROR (
            Error_Code::Corrupted_Format,
            "codepoint not in GPT-2 byte-to-unicode table");
        }
      uint8_t raw_b = static_cast<uint8_t> (k_gpt2_tables.cp_to_byte[cp]);
      out.push_back (static_cast<char> (raw_b));
    }

  return out;
}

} /* namespace obsidian::unicode */
