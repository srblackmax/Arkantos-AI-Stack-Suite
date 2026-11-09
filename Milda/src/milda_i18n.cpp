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

#include "milda/milda_i18n.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace milda
{

namespace
{

[[nodiscard]] constexpr char
to_ascii_lower (char ch) noexcept
{
  if (ch >= 'A' && ch <= 'Z')
    {
      return static_cast<char> (ch + ('a' - 'A'));
    }
  return ch;
}

[[nodiscard]] constexpr bool
is_ascii_alpha (char ch) noexcept
{
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
}

[[nodiscard]] bool
equals_ascii_icase (std::string_view a, std::string_view b) noexcept
{
  if (a.size () != b.size ())
    {
      return false;
    }
  for (std::size_t i = 0U; i < a.size (); ++i)
    {
      if (to_ascii_lower (a[i]) != to_ascii_lower (b[i]))
        {
          return false;
        }
    }
  return true;
}

[[nodiscard]] constexpr bool
is_esperanto_diacritic (uint32_t cp) noexcept
{
  return cp == 0x0108U || cp == 0x0109U || cp == 0x011CU || cp == 0x011DU
         || cp == 0x0124U || cp == 0x0125U || cp == 0x0134U || cp == 0x0135U
         || cp == 0x015CU || cp == 0x015DU || cp == 0x016CU || cp == 0x016DU;
}

[[nodiscard]] constexpr bool
is_spanish_diacritic (uint32_t cp) noexcept
{
  return cp == 0x00A1U || cp == 0x00BFU || cp == 0x00C1U || cp == 0x00C9U
         || cp == 0x00CDU || cp == 0x00D1U || cp == 0x00D3U || cp == 0x00DAU
         || cp == 0x00DCU || cp == 0x00E1U || cp == 0x00E9U || cp == 0x00EDU
         || cp == 0x00F1U || cp == 0x00F3U || cp == 0x00FAU || cp == 0x00FCU;
}

[[nodiscard]] constexpr bool
is_cjk_ideograph (uint32_t cp) noexcept
{
  return (cp >= 0x4E00U && cp <= 0x9FFFU)
         || (cp >= 0x3400U && cp <= 0x4DBFU);
}

[[nodiscard]] constexpr bool
is_cyrillic_codepoint (uint32_t cp) noexcept
{
  return cp >= 0x0400U && cp <= 0x04FFU;
}

/* Decodes a single UTF-8 codepoint starting at text[pos] and advances
   *pos. Returns 0xFFFD on malformed sequences.  */
[[nodiscard]] uint32_t
decode_next_utf8 (std::string_view text, std::size_t *pos) noexcept
{
  const std::size_t idx = *pos;
  const auto b0 = static_cast<uint8_t> (text[idx]);
  if ((b0 & 0x80U) == 0U)
    {
      *pos = idx + 1U;
      return b0;
    }
  if ((b0 & 0xE0U) == 0xC0U && idx + 1U < text.size ())
    {
      const auto b1 = static_cast<uint8_t> (text[idx + 1U]);
      if ((b1 & 0xC0U) == 0x80U)
        {
          *pos = idx + 2U;
          return ((static_cast<uint32_t> (b0 & 0x1FU) << 6U)
                  | static_cast<uint32_t> (b1 & 0x3FU));
        }
    }
  else if ((b0 & 0xF0U) == 0xE0U && idx + 2U < text.size ())
    {
      const auto b1 = static_cast<uint8_t> (text[idx + 1U]);
      const auto b2 = static_cast<uint8_t> (text[idx + 2U]);
      if ((b1 & 0xC0U) == 0x80U && (b2 & 0xC0U) == 0x80U)
        {
          *pos = idx + 3U;
          return ((static_cast<uint32_t> (b0 & 0x0FU) << 12U)
                  | (static_cast<uint32_t> (b1 & 0x3FU) << 6U)
                  | static_cast<uint32_t> (b2 & 0x3FU));
        }
    }
  else if ((b0 & 0xF8U) == 0xF0U && idx + 3U < text.size ())
    {
      const auto b1 = static_cast<uint8_t> (text[idx + 1U]);
      const auto b2 = static_cast<uint8_t> (text[idx + 2U]);
      const auto b3 = static_cast<uint8_t> (text[idx + 3U]);
      if ((b1 & 0xC0U) == 0x80U && (b2 & 0xC0U) == 0x80U
          && (b3 & 0xC0U) == 0x80U)
        {
          *pos = idx + 4U;
          return ((static_cast<uint32_t> (b0 & 0x07U) << 18U)
                  | (static_cast<uint32_t> (b1 & 0x3FU) << 12U)
                  | (static_cast<uint32_t> (b2 & 0x3FU) << 6U)
                  | static_cast<uint32_t> (b3 & 0x3FU));
        }
    }
  *pos = idx + 1U;
  return 0xFFFDU;
}

[[nodiscard]] int32_t
score_esperanto_word (std::string_view word) noexcept
{
  constexpr std::array<std::string_view, 22U> k_eo_strong =
    {
      "kaj", "estas", "kun", "pri", "saluton", "dankon", "kiel", "kiu",
      "kio", "kion", "kiun", "fartas", "cxu", "sxi", "gxi", "cxi",
      "hodiaux", "morgaux", "bonan", "tagon", "esperanto", "esperanton"
    };
  for (const std::string_view w : k_eo_strong)
    {
      if (equals_ascii_icase (word, w))
        {
          return 2;
        }
    }
  if (equals_ascii_icase (word, "al") || equals_ascii_icase (word, "vi")
      || equals_ascii_icase (word, "mi") || equals_ascii_icase (word, "ni")
      || equals_ascii_icase (word, "sed"))
    {
      return 1;
    }
  return 0;
}

[[nodiscard]] int32_t
score_spanish_word (std::string_view word) noexcept
{
  constexpr std::array<std::string_view, 19U> k_es_strong =
    {
      "el", "la", "los", "las", "que", "para", "con", "por", "una", "uno",
      "hola", "gracias", "como", "buenos", "buenas", "dias", "noches",
      "pero", "muy"
    };
  for (const std::string_view w : k_es_strong)
    {
      if (equals_ascii_icase (word, w))
        {
          return 2;
        }
    }
  if (equals_ascii_icase (word, "del") || equals_ascii_icase (word, "es")
      || equals_ascii_icase (word, "son"))
    {
      return 1;
    }
  return 0;
}

[[nodiscard]] constexpr std::size_t
utf8_expected_seq_len (uint8_t lead) noexcept
{
  if ((lead & 0x80U) == 0U)
    {
      return 1U;
    }
  if ((lead & 0xE0U) == 0xC0U)
    {
      return 2U;
    }
  if ((lead & 0xF0U) == 0xE0U)
    {
      return 3U;
    }
  if ((lead & 0xF8U) == 0xF0U)
    {
      return 4U;
    }
  return 1U;
}

/* Computes the byte length of the prefix containing only complete UTF-8
   codepoints. If the trailing 1-3 bytes form an incomplete multi-byte
   sequence, they are excluded from the returned length.  */
[[nodiscard]] std::size_t
compute_complete_utf8_prefix (std::string_view bytes) noexcept
{
  const std::size_t len = bytes.size ();
  if (len == 0U)
    {
      return 0U;
    }

  std::size_t cont_cnt = 0U;
  while (cont_cnt < len && cont_cnt < 3U)
    {
      const auto b = static_cast<uint8_t> (bytes[len - 1U - cont_cnt]);
      if ((b & 0xC0U) != 0x80U)
        {
          break;
        }
      ++cont_cnt;
    }

  if (cont_cnt < len && cont_cnt <= 3U)
    {
      const std::size_t lead_idx = len - 1U - cont_cnt;
      const auto lead = static_cast<uint8_t> (bytes[lead_idx]);
      const std::size_t expected = utf8_expected_seq_len (lead);
      const std::size_t avail = cont_cnt + 1U;
      if (avail < expected)
        {
          return lead_idx;
        }
    }
  return len;
}

void
emit_raw_utf8 (const char *data, std::size_t len, bool stream_stdout,
               std::string *out_str) noexcept
{
  if (len == 0U)
    {
      return;
    }
  if (out_str != nullptr)
    {
      out_str->append (data, len);
    }
  if (stream_stdout)
    {
      (void)std::fwrite (data, 1U, len, stdout);
      (void)std::fflush (stdout);
    }
}

} /* namespace */

bool
parse_milda_lang (std::string_view text, MildaLang *out_lang) noexcept
{
  if (out_lang == nullptr)
    {
      return false;
    }
  const std::string_view trimmed = trim_ascii_whitespace (text);
  if (trimmed.empty ())
    {
      return false;
    }
  if (equals_ascii_icase (trimmed, "auto"))
    {
      *out_lang = MildaLang::Auto;
      return true;
    }
  if (equals_ascii_icase (trimmed, "eo")
      || equals_ascii_icase (trimmed, "esperanto"))
    {
      *out_lang = MildaLang::Esperanto;
      return true;
    }
  if (equals_ascii_icase (trimmed, "en")
      || equals_ascii_icase (trimmed, "english"))
    {
      *out_lang = MildaLang::English;
      return true;
    }
  if (equals_ascii_icase (trimmed, "es")
      || equals_ascii_icase (trimmed, "espanol")
      || trimmed == "español" || trimmed == "Español"
      || equals_ascii_icase (trimmed, "spanish"))
    {
      *out_lang = MildaLang::Spanish;
      return true;
    }
  if (equals_ascii_icase (trimmed, "zh")
      || equals_ascii_icase (trimmed, "zh-hans")
      || equals_ascii_icase (trimmed, "zh_hans")
      || equals_ascii_icase (trimmed, "zh-cn")
      || equals_ascii_icase (trimmed, "zh_cn")
      || equals_ascii_icase (trimmed, "chinese")
      || equals_ascii_icase (trimmed, "mandarin"))
    {
      *out_lang = MildaLang::MandarinSimplified;
      return true;
    }
  if (equals_ascii_icase (trimmed, "ru")
      || equals_ascii_icase (trimmed, "rus")
      || equals_ascii_icase (trimmed, "russian"))
    {
      *out_lang = MildaLang::Russian;
      return true;
    }
  return false;
}

std::string_view
lang_switch_confirmation (MildaLang lang) noexcept
{
  switch (lang)
    {
    case MildaLang::Auto:
      return "[milda] Lingvo-reĝimo: Aŭtomata / Auto "
             "(auto: eo, en, es, zh-Hans, ru).";
    case MildaLang::Esperanto:
      return "[milda] Lingvo ŝanĝita al: Esperanto (eo). "
             "Sistempromeso ĝisdatigita.";
    case MildaLang::English:
      return "[milda] Language switched to: English (en). "
             "System prompt updated.";
    case MildaLang::Spanish:
      return "[milda] Idioma cambiado a: Español (es). "
             "Instrucción del sistema actualizada.";
    case MildaLang::MandarinSimplified:
      return "[milda] 语言已切换为：简体中文 (zh-Hans)。"
             "系统提示词已更新。";
    case MildaLang::Russian:
      return "[milda] Язык переключён на: "
             "Русский (ru). Системный промпт "
             "обновлён.";
    }
  return "[milda] Language updated.";
}

MildaLang
detect_prompt_language (std::string_view utf8_text) noexcept
{
  std::size_t cjk_count = 0U;
  std::size_t cyrillic_count = 0U;
  std::size_t eo_diacritics = 0U;
  std::size_t es_diacritics = 0U;

  std::size_t pos = 0U;
  while (pos < utf8_text.size ())
    {
      const uint32_t cp = decode_next_utf8 (utf8_text, &pos);
      if (is_cjk_ideograph (cp))
        {
          ++cjk_count;
        }
      else if (is_cyrillic_codepoint (cp))
        {
          ++cyrillic_count;
        }
      else if (is_esperanto_diacritic (cp))
        {
          ++eo_diacritics;
        }
      else if (is_spanish_diacritic (cp))
        {
          ++es_diacritics;
        }
    }

  if (cjk_count > 0U || cyrillic_count > 0U)
    {
      return (cjk_count >= cyrillic_count) ? MildaLang::MandarinSimplified
                                           : MildaLang::Russian;
    }
  if (eo_diacritics > 0U || es_diacritics > 0U)
    {
      return (eo_diacritics >= es_diacritics) ? MildaLang::Esperanto
                                              : MildaLang::Spanish;
    }

  int32_t eo_score = 0;
  int32_t es_score = 0;
  std::size_t idx = 0U;
  while (idx < utf8_text.size ())
    {
      while (idx < utf8_text.size () && !is_ascii_alpha (utf8_text[idx]))
        {
          ++idx;
        }
      const std::size_t word_start = idx;
      while (idx < utf8_text.size () && is_ascii_alpha (utf8_text[idx]))
        {
          ++idx;
        }
      if (idx > word_start)
        {
          const std::string_view word
              = utf8_text.substr (word_start, idx - word_start);
          eo_score += score_esperanto_word (word);
          es_score += score_spanish_word (word);
        }
    }

  if (eo_score > es_score && eo_score >= 2)
    {
      return MildaLang::Esperanto;
    }
  if (es_score > eo_score && es_score >= 2)
    {
      return MildaLang::Spanish;
    }
  return MildaLang::English;
}

bool
detect_explicit_language_request (std::string_view utf8_text,
                                  MildaLang *out_lang) noexcept
{
  if (out_lang == nullptr || utf8_text.empty ())
    {
      return false;
    }

  std::string lower;
  lower.reserve (utf8_text.size ());
  for (const char ch : utf8_text)
    {
      lower.push_back (to_ascii_lower (ch));
    }
  const std::string_view sv (lower);

  auto has_any = [&] (std::span<const std::string_view> needles) noexcept
  {
    for (const std::string_view n : needles)
      {
        if (sv.find (n) != std::string_view::npos)
          {
            return true;
          }
      }
    return false;
  };

  /* Reject non-directive statements/questions ("i speak", "do you speak",
     "si hablas", etc.) unless accompanied by an explicit command.  */
  constexpr std::array<std::string_view, 9U> k_non_directives =
    {
      "si hablas", "do you speak", "can you speak", "i speak",
      "yo hablo",
      "\321\217 \320\263\320\276\320\262\320\276\321\200\321\216",
      "\321\202\321\213 \320\263\320\276\320\262\320\276\321\200\320\270",
      "cxu vi parolas", "\304\211u vi parolas"
    };
  constexpr std::array<std::string_view, 28U> k_switch_verbs =
    {
      "speak ", "talk ", "reply ", "respond ", "switch ", "change ",
      "in english", "in spanish", "in russian", "in esperanto",
      "in chinese", "in mandarin", "habla ", "hablame", "h\303\241blame",
      "responde ", "cambia ", "en espa\303\261ol", "en espanol",
      "en ingl\303\251s", "en ingles", "en ruso", "parolu ", "respondu ",
      "\320\263\320\276\320\262\320\276\321\200\320\270",
      "\320\276\321\202\320\262\320\265\321\207\320\260\320\271",
      "\320\277\320\265\321\200\320\265\320\272\320\273\321\216\321\207",
      "\347\224\250\344\270\255\346\226\207"
    };

  if (has_any (k_non_directives) && !has_any (k_switch_verbs))
    {
      return false;
    }
  if (!has_any (k_switch_verbs)
      && sv.find ("\350\257\264\344\270\255\346\226\207")
             == std::string_view::npos
      && sv.find ("\350\257\264\350\213\261\350\257\255")
             == std::string_view::npos)
    {
      return false;
    }

  constexpr std::array<std::string_view, 7U> k_ru_names =
    {
      "russian", "ruso", "rusa", "\321\200\321\203\321\201\321\201\320\272",
      "\320\277\320\276-\321\200\321\203\321\201\321\201\320\272\320\270",
      "\344\277\204\350\257\255", "\344\277\204\346\226\207"
    };
  constexpr std::array<std::string_view, 6U> k_es_names =
    {
      "spanish", "espanol", "espa\303\261ol", "hispana",
      "\320\270\321\201\320\277\320\260\320\275\321\201\320\272",
      "\350\245\277\347\217\255\347\211\231\350\257\255"
    };
  constexpr std::array<std::string_view, 7U> k_en_names =
    {
      "english", "ingles", "ingl\303\251s", "angla",
      "\320\260\320\275\320\263\320\273\320\270\320\271\321\201\320\272",
      "\350\213\261\350\257\255", "\350\213\261\346\226\207"
    };
  constexpr std::array<std::string_view, 4U> k_eo_names =
    {
      "esperanto", "esperante",
      "\321\215\321\201\320\277\320\265\321\200\320\260\320\275\321\202",
      "\344\270\226\347\225\214\350\257\255"
    };
  constexpr std::array<std::string_view, 8U> k_zh_names =
    {
      "chinese", "mandarin", "chino", "cxina", "\304\211ina",
      "\320\272\320\270\321\202\320\260\320\271\321\201\320\272",
      "\344\270\255\346\226\207", "\346\261\211\350\257\255"
    };

  if (has_any (k_ru_names))
    {
      *out_lang = MildaLang::Russian;
      return true;
    }
  if (has_any (k_es_names))
    {
      *out_lang = MildaLang::Spanish;
      return true;
    }
  if (has_any (k_en_names))
    {
      *out_lang = MildaLang::English;
      return true;
    }
  if (has_any (k_eo_names))
    {
      *out_lang = MildaLang::Esperanto;
      return true;
    }
  if (has_any (k_zh_names))
    {
      *out_lang = MildaLang::MandarinSimplified;
      return true;
    }
  return false;
}

bool
is_token_violating_language_lock (std::string_view piece,
                                  MildaLang target_lang) noexcept
{
  if (piece.empty ())
    {
      return false;
    }

  for (std::size_t i = 0U; i < piece.size (); ++i)
    {
      const auto b0 = static_cast<uint8_t> (piece[i]);
      if (b0 >= 0xF0U)
        {
          return true;
        }
      if (b0 == 0xE2U && i + 1U < piece.size ())
        {
          const auto b1 = static_cast<uint8_t> (piece[i + 1U]);
          if (b1 >= 0x98U && b1 <= 0x9EU)
            {
              return true;
            }
        }
      if (target_lang == MildaLang::Russian)
        {
          if ((b0 >= 'a' && b0 <= 'z') || (b0 >= 'A' && b0 <= 'Z')
              || (b0 >= 0xC3U && b0 <= 0xC5U)
              || (b0 >= 0xE3U && b0 <= 0xE9U))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::MandarinSimplified)
        {
          if ((b0 >= 'a' && b0 <= 'z') || (b0 >= 'A' && b0 <= 'Z')
              || (b0 >= 0xC3U && b0 <= 0xD3U))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::English)
        {
          if ((b0 >= 0xC3U && b0 <= 0xC5U)
              || (b0 >= 0xD0U && b0 <= 0xD3U)
              || (b0 >= 0xE3U && b0 <= 0xE9U))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::Spanish)
        {
          if ((b0 >= 0xC4U && b0 <= 0xC5U)
              || (b0 >= 0xD0U && b0 <= 0xD3U)
              || (b0 >= 0xE3U && b0 <= 0xE9U))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::Esperanto)
        {
          if (b0 == 0xC3U || (b0 >= 0xD0U && b0 <= 0xD3U)
              || (b0 >= 0xE3U && b0 <= 0xE9U))
            {
              return true;
            }
        }
    }

  if (target_lang == MildaLang::Auto)
    {
      return false;
    }

  std::size_t pos = 0U;
  while (pos < piece.size ())
    {
      const uint32_t cp = decode_next_utf8 (piece, &pos);
      if ((cp >= 0x1F000U && cp <= 0x1FAFFU)
          || (cp >= 0x2600U && cp <= 0x27BFU) || cp == 0xFE0FU)
        {
          return true;
        }
      const bool is_lat
          = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
      if (target_lang == MildaLang::Russian)
        {
          if (is_lat || is_cjk_ideograph (cp)
              || is_esperanto_diacritic (cp) || is_spanish_diacritic (cp))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::MandarinSimplified)
        {
          if (is_lat || is_cyrillic_codepoint (cp)
              || is_esperanto_diacritic (cp) || is_spanish_diacritic (cp))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::English)
        {
          if (is_cyrillic_codepoint (cp) || is_cjk_ideograph (cp)
              || is_esperanto_diacritic (cp) || is_spanish_diacritic (cp))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::Spanish)
        {
          if (is_cyrillic_codepoint (cp) || is_cjk_ideograph (cp)
              || is_esperanto_diacritic (cp))
            {
              return true;
            }
        }
      else if (target_lang == MildaLang::Esperanto)
        {
          if (is_cyrillic_codepoint (cp) || is_cjk_ideograph (cp)
              || is_spanish_diacritic (cp))
            {
              return true;
            }
        }
    }
  return false;
}

std::size_t
preprocess_user_input_in_place (char *buf, std::size_t len,
                                MildaLang configured_lang,
                                MildaLang *out_effective_lang,
                                x_system_mode mode) noexcept
{
  if (buf == nullptr || len == 0U)
    {
      if (out_effective_lang != nullptr)
        {
          *out_effective_lang = (configured_lang == MildaLang::Auto)
                                    ? MildaLang::English
                                    : configured_lang;
        }
      return 0U;
    }

  const std::string_view raw_view (buf, len);
  const MildaLang detected = detect_prompt_language (raw_view);
  const MildaLang effective
      = (configured_lang == MildaLang::Auto) ? detected : configured_lang;
  if (out_effective_lang != nullptr)
    {
      *out_effective_lang = effective;
    }

  if (should_apply_x_system (configured_lang, detected, mode))
    {
      return transliterate_x_system_in_place (buf, len);
    }
  return len;
}

void
Utf8StreamBuffer::push (std::string_view piece, bool stream_stdout,
                        std::string *out_str) noexcept
{
  if (piece.empty ())
    {
      return;
    }

  std::size_t offset = 0U;
  if (tail_len_ > 0U)
    {
      const auto lead = static_cast<uint8_t> (tail_[0]);
      const std::size_t expected = utf8_expected_seq_len (lead);
      while (tail_len_ < expected && offset < piece.size ())
        {
          const auto next_b = static_cast<uint8_t> (piece[offset]);
          if ((next_b & 0xC0U) != 0x80U)
            {
              /* Malformed continuation; flush existing tail first.  */
              emit_raw_utf8 (tail_.data (), tail_len_, stream_stdout, out_str);
              tail_len_ = 0U;
              break;
            }
          tail_[tail_len_] = piece[offset];
          ++tail_len_;
          ++offset;
        }

      if (tail_len_ > 0U && tail_len_ < expected)
        {
          return;
        }
      if (tail_len_ > 0U)
        {
          emit_raw_utf8 (tail_.data (), tail_len_, stream_stdout, out_str);
          tail_len_ = 0U;
        }
    }

  const std::string_view rem = piece.substr (offset);
  if (rem.empty ())
    {
      return;
    }

  const std::size_t complete_len = compute_complete_utf8_prefix (rem);
  if (complete_len > 0U)
    {
      emit_raw_utf8 (rem.data (), complete_len, stream_stdout, out_str);
    }

  const std::size_t leftover = rem.size () - complete_len;
  if (leftover > 0U && leftover <= tail_.size ())
    {
      std::memcpy (tail_.data (), rem.data () + complete_len, leftover);
      tail_len_ = leftover;
    }
}

std::string
Utf8StreamBuffer::push (std::string_view piece)
{
  std::string out;
  push (piece, false, &out);
  return out;
}

void
Utf8StreamBuffer::flush_remaining (bool stream_stdout,
                                   std::string *out_str) noexcept
{
  if (tail_len_ > 0U)
    {
      emit_raw_utf8 (tail_.data (), tail_len_, stream_stdout, out_str);
      tail_len_ = 0U;
    }
}

std::string
Utf8StreamBuffer::flush_remaining ()
{
  std::string out;
  flush_remaining (false, &out);
  return out;
}

void
Utf8StreamBuffer::reset () noexcept
{
  tail_len_ = 0U;
}

std::size_t
Utf8StreamBuffer::pending_bytes () const noexcept
{
  return tail_len_;
}

} /* namespace milda */
