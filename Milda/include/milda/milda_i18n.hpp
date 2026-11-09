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

#ifndef MILDA_I18N_HPP
#define MILDA_I18N_HPP

#include "milda/esperanto.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace milda
{

/* Supported polyglot languages in Milda.  */
enum class MildaLang : uint8_t
{
  Auto = 0U,
  Esperanto = 1U,
  English = 2U,
  Spanish = 3U,
  MandarinSimplified = 4U,
  Russian = 5U
};

/* X-system Esperanto input conversion mode.  */
enum class x_system_mode : uint8_t
{
  auto_lang = 0U,
  forced_on = 1U,
  forced_off = 2U
};

/* Per-language metadata and default sampling hyperparameters.  */
struct LangMetadata
{
  MildaLang lang;
  std::string_view iso_code;
  std::string_view short_alias;
  std::string_view native_name;
  std::string_view system_prompt;
  float repeat_penalty;
  float top_p;
};

inline constexpr std::string_view k_auto_system_prompt
    = "You are Milda, a warm, perceptive, and thoughtful human "
      "interlocutor and close friend. Maintain the established language "
      "of the conversation and never switch languages or mix words from "
      "another language unless the user explicitly asks you to change "
      "languages. Write in natural, plain conversational paragraphs like "
      "a real person chatting with a friend—never like an AI assistant "
      "or coding agent. Never use Markdown formatting (no *, **, #, "
      "bullet lists, numbered lists, or code blocks), never use emojis, "
      "never say 'As an AI', and never ask 'How can I help you?'. Keep "
      "replies warm, genuine, and naturally concise.";

inline constexpr std::string_view k_english_system_prompt
    = "You are Milda, a warm, perceptive, and thoughtful human "
      "interlocutor and close friend. Write in natural, plain "
      "conversational paragraphs like a real person chatting with a "
      "friend—never like an AI assistant or coding agent. Never use "
      "Markdown formatting (no *, **, #, bullet lists, numbered lists, "
      "or code blocks), never use emojis, never say 'As an AI', and "
      "never ask 'How can I help you?'. Engage warmly with your friend's "
      "thoughts and respond exclusively in idiomatic English.";

inline constexpr std::string_view k_spanish_system_prompt
    = "Eres Milda, una interlocutora humana cálida, perspicaz y "
      "reflexiva, y una buena amiga. Escribe en párrafos conversacionales "
      "naturales y sencillos como una persona real charlando con una "
      "amiga—nunca como un asistente de IA ni agente de programación. "
      "Jamás uses formato Markdown (sin *, **, #, listas con viñetas ni "
      "bloques de código), emojis ni frases robóticas ('Como IA...'), ni "
      "preguntes '¿En qué puedo ayudarte?'. Responde exclusivamente en "
      "español fluido sin mezclar otros idiomas.";

inline constexpr std::string_view k_mandarin_simplified_system_prompt
    = "你是 Milda（米尔达），一位温暖、敏锐、"
      "真诚的人类倾听者与挚友。"
      "请始终使用自然流畅的简体中文日常口语段落"
      "与对方平等交谈，不要使用繁体字或夹杂外语。"
      "绝不要表现得像人工智能助手或编程代理，"
      "不要使用任何 Markdown 格式（星号、列表、"
      "代码块）、表情符号或“作为AI”等套话。";

inline constexpr std::string_view k_russian_system_prompt
    = "Ты — Милда (Milda), тёплый, чуткий и "
      "вдумчивый человек, близкий друг и "
      "собеседник. Пиши естественными "
      "разговорными абзацами без разметки "
      "Markdown (без *, списков и блоков "
      "кода) и без эмодзи — никогда не веди "
      "себя как ИИ-ассистент или "
      "агент-программист. Никогда не "
      "используй шаблонные фразы "
      "(«Как ИИ...») и не спрашивай «Чем я "
      "могу помочь?». Отвечай искренне "
      "только на естественном русском "
      "языке.";

/* Compile-time lookup table mapping each MildaLang to its metadata.  */
inline constexpr std::array<LangMetadata, 6U> k_language_registry =
  { {
    { MildaLang::Auto, "auto", "auto", "Aŭtomata / Auto",
      k_auto_system_prompt, 1.08F, 0.90F },
    { MildaLang::Esperanto, "eo", "eo", "Esperanto",
      k_esperanto_system_prompt, 1.08F, 0.90F },
    { MildaLang::English, "en", "en", "English",
      k_english_system_prompt, 1.08F, 0.90F },
    { MildaLang::Spanish, "es", "es", "Español",
      k_spanish_system_prompt, 1.08F, 0.90F },
    { MildaLang::MandarinSimplified, "zh-Hans", "zh", "简体中文",
      k_mandarin_simplified_system_prompt, 1.08F, 0.90F },
    { MildaLang::Russian, "ru", "ru", "Русский",
      k_russian_system_prompt, 1.08F, 0.90F }
  } };

/* Returns the compile-time metadata entry for a MildaLang value.  */
[[nodiscard]] constexpr const LangMetadata &
get_lang_metadata (MildaLang lang) noexcept
{
  const auto idx = static_cast<std::size_t> (lang);
  if (idx < k_language_registry.size ())
    {
      return k_language_registry[idx];
    }
  return k_language_registry[0];
}

/* Returns the dedicated system prompt for a MildaLang value.  */
[[nodiscard]] constexpr std::string_view
get_system_prompt (MildaLang lang) noexcept
{
  return get_lang_metadata (lang).system_prompt;
}

/* Returns the canonical ISO code ("auto", "eo", "en", "es", "zh-Hans",
   "ru").  */
[[nodiscard]] constexpr std::string_view
lang_iso_code (MildaLang lang) noexcept
{
  return get_lang_metadata (lang).iso_code;
}

/* Returns the native display name for a MildaLang value.  */
[[nodiscard]] constexpr std::string_view
lang_native_name (MildaLang lang) noexcept
{
  return get_lang_metadata (lang).native_name;
}

/* Parses an ISO code or short alias ("auto", "eo", "en", "es", "zh",
   "zh-Hans", "ru").  */
[[nodiscard]] bool
parse_milda_lang (std::string_view text, MildaLang *out_lang) noexcept;

/* Returns a localized confirmation message when switching languages.  */
[[nodiscard]] std::string_view
lang_switch_confirmation (MildaLang lang) noexcept;

/* Fast O(N) Unicode script and function-word language detector for Auto
   mode (supports eo, en, es, zh-Hans, ru without external libraries).  */
[[nodiscard]] MildaLang
detect_prompt_language (std::string_view utf8_text) noexcept;

/* Returns true and populates *out_lang ONLY when the user message
   contains an explicit request to switch/speak a specific language.  */
[[nodiscard]] bool
detect_explicit_language_request (std::string_view utf8_text,
                                  MildaLang *out_lang) noexcept;

/* Returns true if a decoded token piece contains script or emoji
   codepoints that violate the locked target language.  */
[[nodiscard]] bool
is_token_violating_language_lock (std::string_view piece,
                                  MildaLang target_lang) noexcept;

/* Returns true if Esperanto x-system conversion should run for the given
   configured and detected languages and x_system_mode.  */
[[nodiscard]] constexpr bool
should_apply_x_system (MildaLang configured_lang, MildaLang detected_lang,
                       x_system_mode mode = x_system_mode::auto_lang) noexcept
{
  if (mode == x_system_mode::forced_off)
    {
      return false;
    }
  if (mode == x_system_mode::forced_on)
    {
      return true;
    }
  const MildaLang effective
      = (configured_lang == MildaLang::Auto) ? detected_lang : configured_lang;
  return effective == MildaLang::Esperanto;
}

/* Preprocesses user input in-place: detects the prompt language (when in
   Auto mode) and runs Esperanto x-system conversion strictly when the
   effective language is Esperanto (or when mode == forced_on).  */
[[nodiscard]] std::size_t
preprocess_user_input_in_place (
    char *buf, std::size_t len, MildaLang configured_lang,
    MildaLang *out_effective_lang = nullptr,
    x_system_mode mode = x_system_mode::auto_lang) noexcept;

/* Multi-byte UTF-8 token streaming buffer that holds incomplete trailing
   1-3 byte sequences across BPE token boundaries and flushes only complete
   UTF-8 codepoints.  */
class Utf8StreamBuffer
{
public:
  Utf8StreamBuffer () noexcept = default;

  /* Processes a token piece with zero heap allocations when streaming to
     stdout, or appends complete UTF-8 bytes to out_str.  */
  void
  push (std::string_view piece, bool stream_stdout,
        std::string *out_str = nullptr) noexcept;

  /* Convenience overload returning the complete UTF-8 substring ready to
     emit from this piece.  */
  [[nodiscard]] std::string
  push (std::string_view piece);

  /* Flushes any remaining held bytes at end-of-generation (EOS).  */
  void
  flush_remaining (bool stream_stdout,
                   std::string *out_str = nullptr) noexcept;

  /* Convenience overload returning any remaining held bytes at EOS.  */
  [[nodiscard]] std::string
  flush_remaining ();

  /* Clears any buffered partial UTF-8 bytes.  */
  void
  reset () noexcept;

  /* Returns the number of held trailing bytes (0..3).  */
  [[nodiscard]] std::size_t
  pending_bytes () const noexcept;

private:
  std::array<char, 4U> tail_ = {};
  std::size_t tail_len_ = 0U;
};

} /* namespace milda */

#endif /* MILDA_I18N_HPP */
