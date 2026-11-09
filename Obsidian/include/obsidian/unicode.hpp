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

#ifndef OBSIDIAN_UNICODE_HPP
#define OBSIDIAN_UNICODE_HPP

/**
 * @file unicode.hpp
 * @brief UTF-8 codepoint iteration, NFKC compatibility normalization,
 *        Unicode general category predicates, and reversible GPT-2/LLaMA
 *        256-byte-to-Unicode encoding for Byte-Level BPE.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>

namespace obsidian::unicode
{

/**
 * @brief Decode a single UTF-8 codepoint starting at OFFSET and advance
 *        OFFSET to the next codepoint boundary.
 * @pre OFFSET < TEXT.size ().
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API Result<uint32_t>
decode_utf8_codepoint (std::string_view text, size_t &offset) noexcept;

/**
 * @brief Encode a Unicode scalar value CP as UTF-8 and append it to OUT.
 * @pre CP <= 0x10FFFF and CP is not a UTF-16 surrogate.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API void
append_utf8_codepoint (uint32_t cp, std::string &out);

/**
 * @brief Return true if CP is an ASCII or Unicode whitespace separator.
 * @pre None.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API bool
is_unicode_whitespace (uint32_t cp) noexcept;

/**
 * @brief Return true if CP is an ASCII or general Unicode punctuation
 *        character.
 * @pre None.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API bool
is_unicode_punctuation (uint32_t cp) noexcept;

/**
 * @brief Return true if CP is an ignorable control or zero-width format
 *        character (excluding standard tab, newline, and carriage return).
 * @pre None.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API bool
is_ignorable_control (uint32_t cp) noexcept;

/**
 * @brief Normalize UTF-8 TEXT using NFKC compatibility folding (fullwidth
 *        ASCII, typographic quotes/dashes, ligatures, whitespace collapsing,
 *        and optional accent stripping).
 * @pre TEXT must be valid UTF-8.
 * @complexity O(N) time and space in byte length N.
 */
OBSIDIAN_API Result<std::string>
normalize_nfkc_text (std::string_view text, bool strip_accents = false);

/**
 * @brief Map arbitrary raw bytes to printable UTF-8 codepoints using the
 *        standard GPT-2 / LLaMA 256-byte bijection so Byte-Level BPE never
 *        emits out-of-vocabulary tokens.
 * @pre None (accepts arbitrary binary or UTF-8 bytes).
 * @complexity O(N) time and space in byte length N.
 */
OBSIDIAN_API std::string
byte_to_unicode_encode (std::string_view raw_bytes);

/**
 * @brief Decode a GPT-2 / LLaMA byte-level UTF-8 token string back into its
 *        original raw byte sequence.
 * @pre ENCODED_UTF8 must contain only valid GPT-2 byte-level codepoints.
 * @complexity O(N) time and space in byte length N.
 */
OBSIDIAN_API Result<std::string>
byte_to_unicode_decode (std::string_view encoded_utf8);

} /* namespace obsidian::unicode */

#endif /* OBSIDIAN_UNICODE_HPP */
