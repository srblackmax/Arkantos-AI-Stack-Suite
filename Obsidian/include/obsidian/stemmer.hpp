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

#ifndef OBSIDIAN_STEMMER_HPP
#define OBSIDIAN_STEMMER_HPP

/**
 * @file stemmer.hpp
 * @brief Header-only Martin Porter Stemmer, rule-based Lemmatizer, and policy.
 */

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/memory.hpp>

namespace obsidian
{

/**
 * @brief C++23 concept constraining morphological stemmer and lemmatizer
 *        types.
 *
 * A type S satisfies Stemmer_Policy if it can stem a word into a std::string
 * and into a zero-allocation Arena-backed std::string_view.
 */
template <typename S>
concept Stemmer_Policy = requires (const S &stemmer, std::string_view word,
                                   Arena &arena)
{
  { stemmer.stem (word) } -> std::convertible_to<std::string>;
  { stemmer.stem_into_arena (word, arena) }
    -> std::convertible_to<std::string_view>;
};

/**
 * @brief Canonical 5-step Martin Porter (1980) English morphological stemmer.
 *
 * Implemented as a re-entrant, stateless header-only class operating on stack
 * or Arena character buffers without dynamic heap allocations in the hot path.
 */
class OBSIDIAN_API Porter_Stemmer
{
public:
  static constexpr size_t k_stack_word_limit = 256;

  /**
   * @brief Default-construct a stateless Porter_Stemmer.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  constexpr Porter_Stemmer (void) noexcept = default;

  /**
   * @brief Stem a word in-place inside a mutable character buffer.
   * @pre BUF points to at least LEN valid bytes.
   * @complexity O(N) time and O(1) auxiliary space in word length N.
   */
  static size_t
  stem_inplace (char *buf, size_t len) noexcept
  {
    if (buf == nullptr || len == 0)
      {
        return 0;
      }

    for (size_t i = 0; i < len; ++i)
      {
        unsigned char c = static_cast<unsigned char> (buf[i]);
        if (c >= 'A' && c <= 'Z')
          {
            buf[i] = static_cast<char> (c + ('a' - 'A'));
          }
      }

    if (len <= 2)
      {
        return len;
      }

    /* Check that the token consists of ASCII letters before stemming.  */
    for (size_t i = 0; i < len; ++i)
      {
        unsigned char c = static_cast<unsigned char> (buf[i]);
        if (c < 'a' || c > 'z')
          {
            return len;
          }
      }

    State st{buf, static_cast<int> (len) - 1, 0};
    st.step1ab ();
    if (st.k > 0)
      {
        st.step1c ();
        st.step2 ();
        st.step3 ();
        st.step4 ();
        st.step5 ();
      }
    return static_cast<size_t> (st.k + 1);
  }

  /**
   * @brief Stem WORD into a caller-provided output buffer OUT_BUF.
   * @pre OUT_BUF.size() >= WORD.size().
   * @complexity O(N) time and O(1) space in word length N.
   */
  std::string_view
  stem_into_buffer (std::string_view word,
                    std::span<char> out_buf) const noexcept
  {
    if (word.empty () || out_buf.size () < word.size ())
      {
        return {};
      }
    std::memcpy (out_buf.data (), word.data (), word.size ());
    size_t new_len = stem_inplace (out_buf.data (), word.size ());
    return std::string_view (out_buf.data (), new_len);
  }

  /**
   * @brief Stem WORD into ARENA with zero heap allocations.
   * @pre ARENA must remain alive while the returned string_view is accessed.
   * @complexity O(N) time and O(N) arena bump space.
   */
  std::string_view
  stem_into_arena (std::string_view word, Arena &arena) const
  {
    if (word.empty ())
      {
        return {};
      }
    std::span<char> buf = arena.allocate_span<char> (word.size ());
    std::memcpy (buf.data (), word.data (), word.size ());
    size_t new_len = stem_inplace (buf.data (), word.size ());
    return std::string_view (buf.data (), new_len);
  }

  /**
   * @brief Stem WORD and return a newly constructed std::string.
   * @pre None.
   * @complexity O(N) time and O(N) space in word length N.
   */
  std::string
  stem (std::string_view word) const
  {
    if (word.empty ())
      {
        return {};
      }
    if (word.size () <= k_stack_word_limit)
      {
        std::array<char, k_stack_word_limit> stack_buf{};
        std::memcpy (stack_buf.data (), word.data (), word.size ());
        size_t new_len = stem_inplace (stack_buf.data (), word.size ());
        return std::string (stack_buf.data (), new_len);
      }
    std::string out (word);
    size_t new_len = stem_inplace (out.data (), out.size ());
    out.resize (new_len);
    return out;
  }

  /**
   * @brief Function call operator forwarding to stem(word).
   * @pre None.
   * @complexity O(N) time and O(N) space in word length N.
   */
  std::string
  operator() (std::string_view word) const
  {
    return stem (word);
  }

private:
  struct State
  {
    char *b;
    int k;
    int j;

    bool
    cons (int i) const noexcept
    {
      switch (b[i])
        {
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
          return false;
        case 'y':
          return (i == 0) ? true : !cons (i - 1);
        default:
          return true;
        }
    }

    int
    m (void) const noexcept
    {
      int n = 0;
      int i = 0;
      while (true)
        {
          if (i > j)
            {
              return n;
            }
          if (!cons (i))
            {
              break;
            }
          ++i;
        }
      ++i;
      while (true)
        {
          while (true)
            {
              if (i > j)
                {
                  return n;
                }
              if (cons (i))
                {
                  break;
                }
              ++i;
            }
          ++i;
          ++n;
          while (true)
            {
              if (i > j)
                {
                  return n;
                }
              if (!cons (i))
                {
                  break;
                }
              ++i;
            }
          ++i;
        }
    }

    bool
    vowelinstem (void) const noexcept
    {
      for (int i = 0; i <= j; ++i)
        {
          if (!cons (i))
            {
              return true;
            }
        }
      return false;
    }

    bool
    doublec (int i) const noexcept
    {
      if (i < 1)
        {
          return false;
        }
      if (b[i] != b[i - 1])
        {
          return false;
        }
      return cons (i);
    }

    bool
    cvc (int i) const noexcept
    {
      if (i < 2 || !cons (i) || cons (i - 1) || !cons (i - 2))
        {
          return false;
        }
      char ch = b[i];
      if (ch == 'w' || ch == 'x' || ch == 'y')
        {
          return false;
        }
      return true;
    }

    bool
    ends (std::string_view s) noexcept
    {
      int length = static_cast<int> (s.size ());
      if (length > k + 1)
        {
          return false;
        }
      if (std::memcmp (b + k - length + 1, s.data (), s.size ()) != 0)
        {
          return false;
        }
      j = k - length;
      return true;
    }

    void
    setto (std::string_view s) noexcept
    {
      int length = static_cast<int> (s.size ());
      if (length > 0)
        {
          std::memmove (b + j + 1, s.data (), s.size ());
        }
      k = j + length;
    }

    void
    r (std::string_view s) noexcept
    {
      if (m () > 0)
        {
          setto (s);
        }
    }

    void
    step1ab (void) noexcept
    {
      if (b[k] == 's')
        {
          if (ends ("sses"))
            {
              k -= 2;
            }
          else if (ends ("ies"))
            {
              setto ("i");
            }
          else if (b[k - 1] != 's')
            {
              --k;
            }
        }
      if (ends ("eed"))
        {
          if (m () > 0)
            {
              --k;
            }
        }
      else if ((ends ("ed") || ends ("ing")) && vowelinstem ())
        {
          k = j;
          if (ends ("at"))
            {
              setto ("ate");
            }
          else if (ends ("bl"))
            {
              setto ("ble");
            }
          else if (ends ("iz"))
            {
              setto ("ize");
            }
          else if (doublec (k))
            {
              --k;
              char ch = b[k];
              if (ch == 'l' || ch == 's' || ch == 'z')
                {
                  ++k;
                }
            }
          else if (m () == 1 && cvc (k))
            {
              setto ("e");
            }
        }
    }

    void
    step1c (void) noexcept
    {
      if (ends ("y") && vowelinstem ())
        {
          b[k] = 'i';
        }
    }

    void
    step2 (void) noexcept
    {
      if (k < 1)
        {
          return;
        }
      switch (b[k - 1])
        {
        case 'a':
          if (ends ("ational"))
            {
              r ("ate");
              break;
            }
          if (ends ("tional"))
            {
              r ("tion");
              break;
            }
          break;
        case 'c':
          if (ends ("enci"))
            {
              r ("ence");
              break;
            }
          if (ends ("anci"))
            {
              r ("ance");
              break;
            }
          break;
        case 'e':
          if (ends ("izer"))
            {
              r ("ize");
              break;
            }
          break;
        case 'l':
          if (ends ("bli"))
            {
              r ("ble");
              break;
            }
          if (ends ("abli"))
            {
              r ("able");
              break;
            }
          if (ends ("alli"))
            {
              r ("al");
              break;
            }
          if (ends ("entli"))
            {
              r ("ent");
              break;
            }
          if (ends ("eli"))
            {
              r ("e");
              break;
            }
          if (ends ("ousli"))
            {
              r ("ous");
              break;
            }
          break;
        case 'o':
          if (ends ("ization"))
            {
              r ("ize");
              break;
            }
          if (ends ("ation"))
            {
              r ("ate");
              break;
            }
          if (ends ("ator"))
            {
              r ("ate");
              break;
            }
          break;
        case 's':
          if (ends ("alism"))
            {
              r ("al");
              break;
            }
          if (ends ("iveness"))
            {
              r ("ive");
              break;
            }
          if (ends ("fulness"))
            {
              r ("ful");
              break;
            }
          if (ends ("ousness"))
            {
              r ("ous");
              break;
            }
          break;
        case 't':
          if (ends ("aliti"))
            {
              r ("al");
              break;
            }
          if (ends ("iviti"))
            {
              r ("ive");
              break;
            }
          if (ends ("biliti"))
            {
              r ("ble");
              break;
            }
          break;
        case 'g':
          if (ends ("logi"))
            {
              r ("log");
              break;
            }
          break;
        default:
          break;
        }
    }

    void
    step3 (void) noexcept
    {
      switch (b[k])
        {
        case 'e':
          if (ends ("icate"))
            {
              r ("ic");
              break;
            }
          if (ends ("ative"))
            {
              r ("");
              break;
            }
          if (ends ("alize"))
            {
              r ("al");
              break;
            }
          break;
        case 'i':
          if (ends ("iciti"))
            {
              r ("ic");
              break;
            }
          break;
        case 'l':
          if (ends ("ical"))
            {
              r ("ic");
              break;
            }
          if (ends ("ful"))
            {
              r ("");
              break;
            }
          break;
        case 's':
          if (ends ("ness"))
            {
              r ("");
              break;
            }
          break;
        default:
          break;
        }
    }

    void
    step4 (void) noexcept
    {
      if (k < 1)
        {
          return;
        }
      switch (b[k - 1])
        {
        case 'a':
          if (ends ("al"))
            {
              break;
            }
          return;
        case 'c':
          if (ends ("ance"))
            {
              break;
            }
          if (ends ("ence"))
            {
              break;
            }
          return;
        case 'e':
          if (ends ("er"))
            {
              break;
            }
          return;
        case 'i':
          if (ends ("ic"))
            {
              break;
            }
          return;
        case 'l':
          if (ends ("able"))
            {
              break;
            }
          if (ends ("ible"))
            {
              break;
            }
          return;
        case 'n':
          if (ends ("ant"))
            {
              break;
            }
          if (ends ("ement"))
            {
              break;
            }
          if (ends ("ment"))
            {
              break;
            }
          if (ends ("ent"))
            {
              break;
            }
          return;
        case 'o':
          if (ends ("ion") && j >= 0 && (b[j] == 's' || b[j] == 't'))
            {
              break;
            }
          if (ends ("ou"))
            {
              break;
            }
          return;
        case 's':
          if (ends ("ism"))
            {
              break;
            }
          return;
        case 't':
          if (ends ("ate"))
            {
              break;
            }
          if (ends ("iti"))
            {
              break;
            }
          return;
        case 'u':
          if (ends ("ous"))
            {
              break;
            }
          return;
        case 'v':
          if (ends ("ive"))
            {
              break;
            }
          return;
        case 'z':
          if (ends ("ize"))
            {
              break;
            }
          return;
        default:
          return;
        }
      if (m () > 1)
        {
          k = j;
        }
    }

    void
    step5 (void) noexcept
    {
      j = k;
      if (b[k] == 'e')
        {
          int a = m ();
          if (a > 1 || (a == 1 && !cvc (k - 1)))
            {
              --k;
            }
        }
      if (b[k] == 'l' && doublec (k) && m () > 1)
        {
          --k;
        }
    }
  };
};

/**
 * @brief Rule-based English morphological lemmatizer with irregular table.
 *
 * Combines a fast hash table of irregular English inflections (verbs, nouns,
 * adjectives) with part-of-speech-aware morphological suffix rules.
 */
class OBSIDIAN_API Lemmatizer
{
public:
  /**
   * @brief Construct a Lemmatizer preloaded with core English irregular forms.
   * @pre None.
   * @complexity O(I) time and space where I is the default irregular count.
   */
  Lemmatizer (void)
  {
    load_default_irregulars ();
  }

  /**
   * @brief Register or override a custom irregular inflection mapping.
   * @pre INFLECTED and LEMMA must be non-empty.
   * @complexity O(N) amortized time in word length N.
   */
  void
  add_irregular (std::string_view inflected, std::string_view lemma)
  {
    if (inflected.empty () || lemma.empty ())
      {
        return;
      }
    irregulars_[to_lower_ascii (inflected)] = to_lower_ascii (lemma);
  }

  /**
   * @brief Lemmatize WORD given an optional Penn-style or Universal POS_TAG.
   * @pre None.
   * @complexity O(N) time and O(N) space in word length N.
   */
  std::string
  lemmatize (std::string_view word, std::string_view pos_tag = "") const
  {
    if (word.empty ())
      {
        return {};
      }

    std::string lower = to_lower_ascii (word);
    auto it = irregulars_.find (lower);
    if (it != irregulars_.end ())
      {
        return it->second;
      }

    if (lower.size () <= 3)
      {
        return lower;
      }

    char pos = '\0';
    if (!pos_tag.empty ())
      {
        char c = pos_tag[0];
        if (c >= 'a' && c <= 'z')
          {
            c = static_cast<char> (c - ('a' - 'A'));
          }
        pos = c;
      }

    /* Verb morphological rules (POS starting with V, or untagged).  */
    if (pos == 'V' || pos == '\0')
      {
        if (ends_with (lower, "ies") && lower.size () > 4)
          {
            return lower.substr (0, lower.size () - 3) + "y";
          }
        if (ends_with (lower, "ied") && lower.size () > 4)
          {
            return lower.substr (0, lower.size () - 3) + "y";
          }
        if (ends_with (lower, "ying") && lower.size () >= 5)
          {
            if (lower.size () == 5)
              {
                return lower.substr (0, lower.size () - 4) + "ie";
              }
            return lower.substr (0, lower.size () - 3);
          }
        if (ends_with (lower, "ing") && lower.size () > 4)
          {
            std::string base = lower.substr (0, lower.size () - 3);
            if (has_doubled_final_consonant (base))
              {
                base.pop_back ();
                return base;
              }
            if (needs_silent_e (base))
              {
                return base + "e";
              }
            return base;
          }
        if (ends_with (lower, "ed") && !ends_with (lower, "eed")
            && lower.size () > 3)
          {
            std::string base = lower.substr (0, lower.size () - 2);
            if (has_doubled_final_consonant (base))
              {
                base.pop_back ();
                return base;
              }
            if (needs_silent_e (base) || ends_with (base, "at")
                || ends_with (base, "iz") || ends_with (base, "ul")
                || ends_with (base, "ur") || ends_with (base, "ut"))
              {
                return base + "e";
              }
            return base;
          }
        if (pos == 'V')
          {
            if (ends_with (lower, "es")
                && (ends_with (lower, "ches") || ends_with (lower, "shes")
                    || ends_with (lower, "xes") || ends_with (lower, "zes")
                    || ends_with (lower, "sses")))
              {
                return lower.substr (0, lower.size () - 2);
              }
            if (ends_with (lower, "s") && !ends_with (lower, "ss")
                && lower.size () > 3)
              {
                return lower.substr (0, lower.size () - 1);
              }
            return lower;
          }
      }

    /* Adjective and adverb rules (POS starting with J, A, or R).  */
    if (pos == 'J' || pos == 'A' || pos == 'R')
      {
        if (ends_with (lower, "ier") && lower.size () > 4)
          {
            return lower.substr (0, lower.size () - 3) + "y";
          }
        if (ends_with (lower, "iest") && lower.size () > 5)
          {
            return lower.substr (0, lower.size () - 4) + "y";
          }
        if (ends_with (lower, "er") && lower.size () > 4)
          {
            std::string base = lower.substr (0, lower.size () - 2);
            if (has_doubled_final_consonant (base))
              {
                base.pop_back ();
              }
            else if (needs_silent_e (base))
              {
                base.push_back ('e');
              }
            return base;
          }
        if (ends_with (lower, "est") && lower.size () > 5)
          {
            std::string base = lower.substr (0, lower.size () - 3);
            if (has_doubled_final_consonant (base))
              {
                base.pop_back ();
              }
            else if (needs_silent_e (base))
              {
                base.push_back ('e');
              }
            return base;
          }
        if (pos == 'R' && ends_with (lower, "ily") && lower.size () > 4)
          {
            return lower.substr (0, lower.size () - 3) + "y";
          }
        return lower;
      }

    /* Noun plural morphological rules (POS starting with N or untagged).  */
    if (ends_with (lower, "ies") && lower.size () > 4)
      {
        return lower.substr (0, lower.size () - 3) + "y";
      }
    if (ends_with (lower, "ves") && lower.size () > 4)
      {
        std::string prefix = lower.substr (0, lower.size () - 3);
        if (ends_with (prefix, "kn") || ends_with (prefix, "w")
            || ends_with (prefix, "l"))
          {
            return prefix + "fe";
          }
        return prefix + "f";
      }
    if (ends_with (lower, "ches") || ends_with (lower, "shes")
        || ends_with (lower, "xes") || ends_with (lower, "zes")
        || ends_with (lower, "sses"))
      {
        return lower.substr (0, lower.size () - 2);
      }
    if (ends_with (lower, "men") && lower.size () > 4)
      {
        return lower.substr (0, lower.size () - 3) + "man";
      }
    if (ends_with (lower, "s") && !ends_with (lower, "ss")
        && !ends_with (lower, "us") && !ends_with (lower, "is")
        && lower.size () > 3)
      {
        return lower.substr (0, lower.size () - 1);
      }

    return lower;
  }

  /**
   * @brief Stem/lemmatize WORD without a POS tag (satisfies Stemmer_Policy).
   * @pre None.
   * @complexity O(N) time and O(N) space in word length N.
   */
  std::string
  stem (std::string_view word) const
  {
    return lemmatize (word);
  }

  /**
   * @brief Lemmatize WORD and copy the result into ARENA.
   * @pre ARENA must remain alive while the returned string_view is accessed.
   * @complexity O(N) time and O(N) arena bump space.
   */
  std::string_view
  stem_into_arena (std::string_view word, Arena &arena) const
  {
    std::string res = lemmatize (word);
    if (res.empty ())
      {
        return {};
      }
    std::span<char> buf = arena.allocate_span<char> (res.size ());
    std::memcpy (buf.data (), res.data (), res.size ());
    return std::string_view (buf.data (), res.size ());
  }

private:
  std::unordered_map<std::string, std::string> irregulars_;

  static std::string
  to_lower_ascii (std::string_view s)
  {
    std::string out (s);
    for (char &ch : out)
      {
        unsigned char u = static_cast<unsigned char> (ch);
        if (u >= 'A' && u <= 'Z')
          {
            ch = static_cast<char> (u + ('a' - 'A'));
          }
      }
    return out;
  }

  static bool
  ends_with (std::string_view s, std::string_view suffix) noexcept
  {
    return s.size () >= suffix.size ()
           && s.substr (s.size () - suffix.size ()) == suffix;
  }

  static bool
  is_vowel (char c) noexcept
  {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
  }

  static bool
  has_doubled_final_consonant (std::string_view s) noexcept
  {
    if (s.size () < 3)
      {
        return false;
      }
    char last = s.back ();
    char prev = s[s.size () - 2];
    if (last != prev || is_vowel (last))
      {
        return false;
      }
    if (last == 'l' || last == 's' || last == 'z' || last == 'f')
      {
        return false;
      }
    return true;
  }

  static bool
  needs_silent_e (std::string_view base) noexcept
  {
    if (base.size () < 3)
      {
        return false;
      }
    char c1 = base[base.size () - 3];
    char v = base[base.size () - 2];
    char c2 = base[base.size () - 1];
    if (!is_vowel (c1) && is_vowel (v) && !is_vowel (c2)
        && c2 != 'w' && c2 != 'x' && c2 != 'y')
      {
        if (c2 == 'v' || c2 == 'c' || c2 == 'g' || c2 == 'z' || c2 == 'k'
            || (base.size () == 3 && (v == 'a' || v == 'i' || v == 'o')))
          {
            return true;
          }
      }
    return false;
  }

  void
  load_default_irregulars (void)
  {
    struct Pair
    {
      const char *inflected;
      const char *lemma;
    };
    static constexpr Pair k_defaults[] = {
      {"am", "be"}, {"is", "be"}, {"are", "be"}, {"was", "be"},
      {"were", "be"}, {"been", "be"}, {"being", "be"},
      {"has", "have"}, {"had", "have"}, {"having", "have"},
      {"does", "do"}, {"did", "do"}, {"done", "do"}, {"doing", "do"},
      {"went", "go"}, {"gone", "go"}, {"goes", "go"}, {"going", "go"},
      {"ran", "run"}, {"running", "run"}, {"runs", "run"},
      {"ate", "eat"}, {"eaten", "eat"}, {"saw", "see"}, {"seen", "see"},
      {"took", "take"}, {"taken", "take"}, {"gave", "give"},
      {"given", "give"}, {"wrote", "write"}, {"written", "write"},
      {"spoke", "speak"}, {"spoken", "speak"}, {"broke", "break"},
      {"broken", "break"}, {"chose", "choose"}, {"chosen", "choose"},
      {"drove", "drive"}, {"driven", "drive"}, {"fell", "fall"},
      {"fallen", "fall"}, {"flew", "fly"}, {"flown", "fly"},
      {"knew", "know"}, {"known", "know"}, {"threw", "throw"},
      {"thrown", "throw"}, {"grew", "grow"}, {"grown", "grow"},
      {"drew", "draw"}, {"drawn", "draw"}, {"bought", "buy"},
      {"brought", "bring"}, {"thought", "think"}, {"caught", "catch"},
      {"taught", "teach"}, {"sought", "seek"}, {"fought", "fight"},
      {"found", "find"}, {"held", "hold"}, {"told", "tell"},
      {"sold", "sell"}, {"built", "build"}, {"sent", "send"},
      {"spent", "spend"}, {"left", "leave"}, {"kept", "keep"},
      {"slept", "sleep"}, {"felt", "feel"}, {"meant", "mean"},
      {"met", "meet"}, {"led", "lead"}, {"lost", "lose"},
      {"paid", "pay"}, {"said", "say"}, {"made", "make"},
      {"stood", "stand"}, {"understood", "understand"}, {"won", "win"},
      {"sat", "sit"}, {"began", "begin"}, {"begun", "begin"},
      {"sang", "sing"}, {"sung", "sing"}, {"swam", "swim"},
      {"swum", "swim"}, {"drank", "drink"}, {"drunk", "drink"},
      {"children", "child"}, {"men", "man"}, {"women", "woman"},
      {"mice", "mouse"}, {"geese", "goose"}, {"teeth", "tooth"},
      {"feet", "foot"}, {"oxen", "ox"}, {"people", "person"},
      {"leaves", "leaf"}, {"lives", "life"}, {"wives", "wife"},
      {"knives", "knife"}, {"wolves", "wolf"}, {"halves", "half"},
      {"selves", "self"}, {"shelves", "shelf"}, {"thieves", "thief"},
      {"loaves", "loaf"}, {"calves", "calf"},
      {"indices", "index"}, {"matrices", "matrix"}, {"vertices", "vertex"},
      {"analyses", "analysis"}, {"crises", "crisis"}, {"theses", "thesis"},
      {"phenomena", "phenomenon"}, {"criteria", "criterion"},
      {"corpora", "corpus"}, {"better", "good"}, {"best", "good"},
      {"worse", "bad"}, {"worst", "bad"}
    };
    irregulars_.reserve (128);
    for (const auto &p : k_defaults)
      {
        irregulars_.emplace (p.inflected, p.lemma);
      }
  }
};

static_assert (Stemmer_Policy<Porter_Stemmer>);
static_assert (Stemmer_Policy<Lemmatizer>);

} /* namespace obsidian */

#endif /* OBSIDIAN_STEMMER_HPP */
