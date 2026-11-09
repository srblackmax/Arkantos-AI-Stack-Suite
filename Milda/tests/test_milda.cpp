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

#include "milda/engine.hpp"
#include "milda/esperanto.hpp"
#include "milda/hardware.hpp"
#include "milda/milda_i18n.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

[[nodiscard]] std::string
run_x_system (std::string_view input)
{
  std::string buf (input);
  const std::size_t new_len
      = milda::transliterate_x_system_in_place (buf.data (), buf.size ());
  buf.resize (new_len);
  return buf;
}

[[nodiscard]] std::string
run_preprocess (std::string_view input, milda::MildaLang lang_mode,
                milda::MildaLang *out_detected = nullptr,
                milda::x_system_mode x_mode
                = milda::x_system_mode::auto_lang)
{
  std::string buf (input);
  const std::size_t new_len = milda::preprocess_user_input_in_place (
      buf.data (), buf.size (), lang_mode, out_detected, x_mode);
  buf.resize (new_len);
  return buf;
}

[[nodiscard]] bool
test_esperanto_layer ()
{
  constexpr std::string_view expected_prompt
      = "Vi estas Milda, varma, sprita kaj pensiga homa konversacianto kaj "
        "proksima amiko. Babilu nature, kore kaj egale kun via amiko en "
        "simplaj, fluaj konversaciaj alineoj\342\200\224neniam kiel roboto, "
        "AI-asistanto a\305\255 programada agento. Neniam uzu "
        "Markdown-formatadon (*, -, listojn a\305\255 kodblokojn), "
        "miensignojn (emojis), a\305\255 robotajn frazojn, kaj neniam "
        "demandu 'Kiel mi povas helpi vin?'. Respondu rekte kaj koncize "
        "ekskluzive en idiomeca Esperanto la\305\255 la Fundamento kun "
        "\304\235ustaj akuzativoj (-n), supersignoj (\304\211, \304\235, "
        "\304\245, \304\265, \305\235, \305\255), kaj korektaj tabelvortoj "
        "(iu, io, kia, kio), sen miksi aliajn lingvojn.";

  if (milda::k_esperanto_system_prompt != expected_prompt)
    {
      (void)std::fprintf (stderr,
                          "FAIL: k_esperanto_system_prompt mismatch\n");
      return false;
    }

  std::string filtered;
  const std::vector<std::string_view> gemma_pieces =
    {
      "<|channel>",
      "thought",
      "\nThinking in English...\n",
      "<channel|>",
      "\nSaluton, amiko!",
      "<turn|>",
      "IGNORED"
    };
  if (!milda::filter_thought_pieces (gemma_pieces, &filtered)
      || filtered != "Saluton, amiko!")
    {
      (void)std::fprintf (stderr, "FAIL: Gemma 4 thought stream filter\n");
      return false;
    }

  const std::vector<std::string_view> gemma_e4b_pieces =
    {
      "The",
      " user",
      " said \"Saluton!\" (Hello!). I need to respond naturally, friendly, "
      "and in fluent Esperanto, adhering to the persona rules.",
      "<channel|>",
      "Saluton! Kiel vi fartas hodia\305\255?",
      "<turn|>"
    };
  if (!milda::filter_thought_pieces (gemma_e4b_pieces, &filtered)
      || filtered != "Saluton! Kiel vi fartas hodia\305\255?")
    {
      (void)std::fprintf (stderr, "FAIL: Gemma 4 E4B CoT leak filter\n");
      return false;
    }

  const std::vector<std::string_view> bare_thought_pieces =
    {
      "thought\n",
      "Internal English thought...\n",
      "<channel|>",
      "\nSaluton!",
      "<turn|>"
    };
  if (!milda::filter_thought_pieces (bare_thought_pieces, &filtered)
      || filtered != "Saluton!")
    {
      (void)std::fprintf (stderr, "FAIL: Gemma 4 bare thought filter\n");
      return false;
    }

  const std::vector<std::string_view> qwen_pieces =
    {
      "<think>",
      "\nInternal reasoning...\n",
      "</think>",
      "\n\nBonan tagon!",
      "<|im_end|>",
      "IGNORED"
    };
  if (!milda::filter_thought_pieces (qwen_pieces, &filtered)
      || filtered != "Bonan tagon!")
    {
      (void)std::fprintf (stderr, "FAIL: Qwen3 think stream filter\n");
      return false;
    }

  const std::vector<std::string_view> english_the_pieces =
    {
      "The",
      " largest",
      " planet in the solar system is Jupiter.",
      "<|im_end|>"
    };
  if (!milda::filter_thought_pieces (english_the_pieces, &filtered)
      || filtered != "The largest planet in the solar system is Jupiter.")
    {
      (void)std::fprintf (stderr,
                          "FAIL: English 'The ' prefix falsely suppressed\n");
      return false;
    }

  if (run_x_system ("cx gx hx jx sx ux")
      != "\xc4\x89 \xc4\x9d \xc4\xa5 \xc4\xb5 \xc5\x9d \xc5\xad")
    {
      (void)std::fprintf (stderr, "FAIL: lowercase x-system conversion\n");
      return false;
    }

  if (run_x_system ("Cx Gx Hx Jx Sx Ux")
      != "\xc4\x88 \xc4\x9c \xc4\xa4 \xc4\xb4 \xc5\x9c \xc5\xac")
    {
      (void)std::fprintf (stderr, "FAIL: mixed-case x-system conversion\n");
      return false;
    }

  if (run_x_system ("CX GX HX JX SX UX")
      != "\xc4\x88 \xc4\x9c \xc4\xa4 \xc4\xb4 \xc5\x9c \xc5\xac")
    {
      (void)std::fprintf (stderr, "FAIL: uppercase X-system conversion\n");
      return false;
    }

  if (run_x_system ("cxx gxx uxx Bordeauxx xx") != "cx gx ux Bordeaux x")
    {
      (void)std::fprintf (stderr, "FAIL: xx escape handling\n");
      return false;
    }

  if (run_x_system ("Ehxosxangxo cxiumauxde")
      != "E\304\245o\305\235an\304\235o \304\211iuma\305\255de")
    {
      (void)std::fprintf (stderr, "FAIL: Ehxosxangxo cxiumauxde\n");
      return false;
    }

  if (milda::parse_repl_command ("  /eliri \n")
          != milda::repl_command_kind::eliri
      || milda::parse_repl_command ("/vakigi")
             != milda::repl_command_kind::vakigi
      || milda::parse_repl_command ("/stat") != milda::repl_command_kind::stat
      || milda::parse_repl_command ("/konservi mem.bin")
             != milda::repl_command_kind::konservi
      || milda::parse_repl_command ("/sxargi mem.bin")
             != milda::repl_command_kind::sxargi
      || milda::parse_repl_command ("/lang zh-Hans")
             != milda::repl_command_kind::lang
      || milda::parse_repl_command ("/x-system on")
             != milda::repl_command_kind::x_system
      || milda::extract_repl_command_arg ("/konservi  mem.bin ") != "mem.bin"
      || milda::extract_repl_command_arg ("/lang  zh ") != "zh"
      || milda::parse_repl_command ("Saluton")
             != milda::repl_command_kind::none)
    {
      (void)std::fprintf (stderr, "FAIL: parse_repl_command\n");
      return false;
    }

  return true;
}

[[nodiscard]] bool
test_polyglot_layer ()
{
  /* 1. Language registry and code parsing.  */
  milda::MildaLang lang = milda::MildaLang::Esperanto;
  if (!milda::parse_milda_lang ("auto", &lang)
      || lang != milda::MildaLang::Auto
      || !milda::parse_milda_lang ("eo", &lang)
      || lang != milda::MildaLang::Esperanto
      || !milda::parse_milda_lang ("en", &lang)
      || lang != milda::MildaLang::English
      || !milda::parse_milda_lang ("es", &lang)
      || lang != milda::MildaLang::Spanish
      || !milda::parse_milda_lang ("zh", &lang)
      || lang != milda::MildaLang::MandarinSimplified
      || !milda::parse_milda_lang ("zh-Hans", &lang)
      || lang != milda::MildaLang::MandarinSimplified
      || !milda::parse_milda_lang ("ru", &lang)
      || lang != milda::MildaLang::Russian
      || milda::parse_milda_lang ("invalid-lang", &lang))
    {
      (void)std::fprintf (stderr, "FAIL: parse_milda_lang\n");
      return false;
    }

  constexpr std::string_view zh_required
      = "你是 Milda（米尔达），一位温暖、敏锐、"
        "真诚的人类倾听者与挚友。";
  const std::string_view zh_prompt
      = milda::get_system_prompt (milda::MildaLang::MandarinSimplified);
  if (zh_prompt.find (zh_required) == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "FAIL: zh-Hans system prompt mismatch\n");
      return false;
    }

  /* 2. Fast O(N) Unicode script & function-word language detector.  */
  if (milda::detect_prompt_language (
          "\344\275\240\345\245\275\357\274\214\344\270\226\347\225\214")
          != milda::MildaLang::MandarinSimplified
      || milda::detect_prompt_language (
             "\320\237\321\200\320\270\320\262\320\265\321\202, "
             "\320\274\320\270\321\200!")
             != milda::MildaLang::Russian
      || milda::detect_prompt_language (
             "Saluton! Kiel vi fartas hodia\305\255?")
             != milda::MildaLang::Esperanto
      || milda::detect_prompt_language ("Mi estas kun amiko en la urbo")
             != milda::MildaLang::Esperanto
      || milda::detect_prompt_language ("\302\277C\303\263mo est\303\241s en "
                                        "Espa\303\261a?")
             != milda::MildaLang::Spanish
      || milda::detect_prompt_language ("El libro es para la biblioteca")
             != milda::MildaLang::Spanish
      || milda::detect_prompt_language (
             "How does the auxiliary Linux memory allocator work?")
             != milda::MildaLang::English)
    {
      (void)std::fprintf (stderr, "FAIL: detect_prompt_language\n");
      return false;
    }

  /* 3. Multi-Byte UTF-8 Token Streaming Buffer across 1-byte and 2-byte
     splits for 3-byte Chinese and 2-byte Russian/Esperanto/Spanish.  */
  const std::string_view multi_utf8
      = "\344\275\240\345\245\275\357\274\214\344\270\226\347\225\214\357\274"
        "\201 \320\237\321\200\320\270\320\262\320\265\321\202! "
        "E\304\245o\305\235an\304\235o \302\241Espa\303\261ol!";
  milda::Utf8StreamBuffer stream_buf;
  std::string reassembled;
  for (std::size_t i = 0U; i < multi_utf8.size (); ++i)
    {
      const std::string ready
          = stream_buf.push (multi_utf8.substr (i, 1U));
      /* Every emitted chunk must end on a complete UTF-8 codepoint.  */
      if (!ready.empty ())
        {
          const auto last_b = static_cast<unsigned char> (ready.back ());
          if ((last_b & 0xC0U) == 0xC0U)
            {
              (void)std::fprintf (
                  stderr, "FAIL: Utf8StreamBuffer emitted lead byte\n");
              return false;
            }
          reassembled.append (ready);
        }
    }
  reassembled.append (stream_buf.flush_remaining ());
  if (reassembled != multi_utf8)
    {
      (void)std::fprintf (stderr,
                          "FAIL: Utf8StreamBuffer 1-byte split mismatch\n");
      return false;
    }

  /* Also verify thought_stream_filter preserves split 3-byte Chinese
     tokens without corruption.  */
  std::string zh_filtered;
  const std::vector<std::string_view> split_zh_pieces =
    {
      "<think>",
      "Reasoning...",
      "</think>",
      "\n\344\275",
      "\240\345\245",
      "\275\357\274\201",
      "<|im_end|>"
    };
  if (!milda::filter_thought_pieces (split_zh_pieces, &zh_filtered)
      || zh_filtered != "\344\275\240\345\245\275\357\274\201")
    {
      (void)std::fprintf (stderr,
                          "FAIL: filter_thought_pieces split zh-Hans\n");
      return false;
    }

  /* 4. X-system isolation across eo, en, es, zh-Hans, and ru.  */
  if (run_preprocess ("Saluton cxiam", milda::MildaLang::Esperanto)
          != "Saluton \304\211iam"
      || run_preprocess ("Saluton kaj ehxosxangxo", milda::MildaLang::Auto)
             != "Saluton kaj e\304\245o\305\235an\304\235o"
      || run_preprocess ("The auxiliary Linux box", milda::MildaLang::English)
             != "The auxiliary Linux box"
      || run_preprocess ("The auxiliary Linux box", milda::MildaLang::Auto)
             != "The auxiliary Linux box"
      || run_preprocess ("El examen exacto y auxilio",
                         milda::MildaLang::Spanish)
             != "El examen exacto y auxilio"
      || run_preprocess ("\344\275\240\345\245\275 xiao ux cx",
                         milda::MildaLang::MandarinSimplified)
             != "\344\275\240\345\245\275 xiao ux cx"
      || run_preprocess ("\320\237\321\200\320\270\320\262\320\265\321\202 "
                         "aux ux cx",
                         milda::MildaLang::Russian)
             != "\320\237\321\200\320\270\320\262\320\265\321\202 aux ux cx"
      || run_preprocess ("Saluton cxiam", milda::MildaLang::Esperanto,
                         nullptr, milda::x_system_mode::forced_off)
             != "Saluton cxiam"
      || run_preprocess ("Hello cxiam", milda::MildaLang::English, nullptr,
                         milda::x_system_mode::forced_on)
             != "Hello \304\211iam")
    {
      (void)std::fprintf (stderr, "FAIL: x-system language gating\n");
      return false;
    }

  /* 5. Per-language sampler hyperparameter tuning invariants.  */
  const milda::LangMetadata &eo_meta
      = milda::get_lang_metadata (milda::MildaLang::Esperanto);
  const milda::LangMetadata &ru_meta
      = milda::get_lang_metadata (milda::MildaLang::Russian);
  const milda::LangMetadata &zh_meta
      = milda::get_lang_metadata (milda::MildaLang::MandarinSimplified);
  if (eo_meta.repeat_penalty < 1.05F
      || eo_meta.repeat_penalty > 1.08F
      || ru_meta.repeat_penalty < 1.05F
      || ru_meta.repeat_penalty > 1.08F
      || zh_meta.top_p < 0.85F
      || zh_meta.top_p > 0.90F)
    {
      (void)std::fprintf (stderr, "FAIL: per-language sampler metadata\n");
      return false;
    }

  /* 6. Explicit language switch gating & token script lock assertion.  */
  milda::MildaLang req_lang = milda::MildaLang::English;
  if (milda::detect_explicit_language_request (
          "Deseo saber si hablas espa\303\261ol", &req_lang)
      || milda::detect_explicit_language_request (
             "\320\257 \320\263\320\276\320\262\320\276\321\200\321\216 "
             "\320\277\320\276-\321\200\321\203\321\201\321\201\320\272"
             "\320\270",
             &req_lang)
      || !milda::detect_explicit_language_request ("Please speak in Russian",
                                                   &req_lang)
      || req_lang != milda::MildaLang::Russian
      || !milda::detect_explicit_language_request ("Habla en espa\303\261ol",
                                                   &req_lang)
      || req_lang != milda::MildaLang::Spanish
      || !milda::is_token_violating_language_lock (" happening",
                                                   milda::MildaLang::Russian)
      || !milda::is_token_violating_language_lock (" \360\237\230",
                                                   milda::MildaLang::English)
      || !milda::is_token_violating_language_lock (" \360\237\230",
                                                   milda::MildaLang::Auto)
      || milda::is_token_violating_language_lock (
             " \320\266\320\270\320\267\320\275\320\270",
             milda::MildaLang::Russian))
    {
      (void)std::fprintf (stderr, "FAIL: language lock assertions\n");
      return false;
    }

  return true;
}

[[nodiscard]] bool
test_hardware_layer ()
{
  const milda::cpu_topology_info topo = milda::detect_cpu_topology ();
  if (topo.p_core_cpus.empty () || topo.all_physical_cpus.empty ()
      || topo.all_physical_cpus.size ()
             != topo.p_core_cpus.size () + topo.e_core_cpus.size ())
    {
      (void)std::fprintf (stderr, "FAIL: detect_cpu_topology invalid\n");
      return false;
    }

  if (!milda::pin_calling_thread_to_pcores (
          topo, static_cast<int32_t> (topo.all_physical_cpus.size ())))
    {
      (void)std::fprintf (stderr, "FAIL: pin_calling_thread_to_pcores\n");
      return false;
    }

  const std::size_t rss = milda::read_linux_rss_bytes ();
  if (rss == 0U)
    {
      (void)std::fprintf (stderr, "FAIL: read_linux_rss_bytes == 0\n");
      return false;
    }

  milda::pinned_host_buffer pbuf;
  if (!pbuf.allocate (65536U) || pbuf.data () == nullptr
      || pbuf.size () < 65536U)
    {
      (void)std::fprintf (stderr, "FAIL: pinned_host_buffer::allocate\n");
      return false;
    }

  milda::kv_cache_type kt = milda::kv_cache_type::f16;
  if (!milda::parse_kv_cache_type ("q8_0", &kt)
      || kt != milda::kv_cache_type::q8_0
      || !milda::parse_kv_cache_type ("q4_0", &kt)
      || kt != milda::kv_cache_type::q4_0
      || !milda::parse_kv_cache_type ("f16", &kt)
      || kt != milda::kv_cache_type::f16)
    {
      (void)std::fprintf (stderr, "FAIL: parse_kv_cache_type\n");
      return false;
    }

  return true;
}

[[nodiscard]] bool
verify_gnu_style_file (const std::filesystem::path &filepath)
{
  std::ifstream in (filepath);
  if (!in.is_open ())
    {
      (void)std::fprintf (stderr, "FAIL: cannot open %s\n",
                          filepath.c_str ());
      return false;
    }

  bool ok = true;
  bool found_gpl = false;
  bool found_author = false;
  bool found_spdx = false;
  std::size_t line_no = 0U;
  std::string line;

  while (std::getline (in, line))
    {
      ++line_no;
      if (!line.empty () && line.back () == '\r')
        {
          line.pop_back ();
        }
      if (line_no <= 20U)
        {
          if (line.find ("GNU General Public License") != std::string::npos)
            {
              found_gpl = true;
            }
          if (line.find ("Kival") != std::string::npos)
            {
              found_author = true;
            }
          if (line.find ("SPDX-License-Identifier: GPL-3.0-or-later")
              != std::string::npos)
            {
              found_spdx = true;
            }
        }
      if (line.find ('\t') != std::string::npos)
        {
          (void)std::fprintf (stderr, "%s:%zu: tab character found\n",
                              filepath.c_str (), line_no);
          ok = false;
        }
      if (line.size () > 79U)
        {
          (void)std::fprintf (stderr,
                              "%s:%zu: line length %zu exceeds 79 columns\n",
                              filepath.c_str (), line_no, line.size ());
          ok = false;
        }
      if (!line.empty () && line.back () == ' ')
        {
          (void)std::fprintf (stderr, "%s:%zu: trailing whitespace\n",
                              filepath.c_str (), line_no);
          ok = false;
        }
    }

  if (!found_gpl || !found_author || !found_spdx)
    {
      (void)std::fprintf (stderr, "%s:1: missing GPLv3 header\n",
                          filepath.c_str ());
      ok = false;
    }
  return ok;
}

[[nodiscard]] bool
test_gnu_style (const std::filesystem::path &root_dir)
{
  const std::filesystem::path license_path = root_dir / "LICENSE";
  std::ifstream lic_in (license_path);
  if (!lic_in.is_open ())
    {
      (void)std::fprintf (stderr, "FAIL: missing root LICENSE file\n");
      return false;
    }
  std::string lic_text ((std::istreambuf_iterator<char> (lic_in)),
                        std::istreambuf_iterator<char> ());
  if (lic_text.find ("GNU GENERAL PUBLIC LICENSE") == std::string::npos
      || lic_text.find ("Version 3, 29 June 2007") == std::string::npos)
    {
      (void)std::fprintf (stderr, "FAIL: invalid root LICENSE contents\n");
      return false;
    }

  std::vector<std::string> subdirs;
  subdirs.push_back ("include");
  subdirs.push_back ("src");
  subdirs.push_back ("tests");
  bool all_ok = true;
  std::size_t count = 0U;

  for (const std::string &sub : subdirs)
    {
      const std::filesystem::path dir = root_dir / sub;
      if (!std::filesystem::exists (dir))
        {
          continue;
        }
      for (const auto &entry :
           std::filesystem::recursive_directory_iterator (dir))
        {
          if (!entry.is_regular_file ())
            {
              continue;
            }
          const std::string ext = entry.path ().extension ().string ();
          if (ext == ".hpp" || ext == ".cpp")
            {
              ++count;
              if (!verify_gnu_style_file (entry.path ()))
                {
                  all_ok = false;
                }
            }
        }
    }
  return all_ok && (count > 0U);
}

[[nodiscard]] bool
test_live_engine ()
{
  const char *env_home = std::getenv ("HOME");
  const std::string home_dir
      = (env_home != nullptr && env_home[0] != '\0')
            ? std::string (env_home)
            : std::string (".");
  const std::string model_path
      = home_dir + "/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
  std::error_code ec;
  if (!std::filesystem::exists (model_path, ec))
    {
      return true;
    }

  milda::engine_config cfg = milda::engine_config ();
  cfg.model_path = model_path;
  cfg.n_gpu_layers = 99;
  cfg.n_cpu_ffn = 2;
  cfg.n_ctx = 1024U;
  cfg.max_new_tokens = 24;
  cfg.temperature = 0.0F;
  cfg.cache_type_k = milda::kv_cache_type::q8_0;
  cfg.cache_type_v = milda::kv_cache_type::q8_0;
  cfg.flash_attn = true;
  cfg.stream_stdout = false;

  milda::inference_engine eng;
  if (!eng.initialize (cfg))
    {
      (void)std::fprintf (stderr, "FAIL: engine.initialize\n");
      return false;
    }

  const int32_t pinned_sys = eng.pinned_system_tokens ();
  if (pinned_sys <= 0 || eng.cached_kv_tokens () != pinned_sys)
    {
      (void)std::fprintf (stderr, "FAIL: pinned_system_tokens invalid\n");
      return false;
    }

  milda::turn_metrics m1 = milda::turn_metrics ();
  if (!eng.generate_turn ("Saluton! Kiu vi estas?", &m1)
      || m1.generated_tokens == 0U)
    {
      (void)std::fprintf (stderr, "FAIL: turn 1 generation\n");
      return false;
    }

  const int32_t after_turn1 = eng.cached_kv_tokens ();
  if (after_turn1 <= pinned_sys)
    {
      (void)std::fprintf (stderr, "FAIL: KV cache did not advance\n");
      return false;
    }

  milda::turn_metrics m2 = milda::turn_metrics ();
  if (!eng.generate_turn ("Kiel oni diras dankon?", &m2)
      || m2.generated_tokens == 0U)
    {
      (void)std::fprintf (stderr, "FAIL: turn 2 generation\n");
      return false;
    }

  if (!eng.shift_kv_cache_in_place (4)
      || eng.cached_kv_tokens () < pinned_sys)
    {
      (void)std::fprintf (stderr, "FAIL: shift_kv_cache_in_place\n");
      return false;
    }

  const int32_t saved_count = eng.cached_kv_tokens ();
  const char *tmp_sess = "/tmp/milda_test_session.bin";
  if (!eng.save_session (tmp_sess))
    {
      (void)std::fprintf (stderr, "FAIL: save_session (/konservi)\n");
      return false;
    }

  if (!eng.clear_conversation () || eng.cached_kv_tokens () != pinned_sys)
    {
      (void)std::fprintf (stderr, "FAIL: clear_conversation (/vakigi)\n");
      return false;
    }

  if (!eng.load_session (tmp_sess) || eng.cached_kv_tokens () != saved_count)
    {
      (void)std::fprintf (stderr, "FAIL: load_session (/sxargi)\n");
      (void)std::remove (tmp_sess);
      return false;
    }
  (void)std::remove (tmp_sess);

  /* Verify runtime language switch (/lang zh-Hans) re-pins KV cache and
     updates sampler parameters cleanly.  */
  if (!eng.set_language (milda::MildaLang::MandarinSimplified)
      || eng.configured_language () != milda::MildaLang::MandarinSimplified
      || eng.active_top_p () != 0.90F)
    {
      (void)std::fprintf (stderr, "FAIL: set_language (zh-Hans)\n");
      return false;
    }

  return true;
}

} /* namespace */

int
main (int argc, char **argv)
{
  std::filesystem::path root = std::filesystem::current_path ();
  if (argc > 1)
    {
      root = std::filesystem::path (argv[1]);
    }

  if (!test_esperanto_layer ())
    {
      return 1;
    }
  if (!test_polyglot_layer ())
    {
      return 1;
    }
  if (!test_hardware_layer ())
    {
      return 1;
    }
  if (!test_gnu_style (root))
    {
      return 1;
    }
  if (!test_live_engine ())
    {
      return 1;
    }

  (void)std::fprintf (
      stdout,
      "Milda testaro SUKCESIS (Poliglota 5-Lingva, Esperanto, HPC, KV, "
      "GNU).\n");
  return 0;
}
