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

#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>

#if defined (__linux__)
#include <unistd.h>
#endif

namespace
{

constexpr std::size_t k_input_buffer_bytes = 8192U;

void
write_stdout_str (std::string_view text) noexcept
{
  if (!text.empty ())
    {
      (void)std::fwrite (text.data (), 1U, text.size (), stdout);
      (void)std::fflush (stdout);
    }
}

void
write_stderr_str (std::string_view text) noexcept
{
  if (!text.empty ())
    {
      (void)std::fwrite (text.data (), 1U, text.size (), stderr);
      (void)std::fflush (stderr);
    }
}

[[nodiscard]] std::string
discover_default_model_path ()
{
  const char *env_model = std::getenv ("MILDA_MODEL");
  if (env_model != nullptr && env_model[0] != '\0')
    {
      std::error_code ec;
      if (std::filesystem::exists (env_model, ec))
        {
          return std::string (env_model);
        }
    }

  const char *env_home = std::getenv ("HOME");
  const std::string home_dir
      = (env_home != nullptr && env_home[0] != '\0')
            ? std::string (env_home)
            : std::string (".");
  const char *env_user = std::getenv ("USER");
  const std::string user_name
      = (env_user != nullptr && env_user[0] != '\0')
            ? std::string (env_user)
            : std::string ("user");

  const std::array<std::string, 6U> candidates =
    {
      home_dir + "/models/Qwen3-Coder-Next-Q4_K_S.gguf",
      "/media/" + user_name + "/SphinxAI/models/Qwen3-Coder-Next-Q4_K_S.gguf",
      home_dir + "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
      home_dir + "/models/gemma-4-31B-it-Q4_K_M.gguf",
      home_dir + "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
      home_dir + "/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    };

  std::error_code ec;
  for (const std::string &cand : candidates)
    {
      if (std::filesystem::exists (cand, ec))
        {
          return cand;
        }
    }
  return std::string ();
}

void
print_usage (const char *prog_name) noexcept
{
  std::array<char, 2048U> buf;
  const int n = std::snprintf (
      buf.data (), buf.size (),
      "Uzado: %s [opcioj]\n\n"
      "Milda — Poliglota C++23 konversacia AI (eo, en, es, zh-Hans, ru).\n\n"
      "Opcioj:\n"
      "  -m, --model <vojo>          Vojo al GGUF-modelo\n"
      "  -l, --lang <kodo>           Lingvo: auto, eo, en, es, zh, zh-Hans, "
      "ru\n"
      "  --x-system <auto|on|off>    Esperanta x-sistemo (defauxlt: auto)\n"
      "  -ngl, --n-gpu-layers <N>    GPU-tavoloj (-1 = auxtomata, "
      "0 = 100%% CPU)\n"
      "  -ncmoe, --n-cpu-moe <N>     MoE-tavoloj sur CPU RAM "
      "(-1 = auxtomata)\n"
      "  -ncffn, --n-cpu-ffn <N>     FFN-tavoloj sur CPU RAM "
      "(-1 = auxtomata)\n"
      "  -t, --threads <N>           CPU-fadenoj por generado (TG, P-kernoj)\n"
      "  -tb, --threads-batch <N>    CPU-fadenoj por prompt-analizo (PP)\n"
      "  --pcore-pin / --no-pcore-pin  Fiksi TG-fadenojn al P-kernoj "
      "(defauxlt: jes)\n"
      "  -c, --ctx-size <N>          Kunteksta grandeco (defauxlt: 16384)\n"
      "  -b, --batch-size <N>        Logika aro-grandeco (defauxlt: 2048)\n"
      "  -ub, --ubatch-size <N>      Fizika mikro-aro (defauxlt: 512)\n"
      "  --cache-type-k <f16|q8_0|q4_0>  K-kaxsmemora kvantigo "
      "(defauxlt: q8_0)\n"
      "  --cache-type-v <f16|q8_0|q4_0>  V-kaxsmemora kvantigo "
      "(defauxlt: q8_0)\n"
      "  -fa, --flash-attn           Sxalti Flash Attention (defauxlt: jes)\n"
      "  --no-flash-attn             Malsxalti Flash Attention\n"
      "  --mlock                     Sxlosi pezojn en fizika RAM (mlock)\n"
      "  --no-mmap                   Malsxalti mmap\n"
      "  -n, --n-predict <N>         Maksimumaj jxetonoj por respondo (1024)\n"
      "  --temp <F>                  Temperaturo (defauxlt: 0.75)\n"
      "  --seed <N>                  Hazarda semo (defauxlt: 42)\n"
      "  -p, --prompt <teksto>       Unufoja respondo kaj eliro\n"
      "  -s, --session <dosiero>     Konservi/sxargi KV-memoron trans sesioj\n"
      "  --stat                      Montri telemetrion post cxiu vico\n"
      "  -h, --help                  Montri cxi tiun helpon\n\n"
      "Interagaj komandoj:\n"
      "  /lang [kodo]         Montri aux sxangxi lingvon "
      "(auto|eo|en|es|zh|zh-Hans|ru)\n"
      "  /x-system [on|off]   Sxalti/malsxalti Esperantan x-sistemon\n"
      "  /eliri               Eliri el Milda\n"
      "  /vakigi              Vakigi la konversacian KV-kaxsmemoron\n"
      "  /konservi [dosiero]  Konservi la KV-memoron al disko\n"
      "  /sxargi [dosiero]    Sxargi la KV-memoron el disko\n"
      "  /stat                Sxalti/montri realtempan aparatan telemetrion\n",
      (prog_name != nullptr) ? prog_name : "milda");

  if (n > 0)
    {
      write_stdout_str (std::string_view (
          buf.data (), static_cast<std::size_t> (n)));
    }
}

[[nodiscard]] bool
parse_int32_arg (const char *str, int32_t *out_val) noexcept
{
  if (str == nullptr || str[0] == '\0' || out_val == nullptr)
    {
      return false;
    }
  char *end_ptr = nullptr;
  const long val = std::strtol (str, &end_ptr, 10);
  if (end_ptr == str || *end_ptr != '\0')
    {
      return false;
    }
  *out_val = static_cast<int32_t> (val);
  return true;
}

[[nodiscard]] bool
parse_uint32_arg (const char *str, uint32_t *out_val) noexcept
{
  if (str == nullptr || str[0] == '\0' || out_val == nullptr)
    {
      return false;
    }
  char *end_ptr = nullptr;
  const unsigned long val = std::strtoul (str, &end_ptr, 10);
  if (end_ptr == str || *end_ptr != '\0')
    {
      return false;
    }
  *out_val = static_cast<uint32_t> (val);
  return true;
}

[[nodiscard]] bool
parse_float_arg (const char *str, float *out_val) noexcept
{
  if (str == nullptr || str[0] == '\0' || out_val == nullptr)
    {
      return false;
    }
  char *end_ptr = nullptr;
  const float val = std::strtof (str, &end_ptr);
  if (end_ptr == str || *end_ptr != '\0')
    {
      return false;
    }
  *out_val = val;
  return true;
}

[[nodiscard]] bool
parse_x_system_arg (std::string_view val,
                    milda::x_system_mode *out_mode) noexcept
{
  if (out_mode == nullptr)
    {
      return false;
    }
  const std::string_view clean = milda::trim_ascii_whitespace (val);
  if (clean == "auto")
    {
      *out_mode = milda::x_system_mode::auto_lang;
      return true;
    }
  if (clean == "on" || clean == "true" || clean == "1" || clean == "jes")
    {
      *out_mode = milda::x_system_mode::forced_on;
      return true;
    }
  if (clean == "off" || clean == "false" || clean == "0" || clean == "ne")
    {
      *out_mode = milda::x_system_mode::forced_off;
      return true;
    }
  return false;
}

void
print_lang_status (const milda::inference_engine &engine,
                   milda::x_system_mode x_mode) noexcept
{
  const milda::MildaLang cfg_lang = engine.configured_language ();
  const milda::MildaLang act_lang = engine.active_language ();
  const char *x_str = "auto (nur Esperanto)";
  if (x_mode == milda::x_system_mode::forced_on)
    {
      x_str = "on (cxiam sxaltita)";
    }
  else if (x_mode == milda::x_system_mode::forced_off)
    {
      x_str = "off (malsxaltita)";
    }

  std::array<char, 512U> buf;
  const int n = std::snprintf (
      buf.data (), buf.size (),
      "[milda] Lingva re\304\235imo: %.*s (%.*s) | Aktiva prompt-lingvo: "
      "%.*s (%.*s)\n"
      "[milda] X-sistemo: %s | Subtenataj kodoj: auto, eo, en, es, zh, "
      "zh-Hans, ru\n",
      static_cast<int> (milda::lang_iso_code (cfg_lang).size ()),
      milda::lang_iso_code (cfg_lang).data (),
      static_cast<int> (milda::lang_native_name (cfg_lang).size ()),
      milda::lang_native_name (cfg_lang).data (),
      static_cast<int> (milda::lang_iso_code (act_lang).size ()),
      milda::lang_iso_code (act_lang).data (),
      static_cast<int> (milda::lang_native_name (act_lang).size ()),
      milda::lang_native_name (act_lang).data (),
      x_str);
  if (n > 0)
    {
      write_stdout_str (std::string_view (
          buf.data (), static_cast<std::size_t> (n)));
    }
}

} /* namespace */

int
main (int argc, char **argv)
{
#if defined (__linux__)
  (void)std::signal (SIGPIPE, SIG_IGN);
  if (std::getenv ("MILDA_OMP_BOOTSTRAPPED") == nullptr)
    {
      (void)setenv ("MILDA_OMP_BOOTSTRAPPED", "1", 1);
      (void)setenv ("OMP_WAIT_POLICY", "ACTIVE", 0);
      (void)setenv ("GOMP_SPINCOUNT", "30000000", 0);
      std::error_code exe_ec;
      const std::filesystem::path exe_p
          = std::filesystem::read_symlink ("/proc/self/exe", exe_ec);
      if (!exe_ec && !exe_p.empty ())
        {
          (void)execv (exe_p.c_str (), argv);
        }
    }
#endif

  /* Unbuffer stdout for immediate UTF-8 token piece streaming.  */
  (void)std::setvbuf (stdout, nullptr, _IONBF, 0);

  milda::engine_config cfg = milda::engine_config ();
  cfg.model_path = discover_default_model_path ();

  std::string single_prompt;
  bool has_single_prompt = false;
  bool show_telemetry = false;

  for (int i = 1; i < argc; ++i)
    {
      const std::string_view arg (argv[i]);
      if (arg == "-h" || arg == "--help")
        {
          print_usage (argv[0]);
          return 0;
        }
      if ((arg == "-m" || arg == "--model") && i + 1 < argc)
        {
          cfg.model_path = argv[++i];
          continue;
        }
      if ((arg == "-l" || arg == "--lang") && i + 1 < argc)
        {
          if (!milda::parse_milda_lang (argv[++i], &cfg.language))
            {
              write_stderr_str (
                  "Eraro: nevalida lingvo por --lang. Uzu: auto, eo, en, es, "
                  "zh, zh-Hans, ru.\n");
              return 1;
            }
          continue;
        }
      if (arg.starts_with ("--lang="))
        {
          if (!milda::parse_milda_lang (arg.substr (7U), &cfg.language))
            {
              write_stderr_str (
                  "Eraro: nevalida lingvo por --lang. Uzu: auto, eo, en, es, "
                  "zh, zh-Hans, ru.\n");
              return 1;
            }
          continue;
        }
      if (arg == "--x-system" && i + 1 < argc)
        {
          if (!parse_x_system_arg (argv[++i], &cfg.x_mode))
            {
              write_stderr_str (
                  "Eraro: nevalida valoro por --x-system (auto|on|off).\n");
              return 1;
            }
          continue;
        }
      if (arg.starts_with ("--x-system="))
        {
          if (!parse_x_system_arg (arg.substr (11U), &cfg.x_mode))
            {
              write_stderr_str (
                  "Eraro: nevalida valoro por --x-system (auto|on|off).\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-ngl" || arg == "--n-gpu-layers" || arg == "--gpu-layers")
          && i + 1 < argc)
        {
          const std::string_view val (argv[++i]);
          if (val == "auto")
            {
              cfg.n_gpu_layers = -1;
            }
          else if (val == "all")
            {
              cfg.n_gpu_layers = 999;
            }
          else if (!parse_int32_arg (val.data (), &cfg.n_gpu_layers))
            {
              write_stderr_str ("Eraro: nevalida valoro por -ngl.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-ncmoe" || arg == "--n-cpu-moe") && i + 1 < argc)
        {
          if (!parse_int32_arg (argv[++i], &cfg.n_cpu_moe))
            {
              write_stderr_str ("Eraro: nevalida valoro por --n-cpu-moe.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-ncffn" || arg == "--n-cpu-ffn") && i + 1 < argc)
        {
          if (!parse_int32_arg (argv[++i], &cfg.n_cpu_ffn))
            {
              write_stderr_str ("Eraro: nevalida valoro por --n-cpu-ffn.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-t" || arg == "--threads") && i + 1 < argc)
        {
          if (!parse_int32_arg (argv[++i], &cfg.n_threads)
              || cfg.n_threads <= 0)
            {
              write_stderr_str ("Eraro: nevalida valoro por --threads.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-tb" || arg == "--threads-batch") && i + 1 < argc)
        {
          if (!parse_int32_arg (argv[++i], &cfg.n_threads_batch)
              || cfg.n_threads_batch <= 0)
            {
              write_stderr_str (
                  "Eraro: nevalida valoro por --threads-batch.\n");
              return 1;
            }
          continue;
        }
      if (arg == "--pcore-pin")
        {
          cfg.pin_pcores = true;
          continue;
        }
      if (arg == "--no-pcore-pin")
        {
          cfg.pin_pcores = false;
          continue;
        }
      if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc)
        {
          if (!parse_uint32_arg (argv[++i], &cfg.n_ctx) || cfg.n_ctx < 256U)
            {
              write_stderr_str ("Eraro: nevalida valoro por --ctx-size.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc)
        {
          if (!parse_uint32_arg (argv[++i], &cfg.n_batch) || cfg.n_batch == 0U)
            {
              write_stderr_str ("Eraro: nevalida valoro por --batch-size.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-ub" || arg == "--ubatch-size") && i + 1 < argc)
        {
          if (!parse_uint32_arg (argv[++i], &cfg.n_ubatch)
              || cfg.n_ubatch == 0U)
            {
              write_stderr_str ("Eraro: nevalida valoro por --ubatch-size.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-ctk" || arg == "--cache-type-k") && i + 1 < argc)
        {
          if (!milda::parse_kv_cache_type (argv[++i], &cfg.cache_type_k))
            {
              write_stderr_str (
                  "Eraro: --cache-type-k subtenas nur f16, q8_0, q4_0.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-ctv" || arg == "--cache-type-v") && i + 1 < argc)
        {
          if (!milda::parse_kv_cache_type (argv[++i], &cfg.cache_type_v))
            {
              write_stderr_str (
                  "Eraro: --cache-type-v subtenas nur f16, q8_0, q4_0.\n");
              return 1;
            }
          continue;
        }
      if (arg == "-fa" || arg == "--flash-attn")
        {
          if (i + 1 < argc)
            {
              const std::string_view next_v (argv[i + 1]);
              if (next_v == "on" || next_v == "true" || next_v == "1")
                {
                  cfg.flash_attn = true;
                  ++i;
                  continue;
                }
              if (next_v == "off" || next_v == "false" || next_v == "0")
                {
                  cfg.flash_attn = false;
                  ++i;
                  continue;
                }
            }
          cfg.flash_attn = true;
          continue;
        }
      if (arg == "--no-flash-attn")
        {
          cfg.flash_attn = false;
          continue;
        }
      if (arg == "--mlock")
        {
          cfg.use_mlock = true;
          continue;
        }
      if (arg == "--no-mmap")
        {
          cfg.use_mmap = false;
          continue;
        }
      if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc)
        {
          if (!parse_int32_arg (argv[++i], &cfg.max_new_tokens)
              || cfg.max_new_tokens <= 0)
            {
              write_stderr_str ("Eraro: nevalida valoro por --n-predict.\n");
              return 1;
            }
          continue;
        }
      if (arg == "--temp" && i + 1 < argc)
        {
          if (!parse_float_arg (argv[++i], &cfg.temperature))
            {
              write_stderr_str ("Eraro: nevalida valoro por --temp.\n");
              return 1;
            }
          continue;
        }
      if (arg == "--top-k" && i + 1 < argc)
        {
          if (!parse_int32_arg (argv[++i], &cfg.top_k))
            {
              write_stderr_str ("Eraro: nevalida valoro por --top-k.\n");
              return 1;
            }
          continue;
        }
      if (arg == "--top-p" && i + 1 < argc)
        {
          if (!parse_float_arg (argv[++i], &cfg.top_p))
            {
              write_stderr_str ("Eraro: nevalida valoro por --top-p.\n");
              return 1;
            }
          cfg.custom_top_p = true;
          continue;
        }
      if (arg == "--min-p" && i + 1 < argc)
        {
          if (!parse_float_arg (argv[++i], &cfg.min_p))
            {
              write_stderr_str ("Eraro: nevalida valoro por --min-p.\n");
              return 1;
            }
          continue;
        }
      if (arg == "--repeat-penalty" && i + 1 < argc)
        {
          if (!parse_float_arg (argv[++i], &cfg.repeat_penalty))
            {
              write_stderr_str (
                  "Eraro: nevalida valoro por --repeat-penalty.\n");
              return 1;
            }
          cfg.custom_repeat_penalty = true;
          continue;
        }
      if (arg == "--seed" && i + 1 < argc)
        {
          if (!parse_uint32_arg (argv[++i], &cfg.seed))
            {
              write_stderr_str ("Eraro: nevalida valoro por --seed.\n");
              return 1;
            }
          continue;
        }
      if ((arg == "-p" || arg == "--prompt") && i + 1 < argc)
        {
          single_prompt = argv[++i];
          has_single_prompt = true;
          continue;
        }
      if ((arg == "-s" || arg == "--session") && i + 1 < argc)
        {
          cfg.session_path = argv[++i];
          continue;
        }
      if (arg == "--stat")
        {
          show_telemetry = true;
          continue;
        }

      write_stderr_str ("Eraro: nekonata opcio: ");
      write_stderr_str (arg);
      write_stderr_str ("\nUzu --help por vidi la opciojn.\n");
      return 1;
    }

  if (cfg.model_path.empty ())
    {
      write_stderr_str (
          "Eraro: neniu GGUF-modelo trovita. Uzu -m <vojo_al_modelo.gguf>.\n");
      return 1;
    }

  milda::inference_engine engine;
  if (!engine.initialize (cfg))
    {
      write_stderr_str ("Eraro: malsukcesis sxargi modelon: ");
      write_stderr_str (cfg.model_path);
      write_stderr_str ("\n");
      return 1;
    }

  if (!has_single_prompt && cfg.session_path.empty ())
    {
      const char *env_h = std::getenv ("HOME");
      cfg.session_path = (env_h != nullptr && env_h[0] != '\0')
                             ? (std::string (env_h) + "/milda_memoro.bin")
                             : std::string ("milda_memoro.bin");
    }

  std::error_code sess_ec;
  if (!cfg.session_path.empty ()
      && std::filesystem::exists (cfg.session_path, sess_ec))
    {
      if (engine.load_session (cfg.session_path.c_str ()))
        {
          write_stdout_str ("[milda] Konversacia memoro sxargita el: ");
          write_stdout_str (cfg.session_path);
          write_stdout_str ("\n");
        }
    }

  milda::turn_metrics last_metrics = milda::turn_metrics ();

  if (has_single_prompt)
    {
      const std::size_t new_len = milda::preprocess_user_input_in_place (
          single_prompt.data (), single_prompt.size (), cfg.language, nullptr,
          cfg.x_mode);
      const std::string_view prompt_view (single_prompt.data (), new_len);
      write_stdout_str ("Milda > ");
      if (!engine.generate_turn (prompt_view, &last_metrics))
        {
          write_stderr_str ("\nEraro dum generado.\n");
          return 1;
        }
      write_stdout_str ("\n");
      if (!cfg.session_path.empty ())
        {
          (void)engine.save_session (cfg.session_path.c_str ());
        }
      if (show_telemetry)
        {
          milda::print_telemetry (engine.collect_telemetry (last_metrics));
        }
      return 0;
    }

  /* Interactive polyglot REPL with pre-allocated stack/static buffer.  */
  std::array<char, k_input_buffer_bytes> input_buf;
  input_buf.fill ('\0');

  write_stdout_str (
      "Milda (Poliglota C++23 AI: eo, en, es, zh-Hans, ru) — Komandoj: "
      "/lang, /x-system, /eliri, /vakigi, /konservi, /sxargi, /stat\n");

  for (;;)
    {
      write_stdout_str ("Vi > ");
      if (std::fgets (input_buf.data (),
                      static_cast<int> (input_buf.size ()), stdin)
          == nullptr)
        {
          if (!cfg.session_path.empty ())
            {
              (void)engine.save_session (cfg.session_path.c_str ());
            }
          write_stdout_str ("\n\304\234is revido!\n");
          break;
        }

      const std::size_t raw_len = std::strlen (input_buf.data ());
      const std::string_view raw_view (input_buf.data (), raw_len);
      const std::string_view trimmed_raw
          = milda::trim_ascii_whitespace (raw_view);

      if (trimmed_raw.empty ())
        {
          continue;
        }

      const milda::repl_command_kind cmd
          = milda::parse_repl_command (trimmed_raw);
      if (cmd == milda::repl_command_kind::eliri)
        {
          const std::string target_path
              = !cfg.session_path.empty ()
                    ? cfg.session_path
                    : std::string ("milda_memoro.bin");
          if (engine.save_session (target_path.c_str ()))
            {
              write_stdout_str ("[milda] Memoro konservita al: ");
              write_stdout_str (target_path);
              write_stdout_str ("\n");
            }
          write_stdout_str ("\304\234is revido!\n");
          break;
        }
      if (cmd == milda::repl_command_kind::vakigi)
        {
          if (engine.clear_conversation ())
            {
              last_metrics = milda::turn_metrics ();
              write_stdout_str (
                  "[milda] Kunteksto vakigita (sistempromeso konservita).\n");
            }
          else
            {
              write_stderr_str ("[milda] Malsukcesis vakigi kuntekston.\n");
            }
          continue;
        }
      if (cmd == milda::repl_command_kind::lang)
        {
          const std::string_view arg_lang
              = milda::extract_repl_command_arg (trimmed_raw);
          if (arg_lang.empty ())
            {
              print_lang_status (engine, cfg.x_mode);
              continue;
            }
          milda::MildaLang target_lang = milda::MildaLang::Esperanto;
          if (!milda::parse_milda_lang (arg_lang, &target_lang))
            {
              write_stderr_str (
                  "[milda] Nekonata lingvo-kodo. Uzu: auto, eo, en, es, zh, "
                  "zh-Hans, ru.\n");
              continue;
            }
          if (engine.set_language (target_lang))
            {
              cfg.language = target_lang;
              last_metrics = milda::turn_metrics ();
              write_stdout_str (milda::lang_switch_confirmation (target_lang));
              write_stdout_str ("\n");
            }
          else
            {
              write_stderr_str ("[milda] Malsukcesis sxangxi lingvon.\n");
            }
          continue;
        }
      if (cmd == milda::repl_command_kind::x_system)
        {
          const std::string_view arg_x
              = milda::extract_repl_command_arg (trimmed_raw);
          if (arg_x.empty ())
            {
              print_lang_status (engine, cfg.x_mode);
              continue;
            }
          if (!parse_x_system_arg (arg_x, &cfg.x_mode))
            {
              write_stderr_str (
                  "[milda] Uzu: /x-system auto | /x-system on | "
                  "/x-system off\n");
              continue;
            }
          print_lang_status (engine, cfg.x_mode);
          continue;
        }
      if (cmd == milda::repl_command_kind::konservi)
        {
          const std::string_view arg_path
              = milda::extract_repl_command_arg (trimmed_raw);
          const std::string target_path
              = !arg_path.empty ()
                    ? std::string (arg_path)
                    : (!cfg.session_path.empty ()
                           ? cfg.session_path
                           : std::string ("milda_memoro.bin"));
          if (engine.save_session (target_path.c_str ()))
            {
              write_stdout_str ("[milda] Memoro konservita al: ");
              write_stdout_str (target_path);
              write_stdout_str ("\n");
            }
          else
            {
              write_stderr_str ("[milda] Malsukcesis konservi memoron.\n");
            }
          continue;
        }
      if (cmd == milda::repl_command_kind::sxargi)
        {
          const std::string_view arg_path
              = milda::extract_repl_command_arg (trimmed_raw);
          const std::string target_path
              = !arg_path.empty ()
                    ? std::string (arg_path)
                    : (!cfg.session_path.empty ()
                           ? cfg.session_path
                           : std::string ("milda_memoro.bin"));
          if (engine.load_session (target_path.c_str ()))
            {
              write_stdout_str ("[milda] Memoro sxargita el: ");
              write_stdout_str (target_path);
              write_stdout_str ("\n");
            }
          else
            {
              write_stderr_str ("[milda] Malsukcesis sxargi memoron.\n");
            }
          continue;
        }
      if (cmd == milda::repl_command_kind::stat)
        {
          show_telemetry = !show_telemetry;
          write_stdout_str (show_telemetry
                                ? "[milda] Telemetrio sxaltita.\n"
                                : "[milda] Telemetrio malsxaltita.\n");
          milda::print_telemetry (engine.collect_telemetry (last_metrics));
          continue;
        }
      if (cmd == milda::repl_command_kind::helpo)
        {
          write_stdout_str (
              "Komandoj: /lang [auto|eo|en|es|zh|zh-Hans|ru], "
              "/x-system [auto|on|off], /eliri, /vakigi, /konservi [dosiero], "
              "/sxargi [dosiero], /stat\n");
          continue;
        }

      /* Language-gated x-sistemo to UTF-8 conversion before tokenization.  */
      const std::size_t conv_len = milda::preprocess_user_input_in_place (
          input_buf.data (), raw_len, cfg.language, nullptr, cfg.x_mode);
      const std::string_view user_msg = milda::trim_ascii_whitespace (
          std::string_view (input_buf.data (), conv_len));

      if (user_msg.empty ())
        {
          continue;
        }

      write_stdout_str ("Milda > ");
      if (!engine.generate_turn (user_msg, &last_metrics))
        {
          write_stderr_str ("\n[milda] Eraro dum generado de respondo.\n");
          continue;
        }
      write_stdout_str ("\n");
      if (!cfg.session_path.empty ())
        {
          (void)engine.save_session (cfg.session_path.c_str ());
        }

      if (show_telemetry)
        {
          milda::print_telemetry (engine.collect_telemetry (last_metrics));
        }
    }

  return 0;
}
