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

#include <sphinx/sphinx.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/prctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

std::atomic<bool> g_stop_requested{ false };

void
sigint_handler (int /*signum*/) noexcept
{
  g_stop_requested.store (true, std::memory_order_release);
}

void
write_stdout (std::string_view text) noexcept
{
  (void)std::fwrite (text.data (), 1U, text.size (), stdout);
  (void)std::fflush (stdout);
}

void
write_stderr (std::string_view text) noexcept
{
  (void)std::fwrite (text.data (), 1U, text.size (), stderr);
  (void)std::fflush (stderr);
}

[[nodiscard]] auto
is_clear_terminal_input (std::string_view line) noexcept -> bool
{
  while (!line.empty ()
         && std::isspace (static_cast<unsigned char> (line.front ())) != 0)
    {
      line.remove_prefix (1U);
    }
  while (!line.empty ()
         && (std::isspace (static_cast<unsigned char> (line.back ())) != 0
             || line.back () == '.' || line.back () == '!'))
    {
      line.remove_suffix (1U);
    }
  std::string lower;
  lower.reserve (line.size ());
  for (const char c : line)
    {
      lower.push_back (
          static_cast<char> (std::tolower (static_cast<unsigned char> (c))));
    }
  if (lower == "clear" || lower == "/clear" || lower == "cls"
      || lower == "clear screen" || lower == "clear terminal"
      || lower == "clear the screen" || lower == "clear the terminal")
    {
      return true;
    }
  if (lower.size () <= 48U
      && lower.find ("workspace") == std::string::npos
      && lower.find ("memory") == std::string::npos
      && lower.find ("file") == std::string::npos
      && lower.find ("directory") == std::string::npos
      && (lower.find ("clear the screen") != std::string::npos
          || lower.find ("clear the terminal") != std::string::npos
          || lower.find ("clear my screen") != std::string::npos
          || lower.find ("clear my terminal") != std::string::npos))
    {
      return true;
    }
  return false;
}

void
drain_capture_buffer (sphinx::SphinxSession &session) noexcept
{
  std::array<float, 1024U> discard{};
  while (session.audio_io ().pop_capture (discard) > 0U)
    {
    }
}

[[nodiscard]] auto
compute_frame_rms (std::span<const float> samples) noexcept -> float
{
  if (samples.empty ())
    {
      return 0.0F;
    }
  double sum_sq = 0.0;
  for (const float s : samples)
    {
      sum_sq += static_cast<double> (s) * static_cast<double> (s);
    }
  return static_cast<float> (
      std::sqrt (sum_sq / static_cast<double> (samples.size ())));
}

void
normalize_utterance_gain (std::vector<float> &samples) noexcept
{
  float peak = 0.0F;
  for (const float s : samples)
    {
      peak = std::max (peak, std::fabs (s));
    }
  if (peak > 0.005F && peak < 0.45F)
    {
      const float gain = std::min (5.0F, 0.75F / peak);
      for (float &s : samples)
        {
          s = std::clamp (s * gain, -1.0F, 1.0F);
        }
    }
}

void
wait_for_playback_drain (sphinx::SphinxSession &session,
                         bool headless) noexcept
{
  if (headless)
    {
      return;
    }
  const auto start = std::chrono::steady_clock::now ();
  while (!g_stop_requested.load (std::memory_order_acquire))
    {
      const auto st = session.audio_io ().stats ();
      if (st.playback_occupancy == 0U)
        {
          // Allow the final 40 ms ALSA hardware buffer to finish playing.
          std::this_thread::sleep_for (std::chrono::milliseconds (60));
          break;
        }
      if (std::chrono::steady_clock::now () - start
          > std::chrono::seconds (45))
        {
          break;
        }
      std::this_thread::sleep_for (std::chrono::milliseconds (20));
    }
  // Discard any speaker echo captured by the microphone during TTS playback.
  drain_capture_buffer (session);
}

void
print_turn_summary (const sphinx::TurnTelemetry &t)
{
  std::ostringstream oss;
  oss << "[Turn " << t.turn_index
      << "] route=" << sphinx::route_to_string (t.route)
      << " router_ms=" << t.router_ms << " ttfs_ms=" << t.ttfs_ms
      << " tok/s=" << t.tok_per_sec << "\n  User: " << t.user_text
      << "\n  Sphinx: " << t.assistant_text << "\n";
  write_stdout (oss.str ());
}

void
run_live_voice_loop (sphinx::SphinxSession &session, bool headless)
{
  // Flush any audio accumulated in the capture ring buffer during the ~16 s
  // model load phase, then calibrate ambient noise floor over ~300 ms.
  drain_capture_buffer (session);

  constexpr std::size_t kFrameSize = 320U; // 20 ms at 16 kHz
  std::array<float, kFrameSize> frame{};
  std::size_t frame_fill = 0U;

  auto read_exact_frame = [&] () -> bool {
    while (frame_fill < kFrameSize)
      {
        const std::span<float> rem (frame.data () + frame_fill,
                                    kFrameSize - frame_fill);
        const std::size_t got = session.audio_io ().pop_capture (rem);
        if (got == 0U)
          {
            return false;
          }
        frame_fill += got;
      }
    frame_fill = 0U;
    return true;
  };

  float ambient_rms = 0.02F;
  if (!headless)
    {
      double sum_rms = 0.0;
      std::size_t cal_frames = 0U;
      const auto cal_start = std::chrono::steady_clock::now ();
      while (cal_frames < 15U
             && (std::chrono::steady_clock::now () - cal_start)
                    < std::chrono::milliseconds (450))
        {
          if (read_exact_frame ())
            {
              sum_rms += static_cast<double> (compute_frame_rms (frame));
              ++cal_frames;
            }
          else
            {
              std::this_thread::sleep_for (std::chrono::milliseconds (10));
            }
        }
      if (cal_frames > 0U)
        {
          ambient_rms = static_cast<float> (
              sum_rms / static_cast<double> (cal_frames));
        }
    }

  const float init_thresh
      = std::clamp (ambient_rms * 1.85F + 0.012F, 0.022F, 0.25F);
  {
    char cbuf[256];
    (void)std::snprintf (
        cbuf, sizeof (cbuf),
        "Sphinx Live Voice + Text Mode active (session_id=%s, mic_rms=%.3f, "
        "vad_threshold=%.3f).\nSpeak into your microphone or type a message "
        "('quit' to exit).\nYou> ",
        std::string (session.session_id ()).c_str (),
        static_cast<double> (ambient_rms), static_cast<double> (init_thresh));
    write_stdout (cbuf);
  }

  std::vector<float> pre_roll;
  pre_roll.reserve (kFrameSize * 12U);
  std::vector<float> utterance_16k;
  utterance_16k.reserve (16000U * 12U);

  bool capturing_speech = false;
  uint32_t consecutive_voice_frames = 0U;
  uint32_t active_voice_frames = 0U;
  uint32_t trailing_silence_ms = 0U;
  const bool stdin_is_tty = (::isatty (STDIN_FILENO) != 0);

  while (!g_stop_requested.load (std::memory_order_acquire))
    {
      // Also poll stdin so the user can type messages or 'quit' at any time.
      if (stdin_is_tty && !capturing_speech)
        {
          struct pollfd pfd{};
          pfd.fd = STDIN_FILENO;
          pfd.events = POLLIN;
          if (::poll (&pfd, 1U, 0) > 0 && (pfd.revents & POLLIN) != 0)
            {
              std::string line;
              if (!std::getline (std::cin, line))
                {
                  break;
                }
              if (line == "quit" || line == "exit" || line == "/quit"
                  || line == "/exit")
                {
                  break;
                }
              if (is_clear_terminal_input (line))
                {
                  write_stdout ("\033[2J\033[HYou> ");
                  continue;
                }
              if (!line.empty ())
                {
                  auto turn_res = session.process_text_turn (line);
                  if (turn_res)
                    {
                      print_turn_summary (*turn_res);
                      wait_for_playback_drain (session, headless);
                    }
                  else
                    {
                      write_stderr (
                          "Turn error: "
                          + std::string (turn_res.error ().message ()) + "\n");
                    }
                }
              write_stdout ("You> ");
              continue;
            }
        }

      if (!read_exact_frame ())
        {
          std::this_thread::sleep_for (std::chrono::milliseconds (10));
          continue;
        }

      const float f_rms = compute_frame_rms (frame);
      const float start_thresh
          = std::clamp (ambient_rms * 1.85F + 0.012F, 0.022F, 0.25F);
      const float keep_thresh
          = std::clamp (ambient_rms * 1.35F + 0.008F, 0.016F, 0.20F);

      if (!capturing_speech)
        {
          // Keep a 200 ms (10-frame) pre-roll buffer so initial consonants are
          // never clipped when speech starts.
          if (pre_roll.size () >= kFrameSize * 10U)
            {
              pre_roll.erase (pre_roll.begin (),
                              pre_roll.begin ()
                                  + static_cast<std::ptrdiff_t> (kFrameSize));
            }
          pre_roll.insert (pre_roll.end (), frame.begin (), frame.end ());

          if (f_rms >= start_thresh)
            {
              ++consecutive_voice_frames;
              if (consecutive_voice_frames >= 2U)
                {
                  session.audio_io ().clear_playback ();
                  utterance_16k = pre_roll;
                  capturing_speech = true;
                  active_voice_frames = consecutive_voice_frames;
                  trailing_silence_ms = 0U;
                  write_stdout ("\n  [Listening...]\n");
                }
            }
          else
            {
              consecutive_voice_frames = 0U;
              ambient_rms = 0.985F * ambient_rms + 0.015F * f_rms;
            }
        }
      else
        {
          utterance_16k.insert (utterance_16k.end (), frame.begin (),
                                frame.end ());
          if (f_rms >= keep_thresh)
            {
              ++active_voice_frames;
              trailing_silence_ms = 0U;
            }
          else
            {
              trailing_silence_ms += 20U;
            }

          const bool hit_silence_end = (trailing_silence_ms >= 600U);
          const bool hit_max_duration
              = (utterance_16k.size () >= 16000U * 10U);
          if (hit_silence_end || hit_max_duration)
            {
              capturing_speech = false;
              consecutive_voice_frames = 0U;
              pre_roll.clear ();

              if (active_voice_frames >= 6U && utterance_16k.size () >= 3200U)
                {
                  write_stdout ("  [Transcribing...]\n");
                  normalize_utterance_gain (utterance_16k);
                  auto turn_res = session.process_voice_turn (utterance_16k);
                  if (turn_res)
                    {
                      print_turn_summary (*turn_res);
                      wait_for_playback_drain (session, headless);
                    }
                }
              utterance_16k.clear ();
              write_stdout ("You> ");
            }
        }
    }
  write_stdout ("\n");
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  (void)::prctl (PR_SET_NAME, "sphinx", 0UL, 0UL, 0UL);
  if (::geteuid () == 0U)
    {
      write_stderr (
          "[sphinx] WARNING: Running as root (UID 0) is discouraged for "
          "security. Please run sphinx as a standard non-root user.\n");
    }
  (void)std::signal (SIGINT, sigint_handler);

  sphinx::SphinxSessionConfig cfg;
  std::string single_prompt;
  bool demo_mode = false;
  bool voice_mode = false;
  bool text_mode_explicit = false;
  bool clear_memory_requested = false;
  bool session_mode_requested = false;

  for (int i = 1; i < argc; ++i)
    {
      const std::string_view arg (argv[i]);
      if (arg == "--help" || arg == "-h")
        {
          write_stdout (
              "Usage: sphinx [options]\n"
              "  --text                  Interactive text chat (default)\n"
              "  --voice                 Live mic VAD + STT + Bryce TTS\n"
              "  --prompt <text>         Execute a single turn and exit\n"
              "  --demo                  Run a 3-turn CHAT + TASK + SSH demo\n"
              "  --big-model <id|path>   Big TASK model ID or GGUF path\n"
              "  --big-gpu-layers <N>    Layers to offload to GPU VRAM\n"
              "  --big-cpu-moe <N>       MoE expert layers in CPU RAM\n"
              "  --threads, -t <N>       Generation CPU threads (P-cores)\n"
              "  --threads-batch, -tb <N> Prefill batch CPU threads\n"
              "  --lazy-big              Defer big model load until TASK "
              "(default)\n"
              "  --eager-big             Load big model eagerly at startup\n"
              "  --preload-all           Populate CPU-offloaded weights into "
              "RAM upfront\n"
              "  --preload-cpu           Alias for --preload-all\n"
              "  --unload-big-after-task Free big model VRAM/RAM after TASK\n"
              "  --low-mem               Low-memory profile (4K ctx, 7B/8B "
              "default)\n"
              "  --clear-memory          Erase persistent memory on disk\n"
              "  --session-id <id>       Custom Yantra session run ID\n"
              "  --headless              Disable hardware ALSA I/O\n"
              "  --version               Print version information\n");
          return 0;
        }
      if (arg == "--version")
        {
          write_stdout ("sphinx (Sphinx Capstone Voice Assistant) 1.0.0\n");
          return 0;
        }
      if (arg == "--headless")
        {
          cfg.headless_audio = true;
        }
      else if (arg == "--lazy-big")
        {
          cfg.lazy_load_big_model = true;
        }
      else if (arg == "--eager-big" || arg == "--eager-load-big")
        {
          cfg.lazy_load_big_model = false;
        }
      else if (arg == "--preload-all" || arg == "--preload-cpu")
        {
          cfg.preload_cpu_weights = true;
          cfg.lazy_load_big_model = false;
        }
      else if (arg == "--unload-big-after-task")
        {
          cfg.unload_big_after_task = true;
        }
      else if (arg == "--low-mem")
        {
          cfg.low_memory_mode = true;
        }
      else if (arg == "--clear-memory" || arg == "--reset-memory")
        {
          clear_memory_requested = true;
        }
      else if (arg == "--demo")
        {
          demo_mode = true;
          session_mode_requested = true;
        }
      else if (arg == "--voice")
        {
          voice_mode = true;
          text_mode_explicit = false;
          session_mode_requested = true;
        }
      else if (arg == "--text")
        {
          voice_mode = false;
          text_mode_explicit = true;
          session_mode_requested = true;
        }
      else if (arg == "--big-model" && i + 1 < argc)
        {
          cfg.big_model_id = argv[++i];
        }
      else if (arg == "--big-gpu-layers" && i + 1 < argc)
        {
          cfg.big_model_n_gpu_layers
              = static_cast<int32_t> (std::strtol (argv[++i], nullptr, 10));
        }
      else if (arg == "--big-cpu-moe" && i + 1 < argc)
        {
          cfg.big_model_n_cpu_moe
              = static_cast<int32_t> (std::strtol (argv[++i], nullptr, 10));
        }
      else if ((arg == "--threads" || arg == "-t") && i + 1 < argc)
        {
          cfg.big_model_n_threads
              = static_cast<int32_t> (std::strtol (argv[++i], nullptr, 10));
        }
      else if ((arg == "--threads-batch" || arg == "-tb") && i + 1 < argc)
        {
          cfg.big_model_n_threads_batch
              = static_cast<int32_t> (std::strtol (argv[++i], nullptr, 10));
        }
      else if (arg == "--whisper-threads" && i + 1 < argc)
        {
          ++i;
        }
      else if (arg == "--session-id" && i + 1 < argc)
        {
          cfg.session_id = argv[++i];
        }
      else if (arg == "--prompt" && i + 1 < argc)
        {
          single_prompt = argv[++i];
          session_mode_requested = true;
        }
    }

  if (!voice_mode
      && (text_mode_explicit || cfg.headless_audio || !single_prompt.empty ()
          || demo_mode || ::isatty (STDIN_FILENO) == 0))
    {
      cfg.headless_audio = true;
      cfg.skip_tts_synthesis = true;
    }

  if (clear_memory_requested)
    {
      (void)sphinx::MemoryStore::wipe_directory (cfg.memory_dir);
      (void)sphinx::MemoryStore::wipe_directory (
          sphinx::default_session_home_path ("sphinx-workspace/.memory"));
      (void)sphinx::MemoryStore::wipe_directory (
          sphinx::default_session_home_path (".local/share/sphinx/memory"));
      write_stdout ("[sphinx] Persistent memory erased.\n");
      if (!session_mode_requested)
        {
          return 0;
        }
    }

  auto reg_ok = sphinx::ensure_models_registered (cfg.registry_path);
  if (!reg_ok)
    {
      write_stderr ("Error registering models: "
                    + std::string (reg_ok.error ().message ()) + "\n");
      return 1;
    }

  write_stdout ("[sphinx] Loading router model...\n");
  if (!cfg.lazy_load_big_model)
    {
      write_stdout ("[sphinx] Loading big model...\n");
    }
  auto sess_res = sphinx::SphinxSession::create (cfg);
  if (!sess_res)
    {
      write_stderr ("Error initializing SphinxSession: "
                    + std::string (sess_res.error ().message ()) + "\n");
      return 1;
    }
  sphinx::SphinxSession &session = **sess_res;
  {
    char buf[256];
    if (session.is_big_model_loaded ())
      {
        (void)std::snprintf (buf, sizeof (buf),
                             "[sphinx] Router loaded in %.2f s\n"
                             "[sphinx] Big model loaded in %.2f s. VRAM peak: "
                             "%.2f GB.\n"
                             "[sphinx] Ready.\n",
                             session.router_load_seconds (),
                             session.big_model_load_seconds (),
                             session.vram_peak_gb ());
      }
    else
      {
        (void)std::snprintf (buf, sizeof (buf),
                             "[sphinx] Router loaded in %.2f s (big model "
                             "deferred until first TASK). VRAM: %.2f GB.\n"
                             "[sphinx] Ready.\n",
                             session.router_load_seconds (),
                             session.vram_peak_gb ());
      }
    write_stdout (buf);
  }

  if (!single_prompt.empty ())
    {
      auto turn_res = session.process_text_turn (single_prompt);
      if (!turn_res)
        {
          write_stderr ("Turn error: "
                        + std::string (turn_res.error ().message ()) + "\n");
          return 1;
        }
      print_turn_summary (*turn_res);
      wait_for_playback_drain (session, cfg.headless_audio);
    }
  else if (demo_mode)
    {
      const std::vector<std::string_view> demo_prompts = {
        "Hello Sphinx, how are you doing today?",
        "Write a C++ function in fib.cpp that computes Fibonacci numbers and "
        "check if it compiles.",
        "Connect to localhost via SSH and check the kernel version with "
        "uname."
      };
      for (const std::string_view p : demo_prompts)
        {
          auto turn_res = session.process_text_turn (p);
          if (!turn_res)
            {
              write_stderr ("Turn error: "
                            + std::string (turn_res.error ().message ())
                            + "\n");
              return 1;
            }
          print_turn_summary (*turn_res);
          wait_for_playback_drain (session, cfg.headless_audio);
        }
    }
  else if (voice_mode
           || (!cfg.headless_audio && ::isatty (STDIN_FILENO) != 0))
    {
      run_live_voice_loop (session, cfg.headless_audio);
    }
  else
    {
      write_stdout ("Sphinx Interactive Text Chat ready (session_id="
                    + std::string (session.session_id ())
                    + "). Type a message or 'quit' to exit.\nYou> ");
      std::string line;
      while (!g_stop_requested.load (std::memory_order_acquire)
             && std::getline (std::cin, line))
        {
          if (line == "quit" || line == "exit" || line == "/quit"
              || line == "/exit")
            {
              break;
            }
          if (is_clear_terminal_input (line))
            {
              write_stdout ("\033[2J\033[HYou> ");
              continue;
            }
          if (line.empty ())
            {
              write_stdout ("You> ");
              continue;
            }
          auto turn_res = session.process_text_turn (line);
          if (!turn_res)
            {
              write_stderr ("Turn error: "
                            + std::string (turn_res.error ().message ())
                            + "\n");
              write_stdout ("You> ");
              continue;
            }
          print_turn_summary (*turn_res);
          wait_for_playback_drain (session, cfg.headless_audio);
          write_stdout ("You> ");
        }
      write_stdout ("\n");
    }

  auto plots_res = session.finalize_session ();
  if (!plots_res)
    {
      write_stderr ("Error rendering Darshana session plots: "
                    + std::string (plots_res.error ().message ()) + "\n");
      return 1;
    }

  std::ostringstream oss;
  oss << "Session finalized. Darshana plots exported:\n"
      << "  " << plots_res->main_svg_path << " (" << plots_res->main_render_ms
      << " ms)\n"
      << "  " << plots_res->latency_svg_path << " ("
      << plots_res->latency_render_ms << " ms)\n"
      << "  " << plots_res->router_svg_path << " ("
      << plots_res->router_render_ms << " ms)\n";
  write_stdout (oss.str ());
  return 0;
}
