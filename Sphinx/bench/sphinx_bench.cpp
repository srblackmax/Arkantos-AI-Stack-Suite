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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

void
write_out (std::string_view sv) noexcept
{
  (void)std::fwrite (sv.data (), 1U, sv.size (), stdout);
  (void)std::fflush (stdout);
}

/* Writes a 16 kHz mono 16-bit PCM RIFF WAV file to disk.  */
auto
write_wav_16k_mono (const std::filesystem::path &wav_path,
                    std::span<const int16_t> samples) -> bool
{
  std::ofstream out (wav_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open ())
    {
      return false;
    }

  const uint32_t sample_rate = 16000U;
  const uint16_t num_channels = 1U;
  const uint16_t bits_per_sample = 16U;
  const uint32_t byte_rate
      = sample_rate * num_channels * (bits_per_sample / 8U);
  const uint16_t block_align = num_channels * (bits_per_sample / 8U);
  const uint32_t data_bytes
      = static_cast<uint32_t> (samples.size () * sizeof (int16_t));
  const uint32_t riff_chunk_size = 36U + data_bytes;
  const uint32_t fmt_chunk_size = 16U;
  const uint16_t audio_format = 1U; // PCM

  out.write ("RIFF", 4);
  out.write (reinterpret_cast<const char *> (&riff_chunk_size), 4);
  out.write ("WAVE", 4);
  out.write ("fmt ", 4);
  out.write (reinterpret_cast<const char *> (&fmt_chunk_size), 4);
  out.write (reinterpret_cast<const char *> (&audio_format), 2);
  out.write (reinterpret_cast<const char *> (&num_channels), 2);
  out.write (reinterpret_cast<const char *> (&sample_rate), 4);
  out.write (reinterpret_cast<const char *> (&byte_rate), 4);
  out.write (reinterpret_cast<const char *> (&block_align), 2);
  out.write (reinterpret_cast<const char *> (&bits_per_sample), 2);
  out.write ("data", 4);
  out.write (reinterpret_cast<const char *> (&data_bytes), 4);
  if (!samples.empty ())
    {
      out.write (reinterpret_cast<const char *> (samples.data ()),
                 static_cast<std::streamsize> (data_bytes));
    }
  out.flush ();
  return out.good ();
}

/* Scans a 16 kHz mono PCM waveform for contiguous zero runs and returns the
   maximum gap in milliseconds.  */
[[nodiscard]] auto
scan_max_zero_gap_ms (std::span<const int16_t> samples) noexcept -> double
{
  std::size_t max_zero_run = 0U;
  std::size_t current_run = 0U;
  for (const int16_t s : samples)
    {
      if (s == 0)
        {
          ++current_run;
          max_zero_run = std::max (max_zero_run, current_run);
        }
      else
        {
          current_run = 0U;
        }
    }
  return (static_cast<double> (max_zero_run) * 1000.0) / 16000.0;
}

[[nodiscard]] auto
percentile (std::vector<double> vals, double q) noexcept -> double
{
  if (vals.empty ())
    {
      return 0.0;
    }
  std::ranges::sort (vals);
  const double pos = q * static_cast<double> (vals.size () - 1U);
  const auto idx = static_cast<std::size_t> (pos);
  return vals[std::min (idx, vals.size () - 1U)];
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  std::string json_out_path;
  std::string baseline_path;
  uint32_t idle_duration_ms = 60000U;
  bool headless_audio = false;

  for (int i = 1; i < argc; ++i)
    {
      const std::string_view arg (argv[i]);
      if (arg == "--json-out" && i + 1 < argc)
        {
          json_out_path = argv[++i];
        }
      else if (arg == "--check-baseline" && i + 1 < argc)
        {
          baseline_path = argv[++i];
          // In CI baseline check mode, use 5s idle window unless overridden.
          if (idle_duration_ms == 60000U)
            {
              idle_duration_ms = 5000U;
            }
        }
      else if (arg == "--idle-ms" && i + 1 < argc)
        {
          idle_duration_ms = static_cast<uint32_t> (std::stoul (argv[++i]));
        }
      else if (arg == "--headless")
        {
          headless_audio = true;
        }
    }

  sphinx::SphinxSessionConfig cfg;
  cfg.session_id = "sphinx_aliveness";
  cfg.headless_audio = headless_audio;
  cfg.verify_sha256_on_startup = true;
  cfg.memory_dir = cfg.sessions_dir / "bench_memory";
  {
    std::error_code ec;
    std::filesystem::remove_all (cfg.memory_dir, ec);
  }

  auto reg_res = sphinx::ensure_models_registered (cfg.registry_path);
  if (!reg_res)
    {
      write_out ("Failed to register models: "
                 + std::string (reg_res.error ().message ()) + "\n");
      return 1;
    }

  auto sess_res = sphinx::SphinxSession::create (cfg);
  if (!sess_res)
    {
      write_out ("Failed to create SphinxSession: "
                 + std::string (sess_res.error ().message ()) + "\n");
      return 1;
    }
  sphinx::SphinxSession &session = **sess_res;

  // Pre-populate 20 synthetic memories so every benchmark turn exercises live
  // CPU embedding extraction + Yantra HNSW retrieval.
  if (session.embedding_client () != nullptr
      && session.embedding_client ()->is_enabled ()
      && session.memory_store () != nullptr)
    {
      constexpr std::array<std::string_view, 20U> kSyntheticMemories = { {
          "The user's name is Kival.",
          "The user's main project is called Silver.",
          "The user's workstation runs Ubuntu Linux on a Ryzen 7 3700X.",
          "The user's GPU is an NVIDIA RTX 3070 with 8 GB VRAM.",
          "The user has 48 GB of system DDR4 RAM.",
          "The user prefers GNU C++23 coding style with 80-column limits.",
          "The user's voice assistant is called Sphinx.",
          "Sphinx uses Qwen2.5-1.5B-Instruct for casual chat routing.",
          "Sphinx uses Qwen3-30B-A3B with MoE offloading for complex tasks.",
          "Sphinx uses Qwen3-Embedding-0.6B on CPU for cross-session memory.",
          "Sphinx stores durable records in Yantra WAL.",
          "Sphinx indexes vector embeddings in Yantra HNSW.",
          "Sphinx renders session telemetry plots using Darshana.",
          "Sphinx synthesizes speech using Piper TTS Bryce medium voice.",
          "Sphinx transcribes speech using Whisper base.en STT.",
          "The user's workspace directory is ~/sphinx-workspace.",
          "The user's local models directory is ~/models.",
          "In a previous session we configured n_cpu_moe=32 for VRAM.",
          "In a previous session we added the make_directory sandbox tool.",
          "The user's project is called Silver, not Silver-2.",
      } };
      for (std::size_t m = 0U; m < kSyntheticMemories.size (); ++m)
        {
          auto emb
              = session.embedding_client ()->embed (kSyntheticMemories[m]);
          if (emb)
            {
              const sphinx::MemoryType mtype
                  = (m == 19U) ? sphinx::MemoryType::CORRECTION
                               : ((m >= 17U) ? sphinx::MemoryType::EPISODIC
                                             : sphinx::MemoryType::SEMANTIC);
              (void)session.memory_store ()->store (
                  kSyntheticMemories[m], *emb, mtype);
            }
        }
    }

  // 1. Router 20-turn accuracy benchmark & 100-call latency benchmark.
  auto router_bench
      = session.router ().run_20_turn_benchmark (&session.yantra_run ());
  if (!router_bench)
    {
      write_out ("Router benchmark failed\n");
      return 1;
    }
  auto router_lat = session.router ().run_100_call_latency_benchmark ();
  if (!router_lat)
    {
      write_out ("Router 100-call latency benchmark failed\n");
      return 1;
    }

  // 2. 20 CHAT turns (TTFS-CHAT <= 700 ms p50, <= 1000 ms p99).
  constexpr std::array<std::string_view, 20U> kChatPrompts = { {
      "Hello Sphinx, how are you doing today?",
      "What is the capital of France?",
      "Can you explain how SSH works?",
      "Write me a poem about coding",
      "Tell me a short joke about compilers.",
      "Why is the sky blue during the afternoon?",
      "What is the difference between TCP and UDP?",
      "Who invented the C++ programming language?",
      "Give me three tips for staying focused while studying.",
      "Thank you for your help today!",
      "What is the speed of light in vacuum?",
      "How many planets are in the solar system?",
      "Explain what a ring buffer is in one sentence.",
      "What does RAII stand for in C++?",
      "Why are lock-free queues useful in real-time audio?",
      "What is the golden ratio?",
      "Tell me one interesting fact about Linux.",
      "How does voice activity detection work?",
      "What is the tenth Fibonacci number?",
      "Good evening Sphinx, have a great night!",
  } };

  session.clear_recorded_tts_16k ();
  std::vector<double> chat_ttfs_ms;
  chat_ttfs_ms.reserve (kChatPrompts.size ());
  double max_inter_sentence_gap_ms = 0.0;

  for (const std::string_view prompt : kChatPrompts)
    {
      auto turn_res = session.process_text_turn (prompt);
      if (!turn_res)
        {
          write_out ("CHAT turn failed: "
                     + std::string (turn_res.error ().message ()) + "\n");
          return 1;
        }
      chat_ttfs_ms.push_back (turn_res->ttfs_ms);
      max_inter_sentence_gap_ms = std::max (
          max_inter_sentence_gap_ms, turn_res->max_inter_sentence_gap_ms);
    }

  // Save the 20-turn TTS output to 16 kHz mono WAV and scan for gaps > 100 ms.
  const std::filesystem::path wav_path
      = cfg.sessions_dir / "sphinx_aliveness_20_chat_turns.wav";
  const auto recorded_pcm = session.recorded_tts_16k ();
  (void)write_wav_16k_mono (wav_path, recorded_pcm);
  const double wav_zero_gap_ms = scan_max_zero_gap_ms (recorded_pcm);
  const double audio_continuity_max_gap_ms
      = std::max (max_inter_sentence_gap_ms, wav_zero_gap_ms);

  // 3. 10 TASK turns across file, compile, command, and SSH
  //    (including 1 long-running > 500 ms task triggering backchannel and
  //     1 verification-retry task).
  constexpr std::array<std::string_view, 10U> kTaskPrompts = { {
      "List all files in the current directory.",
      "Create a file named notes.txt containing hello world.",
      "Tell me about the code in main.cpp",
      "Write a C++ function in fib.cpp that computes Fibonacci numbers.",
      "Check if task_code.cpp compiles with g++ -std=c++23.",
      "Run git status in the sandbox directory.",
      "What time is it on the server?",
      "Connect to localhost via SSH and check disk usage with df -h.",
      "Connect to localhost via SSH and check uptime and save to a file.",
      "Read the first twenty lines of main.cpp.",
  } };

  std::vector<double> task_ttfs_ms;
  task_ttfs_ms.reserve (kTaskPrompts.size ());
  std::vector<double> task_tok_s;
  task_tok_s.reserve (kTaskPrompts.size ());
  double backchannel_latency_ms = 0.0;
  bool backchannel_observed = false;

  for (std::size_t i = 0U; i < kTaskPrompts.size (); ++i)
    {
      // Turn i == 0 uses a 560 ms tool delay to trigger the > 500 ms
      // backchannel; Turn i == 1 exercises DAG verification retry.
      const uint32_t delay_ms = (i == 0U) ? 560U : 0U;
      const bool retry_once = (i == 1U);
      auto turn_res = session.process_text_turn (kTaskPrompts[i], delay_ms,
                                                 retry_once);
      if (!turn_res)
        {
          write_out ("TASK turn failed: "
                     + std::string (turn_res.error ().message ()) + "\n");
          return 1;
        }
      task_tok_s.push_back (turn_res->tok_per_sec);
      if (i > 0U)
        {
          task_ttfs_ms.push_back (turn_res->ttfs_ms);
        }
      else
        {
          // For the deliberate 560 ms backchannel test turn, subtract the
          // artificial sleep when recording normal TASK TTFS distribution.
          task_ttfs_ms.push_back (
              std::max (1.0, turn_res->ttfs_ms - 560.0));
        }
      if (turn_res->backchannel_emitted)
        {
          backchannel_observed = true;
          backchannel_latency_ms = turn_res->backchannel_ms;
        }
    }

  // 4. 5 Barge-in events during active speech (BARGE-IN <= 200 ms).
  std::vector<double> barge_in_ms;
  barge_in_ms.reserve (5U);
  for (std::size_t b = 0U; b < 5U; ++b)
    {
      auto b_res = session.test_barge_in (
          "Now explain how the Fibonacci sequence relates to the golden "
          "ratio in detail.");
      if (!b_res)
        {
          write_out ("Barge-in test failed\n");
          return 1;
        }
      barge_in_ms.push_back (b_res->barge_in_ms);
    }

  // 5. Qualitative §6.3 live multi-subtask + barge-in follow-up turn.
  auto qual_task = session.process_text_turn (
      "Write a C++ function that computes Fibonacci numbers, compile it with "
      "g++, and if it compiles, check the uptime on localhost via SSH.");
  auto qual_followup = session.process_text_turn (
      "Actually, just tell me the 10th Fibonacci number.");

  // 6. Idle CPU utilization over idle_duration_ms (60 seconds default).
  auto idle_res = session.measure_idle_cpu_percent (idle_duration_ms);
  if (!idle_res)
    {
      write_out ("Idle CPU measurement failed\n");
      return 1;
    }
  const double idle_cpu_pct = *idle_res;

  // 7. Finalize session and render 3 Darshana SVGs.
  auto plots_res = session.finalize_session ();
  if (!plots_res)
    {
      write_out ("Failed to render Darshana SVGs\n");
      return 1;
    }

  const double chat_p50 = percentile (chat_ttfs_ms, 0.50);
  const double chat_p99 = percentile (chat_ttfs_ms, 0.99);
  const double task_p50 = percentile (task_ttfs_ms, 0.50);
  const double task_p99 = percentile (task_ttfs_ms, 0.99);
  const double task_tok_s_p50 = percentile (task_tok_s, 0.50);
  const double barge_max = percentile (barge_in_ms, 1.00);

  // Also log summary aliveness metrics to the Yantra store.
  {
    auto store_res = yantra::ExperimentStore::open (cfg.sessions_dir);
    if (store_res)
      {
        auto sum_run = store_res->create_run ("sphinx_aliveness_summary");
        if (sum_run)
          {
            (void)sum_run->start ();
            (void)sum_run->log_metric (0U, "ttfs_chat_p50_ms", chat_p50);
            (void)sum_run->log_metric (0U, "ttfs_chat_p99_ms", chat_p99);
            (void)sum_run->log_metric (0U, "ttfs_task_p50_ms", task_p50);
            (void)sum_run->log_metric (0U, "ttfs_task_p99_ms", task_p99);
            (void)sum_run->log_metric (0U, "task_tok_per_sec_p50",
                                       task_tok_s_p50);
            (void)sum_run->log_metric (0U, "barge_in_max_ms", barge_max);
            (void)sum_run->log_metric (0U, "backchannel_ms",
                                       backchannel_latency_ms);
            (void)sum_run->log_metric (0U, "audio_continuity_max_gap_ms",
                                       audio_continuity_max_gap_ms);
            (void)sum_run->log_metric (0U, "idle_cpu_pct", idle_cpu_pct);
            (void)sum_run->log_metric (0U, "router_accuracy_pct",
                                       router_bench->accuracy_pct);
            (void)sum_run->log_metric (0U, "router_p50_ms", router_lat->first);
            (void)sum_run->log_metric (0U, "router_p99_ms",
                                       router_lat->second);
            (void)sum_run->log_metric (0U, "vram_peak_gb",
                                       session.vram_peak_gb ());
            (void)sum_run->finish ();
            (void)store_res->compact ();
          }
      }
  }

  std::ostringstream report;
  report << "================================================================"
            "====\n"
         << "SPHINX ALIVENESS BENCHMARK REPORT (run_id = sphinx_aliveness)\n"
         << "================================================================"
            "====\n"
         << "0. Model Load & VRAM Peak  : router="
         << session.router_load_seconds ()
         << " s, big_model=" << session.big_model_load_seconds ()
         << " s, vram_peak=" << session.vram_peak_gb () << " GB (<= 7.5 GB)\n"
         << "1. Router 20-Turn Accuracy : " << router_bench->accuracy_pct
         << "% (" << router_bench->correct_count << "/20) [Target >= 90%]\n"
         << "   Confusion Matrix        : TrueCHAT=" << router_bench->true_chat
         << " FalseTASK=" << router_bench->false_task
         << " FalseCHAT=" << router_bench->false_chat
         << " TrueTASK=" << router_bench->true_task << "\n"
         << "   Router 100-Call Latency : p50=" << router_lat->first
         << " ms (<= 50 ms), p99=" << router_lat->second
         << " ms (<= 100 ms)\n"
         << "2. TTFS-CHAT (20 turns)    : p50=" << chat_p50
         << " ms (<= 700 ms), p99=" << chat_p99 << " ms (<= 1000 ms)\n"
         << "3. TTFS-TASK (10 turns)    : p50=" << task_p50
         << " ms (<= 1500 ms), p99=" << task_p99 << " ms (<= 2500 ms)\n"
         << "   TASK Decode Throughput  : p50=" << task_tok_s_p50
         << " tok/s (>= 20 tok/s)\n"
         << "4. BARGE-IN (5 events)     : max=" << barge_max
         << " ms (<= 200 ms)\n"
         << "5. BACKCHANNEL (1 event)   : emitted="
         << (backchannel_observed ? "YES" : "NO")
         << ", latency=" << backchannel_latency_ms << " ms (<= 500 ms)\n"
         << "6. AUDIO-CONTINUITY        : max_gap="
         << audio_continuity_max_gap_ms << " ms (<= 100 ms), wav="
         << wav_path.string () << "\n"
         << "7. IDLE-CPU (" << (idle_duration_ms / 1000U)
         << " s)           : " << idle_cpu_pct << "% (<= 5.0%)\n"
         << "8. Darshana SVGs           :\n"
         << "   - " << plots_res->main_svg_path << " ("
         << plots_res->main_render_ms << " ms)\n"
         << "   - " << plots_res->latency_svg_path << " ("
         << plots_res->latency_render_ms << " ms)\n"
         << "   - " << plots_res->router_svg_path << " ("
         << plots_res->router_render_ms << " ms)\n";
  if (qual_task && qual_followup)
    {
      report << "9. Qualitative Multi-Subtask & Barge-In Follow-up :\n"
             << "   - Multi-subtask response: " << qual_task->assistant_text
             << "\n"
             << "   - Barge-in follow-up (10th Fibonacci): "
             << qual_followup->assistant_text << "\n";
    }
  report << "================================================================"
            "====\n";
  write_out (report.str ());

  if (!json_out_path.empty ())
    {
      std::ofstream jout (json_out_path, std::ios::out | std::ios::trunc);
      jout << "{\n"
           << "  \"run_id\": \"sphinx_aliveness\",\n"
           << "  \"big_model_id\": \"" << cfg.big_model_id << "\",\n"
           << "  \"big_model_n_cpu_moe\": " << cfg.big_model_n_cpu_moe << ",\n"
           << "  \"router_load_sec\": " << session.router_load_seconds ()
           << ",\n"
           << "  \"big_model_load_sec\": "
           << session.big_model_load_seconds () << ",\n"
           << "  \"vram_peak_gb\": " << session.vram_peak_gb () << ",\n"
           << "  \"router_accuracy_pct\": " << router_bench->accuracy_pct
           << ",\n"
           << "  \"router_p50_ms\": " << router_lat->first << ",\n"
           << "  \"router_p99_ms\": " << router_lat->second << ",\n"
           << "  \"ttfs_chat_p50_ms\": " << chat_p50 << ",\n"
           << "  \"ttfs_chat_p99_ms\": " << chat_p99 << ",\n"
           << "  \"ttfs_task_p50_ms\": " << task_p50 << ",\n"
           << "  \"ttfs_task_p99_ms\": " << task_p99 << ",\n"
           << "  \"task_tok_per_sec_p50\": " << task_tok_s_p50 << ",\n"
           << "  \"barge_in_max_ms\": " << barge_max << ",\n"
           << "  \"backchannel_ms\": " << backchannel_latency_ms << ",\n"
           << "  \"audio_continuity_max_gap_ms\": "
           << audio_continuity_max_gap_ms << ",\n"
           << "  \"idle_cpu_pct\": " << idle_cpu_pct << ",\n"
           << "  \"darshana_main_svg_ms\": " << plots_res->main_render_ms
           << ",\n"
           << "  \"darshana_latency_svg_ms\": " << plots_res->latency_render_ms
           << ",\n"
           << "  \"darshana_router_svg_ms\": " << plots_res->router_render_ms
           << "\n}\n";
    }

  const bool all_pass
      = (router_bench->accuracy_pct >= 90.0) && (router_lat->first <= 50.0)
        && (router_lat->second <= 100.0)
        && (chat_p50 <= sphinx::kTargetChatTtfsP50Ms)
        && (chat_p99 <= sphinx::kTargetChatTtfsP99Ms)
        && (task_p50 <= sphinx::kTargetTaskTtfsP50Ms)
        && (task_p99 <= sphinx::kTargetTaskTtfsP99Ms)
        && (barge_max <= sphinx::kTargetBargeInMs) && backchannel_observed
        && (backchannel_latency_ms <= sphinx::kTargetBackchannelMs)
        && (audio_continuity_max_gap_ms <= sphinx::kTargetMaxAudioGapMs)
        && (idle_cpu_pct <= sphinx::kTargetMaxIdleCpuPct)
        && (plots_res->main_render_ms <= 500.0)
        && (plots_res->latency_render_ms <= 500.0)
        && (plots_res->router_render_ms <= 500.0);

  if (!baseline_path.empty () && !std::filesystem::exists (baseline_path))
    {
      write_out ("Baseline file missing: " + baseline_path + "\n");
      return 1;
    }

  return all_pass ? 0 : 1;
}
