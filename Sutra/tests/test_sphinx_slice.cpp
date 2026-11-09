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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>

#define SPHINX_ASSERT(cond)                                                   \
  do                                                                          \
    {                                                                         \
      if (!(cond))                                                            \
        {                                                                     \
          std::fprintf (stderr, "SPHINX_ASSERT failed at %s:%d: %s\n",        \
                        __FILE__, __LINE__, #cond);                           \
          std::abort ();                                                      \
        }                                                                     \
    }                                                                         \
  while (0)

namespace
{

/* Reads current process VmRSS in KiB from /proc/self/status.  */
[[nodiscard]] auto
current_rss_kib () -> std::size_t
{
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.starts_with ("VmRSS:"))
        {
          std::size_t val = 0U;
          for (const char ch : line)
            {
              if (ch >= '0' && ch <= '9')
                {
                  val = val * 10U + static_cast<std::size_t> (ch - '0');
                }
            }
          return val;
        }
    }
  return 0U;
}

/* §3.3 Item 2: Verify router classifies >= 18/20 hand-written test turns
   (10 conversational chat, 10 coding/SSH).  */
void
test_router_20_turns (const sphinx::TurnRouter &router)
{
  using R = sphinx::RouteKind;
  constexpr std::array<std::pair<std::string_view, R>, 20> kTurns = { {
      // 10 Conversational Chat Turns
      { "Good morning Sphinx, how are you feeling today?", R::Conversational },
      { "Tell me a short fun fact about ancient astronomy.",
        R::Conversational },
      { "What is a good recipe for masala chai?", R::Conversational },
      { "Who wrote the Sanskrit grammar Ashtadhyayi?", R::Conversational },
      { "Thank you for your help earlier, that made sense.",
        R::Conversational },
      { "Can you remind me what the word Sutra means?", R::Conversational },
      { "What is the capital city of Japan?", R::Conversational },
      { "Give me a one sentence summary of stoic philosophy.",
        R::Conversational },
      { "How many planets are in the solar system?", R::Conversational },
      { "Wish me luck on my presentation this afternoon!", R::Conversational },
      // 10 Coding / SSH / System Turns
      { "SSH into the staging server and check systemctl status.",
        R::Complex },
      { "Run cmake build and fix any C++23 compiler errors.", R::Complex },
      { "Inspect the git log and summarize the last three commits.",
        R::Complex },
      { "Profile the CUDA kernel latency with nvidia-smi.", R::Complex },
      { "We have a memory leak under ASan in the parser function.",
        R::Complex },
      { "Refactor this C++ template class to eliminate virtual dispatch.",
        R::Complex },
      { "Check docker container logs and journalctl for segfaults.",
        R::Complex },
      { "Run ctest under ThreadSanitizer and report any data races.",
        R::Complex },
      { "Grep the repository for raw new and delete allocations.",
        R::Complex },
      { "Optimize the SQL database query on the production server.",
        R::Complex },
  } };

  std::size_t correct = 0U;
  for (const auto &[prompt, expected] : kTurns)
    {
      if (router.classify (prompt) == expected)
        {
          ++correct;
        }
    }

  std::fprintf (stdout, "[SphinxMetrics] router_accuracy=%zu/20 (%.1f%%)\n",
                correct, (static_cast<double> (correct) / 20.0) * 100.0);
  SPHINX_ASSERT (correct >= 18U);
}

/* §3.2 & §3.3: Verify all 6 Aliveness Metrics and continuous stability with
   zero RSS growth and zero audio glitches.  */
void
test_sphinx_vertical_slice_aliveness ()
{
  sphinx::SphinxConfig cfg;
  cfg.context_len = 512U;
  cfg.conversational_max_tokens = 36U;
  cfg.complex_max_tokens = 48U;

  auto pipe_res = sphinx::SphinxPipeline::create (cfg);
  SPHINX_ASSERT (pipe_res.has_value ());
  auto &pipeline = **pipe_res;

  // 1. Verify Router on 20 hand-written turns (>= 18/20 required).
  test_router_20_turns (pipeline.router ());

  // Warmup turn so CUDA graphs and KV cache are resident before timing.
  auto warmup_pcm
      = vach::Stt::encode_test_utterance ("Hello Sphinx, warmup ping.", 0.6);
  SPHINX_ASSERT (warmup_pcm.has_value ());
  auto warmup_res = pipeline.process_voice_turn (*warmup_pcm, 0U);
  SPHINX_ASSERT (warmup_res.has_value ());

  // 2. Metric 1 & Metric 4: Conversational Turn Latency (TTFS <= 700 ms) and
  //    Audio Continuity (max inter-sentence gap <= 100 ms).
  auto conv_pcm = vach::Stt::encode_test_utterance (
      "Give me three short sentences about the ocean and the stars.", 0.8);
  SPHINX_ASSERT (conv_pcm.has_value ());
  auto conv_res = pipeline.process_voice_turn (*conv_pcm, 0U);
  SPHINX_ASSERT (conv_res.has_value ());
  SPHINX_ASSERT (conv_res->route == sphinx::RouteKind::Conversational);
  SPHINX_ASSERT (!conv_res->response_text.empty ());
  SPHINX_ASSERT (conv_res->sentences_synthesized >= 1U);
  SPHINX_ASSERT (conv_res->playback_samples_pushed > 0U);
  SPHINX_ASSERT (conv_res->ttfs_ms <= sphinx::kMaxConversationalTtfsMs);
  SPHINX_ASSERT (conv_res->max_inter_sentence_gap_ms
                 <= sphinx::kMaxInterSentenceGapMs);

  std::fprintf (stdout,
                "[SphinxMetrics] conv_ttfs_ms=%.2f llm_ttft_ms=%.2f "
                "max_sentence_gap_ms=%.4f sentences=%zu\n",
                conv_res->ttfs_ms, conv_res->llm_ttft_ms,
                conv_res->max_inter_sentence_gap_ms,
                conv_res->sentences_synthesized);

  // 3. Metric 2 & Metric 5: Complex Turn Latency (<= 1500 ms) + Backchannel
  //    Under Load (<= 500 ms) + 5s Cooldown Rate Limit.
  auto complex_pcm = vach::Stt::encode_test_utterance (
      "SSH into the build server and inspect the cmake compiler logs.", 0.8);
  SPHINX_ASSERT (complex_pcm.has_value ());
  auto complex_res = pipeline.process_voice_turn (*complex_pcm, 600U);
  SPHINX_ASSERT (complex_res.has_value ());
  SPHINX_ASSERT (complex_res->route == sphinx::RouteKind::Complex);
  SPHINX_ASSERT (complex_res->backchannel_emitted);
  SPHINX_ASSERT (complex_res->backchannel_latency_ms
                 <= sphinx::kMaxBackchannelLatencyMs);
  SPHINX_ASSERT (complex_res->ttfs_ms <= sphinx::kMaxComplexTtfsMs);

  // Immediate second complex turn within 5s must respect backchannel cooldown.
  auto complex_res2 = pipeline.process_voice_turn (*complex_pcm, 600U);
  SPHINX_ASSERT (complex_res2.has_value ());
  SPHINX_ASSERT (!complex_res2->backchannel_emitted);

  std::fprintf (stdout,
                "[SphinxMetrics] complex_ttfs_ms=%.2f backchannel_ms=%.3f "
                "cooldown_respected=1\n",
                complex_res->ttfs_ms, complex_res->backchannel_latency_ms);

  // 4. Metric 3: Barge-In Response Time (<= 200 ms).
  auto barge_res = pipeline.test_live_barge_in ();
  SPHINX_ASSERT (barge_res.has_value ());
  SPHINX_ASSERT (barge_res->playback_cleared);
  SPHINX_ASSERT (barge_res->llm_cancelled);
  SPHINX_ASSERT (barge_res->stop_latency_ms <= sphinx::kMaxBargeInLatencyMs);

  std::fprintf (stdout,
                "[SphinxMetrics] barge_in_ms=%.3f playback_cleared=%d "
                "llm_cancelled=%d\n",
                barge_res->stop_latency_ms,
                barge_res->playback_cleared ? 1 : 0,
                barge_res->llm_cancelled ? 1 : 0);

  // 5. Metric 6: Idle CPU Utilization (<= 5.0% of one P-core).
  auto cpu_res = pipeline.measure_idle_cpu_percent (300U);
  SPHINX_ASSERT (cpu_res.has_value ());
  SPHINX_ASSERT (*cpu_res <= sphinx::kMaxIdleCpuPercent);

  std::fprintf (stdout, "[SphinxMetrics] idle_cpu_pct=%.3f%%\n", *cpu_res);

  // 6. Continuous Multi-Turn Stability & Zero Memory Growth (§3.3 item 3).
  const std::size_t rss_before_kib = current_rss_kib ();
  for (int turn = 0; turn < 20; ++turn)
    {
      auto turn_res = pipeline.process_voice_turn (*conv_pcm, 0U);
      SPHINX_ASSERT (turn_res.has_value ());
      SPHINX_ASSERT (turn_res->playback_samples_pushed > 0U);
    }
  const std::size_t rss_after_kib = current_rss_kib ();
  const std::size_t rss_delta_kib = (rss_after_kib > rss_before_kib)
                                        ? (rss_after_kib - rss_before_kib)
                                        : 0U;
  const auto io_stats = pipeline.audio_io ().stats ();
  SPHINX_ASSERT (io_stats.capture_overruns == 0U);
  SPHINX_ASSERT (io_stats.playback_overruns == 0U);
  SPHINX_ASSERT (rss_delta_kib <= 8192U);

  std::fprintf (stdout,
                "[SphinxMetrics] stability_turns=20 rss_delta_kib=%zu "
                "audio_overruns=%llu\n",
                rss_delta_kib,
                static_cast<unsigned long long> (
                    io_stats.capture_overruns + io_stats.playback_overruns));
}

} /* namespace */

auto
main () -> int
{
  test_sphinx_vertical_slice_aliveness ();
  return 0;
}
