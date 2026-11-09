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

#ifndef SPHINX_SPHINX_HPP
#define SPHINX_SPHINX_HPP

/**
 * @file sphinx.hpp
 * @brief Sphinx vertical slice composition layer over Sutra and Vach:
 *        mic -> Vad -> Stt -> Router -> Sutra (LlamaCppClient) ->
 *        sentence-chunked Tts -> speaker.
 */

#include <sutra/sutra.hpp>
#include <vach/vach.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sphinx::inline v1
{

[[nodiscard]] inline auto
default_sphinx_slice_model_path () -> std::string
{
  const char *h = std::getenv ("HOME");
  const std::string base
      = (h != nullptr && h[0] != '\0') ? std::string (h) : std::string (".");
  return base + "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
}

inline constexpr double kMaxConversationalTtfsMs = 700.0;
inline constexpr double kMaxComplexTtfsMs = 1500.0;
inline constexpr double kMaxBargeInLatencyMs = 200.0;
inline constexpr double kMaxInterSentenceGapMs = 100.0;
inline constexpr double kMaxBackchannelLatencyMs = 500.0;
inline constexpr double kMaxIdleCpuPercent = 5.0;
inline constexpr uint64_t kBackchannelCooldownNs = 5000000000ULL;

/**
 * @brief Route classification produced by the Sphinx small-model router.
 */
enum class RouteKind : uint8_t
{
  Conversational = 0,
  Complex = 1
};

/**
 * @brief Telemetry and timing breakdown for a single voice turn through
 *        Sphinx.
 */
struct VoiceTurnMetrics
{
  std::string transcript;
  std::string response_text;
  RouteKind route = RouteKind::Conversational;
  bool backchannel_emitted = false;
  double vad_eos_ms = 0.0;
  double stt_latency_ms = 0.0;
  double llm_ttft_ms = 0.0;
  double ttfs_ms = 0.0;
  double backchannel_latency_ms = 0.0;
  double max_inter_sentence_gap_ms = 0.0;
  std::size_t sentences_synthesized = 0U;
  std::size_t playback_samples_pushed = 0U;
};

/**
 * @brief Telemetry for a barge-in interruption event during TTS playback.
 */
struct BargeInMetrics
{
  double stop_latency_ms = 0.0;
  bool playback_cleared = false;
  bool llm_cancelled = false;
};

/**
 * @brief Configuration for the Sphinx voice assistant pipeline.
 */
struct SphinxConfig
{
  std::string model_path = default_sphinx_slice_model_path ();
  uint32_t context_len = 1024U;
  uint32_t conversational_max_tokens = 48U;
  uint32_t complex_max_tokens = 64U;
  uint64_t backchannel_cooldown_ns = kBackchannelCooldownNs;
};

/**
 * @brief Fast turn router backed by sutra::LlmClient for classifying
 *        conversational vs complex (coding/SSH/tool) turns.
 */
class TurnRouter
{
public:
  explicit TurnRouter (sutra::LlmClient *llm_client = nullptr) noexcept;

  /**
   * @brief Classifies a user transcript into Conversational or Complex.
   * @param transcript User spoken utterance text.
   * @returns RouteKind::Conversational or RouteKind::Complex.
   */
  [[nodiscard]] auto
  classify (std::string_view transcript) const noexcept -> RouteKind;

private:
  sutra::LlmClient *llm_client_ = nullptr;
};

/**
 * @brief End-to-end Sphinx voice pipeline wiring Vach and Sutra.
 */
class SphinxPipeline
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  SphinxPipeline (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates and initializes the Sphinx vertical slice pipeline.
   * @param config Pipeline configuration.
   * @param llm_override Optional external LlmClient (if null, creates a
   *        resident LlamaCppClient from config.model_path).
   * @returns sutra::Result<std::unique_ptr<SphinxPipeline>>.
   */
  [[nodiscard]] static auto create (
      const SphinxConfig &config = SphinxConfig{},
      std::shared_ptr<sutra::LlmClient> llm_override
      = nullptr) noexcept -> sutra::Result<std::unique_ptr<SphinxPipeline>>;

  SphinxPipeline (const SphinxPipeline &) = delete;
  auto operator= (const SphinxPipeline &) -> SphinxPipeline & = delete;
  SphinxPipeline (SphinxPipeline &&) noexcept;
  auto operator= (SphinxPipeline &&) noexcept -> SphinxPipeline &;
  ~SphinxPipeline () noexcept;

  /**
   * @brief Executes a full voice turn from 16 kHz mic audio through VAD ->
   *        STT -> Router -> Sutra live LLM -> sentence-chunked TTS ->
   *        AudioIo speaker buffer.
   * @param utterance_16k Spoken 16 kHz mono audio frame sequence.
   * @param simulated_tool_delay_ms Optional tool execution latency (in ms) for
   *        complex turns to exercise backchannel emission under load.
   * @returns sutra::Result<VoiceTurnMetrics>.
   */
  [[nodiscard]] auto
  process_voice_turn (std::span<const float> utterance_16k,
                      uint32_t simulated_tool_delay_ms
                      = 0U) noexcept -> sutra::Result<VoiceTurnMetrics>;

  /**
   * @brief Exercises live barge-in: starts streaming LLM + TTS playback and
   *        injects a Vad::SpeechStart frame mid-playback, measuring time to
   *        playback silence + LLM cancellation (§3.2 metric 3).
   */
  [[nodiscard]] auto
  test_live_barge_in () noexcept -> sutra::Result<BargeInMetrics>;

  /**
   * @brief Measures single-core CPU utilization of the idle listening loop
   *        (AudioIo + Vad on silence at 10 ms real-time frame cadence) over
   *        duration_ms (§3.2 metric 6).
   */
  [[nodiscard]] auto
  measure_idle_cpu_percent (uint32_t duration_ms
                            = 500U) noexcept -> sutra::Result<double>;

  /**
   * @brief Accesses the TurnRouter instance.
   */
  [[nodiscard]] auto router () const noexcept -> const TurnRouter &;

  /**
   * @brief Accesses the underlying AudioIo ring buffer subsystem.
   */
  [[nodiscard]] auto audio_io () noexcept -> vach::AudioIo &;

private:
  std::unique_ptr<Impl> impl_;
};

} /* namespace sphinx::inline v1 */

#endif /* SPHINX_SPHINX_HPP */
