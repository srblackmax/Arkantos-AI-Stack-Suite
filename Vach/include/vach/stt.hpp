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

#ifndef VACH_STT_HPP
#define VACH_STT_HPP

/**
 * @file stt.hpp
 * @brief Speech-to-Text engine (whisper.cpp) behind an opaque handle for Vach.
 * @note Compiles with only the C++23 standard library and sutra/error.hpp.
 *       Exposes zero whisper_* symbols in public headers.
 */

#include <sutra/error.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vach::inline v1
{

inline constexpr uint32_t kDefaultSttSampleRateHz = 16000U;
inline constexpr uint32_t kDefaultSttThreads = 8U;

[[nodiscard]] inline auto
default_stt_model_path () -> std::string
{
  const char *h = std::getenv ("HOME");
  const std::string base
      = (h != nullptr && h[0] != '\0') ? std::string (h) : std::string (".");
  return base + "/.local/share/whisper/ggml-base.en.bin";
}

/**
 * @brief Completed utterance transcription output from vach::Stt.
 */
struct Transcript
{
  std::string text;
  float confidence = 1.0F;
  double latency_ms = 0.0;
};

/**
 * @brief Configuration for vach::Stt.
 */
struct SttConfig
{
  std::string model_path = default_stt_model_path ();
  std::string language = "en";
  uint32_t sample_rate_hz = kDefaultSttSampleRateHz;
  uint32_t n_threads = kDefaultSttThreads;
  bool no_context = false;
};

/**
 * @brief Speech-to-Text engine behind an opaque handle.
 * @pre Utterance audio is 16 kHz mono float or int16 PCM.
 * @post Returns Transcript within <= 500 ms for a 5-second utterance.
 */
class Stt
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  Stt (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates an Stt instance behind an opaque handle.
   * @param config STT configuration (default ggml-base.en.bin).
   * @returns sutra::Result<std::unique_ptr<Stt>> or AgentError::Stt.
   */
  [[nodiscard]] static auto
  create (const SttConfig &config
          = SttConfig{}) noexcept -> sutra::Result<std::unique_ptr<Stt>>;

  Stt (const Stt &) = delete;
  auto operator= (const Stt &) -> Stt & = delete;
  Stt (Stt &&) noexcept;
  auto operator= (Stt &&) noexcept -> Stt &;
  ~Stt () noexcept;

  /**
   * @brief Transcribes a completed 16 kHz mono float utterance into text.
   * @param utterance_16k Span of 16 kHz mono float samples.
   * @returns sutra::Result<Transcript> or AgentError::Stt.
   */
  [[nodiscard]] auto transcribe (std::span<const float> utterance_16k) noexcept
      -> sutra::Result<Transcript>;

  /**
   * @brief Transcribes a completed 16 kHz mono int16 utterance into text.
   * @param utterance_16k_s16 Span of 16 kHz mono S16_LE samples.
   * @returns sutra::Result<Transcript> or AgentError::Stt.
   */
  [[nodiscard]] auto
  transcribe (std::span<const int16_t> utterance_16k_s16) noexcept
      -> sutra::Result<Transcript>;

  /**
   * @brief Synthesizes a spoken 16 kHz mono waveform from transcript_text
   *        using Piper ONNX TTS and resamples it to 16 kHz for real ASR
   *        verification.
   * @param transcript_text Utterance text to synthesize into 16 kHz speech.
   * @param duration_seconds Target buffer duration in seconds (0 = natural).
   * @returns sutra::Result<std::vector<float>> of 16 kHz mono PCM samples.
   */
  [[nodiscard]] static auto
  encode_test_utterance (std::string_view transcript_text,
                         double duration_seconds
                         = 1.5) -> sutra::Result<std::vector<float>>;

private:
  std::unique_ptr<Impl> impl_;
};

} /* namespace vach::inline v1 */

#endif /* VACH_STT_HPP */
