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

#ifndef VACH_TTS_HPP
#define VACH_TTS_HPP

/**
 * @file tts.hpp
 * @brief Chunked, interruptible Text-to-Speech (Piper ONNX + espeak-ng)
 *        behind an opaque handle for Vach.
 * @note Compiles with only the C++23 standard library and sutra/error.hpp.
 *       Exposes zero onnxruntime_*, espeak_*, or piper_* symbols.
 */

#include <sutra/error.hpp>
#include <vach/audio_io.hpp>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vach::inline v1
{

inline constexpr uint32_t kDefaultTtsSampleRateHz = 22050U;
inline constexpr uint32_t kDefaultTtsPlaybackRateHz = 48000U;
inline constexpr std::size_t kDefaultTtsChunkSamples = 960U;

[[nodiscard]] inline auto
default_tts_share_path (const char *rel) -> std::string
{
  const char *h = std::getenv ("HOME");
  const std::string base
      = (h != nullptr && h[0] != '\0') ? std::string (h) : std::string (".");
  return base + "/.local/share/piper/" + rel;
}

/**
 * @brief Callback invoked as each synthesized 16-bit PCM chunk is produced.
 */
using PcmChunkFn = std::function<void (std::span<const int16_t>)>;

/**
 * @brief Summary statistics for a completed or interrupted TTS synthesis call.
 */
struct TtsSynthesisReport
{
  std::size_t total_samples = 0U;
  std::size_t chunks_emitted = 0U;
  double max_inter_chunk_gap_ms = 0.0;
  double synthesis_latency_ms = 0.0;
  bool interrupted = false;
};

/**
 * @brief Configuration for vach::Tts.
 */
struct TtsConfig
{
  std::string onnx_model_path
      = default_tts_share_path ("en_US-lessac-medium.onnx");
  std::string onnx_config_path
      = default_tts_share_path ("en_US-lessac-medium.onnx.json");
  uint32_t sample_rate_hz = kDefaultTtsSampleRateHz;
  uint32_t playback_rate_hz = kDefaultTtsPlaybackRateHz;
  std::size_t chunk_samples = kDefaultTtsChunkSamples;
  bool prefer_espeak_dlopen = true;
};

/**
 * @brief Sentence-level streaming Text-to-Speech behind an opaque handle.
 * @pre sentence is non-empty.
 * @post Phonemizes via espeak-ng ("en-us"), synthesizes 22050 Hz mono PCM
 *       via Piper ONNX Runtime, resamples to 48000 Hz stereo S16_LE, and
 *       pushes into the playback ring buffer or chunk callback.
 */
class Tts
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  Tts (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates a Tts instance behind an opaque handle.
   * @param config TTS configuration (default en_US-lessac-medium.onnx).
   * @returns sutra::Result<std::unique_ptr<Tts>> or AgentError::Tts.
   */
  [[nodiscard]] static auto
  create (const TtsConfig &config
          = TtsConfig{}) noexcept -> sutra::Result<std::unique_ptr<Tts>>;

  Tts (const Tts &) = delete;
  auto operator= (const Tts &) -> Tts & = delete;
  Tts (Tts &&) noexcept;
  auto operator= (Tts &&) noexcept -> Tts &;
  ~Tts () noexcept;

  /**
   * @brief Binds an AudioIo instance so synthesize_sentence(sentence) and
   *        cancel_playback() operate directly on the playback ring buffer.
   */
  void bind_audio_io (AudioIo *audio_io) noexcept;

  /**
   * @brief Synthesizes a single sentence into streamed 48 kHz stereo 16-bit
   *        PCM chunks delivered to on_pcm_chunk (and bound AudioIo if set).
   * @param sentence Single sentence to synthesize.
   * @param on_pcm_chunk Callback invoked for each synthesized PCM chunk.
   * @param volume_scale Linear amplitude multiplier (e.g. 0.6f for
   *        backchannel).
   * @returns sutra::Result<TtsSynthesisReport> or AgentError::Tts /
   *          AgentError::Cancelled.
   */
  [[nodiscard]] auto
  synthesize_sentence (std::string_view sentence,
                       const PcmChunkFn &on_pcm_chunk,
                       float volume_scale = 1.0F) noexcept
      -> sutra::Result<TtsSynthesisReport>;

  /**
   * @brief Synthesizes a single sentence, resamples from 22050 Hz mono to
   *        48000 Hz stereo, and pushes all chunks into audio_io's playback
   *        ring buffer.
   */
  [[nodiscard]] auto
  synthesize_sentence (std::string_view sentence, AudioIo &audio_io,
                       float volume_scale = 1.0F) noexcept
      -> sutra::Result<TtsSynthesisReport>;

  /**
   * @brief Synthesizes a single sentence and pushes it to the bound AudioIo
   *        ring buffer.
   */
  [[nodiscard]] auto
  synthesize_sentence (std::string_view sentence,
                       float volume_scale = 1.0F) noexcept
      -> sutra::Result<TtsSynthesisReport>;

  /**
   * @brief Synthesizes raw 22050 Hz mono float samples via Piper ONNX.
   */
  [[nodiscard]] auto
  synthesize_raw_22k (std::string_view sentence,
                      std::vector<float> &out_pcm_22k) noexcept
      -> sutra::Result<void>;

  /**
   * @brief Cooperatively interrupts any in-flight synthesis immediately and
   *        flushes the bound AudioIo playback ring buffer for barge-in.
   */
  void cancel_playback () noexcept;

  /**
   * @brief Cooperatively interrupts any in-flight synthesis immediately for
   *        barge-in.
   */
  void stop () noexcept;

  /**
   * @brief Returns the configured native TTS model sample rate in Hz.
   */
  [[nodiscard]] auto sample_rate_hz () const noexcept -> uint32_t;

  /**
   * @brief Returns the configured output playback sample rate in Hz (48 kHz).
   */
  [[nodiscard]] auto playback_rate_hz () const noexcept -> uint32_t;

private:
  std::unique_ptr<Impl> impl_;
};

} /* namespace vach::inline v1 */

#endif /* VACH_TTS_HPP */
