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

#ifndef VACH_VAD_HPP
#define VACH_VAD_HPP

/**
 * @file vad.hpp
 * @brief Voice Activity Detection (energy + zero-crossing rate / opaque ONNX)
 *        for Vach.
 * @note Compiles with only the C++23 standard library and sutra/error.hpp.
 */

#include <sutra/error.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace vach::inline v1
{

inline constexpr uint32_t kDefaultVadSampleRateHz = 16000U;
inline constexpr float kDefaultVadEnergyThreshold = 0.015F;
inline constexpr float kDefaultVadMaxZeroCrossingRate = 0.45F;
inline constexpr uint32_t kDefaultVadSpeechStartMs = 30U;
inline constexpr uint32_t kDefaultVadSilenceHangoverMs = 200U;

/**
 * @brief Discrete state transition events emitted by vach::Vad.
 */
enum class VadEvent : uint8_t
{
  None = 0,
  SpeechStart = 1,
  SpeechActive = 2,
  SpeechEnd = 3
};

/**
 * @brief Configuration parameters for vach::Vad.
 */
struct VadConfig
{
  uint32_t sample_rate_hz = kDefaultVadSampleRateHz;
  float energy_threshold = kDefaultVadEnergyThreshold;
  float max_zcr = kDefaultVadMaxZeroCrossingRate;
  uint32_t speech_start_ms = kDefaultVadSpeechStartMs;
  uint32_t silence_hangover_ms = kDefaultVadSilenceHangoverMs;
};

/**
 * @brief Frame-by-frame Voice Activity Detector behind an opaque handle.
 * @pre sample_rate_hz > 0 && silence_hangover_ms <= 300.
 * @post Detects SpeechEnd within <= 300 ms of trailing silence.
 */
class Vad
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  Vad (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates a Vad instance with the given configuration.
   * @param config VAD parameters.
   * @returns sutra::Result<std::unique_ptr<Vad>> or AgentError::AudioDevice.
   */
  [[nodiscard]] static auto
  create (const VadConfig &config
          = VadConfig{}) noexcept -> sutra::Result<std::unique_ptr<Vad>>;

  Vad (const Vad &) = delete;
  auto operator= (const Vad &) -> Vad & = delete;
  Vad (Vad &&) noexcept;
  auto operator= (Vad &&) noexcept -> Vad &;
  ~Vad () noexcept;

  /**
   * @brief Processes a 16 kHz mono PCM frame and returns any state transition.
   * @param pcm_16k Span of normalized [-1.0f, 1.0f] mono audio samples.
   * @returns VadEvent::SpeechStart, SpeechActive, SpeechEnd, or None.
   */
  [[nodiscard]] auto
  process_frame (std::span<const float> pcm_16k) noexcept -> VadEvent;

  /**
   * @brief Resets internal speech/silence accumulators.
   */
  void reset () noexcept;

  /**
   * @brief Returns true if speech is currently active.
   */
  [[nodiscard]] auto is_in_speech () const noexcept -> bool;

  /**
   * @brief Returns the trailing silence duration (in ms) that triggered the
   *        most recent SpeechEnd event.
   */
  [[nodiscard]] auto last_speech_end_latency_ms () const noexcept -> double;

private:
  std::unique_ptr<Impl> impl_;
};

} /* namespace vach::inline v1 */

#endif /* VACH_VAD_HPP */
