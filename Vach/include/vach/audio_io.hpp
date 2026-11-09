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

#ifndef VACH_AUDIO_IO_HPP
#define VACH_AUDIO_IO_HPP

/**
 * @file audio_io.hpp
 * @brief Lock-free real-time audio capture & playback behind an opaque handle.
 * @note Compiles with only the C++23 standard library and sutra/error.hpp.
 *       The real-time audio path never allocates, locks a std::mutex, logs,
 *       or calls into Sutra.
 */

#include <sutra/error.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace vach::inline v1
{

inline constexpr uint32_t kDefaultCaptureRateHz = 16000U;
inline constexpr uint32_t kDefaultPlaybackRateHz = 48000U;
inline constexpr std::size_t kDefaultRingBufferSamples = 2097152U;

/**
 * @brief Configuration for vach::AudioIo.
 */
struct AudioIoConfig
{
  std::string capture_device = "default";
  std::string playback_device = "default";
  uint32_t capture_rate_hz = kDefaultCaptureRateHz;
  uint32_t playback_rate_hz = kDefaultPlaybackRateHz;
  std::size_t ring_capacity_samples = kDefaultRingBufferSamples;
  bool headless_loopback = false;
};

using VachConfig = AudioIoConfig;

/**
 * @brief Telemetry counters for the lock-free capture and playback ring
 *        buffers and hardware ALSA xruns.
 */
struct RingBufferStats
{
  std::size_t capture_occupancy = 0U;
  std::size_t playback_occupancy = 0U;
  uint64_t capture_overruns = 0U;
  uint64_t playback_overruns = 0U;
  uint64_t playback_underruns = 0U;
  uint64_t total_capture_samples = 0U;
  uint64_t total_playback_samples = 0U;
  uint64_t xrun_count = 0U;
};

/**
 * @brief Real-time audio I/O subsystem with lock-free SPSC ring buffers behind
 *        an opaque handle (ALSA hardware threads / headless loopback).
 */
class AudioIo
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  AudioIo (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates an AudioIo instance, opens real ALSA PCM capture/playback
   *        streams (unless headless_loopback is enabled), and spawns RT
   *        capture/playback threads.
   * @param config Audio device and ring buffer configuration.
   * @returns sutra::Result<std::unique_ptr<AudioIo>> or
   *          AgentError::AudioDevice.
   */
  [[nodiscard]] static auto create (const AudioIoConfig &config
                                    = AudioIoConfig{}) noexcept
      -> sutra::Result<std::unique_ptr<AudioIo>>;

  AudioIo (const AudioIo &) = delete;
  auto operator= (const AudioIo &) -> AudioIo & = delete;
  AudioIo (AudioIo &&) noexcept;
  auto operator= (AudioIo &&) noexcept -> AudioIo &;
  ~AudioIo () noexcept;

  /**
   * @brief Real-time producer push for captured 16 kHz float samples.
   * @note Lock-free, allocation-free, log-free; safe for RT audio callback.
   * @returns Number of samples written to the capture ring buffer.
   */
  auto
  rt_push_capture (std::span<const float> samples) noexcept -> std::size_t;

  /**
   * @brief Worker consumer pop for captured 16 kHz float samples.
   * @returns Number of samples read into out_buffer.
   */
  auto pop_capture (std::span<float> out_buffer) noexcept -> std::size_t;

  /**
   * @brief Worker producer push for synthesized 16-bit PCM playback samples.
   * @returns Number of samples written to the playback ring buffer.
   */
  auto
  push_playback (std::span<const int16_t> samples) noexcept -> std::size_t;

  /**
   * @brief Lock-free producer push alias for 16-bit PCM playback samples.
   */
  auto
  rt_push_playback (std::span<const int16_t> samples) noexcept -> std::size_t
  {
    return push_playback (samples);
  }

  /**
   * @brief Real-time consumer pop for 16-bit PCM playback samples.
   * @note Lock-free, allocation-free, log-free; safe for RT audio callback.
   * @returns Number of samples pulled from the playback ring buffer.
   */
  auto
  rt_pull_playback (std::span<int16_t> out_buffer) noexcept -> std::size_t;

  /**
   * @brief Immediately flushes the playback ring buffer and signals the ALSA
   *        playback thread to drop buffered hardware frames for barge-in.
   * @note Lock-free atomic reset; completes in < 1 us.
   */
  void clear_playback () noexcept;

  /**
   * @brief Returns current lock-free ring buffer and ALSA telemetry.
   */
  [[nodiscard]] auto stats () const noexcept -> RingBufferStats;

  /**
   * @brief Runs a 1-hour continuous stream equivalent (57.6M samples at 16
   *        kHz) across the lock-free capture/playback ring buffers between
   *        concurrent producer and consumer threads, asserting 0 underruns and
   *        0 overruns.
   */
  [[nodiscard]] auto
  run_one_hour_ring_stress_test () noexcept -> sutra::Result<RingBufferStats>;

private:
  std::unique_ptr<Impl> impl_;
};

} /* namespace vach::inline v1 */

#endif /* VACH_AUDIO_IO_HPP */
