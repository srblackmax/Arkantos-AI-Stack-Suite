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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sutra/observability.hpp>
#include <vach/vach.hpp>

#include <alsa/asoundlib.h>
#include <dlfcn.h>
#include <espeak-ng/speak_lib.h>
#include <fcntl.h>
#include <malloc.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <whisper/whisper.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vach::inline v1
{

namespace
{

/* ========================================================================= */
/* Lock-Free Ring Buffer                                                     */
/* ========================================================================= */

template <typename T> class LockFreeRingBuffer
{
public:
  explicit LockFreeRingBuffer (std::size_t capacity_samples)
      : capacity_ (capacity_samples + 1U), buf_ (capacity_samples + 1U, T{})
  {
  }

  auto
  push (std::span<const T> src) noexcept -> std::size_t
  {
    while (guard_.test_and_set (std::memory_order_acquire))
      {
      }
    const std::size_t w = write_pos_.load (std::memory_order_relaxed);
    const std::size_t r = read_pos_.load (std::memory_order_relaxed);
    const std::size_t used = (w >= r) ? (w - r) : (capacity_ - (r - w));
    const std::size_t free_space = (capacity_ - 1U) - used;
    const std::size_t to_write = std::min (src.size (), free_space);
    if (to_write < src.size ())
      {
        overruns_.fetch_add (1U, std::memory_order_relaxed);
      }
    if (to_write > 0U)
      {
        const std::size_t first_chunk = std::min (to_write, capacity_ - w);
        std::memcpy (&buf_[w], src.data (), first_chunk * sizeof (T));
        if (to_write > first_chunk)
          {
            std::memcpy (&buf_[0], src.data () + first_chunk,
                         (to_write - first_chunk) * sizeof (T));
          }
        const std::size_t next_w = (w + to_write) % capacity_;
        write_pos_.store (next_w, std::memory_order_relaxed);
        total_pushed_.fetch_add (to_write, std::memory_order_relaxed);
      }
    guard_.clear (std::memory_order_release);
    return to_write;
  }

  auto
  pop (std::span<T> dst, bool record_underrun) noexcept -> std::size_t
  {
    while (guard_.test_and_set (std::memory_order_acquire))
      {
      }
    const std::size_t r = read_pos_.load (std::memory_order_relaxed);
    const std::size_t w = write_pos_.load (std::memory_order_relaxed);
    const std::size_t avail = (w >= r) ? (w - r) : (capacity_ - (r - w));
    const std::size_t to_read = std::min (dst.size (), avail);
    if (record_underrun && to_read < dst.size ())
      {
        underruns_.fetch_add (1U, std::memory_order_relaxed);
      }
    if (to_read > 0U)
      {
        const std::size_t first_chunk = std::min (to_read, capacity_ - r);
        std::memcpy (dst.data (), &buf_[r], first_chunk * sizeof (T));
        if (to_read > first_chunk)
          {
            std::memcpy (dst.data () + first_chunk, &buf_[0],
                         (to_read - first_chunk) * sizeof (T));
          }
        const std::size_t next_r = (r + to_read) % capacity_;
        read_pos_.store (next_r, std::memory_order_relaxed);
        total_popped_.fetch_add (to_read, std::memory_order_relaxed);
      }
    guard_.clear (std::memory_order_release);
    return to_read;
  }

  void
  clear () noexcept
  {
    while (guard_.test_and_set (std::memory_order_acquire))
      {
      }
    const std::size_t w = write_pos_.load (std::memory_order_relaxed);
    read_pos_.store (w, std::memory_order_relaxed);
    guard_.clear (std::memory_order_release);
  }

  [[nodiscard]] auto
  occupancy () const noexcept -> std::size_t
  {
    while (guard_.test_and_set (std::memory_order_acquire))
      {
      }
    const std::size_t r = read_pos_.load (std::memory_order_relaxed);
    const std::size_t w = write_pos_.load (std::memory_order_relaxed);
    const std::size_t occ = (w >= r) ? (w - r) : (capacity_ - (r - w));
    guard_.clear (std::memory_order_release);
    return occ;
  }

  [[nodiscard]] auto
  overruns () const noexcept -> uint64_t
  {
    return overruns_.load (std::memory_order_relaxed);
  }

  [[nodiscard]] auto
  underruns () const noexcept -> uint64_t
  {
    return underruns_.load (std::memory_order_relaxed);
  }

  [[nodiscard]] auto
  total_pushed () const noexcept -> uint64_t
  {
    return total_pushed_.load (std::memory_order_relaxed);
  }

  [[nodiscard]] auto
  total_popped () const noexcept -> uint64_t
  {
    return total_popped_.load (std::memory_order_relaxed);
  }

private:
  std::size_t capacity_;
  std::vector<T> buf_;
  mutable std::atomic_flag guard_ = ATOMIC_FLAG_INIT;
  alignas (64) std::atomic<std::size_t> write_pos_{ 0U };
  alignas (64) std::atomic<std::size_t> read_pos_{ 0U };
  std::atomic<uint64_t> overruns_{ 0U };
  std::atomic<uint64_t> underruns_{ 0U };
  std::atomic<uint64_t> total_pushed_{ 0U };
  std::atomic<uint64_t> total_popped_{ 0U };
};

/* ========================================================================= */
/* Resident ALSA dlopen Table                                                */
/* ========================================================================= */

struct AlsaApi
{
  void *handle = nullptr;
  decltype (&snd_pcm_open) open = nullptr;
  decltype (&snd_pcm_close) close = nullptr;
  decltype (&snd_pcm_drain) drain = nullptr;
  decltype (&snd_pcm_drop) drop = nullptr;
  decltype (&snd_pcm_prepare) prepare = nullptr;
  decltype (&snd_pcm_start) start = nullptr;
  decltype (&snd_pcm_resume) resume = nullptr;
  decltype (&snd_pcm_wait) wait = nullptr;
  decltype (&snd_pcm_readi) readi = nullptr;
  decltype (&snd_pcm_writei) writei = nullptr;
  decltype (&snd_pcm_hw_params_malloc) hw_malloc = nullptr;
  decltype (&snd_pcm_hw_params_free) hw_free = nullptr;
  decltype (&snd_pcm_hw_params_any) hw_any = nullptr;
  decltype (&snd_pcm_hw_params_set_access) hw_set_access = nullptr;
  decltype (&snd_pcm_hw_params_set_format) hw_set_format = nullptr;
  decltype (&snd_pcm_hw_params_set_channels) hw_set_channels = nullptr;
  decltype (&snd_pcm_hw_params_set_rate_near) hw_set_rate_near = nullptr;
  decltype (&snd_pcm_hw_params_set_period_size_near) hw_set_period_near
      = nullptr;
  decltype (&snd_pcm_hw_params_set_buffer_size_near) hw_set_buffer_near
      = nullptr;
  decltype (&snd_pcm_hw_params) hw_params = nullptr;
  decltype (&snd_pcm_sw_params_malloc) sw_malloc = nullptr;
  decltype (&snd_pcm_sw_params_free) sw_free = nullptr;
  decltype (&snd_pcm_sw_params_current) sw_current = nullptr;
  decltype (&snd_pcm_sw_params_set_start_threshold) sw_set_start_thresh
      = nullptr;
  decltype (&snd_pcm_sw_params_set_avail_min) sw_set_avail_min = nullptr;
  decltype (&snd_pcm_sw_params) sw_params = nullptr;
  decltype (&snd_strerror) strerror_fn = nullptr;
  bool valid = false;
};

template <typename FnPtr>
auto
resolve_sym (void *handle, const char *name, FnPtr &out) noexcept -> bool
{
  void *sym = dlsym (handle, name);
  if (sym == nullptr)
    {
      return false;
    }
  std::memcpy (&out, &sym, sizeof (FnPtr));
  return true;
}

auto
resident_alsa_api () noexcept -> const AlsaApi &
{
  static const AlsaApi api = [] () noexcept {
    AlsaApi a{};
    a.handle = dlopen ("libasound.so.2", RTLD_NOW | RTLD_LOCAL);
    if (a.handle == nullptr)
      {
        a.handle = dlopen ("libasound.so", RTLD_NOW | RTLD_LOCAL);
      }
    if (a.handle == nullptr)
      {
        return a;
      }
    const bool ok
        = resolve_sym (a.handle, "snd_pcm_open", a.open)
          && resolve_sym (a.handle, "snd_pcm_close", a.close)
          && resolve_sym (a.handle, "snd_pcm_drain", a.drain)
          && resolve_sym (a.handle, "snd_pcm_drop", a.drop)
          && resolve_sym (a.handle, "snd_pcm_prepare", a.prepare)
          && resolve_sym (a.handle, "snd_pcm_start", a.start)
          && resolve_sym (a.handle, "snd_pcm_resume", a.resume)
          && resolve_sym (a.handle, "snd_pcm_wait", a.wait)
          && resolve_sym (a.handle, "snd_pcm_readi", a.readi)
          && resolve_sym (a.handle, "snd_pcm_writei", a.writei)
          && resolve_sym (a.handle, "snd_pcm_hw_params_malloc", a.hw_malloc)
          && resolve_sym (a.handle, "snd_pcm_hw_params_free", a.hw_free)
          && resolve_sym (a.handle, "snd_pcm_hw_params_any", a.hw_any)
          && resolve_sym (a.handle, "snd_pcm_hw_params_set_access",
                          a.hw_set_access)
          && resolve_sym (a.handle, "snd_pcm_hw_params_set_format",
                          a.hw_set_format)
          && resolve_sym (a.handle, "snd_pcm_hw_params_set_channels",
                          a.hw_set_channels)
          && resolve_sym (a.handle, "snd_pcm_hw_params_set_rate_near",
                          a.hw_set_rate_near)
          && resolve_sym (a.handle, "snd_pcm_hw_params_set_period_size_near",
                          a.hw_set_period_near)
          && resolve_sym (a.handle, "snd_pcm_hw_params_set_buffer_size_near",
                          a.hw_set_buffer_near)
          && resolve_sym (a.handle, "snd_pcm_hw_params", a.hw_params)
          && resolve_sym (a.handle, "snd_pcm_sw_params_malloc", a.sw_malloc)
          && resolve_sym (a.handle, "snd_pcm_sw_params_free", a.sw_free)
          && resolve_sym (a.handle, "snd_pcm_sw_params_current", a.sw_current)
          && resolve_sym (a.handle, "snd_pcm_sw_params_set_start_threshold",
                          a.sw_set_start_thresh)
          && resolve_sym (a.handle, "snd_pcm_sw_params_set_avail_min",
                          a.sw_set_avail_min)
          && resolve_sym (a.handle, "snd_pcm_sw_params", a.sw_params)
          && resolve_sym (a.handle, "snd_strerror", a.strerror_fn);
    a.valid = ok;
    return a;
  }();
  return api;
}

void
configure_rt_thread (int core_id, const char *thread_label) noexcept
{
  cpu_set_t cpuset;
  CPU_ZERO (&cpuset);
  CPU_SET (core_id, &cpuset);
  (void)sched_setaffinity (0, sizeof (cpu_set_t), &cpuset);

  struct sched_param sp
  {
  };
  sp.sched_priority = 50;
  const int rc = pthread_setschedparam (pthread_self (), SCHED_FIFO, &sp);
  if (rc != 0)
    {
      struct sched_param other_sp
      {
      };
      other_sp.sched_priority = 0;
      (void)pthread_setschedparam (pthread_self (), SCHED_OTHER, &other_sp);
      static std::atomic<bool> logged_once{ false };
      if (!logged_once.exchange (true, std::memory_order_relaxed))
        {
          sutra::Logger::log (
              sutra::LogLevel::Warn, "vach",
              std::string ("SCHED_FIFO priority 50 unavailable on ")
                  + thread_label
                  + " (CAP_SYS_NICE not granted); falling back to "
                    "SCHED_OTHER");
        }
    }
}

auto
setup_alsa_stream (const AlsaApi &alsa, snd_pcm_t *pcm, unsigned int req_rate,
                   unsigned int channels, snd_pcm_uframes_t req_period,
                   snd_pcm_uframes_t req_buffer,
                   snd_pcm_uframes_t start_thresh) noexcept
    -> sutra::Result<void>
{
  snd_pcm_hw_params_t *hw = nullptr;
  if (alsa.hw_malloc (&hw) < 0 || hw == nullptr)
    {
      return std::unexpected (
          sutra::AgentError::AudioDevice ("snd_pcm_hw_params_malloc failed"));
    }

  auto free_hw = [&] () noexcept { alsa.hw_free (hw); };

  if (alsa.hw_any (pcm, hw) < 0
      || alsa.hw_set_access (pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED) < 0
      || alsa.hw_set_format (pcm, hw, SND_PCM_FORMAT_S16_LE) < 0
      || alsa.hw_set_channels (pcm, hw, channels) < 0)
    {
      free_hw ();
      return std::unexpected (sutra::AgentError::AudioDevice (
          "ALSA basic hw_params setup failed"));
    }

  unsigned int actual_rate = req_rate;
  int dir = 0;
  if (alsa.hw_set_rate_near (pcm, hw, &actual_rate, &dir) < 0
      || actual_rate != req_rate)
    {
      free_hw ();
      return std::unexpected (sutra::AgentError::AudioDevice (
          "ALSA sample rate mismatch: requested " + std::to_string (req_rate)
          + " got " + std::to_string (actual_rate)));
    }

  snd_pcm_uframes_t actual_period = req_period;
  dir = 0;
  if (alsa.hw_set_period_near (pcm, hw, &actual_period, &dir) < 0)
    {
      free_hw ();
      return std::unexpected (
          sutra::AgentError::AudioDevice ("ALSA set_period_size_near failed"));
    }

  snd_pcm_uframes_t actual_buffer = req_buffer;
  if (alsa.hw_set_buffer_near (pcm, hw, &actual_buffer) < 0)
    {
      free_hw ();
      return std::unexpected (
          sutra::AgentError::AudioDevice ("ALSA set_buffer_size_near failed"));
    }

  if (actual_period != req_period || actual_buffer != req_buffer)
    {
      free_hw ();
      return std::unexpected (sutra::AgentError::AudioDevice (
          "ALSA period/buffer mismatch: requested "
          + std::to_string (req_period) + "/" + std::to_string (req_buffer)
          + " got " + std::to_string (actual_period) + "/"
          + std::to_string (actual_buffer)));
    }

  if (alsa.hw_params (pcm, hw) < 0)
    {
      free_hw ();
      return std::unexpected (
          sutra::AgentError::AudioDevice ("snd_pcm_hw_params commit failed"));
    }
  free_hw ();

  snd_pcm_sw_params_t *sw = nullptr;
  if (alsa.sw_malloc (&sw) < 0 || sw == nullptr)
    {
      return std::unexpected (
          sutra::AgentError::AudioDevice ("snd_pcm_sw_params_malloc failed"));
    }

  if (alsa.sw_current (pcm, sw) < 0
      || alsa.sw_set_start_thresh (pcm, sw, start_thresh) < 0
      || alsa.sw_set_avail_min (pcm, sw, req_period) < 0
      || alsa.sw_params (pcm, sw) < 0)
    {
      alsa.sw_free (sw);
      return std::unexpected (
          sutra::AgentError::AudioDevice ("snd_pcm_sw_params commit failed"));
    }
  alsa.sw_free (sw);

  if (alsa.prepare (pcm) < 0)
    {
      return std::unexpected (
          sutra::AgentError::AudioDevice ("snd_pcm_prepare failed"));
    }
  return {};
}

/* ========================================================================= */
/* Resident Whisper Backend                                                  */
/* ========================================================================= */

struct WhisperApi
{
  void *handle = nullptr;
  decltype (&whisper_init_from_file_with_params) init_from_file = nullptr;
  decltype (&whisper_context_default_params) context_default_params = nullptr;
  decltype (&whisper_full_default_params) full_default_params = nullptr;
  decltype (&whisper_full) full = nullptr;
  decltype (&whisper_full_n_segments) full_n_segments = nullptr;
  decltype (&whisper_full_get_segment_text) full_get_segment_text = nullptr;
  decltype (&whisper_free) free_ctx = nullptr;
  decltype (&whisper_log_set) log_set = nullptr;
  bool valid = false;
};

void
whisper_silent_log_cb (enum ggml_log_level, const char *, void *)
{
}

auto
resident_whisper_api () noexcept -> const WhisperApi &
{
  static const WhisperApi api = [] () noexcept {
    WhisperApi w{};
    const char *env_home = std::getenv ("HOME");
    if (env_home != nullptr && env_home[0] != '\0')
      {
        const std::string local_lib
            = std::string (env_home) + "/.local/lib/libwhisper.so";
        w.handle = dlopen (local_lib.c_str (), RTLD_NOW | RTLD_LOCAL);
      }
    if (w.handle == nullptr)
      {
        w.handle = dlopen ("libwhisper.so", RTLD_NOW | RTLD_LOCAL);
      }
    if (w.handle == nullptr)
      {
        return w;
      }
    const bool ok
        = resolve_sym (w.handle, "whisper_init_from_file_with_params",
                       w.init_from_file)
          && resolve_sym (w.handle, "whisper_context_default_params",
                          w.context_default_params)
          && resolve_sym (w.handle, "whisper_full_default_params",
                          w.full_default_params)
          && resolve_sym (w.handle, "whisper_full", w.full)
          && resolve_sym (w.handle, "whisper_full_n_segments",
                          w.full_n_segments)
          && resolve_sym (w.handle, "whisper_full_get_segment_text",
                          w.full_get_segment_text)
          && resolve_sym (w.handle, "whisper_free", w.free_ctx)
          && resolve_sym (w.handle, "whisper_log_set", w.log_set);
    if (ok && w.log_set != nullptr)
      {
        w.log_set (whisper_silent_log_cb, nullptr);
      }
    w.valid = ok;
    return w;
  }();
  return api;
}

struct ResidentWhisperModel
{
  std::mutex mtx;
  std::unordered_map<std::string, whisper_context *> contexts;

  auto
  get_or_load (const std::string &model_path) noexcept -> whisper_context *
  {
    std::scoped_lock lk (mtx);
    auto it = contexts.find (model_path);
    if (it != contexts.end ())
      {
        return it->second;
      }
    const auto &w = resident_whisper_api ();
    if (!w.valid || !std::filesystem::exists (model_path))
      {
        return nullptr;
      }
    whisper_context_params cparams = w.context_default_params ();
    cparams.use_gpu = true;
    whisper_context *ctx = w.init_from_file (model_path.c_str (), cparams);
    if (ctx != nullptr)
      {
        contexts.emplace (model_path, ctx);
      }
    return ctx;
  }
};

auto
resident_whisper_models () noexcept -> ResidentWhisperModel &
{
  static ResidentWhisperModel mgr;
  return mgr;
}

/* ========================================================================= */
/* Resident Piper ONNX + eSpeak-NG Backend                                   */
/* ========================================================================= */

auto
next_utf8_cp (const char *&p, const char *end) noexcept -> uint32_t
{
  const auto c0 = static_cast<unsigned char> (*p++);
  if (c0 < 0x80U)
    {
      return c0;
    }
  if ((c0 & 0xE0U) == 0xC0U && p < end)
    {
      const uint32_t c1 = static_cast<unsigned char> (*p++) & 0x3FU;
      return ((c0 & 0x1FU) << 6U) | c1;
    }
  if ((c0 & 0xF0U) == 0xE0U && p + 1 < end)
    {
      const uint32_t c1 = static_cast<unsigned char> (*p++) & 0x3FU;
      const uint32_t c2 = static_cast<unsigned char> (*p++) & 0x3FU;
      return ((c0 & 0x0FU) << 12U) | (c1 << 6U) | c2;
    }
  if ((c0 & 0xF8U) == 0xF0U && p + 2 < end)
    {
      const uint32_t c1 = static_cast<unsigned char> (*p++) & 0x3FU;
      const uint32_t c2 = static_cast<unsigned char> (*p++) & 0x3FU;
      const uint32_t c3 = static_cast<unsigned char> (*p++) & 0x3FU;
      return ((c0 & 0x07U) << 18U) | (c1 << 12U) | (c2 << 6U) | c3;
    }
  return 0U;
}

auto
parse_piper_phoneme_map (const std::string &json_path,
                         std::unordered_map<uint32_t, std::vector<int64_t>>
                             &out_map) noexcept -> bool
{
  std::ifstream in (json_path);
  if (!in.is_open ())
    {
      return false;
    }
  std::string content ((std::istreambuf_iterator<char> (in)),
                       std::istreambuf_iterator<char> ());
  const auto pos = content.find ("\"phoneme_id_map\"");
  if (pos == std::string::npos)
    {
      return false;
    }
  const auto brace_start = content.find ('{', pos);
  if (brace_start == std::string::npos)
    {
      return false;
    }
  std::size_t i = brace_start + 1U;
  int depth = 1;
  while (i < content.size () && depth > 0)
    {
      const char c = content[i];
      if (c == '{')
        {
          ++depth;
          ++i;
        }
      else if (c == '}')
        {
          --depth;
          ++i;
        }
      else if (c == '"' && depth == 1)
        {
          ++i;
          std::string key;
          while (i < content.size () && content[i] != '"')
            {
              if (content[i] == '\\' && i + 1U < content.size ())
                {
                  ++i;
                  if (content[i] == 'n')
                    {
                      key.push_back ('\n');
                    }
                  else if (content[i] == 't')
                    {
                      key.push_back ('\t');
                    }
                  else
                    {
                      key.push_back (content[i]);
                    }
                }
              else
                {
                  key.push_back (content[i]);
                }
              ++i;
            }
          if (i < content.size ())
            {
              ++i;
            }
          const auto arr_start = content.find ('[', i);
          const auto arr_end = content.find (']', arr_start);
          if (arr_start == std::string::npos || arr_end == std::string::npos)
            {
              break;
            }
          std::vector<int64_t> ids;
          std::size_t j = arr_start + 1U;
          while (j < arr_end)
            {
              while (j < arr_end && (content[j] < '0' || content[j] > '9'))
                {
                  ++j;
                }
              if (j < arr_end)
                {
                  int64_t val = 0;
                  while (j < arr_end && content[j] >= '0' && content[j] <= '9')
                    {
                      val = val * 10 + (content[j] - '0');
                      ++j;
                    }
                  ids.push_back (val);
                }
            }
          if (!key.empty ())
            {
              const char *kp = key.data ();
              const uint32_t cp = next_utf8_cp (kp, key.data () + key.size ());
              out_map[cp] = std::move (ids);
            }
          i = arr_end + 1U;
        }
      else
        {
          ++i;
        }
    }
  return !out_map.empty ();
}

struct PiperRuntime
{
  std::mutex mtx;
  void *espeak_handle = nullptr;
  decltype (&espeak_Initialize) espeak_init_fn = nullptr;
  decltype (&espeak_SetVoiceByName) espeak_set_voice_fn = nullptr;
  decltype (&espeak_TextToPhonemes) espeak_t2p_fn = nullptr;

  void *ort_handle = nullptr;
  const OrtApi *ort = nullptr;
  OrtEnv *env = nullptr;
  OrtMemoryInfo *mem_info = nullptr;
  std::unordered_map<std::string, OrtSession *> sessions;
  std::unordered_map<std::string,
                     std::unordered_map<uint32_t, std::vector<int64_t>>>
      phoneme_maps;
  struct CacheEntry
  {
    std::string key;
    std::vector<float> pcm;
  };
  std::array<CacheEntry, 32> synth_cache{};
  std::size_t cache_cursor = 0U;
  bool initialized = false;

  void
  release_ort_status (OrtStatus *st) const noexcept
  {
    if (st != nullptr && ort != nullptr)
      {
        ort->ReleaseStatus (st);
      }
  }

  void
  ensure_init () noexcept
  {
    if (initialized)
      {
        return;
      }
    espeak_handle = dlopen ("libespeak-ng.so.1", RTLD_NOW | RTLD_LOCAL);
    if (espeak_handle == nullptr)
      {
        espeak_handle = dlopen ("libespeak-ng.so", RTLD_NOW | RTLD_LOCAL);
      }
    if (espeak_handle != nullptr)
      {
        (void)resolve_sym (espeak_handle, "espeak_Initialize", espeak_init_fn);
        (void)resolve_sym (espeak_handle, "espeak_SetVoiceByName",
                           espeak_set_voice_fn);
        (void)resolve_sym (espeak_handle, "espeak_TextToPhonemes",
                           espeak_t2p_fn);
        if (espeak_init_fn != nullptr && espeak_set_voice_fn != nullptr)
          {
            (void)espeak_init_fn (AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
            (void)espeak_set_voice_fn ("en-us");
          }
      }

    /* Silence benign ONNX schema registration prints during dlopen/init.  */
    const int saved_out = dup (STDOUT_FILENO);
    const int saved_err = dup (STDERR_FILENO);
    const int devnull = open ("/dev/null", O_WRONLY);
    if (devnull >= 0)
      {
        (void)dup2 (devnull, STDOUT_FILENO);
        (void)dup2 (devnull, STDERR_FILENO);
        close (devnull);
      }
    ort_handle = dlopen ("libonnxruntime.so.1.21.0", RTLD_NOW | RTLD_LOCAL);
    if (ort_handle == nullptr)
      {
        ort_handle = dlopen ("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
      }

    if (ort_handle != nullptr)
      {
        using GetApiBaseFn = const OrtApiBase *(*) ();
        GetApiBaseFn get_base = nullptr;
        if (resolve_sym (ort_handle, "OrtGetApiBase", get_base)
            && get_base != nullptr)
          {
            ort = get_base ()->GetApi (ORT_API_VERSION);
            if (ort != nullptr)
              {
                release_ort_status (ort->CreateEnv (ORT_LOGGING_LEVEL_ERROR,
                                                    "vach_piper", &env));
                release_ort_status (ort->CreateCpuMemoryInfo (
                    OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
              }
          }
      }
    if (saved_out >= 0)
      {
        (void)dup2 (saved_out, STDOUT_FILENO);
        close (saved_out);
      }
    if (saved_err >= 0)
      {
        (void)dup2 (saved_err, STDERR_FILENO);
        close (saved_err);
      }
    initialized = (espeak_t2p_fn != nullptr && ort != nullptr
                   && env != nullptr && mem_info != nullptr);
  }

  auto
  get_session (const std::string &model_path,
               const std::string &config_path) noexcept -> OrtSession *
  {
    ensure_init ();
    if (!initialized)
      {
        return nullptr;
      }
    auto it = sessions.find (model_path);
    if (it != sessions.end ())
      {
        return it->second;
      }
    if (!std::filesystem::exists (model_path)
        || !std::filesystem::exists (config_path))
      {
        return nullptr;
      }
    std::unordered_map<uint32_t, std::vector<int64_t>> pmap;
    if (!parse_piper_phoneme_map (config_path, pmap))
      {
        return nullptr;
      }
    OrtSessionOptions *opts = nullptr;
    OrtStatus *st = ort->CreateSessionOptions (&opts);
    if (st != nullptr || opts == nullptr)
      {
        release_ort_status (st);
        return nullptr;
      }
    release_ort_status (ort->SetIntraOpNumThreads (opts, 8));
    release_ort_status (
        ort->SetSessionGraphOptimizationLevel (opts, ORT_ENABLE_ALL));

    OrtSession *sess = nullptr;
    const int saved_out = dup (STDOUT_FILENO);
    const int saved_err = dup (STDERR_FILENO);
    const int devnull = open ("/dev/null", O_WRONLY);
    if (devnull >= 0)
      {
        (void)dup2 (devnull, STDOUT_FILENO);
        (void)dup2 (devnull, STDERR_FILENO);
        close (devnull);
      }
    st = ort->CreateSession (env, model_path.c_str (), opts, &sess);
    (void)std::fflush (stdout);
    (void)std::fflush (stderr);
    if (saved_out >= 0)
      {
        (void)dup2 (saved_out, STDOUT_FILENO);
        close (saved_out);
      }
    if (saved_err >= 0)
      {
        (void)dup2 (saved_err, STDERR_FILENO);
        close (saved_err);
      }
    ort->ReleaseSessionOptions (opts);
    if (st != nullptr || sess == nullptr)
      {
        release_ort_status (st);
        return nullptr;
      }
    phoneme_maps[model_path] = std::move (pmap);
    sessions[model_path] = sess;
    return sess;
  }

  auto
  synthesize_22k_locked (const std::string &model_path,
                         const std::string &config_path,
                         std::string_view sentence,
                         std::vector<float> &out_pcm_22k) noexcept
      -> sutra::Result<void>
  {
    std::scoped_lock lk (mtx);
    const std::string cache_key = model_path + "|" + std::string (sentence);
    for (const auto &entry : synth_cache)
      {
        if (!entry.key.empty () && entry.key == cache_key)
          {
            out_pcm_22k.assign (entry.pcm.begin (), entry.pcm.end ());
            return {};
          }
      }

    OrtSession *sess = get_session (model_path, config_path);
    if (sess == nullptr)
      {
        return std::unexpected (sutra::AgentError::Tts (
            "Failed to initialize Piper ONNX session for " + model_path));
      }
    const auto &pmap = phoneme_maps[model_path];

    (void)espeak_set_voice_fn ("en-us");
    std::string text_copy (sentence);
    const void *text_ptr = text_copy.c_str ();
    std::string ipa_str;
    while (text_ptr != nullptr
           && *static_cast<const char *> (text_ptr) != '\0')
      {
        const char *ph = espeak_t2p_fn (&text_ptr, espeakCHARS_UTF8, 0x02);
        if (ph != nullptr && *ph != '\0')
          {
            if (!ipa_str.empty ())
              {
                ipa_str.push_back (' ');
              }
            ipa_str.append (ph);
          }
      }

    std::vector<int64_t> phoneme_ids;
    phoneme_ids.reserve (ipa_str.size () * 2U + 4U);
    phoneme_ids.push_back (1); /* '^' BOS  */
    phoneme_ids.push_back (0); /* '_' pad  */

    const char *p = ipa_str.data ();
    const char *end = ipa_str.data () + ipa_str.size ();
    while (p < end)
      {
        const uint32_t cp = next_utf8_cp (p, end);
        auto it = pmap.find (cp);
        if (it != pmap.end ())
          {
            for (const int64_t id : it->second)
              {
                phoneme_ids.push_back (id);
                phoneme_ids.push_back (0);
              }
          }
      }
    phoneme_ids.push_back (2); /* '$' EOS  */

    const std::array<int64_t, 2> input_shape
        = { 1, static_cast<int64_t> (phoneme_ids.size ()) };
    int64_t input_len_val = static_cast<int64_t> (phoneme_ids.size ());
    const std::array<int64_t, 1> len_shape = { 1 };
    std::array<float, 3> scales = { 0.333F, 1.0F, 0.4F };
    const std::array<int64_t, 1> scales_shape = { 3 };

    OrtValue *in_tensor = nullptr;
    OrtValue *len_tensor = nullptr;
    OrtValue *scales_tensor = nullptr;
    OrtStatus *st = ort->CreateTensorWithDataAsOrtValue (
        mem_info, phoneme_ids.data (), phoneme_ids.size () * sizeof (int64_t),
        input_shape.data (), input_shape.size (),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_tensor);
    if (st != nullptr)
      {
        release_ort_status (st);
        return std::unexpected (
            sutra::AgentError::Tts ("ORT input tensor creation failed"));
      }
    st = ort->CreateTensorWithDataAsOrtValue (
        mem_info, &input_len_val, sizeof (int64_t), len_shape.data (),
        len_shape.size (), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor);
    if (st != nullptr)
      {
        release_ort_status (st);
        ort->ReleaseValue (in_tensor);
        return std::unexpected (
            sutra::AgentError::Tts ("ORT input_lengths tensor failed"));
      }
    st = ort->CreateTensorWithDataAsOrtValue (
        mem_info, scales.data (), scales.size () * sizeof (float),
        scales_shape.data (), scales_shape.size (),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &scales_tensor);
    if (st != nullptr)
      {
        release_ort_status (st);
        ort->ReleaseValue (len_tensor);
        ort->ReleaseValue (in_tensor);
        return std::unexpected (
            sutra::AgentError::Tts ("ORT scales tensor failed"));
      }

    const std::array<const char *, 3> in_names
        = { "input", "input_lengths", "scales" };
    const std::array<const OrtValue *, 3> in_vals
        = { in_tensor, len_tensor, scales_tensor };
    const std::array<const char *, 1> out_names = { "output" };
    OrtValue *out_tensor = nullptr;

    st = ort->Run (sess, nullptr, in_names.data (), in_vals.data (), 3U,
                   out_names.data (), 1U, &out_tensor);
    ort->ReleaseValue (scales_tensor);
    ort->ReleaseValue (len_tensor);
    ort->ReleaseValue (in_tensor);

    if (st != nullptr || out_tensor == nullptr)
      {
        release_ort_status (st);
        return std::unexpected (
            sutra::AgentError::Tts ("Piper ONNX Run failed"));
      }

    float *audio_ptr = nullptr;
    st = ort->GetTensorMutableData (out_tensor,
                                    reinterpret_cast<void **> (&audio_ptr));
    if (st != nullptr)
      {
        release_ort_status (st);
        ort->ReleaseValue (out_tensor);
        return std::unexpected (
            sutra::AgentError::Tts ("GetTensorMutableData failed"));
      }

    OrtTensorTypeAndShapeInfo *shape_info = nullptr;
    st = ort->GetTensorTypeAndShape (out_tensor, &shape_info);
    if (st != nullptr)
      {
        release_ort_status (st);
        ort->ReleaseValue (out_tensor);
        return std::unexpected (
            sutra::AgentError::Tts ("GetTensorTypeAndShape failed"));
      }

    std::size_t dim_count = 0U;
    release_ort_status (ort->GetDimensionsCount (shape_info, &dim_count));
    std::vector<int64_t> dims (dim_count, 0);
    release_ort_status (
        ort->GetDimensions (shape_info, dims.data (), dim_count));
    ort->ReleaseTensorTypeAndShapeInfo (shape_info);

    std::size_t total_samples = 1U;
    for (const int64_t d : dims)
      {
        if (d > 0)
          {
            total_samples *= static_cast<std::size_t> (d);
          }
      }
    out_pcm_22k.assign (audio_ptr, audio_ptr + total_samples);
    ort->ReleaseValue (out_tensor);
    synth_cache[cache_cursor].key = cache_key;
    synth_cache[cache_cursor].pcm = out_pcm_22k;
    cache_cursor = (cache_cursor + 1U) % synth_cache.size ();
    return {};
  }
};

auto
resident_piper () noexcept -> PiperRuntime &
{
  static PiperRuntime rt;
  return rt;
}

} /* namespace */

/* ========================================================================= */
/* vach::AudioIo Implementation                                              */
/* ========================================================================= */

struct AudioIo::Impl
{
  AudioIoConfig config;
  LockFreeRingBuffer<float> capture_ring;
  LockFreeRingBuffer<int16_t> playback_ring;
  snd_pcm_t *cap_pcm = nullptr;
  snd_pcm_t *play_pcm = nullptr;
  std::atomic<bool> running{ false };
  std::atomic<bool> drop_requested{ false };
  std::atomic<bool> synthetic_capture_mode{ false };
  std::atomic<uint64_t> xrun_count{ 0U };
  std::atomic<uint64_t> alsa_capture_samples{ 0U };
  std::atomic<uint64_t> stress_capture_samples{ 0U };
  std::atomic<uint64_t> stress_playback_samples{ 0U };
  std::thread cap_thread;
  std::thread play_thread;

  explicit Impl (const AudioIoConfig &cfg)
      : config (cfg), capture_ring (cfg.ring_capacity_samples),
        playback_ring (cfg.ring_capacity_samples * 2U)
  {
  }

  ~Impl () noexcept
  {
    running.store (false, std::memory_order_release);
    if (cap_thread.joinable ())
      {
        cap_thread.join ();
      }
    if (play_thread.joinable ())
      {
        play_thread.join ();
      }
    const auto &alsa = resident_alsa_api ();
    if (alsa.valid)
      {
        if (cap_pcm != nullptr)
          {
            (void)alsa.drain (cap_pcm);
            (void)alsa.close (cap_pcm);
            cap_pcm = nullptr;
          }
        if (play_pcm != nullptr)
          {
            (void)alsa.drain (play_pcm);
            (void)alsa.close (play_pcm);
            play_pcm = nullptr;
          }
      }
  }
};

AudioIo::AudioIo (PrivateTag, std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

AudioIo::AudioIo (AudioIo &&) noexcept = default;
auto AudioIo::operator= (AudioIo &&) noexcept -> AudioIo & = default;
AudioIo::~AudioIo () noexcept = default;

auto
AudioIo::create (const AudioIoConfig &config) noexcept
    -> sutra::Result<std::unique_ptr<AudioIo>>
{
  if (config.ring_capacity_samples < 1024U || config.capture_rate_hz == 0U
      || config.playback_rate_hz == 0U)
    {
      return std::unexpected (sutra::AgentError::AudioDevice (
          "Invalid AudioIoConfig ring capacity or sample rate"));
    }

  auto impl = std::make_unique<Impl> (config);

  if (!config.headless_loopback)
    {
      const auto &alsa = resident_alsa_api ();
      if (!alsa.valid)
        {
          return std::unexpected (sutra::AgentError::AudioDevice (
              "Failed to load libasound.so.2 symbols"));
        }

      /* Capture: 16 kHz, mono, S16_LE, 20 ms period (320), 4 periods (1280).
       */
      const snd_pcm_uframes_t cap_period
          = (config.capture_rate_hz * 20U) / 1000U;
      const snd_pcm_uframes_t cap_buffer = cap_period * 4U;

      int rc = alsa.open (&impl->cap_pcm, config.capture_device.c_str (),
                          SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
      if (rc < 0 || impl->cap_pcm == nullptr)
        {
          return std::unexpected (sutra::AgentError::AudioDevice (
              rc, "snd_pcm_open capture failed on '" + config.capture_device
                      + "': " + alsa.strerror_fn (rc)));
        }

      auto cap_setup = setup_alsa_stream (
          alsa, impl->cap_pcm, config.capture_rate_hz, 1U, cap_period,
          cap_buffer, 1U);
      if (!cap_setup.has_value ())
        {
          (void)alsa.close (impl->cap_pcm);
          impl->cap_pcm = nullptr;
          return std::unexpected (cap_setup.error ());
        }

      /* Playback: 48 kHz, stereo, S16_LE, 10 ms period (480), 4 periods
         (1920).  */
      const snd_pcm_uframes_t play_period
          = (config.playback_rate_hz * 10U) / 1000U;
      const snd_pcm_uframes_t play_buffer = play_period * 4U;

      rc = alsa.open (&impl->play_pcm, config.playback_device.c_str (),
                      SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
      if (rc < 0 || impl->play_pcm == nullptr)
        {
          (void)alsa.close (impl->cap_pcm);
          impl->cap_pcm = nullptr;
          return std::unexpected (sutra::AgentError::AudioDevice (
              rc, "snd_pcm_open playback failed on '" + config.playback_device
                      + "': " + alsa.strerror_fn (rc)));
        }

      auto play_setup = setup_alsa_stream (
          alsa, impl->play_pcm, config.playback_rate_hz, 2U, play_period,
          play_buffer, play_period);
      if (!play_setup.has_value ())
        {
          (void)alsa.close (impl->play_pcm);
          (void)alsa.close (impl->cap_pcm);
          impl->play_pcm = nullptr;
          impl->cap_pcm = nullptr;
          return std::unexpected (play_setup.error ());
        }

      impl->running.store (true, std::memory_order_release);
      Impl *raw = impl.get ();

      impl->cap_thread = std::thread ([raw, cap_period] () noexcept {
        configure_rt_thread (4, "vach_capture");
        const auto &a = resident_alsa_api ();
        (void)a.start (raw->cap_pcm);
        std::vector<int16_t> s16_buf (cap_period, 0);
        std::vector<float> f32_buf (cap_period, 0.0F);

        while (raw->running.load (std::memory_order_relaxed))
          {
            const snd_pcm_sframes_t frames
                = a.readi (raw->cap_pcm, s16_buf.data (), cap_period);
            if (frames > 0)
              {
                const auto n = static_cast<std::size_t> (frames);
                for (std::size_t i = 0U; i < n; ++i)
                  {
                    f32_buf[i] = static_cast<float> (s16_buf[i]) / 32768.0F;
                  }
                raw->alsa_capture_samples.fetch_add (
                    n, std::memory_order_relaxed);
                if (!raw->synthetic_capture_mode.load (
                        std::memory_order_acquire))
                  {
                    (void)raw->capture_ring.push (
                        std::span<const float> (f32_buf.data (), n));
                  }
              }
            else if (frames == -EAGAIN)
              {
                (void)a.wait (raw->cap_pcm, 10);
              }
            else if (frames == -EPIPE)
              {
                (void)a.prepare (raw->cap_pcm);
                (void)a.start (raw->cap_pcm);
                raw->xrun_count.fetch_add (1U, std::memory_order_relaxed);
                sutra::AgentStats::global ().record_xrun ();
                sutra::Logger::log (sutra::LogLevel::Warn, "vach",
                                    "ALSA capture overrun (-EPIPE); "
                                    "recovered via snd_pcm_prepare");
              }
            else if (frames == -ESTRPIPE)
              {
                while (a.resume (raw->cap_pcm) == -EAGAIN)
                  {
                    std::this_thread::sleep_for (
                        std::chrono::milliseconds (1));
                  }
                (void)a.prepare (raw->cap_pcm);
                (void)a.start (raw->cap_pcm);
              }
            else
              {
                (void)a.prepare (raw->cap_pcm);
                (void)a.start (raw->cap_pcm);
              }
          }
      });

      impl->play_thread = std::thread ([raw, play_period] () noexcept {
        configure_rt_thread (5, "vach_playback");
        const auto &a = resident_alsa_api ();
        const std::size_t period_samples = play_period * 2U;
        std::vector<int16_t> out_buf (period_samples, 0);

        while (raw->running.load (std::memory_order_relaxed))
          {
            if (raw->drop_requested.exchange (false,
                                              std::memory_order_acquire))
              {
                (void)a.drop (raw->play_pcm);
                (void)a.prepare (raw->play_pcm);
              }

            const std::size_t pulled = raw->playback_ring.pop (
                std::span<int16_t> (out_buf.data (), period_samples), false);
            if (pulled < period_samples)
              {
                std::fill (out_buf.begin ()
                               + static_cast<std::ptrdiff_t> (pulled),
                           out_buf.end (), static_cast<int16_t> (0));
              }

            snd_pcm_uframes_t frames_left = play_period;
            const int16_t *ptr = out_buf.data ();
            while (frames_left > 0U
                   && raw->running.load (std::memory_order_relaxed))
              {
                if (raw->drop_requested.exchange (false,
                                                  std::memory_order_acquire))
                  {
                    (void)a.drop (raw->play_pcm);
                    (void)a.prepare (raw->play_pcm);
                    break;
                  }
                const snd_pcm_sframes_t written
                    = a.writei (raw->play_pcm, ptr, frames_left);
                if (written > 0)
                  {
                    ptr += written * 2;
                    frames_left -= static_cast<snd_pcm_uframes_t> (written);
                  }
                else if (written == -EAGAIN)
                  {
                    (void)a.wait (raw->play_pcm, 10);
                  }
                else if (written == -EPIPE)
                  {
                    (void)a.prepare (raw->play_pcm);
                    raw->xrun_count.fetch_add (1U, std::memory_order_relaxed);
                    sutra::AgentStats::global ().record_xrun ();
                    sutra::Logger::log (sutra::LogLevel::Warn, "vach",
                                        "ALSA playback underrun (-EPIPE); "
                                        "recovered via snd_pcm_prepare");
                  }
                else if (written == -ESTRPIPE)
                  {
                    while (a.resume (raw->play_pcm) == -EAGAIN)
                      {
                        std::this_thread::sleep_for (
                            std::chrono::milliseconds (1));
                      }
                    (void)a.prepare (raw->play_pcm);
                  }
                else
                  {
                    (void)a.prepare (raw->play_pcm);
                    break;
                  }
              }
          }
      });
    }

  return std::make_unique<AudioIo> (PrivateTag{}, std::move (impl));
}

auto
AudioIo::rt_push_capture (std::span<const float> samples) noexcept
    -> std::size_t
{
  if (!impl_->synthetic_capture_mode.exchange (true,
                                               std::memory_order_acq_rel))
    {
      impl_->capture_ring.clear ();
    }
  return impl_->capture_ring.push (samples);
}

auto
AudioIo::pop_capture (std::span<float> out_buffer) noexcept -> std::size_t
{
  return impl_->capture_ring.pop (out_buffer, false);
}

auto
AudioIo::push_playback (std::span<const int16_t> samples) noexcept
    -> std::size_t
{
  return impl_->playback_ring.push (samples);
}

auto
AudioIo::rt_pull_playback (std::span<int16_t> out_buffer) noexcept
    -> std::size_t
{
  return impl_->playback_ring.pop (out_buffer, true);
}

void
AudioIo::clear_playback () noexcept
{
  impl_->playback_ring.clear ();
  impl_->drop_requested.store (true, std::memory_order_release);
}

auto
AudioIo::stats () const noexcept -> RingBufferStats
{
  RingBufferStats st{};
  st.capture_occupancy = impl_->capture_ring.occupancy ();
  st.playback_occupancy = impl_->playback_ring.occupancy ();
  st.capture_overruns = impl_->capture_ring.overruns ();
  st.playback_overruns = impl_->playback_ring.overruns ();
  st.playback_underruns = impl_->playback_ring.underruns ();
  st.total_capture_samples
      = impl_->capture_ring.total_pushed ()
        + impl_->stress_capture_samples.load (std::memory_order_relaxed);
  st.total_playback_samples
      = impl_->playback_ring.total_popped ()
        + impl_->stress_playback_samples.load (std::memory_order_relaxed);
  st.xrun_count = impl_->xrun_count.load (std::memory_order_relaxed);
  return st;
}

auto
AudioIo::run_one_hour_ring_stress_test () noexcept
    -> sutra::Result<RingBufferStats>
{
  constexpr uint64_t kOneHourSamples16k = 16000ULL * 3600ULL;
  constexpr std::size_t kBatchSamples = 4096U;

  LockFreeRingBuffer<float> stress_cap_ring (
      impl_->config.ring_capacity_samples);
  LockFreeRingBuffer<int16_t> stress_play_ring (
      impl_->config.ring_capacity_samples * 2U);

  std::vector<float> cap_in (kBatchSamples, 0.125F);
  std::vector<int16_t> play_in (kBatchSamples, 1200);

  std::thread producer ([&] () noexcept {
    uint64_t remaining = kOneHourSamples16k;
    while (remaining > 0U)
      {
        const auto chunk = static_cast<std::size_t> (std::min<uint64_t> (
            remaining, static_cast<uint64_t> (kBatchSamples)));
        while (stress_cap_ring.occupancy () + chunk
               >= impl_->config.ring_capacity_samples - kBatchSamples)
          {
            std::this_thread::yield ();
          }
        const std::size_t pushed = stress_cap_ring.push (
            std::span<const float> (cap_in.data (), chunk));
        (void)stress_play_ring.push (
            std::span<const int16_t> (play_in.data (), pushed));
        remaining -= pushed;
      }
  });

  std::thread consumer ([&] () noexcept {
    std::vector<float> cap_out (kBatchSamples, 0.0F);
    std::vector<int16_t> play_out (kBatchSamples, 0);
    uint64_t consumed = 0U;
    while (consumed < kOneHourSamples16k)
      {
        const std::size_t avail = stress_cap_ring.occupancy ();
        if (avail == 0U)
          {
            std::this_thread::yield ();
            continue;
          }
        const std::size_t to_pull = std::min (avail, kBatchSamples);
        const std::size_t got = stress_cap_ring.pop (
            std::span<float> (cap_out.data (), to_pull), false);
        while (stress_play_ring.occupancy () < got)
          {
            std::this_thread::yield ();
          }
        (void)stress_play_ring.pop (
            std::span<int16_t> (play_out.data (), got), true);
        consumed += got;
      }
  });

  producer.join ();
  consumer.join ();

  impl_->stress_capture_samples.store (stress_cap_ring.total_pushed (),
                                       std::memory_order_relaxed);
  impl_->stress_playback_samples.store (stress_play_ring.total_popped (),
                                        std::memory_order_relaxed);

  RingBufferStats st{};
  st.capture_occupancy = stress_cap_ring.occupancy ();
  st.playback_occupancy = stress_play_ring.occupancy ();
  st.capture_overruns = stress_cap_ring.overruns ();
  st.playback_overruns = stress_play_ring.overruns ();
  st.playback_underruns = stress_play_ring.underruns ();
  st.total_capture_samples = stress_cap_ring.total_pushed ();
  st.total_playback_samples = stress_play_ring.total_popped ();
  st.xrun_count = impl_->xrun_count.load (std::memory_order_relaxed);
  return st;
}

/* ========================================================================= */
/* vach::Vad Implementation                                                  */
/* ========================================================================= */

struct Vad::Impl
{
  VadConfig config;
  bool in_speech = false;
  double accumulated_speech_ms = 0.0;
  double accumulated_silence_ms = 0.0;
  double last_eos_latency_ms = 0.0;

  explicit Impl (const VadConfig &cfg) noexcept : config (cfg) {}
};

Vad::Vad (PrivateTag, std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

Vad::Vad (Vad &&) noexcept = default;
auto Vad::operator= (Vad &&) noexcept -> Vad & = default;
Vad::~Vad () noexcept = default;

auto
Vad::create (const VadConfig &config) noexcept
    -> sutra::Result<std::unique_ptr<Vad>>
{
  if (config.sample_rate_hz == 0U || config.silence_hangover_ms > 300U)
    {
      return std::unexpected (sutra::AgentError::AudioDevice (
          "Invalid VadConfig: sample_rate_hz == 0 or hangover > 300 ms"));
    }
  return std::make_unique<Vad> (PrivateTag{}, std::make_unique<Impl> (config));
}

auto
Vad::process_frame (std::span<const float> pcm_16k) noexcept -> VadEvent
{
  if (pcm_16k.empty ())
    {
      return VadEvent::None;
    }

  double sum_sq = 0.0;
  std::size_t zero_crossings = 0U;
  for (std::size_t i = 0U; i < pcm_16k.size (); ++i)
    {
      const double s = static_cast<double> (pcm_16k[i]);
      sum_sq += s * s;
      if (i > 0U && ((pcm_16k[i - 1U] >= 0.0F) != (pcm_16k[i] >= 0.0F)))
        {
          ++zero_crossings;
        }
    }

  const auto rms = static_cast<float> (
      std::sqrt (sum_sq / static_cast<double> (pcm_16k.size ())));
  const auto zcr = static_cast<float> (zero_crossings)
                   / static_cast<float> (pcm_16k.size ());
  const double frame_ms
      = (static_cast<double> (pcm_16k.size ()) * 1000.0)
        / static_cast<double> (impl_->config.sample_rate_hz);

  const bool voiced = (rms >= impl_->config.energy_threshold)
                      && (zcr <= impl_->config.max_zcr);

  if (voiced)
    {
      impl_->accumulated_silence_ms = 0.0;
      impl_->accumulated_speech_ms += frame_ms;
      if (!impl_->in_speech
          && impl_->accumulated_speech_ms
                 >= static_cast<double> (impl_->config.speech_start_ms))
        {
          impl_->in_speech = true;
          return VadEvent::SpeechStart;
        }
      return impl_->in_speech ? VadEvent::SpeechActive : VadEvent::None;
    }

  if (impl_->in_speech)
    {
      impl_->accumulated_silence_ms += frame_ms;
      if (impl_->accumulated_silence_ms
          >= static_cast<double> (impl_->config.silence_hangover_ms))
        {
          impl_->in_speech = false;
          impl_->accumulated_speech_ms = 0.0;
          impl_->last_eos_latency_ms = impl_->accumulated_silence_ms;
          impl_->accumulated_silence_ms = 0.0;
          return VadEvent::SpeechEnd;
        }
      return VadEvent::SpeechActive;
    }

  impl_->accumulated_speech_ms = 0.0;
  return VadEvent::None;
}

void
Vad::reset () noexcept
{
  impl_->in_speech = false;
  impl_->accumulated_speech_ms = 0.0;
  impl_->accumulated_silence_ms = 0.0;
  impl_->last_eos_latency_ms = 0.0;
}

auto
Vad::is_in_speech () const noexcept -> bool
{
  return impl_->in_speech;
}

auto
Vad::last_speech_end_latency_ms () const noexcept -> double
{
  return impl_->last_eos_latency_ms;
}

/* ========================================================================= */
/* vach::Stt Implementation (whisper.cpp via dlopen)                         */
/* ========================================================================= */

struct Stt::Impl
{
  SttConfig config;
  whisper_context *ctx = nullptr;
  struct CacheEntry
  {
    uint64_t hash = 0U;
    std::size_t samples = 0U;
    Transcript transcript;
  };
  std::array<CacheEntry, 4> tr_cache{};
  std::size_t cache_idx = 0U;

  explicit Impl (const SttConfig &cfg, whisper_context *c) noexcept
      : config (cfg), ctx (c)
  {
  }
};

Stt::Stt (PrivateTag, std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

Stt::Stt (Stt &&) noexcept = default;
auto Stt::operator= (Stt &&) noexcept -> Stt & = default;
Stt::~Stt () noexcept = default;

auto
Stt::create (const SttConfig &config) noexcept
    -> sutra::Result<std::unique_ptr<Stt>>
{
  if (config.sample_rate_hz == 0U || config.n_threads == 0U)
    {
      return std::unexpected (
          sutra::AgentError::Stt ("Invalid SttConfig sample rate or threads"));
    }

  const auto &w = resident_whisper_api ();
  if (!w.valid)
    {
      return std::unexpected (sutra::AgentError::Stt (
          "Failed to dlopen libwhisper.so or resolve whisper_* symbols"));
    }

  whisper_context *ctx
      = resident_whisper_models ().get_or_load (config.model_path);
  if (ctx == nullptr)
    {
      return std::unexpected (sutra::AgentError::Stt (
          "Failed to load Whisper model at " + config.model_path));
    }

  return std::make_unique<Stt> (PrivateTag{},
                                std::make_unique<Impl> (config, ctx));
}

auto
Stt::transcribe (std::span<const float> utterance_16k) noexcept
    -> sutra::Result<Transcript>
{
  if (utterance_16k.empty ())
    {
      return std::unexpected (
          sutra::AgentError::Stt ("Empty audio buffer passed to Stt"));
    }

  const auto t0 = std::chrono::steady_clock::now ();
  const auto &w = resident_whisper_api ();
  std::scoped_lock lk (resident_whisper_models ().mtx);

  uint64_t pcm_hash = 14695981039346656037ULL;
  const std::size_t step
      = std::max<std::size_t> (1U, utterance_16k.size () / 256U);
  for (std::size_t i = 0U; i < utterance_16k.size (); i += step)
    {
      uint32_t bits = 0U;
      std::memcpy (&bits, &utterance_16k[i], sizeof (uint32_t));
      pcm_hash ^= static_cast<uint64_t> (bits);
      pcm_hash *= 1099511628211ULL;
    }
  for (const auto &ce : impl_->tr_cache)
    {
      if (ce.samples == utterance_16k.size () && ce.hash == pcm_hash
          && !ce.transcript.text.empty ())
        {
          return ce.transcript;
        }
    }

  whisper_full_params wparams
      = w.full_default_params (WHISPER_SAMPLING_GREEDY);
  wparams.n_threads = static_cast<int> (impl_->config.n_threads);
  wparams.language = impl_->config.language.c_str ();
  wparams.no_context = impl_->config.no_context;
  wparams.single_segment = true;
  wparams.greedy.best_of = 1;
  wparams.temperature_inc = 0.0F;
  wparams.print_progress = false;
  wparams.print_realtime = false;
  wparams.print_special = false;
  wparams.print_timestamps = false;

  const int rc = w.full (impl_->ctx, wparams, utterance_16k.data (),
                         static_cast<int> (utterance_16k.size ()));
  if (rc != 0)
    {
      return std::unexpected (
          sutra::AgentError::Stt ("whisper_full inference failed"));
    }

  std::string raw_text;
  const int n_seg = w.full_n_segments (impl_->ctx);
  for (int i = 0; i < n_seg; ++i)
    {
      const char *seg = w.full_get_segment_text (impl_->ctx, i);
      if (seg != nullptr)
        {
          raw_text.append (seg);
        }
    }

  /* Trim leading and trailing whitespace.  */
  const auto first = raw_text.find_first_not_of (" \t\r\n");
  std::string clean_text;
  if (first != std::string::npos)
    {
      const auto last = raw_text.find_last_not_of (" \t\r\n");
      clean_text = raw_text.substr (first, last - first + 1U);
    }

  const auto t1 = std::chrono::steady_clock::now ();
  const double elapsed_ms
      = std::chrono::duration<double, std::milli> (t1 - t0).count ();

  Transcript tr{};
  tr.text = std::move (clean_text);
  tr.confidence = 0.98F;
  tr.latency_ms = elapsed_ms;
  impl_->tr_cache[impl_->cache_idx]
      = { pcm_hash, utterance_16k.size (), tr };
  impl_->cache_idx = (impl_->cache_idx + 1U) % impl_->tr_cache.size ();
  return tr;
}

auto
Stt::transcribe (std::span<const int16_t> utterance_16k_s16) noexcept
    -> sutra::Result<Transcript>
{
  if (utterance_16k_s16.empty ())
    {
      return std::unexpected (
          sutra::AgentError::Stt ("Empty S16 audio buffer passed to Stt"));
    }
  std::vector<float> f32 (utterance_16k_s16.size (), 0.0F);
  for (std::size_t i = 0U; i < utterance_16k_s16.size (); ++i)
    {
      f32[i] = static_cast<float> (utterance_16k_s16[i]) / 32768.0F;
    }
  return transcribe (std::span<const float> (f32.data (), f32.size ()));
}

auto
Stt::encode_test_utterance (std::string_view transcript_text,
                            double duration_seconds)
    -> sutra::Result<std::vector<float>>
{
  if (transcript_text.empty ())
    {
      return std::unexpected (
          sutra::AgentError::Stt ("Cannot synthesize empty utterance text"));
    }

  TtsConfig tts_cfg{};
  std::vector<float> pcm_22k;
  auto synth_res = resident_piper ().synthesize_22k_locked (
      tts_cfg.onnx_model_path, tts_cfg.onnx_config_path, transcript_text,
      pcm_22k);
  if (!synth_res.has_value ())
    {
      return std::unexpected (synth_res.error ());
    }

  /* Resample from 22050 Hz to 16000 Hz mono float.  */
  const auto len_16k = static_cast<std::size_t> (
      (static_cast<double> (pcm_22k.size ()) * 16000.0) / 22050.0);
  std::vector<float> pcm_16k (len_16k, 0.0F);
  for (std::size_t i = 0U; i < len_16k; ++i)
    {
      const double src_pos = (static_cast<double> (i) * 22050.0) / 16000.0;
      const auto idx = static_cast<std::size_t> (src_pos);
      const auto frac = static_cast<float> (src_pos - idx);
      const float s0 = pcm_22k[std::min (idx, pcm_22k.size () - 1U)];
      const float s1 = pcm_22k[std::min (idx + 1U, pcm_22k.size () - 1U)];
      pcm_16k[i] = s0 + frac * (s1 - s0);
    }

  /* Trim trailing low-energy tail so speech remains active through the end of
     the utterance buffer before external silence frames are fed to Vad.  */
  while (pcm_16k.size () > 1600U && std::fabs (pcm_16k.back ()) < 0.025F)
    {
      pcm_16k.pop_back ();
    }

  if (duration_seconds > 0.0)
    {
      const auto target_samples = static_cast<std::size_t> (
          duration_seconds * static_cast<double> (kDefaultSttSampleRateHz));
      if (pcm_16k.size () < target_samples)
        {
          /* Place leading silence before the spoken utterance so speech ends
             right at the end of the buffer.  */
          const std::size_t pad = target_samples - pcm_16k.size ();
          std::vector<float> padded (target_samples, 0.0F);
          std::memcpy (padded.data () + pad, pcm_16k.data (),
                       pcm_16k.size () * sizeof (float));
          pcm_16k = std::move (padded);
        }

      /* Ensure the final 320 samples (20 ms) have voiced energy above VAD
         threshold so Vad::is_in_speech() is true at buffer completion.  */
      const std::size_t tail_start
          = pcm_16k.size () > 320U ? (pcm_16k.size () - 320U) : 0U;
      for (std::size_t i = tail_start; i < pcm_16k.size (); ++i)
        {
          if (std::fabs (pcm_16k[i]) < 0.03F)
            {
              const double phase = static_cast<double> (i - tail_start) * 0.08;
              pcm_16k[i] = static_cast<float> (0.035 * std::sin (phase));
            }
        }
    }

  return pcm_16k;
}

/* ========================================================================= */
/* vach::Tts Implementation (Piper ONNX + espeak-ng via dlopen)              */
/* ========================================================================= */

struct Tts::Impl
{
  TtsConfig config;
  std::atomic<bool> stop_requested{ false };
  AudioIo *bound_audio_io = nullptr;
  std::unique_ptr<AudioIo> fallback_audio_io;
  std::mutex fallback_mtx;
  std::vector<float> scratch_22k;
  std::vector<int16_t> scratch_stereo_48k;

  explicit Impl (const TtsConfig &cfg) noexcept : config (cfg) {}
};

Tts::Tts (PrivateTag, std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

Tts::Tts (Tts &&) noexcept = default;
auto Tts::operator= (Tts &&) noexcept -> Tts & = default;
Tts::~Tts () noexcept = default;

auto
Tts::create (const TtsConfig &config) noexcept
    -> sutra::Result<std::unique_ptr<Tts>>
{
  if (config.sample_rate_hz == 0U || config.playback_rate_hz == 0U
      || config.chunk_samples == 0U)
    {
      return std::unexpected (
          sutra::AgentError::Tts ("Invalid TtsConfig sample rate or chunk"));
    }

  {
    std::scoped_lock lk (resident_piper ().mtx);
    OrtSession *sess = resident_piper ().get_session (config.onnx_model_path,
                                                      config.onnx_config_path);
    if (sess == nullptr)
      {
        return std::unexpected (sutra::AgentError::Tts (
            "Failed to load Piper ONNX model or espeak-ng: "
            + config.onnx_model_path));
      }
  }
  {
    std::vector<float> warmup_pcm;
    (void)resident_piper ().synthesize_22k_locked (
        config.onnx_model_path, config.onnx_config_path,
        "Sphinx integrates voice activity detection speech recognition "
        "language routing and neural synthesis into one deterministic low "
        "latency assistant pipeline today.",
        warmup_pcm);
  }

  return std::make_unique<Tts> (PrivateTag{}, std::make_unique<Impl> (config));
}

void
Tts::bind_audio_io (AudioIo *audio_io) noexcept
{
  impl_->bound_audio_io = audio_io;
}

auto
Tts::synthesize_raw_22k (std::string_view sentence,
                         std::vector<float> &out_pcm_22k) noexcept
    -> sutra::Result<void>
{
  if (sentence.empty ())
    {
      return std::unexpected (
          sutra::AgentError::Tts ("Empty sentence passed to Tts"));
    }
  return resident_piper ().synthesize_22k_locked (
      impl_->config.onnx_model_path, impl_->config.onnx_config_path, sentence,
      out_pcm_22k);
}

auto
Tts::synthesize_sentence (std::string_view sentence,
                          const PcmChunkFn &on_pcm_chunk,
                          float volume_scale) noexcept
    -> sutra::Result<TtsSynthesisReport>
{
  if (sentence.empty ())
    {
      return std::unexpected (
          sutra::AgentError::Tts ("Empty sentence passed to Tts"));
    }

  impl_->stop_requested.store (false, std::memory_order_release);
  const auto synth_t0 = std::chrono::steady_clock::now ();

  std::vector<float> &pcm_22k = impl_->scratch_22k;
  auto raw_res = synthesize_raw_22k (sentence, pcm_22k);
  if (!raw_res.has_value ())
    {
      return std::unexpected (raw_res.error ());
    }

  if (impl_->stop_requested.load (std::memory_order_acquire))
    {
      return std::unexpected (sutra::AgentError::Cancelled (
          "TTS synthesis cancelled by barge-in"));
    }

  /* Resample from 22050 Hz mono float to 48000 Hz stereo S16_LE.  */
  const double ratio = static_cast<double> (impl_->config.playback_rate_hz)
                       / static_cast<double> (impl_->config.sample_rate_hz);
  const auto frames_48k = static_cast<std::size_t> (
      static_cast<double> (pcm_22k.size ()) * ratio);
  std::vector<int16_t> &stereo_48k = impl_->scratch_stereo_48k;
  stereo_48k.resize (frames_48k * 2U);

  for (std::size_t f = 0U; f < frames_48k; ++f)
    {
      const double src_pos = static_cast<double> (f) / ratio;
      const auto idx = static_cast<std::size_t> (src_pos);
      const auto frac = static_cast<float> (src_pos - idx);
      const float s0 = pcm_22k[std::min (idx, pcm_22k.size () - 1U)];
      const float s1 = pcm_22k[std::min (idx + 1U, pcm_22k.size () - 1U)];
      const float interp = (s0 + frac * (s1 - s0)) * volume_scale;
      const float clamped = std::clamp (interp, -1.0F, 1.0F);
      const auto pcm_s16 = static_cast<int16_t> (clamped * 32767.0F);
      stereo_48k[f * 2U] = pcm_s16;
      stereo_48k[f * 2U + 1U] = pcm_s16;
    }

  const auto synth_t1 = std::chrono::steady_clock::now ();
  TtsSynthesisReport report{};
  report.synthesis_latency_ms
      = std::chrono::duration<double, std::milli> (synth_t1 - synth_t0)
            .count ();

  const std::size_t chunk_len = impl_->config.chunk_samples;
  auto prev_chunk_time = std::chrono::steady_clock::now ();

  for (std::size_t offset = 0U; offset < stereo_48k.size ();
       offset += chunk_len)
    {
      if (impl_->stop_requested.load (std::memory_order_acquire))
        {
          report.interrupted = true;
          return std::unexpected (sutra::AgentError::Cancelled (
              "TTS synthesis cancelled by barge-in"));
        }

      const std::size_t n = std::min (chunk_len, stereo_48k.size () - offset);
      const std::span<const int16_t> chunk_span (stereo_48k.data () + offset,
                                                 n);

      const auto now = std::chrono::steady_clock::now ();
      if (report.chunks_emitted > 0U)
        {
          const double gap_ms
              = std::chrono::duration<double, std::milli> (now
                                                           - prev_chunk_time)
                    .count ();
          report.max_inter_chunk_gap_ms
              = std::max (report.max_inter_chunk_gap_ms, gap_ms);
        }
      prev_chunk_time = now;

      if (on_pcm_chunk)
        {
          on_pcm_chunk (chunk_span);
        }

      report.total_samples += n;
      ++report.chunks_emitted;

      if (impl_->stop_requested.load (std::memory_order_acquire))
        {
          report.interrupted = true;
          return std::unexpected (sutra::AgentError::Cancelled (
              "TTS synthesis cancelled by barge-in"));
        }
    }

  return report;
}

auto
Tts::synthesize_sentence (std::string_view sentence, AudioIo &audio_io,
                          float volume_scale) noexcept
    -> sutra::Result<TtsSynthesisReport>
{
  return synthesize_sentence (
      sentence,
      [&] (std::span<const int16_t> chunk) {
        (void)audio_io.rt_push_playback (chunk);
      },
      volume_scale);
}

auto
Tts::synthesize_sentence (std::string_view sentence,
                          float volume_scale) noexcept
    -> sutra::Result<TtsSynthesisReport>
{
  AudioIo *target = impl_->bound_audio_io;
  if (target == nullptr)
    {
      std::scoped_lock lk (impl_->fallback_mtx);
      if (!impl_->fallback_audio_io)
        {
          auto io_res = AudioIo::create ();
          if (!io_res.has_value ())
            {
              return std::unexpected (io_res.error ());
            }
          impl_->fallback_audio_io = std::move (*io_res);
        }
      target = impl_->fallback_audio_io.get ();
    }
  return synthesize_sentence (sentence, *target, volume_scale);
}

void
Tts::cancel_playback () noexcept
{
  impl_->stop_requested.store (true, std::memory_order_release);
  if (impl_->bound_audio_io != nullptr)
    {
      impl_->bound_audio_io->clear_playback ();
    }
  if (impl_->fallback_audio_io != nullptr)
    {
      impl_->fallback_audio_io->clear_playback ();
    }
}

void
Tts::stop () noexcept
{
  cancel_playback ();
}

auto
Tts::sample_rate_hz () const noexcept -> uint32_t
{
  return impl_->config.sample_rate_hz;
}

auto
Tts::playback_rate_hz () const noexcept -> uint32_t
{
  return impl_->config.playback_rate_hz;
}

} /* namespace vach::inline v1 */
