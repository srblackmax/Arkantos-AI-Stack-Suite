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

#include <yantra/observability.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace yantra::inline v1
{

namespace
{

class StderrJsonSink final : public LogSink
{
public:
  void
  write (LogLevel level, std::string_view subsystem, std::string_view message,
         uint64_t timestamp_ns) noexcept override
  {
    std::string_view lvl_str = "INFO";
    switch (level)
      {
      case LogLevel::Trace:
        lvl_str = "TRACE";
        break;
      case LogLevel::Debug:
        lvl_str = "DEBUG";
        break;
      case LogLevel::Info:
        lvl_str = "INFO";
        break;
      case LogLevel::Warn:
        lvl_str = "WARN";
        break;
      case LogLevel::Error:
        lvl_str = "ERROR";
        break;
      }

    std::lock_guard<std::mutex> lock (mu_);
    std::cerr << "{\"ts_ns\":" << timestamp_ns << ",\"level\":\"" << lvl_str
              << "\",\"subsystem\":\"" << subsystem << "\",\"msg\":\"";
    for (const char c : message)
      {
        if (c == '"' || c == '\\')
          {
            std::cerr << '\\' << c;
          }
        else if (c == '\n')
          {
            std::cerr << "\\n";
          }
        else
          {
            std::cerr << c;
          }
      }
    std::cerr << "\"}\n";
  }

private:
  std::mutex mu_;
};

struct LoggerState
{
  std::atomic<LogLevel> min_level{ LogLevel::Warn };
  std::mutex sink_mu;
  std::shared_ptr<LogSink> custom_sink;
  StderrJsonSink default_sink;
};

auto
logger_state () noexcept -> LoggerState &
{
  static LoggerState state;
  return state;
}

std::atomic<ProfilerHook *> g_profiler_hook{ nullptr };

auto
update_max (std::atomic<uint64_t> &target, uint64_t val) noexcept -> void
{
  uint64_t prev = target.load (std::memory_order_relaxed);
  while (prev < val
         && !target.compare_exchange_weak (prev, val,
                                           std::memory_order_relaxed))
    {
    }
}

constexpr std::array<uint32_t, 64> kSha256K = {
  0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL,
  0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL, 0xd807aa98UL, 0x12835b01UL,
  0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL,
  0xc19bf174UL, 0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
  0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL, 0x983e5152UL,
  0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL,
  0x06ca6351UL, 0x14292967UL, 0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL,
  0x53380d13UL, 0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
  0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL,
  0xd6990624UL, 0xf40e3585UL, 0x106aa070UL, 0x19a4c116UL, 0x1e376c08UL,
  0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL,
  0x682e6ff3UL, 0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
  0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
};

auto
rotr32 (uint32_t x, unsigned int n) noexcept -> uint32_t
{
  return (x >> n) | (x << (32U - n));
}

void
sha256_transform_scalar (std::array<uint32_t, 8> &state, const uint8_t *data,
                         std::size_t num_blocks) noexcept
{
  for (std::size_t blk = 0U; blk < num_blocks; ++blk)
    {
      const uint8_t *ptr = data + blk * 64U;
      std::array<uint32_t, 64> w{};
      for (std::size_t i = 0U; i < 16U; ++i)
        {
          w[i] = (static_cast<uint32_t> (ptr[i * 4U]) << 24U)
                 | (static_cast<uint32_t> (ptr[i * 4U + 1U]) << 16U)
                 | (static_cast<uint32_t> (ptr[i * 4U + 2U]) << 8U)
                 | static_cast<uint32_t> (ptr[i * 4U + 3U]);
        }
      for (std::size_t i = 16U; i < 64U; ++i)
        {
          const uint32_t s0 = rotr32 (w[i - 15U], 7U)
                              ^ rotr32 (w[i - 15U], 18U) ^ (w[i - 15U] >> 3U);
          const uint32_t s1 = rotr32 (w[i - 2U], 17U) ^ rotr32 (w[i - 2U], 19U)
                              ^ (w[i - 2U] >> 10U);
          w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
        }

      uint32_t a = state[0];
      uint32_t b = state[1];
      uint32_t c = state[2];
      uint32_t d = state[3];
      uint32_t e = state[4];
      uint32_t f = state[5];
      uint32_t g = state[6];
      uint32_t h = state[7];

      for (std::size_t i = 0U; i < 64U; ++i)
        {
          const uint32_t s1
              = rotr32 (e, 6U) ^ rotr32 (e, 11U) ^ rotr32 (e, 25U);
          const uint32_t ch = (e & f) ^ ((~e) & g);
          const uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
          const uint32_t s0
              = rotr32 (a, 2U) ^ rotr32 (a, 13U) ^ rotr32 (a, 22U);
          const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
          const uint32_t temp2 = s0 + maj;

          h = g;
          g = f;
          f = e;
          e = d + temp1;
          d = c;
          c = b;
          b = a;
          a = temp1 + temp2;
        }

      state[0] += a;
      state[1] += b;
      state[2] += c;
      state[3] += d;
      state[4] += e;
      state[5] += f;
      state[6] += g;
      state[7] += h;
    }
}

#if defined(__x86_64__)
__attribute__ ((target ("sha,sse4.1,ssse3"))) void
sha256_transform_shani (std::array<uint32_t, 8> &state, const uint8_t *data,
                        std::size_t num_blocks) noexcept
{
  const __m128i shuf_mask
      = _mm_set_epi64x (0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

  __m128i tmp
      = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (&state[0]));
  __m128i st1
      = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (&state[4]));

  tmp = _mm_shuffle_epi32 (tmp, 0xB1);
  st1 = _mm_shuffle_epi32 (st1, 0x1B);
  __m128i st0 = _mm_alignr_epi8 (tmp, st1, 8);
  st1 = _mm_blend_epi16 (st1, tmp, 0xF0);

  for (std::size_t blk = 0U; blk < num_blocks; ++blk)
    {
      const auto *ptr = reinterpret_cast<const __m128i *> (data + blk * 64U);
      const __m128i abef_save = st0;
      const __m128i cdgh_save = st1;

      __m128i msg0 = _mm_shuffle_epi8 (_mm_loadu_si128 (ptr + 0), shuf_mask);
      __m128i msg1 = _mm_shuffle_epi8 (_mm_loadu_si128 (ptr + 1), shuf_mask);
      __m128i msg2 = _mm_shuffle_epi8 (_mm_loadu_si128 (ptr + 2), shuf_mask);
      __m128i msg3 = _mm_shuffle_epi8 (_mm_loadu_si128 (ptr + 3), shuf_mask);

      __m128i k0 = _mm_loadu_si128 (
          reinterpret_cast<const __m128i *> (&kSha256K[0]));
      __m128i msg = _mm_add_epi32 (msg0, k0);
      st1 = _mm_sha256rnds2_epu32 (st1, st0, msg);
      msg = _mm_shuffle_epi32 (msg, 0x0E);
      st0 = _mm_sha256rnds2_epu32 (st0, st1, msg);

      for (std::size_t iter = 1U; iter < 16U; ++iter)
        {
          const __m128i kn = _mm_loadu_si128 (
              reinterpret_cast<const __m128i *> (&kSha256K[iter * 4U]));
          const std::size_t r = iter & 3U;
          if (iter < 13U)
            {
              if (r == 1U)
                {
                  msg0 = _mm_sha256msg1_epu32 (msg0, msg1);
                }
              else if (r == 2U)
                {
                  msg1 = _mm_sha256msg1_epu32 (msg1, msg2);
                }
              else if (r == 3U)
                {
                  msg2 = _mm_sha256msg1_epu32 (msg2, msg3);
                }
              else
                {
                  msg3 = _mm_sha256msg1_epu32 (msg3, msg0);
                }
            }
          if (iter >= 3U)
            {
              if (r == 3U)
                {
                  msg0 = _mm_add_epi32 (msg0, _mm_alignr_epi8 (msg3, msg2, 4));
                  msg0 = _mm_sha256msg2_epu32 (msg0, msg3);
                }
              else if (r == 0U)
                {
                  msg1 = _mm_add_epi32 (msg1, _mm_alignr_epi8 (msg0, msg3, 4));
                  msg1 = _mm_sha256msg2_epu32 (msg1, msg0);
                }
              else if (r == 1U)
                {
                  msg2 = _mm_add_epi32 (msg2, _mm_alignr_epi8 (msg1, msg0, 4));
                  msg2 = _mm_sha256msg2_epu32 (msg2, msg1);
                }
              else
                {
                  msg3 = _mm_add_epi32 (msg3, _mm_alignr_epi8 (msg2, msg1, 4));
                  msg3 = _mm_sha256msg2_epu32 (msg3, msg2);
                }
            }

          const __m128i cur_m
              = (r == 0U) ? msg0 : (r == 1U) ? msg1 : (r == 2U) ? msg2 : msg3;
          msg = _mm_add_epi32 (cur_m, kn);
          st1 = _mm_sha256rnds2_epu32 (st1, st0, msg);
          msg = _mm_shuffle_epi32 (msg, 0x0E);
          st0 = _mm_sha256rnds2_epu32 (st0, st1, msg);
        }

      st0 = _mm_add_epi32 (st0, abef_save);
      st1 = _mm_add_epi32 (st1, cdgh_save);
    }

  tmp = _mm_shuffle_epi32 (st0, 0x1B);
  st1 = _mm_shuffle_epi32 (st1, 0xB1);
  st0 = _mm_blend_epi16 (tmp, st1, 0xF0);
  st1 = _mm_alignr_epi8 (st1, tmp, 8);

  _mm_storeu_si128 (reinterpret_cast<__m128i *> (&state[0]), st0);
  _mm_storeu_si128 (reinterpret_cast<__m128i *> (&state[4]), st1);
}
#endif

constexpr std::array<uint32_t, 8> kBlake3IV = {
  0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
  0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

constexpr std::array<uint8_t, 16> kBlake3MsgPerm
    = { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 };

void
blake3_g (std::array<uint32_t, 16> &v, std::size_t a, std::size_t b,
          std::size_t c, std::size_t d, uint32_t mx, uint32_t my) noexcept
{
  v[a] = v[a] + v[b] + mx;
  v[d] = rotr32 (v[d] ^ v[a], 16U);
  v[c] = v[c] + v[d];
  v[b] = rotr32 (v[b] ^ v[c], 12U);
  v[a] = v[a] + v[b] + my;
  v[d] = rotr32 (v[d] ^ v[a], 8U);
  v[c] = v[c] + v[d];
  v[b] = rotr32 (v[b] ^ v[c], 7U);
}

void
blake3_round (std::array<uint32_t, 16> &v,
              const std::array<uint32_t, 16> &m) noexcept
{
  blake3_g (v, 0, 4, 8, 12, m[0], m[1]);
  blake3_g (v, 1, 5, 9, 13, m[2], m[3]);
  blake3_g (v, 2, 6, 10, 14, m[4], m[5]);
  blake3_g (v, 3, 7, 11, 15, m[6], m[7]);
  blake3_g (v, 0, 5, 10, 15, m[8], m[9]);
  blake3_g (v, 1, 6, 11, 12, m[10], m[11]);
  blake3_g (v, 2, 7, 8, 13, m[12], m[13]);
  blake3_g (v, 3, 4, 9, 14, m[14], m[15]);
}

auto
blake3_compress (const std::array<uint32_t, 8> &cv,
                 const std::array<uint32_t, 16> &block_words, uint64_t counter,
                 uint32_t block_len, uint32_t flags) noexcept
    -> std::array<uint32_t, 8>
{
  std::array<uint32_t, 16> v = { cv[0],
                                 cv[1],
                                 cv[2],
                                 cv[3],
                                 cv[4],
                                 cv[5],
                                 cv[6],
                                 cv[7],
                                 kBlake3IV[0],
                                 kBlake3IV[1],
                                 kBlake3IV[2],
                                 kBlake3IV[3],
                                 static_cast<uint32_t> (counter),
                                 static_cast<uint32_t> (counter >> 32U),
                                 block_len,
                                 flags };
  std::array<uint32_t, 16> m = block_words;
  for (int r = 0; r < 7; ++r)
    {
      blake3_round (v, m);
      if (r < 6)
        {
          std::array<uint32_t, 16> permuted{};
          for (std::size_t i = 0U; i < 16U; ++i)
            {
              permuted[i] = m[kBlake3MsgPerm[i]];
            }
          m = permuted;
        }
    }
  std::array<uint32_t, 8> out{};
  for (std::size_t i = 0U; i < 8U; ++i)
    {
      out[i] = v[i] ^ v[i + 8U];
    }
  return out;
}

auto
blake3_hash_segment (const uint8_t *data, std::size_t len,
                     uint64_t segment_idx) noexcept -> std::array<uint32_t, 8>
{
  std::array<uint32_t, 8> cv = kBlake3IV;
  cv[0] ^= static_cast<uint32_t> (segment_idx);
  cv[1] ^= static_cast<uint32_t> (segment_idx >> 32U);

  std::size_t offset = 0U;
  uint64_t blk_counter = 0U;
  while (offset + 64U <= len)
    {
      std::array<uint32_t, 16> words{};
      std::memcpy (words.data (), data + offset, 64U);
      cv = blake3_compress (cv, words, blk_counter, 64U, 0U);
      offset += 64U;
      ++blk_counter;
    }
  if (offset < len || len == 0U)
    {
      std::array<uint32_t, 16> words{};
      const std::size_t rem = len - offset;
      if (rem > 0U)
        {
          std::memcpy (words.data (), data + offset, rem);
        }
      cv = blake3_compress (cv, words, blk_counter,
                            static_cast<uint32_t> (rem), 1U);
    }
  return cv;
}

auto
to_hex_string (const uint8_t *bytes, std::size_t count) noexcept -> std::string
{
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize (count * 2U);
  for (std::size_t i = 0U; i < count; ++i)
    {
      out[i * 2U] = kHex[(bytes[i] >> 4U) & 0x0FU];
      out[i * 2U + 1U] = kHex[bytes[i] & 0x0FU];
    }
  return out;
}

} /* namespace */

void
Logger::set_level (LogLevel level) noexcept
{
  logger_state ().min_level.store (level, std::memory_order_relaxed);
}

auto
Logger::level () noexcept -> LogLevel
{
  return logger_state ().min_level.load (std::memory_order_relaxed);
}

void
Logger::set_sink (std::shared_ptr<LogSink> sink) noexcept
{
  auto &st = logger_state ();
  std::lock_guard<std::mutex> lock (st.sink_mu);
  st.custom_sink = std::move (sink);
}

void
Logger::log (LogLevel level, std::string_view subsystem,
             std::string_view message) noexcept
{
  auto &st = logger_state ();
  if (static_cast<uint8_t> (level)
      < static_cast<uint8_t> (st.min_level.load (std::memory_order_relaxed)))
    {
      return;
    }
  const uint64_t now = monotonic_time_ns ();
  std::shared_ptr<LogSink> sink_copy;
  {
    std::lock_guard<std::mutex> lock (st.sink_mu);
    sink_copy = st.custom_sink;
  }
  if (sink_copy)
    {
      sink_copy->write (level, subsystem, message, now);
    }
  else
    {
      st.default_sink.write (level, subsystem, message, now);
    }
}

void
TrackingStats::record_metrics_logged (uint64_t count, uint64_t elapsed_ns,
                                      uint64_t wal_bytes) noexcept
{
#if YANTRA_ENABLE_STATS
  metrics_logged_total_.fetch_add (count, std::memory_order_relaxed);
  logging_time_ns_.fetch_add (elapsed_ns, std::memory_order_relaxed);
  wal_size_bytes_.store (wal_bytes, std::memory_order_relaxed);
#else
  (void)count;
  (void)elapsed_ns;
  (void)wal_bytes;
#endif
}

void
TrackingStats::record_query_latency (uint64_t latency_ns) noexcept
{
#if YANTRA_ENABLE_STATS
  last_query_ns_.store (latency_ns, std::memory_order_relaxed);
  update_max (max_query_ns_, latency_ns);
#else
  (void)latency_ns;
#endif
}

void
TrackingStats::record_compaction () noexcept
{
#if YANTRA_ENABLE_STATS
  compactions_completed_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

auto
TrackingStats::snapshot () const noexcept -> TrackingStatsSnapshot
{
  TrackingStatsSnapshot s;
  s.metrics_logged_total
      = metrics_logged_total_.load (std::memory_order_relaxed);
  const uint64_t ns = logging_time_ns_.load (std::memory_order_relaxed);
  if (ns > 0U)
    {
      s.metrics_logged_per_sec = static_cast<double> (s.metrics_logged_total)
                                 * 1.0e9 / static_cast<double> (ns);
    }
  s.query_p50_ns = last_query_ns_.load (std::memory_order_relaxed);
  s.query_p99_ns = max_query_ns_.load (std::memory_order_relaxed);
  s.wal_size_bytes = wal_size_bytes_.load (std::memory_order_relaxed);
  s.compactions_completed
      = compactions_completed_.load (std::memory_order_relaxed);
  return s;
}

void
TrackingStats::reset () noexcept
{
  metrics_logged_total_.store (0U, std::memory_order_relaxed);
  logging_time_ns_.store (0U, std::memory_order_relaxed);
  last_query_ns_.store (0U, std::memory_order_relaxed);
  max_query_ns_.store (0U, std::memory_order_relaxed);
  wal_size_bytes_.store (0U, std::memory_order_relaxed);
  compactions_completed_.store (0U, std::memory_order_relaxed);
}

auto
TrackingStats::global () noexcept -> TrackingStats &
{
  static TrackingStats inst;
  return inst;
}

void
RegistryStats::record_registration (uint64_t duration_ns) noexcept
{
#if YANTRA_ENABLE_STATS
  registrations_total_.fetch_add (1U, std::memory_order_relaxed);
  registration_time_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
#else
  (void)duration_ns;
#endif
}

void
RegistryStats::record_stage_transition () noexcept
{
#if YANTRA_ENABLE_STATS
  stage_transitions_total_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

void
RegistryStats::record_lineage_query (uint64_t latency_ns) noexcept
{
#if YANTRA_ENABLE_STATS
  last_lineage_ns_.store (latency_ns, std::memory_order_relaxed);
  update_max (max_lineage_ns_, latency_ns);
#else
  (void)latency_ns;
#endif
}

auto
RegistryStats::snapshot () const noexcept -> RegistryStatsSnapshot
{
  RegistryStatsSnapshot s;
  s.registrations_total
      = registrations_total_.load (std::memory_order_relaxed);
  const uint64_t ns = registration_time_ns_.load (std::memory_order_relaxed);
  if (ns > 0U)
    {
      s.registrations_per_sec = static_cast<double> (s.registrations_total)
                                * 1.0e9 / static_cast<double> (ns);
    }
  s.stage_transitions_total
      = stage_transitions_total_.load (std::memory_order_relaxed);
  s.lineage_query_p50_ns = last_lineage_ns_.load (std::memory_order_relaxed);
  s.lineage_query_p99_ns = max_lineage_ns_.load (std::memory_order_relaxed);
  return s;
}

void
RegistryStats::reset () noexcept
{
  registrations_total_.store (0U, std::memory_order_relaxed);
  registration_time_ns_.store (0U, std::memory_order_relaxed);
  stage_transitions_total_.store (0U, std::memory_order_relaxed);
  last_lineage_ns_.store (0U, std::memory_order_relaxed);
  max_lineage_ns_.store (0U, std::memory_order_relaxed);
}

auto
RegistryStats::global () noexcept -> RegistryStats &
{
  static RegistryStats inst;
  return inst;
}

void
DagStats::record_dag_run (uint64_t task_count, uint64_t wall_ns,
                          uint64_t schedule_ns) noexcept
{
#if YANTRA_ENABLE_STATS
  tasks_completed_total_.fetch_add (task_count, std::memory_order_relaxed);
  total_wall_ns_.fetch_add (wall_ns, std::memory_order_relaxed);
  last_schedule_ns_.store (schedule_ns, std::memory_order_relaxed);
#else
  (void)task_count;
  (void)wall_ns;
  (void)schedule_ns;
#endif
}

void
DagStats::record_retry () noexcept
{
#if YANTRA_ENABLE_STATS
  retry_count_total_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

void
DagStats::record_steal () noexcept
{
#if YANTRA_ENABLE_STATS
  steal_count_total_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

void
DagStats::record_queue_depth (uint64_t depth) noexcept
{
#if YANTRA_ENABLE_STATS
  update_max (max_queue_depth_, depth);
#else
  (void)depth;
#endif
}

auto
DagStats::snapshot () const noexcept -> DagStatsSnapshot
{
  DagStatsSnapshot s;
  s.tasks_completed_total
      = tasks_completed_total_.load (std::memory_order_relaxed);
  const uint64_t ns = total_wall_ns_.load (std::memory_order_relaxed);
  if (ns > 0U)
    {
      s.tasks_per_sec = static_cast<double> (s.tasks_completed_total) * 1.0e9
                        / static_cast<double> (ns);
    }
  s.scheduling_latency_ns = last_schedule_ns_.load (std::memory_order_relaxed);
  s.max_worker_queue_depth = max_queue_depth_.load (std::memory_order_relaxed);
  s.retry_count_total = retry_count_total_.load (std::memory_order_relaxed);
  s.steal_count_total = steal_count_total_.load (std::memory_order_relaxed);
  return s;
}

void
DagStats::reset () noexcept
{
  tasks_completed_total_.store (0U, std::memory_order_relaxed);
  total_wall_ns_.store (0U, std::memory_order_relaxed);
  last_schedule_ns_.store (0U, std::memory_order_relaxed);
  max_queue_depth_.store (0U, std::memory_order_relaxed);
  retry_count_total_.store (0U, std::memory_order_relaxed);
  steal_count_total_.store (0U, std::memory_order_relaxed);
}

auto
DagStats::global () noexcept -> DagStats &
{
  static DagStats inst;
  return inst;
}

void
IndexStats::record_search (uint64_t latency_ns) noexcept
{
#if YANTRA_ENABLE_STATS
  queries_total_.fetch_add (1U, std::memory_order_relaxed);
  last_search_ns_.store (latency_ns, std::memory_order_relaxed);
  update_max (max_search_ns_, latency_ns);
#else
  (void)latency_ns;
#endif
}

void
IndexStats::set_recall_estimate (double recall) noexcept
{
#if YANTRA_ENABLE_STATS
  const auto fixed = static_cast<uint64_t> (recall * 1000000.0);
  recall_fixed_pt_.store (fixed, std::memory_order_relaxed);
#else
  (void)recall;
#endif
}

void
IndexStats::set_memory_usage (uint64_t bytes) noexcept
{
#if YANTRA_ENABLE_STATS
  memory_usage_bytes_.store (bytes, std::memory_order_relaxed);
#else
  (void)bytes;
#endif
}

auto
IndexStats::snapshot () const noexcept -> IndexStatsSnapshot
{
  IndexStatsSnapshot s;
  s.queries_total = queries_total_.load (std::memory_order_relaxed);
  s.search_p50_ns = last_search_ns_.load (std::memory_order_relaxed);
  s.search_p99_ns = max_search_ns_.load (std::memory_order_relaxed);
  s.recall_estimate
      = static_cast<double> (recall_fixed_pt_.load (std::memory_order_relaxed))
        / 1000000.0;
  s.memory_usage_bytes = memory_usage_bytes_.load (std::memory_order_relaxed);
  return s;
}

void
IndexStats::reset () noexcept
{
  queries_total_.store (0U, std::memory_order_relaxed);
  last_search_ns_.store (0U, std::memory_order_relaxed);
  max_search_ns_.store (0U, std::memory_order_relaxed);
  recall_fixed_pt_.store (1000000U, std::memory_order_relaxed);
  memory_usage_bytes_.store (0U, std::memory_order_relaxed);
}

auto
IndexStats::global () noexcept -> IndexStats &
{
  static IndexStats inst;
  return inst;
}

void
FairnessStats::record_audit (double violation_mag, double threshold_adj,
                             uint64_t duration_ns) noexcept
{
#if YANTRA_ENABLE_STATS
  audits_performed_.fetch_add (1U, std::memory_order_relaxed);
  violation_sum_fixed_.fetch_add (
      static_cast<uint64_t> (violation_mag * 1000000.0),
      std::memory_order_relaxed);
  adjustment_sum_fixed_.fetch_add (
      static_cast<uint64_t> (threshold_adj * 1000000.0),
      std::memory_order_relaxed);
  last_duration_ns_.store (duration_ns, std::memory_order_relaxed);
#else
  (void)violation_mag;
  (void)threshold_adj;
  (void)duration_ns;
#endif
}

auto
FairnessStats::snapshot () const noexcept -> FairnessStatsSnapshot
{
  FairnessStatsSnapshot s;
  s.audits_performed = audits_performed_.load (std::memory_order_relaxed);
  if (s.audits_performed > 0U)
    {
      s.avg_constraint_violation
          = static_cast<double> (
                violation_sum_fixed_.load (std::memory_order_relaxed))
            / (1000000.0 * static_cast<double> (s.audits_performed));
      s.avg_threshold_adjustment
          = static_cast<double> (
                adjustment_sum_fixed_.load (std::memory_order_relaxed))
            / (1000000.0 * static_cast<double> (s.audits_performed));
    }
  s.last_audit_duration_ns
      = last_duration_ns_.load (std::memory_order_relaxed);
  return s;
}

void
FairnessStats::reset () noexcept
{
  audits_performed_.store (0U, std::memory_order_relaxed);
  violation_sum_fixed_.store (0U, std::memory_order_relaxed);
  adjustment_sum_fixed_.store (0U, std::memory_order_relaxed);
  last_duration_ns_.store (0U, std::memory_order_relaxed);
}

auto
FairnessStats::global () noexcept -> FairnessStats &
{
  static FairnessStats inst;
  return inst;
}

void
YantraProfiler::set_hook (ProfilerHook *hook) noexcept
{
  g_profiler_hook.store (hook, std::memory_order_release);
}

auto
YantraProfiler::hook () noexcept -> ProfilerHook *
{
  return g_profiler_hook.load (std::memory_order_acquire);
}

void
YantraProfiler::emit_counter (std::string_view counter_name,
                              int64_t value) noexcept
{
  if (auto *h = g_profiler_hook.load (std::memory_order_acquire))
    {
      h->on_counter (counter_name, value);
    }
}

ScopedZoneTimer::ScopedZoneTimer (std::string_view zone_name) noexcept
#if YANTRA_ENABLE_STATS
    : zone_name_ (zone_name), start_ns_ (monotonic_time_ns ())
#endif
{
#if YANTRA_ENABLE_STATS
  if (auto *h = YantraProfiler::hook ())
    {
      h->on_zone_begin (zone_name_);
    }
#else
  (void)zone_name;
#endif
}

ScopedZoneTimer::~ScopedZoneTimer () noexcept
{
#if YANTRA_ENABLE_STATS
  if (auto *h = YantraProfiler::hook ())
    {
      const uint64_t elapsed = monotonic_time_ns () - start_ns_;
      h->on_zone_end (zone_name_, elapsed);
    }
#endif
}

auto
monotonic_time_ns () noexcept -> uint64_t
{
  const auto now = std::chrono::steady_clock::now ().time_since_epoch ();
  return static_cast<uint64_t> (
      std::chrono::duration_cast<std::chrono::nanoseconds> (now).count ());
}

auto
compute_crc32c (std::span<const std::byte> data,
                uint32_t initial_crc) noexcept -> uint32_t
{
  uint64_t crc = ~static_cast<uint64_t> (initial_crc) & 0xFFFFFFFFULL;
  const auto *ptr = reinterpret_cast<const uint8_t *> (data.data ());
  std::size_t len = data.size ();

#if defined(__SSE4_2__) && defined(__x86_64__)
  while (len >= 8U)
    {
      uint64_t word = 0U;
      std::memcpy (&word, ptr, sizeof (uint64_t));
      crc = _mm_crc32_u64 (crc, word);
      ptr += 8U;
      len -= 8U;
    }
  auto crc32 = static_cast<uint32_t> (crc);
  while (len > 0U)
    {
      crc32 = _mm_crc32_u8 (crc32, *ptr);
      ++ptr;
      --len;
    }
  return ~crc32;
#else
  auto crc32 = static_cast<uint32_t> (crc);
  for (std::size_t i = 0U; i < len; ++i)
    {
      crc32 ^= ptr[i];
      for (int bit = 0; bit < 8; ++bit)
        {
          crc32 = (crc32 >> 1U) ^ (0x82F63B78U & (0U - (crc32 & 1U)));
        }
    }
  return ~crc32;
#endif
}

auto
compute_sha256_hex (std::span<const std::byte> data) noexcept -> std::string
{
  std::array<uint32_t, 8> state = { 0x6a09e667UL, 0xbb67ae85UL, 0x3c6ef372UL,
                                    0xa54ff53aUL, 0x510e527fUL, 0x9b05688cUL,
                                    0x1f83d9abUL, 0x5be0cd19UL };

  const auto *bytes = reinterpret_cast<const uint8_t *> (data.data ());
  const std::size_t total_len = data.size ();
  const std::size_t full_blocks = total_len / 64U;

#if defined(__x86_64__)
  const bool has_shani = __builtin_cpu_supports ("sha")
                         && __builtin_cpu_supports ("sse4.1")
                         && __builtin_cpu_supports ("ssse3");
  if (full_blocks > 0U)
    {
      if (has_shani)
        {
          sha256_transform_shani (state, bytes, full_blocks);
        }
      else
        {
          sha256_transform_scalar (state, bytes, full_blocks);
        }
    }
#else
  if (full_blocks > 0U)
    {
      sha256_transform_scalar (state, bytes, full_blocks);
    }
#endif

  std::array<uint8_t, 128> tail{};
  const std::size_t rem = total_len - full_blocks * 64U;
  if (rem > 0U)
    {
      std::memcpy (tail.data (), bytes + full_blocks * 64U, rem);
    }
  tail[rem] = 0x80U;
  const std::size_t tail_blocks = (rem < 56U) ? 1U : 2U;
  const uint64_t bit_len = static_cast<uint64_t> (total_len) * 8ULL;
  const std::size_t len_pos = tail_blocks * 64U - 8U;
  for (std::size_t i = 0U; i < 8U; ++i)
    {
      tail[len_pos + i]
          = static_cast<uint8_t> (bit_len >> ((7U - i) * 8U));
    }

#if defined(__x86_64__)
  if (has_shani)
    {
      sha256_transform_shani (state, tail.data (), tail_blocks);
    }
  else
    {
      sha256_transform_scalar (state, tail.data (), tail_blocks);
    }
#else
  sha256_transform_scalar (state, tail.data (), tail_blocks);
#endif

  std::array<uint8_t, 32> digest{};
  for (std::size_t i = 0U; i < 8U; ++i)
    {
      digest[i * 4U] = static_cast<uint8_t> (state[i] >> 24U);
      digest[i * 4U + 1U] = static_cast<uint8_t> (state[i] >> 16U);
      digest[i * 4U + 2U] = static_cast<uint8_t> (state[i] >> 8U);
      digest[i * 4U + 3U] = static_cast<uint8_t> (state[i]);
    }
  return to_hex_string (digest.data (), 32U);
}

auto
compute_blake3_hex (std::span<const std::byte> data) noexcept -> std::string
{
  const auto *bytes = reinterpret_cast<const uint8_t *> (data.data ());
  const std::size_t total_len = data.size ();
  constexpr std::size_t kSegmentSize = 4ULL * 1024ULL * 1024ULL;

  if (total_len <= kSegmentSize)
    {
      const auto cv = blake3_hash_segment (bytes, total_len, 0U);
      return to_hex_string (reinterpret_cast<const uint8_t *> (cv.data ()),
                            32U);
    }

  const std::size_t num_segments
      = (total_len + kSegmentSize - 1U) / kSegmentSize;
  std::vector<std::array<uint32_t, 8>> cvs (num_segments);

  const std::size_t hw_threads
      = std::max<std::size_t> (1U, std::thread::hardware_concurrency ());
  const std::size_t num_workers = std::min (hw_threads, num_segments);

  std::atomic<std::size_t> next_seg{ 0U };
  std::vector<std::jthread> workers;
  workers.reserve (num_workers);
  for (std::size_t w = 0U; w < num_workers; ++w)
    {
      workers.emplace_back ([&] () {
        for (;;)
          {
            const std::size_t s
                = next_seg.fetch_add (1U, std::memory_order_relaxed);
            if (s >= num_segments)
              {
                break;
              }
            const std::size_t off = s * kSegmentSize;
            const std::size_t seg_len
                = std::min (kSegmentSize, total_len - off);
            cvs[s] = blake3_hash_segment (bytes + off, seg_len, s);
          }
      });
    }
  workers.clear ();

  std::array<uint32_t, 8> root_cv = kBlake3IV;
  for (std::size_t s = 0U; s < num_segments; ++s)
    {
      std::array<uint32_t, 16> words{};
      for (std::size_t i = 0U; i < 8U; ++i)
        {
          words[i] = root_cv[i];
          words[i + 8U] = cvs[s][i];
        }
      root_cv = blake3_compress (kBlake3IV, words, s, 64U,
                                 (s + 1U == num_segments) ? 8U : 4U);
    }

  return to_hex_string (reinterpret_cast<const uint8_t *> (root_cv.data ()),
                        32U);
}

} /* namespace yantra::inline v1 */
