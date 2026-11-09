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

#include <sutra/observability.hpp>

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

namespace sutra::inline v1
{

namespace
{

constexpr std::size_t kJsonLineReserveBytes = 160U;

/* Converts a LogLevel enumerator to its uppercase JSON string label.  */
[[nodiscard]] constexpr auto
level_to_string (LogLevel level) noexcept -> std::string_view
{
  switch (level)
    {
    case LogLevel::Trace:
      return "TRACE";
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warn:
      return "WARN";
    case LogLevel::Error:
      return "ERROR";
    }
  return "INFO";
}

/* Appends a JSON-escaped string to out without raw backslashes in literals. */
void
append_json_escaped (std::string &out, std::string_view input)
{
  for (const char ch : input)
    {
      if (ch == '"' || ch == '\\')
        {
          out.push_back ('\\');
          out.push_back (ch);
        }
      else if (ch == '\n')
        {
          out.append (R"(\n)");
        }
      else if (ch == '\r')
        {
          out.append (R"(\r)");
        }
      else if (ch == '\t')
        {
          out.append (R"(\t)");
        }
      else
        {
          out.push_back (ch);
        }
    }
}

/* Default sink writing structured JSON-lines to stderr via fwrite.  */
class StderrJsonSink final : public LogSink
{
public:
  void
  write (LogLevel level, std::string_view subsystem, std::string_view message,
         uint64_t timestamp_ns) noexcept override
  {
    try
      {
        std::string line;
        line.reserve (kJsonLineReserveBytes + subsystem.size ()
                      + message.size ());
        line.append (R"({"ts_ns":)");
        line.append (std::to_string (timestamp_ns));
        line.append (R"(,"level":")");
        line.append (level_to_string (level));
        line.append (R"(","subsystem":")");
        append_json_escaped (line, subsystem);
        line.append (R"(","msg":")");
        append_json_escaped (line, message);
        line.append ("\"}\n");

        const std::scoped_lock lock (write_mtx_);
        std::fwrite (line.data (), sizeof (char), line.size (), stderr);
      }
    catch (...)
      {
        // Logging must never throw across ABI boundaries.
      }
  }

private:
  std::mutex write_mtx_;
};

/* Holds process-wide logger and profiler configuration without non-const
   namespace-scope globals.  */
struct ObservabilityRegistry
{
  std::atomic<LogLevel> min_level{ LogLevel::Warn };
  std::mutex sink_mtx;
  std::shared_ptr<LogSink> custom_sink;
  std::shared_ptr<LogSink> default_sink{ std::make_shared<StderrJsonSink> () };
  std::atomic<ProfilerHook *> profiler_hook{ nullptr };

  [[nodiscard]] static auto
  instance () noexcept -> ObservabilityRegistry &
  {
    static ObservabilityRegistry reg;
    return reg;
  }
};

} /* namespace */

/* Queries steady_clock for nanosecond timestamps.  */
auto
monotonic_time_ns () noexcept -> uint64_t
{
  const auto now = std::chrono::steady_clock::now ().time_since_epoch ();
  return static_cast<uint64_t> (
      std::chrono::duration_cast<std::chrono::nanoseconds> (now).count ());
}

/* Updates the minimum log level atomically.  */
void
Logger::set_level (LogLevel level) noexcept
{
  ObservabilityRegistry::instance ().min_level.store (
      level, std::memory_order_relaxed);
}

/* Reads the minimum log level atomically.  */
auto
Logger::level () noexcept -> LogLevel
{
  return ObservabilityRegistry::instance ().min_level.load (
      std::memory_order_relaxed);
}

/* Replaces the active log sink.  */
void
Logger::set_sink (std::shared_ptr<LogSink> sink) noexcept
{
  auto &reg = ObservabilityRegistry::instance ();
  const std::scoped_lock lock (reg.sink_mtx);
  reg.custom_sink = std::move (sink);
}

/* Emits a structured log line if severity >= min_level.  */
void
Logger::log (LogLevel level, std::string_view subsystem,
             std::string_view message) noexcept
{
  auto &reg = ObservabilityRegistry::instance ();
  if (static_cast<uint8_t> (level)
      < static_cast<uint8_t> (reg.min_level.load (std::memory_order_relaxed)))
    {
      return;
    }

  std::shared_ptr<LogSink> active_sink;
  {
    const std::scoped_lock lock (reg.sink_mtx);
    active_sink
        = (reg.custom_sink != nullptr) ? reg.custom_sink : reg.default_sink;
  }
  if (active_sink != nullptr)
    {
      active_sink->write (level, subsystem, message, monotonic_time_ns ());
    }
}

/* Accumulates elapsed nanoseconds and increments call count for stage.  */
void
AgentStats::record_stage (StageKind stage, uint64_t duration_ns) noexcept
{
  switch (stage)
    {
    case StageKind::Planning:
      planning_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      planning_calls_.fetch_add (1U, std::memory_order_relaxed);
      break;
    case StageKind::ToolDispatch:
      tool_dispatch_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      tool_dispatch_calls_.fetch_add (1U, std::memory_order_relaxed);
      break;
    case StageKind::LlmInference:
      llm_inference_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      llm_inference_calls_.fetch_add (1U, std::memory_order_relaxed);
      break;
    case StageKind::ToolExecution:
      tool_execution_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      tool_execution_calls_.fetch_add (1U, std::memory_order_relaxed);
      break;
    case StageKind::CheckpointWrite:
      checkpoint_write_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      checkpoint_write_calls_.fetch_add (1U, std::memory_order_relaxed);
      break;
    }
}

/* Increments the count of LLM tokens generated concurrently with tools.  */
void
AgentStats::record_overlapped_tokens (uint64_t token_count) noexcept
{
  overlapped_tool_tokens_.fetch_add (token_count, std::memory_order_relaxed);
}

/* Updates active and high-water-mark scratch arena byte counters.  */
void
AgentStats::record_arena_alloc (uint64_t bytes) noexcept
{
  const uint64_t current
      = active_arena_bytes_.fetch_add (bytes, std::memory_order_relaxed)
        + bytes;
  uint64_t prev_peak = peak_arena_bytes_.load (std::memory_order_relaxed);
  while (current > prev_peak
         && !peak_arena_bytes_.compare_exchange_weak (
             prev_peak, current, std::memory_order_relaxed))
    {
    }
}

/* Decrements active scratch arena byte counter upon arena reset.  */
void
AgentStats::record_arena_release (uint64_t bytes) noexcept
{
  uint64_t current = active_arena_bytes_.load (std::memory_order_relaxed);
  for (;;)
    {
      const uint64_t next = (current >= bytes) ? (current - bytes) : 0U;
      if (active_arena_bytes_.compare_exchange_weak (
              current, next, std::memory_order_relaxed))
        {
          break;
        }
    }
}

/* Records time-to-first-token (TTFT) latency for a turn.  */
void
AgentStats::record_ttft (uint64_t ttft_ns) noexcept
{
  last_ttft_ns_.store (ttft_ns, std::memory_order_relaxed);
  total_ttft_ns_.fetch_add (ttft_ns, std::memory_order_relaxed);
  ttft_samples_.fetch_add (1U, std::memory_order_relaxed);
  AgentProfiler::emit_counter ("ttft_ns", static_cast<int64_t> (ttft_ns));
}

/* Records per-turn generated token count and generation duration.  */
void
AgentStats::record_generation (uint64_t tokens_generated,
                               uint64_t generation_ns) noexcept
{
  last_tokens_generated_.store (tokens_generated, std::memory_order_relaxed);
  total_tokens_generated_.fetch_add (tokens_generated,
                                     std::memory_order_relaxed);
  last_generation_ns_.store (generation_ns, std::memory_order_relaxed);
  AgentProfiler::emit_counter ("tokens_generated",
                               static_cast<int64_t> (tokens_generated));
}

/* Records sentence-boundary-to-TTS dispatch latency.  */
void
AgentStats::record_sentence_to_tts (uint64_t latency_ns) noexcept
{
  sentence_to_tts_ns_.fetch_add (latency_ns, std::memory_order_relaxed);
  sentence_to_tts_samples_.fetch_add (1U, std::memory_order_relaxed);
  AgentProfiler::emit_counter ("sentence_to_tts_ns",
                               static_cast<int64_t> (latency_ns));
}

/* Updates the TTS queue depth gauge.  */
void
AgentStats::set_tts_queue_depth (uint64_t depth) noexcept
{
  tts_queue_depth_.store (depth, std::memory_order_relaxed);
  AgentProfiler::emit_counter ("tts_queue_depth",
                               static_cast<int64_t> (depth));
}

/* Updates the audio playback ring buffer occupancy gauge.  */
void
AgentStats::set_playback_buffer_occupancy (uint64_t samples) noexcept
{
  playback_buffer_occupancy_.store (samples, std::memory_order_relaxed);
  AgentProfiler::emit_counter ("playback_buffer_occupancy",
                               static_cast<int64_t> (samples));
}

/* Records STT hypothesis latency.  */
void
AgentStats::record_stt_latency (uint64_t latency_ns) noexcept
{
  stt_partial_latency_ns_.store (latency_ns, std::memory_order_relaxed);
  AgentProfiler::emit_counter ("stt_latency_ns",
                               static_cast<int64_t> (latency_ns));
}

/* Increments the voice barge-in interruption counter.  */
void
AgentStats::record_barge_in () noexcept
{
  const uint64_t count
      = barge_in_count_.fetch_add (1U, std::memory_order_relaxed) + 1U;
  AgentProfiler::emit_counter ("barge_in_count", static_cast<int64_t> (count));
}

/* Increments the ALSA PCM overrun/underrun (xrun) counter.  */
void
AgentStats::record_xrun () noexcept
{
  const uint64_t count
      = xrun_count_.fetch_add (1U, std::memory_order_relaxed) + 1U;
  AgentProfiler::emit_counter ("xrun_count", static_cast<int64_t> (count));
}

/* Queries the current ALSA PCM xrun count.  */
auto
AgentStats::xrun_count () const noexcept -> uint64_t
{
  return xrun_count_.load (std::memory_order_relaxed);
}

/* Captures an atomic point-in-time snapshot of all counters.  */
auto
AgentStats::snapshot () const noexcept -> AgentStatsSnapshot
{
  AgentStatsSnapshot snap;
  snap.planning_ns = planning_ns_.load (std::memory_order_relaxed);
  snap.planning_calls = planning_calls_.load (std::memory_order_relaxed);
  snap.tool_dispatch_ns = tool_dispatch_ns_.load (std::memory_order_relaxed);
  snap.tool_dispatch_calls
      = tool_dispatch_calls_.load (std::memory_order_relaxed);
  snap.llm_inference_ns = llm_inference_ns_.load (std::memory_order_relaxed);
  snap.llm_inference_calls
      = llm_inference_calls_.load (std::memory_order_relaxed);
  snap.tool_execution_ns = tool_execution_ns_.load (std::memory_order_relaxed);
  snap.tool_execution_calls
      = tool_execution_calls_.load (std::memory_order_relaxed);
  snap.checkpoint_write_ns
      = checkpoint_write_ns_.load (std::memory_order_relaxed);
  snap.checkpoint_write_calls
      = checkpoint_write_calls_.load (std::memory_order_relaxed);
  snap.overlapped_tool_tokens
      = overlapped_tool_tokens_.load (std::memory_order_relaxed);
  snap.active_arena_bytes
      = active_arena_bytes_.load (std::memory_order_relaxed);
  snap.peak_arena_bytes = peak_arena_bytes_.load (std::memory_order_relaxed);
  snap.last_ttft_ns = last_ttft_ns_.load (std::memory_order_relaxed);
  snap.total_ttft_ns = total_ttft_ns_.load (std::memory_order_relaxed);
  snap.ttft_samples = ttft_samples_.load (std::memory_order_relaxed);
  snap.last_generation_ns
      = last_generation_ns_.load (std::memory_order_relaxed);
  snap.last_tokens_generated
      = last_tokens_generated_.load (std::memory_order_relaxed);
  snap.total_tokens_generated
      = total_tokens_generated_.load (std::memory_order_relaxed);
  snap.last_tokens_per_sec
      = snap.last_generation_ns > 0U
            ? (static_cast<double> (snap.last_tokens_generated) * 1.0e9
               / static_cast<double> (snap.last_generation_ns))
            : 0.0;
  snap.sentence_to_tts_ns
      = sentence_to_tts_ns_.load (std::memory_order_relaxed);
  snap.sentence_to_tts_samples
      = sentence_to_tts_samples_.load (std::memory_order_relaxed);
  snap.tts_queue_depth = tts_queue_depth_.load (std::memory_order_relaxed);
  snap.playback_buffer_occupancy
      = playback_buffer_occupancy_.load (std::memory_order_relaxed);
  snap.stt_partial_latency_ns
      = stt_partial_latency_ns_.load (std::memory_order_relaxed);
  snap.barge_in_count = barge_in_count_.load (std::memory_order_relaxed);
  snap.xrun_count = xrun_count_.load (std::memory_order_relaxed);
  return snap;
}

/* Zeroes all telemetry counters.  */
void
AgentStats::reset () noexcept
{
  planning_ns_.store (0U, std::memory_order_relaxed);
  planning_calls_.store (0U, std::memory_order_relaxed);
  tool_dispatch_ns_.store (0U, std::memory_order_relaxed);
  tool_dispatch_calls_.store (0U, std::memory_order_relaxed);
  llm_inference_ns_.store (0U, std::memory_order_relaxed);
  llm_inference_calls_.store (0U, std::memory_order_relaxed);
  tool_execution_ns_.store (0U, std::memory_order_relaxed);
  tool_execution_calls_.store (0U, std::memory_order_relaxed);
  checkpoint_write_ns_.store (0U, std::memory_order_relaxed);
  checkpoint_write_calls_.store (0U, std::memory_order_relaxed);
  overlapped_tool_tokens_.store (0U, std::memory_order_relaxed);
  active_arena_bytes_.store (0U, std::memory_order_relaxed);
  peak_arena_bytes_.store (0U, std::memory_order_relaxed);
  last_ttft_ns_.store (0U, std::memory_order_relaxed);
  total_ttft_ns_.store (0U, std::memory_order_relaxed);
  ttft_samples_.store (0U, std::memory_order_relaxed);
  last_generation_ns_.store (0U, std::memory_order_relaxed);
  last_tokens_generated_.store (0U, std::memory_order_relaxed);
  total_tokens_generated_.store (0U, std::memory_order_relaxed);
  sentence_to_tts_ns_.store (0U, std::memory_order_relaxed);
  sentence_to_tts_samples_.store (0U, std::memory_order_relaxed);
  tts_queue_depth_.store (0U, std::memory_order_relaxed);
  playback_buffer_occupancy_.store (0U, std::memory_order_relaxed);
  stt_partial_latency_ns_.store (0U, std::memory_order_relaxed);
  barge_in_count_.store (0U, std::memory_order_relaxed);
  xrun_count_.store (0U, std::memory_order_relaxed);
}

/* Returns the process-wide default AgentStats instance.  */
auto
AgentStats::global () noexcept -> AgentStats &
{
  static AgentStats instance;
  return instance;
}

/* Sets the global external profiler hook pointer atomically.  */
void
AgentProfiler::set_hook (ProfilerHook *hook) noexcept
{
  ObservabilityRegistry::instance ().profiler_hook.store (
      hook, std::memory_order_release);
}

/* Loads the global external profiler hook pointer atomically.  */
auto
AgentProfiler::hook () noexcept -> ProfilerHook *
{
  return ObservabilityRegistry::instance ().profiler_hook.load (
      std::memory_order_acquire);
}

/* Forwards a numeric counter event to the attached profiler hook.  */
void
AgentProfiler::emit_counter (std::string_view counter_name,
                             int64_t value) noexcept
{
  ProfilerHook *const active = hook ();
  if (active != nullptr)
    {
      active->on_counter (counter_name, value);
    }
}

/* Starts timing a stage and notifies the external profiler hook.  */
ScopedStageTimer::ScopedStageTimer (
    [[maybe_unused]] AgentStats &stats, [[maybe_unused]] StageKind stage,
    [[maybe_unused]] std::string_view zone_name) noexcept
#if SUTRA_ENABLE_STATS
    : stats_ (&stats), stage_ (stage), zone_name_ (zone_name),
      start_ns_ (monotonic_time_ns ())
#endif
{
#if SUTRA_ENABLE_STATS
  ProfilerHook *const active = AgentProfiler::hook ();
  if (active != nullptr)
    {
      active->on_zone_begin (zone_name_);
    }
#endif
}

/* Stops timing a stage, updates stats, and notifies profiler hook.  */
ScopedStageTimer::~ScopedStageTimer () noexcept
{
#if SUTRA_ENABLE_STATS
  const uint64_t end_ns = monotonic_time_ns ();
  const uint64_t elapsed = (end_ns >= start_ns_) ? (end_ns - start_ns_) : 0U;
  if (stats_ != nullptr)
    {
      stats_->record_stage (stage_, elapsed);
    }
  ProfilerHook *const active = AgentProfiler::hook ();
  if (active != nullptr)
    {
      active->on_zone_end (zone_name_, elapsed);
    }
#endif
}

} /* namespace sutra::inline v1 */
