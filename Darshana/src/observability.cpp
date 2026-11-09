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

#include <darshana/observability.hpp>

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

namespace darshana::inline v1
{

namespace
{

/* Converts LogLevel enum to string representation.  */
auto
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

/* Appends a JSON-escaped string to out.  */
void
append_json_escaped (std::string &out, std::string_view text)
{
  for (const char ch : text)
    {
      if (ch == '"')
        {
          out += "\\\"";
        }
      else if (ch == '\\')
        {
          out += "\\\\";
        }
      else if (ch == '\n')
        {
          out += "\\n";
        }
      else if (ch == '\r')
        {
          out += "\\r";
        }
      else if (ch == '\t')
        {
          out += "\\t";
        }
      else
        {
          out.push_back (ch);
        }
    }
}

/* Default stderr JSON-lines sink using fwrite without stdio formatting.  */
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
        line.reserve (128U + subsystem.size () + message.size ());
        line += "{\"ts_ns\":";
        line += std::to_string (timestamp_ns);
        line += ",\"level\":\"";
        line += level_to_string (level);
        line += "\",\"subsystem\":\"";
        append_json_escaped (line, subsystem);
        line += "\",\"msg\":\"";
        append_json_escaped (line, message);
        line += "\"}\n";
        std::lock_guard<std::mutex> lock (mutex_);
        (void)std::fwrite (line.data (), 1U, line.size (), stderr);
      }
    catch (...)
      {
      }
  }

private:
  std::mutex mutex_;
};

struct LoggerState
{
  std::atomic<LogLevel> min_level{ LogLevel::Warn };
  std::mutex sink_mutex;
  std::shared_ptr<LogSink> custom_sink;
  StderrJsonSink default_sink;
};

/* Accesses the process-wide logger state singleton.  */
auto
logger_state () noexcept -> LoggerState &
{
  static LoggerState state;
  return state;
}

/* Updates an atomic maximum value using compare-and-swap.  */
void
atomic_update_max (std::atomic<uint64_t> &target, uint64_t candidate) noexcept
{
  uint64_t prev = target.load (std::memory_order_relaxed);
  while (prev < candidate
         && !target.compare_exchange_weak (prev, candidate,
                                           std::memory_order_relaxed))
    {
    }
}

std::atomic<ProfilerHook *> g_profiler_hook{ nullptr };

} /* namespace */

/* Sets minimum severity threshold for emitted logs.  */
void
Logger::set_level (LogLevel level) noexcept
{
  logger_state ().min_level.store (level, std::memory_order_relaxed);
}

/* Queries current minimum log severity threshold.  */
auto
Logger::level () noexcept -> LogLevel
{
  return logger_state ().min_level.load (std::memory_order_relaxed);
}

/* Installs a custom LogSink or restores default stderr JSON sink.  */
void
Logger::set_sink (std::shared_ptr<LogSink> sink) noexcept
{
  auto &state = logger_state ();
  std::lock_guard<std::mutex> lock (state.sink_mutex);
  state.custom_sink = std::move (sink);
}

/* Emits a structured log record if level meets threshold.  */
void
Logger::log (LogLevel level, std::string_view subsystem,
             std::string_view message) noexcept
{
  auto &state = logger_state ();
  if (static_cast<uint8_t> (level) < static_cast<uint8_t> (
          state.min_level.load (std::memory_order_relaxed)))
    {
      return;
    }
  const uint64_t ts = monotonic_time_ns ();
  std::shared_ptr<LogSink> sink_copy;
  {
    std::lock_guard<std::mutex> lock (state.sink_mutex);
    sink_copy = state.custom_sink;
  }
  if (sink_copy)
    {
      sink_copy->write (level, subsystem, message, ts);
    }
  else
    {
      state.default_sink.write (level, subsystem, message, ts);
    }
}

/* Records elapsed duration for a frame stage.  */
void
FrameStats::record_stage (FrameStage stage, uint64_t duration_ns) noexcept
{
#if DARSHANA_ENABLE_STATS
  switch (stage)
    {
    case FrameStage::CpuPrep:
      cpu_prep_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      break;
    case FrameStage::GpuUpload:
      gpu_upload_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      break;
    case FrameStage::GpuDraw:
      gpu_draw_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      break;
    case FrameStage::Present:
      present_ns_.fetch_add (duration_ns, std::memory_order_relaxed);
      break;
    }
#else
  (void)stage;
  (void)duration_ns;
#endif
}

/* Records completion of a rendered frame.  */
void
FrameStats::record_frame (uint64_t frame_ns) noexcept
{
#if DARSHANA_ENABLE_STATS
  last_frame_ns_.store (frame_ns, std::memory_order_relaxed);
  total_frame_ns_.fetch_add (frame_ns, std::memory_order_relaxed);
  const uint64_t idx
      = frame_count_.fetch_add (1U, std::memory_order_relaxed) + 1U;
  Profiler::mark_frame (idx);
#else
  (void)frame_ns;
#endif
}

/* Increments static base layer redraw count.  */
void
FrameStats::record_base_redraw () noexcept
{
#if DARSHANA_ENABLE_STATS
  base_layer_redraws_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

/* Increments interactive overlay layer redraw count.  */
void
FrameStats::record_overlay_redraw () noexcept
{
#if DARSHANA_ENABLE_STATS
  overlay_layer_redraws_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

/* Records a GPU memory allocation occurring inside the frame hot path.  */
void
FrameStats::record_gpu_alloc_in_frame () noexcept
{
#if DARSHANA_ENABLE_STATS
  gpu_allocations_in_frame_.fetch_add (1U, std::memory_order_relaxed);
#endif
}

/* Updates active and peak GPU memory pool usage gauges.  */
void
FrameStats::record_gpu_memory (uint64_t active_bytes,
                               uint64_t peak_bytes) noexcept
{
#if DARSHANA_ENABLE_STATS
  active_gpu_bytes_.store (active_bytes, std::memory_order_relaxed);
  atomic_update_max (peak_gpu_bytes_, peak_bytes);
#else
  (void)active_bytes;
  (void)peak_bytes;
#endif
}

/* Updates active and peak PMR scratch arena usage gauges.  */
void
FrameStats::record_arena_usage (uint64_t active_bytes) noexcept
{
#if DARSHANA_ENABLE_STATS
  active_arena_bytes_.store (active_bytes, std::memory_order_relaxed);
  atomic_update_max (peak_arena_bytes_, active_bytes);
#else
  (void)active_bytes;
#endif
}

/* Captures a snapshot of all FrameStats counters.  */
auto
FrameStats::snapshot () const noexcept -> FrameStatsSnapshot
{
  FrameStatsSnapshot snap;
  snap.cpu_prep_ns = cpu_prep_ns_.load (std::memory_order_relaxed);
  snap.gpu_upload_ns = gpu_upload_ns_.load (std::memory_order_relaxed);
  snap.gpu_draw_ns = gpu_draw_ns_.load (std::memory_order_relaxed);
  snap.present_ns = present_ns_.load (std::memory_order_relaxed);
  snap.last_frame_ns = last_frame_ns_.load (std::memory_order_relaxed);
  snap.total_frame_ns = total_frame_ns_.load (std::memory_order_relaxed);
  snap.frame_count = frame_count_.load (std::memory_order_relaxed);
  snap.base_layer_redraws
      = base_layer_redraws_.load (std::memory_order_relaxed);
  snap.overlay_layer_redraws
      = overlay_layer_redraws_.load (std::memory_order_relaxed);
  snap.gpu_allocations_in_frame
      = gpu_allocations_in_frame_.load (std::memory_order_relaxed);
  snap.active_gpu_bytes = active_gpu_bytes_.load (std::memory_order_relaxed);
  snap.peak_gpu_bytes = peak_gpu_bytes_.load (std::memory_order_relaxed);
  snap.active_arena_bytes
      = active_arena_bytes_.load (std::memory_order_relaxed);
  snap.peak_arena_bytes = peak_arena_bytes_.load (std::memory_order_relaxed);
  return snap;
}

/* Resets all telemetry counters to zero.  */
void
FrameStats::reset () noexcept
{
  cpu_prep_ns_.store (0U, std::memory_order_relaxed);
  gpu_upload_ns_.store (0U, std::memory_order_relaxed);
  gpu_draw_ns_.store (0U, std::memory_order_relaxed);
  present_ns_.store (0U, std::memory_order_relaxed);
  last_frame_ns_.store (0U, std::memory_order_relaxed);
  total_frame_ns_.store (0U, std::memory_order_relaxed);
  frame_count_.store (0U, std::memory_order_relaxed);
  base_layer_redraws_.store (0U, std::memory_order_relaxed);
  overlay_layer_redraws_.store (0U, std::memory_order_relaxed);
  gpu_allocations_in_frame_.store (0U, std::memory_order_relaxed);
  active_gpu_bytes_.store (0U, std::memory_order_relaxed);
  peak_gpu_bytes_.store (0U, std::memory_order_relaxed);
  active_arena_bytes_.store (0U, std::memory_order_relaxed);
  peak_arena_bytes_.store (0U, std::memory_order_relaxed);
}

/* Accesses the global default FrameStats instance.  */
auto
FrameStats::global () noexcept -> FrameStats &
{
  static FrameStats instance;
  return instance;
}

/* Default no-op implementation of ProfilerHook::on_frame_mark.  */
void
ProfilerHook::on_frame_mark (uint64_t frame_index) noexcept
{
  (void)frame_index;
}

/* Attaches or detaches a ProfilerHook.  */
void
Profiler::set_hook (ProfilerHook *hook) noexcept
{
  g_profiler_hook.store (hook, std::memory_order_release);
}

/* Retrieves the currently attached ProfilerHook.  */
auto
Profiler::hook () noexcept -> ProfilerHook *
{
  return g_profiler_hook.load (std::memory_order_acquire);
}

/* Emits a counter update to the attached ProfilerHook if present.  */
void
Profiler::emit_counter (std::string_view counter_name, int64_t value) noexcept
{
  if (ProfilerHook *h = g_profiler_hook.load (std::memory_order_acquire))
    {
      h->on_counter (counter_name, value);
    }
}

/* Emits a frame boundary marker to the attached ProfilerHook if present.  */
void
Profiler::mark_frame (uint64_t frame_index) noexcept
{
  if (ProfilerHook *h = g_profiler_hook.load (std::memory_order_acquire))
    {
      h->on_frame_mark (frame_index);
    }
}

/* Begins timing a frame stage and notifies the profiler hook.  */
ScopedFrameTimer::ScopedFrameTimer (FrameStats &stats, FrameStage stage,
                                    std::string_view zone_name) noexcept
#if DARSHANA_ENABLE_STATS
    : stats_ (&stats), stage_ (stage), zone_name_ (zone_name),
      start_ns_ (monotonic_time_ns ())
#endif
{
#if DARSHANA_ENABLE_STATS
  if (ProfilerHook *h = Profiler::hook ())
    {
      h->on_zone_begin (zone_name_);
    }
#else
  (void)stats;
  (void)stage;
  (void)zone_name;
#endif
}

/* Ends timing a frame stage and commits elapsed duration.  */
ScopedFrameTimer::~ScopedFrameTimer () noexcept
{
#if DARSHANA_ENABLE_STATS
  const uint64_t elapsed = monotonic_time_ns () - start_ns_;
  if (stats_ != nullptr)
    {
      stats_->record_stage (stage_, elapsed);
    }
  if (ProfilerHook *h = Profiler::hook ())
    {
      h->on_zone_end (zone_name_, elapsed);
    }
#endif
}

/* Returns steady_clock timestamp in nanoseconds.  */
auto
monotonic_time_ns () noexcept -> uint64_t
{
  const auto now = std::chrono::steady_clock::now ().time_since_epoch ();
  return static_cast<uint64_t> (
      std::chrono::duration_cast<std::chrono::nanoseconds> (now).count ());
}

} /* namespace darshana::inline v1 */
