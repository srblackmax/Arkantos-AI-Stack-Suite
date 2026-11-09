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

#include <surya/core/observability.hpp>

#include <cstdio>
#include <mutex>
#include <string>

namespace surya::inline v1::core
{

namespace
{

constexpr std::string_view
level_to_str (Log_Level level) noexcept
{
  switch (level)
    {
    case Log_Level::Debug:
      return "DEBUG";
    case Log_Level::Info:
      return "INFO";
    case Log_Level::Warn:
      return "WARN";
    case Log_Level::Error:
      return "ERROR";
    case Log_Level::Off:
      return "OFF";
    }
  return "INFO";
}

void
append_json_escaped (std::string &out, std::string_view sv)
{
  for (char c : sv)
    {
      if (c == '"' || c == '\\')
        {
          out.push_back ('\\');
          out.push_back (c);
        }
      else if (c == '\n')
        {
          out.append ("\\n");
        }
      else if (c == '\r')
        {
          out.append ("\\r");
        }
      else
        {
          out.push_back (c);
        }
    }
}

class Stderr_Json_Log_Sink final : public SuryaLogSink
{
public:
  void
  write (Log_Level level,
         std::string_view subsystem,
         std::string_view message,
         uint64_t timestamp_ns) noexcept override
  {
    try
      {
        std::string line;
        line.reserve (128U + subsystem.size () + message.size ());
        line.append ("{\"ts_ns\":");
        line.append (std::to_string (timestamp_ns));
        line.append (",\"level\":\"");
        line.append (level_to_str (level));
        line.append ("\",\"subsystem\":\"");
        append_json_escaped (line, subsystem);
        line.append ("\",\"msg\":\"");
        append_json_escaped (line, message);
        line.append ("\"}\n");

        std::lock_guard<std::mutex> lock (mtx_);
        (void) std::fwrite (line.data (), 1U, line.size (), stderr);
      }
    catch (...)
      {
      }
  }

private:
  std::mutex mtx_;
};

std::atomic<Log_Level> g_log_level {Log_Level::Warn};
std::mutex g_sink_mtx;
std::shared_ptr<SuryaLogSink> g_custom_sink;
std::atomic<SuryaProfilerHook *> g_profiler_hook {nullptr};

std::shared_ptr<SuryaLogSink>
default_stderr_sink (void)
{
  static std::shared_ptr<SuryaLogSink> inst =
    std::make_shared<Stderr_Json_Log_Sink> ();
  return inst;
}

} /* namespace */

void
SuryaLogger::set_sink (std::shared_ptr<SuryaLogSink> sink) noexcept
{
  std::lock_guard<std::mutex> lock (g_sink_mtx);
  g_custom_sink = std::move (sink);
}

void
SuryaLogger::set_level (Log_Level level) noexcept
{
  g_log_level.store (level, std::memory_order_relaxed);
}

Log_Level
SuryaLogger::level (void) noexcept
{
  return g_log_level.load (std::memory_order_relaxed);
}

void
SuryaLogger::log (Log_Level lvl,
                  std::string_view subsystem,
                  std::string_view message) noexcept
{
  const Log_Level min_lvl = g_log_level.load (std::memory_order_relaxed);
  if (lvl < min_lvl || min_lvl == Log_Level::Off)
    {
      return;
    }

  const auto now = std::chrono::steady_clock::now ().time_since_epoch ();
  const uint64_t ts_ns = static_cast<uint64_t> (
    std::chrono::duration_cast<std::chrono::nanoseconds> (now).count ());

  std::shared_ptr<SuryaLogSink> active_sink;
  {
    std::lock_guard<std::mutex> lock (g_sink_mtx);
    active_sink = g_custom_sink ? g_custom_sink : default_stderr_sink ();
  }
  if (active_sink)
    {
      active_sink->write (lvl, subsystem, message, ts_ns);
    }
}

SuryaStatsSnapshot
SuryaStats::snapshot (void) const noexcept
{
  SuryaStatsSnapshot s;
  s.preprocess_ns = preprocess_ns_.load (std::memory_order_relaxed);
  s.h2d_ns = h2d_ns_.load (std::memory_order_relaxed);
  s.inference_ns = inference_ns_.load (std::memory_order_relaxed);
  s.d2h_ns = d2h_ns_.load (std::memory_order_relaxed);
  s.postprocess_ns = postprocess_ns_.load (std::memory_order_relaxed);
  s.total_requests = total_requests_.load (std::memory_order_relaxed);
  s.total_images = total_images_.load (std::memory_order_relaxed);
  s.vram_current_bytes = vram_current_bytes_.load (std::memory_order_relaxed);
  s.vram_peak_bytes = vram_peak_bytes_.load (std::memory_order_relaxed);
  s.cpu_fallback_count = cpu_fallback_count_.load (std::memory_order_relaxed);
  s.cuda_graph_captures =
    cuda_graph_captures_.load (std::memory_order_relaxed);
  s.cuda_graph_launches =
    cuda_graph_launches_.load (std::memory_order_relaxed);
  return s;
}

void
SuryaStats::reset (void) noexcept
{
  preprocess_ns_.store (0ULL, std::memory_order_relaxed);
  h2d_ns_.store (0ULL, std::memory_order_relaxed);
  inference_ns_.store (0ULL, std::memory_order_relaxed);
  d2h_ns_.store (0ULL, std::memory_order_relaxed);
  postprocess_ns_.store (0ULL, std::memory_order_relaxed);
  total_requests_.store (0ULL, std::memory_order_relaxed);
  total_images_.store (0ULL, std::memory_order_relaxed);
  vram_peak_bytes_.store (
    vram_current_bytes_.load (std::memory_order_relaxed),
    std::memory_order_relaxed);
  cpu_fallback_count_.store (0ULL, std::memory_order_relaxed);
  cuda_graph_captures_.store (0ULL, std::memory_order_relaxed);
  cuda_graph_launches_.store (0ULL, std::memory_order_relaxed);
}

SuryaStats &
SuryaStats::global_stats (void) noexcept
{
  static SuryaStats inst;
  return inst;
}

void
SuryaProfiler::set_hook (SuryaProfilerHook *hook) noexcept
{
  g_profiler_hook.store (hook, std::memory_order_release);
}

SuryaProfilerHook *
SuryaProfiler::hook (void) noexcept
{
  return g_profiler_hook.load (std::memory_order_acquire);
}

} /* namespace surya::inline v1::core */
