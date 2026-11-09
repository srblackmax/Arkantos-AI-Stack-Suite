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

#include <darshana/render_loop.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <utility>

namespace darshana::inline v1
{

namespace
{

enum class CommandKind : uint8_t
{
  RenderOnly = 0,
  UpdateOverlay = 1,
  StreamPoint = 2,
  SetViewport = 3
};

struct LoopCommand
{
  CommandKind kind = CommandKind::RenderOnly;
  uint64_t frame_id = 0U;
  OverlaySpec overlay{};
  float stream_x = 0.0F;
  float stream_y = 0.0F;
  double vp_xmin = 0.0;
  double vp_xmax = 1.0;
  double vp_ymin = 0.0;
  double vp_ymax = 1.0;
};

/* Lock-free single-producer single-consumer bounded ring queue.  */
template <typename T, std::size_t Capacity> class LockFreeSpscQueue
{
public:
  [[nodiscard]] auto
  try_push (const T &item) noexcept -> bool
  {
    const std::size_t w = write_idx_.load (std::memory_order_relaxed);
    const std::size_t next = (w + 1U) % Capacity;
    if (next == read_idx_.load (std::memory_order_acquire))
      {
        return false;
      }
    slots_[w] = item;
    write_idx_.store (next, std::memory_order_release);
    return true;
  }

  [[nodiscard]] auto
  try_pop (T &out) noexcept -> bool
  {
    const std::size_t r = read_idx_.load (std::memory_order_relaxed);
    if (r == write_idx_.load (std::memory_order_acquire))
      {
        return false;
      }
    out = slots_[r];
    read_idx_.store ((r + 1U) % Capacity, std::memory_order_release);
    return true;
  }

private:
  std::array<T, Capacity> slots_{};
  std::atomic<std::size_t> write_idx_{ 0U };
  std::atomic<std::size_t> read_idx_{ 0U };
};

} /* namespace */

struct RenderLoop::Impl
{
  Figure *figure = nullptr;
  uint32_t target_fps = 60U;
  std::atomic<bool> running{ false };
  std::atomic<uint64_t> next_frame_id{ 0U };
  LockFreeSpscQueue<LoopCommand, 1024U> cmd_queue;
  LockFreeSpscQueue<CompletedFrame, 1024U> done_queue;
  std::jthread worker;

  Impl () noexcept = default;
  Impl (const Impl &) = delete;
  auto operator= (const Impl &) -> Impl & = delete;
  Impl (Impl &&) = delete;
  auto operator= (Impl &&) -> Impl & = delete;

  ~Impl () noexcept { stop (); }

  void
  stop () noexcept
  {
    running.store (false, std::memory_order_release);
    if (worker.joinable ())
      {
        worker.join ();
      }
  }

  void
  worker_loop () noexcept
  {
    while (running.load (std::memory_order_acquire))
      {
        LoopCommand cmd{};
        bool did_work = false;
        while (cmd_queue.try_pop (cmd))
          {
            did_work = true;
            if (cmd.kind == CommandKind::UpdateOverlay)
              {
                (void)figure->update_overlay (cmd.overlay);
              }
            else if (cmd.kind == CommandKind::StreamPoint)
              {
                (void)figure->append_stream (cmd.stream_x, cmd.stream_y);
              }
            else if (cmd.kind == CommandKind::SetViewport)
              {
                (void)figure->set_viewport (cmd.vp_xmin, cmd.vp_xmax,
                                            cmd.vp_ymin, cmd.vp_ymax);
              }
            auto snap_res = figure->render_frame (false);
            CompletedFrame cf{};
            cf.frame_id = cmd.frame_id;
            cf.timestamp_ns = monotonic_time_ns ();
            if (snap_res.has_value ())
              {
                cf.stats = *snap_res;
              }
            if (!done_queue.try_push (cf))
              {
                CompletedFrame discard{};
                (void)done_queue.try_pop (discard);
                (void)done_queue.try_push (cf);
              }
          }
        if (!did_work)
          {
            std::this_thread::sleep_for (std::chrono::microseconds (100));
          }
      }
  }
};

/* Default-constructs an empty RenderLoop handle.  */
RenderLoop::RenderLoop () noexcept = default;

/* Constructs a RenderLoop from an internal implementation pointer.  */
RenderLoop::RenderLoop (std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

RenderLoop::RenderLoop (RenderLoop &&other) noexcept = default;

auto
RenderLoop::operator= (RenderLoop &&other) noexcept -> RenderLoop & = default;

RenderLoop::~RenderLoop () noexcept = default;

/* Creates and starts a dedicated RenderLoop for the given Figure.  */
auto
RenderLoop::create (Figure &figure,
                    uint32_t target_fps) noexcept -> Result<RenderLoop>
{
  if (target_fps == 0U)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::InvalidState, 0,
          "RenderLoop::create requires target_fps greater than zero"));
    }
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->figure = &figure;
      impl->target_fps = target_fps;
      impl->running.store (true, std::memory_order_release);
      Impl *raw = impl.get ();
      impl->worker = std::jthread ([raw] () { raw->worker_loop (); });
      return RenderLoop (std::move (impl));
    }
  catch (...)
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Failed to spawn dedicated RenderLoop worker thread"));
    }
}

/* Stops the dedicated rendering thread cleanly.  */
void
RenderLoop::stop () noexcept
{
  if (impl_)
    {
      impl_->stop ();
    }
}

/* Enqueues a lock-free interactive overlay update command.  */
auto
RenderLoop::submit_overlay (const OverlaySpec &overlay) noexcept
    -> Result<uint64_t>
{
  if (!impl_ || !impl_->running.load (std::memory_order_acquire))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::NotInitialized, 0,
                            "Cannot submit_overlay to a stopped RenderLoop"));
    }
  const uint64_t fid
      = impl_->next_frame_id.fetch_add (1U, std::memory_order_relaxed) + 1U;
  LoopCommand cmd{};
  cmd.kind = CommandKind::UpdateOverlay;
  cmd.frame_id = fid;
  cmd.overlay = overlay;
  if (!impl_->cmd_queue.try_push (cmd))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "RenderLoop lock-free command queue is full"));
    }
  return fid;
}

/* Enqueues a lock-free streaming point append command.  */
auto
RenderLoop::submit_stream_point (float x, float y) noexcept -> Result<uint64_t>
{
  if (!impl_ || !impl_->running.load (std::memory_order_acquire))
    {
      return std::unexpected (Error::framework (
          FrameworkErrorCode::NotInitialized, 0,
          "Cannot submit_stream_point to a stopped RenderLoop"));
    }
  const uint64_t fid
      = impl_->next_frame_id.fetch_add (1U, std::memory_order_relaxed) + 1U;
  LoopCommand cmd{};
  cmd.kind = CommandKind::StreamPoint;
  cmd.frame_id = fid;
  cmd.stream_x = x;
  cmd.stream_y = y;
  if (!impl_->cmd_queue.try_push (cmd))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "RenderLoop lock-free command queue is full"));
    }
  return fid;
}

/* Enqueues a lock-free viewport pan/zoom update command.  */
auto
RenderLoop::submit_viewport (double x_min, double x_max, double y_min,
                             double y_max) noexcept -> Result<uint64_t>
{
  if (!impl_ || !impl_->running.load (std::memory_order_acquire))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::NotInitialized, 0,
                            "Cannot submit_viewport to a stopped RenderLoop"));
    }
  if (x_min >= x_max || y_min >= y_max)
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidScaleDomain, "viewport",
                       "submit_viewport requires min < max for both axes"));
    }
  const uint64_t fid
      = impl_->next_frame_id.fetch_add (1U, std::memory_order_relaxed) + 1U;
  LoopCommand cmd{};
  cmd.kind = CommandKind::SetViewport;
  cmd.frame_id = fid;
  cmd.vp_xmin = x_min;
  cmd.vp_xmax = x_max;
  cmd.vp_ymin = y_min;
  cmd.vp_ymax = y_max;
  if (!impl_->cmd_queue.try_push (cmd))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "RenderLoop lock-free command queue is full"));
    }
  return fid;
}

/* Enqueues an explicit frame render request on the lock-free queue.  */
auto
RenderLoop::request_frame () noexcept -> Result<uint64_t>
{
  if (!impl_ || !impl_->running.load (std::memory_order_acquire))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::NotInitialized, 0,
                            "Cannot request_frame on a stopped RenderLoop"));
    }
  const uint64_t fid
      = impl_->next_frame_id.fetch_add (1U, std::memory_order_relaxed) + 1U;
  LoopCommand cmd{};
  cmd.kind = CommandKind::RenderOnly;
  cmd.frame_id = fid;
  if (!impl_->cmd_queue.try_push (cmd))
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "RenderLoop lock-free command queue is full"));
    }
  return fid;
}

/* Polls the lock-free completion queue for the next presented frame.  */
auto
RenderLoop::poll_completed () noexcept -> std::optional<CompletedFrame>
{
  if (!impl_)
    {
      return std::nullopt;
    }
  CompletedFrame cf{};
  if (impl_->done_queue.try_pop (cf))
    {
      return cf;
    }
  return std::nullopt;
}

/* Waits until the specified frame_id is presented on the completion queue.  */
auto
RenderLoop::wait_for_frame (uint64_t target_frame_id,
                            uint64_t timeout_ms) noexcept
    -> Result<CompletedFrame>
{
  if (!impl_)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::NotInitialized, 0,
                            "RenderLoop is not initialized"));
    }
  const uint64_t deadline = monotonic_time_ns () + timeout_ms * 1'000'000ULL;
  while (monotonic_time_ns () < deadline)
    {
      CompletedFrame cf{};
      if (impl_->done_queue.try_pop (cf))
        {
          if (cf.frame_id >= target_frame_id)
            {
              return cf;
            }
        }
      else
        {
          std::this_thread::sleep_for (std::chrono::microseconds (50));
        }
    }
  return std::unexpected (
      Error::framework (FrameworkErrorCode::InvalidState, 0,
                        "Timed out waiting for CompletedFrame on RenderLoop"));
}

} /* namespace darshana::inline v1 */
