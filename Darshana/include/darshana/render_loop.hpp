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

#ifndef DARSHANA_RENDER_LOOP_HPP
#define DARSHANA_RENDER_LOOP_HPP

#include <darshana/error.hpp>
#include <darshana/figure.hpp>
#include <darshana/observability.hpp>
#include <darshana/spec.hpp>

#include <cstdint>
#include <memory>
#include <optional>

namespace darshana::inline v1
{

/**
 * @brief Describes a completed frame delivered by RenderLoop's completion
 * queue.
 * @pre None.
 * @post Contains monotonic frame_id, completion timestamp, and frame
 * telemetry.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct CompletedFrame
{
  uint64_t frame_id = 0U;
  uint64_t timestamp_ns = 0U;
  FrameStatsSnapshot stats;
};

/**
 * @brief Dedicated rendering thread abstraction with a lock-free command queue
 * for figure updates and a lock-free completion queue for frame presentation.
 * @pre Figure must outlive the RenderLoop or be stopped before destruction.
 * @post Users submit figure commands without touching raw Vulkan sync
 *       primitives.
 * @note Thread-safety: Command submission and completion polling are lock-free
 *       and safe from the producer/UI thread while the worker thread renders.
 */
class RenderLoop
{
public:
  RenderLoop () noexcept;
  RenderLoop (const RenderLoop &) = delete;
  auto operator= (const RenderLoop &) -> RenderLoop & = delete;

  /**
   * @brief Transfers ownership of a RenderLoop instance.
   * @param other Source RenderLoop to move from.
   * @pre None.
   * @post other is empty.
   */
  RenderLoop (RenderLoop &&other) noexcept;

  /**
   * @brief Move-assigns ownership of a RenderLoop instance.
   * @param other Source RenderLoop to move from.
   * @pre None.
   * @post Previous worker thread is stopped and other is empty.
   * @returns Reference to *this.
   */
  auto operator= (RenderLoop &&other) noexcept -> RenderLoop &;

  /**
   * @brief Stops the dedicated rendering thread and releases resources.
   */
  ~RenderLoop () noexcept;

  /**
   * @brief Creates and starts a dedicated RenderLoop for the given Figure.
   * @param figure Target Figure managed by the rendering thread.
   * @param target_fps Pacing frequency in Hz (default 60 Hz).
   * @pre target_fps > 0.
   * @post Dedicated rendering thread is running and ready for commands.
   * @returns Result<RenderLoop> or FrameworkError on failure.
   * @note Thread-safety: Safe for concurrent creation on distinct Figures.
   */
  [[nodiscard]] static auto create (Figure &figure,
                                    uint32_t target_fps
                                    = 60U) noexcept -> Result<RenderLoop>;

  /**
   * @brief Stops the dedicated rendering thread cleanly.
   * @pre None.
   * @post Rendering thread has joined.
   * @note Thread-safety: Idempotent and safe to call from control thread.
   */
  void stop () noexcept;

  /**
   * @brief Enqueues a lock-free interactive overlay update command.
   * @param overlay New OverlaySpec state.
   * @pre RenderLoop is running.
   * @post Enqueues overlay command and frame request without blocking.
   * @returns Result<uint64_t> scheduled frame_id or FrameworkError if full.
   * @note Thread-safety: Lock-free SPSC ring buffer push.
   */
  [[nodiscard]] auto
  submit_overlay (const OverlaySpec &overlay) noexcept -> Result<uint64_t>;

  /**
   * @brief Enqueues a lock-free streaming point append command.
   * @param x Horizontal coordinate.
   * @param y Vertical coordinate.
   * @pre RenderLoop is running.
   * @post Enqueues streaming sample and frame request without blocking.
   * @returns Result<uint64_t> scheduled frame_id or FrameworkError if full.
   * @note Thread-safety: Lock-free SPSC ring buffer push.
   */
  [[nodiscard]] auto
  submit_stream_point (float x, float y) noexcept -> Result<uint64_t>;

  /**
   * @brief Enqueues a lock-free viewport pan/zoom update command.
   * @param x_min Minimum X domain coordinate.
   * @param x_max Maximum X domain coordinate.
   * @param y_min Minimum Y domain coordinate.
   * @param y_max Maximum Y domain coordinate.
   * @pre RenderLoop is running and x_min < x_max, y_min < y_max.
   * @post Enqueues viewport change and frame request without blocking.
   * @returns Result<uint64_t> scheduled frame_id or FrameworkError if full.
   * @note Thread-safety: Lock-free SPSC ring buffer push.
   */
  [[nodiscard]] auto
  submit_viewport (double x_min, double x_max, double y_min,
                   double y_max) noexcept -> Result<uint64_t>;

  /**
   * @brief Enqueues an explicit frame render request on the lock-free queue.
   * @pre RenderLoop is running.
   * @post Worker thread renders a frame and pushes CompletedFrame.
   * @returns Result<uint64_t> scheduled frame_id or FrameworkError if full.
   * @note Thread-safety: Lock-free SPSC ring buffer push.
   */
  [[nodiscard]] auto request_frame () noexcept -> Result<uint64_t>;

  /**
   * @brief Polls the lock-free completion queue for the next presented frame.
   * @pre None.
   * @post Pops oldest CompletedFrame if available.
   * @returns std::optional<CompletedFrame> or std::nullopt if empty.
   * @note Thread-safety: Lock-free SPSC ring buffer pop.
   */
  [[nodiscard]] auto
  poll_completed () noexcept -> std::optional<CompletedFrame>;

  /**
   * @brief Waits until the specified frame_id is presented on the completion
   * queue.
   * @param target_frame_id Frame identifier returned by submit_* or
   *        request_frame.
   * @param timeout_ms Maximum wait duration in milliseconds.
   * @pre target_frame_id > 0.
   * @post Returns CompletedFrame matching or exceeding target_frame_id.
   * @returns Result<CompletedFrame> or FrameworkError on timeout.
   * @note Thread-safety: Safe to call from consumer thread.
   */
  [[nodiscard]] auto wait_for_frame (uint64_t target_frame_id,
                                     uint64_t timeout_ms = 2000U) noexcept
      -> Result<CompletedFrame>;

private:
  struct Impl;
  explicit RenderLoop (std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_RENDER_LOOP_HPP */
