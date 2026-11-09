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

#ifndef DARSHANA_EXPORT_HPP
#define DARSHANA_EXPORT_HPP

#include <darshana/error.hpp>
#include <darshana/memory.hpp>
#include <darshana/spec.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <string_view>

namespace darshana::inline v1
{

/**
 * @brief Supported publication-quality vector export formats.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class ExportFormat : uint8_t
{
  Svg = 0,
  Pdf = 1
};

/**
 * @brief Summary telemetry and metadata returned by a completed export.
 * @pre None.
 * @post Contains format, output file path, byte size, duration, and point
 *       count.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct ExportResult
{
  ExportFormat format = ExportFormat::Svg;
  std::string file_path;
  std::size_t bytes_written = 0U;
  uint64_t elapsed_ns = 0U;
  std::size_t points_exported = 0U;
};

/**
 * @brief Asynchronous completion callback type for non-blocking exports.
 */
using ExportCallback = std::function<void (const Result<ExportResult> &)>;

/**
 * @brief Generates an in-memory SVG 1.1 document from a Spec and dataset.
 * @param spec Validated declarative plot specification.
 * @param snapshot Optional static SnapshotData pointer (may be nullptr).
 * @param stream Optional StreamingBuffer pointer (may be nullptr).
 * @pre spec.validate() succeeds.
 * @post Returns complete standalone SVG document string.
 * @returns Result<std::string> containing SVG markup or Error.
 * @note Thread-safety: Pure function; parallelized internally across CPU
 *       cores for large point counts.
 */
[[nodiscard]] auto
render_svg_string (const Spec &spec, const SnapshotData *snapshot = nullptr,
                   const StreamingBuffer *stream
                   = nullptr) noexcept -> Result<std::string>;

/**
 * @brief Generates an in-memory PDF 1.4 vector document from a Spec and
 * dataset.
 * @param spec Validated declarative plot specification.
 * @param snapshot Optional static SnapshotData pointer (may be nullptr).
 * @param stream Optional StreamingBuffer pointer (may be nullptr).
 * @pre spec.validate() succeeds.
 * @post Returns complete PDF 1.4 document with cross-reference table.
 * @returns Result<std::string> containing PDF byte stream or Error.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto
render_pdf_string (const Spec &spec, const SnapshotData *snapshot = nullptr,
                   const StreamingBuffer *stream
                   = nullptr) noexcept -> Result<std::string>;

/**
 * @brief Synchronously exports a plot to disk in SVG or PDF format.
 * @param spec Validated declarative plot specification.
 * @param file_path Destination filesystem path.
 * @param format Target ExportFormat (Svg or Pdf).
 * @param snapshot Optional static SnapshotData pointer.
 * @param stream Optional StreamingBuffer pointer.
 * @pre spec.validate() succeeds and !file_path.empty().
 * @post Output file is written atomically and flushed.
 * @returns Result<ExportResult> or ExportError on I/O or format failure.
 * @note Thread-safety: Safe for concurrent calls writing to distinct paths.
 */
[[nodiscard]] auto
export_file_sync (const Spec &spec, std::string_view file_path,
                  ExportFormat format = ExportFormat::Svg,
                  const SnapshotData *snapshot = nullptr,
                  const StreamingBuffer *stream
                  = nullptr) noexcept -> Result<ExportResult>;

/**
 * @brief Asynchronously exports a plot to disk without blocking the caller.
 * @param spec Plot specification copied into the background export task.
 * @param file_path Destination filesystem path.
 * @param format Target ExportFormat (Svg or Pdf).
 * @param snapshot Optional static SnapshotData copied for background export.
 * @param stream Optional StreamingBuffer copied for background export.
 * @param callback Optional completion callback invoked when export finishes.
 * @pre !file_path.empty().
 * @post Spawns non-blocking background export task and returns immediately.
 * @returns std::future<Result<ExportResult>> resolving upon completion.
 * @note Thread-safety: Non-blocking and thread-safe; never stalls interactive
 *       rendering.
 */
[[nodiscard]] auto
export_file_async (Spec spec, std::string file_path,
                   ExportFormat format = ExportFormat::Svg,
                   SnapshotData snapshot = {}, StreamingBuffer stream = {},
                   ExportCallback callback
                   = nullptr) noexcept -> std::future<Result<ExportResult>>;

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_EXPORT_HPP */
