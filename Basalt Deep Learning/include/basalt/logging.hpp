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

#ifndef BASALT_LOGGING_HPP
#define BASALT_LOGGING_HPP

/* Pluggable experiment logging backends for basalt::Trainer.  */

#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <basalt/config.hpp>

namespace basalt
{

/* Abstract base interface for pluggable training loggers.  */
class BASALT_API Logger
{
public:
  Logger (void) = default;
  virtual ~Logger (void) noexcept;

  virtual void
  log_metrics (const std::map<std::string, double> &metrics,
               size_t step,
               size_t epoch) = 0;

  virtual void
  log_hyperparams (const std::map<std::string, std::string> &params);

  virtual void
  finalize (void);
};

/* Human-readable console logger writing to stdout or stderr.  */
class BASALT_API Console_Logger : public Logger
{
public:
  explicit Console_Logger (bool quiet = false);
  ~Console_Logger (void) noexcept override;

  void
  log_metrics (const std::map<std::string, double> &metrics,
               size_t step,
               size_t epoch) override;

private:
  bool quiet_;
  mutable std::mutex mutex_;
};

/* Comma-separated values file logger for tabular metric analysis.  */
class BASALT_API CSV_Logger : public Logger
{
public:
  explicit CSV_Logger (const std::string &file_path);
  ~CSV_Logger (void) noexcept override;

  void
  log_metrics (const std::map<std::string, double> &metrics,
               size_t step,
               size_t epoch) override;

  void
  finalize (void) override;

private:
  std::string path_;
  std::ofstream out_;
  std::vector<std::string> columns_;
  bool header_written_;
  mutable std::mutex mutex_;
};

/* Structured JSON Lines (.jsonl) logger for programmatic experiment
   tracking.  */
class BASALT_API JSON_Lines_Logger : public Logger
{
public:
  explicit JSON_Lines_Logger (const std::string &file_path);
  ~JSON_Lines_Logger (void) noexcept override;

  void
  log_metrics (const std::map<std::string, double> &metrics,
               size_t step,
               size_t epoch) override;

  void
  log_hyperparams (const std::map<std::string, std::string> &params) override;

  void
  finalize (void) override;

private:
  std::string path_;
  std::ofstream out_;
  mutable std::mutex mutex_;
};

/* In-memory logger storing metric history for inspection in tests.  */
class BASALT_API Memory_Logger : public Logger
{
public:
  struct Record
  {
    size_t step = 0;
    size_t epoch = 0;
    std::map<std::string, double> metrics;
  };

  Memory_Logger (void) = default;
  ~Memory_Logger (void) noexcept override;

  void
  log_metrics (const std::map<std::string, double> &metrics,
               size_t step,
               size_t epoch) override;

  const std::vector<Record> &
  history (void) const noexcept
  {
    return history_;
  }

private:
  std::vector<Record> history_;
  mutable std::mutex mutex_;
};

} /* namespace basalt */

#endif /* BASALT_LOGGING_HPP */
