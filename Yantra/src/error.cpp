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

#include <yantra/error.hpp>
#include <yantra/observability.hpp>

#include <utility>

namespace yantra::inline v1
{

Error::Error (ErrorVariant domain, std::string message) noexcept
    : domain_ (std::move (domain)), message_ (std::move (message))
{
  Logger::log (LogLevel::Error, category_name (), message_);
}

auto
Error::tracking (TrackingErrorCode code, std::string run_id,
                 std::string message) noexcept -> Error
{
  return Error (TrackingError{ code, std::move (run_id) },
                std::move (message));
}

auto
Error::registry (RegistryErrorCode code, std::string model_name,
                 uint64_t version, std::string message) noexcept -> Error
{
  return Error (RegistryError{ code, std::move (model_name), version },
                std::move (message));
}

auto
Error::dag (DagErrorCode code, std::string task_id,
            std::string message) noexcept -> Error
{
  return Error (DagError{ code, std::move (task_id) }, std::move (message));
}

auto
Error::index (IndexErrorCode code, std::size_t expected_dim,
              std::size_t actual_dim, std::string message) noexcept -> Error
{
  return Error (IndexError{ code, expected_dim, actual_dim },
                std::move (message));
}

auto
Error::fairness (FairnessErrorCode code, uint32_t group_id,
                 std::string message) noexcept -> Error
{
  return Error (FairnessError{ code, group_id }, std::move (message));
}

auto
Error::variant () const noexcept -> const ErrorVariant &
{
  return domain_;
}

auto
Error::message () const noexcept -> std::string_view
{
  return message_;
}

auto
Error::category_name () const noexcept -> std::string_view
{
  if (std::holds_alternative<TrackingError> (domain_))
    {
      return "Tracking";
    }
  if (std::holds_alternative<RegistryError> (domain_))
    {
      return "Registry";
    }
  if (std::holds_alternative<DagError> (domain_))
    {
      return "Dag";
    }
  if (std::holds_alternative<IndexError> (domain_))
    {
      return "Index";
    }
  return "Fairness";
}

auto
Error::is_tracking_error () const noexcept -> bool
{
  return std::holds_alternative<TrackingError> (domain_);
}

auto
Error::is_registry_error () const noexcept -> bool
{
  return std::holds_alternative<RegistryError> (domain_);
}

auto
Error::is_dag_error () const noexcept -> bool
{
  return std::holds_alternative<DagError> (domain_);
}

auto
Error::is_index_error () const noexcept -> bool
{
  return std::holds_alternative<IndexError> (domain_);
}

auto
Error::is_fairness_error () const noexcept -> bool
{
  return std::holds_alternative<FairnessError> (domain_);
}

} /* namespace yantra::inline v1 */
