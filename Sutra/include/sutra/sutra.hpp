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

#ifndef SUTRA_SUTRA_HPP
#define SUTRA_SUTRA_HPP

#include <sutra/checkpoint.hpp>
#include <sutra/error.hpp>
#include <sutra/gpu.hpp>
#include <sutra/llm_backends.hpp>
#include <sutra/memory.hpp>
#include <sutra/observability.hpp>
#include <sutra/planner.hpp>
#include <sutra/runtime.hpp>
#include <sutra/scheduler.hpp>
#include <sutra/tool.hpp>
#include <sutra/workflow.hpp>

#include <string_view>

namespace sutra::inline v1
{

/**
 * @brief Returns the semantic version string of the Sutra library.
 * @pre None.
 * @post State is unchanged.
 * @returns Compile-time version string ("1.0.0").
 * @note Thread-safety: Pure constexpr function; thread-safe.
 */
[[nodiscard]] constexpr auto
version_string () noexcept -> std::string_view
{
  return "1.0.0";
}

} /* namespace sutra::inline v1 */

#endif /* SUTRA_SUTRA_HPP */
