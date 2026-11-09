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

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include <bellows/bellows.hpp>

namespace bellows
{

const char *
version_string () noexcept
{
  return "0.1.0";
}

std::string
package_version_banner (const std::string &program_name)
{
  std::ostringstream oss;
  oss << program_name << " (Bellows) " << version_string () << "\n"
      << "Copyright (C) 2026 Kival\n"
      << "License GPLv3+: GNU GPL version 3 or later "
          "<https://gnu.org/licenses/gpl.html>.\n"
      << "This is free software: you are free to change and redistribute it.\n"
      << "There is NO WARRANTY, to the extent permitted by law.\n";
  return oss.str ();
}

std::uint64_t
compute_fnv1a_checksum (const char *data, std::size_t length) noexcept
{
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t i = 0; i < length; ++i)
    {
      unsigned char byte = static_cast<unsigned char> (data[i]);
      hash ^= static_cast<std::uint64_t> (byte);
      hash *= 1099511628211ULL;
    }
  return hash;
}

} // namespace bellows
