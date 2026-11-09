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

#ifndef BELLOWS_IO_ARTIFACT_HPP
#define BELLOWS_IO_ARTIFACT_HPP

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <bellows/core/concepts.hpp>

namespace bellows
{

inline constexpr std::uint32_t artifact_magic = 0x53574C42U; // "BLWS"
inline constexpr std::uint32_t artifact_format_version = 1U;

// Compute 64-bit FNV-1a checksum over a byte buffer.
std::uint64_t
compute_fnv1a_checksum (const char *data, std::size_t length) noexcept;

// Serialise a model into a versioned, checksummed binary stream.
template <serialisable Model>
void
save_artifact (const Model &model,
                std::ostream &out,
                std::uint32_t model_tag = 1U)
{
  std::ostringstream payload_stream (std::ios::binary);
  model.write_state (payload_stream);
  std::string payload = payload_stream.str ();

  std::uint32_t magic = artifact_magic;
  std::uint32_t version = artifact_format_version;
  std::uint64_t length = static_cast<std::uint64_t> (payload.size ());
  std::uint64_t checksum =
    compute_fnv1a_checksum (payload.data (), payload.size ());

  out.write (reinterpret_cast<const char *> (&magic), sizeof (magic));
  out.write (reinterpret_cast<const char *> (&version), sizeof (version));
  out.write (reinterpret_cast<const char *> (&model_tag), sizeof (model_tag));
  out.write (reinterpret_cast<const char *> (&length), sizeof (length));
  out.write (reinterpret_cast<const char *> (&checksum), sizeof (checksum));
  out.write (payload.data (), static_cast<std::streamsize> (payload.size ()));
}

// Deserialise and verify a model from a versioned binary stream.
template <serialisable Model>
std::uint32_t
load_artifact (Model &model, std::istream &in)
{
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t model_tag = 0;
  std::uint64_t length = 0;
  std::uint64_t expected_checksum = 0;

  in.read (reinterpret_cast<char *> (&magic), sizeof (magic));
  in.read (reinterpret_cast<char *> (&version), sizeof (version));
  in.read (reinterpret_cast<char *> (&model_tag), sizeof (model_tag));
  in.read (reinterpret_cast<char *> (&length), sizeof (length));
  in.read (reinterpret_cast<char *> (&expected_checksum),
            sizeof (expected_checksum));

  if (!in || magic != artifact_magic)
    {
      throw std::runtime_error (
        "bellows::load_artifact: invalid artifact magic header");
    }
  if (version != artifact_format_version)
    {
      throw std::runtime_error (
        "bellows::load_artifact: unsupported artifact format version");
    }

  std::string payload (static_cast<std::size_t> (length), '\0');
  in.read (payload.data (), static_cast<std::streamsize> (length));
  if (!in)
    {
      throw std::runtime_error (
        "bellows::load_artifact: truncated artifact payload");
    }

  std::uint64_t actual_checksum =
    compute_fnv1a_checksum (payload.data (), payload.size ());
  if (actual_checksum != expected_checksum)
    {
      throw std::runtime_error (
        "bellows::load_artifact: artifact checksum mismatch");
    }

  std::istringstream payload_stream (payload, std::ios::binary);
  model.read_state (payload_stream);
  return model_tag;
}

// Save a serialisable model directly to a filesystem path.
template <serialisable Model>
void
save_artifact_file (const Model &model,
                    const std::string &path,
                    std::uint32_t model_tag = 1U)
{
  std::ofstream out (path, std::ios::binary);
  if (!out)
    {
      throw std::runtime_error (
        "bellows::save_artifact_file: cannot open output file: " + path);
    }
  save_artifact (model, out, model_tag);
}

// Load a serialisable model directly from a filesystem path.
template <serialisable Model>
std::uint32_t
load_artifact_file (Model &model, const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    {
      throw std::runtime_error (
        "bellows::load_artifact_file: cannot open input file: " + path);
    }
  return load_artifact (model, in);
}

} // namespace bellows

#endif // BELLOWS_IO_ARTIFACT_HPP
