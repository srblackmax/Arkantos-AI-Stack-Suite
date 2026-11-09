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

#ifndef OBSIDIAN_SAFETENSORS_HPP
#define OBSIDIAN_SAFETENSORS_HPP

/**
 * @file safetensors.hpp
 * @brief Zero-copy POSIX mmap(2) Hugging Face SafeTensors (.safetensors)
 *        reader, writer, and basalt::nn::Module checkpoint serializer.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <basalt/basalt.hpp>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/tensor_bridge.hpp>

namespace obsidian
{

/**
 * @brief Element data types defined by the SafeTensors binary specification.
 */
enum class SafeTensors_Dtype : uint8_t
{
  F32 = 0,
  F64,
  I64,
  I32,
  F16,
  BF16,
  I16,
  I8,
  U8,
  BOOL
};

/**
 * @brief Descriptor for a single tensor stored in a SafeTensors file.
 */
struct OBSIDIAN_API SafeTensor_Entry
{
  std::string name;
  SafeTensors_Dtype dtype = SafeTensors_Dtype::F32;
  Shape_Vector shape;
  size_t data_offset_begin = 0;
  size_t data_offset_end = 0;
};

/**
 * @brief Memory-mapped read-only SafeTensors (.safetensors) container with
 *        strict header/offset bounds verification and zero-copy byte views.
 */
class OBSIDIAN_API SafeTensors_File
{
public:
  /**
   * @brief Construct an unmapped SafeTensors_File handle.
   * @pre None.
   * @complexity O(1) time and space.
   */
  SafeTensors_File (void) noexcept = default;

  /**
   * @brief Unmap the underlying POSIX memory mapping if active.
   * @pre None.
   * @complexity O(1) time and space.
   */
  ~SafeTensors_File (void) noexcept;

  SafeTensors_File (const SafeTensors_File &) = delete;
  SafeTensors_File &operator= (const SafeTensors_File &) = delete;

  SafeTensors_File (SafeTensors_File &&other) noexcept;
  SafeTensors_File &operator= (SafeTensors_File &&other) noexcept;

  /**
   * @brief Memory-map and validate a .safetensors file at PATH.
   * @pre PATH must refer to a readable, spec-compliant .safetensors file.
   * @complexity O(H) time in JSON header byte length H; O(1) for tensor data.
   */
  static Result<SafeTensors_File>
  open (const std::string &path);

  /**
   * @brief Return the number of tensors present in the file.
   * @pre None.
   * @complexity O(1) time and space.
   */
  size_t
  num_tensors (void) const noexcept
  {
    return entries_.size ();
  }

  /**
   * @brief Return true if a tensor named NAME exists in the file.
   * @pre None.
   * @complexity O(log T) time in tensor count T.
   */
  bool
  contains (std::string_view name) const noexcept;

  /**
   * @brief Return all tensor names in lexicographical order.
   * @pre None.
   * @complexity O(T) time and space in tensor count T.
   */
  std::vector<std::string>
  tensor_names (void) const;

  /**
   * @brief Return the __metadata__ string key-value dictionary.
   * @pre None.
   * @complexity O(1) time and space.
   */
  const std::map<std::string, std::string> &
  metadata (void) const noexcept
  {
    return metadata_;
  }

  /**
   * @brief Retrieve the metadata descriptor for tensor NAME.
   * @pre NAME must exist in the file.
   * @complexity O(log T) time.
   */
  Result<SafeTensor_Entry>
  get_entry (std::string_view name) const;

  /**
   * @brief Return a zero-copy byte span into the mmap region for tensor NAME.
   * @pre File is mapped and NAME exists.
   * @complexity O(log T) time and O(1) space.
   */
  Result<std::span<const std::byte>>
  get_raw_bytes (std::string_view name) const;

  /**
   * @brief Materialize tensor NAME as a Basalt Tensor on DEVICE (converting
   *        F16 and BF16 weights to Float32 automatically).
   * @pre File is mapped and NAME exists.
   * @complexity O(numel) time and space.
   */
  Result<Tensor>
  load_tensor (std::string_view name,
               Device device = Device::cpu ()) const;

  /**
   * @brief Load all tensors in the file as a Basalt State_Dict.
   * @pre File is mapped.
   * @complexity O(total_elements) time and space.
   */
  Result<basalt::nn::State_Dict>
  load_all_tensors (Device device = Device::cpu ()) const;

private:
  void
  close_mapping (void) noexcept;

  const std::byte *mmap_base_ = nullptr;
  size_t mmap_size_ = 0;
  size_t data_section_offset_ = 0;
  int fd_ = -1;
  std::map<std::string, SafeTensor_Entry, std::less<>> entries_;
  std::map<std::string, std::string> metadata_;
};

/**
 * @brief Serialize a dictionary of named Basalt Tensors and optional string
 *        metadata to a spec-compliant Hugging Face .safetensors file.
 * @pre PATH is writable and all tensors in TENSORS are defined.
 * @complexity O(total_bytes) time and O(header_bytes) auxiliary space.
 */
OBSIDIAN_API Result<void>
save_safetensors (const std::string &path,
                  const basalt::nn::State_Dict &tensors,
                  const std::map<std::string, std::string> &metadata = {});

/**
 * @brief Save all parameters and buffers of MODULE to a .safetensors file.
 * @pre PATH is writable.
 * @complexity O(total_parameter_bytes) time.
 */
OBSIDIAN_API Result<void>
save_module_safetensors (
  const basalt::nn::Module &module,
  const std::string &path,
  const std::map<std::string, std::string> &metadata = {});

/**
 * @brief Load parameters and buffers from a .safetensors file at PATH into
 *        MODULE using memory-mapped IO.
 * @pre PATH is a valid .safetensors file; when STRICT is true, every
 *      parameter and buffer in MODULE must exist in the file with matching
 *      shape.
 * @complexity O(total_parameter_elements) time.
 */
OBSIDIAN_API Result<void>
load_module_safetensors (basalt::nn::Module &module,
                         const std::string &path,
                         bool strict = true);

} /* namespace obsidian */

#endif /* OBSIDIAN_SAFETENSORS_HPP */
