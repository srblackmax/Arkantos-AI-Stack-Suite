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

#include <obsidian/safetensors.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace obsidian
{

namespace
{

constexpr size_t k_max_header_size = 100ull * 1024ull * 1024ull;

size_t
safetensors_dtype_size (SafeTensors_Dtype dt) noexcept
{
  switch (dt)
    {
    case SafeTensors_Dtype::F64:
    case SafeTensors_Dtype::I64:
      return 8;
    case SafeTensors_Dtype::F32:
    case SafeTensors_Dtype::I32:
      return 4;
    case SafeTensors_Dtype::F16:
    case SafeTensors_Dtype::BF16:
    case SafeTensors_Dtype::I16:
      return 2;
    case SafeTensors_Dtype::I8:
    case SafeTensors_Dtype::U8:
    case SafeTensors_Dtype::BOOL:
      return 1;
    }
  return 0;
}

Result<SafeTensors_Dtype>
parse_dtype_string (std::string_view s) noexcept
{
  if (s == "F32")
    return SafeTensors_Dtype::F32;
  if (s == "F64")
    return SafeTensors_Dtype::F64;
  if (s == "I64")
    return SafeTensors_Dtype::I64;
  if (s == "I32")
    return SafeTensors_Dtype::I32;
  if (s == "F16")
    return SafeTensors_Dtype::F16;
  if (s == "BF16")
    return SafeTensors_Dtype::BF16;
  if (s == "I16")
    return SafeTensors_Dtype::I16;
  if (s == "I8")
    return SafeTensors_Dtype::I8;
  if (s == "U8")
    return SafeTensors_Dtype::U8;
  if (s == "BOOL")
    return SafeTensors_Dtype::BOOL;
  return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                         "unsupported SafeTensors dtype string");
}

const char *
dtype_to_safetensors_str (Dtype dt) noexcept
{
  switch (dt)
    {
    case Dtype::Float32:
      return "F32";
    case Dtype::Float64:
      return "F64";
    case Dtype::Int64:
      return "I64";
    case Dtype::Int32:
      return "I32";
    default:
      return "F32";
    }
}

/* Convert IEEE 754 binary16 (F16) to binary32 float.  */
float
f16_to_f32 (uint16_t h) noexcept
{
  uint32_t sign = static_cast<uint32_t> (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;
  uint32_t bits = 0;

  if (exp == 0)
    {
      if (mant == 0)
        {
          bits = sign;
        }
      else
        {
          exp = 1;
          while ((mant & 0x0400u) == 0)
            {
              mant <<= 1;
              --exp;
            }
          mant &= 0x03FFu;
          bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    }
  else if (exp == 0x1Fu)
    {
      bits = sign | 0x7F800000u | (mant << 13);
    }
  else
    {
      bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

  float out = 0.0f;
  std::memcpy (&out, &bits, sizeof (float));
  return out;
}

/* Convert bfloat16 (BF16) to binary32 float.  */
float
bf16_to_f32 (uint16_t b) noexcept
{
  uint32_t bits = static_cast<uint32_t> (b) << 16;
  float out = 0.0f;
  std::memcpy (&out, &bits, sizeof (float));
  return out;
}

/* Minimal, zero-dependency recursive-descent parser for SafeTensors JSON
   headers.  */
class Json_Header_Parser
{
public:
  explicit Json_Header_Parser (std::string_view src) noexcept
    : src_ (src),
      pos_ (0)
  {
  }

  Result<void>
  parse_root (
    std::map<std::string, SafeTensor_Entry, std::less<>> &entries,
    std::map<std::string, std::string> &metadata)
  {
    skip_ws ();
    if (!consume ('{'))
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "SafeTensors header must start with '{'");
      }
    skip_ws ();
    if (consume ('}'))
      {
        return {};
      }

    while (true)
      {
        auto key_res = parse_string ();
        if (!key_res.has_value ())
          {
            return std::unexpected (key_res.error ());
          }
        skip_ws ();
        if (!consume (':'))
          {
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "expected ':' after key in SafeTensors");
          }
        skip_ws ();

        if (*key_res == "__metadata__")
          {
            auto meta_res = parse_metadata_object (metadata);
            if (!meta_res.has_value ())
              {
                return meta_res;
              }
          }
        else
          {
            auto entry_res = parse_tensor_entry (*key_res);
            if (!entry_res.has_value ())
              {
                return std::unexpected (entry_res.error ());
              }
            entries.emplace (*key_res, std::move (*entry_res));
          }

        skip_ws ();
        if (consume (','))
          {
            skip_ws ();
            continue;
          }
        if (consume ('}'))
          {
            break;
          }
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected ',' or '}' in SafeTensors root");
      }
    return {};
  }

private:
  void
  skip_ws (void) noexcept
  {
    while (pos_ < src_.size ())
      {
        char c = src_[pos_];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
          {
            ++pos_;
          }
        else
          {
            break;
          }
      }
  }

  bool
  consume (char expected) noexcept
  {
    if (pos_ < src_.size () && src_[pos_] == expected)
      {
        ++pos_;
        return true;
      }
    return false;
  }

  Result<std::string>
  parse_string (void)
  {
    skip_ws ();
    if (!consume ('"'))
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected '\"' to start JSON string");
      }
    std::string out;
    while (pos_ < src_.size ())
      {
        char c = src_[pos_++];
        if (c == '"')
          {
            return out;
          }
        if (c == '\\')
          {
            if (pos_ >= src_.size ())
              {
                return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                       "unterminated JSON escape");
              }
            char esc = src_[pos_++];
            switch (esc)
              {
              case '"':
              case '\\':
              case '/':
                out.push_back (esc);
                break;
              case 'b':
                out.push_back ('\b');
                break;
              case 'f':
                out.push_back ('\f');
                break;
              case 'n':
                out.push_back ('\n');
                break;
              case 'r':
                out.push_back ('\r');
                break;
              case 't':
                out.push_back ('\t');
                break;
              default:
                out.push_back (esc);
                break;
              }
          }
        else
          {
            out.push_back (c);
          }
      }
    return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                           "unterminated JSON string literal");
  }

  Result<size_t>
  parse_uint (void)
  {
    skip_ws ();
    if (pos_ >= src_.size () || src_[pos_] < '0' || src_[pos_] > '9')
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected unsigned integer in JSON");
      }
    size_t val = 0;
    while (pos_ < src_.size () && src_[pos_] >= '0' && src_[pos_] <= '9')
      {
        size_t digit = static_cast<size_t> (src_[pos_] - '0');
        if (val > (std::numeric_limits<size_t>::max () - digit) / 10)
          {
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "integer overflow in SafeTensors header");
          }
        val = val * 10 + digit;
        ++pos_;
      }
    return val;
  }

  Result<std::vector<size_t>>
  parse_uint_array (void)
  {
    skip_ws ();
    if (!consume ('['))
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected '[' for integer array");
      }
    std::vector<size_t> arr;
    skip_ws ();
    if (consume (']'))
      {
        return arr;
      }
    while (true)
      {
        auto v = parse_uint ();
        if (!v.has_value ())
          {
            return std::unexpected (v.error ());
          }
        arr.push_back (*v);
        skip_ws ();
        if (consume (','))
          {
            skip_ws ();
            continue;
          }
        if (consume (']'))
          {
            break;
          }
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected ',' or ']' in integer array");
      }
    return arr;
  }

  Result<void>
  parse_metadata_object (std::map<std::string, std::string> &metadata)
  {
    skip_ws ();
    if (!consume ('{'))
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected '{' for __metadata__");
      }
    skip_ws ();
    if (consume ('}'))
      {
        return {};
      }
    while (true)
      {
        auto k = parse_string ();
        if (!k.has_value ())
          {
            return std::unexpected (k.error ());
          }
        skip_ws ();
        if (!consume (':'))
          {
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "expected ':' in __metadata__");
          }
        auto v = parse_string ();
        if (!v.has_value ())
          {
            return std::unexpected (v.error ());
          }
        metadata[*k] = *v;
        skip_ws ();
        if (consume (','))
          {
            skip_ws ();
            continue;
          }
        if (consume ('}'))
          {
            break;
          }
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected ',' or '}' in __metadata__");
      }
    return {};
  }

  Result<SafeTensor_Entry>
  parse_tensor_entry (const std::string &tensor_name)
  {
    skip_ws ();
    if (!consume ('{'))
      {
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected '{' for tensor descriptor");
      }
    SafeTensor_Entry entry;
    entry.name = tensor_name;
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offsets = false;

    skip_ws ();
    while (!consume ('}'))
      {
        auto field = parse_string ();
        if (!field.has_value ())
          {
            return std::unexpected (field.error ());
          }
        skip_ws ();
        if (!consume (':'))
          {
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "expected ':' after tensor field");
          }
        skip_ws ();

        if (*field == "dtype")
          {
            auto dt_str = parse_string ();
            if (!dt_str.has_value ())
              {
                return std::unexpected (dt_str.error ());
              }
            auto dt = parse_dtype_string (*dt_str);
            if (!dt.has_value ())
              {
                return std::unexpected (dt.error ());
              }
            entry.dtype = *dt;
            has_dtype = true;
          }
        else if (*field == "shape")
          {
            auto sh = parse_uint_array ();
            if (!sh.has_value ())
              {
                return std::unexpected (sh.error ());
              }
            entry.shape = std::move (*sh);
            has_shape = true;
          }
        else if (*field == "data_offsets")
          {
            auto offs = parse_uint_array ();
            if (!offs.has_value () || offs->size () != 2)
              {
                return OBSIDIAN_ERROR (
                  Error_Code::Corrupted_Format,
                  "data_offsets must be a 2-element array");
              }
            entry.data_offset_begin = (*offs)[0];
            entry.data_offset_end = (*offs)[1];
            has_offsets = true;
          }
        else
          {
            return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                   "unknown field in SafeTensors descriptor");
          }

        skip_ws ();
        if (consume (','))
          {
            skip_ws ();
            continue;
          }
        if (consume ('}'))
          {
            break;
          }
        return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                               "expected ',' or '}' in tensor descriptor");
      }

    if (!has_dtype || !has_shape || !has_offsets)
      {
        return OBSIDIAN_ERROR (
          Error_Code::Corrupted_Format,
          "incomplete SafeTensors descriptor (missing dtype/shape/offsets)");
      }
    return entry;
  }

  std::string_view src_;
  size_t pos_;
};

std::string
escape_json_string (std::string_view s)
{
  std::string out;
  out.reserve (s.size () + 4);
  for (char c : s)
    {
      switch (c)
        {
        case '"':
          out.append ("\\\"");
          break;
        case '\\':
          out.append ("\\\\");
          break;
        case '\n':
          out.append ("\\n");
          break;
        case '\r':
          out.append ("\\r");
          break;
        case '\t':
          out.append ("\\t");
          break;
        default:
          out.push_back (c);
          break;
        }
    }
  return out;
}

} /* namespace */

SafeTensors_File::~SafeTensors_File (void) noexcept
{
  close_mapping ();
}

SafeTensors_File::SafeTensors_File (SafeTensors_File &&other) noexcept
  : mmap_base_ (other.mmap_base_),
    mmap_size_ (other.mmap_size_),
    data_section_offset_ (other.data_section_offset_),
    fd_ (other.fd_),
    entries_ (std::move (other.entries_)),
    metadata_ (std::move (other.metadata_))
{
  other.mmap_base_ = nullptr;
  other.mmap_size_ = 0;
  other.data_section_offset_ = 0;
  other.fd_ = -1;
}

SafeTensors_File &
SafeTensors_File::operator= (SafeTensors_File &&other) noexcept
{
  if (this != &other)
    {
      close_mapping ();
      mmap_base_ = other.mmap_base_;
      mmap_size_ = other.mmap_size_;
      data_section_offset_ = other.data_section_offset_;
      fd_ = other.fd_;
      entries_ = std::move (other.entries_);
      metadata_ = std::move (other.metadata_);

      other.mmap_base_ = nullptr;
      other.mmap_size_ = 0;
      other.data_section_offset_ = 0;
      other.fd_ = -1;
    }
  return *this;
}

void
SafeTensors_File::close_mapping (void) noexcept
{
  if (mmap_base_ != nullptr && mmap_size_ > 0)
    {
      ::munmap (const_cast<std::byte *> (mmap_base_), mmap_size_);
    }
  if (fd_ >= 0)
    {
      ::close (fd_);
    }
  mmap_base_ = nullptr;
  mmap_size_ = 0;
  data_section_offset_ = 0;
  fd_ = -1;
}

Result<SafeTensors_File>
SafeTensors_File::open (const std::string &path)
{
  int fd = ::open (path.c_str (), O_RDONLY);
  if (fd < 0)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open SafeTensors file: " + path);
    }

  struct stat st {};
  if (::fstat (fd, &st) != 0 || st.st_size < 8)
    {
      ::close (fd);
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "SafeTensors file too small (< 8 bytes)");
    }

  size_t file_size = static_cast<size_t> (st.st_size);
  void *mapped = ::mmap (nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED)
    {
      ::close (fd);
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "mmap failed on SafeTensors file");
    }

  SafeTensors_File file;
  file.mmap_base_ = static_cast<const std::byte *> (mapped);
  file.mmap_size_ = file_size;
  file.fd_ = fd;

  uint64_t header_len_u64 = 0;
  std::memcpy (&header_len_u64, file.mmap_base_, sizeof (uint64_t));
  size_t header_len = static_cast<size_t> (header_len_u64);

  if (header_len == 0 || header_len > k_max_header_size
      || header_len > file_size - 8)
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "invalid SafeTensors header byte length");
    }

  file.data_section_offset_ = 8 + header_len;
  std::string_view json_str (
    reinterpret_cast<const char *> (file.mmap_base_ + 8), header_len);

  Json_Header_Parser parser (json_str);
  auto parse_res = parser.parse_root (file.entries_, file.metadata_);
  if (!parse_res.has_value ())
    {
      return std::unexpected (parse_res.error ());
    }

  /* Validate every tensor's byte offsets and shape-to-byte-size match.  */
  for (const auto &kv : file.entries_)
    {
      const SafeTensor_Entry &e = kv.second;
      if (e.data_offset_begin > e.data_offset_end)
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "SafeTensors begin offset > end offset");
        }
      if (e.data_offset_end > file_size - file.data_section_offset_)
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "SafeTensors data offset exceeds file size");
        }

      size_t elem_count = 1;
      for (size_t dim : e.shape)
        {
          if (dim > 0
              && elem_count > std::numeric_limits<size_t>::max () / dim)
            {
              return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                                     "SafeTensors shape product overflow");
            }
          elem_count *= dim;
        }
      size_t elem_bytes = safetensors_dtype_size (e.dtype);
      size_t expected_bytes = elem_count * elem_bytes;
      size_t actual_bytes = e.data_offset_end - e.data_offset_begin;
      if (expected_bytes != actual_bytes)
        {
          return OBSIDIAN_ERROR (
            Error_Code::Shape_Mismatch,
            "SafeTensors byte span does not match shape * dtype_size");
        }
    }

  return file;
}

bool
SafeTensors_File::contains (std::string_view name) const noexcept
{
  return entries_.find (name) != entries_.end ();
}

std::vector<std::string>
SafeTensors_File::tensor_names (void) const
{
  std::vector<std::string> names;
  names.reserve (entries_.size ());
  for (const auto &kv : entries_)
    {
      names.push_back (kv.first);
    }
  return names;
}

Result<SafeTensor_Entry>
SafeTensors_File::get_entry (std::string_view name) const
{
  auto it = entries_.find (name);
  if (it == entries_.end ())
    {
      return OBSIDIAN_ERROR (Error_Code::Vocab_Not_Found,
                             "tensor not found in SafeTensors file");
    }
  return it->second;
}

Result<std::span<const std::byte>>
SafeTensors_File::get_raw_bytes (std::string_view name) const
{
  if (mmap_base_ == nullptr)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "SafeTensors_File is not mapped");
    }
  auto entry_res = get_entry (name);
  if (!entry_res.has_value ())
    {
      return std::unexpected (entry_res.error ());
    }
  const SafeTensor_Entry &e = *entry_res;
  const std::byte *start
    = mmap_base_ + data_section_offset_ + e.data_offset_begin;
  size_t len = e.data_offset_end - e.data_offset_begin;
  return std::span<const std::byte> (start, len);
}

Result<Tensor>
SafeTensors_File::load_tensor (std::string_view name, Device device) const
{
  auto entry_res = get_entry (name);
  if (!entry_res.has_value ())
    {
      return std::unexpected (entry_res.error ());
    }
  auto bytes_res = get_raw_bytes (name);
  if (!bytes_res.has_value ())
    {
      return std::unexpected (bytes_res.error ());
    }

  const SafeTensor_Entry &e = *entry_res;
  std::span<const std::byte> raw = *bytes_res;
  Shape_Vector shape = e.shape.empty () ? Shape_Vector {1} : e.shape;

  size_t numel = 1;
  for (size_t d : shape)
    {
      numel *= d;
    }

  Tensor out;
  switch (e.dtype)
    {
    case SafeTensors_Dtype::F32:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Float32));
        if (numel > 0)
          {
            std::memcpy (out.data_ptr<float> (), raw.data (),
                         numel * sizeof (float));
          }
        break;
      }
    case SafeTensors_Dtype::F64:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Float64));
        if (numel > 0)
          {
            std::memcpy (out.data_ptr<double> (), raw.data (),
                         numel * sizeof (double));
          }
        break;
      }
    case SafeTensors_Dtype::I64:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Int64));
        if (numel > 0)
          {
            std::memcpy (out.data_ptr<int64_t> (), raw.data (),
                         numel * sizeof (int64_t));
          }
        break;
      }
    case SafeTensors_Dtype::I32:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Int32));
        if (numel > 0)
          {
            std::memcpy (out.data_ptr<int32_t> (), raw.data (),
                         numel * sizeof (int32_t));
          }
        break;
      }
    case SafeTensors_Dtype::F16:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Float32));
        float *dst = out.data_ptr<float> ();
        const auto *src = reinterpret_cast<const uint16_t *> (raw.data ());
        for (size_t i = 0; i < numel; ++i)
          {
            dst[i] = f16_to_f32 (src[i]);
          }
        break;
      }
    case SafeTensors_Dtype::BF16:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Float32));
        float *dst = out.data_ptr<float> ();
        const auto *src = reinterpret_cast<const uint16_t *> (raw.data ());
        for (size_t i = 0; i < numel; ++i)
          {
            dst[i] = bf16_to_f32 (src[i]);
          }
        break;
      }
    default:
      {
        out = basalt::zeros (
          shape, Tensor_Options ().with_dtype (Dtype::Float32));
        float *dst = out.data_ptr<float> ();
        const auto *src = reinterpret_cast<const uint8_t *> (raw.data ());
        for (size_t i = 0; i < numel; ++i)
          {
            dst[i] = static_cast<float> (src[i]);
          }
        break;
      }
    }

  if (!device.is_cpu ())
    {
      out = out.to (device);
    }
  return out;
}

Result<basalt::nn::State_Dict>
SafeTensors_File::load_all_tensors (Device device) const
{
  basalt::nn::State_Dict dict;
  for (const auto &kv : entries_)
    {
      auto t_res = load_tensor (kv.first, device);
      if (!t_res.has_value ())
        {
          return std::unexpected (t_res.error ());
        }
      dict.emplace (kv.first, std::move (*t_res));
    }
  return dict;
}

Result<void>
save_safetensors (const std::string &path,
                  const basalt::nn::State_Dict &tensors,
                  const std::map<std::string, std::string> &metadata)
{
  struct Prepared_Tensor
  {
    std::string name;
    Tensor contig_cpu;
    const char *dtype_str = "F32";
    size_t begin_offset = 0;
    size_t end_offset = 0;
  };

  std::vector<Prepared_Tensor> prepared;
  prepared.reserve (tensors.size ());
  size_t running_offset = 0;

  for (const auto &kv : tensors)
    {
      if (!kv.second.defined ())
        {
          return OBSIDIAN_ERROR (
            Error_Code::Invalid_Argument,
            "cannot serialize undefined tensor: " + kv.first);
        }
      Tensor t = kv.second.detach ().cpu ().contiguous ();
      size_t elem_size = 4;
      switch (t.dtype ())
        {
        case Dtype::Float64:
        case Dtype::Int64:
          elem_size = 8;
          break;
        case Dtype::Float32:
        case Dtype::Int32:
          elem_size = 4;
          break;
        default:
          t = t.to (Dtype::Float32).contiguous ();
          elem_size = 4;
          break;
        }
      size_t nbytes = t.numel () * elem_size;
      Prepared_Tensor pt;
      pt.name = kv.first;
      pt.contig_cpu = std::move (t);
      pt.dtype_str = dtype_to_safetensors_str (pt.contig_cpu.dtype ());
      pt.begin_offset = running_offset;
      pt.end_offset = running_offset + nbytes;
      running_offset += nbytes;
      prepared.push_back (std::move (pt));
    }

  /* Build JSON header string.  */
  std::string header = "{";
  bool first_entry = true;

  if (!metadata.empty ())
    {
      header.append ("\"__metadata__\":{");
      bool first_meta = true;
      for (const auto &mkv : metadata)
        {
          if (!first_meta)
            {
              header.push_back (',');
            }
          first_meta = false;
          header.push_back ('"');
          header.append (escape_json_string (mkv.first));
          header.append ("\":\"");
          header.append (escape_json_string (mkv.second));
          header.push_back ('"');
        }
      header.push_back ('}');
      first_entry = false;
    }

  for (const auto &pt : prepared)
    {
      if (!first_entry)
        {
          header.push_back (',');
        }
      first_entry = false;
      header.push_back ('"');
      header.append (escape_json_string (pt.name));
      header.append ("\":{\"dtype\":\"");
      header.append (pt.dtype_str);
      header.append ("\",\"shape\":[");
      const auto &sh = pt.contig_cpu.shape ();
      for (size_t i = 0; i < sh.size (); ++i)
        {
          if (i > 0)
            {
              header.push_back (',');
            }
          header.append (std::to_string (sh[i]));
        }
      header.append ("],\"data_offsets\":[");
      header.append (std::to_string (pt.begin_offset));
      header.push_back (',');
      header.append (std::to_string (pt.end_offset));
      header.append ("]}");
    }
  header.push_back ('}');

  /* Pad JSON header with spaces to 8-byte alignment for SIMD/mmap safety.  */
  while ((header.size () % 8) != 0)
    {
      header.push_back (' ');
    }

  std::ofstream ofs (path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open ())
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open file for writing: " + path);
    }

  uint64_t header_len = static_cast<uint64_t> (header.size ());
  ofs.write (reinterpret_cast<const char *> (&header_len), sizeof (uint64_t));
  ofs.write (header.data (), static_cast<std::streamsize> (header.size ()));

  for (const auto &pt : prepared)
    {
      size_t nbytes = pt.end_offset - pt.begin_offset;
      if (nbytes > 0)
        {
          ofs.write (
            reinterpret_cast<const char *> (pt.contig_cpu.raw_data_ptr ()),
            static_cast<std::streamsize> (nbytes));
        }
    }

  if (!ofs.good ())
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "error writing SafeTensors payload: " + path);
    }
  return {};
}

Result<void>
save_module_safetensors (
  const basalt::nn::Module &module,
  const std::string &path,
  const std::map<std::string, std::string> &metadata)
{
  return obsidian::save_safetensors (path, module.state_dict (), metadata);
}

Result<void>
load_module_safetensors (basalt::nn::Module &module,
                         const std::string &path,
                         bool strict)
{
  auto st_res = SafeTensors_File::open (path);
  if (!st_res.has_value ())
    {
      return std::unexpected (st_res.error ());
    }
  const SafeTensors_File &st = *st_res;

  No_Grad_Guard no_grad;
  auto params = module.named_parameters (true);
  auto bufs = module.named_buffers (true);

  auto load_into_target = [&] (const std::string &name,
                               Tensor &target) -> Result<void>
    {
      if (!st.contains (name))
        {
          if (strict)
            {
              return OBSIDIAN_ERROR (
                Error_Code::Vocab_Not_Found,
                "missing tensor in SafeTensors checkpoint: " + name);
            }
          return {};
        }
      auto loaded_res = st.load_tensor (name, target.device ());
      if (!loaded_res.has_value ())
        {
          return std::unexpected (loaded_res.error ());
        }
      if (loaded_res->shape () != target.shape ())
        {
          return OBSIDIAN_ERROR (
            Error_Code::Shape_Mismatch,
            "shape mismatch loading SafeTensors parameter: " + name);
        }
      Tensor src = (loaded_res->dtype () == target.dtype ())
                     ? *loaded_res
                     : loaded_res->to (target.dtype ());
      target.copy_ (src);
      return {};
    };

  for (auto &kv : params)
    {
      auto r = load_into_target (kv.first, kv.second);
      if (!r.has_value ())
        {
          return r;
        }
    }
  for (auto &kv : bufs)
    {
      auto r = load_into_target (kv.first, kv.second);
      if (!r.has_value ())
        {
          return r;
        }
    }
  return {};
}

} /* namespace obsidian */
