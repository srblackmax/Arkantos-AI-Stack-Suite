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

#include <basalt/serialization.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <bellows/io/artifact.hpp>
#include <basalt/error.hpp>
#include <basalt/version.h>

namespace basalt::io
{

namespace
{

size_t
tensor_nbytes (const Tensor &t) noexcept
{
  if (!t.defined ())
    return 0;
  return t.numel () * dtype_size (t.dtype ());
}

template <typename T>
void
write_pod (std::ostream &os, const T &val)
{
  os.write (reinterpret_cast<const char *> (&val), sizeof (T));
}

template <typename T>
T
read_pod (std::istream &is)
{
  T val{};
  is.read (reinterpret_cast<char *> (&val), sizeof (T));
  if (!is)
    BASALT_THROW (IO_Error, "Unexpected EOF reading checkpoint");
  return val;
}

void
write_string (std::ostream &os, const std::string &s)
{
  uint64_t len = static_cast<uint64_t> (s.size ());
  write_pod (os, len);
  if (len > 0)
    os.write (s.data (), static_cast<std::streamsize> (len));
}

std::string
read_string (std::istream &is)
{
  uint64_t len = read_pod<uint64_t> (is);
  std::string s (static_cast<size_t> (len), '\0');
  if (len > 0)
    {
      is.read (s.data (), static_cast<std::streamsize> (len));
      if (!is)
        BASALT_THROW (IO_Error,
                      "Unexpected EOF reading string from checkpoint");
    }
  return s;
}

void
write_tensor_map (std::ostream &os, const std::map<std::string, Tensor> &m)
{
  uint64_t count = static_cast<uint64_t> (m.size ());
  write_pod (os, count);
  for (const auto &kv : m)
    {
      write_string (os, kv.first);
      Tensor cpu_t = kv.second.defined ()
                     ? kv.second.to (Device::cpu ()).contiguous ()
                     : Tensor ();
      uint32_t dt = static_cast<uint32_t> (cpu_t.dtype ());
      write_pod (os, dt);
      uint64_t ndim = static_cast<uint64_t> (cpu_t.ndim ());
      write_pod (os, ndim);
      for (size_t d : cpu_t.shape ())
        write_pod (os, static_cast<uint64_t> (d));
      uint64_t nbytes = static_cast<uint64_t> (tensor_nbytes (cpu_t));
      write_pod (os, nbytes);
      if (nbytes > 0)
        os.write (static_cast<const char *> (cpu_t.raw_data_ptr ()),
                  static_cast<std::streamsize> (nbytes));
    }
}

std::map<std::string, Tensor>
read_tensor_map (std::istream &is)
{
  uint64_t count = read_pod<uint64_t> (is);
  std::map<std::string, Tensor> m;
  for (uint64_t i = 0; i < count; ++i)
    {
      std::string name = read_string (is);
      uint32_t dt_raw = read_pod<uint32_t> (is);
      Dtype dt = static_cast<Dtype> (dt_raw);
      uint64_t ndim = read_pod<uint64_t> (is);
      Shape_Vector shape (static_cast<size_t> (ndim));
      for (uint64_t d = 0; d < ndim; ++d)
        shape[static_cast<size_t> (d)] =
          static_cast<size_t> (read_pod<uint64_t> (is));
      uint64_t nbytes = read_pod<uint64_t> (is);
      Tensor t = empty (
        shape, Tensor_Options ().with_dtype (dt).with_device (Device::cpu ()));
      if (nbytes != static_cast<uint64_t> (tensor_nbytes (t)))
        BASALT_THROW (IO_Error,
                      "Checkpoint tensor byte size mismatch for '"
                      + name + "'");
      if (nbytes > 0)
        {
          is.read (static_cast<char *> (t.raw_data_ptr ()),
                   static_cast<std::streamsize> (nbytes));
          if (!is)
            BASALT_THROW (IO_Error,
                          "Unexpected EOF reading tensor '" + name + "'");
        }
      m.emplace (std::move (name), std::move (t));
    }
  return m;
}

} /* anonymous namespace */

void
save_checkpoint (const std::string &path, const Checkpoint &ckpt)
{
  std::ostringstream payload_os (std::ios::binary);
  write_pod (payload_os, static_cast<uint64_t> (ckpt.epoch));
  write_pod (payload_os, static_cast<uint64_t> (ckpt.global_step));
  write_pod (payload_os, ckpt.best_metric);
  write_pod (payload_os, ckpt.rng_seed);
  write_tensor_map (payload_os, ckpt.model_state);

  if (ckpt.format_version >= 1U)
    {
      write_tensor_map (payload_os, ckpt.optimizer_state);
      uint64_t meta_count = static_cast<uint64_t> (ckpt.metadata.size ());
      write_pod (payload_os, meta_count);
      for (const auto &kv : ckpt.metadata)
        {
          write_string (payload_os, kv.first);
          write_string (payload_os, kv.second);
        }
    }

  std::string payload = payload_os.str ();
  uint64_t payload_len = static_cast<uint64_t> (payload.size ());
  uint64_t checksum =
    bellows::compute_fnv1a_checksum (payload.data (), payload.size ());

  std::ofstream out (path, std::ios::binary | std::ios::trunc);
  if (!out)
    BASALT_THROW (IO_Error,
                  "save_checkpoint: cannot open '" + path + "' for writing");

  uint32_t magic = k_checkpoint_magic;
  uint32_t version = ckpt.format_version;
  write_pod (out, magic);
  write_pod (out, version);
  write_pod (out, payload_len);
  write_pod (out, checksum);
  if (payload_len > 0)
    out.write (payload.data (), static_cast<std::streamsize> (payload_len));
  if (!out)
    BASALT_THROW (IO_Error,
                  "save_checkpoint: failed writing to '" + path + "'");
}

Checkpoint
load_checkpoint (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    BASALT_THROW (IO_Error,
                  "load_checkpoint: cannot open '" + path + "' for reading");

  uint32_t magic = read_pod<uint32_t> (in);
  if (magic != k_checkpoint_magic)
    BASALT_THROW (IO_Error,
                  "load_checkpoint: invalid magic header in '" + path + "'");

  uint32_t version = read_pod<uint32_t> (in);
  if (version > k_checkpoint_version)
    BASALT_THROW (IO_Error,
                  "load_checkpoint: unsupported future checkpoint version");

  uint64_t payload_len = read_pod<uint64_t> (in);
  uint64_t expected_checksum = read_pod<uint64_t> (in);

  std::string payload (static_cast<size_t> (payload_len), '\0');
  if (payload_len > 0)
    {
      in.read (payload.data (), static_cast<std::streamsize> (payload_len));
      if (!in)
        BASALT_THROW (IO_Error,
                      "load_checkpoint: truncated checkpoint payload");
    }

  uint64_t actual_checksum =
    bellows::compute_fnv1a_checksum (payload.data (), payload.size ());
  if (actual_checksum != expected_checksum)
    BASALT_THROW (IO_Error,
                  "load_checkpoint: FNV-1a checksum mismatch in '"
                  + path + "'");

  std::istringstream ps (payload, std::ios::binary);
  Checkpoint ckpt;
  ckpt.format_version = version;
  ckpt.epoch = static_cast<size_t> (read_pod<uint64_t> (ps));
  ckpt.global_step = static_cast<size_t> (read_pod<uint64_t> (ps));
  ckpt.best_metric = read_pod<double> (ps);
  ckpt.rng_seed = read_pod<uint64_t> (ps);
  ckpt.model_state = read_tensor_map (ps);

  if (version >= 1U && ps.peek () != EOF)
    {
      ckpt.optimizer_state = read_tensor_map (ps);
      if (ps.peek () != EOF)
        {
          uint64_t meta_count = read_pod<uint64_t> (ps);
          for (uint64_t i = 0; i < meta_count; ++i)
            {
              std::string k = read_string (ps);
              std::string v = read_string (ps);
              ckpt.metadata.emplace (std::move (k), std::move (v));
            }
        }
    }
  return ckpt;
}

void
save_model (const std::string &path, const nn::Module &module)
{
  Checkpoint ckpt;
  ckpt.model_state = module.state_dict ();
  ckpt.metadata["producer"] = "Basalt Deep Learning " BASALT_VERSION_STRING;
  save_checkpoint (path, ckpt);
}

void
load_model (const std::string &path, nn::Module &module, bool strict)
{
  Checkpoint ckpt = load_checkpoint (path);
  module.load_state_dict (ckpt.model_state, strict);
}

/* --- Safetensors (.safetensors) reader and writer --- */

namespace
{

std::string
dtype_to_safetensors (Dtype dt)
{
  switch (dt)
    {
    case Dtype::Float32:
      return "F32";
    case Dtype::Float64:
      return "F64";
    case Dtype::Float16:
      return "F16";
    case Dtype::BFloat16:
      return "BF16";
    case Dtype::Int64:
      return "I64";
    case Dtype::Int32:
      return "I32";
    case Dtype::UInt8:
      return "U8";
    case Dtype::Bool:
      return "BOOL";
    }
  return "F32";
}

Dtype
safetensors_to_dtype (const std::string &s)
{
  if (s == "F32")
    return Dtype::Float32;
  if (s == "F64")
    return Dtype::Float64;
  if (s == "F16")
    return Dtype::Float16;
  if (s == "BF16")
    return Dtype::BFloat16;
  if (s == "I64")
    return Dtype::Int64;
  if (s == "I32")
    return Dtype::Int32;
  if (s == "U8" || s == "I8")
    return Dtype::UInt8;
  if (s == "BOOL")
    return Dtype::Bool;
  BASALT_THROW (IO_Error,
                "Unsupported Safetensors dtype '" + s + "'");
}

std::string
json_escape (const std::string &s)
{
  std::string out;
  out.reserve (s.size () + 4);
  for (char c : s)
    {
      if (c == '"' || c == '\\')
        {
          out.push_back ('\\');
          out.push_back (c);
        }
      else if (c == '\n')
        out += "\\n";
      else
        out.push_back (c);
    }
  return out;
}

class Json_Cursor
{
public:
  explicit Json_Cursor (const std::string &src)
    : src_ (src), pos_ (0)
  {
  }

  void
  skip_ws (void)
  {
    while (pos_ < src_.size ()
           && std::isspace (static_cast<unsigned char> (src_[pos_])))
      ++pos_;
  }

  char
  peek (void)
  {
    skip_ws ();
    if (pos_ >= src_.size ())
      BASALT_THROW (IO_Error, "Unexpected end of Safetensors JSON");
    return src_[pos_];
  }

  void
  expect (char expected)
  {
    char c = peek ();
    if (c != expected)
      BASALT_THROW (IO_Error,
                    std::string ("Expected '") + expected
                    + "' in Safetensors JSON");
    ++pos_;
  }

  std::string
  read_string (void)
  {
    expect ('"');
    std::string out;
    while (pos_ < src_.size ())
      {
        char c = src_[pos_++];
        if (c == '"')
          return out;
        if (c == '\\')
          {
            if (pos_ >= src_.size ())
              BASALT_THROW (IO_Error,
                            "Unterminated escape in Safetensors JSON");
            char esc = src_[pos_++];
            if (esc == 'n')
              out.push_back ('\n');
            else
              out.push_back (esc);
          }
        else
          out.push_back (c);
      }
    BASALT_THROW (IO_Error, "Unterminated string in Safetensors JSON");
    return out;
  }

  size_t
  read_size (void)
  {
    skip_ws ();
    size_t start = pos_;
    while (pos_ < src_.size ()
           && std::isdigit (static_cast<unsigned char> (src_[pos_])))
      ++pos_;
    if (pos_ == start)
      BASALT_THROW (IO_Error, "Expected integer in Safetensors JSON");
    return static_cast<size_t> (
      std::stoull (src_.substr (start, pos_ - start)));
  }

private:
  const std::string &src_;
  size_t pos_;
};

} /* anonymous namespace */

void
save_safetensors (const std::string &path,
                  const std::map<std::string, Tensor> &tensors,
                  const std::map<std::string, std::string> &metadata)
{
  std::vector<std::pair<std::string, Tensor>> cpu_tensors;
  cpu_tensors.reserve (tensors.size ());
  for (const auto &kv : tensors)
    {
      cpu_tensors.emplace_back (
        kv.first, kv.second.to (Device::cpu ()).contiguous ());
    }

  std::ostringstream json;
  json << "{";
  bool first_entry = true;
  if (!metadata.empty ())
    {
      first_entry = false;
      json << "\"__metadata__\":{";
      bool first_meta = true;
      for (const auto &mkv : metadata)
        {
          if (!first_meta)
            json << ",";
          first_meta = false;
          json << "\"" << json_escape (mkv.first) << "\":\""
               << json_escape (mkv.second) << "\"";
        }
      json << "}";
    }

  size_t current_offset = 0;
  for (const auto &kv : cpu_tensors)
    {
      if (!first_entry)
        json << ",";
      first_entry = false;
      const Tensor &t = kv.second;
      size_t nbytes = tensor_nbytes (t);
      size_t begin_off = current_offset;
      size_t end_off = current_offset + nbytes;
      current_offset = end_off;

      json << "\"" << json_escape (kv.first) << "\":{\"dtype\":\""
           << dtype_to_safetensors (t.dtype ()) << "\",\"shape\":[";
      for (size_t i = 0; i < t.shape ().size (); ++i)
        {
          if (i > 0)
            json << ",";
          json << t.shape ()[i];
        }
      json << "],\"data_offsets\":[" << begin_off << "," << end_off << "]}";
    }
  json << "}";

  std::string header_str = json.str ();
  while (header_str.size () % 8 != 0)
    header_str.push_back (' ');

  std::ofstream out (path, std::ios::binary | std::ios::trunc);
  if (!out)
    BASALT_THROW (IO_Error,
                  "save_safetensors: cannot open '" + path + "'");

  uint64_t header_len = static_cast<uint64_t> (header_str.size ());
  write_pod (out, header_len);
  out.write (header_str.data (),
             static_cast<std::streamsize> (header_str.size ()));
  for (const auto &kv : cpu_tensors)
    {
      size_t nb = tensor_nbytes (kv.second);
      if (nb > 0)
        out.write (static_cast<const char *> (kv.second.raw_data_ptr ()),
                   static_cast<std::streamsize> (nb));
    }
}

std::map<std::string, Tensor>
load_safetensors (const std::string &path,
                  std::map<std::string, std::string> *out_metadata)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    BASALT_THROW (IO_Error,
                  "load_safetensors: cannot open '" + path + "'");

  uint64_t header_len = read_pod<uint64_t> (in);
  std::string header_str (static_cast<size_t> (header_len), '\0');
  in.read (header_str.data (), static_cast<std::streamsize> (header_len));
  if (!in)
    BASALT_THROW (IO_Error, "load_safetensors: truncated JSON header");

  std::streamoff data_base =
    static_cast<std::streamoff> (sizeof (uint64_t) + header_len);

  struct Entry_Info
  {
    std::string name;
    Dtype dtype = Dtype::Float32;
    Shape_Vector shape;
    size_t begin_off = 0;
    size_t end_off = 0;
  };
  std::vector<Entry_Info> entries;

  Json_Cursor cur (header_str);
  cur.expect ('{');
  if (cur.peek () != '}')
    {
      for (;;)
        {
          std::string key = cur.read_string ();
          cur.expect (':');
          if (key == "__metadata__")
            {
              cur.expect ('{');
              if (cur.peek () != '}')
                {
                  for (;;)
                    {
                      std::string mk = cur.read_string ();
                      cur.expect (':');
                      std::string mv = cur.read_string ();
                      if (out_metadata != nullptr)
                        (*out_metadata)[mk] = mv;
                      if (cur.peek () == ',')
                        cur.expect (',');
                      else
                        break;
                    }
                }
              cur.expect ('}');
            }
          else
            {
              Entry_Info info;
              info.name = key;
              cur.expect ('{');
              if (cur.peek () != '}')
                {
                  for (;;)
                    {
                      std::string fk = cur.read_string ();
                      cur.expect (':');
                      if (fk == "dtype")
                        info.dtype = safetensors_to_dtype (cur.read_string ());
                      else if (fk == "shape")
                        {
                          cur.expect ('[');
                          if (cur.peek () != ']')
                            {
                              for (;;)
                                {
                                  info.shape.push_back (cur.read_size ());
                                  if (cur.peek () == ',')
                                    cur.expect (',');
                                  else
                                    break;
                                }
                            }
                          cur.expect (']');
                        }
                      else if (fk == "data_offsets")
                        {
                          cur.expect ('[');
                          info.begin_off = cur.read_size ();
                          cur.expect (',');
                          info.end_off = cur.read_size ();
                          cur.expect (']');
                        }
                      if (cur.peek () == ',')
                        cur.expect (',');
                      else
                        break;
                    }
                }
              cur.expect ('}');
              entries.push_back (std::move (info));
            }
          if (cur.peek () == ',')
            cur.expect (',');
          else
            break;
        }
    }
  cur.expect ('}');

  std::map<std::string, Tensor> result;
  for (const auto &e : entries)
    {
      Tensor t = empty (e.shape, Tensor_Options ()
                                   .with_dtype (e.dtype)
                                   .with_device (Device::cpu ()));
      size_t expected_bytes = e.end_off - e.begin_off;
      if (expected_bytes != tensor_nbytes (t))
        BASALT_THROW (IO_Error,
                      "Safetensors byte length mismatch for '" + e.name + "'");
      if (expected_bytes > 0)
        {
          in.seekg (data_base + static_cast<std::streamoff> (e.begin_off),
                    std::ios::beg);
          in.read (static_cast<char *> (t.raw_data_ptr ()),
                   static_cast<std::streamsize> (expected_bytes));
          if (!in)
            BASALT_THROW (IO_Error,
                          "Safetensors truncated data for '" + e.name + "'");
        }
      result.emplace (e.name, std::move (t));
    }
  return result;
}

/* --- ONNX (.onnx) binary Protobuf wire-format reader and writer --- */

namespace
{

void
pb_write_varint (std::string &buf, uint64_t val)
{
  while (val >= 0x80U)
    {
      buf.push_back (static_cast<char> ((val & 0x7FU) | 0x80U));
      val >>= 7U;
    }
  buf.push_back (static_cast<char> (val));
}

void
pb_write_tag (std::string &buf, uint32_t field_number, uint32_t wire_type)
{
  pb_write_varint (buf, (static_cast<uint64_t> (field_number) << 3U)
                        | static_cast<uint64_t> (wire_type));
}

void
pb_write_bytes (std::string &buf, uint32_t field_number,
                const void *data, size_t len)
{
  pb_write_tag (buf, field_number, 2U);
  pb_write_varint (buf, static_cast<uint64_t> (len));
  if (len > 0)
    buf.append (static_cast<const char *> (data), len);
}

void
pb_write_string (std::string &buf, uint32_t field_number, const std::string &s)
{
  pb_write_bytes (buf, field_number, s.data (), s.size ());
}

uint32_t
dtype_to_onnx (Dtype dt)
{
  switch (dt)
    {
    case Dtype::Float32:
      return 1U;
    case Dtype::UInt8:
      return 2U;
    case Dtype::Int32:
      return 6U;
    case Dtype::Int64:
      return 7U;
    case Dtype::Bool:
      return 9U;
    case Dtype::Float16:
      return 10U;
    case Dtype::Float64:
      return 11U;
    case Dtype::BFloat16:
      return 16U;
    }
  return 1U;
}

Dtype
onnx_to_dtype (uint32_t code)
{
  switch (code)
    {
    case 1U:
      return Dtype::Float32;
    case 2U:
    case 3U:
      return Dtype::UInt8;
    case 6U:
      return Dtype::Int32;
    case 7U:
      return Dtype::Int64;
    case 9U:
      return Dtype::Bool;
    case 10U:
      return Dtype::Float16;
    case 11U:
      return Dtype::Float64;
    case 16U:
      return Dtype::BFloat16;
    default:
      BASALT_THROW (IO_Error,
                    "Unsupported ONNX TensorProto data_type code");
    }
  return Dtype::Float32;
}

std::string
encode_onnx_tensor_proto (const std::string &name, const Tensor &t)
{
  Tensor cpu_t = t.to (Device::cpu ()).contiguous ();
  std::string msg;
  for (size_t d : cpu_t.shape ())
    {
      pb_write_tag (msg, 1U, 0U);
      pb_write_varint (msg, static_cast<uint64_t> (d));
    }
  pb_write_tag (msg, 2U, 0U);
  pb_write_varint (msg, dtype_to_onnx (cpu_t.dtype ()));
  pb_write_string (msg, 8U, name);
  pb_write_bytes (msg, 9U, cpu_t.raw_data_ptr (), tensor_nbytes (cpu_t));
  return msg;
}

std::string
encode_onnx_node_proto (const std::vector<std::string> &inputs,
                        const std::vector<std::string> &outputs,
                        const std::string &name,
                        const std::string &op_type)
{
  std::string msg;
  for (const auto &in_s : inputs)
    pb_write_string (msg, 1U, in_s);
  for (const auto &out_s : outputs)
    pb_write_string (msg, 2U, out_s);
  pb_write_string (msg, 3U, name);
  pb_write_string (msg, 4U, op_type);
  return msg;
}

uint64_t
pb_read_varint (const uint8_t *data, size_t size, size_t &pos)
{
  uint64_t result = 0;
  uint32_t shift = 0;
  while (pos < size)
    {
      uint8_t byte = data[pos++];
      result |= static_cast<uint64_t> (byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0)
        return result;
      shift += 7U;
      if (shift >= 64U)
        BASALT_THROW (IO_Error, "Malformed Protobuf varint");
    }
  BASALT_THROW (IO_Error, "Unexpected EOF in Protobuf varint");
  return 0;
}

void
pb_skip_field (const uint8_t *data, size_t size, size_t &pos,
               uint32_t wire_type)
{
  if (wire_type == 0U)
    (void) pb_read_varint (data, size, pos);
  else if (wire_type == 1U)
    {
      if (pos + 8U > size)
        BASALT_THROW (IO_Error, "Protobuf EOF skipping 64-bit");
      pos += 8U;
    }
  else if (wire_type == 2U)
    {
      uint64_t len = pb_read_varint (data, size, pos);
      if (pos + len > size)
        BASALT_THROW (IO_Error, "Protobuf EOF skipping bytes");
      pos += static_cast<size_t> (len);
    }
  else if (wire_type == 5U)
    {
      if (pos + 4U > size)
        BASALT_THROW (IO_Error, "Protobuf EOF skipping 32-bit");
      pos += 4U;
    }
  else
    BASALT_THROW (IO_Error, "Unsupported Protobuf wire type");
}

std::pair<std::string, Tensor>
decode_onnx_tensor_proto (const uint8_t *data, size_t size)
{
  size_t pos = 0;
  Shape_Vector dims;
  uint32_t dt_code = 1U;
  std::string name;
  const uint8_t *raw_ptr = nullptr;
  size_t raw_len = 0;
  std::vector<float> float_vals;
  std::vector<int64_t> int64_vals;

  while (pos < size)
    {
      uint64_t tag = pb_read_varint (data, size, pos);
      uint32_t field_num = static_cast<uint32_t> (tag >> 3U);
      uint32_t wire_type = static_cast<uint32_t> (tag & 0x7U);

      if (field_num == 1U && wire_type == 0U)
        dims.push_back (
          static_cast<size_t> (pb_read_varint (data, size, pos)));
      else if (field_num == 1U && wire_type == 2U)
        {
          uint64_t len = pb_read_varint (data, size, pos);
          size_t end = pos + static_cast<size_t> (len);
          while (pos < end)
            dims.push_back (
              static_cast<size_t> (pb_read_varint (data, size, pos)));
        }
      else if (field_num == 2U && wire_type == 0U)
        dt_code = static_cast<uint32_t> (pb_read_varint (data, size, pos));
      else if (field_num == 4U && wire_type == 2U)
        {
          uint64_t len = pb_read_varint (data, size, pos);
          size_t count = static_cast<size_t> (len) / sizeof (float);
          for (size_t i = 0; i < count; ++i)
            {
              float v = 0.0f;
              std::memcpy (&v, data + pos + i * sizeof (float),
                           sizeof (float));
              float_vals.push_back (v);
            }
          pos += static_cast<size_t> (len);
        }
      else if (field_num == 7U && wire_type == 2U)
        {
          uint64_t len = pb_read_varint (data, size, pos);
          size_t end = pos + static_cast<size_t> (len);
          while (pos < end)
            int64_vals.push_back (
              static_cast<int64_t> (pb_read_varint (data, size, pos)));
        }
      else if (field_num == 8U && wire_type == 2U)
        {
          uint64_t len = pb_read_varint (data, size, pos);
          name.assign (reinterpret_cast<const char *> (data + pos),
                       static_cast<size_t> (len));
          pos += static_cast<size_t> (len);
        }
      else if (field_num == 9U && wire_type == 2U)
        {
          uint64_t len = pb_read_varint (data, size, pos);
          raw_ptr = data + pos;
          raw_len = static_cast<size_t> (len);
          pos += raw_len;
        }
      else
        pb_skip_field (data, size, pos, wire_type);
    }

  Dtype dt = onnx_to_dtype (dt_code);
  Tensor t = empty (
    dims, Tensor_Options ().with_dtype (dt).with_device (Device::cpu ()));
  size_t nb = tensor_nbytes (t);
  if (raw_ptr != nullptr && raw_len == nb && raw_len > 0)
    std::memcpy (t.raw_data_ptr (), raw_ptr, raw_len);
  else if (!float_vals.empty () && dt == Dtype::Float32
           && float_vals.size () * sizeof (float) == nb)
    std::memcpy (t.raw_data_ptr (), float_vals.data (), nb);
  else if (!int64_vals.empty () && dt == Dtype::Int64
           && int64_vals.size () * sizeof (int64_t) == nb)
    std::memcpy (t.raw_data_ptr (), int64_vals.data (), nb);

  return {name, std::move (t)};
}

} /* anonymous namespace */

void
export_onnx (const std::string &path,
             const nn::Module &module,
             const Shape_Vector &sample_input_shape,
             const std::string &model_name)
{
  (void) sample_input_shape;
  nn::State_Dict sd = module.state_dict ();

  std::string graph_msg;
  std::string prev_out = "input";
  size_t node_idx = 0;
  for (const auto &kv : sd)
    {
      if (kv.first.size () >= 6
          && kv.first.substr (kv.first.size () - 6) == "weight")
        {
          std::string out_name = "node_out_" + std::to_string (node_idx);
          std::string op_type = (kv.second.ndim () == 2) ? "Gemm" : "Conv";
          std::string node_msg = encode_onnx_node_proto (
            {prev_out, kv.first}, {out_name},
            "node_" + std::to_string (node_idx), op_type);
          pb_write_string (graph_msg, 1U, node_msg);
          prev_out = out_name;
          ++node_idx;
        }
    }

  pb_write_string (graph_msg, 2U, model_name);
  for (const auto &kv : sd)
    {
      std::string init_msg = encode_onnx_tensor_proto (kv.first, kv.second);
      pb_write_string (graph_msg, 5U, init_msg);
    }

  std::string opset_msg;
  pb_write_string (opset_msg, 1U, "");
  pb_write_tag (opset_msg, 2U, 0U);
  pb_write_varint (opset_msg, 17U);

  std::string model_msg;
  pb_write_tag (model_msg, 1U, 0U);
  pb_write_varint (model_msg, 8U);
  pb_write_string (model_msg, 2U, "Basalt Deep Learning");
  pb_write_string (model_msg, 3U, BASALT_VERSION_STRING);
  pb_write_string (model_msg, 7U, graph_msg);
  pb_write_string (model_msg, 8U, opset_msg);

  std::ofstream out (path, std::ios::binary | std::ios::trunc);
  if (!out)
    BASALT_THROW (IO_Error, "export_onnx: cannot open '" + path + "'");
  out.write (model_msg.data (),
             static_cast<std::streamsize> (model_msg.size ()));
}

std::map<std::string, Tensor>
import_onnx_weights (const std::string &path,
                     std::string *out_producer,
                     int64_t *out_ir_version)
{
  std::ifstream in (path, std::ios::binary | std::ios::ate);
  if (!in)
    BASALT_THROW (IO_Error,
                  "import_onnx_weights: cannot open '" + path + "'");
  std::streamsize fsize = in.tellg ();
  in.seekg (0, std::ios::beg);
  std::vector<uint8_t> buf (static_cast<size_t> (fsize));
  if (fsize > 0)
    in.read (reinterpret_cast<char *> (buf.data ()), fsize);

  std::map<std::string, Tensor> weights;
  size_t pos = 0;
  size_t size = buf.size ();
  while (pos < size)
    {
      uint64_t tag = pb_read_varint (buf.data (), size, pos);
      uint32_t field_num = static_cast<uint32_t> (tag >> 3U);
      uint32_t wire_type = static_cast<uint32_t> (tag & 0x7U);

      if (field_num == 1U && wire_type == 0U)
        {
          uint64_t ir_v = pb_read_varint (buf.data (), size, pos);
          if (out_ir_version != nullptr)
            *out_ir_version = static_cast<int64_t> (ir_v);
        }
      else if (field_num == 2U && wire_type == 2U)
        {
          uint64_t len = pb_read_varint (buf.data (), size, pos);
          if (out_producer != nullptr)
            out_producer->assign (
              reinterpret_cast<const char *> (buf.data () + pos),
              static_cast<size_t> (len));
          pos += static_cast<size_t> (len);
        }
      else if (field_num == 7U && wire_type == 2U)
        {
          uint64_t glen = pb_read_varint (buf.data (), size, pos);
          size_t gpos = pos;
          size_t gend = pos + static_cast<size_t> (glen);
          while (gpos < gend)
            {
              uint64_t gtag = pb_read_varint (buf.data (), gend, gpos);
              uint32_t gfield = static_cast<uint32_t> (gtag >> 3U);
              uint32_t gwire = static_cast<uint32_t> (gtag & 0x7U);
              if (gfield == 5U && gwire == 2U)
                {
                  uint64_t tlen = pb_read_varint (buf.data (), gend, gpos);
                  auto kv = decode_onnx_tensor_proto (
                    buf.data () + gpos, static_cast<size_t> (tlen));
                  weights.emplace (std::move (kv.first),
                                   std::move (kv.second));
                  gpos += static_cast<size_t> (tlen);
                }
              else
                pb_skip_field (buf.data (), gend, gpos, gwire);
            }
          pos = gend;
        }
      else
        pb_skip_field (buf.data (), size, pos, wire_type);
    }
  return weights;
}

void
load_onnx_model (const std::string &path, nn::Module &module, bool strict)
{
  auto weights = import_onnx_weights (path);
  module.load_state_dict (weights, strict);
}

std::string
inspect_artifact (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    BASALT_THROW (IO_Error,
                  "inspect_artifact: cannot open '" + path + "'");

  uint32_t magic = 0;
  in.read (reinterpret_cast<char *> (&magic), sizeof (magic));
  in.close ();

  std::ostringstream out;
  if (magic == k_checkpoint_magic)
    {
      Checkpoint ckpt = load_checkpoint (path);
      out << "Format: Basalt Native Checkpoint (.bckpt)\n"
          << "Version: " << ckpt.format_version << "\n"
          << "Epoch: " << ckpt.epoch << "\n"
          << "Global step: " << ckpt.global_step << "\n"
          << "Best metric: " << ckpt.best_metric << "\n"
          << "Tensors (" << ckpt.model_state.size () << "):\n";
      for (const auto &kv : ckpt.model_state)
        {
          out << "  " << kv.first << ": dtype="
              << dtype_name (kv.second.dtype ()) << " shape=[";
          for (size_t i = 0; i < kv.second.shape ().size (); ++i)
            {
              if (i > 0)
                out << ", ";
              out << kv.second.shape ()[i];
            }
          out << "] bytes=" << tensor_nbytes (kv.second) << "\n";
        }
      return out.str ();
    }

  if (path.size () >= 5 && path.substr (path.size () - 5) == ".onnx")
    {
      std::string producer;
      int64_t ir_v = 0;
      auto weights = import_onnx_weights (path, &producer, &ir_v);
      out << "Format: ONNX Model (.onnx)\n"
          << "IR version: " << ir_v << "\n"
          << "Producer: " << producer << "\n"
          << "Initializers (" << weights.size () << "):\n";
      for (const auto &kv : weights)
        {
          out << "  " << kv.first << ": dtype="
              << dtype_name (kv.second.dtype ()) << " bytes="
              << tensor_nbytes (kv.second) << "\n";
        }
      return out.str ();
    }

  std::map<std::string, std::string> meta;
  auto st = load_safetensors (path, &meta);
  out << "Format: Safetensors (.safetensors)\n"
      << "Tensors (" << st.size () << "):\n";
  for (const auto &kv : st)
    {
      out << "  " << kv.first << ": dtype="
          << dtype_name (kv.second.dtype ()) << " bytes="
          << tensor_nbytes (kv.second) << "\n";
    }
  return out.str ();
}

} /* namespace basalt::io */
