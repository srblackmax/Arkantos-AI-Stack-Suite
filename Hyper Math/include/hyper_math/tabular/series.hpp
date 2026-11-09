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

#ifndef HYPER_MATH_TABULAR_SERIES_HPP
#define HYPER_MATH_TABULAR_SERIES_HPP

#include "hyper_math/core/bitmask.hpp"
#include "hyper_math/tabular/index.hpp"
#include "hyper_math/tensor/reductions.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace hyper_math::tabular
{

using tensor::nd_tensor;
using tensor::shape_type;

enum class data_type
{
  int64,
  float64,
  boolean,
  utf8_string
};

inline std::string
dtype_to_string (data_type dt)
{
  switch (dt)
    {
    case data_type::int64:
      return "int64";
    case data_type::float64:
      return "float64";
    case data_type::boolean:
      return "bool";
    case data_type::utf8_string:
      return "utf8";
    }
  return "unknown";
}

/* Nullable scalar cell representation: std::monostate represents NULL/NaN. */
using cell_value =
  std::variant<std::monostate, std::int64_t, double, bool, std::string>;

inline std::string
format_cell (const cell_value &cell)
{
  if (std::holds_alternative<std::monostate> (cell))
    {
      return "null";
    }
  else if (auto *iv = std::get_if<std::int64_t> (&cell))
    {
      return std::to_string (*iv);
    }
  else if (auto *dv = std::get_if<double> (&cell))
    {
      std::ostringstream oss;
      oss << std::setprecision (6) << *dv;
      return oss.str ();
    }
  else if (auto *bv = std::get_if<bool> (&cell))
    {
      return *bv ? "true" : "false";
    }
  else if (auto *sv = std::get_if<std::string> (&cell))
    {
      return *sv;
    }
  return "null";
}

/* Lossless typed canonical key for hashing, grouping, joins, and deduplication.
   Prefixes a type tag ('0'=null, '1'=int64, '2'=float64 with 17 digits,
   '3'=bool, '4'=string) so NULL never collides with the string "null" and
   full IEEE-754 double precision is preserved.  */
inline std::string
cell_hash_key (const cell_value &cell)
{
  if (std::holds_alternative<std::monostate> (cell))
    {
      return "0:";
    }
  if (auto *iv = std::get_if<std::int64_t> (&cell))
    {
      return "1:" + std::to_string (*iv);
    }
  if (auto *dv = std::get_if<double> (&cell))
    {
      const double v = (*dv == 0.0) ? 0.0 : *dv;
      std::ostringstream oss;
      oss << "2:" << std::setprecision (17) << v;
      return oss.str ();
    }
  if (auto *bv = std::get_if<bool> (&cell))
    {
      return *bv ? "3:1" : "3:0";
    }
  if (auto *sv = std::get_if<std::string> (&cell))
    {
      return "4:" + *sv;
    }
  return "0:";
}

inline double
cell_as_double (const cell_value &cell)
{
  if (auto *dv = std::get_if<double> (&cell))
    {
      return *dv;
    }
  if (auto *iv = std::get_if<std::int64_t> (&cell))
    {
      return static_cast<double> (*iv);
    }
  if (auto *bv = std::get_if<bool> (&cell))
    {
      return *bv ? 1.0 : 0.0;
    }
  return constants::nan;
}

/* Dual-Bitmask Columnar Series (unifies Pandas Series and Polars Series).
   Pairs a contiguous typed column buffer with a 64-bit word validity_bitmap
   and a hash-indexed index_axis.  */
class series
{
private:
  std::string name_;
  data_type dtype_;
  nd_tensor<std::int64_t> i64_data_;
  nd_tensor<double> f64_data_;
  nd_tensor<std::uint8_t> bool_data_;
  std::vector<std::string> str_data_;
  validity_bitmap validity_;
  index_axis index_;

public:
  series ()
    : name_ (""),
      dtype_ (data_type::float64),
      i64_data_ (),
      f64_data_ (),
      bool_data_ (),
      str_data_ (),
      validity_ (0),
      index_ (0)
  {
  }

  /* Construct float64 series from vector<double> (NaN automatically marked
     invalid in the validity_bitmap).  */
  series (std::string col_name,
          const std::vector<double> &values,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::float64),
      i64_data_ (),
      f64_data_ (shape_type {values.size ()}, values),
      bool_data_ (),
      str_data_ (),
      validity_ (values.size (), true),
      index_ (idx.value_or (index_axis (values.size ())))
  {
    if (index_.size () != values.size ())
      {
        throw shape_error ("series index_axis size does not match values length");
      }
    for (std::size_t i = 0; i < values.size (); ++i)
      {
        if (std::isnan (values[i]))
          {
            validity_.set_valid (i, false);
          }
      }
  }

  series (std::string col_name,
          const nd_tensor<double> &values,
          std::optional<index_axis> idx = std::nullopt)
    : series (std::move (col_name), values,
              validity_bitmap (values.size (), true), std::move (idx))
  {
  }

  series (std::string col_name,
          const nd_tensor<double> &values,
          validity_bitmap valid_mask,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::float64),
      i64_data_ (),
      f64_data_ (values.ravel ()),
      bool_data_ (),
      str_data_ (),
      validity_ (std::move (valid_mask)),
      index_ (idx.value_or (index_axis (values.size ())))
  {
    if (validity_.size () != f64_data_.size ()
        || index_.size () != f64_data_.size ())
      {
        throw shape_error ("series validity or index size mismatch");
      }
    for (std::size_t i = 0; i < f64_data_.size (); ++i)
      {
        if (std::isnan (f64_data_[i]))
          {
            validity_.set_valid (i, false);
          }
      }
  }

  /* Construct int64 series.  */
  series (std::string col_name,
          const std::vector<std::int64_t> &values,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::int64),
      i64_data_ (shape_type {values.size ()}, values),
      f64_data_ (),
      bool_data_ (),
      str_data_ (),
      validity_ (values.size (), true),
      index_ (idx.value_or (index_axis (values.size ())))
  {
    if (index_.size () != values.size ())
      {
        throw shape_error ("series index_axis size does not match values length");
      }
  }

  series (std::string col_name,
          const nd_tensor<std::int64_t> &values,
          std::optional<index_axis> idx = std::nullopt)
    : series (std::move (col_name), values,
              validity_bitmap (values.size (), true), std::move (idx))
  {
  }

  series (std::string col_name,
          const nd_tensor<std::int64_t> &values,
          validity_bitmap valid_mask,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::int64),
      i64_data_ (values.ravel ()),
      f64_data_ (),
      bool_data_ (),
      str_data_ (),
      validity_ (std::move (valid_mask)),
      index_ (idx.value_or (index_axis (values.size ())))
  {
    if (validity_.size () != i64_data_.size ()
        || index_.size () != i64_data_.size ())
      {
        throw shape_error ("series validity or index size mismatch");
      }
  }

  /* Construct boolean series.  */
  series (std::string col_name,
          const std::vector<bool> &values,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::boolean),
      i64_data_ (),
      f64_data_ (),
      bool_data_ (shape_type {values.size ()}),
      str_data_ (),
      validity_ (values.size (), true),
      index_ (idx.value_or (index_axis (values.size ())))
  {
    if (index_.size () != values.size ())
      {
        throw shape_error ("series index_axis size does not match values length");
      }
    for (std::size_t i = 0; i < values.size (); ++i)
      {
        bool_data_[i] = values[i] ? 1 : 0;
      }
  }

  series (std::string col_name,
          const nd_tensor<std::uint8_t> &values,
          validity_bitmap valid_mask,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::boolean),
      i64_data_ (),
      f64_data_ (),
      bool_data_ (values.ravel ()),
      str_data_ (),
      validity_ (std::move (valid_mask)),
      index_ (idx.value_or (index_axis (values.size ())))
  {
    if (validity_.size () != bool_data_.size ()
        || index_.size () != bool_data_.size ())
      {
        throw shape_error ("series validity or index size mismatch");
      }
  }

  /* Construct utf8 string series.  */
  series (std::string col_name,
          std::vector<std::string> values,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::utf8_string),
      i64_data_ (),
      f64_data_ (),
      bool_data_ (),
      str_data_ (std::move (values)),
      validity_ (str_data_.size (), true),
      index_ (idx.value_or (index_axis (str_data_.size ())))
  {
    if (index_.size () != str_data_.size ())
      {
        throw shape_error ("series index_axis size does not match values length");
      }
  }

  series (std::string col_name,
          std::vector<std::string> values,
          validity_bitmap valid_mask,
          std::optional<index_axis> idx = std::nullopt)
    : name_ (std::move (col_name)),
      dtype_ (data_type::utf8_string),
      i64_data_ (),
      f64_data_ (),
      bool_data_ (),
      str_data_ (std::move (values)),
      validity_ (std::move (valid_mask)),
      index_ (idx.value_or (index_axis (str_data_.size ())))
  {
    if (validity_.size () != str_data_.size ()
        || index_.size () != str_data_.size ())
      {
        throw shape_error ("series validity or index size mismatch");
      }
  }

  /* Metadata accessors.  */
  const std::string &
  name () const noexcept
  {
    return name_;
  }

  void
  set_name (std::string new_name)
  {
    name_ = std::move (new_name);
  }

  series
  alias (std::string new_name) const
  {
    series copy = *this;
    copy.name_ = std::move (new_name);
    return copy;
  }

  data_type
  dtype () const noexcept
  {
    return dtype_;
  }

  std::size_t
  size () const noexcept
  {
    return validity_.size ();
  }

  bool
  empty () const noexcept
  {
    return validity_.size () == 0;
  }

  const validity_bitmap &
  validity () const noexcept
  {
    return validity_;
  }

  const index_axis &
  index () const noexcept
  {
    return index_;
  }

  void
  set_index (index_axis new_idx)
  {
    if (new_idx.size () != size ())
      {
        throw shape_error ("new index_axis size does not match series length");
      }
    index_ = std::move (new_idx);
  }

  bool
  is_null (std::size_t pos) const noexcept
  {
    return validity_.is_null (pos);
  }

  bool
  is_valid (std::size_t pos) const noexcept
  {
    return validity_.is_valid (pos);
  }

  std::size_t
  null_count () const noexcept
  {
    return validity_.null_count ();
  }

  std::size_t
  valid_count () const noexcept
  {
    return validity_.valid_count ();
  }

  /* Positional (`iloc`) and label-based (`loc`) element access.  */
  cell_value
  iloc (std::size_t pos) const
  {
    if (pos >= size ())
      {
        throw index_error ("series::iloc index out of bounds");
      }
    if (validity_.is_null (pos))
      {
        return std::monostate {};
      }
    switch (dtype_)
      {
      case data_type::int64:
        return i64_data_[pos];
      case data_type::float64:
        return f64_data_[pos];
      case data_type::boolean:
        return static_cast<bool> (bool_data_[pos] != 0);
      case data_type::utf8_string:
        return str_data_[pos];
      }
    return std::monostate {};
  }

  cell_value
  loc (const std::string &label) const
  {
    return iloc (index_.first_position_of (label));
  }

  double
  get_double (std::size_t pos) const
  {
    if (pos >= size ())
      {
        throw index_error ("series::get_double index out of bounds");
      }
    if (dtype_ == data_type::utf8_string)
      {
        throw schema_error ("cannot convert utf8 series element to double");
      }
    if (!validity_.is_valid (pos))
      {
        return constants::nan;
      }
    if (dtype_ == data_type::float64)
      {
        return f64_data_[pos];
      }
    if (dtype_ == data_type::int64)
      {
        return static_cast<double> (i64_data_[pos]);
      }
    return bool_data_[pos] ? 1.0 : 0.0;
  }

  /* Convert numeric series to contiguous float64 nd_tensor.  */
  nd_tensor<double>
  to_float_tensor () const
  {
    nd_tensor<double> out (shape_type {size ()});
    for (std::size_t i = 0; i < size (); ++i)
      {
        out[i] = get_double (i);
      }
    return out;
  }

  const nd_tensor<double> &
  f64_tensor () const
  {
    return f64_data_;
  }

  const nd_tensor<std::int64_t> &
  i64_tensor () const
  {
    return i64_data_;
  }

  const nd_tensor<std::uint8_t> &
  bool_tensor () const
  {
    return bool_data_;
  }

  const std::vector<std::string> &
  str_values () const
  {
    return str_data_;
  }

  /* Take a subset of rows by positional index vector.  */
  series
  take (const std::vector<std::size_t> &positions) const
  {
    const std::size_t m = positions.size ();
    for (std::size_t i = 0; i < m; ++i)
      {
        if (positions[i] >= size ())
          {
            throw index_error ("series::take index out of bounds");
          }
      }
    validity_bitmap new_valid (m, true);
    index_axis new_idx = index_.take (positions);

    if (dtype_ == data_type::float64)
      {
        nd_tensor<double> buf (shape_type {m});
        for (std::size_t i = 0; i < m; ++i)
          {
            const std::size_t p = positions[i];
            new_valid.set_valid (i, validity_.is_valid (p));
            buf[i] = f64_data_[p];
          }
        return series (name_, buf, std::move (new_valid), std::move (new_idx));
      }
    else if (dtype_ == data_type::int64)
      {
        nd_tensor<std::int64_t> buf (shape_type {m});
        for (std::size_t i = 0; i < m; ++i)
          {
            const std::size_t p = positions[i];
            new_valid.set_valid (i, validity_.is_valid (p));
            buf[i] = i64_data_[p];
          }
        return series (name_, buf, std::move (new_valid), std::move (new_idx));
      }
    else if (dtype_ == data_type::boolean)
      {
        nd_tensor<std::uint8_t> buf (shape_type {m});
        for (std::size_t i = 0; i < m; ++i)
          {
            const std::size_t p = positions[i];
            new_valid.set_valid (i, validity_.is_valid (p));
            buf[i] = bool_data_[p];
          }
        return series (name_, buf, std::move (new_valid), std::move (new_idx));
      }
    else
      {
        std::vector<std::string> buf (m);
        for (std::size_t i = 0; i < m; ++i)
          {
            const std::size_t p = positions[i];
            new_valid.set_valid (i, validity_.is_valid (p));
            buf[i] = str_data_[p];
          }
        return series (name_, std::move (buf),
                       std::move (new_valid), std::move (new_idx));
      }
  }

  series
  slice (std::size_t offset, std::size_t length) const
  {
    const std::size_t start = std::min (offset, size ());
    const std::size_t stop = std::min (start + length, size ());
    std::vector<std::size_t> pos (stop - start);
    for (std::size_t i = start; i < stop; ++i)
      {
        pos[i - start] = i;
      }
    return take (pos);
  }

  series
  head (std::size_t n = 5) const
  {
    return slice (0, n);
  }

  series
  tail (std::size_t n = 5) const
  {
    const std::size_t len = std::min (n, size ());
    return slice (size () - len, len);
  }

  /* Filter rows using a boolean series mask.  */
  series
  filter (const series &mask) const
  {
    if (mask.dtype () != data_type::boolean || mask.size () != size ())
      {
        throw shape_error ("series::filter requires matching boolean series");
      }
    std::vector<std::size_t> selected;
    selected.reserve (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (mask.is_valid (i) && mask.bool_data_[i] != 0)
          {
            selected.push_back (i);
          }
      }
    return take (selected);
  }

  /* Missing data handling: drop_nulls (dropna), fill_null (fillna),
     fill_forward (ffill), fill_backward (bfill).  */
  series
  drop_nulls () const
  {
    std::vector<std::size_t> valid_pos;
    valid_pos.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            valid_pos.push_back (i);
          }
      }
    return take (valid_pos);
  }

  series
  fill_null (double replacement) const
  {
    nd_tensor<double> out = to_float_tensor ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_null (i))
          {
            out[i] = replacement;
          }
      }
    return series (name_, out, validity_bitmap (size (), true), index_);
  }

  series
  fill_null_str (const std::string &replacement) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("fill_null_str requires a utf8_string series");
      }
    std::vector<std::string> out = str_data_;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_null (i))
          {
            out[i] = replacement;
          }
      }
    return series (name_, std::move (out),
                   validity_bitmap (size (), true), index_);
  }

  series
  fill_forward () const
  {
    validity_bitmap new_valid = validity_.clone ();
    if (dtype_ == data_type::utf8_string)
      {
        std::vector<std::string> out = str_data_;
        bool seen = false;
        std::string last_val;
        for (std::size_t i = 0; i < size (); ++i)
          {
            if (new_valid.is_valid (i))
              {
                last_val = out[i];
                seen = true;
              }
            else if (seen)
              {
                out[i] = last_val;
                new_valid.set_valid (i, true);
              }
          }
        return series (name_, std::move (out), std::move (new_valid), index_);
      }
    nd_tensor<double> out = to_float_tensor ();
    bool seen = false;
    double last_val = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i))
          {
            last_val = out[i];
            seen = true;
          }
        else if (seen)
          {
            out[i] = last_val;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  fill_backward () const
  {
    validity_bitmap new_valid = validity_.clone ();
    if (dtype_ == data_type::utf8_string)
      {
        std::vector<std::string> out = str_data_;
        bool seen = false;
        std::string next_val;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (size ()) - 1;
             i >= 0; --i)
          {
            const std::size_t ui = static_cast<std::size_t> (i);
            if (new_valid.is_valid (ui))
              {
                next_val = out[ui];
                seen = true;
              }
            else if (seen)
              {
                out[ui] = next_val;
                new_valid.set_valid (ui, true);
              }
          }
        return series (name_, std::move (out), std::move (new_valid), index_);
      }
    nd_tensor<double> out = to_float_tensor ();
    bool seen = false;
    double next_val = 0.0;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (size ()) - 1;
         i >= 0; --i)
      {
        const std::size_t ui = static_cast<std::size_t> (i);
        if (new_valid.is_valid (ui))
          {
            next_val = out[ui];
            seen = true;
          }
        else if (seen)
          {
            out[ui] = next_val;
            new_valid.set_valid (ui, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  /* Null-aware reductions and statistical aggregations.  */
  double
  sum () const
  {
    double acc = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            acc += get_double (i);
          }
      }
    return acc;
  }

  double
  mean () const
  {
    const std::size_t vc = valid_count ();
    if (vc == 0)
      {
        return constants::nan;
      }
    return sum () / static_cast<double> (vc);
  }

  double
  var (std::size_t ddof = 1) const
  {
    const std::size_t vc = valid_count ();
    if (vc <= ddof)
      {
        return constants::nan;
      }
    const double m = mean ();
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double d = get_double (i) - m;
            sum_sq += d * d;
          }
      }
    return sum_sq / static_cast<double> (vc - ddof);
  }

  double
  std_dev (std::size_t ddof = 1) const
  {
    return std::sqrt (var (ddof));
  }

  double
  min () const
  {
    double best = constants::inf;
    bool found = false;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            best = std::min (best, get_double (i));
            found = true;
          }
      }
    return found ? best : constants::nan;
  }

  double
  max () const
  {
    double best = -constants::inf;
    bool found = false;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            best = std::max (best, get_double (i));
            found = true;
          }
      }
    return found ? best : constants::nan;
  }

  double
  median () const
  {
    std::vector<double> vals;
    vals.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            vals.push_back (get_double (i));
          }
      }
    if (vals.empty ())
      {
        return constants::nan;
      }
    std::sort (vals.begin (), vals.end ());
    const std::size_t mid = vals.size () / 2;
    if (vals.size () % 2 == 1)
      {
        return vals[mid];
      }
    return 0.5 * (vals[mid - 1] + vals[mid]);
  }

  double
  quantile (double q_pct = 50.0) const
  {
    const double p = (q_pct <= 1.0 && q_pct >= 0.0) ? (q_pct * 100.0) : q_pct;
    std::vector<double> vals;
    vals.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            vals.push_back (get_double (i));
          }
      }
    if (vals.empty ())
      {
        return constants::nan;
      }
    const nd_tensor<double> vt (shape_type {vals.size ()}, vals);
    return tensor::percentile (vt, p);
  }

  double
  skew () const
  {
    const std::size_t vc = valid_count ();
    if (vc < 3)
      {
        return constants::nan;
      }
    const double m = mean ();
    const double s = std_dev (0);
    if (s <= 0.0)
      {
        return 0.0;
      }
    double m3 = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double z = (get_double (i) - m) / s;
            m3 += z * z * z;
          }
      }
    return m3 / static_cast<double> (vc);
  }

  double
  kurtosis () const
  {
    const std::size_t vc = valid_count ();
    if (vc < 4)
      {
        return constants::nan;
      }
    const double m = mean ();
    const double s = std_dev (0);
    if (s <= 0.0)
      {
        return 0.0;
      }
    double m4 = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double z = (get_double (i) - m) / s;
            m4 += z * z * z * z;
          }
      }
    return (m4 / static_cast<double> (vc)) - 3.0;
  }

  bool
  any () const
  {
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            if (dtype_ == data_type::utf8_string)
              {
                if (!str_data_[i].empty ())
                  {
                    return true;
                  }
              }
            else if (get_double (i) != 0.0)
              {
                return true;
              }
          }
      }
    return false;
  }

  bool
  all () const
  {
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (!validity_.is_valid (i))
          {
            return false;
          }
        if (dtype_ == data_type::utf8_string)
          {
            if (str_data_[i].empty ())
              {
                return false;
              }
          }
        else if (get_double (i) == 0.0)
          {
            return false;
          }
      }
    return true;
  }

  series
  cum_count () const
  {
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    std::int64_t c = 0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            ++c;
          }
        out[i] = c;
      }
    return series (name_ + "_cum_count", out, validity_bitmap (size (), true), index_);
  }

  std::size_t
  n_unique () const
  {
    std::vector<std::string> reprs;
    reprs.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            reprs.push_back (cell_hash_key (iloc (i)));
          }
      }
    std::sort (reprs.begin (), reprs.end ());
    reprs.erase (std::unique (reprs.begin (), reprs.end ()), reprs.end ());
    return reprs.size ();
  }

  /* Rolling & Exponentially Weighted Moving Windows (Pandas/Polars).  */
  series
  rolling_mean (std::size_t window) const
  {
    if (window == 0)
      {
        throw hyper_math_error ("rolling_mean window must be >= 1");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);

    for (std::size_t i = window - 1; i < size (); ++i)
      {
        double acc = 0.0;
        bool all_ok = true;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (validity_.is_null (k))
              {
                all_ok = false;
                break;
              }
            acc += get_double (k);
          }
        if (all_ok)
          {
            out[i] = acc / static_cast<double> (window);
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  rolling_sum (std::size_t window) const
  {
    if (window == 0)
      {
        throw hyper_math_error ("rolling_sum window must be >= 1");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);

    for (std::size_t i = window - 1; i < size (); ++i)
      {
        double acc = 0.0;
        bool all_ok = true;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (validity_.is_null (k))
              {
                all_ok = false;
                break;
              }
            acc += get_double (k);
          }
        if (all_ok)
          {
            out[i] = acc;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  ewm_mean (double alpha) const
  {
    if (alpha <= 0.0 || alpha > 1.0)
      {
        throw hyper_math_error ("ewm_mean alpha must lie in (0, 1]");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    bool initialized = false;
    double ema = 0.0;

    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double val = get_double (i);
            ema = initialized ? (alpha * val + (1.0 - alpha) * ema) : val;
            initialized = true;
            out[i] = ema;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  diff (std::size_t periods = 1) const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    for (std::size_t i = periods; i < size (); ++i)
      {
        if (validity_.is_valid (i) && validity_.is_valid (i - periods))
          {
            out[i] = get_double (i) - get_double (i - periods);
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  pct_change (std::size_t periods = 1) const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    for (std::size_t i = periods; i < size (); ++i)
      {
        if (validity_.is_valid (i) && validity_.is_valid (i - periods))
          {
            const double prev = get_double (i - periods);
            if (prev != 0.0)
              {
                out[i] = (get_double (i) - prev) / prev;
                new_valid.set_valid (i, true);
              }
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  /* Pandas-style Index-Aligned Arithmetic: aligns two Series by their
     index_axis labels (union of labels); labels missing in either operand
     automatically produce null in the resulting validity_bitmap!  */
  template <typename BinaryOp>
  series
  binary_op_aligned (const series &other, BinaryOp op) const
  {
    if (index_ == other.index_)
      {
        nd_tensor<double> out (shape_type {size ()});
        validity_bitmap out_valid = validity_ & other.validity_;
        for (std::size_t i = 0; i < size (); ++i)
          {
            if (out_valid.is_valid (i))
              {
                out[i] = op (get_double (i), other.get_double (i));
              }
            else
              {
                out[i] = constants::nan;
              }
          }
        return series (name_, out, std::move (out_valid), index_);
      }

    /* Union of labels preserving order of appearance.  */
    std::vector<std::string> union_labels;
    std::unordered_map<std::string, bool> seen;
    for (const auto &lbl : index_.labels ())
      {
        if (!seen[lbl])
          {
            seen[lbl] = true;
            union_labels.push_back (lbl);
          }
      }
    for (const auto &lbl : other.index_.labels ())
      {
        if (!seen[lbl])
          {
            seen[lbl] = true;
            union_labels.push_back (lbl);
          }
      }

    const std::size_t n_out = union_labels.size ();
    nd_tensor<double> out (shape_type {n_out}, constants::nan);
    validity_bitmap out_valid (n_out, false);

    for (std::size_t i = 0; i < n_out; ++i)
      {
        const std::string &lbl = union_labels[i];
        if (index_.contains (lbl) && other.index_.contains (lbl))
          {
            const std::size_t pa = index_.first_position_of (lbl);
            const std::size_t pb = other.index_.first_position_of (lbl);
            if (validity_.is_valid (pa) && other.validity_.is_valid (pb))
              {
                out[i] = op (get_double (pa), other.get_double (pb));
                out_valid.set_valid (i, true);
              }
          }
      }

    return series (name_, out, std::move (out_valid),
                   index_axis (std::move (union_labels)));
  }

  series
  operator+ (const series &other) const
  {
    return binary_op_aligned (other, [] (double a, double b) { return a + b; });
  }

  series
  operator- (const series &other) const
  {
    return binary_op_aligned (other, [] (double a, double b) { return a - b; });
  }

  series
  operator* (const series &other) const
  {
    return binary_op_aligned (other, [] (double a, double b) { return a * b; });
  }

  series
  operator/ (const series &other) const
  {
    return binary_op_aligned (other, [] (double a, double b) { return a / b; });
  }

  series
  operator+ (double scalar) const
  {
    nd_tensor<double> out = to_float_tensor () + scalar;
    return series (name_, out, validity_.clone (), index_);
  }

  series
  operator- (double scalar) const
  {
    nd_tensor<double> out = to_float_tensor () - scalar;
    return series (name_, out, validity_.clone (), index_);
  }

  series
  operator* (double scalar) const
  {
    nd_tensor<double> out = to_float_tensor () * scalar;
    return series (name_, out, validity_.clone (), index_);
  }

  series
  operator/ (double scalar) const
  {
    nd_tensor<double> out = to_float_tensor () / scalar;
    return series (name_, out, validity_.clone (), index_);
  }

  /* Vectorized comparison operators returning boolean series.  */
  template <typename CmpOp>
  series
  compare_scalar (double rhs, CmpOp cmp) const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = cmp (get_double (i), rhs) ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  operator> (double rhs) const
  {
    return compare_scalar (rhs, [] (double a, double b) { return a > b; });
  }

  series
  operator>= (double rhs) const
  {
    return compare_scalar (rhs, [] (double a, double b) { return a >= b; });
  }

  series
  operator< (double rhs) const
  {
    return compare_scalar (rhs, [] (double a, double b) { return a < b; });
  }

  series
  operator<= (double rhs) const
  {
    return compare_scalar (rhs, [] (double a, double b) { return a <= b; });
  }

  series
  operator== (double rhs) const
  {
    return compare_scalar (rhs, [] (double a, double b) { return a == b; });
  }

  series
  operator!= (double rhs) const
  {
    return compare_scalar (rhs, [] (double a, double b) { return a != b; });
  }

  series
  operator== (const std::string &rhs) const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i) && dtype_ == data_type::utf8_string)
          {
            out[i] = (str_data_[i] == rhs) ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  operator!= (const std::string &rhs) const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i) && dtype_ == data_type::utf8_string)
          {
            out[i] = (str_data_[i] != rhs) ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  /* String operations (Pandas .str / Polars .str namespace equivalents).  */
  series
  str_to_uppercase () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_to_uppercase requires utf8_string series");
      }
    std::vector<std::string> out = str_data_;
    for (auto &s : out)
      {
        for (char &c : s)
          {
            c = static_cast<char> (
              std::toupper (static_cast<unsigned char> (c)));
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_contains (const std::string &pattern) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_contains requires utf8_string series");
      }
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] =
              (str_data_[i].find (pattern) != std::string::npos) ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  str_to_lowercase () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_to_lowercase requires utf8_string series");
      }
    std::vector<std::string> out = str_data_;
    for (auto &s : out)
      {
        for (char &c : s)
          {
            c = static_cast<char> (
              std::tolower (static_cast<unsigned char> (c)));
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_lengths () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_lengths requires utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = static_cast<std::int64_t> (str_data_[i].size ());
          }
      }
    return series (name_ + "_len", out, validity_.clone (), index_);
  }

  series
  str_replace (const std::string &pattern, const std::string &replacement) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_replace requires utf8_string series");
      }
    std::vector<std::string> out = str_data_;
    if (!pattern.empty ())
      {
        for (std::size_t i = 0; i < size (); ++i)
          {
            if (validity_.is_valid (i))
              {
                std::string &s = out[i];
                std::size_t pos = 0;
                while ((pos = s.find (pattern, pos)) != std::string::npos)
                  {
                    s.replace (pos, pattern.size (), replacement);
                    pos += replacement.size ();
                  }
              }
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_starts_with (const std::string &prefix) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_starts_with requires utf8_string series");
      }
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = (str_data_[i].rfind (prefix, 0) == 0) ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  str_ends_with (const std::string &suffix) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_ends_with requires utf8_string series");
      }
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i) && str_data_[i].size () >= suffix.size ())
          {
            const std::size_t pos = str_data_[i].size () - suffix.size ();
            out[i] = (str_data_[i].compare (pos, suffix.size (), suffix) == 0)
                     ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  str_strip () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_strip requires utf8_string series");
      }
    std::vector<std::string> out = str_data_;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            std::string &s = out[i];
            std::size_t l = 0;
            while (l < s.size () && std::isspace (static_cast<unsigned char> (s[l])))
              {
                ++l;
              }
            std::size_t r = s.size ();
            while (r > l && std::isspace (static_cast<unsigned char> (s[r - 1])))
              {
                --r;
              }
            s = s.substr (l, r - l);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  between (double low, double high, bool inclusive = true) const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double v = get_double (i);
            const bool inside = inclusive
                                ? (v >= low && v <= high)
                                : (v > low && v < high);
            out[i] = inside ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  /* Linear interpolation of missing (null) values between valid anchors
     (Pandas / Polars interpolate equivalent).  */
  series
  interpolate () const
  {
    nd_tensor<double> out = to_float_tensor ();
    validity_bitmap new_valid = validity_.clone ();
    const std::size_t n = size ();
    std::size_t i = 0;
    while (i < n)
      {
        if (new_valid.is_valid (i))
          {
            ++i;
            continue;
          }
        std::size_t gap_start = i;
        while (i < n && new_valid.is_null (i))
          {
            ++i;
          }
        std::size_t gap_end = i;
        if (gap_start > 0 && gap_end < n)
          {
            const double v0 = out[gap_start - 1];
            const double v1 = out[gap_end];
            const double span =
              static_cast<double> (gap_end - (gap_start - 1));
            for (std::size_t k = gap_start; k < gap_end; ++k)
              {
                const double t =
                  static_cast<double> (k - (gap_start - 1)) / span;
                out[k] = v0 + t * (v1 - v0);
                new_valid.set_valid (k, true);
              }
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  /* Shift values by `periods` (positive shifts down, negative shifts up),
     inserting nulls at newly exposed boundary positions.  */
  series
  shift (std::ptrdiff_t periods = 1) const
  {
    const std::size_t n = size ();
    nd_tensor<double> out (shape_type {n}, constants::nan);
    validity_bitmap new_valid (n, false);
    for (std::size_t i = 0; i < n; ++i)
      {
        const std::ptrdiff_t src =
          static_cast<std::ptrdiff_t> (i) - periods;
        if (src >= 0 && static_cast<std::size_t> (src) < n)
          {
            const std::size_t usrc = static_cast<std::size_t> (src);
            if (validity_.is_valid (usrc))
              {
                out[i] = get_double (usrc);
                new_valid.set_valid (i, true);
              }
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  /* Cumulative sum, min, max, rolling_min, rolling_max, rolling_std.  */
  series
  cum_sum () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid = validity_.clone ();
    double acc = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            acc += get_double (i);
            out[i] = acc;
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  cum_prod () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid = validity_.clone ();
    double acc = 1.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            acc *= get_double (i);
            out[i] = acc;
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  cum_min () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid = validity_.clone ();
    double best = constants::inf;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            best = std::min (best, get_double (i));
            out[i] = best;
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  cum_max () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid = validity_.clone ();
    double best = -constants::inf;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            best = std::max (best, get_double (i));
            out[i] = best;
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  std::size_t
  nunique () const
  {
    return n_unique ();
  }

  series
  unique () const
  {
    std::set<std::string> seen;
    if (dtype_ == data_type::utf8_string)
      {
        std::vector<std::string> out;
        for (std::size_t i = 0; i < size (); ++i)
          {
            if (validity_.is_valid (i) && seen.insert (str_data_[i]).second)
              {
                out.push_back (str_data_[i]);
              }
          }
        return series (name_, std::move (out));
      }
    std::vector<double> out;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string key = cell_hash_key (iloc (i));
            if (seen.insert (key).second)
              {
                out.push_back (get_double (i));
              }
          }
      }
    return series (name_, out);
  }

  series
  rolling_min (std::size_t window) const
  {
    if (window == 0)
      {
        throw hyper_math_error ("rolling_min window must be >= 1");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    for (std::size_t i = window - 1; i < size (); ++i)
      {
        double best = constants::inf;
        bool all_ok = true;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (validity_.is_null (k))
              {
                all_ok = false;
                break;
              }
            best = std::min (best, get_double (k));
          }
        if (all_ok)
          {
            out[i] = best;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  rolling_max (std::size_t window) const
  {
    if (window == 0)
      {
        throw hyper_math_error ("rolling_max window must be >= 1");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    for (std::size_t i = window - 1; i < size (); ++i)
      {
        double best = -constants::inf;
        bool all_ok = true;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (validity_.is_null (k))
              {
                all_ok = false;
                break;
              }
            best = std::max (best, get_double (k));
          }
        if (all_ok)
          {
            out[i] = best;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  rolling_std (std::size_t window, std::size_t ddof = 1) const
  {
    if (window <= ddof)
      {
        throw hyper_math_error ("rolling_std window must exceed ddof");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    for (std::size_t i = window - 1; i < size (); ++i)
      {
        double sum_v = 0.0;
        bool all_ok = true;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (validity_.is_null (k))
              {
                all_ok = false;
                break;
              }
            sum_v += get_double (k);
          }
        if (all_ok)
          {
            const double m = sum_v / static_cast<double> (window);
            double sq = 0.0;
            for (std::size_t k = i + 1 - window; k <= i; ++k)
              {
                const double d = get_double (k) - m;
                sq += d * d;
              }
            out[i] = std::sqrt (sq / static_cast<double> (window - ddof));
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  /* Frequency counts of distinct non-null values (Pandas/Polars value_counts). */
  std::map<std::string, std::size_t>
  value_counts () const
  {
    std::map<std::string, std::size_t> counts;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            counts[format_cell (iloc (i))] += 1;
          }
      }
    return counts;
  }

  /* Set membership check (Pandas isin / Polars is_in).  */
  series
  is_in (const std::vector<std::string> &candidates) const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string repr = format_cell (iloc (i));
            if (std::find (candidates.begin (), candidates.end (), repr)
                != candidates.end ())
              {
                out[i] = 1;
              }
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  /* Elementwise numeric transformations: abs, clip, round, log, exp, sqrt.  */
  series
  abs () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = std::abs (get_double (i));
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  clip (double low, double high) const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = std::clamp (get_double (i), low, high);
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  round () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = std::round (get_double (i));
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  /* Rank values with tie-breaking methods: "average", "min", "max", "dense",
     "ordinal" (Pandas Series.rank / Polars Expr.rank equivalent).  */
  series
  rank (const std::string &method = "average", bool ascending = true) const
  {
    std::vector<std::size_t> valid_idx;
    valid_idx.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            valid_idx.push_back (i);
          }
      }

    std::stable_sort (valid_idx.begin (), valid_idx.end (),
                      [&] (std::size_t a, std::size_t b) {
                        const double va = get_double (a);
                        const double vb = get_double (b);
                        return ascending ? (va < vb) : (va > vb);
                      });

    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    std::size_t dense_rank = 1;
    std::size_t i = 0;
    while (i < valid_idx.size ())
      {
        std::size_t j = i;
        const double ref = get_double (valid_idx[i]);
        while (j + 1 < valid_idx.size ()
               && get_double (valid_idx[j + 1]) == ref)
          {
            ++j;
          }

        for (std::size_t k = i; k <= j; ++k)
          {
            double r_val = 0.0;
            if (method == "average")
              {
                r_val = 0.5 * static_cast<double> (i + j) + 1.0;
              }
            else if (method == "min")
              {
                r_val = static_cast<double> (i + 1);
              }
            else if (method == "max")
              {
                r_val = static_cast<double> (j + 1);
              }
            else if (method == "dense")
              {
                r_val = static_cast<double> (dense_rank);
              }
            else if (method == "ordinal")
              {
                r_val = static_cast<double> (k + 1);
              }
            else
              {
                throw hyper_math_error ("unknown rank method: " + method);
              }
            out[valid_idx[k]] = r_val;
          }
        ++dense_rank;
        i = j + 1;
      }

    return series (name_ + "_rank", out, validity_.clone (), index_);
  }

  /* Equal-width (`cut`) and quantile-based (`qcut`) discretization into
     integer bin codes 0..bins-1 (Pandas cut / qcut equivalent).  */
  series
  cut (std::size_t bins) const
  {
    if (bins == 0)
      {
        throw hyper_math_error ("cut requires bins >= 1");
      }
    const double lo = min ();
    const double hi = max ();
    const double span = (hi > lo) ? (hi - lo) : 1.0;
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double v = get_double (i);
            const double raw_bin =
              std::floor (((v - lo) / span) * static_cast<double> (bins));
            const std::int64_t b = std::clamp<std::int64_t> (
              std::isfinite (raw_bin) ? static_cast<std::int64_t> (raw_bin) : 0,
              0,
              static_cast<std::int64_t> (bins - 1));
            out[i] = b;
          }
      }
    return series (name_ + "_bin", out, validity_.clone (), index_);
  }

  series
  qcut (std::size_t q_bins) const
  {
    if (q_bins == 0)
      {
        throw hyper_math_error ("qcut requires q_bins >= 1");
      }
    std::vector<double> valid_vals;
    valid_vals.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            valid_vals.push_back (get_double (i));
          }
      }
    const nd_tensor<double> vt (shape_type {valid_vals.size ()}, valid_vals);
    std::vector<double> edges (q_bins + 1);
    for (std::size_t k = 0; k <= q_bins; ++k)
      {
        const double q_pct =
          (100.0 * static_cast<double> (k)) / static_cast<double> (q_bins);
        edges[k] = valid_vals.empty () ? 0.0 : tensor::percentile (vt, q_pct);
      }

    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const double v = get_double (i);
            std::size_t b = 0;
            for (std::size_t k = 1; k < q_bins; ++k)
              {
                if (v > edges[k])
                  {
                    b = k;
                  }
              }
            out[i] = static_cast<std::int64_t> (b);
          }
      }
    return series (name_ + "_qbin", out, validity_.clone (), index_);
  }

  /* ISO-8601 Temporal / Date-Time Accessors ("YYYY-MM-DD", Pandas .dt /
     Polars .dt namespace equivalents: dt_year, dt_month, dt_day,
     dt_day_of_week, dt_epoch_days).  */
  series
  dt_year () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_year requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i) && str_data_[i].size () >= 10)
          {
            out[i] = std::stoll (str_data_[i].substr (0, 4));
          }
        else
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_ + "_year", out, std::move (new_valid), index_);
  }

  series
  dt_month () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_month requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i) && str_data_[i].size () >= 10)
          {
            out[i] = std::stoll (str_data_[i].substr (5, 2));
          }
        else
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_ + "_month", out, std::move (new_valid), index_);
  }

  series
  dt_day () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_day requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i) && str_data_[i].size () >= 10)
          {
            out[i] = std::stoll (str_data_[i].substr (8, 2));
          }
        else
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_ + "_day", out, std::move (new_valid), index_);
  }

  /* Howard Hinnant civil-to-days algorithm: days since 1970-01-01 (Unix epoch). */
  series
  dt_epoch_days () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_epoch_days requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i) && str_data_[i].size () >= 10)
          {
            std::int64_t y = std::stoll (str_data_[i].substr (0, 4));
            const std::int64_t m = std::stoll (str_data_[i].substr (5, 2));
            const std::int64_t d = std::stoll (str_data_[i].substr (8, 2));
            y -= (m <= 2) ? 1 : 0;
            const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
            const std::int64_t yoe = y - era * 400;
            const std::int64_t doy =
              (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
            const std::int64_t doe =
              yoe * 365 + yoe / 4 - yoe / 100 + doy;
            out[i] = era * 146097 + doe - 719468;
          }
        else
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_ + "_days", out, std::move (new_valid), index_);
  }

  /* Lag-k Pearson autocorrelation (Pandas Series.autocorr equivalent).  */
  double
  autocorr (std::size_t lag = 1) const
  {
    const std::size_t n = size ();
    if (lag >= n || n - lag < 2)
      {
        return constants::nan;
      }
    const series shifted = shift (static_cast<std::ptrdiff_t> (lag));
    std::vector<double> x;
    std::vector<double> y;
    for (std::size_t i = 0; i < n; ++i)
      {
        if (is_valid (i) && shifted.is_valid (i))
          {
            x.push_back (get_double (i));
            y.push_back (shifted.get_double (i));
          }
      }
    if (x.size () < 2)
      {
        return constants::nan;
      }
    double mx = 0.0;
    double my = 0.0;
    for (std::size_t i = 0; i < x.size (); ++i)
      {
        mx += x[i];
        my += y[i];
      }
    mx /= static_cast<double> (x.size ());
    my /= static_cast<double> (y.size ());
    double num = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    for (std::size_t i = 0; i < x.size (); ++i)
      {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
      }
    return (vx > 0.0 && vy > 0.0) ? (num / std::sqrt (vx * vy))
                                  : constants::nan;
  }

  /* Statistical Mode (most frequent valid value(s)), ArgMin / ArgMax
     (idxmin / idxmax), functional numeric map (`map_numeric`), and ISO-8601
     weekday / day-of-year (`dt_day_of_week`, `dt_day_of_year`)
     (Pandas / Polars Series equivalents).  */
  series
  mode () const
  {
    std::map<std::string, std::size_t> counts;
    std::vector<std::size_t> first_idx;
    std::map<std::string, std::size_t> repr_to_first;
    std::size_t max_freq = 0;

    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            const std::string repr = cell_hash_key (iloc (i));
            if (counts.find (repr) == counts.end ())
              {
                repr_to_first[repr] = i;
              }
            const std::size_t c = ++counts[repr];
            max_freq = std::max (max_freq, c);
          }
      }

    if (max_freq == 0)
      {
        return series (name_ + "_mode", std::vector<double> {});
      }

    std::vector<std::size_t> mode_indices;
    for (const auto &kv : counts)
      {
        if (kv.second == max_freq)
          {
            mode_indices.push_back (repr_to_first[kv.first]);
          }
      }
    std::sort (mode_indices.begin (), mode_indices.end ());

    if (dtype_ == data_type::utf8_string)
      {
        std::vector<std::string> out_str;
        for (std::size_t idx : mode_indices)
          {
            out_str.push_back (str_data_[idx]);
          }
        return series (name_ + "_mode", std::move (out_str));
      }
    std::vector<double> out_num;
    for (std::size_t idx : mode_indices)
      {
        out_num.push_back (get_double (idx));
      }
    return series (name_ + "_mode", out_num);
  }

  std::size_t
  arg_min () const
  {
    std::size_t best_idx = size ();
    double best_val = constants::inf;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            const double v = get_double (i);
            if (v < best_val || best_idx == size ())
              {
                best_val = v;
                best_idx = i;
              }
          }
      }
    if (best_idx == size ())
      {
        throw index_error ("arg_min called on all-null or empty series");
      }
    return best_idx;
  }

  std::size_t
  arg_max () const
  {
    std::size_t best_idx = size ();
    double best_val = -constants::inf;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            const double v = get_double (i);
            if (v > best_val || best_idx == size ())
              {
                best_val = v;
                best_idx = i;
              }
          }
      }
    if (best_idx == size ())
      {
        throw index_error ("arg_max called on all-null or empty series");
      }
    return best_idx;
  }

  std::string
  idxmin () const
  {
    return index_[arg_min ()];
  }

  std::string
  idxmax () const
  {
    return index_[arg_max ()];
  }

  template <typename UnaryFn>
  series
  map_numeric (UnaryFn &&fn) const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = fn (get_double (i));
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  /* ISO-8601 day of week (Monday = 0 .. Sunday = 6, Pandas dt.dayofweek).  */
  series
  dt_day_of_week () const
  {
    const series epoch = dt_epoch_days ();
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (epoch.is_valid (i))
          {
            /* 1970-01-01 was a Thursday (index 3 in Monday=0..Sunday=6).  */
            const std::int64_t z = epoch.i64_tensor ()[i];
            std::int64_t dow = (z + 3) % 7;
            if (dow < 0)
              {
                dow += 7;
              }
            out[i] = dow;
          }
      }
    return series (name_ + "_dow", out, epoch.validity ().clone (), index_);
  }

  /* ISO-8601 day of year (1..366, Pandas dt.dayofyear / Polars dt.ordinal_day). */
  series
  dt_day_of_year () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_day_of_year requires ISO-8601 utf8_string series");
      }
    static constexpr int cum_days[12] = {
      0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
    };
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i) && str_data_[i].size () >= 10)
          {
            const std::int64_t y = std::stoll (str_data_[i].substr (0, 4));
            const std::int64_t m = std::stoll (str_data_[i].substr (5, 2));
            const std::int64_t d = std::stoll (str_data_[i].substr (8, 2));
            const bool leap =
              ((y % 4 == 0) && (y % 100 != 0)) || (y % 400 == 0);
            std::int64_t doy =
              cum_days[std::clamp<std::int64_t> (m - 1, 0, 11)] + d;
            if (leap && m > 2)
              {
                doy += 1;
              }
            out[i] = doy;
          }
        else
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_ + "_doy", out, std::move (new_valid), index_);
  }

  /* Sliding-window rolling variance (`rolling_var`) and expanding window mean
     (`expanding_mean`) (Pandas Series.rolling.var / Series.expanding.mean).  */
  series
  rolling_var (std::size_t window, std::size_t ddof = 1) const
  {
    const series rs = rolling_std (window, ddof);
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (rs.is_valid (i))
          {
            const double v = rs.get_double (i);
            out[i] = v * v;
          }
      }
    return series (name_ + "_rvar", out, rs.validity ().clone (), index_);
  }

  series
  expanding_mean () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    double sum_v = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            sum_v += get_double (i);
            ++count;
          }
        if (count > 0)
          {
            out[i] = sum_v / static_cast<double> (count);
            new_valid.set_valid (i, true);
          }
      }
    return series (name_ + "_exp_mean", out, std::move (new_valid), index_);
  }

  series
  expanding_sum () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    double sum_v = 0.0;
    bool seen = false;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            sum_v += get_double (i);
            seen = true;
          }
        if (seen)
          {
            out[i] = sum_v;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_ + "_exp_sum", out, std::move (new_valid), index_);
  }

  series
  expanding_min () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    double best = constants::inf;
    bool seen = false;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            best = std::min (best, get_double (i));
            seen = true;
          }
        if (seen)
          {
            out[i] = best;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_ + "_exp_min", out, std::move (new_valid), index_);
  }

  series
  expanding_max () const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    double best = -constants::inf;
    bool seen = false;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            best = std::max (best, get_double (i));
            seen = true;
          }
        if (seen)
          {
            out[i] = best;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_ + "_exp_max", out, std::move (new_valid), index_);
  }

  /* Rolling quantile (`q_pct` in [0, 100]) and rolling median over a fixed
     window of size `window` (Pandas Series.rolling.quantile / median).  */
  series
  rolling_quantile (std::size_t window, double q_pct) const
  {
    if (window == 0)
      {
        throw hyper_math_error ("rolling_quantile window must be >= 1");
      }
    if (std::isnan (q_pct) || q_pct < 0.0 || q_pct > 100.0)
      {
        throw hyper_math_error ("rolling_quantile q_pct must lie in [0, 100]");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    std::vector<double> buf;
    buf.reserve (window);

    for (std::size_t i = window - 1; i < size (); ++i)
      {
        buf.clear ();
        bool all_ok = true;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (validity_.is_null (k))
              {
                all_ok = false;
                break;
              }
            buf.push_back (get_double (k));
          }
        if (all_ok)
          {
            std::sort (buf.begin (), buf.end ());
            const double pos =
              (q_pct / 100.0) * static_cast<double> (window - 1);
            const std::size_t lo =
              static_cast<std::size_t> (std::floor (pos));
            const std::size_t hi =
              static_cast<std::size_t> (std::ceil (pos));
            const double frac = pos - static_cast<double> (lo);
            out[i] = buf[lo] * (1.0 - frac) + buf[hi] * frac;
            new_valid.set_valid (i, true);
          }
      }
    return series (name_, out, std::move (new_valid), index_);
  }

  series
  rolling_median (std::size_t window) const
  {
    return rolling_quantile (window, 50.0);
  }

  /* Expanding variance `expanding_var`, expanding standard deviation
     `expanding_std`, lag-k autocorrelation `autocorr`, and elementwise
     transcendental transforms `sqrt`, `log`, `exp`.  */
  series
  expanding_var (std::size_t ddof = 1) const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    double sum_v = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;

    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            const double v = get_double (i);
            ++count;
            const double delta = v - sum_v;
            sum_v += delta / static_cast<double> (count);
            const double delta2 = v - sum_v;
            sum_sq += delta * delta2;
          }
        if (count > ddof)
          {
            out[i] = sum_sq / static_cast<double> (count - ddof);
            new_valid.set_valid (i, true);
          }
      }
    return series (name_ + "_exp_var", out, std::move (new_valid), index_);
  }

  series
  expanding_std (std::size_t ddof = 1) const
  {
    series v = expanding_var (ddof);
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid = v.validity ().clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i))
          {
            out[i] = std::sqrt (v.get_double (i));
          }
      }
    return series (name_ + "_exp_std", out, std::move (new_valid), index_);
  }

  series
  sqrt () const
  {
    return map_numeric ([] (double v) { return std::sqrt (v); });
  }

  series
  log () const
  {
    return map_numeric ([] (double v) { return std::log (v); });
  }

  series
  exp () const
  {
    return map_numeric ([] (double v) { return std::exp (v); });
  }

  /* Standard error of the mean `sem`, mean absolute deviation `mad`,
     missing-value predicate `hasnans`, and trigonometric transforms
     `sin`, `cos`, `tan` (Pandas/Polars Series equivalents).  */
  double
  sem (std::size_t ddof = 1) const
  {
    const std::size_t vc = valid_count ();
    if (vc <= ddof)
      {
        return constants::nan;
      }
    return std_dev (ddof) / std::sqrt (static_cast<double> (vc));
  }

  double
  mad () const
  {
    const std::size_t vc = valid_count ();
    if (vc == 0)
      {
        return constants::nan;
      }
    const double m = mean ();
    double acc = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            acc += std::abs (get_double (i) - m);
          }
      }
    return acc / static_cast<double> (vc);
  }

  bool
  hasnans () const noexcept
  {
    return null_count () > 0;
  }

  series
  sin () const
  {
    return map_numeric ([] (double v) { return std::sin (v); });
  }

  series
  cos () const
  {
    return map_numeric ([] (double v) { return std::cos (v); });
  }

  series
  tan () const
  {
    return map_numeric ([] (double v) { return std::tan (v); });
  }

  /* Exponentially weighted moving variance `ewm_var` and standard deviation
     `ewm_std` with smoothing factor `alpha` in (0, 1]
     (Pandas Series.ewm(alpha=...).var() / .std() equivalents).  */
  series
  ewm_var (double alpha) const
  {
    if (alpha <= 0.0 || alpha > 1.0)
      {
        throw hyper_math_error ("ewm_var alpha must lie in (0, 1]");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap new_valid (size (), false);
    double ema = 0.0;
    double emvar = 0.0;
    std::size_t seen = 0;

    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            const double x = get_double (i);
            if (seen == 0)
              {
                ema = x;
                emvar = 0.0;
              }
            else
              {
                const double delta = x - ema;
                ema += alpha * delta;
                emvar = (1.0 - alpha) * (emvar + alpha * delta * delta);
                out[i] = emvar;
                new_valid.set_valid (i, true);
              }
            ++seen;
          }
      }
    return series (name_ + "_ewm_var", out, std::move (new_valid), index_);
  }

  series
  ewm_std (double alpha) const
  {
    series v = ewm_var (alpha);
    return v.sqrt ().alias (name_ + "_ewm_std");
  }

  /* Null-aware monotonicity checks `is_monotonic_increasing` and
     `is_monotonic_decreasing`, plus empirical percentile tail clipping
     `winsorize(lower_pct, upper_pct)` (Pandas/SciPy equivalents).  */
  bool
  is_monotonic_increasing () const noexcept
  {
    if (dtype_ == data_type::utf8_string)
      {
        bool have_prev = false;
        std::string prev;
        for (std::size_t i = 0; i < size (); ++i)
          {
            if (!is_valid (i))
              {
                return false;
              }
            if (have_prev && str_data_[i] < prev)
              {
                return false;
              }
            prev = str_data_[i];
            have_prev = true;
          }
        return true;
      }
    bool have_prev = false;
    double prev = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (!is_valid (i))
          {
            return false;
          }
        const double cur =
          (dtype_ == data_type::float64)
          ? f64_data_[i]
          : ((dtype_ == data_type::int64)
             ? static_cast<double> (i64_data_[i])
             : (bool_data_[i] ? 1.0 : 0.0));
        if (have_prev && cur < prev)
          {
            return false;
          }
        prev = cur;
        have_prev = true;
      }
    return true;
  }

  bool
  is_monotonic_decreasing () const noexcept
  {
    if (dtype_ == data_type::utf8_string)
      {
        bool have_prev = false;
        std::string prev;
        for (std::size_t i = 0; i < size (); ++i)
          {
            if (!is_valid (i))
              {
                return false;
              }
            if (have_prev && str_data_[i] > prev)
              {
                return false;
              }
            prev = str_data_[i];
            have_prev = true;
          }
        return true;
      }
    bool have_prev = false;
    double prev = 0.0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (!is_valid (i))
          {
            return false;
          }
        const double cur =
          (dtype_ == data_type::float64)
          ? f64_data_[i]
          : ((dtype_ == data_type::int64)
             ? static_cast<double> (i64_data_[i])
             : (bool_data_[i] ? 1.0 : 0.0));
        if (have_prev && cur > prev)
          {
            return false;
          }
        prev = cur;
        have_prev = true;
      }
    return true;
  }

  series
  winsorize (double lower_pct, double upper_pct) const
  {
    std::vector<double> valid_vals;
    valid_vals.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            valid_vals.push_back (get_double (i));
          }
      }
    if (valid_vals.empty ())
      {
        return *this;
      }
    const nd_tensor<double> vt (shape_type {valid_vals.size ()}, valid_vals);
    const double lo = tensor::percentile (vt, lower_pct);
    const double hi = tensor::percentile (vt, upper_pct);
    return clip (lo, hi);
  }

  /* Vectorized String Slicing, Padding, Zero-Fill, and Substring Occurrence
     Counting (`str_slice`, `str_pad_left`, `str_pad_right`, `str_zfill`,
     `str_count`) plus ISO-8601 Time-of-Day Accessors (`dt_hour`, `dt_minute`,
     `dt_second`) (Pandas .str / .dt and Polars .str / .dt equivalents).  */
  series
  str_slice (std::size_t pos, std::size_t len = std::string::npos) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_slice requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            out[i] = (pos < s.size ()) ? s.substr (pos, len) : "";
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_pad_left (std::size_t width, char fill_char = ' ') const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_pad_left requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            out[i] =
              (s.size () < width)
              ? (std::string (width - s.size (), fill_char) + s)
              : s;
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_pad_right (std::size_t width, char fill_char = ' ') const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_pad_right requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            out[i] =
              (s.size () < width)
              ? (s + std::string (width - s.size (), fill_char))
              : s;
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_zfill (std::size_t width) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_zfill requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (s.size () >= width)
              {
                out[i] = s;
              }
            else if (!s.empty () && (s[0] == '+' || s[0] == '-'))
              {
                out[i] =
                  std::string (1, s[0])
                  + std::string (width - s.size (), '0')
                  + s.substr (1);
              }
            else
              {
                out[i] = std::string (width - s.size (), '0') + s;
              }
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_count (const std::string &pat) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_count requires utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i) && !pat.empty ())
          {
            const std::string &s = str_data_[i];
            std::int64_t cnt = 0;
            std::size_t pos = 0;
            while ((pos = s.find (pat, pos)) != std::string::npos)
              {
                ++cnt;
                pos += pat.size ();
              }
            out[i] = cnt;
          }
      }
    return series (name_ + "_count", out, validity_.clone (), index_);
  }

  series
  dt_hour () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_hour requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (s.size () >= 13 && (s[10] == 'T' || s[10] == ' '))
              {
                out[i] = std::stoll (s.substr (11, 2));
              }
            else if (s.size () >= 2 && s.find (':') == 2)
              {
                out[i] = std::stoll (s.substr (0, 2));
              }
            else
              {
                new_valid.set_valid (i, false);
              }
          }
      }
    return series (name_ + "_hour", out, std::move (new_valid), index_);
  }

  series
  dt_minute () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_minute requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (s.size () >= 16 && (s[10] == 'T' || s[10] == ' '))
              {
                out[i] = std::stoll (s.substr (14, 2));
              }
            else if (s.size () >= 5 && s.find (':') == 2)
              {
                out[i] = std::stoll (s.substr (3, 2));
              }
            else
              {
                new_valid.set_valid (i, false);
              }
          }
      }
    return series (name_ + "_minute", out, std::move (new_valid), index_);
  }

  series
  dt_second () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_second requires ISO-8601 utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (new_valid.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (s.size () >= 19 && (s[10] == 'T' || s[10] == ' '))
              {
                out[i] = std::stoll (s.substr (17, 2));
              }
            else if (s.size () >= 8 && s.find (':') == 2)
              {
                out[i] = std::stoll (s.substr (6, 2));
              }
            else
              {
                new_valid.set_valid (i, false);
              }
          }
      }
    return series (name_ + "_second", out, std::move (new_valid), index_);
  }

  /* ISO-8601 Temporal Truncation `dt_truncate(unit)` ("year", "month", "day",
     "hour", "minute"), Delimited Token Extraction `str_split_part`, and
     Numeric Substring Extraction `str_extract_digits`
     (Pandas / Polars Series equivalents).  */
  series
  dt_truncate (const std::string &unit) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_truncate requires ISO-8601 utf8_string series");
      }
    std::vector<std::string> out (size ());
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (!new_valid.is_valid (i))
          {
            continue;
          }
        const std::string &s = str_data_[i];
        if ((unit == "year" || unit == "Y" || unit == "y") && s.size () >= 4)
          {
            out[i] = s.substr (0, 4) + "-01-01";
          }
        else if ((unit == "month" || unit == "M") && s.size () >= 7)
          {
            out[i] = s.substr (0, 7) + "-01";
          }
        else if ((unit == "day" || unit == "D" || unit == "d")
                 && s.size () >= 10)
          {
            out[i] = s.substr (0, 10);
          }
        else if ((unit == "hour" || unit == "H" || unit == "h")
                 && s.size () >= 13)
          {
            out[i] = s.substr (0, 13) + ":00:00";
          }
        else if ((unit == "minute" || unit == "min" || unit == "T")
                 && s.size () >= 16)
          {
            out[i] = s.substr (0, 16) + ":00";
          }
        else
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_, std::move (out), std::move (new_valid), index_);
  }

  series
  str_split_part (char delim, std::size_t part_idx) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_split_part requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (!new_valid.is_valid (i))
          {
            continue;
          }
        const std::string &s = str_data_[i];
        std::size_t start = 0;
        std::size_t cur_part = 0;
        bool found = false;
        while (start <= s.size ())
          {
            const std::size_t pos = s.find (delim, start);
            const std::size_t end =
              (pos == std::string::npos) ? s.size () : pos;
            if (cur_part == part_idx)
              {
                out[i] = s.substr (start, end - start);
                found = true;
                break;
              }
            if (pos == std::string::npos)
              {
                break;
              }
            start = pos + 1;
            ++cur_part;
          }
        if (!found)
          {
            new_valid.set_valid (i, false);
          }
      }
    return series (name_, std::move (out), std::move (new_valid), index_);
  }

  series
  str_extract_digits () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_extract_digits requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    validity_bitmap new_valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (!new_valid.is_valid (i))
          {
            continue;
          }
        const std::string &s = str_data_[i];
        std::string digits;
        for (char c : s)
          {
            if (c >= '0' && c <= '9')
              {
                digits.push_back (c);
              }
            else if (!digits.empty ())
              {
                break;
              }
          }
        if (digits.empty ())
          {
            new_valid.set_valid (i, false);
          }
        else
          {
            out[i] = std::move (digits);
          }
      }
    return series (name_, std::move (out), std::move (new_valid), index_);
  }

  /* Extended `.str` String Methods:
     str_find, str_rfind, str_isalnum, str_isalpha, str_isdigit, str_isspace,
     str_islower, str_isupper, str_title, str_capitalize, str_swapcase,
     str_lstrip, str_rstrip, str_removeprefix, str_removesuffix, str_repeat
     (Pandas Series.str / Polars Expr.str equivalents).  */
  series
  str_find (const std::string &sub) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_find requires utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, -1);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const auto pos = str_data_[i].find (sub);
            out[i] =
              (pos == std::string::npos) ? -1 : static_cast<std::int64_t> (pos);
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  str_rfind (const std::string &sub) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_rfind requires utf8_string series");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, -1);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const auto pos = str_data_[i].rfind (sub);
            out[i] =
              (pos == std::string::npos) ? -1 : static_cast<std::int64_t> (pos);
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  template <typename CharPred>
  series
  str_char_predicate (CharPred &&pred, bool require_cased = false) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("string character predicate requires utf8_string series");
      }
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i) && !str_data_[i].empty ())
          {
            bool ok = true;
            bool has_cased = false;
            for (unsigned char c : str_data_[i])
              {
                if (require_cased && std::isalpha (c))
                  {
                    has_cased = true;
                  }
                if (!pred (c))
                  {
                    ok = false;
                    break;
                  }
              }
            out[i] = (ok && (!require_cased || has_cased)) ? 1 : 0;
          }
      }
    return series (name_, out, validity_.clone (), index_);
  }

  series
  str_isalnum () const
  {
    return str_char_predicate ([] (unsigned char c) {
      return std::isalnum (c) != 0;
    });
  }

  series
  str_isalpha () const
  {
    return str_char_predicate ([] (unsigned char c) {
      return std::isalpha (c) != 0;
    });
  }

  series
  str_isdigit () const
  {
    return str_char_predicate ([] (unsigned char c) {
      return std::isdigit (c) != 0;
    });
  }

  series
  str_isspace () const
  {
    return str_char_predicate ([] (unsigned char c) {
      return std::isspace (c) != 0;
    });
  }

  series
  str_islower () const
  {
    return str_char_predicate ([] (unsigned char c) {
      return !std::isupper (c);
    }, true);
  }

  series
  str_isupper () const
  {
    return str_char_predicate ([] (unsigned char c) {
      return !std::islower (c);
    }, true);
  }

  series
  str_capitalize () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_capitalize requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i) && !str_data_[i].empty ())
          {
            std::string s = str_data_[i];
            s[0] = static_cast<char> (
              std::toupper (static_cast<unsigned char> (s[0])));
            for (std::size_t k = 1; k < s.size (); ++k)
              {
                s[k] = static_cast<char> (
                  std::tolower (static_cast<unsigned char> (s[k])));
              }
            out[i] = std::move (s);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_title () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_title requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            std::string s = str_data_[i];
            bool new_word = true;
            for (char &ch : s)
              {
                const auto uc = static_cast<unsigned char> (ch);
                if (std::isalpha (uc))
                  {
                    ch = static_cast<char> (
                      new_word ? std::toupper (uc) : std::tolower (uc));
                    new_word = false;
                  }
                else
                  {
                    new_word = true;
                  }
              }
            out[i] = std::move (s);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_swapcase () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_swapcase requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            std::string s = str_data_[i];
            for (char &ch : s)
              {
                const auto uc = static_cast<unsigned char> (ch);
                if (std::isupper (uc))
                  {
                    ch = static_cast<char> (std::tolower (uc));
                  }
                else if (std::islower (uc))
                  {
                    ch = static_cast<char> (std::toupper (uc));
                  }
              }
            out[i] = std::move (s);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_lstrip () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_lstrip requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            std::size_t p = 0;
            while (p < s.size ()
                   && std::isspace (static_cast<unsigned char> (s[p])))
              {
                ++p;
              }
            out[i] = s.substr (p);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_rstrip () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_rstrip requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            std::size_t len = s.size ();
            while (len > 0
                   && std::isspace (static_cast<unsigned char> (s[len - 1])))
              {
                --len;
              }
            out[i] = s.substr (0, len);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_removeprefix (const std::string &prefix) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_removeprefix requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (!prefix.empty () && s.rfind (prefix, 0) == 0)
              {
                out[i] = s.substr (prefix.size ());
              }
            else
              {
                out[i] = s;
              }
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_removesuffix (const std::string &suffix) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_removesuffix requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (!suffix.empty () && s.size () >= suffix.size ()
                && s.compare (s.size () - suffix.size (),
                              suffix.size (), suffix) == 0)
              {
                out[i] = s.substr (0, s.size () - suffix.size ());
              }
            else
              {
                out[i] = s;
              }
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_repeat (std::size_t repeats) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_repeat requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            std::string acc;
            acc.reserve (str_data_[i].size () * repeats);
            for (std::size_t r = 0; r < repeats; ++r)
              {
                acc += str_data_[i];
              }
            out[i] = std::move (acc);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  /* Extended `.dt` ISO-8601 Calendar Methods:
     dt_quarter, dt_week, dt_is_leap_year, dt_days_in_month,
     dt_is_month_start, dt_is_month_end, dt_is_year_start, dt_is_year_end,
     dt_is_weekend, dt_add_days, dt_diff_days
     (Pandas Series.dt / Polars Expr.dt equivalents).  */
  series
  dt_quarter () const
  {
    const series mo = dt_month ();
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (mo.is_valid (i))
          {
            out[i] = (mo.i64_tensor ()[i] - 1) / 3 + 1;
          }
      }
    return series (name_ + "_quarter", out, mo.validity ().clone (), index_);
  }

  series
  dt_week () const
  {
    const series doy = dt_day_of_year ();
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (doy.is_valid (i))
          {
            out[i] = (doy.i64_tensor ()[i] - 1) / 7 + 1;
          }
      }
    return series (name_ + "_week", out, doy.validity ().clone (), index_);
  }

  series
  dt_is_leap_year () const
  {
    const series yr = dt_year ();
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (yr.is_valid (i))
          {
            const std::int64_t y = yr.i64_tensor ()[i];
            const bool leap =
              ((y % 4) == 0 && (y % 100) != 0) || ((y % 400) == 0);
            out[i] = leap ? 1 : 0;
          }
      }
    return series (name_ + "_is_leap_year", out, yr.validity ().clone (), index_);
  }

  series
  dt_days_in_month () const
  {
    const series yr = dt_year ();
    const series mo = dt_month ();
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap valid = yr.validity () & mo.validity ();
    const int mdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            const std::int64_t y = yr.i64_tensor ()[i];
            const std::int64_t m = mo.i64_tensor ()[i];
            const bool leap =
              ((y % 4) == 0 && (y % 100) != 0) || ((y % 400) == 0);
            out[i] =
              (m == 2 && leap) ? 29 : mdays[std::clamp<std::int64_t> (m - 1, 0, 11)];
          }
      }
    return series (name_ + "_days_in_month", out, std::move (valid), index_);
  }

  series
  dt_is_month_start () const
  {
    const series dy = dt_day ();
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (dy.is_valid (i))
          {
            out[i] = (dy.i64_tensor ()[i] == 1) ? 1 : 0;
          }
      }
    return series (name_ + "_is_month_start", out, dy.validity ().clone (), index_);
  }

  series
  dt_is_month_end () const
  {
    const series dy = dt_day ();
    const series dim = dt_days_in_month ();
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    validity_bitmap valid = dy.validity () & dim.validity ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            out[i] = (dy.i64_tensor ()[i] == dim.i64_tensor ()[i]) ? 1 : 0;
          }
      }
    return series (name_ + "_is_month_end", out, std::move (valid), index_);
  }

  series
  dt_is_year_start () const
  {
    const series mo = dt_month ();
    const series dy = dt_day ();
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    validity_bitmap valid = mo.validity () & dy.validity ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            out[i] =
              (mo.i64_tensor ()[i] == 1 && dy.i64_tensor ()[i] == 1) ? 1 : 0;
          }
      }
    return series (name_ + "_is_year_start", out, std::move (valid), index_);
  }

  series
  dt_is_year_end () const
  {
    const series mo = dt_month ();
    const series dy = dt_day ();
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    validity_bitmap valid = mo.validity () & dy.validity ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            out[i] =
              (mo.i64_tensor ()[i] == 12 && dy.i64_tensor ()[i] == 31) ? 1 : 0;
          }
      }
    return series (name_ + "_is_year_end", out, std::move (valid), index_);
  }

  series
  dt_is_weekend () const
  {
    const series dow = dt_day_of_week ();
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (dow.is_valid (i))
          {
            out[i] = (dow.i64_tensor ()[i] >= 5) ? 1 : 0;
          }
      }
    return series (name_ + "_is_weekend", out, dow.validity ().clone (), index_);
  }

  series
  dt_diff_days (const series &other) const
  {
    return dt_epoch_days () - other.dt_epoch_days ();
  }

  series
  dt_add_days (std::int64_t delta_days) const
  {
    const series ep = dt_epoch_days ();
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (ep.is_valid (i))
          {
            /* Howard Hinnant inverse civil_from_days algorithm.  */
            std::int64_t z = ep.i64_tensor ()[i] + delta_days + 719468;
            const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
            const auto doe = static_cast<unsigned> (z - era * 146097);
            const unsigned yoe =
              (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            std::int64_t y = static_cast<std::int64_t> (yoe) + era * 400;
            const unsigned doy =
              doe - (365 * yoe + yoe / 4 - yoe / 100);
            const unsigned mp = (5 * doy + 2) / 153;
            const unsigned d = doy - (153 * mp + 2) / 5 + 1;
            const unsigned m = mp < 10 ? mp + 3 : mp - 9;
            if (m <= 2)
              {
                ++y;
              }
            char buf[32];
            std::snprintf (buf, sizeof (buf), "%04lld-%02u-%02u",
                           static_cast<long long> (y), m, d);
            out[i] = buf;
          }
      }
    return series (name_, std::move (out), ep.validity ().clone (), index_);
  }

  /* =========================================================================
     Extended String/Regex/Encoding Operations (`str_extract`,
     `str_replace_regex`, `str_match`, `str_count_regex`, `str_partition`,
     `str_center`, `str_encode_hex`, `str_decode_hex`, `str_encode_base64`,
     `str_decode_base64`, `str_json_extract`), Extended Temporal Operations
     (`dt_strftime`, `dt_day_name`, `dt_month_name`, `dt_days_in_year`,
     `dt_iso_year`, `dt_epoch_seconds`, `dt_from_epoch_seconds`,
     `dt_business_day_count`), and Analytical / Window Methods (`pct_rank`,
     `top_k`, `bottom_k`, `hash`, `rle_id`, `peak_max`, `peak_min`,
     `ewm_cov`, `ewm_corr`, `rolling_cov`, `rolling_corr`, `rolling_apply`,
     `expanding_apply`, `search_sorted`).
     ========================================================================= */

  series
  str_extract (const std::string &pattern, std::size_t group_idx = 1) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_extract requires utf8_string series");
      }
    std::regex re (pattern);
    std::vector<std::string> out (size ());
    validity_bitmap valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            std::smatch m;
            if (std::regex_search (str_data_[i], m, re)
                && group_idx < m.size ())
              {
                out[i] = m[group_idx].str ();
              }
            else
              {
                valid.set_valid (i, false);
              }
          }
      }
    return series (name_, std::move (out), std::move (valid), index_);
  }

  series
  str_replace_regex (const std::string &pattern,
                     const std::string &replacement) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_replace_regex requires utf8_string series");
      }
    std::regex re (pattern);
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = std::regex_replace (str_data_[i], re, replacement);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_match (const std::string &pattern) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_match requires utf8_string series");
      }
    std::regex re (pattern);
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            out[i] = std::regex_search (str_data_[i], re) ? 1 : 0;
          }
      }
    return series (name_ + "_match", out, validity_.clone (), index_);
  }

  series
  str_count_regex (const std::string &pattern) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_count_regex requires utf8_string series");
      }
    std::regex re (pattern);
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            auto begin =
              std::sregex_iterator (str_data_[i].begin (),
                                    str_data_[i].end (), re);
            auto end = std::sregex_iterator ();
            out[i] = static_cast<std::int64_t> (std::distance (begin, end));
          }
      }
    return series (name_ + "_count_re", out, validity_.clone (), index_);
  }

  std::array<series, 3>
  str_partition (const std::string &sep) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_partition requires utf8_string series");
      }
    std::vector<std::string> left (size ());
    std::vector<std::string> mid (size ());
    std::vector<std::string> right (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::size_t pos = str_data_[i].find (sep);
            if (pos != std::string::npos)
              {
                left[i] = str_data_[i].substr (0, pos);
                mid[i] = sep;
                right[i] = str_data_[i].substr (pos + sep.size ());
              }
            else
              {
                left[i] = str_data_[i];
              }
          }
      }
    return {
      series (name_ + "_0", std::move (left), validity_.clone (), index_),
      series (name_ + "_1", std::move (mid), validity_.clone (), index_),
      series (name_ + "_2", std::move (right), validity_.clone (), index_)
    };
  }

  series
  str_center (std::size_t width, char fill_char = ' ') const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_center requires utf8_string series");
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            if (s.size () >= width)
              {
                out[i] = s;
              }
            else
              {
                const std::size_t pad = width - s.size ();
                const std::size_t lpad = pad / 2;
                const std::size_t rpad = pad - lpad;
                out[i] = std::string (lpad, fill_char)
                         + s + std::string (rpad, fill_char);
              }
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_encode_hex () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_encode_hex requires utf8_string series");
      }
    static constexpr char hex_digits[] = "0123456789abcdef";
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            std::string h;
            h.reserve (str_data_[i].size () * 2);
            for (unsigned char ch : str_data_[i])
              {
                h.push_back (hex_digits[(ch >> 4u) & 0xFu]);
                h.push_back (hex_digits[ch & 0xFu]);
              }
            out[i] = std::move (h);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_decode_hex () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_decode_hex requires utf8_string series");
      }
    auto hex_val = [] (char c) -> int {
      if (c >= '0' && c <= '9')
        {
          return c - '0';
        }
      if (c >= 'a' && c <= 'f')
        {
          return 10 + (c - 'a');
        }
      if (c >= 'A' && c <= 'F')
        {
          return 10 + (c - 'A');
        }
      return 0;
    };
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            std::string dec;
            dec.reserve (s.size () / 2);
            for (std::size_t k = 0; k + 1 < s.size (); k += 2)
              {
                dec.push_back (
                  static_cast<char> ((hex_val (s[k]) << 4) | hex_val (s[k + 1])));
              }
            out[i] = std::move (dec);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_encode_base64 () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_encode_base64 requires utf8_string series");
      }
    static constexpr char b64[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            const std::string &s = str_data_[i];
            std::string enc;
            std::uint32_t val = 0;
            int valb = -6;
            for (unsigned char c : s)
              {
                val = (val << 8u) + c;
                valb += 8;
                while (valb >= 0)
                  {
                    enc.push_back (b64[(val >> valb) & 0x3Fu]);
                    valb -= 6;
                  }
              }
            if (valb > -6)
              {
                enc.push_back (b64[((val << 8u) >> (valb + 8)) & 0x3Fu]);
              }
            while (enc.size () % 4 != 0)
              {
                enc.push_back ('=');
              }
            out[i] = std::move (enc);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_decode_base64 () const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_decode_base64 requires utf8_string series");
      }
    static constexpr char b64[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::array<int, 256> T;
    T.fill (-1);
    for (int i = 0; i < 64; ++i)
      {
        T[static_cast<unsigned char> (b64[i])] = i;
      }
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (validity_.is_valid (i))
          {
            std::string dec;
            std::uint32_t val = 0;
            int valb = -8;
            for (unsigned char c : str_data_[i])
              {
                if (T[c] == -1)
                  {
                    break;
                  }
                val = (val << 6u) + static_cast<std::uint32_t> (T[c]);
                valb += 6;
                if (valb >= 0)
                  {
                    dec.push_back (static_cast<char> ((val >> valb) & 0xFFu));
                    valb -= 8;
                  }
              }
            out[i] = std::move (dec);
          }
      }
    return series (name_, std::move (out), validity_.clone (), index_);
  }

  series
  str_json_extract (const std::string &key) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("str_json_extract requires utf8_string series");
      }
    const std::string pat =
      "\"" + key + "\"\\s*:\\s*(?:\"([^\"]*)\"|([^,\\}\\s]+))";
    std::regex re (pat);
    std::vector<std::string> out (size ());
    validity_bitmap valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            std::smatch m;
            if (std::regex_search (str_data_[i], m, re))
              {
                out[i] = m[1].matched ? m[1].str () : m[2].str ();
              }
            else
              {
                valid.set_valid (i, false);
              }
          }
      }
    return series (name_ + "_" + key, std::move (out), std::move (valid), index_);
  }

  series
  dt_strftime (const std::string &fmt) const
  {
    if (dtype_ != data_type::utf8_string)
      {
        throw schema_error ("dt_strftime requires ISO-8601 utf8_string series");
      }
    std::vector<std::string> out (size ());
    validity_bitmap valid = validity_.clone ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i) && str_data_[i].size () >= 10)
          {
            const std::string &s = str_data_[i];
            std::string res = fmt;
            auto rep = [&] (const std::string &tok, const std::string &val) {
              std::size_t p = 0;
              while ((p = res.find (tok, p)) != std::string::npos)
                {
                  res.replace (p, tok.size (), val);
                  p += val.size ();
                }
            };
            rep ("%Y", s.substr (0, 4));
            rep ("%m", s.substr (5, 2));
            rep ("%d", s.substr (8, 2));
            rep ("%H", (s.size () >= 13) ? s.substr (11, 2) : "00");
            rep ("%M", (s.size () >= 16) ? s.substr (14, 2) : "00");
            rep ("%S", (s.size () >= 19) ? s.substr (17, 2) : "00");
            out[i] = std::move (res);
          }
        else
          {
            valid.set_valid (i, false);
          }
      }
    return series (name_, std::move (out), std::move (valid), index_);
  }

  series
  dt_day_name () const
  {
    static const char *names[7] = {
      "Monday", "Tuesday", "Wednesday", "Thursday",
      "Friday", "Saturday", "Sunday"
    };
    const series dow = dt_day_of_week ();
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (dow.is_valid (i))
          {
            const std::int64_t idx =
              std::clamp<std::int64_t> (dow.i64_tensor ()[i], 0, 6);
            out[i] = names[idx];
          }
      }
    return series (name_ + "_day_name", std::move (out),
                   dow.validity ().clone (), index_);
  }

  series
  dt_month_name () const
  {
    static const char *names[12] = {
      "January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"
    };
    const series mo = dt_month ();
    std::vector<std::string> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (mo.is_valid (i))
          {
            const std::int64_t idx =
              std::clamp<std::int64_t> (mo.i64_tensor ()[i] - 1, 0, 11);
            out[i] = names[idx];
          }
      }
    return series (name_ + "_month_name", std::move (out),
                   mo.validity ().clone (), index_);
  }

  series
  dt_days_in_year () const
  {
    const series ly = dt_is_leap_year ();
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (ly.is_valid (i))
          {
            out[i] = (ly.bool_tensor ()[i] != 0) ? 366 : 365;
          }
      }
    return series (name_ + "_days_in_year", out, ly.validity ().clone (), index_);
  }

  series
  dt_iso_year () const
  {
    return dt_year ().alias (name_ + "_iso_year");
  }

  series
  dt_epoch_seconds () const
  {
    const series days = dt_epoch_days ();
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (days.is_valid (i))
          {
            std::int64_t secs = days.i64_tensor ()[i] * 86400;
            const std::string &s = str_data_[i];
            if (s.size () >= 19)
              {
                secs += std::stoll (s.substr (11, 2)) * 3600
                        + std::stoll (s.substr (14, 2)) * 60
                        + std::stoll (s.substr (17, 2));
              }
            out[i] = secs;
          }
      }
    return series (name_ + "_epoch_s", out, days.validity ().clone (), index_);
  }

  static series
  dt_from_epoch_seconds (const std::string &name,
                         const nd_tensor<std::int64_t> &epoch_s)
  {
    std::vector<std::string> out (epoch_s.size ());
    for (std::size_t i = 0; i < epoch_s.size (); ++i)
      {
        std::int64_t total = epoch_s.eval_at (i);
        std::int64_t days = total / 86400;
        std::int64_t rem = total % 86400;
        if (rem < 0)
          {
            rem += 86400;
            --days;
          }
        const unsigned hh = static_cast<unsigned> (rem / 3600);
        const unsigned mm = static_cast<unsigned> ((rem % 3600) / 60);
        const unsigned ss = static_cast<unsigned> (rem % 60);
        std::int64_t z = days + 719468;
        const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
        const auto doe = static_cast<unsigned> (z - era * 146097);
        const unsigned yoe =
          (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        std::int64_t y = static_cast<std::int64_t> (yoe) + era * 400;
        const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const unsigned mp = (5 * doy + 2) / 153;
        const unsigned d = doy - (153 * mp + 2) / 5 + 1;
        const unsigned m = mp < 10 ? mp + 3 : mp - 9;
        if (m <= 2)
          {
            ++y;
          }
        char buf[64];
        std::snprintf (buf, sizeof (buf), "%04lld-%02u-%02uT%02u:%02u:%02u",
                       static_cast<long long> (y), m, d, hh, mm, ss);
        out[i] = buf;
      }
    return series (name, std::move (out));
  }

  series
  dt_business_day_count (const series &other) const
  {
    const series e1 = dt_epoch_days ();
    const series e2 = other.dt_epoch_days ();
    if (e1.size () != e2.size ())
      {
        throw shape_error ("dt_business_day_count requires equal series lengths");
      }
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    validity_bitmap valid = e1.validity () & e2.validity ();
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (valid.is_valid (i))
          {
            const std::int64_t a = e1.i64_tensor ()[i];
            const std::int64_t b = e2.i64_tensor ()[i];
            const std::int64_t sgn = (b >= a) ? 1 : -1;
            const std::int64_t lo = std::min (a, b);
            const std::int64_t hi = std::max (a, b);
            std::int64_t cnt = 0;
            for (std::int64_t z = lo; z < hi; ++z)
              {
                std::int64_t dow = (z + 3) % 7;
                if (dow < 0)
                  {
                    dow += 7;
                  }
                if (dow < 5)
                  {
                    ++cnt;
                  }
              }
            out[i] = sgn * cnt;
          }
      }
    return series (name_ + "_busdays", out, std::move (valid), index_);
  }

  series
  pct_rank () const
  {
    const series rk = rank ();
    const std::size_t vc = valid_count ();
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (rk.is_valid (i) && vc > 0)
          {
            out[i] = rk.get_double (i) / static_cast<double> (vc);
          }
      }
    return series (name_ + "_pct_rank", out, rk.validity ().clone (), index_);
  }

  series
  top_k (std::size_t k) const
  {
    std::vector<std::size_t> pos;
    pos.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            pos.push_back (i);
          }
      }
    std::stable_sort (pos.begin (), pos.end (), [&] (std::size_t a, std::size_t b) {
      return get_double (a) > get_double (b);
    });
    if (pos.size () > k)
      {
        pos.resize (k);
      }
    return take (pos);
  }

  series
  bottom_k (std::size_t k) const
  {
    std::vector<std::size_t> pos;
    pos.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            pos.push_back (i);
          }
      }
    std::stable_sort (pos.begin (), pos.end (), [&] (std::size_t a, std::size_t b) {
      return get_double (a) < get_double (b);
    });
    if (pos.size () > k)
      {
        pos.resize (k);
      }
    return take (pos);
  }

  series
  hash () const
  {
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            std::uint64_t h = 14695981039346656037ull;
            const std::string s = format_cell (iloc (i));
            for (unsigned char c : s)
              {
                h ^= static_cast<std::uint64_t> (c);
                h *= 1099511628211ull;
              }
            out[i] = static_cast<std::int64_t> (h & 0x7FFFFFFFFFFFFFFFull);
          }
      }
    return series (name_ + "_hash", out, validity_.clone (), index_);
  }

  series
  rle_id () const
  {
    nd_tensor<std::int64_t> out (shape_type {size ()}, 0);
    std::int64_t cur_id = 0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (i > 0)
          {
            const bool same_valid = (is_valid (i) == is_valid (i - 1));
            const bool same_val =
              same_valid
              && (!is_valid (i)
                  || format_cell (iloc (i)) == format_cell (iloc (i - 1)));
            if (!same_val)
              {
                ++cur_id;
              }
          }
        out[i] = cur_id;
      }
    return series (name_ + "_rle_id", out, validity_bitmap (size (), true), index_);
  }

  series
  peak_max () const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 1; i + 1 < size (); ++i)
      {
        if (is_valid (i - 1) && is_valid (i) && is_valid (i + 1))
          {
            const double v = get_double (i);
            if (v > get_double (i - 1) && v > get_double (i + 1))
              {
                out[i] = 1;
              }
          }
      }
    return series (name_ + "_peak_max", out, validity_.clone (), index_);
  }

  series
  peak_min () const
  {
    nd_tensor<std::uint8_t> out (shape_type {size ()}, 0);
    for (std::size_t i = 1; i + 1 < size (); ++i)
      {
        if (is_valid (i - 1) && is_valid (i) && is_valid (i + 1))
          {
            const double v = get_double (i);
            if (v < get_double (i - 1) && v < get_double (i + 1))
              {
                out[i] = 1;
              }
          }
      }
    return series (name_ + "_peak_min", out, validity_.clone (), index_);
  }

  series
  ewm_cov (const series &other, double alpha) const
  {
    if (size () != other.size () || alpha <= 0.0 || alpha > 1.0)
      {
        throw value_error ("ewm_cov requires matching size and alpha in (0, 1]");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap valid (size (), false);
    double mx = 0.0;
    double my = 0.0;
    double cov_xy = 0.0;
    std::size_t seen = 0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i) && other.is_valid (i))
          {
            const double x = get_double (i);
            const double y = other.get_double (i);
            if (seen == 0)
              {
                mx = x;
                my = y;
              }
            else
              {
                const double dx = x - mx;
                const double dy = y - my;
                mx += alpha * dx;
                my += alpha * dy;
                cov_xy = (1.0 - alpha) * (cov_xy + alpha * dx * dy);
                out[i] = cov_xy;
                valid.set_valid (i, true);
              }
            ++seen;
          }
      }
    return series (name_ + "_ewm_cov", out, std::move (valid), index_);
  }

  series
  ewm_corr (const series &other, double alpha) const
  {
    const series c = ewm_cov (other, alpha);
    const series sx = ewm_std (alpha);
    const series sy = other.ewm_std (alpha);
    return (c / (sx * sy)).alias (name_ + "_ewm_corr");
  }

  series
  rolling_cov (const series &other, std::size_t window,
               std::size_t ddof = 1) const
  {
    if (size () != other.size () || window <= ddof)
      {
        throw value_error ("rolling_cov requires matching sizes and window > ddof");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap valid (size (), false);
    for (std::size_t i = window - 1; i < size (); ++i)
      {
        bool ok = true;
        double sx = 0.0;
        double sy = 0.0;
        for (std::size_t k = i + 1 - window; k <= i; ++k)
          {
            if (!is_valid (k) || !other.is_valid (k))
              {
                ok = false;
                break;
              }
            sx += get_double (k);
            sy += other.get_double (k);
          }
        if (ok)
          {
            const double mx = sx / static_cast<double> (window);
            const double my = sy / static_cast<double> (window);
            double sxy = 0.0;
            for (std::size_t k = i + 1 - window; k <= i; ++k)
              {
                sxy += (get_double (k) - mx) * (other.get_double (k) - my);
              }
            out[i] = sxy / static_cast<double> (window - ddof);
            valid.set_valid (i, true);
          }
      }
    return series (name_ + "_rcov", out, std::move (valid), index_);
  }

  series
  rolling_corr (const series &other, std::size_t window) const
  {
    const series c = rolling_cov (other, window, 1);
    const series sx = rolling_std (window, 1);
    const series sy = other.rolling_std (window, 1);
    return (c / (sx * sy)).alias (name_ + "_rcorr");
  }

  template <typename Func>
  series
  rolling_apply (std::size_t window, Func &&fn) const
  {
    if (window == 0)
      {
        throw value_error ("rolling_apply requires window >= 1");
      }
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap valid (size (), false);
    for (std::size_t i = window - 1; i < size (); ++i)
      {
        nd_tensor<double> win (shape_type {window});
        bool ok = true;
        for (std::size_t k = 0; k < window; ++k)
          {
            const std::size_t idx = i + 1 - window + k;
            if (!is_valid (idx))
              {
                ok = false;
                break;
              }
            win[k] = get_double (idx);
          }
        if (ok)
          {
            out[i] = fn (win);
            valid.set_valid (i, true);
          }
      }
    return series (name_ + "_rapply", out, std::move (valid), index_);
  }

  template <typename Func>
  series
  expanding_apply (Func &&fn) const
  {
    nd_tensor<double> out (shape_type {size ()}, constants::nan);
    validity_bitmap valid (size (), false);
    std::vector<double> vals;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            vals.push_back (get_double (i));
          }
        if (!vals.empty ())
          {
            out[i] = fn (nd_tensor<double> (shape_type {vals.size ()}, vals));
            valid.set_valid (i, true);
          }
      }
    return series (name_ + "_eapply", out, std::move (valid), index_);
  }

  std::size_t
  search_sorted (double value, const std::string &side = "left") const
  {
    std::vector<double> vals;
    vals.reserve (valid_count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (is_valid (i))
          {
            vals.push_back (get_double (i));
          }
      }
    auto it = (side == "right")
      ? std::upper_bound (vals.begin (), vals.end (), value)
      : std::lower_bound (vals.begin (), vals.end (), value);
    return static_cast<std::size_t> (std::distance (vals.begin (), it));
  }
};

} // namespace hyper_math::tabular

#endif /* HYPER_MATH_TABULAR_SERIES_HPP */
