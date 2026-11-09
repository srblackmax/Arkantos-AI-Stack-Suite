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

#ifndef BELLOWS_CORE_DATA_TABLE_HPP
#define BELLOWS_CORE_DATA_TABLE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <hyper_math/hyper_math.hpp>

namespace bellows
{

// Measurement scale of a tabular feature column.
enum class feature_scale : std::uint8_t
{
  real = 0,
  nominal = 1,
  ordinal = 2
};

// Canonical IEEE-754 quiet NaN sentinel representing a missing entry.
inline constexpr double
missing_value () noexcept
{
  return hyper_math::constants::nan;
}

// Predicate testing whether an observation cell is missing.
inline bool
is_missing (double entry) noexcept
{
  return std::isnan (entry);
}

// Metadata describing a single column in a tabular domain.
struct column_descriptor
{
  std::string name = "";
  feature_scale scale = feature_scale::real;
  std::size_t cardinality = 0;
};

// Schema-aware tabular dataset backed by Hyper Math nd_tensor and bitmasks.
class data_table
{
public:
  using tensor_type = hyper_math::tensor::nd_tensor<double>;
  using bitmap_type = hyper_math::validity_bitmap;
  using frame_type = hyper_math::tabular::data_frame;

  data_table () = default;

  data_table (const data_table &other)
    : rows_ (other.rows_),
      cols_ (other.cols_),
      tensor_ (other.tensor_.copy ()),
      validity_ (),
      schema_ (other.schema_)
  {
    other.sync_validity_bitmap ();
    validity_ = other.validity_.clone ();
  }

  data_table (data_table &&) noexcept = default;

  data_table &
  operator= (const data_table &other)
  {
    if (this != &other)
      {
        other.sync_validity_bitmap ();
        rows_ = other.rows_;
        cols_ = other.cols_;
        tensor_ = other.tensor_.copy ();
        validity_ = other.validity_.clone ();
        schema_ = other.schema_;
      }
    return *this;
  }

  data_table &
  operator= (data_table &&) noexcept = default;

  data_table (std::size_t rows, std::size_t cols, double initial_value = 0.0)
    : rows_ (rows),
      cols_ (cols),
      tensor_ (hyper_math::tensor::shape_type{rows, cols}, initial_value),
      validity_ (rows * cols, !is_missing (initial_value)),
      schema_ (cols)
  {
    for (std::size_t c = 0; c < cols_; ++c)
      {
        schema_[c].name = "col_" + std::to_string (c);
      }
  }

  data_table (std::initializer_list<std::initializer_list<double>> matrix)
    : tensor_ (matrix)
  {
    rows_ = tensor_.ndim () >= 1 ? tensor_.shape ()[0] : 0;
    cols_ = tensor_.ndim () >= 2 ? tensor_.shape ()[1] : 0;
    validity_ = bitmap_type (rows_ * cols_, true);
    schema_.resize (cols_);
    for (std::size_t c = 0; c < cols_; ++c)
      {
        schema_[c].name = "col_" + std::to_string (c);
      }
    const double *raw = tensor_.raw_data ();
    for (std::size_t i = 0; i < rows_ * cols_; ++i)
      {
        if (is_missing (raw[i]))
          {
            validity_.set_valid (i, false);
          }
      }
  }

  // Construct directly from a 2D Hyper Math nd_tensor<double>.
  explicit data_table (const tensor_type &t)
  {
    if (t.ndim () != 2)
      {
        throw std::invalid_argument (
          "bellows::data_table: expected a 2D hyper_math::tensor::nd_tensor");
      }
    rows_ = t.shape ()[0];
    cols_ = t.shape ()[1];
    tensor_ = t.copy ();
    validity_ = bitmap_type (rows_ * cols_, true);
    schema_.resize (cols_);
    for (std::size_t c = 0; c < cols_; ++c)
      {
        schema_[c].name = "col_" + std::to_string (c);
      }
    for (std::size_t r = 0; r < rows_; ++r)
      {
        for (std::size_t c = 0; c < cols_; ++c)
          {
            if (is_missing (tensor_ (r, c)))
              {
                validity_.set_valid (r * cols_ + c, false);
              }
          }
      }
  }

  // Construct from a Hyper Math columnar data_frame, encoding string/bool
  // columns as nominal features and preserving validity bitmasks.
  explicit data_table (const frame_type &df)
    : rows_ (df.nrows ()),
      cols_ (df.ncols ()),
      tensor_ (hyper_math::tensor::shape_type{df.nrows (), df.ncols ()}, 0.0),
      validity_ (df.nrows () * df.ncols (), true),
      schema_ (df.ncols ())
  {
    for (std::size_t c = 0; c < cols_; ++c)
      {
        const auto &ser = df.columns ()[c];
        schema_[c].name = ser.name ();
        if (ser.dtype () == hyper_math::tabular::data_type::utf8_string)
          {
            schema_[c].scale = feature_scale::nominal;
            std::map<std::string, std::size_t> codes;
            for (std::size_t r = 0; r < rows_; ++r)
              {
                if (ser.is_null (r))
                  {
                    tensor_ (r, c) = missing_value ();
                    validity_.set_valid (r * cols_ + c, false);
                  }
                else
                  {
                    std::string s =
                      hyper_math::tabular::format_cell (ser.iloc (r));
                    auto it = codes.find (s);
                    std::size_t code = 0;
                    if (it == codes.end ())
                      {
                        code = codes.size ();
                        codes[s] = code;
                      }
                    else
                      {
                        code = it->second;
                      }
                    tensor_ (r, c) = static_cast<double> (code);
                  }
              }
            schema_[c].cardinality = codes.size ();
          }
        else if (ser.dtype () == hyper_math::tabular::data_type::boolean)
          {
            schema_[c].scale = feature_scale::nominal;
            schema_[c].cardinality = 2;
            for (std::size_t r = 0; r < rows_; ++r)
              {
                if (ser.is_null (r))
                  {
                    tensor_ (r, c) = missing_value ();
                    validity_.set_valid (r * cols_ + c, false);
                  }
                else
                  {
                    tensor_ (r, c) =
                      hyper_math::tabular::cell_as_double (ser.iloc (r));
                  }
              }
          }
        else
          {
            schema_[c].scale = feature_scale::real;
            for (std::size_t r = 0; r < rows_; ++r)
              {
                if (ser.is_null (r))
                  {
                    tensor_ (r, c) = missing_value ();
                    validity_.set_valid (r * cols_ + c, false);
                  }
                else
                  {
                    double v =
                      hyper_math::tabular::cell_as_double (ser.iloc (r));
                    tensor_ (r, c) = v;
                    if (is_missing (v))
                      {
                        validity_.set_valid (r * cols_ + c, false);
                      }
                  }
              }
          }
      }
  }

  std::size_t
  row_count () const noexcept
  {
    return rows_;
  }

  std::size_t
  column_count () const noexcept
  {
    return cols_;
  }

  double
  at (std::size_t row, std::size_t col) const
  {
    if (row >= rows_ || col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::at: index out of bounds");
      }
    return tensor_.raw_data ()[row * cols_ + col];
  }

  double &
  at (std::size_t row, std::size_t col)
  {
    if (row >= rows_ || col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::at: index out of bounds");
      }
    return tensor_.raw_data ()[row * cols_ + col];
  }

  void
  set_cell (std::size_t row, std::size_t col, double val)
  {
    if (row >= rows_ || col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::set_cell: index out of bounds");
      }
    std::size_t flat = row * cols_ + col;
    tensor_.raw_data ()[flat] = val;
    validity_.set_valid (flat, !is_missing (val));
  }

  bool
  cell_valid (std::size_t row, std::size_t col) const
  {
    if (row >= rows_ || col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::cell_valid: index out of bounds");
      }
    double v = tensor_.raw_data ()[row * cols_ + col];
    return !is_missing (v);
  }

  std::span<const double>
  row_span (std::size_t row) const
  {
    if (row >= rows_)
      {
        throw std::out_of_range (
          "bellows::data_table::row_span: row index out of bounds");
      }
    return std::span<const double> (tensor_.raw_data () + row * cols_, cols_);
  }

  std::span<double>
  row_span (std::size_t row)
  {
    if (row >= rows_)
      {
        throw std::out_of_range (
          "bellows::data_table::row_span: row index out of bounds");
      }
    return std::span<double> (tensor_.raw_data () + row * cols_, cols_);
  }

  const tensor_type &
  as_tensor () const noexcept
  {
    return tensor_;
  }

  tensor_type &
  as_tensor () noexcept
  {
    return tensor_;
  }

  frame_type
  to_data_frame () const
  {
    std::vector<hyper_math::tabular::series> cols;
    cols.reserve (cols_);
    for (std::size_t c = 0; c < cols_; ++c)
      {
        std::vector<double> vals (rows_);
        for (std::size_t r = 0; r < rows_; ++r)
          {
            vals[r] = at (r, c);
          }
        std::string cname = schema_[c].name.empty ()
                              ? ("col_" + std::to_string (c))
                              : schema_[c].name;
        cols.emplace_back (cname, vals);
      }
    return frame_type (std::move (cols));
  }

  feature_scale
  scale (std::size_t col) const
  {
    if (col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::scale: column index out of bounds");
      }
    return schema_[col].scale;
  }

  void
  set_scale (std::size_t col, feature_scale s, std::size_t card = 0)
  {
    if (col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::set_scale: column index out of bounds");
      }
    schema_[col].scale = s;
    if (card > 0)
      {
        schema_[col].cardinality = card;
      }
    else if (s == feature_scale::nominal || s == feature_scale::ordinal)
      {
        std::size_t inferred = 0;
        for (std::size_t r = 0; r < rows_; ++r)
          {
            double v = at (r, col);
            if (!is_missing (v) && v >= 0.0)
              {
                std::size_t code = static_cast<std::size_t> (v) + 1;
                if (code > inferred)
                  {
                    inferred = code;
                  }
              }
          }
        schema_[col].cardinality = inferred;
      }
  }

  std::size_t
  cardinality (std::size_t col) const
  {
    if (col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::cardinality: column index out of bounds");
      }
    if (schema_[col].cardinality > 0)
      {
        return schema_[col].cardinality;
      }
    std::size_t inferred = 0;
    for (std::size_t r = 0; r < rows_; ++r)
      {
        double v = at (r, col);
        if (!is_missing (v) && v >= 0.0)
          {
            std::size_t code = static_cast<std::size_t> (v) + 1;
            if (code > inferred)
              {
                inferred = code;
              }
          }
      }
    return inferred;
  }

  const std::string &
  column_name (std::size_t col) const
  {
    if (col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::column_name: column index out of bounds");
      }
    return schema_[col].name;
  }

  void
  set_column_name (std::size_t col, std::string name)
  {
    if (col >= cols_)
      {
        throw std::out_of_range (
          "bellows::data_table::set_column_name: column index out of bounds");
      }
    schema_[col].name = std::move (name);
  }

  const std::vector<column_descriptor> &
  schema () const noexcept
  {
    return schema_;
  }

  void
  set_schema (std::vector<column_descriptor> new_schema)
  {
    if (new_schema.size () == cols_)
      {
        schema_ = std::move (new_schema);
      }
  }

  bool
  has_nominal_columns () const noexcept
  {
    for (const auto &col_desc : schema_)
      {
        if (col_desc.scale == feature_scale::nominal)
          {
            return true;
          }
      }
    return false;
  }

  bool
  has_missing_entries () const noexcept
  {
    const double *raw = tensor_.raw_data ();
    std::size_t total = rows_ * cols_;
    for (std::size_t i = 0; i < total; ++i)
      {
        if (is_missing (raw[i]))
          {
            return true;
          }
      }
    return false;
  }

  void
  sync_validity_bitmap () const
  {
    validity_ = bitmap_type (rows_ * cols_, true);
    const double *raw = tensor_.raw_data ();
    for (std::size_t i = 0; i < rows_ * cols_; ++i)
      {
        if (is_missing (raw[i]))
          {
            validity_.set_valid (i, false);
          }
      }
  }

  const bitmap_type &
  validity () const
  {
    sync_validity_bitmap ();
    return validity_;
  }

  data_table
  select_rows (const std::vector<std::size_t> &row_indices) const
  {
    data_table result (row_indices.size (), cols_);
    result.set_schema (schema_);
    for (std::size_t i = 0; i < row_indices.size (); ++i)
      {
        std::size_t src_row = row_indices[i];
        for (std::size_t c = 0; c < cols_; ++c)
          {
            result.set_cell (i, c, at (src_row, c));
          }
      }
    return result;
  }

  data_table
  select_columns (const std::vector<std::size_t> &col_indices) const
  {
    data_table result (rows_, col_indices.size ());
    for (std::size_t j = 0; j < col_indices.size (); ++j)
      {
        std::size_t src_col = col_indices[j];
        result.set_column_name (j, schema_[src_col].name);
        result.set_scale (j, schema_[src_col].scale,
                          schema_[src_col].cardinality);
        for (std::size_t r = 0; r < rows_; ++r)
          {
            result.set_cell (r, j, at (r, src_col));
          }
      }
    return result;
  }

private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  tensor_type tensor_;
  mutable bitmap_type validity_;
  std::vector<column_descriptor> schema_;
};

// Lightweight non-owning view over a data_table.
class table_view
{
public:
  explicit table_view (const data_table &table) noexcept
    : table_ (&table)
  {
  }

  std::size_t
  row_count () const noexcept
  {
    return table_->row_count ();
  }

  std::size_t
  column_count () const noexcept
  {
    return table_->column_count ();
  }

  double
  at (std::size_t row, std::size_t col) const
  {
    return table_->at (row, col);
  }

  std::span<const double>
  row_span (std::size_t row) const
  {
    return table_->row_span (row);
  }

  feature_scale
  scale (std::size_t col) const
  {
    return table_->scale (col);
  }

  std::size_t
  cardinality (std::size_t col) const
  {
    return table_->cardinality (col);
  }

private:
  const data_table *table_;
};

} // namespace bellows

#endif // BELLOWS_CORE_DATA_TABLE_HPP
