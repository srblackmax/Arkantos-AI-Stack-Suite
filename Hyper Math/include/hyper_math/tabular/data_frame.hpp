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

#ifndef HYPER_MATH_TABULAR_DATA_FRAME_HPP
#define HYPER_MATH_TABULAR_DATA_FRAME_HPP

#include "hyper_math/tabular/series.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hyper_math::tabular
{

enum class agg_op
{
  sum,
  mean,
  min,
  max,
  count,
  std_dev,
  var,
  median,
  first,
  last,
  n_unique,
  arg_min,
  arg_max,
  mode,
  skew,
  kurtosis
};

struct agg_spec
{
  std::string column;
  agg_op op;
  std::string alias_name;

  agg_spec (std::string col, agg_op operation, std::string out_alias = "")
    : column (std::move (col)),
      op (operation),
      alias_name (std::move (out_alias))
  {
    if (alias_name.empty ())
      {
        switch (op)
          {
          case agg_op::sum:
            alias_name = column + "_sum";
            break;
          case agg_op::mean:
            alias_name = column + "_mean";
            break;
          case agg_op::min:
            alias_name = column + "_min";
            break;
          case agg_op::max:
            alias_name = column + "_max";
            break;
          case agg_op::count:
            alias_name = column + "_count";
            break;
          case agg_op::std_dev:
            alias_name = column + "_std";
            break;
          case agg_op::var:
            alias_name = column + "_var";
            break;
          case agg_op::median:
            alias_name = column + "_median";
            break;
          case agg_op::first:
            alias_name = column + "_first";
            break;
          case agg_op::last:
            alias_name = column + "_last";
            break;
          case agg_op::n_unique:
            alias_name = column + "_n_unique";
            break;
          case agg_op::arg_min:
            alias_name = column + "_arg_min";
            break;
          case agg_op::arg_max:
            alias_name = column + "_arg_max";
            break;
          case agg_op::mode:
            alias_name = column + "_mode";
            break;
          case agg_op::skew:
            alias_name = column + "_skew";
            break;
          case agg_op::kurtosis:
            alias_name = column + "_kurtosis";
            break;
          }
      }
  }
};

/* Columnar DataFrame unifying Pandas DataFrame and Polars DataFrame.  */
class data_frame
{
private:
  std::vector<series> columns_;
  std::unordered_map<std::string, std::size_t> col_lookup_;
  index_axis index_;

  void
  rebuild_lookup ()
  {
    col_lookup_.clear ();
    for (std::size_t i = 0; i < columns_.size (); ++i)
      {
        const auto &cname = columns_[i].name ();
        if (col_lookup_.find (cname) != col_lookup_.end ())
          {
            throw schema_error ("duplicate column name in data_frame: "
                                + cname);
          }
        col_lookup_[cname] = i;
      }
  }

public:
  data_frame () = default;

  explicit
  data_frame (std::vector<series> cols,
              std::optional<index_axis> idx = std::nullopt)
    : columns_ (std::move (cols)), col_lookup_ (), index_ ()
  {
    if (!columns_.empty ())
      {
        const std::size_t expected_rows = columns_[0].size ();
        for (const auto &c : columns_)
          {
            if (c.size () != expected_rows)
              {
                throw shape_error ("all columns in data_frame must have equal length");
              }
          }
        index_ = idx.value_or (columns_[0].index ());
        for (auto &c : columns_)
          {
            c.set_index (index_);
          }
      }
    rebuild_lookup ();
  }

  data_frame (std::initializer_list<series> cols)
    : data_frame (std::vector<series> (cols))
  {
  }

  std::size_t
  nrows () const noexcept
  {
    return columns_.empty () ? 0 : columns_[0].size ();
  }

  std::size_t
  ncols () const noexcept
  {
    return columns_.size ();
  }

  std::pair<std::size_t, std::size_t>
  shape () const noexcept
  {
    return {nrows (), ncols ()};
  }

  bool
  empty () const noexcept
  {
    return columns_.empty () || nrows () == 0;
  }

  const index_axis &
  index () const noexcept
  {
    return index_;
  }

  void
  set_index (index_axis new_idx)
  {
    if (!columns_.empty () && new_idx.size () != nrows ())
      {
        throw shape_error ("new index_axis size mismatch in data_frame");
      }
    index_ = std::move (new_idx);
    for (auto &c : columns_)
      {
        c.set_index (index_);
      }
  }

  /* Set an existing column as the row index_axis (Pandas set_index).  */
  data_frame
  set_index_column (const std::string &col_name, bool drop_col = true) const
  {
    const series &idx_col = (*this)[col_name];
    std::vector<std::string> lbls (idx_col.size ());
    for (std::size_t i = 0; i < idx_col.size (); ++i)
      {
        lbls[i] = format_cell (idx_col.iloc (i));
      }
    index_axis new_idx (std::move (lbls), col_name);
    data_frame out = drop_col ? drop ({col_name}) : *this;
    out.set_index (std::move (new_idx));
    return out;
  }

  std::vector<std::string>
  column_names () const
  {
    std::vector<std::string> names;
    names.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        names.push_back (c.name ());
      }
    return names;
  }

  const std::vector<series> &
  columns () const noexcept
  {
    return columns_;
  }

  bool
  has_column (const std::string &col_name) const
  {
    return col_lookup_.find (col_name) != col_lookup_.end ();
  }

  const series &
  operator[] (const std::string &col_name) const
  {
    auto it = col_lookup_.find (col_name);
    if (it == col_lookup_.end ())
      {
        throw schema_error ("column '" + col_name + "' not found in data_frame");
      }
    return columns_[it->second];
  }

  series &
  operator[] (const std::string &col_name)
  {
    auto it = col_lookup_.find (col_name);
    if (it == col_lookup_.end ())
      {
        throw schema_error ("column '" + col_name + "' not found in data_frame");
      }
    return columns_[it->second];
  }

  /* Column projection & mutation.  */
  data_frame
  select (const std::vector<std::string> &col_names) const
  {
    std::vector<series> selected;
    selected.reserve (col_names.size ());
    for (const auto &name : col_names)
      {
        selected.push_back ((*this)[name]);
      }
    return data_frame (std::move (selected), index_);
  }

  data_frame
  drop (const std::vector<std::string> &col_names_to_drop) const
  {
    std::unordered_set<std::string> drop_set (col_names_to_drop.begin (),
                                              col_names_to_drop.end ());
    std::vector<series> kept;
    for (const auto &c : columns_)
      {
        if (drop_set.find (c.name ()) == drop_set.end ())
          {
            kept.push_back (c);
          }
      }
    return data_frame (std::move (kept), index_);
  }

  data_frame
  with_column (series new_col) const
  {
    if (!columns_.empty () && new_col.size () != nrows ())
      {
        throw shape_error ("with_column series length does not match data_frame");
      }
    std::vector<series> cols = columns_;
    auto it = col_lookup_.find (new_col.name ());
    if (it != col_lookup_.end ())
      {
        cols[it->second] = std::move (new_col);
      }
    else
      {
        cols.push_back (std::move (new_col));
      }
    return data_frame (std::move (cols),
                       columns_.empty ()
                       ? std::optional<index_axis> (std::nullopt)
                       : std::optional<index_axis> (index_));
  }

  data_frame
  rename (const std::string &old_name, const std::string &new_name) const
  {
    std::vector<series> cols = columns_;
    auto it = col_lookup_.find (old_name);
    if (it == col_lookup_.end ())
      {
        throw schema_error ("rename: column '" + old_name + "' not found");
      }
    cols[it->second].set_name (new_name);
    return data_frame (std::move (cols), index_);
  }

  /* Row selection: take, slice, head, tail, filter, iloc_row, loc_row.  */
  data_frame
  take (const std::vector<std::size_t> &positions) const
  {
    std::vector<series> cols;
    cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        cols.push_back (c.take (positions));
      }
    return data_frame (std::move (cols), index_.take (positions));
  }

  data_frame
  slice (std::size_t offset, std::size_t length) const
  {
    const std::size_t start = std::min (offset, nrows ());
    const std::size_t stop = std::min (start + length, nrows ());
    std::vector<std::size_t> pos (stop - start);
    std::iota (pos.begin (), pos.end (), start);
    return take (pos);
  }

  data_frame
  head (std::size_t n = 5) const
  {
    return slice (0, n);
  }

  data_frame
  tail (std::size_t n = 5) const
  {
    const std::size_t len = std::min (n, nrows ());
    return slice (nrows () - len, len);
  }

  data_frame
  filter (const series &mask) const
  {
    if (mask.dtype () != data_type::boolean || mask.size () != nrows ())
      {
        throw shape_error ("data_frame::filter requires matching boolean series");
      }
    std::vector<std::size_t> pos;
    pos.reserve (nrows ());
    for (std::size_t i = 0; i < nrows (); ++i)
      {
        if (mask.is_valid (i) && mask.bool_tensor ()[i] != 0)
          {
            pos.push_back (i);
          }
      }
    return take (pos);
  }

  std::map<std::string, cell_value>
  iloc_row (std::size_t pos) const
  {
    if (pos >= nrows ())
      {
        throw index_error ("iloc_row index out of bounds");
      }
    std::map<std::string, cell_value> row;
    for (const auto &c : columns_)
      {
        row[c.name ()] = c.iloc (pos);
      }
    return row;
  }

  std::map<std::string, cell_value>
  loc_row (const std::string &label) const
  {
    return iloc_row (index_.first_position_of (label));
  }

  /* Drop rows containing any null across all columns.  */
  data_frame
  drop_nulls () const
  {
    if (columns_.empty ())
      {
        return *this;
      }
    validity_bitmap combined = columns_[0].validity ().clone ();
    for (std::size_t c = 1; c < columns_.size (); ++c)
      {
        combined = combined & columns_[c].validity ();
      }
    std::vector<std::size_t> valid_rows;
    valid_rows.reserve (combined.valid_count ());
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        if (combined.is_valid (r))
          {
            valid_rows.push_back (r);
          }
      }
    return take (valid_rows);
  }

  data_frame
  fill_null (double replacement) const
  {
    std::vector<series> cols;
    cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::float64 || c.dtype () == data_type::int64)
          {
            cols.push_back (c.fill_null (replacement));
          }
        else
          {
            cols.push_back (c);
          }
      }
    return data_frame (std::move (cols), index_);
  }

  /* Sort rows by a column (ascending or descending, nulls placed last).  */
  data_frame
  sort_by (const std::string &col_name, bool ascending = true) const
  {
    const series &key_col = (*this)[col_name];
    std::vector<std::size_t> order (nrows ());
    std::iota (order.begin (), order.end (), 0);

    std::stable_sort (order.begin (), order.end (),
                      [&] (std::size_t a, std::size_t b) {
                        const bool va = key_col.is_valid (a);
                        const bool vb = key_col.is_valid (b);
                        if (!va && !vb)
                          {
                            return false;
                          }
                        if (!va)
                          {
                            return false;
                          }
                        if (!vb)
                          {
                            return true;
                          }
                        if (key_col.dtype () == data_type::utf8_string)
                          {
                            const auto &sa = key_col.str_values ()[a];
                            const auto &sb = key_col.str_values ()[b];
                            return ascending ? (sa < sb) : (sa > sb);
                          }
                        const double da = key_col.get_double (a);
                        const double db = key_col.get_double (b);
                        return ascending ? (da < db) : (da > db);
                      });

    return take (order);
  }

  /* Vectorized Hash Group-By Builder and Aggregator.  */
  class group_by_builder
  {
  private:
    const data_frame &df_;
    std::vector<std::string> keys_;

  public:
    group_by_builder (const data_frame &df, std::vector<std::string> keys)
      : df_ (df), keys_ (std::move (keys))
    {
      if (keys_.empty ())
        {
          throw schema_error ("group_by requires at least one key column");
        }
      for (const auto &k : keys_)
        {
          if (!df_.has_column (k))
            {
              throw schema_error ("group_by key column '" + k + "' not found");
            }
        }
    }

    data_frame
    agg (const std::vector<agg_spec> &specs) const
    {
      /* Partition row indices by composite key hash in appearance order.  */
      std::unordered_map<std::string, std::size_t> group_id_map;
      std::vector<std::vector<std::size_t>> group_buckets;

      const std::size_t n = df_.nrows ();
      if (keys_.size () == 1
          && df_[keys_[0]].dtype () == data_type::utf8_string
          && df_[keys_[0]].null_count () == 0)
        {
          /* Fast path for single non-null string key column.  */
          const auto &str_vec = df_[keys_[0]].str_values ();
          for (std::size_t r = 0; r < n; ++r)
            {
              const std::string &k_str = str_vec[r];
              auto it = group_id_map.find (k_str);
              if (it == group_id_map.end ())
                {
                  const std::size_t gid = group_buckets.size ();
                  group_id_map.emplace (k_str, gid);
                  group_buckets.push_back ({r});
                }
              else
                {
                  group_buckets[it->second].push_back (r);
                }
            }
        }
      else
        {
          for (std::size_t r = 0; r < n; ++r)
            {
              std::string composite_key;
              for (std::size_t k = 0; k < keys_.size (); ++k)
                {
                  if (k > 0)
                    {
                      composite_key.push_back ('\x1f');
                    }
                  composite_key += cell_hash_key (df_[keys_[k]].iloc (r));
                }

              auto it = group_id_map.find (composite_key);
              if (it == group_id_map.end ())
                {
                  const std::size_t gid = group_buckets.size ();
                  group_id_map.emplace (std::move (composite_key), gid);
                  group_buckets.push_back ({r});
                }
              else
                {
                  group_buckets[it->second].push_back (r);
                }
            }
        }

      const std::size_t n_groups = group_buckets.size ();
      std::vector<series> result_cols;

      /* Materialize key columns for each group.  */
      for (std::size_t k = 0; k < keys_.size (); ++k)
        {
          const series &src_key = df_[keys_[k]];
          std::vector<std::size_t> rep_rows (n_groups);
          for (std::size_t g = 0; g < n_groups; ++g)
            {
              rep_rows[g] = group_buckets[g].front ();
            }
          series k_col = src_key.take (rep_rows);
          k_col.set_index (index_axis (n_groups));
          result_cols.push_back (std::move (k_col));
        }

      /* Compute each requested aggregation directly over group buckets.  */
      for (const auto &sp : specs)
        {
          const series &src = df_[sp.column];
          if (src.dtype () == data_type::utf8_string
              && (sp.op == agg_op::first || sp.op == agg_op::last
                  || sp.op == agg_op::min || sp.op == agg_op::max))
            {
              std::vector<std::string> str_vals (n_groups, "");
              validity_bitmap str_valid (n_groups, false);
              for (std::size_t g = 0; g < n_groups; ++g)
                {
                  const auto &bucket = group_buckets[g];
                  if (sp.op == agg_op::first)
                    {
                      for (std::size_t r : bucket)
                        {
                          if (src.is_valid (r))
                            {
                              str_vals[g] = src.str_values ()[r];
                              str_valid.set_valid (g, true);
                              break;
                            }
                        }
                    }
                  else if (sp.op == agg_op::last)
                    {
                      for (std::size_t idx = bucket.size (); idx > 0; --idx)
                        {
                          const std::size_t r = bucket[idx - 1];
                          if (src.is_valid (r))
                            {
                              str_vals[g] = src.str_values ()[r];
                              str_valid.set_valid (g, true);
                              break;
                            }
                        }
                    }
                  else
                    {
                      for (std::size_t r : bucket)
                        {
                          if (src.is_valid (r))
                            {
                              if (!str_valid.is_valid (g))
                                {
                                  str_vals[g] = src.str_values ()[r];
                                  str_valid.set_valid (g, true);
                                }
                              else if (sp.op == agg_op::min
                                       && src.str_values ()[r] < str_vals[g])
                                {
                                  str_vals[g] = src.str_values ()[r];
                                }
                              else if (sp.op == agg_op::max
                                       && src.str_values ()[r] > str_vals[g])
                                {
                                  str_vals[g] = src.str_values ()[r];
                                }
                            }
                        }
                    }
                }
              result_cols.emplace_back (sp.alias_name, std::move (str_vals),
                                        std::move (str_valid),
                                        index_axis (n_groups));
              continue;
            }

          std::vector<double> agg_vals (n_groups, constants::nan);

          for (std::size_t g = 0; g < n_groups; ++g)
            {
              const auto &bucket = group_buckets[g];
              switch (sp.op)
                {
                case agg_op::sum:
                  {
                    double s = 0.0;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            s += src.get_double (r);
                          }
                      }
                    agg_vals[g] = s;
                    break;
                  }
                case agg_op::mean:
                  {
                    double s = 0.0;
                    std::size_t cnt = 0;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            s += src.get_double (r);
                            ++cnt;
                          }
                      }
                    agg_vals[g] =
                      (cnt > 0) ? (s / static_cast<double> (cnt)) : constants::nan;
                    break;
                  }
                case agg_op::min:
                  {
                    double best = constants::inf;
                    bool found = false;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            best = std::min (best, src.get_double (r));
                            found = true;
                          }
                      }
                    agg_vals[g] = found ? best : constants::nan;
                    break;
                  }
                case agg_op::max:
                  {
                    double best = -constants::inf;
                    bool found = false;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            best = std::max (best, src.get_double (r));
                            found = true;
                          }
                      }
                    agg_vals[g] = found ? best : constants::nan;
                    break;
                  }
                case agg_op::count:
                  {
                    std::size_t cnt = 0;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            ++cnt;
                          }
                      }
                    agg_vals[g] = static_cast<double> (cnt);
                    break;
                  }
                case agg_op::std_dev:
                  agg_vals[g] = src.take (bucket).std_dev ();
                  break;
                case agg_op::var:
                  agg_vals[g] = src.take (bucket).var ();
                  break;
                case agg_op::median:
                  agg_vals[g] = src.take (bucket).median ();
                  break;
                case agg_op::arg_min:
                  {
                    const series sub = src.take (bucket);
                    agg_vals[g] =
                      (sub.valid_count () > 0)
                      ? static_cast<double> (sub.arg_min ())
                      : constants::nan;
                    break;
                  }
                case agg_op::arg_max:
                  {
                    const series sub = src.take (bucket);
                    agg_vals[g] =
                      (sub.valid_count () > 0)
                      ? static_cast<double> (sub.arg_max ())
                      : constants::nan;
                    break;
                  }
                case agg_op::mode:
                  {
                    const series sub = src.take (bucket);
                    const series m = sub.mode ();
                    agg_vals[g] =
                      (m.size () > 0 && m.is_valid (0))
                      ? m.get_double (0)
                      : constants::nan;
                    break;
                  }
                case agg_op::skew:
                  {
                    const series sub = src.take (bucket);
                    const std::size_t vc = sub.valid_count ();
                    if (vc < 3)
                      {
                        agg_vals[g] = constants::nan;
                      }
                    else
                      {
                        const double mu = sub.mean ();
                        const double sd = sub.std_dev ();
                        double m3 = 0.0;
                        for (std::size_t i = 0; i < sub.size (); ++i)
                          {
                            if (sub.is_valid (i))
                              {
                                const double z = (sub.get_double (i) - mu) / sd;
                                m3 += z * z * z;
                              }
                          }
                        agg_vals[g] = m3 / static_cast<double> (vc);
                      }
                    break;
                  }
                case agg_op::kurtosis:
                  {
                    const series sub = src.take (bucket);
                    const std::size_t vc = sub.valid_count ();
                    if (vc < 4)
                      {
                        agg_vals[g] = constants::nan;
                      }
                    else
                      {
                        const double mu = sub.mean ();
                        const double sd = sub.std_dev ();
                        double m4 = 0.0;
                        for (std::size_t i = 0; i < sub.size (); ++i)
                          {
                            if (sub.is_valid (i))
                              {
                                const double z = (sub.get_double (i) - mu) / sd;
                                m4 += z * z * z * z;
                              }
                          }
                        agg_vals[g] = m4 / static_cast<double> (vc) - 3.0;
                      }
                    break;
                  }
                case agg_op::first:
                  {
                    double val = constants::nan;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            val = src.get_double (r);
                            break;
                          }
                      }
                    agg_vals[g] = val;
                    break;
                  }
                case agg_op::last:
                  {
                    double val = constants::nan;
                    for (std::size_t idx = bucket.size (); idx > 0; --idx)
                      {
                        const std::size_t r = bucket[idx - 1];
                        if (src.is_valid (r))
                          {
                            val = src.get_double (r);
                            break;
                          }
                      }
                    agg_vals[g] = val;
                    break;
                  }
                case agg_op::n_unique:
                  {
                    std::unordered_set<std::string> seen;
                    for (std::size_t r : bucket)
                      {
                        if (src.is_valid (r))
                          {
                            seen.insert (cell_hash_key (src.iloc (r)));
                          }
                      }
                    agg_vals[g] = static_cast<double> (seen.size ());
                    break;
                  }
                }
            }

          result_cols.emplace_back (sp.alias_name, agg_vals,
                                    index_axis (n_groups));
        }

      return data_frame (std::move (result_cols), index_axis (n_groups));
    }
  };

  group_by_builder
  group_by (std::vector<std::string> keys) const
  {
    return group_by_builder (*this, std::move (keys));
  }

  group_by_builder
  group_by (const std::string &key) const
  {
    return group_by_builder (*this, {key});
  }

  /* Vectorized Hash Join supporting "inner", "left", "outer", "semi", and
     "anti" joins (Pandas merge / Polars join equivalent).  */
  data_frame
  join (const data_frame &right,
        const std::string &left_on,
        const std::string &right_on,
        const std::string &how = "inner") const
  {
    if (how != "inner" && how != "left" && how != "outer"
        && how != "semi" && how != "anti")
      {
        throw hyper_math_error ("unknown join type: " + how);
      }
    const series &l_key = (*this)[left_on];
    const series &r_key = right[right_on];

    /* Build hash multimap on right key.  */
    std::unordered_map<std::string, std::vector<std::size_t>> right_hash;
    for (std::size_t r = 0; r < right.nrows (); ++r)
      {
        if (r_key.is_valid (r))
          {
            right_hash[cell_hash_key (r_key.iloc (r))].push_back (r);
          }
      }

    if (how == "semi" || how == "anti")
      {
        std::vector<std::size_t> kept_left;
        for (std::size_t l = 0; l < nrows (); ++l)
          {
            const bool matched =
              l_key.is_valid (l)
              && right_hash.find (cell_hash_key (l_key.iloc (l)))
                 != right_hash.end ();
            if ((how == "semi" && matched) || (how == "anti" && !matched))
              {
                kept_left.push_back (l);
              }
          }
        return take (kept_left);
      }

    std::vector<std::optional<std::size_t>> left_matches;
    std::vector<std::optional<std::size_t>> right_matches;
    std::vector<bool> right_visited (right.nrows (), false);

    for (std::size_t l = 0; l < nrows (); ++l)
      {
        bool found = false;
        if (l_key.is_valid (l))
          {
            auto it = right_hash.find (cell_hash_key (l_key.iloc (l)));
            if (it != right_hash.end ())
              {
                for (std::size_t r_idx : it->second)
                  {
                    left_matches.push_back (l);
                    right_matches.push_back (r_idx);
                    right_visited[r_idx] = true;
                    found = true;
                  }
              }
          }
        if (!found && (how == "left" || how == "outer"))
          {
            left_matches.push_back (l);
            right_matches.push_back (std::nullopt);
          }
      }

    if (how == "outer")
      {
        for (std::size_t r = 0; r < right.nrows (); ++r)
          {
            if (!right_visited[r])
              {
                left_matches.push_back (std::nullopt);
                right_matches.push_back (r);
              }
          }
      }

    const std::size_t n_out = left_matches.size ();
    auto materialize_joined_series =
      [n_out] (const series &src,
               const std::vector<std::optional<std::size_t>> &row_map,
               const std::string &col_name) -> series {
      validity_bitmap valid (n_out, false);
      if (src.dtype () == data_type::utf8_string)
        {
          std::vector<std::string> vals (n_out, "");
          for (std::size_t i = 0; i < n_out; ++i)
            {
              if (row_map[i].has_value () && src.is_valid (*row_map[i]))
                {
                  vals[i] = src.str_values ()[*row_map[i]];
                  valid.set_valid (i, true);
                }
            }
          return series (col_name, std::move (vals),
                         std::move (valid), index_axis (n_out));
        }
      else
        {
          nd_tensor<double> vals (shape_type {n_out}, constants::nan);
          for (std::size_t i = 0; i < n_out; ++i)
            {
              if (row_map[i].has_value () && src.is_valid (*row_map[i]))
                {
                  vals[i] = src.get_double (*row_map[i]);
                  valid.set_valid (i, true);
                }
            }
          return series (col_name, vals, std::move (valid), index_axis (n_out));
        }
    };

    std::vector<series> out_cols;
    for (const auto &c : columns_)
      {
        if (c.name () == left_on && left_on == right_on && how == "outer")
          {
            validity_bitmap valid (n_out, false);
            if (c.dtype () == data_type::utf8_string)
              {
                std::vector<std::string> vals (n_out, "");
                for (std::size_t i = 0; i < n_out; ++i)
                  {
                    if (left_matches[i].has_value ()
                        && c.is_valid (*left_matches[i]))
                      {
                        vals[i] = c.str_values ()[*left_matches[i]];
                        valid.set_valid (i, true);
                      }
                    else if (right_matches[i].has_value ()
                             && r_key.is_valid (*right_matches[i]))
                      {
                        vals[i] = r_key.str_values ()[*right_matches[i]];
                        valid.set_valid (i, true);
                      }
                  }
                out_cols.emplace_back (c.name (), std::move (vals),
                                       std::move (valid), index_axis (n_out));
              }
            else
              {
                nd_tensor<double> vals (shape_type {n_out}, constants::nan);
                for (std::size_t i = 0; i < n_out; ++i)
                  {
                    if (left_matches[i].has_value ()
                        && c.is_valid (*left_matches[i]))
                      {
                        vals[i] = c.get_double (*left_matches[i]);
                        valid.set_valid (i, true);
                      }
                    else if (right_matches[i].has_value ()
                             && r_key.is_valid (*right_matches[i]))
                      {
                        vals[i] = r_key.get_double (*right_matches[i]);
                        valid.set_valid (i, true);
                      }
                  }
                out_cols.emplace_back (c.name (), vals,
                                       std::move (valid), index_axis (n_out));
              }
          }
        else
          {
            out_cols.push_back (
              materialize_joined_series (c, left_matches, c.name ()));
          }
      }
    for (const auto &c : right.columns ())
      {
        if (c.name () == right_on && left_on == right_on)
          {
            continue;
          }
        std::string r_name =
          has_column (c.name ()) ? (c.name () + "_right") : c.name ();
        out_cols.push_back (
          materialize_joined_series (c, right_matches, r_name));
      }

    return data_frame (std::move (out_cols), index_axis (n_out));
  }

  /* Pivot table: reshapes long table into wide matrix on (index, columns,
     values) (Pandas pivot_table / Polars pivot equivalent).  */
  data_frame
  pivot (const std::string &index_col,
         const std::string &columns_col,
         const std::string &values_col,
         agg_op op = agg_op::sum) const
  {
    const series &idx_s = (*this)[index_col];
    const series &col_s = (*this)[columns_col];
    const series &val_s = (*this)[values_col];

    std::vector<std::string> row_keys;
    std::unordered_map<std::string, std::size_t> row_map;
    std::vector<std::string> col_keys;
    std::unordered_map<std::string, std::size_t> col_map;

    for (std::size_t r = 0; r < nrows (); ++r)
      {
        const std::string rk = format_cell (idx_s.iloc (r));
        const std::string ck = format_cell (col_s.iloc (r));
        if (row_map.find (rk) == row_map.end ())
          {
            row_map[rk] = row_keys.size ();
            row_keys.push_back (rk);
          }
        if (col_map.find (ck) == col_map.end ())
          {
            col_map[ck] = col_keys.size ();
            col_keys.push_back (ck);
          }
      }

    const std::size_t n_r = row_keys.size ();
    const std::size_t n_c = col_keys.size ();
    std::vector<std::vector<std::vector<double>>> buckets (
      n_r, std::vector<std::vector<double>> (n_c));

    for (std::size_t r = 0; r < nrows (); ++r)
      {
        if (val_s.is_valid (r))
          {
            const std::size_t ri = row_map[format_cell (idx_s.iloc (r))];
            const std::size_t ci = col_map[format_cell (col_s.iloc (r))];
            buckets[ri][ci].push_back (val_s.get_double (r));
          }
      }

    std::vector<series> out_cols;
    out_cols.emplace_back (index_col, row_keys, index_axis (n_r));

    for (std::size_t ci = 0; ci < n_c; ++ci)
      {
        std::vector<double> col_vals (n_r, constants::nan);
        for (std::size_t ri = 0; ri < n_r; ++ri)
          {
            const auto &vec = buckets[ri][ci];
            if (!vec.empty ())
              {
                const double s =
                  std::accumulate (vec.begin (), vec.end (), 0.0);
                if (op == agg_op::sum)
                  {
                    col_vals[ri] = s;
                  }
                else if (op == agg_op::mean)
                  {
                    col_vals[ri] = s / static_cast<double> (vec.size ());
                  }
                else if (op == agg_op::count)
                  {
                    col_vals[ri] = static_cast<double> (vec.size ());
                  }
                else if (op == agg_op::min)
                  {
                    col_vals[ri] =
                      *std::min_element (vec.begin (), vec.end ());
                  }
                else if (op == agg_op::max)
                  {
                    col_vals[ri] =
                      *std::max_element (vec.begin (), vec.end ());
                  }
                else if (op == agg_op::first)
                  {
                    col_vals[ri] = vec.front ();
                  }
                else if (op == agg_op::last)
                  {
                    col_vals[ri] = vec.back ();
                  }
                else if (op == agg_op::std_dev)
                  {
                    col_vals[ri] = series ("tmp", vec).std_dev ();
                  }
                else if (op == agg_op::median)
                  {
                    col_vals[ri] = series ("tmp", vec).median ();
                  }
                else if (op == agg_op::n_unique)
                  {
                    col_vals[ri] =
                      static_cast<double> (series ("tmp", vec).n_unique ());
                  }
              }
          }
        out_cols.emplace_back (col_keys[ci], col_vals, index_axis (n_r));
      }

    return data_frame (std::move (out_cols), index_axis (n_r));
  }

  /* Melt: unpivots wide DataFrame into long format (Pandas/Polars melt).  */
  data_frame
  melt (const std::vector<std::string> &id_vars,
        const std::vector<std::string> &value_vars,
        const std::string &var_name = "variable",
        const std::string &value_name = "value") const
  {
    const std::size_t n_in = nrows ();
    const std::size_t n_val_cols = value_vars.size ();
    const std::size_t n_out = n_in * n_val_cols;

    std::vector<std::size_t> repeat_idx;
    repeat_idx.reserve (n_out);
    std::vector<std::string> var_col;
    var_col.reserve (n_out);
    std::vector<double> val_col (n_out, constants::nan);

    std::size_t pos = 0;
    for (const auto &vname : value_vars)
      {
        const series &vs = (*this)[vname];
        for (std::size_t r = 0; r < n_in; ++r)
          {
            repeat_idx.push_back (r);
            var_col.push_back (vname);
            val_col[pos++] = vs.get_double (r);
          }
      }

    std::vector<series> out_cols;
    for (const auto &id_col : id_vars)
      {
        series taken = (*this)[id_col].take (repeat_idx);
        taken.set_index (index_axis (n_out));
        out_cols.push_back (std::move (taken));
      }
    out_cols.emplace_back (var_name, std::move (var_col), index_axis (n_out));
    out_cols.emplace_back (value_name, val_col, index_axis (n_out));

    return data_frame (std::move (out_cols), index_axis (n_out));
  }

  /* Summary statistics table (Pandas/Polars describe equivalent).  */
  data_frame
  describe () const
  {
    std::vector<std::string> metrics {"count", "mean", "std", "min", "median", "max"};
    std::vector<series> out_cols;
    out_cols.emplace_back ("statistic", metrics);

    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::float64 || c.dtype () == data_type::int64)
          {
            std::vector<double> stats {
              static_cast<double> (c.valid_count ()),
              c.mean (),
              c.std_dev (),
              c.min (),
              c.median (),
              c.max ()
            };
            out_cols.emplace_back (c.name (), stats);
          }
      }
    return data_frame (std::move (out_cols));
  }

  /* Deduplicate rows across all columns or a specified subset of columns
     (Pandas drop_duplicates / Polars unique).  */
  data_frame
  drop_duplicates (std::vector<std::string> subset = {}) const
  {
    if (subset.empty ())
      {
        subset = column_names ();
      }
    std::unordered_set<std::string> seen;
    std::vector<std::size_t> kept;
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        std::string key;
        for (std::size_t k = 0; k < subset.size (); ++k)
          {
            if (k > 0)
              {
                key.push_back ('\x1f');
              }
            key += cell_hash_key ((*this)[subset[k]].iloc (r));
          }
        if (seen.insert (key).second)
          {
            kept.push_back (r);
          }
      }
    return take (kept);
  }

  /* Random row sampling without replacement (Pandas/Polars sample).  */
  data_frame
  sample (std::size_t n, random_engine &rng = global_rng ()) const
  {
    if (n > nrows ())
      {
        throw shape_error ("sample size exceeds data_frame row count");
      }
    std::vector<std::size_t> perm = rng.permutation (nrows ());
    perm.resize (n);
    return take (perm);
  }

  /* Pairwise Pearson correlation matrix across numeric columns (Pandas df.corr). */
  data_frame
  corr () const
  {
    std::vector<const series *> num_cols;
    std::vector<std::string> num_names;
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::float64 || c.dtype () == data_type::int64)
          {
            num_cols.push_back (&c);
            num_names.push_back (c.name ());
          }
      }
    const std::size_t k = num_cols.size ();
    std::vector<series> out_cols;
    out_cols.emplace_back ("column", num_names);

    for (std::size_t j = 0; j < k; ++j)
      {
        std::vector<double> col_r (k, 1.0);
        for (std::size_t i = 0; i < k; ++i)
          {
            if (i == j)
              {
                col_r[i] = 1.0;
                continue;
              }
            double sum_i = 0.0;
            double sum_j = 0.0;
            std::size_t pair_cnt = 0;
            for (std::size_t r = 0; r < nrows (); ++r)
              {
                if (num_cols[i]->is_valid (r) && num_cols[j]->is_valid (r))
                  {
                    sum_i += num_cols[i]->get_double (r);
                    sum_j += num_cols[j]->get_double (r);
                    ++pair_cnt;
                  }
              }
            if (pair_cnt < 2)
              {
                col_r[i] = 0.0;
                continue;
              }
            const double mi = sum_i / static_cast<double> (pair_cnt);
            const double mj = sum_j / static_cast<double> (pair_cnt);
            double cov_ij = 0.0;
            double vi = 0.0;
            double vj = 0.0;
            for (std::size_t r = 0; r < nrows (); ++r)
              {
                if (num_cols[i]->is_valid (r) && num_cols[j]->is_valid (r))
                  {
                    const double di = num_cols[i]->get_double (r) - mi;
                    const double dj = num_cols[j]->get_double (r) - mj;
                    cov_ij += di * dj;
                    vi += di * di;
                    vj += dj * dj;
                  }
              }
            const double denom = std::sqrt (vi * vj);
            col_r[i] = (denom > 0.0) ? (cov_ij / denom) : 0.0;
          }
        out_cols.emplace_back (num_names[j], col_r);
      }

    return data_frame (std::move (out_cols), index_axis ( num_names));
  }

  /* Vertical row concatenation of multiple DataFrames (pd.concat / pl.concat). */
  static data_frame
  concat_rows (const std::vector<data_frame> &frames)
  {
    if (frames.empty ())
      {
        return data_frame ();
      }
    const std::vector<std::string> col_names = frames[0].column_names ();
    std::size_t total_rows = 0;
    for (const auto &df : frames)
      {
        if (df.column_names () != col_names)
          {
            throw schema_error ("concat_rows requires identical column schemas");
          }
        total_rows += df.nrows ();
      }

    std::vector<series> merged_cols;
    for (const auto &cname : col_names)
      {
        const data_type dt = frames[0][cname].dtype ();
        validity_bitmap valid (total_rows, false);
        if (dt == data_type::utf8_string)
          {
            std::vector<std::string> vals (total_rows);
            std::size_t pos = 0;
            for (const auto &df : frames)
              {
                const series &s = df[cname];
                for (std::size_t r = 0; r < s.size (); ++r)
                  {
                    vals[pos] = s.str_values ()[r];
                    valid.set_valid (pos, s.is_valid (r));
                    ++pos;
                  }
              }
            merged_cols.emplace_back (cname, std::move (vals), std::move (valid));
          }
        else
          {
            nd_tensor<double> vals (shape_type {total_rows}, constants::nan);
            std::size_t pos = 0;
            for (const auto &df : frames)
              {
                const series &s = df[cname];
                for (std::size_t r = 0; r < s.size (); ++r)
                  {
                    vals[pos] = s.get_double (r);
                    valid.set_valid (pos, s.is_valid (r));
                    ++pos;
                  }
              }
            merged_cols.emplace_back (cname, vals, std::move (valid));
          }
      }

    return data_frame (std::move (merged_cols));
  }

  /* Clean-room Hyper Math Binary Columnar Format (.hmb) serialization and
     deserialization (analogous to Arrow IPC / Parquet for zero-parse overhead). */
  void
  write_hmb (const std::string &filepath) const
  {
    std::ofstream ofs (filepath, std::ios::binary);
    if (!ofs)
      {
        throw parse_error ("unable to open .hmb file for writing: " + filepath);
      }
    const char magic[4] = {'H', 'M', 'B', '1'};
    ofs.write (magic, 4);
    const std::uint64_t nr = nrows ();
    const std::uint64_t nc = ncols ();
    ofs.write (reinterpret_cast<const char *> (&nr), sizeof (nr));
    ofs.write (reinterpret_cast<const char *> (&nc), sizeof (nc));

    for (const auto &c : columns_)
      {
        const std::uint64_t name_len = c.name ().size ();
        ofs.write (reinterpret_cast<const char *> (&name_len), sizeof (name_len));
        ofs.write (c.name ().data (), static_cast<std::streamsize> (name_len));
        const std::uint8_t dt_code =
          (c.dtype () == data_type::utf8_string) ? 1 : 0;
        ofs.write (reinterpret_cast<const char *> (&dt_code), 1);

        for (std::size_t r = 0; r < c.size (); ++r)
          {
            const std::uint8_t vbit = c.is_valid (r) ? 1 : 0;
            ofs.write (reinterpret_cast<const char *> (&vbit), 1);
            if (dt_code == 0)
              {
                const double dv = c.get_double (r);
                ofs.write (reinterpret_cast<const char *> (&dv), sizeof (dv));
              }
            else
              {
                const std::string &sv = c.str_values ()[r];
                const std::uint64_t slen = sv.size ();
                ofs.write (reinterpret_cast<const char *> (&slen), sizeof (slen));
                ofs.write (sv.data (), static_cast<std::streamsize> (slen));
              }
          }
      }
  }

  static data_frame
  read_hmb (const std::string &filepath)
  {
    std::ifstream ifs (filepath, std::ios::binary);
    if (!ifs)
      {
        throw parse_error ("unable to open .hmb file for reading: " + filepath);
      }
    char magic[4] = {};
    ifs.read (magic, 4);
    if (magic[0] != 'H' || magic[1] != 'M' || magic[2] != 'B' || magic[3] != '1')
      {
        throw parse_error ("invalid HMB1 magic header in " + filepath);
      }
    std::uint64_t nr = 0;
    std::uint64_t nc = 0;
    ifs.read (reinterpret_cast<char *> (&nr), sizeof (nr));
    ifs.read (reinterpret_cast<char *> (&nc), sizeof (nc));

    std::vector<series> cols;
    cols.reserve (static_cast<std::size_t> (nc));
    for (std::uint64_t c = 0; c < nc; ++c)
      {
        std::uint64_t name_len = 0;
        ifs.read (reinterpret_cast<char *> (&name_len), sizeof (name_len));
        std::string cname (static_cast<std::size_t> (name_len), '\0');
        ifs.read (cname.data (), static_cast<std::streamsize> (name_len));
        std::uint8_t dt_code = 0;
        ifs.read (reinterpret_cast<char *> (&dt_code), 1);

        validity_bitmap valid (static_cast<std::size_t> (nr), true);
        if (dt_code == 0)
          {
            nd_tensor<double> vals (shape_type {static_cast<std::size_t> (nr)});
            for (std::size_t r = 0; r < static_cast<std::size_t> (nr); ++r)
              {
                std::uint8_t vbit = 0;
                double dv = 0.0;
                ifs.read (reinterpret_cast<char *> (&vbit), 1);
                ifs.read (reinterpret_cast<char *> (&dv), sizeof (dv));
                valid.set_valid (r, vbit != 0);
                vals[r] = dv;
              }
            cols.emplace_back (std::move (cname), vals, std::move (valid));
          }
        else
          {
            std::vector<std::string> vals (static_cast<std::size_t> (nr));
            for (std::size_t r = 0; r < static_cast<std::size_t> (nr); ++r)
              {
                std::uint8_t vbit = 0;
                std::uint64_t slen = 0;
                ifs.read (reinterpret_cast<char *> (&vbit), 1);
                ifs.read (reinterpret_cast<char *> (&slen), sizeof (slen));
                vals[r].resize (static_cast<std::size_t> (slen));
                ifs.read (vals[r].data (), static_cast<std::streamsize> (slen));
                valid.set_valid (r, vbit != 0);
              }
            cols.emplace_back (std::move (cname), std::move (vals),
                               std::move (valid));
          }
      }

    return data_frame (std::move (cols));
  }

  /* RFC-4180 CSV Serialization & Deserialization with automatic type & null
     inference.  */
  std::string
  to_csv_string (char delimiter = ',') const
  {
    std::ostringstream oss;
    for (std::size_t c = 0; c < ncols (); ++c)
      {
        if (c > 0)
          {
            oss << delimiter;
          }
        oss << columns_[c].name ();
      }
    oss << "\n";

    for (std::size_t r = 0; r < nrows (); ++r)
      {
        for (std::size_t c = 0; c < ncols (); ++c)
          {
            if (c > 0)
              {
                oss << delimiter;
              }
            if (columns_[c].is_valid (r))
              {
                oss << format_cell (columns_[c].iloc (r));
              }
          }
        oss << "\n";
      }
    return oss.str ();
  }

  void
  write_csv (const std::string &filepath, char delimiter = ',') const
  {
    std::ofstream ofs (filepath);
    if (!ofs)
      {
        throw parse_error ("unable to open file for writing: " + filepath);
      }
    ofs << to_csv_string (delimiter);
  }

  static data_frame
  from_csv_string (const std::string &csv_content, char delimiter = ',')
  {
    std::istringstream iss (csv_content);
    std::string line;
    auto split_csv_line = [delimiter] (const std::string &ln) {
      std::vector<std::string> fields;
      std::string cur;
      bool in_quotes = false;
      for (char ch : ln)
        {
          if (ch == '"')
            {
              in_quotes = !in_quotes;
            }
          else if (ch == delimiter && !in_quotes)
            {
              fields.push_back (cur);
              cur.clear ();
            }
          else if (ch != '\r')
            {
              cur.push_back (ch);
            }
        }
      fields.push_back (cur);
      return fields;
    };

    if (!std::getline (iss, line))
      {
        return data_frame ();
      }
    const std::vector<std::string> headers = split_csv_line (line);
    const std::size_t n_cols = headers.size ();
    std::vector<std::vector<std::string>> raw_cols (n_cols);

    while (std::getline (iss, line))
      {
        if (line.empty () || line == "\r")
          {
            continue;
          }
        const auto fields = split_csv_line (line);
        for (std::size_t c = 0; c < n_cols; ++c)
          {
            raw_cols[c].push_back (c < fields.size () ? fields[c] : "");
          }
      }

    const std::size_t n_rows = n_cols > 0 ? raw_cols[0].size () : 0;
    std::vector<series> parsed_cols;
    parsed_cols.reserve (n_cols);

    auto is_null_token = [] (const std::string &s) {
      return s.empty () || s == "null" || s == "NULL" || s == "NaN"
             || s == "nan" || s == "NA";
    };

    for (std::size_t c = 0; c < n_cols; ++c)
      {
        bool all_numeric = true;
        validity_bitmap valid (n_rows, true);
        std::vector<double> num_vals (n_rows, constants::nan);

        for (std::size_t r = 0; r < n_rows; ++r)
          {
            const std::string &tok = raw_cols[c][r];
            if (is_null_token (tok))
              {
                valid.set_valid (r, false);
                continue;
              }
            char *endptr = nullptr;
            const double val = std::strtod (tok.c_str (), &endptr);
            if (endptr != tok.c_str () && *endptr == '\0')
              {
                num_vals[r] = val;
              }
            else
              {
                all_numeric = false;
              }
          }

        if (all_numeric)
          {
            nd_tensor<double> t_vals (shape_type {n_rows}, num_vals);
            parsed_cols.emplace_back (headers[c], t_vals, std::move (valid));
          }
        else
          {
            parsed_cols.emplace_back (headers[c], std::move (raw_cols[c]),
                                      std::move (valid));
          }
      }

    return data_frame (std::move (parsed_cols));
  }

  static data_frame
  read_csv (const std::string &filepath, char delimiter = ',')
  {
    std::ifstream ifs (filepath);
    if (!ifs)
      {
        throw parse_error ("unable to open CSV file: " + filepath);
      }
    std::ostringstream buf;
    buf << ifs.rdbuf ();
    return from_csv_string (buf.str (), delimiter);
  }

  /* Formatted ASCII/Unicode table view (Polars/Pandas display style).  */
  std::string
  to_pretty_string (std::size_t max_rows = 20) const
  {
    std::ostringstream oss;
    oss << "shape: (" << nrows () << ", " << ncols () << ")\n";
    if (columns_.empty ())
      {
        oss << "(empty data_frame)\n";
        return oss.str ();
      }

    const std::size_t show_rows = std::min (nrows (), max_rows);
    std::vector<std::size_t> widths (ncols ());
    for (std::size_t c = 0; c < ncols (); ++c)
      {
        widths[c] = std::max (columns_[c].name ().size (),
                              dtype_to_string (columns_[c].dtype ()).size ());
        for (std::size_t r = 0; r < show_rows; ++r)
          {
            widths[c] =
              std::max (widths[c], format_cell (columns_[c].iloc (r)).size ());
          }
        widths[c] += 2;
      }

    auto print_sep = [&] () {
      oss << "+";
      for (std::size_t w : widths)
        {
          oss << std::string (w, '-') << "+";
        }
      oss << "\n";
    };

    print_sep ();
    oss << "|";
    for (std::size_t c = 0; c < ncols (); ++c)
      {
        oss << std::setw (static_cast<int> (widths[c]))
            << columns_[c].name () << "|";
      }
    oss << "\n|";
    for (std::size_t c = 0; c < ncols (); ++c)
      {
        oss << std::setw (static_cast<int> (widths[c]))
            << dtype_to_string (columns_[c].dtype ()) << "|";
      }
    oss << "\n";
    print_sep ();

    for (std::size_t r = 0; r < show_rows; ++r)
      {
        oss << "|";
        for (std::size_t c = 0; c < ncols (); ++c)
          {
            oss << std::setw (static_cast<int> (widths[c]))
                << format_cell (columns_[c].iloc (r)) << "|";
          }
        oss << "\n";
      }
    print_sep ();
    return oss.str ();
  }

  /* Multi-column composite key Hash Join (Pandas merge / Polars join on
     multiple key columns).  */
  data_frame
  join (const data_frame &right,
        const std::vector<std::string> &left_on,
        const std::vector<std::string> &right_on,
        const std::string &how = "inner") const
  {
    if (left_on.empty () || left_on.size () != right_on.size ())
      {
        throw schema_error ("multi-column join requires equal non-empty key lists");
      }
    if (left_on.size () == 1)
      {
        return join (right, left_on[0], right_on[0], how);
      }

    auto make_composite =
      [] (const data_frame &df, const std::vector<std::string> &keys) {
      std::vector<std::string> comp (df.nrows ());
      validity_bitmap valid (df.nrows (), true);
      for (std::size_t r = 0; r < df.nrows (); ++r)
        {
          std::string s;
          for (std::size_t k = 0; k < keys.size (); ++k)
            {
              if (!df[keys[k]].is_valid (r))
                {
                  valid.set_valid (r, false);
                  break;
                }
              if (k > 0)
                {
                  s.push_back ('\x1f');
                }
              s += cell_hash_key (df[keys[k]].iloc (r));
            }
          comp[r] = std::move (s);
        }
      return series ("__join_key__", std::move (comp), std::move (valid));
    };

    data_frame l_aug = with_column (make_composite (*this, left_on));
    data_frame r_aug = right.with_column (make_composite (right, right_on));
    return l_aug.join (r_aug, "__join_key__", "__join_key__", how)
                .drop ({"__join_key__"});
  }

  /* As-Of Time-Series Join (Pandas merge_asof / Polars join_asof).
     Matches each row in `left` on key column `on` to the closest row in
     `right` according to `strategy` ("backward": right_val <= left_val,
     "forward": right_val >= left_val, "nearest": min |right_val - left_val|). */
  data_frame
  join_asof (const data_frame &right,
             const std::string &on,
             const std::string &strategy = "backward") const
  {
    const series &l_key = (*this)[on];
    const series &r_key = right[on];

    std::vector<std::optional<std::size_t>> right_matches (nrows (),
                                                           std::nullopt);
    for (std::size_t l = 0; l < nrows (); ++l)
      {
        if (!l_key.is_valid (l))
          {
            continue;
          }
        const double lv = l_key.get_double (l);
        double best_dist = constants::inf;
        std::optional<std::size_t> best_r = std::nullopt;

        for (std::size_t r = 0; r < right.nrows (); ++r)
          {
            if (!r_key.is_valid (r))
              {
                continue;
              }
            const double rv = r_key.get_double (r);
            if (strategy == "backward")
              {
                if (rv <= lv && (lv - rv) <= best_dist)
                  {
                    best_dist = lv - rv;
                    best_r = r;
                  }
              }
            else if (strategy == "forward")
              {
                if (rv >= lv && (rv - lv) < best_dist)
                  {
                    best_dist = rv - lv;
                    best_r = r;
                  }
              }
            else if (strategy == "nearest")
              {
                const double d = std::abs (lv - rv);
                if (d < best_dist)
                  {
                    best_dist = d;
                    best_r = r;
                  }
              }
            else
              {
                throw hyper_math_error ("unknown join_asof strategy: "
                                        + strategy);
              }
          }
        right_matches[l] = best_r;
      }

    std::vector<series> out_cols = columns_;
    for (const auto &rc : right.columns ())
      {
        if (rc.name () == on)
          {
            continue;
          }
        std::string r_name =
          has_column (rc.name ()) ? (rc.name () + "_right") : rc.name ();
        validity_bitmap valid (nrows (), false);
        if (rc.dtype () == data_type::utf8_string)
          {
            std::vector<std::string> vals (nrows (), "");
            for (std::size_t i = 0; i < nrows (); ++i)
              {
                if (right_matches[i].has_value ()
                    && rc.is_valid (*right_matches[i]))
                  {
                    vals[i] = rc.str_values ()[*right_matches[i]];
                    valid.set_valid (i, true);
                  }
              }
            out_cols.emplace_back (r_name, std::move (vals),
                                   std::move (valid), index_);
          }
        else
          {
            nd_tensor<double> vals (shape_type {nrows ()}, constants::nan);
            for (std::size_t i = 0; i < nrows (); ++i)
              {
                if (right_matches[i].has_value ()
                    && rc.is_valid (*right_matches[i]))
                  {
                    vals[i] = rc.get_double (*right_matches[i]);
                    valid.set_valid (i, true);
                  }
              }
            out_cols.emplace_back (r_name, vals, std::move (valid), index_);
          }
      }

    return data_frame (std::move (out_cols), index_);
  }

  /* Contingency table frequency cross-tabulation (Pandas crosstab).  */
  static data_frame
  crosstab (const series &row_series, const series &col_series)
  {
    if (row_series.size () != col_series.size ())
      {
        throw shape_error ("crosstab series must have equal length");
      }
    data_frame tmp ({
      row_series.alias ("row_key"),
      col_series.alias ("col_key"),
      series ("__ones__", std::vector<double> (row_series.size (), 1.0))
    });
    return tmp.pivot ("row_key", "col_key", "__ones__", agg_op::count)
              .fill_null (0.0);
  }

  /* Top-k / Bottom-k row selection and column renaming (Pandas nlargest,
     nsmallest, rename / Polars top_k, bottom_k, rename).  */
  data_frame
  nlargest (std::size_t n, const std::string &col_name) const
  {
    return sort_by (col_name, false).head (n);
  }

  data_frame
  nsmallest (std::size_t n, const std::string &col_name) const
  {
    return sort_by (col_name, true).head (n);
  }

  data_frame
  rename (const std::map<std::string, std::string> &mapping) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        auto it = mapping.find (c.name ());
        if (it != mapping.end ())
          {
            out_cols.push_back (c.alias (it->second));
          }
        else
          {
            out_cols.push_back (c);
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  /* Explode a delimited string column into multiple rows while replicating
     other column values (Pandas df.explode / Polars df.explode equivalent). */
  data_frame
  explode_delimited (const std::string &col_name, char delimiter = ',') const
  {
    const series &target = (*this)[col_name];
    if (target.dtype () != data_type::utf8_string)
      {
        throw schema_error ("explode_delimited requires a utf8_string column");
      }

    std::vector<std::size_t> src_rows;
    std::vector<std::string> exploded_tokens;
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        if (target.is_null (r))
          {
            src_rows.push_back (r);
            exploded_tokens.emplace_back ("");
            continue;
          }
        const std::string &raw_s = target.str_values ()[r];
        if (raw_s.empty ())
          {
            src_rows.push_back (r);
            exploded_tokens.emplace_back ("");
            continue;
          }
        std::size_t start = 0;
        while (true)
          {
            const std::size_t pos = raw_s.find (delimiter, start);
            if (pos == std::string::npos)
              {
                src_rows.push_back (r);
                exploded_tokens.push_back (raw_s.substr (start));
                break;
              }
            src_rows.push_back (r);
            exploded_tokens.push_back (raw_s.substr (start, pos - start));
            start = pos + 1;
          }
      }

    const std::size_t out_n = src_rows.size ();
    std::vector<series> out_cols;
    out_cols.reserve (ncols ());

    for (const auto &c : columns_)
      {
        if (c.name () == col_name)
          {
            validity_bitmap valid (out_n, true);
            for (std::size_t i = 0; i < out_n; ++i)
              {
                if (target.is_null (src_rows[i]))
                  {
                    valid.set_valid (i, false);
                  }
              }
            out_cols.emplace_back (col_name, std::move (exploded_tokens),
                                   std::move (valid));
          }
        else if (c.dtype () == data_type::utf8_string)
          {
            std::vector<std::string> vals (out_n);
            validity_bitmap valid (out_n, true);
            for (std::size_t i = 0; i < out_n; ++i)
              {
                const std::size_t sr = src_rows[i];
                if (c.is_valid (sr))
                  {
                    vals[i] = c.str_values ()[sr];
                  }
                else
                  {
                    valid.set_valid (i, false);
                  }
              }
            out_cols.emplace_back (c.name (), std::move (vals),
                                   std::move (valid));
          }
        else
          {
            nd_tensor<double> vals (shape_type {out_n}, constants::nan);
            validity_bitmap valid (out_n, true);
            for (std::size_t i = 0; i < out_n; ++i)
              {
                const std::size_t sr = src_rows[i];
                if (c.is_valid (sr))
                  {
                    vals[i] = c.get_double (sr);
                  }
                else
                  {
                    valid.set_valid (i, false);
                  }
              }
            out_cols.emplace_back (c.name (), vals, std::move (valid));
          }
      }

    return data_frame (std::move (out_cols));
  }

  /* One-Hot Categorical Encoding (Pandas pd.get_dummies / Polars to_dummies). */
  data_frame
  get_dummies (const std::string &col_name,
               const std::string &prefix = "") const
  {
    const series &target = (*this)[col_name];
    const series uniq = target.unique ();
    const std::string pfx = prefix.empty () ? col_name : prefix;

    std::vector<series> out_cols;
    for (const auto &c : columns_)
      {
        if (c.name () != col_name)
          {
            out_cols.push_back (c);
          }
      }

    for (std::size_t k = 0; k < uniq.size (); ++k)
      {
        const std::string cat_label = format_cell (uniq.iloc (k));
        nd_tensor<double> ind (shape_type {nrows ()}, 0.0);
        for (std::size_t r = 0; r < nrows (); ++r)
          {
            if (target.is_valid (r)
                && format_cell (target.iloc (r)) == cat_label)
              {
                ind[r] = 1.0;
              }
          }
        out_cols.emplace_back (pfx + "_" + cat_label, ind,
                               validity_bitmap (nrows (), true), index_);
      }
    return data_frame (std::move (out_cols), index_);
  }

  /* Markdown and LaTeX table exporters (Pandas to_markdown / to_latex).  */
  std::string
  to_markdown () const
  {
    std::ostringstream oss;
    oss << "|";
    for (const auto &c : columns_)
      {
        oss << " " << c.name () << " |";
      }
    oss << "\n|";
    for (std::size_t c = 0; c < ncols (); ++c)
      {
        oss << " --- |";
      }
    oss << "\n";
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        oss << "|";
        for (const auto &c : columns_)
          {
            oss << " " << format_cell (c.iloc (r)) << " |";
          }
        oss << "\n";
      }
    return oss.str ();
  }

  std::string
  to_latex () const
  {
    std::ostringstream oss;
    oss << "\\begin{tabular}{" << std::string (ncols (), 'l') << "}\n\\hline\n";
    for (std::size_t c = 0; c < ncols (); ++c)
      {
        if (c > 0)
          {
            oss << " & ";
          }
        oss << columns_[c].name ();
      }
    oss << " \\\\\n\\hline\n";
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        for (std::size_t c = 0; c < ncols (); ++c)
          {
            if (c > 0)
              {
                oss << " & ";
              }
            oss << format_cell (columns_[c].iloc (r));
          }
        oss << " \\\\\n";
      }
    oss << "\\hline\n\\end{tabular}\n";
    return oss.str ();
  }

  /* Streaming Newline-Delimited JSON (.ndjson / .jsonl) Reader & Writer
     (Polars read_ndjson / write_ndjson & Pandas read_json(lines=True)).  */
  void
  write_ndjson (const std::string &filepath) const
  {
    std::ofstream ofs (filepath);
    if (!ofs)
      {
        throw parse_error ("unable to open .ndjson file for writing: "
                           + filepath);
      }
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        ofs << "{";
        for (std::size_t c = 0; c < ncols (); ++c)
          {
            if (c > 0)
              {
                ofs << ",";
              }
            ofs << "\"" << columns_[c].name () << "\":";
            const cell_value cv = columns_[c].iloc (r);
            if (std::holds_alternative<std::monostate> (cv))
              {
                ofs << "null";
              }
            else if (auto *sv = std::get_if<std::string> (&cv))
              {
                ofs << "\"" << *sv << "\"";
              }
            else
              {
                ofs << format_cell (cv);
              }
          }
        ofs << "}\n";
      }
  }

  static data_frame
  read_ndjson (const std::string &filepath)
  {
    std::ifstream ifs (filepath);
    if (!ifs)
      {
        throw parse_error ("unable to open .ndjson file for reading: "
                           + filepath);
      }

    std::vector<std::string> col_order;
    std::unordered_map<std::string, std::vector<std::string>> raw_cols;
    std::size_t row_idx = 0;
    std::string line;

    while (std::getline (ifs, line))
      {
        if (!line.empty () && line.back () == '\r')
          {
            line.pop_back ();
          }
        const std::size_t lbrace = line.find ('{');
        const std::size_t rbrace = line.rfind ('}');
        if (lbrace == std::string::npos || rbrace == std::string::npos
            || rbrace <= lbrace)
          {
            continue;
          }
        const std::string body =
          line.substr (lbrace + 1, rbrace - lbrace - 1);

        std::size_t pos = 0;
        while (pos < body.size ())
          {
            const std::size_t q1 = body.find ('"', pos);
            if (q1 == std::string::npos)
              {
                break;
              }
            const std::size_t q2 = body.find ('"', q1 + 1);
            if (q2 == std::string::npos)
              {
                break;
              }
            const std::string key = body.substr (q1 + 1, q2 - q1 - 1);
            const std::size_t colon = body.find (':', q2 + 1);
            if (colon == std::string::npos)
              {
                break;
              }
            std::size_t v_start = colon + 1;
            while (v_start < body.size () && body[v_start] == ' ')
              {
                ++v_start;
              }

            std::string val_str;
            if (v_start < body.size () && body[v_start] == '"')
              {
                const std::size_t v_end = body.find ('"', v_start + 1);
                if (v_end == std::string::npos)
                  {
                    val_str = body.substr (v_start + 1);
                    pos = std::string::npos;
                  }
                else
                  {
                    val_str = body.substr (v_start + 1, v_end - v_start - 1);
                    pos = body.find (',', v_end + 1);
                  }
              }
            else
              {
                const std::size_t comma = body.find (',', v_start);
                val_str =
                  (comma == std::string::npos)
                  ? body.substr (v_start)
                  : body.substr (v_start, comma - v_start);
                while (!val_str.empty () && val_str.back () == ' ')
                  {
                    val_str.pop_back ();
                  }
                pos = comma;
              }

            if (raw_cols.find (key) == raw_cols.end ())
              {
                col_order.push_back (key);
                raw_cols[key].resize (row_idx, "null");
              }
            raw_cols[key].push_back (val_str);

            if (pos == std::string::npos)
              {
                break;
              }
            ++pos;
          }

        ++row_idx;
        for (const auto &k : col_order)
          {
            if (raw_cols[k].size () < row_idx)
              {
                raw_cols[k].push_back ("null");
              }
          }
      }

    std::vector<series> typed_cols;
    for (const auto &cname : col_order)
      {
        const auto &cells = raw_cols[cname];
        bool all_numeric = true;
        std::vector<double> f_vals (row_idx, constants::nan);
        validity_bitmap valid (row_idx, true);

        for (std::size_t r = 0; r < row_idx; ++r)
          {
            if (cells[r].empty () || cells[r] == "null" || cells[r] == "NaN")
              {
                valid.set_valid (r, false);
                continue;
              }
            char *endptr = nullptr;
            const double d = std::strtod (cells[r].c_str (), &endptr);
            if (endptr != cells[r].c_str () && *endptr == '\0')
              {
                f_vals[r] = d;
              }
            else
              {
                all_numeric = false;
              }
          }

        if (all_numeric)
          {
            typed_cols.emplace_back (cname,
                                     nd_tensor<double> (shape_type {row_idx}, f_vals),
                                     std::move (valid));
          }
        else
          {
            typed_cols.emplace_back (cname, cells, std::move (valid));
          }
      }

    return data_frame (std::move (typed_cols));
  }

  /* Null-coalescing `combine_first(other)` (Pandas DataFrame.combine_first). */
  data_frame
  combine_first (const data_frame &other) const
  {
    if (other.nrows () != nrows ())
      {
        throw shape_error ("combine_first requires DataFrames of equal row count");
      }
    std::vector<series> out_cols;
    for (const auto &c : columns_)
      {
        if (!other.has_column (c.name ()))
          {
            out_cols.push_back (c);
            continue;
          }
        const series &oc = other[c.name ()];
        if (c.dtype () == data_type::utf8_string)
          {
            std::vector<std::string> vals = c.str_values ();
            validity_bitmap valid = c.validity ().clone ();
            for (std::size_t r = 0; r < nrows (); ++r)
              {
                if (!valid.is_valid (r) && oc.is_valid (r))
                  {
                    vals[r] = format_cell (oc.iloc (r));
                    valid.set_valid (r, true);
                  }
              }
            out_cols.emplace_back (c.name (), std::move (vals),
                                   std::move (valid), index_);
          }
        else
          {
            nd_tensor<double> vals (shape_type {nrows ()}, constants::nan);
            validity_bitmap valid = c.validity ().clone ();
            for (std::size_t r = 0; r < nrows (); ++r)
              {
                if (valid.is_valid (r))
                  {
                    vals[r] = c.get_double (r);
                  }
                else if (oc.is_valid (r))
                  {
                    vals[r] = oc.get_double (r);
                    valid.set_valid (r, true);
                  }
              }
            out_cols.emplace_back (c.name (), vals, std::move (valid), index_);
          }
      }
    /* Append any extra columns from `other` not present in `*this`.  */
    for (const auto &oc : other.columns ())
      {
        if (!has_column (oc.name ()))
          {
            out_cols.push_back (oc);
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  /* Multi-column frequency table returning a DataFrame sorted by descending
     frequency with a `"count"` column (Pandas DataFrame.value_counts).  */
  data_frame
  value_counts (std::vector<std::string> subset = {}) const
  {
    if (subset.empty ())
      {
        subset = column_names ();
      }
    data_frame sub_df = select (subset).with_column (
      series ("__count__", std::vector<double> (nrows (), 1.0)));
    return sub_df
      .group_by (subset)
      .agg ({agg_spec ("__count__", agg_op::count, "count")})
      .sort_by ("count", false);
  }

  /* Prefix or suffix all column names in the DataFrame
     (Pandas DataFrame.add_prefix / DataFrame.add_suffix equivalent).  */
  data_frame
  add_prefix (const std::string &prefix) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        out_cols.push_back (c.alias (prefix + c.name ()));
      }
    return data_frame (std::move (out_cols), index_);
  }

  data_frame
  add_suffix (const std::string &suffix) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        out_cols.push_back (c.alias (c.name () + suffix));
      }
    return data_frame (std::move (out_cols), index_);
  }

  /* Filter columns by `data_type` (`select_dtypes`) and remove + return a
     column in-place (`pop`) (Pandas DataFrame equivalents).  */
  data_frame
  select_dtypes (const std::vector<data_type> &include_types,
                 const std::vector<data_type> &exclude_types = {}) const
  {
    std::vector<series> out_cols;
    for (const auto &c : columns_)
      {
        const bool inc_ok =
          include_types.empty ()
          || (std::find (include_types.begin (), include_types.end (),
                         c.dtype ())
              != include_types.end ());
        const bool exc_ok =
          std::find (exclude_types.begin (), exclude_types.end (),
                     c.dtype ())
          == exclude_types.end ();
        if (inc_ok && exc_ok)
          {
            out_cols.push_back (c);
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  series
  pop (const std::string &col_name)
  {
    for (std::size_t i = 0; i < columns_.size (); ++i)
      {
        if (columns_[i].name () == col_name)
          {
            series extracted = std::move (columns_[i]);
            columns_.erase (columns_.begin ()
                            + static_cast<std::ptrdiff_t> (i));
            rebuild_lookup ();
            return extracted;
          }
      }
    throw schema_error ("pop: column not found: " + col_name);
  }

  /* Multi-column functional assignment `assign` and positional column
     insertion `insert_column` (Pandas DataFrame.assign / DataFrame.insert).  */
  data_frame
  assign (const std::vector<series> &new_cols) const
  {
    data_frame out = *this;
    for (const auto &c : new_cols)
      {
        out = out.with_column (c);
      }
    return out;
  }

  data_frame
  insert_column (std::size_t loc, const series &col) const
  {
    if (loc > columns_.size ())
      {
        throw shape_error ("insert_column location out of bounds");
      }
    if (has_column (col.name ()))
      {
        throw schema_error ("insert_column duplicate column: " + col.name ());
      }
    if (!columns_.empty () && col.size () != nrows ())
      {
        throw shape_error ("insert_column series length does not match nrows");
      }
    std::vector<series> out_cols = columns_;
    out_cols.insert (out_cols.begin () + static_cast<std::ptrdiff_t> (loc),
                     col);
    return data_frame (
      std::move (out_cols),
      columns_.empty () ? std::optional<index_axis> (std::nullopt)
                        : std::optional<index_axis> (index_));
  }

  /* Elementwise conditional replacement `where_cond` (keep where cond is true,
     replace with `other_val` where false) and `mask_cond` (replace where cond
     is true) across all numeric columns (Pandas DataFrame.where / mask).  */
  data_frame
  where_cond (const data_frame &cond_df, double other_val) const
  {
    if (cond_df.nrows () != nrows ())
      {
        throw shape_error ("where_cond requires matching row count");
      }
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        if (!cond_df.has_column (c.name ())
            || c.dtype () == data_type::utf8_string)
          {
            out_cols.push_back (c);
            continue;
          }
        const series &cond_col = cond_df[c.name ()];
        nd_tensor<double> vals (shape_type {nrows ()}, constants::nan);
        validity_bitmap valid = c.validity ().clone ();
        for (std::size_t r = 0; r < nrows (); ++r)
          {
            const bool keep =
              cond_col.is_valid (r) && (cond_col.get_double (r) != 0.0);
            if (keep)
              {
                if (valid.is_valid (r))
                  {
                    vals[r] = c.get_double (r);
                  }
              }
            else
              {
                vals[r] = other_val;
                valid.set_valid (r, !std::isnan (other_val));
              }
          }
        out_cols.emplace_back (c.name (), vals, std::move (valid), index_);
      }
    return data_frame (std::move (out_cols), index_);
  }

  data_frame
  mask_cond (const data_frame &cond_df, double other_val) const
  {
    if (cond_df.nrows () != nrows ())
      {
        throw shape_error ("mask_cond requires matching row count");
      }
    std::vector<series> inv_cols;
    for (const auto &cc : cond_df.columns ())
      {
        nd_tensor<std::uint8_t> inv (shape_type {nrows ()}, 0);
        for (std::size_t r = 0; r < nrows (); ++r)
          {
            const bool active = cc.is_valid (r) && (cc.get_double (r) != 0.0);
            inv[r] = active ? 0 : 1;
          }
        inv_cols.emplace_back (cc.name (), inv,
                               validity_bitmap (nrows (), true), index_);
      }
    return where_cond (data_frame (std::move (inv_cols), index_), other_val);
  }

  /* Per-column distinct non-null counts `nunique()` and exact structural/value
     equality check `equals(other)` (Pandas DataFrame equivalents).  */
  series
  nunique () const
  {
    std::vector<std::string> col_labels;
    std::vector<double> counts;
    col_labels.reserve (columns_.size ());
    counts.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        col_labels.push_back (c.name ());
        counts.push_back (static_cast<double> (c.nunique ()));
      }
    return series ("nunique", counts, index_axis (std::move (col_labels)));
  }

  bool
  equals (const data_frame &other) const
  {
    if (nrows () != other.nrows () || ncols () != other.ncols ())
      {
        return false;
      }
    for (std::size_t j = 0; j < ncols (); ++j)
      {
        const series &ca = columns_[j];
        const series &cb = other.columns_[j];
        if (ca.name () != cb.name () || ca.dtype () != cb.dtype ())
          {
            return false;
          }
        for (std::size_t r = 0; r < nrows (); ++r)
          {
            if (ca.is_valid (r) != cb.is_valid (r))
              {
                return false;
              }
            if (ca.is_valid (r)
                && cell_hash_key (ca.iloc (r)) != cell_hash_key (cb.iloc (r)))
              {
                return false;
              }
          }
      }
    return true;
  }

  /* Multi-column lexicographical sort `sort_by(col_names, ascending)`,
     DataFrame <-> 2-D Tensor bridges `to_tensor()` & `from_tensor(mat, cols)`,
     DataFrame `transpose()`, and table-wide temporal lag methods
     `shift(periods)`, `diff(periods)`, `pct_change(periods)`
     (Pandas / Polars DataFrame equivalents).  */
  data_frame
  sort_by (const std::vector<std::string> &col_names,
           std::vector<bool> ascending = {}) const
  {
    if (col_names.empty ())
      {
        throw schema_error ("sort_by requires at least one column name");
      }
    if (ascending.empty ())
      {
        ascending.assign (col_names.size (), true);
      }
    if (ascending.size () != col_names.size ())
      {
        throw schema_error ("sort_by ascending vector size must match col_names");
      }
    for (const auto &cn : col_names)
      {
        if (!has_column (cn))
          {
            throw schema_error ("sort_by column not found: " + cn);
          }
      }

    std::vector<std::size_t> order (nrows ());
    std::iota (order.begin (), order.end (), 0);

    std::stable_sort (
      order.begin (), order.end (), [&] (std::size_t a, std::size_t b) {
        for (std::size_t k = 0; k < col_names.size (); ++k)
          {
            const series &col = (*this)[col_names[k]];
            const bool va = col.is_valid (a);
            const bool vb = col.is_valid (b);
            if (!va && !vb)
              {
                continue;
              }
            if (!va)
              {
                return false;
              }
            if (!vb)
              {
                return true;
              }
            if (col.dtype () == data_type::utf8_string)
              {
                const auto &sa = col.str_values ()[a];
                const auto &sb = col.str_values ()[b];
                if (sa != sb)
                  {
                    return ascending[k] ? (sa < sb) : (sa > sb);
                  }
              }
            else
              {
                const double da = col.get_double (a);
                const double db = col.get_double (b);
                if (da != db)
                  {
                    return ascending[k] ? (da < db) : (da > db);
                  }
              }
          }
        return false;
      });

    return take (order);
  }

  nd_tensor<double>
  to_tensor () const
  {
    const std::size_t nr = nrows ();
    const std::size_t nc = ncols ();
    nd_tensor<double> mat (shape_type {nr, nc}, constants::nan);
    for (std::size_t c = 0; c < nc; ++c)
      {
        const series &col = columns_[c];
        for (std::size_t r = 0; r < nr; ++r)
          {
            if (col.is_valid (r))
              {
                mat (r, c) = col.get_double (r);
              }
          }
      }
    return mat;
  }

  static data_frame
  from_tensor (const nd_tensor<double> &mat,
               std::vector<std::string> col_names = {})
  {
    if (mat.ndim () != 2)
      {
        throw shape_error ("from_tensor requires a 2D tensor");
      }
    const std::size_t nr = mat.shape ()[0];
    const std::size_t nc = mat.shape ()[1];
    if (col_names.empty ())
      {
        col_names.resize (nc);
        for (std::size_t c = 0; c < nc; ++c)
          {
            col_names[c] = "col_" + std::to_string (c);
          }
      }
    if (col_names.size () != nc)
      {
        throw schema_error ("from_tensor col_names size must match tensor columns");
      }

    std::vector<series> cols;
    cols.reserve (nc);
    for (std::size_t c = 0; c < nc; ++c)
      {
        nd_tensor<double> col_vec (shape_type {nr});
        validity_bitmap valid (nr, true);
        for (std::size_t r = 0; r < nr; ++r)
          {
            const double v = mat (r, c);
            col_vec[r] = v;
            if (std::isnan (v))
              {
                valid.set_valid (r, false);
              }
          }
        cols.emplace_back (col_names[c], col_vec, std::move (valid));
      }
    return data_frame (std::move (cols));
  }

  data_frame
  transpose () const
  {
    const std::size_t nr = nrows ();
    const std::size_t nc = ncols ();
    std::vector<std::string> new_row_labels = column_names ();
    index_axis new_index (std::move (new_row_labels));

    std::vector<series> out_cols;
    out_cols.reserve (nr);
    for (std::size_t r = 0; r < nr; ++r)
      {
        nd_tensor<double> vals (shape_type {nc}, constants::nan);
        validity_bitmap valid (nc, false);
        for (std::size_t c = 0; c < nc; ++c)
          {
            if (columns_[c].is_valid (r))
              {
                vals[c] = columns_[c].get_double (r);
                valid.set_valid (c, true);
              }
          }
        out_cols.emplace_back (index_[r], vals, std::move (valid), new_index);
      }
    return data_frame (std::move (out_cols), new_index);
  }

  data_frame
  shift (std::ptrdiff_t periods = 1) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::utf8_string)
          {
            out_cols.push_back (c);
          }
        else
          {
            out_cols.push_back (c.shift (periods));
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  data_frame
  diff (std::size_t periods = 1) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::utf8_string)
          {
            out_cols.push_back (c);
          }
        else
          {
            out_cols.push_back (c.diff (periods));
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  data_frame
  pct_change (std::size_t periods = 1) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::utf8_string)
          {
            out_cols.push_back (c);
          }
        else
          {
            out_cols.push_back (c.pct_change (periods));
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  /* Time-series downsampling & calendar aggregation `resample`
     (Pandas DataFrame.resample / Polars group_by_dynamic equivalent).
     Truncates `date_col` to `unit` ("Y", "M", "D"), groups by the truncated
     calendar bucket, evaluates `specs`, and sorts ascending by `date_col`.  */
  data_frame
  resample (const std::string &date_col,
            const std::string &unit,
            const std::vector<agg_spec> &specs) const
  {
    if (!has_column (date_col))
      {
        throw schema_error ("resample: date_col not found: " + date_col);
      }
    data_frame truncated =
      with_column ((*this)[date_col].dt_truncate (unit).alias (date_col));
    return truncated.group_by (date_col).agg (specs).sort_by (date_col, true);
  }

  /* Extended Table-Wide Window, Cumulative, Math, & Relational Reshaping
     Operations: rolling_mean, rolling_sum, rolling_std, rolling_min,
     rolling_max, ewm_mean, expanding_mean, cum_sum, cum_prod, cum_min,
     cum_max, rank, clip, abs, round, applymap, join_cross, set_index,
     reset_index, reindex, stack, unstack, update, compare
     (Pandas & Polars DataFrame equivalents).  */
  template <typename ColFn>
  data_frame
  apply_numeric_cols (ColFn &&fn) const
  {
    std::vector<series> out_cols;
    out_cols.reserve (columns_.size ());
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::utf8_string)
          {
            out_cols.push_back (c);
          }
        else
          {
            out_cols.push_back (fn (c));
          }
      }
    return data_frame (std::move (out_cols), index_);
  }

  data_frame
  rolling_mean (std::size_t window) const
  {
    return apply_numeric_cols ([window] (const series &c) {
      return c.rolling_mean (window);
    });
  }

  data_frame
  rolling_sum (std::size_t window) const
  {
    return apply_numeric_cols ([window] (const series &c) {
      return c.rolling_sum (window);
    });
  }

  data_frame
  rolling_std (std::size_t window) const
  {
    return apply_numeric_cols ([window] (const series &c) {
      return c.rolling_std (window);
    });
  }

  data_frame
  rolling_min (std::size_t window) const
  {
    return apply_numeric_cols ([window] (const series &c) {
      return c.rolling_min (window);
    });
  }

  data_frame
  rolling_max (std::size_t window) const
  {
    return apply_numeric_cols ([window] (const series &c) {
      return c.rolling_max (window);
    });
  }

  data_frame
  ewm_mean (double alpha) const
  {
    return apply_numeric_cols ([alpha] (const series &c) {
      return c.ewm_mean (alpha);
    });
  }

  data_frame
  expanding_mean () const
  {
    return apply_numeric_cols ([] (const series &c) {
      return c.expanding_mean ();
    });
  }

  data_frame
  cum_sum () const
  {
    return apply_numeric_cols ([] (const series &c) { return c.cum_sum (); });
  }

  data_frame
  cum_prod () const
  {
    return apply_numeric_cols ([] (const series &c) { return c.cum_prod (); });
  }

  data_frame
  cum_min () const
  {
    return apply_numeric_cols ([] (const series &c) { return c.cum_min (); });
  }

  data_frame
  cum_max () const
  {
    return apply_numeric_cols ([] (const series &c) { return c.cum_max (); });
  }

  data_frame
  rank (const std::string &method = "average", bool ascending = true) const
  {
    return apply_numeric_cols ([&] (const series &c) {
      return c.rank (method, ascending);
    });
  }

  data_frame
  clip (double low, double high) const
  {
    return apply_numeric_cols ([low, high] (const series &c) {
      return c.clip (low, high);
    });
  }

  data_frame
  abs () const
  {
    return apply_numeric_cols ([] (const series &c) { return c.abs (); });
  }

  data_frame
  round () const
  {
    return apply_numeric_cols ([] (const series &c) { return c.round (); });
  }

  template <typename ScalarFn>
  data_frame
  applymap (ScalarFn &&fn) const
  {
    return apply_numeric_cols ([&] (const series &c) {
      return c.map_numeric (fn);
    });
  }

  data_frame
  join_cross (const data_frame &other,
              const std::string &suffix = "_right") const
  {
    const std::size_t nr_a = nrows ();
    const std::size_t nr_b = other.nrows ();
    const std::size_t total = nr_a * nr_b;

    std::vector<std::size_t> idx_a (total);
    std::vector<std::size_t> idx_b (total);
    for (std::size_t i = 0; i < nr_a; ++i)
      {
        for (std::size_t j = 0; j < nr_b; ++j)
          {
            idx_a[i * nr_b + j] = i;
            idx_b[i * nr_b + j] = j;
          }
      }

    data_frame left_part = take (idx_a);
    data_frame right_part = other.take (idx_b);
    std::vector<series> out_cols = left_part.columns ();
    for (const auto &rc : right_part.columns ())
      {
        std::string cname = rc.name ();
        if (has_column (cname))
          {
            cname += suffix;
          }
        out_cols.push_back (rc.alias (cname));
      }
    return data_frame (std::move (out_cols));
  }

  data_frame
  set_index (const std::string &col_name, bool drop_col = true) const
  {
    if (!has_column (col_name))
      {
        throw schema_error ("set_index column not found: " + col_name);
      }
    const series &idx_col = (*this)[col_name];
    std::vector<std::string> labels (nrows ());
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        labels[r] = format_cell (idx_col.iloc (r));
      }
    index_axis new_idx (std::move (labels));
    std::vector<series> out_cols;
    for (const auto &c : columns_)
      {
        if (drop_col && c.name () == col_name)
          {
            continue;
          }
        series copy_c = c;
        copy_c.set_index (new_idx);
        out_cols.push_back (std::move (copy_c));
      }
    return data_frame (std::move (out_cols), new_idx);
  }

  data_frame
  reset_index (const std::string &name = "index") const
  {
    std::vector<std::string> idx_vals (nrows ());
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        idx_vals[r] = index_[r];
      }
    index_axis range_idx (nrows ());
    std::vector<series> out_cols;
    out_cols.reserve (ncols () + 1);
    out_cols.emplace_back (name, std::move (idx_vals), range_idx);
    for (const auto &c : columns_)
      {
        series copy_c = c;
        copy_c.set_index (range_idx);
        out_cols.push_back (std::move (copy_c));
      }
    return data_frame (std::move (out_cols), range_idx);
  }

  data_frame
  reindex (const std::vector<std::string> &new_labels,
           double fill_val = constants::nan) const
  {
    const std::size_t m = new_labels.size ();
    index_axis new_idx (new_labels);
    std::vector<series> out_cols;
    out_cols.reserve (ncols ());
    for (const auto &c : columns_)
      {
        if (c.dtype () == data_type::utf8_string)
          {
            std::vector<std::string> vals (m, "");
            validity_bitmap valid (m, false);
            for (std::size_t i = 0; i < m; ++i)
              {
                if (index_.contains (new_labels[i]))
                  {
                    const std::size_t r =
                      index_.first_position_of (new_labels[i]);
                    if (c.is_valid (r))
                      {
                        vals[i] = c.str_values ()[r];
                        valid.set_valid (i, true);
                      }
                  }
              }
            out_cols.emplace_back (c.name (), std::move (vals),
                                   std::move (valid), new_idx);
          }
        else
          {
            nd_tensor<double> vals (shape_type {m}, fill_val);
            validity_bitmap valid (m, !std::isnan (fill_val));
            for (std::size_t i = 0; i < m; ++i)
              {
                if (index_.contains (new_labels[i]))
                  {
                    const std::size_t r =
                      index_.first_position_of (new_labels[i]);
                    if (c.is_valid (r))
                      {
                        vals[i] = c.get_double (r);
                        valid.set_valid (i, true);
                      }
                    else
                      {
                        vals[i] = fill_val;
                        valid.set_valid (i, !std::isnan (fill_val));
                      }
                  }
              }
            out_cols.emplace_back (c.name (), vals, std::move (valid), new_idx);
          }
      }
    return data_frame (std::move (out_cols), new_idx);
  }

  data_frame
  stack () const
  {
    std::vector<std::string> row_lbls;
    std::vector<std::string> col_lbls;
    std::vector<double> vals;
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string && c.is_valid (r))
              {
                row_lbls.push_back (index_[r]);
                col_lbls.push_back (c.name ());
                vals.push_back (c.get_double (r));
              }
          }
      }
    return data_frame ({
      series ("index", std::move (row_lbls)),
      series ("column", std::move (col_lbls)),
      series ("value", std::move (vals))
    });
  }

  data_frame
  unstack (const std::string &index_col = "index",
           const std::string &col_col = "column",
           const std::string &val_col = "value") const
  {
    return pivot (index_col, col_col, val_col);
  }

  data_frame
  update (const data_frame &other) const
  {
    return other.combine_first (*this);
  }

  data_frame
  compare (const data_frame &other) const
  {
    if (nrows () != other.nrows () || ncols () != other.ncols ())
      {
        throw shape_error ("compare requires identical DataFrame shapes");
      }
    std::vector<series> diff_cols;
    for (std::size_t j = 0; j < ncols (); ++j)
      {
        const series &ca = columns_[j];
        const series &cb = other.columns_[j];
        bool any_diff = false;
        for (std::size_t r = 0; r < nrows (); ++r)
          {
            if (ca.is_valid (r) != cb.is_valid (r)
                || (ca.is_valid (r)
                    && cell_hash_key (ca.iloc (r))
                       != cell_hash_key (cb.iloc (r))))
              {
                any_diff = true;
                break;
              }
          }
        if (any_diff)
          {
            diff_cols.push_back (ca.alias (ca.name () + "_self"));
            diff_cols.push_back (cb.alias (cb.name () + "_other"));
          }
      }
    return data_frame (std::move (diff_cols), index_);
  }

  /* =========================================================================
     Relational & Query Operations (`query`, `eval_expr`, `sql_query`,
     `partition_by`, `group_by_dynamic`, `upsample`, `merge_sorted`),
     Serialization (`to_json`, `read_json`, `to_html`), and Table-Wide
     Imputation / Axis Reductions (`fill_forward`, `fill_backward`,
     `interpolate`, `sum`, `mean`, `min`, `max`, `std_dev`, `var`,
     `median`, `quantile`, `skew`, `kurtosis`, `any`, `all`).
     ========================================================================= */

  data_frame
  query (const std::string &predicate_str) const
  {
    /* Supports predicates of the form "col op val" where op in
       {">=", "<=", "==", "!=", ">", "<"}.  */
    std::istringstream iss (predicate_str);
    std::string col_name;
    std::string op;
    std::string rhs_tok;
    if (!(iss >> col_name >> op >> rhs_tok) || !has_column (col_name))
      {
        throw parse_error ("query invalid predicate or unknown column: "
                           + predicate_str);
      }
    const series &col = (*this)[col_name];
    if (col.dtype () == data_type::utf8_string)
      {
        if (rhs_tok.size () >= 2
            && ((rhs_tok.front () == '\'' && rhs_tok.back () == '\'')
                || (rhs_tok.front () == '"' && rhs_tok.back () == '"')))
          {
            rhs_tok = rhs_tok.substr (1, rhs_tok.size () - 2);
          }
        if (op == "==")
          {
            return filter (col == rhs_tok);
          }
        if (op == "!=")
          {
            return filter (col != rhs_tok);
          }
        throw parse_error ("query unsupported string operator: " + op);
      }
    const double val = std::stod (rhs_tok);
    if (op == ">")
      {
        return filter (col > val);
      }
    if (op == ">=")
      {
        return filter (col >= val);
      }
    if (op == "<")
      {
        return filter (col < val);
      }
    if (op == "<=")
      {
        return filter (col <= val);
      }
    if (op == "==")
      {
        return filter (col == val);
      }
    if (op == "!=")
      {
        return filter (col != val);
      }
    throw parse_error ("query unknown comparison operator: " + op);
  }

  series
  eval_expr (const std::string &expr_str) const
  {
    /* Supports binary arithmetic expressions "lhs op rhs" where lhs/rhs are
       column names or numeric literals and op in {"+", "-", "*", "/"}.  */
    std::istringstream iss (expr_str);
    std::string lhs_tok;
    std::string op;
    std::string rhs_tok;
    if (!(iss >> lhs_tok >> op >> rhs_tok))
      {
        if (has_column (lhs_tok))
          {
            return (*this)[lhs_tok];
          }
        throw parse_error ("eval_expr unable to parse: " + expr_str);
      }
    auto resolve = [&] (const std::string &tok) -> series {
      if (has_column (tok))
        {
          return (*this)[tok];
        }
      const double v = std::stod (tok);
      return series ("const", nd_tensor<double> (shape_type {nrows ()}, v), index_);
    };
    const series a = resolve (lhs_tok);
    const series b = resolve (rhs_tok);
    if (op == "+")
      {
        return a + b;
      }
    if (op == "-")
      {
        return a - b;
      }
    if (op == "*")
      {
        return a * b;
      }
    if (op == "/")
      {
        return a / b;
      }
    throw parse_error ("eval_expr unknown operator: " + op);
  }

  data_frame
  sql_query (const std::string &sql) const
  {
    /* Lightweight SQL SELECT subset:
       SELECT col1, col2 FROM self [WHERE col op val] [ORDER BY col [ASC|DESC]] [LIMIT n] */
    std::string s = sql;
    auto find_kw = [&] (const std::string &kw) -> std::size_t {
      return s.find (kw);
    };
    const std::size_t p_sel = find_kw ("SELECT ");
    const std::size_t p_from = find_kw (" FROM ");
    if (p_sel == std::string::npos || p_from == std::string::npos)
      {
        throw parse_error ("sql_query requires SELECT ... FROM ...");
      }
    const std::string cols_part =
      s.substr (p_sel + 7, p_from - (p_sel + 7));
    data_frame cur = *this;

    const std::size_t p_where = find_kw (" WHERE ");
    const std::size_t p_order = find_kw (" ORDER BY ");
    const std::size_t p_limit = find_kw (" LIMIT ");

    if (p_where != std::string::npos)
      {
        const std::size_t end_w =
          std::min (p_order != std::string::npos ? p_order : s.size (),
                    p_limit != std::string::npos ? p_limit : s.size ());
        cur = cur.query (s.substr (p_where + 7, end_w - (p_where + 7)));
      }
    if (p_order != std::string::npos)
      {
        const std::size_t end_o =
          (p_limit != std::string::npos) ? p_limit : s.size ();
        std::istringstream o_ss (s.substr (p_order + 10, end_o - (p_order + 10)));
        std::string o_col;
        std::string o_dir = "ASC";
        o_ss >> o_col >> o_dir;
        cur = cur.sort_by (o_col, o_dir != "DESC");
      }
    if (cols_part != "*")
      {
        std::vector<std::string> sel_cols;
        std::stringstream css (cols_part);
        std::string item;
        while (std::getline (css, item, ','))
          {
            std::size_t a = item.find_first_not_of (" \t");
            std::size_t b = item.find_last_not_of (" \t");
            if (a != std::string::npos)
              {
                sel_cols.push_back (item.substr (a, b - a + 1));
              }
          }
        cur = cur.select (sel_cols);
      }
    if (p_limit != std::string::npos)
      {
        const std::size_t lim =
          static_cast<std::size_t> (std::stoull (s.substr (p_limit + 7)));
        cur = cur.head (lim);
      }
    return cur;
  }

  std::vector<data_frame>
  partition_by (const std::string &col_name) const
  {
    if (!has_column (col_name))
      {
        throw schema_error ("partition_by column not found: " + col_name);
      }
    const series &key_col = (*this)[col_name];
    std::vector<std::string> order_keys;
    std::map<std::string, std::vector<std::size_t>> groups;
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        const std::string k = cell_hash_key (key_col.iloc (r));
        if (groups.find (k) == groups.end ())
          {
            order_keys.push_back (k);
          }
        groups[k].push_back (r);
      }
    std::vector<data_frame> parts;
    parts.reserve (order_keys.size ());
    for (const auto &k : order_keys)
      {
        parts.push_back (take (groups[k]));
      }
    return parts;
  }

  data_frame
  group_by_dynamic (const std::string &time_col,
                    std::int64_t every_step,
                    const std::string &val_col,
                    const std::string &agg = "sum") const
  {
    if (!has_column (time_col) || !has_column (val_col) || every_step <= 0)
      {
        throw schema_error ("group_by_dynamic invalid columns or step");
      }
    const series &tc = (*this)[time_col];
    const series &vc = (*this)[val_col];
    std::map<std::int64_t, std::vector<double>> buckets;
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        if (tc.is_valid (r) && vc.is_valid (r))
          {
            const auto t_val = static_cast<std::int64_t> (tc.get_double (r));
            const std::int64_t b = (t_val / every_step) * every_step;
            buckets[b].push_back (vc.get_double (r));
          }
      }
    std::vector<std::int64_t> out_t;
    std::vector<double> out_v;
    for (const auto &[b, vals] : buckets)
      {
        out_t.push_back (b);
        const nd_tensor<double> vt (shape_type {vals.size ()}, vals);
        if (agg == "sum")
          {
            out_v.push_back (tensor::sum (vt));
          }
        else if (agg == "mean")
          {
            out_v.push_back (tensor::mean (vt));
          }
        else if (agg == "min")
          {
            out_v.push_back (tensor::min (vt));
          }
        else if (agg == "max")
          {
            out_v.push_back (tensor::max (vt));
          }
        else
          {
            throw value_error ("group_by_dynamic unknown agg: " + agg);
          }
      }
    return data_frame ({
      series (time_col, nd_tensor<std::int64_t> (shape_type {out_t.size ()}, out_t)),
      series (val_col, nd_tensor<double> (shape_type {out_v.size ()}, out_v))
    });
  }

  data_frame
  upsample (const std::string &time_col, std::int64_t step) const
  {
    if (!has_column (time_col) || step <= 0 || nrows () == 0)
      {
        throw schema_error ("upsample requires valid time_col and step > 0");
      }
    const series &tc = (*this)[time_col];
    const auto t_min = static_cast<std::int64_t> (tc.min ());
    const auto t_max = static_cast<std::int64_t> (tc.max ());
    std::map<std::int64_t, std::size_t> row_map;
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        if (tc.is_valid (r))
          {
            row_map[static_cast<std::int64_t> (tc.get_double (r))] = r;
          }
      }
    std::vector<std::int64_t> grid;
    for (std::int64_t t = t_min; t <= t_max; t += step)
      {
        grid.push_back (t);
      }
    const std::size_t m = grid.size ();
    std::vector<series> out_cols;
    for (const auto &c : columns_)
      {
        if (c.name () == time_col)
          {
            out_cols.emplace_back (
              time_col, nd_tensor<std::int64_t> (shape_type {m}, grid));
          }
        else
          {
            nd_tensor<double> vals (shape_type {m}, constants::nan);
            validity_bitmap valid (m, false);
            for (std::size_t i = 0; i < m; ++i)
              {
                auto it = row_map.find (grid[i]);
                if (it != row_map.end () && c.is_valid (it->second))
                  {
                    vals[i] = c.get_double (it->second);
                    valid.set_valid (i, true);
                  }
              }
            out_cols.emplace_back (c.name (), vals, std::move (valid));
          }
      }
    return data_frame (std::move (out_cols));
  }

  data_frame
  merge_sorted (const data_frame &other, const std::string &key_col) const
  {
    return concat_rows ({*this, other}).sort_by (key_col, true);
  }

  data_frame
  with_row_index (const std::string &name = "index") const
  {
    nd_tensor<std::int64_t> idx_col (shape_type {nrows ()});
    for (std::size_t i = 0; i < nrows (); ++i)
      {
        idx_col[i] = static_cast<std::int64_t> (i);
      }
    return insert_column (0, series (name, idx_col, index_));
  }

  std::string
  to_json () const
  {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        if (r > 0)
          {
            oss << ",";
          }
        oss << "{";
        for (std::size_t c = 0; c < ncols (); ++c)
          {
            if (c > 0)
              {
                oss << ",";
              }
            const series &col = columns_[c];
            oss << "\"" << col.name () << "\":";
            if (!col.is_valid (r))
              {
                oss << "null";
              }
            else if (col.dtype () == data_type::utf8_string)
              {
                oss << "\"" << col.str_values ()[r] << "\"";
              }
            else if (col.dtype () == data_type::boolean)
              {
                oss << (col.bool_tensor ()[r] ? "true" : "false");
              }
            else if (col.dtype () == data_type::int64)
              {
                oss << col.i64_tensor ()[r];
              }
            else
              {
                oss << col.f64_tensor ()[r];
              }
          }
        oss << "}";
      }
    oss << "]";
    return oss.str ();
  }

  static data_frame
  read_json (const std::string &json_str)
  {
    std::vector<std::string> col_order;
    std::map<std::string, std::vector<std::string>> raw_cols;
    std::size_t pos = 0;
    std::size_t row_cnt = 0;
    while ((pos = json_str.find ('{', pos)) != std::string::npos)
      {
        const std::size_t end_obj = json_str.find ('}', pos);
        if (end_obj == std::string::npos)
          {
            break;
          }
        const std::string obj = json_str.substr (pos + 1, end_obj - pos - 1);
        std::regex kv_re ("\"([^\"]+)\"\\s*:\\s*(\"[^\"]*\"|[^,\\s]+)");
        auto it = std::sregex_iterator (obj.begin (), obj.end (), kv_re);
        auto end = std::sregex_iterator ();
        for (; it != end; ++it)
          {
            const std::string k = (*it)[1].str ();
            const std::string v = (*it)[2].str ();
            if (raw_cols.find (k) == raw_cols.end ())
              {
                col_order.push_back (k);
              }
            raw_cols[k].resize (row_cnt, "null");
            raw_cols[k].push_back (v);
          }
        ++row_cnt;
        for (auto &[k, vec] : raw_cols)
          {
            if (vec.size () < row_cnt)
              {
                vec.push_back ("null");
              }
          }
        pos = end_obj + 1;
      }
    std::vector<series> cols;
    for (const auto &k : col_order)
      {
        const auto &vec = raw_cols[k];
        bool is_str = false;
        for (const auto &v : vec)
          {
            if (!v.empty () && v.front () == '"')
              {
                is_str = true;
                break;
              }
          }
        if (is_str)
          {
            std::vector<std::string> sv (row_cnt);
            validity_bitmap valid (row_cnt, true);
            for (std::size_t r = 0; r < row_cnt; ++r)
              {
                if (vec[r] == "null")
                  {
                    valid.set_valid (r, false);
                  }
                else if (vec[r].size () >= 2 && vec[r].front () == '"')
                  {
                    sv[r] = vec[r].substr (1, vec[r].size () - 2);
                  }
                else
                  {
                    sv[r] = vec[r];
                  }
              }
            cols.emplace_back (k, std::move (sv), std::move (valid));
          }
        else
          {
            nd_tensor<double> fv (shape_type {row_cnt}, constants::nan);
            validity_bitmap valid (row_cnt, true);
            for (std::size_t r = 0; r < row_cnt; ++r)
              {
                if (vec[r] == "null")
                  {
                    valid.set_valid (r, false);
                  }
                else
                  {
                    fv[r] = std::stod (vec[r]);
                  }
              }
            cols.emplace_back (k, fv, std::move (valid));
          }
      }
    return data_frame (std::move (cols));
  }

  std::string
  to_html () const
  {
    std::ostringstream oss;
    oss << "<table><thead><tr>";
    for (const auto &c : columns_)
      {
        oss << "<th>" << c.name () << "</th>";
      }
    oss << "</tr></thead><tbody>";
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        oss << "<tr>";
        for (const auto &c : columns_)
          {
            oss << "<td>" << format_cell (c.iloc (r)) << "</td>";
          }
        oss << "</tr>";
      }
    oss << "</tbody></table>";
    return oss.str ();
  }

  data_frame
  fill_forward () const
  {
    std::vector<series> out;
    out.reserve (ncols ());
    for (const auto &c : columns_)
      {
        out.push_back (c.fill_forward ());
      }
    return data_frame (std::move (out), index_);
  }

  data_frame
  fill_backward () const
  {
    std::vector<series> out;
    out.reserve (ncols ());
    for (const auto &c : columns_)
      {
        out.push_back (c.fill_backward ());
      }
    return data_frame (std::move (out), index_);
  }

  data_frame
  interpolate () const
  {
    std::vector<series> out;
    out.reserve (ncols ());
    for (const auto &c : columns_)
      {
        out.push_back (
          (c.dtype () == data_type::utf8_string) ? c : c.interpolate ());
      }
    return data_frame (std::move (out), index_);
  }

  series
  sum (std::size_t axis = 0) const
  {
    if (axis == 0)
      {
        std::vector<std::string> lbls;
        std::vector<double> vals;
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string)
              {
                lbls.push_back (c.name ());
                vals.push_back (c.sum ());
              }
          }
        return series ("sum",
                       nd_tensor<double> (shape_type {vals.size ()}, vals),
                       index_axis (std::move (lbls)));
      }
    nd_tensor<double> vals (shape_type {nrows ()}, 0.0);
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string && c.is_valid (r))
              {
                vals[r] += c.get_double (r);
              }
          }
      }
    return series ("sum", vals, index_);
  }

  series
  mean (std::size_t axis = 0) const
  {
    if (axis == 0)
      {
        std::vector<std::string> lbls;
        std::vector<double> vals;
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string)
              {
                lbls.push_back (c.name ());
                vals.push_back (c.mean ());
              }
          }
        return series ("mean",
                       nd_tensor<double> (shape_type {vals.size ()}, vals),
                       index_axis (std::move (lbls)));
      }
    nd_tensor<double> vals (shape_type {nrows ()}, constants::nan);
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        double s = 0.0;
        std::size_t cnt = 0;
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string && c.is_valid (r))
              {
                s += c.get_double (r);
                ++cnt;
              }
          }
        if (cnt > 0)
          {
            vals[r] = s / static_cast<double> (cnt);
          }
      }
    return series ("mean", vals, index_);
  }

  series
  min (std::size_t axis = 0) const
  {
    if (axis == 0)
      {
        std::vector<std::string> lbls;
        std::vector<double> vals;
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string)
              {
                lbls.push_back (c.name ());
                vals.push_back (c.min ());
              }
          }
        return series ("min",
                       nd_tensor<double> (shape_type {vals.size ()}, vals),
                       index_axis (std::move (lbls)));
      }
    nd_tensor<double> vals (shape_type {nrows ()}, constants::inf);
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string && c.is_valid (r))
              {
                vals[r] = std::min (vals[r], c.get_double (r));
              }
          }
      }
    return series ("min", vals, index_);
  }

  series
  max (std::size_t axis = 0) const
  {
    if (axis == 0)
      {
        std::vector<std::string> lbls;
        std::vector<double> vals;
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string)
              {
                lbls.push_back (c.name ());
                vals.push_back (c.max ());
              }
          }
        return series ("max",
                       nd_tensor<double> (shape_type {vals.size ()}, vals),
                       index_axis (std::move (lbls)));
      }
    nd_tensor<double> vals (shape_type {nrows ()}, -constants::inf);
    for (std::size_t r = 0; r < nrows (); ++r)
      {
        for (const auto &c : columns_)
          {
            if (c.dtype () != data_type::utf8_string && c.is_valid (r))
              {
                vals[r] = std::max (vals[r], c.get_double (r));
              }
          }
      }
    return series ("max", vals, index_);
  }

  series
  std_dev (std::size_t ddof = 1) const
  {
    std::vector<std::string> lbls;
    std::vector<double> vals;
    for (const auto &c : columns_)
      {
        if (c.dtype () != data_type::utf8_string)
          {
            lbls.push_back (c.name ());
            vals.push_back (c.std_dev (ddof));
          }
      }
    return series ("std",
                   nd_tensor<double> (shape_type {vals.size ()}, vals),
                   index_axis (std::move (lbls)));
  }

  series
  var (std::size_t ddof = 1) const
  {
    std::vector<std::string> lbls;
    std::vector<double> vals;
    for (const auto &c : columns_)
      {
        if (c.dtype () != data_type::utf8_string)
          {
            lbls.push_back (c.name ());
            vals.push_back (c.var (ddof));
          }
      }
    return series ("var",
                   nd_tensor<double> (shape_type {vals.size ()}, vals),
                   index_axis (std::move (lbls)));
  }

  series
  median () const
  {
    std::vector<std::string> lbls;
    std::vector<double> vals;
    for (const auto &c : columns_)
      {
        if (c.dtype () != data_type::utf8_string)
          {
            lbls.push_back (c.name ());
            vals.push_back (c.median ());
          }
      }
    return series ("median",
                   nd_tensor<double> (shape_type {vals.size ()}, vals),
                   index_axis (std::move (lbls)));
  }

  series
  quantile (double q_pct) const
  {
    std::vector<std::string> lbls;
    std::vector<double> vals;
    for (const auto &c : columns_)
      {
        if (c.dtype () != data_type::utf8_string)
          {
            lbls.push_back (c.name ());
            vals.push_back (c.quantile (q_pct));
          }
      }
    return series ("quantile",
                   nd_tensor<double> (shape_type {vals.size ()}, vals),
                   index_axis (std::move (lbls)));
  }

  series
  skew () const
  {
    std::vector<std::string> lbls;
    std::vector<double> vals;
    for (const auto &c : columns_)
      {
        if (c.dtype () != data_type::utf8_string)
          {
            lbls.push_back (c.name ());
            vals.push_back (c.skew ());
          }
      }
    return series ("skew",
                   nd_tensor<double> (shape_type {vals.size ()}, vals),
                   index_axis (std::move (lbls)));
  }

  series
  kurtosis () const
  {
    std::vector<std::string> lbls;
    std::vector<double> vals;
    for (const auto &c : columns_)
      {
        if (c.dtype () != data_type::utf8_string)
          {
            lbls.push_back (c.name ());
            vals.push_back (c.kurtosis ());
          }
      }
    return series ("kurtosis",
                   nd_tensor<double> (shape_type {vals.size ()}, vals),
                   index_axis (std::move (lbls)));
  }

  series
  any () const
  {
    std::vector<std::string> lbls;
    nd_tensor<std::uint8_t> vals (shape_type {ncols ()}, 0);
    for (std::size_t j = 0; j < ncols (); ++j)
      {
        lbls.push_back (columns_[j].name ());
        vals[j] = columns_[j].any () ? 1 : 0;
      }
    return series ("any", vals, validity_bitmap (ncols (), true),
                   index_axis (std::move (lbls)));
  }

  series
  all () const
  {
    std::vector<std::string> lbls;
    nd_tensor<std::uint8_t> vals (shape_type {ncols ()}, 0);
    for (std::size_t j = 0; j < ncols (); ++j)
      {
        lbls.push_back (columns_[j].name ());
        vals[j] = columns_[j].all () ? 1 : 0;
      }
    return series ("all", vals, validity_bitmap (ncols (), true),
                   index_axis (std::move (lbls)));
  }
};

inline std::ostream &
operator<< (std::ostream &os, const data_frame &df)
{
  os << df.to_pretty_string ();
  return os;
}

} // namespace hyper_math::tabular

#endif /* HYPER_MATH_TABULAR_DATA_FRAME_HPP */
