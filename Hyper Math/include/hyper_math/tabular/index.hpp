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

#ifndef HYPER_MATH_TABULAR_INDEX_HPP
#define HYPER_MATH_TABULAR_INDEX_HPP

#include "hyper_math/core/config.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace hyper_math::tabular
{

/* Bi-directional hash-indexed label axis supporting O(1) RangeIndex mode,
   lazy hash-table materialization (`loc`), and automatic index alignment
   across Series and DataFrames.  */
class index_axis
{
private:
  std::size_t size_ = 0;
  bool is_range_ = true;
  mutable std::vector<std::string> labels_;
  mutable std::unordered_map<std::string, std::vector<std::size_t>> lookup_;
  mutable bool lookup_built_ = false;
  std::string name_;

  void
  ensure_labels () const
  {
    if (is_range_ && labels_.size () != size_)
      {
        labels_.resize (size_);
        for (std::size_t i = 0; i < size_; ++i)
          {
            labels_[i] = std::to_string (i);
          }
      }
  }

  void
  ensure_lookup () const
  {
    if (!lookup_built_)
      {
        ensure_labels ();
        lookup_.clear ();
        lookup_.reserve (labels_.size ());
        for (std::size_t i = 0; i < labels_.size (); ++i)
          {
            lookup_[labels_[i]].push_back (i);
          }
        lookup_built_ = true;
      }
  }

  static bool
  parse_nonneg_size_t (const std::string &s, std::size_t &out) noexcept
  {
    if (s.empty () || (s.size () > 1 && s[0] == '0'))
      {
        return false;
      }
    std::size_t v = 0;
    for (char c : s)
      {
        if (c < '0' || c > '9')
          {
            return false;
          }
        const std::size_t digit = static_cast<std::size_t> (c - '0');
        if (v > (std::numeric_limits<std::size_t>::max () - digit) / 10)
          {
            return false;
          }
        v = v * 10 + digit;
      }
    out = v;
    return true;
  }

public:
  index_axis () = default;

  /* O(1) zero-allocation RangeIndex constructor: [0, 1, ..., n-1].  */
  explicit
  index_axis (std::size_t n, std::string axis_name = "")
    : size_ (n),
      is_range_ (true),
      labels_ (),
      lookup_ (),
      lookup_built_ (false),
      name_ (std::move (axis_name))
  {
  }

  explicit
  index_axis (std::vector<std::string> labels, std::string axis_name = "")
    : size_ (labels.size ()),
      is_range_ (false),
      labels_ (std::move (labels)),
      lookup_ (),
      lookup_built_ (false),
      name_ (std::move (axis_name))
  {
  }

  std::size_t
  size () const noexcept
  {
    return size_;
  }

  bool
  empty () const noexcept
  {
    return size_ == 0;
  }

  bool
  is_range () const noexcept
  {
    return is_range_;
  }

  const std::string &
  name () const noexcept
  {
    return name_;
  }

  const std::vector<std::string> &
  labels () const
  {
    ensure_labels ();
    return labels_;
  }

  const std::string &
  operator[] (std::size_t pos) const
  {
    if (pos >= size_)
      {
        throw index_error ("index_axis position out of bounds");
      }
    ensure_labels ();
    return labels_[pos];
  }

  bool
  contains (const std::string &label) const
  {
    if (is_range_)
      {
        std::size_t idx = 0;
        return parse_nonneg_size_t (label, idx) && idx < size_;
      }
    ensure_lookup ();
    return lookup_.find (label) != lookup_.end ();
  }

  std::size_t
  first_position_of (const std::string &label) const
  {
    if (is_range_)
      {
        std::size_t idx = 0;
        if (parse_nonneg_size_t (label, idx) && idx < size_)
          {
            return idx;
          }
        throw index_error ("label '" + label + "' not found in index_axis");
      }
    ensure_lookup ();
    auto it = lookup_.find (label);
    if (it == lookup_.end () || it->second.empty ())
      {
        throw index_error ("label '" + label + "' not found in index_axis");
      }
    return it->second.front ();
  }

  const std::vector<std::size_t> &
  positions_of (const std::string &label) const
  {
    ensure_lookup ();
    auto it = lookup_.find (label);
    if (it == lookup_.end ())
      {
        throw index_error ("label '" + label + "' not found in index_axis");
      }
    return it->second;
  }

  index_axis
  slice (std::size_t offset, std::size_t length) const
  {
    const std::size_t start = std::min (offset, size_);
    const std::size_t stop = std::min (start + length, size_);
    if (is_range_ && start == 0)
      {
        return index_axis (stop, name_);
      }
    ensure_labels ();
    std::vector<std::string> sub (
      labels_.begin () + static_cast<std::ptrdiff_t> (start),
      labels_.begin () + static_cast<std::ptrdiff_t> (stop));
    return index_axis (std::move (sub), name_);
  }

  index_axis
  take (const std::vector<std::size_t> &indices) const
  {
    for (std::size_t idx : indices)
      {
        if (idx >= size_)
          {
            throw index_error ("index_axis::take index out of bounds");
          }
      }
    if (is_range_)
      {
        return index_axis (indices.size (), name_);
      }

    ensure_labels ();
    std::vector<std::string> sub;
    sub.reserve (indices.size ());
    for (std::size_t idx : indices)
      {
        sub.push_back (labels_[idx]);
      }
    return index_axis (std::move (sub), name_);
  }

  bool
  operator== (const index_axis &other) const
  {
    if (size_ != other.size_)
      {
        return false;
      }
    if (is_range_ && other.is_range_)
      {
        return true;
      }
    ensure_labels ();
    other.ensure_labels ();
    return labels_ == other.labels_;
  }

  bool
  operator!= (const index_axis &other) const
  {
    return !(*this == other);
  }
};

} // namespace hyper_math::tabular

#endif /* HYPER_MATH_TABULAR_INDEX_HPP */
