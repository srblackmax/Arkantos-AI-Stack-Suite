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

#ifndef BELLOWS_COMPOSITION_ENSEMBLE_HPP
#define BELLOWS_COMPOSITION_ENSEMBLE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <bellows/composition/validation.hpp>
#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

enum class aggregation_mode : std::uint8_t
{
  weighted_mean = 0,
  majority_vote = 1
};

namespace detail
{

struct erased_member_predictor
{
  virtual ~erased_member_predictor () = default;
  virtual std::unique_ptr<erased_member_predictor>
  clone () const = 0;
  virtual bool
  is_calibrated () const = 0;
  virtual void
  calibrate (const data_table &data,
              const std::vector<double> &targets) = 0;
  virtual std::vector<double>
  evaluate (const data_table &data) const = 0;
  virtual double
  evaluate_one (std::span<const double> row) const = 0;
  virtual void
  write_state (std::ostream &out) const = 0;
  virtual void
  read_state (std::istream &in) = 0;
};

template <typename Model>
class concrete_member_predictor final : public erased_member_predictor
{
public:
  explicit concrete_member_predictor (Model m)
    : model_ (std::move (m))
  {
  }

  std::unique_ptr<erased_member_predictor>
  clone () const override
  {
    return std::make_unique<concrete_member_predictor<Model>> (model_);
  }

  bool
  is_calibrated () const override
  {
    return model_.is_calibrated ();
  }

  void
  calibrate (const data_table &data,
              const std::vector<double> &targets) override
  {
    model_.calibrate (data, targets);
  }

  std::vector<double>
  evaluate (const data_table &data) const override
  {
    return model_.evaluate (data);
  }

  double
  evaluate_one (std::span<const double> row) const override
  {
    return model_.evaluate_one (row);
  }

  void
  write_state (std::ostream &out) const override
  {
    if constexpr (serialisable<Model>)
      {
        model_.write_state (out);
      }
  }

  void
  read_state (std::istream &in) override
  {
    if constexpr (serialisable<Model>)
      {
        model_.read_state (in);
      }
  }

private:
  Model model_;
};

} // namespace detail

// Heterogeneous weighted consensus ensemble supporting voting and averaging.
class consensus_ensemble
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit consensus_ensemble (
    aggregation_mode mode = aggregation_mode::weighted_mean) noexcept
    : mode_ (mode)
  {
  }

  consensus_ensemble (const consensus_ensemble &other)
    : mode_ (other.mode_),
      weights_ (other.weights_),
      calibrated_ (other.calibrated_)
  {
    members_.reserve (other.members_.size ());
    for (const auto &m : other.members_)
      {
        members_.push_back (m->clone ());
      }
  }

  consensus_ensemble &
  operator= (const consensus_ensemble &other)
  {
    if (this != &other)
      {
        mode_ = other.mode_;
        weights_ = other.weights_;
        calibrated_ = other.calibrated_;
        members_.clear ();
        members_.reserve (other.members_.size ());
        for (const auto &m : other.members_)
          {
            members_.push_back (m->clone ());
          }
      }
    return *this;
  }

  consensus_ensemble (consensus_ensemble &&) noexcept = default;
  consensus_ensemble &
  operator= (consensus_ensemble &&) noexcept = default;

  template <typename Model>
    requires supervised_fittable<Model, data_table, std::vector<double>>
              && predictor<Model, data_table>
  void
  add_member (Model model, double weight = 1.0)
  {
    members_.push_back (
      std::make_unique<detail::concrete_member_predictor<Model>> (
        std::move (model)));
    weights_.push_back (std::max (1.0e-9, weight));
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_ && !members_.empty ();
  }

  std::size_t
  member_count () const noexcept
  {
    return members_.size ();
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &targets)
  {
    if (members_.empty ())
      {
        calibrated_ = false;
        return;
      }
    data_table dt = to_data_table (data);
    std::vector<double> y_vec (targets.size ());
    for (std::size_t i = 0; i < targets.size (); ++i)
      {
        y_vec[i] = static_cast<double> (targets[i]);
      }

    for (auto &m : members_)
      {
        m->calibrate (dt, y_vec);
      }
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> row) const
  {
    if (!is_calibrated ())
      {
        return 0.0;
      }
    if (mode_ == aggregation_mode::weighted_mean)
      {
        double w_sum = 0.0;
        double acc = 0.0;
        for (std::size_t i = 0; i < members_.size (); ++i)
          {
            acc += weights_[i] * members_[i]->evaluate_one (row);
            w_sum += weights_[i];
          }
        return w_sum > 0.0 ? acc / w_sum : 0.0;
      }

    std::map<long, double> ballot;
    for (std::size_t i = 0; i < members_.size (); ++i)
      {
        long cls = std::lround (members_[i]->evaluate_one (row));
        ballot[cls] += weights_[i];
      }
    long best_cls = 0;
    double best_w = -1.0;
    for (const auto &kv : ballot)
      {
        if (kv.second > best_w)
          {
            best_w = kv.second;
            best_cls = kv.first;
          }
      }
    return static_cast<double> (best_cls);
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> out (n, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        out[r] = evaluate_one (buf);
      }
    return out;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    std::uint8_t md = static_cast<std::uint8_t> (mode_);
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&md), sizeof (md));
    std::size_t m = members_.size ();
    out.write (reinterpret_cast<const char *> (&m), sizeof (m));
    for (double w : weights_)
      {
        out.write (reinterpret_cast<const char *> (&w), sizeof (double));
      }
    for (const auto &mem : members_)
      {
        mem->write_state (out);
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    std::uint8_t md = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    in.read (reinterpret_cast<char *> (&md), sizeof (md));
    calibrated_ = (cal != 0);
    mode_ = static_cast<aggregation_mode> (md);
    std::size_t m = 0;
    in.read (reinterpret_cast<char *> (&m), sizeof (m));
    weights_.resize (m, 1.0);
    for (std::size_t i = 0; i < m; ++i)
      {
        in.read (reinterpret_cast<char *> (&weights_[i]), sizeof (double));
      }
    for (std::size_t i = 0; i < std::min (m, members_.size ()); ++i)
      {
        members_[i]->read_state (in);
      }
  }

private:
  template <tabular_source Data>
  static data_table
  to_data_table (const Data &data)
  {
    if constexpr (std::is_same_v<std::remove_cvref_t<Data>, data_table>)
      {
        return data;
      }
    else
      {
        std::size_t n = data.row_count ();
        std::size_t d = data.column_count ();
        data_table dt (n, d);
        for (std::size_t c = 0; c < d; ++c)
          {
            dt.set_scale (c, data.scale (c), data.cardinality (c));
          }
        for (std::size_t r = 0; r < n; ++r)
          {
            for (std::size_t c = 0; c < d; ++c)
              {
                dt.set_cell (r, c, data.at (r, c));
              }
          }
        return dt;
      }
  }

  aggregation_mode mode_ = aggregation_mode::weighted_mean;
  std::vector<std::unique_ptr<detail::erased_member_predictor>> members_;
  std::vector<double> weights_;
  bool calibrated_ = false;
};

// Out-of-fold stacked generalisation meta-estimator.
template <typename Meta_Learner>
  requires supervised_fittable<Meta_Learner, data_table, std::vector<double>>
            && predictor<Meta_Learner, data_table>
class stacked_generaliser
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit stacked_generaliser (Meta_Learner meta = Meta_Learner{},
                                std::size_t fold_count = 4,
                                bool passthrough_features = false,
                                std::uint64_t seed = 42)
    : meta_learner_ (std::move (meta)),
      fold_count_ (std::max<std::size_t> (2, fold_count)),
      passthrough_ (passthrough_features),
      seed_ (seed)
  {
  }

  stacked_generaliser (const stacked_generaliser &other)
    : meta_learner_ (other.meta_learner_),
      fold_count_ (other.fold_count_),
      passthrough_ (other.passthrough_),
      seed_ (other.seed_),
      calibrated_ (other.calibrated_)
  {
    base_learners_.reserve (other.base_learners_.size ());
    for (const auto &b : other.base_learners_)
      {
        base_learners_.push_back (b->clone ());
      }
  }

  stacked_generaliser &
  operator= (const stacked_generaliser &other)
  {
    if (this != &other)
      {
        meta_learner_ = other.meta_learner_;
        fold_count_ = other.fold_count_;
        passthrough_ = other.passthrough_;
        seed_ = other.seed_;
        calibrated_ = other.calibrated_;
        base_learners_.clear ();
        base_learners_.reserve (other.base_learners_.size ());
        for (const auto &b : other.base_learners_)
          {
            base_learners_.push_back (b->clone ());
          }
      }
    return *this;
  }

  stacked_generaliser (stacked_generaliser &&) noexcept = default;
  stacked_generaliser &
  operator= (stacked_generaliser &&) noexcept = default;

  template <typename Model>
    requires supervised_fittable<Model, data_table, std::vector<double>>
              && predictor<Model, data_table>
  void
  add_base_learner (Model model)
  {
    base_learners_.push_back (
      std::make_unique<detail::concrete_member_predictor<Model>> (
        std::move (model)));
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_ && !base_learners_.empty ()
            && meta_learner_.is_calibrated ();
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &targets)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::size_t m = base_learners_.size ();
    if (n == 0 || d == 0 || m == 0 || targets.size () < n)
      {
        calibrated_ = false;
        return;
      }

    data_table full_x (n, d);
    for (std::size_t c = 0; c < d; ++c)
      {
        full_x.set_scale (c, data.scale (c), data.cardinality (c));
      }
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            full_x.set_cell (r, c, data.at (r, c));
          }
      }

    std::vector<double> full_y (n);
    for (std::size_t r = 0; r < n; ++r)
      {
        full_y[r] = static_cast<double> (targets[r]);
      }

    std::size_t meta_cols = m + (passthrough_ ? d : 0);
    data_table meta_x (n, meta_cols, 0.0);

    fold_splitter splitter (fold_count_, true, seed_);
    auto partitions = splitter.split (full_x, full_y);

    if (partitions.empty ())
      {
        for (std::size_t b = 0; b < m; ++b)
          {
            base_learners_[b]->calibrate (full_x, full_y);
            auto preds = base_learners_[b]->evaluate (full_x);
            for (std::size_t r = 0; r < n; ++r)
              {
                meta_x.at (r, b) = preds[r];
              }
          }
      }
    else
      {
        for (const auto &part : partitions)
          {
            data_table x_tr = full_x.select_rows (part.train_rows);
            data_table x_val = full_x.select_rows (part.validation_rows);
            std::vector<double> y_tr;
            y_tr.reserve (part.train_rows.size ());
            for (std::size_t r : part.train_rows)
              {
                y_tr.push_back (full_y[r]);
              }

            for (std::size_t b = 0; b < m; ++b)
              {
                auto fold_clone = base_learners_[b]->clone ();
                fold_clone->calibrate (x_tr, y_tr);
                auto val_preds = fold_clone->evaluate (x_val);
                for (std::size_t idx = 0;
                      idx < part.validation_rows.size ();
                      ++idx)
                  {
                    meta_x.at (part.validation_rows[idx], b) = val_preds[idx];
                  }
              }
          }

        for (std::size_t b = 0; b < m; ++b)
          {
            base_learners_[b]->calibrate (full_x, full_y);
          }
      }

    if (passthrough_)
      {
        for (std::size_t r = 0; r < n; ++r)
          {
            for (std::size_t c = 0; c < d; ++c)
              {
                meta_x.set_cell (r, m + c, full_x.at (r, c));
              }
          }
      }

    meta_learner_.calibrate (meta_x, full_y);
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> row) const
  {
    if (!is_calibrated ())
      {
        return 0.0;
      }
    std::size_t m = base_learners_.size ();
    std::size_t meta_cols = m + (passthrough_ ? row.size () : 0);
    std::vector<double> meta_row (meta_cols, 0.0);
    for (std::size_t b = 0; b < m; ++b)
      {
        meta_row[b] = base_learners_[b]->evaluate_one (row);
      }
    if (passthrough_)
      {
        for (std::size_t c = 0; c < row.size (); ++c)
          {
            meta_row[m + c] = row[c];
          }
      }
    return meta_learner_.evaluate_one (meta_row);
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> out (n, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        out[r] = evaluate_one (buf);
      }
    return out;
  }

  void
  write_state (std::ostream &out) const
    requires serialisable<Meta_Learner>
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    std::uint8_t pass = passthrough_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&pass), sizeof (pass));
    for (const auto &b : base_learners_)
      {
        b->write_state (out);
      }
    meta_learner_.write_state (out);
  }

  void
  read_state (std::istream &in)
    requires serialisable<Meta_Learner>
  {
    std::uint8_t cal = 0;
    std::uint8_t pass = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    in.read (reinterpret_cast<char *> (&pass), sizeof (pass));
    calibrated_ = (cal != 0);
    passthrough_ = (pass != 0);
    for (auto &b : base_learners_)
      {
        b->read_state (in);
      }
    meta_learner_.read_state (in);
  }

private:
  std::vector<std::unique_ptr<detail::erased_member_predictor>> base_learners_;
  Meta_Learner meta_learner_;
  std::size_t fold_count_ = 4;
  bool passthrough_ = false;
  std::uint64_t seed_ = 42;
  bool calibrated_ = false;
};

} // namespace bellows

#endif // BELLOWS_COMPOSITION_ENSEMBLE_HPP
