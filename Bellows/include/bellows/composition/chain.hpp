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

#ifndef BELLOWS_COMPOSITION_CHAIN_HPP
#define BELLOWS_COMPOSITION_CHAIN_HPP

#include <cstddef>
#include <iostream>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/operations.hpp>

namespace bellows
{

// Zero-overhead compile-time composition of a transformer and a downstream stage.
template <typename Upstream_Stage, typename Downstream_Stage>
  requires transformer<Upstream_Stage, data_table>
            && estimator<Downstream_Stage>
class composite_chain
{
public:
  static constexpr bool supports_nominal_columns =
    categorical_aware<Upstream_Stage>;
  static constexpr bool supports_missing_entries =
    missing_aware<Upstream_Stage>;

  composite_chain () = default;

  composite_chain (Upstream_Stage upstream, Downstream_Stage downstream)
    : upstream_ (std::move (upstream)),
      downstream_ (std::move (downstream))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return upstream_.is_calibrated () && downstream_.is_calibrated ();
  }

  template <tabular_source Data>
    requires unsupervised_fittable<Upstream_Stage, Data>
              && unsupervised_fittable<Downstream_Stage, data_table>
  void
  calibrate (const Data &data)
  {
    upstream_.calibrate (data);
    data_table mid = upstream_.project (data);
    downstream_.calibrate (mid);
  }

  template <tabular_source Data, target_sequence Targets>
    requires supervised_fittable<Upstream_Stage, Data, Targets>
              && supervised_fittable<Downstream_Stage, data_table, Targets>
  void
  calibrate (const Data &data, const Targets &targets)
  {
    upstream_.calibrate (data, targets);
    data_table mid = [&] ()
    {
      if constexpr (requires { upstream_.project_ordered (data, targets); })
        {
          return upstream_.project_ordered (data, targets);
        }
      else
        {
          return upstream_.project (data);
        }
    }();
    downstream_.calibrate (mid, targets);
  }

  template <tabular_source Data>
    requires transformer<Downstream_Stage, data_table>
  data_table
  project (const Data &data) const
  {
    data_table mid = upstream_.project (data);
    return downstream_.project (mid);
  }

  template <tabular_source Data>
    requires predictor<Downstream_Stage, data_table>
  std::vector<double>
  evaluate (const Data &data) const
  {
    data_table mid = upstream_.project (data);
    return downstream_.evaluate (mid);
  }

  double
  evaluate_one (std::span<const double> row) const
    requires predictor<Downstream_Stage, data_table>
  {
    data_table single (1, row.size ());
    for (std::size_t c = 0; c < row.size (); ++c)
      {
        single.set_cell (0, c, row[c]);
      }
    data_table mid = upstream_.project (single);
    return downstream_.evaluate_one (mid.row_span (0));
  }

  template <tabular_source Data>
    requires belief_predictor<Downstream_Stage, data_table>
  data_table
  evaluate_beliefs (const Data &data) const
  {
    data_table mid = upstream_.project (data);
    return downstream_.evaluate_beliefs (mid);
  }

  const Upstream_Stage &
  upstream () const noexcept
  {
    return upstream_;
  }

  const Downstream_Stage &
  downstream () const noexcept
  {
    return downstream_;
  }

  std::vector<parameter_knob>
  parameter_space () const
    requires tunable<Downstream_Stage>
  {
    std::vector<parameter_knob> space;
    if constexpr (tunable<Upstream_Stage>)
      {
        auto up_space = upstream_.parameter_space ();
        space.insert (space.end (), up_space.begin (), up_space.end ());
      }
    auto down_space = downstream_.parameter_space ();
    space.insert (space.end (), down_space.begin (), down_space.end ());
    return space;
  }

  bool
  apply_parameter (const std::string &name, double val)
    requires tunable<Downstream_Stage>
  {
    bool applied = false;
    if constexpr (tunable<Upstream_Stage>)
      {
        applied = upstream_.apply_parameter (name, val) || applied;
      }
    applied = downstream_.apply_parameter (name, val) || applied;
    return applied;
  }

  void
  write_state (std::ostream &out) const
    requires serialisable<Upstream_Stage> && serialisable<Downstream_Stage>
  {
    upstream_.write_state (out);
    downstream_.write_state (out);
  }

  void
  read_state (std::istream &in)
    requires serialisable<Upstream_Stage> && serialisable<Downstream_Stage>
  {
    upstream_.read_state (in);
    downstream_.read_state (in);
  }

private:
  Upstream_Stage upstream_;
  Downstream_Stage downstream_;
};

// Pipe operator composing a transformer stage with a downstream estimator.
template <typename Upstream_Stage, typename Downstream_Stage>
  requires transformer<std::remove_cvref_t<Upstream_Stage>, data_table>
            && estimator<std::remove_cvref_t<Downstream_Stage>>
auto
operator| (Upstream_Stage &&up, Downstream_Stage &&down)
{
  return composite_chain<std::remove_cvref_t<Upstream_Stage>,
                          std::remove_cvref_t<Downstream_Stage>> (
    std::forward<Upstream_Stage> (up),
    std::forward<Downstream_Stage> (down));
}

// Type-erased runtime chain for storing heterogeneous pipeline steps.
class dynamic_chain
{
public:
  dynamic_chain () = default;

  dynamic_chain (const dynamic_chain &other)
    : calibrated_ (other.calibrated_)
  {
    steps_.reserve (other.steps_.size ());
    for (const auto &s : other.steps_)
      {
        steps_.push_back (s->clone ());
      }
    if (other.terminal_)
      {
        terminal_ = other.terminal_->clone ();
      }
  }

  dynamic_chain &
  operator= (const dynamic_chain &other)
  {
    if (this != &other)
      {
        calibrated_ = other.calibrated_;
        steps_.clear ();
        for (const auto &s : other.steps_)
          {
            steps_.push_back (s->clone ());
          }
        terminal_ = other.terminal_ ? other.terminal_->clone () : nullptr;
      }
    return *this;
  }

  template <typename Step>
    requires transformer<Step, data_table>
              && supervised_fittable<Step, data_table, std::vector<double>>
  void
  append_transformer (Step step)
  {
    steps_.push_back (
      std::make_unique<transformer_holder<Step>> (std::move (step)));
  }

  template <typename Terminal>
    requires predictor<Terminal, data_table>
              && supervised_fittable<Terminal, data_table, std::vector<double>>
  void
  set_predictor (Terminal term)
  {
    terminal_ = std::make_unique<predictor_holder<Terminal>> (std::move (term));
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &targets)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    data_table current (n, d);
    for (std::size_t c = 0; c < d; ++c)
      {
        current.set_scale (c, data.scale (c), data.cardinality (c));
        for (std::size_t r = 0; r < n; ++r)
          {
            current.set_cell (r, c, data.at (r, c));
          }
      }
    std::vector<double> y_vec (n);
    for (std::size_t r = 0; r < n; ++r)
      {
        y_vec[r] = targets[r];
      }

    for (auto &step : steps_)
      {
        current = step->fit_and_project_step (current, y_vec);
      }
    if (terminal_)
      {
        terminal_->fit_terminal (current, y_vec);
      }
    calibrated_ = true;
  }

  template <tabular_source Data>
  data_table
  project (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    data_table current (n, d);
    for (std::size_t c = 0; c < d; ++c)
      {
        current.set_scale (c, data.scale (c), data.cardinality (c));
        for (std::size_t r = 0; r < n; ++r)
          {
            current.set_cell (r, c, data.at (r, c));
          }
      }
    for (const auto &step : steps_)
      {
        current = step->project_step (current);
      }
    return current;
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    data_table transformed = project (data);
    if (!terminal_)
      {
        return std::vector<double> (transformed.row_count (), 0.0);
      }
    return terminal_->evaluate_terminal (transformed);
  }

  double
  evaluate_one (std::span<const double> row) const
  {
    data_table single (1, row.size ());
    for (std::size_t c = 0; c < row.size (); ++c)
      {
        single.set_cell (0, c, row[c]);
      }
    auto preds = evaluate (single);
    return preds.empty () ? 0.0 : preds[0];
  }

private:
  struct transformer_concept
  {
    virtual ~transformer_concept () = default;
    virtual std::unique_ptr<transformer_concept>
    clone () const = 0;
    virtual data_table
    fit_and_project_step (const data_table &data,
                          const std::vector<double> &targets) = 0;
    virtual data_table
    project_step (const data_table &data) const = 0;
  };

  template <typename Step>
  struct transformer_holder final : transformer_concept
  {
    explicit transformer_holder (Step s)
      : step_ (std::move (s))
    {
    }

    std::unique_ptr<transformer_concept>
    clone () const override
    {
      return std::make_unique<transformer_holder<Step>> (step_);
    }

    data_table
    fit_and_project_step (const data_table &data,
                          const std::vector<double> &targets) override
    {
      step_.calibrate (data, targets);
      if constexpr (requires { step_.project_ordered (data, targets); })
        {
          return step_.project_ordered (data, targets);
        }
      else
        {
          return step_.project (data);
        }
    }

    data_table
    project_step (const data_table &data) const override
    {
      return step_.project (data);
    }

    Step step_;
  };

  struct predictor_concept
  {
    virtual ~predictor_concept () = default;
    virtual std::unique_ptr<predictor_concept>
    clone () const = 0;
    virtual void
    fit_terminal (const data_table &data,
                  const std::vector<double> &targets) = 0;
    virtual std::vector<double>
    evaluate_terminal (const data_table &data) const = 0;
  };

  template <typename Terminal>
  struct predictor_holder final : predictor_concept
  {
    explicit predictor_holder (Terminal t)
      : term_ (std::move (t))
    {
    }

    std::unique_ptr<predictor_concept>
    clone () const override
    {
      return std::make_unique<predictor_holder<Terminal>> (term_);
    }

    void
    fit_terminal (const data_table &data,
                  const std::vector<double> &targets) override
    {
      term_.calibrate (data, targets);
    }

    std::vector<double>
    evaluate_terminal (const data_table &data) const override
    {
      return term_.evaluate (data);
    }

    Terminal term_;
  };

  bool calibrated_ = false;
  std::vector<std::unique_ptr<transformer_concept>> steps_;
  std::unique_ptr<predictor_concept> terminal_;
};

} // namespace bellows

#endif // BELLOWS_COMPOSITION_CHAIN_HPP
