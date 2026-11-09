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

#ifndef HYPER_MATH_TABULAR_LAZY_HPP
#define HYPER_MATH_TABULAR_LAZY_HPP

#include "hyper_math/tabular/data_frame.hpp"

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace hyper_math::tabular
{

enum class expr_kind
{
  column_ref,
  literal_scalar,
  literal_string,
  binary_arith,
  binary_compare,
  binary_logical,
  ternary_when,
  aggregation,
  window_over,
  fill_null_op,
  unary_transform,
  aliased
};

enum class binary_op_code
{
  add,
  sub,
  mul,
  div,
  eq,
  neq,
  lt,
  le,
  gt,
  ge,
  logic_and,
  logic_or
};

/* Declarative Expression AST node (Polars pl.Expr equivalent).  */
class expr
{
public:
  struct node
  {
    expr_kind kind;
    std::string text;
    double num_val = 0.0;
    double num_val2 = 0.0;
    binary_op_code bin_op = binary_op_code::add;
    agg_op agg = agg_op::sum;
    std::shared_ptr<node> left;
    std::shared_ptr<node> right;
    std::shared_ptr<node> third;
  };

private:
  std::shared_ptr<node> root_;

public:
  expr () = default;

  explicit
  expr (std::shared_ptr<node> n)
    : root_ (std::move (n))
  {
  }

  const std::shared_ptr<node> &
  root () const noexcept
  {
    return root_;
  }

  static expr
  col (std::string name)
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::column_ref;
    n->text = std::move (name);
    return expr (std::move (n));
  }

  static expr
  lit (double val)
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::literal_scalar;
    n->num_val = val;
    return expr (std::move (n));
  }

  static expr
  lit (std::string val)
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::literal_string;
    n->text = std::move (val);
    return expr (std::move (n));
  }

  expr
  alias (std::string alias_name) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::aliased;
    n->text = std::move (alias_name);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  fill_null (double val) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::fill_null_op;
    n->num_val = val;
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  make_unary_transform (std::string op_tag) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = std::move (op_tag);
    n->left = root_;
    return expr (std::move (n));
  }

  expr abs () const { return make_unary_transform ("abs"); }
  expr round () const { return make_unary_transform ("round"); }
  expr sqrt () const { return make_unary_transform ("sqrt"); }
  expr log () const { return make_unary_transform ("log"); }
  expr exp () const { return make_unary_transform ("exp"); }
  expr sin () const { return make_unary_transform ("sin"); }
  expr cos () const { return make_unary_transform ("cos"); }
  expr tan () const { return make_unary_transform ("tan"); }
  expr rank () const { return make_unary_transform ("rank"); }
  expr cum_sum () const { return make_unary_transform ("cum_sum"); }
  expr cum_prod () const { return make_unary_transform ("cum_prod"); }
  expr cum_min () const { return make_unary_transform ("cum_min"); }
  expr cum_max () const { return make_unary_transform ("cum_max"); }
  expr diff () const { return make_unary_transform ("diff"); }
  expr pct_change () const { return make_unary_transform ("pct_change"); }
  expr fill_forward () const { return make_unary_transform ("fill_forward"); }
  expr fill_backward () const { return make_unary_transform ("fill_backward"); }
  expr is_null () const { return make_unary_transform ("is_null"); }
  expr is_not_null () const { return make_unary_transform ("is_not_null"); }
  expr dt_year () const { return make_unary_transform ("dt_year"); }
  expr dt_month () const { return make_unary_transform ("dt_month"); }
  expr dt_day () const { return make_unary_transform ("dt_day"); }
  expr str_to_uppercase () const
  {
    return make_unary_transform ("str_to_uppercase");
  }
  expr str_to_lowercase () const
  {
    return make_unary_transform ("str_to_lowercase");
  }
  expr str_lengths () const { return make_unary_transform ("str_lengths"); }
  expr str_strip () const { return make_unary_transform ("str_strip"); }
  expr floor () const { return make_unary_transform ("floor"); }
  expr ceil () const { return make_unary_transform ("ceil"); }
  expr sign () const { return make_unary_transform ("sign"); }
  expr is_nan () const { return make_unary_transform ("is_nan"); }
  expr is_finite () const { return make_unary_transform ("is_finite"); }
  expr is_duplicated () const { return make_unary_transform ("is_duplicated"); }
  expr is_unique () const { return make_unary_transform ("is_unique"); }
  expr dt_quarter () const { return make_unary_transform ("dt_quarter"); }
  expr dt_week () const { return make_unary_transform ("dt_week"); }
  expr dt_day_of_week () const
  {
    return make_unary_transform ("dt_day_of_week");
  }
  expr dt_day_of_year () const
  {
    return make_unary_transform ("dt_day_of_year");
  }
  expr dt_hour () const { return make_unary_transform ("dt_hour"); }
  expr dt_minute () const { return make_unary_transform ("dt_minute"); }
  expr dt_second () const { return make_unary_transform ("dt_second"); }

  expr
  shift (std::ptrdiff_t periods = 1) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "shift";
    n->num_val = static_cast<double> (periods);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  pow (double exp_val) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "pow";
    n->num_val = exp_val;
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  mod (double divisor) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "mod";
    n->num_val = divisor;
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  rolling_min (std::size_t window) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "rolling_min";
    n->num_val = static_cast<double> (window);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  rolling_max (std::size_t window) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "rolling_max";
    n->num_val = static_cast<double> (window);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  rolling_std (std::size_t window) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "rolling_std";
    n->num_val = static_cast<double> (window);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  rolling_var (std::size_t window) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "rolling_var";
    n->num_val = static_cast<double> (window);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  ewm_mean (double alpha) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "ewm_mean";
    n->num_val = alpha;
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_contains (std::string pat) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_contains";
    n->right = lit (std::move (pat)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_starts_with (std::string pat) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_starts_with";
    n->right = lit (std::move (pat)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_ends_with (std::string pat) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_ends_with";
    n->right = lit (std::move (pat)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_slice (std::size_t start, std::size_t len) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_slice";
    n->num_val = static_cast<double> (start);
    n->num_val2 = static_cast<double> (len);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_replace (std::string pat, std::string repl) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_replace";
    n->right = lit (std::move (pat)).root ();
    n->third = lit (std::move (repl)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  is_in (const std::vector<double> &vals) const
  {
    auto make_eq = [this] (double v) -> expr {
      auto n = std::make_shared<node> ();
      n->kind = expr_kind::binary_compare;
      n->bin_op = binary_op_code::eq;
      n->left = root_;
      n->right = lit (v).root ();
      return expr (std::move (n));
    };
    if (vals.empty ())
      {
        return make_eq (constants::nan);
      }
    expr acc = make_eq (vals[0]);
    for (std::size_t i = 1; i < vals.size (); ++i)
      {
        auto n = std::make_shared<node> ();
        n->kind = expr_kind::binary_logical;
        n->bin_op = binary_op_code::logic_or;
        n->left = acc.root ();
        n->right = make_eq (vals[i]).root ();
        acc = expr (std::move (n));
      }
    return acc;
  }

  expr
  is_in (const std::vector<std::string> &vals) const
  {
    auto make_eq = [this] (const std::string &v) -> expr {
      auto n = std::make_shared<node> ();
      n->kind = expr_kind::binary_compare;
      n->bin_op = binary_op_code::eq;
      n->left = root_;
      n->right = lit (v).root ();
      return expr (std::move (n));
    };
    if (vals.empty ())
      {
        return make_eq ("__never_match__");
      }
    expr acc = make_eq (vals[0]);
    for (std::size_t i = 1; i < vals.size (); ++i)
      {
        auto n = std::make_shared<node> ();
        n->kind = expr_kind::binary_logical;
        n->bin_op = binary_op_code::logic_or;
        n->left = acc.root ();
        n->right = make_eq (vals[i]).root ();
        acc = expr (std::move (n));
      }
    return acc;
  }

  expr
  dt_truncate (std::string unit) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "dt_truncate";
    n->right = lit (std::move (unit)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  clip (double low, double high) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "clip";
    n->num_val = low;
    n->num_val2 = high;
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  is_between (double low, double high) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "is_between";
    n->num_val = low;
    n->num_val2 = high;
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  rolling_mean (std::size_t window) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "rolling_mean";
    n->num_val = static_cast<double> (window);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  rolling_sum (std::size_t window) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "rolling_sum";
    n->num_val = static_cast<double> (window);
    n->left = root_;
    return expr (std::move (n));
  }

  expr rle_id () const { return make_unary_transform ("rle_id"); }
  expr hash () const { return make_unary_transform ("hash"); }
  expr pct_rank () const { return make_unary_transform ("pct_rank"); }
  expr interpolate () const { return make_unary_transform ("interpolate"); }
  expr cum_count () const { return make_unary_transform ("cum_count"); }
  expr dt_epoch_seconds () const
  {
    return make_unary_transform ("dt_epoch_seconds");
  }
  expr log10 () const { return make_unary_transform ("log10"); }
  expr log2 () const { return make_unary_transform ("log2"); }
  expr exp2 () const { return make_unary_transform ("exp2"); }
  expr cbrt () const { return make_unary_transform ("cbrt"); }
  expr sinh () const { return make_unary_transform ("sinh"); }
  expr cosh () const { return make_unary_transform ("cosh"); }
  expr tanh () const { return make_unary_transform ("tanh"); }
  expr asin () const { return make_unary_transform ("asin"); }
  expr acos () const { return make_unary_transform ("acos"); }
  expr atan () const { return make_unary_transform ("atan"); }
  expr degrees () const { return make_unary_transform ("degrees"); }
  expr radians () const { return make_unary_transform ("radians"); }

  expr
  top_k (std::size_t k) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "top_k";
    n->num_val = static_cast<double> (k);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  bottom_k (std::size_t k) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "bottom_k";
    n->num_val = static_cast<double> (k);
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_extract (std::string pattern, std::size_t group_idx = 1) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_extract";
    n->num_val = static_cast<double> (group_idx);
    n->right = lit (std::move (pattern)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  str_json_extract (std::string key) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "str_json_extract";
    n->right = lit (std::move (key)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  expr
  dt_strftime (std::string fmt) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::unary_transform;
    n->text = "dt_strftime";
    n->right = lit (std::move (fmt)).root ();
    n->left = root_;
    return expr (std::move (n));
  }

  /* Aggregation expressions for lazy group_by.agg(...).  */
  expr
  make_agg (agg_op op) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::aggregation;
    n->agg = op;
    n->left = root_;
    return expr (std::move (n));
  }

  expr sum () const { return make_agg (agg_op::sum); }
  expr mean () const { return make_agg (agg_op::mean); }
  expr min () const { return make_agg (agg_op::min); }
  expr max () const { return make_agg (agg_op::max); }
  expr count () const { return make_agg (agg_op::count); }
  expr std_dev () const { return make_agg (agg_op::std_dev); }
  expr var () const { return make_agg (agg_op::var); }
  expr median () const { return make_agg (agg_op::median); }
  expr first () const { return make_agg (agg_op::first); }
  expr last () const { return make_agg (agg_op::last); }
  expr n_unique () const { return make_agg (agg_op::n_unique); }
  expr arg_min () const { return make_agg (agg_op::arg_min); }
  expr arg_max () const { return make_agg (agg_op::arg_max); }
  expr mode () const { return make_agg (agg_op::mode); }
  expr skew () const { return make_agg (agg_op::skew); }
  expr kurtosis () const { return make_agg (agg_op::kurtosis); }

  /* Polars-style window expression: expr.over("partition_col").
     Computes a group aggregation per partition and broadcasts it back to
     every row in that partition!  */
  expr
  over (std::string partition_col) const
  {
    auto n = std::make_shared<node> ();
    n->kind = expr_kind::window_over;
    n->text = std::move (partition_col);
    n->left = root_;
    return expr (std::move (n));
  }

  /* Determine output column name of this expression.  */
  std::string
  output_name () const
  {
    if (!root_)
      {
        return "expr";
      }
    if (root_->kind == expr_kind::aliased
        || root_->kind == expr_kind::column_ref)
      {
        return root_->text;
      }
    if (root_->kind == expr_kind::window_over && root_->left
        && root_->left->left)
      {
        return expr (root_->left->left).output_name () + "_over_" + root_->text;
      }
    if (root_->kind == expr_kind::aggregation && root_->left)
      {
        agg_spec sp (expr (root_->left).output_name (), root_->agg);
        return sp.alias_name;
      }
    if (root_->left)
      {
        const std::string l_name = expr (root_->left).output_name ();
        if (l_name != "literal" || !root_->right)
          {
            return l_name;
          }
        return expr (root_->right).output_name ();
      }
    return "literal";
  }

  /* Check if this expression can be evaluated independently per row without
     depending on row order or cross-row context.  */
  bool
  is_row_independent () const
  {
    if (!root_)
      {
        return true;
      }
    if (root_->kind == expr_kind::window_over
        || root_->kind == expr_kind::aggregation)
      {
        return false;
      }
    if (root_->kind == expr_kind::unary_transform)
      {
        const std::string &t = root_->text;
        if (t == "rank" || t == "cum_sum" || t == "cum_prod"
            || t == "cum_min" || t == "cum_max" || t == "diff"
            || t == "pct_change" || t == "fill_forward"
            || t == "fill_backward" || t == "rolling_mean"
            || t == "rolling_sum" || t == "rolling_min"
            || t == "rolling_max" || t == "rolling_std"
            || t == "rolling_var" || t == "ewm_mean"
            || t == "shift" || t == "is_duplicated" || t == "is_unique")
          {
            return false;
          }
      }
    if (root_->left && !expr (root_->left).is_row_independent ())
      {
        return false;
      }
    if (root_->right && !expr (root_->right).is_row_independent ())
      {
        return false;
      }
    if (root_->third && !expr (root_->third).is_row_independent ())
      {
        return false;
      }
    return true;
  }

  /* Collect all source column names referenced by this expression AST.  */
  void
  collect_referenced_columns (std::set<std::string> &out) const
  {
    if (!root_)
      {
        return;
      }
    if (root_->kind == expr_kind::column_ref)
      {
        out.insert (root_->text);
      }
    if (root_->kind == expr_kind::window_over)
      {
        out.insert (root_->text);
      }
    if (root_->left)
      {
        expr (root_->left).collect_referenced_columns (out);
      }
    if (root_->right)
      {
        expr (root_->right).collect_referenced_columns (out);
      }
    if (root_->third)
      {
        expr (root_->third).collect_referenced_columns (out);
      }
  }

  /* Constant folding optimization pass over the expression tree.  */
  expr
  fold_constants () const
  {
    if (!root_)
      {
        return *this;
      }
    auto copy_node = std::make_shared<node> (*root_);
    if (copy_node->left)
      {
        copy_node->left = expr (copy_node->left).fold_constants ().root ();
      }
    if (copy_node->right)
      {
        copy_node->right = expr (copy_node->right).fold_constants ().root ();
      }
    if (copy_node->third)
      {
        copy_node->third = expr (copy_node->third).fold_constants ().root ();
      }

    if (copy_node->kind == expr_kind::binary_arith
        && copy_node->left
        && copy_node->right
        && copy_node->left->kind == expr_kind::literal_scalar
        && copy_node->right->kind == expr_kind::literal_scalar)
      {
        const double a = copy_node->left->num_val;
        const double b = copy_node->right->num_val;
        double res = 0.0;
        switch (copy_node->bin_op)
          {
          case binary_op_code::add:
            res = a + b;
            break;
          case binary_op_code::sub:
            res = a - b;
            break;
          case binary_op_code::mul:
            res = a * b;
            break;
          case binary_op_code::div:
            res = a / b;
            break;
          default:
            break;
          }
        return lit (res);
      }
    return expr (std::move (copy_node));
  }

  /* Format expression AST as a human-readable string for explain plans.  */
  std::string
  to_string () const
  {
    if (!root_)
      {
        return "<null_expr>";
      }
    switch (root_->kind)
      {
      case expr_kind::column_ref:
        return "col(\"" + root_->text + "\")";
      case expr_kind::literal_scalar:
        {
          std::ostringstream oss;
          oss << root_->num_val;
          return oss.str ();
        }
      case expr_kind::literal_string:
        return "\"" + root_->text + "\"";
      case expr_kind::aliased:
        return expr (root_->left).to_string () + ".alias(\""
               + root_->text + "\")";
      case expr_kind::fill_null_op:
        return expr (root_->left).to_string () + ".fill_null("
               + std::to_string (root_->num_val) + ")";
      case expr_kind::unary_transform:
        return expr (root_->left).to_string () + "." + root_->text + "()";
      case expr_kind::aggregation:
        return expr (root_->left).to_string () + ".agg()";
      case expr_kind::window_over:
        return expr (root_->left).to_string () + ".over(\""
               + root_->text + "\")";
      case expr_kind::ternary_when:
        return "when(" + expr (root_->left).to_string () + ").then("
               + expr (root_->right).to_string () + ").otherwise("
               + expr (root_->third).to_string () + ")";
      case expr_kind::binary_arith:
      case expr_kind::binary_compare:
      case expr_kind::binary_logical:
        {
          std::string op_str = "?";
          switch (root_->bin_op)
            {
            case binary_op_code::add: op_str = "+"; break;
            case binary_op_code::sub: op_str = "-"; break;
            case binary_op_code::mul: op_str = "*"; break;
            case binary_op_code::div: op_str = "/"; break;
            case binary_op_code::eq:  op_str = "=="; break;
            case binary_op_code::neq: op_str = "!="; break;
            case binary_op_code::lt:  op_str = "<"; break;
            case binary_op_code::le:  op_str = "<="; break;
            case binary_op_code::gt:  op_str = ">"; break;
            case binary_op_code::ge:  op_str = ">="; break;
            case binary_op_code::logic_and: op_str = "&&"; break;
            case binary_op_code::logic_or:  op_str = "||"; break;
            }
          return "(" + expr (root_->left).to_string () + " " + op_str + " "
                 + expr (root_->right).to_string () + ")";
        }
      }
    return "expr";
  }

  /* Evaluate expression physically against a materialized data_frame.  */
  series
  evaluate (const data_frame &df) const
  {
    if (!root_)
      {
        throw schema_error ("cannot evaluate empty expression");
      }

    switch (root_->kind)
      {
      case expr_kind::column_ref:
        return df[root_->text];

      case expr_kind::literal_scalar:
        return series ("literal",
                       std::vector<double> (df.nrows (), root_->num_val),
                       df.index ());

      case expr_kind::literal_string:
        return series ("literal",
                       std::vector<std::string> (df.nrows (), root_->text),
                       df.index ());

      case expr_kind::aliased:
        return expr (root_->left).evaluate (df).alias (root_->text);

      case expr_kind::fill_null_op:
        return expr (root_->left).evaluate (df).fill_null (root_->num_val);

      case expr_kind::unary_transform:
        {
          const series s = expr (root_->left).evaluate (df);
          if (root_->text == "abs")
            {
              return s.abs ();
            }
          if (root_->text == "round")
            {
              return s.round ();
            }
          if (root_->text == "sqrt")
            {
              return s.sqrt ();
            }
          if (root_->text == "log")
            {
              return s.log ();
            }
          if (root_->text == "exp")
            {
              return s.exp ();
            }
          if (root_->text == "sin")
            {
              return s.sin ();
            }
          if (root_->text == "cos")
            {
              return s.cos ();
            }
          if (root_->text == "tan")
            {
              return s.tan ();
            }
          if (root_->text == "rank")
            {
              return s.rank ("average", true);
            }
          if (root_->text == "cum_sum")
            {
              return s.cum_sum ();
            }
          if (root_->text == "cum_prod")
            {
              return s.cum_prod ();
            }
          if (root_->text == "cum_min")
            {
              return s.cum_min ();
            }
          if (root_->text == "cum_max")
            {
              return s.cum_max ();
            }
          if (root_->text == "diff")
            {
              return s.diff (1);
            }
          if (root_->text == "pct_change")
            {
              return s.pct_change (1);
            }
          if (root_->text == "fill_forward")
            {
              return s.fill_forward ();
            }
          if (root_->text == "fill_backward")
            {
              return s.fill_backward ();
            }
          if (root_->text == "clip")
            {
              return s.clip (root_->num_val, root_->num_val2);
            }
          if (root_->text == "is_between")
            {
              nd_tensor<std::uint8_t> m (shape_type {s.size ()}, 0);
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  if (s.is_valid (i))
                    {
                      const double v = s.get_double (i);
                      m[i] =
                        (v >= root_->num_val && v <= root_->num_val2) ? 1 : 0;
                    }
                }
              return series (s.name (), m, s.validity ().clone (), df.index ());
            }
          if (root_->text == "rolling_mean")
            {
              return s.rolling_mean (
                static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "rolling_sum")
            {
              return s.rolling_sum (
                static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "is_null")
            {
              nd_tensor<std::uint8_t> m (shape_type {s.size ()}, 0);
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  m[i] = s.is_null (i) ? 1 : 0;
                }
              return series (s.name () + "_isnull", m,
                             validity_bitmap (s.size (), true), df.index ());
            }
          if (root_->text == "is_not_null")
            {
              nd_tensor<std::uint8_t> m (shape_type {s.size ()}, 0);
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  m[i] = s.is_valid (i) ? 1 : 0;
                }
              return series (s.name () + "_notnull", m,
                             validity_bitmap (s.size (), true), df.index ());
            }
          if (root_->text == "dt_year")
            {
              return s.dt_year ();
            }
          if (root_->text == "dt_month")
            {
              return s.dt_month ();
            }
          if (root_->text == "dt_day")
            {
              return s.dt_day ();
            }
          if (root_->text == "str_to_uppercase")
            {
              return s.str_to_uppercase ();
            }
          if (root_->text == "str_to_lowercase")
            {
              return s.str_to_lowercase ();
            }
          if (root_->text == "str_lengths")
            {
              return s.str_lengths ();
            }
          if (root_->text == "str_strip")
            {
              return s.str_strip ();
            }
          if (root_->text == "floor")
            {
              return s.map_numeric ([] (double v) { return std::floor (v); });
            }
          if (root_->text == "ceil")
            {
              return s.map_numeric ([] (double v) { return std::ceil (v); });
            }
          if (root_->text == "sign")
            {
              return s.map_numeric ([] (double v) {
                return (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0);
              });
            }
          if (root_->text == "pow")
            {
              const double p = root_->num_val;
              return s.map_numeric ([p] (double v) { return std::pow (v, p); });
            }
          if (root_->text == "mod")
            {
              const double d = root_->num_val;
              return s.map_numeric ([d] (double v) { return std::fmod (v, d); });
            }
          if (root_->text == "shift")
            {
              return s.shift (static_cast<std::ptrdiff_t> (root_->num_val));
            }
          if (root_->text == "rolling_min")
            {
              return s.rolling_min (static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "rolling_max")
            {
              return s.rolling_max (static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "rolling_std")
            {
              return s.rolling_std (static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "rolling_var")
            {
              return s.rolling_var (static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "ewm_mean")
            {
              return s.ewm_mean (root_->num_val);
            }
          if (root_->text == "is_nan")
            {
              nd_tensor<std::uint8_t> m (shape_type {s.size ()}, 0);
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  m[i] =
                    (!s.is_valid (i) || std::isnan (s.get_double (i))) ? 1 : 0;
                }
              return series (s.name (), m,
                             validity_bitmap (s.size (), true), df.index ());
            }
          if (root_->text == "is_finite")
            {
              nd_tensor<std::uint8_t> m (shape_type {s.size ()}, 0);
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  m[i] =
                    (s.is_valid (i) && std::isfinite (s.get_double (i))) ? 1 : 0;
                }
              return series (s.name (), m,
                             validity_bitmap (s.size (), true), df.index ());
            }
          if (root_->text == "is_duplicated" || root_->text == "is_unique")
            {
              std::unordered_map<std::string, std::size_t> counts;
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  if (s.is_valid (i))
                    {
                      ++counts[cell_hash_key (s.iloc (i))];
                    }
                }
              nd_tensor<std::uint8_t> m (shape_type {s.size ()}, 0);
              const bool want_dup = (root_->text == "is_duplicated");
              for (std::size_t i = 0; i < s.size (); ++i)
                {
                  if (s.is_valid (i))
                    {
                      const std::size_t c = counts[cell_hash_key (s.iloc (i))];
                      m[i] = (want_dup ? (c > 1) : (c == 1)) ? 1 : 0;
                    }
                }
              return series (s.name (), m, s.validity ().clone (), df.index ());
            }
          if (root_->text == "str_contains" && root_->right)
            {
              return s.str_contains (root_->right->text);
            }
          if (root_->text == "str_starts_with" && root_->right)
            {
              return s.str_starts_with (root_->right->text);
            }
          if (root_->text == "str_ends_with" && root_->right)
            {
              return s.str_ends_with (root_->right->text);
            }
          if (root_->text == "str_slice")
            {
              return s.str_slice (static_cast<std::size_t> (root_->num_val),
                                  static_cast<std::size_t> (root_->num_val2));
            }
          if (root_->text == "str_replace" && root_->right && root_->third)
            {
              return s.str_replace (root_->right->text, root_->third->text);
            }
          if (root_->text == "dt_quarter")
            {
              return s.dt_quarter ();
            }
          if (root_->text == "dt_week")
            {
              return s.dt_week ();
            }
          if (root_->text == "dt_day_of_week")
            {
              return s.dt_day_of_week ();
            }
          if (root_->text == "dt_day_of_year")
            {
              return s.dt_day_of_year ();
            }
          if (root_->text == "dt_hour")
            {
              return s.dt_hour ();
            }
          if (root_->text == "dt_minute")
            {
              return s.dt_minute ();
            }
          if (root_->text == "dt_second")
            {
              return s.dt_second ();
            }
          if (root_->text == "dt_truncate" && root_->right)
            {
              return s.dt_truncate (root_->right->text);
            }
          if (root_->text == "rle_id")
            {
              return s.rle_id ();
            }
          if (root_->text == "hash")
            {
              return s.hash ();
            }
          if (root_->text == "pct_rank")
            {
              return s.pct_rank ();
            }
          if (root_->text == "interpolate")
            {
              return s.interpolate ();
            }
          if (root_->text == "cum_count")
            {
              return s.cum_count ();
            }
          if (root_->text == "dt_epoch_seconds")
            {
              return s.dt_epoch_seconds ();
            }
          if (root_->text == "top_k")
            {
              return s.top_k (static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "bottom_k")
            {
              return s.bottom_k (static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "str_extract" && root_->right)
            {
              return s.str_extract (
                root_->right->text,
                static_cast<std::size_t> (root_->num_val));
            }
          if (root_->text == "str_json_extract" && root_->right)
            {
              return s.str_json_extract (root_->right->text);
            }
          if (root_->text == "dt_strftime" && root_->right)
            {
              return s.dt_strftime (root_->right->text);
            }
          if (root_->text == "log10")
            {
              return s.map_numeric ([] (double v) { return std::log10 (v); });
            }
          if (root_->text == "log2")
            {
              return s.map_numeric ([] (double v) { return std::log2 (v); });
            }
          if (root_->text == "exp2")
            {
              return s.map_numeric ([] (double v) { return std::exp2 (v); });
            }
          if (root_->text == "cbrt")
            {
              return s.map_numeric ([] (double v) { return std::cbrt (v); });
            }
          if (root_->text == "sinh")
            {
              return s.map_numeric ([] (double v) { return std::sinh (v); });
            }
          if (root_->text == "cosh")
            {
              return s.map_numeric ([] (double v) { return std::cosh (v); });
            }
          if (root_->text == "tanh")
            {
              return s.map_numeric ([] (double v) { return std::tanh (v); });
            }
          if (root_->text == "asin")
            {
              return s.map_numeric ([] (double v) { return std::asin (v); });
            }
          if (root_->text == "acos")
            {
              return s.map_numeric ([] (double v) { return std::acos (v); });
            }
          if (root_->text == "atan")
            {
              return s.map_numeric ([] (double v) { return std::atan (v); });
            }
          if (root_->text == "degrees")
            {
              return s.map_numeric ([] (double v) {
                return v * (180.0 / constants::pi);
              });
            }
          if (root_->text == "radians")
            {
              return s.map_numeric ([] (double v) {
                return v * (constants::pi / 180.0);
              });
            }
          return s;
        }

      case expr_kind::ternary_when:
        {
          const series cond_s = expr (root_->left).evaluate (df);
          const series then_s = expr (root_->right).evaluate (df);
          const series else_s = expr (root_->third).evaluate (df);
          validity_bitmap valid (df.nrows (), false);
          if (then_s.dtype () == data_type::utf8_string
              && else_s.dtype () == data_type::utf8_string)
            {
              std::vector<std::string> out (df.nrows (), "");
              for (std::size_t i = 0; i < df.nrows (); ++i)
                {
                  const bool take_then =
                    cond_s.is_valid (i)
                    && ((cond_s.dtype () == data_type::boolean)
                        ? (cond_s.bool_tensor ()[i] != 0)
                        : (cond_s.get_double (i) != 0.0));
                  const series &chosen = take_then ? then_s : else_s;
                  if (chosen.is_valid (i))
                    {
                      out[i] = chosen.str_values ()[i];
                      valid.set_valid (i, true);
                    }
                }
              return series (then_s.name (), std::move (out),
                             std::move (valid), df.index ());
            }
          nd_tensor<double> out (shape_type {df.nrows ()}, constants::nan);
          for (std::size_t i = 0; i < df.nrows (); ++i)
            {
              const bool take_then =
                cond_s.is_valid (i)
                && ((cond_s.dtype () == data_type::boolean)
                    ? (cond_s.bool_tensor ()[i] != 0)
                    : (cond_s.get_double (i) != 0.0));
              const series &chosen = take_then ? then_s : else_s;
              if (chosen.is_valid (i))
                {
                  out[i] = chosen.get_double (i);
                  valid.set_valid (i, true);
                }
            }
          return series (then_s.name (), out, std::move (valid), df.index ());
        }

      case expr_kind::binary_arith:
        {
          const series lhs = expr (root_->left).evaluate (df);
          const series rhs = expr (root_->right).evaluate (df);
          switch (root_->bin_op)
            {
            case binary_op_code::add: return lhs + rhs;
            case binary_op_code::sub: return lhs - rhs;
            case binary_op_code::mul: return lhs * rhs;
            case binary_op_code::div: return lhs / rhs;
            default: break;
            }
          break;
        }

      case expr_kind::binary_compare:
        {
          const series lhs = expr (root_->left).evaluate (df);
          const series rhs = expr (root_->right).evaluate (df);
          nd_tensor<std::uint8_t> out (shape_type {df.nrows ()}, 0);
          validity_bitmap valid = lhs.validity () & rhs.validity ();
          if (lhs.dtype () == data_type::utf8_string
              && rhs.dtype () == data_type::utf8_string)
            {
              for (std::size_t i = 0; i < df.nrows (); ++i)
                {
                  if (valid.is_valid (i))
                    {
                      const std::string &a = lhs.str_values ()[i];
                      const std::string &b = rhs.str_values ()[i];
                      bool res = false;
                      switch (root_->bin_op)
                        {
                        case binary_op_code::eq:  res = (a == b); break;
                        case binary_op_code::neq: res = (a != b); break;
                        case binary_op_code::lt:  res = (a < b); break;
                        case binary_op_code::le:  res = (a <= b); break;
                        case binary_op_code::gt:  res = (a > b); break;
                        case binary_op_code::ge:  res = (a >= b); break;
                        default: break;
                        }
                      out[i] = res ? 1 : 0;
                    }
                }
              return series (lhs.name (), out, std::move (valid), df.index ());
            }
          for (std::size_t i = 0; i < df.nrows (); ++i)
            {
              if (valid.is_valid (i))
                {
                  const double a = lhs.get_double (i);
                  const double b = rhs.get_double (i);
                  bool res = false;
                  switch (root_->bin_op)
                    {
                    case binary_op_code::eq:  res = (a == b); break;
                    case binary_op_code::neq: res = (a != b); break;
                    case binary_op_code::lt:  res = (a < b); break;
                    case binary_op_code::le:  res = (a <= b); break;
                    case binary_op_code::gt:  res = (a > b); break;
                    case binary_op_code::ge:  res = (a >= b); break;
                    default: break;
                    }
                  out[i] = res ? 1 : 0;
                }
            }
          return series (lhs.name (), out, std::move (valid), df.index ());
        }

      case expr_kind::binary_logical:
        {
          const series lhs = expr (root_->left).evaluate (df);
          const series rhs = expr (root_->right).evaluate (df);
          nd_tensor<std::uint8_t> out (shape_type {df.nrows ()}, 0);
          validity_bitmap valid = lhs.validity () & rhs.validity ();
          for (std::size_t i = 0; i < df.nrows (); ++i)
            {
              if (valid.is_valid (i))
                {
                  const bool a =
                    (lhs.dtype () == data_type::boolean)
                    ? (lhs.bool_tensor ()[i] != 0)
                    : (lhs.get_double (i) != 0.0);
                  const bool b =
                    (rhs.dtype () == data_type::boolean)
                    ? (rhs.bool_tensor ()[i] != 0)
                    : (rhs.get_double (i) != 0.0);
                  out[i] =
                    (root_->bin_op == binary_op_code::logic_and ? (a && b)
                                                                : (a || b))
                    ? 1
                    : 0;
                }
            }
          return series (lhs.name (), out, std::move (valid), df.index ());
        }

      case expr_kind::aggregation:
        throw schema_error ("aggregation expr must be evaluated in group_by");

      case expr_kind::window_over:
        {
          if (!root_->left || root_->left->kind != expr_kind::aggregation
              || !root_->left->left)
            {
              throw schema_error ("over() requires an aggregation expression");
            }
          const std::string part_col = root_->text;
          const expr inner_expr (root_->left->left);
          const std::string target_col = inner_expr.output_name ();
          const agg_op op = root_->left->agg;

          data_frame work_df = df;
          std::string src_col_name = target_col;
          if (inner_expr.root ()->kind != expr_kind::column_ref)
            {
              src_col_name = "__win_inner__";
              work_df = work_df.with_column (
                inner_expr.evaluate (df).alias (src_col_name));
            }

          const data_frame grouped =
            work_df.group_by (part_col).agg (
              {agg_spec (src_col_name, op, "__win_val")});
          std::unordered_map<std::string, double> lookup;
          for (std::size_t r = 0; r < grouped.nrows (); ++r)
            {
              lookup[cell_hash_key (grouped[part_col].iloc (r))] =
                grouped["__win_val"].get_double (r);
            }

          nd_tensor<double> out (shape_type {df.nrows ()}, constants::nan);
          validity_bitmap valid (df.nrows (), false);
          for (std::size_t r = 0; r < df.nrows (); ++r)
            {
              if (df[part_col].is_valid (r))
                {
                  auto it = lookup.find (cell_hash_key (df[part_col].iloc (r)));
                  if (it != lookup.end () && !std::isnan (it->second))
                    {
                      out[r] = it->second;
                      valid.set_valid (r, true);
                    }
                }
            }
          return series (target_col + "_over_" + part_col,
                         out, std::move (valid), df.index ());
        }
      }

    throw schema_error ("unsupported expression node in evaluate");
  }
};

/* Free-standing Polars-style DSL builders: col, lit, when.  */
inline expr
col (std::string name)
{
  return expr::col (std::move (name));
}

inline expr
lit (double val)
{
  return expr::lit (val);
}

inline expr
lit (std::string val)
{
  return expr::lit (std::move (val));
}

struct when_then_builder
{
  expr cond_expr;
  expr then_expr;

  expr
  otherwise (expr else_expr) const
  {
    auto n = std::make_shared<expr::node> ();
    n->kind = expr_kind::ternary_when;
    n->left = cond_expr.root ();
    n->right = then_expr.root ();
    n->third = else_expr.root ();
    return expr (std::move (n));
  }
};

struct when_builder
{
  expr cond_expr;

  when_then_builder
  then (expr then_expr) const
  {
    return when_then_builder {cond_expr, std::move (then_expr)};
  }
};

inline when_builder
when (expr condition)
{
  return when_builder {std::move (condition)};
}

/* Polars `coalesce` expression: evaluates to the first non-null value across
   `exprs` for each row (Polars pl.coalesce equivalent).  */
inline expr
coalesce (const std::vector<expr> &exprs)
{
  if (exprs.empty ())
    {
      throw schema_error ("coalesce requires at least one expression");
    }
  expr acc = exprs.back ();
  for (std::size_t k = exprs.size () - 1; k > 0; --k)
    {
      const expr &cur = exprs[k - 1];
      acc = when (cur.is_not_null ()).then (cur).otherwise (acc);
    }
  return acc.alias (exprs.front ().output_name ());
}

/* Expression binary operators.  */
inline expr
make_bin_expr (expr_kind k, binary_op_code op, const expr &a, const expr &b)
{
  auto n = std::make_shared<expr::node> ();
  n->kind = k;
  n->bin_op = op;
  n->left = a.root ();
  n->right = b.root ();
  return expr (std::move (n));
}

inline expr operator+ (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_arith, binary_op_code::add, a, b);
}
inline expr operator+ (const expr &a, double b) { return a + lit (b); }
inline expr operator+ (double a, const expr &b) { return lit (a) + b; }

inline expr operator- (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_arith, binary_op_code::sub, a, b);
}
inline expr operator- (const expr &a, double b) { return a - lit (b); }
inline expr operator- (double a, const expr &b) { return lit (a) - b; }

inline expr operator* (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_arith, binary_op_code::mul, a, b);
}
inline expr operator* (const expr &a, double b) { return a * lit (b); }
inline expr operator* (double a, const expr &b) { return lit (a) * b; }

inline expr operator/ (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_arith, binary_op_code::div, a, b);
}
inline expr operator/ (const expr &a, double b) { return a / lit (b); }
inline expr operator/ (double a, const expr &b) { return lit (a) / b; }

inline expr operator== (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_compare, binary_op_code::eq, a, b);
}
inline expr operator== (const expr &a, double b) { return a == lit (b); }
inline expr operator== (const expr &a, std::string b)
{
  return a == lit (std::move (b));
}

inline expr operator!= (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_compare, binary_op_code::neq, a, b);
}
inline expr operator!= (const expr &a, double b) { return a != lit (b); }
inline expr operator!= (const expr &a, std::string b)
{
  return a != lit (std::move (b));
}

inline expr operator< (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_compare, binary_op_code::lt, a, b);
}
inline expr operator< (const expr &a, double b) { return a < lit (b); }
inline expr operator< (const expr &a, std::string b)
{
  return a < lit (std::move (b));
}

inline expr operator<= (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_compare, binary_op_code::le, a, b);
}
inline expr operator<= (const expr &a, double b) { return a <= lit (b); }
inline expr operator<= (const expr &a, std::string b)
{
  return a <= lit (std::move (b));
}

inline expr operator> (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_compare, binary_op_code::gt, a, b);
}
inline expr operator> (const expr &a, double b) { return a > lit (b); }
inline expr operator> (const expr &a, std::string b)
{
  return a > lit (std::move (b));
}

inline expr operator>= (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_compare, binary_op_code::ge, a, b);
}
inline expr operator>= (const expr &a, double b) { return a >= lit (b); }
inline expr operator>= (const expr &a, std::string b)
{
  return a >= lit (std::move (b));
}

inline expr operator&& (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_logical,
                        binary_op_code::logic_and, a, b);
}

inline expr operator|| (const expr &a, const expr &b)
{
  return make_bin_expr (expr_kind::binary_logical,
                        binary_op_code::logic_or, a, b);
}

/* Logical Query Plan Nodes and Rule-Based Algebraic Optimizer.  */
enum class plan_node_kind
{
  scan,
  filter,
  select,
  with_columns,
  aggregate,
  sort,
  limit,
  slice,
  unique,
  join
};

struct logical_plan_node
{
  plan_node_kind kind;
  data_frame source_df;
  std::vector<std::string> projected_scan_cols;
  expr predicate;
  std::vector<expr> exprs;
  std::vector<std::string> group_keys;
  std::string sort_col;
  bool sort_asc = true;
  std::vector<std::string> sort_cols;
  std::vector<bool> sort_ascs;
  std::size_t limit_count = 0;
  std::size_t slice_offset = 0;
  std::string left_on;
  std::string right_on;
  std::string join_how = "inner";
  std::shared_ptr<logical_plan_node> input;
  std::shared_ptr<logical_plan_node> right_input;
};

namespace detail
{

  /* Rule 1 & 2 & 3: Constant folding, Filter Fusion, and Predicate Pushdown.
     Pushes filter predicates past select, with_columns, and sort nodes down
     toward the scan boundary whenever the predicate only references columns
     already available in the child input.  */
  inline std::shared_ptr<logical_plan_node>
  pushdown_predicates (const std::shared_ptr<logical_plan_node> &plan)
  {
    if (!plan)
      {
        return nullptr;
      }
    auto node = std::make_shared<logical_plan_node> (*plan);
    if (node->input)
      {
        node->input = pushdown_predicates (node->input);
      }
    if (node->right_input)
      {
        node->right_input = pushdown_predicates (node->right_input);
      }

    if (node->kind == plan_node_kind::filter)
      {
        node->predicate = node->predicate.fold_constants ();
        const bool row_indep = node->predicate.is_row_independent ();

        /* Filter fusion: Filter(p2, Filter(p1, child)) -> Filter(p1 && p2). */
        if (row_indep
            && node->input && node->input->kind == plan_node_kind::filter)
          {
            auto fused = std::make_shared<logical_plan_node> (*node->input);
            fused->predicate =
              (fused->predicate && node->predicate).fold_constants ();
            return pushdown_predicates (fused);
          }

        /* Push Filter below Sort: Filter(p, Sort(child)) -> Sort(Filter(p)). */
        if (row_indep
            && node->input && node->input->kind == plan_node_kind::sort)
          {
            auto sort_node = std::make_shared<logical_plan_node> (*node->input);
            auto new_filter = std::make_shared<logical_plan_node> (*node);
            new_filter->input = sort_node->input;
            sort_node->input = pushdown_predicates (new_filter);
            return sort_node;
          }

        /* Push Filter below Select / WithColumns if predicate only uses
           pure source columns not redefined by the projection.  */
        if (node->input
            && (node->input->kind == plan_node_kind::select
                || node->input->kind == plan_node_kind::with_columns))
          {
            std::set<std::string> pred_cols;
            node->predicate.collect_referenced_columns (pred_cols);

            std::set<std::string> computed_aliases;
            for (const auto &e : node->input->exprs)
              {
                if (e.root () && e.root ()->kind != expr_kind::column_ref)
                  {
                    computed_aliases.insert (e.output_name ());
                  }
              }

            bool depends_on_computed = false;
            for (const auto &pc : pred_cols)
              {
                if (computed_aliases.find (pc) != computed_aliases.end ())
                  {
                    depends_on_computed = true;
                    break;
                  }
              }

            if (!depends_on_computed)
              {
                auto proj_node =
                  std::make_shared<logical_plan_node> (*node->input);
                auto new_filter = std::make_shared<logical_plan_node> (*node);
                new_filter->input = proj_node->input;
                proj_node->input = pushdown_predicates (new_filter);
                return proj_node;
              }
          }
      }

    for (auto &e : node->exprs)
      {
        e = e.fold_constants ();
      }
    return node;
  }

  /* Rule 4: Projection Pushdown (Column Pruning).
     Walks top-down collecting the minimal set of columns required by
     downstream operators and prunes unused columns right at the Scan node.  */
  inline std::shared_ptr<logical_plan_node>
  pushdown_projections (const std::shared_ptr<logical_plan_node> &plan,
                        std::set<std::string> required_cols)
  {
    if (!plan)
      {
        return nullptr;
      }
    auto node = std::make_shared<logical_plan_node> (*plan);

    switch (node->kind)
      {
      case plan_node_kind::scan:
        {
          if (!required_cols.empty ())
            {
              std::vector<std::string> pruned;
              for (const auto &cname : node->source_df.column_names ())
                {
                  if (required_cols.find (cname) != required_cols.end ())
                    {
                      pruned.push_back (cname);
                    }
                }
              if (!pruned.empty ()
                  && pruned.size () < node->source_df.ncols ())
                {
                  node->projected_scan_cols = std::move (pruned);
                }
            }
          return node;
        }

      case plan_node_kind::filter:
        if (!required_cols.empty ())
          {
            node->predicate.collect_referenced_columns (required_cols);
          }
        node->input =
          pushdown_projections (node->input, std::move (required_cols));
        return node;

      case plan_node_kind::select:
        {
          std::set<std::string> child_req;
          for (const auto &e : node->exprs)
            {
              e.collect_referenced_columns (child_req);
            }
          node->input =
            pushdown_projections (node->input, std::move (child_req));
          return node;
        }

      case plan_node_kind::with_columns:
        {
          /* Prune any computed with_columns expression whose output column is
             never referenced downstream when required_cols is known.  */
          if (!required_cols.empty ())
            {
              std::vector<expr> live_exprs;
              for (const auto &e : node->exprs)
                {
                  if (required_cols.find (e.output_name ())
                      != required_cols.end ())
                    {
                      live_exprs.push_back (e);
                    }
                }
              node->exprs = std::move (live_exprs);
              for (const auto &e : node->exprs)
                {
                  e.collect_referenced_columns (required_cols);
                }
            }
          node->input =
            pushdown_projections (node->input, std::move (required_cols));
          if (node->exprs.empty () && node->input)
            {
              return node->input;
            }
          return node;
        }

      case plan_node_kind::aggregate:
        {
          std::set<std::string> child_req (node->group_keys.begin (),
                                           node->group_keys.end ());
          for (const auto &e : node->exprs)
            {
              e.collect_referenced_columns (child_req);
            }
          node->input =
            pushdown_projections (node->input, std::move (child_req));
          return node;
        }

      case plan_node_kind::sort:
        if (! required_cols.empty ())
          {
            if (!node->sort_col.empty ())
              {
                required_cols.insert (node->sort_col);
              }
            for (const auto &sc : node->sort_cols)
              {
                required_cols.insert (sc);
              }
          }
        node->input =
          pushdown_projections (node->input, std::move (required_cols));
        return node;

      case plan_node_kind::limit:
      case plan_node_kind::slice:
        node->input =
          pushdown_projections (node->input, std::move (required_cols));
        return node;

      case plan_node_kind::unique:
        if (!required_cols.empty ())
          {
            for (const auto &k : node->group_keys)
              {
                required_cols.insert (k);
              }
          }
        node->input =
          pushdown_projections (node->input, std::move (required_cols));
        return node;

      case plan_node_kind::join:
        node->input = pushdown_projections (node->input, {});
        node->right_input = pushdown_projections (node->right_input, {});
        return node;
      }

    return node;
  }

  inline std::shared_ptr<logical_plan_node>
  optimize_plan (const std::shared_ptr<logical_plan_node> &plan)
  {
    auto after_pred = pushdown_predicates (plan);
    return pushdown_projections (after_pred, {});
  }

  inline void
  format_plan_tree (const std::shared_ptr<logical_plan_node> &node,
                    std::ostringstream &oss,
                    std::size_t indent = 0)
  {
    if (!node)
      {
        return;
      }
    const std::string pad (indent * 2, ' ');
    switch (node->kind)
      {
      case plan_node_kind::scan:
        oss << pad << "SCAN [rows=" << node->source_df.nrows ()
            << ", cols=";
        if (node->projected_scan_cols.empty ())
          {
            oss << "ALL(" << node->source_df.ncols () << ")";
          }
        else
          {
            oss << "[";
            for (std::size_t i = 0; i < node->projected_scan_cols.size (); ++i)
              {
                if (i > 0)
                  {
                    oss << ", ";
                  }
                oss << node->projected_scan_cols[i];
              }
            oss << "]";
          }
        oss << "]\n";
        break;

      case plan_node_kind::filter:
        oss << pad << "FILTER [" << node->predicate.to_string () << "]\n";
        break;

      case plan_node_kind::select:
        oss << pad << "SELECT [";
        for (std::size_t i = 0; i < node->exprs.size (); ++i)
          {
            if (i > 0)
              {
                oss << ", ";
              }
            oss << node->exprs[i].to_string ();
          }
        oss << "]\n";
        break;

      case plan_node_kind::with_columns:
        oss << pad << "WITH_COLUMNS [";
        for (std::size_t i = 0; i < node->exprs.size (); ++i)
          {
            if (i > 0)
              {
                oss << ", ";
              }
            oss << node->exprs[i].to_string ();
          }
        oss << "]\n";
        break;

      case plan_node_kind::aggregate:
        oss << pad << "AGGREGATE by=[";
        for (std::size_t i = 0; i < node->group_keys.size (); ++i)
          {
            if (i > 0)
              {
                oss << ", ";
              }
            oss << node->group_keys[i];
          }
        oss << "]\n";
        break;

      case plan_node_kind::sort:
        oss << pad << "SORT by=" << node->sort_col
            << (node->sort_asc ? " ASC\n" : " DESC\n");
        break;

      case plan_node_kind::limit:
        oss << pad << "LIMIT n=" << node->limit_count << "\n";
        break;

      case plan_node_kind::slice:
        oss << pad << "SLICE offset=" << node->slice_offset
            << " len=" << node->limit_count << "\n";
        break;

      case plan_node_kind::unique:
        oss << pad << "UNIQUE\n";
        break;

      case plan_node_kind::join:
        oss << pad << "JOIN (" << node->join_how << ") on ["
            << node->left_on << " == " << node->right_on << "]\n";
        break;
      }

    if (node->input)
      {
        format_plan_tree (node->input, oss, indent + 1);
      }
    if (node->right_input)
      {
        format_plan_tree (node->right_input, oss, indent + 1);
      }
  }

  inline data_frame
  execute_plan (const std::shared_ptr<logical_plan_node> &node)
  {
    if (!node)
      {
        return data_frame ();
      }

    switch (node->kind)
      {
      case plan_node_kind::scan:
        if (!node->projected_scan_cols.empty ())
          {
            return node->source_df.select (node->projected_scan_cols);
          }
        return node->source_df;

      case plan_node_kind::filter:
        {
          const data_frame child_df = execute_plan (node->input);
          const series mask = node->predicate.evaluate (child_df);
          return child_df.filter (mask);
        }

      case plan_node_kind::select:
        {
          const data_frame child_df = execute_plan (node->input);
          std::vector<series> cols;
          cols.reserve (node->exprs.size ());
          for (const auto &e : node->exprs)
            {
              series s = e.evaluate (child_df);
              s.set_name (e.output_name ());
              cols.push_back (std::move (s));
            }
          return data_frame (std::move (cols), child_df.index ());
        }

      case plan_node_kind::with_columns:
        {
          data_frame out_df = execute_plan (node->input);
          for (const auto &e : node->exprs)
            {
              series s = e.evaluate (out_df);
              s.set_name (e.output_name ());
              out_df = out_df.with_column (std::move (s));
            }
          return out_df;
        }

      case plan_node_kind::aggregate:
        {
          data_frame child_df = execute_plan (node->input);
          std::vector<agg_spec> specs;
          std::size_t temp_idx = 0;
          for (const auto &e : node->exprs)
            {
              std::string out_alias = e.output_name ();
              auto cur = e.root ();
              if (cur && cur->kind == expr_kind::aliased)
                {
                  cur = cur->left;
                }
              if (!cur || cur->kind != expr_kind::aggregation || !cur->left)
                {
                  throw schema_error ("group_by.agg requires aggregation expr");
                }
              const expr inner_expr (cur->left);
              std::string target_col = inner_expr.output_name ();
              if (cur->left->kind != expr_kind::column_ref)
                {
                  target_col = "__agg_tmp_" + std::to_string (temp_idx++);
                  child_df = child_df.with_column (
                    inner_expr.evaluate (child_df).alias (target_col));
                }
              specs.emplace_back (target_col, cur->agg, out_alias);
            }
          return child_df.group_by (node->group_keys).agg (specs);
        }

      case plan_node_kind::sort:
        if (!node->sort_cols.empty ())
          {
            return execute_plan (node->input).sort_by (node->sort_cols,
                                                       node->sort_ascs);
          }
        return execute_plan (node->input).sort_by (node->sort_col,
                                                   node->sort_asc);

      case plan_node_kind::limit:
        return execute_plan (node->input).head (node->limit_count);

      case plan_node_kind::slice:
        return execute_plan (node->input).slice (node->slice_offset,
                                                 node->limit_count);

      case plan_node_kind::unique:
        return execute_plan (node->input).drop_duplicates (node->group_keys);

      case plan_node_kind::join:
        {
          const data_frame l_df = execute_plan (node->input);
          const data_frame r_df = execute_plan (node->right_input);
          return l_df.join (r_df, node->left_on, node->right_on, node->join_how);
        }
      }

    return data_frame ();
  }

} // namespace detail

/* Polars LazyFrame equivalent: builds a declarative query plan DAG and
   executes it through the algebraic query optimizer on `.collect ()`.  */
class lazy_frame
{
private:
  std::shared_ptr<logical_plan_node> plan_;

  explicit
  lazy_frame (std::shared_ptr<logical_plan_node> p)
    : plan_ (std::move (p))
  {
  }

public:
  explicit
  lazy_frame (const data_frame &df)
    : plan_ (std::make_shared<logical_plan_node> ())
  {
    plan_->kind = plan_node_kind::scan;
    plan_->source_df = df;
  }

  lazy_frame
  filter (expr predicate) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::filter;
    n->predicate = std::move (predicate);
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  select (std::vector<expr> exprs) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::select;
    n->exprs = std::move (exprs);
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  with_columns (std::vector<expr> exprs) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::with_columns;
    n->exprs = std::move (exprs);
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  sort (std::string col_name, bool ascending = true) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::sort;
    n->sort_col = std::move (col_name);
    n->sort_asc = ascending;
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  sort (std::vector<std::string> col_names,
        std::vector<bool> ascending = {}) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::sort;
    n->sort_cols = std::move (col_names);
    n->sort_ascs = std::move (ascending);
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  limit (std::size_t count) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::limit;
    n->limit_count = count;
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  top_k (std::size_t k, std::string col_name) const
  {
    return sort (std::move (col_name), false).limit (k);
  }

  lazy_frame
  bottom_k (std::size_t k, std::string col_name) const
  {
    return sort (std::move (col_name), true).limit (k);
  }

  lazy_frame
  slice (std::size_t offset, std::size_t length) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::slice;
    n->slice_offset = offset;
    n->limit_count = length;
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  unique (std::vector<std::string> subset = {}) const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::unique;
    n->group_keys = std::move (subset);
    n->input = plan_;
    return lazy_frame (std::move (n));
  }

  lazy_frame
  join (const lazy_frame &other,
        std::string left_on,
        std::string right_on,
        std::string how = "inner") const
  {
    auto n = std::make_shared<logical_plan_node> ();
    n->kind = plan_node_kind::join;
    n->left_on = std::move (left_on);
    n->right_on = std::move (right_on);
    n->join_how = std::move (how);
    n->input = plan_;
    n->right_input = other.plan_;
    return lazy_frame (std::move (n));
  }

  struct lazy_group_by
  {
    std::shared_ptr<logical_plan_node> input_plan;
    std::vector<std::string> keys;

    lazy_frame
    agg (std::vector<expr> agg_exprs) const
    {
      auto n = std::make_shared<logical_plan_node> ();
      n->kind = plan_node_kind::aggregate;
      n->group_keys = keys;
      n->exprs = std::move (agg_exprs);
      n->input = input_plan;
      return lazy_frame (std::move (n));
    }
  };

  lazy_group_by
  group_by (std::vector<std::string> keys) const
  {
    return lazy_group_by {plan_, std::move (keys)};
  }

  lazy_group_by
  group_by (const std::string &key) const
  {
    return lazy_group_by {plan_, {key}};
  }

  std::string
  explain (bool optimized = true) const
  {
    std::ostringstream oss;
    const auto target =
      optimized ? detail::optimize_plan (plan_) : plan_;
    detail::format_plan_tree (target, oss, 0);
    return oss.str ();
  }

  data_frame
  collect (bool optimize = true) const
  {
    const auto target =
      optimize ? detail::optimize_plan (plan_) : plan_;
    return detail::execute_plan (target);
  }

  /* Extended Polars `lazy_frame` relational operations:
     drop, rename, reverse, shift, fill_null, drop_nulls, tail,
     join_cross, join_asof, unpivot (melt), and concat_lazy.  */
  lazy_frame
  drop (const std::vector<std::string> &cols) const
  {
    return lazy_frame (collect (true).drop (cols));
  }

  lazy_frame
  rename (const std::string &old_name, const std::string &new_name) const
  {
    return lazy_frame (collect (true).rename (old_name, new_name));
  }

  lazy_frame
  reverse () const
  {
    const data_frame df = collect (true);
    std::vector<std::size_t> order (df.nrows ());
    for (std::size_t i = 0; i < df.nrows (); ++i)
      {
        order[i] = df.nrows () - 1 - i;
      }
    return lazy_frame (df.take (order));
  }

  lazy_frame
  shift (std::ptrdiff_t periods = 1) const
  {
    return lazy_frame (collect (true).shift (periods));
  }

  lazy_frame
  fill_null (double val) const
  {
    return lazy_frame (collect (true).fill_null (val));
  }

  lazy_frame
  drop_nulls () const
  {
    return lazy_frame (collect (true).drop_nulls ());
  }

  lazy_frame
  tail (std::size_t n) const
  {
    return lazy_frame (collect (true).tail (n));
  }

  lazy_frame
  join_cross (const lazy_frame &other,
              const std::string &suffix = "_right") const
  {
    return lazy_frame (collect (true).join_cross (other.collect (true), suffix));
  }

  lazy_frame
  join_asof (const lazy_frame &other, const std::string &on_col) const
  {
    return lazy_frame (collect (true).join_asof (other.collect (true), on_col));
  }

  lazy_frame
  unpivot (const std::vector<std::string> &id_vars,
           const std::vector<std::string> &value_vars) const
  {
    return lazy_frame (collect (true).melt (id_vars, value_vars));
  }

  lazy_frame
  pivot (const std::string &index_col,
         const std::string &columns_col,
         const std::string &values_col,
         agg_op op = agg_op::sum) const
  {
    return lazy_frame (
      collect (true).pivot (index_col, columns_col, values_col, op));
  }

  lazy_frame
  with_row_index (const std::string &name = "index") const
  {
    return lazy_frame (collect (true).with_row_index (name));
  }

  lazy_frame
  merge_sorted (const lazy_frame &other, const std::string &key_col) const
  {
    return lazy_frame (
      collect (true).merge_sorted (other.collect (true), key_col));
  }

  lazy_frame
  group_by_dynamic (const std::string &time_col,
                    std::int64_t every_step,
                    const std::string &val_col,
                    const std::string &agg = "sum") const
  {
    return lazy_frame (
      collect (true).group_by_dynamic (time_col, every_step, val_col, agg));
  }

  lazy_frame
  explode_delimited (const std::string &col_name, char delim = ',') const
  {
    return lazy_frame (collect (true).explode_delimited (col_name, delim));
  }

  void
  sink_csv (const std::string &filepath) const
  {
    collect (true).write_csv (filepath);
  }

  void
  sink_ndjson (const std::string &filepath) const
  {
    collect (true).write_ndjson (filepath);
  }

  void
  sink_hmb (const std::string &filepath) const
  {
    collect (true).write_hmb (filepath);
  }
};

inline lazy_frame
concat_lazy (const std::vector<lazy_frame> &frames)
{
  if (frames.empty ())
    {
      throw schema_error ("concat_lazy requires at least one lazy_frame");
    }
  std::vector<data_frame> dfs;
  dfs.reserve (frames.size ());
  for (const auto &lf : frames)
    {
      dfs.push_back (lf.collect (true));
    }
  return lazy_frame (data_frame::concat_rows (dfs));
}

inline lazy_frame
scan_csv (const std::string &filepath)
{
  return lazy_frame (data_frame::read_csv (filepath));
}

inline lazy_frame
scan_ndjson (const std::string &filepath)
{
  return lazy_frame (data_frame::read_ndjson (filepath));
}

inline lazy_frame
scan_hmb (const std::string &filepath)
{
  return lazy_frame (data_frame::read_hmb (filepath));
}

class sql_context
{
private:
  std::map<std::string, data_frame> tables_;

public:
  void
  register_table (const std::string &name, const data_frame &df)
  {
    tables_[name] = df;
  }

  void
  register_table (const std::string &name, const lazy_frame &lf)
  {
    tables_[name] = lf.collect (true);
  }

  data_frame
  execute (const std::string &sql) const
  {
    const std::size_t p_from = sql.find (" FROM ");
    if (p_from == std::string::npos)
      {
        throw parse_error ("sql_context::execute requires FROM <table_name>");
      }
    std::istringstream iss (sql.substr (p_from + 6));
    std::string tbl_name;
    iss >> tbl_name;
    auto it = tables_.find (tbl_name);
    if (it == tables_.end ())
      {
        throw schema_error ("sql_context unknown table: " + tbl_name);
      }
    return it->second.sql_query (sql);
  }
};

} // namespace hyper_math::tabular

#endif /* HYPER_MATH_TABULAR_LAZY_HPP */
