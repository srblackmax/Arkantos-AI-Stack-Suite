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

#include "hyper_math/hyper_math.hpp"
#include "hyper_math/hyper_math_c.h"

#include <cmath>
#include <iostream>

#define CHECK_TRUE(COND)                                                      \
  do                                                                          \
    {                                                                         \
      if (!(COND))                                                            \
        {                                                                     \
          std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: "       \
                    << #COND << "\n";                                         \
          return 1;                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

#define CHECK_NEAR(A, B, TOL)                                                 \
  do                                                                          \
    {                                                                         \
      const double _va = static_cast<double> (A);                             \
      const double _vb = static_cast<double> (B);                             \
      if (std::abs (_va - _vb) > (TOL))                                       \
        {                                                                     \
          std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: "       \
                    << #A << " (" << _va << ") != " << #B << " (" << _vb      \
                    << ")\n";                                                 \
          return 1;                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

int
run_tabular_tests ()
{
  using namespace hyper_math;
  using namespace hyper_math::tabular;

  std::cout << "[Suite 3/3] Dual-Bitmask Tabular & Lazy Engine (Pandas/Polars)...\n";

  /* 1. Pandas-style Series: Index alignment, loc/iloc, null bitmask,
     ffill/bfill, and rolling windows.  */
  {
    series s1 ("a", std::vector<double> {10.0, 20.0, 30.0},
               index_axis ({"x", "y", "z"}));
    series s2 ("b", std::vector<double> {100.0, 200.0, 300.0},
               index_axis ({"y", "z", "w"}));

    CHECK_NEAR (cell_as_double (s1.loc ("y")), 20.0, 1e-12);
    CHECK_NEAR (cell_as_double (s1.iloc (0)), 10.0, 1e-12);

    /* Aligned addition over ("x", "y", "z") + ("y", "z", "w")
       produces union index ("x", "y", "z", "w") where "x" and "w" are null
       and "y" = 120.0, "z" = 230.0.  */
    series s_sum = s1 + s2;
    CHECK_TRUE (s_sum.size () == 4);
    CHECK_TRUE (s_sum.null_count () == 2);
    CHECK_NEAR (cell_as_double (s_sum.loc ("y")), 120.0, 1e-12);
    CHECK_NEAR (cell_as_double (s_sum.loc ("z")), 230.0, 1e-12);

    auto dropped = s_sum.drop_nulls ();
    CHECK_TRUE (dropped.size () == 2);
    CHECK_NEAR (dropped.sum (), 350.0, 1e-12);

    auto ffilled = s_sum.fill_forward ();
    CHECK_NEAR (cell_as_double (ffilled.loc ("w")), 230.0, 1e-12);

    auto bfilled = s_sum.fill_backward ();
    CHECK_NEAR (cell_as_double (bfilled.loc ("x")), 120.0, 1e-12);

    /* Rolling mean window.  */
    series ts ("price", std::vector<double> {2.0, 4.0, 6.0, 8.0, 10.0});
    auto rmean = ts.rolling_mean (3);
    CHECK_TRUE (rmean.is_null (0) && rmean.is_null (1));
    CHECK_NEAR (rmean.get_double (2), 4.0, 1e-12);
    CHECK_NEAR (rmean.get_double (4), 8.0, 1e-12);
  }

  /* 2. DataFrame: Group-By, Joins (inner, left, semi, anti), Pivot, Melt,
     and RFC-4180 CSV roundtrip.  */
  {
    data_frame emp ({
      series ("dept", std::vector<std::string> {"Eng", "Eng", "Sales", "Sales"}),
      series ("quarter", std::vector<std::string> {"Q1", "Q2", "Q1", "Q2"}),
      series ("revenue", std::vector<double> {100.0, 150.0, 80.0, 120.0})
    });

    auto grp = emp.group_by ("dept").agg ({
      agg_spec ("revenue", agg_op::sum, "rev_sum"),
      agg_spec ("revenue", agg_op::mean, "rev_mean")
    });
    CHECK_TRUE (grp.nrows () == 2);
    CHECK_NEAR (grp["rev_sum"].get_double (0), 250.0, 1e-12);
    CHECK_NEAR (grp["rev_mean"].get_double (1), 100.0, 1e-12);

    /* Pivot table: rows = dept, cols = quarter, values = revenue.  */
    auto pivoted = emp.pivot ("dept", "quarter", "revenue", agg_op::sum);
    CHECK_TRUE (pivoted.nrows () == 2);
    CHECK_TRUE (pivoted.has_column ("Q1") && pivoted.has_column ("Q2"));
    CHECK_NEAR (pivoted["Q1"].get_double (0), 100.0, 1e-12);
    CHECK_NEAR (pivoted["Q2"].get_double (0), 150.0, 1e-12);

    /* Melt back to long format.  */
    auto melted = pivoted.melt ({"dept"}, {"Q1", "Q2"}, "quarter", "revenue");
    CHECK_TRUE (melted.nrows () == 4);

    /* Hash Join tests.  */
    data_frame meta ({
      series ("dept", std::vector<std::string> {"Eng", "HR"}),
      series ("budget", std::vector<double> {500.0, 90.0})
    });
    auto inner_j = emp.join (meta, "dept", "dept", "inner");
    CHECK_TRUE (inner_j.nrows () == 2);
    auto left_j = emp.join (meta, "dept", "dept", "left");
    CHECK_TRUE (left_j.nrows () == 4);
    CHECK_TRUE (left_j["budget"].null_count () == 2);
    auto anti_j = emp.join (meta, "dept", "dept", "anti");
    CHECK_TRUE (anti_j.nrows () == 2);

    /* CSV serialization and automatic schema/null inference roundtrip.  */
    const std::string csv_txt = left_j.to_csv_string ();
    const data_frame parsed = data_frame::from_csv_string (csv_txt);
    CHECK_TRUE (parsed.nrows () == 4);
    CHECK_TRUE (parsed["budget"].null_count () == 2);
    CHECK_NEAR (parsed["revenue"].sum (), 450.0, 1e-12);
  }

  /* 3. Polars LazyFrame & Algebraic Query Optimizer (Constant Folding,
     Filter Fusion, Predicate Pushdown, and Projection Pushdown).  */
  {
    data_frame sales ({
      series ("region", std::vector<std::string> {"NA", "EU", "NA", "APAC"}),
      series ("price", std::vector<double> {50.0, 20.0, 80.0, 100.0}),
      series ("qty", std::vector<double> {2.0, 5.0, 3.0, 1.0}),
      series ("unused_notes", std::vector<std::string> {"a", "b", "c", "d"})
    });

    lazy_frame query =
      lazy_frame (sales)
        .select ({
          col ("region"),
          col ("price"),
          (col ("price") * col ("qty")).alias ("gross")
        })
        .filter (col ("price") > lit (10.0) + lit (20.0))
        .filter (col ("price") < lit (95.0))
        .sort ("gross", false);

    const std::string unopt_plan = query.explain (false);
    const std::string opt_plan = query.explain (true);

    /* Verify that in the optimized plan:
       1. Constant folding simplified (10 + 20) into 30.
       2. Consecutive filters fused into a single '&&' filter.
       3. Predicate pushdown moved FILTER below SELECT directly above SCAN.
       4. Projection pushdown pruned 'unused_notes' from SCAN!  */
    CHECK_TRUE (opt_plan.find ("30") != std::string::npos);
    CHECK_TRUE (opt_plan.find ("&&") != std::string::npos);
    CHECK_TRUE (opt_plan.find ("unused_notes") == std::string::npos);
    CHECK_TRUE (opt_plan.find ("SELECT") < opt_plan.find ("FILTER"));

    data_frame res = query.collect (true);
    CHECK_TRUE (res.nrows () == 2);
    CHECK_NEAR (res["gross"].get_double (0), 240.0, 1e-12);
    CHECK_NEAR (res["gross"].get_double (1), 100.0, 1e-12);
  }

  /* 4. Phase 2 Tabular & Lazy Features: Series linear interpolation,
     string ops, value_counts, is_in, DataFrame corr, drop_duplicates,
     concat_rows, Binary .hmb columnar I/O, and Lazy Window `.over()`.  */
  {
    series gappy ("sensor",
                  std::vector<double> {10.0, constants::nan, constants::nan, 40.0});
    auto interp_s = gappy.interpolate ();
    CHECK_TRUE (interp_s.null_count () == 0);
    CHECK_NEAR (interp_s.get_double (1), 20.0, 1e-12);
    CHECK_NEAR (interp_s.get_double (2), 30.0, 1e-12);

    series words ("tag", std::vector<std::string> {"Alpha", "Beta", "Alpha"});
    auto vc = words.value_counts ();
    CHECK_TRUE (vc["Alpha"] == 2 && vc["Beta"] == 1);
    auto replaced = words.str_replace ("Alpha", "Omega");
    CHECK_TRUE (replaced.str_values ()[0] == "Omega");

    /* DataFrame drop_duplicates, corr, concat_rows, and .hmb binary I/O.  */
    data_frame df1 ({
      series ("k", std::vector<std::string> {"A", "A", "B"}),
      series ("x", std::vector<double> {1.0, 1.0, 2.0}),
      series ("y", std::vector<double> {2.0, 2.0, 4.0})
    });
    auto dedup = df1.drop_duplicates ();
    CHECK_TRUE (dedup.nrows () == 2);

    auto corr_df = df1.corr ();
    CHECK_NEAR (corr_df["y"].get_double (0), 1.0, 1e-12);

    auto stacked = data_frame::concat_rows ({dedup, dedup});
    CHECK_TRUE (stacked.nrows () == 4);

    const std::string hmb_path = "/tmp/test_hyper_math_table.hmb";
    df1.write_hmb (hmb_path);
    auto loaded_hmb = data_frame::read_hmb (hmb_path);
    CHECK_TRUE (loaded_hmb.nrows () == 3 && loaded_hmb.ncols () == 3);
    CHECK_NEAR (loaded_hmb["y"].sum (), 8.0, 1e-12);

    /* Polars-style Window Expression: col("val").mean().over("group").  */
    data_frame w_df ({
      series ("grp", std::vector<std::string> {"G1", "G1", "G2", "G2"}),
      series ("val", std::vector<double> {10.0, 30.0, 100.0, 200.0})
    });
    data_frame w_res =
      lazy_frame (w_df)
        .with_columns ({
          (col ("val") - col ("val").mean ().over ("grp")).alias ("centered")
        })
        .collect ();
    CHECK_NEAR (w_res["centered"].get_double (0), -10.0, 1e-12);
    CHECK_NEAR (w_res["centered"].get_double (1), 10.0, 1e-12);
    CHECK_NEAR (w_res["centered"].get_double (2), -50.0, 1e-12);
    CHECK_NEAR (w_res["centered"].get_double (3), 50.0, 1e-12);

    /* NaN-to-null bitmap propagation in arithmetic and 0-row filter.  */
    series zero_s ("z", std::vector<double> {0.0, 4.0});
    series div_s = zero_s / zero_s;
    CHECK_TRUE (div_s.is_null (0));
    CHECK_TRUE (div_s.is_valid (1));
    CHECK_NEAR (div_s.get_double (1), 1.0, 1e-12);

    data_frame empty_filtered = w_df.filter (w_df["val"] > 9999.0);
    CHECK_TRUE (empty_filtered.nrows () == 0 && empty_filtered.ncols () == 2);

    /* Phase 3: rank, cut/qcut, ISO-8601 .dt accessors, multi-column join,
       join_asof, crosstab, NDJSON roundtrip, Markdown/LaTeX export, and
       dead-column pruning in lazy with_columns.  */
    series s_rank ("v", std::vector<double> {30.0, 10.0, 20.0, 20.0});
    auto r_avg = s_rank.rank ("average");
    CHECK_NEAR (r_avg.get_double (1), 1.0, 1e-12);
    CHECK_NEAR (r_avg.get_double (2), 2.5, 1e-12);
    CHECK_NEAR (r_avg.get_double (0), 4.0, 1e-12);

    auto qbins = s_rank.qcut (2);
    CHECK_TRUE (qbins.get_double (1) == 0.0 && qbins.get_double (0) == 1.0);

    series dates ("dt", std::vector<std::string> {"1970-01-01", "2026-10-30"});
    CHECK_NEAR (dates.dt_year ().get_double (1), 2026.0, 1e-12);
    CHECK_NEAR (dates.dt_month ().get_double (1), 10.0, 1e-12);
    CHECK_NEAR (dates.dt_day ().get_double (1), 30.0, 1e-12);
    CHECK_NEAR (dates.dt_epoch_days ().get_double (0), 0.0, 1e-12);

    /* Time-series join_asof (backward match).  */
    data_frame trades ({
      series ("ts", std::vector<double> {2.0, 5.0, 9.0}),
      series ("qty", std::vector<double> {10.0, 20.0, 30.0})
    });
    data_frame quotes ({
      series ("ts", std::vector<double> {1.0, 4.0, 6.0, 8.0}),
      series ("bid", std::vector<double> {100.0, 101.0, 102.0, 105.0})
    });
    auto asof_df = trades.join_asof (quotes, "ts", "backward");
    CHECK_NEAR (asof_df["bid"].get_double (0), 100.0, 1e-12);
    CHECK_NEAR (asof_df["bid"].get_double (1), 101.0, 1e-12);
    CHECK_NEAR (asof_df["bid"].get_double (2), 105.0, 1e-12);

    /* NDJSON roundtrip & Markdown/LaTeX exporters.  */
    const std::string ndjson_path = "/tmp/test_hyper_math.ndjson";
    trades.write_ndjson (ndjson_path);
    auto from_ndjson = data_frame::read_ndjson (ndjson_path);
    CHECK_TRUE (from_ndjson.nrows () == 3 && from_ndjson.ncols () == 2);
    CHECK_NEAR (from_ndjson["qty"].sum (), 60.0, 1e-12);
    CHECK_TRUE (trades.to_markdown ().find ("| ts | qty |") != std::string::npos);
    CHECK_TRUE (trades.to_latex ().find ("\\begin{tabular}") != std::string::npos);

    /* Phase 4: Pure C ABI (hyper_math_c.h) verification.  */
    const size_t c_shape[2] = {2, 2};
    const double c_data[4] = {4.0, 1.0, 2.0, 3.0};
    hm_tensor_t *ct = hm_tensor_from_data (c_shape, 2, c_data);
    CHECK_TRUE (ct != nullptr && hm_tensor_size (ct) == 4);
    CHECK_NEAR (hm_tensor_sum (ct), 10.0, 1e-12);
    CHECK_NEAR (hm_linalg_det (ct), 10.0, 1e-12);
    hm_tensor_t *cinv = hm_linalg_inv (ct);
    hm_tensor_t *cprod = hm_tensor_matmul (ct, cinv);
    CHECK_NEAR (hm_tensor_get_flat (cprod, 0), 1.0, 1e-12);
    CHECK_NEAR (hm_tensor_get_flat (cprod, 3), 1.0, 1e-12);
    hm_tensor_free (cprod);
    hm_tensor_free (cinv);
    hm_tensor_free (ct);

    /* Phase 6: diff, pct_change, cum_prod, cum_min, cum_max, unique, nunique,
       nlargest, nsmallest, rename, and lazy top_k / bottom_k.  */
    series fin_s ("price", std::vector<double> {100.0, 110.0, 99.0, 110.0});
    CHECK_TRUE (fin_s.nunique () == 3);
    CHECK_TRUE (fin_s.unique ().size () == 3);
    CHECK_TRUE (fin_s.diff ().is_null (0));
    CHECK_NEAR (fin_s.diff ().get_double (1), 10.0, 1e-12);
    CHECK_NEAR (fin_s.pct_change ().get_double (1), 0.10, 1e-12);
    CHECK_NEAR (fin_s.cum_min ().get_double (2), 99.0, 1e-12);
    CHECK_NEAR (fin_s.cum_max ().get_double (3), 110.0, 1e-12);
    CHECK_NEAR (series ("p", std::vector<double> {2.0, 3.0, 4.0}).cum_prod ().get_double (2),
                24.0, 1e-12);

    auto top2 = trades.nlargest (2, "qty");
    CHECK_TRUE (top2.nrows () == 2);
    CHECK_NEAR (top2["qty"].get_double (0), 30.0, 1e-12);
    auto bot1 = trades.nsmallest (1, "qty");
    CHECK_NEAR (bot1["qty"].get_double (0), 10.0, 1e-12);

    auto renamed_df = trades.rename ({{"qty", "volume"}});
    CHECK_TRUE (renamed_df.has_column ("volume") && !renamed_df.has_column ("qty"));

    auto lazy_top =
      lazy_frame (trades)
        .with_columns ({col ("qty").diff ().alias ("dq")})
        .top_k (2, "qty")
        .collect ();
    CHECK_TRUE (lazy_top.nrows () == 2);
    CHECK_NEAR (lazy_top["qty"].get_double (0), 30.0, 1e-12);

    /* Phase 7: str_starts_with, str_ends_with, str_strip, between, and
       explode_delimited.  */
    series raw_tags ("tags", std::vector<std::string> {"  alpha_1 ", "beta_2"});
    auto stripped = raw_tags.str_strip ();
    CHECK_TRUE (stripped.str_values ()[0] == "alpha_1");
    CHECK_TRUE (stripped.str_starts_with ("alpha").bool_tensor ()[0] == 1);
    CHECK_TRUE (stripped.str_ends_with ("_2").bool_tensor ()[1] == 1);
    CHECK_TRUE (fin_s.between (100.0, 105.0).bool_tensor ()[0] == 1);
    CHECK_TRUE (fin_s.between (100.0, 105.0).bool_tensor ()[1] == 0);

    data_frame nested_df ({
      series ("id", std::vector<double> {1.0, 2.0}),
      series ("items", std::vector<std::string> {"a,b", "c"})
    });
    auto exp_df = nested_df.explode_delimited ("items", ',');
    CHECK_TRUE (exp_df.nrows () == 3);
    CHECK_TRUE (exp_df["items"].str_values ()[1] == "b");
    CHECK_NEAR (exp_df["id"].get_double (1), 1.0, 1e-12);

    /* Phase 8: autocorr and get_dummies (one-hot encoding).  */
    series lin_s ("lin", std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0});
    CHECK_NEAR (lin_s.autocorr (1), 1.0, 1e-12);

    auto dum_df = exp_df.get_dummies ("items");
    CHECK_TRUE (dum_df.has_column ("items_a") && dum_df.has_column ("items_b")
                && dum_df.has_column ("items_c"));
    CHECK_NEAR (dum_df["items_a"].get_double (0), 1.0, 1e-12);
    CHECK_NEAR (dum_df["items_b"].get_double (1), 1.0, 1e-12);

    /* Phase 9: series::mode, arg_min, arg_max, map_numeric, dt_day_of_week,
       dt_day_of_year, data_frame::combine_first, slice, and lazy_frame::unique/slice. */
    series m_s ("vals", std::vector<double> {3.0, 1.0, 3.0, 9.0, 2.0});
    CHECK_NEAR (m_s.mode ().get_double (0), 3.0, 1e-12);
    CHECK_TRUE (m_s.arg_min () == 1 && m_s.arg_max () == 3);
    CHECK_NEAR (m_s.map_numeric ([] (double v) { return v * 2.0; }).get_double (1),
                2.0, 1e-12);

    /* 1970-01-01 is Thursday (3) and day-of-year 1.  */
    CHECK_TRUE (dates.dt_day_of_week ().i64_tensor ()[0] == 3);
    CHECK_TRUE (dates.dt_day_of_year ().i64_tensor ()[0] == 1);

    validity_bitmap v_left (3, true);
    v_left.set_valid (1, false);
    data_frame df_primary ({
      series ("x", nd_tensor<double> {10.0, 0.0, 30.0}, std::move (v_left))
    });
    data_frame df_fallback ({
      series ("x", std::vector<double> {99.0, 20.0, 99.0}),
      series ("extra", std::vector<double> {1.0, 2.0, 3.0})
    });
    auto coalesced = df_primary.combine_first (df_fallback);
    CHECK_NEAR (coalesced["x"].get_double (1), 20.0, 1e-12);
    CHECK_TRUE (coalesced.has_column ("extra"));

    auto lazy_uniq =
      lazy_frame (df_fallback)
        .unique ({"x"})
        .slice (0, 2)
        .collect ();
    CHECK_TRUE (lazy_uniq.nrows () == 2);

    /* Phase 10: rolling_var, expanding_mean, data_frame::value_counts, and
       lazy coalesce.  */
    series rv_s ("v", std::vector<double> {2.0, 4.0, 6.0});
    CHECK_NEAR (rv_s.rolling_var (2, 1).get_double (1), 2.0, 1e-12);
    CHECK_NEAR (rv_s.expanding_mean ().get_double (2), 4.0, 1e-12);

    data_frame vc_df ({
      series ("k1", std::vector<std::string> {"a", "a", "b"}),
      series ("k2", std::vector<std::string> {"x", "x", "y"})
    });
    auto vc_out = vc_df.value_counts ({"k1", "k2"});
    CHECK_TRUE (vc_out.nrows () == 2);
    CHECK_NEAR (vc_out["count"].get_double (0), 2.0, 1e-12);

    data_frame coal_df ({
      df_primary["x"],
      series ("y", std::vector<double> {100.0, 200.0, 300.0})
    });
    auto coal_res =
      lazy_frame (coal_df)
        .select ({coalesce ({col ("x"), col ("y")}).alias ("z")})
        .collect ();
    CHECK_NEAR (coal_res["z"].get_double (0), 10.0, 1e-12);
    CHECK_NEAR (coal_res["z"].get_double (1), 200.0, 1e-12);

    /* Phase 11: expanding_sum, expanding_min, and expanding_max.  */
    CHECK_NEAR (rv_s.expanding_sum ().get_double (2), 12.0, 1e-12);
    CHECK_NEAR (rv_s.expanding_min ().get_double (2), 2.0, 1e-12);
    CHECK_NEAR (rv_s.expanding_max ().get_double (2), 6.0, 1e-12);

    /* Phase 12: rolling_median, rolling_quantile, add_prefix, add_suffix,
       and lazy expr::clip / expr::fill_forward.  */
    series rq_s ("q", std::vector<double> {10.0, 50.0, 20.0, 40.0});
    CHECK_NEAR (rq_s.rolling_median (3).get_double (2), 20.0, 1e-12);
    CHECK_NEAR (rq_s.rolling_quantile (3, 50.0).get_double (3), 40.0, 1e-12);

    auto pref_df = vc_df.add_prefix ("pre_").add_suffix ("_suf");
    CHECK_TRUE (pref_df.has_column ("pre_k1_suf"));
    CHECK_TRUE (pref_df.has_column ("pre_k2_suf"));

    auto lazy_p12 =
      lazy_frame (coal_df)
        .select ({
          col ("x").fill_forward ().alias ("x_ff"),
          col ("y").clip (150.0, 250.0).alias ("y_clip")
        })
        .collect ();
    CHECK_NEAR (lazy_p12["x_ff"].get_double (1), 10.0, 1e-12);
    CHECK_NEAR (lazy_p12["y_clip"].get_double (0), 150.0, 1e-12);
    CHECK_NEAR (lazy_p12["y_clip"].get_double (2), 250.0, 1e-12);

    /* Phase 13: expanding_var, expanding_std, autocorr, select_dtypes, pop,
       and lazy expr::sqrt / expr::log / expr::exp.  */
    CHECK_NEAR (rv_s.expanding_var (1).get_double (2), 4.0, 1e-12);
    CHECK_NEAR (rv_s.expanding_std (1).get_double (2), 2.0, 1e-12);
    CHECK_NEAR (rv_s.autocorr (1), 1.0, 1e-12);

    data_frame mixed_df ({
      series ("num", std::vector<double> {1.0, 4.0, 9.0}),
      series ("tag", std::vector<std::string> {"a", "b", "c"})
    });
    auto num_only = mixed_df.select_dtypes ({data_type::float64});
    CHECK_TRUE (num_only.ncols () == 1 && num_only.has_column ("num"));

    series popped = mixed_df.pop ("tag");
    CHECK_TRUE (popped.size () == 3 && !mixed_df.has_column ("tag"));

    auto lazy_p13 =
      lazy_frame (num_only)
        .select ({
          col ("num").sqrt ().alias ("rt"),
          col ("num").log ().exp ().alias ("roundtrip")
        })
        .collect ();
    CHECK_NEAR (lazy_p13["rt"].get_double (2), 3.0, 1e-12);
    CHECK_NEAR (lazy_p13["roundtrip"].get_double (1), 4.0, 1e-12);

    /* Phase 14: series::sem, mad, hasnans, data_frame::assign, insert_column,
       and lazy expr::sin / expr::cos / expr::tan.  */
    CHECK_NEAR (rv_s.sem (1), 2.0 / std::sqrt (3.0), 1e-12);
    CHECK_NEAR (rv_s.mad (), 4.0 / 3.0, 1e-12);
    CHECK_TRUE (!rv_s.hasnans () && df_primary["x"].hasnans ());

    auto assigned_df =
      num_only
        .assign ({series ("b", std::vector<double> {10.0, 20.0, 30.0})})
        .insert_column (0, series ("first", std::vector<double> {0.0, 0.0, 0.0}));
    CHECK_TRUE (assigned_df.column_names ()[0] == "first");
    CHECK_TRUE (assigned_df.ncols () == 3);

    data_frame trig_df ({
      series ("theta", std::vector<double> {0.0, 0.5 * constants::pi})
    });
    auto trig_out =
      lazy_frame (trig_df)
        .select ({
          col ("theta").sin ().alias ("s"),
          col ("theta").cos ().alias ("c")
        })
        .collect ();
    CHECK_NEAR (trig_out["s"].get_double (1), 1.0, 1e-12);
    CHECK_NEAR (trig_out["c"].get_double (0), 1.0, 1e-12);

    /* Phase 15: series::ewm_var, ewm_std, data_frame::where_cond, mask_cond. */
    auto ew_v = rv_s.ewm_var (0.5);
    auto ew_s = rv_s.ewm_std (0.5);
    CHECK_TRUE (ew_v.is_valid (1) && ew_s.get_double (2) > 0.0);

    data_frame cond_mask ({
      series ("num", std::vector<double> {1.0, 0.0, 1.0})
    });
    auto wh_df = num_only.where_cond (cond_mask, -1.0);
    auto mk_df = num_only.mask_cond (cond_mask, -1.0);
    CHECK_NEAR (wh_df["num"].get_double (1), -1.0, 1e-12);
    CHECK_NEAR (mk_df["num"].get_double (0), -1.0, 1e-12);
    CHECK_NEAR (mk_df["num"].get_double (1), 4.0, 1e-12);

    /* Phase 16: series::is_monotonic_increasing, is_monotonic_decreasing,
       winsorize, data_frame::nunique, and data_frame::equals.  */
    CHECK_TRUE (rv_s.is_monotonic_increasing ());
    CHECK_TRUE (!rv_s.is_monotonic_decreasing ());
    series dec_s ("d", std::vector<double> {9.0, 5.0, 1.0});
    CHECK_TRUE (dec_s.is_monotonic_decreasing ());

    series win_s ("w", std::vector<double> {-100.0, 2.0, 3.0, 4.0, 100.0});
    auto win_out = win_s.winsorize (25.0, 75.0);
    CHECK_NEAR (win_out.get_double (0), 2.0, 1e-12);
    CHECK_NEAR (win_out.get_double (4), 4.0, 1e-12);

    auto nu_s = vc_df.nunique ();
    CHECK_NEAR (std::get<double> (nu_s.loc ("k1")), 2.0, 1e-12);
    CHECK_TRUE (vc_df.equals (vc_df));
    CHECK_TRUE (!vc_df.equals (num_only));

    /* Bug-audit regression tests: 17-digit double precision in nunique/equals,
       utf8_string monotonicity & fill_forward, lazy filter column retention,
       and lazy string inequality comparison.  */
    series hp_s ("hp", std::vector<double> {1.0000001, 1.0000002});
    CHECK_TRUE (hp_s.nunique () == 2);

    series str_mono ("sm", std::vector<std::string> {"a", "b", "c"});
    CHECK_TRUE (str_mono.is_monotonic_increasing ());
    CHECK_TRUE (!str_mono.is_monotonic_decreasing ());

    validity_bitmap sv_mask (3, true);
    sv_mask.set_valid (1, false);
    series str_null ("sn", std::vector<std::string> {"x", "", "z"}, sv_mask);
    auto str_ff = str_null.fill_forward ();
    CHECK_TRUE (str_ff.is_valid (1)
                && std::get<std::string> (str_ff.iloc (1)) == "x");

    auto lazy_neq =
      lazy_frame (vc_df)
        .filter (col ("k1") != std::string ("a"))
        .collect (true);
    CHECK_TRUE (lazy_neq.ncols () == vc_df.ncols ());
    CHECK_TRUE (lazy_neq.nrows () == 1);

    /* Phase 17: str_slice, str_pad_left, str_pad_right, str_zfill, str_count,
       dt_hour, dt_minute, dt_second, DataFrame to_tensor/from_tensor/transpose/
       shift/diff/pct_change, multi-col sort_by, and lazy is_between/
       rolling_mean/rolling_sum/n_unique/sort.  */
    series s_str ("txt", std::vector<std::string> {"hello_world", "-42"});
    CHECK_TRUE (std::get<std::string> (s_str.str_slice (0, 5).iloc (0))
                == "hello");
    CHECK_TRUE (std::get<std::string> (s_str.str_pad_left (5, '_').iloc (1))
                == "__-42");
    CHECK_TRUE (std::get<std::string> (s_str.str_pad_right (5, '_').iloc (1))
                == "-42__");
    CHECK_TRUE (std::get<std::string> (s_str.str_zfill (5).iloc (1))
                == "-0042");
    CHECK_TRUE (std::get<std::int64_t> (s_str.str_count ("l").iloc (0)) == 3);

    series s_iso ("ts", std::vector<std::string> {"2026-10-30T14:35:09"});
    CHECK_TRUE (std::get<std::int64_t> (s_iso.dt_hour ().iloc (0)) == 14);
    CHECK_TRUE (std::get<std::int64_t> (s_iso.dt_minute ().iloc (0)) == 35);
    CHECK_TRUE (std::get<std::int64_t> (s_iso.dt_second ().iloc (0)) == 9);

    nd_tensor<double> raw_mat = {{1.0, 2.0}, {3.0, 4.0}};
    data_frame t_df = data_frame::from_tensor (raw_mat, {"c1", "c2"});
    CHECK_TRUE (allclose (t_df.to_tensor (), raw_mat, 1e-12));
    auto tr_df = t_df.transpose ();
    CHECK_TRUE (tr_df.nrows () == 2 && tr_df.ncols () == 2);
    CHECK_NEAR (tr_df["0"].get_double (1), 2.0, 1e-12);

    auto df_diff = t_df.diff (1);
    CHECK_NEAR (df_diff["c1"].get_double (1), 2.0, 1e-12);
    auto df_sh = t_df.shift (1);
    CHECK_TRUE (df_sh["c1"].is_null (0));
    CHECK_NEAR (df_sh["c1"].get_double (1), 1.0, 1e-12);
    auto df_pct = t_df.pct_change (1);
    CHECK_NEAR (df_pct["c1"].get_double (1), 2.0, 1e-12);

    data_frame ms_df ({
      series ("g", std::vector<std::string> {"b", "a", "a"}),
      series ("v", std::vector<double> {1.0, 20.0, 10.0})
    });
    auto ms_sorted = ms_df.sort_by (std::vector<std::string> {"g", "v"},
                                    std::vector<bool> {true, true});
    CHECK_TRUE (std::get<std::string> (ms_sorted["g"].iloc (0)) == "a");
    CHECK_NEAR (ms_sorted["v"].get_double (0), 10.0, 1e-12);
    CHECK_NEAR (ms_sorted["v"].get_double (1), 20.0, 1e-12);

    auto lz_res =
      lazy_frame (ms_df)
        .with_columns ({
          col ("v").rolling_sum (2).alias ("rsum"),
          col ("v").rolling_mean (2).alias ("rmean")
        })
        .filter (col ("v").is_between (5.0, 25.0))
        .sort (std::vector<std::string> {"g", "v"},
               std::vector<bool> {true, true})
        .collect (true);
    CHECK_TRUE (lz_res.nrows () == 2);
    CHECK_NEAR (lz_res["v"].get_double (0), 10.0, 1e-12);
    CHECK_NEAR (lz_res["rsum"].get_double (0), 30.0, 1e-12);

    auto lz_nu =
      lazy_frame (ms_df)
        .group_by ("g")
        .agg ({col ("v").n_unique ().alias ("nu")})
        .sort ("g", true)
        .collect (true);
    CHECK_NEAR (lz_nu["nu"].get_double (0), 2.0, 1e-12);

    /* Phase 18: dt_truncate, str_split_part, str_extract_digits,
       data_frame::resample, and lazy dt_year / dt_month / dt_day /
       str_to_uppercase / str_to_lowercase / str_lengths.  */
    {
      series dt_s ("date", std::vector<std::string> {
        "2026-03-15", "2026-03-28", "2026-04-02"
      });
      auto dt_m = dt_s.dt_truncate ("M");
      CHECK_TRUE (std::get<std::string> (dt_m.iloc (0)) == "2026-03-01");
      CHECK_TRUE (std::get<std::string> (dt_m.iloc (2)) == "2026-04-01");

      series sp_s ("code", std::vector<std::string> {
        "usr_104_admin", "dev_99_guest", "ops_7_root"
      });
      CHECK_TRUE (std::get<std::string> (sp_s.str_split_part ('_', 2).iloc (0))
                  == "admin");
      CHECK_TRUE (std::get<std::string> (sp_s.str_extract_digits ().iloc (0))
                  == "104");

      data_frame ts_df ({
        dt_s,
        series ("rev", std::vector<double> {100.0, 150.0, 300.0})
      });
      auto res_m = ts_df.resample ("date", "M", {
        agg_spec ("rev", agg_op::sum, "monthly_rev")
      });
      CHECK_TRUE (res_m.nrows () == 2);
      CHECK_TRUE (std::get<std::string> (res_m["date"].iloc (0)) == "2026-03-01");
      CHECK_NEAR (res_m["monthly_rev"].get_double (0), 250.0, 1e-12);
      CHECK_NEAR (res_m["monthly_rev"].get_double (1), 300.0, 1e-12);

      auto lz_dt_str =
        lazy_frame (ts_df.with_column (sp_s))
          .with_columns ({
            col ("date").dt_year ().alias ("yr"),
            col ("date").dt_month ().alias ("mo"),
            col ("code").str_to_uppercase ().alias ("up"),
            col ("code").str_lengths ().alias ("len")
          })
          .collect (true);
      CHECK_NEAR (lz_dt_str["yr"].get_double (0), 2026.0, 1e-12);
      CHECK_NEAR (lz_dt_str["mo"].get_double (0), 3.0, 1e-12);
      CHECK_TRUE (std::get<std::string> (lz_dt_str["up"].iloc (0))
                  == "USR_104_ADMIN");
      CHECK_NEAR (lz_dt_str["len"].get_double (0), 13.0, 1e-12);
    }

    /* Phase 19: extended .str & .dt methods, table-wide DataFrame window /
       cumulative / reshaping operations, and extended Polars expr &
       lazy_frame operations.  */
    {
      series s_str ("txt", std::vector<std::string> {
        "  hello world  ", "hyperMath", "12345"
      });
      CHECK_TRUE (s_str.str_find ("world").i64_tensor ()[0] == 8);
      CHECK_TRUE (s_str.str_isdigit ().bool_tensor ()[2] == 1);
      CHECK_TRUE (s_str.str_isalpha ().bool_tensor ()[1] == 1);
      CHECK_TRUE (std::get<std::string> (s_str.str_lstrip ().iloc (0))
                  == "hello world  ");
      CHECK_TRUE (std::get<std::string> (s_str.str_rstrip ().iloc (0))
                  == "  hello world");
      CHECK_TRUE (std::get<std::string> (s_str.str_title ().iloc (1))
                  == "Hypermath");
      CHECK_TRUE (std::get<std::string> (
                    s_str.str_removeprefix ("hyper").iloc (1)) == "Math");
      CHECK_TRUE (std::get<std::string> (
                    s_str.str_removesuffix ("Math").iloc (1)) == "hyper");
      CHECK_TRUE (std::get<std::string> (s_str.str_repeat (2).iloc (2))
                  == "1234512345");

      series s_dt ("d", std::vector<std::string> {
        "2024-02-29", "2026-01-01", "2026-12-31"
      });
      CHECK_TRUE (s_dt.dt_is_leap_year ().bool_tensor ()[0] == 1);
      CHECK_TRUE (s_dt.dt_is_leap_year ().bool_tensor ()[1] == 0);
      CHECK_TRUE (s_dt.dt_days_in_month ().i64_tensor ()[0] == 29);
      CHECK_TRUE (s_dt.dt_quarter ().i64_tensor ()[2] == 4);
      CHECK_TRUE (s_dt.dt_is_year_start ().bool_tensor ()[1] == 1);
      CHECK_TRUE (s_dt.dt_is_year_end ().bool_tensor ()[2] == 1);
      CHECK_TRUE (std::get<std::string> (s_dt.dt_add_days (1).iloc (0))
                  == "2024-03-01");

      data_frame w_df ({
        series ("a", std::vector<double> {1.0, 2.0, 3.0}),
        series ("b", std::vector<double> {10.0, 20.0, 30.0})
      });
      CHECK_NEAR (w_df.cum_sum ()["a"].get_double (2), 6.0, 1e-12);
      CHECK_NEAR (w_df.rolling_mean (2)["b"].get_double (2), 25.0, 1e-12);
      CHECK_NEAR (w_df.applymap ([] (double v) { return v * 2.0; })["a"]
                    .get_double (1), 4.0, 1e-12);

      auto cj = w_df.head (2).join_cross (w_df.head (2));
      CHECK_TRUE (cj.nrows () == 4);

      auto idx_df = w_df.set_index ("a").reset_index ("a");
      CHECK_TRUE (idx_df.nrows () == 3 && idx_df.has_column ("a"));

      auto stacked = w_df.stack ();
      auto unstacked = stacked.unstack ();
      CHECK_TRUE (stacked.nrows () == 6 && unstacked.nrows () == 3);

      data_frame w_df2 = w_df;
      auto cmp_df = w_df.compare (w_df2);
      CHECK_TRUE (cmp_df.nrows () == 0);

      auto lz_p19 =
        lazy_frame (w_df)
          .with_columns ({
            col ("a").pow (2.0).alias ("a2"),
            col ("b").shift (1).fill_null (0.0).alias ("b_prev"),
            col ("a").is_in (std::vector<double> {1.0, 3.0}).alias ("in_set")
          })
          .reverse ()
          .collect (true);
      CHECK_NEAR (lz_p19["a2"].get_double (0), 9.0, 1e-12);
      CHECK_NEAR (lz_p19["b_prev"].get_double (0), 20.0, 1e-12);
      CHECK_TRUE (lz_p19["in_set"].bool_tensor ()[0] == 1);

      auto lz_cat = concat_lazy ({lazy_frame (w_df), lazy_frame (w_df)})
                      .tail (2)
                      .collect (true);
      CHECK_TRUE (lz_cat.nrows () == 2);
    }

    /* Phase 20: Series regex/encoding/JSON/temporal/analytical methods,
       DataFrame query/eval_expr/sql_query/partition_by/group_by_dynamic/
       upsample/to_json/read_json/to_html/axis reductions, and LazyFrame +
       sql_context.  */
    {
      series s_re ("s", std::vector<std::string> {"id:42", "id:99"});
      CHECK_TRUE (std::get<std::string> (s_re.str_extract ("id:(\\d+)").iloc (0))
                  == "42");
      CHECK_TRUE (std::get<std::string> (
                    s_re.str_replace_regex ("\\d+", "X").iloc (1)) == "id:X");
      CHECK_TRUE (s_re.str_match ("^id:").bool_tensor ()[0] == 1);
      CHECK_TRUE (s_re.str_count_regex ("\\d").i64_tensor ()[0] == 2);
      auto parts = s_re.str_partition (":");
      CHECK_TRUE (std::get<std::string> (parts[0].iloc (0)) == "id"
                  && std::get<std::string> (parts[2].iloc (0)) == "42");
      CHECK_TRUE (std::get<std::string> (
                    s_re.str_encode_hex ().str_decode_hex ().iloc (0)) == "id:42");
      CHECK_TRUE (std::get<std::string> (
                    s_re.str_encode_base64 ().str_decode_base64 ().iloc (1))
                  == "id:99");

      series s_js ("j", std::vector<std::string> {
        "{\"user\":\"alice\",\"score\":10}",
        "{\"user\":\"bob\",\"score\":20}"
      });
      CHECK_TRUE (std::get<std::string> (s_js.str_json_extract ("user").iloc (0))
                  == "alice");
      CHECK_TRUE (std::get<std::string> (s_js.str_json_extract ("score").iloc (1))
                  == "20");

      series s_dt ("d", std::vector<std::string> {"2026-01-01", "2026-01-08"});
      CHECK_TRUE (std::get<std::string> (s_dt.dt_strftime ("%Y/%m/%d").iloc (0))
                  == "2026/01/01");
      CHECK_TRUE (std::get<std::string> (s_dt.dt_day_name ().iloc (0))
                  == "Thursday");
      CHECK_TRUE (std::get<std::string> (s_dt.dt_month_name ().iloc (0))
                  == "January");
      CHECK_TRUE (s_dt.dt_days_in_year ().i64_tensor ()[0] == 365);
      auto ep_s = s_dt.dt_epoch_seconds ();
      auto from_ep = series::dt_from_epoch_seconds ("d", ep_s.i64_tensor ());
      CHECK_TRUE (std::get<std::string> (from_ep.iloc (0)).substr (0, 10)
                  == "2026-01-01");
      series s_dt2 ("d2", std::vector<std::string> {"2026-01-08", "2026-01-15"});
      CHECK_TRUE (s_dt.dt_business_day_count (s_dt2).i64_tensor ()[0] == 5);

      series s_an ("v", std::vector<double> {10.0, 10.0, 30.0, 20.0});
      CHECK_NEAR (s_an.pct_rank ().get_double (2), 1.0, 1e-12);
      CHECK_TRUE (s_an.rle_id ().i64_tensor ()[0] == 0
                  && s_an.rle_id ().i64_tensor ()[1] == 0
                  && s_an.rle_id ().i64_tensor ()[2] == 1);
      CHECK_TRUE (s_an.peak_max ().bool_tensor ()[2] == 1);
      CHECK_TRUE (s_an.hash ().size () == 4);
      CHECK_NEAR (s_an.rolling_corr (s_an, 3).get_double (2), 1.0, 1e-9);
      CHECK_NEAR (s_an.rolling_apply (2, [] (const nd_tensor<double> &w) {
        return tensor::sum (w);
      }).get_double (1), 20.0, 1e-12);
      CHECK_TRUE (s_an.search_sorted (15.0) == 2);

      data_frame q_df ({
        series ("t", std::vector<double> {0.0, 2.0, 4.0}),
        series ("x", std::vector<double> {10.0, 20.0, 30.0}),
        series ("y", std::vector<double> {1.0, 2.0, 3.0})
      });
      CHECK_TRUE (q_df.query ("x >= 20").nrows () == 2);
      CHECK_NEAR (q_df.eval_expr ("x + y").get_double (1), 22.0, 1e-12);
      CHECK_TRUE (q_df.sql_query ("SELECT x, y FROM self WHERE x > 10 ORDER BY x DESC LIMIT 1")
                    .nrows () == 1);
      CHECK_TRUE (q_df.partition_by ("x").size () == 3);
      CHECK_TRUE (q_df.group_by_dynamic ("t", 2, "x", "sum").nrows () == 3);
      CHECK_TRUE (q_df.upsample ("t", 1).interpolate ().nrows () == 5);

      const std::string js_out = q_df.to_json ();
      const data_frame js_in = data_frame::read_json (js_out);
      CHECK_TRUE (js_in.nrows () == 3 && js_in.ncols () == 3);
      CHECK_TRUE (q_df.to_html ().find ("<table>") != std::string::npos);
      CHECK_NEAR (std::get<double> (q_df.sum (0).loc ("x")), 60.0, 1e-12);
      CHECK_NEAR (q_df.sum (1).get_double (2), 37.0, 1e-12);
      CHECK_NEAR (std::get<double> (q_df.mean (0).loc ("x")), 20.0, 1e-12);

      sql_context ctx;
      ctx.register_table ("metrics", lazy_frame (q_df));
      auto sql_res = ctx.execute ("SELECT x, y FROM metrics WHERE x >= 20 ORDER BY x ASC");
      CHECK_TRUE (sql_res.nrows () == 2);

      auto lz_p20 =
        lazy_frame (q_df)
          .with_row_index ("row_id")
          .with_columns ({
            col ("x").pct_rank ().alias ("pr"),
            col ("x").rle_id ().alias ("rle"),
            col ("y").log10 ().alias ("ly")
          })
          .collect (true);
      CHECK_TRUE (lz_p20.has_column ("row_id") && lz_p20.has_column ("pr"));

      /* Extended Pure C ABI verification (Phase 20).  */
      CHECK_NEAR (hm_special_gamma (5.0), 24.0, 1e-10);
      CHECK_NEAR (hm_special_erf (0.0), 0.0, 1e-12);
      CHECK_NEAR (hm_special_bessel_j0 (0.0), 1.0, 1e-12);
      CHECK_NEAR (hm_special_lambertw (constants::e), 1.0, 1e-10);
      CHECK_NEAR (hm_constants_convert_temperature (100.0, "Celsius", "Kelvin"),
                  373.15, 1e-9);

      hm_tensor_t *qmc_h = hm_qmc_halton (8, 2);
      CHECK_TRUE (qmc_h != nullptr && hm_tensor_size (qmc_h) == 16);
      hm_tensor_free (qmc_h);

      const size_t svd_shape[2] = {2, 2};
      const double svd_data[4] = {3.0, 0.0, 0.0, 5.0};
      hm_tensor_t *svd_in = hm_tensor_from_data (svd_shape, 2, svd_data);
      hm_tensor_t *svals = hm_linalg_svdvals (svd_in);
      CHECK_TRUE (svals != nullptr && hm_tensor_size (svals) == 2);
      CHECK_NEAR (hm_tensor_get_flat (svals, 0), 5.0, 1e-10);
      CHECK_NEAR (hm_tensor_get_flat (svals, 1), 3.0, 1e-10);
      hm_tensor_free (svals);
      hm_tensor_free (svd_in);

      const size_t fft_shape[1] = {4};
      const double fft_data[4] = {1.0, 1.0, 1.0, 1.0};
      hm_tensor_t *fft_in = hm_tensor_from_data (fft_shape, 1, fft_data);
      hm_tensor_t *fft_mag = hm_fft_rfft_abs (fft_in);
      CHECK_TRUE (fft_mag != nullptr && hm_tensor_size (fft_mag) == 3);
      CHECK_NEAR (hm_tensor_get_flat (fft_mag, 0), 4.0, 1e-10);
      CHECK_NEAR (hm_tensor_get_flat (fft_mag, 1), 0.0, 1e-10);
      hm_tensor_free (fft_mag);
      hm_tensor_free (fft_in);

      const std::string c_sql_csv = "/tmp/test_hm_c_sql.csv";
      q_df.write_csv (c_sql_csv);
      hm_dataframe_t *c_df = hm_dataframe_read_csv (c_sql_csv.c_str ());
      hm_dataframe_t *c_sql_res = hm_dataframe_sql_query (
        c_df, "SELECT x, y FROM self WHERE x >= 20 ORDER BY x ASC");
      CHECK_TRUE (c_sql_res != nullptr && hm_dataframe_nrows (c_sql_res) == 2);
      CHECK_NEAR (hm_dataframe_col_sum (c_sql_res, "x"), 50.0, 1e-12);
      hm_dataframe_free (c_sql_res);
      hm_dataframe_free (c_df);
    }
  }

  std::cout << "  -> Suite 3 passed.\n";
  return 0;
}
