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

#include <darshana/stat.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

namespace darshana::inline v1::stat
{

namespace
{

/* Evaluates linear interpolation quantile p in [0, 1] on a sorted vector.  */
auto
sorted_quantile (const std::vector<float> &sorted, double p) noexcept -> double
{
  if (sorted.empty ())
    {
      return 0.0;
    }
  if (sorted.size () == 1U)
    {
      return static_cast<double> (sorted.front ());
    }
  const double idx = p * static_cast<double> (sorted.size () - 1U);
  const auto lo = static_cast<std::size_t> (std::floor (idx));
  const auto hi = static_cast<std::size_t> (std::ceil (idx));
  const double frac = idx - static_cast<double> (lo);
  return static_cast<double> (sorted[lo]) * (1.0 - frac)
         + static_cast<double> (sorted[hi]) * frac;
}

/* Approximates normal critical value z for a two-sided confidence level.  */
auto
normal_critical_z (float confidence_level) noexcept -> double
{
  if (confidence_level >= 0.99F)
    {
      return 2.575829;
    }
  if (confidence_level >= 0.95F)
    {
      return 1.959964;
    }
  if (confidence_level >= 0.90F)
    {
      return 1.644854;
    }
  return 1.281552;
}

} /* namespace */

/* Computes descriptive statistics and quartiles for a 1D sample.  */
auto
compute_summary (std::span<const float> values) noexcept
    -> Result<SummaryStats>
{
  if (values.empty ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::EmptyDataset, "values",
          "stat::compute_summary requires a non-empty sample span"));
    }
  for (const float v : values)
    {
      if (!std::isfinite (v))
        {
          return std::unexpected (Error::data (
              DataErrorCode::UnsupportedDataType, "values",
              "stat::compute_summary requires finite floating-point values"));
        }
    }
  try
    {
      std::vector<float> sorted (values.begin (), values.end ());
      std::ranges::sort (sorted);

      double sum = 0.0;
      for (const float v : sorted)
        {
          sum += static_cast<double> (v);
        }
      const double n = static_cast<double> (sorted.size ());
      const double mean = sum / n;

      double sq_diff = 0.0;
      for (const float v : sorted)
        {
          const double d = static_cast<double> (v) - mean;
          sq_diff += d * d;
        }
      const double var = (sorted.size () > 1U) ? (sq_diff / (n - 1.0)) : 0.0;

      SummaryStats out;
      out.mean = mean;
      out.variance = var;
      out.std_dev = std::sqrt (var);
      out.min_val = static_cast<double> (sorted.front ());
      out.max_val = static_cast<double> (sorted.back ());
      out.q1 = sorted_quantile (sorted, 0.25);
      out.median = sorted_quantile (sorted, 0.50);
      out.q3 = sorted_quantile (sorted, 0.75);
      out.iqr = out.q3 - out.q1;
      return out;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::compute_summary"));
    }
}

/* Fits an ordinary least squares (OLS) linear regression model.  */
auto
compute_ols (std::span<const float> x,
             std::span<const float> y) noexcept -> Result<RegressionFit>
{
  if (x.size () < 2U || y.size () < 2U)
    {
      return std::unexpected (Error::data (
          DataErrorCode::EmptyDataset, "x/y",
          "stat::compute_ols requires at least two (x, y) observations"));
    }
  if (x.size () != y.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "x/y",
          "stat::compute_ols requires x and y spans of equal length"));
    }
  const std::size_t n = x.size ();
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (std::size_t i = 0U; i < n; ++i)
    {
      if (!std::isfinite (x[i]) || !std::isfinite (y[i]))
        {
          return std::unexpected (Error::data (
              DataErrorCode::UnsupportedDataType, "x/y",
              "stat::compute_ols requires finite floating-point inputs"));
        }
      sum_x += static_cast<double> (x[i]);
      sum_y += static_cast<double> (y[i]);
    }
  const double dn = static_cast<double> (n);
  const double mean_x = sum_x / dn;
  const double mean_y = sum_y / dn;

  double sxx = 0.0;
  double sxy = 0.0;
  double syy = 0.0;
  for (std::size_t i = 0U; i < n; ++i)
    {
      const double dx = static_cast<double> (x[i]) - mean_x;
      const double dy = static_cast<double> (y[i]) - mean_y;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
  if (sxx <= 1e-15)
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidScaleDomain, "x",
          "stat::compute_ols predictor variable x has zero variance"));
    }
  RegressionFit fit;
  fit.slope = sxy / sxx;
  fit.intercept = mean_y - fit.slope * mean_x;
  fit.r_squared
      = (syy > 1e-15) ? std::clamp ((sxy * sxy) / (sxx * syy), 0.0, 1.0) : 1.0;

  double ssr = 0.0;
  for (std::size_t i = 0U; i < n; ++i)
    {
      const double y_hat
          = fit.intercept + fit.slope * static_cast<double> (x[i]);
      const double err = static_cast<double> (y[i]) - y_hat;
      ssr += err * err;
    }
  fit.std_err
      = (n > 2U) ? std::sqrt (ssr / static_cast<double> (n - 2U)) : 0.0;
  return fit;
}

/* Builds a declarative histogram Spec from a 1D sample.  */
auto
histplot (std::span<const float> data, std::size_t bins,
          std::string_view title) noexcept -> Result<Spec>
{
  if (bins == 0U)
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "bins",
                       "stat::histplot requires at least one histogram bin"));
    }
  auto sum_res = compute_summary (data);
  if (!sum_res.has_value ())
    {
      return std::unexpected (sum_res.error ());
    }
  try
    {
      double lo = sum_res->min_val;
      double hi = sum_res->max_val;
      if (lo == hi)
        {
          lo -= 0.5;
          hi += 0.5;
        }
      const double bin_w = (hi - lo) / static_cast<double> (bins);

      LayerSpec layer;
      layer.mark = MarkType::Histogram;
      layer.encoding.label = "Histogram";
      layer.aesthetics.fill_color = ColorRgba::from_hex (0x1F77B4U, 195U);
      layer.aesthetics.stroke_color = ColorRgba::from_hex (0x0D3B66U, 255U);
      layer.inline_x.resize (bins);
      layer.inline_y.assign (bins, 0.0F);

      for (std::size_t b = 0U; b < bins; ++b)
        {
          layer.inline_x[b] = static_cast<float> (
              lo + (static_cast<double> (b) + 0.5) * bin_w);
        }
      float max_count = 1.0F;
      for (const float v : data)
        {
          const double rel = (static_cast<double> (v) - lo) / bin_w;
          auto idx = static_cast<std::size_t> (std::clamp (
              std::floor (rel), 0.0, static_cast<double> (bins - 1U)));
          layer.inline_y[idx] += 1.0F;
          max_count = std::max (max_count, layer.inline_y[idx]);
        }

      Spec spec;
      spec.title = std::string (title);
      spec.x_label = "value";
      spec.y_label = "count";
      spec.x_scale.domain_min = lo;
      spec.x_scale.domain_max = hi;
      spec.y_scale.domain_min = 0.0;
      spec.y_scale.domain_max = static_cast<double> (max_count) * 1.1;
      spec.layers.push_back (std::move (layer));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::histplot"));
    }
}

/* Builds a Gaussian Kernel Density Estimate (KDE) Spec from a 1D sample.  */
auto
kdeplot (std::span<const float> data, std::size_t grid_points, float bandwidth,
         std::string_view title) noexcept -> Result<Spec>
{
  if (grid_points < 2U)
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidSpec, "grid_points",
          "stat::kdeplot requires at least 2 evaluation grid points"));
    }
  auto sum_res = compute_summary (data);
  if (!sum_res.has_value ())
    {
      return std::unexpected (sum_res.error ());
    }
  try
    {
      const double n = static_cast<double> (data.size ());
      double bw = static_cast<double> (bandwidth);
      if (bw <= 0.0)
        {
          const double sigma = std::max (sum_res->std_dev, 1e-3);
          const double iqr_scale
              = (sum_res->iqr > 0.0) ? (sum_res->iqr / 1.34) : sigma;
          const double spread = std::min (sigma, iqr_scale);
          bw = 0.9 * std::max (spread, 1e-3) * std::pow (n, -0.2);
        }
      const double lo = sum_res->min_val - 3.0 * bw;
      const double hi = sum_res->max_val + 3.0 * bw;
      const double step = (hi - lo) / static_cast<double> (grid_points - 1U);
      const double inv_norm
          = 1.0 / (n * bw * std::sqrt (2.0 * std::numbers::pi));

      LayerSpec layer;
      layer.mark = MarkType::Kde;
      layer.encoding.label = "KDE";
      layer.aesthetics.stroke_color = ColorRgba::from_hex (0xD62728U, 255U);
      layer.aesthetics.fill_color = ColorRgba::from_hex (0xD62728U, 90U);
      layer.aesthetics.stroke_width = 2.0F;
      layer.inline_x.resize (grid_points);
      layer.inline_y.resize (grid_points);

      double max_density = 1e-6;
      for (std::size_t g = 0U; g < grid_points; ++g)
        {
          const double gx = lo + static_cast<double> (g) * step;
          double acc = 0.0;
          for (const float sample : data)
            {
              const double u = (gx - static_cast<double> (sample)) / bw;
              acc += std::exp (-0.5 * u * u);
            }
          const double density = acc * inv_norm;
          layer.inline_x[g] = static_cast<float> (gx);
          layer.inline_y[g] = static_cast<float> (density);
          max_density = std::max (max_density, density);
        }

      Spec spec;
      spec.title = std::string (title);
      spec.x_label = "value";
      spec.y_label = "density";
      spec.x_scale.domain_min = lo;
      spec.x_scale.domain_max = hi;
      spec.y_scale.domain_min = 0.0;
      spec.y_scale.domain_max = max_density * 1.1;
      spec.layers.push_back (std::move (layer));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::kdeplot"));
    }
}

/* Builds a composite distribution Spec (normalized histogram + KDE).  */
auto
distplot (std::span<const float> data, std::size_t bins,
          std::string_view title) noexcept -> Result<Spec>
{
  auto hist_res = histplot (data, bins, title);
  if (!hist_res.has_value ())
    {
      return std::unexpected (hist_res.error ());
    }
  auto kde_res = kdeplot (data, 128U, 0.0F, title);
  if (!kde_res.has_value ())
    {
      return std::unexpected (kde_res.error ());
    }
  try
    {
      Spec spec = std::move (*hist_res);
      spec.y_label = "density";
      auto &hist_layer = spec.layers.front ();
      const double bin_w = (spec.x_scale.domain_max - spec.x_scale.domain_min)
                           / static_cast<double> (bins);
      const double norm
          = static_cast<double> (data.size ()) * std::max (bin_w, 1e-9);
      double max_y = kde_res->y_scale.domain_max;
      for (float &c : hist_layer.inline_y)
        {
          c = static_cast<float> (static_cast<double> (c) / norm);
          max_y = std::max (max_y, static_cast<double> (c) * 1.1);
        }
      spec.x_scale = kde_res->x_scale;
      spec.y_scale.domain_min = 0.0;
      spec.y_scale.domain_max = max_y;
      spec.layers.push_back (std::move (kde_res->layers.front ()));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::distplot"));
    }
}

/* Builds an Empirical Cumulative Distribution Function (ECDF) Spec.  */
auto
ecdfplot (std::span<const float> data,
          std::string_view title) noexcept -> Result<Spec>
{
  auto sum_res = compute_summary (data);
  if (!sum_res.has_value ())
    {
      return std::unexpected (sum_res.error ());
    }
  try
    {
      std::vector<float> sorted (data.begin (), data.end ());
      std::ranges::sort (sorted);

      LayerSpec layer;
      layer.mark = MarkType::Ecdf;
      layer.encoding.label = "ECDF";
      layer.aesthetics.stroke_color = ColorRgba::from_hex (0x2CA02CU, 255U);
      layer.aesthetics.stroke_width = 2.0F;
      layer.inline_x = std::move (sorted);
      layer.inline_y.resize (data.size ());
      const float inv_n = 1.0F / static_cast<float> (data.size ());
      for (std::size_t i = 0U; i < data.size (); ++i)
        {
          layer.inline_y[i] = static_cast<float> (i + 1U) * inv_n;
        }

      Spec spec;
      spec.title = std::string (title);
      spec.x_label = "value";
      spec.y_label = "cumulative probability";
      spec.x_scale.domain_min = sum_res->min_val;
      spec.x_scale.domain_max = (sum_res->max_val > sum_res->min_val)
                                    ? sum_res->max_val
                                    : (sum_res->min_val + 1.0);
      spec.y_scale.domain_min = 0.0;
      spec.y_scale.domain_max = 1.0;
      spec.layers.push_back (std::move (layer));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::ecdfplot"));
    }
}

/* Builds a regression plot Spec with scatter, OLS line, and confidence band.
 */
auto
regplot (std::span<const float> x, std::span<const float> y,
         float confidence_level,
         std::string_view title) noexcept -> Result<Spec>
{
  if (confidence_level <= 0.0F || confidence_level >= 1.0F)
    {
      return std::unexpected (Error::data (
          DataErrorCode::InvalidSpec, "confidence_level",
          "stat::regplot confidence_level must lie strictly in (0, 1)"));
    }
  auto fit_res = compute_ols (x, y);
  if (!fit_res.has_value ())
    {
      return std::unexpected (fit_res.error ());
    }
  auto sx_res = compute_summary (x);
  auto sy_res = compute_summary (y);
  if (!sx_res.has_value () || !sy_res.has_value ())
    {
      return std::unexpected (
          Error::data (DataErrorCode::InvalidSpec, "x/y",
                       "Failed to compute summary statistics for regplot"));
    }
  try
    {
      const double n = static_cast<double> (x.size ());
      const double sxx = sx_res->variance * (n - 1.0);
      const double z = normal_critical_z (confidence_level);

      constexpr std::size_t kGrid = 50U;
      LayerSpec band;
      band.mark = MarkType::ConfidenceBand;
      band.encoding.label = "Confidence Band";
      band.aesthetics.fill_color = ColorRgba::from_hex (0xFF7F0EU, 75U);
      band.aesthetics.stroke_width = 0.5F;
      band.inline_x.resize (kGrid);
      band.inline_y.resize (kGrid);
      band.inline_y2.resize (kGrid);

      LayerSpec line;
      line.mark = MarkType::RegressionLine;
      line.encoding.label = "OLS Fit";
      line.aesthetics.stroke_color = ColorRgba::from_hex (0xD62728U, 255U);
      line.aesthetics.stroke_width = 2.2F;
      line.inline_x.resize (kGrid);
      line.inline_y.resize (kGrid);

      const double x_lo = sx_res->min_val;
      const double x_hi = sx_res->max_val;
      const double step = (x_hi - x_lo) / static_cast<double> (kGrid - 1U);
      double y_min = sy_res->min_val;
      double y_max = sy_res->max_val;

      for (std::size_t i = 0U; i < kGrid; ++i)
        {
          const double gx = x_lo + static_cast<double> (i) * step;
          const double gy = fit_res->intercept + fit_res->slope * gx;
          const double dx = gx - sx_res->mean;
          const double se_fit
              = fit_res->std_err
                * std::sqrt ((1.0 / n) + (dx * dx) / std::max (sxx, 1e-12));
          const double margin = z * se_fit;

          band.inline_x[i] = static_cast<float> (gx);
          band.inline_y[i] = static_cast<float> (gy - margin);
          band.inline_y2[i] = static_cast<float> (gy + margin);
          line.inline_x[i] = static_cast<float> (gx);
          line.inline_y[i] = static_cast<float> (gy);
          y_min = std::min (y_min, gy - margin);
          y_max = std::max (y_max, gy + margin);
        }

      LayerSpec scatter;
      scatter.mark = MarkType::Scatter;
      scatter.encoding.label = "Observations";
      scatter.aesthetics.fill_color = ColorRgba::from_hex (0x1F77B4U, 210U);
      scatter.aesthetics.marker_size = 4.5F;
      scatter.inline_x.assign (x.begin (), x.end ());
      scatter.inline_y.assign (y.begin (), y.end ());

      Spec spec;
      spec.title = std::string (title);
      spec.x_scale.domain_min = x_lo;
      spec.x_scale.domain_max = x_hi;
      spec.y_scale.domain_min = y_min;
      spec.y_scale.domain_max = (y_max > y_min) ? y_max : (y_min + 1.0);
      spec.layers.push_back (std::move (band));
      spec.layers.push_back (std::move (scatter));
      spec.layers.push_back (std::move (line));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::regplot"));
    }
}

/* Builds a categorical box-and-whisker Spec from grouped observations.  */
auto
boxplot (std::span<const std::string> categories,
         std::span<const std::vector<float>> groups,
         std::string_view title) noexcept -> Result<Spec>
{
  if (categories.empty () || groups.empty ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::EmptyDataset, "categories/groups",
          "stat::boxplot requires non-empty category and group spans"));
    }
  if (categories.size () != groups.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "categories/groups",
          "stat::boxplot requires equal numbers of categories and groups"));
    }
  try
    {
      LayerSpec box;
      box.mark = MarkType::Box;
      box.encoding.label = "Box";
      box.aesthetics.fill_color = ColorRgba::from_hex (0x1F77B4U, 180U);
      box.aesthetics.stroke_color = ColorRgba::from_hex (0x111111U, 255U);

      double global_min = 1e30;
      double global_max = -1e30;
      for (std::size_t i = 0U; i < groups.size (); ++i)
        {
          auto s_res = compute_summary (groups[i]);
          if (!s_res.has_value ())
            {
              return std::unexpected (s_res.error ());
            }
          const double whisker_lo
              = std::max (s_res->min_val, s_res->q1 - 1.5 * s_res->iqr);
          const double whisker_hi
              = std::min (s_res->max_val, s_res->q3 + 1.5 * s_res->iqr);
          box.inline_x.push_back (static_cast<float> (i));
          box.inline_y.push_back (static_cast<float> (s_res->median));
          box.inline_q1.push_back (static_cast<float> (s_res->q1));
          box.inline_q3.push_back (static_cast<float> (s_res->q3));
          box.inline_y2.push_back (static_cast<float> (whisker_hi));
          global_min = std::min (global_min, whisker_lo);
          global_max = std::max (global_max, whisker_hi);
        }

      Spec spec;
      spec.title = std::string (title);
      spec.x_label = "category";
      spec.y_label = "value";
      spec.x_scale.type = ScaleType::Categorical;
      spec.x_scale.domain_min = -0.5;
      spec.x_scale.domain_max = static_cast<double> (categories.size ()) - 0.5;
      spec.x_scale.categories.assign (categories.begin (), categories.end ());
      spec.y_scale.domain_min = global_min;
      spec.y_scale.domain_max
          = (global_max > global_min) ? global_max : (global_min + 1.0);
      spec.layers.push_back (std::move (box));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::boxplot"));
    }
}

/* Builds a categorical violin plot Spec with KDE envelopes and IQR.  */
auto
violinplot (std::span<const std::string> categories,
            std::span<const std::vector<float>> groups,
            std::string_view title) noexcept -> Result<Spec>
{
  auto box_res = boxplot (categories, groups, title);
  if (!box_res.has_value ())
    {
      return std::unexpected (box_res.error ());
    }
  Spec spec = std::move (*box_res);
  spec.layers.front ().mark = MarkType::Violin;
  spec.layers.front ().encoding.label = "Violin";
  spec.layers.front ().aesthetics.fill_color
      = ColorRgba::from_hex (0x9467BDU, 185U);
  return spec;
}

/* Builds a categorical bar plot Spec with group means and error bars.  */
auto
barplot (std::span<const std::string> categories,
         std::span<const std::vector<float>> groups,
         std::string_view title) noexcept -> Result<Spec>
{
  if (categories.empty () || groups.empty ()
      || categories.size () != groups.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "categories/groups",
          "stat::barplot requires matching non-empty categories and groups"));
    }
  try
    {
      LayerSpec bars;
      bars.mark = MarkType::Bar;
      bars.encoding.label = "Mean";
      bars.aesthetics.fill_color = ColorRgba::from_hex (0x2CA02CU, 200U);

      LayerSpec errs;
      errs.mark = MarkType::ErrorBar;
      errs.encoding.label = "95% CI";
      errs.aesthetics.stroke_color = ColorRgba::from_hex (0x111111U, 255U);

      double min_y = 0.0;
      double max_y = 1.0;
      for (std::size_t i = 0U; i < groups.size (); ++i)
        {
          auto s_res = compute_summary (groups[i]);
          if (!s_res.has_value ())
            {
              return std::unexpected (s_res.error ());
            }
          const double sem
              = s_res->std_dev
                / std::sqrt (static_cast<double> (groups[i].size ()));
          const double ci = 1.96 * sem;
          bars.inline_x.push_back (static_cast<float> (i));
          bars.inline_y.push_back (static_cast<float> (s_res->mean));
          errs.inline_x.push_back (static_cast<float> (i));
          errs.inline_y.push_back (static_cast<float> (s_res->mean - ci));
          errs.inline_y2.push_back (static_cast<float> (s_res->mean + ci));
          min_y = std::min (min_y, s_res->mean - ci);
          max_y = std::max (max_y, s_res->mean + ci);
        }

      Spec spec;
      spec.title = std::string (title);
      spec.x_scale.type = ScaleType::Categorical;
      spec.x_scale.domain_min = -0.5;
      spec.x_scale.domain_max = static_cast<double> (categories.size ()) - 0.5;
      spec.x_scale.categories.assign (categories.begin (), categories.end ());
      spec.y_scale.domain_min = min_y;
      spec.y_scale.domain_max
          = (max_y > min_y) ? (max_y * 1.1) : (min_y + 1.0);
      spec.layers.push_back (std::move (bars));
      spec.layers.push_back (std::move (errs));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::barplot"));
    }
}

/* Builds a 2D color-mapped matrix heatmap Spec.  */
auto
heatmap (std::span<const float> matrix, std::size_t rows, std::size_t cols,
         std::string_view title) noexcept -> Result<Spec>
{
  if (rows == 0U || cols == 0U || matrix.size () != rows * cols)
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "matrix",
          "stat::heatmap requires non-zero rows*cols equal to matrix.size()"));
    }
  for (const float v : matrix)
    {
      if (!std::isfinite (v))
        {
          return std::unexpected (
              Error::data (DataErrorCode::UnsupportedDataType, "matrix",
                           "stat::heatmap matrix contains non-finite values"));
        }
    }
  try
    {
      LayerSpec hm;
      hm.mark = MarkType::Heatmap;
      hm.encoding.label = "Heatmap";
      hm.matrix_rows = rows;
      hm.matrix_cols = cols;
      hm.matrix_values.assign (matrix.begin (), matrix.end ());

      Spec spec;
      spec.title = std::string (title);
      spec.x_label = "column";
      spec.y_label = "row";
      spec.x_scale.domain_min = 0.0;
      spec.x_scale.domain_max = static_cast<double> (cols);
      spec.y_scale.domain_min = 0.0;
      spec.y_scale.domain_max = static_cast<double> (rows);
      spec.layers.push_back (std::move (hm));
      return spec;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::heatmap"));
    }
}

/* Computes a Pearson correlation matrix and builds a heatmap Spec.  */
auto
corrplot (std::span<const std::vector<float>> variables,
          std::span<const std::string> names,
          std::string_view title) noexcept -> Result<Spec>
{
  if (variables.size () < 2U || variables.size () != names.size ())
    {
      return std::unexpected (Error::data (
          DataErrorCode::DimensionMismatch, "variables/names",
          "stat::corrplot requires at least 2 variables matching names"));
    }
  const std::size_t k = variables.size ();
  const std::size_t n = variables.front ().size ();
  if (n < 2U)
    {
      return std::unexpected (Error::data (
          DataErrorCode::EmptyDataset, "variables",
          "stat::corrplot requires at least 2 observations per variable"));
    }
  for (const auto &v : variables)
    {
      if (v.size () != n)
        {
          return std::unexpected (Error::data (
              DataErrorCode::DimensionMismatch, "variables",
              "stat::corrplot requires all variable vectors to have equal "
              "length"));
        }
    }
  try
    {
      std::vector<float> corr (k * k, 1.0F);
      for (std::size_t i = 0U; i < k; ++i)
        {
          for (std::size_t j = i + 1U; j < k; ++j)
            {
              auto fit = compute_ols (variables[i], variables[j]);
              float r = 0.0F;
              if (fit.has_value ())
                {
                  const double mag = std::sqrt (fit->r_squared);
                  r = static_cast<float> ((fit->slope >= 0.0) ? mag : -mag);
                }
              corr[i * k + j] = r;
              corr[j * k + i] = r;
            }
        }
      auto spec_res = heatmap (corr, k, k, title);
      if (!spec_res.has_value ())
        {
          return std::unexpected (spec_res.error ());
        }
      spec_res->x_scale.categories.assign (names.begin (), names.end ());
      spec_res->y_scale.categories.assign (names.begin (), names.end ());
      return spec_res;
    }
  catch (...)
    {
      return std::unexpected (
          Error::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                            "Out of memory in stat::corrplot"));
    }
}

} /* namespace darshana::inline v1::stat */
