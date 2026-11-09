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

#ifndef DARSHANA_STAT_HPP
#define DARSHANA_STAT_HPP

#include <darshana/error.hpp>
#include <darshana/spec.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace darshana::inline v1::stat
{

/**
 * @brief Five-number summary and moments for a univariate numerical sample.
 * @pre None.
 * @post Holds sample mean, variance, standard deviation, quartiles, and
 *       extrema.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct SummaryStats
{
  double mean = 0.0;
  double variance = 0.0;
  double std_dev = 0.0;
  double median = 0.0;
  double q1 = 0.0;
  double q3 = 0.0;
  double iqr = 0.0;
  double min_val = 0.0;
  double max_val = 0.0;
};

/**
 * @brief Ordinary least squares (OLS) linear regression model parameters.
 * @pre None.
 * @post Holds slope, intercept, coefficient of determination (R^2), and
 *       standard error of the estimate.
 * @note Thread-safety: Value type; safe for concurrent reads.
 */
struct RegressionFit
{
  double slope = 0.0;
  double intercept = 0.0;
  double r_squared = 0.0;
  double std_err = 0.0;
};

/**
 * @brief Computes descriptive statistics and quartiles for a 1D sample.
 * @param values Non-empty span of finite float samples.
 * @pre !values.empty().
 * @post State is unchanged; returns computed SummaryStats.
 * @returns Result<SummaryStats> or DataError on empty/non-finite input.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto compute_summary (std::span<const float> values) noexcept
    -> Result<SummaryStats>;

/**
 * @brief Fits an ordinary least squares (OLS) linear regression model.
 * @param x Predictor variable samples.
 * @param y Response variable samples.
 * @pre x.size() >= 2 and x.size() == y.size().
 * @post Returns fitted slope, intercept, R^2, and residual standard error.
 * @returns Result<RegressionFit> or DataError on invalid input.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto
compute_ols (std::span<const float> x,
             std::span<const float> y) noexcept -> Result<RegressionFit>;

/**
 * @brief Builds a declarative histogram Spec from a 1D sample.
 * @param data Non-empty span of numerical samples.
 * @param bins Number of equal-width histogram bins (> 0).
 * @param title Plot title string.
 * @pre !data.empty() and bins > 0.
 * @post Returned Spec contains a validated Histogram layer with bin counts.
 * @returns Result<Spec> or DataError on empty input or zero bins.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto
histplot (std::span<const float> data, std::size_t bins = 30U,
          std::string_view title = "Histogram") noexcept -> Result<Spec>;

/**
 * @brief Builds a Gaussian Kernel Density Estimate (KDE) Spec from a 1D
 * sample.
 * @param data Non-empty span of numerical samples.
 * @param grid_points Number of evaluation points across the support (> 1).
 * @param bandwidth Kernel bandwidth (<= 0 selects Silverman's rule of thumb).
 * @param title Plot title string.
 * @pre !data.empty() and grid_points >= 2.
 * @post Returned Spec contains a validated Kde curve layer.
 * @returns Result<Spec> or DataError on invalid input.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto
kdeplot (std::span<const float> data, std::size_t grid_points = 128U,
         float bandwidth = 0.0F,
         std::string_view title
         = "Kernel Density Estimate") noexcept -> Result<Spec>;

/**
 * @brief Builds a composite distribution Spec (normalized histogram + KDE).
 * @param data Non-empty span of numerical samples.
 * @param bins Number of histogram bins (> 0).
 * @param title Plot title string.
 * @pre !data.empty() and bins > 0.
 * @post Returned Spec contains both Histogram and Kde layers.
 * @returns Result<Spec> or DataError on invalid input.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto distplot (std::span<const float> data,
                             std::size_t bins = 30U,
                             std::string_view title
                             = "Distribution Plot") noexcept -> Result<Spec>;

/**
 * @brief Builds an Empirical Cumulative Distribution Function (ECDF) Spec.
 * @param data Non-empty span of numerical samples.
 * @param title Plot title string.
 * @pre !data.empty().
 * @post Returned Spec contains a monotonically non-decreasing Ecdf step layer.
 * @returns Result<Spec> or DataError on empty input.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto ecdfplot (std::span<const float> data,
                             std::string_view title
                             = "Empirical CDF") noexcept -> Result<Spec>;

/**
 * @brief Builds a regression plot Spec with scatter, OLS line, and confidence
 * band.
 * @param x Predictor variable samples.
 * @param y Response variable samples.
 * @param confidence_level Confidence interval level in (0.0, 1.0).
 * @param title Plot title string.
 * @pre x.size() >= 2, x.size() == y.size(), and 0 < confidence_level < 1.
 * @post Returned Spec contains ConfidenceBand, Scatter, and RegressionLine
 *       layers.
 * @returns Result<Spec> or DataError on invalid input.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto regplot (std::span<const float> x, std::span<const float> y,
                            float confidence_level = 0.95F,
                            std::string_view title
                            = "Linear Regression") noexcept -> Result<Spec>;

/**
 * @brief Builds a categorical box-and-whisker Spec from grouped observations.
 * @param categories Category names corresponding to each group.
 * @param groups Vector of sample vectors for each category.
 * @param title Plot title string.
 * @pre !categories.empty() and categories.size() == groups.size().
 * @post Returned Spec contains a Box layer with medians, quartiles, and
 *       whiskers.
 * @returns Result<Spec> or DataError on empty or mismatched groups.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto boxplot (std::span<const std::string> categories,
                            std::span<const std::vector<float>> groups,
                            std::string_view title
                            = "Box Plot") noexcept -> Result<Spec>;

/**
 * @brief Builds a categorical violin plot Spec with KDE envelopes and IQR.
 * @param categories Category names corresponding to each group.
 * @param groups Vector of sample vectors for each category.
 * @param title Plot title string.
 * @pre !categories.empty() and categories.size() == groups.size().
 * @post Returned Spec contains Violin density envelope and quartile layers.
 * @returns Result<Spec> or DataError on empty or mismatched groups.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto violinplot (std::span<const std::string> categories,
                               std::span<const std::vector<float>> groups,
                               std::string_view title
                               = "Violin Plot") noexcept -> Result<Spec>;

/**
 * @brief Builds a categorical bar plot Spec with group means and error bars.
 * @param categories Category names corresponding to each group.
 * @param groups Vector of sample vectors for each category.
 * @param title Plot title string.
 * @pre !categories.empty() and categories.size() == groups.size().
 * @post Returned Spec contains Bar and ErrorBar layers.
 * @returns Result<Spec> or DataError on empty or mismatched groups.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto barplot (std::span<const std::string> categories,
                            std::span<const std::vector<float>> groups,
                            std::string_view title
                            = "Bar Plot") noexcept -> Result<Spec>;

/**
 * @brief Builds a 2D color-mapped matrix heatmap Spec.
 * @param matrix Row-major matrix values of length rows * cols.
 * @param rows Number of matrix rows (> 0).
 * @param cols Number of matrix columns (> 0).
 * @param title Plot title string.
 * @pre rows > 0, cols > 0, and matrix.size() == rows * cols.
 * @post Returned Spec contains a Heatmap layer with matrix dimensions.
 * @returns Result<Spec> or DataError on dimension mismatch.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto
heatmap (std::span<const float> matrix, std::size_t rows, std::size_t cols,
         std::string_view title = "Heatmap") noexcept -> Result<Spec>;

/**
 * @brief Computes a Pearson correlation matrix and builds a heatmap Spec.
 * @param variables Span of variable sample vectors of equal length (>= 2).
 * @param names Variable names matching variables.size().
 * @param title Plot title string.
 * @pre variables.size() >= 2 and variables.size() == names.size().
 * @post Returned Spec contains a symmetric k x k correlation Heatmap layer.
 * @returns Result<Spec> or DataError on mismatched dimensions.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto corrplot (std::span<const std::vector<float>> variables,
                             std::span<const std::string> names,
                             std::string_view title
                             = "Correlation Matrix") noexcept -> Result<Spec>;

} /* namespace darshana::inline v1::stat */

#endif /* DARSHANA_STAT_HPP */
