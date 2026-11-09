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

#ifndef YANTRA_FAIRNESS_HPP
#define YANTRA_FAIRNESS_HPP

#include <yantra/error.hpp>
#include <yantra/observability.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace yantra::inline v1
{

/**
 * @brief Zero-copy view over model prediction scores, ground-truth binary
 * labels, and sensitive group identifiers produced by Hyper Math / Bellows.
 * @note Thread-safety: Read-only non-owning view; safe for concurrent reads.
 */
struct DatasetView
{
  std::span<const double> prediction_scores;
  std::span<const uint8_t> true_labels;
  std::span<const uint32_t> sensitive_groups;
  uint32_t num_groups = 0U;
  double classification_threshold = 0.5;
  std::string score_column_name = "y_score";
  std::string label_column_name = "y_true";
  std::string group_column_name = "group_id";
};

/**
 * @brief Per-group confusion matrix cell counts and derived rates.
 */
struct GroupConfusionMatrix
{
  uint32_t group_id = 0U;
  uint64_t tp = 0U;
  uint64_t fp = 0U;
  uint64_t tn = 0U;
  uint64_t fn = 0U;

  [[nodiscard]] auto
  total () const noexcept -> uint64_t
  {
    return tp + fp + tn + fn;
  }

  [[nodiscard]] auto
  positive_rate () const noexcept -> double
  {
    const uint64_t n = total ();
    return n > 0U ? static_cast<double> (tp + fp) / static_cast<double> (n)
                  : 0.0;
  }

  [[nodiscard]] auto
  tpr () const noexcept -> double
  {
    const uint64_t pos = tp + fn;
    return pos > 0U ? static_cast<double> (tp) / static_cast<double> (pos)
                    : 0.0;
  }

  [[nodiscard]] auto
  fpr () const noexcept -> double
  {
    const uint64_t neg = fp + tn;
    return neg > 0U ? static_cast<double> (fp) / static_cast<double> (neg)
                    : 0.0;
  }

  [[nodiscard]] auto
  ppv () const noexcept -> double
  {
    const uint64_t pred_pos = tp + fp;
    return pred_pos > 0U
               ? static_cast<double> (tp) / static_cast<double> (pred_pos)
               : 0.0;
  }
};

/**
 * @brief Group fairness audit report containing all standard parity and odds
 * metrics and per-group confusion matrices.
 */
struct FairnessReport
{
  double demographic_parity_difference = 0.0;
  double demographic_parity_ratio = 1.0;
  double equalized_odds_difference = 0.0;
  double equal_opportunity_difference = 0.0;
  double predictive_parity_difference = 0.0;
  double overall_accuracy = 0.0;
  uint64_t total_rows = 0U;
  std::vector<GroupConfusionMatrix> group_metrics;
};

/**
 * @brief Outcome of post-processing group threshold optimization to satisfy
 * an equalized odds constraint while minimizing accuracy loss.
 */
struct ThresholdOptimizationResult
{
  std::vector<double> group_thresholds;
  std::size_t iterations = 0U;
  bool converged = false;
  double initial_equalized_odds_diff = 0.0;
  double optimized_equalized_odds_diff = 0.0;
  double initial_accuracy = 0.0;
  double optimized_accuracy = 0.0;
};

/**
 * @brief High-throughput algorithmic bias and group fairness auditor with
 * post-processing threshold optimization.
 * @note Thread-safety: Stateless methods; safe for concurrent invocation.
 */
class FairnessAuditor
{
public:
  FairnessAuditor () noexcept = default;

  /**
   * @brief Computes all group fairness metrics over a DatasetView using
   * vectorized / multi-threaded histogram reduction.
   * @param dataset Zero-copy dataset view.
   * @returns Result<FairnessReport> or FairnessError.
   */
  [[nodiscard]] static auto
  audit (const DatasetView &dataset) noexcept -> Result<FairnessReport>;

  /**
   * @brief Computes all group fairness metrics using a single-threaded scalar
   * fp64 reference loop for numerical equivalence verification.
   * @param dataset Zero-copy dataset view.
   * @returns Result<FairnessReport> or FairnessError.
   */
  [[nodiscard]] static auto
  audit_scalar_reference (const DatasetView &dataset) noexcept
      -> Result<FairnessReport>;

  /**
   * @brief Optimizes per-group classification thresholds to satisfy an
   * Equalized Odds fairness constraint within tolerance_epsilon while
   * minimizing overall accuracy loss.
   * @param dataset Zero-copy dataset view.
   * @param tolerance_epsilon Maximum allowed Equalized Odds Difference (> 0).
   * @param max_iterations Maximum optimization iterations (default 100).
   * @returns Result<ThresholdOptimizationResult> or FairnessError.
   */
  [[nodiscard]] static auto optimize_thresholds (
      const DatasetView &dataset, double tolerance_epsilon = 0.02,
      std::size_t max_iterations = 100U) noexcept
      -> Result<ThresholdOptimizationResult>;
};

/**
 * @brief Report of statistical distribution drift between reference and
 * production score distributions.
 */
struct DriftReport
{
  double psi = 0.0;
  double ks_statistic = 0.0;
  bool drift_detected = false;
};

/**
 * @brief High-throughput distribution drift detector computing Population
 * Stability Index (PSI) and exact two-sample Kolmogorov-Smirnov (D_KS).
 * @note Thread-safety: Stateless methods; safe for concurrent invocation.
 */
class DriftDetector
{
public:
  DriftDetector () noexcept = default;

  /**
   * @brief Computes PSI (over num_bins quantile bins) and exact two-sample
   * Kolmogorov-Smirnov statistic D_KS using SIMD-accelerated histogramming.
   * @param reference_scores Reference baseline distribution scores.
   * @param production_scores Production distribution scores.
   * @param num_bins Number of quantile bins for PSI (default 10).
   * @returns Result<DriftReport> or FairnessError.
   */
  [[nodiscard]] static auto
  compute_drift (std::span<const double> reference_scores,
                 std::span<const double> production_scores,
                 std::size_t num_bins = 10U) noexcept -> Result<DriftReport>;

  /**
   * @brief Computes PSI and exact D_KS using a pure scalar fp64 reference
   * path for 1e-12 numerical equivalence verification.
   * @param reference_scores Reference baseline distribution scores.
   * @param production_scores Production distribution scores.
   * @param num_bins Number of quantile bins for PSI (default 10).
   * @returns Result<DriftReport> or FairnessError.
   */
  [[nodiscard]] static auto
  compute_drift_scalar_reference (std::span<const double> reference_scores,
                                  std::span<const double> production_scores,
                                  std::size_t num_bins = 10U) noexcept
      -> Result<DriftReport>;
};

namespace fairness
{
using yantra::v1::DatasetView;
using yantra::v1::DriftDetector;
using yantra::v1::DriftReport;
using yantra::v1::FairnessAuditor;
using yantra::v1::FairnessReport;
using yantra::v1::GroupConfusionMatrix;
using yantra::v1::ThresholdOptimizationResult;
} /* namespace fairness */

} /* namespace yantra::inline v1 */

#endif /* YANTRA_FAIRNESS_HPP */
