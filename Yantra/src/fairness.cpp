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

#include <yantra/fairness.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

namespace yantra::inline v1
{

namespace
{

auto
validate_dataset (const DatasetView &dataset) noexcept -> Result<uint32_t>
{
  if (dataset.prediction_scores.empty () || dataset.true_labels.empty ()
      || dataset.sensitive_groups.empty ())
    {
      return std::unexpected (Error::fairness (
          FairnessErrorCode::MissingSensitiveAttribute, 0U,
          "DatasetView spans must be non-empty for fairness auditing"));
    }
  const std::size_t n = dataset.prediction_scores.size ();
  if (dataset.true_labels.size () != n
      || dataset.sensitive_groups.size () != n)
    {
      return std::unexpected (Error::fairness (
          FairnessErrorCode::MissingSensitiveAttribute, 0U,
          "DatasetView column lengths do not match"));
    }
  if (dataset.classification_threshold < 0.0
      || dataset.classification_threshold > 1.0)
    {
      return std::unexpected (Error::fairness (
          FairnessErrorCode::InvalidThresholdBounds, 0U,
          "Classification threshold must lie in [0.0, 1.0]"));
    }

  uint32_t g_count = dataset.num_groups;
  if (g_count == 0U)
    {
      uint32_t max_g = 0U;
      for (const uint32_t g : dataset.sensitive_groups)
        {
          max_g = std::max (max_g, g);
        }
      g_count = max_g + 1U;
    }
  if (g_count < 2U)
    {
      return std::unexpected (Error::fairness (
          FairnessErrorCode::InsufficientSamplesPerGroup, 0U,
          "Fairness auditing requires at least 2 distinct sensitive groups"));
    }
  return g_count;
}

auto
finalize_report (std::vector<GroupConfusionMatrix> &&groups) noexcept
    -> Result<FairnessReport>
{
  double min_pr = 1.0;
  double max_pr = 0.0;
  double min_tpr = 1.0;
  double max_tpr = 0.0;
  double min_fpr = 1.0;
  double max_fpr = 0.0;
  double min_ppv = 1.0;
  double max_ppv = 0.0;

  uint64_t total_correct = 0U;
  uint64_t total_rows = 0U;

  for (std::size_t g = 0U; g < groups.size (); ++g)
    {
      groups[g].group_id = static_cast<uint32_t> (g);
      const uint64_t n_g = groups[g].total ();
      if (n_g < 2U)
        {
          return std::unexpected (Error::fairness (
              FairnessErrorCode::InsufficientSamplesPerGroup,
              static_cast<uint32_t> (g),
              "Sensitive group has fewer than 2 samples"));
        }

      total_rows += n_g;
      total_correct += groups[g].tp + groups[g].tn;

      const double pr = groups[g].positive_rate ();
      const double tpr = groups[g].tpr ();
      const double fpr = groups[g].fpr ();
      const double ppv = groups[g].ppv ();

      min_pr = std::min (min_pr, pr);
      max_pr = std::max (max_pr, pr);
      min_tpr = std::min (min_tpr, tpr);
      max_tpr = std::max (max_tpr, tpr);
      min_fpr = std::min (min_fpr, fpr);
      max_fpr = std::max (max_fpr, fpr);
      min_ppv = std::min (min_ppv, ppv);
      max_ppv = std::max (max_ppv, ppv);
    }

  FairnessReport rep;
  rep.demographic_parity_difference = max_pr - min_pr;
  rep.demographic_parity_ratio = (max_pr > 1.0e-12) ? (min_pr / max_pr) : 1.0;
  rep.equal_opportunity_difference = max_tpr - min_tpr;
  rep.equalized_odds_difference
      = std::max (max_tpr - min_tpr, max_fpr - min_fpr);
  rep.predictive_parity_difference = max_ppv - min_ppv;
  rep.overall_accuracy
      = total_rows > 0U
            ? static_cast<double> (total_correct)
                  / static_cast<double> (total_rows)
            : 0.0;
  rep.total_rows = total_rows;
  rep.group_metrics = std::move (groups);
  return rep;
}

struct FairnessPool
{
  static constexpr std::size_t kWorkers = 8U;

  std::mutex dispatch_mu;
  std::mutex cv_mu;
  std::condition_variable cv_start;
  std::condition_variable cv_done;

  const double *scores = nullptr;
  const uint8_t *labels = nullptr;
  const uint32_t *groups = nullptr;
  std::size_t n = 0U;
  uint32_t num_groups = 0U;
  double thresh = 0.5;
  std::vector<std::vector<GroupConfusionMatrix>> *local_mats = nullptr;

  uint64_t epoch = 0U;
  std::atomic<uint32_t> remaining{ 0U };
  bool stop = false;
  std::vector<std::thread> threads;

  FairnessPool ()
  {
    threads.reserve (kWorkers - 1U);
    for (std::size_t t = 1U; t < kWorkers; ++t)
      {
        threads.emplace_back ([this, t] () { worker_loop (t); });
      }
  }

  ~FairnessPool ()
  {
    {
      std::lock_guard<std::mutex> lk (cv_mu);
      stop = true;
      ++epoch;
    }
    cv_start.notify_all ();
    for (auto &th : threads)
      {
        if (th.joinable ())
          {
            th.join ();
          }
      }
  }

  void
  run_slice (std::size_t t) noexcept
  {
    const std::size_t chunk = (n + kWorkers - 1U) / kWorkers;
    const std::size_t beg = std::min (n, t * chunk);
    const std::size_t end = std::min (n, beg + chunk);
    auto &loc = (*local_mats)[t];
    for (std::size_t i = beg; i < end; ++i)
      {
        const uint32_t g = groups[i];
        if (g >= num_groups)
          {
            continue;
          }
        const bool pred = (scores[i] >= thresh);
        const bool truth = (labels[i] != 0U);
        auto &m = loc[g];
        m.tp += static_cast<uint64_t> (pred && truth);
        m.fp += static_cast<uint64_t> (pred && !truth);
        m.tn += static_cast<uint64_t> (!pred && !truth);
        m.fn += static_cast<uint64_t> (!pred && truth);
      }
  }

  void
  worker_loop (std::size_t t) noexcept
  {
    uint64_t seen = 0U;
    for (;;)
      {
        {
          std::unique_lock<std::mutex> lk (cv_mu);
          cv_start.wait (lk, [&] () { return stop || epoch != seen; });
          if (stop)
            {
              return;
            }
          seen = epoch;
        }
        run_slice (t);
        if (remaining.fetch_sub (1U, std::memory_order_acq_rel) == 1U)
          {
            std::lock_guard<std::mutex> lk (cv_mu);
            cv_done.notify_one ();
          }
      }
  }

  void
  execute (const double *sc, const uint8_t *lb, const uint32_t *gr,
           std::size_t rows, uint32_t ng, double th,
           std::vector<std::vector<GroupConfusionMatrix>> &mats)
  {
    std::lock_guard<std::mutex> guard (dispatch_mu);
    {
      std::lock_guard<std::mutex> lk (cv_mu);
      scores = sc;
      labels = lb;
      groups = gr;
      n = rows;
      num_groups = ng;
      thresh = th;
      local_mats = &mats;
      remaining.store (static_cast<uint32_t> (kWorkers - 1U),
                       std::memory_order_release);
      ++epoch;
    }
    cv_start.notify_all ();
    run_slice (0U);
    if (remaining.load (std::memory_order_acquire) != 0U)
      {
        std::unique_lock<std::mutex> lk (cv_mu);
        cv_done.wait (lk, [&] () {
          return remaining.load (std::memory_order_acquire) == 0U;
        });
      }
  }
};

} /* namespace */

auto
FairnessAuditor::audit (const DatasetView &dataset) noexcept
    -> Result<FairnessReport>
{
  const uint64_t t0 = monotonic_time_ns ();
  auto g_res = validate_dataset (dataset);
  if (!g_res.has_value ())
    {
      return std::unexpected (g_res.error ());
    }
  const uint32_t num_groups = *g_res;
  const std::size_t n = dataset.prediction_scores.size ();
  const double thresh = dataset.classification_threshold;
  const double *scores = dataset.prediction_scores.data ();
  const uint8_t *labels = dataset.true_labels.data ();
  const uint32_t *groups = dataset.sensitive_groups.data ();

  std::vector<GroupConfusionMatrix> g_mats (num_groups);

  if (n >= 250000U)
    {
      static FairnessPool pool;
      std::vector<std::vector<GroupConfusionMatrix>> local_mats (
          FairnessPool::kWorkers,
          std::vector<GroupConfusionMatrix> (num_groups));
      pool.execute (scores, labels, groups, n, num_groups, thresh, local_mats);

      for (std::size_t t = 0U; t < FairnessPool::kWorkers; ++t)
        {
          for (uint32_t g = 0U; g < num_groups; ++g)
            {
              g_mats[g].tp += local_mats[t][g].tp;
              g_mats[g].fp += local_mats[t][g].fp;
              g_mats[g].tn += local_mats[t][g].tn;
              g_mats[g].fn += local_mats[t][g].fn;
            }
        }
    }
  else
    {
      for (std::size_t i = 0U; i < n; ++i)
        {
          const uint32_t g = groups[i];
          if (g >= num_groups)
            {
              return std::unexpected (Error::fairness (
                  FairnessErrorCode::MissingSensitiveAttribute, g,
                  "Row sensitive group index exceeds num_groups"));
            }
          const bool pred = (scores[i] >= thresh);
          const bool truth = (labels[i] != 0U);
          auto &m = g_mats[g];
          m.tp += static_cast<uint64_t> (pred && truth);
          m.fp += static_cast<uint64_t> (pred && !truth);
          m.tn += static_cast<uint64_t> (!pred && !truth);
          m.fn += static_cast<uint64_t> (!pred && truth);
        }
    }

  auto rep_res = finalize_report (std::move (g_mats));
  if (rep_res.has_value ())
    {
      FairnessStats::global ().record_audit (
          rep_res->equalized_odds_difference, 0.0, monotonic_time_ns () - t0);
    }
  return rep_res;
}

auto
FairnessAuditor::audit_scalar_reference (const DatasetView &dataset) noexcept
    -> Result<FairnessReport>
{
  auto g_res = validate_dataset (dataset);
  if (!g_res.has_value ())
    {
      return std::unexpected (g_res.error ());
    }
  const uint32_t num_groups = *g_res;
  const std::size_t n = dataset.prediction_scores.size ();
  const double thresh = dataset.classification_threshold;

  std::vector<GroupConfusionMatrix> g_mats (num_groups);
  for (std::size_t i = 0U; i < n; ++i)
    {
      const uint32_t g = dataset.sensitive_groups[i];
      if (g >= num_groups)
        {
          return std::unexpected (Error::fairness (
              FairnessErrorCode::MissingSensitiveAttribute, g,
              "Row sensitive group index exceeds num_groups"));
        }
      const bool pred = (dataset.prediction_scores[i] >= thresh);
      const bool truth = (dataset.true_labels[i] != 0U);
      if (pred && truth)
        {
          ++g_mats[g].tp;
        }
      else if (pred && !truth)
        {
          ++g_mats[g].fp;
        }
      else if (!pred && !truth)
        {
          ++g_mats[g].tn;
        }
      else
        {
          ++g_mats[g].fn;
        }
    }

  return finalize_report (std::move (g_mats));
}

auto
FairnessAuditor::optimize_thresholds (const DatasetView &dataset,
                                      double tolerance_epsilon,
                                      std::size_t max_iterations) noexcept
    -> Result<ThresholdOptimizationResult>
{
  const uint64_t t0 = monotonic_time_ns ();
  if (tolerance_epsilon <= 0.0 || max_iterations == 0U)
    {
      return std::unexpected (Error::fairness (
          FairnessErrorCode::ConstraintUnsatisfiable, 0U,
          "Fairness tolerance epsilon and max_iterations must be positive"));
    }

  auto init_rep = audit (dataset);
  if (!init_rep.has_value ())
    {
      return std::unexpected (init_rep.error ());
    }

  const uint32_t num_groups
      = static_cast<uint32_t> (init_rep->group_metrics.size ());
  constexpr std::size_t kBins = 1000U;

  std::vector<std::vector<uint64_t>> pos_hist (
      num_groups, std::vector<uint64_t> (kBins + 1U, 0U));
  std::vector<std::vector<uint64_t>> neg_hist (
      num_groups, std::vector<uint64_t> (kBins + 1U, 0U));
  std::vector<uint64_t> total_pos (num_groups, 0U);
  std::vector<uint64_t> total_neg (num_groups, 0U);

  const std::size_t n = dataset.prediction_scores.size ();
  for (std::size_t i = 0U; i < n; ++i)
    {
      const uint32_t g = dataset.sensitive_groups[i];
      const double s = std::clamp (dataset.prediction_scores[i], 0.0, 1.0);
      const auto bin = static_cast<std::size_t> (
          std::llround (s * static_cast<double> (kBins)));
      if (dataset.true_labels[i] != 0U)
        {
          ++pos_hist[g][bin];
          ++total_pos[g];
        }
      else
        {
          ++neg_hist[g][bin];
          ++total_neg[g];
        }
    }

  for (uint32_t g = 0U; g < num_groups; ++g)
    {
      if (total_pos[g] == 0U || total_neg[g] == 0U)
        {
          return std::unexpected (Error::fairness (
              FairnessErrorCode::ConstraintUnsatisfiable, g,
              "Equalized odds optimization requires both positive and "
              "negative labels in every sensitive group"));
        }
      /* Convert histograms into suffix cumulative sums >= bin.  */
      for (std::size_t b = kBins; b > 0U; --b)
        {
          pos_hist[g][b - 1U] += pos_hist[g][b];
          neg_hist[g][b - 1U] += neg_hist[g][b];
        }
    }

  std::vector<double> thresholds (num_groups,
                                  dataset.classification_threshold);

  auto eval_group = [&] (uint32_t g, double th, double &tpr, double &fpr,
                         uint64_t &correct) noexcept {
    const auto b = static_cast<std::size_t> (std::clamp (
        std::llround (th * static_cast<double> (kBins)), 0LL,
        static_cast<long long> (kBins)));
    const uint64_t tp = pos_hist[g][b];
    const uint64_t fp = neg_hist[g][b];
    const uint64_t tn = total_neg[g] - fp;
    tpr = static_cast<double> (tp) / static_cast<double> (total_pos[g]);
    fpr = static_cast<double> (fp) / static_cast<double> (total_neg[g]);
    correct = tp + tn;
  };

  double cur_eo_diff = init_rep->equalized_odds_difference;
  double cur_acc = init_rep->overall_accuracy;
  std::size_t iter_used = 0U;
  bool converged = (cur_eo_diff <= tolerance_epsilon);

  for (std::size_t iter = 1U; iter <= max_iterations && !converged; ++iter)
    {
      iter_used = iter;
      std::vector<double> tprs (num_groups, 0.0);
      std::vector<double> fprs (num_groups, 0.0);
      uint64_t tot_corr = 0U;

      for (uint32_t g = 0U; g < num_groups; ++g)
        {
          uint64_t cg = 0U;
          eval_group (g, thresholds[g], tprs[g], fprs[g], cg);
          tot_corr += cg;
        }

      const double min_tpr = *std::min_element (tprs.begin (), tprs.end ());
      const double max_tpr = *std::max_element (tprs.begin (), tprs.end ());
      const double min_fpr = *std::min_element (fprs.begin (), fprs.end ());
      const double max_fpr = *std::max_element (fprs.begin (), fprs.end ());

      cur_eo_diff = std::max (max_tpr - min_tpr, max_fpr - min_fpr);
      cur_acc = static_cast<double> (tot_corr) / static_cast<double> (n);
      if (cur_eo_diff <= tolerance_epsilon)
        {
          converged = true;
          break;
        }

      const double target_tpr = 0.5 * (min_tpr + max_tpr);
      const double target_fpr = 0.5 * (min_fpr + max_fpr);

      /* Coordinate search per group to align (TPR_g, FPR_g) with target while
         preserving accuracy.  */
      for (uint32_t g = 0U; g < num_groups; ++g)
        {
          double best_t = thresholds[g];
          double best_loss = 1.0e30;
          for (std::size_t b = 10U; b < kBins - 10U; b += 5U)
            {
              const double cand_t
                  = static_cast<double> (b) / static_cast<double> (kBins);
              double ctpr = 0.0;
              double cfpr = 0.0;
              uint64_t ccorr = 0U;
              eval_group (g, cand_t, ctpr, cfpr, ccorr);
              const double eo_err = std::max (std::abs (ctpr - target_tpr),
                                              std::abs (cfpr - target_fpr));
              const double acc_pen
                  = 0.05
                    * (1.0
                       - static_cast<double> (ccorr)
                             / static_cast<double> (total_pos[g]
                                                    + total_neg[g]));
              const double loss = eo_err + acc_pen;
              if (loss < best_loss)
                {
                  best_loss = loss;
                  best_t = cand_t;
                }
            }
          thresholds[g] = 0.5 * thresholds[g] + 0.5 * best_t;
        }
    }

  /* Final evaluation after loop.  */
  {
    std::vector<double> tprs (num_groups, 0.0);
    std::vector<double> fprs (num_groups, 0.0);
    uint64_t tot_corr = 0U;
    for (uint32_t g = 0U; g < num_groups; ++g)
      {
        uint64_t cg = 0U;
        eval_group (g, thresholds[g], tprs[g], fprs[g], cg);
        tot_corr += cg;
      }
    const double min_tpr = *std::min_element (tprs.begin (), tprs.end ());
    const double max_tpr = *std::max_element (tprs.begin (), tprs.end ());
    const double min_fpr = *std::min_element (fprs.begin (), fprs.end ());
    const double max_fpr = *std::max_element (fprs.begin (), fprs.end ());
    cur_eo_diff = std::max (max_tpr - min_tpr, max_fpr - min_fpr);
    cur_acc = static_cast<double> (tot_corr) / static_cast<double> (n);
    if (cur_eo_diff <= tolerance_epsilon)
      {
        converged = true;
      }
  }

  double avg_shift = 0.0;
  for (const double th : thresholds)
    {
      avg_shift += std::abs (th - dataset.classification_threshold);
    }
  avg_shift /= static_cast<double> (num_groups);

  FairnessStats::global ().record_audit (cur_eo_diff, avg_shift,
                                         monotonic_time_ns () - t0);

  ThresholdOptimizationResult res;
  res.group_thresholds = std::move (thresholds);
  res.iterations = iter_used;
  res.converged = converged;
  res.initial_equalized_odds_diff = init_rep->equalized_odds_difference;
  res.optimized_equalized_odds_diff = cur_eo_diff;
  res.initial_accuracy = init_rep->overall_accuracy;
  res.optimized_accuracy = cur_acc;
  return res;
}

auto
DriftDetector::compute_drift (std::span<const double> reference_scores,
                              std::span<const double> production_scores,
                              std::size_t num_bins) noexcept
    -> Result<DriftReport>
{
  if (reference_scores.size () < 10U || production_scores.size () < 10U
      || num_bins < 2U)
    {
      return std::unexpected (Error::fairness (
          FairnessErrorCode::InsufficientSamplesPerGroup, 0U,
          "DriftDetector requires at least 10 samples and 2 quantile bins"));
    }

  std::vector<double> ref_sorted (reference_scores.begin (),
                                  reference_scores.end ());
  std::vector<double> prod_sorted (production_scores.begin (),
                                   production_scores.end ());
  std::sort (ref_sorted.begin (), ref_sorted.end ());
  std::sort (prod_sorted.begin (), prod_sorted.end ());

  const std::size_t n_ref = ref_sorted.size ();
  const std::size_t n_prod = prod_sorted.size ();
  const double inv_ref = 1.0 / static_cast<double> (n_ref);
  const double inv_prod = 1.0 / static_cast<double> (n_prod);

  /* Exact two-sample Kolmogorov-Smirnov statistic D_KS via merged scan.  */
  std::size_t i = 0U;
  std::size_t j = 0U;
  double max_ks = 0.0;
  while (i < n_ref && j < n_prod)
    {
      const double v_ref = ref_sorted[i];
      const double v_prod = prod_sorted[j];
      const double x = std::min (v_ref, v_prod);
      while (i < n_ref && ref_sorted[i] <= x)
        {
          ++i;
        }
      while (j < n_prod && prod_sorted[j] <= x)
        {
          ++j;
        }
      const double cdf_ref = static_cast<double> (i) * inv_ref;
      const double cdf_prod = static_cast<double> (j) * inv_prod;
      max_ks = std::max (max_ks, std::abs (cdf_ref - cdf_prod));
    }

  /* Population Stability Index (PSI) across num_bins reference quantiles.  */
  std::vector<double> cutpoints (num_bins - 1U);
  for (std::size_t b = 1U; b < num_bins; ++b)
    {
      const std::size_t idx = (b * n_ref) / num_bins;
      cutpoints[b - 1U] = ref_sorted[std::min (idx, n_ref - 1U)];
    }

  double psi = 0.0;
  constexpr double kEps = 1.0e-6;
  std::size_t prev_r = 0U;
  std::size_t prev_p = 0U;

  for (std::size_t b = 0U; b < num_bins; ++b)
    {
      std::size_t next_r = n_ref;
      std::size_t next_p = n_prod;
      if (b + 1U < num_bins)
        {
          const double edge = cutpoints[b];
          next_r = static_cast<std::size_t> (
              std::upper_bound (ref_sorted.begin (), ref_sorted.end (), edge)
              - ref_sorted.begin ());
          next_p = static_cast<std::size_t> (
              std::upper_bound (prod_sorted.begin (), prod_sorted.end (), edge)
              - prod_sorted.begin ());
        }
      const double p_ref
          = std::max (kEps, static_cast<double> (next_r - prev_r) * inv_ref);
      const double p_prod
          = std::max (kEps, static_cast<double> (next_p - prev_p) * inv_prod);
      psi += (p_prod - p_ref) * std::log (p_prod / p_ref);
      prev_r = next_r;
      prev_p = next_p;
    }

  DriftReport rep;
  rep.psi = psi;
  rep.ks_statistic = max_ks;
  rep.drift_detected = (psi >= 0.2 || max_ks >= 0.1);
  return rep;
}

auto
DriftDetector::compute_drift_scalar_reference (
    std::span<const double> reference_scores,
    std::span<const double> production_scores,
    std::size_t num_bins) noexcept -> Result<DriftReport>
{
  return compute_drift (reference_scores, production_scores, num_bins);
}

} /* namespace yantra::inline v1 */
