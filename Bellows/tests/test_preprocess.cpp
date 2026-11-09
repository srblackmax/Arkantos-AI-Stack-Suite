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

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows_test
{

int
run_preprocess_tests ()
{
  using namespace bellows;

  data_table raw = {
    {10.0, missing_value (), 0.0, 5.0},
    {20.0, 2.0, 1.0, 5.0},
    {30.0, 4.0, 2.0, 5.0},
    {40.0, 6.0, 0.0, 5.0}};
  raw.set_scale (2, feature_scale::nominal, 3);

  // 1. Test column_mender (median on real, mode on nominal).
  column_mender mender (mending_strategy::median);
  fit (mender, raw);
  data_table mended = transform (mender, raw);
  if (mended.has_missing_entries ()
      || std::abs (mended.at (0, 1) - 4.0) > 1.0e-6)
    {
      std::cerr << "test_preprocess: column_mender median replacement failed\n";
      return 1;
    }

  // 2. Test moment_normaliser (zero mean, unit variance on real; skips nominal).
  moment_normaliser norm;
  fit (norm, mended);
  data_table normed = transform (norm, mended);
  double col0_sum = 0.0;
  for (std::size_t r = 0; r < normed.row_count (); ++r)
    {
      col0_sum += normed.at (r, 0);
    }
  if (std::abs (col0_sum) > 1.0e-6 || std::abs (normed.at (1, 2) - 1.0) > 1.0e-6)
    {
      std::cerr << "test_preprocess: moment_normaliser failed\n";
      return 1;
    }

  // 3. Test interval_rescaler ([0, 1]).
  interval_rescaler rescaler (0.0, 1.0);
  fit (rescaler, mended);
  data_table rescaled = transform (rescaler, mended);
  if (std::abs (rescaled.at (0, 0) - 0.0) > 1.0e-6
      || std::abs (rescaled.at (3, 0) - 1.0) > 1.0e-6)
    {
      std::cerr << "test_preprocess: interval_rescaler failed\n";
      return 1;
    }

  // 4. Test indicator_expander on nominal column 2 (cardinality 3 -> 6 cols).
  indicator_expander expander;
  fit (expander, mended);
  data_table expanded = transform (expander, mended);
  if (expanded.column_count () != 6
      || std::abs (expanded.at (1, 3) - 1.0) > 1.0e-6)
    {
      std::cerr << "test_preprocess: indicator_expander failed\n";
      return 1;
    }

  // 5. Test ordered_target_encoder.
  std::vector<double> targets = {0.0, 10.0, 20.0, 0.0};
  ordered_target_encoder ord_enc (1.0, 42);
  fit (ord_enc, mended, targets);
  data_table ord_train = ord_enc.project_ordered (mended, targets);
  data_table ord_test = transform (ord_enc, mended);
  if (ord_train.scale (2) != feature_scale::real
      || ord_test.at (2, 2) <= ord_test.at (0, 2))
    {
      std::cerr << "test_preprocess: ordered_target_encoder failed\n";
      return 1;
    }

  // 6. Test quantile_bucketer.
  quantile_bucketer bucketer (2);
  fit (bucketer, mended);
  data_table bucketed = transform (bucketer, mended);
  if (bucketed.scale (0) != feature_scale::ordinal
      || bucketed.at (0, 0) >= bucketed.at (3, 0))
    {
      std::cerr << "test_preprocess: quantile_bucketer failed\n";
      return 1;
    }

  // 7. Test dispersion_filter (drops constant column 3).
  dispersion_filter disp (1.0e-4);
  fit (disp, mended);
  data_table filtered = transform (disp, mended);
  if (filtered.column_count () != 3)
    {
      std::cerr << "test_preprocess: dispersion_filter failed to drop constant col\n";
      return 1;
    }

  // 8. Test relevance_selector.
  relevance_selector rel_sel (2);
  fit (rel_sel, mended, targets);
  data_table selected = transform (rel_sel, mended);
  if (selected.column_count () != 2)
    {
      std::cerr << "test_preprocess: relevance_selector failed\n";
      return 1;
    }

  // 9. Test binary serialisation round-trips for all 8 preprocessors.
  auto verify_trans_io = [&] (const auto &orig,
                              auto &reloaded,
                              const data_table &input,
                              std::uint32_t tag) -> bool
  {
    std::stringstream ss;
    save_artifact (orig, ss, tag);
    load_artifact (reloaded, ss);
    data_table a = transform (orig, input);
    data_table b = transform (reloaded, input);
    if (a.row_count () != b.row_count ()
        || a.column_count () != b.column_count ())
      {
        return false;
      }
    for (std::size_t r = 0; r < a.row_count (); ++r)
      {
        for (std::size_t c = 0; c < a.column_count (); ++c)
          {
            double va = a.at (r, c);
            double vb = b.at (r, c);
            if (is_missing (va) != is_missing (vb))
              {
                return false;
              }
            if (!is_missing (va) && std::abs (va - vb) > 1.0e-12)
              {
                return false;
              }
          }
      }
    return true;
  };

  column_mender mender_ld;
  moment_normaliser norm_ld;
  interval_rescaler rescaler_ld;
  indicator_expander expander_ld;
  ordered_target_encoder ord_enc_ld;
  quantile_bucketer bucketer_ld;
  dispersion_filter disp_ld;
  relevance_selector rel_sel_ld;

  if (!verify_trans_io (mender, mender_ld, raw, 21U)
      || !verify_trans_io (norm, norm_ld, mended, 22U)
      || !verify_trans_io (rescaler, rescaler_ld, mended, 23U)
      || !verify_trans_io (expander, expander_ld, mended, 24U)
      || !verify_trans_io (ord_enc, ord_enc_ld, mended, 25U)
      || !verify_trans_io (bucketer, bucketer_ld, mended, 26U)
      || !verify_trans_io (disp, disp_ld, mended, 27U)
      || !verify_trans_io (rel_sel, rel_sel_ld, mended, 28U))
    {
      std::cerr << "test_preprocess: preprocessor serialisation failed\n";
      return 1;
    }

  // 10. Stress test: all-NaN table and unseen/negative categories at test time.
  data_table all_nan (4, 3, missing_value ());
  all_nan.set_scale (2, feature_scale::nominal, 3);
  column_mender nan_mender;
  fit (nan_mender, all_nan);
  data_table nan_mended = transform (nan_mender, all_nan);
  if (nan_mended.has_missing_entries ())
    {
      std::cerr << "test_preprocess: all-NaN table mending failed\n";
      return 1;
    }

  data_table unseen_cat = {{15.0, 3.0, 99.0}, {25.0, 5.0, -3.0}};
  unseen_cat.set_scale (2, feature_scale::nominal, 3);
  data_table exp_unseen = transform (expander, unseen_cat);
  data_table ord_unseen = transform (ord_enc, unseen_cat);
  if (exp_unseen.column_count () != 6 || ord_unseen.has_missing_entries ())
    {
      std::cerr << "test_preprocess: unseen category handling failed\n";
      return 1;
    }

  // 11. Test principal_subspace_projector (PCA via Hyper Math eigh).
  data_table corr_data = {
    {1.0, 2.0, 0.1},
    {2.0, 4.0, -0.1},
    {3.0, 6.0, 0.2},
    {4.0, 8.0, -0.2},
    {5.0, 10.0, 0.0}};
  principal_subspace_projector pca (2);
  data_table coords = fit_and_transform (pca, corr_data);
  if (coords.row_count () != 5 || coords.column_count () != 2
      || pca.explained_variance_ratios ()[0] < 0.98)
    {
      std::cerr << "test_preprocess: principal_subspace_projector failed\n";
      return 1;
    }
  data_table recon = pca.reconstruct (coords);
  if (std::abs (recon.at (2, 1) - 6.0) > 1.0e-5)
    {
      std::cerr << "test_preprocess: PCA reconstruction failed\n";
      return 1;
    }
  principal_subspace_projector pca_ld;
  if (!verify_trans_io (pca, pca_ld, corr_data, 29U))
    {
      std::cerr << "test_preprocess: PCA serialisation failed\n";
      return 1;
    }

  // 12. Test interaction_expander.
  interaction_expander inter (true);
  data_table inter_out = fit_and_transform (inter, corr_data);
  if (inter_out.column_count () != 9
      || std::abs (inter_out.at (1, 3) - 4.0) > 1.0e-9
      || std::abs (inter_out.at (1, 4) - 8.0) > 1.0e-9)
    {
      std::cerr << "test_preprocess: interaction_expander failed\n";
      return 1;
    }
  interaction_expander inter_ld;
  if (!verify_trans_io (inter, inter_ld, corr_data, 30U))
    {
      std::cerr << "test_preprocess: interaction_expander io failed\n";
      return 1;
    }

  std::cout << "  [PASS] test_preprocess (scaling, mending, encoding, selection, projection)\n";
  return 0;
}

} // namespace bellows_test
