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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows_test
{

int
run_io_and_cli_tests ()
{
  using namespace bellows;

  data_table x (30, 2);
  x.set_scale (1, feature_scale::nominal, 3);
  std::vector<double> y (30, 0.0);
  for (std::size_t i = 0; i < 30; ++i)
    {
      double f0 = static_cast<double> (i) * 0.25;
      double cat = static_cast<double> (i % 3);
      x.at (i, 0) = f0;
      x.at (i, 1) = cat;
      y[i] = 2.0 * f0 + (cat == 2.0 ? 5.0 : 0.0);
    }

  // 1. Test in-memory binary serialisation & checksum verification for booster.
  using obl_booster = stagewise_booster<squared_residual_loss,
                                        quantile_bin_policy,
                                        oblivious_symmetric_growth>;
  obl_booster trained (with_stage_count{30}, with_step_rate{0.15});
  fit (trained, x, y);

  std::stringstream buf (std::ios::in | std::ios::out | std::ios::binary);
  save_artifact (trained, buf, 1U);

  obl_booster loaded;
  load_artifact (loaded, buf);

  for (std::size_t i = 0; i < x.row_count (); ++i)
    {
      double p_orig = infer_one (trained, x.row_span (i));
      double p_load = infer_one (loaded, x.row_span (i));
      if (std::abs (p_orig - p_load) > 1.0e-12)
        {
          std::cerr << "test_io_and_cli: booster serialisation mismatch\n";
          return 1;
        }
    }

  // 2. Test corruption detection via FNV-1a checksum.
  std::string raw_bytes = buf.str ();
  raw_bytes.back () ^= 0x55;
  std::istringstream corrupt_stream (raw_bytes, std::ios::binary);
  bool caught_corruption = false;
  try
    {
      obl_booster dummy;
      load_artifact (dummy, corrupt_stream);
    }
  catch (const std::exception &)
    {
      caught_corruption = true;
    }
  if (!caught_corruption)
    {
      std::cerr << "test_io_and_cli: failed to detect corrupted artifact\n";
      return 1;
    }

  // 3. Test end-to-end CLI tools (`bellows-train`, `bellows`, `bellows-infer`).
  const std::string csv_train = "test_train_tmp.csv";
  const std::string csv_infer = "test_infer_tmp.csv";
  const std::string csv_ragged = "test_ragged_tmp.csv";
  const std::string model_file = "test_model_tmp.blws";
  const std::string corrupt_file = "test_corrupt_tmp.blws";

  {
    std::ofstream out_tr (csv_train);
    std::ofstream out_in (csv_infer);
    for (std::size_t i = 0; i < 20; ++i)
      {
        double a = static_cast<double> (i);
        double b = static_cast<double> (i % 4);
        double target = 2.0 * a + b;
        out_tr << a << "," << b << "," << target << "\n";
        out_in << a << "," << b << "\n";
      }
    std::ofstream out_rag (csv_ragged);
    out_rag << "1.0,2.0,3.0\n4.0,5.0\n";
    std::ofstream out_cor (corrupt_file, std::ios::binary);
    out_cor.write (raw_bytes.data (),
                    static_cast<std::streamsize> (raw_bytes.size ()));
  }

  int rc_train = std::system (
    ("./bellows-train --input " + csv_train + " --output " + model_file
      + " --stages 25 --rate 0.15 > /dev/null")
      .c_str ());
  int rc_inspect = std::system (
    ("./bellows --inspect " + model_file + " > /dev/null").c_str ());
  int rc_inspect_corrupt = std::system (
    ("./bellows --inspect " + corrupt_file + " > /dev/null 2>&1").c_str ());
  int rc_train_ragged = std::system (
    ("./bellows-train --input " + csv_ragged + " --output " + model_file
      + " > /dev/null 2>&1")
      .c_str ());
  int rc_infer_row = std::system (
    ("./bellows-infer --model " + model_file + " --row 5.0,1.0 > /dev/null")
      .c_str ());
  int rc_infer_batch = std::system (
    ("./bellows-infer --model " + model_file + " --input " + csv_infer
      + " > /dev/null")
      .c_str ());

  std::remove (csv_train.c_str ());
  std::remove (csv_infer.c_str ());
  std::remove (csv_ragged.c_str ());
  std::remove (model_file.c_str ());
  std::remove (corrupt_file.c_str ());

  if (rc_train != 0 || rc_inspect != 0 || rc_inspect_corrupt == 0
      || rc_train_ragged == 0 || rc_infer_row != 0 || rc_infer_batch != 0)
    {
      std::cerr << "test_io_and_cli: CLI end-to-end execution failed\n";
      return 1;
    }

  std::cout << "  [PASS] test_io_and_cli (binary artifact, checksum, infer_one, CLI round-trip)\n";
  return 0;
}

} // namespace bellows_test
