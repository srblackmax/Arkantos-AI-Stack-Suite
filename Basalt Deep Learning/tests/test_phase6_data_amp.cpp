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

#include <basalt/amp.hpp>
#include <basalt/autograd.hpp>
#include <basalt/data.hpp>
#include <basalt/error.hpp>
#include <basalt/loss.hpp>
#include <basalt/nn.hpp>
#include <basalt/ops.hpp>
#include <basalt/optim.hpp>
#include <basalt/random.hpp>
#include <basalt/tensor.hpp>
#include <bellows/core/data_table.hpp>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace
{

void
test_dataloader_determinism_and_prefetch (void)
{
  using namespace basalt;

  Tensor x = arange (0.0, 40.0, 1.0).reshape ({20, 2});
  Tensor y = arange (0.0, 20.0, 1.0).reshape ({20, 1});
  auto ds = std::make_shared<data::Tensor_Dataset> (x, y);

  data::Data_Loader loader_0w (ds, 4, true, 0, 2, false, 777ULL);
  data::Data_Loader loader_3w (ds, 4, true, 3, 2, false, 777ULL);

  auto batches_0 = loader_0w.collect_epoch ();
  auto batches_3 = loader_3w.collect_epoch ();
  BASALT_ASSERT (batches_0.size () == 5 && batches_3.size () == 5,
                 "expected 5 batches");

  for (size_t i = 0; i < 5; ++i)
    {
      BASALT_ASSERT (allclose (batches_0[i].inputs, batches_3[i].inputs),
                     "DataLoader must be deterministic across worker counts");
      BASALT_ASSERT (allclose (batches_0[i].targets, batches_3[i].targets),
                     "DataLoader targets deterministic across worker counts");
    }
}

void
test_bellows_tabular_and_posix_mmap_dataset (void)
{
  using namespace basalt;

  /* Tabular_Dataset with Bellows moment_normaliser.  */
  Tensor raw_feat = tensor ({10.0, 100.0, 20.0, 200.0, 30.0, 300.0})
                      .reshape ({3, 2});
  bellows::data_table tbl = to_bellows_table (raw_feat);
  data::Tabular_Dataset tab_ds (tbl, {1.0, 2.0, 3.0}, true);
  BASALT_ASSERT (tab_ds.size () == 3, "Tabular_Dataset size");
  data::Batch s1 = tab_ds.get (1);
  /* Middle row of z-score standardized {10, 20, 30} must be 0.0.  */
  BASALT_ASSERT (std::fabs (s1.inputs.read_scalar_at (0)) < 1e-5,
                 "Bellows moment_normaliser standardization check");

  /* POSIX Mmap_Dataset write & zero-copy read.  */
  std::string path = "/tmp/basalt_test_mmap.bin";
  Tensor mx = arange (1.0, 13.0, 1.0).reshape ({4, 3});
  Tensor my = tensor ({0.0, 1.0, 2.0, 3.0}).reshape ({4, 1});
  data::Mmap_Dataset::write_binary_file (path, mx, my);

  {
    data::Mmap_Dataset mmap_ds (path);
    BASALT_ASSERT (mmap_ds.size () == 4, "Mmap_Dataset sample count");
    data::Batch b2 = mmap_ds.get (2);
    BASALT_ASSERT (b2.inputs.read_scalar_at (0) == 7.0f
                   && b2.targets.read_scalar_at (0) == 2.0f,
                   "Mmap_Dataset record contents");
  }
  std::remove (path.c_str ());
}

void
test_mixed_precision_and_grad_scaler (void)
{
  using namespace basalt;

  nn::Linear lin (4, 2);
  optim::Adam opt (lin.parameters (), 1e-2);
  amp::Grad_Scaler scaler (1024.0, 2.0, 0.5, 2, true);

  Tensor x = ones ({3, 4}, Tensor_Options ().with_requires_grad (true));
  Tensor target = zeros ({3, 2});

  opt.zero_grad ();
  Tensor pred;
  {
    amp::Autocast_Guard ac (Dtype::Float16);
    pred = lin (x);
    BASALT_ASSERT (pred.dtype () == Dtype::Float16,
                   "Autocast_Guard must cast Linear output to Float16");
  }
  Tensor l = loss::mse_loss (pred.to (Dtype::Float32), target);
  scaler.scale (l).backward ();
  bool stepped = scaler.step (opt);
  BASALT_ASSERT (stepped, "Grad_Scaler should step when gradients are finite");
  scaler.update ();

  /* Inject Inf into gradient and verify Grad_Scaler skips step and backs off
     scale factor.  */
  double prev_scale = scaler.get_scale ();
  opt.zero_grad ();
  Tensor l2 = loss::mse_loss (lin (x), target);
  scaler.scale (l2).backward ();
  lin.weight.grad ().write_scalar_at (
    0, std::numeric_limits<double>::infinity ());
  bool stepped_inf = scaler.step (opt);
  BASALT_ASSERT (!stepped_inf, "Grad_Scaler must skip step on Inf gradient");
  scaler.update ();
  BASALT_ASSERT (std::fabs (scaler.get_scale () - prev_scale * 0.5) < 1e-6,
                 "Grad_Scaler must back off scale after Inf gradient");
}

} /* namespace */

int
main (void)
{
  test_dataloader_determinism_and_prefetch ();
  test_bellows_tabular_and_posix_mmap_dataset ();
  test_mixed_precision_and_grad_scaler ();
  return 0;
}
