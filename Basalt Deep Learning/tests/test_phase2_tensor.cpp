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

#include <basalt/device.hpp>
#include <basalt/dtype.hpp>
#include <basalt/error.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>
#include <basalt/tensor.hpp>
#include <bellows/core/data_table.hpp>
#include <cmath>
#include <vector>

namespace
{

void
test_strided_views_and_broadcasting (void)
{
  using namespace basalt;

  Tensor a = arange (0.0, 12.0, 1.0).reshape ({3, 4});
  BASALT_ASSERT (a.is_contiguous (), "expected contiguous 3x4");

  Tensor at = a.t ();
  BASALT_ASSERT (!at.is_contiguous (), "transposed 3x4 should not be contig");
  BASALT_ASSERT (at.size (0) == 4 && at.size (1) == 3, "t() shape");
  BASALT_ASSERT (at.read_scalar_at (1 * 3 + 2) == a.read_scalar_at (2 * 4 + 1),
                 "transpose element mapping");

  /* Strided slice along rows and cols.  */
  Tensor sub = a.slice (0, 0, 3, 2).slice (1, 1, 4, 1);
  BASALT_ASSERT (sub.size (0) == 2 && sub.size (1) == 3, "slice shape");
  BASALT_ASSERT (sub.read_scalar_at (0) == 1.0, "slice[0,0]");
  BASALT_ASSERT (sub.read_scalar_at (3) == 9.0, "slice[1,0]");

  /* Broadcasting (3, 1) + (1, 4) -> (3, 4).  */
  Tensor col = tensor ({10.0, 20.0, 30.0}).reshape ({3, 1});
  Tensor row = tensor ({1.0, 2.0, 3.0, 4.0}).reshape ({1, 4});
  Tensor sum_b = col + row;
  BASALT_ASSERT (sum_b.size (0) == 3 && sum_b.size (1) == 4, "bcast shape");
  BASALT_ASSERT (sum_b.read_scalar_at (2 * 4 + 3) == 34.0, "bcast value");
}

void
test_multi_dtype_and_bellows_bridge (void)
{
  using namespace basalt;

  for (Dtype dt : {Dtype::Float16, Dtype::BFloat16,
                   Dtype::Float32, Dtype::Float64})
    {
      Tensor x = ones ({2, 3}, Tensor_Options ().with_dtype (dt));
      Tensor y = full ({2, 3}, 2.5, Tensor_Options ().with_dtype (dt));
      Tensor z = x + y;
      BASALT_ASSERT (z.dtype () == dt, "dtype preserved");
      BASALT_ASSERT (std::fabs (z.read_scalar_at (0) - 3.5) < 5e-2,
                     "half/full precision addition");
    }

  /* Bridge to and from bellows::data_table.  */
  Tensor src = arange (1.0, 7.0, 1.0).reshape ({2, 3});
  bellows::data_table tbl = to_bellows_table (src);
  BASALT_ASSERT (tbl.row_count () == 2 && tbl.column_count () == 3,
                 "bellows table shape");
  Tensor roundtrip = from_bellows_table (tbl);
  BASALT_ASSERT (allclose (src, roundtrip), "bellows table round-trip");
}

void
test_cross_backend_cpu_cuda_equivalence (void)
{
  using namespace basalt;

  seed_everything (99ULL);
  Tensor a_cpu = randn ({4, 6},
                        Tensor_Options ().with_dtype (Dtype::Float64));
  Tensor b_cpu = randn ({6, 5},
                        Tensor_Options ().with_dtype (Dtype::Float64));

  Tensor a_gpu = a_cpu.cuda (0);
  Tensor b_gpu = b_cpu.cuda (0);

  Tensor c_cpu = softmax (matmul (relu (a_cpu), b_cpu), -1);
  Tensor c_gpu = softmax (matmul (relu (a_gpu), b_gpu), -1);

  BASALT_ASSERT (c_gpu.device ().is_cuda (), "expected CUDA output");
  BASALT_ASSERT (allclose (c_cpu, c_gpu.cpu (), 1e-12, 1e-12),
                 "CPU and CUDA backends must produce identical results");

  /* Verify cross-device mismatch raises Device_Error.  */
  bool caught_device = false;
  try
    {
      Tensor bad = a_cpu + a_gpu;
      (void) bad;
    }
  catch (const Device_Error &)
    {
      caught_device = true;
    }
  BASALT_ASSERT (caught_device, "expected Device_Error on mixed devices");
}

} /* namespace */

int
main (void)
{
  test_strided_views_and_broadcasting ();
  test_multi_dtype_and_bellows_bridge ();
  test_cross_backend_cpu_cuda_equivalence ();
  return 0;
}
