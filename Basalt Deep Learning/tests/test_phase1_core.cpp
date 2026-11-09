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

#include <basalt/config.hpp>
#include <basalt/device.hpp>
#include <basalt/dtype.hpp>
#include <basalt/error.hpp>
#include <basalt/random.hpp>
#include <basalt/version.h>
#include "bellows-compat/bellows_bridge.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace
{

void
test_version_and_diagnostics (void)
{
  BASALT_ASSERT (std::strcmp (basalt_version_string (), "0.1.0") == 0,
                 "version string mismatch");
  BASALT_ASSERT (std::strcmp (basalt_version_banner (),
                              "Basalt Deep Learning 0.1.0") == 0,
                 "version banner mismatch");
  BASALT_ASSERT (std::strcmp (basalt::version_string (), "0.1.0") == 0,
                 "C++ version string mismatch");

  bool caught = false;
  try
    {
      BASALT_ASSERT_SHAPE (false, "expected shape failure");
    }
  catch (const basalt::Shape_Error &err)
    {
      caught = true;
      std::string msg = err.what ();
      BASALT_ASSERT (msg.find ("basalt: test_phase1_core.cpp:") == 0,
                     "diagnostic prefix format mismatch: " + msg);
    }
  BASALT_ASSERT (caught, "Shape_Error was not thrown");
}

void
test_dtypes_and_half_precision (void)
{
  using basalt::BFloat16;
  using basalt::Dtype;
  using basalt::Float16;

  Float16 h1 (1.5f);
  BASALT_ASSERT (std::fabs (static_cast<float> (h1) - 1.5f) < 1e-6f,
                 "Float16 round-trip failed");
  Float16 h_neg (-0.125f);
  BASALT_ASSERT (std::fabs (static_cast<float> (h_neg) + 0.125f) < 1e-6f,
                 "Float16 negative round-trip failed");

  BFloat16 bf1 (3.25f);
  BASALT_ASSERT (std::fabs (static_cast<float> (bf1) - 3.25f) < 1e-2f,
                 "BFloat16 round-trip failed");

  BASALT_ASSERT (basalt::dtype_size (Dtype::Float16) == 2, "f16 size");
  BASALT_ASSERT (basalt::dtype_size (Dtype::BFloat16) == 2, "bf16 size");
  BASALT_ASSERT (basalt::dtype_size (Dtype::Float32) == 4, "f32 size");
  BASALT_ASSERT (basalt::dtype_size (Dtype::Float64) == 8, "f64 size");
  BASALT_ASSERT (basalt::promote_dtypes (Dtype::Float16, Dtype::BFloat16)
                 == Dtype::Float32,
                 "f16 + bf16 should promote to f32");
  BASALT_ASSERT (basalt::dtype_from_name ("bf16") == Dtype::BFloat16,
                 "dtype_from_name failed");
}

void
test_device_and_bellows_storage (void)
{
  using basalt::Device;
  using basalt::bellows_compat::Bellows_Storage;

  Device cpu = Device::parse ("cpu");
  Device cuda0 = Device::parse ("cuda:0");
  BASALT_ASSERT (cpu.is_cpu (), "expected cpu");
  BASALT_ASSERT (cuda0.is_cuda () && cuda0.index () == 0, "expected cuda:0");
  BASALT_ASSERT (basalt::cuda_is_available (), "expected CUDA availability");

  Bellows_Storage s_cpu (64, cpu);
  Bellows_Storage s_gpu (64, cuda0);
  float payload[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float readback[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  s_cpu.copy_from_host (payload, 0, sizeof (payload));
  s_gpu.copy_from_host (payload, 0, sizeof (payload));
  s_gpu.copy_to_host (readback, 0, sizeof (readback));
  for (size_t i = 0; i < 4; ++i)
    {
      BASALT_ASSERT (readback[i] == payload[i], "CUDA storage round-trip");
    }
}

void
test_deterministic_random (void)
{
  basalt::seed_everything (12345ULL);
  double a1 = basalt::default_generator ().uniform (0.0, 1.0);
  double a2 = basalt::default_generator ().normal (0.0, 1.0);
  std::vector<size_t> p1 = basalt::default_generator ().permutation (16);

  basalt::seed_everything (12345ULL);
  double b1 = basalt::default_generator ().uniform (0.0, 1.0);
  double b2 = basalt::default_generator ().normal (0.0, 1.0);
  std::vector<size_t> p2 = basalt::default_generator ().permutation (16);

  BASALT_ASSERT (a1 == b1 && a2 == b2 && p1 == p2,
                 "seed_everything must produce deterministic sequences");
}

} /* namespace */

int
main (void)
{
  test_version_and_diagnostics ();
  test_dtypes_and_half_precision ();
  test_device_and_bellows_storage ();
  test_deterministic_random ();
  return 0;
}
