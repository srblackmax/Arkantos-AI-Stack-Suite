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

#include <sutra/sutra.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace
{

constexpr float kFp32EquivalenceTolerance = 1.0e-6F;

#define SUTRA_ASSERT(cond)                                                    \
  do                                                                          \
    {                                                                         \
      if (!(cond))                                                            \
        {                                                                     \
          std::cerr << __FILE__ << ":" << __LINE__                            \
                    << ": assertion failed: " #cond "\n";                     \
          std::abort ();                                                      \
        }                                                                     \
    }                                                                         \
  while (false)

void
test_simd_vs_scalar_equivalence ()
{
  std::mt19937 rng (0x12345678U);
  std::uniform_real_distribution<float> dist (-1.0F, 1.0F);

  const std::array<std::size_t, 9U> dims{ 1U,   7U,   16U,   64U,  128U,
                                          256U, 512U, 1024U, 1536U };

  for (const std::size_t dim : dims)
    {
      for (int trial = 0; trial < 25; ++trial)
        {
          std::vector<float> a (dim);
          std::vector<float> b (dim);
          for (std::size_t i = 0U; i < dim; ++i)
            {
              a.at (i) = dist (rng);
              b.at (i) = dist (rng);
            }

          const float ref_val = sutra::simd_dot_f32_scalar (a, b);
          const float simd_val = sutra::simd_dot_f32 (a, b);
          const float abs_err = std::fabs (ref_val - simd_val);
          SUTRA_ASSERT (abs_err <= kFp32EquivalenceTolerance);
        }
    }
}

void
test_gpu_vs_cpu_equivalence ()
{
  if (!sutra::is_cuda_available ())
    {
      std::cout << "CUDA GPU not detected; skipping GPU kernel equivalence "
                   "check.\n";
      return;
    }

  std::mt19937 rng (0x87654321U);
  std::uniform_real_distribution<float> dist (-1.0F, 1.0F);

  constexpr std::size_t kNumRows = 256U;
  constexpr std::size_t kDim = 128U;

  std::vector<float> host_matrix (kNumRows * kDim);
  std::vector<float> host_query (kDim);
  for (float &v : host_matrix)
    {
      v = dist (rng);
    }
  for (float &v : host_query)
    {
      v = dist (rng);
    }

  auto stream_res = sutra::Stream::create ();
  SUTRA_ASSERT (stream_res.has_value ());
  auto dev_mat = sutra::Tensor::allocate (kNumRows * kDim);
  auto dev_qry = sutra::Tensor::allocate (kDim);
  auto dev_out = sutra::Tensor::allocate (kNumRows);
  SUTRA_ASSERT (dev_mat.has_value () && dev_qry.has_value ()
                && dev_out.has_value ());

  SUTRA_ASSERT (
      dev_mat->copy_from_host (host_matrix, *stream_res).has_value ());
  SUTRA_ASSERT (
      dev_qry->copy_from_host (host_query, *stream_res).has_value ());
  SUTRA_ASSERT (sutra::gpu_batch_dot_f32 (*dev_mat, *dev_qry, *dev_out,
                                          kNumRows, kDim, *stream_res)
                    .has_value ());

  std::vector<float> gpu_scores (kNumRows, 0.0F);
  SUTRA_ASSERT (dev_out->copy_to_host (gpu_scores, *stream_res).has_value ());
  SUTRA_ASSERT (stream_res->synchronize ().has_value ());

  const std::span<const float> mat_span (host_matrix.data (),
                                         host_matrix.size ());
  for (std::size_t r = 0U; r < kNumRows; ++r)
    {
      const float cpu_ref = sutra::simd_dot_f32_scalar (
          mat_span.subspan (r * kDim, kDim), host_query);
      const float err = std::fabs (cpu_ref - gpu_scores.at (r));
      SUTRA_ASSERT (err <= kFp32EquivalenceTolerance);
    }

  // Verify SemanticMemory::search equivalence between CPU and GPU paths.
  auto sem_res = sutra::SemanticMemory::create (kDim);
  SUTRA_ASSERT (sem_res.has_value ());
  for (std::size_t r = 0U; r < 64U; ++r)
    {
      SUTRA_ASSERT (sem_res
                        ->add_document ("doc_" + std::to_string (r),
                                        "payload_" + std::to_string (r),
                                        mat_span.subspan (r * kDim, kDim))
                        .has_value ());
    }

  const auto cpu_top = sem_res->search (host_query, 10U, /*prefer_gpu=*/false);
  const auto gpu_top = sem_res->search (host_query, 10U, /*prefer_gpu=*/true);
  SUTRA_ASSERT (cpu_top.has_value () && gpu_top.has_value ());
  SUTRA_ASSERT (cpu_top->size () == gpu_top->size ());

  for (std::size_t i = 0U; i < cpu_top->size (); ++i)
    {
      SUTRA_ASSERT (cpu_top->at (i).id == gpu_top->at (i).id);
      const float score_diff
          = std::fabs (cpu_top->at (i).score - gpu_top->at (i).score);
      SUTRA_ASSERT (score_diff <= kFp32EquivalenceTolerance);
    }
}

} /* namespace */

auto
main () -> int
{
  test_simd_vs_scalar_equivalence ();
  test_gpu_vs_cpu_equivalence ();
  std::cout << "All Sutra numerical and GPU/CPU equivalence tests PASSED.\n";
  return 0;
}
