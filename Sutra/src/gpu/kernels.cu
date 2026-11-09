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

#include "gpu/kernels.hpp"

#include <cuda_runtime.h>

namespace sutra::inline v1::detail
{

namespace
{

constexpr unsigned int kThreadsPerBlock = 256U;

/* Computes one row dot product per thread in double accumulation precision
   to guarantee <= 1e-6 fp32 numerical equivalence with CPU reference.  */
__global__ void
batch_dot_f32_kernel (const float *__restrict__ matrix,
                      const float *__restrict__ query,
                      float *__restrict__ out_scores, std::size_t num_rows,
                      std::size_t dim)
{
  const std::size_t row
      = static_cast<std::size_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows)
    {
      return;
    }

  const float *const row_ptr = matrix + row * dim;
  double acc = 0.0;
  for (std::size_t k = 0U; k < dim; ++k)
    {
      acc += static_cast<double> (row_ptr[k]) * static_cast<double> (query[k]);
    }
  out_scores[row] = static_cast<float> (acc);
}

} /* namespace */

/* Launches batch_dot_f32_kernel on the provided CUDA stream.  */
auto
launch_batch_dot_f32_kernel (const float *matrix, const float *query,
                             float *out_scores, std::size_t num_rows,
                             std::size_t dim,
                             void *stream_handle) noexcept -> int
{
  if (num_rows == 0U || dim == 0U)
    {
      return static_cast<int> (cudaSuccess);
    }

  const auto grid_x = static_cast<unsigned int> (
      (num_rows + kThreadsPerBlock - 1U) / kThreadsPerBlock);
  auto *const stream = static_cast<cudaStream_t> (stream_handle);

  batch_dot_f32_kernel<<<grid_x, kThreadsPerBlock, 0U, stream>>> (
      matrix, query, out_scores, num_rows, dim);

  return static_cast<int> (cudaGetLastError ());
}

} /* namespace sutra::inline v1::detail */
