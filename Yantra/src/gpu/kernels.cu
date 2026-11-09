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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace yantra::inline v1::gpu_detail
{

namespace
{

__global__ void
batch_l2_distance_kernel (const float *__restrict__ vectors,
                          const float *__restrict__ queries,
                          float *__restrict__ out_dists,
                          std::size_t num_vectors, std::size_t num_queries,
                          std::size_t dim)
{
  const std::size_t q_idx = blockIdx.y;
  const std::size_t v_idx
      = static_cast<std::size_t> (blockIdx.x) * blockDim.y + threadIdx.y;
  if (q_idx >= num_queries || v_idx >= num_vectors)
    {
      return;
    }

  const float *v_ptr = vectors + v_idx * dim;
  const float *q_ptr = queries + q_idx * dim;
  const unsigned int lane = threadIdx.x;

  float sum = 0.0F;
  for (std::size_t d = lane; d < dim; d += 32U)
    {
      const float diff = v_ptr[d] - q_ptr[d];
      sum += diff * diff;
    }

  for (int offset = 16; offset > 0; offset /= 2)
    {
      sum += __shfl_down_sync (0xFFFFFFFFU, sum, offset);
    }

  if (lane == 0U)
    {
      out_dists[q_idx * num_vectors + v_idx] = sum;
    }
}

__global__ void
cagra_beam_search_kernel (const float *__restrict__ vectors,
                          const uint32_t *__restrict__ graph_neighbors,
                          const float *__restrict__ queries,
                          uint64_t *__restrict__ out_ids,
                          float *__restrict__ out_dists,
                          std::size_t num_vectors, std::size_t num_queries,
                          std::size_t dim, std::size_t degree,
                          std::size_t k_out)
{
  const std::size_t q_idx = blockIdx.x;
  if (q_idx >= num_queries)
    {
      return;
    }

  const float *q_ptr = queries + q_idx * dim;
  const unsigned int lane = threadIdx.x;

  constexpr int kBeamSize = 32;
  __shared__ uint32_t beam_ids[kBeamSize];
  __shared__ float beam_dists[kBeamSize];

  auto warp_l2 = [&] (uint32_t node_id) -> float {
    const float *v_ptr = vectors + static_cast<std::size_t> (node_id) * dim;
    float s = 0.0F;
    for (std::size_t d = lane; d < dim; d += 32U)
      {
        const float diff = v_ptr[d] - q_ptr[d];
        s += diff * diff;
      }
    for (int off = 16; off > 0; off /= 2)
      {
        s += __shfl_down_sync (0xFFFFFFFFU, s, off);
      }
    return __shfl_sync (0xFFFFFFFFU, s, 0);
  };

  const int init_seeds
      = (static_cast<int> (num_vectors) < kBeamSize)
            ? static_cast<int> (num_vectors)
            : kBeamSize;

  for (int i = 0; i < kBeamSize; ++i)
    {
      if (i < init_seeds)
        {
          const auto seed_id = static_cast<uint32_t> (
              (static_cast<std::size_t> (i) * num_vectors)
              / static_cast<std::size_t> (init_seeds));
          const float d = warp_l2 (seed_id);
          if (lane == 0U)
            {
              beam_ids[i] = seed_id;
              beam_dists[i] = d;
            }
        }
      else if (lane == 0U)
        {
          beam_ids[i] = UINT32_MAX;
          beam_dists[i] = 1.0e30F;
        }
    }
  __syncwarp ();

  for (int step = 0; step < 12; ++step)
    {
      bool improved = false;
      for (int b = 0; b < 8 && b < init_seeds; ++b)
        {
          const uint32_t curr = beam_ids[b];
          if (curr >= num_vectors)
            {
              continue;
            }
          const uint32_t *nbrs
              = graph_neighbors + static_cast<std::size_t> (curr) * degree;
          for (std::size_t deg = 0U; deg < degree; ++deg)
            {
              const uint32_t cand = nbrs[deg];
              if (cand >= num_vectors)
                {
                  break;
                }
              bool seen = false;
              if (lane < static_cast<unsigned int> (kBeamSize))
                {
                  seen = (beam_ids[lane] == cand);
                }
              if (__any_sync (0xFFFFFFFFU, seen))
                {
                  continue;
                }

              const float d = warp_l2 (cand);
              if (lane == 0U)
                {
                  int worst_idx = 0;
                  float worst_dist = beam_dists[0];
                  for (int j = 1; j < kBeamSize; ++j)
                    {
                      if (beam_dists[j] > worst_dist)
                        {
                          worst_dist = beam_dists[j];
                          worst_idx = j;
                        }
                    }
                  if (d < worst_dist)
                    {
                      beam_ids[worst_idx] = cand;
                      beam_dists[worst_idx] = d;
                      improved = true;
                    }
                }
              __syncwarp ();
            }
        }
      const int any_imp
          = __shfl_sync (0xFFFFFFFFU, improved ? 1 : 0, 0);
      if (any_imp == 0)
        {
          break;
        }
    }

  if (lane == 0U)
    {
      for (int i = 0; i < kBeamSize - 1; ++i)
        {
          for (int j = i + 1; j < kBeamSize; ++j)
            {
              if (beam_dists[j] < beam_dists[i])
                {
                  const float td = beam_dists[i];
                  beam_dists[i] = beam_dists[j];
                  beam_dists[j] = td;
                  const uint32_t ti = beam_ids[i];
                  beam_ids[i] = beam_ids[j];
                  beam_ids[j] = ti;
                }
            }
        }
      for (std::size_t r = 0U; r < k_out; ++r)
        {
          if (r < static_cast<std::size_t> (kBeamSize)
              && beam_ids[r] != UINT32_MAX)
            {
              out_ids[q_idx * k_out + r] = beam_ids[r];
              out_dists[q_idx * k_out + r] = beam_dists[r];
            }
          else
            {
              out_ids[q_idx * k_out + r] = 0U;
              out_dists[q_idx * k_out + r] = 1.0e30F;
            }
        }
    }
}

} /* namespace */

auto
launch_batch_l2_f32 (const float *d_vectors, const float *d_queries,
                     float *d_out_dists, std::size_t num_vectors,
                     std::size_t num_queries, std::size_t dim,
                     cudaStream_t stream) noexcept -> cudaError_t
{
  const dim3 block (32U, 8U, 1U);
  const dim3 grid (static_cast<unsigned int> ((num_vectors + 7U) / 8U),
                   static_cast<unsigned int> (num_queries), 1U);
  batch_l2_distance_kernel<<<grid, block, 0U, stream>>> (
      d_vectors, d_queries, d_out_dists, num_vectors, num_queries, dim);
  return cudaGetLastError ();
}

auto
launch_cagra_search (const float *d_vectors, const uint32_t *d_neighbors,
                     const float *d_queries, uint64_t *d_out_ids,
                     float *d_out_dists, std::size_t num_vectors,
                     std::size_t num_queries, std::size_t dim,
                     std::size_t degree, std::size_t k_out,
                     cudaStream_t stream) noexcept -> cudaError_t
{
  const dim3 block (32U, 1U, 1U);
  const dim3 grid (static_cast<unsigned int> (num_queries), 1U, 1U);
  cagra_beam_search_kernel<<<grid, block, 0U, stream>>> (
      d_vectors, d_neighbors, d_queries, d_out_ids, d_out_dists, num_vectors,
      num_queries, dim, degree, k_out);
  return cudaGetLastError ();
}

} /* namespace yantra::inline v1::gpu_detail */
