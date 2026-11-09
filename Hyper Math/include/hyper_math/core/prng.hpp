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

#ifndef HYPER_MATH_CORE_PRNG_HPP
#define HYPER_MATH_CORE_PRNG_HPP

#include "hyper_math/core/config.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace hyper_math
{

/* High-speed, statistically robust 256-bit state PRNG using SplitMix64 state
   initialization and the xoshiro256** scrambling recurrence, paired with
   Marsaglia polar normal sampling.  */
class random_engine
{
private:
  std::array<std::uint64_t, 4> state_;
  bool has_spare_normal_;
  double spare_normal_;

  static constexpr std::uint64_t
  splitmix64_next (std::uint64_t &x) noexcept
  {
    std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

public:
  explicit
  random_engine (std::uint64_t seed_val = 0x48595045524d4154ULL) noexcept
    : state_ {}, has_spare_normal_ (false), spare_normal_ (0.0)
  {
    seed (seed_val);
  }

  void
  seed (std::uint64_t seed_val) noexcept
  {
    std::uint64_t sm = seed_val;
    for (std::size_t i = 0; i < 4; ++i)
      {
        state_[i] = splitmix64_next (sm);
      }
    has_spare_normal_ = false;
    spare_normal_ = 0.0;
  }

  void
  get_state (std::uint64_t out_state[4],
             bool &out_has_spare,
             double &out_spare) const noexcept
  {
    for (std::size_t i = 0; i < 4; ++i)
      {
        out_state[i] = state_[i];
      }
    out_has_spare = has_spare_normal_;
    out_spare = spare_normal_;
  }

  void
  set_state (const std::uint64_t in_state[4],
             bool in_has_spare,
             double in_spare) noexcept
  {
    for (std::size_t i = 0; i < 4; ++i)
      {
        state_[i] = in_state[i];
      }
    has_spare_normal_ = in_has_spare;
    spare_normal_ = in_spare;
  }

  std::uint64_t
  next_u64 () noexcept
  {
    const std::uint64_t result =
      std::rotl (state_[1] * 5ULL, 7) * 9ULL;
    const std::uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;
    state_[3] = std::rotl (state_[3], 45);

    return result;
  }

  /* Uniform double in [0.0, 1.0) using 53 high-precision mantissa bits.  */
  double
  next_uniform01 () noexcept
  {
    return static_cast<double> (next_u64 () >> 11) * 0x1.0p-53;
  }

  double
  uniform (double low = 0.0, double high = 1.0) noexcept
  {
    return low + (high - low) * next_uniform01 ();
  }

  std::int64_t
  randint (std::int64_t low, std::int64_t high)
  {
    if (high <= low)
      {
        throw hyper_math_error ("randint requires high > low");
      }
    const std::uint64_t span =
      static_cast<std::uint64_t> (high) - static_cast<std::uint64_t> (low);
    return static_cast<std::int64_t> (
      static_cast<std::uint64_t> (low) + (next_u64 () % span));
  }

  /* Marsaglia polar method for standard normal N(mean, stddev^2) variates.  */
  double
  normal (double mean = 0.0, double stddev = 1.0) noexcept
  {
    if (has_spare_normal_)
      {
        has_spare_normal_ = false;
        return mean + stddev * spare_normal_;
      }

    double u = 0.0;
    double v = 0.0;
    double s = 0.0;
    do
      {
        u = 2.0 * next_uniform01 () - 1.0;
        v = 2.0 * next_uniform01 () - 1.0;
        s = u * u + v * v;
      }
    while (s >= 1.0 || s == 0.0);

    const double mul = std::sqrt (-2.0 * std::log (s) / s);
    spare_normal_ = v * mul;
    has_spare_normal_ = true;
    return mean + stddev * (u * mul);
  }

  /* Generate random permutation of [0, n).  */
  std::vector<std::size_t>
  permutation (std::size_t n)
  {
    std::vector<std::size_t> idx (n);
    std::iota (idx.begin (), idx.end (), 0);
    for (std::size_t i = n; i > 1; --i)
      {
        const std::size_t j = static_cast<std::size_t> (next_u64 () % i);
        std::swap (idx[i - 1], idx[j]);
      }
    return idx;
  }
};

inline random_engine &
global_rng () noexcept
{
  static thread_local random_engine instance (0x48595045524d4154ULL);
  return instance;
}

} // namespace hyper_math

#endif /* HYPER_MATH_CORE_PRNG_HPP */
