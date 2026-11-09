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

#ifndef BASALT_RANDOM_HPP
#define BASALT_RANDOM_HPP

/* Deterministic pseudo-random number generation built directly upon the
   SplitMix64 + xoshiro256** engine provided by Bellows / Hyper Math.
   Guarantees bit-identical sequences across backends and thread counts when
   seeded via seed_everything.  */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <basalt/config.hpp>

namespace basalt
{

/* Thread-safe deterministic PRNG wrapper around hyper_math::random_engine.  */
class BASALT_API Random_Generator
{
public:
  explicit Random_Generator (uint64_t seed = 0x853C49E6748FEA9BULL) noexcept;
  ~Random_Generator (void) noexcept = default;

  Random_Generator (const Random_Generator &) noexcept = default;
  Random_Generator &operator= (const Random_Generator &) noexcept = default;

  /* Reset the generator state from SEED.  */
  void
  seed (uint64_t new_seed) noexcept;

  /* Return the most recently configured seed value.  */
  uint64_t
  current_seed (void) const noexcept
  {
    return seed_;
  }

  /* Advance and return the next 64-bit unsigned random word.  */
  uint64_t
  next_u64 (void) noexcept;

  /* Sample a uniform double in [low, high).  */
  double
  uniform (double low = 0.0, double high = 1.0) noexcept;

  /* Sample a Gaussian normal double with mean MU and standard deviation
     SIGMA using the Marsaglia polar method.  */
  double
  normal (double mu = 0.0, double sigma = 1.0) noexcept;

  /* Sample an integer in [low, high).  */
  int64_t
  randint (int64_t low, int64_t high);

  /* Generate a Fisher-Yates permutation of [0, n).  */
  std::vector<size_t>
  permutation (size_t n) noexcept;

  /* Deterministically fork an independent child generator for worker ID.  */
  Random_Generator
  fork (size_t stream_id) const noexcept;

private:
  uint64_t seed_;
  uint64_t state_[4];
  bool has_spare_;
  double spare_;
};

/* Return a reference to the calling thread's default Random_Generator.  */
BASALT_API Random_Generator &
default_generator (void) noexcept;

/* Seed the global base seed and the calling thread's default generator.
   Returns the applied seed.  */
BASALT_API uint64_t
seed_everything (uint64_t seed) noexcept;

/* Return the most recently configured global seed.  */
BASALT_API uint64_t
initial_seed (void) noexcept;

} /* namespace basalt */

#endif /* BASALT_RANDOM_HPP */
