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

#include "config.h"
#include <basalt/random.hpp>
#include <basalt/error.hpp>
#include <hyper_math/core/prng.hpp>
#include <atomic>
#include <new>

namespace basalt
{

namespace
{

std::atomic<uint64_t> &
global_seed_slot (void) noexcept
{
  static std::atomic<uint64_t> s{0x853C49E6748FEA9BULL};
  return s;
}

std::atomic<uint64_t> &
global_epoch_counter (void) noexcept
{
  static std::atomic<uint64_t> c{1ULL};
  return c;
}

} /* namespace */

Random_Generator::Random_Generator (uint64_t initial) noexcept
  : seed_ (initial),
    state_{0, 0, 0, 0},
    has_spare_ (false),
    spare_ (0.0)
{
  seed (initial);
}

void
Random_Generator::seed (uint64_t new_seed) noexcept
{
  seed_ = new_seed;
  hyper_math::random_engine eng (new_seed);
  eng.get_state (state_, has_spare_, spare_);
}

uint64_t
Random_Generator::next_u64 (void) noexcept
{
  hyper_math::random_engine eng (0);
  eng.set_state (state_, has_spare_, spare_);
  uint64_t val = eng.next_u64 ();
  eng.get_state (state_, has_spare_, spare_);
  return val;
}

double
Random_Generator::uniform (double low, double high) noexcept
{
  hyper_math::random_engine eng (0);
  eng.set_state (state_, has_spare_, spare_);
  double val = eng.uniform (low, high);
  eng.get_state (state_, has_spare_, spare_);
  return val;
}

double
Random_Generator::normal (double mu, double sigma) noexcept
{
  hyper_math::random_engine eng (0);
  eng.set_state (state_, has_spare_, spare_);
  double val = eng.normal (mu, sigma);
  eng.get_state (state_, has_spare_, spare_);
  return val;
}

int64_t
Random_Generator::randint (int64_t low, int64_t high)
{
  BASALT_ASSERT_ARG (low < high, "randint requires low < high");
  hyper_math::random_engine eng (0);
  eng.set_state (state_, has_spare_, spare_);
  int64_t val = eng.randint (low, high);
  eng.get_state (state_, has_spare_, spare_);
  return val;
}

std::vector<size_t>
Random_Generator::permutation (size_t n) noexcept
{
  hyper_math::random_engine eng (0);
  eng.set_state (state_, has_spare_, spare_);
  std::vector<size_t> perm = eng.permutation (n);
  eng.get_state (state_, has_spare_, spare_);
  return perm;
}

Random_Generator
Random_Generator::fork (size_t stream_id) const noexcept
{
  uint64_t mixed = seed_ ^ (static_cast<uint64_t> (stream_id + 1U)
                            * 0x9E3779B97F4A7C15ULL);
  return Random_Generator (mixed);
}

Random_Generator &
default_generator (void) noexcept
{
  thread_local Random_Generator tls_gen (global_seed_slot ().load ());
  thread_local uint64_t tls_epoch = 0;
  uint64_t cur_epoch
    = global_epoch_counter ().load (std::memory_order_acquire);
  if (tls_epoch != cur_epoch)
    {
      tls_gen.seed (global_seed_slot ().load (std::memory_order_relaxed));
      tls_epoch = cur_epoch;
    }
  return tls_gen;
}

uint64_t
seed_everything (uint64_t seed) noexcept
{
  global_seed_slot ().store (seed, std::memory_order_relaxed);
  global_epoch_counter ().fetch_add (1ULL, std::memory_order_release);
  default_generator ().seed (seed);
  return seed;
}

uint64_t
initial_seed (void) noexcept
{
  return global_seed_slot ().load (std::memory_order_relaxed);
}

} /* namespace basalt */
