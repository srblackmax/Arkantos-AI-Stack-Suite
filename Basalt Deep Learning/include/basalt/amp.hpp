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

#ifndef BASALT_AMP_HPP
#define BASALT_AMP_HPP

/* Automatic Mixed Precision (fp16 and bf16) execution scopes and dynamic
   gradient loss scaling (Grad_Scaler).  */

#include <cstddef>
#include <basalt/config.hpp>
#include <basalt/dtype.hpp>
#include <basalt/optim.hpp>
#include <basalt/tensor.hpp>

namespace basalt::amp
{

/* Thread-local automatic mixed-precision state.  */
class BASALT_API Autocast_Mode
{
public:
  static bool
  is_enabled (void) noexcept;

  static void
  set_enabled (bool enabled) noexcept;

  static Dtype
  autocast_dtype (void) noexcept;

  static void
  set_autocast_dtype (Dtype dtype) noexcept;
};

/* RAII guard enabling fp16 or bf16 automatic mixed precision within a
   lexical block.  */
class BASALT_API Autocast_Guard
{
public:
  explicit Autocast_Guard (Dtype dtype = Dtype::Float16,
                           bool enabled = true) noexcept;
  ~Autocast_Guard (void) noexcept;

  Autocast_Guard (const Autocast_Guard &) = delete;
  Autocast_Guard &operator= (const Autocast_Guard &) = delete;

private:
  bool prev_enabled_;
  Dtype prev_dtype_;
};

using AutocastGuard = Autocast_Guard;

/* Cast floating-point tensor T to the active autocast dtype if Autocast_Mode
   is enabled; returns T unchanged otherwise.  */
BASALT_API Tensor
autocast_cast (const Tensor &t);

/* Dynamic loss scaler that prevents fp16/bf16 gradient underflow and skips
   optimizer steps when Inf/NaN gradients are detected.  */
class BASALT_API Grad_Scaler
{
public:
  explicit Grad_Scaler (double init_scale = 65536.0,
                        double growth_factor = 2.0,
                        double backoff_factor = 0.5,
                        size_t growth_interval = 2000,
                        bool enabled = true);

  Tensor
  scale (const Tensor &loss) const;

  void
  unscale_ (optim::Optimizer &optimizer);

  bool
  step (optim::Optimizer &optimizer);

  void
  update (void);

  double
  get_scale (void) const noexcept
  {
    return scale_;
  }

  bool
  is_enabled (void) const noexcept
  {
    return enabled_;
  }

  bool
  found_inf_or_nan (void) const noexcept
  {
    return found_inf_;
  }

private:
  double scale_;
  double growth_factor_;
  double backoff_factor_;
  size_t growth_interval_;
  size_t growth_tracker_ = 0;
  bool enabled_;
  bool unscaled_ = false;
  bool found_inf_ = false;
};

using GradScaler = Grad_Scaler;

} /* namespace basalt::amp */

namespace basalt
{
using amp::Autocast_Guard;
using amp::AutocastGuard;
using amp::Grad_Scaler;
using amp::GradScaler;
} /* namespace basalt */

#endif /* BASALT_AMP_HPP */
