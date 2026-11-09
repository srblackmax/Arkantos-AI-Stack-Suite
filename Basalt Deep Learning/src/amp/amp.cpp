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
#include <basalt/amp.hpp>
#include <basalt/autograd.hpp>
#include <basalt/error.hpp>
#include <basalt/ops.hpp>
#include "bellows-compat/bellows_bridge.hpp"
#include <cmath>

namespace basalt::amp
{

namespace
{

bool &
tls_autocast_enabled (void) noexcept
{
  thread_local bool en = false;
  return en;
}

Dtype &
tls_autocast_dtype (void) noexcept
{
  thread_local Dtype dt = Dtype::Float16;
  return dt;
}

} /* namespace */

bool
Autocast_Mode::is_enabled (void) noexcept
{
  return tls_autocast_enabled ();
}

void
Autocast_Mode::set_enabled (bool enabled) noexcept
{
  tls_autocast_enabled () = enabled;
}

Dtype
Autocast_Mode::autocast_dtype (void) noexcept
{
  return tls_autocast_dtype ();
}

void
Autocast_Mode::set_autocast_dtype (Dtype dtype) noexcept
{
  tls_autocast_dtype () = dtype;
}

Autocast_Guard::Autocast_Guard (Dtype dtype, bool enabled) noexcept
  : prev_enabled_ (Autocast_Mode::is_enabled ()),
    prev_dtype_ (Autocast_Mode::autocast_dtype ())
{
  Autocast_Mode::set_autocast_dtype (dtype);
  Autocast_Mode::set_enabled (enabled);
}

Autocast_Guard::~Autocast_Guard (void) noexcept
{
  Autocast_Mode::set_enabled (prev_enabled_);
  Autocast_Mode::set_autocast_dtype (prev_dtype_);
}

Tensor
autocast_cast (const Tensor &t)
{
  if (!t.defined () || !Autocast_Mode::is_enabled ()
      || !is_floating_point (t.dtype ()))
    {
      return t;
    }
  Dtype target = Autocast_Mode::autocast_dtype ();
  if (t.dtype () == target)
    {
      return t;
    }
  return t.to (target);
}

Grad_Scaler::Grad_Scaler (double init_scale,
                          double growth_factor,
                          double backoff_factor,
                          size_t growth_interval,
                          bool enabled)
  : scale_ (init_scale),
    growth_factor_ (growth_factor),
    backoff_factor_ (backoff_factor),
    growth_interval_ (growth_interval),
    growth_tracker_ (0),
    enabled_ (enabled),
    unscaled_ (false),
    found_inf_ (false)
{
  BASALT_ASSERT_ARG (init_scale > 0.0 && growth_factor > 1.0
                     && backoff_factor > 0.0 && backoff_factor < 1.0,
                     "Grad_Scaler invalid scaling factors");
}

Tensor
Grad_Scaler::scale (const Tensor &loss) const
{
  if (!enabled_)
    {
      return loss;
    }
  return ops::mul_scalar (loss, scale_);
}

void
Grad_Scaler::unscale_ (optim::Optimizer &optimizer)
{
  if (!enabled_ || unscaled_)
    {
      return;
    }
  No_Grad_Guard no_grad;
  unscaled_ = true;
  found_inf_ = false;
  double inv_scale = 1.0 / scale_;

  for (const Tensor &p : optimizer.parameters ())
    {
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      size_t n = g.numel ();
      if (g.dtype () == Dtype::Float32 && g.is_contiguous () && n > 0)
        {
          if (g.device ().is_cuda ())
            {
              bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
            }
          float *gp = g.data_ptr<float> ();
          float finv = static_cast<float> (inv_scale);
          for (size_t i = 0; i < n; ++i)
            {
              float v = gp[i] * finv;
              if (!std::isfinite (v))
                {
                  found_inf_ = true;
                }
              gp[i] = v;
            }
          continue;
        }
      for (size_t i = 0; i < n; ++i)
        {
          double v = g.read_scalar_at (i) * inv_scale;
          if (!std::isfinite (v))
            {
              found_inf_ = true;
            }
          g.write_scalar_at (i, v);
        }
    }
}

bool
Grad_Scaler::step (optim::Optimizer &optimizer)
{
  if (!enabled_)
    {
      optimizer.step ();
      return true;
    }
  if (!unscaled_)
    {
      unscale_ (optimizer);
    }
  if (found_inf_)
    {
      return false;
    }
  optimizer.step ();
  return true;
}

void
Grad_Scaler::update (void)
{
  if (!enabled_)
    {
      return;
    }
  if (found_inf_)
    {
      scale_ *= backoff_factor_;
      growth_tracker_ = 0;
    }
  else
    {
      ++growth_tracker_;
      if (growth_tracker_ >= growth_interval_)
        {
          scale_ *= growth_factor_;
          growth_tracker_ = 0;
        }
    }
  unscaled_ = false;
  found_inf_ = false;
}

} /* namespace basalt::amp */
