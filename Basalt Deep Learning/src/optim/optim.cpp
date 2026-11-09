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
#include <basalt/optim.hpp>
#include <basalt/autograd.hpp>
#include <basalt/error.hpp>
#include "bellows-compat/bellows_bridge.hpp"
#include <algorithm>
#include <cmath>

namespace basalt::optim
{

Optimizer::Optimizer (std::vector<Tensor> params, double lr)
  : params_ (std::move (params)),
    lr_ (lr),
    step_count_ (0)
{
  BASALT_ASSERT_ARG (lr >= 0.0, "learning rate must be non-negative");
}

Optimizer::~Optimizer (void) noexcept = default;

void
Optimizer::zero_grad (void)
{
  for (auto &p : params_)
    {
      if (p.defined ())
        {
          p.zero_grad ();
        }
    }
}

std::map<std::string, Tensor>
Optimizer::state_dict (void) const
{
  std::map<std::string, Tensor> out = state_buffers_;
  out["__lr__"] = scalar_tensor (lr_);
  out["__step__"] = scalar_tensor (static_cast<double> (step_count_));
  return out;
}

void
Optimizer::load_state_dict (const std::map<std::string, Tensor> &state)
{
  No_Grad_Guard no_grad;
  for (const auto &kv : state)
    {
      if (kv.first == "__lr__")
        {
          lr_ = kv.second.item_double ();
        }
      else if (kv.first == "__step__")
        {
          step_count_ = static_cast<size_t> (kv.second.item_double ());
        }
      else
        {
          state_buffers_[kv.first] = kv.second.clone ().detach ();
        }
    }
}

SGD::SGD (std::vector<Tensor> params, double lr, double momentum,
          double dampening, double weight_decay, bool nesterov)
  : Optimizer (std::move (params), lr),
    momentum_ (momentum),
    dampening_ (dampening),
    weight_decay_ (weight_decay),
    nesterov_ (nesterov)
{
  BASALT_ASSERT_ARG (!nesterov || (momentum > 0.0 && dampening == 0.0),
                     "Nesterov SGD requires momentum > 0 and dampening == 0");
}

void
SGD::step (void)
{
  No_Grad_Guard no_grad;
  ++step_count_;

  for (size_t idx = 0; idx < params_.size (); ++idx)
    {
      Tensor &p = params_[idx];
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      size_t n = p.numel ();
      std::string key = "buf_" + std::to_string (idx);

      if (p.dtype () == Dtype::Float32 && g.dtype () == Dtype::Float32
          && p.is_contiguous () && g.is_contiguous () && n > 0)
        {
          bool has_mom = (momentum_ > 0.0);
          bool init_buf = false;
          float *buf_ptr = nullptr;
          if (has_mom)
            {
              auto it = state_buffers_.find (key);
              if (it == state_buffers_.end ())
                {
                  state_buffers_[key] = zeros_like (p);
                  init_buf = true;
                }
              buf_ptr = state_buffers_[key].data_ptr<float> ();
            }
          float *__restrict__ pp = p.data_ptr<float> ();
          const float *__restrict__ gp = g.data_ptr<float> ();
          float flr = static_cast<float> (lr_);
          float fwd = static_cast<float> (weight_decay_);
          float fmom = static_cast<float> (momentum_);
          float fdamp = static_cast<float> (dampening_);
          auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
          if (p.device ().is_cuda ()
              && cuda_drv.launch_sgd_step_f32 (
                   p.device ().index (), pp, gp, buf_ptr, flr, fwd, fmom,
                   fdamp, nesterov_, has_mom, init_buf, n))
            {
              p.bump_version ();
              continue;
            }
          if (p.device ().is_cuda ())
            {
              cuda_drv.sync_if_busy ();
            }
          for (size_t i = 0; i < n; ++i)
            {
              float pi = pp[i];
              float dp = gp[i] + fwd * pi;
              if (has_mom)
                {
                  float bi = init_buf
                             ? dp
                             : (fmom * buf_ptr[i] + (1.0f - fdamp) * dp);
                  buf_ptr[i] = bi;
                  dp = nesterov_ ? (dp + fmom * bi) : bi;
                }
              pp[i] = pi - flr * dp;
            }
          p.bump_version ();
          continue;
        }

      Tensor buf;
      if (momentum_ > 0.0)
        {
          auto it = state_buffers_.find (key);
          if (it == state_buffers_.end ())
            {
              buf = zeros_like (p);
              for (size_t i = 0; i < n; ++i)
                {
                  double d_p = g.read_scalar_at (i)
                               + weight_decay_ * p.read_scalar_at (i);
                  buf.write_scalar_at (i, d_p);
                }
              state_buffers_[key] = buf;
            }
          else
            {
              buf = it->second;
              for (size_t i = 0; i < n; ++i)
                {
                  double d_p = g.read_scalar_at (i)
                               + weight_decay_ * p.read_scalar_at (i);
                  double b_val = momentum_ * buf.read_scalar_at (i)
                                 + (1.0 - dampening_) * d_p;
                  buf.write_scalar_at (i, b_val);
                }
            }
        }

      for (size_t i = 0; i < n; ++i)
        {
          double d_p = g.read_scalar_at (i)
                       + weight_decay_ * p.read_scalar_at (i);
          if (momentum_ > 0.0)
            {
              if (nesterov_)
                {
                  d_p += momentum_ * buf.read_scalar_at (i);
                }
              else
                {
                  d_p = buf.read_scalar_at (i);
                }
            }
          p.write_scalar_at (i, p.read_scalar_at (i) - lr_ * d_p);
        }
      p.bump_version ();
    }
}

Adam::Adam (std::vector<Tensor> params, double lr, double beta1, double beta2,
            double eps, double weight_decay)
  : Optimizer (std::move (params), lr),
    beta1_ (beta1),
    beta2_ (beta2),
    eps_ (eps),
    weight_decay_ (weight_decay)
{
}

void
Adam::step (void)
{
  No_Grad_Guard no_grad;
  ++step_count_;
  double bc1 = 1.0 - std::pow (beta1_, static_cast<double> (step_count_));
  double bc2 = 1.0 - std::pow (beta2_, static_cast<double> (step_count_));

  for (size_t idx = 0; idx < params_.size (); ++idx)
    {
      Tensor &p = params_[idx];
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      std::string m_key = "m_" + std::to_string (idx);
      std::string v_key = "v_" + std::to_string (idx);
      if (state_buffers_.find (m_key) == state_buffers_.end ())
        {
          state_buffers_[m_key] = zeros_like (p);
          state_buffers_[v_key] = zeros_like (p);
        }
      Tensor &m = state_buffers_[m_key];
      Tensor &v = state_buffers_[v_key];
      size_t n = p.numel ();

      if (p.dtype () == Dtype::Float32 && g.dtype () == Dtype::Float32
          && p.is_contiguous () && g.is_contiguous () && n > 0)
        {
          float *__restrict__ pp = p.data_ptr<float> ();
          const float *__restrict__ gp = g.data_ptr<float> ();
          float *__restrict__ mp = m.data_ptr<float> ();
          float *__restrict__ vp = v.data_ptr<float> ();
          float flr = static_cast<float> (lr_);
          float fb1 = static_cast<float> (beta1_);
          float fb2 = static_cast<float> (beta2_);
          float feps = static_cast<float> (eps_);
          float fwd = static_cast<float> (weight_decay_);
          float fbc1 = static_cast<float> (bc1);
          float fbc2 = static_cast<float> (bc2);
          auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
          if (p.device ().is_cuda ()
              && cuda_drv.launch_adam_step_f32 (
                   p.device ().index (), pp, gp, mp, vp, flr, fb1, fb2,
                   feps, fwd, fbc1, fbc2, false, n))
            {
              p.bump_version ();
              continue;
            }
          if (p.device ().is_cuda ())
            {
              cuda_drv.sync_if_busy ();
            }
          for (size_t i = 0; i < n; ++i)
            {
              float pv = pp[i];
              float gv = gp[i] + fwd * pv;
              float mv = fb1 * mp[i] + (1.0f - fb1) * gv;
              float vv = fb2 * vp[i] + (1.0f - fb2) * gv * gv;
              mp[i] = mv;
              vp[i] = vv;
              float m_hat = mv / fbc1;
              float v_hat = vv / fbc2;
              pp[i] = pv - flr * m_hat / (std::sqrt (v_hat) + feps);
            }
          p.bump_version ();
          continue;
        }

      for (size_t i = 0; i < n; ++i)
        {
          double pv = p.read_scalar_at (i);
          double gv = g.read_scalar_at (i) + weight_decay_ * pv;
          double mv = beta1_ * m.read_scalar_at (i) + (1.0 - beta1_) * gv;
          double vv = beta2_ * v.read_scalar_at (i) + (1.0 - beta2_) * gv * gv;
          m.write_scalar_at (i, mv);
          v.write_scalar_at (i, vv);

          double m_hat = mv / bc1;
          double v_hat = vv / bc2;
          p.write_scalar_at (
            i, pv - lr_ * m_hat / (std::sqrt (v_hat) + eps_));
        }
      p.bump_version ();
    }
}

AdamW::AdamW (std::vector<Tensor> params, double lr, double beta1,
              double beta2, double eps, double weight_decay)
  : Optimizer (std::move (params), lr),
    beta1_ (beta1),
    beta2_ (beta2),
    eps_ (eps),
    weight_decay_ (weight_decay)
{
}

void
AdamW::step (void)
{
  No_Grad_Guard no_grad;
  ++step_count_;
  double bc1 = 1.0 - std::pow (beta1_, static_cast<double> (step_count_));
  double bc2 = 1.0 - std::pow (beta2_, static_cast<double> (step_count_));

  for (size_t idx = 0; idx < params_.size (); ++idx)
    {
      Tensor &p = params_[idx];
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      std::string m_key = "m_" + std::to_string (idx);
      std::string v_key = "v_" + std::to_string (idx);
      if (state_buffers_.find (m_key) == state_buffers_.end ())
        {
          state_buffers_[m_key] = zeros_like (p);
          state_buffers_[v_key] = zeros_like (p);
        }
      Tensor &m = state_buffers_[m_key];
      Tensor &v = state_buffers_[v_key];
      size_t n = p.numel ();

      if (p.dtype () == Dtype::Float32 && g.dtype () == Dtype::Float32
          && p.is_contiguous () && g.is_contiguous () && n > 0)
        {
          float *__restrict__ pp = p.data_ptr<float> ();
          const float *__restrict__ gp = g.data_ptr<float> ();
          float *__restrict__ mp = m.data_ptr<float> ();
          float *__restrict__ vp = v.data_ptr<float> ();
          float flr = static_cast<float> (lr_);
          float fb1 = static_cast<float> (beta1_);
          float fb2 = static_cast<float> (beta2_);
          float feps = static_cast<float> (eps_);
          float fwd = static_cast<float> (weight_decay_);
          float fbc1 = static_cast<float> (bc1);
          float fbc2 = static_cast<float> (bc2);
          auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
          if (p.device ().is_cuda ()
              && cuda_drv.launch_adam_step_f32 (
                   p.device ().index (), pp, gp, mp, vp, flr, fb1, fb2,
                   feps, fwd, fbc1, fbc2, true, n))
            {
              p.bump_version ();
              continue;
            }
          if (p.device ().is_cuda ())
            {
              cuda_drv.sync_if_busy ();
            }
          for (size_t i = 0; i < n; ++i)
            {
              float pv = pp[i] * (1.0f - flr * fwd);
              float gv = gp[i];
              float mv = fb1 * mp[i] + (1.0f - fb1) * gv;
              float vv = fb2 * vp[i] + (1.0f - fb2) * gv * gv;
              mp[i] = mv;
              vp[i] = vv;
              float m_hat = mv / fbc1;
              float v_hat = vv / fbc2;
              pp[i] = pv - flr * m_hat / (std::sqrt (v_hat) + feps);
            }
          p.bump_version ();
          continue;
        }

      for (size_t i = 0; i < n; ++i)
        {
          double pv = p.read_scalar_at (i) * (1.0 - lr_ * weight_decay_);
          double gv = g.read_scalar_at (i);
          double mv = beta1_ * m.read_scalar_at (i) + (1.0 - beta1_) * gv;
          double vv = beta2_ * v.read_scalar_at (i) + (1.0 - beta2_) * gv * gv;
          m.write_scalar_at (i, mv);
          v.write_scalar_at (i, vv);

          double m_hat = mv / bc1;
          double v_hat = vv / bc2;
          p.write_scalar_at (
            i, pv - lr_ * m_hat / (std::sqrt (v_hat) + eps_));
        }
      p.bump_version ();
    }
}

RMSprop::RMSprop (std::vector<Tensor> params, double lr, double alpha,
                  double eps, double weight_decay, double momentum,
                  bool centered)
  : Optimizer (std::move (params), lr),
    alpha_ (alpha),
    eps_ (eps),
    weight_decay_ (weight_decay),
    momentum_ (momentum),
    centered_ (centered)
{
}

void
RMSprop::step (void)
{
  No_Grad_Guard no_grad;
  ++step_count_;

  for (size_t idx = 0; idx < params_.size (); ++idx)
    {
      Tensor &p = params_[idx];
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      std::string sq_key = "sq_" + std::to_string (idx);
      std::string ga_key = "ga_" + std::to_string (idx);
      std::string mb_key = "mb_" + std::to_string (idx);
      if (state_buffers_.find (sq_key) == state_buffers_.end ())
        {
          state_buffers_[sq_key] = zeros_like (p);
          if (centered_)
            {
              state_buffers_[ga_key] = zeros_like (p);
            }
          if (momentum_ > 0.0)
            {
              state_buffers_[mb_key] = zeros_like (p);
            }
        }
      Tensor &sq = state_buffers_[sq_key];
      size_t n = p.numel ();

      for (size_t i = 0; i < n; ++i)
        {
          double pv = p.read_scalar_at (i);
          double gv = g.read_scalar_at (i) + weight_decay_ * pv;
          double sq_v = alpha_ * sq.read_scalar_at (i)
                        + (1.0 - alpha_) * gv * gv;
          sq.write_scalar_at (i, sq_v);

          double avg = sq_v;
          if (centered_)
            {
              Tensor &ga = state_buffers_[ga_key];
              double ga_v = alpha_ * ga.read_scalar_at (i)
                            + (1.0 - alpha_) * gv;
              ga.write_scalar_at (i, ga_v);
              avg = std::max (0.0, sq_v - ga_v * ga_v);
            }
          double denom = std::sqrt (avg) + eps_;
          if (momentum_ > 0.0)
            {
              Tensor &mb = state_buffers_[mb_key];
              double b_v = momentum_ * mb.read_scalar_at (i) + gv / denom;
              mb.write_scalar_at (i, b_v);
              p.write_scalar_at (i, pv - lr_ * b_v);
            }
          else
            {
              p.write_scalar_at (i, pv - lr_ * gv / denom);
            }
        }
      p.bump_version ();
    }
}

Adagrad::Adagrad (std::vector<Tensor> params, double lr, double lr_decay,
                  double weight_decay, double eps)
  : Optimizer (std::move (params), lr),
    lr_decay_ (lr_decay),
    weight_decay_ (weight_decay),
    eps_ (eps)
{
}

void
Adagrad::step (void)
{
  No_Grad_Guard no_grad;
  ++step_count_;
  double clr = lr_ / (1.0 + static_cast<double> (step_count_ - 1) * lr_decay_);

  for (size_t idx = 0; idx < params_.size (); ++idx)
    {
      Tensor &p = params_[idx];
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      std::string sum_key = "sum_" + std::to_string (idx);
      if (state_buffers_.find (sum_key) == state_buffers_.end ())
        {
          state_buffers_[sum_key] = zeros_like (p);
        }
      Tensor &acc = state_buffers_[sum_key];
      size_t n = p.numel ();

      for (size_t i = 0; i < n; ++i)
        {
          double pv = p.read_scalar_at (i);
          double gv = g.read_scalar_at (i) + weight_decay_ * pv;
          double sv = acc.read_scalar_at (i) + gv * gv;
          acc.write_scalar_at (i, sv);
          p.write_scalar_at (i, pv - clr * gv / (std::sqrt (sv) + eps_));
        }
      p.bump_version ();
    }
}

LAMB::LAMB (std::vector<Tensor> params, double lr, double beta1, double beta2,
            double eps, double weight_decay)
  : Optimizer (std::move (params), lr),
    beta1_ (beta1),
    beta2_ (beta2),
    eps_ (eps),
    weight_decay_ (weight_decay)
{
}

void
LAMB::step (void)
{
  No_Grad_Guard no_grad;
  ++step_count_;
  double bc1 = 1.0 - std::pow (beta1_, static_cast<double> (step_count_));
  double bc2 = 1.0 - std::pow (beta2_, static_cast<double> (step_count_));

  for (size_t idx = 0; idx < params_.size (); ++idx)
    {
      Tensor &p = params_[idx];
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      std::string m_key = "m_" + std::to_string (idx);
      std::string v_key = "v_" + std::to_string (idx);
      if (state_buffers_.find (m_key) == state_buffers_.end ())
        {
          state_buffers_[m_key] = zeros_like (p);
          state_buffers_[v_key] = zeros_like (p);
        }
      Tensor &m = state_buffers_[m_key];
      Tensor &v = state_buffers_[v_key];
      size_t n = p.numel ();

      std::vector<double> update_vec (n);
      double w_norm_sq = 0.0;
      double r_norm_sq = 0.0;

      for (size_t i = 0; i < n; ++i)
        {
          double pv = p.read_scalar_at (i);
          double gv = g.read_scalar_at (i);
          double mv = beta1_ * m.read_scalar_at (i) + (1.0 - beta1_) * gv;
          double vv = beta2_ * v.read_scalar_at (i) + (1.0 - beta2_) * gv * gv;
          m.write_scalar_at (i, mv);
          v.write_scalar_at (i, vv);

          double m_hat = mv / bc1;
          double v_hat = vv / bc2;
          double r_i = m_hat / (std::sqrt (v_hat) + eps_)
                       + weight_decay_ * pv;
          update_vec[i] = r_i;
          w_norm_sq += pv * pv;
          r_norm_sq += r_i * r_i;
        }

      double w_norm = std::sqrt (w_norm_sq);
      double r_norm = std::sqrt (r_norm_sq);
      double trust_ratio = (w_norm > 0.0 && r_norm > 0.0)
                           ? (w_norm / r_norm)
                           : 1.0;

      for (size_t i = 0; i < n; ++i)
        {
          p.write_scalar_at (
            i, p.read_scalar_at (i) - lr_ * trust_ratio * update_vec[i]);
        }
      p.bump_version ();
    }
}

double
clip_grad_norm_ (const std::vector<Tensor> &parameters,
                 double max_norm,
                 double norm_type)
{
  No_Grad_Guard no_grad;
  BASALT_ASSERT_ARG (max_norm > 0.0 && norm_type > 0.0,
                     "clip_grad_norm_() requires positive max_norm and p");
  double total_p = 0.0;
  for (const Tensor &p : parameters)
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
          const float *gp = g.data_ptr<float> ();
          if (norm_type == 2.0)
            {
              for (size_t i = 0; i < n; ++i)
                {
                  double v = static_cast<double> (gp[i]);
                  total_p += v * v;
                }
            }
          else
            {
              for (size_t i = 0; i < n; ++i)
                {
                  total_p += std::pow (
                    std::fabs (static_cast<double> (gp[i])), norm_type);
                }
            }
        }
      else
        {
          for (size_t i = 0; i < n; ++i)
            {
              total_p += std::pow (std::fabs (g.read_scalar_at (i)),
                                   norm_type);
            }
        }
    }
  double total_norm = std::pow (total_p, 1.0 / norm_type);
  double clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1.0)
    {
      for (const Tensor &p : parameters)
        {
          if (p.defined () && p.grad ().defined ())
            {
              p.grad ().mul_ (clip_coef);
            }
        }
    }
  return total_norm;
}

void
clip_grad_value_ (const std::vector<Tensor> &parameters,
                  double clip_value)
{
  No_Grad_Guard no_grad;
  BASALT_ASSERT_ARG (clip_value > 0.0,
                     "clip_grad_value_() requires clip_value > 0");
  for (const Tensor &p : parameters)
    {
      if (!p.defined () || !p.grad ().defined ())
        {
          continue;
        }
      Tensor g = p.grad ();
      size_t n = g.numel ();
      for (size_t i = 0; i < n; ++i)
        {
          double v = std::clamp (g.read_scalar_at (i),
                                 -clip_value, clip_value);
          g.write_scalar_at (i, v);
        }
    }
}

} /* namespace basalt::optim */
