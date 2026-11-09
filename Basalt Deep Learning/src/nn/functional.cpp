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
#include <basalt/functional.hpp>
#include <basalt/amp.hpp>
#include <basalt/autograd.hpp>
#include <basalt/random.hpp>
#include "tensor/tensor_internal.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace basalt::nn::functional
{

Tensor
linear (const Tensor &input, const Tensor &weight, const Tensor &bias)
{
  BASALT_ASSERT_SHAPE (input.ndim () >= 1 && weight.ndim () == 2,
                       "linear() expects weight of rank 2");
  Tensor x = amp::autocast_cast (input);
  Tensor w = amp::autocast_cast (weight);
  Tensor out = ops::matmul (x, w.transpose (0, 1));
  if (bias.defined ())
    {
      out = ops::add (out, amp::autocast_cast (bias));
    }
  return out;
}

namespace
{

class Conv3d_Backward : public Autograd_Node
{
public:
  Conv3d_Backward (const Tensor &input, const Tensor &weight,
                   bool has_bias,
                   size_t sd, size_t sh, size_t sw,
                   size_t pd, size_t ph, size_t pw,
                   size_t dd, size_t dh, size_t dw,
                   size_t groups)
    : input_ (input),
      weight_ (weight),
      has_bias_ (has_bias),
      sd_ (sd), sh_ (sh), sw_ (sw),
      pd_ (pd), ph_ (ph), pw_ (pw),
      dd_ (dd), dh_ (dh), dw_ (dw),
      groups_ (groups)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Conv3d_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    Tensor x = input_.unpack ();
    Tensor w = weight_.unpack ();
    const Tensor &gy = grads[0];

    size_t batch = x.size (0);
    size_t c_in = x.size (1);
    size_t in_d = x.size (2);
    size_t in_h = x.size (3);
    size_t in_w = x.size (4);

    size_t c_out = w.size (0);
    size_t c_per_grp = w.size (1);
    size_t kd = w.size (2);
    size_t kh = w.size (3);
    size_t kw = w.size (4);

    size_t out_d = gy.size (2);
    size_t out_h = gy.size (3);
    size_t out_w = gy.size (4);
    size_t cout_per_grp = c_out / groups_;

    Tensor gx = zeros (x.shape (), x.options ().with_requires_grad (false));
    Tensor gw = zeros (w.shape (), w.options ().with_requires_grad (false));
    Tensor gb;
    if (has_bias_)
      {
        gb = zeros ({c_out}, w.options ().with_requires_grad (false));
      }

    if (x.dtype () == Dtype::Float32 && w.dtype () == Dtype::Float32
        && gy.dtype () == Dtype::Float32)
      {
        Tensor x_c = x.is_contiguous () ? x : x.contiguous ();
        Tensor w_c = w.is_contiguous () ? w : w.contiguous ();
        Tensor gy_c = gy.is_contiguous () ? gy : gy.contiguous ();
        if (x.device ().is_cuda ())
          {
            bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
          }
        const float *xp = x_c.data_ptr<float> ();
        const float *wp = w_c.data_ptr<float> ();
        const float *gyp = gy_c.data_ptr<float> ();
        float *gxp = gx.data_ptr<float> ();
        float *gwp = gw.data_ptr<float> ();
        float *gbp = has_bias_ ? gb.data_ptr<float> () : nullptr;

        size_t k_col = c_per_grp * kd * kh * kw;
        size_t spatial_out = out_d * out_h * out_w;
        if (groups_ == 1 && spatial_out >= 16 && k_col >= 8)
          {
            Tensor w_2d = w_c.view ({c_out, k_col});
            Tensor w_2d_t = w_2d.transpose (0, 1);
            Tensor col ({k_col, spatial_out},
                        Tensor_Options ()
                          .with_dtype (Dtype::Float32)
                          .with_device (x.device ()));
            Tensor gw_2d = gw.view ({c_out, k_col});
            for (size_t n = 0; n < batch; ++n)
              {
                if (x.device ().is_cuda ())
                  {
                    bellows_compat::Cuda_Runtime_Driver::instance ()
                      .sync_if_busy ();
                  }
                float *col_ptr = col.data_ptr<float> ();
                const float *in_n = xp + n * c_in * in_d * in_h * in_w;
                for (size_t ci = 0; ci < c_per_grp; ++ci)
                  {
                    for (size_t fd = 0; fd < kd; ++fd)
                      {
                        for (size_t fh = 0; fh < kh; ++fh)
                          {
                            for (size_t fw = 0; fw < kw; ++fw)
                              {
                                size_t col_r
                                  = ((ci * kd + fd) * kh + fh) * kw + fw;
                                float *col_row = col_ptr + col_r * spatial_out;
                                for (size_t od = 0; od < out_d; ++od)
                                  {
                                    ptrdiff_t id
                                      = static_cast<ptrdiff_t> (
                                          od * sd_ + fd * dd_)
                                        - static_cast<ptrdiff_t> (pd_);
                                    for (size_t oh = 0; oh < out_h; ++oh)
                                      {
                                        ptrdiff_t ih
                                          = static_cast<ptrdiff_t> (
                                              oh * sh_ + fh * dh_)
                                            - static_cast<ptrdiff_t> (ph_);
                                        float *dst_w
                                          = col_row
                                            + (od * out_h + oh) * out_w;
                                        if (id < 0
                                            || id >= static_cast<ptrdiff_t> (
                                                 in_d)
                                            || ih < 0
                                            || ih >= static_cast<ptrdiff_t> (
                                                 in_h))
                                          {
                                            std::fill_n (dst_w, out_w, 0.0f);
                                            continue;
                                          }
                                        const float *src_w
                                          = in_n
                                            + ((ci * in_d
                                                + static_cast<size_t> (id))
                                               * in_h
                                               + static_cast<size_t> (ih))
                                              * in_w;
                                        for (size_t ow = 0; ow < out_w; ++ow)
                                          {
                                            ptrdiff_t iw
                                              = static_cast<ptrdiff_t> (
                                                  ow * sw_ + fw * dw_)
                                                - static_cast<ptrdiff_t> (pw_);
                                            dst_w[ow]
                                              = (iw >= 0
                                                 && iw < static_cast<
                                                      ptrdiff_t> (in_w))
                                                ? src_w[iw]
                                                : 0.0f;
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
                Tensor gy_n = gy_c.select (0, static_cast<ptrdiff_t> (n))
                                .view ({c_out, spatial_out});
                if (has_bias_)
                  {
                    if (x.device ().is_cuda ())
                      {
                        bellows_compat::Cuda_Runtime_Driver::instance ()
                          .sync_if_busy ();
                      }
                    const float *gy_np = gy_n.data_ptr<float> ();
                    for (size_t oc = 0; oc < c_out; ++oc)
                      {
                        const float *r_ptr = gy_np + oc * spatial_out;
                        float acc = 0.0f;
                        for (size_t s = 0; s < spatial_out; ++s)
                          {
                            acc += r_ptr[s];
                          }
                        gbp[oc] += acc;
                      }
                  }
                Tensor dw_n = ops::matmul (gy_n, col.transpose (0, 1));
                gw_2d.add_ (dw_n);

                Tensor dcol_n = ops::matmul (w_2d_t, gy_n);
                if (x.device ().is_cuda ())
                  {
                    bellows_compat::Cuda_Runtime_Driver::instance ()
                      .sync_if_busy ();
                  }
                const float *dcol_ptr = dcol_n.data_ptr<float> ();
                float *gx_n = gxp + n * c_in * in_d * in_h * in_w;
                for (size_t ci = 0; ci < c_per_grp; ++ci)
                  {
                    for (size_t fd = 0; fd < kd; ++fd)
                      {
                        for (size_t fh = 0; fh < kh; ++fh)
                          {
                            for (size_t fw = 0; fw < kw; ++fw)
                              {
                                size_t col_r
                                  = ((ci * kd + fd) * kh + fh) * kw + fw;
                                const float *col_row
                                  = dcol_ptr + col_r * spatial_out;
                                for (size_t od = 0; od < out_d; ++od)
                                  {
                                    ptrdiff_t id
                                      = static_cast<ptrdiff_t> (
                                          od * sd_ + fd * dd_)
                                        - static_cast<ptrdiff_t> (pd_);
                                    if (id < 0
                                        || id >= static_cast<ptrdiff_t> (in_d))
                                      {
                                        continue;
                                      }
                                    for (size_t oh = 0; oh < out_h; ++oh)
                                      {
                                        ptrdiff_t ih
                                          = static_cast<ptrdiff_t> (
                                              oh * sh_ + fh * dh_)
                                            - static_cast<ptrdiff_t> (ph_);
                                        if (ih < 0
                                            || ih >= static_cast<ptrdiff_t> (
                                                 in_h))
                                          {
                                            continue;
                                          }
                                        const float *src_w
                                          = col_row
                                            + (od * out_h + oh) * out_w;
                                        float *dst_w
                                          = gx_n
                                            + ((ci * in_d
                                                + static_cast<size_t> (id))
                                               * in_h
                                               + static_cast<size_t> (ih))
                                              * in_w;
                                        for (size_t ow = 0; ow < out_w; ++ow)
                                          {
                                            ptrdiff_t iw
                                              = static_cast<ptrdiff_t> (
                                                  ow * sw_ + fw * dw_)
                                                - static_cast<ptrdiff_t> (pw_);
                                            if (iw >= 0
                                                && iw < static_cast<
                                                     ptrdiff_t> (in_w))
                                              {
                                                dst_w[iw] += src_w[ow];
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
            if (has_bias_)
              {
                return {gx, gw, gb};
              }
            return {gx, gw};
          }

        /* 1. Compute gw and gb in parallel across output channels oc.  */
        hyper_math::parallel_for (
          size_t (0), c_out, size_t (1), [&] (size_t oc)
          {
            size_t g = oc / cout_per_grp;
            float b_acc = 0.0f;
            for (size_t n = 0; n < batch; ++n)
              {
                for (size_t od = 0; od < out_d; ++od)
                  {
                    for (size_t oh = 0; oh < out_h; ++oh)
                      {
                        const float *gy_row
                          = gyp + (((n * c_out + oc) * out_d + od) * out_h
                                   + oh) * out_w;
                        if (has_bias_)
                          {
                            for (size_t ow = 0; ow < out_w; ++ow)
                              {
                                b_acc += gy_row[ow];
                              }
                          }
                        for (size_t ci = 0; ci < c_per_grp; ++ci)
                          {
                            size_t ic = g * c_per_grp + ci;
                            for (size_t fd = 0; fd < kd; ++fd)
                              {
                                ptrdiff_t id
                                  = static_cast<ptrdiff_t> (
                                      od * sd_ + fd * dd_)
                                    - static_cast<ptrdiff_t> (pd_);
                                if (id < 0
                                    || id >= static_cast<ptrdiff_t> (in_d))
                                  {
                                    continue;
                                  }
                                for (size_t fh = 0; fh < kh; ++fh)
                                  {
                                    ptrdiff_t ih
                                      = static_cast<ptrdiff_t> (
                                          oh * sh_ + fh * dh_)
                                        - static_cast<ptrdiff_t> (ph_);
                                    if (ih < 0
                                        || ih >= static_cast<ptrdiff_t> (in_h))
                                      {
                                        continue;
                                      }
                                    const float *x_row
                                      = xp + (((n * c_in + ic) * in_d
                                               + static_cast<size_t> (id))
                                              * in_h
                                              + static_cast<size_t> (ih))
                                             * in_w;
                                    float *gw_row
                                      = gwp + (((oc * c_per_grp + ci) * kd
                                                + fd) * kh + fh) * kw;
                                    for (size_t fw = 0; fw < kw; ++fw)
                                      {
                                        float acc = 0.0f;
                                        for (size_t ow = 0; ow < out_w; ++ow)
                                          {
                                            ptrdiff_t iw
                                              = static_cast<ptrdiff_t> (
                                                  ow * sw_ + fw * dw_)
                                                - static_cast<ptrdiff_t> (pw_);
                                            if (iw >= 0
                                                && iw < static_cast<
                                                     ptrdiff_t> (in_w))
                                              {
                                                acc += gy_row[ow]
                                                       * x_row[iw];
                                              }
                                          }
                                        gw_row[fw] += acc;
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
            if (has_bias_)
              {
                gbp[oc] = b_acc;
              }
          });

        /* 2. Compute gx in parallel across batch items n.  */
        hyper_math::parallel_for (
          size_t (0), batch, size_t (1), [&] (size_t n)
          {
            for (size_t g = 0; g < groups_; ++g)
              {
                for (size_t co = 0; co < cout_per_grp; ++co)
                  {
                    size_t oc = g * cout_per_grp + co;
                    for (size_t od = 0; od < out_d; ++od)
                      {
                        for (size_t oh = 0; oh < out_h; ++oh)
                          {
                            const float *gy_row
                              = gyp + (((n * c_out + oc) * out_d + od) * out_h
                                       + oh) * out_w;
                            for (size_t ci = 0; ci < c_per_grp; ++ci)
                              {
                                size_t ic = g * c_per_grp + ci;
                                for (size_t fd = 0; fd < kd; ++fd)
                                  {
                                    ptrdiff_t id
                                      = static_cast<ptrdiff_t> (
                                          od * sd_ + fd * dd_)
                                        - static_cast<ptrdiff_t> (pd_);
                                    if (id < 0
                                        || id >= static_cast<ptrdiff_t> (in_d))
                                      {
                                        continue;
                                      }
                                    for (size_t fh = 0; fh < kh; ++fh)
                                      {
                                        ptrdiff_t ih
                                          = static_cast<ptrdiff_t> (
                                              oh * sh_ + fh * dh_)
                                            - static_cast<ptrdiff_t> (ph_);
                                        if (ih < 0
                                            || ih >= static_cast<ptrdiff_t> (
                                                 in_h))
                                          {
                                            continue;
                                          }
                                        float *gx_row
                                          = gxp + (((n * c_in + ic) * in_d
                                                    + static_cast<size_t> (id))
                                                   * in_h
                                                   + static_cast<size_t> (ih))
                                                  * in_w;
                                        const float *w_row
                                          = wp + (((oc * c_per_grp + ci) * kd
                                                   + fd) * kh + fh) * kw;
                                        for (size_t fw = 0; fw < kw; ++fw)
                                          {
                                            float wv = w_row[fw];
                                            for (size_t ow = 0; ow < out_w;
                                                 ++ow)
                                              {
                                                ptrdiff_t iw
                                                  = static_cast<ptrdiff_t> (
                                                      ow * sw_ + fw * dw_)
                                                    - static_cast<ptrdiff_t> (
                                                        pw_);
                                                if (iw >= 0
                                                    && iw < static_cast<
                                                         ptrdiff_t> (in_w))
                                                  {
                                                    gx_row[iw]
                                                      += gy_row[ow] * wv;
                                                  }
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          });

        if (has_bias_)
          {
            return {gx, gw, gb};
          }
        return {gx, gw};
      }

    for (size_t n = 0; n < batch; ++n)
      {
        for (size_t g = 0; g < groups_; ++g)
          {
            for (size_t co = 0; co < cout_per_grp; ++co)
              {
                size_t oc = g * cout_per_grp + co;
                for (size_t od = 0; od < out_d; ++od)
                  {
                    for (size_t oh = 0; oh < out_h; ++oh)
                      {
                        for (size_t ow = 0; ow < out_w; ++ow)
                          {
                            size_t gy_idx
                              = (((n * c_out + oc) * out_d + od) * out_h + oh)
                                * out_w + ow;
                            double grad_val = gy.read_scalar_at (gy_idx);
                            if (has_bias_)
                              {
                                gb.write_scalar_at (
                                  oc, gb.read_scalar_at (oc) + grad_val);
                              }
                            for (size_t ci = 0; ci < c_per_grp; ++ci)
                              {
                                size_t ic = g * c_per_grp + ci;
                                for (size_t fd = 0; fd < kd; ++fd)
                                  {
                                    ptrdiff_t id
                                      = static_cast<ptrdiff_t> (
                                          od * sd_ + fd * dd_)
                                        - static_cast<ptrdiff_t> (pd_);
                                    if (id < 0
                                        || id >= static_cast<ptrdiff_t> (in_d))
                                      {
                                        continue;
                                      }
                                    for (size_t fh = 0; fh < kh; ++fh)
                                      {
                                        ptrdiff_t ih
                                          = static_cast<ptrdiff_t> (
                                              oh * sh_ + fh * dh_)
                                            - static_cast<ptrdiff_t> (ph_);
                                        if (ih < 0
                                            || ih >= static_cast<ptrdiff_t> (
                                                 in_h))
                                          {
                                            continue;
                                          }
                                        for (size_t fw = 0; fw < kw; ++fw)
                                          {
                                            ptrdiff_t iw
                                              = static_cast<ptrdiff_t> (
                                                  ow * sw_ + fw * dw_)
                                                - static_cast<ptrdiff_t> (pw_);
                                            if (iw < 0
                                                || iw >= static_cast<ptrdiff_t>
                                                     (in_w))
                                              {
                                                continue;
                                              }
                                            size_t x_idx
                                              = (((n * c_in + ic) * in_d
                                                  + static_cast<size_t> (id))
                                                 * in_h
                                                 + static_cast<size_t> (ih))
                                                * in_w
                                                + static_cast<size_t> (iw);
                                            size_t w_idx
                                              = (((oc * c_per_grp + ci) * kd
                                                  + fd) * kh + fh) * kw + fw;

                                            double xv
                                              = x.read_scalar_at (x_idx);
                                            double wv
                                              = w.read_scalar_at (w_idx);
                                            gx.write_scalar_at (
                                              x_idx,
                                              gx.read_scalar_at (x_idx)
                                              + grad_val * wv);
                                            gw.write_scalar_at (
                                              w_idx,
                                              gw.read_scalar_at (w_idx)
                                              + grad_val * xv);
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }
      }

    if (has_bias_)
      {
        return {gx, gw, gb};
      }
    return {gx, gw};
  }

  void
  release_saved_variables (void) noexcept override
  {
    input_ = Saved_Variable ();
    weight_ = Saved_Variable ();
  }

private:
  Saved_Variable input_;
  Saved_Variable weight_;
  bool has_bias_;
  size_t sd_, sh_, sw_;
  size_t pd_, ph_, pw_;
  size_t dd_, dh_, dw_;
  size_t groups_;
};

class Max_Pool3d_Backward : public Autograd_Node
{
public:
  Max_Pool3d_Backward (Shape_Vector in_shape, Tensor_Options opts,
                       std::vector<size_t> max_indices)
    : in_shape_ (std::move (in_shape)),
      opts_ (opts.with_requires_grad (false)),
      max_indices_ (std::move (max_indices))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Max_Pool3d_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    const Tensor &gy = grads[0];
    Tensor gx = zeros (in_shape_, opts_);
    size_t n = gy.numel ();
    for (size_t i = 0; i < n; ++i)
      {
        size_t dst_idx = max_indices_[i];
        gx.write_scalar_at (
          dst_idx, gx.read_scalar_at (dst_idx) + gy.read_scalar_at (i));
      }
    return {gx};
  }

private:
  Shape_Vector in_shape_;
  Tensor_Options opts_;
  std::vector<size_t> max_indices_;
};

class Avg_Pool3d_Backward : public Autograd_Node
{
public:
  Avg_Pool3d_Backward (Shape_Vector in_shape, Tensor_Options opts,
                       size_t kd, size_t kh, size_t kw,
                       size_t sd, size_t sh, size_t sw,
                       size_t pd, size_t ph, size_t pw)
    : in_shape_ (std::move (in_shape)),
      opts_ (opts.with_requires_grad (false)),
      kd_ (kd), kh_ (kh), kw_ (kw),
      sd_ (sd), sh_ (sh), sw_ (sw),
      pd_ (pd), ph_ (ph), pw_ (pw)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Avg_Pool3d_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    const Tensor &gy = grads[0];
    Tensor gx = zeros (in_shape_, opts_);

    size_t batch = in_shape_[0];
    size_t channels = in_shape_[1];
    size_t in_d = in_shape_[2];
    size_t in_h = in_shape_[3];
    size_t in_w = in_shape_[4];

    size_t out_d = gy.size (2);
    size_t out_h = gy.size (3);
    size_t out_w = gy.size (4);
    double inv_vol = 1.0 / static_cast<double> (kd_ * kh_ * kw_);

    for (size_t n = 0; n < batch; ++n)
      {
        for (size_t c = 0; c < channels; ++c)
          {
            for (size_t od = 0; od < out_d; ++od)
              {
                for (size_t oh = 0; oh < out_h; ++oh)
                  {
                    for (size_t ow = 0; ow < out_w; ++ow)
                      {
                        size_t gy_idx
                          = (((n * channels + c) * out_d + od) * out_h + oh)
                            * out_w + ow;
                        double share = gy.read_scalar_at (gy_idx) * inv_vol;
                        for (size_t fd = 0; fd < kd_; ++fd)
                          {
                            ptrdiff_t id
                              = static_cast<ptrdiff_t> (od * sd_ + fd)
                                - static_cast<ptrdiff_t> (pd_);
                            if (id < 0
                                || id >= static_cast<ptrdiff_t> (in_d))
                              {
                                continue;
                              }
                            for (size_t fh = 0; fh < kh_; ++fh)
                              {
                                ptrdiff_t ih
                                  = static_cast<ptrdiff_t> (oh * sh_ + fh)
                                    - static_cast<ptrdiff_t> (ph_);
                                if (ih < 0
                                    || ih >= static_cast<ptrdiff_t> (in_h))
                                  {
                                    continue;
                                  }
                                for (size_t fw = 0; fw < kw_; ++fw)
                                  {
                                    ptrdiff_t iw
                                      = static_cast<ptrdiff_t> (ow * sw_ + fw)
                                        - static_cast<ptrdiff_t> (pw_);
                                    if (iw < 0
                                        || iw >= static_cast<ptrdiff_t> (in_w))
                                      {
                                        continue;
                                      }
                                    size_t x_idx
                                      = (((n * channels + c) * in_d
                                          + static_cast<size_t> (id)) * in_h
                                         + static_cast<size_t> (ih)) * in_w
                                        + static_cast<size_t> (iw);
                                    gx.write_scalar_at (
                                      x_idx,
                                      gx.read_scalar_at (x_idx) + share);
                                  }
                              }
                          }
                      }
                  }
              }
          }
      }
    return {gx};
  }

private:
  Shape_Vector in_shape_;
  Tensor_Options opts_;
  size_t kd_, kh_, kw_;
  size_t sd_, sh_, sw_;
  size_t pd_, ph_, pw_;
};

} /* namespace */

Tensor
conv1d (const Tensor &input, const Tensor &weight, const Tensor &bias,
        size_t stride, size_t padding, size_t dilation, size_t groups)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 3 && weight.ndim () == 3,
                       "conv1d() expects 3-D input and weight");
  Tensor x5 = input.unsqueeze (2).unsqueeze (3);
  Tensor w5 = weight.unsqueeze (2).unsqueeze (3);
  Tensor y5 = conv3d (x5, w5, bias,
                      1, 1, stride,
                      0, 0, padding,
                      1, 1, dilation,
                      groups);
  return y5.squeeze (2).squeeze (2);
}

Tensor
conv2d (const Tensor &input, const Tensor &weight, const Tensor &bias,
        size_t stride_h, size_t stride_w,
        size_t pad_h, size_t pad_w,
        size_t dilation_h, size_t dilation_w,
        size_t groups)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 4 && weight.ndim () == 4,
                       "conv2d() expects 4-D input and weight");
  Tensor x5 = input.unsqueeze (2);
  Tensor w5 = weight.unsqueeze (2);
  Tensor y5 = conv3d (x5, w5, bias,
                      1, stride_h, stride_w,
                      0, pad_h, pad_w,
                      1, dilation_h, dilation_w,
                      groups);
  return y5.squeeze (2);
}

Tensor
conv3d (const Tensor &input, const Tensor &weight, const Tensor &bias,
        size_t stride_d, size_t stride_h, size_t stride_w,
        size_t pad_d, size_t pad_h, size_t pad_w,
        size_t dilation_d, size_t dilation_h, size_t dilation_w,
        size_t groups)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 5 && weight.ndim () == 5,
                       "conv3d() expects 5-D input and weight");
  BASALT_ASSERT_ARG (groups >= 1 && stride_d >= 1 && stride_h >= 1
                     && stride_w >= 1,
                     "conv3d() invalid stride or groups");

  size_t batch = input.size (0);
  size_t c_in = input.size (1);
  size_t in_d = input.size (2);
  size_t in_h = input.size (3);
  size_t in_w = input.size (4);

  size_t c_out = weight.size (0);
  size_t c_per_grp = weight.size (1);
  size_t kd = weight.size (2);
  size_t kh = weight.size (3);
  size_t kw = weight.size (4);

  BASALT_ASSERT_SHAPE (c_in == c_per_grp * groups && c_out % groups == 0,
                       "conv3d() channel/group mismatch");

  size_t eff_kd = dilation_d * (kd - 1) + 1;
  size_t eff_kh = dilation_h * (kh - 1) + 1;
  size_t eff_kw = dilation_w * (kw - 1) + 1;
  BASALT_ASSERT_SHAPE (in_d + 2 * pad_d >= eff_kd
                       && in_h + 2 * pad_h >= eff_kh
                       && in_w + 2 * pad_w >= eff_kw,
                       "conv3d() kernel exceeds padded spatial size");

  size_t out_d = (in_d + 2 * pad_d - eff_kd) / stride_d + 1;
  size_t out_h = (in_h + 2 * pad_h - eff_kh) / stride_h + 1;
  size_t out_w = (in_w + 2 * pad_w - eff_kw) / stride_w + 1;
  size_t cout_per_grp = c_out / groups;

  Dtype dt = promote_dtypes (input.dtype (), weight.dtype ());
  Tensor out ({batch, c_out, out_d, out_h, out_w},
              Tensor_Options ().with_dtype (dt).with_device (input.device ()));

  if (dt == Dtype::Float32 && input.dtype () == Dtype::Float32
      && weight.dtype () == Dtype::Float32)
    {
      No_Grad_Guard ng;
      Tensor in_c = input.is_contiguous () ? input : input.contiguous ();
      Tensor w_c = weight.is_contiguous () ? weight : weight.contiguous ();
      if (input.device ().is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      const float *in_ptr = in_c.data_ptr<float> ();
      const float *w_ptr = w_c.data_ptr<float> ();
      float *out_ptr = out.data_ptr<float> ();

      /* Use im2col + GEMM when groups == 1 and spatial output is large enough
         to amortize patch extraction, routing through AVX2/FMA or cuBLAS.  */
      size_t k_col = c_per_grp * kd * kh * kw;
      size_t spatial_out = out_d * out_h * out_w;
      if (groups == 1 && spatial_out >= 16 && k_col >= 8)
        {
          Tensor w_2d = w_c.view ({c_out, k_col});
          Tensor col ({k_col, spatial_out},
                      Tensor_Options ()
                        .with_dtype (Dtype::Float32)
                        .with_device (input.device ()));
          if (input.device ().is_cuda ())
            {
              bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
            }
          for (size_t n = 0; n < batch; ++n)
            {
              float *col_ptr = col.data_ptr<float> ();
              const float *in_n = in_ptr + n * c_in * in_d * in_h * in_w;
              hyper_math::parallel_for (
                size_t (0), c_per_grp, size_t (1), [&] (size_t ci)
                {
                  for (size_t fd = 0; fd < kd; ++fd)
                    {
                      for (size_t fh = 0; fh < kh; ++fh)
                        {
                          for (size_t fw = 0; fw < kw; ++fw)
                            {
                              size_t col_r
                                = ((ci * kd + fd) * kh + fh) * kw + fw;
                              float *col_row = col_ptr + col_r * spatial_out;
                              for (size_t od = 0; od < out_d; ++od)
                                {
                                  ptrdiff_t id
                                    = static_cast<ptrdiff_t> (
                                        od * stride_d + fd * dilation_d)
                                      - static_cast<ptrdiff_t> (pad_d);
                                  for (size_t oh = 0; oh < out_h; ++oh)
                                    {
                                      ptrdiff_t ih
                                        = static_cast<ptrdiff_t> (
                                            oh * stride_h + fh * dilation_h)
                                          - static_cast<ptrdiff_t> (pad_h);
                                      float *dst_w
                                        = col_row + (od * out_h + oh) * out_w;
                                      if (id < 0
                                          || id >= static_cast<
                                               ptrdiff_t> (in_d)
                                          || ih < 0
                                          || ih >= static_cast<ptrdiff_t> (
                                               in_h))
                                        {
                                          std::fill_n (dst_w, out_w, 0.0f);
                                          continue;
                                        }
                                      const float *src_w
                                        = in_n
                                          + ((ci * in_d
                                              + static_cast<size_t> (id))
                                             * in_h
                                             + static_cast<size_t> (ih))
                                            * in_w;
                                      for (size_t ow = 0; ow < out_w; ++ow)
                                        {
                                          ptrdiff_t iw
                                            = static_cast<ptrdiff_t> (
                                                ow * stride_w
                                                + fw * dilation_w)
                                              - static_cast<ptrdiff_t> (pad_w);
                                          dst_w[ow]
                                            = (iw >= 0
                                               && iw < static_cast<ptrdiff_t> (
                                                    in_w))
                                              ? src_w[iw]
                                              : 0.0f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
              Tensor gemm_n = ops::matmul (w_2d, col);
              if (input.device ().is_cuda ())
                {
                  bellows_compat::Cuda_Runtime_Driver::instance ()
                    .sync_if_busy ();
                }
              const float *gp = gemm_n.data_ptr<float> ();
              float *out_n = out_ptr + n * c_out * spatial_out;
              if (bias.defined ())
                {
                  for (size_t oc = 0; oc < c_out; ++oc)
                    {
                      float bv = static_cast<float> (bias.read_scalar_at (oc));
                      const float *gr = gp + oc * spatial_out;
                      float *orow = out_n + oc * spatial_out;
                      for (size_t s = 0; s < spatial_out; ++s)
                        {
                          orow[s] = gr[s] + bv;
                        }
                    }
                }
              else
                {
                  std::memcpy (out_n, gp,
                               c_out * spatial_out * sizeof (float));
                }
            }
        }
      else
        {
          size_t total_slices = batch * c_out;
          hyper_math::parallel_for (
            size_t (0), total_slices, size_t (1), [&] (size_t slice_idx)
            {
              size_t n = slice_idx / c_out;
              size_t oc = slice_idx % c_out;
              size_t g = oc / cout_per_grp;
              float b_val = bias.defined ()
                            ? static_cast<float> (bias.read_scalar_at (oc))
                            : 0.0f;
              for (size_t od = 0; od < out_d; ++od)
                {
                  for (size_t oh = 0; oh < out_h; ++oh)
                    {
                      for (size_t ow = 0; ow < out_w; ++ow)
                        {
                          float acc = b_val;
                          for (size_t ci = 0; ci < c_per_grp; ++ci)
                            {
                              size_t ic = g * c_per_grp + ci;
                              for (size_t fd = 0; fd < kd; ++fd)
                                {
                                  ptrdiff_t id
                                    = static_cast<ptrdiff_t> (
                                        od * stride_d + fd * dilation_d)
                                      - static_cast<ptrdiff_t> (pad_d);
                                  if (id < 0
                                      || id >= static_cast<ptrdiff_t> (in_d))
                                    continue;
                                  for (size_t fh = 0; fh < kh; ++fh)
                                    {
                                      ptrdiff_t ih
                                        = static_cast<ptrdiff_t> (
                                            oh * stride_h + fh * dilation_h)
                                          - static_cast<ptrdiff_t> (pad_h);
                                      if (ih < 0
                                          || ih >= static_cast<ptrdiff_t> (
                                               in_h))
                                        continue;
                                      size_t x_row
                                        = (((n * c_in + ic) * in_d
                                            + static_cast<size_t> (id)) * in_h
                                           + static_cast<size_t> (ih)) * in_w;
                                      size_t w_row
                                        = (((oc * c_per_grp + ci) * kd + fd)
                                           * kh + fh) * kw;
                                      for (size_t fw = 0; fw < kw; ++fw)
                                        {
                                          ptrdiff_t iw
                                            = static_cast<ptrdiff_t> (
                                                ow * stride_w
                                                + fw * dilation_w)
                                              - static_cast<ptrdiff_t> (
                                                  pad_w);
                                          if (iw >= 0
                                              && iw < static_cast<ptrdiff_t> (
                                                   in_w))
                                            {
                                              acc
                                                += in_ptr[x_row
                                                          + static_cast<size_t>
                                                              (iw)]
                                                   * w_ptr[w_row + fw];
                                            }
                                        }
                                    }
                                }
                            }
                          size_t out_idx
                            = (((n * c_out + oc) * out_d + od) * out_h + oh)
                              * out_w + ow;
                          out_ptr[out_idx] = acc;
                        }
                    }
                }
            });
        }
    }
  else
    {
      for (size_t n = 0; n < batch; ++n)
        {
          for (size_t g = 0; g < groups; ++g)
            {
              for (size_t co = 0; co < cout_per_grp; ++co)
                {
                  size_t oc = g * cout_per_grp + co;
                  double b_val =
                    bias.defined () ? bias.read_scalar_at (oc) : 0.0;
                  for (size_t od = 0; od < out_d; ++od)
                    {
                      for (size_t oh = 0; oh < out_h; ++oh)
                        {
                          for (size_t ow = 0; ow < out_w; ++ow)
                            {
                              double acc = b_val;
                              for (size_t ci = 0; ci < c_per_grp; ++ci)
                                {
                                  size_t ic = g * c_per_grp + ci;
                                  for (size_t fd = 0; fd < kd; ++fd)
                                    {
                                      ptrdiff_t id =
                                        static_cast<ptrdiff_t> (
                                          od * stride_d + fd * dilation_d)
                                        - static_cast<ptrdiff_t> (pad_d);
                                      if (id < 0
                                          || id >= static_cast<ptrdiff_t> (
                                               in_d))
                                        continue;
                                      for (size_t fh = 0; fh < kh; ++fh)
                                        {
                                          ptrdiff_t ih =
                                            static_cast<ptrdiff_t> (
                                              oh * stride_h + fh * dilation_h)
                                            - static_cast<ptrdiff_t> (pad_h);
                                          if (ih < 0
                                              || ih >= static_cast<ptrdiff_t> (
                                                   in_h))
                                            continue;
                                          for (size_t fw = 0; fw < kw; ++fw)
                                            {
                                              ptrdiff_t iw =
                                                static_cast<ptrdiff_t> (
                                                  ow * stride_w
                                                  + fw * dilation_w)
                                                - static_cast<ptrdiff_t> (
                                                    pad_w);
                                              if (iw < 0
                                                  || iw >= static_cast<
                                                       ptrdiff_t> (in_w))
                                                continue;
                                              size_t x_idx =
                                                (((n * c_in + ic) * in_d
                                                  + static_cast<size_t> (id))
                                                 * in_h
                                                 + static_cast<size_t> (ih))
                                                * in_w
                                                + static_cast<size_t> (iw);
                                              size_t w_idx =
                                                (((oc * c_per_grp + ci) * kd
                                                  + fd) * kh + fh) * kw + fw;
                                              acc +=
                                                input.read_scalar_at (x_idx)
                                                * weight.read_scalar_at (
                                                    w_idx);
                                            }
                                        }
                                    }
                                }
                              size_t out_idx =
                                (((n * c_out + oc) * out_d + od) * out_h + oh)
                                * out_w + ow;
                              out.write_scalar_at (out_idx, acc);
                            }
                        }
                    }
                }
            }
        }
    }

  bool need_grad = Grad_Mode::is_enabled ()
                   && (input.requires_grad () || weight.requires_grad ()
                       || (bias.defined () && bias.requires_grad ()));
  if (need_grad)
    {
      auto node = std::make_shared<Conv3d_Backward> (
        input, weight, bias.defined (),
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        groups);
      if (bias.defined ())
        {
          node->set_next_edges (
            autograd::collect_next_edges ({input, weight, bias}));
        }
      else
        {
          node->set_next_edges (
            autograd::collect_next_edges ({input, weight}));
        }
      out.set_requires_grad (true);
      out.set_grad_fn (std::move (node));
    }
  return out;
}

Tensor
max_pool1d (const Tensor &input, size_t kernel_size,
            size_t stride, size_t padding)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 3, "max_pool1d() expects 3-D input");
  size_t st = (stride == 0) ? kernel_size : stride;
  Tensor x5 = input.unsqueeze (2).unsqueeze (3);
  Tensor y5 = max_pool3d (x5, 1, 1, kernel_size, 1, 1, st, 0, 0, padding);
  return y5.squeeze (2).squeeze (2);
}

Tensor
max_pool2d (const Tensor &input, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 4, "max_pool2d() expects 4-D input");
  size_t sh = (stride_h == 0) ? kernel_h : stride_h;
  size_t sw = (stride_w == 0) ? kernel_w : stride_w;
  Tensor x5 = input.unsqueeze (2);
  Tensor y5 = max_pool3d (x5, 1, kernel_h, kernel_w,
                          1, sh, sw, 0, pad_h, pad_w);
  return y5.squeeze (2);
}

Tensor
max_pool3d (const Tensor &input,
            size_t kernel_d, size_t kernel_h, size_t kernel_w,
            size_t stride_d, size_t stride_h, size_t stride_w,
            size_t pad_d, size_t pad_h, size_t pad_w)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 5, "max_pool3d() expects 5-D input");
  size_t sd = (stride_d == 0) ? kernel_d : stride_d;
  size_t sh = (stride_h == 0) ? kernel_h : stride_h;
  size_t sw = (stride_w == 0) ? kernel_w : stride_w;

  size_t batch = input.size (0);
  size_t channels = input.size (1);
  size_t in_d = input.size (2);
  size_t in_h = input.size (3);
  size_t in_w = input.size (4);

  size_t out_d = (in_d + 2 * pad_d - kernel_d) / sd + 1;
  size_t out_h = (in_h + 2 * pad_h - kernel_h) / sh + 1;
  size_t out_w = (in_w + 2 * pad_w - kernel_w) / sw + 1;

  Tensor out ({batch, channels, out_d, out_h, out_w},
              input.options ().with_requires_grad (false));
  std::vector<size_t> max_indices (out.numel (), 0);

  if (input.dtype () == Dtype::Float32)
    {
      Tensor x_c = input.is_contiguous () ? input : input.contiguous ();
      if (input.device ().is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      const float *xp = x_c.data_ptr<float> ();
      float *op = out.data_ptr<float> ();
      size_t nc_total = batch * channels;
      auto nc_fn = [&] (size_t nc)
      {
        for (size_t od = 0; od < out_d; ++od)
          {
            for (size_t oh = 0; oh < out_h; ++oh)
              {
                for (size_t ow = 0; ow < out_w; ++ow)
                  {
                    float best_val = -std::numeric_limits<float>::infinity ();
                    size_t best_idx = 0;
                    for (size_t fd = 0; fd < kernel_d; ++fd)
                      {
                        ptrdiff_t id = static_cast<ptrdiff_t> (od * sd + fd)
                                       - static_cast<ptrdiff_t> (pad_d);
                        if (id < 0 || id >= static_cast<ptrdiff_t> (in_d))
                          {
                            continue;
                          }
                        for (size_t fh = 0; fh < kernel_h; ++fh)
                          {
                            ptrdiff_t ih
                              = static_cast<ptrdiff_t> (oh * sh + fh)
                                - static_cast<ptrdiff_t> (pad_h);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t> (in_h))
                              {
                                continue;
                              }
                            for (size_t fw = 0; fw < kernel_w; ++fw)
                              {
                                ptrdiff_t iw
                                  = static_cast<ptrdiff_t> (ow * sw + fw)
                                    - static_cast<ptrdiff_t> (pad_w);
                                if (iw < 0
                                    || iw >= static_cast<ptrdiff_t> (in_w))
                                  {
                                    continue;
                                  }
                                size_t x_idx
                                  = ((nc * in_d + static_cast<size_t> (id))
                                     * in_h + static_cast<size_t> (ih)) * in_w
                                    + static_cast<size_t> (iw);
                                float v = xp[x_idx];
                                if (v > best_val)
                                  {
                                    best_val = v;
                                    best_idx = x_idx;
                                  }
                              }
                          }
                      }
                    size_t out_idx = ((nc * out_d + od) * out_h + oh) * out_w
                                     + ow;
                    op[out_idx] = best_val;
                    max_indices[out_idx] = best_idx;
                  }
              }
          }
      };
      if (out.numel () * kernel_d * kernel_h * kernel_w >= 65536)
        {
          hyper_math::parallel_for (size_t (0), nc_total, size_t (2), nc_fn);
        }
      else
        {
          for (size_t nc = 0; nc < nc_total; ++nc)
            {
              nc_fn (nc);
            }
        }
    }
  else
    {
      for (size_t n = 0; n < batch; ++n)
        {
          for (size_t c = 0; c < channels; ++c)
            {
              for (size_t od = 0; od < out_d; ++od)
                {
                  for (size_t oh = 0; oh < out_h; ++oh)
                    {
                      for (size_t ow = 0; ow < out_w; ++ow)
                        {
                          double best_val
                            = -std::numeric_limits<double>::infinity ();
                          size_t best_idx = 0;
                          for (size_t fd = 0; fd < kernel_d; ++fd)
                            {
                              ptrdiff_t id
                                = static_cast<ptrdiff_t> (od * sd + fd)
                                  - static_cast<ptrdiff_t> (pad_d);
                              if (id < 0
                                  || id >= static_cast<ptrdiff_t> (in_d))
                                {
                                  continue;
                                }
                              for (size_t fh = 0; fh < kernel_h; ++fh)
                                {
                                  ptrdiff_t ih
                                    = static_cast<ptrdiff_t> (oh * sh + fh)
                                      - static_cast<ptrdiff_t> (pad_h);
                                  if (ih < 0
                                      || ih >= static_cast<ptrdiff_t> (in_h))
                                    {
                                      continue;
                                    }
                                  for (size_t fw = 0; fw < kernel_w; ++fw)
                                    {
                                      ptrdiff_t iw
                                        = static_cast<ptrdiff_t> (ow * sw + fw)
                                          - static_cast<ptrdiff_t> (pad_w);
                                      if (iw < 0
                                          || iw
                                             >= static_cast<ptrdiff_t> (in_w))
                                        {
                                          continue;
                                        }
                                      size_t x_idx
                                        = (((n * channels + c) * in_d
                                            + static_cast<size_t> (id)) * in_h
                                           + static_cast<size_t> (ih)) * in_w
                                          + static_cast<size_t> (iw);
                                      double v = input.read_scalar_at (x_idx);
                                      if (v > best_val)
                                        {
                                          best_val = v;
                                          best_idx = x_idx;
                                        }
                                    }
                                }
                            }
                          size_t out_idx
                            = (((n * channels + c) * out_d + od) * out_h + oh)
                              * out_w + ow;
                          out.write_scalar_at (out_idx, best_val);
                          max_indices[out_idx] = best_idx;
                        }
                    }
                }
            }
        }
    }

  if (Grad_Mode::is_enabled () && input.requires_grad ())
    {
      auto node = std::make_shared<Max_Pool3d_Backward> (
        input.shape (), input.options (), std::move (max_indices));
      node->set_next_edges (autograd::collect_next_edges ({input}));
      out.set_requires_grad (true);
      out.set_grad_fn (std::move (node));
    }
  return out;
}

Tensor
avg_pool1d (const Tensor &input, size_t kernel_size,
            size_t stride, size_t padding)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 3, "avg_pool1d() expects 3-D input");
  size_t st = (stride == 0) ? kernel_size : stride;
  Tensor x5 = input.unsqueeze (2).unsqueeze (3);
  Tensor y5 = avg_pool3d (x5, 1, 1, kernel_size, 1, 1, st, 0, 0, padding);
  return y5.squeeze (2).squeeze (2);
}

Tensor
avg_pool2d (const Tensor &input, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 4, "avg_pool2d() expects 4-D input");
  size_t sh = (stride_h == 0) ? kernel_h : stride_h;
  size_t sw = (stride_w == 0) ? kernel_w : stride_w;
  Tensor x5 = input.unsqueeze (2);
  Tensor y5 = avg_pool3d (x5, 1, kernel_h, kernel_w,
                          1, sh, sw, 0, pad_h, pad_w);
  return y5.squeeze (2);
}

Tensor
avg_pool3d (const Tensor &input,
            size_t kernel_d, size_t kernel_h, size_t kernel_w,
            size_t stride_d, size_t stride_h, size_t stride_w,
            size_t pad_d, size_t pad_h, size_t pad_w)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 5, "avg_pool3d() expects 5-D input");
  size_t sd = (stride_d == 0) ? kernel_d : stride_d;
  size_t sh = (stride_h == 0) ? kernel_h : stride_h;
  size_t sw = (stride_w == 0) ? kernel_w : stride_w;

  size_t batch = input.size (0);
  size_t channels = input.size (1);
  size_t in_d = input.size (2);
  size_t in_h = input.size (3);
  size_t in_w = input.size (4);

  size_t out_d = (in_d + 2 * pad_d - kernel_d) / sd + 1;
  size_t out_h = (in_h + 2 * pad_h - kernel_h) / sh + 1;
  size_t out_w = (in_w + 2 * pad_w - kernel_w) / sw + 1;
  double inv_vol = 1.0 / static_cast<double> (kernel_d * kernel_h * kernel_w);

  Tensor out ({batch, channels, out_d, out_h, out_w},
              input.options ().with_requires_grad (false));

  if (input.dtype () == Dtype::Float32)
    {
      Tensor x_c = input.is_contiguous () ? input : input.contiguous ();
      if (input.device ().is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      const float *xp = x_c.data_ptr<float> ();
      float *op = out.data_ptr<float> ();
      float finv_vol = static_cast<float> (inv_vol);
      size_t nc_total = batch * channels;
      auto nc_fn = [&] (size_t nc)
      {
        for (size_t od = 0; od < out_d; ++od)
          {
            for (size_t oh = 0; oh < out_h; ++oh)
              {
                for (size_t ow = 0; ow < out_w; ++ow)
                  {
                    float acc = 0.0f;
                    for (size_t fd = 0; fd < kernel_d; ++fd)
                      {
                        ptrdiff_t id = static_cast<ptrdiff_t> (od * sd + fd)
                                       - static_cast<ptrdiff_t> (pad_d);
                        if (id < 0 || id >= static_cast<ptrdiff_t> (in_d))
                          {
                            continue;
                          }
                        for (size_t fh = 0; fh < kernel_h; ++fh)
                          {
                            ptrdiff_t ih
                              = static_cast<ptrdiff_t> (oh * sh + fh)
                                - static_cast<ptrdiff_t> (pad_h);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t> (in_h))
                              {
                                continue;
                              }
                            for (size_t fw = 0; fw < kernel_w; ++fw)
                              {
                                ptrdiff_t iw
                                  = static_cast<ptrdiff_t> (ow * sw + fw)
                                    - static_cast<ptrdiff_t> (pad_w);
                                if (iw < 0
                                    || iw >= static_cast<ptrdiff_t> (in_w))
                                  {
                                    continue;
                                  }
                                size_t x_idx
                                  = ((nc * in_d + static_cast<size_t> (id))
                                     * in_h + static_cast<size_t> (ih)) * in_w
                                    + static_cast<size_t> (iw);
                                acc += xp[x_idx];
                              }
                          }
                      }
                    size_t out_idx = ((nc * out_d + od) * out_h + oh) * out_w
                                     + ow;
                    op[out_idx] = acc * finv_vol;
                  }
              }
          }
      };
      if (out.numel () * kernel_d * kernel_h * kernel_w >= 65536)
        {
          hyper_math::parallel_for (size_t (0), nc_total, size_t (2), nc_fn);
        }
      else
        {
          for (size_t nc = 0; nc < nc_total; ++nc)
            {
              nc_fn (nc);
            }
        }
    }
  else
    {
      for (size_t n = 0; n < batch; ++n)
        {
          for (size_t c = 0; c < channels; ++c)
            {
              for (size_t od = 0; od < out_d; ++od)
                {
                  for (size_t oh = 0; oh < out_h; ++oh)
                    {
                      for (size_t ow = 0; ow < out_w; ++ow)
                        {
                          double acc = 0.0;
                          for (size_t fd = 0; fd < kernel_d; ++fd)
                            {
                              ptrdiff_t id
                                = static_cast<ptrdiff_t> (od * sd + fd)
                                  - static_cast<ptrdiff_t> (pad_d);
                              if (id < 0
                                  || id >= static_cast<ptrdiff_t> (in_d))
                                {
                                  continue;
                                }
                              for (size_t fh = 0; fh < kernel_h; ++fh)
                                {
                                  ptrdiff_t ih
                                    = static_cast<ptrdiff_t> (oh * sh + fh)
                                      - static_cast<ptrdiff_t> (pad_h);
                                  if (ih < 0
                                      || ih >= static_cast<ptrdiff_t> (in_h))
                                    {
                                      continue;
                                    }
                                  for (size_t fw = 0; fw < kernel_w; ++fw)
                                    {
                                      ptrdiff_t iw
                                        = static_cast<ptrdiff_t> (ow * sw + fw)
                                          - static_cast<ptrdiff_t> (pad_w);
                                      if (iw < 0
                                          || iw
                                             >= static_cast<ptrdiff_t> (in_w))
                                        {
                                          continue;
                                        }
                                      size_t x_idx
                                        = (((n * channels + c) * in_d
                                            + static_cast<size_t> (id)) * in_h
                                           + static_cast<size_t> (ih)) * in_w
                                          + static_cast<size_t> (iw);
                                      acc += input.read_scalar_at (x_idx);
                                    }
                                }
                            }
                          size_t out_idx
                            = (((n * channels + c) * out_d + od) * out_h + oh)
                              * out_w + ow;
                          out.write_scalar_at (out_idx, acc * inv_vol);
                        }
                    }
                }
            }
        }
    }

  if (Grad_Mode::is_enabled () && input.requires_grad ())
    {
      auto node = std::make_shared<Avg_Pool3d_Backward> (
        input.shape (), input.options (),
        kernel_d, kernel_h, kernel_w,
        sd, sh, sw, pad_d, pad_h, pad_w);
      node->set_next_edges (autograd::collect_next_edges ({input}));
      out.set_requires_grad (true);
      out.set_grad_fn (std::move (node));
    }
  return out;
}

Tensor
adaptive_avg_pool2d (const Tensor &input, size_t output_h, size_t output_w)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 4,
                       "adaptive_avg_pool2d() expects 4-D input");
  size_t in_h = input.size (2);
  size_t in_w = input.size (3);
  if (output_h == 1 && output_w == 1)
    {
      return ops::mean (ops::mean (input, 3, true), 2, true);
    }
  size_t sh = std::max<size_t> (1, in_h / output_h);
  size_t sw = std::max<size_t> (1, in_w / output_w);
  size_t kh = in_h - (output_h - 1) * sh;
  size_t kw = in_w - (output_w - 1) * sw;
  return avg_pool2d (input, kh, kw, sh, sw, 0, 0);
}

Tensor
batch_norm (const Tensor &input,
            Tensor &running_mean,
            Tensor &running_var,
            const Tensor &weight,
            const Tensor &bias,
            bool training,
            double momentum,
            double eps)
{
  BASALT_ASSERT_SHAPE (input.ndim () >= 2,
                       "batch_norm() requires ndim >= 2");
  size_t c = input.size (1);
  /* Permute channel dimension (1) to the last axis so we can reduce over all
     other axes cleanly, then permute back.  */
  std::vector<ptrdiff_t> to_last;
  std::vector<ptrdiff_t> from_last;
  to_last.push_back (0);
  for (size_t i = 2; i < input.ndim (); ++i)
    {
      to_last.push_back (static_cast<ptrdiff_t> (i));
    }
  to_last.push_back (1);

  from_last.push_back (0);
  from_last.push_back (static_cast<ptrdiff_t> (input.ndim () - 1));
  for (size_t i = 1; i + 1 < input.ndim (); ++i)
    {
      from_last.push_back (static_cast<ptrdiff_t> (i));
    }

  size_t total_per_c = input.numel () / c;
  Tensor flat = input.permute (to_last).reshape ({total_per_c, c});

  Tensor mean_c;
  Tensor var_c;
  if (training || !running_mean.defined () || !running_var.defined ())
    {
      mean_c = ops::mean (flat, 0, true);
      Tensor diff = ops::sub (flat, mean_c);
      var_c = ops::mean (ops::mul (diff, diff), 0, true);

      if (training && running_mean.defined () && running_var.defined ())
        {
          No_Grad_Guard no_grad;
          Tensor m_1d = mean_c.detach ().reshape ({c});
          Tensor v_unbiased = ops::mul_scalar (
            var_c.detach ().reshape ({c}),
            (total_per_c > 1)
              ? (static_cast<double> (total_per_c)
                 / static_cast<double> (total_per_c - 1))
              : 1.0);
          running_mean.mul_ (1.0 - momentum).add_ (m_1d, momentum);
          running_var.mul_ (1.0 - momentum).add_ (v_unbiased, momentum);
        }
    }
  else
    {
      mean_c = running_mean.reshape ({1, c});
      var_c = running_var.reshape ({1, c});
    }

  Tensor inv_std = ops::rsqrt (ops::add_scalar (var_c, eps));
  Tensor norm_flat = ops::mul (ops::sub (flat, mean_c), inv_std);
  if (weight.defined ())
    {
      norm_flat = ops::mul (norm_flat, weight.reshape ({1, c}));
    }
  if (bias.defined ())
    {
      norm_flat = ops::add (norm_flat, bias.reshape ({1, c}));
    }

  Shape_Vector perm_shape;
  perm_shape.push_back (input.size (0));
  for (size_t i = 2; i < input.ndim (); ++i)
    {
      perm_shape.push_back (input.size (static_cast<ptrdiff_t> (i)));
    }
  perm_shape.push_back (c);

  return norm_flat.reshape (perm_shape).permute (from_last);
}

namespace
{

class Layer_Norm_Backward : public Autograd_Node
{
public:
  Layer_Norm_Backward (const Tensor &input, const Tensor &weight,
                       bool has_weight, bool has_bias,
                       size_t outer_size, size_t inner_size, float eps)
    : input_ (input),
      weight_ (weight),
      has_weight_ (has_weight),
      has_bias_ (has_bias),
      outer_size_ (outer_size),
      inner_size_ (inner_size),
      eps_ (eps)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Layer_Norm_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard ng;
    Tensor x = input_.unpack ();
    Tensor w = has_weight_ ? weight_.unpack () : Tensor ();
    const Tensor &gy = grads[0];

    Tensor x_c = x.is_contiguous () ? x : x.contiguous ();
    Tensor gy_c = gy.is_contiguous () ? gy : gy.contiguous ();
    Tensor w_c = (has_weight_ && !w.is_contiguous ()) ? w.contiguous () : w;

    Tensor gx (x.shape (), x.options ().with_requires_grad (false));
    Tensor gw = has_weight_
                ? zeros (w.shape (), w.options ().with_requires_grad (false))
                : Tensor ();
    Tensor gb = has_bias_
                ? zeros ({inner_size_},
                         x.options ().with_requires_grad (false))
                : Tensor ();
    if (x.device ().is_cuda ())
      {
        bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
      }

    const float *xp = x_c.data_ptr<float> ();
    const float *gyp = gy_c.data_ptr<float> ();
    const float *wp = has_weight_ ? w_c.data_ptr<float> () : nullptr;
    float *gxp = gx.data_ptr<float> ();
    float *gwp = has_weight_ ? gw.data_ptr<float> () : nullptr;
    float *gbp = has_bias_ ? gb.data_ptr<float> () : nullptr;
    float inv_n = 1.0f / static_cast<float> (inner_size_);

    for (size_t r = 0; r < outer_size_; ++r)
      {
        const float *xr = xp + r * inner_size_;
        const float *gyr = gyp + r * inner_size_;
        float *gxr = gxp + r * inner_size_;

        float mean = 0.0f;
        for (size_t c = 0; c < inner_size_; ++c)
          {
            mean += xr[c];
          }
        mean *= inv_n;

        float var = 0.0f;
        for (size_t c = 0; c < inner_size_; ++c)
          {
            float d = xr[c] - mean;
            var += d * d;
          }
        float inv_std = 1.0f / std::sqrt (var * inv_n + eps_);

        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (size_t c = 0; c < inner_size_; ++c)
          {
            float xhat = (xr[c] - mean) * inv_std;
            float gy_v = gyr[c];
            if (has_weight_)
              {
                gwp[c] += gy_v * xhat;
              }
            if (has_bias_)
              {
                gbp[c] += gy_v;
              }
            float dy = has_weight_ ? (gy_v * wp[c]) : gy_v;
            sum_dy += dy;
            sum_dy_xhat += dy * xhat;
          }

        float scale = inv_std * inv_n;
        for (size_t c = 0; c < inner_size_; ++c)
          {
            float xhat = (xr[c] - mean) * inv_std;
            float dy = has_weight_ ? (gyr[c] * wp[c]) : gyr[c];
            gxr[c] = scale * (static_cast<float> (inner_size_) * dy
                              - sum_dy - xhat * sum_dy_xhat);
          }
      }

    std::vector<Tensor> out;
    out.push_back (gx);
    if (has_weight_)
      {
        out.push_back (gw);
      }
    if (has_bias_)
      {
        out.push_back (gb);
      }
    return out;
  }

  void
  release_saved_variables (void) noexcept override
  {
    input_ = Saved_Variable ();
    weight_ = Saved_Variable ();
  }

private:
  Saved_Variable input_;
  Saved_Variable weight_;
  bool has_weight_;
  bool has_bias_;
  size_t outer_size_;
  size_t inner_size_;
  float eps_;
};

} /* namespace */

Tensor
layer_norm (const Tensor &input,
            const Shape_Vector &normalized_shape,
            const Tensor &weight,
            const Tensor &bias,
            double eps)
{
  BASALT_ASSERT_SHAPE (!normalized_shape.empty ()
                       && input.ndim () >= normalized_shape.size (),
                       "layer_norm() invalid normalized_shape");
  size_t norm_dims = normalized_shape.size ();
  size_t outer_dims = input.ndim () - norm_dims;
  size_t outer_size = 1;
  for (size_t i = 0; i < outer_dims; ++i)
    {
      outer_size *= input.size (static_cast<ptrdiff_t> (i));
    }
  size_t inner_size = 1;
  for (size_t i = 0; i < norm_dims; ++i)
    {
      size_t actual = input.size (static_cast<ptrdiff_t> (outer_dims + i));
      BASALT_ASSERT_SHAPE (actual == normalized_shape[i],
                           "layer_norm() trailing shape mismatch");
      inner_size *= actual;
    }

  if (input.is_contiguous () && input.dtype () == Dtype::Float32
      && (!weight.defined ()
          || (weight.is_contiguous () && weight.dtype () == Dtype::Float32))
      && (!bias.defined ()
          || (bias.is_contiguous () && bias.dtype () == Dtype::Float32))
      && inner_size > 0)
    {
      Tensor out (input.shape (), input.options ().with_requires_grad (false));
      const float *xp = input.data_ptr<float> ();
      const float *wp
        = weight.defined () ? weight.data_ptr<float> () : nullptr;
      const float *bp
        = bias.defined () ? bias.data_ptr<float> () : nullptr;
      float *yp = out.data_ptr<float> ();
      float inv_n = 1.0f / static_cast<float> (inner_size);
      float feps = static_cast<float> (eps);

      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      bool gpu_done = false;
      if (input.device ().is_cuda ()
          && cuda_drv.launch_layernorm_fwd_f32 (
               input.device ().index (), xp, wp, bp, yp,
               outer_size, inner_size, feps))
        {
          gpu_done = true;
        }
      if (!gpu_done && input.device ().is_cuda ())
        {
          cuda_drv.sync_if_busy ();
        }

      auto row_fn = [&] (size_t r)
      {
        const float *xr = xp + r * inner_size;
        float *yr = yp + r * inner_size;
        float mean = 0.0f;
        for (size_t c = 0; c < inner_size; ++c)
          {
            mean += xr[c];
          }
        mean *= inv_n;
        float var = 0.0f;
        for (size_t c = 0; c < inner_size; ++c)
          {
            float d = xr[c] - mean;
            var += d * d;
          }
        float inv_std = 1.0f / std::sqrt (var * inv_n + feps);
        if (wp != nullptr && bp != nullptr)
          {
            for (size_t c = 0; c < inner_size; ++c)
              {
                yr[c] = (xr[c] - mean) * inv_std * wp[c] + bp[c];
              }
          }
        else if (wp != nullptr)
          {
            for (size_t c = 0; c < inner_size; ++c)
              {
                yr[c] = (xr[c] - mean) * inv_std * wp[c];
              }
          }
        else if (bp != nullptr)
          {
            for (size_t c = 0; c < inner_size; ++c)
              {
                yr[c] = (xr[c] - mean) * inv_std + bp[c];
              }
          }
        else
          {
            for (size_t c = 0; c < inner_size; ++c)
              {
                yr[c] = (xr[c] - mean) * inv_std;
              }
          }
      };

      if (!gpu_done)
        {
          if (outer_size * inner_size >= 65536 && outer_size >= 8)
            {
              hyper_math::parallel_for (size_t (0), outer_size, size_t (16),
                                        row_fn);
            }
          else
            {
              for (size_t r = 0; r < outer_size; ++r)
                {
                  row_fn (r);
                }
            }
        }

      bool need_grad = Grad_Mode::is_enabled ()
                       && (input.requires_grad ()
                           || (weight.defined () && weight.requires_grad ())
                           || (bias.defined () && bias.requires_grad ()));
      if (need_grad)
        {
          auto node = std::make_shared<Layer_Norm_Backward> (
            input, weight, weight.defined (), bias.defined (),
            outer_size, inner_size, feps);
          std::vector<Tensor> inputs_vec;
          inputs_vec.push_back (input);
          if (weight.defined ())
            {
              inputs_vec.push_back (weight);
            }
          if (bias.defined ())
            {
              inputs_vec.push_back (bias);
            }
          node->set_next_edges (autograd::collect_next_edges (inputs_vec));
          out.set_requires_grad (true);
          out.set_grad_fn (std::move (node));
        }
      return out;
    }

  Tensor flat = input.reshape ({outer_size, inner_size});
  Tensor m = ops::mean (flat, 1, true);
  Tensor diff = ops::sub (flat, m);
  Tensor v = ops::mean (ops::mul (diff, diff), 1, true);
  Tensor normed = ops::mul (diff, ops::rsqrt (ops::add_scalar (v, eps)))
                    .reshape (input.shape ());

  if (weight.defined ())
    {
      normed = ops::mul (normed, weight);
    }
  if (bias.defined ())
    {
      normed = ops::add (normed, bias);
    }
  return normed;
}

Tensor
group_norm (const Tensor &input,
            size_t num_groups,
            const Tensor &weight,
            const Tensor &bias,
            double eps)
{
  BASALT_ASSERT_SHAPE (input.ndim () >= 2, "group_norm() requires ndim >= 2");
  size_t batch = input.size (0);
  size_t c = input.size (1);
  BASALT_ASSERT_SHAPE (num_groups >= 1 && c % num_groups == 0,
                       "group_norm() channels must be divisible by groups");
  size_t spatial = input.numel () / (batch * c);
  size_t group_elems = (c / num_groups) * spatial;

  Tensor grouped = input.reshape ({batch * num_groups, group_elems});
  Tensor m = ops::mean (grouped, 1, true);
  Tensor diff = ops::sub (grouped, m);
  Tensor v = ops::mean (ops::mul (diff, diff), 1, true);
  Tensor normed = ops::mul (diff, ops::rsqrt (ops::add_scalar (v, eps)))
                    .reshape (input.shape ());

  if (weight.defined () || bias.defined ())
    {
      Shape_Vector affine_shape (input.ndim (), 1);
      affine_shape[1] = c;
      if (weight.defined ())
        {
          normed = ops::mul (normed, weight.reshape (affine_shape));
        }
      if (bias.defined ())
        {
          normed = ops::add (normed, bias.reshape (affine_shape));
        }
    }
  return normed;
}

Tensor
rms_norm (const Tensor &input,
          const Shape_Vector &normalized_shape,
          const Tensor &weight,
          double eps)
{
  BASALT_ASSERT_SHAPE (!normalized_shape.empty ()
                       && input.ndim () >= normalized_shape.size (),
                       "rms_norm() invalid normalized_shape");
  size_t norm_dims = normalized_shape.size ();
  size_t outer_dims = input.ndim () - norm_dims;
  size_t outer_size = 1;
  for (size_t i = 0; i < outer_dims; ++i)
    {
      outer_size *= input.size (static_cast<ptrdiff_t> (i));
    }
  size_t inner_size = 1;
  for (size_t i = 0; i < norm_dims; ++i)
    {
      size_t actual = input.size (static_cast<ptrdiff_t> (outer_dims + i));
      BASALT_ASSERT_SHAPE (actual == normalized_shape[i],
                           "rms_norm() trailing shape mismatch");
      inner_size *= actual;
    }

  Tensor flat = input.reshape ({outer_size, inner_size});
  Tensor ms = ops::mean (ops::square (flat), 1, true);
  Tensor normed = ops::mul (flat, ops::rsqrt (ops::add_scalar (ms, eps)))
                    .reshape (input.shape ());
  if (weight.defined ())
    {
      normed = ops::mul (normed, weight);
    }
  return normed;
}

Tensor
embedding (const Tensor &indices, const Tensor &weight, ptrdiff_t padding_idx)
{
  BASALT_ASSERT_SHAPE (weight.ndim () == 2,
                       "embedding() requires 2-D weight matrix");
  Tensor flat_idx = indices.reshape ({indices.numel ()});
  Tensor selected = ops::index_select (weight, 0, flat_idx);

  Shape_Vector out_shape = indices.shape ();
  out_shape.push_back (weight.size (1));
  Tensor out = selected.reshape (out_shape);

  if (padding_idx >= 0)
    {
      Tensor pad_val = full (indices.shape (),
                             static_cast<double> (padding_idx),
                             indices.options ().with_requires_grad (false));
      Tensor is_pad = ops::eq (indices, pad_val).unsqueeze (-1)
                        .expand (out_shape);
      out = ops::where (is_pad, zeros_like (out), out);
    }
  return out;
}

Tensor
dropout (const Tensor &input, double p, bool training)
{
  BASALT_ASSERT_ARG (p >= 0.0 && p <= 1.0,
                     "dropout() probability p must be in [0, 1]");
  if (!training || p == 0.0)
    {
      return input;
    }
  if (p == 1.0)
    {
      return zeros_like (input);
    }
  double keep_prob = 1.0 - p;
  double scale = 1.0 / keep_prob;
  Tensor mask (input.shape (),
               input.options ().with_requires_grad (false));
  size_t n = mask.numel ();
  Random_Generator &gen = default_generator ();
  for (size_t i = 0; i < n; ++i)
    {
      double u = gen.uniform (0.0, 1.0);
      mask.write_scalar_at (i, (u < keep_prob) ? scale : 0.0);
    }
  return ops::mul (input, mask);
}

namespace
{

class Flash_Attention_Backward : public Autograd_Node
{
public:
  Flash_Attention_Backward (const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &out, float scale, bool is_causal)
    : q_ (q),
      k_ (k),
      v_ (v),
      out_ (out, true),
      scale_ (scale),
      is_causal_ (is_causal)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Flash_Attention_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard ng;
    Tensor q = q_.unpack ();
    Tensor k = k_.unpack ();
    Tensor v = v_.unpack ();
    Tensor o = out_.unpack ();
    const Tensor &do_t = grads[0];

    Tensor q_c = q.is_contiguous () ? q : q.contiguous ();
    Tensor k_c = k.is_contiguous () ? k : k.contiguous ();
    Tensor v_c = v.is_contiguous () ? v : v.contiguous ();
    Tensor o_c = o.is_contiguous () ? o : o.contiguous ();
    Tensor do_c = do_t.is_contiguous () ? do_t : do_t.contiguous ();

    Tensor dq = zeros (q.shape (), q.options ().with_requires_grad (false));
    Tensor dk_t = zeros (k.shape (), k.options ().with_requires_grad (false));
    Tensor dv_t = zeros (v.shape (), v.options ().with_requires_grad (false));

    if (q.device ().is_cuda ())
      {
        bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
      }

    size_t q_len = q.size (-2);
    size_t k_len = k.size (-2);
    size_t dk = q.size (-1);
    size_t dv = v.size (-1);
    size_t heads = q.numel () / (q_len * dk);

    const float *qp = q_c.data_ptr<float> ();
    const float *kp = k_c.data_ptr<float> ();
    const float *vp = v_c.data_ptr<float> ();
    const float *op = o_c.data_ptr<float> ();
    const float *dop = do_c.data_ptr<float> ();
    float *dqp = dq.data_ptr<float> ();
    float *dkp = dk_t.data_ptr<float> ();
    float *dvp = dv_t.data_ptr<float> ();

    for (size_t b = 0; b < heads; ++b)
      {
        const float *q_mat = qp + b * q_len * dk;
        const float *k_mat = kp + b * k_len * dk;
        const float *v_mat = vp + b * k_len * dv;
        const float *o_mat = op + b * q_len * dv;
        const float *do_mat = dop + b * q_len * dv;
        float *dq_mat = dqp + b * q_len * dk;
        float *dk_mat = dkp + b * k_len * dk;
        float *dv_mat = dvp + b * k_len * dv;

        std::vector<float> p_row (k_len, 0.0f);
        for (size_t qi = 0; qi < q_len; ++qi)
          {
            const float *q_row = q_mat + qi * dk;
            const float *o_row = o_mat + qi * dv;
            const float *do_row = do_mat + qi * dv;
            float *dq_row = dq_mat + qi * dk;

            size_t j_limit = is_causal_ ? std::min (k_len, qi + 1) : k_len;
            float m_i = -std::numeric_limits<float>::infinity ();
            for (size_t kj = 0; kj < j_limit; ++kj)
              {
                const float *k_row = k_mat + kj * dk;
                float dot = 0.0f;
                for (size_t d = 0; d < dk; ++d)
                  {
                    dot += q_row[d] * k_row[d];
                  }
                float s_ij = dot * scale_;
                p_row[kj] = s_ij;
                if (s_ij > m_i)
                  {
                    m_i = s_ij;
                  }
              }
            float l_i = 0.0f;
            for (size_t kj = 0; kj < j_limit; ++kj)
              {
                float e = std::exp (p_row[kj] - m_i);
                p_row[kj] = e;
                l_i += e;
              }
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            for (size_t kj = 0; kj < j_limit; ++kj)
              {
                p_row[kj] *= inv_l;
              }

            /* D_i = sum_d (do_row[d] * o_row[d]) */
            float d_i = 0.0f;
            for (size_t d = 0; d < dv; ++d)
              {
                d_i += do_row[d] * o_row[d];
              }

            for (size_t kj = 0; kj < j_limit; ++kj)
              {
                float p_ij = p_row[kj];
                const float *k_row = k_mat + kj * dk;
                const float *v_row = v_mat + kj * dv;
                float *dk_row = dk_mat + kj * dk;
                float *dv_row = dv_mat + kj * dv;

                float dp_ij = 0.0f;
                for (size_t d = 0; d < dv; ++d)
                  {
                    dv_row[d] += p_ij * do_row[d];
                    dp_ij += do_row[d] * v_row[d];
                  }
                float ds_ij = p_ij * (dp_ij - d_i) * scale_;
                for (size_t d = 0; d < dk; ++d)
                  {
                    dq_row[d] += ds_ij * k_row[d];
                    dk_row[d] += ds_ij * q_row[d];
                  }
              }
          }
      }
    return {dq, dk_t, dv_t};
  }

  void
  release_saved_variables (void) noexcept override
  {
    q_ = Saved_Variable ();
    k_ = Saved_Variable ();
    v_ = Saved_Variable ();
    out_ = Saved_Variable ();
  }

private:
  Saved_Variable q_;
  Saved_Variable k_;
  Saved_Variable v_;
  Saved_Variable out_;
  float scale_;
  bool is_causal_;
};

} /* namespace */

Tensor
scaled_dot_product_attention (const Tensor &query,
                              const Tensor &key,
                              const Tensor &value,
                              const Tensor &attn_mask,
                              double dropout_p,
                              bool is_causal,
                              bool training)
{
  BASALT_ASSERT_SHAPE (query.ndim () >= 2 && key.ndim () >= 2
                       && value.ndim () >= 2,
                       "scaled_dot_product_attention() requires rank >= 2");
  double d_k = static_cast<double> (query.size (-1));
  double scale = 1.0 / std::sqrt (d_k);

  bool need_grad = Grad_Mode::is_enabled ()
                   && (query.requires_grad () || key.requires_grad ()
                       || value.requires_grad ()
                       || (attn_mask.defined ()
                           && attn_mask.requires_grad ()));

  /* Online-softmax FlashAttention fast path (O(1) extra memory per row, both
     forward and backward) on CPU when no arbitrary attention or dropout mask
     is active and either gradients are disabled or the sequence attention
     matrix exceeds L1 cache (q_len * k_len >= 1024).  On CUDA, we dispatch
     through cuBLAS strided batched GEMM + NVRTC softmax below.  */
  if (!query.device ().is_cuda () && !attn_mask.defined ()
      && (!training || dropout_p == 0.0)
      && query.dtype () == Dtype::Float32 && key.dtype () == Dtype::Float32
      && value.dtype () == Dtype::Float32
      && query.ndim () == key.ndim () && query.ndim () == value.ndim ())
    {
      size_t q_len = query.size (-2);
      size_t k_len = key.size (-2);
      size_t dk = query.size (-1);
      size_t dv = value.size (-1);
      size_t q_heads = query.numel () / (q_len * dk);
      size_t k_heads = key.numel () / (k_len * dk);
      size_t v_heads = value.numel () / (k_len * dv);
      if (q_heads == k_heads && q_heads == v_heads && q_len > 0 && k_len > 0
          && (!need_grad || (q_len * k_len >= 1024)))
        {
          No_Grad_Guard ng;
          Tensor q_c = query.is_contiguous () ? query : query.contiguous ();
          Tensor k_c = key.is_contiguous () ? key : key.contiguous ();
          Tensor v_c = value.is_contiguous () ? value : value.contiguous ();
          Shape_Vector out_shape = query.shape ();
          out_shape.back () = dv;
          Tensor out (out_shape,
                      query.options ().with_requires_grad (false));
          if (query.device ().is_cuda ())
            {
              bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
            }

          const float *qp = q_c.data_ptr<float> ();
          const float *kp = k_c.data_ptr<float> ();
          const float *vp = v_c.data_ptr<float> ();
          float *op = out.data_ptr<float> ();
          float fscale = static_cast<float> (scale);
          size_t total_q_rows = q_heads * q_len;

          auto q_row_fn = [&] (size_t row_idx)
          {
            size_t b = row_idx / q_len;
            size_t qi = row_idx % q_len;
            const float *q_row = qp + (b * q_len + qi) * dk;
            const float *k_mat = kp + b * k_len * dk;
            const float *v_mat = vp + b * k_len * dv;
            float *o_row = op + (b * q_len + qi) * dv;
            std::fill_n (o_row, dv, 0.0f);

            size_t j_limit = is_causal ? std::min (k_len, qi + 1) : k_len;
            float m_i = -std::numeric_limits<float>::infinity ();
            float l_i = 0.0f;

            for (size_t kj = 0; kj < j_limit; ++kj)
              {
                const float *k_row = k_mat + kj * dk;
                float dot = 0.0f;
                for (size_t d = 0; d < dk; ++d)
                  {
                    dot += q_row[d] * k_row[d];
                  }
                float s_ij = dot * fscale;
                float m_new = std::max (m_i, s_ij);
                float neg_inf = -std::numeric_limits<float>::infinity ();
                float alpha = (m_i == neg_inf)
                              ? 0.0f
                              : std::exp (m_i - m_new);
                float beta = std::exp (s_ij - m_new);
                l_i = alpha * l_i + beta;
                m_i = m_new;
                const float *v_row = v_mat + kj * dv;
                for (size_t d = 0; d < dv; ++d)
                  {
                    o_row[d] = alpha * o_row[d] + beta * v_row[d];
                  }
              }
            if (l_i > 0.0f)
              {
                float inv_l = 1.0f / l_i;
                for (size_t d = 0; d < dv; ++d)
                  {
                    o_row[d] *= inv_l;
                  }
              }
          };
          if (total_q_rows * k_len * dk >= 65536)
            {
              hyper_math::parallel_for (
                size_t (0), total_q_rows, size_t (8), q_row_fn);
            }
          else
            {
              for (size_t row_idx = 0; row_idx < total_q_rows; ++row_idx)
                {
                  q_row_fn (row_idx);
                }
            }
          if (need_grad)
            {
              auto node = std::make_shared<Flash_Attention_Backward> (
                query, key, value, out, fscale, is_causal);
              node->set_next_edges (
                autograd::collect_next_edges ({query, key, value}));
              out.set_requires_grad (true);
              out.set_grad_fn (std::move (node));
            }
          return out;
        }
    }

  Tensor scores = ops::mul_scalar (
    ops::matmul (query, key.transpose (-2, -1)), scale);

  if (is_causal)
    {
      size_t q_len = scores.size (-2);
      size_t k_len = scores.size (-1);
      Tensor upper_mask ({q_len, k_len},
                         Tensor_Options ()
                           .with_dtype (Dtype::Bool)
                           .with_device (scores.device ()));
      for (size_t r = 0; r < q_len; ++r)
        {
          for (size_t c = 0; c < k_len; ++c)
            {
              upper_mask.write_scalar_at (r * k_len + c,
                                          (c > r) ? 1.0 : 0.0);
            }
        }
      scores = ops::masked_fill (scores, upper_mask, -1e9);
    }

  if (attn_mask.defined ())
    {
      if (attn_mask.dtype () == Dtype::Bool)
        {
          scores = ops::masked_fill (scores, attn_mask, -1e9);
        }
      else
        {
          scores = ops::add (scores, attn_mask);
        }
    }

  Tensor attn_weights = ops::softmax (scores, -1);
  if (dropout_p > 0.0 && training)
    {
      attn_weights = dropout (attn_weights, dropout_p, training);
    }
  return ops::matmul (attn_weights, value);
}

Tensor
apply_rotary_pos_emb (const Tensor &x,
                      const Tensor &cos_cached,
                      const Tensor &sin_cached)
{
  BASALT_ASSERT_SHAPE (x.ndim () >= 2,
                       "apply_rotary_pos_emb() requires ndim >= 2");
  size_t seq_len = x.size (-2);
  size_t head_dim = x.size (-1);
  BASALT_ASSERT_SHAPE (head_dim % 2 == 0,
                       "apply_rotary_pos_emb() requires even head_dim");
  size_t half = head_dim / 2;

  Tensor c = (cos_cached.size (-2) > seq_len)
             ? cos_cached.narrow (-2, 0, seq_len)
             : cos_cached;
  Tensor s = (sin_cached.size (-2) > seq_len)
             ? sin_cached.narrow (-2, 0, seq_len)
             : sin_cached;

  Tensor x1 = x.narrow (-1, 0, half);
  Tensor x2 = x.narrow (-1, half, half);
  Tensor rot_x = ops::cat ({ops::neg (x2), x1}, -1);
  return ops::add (ops::mul (x, c), ops::mul (rot_x, s));
}

namespace
{

class Conv_Transpose2d_Backward : public Autograd_Node
{
public:
  Conv_Transpose2d_Backward (Tensor input, Tensor weight, bool has_bias,
                             size_t stride_h, size_t stride_w,
                             size_t pad_h, size_t pad_w,
                             size_t dilation_h, size_t dilation_w,
                             size_t groups)
    : input_ (std::move (input)),
      weight_ (std::move (weight)),
      has_bias_ (has_bias),
      stride_h_ (stride_h),
      stride_w_ (stride_w),
      pad_h_ (pad_h),
      pad_w_ (pad_w),
      dilation_h_ (dilation_h),
      dilation_w_ (dilation_w),
      groups_ (groups)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Conv_Transpose2d_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard ng;
    Device orig_dev = input_.device ();
    Tensor gy = grads[0].cpu ().contiguous ();
    Tensor x_cpu = input_.cpu ().contiguous ();
    Tensor w_cpu = weight_.cpu ().contiguous ();

    size_t batch = x_cpu.size (0);
    size_t c_in = x_cpu.size (1);
    size_t in_h = x_cpu.size (2);
    size_t in_w = x_cpu.size (3);

    size_t c_out_g = w_cpu.size (1);
    size_t k_h = w_cpu.size (2);
    size_t k_w = w_cpu.size (3);
    size_t c_out = c_out_g * groups_;
    size_t c_in_g = c_in / groups_;
    size_t out_h = gy.size (2);
    size_t out_w = gy.size (3);

    Tensor_Options cpu_opts
      = x_cpu.options ().with_device (Device::cpu ())
                        .with_requires_grad (false);
    Tensor gx = zeros (x_cpu.shape (), cpu_opts);
    Tensor gw = zeros (w_cpu.shape (), cpu_opts);
    Tensor gb = has_bias_ ? zeros ({c_out}, cpu_opts) : Tensor ();

    size_t patch_len = c_out_g * k_h * k_w;
    size_t in_spatial = in_h * in_w;

    for (size_t n = 0; n < batch; ++n)
      {
        if (has_bias_)
          {
            for (size_t oc = 0; oc < c_out; ++oc)
              {
                double b_sum = gb.read_scalar_at (oc);
                size_t base = (n * c_out + oc) * out_h * out_w;
                for (size_t idx = 0; idx < out_h * out_w; ++idx)
                  {
                    b_sum += gy.read_scalar_at (base + idx);
                  }
                gb.write_scalar_at (oc, b_sum);
              }
          }

        for (size_t g = 0; g < groups_; ++g)
          {
            Tensor gy_col = zeros ({patch_len, in_spatial}, cpu_opts);
            for (size_t oc = 0; oc < c_out_g; ++oc)
              {
                size_t abs_oc = g * c_out_g + oc;
                for (size_t kh = 0; kh < k_h; ++kh)
                  {
                    for (size_t kw = 0; kw < k_w; ++kw)
                      {
                        size_t row = (oc * k_h + kh) * k_w + kw;
                        for (size_t ih = 0; ih < in_h; ++ih)
                          {
                            ptrdiff_t oh
                              = static_cast<ptrdiff_t> (ih * stride_h_
                                                        + kh * dilation_h_)
                                - static_cast<ptrdiff_t> (pad_h_);
                            if (oh < 0
                                || static_cast<size_t> (oh) >= out_h)
                              {
                                continue;
                              }
                            for (size_t iw = 0; iw < in_w; ++iw)
                              {
                                ptrdiff_t ow
                                  = static_cast<ptrdiff_t> (iw * stride_w_
                                                            + kw * dilation_w_)
                                    - static_cast<ptrdiff_t> (pad_w_);
                                if (ow < 0
                                    || static_cast<size_t> (ow) >= out_w)
                                  {
                                    continue;
                                  }
                                size_t col = ih * in_w + iw;
                                size_t gy_idx
                                  = ((n * c_out + abs_oc) * out_h
                                     + static_cast<size_t> (oh))
                                      * out_w
                                    + static_cast<size_t> (ow);
                                gy_col.write_scalar_at (
                                  row * in_spatial + col,
                                  gy.read_scalar_at (gy_idx));
                              }
                          }
                      }
                  }
              }

            Tensor w_g = w_cpu
                           .narrow (0, static_cast<ptrdiff_t> (g * c_in_g),
                                    c_in_g)
                           .contiguous ()
                           .reshape ({c_in_g, patch_len});
            Tensor x_g = x_cpu
                           .select (0, static_cast<ptrdiff_t> (n))
                           .narrow (0, static_cast<ptrdiff_t> (g * c_in_g),
                                    c_in_g)
                           .contiguous ()
                           .reshape ({c_in_g, in_spatial});

            Tensor gx_g = ops::matmul (w_g, gy_col)
                            .reshape ({c_in_g, in_h, in_w});
            gx.select (0, static_cast<ptrdiff_t> (n))
              .narrow (0, static_cast<ptrdiff_t> (g * c_in_g), c_in_g)
              .copy_ (gx_g);

            Tensor gw_g = ops::matmul (x_g, gy_col.transpose (0, 1))
                            .reshape ({c_in_g, c_out_g, k_h, k_w});
            Tensor gw_slice
              = gw.narrow (0, static_cast<ptrdiff_t> (g * c_in_g), c_in_g);
            gw_slice.copy_ (ops::add (gw_slice, gw_g));
          }
      }

    if (has_bias_)
      {
        return {gx.to (orig_dev), gw.to (orig_dev), gb.to (orig_dev)};
      }
    return {gx.to (orig_dev), gw.to (orig_dev)};
  }

private:
  Tensor input_;
  Tensor weight_;
  bool has_bias_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  size_t dilation_h_;
  size_t dilation_w_;
  size_t groups_;
};

} /* namespace */

Tensor
conv_transpose2d (const Tensor &input, const Tensor &weight,
                  const Tensor &bias,
                  size_t stride_h, size_t stride_w,
                  size_t pad_h, size_t pad_w,
                  size_t out_pad_h, size_t out_pad_w,
                  size_t dilation_h, size_t dilation_w,
                  size_t groups)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 4 && weight.ndim () == 4,
                       "conv_transpose2d() expects 4-D input and weight");
  BASALT_ASSERT_ARG (stride_h >= 1 && stride_w >= 1
                     && dilation_h >= 1 && dilation_w >= 1 && groups >= 1,
                     "conv_transpose2d() stride, dilation, groups must >= 1");
  BASALT_ASSERT_ARG (out_pad_h < stride_h && out_pad_w < stride_w,
                     "conv_transpose2d() output_padding must be < stride");

  size_t batch = input.size (0);
  size_t c_in = input.size (1);
  size_t in_h = input.size (2);
  size_t in_w = input.size (3);

  BASALT_ASSERT_SHAPE (weight.size (0) == c_in && c_in % groups == 0,
                       "conv_transpose2d() weight in_channels mismatch");
  size_t c_out_g = weight.size (1);
  size_t c_out = c_out_g * groups;
  size_t k_h = weight.size (2);
  size_t k_w = weight.size (3);
  size_t c_in_g = c_in / groups;

  ptrdiff_t out_h_signed
    = static_cast<ptrdiff_t> ((in_h - 1) * stride_h
                              + dilation_h * (k_h - 1) + out_pad_h + 1)
      - static_cast<ptrdiff_t> (2 * pad_h);
  ptrdiff_t out_w_signed
    = static_cast<ptrdiff_t> ((in_w - 1) * stride_w
                              + dilation_w * (k_w - 1) + out_pad_w + 1)
      - static_cast<ptrdiff_t> (2 * pad_w);
  BASALT_ASSERT_SHAPE (out_h_signed > 0 && out_w_signed > 0,
                       "conv_transpose2d() non-positive output spatial size");
  size_t out_h = static_cast<size_t> (out_h_signed);
  size_t out_w = static_cast<size_t> (out_w_signed);

  bool need_grad = Grad_Mode::is_enabled ()
                   && (input.requires_grad () || weight.requires_grad ()
                       || (bias.defined () && bias.requires_grad ()));

  No_Grad_Guard ng;
  Tensor x_cpu = input.detach ().cpu ().contiguous ();
  Tensor w_cpu = weight.detach ().cpu ().contiguous ();
  Tensor b_cpu = bias.defined () ? bias.detach ().cpu ().contiguous ()
                                 : Tensor ();

  Tensor_Options cpu_opts
    = input.options ().with_device (Device::cpu ())
                      .with_requires_grad (false);
  Tensor out_cpu = zeros ({batch, c_out, out_h, out_w}, cpu_opts);

  size_t patch_len = c_out_g * k_h * k_w;
  size_t in_spatial = in_h * in_w;

  for (size_t n = 0; n < batch; ++n)
    {
      for (size_t g = 0; g < groups; ++g)
        {
          Tensor w_g = w_cpu
                         .narrow (0, static_cast<ptrdiff_t> (g * c_in_g),
                                  c_in_g)
                         .contiguous ()
                         .reshape ({c_in_g, patch_len});
          Tensor x_g = x_cpu
                         .select (0, static_cast<ptrdiff_t> (n))
                         .narrow (0, static_cast<ptrdiff_t> (g * c_in_g),
                                  c_in_g)
                         .contiguous ()
                         .reshape ({c_in_g, in_spatial});

          Tensor cols = ops::matmul (w_g.transpose (0, 1), x_g);
          for (size_t oc = 0; oc < c_out_g; ++oc)
            {
              size_t abs_oc = g * c_out_g + oc;
              for (size_t kh = 0; kh < k_h; ++kh)
                {
                  for (size_t kw = 0; kw < k_w; ++kw)
                    {
                      size_t row = (oc * k_h + kh) * k_w + kw;
                      for (size_t ih = 0; ih < in_h; ++ih)
                        {
                          ptrdiff_t oh
                            = static_cast<ptrdiff_t> (ih * stride_h
                                                      + kh * dilation_h)
                              - static_cast<ptrdiff_t> (pad_h);
                          if (oh < 0 || static_cast<size_t> (oh) >= out_h)
                            {
                              continue;
                            }
                          for (size_t iw = 0; iw < in_w; ++iw)
                            {
                              ptrdiff_t ow
                                = static_cast<ptrdiff_t> (iw * stride_w
                                                          + kw * dilation_w)
                                  - static_cast<ptrdiff_t> (pad_w);
                              if (ow < 0 || static_cast<size_t> (ow) >= out_w)
                                {
                                  continue;
                                }
                              size_t col = ih * in_w + iw;
                              size_t out_idx
                                = ((n * c_out + abs_oc) * out_h
                                   + static_cast<size_t> (oh))
                                    * out_w
                                  + static_cast<size_t> (ow);
                              out_cpu.write_scalar_at (
                                out_idx,
                                out_cpu.read_scalar_at (out_idx)
                                  + cols.read_scalar_at (
                                      row * in_spatial + col));
                            }
                        }
                    }
                }
            }
        }

      if (b_cpu.defined ())
        {
          for (size_t oc = 0; oc < c_out; ++oc)
            {
              double b_val = b_cpu.read_scalar_at (oc);
              size_t base = (n * c_out + oc) * out_h * out_w;
              for (size_t idx = 0; idx < out_h * out_w; ++idx)
                {
                  out_cpu.write_scalar_at (
                    base + idx, out_cpu.read_scalar_at (base + idx) + b_val);
                }
            }
        }
    }

  Tensor out = out_cpu.to (input.device ());
  if (need_grad)
    {
      auto node = std::make_shared<Conv_Transpose2d_Backward> (
        input.detach (), weight.detach (), bias.defined (),
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups);
      if (bias.defined ())
        {
          node->set_next_edges (
            autograd::collect_next_edges ({input, weight, bias}));
        }
      else
        {
          node->set_next_edges (
            autograd::collect_next_edges ({input, weight}));
        }
      out.set_requires_grad (true);
      out.set_grad_fn (std::move (node));
    }
  return out;
}

Tensor
cosine_similarity (const Tensor &x1, const Tensor &x2,
                   ptrdiff_t dim, double eps)
{
  BASALT_ASSERT_SHAPE (x1.shape () == x2.shape () && x1.ndim () >= 1,
                       "cosine_similarity() requires matching tensor shapes");
  Tensor dot = ops::sum (ops::mul (x1, x2), dim, false);
  Tensor n1 = ops::sqrt (ops::add_scalar (
    ops::sum (ops::square (x1), dim, false), eps * eps));
  Tensor n2 = ops::sqrt (ops::add_scalar (
    ops::sum (ops::square (x2), dim, false), eps * eps));
  Tensor denom = ops::clamp (ops::mul (n1, n2), eps, 1e30);
  return ops::div (dot, denom);
}

} /* namespace basalt::nn::functional */
