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

#include <surya/nn/vision_blocks.hpp>

#include <basalt/basalt.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace surya::inline v1::nn
{

namespace
{

basalt::Tensor
surya_to_basalt (const core::SuryaTensor &t)
{
  basalt::Tensor out (t.shape (),
                      basalt::Tensor_Options ().with_dtype (
                        basalt::Dtype::Float32));
  if (t.numel () > 0U && t.data () != nullptr)
    {
      std::memcpy (out.data_ptr<float> (), t.data (),
                   t.numel () * sizeof (float));
    }
  return out;
}

core::SuryaTensor
basalt_to_surya (const basalt::Tensor &t)
{
  const basalt::Tensor contig =
    t.to (basalt::Device::cpu (), basalt::Dtype::Float32).contiguous ();
  core::SuryaTensor out (contig.shape (), 0.0F);
  if (out.numel () > 0U && out.data () != nullptr)
    {
      std::memcpy (out.data (), contig.data_ptr<float> (),
                   out.numel () * sizeof (float));
    }
  return out;
}

basalt::Tensor
apply_activation (const basalt::Tensor &x, Activation_Kind act)
{
  switch (act)
    {
    case Activation_Kind::Identity:
      return x;
    case Activation_Kind::ReLU:
      return basalt::ops::relu (x);
    case Activation_Kind::SiLU:
      return basalt::ops::silu (x);
    case Activation_Kind::GELU:
      return basalt::ops::gelu (x);
    }
  return x;
}

class Basalt_Conv_BN_Act : public basalt::nn::Module
{
public:
  Basalt_Conv_BN_Act (size_t in_channels,
                      size_t out_channels,
                      size_t kernel_size,
                      size_t stride,
                      size_t padding,
                      Activation_Kind act,
                      double bn_eps)
    : act_ (act), bn_eps_ (bn_eps)
  {
    conv = register_module (
      "conv",
      std::make_shared<basalt::nn::Conv2d> (
        in_channels, out_channels, kernel_size, stride, padding, 1U, 1U,
        true));
    bn = register_module (
      "bn",
      std::make_shared<basalt::nn::Batch_Norm> (out_channels, bn_eps));
  }

  basalt::Tensor
  forward (const basalt::Tensor &input) override
  {
    basalt::Tensor y = conv->forward (input);
    if (!fused_)
      {
        y = bn->forward (y);
      }
    return apply_activation (y, act_);
  }

  void
  fuse_for_inference (void)
  {
    if (fused_)
      {
        return;
      }
    basalt::No_Grad_Guard no_grad;
    eval ();

    const size_t out_ch = conv->weight.size (0);
    const size_t fan_elems = conv->weight.numel () / out_ch;

    float *w_ptr = conv->weight.data_ptr<float> ();
    float *b_ptr = conv->bias.data_ptr<float> ();
    const float *gamma = bn->weight.data_ptr<float> ();
    const float *beta = bn->bias.data_ptr<float> ();
    const float *mean = bn->running_mean.data_ptr<float> ();
    const float *var = bn->running_var.data_ptr<float> ();

    for (size_t c = 0U; c < out_ch; ++c)
      {
        const float scale =
          gamma[c] / std::sqrt (var[c] + static_cast<float> (bn_eps_));
        float *w_c = w_ptr + c * fan_elems;
        for (size_t i = 0U; i < fan_elems; ++i)
          {
            w_c[i] *= scale;
          }
        b_ptr[c] = (b_ptr[c] - mean[c]) * scale + beta[c];
      }

    fused_ = true;
  }

  bool
  is_fused (void) const noexcept
  {
    return fused_;
  }

  std::shared_ptr<basalt::nn::Conv2d> conv;
  std::shared_ptr<basalt::nn::Batch_Norm> bn;

private:
  Activation_Kind act_;
  double bn_eps_;
  bool fused_ = false;
};

class Basalt_Residual_Bottleneck : public basalt::nn::Module
{
public:
  Basalt_Residual_Bottleneck (size_t in_channels,
                              size_t out_channels,
                              bool shortcut,
                              double expansion,
                              Activation_Kind act)
    : add_shortcut_ (shortcut && (in_channels == out_channels))
  {
    const size_t hidden = std::max<size_t> (
      1U,
      static_cast<size_t> (static_cast<double> (out_channels) * expansion));
    cv1 = register_module (
      "cv1",
      std::make_shared<Basalt_Conv_BN_Act> (
        in_channels, hidden, 1U, 1U, 0U, act, 1e-5));
    cv2 = register_module (
      "cv2",
      std::make_shared<Basalt_Conv_BN_Act> (
        hidden, out_channels, 3U, 1U, 1U, act, 1e-5));
  }

  basalt::Tensor
  forward (const basalt::Tensor &input) override
  {
    basalt::Tensor y = cv2->forward (cv1->forward (input));
    if (add_shortcut_)
      {
        return basalt::ops::add (input, y);
      }
    return y;
  }

  void
  fuse_for_inference (void)
  {
    cv1->fuse_for_inference ();
    cv2->fuse_for_inference ();
  }

  std::shared_ptr<Basalt_Conv_BN_Act> cv1;
  std::shared_ptr<Basalt_Conv_BN_Act> cv2;

private:
  bool add_shortcut_ = false;
};

class Basalt_CSP_C2f : public basalt::nn::Module
{
public:
  Basalt_CSP_C2f (size_t in_channels,
                  size_t out_channels,
                  size_t num_bottlenecks,
                  bool shortcut,
                  Activation_Kind act)
    : hidden_channels_ (std::max<size_t> (1U, out_channels / 2U))
  {
    cv1 = register_module (
      "cv1",
      std::make_shared<Basalt_Conv_BN_Act> (
        in_channels, 2U * hidden_channels_, 1U, 1U, 0U, act, 1e-5));

    const size_t n_blocks = std::max<size_t> (1U, num_bottlenecks);
    for (size_t i = 0U; i < n_blocks; ++i)
      {
        auto b = register_module (
          "m_" + std::to_string (i),
          std::make_shared<Basalt_Residual_Bottleneck> (
            hidden_channels_, hidden_channels_, shortcut, 1.0, act));
        bottlenecks.push_back (std::move (b));
      }

    const size_t concat_channels = (2U + n_blocks) * hidden_channels_;
    cv2 = register_module (
      "cv2",
      std::make_shared<Basalt_Conv_BN_Act> (
        concat_channels, out_channels, 1U, 1U, 0U, act, 1e-5));
  }

  basalt::Tensor
  forward (const basalt::Tensor &input) override
  {
    const basalt::Tensor y01 = cv1->forward (input);
    std::vector<basalt::Tensor> chunks = basalt::ops::chunk (y01, 2U, 1);
    basalt::Tensor cur = chunks.back ();
    for (const auto &b : bottlenecks)
      {
        cur = b->forward (cur);
        chunks.push_back (cur);
      }
    const basalt::Tensor cat_feat = basalt::ops::cat (chunks, 1);
    return cv2->forward (cat_feat);
  }

  void
  fuse_for_inference (void)
  {
    cv1->fuse_for_inference ();
    for (auto &b : bottlenecks)
      {
        b->fuse_for_inference ();
      }
    cv2->fuse_for_inference ();
  }

  std::shared_ptr<Basalt_Conv_BN_Act> cv1;
  std::shared_ptr<Basalt_Conv_BN_Act> cv2;
  std::vector<std::shared_ptr<Basalt_Residual_Bottleneck>> bottlenecks;

private:
  size_t hidden_channels_ = 0U;
};

class Basalt_SPPF : public basalt::nn::Module
{
public:
  Basalt_SPPF (size_t in_channels, size_t out_channels, size_t pool_kernel)
  {
    const size_t hidden = std::max<size_t> (1U, in_channels / 2U);
    cv1 = register_module (
      "cv1",
      std::make_shared<Basalt_Conv_BN_Act> (
        in_channels, hidden, 1U, 1U, 0U, Activation_Kind::SiLU, 1e-5));
    cv2 = register_module (
      "cv2",
      std::make_shared<Basalt_Conv_BN_Act> (
        hidden * 4U, out_channels, 1U, 1U, 0U, Activation_Kind::SiLU, 1e-5));
    maxpool = register_module (
      "maxpool",
      std::make_shared<basalt::nn::Max_Pool2d> (
        pool_kernel, 1U, pool_kernel / 2U));
  }

  basalt::Tensor
  forward (const basalt::Tensor &input) override
  {
    const basalt::Tensor x = cv1->forward (input);
    const basalt::Tensor y1 = maxpool->forward (x);
    const basalt::Tensor y2 = maxpool->forward (y1);
    const basalt::Tensor y3 = maxpool->forward (y2);
    const basalt::Tensor cat_feat = basalt::ops::cat ({x, y1, y2, y3}, 1);
    return cv2->forward (cat_feat);
  }

  void
  fuse_for_inference (void)
  {
    cv1->fuse_for_inference ();
    cv2->fuse_for_inference ();
  }

  std::shared_ptr<Basalt_Conv_BN_Act> cv1;
  std::shared_ptr<Basalt_Conv_BN_Act> cv2;
  std::shared_ptr<basalt::nn::Max_Pool2d> maxpool;
};

class Basalt_Vision_Classifier : public basalt::nn::Module
{
public:
  Basalt_Vision_Classifier (size_t in_channels,
                            size_t base_channels,
                            size_t num_classes)
  {
    const size_t c1 = std::max<size_t> (4U, base_channels);
    const size_t c2 = c1 * 2U;

    stem = register_module (
      "stem",
      std::make_shared<Basalt_Conv_BN_Act> (
        in_channels, c1, 3U, 2U, 1U, Activation_Kind::SiLU, 1e-5));
    stage1 = register_module (
      "stage1",
      std::make_shared<Basalt_CSP_C2f> (
        c1, c2, 1U, false, Activation_Kind::SiLU));
    sppf = register_module (
      "sppf",
      std::make_shared<Basalt_SPPF> (c2, c2, 5U));
    pool = register_module (
      "pool",
      std::make_shared<basalt::nn::Adaptive_Avg_Pool2d> (1U, 1U));
    flatten = register_module (
      "flatten",
      std::make_shared<basalt::nn::Flatten> (1, -1));
    head = register_module (
      "head",
      std::make_shared<basalt::nn::Linear> (c2, num_classes, true));
  }

  basalt::Tensor
  forward (const basalt::Tensor &input) override
  {
    basalt::Tensor x = stem->forward (input);
    x = stage1->forward (x);
    x = sppf->forward (x);
    x = pool->forward (x);
    x = flatten->forward (x);
    return head->forward (x);
  }

  void
  fuse_for_inference (void)
  {
    stem->fuse_for_inference ();
    stage1->fuse_for_inference ();
    sppf->fuse_for_inference ();
  }

  std::shared_ptr<Basalt_Conv_BN_Act> stem;
  std::shared_ptr<Basalt_CSP_C2f> stage1;
  std::shared_ptr<Basalt_SPPF> sppf;
  std::shared_ptr<basalt::nn::Adaptive_Avg_Pool2d> pool;
  std::shared_ptr<basalt::nn::Flatten> flatten;
  std::shared_ptr<basalt::nn::Linear> head;
};

} /* namespace */

/* 1. Conv_BN_Act wrapper.  */
struct Conv_BN_Act::Impl
{
  Basalt_Conv_BN_Act mod;
  Impl (size_t ic, size_t oc, size_t ks, size_t st, size_t pad,
        Activation_Kind act, double eps)
    : mod (ic, oc, ks, st, pad, act, eps)
  {
  }
};

Conv_BN_Act::Conv_BN_Act (size_t in_channels,
                          size_t out_channels,
                          size_t kernel_size,
                          size_t stride,
                          size_t padding,
                          Activation_Kind act,
                          double bn_eps)
  : impl_ (std::make_shared<Impl> (
      in_channels, out_channels, kernel_size, stride, padding, act, bn_eps))
{
}

Conv_BN_Act::~Conv_BN_Act (void) noexcept = default;

core::SuryaTensor
Conv_BN_Act::forward (const core::SuryaTensor &input)
{
  basalt::No_Grad_Guard ng;
  return basalt_to_surya (impl_->mod.forward (surya_to_basalt (input)));
}

void
Conv_BN_Act::eval (void) noexcept
{
  impl_->mod.eval ();
}

void
Conv_BN_Act::fuse_for_inference (void)
{
  impl_->mod.fuse_for_inference ();
}

bool
Conv_BN_Act::is_fused (void) const noexcept
{
  return impl_->mod.is_fused ();
}

void
Conv_BN_Act::randomize_bn_parameters (void)
{
  basalt::nn::init::uniform_ (impl_->mod.bn->running_mean, -0.5, 0.5);
  basalt::nn::init::uniform_ (impl_->mod.bn->running_var, 0.5, 2.0);
  basalt::nn::init::uniform_ (impl_->mod.bn->weight, 0.5, 1.5);
  basalt::nn::init::uniform_ (impl_->mod.bn->bias, -0.2, 0.2);
}

void *
Conv_BN_Act::native_module_ptr (void) noexcept
{
  return static_cast<basalt::nn::Module *> (&impl_->mod);
}

const void *
Conv_BN_Act::native_module_ptr (void) const noexcept
{
  return static_cast<const basalt::nn::Module *> (&impl_->mod);
}

/* 2. Residual_Bottleneck wrapper.  */
struct Residual_Bottleneck::Impl
{
  Basalt_Residual_Bottleneck mod;
  Impl (size_t ic, size_t oc, bool sc, double exp, Activation_Kind act)
    : mod (ic, oc, sc, exp, act)
  {
  }
};

Residual_Bottleneck::Residual_Bottleneck (size_t in_channels,
                                          size_t out_channels,
                                          bool shortcut,
                                          double expansion,
                                          Activation_Kind act)
  : impl_ (std::make_shared<Impl> (
      in_channels, out_channels, shortcut, expansion, act))
{
}

Residual_Bottleneck::~Residual_Bottleneck (void) noexcept = default;

core::SuryaTensor
Residual_Bottleneck::forward (const core::SuryaTensor &input)
{
  basalt::No_Grad_Guard ng;
  return basalt_to_surya (impl_->mod.forward (surya_to_basalt (input)));
}

void
Residual_Bottleneck::eval (void) noexcept
{
  impl_->mod.eval ();
}

void
Residual_Bottleneck::fuse_for_inference (void)
{
  impl_->mod.fuse_for_inference ();
}

void *
Residual_Bottleneck::native_module_ptr (void) noexcept
{
  return static_cast<basalt::nn::Module *> (&impl_->mod);
}

const void *
Residual_Bottleneck::native_module_ptr (void) const noexcept
{
  return static_cast<const basalt::nn::Module *> (&impl_->mod);
}

/* 3. CSP_C2f_Block wrapper.  */
struct CSP_C2f_Block::Impl
{
  Basalt_CSP_C2f mod;
  Impl (size_t ic, size_t oc, size_t nb, bool sc, Activation_Kind act)
    : mod (ic, oc, nb, sc, act)
  {
  }
};

CSP_C2f_Block::CSP_C2f_Block (size_t in_channels,
                              size_t out_channels,
                              size_t num_bottlenecks,
                              bool shortcut,
                              Activation_Kind act)
  : impl_ (std::make_shared<Impl> (
      in_channels, out_channels, num_bottlenecks, shortcut, act))
{
}

CSP_C2f_Block::~CSP_C2f_Block (void) noexcept = default;

core::SuryaTensor
CSP_C2f_Block::forward (const core::SuryaTensor &input)
{
  basalt::No_Grad_Guard ng;
  return basalt_to_surya (impl_->mod.forward (surya_to_basalt (input)));
}

void
CSP_C2f_Block::eval (void) noexcept
{
  impl_->mod.eval ();
}

void
CSP_C2f_Block::fuse_for_inference (void)
{
  impl_->mod.fuse_for_inference ();
}

void *
CSP_C2f_Block::native_module_ptr (void) noexcept
{
  return static_cast<basalt::nn::Module *> (&impl_->mod);
}

const void *
CSP_C2f_Block::native_module_ptr (void) const noexcept
{
  return static_cast<const basalt::nn::Module *> (&impl_->mod);
}

/* 4. SPPF_Block wrapper.  */
struct SPPF_Block::Impl
{
  Basalt_SPPF mod;
  Impl (size_t ic, size_t oc, size_t pk) : mod (ic, oc, pk) {}
};

SPPF_Block::SPPF_Block (size_t in_channels,
                        size_t out_channels,
                        size_t pool_kernel)
  : impl_ (std::make_shared<Impl> (in_channels, out_channels, pool_kernel))
{
}

SPPF_Block::~SPPF_Block (void) noexcept = default;

core::SuryaTensor
SPPF_Block::forward (const core::SuryaTensor &input)
{
  basalt::No_Grad_Guard ng;
  return basalt_to_surya (impl_->mod.forward (surya_to_basalt (input)));
}

void
SPPF_Block::eval (void) noexcept
{
  impl_->mod.eval ();
}

void
SPPF_Block::fuse_for_inference (void)
{
  impl_->mod.fuse_for_inference ();
}

void *
SPPF_Block::native_module_ptr (void) noexcept
{
  return static_cast<basalt::nn::Module *> (&impl_->mod);
}

const void *
SPPF_Block::native_module_ptr (void) const noexcept
{
  return static_cast<const basalt::nn::Module *> (&impl_->mod);
}

/* 5. Vision_Classifier wrapper.  */
struct Vision_Classifier::Impl
{
  Basalt_Vision_Classifier mod;
  Impl (size_t ic, size_t bc, size_t nc) : mod (ic, bc, nc) {}
};

Vision_Classifier::Vision_Classifier (size_t in_channels,
                                      size_t base_channels,
                                      size_t num_classes)
  : impl_ (std::make_shared<Impl> (in_channels, base_channels, num_classes))
{
}

Vision_Classifier::~Vision_Classifier (void) noexcept = default;

core::SuryaTensor
Vision_Classifier::forward (const core::SuryaTensor &input)
{
  basalt::No_Grad_Guard ng;
  return basalt_to_surya (impl_->mod.forward (surya_to_basalt (input)));
}

void
Vision_Classifier::eval (void) noexcept
{
  impl_->mod.eval ();
}

void
Vision_Classifier::fuse_for_inference (void)
{
  impl_->mod.fuse_for_inference ();
}

void
Vision_Classifier::zero_head_weights (void)
{
  basalt::nn::init::zeros_ (impl_->mod.head->weight);
}

void *
Vision_Classifier::native_module_ptr (void) noexcept
{
  return static_cast<basalt::nn::Module *> (&impl_->mod);
}

const void *
Vision_Classifier::native_module_ptr (void) const noexcept
{
  return static_cast<const basalt::nn::Module *> (&impl_->mod);
}

} /* namespace surya::inline v1::nn */
