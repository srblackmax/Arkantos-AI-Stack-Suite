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
#include <basalt/nn.hpp>
#include <basalt/autograd.hpp>
#include <basalt/random.hpp>
#include <hyper_math/scientific/linalg.hpp>
#include <algorithm>
#include <cmath>

namespace basalt::nn
{

namespace init
{

namespace
{

std::pair<size_t, size_t>
compute_fan_in_and_out (const Tensor &tensor)
{
  BASALT_ASSERT_SHAPE (tensor.ndim () >= 2,
                       "fan_in/fan_out requires tensor of rank >= 2");
  size_t num_out = tensor.size (0);
  size_t num_in = tensor.size (1);
  size_t receptive = 1;
  for (size_t i = 2; i < tensor.ndim (); ++i)
    {
      receptive *= tensor.size (static_cast<ptrdiff_t> (i));
    }
  return {num_in * receptive, num_out * receptive};
}

} /* namespace */

Tensor &
zeros_ (Tensor &tensor)
{
  No_Grad_Guard no_grad;
  return tensor.fill_ (0.0);
}

Tensor &
ones_ (Tensor &tensor)
{
  No_Grad_Guard no_grad;
  return tensor.fill_ (1.0);
}

Tensor &
constant_ (Tensor &tensor, double val)
{
  No_Grad_Guard no_grad;
  return tensor.fill_ (val);
}

Tensor &
uniform_ (Tensor &tensor, double low, double high)
{
  No_Grad_Guard no_grad;
  size_t n = tensor.numel ();
  Random_Generator &gen = default_generator ();
  for (size_t i = 0; i < n; ++i)
    {
      tensor.write_scalar_at (i, gen.uniform (low, high));
    }
  tensor.bump_version ();
  return tensor;
}

Tensor &
normal_ (Tensor &tensor, double mean, double std_dev)
{
  No_Grad_Guard no_grad;
  size_t n = tensor.numel ();
  Random_Generator &gen = default_generator ();
  for (size_t i = 0; i < n; ++i)
    {
      tensor.write_scalar_at (i, gen.normal (mean, std_dev));
    }
  tensor.bump_version ();
  return tensor;
}

Tensor &
xavier_uniform_ (Tensor &tensor, double gain)
{
  auto [fan_in, fan_out] = compute_fan_in_and_out (tensor);
  double std_dev
    = gain * std::sqrt (2.0 / static_cast<double> (fan_in + fan_out));
  double bound = std::sqrt (3.0) * std_dev;
  return uniform_ (tensor, -bound, bound);
}

Tensor &
xavier_normal_ (Tensor &tensor, double gain)
{
  auto [fan_in, fan_out] = compute_fan_in_and_out (tensor);
  double std_dev
    = gain * std::sqrt (2.0 / static_cast<double> (fan_in + fan_out));
  return normal_ (tensor, 0.0, std_dev);
}

Tensor &
kaiming_uniform_ (Tensor &tensor, double a, bool fan_in_mode)
{
  auto [fan_in, fan_out] = compute_fan_in_and_out (tensor);
  size_t fan = fan_in_mode ? fan_in : fan_out;
  double gain = std::sqrt (2.0 / (1.0 + a * a));
  double std_dev = gain / std::sqrt (static_cast<double> (fan));
  double bound = std::sqrt (3.0) * std_dev;
  return uniform_ (tensor, -bound, bound);
}

Tensor &
kaiming_normal_ (Tensor &tensor, double a, bool fan_in_mode)
{
  auto [fan_in, fan_out] = compute_fan_in_and_out (tensor);
  size_t fan = fan_in_mode ? fan_in : fan_out;
  double gain = std::sqrt (2.0 / (1.0 + a * a));
  double std_dev = gain / std::sqrt (static_cast<double> (fan));
  return normal_ (tensor, 0.0, std_dev);
}

Tensor &
orthogonal_ (Tensor &tensor, double gain)
{
  No_Grad_Guard no_grad;
  BASALT_ASSERT_SHAPE (tensor.ndim () >= 2,
                       "orthogonal_() requires rank >= 2");
  size_t rows = tensor.size (0);
  size_t cols = tensor.numel () / rows;
  size_t m = std::max (rows, cols);
  size_t n = std::min (rows, cols);

  hyper_math::tensor::nd_tensor<double> rand_mat (
    hyper_math::tensor::shape_type{m, n});
  Random_Generator &gen = default_generator ();
  for (size_t r = 0; r < m; ++r)
    {
      for (size_t c = 0; c < n; ++c)
        {
          rand_mat (r, c) = gen.normal (0.0, 1.0);
        }
    }

  auto [q, r_mat] = hyper_math::scientific::linalg::qr (rand_mat);
  for (size_t j = 0; j < n; ++j)
    {
      double diag = r_mat (j, j);
      double s = (diag < 0.0) ? -1.0 : 1.0;
      for (size_t i = 0; i < m; ++i)
        {
          q (i, j) *= s * gain;
        }
    }

  for (size_t r = 0; r < rows; ++r)
    {
      for (size_t c = 0; c < cols; ++c)
        {
          double val = (rows >= cols) ? q (r, c) : q (c, r);
          tensor.write_scalar_at (r * cols + c, val);
        }
    }
  tensor.bump_version ();
  return tensor;
}

} /* namespace init */

/* Module implementation.  */

Module::~Module (void) noexcept = default;

Tensor
Module::forward (const Tensor & /*input*/)
{
  BASALT_THROW (Basalt_Error,
                "Module::forward() not implemented by subclass");
}

Tensor &
Module::register_parameter (const std::string &name, Tensor tensor,
                            bool requires_grad)
{
  BASALT_ASSERT_ARG (!name.empty (), "parameter name cannot be empty");
  if (tensor.defined () && requires_grad)
    {
      tensor.set_requires_grad (true);
    }
  for (auto &kv : parameters_)
    {
      if (kv.first == name)
        {
          kv.second = std::move (tensor);
          return kv.second;
        }
    }
  parameters_.emplace_back (name, std::move (tensor));
  return parameters_.back ().second;
}

Tensor &
Module::register_buffer (const std::string &name, Tensor tensor)
{
  BASALT_ASSERT_ARG (!name.empty (), "buffer name cannot be empty");
  for (auto &kv : buffers_)
    {
      if (kv.first == name)
        {
          kv.second = std::move (tensor);
          return kv.second;
        }
    }
  buffers_.emplace_back (name, std::move (tensor));
  return buffers_.back ().second;
}

void
Module::register_module_base (const std::string &name,
                              std::shared_ptr<Module> submodule)
{
  BASALT_ASSERT_ARG (!name.empty () && submodule != nullptr,
                     "register_module() requires non-empty name and module");
  for (auto &kv : submodules_)
    {
      if (kv.first == name)
        {
          kv.second = std::move (submodule);
          return;
        }
    }
  submodules_.emplace_back (name, std::move (submodule));
}

std::vector<Tensor>
Module::parameters (bool recurse) const
{
  auto named = named_parameters (recurse);
  std::vector<Tensor> out;
  out.reserve (named.size ());
  for (auto &kv : named)
    {
      out.push_back (kv.second);
    }
  return out;
}

std::vector<std::pair<std::string, Tensor>>
Module::named_parameters (bool recurse, const std::string &prefix) const
{
  std::vector<std::pair<std::string, Tensor>> out;
  std::string pfx = prefix.empty () ? "" : (prefix + ".");
  for (const auto &kv : parameters_)
    {
      if (kv.second.defined ())
        {
          out.emplace_back (pfx + kv.first, kv.second);
        }
    }
  if (recurse)
    {
      for (const auto &sub : submodules_)
        {
          auto sub_params = sub.second->named_parameters (
            true, pfx + sub.first);
          out.insert (out.end (), sub_params.begin (), sub_params.end ());
        }
    }
  return out;
}

std::vector<Tensor>
Module::buffers (bool recurse) const
{
  auto named = named_buffers (recurse);
  std::vector<Tensor> out;
  out.reserve (named.size ());
  for (auto &kv : named)
    {
      out.push_back (kv.second);
    }
  return out;
}

std::vector<std::pair<std::string, Tensor>>
Module::named_buffers (bool recurse, const std::string &prefix) const
{
  std::vector<std::pair<std::string, Tensor>> out;
  std::string pfx = prefix.empty () ? "" : (prefix + ".");
  for (const auto &kv : buffers_)
    {
      if (kv.second.defined ())
        {
          out.emplace_back (pfx + kv.first, kv.second);
        }
    }
  if (recurse)
    {
      for (const auto &sub : submodules_)
        {
          auto sub_bufs = sub.second->named_buffers (true, pfx + sub.first);
          out.insert (out.end (), sub_bufs.begin (), sub_bufs.end ());
        }
    }
  return out;
}

State_Dict
Module::state_dict (const std::string &prefix) const
{
  State_Dict sd;
  for (const auto &kv : named_parameters (true, prefix))
    {
      sd[kv.first] = kv.second.detach ();
    }
  for (const auto &kv : named_buffers (true, prefix))
    {
      sd[kv.first] = kv.second.detach ();
    }
  return sd;
}

void
Module::load_state_dict (const State_Dict &dict, bool strict)
{
  No_Grad_Guard no_grad;
  auto params = named_parameters (true);
  auto bufs = named_buffers (true);

  for (auto &kv : params)
    {
      auto it = dict.find (kv.first);
      if (it == dict.end ())
        {
          BASALT_ASSERT (!strict,
                         "missing parameter in state_dict: " + kv.first);
          continue;
        }
      BASALT_ASSERT_SHAPE (kv.second.shape () == it->second.shape (),
                           "shape mismatch loading parameter: " + kv.first);
      kv.second.copy_ (it->second);
    }
  for (auto &kv : bufs)
    {
      auto it = dict.find (kv.first);
      if (it == dict.end ())
        {
          BASALT_ASSERT (!strict,
                         "missing buffer in state_dict: " + kv.first);
          continue;
        }
      BASALT_ASSERT_SHAPE (kv.second.shape () == it->second.shape (),
                           "shape mismatch loading buffer: " + kv.first);
      kv.second.copy_ (it->second);
    }
}

void
Module::train (bool mode)
{
  training_ = mode;
  for (auto &sub : submodules_)
    {
      sub.second->train (mode);
    }
}

void
Module::to (Device device)
{
  No_Grad_Guard no_grad;
  for (auto &kv : parameters_)
    {
      if (kv.second.defined ())
        {
          bool rg = kv.second.requires_grad ();
          kv.second = kv.second.to (device).detach ();
          kv.second.set_requires_grad (rg);
        }
    }
  for (auto &kv : buffers_)
    {
      if (kv.second.defined ())
        {
          kv.second = kv.second.to (device).detach ();
        }
    }
  for (auto &sub : submodules_)
    {
      sub.second->to (device);
    }
}

void
Module::to (Dtype dtype)
{
  No_Grad_Guard no_grad;
  for (auto &kv : parameters_)
    {
      if (kv.second.defined () && is_floating_point (kv.second.dtype ()))
        {
          bool rg = kv.second.requires_grad ();
          kv.second = kv.second.to (dtype).detach ();
          kv.second.set_requires_grad (rg);
        }
    }
  for (auto &kv : buffers_)
    {
      if (kv.second.defined () && is_floating_point (kv.second.dtype ()))
        {
          kv.second = kv.second.to (dtype).detach ();
        }
    }
  for (auto &sub : submodules_)
    {
      sub.second->to (dtype);
    }
}

void
Module::to (Device device, Dtype dtype)
{
  to (device);
  to (dtype);
}

void
Module::zero_grad (void)
{
  for (auto &p : parameters (true))
    {
      p.zero_grad ();
    }
}

size_t
Module::num_parameters (bool only_trainable) const
{
  size_t total = 0;
  for (const auto &p : parameters (true))
    {
      if (!only_trainable || p.requires_grad ())
        {
          total += p.numel ();
        }
    }
  return total;
}

/* Sequential, Module_List, and Module_Dict.  */

Sequential::Sequential (std::initializer_list<std::shared_ptr<Module>> modules)
{
  for (const auto &m : modules)
    {
      add (m);
    }
}

Sequential::Sequential (std::vector<std::shared_ptr<Module>> modules)
{
  for (auto &m : modules)
    {
      add (std::move (m));
    }
}

void
Sequential::add (std::shared_ptr<Module> module)
{
  add (std::to_string (layers_.size ()), std::move (module));
}

void
Sequential::add (const std::string &name, std::shared_ptr<Module> module)
{
  register_module (name, module);
  layers_.push_back (std::move (module));
}

Tensor
Sequential::forward (const Tensor &input)
{
  Tensor cur = input;
  for (const auto &layer : layers_)
    {
      cur = layer->forward (cur);
    }
  return cur;
}

Module_List::Module_List (std::vector<std::shared_ptr<Module>> modules)
{
  for (auto &m : modules)
    {
      append (std::move (m));
    }
}

void
Module_List::append (std::shared_ptr<Module> module)
{
  register_module (std::to_string (modules_.size ()), module);
  modules_.push_back (std::move (module));
}

void
Module_Dict::insert (const std::string &key, std::shared_ptr<Module> module)
{
  register_module (key, module);
  dict_[key] = std::move (module);
}

std::shared_ptr<Module>
Module_Dict::at (const std::string &key) const
{
  auto it = dict_.find (key);
  BASALT_ASSERT_ARG (it != dict_.end (),
                     "key not found in Module_Dict: " + key);
  return it->second;
}

bool
Module_Dict::contains (const std::string &key) const
{
  return dict_.find (key) != dict_.end ();
}

/* Linear and Convolutional layers.  */

Linear::Linear (size_t in_features, size_t out_features, bool use_bias,
                Tensor_Options options)
  : in_features_ (in_features),
    out_features_ (out_features),
    use_bias_ (use_bias)
{
  weight = empty ({out_features, in_features},
                  options.with_requires_grad (true));
  init::kaiming_uniform_ (weight, std::sqrt (5.0));
  register_parameter ("weight", weight, true);

  if (use_bias_)
    {
      bias = empty ({out_features}, options.with_requires_grad (true));
      double bound = 1.0 / std::sqrt (static_cast<double> (in_features));
      init::uniform_ (bias, -bound, bound);
      register_parameter ("bias", bias, true);
    }
}

Tensor
Linear::forward (const Tensor &input)
{
  const Tensor &w = parameters_[0].second;
  const Tensor &b = use_bias_ ? parameters_[1].second : Tensor ();
  return functional::linear (input, w, b);
}

Conv1d::Conv1d (size_t in_channels, size_t out_channels, size_t kernel_size,
                size_t stride, size_t padding, size_t dilation, size_t groups,
                bool use_bias, Tensor_Options options)
  : stride_ (stride),
    padding_ (padding),
    dilation_ (dilation),
    groups_ (groups)
{
  BASALT_ASSERT_ARG (in_channels % groups == 0 && out_channels % groups == 0,
                     "Conv1d channels must be divisible by groups");
  weight = empty ({out_channels, in_channels / groups, kernel_size},
                  options.with_requires_grad (true));
  init::kaiming_uniform_ (weight, std::sqrt (5.0));
  register_parameter ("weight", weight, true);

  if (use_bias)
    {
      bias = empty ({out_channels}, options.with_requires_grad (true));
      double fan_in = static_cast<double> ((in_channels / groups)
                                           * kernel_size);
      double bound = 1.0 / std::sqrt (fan_in);
      init::uniform_ (bias, -bound, bound);
      register_parameter ("bias", bias, true);
    }
}

Tensor
Conv1d::forward (const Tensor &input)
{
  const Tensor &w = parameters_[0].second;
  const Tensor &b = (parameters_.size () > 1)
                    ? parameters_[1].second
                    : Tensor ();
  return functional::conv1d (input, w, b, stride_, padding_,
                             dilation_, groups_);
}

Conv2d::Conv2d (size_t in_channels, size_t out_channels, size_t kernel_size,
                size_t stride, size_t padding, size_t dilation, size_t groups,
                bool use_bias, Tensor_Options options)
  : stride_ (stride),
    padding_ (padding),
    dilation_ (dilation),
    groups_ (groups)
{
  BASALT_ASSERT_ARG (in_channels % groups == 0 && out_channels % groups == 0,
                     "Conv2d channels must be divisible by groups");
  weight = empty (
    {out_channels, in_channels / groups, kernel_size, kernel_size},
    options.with_requires_grad (true));
  init::kaiming_uniform_ (weight, std::sqrt (5.0));
  register_parameter ("weight", weight, true);

  if (use_bias)
    {
      bias = empty ({out_channels}, options.with_requires_grad (true));
      double fan_in = static_cast<double> (
        (in_channels / groups) * kernel_size * kernel_size);
      double bound = 1.0 / std::sqrt (fan_in);
      init::uniform_ (bias, -bound, bound);
      register_parameter ("bias", bias, true);
    }
}

Tensor
Conv2d::forward (const Tensor &input)
{
  const Tensor &w = parameters_[0].second;
  const Tensor &b = (parameters_.size () > 1)
                    ? parameters_[1].second
                    : Tensor ();
  return functional::conv2d (input, w, b, stride_, stride_,
                             padding_, padding_,
                             dilation_, dilation_, groups_);
}

Conv3d::Conv3d (size_t in_channels, size_t out_channels, size_t kernel_size,
                size_t stride, size_t padding, size_t dilation, size_t groups,
                bool use_bias, Tensor_Options options)
  : stride_ (stride),
    padding_ (padding),
    dilation_ (dilation),
    groups_ (groups)
{
  BASALT_ASSERT_ARG (in_channels % groups == 0 && out_channels % groups == 0,
                     "Conv3d channels must be divisible by groups");
  weight = empty ({out_channels, in_channels / groups,
                   kernel_size, kernel_size, kernel_size},
                  options.with_requires_grad (true));
  init::kaiming_uniform_ (weight, std::sqrt (5.0));
  register_parameter ("weight", weight, true);

  if (use_bias)
    {
      bias = empty ({out_channels}, options.with_requires_grad (true));
      double fan_in = static_cast<double> (
        (in_channels / groups) * kernel_size * kernel_size * kernel_size);
      double bound = 1.0 / std::sqrt (fan_in);
      init::uniform_ (bias, -bound, bound);
      register_parameter ("bias", bias, true);
    }
}

Tensor
Conv3d::forward (const Tensor &input)
{
  const Tensor &w = parameters_[0].second;
  const Tensor &b = (parameters_.size () > 1)
                    ? parameters_[1].second
                    : Tensor ();
  return functional::conv3d (input, w, b,
                             stride_, stride_, stride_,
                             padding_, padding_, padding_,
                             dilation_, dilation_, dilation_,
                             groups_);
}

/* Pooling layers.  */

Max_Pool1d::Max_Pool1d (size_t kernel_size, size_t stride, size_t padding)
  : kernel_size_ (kernel_size),
    stride_ (stride),
    padding_ (padding)
{
}

Tensor
Max_Pool1d::forward (const Tensor &input)
{
  return functional::max_pool1d (input, kernel_size_, stride_, padding_);
}

Max_Pool2d::Max_Pool2d (size_t kernel_size, size_t stride, size_t padding)
  : kernel_size_ (kernel_size),
    stride_ (stride),
    padding_ (padding)
{
}

Tensor
Max_Pool2d::forward (const Tensor &input)
{
  return functional::max_pool2d (input, kernel_size_, kernel_size_,
                                 stride_, stride_, padding_, padding_);
}

Max_Pool3d::Max_Pool3d (size_t kernel_size, size_t stride, size_t padding)
  : kernel_size_ (kernel_size),
    stride_ (stride),
    padding_ (padding)
{
}

Tensor
Max_Pool3d::forward (const Tensor &input)
{
  return functional::max_pool3d (input,
                                 kernel_size_, kernel_size_, kernel_size_,
                                 stride_, stride_, stride_,
                                 padding_, padding_, padding_);
}

Avg_Pool1d::Avg_Pool1d (size_t kernel_size, size_t stride, size_t padding)
  : kernel_size_ (kernel_size),
    stride_ (stride),
    padding_ (padding)
{
}

Tensor
Avg_Pool1d::forward (const Tensor &input)
{
  return functional::avg_pool1d (input, kernel_size_, stride_, padding_);
}

Avg_Pool2d::Avg_Pool2d (size_t kernel_size, size_t stride, size_t padding)
  : kernel_size_ (kernel_size),
    stride_ (stride),
    padding_ (padding)
{
}

Tensor
Avg_Pool2d::forward (const Tensor &input)
{
  return functional::avg_pool2d (input, kernel_size_, kernel_size_,
                                 stride_, stride_, padding_, padding_);
}

Avg_Pool3d::Avg_Pool3d (size_t kernel_size, size_t stride, size_t padding)
  : kernel_size_ (kernel_size),
    stride_ (stride),
    padding_ (padding)
{
}

Tensor
Avg_Pool3d::forward (const Tensor &input)
{
  return functional::avg_pool3d (input,
                                 kernel_size_, kernel_size_, kernel_size_,
                                 stride_, stride_, stride_,
                                 padding_, padding_, padding_);
}

Adaptive_Avg_Pool2d::Adaptive_Avg_Pool2d (size_t output_h, size_t output_w)
  : output_h_ (output_h),
    output_w_ (output_w)
{
}

Adaptive_Avg_Pool2d::Adaptive_Avg_Pool2d (size_t output_size)
  : output_h_ (output_size),
    output_w_ (output_size)
{
}

Tensor
Adaptive_Avg_Pool2d::forward (const Tensor &input)
{
  return functional::adaptive_avg_pool2d (input, output_h_, output_w_);
}

Flatten::Flatten (ptrdiff_t start_dim, ptrdiff_t end_dim)
  : start_dim_ (start_dim),
    end_dim_ (end_dim)
{
}

Tensor
Flatten::forward (const Tensor &input)
{
  return input.flatten (start_dim_, end_dim_);
}

/* Normalization layers.  */

Batch_Norm::Batch_Norm (size_t num_features, double eps, double momentum,
                        bool affine, Tensor_Options options)
  : eps_ (eps),
    momentum_ (momentum)
{
  running_mean = zeros ({num_features}, options.with_requires_grad (false));
  running_var = ones ({num_features}, options.with_requires_grad (false));
  register_buffer ("running_mean", running_mean);
  register_buffer ("running_var", running_var);

  if (affine)
    {
      weight = ones ({num_features}, options.with_requires_grad (true));
      bias = zeros ({num_features}, options.with_requires_grad (true));
      register_parameter ("weight", weight, true);
      register_parameter ("bias", bias, true);
    }
}

Tensor
Batch_Norm::forward (const Tensor &input)
{
  Tensor &rm = buffers_[0].second;
  Tensor &rv = buffers_[1].second;
  const Tensor &w = !parameters_.empty () ? parameters_[0].second : Tensor ();
  const Tensor &b = (parameters_.size () > 1)
                    ? parameters_[1].second
                    : Tensor ();
  return functional::batch_norm (input, rm, rv, w, b,
                                 training_, momentum_, eps_);
}

Layer_Norm::Layer_Norm (Shape_Vector normalized_shape, double eps,
                        bool elementwise_affine, Tensor_Options options)
  : normalized_shape_ (std::move (normalized_shape)),
    eps_ (eps)
{
  if (elementwise_affine)
    {
      weight = ones (normalized_shape_, options.with_requires_grad (true));
      bias = zeros (normalized_shape_, options.with_requires_grad (true));
      register_parameter ("weight", weight, true);
      register_parameter ("bias", bias, true);
    }
}

Layer_Norm::Layer_Norm (size_t normalized_dim, double eps,
                        bool elementwise_affine, Tensor_Options options)
  : Layer_Norm (Shape_Vector{normalized_dim}, eps, elementwise_affine,
                options)
{
}

Tensor
Layer_Norm::forward (const Tensor &input)
{
  const Tensor &w = !parameters_.empty () ? parameters_[0].second : Tensor ();
  const Tensor &b = (parameters_.size () > 1)
                    ? parameters_[1].second
                    : Tensor ();
  return functional::layer_norm (input, normalized_shape_, w, b, eps_);
}

Group_Norm::Group_Norm (size_t num_groups, size_t num_channels, double eps,
                        bool affine, Tensor_Options options)
  : num_groups_ (num_groups),
    num_channels_ (num_channels),
    eps_ (eps)
{
  BASALT_ASSERT_ARG (num_channels % num_groups == 0,
                     "Group_Norm channels must be divisible by num_groups");
  if (affine)
    {
      weight = ones ({num_channels}, options.with_requires_grad (true));
      bias = zeros ({num_channels}, options.with_requires_grad (true));
      register_parameter ("weight", weight, true);
      register_parameter ("bias", bias, true);
    }
}

Tensor
Group_Norm::forward (const Tensor &input)
{
  const Tensor &w = !parameters_.empty () ? parameters_[0].second : Tensor ();
  const Tensor &b = (parameters_.size () > 1)
                    ? parameters_[1].second
                    : Tensor ();
  return functional::group_norm (input, num_groups_, w, b, eps_);
}

RMS_Norm::RMS_Norm (Shape_Vector normalized_shape, double eps,
                    bool elementwise_affine, Tensor_Options options)
  : normalized_shape_ (std::move (normalized_shape)),
    eps_ (eps)
{
  if (elementwise_affine)
    {
      weight = ones (normalized_shape_, options.with_requires_grad (true));
      register_parameter ("weight", weight, true);
    }
}

RMS_Norm::RMS_Norm (size_t normalized_dim, double eps,
                    bool elementwise_affine, Tensor_Options options)
  : RMS_Norm (Shape_Vector{normalized_dim}, eps, elementwise_affine, options)
{
}

Tensor
RMS_Norm::forward (const Tensor &input)
{
  const Tensor &w = !parameters_.empty () ? parameters_[0].second : Tensor ();
  return functional::rms_norm (input, normalized_shape_, w, eps_);
}

/* Activation modules.  */

Tensor
ReLU::forward (const Tensor &input)
{
  return ops::relu (input);
}

Leaky_ReLU::Leaky_ReLU (double negative_slope)
  : negative_slope_ (negative_slope)
{
}

Tensor
Leaky_ReLU::forward (const Tensor &input)
{
  return ops::leaky_relu (input, negative_slope_);
}

ELU::ELU (double alpha)
  : alpha_ (alpha)
{
}

Tensor
ELU::forward (const Tensor &input)
{
  return ops::elu (input, alpha_);
}

Tensor
GELU::forward (const Tensor &input)
{
  return ops::gelu (input);
}

Tensor
SiLU::forward (const Tensor &input)
{
  return ops::silu (input);
}

Tensor
Mish::forward (const Tensor &input)
{
  return ops::mish (input);
}

Tensor
Sigmoid::forward (const Tensor &input)
{
  return ops::sigmoid (input);
}

Tensor
Tanh::forward (const Tensor &input)
{
  return ops::tanh (input);
}

Softmax::Softmax (ptrdiff_t dim)
  : dim_ (dim)
{
}

Tensor
Softmax::forward (const Tensor &input)
{
  return ops::softmax (input, dim_);
}

Log_Softmax::Log_Softmax (ptrdiff_t dim)
  : dim_ (dim)
{
}

Tensor
Log_Softmax::forward (const Tensor &input)
{
  return ops::log_softmax (input, dim_);
}

/* Embedding and Dropout.  */

Embedding::Embedding (size_t num_embeddings, size_t embedding_dim,
                      ptrdiff_t padding_idx, Tensor_Options options)
  : num_embeddings_ (num_embeddings),
    embedding_dim_ (embedding_dim),
    padding_idx_ (padding_idx)
{
  weight = empty ({num_embeddings_, embedding_dim_},
                  options.with_requires_grad (true));
  init::normal_ (weight, 0.0, 1.0);
  if (padding_idx_ >= 0
      && static_cast<size_t> (padding_idx_) < num_embeddings_)
    {
      No_Grad_Guard no_grad;
      weight.select (0, padding_idx_).fill_ (0.0);
    }
  register_parameter ("weight", weight, true);
}

Tensor
Embedding::forward (const Tensor &indices)
{
  return functional::embedding (indices, parameters_[0].second, padding_idx_);
}

Dropout::Dropout (double p)
  : p_ (p)
{
}

Tensor
Dropout::forward (const Tensor &input)
{
  return functional::dropout (input, p_, training_);
}

/* Multihead_Attention and Transformer blocks.  */

Multihead_Attention::Multihead_Attention (size_t embed_dim, size_t num_heads,
                                          double dropout_p, bool use_bias,
                                          Tensor_Options options)
  : embed_dim_ (embed_dim),
    num_heads_ (num_heads),
    head_dim_ (embed_dim / std::max<size_t> (1, num_heads)),
    dropout_p_ (dropout_p)
{
  BASALT_ASSERT_ARG (embed_dim % num_heads == 0,
                     "Multihead_Attention embed_dim must be divisible by "
                     "num_heads");
  q_proj = register_module (
    "q_proj", std::make_shared<Linear> (embed_dim, embed_dim, use_bias,
                                        options));
  k_proj = register_module (
    "k_proj", std::make_shared<Linear> (embed_dim, embed_dim, use_bias,
                                        options));
  v_proj = register_module (
    "v_proj", std::make_shared<Linear> (embed_dim, embed_dim, use_bias,
                                        options));
  out_proj = register_module (
    "out_proj", std::make_shared<Linear> (embed_dim, embed_dim, use_bias,
                                          options));
}

Tensor
Multihead_Attention::forward (const Tensor &input)
{
  return forward (input, input, input, Tensor (), false);
}

Tensor
Multihead_Attention::forward (const Tensor &query, const Tensor &key,
                              const Tensor &value, const Tensor &attn_mask,
                              bool is_causal)
{
  BASALT_ASSERT_SHAPE (query.ndim () == 3 && key.ndim () == 3
                       && value.ndim () == 3,
                       "Multihead_Attention expects (batch, seq, dim) inputs");
  size_t batch = query.size (0);
  size_t q_len = query.size (1);
  size_t k_len = key.size (1);

  Tensor q = q_proj->forward (query)
               .reshape ({batch, q_len, num_heads_, head_dim_})
               .permute ({0, 2, 1, 3});
  Tensor k = k_proj->forward (key)
               .reshape ({batch, k_len, num_heads_, head_dim_})
               .permute ({0, 2, 1, 3});
  Tensor v = v_proj->forward (value)
               .reshape ({batch, k_len, num_heads_, head_dim_})
               .permute ({0, 2, 1, 3});

  Tensor attn_out = functional::scaled_dot_product_attention (
    q, k, v, attn_mask, dropout_p_, is_causal, training_);

  Tensor merged = attn_out.permute ({0, 2, 1, 3})
                    .reshape ({batch, q_len, embed_dim_});
  return out_proj->forward (merged);
}

Transformer_Encoder_Layer::Transformer_Encoder_Layer (
  size_t d_model, size_t nhead, size_t dim_feedforward,
  double dropout_p, bool norm_first, Tensor_Options options)
  : norm_first_ (norm_first)
{
  self_attn = register_module (
    "self_attn",
    std::make_shared<Multihead_Attention> (d_model, nhead, dropout_p, true,
                                           options));
  linear1 = register_module (
    "linear1",
    std::make_shared<Linear> (d_model, dim_feedforward, true, options));
  linear2 = register_module (
    "linear2",
    std::make_shared<Linear> (dim_feedforward, d_model, true, options));
  norm1 = register_module (
    "norm1", std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
  norm2 = register_module (
    "norm2", std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
  dropout = register_module (
    "dropout", std::make_shared<Dropout> (dropout_p));
}

Tensor
Transformer_Encoder_Layer::forward (const Tensor &src)
{
  return forward (src, Tensor (), false);
}

Tensor
Transformer_Encoder_Layer::forward (const Tensor &src,
                                    const Tensor &src_mask,
                                    bool is_causal)
{
  Tensor x = src;
  if (norm_first_)
    {
      Tensor n1 = norm1->forward (x);
      Tensor attn = self_attn->forward (n1, n1, n1, src_mask, is_causal);
      x = ops::add (x, dropout->forward (attn));
      Tensor n2 = norm2->forward (x);
      Tensor ff = linear2->forward (
        dropout->forward (ops::gelu (linear1->forward (n2))));
      x = ops::add (x, dropout->forward (ff));
    }
  else
    {
      Tensor attn = self_attn->forward (x, x, x, src_mask, is_causal);
      x = norm1->forward (ops::add (x, dropout->forward (attn)));
      Tensor ff = linear2->forward (
        dropout->forward (ops::gelu (linear1->forward (x))));
      x = norm2->forward (ops::add (x, dropout->forward (ff)));
    }
  return x;
}

Transformer_Decoder_Layer::Transformer_Decoder_Layer (
  size_t d_model, size_t nhead, size_t dim_feedforward,
  double dropout_p, bool norm_first, Tensor_Options options)
  : norm_first_ (norm_first)
{
  self_attn = register_module (
    "self_attn",
    std::make_shared<Multihead_Attention> (d_model, nhead, dropout_p, true,
                                           options));
  multihead_attn = register_module (
    "multihead_attn",
    std::make_shared<Multihead_Attention> (d_model, nhead, dropout_p, true,
                                           options));
  linear1 = register_module (
    "linear1",
    std::make_shared<Linear> (d_model, dim_feedforward, true, options));
  linear2 = register_module (
    "linear2",
    std::make_shared<Linear> (dim_feedforward, d_model, true, options));
  norm1 = register_module (
    "norm1", std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
  norm2 = register_module (
    "norm2", std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
  norm3 = register_module (
    "norm3", std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
  dropout = register_module (
    "dropout", std::make_shared<Dropout> (dropout_p));
}

Tensor
Transformer_Decoder_Layer::forward (const Tensor &tgt)
{
  return forward (tgt, tgt, Tensor (), Tensor (), true);
}

Tensor
Transformer_Decoder_Layer::forward (const Tensor &tgt,
                                    const Tensor &memory,
                                    const Tensor &tgt_mask,
                                    const Tensor &memory_mask,
                                    bool tgt_is_causal)
{
  Tensor x = tgt;
  if (norm_first_)
    {
      Tensor n1 = norm1->forward (x);
      Tensor sa = self_attn->forward (n1, n1, n1, tgt_mask, tgt_is_causal);
      x = ops::add (x, dropout->forward (sa));

      Tensor n2 = norm2->forward (x);
      Tensor ca = multihead_attn->forward (
        n2, memory, memory, memory_mask, false);
      x = ops::add (x, dropout->forward (ca));

      Tensor n3 = norm3->forward (x);
      Tensor ff = linear2->forward (
        dropout->forward (ops::gelu (linear1->forward (n3))));
      x = ops::add (x, dropout->forward (ff));
    }
  else
    {
      Tensor sa = self_attn->forward (x, x, x, tgt_mask, tgt_is_causal);
      x = norm1->forward (ops::add (x, dropout->forward (sa)));

      Tensor ca = multihead_attn->forward (
        x, memory, memory, memory_mask, false);
      x = norm2->forward (ops::add (x, dropout->forward (ca)));

      Tensor ff = linear2->forward (
        dropout->forward (ops::gelu (linear1->forward (x))));
      x = norm3->forward (ops::add (x, dropout->forward (ff)));
    }
  return x;
}

Transformer_Encoder::Transformer_Encoder (
  size_t d_model, size_t nhead, size_t num_layers,
  size_t dim_feedforward, double dropout_p, bool norm_first,
  Tensor_Options options)
{
  for (size_t i = 0; i < num_layers; ++i)
    {
      auto layer = std::make_shared<Transformer_Encoder_Layer> (
        d_model, nhead, dim_feedforward, dropout_p, norm_first, options);
      register_module ("layer_" + std::to_string (i), layer);
      layers_.push_back (std::move (layer));
    }
  final_norm_ = register_module (
    "final_norm",
    std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
}

Tensor
Transformer_Encoder::forward (const Tensor &src)
{
  return forward (src, Tensor (), false);
}

Tensor
Transformer_Encoder::forward (const Tensor &src, const Tensor &mask,
                              bool is_causal)
{
  Tensor cur = src;
  for (const auto &layer : layers_)
    {
      cur = layer->forward (cur, mask, is_causal);
    }
  return final_norm_->forward (cur);
}

Transformer_Decoder::Transformer_Decoder (
  size_t d_model, size_t nhead, size_t num_layers,
  size_t dim_feedforward, double dropout_p, bool norm_first,
  Tensor_Options options)
{
  for (size_t i = 0; i < num_layers; ++i)
    {
      auto layer = std::make_shared<Transformer_Decoder_Layer> (
        d_model, nhead, dim_feedforward, dropout_p, norm_first, options);
      register_module ("layer_" + std::to_string (i), layer);
      layers_.push_back (std::move (layer));
    }
  final_norm_ = register_module (
    "final_norm",
    std::make_shared<Layer_Norm> (d_model, 1e-5, true, options));
}

Tensor
Transformer_Decoder::forward (const Tensor &tgt)
{
  return forward (tgt, tgt, Tensor (), Tensor (), true);
}

Tensor
Transformer_Decoder::forward (const Tensor &tgt, const Tensor &memory,
                              const Tensor &tgt_mask,
                              const Tensor &memory_mask,
                              bool tgt_is_causal)
{
  Tensor cur = tgt;
  for (const auto &layer : layers_)
    {
      cur = layer->forward (cur, memory, tgt_mask, memory_mask,
                            tgt_is_causal);
    }
  return final_norm_->forward (cur);
}

/* Recurrent sequence models: RNN, LSTM, and GRU.  */

RNN::RNN (size_t input_size, size_t hidden_size, size_t num_layers,
          bool batch_first, double dropout_p, Tensor_Options options)
  : input_size_ (input_size),
    hidden_size_ (hidden_size),
    num_layers_ (std::max<size_t> (1, num_layers)),
    batch_first_ (batch_first),
    dropout_p_ (dropout_p)
{
  for (size_t l = 0; l < num_layers_; ++l)
    {
      size_t in_dim = (l == 0) ? input_size_ : hidden_size_;
      auto ih = register_module (
        "w_ih_l" + std::to_string (l),
        std::make_shared<Linear> (in_dim, hidden_size_, true, options));
      auto hh = register_module (
        "w_hh_l" + std::to_string (l),
        std::make_shared<Linear> (hidden_size_, hidden_size_, true, options));
      w_ih_.push_back (std::move (ih));
      w_hh_.push_back (std::move (hh));
    }
}

Tensor
RNN::forward (const Tensor &input)
{
  return forward_with_state (input, Tensor ()).first;
}

std::pair<Tensor, Tensor>
RNN::forward_with_state (const Tensor &input, const Tensor &h0)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 3,
                       "RNN expects 3-D (batch, seq, feat) or "
                       "(seq, batch, feat) input");
  Tensor x = batch_first_ ? input : input.transpose (0, 1);
  size_t batch = x.size (0);
  size_t seq_len = x.size (1);

  std::vector<Tensor> h_last;
  h_last.reserve (num_layers_);

  for (size_t l = 0; l < num_layers_; ++l)
    {
      Tensor h_t = (h0.defined () && h0.ndim () == 3)
                   ? h0.select (0, static_cast<ptrdiff_t> (l))
                   : zeros ({batch, hidden_size_},
                            x.options ().with_requires_grad (false));
      Tensor x_proj = w_ih_[l]->forward (x);
      std::vector<Tensor> step_outs;
      step_outs.reserve (seq_len);
      for (size_t t = 0; t < seq_len; ++t)
        {
          Tensor xt = x_proj.select (1, static_cast<ptrdiff_t> (t));
          Tensor ht_proj = w_hh_[l]->forward (h_t);
          h_t = ops::tanh (ops::add (xt, ht_proj));
          step_outs.push_back (h_t.unsqueeze (1));
        }
      x = ops::cat (step_outs, 1);
      if (l + 1 < num_layers_ && dropout_p_ > 0.0 && training_)
        {
          x = functional::dropout (x, dropout_p_, training_);
        }
      h_last.push_back (h_t.unsqueeze (0));
    }

  Tensor h_n = ops::cat (h_last, 0);
  Tensor out = batch_first_ ? x : x.transpose (0, 1);
  return {out, h_n};
}

LSTM::LSTM (size_t input_size, size_t hidden_size, size_t num_layers,
            bool batch_first, double dropout_p, Tensor_Options options)
  : input_size_ (input_size),
    hidden_size_ (hidden_size),
    num_layers_ (std::max<size_t> (1, num_layers)),
    batch_first_ (batch_first),
    dropout_p_ (dropout_p)
{
  for (size_t l = 0; l < num_layers_; ++l)
    {
      size_t in_dim = (l == 0) ? input_size_ : hidden_size_;
      auto ih = register_module (
        "w_ih_l" + std::to_string (l),
        std::make_shared<Linear> (in_dim, 4 * hidden_size_, true, options));
      auto hh = register_module (
        "w_hh_l" + std::to_string (l),
        std::make_shared<Linear> (hidden_size_, 4 * hidden_size_, true,
                                  options));
      w_ih_.push_back (std::move (ih));
      w_hh_.push_back (std::move (hh));
    }
}

Tensor
LSTM::forward (const Tensor &input)
{
  return std::get<0> (forward_with_state (input, Tensor (), Tensor ()));
}

std::tuple<Tensor, Tensor, Tensor>
LSTM::forward_with_state (const Tensor &input,
                          const Tensor &h0,
                          const Tensor &c0)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 3, "LSTM expects 3-D input");
  Tensor x = batch_first_ ? input : input.transpose (0, 1);
  size_t batch = x.size (0);
  size_t seq_len = x.size (1);

  std::vector<Tensor> h_last;
  std::vector<Tensor> c_last;
  h_last.reserve (num_layers_);
  c_last.reserve (num_layers_);

  for (size_t l = 0; l < num_layers_; ++l)
    {
      Tensor h_t = (h0.defined () && h0.ndim () == 3)
                   ? h0.select (0, static_cast<ptrdiff_t> (l))
                   : zeros ({batch, hidden_size_},
                            x.options ().with_requires_grad (false));
      Tensor c_t = (c0.defined () && c0.ndim () == 3)
                   ? c0.select (0, static_cast<ptrdiff_t> (l))
                   : zeros ({batch, hidden_size_},
                            x.options ().with_requires_grad (false));
      Tensor x_proj = w_ih_[l]->forward (x);
      std::vector<Tensor> step_outs;
      step_outs.reserve (seq_len);
      for (size_t t = 0; t < seq_len; ++t)
        {
          Tensor xt = x_proj.select (1, static_cast<ptrdiff_t> (t));
          Tensor gates = ops::add (xt, w_hh_[l]->forward (h_t));
          Tensor i_gate = ops::sigmoid (
            gates.narrow (1, 0, hidden_size_));
          Tensor f_gate = ops::sigmoid (
            gates.narrow (1, static_cast<ptrdiff_t> (hidden_size_),
                          hidden_size_));
          Tensor g_gate = ops::tanh (
            gates.narrow (1, static_cast<ptrdiff_t> (2 * hidden_size_),
                          hidden_size_));
          Tensor o_gate = ops::sigmoid (
            gates.narrow (1, static_cast<ptrdiff_t> (3 * hidden_size_),
                          hidden_size_));
          c_t = ops::add (ops::mul (f_gate, c_t), ops::mul (i_gate, g_gate));
          h_t = ops::mul (o_gate, ops::tanh (c_t));
          step_outs.push_back (h_t.unsqueeze (1));
        }
      x = ops::cat (step_outs, 1);
      if (l + 1 < num_layers_ && dropout_p_ > 0.0 && training_)
        {
          x = functional::dropout (x, dropout_p_, training_);
        }
      h_last.push_back (h_t.unsqueeze (0));
      c_last.push_back (c_t.unsqueeze (0));
    }

  Tensor h_n = ops::cat (h_last, 0);
  Tensor c_n = ops::cat (c_last, 0);
  Tensor out = batch_first_ ? x : x.transpose (0, 1);
  return {out, h_n, c_n};
}

GRU::GRU (size_t input_size, size_t hidden_size, size_t num_layers,
          bool batch_first, double dropout_p, Tensor_Options options)
  : input_size_ (input_size),
    hidden_size_ (hidden_size),
    num_layers_ (std::max<size_t> (1, num_layers)),
    batch_first_ (batch_first),
    dropout_p_ (dropout_p)
{
  for (size_t l = 0; l < num_layers_; ++l)
    {
      size_t in_dim = (l == 0) ? input_size_ : hidden_size_;
      auto ih = register_module (
        "w_ih_l" + std::to_string (l),
        std::make_shared<Linear> (in_dim, 3 * hidden_size_, true, options));
      auto hh = register_module (
        "w_hh_l" + std::to_string (l),
        std::make_shared<Linear> (hidden_size_, 3 * hidden_size_, true,
                                  options));
      w_ih_.push_back (std::move (ih));
      w_hh_.push_back (std::move (hh));
    }
}

Tensor
GRU::forward (const Tensor &input)
{
  return forward_with_state (input, Tensor ()).first;
}

std::pair<Tensor, Tensor>
GRU::forward_with_state (const Tensor &input, const Tensor &h0)
{
  BASALT_ASSERT_SHAPE (input.ndim () == 3, "GRU expects 3-D input");
  Tensor x = batch_first_ ? input : input.transpose (0, 1);
  size_t batch = x.size (0);
  size_t seq_len = x.size (1);

  std::vector<Tensor> h_last;
  h_last.reserve (num_layers_);

  for (size_t l = 0; l < num_layers_; ++l)
    {
      Tensor h_t = (h0.defined () && h0.ndim () == 3)
                   ? h0.select (0, static_cast<ptrdiff_t> (l))
                   : zeros ({batch, hidden_size_},
                            x.options ().with_requires_grad (false));
      Tensor x_proj = w_ih_[l]->forward (x);
      std::vector<Tensor> step_outs;
      step_outs.reserve (seq_len);
      for (size_t t = 0; t < seq_len; ++t)
        {
          Tensor xt = x_proj.select (1, static_cast<ptrdiff_t> (t));
          Tensor ht_proj = w_hh_[l]->forward (h_t);
          Tensor x_r = xt.narrow (1, 0, hidden_size_);
          Tensor x_z = xt.narrow (1, static_cast<ptrdiff_t> (hidden_size_),
                                  hidden_size_);
          Tensor x_n = xt.narrow (1, static_cast<ptrdiff_t> (2 * hidden_size_),
                                  hidden_size_);
          Tensor h_r = ht_proj.narrow (1, 0, hidden_size_);
          Tensor h_z = ht_proj.narrow (
            1, static_cast<ptrdiff_t> (hidden_size_), hidden_size_);
          Tensor h_n = ht_proj.narrow (
            1, static_cast<ptrdiff_t> (2 * hidden_size_), hidden_size_);

          Tensor r_gate = ops::sigmoid (ops::add (x_r, h_r));
          Tensor z_gate = ops::sigmoid (ops::add (x_z, h_z));
          Tensor n_gate = ops::tanh (ops::add (x_n, ops::mul (r_gate, h_n)));
          Tensor one_minus_z = ops::add_scalar (ops::neg (z_gate), 1.0);
          h_t = ops::add (ops::mul (one_minus_z, n_gate),
                          ops::mul (z_gate, h_t));
          step_outs.push_back (h_t.unsqueeze (1));
        }
      x = ops::cat (step_outs, 1);
      if (l + 1 < num_layers_ && dropout_p_ > 0.0 && training_)
        {
          x = functional::dropout (x, dropout_p_, training_);
        }
      h_last.push_back (h_t.unsqueeze (0));
    }

  Tensor h_n = ops::cat (h_last, 0);
  Tensor out = batch_first_ ? x : x.transpose (0, 1);
  return {out, h_n};
}

/* Rotary Position Embedding (RoPE) and KV_Cache.  */

Rotary_Embedding::Rotary_Embedding (size_t dim, size_t max_seq_len,
                                    double base, Tensor_Options options)
  : dim_ (dim),
    max_seq_len_ (max_seq_len),
    base_ (base)
{
  BASALT_ASSERT_ARG (dim_ > 0 && dim_ % 2 == 0,
                     "Rotary_Embedding requires positive even dimension");
  No_Grad_Guard ng;
  Tensor_Options cpu_opts
    = Tensor_Options ().with_dtype (Dtype::Float32).with_requires_grad (false);
  cos_cached = empty ({max_seq_len_, dim_}, cpu_opts);
  sin_cached = empty ({max_seq_len_, dim_}, cpu_opts);
  float *cp = cos_cached.data_ptr<float> ();
  float *sp = sin_cached.data_ptr<float> ();
  size_t half = dim_ / 2;
  for (size_t pos = 0; pos < max_seq_len_; ++pos)
    {
      for (size_t i = 0; i < half; ++i)
        {
          double exponent
            = static_cast<double> (2 * i) / static_cast<double> (dim_);
          double theta
            = static_cast<double> (pos) / std::pow (base_, exponent);
          float c = static_cast<float> (std::cos (theta));
          float s = static_cast<float> (std::sin (theta));
          cp[pos * dim_ + i] = c;
          cp[pos * dim_ + half + i] = c;
          sp[pos * dim_ + i] = s;
          sp[pos * dim_ + half + i] = s;
        }
    }
  if (options.device != Device::cpu ()
      || options.dtype != Dtype::Float32)
    {
      cos_cached = cos_cached.to (options.device, options.dtype);
      sin_cached = sin_cached.to (options.device, options.dtype);
    }
  register_buffer ("cos_cached", cos_cached);
  register_buffer ("sin_cached", sin_cached);
}

Tensor
Rotary_Embedding::forward (const Tensor &input)
{
  return functional::apply_rotary_pos_emb (
    input, buffers_[0].second, buffers_[1].second);
}

KV_Cache::KV_Cache (size_t batch_size, size_t num_heads, size_t max_seq_len,
                    size_t head_dim, Tensor_Options options)
  : k_cache (zeros ({batch_size, num_heads, max_seq_len, head_dim},
                    options.with_requires_grad (false))),
    v_cache (zeros ({batch_size, num_heads, max_seq_len, head_dim},
                    options.with_requires_grad (false))),
    max_seq_len_ (max_seq_len),
    seq_pos_ (0)
{
}

std::pair<Tensor, Tensor>
KV_Cache::update (const Tensor &k_step, const Tensor &v_step)
{
  BASALT_ASSERT_SHAPE (k_step.ndim () == 4 && v_step.ndim () == 4,
                       "KV_Cache::update expects 4-D (B, H, S, D) tensors");
  size_t step_len = k_step.size (2);
  BASALT_ASSERT_SHAPE (seq_pos_ + step_len <= max_seq_len_,
                       "KV_Cache::update exceeds max_seq_len capacity");
  No_Grad_Guard ng;
  k_cache.narrow (2, static_cast<ptrdiff_t> (seq_pos_), step_len)
    .copy_ (k_step);
  v_cache.narrow (2, static_cast<ptrdiff_t> (seq_pos_), step_len)
    .copy_ (v_step);
  seq_pos_ += step_len;
  return {k_cache.narrow (2, 0, seq_pos_), v_cache.narrow (2, 0, seq_pos_)};
}

void
KV_Cache::reset (void) noexcept
{
  seq_pos_ = 0;
}

size_t
KV_Cache::seq_len (void) const noexcept
{
  return seq_pos_;
}

Conv_Transpose2d::Conv_Transpose2d (size_t in_channels, size_t out_channels,
                                    size_t kernel_size,
                                    size_t stride, size_t padding,
                                    size_t output_padding,
                                    size_t dilation, size_t groups,
                                    bool use_bias,
                                    Tensor_Options options)
  : stride_h_ (stride),
    stride_w_ (stride),
    pad_h_ (padding),
    pad_w_ (padding),
    out_pad_h_ (output_padding),
    out_pad_w_ (output_padding),
    dilation_h_ (dilation),
    dilation_w_ (dilation),
    groups_ (groups),
    use_bias_ (use_bias)
{
  BASALT_ASSERT_ARG (in_channels > 0 && out_channels > 0 && kernel_size > 0
                     && groups > 0 && in_channels % groups == 0
                     && out_channels % groups == 0,
                     "Conv_Transpose2d invalid channel or group dimensions");
  size_t c_out_g = out_channels / groups;
  weight = register_parameter (
    "weight", empty ({in_channels, c_out_g, kernel_size, kernel_size},
                     options.with_requires_grad (true)));
  init::kaiming_uniform_ (weight, std::sqrt (5.0));

  if (use_bias_)
    {
      double fan_in
        = static_cast<double> (in_channels * kernel_size * kernel_size)
          / static_cast<double> (groups);
      double bound = 1.0 / std::sqrt (std::max (1.0, fan_in));
      bias = register_parameter (
        "bias", empty ({out_channels}, options.with_requires_grad (true)));
      init::uniform_ (bias, -bound, bound);
    }
}

Tensor
Conv_Transpose2d::forward (const Tensor &input)
{
  return functional::conv_transpose2d (
    input, weight, use_bias_ ? bias : Tensor (),
    stride_h_, stride_w_, pad_h_, pad_w_,
    out_pad_h_, out_pad_w_, dilation_h_, dilation_w_, groups_);
}

Instance_Norm::Instance_Norm (size_t num_features,
                              double eps,
                              bool affine,
                              Tensor_Options options)
  : num_features_ (num_features),
    eps_ (eps),
    affine_ (affine)
{
  BASALT_ASSERT_ARG (num_features > 0 && eps > 0.0,
                     "Instance_Norm requires positive num_features and eps");
  if (affine_)
    {
      weight = register_parameter (
        "weight", ones ({num_features}, options.with_requires_grad (true)));
      bias = register_parameter (
        "bias", zeros ({num_features}, options.with_requires_grad (true)));
    }
}

Tensor
Instance_Norm::forward (const Tensor &input)
{
  return functional::group_norm (
    input, num_features_,
    affine_ ? weight : Tensor (),
    affine_ ? bias : Tensor (),
    eps_);
}

} /* namespace basalt::nn */
