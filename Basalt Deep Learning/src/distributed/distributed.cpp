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

#include <basalt/distributed.hpp>

#include <algorithm>
#include <hyper_math/core/parallel.hpp>
#include <basalt/autograd.hpp>
#include <basalt/error.hpp>
#include <basalt/ops.hpp>

namespace basalt::distributed
{

std::vector<Tensor>
scatter (const Tensor &batch, const std::vector<Device> &devices)
{
  BASALT_ASSERT_ARG (!devices.empty (),
                     "scatter requires at least one target device");
  BASALT_ASSERT_SHAPE (batch.defined () && batch.ndim () >= 1,
                       "scatter requires a defined tensor with ndim >= 1");

  std::vector<Tensor> chunks = ops::chunk (batch, devices.size (), 0);
  std::vector<Tensor> out;
  out.reserve (chunks.size ());
  for (size_t i = 0; i < chunks.size (); ++i)
    {
      Tensor moved = chunks[i].contiguous ().to (devices[i]).detach ();
      moved.set_requires_grad (batch.requires_grad ());
      out.push_back (std::move (moved));
    }
  return out;
}

Tensor
gather (const std::vector<Tensor> &outputs, Device target_device)
{
  BASALT_ASSERT_ARG (!outputs.empty (),
                     "gather requires at least one tensor");
  std::vector<Tensor> moved;
  moved.reserve (outputs.size ());
  for (const auto &t : outputs)
    {
      if (t.defined ())
        moved.push_back (t.to (target_device));
    }
  BASALT_ASSERT_ARG (!moved.empty (),
                     "gather requires at least one defined tensor");
  if (moved.size () == 1)
    return moved[0];
  return ops::cat (moved, 0);
}

void
broadcast_parameters (
  const std::vector<std::shared_ptr<nn::Module>> &replicas)
{
  if (replicas.size () <= 1)
    return;

  No_Grad_Guard no_grad;
  std::vector<Tensor> src_params = replicas[0]->parameters ();
  std::vector<Tensor> src_bufs = replicas[0]->buffers ();

  for (size_t r = 1; r < replicas.size (); ++r)
    {
      std::vector<Tensor> dst_params = replicas[r]->parameters ();
      BASALT_ASSERT_SHAPE (
        dst_params.size () == src_params.size (),
        "broadcast_parameters: replica parameter count mismatch");
      for (size_t p = 0; p < src_params.size (); ++p)
        {
          Tensor copied = src_params[p].detach ().to (
            dst_params[p].device (), dst_params[p].dtype ());
          dst_params[p].copy_ (copied);
        }

      std::vector<Tensor> dst_bufs = replicas[r]->buffers ();
      BASALT_ASSERT_SHAPE (
        dst_bufs.size () == src_bufs.size (),
        "broadcast_parameters: replica buffer count mismatch");
      for (size_t b = 0; b < src_bufs.size (); ++b)
        {
          Tensor copied = src_bufs[b].detach ().to (
            dst_bufs[b].device (), dst_bufs[b].dtype ());
          dst_bufs[b].copy_ (copied);
        }
    }
}

void
all_reduce_gradients (
  const std::vector<std::shared_ptr<nn::Module>> &replicas)
{
  if (replicas.size () <= 1)
    return;

  No_Grad_Guard no_grad;
  size_t num_replicas = replicas.size ();
  std::vector<std::vector<Tensor>> all_params (num_replicas);
  for (size_t r = 0; r < num_replicas; ++r)
    all_params[r] = replicas[r]->parameters ();

  size_t num_params = all_params[0].size ();
  double inv_replicas = 1.0 / static_cast<double> (num_replicas);

  for (size_t p = 0; p < num_params; ++p)
    {
      Device primary_dev = all_params[0][p].device ();
      Dtype primary_dt = all_params[0][p].dtype ();
      Tensor accum;
      size_t contrib_count = 0;

      for (size_t r = 0; r < num_replicas; ++r)
        {
          Tensor g = all_params[r][p].grad ();
          if (!g.defined ())
            continue;
          Tensor g_primary = g.detach ().to (primary_dev, primary_dt);
          if (!accum.defined ())
            accum = g_primary.clone ();
          else
            accum = ops::add (accum, g_primary);
          ++contrib_count;
        }

      if (accum.defined () && contrib_count > 0)
        {
          Tensor avg_grad = ops::mul_scalar (accum, inv_replicas);
          for (size_t r = 0; r < num_replicas; ++r)
            {
              Device r_dev = all_params[r][p].device ();
              Dtype r_dt = all_params[r][p].dtype ();
              all_params[r][p].set_grad (avg_grad.to (r_dev, r_dt));
            }
        }
    }
}

Data_Parallel::Data_Parallel (Replica_Factory factory,
                              std::vector<Device> devices)
  : devices_ (std::move (devices))
{
  if (devices_.empty ())
    {
      size_t n_cuda = cuda_device_count ();
      if (n_cuda >= 2)
        {
          for (size_t i = 0; i < n_cuda; ++i)
            devices_.push_back (Device::cuda (i));
        }
      else
        {
          devices_.push_back (Device::cpu ());
          devices_.push_back (Device::cpu ());
        }
    }

  replicas_.reserve (devices_.size ());
  for (size_t i = 0; i < devices_.size (); ++i)
    {
      std::shared_ptr<nn::Module> rep = factory ();
      BASALT_ASSERT_ARG (rep != nullptr,
                         "Data_Parallel replica factory returned nullptr");
      rep->to (devices_[i]);
      replicas_.push_back (rep);
    }

  register_module ("module", replicas_[0]);
  broadcast_parameters (replicas_);
}

Data_Parallel::~Data_Parallel (void) noexcept = default;

void
Data_Parallel::sync_replicas (void)
{
  broadcast_parameters (replicas_);
}

void
Data_Parallel::reduce_gradients (void)
{
  all_reduce_gradients (replicas_);
}

Tensor
Data_Parallel::forward (const Tensor &input)
{
  sync_replicas ();
  std::vector<Tensor> in_chunks = scatter (input, devices_);
  size_t n_active = in_chunks.size ();
  std::vector<Tensor> out_chunks (n_active);

  hyper_math::parallel_for (
    static_cast<size_t> (0), n_active, static_cast<size_t> (1),
    [&] (size_t r)
    {
      out_chunks[r] = replicas_[r]->forward (in_chunks[r]);
    });

  return gather (out_chunks, devices_[0]);
}

Tensor
Data_Parallel::parallel_step (
  const Tensor &inputs,
  const Tensor &targets,
  const std::function<Tensor (const Tensor &, const Tensor &)> &loss_fn)
{
  sync_replicas ();
  for (auto &rep : replicas_)
    rep->zero_grad ();

  std::vector<Tensor> in_chunks = scatter (inputs, devices_);
  std::vector<Tensor> tgt_chunks = scatter (targets, devices_);
  size_t n_active = in_chunks.size ();
  size_t total_batch = inputs.size (0);

  std::vector<double> shard_losses (n_active, 0.0);

  /* Execute forward passes across device replicas in parallel over Bellows'
     thread pool, then run reverse-mode backward and all-reduce gradients.  */
  std::vector<Tensor> preds (n_active);
  hyper_math::parallel_for (
    static_cast<size_t> (0), n_active, static_cast<size_t> (1),
    [&] (size_t r)
    {
      preds[r] = replicas_[r]->forward (in_chunks[r]);
    });

  double weighted_loss_sum = 0.0;
  for (size_t r = 0; r < n_active; ++r)
    {
      size_t shard_bs = in_chunks[r].size (0);
      double weight = (static_cast<double> (shard_bs)
                       * static_cast<double> (replicas_.size ()))
                      / static_cast<double> (total_batch);
      Tensor raw_loss = loss_fn (preds[r], tgt_chunks[r]);
      shard_losses[r] = raw_loss.item_double ();
      weighted_loss_sum += shard_losses[r]
                           * (static_cast<double> (shard_bs)
                              / static_cast<double> (total_batch));
      Tensor scaled_loss = ops::mul_scalar (raw_loss, weight);
      scaled_loss.backward ();
    }

  reduce_gradients ();
  return scalar_tensor (weighted_loss_sum,
                        Tensor_Options ().with_device (devices_[0]));
}

} /* namespace basalt::distributed */
