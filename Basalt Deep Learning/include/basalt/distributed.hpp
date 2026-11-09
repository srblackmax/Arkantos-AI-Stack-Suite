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

#ifndef BASALT_DISTRIBUTED_HPP
#define BASALT_DISTRIBUTED_HPP

/* Multi-device data-parallel training primitives: batch scatter/gather,
   parallel replica execution over Bellows thread pools, parameter broadcast,
   and deterministic all-reduce gradient synchronization.  */

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/device.hpp>
#include <basalt/nn.hpp>
#include <basalt/tensor.hpp>

namespace basalt::distributed
{

/* Split a batch tensor along dimension 0 across the specified devices.  */
BASALT_API std::vector<Tensor>
scatter (const Tensor &batch, const std::vector<Device> &devices);

/* Gather tensors from multiple devices onto TARGET_DEVICE along dim 0.  */
BASALT_API Tensor
gather (const std::vector<Tensor> &outputs, Device target_device);

/* Synchronize (average) gradients across a list of replica modules and copy
   the averaged gradients into every replica's parameters.  */
BASALT_API void
all_reduce_gradients (
  const std::vector<std::shared_ptr<nn::Module>> &replicas);

/* Broadcast parameters and buffers from REPLICAS[0] to all other replicas.  */
BASALT_API void
broadcast_parameters (
  const std::vector<std::shared_ptr<nn::Module>> &replicas);

/* Multi-device data-parallel module wrapper.
   Maintains one replica per device, synchronizes parameters before each
   forward pass, executes replicas in parallel across devices, and all-reduces
   parameter gradients across devices.  */
class BASALT_API Data_Parallel : public nn::Module
{
public:
  using Replica_Factory = std::function<std::shared_ptr<nn::Module> (void)>;

  Data_Parallel (Replica_Factory factory,
                 std::vector<Device> devices = {});

  ~Data_Parallel (void) noexcept override;

  Tensor
  forward (const Tensor &input) override;

  /* Execute data-parallel forward + backward across all devices and
     all-reduce gradients into the primary module's parameters.
     Returns the weighted average loss scalar on the primary device.  */
  Tensor
  parallel_step (
    const Tensor &inputs,
    const Tensor &targets,
    const std::function<Tensor (const Tensor &, const Tensor &)> &loss_fn);

  void
  sync_replicas (void);

  void
  reduce_gradients (void);

  std::shared_ptr<nn::Module>
  primary_module (void) const noexcept
  {
    return replicas_.empty () ? nullptr : replicas_[0];
  }

  const std::vector<std::shared_ptr<nn::Module>> &
  replicas (void) const noexcept
  {
    return replicas_;
  }

  const std::vector<Device> &
  devices (void) const noexcept
  {
    return devices_;
  }

private:
  std::vector<Device> devices_;
  std::vector<std::shared_ptr<nn::Module>> replicas_;
};

} /* namespace basalt::distributed */

namespace basalt
{
using distributed::Data_Parallel;
} /* namespace basalt */

#endif /* BASALT_DISTRIBUTED_HPP */
