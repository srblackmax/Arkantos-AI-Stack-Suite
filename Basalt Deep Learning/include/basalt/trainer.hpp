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

#ifndef BASALT_TRAINER_HPP
#define BASALT_TRAINER_HPP

/* PyTorch Lightning-style Trainer and Trainable_Module abstractions with
   callbacks, lifecycle hooks, early stopping, checkpoint resume, mixed
   precision (fp16/bf16), gradient accumulation, gradient clipping,
   multi-device Data_Parallel execution, pluggable loggers, and reproducible
   seeding.  */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <basalt/amp.hpp>
#include <basalt/callbacks.hpp>
#include <basalt/config.hpp>
#include <basalt/data.hpp>
#include <basalt/device.hpp>
#include <basalt/distributed.hpp>
#include <basalt/dtype.hpp>
#include <basalt/logging.hpp>
#include <basalt/nn.hpp>
#include <basalt/optim.hpp>
#include <basalt/scheduler.hpp>
#include <basalt/serialization.hpp>
#include <basalt/tensor.hpp>

namespace basalt
{

/* Bundle returned by Trainable_Module::configure_optimizers().  */
struct BASALT_API Optimizer_Configuration
{
  std::shared_ptr<optim::Optimizer> optimizer;
  std::shared_ptr<optim::LR_Scheduler> scheduler;
};

/* Lightning-style self-contained trainable module defining training,
   validation, and test steps alongside optimizer configuration.  */
class BASALT_API Trainable_Module : public nn::Module
{
public:
  Trainable_Module (void) = default;
  ~Trainable_Module (void) noexcept override;

  /* Compute training loss for a batch and optionally call log().  */
  virtual Tensor
  training_step (const data::Batch &batch, size_t batch_idx) = 0;

  /* Optional validation step; default returns undefined Tensor.  */
  virtual Tensor
  validation_step (const data::Batch &batch, size_t batch_idx);

  /* Optional test step; default delegates to validation_step.  */
  virtual Tensor
  test_step (const data::Batch &batch, size_t batch_idx);

  /* Construct optimizer and optional learning-rate scheduler.  */
  virtual Optimizer_Configuration
  configure_optimizers (void) = 0;

  /* Lifecycle hooks invoked by basalt::Trainer.  */
  virtual void
  on_fit_start (void);

  virtual void
  on_fit_end (void);

  virtual void
  on_epoch_start (void);

  virtual void
  on_epoch_end (void);

  virtual void
  on_before_backward (const Tensor &loss);

  virtual void
  on_after_backward (void);

  virtual void
  on_before_optimizer_step (optim::Optimizer &opt);

  /* Record a named scalar metric for the current step/epoch.  */
  void
  log (const std::string &name, double value);

  const std::map<std::string, double> &
  logged_metrics (void) const noexcept
  {
    return step_metrics_;
  }

  void
  clear_logged_metrics (void) noexcept
  {
    step_metrics_.clear ();
  }

private:
  std::map<std::string, double> step_metrics_;
};

/* Configuration parameters controlling basalt::Trainer execution.  */
struct BASALT_API Trainer_Config
{
  size_t max_epochs = 10;
  size_t max_steps = 0; /* 0 = unlimited.  */
  size_t accumulate_grad_batches = 1;
  double gradient_clip_val = 0.0; /* 0.0 = disabled.  */
  bool use_amp = false;
  Dtype amp_dtype = Dtype::Float16;
  Device device = Device::cpu ();
  std::vector<Device> devices; /* >1 enables multi-device Data_Parallel.  */
  uint64_t seed = 0;
  bool deterministic = true;
  std::string resume_from_checkpoint;
  size_t log_every_n_steps = 10;
  bool enable_console_log = false;
};

/* Orchestrates training, validation, testing, checkpointing, mixed precision,
   gradient accumulation, and multi-device data parallelism.  */
class BASALT_API Trainer
{
public:
  using Loss_Function =
    std::function<Tensor (const Tensor &pred, const Tensor &target)>;

  explicit Trainer (
    Trainer_Config config = Trainer_Config (),
    std::vector<std::shared_ptr<Callback>> callbacks = {},
    std::vector<std::shared_ptr<Logger>> loggers = {});

  ~Trainer (void) noexcept = default;

  /* Train a Lightning-style Trainable_Module.  */
  void
  fit (Trainable_Module &model,
       data::Data_Loader &train_loader,
       data::Data_Loader *val_loader = nullptr);

  /* Train any nn::Module with an explicit optimizer, loss function, and
     optional scheduler (and optional replica factory for multi-device
     Data_Parallel execution).  */
  void
  fit (nn::Module &model,
       optim::Optimizer &optimizer,
       const Loss_Function &loss_fn,
       data::Data_Loader &train_loader,
       data::Data_Loader *val_loader = nullptr,
       optim::LR_Scheduler *scheduler = nullptr,
       distributed::Data_Parallel::Replica_Factory dp_factory = nullptr);

  /* Run evaluation over a Data_Loader and return aggregated metrics.  */
  std::map<std::string, double>
  validate (Trainable_Module &model, data::Data_Loader &val_loader);

  std::map<std::string, double>
  validate (nn::Module &model,
            const Loss_Function &loss_fn,
            data::Data_Loader &val_loader);

  std::map<std::string, double>
  test (Trainable_Module &model, data::Data_Loader &test_loader);

  /* Save and restore full Trainer state checkpoints (.bckpt).  */
  void
  save_checkpoint (const std::string &path,
                   const nn::Module &model,
                   const optim::Optimizer *optimizer = nullptr) const;

  void
  load_checkpoint (const std::string &path,
                   nn::Module &model,
                   optim::Optimizer *optimizer = nullptr);

  /* Accessors and state inspection for callbacks.  */
  size_t
  current_epoch (void) const noexcept
  {
    return current_epoch_;
  }

  size_t
  global_step (void) const noexcept
  {
    return global_step_;
  }

  double
  best_metric (void) const noexcept
  {
    return best_metric_;
  }

  void
  set_best_metric (double val) noexcept
  {
    best_metric_ = val;
  }

  bool
  should_stop (void) const noexcept
  {
    return should_stop_;
  }

  void
  request_stop (bool stop = true) noexcept
  {
    should_stop_ = stop;
  }

  const std::map<std::string, double> &
  callback_metrics (void) const noexcept
  {
    return callback_metrics_;
  }

  void
  record_metric (const std::string &name, double value)
  {
    callback_metrics_[name] = value;
  }

  optim::Optimizer *
  active_optimizer (void) const noexcept
  {
    return active_optimizer_;
  }

  const Trainer_Config &
  config (void) const noexcept
  {
    return config_;
  }

private:
  Trainer_Config config_;
  std::vector<std::shared_ptr<Callback>> callbacks_;
  std::vector<std::shared_ptr<Logger>> loggers_;
  size_t current_epoch_ = 0;
  size_t global_step_ = 0;
  double best_metric_ = 0.0;
  bool should_stop_ = false;
  std::map<std::string, double> callback_metrics_;
  std::shared_ptr<optim::Optimizer> owned_optimizer_;
  optim::Optimizer *active_optimizer_ = nullptr;
  amp::Grad_Scaler scaler_;
};

} /* namespace basalt */

#endif /* BASALT_TRAINER_HPP */
