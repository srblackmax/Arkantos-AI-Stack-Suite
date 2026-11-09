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

#include <basalt/trainer.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <basalt/autograd.hpp>
#include <basalt/error.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>

namespace basalt
{

/* --- Loggers --- */

Logger::~Logger (void) noexcept = default;

void
Logger::log_hyperparams (const std::map<std::string, std::string> &params)
{
  (void) params;
}

void
Logger::finalize (void)
{
}

Console_Logger::Console_Logger (bool quiet)
  : quiet_ (quiet)
{
}

Console_Logger::~Console_Logger (void) noexcept = default;

void
Console_Logger::log_metrics (const std::map<std::string, double> &metrics,
                             size_t step,
                             size_t epoch)
{
  if (quiet_)
    return;
  std::lock_guard<std::mutex> lock (mutex_);
  std::cerr << "[epoch " << epoch << " | step " << step << "]";
  for (const auto &kv : metrics)
    std::cerr << " " << kv.first << "=" << std::fixed << std::setprecision (4)
              << kv.second;
  std::cerr << "\n";
}

CSV_Logger::CSV_Logger (const std::string &file_path)
  : path_ (file_path),
    out_ (file_path, std::ios::out | std::ios::trunc),
    header_written_ (false)
{
  if (!out_)
    BASALT_THROW (IO_Error,
                  "CSV_Logger: cannot open '" + file_path + "' for writing");
}

CSV_Logger::~CSV_Logger (void) noexcept = default;

void
CSV_Logger::log_metrics (const std::map<std::string, double> &metrics,
                         size_t step,
                         size_t epoch)
{
  std::lock_guard<std::mutex> lock (mutex_);
  if (!header_written_)
    {
      columns_.clear ();
      for (const auto &kv : metrics)
        columns_.push_back (kv.first);
      out_ << "epoch,step";
      for (const auto &col : columns_)
        out_ << "," << col;
      out_ << "\n";
      header_written_ = true;
    }

  out_ << epoch << "," << step;
  for (const auto &col : columns_)
    {
      out_ << ",";
      auto it = metrics.find (col);
      if (it != metrics.end ())
        out_ << it->second;
    }
  out_ << "\n";
  out_.flush ();
}

void
CSV_Logger::finalize (void)
{
  std::lock_guard<std::mutex> lock (mutex_);
  if (out_.is_open ())
    out_.flush ();
}

JSON_Lines_Logger::JSON_Lines_Logger (const std::string &file_path)
  : path_ (file_path),
    out_ (file_path, std::ios::out | std::ios::trunc)
{
  if (!out_)
    BASALT_THROW (IO_Error,
                  "JSON_Lines_Logger: cannot open '" + file_path + "'");
}

JSON_Lines_Logger::~JSON_Lines_Logger (void) noexcept = default;

void
JSON_Lines_Logger::log_metrics (const std::map<std::string, double> &metrics,
                                size_t step,
                                size_t epoch)
{
  std::lock_guard<std::mutex> lock (mutex_);
  out_ << "{\"epoch\":" << epoch << ",\"step\":" << step;
  for (const auto &kv : metrics)
    out_ << ",\"" << kv.first << "\":" << kv.second;
  out_ << "}\n";
  out_.flush ();
}

void
JSON_Lines_Logger::log_hyperparams (
  const std::map<std::string, std::string> &params)
{
  std::lock_guard<std::mutex> lock (mutex_);
  out_ << "{\"type\":\"hyperparams\"";
  for (const auto &kv : params)
    out_ << ",\"" << kv.first << "\":\"" << kv.second << "\"";
  out_ << "}\n";
  out_.flush ();
}

void
JSON_Lines_Logger::finalize (void)
{
  std::lock_guard<std::mutex> lock (mutex_);
  if (out_.is_open ())
    out_.flush ();
}

Memory_Logger::~Memory_Logger (void) noexcept = default;

void
Memory_Logger::log_metrics (const std::map<std::string, double> &metrics,
                            size_t step,
                            size_t epoch)
{
  std::lock_guard<std::mutex> lock (mutex_);
  Record rec;
  rec.step = step;
  rec.epoch = epoch;
  rec.metrics = metrics;
  history_.push_back (std::move (rec));
}

/* --- Callbacks --- */

Callback::~Callback (void) noexcept = default;

void
Callback::on_fit_start (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
}

void
Callback::on_fit_end (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
}

void
Callback::on_epoch_start (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
}

void
Callback::on_epoch_end (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
}

void
Callback::on_batch_start (Trainer &trainer, nn::Module &model,
                          size_t batch_idx)
{
  (void) trainer;
  (void) model;
  (void) batch_idx;
}

void
Callback::on_batch_end (Trainer &trainer, nn::Module &model, size_t batch_idx)
{
  (void) trainer;
  (void) model;
  (void) batch_idx;
}

void
Callback::on_validation_end (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
}

Early_Stopping::Early_Stopping (std::string monitor,
                                size_t patience,
                                double min_delta,
                                bool minimize)
  : monitor_ (std::move (monitor)),
    patience_ (patience),
    min_delta_ (min_delta),
    minimize_ (minimize),
    wait_count_ (0),
    best_score_ (minimize ? std::numeric_limits<double>::infinity ()
                          : -std::numeric_limits<double>::infinity ()),
    triggered_ (false)
{
}

Early_Stopping::~Early_Stopping (void) noexcept = default;

void
Early_Stopping::on_fit_start (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
  wait_count_ = 0;
  best_score_ = minimize_ ? std::numeric_limits<double>::infinity ()
                          : -std::numeric_limits<double>::infinity ();
  triggered_ = false;
}

void
Early_Stopping::on_epoch_end (Trainer &trainer, nn::Module &model)
{
  (void) model;
  const auto &metrics = trainer.callback_metrics ();
  auto it = metrics.find (monitor_);
  if (it == metrics.end ())
    return;

  double current = it->second;
  bool improved = minimize_ ? (current < (best_score_ - min_delta_))
                            : (current > (best_score_ + min_delta_));
  if (improved)
    {
      best_score_ = current;
      wait_count_ = 0;
      trainer.set_best_metric (current);
    }
  else
    {
      ++wait_count_;
      if (wait_count_ >= patience_)
        {
          triggered_ = true;
          trainer.request_stop (true);
        }
    }
}

Model_Checkpoint::Model_Checkpoint (std::string file_path,
                                    std::string monitor,
                                    bool save_best_only,
                                    bool minimize)
  : file_path_ (std::move (file_path)),
    monitor_ (std::move (monitor)),
    save_best_only_ (save_best_only),
    minimize_ (minimize),
    best_score_ (minimize ? std::numeric_limits<double>::infinity ()
                          : -std::numeric_limits<double>::infinity ())
{
}

Model_Checkpoint::~Model_Checkpoint (void) noexcept = default;

void
Model_Checkpoint::on_fit_start (Trainer &trainer, nn::Module &model)
{
  (void) trainer;
  (void) model;
  best_score_ = minimize_ ? std::numeric_limits<double>::infinity ()
                          : -std::numeric_limits<double>::infinity ();
}

void
Model_Checkpoint::on_epoch_end (Trainer &trainer, nn::Module &model)
{
  if (!save_best_only_)
    {
      trainer.save_checkpoint (file_path_, model, trainer.active_optimizer ());
      best_path_ = file_path_;
      return;
    }

  const auto &metrics = trainer.callback_metrics ();
  auto it = metrics.find (monitor_);
  if (it == metrics.end ())
    return;

  double current = it->second;
  bool improved = minimize_ ? (current < best_score_)
                            : (current > best_score_);
  if (improved)
    {
      best_score_ = current;
      trainer.set_best_metric (current);
      trainer.save_checkpoint (file_path_, model, trainer.active_optimizer ());
      best_path_ = file_path_;
    }
}

LR_Monitor::LR_Monitor (std::string metric_name)
  : metric_name_ (std::move (metric_name))
{
}

LR_Monitor::~LR_Monitor (void) noexcept = default;

void
LR_Monitor::on_epoch_end (Trainer &trainer, nn::Module &model)
{
  (void) model;
  if (trainer.active_optimizer () != nullptr)
    trainer.record_metric (metric_name_, trainer.active_optimizer ()->lr ());
}

/* --- Trainable_Module --- */

Trainable_Module::~Trainable_Module (void) noexcept = default;

Tensor
Trainable_Module::validation_step (const data::Batch &batch, size_t batch_idx)
{
  (void) batch;
  (void) batch_idx;
  return Tensor ();
}

Tensor
Trainable_Module::test_step (const data::Batch &batch, size_t batch_idx)
{
  return validation_step (batch, batch_idx);
}

void
Trainable_Module::on_fit_start (void)
{
}

void
Trainable_Module::on_fit_end (void)
{
}

void
Trainable_Module::on_epoch_start (void)
{
}

void
Trainable_Module::on_epoch_end (void)
{
}

void
Trainable_Module::on_before_backward (const Tensor &loss)
{
  (void) loss;
}

void
Trainable_Module::on_after_backward (void)
{
}

void
Trainable_Module::on_before_optimizer_step (optim::Optimizer &opt)
{
  (void) opt;
}

void
Trainable_Module::log (const std::string &name, double value)
{
  step_metrics_[name] = value;
}

/* --- Trainer --- */

Trainer::Trainer (Trainer_Config config,
                  std::vector<std::shared_ptr<Callback>> callbacks,
                  std::vector<std::shared_ptr<Logger>> loggers)
  : config_ (std::move (config)),
    callbacks_ (std::move (callbacks)),
    loggers_ (std::move (loggers)),
    scaler_ (65536.0, 2.0, 0.5, 2000, config_.use_amp)
{
  if (config_.enable_console_log)
    loggers_.push_back (std::make_shared<Console_Logger> (false));
}

void
Trainer::save_checkpoint (const std::string &path,
                          const nn::Module &model,
                          const optim::Optimizer *optimizer) const
{
  io::Checkpoint ckpt;
  ckpt.format_version = io::k_checkpoint_version;
  ckpt.epoch = current_epoch_;
  ckpt.global_step = global_step_;
  ckpt.best_metric = best_metric_;
  ckpt.rng_seed = config_.seed;
  ckpt.model_state = model.state_dict ();
  if (optimizer != nullptr)
    ckpt.optimizer_state = optimizer->state_dict ();
  io::save_checkpoint (path, ckpt);
}

void
Trainer::load_checkpoint (const std::string &path,
                          nn::Module &model,
                          optim::Optimizer *optimizer)
{
  io::Checkpoint ckpt = io::load_checkpoint (path);
  current_epoch_ = ckpt.epoch;
  global_step_ = ckpt.global_step;
  best_metric_ = ckpt.best_metric;
  model.load_state_dict (ckpt.model_state);
  if (optimizer != nullptr && !ckpt.optimizer_state.empty ())
    optimizer->load_state_dict (ckpt.optimizer_state);
}

void
Trainer::fit (Trainable_Module &model,
              data::Data_Loader &train_loader,
              data::Data_Loader *val_loader)
{
  if (config_.seed != 0)
    seed_everything (config_.seed);

  Device primary_dev = config_.devices.empty ()
                       ? config_.device
                       : config_.devices[0];
  model.to (primary_dev);

  Optimizer_Configuration opt_cfg = model.configure_optimizers ();
  BASALT_ASSERT_ARG (opt_cfg.optimizer != nullptr,
                     "configure_optimizers() must return a valid Optimizer");
  owned_optimizer_ = opt_cfg.optimizer;
  active_optimizer_ = owned_optimizer_.get ();

  size_t start_epoch = 0;
  if (!config_.resume_from_checkpoint.empty ())
    {
      load_checkpoint (config_.resume_from_checkpoint, model,
                       active_optimizer_);
      start_epoch = current_epoch_ + 1;
    }
  else
    {
      current_epoch_ = 0;
      global_step_ = 0;
    }

  should_stop_ = false;
  model.on_fit_start ();
  for (auto &cb : callbacks_)
    cb->on_fit_start (*this, model);

  size_t accum_steps =
    config_.accumulate_grad_batches > 0 ? config_.accumulate_grad_batches : 1;

  for (size_t epoch = start_epoch; epoch < config_.max_epochs; ++epoch)
    {
      if (should_stop_)
        break;
      current_epoch_ = epoch;
      train_loader.set_epoch (epoch);
      model.train (true);
      model.on_epoch_start ();
      for (auto &cb : callbacks_)
        cb->on_epoch_start (*this, model);

      size_t num_batches = train_loader.num_batches ();
      double epoch_loss_sum = 0.0;
      size_t epoch_batch_count = 0;

      active_optimizer_->zero_grad ();

      train_loader.for_each_batch (
        [&] (size_t b, const data::Batch &raw_batch)
        {
          if (should_stop_)
            return;
          if (config_.max_steps > 0 && global_step_ >= config_.max_steps)
            {
              should_stop_ = true;
              return;
            }

          model.clear_logged_metrics ();
          for (auto &cb : callbacks_)
            cb->on_batch_start (*this, model, b);

          data::Batch batch;
          batch.inputs = raw_batch.inputs.defined ()
                         ? raw_batch.inputs.to (primary_dev)
                         : Tensor ();
          batch.targets = raw_batch.targets.defined ()
                          ? raw_batch.targets.to (primary_dev)
                          : Tensor ();

          Tensor loss;
          {
            amp::Autocast_Guard amp_guard (config_.amp_dtype, config_.use_amp);
            loss = model.training_step (batch, b);
          }

          double raw_loss_val = loss.item_double ();
          epoch_loss_sum += raw_loss_val;
          ++epoch_batch_count;

          Tensor scaled_loss =
            (accum_steps > 1)
            ? ops::div_scalar (loss, static_cast<double> (accum_steps))
            : loss;
          if (config_.use_amp)
            scaled_loss = scaler_.scale (scaled_loss);

          model.on_before_backward (scaled_loss);
          scaled_loss.backward ();
          model.on_after_backward ();

          bool is_step_boundary =
            ((b + 1) % accum_steps == 0) || (b + 1 == num_batches);
          if (is_step_boundary)
            {
              if (config_.use_amp)
                scaler_.unscale_ (*active_optimizer_);

              if (config_.gradient_clip_val > 0.0)
                optim::clip_grad_norm_ (model.parameters (),
                                        config_.gradient_clip_val);

              model.on_before_optimizer_step (*active_optimizer_);
              if (config_.use_amp)
                {
                  scaler_.step (*active_optimizer_);
                  scaler_.update ();
                }
              else
                {
                  active_optimizer_->step ();
                }
              active_optimizer_->zero_grad ();
              ++global_step_;

              for (const auto &mkv : model.logged_metrics ())
                callback_metrics_[mkv.first] = mkv.second;
              callback_metrics_["train_loss"] = raw_loss_val;

              if (config_.log_every_n_steps > 0
                  && global_step_ % config_.log_every_n_steps == 0)
                {
                  for (auto &lg : loggers_)
                    lg->log_metrics (callback_metrics_, global_step_, epoch);
                }
            }

          for (auto &cb : callbacks_)
            cb->on_batch_end (*this, model, b);
        });

      if (epoch_batch_count > 0)
        callback_metrics_["train_loss"] =
          epoch_loss_sum / static_cast<double> (epoch_batch_count);

      if (val_loader != nullptr)
        {
          auto val_metrics = validate (model, *val_loader);
          for (const auto &kv : val_metrics)
            callback_metrics_[kv.first] = kv.second;
          for (auto &cb : callbacks_)
            cb->on_validation_end (*this, model);
        }

      if (opt_cfg.scheduler != nullptr)
        {
          if (auto *plateau = dynamic_cast<optim::Reduce_LR_On_Plateau *> (
                opt_cfg.scheduler.get ()))
            {
              double metric_for_sched = callback_metrics_.count ("val_loss")
                                        ? callback_metrics_["val_loss"]
                                        : callback_metrics_["train_loss"];
              plateau->step (metric_for_sched);
            }
          else
            {
              opt_cfg.scheduler->step ();
            }
        }

      model.on_epoch_end ();
      for (auto &cb : callbacks_)
        cb->on_epoch_end (*this, model);

      for (auto &lg : loggers_)
        lg->log_metrics (callback_metrics_, global_step_, epoch);
    }

  model.on_fit_end ();
  for (auto &cb : callbacks_)
    cb->on_fit_end (*this, model);
  for (auto &lg : loggers_)
    lg->finalize ();
}

void
Trainer::fit (nn::Module &model,
              optim::Optimizer &optimizer,
              const Loss_Function &loss_fn,
              data::Data_Loader &train_loader,
              data::Data_Loader *val_loader,
              optim::LR_Scheduler *scheduler,
              distributed::Data_Parallel::Replica_Factory dp_factory)
{
  if (config_.seed != 0)
    seed_everything (config_.seed);

  Device primary_dev = config_.devices.empty ()
                       ? config_.device
                       : config_.devices[0];
  model.to (primary_dev);
  active_optimizer_ = &optimizer;

  std::unique_ptr<distributed::Data_Parallel> dp_wrapper;
  if (config_.devices.size () > 1 && dp_factory != nullptr)
    {
      dp_wrapper = std::make_unique<distributed::Data_Parallel> (
        dp_factory, config_.devices);
      No_Grad_Guard ng;
      dp_wrapper->primary_module ()->load_state_dict (model.state_dict ());
      dp_wrapper->sync_replicas ();
    }

  size_t start_epoch = 0;
  if (!config_.resume_from_checkpoint.empty ())
    {
      load_checkpoint (config_.resume_from_checkpoint, model, &optimizer);
      start_epoch = current_epoch_ + 1;
    }
  else
    {
      current_epoch_ = 0;
      global_step_ = 0;
    }

  should_stop_ = false;
  for (auto &cb : callbacks_)
    cb->on_fit_start (*this, model);

  size_t accum_steps =
    config_.accumulate_grad_batches > 0 ? config_.accumulate_grad_batches : 1;

  for (size_t epoch = start_epoch; epoch < config_.max_epochs; ++epoch)
    {
      if (should_stop_)
        break;
      current_epoch_ = epoch;
      train_loader.set_epoch (epoch);
      model.train (true);
      for (auto &cb : callbacks_)
        cb->on_epoch_start (*this, model);

      size_t num_batches = train_loader.num_batches ();
      double epoch_loss_sum = 0.0;
      size_t epoch_batch_count = 0;

      optimizer.zero_grad ();

      train_loader.for_each_batch (
        [&] (size_t b, const data::Batch &raw_batch)
        {
          if (should_stop_)
            return;
          if (config_.max_steps > 0 && global_step_ >= config_.max_steps)
            {
              should_stop_ = true;
              return;
            }

          for (auto &cb : callbacks_)
            cb->on_batch_start (*this, model, b);

          Tensor inputs = raw_batch.inputs.to (primary_dev);
          Tensor targets = raw_batch.targets.to (primary_dev);

          if (dp_wrapper != nullptr)
            {
              No_Grad_Guard ng;
              dp_wrapper->primary_module ()->load_state_dict (
                model.state_dict ());
            }

          double raw_loss_val = 0.0;
          if (dp_wrapper != nullptr)
            {
              Tensor dp_loss = dp_wrapper->parallel_step (
                inputs, targets, loss_fn);
              raw_loss_val = dp_loss.item_double ();
              auto src_p = dp_wrapper->primary_module ()->parameters ();
              auto dst_p = model.parameters ();
              for (size_t p = 0; p < dst_p.size (); ++p)
                {
                  if (src_p[p].grad ().defined ())
                    {
                      Tensor g = src_p[p].grad ();
                      if (accum_steps > 1)
                        g = ops::div_scalar (
                          g, static_cast<double> (accum_steps));
                      if (!dst_p[p].grad ().defined ())
                        dst_p[p].set_grad (g.clone ());
                      else
                        dst_p[p].set_grad (ops::add (dst_p[p].grad (), g));
                    }
                }
            }
          else
            {
              Tensor loss;
              {
                amp::Autocast_Guard amp_guard (
                  config_.amp_dtype, config_.use_amp);
                Tensor preds = model.forward (inputs);
                loss = loss_fn (preds, targets);
              }
              raw_loss_val = loss.item_double ();
              Tensor scaled_loss =
                (accum_steps > 1)
                ? ops::div_scalar (loss, static_cast<double> (accum_steps))
                : loss;
              if (config_.use_amp)
                scaled_loss = scaler_.scale (scaled_loss);
              scaled_loss.backward ();
            }

          epoch_loss_sum += raw_loss_val;
          ++epoch_batch_count;

          bool is_step_boundary =
            ((b + 1) % accum_steps == 0) || (b + 1 == num_batches);
          if (is_step_boundary)
            {
              if (config_.use_amp && dp_wrapper == nullptr)
                scaler_.unscale_ (optimizer);

              if (config_.gradient_clip_val > 0.0)
                optim::clip_grad_norm_ (model.parameters (),
                                        config_.gradient_clip_val);

              if (config_.use_amp && dp_wrapper == nullptr)
                {
                  scaler_.step (optimizer);
                  scaler_.update ();
                }
              else
                {
                  optimizer.step ();
                }
              optimizer.zero_grad ();
              ++global_step_;

              callback_metrics_["train_loss"] = raw_loss_val;
              if (config_.log_every_n_steps > 0
                  && global_step_ % config_.log_every_n_steps == 0)
                {
                  for (auto &lg : loggers_)
                    lg->log_metrics (callback_metrics_, global_step_, epoch);
                }
            }

          for (auto &cb : callbacks_)
            cb->on_batch_end (*this, model, b);
        });

      if (epoch_batch_count > 0)
        callback_metrics_["train_loss"] =
          epoch_loss_sum / static_cast<double> (epoch_batch_count);

      if (val_loader != nullptr)
        {
          auto val_metrics = validate (model, loss_fn, *val_loader);
          for (const auto &kv : val_metrics)
            callback_metrics_[kv.first] = kv.second;
          for (auto &cb : callbacks_)
            cb->on_validation_end (*this, model);
        }

      if (scheduler != nullptr)
        {
          if (auto *plateau = dynamic_cast<optim::Reduce_LR_On_Plateau *> (
                scheduler))
            {
              double metric_for_sched = callback_metrics_.count ("val_loss")
                                        ? callback_metrics_["val_loss"]
                                        : callback_metrics_["train_loss"];
              plateau->step (metric_for_sched);
            }
          else
            {
              scheduler->step ();
            }
        }

      for (auto &cb : callbacks_)
        cb->on_epoch_end (*this, model);

      for (auto &lg : loggers_)
        lg->log_metrics (callback_metrics_, global_step_, epoch);
    }

  for (auto &cb : callbacks_)
    cb->on_fit_end (*this, model);
  for (auto &lg : loggers_)
    lg->finalize ();
}

std::map<std::string, double>
Trainer::validate (Trainable_Module &model, data::Data_Loader &val_loader)
{
  No_Grad_Guard no_grad;
  model.eval ();
  Device primary_dev = config_.devices.empty ()
                       ? config_.device
                       : config_.devices[0];

  double loss_sum = 0.0;
  size_t count = 0;
  size_t num_batches = 0;
  std::map<std::string, double> metric_sums;

  val_loader.for_each_batch (
    [&] (size_t b, const data::Batch &raw_batch)
    {
      ++num_batches;
      model.clear_logged_metrics ();
      data::Batch batch;
      batch.inputs = raw_batch.inputs.defined ()
                     ? raw_batch.inputs.to (primary_dev)
                     : Tensor ();
      batch.targets = raw_batch.targets.defined ()
                      ? raw_batch.targets.to (primary_dev)
                      : Tensor ();

      Tensor loss = model.validation_step (batch, b);
      if (loss.defined ())
        {
          loss_sum += loss.item_double ();
          ++count;
        }
      for (const auto &kv : model.logged_metrics ())
        metric_sums[kv.first] += kv.second;
    });

  std::map<std::string, double> result;
  if (count > 0)
    result["val_loss"] = loss_sum / static_cast<double> (count);
  if (num_batches > 0)
    {
      double inv_b = 1.0 / static_cast<double> (num_batches);
      for (const auto &kv : metric_sums)
        result[kv.first] = kv.second * inv_b;
    }
  model.train (true);
  return result;
}

std::map<std::string, double>
Trainer::validate (nn::Module &model,
                   const Loss_Function &loss_fn,
                   data::Data_Loader &val_loader)
{
  No_Grad_Guard no_grad;
  model.eval ();
  Device primary_dev = config_.devices.empty ()
                       ? config_.device
                       : config_.devices[0];

  double loss_sum = 0.0;
  size_t count = 0;

  val_loader.for_each_batch (
    [&] (size_t, const data::Batch &raw_batch)
    {
      Tensor inputs = raw_batch.inputs.to (primary_dev);
      Tensor targets = raw_batch.targets.to (primary_dev);
      Tensor preds = model.forward (inputs);
      Tensor loss = loss_fn (preds, targets);
      loss_sum += loss.item_double ();
      ++count;
    });

  std::map<std::string, double> result;
  if (count > 0)
    result["val_loss"] = loss_sum / static_cast<double> (count);
  model.train (true);
  return result;
}

std::map<std::string, double>
Trainer::test (Trainable_Module &model, data::Data_Loader &test_loader)
{
  No_Grad_Guard no_grad;
  model.eval ();
  Device primary_dev = config_.devices.empty ()
                       ? config_.device
                       : config_.devices[0];

  double loss_sum = 0.0;
  size_t count = 0;
  size_t num_batches = 0;
  std::map<std::string, double> metric_sums;

  test_loader.for_each_batch (
    [&] (size_t b, const data::Batch &raw_batch)
    {
      ++num_batches;
      model.clear_logged_metrics ();
      data::Batch batch;
      batch.inputs = raw_batch.inputs.defined ()
                     ? raw_batch.inputs.to (primary_dev)
                     : Tensor ();
      batch.targets = raw_batch.targets.defined ()
                      ? raw_batch.targets.to (primary_dev)
                      : Tensor ();

      Tensor loss = model.test_step (batch, b);
      if (loss.defined ())
        {
          loss_sum += loss.item_double ();
          ++count;
        }
      for (const auto &kv : model.logged_metrics ())
        metric_sums[kv.first] += kv.second;
    });

  std::map<std::string, double> result;
  if (count > 0)
    result["test_loss"] = loss_sum / static_cast<double> (count);
  if (num_batches > 0)
    {
      double inv_b = 1.0 / static_cast<double> (num_batches);
      for (const auto &kv : metric_sums)
        result[kv.first] = kv.second * inv_b;
    }
  model.train (true);
  return result;
}

} /* namespace basalt */
