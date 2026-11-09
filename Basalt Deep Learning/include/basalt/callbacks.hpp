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

#ifndef BASALT_CALLBACKS_HPP
#define BASALT_CALLBACKS_HPP

/* Extensible lifecycle callbacks for basalt::Trainer: early stopping,
   automated model checkpointing, and learning-rate monitoring.  */

#include <cstddef>
#include <string>
#include <basalt/config.hpp>
#include <basalt/nn.hpp>

namespace basalt
{

class Trainer;

/* Base class for Trainer lifecycle hooks.  */
class BASALT_API Callback
{
public:
  Callback (void) = default;
  virtual ~Callback (void) noexcept;

  virtual void
  on_fit_start (Trainer &trainer, nn::Module &model);

  virtual void
  on_fit_end (Trainer &trainer, nn::Module &model);

  virtual void
  on_epoch_start (Trainer &trainer, nn::Module &model);

  virtual void
  on_epoch_end (Trainer &trainer, nn::Module &model);

  virtual void
  on_batch_start (Trainer &trainer, nn::Module &model, size_t batch_idx);

  virtual void
  on_batch_end (Trainer &trainer, nn::Module &model, size_t batch_idx);

  virtual void
  on_validation_end (Trainer &trainer, nn::Module &model);
};

/* Monitor a validation or training metric and request training termination
   when no improvement exceeding MIN_DELTA is seen for PATIENCE epochs.  */
class BASALT_API Early_Stopping : public Callback
{
public:
  Early_Stopping (std::string monitor = "val_loss",
                  size_t patience = 5,
                  double min_delta = 0.0,
                  bool minimize = true);

  ~Early_Stopping (void) noexcept override;

  void
  on_fit_start (Trainer &trainer, nn::Module &model) override;

  void
  on_epoch_end (Trainer &trainer, nn::Module &model) override;

  size_t
  wait_count (void) const noexcept
  {
    return wait_count_;
  }

  double
  best_score (void) const noexcept
  {
    return best_score_;
  }

  bool
  triggered (void) const noexcept
  {
    return triggered_;
  }

private:
  std::string monitor_;
  size_t patience_;
  double min_delta_;
  bool minimize_;
  size_t wait_count_;
  double best_score_;
  bool triggered_;
};

/* Automatically save native Basalt checkpoints (.bckpt) when the monitored
   metric improves or at the end of every epoch.  */
class BASALT_API Model_Checkpoint : public Callback
{
public:
  Model_Checkpoint (std::string file_path,
                    std::string monitor = "val_loss",
                    bool save_best_only = true,
                    bool minimize = true);

  ~Model_Checkpoint (void) noexcept override;

  void
  on_fit_start (Trainer &trainer, nn::Module &model) override;

  void
  on_epoch_end (Trainer &trainer, nn::Module &model) override;

  const std::string &
  best_model_path (void) const noexcept
  {
    return best_path_;
  }

  double
  best_score (void) const noexcept
  {
    return best_score_;
  }

private:
  std::string file_path_;
  std::string monitor_;
  bool save_best_only_;
  bool minimize_;
  double best_score_;
  std::string best_path_;
};

/* Record the active optimizer learning rate into Trainer metrics at the end
   of each epoch.  */
class BASALT_API LR_Monitor : public Callback
{
public:
  explicit LR_Monitor (std::string metric_name = "lr");
  ~LR_Monitor (void) noexcept override;

  void
  on_epoch_end (Trainer &trainer, nn::Module &model) override;

private:
  std::string metric_name_;
};

} /* namespace basalt */

#endif /* BASALT_CALLBACKS_HPP */
