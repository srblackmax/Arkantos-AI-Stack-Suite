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

#include <cmath>
#include <cstdio>
#include <memory>
#include <basalt/callbacks.hpp>
#include <basalt/data.hpp>
#include <basalt/distributed.hpp>
#include <basalt/error.hpp>
#include <basalt/logging.hpp>
#include <basalt/loss.hpp>
#include <basalt/metrics.hpp>
#include <basalt/nn.hpp>
#include <basalt/ops.hpp>
#include <basalt/optim.hpp>
#include <basalt/random.hpp>
#include <basalt/scheduler.hpp>
#include <basalt/trainer.hpp>

using namespace basalt;

class Synthetic_Regressor : public Trainable_Module
{
public:
  Synthetic_Regressor (void)
  {
    fc1_ = register_module ("fc1", std::make_shared<nn::Linear> (4, 16));
    relu_ = register_module ("relu", std::make_shared<nn::ReLU> ());
    fc2_ = register_module ("fc2", std::make_shared<nn::Linear> (16, 1));
  }

  Tensor
  forward (const Tensor &x) override
  {
    return fc2_->forward (relu_->forward (fc1_->forward (x)));
  }

  Tensor
  training_step (const data::Batch &batch, size_t batch_idx) override
  {
    (void) batch_idx;
    Tensor pred = forward (batch.inputs);
    Tensor loss = loss::mse_loss (pred, batch.targets);
    log ("batch_mse", loss.item_double ());
    return loss;
  }

  Tensor
  validation_step (const data::Batch &batch, size_t batch_idx) override
  {
    (void) batch_idx;
    Tensor pred = forward (batch.inputs);
    Tensor loss = loss::mse_loss (pred, batch.targets);
    log ("val_mae", metrics::mean_absolute_error (pred, batch.targets));
    return loss;
  }

  Optimizer_Configuration
  configure_optimizers (void) override
  {
    auto opt = std::make_shared<optim::Adam> (parameters (), 0.02);
    auto sched = std::make_shared<optim::Step_LR> (*opt, 5, 0.8);
    return {opt, sched};
  }

private:
  std::shared_ptr<nn::Linear> fc1_;
  std::shared_ptr<nn::ReLU> relu_;
  std::shared_ptr<nn::Linear> fc2_;
};

static void
test_trainer_end_to_end_and_resume (void)
{
  seed_everything (1234);

  /* Generate synthetic regression target y = 2*x0 - x1 + 0.5*x2 + 1.0.  */
  size_t n_samples = 64;
  Tensor x = randn ({n_samples, 4});
  Tensor w_true = tensor ({2.0, -1.0, 0.5, 0.0}).reshape ({4, 1});
  Tensor y = ops::add_scalar (ops::matmul (x, w_true), 1.0);

  auto ds = std::make_shared<data::Tensor_Dataset> (x, y);
  data::Data_Loader train_loader (ds, 16, true, 2, 2, false, 77);
  data::Data_Loader val_loader (ds, 32, false, 0);

  const std::string ckpt_path = "trainer_best.bckpt";
  const std::string csv_path = "trainer_metrics.csv";
  const std::string jsonl_path = "trainer_metrics.jsonl";

  auto ckpt_cb = std::make_shared<Model_Checkpoint> (
    ckpt_path, "val_loss", true, true);
  auto es_cb = std::make_shared<Early_Stopping> ("val_loss", 20, 1e-6, true);
  auto lr_cb = std::make_shared<LR_Monitor> ("lr");

  auto mem_logger = std::make_shared<Memory_Logger> ();
  auto csv_logger = std::make_shared<CSV_Logger> (csv_path);
  auto jsonl_logger = std::make_shared<JSON_Lines_Logger> (jsonl_path);

  Synthetic_Regressor model;
  auto init_val = Trainer ().validate (model, val_loader);
  double initial_loss = init_val["val_loss"];

  Trainer_Config cfg;
  cfg.max_epochs = 25;
  cfg.accumulate_grad_batches = 2;
  cfg.gradient_clip_val = 5.0;
  cfg.use_amp = true;
  cfg.amp_dtype = Dtype::BFloat16;
  cfg.seed = 1234;
  cfg.log_every_n_steps = 2;

  Trainer trainer (
    cfg, {ckpt_cb, es_cb, lr_cb}, {mem_logger, csv_logger, jsonl_logger});
  trainer.fit (model, train_loader, &val_loader);

  auto final_val = trainer.validate (model, val_loader);
  double final_loss = final_val["val_loss"];
  BASALT_ASSERT (final_loss < initial_loss * 0.1,
                 "Trainer should converge to < 10% of initial MSE loss");
  BASALT_ASSERT (!mem_logger->history ().empty (),
                 "Memory_Logger should have recorded metrics");
  BASALT_ASSERT (!ckpt_cb->best_model_path ().empty (),
                 "Model_Checkpoint should have saved best model");

  /* Test resuming from the saved checkpoint for 5 more epochs.  */
  Trainer_Config resume_cfg = cfg;
  resume_cfg.max_epochs = 30;
  resume_cfg.resume_from_checkpoint = ckpt_path;

  Synthetic_Regressor resumed_model;
  Trainer resume_trainer (resume_cfg);
  resume_trainer.fit (resumed_model, train_loader, &val_loader);

  auto resumed_val = resume_trainer.validate (resumed_model, val_loader);
  BASALT_ASSERT (resumed_val["val_loss"] < initial_loss * 0.1,
                 "Resumed model should preserve trained performance");

  std::remove (ckpt_path.c_str ());
  std::remove (csv_path.c_str ());
  std::remove (jsonl_path.c_str ());
}

static void
test_multi_device_data_parallel_equivalence (void)
{
  auto make_mlp = [] (void) -> std::shared_ptr<nn::Module>
  {
    auto seq = std::make_shared<nn::Sequential> ();
    seq->add ("fc1", std::make_shared<nn::Linear> (4, 8));
    seq->add ("act", std::make_shared<nn::ReLU> ());
    seq->add ("fc2", std::make_shared<nn::Linear> (8, 2));
    return seq;
  };

  seed_everything (555);
  auto single_model = make_mlp ();

  std::vector<Device> dp_devices = {Device::cuda (0), Device::cuda (1)};
  distributed::Data_Parallel dp_model (make_mlp, dp_devices);

  /* Copy exact weights from single_model to dp_model primary module and
     broadcast to all device replicas.  */
  dp_model.primary_module ()->load_state_dict (single_model->state_dict ());
  dp_model.sync_replicas ();

  Tensor batch_x = randn ({8, 4});
  Tensor batch_y = randn ({8, 2});

  /* 1. Single-device forward + backward on full batch of 8.  */
  single_model->zero_grad ();
  Tensor single_pred = single_model->forward (batch_x);
  Tensor single_loss = loss::mse_loss (single_pred, batch_y);
  single_loss.backward ();

  /* 2. Multi-device Data_Parallel forward+backward across cuda:0 & cuda:1.  */
  Tensor dp_out = dp_model.forward (batch_x);
  for (size_t i = 0; i < single_pred.numel (); ++i)
    BASALT_ASSERT (std::fabs (single_pred.read_scalar_at (i)
                              - dp_out.read_scalar_at (i)) < 1e-5,
                   "Data_Parallel forward output must match single-device");

  Tensor dp_loss = dp_model.parallel_step (
    batch_x, batch_y,
    [] (const Tensor &p, const Tensor &t)
    {
      return loss::mse_loss (p, t);
    });

  BASALT_ASSERT (std::fabs (single_loss.item_double ()
                            - dp_loss.item_double ()) < 1e-5,
                 "Data_Parallel loss must match single-device loss");

  auto p_single = single_model->parameters ();
  auto p_dp = dp_model.primary_module ()->parameters ();
  BASALT_ASSERT (p_single.size () == p_dp.size (),
                 "Parameter count must match");

  for (size_t k = 0; k < p_single.size (); ++k)
    {
      Tensor g1 = p_single[k].grad ();
      Tensor g2 = p_dp[k].grad ().to (Device::cpu ());
      for (size_t i = 0; i < g1.numel (); ++i)
        {
          BASALT_ASSERT (std::fabs (g1.read_scalar_at (i)
                                    - g2.read_scalar_at (i)) < 1e-5,
                         "Data_Parallel all-reduced gradient mismatch");
        }
    }
}

int
main (void)
{
  test_trainer_end_to_end_and_resume ();
  test_multi_device_data_parallel_equivalence ();
  return 0;
}
