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

/* Example program: train a 2-layer MLP to solve the non-linear XOR problem
   using basalt::Trainable_Module and basalt::Trainer.  */

#include <iostream>
#include <memory>
#include <basalt/basalt.hpp>

class XOR_Module : public basalt::Trainable_Module
{
public:
  XOR_Module (void)
  {
    fc1_ = register_module ("fc1",
                            std::make_shared<basalt::nn::Linear> (2, 16));
    act_ = register_module ("act", std::make_shared<basalt::nn::GELU> ());
    fc2_ = register_module ("fc2",
                            std::make_shared<basalt::nn::Linear> (16, 1));
  }

  basalt::Tensor
  forward (const basalt::Tensor &x) override
  {
    return fc2_->forward (act_->forward (fc1_->forward (x)));
  }

  basalt::Tensor
  training_step (const basalt::data::Batch &batch, size_t batch_idx) override
  {
    (void) batch_idx;
    basalt::Tensor logits = forward (batch.inputs);
    return basalt::loss::bce_with_logits_loss (logits, batch.targets);
  }

  basalt::Optimizer_Configuration
  configure_optimizers (void) override
  {
    auto opt = std::make_shared<basalt::optim::AdamW> (
      parameters (), 0.05, 0.9, 0.999, 1e-8, 1e-4);
    return {opt, nullptr};
  }

private:
  std::shared_ptr<basalt::nn::Linear> fc1_;
  std::shared_ptr<basalt::nn::GELU> act_;
  std::shared_ptr<basalt::nn::Linear> fc2_;
};

int
main (void)
{
  basalt::seed_everything (42);

  basalt::Tensor x =
    basalt::tensor ({0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}).reshape ({4, 2});
  basalt::Tensor y =
    basalt::tensor ({0.0, 1.0, 1.0, 0.0}).reshape ({4, 1});

  auto ds = std::make_shared<basalt::data::Tensor_Dataset> (x, y);
  basalt::data::Data_Loader loader (ds, 4, true, 0, 2, false, 42);

  XOR_Module model;
  basalt::Trainer_Config cfg;
  cfg.max_epochs = 120;
  cfg.seed = 42;

  basalt::Trainer trainer (cfg);
  trainer.fit (model, loader);

  basalt::Tensor probs = basalt::ops::sigmoid (model.forward (x));
  std::cout << "XOR predictions after training:\n"
            << probs.to_string () << "\n";
  return 0;
}
