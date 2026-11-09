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

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <basalt/basalt.hpp>

namespace
{

const char *const k_program_name = "basalt-train";

void
print_version (void)
{
  std::cout
    << "Basalt Deep Learning " << basalt_version_string () << "\n"
    << "Copyright (C) 2026 Kival\n"
    << "License GPLv3+: GNU GPL version 3 or later "
       "<https://gnu.org/licenses/gpl.html>\n"
    << "This is free software: you are free to change and redistribute it.\n"
    << "There is NO WARRANTY, to the extent permitted by law.\n";
}

void
print_help (void)
{
  std::cout
    << "Usage: " << k_program_name << " [OPTION]...\n"
    << "Train a neural network model using Basalt Deep Learning.\n\n"
    << "  -e, --epochs=N         number of training epochs (default: 20)\n"
    << "  -b, --batch-size=N     mini-batch size (default: 16)\n"
    << "  -l, --lr=RATE          learning rate (default: 0.01)\n"
    << "  -s, --seed=SEED        deterministic RNG seed (default: 42)\n"
    << "  -d, --device=DEV       target device (cpu or cuda:0, default: cpu)\n"
    << "  -o, --output=FILE      save trained checkpoint to FILE\n"
    << "  -r, --resume=FILE      resume training from checkpoint FILE\n"
    << "      --amp              enable mixed-precision (bf16) training\n"
    << "  -n, --dry-run          validate configuration without training\n"
    << "  -v, --verbose          print detailed progress per epoch\n"
    << "  -q, --quiet            suppress non-error output\n"
    << "  -h, --help             display this help and exit\n"
    << "  -V, --version          output version information and exit\n\n"
    << "Report bugs to <Kival@basalt-dl.org>.\n";
}

class CLI_MLP_Model : public basalt::Trainable_Module
{
public:
  explicit CLI_MLP_Model (double lr)
    : lr_ (lr)
  {
    fc1_ = register_module ("fc1",
                            std::make_shared<basalt::nn::Linear> (4, 16));
    relu_ = register_module ("relu", std::make_shared<basalt::nn::ReLU> ());
    fc2_ = register_module ("fc2",
                            std::make_shared<basalt::nn::Linear> (16, 1));
  }

  basalt::Tensor
  forward (const basalt::Tensor &x) override
  {
    return fc2_->forward (relu_->forward (fc1_->forward (x)));
  }

  basalt::Tensor
  training_step (const basalt::data::Batch &batch, size_t batch_idx) override
  {
    (void) batch_idx;
    basalt::Tensor pred = forward (batch.inputs);
    return basalt::loss::mse_loss (pred, batch.targets);
  }

  basalt::Optimizer_Configuration
  configure_optimizers (void) override
  {
    auto opt = std::make_shared<basalt::optim::Adam> (parameters (), lr_);
    return {opt, nullptr};
  }

private:
  double lr_;
  std::shared_ptr<basalt::nn::Linear> fc1_;
  std::shared_ptr<basalt::nn::ReLU> relu_;
  std::shared_ptr<basalt::nn::Linear> fc2_;
};

} /* anonymous namespace */

int
main (int argc, char **argv)
{
  size_t epochs = 20;
  size_t batch_size = 16;
  double lr = 0.01;
  uint64_t seed = 42;
  std::string device_str = "cpu";
  std::string output_path;
  std::string resume_path;
  bool use_amp = false;
  bool dry_run = false;
  bool verbose = false;
  bool quiet = false;

  static const struct option long_options[] = {
    {"epochs", required_argument, nullptr, 'e'},
    {"batch-size", required_argument, nullptr, 'b'},
    {"lr", required_argument, nullptr, 'l'},
    {"seed", required_argument, nullptr, 's'},
    {"device", required_argument, nullptr, 'd'},
    {"output", required_argument, nullptr, 'o'},
    {"resume", required_argument, nullptr, 'r'},
    {"amp", no_argument, nullptr, 1001},
    {"dry-run", no_argument, nullptr, 'n'},
    {"verbose", no_argument, nullptr, 'v'},
    {"quiet", no_argument, nullptr, 'q'},
    {"help", no_argument, nullptr, 'h'},
    {"version", no_argument, nullptr, 'V'},
    {nullptr, 0, nullptr, 0}
  };

  opterr = 0;
  int opt = 0;
  while ((opt = getopt_long (
            argc, argv, "e:b:l:s:d:o:r:nvqhV", long_options, nullptr)) != -1)
    {
      switch (opt)
        {
        case 'e':
          epochs = static_cast<size_t> (std::stoull (optarg));
          break;
        case 'b':
          batch_size = static_cast<size_t> (std::stoull (optarg));
          break;
        case 'l':
          lr = std::stod (optarg);
          break;
        case 's':
          seed = static_cast<uint64_t> (std::stoull (optarg));
          break;
        case 'd':
          device_str = optarg;
          break;
        case 'o':
          output_path = optarg;
          break;
        case 'r':
          resume_path = optarg;
          break;
        case 1001:
          use_amp = true;
          break;
        case 'n':
          dry_run = true;
          break;
        case 'v':
          verbose = true;
          break;
        case 'q':
          quiet = true;
          break;
        case 'h':
          print_help ();
          return EXIT_SUCCESS;
        case 'V':
          print_version ();
          return EXIT_SUCCESS;
        default:
          basalt::report_tool_diagnostic (
            k_program_name, "unrecognized option; try '--help'");
          return EXIT_FAILURE;
        }
    }

  try
    {
      if (dry_run)
        {
          if (!quiet)
            std::cout << k_program_name << ": dry-run valid (epochs="
                      << epochs << ", batch_size=" << batch_size
                      << ", lr=" << lr << ", seed=" << seed << ")\n";
          return EXIT_SUCCESS;
        }

      basalt::seed_everything (seed);
      basalt::Tensor x = basalt::randn ({64, 4});
      basalt::Tensor w =
        basalt::tensor ({1.5, -0.5, 2.0, 0.25}).reshape ({4, 1});
      basalt::Tensor y = basalt::ops::matmul (x, w);
      auto ds = std::make_shared<basalt::data::Tensor_Dataset> (x, y);
      basalt::data::Data_Loader loader (
        ds, batch_size, true, 0, 2, false, seed);

      CLI_MLP_Model model (lr);
      basalt::Trainer_Config cfg;
      cfg.max_epochs = epochs;
      cfg.seed = seed;
      cfg.device = basalt::Device::parse (device_str);
      cfg.use_amp = use_amp;
      cfg.amp_dtype = basalt::Dtype::BFloat16;
      cfg.resume_from_checkpoint = resume_path;
      cfg.enable_console_log = verbose && !quiet;

      basalt::Trainer trainer (cfg);
      trainer.fit (model, loader);

      if (!output_path.empty ())
        trainer.save_checkpoint (output_path, model,
                                 trainer.active_optimizer ());

      if (!quiet)
        std::cout << "Final train_loss: "
                  << trainer.callback_metrics ().at ("train_loss") << "\n";
      return EXIT_SUCCESS;
    }
  catch (const std::exception &e)
    {
      basalt::report_tool_diagnostic (k_program_name, e.what ());
      return EXIT_FAILURE;
    }
}
