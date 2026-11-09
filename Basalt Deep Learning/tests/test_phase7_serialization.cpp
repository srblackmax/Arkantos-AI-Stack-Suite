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
#include <fstream>
#include <memory>
#include <basalt/error.hpp>
#include <basalt/nn.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>
#include <basalt/serialization.hpp>
#include <basalt/tensor.hpp>

using namespace basalt;

static void
test_checkpoint_roundtrip (void)
{
  seed_everything (101);
  nn::Sequential model;
  model.add ("fc1", std::make_shared<nn::Linear> (4, 6));
  model.add ("relu", std::make_shared<nn::ReLU> ());
  model.add ("fc2", std::make_shared<nn::Linear> (6, 2));

  Tensor x = randn ({3, 4});
  Tensor expected_y = model.forward (x);

  const std::string path = "test_model_v1.bckpt";
  Checkpoint ckpt;
  ckpt.format_version = 1U;
  ckpt.epoch = 12;
  ckpt.global_step = 480;
  ckpt.best_metric = 0.985;
  ckpt.rng_seed = 101;
  ckpt.model_state = model.state_dict ();
  ckpt.optimizer_state["step"] = full ({1}, 480.0);
  ckpt.metadata["arch"] = "mlp_4_6_2";

  save_checkpoint (path, ckpt);

  Checkpoint loaded = load_checkpoint (path);
  BASALT_ASSERT (loaded.format_version == 1U, "Checkpoint version mismatch");
  BASALT_ASSERT (loaded.epoch == 12, "Checkpoint epoch mismatch");
  BASALT_ASSERT (loaded.global_step == 480, "Checkpoint step mismatch");
  BASALT_ASSERT (std::fabs (loaded.best_metric - 0.985) < 1e-12,
                 "Checkpoint best_metric mismatch");
  BASALT_ASSERT (loaded.metadata["arch"] == "mlp_4_6_2",
                 "Checkpoint metadata mismatch");

  nn::Sequential model2;
  model2.add ("fc1", std::make_shared<nn::Linear> (4, 6));
  model2.add ("relu", std::make_shared<nn::ReLU> ());
  model2.add ("fc2", std::make_shared<nn::Linear> (6, 2));
  model2.load_state_dict (loaded.model_state);

  Tensor actual_y = model2.forward (x);
  for (size_t i = 0; i < expected_y.numel (); ++i)
    BASALT_ASSERT (std::fabs (expected_y.read_scalar_at (i)
                              - actual_y.read_scalar_at (i)) < 1e-6,
                   "Checkpoint model output mismatch");

  std::string summary = inspect_artifact (path);
  BASALT_ASSERT (summary.find ("Basalt Native Checkpoint")
                 != std::string::npos,
                 "inspect_artifact should identify .bckpt");

  /* Verify FNV-1a corruption detection.  */
  {
    std::fstream f (path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp (-4, std::ios::end);
    char bad = 0x5A;
    f.write (&bad, 1);
  }
  bool caught_corruption = false;
  try
    {
      (void) load_checkpoint (path);
    }
  catch (const Basalt_Error &)
    {
      caught_corruption = true;
    }
  BASALT_ASSERT (caught_corruption,
                 "Corrupted checkpoint should fail FNV-1a verification");
  std::remove (path.c_str ());
}

static void
test_checkpoint_v0_backward_compat (void)
{
  const std::string path = "test_model_v0.bckpt";
  Checkpoint ckpt;
  ckpt.format_version = 0U;
  ckpt.epoch = 3;
  ckpt.global_step = 30;
  ckpt.model_state["w"] = tensor ({1.0, 2.0, 3.0});

  save_checkpoint (path, ckpt);
  Checkpoint loaded = load_checkpoint (path);
  BASALT_ASSERT (loaded.format_version == 0U, "v0 version preserved");
  BASALT_ASSERT (loaded.epoch == 3, "v0 epoch preserved");
  BASALT_ASSERT (loaded.model_state.count ("w") == 1, "v0 tensor preserved");
  BASALT_ASSERT (std::fabs (loaded.model_state["w"].read_scalar_at (1) - 2.0)
                 < 1e-6,
                 "v0 tensor value preserved");
  std::remove (path.c_str ());
}

static void
test_safetensors_roundtrip (void)
{
  const std::string path = "test_weights.safetensors";
  std::map<std::string, Tensor> tensors;
  tensors["f32_w"] = tensor ({1.5, -2.5, 3.25, 4.0}).reshape ({2, 2});
  tensors["f16_w"] = tensors["f32_w"].to (Dtype::Float16);
  tensors["bf16_w"] = tensors["f32_w"].to (Dtype::BFloat16);
  tensors["i64_idx"] = tensor (
    {10.0, 20.0, 30.0}, Tensor_Options ().with_dtype (Dtype::Int64));

  std::map<std::string, std::string> meta;
  meta["format"] = "pt";
  meta["framework"] = "Basalt Deep Learning";

  save_safetensors (path, tensors, meta);

  std::map<std::string, std::string> loaded_meta;
  auto loaded = load_safetensors (path, &loaded_meta);

  BASALT_ASSERT (loaded_meta["framework"] == "Basalt Deep Learning",
                 "Safetensors metadata round-trip mismatch");
  BASALT_ASSERT (loaded.size () == 4, "Safetensors tensor count mismatch");
  BASALT_ASSERT (loaded["f16_w"].dtype () == Dtype::Float16,
                 "Safetensors fp16 dtype preserved");
  BASALT_ASSERT (loaded["bf16_w"].dtype () == Dtype::BFloat16,
                 "Safetensors bf16 dtype preserved");
  BASALT_ASSERT (std::fabs (loaded["f32_w"].read_scalar_at (1) - (-2.5))
                 < 1e-6,
                 "Safetensors f32 value mismatch");
  BASALT_ASSERT (std::fabs (loaded["i64_idx"].read_scalar_at (2) - 30.0)
                 < 1e-6,
                 "Safetensors i64 value mismatch");

  std::string summary = inspect_artifact (path);
  BASALT_ASSERT (summary.find ("Safetensors") != std::string::npos,
                 "inspect_artifact should identify Safetensors");
  std::remove (path.c_str ());
}

static void
test_onnx_roundtrip (void)
{
  seed_everything (202);
  nn::Sequential model1;
  model1.add ("fc1", std::make_shared<nn::Linear> (4, 5));
  model1.add ("act", std::make_shared<nn::ReLU> ());
  model1.add ("fc2", std::make_shared<nn::Linear> (5, 2));

  Tensor x = randn ({2, 4});
  Tensor y1 = model1.forward (x);

  const std::string path = "test_graph.onnx";
  export_onnx (path, model1, {2, 4}, "test_mlp");

  std::string producer;
  int64_t ir_v = 0;
  auto weights = import_onnx_weights (path, &producer, &ir_v);
  BASALT_ASSERT (producer == "Basalt Deep Learning",
                 "ONNX producer_name should be 'Basalt Deep Learning'");
  BASALT_ASSERT (ir_v == 8, "ONNX ir_version should be 8");
  BASALT_ASSERT (weights.size () == 4, "ONNX initializer count should be 4");

  nn::Sequential model2;
  model2.add ("fc1", std::make_shared<nn::Linear> (4, 5));
  model2.add ("act", std::make_shared<nn::ReLU> ());
  model2.add ("fc2", std::make_shared<nn::Linear> (5, 2));
  load_onnx_model (path, model2);

  Tensor y2 = model2.forward (x);
  for (size_t i = 0; i < y1.numel (); ++i)
    BASALT_ASSERT (std::fabs (y1.read_scalar_at (i)
                              - y2.read_scalar_at (i)) < 1e-6,
                   "ONNX round-trip output mismatch");

  std::string summary = inspect_artifact (path);
  BASALT_ASSERT (summary.find ("ONNX Model") != std::string::npos,
                 "inspect_artifact should identify ONNX");
  std::remove (path.c_str ());
}

int
main (void)
{
  test_checkpoint_roundtrip ();
  test_checkpoint_v0_backward_compat ();
  test_safetensors_roundtrip ();
  test_onnx_roundtrip ();
  return 0;
}
