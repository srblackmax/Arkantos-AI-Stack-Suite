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

#ifndef BASALT_SERIALIZATION_HPP
#define BASALT_SERIALIZATION_HPP

/* Versioned, self-describing native checkpoint format with FNV-1a integrity
   verification and backward-compatibility guarantees, plus Safetensors and
   ONNX binary protobuf import/export.  */

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/nn.hpp>
#include <basalt/optim.hpp>
#include <basalt/tensor.hpp>

namespace basalt::io
{

inline constexpr uint32_t k_checkpoint_magic = 0x544C5342U; /* "BSLT" */
inline constexpr uint32_t k_checkpoint_version = 1U;

/* Complete training and model state stored in a native Basalt checkpoint.  */
struct BASALT_API Checkpoint
{
  uint32_t format_version = k_checkpoint_version;
  size_t epoch = 0;
  size_t global_step = 0;
  double best_metric = 0.0;
  uint64_t rng_seed = 0;
  nn::State_Dict model_state;
  std::map<std::string, Tensor> optimizer_state;
  std::map<std::string, std::string> metadata;
};

/* Save and load versioned self-describing native checkpoints (.bckpt).  */
BASALT_API void
save_checkpoint (const std::string &path, const Checkpoint &ckpt);

BASALT_API Checkpoint
load_checkpoint (const std::string &path);

BASALT_API void
save_model (const std::string &path, const nn::Module &module);

BASALT_API void
load_model (const std::string &path, nn::Module &module, bool strict = true);

/* Import and export Hugging Face Safetensors (.safetensors) files.  */
BASALT_API void
save_safetensors (
  const std::string &path,
  const std::map<std::string, Tensor> &tensors,
  const std::map<std::string, std::string> &metadata = {});

BASALT_API std::map<std::string, Tensor>
load_safetensors (const std::string &path,
                  std::map<std::string, std::string> *out_metadata = nullptr);

/* Import and export ONNX (.onnx) binary protobuf model files.  */
BASALT_API void
export_onnx (const std::string &path,
             const nn::Module &module,
             const Shape_Vector &sample_input_shape = {},
             const std::string &model_name = "basalt_model");

BASALT_API std::map<std::string, Tensor>
import_onnx_weights (const std::string &path,
                     std::string *out_producer = nullptr,
                     int64_t *out_ir_version = nullptr);

BASALT_API void
load_onnx_model (const std::string &path, nn::Module &module,
                 bool strict = true);

/* Inspect any supported artifact (.bckpt, .safetensors, or .onnx) and return
   a human-readable summary string.  */
BASALT_API std::string
inspect_artifact (const std::string &path);

} /* namespace basalt::io */

namespace basalt
{
using io::Checkpoint;
using io::export_onnx;
using io::import_onnx_weights;
using io::inspect_artifact;
using io::load_checkpoint;
using io::load_model;
using io::load_onnx_model;
using io::load_safetensors;
using io::save_checkpoint;
using io::save_model;
using io::save_safetensors;
} /* namespace basalt */

#endif /* BASALT_SERIALIZATION_HPP */
