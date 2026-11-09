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

#include <surya/nn/runtime.hpp>

#include <basalt/basalt.hpp>
#include <obsidian/safetensors.hpp>

namespace surya::inline v1::nn
{

Vision_Runtime::Vision_Runtime (std::shared_ptr<Vision_Module> model,
                                Normalize_Params norm,
                                bool swap_rb)
  : model_ (std::move (model)), norm_ (norm), swap_rb_ (swap_rb)
{
  if (model_ != nullptr)
    {
      model_->eval ();
    }
}

core::Result<void>
Vision_Runtime::save_safetensors (
  const std::string &path,
  const std::map<std::string, std::string> &metadata) const
{
  if (model_ == nullptr || model_->native_module_ptr () == nullptr)
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Vision_Runtime::save_safetensors: null model");
    }
  const auto *native_mod =
    static_cast<const basalt::nn::Module *> (model_->native_module_ptr ());
  auto res = obsidian::save_module_safetensors (*native_mod, path, metadata);
  if (!res.has_value ())
    {
      return core::make_error (core::Error_Code::Io_Error,
                               res.error ().message ());
    }
  return {};
}

core::Result<void>
Vision_Runtime::load_safetensors (const std::string &path, bool strict)
{
  if (model_ == nullptr || model_->native_module_ptr () == nullptr)
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Vision_Runtime::load_safetensors: null model");
    }
  auto *native_mod =
    static_cast<basalt::nn::Module *> (model_->native_module_ptr ());
  auto res = obsidian::load_module_safetensors (*native_mod, path, strict);
  if (!res.has_value ())
    {
      return core::make_error (core::Error_Code::Io_Error,
                               res.error ().message ());
    }
  model_->eval ();
  return {};
}

core::Result<core::SuryaTensor>
Vision_Runtime::infer_tensor (const core::SuryaTensor &input_nchw) const
{
  if (model_ == nullptr)
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Vision_Runtime::infer_tensor: null model");
    }
  if (input_nchw.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Vision_Runtime::infer_tensor: empty tensor");
    }
  model_->eval ();
  return model_->forward (input_nchw);
}

core::Result<core::SuryaTensor>
Vision_Runtime::infer_image (core::Const_View_U8C3 image) const
{
  auto t_res = image_to_tensor_nchw (image, norm_, swap_rb_);
  if (!t_res.has_value ())
    {
      return t_res;
    }
  return infer_tensor (*t_res);
}

core::Result<core::SuryaTensor>
Vision_Runtime::infer_batch (
  std::span<const core::Const_View_U8C3> batch) const
{
  auto t_res = batch_images_to_tensor_nchw (batch, norm_, swap_rb_);
  if (!t_res.has_value ())
    {
      return t_res;
    }
  return infer_tensor (*t_res);
}

core::Result<std::vector<std::string>>
Vision_Runtime::inspect_checkpoint_tensors (const std::string &path)
{
  auto st_res = obsidian::SafeTensors_File::open (path);
  if (!st_res.has_value ())
    {
      return core::make_error (core::Error_Code::Io_Error,
                               st_res.error ().message ());
    }
  return st_res->tensor_names ();
}

} /* namespace surya::inline v1::nn */
