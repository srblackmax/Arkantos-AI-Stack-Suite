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

#include <surya/nn.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>

using namespace surya;
using namespace surya::core;
using namespace surya::nn;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_nn.cpp:" << line << "]: " << expr << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_tensor_bridge (void)
{
  Image_U8C3 img (16U, 12U, Scalar<uint8_t, 3U> (51U, 102U, 153U));
  auto t_res = image_to_tensor_nchw (img.const_view ());
  CHECK (t_res.has_value ());
  CHECK (t_res->ndim () == 4U);
  CHECK (t_res->size (0) == 1U);
  CHECK (t_res->size (1) == 3U);
  CHECK (t_res->size (2) == 12U);
  CHECK (t_res->size (3) == 16U);

  auto back = tensor_nchw_to_image_u8 (*t_res, 0U, 255.0F);
  CHECK (back.has_value ());
  CHECK (back->at (5U, 5U, 0U) == 51U);
  CHECK (back->at (5U, 5U, 1U) == 102U);
  CHECK (back->at (5U, 5U, 2U) == 153U);

  const std::array<Const_View_U8C3, 2U> views = {img.const_view (),
                                                 img.const_view ()};
  auto bt = batch_images_to_tensor_nchw (views);
  CHECK (bt.has_value ());
  CHECK (bt->size (0) == 2U);
}

void
test_conv_bn_fusion_and_safetensors (void)
{
  /* 1. Verify analytical Conv2d + BatchNorm2d folding matches unfused.  */
  auto block = std::make_shared<Conv_BN_Act> (
    3U, 8U, 3U, 1U, 1U, Activation_Kind::SiLU);
  block->randomize_bn_parameters ();
  block->eval ();

  Image_U8C3 sample (16U, 16U, Scalar<uint8_t, 3U> (80U, 120U, 200U));
  auto x_res = image_to_tensor_nchw (sample.const_view ());
  CHECK (x_res.has_value ());

  SuryaTensor y_unfused = block->forward (*x_res);

  block->fuse_for_inference ();
  CHECK (block->is_fused ());

  SuryaTensor y_fused = block->forward (*x_res);
  CHECK (y_unfused.allclose (y_fused, 1e-4F, 1e-5F));

  /* 2. Full Vision_Classifier + Obsidian SafeTensors round-trip.  */
  auto model1 = std::make_shared<Vision_Classifier> (3U, 4U, 5U);
  Vision_Runtime rt1 (model1, Normalize_Params::imagenet ());

  auto out1 = rt1.infer_image (sample.const_view ());
  CHECK (out1.has_value ());
  CHECK (out1->ndim () == 2U);
  CHECK (out1->size (0) == 1U);
  CHECK (out1->size (1) == 5U);

  const std::string ckpt_path = "/tmp/surya_vision_classifier.safetensors";
  auto save_res = rt1.save_safetensors (ckpt_path, {{"arch", "surya_cnn"}});
  CHECK (save_res.has_value ());

  auto names_res = Vision_Runtime::inspect_checkpoint_tensors (ckpt_path);
  CHECK (names_res.has_value () && !names_res->empty ());

  auto model2 = std::make_shared<Vision_Classifier> (3U, 4U, 5U);
  /* Perturb model2 weights before loading checkpoint.  */
  model2->zero_head_weights ();
  Vision_Runtime rt2 (model2, Normalize_Params::imagenet ());

  auto load_res = rt2.load_safetensors (ckpt_path, true);
  CHECK (load_res.has_value ());

  auto out2 = rt2.infer_image (sample.const_view ());
  CHECK (out2.has_value ());
  CHECK (out1->allclose (*out2, 1e-5F, 1e-6F));

  std::filesystem::remove (ckpt_path);
}

} /* namespace */

int
main (void)
{
  test_tensor_bridge ();
  test_conv_bn_fusion_and_safetensors ();

  if (g_failures > 0)
    {
      std::cerr << "test_nn FAILED with " << g_failures << " error(s).\n";
      return 1;
    }
  std::cout << "test_nn PASSED.\n";
  return 0;
}
