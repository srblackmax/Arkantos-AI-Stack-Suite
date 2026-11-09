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

#include <surya/detection.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace surya;
using namespace surya::core;
using namespace surya::detection;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_detection.cpp:" << line << "]: " << expr
                << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_letterbox_and_iou_metrics (void)
{
  Image_U8C3 src (1280U, 720U, Scalar<uint8_t, 3U> (20U, 60U, 180U));
  auto lb_res = letterbox_image (src.const_view (), 640U, 640U);
  CHECK (lb_res.has_value ());
  CHECK (lb_res->image.width () == 640U);
  CHECK (lb_res->image.height () == 640U);
  CHECK (std::fabs (lb_res->transform.scale - 0.5F) < 1e-5F);
  CHECK (std::fabs (lb_res->transform.pad_left - 0.0F) < 1e-5F);
  CHECK (std::fabs (lb_res->transform.pad_top - 140.0F) < 1e-5F);

  /* Top pad row should have pad color 114, center row should have image.  */
  CHECK (lb_res->image.at (10U, 320U, 0U) == 114U);
  CHECK (lb_res->image.at (320U, 320U, 2U) == 180U);

  /* Unletterbox a box from 640x640 letterbox coords back to 1280x720.  */
  const Rectf lb_box (100.0F, 240.0F, 200.0F, 100.0F);
  const Rectf orig_box = lb_res->transform.unletterbox_box (lb_box);
  CHECK (std::fabs (orig_box.x - 200.0F) < 1e-4F);
  CHECK (std::fabs (orig_box.y - 200.0F) < 1e-4F);
  CHECK (std::fabs (orig_box.width - 400.0F) < 1e-4F);
  CHECK (std::fabs (orig_box.height - 200.0F) < 1e-4F);

  /* Verify IoU >= DIoU >= CIoU and GIoU properties.  */
  const Rectf a (10.0F, 10.0F, 100.0F, 100.0F);
  const Rectf b (60.0F, 60.0F, 100.0F, 100.0F);
  const float iou = box_iou (a, b);
  const float giou = box_giou (a, b);
  const float diou = box_diou (a, b);
  const float ciou = box_ciou (a, b);
  CHECK (std::fabs (iou - (2500.0F / 17500.0F)) < 1e-5F);
  CHECK (giou < iou);
  CHECK (diou < iou);
  CHECK ( std::fabs (ciou - diou) < 1e-5F);
}

void
test_nms_dfl_masks_and_oks (void)
{
  /* 1. Standard NMS, Batched NMS, and Soft-NMS.  */
  const std::vector<Detection_Box> boxes = {
    {Rectf (10.0F, 10.0F, 100.0F, 100.0F), 0, 0.95F},
    {Rectf (14.0F, 12.0F, 100.0F, 100.0F), 0, 0.80F},
    {Rectf (14.0F, 12.0F, 100.0F, 100.0F), 1, 0.85F},
    {Rectf (300.0F, 300.0F, 80.0F, 80.0F), 0, 0.70F}
  };

  auto keep_agn = nms (boxes, 0.5F);
  CHECK (keep_agn.size () == 2U);

  auto keep_cls = batched_nms (boxes, 0.5F);
  CHECK (keep_cls.size () == 3U);

  auto soft_out = soft_nms (boxes, 0.5F, 0.5F, 0.05F, Soft_NMS_Method::Linear);
  CHECK (soft_out.size () == 4U);

  /* 2. DFL 16-bin box decoder.  */
  std::vector<float> dfl_logits (1U * 4U * 16U, -20.0F);
  /* Peak bin 2 for left/top, bin 4 for right/bottom -> 2*8=16, 4*8=32.  */
  dfl_logits[0U * 16U + 2U] = 20.0F;
  dfl_logits[1U * 16U + 2U] = 20.0F;
  dfl_logits[2U * 16U + 4U] = 20.0F;
  dfl_logits[3U * 16U + 4U] = 20.0F;

  const std::vector<float> cls_scores = {0.1F, 0.92F, 0.05F};
  const std::vector<Point2f> anchors = {Point2f (10.0F, 10.0F)};
  const std::vector<float> strides = {8.0F};

  auto dfl_dets = decode_dfl_detections (
    dfl_logits, cls_scores, anchors, strides, 3U, 16U, 0.5F);
  CHECK (dfl_dets.has_value ());
  CHECK (dfl_dets->size () == 1U);
  CHECK ((*dfl_dets)[0].class_id == 1);
  CHECK (std::fabs ((*dfl_dets)[0].box.x - 64.0F) < 1e-2F);
  CHECK (std::fabs ((*dfl_dets)[0].box.width - 48.0F) < 1e-2F);

  /* 3. Prototype instance mask assembly.  */
  std::vector<float> protos (2U * 16U * 16U, -5.0F);
  for (size_t y = 4U; y < 12U; ++y)
    {
      for (size_t x = 4U; x < 12U; ++x)
        {
          protos[0U * 256U + y * 16U + x] = 5.0F;
        }
    }
  const std::vector<float> coeffs = {1.0F, 0.0F};
  const std::vector<Detection_Box> mask_dets = {
    {Rectf (8.0F, 8.0F, 16.0F, 16.0F), 0, 0.9F}
  };

  auto masks_res = decode_instance_masks (
    protos, 2U, 16U, 16U, coeffs, mask_dets, 32U, 32U, 0.5F);
  CHECK (masks_res.has_value () && masks_res->size () == 1U);
  CHECK ((*masks_res)[0].at (16U, 16U, 0U) == 255U);
  CHECK ((*masks_res)[0].at (2U, 2U, 0U) == 0U);

  /* 4. Pose OKS & OKS-NMS.  */
  Pose_Detection p1 {
    Rectf (10.0F, 10.0F, 50.0F, 100.0F), 0.95F,
    {{25.0F, 30.0F, 0.9F}, {35.0F, 60.0F, 0.9F}}
  };
  Pose_Detection p2 {
    Rectf (11.0F, 10.0F, 50.0F, 100.0F), 0.75F,
    {{25.2F, 30.1F, 0.9F}, {35.1F, 60.2F, 0.9F}}
  };
  const std::vector<float> sigmas = {0.05F, 0.05F};
  CHECK (compute_oks (p1, p2, sigmas) > 0.95F);
  const std::vector<Pose_Detection> poses = {p1, p2};
  auto oks_keep = oks_nms (poses, sigmas, 0.5F);
  CHECK (oks_keep.size () == 1U && oks_keep[0] == 0U);
}

void
test_byte_tracker_hyper_math_kalman (void)
{
  Byte_Tracker tracker (0.5F, 0.1F, 0.3F, 10U);

  /* Frame 1: High-confidence detection spawns Track ID 1.  */
  const std::vector<Detection_Box> f1 = {
    {Rectf (100.0F, 100.0F, 50.0F, 80.0F), 0, 0.90F}
  };
  auto t1 = tracker.update (f1);
  CHECK (t1.size () == 1U);
  CHECK (t1[0].track_id == 1ULL);

  /* Frame 2: Object moves +4px right, occluded low score (0.25) rescued!  */
  const std::vector<Detection_Box> f2 = {
    {Rectf (104.0F, 100.0F, 50.0F, 80.0F), 0, 0.25F}
  };
  auto t2 = tracker.update (f2);
  CHECK (t2.size () == 1U);
  CHECK (t2[0].track_id == 1ULL);
  CHECK (t2[0].hits == 2U);

  /* Frame 3: Object moves +4px right, high score (0.88) continues Track 1.  */
  const std::vector<Detection_Box> f3 = {
    {Rectf (108.0F, 100.0F, 50.0F, 80.0F), 0, 0.88F}
  };
  auto t3 = tracker.update (f3);
  CHECK (t3.size () == 1U);
  CHECK (t3[0].track_id == 1ULL);
  CHECK (t3[0].hits == 3U);
  CHECK (std::fabs (t3[0].box.x - 108.0F) < 3.0F);
}

} /* namespace */

int
main (void)
{
  test_letterbox_and_iou_metrics ();
  test_nms_dfl_masks_and_oks ();
  test_byte_tracker_hyper_math_kalman ();

  if (g_failures != 0)
    {
      std::cerr << "test_detection FAILED with " << g_failures
                << " error(s).\n";
      return EXIT_FAILURE;
    }
  std::cout << "test_detection PASSED all assertions.\n";
  return EXIT_SUCCESS;
}
