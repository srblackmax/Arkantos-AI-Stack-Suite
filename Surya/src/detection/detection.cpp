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

#include <surya/detection/detection.hpp>
#include <surya/imgproc/geometry.hpp>

#include <hyper_math/scientific/linalg.hpp>
#include <hyper_math/tensor/nd_tensor.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

namespace surya::inline v1::detection
{

core::Result<Letterbox_Result>
letterbox_image (core::Const_View_U8C3 src,
                 size_t dst_width,
                 size_t dst_height,
                 core::Scalar<uint8_t, 3U> pad_color)
{
  if (src.empty () || dst_width == 0U || dst_height == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "letterbox_image requires non-empty dims");
    }

  const float scale_w =
    static_cast<float> (dst_width) / static_cast<float> (src.width ());
  const float scale_h =
    static_cast<float> (dst_height) / static_cast<float> (src.height ());
  const float scale = std::min (scale_w, scale_h);

  const size_t new_w = std::clamp (
    static_cast<size_t> (
      std::lround (static_cast<float> (src.width ()) * scale)),
    size_t {1U}, dst_width);
  const size_t new_h = std::clamp (
    static_cast<size_t> (
      std::lround (static_cast<float> (src.height ()) * scale)),
    size_t {1U}, dst_height);

  auto resized_res = imgproc::resize (
    src, new_w, new_h, imgproc::Interpolation_Mode::Bilinear);
  if (!resized_res)
    {
      return std::unexpected (resized_res.error ());
    }

  const size_t pad_x = (dst_width - new_w) / 2U;
  const size_t pad_y = (dst_height - new_h) / 2U;

  core::Image_U8C3 canvas (dst_width, dst_height, pad_color);
  const core::Image_U8C3 &resized = *resized_res;
  for (size_t y = 0U; y < new_h; ++y)
    {
      const uint8_t *srow = resized.row_ptr (y);
      uint8_t *drow = canvas.row_ptr (pad_y + y) + pad_x * 3U;
      std::copy (srow, srow + new_w * 3U, drow);
    }

  Letterbox_Result out;
  out.image = std::move (canvas);
  out.transform.src_width = src.width ();
  out.transform.src_height = src.height ();
  out.transform.dst_width = dst_width;
  out.transform.dst_height = dst_height;
  out.transform.scale = scale;
  out.transform.pad_left = static_cast<float> (pad_x);
  out.transform.pad_top = static_cast<float> (pad_y);
  return out;
}

float
box_iou (const core::Rectf &a, const core::Rectf &b) noexcept
{
  const float x1 = std::max (a.x, b.x);
  const float y1 = std::max (a.y, b.y);
  const float x2 = std::min (a.x + a.width, b.x + b.width);
  const float y2 = std::min (a.y + a.height, b.y + b.height);

  const float inter_w = std::max (0.0F, x2 - x1);
  const float inter_h = std::max (0.0F, y2 - y1);
  const float inter_area = inter_w * inter_h;
  const float union_area =
    std::max (0.0F, a.width * a.height)
    + std::max (0.0F, b.width * b.height) - inter_area;

  return (union_area > 1e-8F) ? (inter_area / union_area) : 0.0F;
}

float
box_giou (const core::Rectf &a, const core::Rectf &b) noexcept
{
  const float iou = box_iou (a, b);
  const float cx1 = std::min (a.x, b.x);
  const float cy1 = std::min (a.y, b.y);
  const float cx2 = std::max (a.x + a.width, b.x + b.width);
  const float cy2 = std::max (a.y + a.height, b.y + b.height);

  const float c_area = std::max (0.0F, cx2 - cx1) * std::max (0.0F, cy2 - cy1);
  if (c_area <= 1e-8F)
    {
      return iou;
    }

  const float ix1 = std::max (a.x, b.x);
  const float iy1 = std::max (a.y, b.y);
  const float ix2 = std::min (a.x + a.width, b.x + b.width);
  const float iy2 = std::min (a.y + a.height, b.y + b.height);
  const float inter_area =
    std::max (0.0F, ix2 - ix1) * std::max (0.0F, iy2 - iy1);
  const float union_area =
    a.width * a.height + b.width * b.height - inter_area;

  return iou - (c_area - union_area) / c_area;
}

float
box_diou (const core::Rectf &a, const core::Rectf &b) noexcept
{
  const float iou = box_iou (a, b);
  const float cx1 = std::min (a.x, b.x);
  const float cy1 = std::min (a.y, b.y);
  const float cx2 = std::max (a.x + a.width, b.x + b.width);
  const float cy2 = std::max (a.y + a.height, b.y + b.height);

  const float c2 = (cx2 - cx1) * (cx2 - cx1) + (cy2 - cy1) * (cy2 - cy1);
  if (c2 <= 1e-8F)
    {
      return iou;
    }

  const float acx = a.x + 0.5F * a.width;
  const float acy = a.y + 0.5F * a.height;
  const float bcx = b.x + 0.5F * b.width;
  const float bcy = b.y + 0.5F * b.height;
  const float rho2 = (acx - bcx) * (acx - bcx) + (acy - bcy) * (acy - bcy);

  return iou - (rho2 / c2);
}

float
box_ciou (const core::Rectf &a, const core::Rectf &b) noexcept
{
  const float diou = box_diou (a, b);
  const float iou = box_iou (a, b);
  if (a.height <= 1e-8F || b.height <= 1e-8F)
    {
      return diou;
    }

  const float atan_diff =
    std::atan (a.width / a.height) - std::atan (b.width / b.height);
  const float pi = std::numbers::pi_v<float>;
  const float v = (4.0F / (pi * pi)) * atan_diff * atan_diff;
  const float denom = (1.0F - iou) + v;
  const float alpha = (denom > 1e-8F) ? (v / denom) : 0.0F;
  return diou - alpha * v;
}

std::vector<size_t>
nms (std::span<const Detection_Box> boxes, float iou_threshold) noexcept
{
  const size_t n = boxes.size ();
  std::vector<size_t> order (n);
  std::iota (order.begin (), order.end (), 0U);
  std::sort (order.begin (), order.end (),
             [&] (size_t i, size_t j)
             {
               return boxes[i].score > boxes[j].score;
             });

  std::vector<uint8_t> suppressed (n, 0U);
  std::vector<size_t> keep;
  keep.reserve (n);

  for (size_t i = 0U; i < n; ++i)
    {
      const size_t idx_i = order[i];
      if (suppressed[idx_i] != 0U)
        {
          continue;
        }
      keep.push_back (idx_i);
      for (size_t j = i + 1U; j < n; ++j)
        {
          const size_t idx_j = order[j];
          if (suppressed[idx_j] == 0U
              && box_iou (boxes[idx_i].box, boxes[idx_j].box) > iou_threshold)
            {
              suppressed[idx_j] = 1U;
            }
        }
    }
  return keep;
}

std::vector<size_t>
batched_nms (std::span<const Detection_Box> boxes,
             float iou_threshold) noexcept
{
  if (boxes.empty ())
    {
      return {};
    }

  float max_coord = 0.0F;
  for (const auto &b : boxes)
    {
      max_coord = std::max ({max_coord, b.box.x + b.box.width,
                             b.box.y + b.box.height});
    }
  const float stride = max_coord + 4096.0F;

  std::vector<Detection_Box> shifted (boxes.size ());
  for (size_t i = 0U; i < boxes.size (); ++i)
    {
      shifted[i] = boxes[i];
      const float offset = static_cast<float> (boxes[i].class_id) * stride;
      shifted[i].box.x += offset;
      shifted[i].box.y += offset;
    }

  return nms (shifted, iou_threshold);
}

std::vector<Detection_Box>
soft_nms (std::span<const Detection_Box> boxes,
          float iou_threshold,
          float sigma,
          float score_threshold,
          Soft_NMS_Method method)
{
  std::vector<Detection_Box> pool (boxes.begin (), boxes.end ());
  std::vector<Detection_Box> keep;
  keep.reserve (pool.size ());

  while (!pool.empty ())
    {
      auto max_it = std::max_element (
        pool.begin (), pool.end (),
        [] (const Detection_Box &a, const Detection_Box &b)
        {
          return a.score < b.score;
        });

      Detection_Box best = *max_it;
      *max_it = pool.back ();
      pool.pop_back ();

      if (best.score < score_threshold)
        {
          break;
        }
      keep.push_back (best);

      const float inv_sigma = (sigma > 1e-6F) ? (1.0F / sigma) : 2.0F;
      for (size_t i = 0U; i < pool.size ();)
        {
          if (pool[i].class_id == best.class_id)
            {
              const float ov = box_iou (best.box, pool[i].box);
              if (method == Soft_NMS_Method::Linear)
                {
                  if (ov > iou_threshold)
                    {
                      pool[i].score *= (1.0F - ov);
                    }
                }
              else
                {
                  pool[i].score *= std::exp (-(ov * ov) * inv_sigma);
                }
            }

          if (pool[i].score < score_threshold)
            {
              pool[i] = pool.back ();
              pool.pop_back ();
            }
          else
            {
              ++i;
            }
        }
    }
  return keep;
}

core::Result<std::vector<Detection_Box>>
decode_dfl_detections (std::span<const float> dfl_logits,
                       std::span<const float> cls_scores,
                       std::span<const core::Point2f> anchor_points,
                       std::span<const float> anchor_strides,
                       size_t num_classes,
                       size_t reg_max,
                       float conf_threshold)
{
  const size_t num_anchors = anchor_points.size ();
  if (num_anchors == 0U || num_classes == 0U || reg_max < 2U
      || anchor_strides.size () != num_anchors
      || dfl_logits.size () != num_anchors * 4U * reg_max
      || cls_scores.size () != num_anchors * num_classes)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "decode_dfl_detections shape mismatch");
    }

  std::vector<Detection_Box> detections;
  for (size_t a = 0U; a < num_anchors; ++a)
    {
      const float *scores_row = cls_scores.data () + a * num_classes;
      size_t best_cls = 0U;
      float best_score = scores_row[0];
      for (size_t c = 1U; c < num_classes; ++c)
        {
          if (scores_row[c] > best_score)
            {
              best_score = scores_row[c];
              best_cls = c;
            }
        }

      if (best_score < conf_threshold)
        {
          continue;
        }

      /* Compute DFL 16-bin softmax integral for [l, t, r, b].  */
      float dist[4] = {0.0F, 0.0F, 0.0F, 0.0F};
      const float *dfl_row = dfl_logits.data () + a * 4U * reg_max;
      for (size_t side = 0U; side < 4U; ++side)
        {
          const float *bins = dfl_row + side * reg_max;
          float max_logit = bins[0];
          for (size_t b = 1U; b < reg_max; ++b)
            {
              max_logit = std::max (max_logit, bins[b]);
            }

          float exp_sum = 0.0F;
          float weighted_sum = 0.0F;
          for (size_t b = 0U; b < reg_max; ++b)
            {
              const float p = std::exp (bins[b] - max_logit);
              exp_sum += p;
              weighted_sum += static_cast<float> (b) * p;
            }
          dist[side] = (exp_sum > 1e-12F) ? (weighted_sum / exp_sum) : 0.0F;
        }

      const float stride = anchor_strides[a];
      const float cx = anchor_points[a].x * stride;
      const float cy = anchor_points[a].y * stride;
      const float x1 = cx - dist[0] * stride;
      const float y1 = cy - dist[1] * stride;
      const float x2 = cx + dist[2] * stride;
      const float y2 = cy + dist[3] * stride;

      detections.push_back (Detection_Box {
        core::Rectf (x1, y1, std::max (0.0F, x2 - x1),
                     std::max (0.0F, y2 - y1)),
        static_cast<int> (best_cls), best_score});
    }

  return detections;
}

core::Result<std::vector<core::Image_U8C1>>
decode_instance_masks (std::span<const float> proto_masks,
                       size_t num_protos,
                       size_t proto_h,
                       size_t proto_w,
                       std::span<const float> mask_coeffs,
                       std::span<const Detection_Box> detections,
                       size_t image_width,
                       size_t image_height,
                       float mask_threshold)
{
  const size_t num_dets = detections.size ();
  if (num_protos == 0U || proto_h == 0U || proto_w == 0U
      || image_width == 0U || image_height == 0U
      || proto_masks.size () != num_protos * proto_h * proto_w
      || mask_coeffs.size () != num_dets * num_protos)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "decode_instance_masks shape mismatch");
    }

  const float scale_x =
    static_cast<float> (proto_w) / static_cast<float> (image_width);
  const float scale_y =
    static_cast<float> (proto_h) / static_cast<float> (image_height);

  std::vector<core::Image_U8C1> masks;
  masks.reserve (num_dets);

  for (size_t d = 0U; d < num_dets; ++d)
    {
      const float *coeffs = mask_coeffs.data () + d * num_protos;
      const core::Rectf &box = detections[d].box;

      core::Image_U8C1 bin_mask (
        image_width, image_height, core::Scalar<uint8_t, 1U> (0U));

      const size_t x_start = static_cast<size_t> (
        std::clamp (std::floor (box.x), 0.0F,
                    static_cast<float> (image_width)));
      const size_t y_start = static_cast<size_t> (
        std::clamp (std::floor (box.y), 0.0F,
                    static_cast<float> (image_height)));
      const size_t x_end = static_cast<size_t> (
        std::clamp (std::ceil (box.x + box.width), 0.0F,
                    static_cast<float> (image_width)));
      const size_t y_end = static_cast<size_t> (
        std::clamp (std::ceil (box.y + box.height), 0.0F,
                    static_cast<float> (image_height)));

      for (size_t y = y_start; y < y_end; ++y)
        {
          uint8_t *mrow = bin_mask.row_ptr (y);
          const size_t py = std::min (
            static_cast<size_t> (static_cast<float> (y) * scale_y),
            proto_h - 1U);
          for (size_t x = x_start; x < x_end; ++x)
            {
              const size_t px = std::min (
                static_cast<size_t> (static_cast<float> (x) * scale_x),
                proto_w - 1U);

              float dot = 0.0F;
              const size_t proto_offset = py * proto_w + px;
              const size_t proto_stride = proto_h * proto_w;
              for (size_t k = 0U; k < num_protos; ++k)
                {
                  dot += coeffs[k]
                         * proto_masks[k * proto_stride + proto_offset];
                }
              const float prob = 1.0F / (1.0F + std::exp (-dot));
              mrow[x] = (prob >= mask_threshold) ? 255U : 0U;
            }
        }
      masks.push_back (std::move (bin_mask));
    }

  return masks;
}

float
compute_oks (const Pose_Detection &a,
             const Pose_Detection &b,
             std::span<const float> sigmas) noexcept
{
  const size_t k = std::min ({a.keypoints.size (), b.keypoints.size (),
                              sigmas.size ()});
  if (k == 0U)
    {
      return 0.0F;
    }

  const float area =
    std::max (1.0F, 0.5F * (a.box.width * a.box.height
                            + b.box.width * b.box.height));

  float sum_exp = 0.0F;
  size_t valid_count = 0U;
  for (size_t i = 0U; i < k; ++i)
    {
      if (a.keypoints[i].confidence <= 0.1F
          || b.keypoints[i].confidence <= 0.1F)
        {
          continue;
        }
      const float dx = a.keypoints[i].x - b.keypoints[i].x;
      const float dy = a.keypoints[i].y - b.keypoints[i].y;
      const float d2 = dx * dx + dy * dy;
      const float k_var = 2.0F * sigmas[i] * sigmas[i];
      const float denom = 2.0F * area * k_var + 1e-8F;
      sum_exp += std::exp (-d2 / denom);
      ++valid_count;
    }

  return (valid_count > 0U)
           ? (sum_exp / static_cast<float> (valid_count))
           : 0.0F;
}

std::vector<size_t>
oks_nms (std::span<const Pose_Detection> poses,
         std::span<const float> sigmas,
         float oks_threshold) noexcept
{
  const size_t n = poses.size ();
  std::vector<size_t> order (n);
  std::iota (order.begin (), order.end (), 0U);
  std::sort (order.begin (), order.end (),
             [&] (size_t i, size_t j)
             {
               return poses[i].score > poses[j].score;
             });

  std::vector<uint8_t> suppressed (n, 0U);
  std::vector<size_t> keep;
  keep.reserve (n);

  for (size_t i = 0U; i < n; ++i)
    {
      const size_t idx_i = order[i];
      if (suppressed[idx_i] != 0U)
        {
          continue;
        }
      keep.push_back (idx_i);
      for (size_t j = i + 1U; j < n; ++j)
        {
          const size_t idx_j = order[j];
          if (suppressed[idx_j] == 0U
              && compute_oks (poses[idx_i], poses[idx_j], sigmas)
                   > oks_threshold)
            {
              suppressed[idx_j] = 1U;
            }
        }
    }
  return keep;
}

/* 6. Kalman_Box_Tracker & Byte_Tracker implementation using Hyper Math.  */
Kalman_Box_Tracker::Kalman_Box_Tracker (const core::Rectf &initial_box,
                                        int class_id,
                                        float score,
                                        uint64_t track_id) noexcept
  : track_id_ (track_id),
    class_id_ (class_id),
    score_ (score),
    hits_ (1U),
    time_since_update_ (0U),
    state_ {},
    cov_ {}
{
  const double w = std::max (1.0, static_cast<double> (initial_box.width));
  const double h = std::max (1.0, static_cast<double> (initial_box.height));
  const double cx = static_cast<double> (initial_box.x) + 0.5 * w;
  const double cy = static_cast<double> (initial_box.y) + 0.5 * h;

  state_[0] = cx;
  state_[1] = cy;
  state_[2] = w / h;
  state_[3] = h;

  for (size_t i = 0U; i < 4U; ++i)
    {
      cov_[i * 8U + i] = 10.0;
      cov_[(i + 4U) * 8U + (i + 4U)] = 100.0;
    }
}

core::Rectf
Kalman_Box_Tracker::current_box (void) const noexcept
{
  const double cx = state_[0];
  const double cy = state_[1];
  const double a = std::max (0.05, state_[2]);
  const double h = std::max (1.0, state_[3]);
  const double w = a * h;
  return core::Rectf (static_cast<float> (cx - 0.5 * w),
                      static_cast<float> (cy - 0.5 * h),
                      static_cast<float> (w),
                      static_cast<float> (h));
}

core::Rectf
Kalman_Box_Tracker::predict (void) noexcept
{
  if (state_[3] + state_[7] <= 1.0)
    {
      state_[7] = 0.0;
    }

  /* State transition x' = F * x (constant velocity model).  */
  for (size_t i = 0U; i < 4U; ++i)
    {
      state_[i] += state_[i + 4U];
    }

  /* Covariance update P' = F * P * F^T + Q.  */
  std::array<double, 64U> next_cov = cov_;
  for (size_t i = 0U; i < 4U; ++i)
    {
      for (size_t j = 0U; j < 4U; ++j)
        {
          next_cov[i * 8U + j] =
            cov_[i * 8U + j] + cov_[(i + 4U) * 8U + j]
            + cov_[i * 8U + (j + 4U)] + cov_[(i + 4U) * 8U + (j + 4U)];
        }
      for (size_t j = 0U; j < 4U; ++j)
        {
          next_cov[i * 8U + (j + 4U)] =
            cov_[i * 8U + (j + 4U)] + cov_[(i + 4U) * 8U + (j + 4U)];
          next_cov[(j + 4U) * 8U + i] =
            cov_[(j + 4U) * 8U + i] + cov_[(j + 4U) * 8U + (i + 4U)];
        }
    }

  for (size_t i = 0U; i < 4U; ++i)
    {
      next_cov[i * 8U + i] += 1.0;
      next_cov[(i + 4U) * 8U + (i + 4U)] += 0.1;
    }

  cov_ = next_cov;
  ++time_since_update_;
  return current_box ();
}

void
Kalman_Box_Tracker::update (const core::Rectf &observed_box,
                            float score) noexcept
{
  const double w = std::max (1.0, static_cast<double> (observed_box.width));
  const double h = std::max (1.0, static_cast<double> (observed_box.height));
  const double z[4] = {
    static_cast<double> (observed_box.x) + 0.5 * w,
    static_cast<double> (observed_box.y) + 0.5 * h,
    w / h,
    h
  };

  try
    {
      /* Innovation covariance S = H * P * H^T + R (4x4 upper-left + R).  */
      hyper_math::tensor::nd_tensor<double> s_mat (
        hyper_math::tensor::shape_type {4U, 4U}, 0.0);
      const double r_diag[4] = {1.0, 1.0, 0.01, 1.0};
      for (size_t i = 0U; i < 4U; ++i)
        {
          for (size_t j = 0U; j < 4U; ++j)
            {
              s_mat (i, j) = cov_[i * 8U + j];
            }
          s_mat (i, i) += r_diag[i];
        }

      /* Solve S * (K^T) = H * P^T using Hyper Math Doolittle LU solver!  */
      auto lu_s = hyper_math::scientific::linalg::lu_factor (s_mat);

      hyper_math::tensor::nd_tensor<double> h_pt (
        hyper_math::tensor::shape_type {4U, 8U}, 0.0);
      for (size_t i = 0U; i < 4U; ++i)
        {
          for (size_t j = 0U; j < 8U; ++j)
            {
              h_pt (i, j) = cov_[j * 8U + i];
            }
        }

      /* kt_mat has shape [4, 8] = S^{-1} * H * P^T = K^T.  */
      hyper_math::tensor::nd_tensor<double> kt_mat =
        hyper_math::scientific::linalg::lu_solve (lu_s, h_pt);

      /* Innovation vector y = z - H * x.  */
      double innov[4];
      for (size_t i = 0U; i < 4U; ++i)
        {
          innov[i] = z[i] - state_[i];
        }

      /* State update x = x + K * y = x + (K^T)^T * y.  */
      for (size_t i = 0U; i < 8U; ++i)
        {
          double delta = 0.0;
          for (size_t k = 0U; k < 4U; ++k)
            {
              delta += kt_mat (k, i) * innov[k];
            }
          state_[i] += delta;
        }

      /* Covariance update P = (I - K * H) * P.  */
      std::array<double, 64U> new_cov = cov_;
      for (size_t i = 0U; i < 8U; ++i)
        {
          for (size_t j = 0U; j < 8U; ++j)
            {
              double kh_p = 0.0;
              for (size_t k = 0U; k < 4U; ++k)
                {
                  kh_p += kt_mat (k, i) * cov_[k * 8U + j];
                }
              new_cov[i * 8U + j] = cov_[i * 8U + j] - kh_p;
            }
        }
      cov_ = new_cov;
    }
  catch (...)
    {
      for (size_t i = 0U; i < 4U; ++i)
        {
          state_[i] = z[i];
        }
    }

  score_ = score;
  time_since_update_ = 0U;
  ++hits_;
}

std::vector<Tracked_Object>
Byte_Tracker::update (std::span<const Detection_Box> detections)
{
  /* 1. Predict all existing tracklets forward one frame.  */
  for (auto &trk : tracks_)
    {
      (void) trk.predict ();
    }

  /* 2. Partition detections into high-confidence and low-confidence sets.  */
  std::vector<Detection_Box> dets_high;
  std::vector<Detection_Box> dets_low;
  for (const auto &d : detections)
    {
      if (d.score >= high_thresh_)
        {
          dets_high.push_back (d);
        }
      else if (d.score >= low_thresh_)
        {
          dets_low.push_back (d);
        }
    }

  std::vector<uint8_t> trk_matched (tracks_.size (), 0U);
  std::vector<uint8_t> high_matched (dets_high.size (), 0U);

  /* 3. Stage 1 Association: Match high-score detections to tracklets.  */
  for (size_t t = 0U; t < tracks_.size (); ++t)
    {
      float best_iou = match_iou_thresh_;
      size_t best_d = dets_high.size ();
      const core::Rectf tbox = tracks_[t].current_box ();

      for (size_t d = 0U; d < dets_high.size (); ++d)
        {
          if (high_matched[d] != 0U
              || dets_high[d].class_id != tracks_[t].class_id ())
            {
              continue;
            }
          const float ov = box_iou (tbox, dets_high[d].box);
          if (ov > best_iou)
            {
              best_iou = ov;
              best_d = d;
            }
        }

      if (best_d < dets_high.size ())
        {
          tracks_[t].update (dets_high[best_d].box, dets_high[best_d].score);
          trk_matched[t] = 1U;
          high_matched[best_d] = 1U;
        }
    }

  /* 4. Stage 2 Association (ByteTrack): Match remaining active tracklets
        against low-confidence occluded detections.  */
  std::vector<uint8_t> low_matched (dets_low.size (), 0U);
  for (size_t t = 0U; t < tracks_.size (); ++t)
    {
      if (trk_matched[t] != 0U)
        {
          continue;
        }

      float best_iou = match_iou_thresh_;
      size_t best_d = dets_low.size ();
      const core::Rectf tbox = tracks_[t].current_box ();

      for (size_t d = 0U; d < dets_low.size (); ++d)
        {
          if (low_matched[d] != 0U
              || dets_low[d].class_id != tracks_[t].class_id ())
            {
              continue;
            }
          const float ov = box_iou (tbox, dets_low[d].box);
          if (ov > best_iou)
            {
              best_iou = ov;
              best_d = d;
            }
        }

      if (best_d < dets_low.size ())
        {
          tracks_[t].update (dets_low[best_d].box, dets_low[best_d].score);
          trk_matched[t] = 1U;
          low_matched[best_d] = 1U;
        }
    }

  /* 5. Spawn new tracklets for unmatched high-confidence detections.  */
  for (size_t d = 0U; d < dets_high.size (); ++d)
    {
      if (high_matched[d] == 0U)
        {
          tracks_.emplace_back (dets_high[d].box, dets_high[d].class_id,
                                dets_high[d].score, next_id_++);
        }
    }

  /* 6. Prune expired lost tracklets & collect active output tracks.  */
  std::vector<Tracked_Object> active_out;
  for (size_t t = 0U; t < tracks_.size ();)
    {
      if (tracks_[t].time_since_update () > max_time_lost_)
        {
          tracks_.erase (tracks_.begin () + static_cast<ptrdiff_t> (t));
        }
      else
        {
          if (tracks_[t].time_since_update () == 0U)
            {
              active_out.push_back (Tracked_Object {
                tracks_[t].track_id (),
                tracks_[t].current_box (),
                tracks_[t].class_id (),
                tracks_[t].score (),
                tracks_[t].hits ()});
            }
          ++t;
        }
    }

  return active_out;
}

} /* namespace surya::inline v1::detection */
