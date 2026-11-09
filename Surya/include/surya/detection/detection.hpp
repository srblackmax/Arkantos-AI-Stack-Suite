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

#ifndef SURYA_DETECTION_DETECTION_HPP
#define SURYA_DETECTION_DETECTION_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include <surya/core/error.hpp>
#include <surya/core/image.hpp>
#include <surya/core/types.hpp>

namespace surya::inline v1::detection
{

/**
 * @brief Reversible aspect-ratio preserving letterbox coordinate transform.
 * @note Thread-safety: Value type; safe to share across threads.
 */
struct Letterbox_Transform
{
  size_t src_width = 0U;
  size_t src_height = 0U;
  size_t dst_width = 0U;
  size_t dst_height = 0U;
  float scale = 1.0F;
  float pad_left = 0.0F;
  float pad_top = 0.0F;

  /**
   * @brief Map a bounding box from letterboxed canvas coordinates back to
   *        original source image coordinates.
   * @pre None.
   * @post Returns unletterboxed Rectf.
   * @returns Mapped Rectf in source coordinate frame.
   * @note Thread-safety: Pure constexpr function.
   */
  constexpr core::Rectf
  unletterbox_box (const core::Rectf &box) const noexcept
  {
    const float inv_s = (scale > 1e-8F) ? (1.0F / scale) : 1.0F;
    const float x = (box.x - pad_left) * inv_s;
    const float y = (box.y - pad_top) * inv_s;
    const float w = box.width * inv_s;
    const float h = box.height * inv_s;
    return core::Rectf (x, y, w, h);
  }

  /**
   * @brief Map a 2D point from letterboxed canvas coordinates back to
   *        original source image coordinates.
   * @pre None.
   * @post Returns unletterboxed Point2f.
   * @returns Mapped Point2f in source coordinate frame.
   * @note Thread-safety: Pure constexpr function.
   */
  constexpr core::Point2f
  unletterbox_point (const core::Point2f &pt) const noexcept
  {
    const float inv_s = (scale > 1e-8F) ? (1.0F / scale) : 1.0F;
    return core::Point2f ((pt.x - pad_left) * inv_s,
                          (pt.y - pad_top) * inv_s);
  }
};

struct Letterbox_Result
{
  core::Image_U8C3 image;
  Letterbox_Transform transform;
};

/**
 * @brief Resize SRC isotropically with centered padding into a DST_WIDTH x
 *        DST_HEIGHT canvas.
 * @pre !src.empty() && dst_width > 0 && dst_height > 0.
 * @post Returns Letterbox_Result with padded image and inverse transform.
 * @returns std::expected<Letterbox_Result, SuryaError>.
 * @note Thread-safety: Safe to call concurrently on distinct views.
 */
SURYA_API core::Result<Letterbox_Result>
letterbox_image (core::Const_View_U8C3 src,
                 size_t dst_width,
                 size_t dst_height,
                 core::Scalar<uint8_t, 3U> pad_color =
                   core::Scalar<uint8_t, 3U> (114U, 114U, 114U));

struct Detection_Box
{
  core::Rectf box;
  int class_id = 0;
  float score = 0.0F;
};

struct Pose_Keypoint
{
  float x = 0.0F;
  float y = 0.0F;
  float confidence = 0.0F;
};

struct Pose_Detection
{
  core::Rectf box;
  float score = 0.0F;
  std::vector<Pose_Keypoint> keypoints;
};

/**
 * @brief Compute standard Intersection-over-Union (IoU) of two boxes.
 * @pre None.
 * @post Returns value in [0.0, 1.0].
 * @returns IoU overlap score.
 * @note Thread-safety: Pure function.
 */
SURYA_API float
box_iou (const core::Rectf &a, const core::Rectf &b) noexcept;

/**
 * @brief Compute Generalized IoU (GIoU) of two boxes.
 * @pre None.
 * @post Returns value in [-1.0, 1.0].
 * @returns GIoU score.
 * @note Thread-safety: Pure function.
 */
SURYA_API float
box_giou (const core::Rectf &a, const core::Rectf &b) noexcept;

/**
 * @brief Compute Distance IoU (DIoU) of two boxes.
 * @pre None.
 * @post Returns value in [-1.0, 1.0].
 * @returns DIoU score.
 * @note Thread-safety: Pure function.
 */
SURYA_API float
box_diou (const core::Rectf &a, const core::Rectf &b) noexcept;

/**
 * @brief Compute Complete IoU (CIoU) of two boxes.
 * @pre None.
 * @post Returns value in [-1.0, 1.0].
 * @returns CIoU score.
 * @note Thread-safety: Pure function.
 */
SURYA_API float
box_ciou (const core::Rectf &a, const core::Rectf &b) noexcept;

/**
 * @brief Class-agnostic greedy Non-Maximum Suppression.
 * @pre None.
 * @post Returns kept indices sorted by descending score.
 * @returns Vector of kept box indices.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API std::vector<size_t>
nms (std::span<const Detection_Box> boxes,
     float iou_threshold) noexcept;

/**
 * @brief Class-aware batched Non-Maximum Suppression using coordinate offsets.
 * @pre None.
 * @post Returns kept indices sorted by descending score.
 * @returns Vector of kept box indices.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API std::vector<size_t>
batched_nms (std::span<const Detection_Box> boxes,
             float iou_threshold) noexcept;

enum class Soft_NMS_Method : uint8_t
{
  Linear = 0U,
  Gaussian = 1U
};

/**
 * @brief Soft-NMS with continuous score decay (Linear or Gaussian).
 * @pre sigma > 0.
 * @post Returns detections whose decayed scores remain >= score_threshold.
 * @returns Vector of retained Detection_Box instances.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API std::vector<Detection_Box>
soft_nms (std::span<const Detection_Box> boxes,
          float iou_threshold,
          float sigma = 0.5F,
          float score_threshold = 0.001F,
          Soft_NMS_Method method = Soft_NMS_Method::Gaussian);

/**
 * @brief Decode anchor-free Distribution Focal Loss (DFL) 16-bin logits.
 * @pre dfl_logits.size() == anchor_points.size() * 4 * reg_max.
 * @post Returns filtered Detection_Box vector.
 * @returns std::expected<std::vector<Detection_Box>, SuryaError>.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API core::Result<std::vector<Detection_Box>>
decode_dfl_detections (std::span<const float> dfl_logits,
                       std::span<const float> cls_scores,
                       std::span<const core::Point2f> anchor_points,
                       std::span<const float> anchor_strides,
                       size_t num_classes,
                       size_t reg_max = 16U,
                       float conf_threshold = 0.25F);

/**
 * @brief Assemble binary instance masks from prototype masks and coefficients.
 * @pre proto_masks.size() == num_protos * proto_h * proto_w.
 * @post Returns one binary Image_U8C1 per detection.
 * @returns std::expected<std::vector<core::Image_U8C1>, SuryaError>.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API core::Result<std::vector<core::Image_U8C1>>
decode_instance_masks (std::span<const float> proto_masks,
                       size_t num_protos,
                       size_t proto_h,
                       size_t proto_w,
                       std::span<const float> mask_coeffs,
                       std::span<const Detection_Box> detections,
                       size_t image_width,
                       size_t image_height,
                       float mask_threshold = 0.5F);

/**
 * @brief Compute Object Keypoint Similarity (OKS) between two pose detections.
 * @pre None.
 * @post Returns similarity in [0.0, 1.0].
 * @returns OKS similarity score.
 * @note Thread-safety: Pure function.
 */
SURYA_API float
compute_oks (const Pose_Detection &a,
             const Pose_Detection &b,
             std::span<const float> sigmas) noexcept;

/**
 * @brief Perform Non-Maximum Suppression over pose detections using OKS.
 * @pre None.
 * @post Returns kept pose indices sorted by descending score.
 * @returns Vector of kept pose indices.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API std::vector<size_t>
oks_nms (std::span<const Pose_Detection> poses,
         std::span<const float> sigmas,
         float oks_threshold = 0.5F) noexcept;

/**
 * @brief 8-DOF constant-velocity Kalman bounding-box tracklet.
 *        Uses Hyper Math Doolittle LU solver internally without exposing
 *        third-party types in the public header.
 * @note Thread-safety: Distinct instances may be updated concurrently.
 */
class SURYA_API Kalman_Box_Tracker
{
public:
  Kalman_Box_Tracker (const core::Rectf &initial_box,
                      int class_id,
                      float score,
                      uint64_t track_id) noexcept;

  core::Rectf
  predict (void) noexcept;

  void
  update (const core::Rectf &observed_box, float score) noexcept;

  core::Rectf
  current_box (void) const noexcept;

  uint64_t
  track_id (void) const noexcept
  {
    return track_id_;
  }

  int
  class_id (void) const noexcept
  {
    return class_id_;
  }

  float
  score (void) const noexcept
  {
    return score_;
  }

  size_t
  hits (void) const noexcept
  {
    return hits_;
  }

  size_t
  time_since_update (void) const noexcept
  {
    return time_since_update_;
  }

private:
  uint64_t track_id_ = 0ULL;
  int class_id_ = 0;
  float score_ = 0.0F;
  size_t hits_ = 1U;
  size_t time_since_update_ = 0U;
  std::array<double, 8U> state_ {};
  std::array<double, 64U> cov_ {};
};

struct Tracked_Object
{
  uint64_t track_id = 0ULL;
  core::Rectf box;
  int class_id = 0;
  float score = 0.0F;
  size_t hits = 0U;
};

/**
 * @brief Two-stage ByteTrack multi-object tracker associating high-score
 *        and occluded low-score detections across frames.
 * @note Thread-safety: Per-stream stateful tracker; distinct instances are
 *       independent and thread-safe.
 */
class SURYA_API Byte_Tracker
{
public:
  explicit Byte_Tracker (float high_thresh = 0.5F,
                         float low_thresh = 0.1F,
                         float match_iou_thresh = 0.3F,
                         size_t max_time_lost = 30U) noexcept
    : high_thresh_ (high_thresh),
      low_thresh_ (low_thresh),
      match_iou_thresh_ (match_iou_thresh),
      max_time_lost_ (max_time_lost)
  {
  }

  std::vector<Tracked_Object>
  update (std::span<const Detection_Box> detections);

  size_t
  active_track_count (void) const noexcept
  {
    return tracks_.size ();
  }

private:
  float high_thresh_ = 0.5F;
  float low_thresh_ = 0.1F;
  float match_iou_thresh_ = 0.3F;
  size_t max_time_lost_ = 30U;
  uint64_t next_id_ = 1ULL;
  std::vector<Kalman_Box_Tracker> tracks_;
};

} /* namespace surya::inline v1::detection */

#endif /* SURYA_DETECTION_DETECTION_HPP */
