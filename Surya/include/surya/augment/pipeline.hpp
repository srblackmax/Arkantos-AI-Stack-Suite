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

#ifndef SURYA_AUGMENT_PIPELINE_HPP
#define SURYA_AUGMENT_PIPELINE_HPP

#include <array>
#include <cstdint>
#include <random>
#include <span>
#include <tuple>
#include <variant>
#include <vector>
#include <surya/core.hpp>
#include <surya/imgproc.hpp>

namespace surya::inline v1::augment
{

/**
 * @brief Labeled 2D bounding box carried through augmentation pipelines.
 */
struct Annotated_Box
{
  core::Rectf rect {};
  int32_t class_id = 0;
  float weight = 1.0F;
};

/**
 * @brief Labeled 2D keypoint with COCO-style visibility flag (0=absent,
 *        1=occluded, 2=visible).
 */
struct Annotated_Keypoint
{
  core::Point2f pt {};
  uint8_t visibility = 2U;
  int32_t id = 0;
};

/**
 * @brief Composite sample bundling an RGB image, optional binary segmentation
 *        mask, bounding boxes, and keypoints for joint transformation.
 */
struct Augment_Sample
{
  core::Image_U8C3 image;
  core::Image_U8C1 mask;
  std::vector<Annotated_Box> boxes;
  std::vector<Annotated_Keypoint> keypoints;

  Augment_Sample
  clone (void) const noexcept
  {
    Augment_Sample out;
    out.image = image.clone ();
    out.mask = mask.clone ();
    out.boxes = boxes;
    out.keypoints = keypoints;
    return out;
  }
};

/**
 * @brief Internal accumulator fusing consecutive geometric transforms into a
 *        single 3x3 homography and consecutive photometric ops into a single
 *        3x256 lookup table (LUT).
 */
class SURYA_API Augment_Fused_State
{
public:
  Augment_Fused_State (void) noexcept;

  void
  reset (void) noexcept;

  void
  append_homography (const imgproc::Homography_3x3 &h) noexcept;

  void
  append_lut (
    const std::array<std::array<uint8_t, 256U>, 3U> &op_lut) noexcept;

  bool
  has_geometric (void) const noexcept
  {
    return has_geo_;
  }

  bool
  has_photometric (void) const noexcept
  {
    return has_lut_;
  }

  const imgproc::Homography_3x3 &
  homography (void) const noexcept
  {
    return h_total_;
  }

  /**
   * @brief Flush accumulated geometric homography and photometric LUT onto
   *        SAMPLE in at most one spatial resample pass and one LUT pass.
   */
  core::Result<void>
  flush (Augment_Sample &sample, float min_box_visibility = 0.1F);

private:
  imgproc::Homography_3x3 h_total_;
  std::array<std::array<uint8_t, 256U>, 3U> lut_ {};
  bool has_geo_ = false;
  bool has_lut_ = false;
};

/* =========================================================================
   Declarative Augmentation Stage Descriptors
   ========================================================================= */

struct Horizontal_Flip
{
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Vertical_Flip
{
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Shift_Scale_Rotate
{
  float max_angle_deg = 15.0F;
  float min_scale = 0.85F;
  float max_scale = 1.15F;
  float max_shift_frac = 0.10F;
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Brightness_Contrast
{
  float max_brightness_delta = 32.0F;
  float min_contrast = 0.8F;
  float max_contrast = 1.2F;
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Gamma_Adjust
{
  float min_gamma = 0.8F;
  float max_gamma = 1.2F;
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Solarize
{
  uint8_t threshold = 128U;
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Posterize
{
  uint8_t bits = 4U;
  float prob = 0.5F;

  void
  record (Augment_Fused_State &st,
          size_t width,
          size_t height,
          std::mt19937_64 &rng) const noexcept;
};

struct Coarse_Dropout
{
  size_t max_holes = 4U;
  size_t max_hole_width = 32U;
  size_t max_hole_height = 32U;
  uint8_t fill_value = 0U;
  float prob = 0.5F;

  core::Result<void>
  apply_eager (Augment_Sample &sample, std::mt19937_64 &rng) const;
};

/**
 * @brief Compile-time template augmentation pipeline that algebraically fuses
 *        adjacent geometric and photometric stages before touching pixel data.
 */
template <typename... Stages>
class Augment_Pipeline
{
public:
  explicit constexpr Augment_Pipeline (Stages... stages) noexcept
    : stages_ (std::move (stages)...)
  {
  }

  /**
   * @brief Apply the fused pipeline to a single Augment_Sample in-place.
   */
  core::Result<void>
  apply (Augment_Sample &sample,
         std::mt19937_64 &rng,
         float min_box_visibility = 0.1F) const
  {
    if (sample.image.empty ())
      {
        return core::make_error (core::Error_Code::Empty_Input,
                                 "Augment_Pipeline::apply: empty image");
      }
    Augment_Fused_State state;
    core::Result<void> status {};
    std::apply (
      [&] (const auto &...stage)
      {
        (void) ((status.has_value ()
                   ? (dispatch_stage (stage, state, sample, rng,
                                      min_box_visibility, status),
                      true)
                   : false)
                && ...);
      },
      stages_);
    if (!status.has_value ())
      {
        return status;
      }
    return state.flush (sample, min_box_visibility);
  }

  /**
   * @brief Apply the fused pipeline across a batch of samples in parallel
   *        using deterministic per-sample seeds derived from BASE_SEED.
   */
  core::Result<void>
  apply_batch (std::span<Augment_Sample> batch,
               uint64_t base_seed = 42ULL,
               float min_box_visibility = 0.1F) const
  {
    for (size_t i = 0U; i < batch.size (); ++i)
      {
        if (batch[i].image.empty ())
          {
            return core::make_error (
              core::Error_Code::Empty_Input,
              "Augment_Pipeline::apply_batch: sample image is empty");
          }
      }
    core::parallel_for_1d (
      0U, batch.size (), 1U,
      [&] (size_t b0, size_t b1)
      {
        for (size_t i = b0; i < b1; ++i)
          {
            std::mt19937_64 rng (base_seed + i * 0x9E3779B97F4A7C15ULL);
            auto r = apply (batch[i], rng, min_box_visibility);
            (void) r;
          }
      });
    return {};
  }

private:
  template <typename StageT>
  static void
  dispatch_stage (const StageT &stage,
                  Augment_Fused_State &state,
                  Augment_Sample &sample,
                  std::mt19937_64 &rng,
                  float min_vis,
                  core::Result<void> &status)
  {
    if constexpr (requires {
                    stage.record (state, sample.image.width (),
                                  sample.image.height (), rng);
                  })
      {
        stage.record (state, sample.image.width (), sample.image.height (),
                      rng);
      }
    else
      {
        status = state.flush (sample, min_vis);
        if (status.has_value ())
          {
            status = stage.apply_eager (sample, rng);
          }
      }
  }

  std::tuple<Stages...> stages_;
};

/**
 * @brief Runtime-configurable augmentation DAG / instruction tape sharing the
 *        exact same homography and LUT fusion engine as Augment_Pipeline.
 */
class SURYA_API Augment_Graph
{
public:
  using Node_Variant =
    std::variant<Horizontal_Flip,
                 Vertical_Flip,
                 Shift_Scale_Rotate,
                 Brightness_Contrast,
                 Gamma_Adjust,
                 Solarize,
                 Posterize,
                 Coarse_Dropout>;

  Augment_Graph (void) = default;

  void
  add_stage (Node_Variant stage)
  {
    nodes_.push_back (std::move (stage));
  }

  size_t
  num_stages (void) const noexcept
  {
    return nodes_.size ();
  }

  core::Result<void>
  apply (Augment_Sample &sample,
         std::mt19937_64 &rng,
         float min_box_visibility = 0.1F) const;

  core::Result<void>
  apply_batch (std::span<Augment_Sample> batch,
               uint64_t base_seed = 42ULL,
               float min_box_visibility = 0.1F) const;

private:
  std::vector<Node_Variant> nodes_;
};

/* =========================================================================
   Batch-Level Multi-Image Augmentations (MixUp, CutMix, Mosaic4)
   ========================================================================= */

/**
 * @brief Blend two Augment_Sample instances with weight ALPHA in [0, 1]:
 *        out.image = alpha * s1.image + (1 - alpha) * s2.image, merging boxes
 *        with corresponding confidence weights.
 */
SURYA_API core::Result<Augment_Sample>
mixup (const Augment_Sample &s1, const Augment_Sample &s2, float alpha);

/**
 * @brief Paste rectangular region Patch_Rect from S2 into a clone of S1 and
 *        synchronize bounding boxes from both samples.
 */
SURYA_API core::Result<Augment_Sample>
cutmix (const Augment_Sample &s1,
        const Augment_Sample &s2,
        core::Rectu patch_rect);

/**
 * @brief Assemble 4 Augment_Sample instances into a 2x2 Mosaic canvas of size
 *        OUT_WIDTH x OUT_HEIGHT centered at divider (DIV_X, DIV_Y), shifting
 *        and clipping all bounding boxes and keypoints.
 */
SURYA_API core::Result<Augment_Sample>
mosaic4 (const std::array<const Augment_Sample *, 4U> &samples,
         size_t out_width,
         size_t out_height,
         size_t div_x,
         size_t div_y);

} /* namespace surya::inline v1::augment */

#endif /* SURYA_AUGMENT_PIPELINE_HPP */
