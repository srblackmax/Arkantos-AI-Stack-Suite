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

#ifndef SURYA_CLASSICAL_CLASSICAL_HPP
#define SURYA_CLASSICAL_CLASSICAL_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <surya/core/error.hpp>
#include <surya/core/image.hpp>
#include <surya/core/tensor.hpp>
#include <surya/core/types.hpp>

namespace surya::inline v1::classical
{

/* 1. Keypoint Detection & 256-bit ORB Binary Descriptors.  */
struct Keypoint
{
  float x = 0.0F;
  float y = 0.0F;
  float response = 0.0F;
  float angle_rad = 0.0F;
};

struct Orb_Descriptor
{
  std::array<uint64_t, 4U> words = {0ULL, 0ULL, 0ULL, 0ULL};

  constexpr uint32_t
  hamming_distance (const Orb_Descriptor &other) const noexcept
  {
    return static_cast<uint32_t> (
      std::popcount (words[0] ^ other.words[0])
      + std::popcount (words[1] ^ other.words[1])
      + std::popcount (words[2] ^ other.words[2])
      + std::popcount (words[3] ^ other.words[3]));
  }
};

struct Orb_Feature
{
  Keypoint kp;
  Orb_Descriptor desc;
};

struct Descriptor_Match
{
  size_t query_idx = 0U;
  size_t train_idx = 0U;
  uint32_t distance = 0U;
};

/**
 * @brief Detect FAST-9 corner keypoints using a 16-pixel Bresenham ring.
 * @pre src.width() >= 7 && src.height() >= 7.
 * @post Returns up to MAX_KEYPOINTS sorted by descending corner score.
 * @returns std::expected<std::vector<Keypoint>, SuryaError>.
 * @note Thread-safety: Safe to call concurrently on distinct views.
 */
SURYA_API core::Result<std::vector<Keypoint>>
detect_fast9_keypoints (core::Const_View_U8C1 src,
                        uint8_t threshold = 20U,
                        bool nonmax_suppression = true,
                        size_t max_keypoints = 2000U);

/**
 * @brief Extract oriented FAST keypoints and 256-bit steered BRIEF (ORB)
 *        descriptors.
 * @pre src.width() >= 32 && src.height() >= 32.
 * @post Returns up to MAX_FEATURES Orb_Feature descriptors.
 * @returns std::expected<std::vector<Orb_Feature>, SuryaError>.
 * @note Thread-safety: Safe to call concurrently on distinct views.
 */
SURYA_API core::Result<std::vector<Orb_Feature>>
extract_orb_features (core::Const_View_U8C1 src,
                      uint8_t fast_threshold = 20U,
                      size_t max_features = 1000U);

/**
 * @brief Match 256-bit ORB descriptors using hardware POPCNT and Lowe's
 *        second-nearest ratio test.
 * @pre None.
 * @post Returns matches with distance <= max_distance.
 * @returns Vector of Descriptor_Match pairs.
 * @note Thread-safety: Pure function; safe to call concurrently.
 */
SURYA_API std::vector<Descriptor_Match>
match_descriptors_hamming (std::span<const Orb_Feature> query,
                           std::span<const Orb_Feature> train,
                           float lowe_ratio = 0.75F,
                           uint32_t max_distance = 64U) noexcept;

/* 2. Histogram of Oriented Gradients (HOG) Feature Extractor.  */
class SURYA_API HOG_Extractor
{
public:
  explicit HOG_Extractor (size_t cell_size = 8U,
                          size_t block_cells = 2U,
                          size_t num_bins = 9U) noexcept
    : cell_size_ (cell_size),
      block_cells_ (block_cells),
      num_bins_ (num_bins)
  {
  }

  size_t
  descriptor_length (size_t width, size_t height) const noexcept;

  core::Result<std::vector<double>>
  compute (core::Const_View_U8C1 src) const;

private:
  size_t cell_size_ = 8U;
  size_t block_cells_ = 2U;
  size_t num_bins_ = 9U;
};

/* 3. RANSAC Projective Homography Estimation via Hyper Math eigh (A^T A).  */
struct Homography_Result
{
  std::array<double, 9U> h_matrix = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
  };
  std::vector<uint8_t> inlier_mask;
  size_t num_inliers = 0U;

  core::Point2f
  project_point (const core::Point2f &pt) const noexcept
  {
    const double x = static_cast<double> (pt.x);
    const double y = static_cast<double> (pt.y);
    const double u = h_matrix[0] * x + h_matrix[1] * y + h_matrix[2];
    const double v = h_matrix[3] * x + h_matrix[4] * y + h_matrix[5];
    const double w = h_matrix[6] * x + h_matrix[7] * y + h_matrix[8];
    const double inv_w = (std::abs (w) > 1e-12) ? (1.0 / w) : 1.0;
    return core::Point2f (static_cast<float> (u * inv_w),
                          static_cast<float> (v * inv_w));
  }
};

/**
 * @brief Estimate a 3x3 projective homography from 2D correspondences using
 *        RANSAC and normalized Direct Linear Transform (DLT).
 * @pre src_pts.size() >= 4 && src_pts.size() == dst_pts.size().
 * @post Returns refined Homography_Result and inlier mask.
 * @returns std::expected<Homography_Result, SuryaError>.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API core::Result<Homography_Result>
estimate_homography_ransac (std::span<const core::Point2f> src_pts,
                            std::span<const core::Point2f> dst_pts,
                            float reproj_threshold = 3.0F,
                            size_t max_iterations = 500U,
                            uint64_t seed = 42ULL);

/* 4. Template Matching & Sparse Lucas-Kanade Optical Flow.  */
enum class Template_Match_Mode : uint8_t
{
  SAD = 0U,
  SSD = 1U,
  NCC = 2U
};

struct Template_Match_Result
{
  core::Image_F32C1 response_map;
  size_t best_x = 0U;
  size_t best_y = 0U;
  float best_score = 0.0F;
};

SURYA_API core::Result<Template_Match_Result>
match_template (core::Const_View_U8C1 image,
                core::Const_View_U8C1 templ,
                Template_Match_Mode mode = Template_Match_Mode::NCC);

struct Flow_Track_Result
{
  std::vector<core::Point2f> next_points;
  std::vector<uint8_t> status;
  std::vector<float> error;
};

SURYA_API core::Result<Flow_Track_Result>
calc_optical_flow_lk (core::Const_View_U8C1 prev_img,
                      core::Const_View_U8C1 next_img,
                      std::span<const core::Point2f> prev_pts,
                      size_t win_radius = 5U,
                      size_t max_iterations = 20U,
                      float epsilon = 0.01F);

/* 5. Classical ML Vision Pipelines Powered by Bellows (Opaque ABI).  */
class SURYA_API HOG_SVM_Classifier
{
public:
  explicit HOG_SVM_Classifier (HOG_Extractor hog = HOG_Extractor (),
                               double penalty = 0.01,
                               double learning_rate = 0.08,
                               size_t max_epochs = 300U);

  ~HOG_SVM_Classifier (void) noexcept;

  core::Result<void>
  fit (std::span<const core::Const_View_U8C1> patches,
       std::span<const int> labels);

  core::Result<int>
  predict (core::Const_View_U8C1 patch) const;

  core::Result<std::vector<double>>
  predict_probabilities (core::Const_View_U8C1 patch) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

class SURYA_API Visual_Vocabulary_BoVW
{
public:
  explicit Visual_Vocabulary_BoVW (size_t num_clusters = 8U,
                                   size_t max_iterations = 100U,
                                   uint64_t seed = 42ULL);

  ~Visual_Vocabulary_BoVW (void) noexcept;

  core::Result<void>
  fit (const core::SuryaTensor &descriptors);

  core::Result<std::vector<double>>
  encode_histogram (const core::SuryaTensor &image_descriptors) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

class SURYA_API Patch_Forest_Classifier
{
public:
  explicit Patch_Forest_Classifier (HOG_Extractor hog = HOG_Extractor (),
                                    size_t tree_count = 25U,
                                    size_t max_depth = 6U,
                                    uint64_t seed = 42ULL);

  ~Patch_Forest_Classifier (void) noexcept;

  core::Result<void>
  fit (std::span<const core::Const_View_U8C1> patches,
       std::span<const int> labels);

  core::Result<int>
  predict (core::Const_View_U8C1 patch) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} /* namespace surya::inline v1::classical */

#endif /* SURYA_CLASSICAL_CLASSICAL_HPP */
