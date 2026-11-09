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

#ifndef YANTRA_REGISTRY_HPP
#define YANTRA_REGISTRY_HPP

#include <yantra/error.hpp>
#include <yantra/observability.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace yantra::inline v1
{

/**
 * @brief Deployment lifecycle stages for a registered model version.
 */
enum class ModelStage : uint8_t
{
  Staging = 0,
  Production = 1,
  Archived = 2
};

/**
 * @brief Provenance and lineage metadata for a registered model version.
 */
struct ModelLineage
{
  std::string run_id;
  std::string dataset_hash;
  std::string code_commit_hash;
  std::vector<std::string> parent_model_versions;
};

/**
 * @brief Immutable manifest record describing a registered model version.
 */
struct ModelManifest
{
  std::string model_name;
  uint64_t version = 0U;
  std::string semver_tag;
  ModelStage stage = ModelStage::Staging;
  std::string blake3_hex;
  std::string sha256_hex;
  std::size_t artifact_size_bytes = 0U;
  uint32_t tensor_count = 0U;
  ModelLineage lineage;
  uint64_t registered_at_ns = 0U;
  std::filesystem::path artifact_path;
};

/**
 * @brief Audit trail entry recording an explicit stage transition.
 */
struct StageTransitionAudit
{
  std::string model_name;
  uint64_t version = 0U;
  ModelStage from_stage = ModelStage::Staging;
  ModelStage to_stage = ModelStage::Staging;
  std::string justification;
  uint64_t timestamp_ns = 0U;
};

/**
 * @brief Result of loading and cryptographically verifying a model artifact.
 */
struct VerifiedArtifactLoad
{
  std::string sha256_hex;
  std::string blake3_hex;
  std::size_t bytes_verified = 0U;
  uint32_t tensor_count = 0U;
};

/**
 * @brief Transactional model registry with content-addressed storage,
 * optimistic concurrency version checks, explicit stage transitions, and
 * lineage graph queries.
 * @note Thread-safety: Safe for concurrent registrations, lookups, stage
 * transitions, and lineage queries across threads.
 */
class ModelRegistry
{
public:
  ModelRegistry () noexcept;
  ModelRegistry (const ModelRegistry &) = delete;
  auto operator= (const ModelRegistry &) -> ModelRegistry & = delete;
  ModelRegistry (ModelRegistry &&other) noexcept;
  auto operator= (ModelRegistry &&other) noexcept -> ModelRegistry &;
  ~ModelRegistry () noexcept;

  /**
   * @brief Opens or initializes a ModelRegistry backed by the given directory.
   * @param registry_dir Directory for content-addressed model artifacts.
   * @returns Result<ModelRegistry> or RegistryErrorCode::IoFailure.
   */
  [[nodiscard]] static auto
  open (const std::filesystem::path &registry_dir) noexcept
      -> Result<ModelRegistry>;

  /**
   * @brief Transactionally hashes, stores, and indexes an immutable model
   * version.
   * @param model_name Logical model family name (e.g., "basalt-resnet18").
   * @param semver_tag Semantic version tag (e.g., "v1.2.0").
   * @param artifact_bytes Raw model checkpoint payload.
   * @param lineage Provenance metadata (run ID, dataset hash, commit hash).
   * @param tensor_count Number of tensors in checkpoint header (default 1).
   * @param compute_sha256 If true, computes SHA-256 alongside BLAKE3.
   * @returns Result<ModelManifest> with assigned monotonic version number.
   */
  [[nodiscard]] auto
  register_version (std::string_view model_name, std::string_view semver_tag,
                    std::span<const std::byte> artifact_bytes,
                    const ModelLineage &lineage, uint32_t tensor_count = 1U,
                    bool compute_sha256 = true) noexcept
      -> Result<ModelManifest>;

  /**
   * @brief Transactionally hashes, stores, and indexes a model file on disk.
   * @param model_name Logical model family name.
   * @param semver_tag Semantic version tag.
   * @param file_path Path to the source model artifact file.
   * @param lineage Provenance metadata.
   * @param tensor_count Number of tensors in checkpoint header (default 1).
   * @param compute_sha256 If true, computes SHA-256 alongside BLAKE3.
   * @returns Result<ModelManifest> or RegistryErrorCode::ArtifactNotFound.
   */
  [[nodiscard]] auto
  register_file (std::string_view model_name, std::string_view semver_tag,
                 const std::filesystem::path &file_path,
                 const ModelLineage &lineage, uint32_t tensor_count = 1U,
                 bool compute_sha256 = true) noexcept
      -> Result<ModelManifest>;

  /**
   * @brief Transactionally registers a model version with an explicit
   * optimistic concurrency check on the expected prior version number.
   * @param model_name Logical model name.
   * @param expected_prev_version Expected current latest version (0 if none).
   * @param semver_tag Semantic version tag.
   * @param artifact_bytes Raw artifact payload.
   * @param lineage Provenance metadata.
   * @returns Result<ModelManifest> or RegistryErrorCode::VersionConflict.
   */
  [[nodiscard]] auto register_version_cas (
      std::string_view model_name, uint64_t expected_prev_version,
      std::string_view semver_tag, std::span<const std::byte> artifact_bytes,
      const ModelLineage &lineage) noexcept -> Result<ModelManifest>;

  /**
   * @brief Indexes a pre-hashed model manifest directly into the catalog
   * (used during fast catalog population and recovery).
   * @param manifest Pre-populated ModelManifest record.
   * @returns Result<ModelManifest> or RegistryErrorCode::LineageCycleDetected.
   */
  [[nodiscard]] auto
  index_manifest (ModelManifest manifest) noexcept -> Result<ModelManifest>;

  /**
   * @brief Transitions a model version to a target deployment stage with a
   * mandatory audit justification string.
   * @param model_name Model name.
   * @param version Version number to transition.
   * @param target_stage Target ModelStage (Staging, Production, Archived).
   * @param justification Non-empty audit justification sentence.
   * @returns Result<void> or RegistryErrorCode::InvalidStageTransition.
   */
  [[nodiscard]] auto
  transition_stage (std::string_view model_name, uint64_t version,
                    ModelStage target_stage,
                    std::string_view justification) noexcept -> Result<void>;

  /**
   * @brief Resolves a "model_name:semver_tag" string (e.g.,
   * "basalt-resnet18:v1.2.0") to its ModelManifest in O(1) time.
   * @param name_and_tag Composite identifier "name:tag".
   * @returns Result<ModelManifest> or RegistryErrorCode::ArtifactNotFound.
   */
  [[nodiscard]] auto
  lookup (std::string_view name_and_tag) const noexcept
      -> Result<ModelManifest>;

  /**
   * @brief Retrieves a specific model version by name and version number.
   * @param model_name Model name.
   * @param version 1-based version number.
   * @returns Result<ModelManifest> or RegistryErrorCode::ArtifactNotFound.
   */
  [[nodiscard]] auto
  get_version (std::string_view model_name,
               uint64_t version) const noexcept -> Result<ModelManifest>;

  /**
   * @brief Lists all registered versions of a model in ascending order.
   * @param model_name Model name.
   * @returns Result<std::vector<ModelManifest>>.
   */
  [[nodiscard]] auto
  list_versions (std::string_view model_name) const noexcept
      -> Result<std::vector<ModelManifest>>;

  /**
   * @brief Queries the lineage graph starting from (model_name, version),
   * returning the version and all transitive ancestor manifests.
   * @param model_name Model name.
   * @param version Starting version number.
   * @returns Result<std::vector<ModelManifest>> in topological lineage order.
   */
  [[nodiscard]] auto
  query_lineage (std::string_view model_name,
                 uint64_t version) const noexcept
      -> Result<std::vector<ModelManifest>>;

  /**
   * @brief Retrieves the immutable stage transition audit trail for a model.
   * @param model_name Model name.
   * @returns Result<std::vector<StageTransitionAudit>>.
   */
  [[nodiscard]] auto
  audit_trail (std::string_view model_name) const noexcept
      -> Result<std::vector<StageTransitionAudit>>;

  /**
   * @brief Maps a registered model artifact via mmap(2) from warm OS page
   * cache, verifies its full SHA-256 digest against the manifest, and parses
   * the checkpoint tensor count.
   * @param model_name Model name.
   * @param version Version number.
   * @returns Result<VerifiedArtifactLoad> or RegistryErrorCode::HashMismatch.
   */
  [[nodiscard]] auto
  load_artifact_warm_mmap (std::string_view model_name,
                           uint64_t version) const noexcept
      -> Result<VerifiedArtifactLoad>;

  /**
   * @brief Evicts OS page-cache pages via posix_fadvise(POSIX_FADV_DONTNEED),
   * reads the model artifact via read(2) from disk, and verifies its full
   * SHA-256 digest against the manifest.
   * @param model_name Model name.
   * @param version Version number.
   * @returns Result<VerifiedArtifactLoad> or RegistryErrorCode::HashMismatch.
   */
  [[nodiscard]] auto
  load_artifact_cold_nvme (std::string_view model_name,
                           uint64_t version) const noexcept
      -> Result<VerifiedArtifactLoad>;

  /**
   * @brief Queries the total number of model versions registered.
   */
  [[nodiscard]] auto total_versions () const noexcept -> std::size_t;

  /**
   * @brief Registers and verifies all standard AI stack GGUF models present in
   * models_dir (including qwen3-coder-next-80b-q4ks, gemma-4-31b-it-q4km,
   * qwen3-30b-a3b-q4km, qwen2.5-7b-instruct-q4km,
   * qwen2.5-1.5b-instruct-q4km, and qwen3-embedding-0.6b-q8_0).
   */
  [[nodiscard]] auto ensure_standard_models (
      const std::filesystem::path &models_dir = {}) noexcept -> Result<void>;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

namespace registry
{
using yantra::v1::ModelLineage;
using yantra::v1::ModelManifest;
using yantra::v1::ModelRegistry;
using yantra::v1::ModelStage;
using yantra::v1::StageTransitionAudit;
using yantra::v1::VerifiedArtifactLoad;
} /* namespace registry */

} /* namespace yantra::inline v1 */

#endif /* YANTRA_REGISTRY_HPP */
