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

#ifndef YANTRA_ERROR_HPP
#define YANTRA_ERROR_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <variant>

namespace yantra::inline v1
{

/**
 * @brief Enumerates failure modes originating from experiment tracking & WAL.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class TrackingErrorCode : uint8_t
{
  WalWriteFailure = 0,
  CorruptedSegment = 1,
  QueryParseError = 2,
  ImmutableParameterViolation = 3,
  RunNotFound = 4,
  CompactionFailed = 5
};

/**
 * @brief Enumerates failure modes originating from the model registry.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class RegistryErrorCode : uint8_t
{
  ArtifactNotFound = 0,
  InvalidStageTransition = 1,
  LineageCycleDetected = 2,
  VersionConflict = 3,
  HashMismatch = 4,
  IoFailure = 5
};

/**
 * @brief Enumerates failure modes originating from DAG workflow orchestration.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class DagErrorCode : uint8_t
{
  CycleDetected = 0,
  ResourceDeadlock = 1,
  TaskTimeout = 2,
  MissingDependency = 3,
  TaskExecutionFailed = 4,
  InvalidDefinition = 5
};

/**
 * @brief Enumerates failure modes originating from the vector index.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class IndexErrorCode : uint8_t
{
  DimensionMismatch = 0,
  IndexNotBuilt = 1,
  PersistenceCorruption = 2,
  GpuOutOfMemory = 3,
  InvalidParameter = 4,
  CapacityExceeded = 5
};

/**
 * @brief Enumerates failure modes originating from bias & fairness auditing.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class FairnessErrorCode : uint8_t
{
  MissingSensitiveAttribute = 0,
  InsufficientSamplesPerGroup = 1,
  ConstraintUnsatisfiable = 2,
  InvalidThresholdBounds = 3
};

/**
 * @brief Describes an experiment tracking error payload within yantra::Error.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct TrackingError
{
  TrackingErrorCode code = TrackingErrorCode::WalWriteFailure;
  std::string run_id;
};

/**
 * @brief Describes a model registry error payload within yantra::Error.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct RegistryError
{
  RegistryErrorCode code = RegistryErrorCode::ArtifactNotFound;
  std::string model_name;
  uint64_t version = 0U;
};

/**
 * @brief Describes a workflow DAG error payload within yantra::Error.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct DagError
{
  DagErrorCode code = DagErrorCode::CycleDetected;
  std::string task_id;
};

/**
 * @brief Describes a vector index error payload within yantra::Error.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct IndexError
{
  IndexErrorCode code = IndexErrorCode::DimensionMismatch;
  std::size_t expected_dim = 0U;
  std::size_t actual_dim = 0U;
};

/**
 * @brief Describes a fairness or drift audit error payload within Error.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct FairnessError
{
  FairnessErrorCode code = FairnessErrorCode::MissingSensitiveAttribute;
  uint32_t group_id = 0U;
};

/**
 * @brief Discriminated union holding the specific domain of a yantra::Error.
 */
using ErrorVariant = std::variant<TrackingError, RegistryError, DagError,
                                  IndexError, FairnessError>;

/**
 * @brief Represents a structured, non-throwing error across all Yantra APIs.
 * @pre Error message must be a complete sentence without trailing period.
 * @post Holds a valid ErrorVariant and logs the diagnostic on construction.
 * @note Thread-safety: Const methods are safe for concurrent reads.
 */
class Error
{
public:
  /**
   * @brief Constructs a default tracking error.
   * @pre None.
   * @post Initializes variant to TrackingError.
   */
  Error () noexcept = default;

  /**
   * @brief Constructs an Error from a domain variant and diagnostic message.
   * @param domain Typed error domain payload.
   * @param message Complete-sentence diagnostic without trailing period.
   * @pre None.
   * @post Stores domain and message and emits structured error log entry.
   */
  Error (ErrorVariant domain, std::string message) noexcept;

  /**
   * @brief Creates a TrackingError-domain Error.
   * @param code Specific TrackingErrorCode value.
   * @param run_id Identifier of the run associated with the error.
   * @param message Complete-sentence diagnostic without trailing period.
   * @pre None.
   * @post is_tracking_error() returns true on the returned object.
   * @returns Constructed Error for the Tracking domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  tracking (TrackingErrorCode code, std::string run_id,
            std::string message) noexcept -> Error;

  /**
   * @brief Creates a RegistryError-domain Error.
   * @param code Specific RegistryErrorCode value.
   * @param model_name Name of the model associated with the error.
   * @param version Version number associated with the error.
   * @param message Complete-sentence diagnostic without trailing period.
   * @pre None.
   * @post is_registry_error() returns true on the returned object.
   * @returns Constructed Error for the Registry domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  registry (RegistryErrorCode code, std::string model_name, uint64_t version,
            std::string message) noexcept -> Error;

  /**
   * @brief Creates a DagError-domain Error.
   * @param code Specific DagErrorCode value.
   * @param task_id Identifier of the task node where the failure occurred.
   * @param message Complete-sentence diagnostic without trailing period.
   * @pre None.
   * @post is_dag_error() returns true on the returned object.
   * @returns Constructed Error for the DAG domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto dag (DagErrorCode code, std::string task_id,
                                 std::string message) noexcept -> Error;

  /**
   * @brief Creates an IndexError-domain Error.
   * @param code Specific IndexErrorCode value.
   * @param expected_dim Expected vector dimension if applicable.
   * @param actual_dim Actual vector dimension if applicable.
   * @param message Complete-sentence diagnostic without trailing period.
   * @pre None.
   * @post is_index_error() returns true on the returned object.
   * @returns Constructed Error for the Index domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  index (IndexErrorCode code, std::size_t expected_dim, std::size_t actual_dim,
         std::string message) noexcept -> Error;

  /**
   * @brief Creates a FairnessError-domain Error.
   * @param code Specific FairnessErrorCode value.
   * @param group_id Sensitive group identifier if applicable.
   * @param message Complete-sentence diagnostic without trailing period.
   * @pre None.
   * @post is_fairness_error() returns true on the returned object.
   * @returns Constructed Error for the Fairness domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  fairness (FairnessErrorCode code, uint32_t group_id,
            std::string message) noexcept -> Error;

  /**
   * @brief Retrieves the underlying error variant.
   * @pre None.
   * @post State is unchanged.
   * @returns Reference to the stored ErrorVariant.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto variant () const noexcept -> const ErrorVariant &;

  /**
   * @brief Retrieves the human-readable error message.
   * @pre None.
   * @post State is unchanged.
   * @returns View of the diagnostic string.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto message () const noexcept -> std::string_view;

  /**
   * @brief Retrieves the top-level category name for structured logging.
   * @pre None.
   * @post State is unchanged.
   * @returns One of "Tracking", "Registry", "Dag", "Index", or "Fairness".
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto category_name () const noexcept -> std::string_view;

  /**
   * @brief Checks whether the error originated from experiment tracking.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds TrackingError.
   */
  [[nodiscard]] auto is_tracking_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from the model registry.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds RegistryError.
   */
  [[nodiscard]] auto is_registry_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from DAG orchestration.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds DagError.
   */
  [[nodiscard]] auto is_dag_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from the vector index.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds IndexError.
   */
  [[nodiscard]] auto is_index_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from fairness/drift auditing.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds FairnessError.
   */
  [[nodiscard]] auto is_fairness_error () const noexcept -> bool;

private:
  ErrorVariant domain_{ TrackingError{} };
  std::string message_;
};

/**
 * @brief Standard return type for all fallible Yantra public APIs.
 */
template <typename T> using Result = std::expected<T, Error>;

} /* namespace yantra::inline v1 */

#endif /* YANTRA_ERROR_HPP */
