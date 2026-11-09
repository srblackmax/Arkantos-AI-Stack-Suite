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

#ifndef SUTRA_ERROR_HPP
#define SUTRA_ERROR_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <variant>

namespace sutra::inline v1
{

/**
 * @brief Enumerates failure modes originating from the LLM inference backend.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class LlmErrorCode : uint8_t
{
  ConnectionFailed = 0,
  RateLimited = 1,
  ContextOverflow = 2,
  InvalidResponse = 3,
  GpuExecutionFailed = 4,
  Cancelled = 5,
  Router = 6
};

/**
 * @brief Enumerates failure modes originating from tool validation or runtime.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class ToolErrorCode : uint8_t
{
  ToolNotFound = 0,
  InvalidInput = 1,
  ExecutionFailure = 2,
  Timeout = 3,
  SchemaViolation = 4,
  SandboxViolation = 5,
  NotAllowed = 6,
  SshAuth = 7,
  SshDenied = 8,
  SshTimeout = 9
};

/**
 * @brief Enumerates failure modes originating from planning and linting.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class PlanningErrorCode : uint8_t
{
  NoValidPlan = 0,
  MaxStepsExceeded = 1,
  CycleDetected = 2,
  DanglingDependency = 3,
  SchemaMismatch = 4
};

/**
 * @brief Enumerates failure modes originating from the framework runtime.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class FrameworkErrorCode : uint8_t
{
  ResourceExhaustion = 0,
  InvalidConfiguration = 1,
  CheckpointCorrupted = 2,
  ConcurrencyViolation = 3,
  CudaError = 4
};

/**
 * @brief Enumerates failure modes originating from voice/audio I/O subsystems.
 * @note Thread-safety: Immutable value type; safe for concurrent access.
 */
enum class VoiceErrorCode : uint8_t
{
  AudioDevice = 0,
  Stt = 1,
  Tts = 2,
  AudioXrun = 3
};

/**
 * @brief Describes an LLM error payload within AgentError.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct LlmError
{
  LlmErrorCode code = LlmErrorCode::ConnectionFailed;
};

/**
 * @brief Describes a tool error payload within AgentError.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct ToolError
{
  ToolErrorCode code = ToolErrorCode::ExecutionFailure;
  std::string tool_name;
};

/**
 * @brief Describes a planning or workflow linting error payload.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct PlanningError
{
  PlanningErrorCode code = PlanningErrorCode::NoValidPlan;
  std::size_t step_index = 0U;
};

/**
 * @brief Describes an internal framework or hardware resource error payload.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct FrameworkError
{
  FrameworkErrorCode code = FrameworkErrorCode::InvalidConfiguration;
  int platform_code = 0;
};

/**
 * @brief Describes an audio device, STT, or TTS error payload.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct VoiceError
{
  VoiceErrorCode code = VoiceErrorCode::AudioDevice;
  int platform_code = 0;
};

/**
 * @brief Describes an upstream Yantra model registry error propagated
 *        through Sutra.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct YantraRegistryError
{
  uint8_t yantra_code = 0U;
  std::string model_name;
  uint64_t version = 0U;
};

/**
 * @brief Discriminated union holding the specific domain of an AgentError.
 */
using ErrorVariant = std::variant<LlmError, ToolError, PlanningError,
                                  FrameworkError, VoiceError,
                                  YantraRegistryError>;

/**
 * @brief Represents a structured, non-throwing error across Sutra APIs.
 * @pre Error message should follow GNU diagnostics formatting.
 * @post Holds a valid ErrorVariant and human-readable diagnostic message.
 * @note Thread-safety: Const methods are safe for concurrent reads.
 */
class AgentError
{
public:
  /**
   * @brief Constructs a default framework configuration error.
   * @pre None.
   * @post Initializes variant to FrameworkError.
   */
  AgentError () noexcept = default;

  /**
   * @brief Constructs an AgentError from a domain variant and message.
   * @param domain The typed error domain payload.
   * @param message Human-readable diagnostic sentence without trailing period.
   * @pre None.
   * @post Stores domain and message in the error instance.
   */
  AgentError (ErrorVariant domain, std::string message) noexcept;

  /**
   * @brief Creates an LLM-domain AgentError.
   * @param code The specific LlmErrorCode value.
   * @param message Diagnostic description suggesting corrective action.
   * @pre None.
   * @post is_llm_error() returns true on the returned object.
   * @returns Constructed AgentError for the LLM domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto llm (LlmErrorCode code,
                                 std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a Tool-domain AgentError.
   * @param code The specific ToolErrorCode value.
   * @param tool_name Identifier of the tool that failed.
   * @param message Diagnostic description suggesting corrective action.
   * @pre None.
   * @post is_tool_error() returns true on the returned object.
   * @returns Constructed AgentError for the Tool domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto tool (ToolErrorCode code, std::string tool_name,
                                  std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a Planning-domain AgentError.
   * @param code The specific PlanningErrorCode value.
   * @param step_index Zero-based step index where planning failed.
   * @param message Diagnostic description suggesting corrective action.
   * @pre None.
   * @post is_planning_error() returns true on the returned object.
   * @returns Constructed AgentError for the Planning domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  planning (PlanningErrorCode code, std::size_t step_index,
            std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a Framework-domain AgentError.
   * @param code The specific FrameworkErrorCode value.
   * @param platform_code OS or CUDA driver error code if applicable.
   * @param message Diagnostic description suggesting corrective action.
   * @pre None.
   * @post is_framework_error() returns true on the returned object.
   * @returns Constructed AgentError for the Framework domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto
  framework (FrameworkErrorCode code, int platform_code,
             std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a cooperative cancellation LLM AgentError.
   * @param message Diagnostic description of the cancellation.
   * @returns Constructed AgentError with LlmErrorCode::Cancelled.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto Cancelled (
      std::string message
      = "LLM generation was cooperatively cancelled") noexcept -> AgentError;

  /**
   * @brief Creates a context overflow LLM AgentError before generation starts.
   * @param message Diagnostic description of the context overflow.
   * @returns Constructed AgentError with LlmErrorCode::ContextOverflow.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto LlmContextOverflow (
      std::string message
      = "Prompt exceeds model context window length") noexcept -> AgentError;

  /**
   * @brief Creates an audio capture/playback device AgentError.
   * @param platform_code Native ALSA/audio driver error code.
   * @param message Diagnostic description suggesting corrective action.
   * @returns Constructed AgentError with VoiceErrorCode::AudioDevice.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  AudioDevice (int platform_code, std::string message) noexcept -> AgentError;

  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  AudioDevice (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an audio buffer overrun/underrun (xrun) AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  AudioXrun (int platform_code, std::string message) noexcept -> AgentError;

  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  AudioXrun (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a speech-to-text (STT) AgentError.
   * @param message Diagnostic description suggesting corrective action.
   * @returns Constructed AgentError with VoiceErrorCode::Stt.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto Stt (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a text-to-speech (TTS) AgentError.
   * @param message Diagnostic description suggesting corrective action.
   * @returns Constructed AgentError with VoiceErrorCode::Tts.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto Tts (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an SSH authentication failure AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  SshAuth (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an SSH command denylist violation AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  SshDenied (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an SSH connection/execution timeout AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  SshTimeout (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an AgentError when a model ID is missing from Yantra.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  ModelNotRegistered (std::string model_id,
                      std::string registry_path) noexcept -> AgentError;

  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  ModelNotRegistered (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an AgentError for a general Yantra ModelRegistry failure.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  ModelRegistry (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a turn router classification AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  Router (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a tool sandbox path escape violation AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  ToolSandboxViolation (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a disallowed command execution AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  ToolNotAllowed (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates a tool execution timeout AgentError.
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] static auto
  ToolTimeout (std::string message) noexcept -> AgentError;

  /**
   * @brief Creates an AgentError wrapping an upstream Yantra RegistryError.
   * @param yantra_code Numeric Yantra RegistryErrorCode value.
   * @param model_name Model name from the Yantra RegistryError.
   * @param version Version number from the Yantra RegistryError.
   * @param message Diagnostic description from Yantra.
   * @returns Constructed AgentError for the YantraRegistry domain.
   */
  [[nodiscard]] static auto
  yantra_registry (uint8_t yantra_code, std::string model_name,
                   uint64_t version,
                   std::string message) noexcept -> AgentError;

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
   * @returns One of "LLM", "Tool", "Planning", or "Framework".
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto category_name () const noexcept -> std::string_view;

  /**
   * @brief Checks whether the error originated from the LLM layer.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds LlmError.
   */
  [[nodiscard]] auto is_llm_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from a tool invocation.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds ToolError.
   */
  [[nodiscard]] auto is_tool_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from planning or linting.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds PlanningError.
   */
  [[nodiscard]] auto is_planning_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from the framework runtime.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds FrameworkError.
   */
  [[nodiscard]] auto is_framework_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from voice/audio I/O.
   */
  [[nodiscard]] auto is_voice_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error is an audio device error.
   */
  [[nodiscard]] auto is_audio_device () const noexcept -> bool;

  /**
   * @brief Checks whether the error is an audio xrun error.
   */
  [[nodiscard]] auto is_audio_xrun () const noexcept -> bool;

  /**
   * @brief Checks whether the error is an STT error.
   */
  [[nodiscard]] auto is_stt_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a TTS error.
   */
  [[nodiscard]] auto is_tts_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error is an SSH authentication failure.
   */
  [[nodiscard]] auto is_ssh_auth () const noexcept -> bool;

  /**
   * @brief Checks whether the error is an SSH denylist violation.
   */
  [[nodiscard]] auto is_ssh_denied () const noexcept -> bool;

  /**
   * @brief Checks whether the error is an SSH timeout.
   */
  [[nodiscard]] auto is_ssh_timeout () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a missing registered model error.
   */
  [[nodiscard]] auto is_model_not_registered () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a general model registry error.
   */
  [[nodiscard]] auto is_model_registry () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a turn router error.
   */
  [[nodiscard]] auto is_router_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a tool sandbox violation.
   */
  [[nodiscard]] auto is_tool_sandbox_violation () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a disallowed command error.
   */
  [[nodiscard]] auto is_tool_not_allowed () const noexcept -> bool;

  /**
   * @brief Checks whether the error is a tool timeout error.
   */
  [[nodiscard]] auto is_tool_timeout () const noexcept -> bool;

  /**
   * @brief Checks whether the error represents cooperative LLM cancellation.
   */
  [[nodiscard]] auto is_cancelled () const noexcept -> bool;

  /**
   * @brief Checks whether the error represents LLM context window overflow.
   */
  [[nodiscard]] auto is_context_overflow () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from a Yantra RegistryError.
   */
  [[nodiscard]] auto is_yantra_registry_error () const noexcept -> bool;

  /**
   * @brief Retrieves the underlying Yantra RegistryErrorCode byte if present.
   */
  [[nodiscard]] auto yantra_error_code () const noexcept -> uint8_t;

private:
  ErrorVariant domain_{ FrameworkError{} };
  std::string message_;
};

/**
 * @brief Free helper checking if an AgentError wraps a Yantra RegistryError.
 */
[[nodiscard]] inline auto
is_yantra_registry_error (const AgentError &err) noexcept -> bool
{
  return err.is_yantra_registry_error ();
}

/**
 * @brief Standard return type for all fallible Sutra public APIs.
 */
template <typename T> using Result = std::expected<T, AgentError>;

} /* namespace sutra::inline v1 */

#endif /* SUTRA_ERROR_HPP */
