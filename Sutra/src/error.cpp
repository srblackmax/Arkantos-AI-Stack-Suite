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

#include <sutra/error.hpp>

#include <utility>

namespace sutra::inline v1
{

/* Initializes an AgentError with a typed domain variant and message.  */
AgentError::AgentError (ErrorVariant domain, std::string message) noexcept
    : domain_ (std::move (domain)), message_ (std::move (message))
{
}

/* Constructs an LLM-category error instance.  */
auto
AgentError::llm (LlmErrorCode code, std::string message) noexcept -> AgentError
{
  return AgentError (LlmError{ code }, std::move (message));
}

/* Constructs a Tool-category error instance.  */
auto
AgentError::tool (ToolErrorCode code, std::string tool_name,
                  std::string message) noexcept -> AgentError
{
  return AgentError (ToolError{ code, std::move (tool_name) },
                     std::move (message));
}

/* Constructs a Planning-category error instance.  */
auto
AgentError::planning (PlanningErrorCode code, std::size_t step_index,
                      std::string message) noexcept -> AgentError
{
  return AgentError (PlanningError{ code, step_index }, std::move (message));
}

/* Constructs a Framework-category error instance.  */
auto
AgentError::framework (FrameworkErrorCode code, int platform_code,
                       std::string message) noexcept -> AgentError
{
  return AgentError (FrameworkError{ code, platform_code },
                     std::move (message));
}

/* Constructs a cooperative cancellation LLM error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::Cancelled (std::string message) noexcept -> AgentError
{
  return AgentError (LlmError{ LlmErrorCode::Cancelled }, std::move (message));
}

/* Constructs a context overflow LLM error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::LlmContextOverflow (std::string message) noexcept -> AgentError
{
  return AgentError (LlmError{ LlmErrorCode::ContextOverflow },
                     std::move (message));
}

/* Constructs an audio device voice error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::AudioDevice (int platform_code,
                         std::string message) noexcept -> AgentError
{
  return AgentError (VoiceError{ VoiceErrorCode::AudioDevice, platform_code },
                     std::move (message));
}

// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::AudioDevice (std::string message) noexcept -> AgentError
{
  return AgentError (VoiceError{ VoiceErrorCode::AudioDevice, 0 },
                     std::move (message));
}

/* Constructs an audio buffer xrun error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::AudioXrun (int platform_code,
                       std::string message) noexcept -> AgentError
{
  return AgentError (VoiceError{ VoiceErrorCode::AudioXrun, platform_code },
                     std::move (message));
}

// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::AudioXrun (std::string message) noexcept -> AgentError
{
  return AgentError (VoiceError{ VoiceErrorCode::AudioXrun, 0 },
                     std::move (message));
}

/* Constructs a speech-to-text voice error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::Stt (std::string message) noexcept -> AgentError
{
  return AgentError (VoiceError{ VoiceErrorCode::Stt, 0 },
                     std::move (message));
}

/* Constructs a text-to-speech voice error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::Tts (std::string message) noexcept -> AgentError
{
  return AgentError (VoiceError{ VoiceErrorCode::Tts, 0 },
                     std::move (message));
}

/* Constructs an SSH authentication failure error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::SshAuth (std::string message) noexcept -> AgentError
{
  return AgentError (ToolError{ ToolErrorCode::SshAuth, "ssh" },
                     std::move (message));
}

/* Constructs an SSH command denylist violation error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::SshDenied (std::string message) noexcept -> AgentError
{
  return AgentError (ToolError{ ToolErrorCode::SshDenied, "ssh" },
                     std::move (message));
}

/* Constructs an SSH connection or command timeout error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::SshTimeout (std::string message) noexcept -> AgentError
{
  return AgentError (ToolError{ ToolErrorCode::SshTimeout, "ssh" },
                     std::move (message));
}

/* Constructs a missing registered model error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::ModelNotRegistered (std::string model_id,
                                std::string registry_path) noexcept
    -> AgentError
{
  std::string msg = "Model ID '" + model_id + "' not registered in "
                    + registry_path;
  return AgentError (
      YantraRegistryError{ 1U, std::move (model_id), 0U }, std::move (msg));
}

// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::ModelNotRegistered (std::string message) noexcept -> AgentError
{
  return AgentError (YantraRegistryError{ 1U, "unknown", 0U },
                     std::move (message));
}

/* Constructs a general Yantra ModelRegistry failure error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::ModelRegistry (std::string message) noexcept -> AgentError
{
  return AgentError (YantraRegistryError{ 2U, "registry", 0U },
                     std::move (message));
}

/* Constructs a turn router classification error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::Router (std::string message) noexcept -> AgentError
{
  return AgentError (LlmError{ LlmErrorCode::Router }, std::move (message));
}

/* Constructs a tool sandbox path escape violation error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::ToolSandboxViolation (std::string message) noexcept -> AgentError
{
  return AgentError (
      ToolError{ ToolErrorCode::SandboxViolation, "sandbox" },
      std::move (message));
}

/* Constructs a disallowed command execution error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::ToolNotAllowed (std::string message) noexcept -> AgentError
{
  return AgentError (
      ToolError{ ToolErrorCode::NotAllowed, "run_command" },
      std::move (message));
}

/* Constructs a tool execution timeout error instance.  */
// NOLINTNEXTLINE(readability-identifier-naming)
auto
AgentError::ToolTimeout (std::string message) noexcept -> AgentError
{
  return AgentError (ToolError{ ToolErrorCode::Timeout, "tool" },
                     std::move (message));
}

/* Constructs a YantraRegistry-category error instance.  */
auto
AgentError::yantra_registry (uint8_t yantra_code, std::string model_name,
                             uint64_t version,
                             std::string message) noexcept -> AgentError
{
  return AgentError (
      YantraRegistryError{ yantra_code, std::move (model_name), version },
      std::move (message));
}

/* Returns the underlying error domain variant.  */
auto
AgentError::variant () const noexcept -> const ErrorVariant &
{
  return domain_;
}

/* Returns the diagnostic error message view.  */
auto
AgentError::message () const noexcept -> std::string_view
{
  return message_;
}

/* Returns the category name corresponding to the active variant arm.  */
auto
AgentError::category_name () const noexcept -> std::string_view
{
  if (std::holds_alternative<LlmError> (domain_))
    {
      return "LLM";
    }
  if (std::holds_alternative<ToolError> (domain_))
    {
      return "Tool";
    }
  if (std::holds_alternative<PlanningError> (domain_))
    {
      return "Planning";
    }
  if (std::holds_alternative<VoiceError> (domain_))
    {
      return "Voice";
    }
  if (std::holds_alternative<YantraRegistryError> (domain_))
    {
      return "YantraRegistry";
    }
  return "Framework";
}

/* Checks whether the active variant arm is LlmError.  */
auto
AgentError::is_llm_error () const noexcept -> bool
{
  return std::holds_alternative<LlmError> (domain_);
}

/* Checks whether the active variant arm is ToolError.  */
auto
AgentError::is_tool_error () const noexcept -> bool
{
  return std::holds_alternative<ToolError> (domain_);
}

/* Checks whether the active variant arm is PlanningError.  */
auto
AgentError::is_planning_error () const noexcept -> bool
{
  return std::holds_alternative<PlanningError> (domain_);
}

/* Checks whether the active variant arm is FrameworkError.  */
auto
AgentError::is_framework_error () const noexcept -> bool
{
  return std::holds_alternative<FrameworkError> (domain_);
}

/* Checks whether the active variant arm is VoiceError.  */
auto
AgentError::is_voice_error () const noexcept -> bool
{
  return std::holds_alternative<VoiceError> (domain_);
}

auto
AgentError::is_audio_device () const noexcept -> bool
{
  if (const auto *v_err = std::get_if<VoiceError> (&domain_))
    {
      return v_err->code == VoiceErrorCode::AudioDevice;
    }
  return false;
}

auto
AgentError::is_audio_xrun () const noexcept -> bool
{
  if (const auto *v_err = std::get_if<VoiceError> (&domain_))
    {
      return v_err->code == VoiceErrorCode::AudioXrun;
    }
  return false;
}

auto
AgentError::is_stt_error () const noexcept -> bool
{
  if (const auto *v_err = std::get_if<VoiceError> (&domain_))
    {
      return v_err->code == VoiceErrorCode::Stt;
    }
  return false;
}

auto
AgentError::is_tts_error () const noexcept -> bool
{
  if (const auto *v_err = std::get_if<VoiceError> (&domain_))
    {
      return v_err->code == VoiceErrorCode::Tts;
    }
  return false;
}

auto
AgentError::is_ssh_auth () const noexcept -> bool
{
  if (const auto *t_err = std::get_if<ToolError> (&domain_))
    {
      return t_err->code == ToolErrorCode::SshAuth;
    }
  return false;
}

auto
AgentError::is_ssh_denied () const noexcept -> bool
{
  if (const auto *t_err = std::get_if<ToolError> (&domain_))
    {
      return t_err->code == ToolErrorCode::SshDenied;
    }
  return false;
}

auto
AgentError::is_ssh_timeout () const noexcept -> bool
{
  if (const auto *t_err = std::get_if<ToolError> (&domain_))
    {
      return t_err->code == ToolErrorCode::SshTimeout;
    }
  return false;
}

auto
AgentError::is_model_not_registered () const noexcept -> bool
{
  if (const auto *y_err = std::get_if<YantraRegistryError> (&domain_))
    {
      return y_err->yantra_code == 1U;
    }
  return false;
}

auto
AgentError::is_model_registry () const noexcept -> bool
{
  return std::holds_alternative<YantraRegistryError> (domain_);
}

auto
AgentError::is_router_error () const noexcept -> bool
{
  if (const auto *l_err = std::get_if<LlmError> (&domain_))
    {
      return l_err->code == LlmErrorCode::Router;
    }
  return false;
}

auto
AgentError::is_tool_sandbox_violation () const noexcept -> bool
{
  if (const auto *t_err = std::get_if<ToolError> (&domain_))
    {
      return t_err->code == ToolErrorCode::SandboxViolation;
    }
  return false;
}

auto
AgentError::is_tool_not_allowed () const noexcept -> bool
{
  if (const auto *t_err = std::get_if<ToolError> (&domain_))
    {
      return t_err->code == ToolErrorCode::NotAllowed;
    }
  return false;
}

auto
AgentError::is_tool_timeout () const noexcept -> bool
{
  if (const auto *t_err = std::get_if<ToolError> (&domain_))
    {
      return t_err->code == ToolErrorCode::Timeout;
    }
  return false;
}

/* Checks whether the error is LlmError with LlmErrorCode::Cancelled.  */
auto
AgentError::is_cancelled () const noexcept -> bool
{
  if (const auto *llm_err = std::get_if<LlmError> (&domain_))
    {
      return llm_err->code == LlmErrorCode::Cancelled;
    }
  return false;
}

/* Checks whether the error is LlmError with LlmErrorCode::ContextOverflow.  */
auto
AgentError::is_context_overflow () const noexcept -> bool
{
  if (const auto *llm_err = std::get_if<LlmError> (&domain_))
    {
      return llm_err->code == LlmErrorCode::ContextOverflow;
    }
  return false;
}

/* Checks whether the active variant arm is YantraRegistryError.  */
auto
AgentError::is_yantra_registry_error () const noexcept -> bool
{
  return std::holds_alternative<YantraRegistryError> (domain_);
}

/* Returns the underlying Yantra RegistryErrorCode byte if present.  */
auto
AgentError::yantra_error_code () const noexcept -> uint8_t
{
  if (const auto *y_err = std::get_if<YantraRegistryError> (&domain_))
    {
      return y_err->yantra_code;
    }
  return 0U;
}

} /* namespace sutra::inline v1 */
