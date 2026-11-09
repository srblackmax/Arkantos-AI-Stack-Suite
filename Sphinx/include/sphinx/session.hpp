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

#ifndef SPHINX_SESSION_HPP
#define SPHINX_SESSION_HPP

/**
 * @file session.hpp
 * @brief Yantra-supervised Sphinx voice assistant session orchestrating Vach,
 *        Sutra, Yantra (ModelRegistry, Run, PipelineDefinition DAG), and
 *        Darshana.
 * @note Compiles with only the C++23 standard library and public headers of
 *       Sutra, Vach, Yantra, and Darshana.
 */

#include <sphinx/memory.hpp>
#include <sphinx/router.hpp>
#include <sphinx/tools.hpp>
#include <sphinx/viz.hpp>
#include <sutra/sutra.hpp>
#include <vach/vach.hpp>
#include <yantra/yantra.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sphinx::inline v1
{

inline constexpr double kTargetChatTtfsP50Ms = 700.0;
inline constexpr double kTargetChatTtfsP99Ms = 1000.0;
inline constexpr double kTargetTaskTtfsP50Ms = 1500.0;
inline constexpr double kTargetTaskTtfsP99Ms = 2500.0;
inline constexpr double kTargetBargeInMs = 200.0;
inline constexpr double kTargetBackchannelMs = 500.0;
inline constexpr double kTargetMaxAudioGapMs = 100.0;
inline constexpr double kTargetMaxIdleCpuPct = 5.0;

/**
 * @brief Resolves a relative path under $HOME (or "." if unset).
 */
[[nodiscard]] inline auto
default_session_home_path (std::string_view rel) -> std::filesystem::path
{
  const char *home = std::getenv ("HOME");
  const std::string base = (home != nullptr && home[0] != '\0') ? home : ".";
  return std::filesystem::path (base) / rel;
}

/**
 * @brief Configuration for a SphinxSession.
 */
struct SphinxSessionConfig
{
  std::string session_id;
  std::filesystem::path registry_path
      = default_session_home_path ("models/registry.db");
  std::filesystem::path sessions_dir
      = default_session_home_path ("sphinx/sessions");
  std::filesystem::path sandbox_dir
      = default_session_home_path ("sphinx-workspace");
  std::string big_model_id = "qwen3-30b-a3b-q4km";
  uint64_t big_model_version = 1U;
  int32_t big_model_n_gpu_layers = -1;
  int32_t big_model_n_cpu_moe = 32;
  int32_t big_model_n_threads = 12;
  int32_t big_model_n_threads_batch = 12;
  std::string router_model_id = "qwen2.5-1.5b-q4km";
  uint64_t router_model_version = 1U;
  uint32_t big_context_len = 8192U;
  uint32_t router_context_len = 512U;
  uint32_t chat_max_tokens = 160U;
  uint32_t task_max_tokens = 64U;
  bool task_thinking_enabled = false;
  std::filesystem::path tts_model_path = default_session_home_path (
      ".local/share/piper/en_US-bryce-medium.onnx");
  std::filesystem::path tts_config_path = default_session_home_path (
      ".local/share/piper/en_US-bryce-medium.onnx.json");
  std::filesystem::path embedding_model_path = default_session_home_path (
      "models/Qwen3-Embedding-0.6B-Q8_0.gguf");
  std::filesystem::path memory_dir
      = default_session_home_path ("sphinx-workspace/.memory");
  std::size_t memory_top_k = kDefaultRetrieveTopK;
  float memory_threshold = kDefaultMemoryThreshold;
  bool enable_memory = true;
  bool headless_audio = false;
  bool verify_sha256_on_startup = false;
  bool lazy_load_big_model = true;
  bool low_memory_mode = false;
  bool unload_big_after_task = false;
  bool skip_tts_synthesis = false;
  bool preload_cpu_weights = false;
};

/**
 * @brief Verified model manifest resolution result from Yantra ModelRegistry.
 */
struct ResolvedModelArtifact
{
  std::string model_id;
  uint64_t version = 1U;
  std::filesystem::path artifact_path;
  std::string sha256_hex;
  std::string blake3_hex;
};

/**
 * @brief Splits streaming LLM text into complete sentences while respecting
 *        abbreviations (Dr., Mr., Mrs., Ms., Prof., Inc., Ltd., e.g., i.e.,
 *        etc.) and decimal numbers (3.14).
 */
[[nodiscard]] auto
find_next_sentence_boundary (std::string_view buffer) noexcept -> std::size_t;

/**
 * @brief Per-tool head and tail byte truncation limits.
 */
struct TruncationPolicy
{
  std::size_t head = 2048U;
  std::size_t tail = 0U;
};

/**
 * @brief Truncates a tool result using the per-tool TruncationPolicy table.
 *        If result.size() <= head + tail, returns result verbatim; otherwise
 *        returns "<head bytes>\n...[truncated N bytes]...\n<tail bytes>".
 */
[[nodiscard]] auto
truncate_tool_output (std::string_view tool_name,
                      std::string_view result) -> std::string;

/**
 * @brief Extracts requested file content from a user prompt (prioritizing
 *        quoted payloads inside "...", '...', or ```...``` distinct from
 *        target_path, followed by unquoted natural-language content clauses).
 */
[[nodiscard]] auto
extract_file_content (std::string_view text,
                      std::string_view target_path = {}) -> std::string;

/**
 * @brief Extracts all existing relative file paths inside sandbox_dir
 *        explicitly named in user_text for reading/inspection.
 */
[[nodiscard]] auto
extract_explicit_read_files (std::string_view user_text,
                             const std::filesystem::path &sandbox_dir)
    -> std::vector<std::string>;

/**
 * @brief Sets an optional epoch-seconds override for testing per-turn date
 *        regeneration without modifying the OS system clock. Pass std::nullopt
 *        to restore real std::time(nullptr).
 */
void
set_clock_override_for_testing (std::optional<std::time_t> epoch_sec) noexcept;

/**
 * @brief Gathers the read-only host system profile at session start using a
 *        4-worker subprocess pool (500 ms per command, 1500 ms total budget,
 *        1000-byte maximum size).
 */
[[nodiscard]] auto
gather_session_profile (uint32_t total_budget_ms = 1500U,
                        uint32_t per_cmd_timeout_ms = 500U,
                        bool allow_gpu_probe = true) noexcept -> std::string;

/**
 * @brief Ensures both Qwen2.5-7B-Instruct-Q4_K_M and
 *        Qwen2.5-1.5B-Instruct-Q4_K_M are registered in the Yantra
 *        ModelRegistry at registry_path.
 */
[[nodiscard]] auto
ensure_models_registered (const std::filesystem::path &registry_path
                          = default_session_home_path (
                              "models/registry.db")) noexcept
    -> sutra::Result<void>;

/**
 * @brief Resolves a model path and SHA-256 hash exclusively through Yantra
 *        ModelRegistry; returns AgentError::ModelNotRegistered if absent.
 */
[[nodiscard]] auto
resolve_model_from_registry (const std::filesystem::path &registry_path,
                             std::string_view model_id, uint64_t version = 1U,
                             bool verify_mmap_hash = false) noexcept
    -> sutra::Result<ResolvedModelArtifact>;

/**
 * @brief Yantra-supervised real-time voice assistant session.
 */
class SphinxSession
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  SphinxSession (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates and initializes a SphinxSession, opening a Yantra Run,
   *        resolving models via Yantra ModelRegistry, and initializing Vach
   *        audio/STT/TTS, TurnRouter, and Sutra ToolRegistry.
   */
  [[nodiscard]] static auto
  create (const SphinxSessionConfig &config = SphinxSessionConfig{}) noexcept
      -> sutra::Result<std::unique_ptr<SphinxSession>>;

  SphinxSession (const SphinxSession &) = delete;
  auto operator= (const SphinxSession &) -> SphinxSession & = delete;
  SphinxSession (SphinxSession &&) noexcept;
  auto operator= (SphinxSession &&) noexcept -> SphinxSession &;
  ~SphinxSession () noexcept;

  /**
   * @brief Executes a voice turn from a 16 kHz mono float audio buffer:
   *        Vad -> Stt -> TurnRouter -> CHAT or TASK (Yantra DAG) ->
   *        sentence-chunked Tts -> AudioIo speaker playback.
   */
  [[nodiscard]] auto
  process_voice_turn (std::span<const float> utterance_16k,
                      uint32_t simulated_tool_delay_ms = 0U,
                      bool force_verify_retry_once = false) noexcept
      -> sutra::Result<TurnTelemetry>;

  /**
   * @brief Executes a text turn directly through TurnRouter -> CHAT or TASK
   *        (Yantra DAG) -> sentence-chunked Tts -> AudioIo speaker playback.
   */
  [[nodiscard]] auto
  process_text_turn (std::string_view user_text,
                     uint32_t simulated_tool_delay_ms = 0U,
                     bool force_verify_retry_once = false) noexcept
      -> sutra::Result<TurnTelemetry>;

  /**
   * @brief Exercises live barge-in during active LLM + TTS output and returns
   *        measured stop latency in milliseconds (must be <= 200 ms).
   */
  [[nodiscard]] auto test_barge_in (std::string_view prompt_text
                                    = "Explain the architecture of modern "
                                      "operating systems in detail.") noexcept
      -> sutra::Result<TurnTelemetry>;

  /**
   * @brief Measures single-core idle CPU utilization of the VAD + AudioIo
   *        capture loop over duration_ms.
   */
  [[nodiscard]] auto measure_idle_cpu_percent (uint32_t duration_ms) noexcept
      -> sutra::Result<double>;

  /**
   * @brief Finalizes the Yantra Run and renders the 3 Darshana session SVG
   *        plots in sessions_dir.
   */
  [[nodiscard]] auto finalize_session () noexcept
      -> sutra::Result<SessionPlotsReport>;

  /**
   * @brief Completely erases persistent cross-session memory (SEMANTIC,
   *        EPISODIC, and CORRECTION entries) in memory_dir and
   *        ~/.local/share/sphinx/memory, and suppresses episodic summary
   *        generation for prior turns of the current session.
   */
  [[nodiscard]] auto clear_memory () noexcept -> sutra::Result<void>;

  /**
   * @brief Returns the active session ID.
   */
  [[nodiscard]] auto session_id () const noexcept -> std::string_view;

  /**
   * @brief Returns all TurnTelemetry records collected in this session.
   */
  [[nodiscard]] auto turns () const noexcept -> std::span<const TurnTelemetry>;

  /**
   * @brief Returns all 16 kHz mono float samples recorded from TTS output
   *        during the session (for WAV export and continuity gap analysis).
   */
  [[nodiscard]] auto
  recorded_tts_16k () const noexcept -> std::span<const int16_t>;

  /**
   * @brief Clears the recorded 16 kHz mono TTS waveform accumulator.
   */
  void clear_recorded_tts_16k () noexcept;

  /**
   * @brief Accesses the TurnRouter instance.
   */
  [[nodiscard]] auto router () noexcept -> TurnRouter &;

  /**
   * @brief Accesses the Vach AudioIo instance.
   */
  [[nodiscard]] auto audio_io () noexcept -> vach::AudioIo &;

  /**
   * @brief Accesses the Sutra ToolRegistry instance.
   */
  [[nodiscard]] auto tool_registry () noexcept -> sutra::ToolRegistry &;

  /**
   * @brief Accesses the active Yantra Run.
   */
  [[nodiscard]] auto yantra_run () noexcept -> yantra::Run &;

  /**
   * @brief Returns startup load time of the router model in seconds.
   */
  [[nodiscard]] auto router_load_seconds () const noexcept -> double;

  /**
   * @brief Returns startup load time of the big model in seconds.
   */
  [[nodiscard]] auto big_model_load_seconds () const noexcept -> double;

  /**
   * @brief Returns measured peak GPU VRAM at startup in GiB.
   */
  [[nodiscard]] auto vram_peak_gb () const noexcept -> double;

  /**
   * @brief Accesses the EmbeddingClient (or nullptr if disabled).
   */
  [[nodiscard]] auto embedding_client () noexcept -> EmbeddingClient *;

  /**
   * @brief Accesses the MemoryStore (or nullptr if disabled).
   */
  [[nodiscard]] auto memory_store () noexcept -> MemoryStore *;

  /**
   * @brief Returns the system profile gathered at session start.
   */
  [[nodiscard]] auto
  system_profile () const noexcept -> std::string_view;

  /**
   * @brief Lazily loads and warms up the big TASK model if not yet resident.
   */
  [[nodiscard]] auto ensure_big_model_loaded () noexcept
      -> sutra::Result<void>;

  /**
   * @brief Unloads the resident big TASK model and returns its VRAM/RAM to OS.
   */
  void unload_big_model () noexcept;

  /**
   * @brief Returns true if the big TASK model is currently resident in memory.
   */
  [[nodiscard]] auto is_big_model_loaded () const noexcept -> bool;

private:
  std::unique_ptr<Impl> impl_;
};

} /* namespace sphinx::inline v1 */

#endif /* SPHINX_SESSION_HPP */
