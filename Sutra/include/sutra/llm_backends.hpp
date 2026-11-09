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

#ifndef SUTRA_LLM_BACKENDS_HPP
#define SUTRA_LLM_BACKENDS_HPP

#include <sutra/error.hpp>
#include <sutra/planner.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sutra::inline v1
{

inline constexpr uint32_t kDefaultLlamaCtxTokens = 4096U;
inline constexpr int32_t kDefaultLlamaGpuLayers = 99;
inline constexpr int32_t kDefaultSafeDense31bGpuLayers = 17;
inline constexpr int32_t kDefaultLlamaThreads = 12;
inline constexpr int32_t kDefaultLlamaThreadsBatch = 12;
inline constexpr uint32_t kDefaultMaxNewTokens = 256U;
inline constexpr uint64_t kDefaultParitySeed = 42U;
inline constexpr float kDefaultLogitParityTolerance = 1.0e-4F;
inline constexpr uint32_t kGgmlTypeF32 = 0U;
inline constexpr uint32_t kGgmlTypeF16 = 1U;
inline constexpr uint32_t kGgmlTypeQ4_0 = 2U;
inline constexpr uint32_t kGgmlTypeQ8_0 = 8U;

/**
 * @brief Streaming-safe model-family-aware parser for LLM tool invocations.
 * @pre None.
 * @post Buffers only the minimal disambiguating tail across token chunks.
 * @note Thread-safety: Per-stream stateful parser; use one instance per turn.
 */
class ToolCallParser
{
public:
  /**
   * @brief Constructs a parser for the given model-family ToolCallStyle.
   * @param style Model family tool-call syntax style.
   */
  explicit ToolCallParser (ToolCallStyle style
                           = ToolCallStyle::Fallback) noexcept;

  /**
   * @brief Creates a ToolCallParser configured for style.
   * @param style Target model family style.
   * @returns Configured ToolCallParser.
   */
  [[nodiscard]] static auto
  for_style (ToolCallStyle style) noexcept -> ToolCallParser;

  /**
   * @brief Detects ToolCallStyle from a GGUF model architecture or filename.
   * @param family_or_path Model family string or GGUF path.
   * @returns Matching ToolCallStyle (defaults to Fallback if unknown).
   */
  [[nodiscard]] static auto
  detect_style (std::string_view family_or_path) noexcept -> ToolCallStyle;

  /**
   * @brief Feeds a newly streamed token piece or accumulated text and extracts
   *        any newly completed ToolCall items.
   * @param text_chunk Streamed text fragment.
   * @pre None.
   * @post Retains only the minimal unclosed tag/JSON tail in internal buffer.
   * @returns Vector of newly completed ToolCall instances.
   */
  [[nodiscard]] auto
  parse (std::string_view text_chunk) -> std::vector<ToolCall>;

  /**
   * @brief Statelessly parses a complete final generation string for tool
   * calls.
   * @param final_text Full completion text.
   * @returns Vector of all extracted ToolCall items.
   */
  [[nodiscard]] auto
  parse_complete (std::string_view final_text) const -> std::vector<ToolCall>;

  /**
   * @brief Clears any buffered streaming tail state for reuse on a next turn.
   */
  void reset () noexcept;

  /**
   * @brief Returns the configured ToolCallStyle.
   */
  [[nodiscard]] auto style () const noexcept -> ToolCallStyle;

private:
  ToolCallStyle style_ = ToolCallStyle::Fallback;
  std::string tail_buffer_;
  std::size_t emitted_count_ = 0U;
};

/**
 * @brief Configuration for LlamaCppClient resident GGUF model loading.
 * @note MoE offloading (n_cpu_moe > 0) uses llama.cpp's tensor buffer override
 *       table (llama_model_params::tensor_buft_overrides) with pattern
 *       "blk\\.<idx>\\.ffn_(up|down|gate|gate_up)_(ch|)exps" mapped to
 *       ggml_backend_cpu_buffer_type() for layers 0 .. n_cpu_moe - 1, matching
 *       llama.cpp's --n-cpu-moe implementation.
 */
struct LlamaCppConfig
{
  std::string model_path;
  uint32_t n_ctx = kDefaultLlamaCtxTokens;
  uint32_t n_batch = 512U;
  uint32_t n_ubatch = 256U;
  int32_t n_gpu_layers = kDefaultLlamaGpuLayers;
  int32_t n_cpu_moe = 0;
  int32_t n_threads = kDefaultLlamaThreads;
  int32_t n_threads_batch = kDefaultLlamaThreadsBatch;
  bool flash_attn = true;
  bool use_mmap = true;
  bool use_mlock = false;
  bool preload_cpu_weights = false;
  uint32_t type_k = kGgmlTypeQ8_0;
  uint32_t type_v = kGgmlTypeQ8_0;
  uint32_t default_max_tokens = kDefaultMaxNewTokens;
  std::optional<ToolCallStyle> override_tool_style;
};

/**
 * @brief Live GGUF LLM backend wrapping llama.cpp behind an opaque handle.
 * @pre model_path must reference a valid GGUF model file.
 * @post Holds the model resident in memory/VRAM; streams tokens in real time.
 * @note Thread-safety: Non-blocking stream_completion dispatches to an
 *       internal worker thread; cancel() is lock-free and thread-safe.
 */
class LlamaCppClient final : public LlmClient
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  LlamaCppClient (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Initializes the process-wide llama.cpp and ggml GPU/CPU backend
   *        singletons (including CUDA/cuBLAS handles) once at startup so
   *        per-client memory hygiene baselines exclude one-time driver init.
   */
  static void init_backend () noexcept;

  /**
   * @brief Loads a GGUF model once and initializes the resident context.
   * @param config LlamaCppConfig specifying model path, n_ctx, and GPU layers.
   * @returns Result<std::unique_ptr<LlamaCppClient>> or AgentError.
   */
  [[nodiscard]] static auto create (LlamaCppConfig config) noexcept
      -> Result<std::unique_ptr<LlamaCppClient>>;

  /**
   * @brief Loads a GGUF model from a Yantra ModelRegistry lookup or
   *        registration result, propagating any Yantra RegistryError into
   *        AgentError while preserving the original Yantra error code.
   * @tparam ManifestResult Yantra Result<ModelManifest> type.
   * @param manifest_res Result returned by Yantra ModelRegistry.
   * @param base_config Base LlamaCppConfig (model_path is populated from
   *        manifest_res->artifact_path).
   * @returns Result<std::unique_ptr<LlamaCppClient>> or AgentError.
   */
  template <typename ManifestResult>
  [[nodiscard]] static auto
  create_from_registry (const ManifestResult &manifest_res,
                        LlamaCppConfig base_config = {}) noexcept
      -> Result<std::unique_ptr<LlamaCppClient>>
  {
    if (!manifest_res.has_value ())
      {
        const auto &yerr = manifest_res.error ();
        const uint8_t code = std::visit (
            [] (const auto &dom) noexcept -> uint8_t {
              return static_cast<uint8_t> (dom.code);
            },
            yerr.variant ());
        return std::unexpected (AgentError::yantra_registry (
            code, std::string (yerr.category_name ()), 0U,
            std::string (yerr.message ())));
      }
    base_config.model_path = manifest_res->artifact_path.string ();
    return create (std::move (base_config));
  }

  LlamaCppClient (const LlamaCppClient &) = delete;
  auto operator= (const LlamaCppClient &) -> LlamaCppClient & = delete;
  LlamaCppClient (LlamaCppClient &&) noexcept;
  auto operator= (LlamaCppClient &&) noexcept -> LlamaCppClient &;
  ~LlamaCppClient () noexcept override;

  /**
   * @brief Non-blocking streaming completion dispatching to background worker.
   */
  [[nodiscard]] auto
  stream_completion (const CompletionRequest &req, TokenStreamFn on_token,
                     CompletionDone on_done) noexcept -> Result<void> override;

  /**
   * @brief Halts active generation within 50 ms via llama_abort_callback.
   */
  [[nodiscard]] auto cancel () noexcept -> Result<void> override;

  /**
   * @brief Queries resident model context length and ToolCallStyle.
   */
  [[nodiscard]] auto
  capabilities () const noexcept -> ModelCapabilities override;

  /**
   * @brief Synchronous Sutra Planner adapter with streaming early-tool
   * dispatch.
   */
  [[nodiscard]] auto
  generate (const ConversationSpanView &history, std::size_t step_index,
            const StreamingToolDispatchFn &on_early_tool,
            std::pmr::memory_resource *scratch_arena) noexcept
      -> Result<LlmStepOutput> override;

  /**
   * @brief Evaluates a prompt at temperature=0 and returns the full fp32 logit
   *        vector at the next generated token position.
   * @param prompt Input text prompt.
   * @param seed Deterministic RNG seed.
   * @returns Result<std::vector<float>> of length capabilities().vocab_size.
   */
  [[nodiscard]] auto evaluate_next_token_logits (std::string_view prompt,
                                                 uint64_t seed
                                                 = kDefaultParitySeed) noexcept
      -> Result<std::vector<float>>;

private:
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Describes a single GGUF tensor inspected by NativeEngineClient.
 */
struct GgufTensorInfo
{
  std::string name;
  uint32_t ggml_type = 0U;
  std::vector<uint64_t> dimensions;
};

/**
 * @brief Second LlmClient subclass backed by the step-3 Basalt DL GGUF bridge.
 * @note Used by test_logit_parity to verify numerical parity or emit a
 *       per-layer divergence report.
 */
class NativeEngineClient final : public LlmClient
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  NativeEngineClient (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Opens and parses a GGUF v3 model file via the native DL loader.
   * @param model_path Path to the GGUF model file.
   * @param n_ctx Context window length.
   * @param enable_gguf_q4k_bridge Enables the GGUF v3 K-quant execution bridge
   *        (set false to simulate unquantized-only layer-0 divergence).
   * @returns Result<std::unique_ptr<NativeEngineClient>> or AgentError.
   */
  [[nodiscard]] static auto
  create (std::string_view model_path, uint32_t n_ctx = kDefaultLlamaCtxTokens,
          bool enable_gguf_q4k_bridge = true) noexcept
      -> Result<std::unique_ptr<NativeEngineClient>>;

  NativeEngineClient (const NativeEngineClient &) = delete;
  auto operator= (const NativeEngineClient &) -> NativeEngineClient & = delete;
  NativeEngineClient (NativeEngineClient &&) noexcept;
  auto operator= (NativeEngineClient &&) noexcept -> NativeEngineClient &;
  ~NativeEngineClient () noexcept override;

  [[nodiscard]] auto
  stream_completion (const CompletionRequest &req, TokenStreamFn on_token,
                     CompletionDone on_done) noexcept -> Result<void> override;

  [[nodiscard]] auto cancel () noexcept -> Result<void> override;

  [[nodiscard]] auto
  capabilities () const noexcept -> ModelCapabilities override;

  [[nodiscard]] auto
  generate (const ConversationSpanView &history, std::size_t step_index,
            const StreamingToolDispatchFn &on_early_tool,
            std::pmr::memory_resource *scratch_arena) noexcept
      -> Result<LlmStepOutput> override;

  /**
   * @brief Evaluates the native DL forward pass to produce next-token logits.
   */
  [[nodiscard]] auto evaluate_next_token_logits (std::string_view prompt,
                                                 uint64_t seed
                                                 = kDefaultParitySeed) noexcept
      -> Result<std::vector<float>>;

  /**
   * @brief Returns parsed GGUF v3 tensor descriptors for divergence reporting.
   */
  [[nodiscard]] auto
  inspected_tensors () const noexcept -> const std::vector<GgufTensorInfo> &;

private:
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Per-layer diagnostic report emitted when logit parity fails.
 */
struct LogitParityReport
{
  std::size_t prompts_tested = 0U;
  std::size_t positions_compared = 0U;
  std::size_t top1_matches = 0U;
  float max_abs_logit_delta = 0.0F;
  float tolerance = kDefaultLogitParityTolerance;
  bool passed = false;
  int32_t first_divergent_layer = -1;
  std::string divergent_tensor_name;
  std::string quantization_format;
  std::string root_cause_report;
};

/**
 * @brief Runs the numerical logit parity harness across N prompts between
 *        LlamaCppClient and NativeEngineClient.
 * @param ref_client Resident LlamaCppClient reference backend.
 * @param native_client Resident NativeEngineClient under test.
 * @param prompts Span of deterministic test prompts (short, long, unicode,
 * tools).
 * @param tolerance Maximum allowed |Delta logit| in fp32 (default 1e-4f).
 * @returns Result<LogitParityReport> with pass/fail and layer divergence info.
 */
[[nodiscard]] auto run_logit_parity_harness (
    LlamaCppClient &ref_client, NativeEngineClient &native_client,
    std::span<const std::string_view> prompts,
    float tolerance
    = kDefaultLogitParityTolerance) noexcept -> Result<LogitParityReport>;

} /* namespace sutra::inline v1 */

#endif /* SUTRA_LLM_BACKENDS_HPP */
