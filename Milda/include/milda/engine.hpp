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

#ifndef MILDA_ENGINE_HPP
#define MILDA_ENGINE_HPP

#include "milda/hardware.hpp"
#include "milda/milda_i18n.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_sampler;
struct llama_model_tensor_buft_override;
struct ggml_threadpool;

namespace milda
{

/* Supported KV cache quantization types (--cache-type-k / --cache-type-v).  */
enum class kv_cache_type : uint8_t
{
  f16 = 0U,
  q8_0 = 1U,
  q4_0 = 2U
};

/* Parses a KV cache quantization type string ("f16", "q8_0", "q4_0").  */
[[nodiscard]] bool
parse_kv_cache_type (std::string_view text, kv_cache_type *out_type) noexcept;

/* Returns the canonical string name for a kv_cache_type.  */
[[nodiscard]] const char *
kv_cache_type_name (kv_cache_type type) noexcept;

/* Filters a sequence of streamed token pieces, suppressing any
   <|channel>thought...<channel|> or <think>...</think> blocks and stopping
   at end-of-turn markers (<turn|>, <|im_end|>).  */
[[nodiscard]] bool
filter_thought_pieces (const std::vector<std::string_view> &pieces,
                       std::string *out_visible) noexcept;

/* Complete configuration for Milda's hybrid CPU/GPU execution engine.  */
struct engine_config
{
  std::string model_path;
  std::string session_path;
  int32_t n_gpu_layers = -1;
  int32_t n_cpu_moe = -1;
  int32_t n_cpu_ffn = -1;
  int32_t n_threads = 0;
  int32_t n_threads_batch = 0;
  uint32_t n_ctx = 16384U;
  uint32_t n_batch = 2048U;
  uint32_t n_ubatch = 512U;
  int32_t max_new_tokens = 1024;
  float temperature = 0.75F;
  int32_t top_k = 40;
  float top_p = 0.90F;
  float min_p = 0.05F;
  float repeat_penalty = 1.08F;
  uint32_t seed = 42U;
  kv_cache_type cache_type_k = kv_cache_type::q8_0;
  kv_cache_type cache_type_v = kv_cache_type::q8_0;
  MildaLang language = MildaLang::Auto;
  x_system_mode x_mode = x_system_mode::auto_lang;
  bool custom_repeat_penalty = false;
  bool custom_top_p = false;
  bool flash_attn = true;
  bool use_mmap = true;
  bool use_mlock = false;
  bool pin_pcores = true;
  bool stream_stdout = true;
};

/* Per-turn timing and token counters updated by the hot inference loop.  */
struct turn_metrics
{
  std::size_t prompt_tokens = 0U;
  std::size_t generated_tokens = 0U;
  double prompt_eval_ms = 0.0;
  double token_gen_ms = 0.0;
  double ttft_ms = 0.0;
  double prompt_eval_tok_per_s = 0.0;
  double token_gen_tok_per_s = 0.0;
  int32_t kv_shifts_performed = 0;
};

/* Ultra-lean C++23 conversational inference engine with hybrid CPU/GPU
   offloading, P-core thread pinning, quantized KV cache, and stateful
   Esperanto system prompt prefix pinning.  */
class inference_engine
{
public:
  inference_engine () noexcept;
  inference_engine (const inference_engine &) = delete;
  inference_engine &
  operator= (const inference_engine &) = delete;
  inference_engine (inference_engine &&) = delete;
  inference_engine &
  operator= (inference_engine &&) = delete;

  ~inference_engine () noexcept;

  /* Initializes the backend, maps weights, configures hybrid CPU/GPU
     offloading, pre-allocates hot-loop buffers, and evaluates & pins
     the Esperanto system prompt at KV index 0.  */
  [[nodiscard]] bool
  initialize (const engine_config &cfg);

  /* Shuts down the engine and releases all model, context, and pinned
     memory resources.  */
  void
  shutdown () noexcept;

  /* Evaluates a single user turn (already x-sistemo preprocessed),
     appending only the delta tokens to the KV cache and streaming the
     Esperanto response with zero heap allocations in the hot loop.  */
  [[nodiscard]] bool
  generate_turn (std::string_view user_text,
                 turn_metrics *out_metrics) noexcept;

  /* Truncates the KV cache back to the pinned Esperanto system prompt
     (/vakigi) without re-tokenizing or re-evaluating the system prompt.  */
  [[nodiscard]] bool
  clear_conversation () noexcept;

  /* Performs an in-place KV cache shift preserving [0, n_keep_) and
     evicting the oldest post-system conversation tokens.  */
  [[nodiscard]] bool
  shift_kv_cache_in_place (int32_t needed_tokens) noexcept;

  /* Serializes the active KV cache sequence and token history directly to a
     binary session file (/konservi).  */
  [[nodiscard]] bool
  save_session (const char *filepath) const noexcept;

  /* Restores a binary KV cache sequence and token history from disk
     (/sxargi) without re-evaluating past turns.  */
  [[nodiscard]] bool
  load_session (const char *filepath) noexcept;

  /* Populates a real-time hardware and performance telemetry snapshot.  */
  [[nodiscard]] telemetry_snapshot
  collect_telemetry (const turn_metrics &last_turn) const noexcept;

  [[nodiscard]] int32_t
  pinned_system_tokens () const noexcept;

  [[nodiscard]] int32_t
  cached_kv_tokens () const noexcept;

  [[nodiscard]] int32_t
  context_size () const noexcept;

  /* Dynamically switches the active language mode, updates the pinned
     system prompt context, and tunes per-language sampler parameters.  */
  [[nodiscard]] bool
  set_language (MildaLang lang) noexcept;

  [[nodiscard]] MildaLang
  configured_language () const noexcept;

  [[nodiscard]] MildaLang
  active_language () const noexcept;

  [[nodiscard]] float
  active_repeat_penalty () const noexcept;

  [[nodiscard]] float
  active_top_p () const noexcept;

private:
  [[nodiscard]] bool
  pin_system_prompt () noexcept;

  [[nodiscard]] bool
  rebuild_sampler (MildaLang target_lang) noexcept;

  [[nodiscard]] int32_t
  tokenize_into_preallocated (std::string_view text,
                              bool add_special) noexcept;

  [[nodiscard]] bool
  decode_tokens_chunked (const int32_t *tokens, int32_t count) noexcept;

  void
  format_turn_delta (std::string_view user_text) noexcept;

  engine_config config_;
  cpu_topology_info topo_;
  mapped_weight_file weight_map_;
  pinned_host_buffer pinned_staging_;

  std::vector<std::string> override_patterns_;
  std::vector<llama_model_tensor_buft_override> tensor_overrides_;

  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  ggml_threadpool *threadpool_decode_ = nullptr;
  ggml_threadpool *threadpool_batch_ = nullptr;

  /* Pre-allocated hot-loop buffers (zero dynamic allocations during TG).  */
  std::string turn_prompt_buf_;
  std::vector<int32_t> token_buf_;
  std::vector<int32_t> kv_tokens_;
  std::vector<uint8_t> pinned_sys_state_;
  std::array<char, 256U> decode_piece_buf_;

  std::string model_arch_;
  int32_t n_ctx_ = 0;
  int32_t n_keep_ = 0;
  int32_t n_past_ = 0;
  uint64_t turn_count_ = 0ULL;
  int32_t active_threads_decode_ = 1;
  int32_t active_threads_prefill_ = 1;
  int32_t p_cores_pinned_ = 0;
  int32_t e_cores_pinned_ = 0;
  int32_t offloaded_gpu_layers_ = 0;
  int32_t total_model_layers_ = 0;
  int32_t hybrid_cpu_ffn_layers_ = 0;
  float active_repeat_penalty_ = 1.08F;
  float active_top_p_ = 0.90F;
  MildaLang active_turn_lang_ = MildaLang::Esperanto;
  MildaLang active_sampler_lang_ = MildaLang::Auto;
  bool session_lang_locked_ = false;
  bool is_gemma_arch_ = false;
  bool is_qwen3_arch_ = false;
};

} /* namespace milda */

#endif /* MILDA_ENGINE_HPP */
