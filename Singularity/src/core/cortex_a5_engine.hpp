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

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "linux4_arena_and_mmap.hpp"

namespace singularity {

// GGUF v3 Constants & Types
constexpr uint32_t GGUF_MAGIC   = 0x46554747u; // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION = 3u;

enum GGMLType : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
};

enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct ModelConfig {
    uint32_t dim;            // Transformer embedding dimension (multiple of 32)
    uint32_t hidden_dim;     // SwiGLU FFN intermediate dimension (multiple of 32)
    uint32_t n_layers;       // Number of Transformer layers
    uint32_t n_heads;        // Number of query heads
    uint32_t n_kv_heads;     // Number of key/value heads (GQA/MHA)
    uint32_t vocab_size;     // Vocabulary size (up to 151,936+ for Qwen2.5)
    uint32_t max_seq_len;    // Maximum sequence context length (clamped for 35 MB KV budget)
    float    rms_norm_eps;   // RMSNorm epsilon
    float    rope_freq_base; // RoPE theta base (e.g. 10000.0f or 1000000.0f for Qwen2.5)
    uint32_t weight_type;    // Default layer weight type (GGML_TYPE_Q4_0 or GGML_TYPE_Q8_0)
    uint32_t bos_token_id;   // Beginning-of-sequence token ID
    uint32_t eos_token_id;   // End-of-sequence token ID
    bool     is_qwen2;       // True when general.architecture == "qwen2"
};

constexpr uint32_t MAX_LAYERS_SUPPORTED = 32;
constexpr uint32_t MAX_VOCAB_SUPPORTED  = 160000;

struct LayerWeights {
    const float* rms_att_weight; // [dim]
    const float* rms_ffn_weight; // [dim]
    const void*  wq;             // [dim * dim] in Q4_0 or Q8_0
    const void*  wk;             // [kv_dim * dim] in Q4_0 or Q8_0
    const void*  wv;             // [kv_dim * dim] in Q4_0 or Q8_0
    const void*  wo;             // [dim * dim] in Q4_0 or Q8_0
    const float* bq;             // Optional [dim] attention Q bias (Qwen2)
    const float* bk;             // Optional [kv_dim] attention K bias (Qwen2)
    const float* bv;             // Optional [kv_dim] attention V bias (Qwen2)
    const void*  w1;             // [hidden_dim * dim] in Q4_0 or Q8_0 (ffn_gate)
    const void*  w2;             // [dim * hidden_dim] in Q4_0 or Q8_0 (ffn_down)
    const void*  w3;             // [hidden_dim * dim] in Q4_0 or Q8_0 (ffn_up)
    size_t       file_offset;    // Min byte offset in GGUF file for madvise eviction
    size_t       max_offset;     // Max byte offset in GGUF file for madvise eviction
    size_t       byte_size;      // Total byte span of layer for madvise eviction
};

struct TransformerWeights {
    const void*  token_embedding_table; // [vocab_size * dim] in F32, Q4_0, or Q8_0
    uint32_t     embd_type;             // GGMLType of token_embedding_table
    const float* rms_final_weight;      // [dim]
    const void*  wcls;                  // [vocab_size * dim] in Q4_0 or Q8_0
    uint32_t     wcls_type;             // GGMLType of wcls
    size_t       wcls_file_offset;      // File offset of wcls for madvise eviction
    size_t       wcls_byte_size;        // Byte size of wcls
    LayerWeights layers[MAX_LAYERS_SUPPORTED];
};

struct InferenceStats {
    uint32_t tokens_generated;
    uint32_t prompt_tokens;
    uint32_t script_mask;
    uint32_t dominant_lang;
    uint32_t available_ram_kb;
    uint32_t madvise_evictions;
    uint64_t elapsed_ns;
};

// Persistent Cross-Session Memory Snapshot Constants & Structures
constexpr uint32_t SESSION_MAGIC        = 0x534E474Du; // "SNGM"
constexpr uint32_t SESSION_VERSION      = 1u;
constexpr uint32_t MAX_SESSION_HISTORY  = 8u;
constexpr const char* DEFAULT_SESSION_PATH  = "/opt/singularity/session.bin";
constexpr const char* FALLBACK_SESSION_PATH = "/var/tmp/singularity_session.bin";

struct SessionTurnRecord {
    uint32_t session_id;
    uint32_t turn_id;
    uint32_t lang_id;
    char     user_prompt[96];
    char     assistant_reply[160];
};

struct SessionSnapshotHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t session_count;
    uint32_t turn_count;
    int32_t  forced_lang;
    uint32_t last_dominant_lang;
    uint32_t model_dim;
    uint32_t model_layers;
    uint32_t model_kv_heads;
    uint32_t model_vocab_size;
    uint32_t kv_session_pos;
    uint32_t last_best_tok;
    uint32_t history_count;
    char     user_fact[128];
    SessionTurnRecord history[MAX_SESSION_HISTORY];
};

class CortexA5Engine {
public:
    constexpr CortexA5Engine()
        : cfg_{0, 0, 0, 0, 0, 0, 0, 1e-5f, 10000.0f, GGML_TYPE_Q4_0, 0, 31, false},
          weights_{},
          mmap_file_(),
          synthetic_gguf_arena_(),
          scratch_arena_(),
          kv_arena_(),
          mem_guard_(),
          raw_gguf_tokens_ptr_(nullptr),
          raw_gguf_tokens_count_(0),
          vocab_tokens_(nullptr),
          vocab_lens_(nullptr),
          vocab_script_(nullptr),
          x_(nullptr),
          xb_(nullptr),
          xb2_(nullptr),
          hb_(nullptr),
          hb2_(nullptr),
          q_(nullptr),
          k_(nullptr),
          v_(nullptr),
          att_(nullptr),
          logits_(nullptr),
          xq_(nullptr),
          hbq_(nullptr),
          rope_cos_sin_(nullptr),
          k_f32_tmp_(nullptr),
          v_f32_tmp_(nullptr),
          key_cache_f16_(nullptr),
          val_cache_f16_(nullptr),
          session_pos_(0),
          turn_count_(0),
          session_count_(0),
          forced_lang_(-1),
          last_dominant_lang_(LANG_ENGLISH),
          last_best_tok_(0),
          history_count_(0),
          user_fact_{},
          session_file_path_{},
          history_{},
          initialized_(false) {}

    ~CortexA5Engine() { shutdown(); }

    // Load an external GGUF v2/v3 file via sys_mmap2
    bool load_gguf_file(const char* path);

    // Build a valid GGUF v3 model image for Milda's 5 languages (either written
    // to disk if out_path != nullptr, or mapped directly in memory)
    bool init_milda_5lang_model(const char* optional_out_path = nullptr);

    // Run a single token forward pass at sequence position `pos`
    // Zero malloc/new calls; 100% of math dispatched to .S kernels.
    const float* forward_token(
        uint32_t token,
        uint32_t pos,
        bool force_layer_stream = false,
        bool compute_logits = true,
        uint32_t active_lang = LANG_ENGLISH
    );

    // Validate UTF-8 prompt, route script, and stream generated tokens
    bool generate_stream(
        const char* prompt,
        uint32_t max_new_tokens,
        bool print_output,
        InferenceStats* out_stats
    );

    // Execute one conversational turn maintaining persistent fp16 KV cache state
    bool chat_turn(
        const char* user_prompt,
        uint32_t max_new_tokens,
        bool print_output,
        InferenceStats* out_stats
    );

    // Persistent Cross-Session Memory API (saves/restores fp16 KV cache + episodic history)
    bool load_session_state(const char* path = DEFAULT_SESSION_PATH);
    bool save_session_state(const char* path = nullptr) const;
    bool clear_session_state(const char* path = nullptr);
    void set_session_file_path(const char* path);
    const char* session_file_path() const { return session_file_path_; }
    void set_user_fact(const char* fact);
    const char* user_fact() const { return user_fact_; }
    void print_memory_summary() const;

    void reset_session() {
        session_pos_ = 0;
        turn_count_  = 0;
    }

    void set_forced_language(int32_t lang_id) { forced_lang_ = lang_id; }
    int32_t forced_language() const { return forced_lang_; }
    uint32_t session_pos() const { return session_pos_; }
    uint32_t turn_count() const { return turn_count_; }
    uint32_t session_count() const { return session_count_; }
    uint32_t history_count() const { return history_count_; }
    const SessionTurnRecord& history_entry(uint32_t idx) const { return history_[idx]; }

    const ModelConfig& config() const { return cfg_; }
    size_t scratch_arena_used() const { return scratch_arena_.used; }
    size_t kv_arena_used() const { return kv_arena_.used; }
    uint32_t available_ram_kb() { return mem_guard_.query_available_ram_kb(); }
    uint32_t free_ram_kb() {
        mem_guard_.query_available_ram_kb();
        return mem_guard_.last_mem_free_kb;
    }

    static const char* language_name(uint32_t lang_id);

    void shutdown();

private:
    bool parse_gguf_from_memory(const uint8_t* base, size_t size);
    bool allocate_runtime_arenas();
    void matmul_quantized(
        float* out,
        const block_q8_0* x_q8,
        const void* w,
        uint32_t w_type,
        uint32_t n_in,
        uint32_t n_out
    );
    uint32_t encode_qwen2_bpe(const char* utf8_text, uint32_t* out_tokens, uint32_t max_tokens) const;
    size_t decode_qwen2_token(uint32_t token_id, char* out_buf, size_t max_len) const;
    uint32_t detect_milda_language(const char* prompt, uint32_t script_mask) const;
    void extract_user_fact_from_prompt(const char* prompt);
    void record_turn_history(uint32_t lang_id, const char* prompt, const char* reply);

    ModelConfig        cfg_;
    TransformerWeights weights_;
    Linux4MmapFile     mmap_file_;
    Linux4Arena        synthetic_gguf_arena_;
    Linux4Arena        scratch_arena_;
    Linux4Arena        kv_arena_;
    Linux4MemoryGuard  mem_guard_;

    const uint8_t* raw_gguf_tokens_ptr_;
    uint32_t       raw_gguf_tokens_count_;
    const char**   vocab_tokens_;
    uint16_t*      vocab_lens_;
    uint8_t*       vocab_script_;

    // Pre-allocated 32-byte aligned activation buffers in scratch_arena_ (<= 16 MB)
    float*      x_;
    float*      xb_;
    float*      xb2_;
    float*      hb_;
    float*      hb2_;
    float*      q_;
    float*      k_;
    float*      v_;
    float*      att_;
    float*      logits_;
    block_q8_0* xq_;
    block_q8_0* hbq_;
    float*      rope_cos_sin_;
    float*      k_f32_tmp_;
    float*      v_f32_tmp_;

    // Pre-allocated fp16 KV cache in kv_arena_ (<= 35 MB)
    uint16_t*   key_cache_f16_;
    uint16_t*   val_cache_f16_;

    uint32_t    session_pos_;
    uint32_t    turn_count_;
    uint32_t    session_count_;
    int32_t     forced_lang_;
    uint32_t    last_dominant_lang_;
    uint32_t    last_best_tok_;
    uint32_t    history_count_;
    char        user_fact_[128];
    char        session_file_path_[128];
    SessionTurnRecord history_[MAX_SESSION_HISTORY];
    bool        initialized_;
};

} // namespace singularity
