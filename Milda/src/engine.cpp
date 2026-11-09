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

#include "milda/engine.hpp"
#include "milda/esperanto.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#if defined (__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace milda
{

namespace
{

constexpr uint32_t k_gguf_magic = 0x46554747U;
constexpr uint64_t k_max_gguf_str = 65536ULL;
constexpr std::size_t k_pinned_staging_bytes = 16ULL << 20U;
constexpr std::size_t k_turn_prompt_reserve = 16384U;
constexpr int32_t k_default_prefill_chunk = 512;

struct gguf_header_probe
{
  std::string architecture;
  uint32_t block_count = 0U;
  uint32_t expert_count = 0U;
  std::size_t file_size_bytes = 0U;
  bool valid = false;
};

void
silent_llama_log_callback (enum ggml_log_level level, const char *text,
                           void *user_data)
{
  (void)user_data;
  if (level == GGML_LOG_LEVEL_ERROR && text != nullptr)
    {
      (void)std::fwrite (text, 1U, std::strlen (text), stderr);
    }
}

[[nodiscard]] ggml_type
to_ggml_kv_type (kv_cache_type type) noexcept
{
  switch (type)
    {
    case kv_cache_type::f16:
      return GGML_TYPE_F16;
    case kv_cache_type::q8_0:
      return GGML_TYPE_Q8_0;
    case kv_cache_type::q4_0:
      return GGML_TYPE_Q4_0;
    }
  return GGML_TYPE_F16;
}

[[nodiscard]] bool
skip_gguf_value (std::ifstream &in, uint32_t vtype) noexcept
{
  switch (vtype)
    {
    case 0U:
    case 1U:
    case 7U:
      in.seekg (1, std::ios::cur);
      return in.good ();
    case 2U:
    case 3U:
      in.seekg (2, std::ios::cur);
      return in.good ();
    case 4U:
    case 5U:
    case 6U:
      in.seekg (4, std::ios::cur);
      return in.good ();
    case 10U:
    case 11U:
    case 12U:
      in.seekg (8, std::ios::cur);
      return in.good ();
    case 8U:
      {
        uint64_t slen = 0ULL;
        in.read (std::bit_cast<char *> (&slen), sizeof (slen));
        if (!in.good () || slen > (16ULL << 20U))
          {
            return false;
          }
        in.seekg (static_cast<std::streamoff> (slen), std::ios::cur);
        return in.good ();
      }
    case 9U:
      {
        uint32_t elem_type = 0U;
        uint64_t count = 0ULL;
        in.read (std::bit_cast<char *> (&elem_type), sizeof (elem_type));
        in.read (std::bit_cast<char *> (&count), sizeof (count));
        if (!in.good ())
          {
            return false;
          }
        for (uint64_t idx = 0ULL; idx < count; ++idx)
          {
            if (!skip_gguf_value (in, elem_type))
              {
                return false;
              }
          }
        return true;
      }
    default:
      return false;
    }
}

[[nodiscard]] gguf_header_probe
probe_gguf_file (const std::string &path) noexcept
{
  gguf_header_probe probe = gguf_header_probe ();
  std::error_code ec;
  const auto fsz = std::filesystem::file_size (path, ec);
  if (!ec)
    {
      probe.file_size_bytes = static_cast<std::size_t> (fsz);
    }

  std::ifstream in (path, std::ios::binary);
  if (!in.is_open ())
    {
      return probe;
    }

  uint32_t magic = 0U;
  uint32_t version = 0U;
  uint64_t tensor_count = 0ULL;
  uint64_t kv_count = 0ULL;
  in.read (std::bit_cast<char *> (&magic), sizeof (magic));
  in.read (std::bit_cast<char *> (&version), sizeof (version));
  in.read (std::bit_cast<char *> (&tensor_count), sizeof (tensor_count));
  in.read (std::bit_cast<char *> (&kv_count), sizeof (kv_count));
  if (!in.good () || magic != k_gguf_magic)
    {
      return probe;
    }
  probe.valid = true;

  for (uint64_t i = 0ULL; i < kv_count; ++i)
    {
      uint64_t klen = 0ULL;
      in.read (std::bit_cast<char *> (&klen), sizeof (klen));
      if (!in.good () || klen > k_max_gguf_str)
        {
          break;
        }
      std::string key (static_cast<std::size_t> (klen), '\0');
      in.read (key.data (), static_cast<std::streamsize> (key.size ()));
      uint32_t vtype = 0U;
      in.read (std::bit_cast<char *> (&vtype), sizeof (vtype));
      if (!in.good ())
        {
          break;
        }

      if (key == "general.architecture" && vtype == 8U)
        {
          uint64_t slen = 0ULL;
          in.read (std::bit_cast<char *> (&slen), sizeof (slen));
          if (in.good () && slen <= 256ULL)
            {
              probe.architecture.resize (static_cast<std::size_t> (slen));
              in.read (probe.architecture.data (),
                       static_cast<std::streamsize> (slen));
              continue;
            }
          break;
        }
      if (key.ends_with (".block_count") && vtype == 4U)
        {
          uint32_t bcnt = 0U;
          in.read (std::bit_cast<char *> (&bcnt), sizeof (bcnt));
          if (in.good ())
            {
              probe.block_count = bcnt;
              continue;
            }
          break;
        }
      if (key.ends_with (".expert_count") && vtype == 4U)
        {
          uint32_t ecnt = 0U;
          in.read (std::bit_cast<char *> (&ecnt), sizeof (ecnt));
          if (in.good ())
            {
              probe.expert_count = ecnt;
              continue;
            }
          break;
        }
      if (!skip_gguf_value (in, vtype))
        {
          break;
        }
    }
  return probe;
}

enum class thought_filter_mode : uint8_t
{
  normal = 0U,
  gemma_channel = 1U,
  qwen_think = 2U
};

struct thought_stream_filter
{
  std::array<char, 512U> pending = {};
  std::size_t pending_len = 0U;
  Utf8StreamBuffer utf8_buf = Utf8StreamBuffer ();
  thought_filter_mode mode = thought_filter_mode::normal;
  bool trim_leading_ws = true;
  bool has_emitted_any = false;
  bool hit_stop = false;
  bool suppressed_on_step = false;
  bool emitted_on_step = false;
};

[[nodiscard]] bool
is_stop_token_text (std::string_view tok_text) noexcept
{
  return tok_text == "<turn|>" || tok_text == "<|im_end|>"
         || tok_text == "<end_of_turn>" || tok_text == "<|eot_id|>"
         || tok_text == "<eos>" || tok_text == "<|endoftext|>";
}

[[nodiscard]] bool
matches_cot_start (std::string_view text) noexcept
{
  constexpr std::array<std::string_view, 8U> k_cot_prefixes =
    {
      "thought\n",
      "thought ",
      "thought:",
      "The user ",
      "Thinking Process:",
      "Internal thought:",
      "I need to respond",
      "To respond to the user"
    };
  for (const std::string_view pfx : k_cot_prefixes)
    {
      if (text.starts_with (pfx))
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] bool
is_cot_start_prefix (std::string_view text) noexcept
{
  if (text.empty ())
    {
      return false;
    }
  constexpr std::array<std::string_view, 8U> k_cot_prefixes =
    {
      "thought\n",
      "thought ",
      "thought:",
      "The user ",
      "Thinking Process:",
      "Internal thought:",
      "I need to respond",
      "To respond to the user"
    };
  for (const std::string_view pfx : k_cot_prefixes)
    {
      if (text.size () < pfx.size ()
          && pfx.substr (0U, text.size ()) == text)
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] bool
is_special_tag_prefix (std::string_view text) noexcept
{
  if (text.empty ())
    {
      return false;
    }
  constexpr std::array<std::string_view, 11U> k_tags =
    {
      "<|channel>",
      "<channel|>",
      "<think>",
      "</think>",
      "<|think|>",
      "<turn|>",
      "<|im_end|>",
      "<|endoftext|>",
      "<end_of_turn>",
      "<|eot_id|>",
      "<eos>"
    };
  for (const std::string_view tag : k_tags)
    {
      if (text.size () <= tag.size ()
          && tag.substr (0U, text.size ()) == text)
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] std::size_t
longest_closing_suffix (std::string_view text, std::string_view close_tag,
                        std::string_view stop_tag) noexcept
{
  const std::size_t max_tag = std::max (close_tag.size (), stop_tag.size ());
  const std::size_t max_len
      = std::min (text.size (), (max_tag > 0U) ? (max_tag - 1U) : 0U);
  for (std::size_t len = max_len; len > 0U; --len)
    {
      const std::string_view tail = text.substr (text.size () - len);
      if ((len < close_tag.size () && close_tag.substr (0U, len) == tail)
          || (len < stop_tag.size () && stop_tag.substr (0U, len) == tail))
        {
          return len;
        }
    }
  return 0U;
}

void
consume_pending (thought_stream_filter *flt, std::size_t count) noexcept
{
  if (count >= flt->pending_len)
    {
      flt->pending_len = 0U;
      return;
    }
  const std::size_t rem = flt->pending_len - count;
  std::memmove (flt->pending.data (), flt->pending.data () + count, rem);
  flt->pending_len = rem;
}

void
emit_filtered_bytes (const char *data, std::size_t len, bool stream_stdout,
                     std::string *out_str,
                     thought_stream_filter *flt) noexcept
{
  if (len == 0U)
    {
      return;
    }
  flt->emitted_on_step = true;
  flt->has_emitted_any = true;
  flt->trim_leading_ws = false;
  flt->utf8_buf.push (std::string_view (data, len), stream_stdout, out_str);
}

void
process_thought_stream_piece (thought_stream_filter *flt,
                              std::string_view piece, bool stream_stdout,
                              std::string *out_str) noexcept
{
  flt->suppressed_on_step = (flt->mode != thought_filter_mode::normal);
  flt->emitted_on_step = false;

  if (piece.empty () || flt->hit_stop)
    {
      return;
    }

  const std::size_t room = flt->pending.size () - flt->pending_len;
  const std::size_t to_copy = std::min (room, piece.size ());
  if (to_copy > 0U)
    {
      std::memcpy (flt->pending.data () + flt->pending_len, piece.data (),
                   to_copy);
      flt->pending_len += to_copy;
    }

  while (flt->pending_len > 0U && !flt->hit_stop)
    {
      const std::string_view view (flt->pending.data (), flt->pending_len);

      if (flt->mode == thought_filter_mode::gemma_channel)
        {
          flt->suppressed_on_step = true;
          const std::size_t stop_pos = view.find ("<turn|>");
          const std::size_t close_pos = view.find ("<channel|>");
          if (stop_pos != std::string_view::npos
              && (close_pos == std::string_view::npos
                  || stop_pos < close_pos))
            {
              flt->pending_len = 0U;
              flt->hit_stop = true;
              return;
            }
          if (close_pos != std::string_view::npos)
            {
              consume_pending (flt, close_pos + 10U);
              flt->mode = thought_filter_mode::normal;
              flt->trim_leading_ws = true;
              continue;
            }
          const std::size_t keep_tail
              = longest_closing_suffix (view, "<channel|>", "<turn|>");
          consume_pending (flt, flt->pending_len - keep_tail);
          return;
        }

      if (flt->mode == thought_filter_mode::qwen_think)
        {
          flt->suppressed_on_step = true;
          const std::size_t im_pos = view.find ("<|im_end|>");
          const std::size_t eot_pos = view.find ("<|endoftext|>");
          const std::size_t stop_pos = std::min (im_pos, eot_pos);
          const std::size_t close_pos = view.find ("</think>");
          if (stop_pos != std::string_view::npos
              && (close_pos == std::string_view::npos
                  || stop_pos < close_pos))
            {
              flt->pending_len = 0U;
              flt->hit_stop = true;
              return;
            }
          if (close_pos != std::string_view::npos)
            {
              consume_pending (flt, close_pos + 8U);
              flt->mode = thought_filter_mode::normal;
              flt->trim_leading_ws = true;
              continue;
            }
          const std::size_t keep_im
              = longest_closing_suffix (view, "</think>", "<|im_end|>");
          const std::size_t keep_eot
              = longest_closing_suffix (view, "</think>", "<|endoftext|>");
          const std::size_t keep_tail = std::max (keep_im, keep_eot);
          consume_pending (flt, flt->pending_len - keep_tail);
          return;
        }

      if (flt->trim_leading_ws)
        {
          std::size_t ws_cnt = 0U;
          while (ws_cnt < view.size ()
                 && (view[ws_cnt] == '\n' || view[ws_cnt] == '\r'
                     || view[ws_cnt] == ' ' || view[ws_cnt] == '\t'))
            {
              ++ws_cnt;
            }
          if (ws_cnt > 0U)
            {
              consume_pending (flt, ws_cnt);
              flt->suppressed_on_step = true;
              continue;
            }
        }

      if (!flt->has_emitted_any && view[0] != '<')
        {
          if (matches_cot_start (view))
            {
              flt->mode = thought_filter_mode::gemma_channel;
              flt->suppressed_on_step = true;
              continue;
            }
          if (is_cot_start_prefix (view))
            {
              return;
            }
        }

      const std::size_t lt_pos = view.find ('<');
      if (lt_pos == std::string_view::npos)
        {
          emit_filtered_bytes (view.data (), view.size (), stream_stdout,
                               out_str, flt);
          flt->pending_len = 0U;
          return;
        }

      if (lt_pos > 0U)
        {
          emit_filtered_bytes (view.data (), lt_pos, stream_stdout, out_str,
                               flt);
          consume_pending (flt, lt_pos);
          continue;
        }

      if (view.starts_with ("<|channel>thought"))
        {
          consume_pending (flt, 17U);
          flt->mode = thought_filter_mode::gemma_channel;
          flt->suppressed_on_step = true;
          continue;
        }
      if (view.starts_with ("<|channel>"))
        {
          consume_pending (flt, 10U);
          flt->mode = thought_filter_mode::gemma_channel;
          flt->suppressed_on_step = true;
          continue;
        }
      if (view.starts_with ("<think>"))
        {
          consume_pending (flt, 7U);
          flt->mode = thought_filter_mode::qwen_think;
          flt->suppressed_on_step = true;
          continue;
        }
      if (view.starts_with ("<channel|>"))
        {
          consume_pending (flt, 10U);
          flt->trim_leading_ws = true;
          flt->suppressed_on_step = true;
          continue;
        }
      if (view.starts_with ("</think>"))
        {
          consume_pending (flt, 8U);
          flt->trim_leading_ws = true;
          flt->suppressed_on_step = true;
          continue;
        }
      if (view.starts_with ("<|think|>"))
        {
          consume_pending (flt, 9U);
          flt->trim_leading_ws = true;
          flt->suppressed_on_step = true;
          continue;
        }
      if (view.starts_with ("<turn|>") || view.starts_with ("<|im_end|>")
          || view.starts_with ("<|endoftext|>")
          || view.starts_with ("<end_of_turn>")
          || view.starts_with ("<|eot_id|>") || view.starts_with ("<eos>"))
        {
          flt->pending_len = 0U;
          flt->hit_stop = true;
          return;
        }

      if (is_special_tag_prefix (view))
        {
          return;
        }

      emit_filtered_bytes (view.data (), 1U, stream_stdout, out_str, flt);
      consume_pending (flt, 1U);
    }
}

void
finish_thought_stream_filter (thought_stream_filter *flt, bool stream_stdout,
                              std::string *out_str) noexcept
{
  if (flt->mode == thought_filter_mode::normal && !flt->hit_stop
      && flt->pending_len > 0U)
    {
      const std::string_view rem (flt->pending.data (), flt->pending_len);
      if (!rem.starts_with ("<|") && !rem.starts_with ("<channel")
          && !rem.starts_with ("<think") && !rem.starts_with ("</think")
          && !rem.starts_with ("<turn") && !rem.starts_with ("<end_of_turn")
          && !rem.starts_with ("<eos"))
        {
          emit_filtered_bytes (rem.data (), rem.size (), stream_stdout,
                               out_str, flt);
        }
    }
  flt->pending_len = 0U;
  flt->utf8_buf.flush_remaining (stream_stdout, out_str);
}

} /* namespace */

bool
filter_thought_pieces (const std::vector<std::string_view> &pieces,
                       std::string *out_visible) noexcept
{
  if (out_visible == nullptr)
    {
      return false;
    }
  out_visible->clear ();
  thought_stream_filter flt = thought_stream_filter ();
  for (const std::string_view piece : pieces)
    {
      process_thought_stream_piece (&flt, piece, false, out_visible);
      if (flt.hit_stop)
        {
          break;
        }
    }
  finish_thought_stream_filter (&flt, false, out_visible);
  return true;
}

bool
parse_kv_cache_type (std::string_view text, kv_cache_type *out_type) noexcept
{
  if (out_type == nullptr)
    {
      return false;
    }
  if (text == "f16" || text == "F16" || text == "fp16")
    {
      *out_type = kv_cache_type::f16;
      return true;
    }
  if (text == "q8_0" || text == "Q8_0" || text == "q8")
    {
      *out_type = kv_cache_type::q8_0;
      return true;
    }
  if (text == "q4_0" || text == "Q4_0" || text == "q4")
    {
      *out_type = kv_cache_type::q4_0;
      return true;
    }
  return false;
}

const char *
kv_cache_type_name (kv_cache_type type) noexcept
{
  switch (type)
    {
    case kv_cache_type::f16:
      return "f16";
    case kv_cache_type::q8_0:
      return "q8_0";
    case kv_cache_type::q4_0:
      return "q4_0";
    }
  return "f16";
}

inference_engine::inference_engine () noexcept
    : config_ (), topo_ (), weight_map_ (), pinned_staging_ (),
      override_patterns_ (), tensor_overrides_ (), turn_prompt_buf_ (),
      token_buf_ (), kv_tokens_ (), pinned_sys_state_ (), decode_piece_buf_ (),
      model_arch_ ()
{
  decode_piece_buf_.fill ('\0');
}

inference_engine::~inference_engine () noexcept
{
  shutdown ();
}

bool
inference_engine::initialize (const engine_config &cfg)
{
  shutdown ();
  config_ = cfg;

#if defined (__linux__)
  (void)setenv ("OMP_WAIT_POLICY", "ACTIVE", 0);
  (void)setenv ("GOMP_SPINCOUNT", "30000000", 0);
#endif

  llama_log_set (silent_llama_log_callback, nullptr);
  llama_backend_init ();

  topo_ = detect_cpu_topology ();

  const int32_t default_phys_threads
      = static_cast<int32_t> (topo_.all_physical_cpus.size ());

  active_threads_decode_ = (config_.n_threads > 0)
                               ? config_.n_threads
                               : std::max (1, default_phys_threads);
  active_threads_prefill_
      = (config_.n_threads_batch > 0)
            ? config_.n_threads_batch
            : ((config_.n_threads > 0)
                   ? config_.n_threads
                   : std::max (active_threads_decode_, default_phys_threads));

  if (config_.pin_pcores)
    {
      if (pin_calling_thread_to_pcores (topo_, active_threads_decode_))
        {
          p_cores_pinned_ = std::min (
              active_threads_decode_,
              static_cast<int32_t> (topo_.p_core_cpus.size ()));
          e_cores_pinned_ = std::clamp (
              active_threads_decode_ - p_cores_pinned_, 0,
              static_cast<int32_t> (topo_.e_core_cpus.size ()));
        }
    }

  /* 1. Direct mmap & mlock pipeline for model weights.  */
  if (config_.use_mmap)
    {
      (void)weight_map_.map_file (config_.model_path.c_str (),
                                  config_.use_mlock);
    }

  /* 2. Page-locked (pinned) host staging buffer for CPU-to-GPU transfers.  */
  (void)pinned_staging_.allocate (k_pinned_staging_bytes);

  /* 3. Inspect GGUF architecture and VRAM budget for Smart Hybrid
        Offloading.  */
  const gguf_header_probe probe = probe_gguf_file (config_.model_path);
  model_arch_ = probe.architecture;
  is_gemma_arch_ = (model_arch_.find ("gemma") != std::string::npos);
  is_qwen3_arch_ = (model_arch_.find ("qwen3") != std::string::npos
                    && model_arch_ != "qwen3next");

  const int32_t detected_layers
      = (probe.block_count > 0U) ? static_cast<int32_t> (probe.block_count)
                                 : 32;
  total_model_layers_ = detected_layers;

#if defined (MILDA_ENABLE_CUDA)
  const bool gpu_supported = llama_supports_gpu_offload ();
#else
  const bool gpu_supported = false;
#endif
  std::size_t vram_used = 0U;
  std::size_t vram_total = 0U;
  const bool has_gpu
      = gpu_supported && probe_vram_metrics (&vram_used, &vram_total);
  const std::size_t vram_free
      = (has_gpu && vram_total > vram_used) ? (vram_total - vram_used) : 0U;

  int32_t effective_ngl = config_.n_gpu_layers;
  int32_t effective_cpu_moe = (config_.n_cpu_moe > 0) ? config_.n_cpu_moe : 0;
  int32_t effective_cpu_ffn = (config_.n_cpu_ffn > 0) ? config_.n_cpu_ffn : 0;

  if (!has_gpu || effective_ngl == 0)
    {
      /* Seamless fallback to 100% CPU execution.  */
      effective_ngl = 0;
      effective_cpu_moe = 0;
      effective_cpu_ffn = 0;
    }
  else
    {
      const bool is_moe = (probe.expert_count > 0U);
      const std::size_t safe_vram_budget
          = (vram_free > (768ULL << 20U)) ? (vram_free - (768ULL << 20U)) : 0U;

      if (effective_ngl < 0)
        {
          effective_ngl = 99;
        }

      /* Smart Hybrid Offloading when VRAM is constrained: keep KV cache and
         Attention tensors pinned in fast VRAM while selectively offloading
         heavy FFN / MoE expert tensors to System RAM + CPU compute.  */
      if (is_moe && probe.file_size_bytes > safe_vram_budget)
        {
          if (probe.file_size_bytes > (30ULL << 30U))
            {
              const std::size_t used_mib = vram_used >> 20U;
              const std::size_t free_mib = vram_free >> 20U;
              constexpr std::size_t k_target_max_mib = 7150U;
              constexpr std::size_t k_base_shared_kv_mib = 2050U;
              constexpr std::size_t k_per_moe_layer_mib = 875U;
              const std::size_t cap_budget
                  = (used_mib < k_target_max_mib)
                        ? (k_target_max_mib - used_mib)
                        : 0U;
              const std::size_t free_budget
                  = (free_mib > 300U) ? (free_mib - 300U) : 0U;
              const std::size_t avail_vram_mib
                  = std::min (cap_budget, free_budget);
              int32_t max_gpu_moe = 0;
              if (avail_vram_mib > k_base_shared_kv_mib)
                {
                  max_gpu_moe = static_cast<int32_t> (
                      (avail_vram_mib - k_base_shared_kv_mib)
                      / k_per_moe_layer_mib);
                }
              const int32_t min_safe_moe = std::clamp (
                  detected_layers - max_gpu_moe, 44, detected_layers);
              if (config_.n_cpu_moe < 0 || effective_cpu_moe < min_safe_moe)
                {
                  effective_cpu_moe = min_safe_moe;
                }
            }
          else if (config_.n_cpu_moe < 0)
            {
              effective_cpu_moe = std::min (detected_layers, 38);
            }
        }
      else if (!is_moe && config_.n_cpu_ffn < 0
               && probe.file_size_bytes > safe_vram_budget
               && safe_vram_budget > 0U)
        {
          /* Dense model larger than VRAM: offload dense FFN tensors for the
             first N layers and cap GPU layers if needed to prevent OOM.  */
          const std::size_t excess_bytes
              = probe.file_size_bytes - safe_vram_budget;
          const std::size_t bytes_per_layer
              = std::max<std::size_t> (1U, probe.file_size_bytes
                                               / static_cast<std::size_t> (
                                                   detected_layers));
          const int32_t layers_to_shed = static_cast<int32_t> (
              (excess_bytes + bytes_per_layer - 1U) / bytes_per_layer);
          if (probe.file_size_bytes > (10ULL << 30U))
            {
              effective_ngl = std::clamp (detected_layers - layers_to_shed, 1,
                                          17);
            }
          else
            {
              effective_cpu_ffn
                  = std::clamp (layers_to_shed + 2, 1, detected_layers);
            }
        }
    }

  override_patterns_.clear ();
  tensor_overrides_.clear ();

  ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type ();
  if (effective_ngl > 0 && (effective_cpu_moe > 0 || effective_cpu_ffn > 0))
    {
      const int32_t count_moe
          = std::min (effective_cpu_moe, detected_layers);
      const int32_t count_ffn
          = std::min (effective_cpu_ffn, detected_layers);
      const auto total_patterns
          = static_cast<std::size_t> (count_moe + count_ffn);
      override_patterns_.reserve (total_patterns);
      tensor_overrides_.reserve (total_patterns + 1U);

      for (int32_t i = 0; i < count_moe; ++i)
        {
          override_patterns_.push_back (
              "blk\\." + std::to_string (i)
              + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        }
      for (int32_t i = 0; i < count_ffn; ++i)
        {
          override_patterns_.push_back (
              "blk\\." + std::to_string (i) + "\\.ffn_(up|down|gate)\\.");
        }
      for (const std::string &pat : override_patterns_)
        {
          llama_model_tensor_buft_override ov
              = llama_model_tensor_buft_override ();
          ov.pattern = pat.c_str ();
          ov.buft = cpu_buft;
          tensor_overrides_.push_back (ov);
        }
      llama_model_tensor_buft_override term
          = llama_model_tensor_buft_override ();
      term.pattern = nullptr;
      term.buft = nullptr;
      tensor_overrides_.push_back (term);
    }

  llama_model_params mparams = llama_model_default_params ();
  mparams.n_gpu_layers = effective_ngl;
  mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
  mparams.use_extra_bufts = !config_.use_mmap;
  mparams.no_host = false;

  if (config_.use_mmap)
    {
      mparams.load_mode = config_.use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK
                                            : LLAMA_LOAD_MODE_MMAP;
    }
  else
    {
      mparams.load_mode
          = config_.use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
    }

  if (!tensor_overrides_.empty ())
    {
      mparams.tensor_buft_overrides = tensor_overrides_.data ();
    }

  model_ = llama_model_load_from_file (config_.model_path.c_str (), mparams);
  if (model_ == nullptr && effective_ngl > 0)
    {
      /* Seamless automatic fallback to 100% CPU execution on GPU failure.  */
      effective_ngl = 0;
      effective_cpu_moe = 0;
      effective_cpu_ffn = 0;
      mparams.n_gpu_layers = 0;
      mparams.tensor_buft_overrides = nullptr;
      model_
          = llama_model_load_from_file (config_.model_path.c_str (), mparams);
    }
  if (model_ == nullptr)
    {
      return false;
    }

#if defined (__linux__)
  if ((effective_cpu_moe > 0 || effective_cpu_ffn > 0
       || probe.file_size_bytes > (10ULL << 30U))
      && !config_.model_path.empty ())
    {
      std::error_code canon_ec;
      const std::filesystem::path canon_p
          = std::filesystem::canonical (config_.model_path, canon_ec);
      const std::string canon_str
          = canon_ec ? config_.model_path : canon_p.string ();
      const std::size_t avail_ram = query_available_ram_bytes ();
      std::size_t budget_bytes
          = (avail_ram > (2560ULL << 20U))
                ? (avail_ram - (2560ULL << 20U))
                : 0U;
      struct mapped_vma_seg
      {
        unsigned long long start_addr = 0ULL;
        unsigned long long end_addr = 0ULL;
        unsigned long long file_off = 0ULL;
      };
      std::vector<mapped_vma_seg> vma_segs;
      std::vector<std::pair<unsigned long long, unsigned long long>>
          mapped_ranges;
      std::ifstream maps ("/proc/self/maps");
      std::string line;
      while (std::getline (maps, line))
        {
          if (line.find (config_.model_path) == std::string::npos
              && line.find (canon_str) == std::string::npos)
            {
              continue;
            }
          unsigned long long start_addr = 0ULL;
          unsigned long long end_addr = 0ULL;
          unsigned long long file_off = 0ULL;
          if (std::sscanf (line.c_str (), "%llx-%llx %*s %llx", &start_addr,
                           &end_addr, &file_off)
                  == 3
              && end_addr > start_addr)
            {
              const auto seg_len
                  = static_cast<std::size_t> (end_addr - start_addr);
              void *seg_ptr = reinterpret_cast<void *> (start_addr);
              (void)madvise (seg_ptr, seg_len, MADV_NORMAL);
              mapped_vma_seg vseg = mapped_vma_seg ();
              vseg.start_addr = start_addr;
              vseg.end_addr = end_addr;
              vseg.file_off = file_off;
              vma_segs.push_back (vseg);
              mapped_ranges.emplace_back (file_off,
                                          file_off + (end_addr - start_addr));
              if (seg_len >= (256ULL << 20U))
                {
                  const std::size_t pop_len = std::min (seg_len, budget_bytes);
                  if (pop_len > 0U && seg_len + (6ULL << 30U) <= budget_bytes)
                    {
                      (void)madvise (seg_ptr, pop_len, MADV_WILLNEED);
                      budget_bytes = (budget_bytes > pop_len)
                                         ? (budget_bytes - pop_len)
                                         : 0U;
                    }
                }
            }
        }
      if (!mapped_ranges.empty () && effective_ngl > 0)
        {
          std::sort (mapped_ranges.begin (), mapped_ranges.end ());
          const int fd = open (canon_str.c_str (), O_RDONLY | O_CLOEXEC);
          if (fd >= 0)
            {
              unsigned long long prev_end = 0ULL;
              for (const auto &rng : mapped_ranges)
                {
                  if (rng.first > prev_end)
                    {
                      const unsigned long long gap_start
                          = (prev_end + 4095ULL) & ~4095ULL;
                      const unsigned long long gap_end
                          = rng.first & ~4095ULL;
                      if (gap_end > gap_start)
                        {
                          (void)posix_fadvise (
                              fd, static_cast<off_t> (gap_start),
                              static_cast<off_t> (gap_end - gap_start),
                              POSIX_FADV_DONTNEED);
                        }
                    }
                  if (rng.second > prev_end)
                    {
                      prev_end = rng.second;
                    }
                }
              if (probe.file_size_bytes > prev_end)
                {
                  const unsigned long long gap_start
                      = (prev_end + 4095ULL) & ~4095ULL;
                  const unsigned long long gap_end
                      = static_cast<unsigned long long> (probe.file_size_bytes)
                        & ~4095ULL;
                  if (gap_end > gap_start)
                    {
                      (void)posix_fadvise (
                          fd, static_cast<off_t> (gap_start),
                          static_cast<off_t> (gap_end - gap_start),
                          POSIX_FADV_DONTNEED);
                    }
                }

              /* Evict GPU-offloaded tensors lying inside CPU-mapped VMAs so
                 100% of host DDR4 page-cache capacity is reserved for CPU
                 MoE/FFN tensors.  */
              if (effective_cpu_moe > 0 || effective_cpu_ffn > 0)
                {
                  gguf_init_params ginit = { true, nullptr };
                  gguf_context *gctx
                      = gguf_init_from_file (canon_str.c_str (), ginit);
                  if (gctx != nullptr)
                    {
                      const auto data_off = static_cast<unsigned long long> (
                          gguf_get_data_offset (gctx));
                      const int64_t n_tensors = gguf_get_n_tensors (gctx);
                      for (int64_t ti = 0; ti < n_tensors; ++ti)
                        {
                          const char *tname = gguf_get_tensor_name (gctx, ti);
                          if (tname == nullptr)
                            {
                              continue;
                            }
                          bool on_cpu = false;
                          if (std::strncmp (tname, "token_embd", 10U) == 0
                              || std::strncmp (tname, "token_types", 11U) == 0
                              || std::strncmp (tname, "rope_freqs", 10U) == 0)
                            {
                              on_cpu = true;
                            }
                          else if (std::strncmp (tname, "blk.", 4U) == 0)
                            {
                              const int32_t blk_id
                                  = std::atoi (tname + 4U);
                              if (blk_id >= effective_ngl)
                                {
                                  on_cpu = true;
                                }
                              else if (blk_id < effective_cpu_moe
                                       && std::strstr (tname, "_exps")
                                              != nullptr)
                                {
                                  on_cpu = true;
                                }
                              else if (blk_id < effective_cpu_ffn
                                       && (std::strstr (tname, ".ffn_up.")
                                               != nullptr
                                           || std::strstr (tname, ".ffn_down.")
                                                  != nullptr
                                           || std::strstr (tname, ".ffn_gate.")
                                                  != nullptr))
                                {
                                  on_cpu = true;
                                }
                            }
                          if (!on_cpu)
                            {
                              const auto t_off
                                  = data_off
                                    + static_cast<unsigned long long> (
                                        gguf_get_tensor_offset (gctx, ti));
                              const auto t_sz
                                  = static_cast<unsigned long long> (
                                      gguf_get_tensor_size (gctx, ti));
                              const unsigned long long p_start
                                  = (t_off + 4095ULL) & ~4095ULL;
                              const unsigned long long p_end
                                  = (t_off + t_sz) & ~4095ULL;
                              if (p_end > p_start)
                                {
                                  for (const mapped_vma_seg &vs : vma_segs)
                                    {
                                      const unsigned long long vs_end_off
                                          = vs.file_off
                                            + (vs.end_addr - vs.start_addr);
                                      const unsigned long long i_start
                                          = std::max (p_start, vs.file_off);
                                      const unsigned long long i_end
                                          = std::min (p_end, vs_end_off);
                                      if (i_end > i_start)
                                        {
                                          void *m_ptr
                                              = reinterpret_cast<void *> (
                                                  vs.start_addr
                                                  + (i_start - vs.file_off));
                                          (void)madvise (
                                              m_ptr,
                                              static_cast<std::size_t> (
                                                  i_end - i_start),
                                              MADV_DONTNEED);
                                        }
                                    }
                                  (void)posix_fadvise (
                                      fd, static_cast<off_t> (p_start),
                                      static_cast<off_t> (p_end - p_start),
                                      POSIX_FADV_DONTNEED);
                                }
                            }
                        }
                      gguf_free (gctx);
                    }
                }
              (void)close (fd);
            }
        }
    }
#endif

  total_model_layers_ = llama_model_n_layer (model_);
  offloaded_gpu_layers_
      = (effective_ngl > 0)
            ? std::min (effective_ngl, total_model_layers_)
            : 0;
  hybrid_cpu_ffn_layers_ = std::max (effective_cpu_moe, effective_cpu_ffn);

  vocab_ = llama_model_get_vocab (model_);
  if (vocab_ == nullptr)
    {
      shutdown ();
      return false;
    }

  const char *raw_tmpl = llama_model_chat_template (model_, nullptr);
  const std::string_view chat_tmpl
      = (raw_tmpl != nullptr) ? std::string_view (raw_tmpl)
                              : std::string_view ();
  if (chat_tmpl.find ("<|turn>") != std::string_view::npos
      || chat_tmpl.find ("<|channel>thought") != std::string_view::npos)
    {
      is_gemma_arch_ = true;
      is_qwen3_arch_ = false;
    }
  else if (chat_tmpl.find ("<think>") != std::string_view::npos
           && model_arch_ != "qwen3next")
    {
      is_qwen3_arch_ = true;
    }

  llama_context_params cparams = llama_context_default_params ();
  uint32_t effective_ctx = config_.n_ctx;
  if (effective_ctx > 8192U && query_available_ram_bytes () < (8ULL << 30U))
    {
      effective_ctx = 8192U;
    }
  const bool is_hybrid_cpu = (effective_cpu_moe > 0 || effective_cpu_ffn > 0);
  cparams.n_ctx = effective_ctx;
  cparams.n_batch = is_hybrid_cpu ? std::min (config_.n_batch, 512U)
                                  : config_.n_batch;
  cparams.n_ubatch = is_hybrid_cpu ? std::min (cparams.n_batch, 256U)
                                   : config_.n_ubatch;
  cparams.n_seq_max = 1U;
  cparams.n_threads = active_threads_decode_;
  cparams.n_threads_batch = active_threads_prefill_;
  cparams.type_k = to_ggml_kv_type (config_.cache_type_k);
  cparams.type_v = to_ggml_kv_type (config_.cache_type_v);
  cparams.flash_attn_type = config_.flash_attn
                                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  cparams.offload_kqv = (effective_ngl > 0);
  cparams.op_offload = (effective_ngl > 0 && !is_hybrid_cpu);
  cparams.no_perf = false;
  cparams.kv_unified = true;
  cparams.swa_full = false;

  ctx_ = llama_init_from_model (model_, cparams);
  if (ctx_ == nullptr && effective_ngl > 0)
    {
      /* Fallback to 100% CPU execution if GPU context/KV allocation runs
         out of VRAM.  */
      llama_model_free (model_);
      model_ = nullptr;
      effective_ngl = 0;
      effective_cpu_moe = 0;
      effective_cpu_ffn = 0;
      mparams.n_gpu_layers = 0;
      mparams.tensor_buft_overrides = nullptr;
      cparams.offload_kqv = false;
      cparams.op_offload = false;
      model_
          = llama_model_load_from_file (config_.model_path.c_str (), mparams);
      if (model_ != nullptr)
        {
          vocab_ = llama_model_get_vocab (model_);
          offloaded_gpu_layers_ = 0;
          hybrid_cpu_ffn_layers_ = 0;
          ctx_ = llama_init_from_model (model_, cparams);
        }
    }
  if (ctx_ == nullptr)
    {
      shutdown ();
      return false;
    }

  n_ctx_ = static_cast<int32_t> (llama_n_ctx (ctx_));

  /* Configure and attach 12-physical-core-pinned ggml threadpool for decode
     and batch.  */
  const int32_t pool_threads
      = std::max (active_threads_decode_, active_threads_prefill_);
  ggml_threadpool_params tp_decode_params
      = ggml_threadpool_params_default (pool_threads);
  tp_decode_params.poll = 100;
  tp_decode_params.paused = true;
  if (config_.pin_pcores && !topo_.all_physical_cpus.empty ())
    {
      std::memset (tp_decode_params.cpumask, 0,
                   sizeof (tp_decode_params.cpumask));
      const std::size_t pin_cnt = std::min (
          topo_.all_physical_cpus.size (),
          static_cast<std::size_t> (pool_threads));
      for (std::size_t i = 0U; i < pin_cnt; ++i)
        {
          const int32_t cpu_id = topo_.all_physical_cpus[i];
          if (cpu_id >= 0 && cpu_id < GGML_MAX_N_THREADS)
            {
              tp_decode_params.cpumask[cpu_id] = true;
            }
        }
      if (pool_threads
          > static_cast<int32_t> (topo_.all_physical_cpus.size ()))
        {
          const int32_t max_log
              = std::min (topo_.logical_cpu_count, GGML_MAX_N_THREADS);
          std::size_t total_set = pin_cnt;
          const auto req_dec
              = static_cast<std::size_t> (pool_threads);
          for (int32_t c = 0; c < max_log && total_set < req_dec; ++c)
            {
              if (!tp_decode_params.cpumask[c])
                {
                  tp_decode_params.cpumask[c] = true;
                  ++total_set;
                }
            }
          tp_decode_params.strict_cpu = false;
        }
      else
        {
          tp_decode_params.strict_cpu = true;
        }
      tp_decode_params.prio = GGML_SCHED_PRIO_NORMAL;
    }
  threadpool_decode_ = ggml_threadpool_new(&tp_decode_params);
  threadpool_batch_ = nullptr;

  if (threadpool_decode_ != nullptr)
    {
      llama_attach_threadpool (ctx_, threadpool_decode_, threadpool_decode_);
    }
  llama_set_n_threads (ctx_, active_threads_decode_, active_threads_prefill_);

  /* Initialize reusable sampler chain with per-language tuning.  */
  active_turn_lang_ = (config_.language == MildaLang::Auto)
                          ? MildaLang::English
                          : config_.language;
  session_lang_locked_ = (config_.language != MildaLang::Auto);
  if (!rebuild_sampler (active_turn_lang_))
    {
      shutdown ();
      return false;
    }

  /* Pre-allocate all hot-loop buffers once at startup.  */
  turn_prompt_buf_.reserve (k_turn_prompt_reserve);
  token_buf_.resize (static_cast<std::size_t> (n_ctx_), 0);
  kv_tokens_.resize (static_cast<std::size_t> (n_ctx_), 0);
  decode_piece_buf_.fill ('\0');

  /* Evaluate and pin the active system prompt at KV index 0.  */
  if (!pin_system_prompt ())
    {
      shutdown ();
      return false;
    }

  return true;
}

bool
inference_engine::rebuild_sampler (MildaLang target_lang) noexcept
{
  const LangMetadata &meta = get_lang_metadata (target_lang);
  active_repeat_penalty_ = config_.custom_repeat_penalty
                               ? config_.repeat_penalty
                               : meta.repeat_penalty;
  active_top_p_ = config_.custom_top_p ? config_.top_p : meta.top_p;
  active_sampler_lang_ = target_lang;

  if (vocab_ == nullptr)
    {
      return true;
    }

  if (sampler_ != nullptr)
    {
      llama_sampler_free (sampler_);
      sampler_ = nullptr;
    }

  llama_sampler_chain_params sparams = llama_sampler_chain_default_params ();
  sparams.no_perf = true;
  sampler_ = llama_sampler_chain_init (sparams);
  if (sampler_ == nullptr)
    {
      return false;
    }

  if (active_repeat_penalty_ > 1.0F)
    {
      const int32_t n_vocab = llama_vocab_n_tokens (vocab_);
      llama_sampler_chain_add (
          sampler_, llama_sampler_init_penalties (
                        n_vocab, 64, active_repeat_penalty_, 0.0F, 0.0F));
    }
  if (config_.temperature <= 0.0F)
    {
      llama_sampler_chain_add (sampler_, llama_sampler_init_greedy ());
    }
  else
    {
      if (config_.top_k > 0)
        {
          llama_sampler_chain_add (sampler_,
                                   llama_sampler_init_top_k (config_.top_k));
        }
      if (active_top_p_ > 0.0F && active_top_p_ < 1.0F)
        {
          llama_sampler_chain_add (
              sampler_, llama_sampler_init_top_p (active_top_p_, 1U));
        }
      if (config_.min_p > 0.0F)
        {
          llama_sampler_chain_add (
              sampler_, llama_sampler_init_min_p (config_.min_p, 1U));
        }
      llama_sampler_chain_add (sampler_,
                               llama_sampler_init_temp (config_.temperature));
      llama_sampler_chain_add (sampler_,
                               llama_sampler_init_dist (config_.seed));
    }
  return true;
}

void
inference_engine::shutdown () noexcept
{
  if (ctx_ != nullptr)
    {
      llama_detach_threadpool (ctx_);
    }
  if (sampler_ != nullptr)
    {
      llama_sampler_free (sampler_);
      sampler_ = nullptr;
    }
  if (ctx_ != nullptr)
    {
      llama_free (ctx_);
      ctx_ = nullptr;
    }
  if (threadpool_decode_ != nullptr)
    {
      ggml_threadpool_free (threadpool_decode_);
      threadpool_decode_ = nullptr;
    }
  if (threadpool_batch_ != nullptr)
    {
      ggml_threadpool_free (threadpool_batch_);
      threadpool_batch_ = nullptr;
    }
  if (model_ != nullptr)
    {
      llama_model_free (model_);
      model_ = nullptr;
    }
  vocab_ = nullptr;
  pinned_staging_.release ();
  weight_map_.unmap_file ();
  n_ctx_ = 0;
  n_keep_ = 0;
  n_past_ = 0;
  p_cores_pinned_ = 0;
  e_cores_pinned_ = 0;
  turn_count_ = 0ULL;
}

bool
inference_engine::pin_system_prompt () noexcept
{
  struct tp_guard
  {
    ggml_threadpool_t tp;
    explicit tp_guard (ggml_threadpool_t p) noexcept : tp (p)
    {
    }
    ~tp_guard ()
    {
      if (tp != nullptr)
        {
          ggml_threadpool_pause (tp);
        }
    }
  } guard (threadpool_decode_);

  const std::string_view sys_prompt = get_system_prompt (config_.language);
  turn_prompt_buf_.clear ();
  if (is_gemma_arch_)
    {
      turn_prompt_buf_.append ("<|turn>system\n");
      turn_prompt_buf_.append (sys_prompt);
      turn_prompt_buf_.append ("<turn|>\n");
    }
  else
    {
      turn_prompt_buf_.append ("<|im_start|>system\n");
      turn_prompt_buf_.append (sys_prompt);
      turn_prompt_buf_.append ("<|im_end|>\n");
    }

  const bool add_bos = llama_vocab_get_add_bos (vocab_);
  const int32_t n_sys_tokens
      = tokenize_into_preallocated (turn_prompt_buf_, add_bos);
  if (n_sys_tokens <= 0 || n_sys_tokens >= n_ctx_ - 64)
    {
      return false;
    }

  llama_memory_t mem = llama_get_memory (ctx_);
  llama_memory_clear (mem, true);

  if (!decode_tokens_chunked (token_buf_.data (), n_sys_tokens))
    {
      return false;
    }

  std::memcpy (kv_tokens_.data (), token_buf_.data (),
               static_cast<std::size_t> (n_sys_tokens) * sizeof (int32_t));
  n_keep_ = n_sys_tokens;
  n_past_ = n_sys_tokens;
  turn_count_ = 0ULL;
  session_lang_locked_ = (config_.language != MildaLang::Auto);

  /* Snapshot the sequence state for recurrent/hybrid architectures where
     partial sequence removal is unsupported.  */
  const std::size_t seq_state_sz = llama_state_seq_get_size (ctx_, 0);
  if (seq_state_sz > 0U && seq_state_sz <= (128ULL << 20U))
    {
      pinned_sys_state_.resize (seq_state_sz);
      const std::size_t copied = llama_state_seq_get_data (
          ctx_, pinned_sys_state_.data (), seq_state_sz, 0);
      if (copied != seq_state_sz)
        {
          pinned_sys_state_.clear ();
        }
    }

  return true;
}

int32_t
inference_engine::tokenize_into_preallocated (std::string_view text,
                                              bool add_special) noexcept
{
  if (vocab_ == nullptr || text.empty () || token_buf_.empty ())
    {
      return 0;
    }
  const int32_t cap = static_cast<int32_t> (token_buf_.size ());
  const int32_t n_tok = llama_tokenize (
      vocab_, text.data (), static_cast<int32_t> (text.size ()),
      token_buf_.data (), cap, add_special, true);
  if (n_tok < 0)
    {
      return -1;
    }
  return n_tok;
}

bool
inference_engine::decode_tokens_chunked (const int32_t *tokens,
                                         int32_t count) noexcept
{
  if (ctx_ == nullptr || tokens == nullptr || count <= 0)
    {
      return false;
    }

  const int32_t max_chunk = std::min (
      static_cast<int32_t> (config_.n_batch), k_default_prefill_chunk);
  int32_t offset = 0;

  while (offset < count)
    {
      const int32_t step = std::min (max_chunk, count - offset);
      /* llama_batch_get_one returns a stack-allocated descriptor pointing to
         our pre-allocated token buffer (zero heap allocations).  */
      llama_batch batch = llama_batch_get_one (
          const_cast<int32_t *> (tokens + offset), step);
      if (llama_decode (ctx_, batch) != 0)
        {
          return false;
        }
      offset += step;
    }
  return true;
}

void
inference_engine::format_turn_delta (std::string_view user_text) noexcept
{
  std::string_view lang_lock_note;
  if (config_.language == MildaLang::Auto)
    {
      if (active_turn_lang_ == MildaLang::English)
        {
          lang_lock_note = "Active language: English. Respond naturally "
                           "100% in English without mentioning this rule.";
        }
      else if (active_turn_lang_ == MildaLang::Spanish)
        {
          lang_lock_note = "Idioma activo: espa\303\261ol. Responde con "
                           "naturalidad 100% en espa\303\261ol sin mencionar "
                           "esta regla.";
        }
      else if (active_turn_lang_ == MildaLang::Russian)
        {
          lang_lock_note
              = "\320\220\320\272\321\202\320\270\320\262\320\275\321\213"
                "\320\271 \321\217\320\267\321\213\320\272: "
                "\321\200\321\203\321\201\321\201\320\272\320\270\320\271. "
                "\320\236\321\202\320\262\320\265\321\207\320\260\320\271 "
                "\320\265\321\201\321\202\320\265\321\201\321\202\320\262"
                "\320\265\320\275\320\275\320\276 \321\202\320\276\320\273"
                "\321\214\320\272\320\276 \320\277\320\276-\321\200\321\203"
                "\321\201\321\201\320\272\320\270 \320\261\320\265\320\267 "
                "\321\203\320\277\320\276\320\274\320\270\320\275\320\260"
                "\320\275\320\270\321\217 \320\277\321\200\320\260\320\262"
                "\320\270\320\273\320\260.";
        }
      else if (active_turn_lang_ == MildaLang::Esperanto)
        {
          lang_lock_note = "Aktiva lingvo: Esperanto. Respondu nature 100% "
                           "en Esperanto sen mencii \304\211i tiun regulon.";
        }
      else if (active_turn_lang_ == MildaLang::MandarinSimplified)
        {
          lang_lock_note
              = "\345\275\223\345\211\215\350\257\255\350\250\200\357\274"
                "\232\347\256\200\344\275\223\344\270\255\346\226\207\343"
                "\200\202\350\257\267\350\207\252\347\204\266\345\234\260"
                "\344\273\205\347\224\250\347\256\200\344\275\223\344\270"
                "\255\346\226\207\345\233\236\347\255\224\357\274\214\344"
                "\270\215\350\246\201\346\217\220\345\217\212\346\255\244"
                "\350\247\204\345\210\231\343\200\202";
        }
    }

  turn_prompt_buf_.clear ();
  if (is_gemma_arch_)
    {
      if (turn_count_ > 0ULL)
        {
          turn_prompt_buf_.append ("<turn|>\n");
        }
      turn_prompt_buf_.append ("<|turn>user\n");
      turn_prompt_buf_.append (user_text);
      turn_prompt_buf_.append ("<turn|>\n");
      if (!lang_lock_note.empty ())
        {
          turn_prompt_buf_.append ("<|turn>system\n");
          turn_prompt_buf_.append (lang_lock_note);
          turn_prompt_buf_.append ("<turn|>\n");
        }
      turn_prompt_buf_.append ("<|turn>model\n");
      turn_prompt_buf_.append ("<|channel>thought\n<channel|>");
    }
  else
    {
      if (turn_count_ > 0ULL)
        {
          turn_prompt_buf_.append ("<|im_end|>\n");
        }
      turn_prompt_buf_.append ("<|im_start|>user\n");
      turn_prompt_buf_.append (user_text);
      turn_prompt_buf_.append ("<|im_end|>\n");
      if (!lang_lock_note.empty ())
        {
          turn_prompt_buf_.append ("<|im_start|>system\n");
          turn_prompt_buf_.append (lang_lock_note);
          turn_prompt_buf_.append ("<|im_end|>\n");
        }
      turn_prompt_buf_.append ("<|im_start|>assistant\n");
      if (is_qwen3_arch_)
        {
          turn_prompt_buf_.append ("<think>\n\n</think>\n");
        }
    }
}

bool
inference_engine::shift_kv_cache_in_place (int32_t needed_tokens) noexcept
{
  if (ctx_ == nullptr || n_past_ <= n_keep_)
    {
      return false;
    }

  llama_memory_t mem = llama_get_memory (ctx_);
  const int32_t removable = n_past_ - n_keep_;
  const int32_t n_discard
      = std::clamp (std::max (needed_tokens, removable / 2), 1, removable);

  if (llama_memory_can_shift (mem)
      && llama_memory_seq_rm (mem, 0, n_keep_, n_keep_ + n_discard))
    {
      llama_memory_seq_add (mem, 0, n_keep_ + n_discard, -1, -n_discard);
      const int32_t remaining = n_past_ - (n_keep_ + n_discard);
      if (remaining > 0)
        {
          std::memmove (&kv_tokens_[static_cast<std::size_t> (n_keep_)],
                        &kv_tokens_[static_cast<std::size_t> (n_keep_
                                                              + n_discard)],
                        static_cast<std::size_t> (remaining)
                            * sizeof (int32_t));
        }
      n_past_ -= n_discard;
      return true;
    }

  /* Fallback if the architecture does not support partial KV shifts:
     truncate back to the pinned Esperanto system prompt.  */
  return clear_conversation ();
}

bool
inference_engine::clear_conversation () noexcept
{
  if (ctx_ == nullptr || n_keep_ <= 0)
    {
      return false;
    }

  llama_memory_t mem = llama_get_memory (ctx_);
  if (llama_memory_seq_rm (mem, 0, n_keep_, -1))
    {
      n_past_ = n_keep_;
      turn_count_ = 0ULL;
      session_lang_locked_ = (config_.language != MildaLang::Auto);
      if (sampler_ != nullptr)
        {
          llama_sampler_reset (sampler_);
        }
      return true;
    }

  if (!pinned_sys_state_.empty ())
    {
      llama_memory_clear (mem, true);
      if (llama_state_seq_set_data (ctx_, pinned_sys_state_.data (),
                                    pinned_sys_state_.size (), 0)
          > 0U)
        {
          n_past_ = n_keep_;
          turn_count_ = 0ULL;
          session_lang_locked_ = (config_.language != MildaLang::Auto);
          if (sampler_ != nullptr)
            {
              llama_sampler_reset (sampler_);
            }
          return true;
        }
    }

  return pin_system_prompt ();
}

bool
inference_engine::save_session (const char *filepath) const noexcept
{
  if (ctx_ == nullptr || filepath == nullptr || filepath[0] == '\0'
      || n_past_ <= 0)
    {
      return false;
    }

  std::FILE *fp = std::fopen (filepath, "wb");
  if (fp == nullptr)
    {
      return false;
    }

  const uint32_t k_magic = 0x4D494C44U;
  const int32_t saved_past = n_past_;
  const uint32_t saved_lang = static_cast<uint32_t> (active_turn_lang_);
  const uint32_t saved_locked = session_lang_locked_ ? 1U : 0U;
  const uint64_t saved_turns = turn_count_;
  const auto token_cnt = static_cast<std::size_t> (n_past_);

  bool ok = (std::fwrite (&k_magic, sizeof (k_magic), 1U, fp) == 1U)
            && (std::fwrite (&saved_past, sizeof (saved_past), 1U, fp) == 1U)
            && (std::fwrite (&saved_lang, sizeof (saved_lang), 1U, fp) == 1U)
            && (std::fwrite (&saved_locked, sizeof (saved_locked), 1U, fp)
                == 1U)
            && (std::fwrite (&saved_turns, sizeof (saved_turns), 1U, fp) == 1U)
            && (std::fwrite (kv_tokens_.data (), sizeof (int32_t), token_cnt,
                             fp)
                == token_cnt);
  std::fclose (fp);
  return ok;
}

bool
inference_engine::load_session (const char *filepath) noexcept
{
  if (ctx_ == nullptr || filepath == nullptr || filepath[0] == '\0'
      || n_keep_ <= 0)
    {
      return false;
    }

  std::FILE *fp = std::fopen (filepath, "rb");
  if (fp == nullptr)
    {
      return false;
    }

  uint32_t magic = 0U;
  int32_t saved_past = 0;
  uint32_t saved_lang = 0U;
  uint32_t saved_locked = 0U;
  uint64_t saved_turns = 0ULL;

  bool hdr_ok = (std::fread (&magic, sizeof (magic), 1U, fp) == 1U)
                && (std::fread (&saved_past, sizeof (saved_past), 1U, fp)
                    == 1U)
                && (std::fread (&saved_lang, sizeof (saved_lang), 1U, fp)
                    == 1U)
                && (std::fread (&saved_locked, sizeof (saved_locked), 1U, fp)
                    == 1U)
                && (std::fread (&saved_turns, sizeof (saved_turns), 1U, fp)
                    == 1U);
  if (!hdr_ok || magic != 0x4D494C44U || saved_past < n_keep_
      || saved_past >= n_ctx_
      || static_cast<std::size_t> (saved_past) > kv_tokens_.size ())
    {
      std::fclose (fp);
      return false;
    }

  if (n_past_ != n_keep_)
    {
      (void)clear_conversation ();
    }

  const auto loaded_tokens = static_cast<std::size_t> (saved_past);
  if (std::fread (kv_tokens_.data (), sizeof (int32_t), loaded_tokens, fp)
      != loaded_tokens)
    {
      std::fclose (fp);
      (void)pin_system_prompt ();
      return false;
    }
  std::fclose (fp);

  if (saved_past > n_keep_)
    {
      struct tp_guard
      {
        ggml_threadpool_t tp;
        explicit tp_guard (ggml_threadpool_t p) noexcept : tp (p)
        {
        }
        ~tp_guard ()
        {
          if (tp != nullptr)
            {
              ggml_threadpool_pause (tp);
            }
        }
      } guard (threadpool_decode_);

      if (!decode_tokens_chunked (kv_tokens_.data () + n_keep_,
                                  saved_past - n_keep_))
        {
          (void)pin_system_prompt ();
          return false;
        }
    }

  n_past_ = saved_past;
  active_turn_lang_ = static_cast<MildaLang> (saved_lang);
  session_lang_locked_ = (saved_locked != 0U);
  turn_count_ = (saved_turns > 0ULL) ? saved_turns
                                     : ((n_past_ > n_keep_) ? 1ULL : 0ULL);
  if (sampler_ != nullptr)
    {
      llama_sampler_reset (sampler_);
    }
  return true;
}

bool
inference_engine::generate_turn (std::string_view user_text,
                                 turn_metrics *out_metrics) noexcept
{
  if (ctx_ == nullptr || vocab_ == nullptr || sampler_ == nullptr)
    {
      return false;
    }

  turn_metrics metrics = turn_metrics ();

  MildaLang turn_lang = config_.language;
  if (config_.language == MildaLang::Auto)
    {
      MildaLang req_lang = active_turn_lang_;
      if (!session_lang_locked_)
        {
          turn_lang = detect_prompt_language (user_text);
          session_lang_locked_ = true;
        }
      else if (detect_explicit_language_request (user_text, &req_lang))
        {
          turn_lang = req_lang;
        }
      else
        {
          turn_lang = active_turn_lang_;
        }
    }
  active_turn_lang_ = turn_lang;
  if (turn_lang != active_sampler_lang_)
    {
      if (!rebuild_sampler (turn_lang))
        {
          return false;
        }
    }

  if (config_.pin_pcores)
    {
      (void)pin_calling_thread_to_pcores (topo_, active_threads_decode_);
    }

  struct tp_guard
  {
    ggml_threadpool_t tp;
    explicit tp_guard (ggml_threadpool_t p) noexcept : tp (p)
    {
    }
    ~tp_guard ()
    {
      if (tp != nullptr)
        {
          ggml_threadpool_pause (tp);
        }
    }
  } guard (threadpool_decode_);

  /* 1. Format only the newly appended user turn delta into pre-allocated
        turn_prompt_buf_.  */
  format_turn_delta (user_text);

  /* 2. Tokenize delta into pre-allocated token_buf_.  */
  const int32_t n_delta_tokens
      = tokenize_into_preallocated (turn_prompt_buf_, false);
  if (n_delta_tokens <= 0)
    {
      return false;
    }
  metrics.prompt_tokens = static_cast<std::size_t> (n_delta_tokens);

  /* 3. Ensure KV cache has room for the prompt delta + generation headroom;
        perform in-place context shifting if needed.  */
  const int32_t min_headroom = std::min (config_.max_new_tokens, 256);
  if (n_past_ + n_delta_tokens + min_headroom >= n_ctx_)
    {
      const int32_t needed
          = (n_past_ + n_delta_tokens + min_headroom) - n_ctx_ + 64;
      if (!shift_kv_cache_in_place (needed))
        {
          return false;
        }
      ++metrics.kv_shifts_performed;
    }

  /* 4. Evaluate only the delta tokens into the KV cache.  */
  const uint64_t t_prefill_start = monotonic_time_ns ();
  if (!decode_tokens_chunked (token_buf_.data (), n_delta_tokens))
    {
      return false;
    }
  const uint64_t t_prefill_end = monotonic_time_ns ();
  metrics.prompt_eval_ms
      = static_cast<double> (t_prefill_end - t_prefill_start) / 1.0e6;

  if (n_past_ + n_delta_tokens <= n_ctx_)
    {
      std::memcpy (&kv_tokens_[static_cast<std::size_t> (n_past_)],
                   token_buf_.data (),
                   static_cast<std::size_t> (n_delta_tokens)
                       * sizeof (int32_t));
    }
  n_past_ += n_delta_tokens;
  ++turn_count_;

  /* 5. Zero-heap-allocation hot token generation loop with backup
        <|channel>thought...<channel|> and <think>...</think> stream
        suppression.  */
  const int32_t avail_headroom = std::max (1, n_ctx_ - n_past_ - 16);
  const int32_t max_steps = std::min (config_.max_new_tokens, avail_headroom);
  const int32_t max_total_steps = std::max (max_steps * 4, 1024);
  uint64_t t_gen_start = t_prefill_end;
  int32_t current_token = 0;
  int32_t gen_start_past = n_past_;
  int32_t pending_tag_start_pos = -1;
  bool any_suppressed_tokens = false;
  thought_stream_filter flt = thought_stream_filter ();

  for (int32_t step = 0; step < max_total_steps; ++step)
    {
      std::array<char, 128U> spec_buf = {};
      std::string_view spec_sv;
      std::string_view piece_sv;

      for (int32_t retry = 0; retry < 24; ++retry)
        {
          current_token = llama_sampler_sample (sampler_, ctx_, -1);
          const int32_t spec_len = llama_token_to_piece (
              vocab_, current_token, spec_buf.data (),
              static_cast<int32_t> (spec_buf.size ()), 0, true);
          spec_sv = std::string_view ();
          if (spec_len > 0)
            {
              spec_sv = std::string_view (
                  spec_buf.data (), static_cast<std::size_t> (spec_len));
            }
          else
            {
              const char *raw_tok_text
                  = llama_vocab_get_text (vocab_, current_token);
              if (raw_tok_text != nullptr)
                {
                  spec_sv = std::string_view (raw_tok_text);
                }
            }

          if (llama_vocab_is_eog (vocab_, current_token)
              || is_stop_token_text (spec_sv))
            {
              break;
            }

          const int32_t piece_len = llama_token_to_piece (
              vocab_, current_token, decode_piece_buf_.data (),
              static_cast<int32_t> (decode_piece_buf_.size ()), 0, false);
          piece_sv = std::string_view ();
          if (spec_sv == "<|channel>" || spec_sv == "<channel|>"
              || spec_sv == "<think>" || spec_sv == "</think>"
              || spec_sv == "<|think|>")
            {
              piece_sv = spec_sv;
            }
          else if (piece_len > 0)
            {
              piece_sv = std::string_view (
                  decode_piece_buf_.data (),
                  static_cast<std::size_t> (piece_len));
            }

          if (flt.mode == thought_filter_mode::normal
              && !piece_sv.starts_with ("<")
              && !is_cot_start_prefix (piece_sv)
              && is_token_violating_language_lock (piece_sv,
                                                   active_turn_lang_))
            {
              float *logits = llama_get_logits_ith (ctx_, -1);
              if (logits != nullptr && current_token >= 0)
                {
                  logits[current_token] = -1.0e9F;
                  continue;
                }
            }
          break;
        }

      llama_sampler_accept (sampler_, current_token);

      if (step == 0)
        {
          const uint64_t t_first_tok = monotonic_time_ns ();
          metrics.ttft_ms
              = static_cast<double> (t_first_tok - t_prefill_start) / 1.0e6;
          t_gen_start = t_first_tok;
        }

      if (llama_vocab_is_eog (vocab_, current_token)
          || is_stop_token_text (spec_sv))
        {
          break;
        }

      process_thought_stream_piece (&flt, piece_sv, config_.stream_stdout,
                                    nullptr);
      if (flt.hit_stop)
        {
          break;
        }

      const bool is_thought_token
          = flt.suppressed_on_step && !flt.emitted_on_step;
      if (flt.emitted_on_step || (!is_thought_token && flt.pending_len == 0U))
        {
          ++metrics.generated_tokens;
        }

      if (static_cast<int32_t> (metrics.generated_tokens) >= max_steps)
        {
          break;
        }

      /* In-place KV cache shift if generation reaches context boundary.  */
      if (n_past_ + 1 >= n_ctx_)
        {
          if (!shift_kv_cache_in_place (256))
            {
              break;
            }
          ++metrics.kv_shifts_performed;
          gen_start_past = n_past_;
          pending_tag_start_pos = -1;
          any_suppressed_tokens = false;
        }

      llama_batch step_batch = llama_batch_get_one (&current_token, 1);
      if (llama_decode (ctx_, step_batch) != 0)
        {
          break;
        }

      if (n_past_ < n_ctx_)
        {
          const auto kv_idx = static_cast<std::size_t> (n_past_);
          kv_tokens_[kv_idx] = current_token;
          if (is_thought_token)
            {
              token_buf_[kv_idx] = 1;
              any_suppressed_tokens = true;
              if (pending_tag_start_pos >= gen_start_past)
                {
                  for (int32_t p = pending_tag_start_pos; p < n_past_; ++p)
                    {
                      token_buf_[static_cast<std::size_t> (p)] = 1;
                    }
                  pending_tag_start_pos = -1;
                }
            }
          else
            {
              token_buf_[kv_idx] = 0;
              if (flt.pending_len > 0U && !flt.emitted_on_step)
                {
                  if (pending_tag_start_pos < 0)
                    {
                      pending_tag_start_pos = n_past_;
                    }
                }
              else
                {
                  pending_tag_start_pos = -1;
                }
            }
        }
      ++n_past_;
    }

  finish_thought_stream_filter (&flt, config_.stream_stdout, nullptr);

  /* Excise any suppressed thought/channel tokens from the KV cache and
     multi-turn conversation token history.  */
  if (any_suppressed_tokens && n_past_ > gen_start_past)
    {
      llama_memory_t mem = llama_get_memory (ctx_);
      const bool can_shift = llama_memory_can_shift (mem);
      int32_t idx = n_past_ - 1;
      while (idx >= gen_start_past)
        {
          if (token_buf_[static_cast<std::size_t> (idx)] == 0)
            {
              --idx;
              continue;
            }
          const int32_t span_end = idx + 1;
          while (idx >= gen_start_past
                 && token_buf_[static_cast<std::size_t> (idx)] != 0)
            {
              --idx;
            }
          const int32_t span_start = idx + 1;
          const int32_t span_len = span_end - span_start;
          if (!can_shift
              || llama_memory_seq_rm (mem, 0, span_start, span_end))
            {
              if (can_shift)
                {
                  llama_memory_seq_add (mem, 0, span_end, -1, -span_len);
                }
              const int32_t tail_cnt = n_past_ - span_end;
              if (tail_cnt > 0)
                {
                  std::memmove (
                      &kv_tokens_[static_cast<std::size_t> (span_start)],
                      &kv_tokens_[static_cast<std::size_t> (span_end)],
                      static_cast<std::size_t> (tail_cnt)
                          * sizeof (int32_t));
                }
              n_past_ -= span_len;
            }
        }
    }

  const uint64_t t_gen_end = monotonic_time_ns ();
  metrics.token_gen_ms = (t_gen_end > t_gen_start)
                             ? (static_cast<double> (t_gen_end - t_gen_start)
                                / 1.0e6)
                             : 0.001;

  if (metrics.prompt_eval_ms > 0.0)
    {
      metrics.prompt_eval_tok_per_s
          = (static_cast<double> (metrics.prompt_tokens) * 1000.0)
            / metrics.prompt_eval_ms;
    }
  if (metrics.token_gen_ms > 0.0 && metrics.generated_tokens > 0U)
    {
      /* Use total decode time from prefill end to generation end so even
         1-token generations report accurate tok/s.  */
      const double total_decode_ms
          = static_cast<double> (t_gen_end - t_prefill_end) / 1.0e6;
      if (total_decode_ms > 0.0)
        {
          metrics.token_gen_tok_per_s
              = (static_cast<double> (metrics.generated_tokens) * 1000.0)
                / total_decode_ms;
        }
    }

  if (out_metrics != nullptr)
    {
      *out_metrics = metrics;
    }
  return true;
}

telemetry_snapshot
inference_engine::collect_telemetry (
    const turn_metrics &last_turn) const noexcept
{
  telemetry_snapshot snap = telemetry_snapshot ();
  snap.prompt_eval_tok_per_s = last_turn.prompt_eval_tok_per_s;
  snap.token_gen_tok_per_s = last_turn.token_gen_tok_per_s;
  snap.ttft_ms = last_turn.ttft_ms;
  snap.prompt_tokens = last_turn.prompt_tokens;
  snap.generated_tokens = last_turn.generated_tokens;
  snap.kv_cached_tokens = n_past_;
  snap.kv_pinned_tokens = n_keep_;
  snap.n_ctx = n_ctx_;
  snap.active_threads_decode = active_threads_decode_;
  snap.active_threads_prefill = active_threads_prefill_;
  snap.p_cores_pinned = p_cores_pinned_;
  snap.e_cores_pinned = e_cores_pinned_;
  snap.offloaded_gpu_layers = offloaded_gpu_layers_;
  snap.total_model_layers = total_model_layers_;
  snap.hybrid_cpu_ffn_layers = hybrid_cpu_ffn_layers_;
  snap.rss_bytes = read_linux_rss_bytes ();
  snap.pinned_host_bytes
      = pinned_staging_.is_pinned () ? pinned_staging_.size () : 0U;
  (void)probe_vram_metrics (&snap.vram_allocated_bytes,
                            &snap.vram_total_bytes);
  return snap;
}

int32_t
inference_engine::pinned_system_tokens () const noexcept
{
  return n_keep_;
}

int32_t
inference_engine::cached_kv_tokens () const noexcept
{
  return n_past_;
}

int32_t
inference_engine::context_size () const noexcept
{
  return n_ctx_;
}

bool
inference_engine::set_language (MildaLang lang) noexcept
{
  config_.language = lang;
  active_turn_lang_ = (lang == MildaLang::Auto) ? MildaLang::Esperanto : lang;
  if (!rebuild_sampler (active_turn_lang_))
    {
      return false;
    }
  if (ctx_ != nullptr && vocab_ != nullptr)
    {
      return pin_system_prompt ();
    }
  return true;
}

MildaLang
inference_engine::configured_language () const noexcept
{
  return config_.language;
}

MildaLang
inference_engine::active_language () const noexcept
{
  return (config_.language == MildaLang::Auto) ? active_turn_lang_
                                               : config_.language;
}

float
inference_engine::active_repeat_penalty () const noexcept
{
  return active_repeat_penalty_;
}

float
inference_engine::active_top_p () const noexcept
{
  return active_top_p_;
}

} /* namespace milda */
