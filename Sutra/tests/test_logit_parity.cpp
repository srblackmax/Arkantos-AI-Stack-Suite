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

#include <sutra/sutra.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>

#define SUTRA_ASSERT(cond)                                                    \
  do                                                                          \
    {                                                                         \
      if (!(cond))                                                            \
        {                                                                     \
          std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__  \
                    << " -> " << #cond << "\n";                               \
          std::exit (1);                                                      \
        }                                                                     \
    }                                                                         \
  while (0)

namespace
{

[[nodiscard]] auto
default_parity_model_path () -> std::string
{
  const char *h = std::getenv ("HOME");
  const std::string base
      = (h != nullptr && h[0] != '\0') ? std::string (h) : std::string (".");
  return base + "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
}

const std::string kModelPath = default_parity_model_path ();

/* 50 fixed prompts covering short, long, unicode, and tool-call-inducing
   inputs for numerical logit parity evaluation (§1.4).  */
constexpr std::array<std::string_view, 50U> kParityPrompts{
  "Hello",
  "2 + 2 =",
  "The capital of France is",
  "int main() { return",
  "std::expected<int, AgentError>",
  "Sūtra (सूत्र) means thread in Sanskrit.",
  "Vāc (वाच्) is the Vedic goddess of speech.",
  "¡Hola! ¿Cómo estás hoy?",
  "日本語のテスト文字列です。",
  "Математика и программирование на C++23.",
  "<|im_start|>user\nCall get_weather for "
  "Tokyo<|im_end|>\n<|im_start|>assistant\n<tool_call>",
  "<|im_start|>user\nRun git "
  "status<|im_end|>\n<|im_start|>assistant\n<tool_call>",
  "<|im_start|>user\nCompute 19 * 23 using calc "
  "tool<|im_end|>\n<|im_start|>assistant\n",
  "Explain lock-free ring buffers in one sentence:",
  "Explain AVX-512F FMA accumulation in double precision:",
  "SELECT id, name FROM users WHERE",
  "cmake_minimum_required(VERSION 3.28)",
  "struct TokenChunk { std::string_view text;",
  "CUDA kernel warp shuffle reduction __shfl_down_sync:",
  "Kahn's topological sort algorithm detects cycles when",
  "IEEE 802.3 CRC32 polynomial 0xEDB88320:",
  "ALSA snd_pcm_writei non-blocking ring buffer:",
  "Voice Activity Detection (VAD) energy and zero-crossing rate:",
  "Barge-in interrupts TTS playback within 200 ms:",
  "Router classifies coding vs conversational turns:",
  "template <typename T> class LockFreeQueue {",
  "std::pmr::monotonic_buffer_resource arena;",
  "ShortTermMemory ring buffer O(1) append:",
  "LongTermMemory binary persistence with magic header:",
  "SemanticMemory cosine and dot-product retrieval:",
  "ToolCallStyle::Qwen uses <tool_call>{...}</tool_call>",
  "ToolCallStyle::Llama3 uses <|python_tag|>",
  "ToolCallStyle::Mistral uses [TOOL_CALLS]",
  "ToolCallStyle::DeepSeek uses full-width vertical pipes",
  "ToolCallStyle::Phi uses fenced markdown JSON blocks",
  "ToolCallStyle::Hermes uses <tool_call> tags",
  "ToolCallStyle::Fallback extracts top-level JSON objects",
  "AddressSanitizer and UndefinedBehaviorSanitizer clean:",
  "ThreadSanitizer (-fsanitize=thread) zero race reports:",
  "GNU Coding Standards two-space indentation:",
  "Intel Core i7-12700K 8 P-cores and 4 E-cores:",
  "NVIDIA GeForce RTX 3070 8 GB GDDR6 sm_86:",
  "Whisper.cpp speech-to-text transcription:",
  "Piper ONNX Runtime neural text-to-speech:",
  "Sphinx vertical slice time-to-first-syllable <= 700 ms:",
  "Backchannel utterance emitted after 500 ms planning:",
  "Zero Python dependencies across the entire stack:",
  "Deterministic RAII resource cleanup without new/delete:",
  "std::shared_mutex reader-writer lock on memory tiers:",
  "Final prompt #50 for numerical logit parity verification."
};

} /* namespace */

auto
main () -> int
{
  if (!std::filesystem::exists (kModelPath))
    {
      return 0;
    }

  sutra::LlamaCppConfig cfg;
  cfg.model_path = std::string (kModelPath);
  cfg.n_ctx = 512U;
  cfg.n_gpu_layers = 99;
  cfg.n_threads = 8;

  auto ref_res = sutra::LlamaCppClient::create (cfg);
  SUTRA_ASSERT (ref_res.has_value ());

  auto nat_res = sutra::NativeEngineClient::create (kModelPath, 512U, true);
  SUTRA_ASSERT (nat_res.has_value ());
  SUTRA_ASSERT (!(*nat_res)->inspected_tensors ().empty ());

  auto report_res = sutra::run_logit_parity_harness (**ref_res, **nat_res,
                                                     kParityPrompts, 1.0e-4F);
  SUTRA_ASSERT (report_res.has_value ());
  SUTRA_ASSERT (report_res->prompts_tested == 50U);
  SUTRA_ASSERT (report_res->positions_compared == 50U);
  SUTRA_ASSERT (report_res->top1_matches == 50U);
  SUTRA_ASSERT (report_res->max_abs_logit_delta < 1.0e-4F);
  SUTRA_ASSERT (report_res->passed);

  std::fprintf (
      stdout,
      "[LogitParity] PASSED prompts=%zu positions=%zu top1_matches=%zu "
      "max_abs_delta=%.6f\n",
      report_res->prompts_tested, report_res->positions_compared,
      report_res->top1_matches,
      static_cast<double> (report_res->max_abs_logit_delta));

  // Verify per-layer divergence diagnostic when unquantized-only mode is used.
  auto unquant_nat_res
      = sutra::NativeEngineClient::create (kModelPath, 512U, false);
  SUTRA_ASSERT (unquant_nat_res.has_value ());
  const std::span<const std::string_view> diag_prompts (kParityPrompts.data (),
                                                        4U);
  auto diag_res = sutra::run_logit_parity_harness (
      **ref_res, **unquant_nat_res, diag_prompts, 1.0e-4F);
  SUTRA_ASSERT (diag_res.has_value ());
  SUTRA_ASSERT (!diag_res->passed);
  SUTRA_ASSERT (diag_res->first_divergent_layer == 0);
  SUTRA_ASSERT (!diag_res->divergent_tensor_name.empty ());
  SUTRA_ASSERT (!diag_res->quantization_format.empty ());
  SUTRA_ASSERT (!diag_res->root_cause_report.empty ());

  return 0;
}
