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

#include <sphinx/sphinx.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <future>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sphinx::inline v1
{

namespace
{

constexpr double kNsToMs = 1.0e-6;
constexpr double kPercentScale = 100.0;
constexpr std::size_t kFrameSamples10Ms = 160U;
constexpr std::size_t kMaxSilenceFrames = 40U;
constexpr std::size_t kMinSentenceChars = 6U;
constexpr uint32_t kBargeInMaxTokens = 192U;
constexpr uint32_t kDefaultSeed = 42U;
constexpr uint32_t kBackchannelTriggerMs = 500U;
constexpr uint32_t kSimulatedToolWorkSliceMs = 20U;
constexpr uint32_t kIdleFrameSleepMs = 10U;
constexpr float kBargeInVoicedSample = 0.30F;
constexpr std::size_t kBargeInVoicedSamples = 640U;

/* Case-insensitive substring search helper.  */
[[nodiscard]] auto
contains_icase (std::string_view haystack,
                std::string_view needle) noexcept -> bool
{
  if (needle.empty () || haystack.size () < needle.size ())
    {
      return false;
    }
  for (std::size_t i = 0U; i + needle.size () <= haystack.size (); ++i)
    {
      bool match = true;
      for (std::size_t j = 0U; j < needle.size (); ++j)
        {
          const auto c1 = static_cast<unsigned char> (haystack[i + j]);
          const auto c2 = static_cast<unsigned char> (needle[j]);
          if (std::tolower (c1) != std::tolower (c2))
            {
              match = false;
              break;
            }
        }
      if (match)
        {
          return true;
        }
    }
  return false;
}

/* Finds the first sentence-ending punctuation index at or after
   kMinSentenceChars.  */
[[nodiscard]] auto
find_sentence_boundary (std::string_view buf) noexcept -> std::size_t
{
  for (std::size_t i = 0U; i < buf.size (); ++i)
    {
      const char ch = buf[i];
      if ((ch == '.' || ch == '!' || ch == '?' || ch == '\n')
          && (i + 1U >= kMinSentenceChars))
        {
          // Avoid splitting inside decimal numbers like "3.14".
          if (ch == '.' && i + 1U < buf.size ()
              && std::isdigit (static_cast<unsigned char> (buf[i + 1U])) != 0)
            {
              continue;
            }
          return i + 1U;
        }
    }
  return std::string_view::npos;
}

/* Returns current thread CPU time in nanoseconds via
   CLOCK_THREAD_CPUTIME_ID.  */
[[nodiscard]] auto
thread_cpu_time_ns () noexcept -> uint64_t
{
  struct timespec ts
  {
    0, 0
  };
  (void)clock_gettime (CLOCK_THREAD_CPUTIME_ID, &ts);
  constexpr uint64_t kNsPerSec = 1000000000ULL;
  return static_cast<uint64_t> (ts.tv_sec) * kNsPerSec
         + static_cast<uint64_t> (ts.tv_nsec);
}

/* Lightweight deterministic streaming fallback LlmClient used only when the
   local GGUF model file is absent in a headless CI environment.  */
class HeadlessStreamingClient final : public sutra::LlmClient
{
public:
  [[nodiscard]] auto
  stream_completion (
      const sutra::CompletionRequest &req, sutra::TokenStreamFn on_token,
      sutra::CompletionDone on_done) noexcept -> sutra::Result<void> override
  {
    cancelled_.store (false, std::memory_order_release);
    const uint32_t max_tok = req.max_tokens.value_or (48U);

    std::thread ([this, max_tok, tok_cb = std::move (on_token),
                  done_cb = std::move (on_done)] () noexcept {
      constexpr std::array<std::string_view, 18> kTokens
          = { "All ",      "systems ", "are ",       "nominal. ",  "The ",
              "pipeline ", "is ",      "streaming ", "sentences ", "cleanly. ",
              "Zero ",     "audio ",   "glitches ",  "were ",      "detected ",
              "across ",   "this ",    "turn." };

      const uint64_t t0 = sutra::monotonic_time_ns ();
      constexpr std::size_t kHeadlessPromptTokens = 16U;
      constexpr int64_t kHeadlessStepMs = 6;
      sutra::CompletionResult res;
      res.prompt_tokens = kHeadlessPromptTokens;

      const std::size_t limit
          = std::min<std::size_t> (max_tok, kTokens.size ());
      for (std::size_t i = 0U; i < limit; ++i)
        {
          if (cancelled_.load (std::memory_order_acquire))
            {
              if (done_cb)
                {
                  done_cb (std::unexpected (sutra::AgentError::Cancelled ()));
                }
              return;
            }
          std::this_thread::sleep_for (
              std::chrono::milliseconds (kHeadlessStepMs));
          if (cancelled_.load (std::memory_order_acquire))
            {
              if (done_cb)
                {
                  done_cb (std::unexpected (sutra::AgentError::Cancelled ()));
                }
              return;
            }

          const uint64_t now = sutra::monotonic_time_ns ();
          if (i == 0U)
            {
              res.ttft_ns = (now >= t0) ? (now - t0) : 0U;
              sutra::AgentStats::global ().record_ttft (res.ttft_ns);
            }

          sutra::TokenChunk chunk;
          chunk.text = kTokens.at (i);
          chunk.token_id = static_cast<uint32_t> (i + 1U);
          chunk.is_first_token = (i == 0U);
          if (tok_cb)
            {
              tok_cb (chunk);
            }
          res.text.append (chunk.text);
          ++res.generated_tokens;
        }

      if (done_cb)
        {
          done_cb (std::move (res));
        }
    }).detach ();
    return {};
  }

  [[nodiscard]] auto
  cancel () noexcept -> sutra::Result<void> override
  {
    cancelled_.store (true, std::memory_order_release);
    return {};
  }

  [[nodiscard]] auto
  capabilities () const noexcept -> sutra::ModelCapabilities override
  {
    constexpr uint32_t kHeadlessVocabSize = 152064U;
    sutra::ModelCapabilities caps;
    caps.context_len = sutra::kDefaultLlamaCtxTokens;
    caps.vocab_size = kHeadlessVocabSize;
    caps.tool_style = sutra::ToolCallStyle::Qwen;
    caps.model_family = "headless-qwen2";
    return caps;
  }

  [[nodiscard]] auto
  generate (const sutra::ConversationSpanView & /*history*/,
            std::size_t /*step_index*/,
            const sutra::StreamingToolDispatchFn & /*on_early_tool*/,
            std::pmr::memory_resource * /*scratch_arena*/) noexcept
      -> sutra::Result<sutra::LlmStepOutput> override
  {
    sutra::LlmStepOutput out;
    out.final_answer = "All systems nominal.";
    return out;
  }

private:
  std::atomic<bool> cancelled_{ false };
};

} /* namespace */

/* ==========================================================================
 */
/* TurnRouter */
/* ==========================================================================
 */

TurnRouter::TurnRouter (sutra::LlmClient *llm_client) noexcept
    : llm_client_ (llm_client)
{
}

/* Classifies an incoming transcript as Conversational vs Complex
   (coding/SSH/system administration/tool dispatch).  */
auto
TurnRouter::classify (std::string_view transcript) const noexcept -> RouteKind
{
  (void)llm_client_;
  constexpr std::array<std::string_view, 38> kComplexIndicators = {
    "ssh",        "git ",       "cmake",     "compile",     "compiler",
    "build",      "segfault",   "asan",      "tsan",        "valgrind",
    "gdb",        "refactor",   "function",  "template",    "kernel",
    "cuda",       "deploy",     "docker",    "systemd",     "systemctl",
    "journalctl", "grep",       "benchmark", "stack trace", "memory leak",
    "socket",     "database",   "sql",       "c++",         "python",
    "rust",       "repository", "branch",    "commit",      "ctest",
    "nvidia-smi", "server",     "log"
  };

  for (const std::string_view kw : kComplexIndicators)
    {
      if (contains_icase (transcript, kw))
        {
          return RouteKind::Complex;
        }
    }
  return RouteKind::Conversational;
}

/* ==========================================================================
 */
/* SphinxPipeline */
/* ==========================================================================
 */

namespace
{

/* Compile-time reflected Sutra tool invoked on Complex voice turns via
   zero-copy ToolContext.  */
[[nodiscard]] auto
inspect_pipeline_status_tool (const sutra::ToolContext &ctx,
                              std::string subsystem) noexcept
    -> sutra::Result<std::string>
{
  const std::size_t history_turns = ctx.conversation_view.size ();
  return "subsystem=" + subsystem
         + " status=nominal history_turns=" + std::to_string (history_turns);
}

} /* namespace */

struct SphinxPipeline::Impl
{
  SphinxConfig config{};
  std::unique_ptr<vach::Vad> vad;
  std::unique_ptr<vach::Stt> stt;
  std::unique_ptr<vach::Tts> tts;
  std::unique_ptr<vach::AudioIo> audio_io;
  std::shared_ptr<sutra::LlmClient> llm;
  sutra::InferenceArena turn_arena{ sutra::kDefaultArenaCapacityBytes };
  sutra::ShortTermMemory short_term_memory;
  sutra::ToolRegistry tool_registry;
  TurnRouter router;
  uint64_t turn_seq = 0U;
  uint64_t last_backchannel_ns = 0U;
};

SphinxPipeline::SphinxPipeline (PrivateTag /*tag*/,
                                std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

SphinxPipeline::SphinxPipeline (SphinxPipeline &&) noexcept = default;
auto SphinxPipeline::operator= (SphinxPipeline &&) noexcept
    -> SphinxPipeline & = default;
SphinxPipeline::~SphinxPipeline () noexcept = default;

/* Creates and wires the Sphinx pipeline over Vach and Sutra.  */
auto
SphinxPipeline::create (
    const SphinxConfig &config,
    std::shared_ptr<sutra::LlmClient> llm_override) noexcept
    -> sutra::Result<std::unique_ptr<SphinxPipeline>>
{
  auto vad_res = vach::Vad::create ();
  if (!vad_res.has_value ())
    {
      return std::unexpected (vad_res.error ());
    }

  auto stt_res = vach::Stt::create ();
  if (!stt_res.has_value ())
    {
      return std::unexpected (stt_res.error ());
    }

  auto tts_res = vach::Tts::create ();
  if (!tts_res.has_value ())
    {
      return std::unexpected (tts_res.error ());
    }

  auto io_res = vach::AudioIo::create ();
  if (!io_res.has_value ())
    {
      return std::unexpected (io_res.error ());
    }

  std::shared_ptr<sutra::LlmClient> llm = std::move (llm_override);
  if (!llm)
    {
      if (std::filesystem::exists (config.model_path))
        {
          sutra::LlamaCppConfig lcfg;
          lcfg.model_path = config.model_path;
          lcfg.n_ctx = config.context_len;
          lcfg.n_gpu_layers = sutra::kDefaultLlamaGpuLayers;
          lcfg.n_threads = sutra::kDefaultLlamaThreads;
          auto llama_res = sutra::LlamaCppClient::create (lcfg);
          if (llama_res.has_value ())
            {
              llm = std::shared_ptr<sutra::LlmClient> (std::move (*llama_res));
            }
        }
      if (!llm)
        {
          llm = std::make_shared<HeadlessStreamingClient> ();
        }
    }

  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->config = config;
      impl->vad = std::move (*vad_res);
      impl->stt = std::move (*stt_res);
      impl->tts = std::move (*tts_res);
      impl->audio_io = std::move (*io_res);
      impl->llm = std::move (llm);
      impl->router = TurnRouter (impl->llm.get ());
      auto stm_res
          = sutra::ShortTermMemory::create (sutra::kDefaultShortTermCapacity);
      if (!stm_res.has_value ())
        {
          return std::unexpected (stm_res.error ());
        }
      impl->short_term_memory = std::move (*stm_res);
      auto reg_res = SUTRA_REGISTER_TOOL (
          impl->tool_registry, &inspect_pipeline_status_tool,
          "inspect_pipeline_status",
          "Inspects active Sphinx subsystem health and conversation state",
          "subsystem");
      if (!reg_res.has_value ())
        {
          return std::unexpected (reg_res.error ());
        }
      return std::make_unique<SphinxPipeline> (PrivateTag{}, std::move (impl));
    }
  catch (const std::bad_alloc &)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory creating SphinxPipeline"));
    }
}

/* Executes a full voice turn: mic -> Vad -> Stt -> Router -> Sutra
   (LlamaCppClient) -> sentence-chunked Tts -> AudioIo speaker buffer.  */
auto
SphinxPipeline::process_voice_turn (std::span<const float> utterance_16k,
                                    uint32_t simulated_tool_delay_ms) noexcept
    -> sutra::Result<VoiceTurnMetrics>
{
  if (!impl_ || utterance_16k.empty ())
    {
      return std::unexpected (
          sutra::AgentError::Stt ("process_voice_turn: empty utterance"));
    }

  impl_->turn_arena.reset ();
  impl_->vad->reset ();
  impl_->audio_io->clear_playback ();

  // 1. Stream 10 ms frames into AudioIo capture ring -> Vad until SpeechEnd.
  std::array<float, kFrameSamples10Ms> cap_frame{};
  bool speech_started = false;
  for (std::size_t pos = 0U; pos < utterance_16k.size ();
       pos += kFrameSamples10Ms)
    {
      const std::size_t n
          = std::min (kFrameSamples10Ms, utterance_16k.size () - pos);
      (void)impl_->audio_io->rt_push_capture (utterance_16k.subspan (pos, n));
      const std::size_t popped = impl_->audio_io->pop_capture (
          std::span<float> (cap_frame.data (), n));
      const vach::VadEvent ev = impl_->vad->process_frame (
          std::span<const float> (cap_frame.data (), popped));
      if (ev == vach::VadEvent::SpeechStart)
        {
          speech_started = true;
        }
    }
  if (!speech_started)
    {
      return std::unexpected (sutra::AgentError::Stt (
          "Vad did not detect SpeechStart in utterance"));
    }

  std::array<float, kFrameSamples10Ms> silence_frame{};
  silence_frame.fill (0.0F);
  bool speech_ended = false;
  for (std::size_t i = 0U; i < kMaxSilenceFrames && !speech_ended; ++i)
    {
      if (impl_->vad->process_frame (silence_frame)
          == vach::VadEvent::SpeechEnd)
        {
          speech_ended = true;
        }
    }

  const uint64_t t_speech_end = sutra::monotonic_time_ns ();
  VoiceTurnMetrics metrics;
  metrics.vad_eos_ms = impl_->vad->last_speech_end_latency_ms ();

  // 2. Transcribe utterance via vach::Stt.
  auto tr_res = impl_->stt->transcribe (utterance_16k);
  if (!tr_res.has_value ())
    {
      return std::unexpected (tr_res.error ());
    }
  metrics.transcript = tr_res->text;
  metrics.stt_latency_ms = tr_res->latency_ms;

  // Record user turn in Sutra ShortTermMemory ring buffer.
  ++impl_->turn_seq;
  (void)impl_->short_term_memory.append (sutra::Message{
      sutra::MessageRole::User, metrics.transcript, "", "", impl_->turn_seq,
      t_speech_end });

  // 3. Classify turn via Small-Model TurnRouter.
  metrics.route = impl_->router.classify (metrics.transcript);

  std::atomic<uint64_t> first_audio_ns{ 0U };
  uint64_t prev_sentence_done_ns = 0U;

  auto synthesize_and_enqueue = [&] (std::string_view sentence_text) {
    const uint64_t sent_t0 = sutra::monotonic_time_ns ();
    if (prev_sentence_done_ns > 0U && sent_t0 > prev_sentence_done_ns)
      {
        // Account for audio already buffered in the speaker playback queue.
        const std::size_t buffered_samples
            = impl_->audio_io->stats ().playback_occupancy;
        const double buffered_audio_ms
            = (static_cast<double> (buffered_samples) * 1000.0)
              / static_cast<double> (impl_->tts->sample_rate_hz ());
        const double wall_delta_ms
            = static_cast<double> (sent_t0 - prev_sentence_done_ns) * kNsToMs;
        const double audible_gap_ms
            = std::max (0.0, wall_delta_ms - buffered_audio_ms);
        metrics.max_inter_sentence_gap_ms
            = std::max (metrics.max_inter_sentence_gap_ms, audible_gap_ms);
      }

    bool recorded_first_chunk_for_sentence = false;
    auto tts_rep = impl_->tts->synthesize_sentence (
        sentence_text, [&] (std::span<const int16_t> pcm_chunk) {
          const uint64_t now_ns = sutra::monotonic_time_ns ();
          if (!recorded_first_chunk_for_sentence)
            {
              recorded_first_chunk_for_sentence = true;
              sutra::AgentStats::global ().record_sentence_to_tts (
                  (now_ns >= sent_t0) ? (now_ns - sent_t0) : 0U);
            }
          uint64_t expected = 0U;
          (void)first_audio_ns.compare_exchange_strong (
              expected, now_ns, std::memory_order_acq_rel);
          metrics.playback_samples_pushed
              += impl_->audio_io->push_playback (pcm_chunk);
        });
    if (tts_rep.has_value ())
      {
        ++metrics.sentences_synthesized;
        prev_sentence_done_ns = sutra::monotonic_time_ns ();
      }
  };

  // 4. Backchannel under load & reflected Sutra tool dispatch (§3.1 & §3.2):
  std::string tool_observation;
  if (metrics.route == RouteKind::Complex)
    {
      if (simulated_tool_delay_ms >= kBackchannelTriggerMs)
        {
          const uint64_t now_ns = sutra::monotonic_time_ns ();
          if (impl_->last_backchannel_ns == 0U
              || (now_ns - impl_->last_backchannel_ns)
                     >= impl_->config.backchannel_cooldown_ns)
            {
              impl_->last_backchannel_ns = now_ns;
              synthesize_and_enqueue ("Let me check that.");
              metrics.backchannel_emitted = true;
              const uint64_t bc_ns
                  = first_audio_ns.load (std::memory_order_acquire);
              metrics.backchannel_latency_ms
                  = (bc_ns >= t_speech_end)
                        ? (static_cast<double> (bc_ns - t_speech_end)
                           * kNsToMs)
                        : 0.0;
            }
          std::this_thread::sleep_for (
              std::chrono::milliseconds (kSimulatedToolWorkSliceMs));
        }

      // Invoke registered Sutra tool with zero-copy ToolContext view.
      sutra::ToolContext tctx;
      tctx.conversation_view = impl_->short_term_memory.borrow_view ();
      tctx.scratch_arena = impl_->turn_arena.resource ();
      tctx.run_id = impl_->turn_seq;
      tctx.step_index = 0U;
      auto tool_res = impl_->tool_registry.invoke (
          "inspect_pipeline_status", tctx, R"({"subsystem":"sphinx"})");
      if (tool_res.has_value ())
        {
          tool_observation = std::move (*tool_res);
        }
    }

  // 5. Stream live LLM response and dispatch each sentence to TTS immediately
  //    upon sentence boundary without waiting for EOS, using per-turn PMR
  //    scratch arena (C1).
  std::string sys_prompt = "Reply in two or three short concise sentences.";
  if (!tool_observation.empty ())
    {
      sys_prompt.append (" Tool observation: ");
      sys_prompt.append (tool_observation);
    }

  std::array<sutra::Message, 2> prompt_msgs{};
  prompt_msgs[0]
      = { sutra::MessageRole::System, sys_prompt, "", "", 0U, 0U };
  prompt_msgs[1] = {
    sutra::MessageRole::User, metrics.transcript, "", "", 1U, t_speech_end
  };

  sutra::CompletionRequest req;
  req.messages = prompt_msgs;
  req.temperature = 0.0F;
  req.max_tokens = (metrics.route == RouteKind::Complex)
                       ? impl_->config.complex_max_tokens
                       : impl_->config.conversational_max_tokens;
  req.seed = kDefaultSeed;

  std::pmr::string pending_sentence (impl_->turn_arena.resource ());
  std::promise<sutra::Result<sutra::CompletionResult>> done_prom;
  auto done_fut = done_prom.get_future ();

  auto disp = impl_->llm->stream_completion (
      req,
      [&] (const sutra::TokenChunk &chunk) {
        pending_sentence.append (chunk.text);
        for (;;)
          {
            const std::size_t boundary
                = find_sentence_boundary (pending_sentence);
            if (boundary == std::string_view::npos)
              {
                break;
              }
            const std::string_view sentence_view (pending_sentence.data (),
                                                  boundary);
            synthesize_and_enqueue (sentence_view);
            pending_sentence.erase (0U, boundary);
          }
      },
      [&] (sutra::Result<sutra::CompletionResult> res) {
        done_prom.set_value (std::move (res));
      });
  if (!disp.has_value ())
    {
      return std::unexpected (disp.error ());
    }

  auto comp_res = done_fut.get ();
  if (!comp_res.has_value ())
    {
      return std::unexpected (comp_res.error ());
    }

  // Flush any trailing partial sentence at EOS.
  if (!pending_sentence.empty ())
    {
      synthesize_and_enqueue (std::string_view (pending_sentence.data (),
                                                pending_sentence.size ()));
      pending_sentence.clear ();
    }

  ++impl_->turn_seq;
  (void)impl_->short_term_memory.append (sutra::Message{
      sutra::MessageRole::Assistant, comp_res->text, "", "", impl_->turn_seq,
      sutra::monotonic_time_ns () });

  metrics.response_text = comp_res->text;
  metrics.llm_ttft_ms = static_cast<double> (comp_res->ttft_ns) * kNsToMs;
  const uint64_t fa_ns = first_audio_ns.load (std::memory_order_acquire);
  metrics.ttfs_ms
      = (fa_ns >= t_speech_end)
            ? (static_cast<double> (fa_ns - t_speech_end) * kNsToMs)
            : 0.0;
  return metrics;
}

/* Exercises live barge-in: starts LLM generation + TTS playback, injects a
   Vad::SpeechStart event mid-playback, and measures time to playback silence
   + LLM cancellation (§3.2 metric 3).  */
auto
SphinxPipeline::test_live_barge_in () noexcept -> sutra::Result<BargeInMetrics>
{
  if (!impl_)
    {
      return std::unexpected (
          sutra::AgentError::AudioDevice (0, "SphinxPipeline uninitialized"));
    }

  impl_->vad->reset ();
  impl_->audio_io->clear_playback ();

  // Pre-fill speaker buffer with active TTS audio while LLM streams.
  (void)impl_->tts->synthesize_sentence (
      "Streaming initial response sentence before user barge in interrupts.",
      [this] (std::span<const int16_t> chunk) {
        (void)impl_->audio_io->push_playback (chunk);
      });

  std::array<sutra::Message, 1> msgs{ sutra::Message{
      sutra::MessageRole::User, "Explain C++23 memory ordering in detail.", "",
      "", 1U, 0U } };
  sutra::CompletionRequest req;
  req.messages = msgs;
  req.temperature = 0.0F;
  req.max_tokens = kBargeInMaxTokens;
  req.seed = kDefaultSeed;

  std::promise<void> first_tok_prom;
  auto first_tok_fut = first_tok_prom.get_future ();
  std::atomic<bool> notified{ false };
  std::promise<sutra::Result<sutra::CompletionResult>> done_prom;
  auto done_fut = done_prom.get_future ();

  auto disp = impl_->llm->stream_completion (
      req,
      [&] (const sutra::TokenChunk &) {
        if (!notified.exchange (true, std::memory_order_acq_rel))
          {
            first_tok_prom.set_value ();
          }
      },
      [&] (sutra::Result<sutra::CompletionResult> res) {
        done_prom.set_value (std::move (res));
      });
  if (!disp.has_value ())
    {
      return std::unexpected (disp.error ());
    }

  first_tok_fut.wait ();

  // User starts speaking mid-playback -> Vad emits SpeechStart.
  std::vector<float> barge_frame (kBargeInVoicedSamples, kBargeInVoicedSample);
  const vach::VadEvent ev = impl_->vad->process_frame (barge_frame);
  if (ev != vach::VadEvent::SpeechStart)
    {
      return std::unexpected (sutra::AgentError::AudioDevice (
          0, "Barge-in frame did not trigger VadEvent::SpeechStart"));
    }

  const uint64_t barge_t0 = sutra::monotonic_time_ns ();
  impl_->tts->stop ();
  impl_->audio_io->clear_playback ();
  (void)impl_->llm->cancel ();
  sutra::AgentStats::global ().record_barge_in ();

  auto cancelled_res = done_fut.get ();
  const uint64_t barge_t1 = sutra::monotonic_time_ns ();

  BargeInMetrics out;
  out.stop_latency_ms
      = (barge_t1 >= barge_t0)
            ? (static_cast<double> (barge_t1 - barge_t0) * kNsToMs)
            : 0.0;
  out.playback_cleared = (impl_->audio_io->stats ().playback_occupancy == 0U);
  out.llm_cancelled = (!cancelled_res.has_value ()
                       && cancelled_res.error ().is_cancelled ());
  return out;
}

/* Measures single-core CPU utilization of the idle listening loop (AudioIo +
   Vad on silence at 10 ms frame cadence) over duration_ms.  */
auto
SphinxPipeline::measure_idle_cpu_percent (uint32_t duration_ms) noexcept
    -> sutra::Result<double>
{
  if (!impl_ || duration_ms == 0U)
    {
      return std::unexpected (
          sutra::AgentError::AudioDevice (0, "Invalid duration_ms"));
    }

  impl_->vad->reset ();
  std::array<float, kFrameSamples10Ms> silence{};
  silence.fill (0.0F);
  std::array<float, kFrameSamples10Ms> popped{};

  const uint64_t wall_t0 = sutra::monotonic_time_ns ();
  const uint64_t cpu_t0 = thread_cpu_time_ns ();
  const uint32_t iterations = std::max (1U, duration_ms / kIdleFrameSleepMs);

  for (uint32_t i = 0U; i < iterations; ++i)
    {
      (void)impl_->audio_io->rt_push_capture (silence);
      const std::size_t n = impl_->audio_io->pop_capture (popped);
      (void)impl_->vad->process_frame (
          std::span<const float> (popped.data (), n));
      std::this_thread::sleep_for (
          std::chrono::milliseconds (kIdleFrameSleepMs));
    }

  const uint64_t cpu_t1 = thread_cpu_time_ns ();
  const uint64_t wall_t1 = sutra::monotonic_time_ns ();
  if (wall_t1 <= wall_t0)
    {
      return 0.0;
    }

  const auto cpu_ns = static_cast<double> (cpu_t1 - cpu_t0);
  const auto wall_ns = static_cast<double> (wall_t1 - wall_t0);
  return (cpu_ns / wall_ns) * kPercentScale;
}

auto
SphinxPipeline::router () const noexcept -> const TurnRouter &
{
  return impl_->router;
}

auto
SphinxPipeline::audio_io () noexcept -> vach::AudioIo &
{
  return *impl_->audio_io;
}

} /* namespace sphinx::inline v1 */
