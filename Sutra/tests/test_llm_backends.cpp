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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

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
default_backend_test_model_path () -> std::string
{
  const char *h = std::getenv ("HOME");
  const std::string base
      = (h != nullptr && h[0] != '\0') ? std::string (h) : std::string (".");
  return base + "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
}

const std::string kModelPath = default_backend_test_model_path ();

void
test_all_seven_tool_call_parsers ()
{
  // 1. Llama 3.1 / 3.2 (<|python_tag|>{...}</|python_tag|> and bare JSON)
  {
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Llama3);
    const std::array<std::string_view, 4U> chunks{
      "<|python_", "tag|>{\"name\":\"get_weather\",\"arguments\":{\"city\":",
      "\"Caracas\"}}</|python_", "tag|>{\"name\":\"ping\",\"arguments\":{}}"
    };
    std::vector<sutra::ToolCall> streamed;
    for (const std::string_view ch : chunks)
      {
        auto got = parser.parse (ch);
        streamed.insert (streamed.end (), got.begin (), got.end ());
      }
    SUTRA_ASSERT (streamed.size () == 2U);
    SUTRA_ASSERT (streamed[0].name == "get_weather");
    SUTRA_ASSERT (streamed[0].arguments_json == R"({"city":"Caracas"})");
    SUTRA_ASSERT (streamed[1].name == "ping");
  }

  // 2. Qwen 2.5 / 3 (<tool_call>{...}</tool_call>, split across chunks)
  {
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Qwen);
    const std::array<std::string_view, 5U> chunks{
      "Thinking... <tool_", "call>\n{\"name\":\"run_cmd\",",
      "\"arguments\":{\"cmd\":\"ls\"}}\n</tool_", "call><tool_call>",
      "{\"name\":\"pwd\",\"arguments\":{}}</tool_call>"
    };
    std::vector<sutra::ToolCall> streamed;
    for (const std::string_view ch : chunks)
      {
        auto got = parser.parse (ch);
        streamed.insert (streamed.end (), got.begin (), got.end ());
      }
    SUTRA_ASSERT (streamed.size () == 2U);
    SUTRA_ASSERT (streamed[0].name == "run_cmd");
    SUTRA_ASSERT (streamed[1].name == "pwd");
  }

  // 3. Mistral / Mixtral ([TOOL_CALLS][{...}])
  {
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Mistral);
    const std::array<std::string_view, 3U> chunks{
      "[TOOL_", "CALLS][{\"name\":\"read_db\",\"arguments\":{\"id\":7}},",
      "{\"name\":\"close_db\",\"arguments\":{}}]"
    };
    std::vector<sutra::ToolCall> streamed;
    for (const std::string_view ch : chunks)
      {
        auto got = parser.parse (ch);
        streamed.insert (streamed.end (), got.begin (), got.end ());
      }
    SUTRA_ASSERT (streamed.size () == 2U);
    SUTRA_ASSERT (streamed[0].name == "read_db");
    SUTRA_ASSERT (streamed[1].name == "close_db");
  }

  // 4. Hermes / Nous (<tool_call>{...}</tool_call> with "parameters" schema)
  {
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Hermes);
    const auto complete
        = parser.parse_complete ("<tool_call>{\"name\":\"calc\","
                                 "\"parameters\":{\"x\":42}}</tool_call>");
    SUTRA_ASSERT (complete.size () == 1U);
    SUTRA_ASSERT (complete[0].name == "calc");
    SUTRA_ASSERT (complete[0].arguments_json == R"({"x":42})");
  }

  // 5. DeepSeek (<｜tool▁calls｜> with full-width pipes)
  {
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::DeepSeek);
    const std::array<std::string_view, 3U> chunks{
      "<\xef\xbd\x9ctool\xe2\x96\x81",
      "calls\xef\xbd\x9c>{\"name\":\"git_status\",",
      "\"arguments\":{\"repo\":\"sutra\"}}<\xef\xbd\x9ctool\xe2\x96\x81"
      "call\xe2\x96\x81"
      "end\xef\xbd\x9c>"
    };
    std::vector<sutra::ToolCall> streamed;
    for (const std::string_view ch : chunks)
      {
        auto got = parser.parse (ch);
        streamed.insert (streamed.end (), got.begin (), got.end ());
      }
    SUTRA_ASSERT (streamed.size () == 1U);
    SUTRA_ASSERT (streamed[0].name == "git_status");
  }

  // 6. Phi-3 / Phi-4 (fenced ```json ... ``` block preceded by prose)
  {
    auto parser = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Phi);
    const std::array<std::string_view, 3U> chunks{
      "I will invoke the tool now:\n``",
      "`json\n{\"name\":\"fetch_url\",\"arguments\":{\"url\":\"https://"
      "gnu.org\"}}\n`",
      "``"
    };
    std::vector<sutra::ToolCall> streamed;
    for (const std::string_view ch : chunks)
      {
        auto got = parser.parse (ch);
        streamed.insert (streamed.end (), got.begin (), got.end ());
      }
    SUTRA_ASSERT (streamed.size () == 1U);
    SUTRA_ASSERT (streamed[0].name == "fetch_url");
  }

  // 7. Fallback (top-level {...} with "name" + "arguments")
  {
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Fallback);
    const auto complete = parser.parse_complete (
        "Result: {\"name\":\"fallback_fn\",\"arguments\":{\"k\":\"v\"}}");
    SUTRA_ASSERT (complete.size () == 1U);
    SUTRA_ASSERT (complete[0].name == "fallback_fn");
  }

  // 8. Gemma 4 dual-format (<|tool_call>call:tool_name{...}<|tool_call|> and
  //    <tool_call>{"name":...}</tool_call>)
  {
    SUTRA_ASSERT (sutra::ToolCallParser::detect_style ("gemma4")
                  == sutra::ToolCallStyle::Gemma4);
    auto parser
        = sutra::ToolCallParser::for_style (sutra::ToolCallStyle::Gemma4);
    const std::array<std::string_view, 4U> chunks{
      "<|tool_", "call>call:read_file{path:<|\"|>notes.txt<|\"|>}<|tool_",
      "call|><tool_call>{\"name\":\"list_dir\",",
      "\"arguments\":{\"path\":\".\"}}</tool_call>"
    };
    std::vector<sutra::ToolCall> streamed;
    for (const std::string_view ch : chunks)
      {
        auto got = parser.parse (ch);
        streamed.insert (streamed.end (), got.begin (), got.end ());
      }
    SUTRA_ASSERT (streamed.size () == 2U);
    SUTRA_ASSERT (streamed[0].name == "read_file");
    SUTRA_ASSERT (streamed[0].arguments_json == R"({"path":"notes.txt"})");
    SUTRA_ASSERT (streamed[1].name == "list_dir");
    SUTRA_ASSERT (streamed[1].arguments_json == R"({"path":"."})");
  }
}

void
test_live_llamacpp_streaming_cancel_and_overflow ()
{
  if (!std::filesystem::exists (kModelPath))
    {
      return;
    }

  sutra::LlamaCppConfig cfg;
  cfg.model_path = std::string (kModelPath);
  cfg.n_ctx = 2048U;
  cfg.n_gpu_layers = 99;
  cfg.n_threads = 8;

  auto client_res = sutra::LlamaCppClient::create (cfg);
  SUTRA_ASSERT (client_res.has_value ());
  auto &client = *client_res;

  const auto caps = client->capabilities ();
  SUTRA_ASSERT (caps.context_len == 2048U);
  SUTRA_ASSERT (caps.tool_style == sutra::ToolCallStyle::Qwen);

  // 1. Context overflow before generation begins (§1.5 item 5).
  {
    std::vector<sutra::Message> msgs{ { sutra::MessageRole::User,
                                        "Hello world", "", "", 1U, 0U } };
    sutra::CompletionRequest overflow_req;
    overflow_req.messages = msgs;
    overflow_req.max_tokens = 4096U; // Exceeds n_ctx = 2048

    std::atomic<bool> token_fired{ false };
    std::atomic<bool> done_saw_overflow{ false };
    auto ov_res = client->stream_completion (
        overflow_req,
        [&token_fired] (const sutra::TokenChunk &) {
          token_fired.store (true, std::memory_order_release);
        },
        [&done_saw_overflow] (sutra::Result<sutra::CompletionResult> r) {
          if (!r && r.error ().is_context_overflow ())
            {
              done_saw_overflow.store (true, std::memory_order_release);
            }
        });
    SUTRA_ASSERT (!ov_res.has_value ());
    SUTRA_ASSERT (ov_res.error ().is_context_overflow ());
    SUTRA_ASSERT (done_saw_overflow.load (std::memory_order_acquire));
    SUTRA_ASSERT (!token_fired.load (std::memory_order_acquire));
  }

  // 2. Streaming proof: is_first_token precedes on_done by > 100 ms.
  {
    std::vector<sutra::Message> msgs{
      { sutra::MessageRole::User,
        "Count from 1 to 30 separated by commas with a short explanation.", "",
        "", 1U, 0U }
    };
    sutra::CompletionRequest req;
    req.messages = msgs;
    req.temperature = 0.0F;
    req.max_tokens = 28U;
    req.seed = 42U;

    std::atomic<uint64_t> first_tok_ts{ 0U };
    std::atomic<uint64_t> done_ts{ 0U };
    std::atomic<int> first_tok_count{ 0 };
    std::promise<sutra::Result<sutra::CompletionResult>> done_promise;
    auto done_future = done_promise.get_future ();

    auto disp = client->stream_completion (
        req,
        [&] (const sutra::TokenChunk &chunk) {
          if (chunk.is_first_token)
            {
              first_tok_ts.store (sutra::monotonic_time_ns (),
                                  std::memory_order_release);
              first_tok_count.fetch_add (1, std::memory_order_relaxed);
            }
        },
        [&] (sutra::Result<sutra::CompletionResult> res) {
          done_ts.store (sutra::monotonic_time_ns (),
                         std::memory_order_release);
          done_promise.set_value (std::move (res));
        });
    SUTRA_ASSERT (disp.has_value ());

    auto final_res = done_future.get ();
    SUTRA_ASSERT (final_res.has_value ());
    SUTRA_ASSERT (first_tok_count.load () == 1);
    const uint64_t t_first = first_tok_ts.load (std::memory_order_acquire);
    const uint64_t t_done = done_ts.load (std::memory_order_acquire);
    SUTRA_ASSERT (t_first > 0U && t_done > t_first);
    const double delta_ms = static_cast<double> (t_done - t_first) / 1.0e6;
    const double ttft_ms = static_cast<double> (final_res->ttft_ns) / 1.0e6;
    std::fprintf (stdout,
                  "[Phase1Metrics] ttft_ms=%.3f stream_lead_ms=%.3f "
                  "gen_tokens=%u\n",
                  ttft_ms, delta_ms,
                  static_cast<unsigned> (final_res->generated_tokens));
    SUTRA_ASSERT (delta_ms > 100.0);
  }

  // 3. Cooperative cancel() halts within <= 50 ms and fires no callback after
  //    on_done (§1.5 item 4).
  {
    std::vector<sutra::Message> msgs{
      { sutra::MessageRole::User,
        "Write a detailed 500-word essay about C++23 memory models.", "", "",
        2U, 0U }
    };
    sutra::CompletionRequest req;
    req.messages = msgs;
    req.temperature = 0.0F;
    req.max_tokens = 256U;
    req.seed = 42U;

    std::promise<void> first_token_seen;
    auto first_token_fut = first_token_seen.get_future ();
    std::atomic<bool> notified_first{ false };
    std::atomic<bool> done_fired_flag{ false };
    std::atomic<int> callbacks_after_done{ 0 };
    std::atomic<uint64_t> done_timestamp_ns{ 0U };
    std::promise<sutra::Result<sutra::CompletionResult>> done_promise;
    auto done_fut = done_promise.get_future ();

    auto disp = client->stream_completion (
        req,
        [&] (const sutra::TokenChunk &) {
          if (done_fired_flag.load (std::memory_order_acquire))
            {
              callbacks_after_done.fetch_add (1, std::memory_order_relaxed);
            }
          if (!notified_first.exchange (true, std::memory_order_acq_rel))
            {
              first_token_seen.set_value ();
            }
        },
        [&] (sutra::Result<sutra::CompletionResult> res) {
          done_fired_flag.store (true, std::memory_order_release);
          done_timestamp_ns.store (sutra::monotonic_time_ns (),
                                   std::memory_order_release);
          done_promise.set_value (std::move (res));
        });
    SUTRA_ASSERT (disp.has_value ());

    first_token_fut.wait ();
    const uint64_t cancel_t0 = sutra::monotonic_time_ns ();
    const auto cancel_res = client->cancel ();
    SUTRA_ASSERT (cancel_res.has_value ());

    auto res = done_fut.get ();
    const uint64_t cancel_t1
        = done_timestamp_ns.load (std::memory_order_acquire);
    const double cancel_ms
        = (cancel_t1 >= cancel_t0)
              ? (static_cast<double> (cancel_t1 - cancel_t0) / 1.0e6)
              : 0.0;
    std::fprintf (stdout, "[Phase1Metrics] cancel_ms=%.3f\n", cancel_ms);

    SUTRA_ASSERT (!res.has_value ());
    SUTRA_ASSERT (res.error ().is_cancelled ());
    SUTRA_ASSERT (cancel_ms <= 50.0);

    std::this_thread::sleep_for (std::chrono::milliseconds (25));
    SUTRA_ASSERT (callbacks_after_done.load () == 0);
  }

  // 4. 500-turn live conversation with flat per-turn orchestration latency
  //    (excluding model generation time, §1.5 item 1).
  {
    auto stm_res = sutra::ShortTermMemory::create (64U);
    SUTRA_ASSERT (stm_res.has_value ());
    auto &short_term_mem = *stm_res;
    (void)short_term_mem.append ({ sutra::MessageRole::System,
                                   "You are Sutra concise assistant.", "", "",
                                   0U, 0U });

    std::array<sutra::Message, 2> active_window{};
    active_window[0] = { sutra::MessageRole::System,
                         "You are Sutra concise assistant.",
                         "",
                         "",
                         0U,
                         0U };

    double first_25_orch_us_sum = 0.0;
    double last_25_orch_us_sum = 0.0;
    double first_25_model_ms_sum = 0.0;
    double last_25_model_ms_sum = 0.0;
    const uint64_t loop_t0 = sutra::monotonic_time_ns ();

    for (uint64_t turn = 1U; turn <= 500U; ++turn)
      {
        const uint64_t turn_t0 = sutra::monotonic_time_ns ();
        sutra::Message user_msg{
          sutra::MessageRole::User, "Status ping ack.", "", "", turn, turn_t0
        };
        (void)short_term_mem.append (user_msg);
        const auto view = short_term_mem.borrow_view ();
        SUTRA_ASSERT (view.size () > 0U);
        active_window[1] = user_msg;

        sutra::CompletionRequest req;
        req.messages = active_window;
        req.temperature = 0.0F;
        req.max_tokens = 1U;
        req.seed = 42U;

        std::promise<sutra::Result<sutra::CompletionResult>> prom;
        auto fut = prom.get_future ();
        uint64_t model_t0 = 0U;
        uint64_t model_t1 = 0U;

        const uint64_t pre_dispatch_ns = sutra::monotonic_time_ns ();
        model_t0 = pre_dispatch_ns;
        auto disp = client->stream_completion (
            req, [] (const sutra::TokenChunk &) {},
            [&] (sutra::Result<sutra::CompletionResult> r) {
              model_t1 = sutra::monotonic_time_ns ();
              prom.set_value (std::move (r));
            });
        SUTRA_ASSERT (disp.has_value ());

        auto comp = fut.get ();
        SUTRA_ASSERT (comp.has_value ());
        SUTRA_ASSERT (comp->generated_tokens >= 1U);

        (void)short_term_mem.append ({ sutra::MessageRole::Assistant,
                                       comp->text, "", "", turn, model_t1 });
        const uint64_t turn_t1 = sutra::monotonic_time_ns ();

        const uint64_t total_turn_ns = turn_t1 - turn_t0;
        const uint64_t model_ns
            = (model_t1 >= model_t0) ? (model_t1 - model_t0) : 0U;
        const uint64_t orch_ns
            = (total_turn_ns >= model_ns) ? (total_turn_ns - model_ns) : 0U;
        const double orch_us = static_cast<double> (orch_ns) / 1.0e3;
        const double model_ms = static_cast<double> (model_ns) / 1.0e6;

        if (turn <= 25U)
          {
            first_25_orch_us_sum += orch_us;
            first_25_model_ms_sum += model_ms;
          }
        if (turn > 475U)
          {
            last_25_orch_us_sum += orch_us;
            last_25_model_ms_sum += model_ms;
          }
      }

    const uint64_t loop_t1 = sutra::monotonic_time_ns ();
    const double first_orch_us = first_25_orch_us_sum / 25.0;
    const double last_orch_us = last_25_orch_us_sum / 25.0;
    const double first_model_ms = first_25_model_ms_sum / 25.0;
    const double last_model_ms = last_25_model_ms_sum / 25.0;
    const double loop_s = static_cast<double> (loop_t1 - loop_t0) / 1.0e9;

    std::fprintf (
        stdout,
        "[Phase1Metrics] 500_turns_s=%.2f orch_first25_us=%.2f "
        "orch_last25_us=%.2f model_first25_ms=%.2f model_last25_ms=%.2f\n",
        loop_s, first_orch_us, last_orch_us, first_model_ms, last_model_ms);
    SUTRA_ASSERT (last_orch_us < 250.0);
  }
}

} /* namespace */

auto
main () -> int
{
  test_all_seven_tool_call_parsers ();
  test_live_llamacpp_streaming_cancel_and_overflow ();
  return 0;
}
