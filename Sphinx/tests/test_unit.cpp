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

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace
{

auto
test_sentence_splitter () -> bool
{
  // Must not split on Dr., e.g., or 3.14, and must require >= 5 words.
  const std::string text
      = "Dr. Smith measured 3.14 radians e.g. in the lab today. Next sentence "
        "is here.";
  const std::size_t bound = sphinx::find_next_sentence_boundary (text);
  if (bound == std::string_view::npos)
    {
      return false;
    }
  const std::string first = text.substr (0U, bound);
  if (first != "Dr. Smith measured 3.14 radians e.g. in the lab today.")
    {
      (void)std::fprintf (stderr, "Unexpected sentence split: '%s'\n",
                          first.c_str ());
      return false;
    }

  // Short < 5 word sentence should merge with following text.
  const std::string short_text
      = "Hi there. This is the second part of the utterance.";
  const std::size_t bound2 = sphinx::find_next_sentence_boundary (short_text);
  if (bound2 != short_text.size () - 1U && bound2 != std::string_view::npos)
    {
      // Notice short_text has no trailing space after the final period, so
      // let's append a space and verify it merges "Hi there." into the second
      // sentence.
    }
  const std::string short_with_space
      = "Hi there. This is the second part of the utterance. ";
  const std::size_t bound3
      = sphinx::find_next_sentence_boundary (short_with_space);
  if (bound3 == std::string_view::npos || bound3 < 20U)
    {
      (void)std::fprintf (stderr, "Short sentence failed to merge: %zu\n",
                          bound3);
      return false;
    }
  return true;
}

auto
test_tools_and_sandbox () -> bool
{
  const std::filesystem::path sb = "/tmp/sphinx_unit_sandbox";
  std::error_code ec;
  std::filesystem::remove_all (sb, ec);
  auto sb_res = sphinx::ToolEnvironment::set_sandbox_dir (sb);
  if (!sb_res)
    {
      return false;
    }

  const std::filesystem::path store_dir = "/tmp/sphinx_unit_yantra";
  std::filesystem::remove_all (store_dir, ec);
  auto store_res = yantra::ExperimentStore::open (store_dir);
  if (!store_res)
    {
      return false;
    }
  auto run_res = store_res->create_run ("unit_tool_run");
  if (!run_res)
    {
      return false;
    }
  (void)run_res->start ();
  sphinx::ToolEnvironment::bind_yantra_run (&(*run_res), 1U);

  sutra::ToolRegistry reg;
  if (!sphinx::register_all_sphinx_tools (reg))
    {
      return false;
    }
  if (reg.size () != 13U)
    {
      return false;
    }

  sutra::ToolContext ctx{};

  // 1. write_file & read_file & list_dir & echo
  auto w_res = reg.invoke (
      "write_file", ctx,
      "{\"path\":\"hello.cpp\",\"content\":"
      "\"#include <iostream>\\nint main() { return 0; }\\n\"}");
  if (!w_res)
    {
      return false;
    }

  auto r_res = reg.invoke ("read_file", ctx,
                           R"({"path":"hello.cpp","max_lines":10})");
  if (!r_res || r_res->find ("int main()") == std::string::npos)
    {
      return false;
    }

  auto l_res = reg.invoke ("list_dir", ctx, R"({"path":"."})");
  if (!l_res || l_res->find ("hello.cpp") == std::string::npos)
    {
      return false;
    }

  auto e_res = reg.invoke ("echo", ctx, R"({"text":"ping_pong"})");
  if (!e_res || *e_res != "ping_pong")
    {
      return false;
    }

  // 2. compile_check & run_command
  auto c_res = reg.invoke ("compile_check", ctx,
                           R"({"file":"hello.cpp","flags":"-O2"})");
  if (!c_res || *c_res != "ok")
    {
      return false;
    }

  auto cmd_res = reg.invoke ("run_command", ctx,
                             R"({"cmd":"ls -la","cwd":"."})");
  if (!cmd_res || cmd_res->find ("hello.cpp") == std::string::npos)
    {
      return false;
    }

  auto build_res = reg.invoke (
      "run_command", ctx,
      R"({"cmd":"g++ -std=c++23 -O2 hello.cpp -o hello.out","cwd":"."})");
  if (!build_res)
    {
      return false;
    }
  auto run_bin_res = reg.invoke ("run_command", ctx,
                                 R"({"cmd":"./hello.out","cwd":"."})");
  if (!run_bin_res)
    {
      return false;
    }
  auto run_nonexec = reg.invoke ("run_command", ctx,
                                 R"({"cmd":"./hello.cpp","cwd":"."})");
  if (run_nonexec || !run_nonexec.error ().is_tool_not_allowed ())
    {
      return false;
    }

  auto touch_empty = reg.invoke ("run_command", ctx,
                                 R"({"cmd":"touch empty.cpp","cwd":"."})");
  auto fail_build = reg.invoke (
      "run_command", ctx,
      R"({"cmd":"g++ -std=c++23 -O2 empty.cpp -o empty.out","cwd":"."})");
  if (!touch_empty || fail_build
      || fail_build.error ().message ().find ("Exit ")
             == std::string_view::npos
      || std::filesystem::exists (sb / "empty.out", ec))
    {
      return false;
    }

  std::filesystem::create_directories (sb / ".memory", ec);
  auto rm_bin = reg.invoke ("remove_file", ctx, R"({"path":"hello.out"})");
  if (!rm_bin || std::filesystem::exists (sb / "hello.out", ec))
    {
      return false;
    }
  auto rm_mem = reg.invoke ("remove_file", ctx, R"({"path":".memory"})");
  if (rm_mem || !rm_mem.error ().is_tool_sandbox_violation ())
    {
      return false;
    }
  auto rm_ws = reg.invoke ("remove_file", ctx, R"({"path":"*"})");
  if (!rm_ws || std::filesystem::exists (sb / "hello.cpp", ec)
      || !std::filesystem::exists (sb / ".memory", ec))
    {
      return false;
    }

  // Verify read_file start_line pagination and directory guard.
  std::string hundred_lines;
  for (int ln = 1; ln <= 100; ++ln)
    {
      hundred_lines += "line_" + std::to_string (ln) + "\n";
    }
  {
    std::ofstream ofs (sb / "lines.txt");
    ofs << hundred_lines;
  }
  auto rf_slice = reg.invoke (
      "read_file", ctx,
      R"({"path":"lines.txt","start_line":50,"max_lines":30})");
  std::filesystem::create_directories (sb / "Aurora/src/app", ec);
  auto rf_dir = reg.invoke ("read_file", ctx,
                            R"({"path":"Aurora/src/app"})");
  if (!rf_slice || !rf_slice->starts_with ("line_50\n")
      || rf_slice->find ("line_79\n") == std::string::npos
      || rf_slice->find ("line_80\n") != std::string::npos
      || rf_dir
      || rf_dir.error ().message ().find ("is a directory; use list_dir")
             == std::string_view::npos)
    {
      (void)std::fprintf (
          stderr, "read_file start_line or directory guard check failed\n");
      return false;
    }

  // 3. Sandbox escape rejection: /etc/passwd and ../../etc/passwd
  auto esc1 = reg.invoke ("read_file", ctx,
                          R"({"path":"/etc/passwd","max_lines":5})");
  if (esc1 || !esc1.error ().is_tool_sandbox_violation ())
    {
      (void)std::fprintf (stderr, "Expected ToolSandboxViolation on /etc\n");
      return false;
    }

  auto esc2 = reg.invoke ("read_file", ctx,
                          R"({"path":"../../etc/passwd","max_lines":5})");
  if (esc2 || !esc2.error ().is_tool_sandbox_violation ())
    {
      (void)std::fprintf (stderr, "Expected ToolSandboxViolation on ../..\n");
      return false;
    }

  // 4. Disallowed command rejection (privilege escalation, outside sandbox,
  // and .memory protection) + allowed non-root workspace commands
  auto bad_cmd = reg.invoke ("run_command", ctx,
                             R"({"cmd":"rm -rf /","cwd":"."})");
  auto bad_sudo = reg.invoke ("run_command", ctx,
                              R"({"cmd":"sudo ls","cwd":"."})");
  auto bad_tmp = reg.invoke ("run_command", ctx,
                             R"({"cmd":"rm -rf /tmp","cwd":"."})");
  auto bad_mem = reg.invoke ("run_command", ctx,
                             R"({"cmd":"rm -rf .memory","cwd":"."})");
  auto ok_touch = reg.invoke ("run_command", ctx,
                              R"({"cmd":"touch touched_file.txt","cwd":"."})");
  auto ok_cp = reg.invoke (
      "run_command", ctx,
      R"({"cmd":"cp touched_file.txt copied_file.txt","cwd":"."})");
  auto ok_rm = reg.invoke (
      "run_command", ctx,
      R"({"cmd":"rm touched_file.txt copied_file.txt","cwd":"."})");
  if (bad_cmd || !bad_cmd.error ().is_tool_not_allowed ()
      || bad_sudo || !bad_sudo.error ().is_tool_not_allowed ()
      || bad_sudo.error ().message ().find ("Privilege escalation")
             == std::string_view::npos
      || bad_tmp || !bad_tmp.error ().is_tool_not_allowed ()
      || bad_mem || !bad_mem.error ().is_tool_not_allowed ()
      || !sphinx::validate_allowed_command ("cargo init --bin hello_rust")
      || !sphinx::validate_allowed_command (
             "cargo run --manifest-path hello_rust/Cargo.toml")
      || !sphinx::validate_allowed_command ("git init")
      || !ok_touch || !ok_cp || !ok_rm)
    {
      (void)std::fprintf (stderr,
                          "Sandbox non-root command validation failed\n");
      return false;
    }

  // 5. Subprocess timeout test ("sleep 60" killed after configured timeout)
  sphinx::ToolEnvironment::set_command_timeout_ms (250U);
  auto t_res = reg.invoke ("run_command", ctx,
                           R"({"cmd":"sleep 60","cwd":"."})");
  sphinx::ToolEnvironment::set_command_timeout_ms (30000U);
  if (t_res || !t_res.error ().is_tool_timeout ())
    {
      (void)std::fprintf (stderr, "Expected ToolTimeout on sleep 60\n");
      return false;
    }

  // 6. SSH tools: live localhost test, denylist rejection, unlisted host
  auto ssh_ok = reg.invoke ("ssh_test", ctx, R"({"host":"localhost"})");
  if (!ssh_ok || *ssh_ok != "ok")
    {
      (void)std::fprintf (stderr, "ssh_test(localhost) failed: %s\n",
                          ssh_ok ? ssh_ok->c_str ()
                                 : std::string (ssh_ok.error ().message ())
                                       .c_str ());
      return false;
    }

  auto ssh_uname = reg.invoke (
      "ssh_exec", ctx, R"({"host":"localhost","cmd":"uname -a"})");
  if (!ssh_uname || ssh_uname->find ("Linux") == std::string::npos)
    {
      (void)std::fprintf (stderr, "ssh_exec(localhost, uname -a) failed\n");
      return false;
    }

  auto ssh_deny = reg.invoke (
      "ssh_exec", ctx, R"({"host":"localhost","cmd":"rm -rf /tmp/foo"})");
  if (ssh_deny || !ssh_deny.error ().is_ssh_denied ())
    {
      (void)std::fprintf (stderr, "Expected SshDenied on rm -rf\n");
      return false;
    }

  auto ssh_unlisted = reg.invoke (
      "ssh_exec", ctx, R"({"host":"unlisted.invalid","cmd":"uname -a"})");
  if (ssh_unlisted || !ssh_unlisted.error ().is_ssh_denied ())
    {
      (void)std::fprintf (stderr, "Expected SshDenied on unlisted host\n");
      return false;
    }

  sphinx::ToolEnvironment::bind_yantra_run (nullptr, 0U);
  return true;
}

auto
test_make_directory () -> bool
{
  const std::filesystem::path sb = "/tmp/sphinx_mkdir_unit_sandbox";
  const std::filesystem::path outside_escape = "/tmp/escape";
  std::error_code ec;
  std::filesystem::remove_all (sb, ec);
  std::filesystem::remove_all (outside_escape, ec);

  auto sb_res = sphinx::ToolEnvironment::set_sandbox_dir (sb);
  if (!sb_res)
    {
      return false;
    }
  const std::filesystem::path canon_sb = *sb_res;

  // T1: make_directory("test_dir") on empty sandbox -> returns
  // <sandbox>/test_dir, directory exists.
  auto t1 = sphinx::make_directory ("test_dir");
  if (!t1 || *t1 != (canon_sb / "test_dir")
      || !std::filesystem::is_directory (*t1, ec))
    {
      (void)std::fprintf (stderr, "T1 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T2: make_directory("a/b/c") on empty sandbox -> returns <sandbox>/a/b/c,
  // all three directories exist.
  auto t2 = sphinx::make_directory ("a/b/c");
  if (!t2 || *t2 != (canon_sb / "a" / "b" / "c")
      || !std::filesystem::is_directory (canon_sb / "a", ec)
      || !std::filesystem::is_directory (canon_sb / "a" / "b", ec)
      || !std::filesystem::is_directory (canon_sb / "a" / "b" / "c", ec))
    {
      (void)std::fprintf (stderr, "T2 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T3: make_directory("test_dir") when test_dir already exists as dir ->
  // returns same path, no error (idempotent).
  auto t3 = sphinx::make_directory ("test_dir");
  if (!t3 || *t3 != *t1)
    {
      (void)std::fprintf (stderr, "T3 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T4: make_directory("test_file") when target exists as a file -> returns
  // ToolSandboxViolation.
  std::filesystem::remove_all (canon_sb / "test_dir", ec);
  {
    std::ofstream ofs (canon_sb / "test_dir", std::ios::out);
    ofs << "regular file\n";
  }
  auto t4 = sphinx::make_directory ("test_dir");
  if (t4 || !t4.error ().is_tool_sandbox_violation ())
    {
      (void)std::fprintf (stderr, "T4 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T5: make_directory("../escape") -> returns ToolSandboxViolation, no
  // directory created outside sandbox.
  auto t5 = sphinx::make_directory ("../escape");
  if (t5 || !t5.error ().is_tool_sandbox_violation ()
      || std::filesystem::exists (outside_escape, ec))
    {
      (void)std::fprintf (stderr, "T5 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T6: make_directory("/etc/evil") -> returns ToolSandboxViolation.
  auto t6 = sphinx::make_directory ("/etc/evil");
  if (t6 || !t6.error ().is_tool_sandbox_violation ())
    {
      (void)std::fprintf (stderr, "T6 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T7: make_directory("a/../../escape") -> returns ToolSandboxViolation, plus
  // symlink inside sandbox pointing outside.
  auto t7 = sphinx::make_directory ("a/../../escape");
  if (t7 || !t7.error ().is_tool_sandbox_violation ()
      || std::filesystem::exists (outside_escape, ec))
    {
      (void)std::fprintf (stderr, "T7 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }
  std::filesystem::create_directory_symlink ("/tmp", canon_sb / "sym_out", ec);
  auto t7_sym = sphinx::make_directory ("sym_out/sphinx_sym_escape");
  if (t7_sym || !t7_sym.error ().is_tool_sandbox_violation ()
      || std::filesystem::exists ("/tmp/sphinx_sym_escape", ec))
    {
      (void)std::fprintf (stderr, "T7 symlink escape failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T8: make_directory("") -> returns ToolExecutionFailed with "empty path"
  // message.
  auto t8 = sphinx::make_directory ("");
  if (t8 || !t8.error ().is_tool_error ()
      || std::get<sutra::ToolError> (t8.error ().variant ()).code
             != sutra::ToolErrorCode::ExecutionFailure
      || t8.error ().message ().find ("empty path") == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "T8 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T9: make_directory("name_with_nul\0byte") -> returns ToolSandboxViolation.
  constexpr std::string_view kNulPath ("name_with_nul\0byte", 18U);
  auto t9 = sphinx::make_directory (kNulPath);
  if (t9 || !t9.error ().is_tool_sandbox_violation ())
    {
      (void)std::fprintf (stderr, "T9 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  // T10: make_directory with path component > 255 bytes -> returns
  // ToolExecutionFailed with clear message.
  const std::string long_comp (256U, 'x');
  auto t10 = sphinx::make_directory (long_comp);
  if (t10 || !t10.error ().is_tool_error ()
      || std::get<sutra::ToolError> (t10.error ().variant ()).code
             != sutra::ToolErrorCode::ExecutionFailure
      || t10.error ().message ().find ("255") == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "T10 failed\n");
      std::filesystem::remove_all (sb, ec);
      return false;
    }

  std::filesystem::remove_all (sb, ec);
  return true;
}

auto
test_registry_and_darshana () -> bool
{
  const std::filesystem::path tmp_reg = "/tmp/sphinx_unit_empty_registry";
  std::error_code ec;
  std::filesystem::remove_all (tmp_reg, ec);

  auto missing = sphinx::resolve_model_from_registry (
      tmp_reg, "nonexistent-model-id", 1U, false);
  if (missing || !missing.error ().is_model_not_registered ())
    {
      (void)std::fprintf (stderr, "Expected ModelNotRegistered error\n");
      return false;
    }

  const std::filesystem::path svg_dir = "/tmp/sphinx_unit_svgs";
  std::filesystem::remove_all (svg_dir, ec);

  std::vector<sphinx::TurnTelemetry> turns (2U);
  turns[0].turn_index = 1U;
  turns[0].route = sphinx::Route::CHAT;
  turns[0].router_confidence = 0.98F;
  turns[0].vad_eos_ms = 200.0;
  turns[0].stt_ms = 75.0;
  turns[0].router_ms = 18.0;
  turns[0].llm_ttft_ms = 22.0;
  turns[0].llm_total_ms = 90.0;
  turns[0].tts_first_sample_ms = 110.0;
  turns[0].ttfs_ms = 395.0;
  turns[0].tok_per_sec = 92.0;

  turns[1].turn_index = 2U;
  turns[1].route = sphinx::Route::TASK;
  turns[1].router_confidence = 0.96F;
  turns[1].vad_eos_ms = 200.0;
  turns[1].stt_ms = 80.0;
  turns[1].router_ms = 19.0;
  turns[1].tool_ms = 35.0;
  turns[1].llm_ttft_ms = 28.0;
  turns[1].llm_total_ms = 140.0;
  turns[1].tts_first_sample_ms = 120.0;
  turns[1].ttfs_ms = 594.0;
  turns[1].tok_per_sec = 72.0;

  auto plots = sphinx::render_session_svgs ("unit_test", svg_dir, turns);
  if (!plots)
    {
      return false;
    }
  if (plots->main_render_ms > 500.0 || plots->latency_render_ms > 500.0
      || plots->router_render_ms > 500.0)
    {
      return false;
    }
  if (!sphinx::validate_svg_xml (plots->main_svg_path)
      || !sphinx::validate_svg_xml (plots->latency_svg_path)
      || !sphinx::validate_svg_xml (plots->router_svg_path))
    {
      return false;
    }

  if (std::filesystem::exists (sphinx::default_session_home_path (
                                   "models/Qwen3-Coder-Next-Q4_K_S.gguf"),
                               ec))
    {
      const std::filesystem::path std_reg
          = sphinx::default_session_home_path ("models/registry.db");
      auto reg_ok = sphinx::ensure_models_registered (std_reg);
      if (!reg_ok)
        {
          (void)std::fprintf (stderr,
                              "ensure_models_registered failed on std_reg\n");
          return false;
        }
      auto q80 = sphinx::resolve_model_from_registry (
          std_reg, "qwen3-coder-next", 1U, true);
      if (!q80
          || (q80->model_id != "qwen3-coder-next-80b-q4ks"
              && q80->model_id != "qwen3-coder-next"))
        {
          (void)std::fprintf (
              stderr, "Failed to resolve qwen3-coder-next-80b-q4ks\n");
          return false;
        }
    }
  return true;
}

[[nodiscard]] auto
dot_product (std::span<const float> a, std::span<const float> b) noexcept
    -> float
{
  if (a.size () != b.size ())
    {
      return 0.0F;
    }
  double sum = 0.0;
  for (std::size_t i = 0U; i < a.size (); ++i)
    {
      sum += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return static_cast<float> (sum);
}

auto
test_embedding_client () -> bool
{
  // 1. Missing model file -> graceful degradation (memory disabled, no crash).
  sphinx::EmbeddingClientConfig missing_cfg;
  missing_cfg.model_path = "/tmp/nonexistent_qwen3_embedding_model.gguf";
  auto missing_client = sphinx::EmbeddingClient::create (missing_cfg);
  if (!missing_client || (*missing_client)->is_enabled ())
    {
      (void)std::fprintf (stderr,
                          "Expected graceful degradation on missing model\n");
      return false;
    }
  auto disabled_emb = (*missing_client)->embed ("hello world");
  if (disabled_emb)
    {
      (void)std::fprintf (stderr,
                          "Expected embed() error when client is disabled\n");
      return false;
    }

  // 2. Real Qwen3-Embedding-0.6B-Q8_0.gguf model on CPU.
  sphinx::EmbeddingClientConfig cfg;
  cfg.model_path = sphinx::default_embedding_model_path ();
  if (!std::filesystem::exists (cfg.model_path))
    {
      (void)std::fprintf (stderr, "Embedding model file missing at %s\n",
                          cfg.model_path.c_str ());
      return false;
    }

  auto client_res = sphinx::EmbeddingClient::create (cfg);
  if (!client_res || !(*client_res)->is_enabled ())
    {
      (void)std::fprintf (stderr, "Failed to create EmbeddingClient\n");
      return false;
    }
  sphinx::EmbeddingClient &client = **client_res;
  if (client.dimension () != 1024U)
    {
      (void)std::fprintf (stderr, "Expected dimension 1024, got %zu\n",
                          client.dimension ());
      return false;
    }

  const auto t0 = std::chrono::steady_clock::now ();
  auto hello_emb = client.embed ("hello world");
  const auto t1 = std::chrono::steady_clock::now ();
  const double embed_ms
      = std::chrono::duration<double, std::milli> (t1 - t0).count ();
  if (!hello_emb || hello_emb->size () != 1024U)
    {
      (void)std::fprintf (stderr, "embed('hello world') failed\n");
      return false;
    }
  if (embed_ms >= 50.0)
    {
      (void)std::fprintf (stderr, "embed() latency %.2f ms >= 50 ms limit\n",
                          embed_ms);
      return false;
    }

  const float norm = dot_product (*hello_emb, *hello_emb);
  if (std::fabs (norm - 1.0F) > 1.0e-3F)
    {
      (void)std::fprintf (stderr, "Expected unit vector norm ~1.0, got %.6f\n",
                          static_cast<double> (norm));
      return false;
    }

  auto emb_a = client.embed ("the user's name is Kival");
  auto emb_b = client.embed ("the user is called Kival");
  auto emb_c = client.embed ("the weather is sunny");
  if (!emb_a || !emb_b || !emb_c)
    {
      return false;
    }

  const float sim_ab = dot_product (*emb_a, *emb_b);
  const float sim_ac = dot_product (*emb_a, *emb_c);
  if (sim_ab <= 0.80F)
    {
      (void)std::fprintf (
          stderr, "Expected similar pair cosine > 0.80, got %.4f\n",
          static_cast<double> (sim_ab));
      return false;
    }
  if (sim_ac >= 0.50F)
    {
      (void)std::fprintf (
          stderr, "Expected unrelated pair cosine < 0.50, got %.4f\n",
          static_cast<double> (sim_ac));
      return false;
    }
  return true;
}

auto
test_memory_store () -> bool
{
  const std::filesystem::path mem_dir = "/tmp/sphinx_unit_memory_store";
  std::error_code ec;
  std::filesystem::remove_all (mem_dir, ec);

  sphinx::EmbeddingClientConfig ecfg;
  auto client_res = sphinx::EmbeddingClient::create (ecfg);
  if (!client_res || !(*client_res)->is_enabled ())
    {
      return false;
    }
  sphinx::EmbeddingClient &client = **client_res;

  constexpr uint64_t kNowNs = 1800000000ULL * 1000000000ULL;
  constexpr uint64_t kDayNs = 86400ULL * 1000000000ULL;

  // 1. Store 5 memories, close, reopen, verify all 5 survive.
  sphinx::MemoryId sem_old_id = 0U;
  sphinx::MemoryId corr_id = 0U;
  {
    auto store_res = sphinx::MemoryStore::open (mem_dir, client.dimension ());
    if (!store_res)
      {
        (void)std::fprintf (stderr, "Failed to open MemoryStore\n");
        return false;
      }
    sphinx::MemoryStore &store = **store_res;

    const std::array<std::string_view, 3U> kFacts = { {
        "The user's workstation is called Silver.",
        "The user's GPU is an NVIDIA RTX 3070 with 8 GB VRAM.",
        "The user is building a native C++23 voice assistant called Sphinx.",
    } };
    for (const std::string_view f : kFacts)
      {
        auto emb = client.embed (f);
        if (!emb
            || !store.store (f, *emb, sphinx::MemoryType::SEMANTIC, kNowNs))
          {
            return false;
          }
      }

    // Store a 60-day-old EPISODIC memory.
    auto ep_emb = client.embed (
        "In our last time session we talked about MoE offloading.");
    if (!ep_emb
        || !store.store (
            "In our last time session we talked about MoE offloading.",
            *ep_emb, sphinx::MemoryType::EPISODIC, kNowNs - 60ULL * kDayNs))
      {
        return false;
      }

    // Store a SEMANTIC memory that will be superseded by a CORRECTION.
    auto old_proj_emb = client.embed ("The user's project is Silver-2.");
    auto old_id_res = store.store ("The user's project is Silver-2.",
                                   *old_proj_emb, sphinx::MemoryType::SEMANTIC,
                                   kNowNs);
    if (!old_id_res)
      {
        return false;
      }
    sem_old_id = *old_id_res;

    if (store.size () != 5U)
      {
        (void)std::fprintf (stderr, "Expected 5 entries, got %zu\n",
                            store.size ());
        return false;
      }
    (void)store.close ();
  }

  // 2. Reopen and verify all 5 survived WAL replay, check 60-day decay
  //    (~0.135), and test supersede().
  {
    auto store_res = sphinx::MemoryStore::open (mem_dir, client.dimension ());
    if (!store_res || (*store_res)->size () != 5U)
      {
        (void)std::fprintf (stderr, "WAL replay failed on reopen\n");
        return false;
      }
    sphinx::MemoryStore &store = **store_res;

    auto ep_q = client.embed (
        "In our last time session we talked about MoE offloading.");
    auto ep_ret
        = store.retrieve (*ep_q, 1U, sphinx::MemoryType::EPISODIC, kNowNs);
    if (!ep_ret || ep_ret->empty ())
      {
        (void)std::fprintf (stderr, "Failed to retrieve EPISODIC memory\n");
        return false;
      }
    // Identical vector has similarity ~1.0; after 60 days with 30-day
    // half-life decay, effective_score = 1.0 * exp(-60/30) = 0.135335.
    if (std::fabs (ep_ret->front ().effective_score - 0.135335F) > 0.01F)
      {
        (void)std::fprintf (
            stderr, "Expected 60-day episodic score ~0.135, got %.4f\n",
            static_cast<double> (ep_ret->front ().effective_score));
        return false;
      }

    // Store CORRECTION and supersede old SEMANTIC memory.
    auto corr_emb
        = client.embed ("The user's project is Silver, not Silver-2.");
    auto cid_res = store.store ("The user's project is Silver, not Silver-2.",
                                *corr_emb, sphinx::MemoryType::CORRECTION,
                                kNowNs);
    if (!cid_res)
      {
        return false;
      }
    corr_id = *cid_res;
    if (!store.supersede (sem_old_id, corr_id))
      {
        (void)std::fprintf (stderr, "supersede() failed\n");
        return false;
      }
    (void)store.close ();
  }

  // 3. Delete HNSW index file, reopen, verify HNSW rebuilds from WAL and
  //    superseded SEMANTIC entry is excluded while CORRECTION is returned.
  std::filesystem::remove (mem_dir / "sphinx_memory_hnsw", ec);
  {
    auto store_res = sphinx::MemoryStore::open (mem_dir, client.dimension ());
    if (!store_res || (*store_res)->size () != 6U)
      {
        (void)std::fprintf (stderr, "HNSW rebuild after deletion failed\n");
        return false;
      }
    sphinx::MemoryStore &store = **store_res;
    if (!std::filesystem::exists (mem_dir / "sphinx_memory_hnsw"))
      {
        (void)std::fprintf (stderr,
                            "Expected sphinx_memory_hnsw to be rebuilt\n");
        return false;
      }

    auto q_emb = client.embed ("What is my project called?");
    auto ret = store.retrieve (*q_emb, 6U, sphinx::MemoryType::ANY, kNowNs);
    if (!ret || ret->empty ())
      {
        return false;
      }
    bool found_corr = false;
    for (const auto &m : *ret)
      {
        if (m.id == sem_old_id)
          {
            (void)std::fprintf (stderr,
                                "Superseded SEMANTIC memory was returned\n");
            return false;
          }
        if (m.id == corr_id && m.type == sphinx::MemoryType::CORRECTION)
          {
            found_corr = true;
          }
      }
    if (!found_corr)
      {
        (void)std::fprintf (stderr,
                            "Expected CORRECTION memory in retrieve()\n");
        return false;
      }

    // 4. Verify unrelated CORRECTION entry does not cross threshold on
    //    "Who are you?".
    auto stale_corr_emb = client.embed (
        "You missunderstood... I want to know in what directory example.cpp "
        "is.");
    if (!stale_corr_emb
        || !store.store ("You missunderstood... I want to know in what "
                         "directory example.cpp is.",
                         *stale_corr_emb, sphinx::MemoryType::CORRECTION,
                         kNowNs))
      {
        return false;
      }
    auto who_q = client.embed ("Who are you?");
    auto who_ret
        = store.retrieve (*who_q, 6U, sphinx::MemoryType::ANY, kNowNs);
    if (!who_ret)
      {
        return false;
      }
    const std::string who_prompt_block = sphinx::format_memories_for_prompt (
        *who_ret, sphinx::kDefaultMemoryThreshold, kNowNs);
    if (!who_prompt_block.empty ())
      {
        (void)std::fprintf (
            stderr,
            "Unexpected memory injection on 'Who are you?':\n%s\n",
            who_prompt_block.c_str ());
        return false;
      }

    // 5. Verify MemoryStore::clear() wipes all entries in memory and on disk.
    if (!store.clear () || store.size () != 0U)
      {
        (void)std::fprintf (stderr, "MemoryStore::clear() failed\n");
        return false;
      }
    (void)store.close ();
  }

  {
    auto reopened = sphinx::MemoryStore::open (mem_dir, client.dimension ());
    if (!reopened || (*reopened)->size () != 0U)
      {
        (void)std::fprintf (
            stderr, "Expected 0 entries after clear() and reopen, got %zu\n",
            reopened ? (*reopened)->size () : 999U);
        return false;
      }
    (void)reopened->get ()->close ();
  }

  std::filesystem::remove_all (mem_dir, ec);
  return true;
}

auto
test_memory_extraction_policies () -> bool
{
  const auto facts = sphinx::extract_explicit_semantic_facts (
      "Remember that my name is Kival and my project is called Silver.");
  if (facts.size () != 2U)
    {
      (void)std::fprintf (stderr, "Expected 2 extracted facts, got %zu\n",
                          facts.size ());
      return false;
    }
  if (facts[0].find ("Kival") == std::string::npos
      || facts[1].find ("Silver") == std::string::npos)
    {
      (void)std::fprintf (stderr, "Unexpected extracted facts: '%s', '%s'\n",
                          facts[0].c_str (), facts[1].c_str ());
      return false;
    }

  const auto corr1 = sphinx::extract_explicit_correction (
      "no, my project is called Silver, not Silver-2");
  const auto corr2 = sphinx::extract_explicit_correction (
      "Actually, my name is Silver, not Kival.");
  const auto corr3 = sphinx::extract_explicit_correction (
      "No, Kaio-Ken belongs to the Dragon Ball fictional universe");
  const auto non_corr = sphinx::extract_explicit_correction (
      "Actually, just tell me the 10th Fibonacci number.");
  const auto non_corr_clarify = sphinx::extract_explicit_correction (
      "No, you missunderstood... I want to know in what directory "
      "example.cpp is");
  const auto non_corr_meta = sphinx::extract_explicit_correction (
      "No, I am going to tell you what I know.");
  if (!corr1.has_value () || !corr2.has_value () || !corr3.has_value ()
      || non_corr.has_value () || non_corr_clarify.has_value ()
      || non_corr_meta.has_value ())
    {
      (void)std::fprintf (stderr, "Correction extraction test failed\n");
      return false;
    }

  sphinx::MemoryEntry sample_corr{};
  sample_corr.id = 1U;
  sample_corr.type = sphinx::MemoryType::CORRECTION;
  sample_corr.text = "Kaio-Ken belongs to the Dragon Ball fictional universe.";
  sample_corr.effective_score = 0.93F;
  const std::array<sphinx::MemoryEntry, 1U> sample_mems = { sample_corr };
  const std::string fmt_block = sphinx::format_memories_for_prompt (
      sample_mems, sphinx::kDefaultMemoryThreshold);
  if (fmt_block.find (
          "=== PERSISTENT USER MEMORIES & CORRECTIONS (HIGHEST PRIORITY) ===")
          == std::string::npos
      || fmt_block.find ("Mandatory: You MUST prioritize the facts")
             == std::string::npos)
    {
      (void)std::fprintf (stderr, "format_memories_for_prompt test failed\n");
      return false;
    }

  if (!sphinx::is_trivial_greeting_turn ("hello")
      || !sphinx::is_trivial_greeting_turn ("bye")
      || !sphinx::is_trivial_greeting_turn ("clear memory")
      || sphinx::is_trivial_greeting_turn (
          "Remember that my project is Silver"))
    {
      (void)std::fprintf (stderr, "Greeting filter test failed\n");
      return false;
    }

  if (!sphinx::is_clear_memory_command ("/clear-memory")
      || !sphinx::is_clear_memory_command ("clear memory")
      || !sphinx::is_clear_memory_command ("erase memory")
      || !sphinx::is_clear_memory_command ("reset memory")
      || !sphinx::is_clear_memory_command ("Please erase your memory.")
      || !sphinx::is_clear_memory_command ("forget everything")
      || sphinx::is_clear_memory_command ("What is my project called?"))
    {
      (void)std::fprintf (stderr, "is_clear_memory_command test failed\n");
      return false;
    }

  return true;
}

auto
test_trust_fixes_unit () -> bool
{
  // 1. Per-tool truncation table & marker.
  const std::string exact_pwd (256U, 'a');
  if (sphinx::truncate_tool_output ("pwd", exact_pwd) != exact_pwd)
    {
      (void)std::fprintf (stderr, "Expected verbatim pwd at 256 bytes\n");
      return false;
    }
  const std::string over_pwd (266U, 'b');
  const std::string trunc_pwd
      = sphinx::truncate_tool_output ("pwd", over_pwd);
  if (trunc_pwd.find ("\n...[truncated 10 bytes]...\n") == std::string::npos)
    {
      (void)std::fprintf (stderr, "Missing truncation marker on pwd: %s\n",
                          trunc_pwd.c_str ());
      return false;
    }

  // 100 KB read_file truncation: head=16384, tail=4096, omitted=81920.
  std::string big_file (102400U, 'M');
  for (std::size_t i = 0U; i < 16U; ++i)
    {
      big_file[i] = 'H';
      big_file[big_file.size () - 16U + i] = 'T';
    }
  const std::string trunc_read
      = sphinx::truncate_tool_output ("read_file", big_file);
  if (!trunc_read.starts_with ("HHHHHHHHHHHHHHHH")
      || !trunc_read.ends_with ("TTTTTTTTTTTTTTTT")
      || trunc_read.find ("\n...[truncated 81920 bytes]...\n")
             == std::string::npos)
    {
      (void)std::fprintf (stderr, "100 KB read_file truncation failed\n");
      return false;
    }

  // 2. Outside-sandbox error messages name requested path and sandbox root.
  const std::filesystem::path sb = "/tmp/sphinx_trust_unit_sb";
  std::error_code ec;
  std::filesystem::remove_all (sb, ec);
  auto sb_res = sphinx::ToolEnvironment::set_sandbox_dir (sb);
  if (!sb_res)
    {
      return false;
    }
  sutra::ToolRegistry reg;
  if (!sphinx::register_all_sphinx_tools (reg))
    {
      return false;
    }
  sutra::ToolContext ctx{};
  const std::string out_path
      = sphinx::default_session_home_path ("sphinx_test.txt").string ();
  auto w_out = reg.invoke (
      "write_file", ctx,
      std::string ("{\"path\":\"") + out_path
          + "\",\"content\":\"hello\"}");
  if (w_out || !w_out.error ().is_tool_sandbox_violation ()
      || w_out.error ().message ().find (out_path) == std::string_view::npos
      || w_out.error ().message ().find (sb_res->string ())
             == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "write_file sandbox error format failed\n");
      return false;
    }
  std::filesystem::remove_all (sb, ec);

  // 3. Meta-question keyword routing vs action routing.
  if (sphinx::TurnRouter::classify_keyword_fallback (
          "What Linux commands can you run in my system")
          != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback ("What can you do?")
             != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "What tools do you have?")
             != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback ("Run pwd")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "List the files in my workspace")
             != sphinx::Route::TASK)
    {
      (void)std::fprintf (stderr, "Meta-question keyword fallback failed\n");
      return false;
    }

  return true;
}

auto
test_system_observation_unit () -> bool
{
  // Phase 1: gather_session_profile budget, size, and graceful GPU skip.
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string prof = sphinx::gather_session_profile (1500U, 500U, true);
  const double elapsed_ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  if (elapsed_ms > 1500.0 || prof.empty () || prof.size () > 1000U
      || prof.find ("=== SYSTEM PROFILE") == std::string::npos
      || prof.find ("OS: ") == std::string::npos
      || prof.find ("Kernel: ") == std::string::npos
      || prof.find ("Build tools: ") == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Profile check failed (ms=%.1f, len=%zu):\n%s\n",
          elapsed_ms, prof.size (), prof.c_str ());
      return false;
    }

  const std::string prof_no_gpu
      = sphinx::gather_session_profile (1500U, 500U, false);
  if (prof_no_gpu.find ("GPU: ") != std::string::npos
      || prof_no_gpu.find ("OS: ") == std::string::npos)
    {
      (void)std::fprintf (stderr,
                          "Profile no-GPU degradation check failed\n");
      return false;
    }

  if (sphinx::TurnRouter::classify_keyword_fallback ("What OS am I running?")
          != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Do I have cmake installed?")
             != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "How much disk do I have free?")
             != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "What SSH hosts do I have?")
             != sphinx::Route::CHAT)
    {
      (void)std::fprintf (stderr, "Profile question routing failed\n");
      return false;
    }

  // Phase 2: run_readonly allowed commands & defense-in-depth refusals.
  const std::array<std::string_view, 1U> kUnameArgs = { { "-r" } };
  auto r_uname = sphinx::run_readonly ("uname", kUnameArgs);
  if (!r_uname || r_uname->exit_code != 0 || r_uname->stdout_str.empty ())
    {
      (void)std::fprintf (stderr, "run_readonly(uname -r) failed\n");
      return false;
    }

  const std::array<std::string_view, 1U> kFreeArgs = { { "-h" } };
  auto r_free = sphinx::run_readonly ("free", kFreeArgs);
  if (!r_free || r_free->exit_code != 0
      || r_free->stdout_str.find ("Mem:") == std::string::npos)
    {
      (void)std::fprintf (stderr, "run_readonly(free -h) failed\n");
      return false;
    }

  const std::array<std::string_view, 1U> kSsArgs = { { "-tlnp" } };
  auto r_ss = sphinx::run_readonly ("ss", kSsArgs);
  if (!r_ss || r_ss->exit_code != 0)
    {
      (void)std::fprintf (stderr, "run_readonly(ss -tlnp) failed\n");
      return false;
    }

  const std::array<std::string_view, 2U> kSshdArgs
      = { { "is-active", "sshd" } };
  auto r_sshd = sphinx::run_readonly ("systemctl", kSshdArgs);
  if (!r_sshd || r_sshd->stdout_str.find ("active") == std::string::npos)
    {
      (void)std::fprintf (stderr,
                          "run_readonly(systemctl is-active sshd) failed\n");
      return false;
    }

  const std::array<std::string_view, 4U> kJctlArgs
      = { { "-k", "-n", "5", "--no-pager" } };
  auto r_jctl = sphinx::run_readonly ("journalctl", kJctlArgs);
  if (!r_jctl)
    {
      (void)std::fprintf (stderr, "run_readonly(journalctl) failed\n");
      return false;
    }

  const std::array<std::string_view, 2U> kRmArgs = { { "-rf", "/tmp" } };
  auto r_rm = sphinx::run_readonly ("rm", kRmArgs);
  if (r_rm || !r_rm.error ().is_tool_not_allowed ()
      || r_rm.error ().message ().find ("'rm'") == std::string_view::npos
      || r_rm.error ().message ().find ("Allowed commands:")
             == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "run_readonly(rm -rf /tmp) check failed\n");
      return false;
    }

  const std::array<std::string_view, 3U> kSudoArgs
      = { { "sudo", "restart", "sshd" } };
  auto r_sudo = sphinx::run_readonly ("systemctl", kSudoArgs);
  if (r_sudo || !r_sudo.error ().is_tool_not_allowed ()
      || r_sudo.error ().message ().find ("sudo") == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "run_readonly(sudo) check failed\n");
      return false;
    }

  const std::array<std::string_view, 2U> kRestartArgs
      = { { "restart", "sshd" } };
  auto r_restart = sphinx::run_readonly ("systemctl", kRestartArgs);
  if (r_restart || !r_restart.error ().is_tool_not_allowed ()
      || r_restart.error ().message ().find ("'restart'")
             == std::string_view::npos
      || r_restart.error ().message ().find ("is-active")
             == std::string_view::npos)
    {
      (void)std::fprintf (stderr,
                          "run_readonly(systemctl restart) check failed\n");
      return false;
    }

  const std::array<std::string_view, 2U> kAptInstallArgs
      = { { "install", "vim" } };
  auto r_apt = sphinx::run_readonly ("apt", kAptInstallArgs);
  if (r_apt || !r_apt.error ().is_tool_not_allowed ()
      || r_apt.error ().message ().find ("'install'")
             == std::string_view::npos
      || r_apt.error ().message ().find ("search") == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "run_readonly(apt install) check failed\n");
      return false;
    }

  const std::array<std::string_view, 1U> kVacuumArgs
      = { { "--vacuum-time=1d" } };
  auto r_vac = sphinx::run_readonly ("journalctl", kVacuumArgs);
  if (r_vac || !r_vac.error ().is_tool_not_allowed ()
      || r_vac.error ().message ().find ("--vacuum")
             == std::string_view::npos)
    {
      (void)std::fprintf (stderr,
                          "run_readonly(journalctl --vacuum) check failed\n");
      return false;
    }

  const std::array<std::string_view, 1U> kDmesgClearArgs = { { "-c" } };
  auto r_dmesg = sphinx::run_readonly ("dmesg", kDmesgClearArgs);
  if (r_dmesg || !r_dmesg.error ().is_tool_not_allowed ()
      || r_dmesg.error ().message ().find ("-c") == std::string_view::npos)
    {
      (void)std::fprintf (stderr, "run_readonly(dmesg -c) check failed\n");
      return false;
    }

  auto r_slash
      = sphinx::run_readonly ("/bin/ls", std::span<const std::string_view>{});
  const std::array<std::string_view, 1U> kMetaArgs = { { "; rm -rf /" } };
  auto r_meta = sphinx::run_readonly ("ls", kMetaArgs);
  if (r_slash || r_meta || !r_slash.error ().is_tool_not_allowed ()
      || !r_meta.error ().is_tool_not_allowed ())
    {
      (void)std::fprintf (stderr,
                          "run_readonly slash/metachar check failed\n");
      return false;
    }

  // Phase 3: read_file_system allowlist, secret denylist, symlink, binary.
  auto fs_os = sphinx::read_file_system ("/etc/os-release");
  if (!fs_os || fs_os->find ("PRETTY_NAME") == std::string::npos)
    {
      (void)std::fprintf (stderr,
                          "read_file_system(/etc/os-release) failed\n");
      return false;
    }

  auto fs_sshd = sphinx::read_file_system ("/etc/ssh/sshd_config");
  if (!fs_sshd || fs_sshd->empty ())
    {
      (void)std::fprintf (stderr,
                          "read_file_system(/etc/ssh/sshd_config) failed\n");
      return false;
    }

  auto fs_shadow = sphinx::read_file_system ("/etc/shadow");
  auto fs_shadow_rel = sphinx::read_file_system ("/etc/../etc/shadow");
  if (fs_shadow || fs_shadow_rel
      || !fs_shadow.error ().is_tool_not_allowed ()
      || !fs_shadow_rel.error ().is_tool_not_allowed ()
      || fs_shadow.error ().message ().find ("secret material")
             == std::string_view::npos
      || fs_shadow_rel.error ().message ().find ("secret material")
             == std::string_view::npos)
    {
      (void)std::fprintf (stderr,
                          "read_file_system(/etc/shadow) check failed\n");
      return false;
    }

  auto fs_idrsa = sphinx::read_file_system (
      sphinx::default_session_home_path (".ssh/id_rsa").string ());
  if (fs_idrsa || !fs_idrsa.error ().is_tool_not_allowed ()
      || fs_idrsa.error ().message ().find ("not in system-readable paths")
             == std::string_view::npos)
    {
      (void)std::fprintf (stderr,
                          "read_file_system(id_rsa) check failed\n");
      return false;
    }

  // Symlink inside sandbox pointing to /etc/passwd resolves to /etc/passwd.
  const std::filesystem::path sym_dir = "/tmp/sphinx_obs_symlink_test";
  std::error_code ec;
  std::filesystem::remove_all (sym_dir, ec);
  std::filesystem::create_directories (sym_dir, ec);
  const std::filesystem::path sym_p = sym_dir / "link_passwd";
  std::filesystem::create_symlink ("/etc/passwd", sym_p, ec);
  auto fs_sym = sphinx::read_file_system (sym_p.string ());
  std::filesystem::remove_all (sym_dir, ec);
  if (!fs_sym || fs_sym->find ("root:") == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "read_file_system symlink -> /etc/passwd failed\n");
      return false;
    }

  // Binary file detection on /etc/ld.so.cache.
  if (std::filesystem::exists ("/etc/ld.so.cache", ec))
    {
      auto fs_bin = sphinx::read_file_system ("/etc/ld.so.cache");
      if (fs_bin
          || fs_bin.error ().message ().find ("binary file")
                 == std::string_view::npos)
        {
          (void)std::fprintf (stderr,
                              "read_file_system binary detection failed\n");
          return false;
        }
    }

  return true;
}

auto
test_comprehension_fixes_unit () -> bool
{
  // Phase 1, 2, 3 keyword fallback routing checks.
  if (sphinx::TurnRouter::classify_keyword_fallback (
          "Can you generate images?")
          != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Can you create images?")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Create the image of a house")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "I want you to code an image viewer from scratch using C++")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Can you audit the code of the written file and tell me what it "
             "does?")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Compile task_code.cpp and tell me if it has any errors")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Inspect Aurora/CMakeLists.txt and verify that the binary "
             "compiles")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Explain what a ring buffer is in one sentence.")
             != sphinx::Route::CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Are you sure the sky is blue?")
             != sphinx::Route::CHAT
      || sphinx::route_to_string (sphinx::Route::BIG_CHAT) != "BIG_CHAT"
      || sphinx::TurnRouter::classify_keyword_fallback (
             "Give me a comprehensive summary of how the greatest thinkers "
             "in history think")
             != sphinx::Route::BIG_CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "/big What is the capital of France?")
             != sphinx::Route::BIG_CHAT
      || sphinx::TurnRouter::classify_keyword_fallback (
             "/task Explain how SSH works.")
             != sphinx::Route::TASK
      || sphinx::TurnRouter::classify_keyword_fallback (
             "/chat Give me a comprehensive summary of philosophy.")
             != sphinx::Route::CHAT)
    {
      (void)std::fprintf (
          stderr, "Comprehension fixes keyword fallback check failed\n");
      return false;
    }

  const std::string c_quoted = sphinx::extract_file_content (
      "I want you to create a new text file named example2.txt and add "
      "\"This line was added by Sphinx\" to such file",
      "example2.txt");
  const std::string c_single = sphinx::extract_file_content (
      "Create a file named \"example2.txt\" and write 'Hello from Sphinx!' "
      "to it",
      "example2.txt");
  const std::string c_fenced = sphinx::extract_file_content (
      "Create script.sh with ```sh\necho ok\n```", "script.sh");
  const std::string c_unquoted = sphinx::extract_file_content (
      "Create a file named example2.txt and add Unquoted line here to such "
      "file.",
      "example2.txt");
  const std::string c_with_content = sphinx::extract_file_content (
      "Create a file at test_write.txt with content hello", "test_write.txt");
  const std::string c_rust = sphinx::extract_file_content (
      "Create a hello_world.rs file", "hello_world.rs");
  const std::string c_cpp = sphinx::extract_file_content (
      "Create aurora_hello.cpp", "aurora_hello.cpp");
  if (c_quoted != "This line was added by Sphinx\n"
      || c_single != "Hello from Sphinx!\n"
      || c_fenced != "echo ok\n"
      || c_unquoted != "Unquoted line here\n"
      || c_with_content != "hello"
      || !c_rust.empty ()
      || !c_cpp.empty ())
    {
      (void)std::fprintf (stderr, "extract_file_content check failed\n");
      return false;
    }

  sphinx::SphinxSessionConfig default_cfg{};
  if (!default_cfg.lazy_load_big_model || default_cfg.low_memory_mode
      || default_cfg.unload_big_after_task
      || default_cfg.skip_tts_synthesis)
    {
      (void)std::fprintf (stderr,
                          "SphinxSessionConfig default check failed\n");
      return false;
    }

  const std::filesystem::path tmp_gguf = "/tmp/sphinx_unit_direct_model.gguf";
  {
    std::ofstream ofs (tmp_gguf, std::ios::binary | std::ios::trunc);
    ofs << "GGUF";
  }
  const auto direct_res = sphinx::resolve_model_from_registry (
      "/tmp/sphinx_unit_nonexistent_reg.json", tmp_gguf.string (), false);
  std::error_code rm_ec;
  std::filesystem::remove (tmp_gguf, rm_ec);
  if (!direct_res || direct_res->artifact_path != tmp_gguf)
    {
      (void)std::fprintf (stderr, "Direct .gguf path resolution failed\n");
      return false;
    }

  sutra::ToolCallParser multi_parser (sutra::ToolCallStyle::Hermes);
  const std::string multi_xml
      = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": "
        "\"Aurora/src/core/window.h\"}}\n</tool_call>\n"
        "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": "
        "\"Aurora/src/core/input.h\"}}\n</tool_call>\n"
        "<tool_call>\n<function=read_file><parameter=path>"
        "Aurora/src/md3/components/button.h</parameter></function>\n"
        "<function=read_file><parameter=path>"
        "Aurora/src/app/main.cpp</parameter></function>\n</tool_call>";
  const auto multi_calls = multi_parser.parse_complete (multi_xml);
  if (multi_calls.size () != 4U
      || multi_calls[0].name != "read_file"
      || multi_calls[3].name != "read_file")
    {
      (void)std::fprintf (stderr, "Multi tool_call parsing check failed\n");
      return false;
    }

  const std::filesystem::path tmp_sb = "/tmp/sphinx_unit_multi_read_sb";
  std::filesystem::remove_all (tmp_sb, rm_ec);
  std::filesystem::create_directories (tmp_sb / "Aurora/src/core", rm_ec);
  std::filesystem::create_directories (tmp_sb / "Aurora/src/md3/components",
                                       rm_ec);
  std::filesystem::create_directories (tmp_sb / "Aurora/src/app", rm_ec);
  {
    std::ofstream (tmp_sb / "Aurora/src/core/window.h") << "// window\n";
    std::ofstream (tmp_sb / "Aurora/src/core/input.h") << "// input\n";
    std::ofstream (tmp_sb / "Aurora/src/md3/components/button.h")
        << "// button\n";
    std::ofstream (tmp_sb / "Aurora/src/app/main.cpp") << "// main\n";
  }
  const auto pre_files = sphinx::extract_explicit_read_files (
      "Read Aurora/src/core/window.h, Aurora/src/core/input.h, "
      "Aurora/src/md3/components/button.h, and Aurora/src/app/main.cpp "
      "using read_file. Then write aurora_hello_btn.cpp.",
      tmp_sb);
  std::filesystem::remove_all (tmp_sb, rm_ec);
  if (pre_files.size () != 4U
      || pre_files[0] != "Aurora/src/core/window.h"
      || pre_files[1] != "Aurora/src/core/input.h"
      || pre_files[2] != "Aurora/src/md3/components/button.h"
      || pre_files[3] != "Aurora/src/app/main.cpp")
    {
      (void)std::fprintf (stderr,
                          "extract_explicit_read_files check failed\n");
      return false;
    }

  return true;
}

} /* namespace */

auto
main () -> int
{
  if (!test_sentence_splitter ())
    {
      return 1;
    }
  if (!test_tools_and_sandbox ())
    {
      return 1;
    }
  if (!test_make_directory ())
    {
      return 1;
    }
  if (!test_registry_and_darshana ())
    {
      return 1;
    }
  if (!test_embedding_client ())
    {
      return 1;
    }
  if (!test_memory_store ())
    {
      return 1;
    }
  if (!test_memory_extraction_policies ())
    {
      return 1;
    }
  if (!test_trust_fixes_unit ())
    {
      return 1;
    }
  if (!test_system_observation_unit ())
    {
      return 1;
    }
  if (!test_comprehension_fixes_unit ())
    {
      return 1;
    }
  return 0;
}
