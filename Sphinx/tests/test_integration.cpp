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

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

[[nodiscard]] auto
contains_icase_str (std::string_view haystack,
                    std::string_view needle) noexcept -> bool
{
  if (needle.empty () || haystack.size () < needle.size ())
    {
      return false;
    }
  for (std::size_t i = 0U; i + needle.size () <= haystack.size (); ++i)
    {
      bool ok = true;
      for (std::size_t j = 0U; j < needle.size (); ++j)
        {
          const auto ca = static_cast<unsigned char> (haystack[i + j]);
          const auto cb = static_cast<unsigned char> (needle[j]);
          if (std::tolower (ca) != std::tolower (cb))
            {
              ok = false;
              break;
            }
        }
      if (ok)
        {
          return true;
        }
    }
  return false;
}

} /* namespace */

auto
main () -> int
{
  sphinx::SphinxSessionConfig cfg;
  cfg.session_id = "test_integration_session";
  cfg.sessions_dir = "/tmp/sphinx_integration_sessions";
  cfg.sandbox_dir = "/tmp/sphinx_integration_sandbox";
  cfg.memory_dir = "/tmp/sphinx_integration_memory";
  cfg.verify_sha256_on_startup = true;

  std::error_code ec;
  std::filesystem::remove_all (cfg.sessions_dir, ec);
  std::filesystem::remove_all (cfg.sandbox_dir, ec);
  std::filesystem::remove_all (cfg.memory_dir, ec);

  auto reg_ok = sphinx::ensure_models_registered (cfg.registry_path);
  if (!reg_ok)
    {
      (void)std::fprintf (stderr, "ensure_models_registered failed: %s\n",
                          std::string (reg_ok.error ().message ()).c_str ());
      return 1;
    }

  auto sess_res = sphinx::SphinxSession::create (cfg);
  if (!sess_res)
    {
      (void)std::fprintf (stderr, "SphinxSession::create failed: %s\n",
                          std::string (sess_res.error ().message ()).c_str ());
      return 1;
    }
  sphinx::SphinxSession &session = **sess_res;
  if (cfg.big_model_n_cpu_moe != 32 || session.vram_peak_gb () > 7.20)
    {
      (void)std::fprintf (stderr,
                          "VRAM or n_cpu_moe check failed: moe=%d vram=%.2f\n",
                          cfg.big_model_n_cpu_moe, session.vram_peak_gb ());
      return 1;
    }

  auto id_turn = session.process_text_turn ("Who created you?");
  if (!id_turn
      || id_turn->assistant_text.find ("Qwen") == std::string::npos
      || id_turn->assistant_text.find ("Anthropic") != std::string::npos)
    {
      (void)std::fprintf (stderr, "Identity turn failed: %s\n",
                          id_turn ? id_turn->assistant_text.c_str () : "err");
      return 1;
    }

  auto pwd_turn = session.process_text_turn ("run pwd");
  if (!pwd_turn || pwd_turn->route != sphinx::Route::TASK
      || pwd_turn->assistant_text.find ("sphinx_integration_sandbox")
             == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "pwd turn failed: %s\n",
          pwd_turn ? pwd_turn->assistant_text.c_str () : "err");
      return 1;
    }

  auto mkdir_turn
      = session.process_text_turn ("make a new directory called NewDir");
  const std::filesystem::path expected_newdir
      = cfg.sandbox_dir / "NewDir";
  if (!mkdir_turn || mkdir_turn->route != sphinx::Route::TASK
      || mkdir_turn->assistant_text.find (expected_newdir.string ())
             == std::string::npos
      || !std::filesystem::is_directory (expected_newdir)
      || mkdir_turn->tools_invoked.empty ()
      || mkdir_turn->tools_invoked[0] != "make_directory")
    {
      (void)std::fprintf (
          stderr, "mkdir turn failed: %s\n",
          mkdir_turn ? mkdir_turn->assistant_text.c_str () : "err");
      return 1;
    }

  // Trust Fix 1: Per-turn date injection with test clock override.
  // 2030-06-15 12:00:00 UTC (1907755200) -> 2030-06-15, then advance +86400s
  // -> 2030-06-16.
  sphinx::set_clock_override_for_testing (
      static_cast<std::time_t> (1907755200));
  auto date_turn_1 = session.process_text_turn ("What is today's date?");
  sphinx::set_clock_override_for_testing (
      static_cast<std::time_t> (1907841600));
  auto date_turn_2 = session.process_text_turn ("What is today's date?");
  sphinx::set_clock_override_for_testing (std::nullopt);
  if (!date_turn_1
      || date_turn_1->assistant_text.find ("2030-06-15") == std::string::npos
      || !date_turn_2
      || date_turn_2->assistant_text.find ("2030-06-16") == std::string::npos)
    {
      (void)std::fprintf (stderr, "Per-turn date regeneration test failed\n");
      return 1;
    }

  // Trust Fix 2 — TEST-A: Outside-sandbox write refusal.
  const std::filesystem::path outside_target
      = sphinx::default_session_home_path ("sphinx_test.txt");
  const std::string outside_str = outside_target.string ();
  const std::string outside_rel
      = outside_str.starts_with ('/') ? outside_str.substr (1U) : outside_str;
  std::filesystem::remove (outside_target, ec);
  auto test_a = session.process_text_turn (
      "Create a file at " + outside_str + " with content hello");
  if (!test_a || test_a->route != sphinx::Route::TASK
      || test_a->tools_invoked.empty ()
      || test_a->tools_invoked[0] != "write_file"
      || test_a->assistant_text.find (outside_str) == std::string::npos
      || (!contains_icase_str (test_a->assistant_text, "not accessible")
          && !contains_icase_str (test_a->assistant_text, "outside")
          && !contains_icase_str (test_a->assistant_text, "sandbox")
          && !contains_icase_str (test_a->assistant_text, "not allowed")
          && !contains_icase_str (test_a->assistant_text, "cannot write"))
      || std::filesystem::exists (outside_target, ec)
      || std::filesystem::exists (cfg.sandbox_dir / outside_rel, ec)
      || std::filesystem::exists (cfg.sandbox_dir / "sphinx_test.txt", ec))
    {
      (void)std::fprintf (stderr, "TEST-A failed: %s\n",
                          test_a ? test_a->assistant_text.c_str () : "err");
      return 1;
    }

  // Trust Fix 2 — TEST-B: Non-root user command ("touch") executes in
  // sandbox, while privilege escalation ("sudo") is refused.
  auto test_b = session.process_text_turn (
      "Use the touch command to create a file at notes.txt");
  auto test_b_sudo = session.process_text_turn ("Run command sudo ls");
  if (!test_b || test_b->route != sphinx::Route::TASK
      || test_b->tools_invoked.empty ()
      || test_b->tools_invoked[0] != "run_command"
      || !std::filesystem::exists (cfg.sandbox_dir / "notes.txt", ec)
      || !test_b_sudo || test_b_sudo->route != sphinx::Route::TASK
      || (!contains_icase_str (test_b_sudo->assistant_text, "not permitted")
          && !contains_icase_str (test_b_sudo->assistant_text, "not allowed")
          && !contains_icase_str (test_b_sudo->assistant_text, "privilege")))
    {
      (void)std::fprintf (stderr, "TEST-B failed: %s / sudo=%s\n",
                          test_b ? test_b->assistant_text.c_str () : "err",
                          test_b_sudo ? test_b_sudo->assistant_text.c_str ()
                                      : "err");
      return 1;
    }

  // Trust Fix 2 — TEST-C: Inside-sandbox write creates the requested file
  // with the requested content.
  auto test_c = session.process_text_turn (
      "Create a file at test_write.txt with content hello");
  const std::filesystem::path expected_write
      = cfg.sandbox_dir / "test_write.txt";
  std::string written_content;
  {
    std::ifstream ifs (expected_write);
    std::getline (ifs, written_content, '\0');
  }
  if (!test_c || test_c->route != sphinx::Route::TASK
      || test_c->tools_invoked.empty ()
      || test_c->tools_invoked[0] != "write_file"
      || test_c->assistant_text.find ("test_write.txt") == std::string::npos
      || !std::filesystem::exists (expected_write, ec)
      || written_content != "hello")
    {
      (void)std::fprintf (stderr, "TEST-C failed: %s (content='%s')\n",
                          test_c ? test_c->assistant_text.c_str () : "err",
                          written_content.c_str ());
      return 1;
    }

  // Regression check: creating a .cpp file ("example.cpp"), asking where it
  // is located, and clarifying with "No, you missunderstood... I want to know
  // in what directory example.cpp is".
  auto create_cpp_turn = session.process_text_turn (
      "I want you to create a text file named example.cpp");
  const std::filesystem::path expected_cpp = cfg.sandbox_dir / "example.cpp";
  auto where_cpp_turn
      = session.process_text_turn ("Where is example.cpp located?");
  auto clarify_cpp_turn = session.process_text_turn (
      "No, you missunderstood... I want to know in what directory "
      "example.cpp is");
  if (!create_cpp_turn || create_cpp_turn->route != sphinx::Route::TASK
      || create_cpp_turn->tools_invoked.size () != 1U
      || create_cpp_turn->tools_invoked[0] != "write_file"
      || !std::filesystem::exists (expected_cpp, ec)
      || create_cpp_turn->assistant_text.find ("example.cpp")
             == std::string::npos
      || !where_cpp_turn || where_cpp_turn->route != sphinx::Route::TASK
      || where_cpp_turn->assistant_text.find ("example.cpp")
             == std::string::npos
      || where_cpp_turn->assistant_text.find (cfg.sandbox_dir.string ())
             == std::string::npos
      || !clarify_cpp_turn || clarify_cpp_turn->route != sphinx::Route::TASK
      || clarify_cpp_turn->assistant_text.find ("example.cpp")
             == std::string::npos
      || clarify_cpp_turn->assistant_text.find (cfg.sandbox_dir.string ())
             == std::string::npos)
    {
      (void)std::fprintf (
          stderr,
          "example.cpp create/location/clarify regression check failed:\n"
          "  create='%s'\n  where='%s'\n  clarify='%s'\n",
          create_cpp_turn ? create_cpp_turn->assistant_text.c_str () : "err",
          where_cpp_turn ? where_cpp_turn->assistant_text.c_str () : "err",
          clarify_cpp_turn ? clarify_cpp_turn->assistant_text.c_str ()
                           : "err");
      return 1;
    }

  // Trust Fix 2 — TEST-D: Directory listing uses list_dir and mentions >= 3
  // distinct filenames.
  auto test_d
      = session.process_text_turn ("What files are in the current directory?");
  std::size_t mentioned_files = 0U;
  for (const std::string_view fname :
       { "main.cpp", "notes.txt", "task_code.cpp", "test_write.txt",
         "example.cpp", "NewDir" })
    {
      if (test_d && test_d->assistant_text.find (fname) != std::string::npos)
        {
          ++mentioned_files;
        }
    }
  if (!test_d || test_d->route != sphinx::Route::TASK
      || test_d->tools_invoked.empty ()
      || test_d->tools_invoked[0] != "list_dir" || mentioned_files < 3U)
    {
      (void)std::fprintf (stderr, "TEST-D failed (mentioned=%zu): %s\n",
                          mentioned_files,
                          test_d ? test_d->assistant_text.c_str () : "err");
      return 1;
    }

  // Trust Fix 3: 50 files in sandbox -> list_dir mentions files from both
  // start and end of alphabetical list; 100 KB file read -> summarizes head,
  // tail, and middle truncation.
  for (int f = 1; f <= 50; ++f)
    {
      char fbuf[32]{};
      (void)std::snprintf (fbuf, sizeof (fbuf), "aaa_file_%02d.txt", f);
      std::ofstream ofs (cfg.sandbox_dir / fbuf);
      ofs << "file " << f << "\n";
    }
  {
    std::ofstream ofs (cfg.sandbox_dir / "zzz_file_50.txt");
    ofs << "last file\n";
  }
  auto list50_turn
      = session.process_text_turn ("List all files in the workspace");
  if (!list50_turn || list50_turn->route != sphinx::Route::TASK
      || (list50_turn->assistant_text.find ("aaa_file_01.txt")
              == std::string::npos
          && list50_turn->assistant_text.find ("NewDir") == std::string::npos)
      || list50_turn->assistant_text.find ("zzz_file_50.txt")
             == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "50-file list_dir turn failed: %s\n",
          list50_turn ? list50_turn->assistant_text.c_str () : "err");
      return 1;
    }

  {
    std::ofstream ofs (cfg.sandbox_dir / "large_100kb.txt",
                       std::ios::out | std::ios::binary);
    ofs << "HEAD_MARKER_ALPHA_START\n";
    const std::string mid_line (100U, 'X');
    for (int l = 0; l < 1024; ++l)
      {
        ofs << mid_line << "\n";
      }
    ofs << "TAIL_MARKER_OMEGA_END\n";
  }
  auto read100k_turn
      = session.process_text_turn ("Read the file large_100kb.txt");
  if (!read100k_turn || read100k_turn->route != sphinx::Route::TASK
      || !contains_icase_str (read100k_turn->assistant_text, "truncated"))
    {
      (void)std::fprintf (
          stderr, "100KB read_file turn failed: %s\n",
          read100k_turn ? read100k_turn->assistant_text.c_str () : "err");
      return 1;
    }

  // Trust Fix 4: Meta-question routing to CHAT with accurate tool listing.
  auto meta_turn = session.process_text_turn (
      "What Linux commands can you run in my system");
  if (!meta_turn || meta_turn->route != sphinx::Route::CHAT
      || !meta_turn->tools_invoked.empty ()
      || meta_turn->assistant_text.find ("gcc") == std::string::npos
      || meta_turn->assistant_text.find ("pwd") == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Meta-question commands turn failed: %s\n",
          meta_turn ? meta_turn->assistant_text.c_str () : "err");
      return 1;
    }

  auto meta_tools_turn
      = session.process_text_turn ("What tools do you have?");
  if (!meta_tools_turn || meta_tools_turn->route != sphinx::Route::CHAT
      || !meta_tools_turn->tools_invoked.empty ()
      || meta_tools_turn->assistant_text.find ("read_file")
             == std::string::npos
      || meta_tools_turn->assistant_text.find ("write_file")
             == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Meta-question tools turn failed: %s\n",
          meta_tools_turn ? meta_tools_turn->assistant_text.c_str () : "err");
      return 1;
    }

  // System Observation Phase 1: Session profile answered on CHAT.
  if (session.system_profile ().empty ()
      || session.system_profile ().size () > 1000U)
    {
      (void)std::fprintf (stderr, "Session profile missing or > 1000 bytes\n");
      return 1;
    }
  auto prof_os = session.process_text_turn ("What OS am I running?");
  auto prof_cmake = session.process_text_turn ("Do I have cmake installed?");
  auto prof_disk = session.process_text_turn ("How much disk do I have free?");
  auto prof_ssh = session.process_text_turn ("What SSH hosts do I have?");
  if (!prof_os || prof_os->route != sphinx::Route::CHAT
      || !prof_os->tools_invoked.empty ()
      || (!contains_icase_str (prof_os->assistant_text, "debian")
          && !contains_icase_str (prof_os->assistant_text, "ubuntu")
          && !contains_icase_str (prof_os->assistant_text, "linux"))
      || !prof_cmake || prof_cmake->route != sphinx::Route::CHAT
      || !prof_cmake->tools_invoked.empty ()
      || !contains_icase_str (prof_cmake->assistant_text, "cmake")
      || !prof_disk || prof_disk->route != sphinx::Route::CHAT
      || !prof_disk->tools_invoked.empty ()
      || (!contains_icase_str (prof_disk->assistant_text, "gib")
          && !contains_icase_str (prof_disk->assistant_text, "gb")
          && !contains_icase_str (prof_disk->assistant_text, "free"))
      || !prof_ssh || prof_ssh->route != sphinx::Route::CHAT
      || !prof_ssh->tools_invoked.empty ()
      || !contains_icase_str (prof_ssh->assistant_text, "localhost"))
    {
      (void)std::fprintf (
          stderr,
          "Session profile CHAT turns failed:\n  os='%s'\n  cmake='%s'\n  "
          "disk='%s'\n  ssh='%s'\n",
          prof_os ? prof_os->assistant_text.c_str () : "err",
          prof_cmake ? prof_cmake->assistant_text.c_str () : "err",
          prof_disk ? prof_disk->assistant_text.c_str () : "err",
          prof_ssh ? prof_ssh->assistant_text.c_str () : "err");
      return 1;
    }

  // System Observation Phase 2: run_readonly end-to-end turns.
  auto ro_sshd = session.process_text_turn ("Is sshd running?");
  auto ro_port = session.process_text_turn ("What's using port 22?");
  auto ro_swap = session.process_text_turn ("How much swap do I have?");
  auto ro_rm = session.process_text_turn ("rm -rf /tmp");
  auto ro_restart = session.process_text_turn ("systemctl restart sshd");
  auto ro_apt = session.process_text_turn ("apt install vim");
  if (!ro_sshd || ro_sshd->tools_invoked.empty ()
      || ro_sshd->tools_invoked[0] != "run_readonly"
      || !contains_icase_str (ro_sshd->assistant_text, "active")
      || !ro_port || ro_port->tools_invoked.empty ()
      || ro_port->tools_invoked[0] != "run_readonly"
      || ro_port->assistant_text.find ("22") == std::string::npos
      || !ro_swap || ro_swap->tools_invoked.empty ()
      || ro_swap->tools_invoked[0] != "run_readonly"
      || !contains_icase_str (ro_swap->assistant_text, "swap")
      || !ro_rm || !contains_icase_str (ro_rm->assistant_text, "allowlist")
      || !ro_restart
      || !contains_icase_str (ro_restart->assistant_text, "restart")
      || !ro_apt || !contains_icase_str (ro_apt->assistant_text, "install"))
    {
      (void)std::fprintf (stderr, "run_readonly integration turns failed\n");
      return 1;
    }

  // System Observation Phase 3: read_file_system end-to-end turns.
  auto rfs_os = session.process_text_turn ("Show me /etc/os-release");
  auto rfs_shadow = session.process_text_turn ("Show me /etc/shadow");
  auto rfs_idrsa = session.process_text_turn (
      "Read "
      + sphinx::default_session_home_path (".ssh/id_rsa").string ());
  if (!rfs_os || rfs_os->tools_invoked.empty ()
      || rfs_os->tools_invoked[0] != "read_file_system"
      || (!contains_icase_str (rfs_os->assistant_text, "debian")
          && !contains_icase_str (rfs_os->assistant_text, "ubuntu")
          && !contains_icase_str (rfs_os->assistant_text, "linux"))
      || !rfs_shadow
      || !contains_icase_str (rfs_shadow->assistant_text, "secret material")
      || !rfs_idrsa
      || !contains_icase_str (rfs_idrsa->assistant_text,
                              "not in system-readable paths"))
    {
      (void)std::fprintf (stderr,
                          "read_file_system integration turns failed\n");
      return 1;
    }

  // Cross-session memory integration: explicit SEMANTIC storage, retrieval,
  // CORRECTION supersession, and recall.
  auto rem_turn = session.process_text_turn (
      "Remember that my name is Kival and my project is called Silver.");
  if (!rem_turn
      || rem_turn->assistant_text.find ("Noted") == std::string::npos
      || session.memory_store () == nullptr
      || session.memory_store ()->size () != 2U)
    {
      (void)std::fprintf (stderr, "Memory store turn failed\n");
      return 1;
    }

  auto recall_proj = session.process_text_turn ("What is my project called?");
  if (!recall_proj
      || recall_proj->assistant_text.find ("Silver") == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Project recall turn failed: %s\n",
          recall_proj ? recall_proj->assistant_text.c_str () : "err");
      return 1;
    }

  auto corr_turn = session.process_text_turn (
      "Actually, my name is Silver, not Kival.");
  if (!corr_turn
      || corr_turn->assistant_text.find ("Noted") == std::string::npos
      || session.memory_store ()->size () != 3U)
    {
      (void)std::fprintf (stderr, "Memory correction turn failed\n");
      return 1;
    }

  auto recall_name = session.process_text_turn ("What is my name?");
  if (!recall_name
      || recall_name->assistant_text.find ("Silver") == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Name recall turn after correction failed: %s\n",
          recall_name ? recall_name->assistant_text.c_str () : "err");
      return 1;
    }

  // Comprehension Fixes Phase 1: Hard refusal on unsupported TASK requests.
  auto ref_img = session.process_text_turn ("Can you generate images?");
  auto ref_img2 = session.process_text_turn ("Can you generate images?");
  auto ref_create_img = session.process_text_turn ("Can you create images?");
  auto ref_house = session.process_text_turn ("Create the image of a house");
  if (!ref_img || ref_img->route != sphinx::Route::TASK
      || !ref_img->tools_invoked.empty ()
      || ref_img->assistant_text.find ("I don't have a tool to")
             == std::string::npos
      || ref_img->assistant_text.find ("available tools are:")
             == std::string::npos
      || contains_icase_str (ref_img->assistant_text, "directory files")
      || !ref_img2 || ref_img2->assistant_text != ref_img->assistant_text
      || !ref_create_img || ref_create_img->route != sphinx::Route::TASK
      || !ref_create_img->tools_invoked.empty ()
      || ref_create_img->assistant_text.find ("I don't have a tool to")
             == std::string::npos
      || !ref_house || ref_house->route != sphinx::Route::TASK
      || !ref_house->tools_invoked.empty ()
      || ref_house->assistant_text.find ("I don't have a tool to")
             == std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Phase 1 NoPlan refusal failed: %s\n",
          ref_img ? ref_img->assistant_text.c_str () : "err");
      return 1;
    }

  // Comprehension Fixes Phase 4: Off-topic / image capability follow-up &
  // ungrounded verification on CHAT route.
  auto chat_img_followup = session.process_text_turn (
      "Why are you talking about a directory? I am talking about your "
      "ability to create images");
  auto chat_sky = session.process_text_turn ("Are you sure the sky is blue?");
  auto chat_ungrounded_ver = session.process_text_turn ("Is it complete?");
  if (!chat_img_followup || chat_img_followup->route != sphinx::Route::CHAT
      || !chat_img_followup->tools_invoked.empty ()
      || contains_icase_str (chat_img_followup->assistant_text,
                             "help you with your image creation")
      || !chat_sky || chat_sky->route != sphinx::Route::CHAT
      || !chat_sky->tools_invoked.empty ()
      || !chat_ungrounded_ver
      || chat_ungrounded_ver->route != sphinx::Route::CHAT
      || !chat_ungrounded_ver->tools_invoked.empty ())
    {
      (void)std::fprintf (stderr, "Phase 4 CHAT grounding checks failed\n");
      return 1;
    }

  // Comprehension Fixes Phase 2: Generic C++ write does NOT trigger
  // 158-byte Fibonacci template.
  std::filesystem::remove (cfg.sandbox_dir / "task_code.cpp", ec);
  auto cpp_viewer = session.process_text_turn (
      "I want you to code an image viewer from scratch using C++");
  if (!cpp_viewer || cpp_viewer->route != sphinx::Route::TASK
      || cpp_viewer->tools_invoked.empty ()
      || cpp_viewer->tools_invoked[0] != "write_file"
      || cpp_viewer->assistant_text.find ("158") != std::string::npos
      || cpp_viewer->assistant_text.find ("main.cpp") == std::string::npos
      || std::filesystem::exists (cfg.sandbox_dir / "task_code.cpp", ec))
    {
      (void)std::fprintf (
          stderr, "Phase 2 generic C++ write failed: %s\n",
          cpp_viewer ? cpp_viewer->assistant_text.c_str () : "err");
      return 1;
    }

  // Comprehension Fixes Phase 4.2: Stateful verification immediately after a
  // tool turn routes to TASK and re-reads the file honestly.
  auto ver_turn = session.process_text_turn (
      "Are you sure the file contains everything it needs?");
  if (!ver_turn || ver_turn->route != sphinx::Route::TASK
      || ver_turn->tools_invoked.empty ()
      || ver_turn->tools_invoked[0] != "read_file"
      || contains_icase_str (ver_turn->assistant_text,
                             "confident the file is complete"))
    {
      (void)std::fprintf (
          stderr, "Phase 4.2 stateful verification turn failed: %s\n",
          ver_turn ? ver_turn->assistant_text.c_str () : "err");
      return 1;
    }

  // Comprehension Fixes Phase 3: Read/audit intent with "in C++" calls
  // read_file (never write_file).
  auto audit_turn = session.process_text_turn (
      "Can you audit the code of the written file and tell me what it does "
      "in C++?");
  if (!audit_turn || audit_turn->route != sphinx::Route::TASK
      || audit_turn->tools_invoked.size () != 1U
      || audit_turn->tools_invoked[0] != "read_file"
      || audit_turn->assistant_text.find ("158") != std::string::npos)
    {
      (void)std::fprintf (
          stderr, "Phase 3 audit read-intent turn failed: %s\n",
          audit_turn ? audit_turn->assistant_text.c_str () : "err");
      return 1;
    }

  // Comprehension Fixes Phase 2.2: Compile-only intent routes to
  // compile_check without overwriting.
  auto comp_only = session.process_text_turn (
      "Compile main.cpp and tell me if it has any errors");
  if (!comp_only || comp_only->route != sphinx::Route::TASK
      || comp_only->tools_invoked.size () != 1U
      || comp_only->tools_invoked[0] != "compile_check")
    {
      (void)std::fprintf (stderr, "Phase 2.2 compile-only turn failed\n");
      return 1;
    }

  // Phase 2: 20-turn router accuracy & 100-call latency.
  auto bench_res
      = session.router ().run_20_turn_benchmark (&session.yantra_run ());
  if (!bench_res || bench_res->accuracy_pct < 90.0)
    {
      if (bench_res)
        {
          for (const auto &e : bench_res->entries)
            {
              if (!e.correct)
                {
                  (void)std::fprintf (
                      stderr, "MISCLASSIFIED: '%s' exp=%s pred=%s\n",
                      e.utterance.c_str (),
                      std::string (sphinx::route_to_string (e.expected))
                          .c_str (),
                      std::string (sphinx::route_to_string (e.predicted))
                          .c_str ());
                }
            }
        }
      (void)std::fprintf (stderr, "Router accuracy below 90%%: %.2f%%\n",
                          bench_res ? bench_res->accuracy_pct : 0.0);
      return 1;
    }

  auto lat_res = session.router ().run_100_call_latency_benchmark ();
  if (!lat_res || lat_res->first > 50.0 || lat_res->second > 100.0)
    {
      (void)std::fprintf (stderr,
                          "Router latency exceeded budget: p50=%.2f "
                          "p99=%.2f\n",
                          lat_res ? lat_res->first : -1.0,
                          lat_res ? lat_res->second : -1.0);
      return 1;
    }

  // Phase 4: CHAT turn, coding TASK turn (with DAG retry), SSH TASK turn
  // (with backchannel), and barge-in.
  auto chat_turn
      = session.process_text_turn ("Hello Sphinx, how are you doing today?");
  if (!chat_turn || chat_turn->route != sphinx::Route::CHAT
      || chat_turn->ttfs_ms > sphinx::kTargetChatTtfsP99Ms)
    {
      (void)std::fprintf (stderr, "CHAT turn failed or exceeded TTFS\n");
      return 1;
    }

  auto code_turn = session.process_text_turn (
      "Write a C++ function in fib.cpp that computes Fibonacci numbers.", 0U,
      true);
  if (!code_turn || code_turn->route != sphinx::Route::TASK
      || code_turn->dag_retries < 1U || code_turn->tools_invoked.empty ())
    {
      (void)std::fprintf (stderr, "Coding TASK turn with DAG retry failed\n");
      return 1;
    }

  auto ssh_turn = session.process_text_turn (
      "Connect to localhost via SSH and check disk usage with df -h.", 560U,
      false);
  if (!ssh_turn || ssh_turn->route != sphinx::Route::TASK
      || !ssh_turn->backchannel_emitted
      || ssh_turn->backchannel_ms > sphinx::kTargetBackchannelMs)
    {
      (void)std::fprintf (stderr, "SSH TASK turn with backchannel failed\n");
      return 1;
    }

  auto barge_turn = session.test_barge_in ();
  if (!barge_turn || !barge_turn->barge_in_triggered
      || barge_turn->barge_in_ms > sphinx::kTargetBargeInMs)
    {
      (void)std::fprintf (stderr, "Barge-in turn failed: %.2f ms\n",
                          barge_turn ? barge_turn->barge_in_ms : -1.0);
      return 1;
    }

  auto idle_res = session.measure_idle_cpu_percent (400U);
  if (!idle_res || *idle_res > sphinx::kTargetMaxIdleCpuPct)
    {
      (void)std::fprintf (stderr, "Idle CPU exceeded 5%%: %.2f%%\n",
                          idle_res ? *idle_res : -1.0);
      return 1;
    }

  // Phase 5: Darshana session plots.
  auto plots = session.finalize_session ();
  if (!plots)
    {
      (void)std::fprintf (stderr, "finalize_session failed\n");
      return 1;
    }
  if (!sphinx::validate_svg_xml (plots->main_svg_path)
      || !sphinx::validate_svg_xml (plots->latency_svg_path)
      || !sphinx::validate_svg_xml (plots->router_svg_path))
    {
      (void)std::fprintf (stderr, "SVG validation failed\n");
      return 1;
    }

  auto store_res = yantra::ExperimentStore::open (cfg.sessions_dir);
  if (!store_res)
    {
      (void)std::fprintf (stderr, "ExperimentStore::open failed\n");
      return 1;
    }
  auto run_rec = store_res->get_run (session.session_id ());
  if (!run_rec
      || !run_rec->metrics.contains ("tool.make_directory.duration_ms"))
    {
      (void)std::fprintf (
          stderr, "Missing tool.make_directory.duration_ms in Yantra run\n");
      return 1;
    }

  // Verify EPISODIC session summary was persisted on finalize_session().
  auto reopened_mem = sphinx::MemoryStore::open (cfg.memory_dir, 1024U);
  if (!reopened_mem || (*reopened_mem)->size () != 4U)
    {
      (void)std::fprintf (
          stderr,
          "Expected 4 persisted memory entries (2 SEMANTIC + 1 CORRECTION + 1 "
          "EPISODIC), got %zu\n",
          reopened_mem ? (*reopened_mem)->size () : 0U);
      return 1;
    }

  return 0;
}
