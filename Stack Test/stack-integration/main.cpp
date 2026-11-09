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

#include <sutra/llm_backends.hpp>
#include <sutra/observability.hpp>
#include <sutra/runtime.hpp>
#include <sutra/sutra.hpp>
#include <yantra/observability.hpp>
#include <yantra/registry.hpp>
#include <yantra/track.hpp>
#include <yantra/yantra.hpp>

#if defined(HAVE_DARSHANA) && HAVE_DARSHANA
#include <darshana/darshana.hpp>
#include <darshana/export.hpp>
#include <darshana/spec.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__linux__)
#include <malloc.h>
#endif

namespace fs = std::filesystem;

namespace
{

struct PromptItem
{
  int id = 0;
  std::string text;
  std::string expects;
  std::string tests;
};

struct TurnTelemetry
{
  int id = 0;
  std::string category;
  std::string prompt;
  std::size_t response_length = 0U;
  double ttft_ms = 0.0;
  double total_ms = 0.0;
  uint64_t tokens = 0U;
  double tokens_per_sec = 0.0;
  bool logged_on_async_worker = false;
};

struct CheckStatus
{
  bool passed = false;
  std::string name;
  std::string details;
  std::string failure_message;
};

struct RssBreakdown
{
  double vm_rss_mb = 0.0;
  double rss_anon_mb = 0.0;
  double rss_file_mb = 0.0;
  double rss_shmem_mb = 0.0;
};

auto
read_rss_breakdown () -> RssBreakdown
{
  RssBreakdown out;
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      auto parse_kb = [&line] (std::string_view prefix) -> double {
        if (!line.starts_with (prefix))
          {
            return -1.0;
          }
        std::istringstream iss (line.substr (prefix.size ()));
        long kb = 0;
        iss >> kb;
        return static_cast<double> (kb) / 1024.0;
      };
      if (const double v = parse_kb ("VmRSS:"); v >= 0.0)
        {
          out.vm_rss_mb = v;
        }
      else if (const double a = parse_kb ("RssAnon:"); a >= 0.0)
        {
          out.rss_anon_mb = a;
        }
      else if (const double f = parse_kb ("RssFile:"); f >= 0.0)
        {
          out.rss_file_mb = f;
        }
      else if (const double s = parse_kb ("RssShmem:"); s >= 0.0)
        {
          out.rss_shmem_mb = s;
        }
    }
  return out;
}

auto
hash_file_sha256 (const fs::path &path) -> yantra::Result<std::string>
{
  const int fd = ::open (path.c_str (), O_RDONLY);
  if (fd < 0)
    {
      return std::unexpected (yantra::Error::registry (
          yantra::RegistryErrorCode::ArtifactNotFound, path.string (), 0U,
          "Cannot open file for SHA-256 verification"));
    }
  struct stat st
  {
  };
  if (::fstat (fd, &st) != 0 || st.st_size < 0)
    {
      (void)::close (fd);
      return std::unexpected (yantra::Error::registry (
          yantra::RegistryErrorCode::IoFailure, path.string (), 0U,
          "Cannot stat file for SHA-256 verification"));
    }
  const auto len = static_cast<std::size_t> (st.st_size);
  if (len == 0U)
    {
      (void)::close (fd);
      return yantra::compute_sha256_hex ({});
    }
  void *mapped = ::mmap (nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  (void)::close (fd);
  if (mapped == MAP_FAILED)
    {
      return std::unexpected (yantra::Error::registry (
          yantra::RegistryErrorCode::IoFailure, path.string (), 0U,
          "Cannot mmap file for SHA-256 verification"));
    }
  const std::span<const std::byte> bytes (
      static_cast<const std::byte *> (mapped), len);
  std::string digest = yantra::compute_sha256_hex (bytes);
  (void)::munmap (mapped, len);
  return digest;
}

auto
query_vram_used_mib () -> uint64_t
{
  FILE *pipe = ::popen (
      "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits "
      "2>/dev/null",
      "r");
  if (pipe == nullptr)
    {
      return 0U;
    }
  char buf[128]{};
  uint64_t val = 0U;
  if (std::fgets (buf, sizeof (buf), pipe) != nullptr)
    {
      val = std::strtoull (buf, nullptr, 10);
    }
  (void)::pclose (pipe);
  return val;
}

auto
query_gpu_name () -> std::string
{
  FILE *pipe = ::popen (
      "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
  if (pipe == nullptr)
    {
      return "Unknown GPU";
    }
  char buf[256]{};
  std::string out = "Unknown GPU";
  if (std::fgets (buf, sizeof (buf), pipe) != nullptr)
    {
      out = buf;
      while (!out.empty () && (out.back () == '\n' || out.back () == '\r'))
        {
          out.pop_back ();
        }
    }
  (void)::pclose (pipe);
  return out;
}

auto
query_cpu_name () -> std::string
{
  std::ifstream in ("/proc/cpuinfo");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.starts_with ("model name"))
        {
          const auto colon = line.find (':');
          if (colon != std::string::npos)
            {
              std::string name = line.substr (colon + 1U);
              while (!name.empty () && name.front () == ' ')
                {
                  name.erase (name.begin ());
                }
              return name;
            }
        }
    }
  return "Unknown CPU";
}

auto
query_kernel_version () -> std::string
{
  struct utsname uts
  {
  };
  if (::uname (&uts) == 0)
    {
      return std::string (uts.sysname) + " " + uts.release;
    }
  return "Linux";
}

auto
query_total_ram () -> std::string
{
  std::ifstream in ("/proc/meminfo");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.starts_with ("MemTotal:"))
        {
          std::istringstream iss (line.substr (9U));
          long kb = 0;
          iss >> kb;
          std::ostringstream oss;
          oss << std::fixed << std::setprecision (1)
              << (static_cast<double> (kb) / (1024.0 * 1024.0)) << " GiB";
          return oss.str ();
        }
    }
  return "Unknown";
}

auto
query_disk_free (const fs::path &p) -> std::string
{
  struct statvfs vfs
  {
  };
  if (::statvfs (p.c_str (), &vfs) == 0)
    {
      const double gib
          = (static_cast<double> (vfs.f_bavail)
             * static_cast<double> (vfs.f_frsize))
            / (1024.0 * 1024.0 * 1024.0);
      std::ostringstream oss;
      oss << std::fixed << std::setprecision (1) << gib << " GiB free";
      return oss.str ();
    }
  return "Unknown";
}

auto
json_escape (std::string_view s) -> std::string
{
  std::string out;
  out.reserve (s.size () + 8U);
  for (const char c : s)
    {
      switch (c)
        {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          out.push_back (c);
          break;
        }
    }
  return out;
}

auto
extract_json_string (std::string_view obj, std::string_view key) -> std::string
{
  const std::string needle = "\"" + std::string (key) + "\"";
  const auto kpos = obj.find (needle);
  if (kpos == std::string_view::npos)
    {
      return "";
    }
  const auto colon = obj.find (':', kpos + needle.size ());
  if (colon == std::string_view::npos)
    {
      return "";
    }
  const auto q1 = obj.find ('"', colon + 1U);
  if (q1 == std::string_view::npos)
    {
      return "";
    }
  std::string out;
  for (std::size_t i = q1 + 1U; i < obj.size (); ++i)
    {
      if (obj[i] == '\\' && i + 1U < obj.size ())
        {
          out.push_back (obj[++i]);
          continue;
        }
      if (obj[i] == '"')
        {
          break;
        }
      out.push_back (obj[i]);
    }
  return out;
}

auto
extract_json_int (std::string_view obj, std::string_view key) -> int
{
  const std::string needle = "\"" + std::string (key) + "\"";
  const auto kpos = obj.find (needle);
  if (kpos == std::string_view::npos)
    {
      return 0;
    }
  const auto colon = obj.find (':', kpos + needle.size ());
  if (colon == std::string_view::npos)
    {
      return 0;
    }
  std::size_t pos = colon + 1U;
  while (pos < obj.size () && (obj[pos] == ' ' || obj[pos] == '\t'))
    {
      ++pos;
    }
  return std::atoi (std::string (obj.substr (pos, 16U)).c_str ());
}

auto
load_prompts (const fs::path &path) -> std::vector<PromptItem>
{
  std::ifstream in (path);
  std::ostringstream ss;
  ss << in.rdbuf ();
  const std::string content = ss.str ();

  std::vector<PromptItem> items;
  std::size_t pos = 0U;
  while (pos < content.size ())
    {
      const auto open_b = content.find ('{', pos);
      if (open_b == std::string::npos)
        {
          break;
        }
      const auto close_b = content.find ('}', open_b);
      if (close_b == std::string::npos)
        {
          break;
        }
      const std::string_view slice (content.data () + open_b,
                                    close_b - open_b + 1U);
      PromptItem item;
      item.id = extract_json_int (slice, "id");
      item.text = extract_json_string (slice, "text");
      item.expects = extract_json_string (slice, "expects");
      item.tests = extract_json_string (slice, "tests");
      if (item.id > 0 && !item.text.empty ())
        {
          items.push_back (std::move (item));
        }
      pos = close_b + 1U;
    }
  return items;
}

auto
percentile (std::vector<double> vals, double p) -> double
{
  if (vals.empty ())
    {
      return 0.0;
    }
  std::sort (vals.begin (), vals.end ());
  const double idx = (p / 100.0) * static_cast<double> (vals.size () - 1U);
  const auto lo = static_cast<std::size_t> (std::floor (idx));
  const auto hi = static_cast<std::size_t> (std::ceil (idx));
  const double frac = idx - static_cast<double> (lo);
  return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

auto
validate_svg_xml (const fs::path &svg_path, std::string &err_out) -> bool
{
  std::error_code ec;
  if (!fs::exists (svg_path, ec) || fs::file_size (svg_path, ec) == 0U)
    {
      err_out = "SVG file does not exist or is empty";
      return false;
    }
  std::ifstream in (svg_path);
  std::ostringstream ss;
  ss << in.rdbuf ();
  const std::string xml = ss.str ();
  if (xml.find ("<svg") == std::string::npos
      || xml.find ("xmlns=\"http://www.w3.org/2000/svg\"") == std::string::npos
      || xml.find ("</svg>") == std::string::npos)
    {
      err_out = "Missing <svg> root element or SVG namespace declaration";
      return false;
    }
  int angle_depth = 0;
  bool in_quotes = false;
  for (const char ch : xml)
    {
      if (ch == '"')
        {
          in_quotes = !in_quotes;
        }
      else if (!in_quotes)
        {
          if (ch == '<')
            {
              ++angle_depth;
              if (angle_depth > 1)
                {
                  err_out = "Malformed XML: nested '<' inside tag";
                  return false;
                }
            }
          else if (ch == '>')
            {
              --angle_depth;
              if (angle_depth < 0)
                {
                  err_out = "Malformed XML: unmatched '>'";
                  return false;
                }
            }
        }
    }
  if (in_quotes || angle_depth != 0)
    {
      err_out = "Malformed XML: unclosed quote or tag";
      return false;
    }
  return true;
}

void
write_results_json (const fs::path &out_path, const fs::path &model_path,
                    const std::string &model_sha256, uint64_t model_size_bytes,
                    uint64_t registry_version,
                    const std::vector<TurnTelemetry> &turns,
                    const std::vector<CheckStatus> &checks,
                    bool darshana_available, int exit_code)
{
  std::vector<double> ttfts;
  std::vector<double> totals;
  std::vector<double> tps;
  for (const auto &t : turns)
    {
      ttfts.push_back (t.ttft_ms);
      totals.push_back (t.total_ms);
      tps.push_back (t.tokens_per_sec);
    }

  std::ofstream out (out_path, std::ios::trunc);
  out << std::fixed << std::setprecision (3);
  out << "{\n";
  out << "  \"hardware\": {\n";
  out << "    \"cpu\": \"" << json_escape (query_cpu_name ()) << "\",\n";
  out << "    \"gpu\": \"" << json_escape (query_gpu_name ()) << "\",\n";
  out << "    \"kernel\": \"" << json_escape (query_kernel_version ())
      << "\",\n";
  out << "    \"ram\": \"" << json_escape (query_total_ram ()) << "\",\n";
  out << "    \"disk\": \""
      << json_escape (query_disk_free (fs::current_path ())) << "\"\n";
  out << "  },\n";
  out << "  \"model\": {\n";
  out << "    \"path\": \"" << json_escape (model_path.string ()) << "\",\n";
  out << "    \"sha256\": \"" << json_escape (model_sha256) << "\",\n";
  out << "    \"size_bytes\": " << model_size_bytes << ",\n";
  out << "    \"registry_version\": " << registry_version << "\n";
  out << "  },\n";
  out << "  \"turns\": [\n";
  for (std::size_t i = 0U; i < turns.size (); ++i)
    {
      const auto &t = turns[i];
      out << "    {\n";
      out << "      \"id\": " << t.id << ",\n";
      out << "      \"category\": \"" << json_escape (t.category) << "\",\n";
      out << "      \"prompt\": \"" << json_escape (t.prompt) << "\",\n";
      out << "      \"response_length\": " << t.response_length << ",\n";
      out << "      \"ttft_ms\": " << t.ttft_ms << ",\n";
      out << "      \"total_ms\": " << t.total_ms << ",\n";
      out << "      \"tokens\": " << t.tokens << ",\n";
      out << "      \"tokens_per_sec\": " << t.tokens_per_sec << "\n";
      out << "    }" << (i + 1U < turns.size () ? "," : "") << "\n";
    }
  out << "  ],\n";
  out << "  \"aggregate\": {\n";
  out << "    \"ttft_ms\": { \"p50\": " << percentile (ttfts, 50.0)
      << ", \"p99\": " << percentile (ttfts, 99.0) << " },\n";
  out << "    \"total_ms\": { \"p50\": " << percentile (totals, 50.0)
      << ", \"p99\": " << percentile (totals, 99.0) << " },\n";
  out << "    \"tok_per_s\": { \"p50\": " << percentile (tps, 50.0)
      << ", \"p99\": " << percentile (tps, 99.0) << " }\n";
  out << "  },\n";
  out << "  \"composability\": {\n";
  for (std::size_t i = 0U; i < checks.size (); ++i)
    {
      const auto &c = checks[i];
      out << "    \"check_" << (i + 1U) << "\": { \"passed\": "
          << (c.passed ? "true" : "false") << ", \"details\": \""
          << json_escape (c.passed ? c.details : c.failure_message) << "\" }"
          << (i + 1U < checks.size () ? "," : "") << "\n";
    }
  out << "  },\n";
  out << "  \"darshana_available\": "
      << (darshana_available ? "true" : "false") << ",\n";
  out << "  \"exit_code\": " << exit_code << "\n";
  out << "}\n";
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  try
    {
      std::string model_arg;
      std::string registry_arg;
      bool has_custom_registry = false;

      for (int i = 1; i < argc; ++i)
        {
          const std::string_view arg (argv[i]);
          if (arg == "--model" && i + 1 < argc)
            {
              model_arg = argv[++i];
            }
          else if (arg.starts_with ("--model="))
            {
              model_arg = std::string (arg.substr (8U));
            }
          else if (arg == "--registry" && i + 1 < argc)
            {
              registry_arg = argv[++i];
              has_custom_registry = true;
            }
          else if (arg.starts_with ("--registry="))
            {
              registry_arg = std::string (arg.substr (11U));
              has_custom_registry = true;
            }
        }

      if (model_arg.empty ())
        {
          std::cout << "FAILED: Model file not found at <empty>. Run "
                       "fetch_model.sh first or supply --model.\n";
          return 2;
        }

      const fs::path model_path (model_arg);
      std::error_code ec;
      if (!fs::exists (model_path, ec) || !fs::is_regular_file (model_path, ec))
        {
          std::cout << "FAILED: Model file not found at "
                    << model_path.string ()
                    << ". Run fetch_model.sh first or supply --model.\n";
          return 2;
        }

      const uint64_t model_size_bytes = fs::file_size (model_path, ec);
      const double model_size_mb
          = static_cast<double> (model_size_bytes) / (1024.0 * 1024.0);

      fs::path prompts_path = "prompts.json";
      if (!fs::exists (prompts_path, ec))
        {
          const fs::path bin_dir = fs::path (argv[0]).parent_path ();
          if (fs::exists (bin_dir / "prompts.json", ec))
            {
              prompts_path = bin_dir / "prompts.json";
            }
          else if (fs::exists (bin_dir / ".." / "prompts.json", ec))
            {
              prompts_path = bin_dir / ".." / "prompts.json";
            }
        }
      const std::vector<PromptItem> prompts = load_prompts (prompts_path);
      if (prompts.size () != 10U)
        {
          std::cerr << "FAILED: Expected 10 prompts in "
                    << prompts_path.string () << ", found " << prompts.size ()
                    << "\n";
          return 1;
        }

      const fs::path registry_root
          = has_custom_registry
                ? fs::path (registry_arg)
                : (fs::current_path () / "yantra_stack_integration_store");
      if (!has_custom_registry)
        {
          fs::remove_all (registry_root, ec);
        }
      else
        {
          fs::remove_all (registry_root / "experiments", ec);
        }

      sutra::LlamaCppClient::init_backend ();
      struct QuietYantraSink final : public yantra::LogSink
      {
        void
        write (yantra::LogLevel /*level*/, std::string_view /*subsystem*/,
               std::string_view /*message*/,
               uint64_t /*timestamp_ns*/) noexcept override
        {
        }
      };
      yantra::Logger::set_sink (std::make_shared<QuietYantraSink> ());
#if defined(__linux__)
      (void)malloc_trim (0);
#endif
      const RssBreakdown pre_run_rss = read_rss_breakdown ();
      double peak_rss_mb = pre_run_rss.vm_rss_mb;

      std::vector<CheckStatus> checks (7U);
      checks[0].name = "Yantra -> Sutra artifact handoff";
      checks[1].name = "Sutra AgentStats -> Yantra tracker";
      checks[2].name = "Async thread interop";
      checks[3].name = "Yantra error -> Sutra error propagation";
      checks[4].name = "Darshana visualization";
      checks[5].name = "Model integrity round trip";
      checks[6].name = "Memory released after teardown";

      std::cout
          << "[stack-integration] Registering model in Yantra registry...\n";
      std::cout << "[stack-integration]   Source: " << model_path.string ()
                << "\n";

      auto source_sha_res = hash_file_sha256 (model_path);
      if (!source_sha_res)
        {
          std::cerr << "FAILED: Yantra SHA-256 computation failed: "
                    << source_sha_res.error ().message () << "\n";
          return 3;
        }
      const std::string source_sha256 = *source_sha_res;

      std::string served_sha256;
      uint64_t registry_version = 0U;
      std::vector<TurnTelemetry> turn_records;
      const fs::path wal_path = registry_root / "experiments" / "wal.bin";

      {
        auto reg_res = yantra::ModelRegistry::open (registry_root / "models");
        if (!reg_res)
          {
            std::cerr << "Yantra registry init failure: "
                      << reg_res.error ().message () << "\n";
            return 3;
          }
        yantra::ModelRegistry registry = std::move (*reg_res);

        auto store_res
            = yantra::ExperimentStore::open (registry_root / "experiments");
        if (!store_res)
          {
            std::cerr << "Yantra tracker store init failure: "
                      << store_res.error ().message () << "\n";
            return 3;
          }
        yantra::ExperimentStore store = std::move (*store_res);

        yantra::Result<yantra::ModelManifest> manifest_res
            = std::unexpected (yantra::Error::registry (
                yantra::RegistryErrorCode::ArtifactNotFound, "", 0U,
                "Not yet registered"));
        if (registry.total_versions () > 0U)
          {
            manifest_res
                = registry.lookup ("Qwen2.5-7B-Instruct-Q4_K_M:1.0.0");
          }
        if (!manifest_res
            || !fs::exists (manifest_res->artifact_path, ec))
          {
            yantra::ModelLineage lineage;
            lineage.run_id = "stack_integration_run";
            lineage.dataset_hash = source_sha256;
            lineage.code_commit_hash = "stack-integration-v1";
            manifest_res = registry.register_file (
                "Qwen2.5-7B-Instruct-Q4_K_M", "1.0.0", model_path, lineage,
                1U, true);
          }

        if (!manifest_res)
          {
            std::cerr << "Yantra model registration failed: "
                      << manifest_res.error ().message () << "\n";
            return 3;
          }

        registry_version = manifest_res->version;
        std::cout << "[stack-integration]   SHA-256: "
                  << manifest_res->sha256_hex << "\n";
        std::cout << "[stack-integration]   Registry version: "
                  << registry_version << "\n";

        peak_rss_mb = std::max (peak_rss_mb, read_rss_breakdown ().vm_rss_mb);

        std::cout << "[stack-integration] Loading via Sutra LlamaCppClient...\n";
        std::cout << "[stack-integration]   n_gpu_layers: 99\n";

        sutra::LlamaCppConfig lcfg;
        lcfg.n_ctx = 4096U;
        lcfg.n_gpu_layers = 99;
        lcfg.n_threads = 8;
        lcfg.default_max_tokens = 256U;

        auto client_res
            = sutra::LlamaCppClient::create_from_registry (manifest_res, lcfg);
        if (!client_res)
          {
            const auto &err = client_res.error ();
            std::cerr << "Sutra model load failure: [" << err.category_name ()
                      << "] " << err.message () << "\n";
            std::cerr << "  Model path: "
                      << manifest_res->artifact_path.string () << "\n";
            std::cerr << "  SHA-256: " << manifest_res->sha256_hex << "\n";
            std::cerr << "  VRAM used: " << query_vram_used_mib () << " MiB\n";
            checks[0].passed = false;
            checks[0].failure_message
                = "SEAM BROKEN: Yantra registry path/artifact is not "
                  "compatible with Sutra LlamaCppClient. Path: "
                  + manifest_res->artifact_path.string ()
                  + ". Error: " + std::string (err.message ()) + ".";
            write_results_json ("stack-integration-results.json", model_path,
                                source_sha256, model_size_bytes,
                                registry_version, turn_records, checks,
#if defined(HAVE_DARSHANA) && HAVE_DARSHANA
                                true,
#else
                                false,
#endif
                                4);
            return 4;
          }

        std::shared_ptr<sutra::LlmClient> llm (std::move (*client_res));
        const uint64_t vram_peak = query_vram_used_mib ();
        std::cout << "[stack-integration]   VRAM peak: " << vram_peak
                  << " MiB\n";

        peak_rss_mb = std::max (peak_rss_mb, read_rss_breakdown ().vm_rss_mb);

        auto rehash_res = hash_file_sha256 (manifest_res->artifact_path);
        if (!rehash_res)
          {
            checks[0].passed = false;
            checks[0].failure_message
                = "SEAM BROKEN: Yantra registry path/artifact is not "
                  "compatible with Sutra LlamaCppClient. Path: "
                  + manifest_res->artifact_path.string () + ". Error: "
                  + std::string (rehash_res.error ().message ()) + ".";
          }
        else
          {
            served_sha256 = *rehash_res;
            if (served_sha256 == manifest_res->sha256_hex
                && llm->capabilities ().vocab_size > 0U)
              {
                checks[0].passed = true;
                checks[0].details
                    = "Loaded CAS artifact "
                      + manifest_res->artifact_path.string ()
                      + " with matching SHA-256 " + served_sha256;
              }
            else
              {
                checks[0].passed = false;
                checks[0].failure_message
                    = "SEAM BROKEN: Yantra registry path/artifact is not "
                      "compatible with Sutra LlamaCppClient. Path: "
                      + manifest_res->artifact_path.string ()
                      + ". Error: SHA-256 or capabilities mismatch.";
              }
          }

        if (source_sha256 == served_sha256
            && manifest_res->sha256_hex == source_sha256)
          {
            checks[5].passed = true;
            checks[5].details
                = "Source and CAS-served SHA-256 match (" + source_sha256 + ")";
          }
        else
          {
            checks[5].passed = false;
            checks[5].failure_message
                = "SEAM BROKEN: Content-addressed artifact SHA-256 mismatch. "
                  "Source: "
                  + source_sha256 + ", Served: " + served_sha256 + ".";
          }

        auto run_res = store.create_run ("stack_integration_10_turns");
        if (!run_res)
          {
            std::cerr << "Failed to start Yantra run: "
                      << run_res.error ().message () << "\n";
            return 3;
          }
        yantra::Run yrun = std::move (*run_res);
        const std::string active_run_id (yrun.run_id ());
        (void)yrun.log_param ("model", "Qwen2.5-7B-Instruct-Q4_K_M");
        (void)yrun.log_param ("temperature", "0.0");

        auto tools = std::make_shared<sutra::ToolRegistry> ();
        sutra::AgentConfig acfg;
        acfg.max_steps = 4U;
        acfg.short_term_capacity = 64U;
        auto agent_res = sutra::Agent::create (acfg, tools, llm);
        if (!agent_res)
          {
            std::cerr << "Failed to create Sutra Agent: "
                      << agent_res.error ().message () << "\n";
            return 4;
          }
        auto agent = std::move (*agent_res);
        sutra::AgentRuntime runtime (4U);

        std::cout << "[stack-integration] Running 10 turns...\n";

        const std::thread::id main_tid = std::this_thread::get_id ();
        bool metric_log_error = false;
        std::string metric_log_error_key;

        for (std::size_t idx = 0U; idx < prompts.size (); ++idx)
          {
            const PromptItem &p = prompts[idx];
            std::promise<sutra::Result<TurnTelemetry>> turn_promise;
            auto turn_future = turn_promise.get_future ();

            runtime.pool ().submit ([&runtime, &agent, &yrun, &p, main_tid,
                                     &metric_log_error,
                                     &metric_log_error_key,
                                     &turn_promise] () noexcept {
              auto handle_res = runtime.run_async (*agent, p.text);
              if (!handle_res)
                {
                  turn_promise.set_value (
                      std::unexpected (handle_res.error ()));
                  return;
                }
              auto run_out = handle_res->wait ();
              if (!run_out)
                {
                  turn_promise.set_value (std::unexpected (run_out.error ()));
                  return;
                }

              const auto snap = sutra::AgentStats::global ().snapshot ();
              TurnTelemetry tel;
              tel.id = p.id;
              tel.category = p.tests;
              tel.prompt = p.text;
              tel.response_length = run_out->final_answer.size ();
              tel.ttft_ms = static_cast<double> (snap.last_ttft_ns) / 1.0e6;
              tel.total_ms
                  = static_cast<double> (snap.last_generation_ns) / 1.0e6;
              tel.tokens = snap.last_tokens_generated;
              tel.tokens_per_sec = snap.last_tokens_per_sec;
              tel.logged_on_async_worker
                  = (std::this_thread::get_id () != main_tid);

              const auto step_u64 = static_cast<uint64_t> (p.id);
              if (!yrun.log_metric (step_u64, "ttft_ms", tel.ttft_ms))
                {
                  metric_log_error = true;
                  metric_log_error_key = "ttft_ms";
                }
              if (!yrun.log_metric (step_u64, "total_ms", tel.total_ms))
                {
                  metric_log_error = true;
                  metric_log_error_key = "total_ms";
                }
              if (!yrun.log_metric (step_u64, "tokens_generated",
                                    static_cast<double> (tel.tokens)))
                {
                  metric_log_error = true;
                  metric_log_error_key = "tokens_generated";
                }
              if (!yrun.log_metric (step_u64, "tokens_per_sec",
                                    tel.tokens_per_sec))
                {
                  metric_log_error = true;
                  metric_log_error_key = "tokens_per_sec";
                }

              turn_promise.set_value (std::move (tel));
            });

            auto tel_res = turn_future.get ();
            if (!tel_res)
              {
                std::cerr << "Turn " << p.id
                          << " failed: " << tel_res.error ().message () << "\n";
                return 1;
              }

            const TurnTelemetry &tel = *tel_res;
            turn_records.push_back (tel);
            peak_rss_mb
                = std::max (peak_rss_mb, read_rss_breakdown ().vm_rss_mb);

            std::cout << "[turn " << std::setw (2) << tel.id << "/10] "
                      << std::left << std::setw (20) << tel.category
                      << std::right << " | TTFT " << std::fixed
                      << std::setprecision (1) << std::setw (5) << tel.ttft_ms
                      << " ms | " << std::setw (3) << tel.tokens
                      << " tokens | " << std::setw (5) << tel.tokens_per_sec
                      << " tok/s\n";
          }

        (void)yrun.finish ();

        // CHECK-2: Sutra AgentStats -> Yantra tracker metric compatibility
        auto queried_run_res = store.get_run (active_run_id);
        if (metric_log_error || !queried_run_res)
          {
            checks[1].passed = false;
            checks[1].failure_message
                = "SEAM BROKEN: Sutra AgentStats metric "
                  + (metric_log_error_key.empty () ? std::string ("<run>")
                                                   : metric_log_error_key)
                  + " does not round-trip through Yantra tracker. Sutra value: "
                    "0, Yantra value: missing.";
          }
        else
          {
            bool check2_ok = true;
            std::string failed_key;
            double failed_sutra_val = 0.0;
            double failed_yantra_val = 0.0;

            const auto &hist = queried_run_res->metric_history;
            const auto verify_series
                = [&] (const std::string &key, auto getter) {
                    if (!check2_ok)
                      {
                        return;
                      }
                    const auto it = hist.find (key);
                    if (it == hist.end ()
                        || it->second.size () != turn_records.size ())
                      {
                        check2_ok = false;
                        failed_key = key;
                        failed_sutra_val = getter (turn_records.back ());
                        failed_yantra_val = -1.0;
                        return;
                      }
                    for (std::size_t i = 0U; i < turn_records.size (); ++i)
                      {
                        const double sv = getter (turn_records[i]);
                        const double yv = it->second[i].value;
                        if (std::abs (yv - sv) >= 1.0e-6)
                          {
                            check2_ok = false;
                            failed_key = key;
                            failed_sutra_val = sv;
                            failed_yantra_val = yv;
                            return;
                          }
                      }
                  };

            verify_series ("ttft_ms",
                           [] (const TurnTelemetry &t) { return t.ttft_ms; });
            verify_series ("total_ms",
                           [] (const TurnTelemetry &t) { return t.total_ms; });
            verify_series ("tokens_generated", [] (const TurnTelemetry &t) {
              return static_cast<double> (t.tokens);
            });
            verify_series ("tokens_per_sec", [] (const TurnTelemetry &t) {
              return t.tokens_per_sec;
            });

            if (check2_ok)
              {
                checks[1].passed = true;
                checks[1].details
                    = "All 40 per-turn metrics (ttft_ms, total_ms, "
                      "tokens_generated, tokens_per_sec) matched within 1e-6";
              }
            else
              {
                checks[1].passed = false;
                std::ostringstream oss;
                oss << std::fixed << std::setprecision (6)
                    << "SEAM BROKEN: Sutra AgentStats metric " << failed_key
                    << " does not round-trip through Yantra tracker. Sutra "
                       "value: "
                    << failed_sutra_val
                    << ", Yantra value: " << failed_yantra_val << ".";
                checks[1].failure_message = oss.str ();
              }
          }

        // CHECK-3: Async thread interop (concurrency)
        {
          constexpr std::size_t kExpectedWalEntries = 44U;
          auto recovered_res = yantra::WalEngine::recover (wal_path);
          const std::size_t valid_wal_entries
              = recovered_res ? recovered_res->size () : 0U;

          bool all_async = true;
          for (const auto &t : turn_records)
            {
              if (!t.logged_on_async_worker)
                {
                  all_async = false;
                }
            }

          if (valid_wal_entries == kExpectedWalEntries && all_async)
            {
              checks[2].passed = true;
              checks[2].details
                  = "WAL contains exact expected 44 CRC-verified entries "
                    "written from Sutra async execution thread";
            }
          else
            {
              checks[2].passed = false;
              checks[2].failure_message
                  = "SEAM BROKEN: Yantra tracker WAL has "
                    + std::to_string (valid_wal_entries)
                    + " entries, expected "
                    + std::to_string (kExpectedWalEntries)
                    + ". Concurrency or durability issue during async writes.";
            }
        }

        // CHECK-4: Yantra error -> Sutra error propagation
        {
          const yantra::ModelLineage empty_lin{};
          const auto bad_reg_res = registry.register_file (
              "nonexistent_model", "1.0.0",
              "/nonexistent/path/to/missing_model.gguf", empty_lin);
          const auto bad_sutra_res
              = sutra::LlamaCppClient::create_from_registry (bad_reg_res, lcfg);

          const auto *reg_err
              = bad_reg_res.has_value ()
                    ? nullptr
                    : std::get_if<yantra::RegistryError> (
                          &bad_reg_res.error ().variant ());

          if (!bad_reg_res.has_value () && reg_err != nullptr
              && !bad_sutra_res.has_value ()
              && sutra::is_yantra_registry_error (bad_sutra_res.error ())
              && bad_sutra_res.error ().yantra_error_code ()
                     == static_cast<uint8_t> (reg_err->code))
            {
              checks[3].passed = true;
              checks[3].details
                  = "Yantra RegistryErrorCode::ArtifactNotFound ("
                    + std::to_string (
                          bad_sutra_res.error ().yantra_error_code ())
                    + ") propagated cleanly through Sutra AgentError";
            }
          else
            {
              checks[3].passed = false;
              const std::string y_err_str
                  = bad_reg_res.has_value ()
                        ? "none"
                        : std::string (bad_reg_res.error ().message ());
              const std::string s_err_str
                  = bad_sutra_res.has_value ()
                        ? "none"
                        : std::string (bad_sutra_res.error ().message ());
              checks[3].failure_message
                  = "SEAM BROKEN: Yantra RegistryError does not propagate "
                    "through Sutra AgentError. Original: "
                    + y_err_str + ". Sutra: " + s_err_str + ".";
            }
        }

        // CHECK-5: Darshana visualization
#if defined(HAVE_DARSHANA) && HAVE_DARSHANA
        {
          darshana::Spec spec;
          spec.title = "Sutra + Yantra Per-Turn Latency (ms)";
          spec.subtitle = "Qwen2.5-7B-Instruct-Q4_K_M (10 turns)";
          spec.x_label = "Turn";
          spec.y_label = "Latency (ms)";
          spec.width = 900U;
          spec.height = 540U;
          spec.x_scale.domain_min = 1.0;
          spec.x_scale.domain_max = 10.0;
          spec.x_scale.auto_domain = false;

          double max_lat = 100.0;
          darshana::LayerSpec ttft_layer;
          ttft_layer.mark = darshana::MarkType::Line;
          ttft_layer.encoding.label = "TTFT (ms)";
          ttft_layer.aesthetics.stroke_color
              = darshana::ColorRgba::from_hex (0x1F77B4U);

          darshana::LayerSpec total_layer;
          total_layer.mark = darshana::MarkType::Line;
          total_layer.encoding.label = "Total Generation Latency (ms)";
          total_layer.aesthetics.stroke_color
              = darshana::ColorRgba::from_hex (0xD62728U);

          if (queried_run_res)
            {
              const auto &mh = queried_run_res->metric_history;
              if (const auto it = mh.find ("ttft_ms"); it != mh.end ())
                {
                  for (const auto &pt : it->second)
                    {
                      ttft_layer.inline_x.push_back (
                          static_cast<float> (pt.step));
                      ttft_layer.inline_y.push_back (
                          static_cast<float> (pt.value));
                      max_lat = std::max (max_lat, pt.value);
                    }
                }
              if (const auto it = mh.find ("total_ms"); it != mh.end ())
                {
                  for (const auto &pt : it->second)
                    {
                      total_layer.inline_x.push_back (
                          static_cast<float> (pt.step));
                      total_layer.inline_y.push_back (
                          static_cast<float> (pt.value));
                      max_lat = std::max (max_lat, pt.value);
                    }
                }
            }

          spec.y_scale.domain_min = 0.0;
          spec.y_scale.domain_max = max_lat * 1.1;
          spec.y_scale.auto_domain = false;
          spec.layers.push_back (std::move (ttft_layer));
          spec.layers.push_back (std::move (total_layer));

          const fs::path svg_path = "results-latency.svg";
          auto exp_res = darshana::export_file_sync (
              spec, svg_path.string (), darshana::ExportFormat::Svg);
          std::string xml_err;
          if (!exp_res)
            {
              checks[4].passed = false;
              checks[4].failure_message
                  = "SEAM BROKEN: Darshana cannot render Yantra tracker "
                    "results. Error: "
                    + std::string (exp_res.error ().message ()) + ".";
            }
          else if (!validate_svg_xml (svg_path, xml_err))
            {
              checks[4].passed = false;
              checks[4].failure_message
                  = "SEAM BROKEN: Darshana cannot render Yantra tracker "
                    "results. Error: "
                    + xml_err + ".";
            }
          else
            {
              checks[4].passed = true;
              checks[4].details
                  = "Rendered results-latency.svg ("
                    + std::to_string (exp_res->bytes_written) + " bytes)";
            }
        }
#else
        std::cerr << "WARNING: Darshana not found; skipping CHECK-5.\n";
        checks[4].passed = true;
        checks[4].details = "Skipped (Darshana not installed)";
#endif
      } // Full stack teardown: Sutra Agent, LlamaCppClient, AgentRuntime,
        // Yantra Run, ExperimentStore, and ModelRegistry are all destroyed.

#if defined(__linux__)
      (void)malloc_trim (0);
#endif

      const RssBreakdown post_teardown_rss = read_rss_breakdown ();

      // CHECK-7: Memory does not leak across the full stack
      {
        const double baseline_mb = pre_run_rss.vm_rss_mb;
        const double post_mb = post_teardown_rss.vm_rss_mb;
        const double delta_mb = post_mb - baseline_mb;
        const bool peak_ok = (peak_rss_mb <= model_size_mb + 500.0);
        const bool teardown_ok = (post_mb <= baseline_mb * 1.10);

        if (peak_ok && teardown_ok)
          {
            checks[6].passed = true;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision (1)
                << "Baseline=" << baseline_mb << " MB, Peak=" << peak_rss_mb
                << " MB, Post-teardown=" << post_mb << " MB";
            checks[6].details = oss.str ();
          }
        else
          {
            checks[6].passed = false;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision (1)
                << "SEAM BROKEN: Memory not released after full stack "
                   "teardown. Baseline: "
                << baseline_mb << " MB, Post-teardown: " << post_mb
                << " MB, Delta: " << delta_mb << " MB.";
            checks[6].failure_message = oss.str ();
          }
      }

      std::cout << "[stack-integration] Composability checks:\n";
      const char *dots[7] = {
        " ............ ",
        " .......... ",
        " ........................ ",
        " ..... ",
        " ...................... ",
        " .................. ",
        " .............. ",
      };

      int failed_check_idx = -1;
      for (std::size_t i = 0U; i < checks.size (); ++i)
        {
          std::cout << "[check " << (i + 1U) << "] " << checks[i].name << dots[i]
                    << (checks[i].passed ? "PASS" : "FAIL") << "\n";
          if (!checks[i].passed && failed_check_idx < 0)
            {
              failed_check_idx = static_cast<int> (i);
            }
        }

      const int exit_code = (failed_check_idx < 0) ? 0 : 1;
      write_results_json ("stack-integration-results.json", model_path,
                          source_sha256, model_size_bytes, registry_version,
                          turn_records, checks,
#if defined(HAVE_DARSHANA) && HAVE_DARSHANA
                          true,
#else
                          false,
#endif
                          exit_code);

      if (!has_custom_registry)
        {
          fs::remove_all (registry_root, ec);
        }

      if (failed_check_idx >= 0)
        {
          std::cout << "[stack-integration] FAILED at check "
                    << (failed_check_idx + 1) << ": "
                    << checks[static_cast<std::size_t> (failed_check_idx)]
                           .failure_message
                    << "\n";
          std::cout << "[stack-integration] Exit: " << exit_code << "\n";
          return exit_code;
        }

      std::cout << "[stack-integration] All checks passed. Results: "
                   "stack-integration-results.json\n";
      std::cout << "[stack-integration] Exit: 0\n";
      return 0;
    }
  catch (const std::exception &ex)
    {
      std::cerr << "[stack-integration] Unhandled exception: " << ex.what ()
                << "\n";
      return 1;
    }
}
