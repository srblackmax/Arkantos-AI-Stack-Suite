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

#include <yantra/registry.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace yantra::inline v1
{

namespace
{

class ScopedFd
{
public:
  explicit ScopedFd (int fd) noexcept : fd_ (fd) {}
  ScopedFd (const ScopedFd &) = delete;
  auto operator= (const ScopedFd &) -> ScopedFd & = delete;

  ~ScopedFd () noexcept
  {
    if (fd_ >= 0)
      {
        (void)::close (fd_);
      }
  }

  [[nodiscard]] auto
  get () const noexcept -> int
  {
    return fd_;
  }

private:
  int fd_ = -1;
};

class ScopedMmap
{
public:
  ScopedMmap (void *addr, std::size_t len) noexcept
      : addr_ (addr), len_ (len)
  {
  }
  ScopedMmap (const ScopedMmap &) = delete;
  auto operator= (const ScopedMmap &) -> ScopedMmap & = delete;

  ~ScopedMmap () noexcept
  {
    if (addr_ != nullptr && addr_ != MAP_FAILED && len_ > 0U)
      {
        (void)::munmap (addr_, len_);
      }
  }

  [[nodiscard]] auto
  data () const noexcept -> const std::byte *
  {
    return static_cast<const std::byte *> (addr_);
  }

private:
  void *addr_ = MAP_FAILED;
  std::size_t len_ = 0U;
};

auto
write_artifact_parallel (const std::filesystem::path &target_path,
                         std::span<const std::byte> data) noexcept -> bool
{
  const auto tmp_path = target_path.string () + ".tmp."
                        + std::to_string (monotonic_time_ns ());
  const int fd
      = ::open (tmp_path.c_str (), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    {
      return false;
    }

  bool ok = true;
  const std::size_t total_size = data.size ();
  if (total_size > 0U)
    {
      if (::ftruncate (fd, static_cast<off_t> (total_size)) != 0)
        {
          (void)::close (fd);
          (void)::unlink (tmp_path.c_str ());
          return false;
        }

      constexpr std::size_t kChunk = 16ULL * 1024ULL * 1024ULL;
      if (total_size <= kChunk)
        {
          const auto *ptr = reinterpret_cast<const uint8_t *> (data.data ());
          std::size_t written = 0U;
          while (written < total_size)
            {
              const ssize_t n = ::pwrite (fd, ptr + written,
                                          total_size - written,
                                          static_cast<off_t> (written));
              if (n <= 0)
                {
                  ok = false;
                  break;
                }
              written += static_cast<std::size_t> (n);
            }
        }
      else
        {
          const std::size_t num_chunks
              = (total_size + kChunk - 1U) / kChunk;
          const std::size_t num_workers = std::min<std::size_t> (
              8U, std::max<std::size_t> (1U, num_chunks));
          std::atomic<std::size_t> next_c{ 0U };
          std::atomic<bool> all_ok{ true };
          std::vector<std::jthread> workers;
          workers.reserve (num_workers);
          for (std::size_t w = 0U; w < num_workers; ++w)
            {
              workers.emplace_back ([&] () {
                const auto *base
                    = reinterpret_cast<const uint8_t *> (data.data ());
                for (;;)
                  {
                    const std::size_t c
                        = next_c.fetch_add (1U, std::memory_order_relaxed);
                    if (c >= num_chunks || !all_ok.load ())
                      {
                        break;
                      }
                    const std::size_t off = c * kChunk;
                    const std::size_t len
                        = std::min (kChunk, total_size - off);
                    std::size_t done = 0U;
                    while (done < len)
                      {
                        const ssize_t n = ::pwrite (
                            fd, base + off + done, len - done,
                            static_cast<off_t> (off + done));
                        if (n <= 0)
                          {
                            all_ok.store (false);
                            break;
                          }
                        done += static_cast<std::size_t> (n);
                      }
                  }
              });
            }
          workers.clear ();
          ok = all_ok.load ();
        }
    }

  (void)::close (fd);
  if (!ok)
    {
      (void)::unlink (tmp_path.c_str ());
      return false;
    }
  if (::rename (tmp_path.c_str (), target_path.c_str ()) != 0)
    {
      (void)::unlink (tmp_path.c_str ());
      return false;
    }
  return true;
}

auto
parse_tensor_count_header (std::span<const std::byte> data,
                           uint32_t fallback_count) noexcept -> uint32_t
{
  if (data.size () >= 8U)
    {
      const auto *bytes = reinterpret_cast<const uint8_t *> (data.data ());
      if (bytes[0] == 'Y' && bytes[1] == 'M' && bytes[2] == 'C'
          && bytes[3] == 'P')
        {
          uint32_t count = 0U;
          std::memcpy (&count, bytes + 4U, sizeof (uint32_t));
          if (count > 0U)
            {
              return count;
            }
        }
      if (data.size () >= 16U && bytes[0] == 'G' && bytes[1] == 'G'
          && bytes[2] == 'U' && bytes[3] == 'F' && fallback_count <= 1U)
        {
          uint64_t gguf_tensors = 0U;
          std::memcpy (&gguf_tensors, bytes + 8U, sizeof (uint64_t));
          if (gguf_tensors > 0U && gguf_tensors <= 0xFFFFFFFFULL)
            {
              return static_cast<uint32_t> (gguf_tensors);
            }
        }
    }
  return fallback_count > 0U ? fallback_count : 1U;
}

[[nodiscard]] auto
canonical_model_alias (std::string_view model_name) noexcept -> std::string
{
  if (model_name == "qwen2.5-1.5b-instruct-q4km")
    {
      return "qwen2.5-1.5b-q4km";
    }
  if (model_name == "qwen3-embedding-0.6b-q8_0")
    {
      return "qwen3-embedding-0.6b-q8";
    }
  if (model_name == "gemma-4-31B-it-Q4_K_M"
      || model_name == "gemma-4-31B-it-Q4_K_M.gguf")
    {
      return "gemma-4-31b-it-q4km";
    }
  if (model_name == "qwen3-coder-next" || model_name == "qwen3-80b"
      || model_name == "qwen3-coder-next-q4_k_s"
      || model_name == "Qwen3-Coder-Next-Q4_K_S"
      || model_name == "Qwen3-Coder-Next-Q4_K_S.gguf")
    {
      return "qwen3-coder-next-80b-q4ks";
    }
  return std::string (model_name);
}

constexpr std::size_t kUltraLargeFileThresholdBytes = 30ULL << 30U;
constexpr std::size_t kFastDigestPrefixBytes = 4ULL << 20U;

[[nodiscard]] auto
build_fast_large_artifact_sample (
    std::span<const std::byte> artifact_bytes) noexcept
    -> std::vector<std::byte>
{
  const std::size_t prefix_len
      = std::min (artifact_bytes.size (), kFastDigestPrefixBytes);
  std::vector<std::byte> sample (prefix_len + sizeof (uint64_t));
  if (prefix_len > 0U)
    {
      std::memcpy (sample.data (), artifact_bytes.data (), prefix_len);
    }
  const uint64_t total_len = static_cast<uint64_t> (artifact_bytes.size ());
  std::memcpy (sample.data () + prefix_len, &total_len, sizeof (total_len));
  return sample;
}

} /* namespace */

struct ModelRegistry::Impl
{
  std::filesystem::path registry_dir;
  mutable std::shared_mutex mu;
  std::unordered_map<std::string, std::vector<ModelManifest>> model_versions;
  std::unordered_map<std::string, ModelManifest> tag_index;
  std::unordered_map<std::string, std::vector<StageTransitionAudit>> audits;
  std::size_t total_version_count = 0U;

  [[nodiscard]] auto
  has_lineage_cycle_locked (std::string_view self_tag,
                            const ModelLineage &lineage) const noexcept -> bool
  {
    std::unordered_set<std::string> visited;
    std::vector<std::string> stack = lineage.parent_model_versions;
    while (!stack.empty ())
      {
        std::string curr = std::move (stack.back ());
        stack.pop_back ();
        if (curr == self_tag)
          {
            return true;
          }
        if (!visited.insert (curr).second)
          {
            continue;
          }
        const auto it = tag_index.find (curr);
        if (it != tag_index.end ())
          {
            for (const auto &p : it->second.lineage.parent_model_versions)
              {
                stack.push_back (p);
              }
          }
      }
    return false;
  }

  void
  save_catalog_locked () const noexcept
  {
    const auto cat_path = registry_dir / "catalog.txt";
    std::ofstream out (cat_path, std::ios::trunc);
    if (!out.is_open ())
      {
        return;
      }
    out << "CATALOG " << total_version_count << "\n";
    for (const auto &[mname, vers] : model_versions)
      {
        (void)mname;
        for (const auto &m : vers)
          {
            out << m.model_name << "\t" << m.version << "\t" << m.semver_tag
                << "\t" << static_cast<int> (m.stage) << "\t" << m.blake3_hex
                << "\t" << m.sha256_hex << "\t" << m.artifact_size_bytes
                << "\t" << m.tensor_count << "\t" << m.registered_at_ns << "\t"
                << m.artifact_path.string () << "\t" << m.lineage.run_id
                << "\t" << m.lineage.dataset_hash << "\t"
                << m.lineage.code_commit_hash << "\n";
          }
      }
  }

  void
  load_catalog_locked () noexcept
  {
    const auto cat_path = registry_dir / "catalog.txt";
    std::error_code ec;
    if (!std::filesystem::exists (cat_path, ec))
      {
        return;
      }
    std::ifstream in (cat_path);
    std::string header;
    std::size_t count = 0U;
    if (!(in >> header >> count) || header != "CATALOG")
      {
        return;
      }
    in.ignore (1, '\n');
    for (std::size_t i = 0U; i < count; ++i)
      {
        std::string line;
        if (!std::getline (in, line) || line.empty ())
          {
            break;
          }
        std::vector<std::string> fields;
        std::size_t start = 0U;
        for (;;)
          {
            const auto tab = line.find ('\t', start);
            if (tab == std::string::npos)
              {
                fields.push_back (line.substr (start));
                break;
              }
            fields.push_back (line.substr (start, tab - start));
            start = tab + 1U;
          }
        if (fields.size () < 13U)
          {
            continue;
          }
        ModelManifest m;
        m.model_name = fields[0];
        m.version = std::stoull (fields[1]);
        m.semver_tag = fields[2];
        m.stage = static_cast<ModelStage> (std::stoi (fields[3]));
        m.blake3_hex = fields[4];
        m.sha256_hex = fields[5];
        m.artifact_size_bytes = std::stoull (fields[6]);
        m.tensor_count = static_cast<uint32_t> (std::stoul (fields[7]));
        m.registered_at_ns = std::stoull (fields[8]);
        m.artifact_path = fields[9];
        m.lineage.run_id = fields[10];
        m.lineage.dataset_hash = fields[11];
        m.lineage.code_commit_hash = fields[12];

        auto &vers = model_versions[m.model_name];
        if (vers.size () < m.version)
          {
            vers.resize (m.version);
          }
        vers[m.version - 1U] = m;
        tag_index[m.model_name + ":" + m.semver_tag] = m;
        tag_index[m.model_name + ":v" + std::to_string (m.version)] = m;
        ++total_version_count;
      }
  }
};

ModelRegistry::ModelRegistry () noexcept = default;
ModelRegistry::ModelRegistry (ModelRegistry &&other) noexcept = default;
auto ModelRegistry::operator= (ModelRegistry &&other) noexcept
    -> ModelRegistry & = default;
ModelRegistry::~ModelRegistry () noexcept = default;

auto
ModelRegistry::open (const std::filesystem::path &registry_dir) noexcept
    -> Result<ModelRegistry>
{
  std::error_code ec;
  std::filesystem::create_directories (registry_dir / "cas", ec);
  if (ec)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::IoFailure, "", 0U,
          "Failed to initialize ModelRegistry directory at "
              + registry_dir.string ()));
    }
  ModelRegistry reg;
  reg.impl_ = std::make_unique<Impl> ();
  reg.impl_->registry_dir = registry_dir;
  reg.impl_->load_catalog_locked ();
  return reg;
}

auto
ModelRegistry::register_version (std::string_view model_name,
                                 std::string_view semver_tag,
                                 std::span<const std::byte> artifact_bytes,
                                 const ModelLineage &lineage,
                                 uint32_t tensor_count,
                                 bool compute_sha256) noexcept
    -> Result<ModelManifest>
{
  if (!impl_ || model_name.empty () || semver_tag.empty ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::InvalidStageTransition, std::string (model_name),
          0U, "Model name and semantic version tag must be non-empty"));
    }

  const uint64_t t0 = monotonic_time_ns ();
  const bool is_ultra_large
      = artifact_bytes.size () > kUltraLargeFileThresholdBytes;
  const std::vector<std::byte> fast_sample
      = is_ultra_large ? build_fast_large_artifact_sample (artifact_bytes)
                       : std::vector<std::byte>{};
  const std::span<const std::byte> hash_input
      = is_ultra_large ? std::span<const std::byte> (fast_sample)
                       : artifact_bytes;
  const std::string blake3_hex = compute_blake3_hex (hash_input);
  const std::string sha256_hex
      = compute_sha256 ? compute_sha256_hex (hash_input) : blake3_hex;

  const auto shard_dir
      = impl_->registry_dir / "cas" / blake3_hex.substr (0U, 2U);
  std::error_code ec;
  std::filesystem::create_directories (shard_dir, ec);
  const auto art_path = shard_dir / blake3_hex;

  if (!std::filesystem::exists (art_path, ec))
    {
      if (!write_artifact_parallel (art_path, artifact_bytes))
        {
          return std::unexpected (Error::registry (
              RegistryErrorCode::IoFailure, std::string (model_name), 0U,
              "Transactional artifact write to content-addressable store "
              "failed"));
        }
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const std::string mname (model_name);
  const std::string composite_tag = mname + ":" + std::string (semver_tag);

  if (impl_->has_lineage_cycle_locked (composite_tag, lineage))
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::LineageCycleDetected, mname, 0U,
          "Lineage graph cycle detected while registering " + composite_tag));
    }

  auto &versions = impl_->model_versions[mname];
  const uint64_t next_ver = versions.size () + 1U;
  const std::string ver_tag = mname + ":v" + std::to_string (next_ver);
  if (impl_->has_lineage_cycle_locked (ver_tag, lineage))
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::LineageCycleDetected, mname, next_ver,
          "Lineage graph cycle detected on version " + ver_tag));
    }

  ModelManifest manifest;
  manifest.model_name = mname;
  manifest.version = next_ver;
  manifest.semver_tag = std::string (semver_tag);
  manifest.stage = ModelStage::Staging;
  manifest.blake3_hex = blake3_hex;
  manifest.sha256_hex = sha256_hex;
  manifest.artifact_size_bytes = artifact_bytes.size ();
  manifest.tensor_count
      = parse_tensor_count_header (artifact_bytes, tensor_count);
  manifest.lineage = lineage;
  manifest.registered_at_ns = monotonic_time_ns ();
  manifest.artifact_path = art_path;

  versions.push_back (manifest);
  impl_->tag_index[composite_tag] = manifest;
  impl_->tag_index[ver_tag] = manifest;
  ++impl_->total_version_count;
  impl_->save_catalog_locked ();

  const uint64_t elapsed = monotonic_time_ns () - t0;
  RegistryStats::global ().record_registration (elapsed);
  return manifest;
}

auto
ModelRegistry::register_file (std::string_view model_name,
                              std::string_view semver_tag,
                              const std::filesystem::path &file_path,
                              const ModelLineage &lineage,
                              uint32_t tensor_count,
                              bool compute_sha256) noexcept
    -> Result<ModelManifest>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name), 0U,
          "ModelRegistry is not initialized"));
    }
  const int raw_fd = ::open (file_path.c_str (), O_RDONLY);
  if (raw_fd < 0)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name), 0U,
          "Model artifact file not found at " + file_path.string ()));
    }
  ScopedFd fd_guard (raw_fd);

  struct stat st
  {
  };
  if (::fstat (fd_guard.get (), &st) != 0 || st.st_size < 0)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::IoFailure, std::string (model_name), 0U,
          "Failed to stat model artifact file at " + file_path.string ()));
    }

  const auto file_len = static_cast<std::size_t> (st.st_size);
  if (file_len == 0U)
    {
      return register_version (model_name, semver_tag, {}, lineage,
                               tensor_count, compute_sha256);
    }

  void *mapped = ::mmap (nullptr, file_len, PROT_READ, MAP_PRIVATE,
                         fd_guard.get (), 0);
  if (mapped == MAP_FAILED)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::IoFailure, std::string (model_name), 0U,
          "Failed to mmap model artifact file at " + file_path.string ()));
    }
  ScopedMmap mmap_guard (mapped, file_len);

  const std::span<const std::byte> span_bytes (mmap_guard.data (), file_len);
  if (file_len >= (64ULL << 20U))
    {
      const bool is_ultra_large = file_len > kUltraLargeFileThresholdBytes;
      const std::vector<std::byte> fast_sample
          = is_ultra_large ? build_fast_large_artifact_sample (span_bytes)
                           : std::vector<std::byte>{};
      const std::string b3 = compute_blake3_hex (
          is_ultra_large ? std::span<const std::byte> (fast_sample)
                         : span_bytes);
      const auto shard_dir = impl_->registry_dir / "cas" / b3.substr (0U, 2U);
      std::error_code ec;
      std::filesystem::create_directories (shard_dir, ec);
      const auto art_path = shard_dir / b3;
      if (!std::filesystem::exists (art_path, ec))
        {
          std::filesystem::create_hard_link (file_path, art_path, ec);
          if (ec)
            {
              ec.clear ();
              std::filesystem::create_symlink (file_path, art_path, ec);
            }
        }
    }
  return register_version (model_name, semver_tag, span_bytes, lineage,
                           tensor_count, compute_sha256);
}

auto
ModelRegistry::register_version_cas (
    std::string_view model_name, uint64_t expected_prev_version,
    std::string_view semver_tag, std::span<const std::byte> artifact_bytes,
    const ModelLineage &lineage) noexcept -> Result<ModelManifest>
{
  if (!impl_ || model_name.empty () || semver_tag.empty ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::VersionConflict, std::string (model_name), 0U,
          "Model name and semantic version tag must be non-empty"));
    }

  const uint64_t t0 = monotonic_time_ns ();
  const std::string blake3_hex = compute_blake3_hex (artifact_bytes);
  const std::string sha256_hex = compute_sha256_hex (artifact_bytes);

  const auto shard_dir
      = impl_->registry_dir / "cas" / blake3_hex.substr (0U, 2U);
  std::error_code ec;
  std::filesystem::create_directories (shard_dir, ec);
  const auto art_path = shard_dir / blake3_hex;

  if (!std::filesystem::exists (art_path, ec))
    {
      if (!write_artifact_parallel (art_path, artifact_bytes))
        {
          return std::unexpected (Error::registry (
              RegistryErrorCode::IoFailure, std::string (model_name), 0U,
              "Failed to persist artifact in content-addressable store"));
        }
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const std::string mname (model_name);
  auto &versions = impl_->model_versions[mname];
  if (versions.size () != expected_prev_version)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::VersionConflict, mname, versions.size (),
          "Optimistic concurrency version check failed during registration"));
    }

  const uint64_t next_ver = versions.size () + 1U;
  const std::string composite_tag = mname + ":" + std::string (semver_tag);
  if (impl_->has_lineage_cycle_locked (composite_tag, lineage))
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::LineageCycleDetected, mname, next_ver,
          "Lineage cycle detected during transactional registration"));
    }

  ModelManifest manifest;
  manifest.model_name = mname;
  manifest.version = next_ver;
  manifest.semver_tag = std::string (semver_tag);
  manifest.stage = ModelStage::Staging;
  manifest.blake3_hex = blake3_hex;
  manifest.sha256_hex = sha256_hex;
  manifest.artifact_size_bytes = artifact_bytes.size ();
  manifest.tensor_count = parse_tensor_count_header (artifact_bytes, 1U);
  manifest.lineage = lineage;
  manifest.registered_at_ns = monotonic_time_ns ();
  manifest.artifact_path = art_path;

  versions.push_back (manifest);
  impl_->tag_index[composite_tag] = manifest;
  impl_->tag_index[mname + ":v" + std::to_string (next_ver)] = manifest;
  ++impl_->total_version_count;

  RegistryStats::global ().record_registration (monotonic_time_ns () - t0);
  return manifest;
}

auto
ModelRegistry::index_manifest (ModelManifest manifest) noexcept
    -> Result<ModelManifest>
{
  if (!impl_ || manifest.model_name.empty () || manifest.semver_tag.empty ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, "", 0U,
          "Cannot index manifest with empty model_name or semver_tag"));
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const std::string composite_tag
      = manifest.model_name + ":" + manifest.semver_tag;
  if (impl_->has_lineage_cycle_locked (composite_tag, manifest.lineage))
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::LineageCycleDetected, manifest.model_name,
          manifest.version,
          "Lineage cycle detected while indexing manifest"));
    }

  auto &versions = impl_->model_versions[manifest.model_name];
  if (manifest.version == 0U)
    {
      manifest.version = versions.size () + 1U;
    }
  if (manifest.registered_at_ns == 0U)
    {
      manifest.registered_at_ns = monotonic_time_ns ();
    }
  versions.push_back (manifest);
  impl_->tag_index[composite_tag] = manifest;
  impl_->tag_index[manifest.model_name + ":v"
                   + std::to_string (manifest.version)]
      = manifest;
  ++impl_->total_version_count;
  impl_->save_catalog_locked ();
  return manifest;
}

auto
ModelRegistry::transition_stage (std::string_view model_name,
                                 uint64_t version, ModelStage target_stage,
                                 std::string_view justification) noexcept
    -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "ModelRegistry is not initialized"));
    }
  if (justification.empty ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::InvalidStageTransition, std::string (model_name),
          version,
          "Stage transition requires a non-empty audit justification string"));
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  const std::string mname (model_name);
  auto it = impl_->model_versions.find (mname);
  if (it == impl_->model_versions.end () || version == 0U
      || version > it->second.size ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, mname, version,
          "Requested model version was not found in ModelRegistry"));
    }

  ModelManifest &manifest = it->second[version - 1U];
  const ModelStage prev_stage = manifest.stage;
  if (prev_stage == ModelStage::Archived
      && target_stage == ModelStage::Production)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::InvalidStageTransition, mname, version,
          "Cannot transition directly from ARCHIVED to PRODUCTION stage"));
    }
  if (prev_stage == target_stage)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::InvalidStageTransition, mname, version,
          "Model version is already in the requested target stage"));
    }

  manifest.stage = target_stage;
  impl_->tag_index[mname + ":" + manifest.semver_tag] = manifest;
  impl_->tag_index[mname + ":v" + std::to_string (version)] = manifest;
  impl_->save_catalog_locked ();

  StageTransitionAudit audit;
  audit.model_name = mname;
  audit.version = version;
  audit.from_stage = prev_stage;
  audit.to_stage = target_stage;
  audit.justification = std::string (justification);
  audit.timestamp_ns = monotonic_time_ns ();
  impl_->audits[mname].push_back (std::move (audit));

  RegistryStats::global ().record_stage_transition ();
  return {};
}

auto
ModelRegistry::lookup (std::string_view name_and_tag) const noexcept
    -> Result<ModelManifest>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (name_and_tag), 0U,
          "ModelRegistry is not initialized"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  auto it = impl_->tag_index.find (std::string (name_and_tag));
  if (it == impl_->tag_index.end ())
    {
      const auto colon = name_and_tag.find (':');
      if (colon != std::string_view::npos)
        {
          const std::string canon_tag
              = canonical_model_alias (name_and_tag.substr (0U, colon))
                + std::string (name_and_tag.substr (colon));
          it = impl_->tag_index.find (canon_tag);
        }
    }
  if (it == impl_->tag_index.end ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (name_and_tag), 0U,
          "Model tag was not found in ModelRegistry"));
    }
  return it->second;
}

auto
ModelRegistry::get_version (std::string_view model_name,
                            uint64_t version) const noexcept
    -> Result<ModelManifest>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "ModelRegistry is not initialized"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  auto it = impl_->model_versions.find (std::string (model_name));
  if (it == impl_->model_versions.end ())
    {
      it = impl_->model_versions.find (canonical_model_alias (model_name));
    }
  if (it == impl_->model_versions.end () || version == 0U
      || version > it->second.size ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "Model version was not found in ModelRegistry"));
    }
  return it->second[version - 1U];
}

auto
ModelRegistry::list_versions (std::string_view model_name) const noexcept
    -> Result<std::vector<ModelManifest>>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name), 0U,
          "ModelRegistry is not initialized"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const auto it = impl_->model_versions.find (std::string (model_name));
  if (it == impl_->model_versions.end ())
    {
      return std::vector<ModelManifest>{};
    }
  return it->second;
}

auto
ModelRegistry::query_lineage (std::string_view model_name,
                              uint64_t version) const noexcept
    -> Result<std::vector<ModelManifest>>
{
  const uint64_t t0 = monotonic_time_ns ();
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "ModelRegistry is not initialized"));
    }

  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const auto it = impl_->model_versions.find (std::string (model_name));
  if (it == impl_->model_versions.end () || version == 0U
      || version > it->second.size ())
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "Model version not found for lineage query"));
    }

  std::vector<ModelManifest> chain;
  std::unordered_set<std::string> visited;
  std::vector<ModelManifest> queue;
  queue.push_back (it->second[version - 1U]);

  while (!queue.empty ())
    {
      ModelManifest curr = std::move (queue.front ());
      queue.erase (queue.begin ());
      const std::string key
          = curr.model_name + ":v" + std::to_string (curr.version);
      if (!visited.insert (key).second)
        {
          continue;
        }
      for (const auto &parent_ref : curr.lineage.parent_model_versions)
        {
          const auto pit = impl_->tag_index.find (parent_ref);
          if (pit != impl_->tag_index.end ())
            {
              queue.push_back (pit->second);
            }
        }
      chain.push_back (std::move (curr));
    }

  RegistryStats::global ().record_lineage_query (monotonic_time_ns () - t0);
  return chain;
}

auto
ModelRegistry::audit_trail (std::string_view model_name) const noexcept
    -> Result<std::vector<StageTransitionAudit>>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name), 0U,
          "ModelRegistry is not initialized"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const auto it = impl_->audits.find (std::string (model_name));
  if (it == impl_->audits.end ())
    {
      return std::vector<StageTransitionAudit>{};
    }
  return it->second;
}

auto
ModelRegistry::load_artifact_warm_mmap (std::string_view model_name,
                                        uint64_t version) const noexcept
    -> Result<VerifiedArtifactLoad>
{
  auto man_res = get_version (model_name, version);
  if (!man_res.has_value ())
    {
      return std::unexpected (man_res.error ());
    }
  const ModelManifest &man = *man_res;

  const int raw_fd = ::open (man.artifact_path.c_str (), O_RDONLY);
  if (raw_fd < 0)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "Artifact file missing on disk for warm mmap"));
    }
  ScopedFd fd_guard (raw_fd);

  struct stat st
  {
  };
  if (::fstat (fd_guard.get (), &st) != 0
      || static_cast<std::size_t> (st.st_size) != man.artifact_size_bytes)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::HashMismatch, std::string (model_name), version,
          "Artifact on-disk byte size does not match manifest"));
    }

  const std::size_t len = man.artifact_size_bytes;
  if (len == 0U)
    {
      VerifiedArtifactLoad res;
      res.sha256_hex = compute_sha256_hex ({});
      res.blake3_hex = man.blake3_hex;
      res.bytes_verified = 0U;
      res.tensor_count = man.tensor_count;
      return res;
    }

  void *mapped = ::mmap (nullptr, len, PROT_READ, MAP_PRIVATE,
                         fd_guard.get (), 0);
  if (mapped == MAP_FAILED)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::IoFailure, std::string (model_name), version,
          "mmap syscall failed while loading model artifact"));
    }
  ScopedMmap mmap_guard (mapped, len);

  const std::span<const std::byte> span_bytes (mmap_guard.data (), len);
  std::string computed_sha;
  if (len > kUltraLargeFileThresholdBytes)
    {
      const std::size_t header_pop
          = std::min (len, kFastDigestPrefixBytes);
      if (header_pop > 0U)
        {
          (void)::madvise (mapped, header_pop, MADV_WILLNEED);
        }
      const std::vector<std::byte> fast_sample
          = build_fast_large_artifact_sample (span_bytes);
      computed_sha = compute_sha256_hex (fast_sample);
    }
  else
    {
      computed_sha = compute_sha256_hex (span_bytes);
    }
  if (computed_sha != man.sha256_hex)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::HashMismatch, std::string (model_name), version,
          "SHA-256 digest mismatch during warm mmap verification"));
    }

  VerifiedArtifactLoad out;
  out.sha256_hex = computed_sha;
  out.blake3_hex = man.blake3_hex;
  out.bytes_verified = len;
  out.tensor_count = parse_tensor_count_header (span_bytes, man.tensor_count);
  return out;
}

auto
ModelRegistry::load_artifact_cold_nvme (std::string_view model_name,
                                        uint64_t version) const noexcept
    -> Result<VerifiedArtifactLoad>
{
  auto man_res = get_version (model_name, version);
  if (!man_res.has_value ())
    {
      return std::unexpected (man_res.error ());
    }
  const ModelManifest &man = *man_res;

  const int raw_fd = ::open (man.artifact_path.c_str (), O_RDONLY);
  if (raw_fd < 0)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, std::string (model_name),
          version, "Artifact file missing on disk for cold NVMe load"));
    }
  ScopedFd fd_guard (raw_fd);

  const std::size_t len = man.artifact_size_bytes;
  if (len > 0U)
    {
      (void)::fdatasync (fd_guard.get ());
      (void)::posix_fadvise (fd_guard.get (), 0, static_cast<off_t> (len),
                             POSIX_FADV_DONTNEED);
    }

  thread_local std::vector<std::byte> read_buf;
  read_buf.resize (len);

  std::size_t total_read = 0U;
  auto *dst = reinterpret_cast<uint8_t *> (read_buf.data ());
  while (total_read < len)
    {
      const ssize_t n = ::pread (fd_guard.get (), dst + total_read,
                                 len - total_read,
                                 static_cast<off_t> (total_read));
      if (n <= 0)
        {
          return std::unexpected (Error::registry (
              RegistryErrorCode::IoFailure, std::string (model_name), version,
              "pread syscall failed during cold NVMe artifact read"));
        }
      total_read += static_cast<std::size_t> (n);
    }

  const std::span<const std::byte> span_bytes (read_buf.data (), len);
  const std::string computed_sha = compute_sha256_hex (span_bytes);
  if (computed_sha != man.sha256_hex)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::HashMismatch, std::string (model_name), version,
          "SHA-256 digest mismatch during cold NVMe verification"));
    }

  VerifiedArtifactLoad out;
  out.sha256_hex = computed_sha;
  out.blake3_hex = man.blake3_hex;
  out.bytes_verified = len;
  out.tensor_count = parse_tensor_count_header (span_bytes, man.tensor_count);
  return out;
}

auto
ModelRegistry::total_versions () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return impl_->total_version_count;
}

auto
ModelRegistry::ensure_standard_models (
    const std::filesystem::path &models_dir) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::registry (
          RegistryErrorCode::ArtifactNotFound, "", 0U,
          "ModelRegistry is not initialized"));
    }

  struct StandardEntry
  {
    std::string_view id;
    std::string_view tag;
    std::string_view filename;
    uint32_t tensors;
  };

  constexpr std::array<StandardEntry, 6U> kEntries = { {
      { "qwen3-coder-next-80b-q4ks", "v1.0.0",
        "Qwen3-Coder-Next-Q4_K_S.gguf", 843U },
      { "qwen3-30b-a3b-q4km", "v1.0.0", "Qwen3-30B-A3B-Q4_K_M.gguf", 579U },
      { "gemma-4-31b-it-q4km", "v1.0.0", "gemma-4-31B-it-Q4_K_M.gguf", 833U },
      { "qwen2.5-7b-instruct-q4km", "v1.0.0",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf", 291U },
      { "qwen2.5-1.5b-q4km", "v1.0.0",
        "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", 291U },
      { "qwen3-embedding-0.6b-q8", "v1.0.0",
        "Qwen3-Embedding-0.6B-Q8_0.gguf", 310U },
  } };

  const char *env_home = std::getenv ("HOME");
  const std::filesystem::path base_dir
      = !models_dir.empty ()
            ? models_dir
            : ((env_home != nullptr && env_home[0] != '\0')
                   ? (std::filesystem::path (env_home) / "models")
                   : std::filesystem::path ("models"));
  const char *env_user = std::getenv ("USER");
  const std::string user_str
      = (env_user != nullptr && env_user[0] != '\0')
            ? std::string (env_user)
            : std::string ("user");

  for (const StandardEntry &entry : kEntries)
    {
      std::filesystem::path fpath = base_dir / entry.filename;
      std::error_code ec;
      if (!std::filesystem::exists (fpath, ec)
          && entry.id == "qwen3-coder-next-80b-q4ks")
        {
          fpath = std::filesystem::path ("/media") / user_str / "SphinxAI"
                  / "models" / "Qwen3-Coder-Next-Q4_K_S.gguf";
        }
      if (!std::filesystem::exists (fpath, ec))
        {
          continue;
        }
      auto existing = get_version (entry.id, 1U);
      if (existing && !existing->sha256_hex.empty ()
          && std::filesystem::exists (existing->artifact_path, ec))
        {
          continue;
        }

      ModelLineage lineage;
      lineage.run_id = "sphinx_bootstrap";
      lineage.dataset_hash
          = "0000000000000000000000000000000000000000000000000000000000000000";
      lineage.code_commit_hash = "sphinx-v1.1.0";

      auto reg_res
          = register_file (entry.id, entry.tag, fpath, lineage, entry.tensors,
                           true);
      if (!reg_res)
        {
          return std::unexpected (reg_res.error ());
        }
      (void)transition_stage (
          entry.id, reg_res->version, ModelStage::Production,
          "Approved for Sphinx production deployment");
    }

  // Ensure canonical aliases are also indexed in O(1) without re-hashing.
  struct AliasEntry
  {
    std::string_view alias_id;
    std::string_view target_id;
  };
  constexpr std::array<AliasEntry, 4U> kAliases = { {
      { "qwen2.5-1.5b-instruct-q4km", "qwen2.5-1.5b-q4km" },
      { "qwen3-embedding-0.6b-q8_0", "qwen3-embedding-0.6b-q8" },
      { "qwen3-coder-next", "qwen3-coder-next-80b-q4ks" },
      { "qwen3-80b", "qwen3-coder-next-80b-q4ks" },
  } };
  for (const AliasEntry &al : kAliases)
    {
      if (get_version (al.alias_id, 1U)
          && impl_->model_versions.contains (std::string (al.alias_id)))
        {
          continue;
        }
      auto target_man = get_version (al.target_id, 1U);
      if (target_man)
        {
          ModelManifest alias_man = *target_man;
          alias_man.model_name = std::string (al.alias_id);
          alias_man.version = 1U;
          (void)index_manifest (std::move (alias_man));
        }
    }
  return {};
}

} /* namespace yantra::inline v1 */
