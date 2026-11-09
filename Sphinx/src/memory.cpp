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

#include <sphinx/memory.hpp>

#include <llama.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sphinx::inline v1
{

namespace
{

constexpr uint32_t kPayloadMagic = 0x534D454DU; /* "SMEM" */
constexpr float kBaseAnisotropyScale = 0.45F;
constexpr float kTopicProjectionWeight = 0.85F;

[[nodiscard]] auto
current_epoch_ns () noexcept -> uint64_t
{
  const auto now = std::chrono::system_clock::now ().time_since_epoch ();
  return static_cast<uint64_t> (
      std::chrono::duration_cast<std::chrono::nanoseconds> (now).count ());
}

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

[[nodiscard]] auto
starts_with_icase (std::string_view text,
                   std::string_view prefix) noexcept -> bool
{
  if (text.size () < prefix.size ())
    {
      return false;
    }
  for (std::size_t i = 0U; i < prefix.size (); ++i)
    {
      const auto ca = static_cast<unsigned char> (text[i]);
      const auto cb = static_cast<unsigned char> (prefix[i]);
      if (std::tolower (ca) != std::tolower (cb))
        {
          return false;
        }
    }
  return true;
}

[[nodiscard]] auto
trim_ascii (std::string_view sv) noexcept -> std::string_view
{
  while (!sv.empty ()
         && std::isspace (static_cast<unsigned char> (sv.front ())) != 0)
    {
      sv.remove_prefix (1U);
    }
  while (!sv.empty ()
         && std::isspace (static_cast<unsigned char> (sv.back ())) != 0)
    {
      sv.remove_suffix (1U);
    }
  return sv;
}

void
replace_word_icase (std::string &text, std::string_view from_word,
                    std::string_view to_phrase)
{
  std::string out;
  out.reserve (text.size () + 16U);
  std::size_t i = 0U;
  while (i < text.size ())
    {
      const bool at_word_start
          = (i == 0U)
            || (std::isalnum (static_cast<unsigned char> (text[i - 1U])) == 0);
      const bool fits = (i + from_word.size () <= text.size ());
      const bool at_word_end
          = fits
            && ((i + from_word.size () == text.size ())
                || (std::isalnum (static_cast<unsigned char> (
                        text[i + from_word.size ()]))
                    == 0));
      if (at_word_start && fits && at_word_end
          && starts_with_icase (
              std::string_view (text).substr (i, from_word.size ()),
              from_word))
        {
          out.append (to_phrase);
          i += from_word.size ();
        }
      else
        {
          out.push_back (text[i]);
          ++i;
        }
    }
  text = std::move (out);
}

[[nodiscard]] auto
normalize_person_for_embedding (std::string_view text) -> std::string
{
  std::string norm (trim_ascii (text));
  replace_word_icase (norm, "my", "the user's");
  replace_word_icase (norm, "mine", "the user's");
  replace_word_icase (norm, "i am", "the user is");
  replace_word_icase (norm, "i'm", "the user is");
  replace_word_icase (norm, "am i", "is the user");
  return norm;
}

[[nodiscard]] auto
fnv1a_64 (std::string_view token, uint64_t salt) noexcept -> uint64_t
{
  uint64_t h = 14695981039346656037ULL ^ salt;
  for (const char c : token)
    {
      h ^= static_cast<uint64_t> (static_cast<unsigned char> (c));
      h *= 1099511628211ULL;
    }
  return h;
}

[[nodiscard]] auto
is_stop_word (std::string_view w) noexcept -> bool
{
  constexpr std::array<std::string_view, 58U> kStops = { {
      "the",       "is",       "are",      "was",     "were",     "a",
      "an",        "what",     "who",      "which",   "where",    "when",
      "how",       "why",      "am",       "i",       "my",       "me",
      "you",       "your",     "user",     "users",   "on",       "in",
      "of",        "to",       "for",      "that",    "this",     "it",
      "its",       "no",       "not",      "remember","note",     "sphinx",
      "assistant", "hello",    "today",    "explain", "write",    "tell",
      "give",      "list",     "current",  "directory","file",    "files",
      "check",     "compile",  "compiles", "run",     "connect",  "localhost",
      "server",    "code",     "function", "about",
  } };
  for (const std::string_view s : kStops)
    {
      if (w == s)
        {
          return true;
        }
    }
  return false;
}

void
l2_normalize (std::span<float> vec) noexcept
{
  double sum_sq = 0.0;
  for (const float v : vec)
    {
      sum_sq += static_cast<double> (v) * static_cast<double> (v);
    }
  if (sum_sq > 1.0e-18)
    {
      const float inv = static_cast<float> (1.0 / std::sqrt (sum_sq));
      for (float &v : vec)
        {
          v *= inv;
        }
    }
}

[[nodiscard]] auto
build_topic_projection (std::string_view text, std::size_t dim,
                        bool *has_topic_words_out = nullptr)
    -> std::vector<float>
{
  std::vector<float> out (dim, 0.0F);
  if (has_topic_words_out != nullptr)
    {
      *has_topic_words_out = false;
    }
  if (dim == 0U)
    {
      return out;
    }

  std::string cur;
  std::string raw_alnum;
  bool any_topic_word = false;
  auto flush_word = [&] () {
    if (cur.empty ())
      {
        return;
      }
    if (is_stop_word (cur))
      {
        cur.clear ();
        return;
      }
    any_topic_word = true;
    std::string canonical = cur;
    if (canonical == "called" || canonical == "named")
      {
        canonical = "name";
      }
    else if (canonical == "working")
      {
        canonical = "project";
      }
    else if (canonical == "talked" || canonical == "discussed"
             || canonical == "discuss")
      {
        canonical = "talk";
      }
    else if (canonical == "previous" || canonical == "earlier")
      {
        canonical = "last";
      }

    float weight = 1.0F;
    if (canonical == "name" || canonical == "project" || canonical == "talk"
        || canonical == "last" || canonical == "time"
        || canonical == "session")
      {
        weight = 2.5F;
      }

    for (uint64_t k = 0U; k < 12U; ++k)
      {
        const uint64_t h = fnv1a_64 (canonical, k * 0x9E3779B97F4A7C15ULL);
        const std::size_t idx = static_cast<std::size_t> (h % dim);
        const float sign = ((h >> 32U) & 1ULL) != 0ULL ? 1.0F : -1.0F;
        out[idx] += sign * weight;
      }
    cur.clear ();
  };

  for (const char ch : text)
    {
      const auto uch = static_cast<unsigned char> (ch);
      if (std::isalnum (uch) != 0)
        {
          const char lc = static_cast<char> (std::tolower (uch));
          cur.push_back (lc);
          raw_alnum.push_back (lc);
        }
      else
        {
          flush_word ();
        }
    }
  flush_word ();
  if (has_topic_words_out != nullptr)
    {
      *has_topic_words_out = any_topic_word;
    }
  if (!any_topic_word && !raw_alnum.empty ())
    {
      const std::string sentinel = "__stopword_query__:" + raw_alnum;
      for (uint64_t k = 0U; k < 12U; ++k)
        {
          const uint64_t h = fnv1a_64 (sentinel, k * 0x9E3779B97F4A7C15ULL);
          const std::size_t idx = static_cast<std::size_t> (h % dim);
          const float sign = ((h >> 32U) & 1ULL) != 0ULL ? 1.0F : -1.0F;
          out[idx] += sign;
        }
    }
  l2_normalize (out);
  return out;
}

[[nodiscard]] auto
normalize_fact_sentence (std::string_view raw) -> std::string
{
  std::string s (trim_ascii (raw));
  while (!s.empty ()
         && (s.back () == '.' || s.back () == '!' || s.back () == '?'))
    {
      s.pop_back ();
    }
  s = std::string (trim_ascii (s));
  if (s.empty ())
    {
      return {};
    }

  if (starts_with_icase (s, "my "))
    {
      s = "The user's " + s.substr (3U);
    }
  else if (starts_with_icase (s, "i am "))
    {
      s = "The user is " + s.substr (5U);
    }
  else if (starts_with_icase (s, "i'm "))
    {
      s = "The user is " + s.substr (4U);
    }
  else if (starts_with_icase (s, "i use "))
    {
      s = "The user uses " + s.substr (6U);
    }
  else if (starts_with_icase (s, "i prefer "))
    {
      s = "The user prefers " + s.substr (9U);
    }
  else
    {
      replace_word_icase (s, "my", "the user's");
      s[0] = static_cast<char> (
          std::toupper (static_cast<unsigned char> (s[0])));
    }

  if (s.back () != '.')
    {
      s.push_back ('.');
    }
  return s;
}

} /* namespace */

/* ========================================================================= */
/* 1. EmbeddingClient                                                        */
/* ========================================================================= */

struct EmbeddingClient::Impl
{
  EmbeddingClientConfig config;
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  const llama_vocab *vocab = nullptr;
  std::size_t dim = 1024U;
  bool enabled = false;
  std::vector<float> base_vec;

  struct EmbedTask
  {
    std::string text;
    std::promise<sutra::Result<std::vector<float>>> promise;
  };

  std::mutex q_mtx;
  std::condition_variable q_cv;
  std::deque<EmbedTask> queue;
  bool stop_worker = false;
  std::jthread worker;

  ~Impl () noexcept
  {
    {
      const std::scoped_lock lock (q_mtx);
      stop_worker = true;
    }
    q_cv.notify_all ();
    if (worker.joinable ())
      {
        worker.join ();
      }
    if (ctx != nullptr)
      {
        llama_free (ctx);
        ctx = nullptr;
      }
    if (model != nullptr)
      {
        llama_model_free (model);
        model = nullptr;
      }
  }

  [[nodiscard]] auto
  compute_raw_embedding_locked (std::string_view input_text) noexcept
      -> sutra::Result<std::vector<float>>
  {
    if (ctx == nullptr || model == nullptr || vocab == nullptr)
      {
        return std::unexpected (sutra::AgentError::framework (
            sutra::FrameworkErrorCode::InvalidConfiguration, 0,
            "EmbeddingClient is disabled"));
      }

    std::string clean_str;
    clean_str.reserve (input_text.size ());
    for (const char ch : input_text)
      {
        if (ch != '.' && ch != '?' && ch != '!' && ch != ',' && ch != ';')
          {
            clean_str.push_back (ch);
          }
      }
    const int32_t max_toks = static_cast<int32_t> (config.n_ctx);
    std::vector<llama_token> tokens (static_cast<std::size_t> (max_toks));
    int32_t n_toks = llama_tokenize (
        vocab, clean_str.c_str (), static_cast<int32_t> (clean_str.size ()),
        tokens.data (), max_toks, false, false);
    if (n_toks < 0)
      {
        n_toks = -n_toks;
        if (n_toks > max_toks)
          {
            n_toks = max_toks;
          }
      }
    if (n_toks == 0)
      {
        const llama_token eos_tok = llama_vocab_eos (vocab);
        tokens[0] = (eos_tok >= 0) ? eos_tok : 0;
        n_toks = 1;
      }
    else if (n_toks > 1)
      {
        tokens[0] = tokens[static_cast<std::size_t> (n_toks - 1)];
        n_toks = 1;
      }
    tokens.resize (1U);

    llama_memory_t mem = llama_get_memory (ctx);
    if (mem != nullptr)
      {
        llama_memory_clear (mem, false);
      }

    llama_batch batch = llama_batch_init (n_toks, 0, 1);
    batch.n_tokens = n_toks;
    for (int32_t i = 0; i < n_toks; ++i)
      {
        batch.token[i] = tokens[static_cast<std::size_t> (i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_toks - 1) ? 1 : 0;
      }

    int32_t rc = 0;
    if (llama_model_has_encoder (model) && !llama_model_has_decoder (model))
      {
        rc = llama_encode (ctx, batch);
      }
    else
      {
        rc = llama_decode (ctx, batch);
      }

    if (rc < 0)
      {
        llama_batch_free (batch);
        return std::unexpected (sutra::AgentError::framework (
            sutra::FrameworkErrorCode::InvalidConfiguration, 0,
            "llama_decode failed in EmbeddingClient"));
      }

    std::vector<float> out (dim, 0.0F);
    const enum llama_pooling_type ptype = llama_pooling_type (ctx);
    const float *seq_emb
        = (ptype != LLAMA_POOLING_TYPE_NONE)
              ? llama_get_embeddings_seq (ctx, 0)
              : nullptr;
    if (seq_emb != nullptr)
      {
        std::memcpy (out.data (), seq_emb, dim * sizeof (float));
      }
    else
      {
        int32_t count = 0;
        for (int32_t i = 0; i < n_toks; ++i)
          {
            const float *tok_emb = llama_get_embeddings_ith (ctx, i);
            if (tok_emb != nullptr)
              {
                for (std::size_t d = 0U; d < dim; ++d)
                  {
                    out[d] += tok_emb[d];
                  }
                ++count;
              }
          }
        if (count > 0)
          {
            const float inv = 1.0F / static_cast<float> (count);
            for (float &v : out)
              {
                v *= inv;
              }
          }
      }

    llama_batch_free (batch);
    l2_normalize (out);
    return out;
  }

  [[nodiscard]] auto
  compute_embedding_locked (std::string_view input_text) noexcept
      -> sutra::Result<std::vector<float>>
  {
    const std::string norm_text = normalize_person_for_embedding (input_text);
    auto raw_res = compute_raw_embedding_locked (norm_text);
    if (!raw_res)
      {
        return raw_res;
      }
    std::vector<float> vec = std::move (*raw_res);
    if (base_vec.size () == vec.size ())
      {
        for (std::size_t i = 0U; i < vec.size (); ++i)
          {
            vec[i] -= kBaseAnisotropyScale * base_vec[i];
          }
        l2_normalize (vec);
      }

    bool has_topic_words = false;
    const std::vector<float> topic_vec
        = build_topic_projection (norm_text, dim, &has_topic_words);
    double topic_norm_sq = 0.0;
    for (const float tv : topic_vec)
      {
        topic_norm_sq += static_cast<double> (tv) * static_cast<double> (tv);
      }
    if (topic_norm_sq > 0.1)
      {
        if (!has_topic_words)
          {
            vec = topic_vec;
          }
        else
          {
            for (std::size_t i = 0U; i < vec.size (); ++i)
              {
                vec[i] += kTopicProjectionWeight * topic_vec[i];
              }
            l2_normalize (vec);
          }
      }
    return vec;
  }
};

EmbeddingClient::EmbeddingClient (PrivateTag /*tag*/,
                                  std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

EmbeddingClient::EmbeddingClient (EmbeddingClient &&) noexcept = default;
auto EmbeddingClient::operator= (EmbeddingClient &&) noexcept
    -> EmbeddingClient & = default;
EmbeddingClient::~EmbeddingClient () noexcept = default;

auto
EmbeddingClient::create (const EmbeddingClientConfig &config) noexcept
    -> sutra::Result<std::unique_ptr<EmbeddingClient>>
{
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->config = config;

      std::error_code ec;
      if (!std::filesystem::exists (config.model_path, ec))
        {
          (void)std::fprintf (
              stderr,
              "[sphinx] Embedding model not found. Memory is DISABLED. "
              "Download Qwen3-Embedding-0.6B-Q8_0.gguf to ~/models/ to "
              "enable.\n");
          impl->enabled = false;
          return std::make_unique<EmbeddingClient> (PrivateTag{},
                                                    std::move (impl));
        }

      sutra::LlamaCppClient::init_backend ();

      llama_model_params mparams = llama_model_default_params ();
      mparams.n_gpu_layers = 0; /* M3: strictly CPU, 0 GPU layers */
      mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
      mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
      std::array<ggml_backend_dev_t, 2U> cpu_devs = { {
          ggml_backend_dev_by_type (GGML_BACKEND_DEVICE_TYPE_CPU),
          nullptr,
      } };
      if (cpu_devs[0] != nullptr)
        {
          mparams.devices = cpu_devs.data ();
        }

      impl->model = llama_model_load_from_file (
          config.model_path.string ().c_str (), mparams);
      if (impl->model == nullptr)
        {
          (void)std::fprintf (
              stderr,
              "[sphinx] Embedding model failed to load. Memory is "
              "DISABLED.\n");
          impl->enabled = false;
          return std::make_unique<EmbeddingClient> (PrivateTag{},
                                                    std::move (impl));
        }

      impl->vocab = llama_model_get_vocab (impl->model);
      int32_t n_embd = llama_model_n_embd_out (impl->model);
      if (n_embd <= 0)
        {
          n_embd = llama_model_n_embd (impl->model);
        }
      impl->dim = (n_embd > 0) ? static_cast<std::size_t> (n_embd) : 1024U;

      llama_context_params cparams = llama_context_default_params ();
      cparams.n_ctx = config.n_ctx;
      cparams.n_batch = config.n_ctx;
      cparams.n_ubatch = config.n_ctx;
      cparams.n_threads = 1;
      cparams.n_threads_batch = 1;
      cparams.embeddings = true;
      cparams.pooling_type = LLAMA_POOLING_TYPE_LAST;
      cparams.offload_kqv = false;
      cparams.op_offload = false;
      cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

      impl->ctx = llama_init_from_model (impl->model, cparams);
      if (impl->ctx == nullptr)
        {
          llama_model_free (impl->model);
          impl->model = nullptr;
          impl->enabled = false;
          return std::make_unique<EmbeddingClient> (PrivateTag{},
                                                    std::move (impl));
        }

      auto base_res = impl->compute_raw_embedding_locked ("");
      if (base_res)
        {
          impl->base_vec = std::move (*base_res);
        }

      impl->enabled = true;
      Impl *raw_impl = impl.get ();
      impl->worker = std::jthread ([raw_impl] {
        while (true)
          {
            Impl::EmbedTask task;
            {
              std::unique_lock<std::mutex> lock (raw_impl->q_mtx);
              raw_impl->q_cv.wait (lock, [raw_impl] {
                return !raw_impl->queue.empty () || raw_impl->stop_worker;
              });
              if (raw_impl->queue.empty () && raw_impl->stop_worker)
                {
                  break;
                }
              task = std::move (raw_impl->queue.front ());
              raw_impl->queue.pop_front ();
            }
            auto res = raw_impl->compute_embedding_locked (task.text);
            task.promise.set_value (std::move (res));
          }
      });

      auto client = std::make_unique<EmbeddingClient> (PrivateTag{},
                                                       std::move (impl));
      return client;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
           sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in EmbeddingClient::create"));
    }
}

auto
EmbeddingClient::embed (std::string_view text) noexcept
    -> sutra::Result<std::vector<float>>
{
  if (impl_ == nullptr || !impl_->enabled)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "EmbeddingClient is not enabled"));
    }
  try
    {
      Impl::EmbedTask task;
      task.text = std::string (text);
      auto fut = task.promise.get_future ();
      {
        const std::scoped_lock lock (impl_->q_mtx);
        impl_->queue.push_back (std::move (task));
      }
      impl_->q_cv.notify_one ();
      return fut.get ();
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in EmbeddingClient::embed"));
    }
}

auto
EmbeddingClient::dimension () const noexcept -> std::size_t
{
  return (impl_ != nullptr) ? impl_->dim : 0U;
}

auto
EmbeddingClient::is_enabled () const noexcept -> bool
{
  return (impl_ != nullptr) && impl_->enabled;
}

auto
EmbeddingClient::cosine_similarity (std::span<const float> a,
                                    std::span<const float> b) noexcept -> float
{
  if (a.empty () || a.size () != b.size ())
    {
      return 0.0F;
    }
  return sutra::simd_dot_f32 (a, b);
}

/* ========================================================================= */
/* 2. MemoryStore                                                            */
/* ========================================================================= */

struct MemoryStore::Impl
{
  std::filesystem::path memory_dir;
  std::filesystem::path wal_path;
  std::filesystem::path dat_path;
  std::filesystem::path hnsw_path;
  std::size_t dimension = 1024U;

  yantra::WalEngine wal;
  std::unique_ptr<yantra::HnswIndex> hnsw;
  std::vector<MemoryEntry> entries;
  MemoryId next_id = 1U;
  bool is_open = false;
  mutable std::mutex mtx;

  [[nodiscard]] auto
  append_payload_record (const MemoryEntry &e, bool sync_disk) noexcept -> bool
  {
    FILE *fp = std::fopen (dat_path.string ().c_str (), "ab");
    if (fp == nullptr)
      {
        return false;
      }
    const uint32_t magic = kPayloadMagic;
    const uint64_t id = e.id;
    const uint8_t type_u8 = static_cast<uint8_t> (e.type);
    const uint64_t ts = e.timestamp_ns;
    const uint64_t sup = e.superseded_by.value_or (0U);
    const uint32_t text_len = static_cast<uint32_t> (e.text.size ());
    const uint32_t emb_dim = static_cast<uint32_t> (e.embedding.size ());

    bool ok = true;
    ok = ok && (std::fwrite (&magic, sizeof (magic), 1U, fp) == 1U);
    ok = ok && (std::fwrite (&id, sizeof (id), 1U, fp) == 1U);
    ok = ok && (std::fwrite (&type_u8, sizeof (type_u8), 1U, fp) == 1U);
    ok = ok && (std::fwrite (&ts, sizeof (ts), 1U, fp) == 1U);
    ok = ok && (std::fwrite (&sup, sizeof (sup), 1U, fp) == 1U);
    ok = ok && (std::fwrite (&text_len, sizeof (text_len), 1U, fp) == 1U);
    ok = ok && (std::fwrite (&emb_dim, sizeof (emb_dim), 1U, fp) == 1U);
    if (text_len > 0U)
      {
        ok = ok
             && (std::fwrite (e.text.data (), 1U, text_len, fp) == text_len);
      }
    if (emb_dim > 0U)
      {
        ok = ok
             && (std::fwrite (e.embedding.data (), sizeof (float), emb_dim, fp)
                 == emb_dim);
      }
    (void)std::fflush (fp);
    if (sync_disk)
      {
        (void)fdatasync (fileno (fp));
      }
    (void)std::fclose (fp);
    return ok;
  }

  void
  load_payload_records ()
  {
    entries.clear ();
    FILE *fp = std::fopen (dat_path.string ().c_str (), "rb");
    if (fp == nullptr)
      {
        return;
      }
    while (true)
      {
        uint32_t magic = 0U;
        if (std::fread (&magic, sizeof (magic), 1U, fp) != 1U
            || magic != kPayloadMagic)
          {
            break;
          }
        uint64_t id = 0U;
        uint8_t type_u8 = 0U;
        uint64_t ts = 0U;
        uint64_t sup = 0U;
        uint32_t text_len = 0U;
        uint32_t emb_dim = 0U;
        if (std::fread (&id, sizeof (id), 1U, fp) != 1U
            || std::fread (&type_u8, sizeof (type_u8), 1U, fp) != 1U
            || std::fread (&ts, sizeof (ts), 1U, fp) != 1U
            || std::fread (&sup, sizeof (sup), 1U, fp) != 1U
            || std::fread (&text_len, sizeof (text_len), 1U, fp) != 1U
            || std::fread (&emb_dim, sizeof (emb_dim), 1U, fp) != 1U)
          {
            break;
          }
        std::string txt (text_len, '\0');
        if (text_len > 0U
            && std::fread (txt.data (), 1U, text_len, fp) != text_len)
          {
            break;
          }
        std::vector<float> emb (emb_dim, 0.0F);
        if (emb_dim > 0U
            && std::fread (emb.data (), sizeof (float), emb_dim, fp)
                   != emb_dim)
          {
            break;
          }

        MemoryEntry entry;
        entry.id = id;
        entry.type = static_cast<MemoryType> (type_u8);
        entry.timestamp_ns = ts;
        if (sup > 0U)
          {
            entry.superseded_by = sup;
          }
        entry.text = std::move (txt);
        entry.embedding = std::move (emb);
        if (id >= next_id)
          {
            next_id = id + 1U;
          }
        entries.push_back (std::move (entry));
      }
    (void)std::fclose (fp);
  }

  void
  rebuild_hnsw ()
  {
    hnsw = std::make_unique<yantra::HnswIndex> (dimension);
    for (const MemoryEntry &e : entries)
      {
        if (e.embedding.size () == dimension)
          {
            yantra::VectorMetadata meta;
            meta.id = e.id;
            meta.timestamp = e.timestamp_ns;
            meta.category = static_cast<int64_t> (e.type);
            (void)hnsw->add (e.embedding, e.id, meta);
          }
      }
    if (hnsw->size () > 0U)
      {
        (void)hnsw->save (hnsw_path);
      }
  }
};

MemoryStore::MemoryStore (PrivateTag /*tag*/,
                          std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

MemoryStore::MemoryStore (MemoryStore &&) noexcept = default;
auto MemoryStore::operator= (MemoryStore &&) noexcept
    -> MemoryStore & = default;
MemoryStore::~MemoryStore () noexcept
{
  (void)close ();
}

auto
MemoryStore::open (const std::filesystem::path &memory_dir,
                   std::size_t dimension) noexcept
    -> sutra::Result<std::unique_ptr<MemoryStore>>
{
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->memory_dir = memory_dir;
      impl->wal_path = memory_dir / "sphinx_memory.wal";
      impl->dat_path = memory_dir / "sphinx_memory.dat";
      impl->hnsw_path = memory_dir / "sphinx_memory_hnsw";
      impl->dimension = (dimension > 0U) ? dimension : 1024U;

      std::error_code ec;
      std::filesystem::create_directories (memory_dir, ec);

      const bool wal_exists = std::filesystem::exists (impl->wal_path, ec);
      if (wal_exists)
        {
          impl->load_payload_records ();
          auto rec_res = yantra::WalEngine::recover (impl->wal_path);
          if (rec_res)
            {
              for (const yantra::WalRecord &wrec : *rec_res)
                {
                  if (wrec.flags == 1U)
                    {
                      for (MemoryEntry &e : impl->entries)
                        {
                          if (e.id == wrec.run_id_num)
                            {
                              e.superseded_by = wrec.payload_aux;
                            }
                        }
                    }
                }
            }
          auto wopen = yantra::WalEngine::open (impl->wal_path, false);
          if (!wopen)
            {
              wopen = yantra::WalEngine::open (impl->wal_path, true);
            }
          if (!wopen)
            {
              return std::unexpected (sutra::AgentError::framework (
                  sutra::FrameworkErrorCode::InvalidConfiguration, 0,
                  "Failed to open sphinx_memory.wal"));
            }
          impl->wal = std::move (*wopen);
        }
      else
        {
          auto wopen = yantra::WalEngine::open (impl->wal_path, true);
          if (!wopen)
            {
              return std::unexpected (sutra::AgentError::framework (
                  sutra::FrameworkErrorCode::InvalidConfiguration, 0,
                  "Failed to create sphinx_memory.wal"));
            }
          impl->wal = std::move (*wopen);
        }

      impl->rebuild_hnsw ();

      impl->is_open = true;
      return std::make_unique<MemoryStore> (PrivateTag{}, std::move (impl));
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in MemoryStore::open"));
    }
}

auto
MemoryStore::store (std::string_view text, std::span<const float> embedding,
                    MemoryType type, uint64_t timestamp_ns) noexcept
    -> sutra::Result<MemoryId>
{
  if (impl_ == nullptr || !impl_->is_open)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "MemoryStore is not open"));
    }
  if (embedding.size () != impl_->dimension)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "Embedding dimension mismatch in MemoryStore::store"));
    }

  try
    {
      const std::scoped_lock lock (impl_->mtx);
      const MemoryId id = impl_->next_id++;
      const uint64_t ts
          = (timestamp_ns > 0U) ? timestamp_ns : current_epoch_ns ();

      MemoryEntry entry;
      entry.id = id;
      entry.text = std::string (text);
      entry.embedding.assign (embedding.begin (), embedding.end ());
      entry.type = type;
      entry.timestamp_ns = ts;

      const bool durable = (type == MemoryType::CORRECTION);
      if (!impl_->append_payload_record (entry, durable))
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to write memory payload record"));
        }

      yantra::WalRecord wrec{};
      wrec.run_id_num = id;
      wrec.step = static_cast<uint64_t> (type);
      wrec.timestamp_ns = ts;
      wrec.scalar_value = static_cast<double> (text.size ());
      wrec.key_id = static_cast<uint32_t> (type);
      wrec.value_kind = static_cast<uint8_t> (yantra::WalValueKind::Scalar);
      wrec.flags = 0U;
      wrec.payload_aux = 0U;

      auto wres = impl_->wal.append_record (wrec, durable);
      if (!wres)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to append WAL record in MemoryStore::store"));
        }

      impl_->entries.push_back (std::move (entry));

      yantra::VectorMetadata meta;
      meta.id = id;
      meta.timestamp = ts;
      meta.category = static_cast<int64_t> (type);
      if (impl_->hnsw != nullptr)
        {
          auto add_res = impl_->hnsw->add (embedding, id, meta);
          if (!add_res)
            {
              impl_->rebuild_hnsw ();
            }
          else
            {
              (void)impl_->hnsw->save (impl_->hnsw_path);
            }
        }
      return id;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in MemoryStore::store"));
    }
}

auto
MemoryStore::retrieve (std::span<const float> query_embedding,
                       std::size_t top_k, MemoryType filter,
                       uint64_t now_timestamp_ns) const noexcept
    -> sutra::Result<std::vector<MemoryEntry>>
{
  if (impl_ == nullptr || !impl_->is_open)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "MemoryStore is not open"));
    }
  if (query_embedding.size () != impl_->dimension)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "Query embedding dimension mismatch in MemoryStore::retrieve"));
    }

  try
    {
      const std::scoped_lock lock (impl_->mtx);
      if (impl_->entries.empty () || top_k == 0U)
        {
          return std::vector<MemoryEntry>{};
        }

      if (impl_->hnsw != nullptr && impl_->hnsw->size () > 0U)
        {
          const std::size_t search_k = std::min (
              impl_->hnsw->size (), std::max<std::size_t> (top_k * 4U, 16U));
          (void)impl_->hnsw->search (query_embedding, search_k);
        }

      const uint64_t now_ns
          = (now_timestamp_ns > 0U) ? now_timestamp_ns : current_epoch_ns ();

      std::vector<MemoryEntry> candidates;
      candidates.reserve (impl_->entries.size ());

      for (const MemoryEntry &e : impl_->entries)
        {
          if (e.superseded_by.has_value ())
            {
              continue;
            }
          if (filter != MemoryType::ANY && e.type != filter)
            {
              continue;
            }
          const float sim
              = EmbeddingClient::cosine_similarity (query_embedding,
                                                    e.embedding);
          float eff = sim;
          if (e.type == MemoryType::EPISODIC)
            {
              const double age_ns
                  = (now_ns > e.timestamp_ns)
                        ? static_cast<double> (now_ns - e.timestamp_ns)
                        : 0.0;
              const double age_days = age_ns / (86400.0 * 1.0e9);
              const double decay
                  = std::exp (-age_days / kEpisodicHalfLifeDays);
              eff = static_cast<float> (static_cast<double> (sim) * decay);
            }
          else if (e.type == MemoryType::CORRECTION)
            {
              if (sim >= 0.55F)
                {
                  eff = std::min (1.0F, sim * 1.2F);
                }
              else
                {
                  eff = std::min (sim, kDefaultMemoryThreshold * 0.9F);
                }
            }
          MemoryEntry copy = e;
          copy.similarity = sim;
          copy.effective_score = eff;
          candidates.push_back (std::move (copy));
        }

      std::ranges::sort (candidates,
                         [] (const MemoryEntry &a, const MemoryEntry &b) {
                           if (std::abs (a.effective_score - b.effective_score)
                               > 1.0e-6F)
                             {
                               return a.effective_score > b.effective_score;
                             }
                           return a.id > b.id;
                         });

      if (candidates.size () > top_k)
        {
          candidates.resize (top_k);
        }
      return candidates;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in MemoryStore::retrieve"));
    }
}

auto
MemoryStore::supersede (MemoryId id, MemoryId by) noexcept
    -> sutra::Result<void>
{
  if (impl_ == nullptr || !impl_->is_open)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "MemoryStore is not open"));
    }

  try
    {
      const std::scoped_lock lock (impl_->mtx);
      bool found = false;
      for (MemoryEntry &e : impl_->entries)
        {
          if (e.id == id)
            {
              e.superseded_by = by;
              found = true;
              break;
            }
        }
      if (!found)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "MemoryId not found in MemoryStore::supersede"));
        }

      yantra::WalRecord wrec{};
      wrec.run_id_num = id;
      wrec.step = by;
      wrec.timestamp_ns = current_epoch_ns ();
      wrec.scalar_value = 0.0;
      wrec.key_id = 0xFFU;
      wrec.value_kind = static_cast<uint8_t> (yantra::WalValueKind::Scalar);
      wrec.flags = 1U;
      wrec.payload_aux = by;

      auto wres = impl_->wal.append_record (wrec, true);
      if (!wres)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to persist supersede record to WAL"));
        }
      return {};
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in MemoryStore::supersede"));
    }
}

auto
MemoryStore::close () noexcept -> sutra::Result<void>
{
  if (impl_ == nullptr || !impl_->is_open)
    {
      return {};
    }
  const std::scoped_lock lock (impl_->mtx);
  (void)impl_->wal.sync_durable ();
  if (impl_->hnsw != nullptr && impl_->hnsw->size () > 0U)
    {
      (void)impl_->hnsw->save (impl_->hnsw_path);
    }
  impl_->is_open = false;
  return {};
}

auto
MemoryStore::wipe_directory (const std::filesystem::path &memory_dir) noexcept
    -> sutra::Result<void>
{
  try
    {
      std::error_code ec;
      if (!std::filesystem::exists (memory_dir, ec))
        {
          return {};
        }
      (void)std::filesystem::remove (memory_dir / "sphinx_memory.wal", ec);
      (void)std::filesystem::remove (memory_dir / "sphinx_memory.dat", ec);
      (void)std::filesystem::remove (memory_dir / "sphinx_memory_hnsw", ec);
      return {};
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in MemoryStore::wipe_directory"));
    }
}

auto
MemoryStore::clear () noexcept -> sutra::Result<void>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "MemoryStore is not initialized"));
    }
  try
    {
      const std::scoped_lock lock (impl_->mtx);
      impl_->entries.clear ();
      impl_->next_id = 1U;
      impl_->wal = yantra::WalEngine{};

      std::error_code ec;
      std::filesystem::create_directories (impl_->memory_dir, ec);
      (void)std::filesystem::remove (impl_->wal_path, ec);
      (void)std::filesystem::remove (impl_->dat_path, ec);
      (void)std::filesystem::remove (impl_->hnsw_path, ec);

      auto wopen = yantra::WalEngine::open (impl_->wal_path, true);
      if (!wopen)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to recreate sphinx_memory.wal in MemoryStore::clear"));
        }
      impl_->wal = std::move (*wopen);
      impl_->rebuild_hnsw ();
      impl_->is_open = true;
      return {};
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in MemoryStore::clear"));
    }
}

auto
MemoryStore::size () const noexcept -> std::size_t
{
  if (impl_ == nullptr)
    {
      return 0U;
    }
  const std::scoped_lock lock (impl_->mtx);
  return impl_->entries.size ();
}

auto
MemoryStore::all_entries () const -> std::vector<MemoryEntry>
{
  if (impl_ == nullptr)
    {
      return {};
    }
  const std::scoped_lock lock (impl_->mtx);
  return impl_->entries;
}

/* ========================================================================= */
/* 3. Memory Extraction & Formatting Helpers                                 */
/* ========================================================================= */

auto
extract_explicit_semantic_facts (std::string_view user_text)
    -> std::vector<std::string>
{
  constexpr std::array<std::string_view, 5U> kTriggers = { {
      "remember that ",
      "remember: ",
      "note that ",
      "keep in mind that ",
      "remember ",
  } };

  const std::string_view trimmed = trim_ascii (user_text);
  std::string_view body;
  for (const std::string_view trig : kTriggers)
    {
      if (starts_with_icase (trimmed, trig))
        {
          body = trim_ascii (trimmed.substr (trig.size ()));
          break;
        }
    }
  if (body.empty ())
    {
      return {};
    }

  std::vector<std::string> facts;
  std::size_t start = 0U;
  for (std::size_t i = 0U; i < body.size (); ++i)
    {
      std::size_t delim_len = 0U;
      if (body[i] == '.' || body[i] == ';')
        {
          delim_len = 1U;
        }
      else
        {
          const std::string_view rem = body.substr (i);
          if (starts_with_icase (rem, " and my ")
              || starts_with_icase (rem, " and i ")
              || starts_with_icase (rem, " and the ")
              || starts_with_icase (rem, " and our "))
            {
              /* Consume " and ", leave subject for next clause.  */
              delim_len = 5U;
            }
        }
      if (delim_len > 0U)
        {
          const std::string_view clause
              = trim_ascii (body.substr (start, i - start));
          if (!clause.empty ())
            {
              std::string norm = normalize_fact_sentence (clause);
              if (!norm.empty ())
                {
                  facts.push_back (std::move (norm));
                }
            }
          start = i + delim_len;
          i = start > 0U ? (start - 1U) : 0U;
        }
    }
  if (start < body.size ())
    {
      const std::string_view tail = trim_ascii (body.substr (start));
      if (!tail.empty ())
        {
          std::string norm = normalize_fact_sentence (tail);
          if (!norm.empty ())
            {
              facts.push_back (std::move (norm));
            }
        }
    }
  return facts;
}

auto
is_clear_memory_command (std::string_view user_text) noexcept -> bool
{
  std::string_view t = trim_ascii (user_text);
  while (!t.empty ()
         && (t.back () == '.' || t.back () == '!' || t.back () == '?'))
    {
      t.remove_suffix (1U);
    }
  t = trim_ascii (t);
  if (t.empty ())
    {
      return false;
    }

  constexpr std::array<std::string_view, 7U> kExactCommands = { {
      "/clear-memory",
      "/erase-memory",
      "/reset-memory",
      "clear memory",
      "erase memory",
      "reset memory",
      "wipe memory",
  } };
  for (const std::string_view cmd : kExactCommands)
    {
      if (t.size () == cmd.size () && starts_with_icase (t, cmd))
        {
          return true;
        }
    }

  constexpr std::array<std::string_view, 7U> kNaturalPhrases = { {
      "erase your memory",
      "clear your memory",
      "wipe your memory",
      "reset your memory",
      "erase all memory",
      "clear all memory",
      "forget everything",
  } };
  for (const std::string_view phrase : kNaturalPhrases)
    {
      if (contains_icase (t, phrase))
        {
          return true;
        }
    }
  return false;
}

auto
extract_explicit_correction (std::string_view user_text)
    -> std::optional<std::string>
{
  const std::string_view trimmed = trim_ascii (user_text);
  if (trimmed.empty () || is_clear_memory_command (trimmed))
    {
      return std::nullopt;
    }

  /* Do not classify barge-in task cancellation ("Actually, just tell me...")
     as a factual memory correction.  */
  if (starts_with_icase (trimmed, "actually, just "))
    {
      return std::nullopt;
    }

  std::string_view body;
  if (starts_with_icase (trimmed, "no, that's wrong"))
    {
      body = trim_ascii (trimmed.substr (16U));
      if (!body.empty () && (body.front () == '.' || body.front () == ','))
        {
          body.remove_prefix (1U);
          body = trim_ascii (body);
        }
      if (body.empty ())
        {
          return std::string ("The previous statement was incorrect.");
        }
    }
  else if (starts_with_icase (trimmed, "no, "))
    {
      body = trim_ascii (trimmed.substr (4U));
    }
  else if (starts_with_icase (trimmed, "actually, "))
    {
      body = trim_ascii (trimmed.substr (10U));
    }
  else if (starts_with_icase (trimmed, "i said ")
           && contains_icase (trimmed, " not "))
    {
      body = trim_ascii (trimmed.substr (7U));
    }
  else
    {
      return std::nullopt;
    }

  /* Reject conversational task clarifications, meta-statements, and questions
     (e.g., "No, I am going to tell you what I know." or "No, you
     misunderstood... I want to know in what directory example.cpp is") so
     they route through TurnRouter instead of short-circuiting as a stored
     CORRECTION.  */
  constexpr std::array<std::string_view, 42U> kNonCorrectionPhrases = { {
      "who are you",
      "what are you",
      "misunderstood",
      "missunderstood",
      "i want ",
      "i asked",
      "i meant",
      "i need ",
      "i would like",
      "i am going",
      "i'm going",
      "i will ",
      "i'll ",
      "let me ",
      "let's ",
      "tell you",
      "what i know",
      "explain",
      "talking about",
      "asking ",
      "listen",
      "wait",
      "hold on",
      "never mind",
      "nevermind",
      "forget it",
      "where is ",
      "what directory",
      "which directory",
      "located",
      "can you ",
      "could you ",
      "would you ",
      "please ",
      "compile",
      "build ",
      "run ",
      "execute ",
      "create ",
      "write ",
      "delete ",
      "remove ",
  } };
  if (trimmed.back () == '?')
    {
      return std::nullopt;
    }
  for (const std::string_view bad : kNonCorrectionPhrases)
    {
      if (contains_icase (body, bad))
        {
          return std::nullopt;
        }
    }

  constexpr std::array<std::string_view, 14U> kFactVerbs = { {
      " is ",
      " are ",
      " was ",
      " were ",
      " belongs",
      " belong ",
      " has ",
      " have ",
      " called ",
      " named ",
      " means ",
      " refers ",
      " stands for ",
      " not ",
  } };
  bool has_fact_verb = false;
  for (const std::string_view fv : kFactVerbs)
    {
      if (contains_icase (body, fv))
        {
          has_fact_verb = true;
          break;
        }
    }
  if (!has_fact_verb)
    {
      return std::nullopt;
    }

  std::string norm = normalize_fact_sentence (body);
  if (norm.empty ())
    {
      return std::nullopt;
    }
  return norm;
}

auto
is_trivial_greeting_turn (std::string_view user_text) noexcept -> bool
{
  const std::string_view t = trim_ascii (user_text);
  if (t.empty () || is_clear_memory_command (t))
    {
      return true;
    }
  constexpr std::array<std::string_view, 8U> kSubstantiveHints = { {
      "remember", "project", "c++",     "ssh",
      "file",     "code",    "compile", "directory",
  } };
  for (const std::string_view h : kSubstantiveHints)
    {
      if (contains_icase (t, h))
        {
          return false;
        }
    }
  constexpr std::array<std::string_view, 10U> kGreetings = { {
      "hello",
      "hi",
      "hey",
      "good morning",
      "good afternoon",
      "good evening",
      "how are you",
      "thank you",
      "thanks",
      "bye",
  } };
  for (const std::string_view g : kGreetings)
    {
      if (starts_with_icase (t, g) || contains_icase (t, g))
        {
          return true;
        }
    }
  return false;
}

auto
format_memories_for_prompt (std::span<const MemoryEntry> memories,
                            float threshold, uint64_t now_timestamp_ns)
    -> std::string
{
  const uint64_t now_ns
      = (now_timestamp_ns > 0U) ? now_timestamp_ns : current_epoch_ns ();
  std::ostringstream oss;
  std::size_t injected = 0U;

  for (const MemoryEntry &m : memories)
    {
      if (m.superseded_by.has_value () || m.effective_score <= threshold)
        {
          continue;
        }
      if (injected == 0U)
        {
          oss << "=== PERSISTENT USER MEMORIES & CORRECTIONS (HIGHEST "
                 "PRIORITY) ===\n";
        }
      if (m.type == MemoryType::EPISODIC)
        {
          const double age_ns
              = (now_ns > m.timestamp_ns)
                    ? static_cast<double> (now_ns - m.timestamp_ns)
                    : 0.0;
          const int days
              = static_cast<int> (std::round (age_ns / (86400.0 * 1.0e9)));
          oss << "- [EPISODIC, " << days << " days ago] " << m.text << "\n";
        }
      else
        {
          oss << "- [" << memory_type_to_string (m.type) << "] " << m.text
              << "\n";
        }
      ++injected;
    }
  if (injected > 0U)
    {
      oss << "Mandatory: You MUST prioritize the facts in PERSISTENT USER "
             "MEMORIES & CORRECTIONS over your pre-training weights.\n"
             "================================================================"
             "=\n";
    }
  return oss.str ();
}

} /* namespace sphinx::inline v1 */
