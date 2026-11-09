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

#include <obsidian/embeddings.hpp>
#include <obsidian/simd.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace obsidian
{

namespace
{

constexpr char k_obvec_magic[8] = {'O', 'B', 'V', 'E', 'C', '0', '0', '1'};

struct Obvec_Header
{
  char magic[8];
  uint64_t num_words = 0;
  uint64_t dim = 0;
  uint64_t strings_offset = 0;
  uint64_t strings_bytes = 0;
  uint64_t floats_offset = 0;
  uint64_t reserved[2] = {0, 0};
};

static_assert (sizeof (Obvec_Header) == 64,
               "Obvec_Header must be 64-byte cache-line aligned");

float
fast_sigmoid (float x) noexcept
{
  if (x > 6.0f)
    {
      return 1.0f;
    }
  if (x < -6.0f)
    {
      return 0.0f;
    }
  return 1.0f / (1.0f + std::exp (-x));
}

} /* namespace */

/* -------------------------------------------------------------------------
   Mmap_Embeddings implementation
   ------------------------------------------------------------------------- */

Mmap_Embeddings::~Mmap_Embeddings (void) noexcept
{
  if (mmap_addr_ != nullptr && mmap_size_ > 0)
    {
      munmap (mmap_addr_, mmap_size_);
      mmap_addr_ = nullptr;
      mmap_size_ = 0;
      mmap_floats_ = nullptr;
    }
}

Mmap_Embeddings::Mmap_Embeddings (Mmap_Embeddings &&other) noexcept
  : vocab_ (std::move (other.vocab_)),
    dim_ (other.dim_),
    owned_floats_ (std::move (other.owned_floats_)),
    mmap_addr_ (other.mmap_addr_),
    mmap_size_ (other.mmap_size_),
    mmap_floats_ (other.mmap_floats_)
{
  other.dim_ = 0;
  other.mmap_addr_ = nullptr;
  other.mmap_size_ = 0;
  other.mmap_floats_ = nullptr;
}

Mmap_Embeddings &
Mmap_Embeddings::operator= (Mmap_Embeddings &&other) noexcept
{
  if (this != &other)
    {
      if (mmap_addr_ != nullptr && mmap_size_ > 0)
        {
          munmap (mmap_addr_, mmap_size_);
        }
      vocab_ = std::move (other.vocab_);
      dim_ = other.dim_;
      owned_floats_ = std::move (other.owned_floats_);
      mmap_addr_ = other.mmap_addr_;
      mmap_size_ = other.mmap_size_;
      mmap_floats_ = other.mmap_floats_;

      other.dim_ = 0;
      other.mmap_addr_ = nullptr;
      other.mmap_size_ = 0;
      other.mmap_floats_ = nullptr;
    }
  return *this;
}

Result<Mmap_Embeddings>
Mmap_Embeddings::from_memory (std::span<const std::string> words,
                              std::span<const float> matrix,
                              size_t dim)
{
  if (words.empty () || dim == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "from_memory requires non-empty words and dim");
    }
  if (matrix.size () != words.size () * dim)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "matrix size does not match words.size() * dim");
    }

  Mmap_Embeddings store;
  store.dim_ = dim;
  store.owned_floats_.assign (matrix.begin (), matrix.end ());
  for (const auto &w : words)
    {
      store.vocab_.add_token (w);
    }
  return store;
}

Result<void>
Mmap_Embeddings::save_obvec_mmap (const std::string &path) const
{
  if (vocab_.size () == 0 || dim_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "cannot save empty Mmap_Embeddings");
    }

  std::string str_blob;
  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      std::string_view w = vocab_.id_to_token (
        static_cast<int32_t> (i)).value ();
      uint32_t len = static_cast<uint32_t> (w.size ());
      const char *len_bytes = reinterpret_cast<const char *> (&len);
      str_blob.append (len_bytes, sizeof (uint32_t));
      str_blob.append (w.data (), w.size ());
    }

  size_t strings_off = sizeof (Obvec_Header);
  size_t raw_after_strings = strings_off + str_blob.size ();
  /* Align float matrix start to 64-byte cache line boundary.  */
  size_t floats_off = (raw_after_strings + 63ULL) & ~63ULL;
  size_t pad_bytes = floats_off - raw_after_strings;

  Obvec_Header hdr;
  std::memcpy (hdr.magic, k_obvec_magic, 8);
  hdr.num_words = vocab_.size ();
  hdr.dim = dim_;
  hdr.strings_offset = strings_off;
  hdr.strings_bytes = str_blob.size ();
  hdr.floats_offset = floats_off;

  std::ofstream ofs (path, std::ios::binary);
  if (!ofs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open .obvec file for writing");
    }

  ofs.write (reinterpret_cast<const char *> (&hdr), sizeof (hdr));
  ofs.write (str_blob.data (),
             static_cast<std::streamsize> (str_blob.size ()));
  if (pad_bytes > 0)
    {
      std::vector<char> zeros (pad_bytes, 0);
      ofs.write (zeros.data (), static_cast<std::streamsize> (pad_bytes));
    }

  const float *mat = raw_matrix_ptr ();
  size_t float_bytes = vocab_.size () * dim_ * sizeof (float);
  ofs.write (reinterpret_cast<const char *> (mat),
             static_cast<std::streamsize> (float_bytes));
  if (!ofs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed while writing .obvec payload");
    }
  return {};
}

Result<Mmap_Embeddings>
Mmap_Embeddings::open_mmap (const std::string &path)
{
  int fd = open (path.c_str (), O_RDONLY);
  if (fd < 0)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "open_mmap failed to open file: " + path);
    }

  struct stat st;
  off_t min_hdr = static_cast<off_t> (sizeof (Obvec_Header));
  if (fstat (fd, &st) != 0 || st.st_size < min_hdr)
    {
      close (fd);
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "file too small for .obvec header");
    }

  size_t file_size = static_cast<size_t> (st.st_size);
  void *addr = mmap (nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close (fd);
  if (addr == MAP_FAILED)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "mmap(2) failed on file: " + path);
    }

  const auto *hdr = static_cast<const Obvec_Header *> (addr);
  if (std::memcmp (hdr->magic, k_obvec_magic, 8) != 0)
    {
      munmap (addr, file_size);
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "invalid .obvec magic header");
    }

  size_t expected_end = hdr->floats_offset
                        + hdr->num_words * hdr->dim * sizeof (float);
  if (expected_end > file_size
      || hdr->strings_offset + hdr->strings_bytes > hdr->floats_offset)
    {
      munmap (addr, file_size);
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "truncated .obvec file payload");
    }

  Mmap_Embeddings store;
  store.dim_ = static_cast<size_t> (hdr->dim);
  store.mmap_addr_ = addr;
  store.mmap_size_ = file_size;

  const char *base = static_cast<const char *> (addr);
  const char *sptr = base + hdr->strings_offset;
  const char *send = sptr + hdr->strings_bytes;

  for (uint64_t i = 0; i < hdr->num_words; ++i)
    {
      if (sptr + sizeof (uint32_t) > send)
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "corrupted string table in .obvec");
        }
      uint32_t len = 0;
      std::memcpy (&len, sptr, sizeof (uint32_t));
      sptr += sizeof (uint32_t);
      if (sptr + len > send)
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "string length overflow in .obvec");
        }
      store.vocab_.add_token (std::string_view (sptr, len));
      sptr += len;
    }

  store.mmap_floats_ = reinterpret_cast<const float *> (
    base + hdr->floats_offset);
  return store;
}

Result<Mmap_Embeddings>
Mmap_Embeddings::load_glove_text (const std::string &path)
{
  std::ifstream ifs (path);
  if (!ifs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open GloVe file: " + path);
    }

  Mmap_Embeddings store;
  std::string line;
  while (std::getline (ifs, line))
    {
      if (line.empty ())
        {
          continue;
        }
      std::istringstream iss (line);
      std::string word;
      if (!(iss >> word))
        {
          continue;
        }
      std::vector<float> vals;
      float v = 0.0f;
      while (iss >> v)
        {
          vals.push_back (v);
        }
      if (vals.empty ())
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "GloVe line has no vector values");
        }
      if (store.dim_ == 0)
        {
          store.dim_ = vals.size ();
        }
      else if (vals.size () != store.dim_)
        {
          return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                                 "inconsistent vector dimension in GloVe");
        }
      store.vocab_.add_token (word);
      store.owned_floats_.insert (store.owned_floats_.end (),
                                  vals.begin (), vals.end ());
    }

  if (store.vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "GloVe file contained zero vectors");
    }
  return store;
}

Result<Mmap_Embeddings>
Mmap_Embeddings::load_word2vec_text (const std::string &path)
{
  std::ifstream ifs (path);
  if (!ifs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open Word2Vec text file: " + path);
    }

  size_t num_words = 0;
  size_t dim = 0;
  if (!(ifs >> num_words >> dim) || num_words == 0 || dim == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "invalid Word2Vec text header");
    }

  Mmap_Embeddings store;
  store.dim_ = dim;
  store.owned_floats_.reserve (num_words * dim);

  for (size_t i = 0; i < num_words; ++i)
    {
      std::string word;
      if (!(ifs >> word))
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "unexpected EOF reading Word2Vec word");
        }
      store.vocab_.add_token (word);
      for (size_t d = 0; d < dim; ++d)
        {
          float val = 0.0f;
          if (!(ifs >> val))
            {
              return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                     "unexpected EOF reading Word2Vec float");
            }
          store.owned_floats_.push_back (val);
        }
    }
  return store;
}

Result<Mmap_Embeddings>
Mmap_Embeddings::load_word2vec_binary (const std::string &path)
{
  std::ifstream ifs (path, std::ios::binary);
  if (!ifs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open Word2Vec binary file: " + path);
    }

  size_t num_words = 0;
  size_t dim = 0;
  if (!(ifs >> num_words >> dim) || num_words == 0 || dim == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                             "invalid Word2Vec binary header");
    }
  /* Consume single newline or whitespace after header.  */
  ifs.get ();

  Mmap_Embeddings store;
  store.dim_ = dim;
  store.owned_floats_.resize (num_words * dim, 0.0f);

  for (size_t i = 0; i < num_words; ++i)
    {
      std::string word;
      int ch = 0;
      while ((ch = ifs.get ()) != EOF && ch != ' ')
        {
          if (ch != '\n')
            {
              word.push_back (static_cast<char> (ch));
            }
        }
      if (word.empty ())
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "empty word in Word2Vec binary file");
        }
      store.vocab_.add_token (word);
      ifs.read (
        reinterpret_cast<char *> (store.owned_floats_.data () + i * dim),
        static_cast<std::streamsize> (dim * sizeof (float)));
      if (!ifs)
        {
          return OBSIDIAN_ERROR (Error_Code::Corrupted_Format,
                                 "truncated binary vector in Word2Vec file");
        }
    }
  return store;
}

Result<void>
Mmap_Embeddings::save_glove_text (const std::string &path) const
{
  if (vocab_.size () == 0 || dim_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "cannot save empty embeddings");
    }
  std::ofstream ofs (path);
  if (!ofs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open GloVe file for writing");
    }
  const float *mat = raw_matrix_ptr ();
  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      ofs << vocab_.id_to_token (static_cast<int32_t> (i)).value ();
      for (size_t d = 0; d < dim_; ++d)
        {
          ofs << " " << mat[i * dim_ + d];
        }
      ofs << "\n";
    }
  return {};
}

Result<void>
Mmap_Embeddings::save_word2vec_binary (const std::string &path) const
{
  if (vocab_.size () == 0 || dim_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "cannot save empty embeddings");
    }
  std::ofstream ofs (path, std::ios::binary);
  if (!ofs)
    {
      return OBSIDIAN_ERROR (Error_Code::IO_Failure,
                             "failed to open Word2Vec binary file");
    }
  ofs << vocab_.size () << " " << dim_ << "\n";
  const float *mat = raw_matrix_ptr ();
  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      std::string_view w = vocab_.id_to_token (
        static_cast<int32_t> (i)).value ();
      ofs.write (w.data (), static_cast<std::streamsize> (w.size ()));
      ofs.put (' ');
      ofs.write (reinterpret_cast<const char *> (mat + i * dim_),
                 static_cast<std::streamsize> (dim_ * sizeof (float)));
      ofs.put ('\n');
    }
  return {};
}

Result<std::span<const float>>
Mmap_Embeddings::lookup (std::string_view word) const noexcept
{
  int32_t id = vocab_.token_to_id (word);
  if (id < 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Vocab_Not_Found,
                             "word not found in embedding store");
    }
  return lookup_by_id (id);
}

Result<std::span<const float>>
Mmap_Embeddings::lookup_by_id (int32_t id) const noexcept
{
  if (id < 0 || static_cast<size_t> (id) >= vocab_.size ())
    {
      return OBSIDIAN_ERROR (Error_Code::Out_Of_Range,
                             "word ID out of embedding store bounds");
    }
  const float *base = raw_matrix_ptr ();
  return std::span<const float> (
    base + static_cast<size_t> (id) * dim_, dim_);
}

Result<float>
Mmap_Embeddings::similarity (std::string_view w1,
                             std::string_view w2) const noexcept
{
  auto v1 = lookup (w1);
  if (!v1.has_value ())
    {
      return std::unexpected (v1.error ());
    }
  auto v2 = lookup (w2);
  if (!v2.has_value ())
    {
      return std::unexpected (v2.error ());
    }
  return simd::cosine_similarity_f32 (*v1, *v2);
}

Result<std::vector<Word_Neighbor>>
Mmap_Embeddings::most_similar_to_vector (
  std::span<const float> query,
  size_t top_k,
  std::span<const std::string_view> exclude) const
{
  if (vocab_.size () == 0 || dim_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "embedding store is empty");
    }
  if (query.size () != dim_)
    {
      return OBSIDIAN_ERROR (Error_Code::Shape_Mismatch,
                             "query vector dimension mismatch");
    }
  if (top_k == 0)
    {
      return std::vector<Word_Neighbor> {};
    }

  std::vector<Word_Neighbor> candidates;
  candidates.reserve (vocab_.size ());
  const float *mat = raw_matrix_ptr ();

  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      std::string_view w = vocab_.id_to_token (
        static_cast<int32_t> (i)).value ();
      bool skip = false;
      for (std::string_view ex : exclude)
        {
          if (w == ex)
            {
              skip = true;
              break;
            }
        }
      if (skip)
        {
          continue;
        }
      std::span<const float> row (mat + i * dim_, dim_);
      float sim = simd::cosine_similarity_f32 (query, row);
      candidates.push_back (
        Word_Neighbor {std::string (w), static_cast<int32_t> (i), sim});
    }

  size_t k = std::min (top_k, candidates.size ());
  std::partial_sort (
    candidates.begin (),
    candidates.begin () + static_cast<ptrdiff_t> (k),
    candidates.end (),
    [] (const Word_Neighbor &lhs, const Word_Neighbor &rhs)
    {
      return lhs.similarity > rhs.similarity;
    });
  candidates.resize (k);
  return candidates;
}

Result<std::vector<Word_Neighbor>>
Mmap_Embeddings::most_similar (std::string_view word, size_t top_k) const
{
  auto vec_res = lookup (word);
  if (!vec_res.has_value ())
    {
      return std::unexpected (vec_res.error ());
    }
  std::string_view excl[1] = {word};
  return most_similar_to_vector (*vec_res, top_k, excl);
}

Result<std::vector<Word_Neighbor>>
Mmap_Embeddings::analogy (std::string_view a,
                          std::string_view a_star,
                          std::string_view b,
                          size_t top_k) const
{
  auto va = lookup (a);
  if (!va.has_value ())
    {
      return std::unexpected (va.error ());
    }
  auto vas = lookup (a_star);
  if (!vas.has_value ())
    {
      return std::unexpected (vas.error ());
    }
  auto vb = lookup (b);
  if (!vb.has_value ())
    {
      return std::unexpected (vb.error ());
    }

  std::vector<float> ua (va->begin (), va->end ());
  std::vector<float> uas (vas->begin (), vas->end ());
  std::vector<float> ub (vb->begin (), vb->end ());
  simd::normalize_inplace_f32 (ua);
  simd::normalize_inplace_f32 (uas);
  simd::normalize_inplace_f32 (ub);

  std::vector<float> target (dim_, 0.0f);
  for (size_t d = 0; d < dim_; ++d)
    {
      target[d] = uas[d] - ua[d] + ub[d];
    }
  simd::normalize_inplace_f32 (target);

  std::string_view excl[3] = {a, a_star, b};
  return most_similar_to_vector (target, top_k, excl);
}

Result<Tensor>
Mmap_Embeddings::to_basalt_tensor (bool requires_grad) const
{
  if (vocab_.size () == 0 || dim_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "embedding store is empty");
    }
  Tensor_Options opts = Tensor_Options ()
    .with_dtype (Dtype::Float32)
    .with_requires_grad (requires_grad);
  return basalt::from_blob (raw_matrix_ptr (), {vocab_.size (), dim_}, opts);
}

Result<Tensor>
Mmap_Embeddings::embed_sequence (std::span<const std::string_view> words) const
{
  if (words.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "embed_sequence requires non-empty words");
    }
  if (dim_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "embedding store is empty");
    }

  Tensor out = basalt::zeros ({words.size (), dim_},
                              Tensor_Options ().with_dtype (Dtype::Float32));
  float *dst = out.data_ptr<float> ();

  for (size_t i = 0; i < words.size (); ++i)
    {
      auto sp = lookup (words[i]);
      if (sp.has_value ())
        {
          std::memcpy (dst + i * dim_, sp->data (), dim_ * sizeof (float));
        }
    }
  return out;
}

/* -------------------------------------------------------------------------
   Word2Vec_Trainer implementation
   ------------------------------------------------------------------------- */

Result<Mmap_Embeddings>
Word2Vec_Trainer::train (std::span<const std::string_view> corpus,
                         const Word2Vec_Config &cfg)
{
  if (corpus.empty () || cfg.dim == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "Word2Vec_Trainer requires non-empty corpus");
    }

  std::unordered_map<std::string, size_t> counts;
  std::vector<std::vector<std::string>> tokenized_docs;
  Arena arena;

  for (std::string_view doc : corpus)
    {
      arena.reset ();
      Pmr_Vector<std::string_view> spans (arena.resource ());
      simd::scan_pretoken_spans (doc, spans, true);
      std::vector<std::string> doc_words;
      doc_words.reserve (spans.size ());
      for (std::string_view w : spans)
        {
          std::string s (w);
          ++counts[s];
          doc_words.push_back (std::move (s));
        }
      tokenized_docs.push_back (std::move (doc_words));
    }

  std::vector<std::string> vocab_words;
  for (const auto &[w, c] : counts)
    {
      if (c >= cfg.min_count)
        {
          vocab_words.push_back (w);
        }
    }
  std::sort (vocab_words.begin (), vocab_words.end ());
  if (vocab_words.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "no words met Word2Vec min_count threshold");
    }

  Vocabulary vocab;
  for (const auto &w : vocab_words)
    {
      vocab.add_token (w);
    }

  const size_t V = vocab.size ();
  const size_t D = cfg.dim;
  std::mt19937_64 rng (cfg.seed);
  float bound = 0.5f / static_cast<float> (D);
  std::uniform_real_distribution<float> uni (-bound, bound);
  std::uniform_int_distribution<size_t> neg_dist (0, V - 1);

  std::vector<float> syn0 (V * D, 0.0f);
  std::vector<float> syn1neg (V * D, 0.0f);
  for (float &v : syn0)
    {
      v = uni (rng);
    }

  std::vector<float> grad_buf (D, 0.0f);
  std::vector<float> cbow_mean (D, 0.0f);

  for (size_t ep = 0; ep < cfg.epochs; ++ep)
    {
      float lr = cfg.learning_rate
                 * (1.0f - static_cast<float> (ep)
                           / static_cast<float> (cfg.epochs + 1));

      for (const auto &doc : tokenized_docs)
        {
          std::vector<int32_t> ids;
          ids.reserve (doc.size ());
          for (const auto &w : doc)
            {
              int32_t id = vocab.token_to_id (w);
              if (id >= 0)
                {
                  ids.push_back (id);
                }
            }

          for (size_t pos = 0; pos < ids.size (); ++pos)
            {
              size_t center = static_cast<size_t> (ids[pos]);
              size_t start = (pos > cfg.window_size)
                               ? (pos - cfg.window_size) : 0;
              size_t end = std::min (ids.size (),
                                     pos + cfg.window_size + 1);

              if (cfg.skip_gram)
                {
                  for (size_t ctx_pos = start; ctx_pos < end; ++ctx_pos)
                    {
                      if (ctx_pos == pos)
                        {
                          continue;
                        }
                      size_t ctx = static_cast<size_t> (ids[ctx_pos]);
                      std::fill (grad_buf.begin (), grad_buf.end (), 0.0f);
                      std::span<float> v_in (syn0.data () + ctx * D, D);

                      for (size_t ns = 0; ns <= cfg.negative_samples; ++ns)
                        {
                          size_t target = (ns == 0) ? center : neg_dist (rng);
                          if (ns > 0 && target == center)
                            {
                              continue;
                            }
                          float label = (ns == 0) ? 1.0f : 0.0f;
                          std::span<float> v_out (
                            syn1neg.data () + target * D, D);
                          float dot = simd::dot_f32 (v_in, v_out);
                          float g = (label - fast_sigmoid (dot)) * lr;
                          simd::axpy_f32 (g, v_out, grad_buf);
                          simd::axpy_f32 (g, v_in, v_out);
                        }
                      simd::axpy_f32 (1.0f, grad_buf, v_in);
                    }
                }
              else
                {
                  std::fill (cbow_mean.begin (), cbow_mean.end (), 0.0f);
                  size_t cw = 0;
                  for (size_t ctx_pos = start; ctx_pos < end; ++ctx_pos)
                    {
                      if (ctx_pos == pos)
                        {
                          continue;
                        }
                      size_t ctx = static_cast<size_t> (ids[ctx_pos]);
                      simd::axpy_f32 (
                        1.0f,
                        std::span<const float> (syn0.data () + ctx * D, D),
                        cbow_mean);
                      ++cw;
                    }
                  if (cw == 0)
                    {
                      continue;
                    }
                  float inv_cw = 1.0f / static_cast<float> (cw);
                  for (float &v : cbow_mean)
                    {
                      v *= inv_cw;
                    }

                  std::fill (grad_buf.begin (), grad_buf.end (), 0.0f);
                  for (size_t ns = 0; ns <= cfg.negative_samples; ++ns)
                    {
                      size_t target = (ns == 0) ? center : neg_dist (rng);
                      if (ns > 0 && target == center)
                        {
                          continue;
                        }
                      float label = (ns == 0) ? 1.0f : 0.0f;
                      std::span<float> v_out (
                        syn1neg.data () + target * D, D);
                      float dot = simd::dot_f32 (cbow_mean, v_out);
                      float g = (label - fast_sigmoid (dot)) * lr;
                      simd::axpy_f32 (g, v_out, grad_buf);
                      simd::axpy_f32 (g, cbow_mean, v_out);
                    }
                  for (size_t ctx_pos = start; ctx_pos < end; ++ctx_pos)
                    {
                      if (ctx_pos == pos)
                        {
                          continue;
                        }
                      size_t ctx = static_cast<size_t> (ids[ctx_pos]);
                      simd::axpy_f32 (
                        inv_cw, grad_buf,
                        std::span<float> (syn0.data () + ctx * D, D));
                    }
                }
            }
        }
    }

  return Mmap_Embeddings::from_memory (vocab_words, syn0, D);
}

/* -------------------------------------------------------------------------
   FastText_Embeddings implementation
   ------------------------------------------------------------------------- */

FastText_Embeddings::FastText_Embeddings (size_t dim,
                                          size_t num_buckets,
                                          size_t min_n,
                                          size_t max_n,
                                          uint64_t seed)
  : dim_ (dim),
    num_buckets_ (num_buckets),
    min_n_ (min_n),
    max_n_ (max_n),
    bucket_vectors_ (num_buckets * dim, 0.0f)
{
  std::mt19937_64 rng (seed);
  float bound = 1.0f / static_cast<float> (dim);
  std::uniform_real_distribution<float> uni (-bound, bound);
  for (float &v : bucket_vectors_)
    {
      v = uni (rng);
    }
}

size_t
FastText_Embeddings::ngram_bucket (std::string_view ngram) const noexcept
{
  uint32_t h = 2166136261U;
  for (unsigned char c : ngram)
    {
      h ^= static_cast<uint32_t> (c);
      h *= 16777619U;
    }
  return static_cast<size_t> (h) % num_buckets_;
}

Result<void>
FastText_Embeddings::train (std::span<const std::string_view> corpus,
                            size_t epochs,
                            float lr)
{
  if (corpus.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "FastText_Embeddings::train empty corpus");
    }

  Word2Vec_Config cfg;
  cfg.dim = dim_;
  cfg.epochs = epochs;
  cfg.learning_rate = lr;
  auto w2v_res = Word2Vec_Trainer::train (corpus, cfg);
  if (!w2v_res.has_value ())
    {
      return std::unexpected (w2v_res.error ());
    }

  const Mmap_Embeddings &trained = *w2v_res;
  word_vectors_.assign (trained.vocab_size () * dim_, 0.0f);

  for (size_t i = 0; i < trained.vocab_size (); ++i)
    {
      std::string_view w = trained.id_to_word (
        static_cast<int32_t> (i)).value ();
      known_words_.add_token (w);
      auto sp = trained.lookup_by_id (static_cast<int32_t> (i)).value ();
      std::copy (sp.begin (), sp.end (), word_vectors_.begin () + i * dim_);

      /* Push word semantics into its constituent character n-gram buckets.  */
      std::string bracketed = "<" + std::string (w) + ">";
      for (size_t n = min_n_; n <= max_n_; ++n)
        {
          if (bracketed.size () < n)
            {
              continue;
            }
          for (size_t pos = 0; pos + n <= bracketed.size (); ++pos)
            {
              size_t b = ngram_bucket (
                std::string_view (bracketed.data () + pos, n));
              simd::axpy_f32 (
                0.25f, sp,
                std::span<float> (bucket_vectors_.data () + b * dim_, dim_));
            }
        }
    }
  return {};
}

void
FastText_Embeddings::compute_word_vector (
  std::string_view word,
  std::span<float> out_vec) const noexcept
{
  OBSIDIAN_ASSERT (out_vec.size () == dim_);
  std::fill (out_vec.begin (), out_vec.end (), 0.0f);
  size_t count = 0;

  int32_t id = known_words_.token_to_id (word);
  if (id >= 0)
    {
      std::span<const float> wv (
        word_vectors_.data () + static_cast<size_t> (id) * dim_, dim_);
      simd::axpy_f32 (1.0f, wv, out_vec);
      ++count;
    }

  std::string bracketed = "<" + std::string (word) + ">";
  for (size_t n = min_n_; n <= max_n_; ++n)
    {
      if (bracketed.size () < n)
        {
          continue;
        }
      for (size_t pos = 0; pos + n <= bracketed.size (); ++pos)
        {
          size_t b = ngram_bucket (
            std::string_view (bracketed.data () + pos, n));
          std::span<const float> bv (
            bucket_vectors_.data () + b * dim_, dim_);
          simd::axpy_f32 (1.0f, bv, out_vec);
          ++count;
        }
    }

  if (count > 0)
    {
      simd::normalize_inplace_f32 (out_vec);
    }
}

} /* namespace obsidian */
