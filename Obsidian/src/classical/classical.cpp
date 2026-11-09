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

#include <obsidian/classical.hpp>
#include <obsidian/simd.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace obsidian
{

namespace
{

std::string
encode_history_key (std::span<const int32_t> ctx)
{
  std::string key;
  key.resize (ctx.size () * sizeof (int32_t));
  if (!ctx.empty ())
    {
      std::memcpy (key.data (), ctx.data (),
                   ctx.size () * sizeof (int32_t));
    }
  return key;
}

std::string
lowercase_word (std::string_view w)
{
  std::string out (w.size (), '\0');
  simd::ascii_to_lower (w, out.data ());
  return out;
}

} /* namespace */

/* -------------------------------------------------------------------------
   Damerau-Levenshtein edit distance
   ------------------------------------------------------------------------- */

size_t
damerau_levenshtein_distance (std::string_view a, std::string_view b)
{
  const size_t m = a.size ();
  const size_t n = b.size ();
  if (m == 0)
    {
      return n;
    }
  if (n == 0)
    {
      return m;
    }

  std::vector<size_t> dp ((m + 1) * (n + 1), 0);
  const size_t stride = n + 1;

  for (size_t i = 0; i <= m; ++i)
    {
      dp[i * stride] = i;
    }
  for (size_t j = 0; j <= n; ++j)
    {
      dp[j] = j;
    }

  for (size_t i = 1; i <= m; ++i)
    {
      for (size_t j = 1; j <= n; ++j)
        {
          size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
          size_t best = std::min ({
            dp[(i - 1) * stride + j] + 1,
            dp[i * stride + (j - 1)] + 1,
            dp[(i - 1) * stride + (j - 1)] + cost
          });
          if (i > 1 && j > 1
              && a[i - 1] == b[j - 2]
              && a[i - 2] == b[j - 1])
            {
              best = std::min (
                best, dp[(i - 2) * stride + (j - 2)] + 1);
            }
          dp[i * stride + j] = best;
        }
    }
  return dp[m * stride + n];
}

/* -------------------------------------------------------------------------
   Sentence_Segmenter implementation
   ------------------------------------------------------------------------- */

Sentence_Segmenter::Sentence_Segmenter (void)
{
  const char *defaults[] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "inc", "ltd", "co", "corp", "fig", "eq", "vol", "no", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "e.g",
    "i.e", "u.s", "ph.d", "a.m", "p.m"
  };
  for (const char *ab : defaults)
    {
      add_abbreviation (ab);
    }
}

void
Sentence_Segmenter::add_abbreviation (std::string_view abbr)
{
  if (!abbr.empty ())
    {
      abbrevs_.add_token (lowercase_word (abbr));
    }
}

std::vector<std::string_view>
Sentence_Segmenter::segment (std::string_view text) const
{
  std::vector<std::string_view> sentences;
  const size_t n = text.size ();
  size_t sent_start = 0;

  while (sent_start < n
         && (text[sent_start] == ' ' || text[sent_start] == '\t'
             || text[sent_start] == '\n' || text[sent_start] == '\r'))
    {
      ++sent_start;
    }

  for (size_t i = sent_start; i < n; ++i)
    {
      char c = text[i];
      if (c == '.' || c == '!' || c == '?')
        {
          if (c == '.')
            {
              /* Extract preceding token to check abbreviation or decimal.  */
              size_t w_end = i;
              size_t w_start = w_end;
              while (w_start > sent_start
                     && text[w_start - 1] != ' '
                     && text[w_start - 1] != '\t'
                     && text[w_start - 1] != '\n')
                {
                  --w_start;
                }
              std::string_view prev_word = text.substr (
                w_start, w_end - w_start);
              if (!prev_word.empty ())
                {
                  std::string low = lowercase_word (prev_word);
                  if (abbrevs_.contains (low) || prev_word.size () == 1)
                    {
                      continue;
                    }
                }
              /* Decimal number check (e.g., 3.14).  */
              if (i + 1 < n && text[i + 1] >= '0' && text[i + 1] <= '9')
                {
                  continue;
                }
            }

          size_t end_pos = i + 1;
          while (end_pos < n
                 && (text[end_pos] == '"' || text[end_pos] == '\''
                     || text[end_pos] == ')'))
            {
              ++end_pos;
            }

          if (end_pos == n || text[end_pos] == ' '
              || text[end_pos] == '\n' || text[end_pos] == '\r'
              || text[end_pos] == '\t')
            {
              sentences.push_back (
                text.substr (sent_start, end_pos - sent_start));
              sent_start = end_pos;
              while (sent_start < n
                     && (text[sent_start] == ' '
                         || text[sent_start] == '\t'
                         || text[sent_start] == '\n'
                         || text[sent_start] == '\r'))
                {
                  ++sent_start;
                }
              i = (sent_start > 0) ? (sent_start - 1) : 0;
            }
        }
    }

  if (sent_start < n)
    {
      size_t end_trim = n;
      while (end_trim > sent_start
             && (text[end_trim - 1] == ' ' || text[end_trim - 1] == '\n'
                 || text[end_trim - 1] == '\r'
                 || text[end_trim - 1] == '\t'))
        {
          --end_trim;
        }
      if (end_trim > sent_start)
        {
          sentences.push_back (
            text.substr (sent_start, end_trim - sent_start));
        }
    }
  return sentences;
}

/* -------------------------------------------------------------------------
   NGram_Model implementation
   ------------------------------------------------------------------------- */

NGram_Model::NGram_Model (size_t order,
                          Smoothing_Method smoothing,
                          double discount_or_alpha)
  : order_ (std::clamp (order, size_t (1), size_t (5))),
    smoothing_ (smoothing),
    param_ (discount_or_alpha)
{
  bos_id_ = vocab_.add_token ("<bos>");
  eos_id_ = vocab_.add_token ("<eos>");
  unk_id_ = vocab_.add_token ("<unk>");
}

Result<void>
NGram_Model::fit (std::span<const std::string_view> corpus)
{
  if (corpus.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "NGram_Model::fit requires non-empty corpus");
    }

  ngram_counts_.clear ();
  context_totals_.clear ();
  total_unigrams_ = 0;
  total_unique_bigrams_ = 0;

  Arena arena;
  std::vector<std::vector<int32_t>> encoded_docs;
  encoded_docs.reserve (corpus.size ());

  for (std::string_view doc : corpus)
    {
      arena.reset ();
      Pmr_Vector<std::string_view> spans (arena.resource ());
      simd::scan_pretoken_spans (doc, spans, true);

      std::vector<int32_t> seq;
      for (size_t k = 1; k < order_; ++k)
        {
          seq.push_back (bos_id_);
        }
      for (std::string_view w : spans)
        {
          seq.push_back (vocab_.add_token (lowercase_word (w)));
        }
      seq.push_back (eos_id_);
      encoded_docs.push_back (std::move (seq));
    }

  unigram_counts_.assign (vocab_.size (), 0);
  continuation_counts_.assign (vocab_.size (), 0);

  for (const auto &seq : encoded_docs)
    {
      size_t start_pos = (order_ > 1) ? (order_ - 1) : 0;
      for (size_t i = start_pos; i < seq.size (); ++i)
        {
          int32_t w_id = seq[i];
          ++unigram_counts_[static_cast<size_t> (w_id)];
          ++total_unigrams_;

          for (size_t ctx_len = 1; ctx_len < order_; ++ctx_len)
            {
              if (i < ctx_len)
                {
                  break;
                }
              std::span<const int32_t> ctx (
                seq.data () + (i - ctx_len), ctx_len);
              std::string hkey = encode_history_key (ctx);
              auto & followers = ngram_counts_[hkey];
              bool is_new_pair = (followers.find (w_id) == followers.end ());
              ++followers[w_id];
              ++context_totals_[hkey];

              if (ctx_len == 1 && is_new_pair)
                {
                  ++continuation_counts_[static_cast<size_t> (w_id)];
                  ++total_unique_bigrams_;
                }
            }
        }
    }

  return {};
}

double
NGram_Model::prob_recursive (std::span<const int32_t> ctx,
                             int32_t word_id) const
{
  const double V = static_cast<double> (std::max (vocab_.size (), size_t (1)));
  if (ctx.empty ())
    {
      size_t u_cnt = (static_cast<size_t> (word_id) < unigram_counts_.size ())
                       ? unigram_counts_[static_cast<size_t> (word_id)] : 0;
      if (smoothing_ == Smoothing_Method::Kneser_Ney
          && total_unique_bigrams_ > 0)
        {
          size_t idx = static_cast<size_t> (word_id);
          size_t c_cnt = (idx < continuation_counts_.size ())
                           ? continuation_counts_[idx] : 0;
          double p_cont = static_cast<double> (c_cnt)
                          / static_cast<double> (total_unique_bigrams_);
          return (p_cont > 0.0) ? p_cont : (1.0 / (V * 100.0));
        }
      if (smoothing_ == Smoothing_Method::Laplace)
        {
          double alpha = std::max (param_, 1e-6);
          return (static_cast<double> (u_cnt) + alpha)
                 / (static_cast<double> (total_unigrams_) + alpha * V);
        }
      if (total_unigrams_ == 0 || u_cnt == 0)
        {
          return 1.0 / (V * 100.0);
        }
      return static_cast<double> (u_cnt)
             / static_cast<double> (total_unigrams_);
    }

  std::string hkey = encode_history_key (ctx);
  auto tot_it = context_totals_.find (hkey);
  if (tot_it == context_totals_.end () || tot_it->second == 0)
    {
      return prob_recursive (ctx.subspan (1), word_id);
    }

  size_t ctx_total = tot_it->second;
  size_t word_cnt = 0;
  size_t unique_followers = 0;
  auto map_it = ngram_counts_.find (hkey);
  if (map_it != ngram_counts_.end ())
    {
      unique_followers = map_it->second.size ();
      auto w_it = map_it->second.find (word_id);
      if (w_it != map_it->second.end ())
        {
          word_cnt = w_it->second;
        }
    }

  if (smoothing_ == Smoothing_Method::MLE)
    {
      return (word_cnt > 0)
               ? (static_cast<double> (word_cnt)
                  / static_cast<double> (ctx_total))
               : prob_recursive (ctx.subspan (1), word_id) * 0.1;
    }

  if (smoothing_ == Smoothing_Method::Laplace)
    {
      double alpha = std::max (param_, 1e-6);
      return (static_cast<double> (word_cnt) + alpha)
             / (static_cast<double> (ctx_total) + alpha * V);
    }

  /* Interpolated Kneser-Ney smoothing.  */
  double d = std::clamp (param_, 0.01, 0.99);
  double first_term = std::max (static_cast<double> (word_cnt) - d, 0.0)
                      / static_cast<double> (ctx_total);
  double lambda = (d * static_cast<double> (unique_followers))
                  / static_cast<double> (ctx_total);
  double lower = prob_recursive (ctx.subspan (1), word_id);
  return first_term + lambda * lower;
}

double
NGram_Model::probability (std::span<const std::string_view> context,
                          std::string_view word) const
{
  std::vector<int32_t> ctx_ids;
  size_t max_ctx = (order_ > 1) ? (order_ - 1) : 0;
  size_t start = (context.size () > max_ctx)
                   ? (context.size () - max_ctx) : 0;
  for (size_t i = start; i < context.size (); ++i)
    {
      std::string low = lowercase_word (context[i]);
      int32_t id = vocab_.token_to_id (low);
      ctx_ids.push_back (id >= 0 ? id : unk_id_);
    }

  std::string w_low = lowercase_word (word);
  int32_t w_id = vocab_.token_to_id (w_low);
  if (w_id < 0)
    {
      w_id = unk_id_;
    }
  return prob_recursive (ctx_ids, w_id);
}

double
NGram_Model::sentence_log_prob (std::string_view text) const
{
  Arena arena;
  Pmr_Vector<std::string_view> spans (arena.resource ());
  simd::scan_pretoken_spans (text, spans, true);

  std::vector<int32_t> seq;
  for (size_t k = 1; k < order_; ++k)
    {
      seq.push_back (bos_id_);
    }
  for (std::string_view w : spans)
    {
      int32_t id = vocab_.token_to_id (lowercase_word (w));
      seq.push_back (id >= 0 ? id : unk_id_);
    }
  seq.push_back (eos_id_);

  double log_p = 0.0;
  size_t start_pos = (order_ > 1) ? (order_ - 1) : 0;
  for (size_t i = start_pos; i < seq.size (); ++i)
    {
      size_t ctx_len = order_ - 1;
      std::span<const int32_t> ctx (seq.data () + (i - ctx_len), ctx_len);
      double p = prob_recursive (ctx, seq[i]);
      log_p += std::log (std::max (p, 1e-15));
    }
  return log_p;
}

Result<double>
NGram_Model::perplexity (std::span<const std::string_view> corpus) const
{
  if (corpus.empty () || total_unigrams_ == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "cannot compute perplexity on empty corpus");
    }

  double total_log_prob = 0.0;
  size_t total_eval_tokens = 0;
  Arena arena;

  for (std::string_view doc : corpus)
    {
      arena.reset ();
      Pmr_Vector<std::string_view> spans (arena.resource ());
      simd::scan_pretoken_spans (doc, spans, true);
      total_eval_tokens += spans.size () + 1;
      total_log_prob += sentence_log_prob (doc);
    }

  if (total_eval_tokens == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "zero tokens in evaluation corpus");
    }
  return std::exp (-total_log_prob / static_cast<double> (total_eval_tokens));
}

/* -------------------------------------------------------------------------
   TFIDF_Vectorizer implementation
   ------------------------------------------------------------------------- */

TFIDF_Vectorizer::TFIDF_Vectorizer (bool sublinear_tf, bool smooth_idf)
  : sublinear_tf_ (sublinear_tf),
    smooth_idf_ (smooth_idf)
{
}

Result<void>
TFIDF_Vectorizer::fit (std::span<const std::string_view> corpus)
{
  if (corpus.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "TFIDF_Vectorizer::fit empty corpus");
    }

  vocab_.clear ();
  std::vector<size_t> doc_freq;
  Arena arena;

  for (std::string_view doc : corpus)
    {
      arena.reset ();
      Pmr_Vector<std::string_view> spans (arena.resource ());
      simd::scan_pretoken_spans (doc, spans, true);

      std::vector<int32_t> seen;
      seen.reserve (spans.size ());
      for (std::string_view w : spans)
        {
          int32_t id = vocab_.add_token (lowercase_word (w));
          if (static_cast<size_t> (id) >= doc_freq.size ())
            {
              doc_freq.resize (static_cast<size_t> (id) + 1, 0);
            }
          seen.push_back (id);
        }
      std::sort (seen.begin (), seen.end ());
      seen.erase (std::unique (seen.begin (), seen.end ()), seen.end ());
      for (int32_t id : seen)
        {
          ++doc_freq[static_cast<size_t> (id)];
        }
    }

  if (vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "corpus produced empty TF-IDF vocabulary");
    }

  const double N = static_cast<double> (corpus.size ());
  idf_.resize (vocab_.size (), 0.0f);
  for (size_t i = 0; i < vocab_.size (); ++i)
    {
      double df = static_cast<double> (doc_freq[i]);
      if (smooth_idf_)
        {
          idf_[i] = static_cast<float> (
            std::log ((1.0 + N) / (1.0 + df)) + 1.0);
        }
      else
        {
          idf_[i] = static_cast<float> (
            std::log (N / std::max (df, 1.0)) + 1.0);
        }
    }
  return {};
}

Result<std::vector<float>>
TFIDF_Vectorizer::transform_doc (std::string_view doc) const
{
  if (vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "TFIDF_Vectorizer is not fitted");
    }

  std::vector<float> vec (vocab_.size (), 0.0f);
  Arena arena;
  Pmr_Vector<std::string_view> spans (arena.resource ());
  simd::scan_pretoken_spans (doc, spans, true);

  for (std::string_view w : spans)
    {
      int32_t id = vocab_.token_to_id (lowercase_word (w));
      if (id >= 0)
        {
          vec[static_cast<size_t> (id)] += 1.0f;
        }
    }

  for (size_t i = 0; i < vec.size (); ++i)
    {
      if (vec[i] > 0.0f)
        {
          float tf = sublinear_tf_ ? (1.0f + std::log (vec[i])) : vec[i];
          vec[i] = tf * idf_[i];
        }
    }

  simd::normalize_inplace_f32 (vec);
  return vec;
}

Result<Tensor>
TFIDF_Vectorizer::transform (std::span<const std::string_view> docs) const
{
  if (docs.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "TFIDF_Vectorizer::transform empty docs");
    }
  if (vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "TFIDF_Vectorizer is not fitted");
    }

  const size_t D = docs.size ();
  const size_t V = vocab_.size ();
  Tensor mat = basalt::zeros ({D, V},
                              Tensor_Options ().with_dtype (Dtype::Float32));
  float *dst = mat.data_ptr<float> ();

  for (size_t d = 0; d < D; ++d)
    {
      auto row_res = transform_doc (docs[d]);
      if (!row_res.has_value ())
        {
          return std::unexpected (row_res.error ());
        }
      std::memcpy (dst + d * V, row_res->data (), V * sizeof (float));
    }
  return mat;
}

Result<Tensor>
TFIDF_Vectorizer::fit_transform (std::span<const std::string_view> corpus)
{
  auto fit_res = fit (corpus);
  if (!fit_res.has_value ())
    {
      return std::unexpected (fit_res.error ());
    }
  return transform (corpus);
}

/* -------------------------------------------------------------------------
   BM25_Index implementation
   ------------------------------------------------------------------------- */

BM25_Index::BM25_Index (float k1, float b)
  : k1_ (k1),
    b_ (b)
{
}

Result<void>
BM25_Index::index_documents (std::span<const std::string_view> docs)
{
  if (docs.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "BM25_Index requires at least one document");
    }

  vocab_.clear ();
  doc_lengths_.assign (docs.size (), 0);
  postings_.clear ();

  size_t total_len = 0;
  Arena arena;

  for (size_t d = 0; d < docs.size (); ++d)
    {
      arena.reset ();
      Pmr_Vector<std::string_view> spans (arena.resource ());
      simd::scan_pretoken_spans (docs[d], spans, true);
      doc_lengths_[d] = static_cast<uint32_t> (spans.size ());
      total_len += spans.size ();

      std::unordered_map<int32_t, uint32_t> term_counts;
      for (std::string_view w : spans)
        {
          int32_t id = vocab_.add_token (lowercase_word (w));
          if (static_cast<size_t> (id) >= postings_.size ())
            {
              postings_.resize (static_cast<size_t> (id) + 1);
            }
          ++term_counts[id];
        }

      for (const auto &[id, tf] : term_counts)
        {
          postings_[static_cast<size_t> (id)].push_back (
            Posting {static_cast<uint32_t> (d), tf});
        }
    }

  avg_doc_len_ = static_cast<float> (total_len)
                 / static_cast<float> (docs.size ());
  const double N = static_cast<double> (docs.size ());
  idf_.resize (vocab_.size (), 0.0f);

  for (size_t t = 0; t < vocab_.size (); ++t)
    {
      double df = static_cast<double> (postings_[t].size ());
      double ratio = (N - df + 0.5) / (df + 0.5) + 1.0;
      idf_[t] = static_cast<float> (std::log (ratio));
    }

  return {};
}

Result<std::vector<Search_Hit>>
BM25_Index::search (std::string_view query, size_t top_k) const
{
  if (doc_lengths_.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "BM25_Index has no indexed documents");
    }

  std::vector<float> scores (doc_lengths_.size (), 0.0f);
  Arena arena;
  Pmr_Vector<std::string_view> q_terms (arena.resource ());
  simd::scan_pretoken_spans (query, q_terms, true);

  for (std::string_view qt : q_terms)
    {
      int32_t tid = vocab_.token_to_id (lowercase_word (qt));
      if (tid < 0)
        {
          continue;
        }
      float term_idf = idf_[static_cast<size_t> (tid)];
      for (const Posting &p : postings_[static_cast<size_t> (tid)])
        {
          float tf = static_cast<float> (p.term_freq);
          float dl = static_cast<float> (doc_lengths_[p.doc_id]);
          float norm = 1.0f - b_ + b_ * (dl / std::max (avg_doc_len_, 1.0f));
          float tf_sat = (tf * (k1_ + 1.0f)) / (tf + k1_ * norm);
          scores[p.doc_id] += term_idf * tf_sat;
        }
    }

  std::vector<Search_Hit> hits;
  hits.reserve (scores.size ());
  for (size_t d = 0; d < scores.size (); ++d)
    {
      hits.push_back (Search_Hit {d, scores[d]});
    }

  size_t k = std::min (top_k, hits.size ());
  std::partial_sort (
    hits.begin (),
    hits.begin () + static_cast<ptrdiff_t> (k),
    hits.end (),
    [] (const Search_Hit &lhs, const Search_Hit &rhs)
    {
      return lhs.score > rhs.score;
    });
  hits.resize (k);
  return hits;
}

} /* namespace obsidian */
