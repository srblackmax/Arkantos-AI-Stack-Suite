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

#include <obsidian/topic.hpp>
#include <obsidian/simd.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace obsidian
{

namespace
{

std::string
to_lower_str (std::string_view sv)
{
  std::string s (sv.size (), '\0');
  simd::ascii_to_lower (sv, s.data ());
  return s;
}

bool
is_punct_only (std::string_view sv) noexcept
{
  for (unsigned char c : sv)
    {
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
          || (c >= '0' && c <= '9'))
        {
          return false;
        }
    }
  return true;
}

} /* namespace */

/* -------------------------------------------------------------------------
   LDA_Model implementation
   ------------------------------------------------------------------------- */

LDA_Model::LDA_Model (size_t num_topics,
                      float alpha,
                      float beta,
                      size_t iterations,
                      uint64_t seed)
  : num_topics_ (std::max (num_topics, size_t (2))),
    alpha_ (std::max (alpha, 1e-4f)),
    beta_ (std::max (beta, 1e-4f)),
    iterations_ (std::max (iterations, size_t (1))),
    seed_ (seed)
{
}

Result<void>
LDA_Model::fit (std::span<const std::string_view> corpus)
{
  if (corpus.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "LDA_Model::fit requires non-empty corpus");
    }

  vocab_.clear ();
  docs_.clear ();
  z_.clear ();

  Arena arena;
  for (std::string_view doc : corpus)
    {
      arena.reset ();
      Pmr_Vector<std::string_view> spans (arena.resource ());
      simd::scan_pretoken_spans (doc, spans, true);

      std::vector<int32_t> doc_ids;
      for (std::string_view w : spans)
        {
          if (is_punct_only (w))
            {
              continue;
            }
          doc_ids.push_back (vocab_.add_token (to_lower_str (w)));
        }
      docs_.push_back (std::move (doc_ids));
    }

  const size_t D = docs_.size ();
  const size_t V = vocab_.size ();
  const size_t K = num_topics_;
  if (V == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Invalid_Argument,
                             "corpus produced empty vocabulary for LDA");
    }

  doc_topic_counts_.assign (D * K, 0);
  topic_word_counts_.assign (K * V, 0);
  topic_totals_.assign (K, 0);
  z_.resize (D);

  std::mt19937_64 rng (seed_);
  std::uniform_int_distribution<size_t> topic_dist (0, K - 1);
  std::uniform_real_distribution<float> uni01 (0.0f, 1.0f);

  for (size_t d = 0; d < D; ++d)
    {
      z_[d].resize (docs_[d].size (), 0);
      for (size_t i = 0; i < docs_[d].size (); ++i)
        {
          size_t k = topic_dist (rng);
          size_t w = static_cast<size_t> (docs_[d][i]);
          z_[d][i] = k;
          ++doc_topic_counts_[d * K + k];
          ++topic_word_counts_[k * V + w];
          ++topic_totals_[k];
        }
    }

  std::vector<float> cdf (K, 0.0f);
  const float v_beta = static_cast<float> (V) * beta_;

  for (size_t iter = 0; iter < iterations_; ++iter)
    {
      for (size_t d = 0; d < D; ++d)
        {
          for (size_t i = 0; i < docs_[d].size (); ++i)
            {
              size_t old_k = z_[d][i];
              size_t w = static_cast<size_t> (docs_[d][i]);

              --doc_topic_counts_[d * K + old_k];
              --topic_word_counts_[old_k * V + w];
              --topic_totals_[old_k];

              float accum = 0.0f;
              for (size_t k = 0; k < K; ++k)
                {
                  float dt = static_cast<float> (doc_topic_counts_[d * K + k])
                             + alpha_;
                  float tw = static_cast<float> (topic_word_counts_[k * V + w])
                             + beta_;
                  float tt = static_cast<float> (topic_totals_[k]) + v_beta;
                  accum += dt * (tw / tt);
                  cdf[k] = accum;
                }

              float u = uni01 (rng) * accum;
              size_t new_k = K - 1;
              for (size_t k = 0; k < K; ++k)
                {
                  if (u <= cdf[k])
                    {
                      new_k = k;
                      break;
                    }
                }

              z_[d][i] = new_k;
              ++doc_topic_counts_[d * K + new_k];
              ++topic_word_counts_[new_k * V + w];
              ++topic_totals_[new_k];
            }
        }
    }

  return {};
}

Result<std::vector<std::pair<std::string, float>>>
LDA_Model::topic_words (size_t topic_id, size_t top_k) const
{
  if (vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "LDA_Model is not fitted");
    }
  if (topic_id >= num_topics_)
    {
      return OBSIDIAN_ERROR (Error_Code::Out_Of_Range,
                             "topic_id out of bounds");
    }

  const size_t V = vocab_.size ();
  float denom = static_cast<float> (topic_totals_[topic_id])
                + static_cast<float> (V) * beta_;

  std::vector<std::pair<std::string, float>> words;
  words.reserve (V);
  for (size_t w = 0; w < V; ++w)
    {
      float p = (static_cast<float> (topic_word_counts_[topic_id * V + w])
                 + beta_) / denom;
      std::string_view sv = vocab_.id_to_token (
        static_cast<int32_t> (w)).value ();
      words.emplace_back (std::string (sv), p);
    }

  size_t k = std::min (top_k, words.size ());
  std::partial_sort (
    words.begin (),
    words.begin () + static_cast<ptrdiff_t> (k),
    words.end (),
    [] (const auto &lhs, const auto &rhs)
    {
      return lhs.second > rhs.second;
    });
  words.resize (k);
  return words;
}

Result<std::vector<float>>
LDA_Model::document_topic_distribution (size_t doc_id) const
{
  if (docs_.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "LDA_Model is not fitted");
    }
  if (doc_id >= docs_.size ())
    {
      return OBSIDIAN_ERROR (Error_Code::Out_Of_Range,
                             "doc_id out of bounds");
    }

  const size_t K = num_topics_;
  std::vector<float> theta (K, 0.0f);
  float denom = static_cast<float> (docs_[doc_id].size ())
                + static_cast<float> (K) * alpha_;

  for (size_t k = 0; k < K; ++k)
    {
      theta[k] = (static_cast<float> (doc_topic_counts_[doc_id * K + k])
                  + alpha_) / denom;
    }
  return theta;
}

Result<std::vector<float>>
LDA_Model::infer_document (std::string_view doc, size_t infer_iters) const
{
  if (vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "LDA_Model is not fitted");
    }

  Arena arena;
  Pmr_Vector<std::string_view> spans (arena.resource ());
  simd::scan_pretoken_spans (doc, spans, true);

  std::vector<int32_t> doc_ids;
  for (std::string_view w : spans)
    {
      int32_t id = vocab_.token_to_id (to_lower_str (w));
      if (id >= 0)
        {
          doc_ids.push_back (id);
        }
    }

  const size_t K = num_topics_;
  const size_t V = vocab_.size ();
  std::vector<size_t> local_counts (K, 0);
  std::vector<size_t> local_z (doc_ids.size (), 0);

  std::mt19937_64 rng (seed_ + 999);
  std::uniform_int_distribution<size_t> topic_dist (0, K - 1);
  std::uniform_real_distribution<float> uni01 (0.0f, 1.0f);

  for (size_t i = 0; i < doc_ids.size (); ++i)
    {
      size_t k = topic_dist (rng);
      local_z[i] = k;
      ++local_counts[k];
    }

  std::vector<float> cdf (K, 0.0f);
  const float v_beta = static_cast<float> (V) * beta_;

  for (size_t it = 0; it < infer_iters; ++it)
    {
      for (size_t i = 0; i < doc_ids.size (); ++i)
        {
          size_t old_k = local_z[i];
          size_t w = static_cast<size_t> (doc_ids[i]);
          --local_counts[old_k];

          float accum = 0.0f;
          for (size_t k = 0; k < K; ++k)
            {
              float dt = static_cast<float> (local_counts[k]) + alpha_;
              float tw = static_cast<float> (topic_word_counts_[k * V + w])
                         + beta_;
              float tt = static_cast<float> (topic_totals_[k]) + v_beta;
              accum += dt * (tw / tt);
              cdf[k] = accum;
            }

          float u = uni01 (rng) * accum;
          size_t new_k = K - 1;
          for (size_t k = 0; k < K; ++k)
            {
              if (u <= cdf[k])
                {
                  new_k = k;
                  break;
                }
            }
          local_z[i] = new_k;
          ++local_counts[new_k];
        }
    }

  std::vector<float> theta (K, 0.0f);
  float denom = static_cast<float> (doc_ids.size ())
                + static_cast<float> (K) * alpha_;
  for (size_t k = 0; k < K; ++k)
    {
      theta[k] = (static_cast<float> (local_counts[k]) + alpha_) / denom;
    }
  return theta;
}

Result<Tensor>
LDA_Model::topic_word_tensor (void) const
{
  if (vocab_.size () == 0)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "LDA_Model is not fitted");
    }
  const size_t K = num_topics_;
  const size_t V = vocab_.size ();
  Tensor mat = basalt::zeros ({K, V},
                              Tensor_Options ().with_dtype (Dtype::Float32));
  float *dst = mat.data_ptr<float> ();
  const float v_beta = static_cast<float> (V) * beta_;

  for (size_t k = 0; k < K; ++k)
    {
      float denom = static_cast<float> (topic_totals_[k]) + v_beta;
      for (size_t w = 0; w < V; ++w)
        {
          dst[k * V + w] = (static_cast<float> (topic_word_counts_[k * V + w])
                            + beta_) / denom;
        }
    }
  return mat;
}

/* -------------------------------------------------------------------------
   LSA_Model implementation (Truncated SVD via Basalt Tensor ops)
   ------------------------------------------------------------------------- */

LSA_Model::LSA_Model (size_t n_components, size_t power_iterations)
  : n_components_ (std::max (n_components, size_t (1))),
    power_iters_ (std::max (power_iterations, size_t (5)))
{
}

Result<Tensor>
LSA_Model::fit_transform (std::span<const std::string_view> corpus)
{
  auto tfidf_res = tfidf_.fit_transform (corpus);
  if (!tfidf_res.has_value ())
    {
      return std::unexpected (tfidf_res.error ());
    }

  Tensor X = *tfidf_res; /* [D, V] */
  const size_t D = X.size (0);
  const size_t V = X.size (1);
  const size_t K = std::min ({n_components_, D, V});

  /* Compute Gram matrix G = X^T * X of shape [V, V] via Basalt matmul.  */
  No_Grad_Guard no_grad;
  Tensor Xt = X.transpose (0, 1).contiguous ();
  Tensor G = basalt::ops::matmul (Xt, X);

  components_ = basalt::zeros ({K, V},
                               Tensor_Options ().with_dtype (Dtype::Float32));
  singular_values_ = basalt::zeros (
    {K}, Tensor_Options ().with_dtype (Dtype::Float32));

  float *comp_ptr = components_.data_ptr<float> ();
  float *sv_ptr = singular_values_.data_ptr<float> ();

  std::mt19937_64 rng (12345);
  std::normal_distribution<float> norm_dist (0.0f, 1.0f);

  for (size_t k = 0; k < K; ++k)
    {
      Tensor v_vec = basalt::zeros (
        {V, 1}, Tensor_Options ().with_dtype (Dtype::Float32));
      float *vp = v_vec.data_ptr<float> ();
      for (size_t i = 0; i < V; ++i)
        {
          vp[i] = norm_dist (rng);
        }
      simd::normalize_inplace_f32 (std::span<float> (vp, V));

      for (size_t it = 0; it < power_iters_; ++it)
        {
          Tensor next_v = basalt::ops::matmul (G, v_vec);
          float *np = next_v.data_ptr<float> ();

          /* Orthogonalize against previously extracted singular vectors.  */
          for (size_t prev = 0; prev < k; ++prev)
            {
              std::span<const float> p_span (comp_ptr + prev * V, V);
              float proj = simd::dot_f32 (std::span<const float> (np, V),
                                          p_span);
              simd::axpy_f32 (-proj, p_span, std::span<float> (np, V));
            }
          simd::normalize_inplace_f32 (std::span<float> (np, V));
          v_vec = std::move (next_v);
          vp = v_vec.data_ptr<float> ();
        }

      std::copy (vp, vp + V, comp_ptr + k * V);
      Tensor xv = basalt::ops::matmul (X, v_vec);
      std::span<const float> xv_sp (xv.data_ptr<float> (), D);
      sv_ptr[k] = simd::norm_f32 (xv_sp);
    }

  Tensor Vt = components_.transpose (0, 1).contiguous ();
  return basalt::ops::matmul (X, Vt);
}

Result<Tensor>
LSA_Model::transform (std::span<const std::string_view> docs) const
{
  if (!components_.defined ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "LSA_Model is not fitted");
    }
  auto tfidf_res = tfidf_.transform (docs);
  if (!tfidf_res.has_value ())
    {
      return std::unexpected (tfidf_res.error ());
    }
  No_Grad_Guard no_grad;
  Tensor Vt = components_.transpose (0, 1).contiguous ();
  return basalt::ops::matmul (*tfidf_res, Vt);
}

} /* namespace obsidian */
