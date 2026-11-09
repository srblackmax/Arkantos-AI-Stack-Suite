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

#include <obsidian/pipeline.hpp>
#include <obsidian/simd.hpp>

#include <algorithm>
#include <cmath>

namespace obsidian
{

/* -------------------------------------------------------------------------
   Linguistic_Pipeline implementation
   ------------------------------------------------------------------------- */

Linguistic_Pipeline::Linguistic_Pipeline (void)
{
  /* Seed default POS tagger with core English syntactic patterns so
     out-of-the-box annotation works even prior to custom training.  */
  using Sent = std::vector<std::pair<std::string, std::string>>;
  std::vector<Sent> seed_corpus = {
    {{"the", "DET"}, {"quick", "ADJ"}, {"fox", "NOUN"}, {"jumps", "VERB"}},
    {{"a", "DET"}, {"modern", "ADJ"}, {"system", "NOUN"}, {"runs", "VERB"},
     {"fast", "ADV"}}
  };
  if (pos_tagger_.train (seed_corpus, 0.05f).has_value ())
    {
      pos_trained_ = true;
    }
}

Result<void>
Linguistic_Pipeline::train_pos (
  const std::vector<std::vector<std::pair<std::string, std::string>>> &sents)
{
  auto res = pos_tagger_.train (sents, 0.01f);
  if (!res.has_value ())
    {
      return res;
    }
  pos_trained_ = true;
  return {};
}

void
Linguistic_Pipeline::add_entity_gazetteer (
  const std::vector<std::string> &tokens,
  std::string_view label)
{
  std::string joined;
  for (const auto &t : tokens)
    {
      if (!joined.empty ())
        {
          joined.push_back (' ');
        }
      joined.append (t);
    }
  ner_.add_gazetteer_entry (joined, label);
}

Result<Document_Annotation>
Linguistic_Pipeline::annotate (std::string_view text) const
{
  auto utf8_ok = simd::validate_and_count_utf8 (text);
  if (!utf8_ok.has_value ())
    {
      return std::unexpected (utf8_ok.error ());
    }

  Document_Annotation doc;
  doc.raw_text.assign (text.data (), text.size ());

  auto sent_views = segmenter_.segment (doc.raw_text);
  doc.sentences.reserve (sent_views.size ());
  for (std::string_view sv : sent_views)
    {
      doc.sentences.emplace_back (sv);
    }

  Arena &arena = thread_local_arena ();
  arena.reset ();
  Pmr_Vector<std::string_view> tok_views (arena.resource ());
  simd::scan_pretoken_spans (doc.raw_text, tok_views, true);

  const size_t n = tok_views.size ();
  doc.tokens.reserve (n);
  doc.stems.reserve (n);
  doc.lemmas.reserve (n);

  for (std::string_view tv : tok_views)
    {
      doc.tokens.emplace_back (tv);
      doc.stems.push_back (stemmer_.stem (tv));
    }

  if (pos_trained_ && !tok_views.empty ())
    {
      auto tags_res = pos_tagger_.tag (tok_views);
      if (tags_res.has_value ())
        {
          doc.pos_tags = std::move (*tags_res);
        }
    }
  if (doc.pos_tags.size () != n)
    {
      doc.pos_tags.assign (n, "NOUN");
    }

  for (size_t i = 0; i < n; ++i)
    {
      doc.lemmas.push_back (
        lemmatizer_.lemmatize (doc.tokens[i], doc.pos_tags[i]));
    }

  if (!tok_views.empty ())
    {
      auto ent_res = ner_.extract_spans (tok_views);
      if (ent_res.has_value ())
        {
          doc.entities = std::move (*ent_res);
        }
    }

  return doc;
}

/* -------------------------------------------------------------------------
   Text_Classification_Pipeline implementation
   ------------------------------------------------------------------------- */

Text_Classification_Pipeline::Text_Classification_Pipeline (
  std::shared_ptr<BPE_Tokenizer> tokenizer,
  std::shared_ptr<Sequence_Classifier> model,
  std::vector<std::string> labels)
  : tokenizer_ (std::move (tokenizer)),
    model_ (std::move (model)),
    labels_ (std::move (labels))
{
}

Result<Classification_Prediction>
Text_Classification_Pipeline::predict (std::string_view text) const
{
  if (tokenizer_ == nullptr || model_ == nullptr || labels_.empty ())
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "Text_Classification_Pipeline not initialized");
    }

  std::string_view batch_input[1] = {text};
  auto batch_res = tokenizer_->encode_batch (batch_input, true, true);
  if (!batch_res.has_value ())
    {
      return std::unexpected (batch_res.error ());
    }

  No_Grad_Guard no_grad;
  model_->eval ();
  auto logits_res = model_->forward_batch (*batch_res);
  if (!logits_res.has_value ())
    {
      return std::unexpected (logits_res.error ());
    }

  Tensor probs = basalt::ops::softmax (*logits_res, -1);
  size_t num_classes = std::min (labels_.size (), probs.size (-1));
  size_t best_idx = 0;
  float best_prob = -1.0f;

  for (size_t c = 0; c < num_classes; ++c)
    {
      float p = static_cast<float> (probs.read_scalar_at (c));
      if (p > best_prob)
        {
          best_prob = p;
          best_idx = c;
        }
    }

  return Classification_Prediction {
    labels_[best_idx],
    static_cast<int32_t> (best_idx),
    best_prob
  };
}

/* -------------------------------------------------------------------------
   Text_Generation_Pipeline implementation
   ------------------------------------------------------------------------- */

Text_Generation_Pipeline::Text_Generation_Pipeline (
  std::shared_ptr<BPE_Tokenizer> tokenizer,
  std::shared_ptr<Causal_LM> model)
  : tokenizer_ (std::move (tokenizer)),
    model_ (std::move (model))
{
}

Result<std::string>
Text_Generation_Pipeline::generate_text (std::string_view prompt,
                                         const Generation_Config &cfg) const
{
  if (tokenizer_ == nullptr || model_ == nullptr)
    {
      return OBSIDIAN_ERROR (Error_Code::Not_Initialized,
                             "Text_Generation_Pipeline not initialized");
    }

  auto prompt_ids = tokenizer_->encode (prompt, true, false);
  if (!prompt_ids.has_value ())
    {
      return std::unexpected (prompt_ids.error ());
    }
  if (prompt_ids->empty ())
    {
      prompt_ids->push_back (tokenizer_->special_tokens ().bos_id);
    }

  auto out_ids = model_->generate (*prompt_ids, cfg);
  if (!out_ids.has_value ())
    {
      return std::unexpected (out_ids.error ());
    }

  return tokenizer_->decode (*out_ids, true);
}

} /* namespace obsidian */
