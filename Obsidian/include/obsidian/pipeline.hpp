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

#ifndef OBSIDIAN_PIPELINE_HPP
#define OBSIDIAN_PIPELINE_HPP

/**
 * @file pipeline.hpp
 * @brief C++23 composable NLP pipelines using pipe (|) syntax, spaCy-caliber
 *        Document_Annotation, and Transformer classification/generation
 *        pipelines.
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <obsidian/classical.hpp>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/models.hpp>
#include <obsidian/stemmer.hpp>
#include <obsidian/tagger.hpp>
#include <obsidian/tokenizer.hpp>

namespace obsidian
{

/**
 * @brief Generic pipeable stage wrapper enabling C++23 range-style (val | fn)
 *        and monadic Result<T> | fn composition.
 */
template <typename Callable>
class Pipe_Stage
{
public:
  /**
   * @brief Construct a Pipe_Stage wrapping FN.
   * @pre FN is a valid callable object.
   * @complexity O(1) move construction time.
   */
  constexpr explicit Pipe_Stage (Callable fn)
    : fn_ (std::move (fn))
  {
  }

  /**
   * @brief Invoke the wrapped stage directly on ARG.
   * @pre FN is invocable with ARG.
   * @complexity Depends on wrapped callable FN.
   */
  template <typename Arg>
    requires std::invocable<const Callable &, Arg>
  constexpr decltype (auto)
  operator() (Arg &&arg) const
  {
    return std::invoke (fn_, std::forward<Arg> (arg));
  }

private:
  Callable fn_;
};

/**
 * @brief Create a composable Pipe_Stage from any callable or lambda.
 * @pre None.
 * @complexity O(1) time and space.
 */
template <typename Callable>
constexpr Pipe_Stage<std::decay_t<Callable>>
make_stage (Callable &&fn)
{
  return Pipe_Stage<std::decay_t<Callable>> (std::forward<Callable> (fn));
}

/**
 * @brief Pipe operator applying STAGE to a direct value LHS.
 * @pre STAGE is invocable with LHS.
 * @complexity Depends on STAGE.
 */
template <typename T, typename Callable>
  requires std::invocable<const Pipe_Stage<Callable> &, T>
constexpr decltype (auto)
operator| (T &&lhs, const Pipe_Stage<Callable> &stage)
{
  return stage (std::forward<T> (lhs));
}

/**
 * @brief Monadic pipe operator propagating Error or applying STAGE to the
 *        unwrapped value of Result<T>.
 * @pre STAGE is invocable with const T&.
 * @complexity O(1) on error; depends on STAGE on success.
 */
template <typename T, typename Callable>
  requires std::invocable<const Pipe_Stage<Callable> &, const T &>
constexpr auto
operator| (const Result<T> &lhs, const Pipe_Stage<Callable> &stage)
  -> Result<std::decay_t<std::invoke_result_t<const Pipe_Stage<Callable> &,
                                              const T &>>>
{
  using Out_Type = std::decay_t<
    std::invoke_result_t<const Pipe_Stage<Callable> &, const T &>>;
  if (!lhs.has_value ())
    {
      return std::unexpected (lhs.error ());
    }
  return Result<Out_Type> (stage (*lhs));
}

/**
 * @brief Unified linguistic annotation document (spaCy-style container).
 */
struct OBSIDIAN_API Document_Annotation
{
  std::string raw_text;
  std::vector<std::string> sentences;
  std::vector<std::string> tokens;
  std::vector<std::string> stems;
  std::vector<std::string> lemmas;
  std::vector<std::string> pos_tags;
  std::vector<Entity_Span> entities;
};

/**
 * @brief End-to-end Classical NLP annotation pipeline combining sentence
 *        segmentation, SIMD tokenization, stemming, POS tagging,
 *        lemmatization, and Named Entity Recognition.
 */
class OBSIDIAN_API Linguistic_Pipeline
{
public:
  /**
   * @brief Construct a Linguistic_Pipeline with default English rules and
   *        optional trained POS tagger and NER extractor.
   * @pre None.
   * @complexity O(1) initialization time.
   */
  Linguistic_Pipeline (void);

  /**
   * @brief Train the internal POS tagger on labeled (word, tag) sentences.
   * @pre SENTENCES must not be empty.
   * @complexity O(total_tokens) time.
   */
  Result<void>
  train_pos (
    const std::vector<std::vector<std::pair<std::string,
                                            std::string>>> &sents);

  /**
   * @brief Register a multi-token named entity in the NER gazetteer.
   * @pre TOKENS must not be empty; LABEL must not be empty.
   * @complexity O(tokens.size()) time.
   */
  void
  add_entity_gazetteer (const std::vector<std::string> &tokens,
                        std::string_view label);

  /**
   * @brief Annotate TEXT and produce a rich Document_Annotation.
   * @pre TEXT must be valid UTF-8.
   * @complexity O(N) time in character/token length N.
   */
  Result<Document_Annotation>
  annotate (std::string_view text) const;

  /**
   * @brief Callable operator alias for annotate(text).
   * @pre TEXT must be valid UTF-8.
   * @complexity O(N) time.
   */
  Result<Document_Annotation>
  operator() (std::string_view text) const
  {
    return annotate (text);
  }

private:
  Sentence_Segmenter segmenter_;
  Porter_Stemmer stemmer_;
  Lemmatizer lemmatizer_;
  HMM_POS_Tagger pos_tagger_;
  bool pos_trained_ = false;
  NER_Span_Extractor ner_;
};

/**
 * @brief Output prediction of Text_Classification_Pipeline.
 */
struct OBSIDIAN_API Classification_Prediction
{
  std::string label;
  int32_t class_id = 0;
  float probability = 0.0f;
};

/**
 * @brief End-to-end Transformer text classification pipeline.
 */
class OBSIDIAN_API Text_Classification_Pipeline
{
public:
  /**
   * @brief Construct a classification pipeline from a BPE tokenizer,
   *        Sequence_Classifier model, and class label names.
   * @pre MODEL != nullptr and LABELS.size() >= 2.
   * @complexity O(1) time.
   */
  Text_Classification_Pipeline (
    std::shared_ptr<BPE_Tokenizer> tokenizer,
    std::shared_ptr<Sequence_Classifier> model,
    std::vector<std::string> labels);

  /**
   * @brief Classify TEXT and return the highest-probability class prediction.
   * @pre TEXT must be valid UTF-8.
   * @complexity O(num_layers * S * d_model^2) time.
   */
  Result<Classification_Prediction>
  predict (std::string_view text) const;

private:
  std::shared_ptr<BPE_Tokenizer> tokenizer_;
  std::shared_ptr<Sequence_Classifier> model_;
  std::vector<std::string> labels_;
};

/**
 * @brief End-to-end Transformer causal text generation pipeline.
 */
class OBSIDIAN_API Text_Generation_Pipeline
{
public:
  /**
   * @brief Construct a generation pipeline from a BPE tokenizer and Causal_LM.
   * @pre TOKENIZER != nullptr and MODEL != nullptr.
   * @complexity O(1) time.
   */
  Text_Generation_Pipeline (
    std::shared_ptr<BPE_Tokenizer> tokenizer,
    std::shared_ptr<Causal_LM> model);

  /**
   * @brief Encode PROMPT, generate continuation tokens, and decode to text.
   * @pre PROMPT must be non-empty valid UTF-8.
   * @complexity O((|PROMPT| + max_new_tokens) * num_layers * d_model^2).
   */
  Result<std::string>
  generate_text (std::string_view prompt,
                 const Generation_Config &cfg = Generation_Config ()) const;

private:
  std::shared_ptr<BPE_Tokenizer> tokenizer_;
  std::shared_ptr<Causal_LM> model_;
};

} /* namespace obsidian */

#endif /* OBSIDIAN_PIPELINE_HPP */
