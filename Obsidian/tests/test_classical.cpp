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

#include <obsidian/stemmer.hpp>
#include <obsidian/tagger.hpp>
#include <obsidian/classical.hpp>
#include "test_harness.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace obsidian;

OBSIDIAN_TEST (porter_stemmer_and_lemmatizer)
{
  Porter_Stemmer stemmer;
  OBSIDIAN_CHECK_EQ (stemmer.stem ("caresses"), "caress");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("ponies"), "poni");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("cats"), "cat");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("walking"), "walk");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("plastered"), "plaster");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("relational"), "relat");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("generalization"), "gener");
  OBSIDIAN_CHECK_EQ (stemmer.stem ("formalize"), "formal");

  Arena arena;
  std::string_view sv = stemmer.stem_into_arena ("computational", arena);
  OBSIDIAN_CHECK_EQ (sv, "comput");

  Lemmatizer lemma;
  OBSIDIAN_CHECK_EQ (lemma.lemmatize ("children", "NNS"), "child");
  OBSIDIAN_CHECK_EQ (lemma.lemmatize ("went", "VBD"), "go");
  OBSIDIAN_CHECK_EQ (lemma.lemmatize ("better", "JJR"), "good");
  OBSIDIAN_CHECK_EQ (lemma.lemmatize ("running", "VBG"), "run");
}

OBSIDIAN_TEST (viterbi_hmm_perceptron_pos_and_ner)
{
  using Sent = std::vector<std::pair<std::string, std::string>>;
  std::vector<Sent> train_sents = {
    {{"the", "DET"}, {"fast", "ADJ"}, {"dog", "NOUN"}, {"runs", "VERB"}},
    {{"a", "DET"}, {"smart", "ADJ"}, {"cat", "NOUN"}, {"sleeps", "VERB"}},
    {{"the", "DET"}, {"cat", "NOUN"}, {"runs", "VERB"}}
  };

  HMM_POS_Tagger hmm;
  OBSIDIAN_CHECK_OK (hmm.train (train_sents, 0.01f));

  std::vector<std::string_view> test_words = {"the", "fast", "cat", "runs"};
  auto hmm_tags = hmm.tag (test_words);
  OBSIDIAN_CHECK_OK (hmm_tags);
  OBSIDIAN_CHECK_EQ (hmm_tags->size (), 4u);
  OBSIDIAN_CHECK_EQ ((*hmm_tags)[0], "DET");
  OBSIDIAN_CHECK_EQ ((*hmm_tags)[2], "NOUN");
  OBSIDIAN_CHECK_EQ ((*hmm_tags)[3], "VERB");

  Perceptron_POS_Tagger perc;
  OBSIDIAN_CHECK_OK (perc.train (train_sents, 8));
  auto perc_tags = perc.tag (test_words);
  OBSIDIAN_CHECK_OK (perc_tags);
  OBSIDIAN_CHECK_EQ ((*perc_tags)[0], "DET");
  OBSIDIAN_CHECK_EQ ((*perc_tags)[1], "ADJ");
  OBSIDIAN_CHECK_EQ ((*perc_tags)[2], "NOUN");
  OBSIDIAN_CHECK_EQ ((*perc_tags)[3], "VERB");

  NER_Span_Extractor ner;
  ner.add_gazetteer_entry ({"Google", "DeepMind"}, "ORG");
  ner.add_gazetteer_entry ({"London"}, "LOC");

  std::vector<std::string_view> ner_tokens = {
    "Researchers", "at", "Google", "DeepMind", "visited", "London"
  };
  auto spans_res = ner.extract_spans (ner_tokens);
  OBSIDIAN_CHECK_OK (spans_res);
  OBSIDIAN_CHECK (spans_res->size () >= 2u);
  OBSIDIAN_CHECK_EQ ((*spans_res)[0].label, "ORG");
  OBSIDIAN_CHECK_EQ ((*spans_res)[0].text, "Google DeepMind");
}

OBSIDIAN_TEST (edit_distance_and_sentence_segmentation)
{
  OBSIDIAN_CHECK_EQ (levenshtein_distance ("kitten", "sitting"), 3u);
  OBSIDIAN_CHECK_EQ (levenshtein_distance ("flaw", "lawn"), 2u);
  /* Adjacent transposition "ab" -> "ba" has Damerau distance 1 vs Lev 2.  */
  OBSIDIAN_CHECK_EQ (damerau_levenshtein_distance ("ca", "ac"), 1u);
  OBSIDIAN_CHECK_EQ (levenshtein_distance ("ca", "ac"), 2u);

  Sentence_Segmenter seg;
  std::string text =
    "Dr. Smith arrived at 3.14 p.m. in Washington. He met Mrs. Jones! "
    "Did they discuss C++23?";
  auto sents = seg.segment (text);
  OBSIDIAN_CHECK_EQ (sents.size (), 3u);
  OBSIDIAN_CHECK_EQ (sents[0],
                     "Dr. Smith arrived at 3.14 p.m. in Washington.");
  OBSIDIAN_CHECK_EQ (sents[1], "He met Mrs. Jones!");
  OBSIDIAN_CHECK_EQ (sents[2], "Did they discuss C++23?");
}

OBSIDIAN_TEST (ngram_language_model_tfidf_and_bm25)
{
  std::vector<std::string_view> corpus = {
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat chased the mouse"
  };

  NGram_Model lm (3, Smoothing_Method::Kneser_Ney, 0.75);
  OBSIDIAN_CHECK_OK (lm.fit (corpus));
  std::vector<std::string_view> ctx = {"the", "cat"};
  double p_sat = lm.probability (ctx, "sat");
  double p_rug = lm.probability (ctx, "rug");
  OBSIDIAN_CHECK (p_sat > p_rug);

  auto ppl_res = lm.perplexity (corpus);
  OBSIDIAN_CHECK_OK (ppl_res);
  OBSIDIAN_CHECK (*ppl_res > 1.0 && *ppl_res < 50.0);

  TFIDF_Vectorizer tfidf;
  auto mat_res = tfidf.fit_transform (corpus);
  OBSIDIAN_CHECK_OK (mat_res);
  OBSIDIAN_CHECK_EQ (mat_res->size (0), 3u);
  OBSIDIAN_CHECK_EQ (mat_res->size (1), tfidf.vocab ().size ());

  BM25_Index bm25;
  OBSIDIAN_CHECK_OK (bm25.index_documents (corpus));
  auto hits_res = bm25.search ("chased mouse", 2);
  OBSIDIAN_CHECK_OK (hits_res);
  OBSIDIAN_CHECK_EQ ((*hits_res)[0].doc_id, 2u);
  OBSIDIAN_CHECK ((*hits_res)[0].score > (*hits_res)[1].score);
}

int
main (void)
{
  return obsidian::testing::run_all_tests ("Phase 4: Classical NLP Suite");
}
