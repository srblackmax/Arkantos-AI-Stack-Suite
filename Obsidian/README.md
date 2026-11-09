# Obsidian: High-Performance C++23 Deep Learning & NLP Library

**Obsidian** is a from-scratch, native **C++23** Deep Learning and Natural
Language Processing library engineered for cache locality, SIMD throughput,
and zero-cost abstractions.  It provides the core capabilities of Hugging Face
Transformers, spaCy, NLTK, and Gensim through an original C++23 architecture
powered by the **Basalt Deep Learning** computational backend (`libbasalt`).

> **License Notice**: Obsidian is free software licensed under the **GNU
> General Public License v3.0 or later** (`GPL-3.0-or-later`; see `COPYING`).
> Obsidian is an independent open-source project and is **not** a GNU project.

---

## 1. Key Capabilities

1. **Basalt Tensor & Autograd Bridge (`<obsidian/tensor_bridge.hpp>`)**:
   - Validated `Tensor_Block` wrapper over `basalt::Tensor` returning
     `obsidian::Result<T>` (`std::expected<T, obsidian::Error>`).
   - `Sequence_Batch` collation, padding/causal/ALiBi attention mask
     builders, sinusoidal positional encodings, differentiable
     `masked_mean_pooling`, `last_token_pooling`, and
     `sequence_cross_entropy_loss` with `ignore_index` and label smoothing.

2. **SIMD-Accelerated Tokenization (`<obsidian/tokenizer.hpp>`)**:
   - **SIMD Pre-Tokenizer**: AVX2-accelerated whitespace and punctuation
     boundary scanner operating in `std::pmr` `Arena` buffers.
   - **Byte-Pair Encoding (`BPE_Tokenizer`)**: Flat open-addressed vocabulary
     and merge tables with stack/arena doubly-linked symbol merging—**zero
     dynamic heap allocations in the hot path** (`encode_into_arena`).
   - **WordPiece (`WordPiece_Tokenizer`)** and **Unigram (`Unigram_Tokenizer`)**
     Viterbi lattice tokenizers.

3. **Zero-Copy Memory-Mapped Word Embeddings (`<obsidian/embeddings.hpp>`)**:
   - **`Mmap_Embeddings`**: POSIX `mmap(2)` read-only vector store over
     64-byte-aligned `.obvec` binary files returning `std::span<const float>`
     in $O(1)$ time with zero copy, plus full GloVe text and Word2Vec
     text/binary import & export.
   - AVX2 SIMD cosine similarity, top-$K$ nearest neighbors, and vector
     analogy solver ($v_{a^*} - v_a + v_b$).
   - Native **Word2Vec** (Skip-Gram & CBOW with negative sampling) and
     **FastText** character n-gram subword embeddings.

4. **Classical NLP & Topic Modeling (`<obsidian/stemmer.hpp>`, `<obsidian/tagger.hpp>`, `<obsidian/classical.hpp>`, `<obsidian/topic.hpp>`)**:
   - Header-only **Martin Porter Stemmer** (5-step algorithm) and rule +
     irregular **Lemmatizer** satisfying the `Stemmer_Policy` concept.
   - Header-only log-space **`viterbi_decode`** template, trainable
     **`HMM_POS_Tagger`**, **Averaged Structured `Perceptron_POS_Tagger`**,
     and BIO/BIOES **`NER_Span_Extractor`**.
   - **N-gram Language Models** (MLE, Laplace, Interpolated Kneser-Ney),
     **TF-IDF Vectorizer**, **Okapi BM25 Inverted Index**, Punkt-inspired
     **`Sentence_Segmenter`**, and Levenshtein / Damerau-Levenshtein distance.
   - **Latent Dirichlet Allocation (`LDA_Model`)** via collapsed Gibbs sampling
     and **Latent Semantic Analysis (`LSA_Model`)** via truncated SVD on Basalt
     tensors.

5. **Transformer Architectures & Pipelines (`<obsidian/transformer.hpp>`, `<obsidian/models.hpp>`, `<obsidian/pipeline.hpp>`)**:
   - **Fused-QKV `Grouped_Query_Attention`** (MHA / GQA / MQA), **Rotary
     Position Embeddings (RoPE)**, **ALiBi**, **SwiGLU / GeGLU** gated
     feed-forward blocks, and `RMS_Norm` / `Layer_Norm` residual stacks.
   - **`Encoder_Model`** (BERT-style), **`Sequence_Classifier`**,
     **`Token_Classifier`**, **`Causal_LM`** (GPT/LLaMA-style with incremental
     `KV_Cache`, Greedy, Temperature, Top-K, Top-P Nucleus, Repetition Penalty,
     and Beam Search decoding), and **`Seq2Seq_Model`**.
   - C++23 composable pipe syntax (`text | stage_1 | stage_2`) and high-level
     `Linguistic_Pipeline`, `Text_Classification_Pipeline`, and
     `Text_Generation_Pipeline`.

6. **Production Hardening & Ecosystem Interoperability (`<obsidian/safetensors.hpp>`, `<obsidian/unicode.hpp>`)**:
   - **Zero-Copy SafeTensors (`SafeTensors_File`)**: POSIX `mmap(2)` reader and
     writer for `.safetensors` checkpoints (`F32`, `F64`, `I64`, `I32`, `F16`,
     `BF16`) with `save_module_safetensors` and `load_module_safetensors` for
     any `basalt::nn::Module`.
   - **Byte-Level BPE & Unicode NFKC (`<obsidian/unicode.hpp>`)**: Reversible
     256-byte-to-Unicode GPT-2/LLaMA mapping (`set_byte_level(true)`), NFKC
     compatibility folding, Hugging Face `tokenizer.json` import/export, and
     OpenMP multi-threaded `encode_batch_parallel`.
   - **Inference Safety (`infer` / `infer_batch`)**: RAII `No_Grad_Guard` +
     `eval()` inference methods and `Causal_LM::generate_batch`.

---

## 2. Dependencies and Build Instructions

### Prerequisites
- **Compiler**: GCC 13+ (`g++` supporting `-std=c++23` and `<expected>`).
- **Primary Dependency**: **Basalt Deep Learning** (`libbasalt`, alongside its
  storage backends `Bellows` and `Hyper Math`).
- **Build System**: GNU Autotools (`autoconf`, `automake`, `libtool`) or CMake
  3.20+.

### Building with GNU Autotools (Recommended)
```bash
autoreconf -fi
./configure --with-basalt="/path/to/Basalt Deep Learning"
make -j$(nproc)
make check
sudo make install
```

### Building with AddressSanitizer and UndefinedBehaviorSanitizer
```bash
./configure --enable-sanitizers
make clean && make -j$(nproc) && make check
```

---

## 3. Pure C++23 Usage Example

```cpp
#include <obsidian/obsidian.hpp>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

int
main (void)
{
  using namespace obsidian;

  /* 1. Train a BPE Tokenizer and encode in a stack-backed PMR Arena.  */
  std::vector<std::string_view> corpus = {
    "obsidian builds transformers on basalt tensors",
    "simd tokenization and memory mapped embeddings"
  };
  auto bpe = std::make_shared<BPE_Tokenizer> ();
  bpe->train_from_corpus (corpus, 64, 1);

  Arena arena;
  std::span<const int64_t> ids = bpe->encode_into_arena (
    "obsidian builds transformers", arena, true, true);
  std::cout << "Decoded: " << bpe->decode (ids, true).value () << "\n";

  /* 2. Run Classical Linguistic Annotation (Sentences, POS, Lemmas, NER).  */
  Linguistic_Pipeline nlp;
  nlp.add_entity_gazetteer ({"Obsidian"}, "LIBRARY");
  Document_Annotation doc = nlp.annotate (
    "Obsidian processes text rapidly.").value ();
  std::cout << "Tokens: " << doc.tokens.size ()
            << ", Entities: " << doc.entities.size () << "\n";

  /* 3. Build a RoPE + SwiGLU + GQA Causal Transformer on Basalt.  */
  Transformer_Config cfg;
  cfg.d_model = 32;
  cfg.num_heads = 4;
  cfg.num_kv_heads = 2;
  cfg.d_ff = 64;
  cfg.num_layers = 2;
  cfg.pos_encoding = Position_Encoding_Kind::Rotary_RoPE;
  cfg.ffn_activation = FFN_Activation_Kind::SwiGLU;

  auto lm = std::make_shared<Causal_LM> (bpe->vocab_size (), cfg);
  Text_Generation_Pipeline generator (bpe, lm);

  Generation_Config gen_cfg;
  gen_cfg.max_new_tokens = 5;
  gen_cfg.do_sample = true;
  gen_cfg.top_k = 4;
  gen_cfg.top_p = 0.9f;

  Result<std::string> out = generator.generate_text ("obsidian", gen_cfg);
  if (out.has_value ())
    {
      std::cout << "Generated: " << *out << "\n";
    }
  return 0;
}
```
