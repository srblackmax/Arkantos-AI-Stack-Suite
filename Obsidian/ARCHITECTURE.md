# Obsidian Architecture Document (`ARCHITECTURE.md`)

> **Project**: Obsidian High-Performance C++23 Deep Learning & NLP Library
> **License**: GPLv3 or later (Non-GNU Project)
> **Backend**: Basalt Deep Learning (`libbasalt`)

---

## 1. Architectural Vision

**Obsidian** unifies modern deep neural NLP (Transformers, subword tokenizers,
autoregressive decoding with KV caching) and classical statistical/symbolic
NLP (stemming, HMM/Viterbi and Perceptron POS tagging, NER, TF-IDF/BM25,
n-gram language models, LDA/LSA topic modeling, and memory-mapped word
embeddings) in a single zero-overhead C++23 library.

Unlike Python frameworks that bridge disparate C/C++/Rust extensions across a
slow interpreter boundary, Obsidian provides a **single-address-space,
composable C++23 pipeline** where raw UTF-8 text flows through SIMD
pre-tokenizers and subword encoders directly into contiguous `basalt::Tensor`
views and policy-configured Transformer architectures without intermediate
heap churn.

```
+-----------------------------------------------------------------------------+
|                        Obsidian C++23 Public API                            |
|                     (Concepts, Ranges, Pipelines, CLI)                      |
+-----------------------------------------------------------------------------+
|  Tokenization     |  Embeddings       |  Classical NLP   |  Transformers    |
|  - SIMD Scanner   |  - Mmap Store     |  - Porter Stem   |  - MHA/GQA/RoPE  |
|  - BPE / WordPiece|  - GloVe/Word2Vec |  - Viterbi POS   |  - ALiBi / SwiGLU|
|  - Unigram        |  - FastText Hash  |  - Averaged Perc |  - Encoder (BERT)|
|  - PMR Arena      |  - SkipGram/CBOW  |  - TF-IDF / BM25 |  - Decoder (GPT) |
|  - Vocab & Batch  |  - SIMD Top-K     |  - N-gram / LDA  |  - Beam / Nucleus|
+-----------------------------------------------------------------------------+
|                  Obsidian Tensor & Sequence Bridge Layer                    |
|     (Tensor_View, Sequence_Batch, Masking, Loss, Autograd Extensions)       |
+-----------------------------------------------------------------------------+
|                     Basalt Deep Learning Backend                            |
|      (basalt::Tensor, basalt::ops, basalt::autograd, basalt::nn::Module)    |
+-----------------------------------------------------------------------------+
```

---

## 2. Core Design Principles

1. **Idiomatic C++23 with Concepts & Policy Templates**:
   - Components are parameterized via C++20/23 `concept` constraints
     (`Token_Range`, `Stemmer_Policy`, `Scoring_Policy`,
     `Position_Encoding_Policy`, `Activation_Policy`, `Norm_Policy`).
   - Classical NLP algorithms (Porter stemmer, Viterbi trellis decoder,
     edit distance, n-gram evaluation) are header-only templates enabling
     aggressive compiler inlining and constant propagation.

2. **Recoverable Error Handling via `std::expected`**:
   - Public operations that parse external files, validate user arguments, or
     process untrusted text return `obsidian::Result<T>`
     (`std::expected<T, obsidian::Error>`).
   - `obsidian::Error` carries an `Error_Code` enum, source location, and
     diagnostic message without throwing exceptions on expected user-input
     errors and never aborting the process.

3. **Zero-Allocation Hot Paths (`std::pmr` & `mmap`)**:
   - `obsidian::Arena` wraps `std::pmr::monotonic_buffer_resource` over a
     caller-supplied or thread-local stack/heap buffer so BPE merging and
     sentence segmentation perform zero system allocator calls in steady state.
   - `obsidian::Mmap_Embeddings` uses POSIX `mmap(2)` (`MAP_PRIVATE`) to map
     binary word vector tables directly from disk into virtual memory,
     returning `std::span<const float>` in $O(1)$ time with zero copy.

4. **SIMD Acceleration**:
   - `obsidian::simd` provides runtime- and compile-time-dispatched `AVX2` /
     `SSE4.2` routines (with clean portable scalar fallbacks) for:
     - Fast whitespace and ASCII punctuation boundary scanning (32 bytes/cycle).
     - UTF-8 validation and codepoint counting.
     - Fused dot product, L2 norm, and cosine similarity for embedding search.

---

## 3. Module Breakdown and Directory Layout

```
Obsidian/
├── COPYING                     # GPLv3 license text
├── AGENTS.md                   # Autonomous agent operating charter
├── SKILL.md                    # Obsidian Development Loop specification
├── ARCHITECTURE.md             # High-level system architecture
├── TASKS.md                    # Phased implementation plan & tracker
├── README.md                   # Build instructions & pure C++ usage guide
├── configure.ac                # GNU Autoconf specification
├── Makefile.am                 # GNU Automake build rules
├── CMakeLists.txt              # Modern CMake build configuration
├── include/obsidian/           # Public Doxygen-documented C++23 headers
│   ├── obsidian.hpp            # Umbrella header
│   ├── version.h.in            # Autoconf-configured version macros
│   ├── config.hpp              # Compiler, visibility, and SIMD macros
│   ├── error.hpp               # Error_Code, Error, and std::expected Result<T>
│   ├── memory.hpp              # PMR Arena and aligned buffer utilities
│   ├── simd.hpp                # SIMD string scanning & vector math kernels
│   ├── tensor_bridge.hpp       # Basalt Tensor wrapper, Sequence_Batch, masks
│   ├── tokenizer.hpp           # BPE, WordPiece, Unigram, SIMD Pre-Tokenizer
│   ├── embeddings.hpp          # Mmap_Embeddings, GloVe/W2V IO, Word2Vec/FastText
│   ├── stemmer.hpp             # Header-only Porter & Porter2 stemmers, Lemmatizer
│   ├── tagger.hpp              # Header-only Viterbi decoder, HMM & Perceptron POS, NER
│   ├── classical.hpp           # N-gram LM, TF-IDF, BM25, Edit Distance, Punkt SBD
│   ├── topic.hpp               # LDA (Gibbs sampling) & LSA (Truncated SVD)
│   ├── transformer.hpp         # Policy-based Attention (MHA/GQA/RoPE/ALiBi) & Blocks
│   ├── models.hpp              # Encoder (BERT), Causal LM (GPT/LLaMA), Seq2Seq, Generation
│   └── pipeline.hpp            # C++23 composable NLP pipelines & range adaptors
├── src/
│   ├── core/                   # Version, error formatting, SIMD & memory pools
│   ├── tensor/                 # Basalt tensor bridge, sequence padding, NLP losses
│   ├── tokenizer/              # High-throughput BPE, WordPiece, Unigram implementations
│   ├── embeddings/             # POSIX mmap store, GloVe/W2V parsers, SkipGram/CBOW
│   ├── classical/              # TF-IDF, BM25, LDA, LSA, corpus utilities
│   ├── transformer/            # Basalt-backed attention, RoPE, ALiBi, KV-cache layers
│   ├── models/                 # Encoder, Decoder, Seq2Seq, Beam/Nucleus decoding
│   ├── pipeline/               # High-level text classification, QA, & generation pipelines
│   └── cli/                    # obsidian-nlp command-line inspection & benchmark tool
├── tests/                      # Unit tests & C++ GNU coding style checker
│   ├── test_harness.hpp        # Lightweight header-only C++23 test framework
│   ├── check_gnu_style.cpp     # Native C++ GNU Coding Standards verifier
│   ├── test_core_tensor.cpp    # Phase 1: Core, Error, SIMD, & Basalt Tensor bridge
│   ├── test_tokenizer.cpp      # Phase 2: BPE, WordPiece, Unigram, SIMD pre-tokenizer
│   ├── test_embeddings.cpp     # Phase 3: Mmap embeddings, GloVe/W2V, FastText, analogies
│   ├── test_classical.cpp      # Phase 4: Stemmer, Viterbi POS, NER, TF-IDF, BM25, EditDist
│   ├── test_topic.cpp          # Phase 5: LDA, LSA, N-gram smoothing
│   ├── test_transformer.cpp    # Phase 6: MHA, GQA, RoPE, ALiBi, Encoder/Decoder blocks
│   └── test_models_pipeline.cpp# Phase 7: BERT/GPT models, Generation, End-to-End Pipelines
├── examples/
│   └── nlp_showcase.cpp        # Pure C++23 end-to-end NLP & Transformer demonstration
└── bench/
    └── bench_obsidian.cpp      # Throughput benchmarks (Tokenizer, Mmap, Attention)
```

---

## 4. Detailed Subsystem Design

### 4.1 Basalt Tensor & Autograd Bridge (`include/obsidian/tensor_bridge.hpp`)
- Wraps and interoperates zero-copy with `basalt::Tensor`, `basalt::Device`,
  `basalt::Dtype`, and `basalt::nn::Module`.
- Provides `Sequence_Batch`: contiguous `(batch_size, max_seq_len)` `Int64`
  token ID tensor paired with an attention mask tensor and sequence lengths.
- Provides NLP-specialized differentiable operations built on `basalt::ops`:
  - `cross_entropy_sequence_loss` with `ignore_index` (e.g., `-100` padding).
  - `masked_mean_pooling` and `last_token_pooling` for sentence embeddings.
  - `build_causal_mask`, `build_padding_mask`, and `build_alibi_slopes`.
  - `gather_token_embeddings` and weight-tying utilities between input
    embeddings and output language modeling heads.

### 4.2 High-Throughput Tokenizer (`include/obsidian/tokenizer.hpp`)
- **SIMD Pre-Tokenizer**: Scans UTF-8 byte streams 32 bytes at a time using
  AVX2 (`_mm256_cmpeq_epi8` / range comparisons) to split text on whitespace
  and punctuation boundaries without allocating `std::string` substrings,
  producing `std::span<const std::string_view>` slices in a `std::pmr` arena.
- **Byte-Pair Encoding (BPE)**:
  - Stores vocabulary token-to-ID mappings and merge pair ranks
    `(uint32_t left_id, uint32_t right_id) -> uint32_t rank` in flat
    open-addressed hash tables with power-of-two capacity and Fibonacci hashing.
  - Performs BPE merges over a pre-allocated doubly-linked symbol array in a
    stack/PMR buffer—zero `new`/`malloc` calls in the hot path.
- **WordPiece & Unigram Tokenizers**:
  - Greedy longest-prefix matching (`##` continuation subwords) for BERT-style
    models.
  - Viterbi minimum-loss lattice segmentation over log-probabilities for
    SentencePiece/Unigram models.

### 4.3 Memory-Mapped Word Embeddings (`include/obsidian/embeddings.hpp`)
- **Binary Mmap Format (`.obvec`) & Word2Vec/GloVe Compatibility**:
  - Supports loading standard GloVe text files, Word2Vec binary/text files, and
    Obsidian's native 64-byte-aligned `.obvec` memory-mapped binary format.
  - `Mmap_Embeddings` maps the file via `mmap(2)` with `PROT_READ` and
    `MAP_PRIVATE`.  Looking up a word hashes the `std::string_view` in an
    open-addressed index and returns a `std::span<const float>` pointing
    directly into the mapped OS page cache ($O(1)$ time, 0 bytes copied).
  - Can also expose the entire embedding matrix as a zero-copy `basalt::Tensor`
    via `basalt::from_blob`.
- **SIMD Vector Queries & Training**:
  - AVX2-accelerated `cosine_similarity`, `most_similar` (top-K min-heap), and
    vector analogy solver ($v_b - v_a + v_c$).
  - Online Word2Vec (Skip-Gram & CBOW with negative sampling) and FastText
    character n-gram subword embedding table.

### 4.4 Classical NLP (`include/obsidian/stemmer.hpp`, `tagger.hpp`, `classical.hpp`, `topic.hpp`)
- **Porter Stemmer**: Complete 5-step Martin Porter algorithm implemented as a
  header-only template operating in-place on character buffers or returning
  views, plus dictionary/rule Lemmatizer.
- **Sequence Tagging (POS & NER)**:
  - Generic header-only `viterbi_decode` template parameterized by transition
    and emission score functors, operating in log-space over a contiguous flat
    trellis buffer.
  - Trainable `HMM_Tagger` (with add-$k$ smoothing), `Perceptron_Tagger`
    (Collins averaged structured perceptron with orthographic suffix/prefix
    features), and rule + BIO-tag `NER_Recognizer`.
- **Information Retrieval & Language Modeling**:
  - `TFIDF_Vectorizer` and `BM25_Index` with inverted index and SIMD scoring.
  - `NGram_Model` supporting MLE, Laplace, Good-Turing / Witten-Bell, and
    Interpolated Kneser-Ney smoothing with perplexity evaluation.
  - Bit-parallel / SIMD Levenshtein and Damerau-Levenshtein edit distance.
  - Punkt-style unsupervised sentence boundary detector.
- **Topic Modeling**:
  - `LDA_Model` using collapsed Gibbs sampling with sparse topic-word counts.
  - `LSA_Model` using randomized / power-iteration truncated SVD bridged with
    `basalt::Tensor`.

### 4.5 Transformer Blocks & Deep NLP Models (`include/obsidian/transformer.hpp`, `models.hpp`)
- Built directly as `basalt::nn::Module` subclasses so parameters, gradients,
  `state_dict` serialization, optimizer updates (`basalt::optim::AdamW`), and
  CPU/CUDA device transfers work seamlessly.
- **Configurable Policy-Based Attention**:
  - Supports Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and
    Multi-Query Attention (MQA).
  - Supports Rotary Position Embeddings (RoPE), ALiBi linear biases, or
    learned/sinusoidal absolute positional embeddings.
  - Integrates incremental `KV_Cache` for $O(1)$ per-step token generation.
- **Feed-Forward & Normalization**:
  - Supports standard MLP (`Linear -> GELU -> Linear`) and gated SwiGLU /
    GeGLU (`(Swish(xW_g) * xW_u) W_d`) with `Layer_Norm` or `RMS_Norm`.
- **End-to-End Models & Decoding**:
  - `Encoder_Model` (BERT-style), `Sequence_Classifier`, `Token_Classifier`.
  - `Causal_LM` (GPT/LLaMA-style) with `generate ()` and `generate_batch ()`
    supporting Greedy, Temperature, Top-K, Top-P (Nucleus), Repetition
    Penalty, and Beam Search.
  - `Seq2Seq_Model` (Encoder-Decoder Transformer).
  - Explicit `infer ()` / `infer_batch ()` methods that automatically guard
    execution with `No_Grad_Guard` and `eval ()` mode.

### 4.6 Production Hardening & Ecosystem Interoperability (`include/obsidian/safetensors.hpp`, `unicode.hpp`)
- **Zero-Copy SafeTensors Serialization (`SafeTensors_File`)**:
  - Native POSIX `mmap(2)` reader and binary writer for the Hugging Face
    `.safetensors` format (`F32`, `F64`, `I64`, `I32`, `F16`, `BF16`).
  - Validates header bounds, JSON structure, offset monotonicity, and shape
    byte counts without external JSON dependencies.
  - Direct `save_module_safetensors` and `load_module_safetensors` for any
    `basalt::nn::Module`.
- **Unicode Normalization, Byte-Level BPE & Parallel Batch Tokenization**:
  - `unicode::normalize_nfkc_text` for canonicalizing UTF-8 punctuation,
    fullwidth ASCII (`U+FF01..U+FF5E`), ligatures, and control characters.
  - GPT-2 / LLaMA reversible 256-byte-to-Unicode mapping (`byte_level = true`)
    eliminating `<unk>` tokens on arbitrary UTF-8 and binary sequences.
  - Hugging Face `tokenizer.json` loader and exporter.
  - OpenMP multi-threaded `encode_batch_parallel` using per-thread
    `thread_local_arena ()` instances for lock-free batch tokenization.
