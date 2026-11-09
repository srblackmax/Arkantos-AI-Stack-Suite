# Obsidian Implementation Plan and Task Tracker (`TASKS.md`)

This document tracks the autonomous development phases of **Obsidian**
following the **Obsidian Development Loop** (`Plan -> Code -> Build -> Test ->
Audit -> Refactor`).

---

## Phase 1: Project Skeleton, Build System, and Basalt Tensor Bridge
- [x] Create project directory structure and `COPYING` (GPLv3).
- [x] Author `AGENTS.md`, `SKILL.md`, `ARCHITECTURE.md`, and `TASKS.md`.
- [x] Configure GNU Autotools (`configure.ac`, `Makefile.am`) and
      `CMakeLists.txt` linking against Basalt Deep Learning (`libbasalt`).
- [x] Implement `tests/check_gnu_style.cpp` (native C++ GNU Coding Standards
      verifier: checks GPLv3 header, no tabs, no trailing spaces, <= 79 cols,
      and Doxygen `@file`/`@brief` presence) and `tests/test_harness.hpp`.
- [x] Implement core foundation (`include/obsidian/config.hpp`,
      `include/obsidian/version.h.in`, `include/obsidian/error.hpp` with
      `std::expected`, `include/obsidian/memory.hpp` with `std::pmr` Arena,
      `include/obsidian/simd.hpp` with AVX2/SSE4.2 + scalar kernels).
- [x] Implement Basalt-integrated tensor wrapper and NLP tensor operations
      (`include/obsidian/tensor_bridge.hpp`, `src/tensor/tensor_bridge.cpp`).
- [x] Build and verify `tests/test_core_tensor` and `tests/check_gnu_style`.

## Phase 2: High-Throughput SIMD Tokenizer Suite
- [x] Implement SIMD-accelerated UTF-8 & whitespace/punctuation pre-tokenizer
      using `std::pmr::monotonic_buffer_resource` (zero hot-path allocations).
- [x] Implement high-throughput Byte-Pair Encoding (`BPE_Tokenizer`) with
      flat open-addressed merge table, BPE training from corpus, batch
      encoding to `Sequence_Batch` (`basalt::Tensor`), and decoding.
- [x] Implement `WordPiece_Tokenizer` (greedy longest-match with `##` prefix)
      and `Unigram_Tokenizer` (Viterbi lattice minimum-loss segmentation).
- [x] Build and verify `tests/test_tokenizer`.

## Phase 3: Memory-Mapped Word Embeddings & Subword Vectors
- [x] Implement POSIX `mmap(2)` read-only embedding store (`Mmap_Embeddings`)
      with 64-byte aligned `.obvec` binary format and $O(1)$ zero-copy
      `std::span<const float>` and `basalt::Tensor` views.
- [x] Implement GloVe text and Word2Vec text/binary format importers and
      exporters.
- [x] Implement SIMD-accelerated `cosine_similarity`, top-K `most_similar`, and
      vector analogy solver ($b - a + c$).
- [x] Implement `Word2Vec_Trainer` (Skip-Gram and CBOW with negative sampling)
      and `FastText_Embeddings` (character n-gram subword hashing).
- [x] Build and verify `tests/test_embeddings`.

## Phase 4: Classical NLP Algorithms (Header-Only Templates & Engines)
- [x] Implement header-only `Porter_Stemmer` (full 5-step algorithm) and
      `Lemmatizer` (`include/obsidian/stemmer.hpp`).
- [x] Implement header-only log-space `viterbi_decode` template, trainable
      `HMM_POS_Tagger`, `Perceptron_POS_Tagger` (averaged structured
      perceptron), and `NER_Span_Extractor` (`include/obsidian/tagger.hpp`).
- [x] Implement `NGram_Model` (MLE, Laplace, Interpolated Kneser-Ney),
      `TFIDF_Vectorizer`, `BM25_Index`, `levenshtein_distance` &
      `damerau_levenshtein_distance`, and Punkt-style `Sentence_Segmenter`
      (`include/obsidian/classical.hpp`, `src/classical/classical.cpp`).
- [x] Build and verify `tests/test_classical`.

## Phase 5: Topic Modeling and Semantic Analysis
- [x] Implement `LDA_Model` (Latent Dirichlet Allocation via collapsed Gibbs
      sampling with sparse count tables and unseen document inference).
- [x] Implement `LSA_Model` (Latent Semantic Analysis via truncated SVD powered
      by `basalt::Tensor` matrix operations).
- [x] Build and verify `tests/test_topic`.

## Phase 6: Basalt-Backed Transformer Blocks & Attention Architectures
- [x] Implement policy-configurable attention modules on `basalt::nn::Module`:
      Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query
      Attention (MQA) with Fused-QKV projection, RoPE, ALiBi, and `KV_Cache`.
- [x] Implement SwiGLU, GeGLU, and GELU feed-forward layers, Pre-Norm /
      Post-Norm `RMS_Norm` and `Layer_Norm` residual blocks, and stacks
      (`include/obsidian/transformer.hpp`, `src/transformer/transformer.cpp`).
- [x] Build and verify `tests/test_transformer` (including `autograd` backward
      pass verification).

## Phase 7: End-to-End NLP Models, Generation Strategies, and C++23 Pipelines
- [x] Implement `Encoder_Model` (BERT-style), `Sequence_Classifier`,
      `Token_Classifier`, `Causal_LM` (GPT/LLaMA-style), and `Seq2Seq_Model`.
- [x] Implement text generation engine supporting Greedy, Temperature, Top-K,
      Nucleus (Top-P), Repetition Penalty, and Beam Search decoding.
- [x] Implement composable C++23 NLP pipelines (`include/obsidian/pipeline.hpp`,
      `src/pipeline/pipeline.cpp`) using range/pipe syntax (`|`).
- [x] Implement CLI tool (`src/cli/obsidian_nlp.cpp`), showcase example
      (`examples/nlp_showcase.cpp`), and benchmark suite
      (`bench/bench_obsidian.cpp`).
- [x] Write comprehensive `README.md`.
- [x] Run full Sanitizer (`-fsanitize=address,undefined`), `make check`,
      `make install` verification, and GNU Coding Standards audit.

## Phase 8: Production Hardening & Ecosystem Interoperability
- [x] Implement `include/obsidian/unicode.hpp` and `src/core/unicode.cpp`
      (UTF-8 codepoint iteration, NFKC compatibility normalization, fullwidth
      ASCII folding, control/zero-width stripping, and GPT-2/LLaMA reversible
      256-byte-to-Unicode encoder/decoder).
- [x] Implement `include/obsidian/safetensors.hpp` and
      `src/tensor/safetensors.cpp` (POSIX `mmap(2)` SafeTensors reader with
      native C++23 JSON header parser, `F32`/`F64`/`I64`/`I32`/`F16`/`BF16`
      tensor deserialization, `.safetensors` writer, and
      `save_module_safetensors` / `load_module_safetensors` for
      `basalt::nn::Module`).
- [x] Upgrade `BPE_Tokenizer` with Byte-Level BPE (`set_byte_level(true)`),
      Hugging Face `tokenizer.json` import/export (`load_hf_tokenizer_json`,
      `save_hf_tokenizer_json`), and OpenMP multi-threaded parallel batch
      encoding (`encode_batch_parallel`).
- [x] Upgrade `Encoder_Model`, `Sequence_Classifier`, `Token_Classifier`,
      `Causal_LM`, and `Seq2Seq_Model` with RAII `No_Grad_Guard` + `eval()`
      inference methods (`infer`, `infer_batch`, `infer_seq2seq`) and batched
      autoregressive generation (`Causal_LM::generate_batch`).
- [x] Create `tests/test_production.cpp`, wire into `Makefile.am` and
      `CMakeLists.txt`, and verify 9/9 test suites pass under both Release and
      Sanitizer (`-fsanitize=address,undefined`) builds with 0 GNU style
      violations.
