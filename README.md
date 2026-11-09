# Arkantos AI Stack Suite

**Pure Native C++20/C++23 High-Performance AI, Cognitive Agent, MLOps & Scientific Computing Stack**  
**Author:** Kival  
**License:** GNU General Public License v3.0 or later (`GPL-3.0-or-later`) with Section 7 Runtime & Hardware Compute Library Exception

---

## 1. Executive Summary

The **Arkantos AI Stack Suite** is a vertically integrated, zero-Python-runtime C++20/C++23 software stack engineered for deterministic, low-latency local AI inference, autonomous ReAct software engineering, real-time voice interaction, cryptographic MLOps lineage, vector/graph storage, computer vision, and scientific computing on Linux workstations.

Every component in the suite is written in strict conformance with the **GNU Coding Standards** (Allman brace formatting, `<= 79` columns per line, zero tabs, zero trailing whitespace, verified via automated `check_gnu_style` test suites) and exposes clean C++ headers with zero leaked third-party backend symbols.

> **Complete Reference Manual:** See **[`DOCUMENTATION.md`](./DOCUMENTATION.md)** for the exhaustive layer-by-layer specification of every feature, mathematical formulation, C++20/C++23 API, hardware optimization, and architectural decision rationale across all 11 components and the `stack_integration_test` harness.

---

## 2. System Architecture

```text
+-----------------------------------------------------------------------------+
|                           ARKANTOS AI STACK SUITE                           |
+-----------------------------------------------------------------------------+
|                                                                             |
|  +-----------------------------------+  +--------------------------------+  |
|  |              SPHINX               |  |             MILDA              |  |
|  |  Tri-Route Cognitive Agent &      |  |  Polyglot 5-Language C++23     |  |
|  |  Real-Time Voice/Text Assistant   |  |  LLM Engine (EO/EN/ES/ZH/RU)   |  |
|  |  [CHAT | BIG_CHAT | TASK (ReAct)] |  |  In-Place KV Shift & Telemetry |  |
|  +-----------------+-----------------+  +---------------+----------------+  |
|                    |                                    |                   |
|        +-----------+-----------+------------------------+                   |
|        |                       |                                            |
|        v                       v                                            |
|  +-----------+           +-----------+           +-----------------------+  |
|  |   SUTRA   |           |   VACH    |           |       DARSHANA        |  |
|  | Agentic   |           | Real-Time |           | Zero-Dependency C++23 |  |
|  | Runtime,  |           | Audio I/O,|           | SVG Scientific &      |  |
|  | Tool Call |           | SileroVAD,|           | Telemetry Plotting    |  |
|  | & Hybrid  |           | Whisper   |           | Engine                |  |
|  | MoE LLM   |           | STT, Piper|           +-----------+-----------+  |
|  +-----+-----+           +-----------+                       |              |
|        |                                                     |              |
|        +-------------------------+---------------------------+              |
|                                  |                                          |
|                                  v                                          |
|  +-----------------------------------------------------------------------+  |
|  |                                YANTRA                                 |  |
|  |   Content-Addressed Model Registry (SHA-256 + BLAKE3), Append-Only    |  |
|  |   WAL Experiment Tracker, DAG Pipeline Orchestrator & HNSW Vector DB  |  |
|  +-----------------------------------+-----------------------------------+  |
|                                      |                                      |
|        +-----------------------------+-----------------------------+        |
|        |                             |                             |        |
|        v                             v                             v        |
|  +------------+               +-------------+               +------------+  |
|  | OBSIDIANDB |               |    SURYA    |               |   BASALT   |  |
|  | Classical  |               | High-Speed  |               | Autograd,  |  |
|  | & Neural   |               | Computer    |               | Neural Net |  |
|  | NLP, BPE   |               | Vision,     |               | Layers,    |  |
|  | Tokenizer, |               | SIMD Filter,|               | Optimizers |  |
|  | Safetensor |               | Detection   |               | & Training |  |
|  +-----+------+               +------+------+               +-----+------+  |
|        |                             |                            |         |
|        +--------------+--------------+--------------+-------------+         |
|                       |                             |                       |
|                       v                             v                       |
|                +-------------+               +-------------+                |
|                |   BELLOWS   |               | HYPER MATH  |                |
|                | Classical   |-------------->| N-D Tensors,|                |
|                | ML, GBDT,   |               | AVX2 SIMD,  |                |
|                | Estimators  |               | SciPy/FFT,  |                |
|                | & Pipelines |               | DataFrames  |                |
|                +-------------+               +-------------+                |
+-----------------------------------------------------------------------------+
```

---

## 3. Suite Components

| Component | Language | Role & Key Capabilities |
| :--- | :--- | :--- |
| **`Sphinx`** | C++23 | Autonomous Tri-Route (`CHAT`, `BIG_CHAT`, `TASK`) voice and text assistant with Yantra-supervised DAG execution, persistent semantic/episodic/correction memory, and sandboxed file/shell/SSH engineering tools. |
| **`Sutra`** | C++23 | Core agentic LLM framework: BNF grammar-constrained tool calling, ReAct state machine, hybrid GPU+CPU MoE weight placement, Linux page-cache residency management, and physical-core `ggml_threadpool` pinning. |
| **`Milda`** | C++23 | High-velocity polyglot (`eo`, `en`, `es`, `zh-Hans`, `ru`) conversational engine with zero-allocation Esperanto `x`-system transliteration, pinned `n_keep` KV cache ring-buffer shifting, and live hardware telemetry. |
| **`Vach`** | C++23 | Real-time lock-free audio I/O (`ALSA`/`PipeWire`), Silero VAD v5, isolated-symbol CUDA Whisper STT (`libwhisper.so`), and streaming `espeak-ng` + ONNX Piper TTS with `< 200 ms` barge-in latency. |
| **`Yantra`** | C++23 | Cryptographic MLOps & storage substrate: SQLite/WAL content-addressed model registry (`SHA-256` + `BLAKE3`), lock-free telemetry ring buffer, SIMD HNSW vector index, and topological DAG runner. |
| **`Darshana`** | C++23 | Zero-dependency publication-grade SVG plotting library generating session latency distributions, confusion matrices, and multi-series scientific charts. |
| **`ObsidianDb`** | C++20 | Native NLP & text analytics engine: Unicode normalization, BPE/WordPiece tokenizers, BM25/TF-IDF, LDA topic modeling, Safetensors zero-copy loader, and Transformer encoder inference. |
| **`Surya`** | C++23 | Computer vision & image processing library: AVX2 morphological/convolution kernels, Canny/Sobel/Harris feature extraction, NMS object detection pipelines, and zero-copy tensor bridges. |
| **`Basalt`** | C++20 | Deep learning & autograd engine: dynamic tape-based reverse-mode automatic differentiation, `nn::Linear`/`Conv2d`/`LayerNorm`/`MultiHeadAttention`, mixed-precision (`AMP`), and `AdamW`/cosine schedulers. |
| **`Bellows`** | C++20 | Classical machine learning toolkit: gradient-boosted decision trees (GBDT), random forests, regularized generalized linear models, preprocessing transformers, and composable pipelines. |
| **`Hyper Math`** | C++20 | Foundational scientific computing library: slab-allocated N-D tensors, AVX2/FMA linear algebra (`SVD`, `QR`, `Cholesky`), FFT/signal processing, ODE solvers, and columnar lazy DataFrames. |

---

## 4. Hardware-Sympathetic Hybrid MoE & CPU Topology Engineering

### 4.1 Reference Hardware Profile (`Silver`)
- **CPU:** 12th Gen Intel Core i7-12700K — **12 Physical Cores** (`8 Golden Cove P-cores` @ 2 SMT threads `[0..15]` + `4 Gracemont E-cores` @ 1 thread `[16..19]` = 20 logical CPUs)
- **RAM:** 48 GiB Dual-Channel DDR4 (`~40.3 GiB/s` sustained page-cache read bandwidth)
- **GPU:** NVIDIA GeForce RTX 3070 (`8 GiB GDDR6 VRAM`, Compute Capability `8.6`)
- **OS:** Debian GNU/Linux 13 (trixie), Linux Kernel `6.12`

### 4.2 12-Physical-Core Topology Pinning (`1200%` CPU Utilization)
On hybrid Alder Lake / Raptor Lake architectures, running 20 threads across 12 physical cores causes SMT sibling contention on the 8 P-cores, while linking `libggml-cpu.so` with `GGML_OPENMP=ON` creates a second OpenMP threadpool that collides with `ggml_threadpool`.

`Sutra` and `Milda` solve both bottlenecks:
1. **Sysfs Topology Discovery (`parse_linux_physical_core_cpus`)**: Parses `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` at runtime to select exactly **1 logical CPU per physical core** (`[0, 2, 4, 6, 8, 10, 12, 14]` across the 8 P-cores + `[16, 17, 18, 19]` across the 4 E-cores = **12 physical cores**).
2. **Unified `ggml_threadpool` with Strict Affinity**: Attaches a single shared `ggml_threadpool` (`n_threads = 12`, `n_threads_batch = 12`, `strict_cpu = true`, `poll = 100`, `prio = GGML_SCHED_PRIO_NORMAL`) for both prompt processing and token generation, pausing worker threads between turns (`0%` idle CPU) and sustaining **`1182%–1208%` (`~1200%`) CPU utilization** during active inference.

### 4.3 Warm-Start Linux Page-Cache Preservation (`0.95s` Preload, `MAJFL = 0`)
When running `Qwen3-Coder-Next-80B-A3B-Q4_K_S.gguf` (`42.35 GiB`) on a system with `48 GiB` RAM and `8 GiB` VRAM:
- `3.84 GiB` of attention, shared-expert, and output norms reside in RTX 3070 VRAM (`n_gpu_layers = 99`, `n_cpu_moe = 45`).
- `38.51 GiB` of MoE routed expert weights (`blk.0..44.ffn_*_exps.weight`) plus CPU-resident input embeddings (`token_embd.weight`) reside in host RAM via `mmap`.

Previously, `POSIX_FADV_SEQUENTIAL` during `llama_model_load_from_file` triggered aggressive kernel readahead across the entire `42.35 GiB` file while CUDA buffers were being allocated, evicting cached MoE pages and causing a `182s` disk reload on warm restarts.

`Sutra` eliminates this thrashing through a three-stage page-cache protocol:
1. **`POSIX_FADV_NORMAL` Initialization**: Opens the GGUF `mmap` with `POSIX_FADV_NORMAL` / `POSIX_MADV_NORMAL` so reading the `3.84 GiB` of GPU tensors does not disturb cached CPU MoE pages.
2. **Selective GPU Span Eviction**: Immediately after CUDA upload, calls `madvise(MADV_DONTNEED)` and `posix_fadvise(POSIX_FADV_DONTNEED)` on the exact page-aligned file byte ranges of GPU-resident tensors (`3.84 GiB` freed from host page cache in `24 ms`), while preserving `token_embd`, `token_types`, and `rope_freqs` in `cpu_spans`.
3. **`MADV_POPULATE_READ` Warm Verification**: Pre-populates all `38.51 GiB` of CPU spans in **`0.95 s` (`40.33 GiB/s`)** on warm starts, achieving **`MAJFL = 0`** (zero major page faults) across all subsequent inference turns.

---

## 5. Tri-Route Cognitive Dispatch (`Sphinx`)

`Sphinx` routes every user turn across three specialized execution paths and supports deterministic prefix overrides (`/chat `, `/big `, `/task `):

| Route | Prefix Override | Active Model | Execution Mode | Measured Performance (`Silver`) |
| :--- | :--- | :--- | :--- | :--- |
| **`CHAT`** | `/chat <prompt>` | `Qwen2.5-1.5B-Instruct-Q4_K_M` (100% GPU) | Ultra-fast conversational & system-profile responses | **`211.53 tok/s`** (`TTFS = 246.97 ms`) |
| **`BIG_CHAT`** | `/big <prompt>` | `Qwen3-Coder-Next-80B-A3B-Q4_K_S` (GPU + 12C MoE) | Direct streaming synthesis for math, physics, history, philosophy, and architecture **with zero tool schemas and zero ReAct loop** | **`22.62 tok/s`** (`MAJFL = 0`, `~1200%` CPU) |
| **`TASK`** | `/task <prompt>` | `Qwen3-Coder-Next-80B-A3B-Q4_K_S` (GPU + 12C MoE) | Full ReAct engineering agent with grammar-constrained tool calls (`read_file`, `write_file`, `list_dir`, `run_command`, `run_readonly`, `ssh_exec`) | **`22.35 tok/s`** + sub-millisecond native tool dispatch |

---

## 6. Building, Installing & Verifying the Suite

### 6.1 Automated Full-Stack Build
Use the included `build_all.sh` script (or build components individually in dependency order) to compile, test, and install all libraries and binaries into `$HOME/.local`:

```bash
export PREFIX="$HOME/.local"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"

# 1. Hyper Math
cd "Hyper Math" && ./configure --prefix="$PREFIX" && make -j"$(nproc)" && make check && make install && cd ..

# 2. Bellows
cd Bellows && make -j"$(nproc)" PREFIX="$PREFIX" HYPER_MATH_DIR="$(pwd)/../Hyper Math" && make check && make install PREFIX="$PREFIX" && cd ..

# 3. Basalt Deep Learning
cd "Basalt Deep Learning" && ./configure --prefix="$PREFIX" && make -j"$(nproc)" && make check && make install && cd ..

# 4. ObsidianDb
cd Obsidian && ./configure --prefix="$PREFIX" && make -j"$(nproc)" && make check && make install && cd ..

# 5. Surya
cmake -S Surya -B Surya/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build Surya/build -j"$(nproc)" && ctest --test-dir Surya/build --output-on-failure && cmake --install Surya/build

# 6. Darshana
cmake -S Darshana -B Darshana/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build Darshana/build -j"$(nproc)" && ctest --test-dir Darshana/build --output-on-failure && cmake --install Darshana/build

# 7. Yantra
cmake -S Yantra -B Yantra/build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build Yantra/build-release -j"$(nproc)" && ctest --test-dir Yantra/build-release --output-on-failure && cmake --install Yantra/build-release

# 8. Sutra
cmake -S Sutra -B Sutra/build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build Sutra/build-release -j"$(nproc)" && ctest --test-dir Sutra/build-release --output-on-failure && cmake --install Sutra/build-release

# 9. Vach
cmake -S Vach -B Vach/build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build Vach/build-release -j"$(nproc)" && ctest --test-dir Vach/build-release --output-on-failure && cmake --install Vach/build-release

# 10. Sphinx
cmake -S Sphinx -B Sphinx/build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build Sphinx/build/release -j"$(nproc)" && ctest --test-dir Sphinx/build/release --output-on-failure && cmake --install Sphinx/build/release

# 11. Milda
cmake -S Milda -B Milda/build -DCMAKE_BUILD_TYPE=Release -DMILDA_NATIVE_OPT=ON -DMILDA_ENABLE_LTO=ON -DMILDA_USE_CUDA_BACKEND=ON
cmake --build Milda/build -j"$(nproc)" && ctest --test-dir Milda/build --output-on-failure
```

### 6.2 Running `Sphinx` in Text Mode (`sphinx-text`)
```bash
sphinx --text --eager-big --preload-all --threads 12 --threads-batch 12
```

### 6.3 Running `Milda` in Polyglot Mode
```bash
milda -m "$HOME/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --lang auto --stat
```

---

## 7. Licensing

Copyright (C) 2026 **Kival**.

All components in the **Arkantos AI Stack Suite** are free software distributed under the terms of the **GNU General Public License version 3 or later (`GPL-3.0-or-later`)**, with an explicit **Section 7 Runtime & Hardware Compute Library Exception** permitting linking and distribution with NVIDIA CUDA (`libcudart`, `libcublas`, `libcuda`), Vulkan, OpenGL/GLES, SDL3, FreeType/HarfBuzz, ONNX Runtime, Whisper.cpp, `llama.cpp`/`ggml`, and system audio runtimes. See [`LICENSE`](./LICENSE) for the complete license text.
