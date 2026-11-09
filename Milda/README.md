# Milda — Esperanto & Multilingual C++23 LLM Inference Engine

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)
[![CMake](https://img.shields.io/badge/CMake-3.28%2B-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](https://cmake.org/)
[![CUDA / AVX2](https://img.shields.io/badge/Compute-CUDA%2012%20%7C%20AVX2%2B%20FMA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Debian 13 GNU/Linux](https://img.shields.io/badge/Platform-Debian%2013%20GNU%2FLinux-A81D33?style=for-the-badge&logo=debian&logoColor=white)](https://www.debian.org/)
[![License: GNU GPLv3+](https://img.shields.io/badge/License-GPLv3%2B-blue?style=for-the-badge&logo=gnu&logoColor=white)](./LICENSE)

---

## Overview / Superrigardo

### English
**Milda** is an ultra-low-overhead, bare-metal **C++23** conversational AI inference engine built on top of `llama.cpp` / `ggml` and formatted strictly according to the GNU Coding Standards with Allman bracing. Designed from the ground up for Esperanto (`eo`) and polyglot (`eo`, `en`, `es`, `zh-Hans`, `ru`) conversational fluency, Milda features:

- **Native Esperanto REPL Controls:** Interactive slash commands (`/eliri`, `/vakigi`, `/konservi`, `/sxargi`, `/stat`, `/lang`, `/x-system`) and real-time in-place `x`-systemorthographic transliteration (`cx` → `ĉ`, `gx` → `ĝ`, `hx` → `ĥ`, `jx` → `ĵ`, `sx` → `ŝ`, `ux` → `ŭ`).
- **Zero-Copy `mmap` Weight Streaming & GPU-Gap Eviction:** Maps GGUF weights via POSIX `mmap` with `MADV_HUGEPAGE` / `POSIX_FADV_WILLNEED` pre-faulting, and automatically evicts GPU-offloaded tensor byte ranges from the host Linux page cache (`POSIX_FADV_DONTNEED`) so large hybrid MoE models stream from system RAM without swap or disk thrashing.
- **12-Physical-Core Hybrid CPU Affinity Pinning (1200% Saturation):** Scans `/sys/devices/system/cpu/` topology to isolate primary physical **P-cores** and physical **E-cores** (`[8 P-cores, 4 E-cores]`, zero SMT twin overlap), combined with active OpenMP spin-waiting (`OMP_WAIT_POLICY=ACTIVE`).
- **Asymmetric MoE & FFN GPU-CPU Offloading:** Pins Multi-Head Attention, Gated DeltaNet, shared experts, and `q8_0` KV caches in fast GPU VRAM while selectively offloading sparse Mixture-of-Experts (`--n-cpu-moe` / `-ncmoe`) or dense Feed-Forward layers (`--n-cpu-ffn` / `-ncffn`) to DDR4/DDR5 system RAM and physical CPU cores.
- **Special-Token Chain-of-Thought Suppression:** Tokenizes control templates with `parse_special = true`, pre-fills closed reasoning channels (`<|turn>model\n<|channel>thought\n<channel|>` for **Gemma 4** and `<|im_start|>assistant\n<think>\n\n</think>\n` for **Qwen3**), and filters streaming output so models reply immediately in natural Esperanto without leaking internal English Chain-of-Thought reasoning.

### Esperanto
**Milda** estas ultrarapida, malalt-superkosta **C++23** konversacia AI-motoro konstruita sur `llama.cpp` / `ggml` laŭ la normoj de GNU kaj Allman-stilo. Kreita unuavice por flua, idiomeca konversacio en **Esperanto** (`eo`) kaj plurlingva uzo (`eo`, `en`, `es`, `zh-Hans`, `ru`), Milda inkluzivas:

- **Denaskajn Esperantajn REPL-komandojn:** Interagajn komandojn (`/eliri`, `/vakigi`, `/konservi`, `/sxargi`, `/stat`, `/lang`, `/x-system`) kaj tujan surlokan konvertadon de la `x`-sistemo (`cx` → `ĉ`, `gx` → `ĝ`, `hx` → `ĥ`, `jx` → `ĵ`, `sx` → `ŝ`, `ux` → `ŭ`) sen dinamikaj memor-asignoj.
- **Senkopian `mmap`-Pezofluon kaj Inteligente Fiksitan KV-Kaŝmemoron:** Ŝargas GGUF-pezojn per POSIX `mmap` kun `MADV_HUGEPAGE` / `POSIX_FADV_WILLNEED`, kaj konservas la sisteman instrukcion (`k_esperanto_system_prompt`) ŝlosita en la KV- kaj rekurenta stato-kaŝmemoro (`n_keep_`).
- **Hibridan 12-Fizik-Kernan CPU-Fiksadon (1200% Plena Eluzo):** Aŭtomate analizas `/sys/devices/system/cpu/` por apartigi fizikajn **P-kernojn** kaj **E-kernojn** sen SMT-konkurenco.
- ** Nesimetrian GPU+CPU- Tavol-Distribuon (`--n-cpu-moe`, `--n-cpu-ffn`):** Tenas la atento-tavolojn kaj `q8_0`-kvantigitan KV-kaŝmemoron en rapida GPU-VRAM dum ĝi fluigas pezan MoE- aŭ FFN-kalkuladon tra sistema RAM kaj ĉiuj fizikaj CPU-kernoj.
- ** silentigon de Internaj Pens-Kanaloj (Gemma 4 & Qwen3):** Preterpasas internajn anglalingvajn pens-blokojn (`<|channel>thought...<channel|>` kaj `<think>...</think>`) por ke la modelo respondu tuj kaj nature en flua Esperanto.

---

## Key Engineering Features / Ĉefaj Inĝenieraj Trajtoj

1. **Dual Build Profiles (`build` vs `build-cpu`)**
   - **Hybrid GPU+CPU Workstation Mode (`-DMILDA_ENABLE_CUDA=ON`):** Links `ggml-cuda` for hardware-accelerated Attention, Flash Attention, and KV cache execution on NVIDIA GPUs alongside multi-core CPU MoE/FFN streaming.
   - **Standalone Portable CPU-Only Mode (`-DMILDA_ENABLE_CUDA=OFF -DMILDA_PORTABLE_CPU=ON`):** Produces a 100% self-contained x86_64 Linux binary with **zero** dynamic or transitive dependencies on `libggml-cuda`, `libcudart`, `libcublas`, or `libcuda`, ready for VirtualBox VMs (e.g., Debian 13 GNU/Linux guests with 6 vCPUs and 8 GB RAM) and portable Linux machines.

2. **Asymmetric MoE & FFN Hybrid Offloading**
   - Inspects GGUF header metadata (`general.architecture`, `.block_count`, `.expert_count`) and live GPU VRAM metrics prior to model initialization.
   - Keeps Multi-Head Attention layers, shared experts, norms, and `q8_0` quantized KV caches pinned inside an 8 GB VRAM envelope while routing `--n-cpu-moe` (`-ncmoe`) expert tensors (`blk.*.ffn_(up|down|gate|gate_up)_(ch|)exps`) or `--n-cpu-ffn` (`-ncffn`) dense feed-forward tensors (`blk.*.ffn_(up|down|gate).`) to host DDR4/DDR5 RAM and CPU compute.
   - Scans `/proc/self/maps` after model load and issues `posix_fadvise(..., POSIX_FADV_DONTNEED)` on unmapped GPU tensor byte ranges so GPU-resident weights do not consume host page-cache RAM.

3. **Hybrid CPU Topology Pinning (1200% Saturation)**
   - Parses `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` and `cpufreq/cpuinfo_max_freq` to build an ordered physical core list (`all_physical_cpus = [8 P-cores, 4 E-cores]` on Intel Alder Lake / Raptor Lake) while gracefully falling back to flat vCPU topologies inside VirtualBox VMs.
   - Configures `ggml` threadpools with `poll = 100`, `OMP_WAIT_POLICY=ACTIVE`, and `GOMP_SPINCOUNT=30000000`, and elevates scheduling priority via `setpriority(PRIO_PROCESS, 0, -20)` when `cap_sys_nice` is granted.

4. **Native Gemma 4 & Qwen3 Thought-Channel Suppression**
   - Invokes `llama_tokenize` with `parse_special = true` so structural chat tokens (`<|im_start|>`, `<|im_end|>`, `<|turn>`, `<turn|>`, `<|channel>`, `<channel|>`) are encoded as single control token IDs.
   - Pre-fills `<|turn>model\n<|channel>thought\n<channel|>` on **Gemma 4** models and `<|im_start|>assistant\n<think>\n\n</think>\n` on **Qwen3** models, paired with a zero-heap-allocation streaming state machine (`thought_stream_filter`) and post-turn KV-cache excision of any suppressed reasoning tokens.

5. **Persistent Cross-Session KV Memory & System Prompt Checkpointing**
   - Evaluates and locks the system prompt into the KV and recurrent/SSM state cache once at startup (`n_keep_` + `pinned_sys_state_` up to `128 MiB`), allowing `/vakigi` to reset conversations instantaneously without re-evaluating system tokens.
   - Supports full binary KV-cache session persistence via `-s <file>`, `/konservi <file>`, and `/sxargi <file>`.

---

## Build Instructions / Konstru-Instrukcioj

### Prerequisites / Antaŭkondiĉoj
- **OS:** Debian 13 GNU/Linux (or any modern x86_64 Linux distribution)
- **Compiler:** GCC 13+ or Clang 17+ with C++23 support
- **Build System:** CMake 3.28+
- **Optional (Workstation GPU Mode):** NVIDIA CUDA Toolkit 12+ and NVIDIA driver

### 1. CUDA Hybrid Workstation Build (`build/`)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMILDA_ENABLE_CUDA=ON \
  && cmake --build build -j 12
```

### 2. Standalone Portable CPU-Only Build (`build-cpu/`)
Produces a self-contained binary with zero CUDA `.so` dependencies (ideal for VirtualBox VMs or CPU-only systems):
```bash
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DMILDA_ENABLE_CUDA=OFF -DMILDA_PORTABLE_CPU=ON \
  && cmake --build build-cpu -j 6
```

### 3. Optional Hardware Priority & RAM Lock Capability Grant
To allow Milda to elevate thread scheduling priority (`nice -20`) on the 12 physical cores and lock pages in RAM (`--mlock`) without running as `root`:
```bash
sudo setcap cap_sys_nice,cap_ipc_lock+ep ./build/milda
```

### 4. Run the Verification Suite (`ctest`)
```bash
ctest --test-dir build --output-on-failure -V
ctest --test-dir build-cpu --output-on-failure -V
```

---

## Recommended Models & Verified Launch Profiles / Rekomenditaj Modeloj kaj Lanĉ-Profiloj

| Profile / Profilo | Recommended Model (`GGUF`) | Hardware Target | Offload & Context Strategy | Typical Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Workstation Hybrid MoE (30B)** | `Qwen3-30B-A3B-Q4_K_M.gguf` | RTX 3070 (8 GB) + 48 GB DDR4 + i7-12700K | `-ngl 48 --n-cpu-moe 36 -c 16384` (`q8_0` KV) | **25–38 tok/s** |
| **Workstation Hybrid MoE (80B)** | `Qwen3-Coder-Next-Q4_K_S.gguf` | RTX 3070 (8 GB) + 48 GB DDR4 + i7-12700K | `-ngl 99 --n-cpu-moe -1 -c 16384` (`q8_0` KV) | **12–18 tok/s** |
| **Workstation Pure VRAM Speed** | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` / `Hermes-3-Llama-3.1-8B.Q4_0.gguf` | RTX 3070 (8 GB) + i7-12700K | `-ngl 99 -c 16384` (100% VRAM + `q8_0` KV) | **65–80+ tok/s** |
| **Workstation Literary Dense Hybrid** | `gemma-4-31B-it-Q4_K_M.gguf` | RTX 3070 (8 GB) + 48 GB DDR4 + i7-12700K | `-ngl 18 -c 4096` (`q8_0` KV) | **6–10 tok/s** |
| **Virtual Machine / 8 GB RAM CPU-Only** | `gemma-4-E4B-it-Q4_K_M.gguf` / `gemma-3-4b-it-Q4_K_M.gguf` | 6 vCPUs + 8 GB RAM (VirtualBox / CPU-Only) | `-t 6 -ngl 0 -c 4096` (`q8_0` KV) | **14–22 tok/s** |

### Copy-Paste Launch Commands

#### 1. Workstation Hybrid MoE Saturation (`Qwen3-30B-A3B-Q4_K_M.gguf`)
```bash
./build/milda \
  -m $HOME/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 48 --n-cpu-moe 36 -c 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -t 12 --threads-batch 12 --mlock --stat
```

#### 2. Workstation 80B MoE Hybrid Saturation (`Qwen3-Coder-Next-Q4_K_S.gguf`)
```bash
./build/milda \
  -m $HOME/models/Qwen3-Coder-Next-Q4_K_S.gguf \
  -ngl 99 --n-cpu-moe -1 -c 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -t 12 --threads-batch 12 -fa \
  -s ~/.milda_qwen3coder.session --stat
```

#### 3. Workstation Pure VRAM Speed (`Qwen2.5-7B-Instruct-Q4_K_M.gguf` / `Hermes-3-Llama-3.1-8B.Q4_0.gguf`)
```bash
./build/milda \
  -m $HOME/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 -c 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -t 8 --threads-batch 12 --mlock --stat
```

#### 4. Workstation Literary Dense Hybrid (`gemma-4-31B-it-Q4_K_M.gguf`)
```bash
./build/milda \
  -m $HOME/models/gemma-4-31B-it-Q4_K_M.gguf \
  -ngl 18 -c 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -t 12 --threads-batch 12 --mlock --stat
```

#### 5. Virtual Machine / 8 GB RAM CPU-Only (`gemma-4-E4B-it-Q4_K_M.gguf` & `gemma-3-4b-it-Q4_K_M.gguf`)
```bash
./build-cpu/milda \
  -m $HOME/models/gemma-4-E4B-it-Q4_K_M.gguf \
  -t 6 -ngl 0 -c 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 --stat
```

> **Tip (Clean Conversation Mode):** Append `2>/dev/null` and omit `--stat` to suppress all hardware initialization banners and per-turn telemetry so only your `Vi >` and `Milda >` conversation appears on screen.

---

## Interactive REPL Commands / Interagaj Komandoj (`Komandoj`)

Inside the interactive `Vi >` prompt, Milda supports the following built-in commands:

| Command / Komando | Description (English) | Priskribo (Esperanto) |
| :--- | :--- | :--- |
| `/eliri` | Exit the REPL cleanly (auto-saves session if `-s` was provided). | Eliri el la programo (kaj aŭtomate konservi la seancon se `-s` estas aktiva). |
| `/vakigi` | Clear conversation history in-place while preserving the pinned system prompt checkpoint (`n_keep_`). | Vakigi la konversacian historion en la KV-kaŝmemoro, konservante la fiksitan sisteman instrukcion (`n_keep_`). |
| `/konservi <dosiero>` | Save the active KV/recurrent state and token history to `<dosiero>` (defaults to `-s` path if omitted). | Konservi la nunan KV-memoron kaj konversacion al `<dosiero>`. |
| `/sxargi <dosiero>` | Load a previously saved KV/recurrent session file from `<dosiero>`. | Ŝargi antaŭe konservitan KV-memoron el `<dosiero>`. |
| `/stat` | Print live hardware telemetry for the last turn: Prompt Processing (`PP tok/s`), Time-To-First-Token (`TTFT ms`), Token Generation (`TG tok/s`), KV cache occupancy (`cached / n_ctx`, `fiksitaj`), active CPU threads (`P-kernoj + E-kernoj`), GPU/CPU hybrid layer split, and VRAM / Linux RSS (`/proc/self/statm`) / Pinned host memory. | Montri detalan aparataran telemetrion: `PP tok/s`, `TTFT ms`, `TG tok/s`, KV-kaŝmemoran uzadon, aktivajn CPU-fadenojn (`P-kernoj + E-kernoj`), GPU/CPU-tavolojn, kaj VRAM / RSS / Pinned memoron. |
| `/lang <auto\|eo\|en\|es\|zh\|zh-Hans\|ru>` | Switch the active system prompt and language-specific sampler parameters (`repeat_penalty`, `top_p`) at runtime. | Ŝanĝi la aktivan lingvon kaj sistem-instrukcion dum rulado. |
| `/x-system <auto\|on\|off>` | Configure Esperanto `x`-system input transliteration (`auto` enables it for Esperanto prompts only). | Agordi la Esperantan `x`-sisteman konvertadon (`auto`, `on`, `off`). |

---

## Authorship, Acknowledgments & License / Aŭtoreco, Dankoj kaj Permesilo

### Author / Aŭtoro
- Architected and maintained by **Kival**.

### Acknowledgments / Dankoj
- Built in modern **C++23** with architectural assistance from **Google Antigravity** and **Gemini**.
- Powered by the MIT-licensed [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and [`ggml`](https://github.com/ggml-org/ggml) tensor library ecosystem created by Georgi Gerganov and contributors.
- Tested and validated with Google's open-weights **Gemma 4** & **Gemma 3** model families and Alibaba Cloud's **Qwen** (`Qwen2.5`, `Qwen3`, `Qwen3-Coder-Next`) open-weights models.

### License / Permesilo
**Milda** is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (GPLv3+)**.

See the root [`LICENSE`](./LICENSE) file for the full verbatim text of the GNU General Public License v3.0.
