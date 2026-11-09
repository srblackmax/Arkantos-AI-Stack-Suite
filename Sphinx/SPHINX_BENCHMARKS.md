# SPHINX Benchmarks (`SPHINX_BENCHMARKS.md`)

## 1. Reference Workstation & Model Footprint

| Component | Specification |
|-----------|---------------|
| **CPU** | 24-thread x86_64 Linux workstation (`6.12.113+deb13-amd64`), AVX2 + FMA |
| **GPU** | NVIDIA GeForce RTX 3070 (8192 MiB GDDR6, CUDA 12.4) |
| **Audio Subsystem** | ALSA (`libasound.so.2`) via PipeWire-ALSA (`16 kHz` mono capture, `48 kHz` stereo playback) |
| **Big Task Model** | `Qwen3-30B-A3B-Q4_K_M.gguf` (`qwen3-30b-a3b-q4km` v1, 18.56 GB, 48 layers, 128 experts / 8 active, `big_model_n_cpu_moe = 32` / 40 MoE layers in CPU RAM, SHA-256 verified via Yantra `ModelRegistry`; `qwen2.5-7b-instruct-q4km` retained in registry) |
| **Router & Chat Model** | `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (`qwen2.5-1.5b-q4km` v1, 986 MB, SHA-256 verified via Yantra `ModelRegistry`) |
| **STT Model** | `whisper.cpp` `ggml-base.en.bin` (148 MB, CUDA + 8 P-core threads) |
| **TTS Voice** | Piper ONNX `en_US-bryce-medium.onnx` (61 MB, 22050 Hz -> 48000 Hz stereo resampled) |
| **Embedding Model (Memory)** | `Qwen3-Embedding-0.6B-Q8_0.gguf` (`qwen3-embedding-0.6b-q8` v1, 610 MB, 310 tensors, 1024 dims, CPU-only `n_gpu_layers = 0`, `< 25 ms` per query, `0.00 GB` VRAM impact) |
| **Combined Resident VRAM** | **6.64–6.89 GB peak** at `big_model_n_cpu_moe = 32` (`<= 7.20 GB` target; down from `7.17 GB` bench / `7.46 GB` live at `n_cpu_moe = 28`; `Qwen3-30B-A3B` attention/shared layers + 8 GPU MoE layers + 8192 KV cache + `Qwen2.5-1.5B` router + `whisper.cpp` resident simultaneously on 8 GB RTX 3070; `~14.06 GiB` MoE experts + `610 MB` embedding model in 46 GiB host RAM) |

---

## 2. Phase 1 — Vach Audio, STT, and TTS Benchmarks

| Benchmark | Target | Measured | Status |
|-----------|--------|----------|--------|
| **ALSA Capture Period / Buffer** | `16 kHz` mono `S16_LE`, 20 ms / 80 ms | `320` frames / `1280` frames exact | **PASS** |
| **ALSA Playback Period / Buffer** | `48 kHz` stereo `S16_LE`, 10 ms / 40 ms | `480` frames / `1920` frames exact | **PASS** |
| **1-Hour Ring Buffer Stress (`57.6M` samples)** | `0` overruns, `0` underruns, `xruns <= 5` | `0` overruns, `0` underruns, `0` xruns | **PASS** |
| **STT 5 s Utterance Latency (`ggml-base.en.bin`)** | `<= 500 ms` | **`69.2 ms` mean** (`57.3–81.0 ms` range) | **PASS** |
| **STT 10-Utterance Accuracy (WER)** | `< 10.0%` WER | **`0.0%` WER** (`80 / 80` words exact) | **PASS** |
| **TTS 20-Word Sentence TTFS (`en_US-bryce-medium`)** | `<= 300 ms` | **`<= 250 ms`** (synthesized + pushed to ring) | **PASS** |
| **Clean Shutdown (`/proc/self/task` & `/dev/snd/*`)** | `0` orphan threads, `0` leaked fds | `0` orphan threads, `0` leaked fds (`< 20 ms` join) | **PASS** |

---

## 3. Phase 2 — LLM-Backed Router (`sphinx::TurnRouter`) Benchmarks

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| **100-Call Router Latency (p50)** | `<= 50.0 ms` | **`13.21 ms`** | **PASS** |
| **100-Call Router Latency (p99)** | `<= 100.0 ms` | **`14.10 ms`** | **PASS** |
| **20-Turn Benchmark Accuracy** | `>= 90.0%` (18/20) | **`100.0%` (20/20)** | **PASS** |
| **4 Ambiguous Turns Accuracy** | `>= 75.0%` | **`100.0%` (4/4)** | **PASS** |
| **Confusion Matrix** | Recorded in `SPHINX.md` | `TrueCHAT=10, FalseTASK=0, FalseCHAT=0, TrueTASK=10` | **PASS** |

---

## 4. Phase 5 — Darshana Session SVG Rendering Benchmarks

| SVG Artifact | Target | Measured (`sphinx_bench`) | XML Well-Formedness |
|--------------|--------|---------------------------|---------------------|
| `<session_id>.svg` (Turn Latency Breakdown) | `<= 500 ms` | **`0.181 ms`** (`0.107 ms` in `bench-ci`) | **PASS** |
| `<session_id>-latency.svg` (TTFS Distribution) | `<= 500 ms` | **`0.052 ms`** (`0.035 ms` in `bench-ci`) | **PASS** |
| `<session_id>-router.svg` (Router Distribution) | `<= 500 ms` | **`0.027 ms`** (`0.019 ms` in `bench-ci`) | **PASS** |

---

## 5. Phase 6 — End-to-End Aliveness Suite (`sphinx_bench`)

Executed via `./build/release/sphinx_bench --idle-ms 60000 --json-out build/release/bench_results.json` and logged to Yantra `ExperimentStore` at `~/sphinx/sessions/sphinx_aliveness/`:

| Aliveness Metric | Target Budget | Measured (`60 s` Full Run) | Measured (`bench-ci` Run) | Status |
|------------------|---------------|----------------------------|---------------------------|--------|
| **`TTFS-CHAT` p50 (20 turns)** | `<= 700.0 ms` | **`167.06 ms`** | **`156.68 ms`** | **PASS** |
| **`TTFS-CHAT` p99 (20 turns)** | `<= 1000.0 ms` | **`284.99 ms`** | **`267.38 ms`** | **PASS** |
| **`TTFS-TASK` p50 (10 turns)** | `<= 1500.0 ms` | **`502.64 ms`** | **`495.92 ms`** | **PASS** |
| **`TTFS-TASK` p99 (10 turns)** | `<= 2500.0 ms` | **`738.33 ms`** | **`768.84 ms`** | **PASS** |
| **`BARGE-IN` max (5 events)** | `<= 200.0 ms` | **`0.028 ms`** | **`0.028 ms`** | **PASS** |
| **`BACKCHANNEL` latency** | `<= 500.0 ms` | **`58.78 ms`** | **`80.99 ms`** | **PASS** |
| **`AUDIO-CONTINUITY` max gap** | `<= 100.0 ms` | **`13.99 ms`** | **`0.063 ms`** | **PASS** |
| **`IDLE-CPU` (60 s silence)** | `<= 5.0%` | **`3.56%`** (60 s) | **`0.09%`** (5 s) | **PASS** |

---

## 6. Phase 7 & 8 — Qwen3-30B-A3B MoE Swap (`n_cpu_moe = 28` & `n_cpu_moe = 32`) vs. Qwen2.5-7B Baseline

Executed via `./build/release/sphinx_bench --headless --json-out build/release/bench_qwen3_30b_moe32.json`:

| Aliveness Metric | Target Budget | `Qwen2.5-7B-Instruct-Q4_K_M` Baseline | `Qwen3-30B-A3B-Q4_K_M` (`n_cpu_moe = 28`) | `Qwen3-30B-A3B-Q4_K_M` (`n_cpu_moe = 32` + Identity Fix) | Status |
|------------------|---------------|---------------------------------------|-------------------------------------------|----------------------------------------------------------|--------|
| **Router Load Time** | Recorded | `0.45 s` | `0.46 s` | **`0.47 s`** | **PASS** |
| **Big Model Load Time** | Recorded | `1.12 s` | `3.18 s` | **`6.67 s`** | **PASS** |
| **Peak VRAM (`nvidia-smi`)** | `<= 7.20 GB` (`<= 7.50 GB` max) | `5.82 GB` | `7.17 GB` (`7.46 GB` live) | **`6.89 GB`** | **PASS** |
| **`TTFS-CHAT` p50 (20 turns)** | `<= 250.0 ms` (`<= 700.0 ms` max) | `167.06 ms` | `199.61 ms` | **`221.90 ms`** | **PASS** |
| **`TTFS-CHAT` p99 (20 turns)** | `<= 1000.0 ms` | `284.99 ms` | `927.32 ms` | **`913.89 ms`** | **PASS** |
| **`TTFS-TASK` p50 (10 turns)** | `<= 950.0 ms` (`<= 1500.0 ms` max) | `502.64 ms` | `790.88 ms` | **`777.13 ms`** | **PASS** |
| **`TTFS-TASK` p99 (10 turns)** | `<= 2500.0 ms` | `738.33 ms` | `1150.64 ms` | **`1264.29 ms`** | **PASS** |
| **`TASK` Decode Throughput (p50)** | `>= 21.0 tok/s` (`>= 20.0` floor) | `66.40 tok/s` | `24.02 tok/s` | **`24.77 tok/s`** | **PASS** |
| **`BARGE-IN` max (5 events)** | `<= 200.0 ms` | `0.028 ms` | `0.020 ms` | **`0.025 ms`** | **PASS** |
| **`BACKCHANNEL` latency** | `<= 500.0 ms` | `58.78 ms` | `0.475 ms` | **`78.86 ms`** | **PASS** |
| **`ROUTER-ACCURACY` (20 turns)** | `>= 95.0%` (`>= 90.0%` floor) | `100.0%` (`13.21 ms` p50) | `100.0%` (`13.45 ms` p50) | **`100.0%` (`13.06 ms` p50, `14.08 ms` p99)** | **PASS** |
| **`AUDIO-CONTINUITY` max gap** | `<= 100.0 ms` | `13.99 ms` | `0.0625 ms` | **`0.0625 ms`** | **PASS** |
| **`IDLE-CPU` (60 s silence)** | `<= 5.0%` | `3.56%` | `0.097%` | **`0.111%`** | **PASS** |

---

## 7. Phase 9 — Cross-Session Memory Benchmarks (20 Pre-Populated Synthetic Memories)

Executed via `./build/release/sphinx_bench --idle-ms 5000 --json-out build/release/bench_results.json --headless` with 20 synthetic memories (`SEMANTIC`, `EPISODIC`, `CORRECTION`) pre-populated in `yantra::WalEngine` + `yantra::HnswIndex`:

| Metric | Target Budget | Measured (`sphinx_bench` with Memory Enabled) | Status |
|--------|---------------|-----------------------------------------------|--------|
| **Single-Sentence CPU Embedding Latency (`embed()`)** | `< 50.0 ms` | **`19.4 ms`–`24.8 ms`** (1024-d unit vector on CPU) | **PASS** |
| **Similar Pair Cosine (`"the user's name is Kival"` vs `"the user is called Kival"`)** | `> 0.80` | **`0.87`** | **PASS** |
| **Unrelated Pair Cosine (`"the user's name is Kival"` vs `"the weather is sunny"`)** | `< 0.50` | **`0.23`** | **PASS** |
| **60-Day Episodic Decay (`exp(-60 / 30)`)** | `~0.135` | **`0.1353`** | **PASS** |
| **Peak VRAM (`nvidia-smi`)** | `<= 7.20 GB` (unchanged) | **`6.64 GB`** (`0.00 GB` GPU VRAM added by CPU embedding model) | **PASS** |
| **`TTFS-CHAT` p50 (20 turns, memory enabled)** | `<= 350.0 ms` | **`320.37 ms` p50** (`809.94 ms` p99 `<= 1000 ms`) | **PASS** |
| **`TTFS-TASK` p50 (10 turns, memory enabled)** | `<= 900.0 ms` | **`843.83 ms` p50** (`1056.32 ms` p99 `<= 2500 ms`) | **PASS** |
| **`TASK` Decode Throughput (p50)** | `>= 20.0 tok/s` | **`24.87 tok/s` p50** | **PASS** |
| **`ROUTER-ACCURACY` (20 turns)** | `>= 95.0%` | **`100.0%` (20/20)** (`12.75 ms` p50, `13.53 ms` p99) | **PASS** |
| **`BARGE-IN` max (5 events)** | `<= 200.0 ms` | **`0.015 ms` max** | **PASS** |
| **`BACKCHANNEL` latency** | `<= 500.0 ms` | **`85.00 ms`** | **PASS** |
| **`AUDIO-CONTINUITY` max gap** | `<= 100.0 ms` | **`0.0625 ms` max gap** | **PASS** |
| **`IDLE-CPU` (5 s silence)** | `<= 5.0%` | **`0.090%`** | **PASS** |

---

## 8. Phase 10–12 — Trust Fixes, System Observation, & Comprehension Fixes Benchmarks

Executed via `./build/release/sphinx_bench --idle-ms 5000 --headless --json-out build/release/bench_comprehension.json` with startup system profile (`<= 1000` bytes), `run_readonly`, `read_file_system`, `PlanOutcome` hard refusal (`NoPlan`), Fibonacci benchmark trigger restriction, 21-keyword read-intent expansion, `kChatGroundingClause`, and stateful verification routing enabled:

| Metric | Target Budget | Measured (`sphinx_bench` Post-Comprehension Fixes) | Status |
|--------|---------------|----------------------------------------------------|--------|
| **Session Startup Profile Gathering** | `<= 1500 ms` (`<= 1000 B`) | **`87–97 ms`** (`< 1000 B`) | **PASS** |
| **Router Load Time** | Recorded | **`0.656 s`** | **PASS** |
| **Big Model Load Time** | Recorded | **`15.53 s`** | **PASS** |
| **Peak VRAM (`nvidia-smi`)** | `<= 7.20 GB` (`0.00 GB` delta) | **`6.61 GB`** (`<= 7.20 GB`) | **PASS** |
| **`TTFS-CHAT` p50 (20 turns)** | `<= 350.0 ms` (`<= 700.0 ms` max) | **`303.12 ms` p50** (`593.20 ms` p99 `<= 1000 ms`) | **PASS** |
| **`TTFS-TASK` p50 (10 turns)** | `<= 950.0 ms` (`<= 1500.0 ms` max) | **`794.35 ms` p50** (`965.76 ms` p99 `<= 2500 ms`) | **PASS** |
| **`TASK` Decode Throughput (p50)** | `>= 20.0 tok/s` | **`23.66 tok/s` p50** | **PASS** |
| **`ROUTER-ACCURACY` (20 turns)** | `100.0%` (`>= 95.0%` target) | **`100.0%` (20/20)** (`13.00 ms` p50, `14.58 ms` p99) | **PASS** |
| **`BARGE-IN` max (5 events)** | `<= 200.0 ms` | **`0.028 ms` max** | **PASS** |
| **`BACKCHANNEL` latency** | `<= 500.0 ms` | **`83.63 ms`** | **PASS** |
| **`AUDIO-CONTINUITY` max gap** | `<= 100.0 ms` | **`63.95 ms` max gap** | **PASS** |
| **`IDLE-CPU` (5 s silence)** | `<= 5.0%` | **`0.107%`** | **PASS** |

---

## 9. Capstone Summary Sentence

> "Sphinx runs on the 8-library C++23 stack, routing voice turns through whisper.cpp STT (69.2 ms for 5 s audio, 0.0% WER), a 1.5B LLM router (13.0 ms p50, 100.0% accuracy), CPU-offloaded `Qwen3-Embedding-0.6B` cross-session memory backed by Yantra WAL + HNSW, and a 30B-A3B Mixture-of-Experts task model (`Qwen3-30B-A3B-Q4_K_M`, `n_cpu_moe = 32`, 23.7 tok/s, 6.61 GB peak VRAM) with code/file/SSH/system-observation tools orchestrated as Yantra DAGs, achieving 303.1 ms p50 chat TTFS, 794.4 ms p50 task TTFS, 0.028 ms barge-in, 83.6 ms backchannel, and 0.11% idle CPU, with every session tracked in Yantra and rendered to SVG by Darshana."
