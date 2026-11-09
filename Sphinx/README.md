# Sphinx — Tri-Route Cognitive Voice & Engineering Assistant

**Part of the Arkantos AI Stack Suite**  
**Author:** Kival  
**License:** GNU General Public License v3.0 or later (`GPL-3.0-or-later`) with Section 7 Runtime & Hardware Compute Library Exception

---

## Overview

**Sphinx** (`libsphinx.so`, `sphinx`) is a native C++23 cognitive voice and text assistant orchestrated over **`Sutra`** (ReAct tool-calling & hybrid GPU+CPU MoE runtime), **`Vach`** (real-time ALSA/PipeWire audio I/O, Silero VAD v5, CUDA Whisper STT, ONNX Piper TTS), **`Yantra`** (content-addressed model registry, WAL telemetry, HNSW vector memory, and DAG execution), and **`Darshana`** (SVG session telemetry plots).

### Key Architectural Capabilities
1. **Tri-Route Cognitive Dispatch (`CHAT`, `BIG_CHAT`, `TASK`)**:
   - **`CHAT`** (`/chat `): Uses the resident GPU router model (`Qwen2.5-1.5B-Instruct-Q4_K_M`) at `>210 tok/s` for greetings, quick status checks, and system-profile queries.
   - **`BIG_CHAT`** (`/big `): Streams directly from the 80B MoE model (`Qwen3-Coder-Next-80B-A3B-Q4_K_S`) at `~22.6 tok/s` **with zero tool schemas and zero ReAct loop overhead** for conceptual, mathematical, historical, philosophical, and architectural treatises.
   - **`TASK`** (`/task `): Executes the full Yantra-supervised ReAct agent loop on the 80B MoE model with grammar-constrained tool calls (`read_file`, `write_file`, `list_dir`, `make_directory`, `run_command`, `run_readonly`, `read_file_system`, `ssh_exec`, `ssh_test`).
2. **12-Physical-Core Pinning & Zero-Fault Warm Starts**:
   - Automatically detects physical CPU cores via Linux sysfs (`8 P-cores + 4 E-cores = 12 physical cores` on i7-12700K) and pins a single shared `ggml_threadpool` (`1200%` CPU utilization, zero SMT sibling collisions).
   - Preserves `38.51 GiB` of CPU MoE expert and embedding weights in the Linux page cache across restarts (`0.95s` preload at `40.33 GiB/s`, `MAJFL = 0` during generation).
3. **Zero Memory-Bleed Guarantee**:
   - When `Yantra` / `MemoryStore` returns no relevant persistent memories above the similarity threshold for a turn, the persistent memory prompt block is omitted completely.
4. **Root Execution Safety Guard**:
   - Emits an immediate security warning if invoked with effective UID `0` (`geteuid() == 0`).

---

## Build & Verification

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build/release -j"$(nproc)"
ctest --test-dir build/release --output-on-failure
cmake --install build/release --prefix "$HOME/.local"
```

## CLI Usage

```bash
# Interactive or piped text mode with eager 80B MoE preload across 12 physical cores:
sphinx --text --eager-big --preload-all --threads 12 --threads-batch 12

# Deterministic route prefix overrides:
#   /chat Hello, how are you?
#   /big Provide a comprehensive mathematical treatise on Noether's Theorem.
#   /task List all files in the workspace directory.
```
