# Stack Composability Proof (`stack-integration/`)

This directory contains `stack_integration_test`, a single, real, runnable C++23 verification binary that exercises the installed AI stack (`Yantra`, `Sutra`, and `Darshana`) end-to-end against a real GGUF language model (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`) without mocks, Python, or source-tree fallbacks. It registers the GGUF artifact into Yantra's content-addressed model registry, loads the Yantra-served artifact into Sutra's resident `LlamaCppClient`, executes a 10-turn multi-turn conversation on Sutra's asynchronous `AgentRuntime`, logs per-turn `AgentStats` telemetry to Yantra's crash-safe Write-Ahead Log (`WalEngine` / `ExperimentStore`) directly from Sutra's async worker threads, verifies round-trip metric fidelity and typed error propagation, renders a per-turn latency plot via Darshana (`results-latency.svg`), and verifies full-stack memory teardown hygiene.

> **Important Note:** This is **NOT Sphinx** and **NOT a benchmark suite** — it is a single-run composability verification harness that tests every cross-library seam and reports hard pass/fail status per seam. Any claim that the stack composes is strictly conditional on all 7 composability checks passing (`Exit: 0`) on the reference hardware.

---

## 1. WHAT THIS PROVES AND WHAT IT DOES NOT

1.1. THIS PROVES:

  - A GGUF model can flow through Yantra registry → Sutra LlamaCppClient
    without manual path manipulation
  - Sutra's AgentStats metrics can be logged to Yantra's tracker without
    type translation errors
  - Yantra's tracker does not deadlock, corrupt, or drop metrics when
    driven from Sutra's async execution thread
  - The Yantra-returned artifact (via mmap) is compatible with llama.cpp's
    GGUF loader (which may do its own mmap, possibly creating double-mmap
    or file-descriptor contention)
  - Darshana's Spec can render the tracked metrics without manual data
    reshaping (if Darshana is installed)
  - The error types compose: a Yantra error surfaced during registry
    loading can propagate through Sutra to the caller as a single typed
    error, not two disjoint types
  - 10 real turns of real inference against a real model, with real
    per-turn telemetry captured

1.2. THIS DOES NOT PROVE:

  - Multi-model routing (that is Sphinx's job)
  - Voice (that is Vach's job)
  - The full Sphinx loop (that is the next milestone)
  - Long-session stability (Sphinx needs a 30-minute soak test; this is
    10 turns, ~2 minutes)
  - Concurrency beyond what Sutra already provides (this test runs single-
    agent; Sutra's 16-agent concurrency is already proven separately)
  - Custom training (Basalt's role; not exercised here)

---

## 2. Build Prerequisites

Before building `stack-integration/`, all 8 stack libraries must be built and installed into `~/.local`:

```bash
cd ../Hyper\ Math           && cmake --install build/release --prefix ~/.local
cd ../Bellows               && cmake --install build/release --prefix ~/.local
cd ../Basalt\ Deep\ Learning && cmake --install build/release --prefix ~/.local
cd ../Obsidian              && cmake --install build/release --prefix ~/.local
cd ../Surya                 && cmake --install build/release --prefix ~/.local
cd ../Sutra                 && cmake --install build/release --prefix ~/.local
cd ../Darshana              && cmake --install build/release-vulkan --prefix ~/.local
cd ../Yantra                && cmake --install build/release --prefix ~/.local
```

`stack-integration/CMakeLists.txt` uses **only** `find_package(Yantra REQUIRED)`, `find_package(Sutra REQUIRED)`, and `find_package(Darshana QUIET)` against `-DCMAKE_PREFIX_PATH=$HOME/.local`. It never falls back to source-tree includes.

---

## 3. Build and Run Commands

```bash
./fetch_model.sh
cmake -B build -DCMAKE_PREFIX_PATH=$HOME/.local
cmake --build build
./build/stack_integration_test --model ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

An optional custom Yantra registry/store root directory can also be passed via `--registry <path>`:

```bash
./build/stack_integration_test --model ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --registry /tmp/yantra_store
```

---

## 4. Composability Checks Verified

| Check | Seam Verified | Hard Pass Criterion |
| :---: | :--- | :--- |
| **CHECK-1** | Yantra `ModelRegistry` → Sutra `LlamaCppClient` artifact handoff | Sutra loads the exact CAS path returned by Yantra and its independently computed SHA-256 matches Yantra's manifest SHA-256. |
| **CHECK-2** | Sutra `AgentStats` → Yantra `ExperimentStore` metric round-trip | Per-turn `ttft_ms`, `total_ms`, `tokens_generated`, and `tokens_per_sec` (40 points across 10 turns) round-trip through Yantra with `\|tracked - sutra\| < 1e-6`. |
| **CHECK-3** | Sutra async worker thread → Yantra `WalEngine` concurrency & durability | All 10 turns execute on Sutra's async worker pool, log metrics from that thread, and produce the exact expected 44 CRC32C-verified WAL entries (`10 * 4 metrics + run_start + run_end + 2 params`). |
| **CHECK-4** | Yantra `RegistryError` → Sutra `AgentError` typed error propagation | A missing-model `RegistryError` from Yantra propagates through `LlamaCppClient::create_from_registry` as a `sutra::AgentError` distinguishable via `sutra::is_yantra_registry_error(err)` with the original Yantra error code preserved. |
| **CHECK-5** | Yantra tracked run → Darshana `Spec` SVG visualization | Darshana renders per-turn TTFT and total generation latency from Yantra's tracked run into `results-latency.svg` (non-empty, valid XML/SVG). |
| **CHECK-6** | Content-addressed GGUF artifact integrity round-trip | Pre-registration SHA-256 of the source GGUF file matches the SHA-256 of the CAS file served by Yantra to Sutra. |
| **CHECK-7** | Full-stack teardown memory hygiene | Peak RSS during the run is `<= model_size + 500 MB`, and post-teardown RSS returns to within `10%` of the pre-run baseline. |

---

## 5. Exit Codes

| Exit Code | Meaning |
| :---: | :--- |
| `0` | All 7 composability checks passed |
| `1` | Test ran, but one or more composability checks failed |
| `2` | Model file not found at `--model <path>` (user/setup error) |
| `3` | Yantra registry or tracker initialization failure (setup error) |
| `4` | Sutra `LlamaCppClient` model load failure (setup error) |

---

## 6. Output Files and Schema

- **`stack-integration-results.json`**: Machine-readable report containing hardware metadata, model manifest details, all 10 per-turn telemetry records, `p50`/`p99` aggregate latency and throughput quantiles, per-check pass/fail status, `darshana_available`, and `exit_code`. Validated against [`expected_results.schema.json`](expected_results.schema.json).
- **`results-latency.svg`**: Vector plot generated by `Darshana` (`CHECK-5`) showing per-turn TTFT and total generation latency across the 10-turn conversation.
- **`KNOWN_ISSUES.md`**: Complete engineering log of every cross-library seam discovered during integration testing, root-cause diagnosis, where the fix was applied, and verification commands (see [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)).
