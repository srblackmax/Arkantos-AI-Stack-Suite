# Obsidian — Audited Benchmark Report (`BENCHMARKS.md`)

All benchmarks were executed on the reference workstation (`12th Gen Intel Core i7-12700K` 12C/20T, AVX2 SIMD active, `48 GiB` Dual-Channel DDR4, Debian Linux `6.12.113+deb13-amd64`, `GCC 14.2.0 -std=c++23 -O3 -march=native -fopenmp`). Full timing protocols, vocabulary/model disclosures, exclusions, and post-measurement sanity checks are documented in [`BENCHMARKS_METHODOLOGY.md`](./BENCHMARKS_METHODOLOGY.md).

---

## 1. Tokenization, Zero-Copy Embeddings & Transformer Attention Benchmarks

| Benchmark Workload & Exact Configuration | `p50` Latency | `p99` Latency | Throughput | Warm Python Competitor Baseline (`HF Tokenizers 0.20` / `Gensim 4.3` / `PyTorch 2.4` on same CPU) | Sanity Check |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **`[Bench 1]` BPE Arena Encode (`encode_into_arena`)**<br>`5,000` calls, `68 B` input (`25` tokens/call), `vocab_size = 96`, 1 thread | `6.588 ms` (`1.32 us/call`) | `8.094 ms` (`1.62 us/call`) | `18,975,182 tok/s`<br>(`49.95 MB/s`)\* | `Hugging Face Tokenizers 0.20` (Rust BPE `encode` single-string Python loop, 5K calls warm): `31.40 ms` (`4.77x` slower due to Python/PyO3 per-call allocation) | `PASS` (`125,000` tokens, lossless decode) |
| **`[Bench 2]` Mmap Zero-Copy Lookup (`Mmap_Embeddings::lookup`)**<br>`100,000` lookups, `.obvec` `V = 1,000, D = 64` `fp32` (`256 KiB` warm page-cache) | `0.702 ms` (`7.0 ns/lookup`) | `0.715 ms` (`7.2 ns/lookup`) | `142.45M lookups/s` (`0` bytes copied)\*\* | `Gensim 4.3` (`KeyedVectors.get_vector("word_500", norm=False)` 100K Python dict+NumPy view lookups): `78.50 ms` (`111.8x` slower) | `PASS` (`checksum = 50000.0`) |
| **`[Bench 3]` Fused-QKV Grouped-Query Attention (`forward_self`)**<br>`[batch=4, seq=32, d_model=64]`, `n_heads=8, n_kv_heads=2`, `RoPE=true`, `fp32` CPU | `0.881 ms/pass` | `1.125 ms/pass` | `145,261 tok/s` (`1,135 passes/s`) | `PyTorch 2.4` (eager GQA + RoPE + causal mask CPU `[4, 32, 64]` warm): `1.940 ms/pass` (`2.20x` slower) | `PASS` (`8,192` finite elements) |

\* **Physical-Bound Footnote (`BPE Arena Encode` — §4.4):** Because `vocab_size = 96` produces short subword tokens (`~2.72 bytes/token` on the 68-byte ASCII input string), `18.98M tokens/s` corresponds to `49.95 MB/s` of single-core UTF-8 text throughput using AVX2 pre-tokenization + `std::pmr` arena linked-list merging with zero `malloc`/`free` calls. On a 32K–50K vocabulary (where average token length is ~4 bytes and merge chains are deeper), byte throughput is ~`25–35 MB/s` per core.

\*\* **Physical-Bound Footnote (`Mmap Zero-Copy Lookup` — §4.4):** The `256 KiB` `.obvec` file is resident in the Linux kernel page cache and the i7-12700K's L2 cache (`1.25 MiB` per P-core). Each lookup computes an inline FNV-1a string hash over 8 bytes (`"word_500"`), probes a single cache-resident open-addressed slot, and returns a 16-byte `std::span<const float>` (`ptr + offset, 64`) without copying the 256-byte embedding vector (`7.0 ns` = ~34 CPU clock cycles at `4.9 GHz`).

---

## 2. Process Memory Footprint (3-Phase RSS Split)

| Memory Phase | Measured Host RSS (`MB`) | Measurement Point & Disclosure |
| :--- | :---: | :--- |
| **4a. Bare Process RSS** | `4.20 MB` | `VmRSS` at `main()` entry before tokenizer initialization |
| **4b. Post-Library-Load RSS** | `4.54 MB` | `VmRSS` after `BPE_Tokenizer` training (pure CPU NLP path, pre-`basalt::randn`) |
| **4c. Peak Workload RSS** | `247.23 MB` | `VmHWM` after `[Bench 3]` `Grouped_Query_Attention` (reflects `libbasalt`'s `libcuda.so.1` driver probe and OpenMP thread pool initialization) |

---

## 3. Superseded Pre-Audit Numbers

In accordance with the Ecosystem Benchmark Integrity Audit rules, the pre-audit single-run console outputs of `bench/bench_obsidian.cpp` (which lacked `p50`/`p99` quantiles, `MB/s` disclosure on BPE, warm page-cache disclosure on `mmap`, and output sanity verification on GQA) are preserved below as `superseded` and cross-referenced in [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md):

| Pre-Audit Metric (`bench/bench_obsidian.cpp` v0.1.0) | Pre-Audit Single-Run Output | Status | Replacement Row |
| :--- | :---: | :---: | :--- |
| `[Bench 1] BPE Arena Encode (5000 calls)` | `~6.5 ms (~19M tokens/sec)` (single run, no `p50`/`p99`, no `vocab_size=96` / `MB/s` disclosure) | **Superseded** | Section 1, Row 1 (`p50=6.588 ms`, `p99=8.094 ms`, `18.98M tok/s`, `49.95 MB/s`, `vocab=96`) |
| `[Bench 2] Mmap Zero-Copy Lookup (100000 lookups)` | `~0.70 ms` (single run, no `p50`/`p99`, warm page-cache undisclosed) | **Superseded** | Section 1, Row 2 (`p50=0.702 ms`, `p99=0.715 ms`, `7.0 ns/lookup` warm page-cache) |
| `[Bench 3] Fused-QKV GQA [4, 32, 64] (50 passes)` | `~44 ms total` (single run, no `p50`/`p99` per pass, output cast to `(void)`) | **Superseded** | Section 1, Row 3 (`p50=0.881 ms/pass`, `p99=1.125 ms/pass`, finite output verified) |

---

## 4. Reproducing & Verifying (`bench-ci`)

```bash
make check
make bench-ci
```
`make bench-ci` runs `./bench/bench_obsidian --json-out=results.json --check-baseline=./benchmarks/baseline.json` and exits non-zero if any post-measurement sanity check fails or if any metric regresses by more than `15%` against `benchmarks/baseline.json`.
