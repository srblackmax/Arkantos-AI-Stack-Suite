# Obsidian — Known Issues & Audit Log (`KNOWN_ISSUES.md`)

## 1. Architectural Scope Limitations (v0.1.0)
1. **Small-Vocabulary vs. Large-Vocabulary BPE Token Rate:** `[Bench 1]` trains a compact `vocab_size = 96` BPE tokenizer on a 2-sentence corpus (`2.72 bytes/token`), achieving `18.98M tokens/s` (`49.95 MB/s`). When reporting tokenization speed against external tokenizers (e.g., `tiktoken` or Hugging Face `Tokenizers` on GPT-2 `vocab_size = 50,257`), always compare **byte throughput (`MB/s`)** alongside `tokens/s`.
2. **Transitive `libcuda.so.1` RSS Footprint via `libbasalt`:** Pure classical/tokenizer workloads (`BPE_Tokenizer`, `Mmap_Embeddings`) consume only `4.54 MB` of host RSS (`Post-Library-Load RSS`), whereas instantiating `basalt::randn` for Transformer layers triggers `libbasalt`'s CUDA driver probe (`Peak Workload RSS = 247.23 MB`).

---

## 2. Superseded Benchmark Claims (Ecosystem Benchmark Integrity Audit)

During the Ecosystem Benchmark Integrity Audit (2026-11-01), the following pre-audit benchmark issues in `bench/bench_obsidian.cpp` were identified and remediated without modifying library source code (`src/*`, `include/*`):

1. **Missing `BENCHMARKS.md` and `p50`/`p99` Quantiles (Check 1.2 & §4.4):**
   - *Pre-audit state:* `bench/bench_obsidian.cpp` timed a single loop per benchmark with no warm-up, no `p50`/`p99` quantiles, and no `BENCHMARKS.md`.
   - *Remediation:* Updated [`bench/bench_obsidian.cpp`](./bench/bench_obsidian.cpp) to run 13–15 timed batches after 3–4 warm-up batches, reporting `p50` and `p99` in [`BENCHMARKS.md`](./BENCHMARKS.md).
2. **BPE `vocab_size = 96` and `MB/s` Disclosure (§4.4):**
   - *Pre-audit state:* `[Bench 1]` printed `tokens/sec` without stating `vocab_size = 96` or byte throughput (`MB/s`).
   - *Remediation:* Added explicit `MB/s` (`49.95 MB/s`) and `vocab=96` reporting in `bench_obsidian.cpp` and `BENCHMARKS.md`, plus lossless `decode()` sanity verification.
3. **Warm Page-Cache Disclosure on `Mmap_Embeddings` (`[Bench 2]` — §4.4):**
   - *Pre-audit state:* `[Bench 2]` did not state that the `256 KiB` `.obvec` file was resident in warm page/CPU cache.
   - *Remediation:* Documented the warm page-cache condition (`7.0 ns/lookup`, ~34 cycles) in `bench_obsidian.cpp`, `BENCHMARKS.md`, and `BENCHMARKS_METHODOLOGY.md`.
4. **Post-Measurement Sanity Check on `[Bench 3]` GQA (Check 1.5):**
   - *Pre-audit state:* `[Bench 3]` cast output tensor `y` to `(void)` without inspecting its contents.
   - *Remediation:* Added `y.numel() == 4 * 32 * 64 && std::isfinite(y.read_scalar_at(0))` verification.
