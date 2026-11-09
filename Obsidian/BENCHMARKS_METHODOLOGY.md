# Obsidian — Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

This document specifies the exact measurement protocol, hardware baseline, timed intervals, exclusions, competitor baselines (`Hugging Face Tokenizers 0.20`, `spaCy 3.8`, `PyTorch 2.4`), and post-measurement sanity checks for every benchmark metric in [`BENCHMARKS.md`](./BENCHMARKS.md).

---

## 0. Hardware & Software Environment Baseline

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 physical cores: 8 Performance-cores + 4 Efficient-cores, 20 hardware threads, 800 MHz – 5,100 MHz turbo, `30 MiB` L3 cache, AVX2 / FMA3)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, Driver `550.163.01` — probed by `libbasalt` when `basalt::randn` is invoked in Benchmark 3)
- **RAM:** `48 GiB` Dual-Channel DDR4
- **Storage:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`, hosting `/tmp/obsidian_bench.obvec`)
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `Linux 6.12.113+deb13-amd64 x86_64`
- **Compiler & Flags:** `GCC 14.2.0` (`-std=c++23 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -Werror -pedantic`)
- **Statistical Protocol:** Every benchmark executes `3–4` untimed warm-up batches followed by `13–15` timed sample batches, recording both `p50` (median) and `p99` latencies in milliseconds (`ms`).

---

## 1. `[Bench 1]` BPE Arena Encode (`encode_into_arena`)

- **Configuration (§4.4):** `BPE_Tokenizer` trained with `vocab_size = 96` (`min_frequency = 1`) on a 2-sentence reference corpus (`corpus[0]` = `"high performance deep learning and natural language processing in cpp"`, `68` UTF-8 bytes, `25` merged subword tokens per call).
- **Timed Interval:** Executes `5,000` consecutive calls to `arena.reset()` + `bpe.encode_into_arena(corpus[0], arena)` (`340,000` input bytes = `0.324 MiB`, `125,000` emitted tokens per batch) on a single thread using a stack-backed `std::pmr::monotonic_buffer_resource` (`obsidian::Arena`) with zero dynamic heap allocations.
- **Units Reported:** Batch latency (`p50 = 6.588 ms`, `p99 = 8.094 ms`), token rate (`18,975,182 tokens/s`), and byte rate (`49.95 MB/s`).
- **Exclusions:** BPE vocabulary merge-table training (`train_from_corpus`) is excluded from the timed encoding loop.
- **Sanity Check:** Verifies `batch_tokens == 125000 >= 5000` and checks that `bpe.decode(sp, true)` round-trips `corpus[0]` without error.

---

## 2. `[Bench 2]` Mmap Zero-Copy Embedding Lookup (`Mmap_Embeddings::lookup`)

- **Configuration (§4.4):** Binary `.obvec` embedding file containing `V = 1,000` words x `D = 64` `fp32` dimensions (`256 KiB` 64-byte-aligned float matrix + open-addressed string-to-row hash index), mapped read-only via POSIX `mmap(2)` (`MAP_PRIVATE`).
- **Timed Interval:** Executes `100,000` consecutive calls to `mmap_emb.lookup("word_500")` returning a zero-copy `std::span<const float>` (`64` floats) and accumulating `sp[0]` (`p50 = 0.702 ms` per `100,000` lookups = `7.0 ns/lookup`).
- **Warm Page-Cache Disclosure (§4.4):** The `.obvec` file (`~280 KiB`) is written to `/tmp` and immediately `mmap`-opened, so all pages reside in the Linux kernel page cache and CPU L1/L2 cache. The measured `7.0 ns/lookup` (~34 CPU clock cycles at `4.9 GHz`) measures the open-addressed string hash + `std::span<const float>` pointer arithmetic, **not** cold NVMe page-fault latency (~`15–25 us` per 4 KiB page fault).
- **Exclusions:** `.obvec` serialization (`save_obvec_mmap`) and `mmap(2)` syscall setup (`open_mmap`) are excluded from the lookup loop.
- **Sanity Check:** Verifies `std::abs(accum - 50000.0f) < 1.0f` (`100,000 * 0.5f = 50,000.0f`).

---

## 3. `[Bench 3]` Fused-QKV Grouped-Query Attention (`Grouped_Query_Attention::forward_self`)

- **Configuration (§4.4):** `Grouped_Query_Attention(d_model=64, num_heads=8, num_kv_heads=2, dropout=0.0, causal=true, use_rope=true, max_seq_len=128)` (4:1 GQA head sharing ratio, `head_dim = 8`) evaluated in `fp32` under `No_Grad_Guard` on input tensor `[batch=4, seq_len=32, d_model=64]` (`128` tokens per forward pass).
- **Timed Interval:** Executes `20` forward passes per sample batch (`gqa.forward_self(x, Tensor(), true)`), including fused `W_qkv` linear projection (`64 -> 80`), Rotary Position Embedding (`RoPE`) application, KV head broadcasting (`2 -> 8`), scaled dot-product causal attention, and output projection (`64 -> 64`), dividing batch time by `20` (`p50 = 0.881 ms/pass`, `p99 = 1.125 ms/pass`).
- **Exclusions:** Module weight initialization and input tensor `randn` synthesis are excluded.
- **Sanity Check:** Verifies `y.numel() == 4 * 32 * 64` (`8,192` elements) and `std::isfinite(y.read_scalar_at(0))`.

---

## 4. Process Memory RSS Phases

- **Timed Interval:** Reads `/proc/self/status` at three checkpoints:
  1. **Bare Process RSS (`VmRSS = 4.20 MB`):** First statement in `main()` before tokenizer construction.
  2. **Post-Library-Load RSS (`VmRSS = 4.54 MB`):** After `BPE_Tokenizer` training (pre-`basalt::randn` CUDA driver probe).
  3. **Peak Workload RSS (`VmHWM = 247.23 MB`):** High-water mark resident set after `Grouped_Query_Attention` forward passes (includes `libbasalt`'s `libcuda.so.1` driver probe and OpenMP thread pool).
