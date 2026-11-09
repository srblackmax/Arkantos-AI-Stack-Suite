# Benchmark Integrity Audit — 2026-11-01

**Audited Stack Root:** `.`  
**Verified Host Hardware Baseline:**
- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 cores: 8 P-cores + 4 E-cores, 20 threads, 800–5,100 MHz)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, Driver `550.163.01`, Vulkan `1.4.309`, CUDA `12.4.131`)
- **RAM:** `48 GiB` (`46 GiB` usable) Dual-Channel DDR4
- **Disk:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`, `/tmp` & OS root) + `WDC WD64PURZ-74BWUY0 6TB` (`$HOME` workspace mount)
- **Kernel:** `Linux 6.12.113+deb13-amd64 x86_64`

---

## Summary

| Library | Metrics audited | Flagged | Clean | Action |
|---|---|---|---|---|
| Hyper Math | 8 | 8 | 0 | Fix (Phase 2) |
| Bellows | 5 | 5 | 0 | Fix (Phase 2) |
| Basalt Deep Learning | 6 | 6 | 0 | Fix (Phase 2) |
| Obsidian | 3 | 3 | 0 | Fix (Phase 2) |
| Surya | 25 | 25 | 0 | Fix (Phase 2) |
| Sutra | 11 | 11 | 0 | Fix (Phase 2) |
| Darshana | 8 | 8 | 0 | Fix (Phase 1) |
| **Total** | **66** | **66** | **0** | **Patch All 7 Libraries** |

### Per-Library, Per-Metric Audit Matrix (All 66 Metrics vs. Checks 1.1–1.10)

| Library | Metric | 1.1 Plausibility | 1.2 Interval | 1.3 Exclusions | 1.4 Mem Phase | 1.5 Hardware | 1.6 Repro Cmd | 1.7 Sanity | 1.8 Rival Comp | 1.9 Bound Note | 1.10 FPS Mode | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hyper Math** | `1M Element Fused UFunc` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Hyper Math** | `256x256 Parallel Matrix Multiply` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | N/A | **Flagged** |
| **Hyper Math** | `65,536-Point Radix-2 Complex FFT` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Hyper Math** | `100K Row Lazy Filter + GroupBy + Prune` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Hyper Math** | `1000-D Reverse-Mode Autograd Gradient` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Hyper Math** | `2000x2000 Sparse Conjugate Gradient` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Hyper Math** | `10,000 x 8-D QMC Halton Sequence` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Hyper Math** | `50K Row SQL Context SELECT+WHERE+ORDER` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Bellows** | `Stagewise Booster Fit (Exact vs Quantile)` | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Bellows** | `Stagewise Booster Batch Predict Throughput` | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Bellows** | `Single-Row Zero-Alloc Inference (infer_one)` | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Bellows** | `Penalised Linear / Logistic Fit & Predict` | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Bellows** | `Composite Pipeline Fit + Transform` | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Basalt** | `matmul (256x256 fp32, CPU AVX2+FMA)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | N/A | **Flagged** |
| **Basalt** | `Conv2d (4x8x16x16 -> 16, k=3, CPU)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Basalt** | `MLP (32x64->128->32 fwd+bwd, CPU)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Basalt** | `Transformer_Encoder (4x16x32, CPU)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Basalt** | `CUDA matmul (1024x1024 fp32, RTX 3070)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | N/A | **Flagged** |
| **Basalt** | `CUDA TensorCore (2048x2048 fp16, RTX 3070)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | N/A | **Flagged** |
| **Obsidian** | `[Bench 1] BPE Arena Encode` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Obsidian** | `[Bench 2] Mmap Zero-Copy Lookup` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | N/A | **Flagged** |
| **Obsidian** | `[Bench 3] Fused-QKV GQA [4, 32, 64]` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Surya** | `Single-Image Latency (CPU-Only, b=1)` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Surya** | `Single-Image Latency (Hybrid No Graph, b=1)` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Surya** | `Single-Image Latency (CUDA Graph, b=1)` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | N/A | **Flagged** |
| **Surya** | `Batched Hybrid Throughput (b=4,8,16,32)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | N/A | **Flagged** |
| **Surya** | `Overlapped Multi-Stream (128 imgs, b=16)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | N/A | **Flagged** |
| **Surya** | `VRAM Residency & Budget (4 rows)` | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | N/A | N/A | N/A | **Flagged** |
| **Surya** | `core::aligned_alloc_64 (1080p RGB)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | PASS | N/A | FAIL | N/A | **Flagged** |
| **Surya** | `core::Aligned_Buffer_Pool (1080p RGB)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | N/A | N/A | N/A | **Flagged** |
| **Surya** | `core::simd_scale_bias_u8_to_f32 (1080p)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | N/A | N/A | **Flagged** |
| **Surya** | `imgproc::rgb_to_gray (1280x720)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `imgproc::gaussian_blur 5x5 (1280x720)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `imgproc::resize Bilinear (720p->640x640)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `io::decode_qoi (640x480 RGB)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `io::decode_jpeg (640x480 RGB)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `augment::Augment_Pipeline 5-stage (640x640)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `nn::Conv_BN_Act unfused (1x3x64x64->16)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `nn::Conv_BN_Act fused (1x3x64x64->16)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `detection::letterbox_image (720p->640x640)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `detection::batched_nms (1000 boxes, 10 cls)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `detection::Byte_Tracker (25 Kalman tracks)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | FAIL | FAIL | **Flagged** |
| **Surya** | `classical::HOG_Extractor (64x128)` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | N/A | N/A | **Flagged** |
| **Surya** | `classical::estimate_homography_ransac` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | N/A | N/A | N/A | **Flagged** |
| **Sutra** | `Cold-Start Initialization Latency` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Sutra** | `Cold-Start Memory Footprint (RSS)` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Sutra** | `10-Tool Parallel Execution Speedup` | PASS | FAIL | PASS | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Sutra** | `Overlapped Inference + Tool CPU Utilization` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Sutra** | `500-Turn Incremental Checkpoint Latency` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | N/A | **Flagged** |
| **Sutra** | `10,000-Turn Sequential Memory Stability` | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged** |
| **Sutra** | `Multi-Agent Throughput (N=1,2,4,8,16)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | N/A | **Flagged** |
| **Sutra** | `Semantic Memory Vector Search (CPU SIMD)` | PASS | FAIL | PASS | FAIL | FAIL | PASS | FAIL | N/A | N/A | N/A | **Flagged** |
| **Sutra** | `Semantic Memory Vector Search (GPU CUDA)` | PASS | FAIL | PASS | FAIL | FAIL | PASS | FAIL | N/A | N/A | N/A | **Flagged** |
| **Darshana** | `Interactive Scatter Plot (10M points)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | **Flagged (Phase 1)** |
| **Darshana** | `10M Scatter Frame Jitter (p99 - p50)` | PASS | PASS | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | N/A | FAIL | **Flagged (Phase 1)** |
| **Darshana** | `Interactive Line Plot (1M points)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | **Flagged (Phase 1)** |
| **Darshana** | `Streaming Frame Jitter (60 Hz ring buffer)` | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | FAIL | **Flagged (Phase 1)** |
| **Darshana** | `SVG Vector Export (1M-point scatter, CPU)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | N/A | **Flagged (Phase 1)** |
| **Darshana** | `Cold-Start Latency` | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | N/A | N/A | **Flagged (Phase 1)** |
| **Darshana** | `Idle Memory Footprint (RSS delta)` | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | N/A | **Flagged (Phase 1)** |
| **Darshana** | `Peak GPU Pool Memory (10M scatter)` | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | N/A | N/A | N/A | **Flagged (Phase 1)** |

---

## Hyper Math

Hyper Math (`benchmarks/bench_main.cpp`, `./bin/bench-hyper-math`) implements 8 benchmarks via `make bench` but lacks a root `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`. All 8 metrics use `measure_ms` (which takes the minimum of 3 runs without warm-up or `p50`/`p99` reporting).

### 1. `1M Element Fused UFunc (sin*2.5+cos-1.25)`
- **Current claim:** `1M Element Fused UFunc (sin*2.5+cos-1.25) ... Mops/s` (in `benchmarks/bench_main.cpp:83-85`; no `BENCHMARKS.md` present).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** `measure_ms` records only the best of 3 runs (`min`) rather than `p50`/`p99` latency; it labels `1000.0 / ms` as `"Mops/s"` when each element performs 5 floating-point operations (`sin`, `*`, `+`, `cos`, `-`), conflating `Melem/s` with `MFLOP/s`. Allocation of `x` and `y` is excluded without disclosure, `out` is not checksum-verified after timing, and no scalar or NumPy rival baseline (with version, warm/cold state, and compiler flags) is documented in a `BENCHMARKS.md`.
- **Recommended fix:** Extend `benchmarks/bench_main.cpp` to measure `p50`/`p99` over $\ge 25$ iterations after explicit warm-up, report both `Melem/s` and `GFLOP/s` alongside an unfused/scalar baseline, verify `sum(out)` finite checksum, and document in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 2. `256x256 Parallel Matrix Multiply (matmul)`
- **Current claim:** `256x256 Parallel Matrix Multiply (matmul) ... GFLOP/s` (`benchmarks/bench_main.cpp:89-102`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9
- **Diagnosis:** A $256 \times 256$ `fp64` matrix occupies $512\text{ KB}$ ($1.5\text{ MB}$ total for $A, B, C$), fitting inside the per-core L2 cache of the i7-12700K. The benchmark does not disclose whether caches are warm or cold, what tiling/blocking algorithm is used, or how it scales to out-of-cache sizes (e.g., $1024 \times 1024$), nor does it verify `c` against a reference norm.
- **Recommended fix:** Report both L2-resident ($256 \times 256$) and L3/DRAM-bound ($1024 \times 1024$) blocked AVX2+FMA `matmul` with `p50`/`p99` latency, warm/cold cache labels, compiler flags (`-O3 -march=native`), Frobenius norm verification, and NumPy/OpenBLAS baseline comparison.

### 3. `65,536-Point Radix-2 Complex FFT`
- **Current claim:** `65,536-Point Radix-2 Complex FFT ... N=65536` (`benchmarks/bench_main.cpp:105-113`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports only minimum latency over 3 runs without throughput units (`Mpts/s` or effective `GFLOP/s` via $5 N \log_2 N$), omits Parseval energy conservation sanity check on `spec`, and lacks `BENCHMARKS.md` documentation.
- **Recommended fix:** Add `p50`/`p99` timing, compute effective FFT `GFLOP/s`, verify Parseval norm $\|x\|_2^2 \approx \frac{1}{N}\|X\|_2^2$ post-measurement, and document in `BENCHMARKS.md`.

### 4. `100K Row Lazy Filter + GroupBy + Prune`
- **Current claim:** `100K Row Lazy Filter + GroupBy + Prune ... 4 groups` (`benchmarks/bench_main.cpp:116-145`).
- **Checks passed:** 1.1, 1.6, 1.7
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.8
- **Diagnosis:** Excludes `data_frame` construction from the timed interval without explicit disclosure in documentation, reports only best-of-3 latency without `p50`/`p99` or throughput (`rows/s`), and lacks Pandas/Polars rival comparison metadata.
- **Recommended fix:** Report `p50`/`p99` latency and `Mrows/s`, disclose that in-memory `data_frame` construction is excluded, verify `res.nrows() == 4` and aggregated sum, and document in `BENCHMARKS.md`.

### 5. `1000-D Reverse-Mode Autograd Gradient`
- **Current claim:** `1000-D Reverse-Mode Autograd Gradient ... N=1000 vars` (`benchmarks/bench_main.cpp:148-169`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports best-of-3 latency without `p50`/`p99`, does not disclose that Wengert tape construction + backward pass are both inside the timed interval, and does not verify the analytical Rosenbrock gradient values on `g`.
- **Recommended fix:** Add `p50`/`p99` latency, verify `g` norm against analytical Rosenbrock derivatives, and document in `BENCHMARKS.md`.

### 6. `2000x2000 Sparse Conjugate Gradient (CG)`
- **Current claim:** `2000x2000 Sparse Conjugate Gradient (CG) ... N=2000 SPD` (`benchmarks/bench_main.cpp:172-205`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Excludes CSR matrix assembly without disclosure in docs, reports best-of-3 latency without `p50`/`p99`, and does not verify residual $\|A x - b\|_2 < 10^{-6}$ after solving.
- **Recommended fix:** Add `p50`/`p99` latency, verify $\|A x - b\|_2$ residual post-solve, and document CSR assembly exclusion in `BENCHMARKS.md`.

### 7. `10,000 x 8-D QMC Halton Sequence`
- **Current claim:** `10,000 x 8-D QMC Halton Sequence ... 80K radical-inv` (`benchmarks/bench_main.cpp:208-218`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports best-of-3 latency without `p50`/`p99` or throughput (`Mpts/s`), and omits a post-measurement bounds/mean sanity check ($\mu \approx 0.5$ in $[0, 1)^8$).
- **Recommended fix:** Add `p50`/`p99` latency, `Mvals/s` throughput, and mean/bounds verification on `pts`.

### 8. `50K Row SQL Context SELECT+WHERE+ORDER`
- **Current claim:** `50K Row SQL Context SELECT+WHERE+ORDER ... 100 rows` (`benchmarks/bench_main.cpp:221-244`).
- **Checks passed:** 1.1, 1.6, 1.7
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.8
- **Diagnosis:** Reports best-of-3 latency without `p50`/`p99`, does not disclose that table registration is excluded from the timed SQL parse+execute interval, and lacks `BENCHMARKS.md` documentation.
- **Recommended fix:** Add `p50`/`p99` latency, verify `sql_out.nrows() == 100` and descending order invariant, and document in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

---

## Bellows

Bellows (`README` line 100) claims `bellows — Inspect datasets, model artefacts, and run benchmarks`, but `src/bellows_main.cpp` lacks a `--bench` flag and the repository has no `BENCHMARKS.md` or `BENCHMARKS_METHODOLOGY.md`.

### 1. `Stagewise Booster Fit (Exact vs Quantile Histogram Cuts)`
- **Current claim:** Unreported in `BENCHMARKS.md` (`README` claims benchmark capability via `bellows`).
- **Checks passed:** 1.1
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
- **Diagnosis:** No benchmark harness or `BENCHMARKS.md` exists to measure `stagewise_booster` training latency (`p50`/`p99`), disclose exclusions (synthetic `data_table` generation vs `bellows::fit()` call), verify post-fit loss reduction, or state `scikit-learn` / `XGBoost` comparison parameters (version, warm/cold, hyperparameter parity).
- **Recommended fix:** Add `--bench` to `bellows` (and standalone `bench-ci` target) measuring `stagewise_booster` fit (`exact_cut_policy` and `quantile_bin_policy`) with `p50`/`p99`, loss sanity check, and full disclosures in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 2. `Stagewise Booster Batch Predict Throughput`
- **Current claim:** Unreported in `BENCHMARKS.md`.
- **Checks passed:** 1.1
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
- **Diagnosis:** Batch prediction throughput (`rows/s`) is not benchmarked with explicit batch size disclosure (`§4.2`), `p50`/`p99` latency, or finite prediction checksum verification.
- **Recommended fix:** Benchmark batch `bellows::predict` with explicit batch sizes ($N = 1{,}000$ and $N = 10{,}000$), `p50`/`p99` latency, `rows/s` throughput, and sanity verification.

### 3. `Single-Row Zero-Allocation Inference (bellows::infer_one)`
- **Current claim:** Unreported in `BENCHMARKS.md` (`README` line 63 claims "zero-allocation single-row inference (`bellows::infer_one`)").
- **Checks passed:** 1.1
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
- **Diagnosis:** No latency (`p50`/`p99` in nanoseconds/microseconds) or batch-size=1 throughput is measured or verified for `bellows::infer_one`.
- **Recommended fix:** Measure `bellows::infer_one` (`batch_size = 1`) over 100,000 calls, report `p50`/`p99` latency and `inferences/s`, and verify numerical agreement with `bellows::predict`.

### 4. `Penalised Linear & Logistic Estimator Fit/Predict`
- **Current claim:** Unreported in `BENCHMARKS.md`.
- **Checks passed:** 1.1
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
- **Diagnosis:** Missing timed `fit()` and `predict()` benchmarks for `penalised_linear_regressor` and `penalised_logistic_classifier` with `scikit-learn` warm/cold comparison metadata.
- **Recommended fix:** Add `penalised_linear_regressor` / `penalised_logistic_classifier` benchmark rows with `p50`/`p99` latency, $R^2$/accuracy sanity checks, and `scikit-learn` baseline disclosures.

### 5. `Composite Pipeline (moment_normaliser | stagewise_booster) End-to-End`
- **Current claim:** Unreported in `BENCHMARKS.md`.
- **Checks passed:** 1.1
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
- **Diagnosis:** Missing timed benchmark separating preprocessing (`moment_normaliser` transform) from model fitting (`stagewise_booster` fit) as required by `§4.2`.
- **Recommended fix:** Benchmark both isolated `fit()` and full `moment_normaliser | stagewise_booster` pipeline fit+predict, disclosing preprocessing vs training time splits and RSS phases.

---

## Basalt Deep Learning

Basalt reports 6 benchmark numbers in `NEWS` (lines 55–61) generated by `./bench/bench_basalt` (`bench/bench_basalt.cpp`), but lacks `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 1. `matmul (256x256 fp32, CPU AVX2+FMA)`
- **Current claim:** `matmul (256x256 fp32, CPU AVX2+FMA) : 0.111 ms/iter (303.380 GFLOP/s)` (`NEWS:56`).
- **Checks passed:** 1.6
- **Checks failed:** 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9
- **Diagnosis:** Three $256 \times 256$ `fp32` tensors occupy $768\text{ KB}$, which resides entirely in the i7-12700K L2 cache (`1.25 MB` per P-core) across the 20-iteration loop. `303.38 GFLOP/s` exceeds DRAM-bandwidth limits and requires a physical-bound footnote (`Check 1.9`) explaining L2 cache residency + 8-P-core OpenMP AVX2+FMA execution. It also lacks `p50`/`p99` percentiles, output checksum verification (`(void) basalt::ops::matmul(a, b)`), and out-of-cache $1024 \times 1024$ comparison.
- **Recommended fix:** Add `p50`/`p99` latency, output checksum verification, an L2-residency footnote (`Check 1.9`), and PyTorch 2.4 CPU comparison in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 2. `Conv2d (4x8x16x16 -> 16, k=3, CPU)`
- **Current claim:** `Conv2d (4x8x16x16 -> 16, k=3, CPU) : 0.088 ms/iter` (`NEWS:57`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports only mean latency over 10 iterations without `p50`/`p99` or throughput (`images/s`), does not disclose that weight initialization and `No_Grad_Guard` setup are excluded, and discards the output tensor without finite/norm verification.
- **Recommended fix:** Report `p50`/`p99` latency and `imgs/s` (with batch size 4, $8 \to 16$ channels, $16 \times 16$ spatial size), verify output tensor sum, and document in `BENCHMARKS.md`.

### 3. `MLP (32x64->128->32 fwd+bwd, CPU)`
- **Current claim:** `MLP (32x64->128->32 fwd+bwd, CPU) : 0.205 ms/iter` (`NEWS:58`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports mean latency over 20 iterations without `p50`/`p99` or `samples/s` throughput, does not disclose that `DataLoader` and optimizer step (`AdamW::step`) are excluded from the `zero_grad + forward + mean + backward` interval, and does not verify non-zero parameter gradients after `l.backward()`.
- **Recommended fix:** Report `p50`/`p99` latency, disclose layer dimensions (`Linear(64,128) -> GELU -> Linear(128,32)`, `batch=32`) and exclusions, and verify parameter gradient norm $> 0$.

### 4. `Transformer_Encoder (4x16x32, CPU)`
- **Current claim:** `Transformer_Encoder (4x16x32, CPU) : 0.341 ms/iter` (`NEWS:59`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports mean latency over 5 iterations without `p50`/`p99` or `tokens/s` throughput, and does not verify output/gradient finite values.
- **Recommended fix:** Increase sample count to $\ge 25$ iterations for `p50`/`p99`, state `batch=4, seq_len=16, d_model=32, nhead=4, d_ff=64` (attention + MLP fwd+bwd), and verify output/gradient checksums.

### 5. `CUDA matmul (1024x1024 fp32, RTX 3070)`
- **Current claim:** `CUDA matmul (1024x1024 fp32, RTX 3070) : 0.209 ms/iter (10261.162 GFLOP/s)` (`NEWS:60`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9
- **Diagnosis:** `a` and `b` are uploaded to VRAM before `t0` (`basalt::randn({1024, 1024}).to(gpu)`), so PCIe H2D/D2H transfers are excluded without disclosure in a `BENCHMARKS.md`. Moreover, 20 iterations are timed with a single `device_synchronize` at the end, hiding per-iteration `p50`/`p99` distribution, and the result tensor is discarded without verification.
- **Recommended fix:** Synchronize per iteration to measure true `p50`/`p99` kernel latency, disclose that tensors are resident in VRAM (excluding PCIe transfer), add a footnote on RTX 3070 GA104 FP32 throughput (`20.3 TFLOP/s` theoretical peak), and verify the output tensor on host.

### 6. `CUDA TensorCore (2048x2048 fp16, RTX 3070)`
- **Current claim:** `CUDA TensorCore (2048x2048 fp16, RTX 3070) : 0.504 ms/iter (34109.968 GFLOP/s)` (`NEWS:61`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9
- **Diagnosis:** `34.11 TFLOP/s` exceeds the RTX 3070's FP32 shader peak (`20.3 TFLOP/s`) because `cuBLAS` dispatches `sm_86` FP16 Tensor Core MMA instructions (`40.6 TFLOP/s` FP16 with FP32 accumulate / `81.3 TFLOP/s` FP16 accumulate) and benefits from the 4 MB GA104 L2 cache (`Check 1.9`). Per-iteration `p50`/`p99`, VRAM/RSS phase labels, and output verification are also missing.
- **Recommended fix:** Measure per-iteration synchronized `p50`/`p99`, verify output finite checksum, and document Tensor Core + L2 cache effects (`Check 1.9`) and PyTorch `torch.matmul` (`fp16`, CUDA 12.4) comparison in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

---

## Obsidian

Obsidian has `./bench/bench_obsidian` (`bench/bench_obsidian.cpp`) with 3 benchmarks, but lacks `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 1. `[Bench 1] BPE Arena Encode`
- **Current claim:** `[Bench 1] BPE Arena Encode: 5000 calls (65000 tokens) in ... ms (... tokens/sec)` (`bench/bench_obsidian.cpp:54-58`).
- **Checks passed:** 1.1, 1.6, 1.7
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.8
- **Diagnosis:** Times 5,000 aggregate calls without `p50`/`p99` per-call latency, does not document the tokenizer type (BPE), vocabulary size (`vocab_size = 96`), and 9-word ASCII input corpus in a `BENCHMARKS.md` (`§4.4`), and lacks a comparison against Hugging Face `tokenizers` with version and warm/cold state.
- **Recommended fix:** Record per-call `p50`/`p99` latency and `tokens/sec`, disclose BPE algorithm, `vocab_size = 96`, 69-byte input string, and exclusion of `train_from_corpus`, and create `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 2. `[Bench 2] Mmap Zero-Copy Lookup`
- **Current claim:** `[Bench 2] Mmap Zero-Copy Lookup: 100000 lookups in ... ms (checksum=50000)` (`bench/bench_obsidian.cpp:88-89`).
- **Checks passed:** 1.1, 1.6, 1.7
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.8, 1.9
- **Diagnosis:** Repeatedly looks up `"word_500"` in a 1,000-word $\times$ 64-D (`256 KB`) memory-mapped `.obvec` table that is resident in L1/L2 CPU cache and OS page cache, excluding disk I/O and `.obvec` serialization. Missing `p50`/`p99` latency, memory phase labels, hardware specs, and Gensim `KeyedVectors` comparison.
- **Recommended fix:** Measure `p50`/`p99` lookup latency across varied keys, disclose OS page-cache/L2 residency (`Check 1.3` & `1.9`), label RSS phases, and document in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

### 3. `[Bench 3] Fused-QKV GQA [4, 32, 64]`
- **Current claim:** `[Bench 3] Fused-QKV GQA [4, 32, 64]: 50 passes in ... ms` (`bench/bench_obsidian.cpp:106-107`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Reports only 50-pass total time without `p50`/`p99` per-pass latency or `tokens/sec` throughput, discards the output tensor `y` (`(void) y;`) without a finite/norm sanity check (`Check 1.7`), and lacks model configuration disclosures (`d_model=64, n_heads=8, n_kv_heads=2, batch=4, seq_len=32, RoPE=true`).
- **Recommended fix:** Add `p50`/`p99` latency, `tokens/sec` throughput, output tensor finite checksum verification, and document in `BENCHMARKS.md` and `BENCHMARKS_METHODOLOGY.md`.

---

## Surya

Surya (`BENCHMARKS.md` and `bench/bench_surya.cpp`) reports 25 metrics across 4 tables.

### 1. `Single-Image Latency (batch = 1): CPU-Only, Hybrid No Graph, and CUDA Graph Replay` (3 rows)
- **Current claim:**
  - `CPU-Only (oneTBB + SIMD): p50 5,326.09 us, p99 6,418.06 us, 186.52 img/s`
  - `CPU+GPU Hybrid (No CUDA Graph): p50 391.10 us, p99 1,045.80 us, 2,506.78 img/s`
  - `CPU+GPU Hybrid (CUDA Graph Replay): p50 332.58 us, p99 950.27 us, 2,915.79 img/s`
- **Checks passed:** 1.1, 1.2, 1.6, 1.7
- **Checks failed:** 1.3, 1.4, 1.5, 1.8, 1.9
- **Diagnosis:** While `p50`/`p95`/`p99` and `img/s` are reported, `BENCHMARKS.md` does not explicitly state whether the `256 MiB` INT8 model weights were warm-resident in VRAM (excluding `SuryaPipeline::create` cold load and CUDA Graph capture time) and whether PCIe H2D/D2H transfers of the `640x480` RGB frame are included inside the `332.58 us` interval (`§4.5`). It also lacks host RSS phase labels (`Check 1.4`), RAM/disk/kernel hardware specs (`Check 1.5`), and a footnote explaining how INT8 downsampled $80 \times 60$ feature maps achieve sub-millisecond VLM head replay (`Check 1.9`).
- **Recommended fix:** Disclose warm-resident weights, CUDA Graph capture exclusion, and inclusion of pinned H2D/D2H transfers per `§4.5`, add host RSS phases and full hardware specs, and document in `BENCHMARKS_METHODOLOGY.md`.

### 2. `Batched Hybrid Throughput (batch = 1, 4, 8, 16, 32) & Overlapped Multi-Stream` (6 rows)
- **Current claim:** `batch = 1..32` (`2,915.79 img/s` to `5,190.26 img/s`) and `Overlapped Multi-Stream (128 imgs, b=16): 23,081.83 us, 180.33 us/img, 5,545.49 img/s`.
- **Checks passed:** 1.1, 1.6, 1.7
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.8, 1.9
- **Diagnosis:** Reports only mean batch latency without `p50`/`p99` batch latency percentiles (`Check 1.2`), omits explicit PCIe transfer inclusion/exclusion notes (`Check 1.3`), and lacks RAM/disk/kernel specs (`Check 1.5`).
- **Recommended fix:** Add `p50`/`p99` batch latency columns, disclose pinned-memory H2D/D2H overlap across 4 CUDA streams, and document in `BENCHMARKS_METHODOLOGY.md`.

### 3. `VRAM Residency & Budget Compliance` (4 rows)
- **Current claim:** `Configured Hard VRAM Cap: 6,656.00 MiB`, `Resident INT8 Model Weights: 256.00 MiB`, `Steady-State VRAM: 2,090.89 MiB`, `Peak VRAM: 3,690.89 MiB`.
- **Checks passed:** 1.1, 1.2, 1.3, 1.6, 1.7
- **Checks failed:** 1.4, 1.5
- **Diagnosis:** VRAM phases are labeled, but host process RSS phases (`(a) Bare process RSS`, `(b) Post-library-load RSS`, `(c) Post-GPU-init RSS`) are absent (`Check 1.4`), and RAM/disk/kernel specs are missing (`Check 1.5`).
- **Recommended fix:** Add the three host RSS phase rows alongside the VRAM table and complete the hardware header.

### 4. `Subsystem / Kernel Microbenchmarks` (12 rows in Section 4)
- **Current claim:**
  - `core::aligned_alloc_64 (1080p RGB): 0.03 us`
  - `core::Aligned_Buffer_Pool (1080p RGB): 0.02 us`
  - `core::simd_scale_bias_u8_to_f32 (1080p): 198.77 us (10.43 GPix/s)`
  - `imgproc::rgb_to_gray (1280x720): 77.34 us (12,930.25 FPS)`
  - `imgproc::gaussian_blur 5x5 (1280x720): 399.68 us (2,502.00 FPS)`
  - `imgproc::resize Bilinear (720p -> 640x640): 345.86 us (2,891.32 FPS)`
  - `io::decode_qoi (640x480 RGB): 120.45 us (8,302.32 FPS)`
  - `io::decode_jpeg (640x480 RGB): 245.15 us (4,079.07 FPS)`
  - `augment::Augment_Pipeline 5-stage (640x640): 827.32 us (1,208.72 FPS)`
  - `nn::Conv_BN_Act unfused / fused (1x3x64x64 -> 16): 659.10 us (1,517.21 FPS) / 224.85 us (4,447.48 FPS)`
  - `detection::letterbox_image (720p -> 640x640): 237.30 us (4,214.03 FPS)`
  - `detection::batched_nms (1000 boxes, 10 cls): 516.38 us (1,936.56 FPS)`
  - `detection::Byte_Tracker (25 Kalman tracks): 14.48 us (69,072.46 FPS)`
  - `classical::HOG_Extractor (64x128): 53.48 us (18,699.57 win/s)`
  - `classical::estimate_homography_ransac: 11.50 us (86,967.07 solves/s)`
- **Checks passed:** 1.6
- **Checks failed:** 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 1.10
- **Diagnosis:**
  1. **Check 1.10 & 1.1 Violation:** 11 rows label in-memory CPU processing rates (`1,208` to `69,072` per second) as bare `FPS` without any display swapchain; these must be labeled `img/s (offscreen CPU compute)` or `updates/s (CPU compute)`.
  2. **Check 1.1 & 1.9 Violation on `aligned_alloc_64`:** `0.03 us` (`30 ns`) for a `6.22 MB` (`1080p RGB`) `posix_memalign`/`free` pair occurs because glibc's allocator recycles the same `mmap` chunk or only touches page `0` (`p[0] = 1U`), never faulting the remaining 1,518 pages (`Check 1.3`/`1.9`).
  3. **Check 1.2 & 1.7 Violation:** `measure_us` reports only mean latency (no `p50`/`p99`) and discards outputs (`(void) g;`, `(void) d;`) without post-measurement sanity verification.
- **Recommended fix:** Relabel all bare `FPS` units in `Surya` to `img/s (offscreen CPU compute)` (`Check 1.10`), report `p50`/`p99` latencies, add post-measurement output checksum assertions (`Check 1.7`), and disclose page-zero touch behavior in `aligned_alloc_64`.

---

## Sutra

Sutra (`BENCHMARKS.md` and `bench/bench_sutra.cpp`) reports 11 metrics across 6 benchmark groups.

### 1. `Cold-Start Initialization Latency`
- **Current claim:** `Cold-Start Initialization Latency (main() to first Agent ready): 0.38 ms` (`Sutra/BENCHMARKS.md:15`).
- **Checks passed:** 1.1, 1.6, 1.7
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.8
- **Diagnosis:** `0.38 ms` measures `ToolRegistry` registration + `Agent::create` + `AgentRuntime(10U)` thread-pool construction with `OverlappedInferenceLlm` (an in-memory mock `LlmClient`), **excluding** GGUF model weight loading (`NativeEngineClient` mmap/dequantization). Per `Check 1.2` and `Check 1.3`, cold start must explicitly state whether model load is included, report `p50`/`p99`, and report both orchestration-only cold start and GGUF model-load cold start.
- **Recommended fix:** Report `p50`/`p99` cold-start latency split into (a) framework orchestration cold-start (excluding model weights) and (b) full `NativeEngineClient` GGUF model-load cold-start.

### 2. `Cold-Start Memory Footprint (RSS)`
- **Current claim:** `Cold-Start Memory Footprint (RSS): 4.59 MB` (`Sutra/BENCHMARKS.md:16`).
- **Checks passed:** 1.1, 1.2, 1.6, 1.7
- **Checks failed:** 1.3, 1.4, 1.5, 1.8
- **Diagnosis:** Fails `Check 1.4` because only a single post-Agent-init RSS number is given without labeling `(a) Bare process RSS`, `(b) Post-library-load RSS (before GPU/GGUF init)`, and `(c) Post-GPU/GGUF-init RSS`.
- **Recommended fix:** Measure and report all three RSS phases (`bare`, `post-library-load`, `post-CUDA/GGUF-init`) in `bench_sutra.cpp` and `BENCHMARKS.md`.

### 3. `10-Tool Parallel Execution Speedup`
- **Current claim:** `10-Tool Parallel Execution Speedup (10 x 20 ms I/O tools): 9.91x (20.36 ms vs 201.69 ms)` (`Sutra/BENCHMARKS.md:17, 38-41`).
- **Checks passed:** 1.1, 1.3, 1.6, 1.7
- **Checks failed:** 1.2, 1.4, 1.5, 1.8
- **Diagnosis:** States the tool set (`10 x simulated_io_tool(20 ms)`), concurrency limit (`max_parallel_tools = 10`, 10 workers), and sequential baseline (`201.69 ms`), satisfying `§4.6`, but reports only a single run without `p50`/`p99` percentiles (`Check 1.2`) and lacks Python rival version/warm-cold metadata (`Check 1.8`).
- **Recommended fix:** Run multiple trials to report `p50`/`p99` for both sequential and parallel modes and verify all 10 tool outputs equal `"io_ok"`.

### 4. `Overlapped Inference + Tool CPU Utilization`
- **Current claim:** `Overlapped Inference + Tool CPU Utilization: 554.72% (~5.55 cores active concurrently)` (`Sutra/BENCHMARKS.md:18`).
- **Checks passed:** 1.1, 1.2, 1.6, 1.7
- **Checks failed:** 1.3, 1.4, 1.5, 1.8
- **Diagnosis:** Uses `OverlappedInferenceLlm` (which dispatches 8 `cpu_compute_tool` tasks via `on_early_tool` while running a trigonometric loop on the caller thread); `BENCHMARKS.md` must explicitly disclose that token decoding is simulated via CPU trigonometric work in this synthetic overlap benchmark (`Check 1.3`, `§4.6`).
- **Recommended fix:** Explicitly label the workload as synthetic CPU token-decode + 8-way `cpu_compute_tool` overlap and document in `BENCHMARKS_METHODOLOGY.md`.

### 5. `500-Turn Incremental Checkpoint Latency`
- **Current claim:** `500-Turn Incremental Checkpoint Latency (Turns 10–60 vs Turns 450–500): 3.96 ms/turn -> 3.90 ms/turn (-1.5% drift)` (`Sutra/BENCHMARKS.md:19, 47-48`).
- **Checks passed:** 1.1, 1.6
- **Checks failed:** 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
- **Diagnosis:** Does not report `p50`/`p99` turn latencies (`Check 1.2`), does not disclose whether `flush_checkpoints()` / `fsync` is inside each turn or asynchronous on the background `std::jthread` (`Check 1.3`), and deletes `/tmp/sutra_bench_cpk.bin` without replaying/verifying the checkpoint file size and CRC32 record count before deletion (`Check 1.7`). Also lists an inaccurate kernel version (`6.17.0-20-generic` instead of actual `6.12.113+deb13-amd64`, `Check 1.5`).
- **Recommended fix:** Report `p50`/`p99` turn latency for early and late windows, disclose background delta-append vs `flush_checkpoints()` synchronization, and verify checkpoint file size $> 0$ and record count before cleanup.

### 6. `10,000-Turn Sequential Memory Stability`
- **Current claim:** `10,000-Turn Sequential Memory Stability (test_stress): < 0.5 MB RSS delta` (`Sutra/BENCHMARKS.md:20`).
- **Checks passed:** 1.1, 1.2, 1.3, 1.6, 1.7
- **Checks failed:** 1.4, 1.5, 1.8
- **Diagnosis:** Reports RSS delta without the absolute starting and ending RSS phases (`Check 1.4`), and the hardware header lacks RAM and disk models (`Check 1.5`).
- **Recommended fix:** Report pre-run and post-10,000-turn RSS values alongside the delta in `bench_sutra.cpp` and `BENCHMARKS.md`.

### 7. `Multi-Agent Parallel Throughput Scaling (N = 1, 2, 4, 8, 16)` (5 rows)
- **Current claim:** `1 Agent: 99,444.16 turns/sec (10.06 us/turn)` through `16 Agents: 729,924.68 turns/sec (1.37 us/turn)` (`Sutra/BENCHMARKS.md:29-35`).
- **Checks passed:** 1.6, 1.7
- **Checks failed:** 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 1.9
- **Diagnosis:** Per `§4.6`, `99,444` to `729,924 turns/sec` is physically impossible for real LLM inference and measures **Mock-LLM (`FastReActLlm`) zero-wait framework orchestration throughput**. `BENCHMARKS.md` must explicitly label these rows as `Mock-LLM Orchestration Throughput (Zero-Wait LlmClient)`, state the theoretical minimum orchestration overhead baseline (2 atomic SPSC queue hops + `InferenceArena::reset` + JSON tool-arg parse + `ConversationMemory` ring push $\approx 1.8\ \mu\text{s}$ single-thread floor) and how it was computed (`§4.6`), and provide a companion real-GGUF (`NativeEngineClient`) turn throughput row.
- **Recommended fix:** Relabel Section 2.1 as Mock-LLM Orchestration Throughput, compute and state the theoretical minimum orchestration latency baseline, add a real GGUF (`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`) turn latency/throughput row, and report `p50`/`p99`.

### 8. `Tier 3 Semantic Memory Vector Search (1,000 x 1,536 fp32, CPU SIMD & GPU CUDA)` (2 rows)
- **Current claim:** `CPU Runtime-Dispatched SIMD: 153.87 us/query`; `GPU CUDA Kernel: 2,477.42 us/query` (`Sutra/BENCHMARKS.md:50-52`).
- **Checks passed:** 1.1, 1.3, 1.6
- **Checks failed:** 1.2, 1.4, 1.5, 1.7
- **Diagnosis:** Reports mean query latency over 50 queries without `p50`/`p99` (`Check 1.2`), and in `bench_sutra.cpp:318,327` discards the returned search results (`(void)sem_mem->search(...)`) without verifying top-5 recall against brute-force ground truth (`Check 1.7`).
- **Recommended fix:** Measure `p50`/`p99` query latencies and verify 100% top-5 recall between CPU SIMD, GPU CUDA, and brute-force reference in `bench_sutra.cpp`.

---

## Darshana

Known issues specified in Prompt §3 (Phase 1):
1. `Interactive Scatter Plot (10M points): 3,270.72 FPS` and `Interactive Line Plot (1M points): 8,024.07 FPS` — bare "FPS" without present mode (`Check 1.1`, `1.2`, `1.10`) and measured on cached base layers (`base_dirty == false` on frames 1..999). Requires the Three-Way FPS split (`offscreen`, `CPU submission`, `presented, <mode>`) in §3.1.
2. `SVG Export (1M-pt scatter): 0.01 s (10.4 ms)` — requires the Four-Way SVG measurement (`validation + layout`, `XML generation`, `write-to-disk + fsync`, `file size + element count`) and post-measurement XML/element verification in §3.2.
3. `Memory Footprint (Idle): 4.31 MB` — requires the Three-Way RSS phase split (`bare process`, `post-library-load`, `post-GPU-init`) and `ldd` `libvulkan.so` output in §3.3.
4. `BENCHMARKS_METHODOLOGY.md`, `results.json` + `bench-ci` CMake target, and `KNOWN_ISSUES.md` supersession note (§3.4–§3.6).

---

## Verdict

- **Libraries requiring Phase 1 fixes:** `Darshana`
- **Libraries requiring Phase 2 fixes:** `Hyper Math`, `Bellows`, `Basalt Deep Learning`, `Obsidian`, `Surya`, `Sutra`
- **Libraries cleared as clean:** None (`0` — every library failed at least Checks 1.2, 1.4, 1.5, and 1.7/1.8).
