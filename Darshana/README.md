# Darshana (दर्शन) — Native C++23 GPU/CPU Visualization & Statistical Experimentation Engine

**Darshana** is a production-grade, zero-Python C++23 visualization and statistical graphics library engineered to rival Matplotlib, Seaborn, Plotly, and Bokeh while eliminating their interpreter overhead, global state, and full-figure redraw bottlenecks. Built around a declarative Grammar of Graphics specification (`darshana::Spec`) that feeds both a **Vulkan 1.2+ compute-accelerated interactive backend** and a **multi-threaded CPU vector export engine (SVG/PDF)**, Darshana sustains **>3,000 FPS on 10-million-point interactive scatter plots**, **>7,700 FPS on 1-million-point line plots**, **10 ms 1M-point CPU SVG exports**, and **2.27 ms cold-start latency** inside a **4.34 MB** idle memory footprint.

---

## Why the Name *Darshana*?

In Sanskrit, **Darshana** (*darśana*, दर्शन — from the verbal root *dṛś*, "to see") carries a deliberate dual meaning: **seeing / vision** and **philosophical inquiry / system of interpretation**. This duality reflects the library's architectural thesis: a scientific plot is never a naive pixel dump of raw numbers; it is a disciplined *way of seeing* data. Just as classical Indian *Darshanas* separate the underlying reality from the structured lens used to examine it, `darshana::Spec` cleanly separates the declarative mathematical description of a visualization (marks, scales, statistical transforms, and aesthetics) from the physical rendering engine (Vulkan GPU compute pipelines or CPU vector generators) that brings it into view.

---

## Key Architectural Pillars

1. **Declarative Grammar of Graphics (`darshana::Spec`):**
   Every visualization is a pure, JSON-serializable C++23 value (`Spec::to_json()` / `Spec::from_json()`) describing coordinate scales (`Linear`, `Log`, `Categorical`), layered marks (`Scatter`, `Line`, `Bar`, `Area`, `Heatmap`, `Box`, `Violin`, `Histogram`, `Kde`, `Ecdf`, `RegressionLine`, `ConfidenceBand`, `ErrorBar`), aesthetics, and interactive overlays.
2. **Vulkan 1.2+ Layered GPU Backend (`darshana::GpuDevice`, `darshana::Figure`):**
   Uses pre-allocated device-local (`256 MiB`) and host-pinned staging (`128 MiB`) memory pools, Vulkan 1.2 Timeline Semaphores, and SPIR-V compute shaders (`clear_grid`, `scatter_splat`, `line_sdf`, `overlay_comp`). Interactive crosshairs, tooltips, and selection brushes render on a dedicated overlay layer composited over the cached static base layer with **zero static data redraws**.
3. **Multi-Threaded CPU Vector Export (`darshana::export_svg_string`, `darshana::export_pdf_bytes`, `darshana::export_file_async`):**
   Generates publication-grade SVG 1.1 and PDF 1.4 documents directly in C++23 without a browser or Python runtime, supporting non-blocking `std::future<Result<ExportResult>>` asynchronous file export.
4. **Dual Data Path (`darshana::StreamingBuffer` & `darshana::SnapshotData`):**
   Provides an $O(1)$ lock-free pre-allocated ring buffer for 60+ Hz real-time telemetry (`< 0.08 ms` frame jitter) alongside a contiguous columnar store for static datasets.
5. **Native Statistical Graphics (`darshana::stat`):**
   Pure C++23 statistical transforms and high-level plot constructors (`histplot`, `kdeplot`, `distplot`, `ecdfplot`, `regplot`, `boxplot`, `violinplot`, `barplot`, `heatmap`, `corrplot`) with SIMD-accelerated moments, quantiles, Gaussian KDE (Silverman's rule), and Ordinary Least Squares (OLS) confidence bands.
6. **Strict Error & Memory Safety:**
   Zero raw `new`/`delete`, per-frame `std::pmr` `ScratchArena`, re-entrant thread-safe `Figure` instances, and `std::expected<T, darshana::Error>` (`darshana::Result<T>`) across all fallible APIs.

---

## Quick-Start Example (C++23)

```cpp
#include <darshana/darshana.hpp>
#include <vector>

auto main () -> int
{
  // 1. Generate sample data and build a declarative regression + CI plot spec.
  const std::vector<float> x = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
  const std::vector<float> y = { 2.1F, 3.9F, 6.2F, 8.0F, 9.9F, 12.1F };

  auto spec_res
      = darshana::stat::regplot (x, y, 0.95F, "Sensor Calibration Fit");
  if (!spec_res)
    {
      return 1;
    }

  // 2. Create a GPU-accelerated Figure from the declarative Spec.
  auto fig_res = darshana::Figure::create (*spec_res);
  if (!fig_res)
    {
      return 1;
    }
  darshana::Figure &fig = *fig_res;

  // 3. Render the initial frame on the Vulkan 1.2+ backend (or CPU fallback).
  auto stats_res = fig.render_frame ();
  if (!stats_res)
    {
      return 1;
    }

  // 4. Update interactive crosshair overlay WITHOUT redrawing the static base layer.
  darshana::OverlaySpec overlay{};
  overlay.crosshair_enabled = true;
  overlay.crosshair_x = 0.5F;
  overlay.crosshair_y = 0.5F;
  (void)fig.update_overlay (overlay);
  (void)fig.render_frame ();

  // 5. Export publication-quality SVG asynchronously without blocking rendering.
  auto export_fut = fig.export_async ("calibration_fit.svg",
                                      darshana::ExportFormat::Svg);
  auto export_res = export_fut.get ();
  return export_res.has_value () ? 0 : 1;
}
```

---

## Building & Running Tests

```bash
# Configure and build with Vulkan 1.2+ and native architecture optimizations
cmake --preset release-vulkan
cmake --build --preset release-vulkan -j

# Run the full verification suite (GNU style, unit, integration, property, golden, stress)
ctest --preset release-vulkan --output-on-failure

# Run the hardware performance benchmark suite
./build-release-vulkan/bench_darshana
```

See [PRODUCTION.md](./PRODUCTION.md) for deployment, memory budgeting, and concurrency architecture, [BENCHMARKS.md](./BENCHMARKS.md) for full benchmark comparisons against Matplotlib, Seaborn, Plotly, and Bokeh, and [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) for the verification engineering log.
