# Yantra (यन्त्र) — Native C++23 MLOps & Production Backbone

**Yantra** (Sanskrit: यन्त्र, *"instrument, machine, mechanism"*) is a
zero-Python, native C++23 MLOps and production reliability library engineered
for high-frequency experiment tracking, cryptographic model registry and
versioning, lock-free DAG workflow orchestration, SIMD/CUDA vector retrieval,
and statistical bias/drift auditing. Named after the classical Sanskrit term
for precision astronomical and mechanical instruments (such as the *Samrat
Yantra* at Jantar Mantar), Yantra serves as the production supervision layer
for the native C++23 AI stack alongside **Hyper Math**, **Bellows**,
**Basalt**, **Obsidian**, **Surya** (vision), **Sutra** (agents), and
**Darshana** (visualization).

## 10-Line C++23 Quick-Start

```cpp
#include <yantra/yantra.hpp>

auto main () -> int {
  auto store = yantra::ExperimentStore::open ("/var/tmp/yantra_runs").value ();
  auto run   = store.create_run ("resnet18_train").value ();
  (void)run.log_param ("lr", "3e-4");
  (void)run.log_metric (1U, "val_loss", 0.142);
  (void)run.finish ();
  return 0;
}
```

## Documentation Index

- **[YANTRA.md](./YANTRA.md)** — Architectural specification, module guide, concurrency model, error taxonomy, and memory budget.
- **[BENCHMARKS_METHODOLOGY.md](./BENCHMARKS_METHODOLOGY.md)** — Pre-implementation measurement protocol, statistical rigor, and physical plausibility gates.
- **[BENCHMARKS.md](./BENCHMARKS.md)** — Audited hardware benchmark results and Python incumbent comparison (MLflow, Airflow/Prefect/Dagster, FAISS, Fairlearn).
- **[KNOWN_ISSUES.md](./KNOWN_ISSUES.md)** — Autonomous bug-hunting log and architectural constraints.

## License

Copyright (C) 2026 Kival**GNU General Public
License v3.0 or later** (`GPL-3.0-or-later`). See [COPYING](./COPYING) and
[LICENSE](./LICENSE).
