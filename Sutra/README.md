# Sutra — Native C++23 Agent Orchestration Framework

**Sutra** (`namespace sutra::inline v1`, CMake target `Sutra::Sutra`) is a production-grade, zero-Python C++23 library for building high-throughput, deterministic AI agents and directed workflow graphs. Designed to rival LangChain, LlamaIndex, PydanticAI, CrewAI, and AutoGen while eliminating their systemic memory and concurrency pathologies, Sutra delivers **0.38 ms** cold-start latency, a **4.59 MB** memory footprint, compile-time reflected JSON Schema tool registration without `std::any`, zero-copy `std::span` conversation state borrowing, lock-free work-stealing parallel tool execution (**9.91x** speedup on 10 tools), incremental CRC32-verified binary checkpointing, and SIMD/CUDA-accelerated three-tier memory.

## Why "Sutra"?

In Sanskrit, **Sūtra** (सूत्र) means *"thread"* or *"the thread that binds"*—referring both to a literal cord that weaves disparate elements into a unified fabric and to a concise collection of aphoristic rules. In this architecture, **Sutra** is the native C++23 thread that binds compile-time reflected tools, three-tier memory (short-term ring buffer, persistent long-term store, and AVX-512/AVX2/CUDA semantic vector retrieval), static plan linting, and lock-free execution into a single coherent agent runtime, serving as the sibling orchestration library to **Surya** (the computer vision library).

## Quick-Start Example (10 Lines)

```cpp
#include <sutra/sutra.hpp>

auto add_ints (int64_t a, int64_t b) noexcept -> sutra::Result<std::string> { return std::to_string (a + b); }

auto main () -> int {
  auto tools = std::make_shared<sutra::ToolRegistry> ();
  (void)SUTRA_REGISTER_TOOL (*tools, &add_ints, "add", "Adds two integers", "a", "b");
  auto mem = sutra::MemorySystem::create (64U, 128U);
  sutra::InferenceArena arena (65536U);
  auto res = tools->invoke ("add", sutra::ToolContext{ mem->short_term ().borrow_view (), arena.resource () }, R"({"a":20,"b":22})");
  return (res && *res == "42") ? 0 : 1;
}
```

See [PRODUCTION.md](./PRODUCTION.md) for architecture, concurrency guarantees, and build instructions, [BENCHMARKS.md](./BENCHMARKS.md) for performance measurements on Intel i7-12700K + NVIDIA RTX 3070, [SUTRA_LLM_BACKENDS.md](./SUTRA_LLM_BACKENDS.md) for the live GGUF `LlamaCppClient` backend, 7-family `ToolCallParser`, and logit parity harness, [VACH.md](./VACH.md) for the `Vach::Vach` native voice I/O library (`Vad`, `Stt`, `Tts`, `AudioIo`), [SPHINX.md](./SPHINX.md) for the end-to-end Sphinx voice assistant vertical slice, and [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) for the verification bug-hunting log.
