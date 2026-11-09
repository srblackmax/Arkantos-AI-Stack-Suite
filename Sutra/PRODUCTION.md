# Sutra Production Operations & Architecture Manual (`PRODUCTION.md`)

**Sutra** (`namespace sutra::inline v1`, CMake target `Sutra::Sutra`) is a native C++23 agent orchestration library engineered for deterministic resource lifetimes, lock-free hot-path concurrency, zero-copy state borrowing, and typed `std::expected<T, sutra::AgentError>` error propagation.

---

## 1. Hardware Requirements

| Component | Minimum Specification | Reference Target Specification |
| :--- | :--- | :--- |
| **CPU Architecture** | x86-64 (SSE4.2 / Scalar fallback) | Intel Core i7-12700K (8 P-cores, 4 E-cores, 20 threads) |
| **SIMD Instruction Sets** | Runtime CPUID dispatch (`AVX-512F`, `AVX2+FMA`, `SSE4.2`, `Scalar`) | `AVX2 + FMA` (native on Alder Lake) / `AVX-512F` auto-detected |
| **System RAM** | 64 MB available RSS | 32 GB DDR4/DDR5 (Cold-start RSS: **4.59 MB**) |
| **GPU Acceleration** | Optional (graceful CPU fallback when absent) | NVIDIA GeForce RTX 3070 (8 GB GDDR6 VRAM, `sm_86`, CUDA 12.4) |
| **VRAM Budget** | 0 MB (CPU mode) / $\ge 64\text{ MB}$ (CUDA mode) | $\le 8\text{ GB}$ (`sizeof(float) * num_docs * dim` per GPU batch query) |
| **Compiler & OS** | GCC $\ge 14$ or Clang $\ge 18$, Linux x86-64 (`-std=c++23`) | Ubuntu Linux 6.17, GCC 14.2.0, CUDA 12.4 (`nvcc` + `g++-13`) |

---

## 2. Concurrency Model & Thread-Safety Guarantees

Sutra eliminates the undefined concurrent behavior of Python and legacy C++ agent frameworks by enforcing explicit, documented synchronization invariants across every public class:

### 2.1. Per-Component Concurrency Matrix

| Class / Component | Thread-Safety Guarantee | Synchronization Primitive |
| :--- | :--- | :--- |
| `sutra::ToolRegistry` | **Thread-Safe** (concurrent `invoke`, `has_tool`, `find_tool`, `export_mcp_tools_json`; serialized `register_tool`) | `std::shared_mutex` (shared read lock during lookup; tool function executes outside lock) |
| `sutra::ShortTermMemory` | **Thread-Safe** (concurrent `borrow_view`, `delta_since`, `size`; serialized `append`, `clear`) | `std::shared_mutex` over fixed-capacity ring buffer |
| `sutra::LongTermMemory` | **Thread-Safe** (concurrent `get`, `save_to_file`, `size`; serialized `put`, `erase`, `load_from_file`) | `std::shared_mutex` |
| `sutra::SemanticMemory` | **Thread-Safe** (concurrent `search`, `size`, `dimension`; serialized `add_document`, `clear`) | `std::shared_mutex` |
| `sutra::LockFreeQueue<T>` | **Lock-Free MPMC** (`try_push`, `try_pop` use zero mutexes) | Cache-line aligned (`alignas(64)`) `std::atomic<std::size_t>` sequence ring |
| `sutra::WorkStealingPool` | **Lock-Free Hot Path** (`submit`, `try_run_one` use zero mutexes) | Round-robin per-worker `LockFreeQueue<TaskFn>` + C++20 `std::atomic<uint32_t>::wait/notify` |
| `sutra::ToolScheduler` | **Thread-Safe & Stateless per Batch** (`execute_batch` safe across $N$ concurrent agents) | Lock-free atomic wave cursor + atomic task-completion barrier |
| `sutra::WorkflowGraph` | **Thread-Safe Execution** (`execute` safe for concurrent runs once validated by `PlanLinter`) | Topological parallel wave dispatch on `WorkStealingPool` |
| `sutra::CheckpointManager` | **Thread-Safe Asynchronous I/O** (`enqueue_delta` is non-blocking on the agent hot path) | Lock-free `LockFreeQueue<CheckpointDelta>` drained by dedicated `std::jthread` writer |
| `sutra::AgentRuntime` | **Thread-Safe Multi-Agent Executor** (`run` and `run_async` support $N$ concurrent agents) | Shared `WorkStealingPool` + per-run `AsyncRunHandle` completion queue |
| `sutra::Agent` | **Explicit Concurrency Policy** (configurable rejection or serialization of same-agent calls) | `std::atomic<uint32_t> active_callers_` + `std::mutex turn_mtx_` |

### 2.2. Explicit Same-Agent Concurrency Policy (`sutra::Agent`)

When two threads invoke `AgentRuntime::run(agent, ...)` on the **same** `sutra::Agent` instance simultaneously:
1. **Default Strict Mode (`AgentConfig::reject_concurrent_same_agent = true`)**:
   `agent.active_callers_.fetch_add(1, std::memory_order_acq_rel)` detects contention in $O(1)$ without blocking and immediately returns `std::unexpected(AgentError::framework(FrameworkErrorCode::ConcurrencyViolation, ...))`.
2. **Serialized Turn Mode (`AgentConfig::reject_concurrent_same_agent = false`)**:
   Concurrent callers on the same `Agent` instance are serialized at the turn boundary via `agent.turn_mtx_`, while all underlying memory tiers (`ShortTermMemory`, `LongTermMemory`, `SemanticMemory`) remain protected by their internal `std::shared_mutex`.
3. **Multi-Agent Parallelism ($N$ Distinct `Agent` Instances)**:
   Running $N$ distinct `sutra::Agent` instances concurrently on a shared `sutra::AgentRuntime` shares zero mutable inference state and scales lock-freely across all CPU cores.

---

## 3. Error Handling Taxonomy (`std::expected<T, sutra::AgentError>`)

All fallible Sutra APIs return `sutra::Result<T>` (`std::expected<T, sutra::AgentError>`). No exceptions cross ABI boundaries. `sutra::AgentError` wraps a `std::variant<LlmError, ToolError, PlanningError, FrameworkError>`:

### 3.1. `sutra::LlmError` (`ErrorCategory::Llm`)
- `LlmErrorCode::ConnectionFailure`: Transport or endpoint unreachable; includes HTTP/transport status code.
- `LlmErrorCode::RateLimited`: Upstream token/request quota exceeded; caller may retry with backoff.
- `LlmErrorCode::ContextOverflow`: Prompt exceeds maximum context window (e.g., $> 4\text{ MB}$ prompt guard).
- `LlmErrorCode::InvalidResponse`: Malformed output from inference backend.

### 3.2. `sutra::ToolError` (`ErrorCategory::Tool`)
- `ToolErrorCode::ToolNotFound`: Requested tool name is not registered in `ToolRegistry`.
- `ToolErrorCode::InvalidInput`: Malformed JSON object passed as tool arguments.
- `ToolErrorCode::SchemaViolation`: Missing required field or JSON type mismatch (`string`, `integer`, `number`, `boolean`, `array`, `object`).
- `ToolErrorCode::ExecutionFailure`: Tool returned an error or threw an internal exception (caught by the non-erased dispatch trampoline).
- `ToolErrorCode::Timeout`: Tool execution exceeded its configured `timeout_ms` budget.
- **Recovery Strategy**: Tool errors never crash the agent. `AgentRuntime::run` formats the error into a `MessageRole::Tool` observation (`"ERROR: <message>"`) and feeds it back to the `Planner` / `LlmClient` on the next step for self-correction. If the agent exhausts `max_steps` after encountering tool errors, `AgentRuntime::run` returns the last `ToolError` as the final `std::unexpected` result.

### 3.3. `sutra::PlanningError` (`ErrorCategory::Planning`)
- `PlanningErrorCode::NoValidPlan`: Planner produced neither a tool call nor a final answer.
- `PlanningErrorCode::MaxStepsExceeded`: Agent loop reached `AgentConfig::max_steps` without converging.
- `PlanningErrorCode::CycleDetected`: `PlanLinter` or `ToolScheduler` detected a directed cycle (Kahn's algorithm visited fewer vertices than total nodes).
- `PlanningErrorCode::DanglingDependency`: An edge or `depends_on_call_ids` entry references a non-existent node or call ID.
- `PlanningErrorCode::SchemaMismatch`: `PlanLinter` detected incompatible `output_schema` and `input_schema` types across a `WorkflowEdge`.

### 3.4. `sutra::FrameworkError` (`ErrorCategory::Framework`)
- `FrameworkErrorCode::ResourceExhaustion`: Memory allocation or lock-free queue capacity exhausted.
- `FrameworkErrorCode::InvalidConfiguration`: Zero capacity, empty name/key, or null dependency passed to a factory method.
- `FrameworkErrorCode::ConcurrencyViolation`: Concurrent call on the same `Agent` instance when `reject_concurrent_same_agent == true`.
- `FrameworkErrorCode::CheckpointCorrupted`: Binary checkpoint header mismatch or CRC32 checksum failure during `CheckpointManager::replay`.
- `FrameworkErrorCode::CudaError`: Wrapped CUDA runtime error returned by `SUTRA_CUDA_CHECK(expr)` including `cudaGetErrorString(err)`.

---

## 4. Memory Budgeting & Deterministic Resource Lifetime

1. **Zero Raw `new`/`delete`**:
   Enforced statically by `tests/check_gnu_style.cpp`. All heap allocations use RAII containers (`std::vector`, `std::unique_ptr`, `std::shared_ptr`).
2. **Cold-Start Memory Footprint ($\le 20\text{ MB}$ SLA)**:
   Measured cold-start RSS is **4.59 MB** on Linux x86-64.
3. **Per-Turn Monotonic Scratch Arena (`sutra::InferenceArena`)**:
   Pre-allocates a contiguous `std::vector<std::byte>` (default `kDefaultArenaCapacityBytes = 64 KB`) backing a `std::pmr::monotonic_buffer_resource`. Reclaimed in $O(1)$ at the start of every planning step via `scratch_arena.reset()`.
4. **Tier 1 Short-Term Memory (`sutra::ShortTermMemory`)**:
   Fixed-capacity ring buffer pre-allocated once at construction:
   $$\text{Bytes}_{\text{STM}} \approx \text{capacity} \times \left(\text{sizeof}(\text{Message}) + \overline{|\text{content}|}\right)$$
   Tool calls borrow history via `sutra::ConversationSpanView` (two `std::span<const Message>` slices) with **zero bytes copied**.
5. **Tier 3 Semantic Memory (`sutra::SemanticMemory`)**:
   Contiguous row-major embedding matrix:
   $$\text{Bytes}_{\text{Semantic}} = N_{\text{docs}} \times D_{\text{dim}} \times 4\text{ bytes} + N_{\text{docs}} \times \text{sizeof}(\text{DocumentMeta})$$
   For $N_{\text{docs}} = 10{,}000$ and $D_{\text{dim}} = 1{,}536$, the flat matrix occupies exactly **61.44 MB** of contiguous RAM (or VRAM when offloaded via `sutra::Tensor`).

---

## 5. Reproducible Build, Verification & Sanitizer Commands

```bash
# 1. Configure, build, and run AddressSanitizer + UndefinedBehaviorSanitizer suite
cmake --preset release-asan
cmake --build --preset release-asan -j 20
ctest --preset release-asan --output-on-failure

# 2. Configure, build, and run ThreadSanitizer (TSan) race-detection suite
cmake --preset release-tsan
cmake --build --preset release-tsan -j 20
ctest --preset release-tsan --output-on-failure

# 3. Configure, build, test, and benchmark Release + CUDA (sm_86 RTX 3070 + fat-binary SIMD)
cmake --preset release-cuda
cmake --build --preset release-cuda -j 20
ctest --preset release-cuda --output-on-failure
./build-release-cuda/bench_sutra

# 4. Verify GNU Coding Formatting & Static Analysis (clang-format + clang-tidy)
$HOME/Android/Sdk/ndk/27.0.12077973/toolchains/llvm/prebuilt/linux-x86_64/bin/clang-format \
  --dry-run --Werror include/sutra/*.hpp src/*.cpp src/gpu/*.hpp tests/*.cpp bench/*.cpp

$HOME/Android/Sdk/ndk/27.0.12077973/toolchains/llvm/prebuilt/linux-x86_64/bin/clang-tidy \
  src/*.cpp -- -std=c++23 --target=x86_64-linux-gnu -D__cpp_concepts=202002L \
  -DSUTRA_ENABLE_STATS=1 -DSUTRA_HAS_CUDA=1 -Iinclude -Isrc \
  -isystem/usr/include/c++/14 -isystem/usr/include/x86_64-linux-gnu/c++/14 \
  -isystem/usr/include/x86_64-linux-gnu -isystem/usr/include
```
