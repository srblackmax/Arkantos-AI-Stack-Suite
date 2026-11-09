---
name: obsidian-development-loop
description: Autonomous C++23 engineering workflow for Obsidian (Plan -> Code -> Build -> Test -> Audit -> Refactor) enforcing GNU Coding Standards, Basalt integration, and zero-overhead performance.
---

# Obsidian Development Loop (`SKILL.md`)

This skill defines the mandatory six-stage autonomous engineering cycle for
developing, verifying, and auditing components in **Obsidian**.

```
+--------+    +--------+    +---------+    +--------+    +---------+    +----------+
| 1.Plan | -> | 2.Code | -> | 3.Build | -> | 4.Test | -> | 5.Audit | -> |6.Refactor|
+--------+    +--------+    +---------+    +--------+    +---------+    +----------+
     ^                                                                        |
     +------------------------------------------------------------------------+
```

---

## Stage 1: Plan

Before writing or modifying implementation files:
1. Review `ARCHITECTURE.md` and `TASKS.md` to confirm how the component fits
   into the `obsidian` namespace hierarchy and interacts with `basalt::Tensor`
   and `basalt::nn::Module`.
2. Specify algorithmic complexity targets (time and space), memory layout
   (contiguous arenas, `std::pmr`, `mmap`, SIMD alignment), and error modes
   (`obsidian::Result<T>` / `std::expected<T, obsidian::Error>`).
3. Define the C++23 concepts, policy templates, and range interfaces required
   by the module.

---

## Stage 2: Code

Write headers in `include/obsidian/` and implementation units in `src/`:
1. **Header Hygiene**:
   - Include the exact GPLv3 license notice at lines 1-14.
   - Guard headers with `#ifndef OBSIDIAN_<MODULE>_HPP`.
   - Document every public class, concept, struct, and function with Doxygen
     block comments (`/** ...  */`), explicitly stating `@pre` preconditions
     and `@complexity` asymptotic bounds.
   - Follow two-space sentences (`/* Sentence one.  Sentence two.  */`).
2. **GNU Formatting**:
   - 2-space indentation, zero tabs, <= 79 characters per line.
   - Return type on the line preceding the function name.
   - Space before `(` in function calls, definitions, and control statements.
   - Braces `{` and `}` on their own lines, indented properly.
3. **High-Performance Idioms**:
   - Use `std::string_view` and `std::span` for non-owning slices.
   - Use `std::pmr::monotonic_buffer_resource` for scratch allocations in
     tokenization and parsing.
   - Dispatch heavy linear algebra, attention, embeddings, and autograd
     through Basalt (`basalt::ops`, `basalt::nn::functional`, `basalt::nn`).

---

## Stage 3: Build

Compile the library, CLI tools, and test suite using GCC with strict flags:
```bash
make -j$(nproc) CXXFLAGS="-std=c++23 -O3 -march=native -Wall -Wextra -Werror -pedantic"
```
- If compilation fails:
  1. Parse the exact file, line number, and diagnostic emitted by `g++`.
  2. Fix the root cause in the source or header file.
  3. Re-run `make` until zero warnings and zero errors remain.

---

## Stage 4: Test

Execute the automated test suite via `make check`:
1. Write deterministic unit tests in `tests/test_<component>.cpp` covering:
   - Normal execution paths and numerical accuracy.
   - Edge cases (empty strings, single-token inputs, unicode/UTF-8 boundaries,
     out-of-vocabulary lookups, zero-length sequences).
   - Recoverable error paths (verifying `Result<T>` returns an `Error` rather
     than aborting on malformed input).
   - Gradient flow verification through Basalt (`backward ()` and
     `basalt::autograd::gradcheck`).
2. If any test fails:
   - Isolate the failure, write a targeted regression test, fix the defect,
     and re-run `make check` until 100% green.

---

## Stage 5: Audit

Continuously audit the codebase across four dimensions:
1. **GNU Coding Style Audit**:
   - Run `./tests/check_gnu_style` (compiled C++ linter) to verify GPLv3
     headers, zero tabs, no trailing whitespace, <= 79 columns, and GNU
     formatting compliance across all `.hpp`, `.h`, and `.cpp` files.
2. **Memory and Undefined Behavior Audit**:
   - Build and run the test suite with AddressSanitizer and
     UndefinedBehaviorSanitizer:
     `./configure --enable-sanitizers && make clean && make check`
3. **Thread-Safety Audit**:
   - Verify immutable shared data structures (e.g., `BPE_Tokenizer`,
     `Mmap_Embeddings`, `Porter_Stemmer`) are lock-free and re-entrant across
     concurrent threads.
4. **Doxygen & Precondition Audit**:
   - Verify all public headers in `include/obsidian/` contain complete Doxygen
     comments with complexity and precondition annotations.

---

## Stage 6: Refactor

Once tests and audits pass:
1. Eliminate redundant copies, tighten cache locality, and hoist loop-invariant
   computations.
2. Re-run Stages 3-5 to confirm no regressions were introduced.
3. Update `TASKS.md` to record completion before advancing to the next module.
