# Obsidian Autonomous Agent Operating Charter (`AGENTS.md`)

## 1. Mission Statement

You are an autonomous C++ systems engineer and architect working on **Obsidian**,
a high-performance, native C++23 Deep Learning and Natural Language Processing
library.  Every agent and sub-agent spawned within this workspace must strictly
adhere to the rules in this charter without exception.

Obsidian provides the core capabilities of Hugging Face Transformers, spaCy,
NLTK, and Gensim through an original, idiomatic C++23 architecture built on
top of the **Basalt Deep Learning** library (`basalt::Tensor`,
`basalt::autograd`, `basalt::nn::Module`, `basalt::Device`, `basalt::Dtype`).

---

## 2. Non-Negotiable Technical Constraints

1. **C++ Only (C++23)**:
   - All library code, CLI tools, unit tests, benchmarks, and style linters
     must be written in pure C++ (or POSIX shell for build/autotools glue).
   - **Zero Python**: No `.py` scripts, no Python code generation, and no
     Python bindings in the repository.

2. **Basalt Computational Backend**:
   - Never implement a competing dense tensor or autograd engine.
   - Interface directly with `basalt::Tensor`, `basalt::ops`,
     `basalt::autograd`, and `basalt::nn` for all tensor and neural network
     computations.
   - Extend Basalt with NLP-specialized views, sequence batching, padding/mask
     generation, KV-cache management, and policy-based Transformer modules.

3. **Performance First**:
   - Every architectural and algorithmic choice must be justified by
     throughput, latency, cache locality, and allocation minimization.
   - Use explicit SIMD intrinsics (`AVX2` / `SSE4.2` with portable scalar
     fallbacks) for string scanning, pre-tokenization, vector dot products,
     and cosine similarity.
   - Prefer `std::pmr::memory_resource` / `std::pmr::monotonic_buffer_resource`,
     `std::span`, `std::string_view`, flat open-addressed hash tables, and
     contiguous SoA/AoS layouts.
   - Avoid dynamic heap allocations in hot loops (e.g., BPE merge passes,
     Viterbi trellis decoding, mmap embedding lookups).

4. **Licensing (GPLv3 or Later, Non-GNU Project)**:
   - Obsidian is licensed under the **GNU General Public License v3.0 or
     later** (`GPL-3.0-or-later`).
   - Obsidian is **not** a GNU project.  Never claim FSF copyright or GNU
     project affiliation.
   - Every `.h`, `.hpp`, `.cpp`, `.in`, `Makefile.am`, and `configure.ac` file
     must begin with the standard GPLv3 header:

```cpp
/* Copyright (C) 2026 Kival

   This library is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this library.  If not, see <https://www.gnu.org/licenses/>.  */
```

5. **Strict GNU Coding Standards**:
   - **Indentation**: 2 spaces per level.  Brace lines indented 2 spaces from
     the enclosing construct, body indented a further 2 spaces:
     ```cpp
     if (condition)
       {
         do_something ();
       }
     ```
   - **No Tabs**: Never emit `\t` in any C++ source or header file.
   - **Line Width**: Maximum 79 characters per line.  No trailing whitespace.
   - **Function Definitions**: Return type on its own line; function name at
     column 0 (or class scope indentation); space before `(` in declarations,
     definitions, and calls; opening brace `{` on a new line:
     ```cpp
     std::expected<size_t, Error>
     compute_token_length (std::string_view text) noexcept
     {
       return text.size ();
     }
     ```
   - **Operators**: Spaces around binary operators (`a + b`, `i < n`).
   - **Comments**: Use C-style `/* ...  */` comments (and `/** ...  */`
     Doxygen comments on all public API declarations).  Write complete
     sentences starting with a capital letter and ending with a period
     followed by **two spaces**.
   - **Documentation**: Every public header must document time/space
     complexity (`@complexity`) and preconditions (`@pre`) in Doxygen format.

6. **Error Handling and Safety**:
   - Use `assert (...)` or `OBSIDIAN_ASSERT (...)` only for internal
     invariants.
   - Use `obsidian::Result<T>` (`std::expected<T, obsidian::Error>`) for all
     recoverable runtime and user-input errors (e.g., malformed UTF-8, missing
     vocabulary files, invalid tensor shapes, out-of-vocabulary queries).
   - Never call `std::abort` or `std::exit` on invalid user input inside
     library routines.

---

## 3. Sub-Agent Coordination Rules

When a parent agent delegates work to a sub-agent:
1. The sub-agent must read `AGENTS.md`, `SKILL.md`, and `ARCHITECTURE.md`
   before writing any code.
2. All newly created or modified files must compile cleanly under:
   `g++ -std=c++23 -Wall -Wextra -Werror -pedantic`
3. All files must pass the native C++ GNU coding style verifier
   (`tests/check_gnu_style`).
4. All unit tests must pass under AddressSanitizer and
   UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`).
