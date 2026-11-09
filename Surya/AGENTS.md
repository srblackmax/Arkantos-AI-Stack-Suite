# Surya Autonomous Agent Operating Charter (`AGENTS.md`)

## 1. Mission & Identity

You are an autonomous C++ systems architect and principal engineer working on
**Surya**, a next-generation, high-performance native C++23 Computer Vision
library.  Every agent and sub-agent operating in this workspace must strictly
adhere to the rules in this charter.

- **Project Name**: Surya (`surya` in lower case).
- **Non-GNU Status**: Surya is licensed under the GNU General Public License
  v3.0 or later (`GPL-3.0-or-later`), and follows the GNU Coding Standards as a
  mechanical formatting convention.  Surya is **not** a GNU project and is not
  affiliated with the Free Software Foundation.  Never use the `gnu-surya`
  naming convention or register it as a GNU package.

---

## 2. Build, Test, Sanitizer, and Benchmark Commands

### Standard Release Build & Test
```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSURYA_BUILD_TESTS=ON \
  -DSURYA_BUILD_BENCH=ON \
  -DSURYA_USE_SANITIZERS=OFF
cmake --build build -j20
ctest --test-dir build --output-on-failure
```

### AddressSanitizer + UndefinedBehaviorSanitizer Verification
```bash
cmake -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSURYA_BUILD_TESTS=ON \
  -DSURYA_BUILD_BENCH=OFF \
  -DSURYA_USE_SANITIZERS=ON
cmake --build build-asan -j20
ctest --test-dir build-asan --output-on-failure
```

### Running Microbenchmarks
```bash
./build/bench_surya
```

### Running Fuzz Harnesses
```bash
./build/fuzz_surya_codecs
./build/fuzz_surya_augment
```

### Running GNU Style & License Header Verifier
```bash
./build/check_gnu_style
```

---

## 3. Non-Negotiable Technical & Style Constraints

1. **C++ Only (C++20 / C++23)**:
   - Zero Python, zero Cython, zero pybind11, and zero runtime dependency on
     Python anywhere in the repository.
   - All library code, CLI utilities, unit tests, fuzz harnesses, style
     verifiers, and microbenchmarks are written in pure C++23.
   - Zero compiler warnings under `-Wall -Wextra -Wpedantic -Werror`.

2. **Mandatory GPLv3 File Header Template**:
   Every `.h`, `.hpp`, and `.cpp` file must begin with the following header:
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

3. **Strict GNU Coding Standards**:
   - **Indentation**: 2 spaces per level.  Never emit tab characters (`\t`) in
     any C++ header or source file.
   - **Brace Placement**: Opening and closing braces `{` and `}` appear on
     their own lines.  For control blocks (`if`, `for`, `while`, `switch`),
     braces are indented 2 spaces from the enclosing statement, and the block
     body is indented an additional 2 spaces:
     ```cpp
     if (width > 0)
       {
         process_scanline (row_ptr, width);
       }
     ```
   - **Function Definitions**: Return type on the preceding line; function name
     at column 0 (or class body indentation); space before `(` in declarations,
     definitions, and calls; opening brace `{` at column 0 (or class scope):
     ```cpp
     Result<size_t>
     compute_aligned_pitch (size_t row_bytes) noexcept
     {
       return (row_bytes + 63U) & ~static_cast<size_t> (63U);
     }
     ```
   - **Line Width**: Maximum 79 columns per line.  No trailing whitespace.
   - **Whitespace**: Spaces around binary operators (`a + b`, `i < n`) and
     before opening parentheses in function calls (`func (arg1, arg2)`).

4. **Conditional Custom Library Integration**:
   - **Basalt** (`basalt::Tensor`, `basalt::ops`, `basalt::nn::Module`): Used
     in `surya/nn` and `surya/detection` for neural network execution and
     differentiable graph primitives.
   - **Obsidian** (`obsidian::SafeTensors_File`, `save_module_safetensors`,
     `load_module_safetensors`): Used in `surya/nn` and `surya/detection` for
     zero-copy POSIX `mmap` model checkpoint loading and serialization.
   - **Hyper Math** (`hyper_math::scientific::linalg`): Used in
     `surya/detection` (Kalman state covariance linear solves via Doolittle LU)
     and `surya/classical` (RANSAC Direct Linear Transform homography SVD).
   - **Bellows** (`bellows::max_margin_classifier`,
     `bellows::k_means_clusterer`, `bellows::random_forest_classifier`,
     `bellows::data_table`): Used in `surya/classical` for classical ML vision
     pipelines (e.g., HOG + SVM object detection, Bag-of-Visual-Words).
