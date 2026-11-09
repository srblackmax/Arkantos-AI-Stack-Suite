# AGENTS.md — Bellows Project Engineering Rules

All automated agents and human contributors working in this repository MUST
read and obey the rules in this file before creating or modifying any code,
build configuration, test, or documentation.

## 1. Project Identity and Naming Prohibitions

- **Project Name**: Bellows (`bellows` in lower case). Bellows is a plain
  name, NOT a recursive acronym.
- **No `gnu-` Prefix**: NEVER use the `gnu-` prefix on any binary, library,
  header, target, script, or file. That prefix is reserved for programs that
  are part of the GNU system, and Bellows is not.
- **No GNU / FSF Affiliation Claims**: NEVER describe Bellows as a GNU
  package, GNU project software, or Free Software Foundation (FSF) affiliated
  in any file, comment, banner, or documentation.
- **Copyright Notices**: Every source and header file must begin with a
  standard GPLv3 license notice whose copyright line reads:
  `Copyright (C) 2026 Kival
  Do NOT attribute copyright to the Free Software Foundation or the GNU
  Project.
- **Mandatory README Disclaimer**: The top-level `README` MUST contain the
  following paragraph verbatim near the top:
  > Bellows is an independent GPLv3 C++ machine-learning library. It
  > follows the GNU Coding Standards as a style convention; it is not part
  > of the GNU Project and is not affiliated with the Free Software
  > Foundation.
- **No Transliterated Names**: Do NOT copy class names, function names,
  hyperparameter strings, or file layouts from Scikit-learn, XGBoost,
  LightGBM, or CatBoost (e.g. no `BaseEstimator`, `fit_transform`,
  `predict_proba`, `StandardScaler`, `OneHotEncoder`, `GridSearchCV`,
  `DMatrix`, `XGBClassifier`, `LGBMRegressor`, `CatBoostClassifier`, etc.).

## 2. Coding Style and Naming Conventions (GNU Coding Standards)

Bellows adopts the GNU Coding Standards strictly as a mechanical formatting
and CLI convention:

- **Indentation and Whitespace**:
  - Two (2) spaces per indentation level.
  - NEVER use tab characters (`\t`) in C++, header, CMake, or documentation
    files (Makefiles are the sole exception where POSIX Make syntax requires
    tab-prefixed recipe lines).
  - Put spaces around binary operators (`a + b`, `x = y`) and before open
    parentheses in function calls and control statements (`foo (x)`,
    `if (cond)`, `for (std::size_t i = 0; i < n; ++i)`).
- **Brace Placement**:
  - Opening and closing braces `{` and `}` go on their own lines.
  - For control blocks (`if`, `for`, `while`, `switch`), braces are indented
    2 spaces from the enclosing statement, and the body is indented a further
    2 spaces:
    ```cpp
    if (row_count > 0)
      {
        process_row (0);
      }
    ```
- **Identifier and File Naming**:
  - **Files**: `lower_case_with_underscores.hpp` and
    `lower_case_with_underscores.cpp`.
  - **Namespaces, Functions, Variables, Constants, and Member Fields**:
    `lower_case_with_underscores`.
  - **Concrete Types (structs, classes, enums) and C++20 Concepts**:
    `lower_case_with_underscores` (aligning with the C++20 standard library
    and `std::ranges` conventions, e.g. `bellows::table_view`,
    `bellows::supervised_fittable`, `bellows::stagewise_booster`).
  - **Template Type Parameters**: `Upper_Snake_Case` or `PascalCase`
    (e.g. `template <typename Cut_Policy, typename Growth_Policy>`) to
    visually distinguish template parameters from concrete types and concepts.
- **Comments**:
  - All comments must be written in complete English sentences with proper
    capitalisation and trailing punctuation, followed by two spaces between
    sentences where applicable.
- **CLI Diagnostics and Flags**:
  - All CLI executables (`bellows`, `bellows-train`, `bellows-infer`,
    `bellows-check`) must support `--help` and `--version`.
  - Diagnostic messages must follow the GNU format:
    `program:source:line: message` or `program: message`.

## 3. Directory Standards

```text
bellows/
├── AGENTS.md               # Project engineering rules (this file)
├── COPYING                 # Full text of the GNU General Public License v3
├── README                  # Build/usage guide with mandatory verbatim disclaimer
├── NEWS                    # Release history
├── ChangeLog               # Chronological change records
├── AUTHORS                 # Author attribution
├── INSTALL                 # Installation instructions
├── Makefile                # GNU-conventional top-level build wrapper
├── CMakeLists.txt          # CMake build definition generating standard targets
├── bellows.pc.in           # pkg-config template producing bellows.pc
├── include/
│   └── bellows/            # Public header root (#include <bellows/...>)
│       ├── core/           # Concepts, table views, parameter descriptors, CPOs
│       ├── preprocess/     # Scaling, mending, encoding, bucketing, selection
│       ├── estimators/     # Linear, kernel/neighbour, density, and clustering
│       ├── boosting/       # Unified stagewise boosting core and policies
│       ├── composition/    # Compile-time and runtime chains, CV, tuning
│       ├── metrics/        # Composable evaluation metric function objects
│       └── io/             # Versioned binary serialisation & low-latency infer
├── src/                    # Compiled library sources (libbellows.a / .so) & CLIs
├── tests/                  # Test suite compiled into bellows-check
├── examples/               # Self-contained usage examples
├── doc/                    # Architectural design and API documentation
└── scripts/                # Style verification and packaging helpers
```

## 4. Verification and Test Command

- **Single Documented Test Command**:
  ```sh
  make check
  ```
- Running `make check` builds the libraries (`libbellows.a` and
  `libbellows.so`), the CLI utilities (`bellows`, `bellows-train`,
  `bellows-infer`), and the test driver (`bellows-check`), then executes both
  the unit/integration test suite and the automated GNU style & license
  compliance verifier.
- Standard GNU targets supported by the top-level `Makefile`:
  `all`, `check`, `install`, `clean`, `dist`.
