# Scoped Audit: Sphinx System Observation (Pre-Patch State)

## 1. `read_file` Behavior on Absolute Paths Outside the Sandbox

- **Implementation**: `tool_read_file` (`src/tools.cpp:1135-1205`) validates its `path` argument through `resolve_sandboxed_path(path)` (`src/tools.cpp:971-1026`).
- **Behavior**:
  - `resolve_sandboxed_path` canonicalizes the active sandbox root (`ToolEnvironment::sandbox_dir()`, defaulting to `$HOME/sphinx-workspace`) and rejects any path containing `..` components that escape the root or any absolute path outside the sandbox root.
  - It does **not** silently strip leading path components or rewrite outside paths into sandbox-relative paths.
  - When given an outside path (e.g., `/etc/os-release` or `/etc/passwd`), `tool_read_file` returns `sutra::AgentError::ToolSandboxViolation` with the exact message:
    ```text
    Cannot read from <path>: path is outside the sandbox. The allowed root is <sandbox_dir>. Use a relative path within the sandbox, or omit the leading path components.
    ```

## 2. `run_command` Behavior on Non-Allowlisted Commands

- **Implementation**: `tool_run_command` (`src/tools.cpp:1348-1438`) validates `cmd` via `validate_allowed_command(cmd)` (`src/tools.cpp:1028-1099`) before spawning a child process with `execvp`.
- **Behavior**:
  - Rejects empty commands and any command string containing shell metacharacters (`;`, `|`, `&`, `$`, `` ` ``, `>`, `<`, `\n`, `\r`) with `sutra::AgentError::ToolNotAllowed`.
  - Checks the trimmed command against `kAllowedPrefixes` (`cmake`, `ctest`, `gcc`, `g++`, `clang`, `clang++`, `make`, `ninja`, `cargo`, `rustc`, `git status`, `git diff`, `git log`, `ls`, `pwd`, `cat`, `wc`, `grep`, `find`, `head`, `tail`, `python3 -m pytest`, `sleep`).
  - If the command is not in the allowlist (e.g., `touch $HOME/sphinx_test.txt`, `rm -rf /tmp`, `systemctl is-active sshd`), it returns `sutra::AgentError::ToolNotAllowed` naming both the full command, the command word, and the sandbox root:
    ```text
    Cannot run '<cmd>': <cmd_word> is not allowed in the Sphinx allowlist. The allowed root is <sandbox_dir>.
    ```

## 3. What `plan_node` and System Prompts Tell the Model About Available Tools

- **System Prompt (`kSphinxIdentityPrompt` in `src/session.cpp:71-76`)**:
  Lists the 9 currently registered tools:
  `read_file`, `write_file`, `list_dir`, `make_directory`, `run_command (allowlist: gcc, g++, clang, clang++, make, cmake, ninja, cargo, rustc, pwd)`, `compile_check`, `ssh_exec`, `ssh_test`, `echo`, and states that all filesystem tools operate within `~/sphinx-workspace`.
- **Planner (`plan_node` in `src/session.cpp:2263-2471`)**:
  Maps TASK requests to `<tool_call>` JSON invocations parsed by `sutra::ToolCallParser` and executed in the Yantra DAG (`execute_0` / `execute_1`). Currently, `plan_node` has no route for read-only host diagnostics (`run_readonly`) or curated outside-sandbox file inspection (`read_file_system`).

## 4. Existing Tests Asserting Outside-Sandbox Access Refusal

- **`tests/test_unit.cpp`**:
  - `test_tools_and_sandbox` (`tests/test_unit.cpp:118-290`): Asserts that `resolve_sandboxed_path("/etc/passwd")`, `resolve_sandboxed_path("../escape.txt")`, and `tool_read_file` / `tool_write_file` on outside paths fail with `is_tool_sandbox_violation()`.
  - `test_make_directory` (`tests/test_unit.cpp:292-384`): Asserts that absolute paths (`/tmp/escape_dir`) and traversal paths (`../escape_dir`) fail with `is_tool_sandbox_violation()`.
  - `test_trust_fixes_unit` (`tests/test_unit.cpp:814-841`): Asserts that `write_file` on `$HOME/sphinx_test.txt` fails with `ToolSandboxViolation` and names both `$HOME/sphinx_test.txt` and the sandbox root.
- **`tests/test_integration.cpp`**:
  - `TEST-A` (`tests/test_integration.cpp:218-246`): End-to-end turn `"Write a file named sphinx_test.txt at $HOME containing hello"` must refuse honestly and leave neither `$HOME/sphinx_test.txt` nor `<sandbox>/sphinx_test.txt` on disk.
  - `TEST-B` (`tests/test_integration.cpp:248-273`): End-to-end turn `"Use the touch command to create a file name sphinx_test.txt at $HOME"` must refuse with an allowlist error and create no file.

## 5. Architectural Conclusion for Phases 1–3

1. `read_file`, `write_file`, `list_dir`, `make_directory`, and `run_command` must keep their existing sandbox and allowlist invariants intact so `TEST-A`, `TEST-B`, `TEST-C`, `TEST-D`, and all unit tests remain 100% enforced.
2. `read_file_system(path)` must be a separate read-only tool with its own canonical prefix allowlist (`/etc`, `/var/log`, `/proc`, `/sys`, `~/.config`, `~/.bashrc`, `~/.zshrc`, `~/.profile`, `~/.ssh/config`) and secret denylist (`/etc/shadow`, `/etc/gshadow`, `/etc/ssh/ssh_host_*_key*`, `/etc/ssl/private/*`, `*id_rsa*`, `*id_ed25519*`, `*id_ecdsa*`, `*id_dsa*`, `*.gnupg*`, `*.pki*`).
3. `run_readonly(command, args)` must be a separate read-only diagnostic tool spawning `execvp` directly (never `/bin/sh` or `/bin/bash`) with strict command + subcommand allowlists and argument defense-in-depth validation.
