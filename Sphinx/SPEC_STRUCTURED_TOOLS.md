# Sphinx Structured Diagnostic Tools Specification (Phase 4 — Spec Only)

## 1. Overview & Status

**Status**: SPEC ONLY (Deferred until Phases 1–3 have been used in live sessions for at least one week).

Phases 1–3 provide Sphinx with three read-only observation layers:
1. **Session Profile Injection** (`SphinxSession::create` -> `gather_session_profile`): Zero-latency turn-1 awareness of OS, kernel, CPU, RAM, GPU, disk mounts, shells, compilers, build tools, Rust toolchain, configured SSH hosts, and active system services.
2. **`run_readonly` Tool**: `execvp`-based execution of a 31-command diagnostic allowlist with per-command subcommand restrictions and defense-in-depth argument validation (no shell invocation).
3. **`read_file_system` Tool**: Direct C++ file reads across allowlisted system prefixes (`/etc`, `/var/log`, `/proc`, `/sys`, `~/.config`, and specific dotfiles) protected by `weakly_canonical` path resolution, secret denylists (`O6`), binary detection, and a 256 KB truncation cap.

Phase 4 defines six higher-level structured diagnostic tools that return pre-parsed, token-compact JSON objects instead of raw command output, saving LLM context tokens and eliminating formatting variance across Linux distributions.

---

## 2. Candidate Structured Tools

### 2.1. `get_system_info()`

- **Signature**:
  ```cpp
  [[nodiscard]] auto
  get_system_info () noexcept -> Result<SystemInfo>;
  ```
- **JSON Schema**:
  ```json
  {
    "os": "Ubuntu 24.04.4 LTS",
    "arch": "x86_64",
    "kernel": "6.17.0-20-generic",
    "uptime_seconds": 84320,
    "cpu": {
      "model": "12th Gen Intel Core i7-12700H",
      "cores": 14,
      "threads": 20,
      "avx512": false,
      "load_avg_1m": 0.42,
      "load_avg_5m": 0.58,
      "load_avg_15m": 0.61
    },
    "ram": {
      "total_gib": 47.0,
      "used_gib": 19.2,
      "available_gib": 27.8,
      "swap_total_gib": 2.0,
      "swap_used_gib": 0.0
    },
    "gpu": {
      "model": "NVIDIA GeForce RTX 3070 Laptop GPU",
      "vram_total_mib": 8192,
      "vram_used_mib": 7055,
      "driver": "590.48.01"
    },
    "disks": [
      {
        "mount": "/",
        "total_gib": 915.0,
        "used_gib": 412.0,
        "free_gib": 456.0,
        "use_pct": 48
      }
    ]
  }
  ```
- **Implementation Strategy**: Read `/proc/uptime`, `/proc/loadavg`, `/proc/meminfo`, `statvfs()`, and NVML / `nvidia-smi` directly in C++ without spawning multiple subprocesses.

### 2.2. `list_processes(sort_by, limit)`

- **Signature**:
  ```cpp
  [[nodiscard]] auto
  list_processes (std::string_view sort_by = "cpu",
                  uint32_t limit = 20U) noexcept -> Result<std::string>;
  ```
- **Arguments**:
  - `sort_by`: `"cpu"` | `"mem"` | `"pid"` (default `"cpu"`)
  - `limit`: `1..100` (default `20`)
- **JSON Schema**:
  ```json
  {
    "sort_by": "cpu",
    "processes": [
      {
        "pid": 14820,
        "user": "developer",
        "cpu_pct": 12.5,
        "rss_mib": 7120.4,
        "command": "sphinx"
      }
    ]
  }
  ```
- **Implementation Strategy**: Enumerate `/proc/[pid]/stat` and `/proc/[pid]/status` directly or invoke `ps -eo pid,user,%cpu,rss,comm --sort=-%cpu` via `run_readonly`.

### 2.3. `service_status(name)`

- **Signature**:
  ```cpp
  [[nodiscard]] auto
  service_status (std::string_view name) noexcept -> Result<std::string>;
  ```
- **Arguments**:
  - `name`: systemd unit name (validated against `[A-Za-z0-9_.@-]+`)
- **JSON Schema**:
  ```json
  {
    "unit": "sshd.service",
    "active_state": "active",
    "sub_state": "running",
    "enabled_state": "enabled",
    "main_pid": 1142,
    "since": "2026-11-02 08:14:02 EST",
    "recent_logs": [
      "Nov 02 08:14:02 Silver sshd[1142]: Server listening on 0.0.0.0 port 22."
    ]
  }
  ```
- **Implementation Strategy**: Query `systemctl show <name> --property=Id,ActiveState,SubState,UnitFileState,MainPID,ActiveEnterTimestamp` and `journalctl -u <name> -n 5 --no-pager` via `execvp`.

### 2.4. `recent_logs(unit, priority, lines)`

- **Signature**:
  ```cpp
  [[nodiscard]] auto
  recent_logs (std::string_view unit = "",
               std::string_view priority = "info",
               uint32_t lines = 50U) noexcept -> Result<std::string>;
  ```
- **Arguments**:
  - `unit`: optional systemd unit name (empty string for system-wide journal)
  - `priority`: `"emerg"` | `"alert"` | `"crit"` | `"err"` | `"warning"` | `"notice"` | `"info"` | `"debug"`
  - `lines`: `1..200` (default `50`)
- **JSON Schema**:
  ```json
  {
    "unit": "kernel",
    "priority": "err",
    "entries": [
      {
        "timestamp": "2026-11-02T13:10:04-05:00",
        "unit": "kernel",
        "priority": 3,
        "message": "..."
      }
    ]
  }
  ```

### 2.5. `network_info()`

- **Signature**:
  ```cpp
  [[nodiscard]] auto
  network_info () noexcept -> Result<std::string>;
  ```
- **JSON Schema**:
  ```json
  {
    "interfaces": [
      {
        "name": "wlan0",
        "state": "UP",
        "ipv4": ["192.168.1.42/24"],
        "ipv6": ["fe80::1/64"]
      }
    ],
    "default_route": {
      "gateway": "192.168.1.1",
      "dev": "wlan0"
    },
    "dns_servers": ["1.1.1.1", "8.8.8.8"],
    "listening_ports": [
      {
        "proto": "tcp",
        "local_address": "0.0.0.0:22",
        "process": "sshd"
      }
    ]
  }
  ```

### 2.6. `gpu_info()`

- **Signature**:
  ```cpp
  [[nodiscard]] auto
  gpu_info () noexcept -> Result<std::string>;
  ```
- **JSON Schema**:
  ```json
  {
    "available": true,
    "name": "NVIDIA GeForce RTX 3070 Laptop GPU",
    "driver_version": "590.48.01",
    "vram_total_mib": 8192,
    "vram_used_mib": 7055,
    "vram_free_mib": 1137,
    "temperature_c": 54,
    "utilization_gpu_pct": 4,
    "power_draw_w": 28.4,
    "compute_processes": [
      {
        "pid": 14820,
        "name": "sphinx",
        "used_vram_mib": 6890
      }
    ]
  }
  ```

---

## 3. Promotion Criteria from Phase 2/3 to Phase 4

Phase 4 tools will be selectively implemented after reviewing one week of `~/.local/share/sphinx/sessions/*.jsonl` telemetry:
1. Any `run_readonly` command accounting for `> 20%` of diagnostic invocations is promoted to its corresponding structured tool.
2. Any `run_readonly` command whose output frequently hits the `2048+1024` byte truncation cap is promoted to return pre-filtered JSON.
3. All Phase 4 tools must remain strictly read-only and obey the same zero-shell (`execvp` or direct `/proc` / syscall) security boundary as Phases 1–3.
