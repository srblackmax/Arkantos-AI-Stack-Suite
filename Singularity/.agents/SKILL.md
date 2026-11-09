# SKILL: Singularity — Bare-Metal Cortex-A5 ARMv7-A (armhf) Assembly & Zero-Bloat Architecture

## 1. Target Hardware & Processor Limitations
- **Project & Binary Name:** `Singularity` (`singularity` executable, `namespace singularity`)
- **Device:** LG Optimus L5 (LG-E612 "Billy")
- **SoC:** Qualcomm MSM7225A — Single-Core ARM Cortex-A5 @ 800 MHz
- **ISA:** 32-bit ARMv7-A (`armhf` / `arm-linux-gnueabihf` ABI)
- **Supported Hardware Coprocessors:**
  - Advanced SIMD (NEON) Media Processing Engine (64-bit physical datapath: 128-bit `q0`–`q15` take 2 cycles; 64-bit `d0`–`d31` take 1 cycle).
  - VFPv4-D32 Floating-Point Unit with hardware half-precision conversion (`vcvt.f32.f16`).
- **UNSUPPORTED / PENALIZED Hardware Features (MUST Replace in Assembly):**
  - **NO Out-of-Order Execution:** 8-stage strictly IN-ORDER pipeline. Any adjacent load-to-use (`vld1` followed immediately by `vmull`) stalls the CPU.
  - **NO Hardware Integer Division in ARM Mode / Slow Scalar Float Division:** Avoid `udiv`/`sdiv`, 64-bit modulo (`__aeabi_uldivmod`), and `vdiv.f32`. Replace with bitshifts (`lsr`, `and`) and NEON reciprocal estimates (`vrecpe.f32`/`vrecps.f32`, `vrsqrte.f32`/`vrsqrts.f32`).
  - **NO Multi-Core SMP Overhead:** Single physical core. NEVER emit `dmb ish` memory barriers, `ldrex`/`strex` atomic spinlocks, or `pthread` mutexes in the inference path.
  - **NO Software `fp16` Emulation:** Never use C/C++ `_Float16` or `std::float16_t` in C++ code (which triggers slow `__gnu_h2f_ieee` calls). Perform all `fp16` scale conversions inside `.S` Assembly via `vcvt.f32.f16`.
  - **NO Linux 5.x+ Syscalls or `libstdc++` Bloat:** Target Linux Kernel 4.x compatibility (`mmap2`, `madvise` with `MADV_WILLNEED`/`MADV_DONTNEED`, POSIX `read`/`write`). Link WITHOUT `libstdc++` (`-fno-exceptions -fno-rtti -nostdlib++` or compile with `gcc`) so the stripped binary stays under 128 KB and fits in the 32 KB L1 I-Cache / 256 KB L2 Cache.

## 2. Hard Memory Budget (512 MB Physical RAM Ceiling)
- **Singularity Binary Size:** <= 128 KB stripped (`-s`), zero `libstdc++` dependency.
- **Activation & Scratch Arena:** Single pre-allocated 32-byte aligned arena (`<= 16 MB`). Zero `malloc`/`new` calls during token generation.
- **KV Cache Budget:** <= 35 MB (`fp16` stored raw, converted via NEON `vcvt.f32.f16`).
- **Mandatory OOM Guard Floor:** 48 MB minimum available RAM reserved at all times; stream layers with `madvise(..., MADV_DONTNEED)` if free RAM drops below 48 MB.

## 3. Mandatory GNU Assembler (`.S`) Rules (AAPCS-VFP / `eabihf`)
1. **Header Directives:**
   ```arm
   .syntax unified
   .arch armv7-a
   .fpu neon-vfpv4
   .arm
   .text
   ```
2. **ABI Register Preservation (`arm-linux-gnueabihf`):**
   - **Scratch:** `r0`–`r3`, `r12` (`ip`), `d0`–`d7` (`q0`–`q3`), `d16`–`d31` (`q8`–`q15`).
   - **Callee-saved (MUST preserve via `push`/`pop` and `vpush`/`vpop`):** `r4`–`r11`, `r14` (`lr`), `d8`–`d15` (`q4`–`q7`).
   - **Stack Alignment:** Keep `sp` (`r13`) strictly 8-byte aligned.
3. **Cortex-A5 In-Order Scheduling:**
   - Separate `vld1` / `ldr` loads from dependent arithmetic instructions by at least 2–3 independent instructions.
   - Insert `pld [rX, #128]` aligned to 32-byte cache lines inside hot loops.
   - Unroll inner dot-product loops 4x across `q8`, `q9`, `q10`, `q11` accumulators.
