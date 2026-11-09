# Copyright (C) 2026 Kival
# This file is part of the Arkantos AI Stack Suite.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

HOST_ARCH := $(shell uname -m)

ifeq ($(HOST_ARCH),armv7l)
    CROSS_PREFIX :=
    QEMU_RUN     :=
else
    CROSS_PREFIX := arm-linux-gnueabihf-
    QEMU_CPU     := $(shell qemu-arm -cpu help 2>&1 | grep -qw cortex-a5 && echo cortex-a5 || echo cortex-a7)
    QEMU_RUN     := qemu-arm -L /usr/arm-linux-gnueabihf -cpu $(QEMU_CPU)
endif

CC      := $(CROSS_PREFIX)gcc
OBJDUMP := $(CROSS_PREFIX)objdump
READELF := $(CROSS_PREFIX)readelf
STRIP   := $(CROSS_PREFIX)strip

# Auto-detect local extracted cc1plus if system g++-arm-linux-gnueabihf is not installed
CC1PLUS_DIR := $HOME/.cache/armhf-gpp/usr/libexec/gcc-cross/arm-linux-gnueabihf/14
ifneq ($(wildcard $(CC1PLUS_DIR)/cc1plus),)
    B_FLAGS := -B$(CC1PLUS_DIR)/
else
    B_FLAGS :=
endif

ARCH_FLAGS := -march=armv7-a -mtune=cortex-a5 -mfpu=neon-vfpv4 -mfloat-abi=hard
CXXFLAGS   := $(B_FLAGS) -std=c++23 -O3 $(ARCH_FLAGS) -fno-exceptions -fno-rtti \
              -fno-threadsafe-statics -fno-unwind-tables -fno-asynchronous-unwind-tables \
              -ffunction-sections -fdata-sections -Wall -Wextra
ASFLAGS    := $(ARCH_FLAGS)
LDFLAGS    := -Wl,--gc-sections -Wl,-z,max-page-size=4096 -s -lm -lc

ASM_SRCS := \
    src/asm/singularity_vec_dot_q4_0_q8_0_a5.S \
    src/asm/singularity_vec_dot_q8_0_q8_0_a5.S \
    src/asm/singularity_quantize_row_q8_0_a5.S \
    src/asm/singularity_rmsnorm_f32_a5.S \
    src/asm/singularity_softmax_f32_a5.S \
    src/asm/singularity_rope_f32_a5.S \
    src/asm/singularity_utf8_stream_a5.S \
    src/asm/singularity_linux4_syscalls_a5.S

CORE_SRCS := \
    src/core/linux4_arena_and_mmap.cpp \
    src/core/cortex_a5_engine.cpp

MAIN_SRC  := src/core/main.cpp
TEST_SRC  := tests/test_and_bench_a5.cpp

ASM_OBJS  := $(ASM_SRCS:.S=.o)
CORE_OBJS := $(CORE_SRCS:.cpp=.o)
MAIN_OBJ  := $(MAIN_SRC:.cpp=.o)
TEST_OBJ  := $(TEST_SRC:.cpp=.o)

TARGET      := singularity
TEST_TARGET := test_and_bench_a5

.PHONY: all test audit run clean

all: $(TARGET) $(TEST_TARGET)

%.o: %.S
	$(CC) $(ASFLAGS) -c $< -o $@

%.o: %.cpp src/core/cortex_a5_engine.hpp src/core/linux4_arena_and_mmap.hpp src/asm/singularity_asm_kernels.h
	$(CC) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(ASM_OBJS) $(CORE_OBJS) $(MAIN_OBJ)
	$(CC) $(ARCH_FLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_TARGET): $(ASM_OBJS) $(CORE_OBJS) $(TEST_OBJ)
	$(CC) $(ARCH_FLAGS) $^ -o $@ $(LDFLAGS)

test: $(TARGET) $(TEST_TARGET)
	$(QEMU_RUN) ./$(TEST_TARGET)
	$(QEMU_RUN) ./$(TARGET)

audit: $(TARGET)
	@echo "=== [Audit 1] Dynamic Library Dependencies (MUST NOT link libstdc++.so) ==="
	$(READELF) -d ./$(TARGET) | grep NEEDED
	@if $(READELF) -d ./$(TARGET) | grep -q "libstdc++"; then \
		echo "[FAIL] libstdc++.so is linked!"; exit 1; \
	else \
		echo "[PASS] Zero libstdc++.so dependency verified."; \
	fi
	@echo "=== [Audit 2] Stripped Binary Size (MUST be <= 131072 bytes / 128 KB) ==="
	@SIZE=$$(stat -c%s ./$(TARGET)); \
	echo "Binary size: $$SIZE bytes (Limit: 131072 bytes)"; \
	if [ $$SIZE -le 131072 ]; then \
		echo "[PASS] Binary fits well within 128 KB budget."; \
	else \
		echo "[FAIL] Binary exceeds 128 KB!"; exit 1; \
	fi
	@echo "=== [Audit 3] Prohibited Instructions & Helper Calls Audit ==="
	@if $(OBJDUMP) -d --no-show-raw-insn ./$(TARGET) | grep -E "(__aeabi_uldivmod|__gnu_h2f_ieee|vdiv\.f32|dmb)"; then \
		echo "[FAIL] Prohibited instruction or helper call detected!"; exit 1; \
	else \
		echo "[PASS] Zero __aeabi_uldivmod, __gnu_h2f_ieee, vdiv.f32, or dmb in ./$(TARGET)!"; \
	fi
	@echo "=== [Audit 4] Cortex-A5 In-Order Pipeline Load-to-Use Separation (vld1 -> vmull/vmlal >= 2 insns) ==="
	@if $(OBJDUMP) -d --no-show-raw-insn src/asm/singularity_vec_dot_q4_0_q8_0_a5.o src/asm/singularity_vec_dot_q8_0_q8_0_a5.o | \
		awk '/vld1/{dist=0; seen=1; next} seen{dist++; if ($$0 ~ /vmull|vmlal/ && dist <= 2) {print "Pipeline stall hazard at: " $$0; exit 1}}'; then \
		echo "[PASS] All vld1 loads separated from dependent vmull/vmlal by >= 2 independent instructions."; \
	else \
		echo "[FAIL] Pipeline load-to-use hazard detected!"; exit 1; \
	fi

clean:
	rm -f $(ASM_OBJS) $(CORE_OBJS) $(MAIN_OBJ) $(TEST_OBJ) $(TARGET) $(TEST_TARGET)
