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

PACKAGE = bellows
VERSION = 0.1.0

CXX ?= g++
AR ?= ar
RANLIB ?= ranlib
PYTHON ?= python3

HYPER_MATH_DIR ?= $(CURDIR)/../Hyper Math

PREFIX ?= /usr/local
EXEC_PREFIX ?= $(PREFIX)
BINDIR ?= $(EXEC_PREFIX)/bin
LIBDIR ?= $(EXEC_PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig

CXXFLAGS ?= -O2 -g
WARNFLAGS = -Wall -Wextra -Wpedantic -Wno-unused-parameter
CPPFLAGS = -Iinclude -I"$(HYPER_MATH_DIR)/include"
LDFLAGS_HYPER_MATH = -L"$(HYPER_MATH_DIR)/lib" -Wl,-rpath,"$(HYPER_MATH_DIR)/lib" -lhypermath -pthread
ALL_CXXFLAGS = -std=c++20 $(WARNFLAGS) $(CXXFLAGS) $(CPPFLAGS)

LIB_STATIC = libbellows.a
LIB_SHARED = libbellows.so
PKGCONFIG_FILE = bellows.pc

CLI_TOOLS = bellows bellows-train bellows-infer
TEST_DRIVER = bellows-check
BENCH_BIN = bench-bellows
EXAMPLE_BIN = examples/workflow_demo

LIB_SOURCES = src/libbellows.cpp
LIB_OBJECTS = build/libbellows.o
LIB_PIC_OBJECTS = build/libbellows.pic.o

TEST_SOURCES = \
	tests/check_main.cpp \
	tests/test_core.cpp \
	tests/test_preprocess.cpp \
	tests/test_estimators.cpp \
	tests/test_boosting.cpp \
	tests/test_composition.cpp \
	tests/test_io_and_cli.cpp

TEST_OBJECTS = $(patsubst tests/%.cpp,build/tests/%.o,$(TEST_SOURCES))

HEADERS = $(shell find include -name '*.hpp' 2>/dev/null)

.PHONY: all check bench bench-ci install clean dist

all: $(LIB_STATIC) $(LIB_SHARED) $(PKGCONFIG_FILE) $(CLI_TOOLS) $(TEST_DRIVER) $(BENCH_BIN) $(EXAMPLE_BIN)

build:
	@mkdir -p build build/tests

build/libbellows.o: src/libbellows.cpp $(HEADERS) | build
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

build/libbellows.pic.o: src/libbellows.cpp $(HEADERS) | build
	$(CXX) $(ALL_CXXFLAGS) -fPIC -c $< -o $@

$(LIB_STATIC): $(LIB_OBJECTS)
	$(AR) rcs $@ $^
	$(RANLIB) $@

$(LIB_SHARED): $(LIB_PIC_OBJECTS)
	$(CXX) -shared $(ALL_CXXFLAGS) $^ $(LDFLAGS_HYPER_MATH) -o $@

$(PKGCONFIG_FILE): bellows.pc.in
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@VERSION@|$(VERSION)|g' $< > $@

bellows: src/bellows_main.cpp src/cli_common.hpp $(LIB_STATIC) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) src/bellows_main.cpp $(LIB_STATIC) $(LDFLAGS_HYPER_MATH) -o $@

bellows-train: src/bellows_train_main.cpp src/cli_common.hpp $(LIB_STATIC) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) src/bellows_train_main.cpp $(LIB_STATIC) $(LDFLAGS_HYPER_MATH) -o $@

bellows-infer: src/bellows_infer_main.cpp src/cli_common.hpp $(LIB_STATIC) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) src/bellows_infer_main.cpp $(LIB_STATIC) $(LDFLAGS_HYPER_MATH) -o $@

build/tests/%.o: tests/%.cpp $(HEADERS) | build
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

$(TEST_DRIVER): $(TEST_OBJECTS) $(LIB_STATIC)
	$(CXX) $(ALL_CXXFLAGS) $(TEST_OBJECTS) $(LIB_STATIC) $(LDFLAGS_HYPER_MATH) -o $@

$(EXAMPLE_BIN): examples/workflow_demo.cpp $(LIB_STATIC) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) examples/workflow_demo.cpp $(LIB_STATIC) $(LDFLAGS_HYPER_MATH) -o $@

$(BENCH_BIN): bench/bench_bellows.cpp $(LIB_STATIC) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) bench/bench_bellows.cpp $(LIB_STATIC) $(LDFLAGS_HYPER_MATH) -o $@

bench: $(BENCH_BIN)
	./$(BENCH_BIN) --json-out=results.json

bench-ci: $(BENCH_BIN)
	./$(BENCH_BIN) --json-out=results.json --check-baseline=benchmarks/baseline.json

check: all
	./$(TEST_DRIVER)
	./bellows --version > /dev/null
	./bellows --help > /dev/null
	./bellows-train --version > /dev/null
	./bellows-train --help > /dev/null
	./bellows-infer --version > /dev/null
	./bellows-infer --help > /dev/null
	./$(EXAMPLE_BIN) > /dev/null
	$(PYTHON) scripts/check_style.py

install: all
	install -d "$(DESTDIR)$(BINDIR)"
	install -m 0755 bellows bellows-train bellows-infer "$(DESTDIR)$(BINDIR)/"
	install -d "$(DESTDIR)$(LIBDIR)"
	install -m 0644 $(LIB_STATIC) "$(DESTDIR)$(LIBDIR)/"
	install -m 0755 $(LIB_SHARED) "$(DESTDIR)$(LIBDIR)/"
	install -d "$(DESTDIR)$(PKGCONFIGDIR)"
	install -m 0644 $(PKGCONFIG_FILE) "$(DESTDIR)$(PKGCONFIGDIR)/"
	install -d "$(DESTDIR)$(INCLUDEDIR)"
	cp -R include/bellows "$(DESTDIR)$(INCLUDEDIR)/"

clean:
	rm -rf build $(LIB_STATIC) $(LIB_SHARED) $(PKGCONFIG_FILE) \
		$(CLI_TOOLS) $(TEST_DRIVER) $(EXAMPLE_BIN) \
		$(PACKAGE)-$(VERSION) $(PACKAGE)-$(VERSION).tar.gz

dist:
	rm -rf $(PACKAGE)-$(VERSION) $(PACKAGE)-$(VERSION).tar.gz
	mkdir -p $(PACKAGE)-$(VERSION)
	cp -R AGENTS.md COPYING README NEWS ChangeLog AUTHORS INSTALL \
		Makefile CMakeLists.txt bellows.pc.in \
		include src tests examples doc scripts $(PACKAGE)-$(VERSION)/
	rm -f $(PACKAGE)-$(VERSION)/examples/workflow_demo
	tar -czf $(PACKAGE)-$(VERSION).tar.gz $(PACKAGE)-$(VERSION)
	rm -rf $(PACKAGE)-$(VERSION)
