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

SHELL = /bin/sh
.SUFFIXES:
.SUFFIXES: .cpp .o

PACKAGE = hyper-math
VERSION = 0.1.0
DISTNAME = $(PACKAGE)-$(VERSION)

# Standard GNU directory variables
srcdir = .
prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
libdir = $(exec_prefix)/lib
includedir = $(prefix)/include
datarootdir = $(prefix)/share
datadir = $(datarootdir)
docdir = $(datarootdir)/doc/$(PACKAGE)
mandir = $(datarootdir)/man
man1dir = $(mandir)/man1
pkgconfigdir = $(libdir)/pkgconfig

# Programs and flags
CXX ?= g++
AR ?= ar
ARFLAGS ?= rcs
INSTALL ?= install
INSTALL_PROGRAM ?= $(INSTALL) -m 755
INSTALL_DATA ?= $(INSTALL) -m 644

CPPFLAGS ?=
CXXFLAGS ?= -O2 -g -Wall -Wextra -Wpedantic -pthread
STD_FLAGS = -std=c++20 -I$(srcdir)/include
LDFLAGS ?= -pthread

BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib

STATIC_LIB = $(LIB_DIR)/libhypermath.a
SHARED_LIB = $(LIB_DIR)/libhypermath.so
CLI_BIN = $(BIN_DIR)/hyper-math
TEST_BIN = $(BIN_DIR)/test-hyper-math
BENCH_BIN = $(BIN_DIR)/bench-hyper-math

HEADERS = \
	$(wildcard $(srcdir)/include/hyper_math/*.hpp) \
	$(wildcard $(srcdir)/include/hyper_math/*.h) \
	$(wildcard $(srcdir)/include/hyper_math/*/*.hpp)

LIB_OBJS = $(BUILD_DIR)/libhypermath.o
CLI_OBJS = $(BUILD_DIR)/main.o
TEST_OBJS = \
	$(BUILD_DIR)/test_main.o \
	$(BUILD_DIR)/test_tensor.o \
	$(BUILD_DIR)/test_scientific.o \
	$(BUILD_DIR)/test_tabular.o

.PHONY: all check test bench bench-ci install uninstall clean mostlyclean distclean \
        maintainer-clean dist distcheck info html dvi pdf

all: $(STATIC_LIB) $(SHARED_LIB) $(CLI_BIN) $(TEST_BIN) $(BENCH_BIN)

$(BUILD_DIR) $(BIN_DIR) $(LIB_DIR):
	mkdir -p $@

$(BUILD_DIR)/libhypermath.o: $(srcdir)/src/libhypermath.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(STD_FLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: $(srcdir)/src/main.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(STD_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_main.o: $(srcdir)/tests/test_main.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(STD_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_tensor.o: $(srcdir)/tests/test_tensor.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(STD_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_scientific.o: $(srcdir)/tests/test_scientific.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(STD_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_tabular.o: $(srcdir)/tests/test_tabular.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(STD_FLAGS) -c $< -o $@

$(BUILD_DIR)/bench_main.o: $(srcdir)/benchmarks/bench_main.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(STD_FLAGS) -c $< -o $@

$(STATIC_LIB): $(LIB_OBJS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $(LIB_OBJS)

$(SHARED_LIB): $(LIB_OBJS) | $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared $(LIB_OBJS) $(LDFLAGS) -o $@

$(CLI_BIN): $(CLI_OBJS) $(STATIC_LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CLI_OBJS) $(STATIC_LIB) $(LDFLAGS) -o $@

$(TEST_BIN): $(TEST_OBJS) $(STATIC_LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(STATIC_LIB) $(LDFLAGS) -o $@

$(BENCH_BIN): $(BUILD_DIR)/bench_main.o $(STATIC_LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(BUILD_DIR)/bench_main.o $(STATIC_LIB) $(LDFLAGS) -o $@

check: $(TEST_BIN) $(CLI_BIN)
	./$(TEST_BIN)
	./$(CLI_BIN) --version > /dev/null
	./$(CLI_BIN) --help > /dev/null

test: check

bench: $(BENCH_BIN)
	./$(BENCH_BIN) --json-out=results.json

bench-ci: $(BENCH_BIN)
	./$(BENCH_BIN) --json-out=results.json --check-baseline=$(srcdir)/benchmarks/baseline.json

install: all
	$(INSTALL) -d "$(DESTDIR)$(bindir)"
	$(INSTALL) -d "$(DESTDIR)$(libdir)"
	$(INSTALL) -d "$(DESTDIR)$(pkgconfigdir)"
	$(INSTALL) -d "$(DESTDIR)$(man1dir)"
	$(INSTALL) -d "$(DESTDIR)$(includedir)/hyper_math/core"
	$(INSTALL) -d "$(DESTDIR)$(includedir)/hyper_math/tensor"
	$(INSTALL) -d "$(DESTDIR)$(includedir)/hyper_math/scientific"
	$(INSTALL) -d "$(DESTDIR)$(includedir)/hyper_math/tabular"
	$(INSTALL_PROGRAM) $(CLI_BIN) "$(DESTDIR)$(bindir)/hyper-math"
	$(INSTALL_DATA) $(STATIC_LIB) "$(DESTDIR)$(libdir)/libhypermath.a"
	$(INSTALL_PROGRAM) $(SHARED_LIB) "$(DESTDIR)$(libdir)/libhypermath.so"
	$(INSTALL_DATA) $(srcdir)/hyper-math.pc "$(DESTDIR)$(pkgconfigdir)/hyper-math.pc"
	$(INSTALL_DATA) $(srcdir)/doc/hyper-math.1 "$(DESTDIR)$(man1dir)/hyper-math.1"
	$(INSTALL_DATA) $(srcdir)/include/hyper_math/hyper_math.hpp "$(DESTDIR)$(includedir)/hyper_math/"
	$(INSTALL_DATA) $(srcdir)/include/hyper_math/hyper_math_c.h "$(DESTDIR)$(includedir)/hyper_math/"
	$(INSTALL_DATA) $(srcdir)/include/hyper_math/core/*.hpp "$(DESTDIR)$(includedir)/hyper_math/core/"
	$(INSTALL_DATA) $(srcdir)/include/hyper_math/tensor/*.hpp "$(DESTDIR)$(includedir)/hyper_math/tensor/"
	$(INSTALL_DATA) $(srcdir)/include/hyper_math/scientific/*.hpp "$(DESTDIR)$(includedir)/hyper_math/scientific/"
	$(INSTALL_DATA) $(srcdir)/include/hyper_math/tabular/*.hpp "$(DESTDIR)$(includedir)/hyper_math/tabular/"

uninstall:
	rm -f "$(DESTDIR)$(bindir)/hyper-math"
	rm -f "$(DESTDIR)$(libdir)/libhypermath.a"
	rm -f "$(DESTDIR)$(libdir)/libhypermath.so"
	rm -f "$(DESTDIR)$(pkgconfigdir)/hyper-math.pc"
	rm -f "$(DESTDIR)$(man1dir)/hyper-math.1"
	rm -rf "$(DESTDIR)$(includedir)/hyper_math"

mostlyclean:
	rm -rf $(BUILD_DIR)

clean: mostlyclean
	rm -rf $(BIN_DIR) $(LIB_DIR) $(DISTNAME).tar.gz

distclean: clean

maintainer-clean: distclean

dist:
	mkdir -p $(DISTNAME)
	cp -r COPYING README NEWS AUTHORS ChangeLog INSTALL THANKS \
	      configure Makefile CMakeLists.txt hyper-math.pc \
	      doc include src tests benchmarks $(DISTNAME)/
	tar -czf $(DISTNAME).tar.gz $(DISTNAME)
	rm -rf $(DISTNAME)

distcheck: dist
	tar -xzf $(DISTNAME).tar.gz
	$(MAKE) -C $(DISTNAME) check
	$(MAKE) -C $(DISTNAME) install DESTDIR="$(CURDIR)/$(DISTNAME)/_inst"
	$(MAKE) -C $(DISTNAME) uninstall DESTDIR="$(CURDIR)/$(DISTNAME)/_inst"
	rm -rf $(DISTNAME)
	@echo "===================================================="
	@echo "$(DISTNAME).tar.gz is ready for distribution."
	@echo "===================================================="

info html dvi pdf:
	@echo "Documentation is provided in README, NEWS, and header comments."
