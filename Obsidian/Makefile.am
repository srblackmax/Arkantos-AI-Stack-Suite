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

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = \
  -I$(top_builddir)/include \
  -I$(top_srcdir)/include \
  $(BASALT_CPPFLAGS)

AM_CXXFLAGS = \
  -std=c++23 -O3 -march=native -funroll-loops -fopenmp \
  -Wall -Wextra -Werror -pedantic \
  $(SANITIZER_FLAGS)

AM_LDFLAGS = \
  -fopenmp \
  $(BASALT_LDFLAGS) \
  $(SANITIZER_FLAGS)

lib_LTLIBRARIES = libobsidian.la

libobsidian_la_SOURCES = \
  src/core/core.cpp \
  src/core/unicode.cpp \
  src/tensor/tensor_bridge.cpp \
  src/tensor/safetensors.cpp \
  src/tokenizer/tokenizer.cpp \
  src/embeddings/embeddings.cpp \
  src/classical/classical.cpp \
  src/classical/topic.cpp \
  src/transformer/transformer.cpp \
  src/models/models.cpp \
  src/pipeline/pipeline.cpp

libobsidian_la_LIBADD = $(BASALT_LIBS)
libobsidian_la_LDFLAGS = $(AM_LDFLAGS) -version-info 0:0:0

bin_PROGRAMS = obsidian-nlp

obsidian_nlp_SOURCES = src/cli/obsidian_nlp.cpp
obsidian_nlp_LDADD = libobsidian.la $(BASALT_LIBS)

noinst_PROGRAMS = \
  examples/nlp_showcase \
  bench/bench_obsidian

examples_nlp_showcase_SOURCES = examples/nlp_showcase.cpp
examples_nlp_showcase_LDADD = libobsidian.la $(BASALT_LIBS)

bench_bench_obsidian_SOURCES = bench/bench_obsidian.cpp
bench_bench_obsidian_LDADD = libobsidian.la $(BASALT_LIBS)

.PHONY: bench bench-ci

bench: bench/bench_obsidian
	./bench/bench_obsidian --json-out=results.json

bench-ci: bench/bench_obsidian
	./bench/bench_obsidian --json-out=results.json --check-baseline=$(srcdir)/benchmarks/baseline.json

obsidianincludedir = $(includedir)/obsidian
obsidianinclude_HEADERS = \
  include/obsidian/config.hpp \
  include/obsidian/error.hpp \
  include/obsidian/memory.hpp \
  include/obsidian/simd.hpp \
  include/obsidian/unicode.hpp \
  include/obsidian/tensor_bridge.hpp \
  include/obsidian/safetensors.hpp \
  include/obsidian/tokenizer.hpp \
  include/obsidian/embeddings.hpp \
  include/obsidian/stemmer.hpp \
  include/obsidian/tagger.hpp \
  include/obsidian/classical.hpp \
  include/obsidian/topic.hpp \
  include/obsidian/transformer.hpp \
  include/obsidian/models.hpp \
  include/obsidian/pipeline.hpp \
  include/obsidian/obsidian.hpp

nodist_obsidianinclude_HEADERS = \
  include/obsidian/version.h

noinst_HEADERS = \
  tests/test_harness.hpp

check_PROGRAMS = \
  tests/check_gnu_style \
  tests/test_core_tensor \
  tests/test_tokenizer \
  tests/test_embeddings \
  tests/test_classical \
  tests/test_topic \
  tests/test_transformer \
  tests/test_models_pipeline \
  tests/test_production

tests_check_gnu_style_SOURCES = tests/check_gnu_style.cpp

tests_test_core_tensor_SOURCES = tests/test_core_tensor.cpp
tests_test_core_tensor_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_tokenizer_SOURCES = tests/test_tokenizer.cpp
tests_test_tokenizer_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_embeddings_SOURCES = tests/test_embeddings.cpp
tests_test_embeddings_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_classical_SOURCES = tests/test_classical.cpp
tests_test_classical_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_topic_SOURCES = tests/test_topic.cpp
tests_test_topic_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_transformer_SOURCES = tests/test_transformer.cpp
tests_test_transformer_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_models_pipeline_SOURCES = tests/test_models_pipeline.cpp
tests_test_models_pipeline_LDADD = libobsidian.la $(BASALT_LIBS)

tests_test_production_SOURCES = tests/test_production.cpp
tests_test_production_LDADD = libobsidian.la $(BASALT_LIBS)

TESTS = \
  tests/check_gnu_style \
  tests/test_core_tensor \
  tests/test_tokenizer \
  tests/test_embeddings \
  tests/test_classical \
  tests/test_topic \
  tests/test_transformer \
  tests/test_models_pipeline \
  tests/test_production

EXTRA_DIST = \
  AGENTS.md \
  SKILL.md \
  ARCHITECTURE.md \
  TASKS.md \
  README.md \
  CMakeLists.txt \
  include/obsidian/version.h.in
