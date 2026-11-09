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
  -I$(top_srcdir)/src \
  $(BELLOWS_CPPFLAGS)

AM_CXXFLAGS = \
  -O3 -march=native -funroll-loops -fopenmp \
  -Wall -Wextra -Wpedantic -Wconversion \
  $(SANITIZER_FLAGS)

AM_CFLAGS = \
  -O3 -march=native -funroll-loops -fopenmp \
  -Wall -Wextra -Wpedantic \
  $(SANITIZER_FLAGS)

AM_LDFLAGS = \
  -fopenmp \
  $(BELLOWS_LDFLAGS) \
  $(SANITIZER_FLAGS)

lib_LTLIBRARIES = libbasalt.la libbasalt-c.la

libbasalt_la_SOURCES = \
  src/core/version.cpp \
  src/core/error.cpp \
  src/core/dtype.cpp \
  src/core/device.cpp \
  src/core/random.cpp \
  src/bellows-compat/bellows_bridge.cpp \
  src/tensor/tensor.cpp \
  src/tensor/ops.cpp \
  src/autograd/autograd.cpp \
  src/nn/functional.cpp \
  src/nn/nn.cpp \
  src/loss/loss.cpp \
  src/metrics/metrics.cpp \
  src/optim/optim.cpp \
  src/optim/scheduler.cpp \
  src/data/data.cpp \
  src/amp/amp.cpp \
  src/io/serialization.cpp \
  src/distributed/distributed.cpp \
  src/trainer/trainer.cpp

libbasalt_la_LIBADD = $(BELLOWS_LIBS)
libbasalt_la_LDFLAGS = $(AM_LDFLAGS) -version-info 0:0:0

libbasalt_c_la_SOURCES = \
  src/c_api/basalt_c.cpp \
  $(libbasalt_la_SOURCES)

libbasalt_c_la_LIBADD = $(BELLOWS_LIBS)
libbasalt_c_la_LDFLAGS = $(AM_LDFLAGS) -version-info 0:0:0

bin_PROGRAMS = \
  basalt-train \
  basalt-inspect \
  basalt-convert

basalt_train_SOURCES = src/cli/basalt_train.cpp
basalt_train_LDADD = libbasalt.la $(BELLOWS_LIBS)

basalt_inspect_SOURCES = src/cli/basalt_inspect.cpp
basalt_inspect_LDADD = libbasalt.la $(BELLOWS_LIBS)

basalt_convert_SOURCES = src/cli/basalt_convert.cpp
basalt_convert_LDADD = libbasalt.la $(BELLOWS_LIBS)

noinst_PROGRAMS = \
  examples/mlp_xor \
  bench/bench_basalt

examples_mlp_xor_SOURCES = examples/mlp_xor.cpp
examples_mlp_xor_LDADD = libbasalt.la $(BELLOWS_LIBS)

bench_bench_basalt_SOURCES = bench/bench_basalt.cpp
bench_bench_basalt_LDADD = libbasalt.la $(BELLOWS_LIBS)

.PHONY: bench bench-ci

bench: bench/bench_basalt
	./bench/bench_basalt --json-out=results.json

bench-ci: bench/bench_basalt
	./bench/bench_basalt --json-out=results.json --check-baseline=$(srcdir)/benchmarks/baseline.json

basaltincludedir = $(includedir)/basalt
basaltinclude_HEADERS = \
  include/basalt/config.hpp \
  include/basalt/error.hpp \
  include/basalt/dtype.hpp \
  include/basalt/device.hpp \
  include/basalt/random.hpp \
  include/basalt/tensor.hpp \
  include/basalt/ops.hpp \
  include/basalt/autograd.hpp \
  include/basalt/functional.hpp \
  include/basalt/nn.hpp \
  include/basalt/loss.hpp \
  include/basalt/metrics.hpp \
  include/basalt/optim.hpp \
  include/basalt/scheduler.hpp \
  include/basalt/data.hpp \
  include/basalt/amp.hpp \
  include/basalt/serialization.hpp \
  include/basalt/distributed.hpp \
  include/basalt/logging.hpp \
  include/basalt/callbacks.hpp \
  include/basalt/trainer.hpp \
  include/basalt/basalt.hpp \
  include/basalt/basalt_c.h

nodist_basaltinclude_HEADERS = \
  include/basalt/version.h

noinst_HEADERS = \
  src/bellows-compat/bellows_bridge.hpp \
  src/tensor/tensor_internal.hpp

check_PROGRAMS = \
  tests/test_phase1_core \
  tests/test_phase2_tensor \
  tests/test_phase3_autograd \
  tests/test_phase4_nn \
  tests/test_phase5_optim \
  tests/test_phase6_data_amp \
  tests/test_phase7_serialization \
  tests/test_phase8_trainer \
  tests/test_phase9_c_api

tests_test_phase1_core_SOURCES = tests/test_phase1_core.cpp
tests_test_phase1_core_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase2_tensor_SOURCES = tests/test_phase2_tensor.cpp
tests_test_phase2_tensor_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase3_autograd_SOURCES = tests/test_phase3_autograd.cpp
tests_test_phase3_autograd_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase4_nn_SOURCES = tests/test_phase4_nn.cpp
tests_test_phase4_nn_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase5_optim_SOURCES = tests/test_phase5_optim.cpp
tests_test_phase5_optim_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase6_data_amp_SOURCES = tests/test_phase6_data_amp.cpp
tests_test_phase6_data_amp_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase7_serialization_SOURCES = tests/test_phase7_serialization.cpp
tests_test_phase7_serialization_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase8_trainer_SOURCES = tests/test_phase8_trainer.cpp
tests_test_phase8_trainer_LDADD = libbasalt.la $(BELLOWS_LIBS)

tests_test_phase9_c_api_SOURCES = tests/test_phase9_c_api.c
tests_test_phase9_c_api_LDADD = libbasalt-c.la $(BELLOWS_LIBS)

TESTS = \
  tests/test_phase1_core \
  tests/test_phase2_tensor \
  tests/test_phase3_autograd \
  tests/test_phase4_nn \
  tests/test_phase5_optim \
  tests/test_phase6_data_amp \
  tests/test_phase7_serialization \
  tests/test_phase8_trainer \
  tests/test_phase9_c_api \
  tests/test_cli.sh \
  tests/check_gnu_style.py

info_TEXINFOS = doc/basalt-dl.texi

EXTRA_DIST = \
  include/basalt/version.h.in \
  python/basalt/__init__.py \
  python/basalt/_basalt.cpp \
  python/basalt/_version.py.in \
  setup.cfg.in \
  pyproject.toml \
  tests/test_cli.sh \
  tests/check_gnu_style.py
