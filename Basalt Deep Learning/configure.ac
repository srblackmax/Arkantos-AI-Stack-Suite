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

AC_PREREQ([2.69])
AC_INIT([Basalt Deep Learning], [0.1.0], [Kival@basalt-dl.org], [basalt-dl])
AC_CONFIG_SRCDIR([include/basalt/version.h.in])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.16 gnu subdir-objects -Wall])
AM_SILENT_RULES([yes])

# Derive numeric version components solely from AC_INIT's PACKAGE_VERSION.
BASALT_VERSION_MAJOR=`echo "$PACKAGE_VERSION" | sed 's/\..*//'`
BASALT_VERSION_MINOR=`echo "$PACKAGE_VERSION" | sed 's/^[0-9]*\.//; s/\..*//'`
BASALT_VERSION_PATCH=`echo "$PACKAGE_VERSION" | sed 's/^[0-9]*\.[0-9]*\.//; s/[^0-9].*//'`
BASALT_VERSION_STRING="$PACKAGE_VERSION"
AC_SUBST([BASALT_VERSION_MAJOR])
AC_SUBST([BASALT_VERSION_MINOR])
AC_SUBST([BASALT_VERSION_PATCH])
AC_SUBST([BASALT_VERSION_STRING])

AC_DEFINE_UNQUOTED([BASALT_VERSION_MAJOR], [$BASALT_VERSION_MAJOR],
  [Major version of Basalt Deep Learning.])
AC_DEFINE_UNQUOTED([BASALT_VERSION_MINOR], [$BASALT_VERSION_MINOR],
  [Minor version of Basalt Deep Learning.])
AC_DEFINE_UNQUOTED([BASALT_VERSION_PATCH], [$BASALT_VERSION_PATCH],
  [Patch version of Basalt Deep Learning.])
AC_DEFINE_UNQUOTED([BASALT_VERSION_STRING], ["$BASALT_VERSION_STRING"],
  [Version string of Basalt Deep Learning.])

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AM_PROG_AR
LT_INIT
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# Require C++20.
CXXFLAGS="$CXXFLAGS -std=c++20"
AC_LANG_PUSH([C++])
AC_MSG_CHECKING([whether $CXX supports C++20])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <concepts>
                     #include <span>]],
                   [[static_assert(std::integral<int>);]])],
  [AC_MSG_RESULT([yes])],
  [AC_MSG_RESULT([no])
   AC_MSG_ERROR([A C++20-compliant compiler is required to build Basalt Deep Learning.])])
AC_LANG_POP([C++])

# Locate Bellows and Hyper Math.
AC_ARG_WITH([bellows],
  [AS_HELP_STRING([--with-bellows=DIR],
    [root directory of the Bellows installation or checkout])],
  [with_bellows="$withval"],
  [with_bellows="auto"])

AC_ARG_WITH([hyper-math],
  [AS_HELP_STRING([--with-hyper-math=DIR],
    [root directory of the Hyper Math installation or checkout])],
  [with_hyper_math="$withval"],
  [with_hyper_math="auto"])

if test "x$with_bellows" = "xauto"; then
  for cand in \
    "$ac_abs_confdir/../Bellows" \
    "$ac_abs_confdir/../Artificial Intelligence/Bellows" \
    "$HOME/.local" \
    "/usr/local" \
    "/usr"; do
    if test -f "$cand/include/bellows/bellows.hpp"; then
      with_bellows="$cand"
      break
    fi
  done
fi

if test "x$with_hyper_math" = "xauto"; then
  for cand in \
    "$ac_abs_confdir/../Hyper Math" \
    "$ac_abs_confdir/../Artificial Intelligence/Hyper Math" \
    "$HOME/.local" \
    "/usr/local" \
    "/usr"; do
    if test -f "$cand/include/hyper_math/hyper_math.hpp"; then
      with_hyper_math="$cand"
      break
    fi
  done
fi

if test ! -f "$with_bellows/include/bellows/bellows.hpp"; then
  AC_MSG_ERROR([Bellows headers not found in $with_bellows. Use --with-bellows=DIR.])
fi
if test ! -f "$with_hyper_math/include/hyper_math/hyper_math.hpp"; then
  AC_MSG_ERROR([Hyper Math headers not found in $with_hyper_math. Use --with-hyper-math=DIR.])
fi

# Create space-free symlinks under /tmp/basalt-deps-kival so Make and Libtool
# never split arguments when workspace paths contain spaces.
BASALT_DEPS_LINK_DIR="/tmp/basalt-deps-kival"
mkdir -p "$BASALT_DEPS_LINK_DIR"
ln -sfn "$with_bellows" "$BASALT_DEPS_LINK_DIR/bellows"
ln -sfn "$with_hyper_math" "$BASALT_DEPS_LINK_DIR/hyper_math"

BELLOWS_CPPFLAGS="-isystem $BASALT_DEPS_LINK_DIR/bellows/include -isystem $BASALT_DEPS_LINK_DIR/hyper_math/include"
if test -d "$BASALT_DEPS_LINK_DIR/bellows/lib"; then
  BELLOWS_LIBDIR="$BASALT_DEPS_LINK_DIR/bellows/lib"
else
  BELLOWS_LIBDIR="$BASALT_DEPS_LINK_DIR/bellows"
fi
if test -d "$BASALT_DEPS_LINK_DIR/hyper_math/lib"; then
  HYPER_MATH_LIBDIR="$BASALT_DEPS_LINK_DIR/hyper_math/lib"
else
  HYPER_MATH_LIBDIR="$BASALT_DEPS_LINK_DIR/hyper_math"
fi
BELLOWS_LDFLAGS="-L$BELLOWS_LIBDIR -L$HYPER_MATH_LIBDIR -Wl,-rpath,$BELLOWS_LIBDIR -Wl,-rpath,$HYPER_MATH_LIBDIR"
BELLOWS_LIBS="-lbellows -lhypermath -pthread -ldl"

AC_SUBST([BELLOWS_CPPFLAGS])
AC_SUBST([BELLOWS_LDFLAGS])
AC_SUBST([BELLOWS_LIBS])

# Optional features.
AC_ARG_ENABLE([cuda],
  [AS_HELP_STRING([--enable-cuda],
    [enable runtime-loaded CUDA driver backend (default=yes)])],
  [enable_cuda="$enableval"],
  [enable_cuda="yes"])
if test "x$enable_cuda" = "xyes"; then
  AC_DEFINE([BASALT_ENABLE_CUDA], [1],
    [Define to 1 if runtime-loaded CUDA backend is enabled.])
fi
AM_CONDITIONAL([ENABLE_CUDA], [test "x$enable_cuda" = "xyes"])

AC_ARG_ENABLE([python],
  [AS_HELP_STRING([--enable-python],
    [build pybind11 Python bindings (default=no)])],
  [enable_python="$enableval"],
  [enable_python="no"])
AM_CONDITIONAL([ENABLE_PYTHON], [test "x$enable_python" = "xyes"])

AC_ARG_ENABLE([sanitizers],
  [AS_HELP_STRING([--enable-sanitizers],
    [enable AddressSanitizer and UndefinedBehaviorSanitizer (default=no)])],
  [enable_sanitizers="$enableval"],
  [enable_sanitizers="no"])
SANITIZER_FLAGS=""
if test "x$enable_sanitizers" = "xyes"; then
  SANITIZER_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
fi

AC_ARG_ENABLE([tsan],
  [AS_HELP_STRING([--enable-tsan],
    [enable ThreadSanitizer (default=no)])],
  [enable_tsan="$enableval"],
  [enable_tsan="no"])
if test "x$enable_tsan" = "xyes"; then
  SANITIZER_FLAGS="-fsanitize=thread -fno-omit-frame-pointer"
fi
AC_SUBST([SANITIZER_FLAGS])

AC_CONFIG_FILES([
  Makefile
  include/basalt/version.h
  python/basalt/_version.py
  setup.cfg
])
AC_OUTPUT
