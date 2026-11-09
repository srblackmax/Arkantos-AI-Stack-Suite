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
AC_INIT([Obsidian], [0.1.0], [Kival@obsidian-nlp.org], [obsidian])
AC_CONFIG_SRCDIR([include/obsidian/version.h.in])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_HEADERS([config.h])

# Obsidian is licensed under GPLv3+ and is not part of the GNU Project.
AM_INIT_AUTOMAKE([1.16 foreign subdir-objects -Wall])
AM_SILENT_RULES([yes])

OBSIDIAN_VERSION_MAJOR=`echo "$PACKAGE_VERSION" | cut -d. -f1`
OBSIDIAN_VERSION_MINOR=`echo "$PACKAGE_VERSION" | cut -d. -f2`
OBSIDIAN_VERSION_PATCH=`echo "$PACKAGE_VERSION" | cut -d. -f3`
OBSIDIAN_VERSION_STRING="$PACKAGE_VERSION"
AC_SUBST([OBSIDIAN_VERSION_MAJOR])
AC_SUBST([OBSIDIAN_VERSION_MINOR])
AC_SUBST([OBSIDIAN_VERSION_PATCH])
AC_SUBST([OBSIDIAN_VERSION_STRING])

AC_DEFINE_UNQUOTED([OBSIDIAN_VERSION_MAJOR], [$OBSIDIAN_VERSION_MAJOR],
  [Major version of Obsidian.])
AC_DEFINE_UNQUOTED([OBSIDIAN_VERSION_MINOR], [$OBSIDIAN_VERSION_MINOR],
  [Minor version of Obsidian.])
AC_DEFINE_UNQUOTED([OBSIDIAN_VERSION_PATCH], [$OBSIDIAN_VERSION_PATCH],
  [Patch version of Obsidian.])
AC_DEFINE_UNQUOTED([OBSIDIAN_VERSION_STRING], ["$OBSIDIAN_VERSION_STRING"],
  [Version string of Obsidian.])

AC_PROG_CC
AC_PROG_CXX
AM_PROG_AR
LT_INIT
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# Require C++23 for std::expected, concepts, and ranges.
CXXFLAGS="$CXXFLAGS -std=c++23"
AC_LANG_PUSH([C++])
AC_MSG_CHECKING([whether $CXX supports C++23])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <concepts>
                     #include <expected>
                     #include <span>]],
                   [[std::expected<int, int> x = 1; (void)x;]])],
  [AC_MSG_RESULT([yes])],
  [AC_MSG_RESULT([no])
   AC_MSG_ERROR([A C++23-compliant compiler is required to build Obsidian.])])
AC_LANG_POP([C++])

# Locate Basalt Deep Learning, Bellows, and Hyper Math.
AC_ARG_WITH([basalt],
  [AS_HELP_STRING([--with-basalt=DIR],
    [root directory of the Basalt Deep Learning installation or checkout])],
  [with_basalt="$withval"],
  [with_basalt="auto"])

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

if test "x$with_basalt" = "xauto"; then
  for cand in \
    "$ac_abs_confdir/../Basalt Deep Learning" \
    ". Deep Learning" \
    "/usr/local" \
    "/usr"; do
    if test -f "$cand/include/basalt/basalt.hpp"; then
      with_basalt="$cand"
      break
    fi
  done
fi

if test "x$with_bellows" = "xauto"; then
  for cand in \
    "$ac_abs_confdir/../Bellows" \
    "." \
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
    ". Math" \
    "/usr/local" \
    "/usr"; do
    if test -f "$cand/include/hyper_math/hyper_math.hpp"; then
      with_hyper_math="$cand"
      break
    fi
  done
fi

if test ! -f "$with_basalt/include/basalt/basalt.hpp"; then
  AC_MSG_ERROR([Basalt headers not found in $with_basalt. Use --with-basalt=DIR.])
fi

# Create space-free symlinks under /tmp/obsidian-deps-kival so Make and Libtool
# never split arguments when workspace paths contain spaces.
OBSIDIAN_DEPS_LINK_DIR="/tmp/obsidian-deps-kival"
mkdir -p "$OBSIDIAN_DEPS_LINK_DIR"
ln -sfn "$with_basalt" "$OBSIDIAN_DEPS_LINK_DIR/basalt"
ln -sfn "$with_bellows" "$OBSIDIAN_DEPS_LINK_DIR/bellows"
ln -sfn "$with_hyper_math" "$OBSIDIAN_DEPS_LINK_DIR/hyper_math"

BASALT_CPPFLAGS="-isystem $OBSIDIAN_DEPS_LINK_DIR/basalt/include -isystem $OBSIDIAN_DEPS_LINK_DIR/bellows/include -isystem $OBSIDIAN_DEPS_LINK_DIR/hyper_math/include"

if test -d "$OBSIDIAN_DEPS_LINK_DIR/basalt/.libs"; then
  BASALT_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/basalt/.libs"
elif test -d "$OBSIDIAN_DEPS_LINK_DIR/basalt/lib"; then
  BASALT_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/basalt/lib"
else
  BASALT_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/basalt"
fi

if test -d "$OBSIDIAN_DEPS_LINK_DIR/bellows/lib"; then
  BELLOWS_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/bellows/lib"
else
  BELLOWS_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/bellows"
fi

if test -d "$OBSIDIAN_DEPS_LINK_DIR/hyper_math/lib"; then
  HYPER_MATH_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/hyper_math/lib"
else
  HYPER_MATH_LIBDIR="$OBSIDIAN_DEPS_LINK_DIR/hyper_math"
fi

BASALT_LDFLAGS="-L$BASALT_LIBDIR -L$BELLOWS_LIBDIR -L$HYPER_MATH_LIBDIR -Wl,-rpath,$BASALT_LIBDIR -Wl,-rpath,$BELLOWS_LIBDIR -Wl,-rpath,$HYPER_MATH_LIBDIR"
BASALT_LIBS="-lbasalt -lbellows -lhypermath -pthread -ldl"

AC_SUBST([BASALT_CPPFLAGS])
AC_SUBST([BASALT_LDFLAGS])
AC_SUBST([BASALT_LIBS])

AC_ARG_ENABLE([sanitizers],
  [AS_HELP_STRING([--enable-sanitizers],
    [enable AddressSanitizer and UndefinedBehaviorSanitizer (default=no)])],
  [enable_sanitizers="$enableval"],
  [enable_sanitizers="no"])
SANITIZER_FLAGS=""
if test "x$enable_sanitizers" = "xyes"; then
  SANITIZER_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
fi
AC_SUBST([SANITIZER_FLAGS])

AC_CONFIG_FILES([
  Makefile
  include/obsidian/version.h
])
AC_OUTPUT
