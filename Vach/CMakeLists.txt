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

cmake_minimum_required(VERSION 3.28)

project(Vach
  VERSION 1.0.0
  DESCRIPTION "Vach: Native C++23 Voice I/O Library (VAD, STT, TTS, Lock-Free AudioIo)"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VACH_BUILD_TESTS "Build Vach test suite" ON)
option(VACH_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(VACH_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

add_library(vach_warnings INTERFACE)
target_compile_options(vach_warnings INTERFACE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  $<$<CONFIG:Release>:-O3 -march=native>
)

if(VACH_ENABLE_ASAN)
  target_compile_options(vach_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(vach_warnings INTERFACE -fsanitize=address,undefined)
endif()

if(VACH_ENABLE_TSAN)
  target_compile_options(vach_warnings INTERFACE -fsanitize=thread -fno-omit-frame-pointer -g)
  target_link_options(vach_warnings INTERFACE -fsanitize=thread)
endif()

find_package(Threads REQUIRED)

add_library(vach SHARED
  src/error.cpp
  src/observability.cpp
  src/vach.cpp
)
add_library(Vach::Vach ALIAS vach)

set_target_properties(vach PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME Vach
)

target_include_directories(vach
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $ENV{HOME}/.local/include
)

target_compile_definitions(vach PUBLIC SUTRA_ENABLE_STATS=1)
target_link_libraries(vach
  PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
  PRIVATE
    vach_warnings
)

if(VACH_BUILD_TESTS)
  enable_testing()

  add_executable(check_gnu_style tests/check_gnu_style.cpp)
  target_link_libraries(check_gnu_style PRIVATE vach_warnings)
  add_test(NAME check_gnu_style COMMAND check_gnu_style ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_vach tests/test_vach.cpp)
  target_link_libraries(test_vach PRIVATE Vach::Vach vach_warnings)
  add_test(NAME test_vach COMMAND test_vach)
  if(VACH_ENABLE_ASAN)
    set_tests_properties(test_vach PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=quarantine_size_mb=1:protect_shadow_gap=0:detect_leaks=0"
    )
  elseif(VACH_ENABLE_TSAN)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tsan.supp"
      "${CMAKE_CURRENT_BINARY_DIR}/tsan.supp"
      COPYONLY
    )
    set_tests_properties(test_vach PROPERTIES
      ENVIRONMENT "TSAN_OPTIONS=suppressions=tsan.supp"
    )
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS vach vach_warnings
  EXPORT VachTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/vach
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT VachTargets
  FILE VachTargets.cmake
  NAMESPACE Vach::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Vach
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VachConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/VachConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Vach
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/VachConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/VachConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/VachConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Vach
)
