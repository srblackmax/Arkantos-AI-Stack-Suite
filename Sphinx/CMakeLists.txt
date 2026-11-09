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

cmake_minimum_required(VERSION 3.25)
project(Sphinx VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
set(CMAKE_INSTALL_RPATH "$ENV{HOME}/.local/lib;$ENV{HOME}/llama.cpp/build/bin")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_link_options(-Wl,--disable-new-dtags)

option(SPHINX_ENABLE_ASAN "Enable AddressSanitizer + UBSan" OFF)
option(SPHINX_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(SPHINX_ENABLE_CUDA "Enable CUDA GPU acceleration via llama.cpp" ON)
option(SPHINX_PORTABLE_CPU "Build portable CPU binary (x86-64-v3 AVX2/FMA, no host-specific -march=native or CUDA runtime dependency)" OFF)
option(SPHINX_ENABLE_LTO "Enable Link-Time Optimization and dead-section garbage collection" ON)
option(SPHINX_HEADLESS "Build in headless audio mode" OFF)
option(SPHINX_BUILD_TESTS "Build Sphinx test suite" ON)
option(SPHINX_BUILD_BENCHMARKS "Build Sphinx aliveness benchmark" ON)

if(NOT SPHINX_ENABLE_ASAN AND NOT SPHINX_ENABLE_TSAN)
  add_compile_options(-ffunction-sections -fdata-sections)
  add_link_options(-Wl,--gc-sections)
  if(SPHINX_PORTABLE_CPU)
    add_compile_options(-march=x86-64-v3)
  endif()
endif()

find_package(Sutra REQUIRED)
find_package(Vach REQUIRED)
find_package(Yantra REQUIRED)
find_package(Darshana REQUIRED)
find_package(Threads REQUIRED)

add_library(Sphinx SHARED
  src/memory.cpp
  src/router.cpp
  src/tools.cpp
  src/viz.cpp
  src/session.cpp
)
set_target_properties(Sphinx PROPERTIES OUTPUT_NAME sphinx)
add_library(Sphinx::Sphinx ALIAS Sphinx)
add_library(Sphinx::sphinx ALIAS Sphinx)

target_include_directories(Sphinx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $ENV{HOME}/llama.cpp/include
    $ENV{HOME}/llama.cpp/ggml/include
)

target_link_libraries(Sphinx
  PUBLIC
    Sutra::Sutra
    Vach::Vach
    Yantra::Yantra
    Darshana::Darshana
    Threads::Threads
    ${CMAKE_DL_LIBS}
  PRIVATE
    $ENV{HOME}/llama.cpp/build/bin/libllama.so
)

target_compile_options(Sphinx PRIVATE
  -Wall -Wextra -Wpedantic -Werror
)

if(SPHINX_HEADLESS)
  target_compile_definitions(Sphinx PUBLIC SPHINX_HEADLESS_DEFAULT=1)
endif()

if(SPHINX_PORTABLE_CPU OR NOT SPHINX_ENABLE_CUDA)
  target_compile_definitions(Sphinx PUBLIC SPHINX_PORTABLE_CPU_BUILD=1)
endif()

if(SPHINX_ENABLE_ASAN)
  target_compile_options(Sphinx PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(Sphinx PUBLIC -fsanitize=address,undefined)
endif()

if(SPHINX_ENABLE_TSAN)
  target_compile_options(Sphinx PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(Sphinx PUBLIC -fsanitize=thread)
endif()

add_executable(sphinx_cli src/main.cpp)
set_target_properties(sphinx_cli PROPERTIES OUTPUT_NAME sphinx)
target_link_libraries(sphinx_cli PRIVATE Sphinx::Sphinx)
target_compile_options(sphinx_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(SPHINX_BUILD_BENCHMARKS)
  add_executable(sphinx_bench bench/sphinx_bench.cpp)
  target_link_libraries(sphinx_bench PRIVATE Sphinx::Sphinx)
  target_compile_options(sphinx_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_custom_target(bench-ci
    COMMAND $<TARGET_FILE:sphinx_bench>
            --json-out ${CMAKE_CURRENT_BINARY_DIR}/bench_ci.json
            --check-baseline ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json
    DEPENDS sphinx_bench
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Sphinx aliveness benchmark against baseline.json"
  )
endif()

if(SPHINX_BUILD_TESTS)
  enable_testing()

  set(TSAN_ENV "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/cmake/tsan.supp:halt_on_error=1")
  set(ASAN_ENV "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0")

  add_executable(check_gnu_style tests/check_gnu_style.cpp)
  target_compile_options(check_gnu_style PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME check_gnu_style COMMAND check_gnu_style ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(test_unit tests/test_unit.cpp)
  target_link_libraries(test_unit PRIVATE Sphinx::sphinx)
  target_compile_options(test_unit PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME test_unit COMMAND test_unit)
  set_tests_properties(test_unit PROPERTIES ENVIRONMENT "${TSAN_ENV};${ASAN_ENV}")

  add_executable(test_integration tests/test_integration.cpp)
  target_link_libraries(test_integration PRIVATE Sphinx::sphinx)
  target_compile_options(test_integration PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME test_integration COMMAND test_integration)
  set_tests_properties(test_integration PROPERTIES ENVIRONMENT "${TSAN_ENV};${ASAN_ENV}")

  add_executable(test_property tests/test_property.cpp)
  target_link_libraries(test_property PRIVATE Sphinx::sphinx)
  target_compile_options(test_property PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME test_property COMMAND test_property)
  set_tests_properties(test_property PROPERTIES ENVIRONMENT "${TSAN_ENV};${ASAN_ENV}")

  add_executable(test_stress tests/test_stress.cpp)
  target_link_libraries(test_stress PRIVATE Sphinx::sphinx)
  target_compile_options(test_stress PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME test_stress COMMAND test_stress)
  set_tests_properties(test_stress PROPERTIES ENVIRONMENT "${TSAN_ENV};${ASAN_ENV}")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS Sphinx sphinx_cli
  EXPORT SphinxTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/sphinx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT SphinxTargets
  FILE SphinxTargets.cmake
  NAMESPACE Sphinx::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sphinx
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SphinxConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SphinxConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sphinx
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SphinxConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SphinxConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/SphinxConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sphinx
)
