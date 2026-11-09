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
project(Darshana VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
enable_testing()

option(DARSHANA_BUILD_TESTS "Build Darshana unit, property, golden, and stress test suite" ON)
option(DARSHANA_BUILD_BENCH "Build Darshana benchmark suite" ON)
option(DARSHANA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(DARSHANA_USE_TSAN "Enable ThreadSanitizer" OFF)
option(DARSHANA_ENABLE_VULKAN "Enable Vulkan 1.2+ GPU interactive backend" ON)
option(DARSHANA_BUILD_FAT_NATIVE "Build with -march=native instead of portable -march=x86-64-v3" OFF)
option(DARSHANA_ENABLE_STATS "Enable per-frame timing counters in FrameStats" ON)

set(DARSHANA_ENABLE_VULKAN_BOOL "FALSE")
if(DARSHANA_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  set(DARSHANA_ENABLE_VULKAN_BOOL "TRUE")
endif()

set(DARSHANA_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)
if(DARSHANA_BUILD_FAT_NATIVE)
  set(DARSHANA_ARCH_FLAGS -march=native)
else()
  set(DARSHANA_ARCH_FLAGS -march=x86-64-v3)
endif()
set(DARSHANA_PERF_FLAGS -O3 ${DARSHANA_ARCH_FLAGS})

set(DARSHANA_SANITIZER_FLAGS "")
if(DARSHANA_USE_SANITIZERS)
  set(DARSHANA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
elseif(DARSHANA_USE_TSAN)
  set(DARSHANA_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer -g)
endif()

add_library(darshana SHARED
  src/error.cpp
  src/observability.cpp
  src/memory.cpp
  src/spec.cpp
  src/stat.cpp
  src/gpu.cpp
  src/export.cpp
  src/figure.cpp
  src/render_loop.cpp
)
add_library(Darshana::Darshana ALIAS darshana)

target_compile_features(darshana PUBLIC cxx_std_23)
set_target_properties(darshana PROPERTIES
  CXX_EXTENSIONS OFF
  EXPORT_NAME Darshana
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

if(DARSHANA_ENABLE_STATS)
  target_compile_definitions(darshana PUBLIC DARSHANA_ENABLE_STATS=1)
else()
  target_compile_definitions(darshana PUBLIC DARSHANA_ENABLE_STATS=0)
endif()

if(DARSHANA_ENABLE_VULKAN)
  target_compile_definitions(darshana PRIVATE DARSHANA_HAS_VULKAN=1)
else()
  target_compile_definitions(darshana PRIVATE DARSHANA_HAS_VULKAN=0)
endif()

target_include_directories(darshana
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(DARSHANA_ENABLE_VULKAN)
  target_include_directories(darshana SYSTEM PRIVATE ${Vulkan_INCLUDE_DIRS})
endif()

target_compile_options(darshana
  PUBLIC
    ${DARSHANA_WARNING_FLAGS}
    ${DARSHANA_SANITIZER_FLAGS}
  PRIVATE
    ${DARSHANA_PERF_FLAGS}
)

target_link_options(darshana
  PUBLIC
    ${DARSHANA_SANITIZER_FLAGS}
)

target_link_libraries(darshana
  PRIVATE
    pthread
    $<$<BOOL:${DARSHANA_ENABLE_VULKAN}>:Vulkan::Vulkan>
)

function(darshana_register_test target_name src_file)
  add_executable(${target_name} ${src_file})
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  set_target_properties(${target_name} PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(${target_name} PRIVATE Darshana::Darshana)
  add_test(NAME ${target_name} COMMAND ${target_name})
  if(DARSHANA_USE_SANITIZERS)
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=quarantine_size_mb=1:protect_shadow_gap=0:detect_leaks=0")
  elseif(DARSHANA_USE_TSAN)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tsan.supp"
      "${CMAKE_CURRENT_BINARY_DIR}/tsan.supp"
      COPYONLY
    )
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "TSAN_OPTIONS=suppressions=tsan.supp")
  endif()
endfunction()

if(DARSHANA_BUILD_TESTS)
  add_executable(check_gnu_style tests/check_gnu_style.cpp)
  target_compile_features(check_gnu_style PRIVATE cxx_std_23)
  target_compile_options(check_gnu_style PRIVATE ${DARSHANA_WARNING_FLAGS} ${DARSHANA_SANITIZER_FLAGS})
  target_link_options(check_gnu_style PRIVATE ${DARSHANA_SANITIZER_FLAGS})
  add_test(NAME check_gnu_style
           COMMAND check_gnu_style "${CMAKE_CURRENT_SOURCE_DIR}")

  darshana_register_test(test_unit tests/test_unit.cpp)
  darshana_register_test(test_integration tests/test_integration.cpp)
  darshana_register_test(test_property tests/test_property.cpp)
  darshana_register_test(test_golden tests/test_golden.cpp)
  darshana_register_test(test_stress tests/test_stress.cpp)
endif()

if(DARSHANA_BUILD_BENCH)
  add_executable(bench_darshana bench/bench_darshana.cpp)
  target_compile_features(bench_darshana PRIVATE cxx_std_23)
  target_compile_options(bench_darshana PRIVATE ${DARSHANA_PERF_FLAGS})
  if(DARSHANA_ENABLE_VULKAN)
    target_compile_definitions(bench_darshana PRIVATE DARSHANA_HAS_VULKAN=1)
    target_include_directories(bench_darshana SYSTEM PRIVATE ${Vulkan_INCLUDE_DIRS})
  else()
    target_compile_definitions(bench_darshana PRIVATE DARSHANA_HAS_VULKAN=0)
  endif()
  target_link_libraries(bench_darshana PRIVATE
    Darshana::Darshana
    $<$<BOOL:${DARSHANA_ENABLE_VULKAN}>:Vulkan::Vulkan>
    $<$<BOOL:${DARSHANA_ENABLE_VULKAN}>:xcb>
  )
  add_custom_target(bench-ci
    COMMAND "$<TARGET_FILE:bench_darshana>"
            --json "${CMAKE_CURRENT_BINARY_DIR}/results.json"
            --check-baseline "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
    DEPENDS bench_darshana
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Run Darshana benchmark harness and verify <= 15% regression against benchmarks/baseline.json"
    VERBATIM
  )
endif()

install(TARGETS darshana
  EXPORT DarshanaTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/darshana
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DarshanaConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DarshanaConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Darshana"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DarshanaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT DarshanaTargets
  FILE DarshanaTargets.cmake
  NAMESPACE Darshana::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Darshana"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DarshanaConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DarshanaConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Darshana"
)
