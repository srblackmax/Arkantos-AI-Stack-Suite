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
project(Yantra VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
enable_testing()

option(YANTRA_BUILD_TESTS "Build Yantra unit, property, stress, and style test suite" ON)
option(YANTRA_BUILD_BENCH "Build Yantra benchmark suite" ON)
option(YANTRA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(YANTRA_USE_TSAN "Enable ThreadSanitizer" OFF)
option(YANTRA_ENABLE_CUDA "Enable CUDA 12.x GPU vector index acceleration" ON)
option(YANTRA_BUILD_FAT_NATIVE "Build with -march=native instead of portable -march=x86-64-v3" OFF)
option(YANTRA_ENABLE_STATS "Enable per-module telemetry counters in Yantra" ON)

set(YANTRA_ENABLE_CUDA_BOOL "FALSE")
if(YANTRA_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-13")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(YANTRA_ENABLE_CUDA_BOOL "TRUE")
endif()

set(YANTRA_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)
if(YANTRA_BUILD_FAT_NATIVE)
  set(YANTRA_ARCH_FLAGS -march=native)
else()
  set(YANTRA_ARCH_FLAGS -march=x86-64-v3)
endif()
set(YANTRA_PERF_FLAGS -O3 ${YANTRA_ARCH_FLAGS})

set(YANTRA_SANITIZER_FLAGS "")
if(YANTRA_USE_SANITIZERS)
  set(YANTRA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
elseif(YANTRA_USE_TSAN)
  set(YANTRA_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer -g)
endif()

set(YANTRA_CUDA_OBJECTS "")
if(YANTRA_ENABLE_CUDA)
  add_library(yantra_cuda_kernels OBJECT
    src/gpu/kernels.cu
  )
  set_target_properties(yantra_cuda_kernels PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(yantra_cuda_kernels PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  set(YANTRA_CUDA_OBJECTS $<TARGET_OBJECTS:yantra_cuda_kernels>)
endif()

add_library(yantra SHARED
  src/error.cpp
  src/observability.cpp
  src/gpu.cpp
  src/track.cpp
  src/registry.cpp
  src/dag.cpp
  src/index.cpp
  src/fairness.cpp
  ${YANTRA_CUDA_OBJECTS}
)
add_library(Yantra::Yantra ALIAS yantra)

target_compile_features(yantra PUBLIC cxx_std_23)
set_target_properties(yantra PROPERTIES
  CXX_EXTENSIONS OFF
  EXPORT_NAME Yantra
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

if(YANTRA_ENABLE_STATS)
  target_compile_definitions(yantra PUBLIC YANTRA_ENABLE_STATS=1)
else()
  target_compile_definitions(yantra PUBLIC YANTRA_ENABLE_STATS=0)
endif()

if(YANTRA_ENABLE_CUDA)
  target_compile_definitions(yantra PRIVATE YANTRA_HAS_CUDA=1)
else()
  target_compile_definitions(yantra PRIVATE YANTRA_HAS_CUDA=0)
endif()

target_include_directories(yantra
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(YANTRA_ENABLE_CUDA)
  target_include_directories(yantra SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_compile_options(yantra
  PUBLIC
    ${YANTRA_WARNING_FLAGS}
    ${YANTRA_SANITIZER_FLAGS}
  PRIVATE
    ${YANTRA_PERF_FLAGS}
)

target_link_options(yantra
  PUBLIC
    ${YANTRA_SANITIZER_FLAGS}
)

target_link_libraries(yantra
  PRIVATE
    pthread
    $<$<BOOL:${YANTRA_ENABLE_CUDA}>:CUDA::cudart>
)

function(yantra_register_test target_name src_file)
  add_executable(${target_name} ${src_file})
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  set_target_properties(${target_name} PROPERTIES CXX_EXTENSIONS OFF)
  target_compile_options(${target_name} PRIVATE ${YANTRA_WARNING_FLAGS} ${YANTRA_SANITIZER_FLAGS})
  target_link_options(${target_name} PRIVATE ${YANTRA_SANITIZER_FLAGS})
  target_link_libraries(${target_name} PRIVATE Yantra::Yantra)
  add_test(NAME ${target_name} COMMAND ${target_name})
  if(YANTRA_USE_SANITIZERS)
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=quarantine_size_mb=1:protect_shadow_gap=0:detect_leaks=0")
  elseif(YANTRA_USE_TSAN)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tsan.supp"
      "${CMAKE_CURRENT_BINARY_DIR}/tsan.supp"
      COPYONLY
    )
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "TSAN_OPTIONS=suppressions=tsan.supp")
  endif()
endfunction()

if(YANTRA_BUILD_TESTS)
  add_executable(check_gnu_style tests/check_gnu_style.cpp)
  target_compile_features(check_gnu_style PRIVATE cxx_std_23)
  target_compile_options(check_gnu_style PRIVATE ${YANTRA_WARNING_FLAGS} ${YANTRA_SANITIZER_FLAGS})
  target_link_options(check_gnu_style PRIVATE ${YANTRA_SANITIZER_FLAGS})
  add_test(NAME check_gnu_style
           COMMAND check_gnu_style "${CMAKE_CURRENT_SOURCE_DIR}")

  yantra_register_test(test_unit tests/test_unit.cpp)
  yantra_register_test(test_property tests/test_property.cpp)
  yantra_register_test(test_stress tests/test_stress.cpp)
  yantra_register_test(test_hnsw_connectivity tests/test_hnsw_connectivity.cpp)
  yantra_register_test(test_ivfpq_recall tests/test_ivfpq_recall.cpp)
  yantra_register_test(test_dag_overhead tests/test_dag_overhead.cpp)

  add_executable(diag_hnsw_1m tests/diag_hnsw_1m.cpp)
  target_compile_features(diag_hnsw_1m PRIVATE cxx_std_23)
  target_compile_options(diag_hnsw_1m PRIVATE ${YANTRA_WARNING_FLAGS} ${YANTRA_PERF_FLAGS})
  target_link_libraries(diag_hnsw_1m PRIVATE Yantra::Yantra)
endif()

if(YANTRA_BUILD_BENCH)
  add_executable(bench_yantra bench/bench_yantra.cpp)
  target_compile_features(bench_yantra PRIVATE cxx_std_23)
  target_compile_options(bench_yantra PRIVATE ${YANTRA_WARNING_FLAGS} ${YANTRA_PERF_FLAGS})
  target_link_libraries(bench_yantra PRIVATE Yantra::Yantra)

  add_custom_target(bench-ci
    COMMAND "$<TARGET_FILE:bench_yantra>"
            "--json-out=${CMAKE_CURRENT_SOURCE_DIR}/results.json"
            "--check-baseline=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
    DEPENDS bench_yantra
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Running Yantra benchmark CI regression & sanity suite"
  )
endif()

install(TARGETS yantra
  EXPORT YantraTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/yantra
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/YantraConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/YantraConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Yantra"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/YantraConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT YantraTargets
  FILE YantraTargets.cmake
  NAMESPACE Yantra::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Yantra"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/YantraConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/YantraConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Yantra"
)
