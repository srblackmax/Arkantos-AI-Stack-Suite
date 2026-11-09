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
project(Sutra VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
enable_testing()

option(SUTRA_BUILD_TESTS "Build Sutra unit, property, and stress test suite" ON)
option(SUTRA_BUILD_BENCH "Build Sutra benchmark suite" ON)
option(SUTRA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(SUTRA_USE_TSAN "Enable ThreadSanitizer" OFF)
option(SUTRA_ENABLE_CUDA "Enable CUDA 12.x GPU vector index acceleration" ON)
option(SUTRA_BUILD_FAT_NATIVE "Build with -march=native instead of portable -march=x86-64-v3" OFF)
option(SUTRA_ENABLE_STATS "Enable per-stage timing counters in AgentStats" ON)

set(SUTRA_ENABLE_CUDA_BOOL "FALSE")
if(SUTRA_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-13")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(SUTRA_ENABLE_CUDA_BOOL "TRUE")
endif()

set(SUTRA_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)
if(SUTRA_BUILD_FAT_NATIVE)
  set(SUTRA_ARCH_FLAGS -march=native)
else()
  set(SUTRA_ARCH_FLAGS -march=x86-64-v3)
endif()
set(SUTRA_PERF_FLAGS -O3 ${SUTRA_ARCH_FLAGS})

set(SUTRA_SANITIZER_FLAGS "")
if(SUTRA_USE_SANITIZERS)
  set(SUTRA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
elseif(SUTRA_USE_TSAN)
  set(SUTRA_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer -g)
endif()

set(SUTRA_CUDA_OBJECTS "")
if(SUTRA_ENABLE_CUDA)
  add_library(sutra_cuda_kernels OBJECT
    src/gpu/kernels.cu
  )
  set_target_properties(sutra_cuda_kernels PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(sutra_cuda_kernels PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  set(SUTRA_CUDA_OBJECTS $<TARGET_OBJECTS:sutra_cuda_kernels>)
endif()

add_library(sutra SHARED
  src/error.cpp
  src/observability.cpp
  src/gpu.cpp
  src/memory.cpp
  src/tool.cpp
  src/scheduler.cpp
  src/workflow.cpp
  src/planner.cpp
  src/llm_backends.cpp
  src/checkpoint.cpp
  src/runtime.cpp
  ${SUTRA_CUDA_OBJECTS}
)
add_library(Sutra::Sutra ALIAS sutra)

set(SUTRA_LLAMACPP_ROOT "$ENV{HOME}/llama.cpp" CACHE PATH "Path to llama.cpp checkout")
find_library(SUTRA_LLAMA_LIB NAMES llama PATHS "${SUTRA_LLAMACPP_ROOT}/build/bin" NO_DEFAULT_PATH)
find_library(SUTRA_GGML_CPU_LIB NAMES ggml-cpu PATHS "${SUTRA_LLAMACPP_ROOT}/build/bin" NO_DEFAULT_PATH)
find_library(SUTRA_GGML_BASE_LIB NAMES ggml-base PATHS "${SUTRA_LLAMACPP_ROOT}/build/bin" NO_DEFAULT_PATH)
if(SUTRA_LLAMA_LIB AND EXISTS "${SUTRA_LLAMACPP_ROOT}/include/llama.h")
  target_compile_definitions(sutra PRIVATE SUTRA_HAS_LLAMACPP=1)
  target_include_directories(sutra SYSTEM PRIVATE
    "${SUTRA_LLAMACPP_ROOT}/include"
    "${SUTRA_LLAMACPP_ROOT}/ggml/include"
  )
  target_link_libraries(sutra PRIVATE "${SUTRA_LLAMA_LIB}")
  if(SUTRA_GGML_CPU_LIB)
    target_link_libraries(sutra PRIVATE "${SUTRA_GGML_CPU_LIB}")
  endif()
  if(SUTRA_GGML_BASE_LIB)
    target_link_libraries(sutra PRIVATE "${SUTRA_GGML_BASE_LIB}")
  endif()
  set_target_properties(sutra PROPERTIES
    BUILD_RPATH "${SUTRA_LLAMACPP_ROOT}/build/bin"
    INSTALL_RPATH "${SUTRA_LLAMACPP_ROOT}/build/bin"
  )
else()
  target_compile_definitions(sutra PRIVATE SUTRA_HAS_LLAMACPP=0)
endif()

target_compile_features(sutra PUBLIC cxx_std_23)
set_target_properties(sutra PROPERTIES
  CXX_EXTENSIONS OFF
  EXPORT_NAME Sutra
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

if(SUTRA_ENABLE_STATS)
  target_compile_definitions(sutra PUBLIC SUTRA_ENABLE_STATS=1)
else()
  target_compile_definitions(sutra PUBLIC SUTRA_ENABLE_STATS=0)
endif()

if(SUTRA_ENABLE_CUDA)
  target_compile_definitions(sutra PRIVATE SUTRA_HAS_CUDA=1)
else()
  target_compile_definitions(sutra PRIVATE SUTRA_HAS_CUDA=0)
endif()

target_include_directories(sutra
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(SUTRA_ENABLE_CUDA)
  target_include_directories(sutra SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_compile_options(sutra
  PUBLIC
    ${SUTRA_WARNING_FLAGS}
    ${SUTRA_SANITIZER_FLAGS}
  PRIVATE
    ${SUTRA_PERF_FLAGS}
)

target_link_options(sutra
  PUBLIC
    ${SUTRA_SANITIZER_FLAGS}
)

target_link_libraries(sutra
  PRIVATE
    pthread
    $<$<BOOL:${SUTRA_ENABLE_CUDA}>:CUDA::cudart>
)

add_library(vach SHARED
  src/vach.cpp
  src/sphinx.cpp
)
add_library(Vach::Vach ALIAS vach)
target_compile_features(vach PUBLIC cxx_std_23)
set_target_properties(vach PROPERTIES
  CXX_EXTENSIONS OFF
  EXPORT_NAME Vach
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)
target_include_directories(vach
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $ENV{HOME}/.local/include
)
target_compile_options(vach
  PUBLIC
    ${SUTRA_WARNING_FLAGS}
    ${SUTRA_SANITIZER_FLAGS}
  PRIVATE
    ${SUTRA_PERF_FLAGS}
)
target_link_options(vach
  PUBLIC
    ${SUTRA_SANITIZER_FLAGS}
)
target_link_libraries(vach
  PUBLIC
    Sutra::Sutra
  PRIVATE
    pthread
    ${CMAKE_DL_LIBS}
)

function(sutra_register_test target_name src_file)
  add_executable(${target_name} ${src_file})
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  set_target_properties(${target_name} PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(${target_name} PRIVATE Sutra::Sutra Vach::Vach)
  add_test(NAME ${target_name} COMMAND ${target_name})
  if(SUTRA_USE_SANITIZERS)
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=quarantine_size_mb=1:protect_shadow_gap=0:detect_leaks=0")
  elseif(SUTRA_USE_TSAN)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tsan.supp"
      "${CMAKE_CURRENT_BINARY_DIR}/tsan.supp"
      COPYONLY
    )
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "TSAN_OPTIONS=suppressions=tsan.supp")
  endif()
endfunction()

if(SUTRA_BUILD_TESTS)
  add_executable(check_gnu_style tests/check_gnu_style.cpp)
  target_compile_features(check_gnu_style PRIVATE cxx_std_23)
  target_compile_options(check_gnu_style PRIVATE ${SUTRA_WARNING_FLAGS} ${SUTRA_SANITIZER_FLAGS})
  target_link_options(check_gnu_style PRIVATE ${SUTRA_SANITIZER_FLAGS})
  add_test(NAME check_gnu_style
           COMMAND check_gnu_style "${CMAKE_CURRENT_SOURCE_DIR}")

  sutra_register_test(test_unit tests/test_unit.cpp)
  sutra_register_test(test_integration tests/test_integration.cpp)
  sutra_register_test(test_property tests/test_property.cpp)
  sutra_register_test(test_equivalence tests/test_equivalence.cpp)
  sutra_register_test(test_stress tests/test_stress.cpp)
  sutra_register_test(test_llm_backends tests/test_llm_backends.cpp)
  sutra_register_test(test_logit_parity tests/test_logit_parity.cpp)
  sutra_register_test(test_vach tests/test_vach.cpp)
  sutra_register_test(test_sphinx_slice tests/test_sphinx_slice.cpp)
endif()

if(SUTRA_BUILD_BENCH)
  add_executable(bench_sutra bench/bench_sutra.cpp)
  target_compile_features(bench_sutra PRIVATE cxx_std_23)
  target_link_libraries(bench_sutra PRIVATE Sutra::Sutra Vach::Vach)

  add_custom_target(bench-ci
    COMMAND "$<TARGET_FILE:bench_sutra>"
            "--json-out=${CMAKE_CURRENT_BINARY_DIR}/results.json"
            "--check-baseline=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
    DEPENDS bench_sutra
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Running Sutra benchmark CI regression & sanity suite"
  )
endif()

install(TARGETS sutra
  EXPORT SutraTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS vach
  EXPORT VachTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/sutra include/vach include/sphinx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SutraConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/SutraConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Sutra"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SutraConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT SutraTargets
  FILE SutraTargets.cmake
  NAMESPACE Sutra::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Sutra"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/SutraConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/SutraConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Sutra"
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VachConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/VachConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vach"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/VachConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT VachTargets
  FILE VachTargets.cmake
  NAMESPACE Vach::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vach"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/VachConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/VachConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vach"
)
