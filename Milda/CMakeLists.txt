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
project(Milda VERSION 1.0.0 LANGUAGES C CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MILDA_ENABLE_CUDA "Enable NVIDIA CUDA backend support" ON)
option(MILDA_PORTABLE_CPU "Use portable AVX2/FMA flags (-march=x86-64-v3) instead of host-specific -march=native for VM portability" OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
  set(MILDA_DEFAULT_LLAMACPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
else()
  set(MILDA_DEFAULT_LLAMACPP_ROOT "$ENV{HOME}/llama.cpp")
endif()

set(MILDA_LLAMACPP_ROOT "${MILDA_DEFAULT_LLAMACPP_ROOT}" CACHE PATH "Path to llama.cpp checkout")
set(MILDA_LLAMACPP_LIB_DIR "$ENV{HOME}/llama.cpp/build/bin")

find_package(Threads REQUIRED)

if(MILDA_PORTABLE_CPU)
  set(MILDA_ARCH_FLAG "-march=x86-64-v3")
else()
  set(MILDA_ARCH_FLAG "-march=native")
endif()

set(MILDA_CXX_FLAGS
  -std=c++23
  -O3
  ${MILDA_ARCH_FLAG}
  -flto
  -fno-rtti
  -Wall
  -Wextra
  -Wpedantic
)

set(MILDA_LINK_FLAGS
  -O3
  ${MILDA_ARCH_FLAG}
  -flto
)

set(MILDA_USE_CUDA_BACKEND OFF)
if(MILDA_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  find_library(MILDA_LLAMA_LIB NAMES llama PATHS "${MILDA_LLAMACPP_LIB_DIR}" NO_DEFAULT_PATH)
  find_library(MILDA_GGML_LIB NAMES ggml PATHS "${MILDA_LLAMACPP_LIB_DIR}" NO_DEFAULT_PATH)
  find_library(MILDA_GGML_BASE_LIB NAMES ggml-base PATHS "${MILDA_LLAMACPP_LIB_DIR}" NO_DEFAULT_PATH)
  find_library(MILDA_GGML_CPU_LIB NAMES ggml-cpu PATHS "${MILDA_LLAMACPP_LIB_DIR}" NO_DEFAULT_PATH)
  find_library(MILDA_GGML_CUDA_LIB NAMES ggml-cuda PATHS "${MILDA_LLAMACPP_LIB_DIR}" NO_DEFAULT_PATH)

  if(MILDA_LLAMA_LIB AND MILDA_GGML_LIB AND MILDA_GGML_BASE_LIB AND MILDA_GGML_CPU_LIB AND MILDA_GGML_CUDA_LIB)
    set(MILDA_USE_CUDA_BACKEND ON)
    message(STATUS "Milda: Hybrid GPU+CPU mode enabled (CUDA backend: ${MILDA_GGML_CUDA_LIB})")
  else()
    message(STATUS "Milda: CUDA libraries not found; falling back to standalone CPU-only mode")
  endif()
else()
  message(STATUS "Milda: Pure portable CPU-only mode enabled (MILDA_ENABLE_CUDA=OFF)")
endif()

if(NOT MILDA_USE_CUDA_BACKEND)
  # The prebuilt shared libggml.so in llama.cpp/build/bin has a hard DT_NEEDED
  # dependency on libggml-cuda.so and libcudart.so. Build static CPU-only
  # llama + ggml + ggml-cpu archives so the resulting milda binary has zero
  # CUDA or external shared-library dependencies.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static llama/ggml libraries" FORCE)
  set(GGML_STATIC OFF CACHE BOOL "Keep standard dynamic glibc/libstdc++" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "Disable CUDA backend" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "Use native ggml_threadpool pthreads" FORCE)
  set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
  set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

  if(MILDA_PORTABLE_CPU)
    set(GGML_NATIVE OFF CACHE BOOL "Disable host-specific -march=native for VM portability" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_BMI2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
  endif()

  add_subdirectory("${MILDA_LLAMACPP_ROOT}" "${CMAKE_BINARY_DIR}/llamacpp-cpu" EXCLUDE_FROM_ALL)
endif()

add_library(milda_core STATIC
  src/esperanto.cpp
  src/milda_i18n.cpp
  src/hardware.cpp
  src/engine.cpp
)

target_include_directories(milda_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  SYSTEM PUBLIC
    "${MILDA_LLAMACPP_ROOT}/include"
    "${MILDA_LLAMACPP_ROOT}/ggml/include"
)

target_compile_options(milda_core PUBLIC ${MILDA_CXX_FLAGS})
target_link_options(milda_core PUBLIC ${MILDA_LINK_FLAGS})

if(MILDA_USE_CUDA_BACKEND)
  target_compile_definitions(milda_core PUBLIC MILDA_ENABLE_CUDA=1)
  target_link_libraries(milda_core
    PUBLIC
      "${MILDA_LLAMA_LIB}"
      "${MILDA_GGML_LIB}"
      "${MILDA_GGML_BASE_LIB}"
      "${MILDA_GGML_CPU_LIB}"
      "${MILDA_GGML_CUDA_LIB}"
      Threads::Threads
  )
else()
  target_link_libraries(milda_core
    PUBLIC
      llama
      ggml
      ggml-cpu
      ggml-base
      Threads::Threads
  )
endif()

add_executable(milda src/main.cpp)
target_compile_options(milda PRIVATE ${MILDA_CXX_FLAGS})
target_link_options(milda PRIVATE ${MILDA_LINK_FLAGS})
target_link_libraries(milda PRIVATE milda_core)

add_executable(test_milda tests/test_milda.cpp)
target_compile_options(test_milda PRIVATE ${MILDA_CXX_FLAGS})
target_link_options(test_milda PRIVATE ${MILDA_LINK_FLAGS})
target_link_libraries(test_milda PRIVATE milda_core)

if(MILDA_USE_CUDA_BACKEND)
  set_target_properties(milda PROPERTIES
    BUILD_RPATH "${MILDA_LLAMACPP_LIB_DIR}"
    INSTALL_RPATH "${MILDA_LLAMACPP_LIB_DIR}"
  )
  set_target_properties(test_milda PROPERTIES
    BUILD_RPATH "${MILDA_LLAMACPP_LIB_DIR}"
    INSTALL_RPATH "${MILDA_LLAMACPP_LIB_DIR}"
  )
endif()

add_test(
  NAME test_milda
  COMMAND test_milda "${CMAKE_CURRENT_SOURCE_DIR}"
)
