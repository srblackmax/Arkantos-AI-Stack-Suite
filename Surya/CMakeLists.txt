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
project(Surya VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
enable_testing()

option(SURYA_BUILD_TESTS "Build Surya unit test suite and style checker" ON)
option(SURYA_BUILD_BENCH "Build Surya microbenchmark suite" ON)
option(SURYA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(SURYA_USE_TSAN "Enable ThreadSanitizer" OFF)
option(SURYA_BUILD_FUZZ "Build Surya fuzzing harnesses" ON)
option(SURYA_ENABLE_CUDA "Enable CUDA 12.x GPU hybrid backend" ON)
option(SURYA_BUILD_FAT_NATIVE "Build with -march=native instead of portable -march=x86-64-v3" OFF)

if(SURYA_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-13")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(TBB REQUIRED)

set(BASALT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Basalt Deep Learning"
    CACHE PATH "Path to Basalt Deep Learning")
set(BELLOWS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Bellows"
    CACHE PATH "Path to Bellows")
set(HYPER_MATH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Hyper Math"
    CACHE PATH "Path to Hyper Math")
set(OBSIDIAN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Obsidian"
    CACHE PATH "Path to Obsidian")

find_library(BASALT_LIB basalt
  PATHS "${BASALT_ROOT}/.libs" "${BASALT_ROOT}/lib" "${BASALT_ROOT}"
  REQUIRED)
find_library(BELLOWS_LIB bellows
  PATHS "${BELLOWS_ROOT}/lib" "${BELLOWS_ROOT}"
  REQUIRED)
find_library(HYPER_MATH_LIB hypermath
  PATHS "${HYPER_MATH_ROOT}/lib" "${HYPER_MATH_ROOT}"
  REQUIRED)
find_library(OBSIDIAN_LIB obsidian
  PATHS "${OBSIDIAN_ROOT}/.libs" "${OBSIDIAN_ROOT}/lib" "${OBSIDIAN_ROOT}"
  REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JPEG REQUIRED IMPORTED_TARGET libjpeg)
pkg_check_modules(PNG REQUIRED IMPORTED_TARGET libpng)
pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
pkg_check_modules(TIFF REQUIRED IMPORTED_TARGET libtiff-4)
pkg_check_modules(ZLIB REQUIRED IMPORTED_TARGET zlib)

set(SURYA_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)
if(SURYA_BUILD_FAT_NATIVE)
  set(SURYA_ARCH_FLAGS -march=native)
else()
  set(SURYA_ARCH_FLAGS -march=x86-64-v3)
endif()
set(SURYA_PERF_FLAGS -O3 ${SURYA_ARCH_FLAGS})

set(SURYA_SANITIZER_FLAGS "")
if(SURYA_USE_SANITIZERS)
  set(SURYA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
elseif(SURYA_USE_TSAN)
  set(SURYA_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer -g)
endif()

add_library(surya_cuda_kernels OBJECT
  src/gpu/kernels.cu
)
set_target_properties(surya_cuda_kernels PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(surya_cuda_kernels PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu"
)

add_library(surya SHARED
  src/core/memory.cpp
  src/core/observability.cpp
  src/core/simd.cpp
  src/core/parallel.cpp
  src/core/image.cpp
  src/imgproc/color.cpp
  src/imgproc/filter.cpp
  src/imgproc/geometry.cpp
  src/imgproc/morphology.cpp
  src/imgproc/analysis.cpp
  src/io/codecs.cpp
  src/io/video.cpp
  src/augment/pipeline.cpp
  src/nn/tensor_bridge.cpp
  src/nn/vision_blocks.cpp
  src/nn/runtime.cpp
  src/detection/detection.cpp
  src/classical/classical.cpp
  src/gpu/pipeline.cpp
  $<TARGET_OBJECTS:surya_cuda_kernels>
)
add_library(Surya::surya ALIAS surya)

target_compile_features(surya PUBLIC cxx_std_23)
set_target_properties(surya PROPERTIES
  CXX_EXTENSIONS OFF
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

target_include_directories(surya
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(surya SYSTEM
  PRIVATE
    "${BASALT_ROOT}/include"
    "${BELLOWS_ROOT}/include"
    "${HYPER_MATH_ROOT}/include"
    "${OBSIDIAN_ROOT}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(surya
  PUBLIC
    ${SURYA_WARNING_FLAGS}
    ${SURYA_SANITIZER_FLAGS}
  PRIVATE
    ${SURYA_PERF_FLAGS}
)

target_link_options(surya
  PUBLIC
    ${SURYA_SANITIZER_FLAGS}
)

target_link_libraries(surya
  PRIVATE
    ${OBSIDIAN_LIB}
    ${BASALT_LIB}
    ${BELLOWS_LIB}
    ${HYPER_MATH_LIB}
    TBB::tbb
    CUDA::cudart
    PkgConfig::JPEG
    PkgConfig::PNG
    PkgConfig::WEBP
    PkgConfig::TIFF
    PkgConfig::ZLIB
    pthread
    dl
    gomp
)

function(surya_register_test target_name src_file)
  add_executable(${target_name} ${src_file})
  target_compile_features(${target_name} PRIVATE cxx_std_23)
  target_link_libraries(${target_name} PRIVATE Surya::surya)
  add_test(NAME ${target_name} COMMAND ${target_name})
  if(SURYA_USE_SANITIZERS)
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0")
  elseif(SURYA_USE_TSAN)
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "TSAN_OPTIONS=suppressions=../cmake/tsan.supp")
  endif()
endfunction()

if(SURYA_BUILD_TESTS)
  add_executable(check_gnu_style tests/check_gnu_style.cpp)
  target_compile_features(check_gnu_style PRIVATE cxx_std_23)
  target_compile_options(check_gnu_style PRIVATE ${SURYA_WARNING_FLAGS} ${SURYA_SANITIZER_FLAGS})
  target_link_options(check_gnu_style PRIVATE ${SURYA_SANITIZER_FLAGS})
  add_test(NAME check_gnu_style
           COMMAND check_gnu_style "${CMAKE_CURRENT_SOURCE_DIR}")

  surya_register_test(test_core tests/test_core.cpp)
  surya_register_test(test_imgproc tests/test_imgproc.cpp)
  surya_register_test(test_io tests/test_io.cpp)
  surya_register_test(fuzz_surya_codecs tests/fuzz_codecs.cpp)
  surya_register_test(test_augment tests/test_augment.cpp)
  surya_register_test(fuzz_surya_augment tests/fuzz_augment.cpp)
  surya_register_test(test_nn tests/test_nn.cpp)
  surya_register_test(test_detection tests/test_detection.cpp)
  surya_register_test(test_classical tests/test_classical.cpp)
  surya_register_test(test_gpu_hybrid tests/test_gpu_hybrid.cpp)
endif()

if(SURYA_BUILD_BENCH)
  add_executable(bench_surya bench/bench_surya.cpp)
  target_compile_features(bench_surya PRIVATE cxx_std_23)
  target_link_libraries(bench_surya PRIVATE Surya::surya)

  add_custom_target(bench-ci
    COMMAND "$<TARGET_FILE:bench_surya>"
            "--json-out=${CMAKE_CURRENT_BINARY_DIR}/results.json"
            "--check-baseline=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
    DEPENDS bench_surya
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Running Surya benchmark CI regression gate (<=15% threshold)"
    VERBATIM
  )
endif()

install(TARGETS surya
  EXPORT SuryaTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/surya
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SuryaConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/SuryaConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Surya"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SuryaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT SuryaTargets
  FILE SuryaTargets.cmake
  NAMESPACE Surya::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Surya"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/SuryaConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/SuryaConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Surya"
)
