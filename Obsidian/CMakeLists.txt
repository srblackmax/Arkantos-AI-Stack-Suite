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

cmake_minimum_required(VERSION 3.20)
project(Obsidian VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OBSIDIAN_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(OBSIDIAN_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(OBSIDIAN_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(OBSIDIAN_VERSION_STRING "${PROJECT_VERSION}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/obsidian/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/obsidian/version.h
  @ONLY
)

set(BASALT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Basalt Deep Learning"
    CACHE PATH "Path to Basalt Deep Learning")
set(BELLOWS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Bellows"
    CACHE PATH "Path to Bellows")
set(HYPER_MATH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../Hyper Math"
    CACHE PATH "Path to Hyper Math")

find_library(BASALT_LIB basalt
  PATHS "${BASALT_ROOT}/.libs" "${BASALT_ROOT}/lib" "${BASALT_ROOT}"
  REQUIRED)
find_library(BELLOWS_LIB bellows
  PATHS "${BELLOWS_ROOT}/lib" "${BELLOWS_ROOT}"
  REQUIRED)
find_library(HYPER_MATH_LIB hypermath
  PATHS "${HYPER_MATH_ROOT}/lib" "${HYPER_MATH_ROOT}"
  REQUIRED)

add_library(obsidian SHARED
  src/core/core.cpp
  src/core/unicode.cpp
  src/tensor/tensor_bridge.cpp
  src/tensor/safetensors.cpp
  src/tokenizer/tokenizer.cpp
  src/embeddings/embeddings.cpp
  src/classical/classical.cpp
  src/classical/topic.cpp
  src/transformer/transformer.cpp
  src/models/models.cpp
  src/pipeline/pipeline.cpp
)

target_include_directories(obsidian
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  SYSTEM PUBLIC
    "${BASALT_ROOT}/include"
    "${BELLOWS_ROOT}/include"
    "${HYPER_MATH_ROOT}/include"
)

target_compile_options(obsidian PUBLIC
  -O3 -march=native -fopenmp -Wall -Wextra -Werror -pedantic
)

target_link_libraries(obsidian PUBLIC
  ${BASALT_LIB} ${BELLOWS_LIB} ${HYPER_MATH_LIB} -fopenmp pthread dl
)

add_executable(obsidian-nlp src/cli/obsidian_nlp.cpp)
target_link_libraries(obsidian-nlp PRIVATE obsidian)

add_executable(bench_obsidian bench/bench_obsidian.cpp)
target_link_libraries(bench_obsidian PRIVATE obsidian)

add_custom_target(bench-ci
  COMMAND "$<TARGET_FILE:bench_obsidian>"
          "--json-out=${CMAKE_CURRENT_BINARY_DIR}/results.json"
          "--check-baseline=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
  DEPENDS bench_obsidian
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Running Obsidian benchmark CI regression gate (<=15% threshold)"
  VERBATIM
)

install(TARGETS obsidian obsidian-nlp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/obsidian
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/obsidian/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/obsidian
)
