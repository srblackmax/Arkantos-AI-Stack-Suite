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

cmake_minimum_required (VERSION 3.20)
project (bellows VERSION 0.1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED HYPER_MATH_DIR)
  set (HYPER_MATH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Hyper Math")
endif ()

include (GNUInstallDirs)

add_compile_options (-Wall -Wextra -Wpedantic -Wno-unused-parameter)

include_directories (
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  "${HYPER_MATH_DIR}/include"
)
link_directories ("${HYPER_MATH_DIR}/lib")
set (CMAKE_BUILD_RPATH "${HYPER_MATH_DIR}/lib")
set (CMAKE_INSTALL_RPATH "${HYPER_MATH_DIR}/lib")

add_library (bellows_static STATIC src/libbellows.cpp)
set_target_properties (bellows_static PROPERTIES OUTPUT_NAME bellows)
target_link_libraries (bellows_static PUBLIC hypermath pthread)

add_library (bellows_shared SHARED src/libbellows.cpp)
set_target_properties (bellows_shared PROPERTIES OUTPUT_NAME bellows)
target_link_libraries (bellows_shared PUBLIC hypermath pthread)

add_executable (bellows-cli src/bellows_main.cpp)
set_target_properties (bellows-cli PROPERTIES OUTPUT_NAME bellows)
target_link_libraries (bellows-cli PRIVATE bellows_static)

add_executable (bellows-train src/bellows_train_main.cpp)
target_link_libraries (bellows-train PRIVATE bellows_static)

add_executable (bellows-infer src/bellows_infer_main.cpp)
target_link_libraries (bellows-infer PRIVATE bellows_static)

add_executable (bellows-check
  tests/check_main.cpp
  tests/test_core.cpp
  tests/test_preprocess.cpp
  tests/test_estimators.cpp
  tests/test_boosting.cpp
  tests/test_composition.cpp
  tests/test_io_and_cli.cpp
)
target_link_libraries (bellows-check PRIVATE bellows_static)

add_executable (workflow_demo examples/workflow_demo.cpp)
target_link_libraries (workflow_demo PRIVATE bellows_static)

add_executable (bench-bellows bench/bench_bellows.cpp)
target_link_libraries (bench-bellows PRIVATE bellows_static)

add_custom_target (bench-ci
  COMMAND "$<TARGET_FILE:bench-bellows>"
          "--json-out=${CMAKE_CURRENT_BINARY_DIR}/results.json"
          "--check-baseline=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
  DEPENDS bench-bellows
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Running Bellows benchmark CI regression gate (<=15% threshold)"
  VERBATIM
)

set (PREFIX "${CMAKE_INSTALL_PREFIX}")
set (VERSION "${PROJECT_VERSION}")
configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/bellows.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/bellows.pc
  @ONLY
)

enable_testing ()
add_test (NAME bellows-check COMMAND bellows-check)

add_custom_target (check
  COMMAND bellows-check
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_style.py
  DEPENDS bellows-check bellows-cli bellows-train bellows-infer workflow_demo
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install (TARGETS bellows_static bellows_shared bellows-cli bellows-train bellows-infer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install (DIRECTORY include/bellows DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/bellows.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
