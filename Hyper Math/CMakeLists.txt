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

cmake_minimum_required(VERSION 3.16)
project(hyper_math VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Threads REQUIRED)

add_library(hypermath STATIC src/libhypermath.cpp)
set_target_properties(hypermath PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hypermath PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(hypermath PUBLIC Threads::Threads)

add_library(hypermath_shared SHARED src/libhypermath.cpp)
set_target_properties(hypermath_shared PROPERTIES OUTPUT_NAME hypermath)
target_include_directories(hypermath_shared PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(hypermath_shared PUBLIC Threads::Threads)

add_executable(hyper-math src/main.cpp)
target_link_libraries(hyper-math PRIVATE hypermath)

add_executable(bench-hyper-math benchmarks/bench_main.cpp)
target_link_libraries(bench-hyper-math PRIVATE hypermath)

add_custom_target(bench-ci
  COMMAND "$<TARGET_FILE:bench-hyper-math>"
          "--json-out=${CMAKE_CURRENT_BINARY_DIR}/results.json"
          "--check-baseline=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/baseline.json"
  DEPENDS bench-hyper-math
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Running Hyper Math benchmark CI regression gate (<=15% threshold)"
  VERBATIM
)

if(BUILD_TESTING)
  add_executable(test-hyper-math
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_scientific.cpp
    tests/test_tabular.cpp
  )
  target_link_libraries(test-hyper-math PRIVATE hypermath)
  add_test(NAME HyperMathUnitSuite COMMAND test-hyper-math)
endif()

install(TARGETS hypermath hypermath_shared hyper-math
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/hyper_math
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES hyper-math.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

install(FILES doc/hyper-math.1
  DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
)
