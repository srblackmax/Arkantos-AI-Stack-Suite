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
project(stack_integration CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Yantra REQUIRED)
find_package(Sutra REQUIRED)
find_package(Darshana QUIET)  # optional

add_executable(stack_integration_test main.cpp)

target_link_libraries(stack_integration_test PRIVATE
    Yantra::Yantra
    Sutra::Sutra)

if(Darshana_FOUND)
    target_link_libraries(stack_integration_test PRIVATE
        Darshana::Darshana)
    target_compile_definitions(stack_integration_test PRIVATE
        HAVE_DARSHANA=1)
endif()

target_compile_options(stack_integration_test PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
