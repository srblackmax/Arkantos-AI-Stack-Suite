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
project(basalt VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

install(DIRECTORY include/basalt
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt.a"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt-c.a"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt.so.0.0.0"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt.so.0"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt.so"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt-c.so.0.0.0"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt-c.so.0"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/libbasalt-c.so"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(PROGRAMS
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/basalt-train"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/basalt-inspect"
  "${CMAKE_CURRENT_SOURCE_DIR}/.libs/basalt-convert"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
