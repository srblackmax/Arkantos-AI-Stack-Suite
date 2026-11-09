#!/usr/bin/env bash
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

MODEL_PATH="${MILDA_MODEL:-/media/${USER}/SphinxAI/models/Qwen3-Coder-Next-Q4_K_S.gguf}"
exec "${HOME}/.local/bin/milda" \
  -m "${MODEL_PATH}" \
  -ngl 99 -ncmoe -1 -t 12 -tb 12 \
  -c 16384 -n 16384 \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa \
  -s "${HOME}/.milda_qwen3coder.session" "$@" 2>/dev/null
