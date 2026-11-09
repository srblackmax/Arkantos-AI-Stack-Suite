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

set -euo pipefail

MODEL_DIR="${1:-$HOME/models}"
MODEL_FILE="$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already present: $MODEL_FILE"
    exit 0
fi

echo "Downloading $MODEL_URL to $MODEL_FILE..."
curl -L --fail --progress-bar -o "$MODEL_FILE.tmp" "$MODEL_URL"
mv "$MODEL_FILE.tmp" "$MODEL_FILE"
echo "Downloaded: $(du -h "$MODEL_FILE" | cut -f1)"
