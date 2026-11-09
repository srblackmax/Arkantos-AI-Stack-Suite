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

EXPECTED_VERSION="Basalt Deep Learning 0.1.0
Copyright (C) 2026 Kival
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law."

for tool in ./basalt-train ./basalt-inspect ./basalt-convert; do
  ver_out="$("$tool" --version)"
  if [[ "$ver_out" != "$EXPECTED_VERSION" ]]; then
    echo "Version output mismatch for $tool:" >&2
    echo "$ver_out" >&2
    exit 1
  fi
  "$tool" --help >/dev/null
done

# Test dry-run and POSIXLY_CORRECT
POSIXLY_CORRECT=1 ./basalt-train --dry-run --epochs=5 --quiet
./basalt-train --epochs=4 --batch-size=16 --output=cli_test.bckpt --quiet

# Test basalt-inspect and basalt-convert across formats
./basalt-inspect cli_test.bckpt | grep -q "Basalt Native Checkpoint"
./basalt-convert --input=cli_test.bckpt --output=cli_test.safetensors
./basalt-inspect -- cli_test.safetensors | grep -q "Safetensors"
./basalt-convert -i cli_test.safetensors -o cli_test.onnx --verbose >/dev/null
./basalt-inspect cli_test.onnx | grep -q "ONNX Model"

# Verify diagnostics format "<tool>: message" on stderr
err_msg="$(./basalt-inspect --invalid-flag 2>&1 || true)"
if [[ "$err_msg" != basalt-inspect:* ]]; then
  echo "Expected 'basalt-inspect: ...' diagnostic, got: $err_msg" >&2
  exit 1
fi

rm -f cli_test.bckpt cli_test.safetensors cli_test.onnx
exit 0
