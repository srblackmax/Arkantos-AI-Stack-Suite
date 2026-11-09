#!/usr/bin/env python3
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

"""Verify GNU Coding Standards across all C and C++ source files."""

import pathlib
import sys

REQUIRED_NOTICE = (
    "/*\n"
    " * Copyright (C) 2026 Kival\n"
    " * This file is part of the Arkantos AI Stack Suite.\n"
    " *\n"
    " * This program is free software: you can redistribute it and/or modify\n"
    " * it under the terms of the GNU General Public License as published by\n"
    " * the Free Software Foundation, either version 3 of the License, or\n"
    " * (at your option) any later version.\n"
    " *\n"
    " * This program is distributed in the hope that it will be useful,\n"
    " * but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    " * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n"
    " * GNU General Public License for more details.\n"
    " *\n"
    " * You should have received a copy of the GNU General Public License\n"
    " * along with this program. If not, see <https://www.gnu.org/licenses/>.\n"
    " */"
)


def check_file(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if REQUIRED_NOTICE not in text:
        errors.append(f"{path}: missing exact GPLv3 library header")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "\t" in line:
            errors.append(f"{path}:{lineno}: tab character found")
        if line.endswith(" ") or line.endswith("\t"):
            errors.append(f"{path}:{lineno}: trailing whitespace")
        if len(line) > 79:
            errors.append(
                f"{path}:{lineno}: line length {len(line)} exceeds 79 columns"
            )
    return errors


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    dirs = ["include", "src", "tests", "examples", "bench"]
    exts = {".h", ".hpp", ".c", ".cpp", ".in"}
    all_errors: list[str] = []
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in exts:
                all_errors.extend(check_file(p))
    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
