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

"""Automated GNU Coding Standards, license, and naming verifier for Bellows."""

import os
import re
import sys


REQUIRED_FILES = [
    "AGENTS.md",
    "COPYING",
    "README",
    "NEWS",
    "ChangeLog",
    "AUTHORS",
    "INSTALL",
    "Makefile",
    "CMakeLists.txt",
    "bellows.pc.in",
    "doc/design.md",
]

REQUIRED_DISCLAIMER = (
    "Bellows is an independent GPLv3 C++ machine-learning library. It "
    "follows the GNU Coding Standards as a style convention; it is not part "
    "of the GNU Project and is not affiliated with the Free Software "
    "Foundation."
)

FORBIDDEN_IDENTIFIERS = [
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "TransformerMixin",
    "fit_transform",
    "predict_proba",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "LabelEncoder",
    "SimpleImputer",
    "GridSearchCV",
    "DMatrix",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
]


def error(path, line_no, msg):
    if line_no > 0:
        sys.stderr.write(f"check_style:{path}:{line_no}: {msg}\n")
    else:
        sys.stderr.write(f"check_style:{path}: {msg}\n")


def check_readme(root):
    readme_path = os.path.join(root, "README")
    if not os.path.isfile(readme_path):
        error("README", 0, "missing README file")
        return 1
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    if REQUIRED_DISCLAIMER not in paragraphs[:4]:
        error(
            "README",
            1,
            "missing required verbatim disclaimer paragraph near the top",
        )
        return 1
    return 0


def check_cpp_file(rel_path, full_path):
    failures = 0
    base = os.path.basename(rel_path)
    if not re.match(r"^[a-z0-9_]+\.(hpp|cpp)$", base):
        error(
            rel_path,
            0,
            f"filename '{base}' must use lower_case_with_underscores",
        )
        failures += 1

    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_block = "".join(lines[:25])
    if "Copyright (C) 2026 Kival" not in header_block:
        error(
            rel_path,
            1,
            "missing 'Copyright (C) 2026 Kival' header",
        )
        failures += 1
    if "GNU General Public License" not in header_block:
        error(rel_path, 1, "missing GPLv3 license header notice")
        failures += 1

    for idx, line in enumerate(lines, start=1):
        if "\t" in line:
            error(rel_path, idx, "tab character found (use 2-space indent)")
            failures += 1
        if "gnu-" in line.lower():
            error(rel_path, idx, "forbidden 'gnu-' prefix found")
            failures += 1
        if (
            "copyright" in line.lower()
            and (
                "free software foundation" in line.lower()
                or "gnu project" in line.lower()
            )
        ):
            error(
                rel_path,
                idx,
                "copyright line must not reference FSF or GNU Project",
            )
            failures += 1
        for forbidden in FORBIDDEN_IDENTIFIERS:
            if re.search(r"\b" + re.escape(forbidden) + r"\b", line):
                error(
                    rel_path,
                    idx,
                    f"forbidden transliterated identifier '{forbidden}'",
                )
                failures += 1

        stripped = line.lstrip(" ")
        if stripped and not stripped.startswith("*") and not stripped.startswith("//"):
            leading_spaces = len(line) - len(stripped)
            if leading_spaces % 2 != 0:
                error(
                    rel_path,
                    idx,
                    f"indentation ({leading_spaces} spaces) is not a multiple of 2",
                )
                failures += 1

    return failures


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    total_failures = 0

    for req in REQUIRED_FILES:
        p = os.path.join(root, req)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            error(req, 0, "required repository file is missing or empty")
            total_failures += 1

    total_failures += check_readme(root)

    scan_dirs = ["include", "src", "tests", "examples"]
    cpp_count = 0
    for d in scan_dirs:
        dir_path = os.path.join(root, d)
        if not os.path.isdir(dir_path):
            continue
        for dirpath, _, filenames in os.walk(dir_path):
            for fname in sorted(filenames):
                if fname.endswith((".hpp", ".cpp")):
                    cpp_count += 1
                    full_p = os.path.join(dirpath, fname)
                    rel_p = os.path.relpath(full_p, root)
                    total_failures += check_cpp_file(rel_p, full_p)

    if cpp_count == 0:
        error("include", 0, "no C++ source or header files found")
        total_failures += 1

    if total_failures > 0:
        sys.stderr.write(
            f"check_style: {total_failures} style/compliance error(s) found\n"
        )
        return 1

    print(
        f"check_style: verified {cpp_count} C++ files and {len(REQUIRED_FILES)} "
        "repository artefacts (0 issues)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
