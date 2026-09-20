#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail closed when public materials cross private import boundaries."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from tools.generate_public_api_manifest import (
    EXPLICIT_PUBLIC_MODULES,
    public_api_record,
)


_API_DIRECTIVE = re.compile(r"^::: (phydrax(?:\.[A-Za-z0-9_]+)*)", re.MULTILINE)
_PRIVATE_IMPORT = re.compile(
    r"^(?:from|import)\s+(phydrax(?:\.[A-Za-z0-9_]+)*\._[A-Za-z0-9_.]*)",
    re.MULTILINE,
)


def _is_private_path(path: str, /) -> bool:
    return any(part.startswith("_") for part in path.split(".")[1:])


def _example_private_imports(path: Path, /) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = (node.module,)
        else:
            continue
        for module in modules:
            if module.startswith("phydrax.") and _is_private_path(module):
                errors.append(
                    f"private-example-import:{path.as_posix()}:{node.lineno}:{module}"
                )
    return tuple(errors)


def import_boundary_errors(root: Path, /) -> tuple[str, ...]:
    record = public_api_record()
    modules = record["modules"]
    if not isinstance(modules, dict):
        raise TypeError("Public API record modules must be a dictionary.")
    public_modules = frozenset(modules)
    public_paths = public_modules | frozenset(
        symbol
        for values in modules.values()
        for symbol in values
        if isinstance(symbol, str)
    )

    errors: list[str] = []
    for module in EXPLICIT_PUBLIC_MODULES:
        if module not in public_modules:
            errors.append(f"missing-explicit-public-module:{module}")

    for path in (root / "examples").rglob("*.py"):
        errors.extend(_example_private_imports(path))

    documentation_paths = (root / "README.md", *(root / "docs").rglob("*.md"))
    for path in documentation_paths:
        text = path.read_text()
        for match in _API_DIRECTIVE.finditer(text):
            symbol = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            if _is_private_path(symbol):
                errors.append(
                    f"private-api-directive:{path.relative_to(root)}:{line}:{symbol}"
                )
            elif not any(
                symbol == public or symbol.startswith(public + ".")
                for public in public_paths
            ):
                errors.append(
                    f"unknown-api-directive:{path.relative_to(root)}:{line}:{symbol}"
                )
        for match in _PRIVATE_IMPORT.finditer(text):
            module = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"private-documentation-import:{path.relative_to(root)}:{line}:{module}"
            )

    return tuple(sorted(set(errors)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    errors = import_boundary_errors(arguments.root)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
