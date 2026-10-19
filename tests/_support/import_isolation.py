#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run importable scenarios while refusing a declared external import family."""

import argparse
import importlib
import importlib.abc
import subprocess
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest


_ROOT = Path(__file__).resolve().parents[2]


def run_without_imports(
    module: str,
    function: str,
    blocked_roots: tuple[str, ...],
    /,
    *,
    prepare: str | None = None,
    timeout: int = 600,
) -> None:
    """Run a bounded child on its natural device topology and retain diagnostics.

    The child inherits the capped parent's memory scope and runtime identity.
    Preparation resolves lazy owners before the import-refusal phase begins.
    """
    command = [sys.executable, "-m", "tests._support.import_isolation", module, function]
    if prepare is not None:
        command.extend(("--prepare", prepare))
    for name in blocked_roots:
        command.extend(("--blocked", name))
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().endswith("isolated"), completed.stdout


class _ImportRefusal(importlib.abc.MetaPathFinder):
    def __init__(self, roots: tuple[str, ...]) -> None:
        self.roots = roots

    def blocked(self, name: str) -> bool:
        return any(name == root or name.startswith(root + ".") for root in self.roots)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if self.blocked(fullname):
            raise ImportError(
                f"isolated_scenario.imports must exclude blocked roots; got {fullname}"
            )
        return None


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("function")
    parser.add_argument("--prepare")
    parser.add_argument("--blocked", action="append", default=[])
    arguments = parser.parse_args()
    pytest.register_assert_rewrite(arguments.module)
    module = importlib.import_module(arguments.module)
    if arguments.prepare is not None:
        vars(module)[arguments.prepare]()
    refusal = _ImportRefusal(tuple(arguments.blocked))
    for name in tuple(sys.modules):
        if refusal.blocked(name):
            del sys.modules[name]
    sys.meta_path.insert(0, refusal)
    try:
        vars(module)[arguments.function]()
        assert not [name for name in sys.modules if refusal.blocked(name)]
    finally:
        sys.meta_path.remove(refusal)
    print("isolated")


if __name__ == "__main__":
    _main()
