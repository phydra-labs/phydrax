#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Isolated child processes for Metal contracts that need a fresh JAX runtime.

A child imports the same ``phydrax`` as the test process (source tree or the
installed wheel under qualification) and the same ``tests`` package. Each
case states its startup environment explicitly, so no platform switch happens
after a JAX backend exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final


_TESTS_PARENT: Final = Path(__file__).resolve().parents[2]
# Startup selections inherited from the parent must never leak into a child.
_RUNTIME_ENVIRONMENT: Final = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS")


def _import_roots(extra: tuple[Path, ...]) -> str:
    import phydrax

    roots: list[str] = []
    for root in (*extra, Path(phydrax.__file__).resolve().parents[1], _TESTS_PARENT):
        text = str(root)
        if text not in roots:
            roots.append(text)
    return os.pathsep.join(roots)


def run_python(
    source: str,
    *arguments: str,
    environment: Mapping[str, str],
    import_roots: tuple[Path, ...] = (),
    stdin: str | None = None,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter with exactly ``environment`` selected."""
    child = {
        key: value for key, value in os.environ.items() if key not in _RUNTIME_ENVIRONMENT
    }
    child.update(environment)
    child["PYTHONPATH"] = _import_roots(import_roots)
    return subprocess.run(
        [sys.executable, "-c", source, *arguments],
        cwd=_TESTS_PARENT,
        env=child,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def json_result(completed: subprocess.CompletedProcess[str], /) -> dict[str, object]:
    """Require a successful child and decode its final JSON line."""
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    lines = completed.stdout.strip().splitlines()
    assert lines, output
    result = json.loads(lines[-1])
    assert isinstance(result, dict), output
    return result


__all__ = ["json_result", "run_python"]
