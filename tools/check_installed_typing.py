#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Check the static typing contract of an installed Phydrax wheel.

The check builds one wheel, requires the PEP 561 `py.typed` marker in it, installs
it with its runtime dependencies into a fresh environment for every requested
Python version, and runs the pinned ty over `tests/typing/installed` from a
directory outside the repository, so every `phydrax` import resolves to the
installed distribution rather than to the source tree. Fixture expectations use
`# ty: ignore[<rule>]` with unused suppressions promoted to errors, and
`assert_type` pins inferred types.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

from tools.check_typing import run_tool, verify_pinned


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "typing" / "installed"
DEFAULT_PYTHONS = ("3.12",)
MARKER = "phydrax/py.typed"


def _run(command: Sequence[str], /, *, cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"{' '.join(command)} failed with exit code {result.returncode}:\n"
            f"{result.stdout}{result.stderr}"
        )


def build_wheel(output: Path, /) -> Path:
    """Build one wheel from the repository and require its `py.typed` marker."""
    _run(("uv", "build", "--wheel", "--out-dir", str(output)), cwd=ROOT)
    (wheel,) = output.glob("phydrax-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        if MARKER not in archive.namelist():
            raise SystemExit(f"{wheel.name} does not contain {MARKER}.")
    return wheel


def check_installed(wheel: Path, python: str, workspace: Path, /) -> None:
    """Install `wheel` for one Python version and type-check the fixtures there."""
    environment = workspace / f"venv-{python}"
    _run(("uv", "venv", "--python", python, str(environment)), cwd=workspace)
    interpreter = environment / "bin" / "python"
    _run(
        ("uv", "pip", "install", "--python", str(interpreter), str(wheel)),
        cwd=workspace,
    )
    consumer = workspace / f"consumer-{python}"
    consumer.mkdir()
    fixtures = sorted(FIXTURES.glob("*.py"))
    if not fixtures:
        raise SystemExit(f"No installed-typing fixtures in {FIXTURES}.")
    for fixture in fixtures:
        shutil.copy2(fixture, consumer / fixture.name)
    result = run_tool(
        consumer,
        "ty",
        (
            "check",
            "--output-format",
            "concise",
            "--error",
            "unused-ignore-comment",
            "--python",
            str(environment),
            "--python-version",
            python,
            *(fixture.name for fixture in fixtures),
        ),
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Installed typing contract failed on Python {python}:\n"
            f"{result.stdout}{result.stderr}"
        )
    print(f"python {python}: {len(fixtures)} fixture file(s) passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", action="append", dest="pythons")
    arguments = parser.parse_args()
    verify_pinned(ROOT, "ty")
    with tempfile.TemporaryDirectory(prefix="phydrax-installed-typing-") as scratch:
        workspace = Path(scratch)
        wheel = build_wheel(workspace / "dist")
        print(f"built {wheel.name} with {MARKER}")
        for python in arguments.pythons or DEFAULT_PYTHONS:
            check_installed(wheel, python, workspace)


if __name__ == "__main__":
    main()
