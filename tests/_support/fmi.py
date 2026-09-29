#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real FMI2 Co-Simulation specimens compiled from original Phydrax C sources.

The specimen sources live in `tests/interchange/data`. A missing FMPy runtime,
unsupported host platform, or missing C compiler skips the caller: that is
inconclusive evidence, never a pass.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


SPECIMENS = Path(__file__).resolve().parents[1] / "interchange" / "data"


def require_fmu_toolchain() -> str:
    """Return the C compiler, or skip when a real FMU cannot be built and loaded."""
    if importlib.util.find_spec("fmpy") is None:
        pytest.skip("inconclusive: requires optional FMPy and a native C compiler")
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("inconclusive: specimen compilation supports POSIX macOS/Linux")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip(
            "inconclusive: requires a native C compiler for the real FMU specimen"
        )
    return compiler


def compile_specimen(root: Path, stem: str, /) -> Path:
    """Compile `<stem>.c` into a host shared library named by its model identifier."""
    compiler = require_fmu_toolchain()
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    library = root / (stem + extension)
    subprocess.run(
        [
            compiler,
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-fPIC",
            "-O2",
            str(SPECIMENS / f"{stem}.c"),
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
        timeout=30,
        capture_output=True,
    )
    return library


def package_fmu(
    root: Path, library: Path, description: str, name: str, /
) -> tuple[Path, str]:
    """Package one compiled library with its model description; return path and pin."""
    platform = importlib.import_module("fmpy").platform
    archive = root / name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("modelDescription.xml", description)
        output.write(library, f"binaries/{platform}/{library.name}")
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def description_only(root: Path, description: str, name: str, /) -> tuple[Path, str]:
    """Archive a model description without binaries for inspection-only checks."""
    archive = root / name
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("modelDescription.xml", description)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()
