#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Location of the packaged Phydrax meshcore shared library."""

from __future__ import annotations

import sys
from pathlib import Path


def _library_name() -> str:
    match sys.platform:
        case "darwin":
            return "libphydrax_meshcore.dylib"
        case "win32":
            return "phydrax_meshcore.dll"
        case _:
            return "libphydrax_meshcore.so"


def library_path() -> Path:
    """Return the absolute path of the installed ``phydrax_meshcore`` library."""

    path = Path(__file__).resolve().parent / _library_name()
    if not path.is_file():
        raise FileNotFoundError(f"phydrax_meshcore library is missing at {path}.")
    return path


__all__ = ["library_path"]
