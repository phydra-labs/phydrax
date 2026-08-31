#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from phydrax._fingerprint import canonical_json


def atomic_write(
    destination: str | Path,
    writer: Callable[[Path], None],
    /,
) -> Path:
    """Write through a sibling temporary path and atomically replace the destination."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def write_json_atomic(destination: str | Path, value: Any, /) -> Path:
    """Write finite, sorted, human-readable JSON with one final newline."""
    canonical_json(value)
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return atomic_write(
        destination,
        lambda temporary: temporary.write_text(payload + "\n", encoding="utf-8"),
    )


__all__ = ["atomic_write", "write_json_atomic"]
