#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import os
import uuid
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
    # Create the temporary like an ordinary file (0o666 masked by the process
    # umask), so the replaced artifact keeps the repository's file mode;
    # tempfile.mkstemp would leave it owner-only (0o600).
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
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
