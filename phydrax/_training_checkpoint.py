#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key


def _state_checksum(path: Path, /) -> str:
    """Return the SHA-256 checksum of one serialized state file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_state(
    directory: Path,
    serialise: Callable[[Path], None],
    /,
) -> tuple[Path, str]:
    """Serialize and atomically publish one content-addressed state file."""

    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "state.tmp.eqx"
    serialise(temporary)
    checksum = _state_checksum(temporary)
    destination = directory / f"state-{checksum[:16]}.eqx"
    os.replace(temporary, destination)
    return destination, checksum


def _verify_state(path: Path, expected_checksum: str, /) -> None:
    """Reject a serialized state whose bytes do not match its manifest."""

    if _state_checksum(path) != expected_checksum:
        raise ValueError("Training checkpoint state checksum mismatch.")


def _prune_state_files(directory: Path, current_state_name: str, /) -> None:
    """Remove unpublished and superseded serialized state files."""

    for stale_state in directory.glob("state-*.eqx"):
        if stale_state.name != current_state_name:
            stale_state.unlink(missing_ok=True)
    (directory / "state.tmp.eqx").unlink(missing_ok=True)


def _publish_manifest(path: Path, manifest: Mapping[str, Any], /) -> None:
    """Atomically publish one canonical, human-readable JSON manifest."""

    temporary = path.with_name("manifest.tmp.json")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_manifest(path: Path, /) -> Any:
    """Read one JSON manifest without imposing a lane-specific root contract."""

    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_root_key(key: Key[Array, ""], /) -> dict[str, Any]:
    """Return the canonical manifest representation of a typed JAX root key."""

    return {
        "key_data": np.asarray(jr.key_data(key)).tolist(),
        "key_impl": str(jr.key_impl(key)),
    }


def _deserialize_root_key(
    key_data: Any,
    key_impl: str,
    /,
) -> Key[Array, ""]:
    """Restore a typed JAX root key from its manifest representation."""

    return jr.wrap_key_data(jnp.asarray(key_data, dtype=jnp.uint32), impl=key_impl)


__all__: list[str] = []
