#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from ._external_resource import read_bounded_resource, ResourceLimits
from ._host_io import open_regular_file
from ._publication import publish_bytes, publish_file


def _state_checksum(path: Path, /) -> str:
    """Return the SHA-256 checksum of one serialized state file."""

    digest = hashlib.sha256()
    with open_regular_file(path) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_state(
    directory: Path,
    serialize: Callable[[BinaryIO], None],
    /,
) -> tuple[Path, str]:
    """Serialize and durably publish one content-addressed state file."""

    with SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as staged:
        serialize(staged)
        staged.flush()
        staged.seek(0)
        digest = hashlib.sha256()
        size = 0
        while block := staged.read(1024 * 1024):
            size += len(block)
            digest.update(block)
        checksum = digest.hexdigest()
        destination = directory / f"state-{checksum[:16]}.eqx"

        def writer(stream: BinaryIO) -> None:
            staged.seek(0)
            while block := staged.read(1024 * 1024):
                stream.write(block)

        receipt = publish_file(
            destination,
            writer,
            maximum_bytes=16 * 1024 * 1024 * 1024,
            mode="atomic_replace",
        )
    if receipt.size_bytes != size or receipt.content_sha256 != checksum:
        raise RuntimeError("Published training state identity changed.")
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

    payload = (
        json.dumps(dict(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_bytes(
        path,
        payload,
        maximum_bytes=16 * 1024 * 1024,
        mode="atomic_replace",
    )


def _read_manifest(path: Path, /) -> Any:
    """Read one bounded JSON manifest without a lane-specific root contract."""

    resource = read_bounded_resource(
        path.name,
        trusted_root=path.parent,
        limits=ResourceLimits(16 * 1024 * 1024, 64, 100_000, 100_000, 0),
    )
    try:
        return json.loads(
            resource.data,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("Training checkpoint manifest is invalid JSON.") from error


def _unique_json_object(pairs: list[tuple[str, Any]], /) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"Duplicate training manifest member {name!r}.")
        value[name] = item
    return value


def _reject_json_constant(value: str, /) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")


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
