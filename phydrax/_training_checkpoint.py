#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from ._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from ._document_resource import decode_json_resource
from ._external_resource import read_bounded_resource, ResourceLimits
from ._host_io import open_regular_beneath
from ._publication import publish_bytes, publish_file


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
            maximum_bytes=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_aggregate_bytes,
            mode="atomic_replace",
        )
    if receipt.size_bytes != size or receipt.content_sha256 != checksum:
        raise RuntimeError("Published training state identity changed.")
    return destination, checksum


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
    """Read one byte-, depth-, node-, and duplicate-bounded JSON manifest."""

    resource = read_bounded_resource(
        path.name,
        trusted_root=path.parent,
        limits=ResourceLimits(16 * 1024 * 1024, 64, 100_000, 100_000, 0),
    )
    try:
        return decode_json_resource(resource).value
    except ValueError as error:
        raise ValueError("Training checkpoint manifest is invalid JSON.") from error


@contextmanager
def _open_verified_state(
    directory: Path,
    state_name: str,
    expected_checksum: str,
    /,
) -> Iterator[BinaryIO]:
    """Hold and verify one canonical state file beneath its checkpoint root."""

    if (
        not isinstance(state_name, str)
        or not state_name
        or "\\" in state_name
        or Path(state_name).name != state_name
        or not state_name.startswith("state-")
        or not state_name.endswith(".eqx")
    ):
        raise ValueError("Training checkpoint state_file must be a canonical basename.")
    if (
        not isinstance(expected_checksum, str)
        or len(expected_checksum) != 64
        or any(character not in "0123456789abcdef" for character in expected_checksum)
    ):
        raise ValueError("Training checkpoint state checksum is invalid.")
    if state_name != f"state-{expected_checksum[:16]}.eqx":
        raise ValueError(
            "Training checkpoint state_file does not match its content identity."
        )
    try:
        with open_regular_beneath(
            state_name,
            trusted_root=directory,
            maximum_depth=1,
        ) as opened:
            if (
                opened.file_status.st_size
                > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_aggregate_bytes
            ):
                raise ValueError("Training checkpoint state exceeds its byte limit.")
            with os.fdopen(os.dup(opened.descriptor), "rb") as stream:
                digest = hashlib.sha256()
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                if digest.hexdigest() != expected_checksum:
                    raise ValueError("Training checkpoint state checksum mismatch.")
                stream.seek(0)
                yield stream
                opened.verify_stable()
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "Training checkpoint state path is unsafe or changed during reading."
        ) from error


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
    """Restore one strictly validated scalar typed JAX root key."""

    words = {"threefry2x32": 2, "rbg": 4, "unsafe_rbg": 4}
    if (
        not isinstance(key_impl, str)
        or key_impl not in words
        or not isinstance(key_data, list)
        or len(key_data) != words[key_impl]
        or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in key_data)
    ):
        raise ValueError("Training checkpoint root key is invalid.")
    return jr.wrap_key_data(jnp.asarray(key_data, dtype=jnp.uint32), impl=key_impl)


__all__: list[str] = []
