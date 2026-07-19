#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from ._posterior import PosteriorProblem


_CHECKPOINT_FORMAT = "phydrax-uq-checkpoint"
_CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for checkpoint read, write, and validation failures."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint archive is incomplete or corrupt."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint cannot resume the requested run."""


def write_checkpoint_archive(
    path: str | os.PathLike[str],
    /,
    *,
    kind: str,
    compatibility: Mapping[str, Any],
    state: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> Path:
    """Atomically write one versioned checkpoint without pickle payloads."""
    manifest = {
        "format": _CHECKPOINT_FORMAT,
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "kind": str(kind),
        "versions": _runtime_versions(),
        "compatibility": dict(compatibility),
        "state": dict(state),
    }
    return _write_array_archive(path, manifest=manifest, arrays=arrays)


def read_checkpoint_archive(
    path: str | os.PathLike[str],
    /,
    *,
    kind: str,
    compatibility: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    """Read and validate one checkpoint against a live problem and run."""
    manifest, arrays = _read_array_archive(path)
    if manifest.get("format") != _CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError("File is not a PhydraX UQ checkpoint.")
    if manifest.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "Checkpoint schema version is incompatible with this PhydraX release."
        )
    if manifest.get("kind") != kind:
        raise CheckpointCompatibilityError(
            f"Checkpoint kind {manifest.get('kind')!r} does not match {kind!r}."
        )
    if manifest.get("versions") != _runtime_versions():
        raise CheckpointCompatibilityError(
            "Checkpoint PhydraX or BlackJAX version does not match the runtime."
        )
    actual = manifest.get("compatibility")
    if actual != dict(compatibility):
        raise CheckpointCompatibilityError(
            _compatibility_difference(dict(compatibility), actual)
        )
    state = manifest.get("state")
    if not isinstance(state, dict):
        raise CheckpointCorruptionError("Checkpoint state manifest must be an object.")
    return state, {name: jnp.asarray(value) for name, value in arrays.items()}


def checkpoint_compatibility(
    problem: PosteriorProblem,
    /,
    *,
    checkpoint_id: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic compatibility contract for one posterior run."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    identifier = str(checkpoint_id)
    if not identifier:
        raise ValueError("checkpoint_id must be a non-empty string.")
    normalized_settings = _json_value(dict(settings), path="settings")
    initial = problem.initial_position
    value, gradient = jax.value_and_grad(problem.log_density)(initial)
    jax.block_until_ready(value)
    dynamic_records = _array_records(problem)
    probe_records = _array_records({"value": value, "gradient": gradient})
    return {
        "checkpoint_id": identifier,
        "problem_type": _qualified_type(problem),
        "parameter_tree": tree_signature(initial),
        "problem_array_digest": _records_digest(dynamic_records),
        "initial_probe_digest": _records_digest(probe_records),
        "settings": normalized_settings,
    }


def pack_array_tree(
    prefix: str,
    tree: PyTree[Any],
    arrays: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Store array PyTree leaves and return a reconstruction specification."""
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    if not path_leaves:
        raise ValueError(f"Array tree {prefix!r} must contain at least one leaf.")
    paths: list[str] = []
    names: list[str] = []
    for index, (path, leaf) in enumerate(path_leaves):
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise TypeError(f"Array tree {prefix!r} cannot contain object arrays.")
        name = f"{prefix}/{index:06d}"
        paths.append(jax.tree_util.keystr(path) or "<root>")
        names.append(name)
        arrays[name] = array
    return {
        "paths": paths,
        "arrays": names,
        "num_leaves": len(names),
        "treedef": str(treedef),
    }


def unpack_array_tree(
    specification: Mapping[str, Any],
    arrays: Mapping[str, Any],
    template: PyTree[Any],
    /,
) -> PyTree[jax.Array]:
    """Reconstruct an array PyTree against a live runtime template."""
    template_path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    expected_paths = [
        jax.tree_util.keystr(path) or "<root>" for path, _ in template_path_leaves
    ]
    paths = specification.get("paths")
    names = specification.get("arrays")
    if (
        paths != expected_paths
        or not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
    ):
        raise CheckpointCompatibilityError(
            "Checkpoint array tree does not match the runtime template."
        )
    array_names = [str(name) for name in names]
    if len(array_names) != len(template_path_leaves):
        raise CheckpointCorruptionError("Checkpoint array tree leaf count is invalid.")
    leaves = []
    for name, (_, template_leaf) in zip(array_names, template_path_leaves, strict=True):
        if name not in arrays:
            raise CheckpointCorruptionError(f"Checkpoint array {name!r} is missing.")
        value = jnp.asarray(arrays[name])
        expected = jnp.asarray(template_leaf)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise CheckpointCompatibilityError(
                f"Checkpoint array {name!r} has shape or dtype incompatible with "
                "the runtime template."
            )
        leaves.append(value)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def tree_signature(tree: PyTree[Any], /) -> list[dict[str, Any]]:
    """Return stable array-path, shape, and dtype records for a PyTree."""
    return [
        {
            "path": jax.tree_util.keystr(path) or "<root>",
            "shape": list(np.asarray(leaf).shape),
            "dtype": np.asarray(leaf).dtype.str,
        }
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    ]


def array_tree_fingerprint(tree: PyTree[Any], /) -> dict[str, Any]:
    """Return a content-sensitive, JSON-compatible fingerprint for an array tree."""
    records = _array_records(tree)
    return {
        "signature": tree_signature(tree),
        "sha256": _records_digest(records),
    }


def _write_array_archive(
    path: str | os.PathLike[str],
    /,
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for index, name in enumerate(sorted(arrays)):
        if not isinstance(name, str) or not name:
            raise TypeError("Archive array names must be non-empty strings.")
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"Archive array {name!r} cannot have object dtype.")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        member = f"arrays/{index:06d}.npy"
        payloads[member] = payload
        inventory[name] = {
            "member": member,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    complete_manifest = dict(manifest)
    complete_manifest["arrays"] = inventory
    try:
        manifest_payload = json.dumps(
            complete_manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("Archive manifest must contain finite JSON values.") from error

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=False,
        ) as archive:
            archive.writestr("manifest.json", manifest_payload)
            for member in sorted(payloads):
                archive.writestr(member, payloads[member])
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_array_archive(
    path: str | os.PathLike[str],
    /,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            if archive.testzip() is not None:
                raise CheckpointCorruptionError("Archive CRC validation failed.")
            names = set(archive.namelist())
            if "manifest.json" not in names:
                raise CheckpointCorruptionError("Archive manifest is missing.")
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CheckpointCorruptionError(
                    "Archive manifest is invalid JSON."
                ) from error
            if not isinstance(manifest, dict):
                raise CheckpointCorruptionError("Archive manifest must be an object.")
            inventory = manifest.get("arrays")
            if not isinstance(inventory, dict):
                raise CheckpointCorruptionError("Archive array inventory is missing.")
            expected_members = {"manifest.json"}
            values: dict[str, np.ndarray] = {}
            for logical_name, record in inventory.items():
                if not isinstance(logical_name, str) or not isinstance(record, dict):
                    raise CheckpointCorruptionError("Archive array inventory is invalid.")
                member = record.get("member")
                if not isinstance(member, str) or member not in names:
                    raise CheckpointCorruptionError(
                        f"Archive member for array {logical_name!r} is missing."
                    )
                expected_members.add(member)
                payload = archive.read(member)
                if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                    raise CheckpointCorruptionError(
                        f"Archive array {logical_name!r} checksum failed."
                    )
                try:
                    value = np.load(io.BytesIO(payload), allow_pickle=False)
                except (OSError, ValueError) as error:
                    raise CheckpointCorruptionError(
                        f"Archive array {logical_name!r} is invalid."
                    ) from error
                if list(value.shape) != record.get(
                    "shape"
                ) or value.dtype.str != record.get("dtype"):
                    raise CheckpointCorruptionError(
                        f"Archive array {logical_name!r} metadata is inconsistent."
                    )
                values[logical_name] = value
            if names != expected_members:
                raise CheckpointCorruptionError("Archive contains unexpected members.")
            return manifest, values
    except CheckpointError:
        raise
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, OSError) as error:
        raise CheckpointCorruptionError(
            f"Cannot read checkpoint archive {source}."
        ) from error


def _array_records(tree: PyTree[Any]) -> list[tuple[str, np.ndarray]]:
    records = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        try:
            value = np.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if value.dtype.hasobject:
            continue
        records.append((jax.tree_util.keystr(path) or "<root>", value))
    return records


def _records_digest(records: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for path, value in records:
        contiguous = np.ascontiguousarray(value)
        digest.update(path.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "phydrax": importlib.metadata.version("phydrax"),
        "blackjax": importlib.metadata.version("blackjax"),
    }


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_value(value: Any, /, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} must contain only finite values.")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return _json_value(array.item(), path=path)
    raise TypeError(f"{path} contains a non-scalar value that is not JSON serializable.")


def _compatibility_difference(expected: dict[str, Any], actual: Any) -> str:
    if not isinstance(actual, dict):
        return "Checkpoint compatibility manifest is missing or invalid."
    for key in expected:
        if actual.get(key) != expected[key]:
            return f"Checkpoint {key.replace('_', ' ')} does not match the requested run."
    return "Checkpoint contains unexpected compatibility metadata."


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointCorruptionError",
    "CheckpointError",
    "array_tree_fingerprint",
]
