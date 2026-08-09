#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from .._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from .._fingerprint import array_tree_fingerprint, array_tree_signature
from ._posterior import PosteriorProblem


_CHECKPOINT_FORMAT = "phydrax-uq-checkpoint"


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
    """Atomically write one pickle-free checkpoint."""
    manifest = {
        "format": _CHECKPOINT_FORMAT,
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
    expected = {"format", "kind", "versions", "compatibility", "state", "arrays"}
    missing = expected - set(manifest)
    unknown = set(manifest) - expected
    if missing or unknown:
        raise CheckpointCorruptionError(
            "Checkpoint manifest must use the current canonical fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    if manifest.get("format") != _CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError("File is not a PhydraX UQ checkpoint.")
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
    return {
        "checkpoint_id": identifier,
        "problem_type": _qualified_type(problem),
        "parameter_tree": array_tree_signature(initial),
        "problem_array_digest": array_tree_fingerprint(problem)["sha256"],
        "initial_probe_digest": array_tree_fingerprint(
            {"value": value, "gradient": gradient}
        )["sha256"],
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


def _write_array_archive(
    path: str | os.PathLike[str],
    /,
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> Path:
    """Compatibility wrapper for the shared portable array archive."""
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def _read_array_archive(
    path: str | os.PathLike[str],
    /,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read a shared array archive using the established UQ error type."""
    try:
        return read_array_archive(path)
    except ArrayArchiveCorruptionError as error:
        raise CheckpointCorruptionError(str(error)) from error


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
]
