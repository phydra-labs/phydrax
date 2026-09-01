#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.mpm import MPMRuntimeState
from ..equations import CompiledMaterialPointProblem


_CHECKPOINT_SCHEMA_VERSION = 1


def _leaf_name(path, index):
    tokens = []
    for item in path:
        if isinstance(item, GetAttrKey):
            tokens.append(str(item.name))
        elif isinstance(item, (SequenceKey, FlattenedIndexKey)):
            tokens.append(str(item.idx))
        elif isinstance(item, DictKey):
            tokens.append(str(item.key))
        else:
            tokens.append(str(item))
    return "/".join(tokens) or f"leaf-{index:06d}"


def _array_bytes(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, content: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


class MPMCheckpointMigration(StrictModule, NonTrainableState):
    source_version: int = eqx.field(static=True)
    target_version: int = eqx.field(static=True)
    migration_id: str = eqx.field(static=True)
    migrate_manifest: Callable = eqx.field(static=True)

    def __init__(
        self,
        source_version: int,
        target_version: int,
        migrate_manifest: Callable,
        /,
        *,
        migration_id: str,
    ):
        source = int(source_version)
        target = int(target_version)
        identifier = str(migration_id)
        if (
            source < 0
            or target != source + 1
            or not callable(migrate_manifest)
            or not identifier
        ):
            raise ValueError("MPM checkpoint migration must advance one schema version.")
        self.source_version = source
        self.target_version = target
        self.migration_id = identifier
        self.migrate_manifest = migrate_manifest


class MPMCheckpointManifest(StrictModule, NonTrainableState):
    schema_version: int = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    claim_id: str | None = eqx.field(static=True)
    generation: int = eqx.field(static=True)
    accepted_step: int = eqx.field(static=True)
    physical_time_hex: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)


class MPMCheckpointPlan(StrictModule, NonTrainableState):
    compiled: CompiledMaterialPointProblem
    template_state: MPMRuntimeState
    migrations: tuple[MPMCheckpointMigration, ...]
    checkpoint_id: str = eqx.field(static=True)
    leaf_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        compiled: CompiledMaterialPointProblem,
        template_state: MPMRuntimeState,
        /,
        *,
        migrations: Sequence[MPMCheckpointMigration] = (),
    ):
        if not isinstance(compiled, CompiledMaterialPointProblem):
            raise TypeError("compiled must be CompiledMaterialPointProblem.")
        if not isinstance(template_state, MPMRuntimeState):
            raise TypeError("template_state must be MPMRuntimeState.")
        paths, leaves = jax.tree_util.tree_flatten_with_path(template_state)[0], None
        names = tuple(_leaf_name(path, index) for index, (path, _) in enumerate(paths))
        if len(set(names)) != len(names):
            raise ValueError("MPM checkpoint leaf names are not unique.")
        migrations_ = tuple(migrations)
        expected = 0
        for value in sorted(migrations_, key=lambda item: item.source_version):
            if not isinstance(value, MPMCheckpointMigration):
                raise TypeError("migrations must contain MPMCheckpointMigration.")
            expected = max(expected, value.target_version)
        self.compiled = compiled
        self.template_state = template_state
        self.migrations = migrations_
        self.leaf_names = names
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "mpm-checkpoint-plan",
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "compilation": compiled.compilation_id,
                "leaf_names": names,
                "leaf_shapes": [list(np.asarray(leaf).shape) for _, leaf in paths],
                "leaf_dtypes": [np.asarray(leaf).dtype.str for _, leaf in paths],
                "migrations": [value.migration_id for value in migrations_],
                "migration_extent": expected,
            }
        )

    def _arrays(self, state: MPMRuntimeState):
        paths, _ = jax.tree_util.tree_flatten_with_path(state)
        names = tuple(_leaf_name(path, index) for index, (path, _) in enumerate(paths))
        if names != self.leaf_names:
            raise ValueError("MPM checkpoint runtime tree layout changed.")
        return {
            name: np.asarray(leaf) for name, (_, leaf) in zip(names, paths, strict=True)
        }

    def write(self, path: str | Path, state: MPMRuntimeState, /, *, generation: int = 0):
        if not isinstance(state, MPMRuntimeState):
            raise TypeError("state must be MPMRuntimeState.")
        arrays = self._arrays(state)
        inventory = {}
        payloads = {}
        for index, (name, value) in enumerate(sorted(arrays.items())):
            payload = _array_bytes(value)
            member = f"arrays/{index:06d}.npy"
            payloads[member] = payload
            inventory[name] = {
                "member": member,
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": _sha256(payload),
            }
        metadata = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "compilation_id": self.compiled.compilation_id,
            "claim_id": self.compiled.claim_id,
            "generation": int(generation),
            "accepted_step": int(np.asarray(state.accepted_step)),
            "physical_time_hex": float(np.asarray(state.time)).hex(),
            "arrays": inventory,
        }
        payload_id = canonical_fingerprint(
            {
                "metadata": metadata,
                "array_checksums": {k: v["sha256"] for k, v in inventory.items()},
            }
        )
        metadata["payload_id"] = payload_id
        metadata["manifest_id"] = canonical_fingerprint(metadata)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(
                    "manifest.json", json.dumps(metadata, sort_keys=True, indent=2)
                )
                for member, payload in payloads.items():
                    archive.writestr(member, payload)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(target)
        return MPMCheckpointManifest(
            _CHECKPOINT_SCHEMA_VERSION,
            self.checkpoint_id,
            self.compiled.compilation_id,
            metadata["claim_id"],
            int(generation),
            metadata["accepted_step"],
            metadata["physical_time_hex"],
            payload_id,
            metadata["manifest_id"],
        )

    def write_generation(
        self, directory: str | Path, state: MPMRuntimeState, /, *, generation: int
    ):
        directory_ = Path(directory)
        directory_.mkdir(parents=True, exist_ok=True)
        path = directory_ / f"generation-{int(generation):08d}.mpmckpt"
        manifest = self.write(path, state, generation=generation)
        pointer = directory_ / "CURRENT"
        _atomic_text(pointer, path.name + "\n")
        directory_fd = os.open(directory_, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest

    def _migrate(self, manifest):
        current = dict(manifest)
        version = int(current.get("schema_version", -1))
        while version < _CHECKPOINT_SCHEMA_VERSION:
            candidates = [
                value for value in self.migrations if value.source_version == version
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"No unique MPM checkpoint migration from schema {version}."
                )
            current = dict(candidates[0].migrate_manifest(current))
            version = int(current.get("schema_version", -1))
        return current

    def read(self, path: str | Path, /):
        target = Path(path)
        with zipfile.ZipFile(target, "r") as archive:
            manifest = self._migrate(json.loads(archive.read("manifest.json")))
            if (
                manifest.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
                or manifest.get("checkpoint_id") != self.checkpoint_id
                or manifest.get("compilation_id") != self.compiled.compilation_id
            ):
                raise ValueError("MPM checkpoint identity is incompatible.")
            inventory = manifest.get("arrays")
            if not isinstance(inventory, dict) or tuple(sorted(inventory)) != tuple(
                sorted(self.leaf_names)
            ):
                raise ValueError("MPM checkpoint array inventory changed.")
            values = {}
            for name, record in inventory.items():
                payload = archive.read(record["member"])
                if _sha256(payload) != record["sha256"]:
                    raise ValueError(f"MPM checkpoint checksum failed for {name}.")
                value = np.load(io.BytesIO(payload), allow_pickle=False)
                if (
                    list(value.shape) != record["shape"]
                    or value.dtype.str != record["dtype"]
                ):
                    raise ValueError(f"MPM checkpoint shape/dtype failed for {name}.")
                values[name] = jnp.asarray(value)
        template_paths, template_tree = jax.tree_util.tree_flatten_with_path(
            self.template_state
        )
        ordered = []
        for index, (path_, template) in enumerate(template_paths):
            name = _leaf_name(path_, index)
            value = values[name]
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(f"MPM checkpoint template mismatch for {name}.")
            ordered.append(value)
        restored = jax.tree_util.tree_unflatten(template_tree, ordered)
        if not isinstance(restored, MPMRuntimeState):
            raise TypeError("Restored checkpoint is not MPMRuntimeState.")
        return restored, manifest

    def read_current(self, directory: str | Path, /):
        directory_ = Path(directory)
        name = (directory_ / "CURRENT").read_text().strip()
        if not name or Path(name).name != name:
            raise ValueError("MPM checkpoint CURRENT pointer is invalid.")
        return self.read(directory_ / name)


__all__ = [
    "MPMCheckpointManifest",
    "MPMCheckpointMigration",
    "MPMCheckpointPlan",
]
