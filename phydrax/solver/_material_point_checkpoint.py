#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

from .._array_archive import (
    array_collection_digest,
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from .._external_resource import read_bounded_resource, ResourceLimits
from .._fingerprint import canonical_fingerprint
from .._publication import publish_bytes
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.mpm import MPMRuntimeState
from ..equations import CompiledMaterialPointProblem


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


class MPMCheckpointManifest(StrictModule, NonTrainableState):
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
    checkpoint_id: str = eqx.field(static=True)
    leaf_names: tuple[str, ...] = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    leaf_dtypes: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        compiled: CompiledMaterialPointProblem,
        template_state: MPMRuntimeState,
        /,
    ):
        if not isinstance(compiled, CompiledMaterialPointProblem):
            raise TypeError("compiled must be CompiledMaterialPointProblem.")
        if not isinstance(template_state, MPMRuntimeState):
            raise TypeError("template_state must be MPMRuntimeState.")
        paths, _leaves = jax.tree_util.tree_flatten_with_path(template_state)[0], None
        names = tuple(_leaf_name(path, index) for index, (path, _) in enumerate(paths))
        if len(set(names)) != len(names):
            raise ValueError("MPM checkpoint leaf names are not unique.")
        shapes = tuple(tuple(np.asarray(leaf).shape) for _, leaf in paths)
        dtypes = tuple(np.asarray(leaf).dtype.str for _, leaf in paths)
        self.compiled = compiled
        self.template_state = template_state
        self.leaf_names = names
        self.leaf_shapes = shapes
        self.leaf_dtypes = dtypes
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "mpm-checkpoint-plan",
                "compilation": compiled.compilation_id,
                "leaf_names": names,
                "leaf_shapes": [list(shape) for shape in shapes],
                "leaf_dtypes": dtypes,
            }
        )

    def _arrays(self, state: MPMRuntimeState):
        paths, _ = jax.tree_util.tree_flatten_with_path(state)
        names = tuple(_leaf_name(path, index) for index, (path, _) in enumerate(paths))
        if names != self.leaf_names:
            raise ValueError("MPM checkpoint runtime tree layout changed.")
        arrays = tuple(np.asarray(leaf) for _, leaf in paths)
        signatures = tuple((tuple(array.shape), array.dtype.str) for array in arrays)
        expected = tuple(zip(self.leaf_shapes, self.leaf_dtypes, strict=True))
        if signatures != expected:
            raise ValueError("MPM checkpoint runtime leaf shape or dtype changed.")
        return dict(zip(names, arrays, strict=True))

    def write(self, path: str | Path, state: MPMRuntimeState, /, *, generation: int = 0):
        if not isinstance(state, MPMRuntimeState):
            raise TypeError("state must be MPMRuntimeState.")
        arrays = self._arrays(state)
        payload_digest = array_collection_digest(arrays)
        metadata = {
            "kind": "material-point-checkpoint",
            "checkpoint_id": self.checkpoint_id,
            "compilation_id": self.compiled.compilation_id,
            "claim_id": self.compiled.claim_id,
            "generation": int(generation),
            "accepted_step": int(np.asarray(state.accepted_step)),
            "physical_time_hex": float(np.asarray(state.time)).hex(),
        }
        payload_id = canonical_fingerprint(
            {
                "metadata": metadata,
                "array_collection_digest": payload_digest,
            }
        )
        metadata["payload_id"] = payload_id
        metadata["manifest_id"] = canonical_fingerprint(metadata)
        write_array_archive(path, manifest=metadata, arrays=arrays)
        return MPMCheckpointManifest(
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
        path = directory_ / f"generation-{int(generation):08d}.mpmckpt"
        manifest = self.write(path, state, generation=generation)
        publish_bytes(
            directory_ / "CURRENT",
            (path.name + "\n").encode("utf-8"),
            maximum_bytes=4096,
            mode="atomic_replace",
        )
        return manifest

    def read(self, path: str | Path, /):
        template_paths, template_tree = jax.tree_util.tree_flatten_with_path(
            self.template_state
        )
        expected = {
            _leaf_name(path_, index): (
                tuple(np.asarray(template).shape),
                np.asarray(template).dtype,
            )
            for index, (path_, template) in enumerate(template_paths)
        }
        try:
            manifest, arrays = read_array_archive(path, expected_inventory=expected)
        except ArrayArchiveCorruptionError as error:
            raise ValueError(
                f"MPM checkpoint checksum or structure failed: {error}"
            ) from error
        if (
            manifest.get("kind") != "material-point-checkpoint"
            or manifest.get("checkpoint_id") != self.checkpoint_id
            or manifest.get("compilation_id") != self.compiled.compilation_id
        ):
            raise ValueError("MPM checkpoint identity is incompatible.")
        payload_id = canonical_fingerprint(
            {
                "metadata": {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"arrays", "payload_id", "manifest_id"}
                },
                "array_collection_digest": array_collection_digest(arrays),
            }
        )
        if payload_id != manifest.get("payload_id"):
            raise ValueError("MPM checkpoint payload identity mismatch.")
        manifest_without_id = {
            key: value
            for key, value in manifest.items()
            if key not in {"arrays", "manifest_id"}
        }
        if canonical_fingerprint(manifest_without_id) != manifest.get("manifest_id"):
            raise ValueError("MPM checkpoint manifest identity mismatch.")
        ordered = [
            jnp.asarray(arrays[_leaf_name(path_, index)])
            for index, (path_, _template) in enumerate(template_paths)
        ]
        restored = jax.tree_util.tree_unflatten(template_tree, ordered)
        if not isinstance(restored, MPMRuntimeState):
            raise TypeError("Restored checkpoint is not MPMRuntimeState.")
        return restored, manifest

    def read_current(self, directory: str | Path, /):
        directory_ = Path(directory)
        resource = read_bounded_resource(
            "CURRENT",
            trusted_root=directory_,
            limits=ResourceLimits(4096, 1, 1, 0, 0),
        )
        try:
            name = resource.data.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError("MPM checkpoint CURRENT pointer is invalid.") from error
        if not name or Path(name).name != name:
            raise ValueError("MPM checkpoint CURRENT pointer is invalid.")
        return self.read(directory_ / name)


__all__ = [
    "MPMCheckpointManifest",
    "MPMCheckpointPlan",
]
