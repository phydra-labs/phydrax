#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import numpy as np
import optax

from ..._fingerprint import canonical_fingerprint
from ..._frozendict import frozendict
from ..._model._structure import deserialise_model_leaf, serialise_model_leaf
from ..._strict import StrictModule
from ..._training_checkpoint import (
    _prune_state_files,
    _publish_manifest,
    _publish_state,
    _read_manifest,
    _verify_state,
)
from ...domain import (
    broken_field,
    DomainFunction,
    LocalFieldFamily,
    partition_of_unity_field,
    SubdomainCover,
)
from .._functional_solver import FunctionalSolver


class FunctionalPatchParticipant(StrictModule):
    """One trainable local functional participant owned by a cover patch."""

    solver: FunctionalSolver
    patch_id: str = eqx.field(static=True)
    participant_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_id: str,
        solver: FunctionalSolver,
        /,
        *,
        participant_id: str | None = None,
    ):
        if not isinstance(solver, FunctionalSolver):
            raise TypeError("solver must be a FunctionalSolver.")
        patch = str(patch_id)
        if not patch:
            raise ValueError("patch_id must be non-empty.")
        self.patch_id = patch
        self.participant_id = (
            f"functional/{patch}" if participant_id is None else str(participant_id)
        )
        self.solver = solver

    @property
    def functions(self):
        return self.solver.functions

    def solve(
        self,
        *,
        num_iter: int,
        optim: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
        seed: int = 0,
        jit: bool = True,
    ) -> FunctionalPatchParticipant:
        trained = self.solver.solve(
            num_iter=num_iter,
            optim=optim,
            seed=seed,
            jit=jit,
            keep_best=False,
            log_every=0,
        )
        return FunctionalPatchParticipant(
            self.patch_id,
            trained,
            participant_id=self.participant_id,
        )


class FixedPatchParticipant(StrictModule):
    """One immutable external or numerical field participant."""

    functions: frozendict[str, DomainFunction]
    patch_id: str = eqx.field(static=True)
    participant_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_id: str,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        participant_id: str | None = None,
    ):
        values = frozendict(functions)
        if not values or any(
            not isinstance(value, DomainFunction) for value in values.values()
        ):
            raise TypeError("functions must contain DomainFunction values.")
        patch = str(patch_id)
        if not patch:
            raise ValueError("patch_id must be non-empty.")
        self.patch_id = patch
        self.participant_id = (
            f"fixed/{patch}" if participant_id is None else str(participant_id)
        )
        self.functions = values


PatchParticipant = FunctionalPatchParticipant | FixedPatchParticipant


class HybridFunctionalDecomposition(StrictModule):
    """One explicit participant per patch with a common trace topology."""

    cover: SubdomainCover
    participants: tuple[PatchParticipant, ...]

    def __init__(
        self,
        cover: SubdomainCover,
        participants: Sequence[PatchParticipant],
        /,
    ):
        values = tuple(participants)
        if any(
            not isinstance(value, (FunctionalPatchParticipant, FixedPatchParticipant))
            for value in values
        ):
            raise TypeError("participants contain an unsupported patch participant.")
        patch_ids = tuple(value.patch_id for value in values)
        if set(patch_ids) != set(cover.patch_ids) or len(set(patch_ids)) != len(
            patch_ids
        ):
            raise ValueError("Hybrid decomposition requires one participant per patch.")
        by_patch = {value.patch_id: value for value in values}
        self.cover = cover
        self.participants = tuple(by_patch[patch_id] for patch_id in cover.patch_ids)

    def participant(self, patch_id: str, /) -> PatchParticipant:
        patch = self.cover.patch(patch_id)
        return self.participants[self.cover.patches.index(patch)]

    def family(self, field_name: str, /) -> LocalFieldFamily:
        name = str(field_name)
        fields = {}
        for participant in self.participants:
            if name not in participant.functions:
                raise KeyError(
                    f"Participant {participant.participant_id!r} has no field {name!r}."
                )
            fields[participant.patch_id] = participant.functions[name]
        return LocalFieldFamily(name, self.cover, fields)


def _field_content_fingerprint(family: LocalFieldFamily, /) -> str:
    records = []
    for leaf in jax.tree_util.tree_leaves(family.fields):
        if not eqx.is_array(leaf):
            continue
        array = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        records.append(
            {
                "shape": tuple(int(value) for value in array.shape),
                "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    return canonical_fingerprint(
        {
            "tree": str(jax.tree_util.tree_structure(family.fields)),
            "arrays": tuple(records),
        }
    )


class DecompositionDeploymentArtifact(StrictModule):
    """Content-identified local fields with explicit deployment assembly semantics."""

    family: LocalFieldFamily
    assembly: Literal["partition-of-unity", "broken-first-owner"] = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: LocalFieldFamily,
        /,
        *,
        assembly: Literal["partition-of-unity", "broken-first-owner"],
    ):
        if assembly not in ("partition-of-unity", "broken-first-owner"):
            raise ValueError("Unknown deployment assembly.")
        self.family = family
        self.assembly = assembly
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "functional-decomposition-deployment",
                "cover_id": family.cover.cover_id,
                "field_id": family.field_id,
                "assembly": assembly,
                "field_content_id": _field_content_fingerprint(family),
            }
        )

    def field(self) -> DomainFunction:
        if self.assembly == "partition-of-unity":
            return partition_of_unity_field(self.family)
        return broken_field(self.family).as_domain_function(ownership="first")


_ARTIFACT_FORMAT = "phydrax-functional-decomposition-deployment"


def save_decomposition_artifact(
    path: str | Path,
    artifact: DecompositionDeploymentArtifact,
    /,
) -> Path:
    if not isinstance(artifact, DecompositionDeploymentArtifact):
        raise TypeError("artifact must be a DecompositionDeploymentArtifact.")
    destination = Path(path)
    state_path, checksum = _publish_state(
        destination,
        lambda target: eqx.tree_serialise_leaves(
            target,
            artifact,
            filter_spec=serialise_model_leaf,
        ),
    )
    _publish_manifest(
        destination / "manifest.json",
        {
            "format": _ARTIFACT_FORMAT,
            "state_file": state_path.name,
            "state_sha256": checksum,
            "artifact_id": artifact.artifact_id,
            "cover_id": artifact.family.cover.cover_id,
            "assembly": artifact.assembly,
        },
    )
    _prune_state_files(destination, state_path.name)
    return destination


def load_decomposition_artifact(
    path: str | Path,
    artifact_like: DecompositionDeploymentArtifact,
    /,
) -> DecompositionDeploymentArtifact:
    if not isinstance(artifact_like, DecompositionDeploymentArtifact):
        raise TypeError("artifact_like must be a DecompositionDeploymentArtifact.")
    source = Path(path)
    manifest = _read_manifest(source / "manifest.json")
    expected = {
        "format",
        "state_file",
        "state_sha256",
        "artifact_id",
        "cover_id",
        "assembly",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("Deployment manifest fields are not canonical.")
    if manifest["format"] != _ARTIFACT_FORMAT:
        raise ValueError("File is not a Phydrax decomposition deployment artifact.")
    if manifest["artifact_id"] != artifact_like.artifact_id:
        raise ValueError("Deployment artifact identity mismatch.")
    if manifest["cover_id"] != artifact_like.family.cover.cover_id:
        raise ValueError("Deployment cover identity mismatch.")
    if manifest["assembly"] != artifact_like.assembly:
        raise ValueError("Deployment assembly mismatch.")
    state_path = source / manifest["state_file"]
    _verify_state(state_path, manifest["state_sha256"])
    return eqx.tree_deserialise_leaves(
        state_path,
        artifact_like,
        filter_spec=deserialise_model_leaf,
    )


__all__ = [
    "DecompositionDeploymentArtifact",
    "FixedPatchParticipant",
    "FunctionalPatchParticipant",
    "HybridFunctionalDecomposition",
    "load_decomposition_artifact",
    "save_decomposition_artifact",
]
