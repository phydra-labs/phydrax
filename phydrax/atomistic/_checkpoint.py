#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import equinox as eqx
import numpy as np

from .._array_archive import (
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._document_resource import decode_json_resource
from .._external_resource import bounded_resource_from_bytes, ResourceLimits
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._publication import publish_resource_set
from .._resource_set import (
    open_bounded_resource_set,
    OpenedResourceSet,
    ResourceSetLimits,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    atomistic_model_identity,
    AtomisticModelArtifact,
    AtomisticModelArtifactError,
    model_artifact_from_section,
    model_artifact_section,
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from ._potential import atomistic_potential_revision
from ._potential_program import (
    PreparedAtomisticPotentialProgram,
    PreparedLearnedGraphPotentialTerm,
)
from ._thermodynamic import PreparedThermodynamicStateTable
from ._training import (
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    AtomisticTrainingResult,
    read_atomistic_training_checkpoint,
    write_atomistic_training_checkpoint,
)
from ._units import AtomisticUnitSystem
from .interchange._mace_checkpoint import MACESourceProvenance


_CHECKPOINT_FORMAT = "phydrax-atomistic-dynamics-checkpoint"
_RESTART_FORMAT = "phydrax-atomistic-dynamics-restart"
_MODEL_LEAF_PREFIX = "model/leaves/"
_RUNTIME_FIELDS = frozenset(
    {
        "format",
        "kind",
        "checkpoint_id",
        "prepared_dynamics_id",
        "thermodynamic_table_id",
        "system_id",
        "potential_id",
        "integrator_id",
        "unit_system",
        "state",
        "payload_id",
        "arrays",
    }
)


class AtomisticCheckpointPlan(StrictModule, NonTrainableState):
    dynamics: PreparedAtomisticDynamics
    thermodynamic: PreparedThermodynamicStateTable
    scope_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        scope_id: str | None = None,
    ) -> None:
        thermodynamic.validate_dynamics(dynamics)
        if scope_id is not None and (
            not isinstance(scope_id, str) or not scope_id or scope_id != scope_id.strip()
        ):
            raise ValueError("Checkpoint scope_id must be a canonical nonempty string.")
        self.scope_id = scope_id
        self.dynamics = dynamics
        self.thermodynamic = thermodynamic
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "atomistic-checkpoint-plan",
                "dynamics": dynamics.prepared_id,
                "thermodynamic": thermodynamic.table_id,
                "system": dynamics.system.prepared_id,
                "potential": dynamics.potential.prepared_id,
                "integrator": dynamics.integrator.plan_id,
                **({} if scope_id is None else {"scope_id": scope_id}),
            }
        )


class AtomisticCheckpoint(StrictModule):
    state: AtomisticDynamicsState
    units: AtomisticUnitSystem
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


def _plan_identities(plan: AtomisticCheckpointPlan, /) -> dict[str, str]:
    identities = {
        "checkpoint_id": plan.checkpoint_id,
        "prepared_dynamics_id": plan.dynamics.prepared_id,
        "thermodynamic_table_id": plan.thermodynamic.table_id,
        "system_id": plan.dynamics.system.prepared_id,
        "potential_id": plan.dynamics.potential.prepared_id,
        "integrator_id": plan.dynamics.integrator.plan_id,
    }
    if plan.scope_id is not None:
        identities["scope_id"] = plan.scope_id
    return identities


def _payload_id(
    plan: AtomisticCheckpointPlan,
    unit_system_id: str,
    state: AtomisticDynamicsState,
    specification: Mapping[str, Any],
    arrays: Mapping[str, Any],
    /,
    **binding: str,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "unit_system": unit_system_id,
            "time": float(state.time),
            "step": int(state.step_index),
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
            **binding,
        }
    )


def _require_runtime(
    plan: AtomisticCheckpointPlan, state: AtomisticDynamicsState, role: str, /
) -> None:
    if not isinstance(plan, AtomisticCheckpointPlan):
        raise TypeError("plan must be AtomisticCheckpointPlan.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError(f"{role} must be AtomisticDynamicsState.")
    if (
        state.prepared_dynamics_id != plan.dynamics.prepared_id
        or state.thermodynamic_table_id != plan.thermodynamic.table_id
    ):
        noun = "state" if role == "state" else "template"
        raise ValueError(f"Checkpoint {noun} belongs to another dynamics runtime.")


def _runtime_manifest(
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    format: str,
    kind: str,
    arrays: dict[str, object],
    /,
) -> dict[str, Any]:
    specification = pack_array_tree("runtime", state, arrays)
    return {
        "format": format,
        "kind": kind,
        **_plan_identities(plan),
        "unit_system": plan.dynamics.system.plan.units.to_dict(),
        "state": specification,
    }


def write_atomistic_checkpoint(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    """Persist dynamics runtime state for an explicitly matching prepared model.

    The checkpoint holds no model: restart requires recreating the identical
    prepared dynamics, including its potential. Use ``write_atomistic_restart``
    for a portable bundle carrying the native model artifact.
    """

    _require_runtime(plan, state, "state")
    arrays: dict[str, object] = {}
    manifest = _runtime_manifest(
        plan, state, _CHECKPOINT_FORMAT, "atomistic-dynamics-runtime", arrays
    )
    payload_id = _payload_id(
        plan,
        plan.dynamics.system.plan.units.unit_system_id,
        state,
        manifest["state"],
        arrays,
    )
    write_array_archive(
        path,
        manifest={**manifest, "payload_id": payload_id},
        arrays=arrays,
    )
    return AtomisticCheckpoint(
        state,
        plan.dynamics.system.plan.units,
        payload_id,
        plan.checkpoint_id,
    )


def _restored_runtime(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
    **binding: str,
) -> AtomisticCheckpoint:
    identities = _plan_identities(plan)
    units = AtomisticUnitSystem.from_dict(manifest["unit_system"])
    if units.unit_system_id != plan.dynamics.system.plan.units.unit_system_id:
        raise ValueError("Atomistic checkpoint complete unit descriptor is incompatible.")
    for name, expected_value in identities.items():
        if manifest[name] != expected_value:
            raise ValueError(f"Atomistic checkpoint {name} does not match the runtime.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("Checkpoint did not reconstruct AtomisticDynamicsState.")
    payload_id = str(manifest["payload_id"])
    if not payload_id:
        raise ValueError("Checkpoint payload_id is empty.")
    expected_payload_id = _payload_id(
        plan, units.unit_system_id, state, manifest["state"], arrays, **binding
    )
    if payload_id != expected_payload_id:
        raise ValueError("Atomistic checkpoint payload identity is corrupt.")
    return AtomisticCheckpoint(state, units, payload_id, plan.checkpoint_id)


def read_atomistic_checkpoint(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    _require_runtime(plan, template, "template")
    manifest, arrays = read_array_archive(path)
    expected = set(_RUNTIME_FIELDS)
    if plan.scope_id is not None:
        expected.add("scope_id")
    if set(manifest) != expected:
        raise ValueError(
            "Atomistic checkpoint manifest is not the canonical current format."
        )
    if (
        manifest["format"] != _CHECKPOINT_FORMAT
        or manifest["kind"] != "atomistic-dynamics-runtime"
    ):
        raise ValueError("File is not an atomistic dynamics checkpoint.")
    return _restored_runtime(manifest, arrays, plan, template)


def _executed_models(plan: AtomisticCheckpointPlan, /) -> tuple[Any, ...]:
    potential = plan.dynamics.potential
    if not isinstance(potential, PreparedAtomisticPotentialProgram):
        return ()
    return tuple(
        term.plan.potential
        for term in potential.terms
        if isinstance(term, PreparedLearnedGraphPotentialTerm)
    )


def _require_model_binding(
    plan: AtomisticCheckpointPlan, identity: Mapping[str, str], /
) -> None:
    models = _executed_models(plan)
    if not models:
        raise ValueError("A model restart requires dynamics executing a learned model.")
    for model in models:
        if atomistic_model_identity(model) != identity:
            raise AtomisticModelArtifactError(
                "The dynamics execute a model that differs from the bundled artifact."
            )


def _intrinsic(identity: Mapping[str, str], /) -> dict[str, str]:
    return {name: value for name, value in identity.items() if name != "artifact_id"}


def write_atomistic_restart(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    /,
    *,
    model: Any,
    source: MACESourceProvenance | None = None,
    licenses: Sequence[str] = (),
) -> AtomisticCheckpoint:
    """Write a portable restart: the native model artifact plus dynamics state.

    Every learned term executed by ``plan`` must run exactly ``model``
    (architecture, structure, parameters and fixed leaves). A fresh process
    restores the model with ``read_atomistic_restart_model``, rebuilds its
    dynamics over that model, and resumes with ``read_atomistic_restart``.
    Compiled caches are not part of restart correctness.
    """

    _require_runtime(plan, state, "state")
    section, model_arrays, model_manifest = model_artifact_section(
        model, source=source, licenses=licenses
    )
    _require_model_binding(plan, _intrinsic(section["identity"]))
    arrays: dict[str, object] = dict(model_arrays)
    manifest = _runtime_manifest(
        plan, state, _RESTART_FORMAT, "atomistic-dynamics-restart", arrays
    )
    payload_id = _payload_id(
        plan,
        plan.dynamics.system.plan.units.unit_system_id,
        state,
        manifest["state"],
        {name: value for name, value in arrays.items() if name.startswith("runtime/")},
        model=model_manifest.artifact_id,
    )
    write_array_archive(
        path,
        manifest={**manifest, "model": section, "payload_id": payload_id},
        limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
        arrays=arrays,
    )
    return AtomisticCheckpoint(
        state, plan.dynamics.system.plan.units, payload_id, plan.checkpoint_id
    )


def _restart_payload(
    path: str | Path, /
) -> tuple[dict[str, Any], AtomisticModelArtifact, dict[str, np.ndarray]]:
    manifest, arrays = read_array_archive(path, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS)
    if (
        manifest.get("format") != _RESTART_FORMAT
        or manifest.get("kind") != "atomistic-dynamics-restart"
        or not isinstance(manifest.get("model"), dict)
    ):
        raise ValueError("File is not an atomistic dynamics restart.")
    model_arrays = {
        name: value
        for name, value in arrays.items()
        if name.startswith(_MODEL_LEAF_PREFIX)
    }
    runtime_arrays = {
        name: value for name, value in arrays.items() if name.startswith("runtime/")
    }
    if len(model_arrays) + len(runtime_arrays) != len(arrays):
        raise ValueError("Atomistic restart contains arrays outside its namespaces.")
    artifact = model_artifact_from_section(manifest["model"], model_arrays)
    return manifest, artifact, runtime_arrays


def read_atomistic_restart_model(path: str | Path, /) -> AtomisticModelArtifact:
    """Restore and fully validate the native model bundled in a restart."""

    return _restart_payload(path)[1]


def read_atomistic_restart(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    """Resume dynamics state after verifying the bundled model executes in ``plan``.

    ``plan`` must be rebuilt over the restored model; any altered model,
    source binding, system, integrator, thermodynamic table, scope or graph
    preparation refuses before state is returned.
    """

    _require_runtime(plan, template, "template")
    manifest, artifact, runtime_arrays = _restart_payload(path)
    expected = set(_RUNTIME_FIELDS) | {"model"}
    if plan.scope_id is not None:
        expected.add("scope_id")
    if set(manifest) != expected:
        raise ValueError("Atomistic restart manifest is not the canonical format.")
    identity = {
        "architecture_id": artifact.manifest.architecture_id,
        "method_id": artifact.manifest.method_id,
        "structure_id": artifact.manifest.structure_id,
        "semantic_id": artifact.manifest.semantic_id,
        "numeric_revision_id": artifact.manifest.numeric_revision.revision_id,
        "content_id": artifact.manifest.content_id,
    }
    _require_model_binding(plan, identity)
    return _restored_runtime(
        manifest,
        runtime_arrays,
        plan,
        template,
        model=artifact.manifest.artifact_id,
    )


_TRAINING_RESTART_FORMAT = "phydrax-atomistic-training-restart"
_TRAINING_RECEIPT_FILE = "restart.json"
_TRAINING_MODEL_FILE = "model.phydrax"
_TRAINING_STATE_DIRECTORY = "training"
_TRAINING_STATE_MANIFEST = f"{_TRAINING_STATE_DIRECTORY}/manifest.json"
_TRAINING_RECEIPT_FIELDS = frozenset(
    {
        "format",
        "members",
        "model_artifact_id",
        "result_id",
        "problem_id",
        "policy_id",
        "continuation_id",
        "normalization_id",
        "training_checkpoint_id",
    }
)
_TRAINING_RECEIPT_BYTES = 1_048_576
_TRAINING_STATE_BYTES = DEFAULT_ARRAY_ARCHIVE_LIMITS.max_aggregate_bytes
_TRAINING_MANIFEST_BYTES = 16 * 1024 * 1024
# Receipt, model artifact, training directory, its manifest and its state file.
_TRAINING_RESTART_LIMITS = ResourceSetLimits(
    max_total_bytes=ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_container_bytes
    + _TRAINING_STATE_BYTES
    + _TRAINING_MANIFEST_BYTES
    + _TRAINING_RECEIPT_BYTES,
    max_member_bytes=max(
        ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_container_bytes, _TRAINING_STATE_BYTES
    ),
    max_members=5,
    max_depth=2,
)


def _training_receipt(
    result: AtomisticTrainingResult, model_artifact_id: str, /
) -> dict[str, str]:
    return {
        "model_artifact_id": model_artifact_id,
        "result_id": result.result_id,
        "problem_id": result.problem_id,
        "policy_id": result.policy_id,
        "continuation_id": result.continuation_id,
        "normalization_id": result.normalization.normalization_id,
        "training_checkpoint_id": result.training_checkpoint_id,
    }


def _staged_training_members(staging: Path, /) -> dict[str, bytes]:
    with open_bounded_resource_set(
        staging.name, trusted_root=staging.parent, limits=_TRAINING_RESTART_LIMITS
    ) as staged:
        return {
            member.relative_path: staged.read_member(member.relative_path)
            for member in staged.manifest.members
        }


def write_atomistic_training_restart(
    directory: str | Path,
    result: AtomisticTrainingResult,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    source: MACESourceProvenance | None = None,
    licenses: Sequence[str] = (),
) -> Path:
    """Atomically publish a fresh-process training continuation for one model.

    The bundle directory holds the native artifact of ``result.potential``, the
    training owner's canonical kernel checkpoint (optimizer state, root key,
    cursors, best potential and histories) and a ``restart.json`` receipt
    binding every member digest and the model, result, problem, policy,
    continuation, normalization and kernel identities. Both components are
    written and validated in private staging, then the complete bundle replaces
    ``directory`` in one atomic exchange: a failed or interrupted publication
    leaves any previous bundle readable and unchanged. The training data recipe
    is not stored: continuation rebuilds the same ``AtomisticTrainingProblem``.
    """

    if not isinstance(result, AtomisticTrainingResult):
        raise TypeError("result must be an AtomisticTrainingResult.")
    destination = Path(directory)
    with TemporaryDirectory(prefix="phydrax-training-restart-") as scratch:
        staging = Path(scratch).resolve() / "bundle"
        staging.mkdir(mode=0o700)
        model = write_atomistic_model_artifact(
            staging / _TRAINING_MODEL_FILE,
            result.potential,
            source=source,
            licenses=licenses,
        )
        write_atomistic_training_checkpoint(
            staging / _TRAINING_STATE_DIRECTORY, result, policy
        )
        members = _staged_training_members(staging)
    receipt = {
        "format": _TRAINING_RESTART_FORMAT,
        "members": {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        },
        **_training_receipt(result, model.artifact_id),
    }
    members[_TRAINING_RECEIPT_FILE] = (
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_resource_set(
        destination, members, limits=_TRAINING_RESTART_LIMITS, mode="atomic_replace"
    )
    return destination


def _admitted_training_receipt(bundle: OpenedResourceSet, /) -> dict[str, Any]:
    inventory = {
        member.relative_path: member.content_sha256 for member in bundle.manifest.members
    }
    if _TRAINING_RECEIPT_FILE not in inventory:
        raise ValueError(
            "Directory is not a canonical atomistic training restart bundle."
        )
    receipt = decode_json_resource(
        bounded_resource_from_bytes(
            bundle.read_member(
                _TRAINING_RECEIPT_FILE, maximum_bytes=_TRAINING_RECEIPT_BYTES
            ),
            limits=ResourceLimits(_TRAINING_RECEIPT_BYTES, 8, 1_000, 1_000, 0),
            source_path=_TRAINING_RECEIPT_FILE,
        )
    ).value
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _TRAINING_RECEIPT_FIELDS
        or receipt["format"] != _TRAINING_RESTART_FORMAT
        or any(
            not isinstance(receipt[name], str) or not receipt[name]
            for name in _TRAINING_RECEIPT_FIELDS - {"members"}
        )
        or not isinstance(receipt["members"], dict)
    ):
        raise ValueError("Atomistic training restart receipt is not canonical.")
    members = receipt["members"]
    states = [
        name
        for name in members
        if name.startswith(f"{_TRAINING_STATE_DIRECTORY}/state-")
        and name.endswith(".eqx")
        and name.count("/") == 1
    ]
    canonical = {_TRAINING_MODEL_FILE, _TRAINING_STATE_MANIFEST, *states}
    if len(states) != 1 or set(members) != canonical:
        raise ValueError("Atomistic training restart receipt members are not canonical.")
    if set(inventory) != {_TRAINING_RECEIPT_FILE, *members} or any(
        inventory[name] != digest for name, digest in members.items()
    ):
        raise ValueError(
            "Atomistic training restart members differ from their publication receipt."
        )
    return receipt


def read_atomistic_training_restart(
    directory: str | Path,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
) -> tuple[AtomisticModelArtifact, AtomisticTrainingResult]:
    """Restore the trained model artifact and its exact training continuation.

    The bundle is admitted as one bounded resource set whose inventory and
    member digests must equal its publication receipt; the model and training
    owners then restore exactly those admitted bytes. The artifact's model is
    the structural template of the kernel restore; the restored committed
    parameters and every fixed leaf must equal the artifact exactly, and the
    restored model, result, problem, policy, continuation, normalization and
    kernel identities must equal the receipt, so mixed or stale bundles refuse.
    Resume with ``fit_atomistic_potential(..., continuation=result)``.
    """

    source = Path(directory).expanduser().absolute()
    with TemporaryDirectory(prefix="phydrax-training-restart-") as scratch:
        admitted = Path(scratch).resolve()
        with open_bounded_resource_set(
            source.name, trusted_root=source.parent, limits=_TRAINING_RESTART_LIMITS
        ) as bundle:
            receipt = _admitted_training_receipt(bundle)
            (admitted / _TRAINING_STATE_DIRECTORY).mkdir(mode=0o700)
            for name in receipt["members"]:
                (admitted / name).write_bytes(bundle.read_member(name))
        artifact = read_atomistic_model_artifact(admitted / _TRAINING_MODEL_FILE)
        result = read_atomistic_training_checkpoint(
            admitted / _TRAINING_STATE_DIRECTORY, artifact.model, problem, policy
        )
    if (
        atomistic_potential_revision(result.potential).revision_id
        != artifact.manifest.numeric_revision.revision_id
        or atomistic_model_identity(result.potential)
        != atomistic_model_identity(artifact.model)
    ):
        raise AtomisticModelArtifactError(
            "The training continuation belongs to a different model artifact."
        )
    if _training_receipt(result, artifact.manifest.artifact_id) != {
        name: receipt[name] for name in _TRAINING_RECEIPT_FIELDS - {"format", "members"}
    }:
        raise ValueError(
            "Atomistic training restart components differ from their receipt identities."
        )
    return artifact, result


__all__ = [
    "AtomisticCheckpoint",
    "AtomisticCheckpointPlan",
    "read_atomistic_checkpoint",
    "read_atomistic_restart",
    "read_atomistic_restart_model",
    "read_atomistic_training_restart",
    "write_atomistic_checkpoint",
    "write_atomistic_restart",
    "write_atomistic_training_restart",
]
