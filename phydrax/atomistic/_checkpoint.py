#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, final

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
from .._model import model_structure_recipe
from .._model._structure import (
    model_from_array_recipe,
    model_recipe_array_inventory,
    pack_model_array_tree,
)
from .._publication import publish_resource_set
from .._resource_set import open_bounded_resource_set, ResourceSetLimits
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._conditioning import AtomisticConditioning, AtomisticObservableRequest
from ._dynamics import (
    _underlying_program,
    AtomisticDynamicsState,
    PreparedAtomisticDynamics,
)
from ._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    atomistic_model_identity,
    AtomisticModelArtifact,
    AtomisticModelArtifactError,
    AtomisticSourceProvenance,
    model_artifact_from_section,
    model_artifact_section,
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundlePlan,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential import atomistic_model_revision
from ._potential_program import (
    AbstractPreparedAtomisticHamiltonian,
    LearnedGraphPotentialTerm,
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
from ._training_objectives import AbstractAtomisticObjectiveReduction
from ._units import AtomisticUnitSystem


_CHECKPOINT_FORMAT = "phydrax-atomistic-dynamics-checkpoint"
_RESTART_FORMAT = "phydrax-atomistic-dynamics-restart"
_CHECKPOINT_KIND = "atomistic-dynamics-runtime"
_RESTART_KIND = "atomistic-dynamics-restart"
_MODEL_LEAF_PREFIX = "model/leaves/"
_ARTIFACT_PREFIX = "artifacts"
_TERM_PREFIX = "terms"
_CONDITIONING_MEMBER = "conditioning"
_KEY_MEMBER = "key"
_MODELS_FIELDS = frozenset({"artifacts", "terms"})
_TERM_RECORD_FIELDS = frozenset(
    {
        "name",
        "artifact",
        "force_group",
        "allow_periodic",
        "conditioning",
        "key",
        "request",
        "term_id",
        "binding_id",
    }
)
_RUNTIME_PREFIX = "runtime"
_BUNDLE_FIELDS = frozenset({"plan", "source_levels", "role_levels"})
_RUNTIME_FIELDS = frozenset(
    {
        "format",
        "kind",
        "checkpoint_id",
        "prepared_dynamics_id",
        "thermodynamic_table_id",
        "system_id",
        "hamiltonian_id",
        "program_id",
        "bundle_id",
        "bundle_plan_id",
        "integrator_id",
        "neighborhood_bundle",
        "unit_system",
        "state",
        "payload_id",
        "arrays",
    }
)
# A dynamics state flattens its neighborhood bundle state (per-source epochs,
# per-role views, evidence and graph schedules), typed controllers, ledger and
# term states into one array per leaf.
_RUNTIME_LEAF_LIMIT = 4_096
_CHECKPOINT_LIMITS = dataclasses.replace(
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    max_members=_RUNTIME_LEAF_LIMIT + 1,
    max_manifest_bytes=16_777_216,
    max_central_directory_bytes=4_194_304,
)
# A restart carries at most this many distinct native model artifacts (each
# within the standalone artifact bounds) bound by at most this many learned
# terms, each with a bounded number of conditioning and random-key leaves.
_RESTART_ARTIFACT_LIMIT = 8
_RESTART_TERM_LIMIT = 32
_TERM_LEAF_LIMIT = 64
_ARTIFACT_LEAF_LIMIT = ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_members - 1
_RESTART_LIMITS = dataclasses.replace(
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    max_container_bytes=(_RESTART_ARTIFACT_LIMIT + 1)
    * ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_container_bytes,
    max_aggregate_bytes=(_RESTART_ARTIFACT_LIMIT + 1)
    * ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_aggregate_bytes,
    max_total_array_elements=(_RESTART_ARTIFACT_LIMIT + 1)
    * ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_total_array_elements,
    max_members=_RESTART_ARTIFACT_LIMIT * _ARTIFACT_LEAF_LIMIT
    + _RESTART_TERM_LIMIT * _TERM_LEAF_LIMIT
    + _RUNTIME_LEAF_LIMIT
    + 1,
    max_manifest_bytes=(_RESTART_ARTIFACT_LIMIT + 1)
    * ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_manifest_bytes,
    max_central_directory_bytes=(_RESTART_ARTIFACT_LIMIT + 2)
    * ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_central_directory_bytes,
    max_manifest_nesting=ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_manifest_nesting + 4,
)


@final
class AtomisticCheckpointPlan(StrictModule, NonTrainableState):
    """Identity binding of one dynamics runtime for checkpoint and restart.

    The checkpoint identity binds the prepared dynamics, the thermodynamic
    table, the system, the executed Hamiltonian and its underlying program,
    the prepared neighborhood bundle (plan and capacity levels), the
    integrator and the optional ``scope_id``.
    """

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
        identities = _runtime_identities(dynamics, thermodynamic)
        self.scope_id = scope_id
        self.dynamics = dynamics
        self.thermodynamic = thermodynamic
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "atomistic-checkpoint-plan",
                **identities,
                **({} if scope_id is None else {"scope_id": scope_id}),
            }
        )


@final
class AtomisticCheckpoint(StrictModule):
    state: AtomisticDynamicsState
    units: AtomisticUnitSystem
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


def _runtime_identities(
    dynamics: PreparedAtomisticDynamics,
    thermodynamic: PreparedThermodynamicStateTable,
    /,
) -> dict[str, str]:
    return {
        "prepared_dynamics_id": dynamics.prepared_id,
        "thermodynamic_table_id": thermodynamic.table_id,
        "system_id": dynamics.system.prepared_id,
        "hamiltonian_id": dynamics.potential.prepared_id,
        "program_id": _underlying_program(dynamics.potential).prepared_id,
        "bundle_id": dynamics.bundle.prepared_id,
        "bundle_plan_id": dynamics.bundle.plan.plan_id,
        "integrator_id": dynamics.integrator.plan_id,
    }


def _plan_identities(plan: AtomisticCheckpointPlan, /) -> dict[str, str]:
    identities = {
        "checkpoint_id": plan.checkpoint_id,
        **_runtime_identities(plan.dynamics, plan.thermodynamic),
    }
    if plan.scope_id is not None:
        identities["scope_id"] = plan.scope_id
    return identities


def _bundle_record(bundle: PreparedAtomisticNeighborhoodBundle, /) -> dict[str, Any]:
    return {
        "plan": bundle.plan.to_dict(),
        "source_levels": list(bundle.source_levels),
        "role_levels": list(bundle.role_levels),
    }


def _levels(value: object, name: str, /) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(level, bool) or not isinstance(level, int) or level < 0
        for level in value
    ):
        raise ValueError(f"Checkpoint neighborhood bundle {name} are not canonical.")
    return tuple(value)


@final
class AtomisticCheckpointResources(StrictModule, NonTrainableState):
    """Persisted neighborhood resource binding of a dynamics checkpoint.

    ``plan`` is the bundle plan replayed from its pickle-free recipe through
    the owning constructors; ``source_levels``/``role_levels`` are the
    declared capacity levels the saved state was accepted under (grown by
    capacity replay or not) and ``bundle_id`` the recorded prepared identity.
    """

    plan: AtomisticNeighborhoodBundlePlan
    source_levels: tuple[int, ...] = eqx.field(static=True)
    role_levels: tuple[int, ...] = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: AtomisticNeighborhoodBundlePlan,
        source_levels: tuple[int, ...],
        role_levels: tuple[int, ...],
        bundle_id: str,
        /,
    ) -> None:
        if len(source_levels) != len(plan.sources) or len(role_levels) != len(
            plan.requirements
        ):
            raise ValueError("Neighborhood capacity levels do not match the bundle plan.")
        if not isinstance(bundle_id, str) or not bundle_id:
            raise ValueError("bundle_id must be a nonempty string.")
        self.plan = plan
        self.source_levels = tuple(int(level) for level in source_levels)
        self.role_levels = tuple(int(level) for level in role_levels)
        self.bundle_id = bundle_id

    def bind(
        self, potential: AbstractPreparedAtomisticHamiltonian, /
    ) -> AbstractPreparedAtomisticHamiltonian:
        """Rebind ``potential`` to the recorded capacity levels of this checkpoint.

        ``potential`` is a fresh preparation of the same scientific
        Hamiltonian over the same bundle plan (any capacity level); its
        system, cell and graph schedule plans are retained.  The rebuilt
        bundle must reproduce ``bundle_id`` exactly.
        """
        if not isinstance(potential, AbstractPreparedAtomisticHamiltonian):
            raise TypeError("potential must be a prepared atomistic Hamiltonian.")
        bundle = potential.neighborhood_bundle
        if bundle.plan.plan_id != self.plan.plan_id:
            raise ValueError(
                "The Hamiltonian binds another neighborhood bundle plan than the "
                "checkpoint resources."
            )
        if bundle.prepared_id == self.bundle_id:
            return potential
        rebuilt = PreparedAtomisticNeighborhoodBundle(
            self.plan,
            bundle.system,
            graph_plans=bundle.graph_plans,
            source_levels=self.source_levels,
            role_levels=self.role_levels,
        )
        if rebuilt.prepared_id != self.bundle_id:
            raise ValueError(
                "The rebuilt neighborhood bundle does not reproduce the checkpoint "
                "binding: its system, cell or graph schedule plans differ."
            )
        return potential.with_neighborhood_bundle(rebuilt)


def _resources(manifest: Mapping[str, Any], /) -> AtomisticCheckpointResources:
    record = manifest["neighborhood_bundle"]
    if not isinstance(record, Mapping) or set(record) != _BUNDLE_FIELDS:
        raise ValueError("Checkpoint neighborhood bundle record is not canonical.")
    plan = AtomisticNeighborhoodBundlePlan.from_dict(record["plan"])
    if plan.plan_id != manifest["bundle_plan_id"]:
        raise ValueError(
            "Checkpoint neighborhood bundle recipe differs from its identity."
        )
    bundle_id = manifest["bundle_id"]
    if not isinstance(bundle_id, str):
        raise ValueError("Checkpoint bundle_id is not canonical.")
    return AtomisticCheckpointResources(
        plan,
        _levels(record["source_levels"], "source levels"),
        _levels(record["role_levels"], "role levels"),
        bundle_id,
    )


def _runtime_payload_id(
    checkpoint_id: str,
    unit_system_id: str,
    state: AtomisticDynamicsState,
    specification: Mapping[str, Any],
    bundle: Mapping[str, Any],
    arrays: Mapping[str, Any],
    /,
    **binding: str,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-checkpoint-payload",
            "checkpoint": checkpoint_id,
            "unit_system": unit_system_id,
            "time": float(state.time),
            "step": int(state.step_index),
            "configuration_revision": int(state.configuration_revision),
            "neighborhood_bundle": dict(bundle),
            "state": dict(specification),
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
    dynamics = plan.dynamics
    if (
        state.prepared_dynamics_id != dynamics.prepared_id
        or state.thermodynamic_table_id != plan.thermodynamic.table_id
    ):
        raise ValueError(f"Checkpoint {role} belongs to another dynamics runtime.")
    if (
        state.neighborhoods.bundle_id != dynamics.bundle.prepared_id
        or state.force.bundle_id != dynamics.bundle.prepared_id
        or state.force.program_id != dynamics.potential.prepared_id
    ):
        raise ValueError(
            f"Checkpoint {role} neighborhoods or force cache belong to another "
            "Hamiltonian or neighborhood bundle binding; rebind and re-prime it first."
        )


def _canonical_specification(template: AtomisticDynamicsState, /) -> dict[str, object]:
    return pack_array_tree(_RUNTIME_PREFIX, template, {})


def _runtime_manifest(
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    format: str,
    kind: str,
    arrays: dict[str, object],
    /,
) -> dict[str, Any]:
    specification = pack_array_tree(_RUNTIME_PREFIX, state, arrays)
    return {
        "format": format,
        "kind": kind,
        **_plan_identities(plan),
        "neighborhood_bundle": _bundle_record(plan.dynamics.bundle),
        "unit_system": plan.dynamics.system.plan.units.to_dict(),
        "state": specification,
    }


def _published(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    manifest: dict[str, Any],
    arrays: dict[str, object],
    limits: Any,
    /,
    **binding: str,
) -> AtomisticCheckpoint:
    units = plan.dynamics.system.plan.units
    payload_id = _runtime_payload_id(
        plan.checkpoint_id,
        units.unit_system_id,
        state,
        manifest["state"],
        manifest["neighborhood_bundle"],
        {
            name: value
            for name, value in arrays.items()
            if name.startswith(f"{_RUNTIME_PREFIX}/")
        },
        **binding,
    )
    write_array_archive(
        path,
        manifest={**manifest, "payload_id": payload_id},
        limits=limits,
        arrays=arrays,
    )
    return AtomisticCheckpoint(state, units, payload_id, plan.checkpoint_id)


def write_atomistic_checkpoint(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    """Persist dynamics runtime state for an explicitly matching prepared model.

    The checkpoint holds the complete accepted state (kinematics, cell,
    neighborhood bundle state, force cache, typed controllers, ledger, RNG,
    counters and configuration revision) and the neighborhood bundle recipe
    with its capacity levels, but no model: restart requires recreating the
    identical prepared Hamiltonian, rebound with
    ``read_atomistic_checkpoint_resources(path).bind`` when the saved run grew
    its capacities. Use ``write_atomistic_restart`` for a portable bundle
    carrying the native model artifacts and learned term bindings.
    """

    _require_runtime(plan, state, "state")
    arrays: dict[str, object] = {}
    manifest = _runtime_manifest(
        plan, state, _CHECKPOINT_FORMAT, _CHECKPOINT_KIND, arrays
    )
    return _published(path, plan, state, manifest, arrays, _CHECKPOINT_LIMITS)


def _restored_runtime(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
    **binding: str,
) -> AtomisticCheckpoint:
    bundle = plan.dynamics.bundle
    if manifest["bundle_plan_id"] != bundle.plan.plan_id:
        raise ValueError(
            "Atomistic checkpoint was written for another neighborhood bundle plan."
        )
    if manifest["bundle_id"] != bundle.prepared_id:
        raise ValueError(
            "Atomistic checkpoint state belongs to another neighborhood bundle binding "
            "(capacity levels, system or graph schedules); rebuild the Hamiltonian "
            "with read_atomistic_checkpoint_resources(path).bind(potential)."
        )
    units = AtomisticUnitSystem.from_dict(manifest["unit_system"])
    if units.unit_system_id != plan.dynamics.system.plan.units.unit_system_id:
        raise ValueError("Atomistic checkpoint complete unit descriptor is incompatible.")
    for name, expected_value in _plan_identities(plan).items():
        if manifest[name] != expected_value:
            raise ValueError(f"Atomistic checkpoint {name} does not match the runtime.")
    resources = _resources(manifest)
    if (
        resources.source_levels != bundle.source_levels
        or resources.role_levels != bundle.role_levels
    ):
        raise ValueError(
            "Atomistic checkpoint capacity levels differ from its bundle identity."
        )
    if manifest["state"] != _canonical_specification(template):
        raise ValueError(
            "Atomistic checkpoint state recipe is not the canonical recipe of the "
            "runtime template."
        )
    state = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("Checkpoint did not reconstruct AtomisticDynamicsState.")
    payload_id = manifest["payload_id"]
    if not isinstance(payload_id, str) or not payload_id:
        raise ValueError("Checkpoint payload_id is empty.")
    expected_payload_id = _runtime_payload_id(
        plan.checkpoint_id,
        units.unit_system_id,
        state,
        manifest["state"],
        manifest["neighborhood_bundle"],
        arrays,
        **binding,
    )
    if payload_id != expected_payload_id:
        raise ValueError("Atomistic checkpoint payload identity is corrupt.")
    return AtomisticCheckpoint(state, units, payload_id, plan.checkpoint_id)


def _expected_fields(plan: AtomisticCheckpointPlan, /, *extra: str) -> set[str]:
    expected = set(_RUNTIME_FIELDS) | set(extra)
    if plan.scope_id is not None:
        expected.add("scope_id")
    return expected


def read_atomistic_checkpoint(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    """Restore a dynamics checkpoint against a template of the same runtime.

    ``template`` must come from ``plan.dynamics.initialize_state`` of the
    rebuilt runtime: which controllers exist and the bundle state structure
    depend on the method and the bundle binding. The archived state recipe
    must equal the template's canonical recipe exactly; any other layout
    refuses. A state written under another bundle binding refuses with the
    instruction to rebind via ``read_atomistic_checkpoint_resources``.
    """

    _require_runtime(plan, template, "template")
    manifest, arrays = read_array_archive(path, limits=_CHECKPOINT_LIMITS)
    if set(manifest) != _expected_fields(plan):
        raise ValueError(
            "Atomistic checkpoint manifest is not the canonical current format."
        )
    if manifest["format"] != _CHECKPOINT_FORMAT or manifest["kind"] != _CHECKPOINT_KIND:
        raise ValueError("File is not an atomistic dynamics checkpoint.")
    return _restored_runtime(manifest, arrays, plan, template)


def read_atomistic_checkpoint_resources(
    path: str | Path, /
) -> AtomisticCheckpointResources:
    """Read the neighborhood resource binding of a checkpoint or restart.

    A fresh process prepares its Hamiltonian over ``resources.plan`` and
    rebinds it with ``resources.bind(potential)`` before building the
    dynamics plan, so states accepted under grown capacities resume exactly.
    """

    manifest, _ = read_array_archive(path, limits=_RESTART_LIMITS)
    if (manifest.get("format"), manifest.get("kind")) not in {
        (_CHECKPOINT_FORMAT, _CHECKPOINT_KIND),
        (_RESTART_FORMAT, _RESTART_KIND),
    } or not {"neighborhood_bundle", "bundle_id", "bundle_plan_id"} <= set(manifest):
        raise ValueError("File is not an atomistic dynamics checkpoint or restart.")
    return _resources(manifest)


@final
@dataclasses.dataclass(frozen=True, slots=True)
class AtomisticRestartModelBinding:
    """One learned term of a restart and the native artifact it executes.

    ``artifact`` is the restored, fully validated model artifact. ``term`` is
    rebuilt through ``LearnedGraphPotentialTerm`` over exactly
    ``artifact.model`` with the recorded name, force group, periodic
    admission, bound conditioning (specification, ports, units, values,
    optional absence and default-use evidence), typed random key and
    observable request, and reproduces its recorded term identity. Terms whose
    artifacts share one full artifact identity share one restored model.
    """

    artifact: AtomisticModelArtifact
    term: LearnedGraphPotentialTerm

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AtomisticModelArtifact):
            raise TypeError("artifact must be an AtomisticModelArtifact.")
        if not isinstance(self.term, LearnedGraphPotentialTerm):
            raise TypeError("term must be a LearnedGraphPotentialTerm.")
        if self.term.potential is not self.artifact.model:
            raise ValueError(
                "A restart binding term must execute exactly its artifact model."
            )

    @property
    def binding_id(self) -> str:
        """Identity of the learned term together with its full artifact identity."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-restart-model-binding",
                "term": self.term.term_id,
                "artifact": self.artifact.manifest.artifact_id,
            }
        )


def _learned_terms(
    plan: AtomisticCheckpointPlan, /
) -> dict[str, LearnedGraphPotentialTerm]:
    """Every learned term the prepared Hamiltonian executes, in term-name order."""
    program = _underlying_program(plan.dynamics.potential)
    if not isinstance(program, PreparedAtomisticPotentialProgram):
        return {}
    terms: dict[str, LearnedGraphPotentialTerm] = {}
    for term in program.terms:
        if isinstance(term, PreparedLearnedGraphPotentialTerm):
            if term.name in terms:
                raise ValueError(f"Learned term name {term.name!r} is not unique.")
            terms[term.name] = term.plan
    return {name: terms[name] for name in sorted(terms)}


def _term_options[T](
    values: Mapping[str, T] | None,
    terms: Mapping[str, LearnedGraphPotentialTerm],
    role: str,
    /,
) -> dict[str, T]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{role} must be a mapping keyed by learned term name.")
    unknown = sorted(repr(name) for name in set(values).difference(terms))
    if unknown:
        raise ValueError(
            f"{role} name no learned term executed by the dynamics: {', '.join(unknown)}."
        )
    return dict(values)


def _namespace(prefix: str, index: int, /) -> str:
    return f"{prefix}/{index:04d}"


def _conditioning_member(index: int, /) -> str:
    return f"{_CONDITIONING_MEMBER}/{index:06d}"


def _artifact_identity(artifact: AtomisticModelArtifact, /) -> dict[str, str]:
    manifest = artifact.manifest
    return {
        "architecture_id": manifest.architecture_id,
        "method_id": manifest.method_id,
        "structure_id": manifest.structure_id,
        "semantic_id": manifest.semantic_id,
        "numeric_revision_id": manifest.numeric_revision.revision_id,
        "content_id": manifest.content_id,
    }


def _term_record(
    term: LearnedGraphPotentialTerm, artifact: int, /
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Host record and term-relative arrays of one learned term binding."""
    arrays: dict[str, np.ndarray] = {}
    conditioning = None
    if term.conditioning is not None:
        conditioning, values = term.conditioning.to_record()
        arrays.update(
            {_conditioning_member(index): value for index, value in enumerate(values)}
        )
    key = None
    if term.key is not None:
        key = model_structure_recipe(term.key)
        arrays.update(pack_model_array_tree(term.key, key, prefix=_KEY_MEMBER))
    if len(arrays) > _TERM_LEAF_LIMIT:
        raise ValueError(
            f"Learned term {term.name!r} binds more than {_TERM_LEAF_LIMIT} "
            "conditioning and random-key arrays."
        )
    record = {
        "name": term.name,
        "artifact": artifact,
        "force_group": term.force_group,
        "allow_periodic": term.allow_periodic,
        "conditioning": conditioning,
        "key": key,
        "request": None if term.request is None else term.request.to_dict(),
        "term_id": term.term_id,
    }
    return record, arrays


def _expected_term_members(record: Mapping[str, Any], /) -> set[str]:
    """Exact term-relative array inventory declared by one term record."""
    members: set[str] = set()
    conditioning = record["conditioning"]
    if conditioning is not None:
        bindings = (
            conditioning.get("bindings") if isinstance(conditioning, dict) else None
        )
        if not isinstance(bindings, list):
            raise ValueError("Restart term conditioning record is not canonical.")
        count = sum(1 for binding in bindings if binding != "absent")
        members.update(_conditioning_member(index) for index in range(count))
    key = record["key"]
    if key is not None:
        if not isinstance(key, dict) or key.get("kind") not in {"prng_key", "array"}:
            raise ValueError("Restart term random key record is not canonical.")
        try:
            inventory = model_recipe_array_inventory(key, prefix=_KEY_MEMBER)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Restart term random key recipe is invalid.") from error
        members.update(entry.name for entry in inventory)
    if len(members) > _TERM_LEAF_LIMIT:
        raise ValueError("Restart term binds more arrays than its bound admits.")
    return members


def _rebuilt_binding(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    artifact: AtomisticModelArtifact,
    /,
) -> AtomisticRestartModelBinding:
    """Rebuild one learned term binding through its constructors and verify it.

    ``arrays`` must already equal the record's exact term-relative inventory.
    """
    name = record["name"]
    prefix = f"{_CONDITIONING_MEMBER}/"
    try:
        conditioning = (
            None
            if record["conditioning"] is None
            else AtomisticConditioning.from_record(
                record["conditioning"],
                tuple(
                    arrays[member]
                    for member in sorted(arrays)
                    if member.startswith(prefix)
                ),
            )
        )
        key = (
            None
            if record["key"] is None
            else model_from_array_recipe(
                record["key"],
                {
                    member: value
                    for member, value in arrays.items()
                    if member.startswith(f"{_KEY_MEMBER}/")
                },
                prefix=_KEY_MEMBER,
            )
        )
        request = (
            None
            if record["request"] is None
            else AtomisticObservableRequest.from_dict(record["request"])
        )
        term = LearnedGraphPotentialTerm(
            artifact.model,
            name=name,
            force_group=record["force_group"],
            allow_periodic=record["allow_periodic"],
            conditioning=conditioning,
            key=key,
            request=request,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AtomisticModelArtifactError(
            f"Restart learned term {name!r} binding is invalid."
        ) from error
    if term.name != name or term.term_id != record["term_id"]:
        raise AtomisticModelArtifactError(
            f"Restart learned term {name!r} does not reproduce its recorded identity."
        )
    return AtomisticRestartModelBinding(artifact, term)


def _models_binding_id(bindings: Mapping[str, AtomisticRestartModelBinding], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-restart-models",
            "bindings": {name: binding.binding_id for name, binding in bindings.items()},
        }
    )


def _require_model_bindings(
    plan: AtomisticCheckpointPlan,
    bindings: Mapping[str, AtomisticRestartModelBinding],
    /,
) -> None:
    """Require the executed learned terms to be exactly the bundled bindings."""
    terms = _learned_terms(plan)
    if not terms:
        raise ValueError("A model restart requires dynamics executing a learned model.")
    if set(terms) != set(bindings):
        raise AtomisticModelArtifactError(
            "The dynamics execute learned terms that differ from the bundled "
            f"bindings: unbundled {sorted(set(terms) - set(bindings))!r}, "
            f"not executed {sorted(set(bindings) - set(terms))!r}."
        )
    for name, term in terms.items():
        binding = bindings[name]
        if atomistic_model_identity(term.potential) != _artifact_identity(
            binding.artifact
        ):
            raise AtomisticModelArtifactError(
                f"Learned term {name!r} executes a model that differs from the "
                "bundled artifact."
            )
        if term.term_id != binding.term.term_id:
            raise AtomisticModelArtifactError(
                f"Learned term {name!r} differs from the bundled binding: its force "
                "group, periodic admission, conditioning, random key or observable "
                "request changed."
            )


def write_atomistic_restart(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    /,
    *,
    sources: Mapping[str, AtomisticSourceProvenance] | None = None,
    licenses: Mapping[str, Sequence[str]] | None = None,
) -> AtomisticCheckpoint:
    """Write a portable restart: every executed native model plus dynamics state.

    The learned terms are derived from the prepared Hamiltonian of ``plan``
    (through any controlled Hamiltonian); there is no separate model argument.
    Each term's exact raw or prepared model is archived as a native model
    artifact and its binding (name, force group, periodic admission, bound
    conditioning, typed random key and observable request) as a pickle-free
    record that is rebuilt and identity-checked before anything is written.
    ``sources`` and ``licenses`` are keyed by learned term name and bind the
    conversion provenance and rights identifiers of that term's model; a key
    naming no executed learned term refuses, and a source bound to another
    model refuses. Terms whose artifacts agree in full identity (model,
    preparation, source and licenses) share one archived artifact; distinct
    revisions or preparations never alias.

    A fresh process restores the bindings with ``read_atomistic_restart_models``,
    prepares its Hamiltonian over the restored terms and the recorded bundle
    plan, rebinds it with ``read_atomistic_checkpoint_resources(path).bind``,
    rebuilds the dynamics and resumes with ``read_atomistic_restart``. Compiled
    caches are not part of restart correctness. The restored state is exact;
    continued steps equal the uninterrupted trajectory bitwise only when the
    executed graph plans select ordered streamed accumulation and the resuming
    process uses the same device type, runtime versions, and a fixed compiler
    profile (on GPUs, XLA autotuning disabled).
    """

    _require_runtime(plan, state, "state")
    terms = _learned_terms(plan)
    if not terms:
        raise ValueError("A model restart requires dynamics executing a learned model.")
    if len(terms) > _RESTART_TERM_LIMIT:
        raise ValueError(f"A restart binds at most {_RESTART_TERM_LIMIT} learned terms.")
    source_options = _term_options(sources, terms, "sources")
    license_options = _term_options(licenses, terms, "licenses")
    arrays: dict[str, object] = {}
    sections: list[dict[str, Any]] = []
    artifacts: dict[str, tuple[int, AtomisticModelArtifact]] = {}
    records: list[dict[str, Any]] = []
    bindings: dict[str, AtomisticRestartModelBinding] = {}
    for position, (name, term) in enumerate(terms.items()):
        section, leaves, manifest = model_artifact_section(
            term.potential,
            source=source_options.get(name),
            licenses=license_options.get(name, ()),
        )
        entry = artifacts.get(manifest.artifact_id)
        if entry is None:
            if len(sections) == _RESTART_ARTIFACT_LIMIT:
                raise ValueError(
                    f"A restart carries at most {_RESTART_ARTIFACT_LIMIT} distinct "
                    "model artifacts."
                )
            entry = (
                len(sections),
                AtomisticModelArtifact(model=term.potential, manifest=manifest),
            )
            artifacts[manifest.artifact_id] = entry
            sections.append(section)
            namespace = _namespace(_ARTIFACT_PREFIX, entry[0])
            arrays.update(
                {f"{namespace}/{leaf}": value for leaf, value in leaves.items()}
            )
        index, artifact = entry
        record, term_arrays = _term_record(term, index)
        binding = _rebuilt_binding(record, term_arrays, artifact)
        record["binding_id"] = binding.binding_id
        records.append(record)
        bindings[name] = binding
        namespace = _namespace(_TERM_PREFIX, position)
        arrays.update(
            {f"{namespace}/{member}": value for member, value in term_arrays.items()}
        )
    manifest = _runtime_manifest(plan, state, _RESTART_FORMAT, _RESTART_KIND, arrays)
    return _published(
        path,
        plan,
        state,
        {**manifest, "models": {"artifacts": sections, "terms": records}},
        arrays,
        _RESTART_LIMITS,
        models=_models_binding_id(bindings),
    )


def _partitioned_arrays(
    arrays: Mapping[str, np.ndarray], artifact_count: int, term_count: int, /
) -> tuple[
    list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict[str, np.ndarray]
]:
    """Split archive arrays into artifact, term and runtime namespaces exactly."""
    artifacts: list[dict[str, np.ndarray]] = [{} for _ in range(artifact_count)]
    terms: list[dict[str, np.ndarray]] = [{} for _ in range(term_count)]
    runtime: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        namespace, _, rest = name.partition("/")
        if namespace == _RUNTIME_PREFIX and rest:
            runtime[name] = value
            continue
        position, _, member = rest.partition("/")
        index = (
            int(position)
            if len(position) == 4 and position.isascii() and position.isdigit()
            else None
        )
        if (
            namespace == _ARTIFACT_PREFIX
            and index is not None
            and index < artifact_count
            and member.startswith(_MODEL_LEAF_PREFIX)
        ):
            artifacts[index][member] = value
        elif (
            namespace == _TERM_PREFIX
            and index is not None
            and index < term_count
            and member
        ):
            terms[index][member] = value
        else:
            raise ValueError("Atomistic restart contains arrays outside its namespaces.")
    return artifacts, terms, runtime


def _canonical_term_records(
    records: object, artifact_count: int, /
) -> list[dict[str, Any]]:
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= _RESTART_TERM_LIMIT
        or any(
            not isinstance(record, dict)
            or set(record) != _TERM_RECORD_FIELDS
            or not isinstance(record["name"], str)
            or type(record["artifact"]) is not int
            or type(record["force_group"]) is not int
            or type(record["allow_periodic"]) is not bool
            or not isinstance(record["term_id"], str)
            or not isinstance(record["binding_id"], str)
            or not record["term_id"]
            or not record["binding_id"]
            for record in records
        )
    ):
        raise ValueError("Atomistic restart learned term records are not canonical.")
    names = [record["name"] for record in records]
    if names != sorted(set(names)):
        raise ValueError(
            "Atomistic restart learned terms are not unique and in term-name order."
        )
    references = [record["artifact"] for record in records]
    if list(dict.fromkeys(references)) != list(range(artifact_count)):
        raise ValueError(
            "Atomistic restart artifacts are not each referenced in canonical order."
        )
    return records


def _restart_payload(
    path: str | Path, /
) -> tuple[
    dict[str, Any], dict[str, AtomisticRestartModelBinding], dict[str, np.ndarray]
]:
    """Admit a restart archive and rebuild every learned term binding.

    The complete field, namespace and array inventory is checked before any
    model is restored; every artifact and term identity is recomputed.
    """
    manifest, arrays = read_array_archive(path, limits=_RESTART_LIMITS)
    if manifest.get("format") != _RESTART_FORMAT or manifest.get("kind") != _RESTART_KIND:
        raise ValueError("File is not an atomistic dynamics restart.")
    if set(manifest) - {"scope_id"} != _RUNTIME_FIELDS | {"models"}:
        raise ValueError("Atomistic restart manifest is not the canonical format.")
    models = manifest["models"]
    if not isinstance(models, dict) or set(models) != _MODELS_FIELDS:
        raise ValueError("Atomistic restart model bindings are not canonical.")
    sections = models["artifacts"]
    if (
        not isinstance(sections, list)
        or not 1 <= len(sections) <= _RESTART_ARTIFACT_LIMIT
    ):
        raise ValueError("Atomistic restart model artifacts are not canonical.")
    records = _canonical_term_records(models["terms"], len(sections))
    artifact_arrays, term_arrays, runtime_arrays = _partitioned_arrays(
        arrays, len(sections), len(records)
    )
    if any(len(members) > _ARTIFACT_LEAF_LIMIT for members in artifact_arrays):
        raise ValueError("Atomistic restart model artifact exceeds its leaf bound.")
    for record, members in zip(records, term_arrays, strict=True):
        if set(members) != _expected_term_members(record):
            raise ValueError(
                f"Atomistic restart term {record['name']!r} arrays do not match its "
                "record inventory."
            )
    artifacts = [
        model_artifact_from_section(section, members)
        for section, members in zip(sections, artifact_arrays, strict=True)
    ]
    if len({artifact.manifest.artifact_id for artifact in artifacts}) != len(artifacts):
        raise AtomisticModelArtifactError(
            "Atomistic restart archives one model artifact more than once."
        )
    bindings: dict[str, AtomisticRestartModelBinding] = {}
    for record, members in zip(records, term_arrays, strict=True):
        binding = _rebuilt_binding(record, members, artifacts[record["artifact"]])
        if binding.binding_id != record["binding_id"]:
            raise AtomisticModelArtifactError(
                f"Restart learned term {record['name']!r} binding identity is corrupt."
            )
        bindings[record["name"]] = binding
    return manifest, bindings, runtime_arrays


def read_atomistic_restart_models(
    path: str | Path, /
) -> Mapping[str, AtomisticRestartModelBinding]:
    """Restore and fully validate every learned term binding of a restart.

    The read-only mapping is keyed by learned term name in name order. Each
    artifact is restored through its registered validator and identity check;
    each term is rebuilt through its constructors over exactly that restored
    model and must reproduce its recorded term and binding identities.
    """

    return MappingProxyType(_restart_payload(path)[1])


def read_atomistic_restart(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    """Resume dynamics state after verifying the bundled bindings execute in ``plan``.

    The learned terms executed by ``plan`` must be exactly the bundled
    bindings: no missing or additional term, every term running its bundled
    model (architecture, structure, parameters and fixed leaves) with the same
    force group, periodic admission, conditioning, random key and request.
    ``plan`` must also be rebuilt over the recorded neighborhood binding; any
    altered system, Hamiltonian, bundle plan or capacity levels, integrator,
    thermodynamic table, scope or state recipe refuses before state is
    returned.
    """

    _require_runtime(plan, template, "template")
    manifest, bindings, runtime_arrays = _restart_payload(path)
    if set(manifest) != _expected_fields(plan, "models"):
        raise ValueError("Atomistic restart manifest is not the canonical format.")
    _require_model_bindings(plan, bindings)
    return _restored_runtime(
        manifest,
        runtime_arrays,
        plan,
        template,
        models=_models_binding_id(bindings),
    )


# ------------------------------------------------------------- training restart

_TRAINING_RESTART_FORMAT = "phydrax-atomistic-training-restart"
_TRAINING_RECEIPT_FILE = "restart.json"
_TRAINING_MODEL_FILE = "model.phydrax"
_TRAINING_BASE_FILE = "base.phydrax"
_TRAINING_STATE_DIRECTORY = "training"
_TRAINING_STATE_MANIFEST = f"{_TRAINING_STATE_DIRECTORY}/manifest.json"
_TRAINING_IDENTITY_FIELDS = (
    "model_artifact_id",
    "result_id",
    "problem_id",
    "execution_id",
    "policy_id",
    "continuation_id",
    "reduction_id",
    "capabilities_id",
    "normalization_id",
    "training_checkpoint_id",
    "cursor_id",
)
_TRAINING_RECEIPT_FIELDS = frozenset(
    {"format", "members", "base_artifact_id", *_TRAINING_IDENTITY_FIELDS}
)
_TRAINING_RECEIPT_BYTES = 1_048_576
_TRAINING_STATE_BYTES = DEFAULT_ARRAY_ARCHIVE_LIMITS.max_aggregate_bytes
_TRAINING_MANIFEST_BYTES = 16 * 1024 * 1024
# Receipt, model and base artifacts, training directory, its manifest and its
# state file (entries include the directory).
_TRAINING_RESTART_LIMITS = ResourceSetLimits(
    max_total_bytes=2 * ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_container_bytes
    + _TRAINING_STATE_BYTES
    + _TRAINING_MANIFEST_BYTES
    + _TRAINING_RECEIPT_BYTES,
    max_member_bytes=max(
        ATOMISTIC_MODEL_ARTIFACT_LIMITS.max_container_bytes, _TRAINING_STATE_BYTES
    ),
    max_members=6,
    max_depth=2,
)


def _training_identities(
    result: AtomisticTrainingResult, model_artifact_id: str, /
) -> dict[str, str]:
    return {
        "model_artifact_id": model_artifact_id,
        "result_id": result.result_id,
        "problem_id": result.problem_id,
        "execution_id": result.execution_id,
        "policy_id": result.policy_id,
        "continuation_id": result.continuation_id,
        "reduction_id": result.reduction_id,
        "capabilities_id": result.capabilities_id,
        "normalization_id": result.normalization.normalization_id,
        "training_checkpoint_id": result.training_checkpoint_id,
        "cursor_id": result.cursor.cursor_id,
    }


def _staged_members(staging: Path, limits: ResourceSetLimits, /) -> dict[str, bytes]:
    with open_bounded_resource_set(
        staging.name, trusted_root=staging.parent, limits=limits
    ) as staged:
        return {
            member.relative_path: staged.read_member(member.relative_path)
            for member in staged.manifest.members
        }


def _training_restart_members(
    result: AtomisticTrainingResult,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    source: AtomisticSourceProvenance | None = None,
    licenses: Sequence[str] = (),
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> dict[str, bytes]:
    """Stage and validate the complete member set of one training restart.

    Returns bundle-relative member bytes including the ``restart.json``
    receipt; composite owners publish them (optionally below a prefix) in
    their own resource set and restore them with
    ``_restore_training_restart_members``.
    """

    if not isinstance(result, AtomisticTrainingResult):
        raise TypeError("result must be an AtomisticTrainingResult.")
    if not isinstance(policy, AtomisticTrainingPolicy):
        raise TypeError("policy must be an AtomisticTrainingPolicy.")
    with TemporaryDirectory(prefix="phydrax-training-restart-") as scratch:
        staging = Path(scratch).resolve() / "bundle"
        staging.mkdir(mode=0o700)
        model = write_atomistic_model_artifact(
            staging / _TRAINING_MODEL_FILE,
            result.potential,
            source=source,
            licenses=licenses,
        )
        base_artifact_id = None
        if policy.adaptation is not None:
            base_artifact_id = write_atomistic_model_artifact(
                staging / _TRAINING_BASE_FILE,
                result.trained_tree.base_potential(),
                source=source,
                licenses=licenses,
            ).artifact_id
        write_atomistic_training_checkpoint(
            staging / _TRAINING_STATE_DIRECTORY,
            result,
            problem,
            policy,
            reduction=reduction,
        )
        members = _staged_members(staging, _TRAINING_RESTART_LIMITS)
    receipt = {
        "format": _TRAINING_RESTART_FORMAT,
        "members": {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        },
        "base_artifact_id": base_artifact_id,
        **_training_identities(result, model.artifact_id),
    }
    members[_TRAINING_RECEIPT_FILE] = (
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return members


def write_atomistic_training_restart(
    directory: str | Path,
    result: AtomisticTrainingResult,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    source: AtomisticSourceProvenance | None = None,
    licenses: Sequence[str] = (),
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> Path:
    """Atomically publish a fresh-process training continuation for one model.

    The bundle directory holds the native artifact of the merged
    ``result.potential``, the training owner's canonical kernel checkpoint
    (parameters, optimizer and plateau state, EMA target, open accumulation
    window, root key, loader cursor, selection, best tree and histories) and a
    ``restart.json`` receipt binding every member digest and the model,
    result, problem, execution, policy, continuation, reduction, capabilities,
    normalization, kernel and loader-cursor identities. Under
    ``policy.adaptation`` the merged model is not an admissible restore
    template (the kernel's FIXED lane binds the frozen base), so the adapted
    base is also stored as ``base.phydrax`` and its artifact identity
    recorded. Every component is written and validated in private staging,
    then the complete bundle replaces ``directory`` in one atomic exchange: a
    failed or interrupted publication leaves any previous bundle readable and
    unchanged. The training data recipe is not stored: continuation rebuilds
    the same ``AtomisticTrainingProblem``. The restored state is exact;
    resumed updates are bitwise only under the ordered-accumulation and
    fixed-profile preconditions of ``fit_atomistic_potential``.
    """

    members = _training_restart_members(
        result, problem, policy, source=source, licenses=licenses, reduction=reduction
    )
    destination = Path(directory)
    publish_resource_set(
        destination, members, limits=_TRAINING_RESTART_LIMITS, mode="atomic_replace"
    )
    return destination


def _training_receipt(members: Mapping[str, bytes], /) -> dict[str, Any]:
    if _TRAINING_RECEIPT_FILE not in members:
        raise ValueError(
            "Directory is not a canonical atomistic training restart bundle."
        )
    data = members[_TRAINING_RECEIPT_FILE]
    if len(data) > _TRAINING_RECEIPT_BYTES:
        raise ValueError("Atomistic training restart receipt exceeds its bound.")
    receipt = decode_json_resource(
        bounded_resource_from_bytes(
            data,
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
            for name in _TRAINING_IDENTITY_FIELDS
        )
        or not (
            receipt["base_artifact_id"] is None
            or (
                isinstance(receipt["base_artifact_id"], str)
                and receipt["base_artifact_id"]
            )
        )
        or not isinstance(receipt["members"], dict)
        or any(not isinstance(digest, str) for digest in receipt["members"].values())
    ):
        raise ValueError("Atomistic training restart receipt is not canonical.")
    digests = receipt["members"]
    states = [
        name
        for name in digests
        if name.startswith(f"{_TRAINING_STATE_DIRECTORY}/state-")
        and name.endswith(".eqx")
        and name.count("/") == 1
    ]
    canonical = {_TRAINING_MODEL_FILE, _TRAINING_STATE_MANIFEST, *states}
    if receipt["base_artifact_id"] is not None:
        canonical.add(_TRAINING_BASE_FILE)
    if len(states) != 1 or set(digests) != canonical:
        raise ValueError("Atomistic training restart receipt members are not canonical.")
    if set(members) != {_TRAINING_RECEIPT_FILE, *digests} or any(
        hashlib.sha256(members[name]).hexdigest() != digest
        for name, digest in digests.items()
    ):
        raise ValueError(
            "Atomistic training restart members differ from their publication receipt."
        )
    return receipt


def _restore_training_restart_members(
    members: Mapping[str, bytes],
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    components: Mapping[str, Any] | None = None,
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> tuple[AtomisticModelArtifact, AtomisticTrainingResult]:
    """Admit bundle-relative member bytes against their receipt and restore them."""

    receipt = _training_receipt(members)
    if (receipt["base_artifact_id"] is None) != (policy.adaptation is None):
        raise ValueError(
            "The training restart adaptation differs from the policy: an adapted "
            "continuation records its base model artifact and a plain one does not."
        )
    with TemporaryDirectory(prefix="phydrax-training-restart-") as scratch:
        admitted = Path(scratch).resolve()
        (admitted / _TRAINING_STATE_DIRECTORY).mkdir(mode=0o700)
        for name in receipt["members"]:
            (admitted / name).write_bytes(members[name])
        artifact = read_atomistic_model_artifact(admitted / _TRAINING_MODEL_FILE)
        base = None
        if receipt["base_artifact_id"] is not None:
            base = read_atomistic_model_artifact(admitted / _TRAINING_BASE_FILE)
            if base.manifest.artifact_id != receipt["base_artifact_id"]:
                raise AtomisticModelArtifactError(
                    "The training restart base artifact differs from its receipt."
                )
        result = read_atomistic_training_checkpoint(
            admitted / _TRAINING_STATE_DIRECTORY,
            artifact.model if base is None else base.model,
            problem,
            policy,
            components=components,
            reduction=reduction,
        )
    if atomistic_model_revision(result.potential).revision_id != (
        artifact.manifest.numeric_revision.revision_id
    ) or atomistic_model_identity(result.potential) != atomistic_model_identity(
        artifact.model
    ):
        raise AtomisticModelArtifactError(
            "The training continuation belongs to a different model artifact."
        )
    if base is not None and atomistic_model_identity(
        result.trained_tree.base_potential()
    ) != atomistic_model_identity(base.model):
        raise AtomisticModelArtifactError(
            "The training continuation belongs to a different adapted base model."
        )
    # The execution identity is recorded provenance of the writing run; the
    # graph plan and loader layouts may change across a continuation.
    restored = _training_identities(result, artifact.manifest.artifact_id)
    if any(
        restored[name] != receipt[name]
        for name in _TRAINING_IDENTITY_FIELDS
        if name != "execution_id"
    ):
        raise ValueError(
            "Atomistic training restart components differ from their receipt identities."
        )
    return artifact, result


def read_atomistic_training_restart(
    directory: str | Path,
    problem: AtomisticTrainingProblem,
    policy: AtomisticTrainingPolicy,
    /,
    *,
    components: Mapping[str, Any] | None = None,
    reduction: AbstractAtomisticObjectiveReduction | None = None,
) -> tuple[AtomisticModelArtifact, AtomisticTrainingResult]:
    """Restore the trained model artifact and its exact training continuation.

    The bundle is admitted as one bounded resource set whose inventory and
    member digests must equal its publication receipt; the model and training
    owners then restore exactly those admitted bytes. The restore template is
    the artifact's model, or under ``policy.adaptation`` the recorded adapted
    base; the restored merged model must equal the model artifact, the
    restored base the base artifact, and the restored model, result, problem,
    policy, continuation, reduction, capabilities, normalization, kernel and
    loader-cursor identities the receipt, so tampered, mixed or stale bundles
    refuse. Resume with ``fit_atomistic_potential(..., continuation=result)``
    and a fresh (or no) iteration session; resumed updates are bitwise only
    under its ordered-accumulation and fixed-profile preconditions.
    """

    source = Path(directory).expanduser().absolute()
    with open_bounded_resource_set(
        source.name, trusted_root=source.parent, limits=_TRAINING_RESTART_LIMITS
    ) as bundle:
        members = {
            member.relative_path: bundle.read_member(member.relative_path)
            for member in bundle.manifest.members
        }
    return _restore_training_restart_members(
        members, problem, policy, components=components, reduction=reduction
    )


__all__ = [
    "AtomisticCheckpoint",
    "AtomisticCheckpointPlan",
    "AtomisticCheckpointResources",
    "AtomisticRestartModelBinding",
    "read_atomistic_checkpoint",
    "read_atomistic_checkpoint_resources",
    "read_atomistic_restart",
    "read_atomistic_restart_models",
    "read_atomistic_training_restart",
    "write_atomistic_checkpoint",
    "write_atomistic_restart",
    "write_atomistic_training_restart",
]
