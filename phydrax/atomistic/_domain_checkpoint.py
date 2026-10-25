#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Durable continuation of distributed atomistic domain states.

Every dynamic leaf of an `AtomisticDomainState` is a continuation leaf:
owner-blocked per-atom rows (positions, momenta, masses, species, image
counts, stable/logical identities, forces, per-atom payload), replicated
controllers, energy ledger, conditioning, cell, typed RNG key (stored as key
data plus its implementation), step/time, case identity, constraint rows,
every role topology epoch (routes, relation leaves, reference geometry,
owner-epoch witness, evidence, halo plan) and the layout's owner epochs.
Named ``sections`` (global array pytrees such as a training kernel state),
``process_sections`` (per-process host pytrees such as a stage-transport
state with credits and pending packets), loader cursors and the training
distribution plan identity complete a run's continuation. One canonical
`AtomisticDomainCheckpointRecord` binds plan, runtime, model revision, run,
execution topology, structure, RNG lineage and every section schema.

- `checkpoint_domain_state` / `restore_domain_state`: in-memory host
  checkpoint of a fully addressable state with a content identity.
- `publish_domain_checkpoint`: every process publishes its canonical
  addressable shards (owner-blocked rows as authenticated row-region
  records), then the coordinator re-derives complete coverage and writes the
  single global commit (`commit_distributed_checkpoint`). Without that commit
  nothing restores.
- `restore_domain_checkpoint` (same execution topology): every leaf is
  restored bit for bit into the template's shardings, each process reading
  only the records its destinations intersect. Continued execution is
  bitwise reproducible exactly where the run's reductions are deterministic.
- `migrate_domain_checkpoint` (changed owner/process count): admitted through
  `lifecycle._restart_topology` as a tolerance-class relation; each new
  process reads one canonical row range of the source rows (region-only I/O),
  re-ingresses them with their complete continuation through
  `PreparedAtomisticDomainRuntime.ingress_process_local` (wrap in the current
  cell, migrate every leaf to its fractional owner, rebuild every role
  topology and the constraint import) and verifies the RNG lineage.
  Scientific state and RNG lineage are preserved; future floating-point
  trajectories are not promised bitwise.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import GetAttrKey, PyTreeDef
from jaxtyping import PyTree

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._execution_array import replicate_tree
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from .._jax_config import default_device
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle._chunk_repository import ArtifactRepository, ChunkEncoding
from ..lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    commit_distributed_checkpoint,
    load_distributed_checkpoint_commit,
    plan_checkpoint_region_restore,
    publish_process_checkpoint,
    read_checkpoint_region,
    restore_global_array_from_checkpoint,
)
from ..lifecycle._models import CheckpointManifest
from ..lifecycle._restart_topology import (
    admit_topology_restart,
    RestartClass,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)
from ..qualification._registry import SupportTuple
from ._domain_dynamics import AtomisticDomainConstraintRows
from ._domain_runtime import AtomisticDomainForceRecord, PreparedAtomisticDomainRuntime
from ._domain_state import AtomisticDomainState
from ._training_data import AtomisticLoaderCursor


_RECORD_KIND = "atomistic-domain-checkpoint"
_SECTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DEFAULT_REGION_BYTES = 1 << 20
# Owner-blocked per-atom continuation rows by attribute path.
_ROW_PATHS = {
    "positions": ("positions",),
    "momenta": ("momenta",),
    "masses": ("masses",),
    "species": ("species",),
    "image_counts": ("image_counts",),
    "stable_ids": ("layout", "stable_ids"),
    "active": ("layout", "active"),
    "logical_indices": ("layout", "logical_indices"),
    "forces": ("force", "forces"),
}
# Replicated leaves read on the host during migration.
_REPLICATED_PATHS = {
    "cell_vectors": ("cell_vectors",),
    "step_index": ("step_index",),
    "time": ("time",),
    "case_identity_words": ("case_identity_words",),
    "force_energy": ("force", "energy"),
    "force_strain_derivative": ("force", "strain_derivative"),
    "force_stress": ("force", "stress"),
    "force_strain_available": ("force", "strain_available"),
    "force_step": ("force", "step"),
    "force_successful": ("force", "successful"),
}
_CONSTRAINT_ROW_PATHS = {
    "row_ids": ("constraints", "rows", "stable_ids"),
    "active": ("constraints", "rows", "active"),
    "atoms": ("constraints", "atoms"),
    "components": ("constraints", "components"),
    "targets": ("constraints", "targets"),
    "lagrange": ("constraints", "lagrange"),
}
_CONSTRAINT_REPLICATED_PATHS = {
    "position_residual": ("constraints", "position_residual"),
    "velocity_residual": ("constraints", "velocity_residual"),
    "topology_epoch": ("constraints", "topology_epoch"),
}
_REPLICATED_TREES = ("replicated_payload", "controllers", "ledger", "conditioning")


class AtomisticDomainCheckpointError(ValueError):
    """A domain checkpoint does not match the runtime, template or its identity."""


class AtomisticDomainMigrationCapacityError(ValueError):
    """Changed-topology migration exceeds a destination capacity.

    Capacity refusal, not scientific failure: the source checkpoint is
    untouched and remains the last durable commit.
    """


# ---------------------------------------------------------------------------
# Leaves and the canonical record
# ---------------------------------------------------------------------------


def _attribute_path(names: Sequence[str], /) -> str:
    return jax.tree_util.keystr(tuple(GetAttrKey(name) for name in names))


def _is_key(leaf: Array, /) -> bool:
    return jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)


def _leaf_data(leaf: Array, /) -> tuple[Array, str | None]:
    """Storable data of one leaf and its key implementation (keys only)."""
    if _is_key(leaf):
        return jax.random.key_data(leaf), str(jax.random.key_impl(leaf))
    return leaf, None


def _leaf_spec(
    path: str, leaf: Array | np.ndarray, /
) -> tuple[str, tuple[int, ...], str, str | None]:
    if isinstance(leaf, jax.Array):
        data, implementation = _leaf_data(leaf)
        return (
            path,
            tuple(int(n) for n in data.shape),
            np.dtype(data.dtype).str,
            implementation,
        )
    array = np.asarray(leaf)
    if array.dtype.hasobject:
        raise TypeError("Checkpoint section leaves cannot have object dtype.")
    return (path, tuple(int(n) for n in array.shape), array.dtype.str, None)


def _flat_state(
    state: AtomisticDomainState, /
) -> tuple[tuple[str, ...], tuple[Array, ...], PyTreeDef]:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    paths = []
    leaves = []
    for path, leaf in path_leaves:
        if not isinstance(leaf, jax.Array):
            raise TypeError(
                "Every dynamic AtomisticDomainState leaf must be a JAX array."
            )
        paths.append(jax.tree_util.keystr(path))
        leaves.append(leaf)
    return tuple(paths), tuple(leaves), treedef


def _section_leaves(
    name: str, tree: PyTree[Array | np.ndarray], /
) -> tuple[tuple[str, ...], tuple[Array | np.ndarray, ...], PyTreeDef]:
    if not isinstance(name, str) or _SECTION_NAME.fullmatch(name) is None:
        raise ValueError("Section names must match [A-Za-z0-9_-]{1,64}.")
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    return (
        tuple(jax.tree_util.keystr(path) for path, _ in path_leaves),
        tuple(leaf for _, leaf in path_leaves),
        treedef,
    )


def _replicated_host(leaf: Array, /) -> np.ndarray:
    """Host copy of a fully replicated leaf (this process's own replica)."""
    if not leaf.is_fully_replicated:
        raise AtomisticDomainCheckpointError("A replicated leaf is not replicated.")
    return np.asarray(leaf.addressable_data(0))


def _rng_lineage_id(implementation: str, key_data: np.ndarray, /) -> str:
    data = np.ascontiguousarray(key_data)
    return canonical_fingerprint(
        {
            "kind": "atomistic-domain-rng-lineage",
            "implementation": implementation,
            "dtype": data.dtype.str,
            "shape": list(data.shape),
            "data": hashlib.sha256(data.tobytes()).hexdigest(),
        }
    )


def domain_execution_id(
    runtime: PreparedAtomisticDomainRuntime, /, *, process_count: int | None = None
) -> str:
    """Execution-topology identity: plan (ownership, owners, roles) and processes."""
    count = jax.process_count() if process_count is None else int(process_count)
    return canonical_fingerprint(
        {
            "kind": "atomistic-domain-execution",
            "plan_id": runtime.plan.plan_id,
            "owner_count": runtime.plan.ownership.owner_count,
            "local_capacity": runtime.plan.ownership.local_capacity,
            "process_count": count,
        }
    )


def _analysis_id(state: AtomisticDomainState, /, *, run_id: str | None = None) -> str:
    """Topology-neutral scientific identity of one run.

    Binds the run identity (``run_id`` overrides the state's, so a fresh
    destination template whose default run id derives from its own plan can
    be checked against the source run) and every role/strategy identity.
    """
    return canonical_fingerprint(
        {
            "kind": "atomistic-domain-analysis",
            "run_id": state.run_id if run_id is None else run_id,
            "roles": [topology.role_id for topology in state.topologies],
            "strategies": [str(topology.strategy) for topology in state.topologies],
        }
    )


def _structure_id(
    state: AtomisticDomainState, paths: Sequence[str], leaves: Sequence[Array], /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-domain-state-structure",
            "plan_id": state.plan_id,
            "model_revision_id": state.model_revision_id,
            "run_id": state.run_id,
            "roles": [topology.role_id for topology in state.topologies],
            "strategies": [str(topology.strategy) for topology in state.topologies],
            "streaming": [topology.streaming_plan_id for topology in state.topologies],
            "leaves": [
                [spec[0], list(spec[1]), spec[2], spec[3]]
                for spec in (
                    _leaf_spec(path, leaf)
                    for path, leaf in zip(paths, leaves, strict=True)
                )
            ],
        }
    )


def _support(execution_id: str, owners: int, processes: int, /) -> SupportTuple:
    return SupportTuple(
        "atomistic-domain-execution",
        {"execution_id": execution_id, "owner_count": owners, "process_count": processes},
    )


def _specs_payload(
    specs: Sequence[tuple[str, tuple[int, ...], str, str | None]], /
) -> list[list[object]]:
    return [[path, list(shape), dtype, key] for path, shape, dtype, key in specs]


def _specs_from_payload(
    value: object, /
) -> tuple[tuple[str, tuple[int, ...], str, str | None], ...]:
    if not isinstance(value, list):
        raise AtomisticDomainCheckpointError("The record leaf inventory is invalid.")
    specs = []
    for entry in value:
        if (
            not isinstance(entry, list)
            or len(entry) != 4
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], list)
            or any(type(n) is not int or n < 0 for n in entry[1])
            or not isinstance(entry[2], str)
            or (entry[3] is not None and not isinstance(entry[3], str))
        ):
            raise AtomisticDomainCheckpointError("The record leaf inventory is invalid.")
        specs.append((entry[0], tuple(entry[1]), entry[2], entry[3]))
    return tuple(specs)


@final
class AtomisticDomainConstraintStatics(StrictModule, NonTrainableState):
    """Static capacities of a checkpointed constraint state (migration import)."""

    row_capacity: int = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    directory_capacity: int = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class AtomisticDomainCheckpointRecord(StrictModule, NonTrainableState):
    """Canonical identity and schema of one domain continuation.

    ``leaves`` lists every state leaf ``(path, shape, dtype, key
    implementation)`` in canonical flatten order (typed keys as key data);
    ``sections``/``process_sections`` list each named section's leaves.
    ``scientific_id`` and ``hamiltonian_revision_id`` are the source runtime's
    layout-independent domain physics and its Hamiltonian numeric revision
    (`PreparedAtomisticDomainRuntime`), the identities every restore and
    changed-topology migration must preserve. ``record_id`` content-addresses
    every field.
    """

    plan_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    scientific_id: str = eqx.field(static=True)
    hamiltonian_revision_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    logical_count: int = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    leaves: tuple[tuple[str, tuple[int, ...], str, str | None], ...] = eqx.field(
        static=True
    )
    atom_payload: tuple[tuple[str, str], ...] = eqx.field(static=True)
    constraints: AtomisticDomainConstraintStatics | None
    rng_implementation: str = eqx.field(static=True)
    rng_lineage_id: str = eqx.field(static=True)
    sections: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...], str, str | None], ...]], ...
    ] = eqx.field(static=True)
    process_sections: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...], str, str | None], ...]], ...
    ] = eqx.field(static=True)
    loader_cursors: tuple[str, ...] = eqx.field(static=True)
    training_plan_id: str | None = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan_id: str,
        runtime_id: str,
        model_revision_id: str,
        scientific_id: str,
        hamiltonian_revision_id: str,
        run_id: str,
        analysis_id: str,
        execution_id: str,
        owner_count: int,
        local_capacity: int,
        process_count: int,
        logical_count: int,
        structure_id: str,
        leaves: Sequence[tuple[str, tuple[int, ...], str, str | None]],
        atom_payload: Sequence[tuple[str, str]],
        constraints: AtomisticDomainConstraintStatics | None,
        rng_implementation: str,
        rng_lineage_id: str,
        sections: Sequence[
            tuple[str, Sequence[tuple[str, tuple[int, ...], str, str | None]]]
        ],
        process_sections: Sequence[
            tuple[str, Sequence[tuple[str, tuple[int, ...], str, str | None]]]
        ],
        loader_cursors: Sequence[str],
        training_plan_id: str | None,
    ) -> None:
        for name, value in (
            ("owner_count", owner_count),
            ("local_capacity", local_capacity),
            ("process_count", process_count),
            ("logical_count", logical_count),
        ):
            if type(value) is not int or value <= 0:
                raise AtomisticDomainCheckpointError(f"Record {name} is invalid.")
        self.plan_id = str(plan_id)
        self.runtime_id = str(runtime_id)
        self.model_revision_id = str(model_revision_id)
        self.scientific_id = str(scientific_id)
        self.hamiltonian_revision_id = str(hamiltonian_revision_id)
        self.run_id = str(run_id)
        self.analysis_id = str(analysis_id)
        self.execution_id = str(execution_id)
        self.owner_count = owner_count
        self.local_capacity = local_capacity
        self.process_count = process_count
        self.logical_count = logical_count
        self.structure_id = str(structure_id)
        self.leaves = tuple(leaves)
        self.atom_payload = tuple((str(a), str(b)) for a, b in atom_payload)
        self.constraints = constraints
        self.rng_implementation = str(rng_implementation)
        self.rng_lineage_id = str(rng_lineage_id)
        self.sections = tuple((str(name), tuple(specs)) for name, specs in sections)
        self.process_sections = tuple(
            (str(name), tuple(specs)) for name, specs in process_sections
        )
        self.loader_cursors = tuple(str(value) for value in loader_cursors)
        self.training_plan_id = (
            None if training_plan_id is None else str(training_plan_id)
        )
        self.record_id = canonical_fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        constraints = self.constraints
        return {
            "kind": _RECORD_KIND,
            "plan_id": self.plan_id,
            "runtime_id": self.runtime_id,
            "model_revision_id": self.model_revision_id,
            "scientific_id": self.scientific_id,
            "hamiltonian_revision_id": self.hamiltonian_revision_id,
            "run_id": self.run_id,
            "analysis_id": self.analysis_id,
            "execution_id": self.execution_id,
            "owner_count": self.owner_count,
            "local_capacity": self.local_capacity,
            "process_count": self.process_count,
            "logical_count": self.logical_count,
            "structure_id": self.structure_id,
            "leaves": _specs_payload(self.leaves),
            "atom_payload": [list(item) for item in self.atom_payload],
            "constraints": None
            if constraints is None
            else {
                "row_capacity": constraints.row_capacity,
                "row_count": constraints.row_count,
                "halo_capacity": constraints.halo_capacity,
                "directory_capacity": constraints.directory_capacity,
                "migration_capacity": constraints.migration_capacity,
                "plan_id": constraints.plan_id,
            },
            "rng_implementation": self.rng_implementation,
            "rng_lineage_id": self.rng_lineage_id,
            "sections": [[name, _specs_payload(specs)] for name, specs in self.sections],
            "process_sections": [
                [name, _specs_payload(specs)] for name, specs in self.process_sections
            ],
            "loader_cursors": list(self.loader_cursors),
            "training_plan_id": self.training_plan_id,
        }

    def to_json(self) -> str:
        return canonical_json({**self._payload(), "record_id": self.record_id})

    @classmethod
    def from_json(cls, text: str, /) -> AtomisticDomainCheckpointRecord:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise AtomisticDomainCheckpointError(
                "The domain record is invalid."
            ) from error
        keys = {
            "kind", "plan_id", "runtime_id", "model_revision_id", "scientific_id",
            "hamiltonian_revision_id", "run_id", "analysis_id", "execution_id",
            "owner_count", "local_capacity", "process_count", "logical_count",
            "structure_id", "leaves", "atom_payload", "constraints",
            "rng_implementation", "rng_lineage_id", "sections", "process_sections",
            "loader_cursors", "training_plan_id", "record_id",
        }  # fmt: skip
        if (
            isinstance(payload, dict)
            and payload.get("kind") == _RECORD_KIND
            and set(payload) < keys
            and keys - set(payload) <= {"scientific_id", "hamiltonian_revision_id"}
        ):
            raise AtomisticDomainCheckpointError(
                "The domain record predates the layout-independent scientific "
                "identity; publish a new checkpoint from a prepared runtime."
            )
        if (
            not isinstance(payload, dict)
            or set(payload) != keys
            or payload["kind"] != _RECORD_KIND
        ):
            raise AtomisticDomainCheckpointError("The checkpoint is not a domain record.")
        constraints = payload["constraints"]
        try:
            record = cls(
                plan_id=payload["plan_id"],
                runtime_id=payload["runtime_id"],
                model_revision_id=payload["model_revision_id"],
                scientific_id=payload["scientific_id"],
                hamiltonian_revision_id=payload["hamiltonian_revision_id"],
                run_id=payload["run_id"],
                analysis_id=payload["analysis_id"],
                execution_id=payload["execution_id"],
                owner_count=payload["owner_count"],
                local_capacity=payload["local_capacity"],
                process_count=payload["process_count"],
                logical_count=payload["logical_count"],
                structure_id=payload["structure_id"],
                leaves=_specs_from_payload(payload["leaves"]),
                atom_payload=tuple(tuple(item) for item in payload["atom_payload"]),
                constraints=None
                if constraints is None
                else AtomisticDomainConstraintStatics(**constraints),
                rng_implementation=payload["rng_implementation"],
                rng_lineage_id=payload["rng_lineage_id"],
                sections=tuple(
                    (name, _specs_from_payload(specs))
                    for name, specs in payload["sections"]
                ),
                process_sections=tuple(
                    (name, _specs_from_payload(specs))
                    for name, specs in payload["process_sections"]
                ),
                loader_cursors=payload["loader_cursors"],
                training_plan_id=payload["training_plan_id"],
            )
        except (TypeError, ValueError) as error:
            raise AtomisticDomainCheckpointError(
                "The domain record is invalid."
            ) from error
        if record.record_id != payload["record_id"]:
            raise AtomisticDomainCheckpointError(
                "The domain checkpoint record identity is corrupt."
            )
        return record

    def ordinal(self, names: Sequence[str], /) -> int:
        """Canonical leaf ordinal of one attribute path."""
        return self.path_ordinal(_attribute_path(names))

    def path_ordinal(self, path: str, /) -> int:
        for index, spec in enumerate(self.leaves):
            if spec[0] == path:
                return index
        raise AtomisticDomainCheckpointError(f"The checkpoint lacks leaf {path}.")

    def section(
        self, name: str, /, *, process: bool = False
    ) -> tuple[tuple[str, tuple[int, ...], str, str | None], ...]:
        for section_name, specs in self.process_sections if process else self.sections:
            if section_name == name:
                return specs
        raise AtomisticDomainCheckpointError(f"The checkpoint lacks section {name!r}.")

    def cursors(self) -> tuple[AtomisticLoaderCursor, ...]:
        return tuple(
            AtomisticLoaderCursor.from_payload(json.loads(text))
            for text in self.loader_cursors
        )


@eqx.filter_jit
def _coherent(state: AtomisticDomainState) -> Array:
    """Accepted coherent epoch: uniform owner epochs and successful topologies."""
    epochs = state.layout.owner_epochs.astype(jnp.int64)
    admitted = jnp.all(epochs == epochs[0])
    for topology in state.topologies:
        admitted = (
            admitted
            & topology.evidence.successful
            & jnp.all(topology.owner_epochs.astype(jnp.int64) == epochs)
        )
    return admitted


def _require_runtime_state(
    runtime: PreparedAtomisticDomainRuntime, state: AtomisticDomainState, role: str, /
) -> None:
    if not isinstance(runtime, PreparedAtomisticDomainRuntime):
        raise TypeError("runtime must be a PreparedAtomisticDomainRuntime.")
    if not isinstance(state, AtomisticDomainState):
        raise TypeError(f"{role} must be an AtomisticDomainState.")
    if state.plan_id != runtime.plan.plan_id:
        raise AtomisticDomainCheckpointError(f"The {role} belongs to another plan.")
    if state.model_revision_id != runtime.model_revision_id:
        raise AtomisticDomainCheckpointError(
            f"The {role} is bound to another model revision."
        )


def _global_section_leaves(
    name: str, tree: PyTree[Array], /
) -> tuple[tuple[str, ...], tuple[Array, ...]]:
    """Paths and device leaves of one global section (JAX arrays only)."""
    paths, leaves, _ = _section_leaves(name, tree)
    arrays = tuple(leaf for leaf in leaves if isinstance(leaf, jax.Array))
    if len(arrays) != len(leaves):
        raise TypeError("Global sections hold JAX arrays only.")
    return paths, arrays


def _process_section_leaves(
    name: str, tree: PyTree[np.ndarray], /
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Paths and host leaves of one process section (no JAX arrays)."""
    paths, leaves, _ = _section_leaves(name, tree)
    if any(isinstance(leaf, jax.Array) for leaf in leaves):
        raise TypeError("Process sections hold host NumPy leaves only.")
    return paths, tuple(np.asarray(leaf) for leaf in leaves)


def _section_specs(
    sections: Mapping[str, PyTree[Array | np.ndarray]], process: bool, /
) -> tuple[tuple[str, tuple[tuple[str, tuple[int, ...], str, str | None], ...]], ...]:
    result = []
    for name, tree in sorted(sections.items()):
        paths, leaves = (
            _process_section_leaves(name, tree)
            if process
            else _global_section_leaves(name, tree)
        )
        result.append(
            (name, tuple(_leaf_spec(path, leaf) for path, leaf in zip(paths, leaves)))
        )
    return tuple(result)


def _build_record(
    runtime: PreparedAtomisticDomainRuntime,
    state: AtomisticDomainState,
    paths: Sequence[str],
    leaves: Sequence[Array],
    sections: Mapping[str, PyTree[Array]],
    process_sections: Mapping[str, PyTree[np.ndarray]],
    loader_cursors: Sequence[AtomisticLoaderCursor],
    training_plan_id: str | None,
    /,
) -> AtomisticDomainCheckpointRecord:
    _require_runtime_state(runtime, state, "state")
    if not bool(_coherent(state)):
        raise AtomisticDomainCheckpointError(
            "Only a coherent accepted state (uniform owner epochs, successful "
            "topologies for the current layout epoch) is checkpointed."
        )
    cursors = tuple(loader_cursors)
    if not all(isinstance(cursor, AtomisticLoaderCursor) for cursor in cursors):
        raise TypeError("loader_cursors must be AtomisticLoaderCursor values.")
    implementation = str(jax.random.key_impl(state.rng_key))
    key_data = _replicated_host(jax.random.key_data(state.rng_key))
    payload_prefix = _attribute_path(("atom_payload",))
    payload_entries = jax.tree_util.tree_flatten_with_path(state.atom_payload)[0]
    payload_paths = tuple(
        (str(getattr(path[0], "key", "")), payload_prefix + jax.tree_util.keystr(path))
        for path, _ in payload_entries
    )
    if len(payload_paths) != len(state.atom_payload) or {
        name for name, _ in payload_paths
    } != set(state.atom_payload):
        raise AtomisticDomainCheckpointError(
            "Every per-atom payload entry must be one owner-blocked array."
        )
    constraints = state.constraints
    statics = (
        None
        if constraints is None
        else AtomisticDomainConstraintStatics(
            row_capacity=constraints.rows.plan.local_capacity,
            row_count=constraints.rows.logical_count,
            halo_capacity=constraints.halo.halo_capacity,
            directory_capacity=constraints.directory_capacity,
            migration_capacity=constraints.migration_capacity,
            plan_id=constraints.plan_id,
        )
    )
    return AtomisticDomainCheckpointRecord(
        plan_id=state.plan_id,
        runtime_id=runtime.runtime_id,
        model_revision_id=state.model_revision_id,
        scientific_id=runtime.scientific_id,
        hamiltonian_revision_id=runtime.hamiltonian_revision_id,
        run_id=state.run_id,
        analysis_id=_analysis_id(state),
        execution_id=domain_execution_id(runtime),
        owner_count=runtime.plan.ownership.owner_count,
        local_capacity=runtime.plan.ownership.local_capacity,
        process_count=jax.process_count(),
        logical_count=state.layout.logical_count,
        structure_id=_structure_id(state, paths, leaves),
        leaves=tuple(_leaf_spec(path, leaf) for path, leaf in zip(paths, leaves)),
        atom_payload=payload_paths,
        constraints=statics,
        rng_implementation=implementation,
        rng_lineage_id=_rng_lineage_id(implementation, key_data),
        sections=_section_specs(sections, False),
        process_sections=_section_specs(process_sections, True),
        loader_cursors=tuple(canonical_json(cursor.to_payload()) for cursor in cursors),
        training_plan_id=training_plan_id,
    )


def _require_template(
    record: AtomisticDomainCheckpointRecord,
    runtime: PreparedAtomisticDomainRuntime,
    template: AtomisticDomainState,
    /,
) -> tuple[tuple[Array, ...], PyTreeDef]:
    _require_runtime_state(runtime, template, "template")
    paths, leaves, treedef = _flat_state(template)
    if _structure_id(template, paths, leaves) != record.structure_id:
        raise AtomisticDomainCheckpointError(
            "The template's continuation structure differs from the checkpoint "
            "(leaf paths, shapes, dtypes, roles, plan, model revision or run)."
        )
    return leaves, treedef


def _restore_key(data: Array | np.ndarray, implementation: str | None, /) -> Array:
    """A restored device leaf; recorded key data is re-wrapped as a typed key."""
    array = data if isinstance(data, jax.Array) else jnp.asarray(data)
    if implementation is None:
        return array
    return jax.random.wrap_key_data(array, impl=implementation)


def _restore_host_leaf(
    value: np.ndarray, template: Array, implementation: str | None, /
) -> Array:
    """Retain placement commitment as well as the template's device layout."""
    if template.committed:
        data = jax.device_put(value, template.sharding)
    else:
        # Explicit device_put would commit this local value and prevent JAX
        # from placing it alongside the restored distributed arguments.
        with default_device(next(iter(template.sharding.device_set))):
            data = jnp.asarray(value)
    if data.dtype != value.dtype:
        raise AtomisticDomainCheckpointError(
            "The active precision cannot restore this array dtype."
        )
    return _restore_key(data, implementation)


def _section_templates(
    record: AtomisticDomainCheckpointRecord,
    templates: Mapping[str, PyTree[Array | np.ndarray]],
    process: bool,
    /,
) -> dict[str, tuple[tuple[Array | np.ndarray, ...], PyTreeDef]]:
    expected = record.process_sections if process else record.sections
    if set(templates) != {name for name, _ in expected}:
        raise AtomisticDomainCheckpointError(
            "Section templates must name exactly the checkpointed sections "
            f"{sorted(name for name, _ in expected)}."
        )
    resolved = {}
    for name, tree in templates.items():
        paths, leaves, treedef = _section_leaves(name, tree)
        observed = tuple(_leaf_spec(path, leaf) for path, leaf in zip(paths, leaves))
        if observed != record.section(name, process=process):
            raise AtomisticDomainCheckpointError(
                f"The section template {name!r} differs from the checkpoint."
            )
        resolved[name] = (leaves, treedef)
    return resolved


def _global_section_templates(
    record: AtomisticDomainCheckpointRecord,
    templates: Mapping[str, PyTree[Array]],
    /,
) -> dict[str, tuple[tuple[Array, ...], PyTreeDef]]:
    """Admitted global section templates; their leaves carry target shardings."""
    resolved = {}
    for name, (leaves, treedef) in _section_templates(record, templates, False).items():
        arrays = tuple(leaf for leaf in leaves if isinstance(leaf, jax.Array))
        if len(arrays) != len(leaves):
            raise TypeError("Global section templates hold JAX arrays.")
        resolved[name] = (arrays, treedef)
    return resolved


# ---------------------------------------------------------------------------
# Restore results and evidence
# ---------------------------------------------------------------------------


@final
class AtomisticDomainMigrationEvidence(StrictModule, NonTrainableState):
    """Explicit lineage of one changed-topology continuation (this process).

    ``source_rows`` is this process's canonical source row range and
    ``restored_rows``/``restored_constraint_rows`` its active rows;
    ``read_payload_bytes`` is the authenticated region I/O it performed for
    owner-blocked leaves. ``rng_lineage_id`` equals the source record's
    lineage and the re-ingressed key's lineage (both verified).
    """

    source_checkpoint_id: str = eqx.field(static=True)
    source_manifest_id: str = eqx.field(static=True)
    source_execution_id: str = eqx.field(static=True)
    target_execution_id: str = eqx.field(static=True)
    target_runtime_id: str = eqx.field(static=True)
    source_owner_count: int = eqx.field(static=True)
    target_owner_count: int = eqx.field(static=True)
    source_process_count: int = eqx.field(static=True)
    target_process_count: int = eqx.field(static=True)
    source_rows: tuple[int, int] = eqx.field(static=True)
    restored_rows: int = eqx.field(static=True)
    restored_constraint_rows: int = eqx.field(static=True)
    read_payload_bytes: int = eqx.field(static=True)
    rng_lineage_id: str = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


@final
class AtomisticDomainRestore(StrictModule):
    """A restored domain continuation plus its restart-equivalence evidence.

    ``restart_class="bitwise"`` restores every leaf exactly; continued
    evolution is bitwise only under deterministic reductions.
    ``"tolerance"`` (changed topology) preserves scientific state and RNG
    lineage; ``migration`` holds the explicit lineage evidence.
    """

    state: AtomisticDomainState
    sections: Mapping[str, PyTree[Array]]
    process_sections: Mapping[str, PyTree[np.ndarray]]
    loader_cursors: tuple[AtomisticLoaderCursor, ...]
    migration: AtomisticDomainMigrationEvidence | None
    record: AtomisticDomainCheckpointRecord
    restart_class: RestartClass = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)
    source_checkpoint_id: str = eqx.field(static=True)


def _admission(
    record: AtomisticDomainCheckpointRecord,
    target_execution_id: str,
    target_owners: int,
    target_processes: int,
    restart_class: RestartClass,
    policy: TopologyRestartPolicy,
    relative_tolerance: float,
    /,
) -> tuple[str, str]:
    relation = TopologyRestartRelation(
        _support(record.execution_id, record.owner_count, record.process_count),
        _support(target_execution_id, target_owners, target_processes),
        restart_class,
        relative_tolerance=relative_tolerance,
        reason="Changed ownership re-blocks atoms, rebuilds every role topology "
        "and reorders owner reductions.",
    )
    admission = admit_topology_restart(relation, policy)
    if not admission.admitted:
        raise AtomisticDomainCheckpointError(
            f"The domain restart is not admitted: {admission.reason}"
        )
    return relation.relation_id, admission.admission_id


# ---------------------------------------------------------------------------
# In-memory checkpoint
# ---------------------------------------------------------------------------


@final
class AtomisticDomainCheckpoint(StrictModule, NonTrainableState):
    """Host continuation record of a fully addressable domain state.

    Replaces the owner-local checkpoint: ``arrays`` hold every continuation
    leaf in canonical leaf order (typed keys as key data), ``sections`` each
    named section's leaves, ``record`` the canonical record shared with
    distributed publication, and ``checkpoint_id`` the content identity over
    the record and every array.
    """

    arrays: tuple[np.ndarray, ...]
    sections: Mapping[str, tuple[np.ndarray, ...]]
    record: AtomisticDomainCheckpointRecord
    checkpoint_id: str = eqx.field(static=True)


def _content_id(
    record: AtomisticDomainCheckpointRecord,
    arrays: Sequence[np.ndarray],
    sections: Mapping[str, Sequence[np.ndarray]],
    /,
) -> str:
    named = {f"state/{index:06d}": value for index, value in enumerate(arrays)}
    for name, values in sections.items():
        for index, value in enumerate(values):
            named[f"section/{name}/{index:06d}"] = value
    return canonical_fingerprint(
        {
            "kind": "atomistic-domain-checkpoint-content",
            "record_id": record.record_id,
            "arrays": array_tree_fingerprint(named),
        }
    )


def _host_data(leaf: Array | np.ndarray, /) -> np.ndarray:
    if not isinstance(leaf, jax.Array):
        return np.array(leaf, copy=True)
    data, _ = _leaf_data(leaf)
    if not data.is_fully_addressable:
        raise AtomisticDomainCheckpointError(
            "In-memory checkpoints need fully addressable states; use "
            "publish_domain_checkpoint for multi-process states."
        )
    return np.asarray(jax.device_get(data))


def checkpoint_domain_state(
    runtime: PreparedAtomisticDomainRuntime,
    state: AtomisticDomainState,
    /,
    *,
    sections: Mapping[str, PyTree[Array]] | None = None,
    loader_cursors: Sequence[AtomisticLoaderCursor] = (),
    training_plan_id: str | None = None,
) -> AtomisticDomainCheckpoint:
    """Explicit host egress of a complete accepted, fully addressable state."""
    sections_ = dict(sections or {})
    paths, leaves, _ = _flat_state(state)
    record = _build_record(
        runtime, state, paths, leaves, sections_, {}, loader_cursors, training_plan_id
    )
    arrays = tuple(_host_data(leaf) for leaf in leaves)
    section_arrays = {
        name: tuple(_host_data(leaf) for leaf in _section_leaves(name, tree)[1])
        for name, tree in sorted(sections_.items())
    }
    return AtomisticDomainCheckpoint(
        arrays=arrays,
        sections=section_arrays,
        record=record,
        checkpoint_id=_content_id(record, arrays, section_arrays),
    )


def _bitwise_admission(
    record: AtomisticDomainCheckpointRecord, runtime: PreparedAtomisticDomainRuntime, /
) -> tuple[str, str]:
    execution_id = domain_execution_id(runtime)
    if record.execution_id != execution_id:
        raise AtomisticDomainCheckpointError(
            "The checkpoint was written by another execution topology; use "
            "migrate_domain_checkpoint."
        )
    if (
        record.scientific_id != runtime.scientific_id
        or record.hamiltonian_revision_id != runtime.hamiltonian_revision_id
    ):
        raise AtomisticDomainCheckpointError(
            "The checkpoint was written by another domain physics (model, terms, "
            "roles, reciprocal definition, cell or their numeric revision)."
        )
    return _admission(
        record,
        execution_id,
        runtime.plan.ownership.owner_count,
        jax.process_count(),
        "bitwise",
        TopologyRestartPolicy(allow_topology_change=False),
        0.0,
    )


def restore_domain_state(
    runtime: PreparedAtomisticDomainRuntime,
    checkpoint: AtomisticDomainCheckpoint,
    template: AtomisticDomainState,
    /,
    *,
    section_templates: Mapping[str, PyTree[Array]] | None = None,
) -> AtomisticDomainRestore:
    """Bitwise restore of an in-memory checkpoint into ``template``'s shardings.

    ``template`` is any state of ``runtime`` with the checkpoint's exact
    continuation structure (for example the run's initial ingress); only its
    structure, shardings and placement commitment are used.
    """
    if not isinstance(checkpoint, AtomisticDomainCheckpoint):
        raise TypeError("checkpoint must be an AtomisticDomainCheckpoint.")
    record = checkpoint.record
    if _content_id(record, checkpoint.arrays, checkpoint.sections) != (
        checkpoint.checkpoint_id
    ):
        raise AtomisticDomainCheckpointError(
            "The domain checkpoint content identity is corrupt."
        )
    relation_id, admission_id = _bitwise_admission(record, runtime)
    template_leaves, treedef = _require_template(record, runtime, template)
    restored = []
    for value, leaf, spec in zip(
        checkpoint.arrays, template_leaves, record.leaves, strict=True
    ):
        if tuple(value.shape) != spec[1] or value.dtype.str != spec[2]:
            raise AtomisticDomainCheckpointError("A checkpoint array changed shape.")
        restored.append(_restore_host_leaf(value, leaf, spec[3]))
    sections = {}
    for name, (leaves, section_treedef) in _global_section_templates(
        record, dict(section_templates or {})
    ).items():
        specs = record.section(name)
        sections[name] = jax.tree_util.tree_unflatten(
            section_treedef,
            [
                _restore_host_leaf(value, leaf, spec[3])
                for value, leaf, spec in zip(
                    checkpoint.sections[name], leaves, specs, strict=True
                )
            ],
        )
    return AtomisticDomainRestore(
        state=jax.tree_util.tree_unflatten(treedef, restored),
        sections=sections,
        process_sections={},
        loader_cursors=record.cursors(),
        migration=None,
        record=record,
        restart_class="bitwise",
        relation_id=relation_id,
        admission_id=admission_id,
        source_checkpoint_id=checkpoint.checkpoint_id,
    )


# ---------------------------------------------------------------------------
# Distributed publication
# ---------------------------------------------------------------------------


@final
class AtomisticDomainPublication(StrictModule, NonTrainableState):
    """Globally committed distributed domain checkpoint."""

    manifest: CheckpointManifest
    record: AtomisticDomainCheckpointRecord
    checkpoint_id: str = eqx.field(static=True)


def _lane_mesh(runtime: PreparedAtomisticDomainRuntime, /) -> Mesh:
    """One device per process of the ownership group (process-blocked leaves)."""
    lanes: dict[int, jax.Device] = {}
    for device in runtime.plan.ownership.execution_group.devices:
        lanes.setdefault(device.process_index, device)
    return Mesh(
        np.asarray([lanes[process] for process in sorted(lanes)], dtype=object),
        ("process",),
    )


def _process_sharding(mesh: Mesh, rank: int, /) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec("process", *([None] * rank)))


def _replicated_record(text: str, mesh: Mesh, /) -> Array:
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
    return jax.make_array_from_callback(
        data.shape, NamedSharding(mesh, PartitionSpec()), lambda index: data[index]
    )


def _process_blocked(leaves: Sequence[np.ndarray], mesh: Mesh, /) -> dict[str, Array]:
    process = jax.process_index()
    device = next(
        device for device in mesh.devices.reshape(-1) if device.process_index == process
    )
    blocked = {}
    for index, leaf in enumerate(leaves):
        value = np.asarray(leaf)
        blocked[f"{index:06d}"] = jax.make_array_from_single_device_arrays(
            (mesh.devices.size,) + value.shape,
            _process_sharding(mesh, value.ndim),
            [jax.device_put(value[None], device)],
        )
    return blocked


def _barrier(name: str, /) -> None:
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


def publish_domain_checkpoint(
    repository: ArtifactRepository,
    runtime: PreparedAtomisticDomainRuntime,
    state: AtomisticDomainState,
    /,
    *,
    checkpoint_id: str,
    writer_id: str,
    sections: Mapping[str, PyTree[Array]] | None = None,
    process_sections: Mapping[str, PyTree[np.ndarray]] | None = None,
    loader_cursors: Sequence[AtomisticLoaderCursor] = (),
    training_plan_id: str | None = None,
    parent_manifest: CheckpointManifest | None = None,
    region_bytes: int = _DEFAULT_REGION_BYTES,
    attempt_id: str | None = None,
    encoding: ChunkEncoding = "identity",
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> AtomisticDomainPublication:
    """Collective publication: per-process shards, then one global commit.

    Every process of the job calls this in the same order. Each process
    publishes only its canonical addressable shards (owner-blocked rows as
    authenticated row-region records of at most ``region_bytes``) plus its
    ``process_sections`` block; after a barrier the coordinator re-derives the
    complete manifest from the durable artifacts and writes the single global
    commit; after a second barrier every process loads that commit. A process
    lost before the commit leaves only uncommitted artifacts, which
    `load_distributed_checkpoint_commit` refuses.
    """
    sections_ = dict(sections or {})
    process_ = dict(process_sections or {})
    paths, leaves, _ = _flat_state(state)
    record = _build_record(
        runtime,
        state,
        paths,
        leaves,
        sections_,
        process_,
        loader_cursors,
        training_plan_id,
    )
    mesh = _lane_mesh(runtime)
    tree = {
        "record": _replicated_record(record.to_json(), mesh),
        "state": {
            f"{index:06d}": _leaf_data(leaf)[0] for index, leaf in enumerate(leaves)
        },
        "sections": {
            name: {
                f"{index:06d}": _leaf_data(leaf)[0]
                for index, leaf in enumerate(_global_section_leaves(name, section)[1])
            }
            for name, section in sorted(sections_.items())
        },
        "process_sections": {
            name: _process_blocked(_process_section_leaves(name, section)[1], mesh)
            for name, section in sorted(process_.items())
        },
    }
    group = runtime.plan.ownership.execution_group
    if len(group.spec.process_indices) > 1:
        # A local copy of a replicated scientific field is not an independent
        # global shard. Bind its replicas before choosing the canonical writer.
        local, distributed = eqx.partition(
            tree,
            lambda leaf: (
                isinstance(leaf, Array)
                and leaf.is_fully_addressable
                and leaf.is_fully_replicated
            ),
        )
        tree = eqx.combine(replicate_tree(local, group), distributed)
    publish_process_checkpoint(
        repository,
        checkpoint_id,
        record.execution_id,
        tree,
        analysis_plan_id=record.analysis_id,
        numeric_revision_id=record.model_revision_id,
        writer_id=writer_id,
        attempt_id=attempt_id,
        encoding=encoding,
        limits=limits,
        parent_manifest=parent_manifest,
        region_bytes=region_bytes,
    )
    _barrier(f"atomistic-domain-checkpoint:{checkpoint_id}:published")
    if jax.process_index() == 0:
        manifest = assemble_distributed_checkpoint_from_repository(
            repository,
            checkpoint_id,
            record.analysis_id,
            record.model_revision_id,
            record.execution_id,
            expected_process_count=jax.process_count(),
            parent_manifest=parent_manifest,
            limits=limits,
        )
        commit_distributed_checkpoint(
            repository,
            manifest,
            expected_process_count=jax.process_count(),
            writer_id=writer_id,
            attempt_id=attempt_id,
            parent_manifest=parent_manifest,
            limits=limits,
        )
    _barrier(f"atomistic-domain-checkpoint:{checkpoint_id}:committed")
    committed = load_distributed_checkpoint_commit(
        repository, checkpoint_id, parent_manifest=parent_manifest, limits=limits
    )
    return AtomisticDomainPublication(
        manifest=committed, record=record, checkpoint_id=checkpoint_id
    )


def _committed_record(
    repository: ArtifactRepository,
    checkpoint_id: str,
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
    /,
) -> tuple[CheckpointManifest, AtomisticDomainCheckpointRecord]:
    manifest = load_distributed_checkpoint_commit(
        repository, checkpoint_id, parent_manifest=parent_manifest, limits=limits
    )
    data = read_checkpoint_region(
        repository,
        manifest,
        "['record']",
        (slice(None),),
        limits=limits,
        parent_manifest=parent_manifest,
    )
    try:
        text = data.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise AtomisticDomainCheckpointError("The domain record is invalid.") from error
    record = AtomisticDomainCheckpointRecord.from_json(text)
    if (
        record.execution_id != manifest.execution_plan_id
        or record.analysis_id != manifest.analysis_plan_id
        or record.model_revision_id != manifest.numeric_revision_id
    ):
        raise AtomisticDomainCheckpointError(
            "The domain record disagrees with its committed manifest."
        )
    return manifest, record


def _restore_sections(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    record: AtomisticDomainCheckpointRecord,
    templates: Mapping[str, PyTree[Array]],
    limits: ArrayArchiveLimits,
    parent_manifest: CheckpointManifest | None,
    /,
) -> dict[str, PyTree[Array]]:
    sections = {}
    for name, (leaves, treedef) in _global_section_templates(record, templates).items():
        restored = []
        for index, (leaf, spec) in enumerate(
            zip(leaves, record.section(name), strict=True)
        ):
            data = restore_global_array_from_checkpoint(
                repository,
                manifest,
                f"['sections']['{name}']['{index:06d}']",
                leaf.sharding,
                limits=limits,
                parent_manifest=parent_manifest,
                committed=leaf.committed,
            )
            restored.append(_restore_key(data, spec[3]))
        sections[name] = jax.tree_util.tree_unflatten(treedef, restored)
    return sections


def restore_domain_checkpoint(
    repository: ArtifactRepository,
    runtime: PreparedAtomisticDomainRuntime,
    template: AtomisticDomainState,
    checkpoint_id: str,
    /,
    *,
    section_templates: Mapping[str, PyTree[Array]] | None = None,
    process_section_templates: Mapping[str, PyTree[np.ndarray]] | None = None,
    parent_manifest: CheckpointManifest | None = None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> AtomisticDomainRestore:
    """Same-topology bitwise restore of a globally committed checkpoint.

    The execution topology (plan, ownership, owner and process counts) must
    equal the checkpoint's; a changed topology is refused and must use
    `migrate_domain_checkpoint`. Every leaf is restored exactly into the
    template's sharding; each process reads only intersecting authenticated
    records. Process sections restore this process's own block.
    """
    manifest, record = _committed_record(
        repository, checkpoint_id, limits, parent_manifest
    )
    relation_id, admission_id = _bitwise_admission(record, runtime)
    template_leaves, treedef = _require_template(record, runtime, template)
    restored = []
    for index, (leaf, spec) in enumerate(
        zip(template_leaves, record.leaves, strict=True)
    ):
        data = restore_global_array_from_checkpoint(
            repository,
            manifest,
            f"['state']['{index:06d}']",
            leaf.sharding,
            limits=limits,
            parent_manifest=parent_manifest,
            committed=leaf.committed,
        )
        restored.append(_restore_key(data, spec[3]))
    sections = _restore_sections(
        repository,
        manifest,
        record,
        dict(section_templates or {}),
        limits,
        parent_manifest,
    )
    mesh = _lane_mesh(runtime)
    process_sections = {}
    for name, (leaves, section_treedef) in _section_templates(
        record, dict(process_section_templates or {}), True
    ).items():
        blocks = []
        for index, leaf in enumerate(leaves):
            restored_block = restore_global_array_from_checkpoint(
                repository,
                manifest,
                f"['process_sections']['{name}']['{index:06d}']",
                _process_sharding(mesh, np.asarray(leaf).ndim),
                limits=limits,
                parent_manifest=parent_manifest,
            )
            blocks.append(np.array(restored_block.addressable_data(0))[0])
        process_sections[name] = jax.tree_util.tree_unflatten(section_treedef, blocks)
    return AtomisticDomainRestore(
        state=jax.tree_util.tree_unflatten(treedef, restored),
        sections=sections,
        process_sections=process_sections,
        loader_cursors=record.cursors(),
        migration=None,
        record=record,
        restart_class="bitwise",
        relation_id=relation_id,
        admission_id=admission_id,
        source_checkpoint_id=checkpoint_id,
    )


def _canonical_range(rows: int, /) -> tuple[int, int]:
    """This process's contiguous share of ``rows`` source rows."""
    count = jax.process_count()
    index = jax.process_index()
    return rows * index // count, rows * (index + 1) // count


class _RegionReader:
    """Authenticated region reads of one committed checkpoint's state leaves."""

    def __init__(
        self,
        repository: ArtifactRepository,
        manifest: CheckpointManifest,
        record: AtomisticDomainCheckpointRecord,
        limits: ArrayArchiveLimits,
        parent_manifest: CheckpointManifest | None,
        /,
    ) -> None:
        self.repository = repository
        self.manifest = manifest
        self.record = record
        self.limits = limits
        self.parent_manifest = parent_manifest
        self.read_bytes = 0

    def _read(self, ordinal: int, region: tuple[slice, ...], /) -> np.ndarray:
        path = f"['state']['{ordinal:06d}']"
        self.read_bytes += plan_checkpoint_region_restore(
            self.manifest, path, (region,), limits=self.limits
        ).read_payload_bytes
        return read_checkpoint_region(
            self.repository,
            self.manifest,
            path,
            region,
            limits=self.limits,
            parent_manifest=self.parent_manifest,
        )

    def rows(self, ordinal: int, start: int, stop: int, /) -> np.ndarray:
        """One canonical leading-axis row range."""
        shape = self.record.leaves[ordinal][1]
        return self._read(
            ordinal, (slice(start, stop),) + tuple(slice(None) for _ in shape[1:])
        )

    def whole(self, ordinal: int, /) -> np.ndarray:
        """A complete replicated leaf."""
        return self._read(
            ordinal, tuple(slice(None) for _ in self.record.leaves[ordinal][1])
        )


def _replicated_tree(
    reader: _RegionReader, template: AtomisticDomainState, name: str, /
) -> PyTree[Array]:
    """Device leaves of one replicated subtree, structured and placed by ``template``."""
    record = reader.record
    prefix = _attribute_path((name,))
    sub_paths, treedef = jax.tree_util.tree_flatten_with_path(getattr(template, name))
    recorded = [
        spec[0]
        for spec in record.leaves
        if spec[0].startswith(prefix + ".") or spec[0].startswith(prefix + "[")
    ]
    if len(recorded) != len(sub_paths):
        raise AtomisticDomainCheckpointError(
            f"The template's {name} differs from the checkpoint."
        )
    leaves = []
    for sub_path, leaf in sub_paths:
        path = prefix + jax.tree_util.keystr(sub_path)
        ordinal = record.path_ordinal(path)
        if _leaf_spec(path, leaf) != record.leaves[ordinal]:
            raise AtomisticDomainCheckpointError(
                f"The template's {name} leaf {path} differs from the checkpoint."
            )
        leaves.append(
            _restore_host_leaf(reader.whole(ordinal), leaf, record.leaves[ordinal][3])
        )
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _pack(
    values: np.ndarray, active: np.ndarray, capacity: int, fill: int | float | bool, /
) -> np.ndarray:
    """Active rows first, then inactive padding up to ``capacity`` rows."""
    packed = np.full((capacity,) + values.shape[1:], fill, dtype=values.dtype)
    selected = values[active]
    packed[: selected.shape[0]] = selected
    return packed


def _admit_migration(
    record: AtomisticDomainCheckpointRecord,
    runtime: PreparedAtomisticDomainRuntime,
    template: AtomisticDomainState,
    policy: TopologyRestartPolicy,
    relative_tolerance: float,
    /,
) -> tuple[str, str, str]:
    """Admit a changed-topology continuation of ``record`` on ``runtime``.

    Returns the destination execution, relation and admission identities.
    """
    _require_runtime_state(runtime, template, "template")
    execution_id = domain_execution_id(runtime)
    if record.execution_id == execution_id:
        raise AtomisticDomainCheckpointError(
            "The checkpoint has this execution topology; use restore_domain_checkpoint."
        )
    # The runtime's layout-independent physics, not ``model_revision_id``: a
    # term-only revision digests placement-bearing term identities.
    if (
        record.scientific_id != runtime.scientific_id
        or record.hamiltonian_revision_id != runtime.hamiltonian_revision_id
        or record.analysis_id != _analysis_id(template, run_id=record.run_id)
    ):
        raise AtomisticDomainCheckpointError(
            "Migration preserves the domain physics (model, terms, roles, "
            "reciprocal definition, cell, their numeric revision), roles and "
            "strategies of the checkpoint."
        )
    if record.process_sections:
        raise AtomisticDomainCheckpointError(
            "Process-bound sections (for example pending transport packets) cannot "
            "change topology; drain them and checkpoint again."
        )
    relation_id, admission_id = _admission(
        record,
        execution_id,
        runtime.plan.ownership.owner_count,
        jax.process_count(),
        "tolerance",
        policy,
        relative_tolerance,
    )
    return execution_id, relation_id, admission_id


def _source_rng_key(reader: _RegionReader, /) -> tuple[np.ndarray, str, str]:
    """The checkpointed RNG key data, its implementation and verified lineage."""
    record = reader.record
    key_ordinal = record.ordinal(("rng_key",))
    key_data = reader.whole(key_ordinal)
    implementation = record.leaves[key_ordinal][3]
    if implementation is None or implementation != record.rng_implementation:
        raise AtomisticDomainCheckpointError("The checkpoint RNG key is invalid.")
    lineage = _rng_lineage_id(implementation, key_data)
    if lineage != record.rng_lineage_id:
        raise AtomisticDomainCheckpointError("The restored RNG lineage is corrupt.")
    return key_data, implementation, lineage


def _migrated_constraints(
    reader: _RegionReader, local_owners: int, /
) -> tuple[AtomisticDomainConstraintRows | None, int]:
    """This process's canonical share of the active constraint rows, packed.

    Returns the rows (None without constraints) and the active row count.
    """
    record = reader.record
    statics = record.constraints
    if statics is None:
        return None, 0
    constraint_total = record.leaves[record.ordinal(_CONSTRAINT_ROW_PATHS["row_ids"])][1][
        0
    ]
    c_start, c_stop = _canonical_range(constraint_total)
    c_source = {
        name: reader.rows(record.ordinal(names), c_start, c_stop)
        for name, names in _CONSTRAINT_ROW_PATHS.items()
    }
    c_active = np.asarray(c_source["active"], dtype=np.bool_)
    constraint_rows = int(np.count_nonzero(c_active))
    c_capacity = local_owners * statics.row_capacity
    if constraint_rows > c_capacity:
        raise AtomisticDomainMigrationCapacityError(
            f"Process {jax.process_index()} restores {constraint_rows} constraint "
            f"rows but holds {c_capacity} constraint row slots."
        )
    c_replicated = {
        name: reader.whole(record.ordinal(names))
        for name, names in _CONSTRAINT_REPLICATED_PATHS.items()
    }
    constraints = AtomisticDomainConstraintRows(
        row_ids=_pack(c_source["row_ids"], c_active, c_capacity, -1),
        atoms=_pack(c_source["atoms"], c_active, c_capacity, -1),
        components=_pack(c_source["components"], c_active, c_capacity, -1),
        targets=_pack(c_source["targets"], c_active, c_capacity, 0),
        lagrange=_pack(c_source["lagrange"], c_active, c_capacity, 0),
        active=_pack(c_active, c_active, c_capacity, False),
        position_residual=c_replicated["position_residual"],
        velocity_residual=c_replicated["velocity_residual"],
        topology_epoch=c_replicated["topology_epoch"],
        row_capacity=statics.row_capacity,
        row_count=statics.row_count,
        halo_capacity=statics.halo_capacity,
        directory_capacity=statics.directory_capacity,
        migration_capacity=statics.migration_capacity,
        plan_id=statics.plan_id,
    )
    return constraints, constraint_rows


def migrate_domain_checkpoint(
    repository: ArtifactRepository,
    runtime: PreparedAtomisticDomainRuntime,
    template: AtomisticDomainState,
    checkpoint_id: str,
    /,
    *,
    policy: TopologyRestartPolicy,
    relative_tolerance: float,
    section_templates: Mapping[str, PyTree[Array]] | None = None,
    parent_manifest: CheckpointManifest | None = None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> AtomisticDomainRestore:
    """Continue a committed checkpoint on a changed owner/process topology.

    Collective over the destination processes. ``template`` is a state of the
    destination ``runtime`` (for example its initial ingress); it fixes the
    structure of the replicated payload, controller, ledger and conditioning
    trees. The relation is tolerance-class (``relative_tolerance`` for future
    trajectories) and must be admitted by ``policy``. Each process reads one
    canonical range of source rows (authenticated region records only), keeps
    the active rows and re-ingresses them with their complete continuation
    (forces and force record, RNG key, controllers, ledger, conditioning,
    constraints, step, time, case identity, run id) through
    ``runtime.ingress_process_local``, which owns wrapping, migration to
    current-cell owners and every topology. Process sections (process-bound
    transport state) are refused: drain the transport first.
    """
    manifest, record = _committed_record(
        repository, checkpoint_id, limits, parent_manifest
    )
    execution_id, relation_id, admission_id = _admit_migration(
        record, runtime, template, policy, relative_tolerance
    )
    ownership = runtime.plan.ownership
    reader = _RegionReader(repository, manifest, record, limits, parent_manifest)
    me = jax.process_index()
    local_owners = sum(1 for process in ownership.owner_processes if process == me)
    rows_here = local_owners * ownership.local_capacity
    total_rows = record.leaves[record.ordinal(("positions",))][1][0]
    start, stop = _canonical_range(total_rows)
    source = {
        name: reader.rows(record.ordinal(names), start, stop)
        for name, names in _ROW_PATHS.items()
    }
    payload = {
        name: reader.rows(record.path_ordinal(path), start, stop)
        for name, path in record.atom_payload
    }
    active = np.asarray(source["active"], dtype=np.bool_)
    count = int(np.count_nonzero(active))
    if count > rows_here:
        raise AtomisticDomainMigrationCapacityError(
            f"Process {me} restores {count} active atoms but holds {rows_here} "
            "owner rows; raise local_capacity or the owner count."
        )
    mask = np.zeros((rows_here,), np.bool_)
    mask[:count] = True
    replicated = {
        name: reader.whole(record.ordinal(names))
        for name, names in _REPLICATED_PATHS.items()
    }
    key_data, implementation, lineage = _source_rng_key(reader)
    trees = {name: _replicated_tree(reader, template, name) for name in _REPLICATED_TREES}
    constraints, constraint_rows = _migrated_constraints(reader, local_owners)
    state = runtime.ingress_process_local(
        _pack(source["positions"], active, rows_here, 0),
        _pack(source["species"], active, rows_here, 0),
        rng_key=_restore_key(key_data, implementation),
        active=mask,
        stable_ids=_pack(source["stable_ids"], active, rows_here, -1),
        logical_indices=_pack(source["logical_indices"], active, rows_here, -1),
        logical_count=record.logical_count,
        momenta=_pack(source["momenta"], active, rows_here, 0),
        masses=_pack(source["masses"], active, rows_here, 1),
        image_counts=_pack(source["image_counts"], active, rows_here, 0),
        forces=_pack(source["forces"], active, rows_here, 0),
        force_record=AtomisticDomainForceRecord(
            energy=replicated["force_energy"],
            strain_derivative=replicated["force_strain_derivative"],
            stress=replicated["force_stress"],
            strain_available=replicated["force_strain_available"],
            step=replicated["force_step"],
            successful=replicated["force_successful"],
        ),
        cell_vectors=replicated["cell_vectors"],
        atom_payload={
            name: _pack(value, active, rows_here, 0) for name, value in payload.items()
        },
        replicated_payload=trees["replicated_payload"],
        conditioning=trees["conditioning"],
        controllers=trees["controllers"],
        ledger=trees["ledger"],
        constraints=constraints,
        step_index=int(replicated["step_index"]),
        time=float(replicated["time"]),
        case_identity=replicated["case_identity_words"],
        run_id=record.run_id,
    )
    restored_key = _replicated_host(jax.random.key_data(state.rng_key))
    if _rng_lineage_id(str(jax.random.key_impl(state.rng_key)), restored_key) != lineage:
        raise AtomisticDomainCheckpointError(
            "Re-ingress did not preserve the RNG lineage."
        )
    sections = _restore_sections(
        repository,
        manifest,
        record,
        dict(section_templates or {}),
        limits,
        parent_manifest,
    )
    migration = AtomisticDomainMigrationEvidence(
        source_checkpoint_id=checkpoint_id,
        source_manifest_id=manifest.manifest_id,
        source_execution_id=record.execution_id,
        target_execution_id=execution_id,
        target_runtime_id=runtime.runtime_id,
        source_owner_count=record.owner_count,
        target_owner_count=ownership.owner_count,
        source_process_count=record.process_count,
        target_process_count=jax.process_count(),
        source_rows=(start, stop),
        restored_rows=count,
        restored_constraint_rows=constraint_rows,
        read_payload_bytes=reader.read_bytes,
        rng_lineage_id=lineage,
        relation_id=relation_id,
        admission_id=admission_id,
        evidence_id=canonical_fingerprint(
            {
                "kind": "atomistic-domain-migration",
                "source_manifest_id": manifest.manifest_id,
                "source_execution_id": record.execution_id,
                "target_execution_id": execution_id,
                "target_runtime_id": runtime.runtime_id,
                "relation_id": relation_id,
                "admission_id": admission_id,
                "rng_lineage_id": lineage,
            }
        ),
    )
    return AtomisticDomainRestore(
        state=state,
        sections=sections,
        process_sections={},
        loader_cursors=record.cursors(),
        migration=migration,
        record=record,
        restart_class="tolerance",
        relation_id=relation_id,
        admission_id=admission_id,
        source_checkpoint_id=checkpoint_id,
    )


__all__ = (
    "AtomisticDomainCheckpoint",
    "AtomisticDomainCheckpointError",
    "AtomisticDomainCheckpointRecord",
    "AtomisticDomainConstraintStatics",
    "AtomisticDomainMigrationCapacityError",
    "AtomisticDomainMigrationEvidence",
    "AtomisticDomainPublication",
    "AtomisticDomainRestore",
    "checkpoint_domain_state",
    "domain_execution_id",
    "migrate_domain_checkpoint",
    "publish_domain_checkpoint",
    "restore_domain_checkpoint",
    "restore_domain_state",
)
