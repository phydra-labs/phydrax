#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Staged heterogeneous atomistic campaigns over the native execution owners.

A campaign composes native scientific transitions — molecular dynamics,
relaxation, grouped reaction paths, generation and labeling — into stages of
bounded lane banks.  It is not another scheduler: signature grouping and
semantic RNG indices come from `ExecutionWorksetPlan`, lane refill from
`refill_completed_tasks`, bounded group admission and backpressure from
`allocate_frontier_slots`, durable output from the lifecycle publication
owner (`write_array_archive` / `publish_file`) and observation from
`IterationSession`.  Every scientific transition stays inside its owner:
the campaign never re-implements an integrator, optimizer, path method,
sampler or Hamiltonian.

Stage operations
----------------
An `AbstractAtomisticStageOperation` is a shared lane executor bound to one
prepared native runtime.  Its static `AtomisticStageSignature` holds only
layout identities — the system, resource-bundle and Hamiltonian
``layout_id`` (`atomistic_hamiltonian_layout_id`), derivative request,
precision, method layout and placement — never a scientific identity,
numeric model revision, task, material record or budget, so signature
grouping equals compilation: operations are content-free PyTrees (their
identities are dynamic `IdentityWords`) and every material and revision of
one layout shares one executable per entrypoint.  Two stages that reference
the same operation (equal ``binding_id``) share one lane bank, so their
lanes advance in one batched native call per round; lanes carry their own
stage index, budget and consumption.  An evaluation computed by one stage
is reused by another only when their `AtomisticEvaluationKey` (Hamiltonian
definition, numeric revisions, conditions/controls, bias and requested
quantities) agrees; the key is a host record and never enters compiled code.

Round transaction
-----------------
``evaluate`` advances admitted lanes by one native segment and runs the
operation-owned capacity lifecycle; ``stage_outputs`` classifies terminal
lanes, holds incomplete groups, and moves whole groups to the next stage
queue or the bounded pending-publication bank (overflow parks the lane
without advancing physics); ``publish`` obtains durable receipts from the
sink or leaves records pending; ``commit`` writes the complete continuation
envelope atomically; ``refill`` admits queued groups into free lanes.  A
state returned by any phase may be discarded: `resume` restores the last
durable commit, refills deterministically and never duplicates a published
output (sinks are idempotent by semantic key).

Dynamics continuation
---------------------
A dynamics lane exports its complete accepted native state as an
`AtomisticDynamicsContinuation` bound to the exporting operation: thermostat
chains, barostat, GLE memory and Monte Carlo counters, step, time, RNG key,
images, term states, force cache and neighborhood resources.  A stage
boundary is an observation cadence, not a new run: the next dynamics stage
adopts the same runtime's state bitwise, enters another row of the same
thermodynamic table by the owner's warm rebase, and a changed Hamiltonian
of the same method by the owner's ``transition_hamiltonian`` (lineage
recorded).  A changed integrator, table, constraint set or system is refused
unless the stage declares `AtomisticContinuationPolicy.RESTART`, an explicit
new run initialized by the owner from the incoming phase point.  Label
stages evaluate a configuration without moving it, so a walker's
continuation passes through them; relaxation and path stages move the
configuration and end it.  Terminal walkers keep their final continuation
(`PreparedAtomisticCampaign.terminal_packets`).

Outcomes
--------
`AtomisticStageOutcome` keeps scientific convergence (``CONVERGED``),
completion of a fixed budget (``COMPLETED``), exhaustion before convergence
(``EXHAUSTED``), scientific/numerical failure (``FAILED``), an unconsumed
capacity attempt (``CAPACITY_PENDING``) and an exhausted declared capacity
ladder (``CAPACITY_REFUSED``) distinct.  Capacity is repaired by the owner:
dynamics and labeling move only the offending task to a bucket at grown
declared capacities (rebucketing, re-primed without physical or RNG
advancement); relaxation and reaction paths grow their whole bank through
their own ``grow`` transaction, which preserves optimizer state.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag, StrEnum
from pathlib import Path
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._array_archive import (
    ArrayArchiveLimits,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._execution_pool import (
    allocate_frontier_slots,
    PoolExecutionSignature,
    refill_completed_tasks,
)
from .._execution_resources import ExecutionGroupSpec
from .._execution_workset import ExecutionWorksetPlan
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords, NumericRevision
from .._iteration import (
    IterationCoordinates,
    IterationPhase,
    IterationRecord,
    IterationScope,
    IterationSession,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticDynamicsState,
    AtomisticInitialStateEvaluation,
    AtomisticStepRejectionReason,
    AtomisticThermodynamicRebaseEvaluation,
    CAPACITY_REJECTIONS,
    PreparedAtomisticDynamics,
)
from ._generation import (
    AbstractPreparedAtomisticSampler,
    AbstractProposalFilter,
    atomistic_proposal_id,
    AtomisticProposalBlock,
    screen_atomistic_proposals,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundleState,
    PreparedAtomisticNeighborhoodBundle,
)
from ._potential_program import AbstractPreparedAtomisticHamiltonian
from ._relaxation import (
    _identity_rows,
    _identity_strings,
    AtomisticRelaxationState,
    AtomisticRelaxationStatus,
    AtomisticSymmetryChart,
    PreparedAtomisticRelaxation,
    trial_geometry,
)
from ._system import PreparedAtomisticSystem
from ._thermodynamic import (
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)
from ._types import AtomicStructure


if TYPE_CHECKING:
    from ..chemistry.reaction._path import PreparedAtomisticNudgedElasticBand


_CHECKPOINT_FORMAT = "phydrax-atomistic-campaign-checkpoint"
_OUTPUT_FORMAT = "phydrax-atomistic-campaign-output"
_INT32_MAX = int(np.iinfo(np.int32).max)
_UINT32_LIMIT = 2**32
# One member per lane-bank leaf: a campaign envelope holds many banks, so the
# member, manifest and directory bounds are explicit rather than the
# single-state default (all other admission limits keep their defaults).
_CAMPAIGN_ARCHIVE_LIMITS = ArrayArchiveLimits(
    max_members=65_536,
    max_manifest_bytes=67_108_864,
    max_central_directory_bytes=67_108_864,
    max_manifest_nesting=32,
)


# -- vocabulary -------------------------------------------------------------------


class AtomisticStageOperationKind(StrEnum):
    """Scientific family of a stage operation."""

    DYNAMICS = "dynamics"
    RELAXATION = "relaxation"
    REACTION_PATH = "reaction-path"
    GENERATION = "generation"
    LABEL = "label"


class AtomisticStageOutcome(IntEnum):
    """Per-lane outcome of a stage attempt.

    ``CAPACITY_PENDING`` is the only nonterminal non-active outcome: the
    attempt was not consumed and is replayed after the capacity lifecycle.
    """

    ACTIVE = 0
    CONVERGED = 1
    COMPLETED = 2
    EXHAUSTED = 3
    FAILED = 4
    CAPACITY_PENDING = 5
    CAPACITY_REFUSED = 6


_TERMINAL = frozenset(
    {
        int(AtomisticStageOutcome.CONVERGED),
        int(AtomisticStageOutcome.COMPLETED),
        int(AtomisticStageOutcome.EXHAUSTED),
        int(AtomisticStageOutcome.FAILED),
        int(AtomisticStageOutcome.CAPACITY_REFUSED),
    }
)


class AtomisticCapacityMode(StrEnum):
    """Who repairs a capacity-pending attempt.

    ``REBUCKET`` moves only the offending task to a bank at grown declared
    capacities; ``GROW`` grows the operation's whole bank in place.
    """

    REBUCKET = "rebucket"
    GROW = "grow"


class AtomisticRevisionPolicy(StrEnum):
    """Treatment of in-flight lanes when a stage operation is rebound.

    ``DRAIN`` finishes admitted lanes under the old binding and admits new
    tasks under the new one; ``REPRIME`` rebinds every admitted lane and
    re-primes its caches without physical, RNG, optimizer or budget events.
    """

    DRAIN = "drain"
    REPRIME = "reprime"


class AtomisticContinuationPolicy(StrEnum):
    """How a dynamics stage enters an incoming walker's dynamics continuation.

    ``CONTINUE`` resumes the complete native state; a change of integrator,
    thermodynamic table, constraints or system is refused
    (``CONTINUATION_REFUSED``).  ``RESTART`` declares a new dynamics run: the
    owner initializes it from the incoming phase point (positions, momenta,
    cell, time) with the campaign's semantic key, and the visit records
    ``DYNAMICS_RESTARTED``.
    """

    CONTINUE = "continue"
    RESTART = "restart"


class AtomisticEntryPolicy(StrEnum):
    """Whether a stage may reuse an incoming compatible evaluation."""

    EVALUATE = "evaluate"
    SHARE = "share"


class AtomisticCampaignReason(IntFlag):
    """Campaign-level causes recorded beside method-owned reason codes."""

    NONE = 0
    CELL_MISMATCH = 1 << 0
    IDENTITY_MISMATCH = 1 << 1
    UNROUTABLE = 1 << 2
    INVALID_PACKET = 1 << 3
    LADDER_EXHAUSTED = 1 << 4
    REPRIME_FAILED = 1 << 5
    SOURCE_CAPACITY = 1 << 6
    ROLE_CAPACITY = 1 << 7
    SHARED_EVALUATION = 1 << 8
    ADMISSION_REFUSED = 1 << 9
    CONTINUATION_REFUSED = 1 << 10
    DYNAMICS_RESTARTED = 1 << 11


_QUANTITIES = ("energy", "forces", "stress")


def _identifier(value: object, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


# -- evaluation identity ----------------------------------------------------------


def _hamiltonian_identity(program: AbstractPreparedAtomisticHamiltonian, /) -> str:
    """Canonical scientific identity of the prepared Hamiltonian (``scientific_id``).

    Changing model weights changes the physical Hamiltonian and therefore this
    identity; resource bindings do not.  Numeric content is additionally
    fingerprinted by `atomistic_hamiltonian_revision`.
    """
    return _identifier(program.scientific_id, "scientific_id")


def atomistic_hamiltonian_revision(
    program: AbstractPreparedAtomisticHamiltonian, /
) -> NumericRevision:
    """Numeric revision of every inexact numeric leaf of a prepared Hamiltonian.

    The semantic identity is the resource-independent program definition; the
    numeric content is every inexact array of the prepared terms and the term
    coefficients (model parameters, classical parameters and prepared tables).
    Host boundary: concrete arrays are hashed once, outside transformed code.
    """
    if not isinstance(program, AbstractPreparedAtomisticHamiltonian):
        raise TypeError("program must be a prepared atomistic Hamiltonian.")
    leaves = jax.tree_util.tree_leaves(
        eqx.filter((program.terms, program.coefficients), eqx.is_inexact_array)
    )
    return NumericRevision(
        f"atomistic-hamiltonian:{_hamiltonian_identity(program)}",
        tuple(np.asarray(leaf) for leaf in leaves),
    )


@final
class AtomisticEvaluationKey(StrictModule, NonTrainableState):
    """Everything that decides whether two evaluations are the same evaluation.

    ``quantities`` are the available (or requested) responses among
    ``energy``, ``forces`` and ``stress`` (tension-positive ``sym(dE/d
    strain)/V``).  A request is served by an available evaluation exactly when
    Hamiltonian, numeric revisions, conditions and bias agree and the
    requested quantities are a subset of the available ones.
    """

    hamiltonian_id: str = eqx.field(static=True)
    numeric_revision_ids: tuple[str, ...] = eqx.field(static=True)
    conditions_id: str = eqx.field(static=True)
    bias_id: str | None = eqx.field(static=True)
    quantities: tuple[str, ...] = eqx.field(static=True)
    key_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian_id: str,
        /,
        *,
        numeric_revision_ids: Sequence[str],
        conditions_id: str,
        bias_id: str | None = None,
        quantities: Sequence[str],
    ) -> None:
        revisions = tuple(
            sorted(_identifier(value, "revision") for value in numeric_revision_ids)
        )
        if len(set(revisions)) != len(revisions):
            raise ValueError("numeric_revision_ids must be distinct.")
        requested = tuple(value for value in _QUANTITIES if value in set(quantities))
        if len(requested) != len(set(quantities)) or not requested:
            raise ValueError(f"quantities must be a non-empty subset of {_QUANTITIES}.")
        self.hamiltonian_id = _identifier(hamiltonian_id, "hamiltonian_id")
        self.numeric_revision_ids = revisions
        self.conditions_id = _identifier(conditions_id, "conditions_id")
        self.bias_id = None if bias_id is None else _identifier(bias_id, "bias_id")
        self.quantities = requested
        self.key_id = canonical_fingerprint(
            {
                "kind": "atomistic-evaluation-key",
                "hamiltonian": self.hamiltonian_id,
                "revisions": list(revisions),
                "conditions": self.conditions_id,
                "bias": self.bias_id,
                "quantities": list(requested),
            }
        )

    def served_by(self, available: AtomisticEvaluationKey, /) -> bool:
        if not isinstance(available, AtomisticEvaluationKey):
            raise TypeError("available must be an AtomisticEvaluationKey.")
        return (
            self.hamiltonian_id == available.hamiltonian_id
            and self.numeric_revision_ids == available.numeric_revision_ids
            and self.conditions_id == available.conditions_id
            and self.bias_id == available.bias_id
            and set(self.quantities) <= set(available.quantities)
        )

    def restricted(self, quantities: Sequence[str], /) -> AtomisticEvaluationKey:
        return AtomisticEvaluationKey(
            self.hamiltonian_id,
            numeric_revision_ids=self.numeric_revision_ids,
            conditions_id=self.conditions_id,
            bias_id=self.bias_id,
            quantities=quantities,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "hamiltonian_id": self.hamiltonian_id,
            "numeric_revision_ids": list(self.numeric_revision_ids),
            "conditions_id": self.conditions_id,
            "bias_id": self.bias_id,
            "quantities": list(self.quantities),
            "key_id": self.key_id,
        }

    @staticmethod
    def from_record(record: Mapping[str, Any], /) -> AtomisticEvaluationKey:
        key = AtomisticEvaluationKey(
            record["hamiltonian_id"],
            numeric_revision_ids=tuple(record["numeric_revision_ids"]),
            conditions_id=record["conditions_id"],
            bias_id=record["bias_id"],
            quantities=tuple(record["quantities"]),
        )
        if key.key_id != record["key_id"]:
            raise ValueError("Evaluation key record identity is corrupt.")
        return key


def _conditions_id(controls: Any, control_ids: Sequence[str], /) -> str:
    values = np.asarray(controls, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Evaluation controls must be finite.")
    return canonical_fingerprint(
        {
            "kind": "atomistic-evaluation-conditions",
            "control_ids": list(control_ids),
            "controls": values.tolist(),
        }
    )


# -- packets ----------------------------------------------------------------------


def _optional_array(value: Any, dtype: Any, name: str, /) -> np.ndarray | None:
    """Owned copy; finiteness is checked where a consumer admits the value.

    Failure outputs legitimately carry nonfinite accepted-state evidence.
    """
    del name
    if value is None:
        return None
    return np.array(value, dtype=dtype, copy=True)


@final
class AtomisticEvaluationRecord(StrictModule, NonTrainableState):
    """Accepted energy/forces/stress of one configuration and their identity."""

    energy: np.ndarray
    forces: np.ndarray
    stress: np.ndarray | None
    key: AtomisticEvaluationKey

    def __init__(
        self,
        energy: Any,
        forces: Any,
        stress: Any,
        key: AtomisticEvaluationKey,
        /,
    ) -> None:
        if not isinstance(key, AtomisticEvaluationKey):
            raise TypeError("key must be an AtomisticEvaluationKey.")
        energy_ = np.asarray(energy, dtype=np.float64).reshape(())
        forces_ = np.array(forces, dtype=np.float64, copy=True)
        if forces_.ndim != 2 or forces_.shape[1] != 3:
            raise ValueError("forces must have shape (N, 3).")
        stress_ = (
            None if stress is None else np.array(stress, dtype=np.float64, copy=True)
        )
        if stress_ is not None and stress_.shape != (3, 3):
            raise ValueError("stress must have shape (3, 3).")
        if ("stress" in key.quantities) != (stress_ is not None):
            raise ValueError("The key's quantities must match the stored stress.")
        if not (
            np.isfinite(energy_)
            and np.all(np.isfinite(forces_))
            and (stress_ is None or np.all(np.isfinite(stress_)))
        ):
            raise ValueError("Evaluation records hold only finite accepted values.")
        self.energy = energy_
        self.forces = forces_
        self.stress = stress_
        self.key = key


@final
class AtomisticTaskPacket(StrictModule, NonTrainableState):
    """Host continuation packet of one task between stages and lane banks.

    ``positions`` are unwrapped Cartesian coordinates of the family system
    capacity; ``species`` are atomic numbers and ``atom_type_ids`` the
    method's type identities; ``path`` holds whole bands ``(M, N, 3)``.
    ``continuation`` carries complete method state: a dynamics walker's
    `AtomisticDynamicsContinuation` (across stages and campaigns) or another
    method's state of a same-stage capacity transfer.  ``child_id``/``family``
    mark a spawned task (generation proposals); a packet without
    ``child_id`` continues its own task.
    """

    positions: np.ndarray | None
    cell_vectors: np.ndarray | None
    momenta: np.ndarray | None
    species: np.ndarray | None
    atom_type_ids: np.ndarray | None
    atom_mask: np.ndarray | None
    path: np.ndarray | None
    path_energies: np.ndarray | None
    time: np.ndarray
    evaluation: AtomisticEvaluationRecord | None
    continuation: Any
    child_id: str | None = eqx.field(static=True)
    family: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        positions: Any = None,
        cell_vectors: Any = None,
        momenta: Any = None,
        species: Any = None,
        atom_type_ids: Any = None,
        atom_mask: Any = None,
        path: Any = None,
        path_energies: Any = None,
        time: Any = 0.0,
        evaluation: AtomisticEvaluationRecord | None = None,
        continuation: Any = None,
        child_id: str | None = None,
        family: str | None = None,
    ) -> None:
        positions_ = _optional_array(positions, np.float64, "positions")
        if positions_ is not None and (positions_.ndim != 2 or positions_.shape[1] != 3):
            raise ValueError("positions must have shape (N, 3).")
        cell = _optional_array(cell_vectors, np.float64, "cell_vectors")
        if cell is not None and (cell.ndim != 2 or cell.shape[1] != 3):
            raise ValueError("cell_vectors must have shape (r, 3).")
        momenta_ = _optional_array(momenta, np.float64, "momenta")
        if (
            momenta_ is not None
            and positions_ is not None
            and momenta_.shape != positions_.shape
        ):
            raise ValueError("momenta must match positions.")
        path_ = _optional_array(path, np.float64, "path")
        if path_ is not None and (
            path_.ndim != 3 or path_.shape[-1] != 3 or path_.shape[0] < 2
        ):
            raise ValueError("path must have shape (M >= 2, N, 3).")
        if evaluation is not None and not isinstance(
            evaluation, AtomisticEvaluationRecord
        ):
            raise TypeError("evaluation must be an AtomisticEvaluationRecord.")
        time_ = np.asarray(time, dtype=np.float64).reshape(())
        if not np.isfinite(time_):
            raise ValueError("time must be finite.")
        self.positions = positions_
        self.cell_vectors = cell
        self.momenta = momenta_
        self.species = _optional_array(species, np.int32, "species")
        self.atom_type_ids = _optional_array(atom_type_ids, np.int32, "atom_type_ids")
        self.atom_mask = _optional_array(atom_mask, np.bool_, "atom_mask")
        self.path = path_
        self.path_energies = _optional_array(path_energies, np.float64, "path_energies")
        self.time = time_
        self.evaluation = evaluation
        self.continuation = continuation
        self.child_id = None if child_id is None else _identifier(child_id, "child_id")
        self.family = None if family is None else _identifier(family, "family")

    def ports(self) -> dict[str, np.ndarray]:
        """Published port arrays (exact keys of `AtomisticCampaignOutput.ports`)."""
        ports: dict[str, np.ndarray] = {}
        for name in (
            "positions",
            "cell_vectors",
            "momenta",
            "species",
            "atom_type_ids",
            "atom_mask",
            "path",
            "path_energies",
        ):
            value = getattr(self, name)
            if value is not None:
                ports[name] = value
        if self.evaluation is not None:
            ports["energy"] = self.evaluation.energy
            ports["forces"] = self.evaluation.forces
            if self.evaluation.stress is not None:
                ports["stress"] = self.evaluation.stress
        ports["time"] = self.time
        return ports

    def stripped(self) -> AtomisticTaskPacket:
        """The same configuration without method continuation or spawn identity."""
        return AtomisticTaskPacket(
            positions=self.positions,
            cell_vectors=self.cell_vectors,
            momenta=self.momenta,
            species=self.species,
            atom_type_ids=self.atom_type_ids,
            atom_mask=self.atom_mask,
            path=self.path,
            path_energies=self.path_energies,
            time=self.time,
            evaluation=self.evaluation,
        )


def _packet_digest(packet: AtomisticTaskPacket, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-task-packet",
            "ports": array_tree_fingerprint(packet.ports()),
            "evaluation": None
            if packet.evaluation is None
            else packet.evaluation.key.key_id,
            "child_id": packet.child_id,
            "family": packet.family,
            "continuation": _continuation_digest(packet.continuation),
        }
    )


def _continuation_digest(continuation: Any, /) -> Any:
    if continuation is None:
        return None
    if isinstance(continuation, AtomisticDynamicsContinuation):
        return {
            "kind": "atomistic-dynamics-continuation",
            "source": continuation.source_binding_id,
            "layout": continuation.layout_id,
            "lineage": list(continuation.lineage),
            "state": array_tree_fingerprint(continuation.state),
        }
    return array_tree_fingerprint(jax.device_get(continuation))


# -- signatures -------------------------------------------------------------------


@final
class AtomisticStageSignature(StrictModule, NonTrainableState):
    """Static executable signature of one operation lane bank.

    Built on `PoolExecutionSignature`; ``topology``, ``resources`` and
    ``model`` are the system, bundle and Hamiltonian ``layout_id`` values, so
    it contains no scientific identity, numeric revision, task, material
    record or budget, and equal signatures share one executable.
    """

    pool: PoolExecutionSignature
    kind: str = eqx.field(static=True)
    lanes: int = eqx.field(static=True)
    facts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: AtomisticStageOperationKind,
        lanes: int,
        /,
        *,
        topology: str,
        resources: str,
        model: str,
        derivatives: str,
        precision: str,
        method_state: str,
        execution_group: ExecutionGroupSpec | None = None,
    ) -> None:
        kind_ = AtomisticStageOperationKind(kind)
        count = int(lanes)
        if count < 1:
            raise ValueError("lanes must be positive.")
        facts = (
            ("topology", str(topology)),
            ("resources", str(resources)),
            ("model", str(model)),
            ("derivatives", str(derivatives)),
            ("precision", str(precision)),
            ("method_state", str(method_state)),
        )
        self.pool = PoolExecutionSignature(
            topology_id=canonical_fingerprint(
                {
                    "kind": "atomistic-stage-topology",
                    "topology": str(topology),
                    "resources": str(resources),
                    "lanes": count,
                }
            ),
            method_id=canonical_fingerprint(
                {
                    "kind": "atomistic-stage-method",
                    "operation": kind_.value,
                    "model": str(model),
                    "derivatives": str(derivatives),
                    "method_state": str(method_state),
                }
            ),
            precision_id=str(precision),
            backend_id=jax.default_backend(),
            execution_group=execution_group,
        )
        self.kind = kind_.value
        self.lanes = count
        self.facts = facts
        self.signature_id = self.pool.signature_id


# -- lane utilities ---------------------------------------------------------------


def _lane_slice(tree: Any, lane: int, /) -> Any:
    return jax.tree.map(lambda leaf: leaf[lane], tree)


def _lane_broadcast(tree: Any, lanes: int, /) -> Any:
    return jax.tree.map(
        lambda leaf: jnp.repeat(jnp.asarray(leaf)[None], lanes, axis=0), tree
    )


def _lane_insert(bank: Any, lane: int, single: Any, /) -> Any:
    """Write one lane of ``bank`` from ``single`` under the bank's identities."""
    bank_leaves, structure = jax.tree_util.tree_flatten(bank)
    single_leaves = jax.tree_util.tree_leaves(single)
    if len(bank_leaves) != len(single_leaves):
        raise ValueError("The lane state does not match the bank structure.")
    leaves = []
    for target, value in zip(bank_leaves, single_leaves, strict=True):
        source = jnp.asarray(value)
        if tuple(source.shape) != tuple(target.shape[1:]) or source.dtype != target.dtype:
            raise ValueError("The lane state shape or dtype differs from the bank.")
        leaves.append(target.at[lane].set(source))
    return jax.tree_util.tree_unflatten(structure, leaves)


def _lane_where(mask: Array, new: Any, old: Any, /) -> Any:
    def select(left: Array, right: Array) -> Array:
        shape = mask.shape + (1,) * (left.ndim - mask.ndim)
        return jnp.where(jnp.reshape(mask, shape), left, right)

    return jax.tree.map(select, new, old)


def _is_identity(value: Any, /) -> bool:
    return isinstance(value, IdentityWords)


def _group_merge(mask: np.ndarray, fresh: Any, bank: Any, /) -> Any:
    """Select the masked groups of a native group-batched state from ``fresh``.

    Identity words describe the whole bank (plan, binding, lineage), not a
    group: host admission requires them to be exactly equal and keeps them.
    """
    fresh_leaves, fresh_structure = jax.tree_util.tree_flatten(
        fresh, is_leaf=_is_identity
    )
    bank_leaves, bank_structure = jax.tree_util.tree_flatten(bank, is_leaf=_is_identity)
    if fresh_structure != bank_structure:
        raise ValueError("Admitted groups belong to another native runtime binding.")
    lanes = int(mask.shape[0])
    selector = jnp.asarray(mask)
    leaves = []
    for new, old in zip(fresh_leaves, bank_leaves, strict=True):
        if isinstance(new, IdentityWords):
            if new.words.shape != old.words.shape or not np.array_equal(
                np.asarray(new.words), np.asarray(old.words)
            ):
                raise ValueError(
                    "Admitted groups belong to another native runtime binding."
                )
            leaves.append(old)
        elif new.ndim >= 1 and new.shape[0] == lanes and new.shape == old.shape:
            shape = (lanes,) + (1,) * (new.ndim - 1)
            leaves.append(jnp.where(jnp.reshape(selector, shape), new, old))
        else:
            leaves.append(old)
    return jax.tree_util.tree_unflatten(bank_structure, leaves)


def _capacity_kinds(state: AtomisticNeighborhoodBundleState, /) -> int:
    """Separated source-search versus role-storage capacity causes."""
    reasons = 0
    for view in state.views:
        evidence = view.evidence
        if bool(np.any(np.asarray(evidence.source_capacity_failure))):
            reasons |= int(AtomisticCampaignReason.SOURCE_CAPACITY)
        if bool(np.any(np.asarray(evidence.route_overflow | evidence.degree_overflow))):
            reasons |= int(AtomisticCampaignReason.ROLE_CAPACITY)
    return reasons


def _rebuilt_bundle(
    bundle: PreparedAtomisticNeighborhoodBundle,
    levels: tuple[tuple[int, ...], tuple[int, ...]],
    /,
) -> PreparedAtomisticNeighborhoodBundle:
    sources, roles = levels
    if tuple(sources) == bundle.source_levels and tuple(roles) == bundle.role_levels:
        return bundle
    return PreparedAtomisticNeighborhoodBundle(
        bundle.plan,
        bundle.system,
        graph_plans=bundle.graph_plans,
        source_levels=tuple(int(value) for value in sources),
        role_levels=tuple(int(value) for value in roles),
    )


def _system_ports(system: PreparedAtomisticSystem, /) -> dict[str, np.ndarray]:
    plan = system.plan
    return {
        "species": np.asarray(plan.atomic_numbers, dtype=np.int32),
        "atom_type_ids": np.asarray(plan.atom_type_ids, dtype=np.int32),
        "atom_mask": np.asarray(system.active_mask, dtype=np.bool_),
    }


def _system_cell(system: PreparedAtomisticSystem, /) -> np.ndarray | None:
    cell = system.cell
    return None if cell is None else np.asarray(cell.vectors, dtype=np.float64)


def _packet_positions(
    packet: AtomisticTaskPacket, system: PreparedAtomisticSystem, /
) -> np.ndarray | None:
    positions = packet.positions
    if positions is None or positions.shape != (system.capacity, 3):
        return None
    active = np.asarray(system.active_mask, dtype=np.bool_)
    if not np.all(np.isfinite(positions[active])):
        return None
    return positions


# -- operation protocol -----------------------------------------------------------


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticLaneAdmission:
    """Host result of admitting packets into chosen lanes.

    ``refused[i]`` packets were not admitted and become terminal with
    ``refusal[i]``; ``transfers`` name packets that must be admitted to another
    operation instead (capacity found at admission).  ``lineage`` holds the
    owner Hamiltonian transitions applied to an admitted continuation
    (packet index, transition lineage IDs in order).
    """

    lanes: Any
    refused: np.ndarray
    refusal: np.ndarray
    reasons: np.ndarray
    shared: np.ndarray
    transfers: tuple[tuple[int, AbstractAtomisticStageOperation], ...] = ()
    method_reasons: np.ndarray | None = None
    lineage: tuple[tuple[int, tuple[str, ...]], ...] = ()


class AtomisticLaneAdvance(StrictModule):
    lanes: Any
    outcome: Array
    consumed: Array
    reasons: Array


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticCapacityTransfer:
    """Capacity repair of one lane: a target operation and packet, or a refusal."""

    slot: int
    operation: AbstractAtomisticStageOperation | None
    packet: AtomisticTaskPacket | None
    outcome: int
    reasons: int


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticCapacityGrowth:
    """Result of an operation's capacity lifecycle on its bank."""

    operation: AbstractAtomisticStageOperation
    lanes: Any
    transfers: tuple[AtomisticCapacityTransfer, ...] = ()
    outcome: np.ndarray | None = None


class AbstractAtomisticStageOperation(StrictModule):
    """Shared lane executor of one prepared native scientific runtime.

    Every lane-state leaf carries the leading lane axis.  ``advance`` is pure
    and leaves inactive or terminal lanes bitwise unchanged; ``admit``,
    ``export``, ``grow`` and ``rebind`` are host transactions.

    Operations are passed whole into the compiled entrypoints, so their
    treedefs are content-free: the binding digest (``operation_identity``)
    and the numeric revisions (stacked ``revision_identities``) are dynamic
    `IdentityWords`; ``binding_id``/``numeric_revision_ids`` decode the exact
    strings on the host.  Bank grouping is the layout-only `signature`.
    """

    operation_identity: eqx.AbstractVar[IdentityWords]
    revision_identities: eqx.AbstractVar[IdentityWords]

    @property
    @abc.abstractmethod
    def kind(self) -> AtomisticStageOperationKind:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capacity_mode(self) -> AtomisticCapacityMode:
        raise NotImplementedError

    @property
    def operation_id(self) -> str:
        """Prepared runtime binding: resources, numeric content and method controls."""
        return self.operation_identity.fingerprint()

    @property
    def binding_id(self) -> str:
        return self.operation_id

    @property
    def revision_ids(self) -> tuple[str, ...]:
        return _identity_strings(self.revision_identities)

    @property
    def numeric_revision_ids(self) -> tuple[str, ...]:
        return self.revision_ids

    @property
    @abc.abstractmethod
    def supports_rebind(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def supports_shared_entry(self) -> bool:
        raise NotImplementedError

    @property
    def native_lanes(self) -> int | None:
        """Lane count fixed by a natively group-batched runtime (else ``None``)."""
        return None

    @property
    def preserves_configuration(self) -> bool:
        """Whether lanes evaluate the admitted configuration without moving it.

        A walker's dynamics continuation then passes through the stage.
        """
        return False

    @abc.abstractmethod
    def signature(self, lanes: int, /) -> AtomisticStageSignature:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluation_key(self) -> AtomisticEvaluationKey | None:
        raise NotImplementedError

    @abc.abstractmethod
    def admit(
        self,
        lanes: Any,
        lane_count: int,
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        keys: Array,
        ordinals: np.ndarray,
        /,
        *,
        sources: tuple[AbstractAtomisticStageOperation, ...] = (),
    ) -> AtomisticLaneAdmission:
        """Admit packets into ``slots``.

        ``sources`` are the campaign's plan and resident operations; one of
        them may own the runtime an incoming continuation is bound to.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def advance(
        self, lanes: Any, active: Array, budget: Array, consumed: Array, /
    ) -> AtomisticLaneAdvance:
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, lanes: Any, slot: int, /) -> tuple[AtomisticTaskPacket, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def grow(self, lanes: Any, slots: np.ndarray, /) -> AtomisticCapacityGrowth:
        raise NotImplementedError

    @abc.abstractmethod
    def rebind(
        self,
        lanes: Any,
        occupied: np.ndarray,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> tuple[AbstractAtomisticStageOperation, Any, dict[int, str]]:
        """Explicit owner Hamiltonian transition of every occupied lane (``REPRIME``).

        Atomic: if any occupied lane's transition fails a ``ValueError``
        carries the owner's reason and the bank is untouched.  Returns the
        bank's operation after the transition, its lanes (``None`` when no lane
        is occupied) and the owner's lineage ID of every transitioned lane.
        """
        raise NotImplementedError

    @property
    def hamiltonian_lineage(self) -> tuple[str, ...]:
        """Owner transition history bound into a natively batched bank state."""
        return ()

    def with_lineage(
        self, lineage: tuple[str, ...], /
    ) -> AbstractAtomisticStageOperation:
        """The same binding carrying a native bank lineage (restart reconstruction)."""
        if tuple(lineage):
            raise ValueError(
                f"{self.kind.value} banks carry no native Hamiltonian lineage."
            )
        return self

    @abc.abstractmethod
    def lane_random_words(self, lanes: Any, slot: int, /) -> np.ndarray | None:
        """Method-owned RNG identity words of a lane (``None``: admission key)."""
        raise NotImplementedError

    @abc.abstractmethod
    def capacity_levels(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def with_capacity_levels(
        self, levels: tuple[tuple[int, ...], tuple[int, ...]], /
    ) -> AbstractAtomisticStageOperation:
        raise NotImplementedError

    @abc.abstractmethod
    def lane_template(
        self, lane_count: int, packet: AtomisticTaskPacket | None, /
    ) -> Any:
        """Bank structure for restoring a checkpoint (abstract or concrete leaves).

        ``packet`` is the bank's stored template configuration (no
        continuation, no evaluation, the system cell) for operations whose
        structure needs a concrete host admission.
        """
        raise NotImplementedError


def _no_admission(count: int, lanes: Any, /) -> AtomisticLaneAdmission:
    return AtomisticLaneAdmission(
        lanes,
        np.zeros((count,), dtype=np.bool_),
        np.zeros((count,), dtype=np.int32),
        np.zeros((count,), dtype=np.int32),
        np.zeros((count,), dtype=np.bool_),
    )


def _refuse(admission_arrays: tuple[np.ndarray, ...], index: int, reason: int, /) -> None:
    refused, refusal, reasons, _ = admission_arrays
    refused[index] = True
    refusal[index] = int(AtomisticStageOutcome.FAILED)
    reasons[index] = int(reason) | int(AtomisticCampaignReason.ADMISSION_REFUSED)


def _abstract_lanes(single: Any, lanes: int, /) -> Any:
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct((int(lanes), *tuple(leaf.shape)), leaf.dtype),
        single,
    )


def _template_packet(packet: AtomisticTaskPacket, /) -> AtomisticTaskPacket:
    """Configuration-only packet that fixes a bank's structure (system cell)."""
    return AtomisticTaskPacket(
        positions=packet.positions,
        species=packet.species,
        atom_type_ids=packet.atom_type_ids,
        atom_mask=packet.atom_mask,
        path=packet.path,
    )


def _concrete_template(
    operation: AbstractAtomisticStageOperation,
    lanes: int,
    packet: AtomisticTaskPacket | None,
    /,
) -> Any:
    if packet is None:
        raise ValueError("This bank's structure needs its stored template packet.")
    admission = operation.admit(
        None,
        int(lanes),
        np.zeros((1,), dtype=np.int32),
        (packet,),
        jax.random.split(jax.random.key(0), 1),
        np.zeros((1,), dtype=np.int64),
    )
    if admission.lanes is None or admission.transfers or bool(admission.refused[0]):
        raise ValueError("The stored template packet is not admissible by the bank.")
    return admission.lanes


# -- dynamics -------------------------------------------------------------------------


_REPAIRABLE_DYNAMICS = CAPACITY_REJECTIONS | int(
    AtomisticStepRejectionReason.POTENTIAL | AtomisticStepRejectionReason.NONFINITE
)


@final
class AtomisticDynamicsLanes(StrictModule):
    """Accepted dynamics states of a bank and the last candidate resources."""

    state: AtomisticDynamicsState
    candidate: AtomisticNeighborhoodBundleState


def _initialize_dynamics_lane(
    operation: AtomisticDynamicsStageOperation,
    positions: Array,
    momenta: Array,
    time: Array,
    key: Array,
    cell_vectors: Array | None,
) -> AtomisticInitialStateEvaluation:
    """Owner admission of one initial state; failures are returned, not raised."""
    return operation.dynamics.initialize_state_detailed(
        positions,
        operation.thermodynamic,
        state_index=operation.state_index,
        momentum=momenta,
        time=time,
        key=key,
        cell_vectors=cell_vectors,
    )


_INITIALIZE_DYNAMICS = eqx.filter_jit(_initialize_dynamics_lane)


def _rebase_dynamics_lane(
    operation: AtomisticDynamicsStageOperation, state: AtomisticDynamicsState
) -> AtomisticThermodynamicRebaseEvaluation:
    """Owner warm rebase of an accepted state onto the operation's table row."""
    return operation.dynamics.rebase_thermodynamic_state(
        state, operation.thermodynamic, operation.state_index
    )


_REBASE_DYNAMICS = eqx.filter_jit(_rebase_dynamics_lane)


@final
class AtomisticDynamicsContinuation(StrictModule, NonTrainableState):
    """Complete native dynamics continuation of one walker and its source binding.

    ``state`` is the accepted `AtomisticDynamicsState` with host leaves:
    controllers (thermostat chains, barostat, GLE memory, Monte Carlo
    counters), step, time, RNG key, images, term states, force cache and
    neighborhood resources.  ``source_binding_id`` is the ``binding_id`` of
    the stage operation whose runtime and table the state is bound to,
    ``layout_id`` that operation's ``continuation_layout_id`` (which fixes
    the state structure) and ``lineage`` the owner Hamiltonian transitions
    applied since the walker left its previous visit.  Build it with
    `AtomisticDynamicsStageOperation.continuation`, which authenticates the
    state's runtime, table and resource words against that operation; a
    consumer adopts or transitions it only through an operation whose
    runtime the state is bound to (never by the binding string alone).
    """

    state: AtomisticDynamicsState
    source_binding_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    lineage: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        state: AtomisticDynamicsState,
        /,
        *,
        source_binding_id: str,
        layout_id: str,
        lineage: Sequence[str] = (),
    ) -> None:
        if not isinstance(state, AtomisticDynamicsState):
            raise TypeError("state must be an AtomisticDynamicsState.")
        self.state = state
        self.source_binding_id = _identifier(source_binding_id, "source_binding_id")
        self.layout_id = _identifier(layout_id, "layout_id")
        self.lineage = tuple(_identifier(value, "lineage") for value in lineage)

    @property
    def prepared_dynamics_id(self) -> str:
        return self.state.prepared_dynamics_id

    @property
    def thermodynamic_table_id(self) -> str:
        return self.state.thermodynamic_table_id


@dataclass(frozen=True, slots=True, eq=False)
class _ContinuationEntry:
    """Host decision of entering one continuation into a dynamics operation.

    Exactly one of: an admitted ``state`` (with appended ``lineage``), a
    capacity ``transfer`` operation, or a refusal (``refusal`` outcome with
    campaign ``reasons`` and owner ``method_reasons``).
    """

    state: AtomisticDynamicsState | None = None
    lineage: tuple[str, ...] = ()
    transfer: AtomisticDynamicsStageOperation | None = None
    refusal: int = int(AtomisticStageOutcome.FAILED)
    reasons: int = 0
    method_reasons: int = 0


def _continuation_refusal(reasons: int, method_reasons: int = 0, /) -> _ContinuationEntry:
    return _ContinuationEntry(reasons=int(reasons), method_reasons=int(method_reasons))


@final
class AtomisticDynamicsStageOperation(AbstractAtomisticStageOperation):
    """Native dynamics of one prepared method over a lane bank.

    Each round advances every active lane by up to ``steps_per_round``
    accepted native transactions (`PreparedAtomisticDynamics.step_detailed`);
    a lane completes when its accepted steps reach its stage budget.  A
    rejection whose causes are only source/route capacity (with the overflow
    induced potential/nonfinite flags) is ``CAPACITY_PENDING``: the accepted
    state, step index and stochastic address are untouched and the task is
    rebucketed to the next declared capacities via `rebind_hamiltonian` and
    `prime_state`.  A packet without continuation is admitted by the owner's
    `initialize_state_detailed` from unwrapped positions, carried momenta and
    the packet's current cell with the campaign's semantic key; an
    owner-refused initial state is an explicit ``FAILED`` terminal outcome
    carrying the owner's rejection reasons, and capacity-only refusals take
    the declared capacity ladder.  A packet carrying an
    `AtomisticDynamicsContinuation` is continued, never re-initialized: a
    state of this runtime and table is adopted bitwise; a state of the same
    runtime at grown capacities moves to that bucket; a resource-only
    change of the same scientific Hamiltonian uses the owner's
    `rebind_hamiltonian` and `prime_state`; a changed Hamiltonian of the same
    method uses the source runtime's `transition_hamiltonian` (lineage
    recorded); another row of the table is entered by the owner's
    `rebase_thermodynamic_state`.  Any other change (integrator, table,
    constraints, strain response, force ports, system) is refused with
    ``CONTINUATION_REFUSED``.  Exported packets always carry the lane's
    complete continuation.

    PyTree: the thermodynamic row ``state_index`` is a dynamic ``int32``
    scalar (host ``state_row``); the Hamiltonian's scientific identity
    (``hamiltonian_identity``), the numeric revisions and the operation
    binding are dynamic `IdentityWords`.  Only ``steps_per_round`` (a scan
    length) is static, so operations of every material and revision of one
    layout share the compiled admission and advance entrypoints.
    """

    dynamics: PreparedAtomisticDynamics
    thermodynamic: PreparedThermodynamicStateTable
    states: tuple[AtomisticThermodynamicStatePlan, ...]
    state_index: Array
    hamiltonian_identity: IdentityWords
    revision_identities: IdentityWords
    operation_identity: IdentityWords
    steps_per_round: int = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        states: Sequence[AtomisticThermodynamicStatePlan],
        /,
        *,
        state_index: int = 0,
        steps_per_round: int = 1,
        _revision_ids: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        rows = tuple(states)
        table = PreparedThermodynamicStateTable(dynamics, rows)
        index = int(state_index)
        if not 0 <= index < table.state_count:
            raise ValueError("state_index must name a row of the thermodynamic table.")
        if int(steps_per_round) < 1:
            raise ValueError("steps_per_round must be positive.")
        revisions = (
            (atomistic_hamiltonian_revision(dynamics.potential).revision_id,)
            if _revision_ids is None
            else tuple(_revision_ids)
        )
        self.dynamics = dynamics
        self.thermodynamic = table
        self.states = rows
        self.state_index = jnp.asarray(index, dtype=jnp.int32)
        self.steps_per_round = int(steps_per_round)
        self.hamiltonian_identity = dynamics.potential.scientific_identity
        self.revision_identities = _identity_rows(revisions)
        self.operation_identity = IdentityWords.of(
            {
                "kind": "atomistic-dynamics-stage-operation",
                "dynamics": dynamics.prepared_id,
                "table": table.table_id,
                "state_index": index,
                "steps_per_round": self.steps_per_round,
                "revisions": list(revisions),
            }
        )

    @property
    def hamiltonian_id(self) -> str:
        return self.hamiltonian_identity.fingerprint()

    @property
    def state_row(self) -> int:
        """Host thermodynamic row index."""
        return int(self.state_index)

    @property
    def kind(self) -> AtomisticStageOperationKind:
        return AtomisticStageOperationKind.DYNAMICS

    @property
    def capacity_mode(self) -> AtomisticCapacityMode:
        return AtomisticCapacityMode.REBUCKET

    @property
    def supports_rebind(self) -> bool:
        # REPRIME uses the owner's explicit `transition_hamiltonian`, never the
        # resource-only `rebind_hamiltonian`.
        return True

    @property
    def supports_shared_entry(self) -> bool:
        return False

    def _replace(
        self, dynamics: PreparedAtomisticDynamics, /
    ) -> AtomisticDynamicsStageOperation:
        return AtomisticDynamicsStageOperation(
            dynamics,
            self.states,
            state_index=self.state_row,
            steps_per_round=self.steps_per_round,
            _revision_ids=self.revision_ids,
        )

    def signature(self, lanes: int, /) -> AtomisticStageSignature:
        system = self.dynamics.system
        return AtomisticStageSignature(
            self.kind,
            lanes,
            topology=system.layout_id,
            resources=self.dynamics.bundle.layout_id,
            model=self.dynamics.potential.layout_id,
            derivatives="energy,forces,strain"
            if self.dynamics.strain_response
            else "energy,forces",
            precision=system.plan.coordinate_dtype,
            method_state=canonical_fingerprint(
                {
                    "dynamics": self.dynamics.layout_id,
                    "table": self.thermodynamic.layout_id,
                    "table_states": self.thermodynamic.state_count,
                    "steps_per_round": self.steps_per_round,
                }
            ),
        )

    def evaluation_key(self) -> AtomisticEvaluationKey:
        row = self.thermodynamic.state_at_replica(self.state_index)
        quantities = (
            ("energy", "forces", "stress")
            if self.dynamics.strain_response
            else ("energy", "forces")
        )
        return AtomisticEvaluationKey(
            self.hamiltonian_id,
            numeric_revision_ids=self.revision_ids,
            conditions_id=_conditions_id(
                np.asarray(row.controls), self.thermodynamic.control_ids
            ),
            bias_id=self.thermodynamic.bias_ids[self.state_row],
            quantities=quantities,
        )

    def admit(
        self,
        lanes: AtomisticDynamicsLanes | None,
        lane_count: int,
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        keys: Array,
        ordinals: np.ndarray,
        /,
        *,
        sources: tuple[AbstractAtomisticStageOperation, ...] = (),
    ) -> AtomisticLaneAdmission:
        del ordinals
        count = len(packets)
        admission = _no_admission(count, lanes)
        arrays = (
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
        )
        system = self.dynamics.system
        dtype = system.plan.coordinate_dtype
        periodic = system.cell is not None
        transfers: list[tuple[int, AbstractAtomisticStageOperation]] = []
        method_reasons = np.zeros((count,), dtype=np.int32)
        lineage: list[tuple[int, tuple[str, ...]]] = []
        bank = lanes
        for index, packet in enumerate(packets):
            continuation = packet.continuation
            if continuation is not None:
                if not isinstance(continuation, AtomisticDynamicsContinuation):
                    _refuse(arrays, index, AtomisticCampaignReason.IDENTITY_MISMATCH)
                    continue
                entry = self._enter(continuation, sources)
                if entry.transfer is not None:
                    transfers.append((index, entry.transfer))
                    continue
                if entry.state is None:
                    _refuse(arrays, index, entry.reasons)
                    admission.refusal[index] = int(entry.refusal)
                    method_reasons[index] = int(entry.method_reasons)
                    continue
                state = entry.state
                if entry.lineage:
                    lineage.append((index, entry.lineage))
            else:
                positions = _packet_positions(packet, system)
                if positions is None:
                    _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                    continue
                if not periodic and packet.cell_vectors is not None:
                    _refuse(arrays, index, AtomisticCampaignReason.CELL_MISMATCH)
                    continue
                momenta = (
                    np.zeros_like(positions) if packet.momenta is None else packet.momenta
                )
                evaluation = _INITIALIZE_DYNAMICS(
                    self,
                    jnp.asarray(positions, dtype=dtype),
                    jnp.asarray(momenta, dtype=dtype),
                    jnp.asarray(packet.time, dtype=dtype),
                    keys[index],
                    None
                    if packet.cell_vectors is None
                    else jnp.asarray(packet.cell_vectors, dtype=dtype),
                )
                if not bool(np.asarray(evaluation.successful)):
                    code = int(np.asarray(evaluation.rejection_reasons))
                    method_reasons[index] = code
                    if code & CAPACITY_REJECTIONS and not code & ~_REPAIRABLE_DYNAMICS:
                        resources = evaluation.state.neighborhoods
                        kinds = _capacity_kinds(resources)
                        try:
                            grown = self.dynamics.bundle.grow_capacities(resources)
                        except (RuntimeError, ValueError):
                            _refuse(
                                arrays,
                                index,
                                kinds | int(AtomisticCampaignReason.LADDER_EXHAUSTED),
                            )
                            admission.refusal[index] = int(
                                AtomisticStageOutcome.CAPACITY_REFUSED
                            )
                            continue
                        transfers.append(
                            (
                                index,
                                self.with_capacity_levels(
                                    (grown.source_levels, grown.role_levels)
                                ),
                            )
                        )
                        continue
                    # Owner-refused initial state: an explicit FAILED terminal
                    # outcome carrying the owner's rejection reasons.
                    _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                    continue
                state = evaluation.state
            single = AtomisticDynamicsLanes(state, state.neighborhoods)
            if bank is None:
                bank = _lane_broadcast(single, lane_count)
            bank = _lane_insert(bank, int(slots[index]), single)
        return AtomisticLaneAdmission(
            bank,
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
            tuple(transfers),
            method_reasons,
            tuple(lineage),
        )

    # -- continuation ---------------------------------------------------------------

    @property
    def continuation_layout_id(self) -> str:
        """Structure of this runtime's dynamics state (dynamics and table layouts)."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-dynamics-continuation-layout",
                "dynamics": self.dynamics.layout_id,
                "table": self.thermodynamic.layout_id,
            }
        )

    def binds(self, state: AtomisticDynamicsState, /) -> bool:
        """Whether ``state`` is bound to this runtime, table and resource bundle."""
        return (
            state.prepared_dynamics_id == self.dynamics.prepared_id
            and state.thermodynamic_table_id == self.thermodynamic.table_id
            and state.neighborhoods.bundle_id == self.dynamics.bundle.prepared_id
        )

    def continuation(
        self, state: AtomisticDynamicsState, /, *, lineage: Sequence[str] = ()
    ) -> AtomisticDynamicsContinuation:
        """Authenticated host continuation of an accepted state of this runtime."""
        if not isinstance(state, AtomisticDynamicsState):
            raise TypeError("state must be an AtomisticDynamicsState.")
        if not self.binds(state):
            raise ValueError(
                "The dynamics state is bound to another runtime, table or resource "
                "bundle; transition it through its own operation first."
            )
        return AtomisticDynamicsContinuation(
            jax.device_get(state),
            source_binding_id=self.binding_id,
            layout_id=self.continuation_layout_id,
            lineage=lineage,
        )

    def continuation_packet(
        self, continuation: AtomisticDynamicsContinuation, /
    ) -> AtomisticTaskPacket:
        """The walker packet (ports, evaluation, continuation) of a continuation."""
        if not isinstance(continuation, AtomisticDynamicsContinuation):
            raise TypeError("continuation must be an AtomisticDynamicsContinuation.")
        if continuation.source_binding_id != self.binding_id or not self.binds(
            continuation.state
        ):
            raise ValueError("The continuation is bound to another stage operation.")
        return self._packet(continuation.state, lineage=continuation.lineage)

    def state_template(self) -> AtomisticDynamicsState:
        """Shape/dtype structure of one accepted state (restoring continuations)."""
        system = self.dynamics.system
        dtype = jnp.dtype(system.plan.coordinate_dtype)
        shape = jax.ShapeDtypeStruct((system.capacity, 3), dtype)
        return eqx.filter_eval_shape(
            _initialize_dynamics_lane,
            self,
            shape,
            shape,
            jax.ShapeDtypeStruct((), dtype),
            jax.random.key(0),
            None,
        ).state

    def transition_packet(
        self,
        packet: AtomisticTaskPacket,
        target: AtomisticDynamicsStageOperation,
        /,
    ) -> AtomisticTaskPacket:
        """Continue a walker of this operation under ``target`` (host transaction).

        The canonical promotion path while this (source) runtime is still
        available: the same rules as campaign admission (adoption, resource
        rebinding, the owner's ``transition_hamiltonian`` with recorded
        lineage, warm table-row rebase).  Refusals raise ``ValueError`` with
        the campaign and owner rejection reasons; nothing is re-initialized.
        """
        if not isinstance(target, AtomisticDynamicsStageOperation):
            raise TypeError("target must be an AtomisticDynamicsStageOperation.")
        continuation = packet.continuation
        if (
            not isinstance(continuation, AtomisticDynamicsContinuation)
            or continuation.source_binding_id != self.binding_id
            or not self.binds(continuation.state)
        ):
            raise ValueError("The packet carries no continuation of this operation.")
        entry = target._enter(continuation, (self,))
        if entry.transfer is not None:
            raise ValueError(
                "The continued walker needs the target's grown capacities "
                f"{entry.transfer.capacity_levels()}; continue into that operation."
            )
        if entry.state is None:
            raise ValueError(
                "The dynamics continuation was refused: campaign reasons "
                f"{AtomisticCampaignReason(entry.reasons)!r}, owner rejection "
                f"reasons {entry.method_reasons}."
            )
        return target._packet(entry.state, lineage=entry.lineage)

    def _enter(
        self,
        continuation: AtomisticDynamicsContinuation,
        sources: tuple[AbstractAtomisticStageOperation, ...],
        /,
    ) -> _ContinuationEntry:
        """Owner transition of an incoming continuation into this runtime."""
        state = continuation.state
        lineage = continuation.lineage
        if not self.binds(state):
            source = next(
                (
                    operation
                    for operation in sources
                    if isinstance(operation, AtomisticDynamicsStageOperation)
                    and operation.binding_id == continuation.source_binding_id
                    and operation.binds(state)
                ),
                None,
            )
            if source is None:
                # No available runtime reproduces the continuation's binding.
                return _continuation_refusal(AtomisticCampaignReason.IDENTITY_MISMATCH)
            try:
                bucket = self.with_capacity_levels(source.capacity_levels())
            except (IndexError, RuntimeError, TypeError, ValueError):
                bucket = self
            if bucket is not self and bucket.binds(state):
                return _ContinuationEntry(transfer=bucket)
            moved = self._moved(source, state)
            if moved.state is None:
                return moved
            state = moved.state
            lineage = (*lineage, *moved.lineage)
        if int(np.asarray(state.thermodynamic_state_index)) != self.state_row:
            rebase = _REBASE_DYNAMICS(self, state)
            if not bool(np.asarray(rebase.successful)):
                return _continuation_refusal(AtomisticCampaignReason.REPRIME_FAILED)
            state = rebase.accepted_state
        return _ContinuationEntry(state=state, lineage=lineage)

    def _moved(
        self, source: AtomisticDynamicsStageOperation, state: AtomisticDynamicsState, /
    ) -> _ContinuationEntry:
        """Resource rebinding or Hamiltonian transition of a source-bound state."""
        mine, theirs = self.dynamics, source.dynamics
        same_method = (
            theirs.system.prepared_id == mine.system.prepared_id
            and theirs.integrator.plan_id == mine.integrator.plan_id
            and theirs.schedule.plan_id == mine.schedule.plan_id
            and (None if theirs.constraints is None else theirs.constraints.prepared_id)
            == (None if mine.constraints is None else mine.constraints.prepared_id)
            and theirs.strain_response == mine.strain_response
            and tuple(port.port_id for port in theirs.force_ports)
            == tuple(port.port_id for port in mine.force_ports)
        )
        if not same_method:
            # An integrator, schedule, constraint, port or system change is a
            # new run; only an explicit RESTART stage may begin one.
            return _continuation_refusal(AtomisticCampaignReason.CONTINUATION_REFUSED)
        transition_lineage: tuple[str, ...] = ()
        try:
            if theirs.potential.scientific_id == mine.potential.scientific_id:
                dynamics, table, rebound = theirs.rebind_hamiltonian(
                    mine.potential, state, source.thermodynamic, self.states
                )
                if (
                    dynamics.prepared_id != mine.prepared_id
                    or table.table_id != self.thermodynamic.table_id
                ):
                    return _continuation_refusal(
                        AtomisticCampaignReason.CONTINUATION_REFUSED
                    )
                primed = mine.prime_state(rebound, self.thermodynamic)
                successful = bool(np.asarray(primed.successful))
                code = int(np.asarray(primed.rejection_reasons))
                moved = primed.accepted_state
            else:
                transition = theirs.transition_hamiltonian(
                    mine.potential, state, source.thermodynamic, self.states
                )
                successful = bool(np.asarray(transition.successful))
                code = int(np.asarray(transition.rejection_reasons))
                if successful and (
                    transition.dynamics.prepared_id != mine.prepared_id
                    or transition.thermodynamic.table_id != self.thermodynamic.table_id
                ):
                    return _continuation_refusal(
                        AtomisticCampaignReason.CONTINUATION_REFUSED
                    )
                moved = transition.state
                transition_lineage = (transition.lineage_id,)
        except ValueError:
            # The owner refused the change (table intensive variables, term
            # owners or resource layout): never a silent re-initialization.
            return _continuation_refusal(AtomisticCampaignReason.CONTINUATION_REFUSED)
        if successful:
            return _ContinuationEntry(state=moved, lineage=transition_lineage)
        if code & CAPACITY_REJECTIONS and not code & ~_REPAIRABLE_DYNAMICS:
            resources = mine.initialize_neighborhoods(
                state.kinematics.positions,
                cell_vectors=state.cell_vectors,
                image_counts=state.kinematics.image_counts,
            )
            kinds = _capacity_kinds(resources)
            try:
                grown = mine.bundle.grow_capacities(resources)
            except (RuntimeError, ValueError):
                return _ContinuationEntry(
                    refusal=int(AtomisticStageOutcome.CAPACITY_REFUSED),
                    reasons=kinds | int(AtomisticCampaignReason.LADDER_EXHAUSTED),
                    method_reasons=code,
                )
            return _ContinuationEntry(
                transfer=self.with_capacity_levels(
                    (grown.source_levels, grown.role_levels)
                )
            )
        return _continuation_refusal(AtomisticCampaignReason.REPRIME_FAILED, code)

    def advance(
        self,
        lanes: AtomisticDynamicsLanes,
        active: Array,
        budget: Array,
        consumed: Array,
        /,
    ) -> AtomisticLaneAdvance:
        dynamics = self.dynamics
        table = self.thermodynamic
        start = jnp.where(
            active & (consumed >= budget),
            int(AtomisticStageOutcome.COMPLETED),
            int(AtomisticStageOutcome.ACTIVE),
        ).astype(jnp.int32)

        def body(
            carry: tuple[AtomisticDynamicsLanes, Array, Array, Array], _: None
        ) -> tuple[tuple[AtomisticDynamicsLanes, Array, Array, Array], None]:
            current, used, outcome, reasons = carry
            run = (
                active & (outcome == int(AtomisticStageOutcome.ACTIVE)) & (used < budget)
            )
            evaluation = jax.vmap(lambda state: dynamics.step_detailed(state, table))(
                current.state
            )
            ok = evaluation.successful
            code = evaluation.rejection_reasons
            state = _lane_where(run, evaluation.accepted_state, current.state)
            candidate = _lane_where(
                run & ~ok, evaluation.candidate_state.neighborhoods, current.candidate
            )
            capacity = (
                run
                & ~ok
                & ((code & CAPACITY_REJECTIONS) != 0)
                & ((code & ~_REPAIRABLE_DYNAMICS) == 0)
            )
            failed = run & ~ok & ~capacity
            used = used + (run & ok).astype(jnp.int32)
            done = run & ok & (used >= budget)
            outcome = jnp.where(
                capacity,
                int(AtomisticStageOutcome.CAPACITY_PENDING),
                jnp.where(
                    failed,
                    int(AtomisticStageOutcome.FAILED),
                    jnp.where(done, int(AtomisticStageOutcome.COMPLETED), outcome),
                ),
            ).astype(jnp.int32)
            reasons = jnp.where(run & ~ok, code, reasons).astype(jnp.int32)
            return (
                AtomisticDynamicsLanes(state, candidate),
                used,
                outcome,
                reasons,
            ), None

        (final_lanes, used, outcome, reasons), _ = jax.lax.scan(
            body,
            (
                lanes,
                consumed.astype(jnp.int32),
                start,
                jnp.zeros_like(consumed, dtype=jnp.int32),
            ),
            None,
            length=self.steps_per_round,
        )
        return AtomisticLaneAdvance(final_lanes, outcome, used, reasons)

    def _packet(
        self, state: AtomisticDynamicsState, /, *, lineage: Sequence[str] = ()
    ) -> AtomisticTaskPacket:
        """Walker packet of an accepted state: ports plus its complete continuation."""
        continuation = self.continuation(state, lineage=lineage)
        host = continuation.state
        system = self.dynamics.system
        positions = np.asarray(host.kinematics.positions, dtype=np.float64)
        cell = None
        if system.cell is not None:
            cell = np.asarray(host.cell_vectors, dtype=np.float64)
            positions = (
                positions
                + np.asarray(host.kinematics.image_counts, dtype=np.float64) @ cell
            )
        force = host.force
        evaluation = None
        if bool(force.successful) and bool(np.isfinite(force.potential_energy)):
            key = self.evaluation_key()
            revisions = getattr(force, "term_revisions", None)
            if revisions is not None and np.asarray(revisions).size:
                key = AtomisticEvaluationKey(
                    key.hamiltonian_id,
                    numeric_revision_ids=key.numeric_revision_ids,
                    conditions_id=canonical_fingerprint(
                        {
                            "kind": "atomistic-evaluation-term-conditions",
                            "conditions": key.conditions_id,
                            "term_revisions": np.asarray(revisions).reshape(-1).tolist(),
                        }
                    ),
                    bias_id=key.bias_id,
                    quantities=key.quantities,
                )
            available = bool(force.strain_available)
            stress = np.asarray(force.stress) if available else None
            if not available and "stress" in key.quantities:
                key = key.restricted(("energy", "forces"))
            evaluation = AtomisticEvaluationRecord(
                force.potential_energy, force.forces, stress, key
            )
        ports = _system_ports(system)
        return AtomisticTaskPacket(
            positions=positions,
            cell_vectors=cell,
            momenta=np.asarray(host.kinematics.momenta),
            species=ports["species"],
            atom_type_ids=np.asarray(host.species),
            atom_mask=ports["atom_mask"],
            time=np.asarray(host.time),
            evaluation=evaluation,
            continuation=continuation,
        )

    def export(
        self, lanes: AtomisticDynamicsLanes, slot: int, /
    ) -> tuple[AtomisticTaskPacket, ...]:
        return (self._packet(_lane_slice(lanes.state, int(slot))),)

    def grow(
        self, lanes: AtomisticDynamicsLanes, slots: np.ndarray, /
    ) -> AtomisticCapacityGrowth:
        transfers: list[AtomisticCapacityTransfer] = []
        for slot in (int(value) for value in slots):
            candidate = _lane_slice(lanes.candidate, slot)
            state = _lane_slice(lanes.state, slot)
            kinds = _capacity_kinds(candidate)
            try:
                grown = self.dynamics.bundle.grow_capacities(candidate)
            except (RuntimeError, ValueError):
                transfers.append(
                    AtomisticCapacityTransfer(
                        slot,
                        None,
                        None,
                        int(AtomisticStageOutcome.CAPACITY_REFUSED),
                        kinds | int(AtomisticCampaignReason.LADDER_EXHAUSTED),
                    )
                )
                continue
            dynamics, table, rebound = self.dynamics.rebind_hamiltonian(
                self.dynamics.potential.with_neighborhood_bundle(grown),
                state,
                self.thermodynamic,
                self.states,
            )
            primed = dynamics.prime_state(rebound, table)
            if not bool(np.asarray(primed.successful)):
                transfers.append(
                    AtomisticCapacityTransfer(
                        slot,
                        None,
                        None,
                        int(AtomisticStageOutcome.FAILED),
                        kinds | int(AtomisticCampaignReason.REPRIME_FAILED),
                    )
                )
                continue
            operation = self._replace(dynamics)
            transfers.append(
                AtomisticCapacityTransfer(
                    slot,
                    operation,
                    operation._packet(primed.accepted_state),
                    int(AtomisticStageOutcome.ACTIVE),
                    kinds,
                )
            )
        return AtomisticCapacityGrowth(self, lanes, tuple(transfers))

    def rebind(
        self,
        lanes: AtomisticDynamicsLanes,
        occupied: np.ndarray,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> tuple[
        AbstractAtomisticStageOperation, AtomisticDynamicsLanes | None, dict[int, str]
    ]:
        """Owner `transition_hamiltonian` of every occupied lane (atomic).

        Configuration, momenta, time, step, configuration revision, RNG,
        controllers and counters are preserved by the owner; forces are
        re-primed and the energy jump is booked as external work.
        """
        if not isinstance(operation, AtomisticDynamicsStageOperation):
            raise TypeError("A dynamics bank transitions only to a dynamics operation.")
        if operation.state_row != self.state_row:
            raise ValueError("A Hamiltonian transition keeps the thermodynamic row.")
        transitioned: list[tuple[int, AtomisticDynamicsState]] = []
        lineage: dict[int, str] = {}
        for lane in (int(value) for value in np.flatnonzero(occupied)):
            transition = self.dynamics.transition_hamiltonian(
                operation.dynamics.potential,
                _lane_slice(lanes.state, lane),
                self.thermodynamic,
                operation.states,
            )
            if not bool(np.asarray(transition.successful)):
                raise ValueError(
                    f"The Hamiltonian transition of lane {lane} failed; rejection "
                    f"reasons {int(np.asarray(transition.rejection_reasons))}."
                )
            if (
                transition.dynamics.prepared_id != operation.dynamics.prepared_id
                or transition.thermodynamic.table_id != operation.thermodynamic.table_id
            ):
                raise ValueError(
                    "The transitioned runtime differs from the target operation."
                )
            transitioned.append((lane, transition.state))
            lineage[lane] = transition.lineage_id
        if not transitioned:
            return operation, None, lineage
        first = transitioned[0][1]
        bank = _lane_broadcast(
            AtomisticDynamicsLanes(first, first.neighborhoods), int(occupied.shape[0])
        )
        for lane, state in transitioned:
            bank = _lane_insert(
                bank, lane, AtomisticDynamicsLanes(state, state.neighborhoods)
            )
        return operation, bank, lineage

    def lane_template(
        self, lane_count: int, packet: AtomisticTaskPacket | None, /
    ) -> Any:
        del packet
        single = self.state_template()
        return _abstract_lanes(
            AtomisticDynamicsLanes(single, single.neighborhoods), lane_count
        )

    def lane_random_words(
        self, lanes: AtomisticDynamicsLanes, slot: int, /
    ) -> np.ndarray:
        return np.asarray(
            jax.device_get(lanes.state.random_key[int(slot)]), dtype=np.uint32
        )

    def capacity_levels(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        bundle = self.dynamics.bundle
        return bundle.source_levels, bundle.role_levels

    def with_capacity_levels(
        self, levels: tuple[tuple[int, ...], tuple[int, ...]], /
    ) -> AtomisticDynamicsStageOperation:
        bundle = _rebuilt_bundle(self.dynamics.bundle, levels)
        if bundle is self.dynamics.bundle:
            return self
        dynamics = AtomisticDynamicsPlan(
            self.dynamics.system,
            self.dynamics.potential.with_neighborhood_bundle(bundle),
            self.dynamics.integrator,
            constraints=self.dynamics.constraints,
            strain_response=self.dynamics.strain_response,
        ).prepare()
        return self._replace(dynamics)


# -- relaxation -----------------------------------------------------------------------


def _optimizer_outcome(status: Array, /) -> Array:
    relaxation = AtomisticRelaxationStatus
    return jnp.select(
        [
            status == int(relaxation.CONVERGED),
            status == int(relaxation.EVALUATION_LIMIT),
            status == int(relaxation.CAPACITY_PENDING),
            (status == int(relaxation.SCIENTIFIC_FAILURE))
            | (status == int(relaxation.CELL_FAILURE)),
            status == int(relaxation.RESOURCE_EXHAUSTED),
        ],
        [
            int(AtomisticStageOutcome.CONVERGED),
            int(AtomisticStageOutcome.EXHAUSTED),
            int(AtomisticStageOutcome.CAPACITY_PENDING),
            int(AtomisticStageOutcome.FAILED),
            int(AtomisticStageOutcome.CAPACITY_REFUSED),
        ],
        int(AtomisticStageOutcome.ACTIVE),
    ).astype(jnp.int32)


@final
class AtomisticRelaxationStageOperation(AbstractAtomisticStageOperation):
    """Native group-batched relaxation; its ``G`` groups are the bank's lanes.

    Convergence, evaluation budget, optimizer state and rollback are the
    owner's (`PreparedAtomisticRelaxation.run`); the stage budget is not
    used.  Capacity is repaired by the owner's bank-wide ``grow``, which
    keeps every optimizer history and re-primes accepted evidence.

    PyTree: the scientific Hamiltonian identity, numeric revisions, binding
    and the bank's Hamiltonian lineage (stacked ``(L, 8)``) are dynamic
    `IdentityWords`; ``rounds_per_segment`` is static.  The lineage depth is
    an array shape (one compile per transition depth, never per content).
    """

    runtime: PreparedAtomisticRelaxation
    hamiltonian_identity: IdentityWords
    revision_identities: IdentityWords
    lineage_identities: IdentityWords
    operation_identity: IdentityWords
    rounds_per_segment: int = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedAtomisticRelaxation,
        /,
        *,
        rounds_per_segment: int = 1,
        hamiltonian_lineage: tuple[str, ...] = (),
        _revision_ids: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(runtime, PreparedAtomisticRelaxation):
            raise TypeError("runtime must be PreparedAtomisticRelaxation.")
        if int(rounds_per_segment) < 1:
            raise ValueError("rounds_per_segment must be positive.")
        revisions = (
            (atomistic_hamiltonian_revision(runtime.program).revision_id,)
            if _revision_ids is None
            else tuple(_revision_ids)
        )
        lineage = tuple(_identifier(value, "lineage") for value in hamiltonian_lineage)
        self.runtime = runtime
        self.rounds_per_segment = int(rounds_per_segment)
        self.hamiltonian_identity = runtime.program.scientific_identity
        self.revision_identities = _identity_rows(revisions)
        self.lineage_identities = _identity_rows(lineage)
        self.operation_identity = IdentityWords.of(
            {
                "kind": "atomistic-relaxation-stage-operation",
                "runtime": runtime.prepared_id,
                "rounds": self.rounds_per_segment,
                "revisions": list(revisions),
                "lineage": list(lineage),
            }
        )

    def _rebuilt(
        self,
        runtime: PreparedAtomisticRelaxation,
        /,
        *,
        lineage: tuple[str, ...] | None = None,
    ) -> AtomisticRelaxationStageOperation:
        return AtomisticRelaxationStageOperation(
            runtime,
            rounds_per_segment=self.rounds_per_segment,
            hamiltonian_lineage=self.hamiltonian_lineage if lineage is None else lineage,
            _revision_ids=self.revision_ids,
        )

    @property
    def hamiltonian_id(self) -> str:
        return self.hamiltonian_identity.fingerprint()

    @property
    def hamiltonian_lineage(self) -> tuple[str, ...]:
        return _identity_strings(self.lineage_identities)

    def with_lineage(
        self, lineage: tuple[str, ...], /
    ) -> AtomisticRelaxationStageOperation:
        lineage_ = tuple(lineage)
        if lineage_ == self.hamiltonian_lineage:
            return self
        return self._rebuilt(self.runtime, lineage=lineage_)

    @property
    def kind(self) -> AtomisticStageOperationKind:
        return AtomisticStageOperationKind.RELAXATION

    @property
    def capacity_mode(self) -> AtomisticCapacityMode:
        return AtomisticCapacityMode.GROW

    @property
    def supports_rebind(self) -> bool:
        return True

    @property
    def supports_shared_entry(self) -> bool:
        return False

    @property
    def native_lanes(self) -> int:
        return self.runtime.plan.group_count

    def signature(self, lanes: int, /) -> AtomisticStageSignature:
        plan = self.runtime.plan
        system = self.runtime.system
        return AtomisticStageSignature(
            self.kind,
            lanes,
            topology=system.layout_id,
            resources=self.runtime.bundle.layout_id,
            model=self.runtime.program.layout_id,
            derivatives="energy,forces,cell"
            if plan.chart.variable_cell
            else "energy,forces",
            precision=system.plan.coordinate_dtype,
            method_state=canonical_fingerprint(
                {
                    "relaxation": self.runtime.layout_id,
                    "rounds": self.rounds_per_segment,
                }
            ),
        )

    def evaluation_key(self) -> AtomisticEvaluationKey:
        program = self.runtime.program
        quantities = (
            ("energy", "forces", "stress")
            if self.runtime.plan.chart.variable_cell
            else ("energy", "forces")
        )
        return AtomisticEvaluationKey(
            self.hamiltonian_id,
            numeric_revision_ids=self.revision_ids,
            conditions_id=_conditions_id(np.zeros((0,)), program.control_ids),
            quantities=quantities,
        )

    def admit(
        self,
        lanes: AtomisticRelaxationState | None,
        lane_count: int,
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        keys: Array,
        ordinals: np.ndarray,
        /,
        *,
        sources: tuple[AbstractAtomisticStageOperation, ...] = (),
    ) -> AtomisticLaneAdmission:
        del keys, ordinals, sources
        groups = self.runtime.plan.group_count
        if int(lane_count) != groups:
            raise ValueError("A relaxation bank has exactly group_count lanes.")
        count = len(packets)
        admission = _no_admission(count, lanes)
        arrays = (
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
        )
        system = self.runtime.system
        chart = self.runtime.plan.chart
        variable = chart.variable_cell
        symmetry = isinstance(chart, AtomisticSymmetryChart)
        cell = _system_cell(system)
        accepted: list[int] = []
        coordinates: dict[int, np.ndarray] = {}
        for index, packet in enumerate(packets):
            if symmetry:
                if packet.continuation is None:
                    _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                    continue
            else:
                admitted = _packet_positions(packet, system)
                if admitted is None:
                    _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                    continue
                coordinates[index] = admitted
            if (
                not variable
                and cell is not None
                and packet.cell_vectors is not None
                and not np.array_equal(packet.cell_vectors, cell)
            ):
                _refuse(arrays, index, AtomisticCampaignReason.CELL_MISMATCH)
                continue
            accepted.append(index)
        if not accepted:
            return admission
        mask = np.zeros((groups,), dtype=np.bool_)
        first = packets[accepted[0]]
        positions = None
        vectors = None
        chart_states = None
        if symmetry:
            chosen = [first.continuation] * groups
            for index in accepted:
                chosen[int(slots[index])] = packets[index].continuation
            chart_states = jax.tree.map(lambda *leaves: jnp.stack(leaves), *chosen)
        else:
            stacked = np.repeat(coordinates[accepted[0]][None], groups, axis=0)
            for index in accepted:
                stacked[int(slots[index])] = coordinates[index]
            positions = stacked
        if variable:
            base = first.cell_vectors if first.cell_vectors is not None else cell
            if base is None:
                raise ValueError("A variable-cell relaxation needs a periodic cell.")
            cells = np.repeat(np.asarray(base)[None], groups, axis=0)
            for index in accepted:
                value = packets[index].cell_vectors
                cells[int(slots[index])] = base if value is None else value
            vectors = cells
        for index in accepted:
            mask[int(slots[index])] = True
        fresh = self.runtime.init(
            positions=positions,
            cell_vectors=vectors,
            chart_states=chart_states,
            active=mask,
            hamiltonian_lineage=self.hamiltonian_lineage,
        )
        bank = fresh if lanes is None else _group_merge(mask, fresh, lanes)
        return AtomisticLaneAdmission(
            bank,
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
        )

    def advance(
        self,
        lanes: AtomisticRelaxationState,
        active: Array,
        budget: Array,
        consumed: Array,
        /,
    ) -> AtomisticLaneAdvance:
        del active, budget, consumed
        state = self.runtime.run(lanes, maximum_rounds=self.rounds_per_segment)
        return AtomisticLaneAdvance(
            state,
            _optimizer_outcome(state.status),
            state.evaluations.astype(jnp.int32),
            state.trial_outcome.astype(jnp.int32),
        )

    def export(
        self, lanes: AtomisticRelaxationState, slot: int, /
    ) -> tuple[AtomisticTaskPacket, ...]:
        index = int(slot)
        host = jax.device_get(
            (
                lanes.positions[index],
                None if lanes.cell_vectors is None else lanes.cell_vectors[index],
                lanes.energy[index],
                lanes.forces[index],
                None if lanes.stress is None else lanes.stress[index],
                lanes.optimizer.has_accepted[index],
            )
        )
        positions, vectors, energy, forces, stress, has_accepted = host
        system = self.runtime.system
        cell = _system_cell(system) if vectors is None else np.asarray(vectors)
        evaluation = None
        if (
            bool(has_accepted)
            and bool(np.isfinite(energy))
            and bool(np.all(np.isfinite(forces)))
        ):
            evaluation = AtomisticEvaluationRecord(
                energy, forces, stress, self.evaluation_key()
            )
        ports = _system_ports(system)
        return (
            AtomisticTaskPacket(
                positions=np.asarray(positions),
                cell_vectors=cell,
                species=ports["species"],
                atom_type_ids=ports["atom_type_ids"],
                atom_mask=ports["atom_mask"],
                evaluation=evaluation,
            ),
        )

    def grow(
        self, lanes: AtomisticRelaxationState, slots: np.ndarray, /
    ) -> AtomisticCapacityGrowth:
        del slots
        runtime, state = self.runtime.grow(lanes)
        operation = self._rebuilt(runtime)
        return AtomisticCapacityGrowth(
            operation, state, (), np.asarray(_optimizer_outcome(state.status))
        )

    def rebind(
        self,
        lanes: AtomisticRelaxationState,
        occupied: np.ndarray,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> tuple[AbstractAtomisticStageOperation, AtomisticRelaxationState, dict[int, str]]:
        """Owner `transition_hamiltonian` of the whole group bank (atomic).

        The owner keeps accepted geometry, re-primes the accepted iterate under
        the new Hamiltonian, resets optimizer history explicitly and keeps the
        budget bookkeeping; vacant groups are carried unchanged.
        """
        if not isinstance(operation, AtomisticRelaxationStageOperation):
            raise TypeError(
                "A relaxation bank transitions only to a relaxation operation."
            )
        result = self.runtime.transition_hamiltonian(operation.runtime.program, lanes)
        record = result.record
        if not record.successful:
            raise ValueError(
                f"The relaxation Hamiltonian transition failed: {record.reason}"
            )
        target = operation.with_lineage(result.state.hamiltonian_lineage)
        if result.runtime.prepared_id != target.runtime.prepared_id:
            raise ValueError(
                "The transitioned runtime differs from the target operation."
            )
        return (
            target,
            result.state,
            {int(lane): record.transition_id for lane in np.flatnonzero(occupied)},
        )

    def lane_template(
        self, lane_count: int, packet: AtomisticTaskPacket | None, /
    ) -> Any:
        return _concrete_template(self, lane_count, packet)

    def lane_random_words(self, lanes: AtomisticRelaxationState, slot: int, /) -> None:
        return None

    def capacity_levels(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        bundle = self.runtime.bundle
        return bundle.source_levels, bundle.role_levels

    def with_capacity_levels(
        self, levels: tuple[tuple[int, ...], tuple[int, ...]], /
    ) -> AtomisticRelaxationStageOperation:
        bundle = _rebuilt_bundle(self.runtime.bundle, levels)
        if bundle is self.runtime.bundle:
            return self
        runtime = PreparedAtomisticRelaxation(
            self.runtime.plan, self.runtime.program.with_neighborhood_bundle(bundle)
        )
        return self._rebuilt(runtime)


# -- grouped reaction paths ---------------------------------------------------------


def _endpoint_structure(
    system: PreparedAtomisticSystem, positions: np.ndarray, /
) -> AtomicStructure:
    plan = system.plan
    cell = system.cell
    return AtomicStructure(
        np.asarray(plan.atomic_numbers, dtype=np.int32),
        np.asarray(positions, dtype=plan.coordinate_dtype),
        np.asarray(plan.masses),
        plan.units.scale,
        particle_ids=np.asarray(plan.particle_ids),
        active_mask=np.asarray(plan.active_mask, dtype=np.bool_),
        cell=None if cell is None else np.asarray(cell.vectors),
        periodic_axes=None
        if cell is None
        else np.asarray(cell.periodic_mask, dtype=np.bool_),
    )


@final
class AtomisticReactionPathStageOperation(AbstractAtomisticStageOperation):
    """Native grouped NEB; every lane is one whole path band.

    A task packet carries its whole initial band (``path``); endpoints are
    its first and last images.  Whole bands are admitted, advanced,
    checkpointed and exported together; image slots are never refilled
    individually.  Climbing, convergence, band-atomic failure and the
    evaluation budget are the owner's.  A band that needs larger capacities
    at admission is transferred to the bank at the grown binding; in flight
    the owner's bank-wide ``grow`` replays capacity-pending bands.

    PyTree: the scientific Hamiltonian identity, numeric revisions, binding
    and the bank's Hamiltonian lineage (stacked ``(L, 8)``) are dynamic
    `IdentityWords`; ``rounds_per_segment`` is static.  The lineage depth is
    an array shape (one compile per transition depth, never per content).
    """

    runtime: Any
    hamiltonian_identity: IdentityWords
    revision_identities: IdentityWords
    lineage_identities: IdentityWords
    operation_identity: IdentityWords
    rounds_per_segment: int = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedAtomisticNudgedElasticBand,
        /,
        *,
        rounds_per_segment: int = 1,
        hamiltonian_lineage: tuple[str, ...] = (),
        _revision_ids: tuple[str, ...] | None = None,
    ) -> None:
        from ..chemistry.reaction._path import PreparedAtomisticNudgedElasticBand

        if not isinstance(runtime, PreparedAtomisticNudgedElasticBand):
            raise TypeError("runtime must be PreparedAtomisticNudgedElasticBand.")
        if int(rounds_per_segment) < 1:
            raise ValueError("rounds_per_segment must be positive.")
        revisions = (
            (atomistic_hamiltonian_revision(runtime.program).revision_id,)
            if _revision_ids is None
            else tuple(_revision_ids)
        )
        lineage = tuple(_identifier(value, "lineage") for value in hamiltonian_lineage)
        self.runtime = runtime
        self.rounds_per_segment = int(rounds_per_segment)
        self.hamiltonian_identity = runtime.program.scientific_identity
        self.revision_identities = _identity_rows(revisions)
        self.lineage_identities = _identity_rows(lineage)
        self.operation_identity = IdentityWords.of(
            {
                "kind": "atomistic-reaction-path-stage-operation",
                "runtime": runtime.prepared_id,
                "rounds": self.rounds_per_segment,
                "revisions": list(revisions),
                "lineage": list(lineage),
            }
        )

    @property
    def hamiltonian_id(self) -> str:
        return self.hamiltonian_identity.fingerprint()

    @property
    def hamiltonian_lineage(self) -> tuple[str, ...]:
        return _identity_strings(self.lineage_identities)

    def with_lineage(
        self, lineage: tuple[str, ...], /
    ) -> AtomisticReactionPathStageOperation:
        lineage_ = tuple(lineage)
        if lineage_ == self.hamiltonian_lineage:
            return self
        return self._replace(self.runtime, lineage=lineage_)

    @property
    def kind(self) -> AtomisticStageOperationKind:
        return AtomisticStageOperationKind.REACTION_PATH

    @property
    def capacity_mode(self) -> AtomisticCapacityMode:
        return AtomisticCapacityMode.GROW

    @property
    def supports_rebind(self) -> bool:
        return True

    @property
    def supports_shared_entry(self) -> bool:
        return False

    @property
    def native_lanes(self) -> int:
        return self.runtime.plan.group_count

    def _replace(
        self, runtime: Any, /, *, lineage: tuple[str, ...] | None = None
    ) -> AtomisticReactionPathStageOperation:
        return AtomisticReactionPathStageOperation(
            runtime,
            rounds_per_segment=self.rounds_per_segment,
            hamiltonian_lineage=self.hamiltonian_lineage if lineage is None else lineage,
            _revision_ids=self.revision_ids,
        )

    def signature(self, lanes: int, /) -> AtomisticStageSignature:
        system = self.runtime.system
        return AtomisticStageSignature(
            self.kind,
            lanes,
            topology=system.layout_id,
            resources=self.runtime.program.neighborhood_bundle.layout_id,
            model=self.runtime.program.layout_id,
            derivatives="energy,forces",
            precision=system.plan.coordinate_dtype,
            method_state=canonical_fingerprint(
                {
                    "path": self.runtime.layout_id,
                    "rounds": self.rounds_per_segment,
                }
            ),
        )

    def evaluation_key(self) -> AtomisticEvaluationKey:
        return AtomisticEvaluationKey(
            self.hamiltonian_id,
            numeric_revision_ids=self.revision_ids,
            conditions_id=_conditions_id(
                np.zeros((0,)), self.runtime.program.control_ids
            ),
            quantities=("energy", "forces"),
        )

    def admit(
        self,
        lanes: Any,
        lane_count: int,
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        keys: Array,
        ordinals: np.ndarray,
        /,
        *,
        sources: tuple[AbstractAtomisticStageOperation, ...] = (),
    ) -> AtomisticLaneAdmission:
        del keys, ordinals, sources
        plan = self.runtime.plan
        groups = plan.group_count
        if int(lane_count) != groups:
            raise ValueError("A reaction-path bank has exactly group_count lanes.")
        count = len(packets)
        admission = _no_admission(count, lanes)
        arrays = (
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
        )
        system = self.runtime.system
        cell = _system_cell(system)
        active_atoms = np.asarray(system.active_mask, dtype=np.bool_)
        transfers: list[tuple[int, AbstractAtomisticStageOperation]] = []
        bank = lanes
        for index, packet in enumerate(packets):
            path = packet.path
            if (
                path is None
                or path.shape != (plan.image_count, system.capacity, 3)
                or not np.all(np.isfinite(path[:, active_atoms]))
            ):
                _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                continue
            if (
                cell is not None
                and packet.cell_vectors is not None
                and not np.array_equal(packet.cell_vectors, cell)
            ):
                _refuse(arrays, index, AtomisticCampaignReason.CELL_MISMATCH)
                continue
            lane = int(slots[index])
            mask = np.zeros((groups,), dtype=np.bool_)
            mask[lane] = True
            endpoints = (
                _endpoint_structure(system, path[0]),
                _endpoint_structure(system, path[-1]),
            )
            try:
                runtime, state, _ = self.runtime.init(
                    [endpoints] * groups,
                    initial_images=np.repeat(path[None], groups, axis=0),
                    active=mask,
                    hamiltonian_lineage=self.hamiltonian_lineage,
                )
            except ValueError:
                _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                continue
            if runtime.prepared_id != self.runtime.prepared_id:
                transfers.append((index, self._replace(runtime)))
                continue
            bank = state if bank is None else _group_merge(mask, state, bank)
        return AtomisticLaneAdmission(
            bank,
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
            tuple(transfers),
        )

    def advance(
        self, lanes: Any, active: Array, budget: Array, consumed: Array, /
    ) -> AtomisticLaneAdvance:
        del active, budget, consumed
        state = self.runtime.run(lanes, maximum_rounds=self.rounds_per_segment)
        return AtomisticLaneAdvance(
            state,
            _optimizer_outcome(state.status),
            state.evaluations.astype(jnp.int32),
            state.trial_outcome.astype(jnp.int32),
        )

    def export(self, lanes: Any, slot: int, /) -> tuple[AtomisticTaskPacket, ...]:
        result = self.runtime.path_result(lanes, int(slot))
        images = np.asarray(result.images, dtype=np.float64)
        energies = np.asarray(result.energies, dtype=np.float64)
        system = self.runtime.system
        ports = _system_ports(system)
        return (
            AtomisticTaskPacket(
                positions=images[result.climbing_image],
                cell_vectors=_system_cell(system),
                species=ports["species"],
                atom_type_ids=ports["atom_type_ids"],
                atom_mask=ports["atom_mask"],
                path=images,
                path_energies=energies,
            ),
        )

    def grow(self, lanes: Any, slots: np.ndarray, /) -> AtomisticCapacityGrowth:
        del slots
        runtime, state = self.runtime.grow(lanes)
        return AtomisticCapacityGrowth(
            self._replace(runtime),
            state,
            (),
            np.asarray(_optimizer_outcome(state.status)),
        )

    def rebind(
        self,
        lanes: Any,
        occupied: np.ndarray,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> tuple[AbstractAtomisticStageOperation, Any, dict[int, str]]:
        """Owner whole-band `transition_hamiltonian` of the path bank (atomic)."""
        if not isinstance(operation, AtomisticReactionPathStageOperation):
            raise TypeError("A path bank transitions only to a reaction-path operation.")
        result = self.runtime.transition_hamiltonian(operation.runtime.program, lanes)
        record = result.record
        if not record.successful:
            raise ValueError(f"The path Hamiltonian transition failed: {record.reason}")
        target = operation.with_lineage(result.state.hamiltonian_lineage)
        if result.runtime.prepared_id != target.runtime.prepared_id:
            raise ValueError(
                "The transitioned runtime differs from the target operation."
            )
        return (
            target,
            result.state,
            {int(lane): record.transition_id for lane in np.flatnonzero(occupied)},
        )

    def lane_template(
        self, lane_count: int, packet: AtomisticTaskPacket | None, /
    ) -> Any:
        return _concrete_template(self, lane_count, packet)

    def lane_random_words(self, lanes: Any, slot: int, /) -> None:
        return None

    def capacity_levels(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        bundle = self.runtime.program.neighborhood_bundle
        return bundle.source_levels, bundle.role_levels

    def with_capacity_levels(
        self, levels: tuple[tuple[int, ...], tuple[int, ...]], /
    ) -> AtomisticReactionPathStageOperation:
        from ..chemistry.reaction._path import PreparedAtomisticNudgedElasticBand

        bundle = _rebuilt_bundle(self.runtime.program.neighborhood_bundle, levels)
        if bundle is self.runtime.program.neighborhood_bundle:
            return self
        return self._replace(
            PreparedAtomisticNudgedElasticBand(
                self.runtime.plan, self.runtime.program.with_neighborhood_bundle(bundle)
            )
        )


# -- labeling -------------------------------------------------------------------------


@final
class AtomisticLabelLanes(StrictModule):
    """Label inputs, resources and accepted labels of a bank."""

    unwrapped: Array
    positions: Array
    cell_vectors: Array
    images: Array
    resources: AtomisticNeighborhoodBundleState
    energy: Array
    forces: Array
    stress: Array
    done: Array
    shared: Array


def _label_lane(
    operation: AtomisticLabelStageOperation, unwrapped: Array, cell_vectors: Array
) -> tuple[Array, Array, AtomisticNeighborhoodBundleState]:
    program = operation.program
    bundle = program.neighborhood_bundle
    geometry, wrapped, images = trial_geometry(
        program, bundle, unwrapped, cell_vectors if operation.periodic else None
    )
    resources = bundle.initialize(geometry)
    counts = (
        jnp.zeros(unwrapped.shape, dtype=jnp.int32)
        if images is None
        else jnp.asarray(images, dtype=jnp.int32)
    )
    return wrapped, counts, resources


_LABEL_LANE = eqx.filter_jit(_label_lane)


@final
class AtomisticLabelStageOperation(AbstractAtomisticStageOperation):
    """Native E/F/S labels of a prepared Hamiltonian in one batched evaluation.

    Every active lane without a label is evaluated by one vmapped
    ``program.evaluate`` per round.  With ``reuse`` (and the stage entry policy
    ``SHARE``) a packet whose evaluation key serves this operation's key is
    labeled from that evaluation without evaluating again; the reuse is
    recorded as ``SHARED_EVALUATION``.  A packet cell is the runtime cell of
    the evaluation.  Capacity rebuckets the task to grown declared capacities.

    PyTree: the teacher's scientific identity, numeric revisions and the
    binding are dynamic `IdentityWords`; ``stress``, ``reuse`` and
    ``periodic`` are static selectors.
    """

    program: AbstractPreparedAtomisticHamiltonian
    hamiltonian_identity: IdentityWords
    revision_identities: IdentityWords
    operation_identity: IdentityWords
    stress: bool = eqx.field(static=True)
    reuse: bool = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)

    def __init__(
        self,
        program: AbstractPreparedAtomisticHamiltonian,
        /,
        *,
        stress: bool = False,
        reuse: bool = True,
        _revision_ids: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(program, AbstractPreparedAtomisticHamiltonian):
            raise TypeError("program must be a prepared atomistic Hamiltonian.")
        if not isinstance(stress, bool) or not isinstance(reuse, bool):
            raise TypeError("stress and reuse must be bools.")
        if stress and program.system.cell is None:
            raise ValueError("Stress labels require a periodic system.")
        revisions = (
            (atomistic_hamiltonian_revision(program).revision_id,)
            if _revision_ids is None
            else tuple(_revision_ids)
        )
        self.program = program
        self.stress = stress
        self.reuse = reuse
        self.periodic = program.system.cell is not None
        self.hamiltonian_identity = program.scientific_identity
        self.revision_identities = _identity_rows(revisions)
        self.operation_identity = IdentityWords.of(
            {
                "kind": "atomistic-label-stage-operation",
                "program": program.prepared_id,
                "stress": stress,
                "reuse": reuse,
                "revisions": list(revisions),
            }
        )

    @property
    def hamiltonian_id(self) -> str:
        return self.hamiltonian_identity.fingerprint()

    @property
    def kind(self) -> AtomisticStageOperationKind:
        return AtomisticStageOperationKind.LABEL

    @property
    def capacity_mode(self) -> AtomisticCapacityMode:
        return AtomisticCapacityMode.REBUCKET

    @property
    def supports_rebind(self) -> bool:
        return True

    @property
    def supports_shared_entry(self) -> bool:
        return self.reuse

    @property
    def preserves_configuration(self) -> bool:
        return True

    def _replace(
        self, program: AbstractPreparedAtomisticHamiltonian, /
    ) -> AtomisticLabelStageOperation:
        return AtomisticLabelStageOperation(
            program, stress=self.stress, reuse=self.reuse, _revision_ids=self.revision_ids
        )

    def signature(self, lanes: int, /) -> AtomisticStageSignature:
        system = self.program.system
        labels = ["energy", "forces", "stress"] if self.stress else ["energy", "forces"]
        return AtomisticStageSignature(
            self.kind,
            lanes,
            topology=system.layout_id,
            resources=self.program.neighborhood_bundle.layout_id,
            model=self.program.layout_id,
            derivatives="energy,forces,stress" if self.stress else "energy,forces",
            precision=system.plan.coordinate_dtype,
            method_state=canonical_fingerprint({"labels": labels}),
        )

    def evaluation_key(self) -> AtomisticEvaluationKey:
        return AtomisticEvaluationKey(
            self.hamiltonian_id,
            numeric_revision_ids=self.revision_ids,
            conditions_id=_conditions_id(np.zeros((0,)), self.program.control_ids),
            quantities=("energy", "forces", "stress")
            if self.stress
            else ("energy", "forces"),
        )

    def admit(
        self,
        lanes: AtomisticLabelLanes | None,
        lane_count: int,
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        keys: Array,
        ordinals: np.ndarray,
        /,
        *,
        sources: tuple[AbstractAtomisticStageOperation, ...] = (),
    ) -> AtomisticLaneAdmission:
        del keys, ordinals, sources
        count = len(packets)
        admission = _no_admission(count, lanes)
        arrays = (
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
        )
        system = self.program.system
        dtype = jnp.dtype(system.plan.coordinate_dtype)
        cell = _system_cell(system)
        key = self.evaluation_key()
        bank = lanes
        for index, packet in enumerate(packets):
            positions = _packet_positions(packet, system)
            if positions is None:
                _refuse(arrays, index, AtomisticCampaignReason.INVALID_PACKET)
                continue
            if cell is None and packet.cell_vectors is not None:
                _refuse(arrays, index, AtomisticCampaignReason.CELL_MISMATCH)
                continue
            vectors = (
                np.zeros((3, 3))
                if cell is None
                else (cell if packet.cell_vectors is None else packet.cell_vectors)
            )
            if vectors.shape != (3, 3) or not np.all(np.isfinite(vectors)):
                _refuse(arrays, index, AtomisticCampaignReason.CELL_MISMATCH)
                continue
            unwrapped = jnp.asarray(positions, dtype=dtype)
            vectors_ = jnp.asarray(vectors, dtype=dtype)
            wrapped, images, resources = _LABEL_LANE(self, unwrapped, vectors_)
            evaluation = packet.evaluation
            shared = bool(
                self.reuse and evaluation is not None and key.served_by(evaluation.key)
            )
            nan = jnp.asarray(jnp.nan, dtype=dtype)
            energy = jnp.asarray(evaluation.energy, dtype=dtype) if shared else nan
            forces = (
                jnp.asarray(evaluation.forces, dtype=dtype)
                if shared
                else jnp.full((system.capacity, 3), jnp.nan, dtype=dtype)
            )
            stress = (
                jnp.asarray(evaluation.stress, dtype=dtype)
                if shared and self.stress
                else jnp.full((3, 3), jnp.nan, dtype=dtype)
            )
            single = AtomisticLabelLanes(
                unwrapped,
                wrapped,
                vectors_,
                images,
                resources,
                energy,
                forces,
                stress,
                jnp.asarray(shared),
                jnp.asarray(shared),
            )
            admission.shared[index] = shared
            if bank is None:
                bank = _lane_broadcast(single, lane_count)
            bank = _lane_insert(bank, int(slots[index]), single)
        return AtomisticLaneAdmission(
            bank,
            admission.refused,
            admission.refusal,
            admission.reasons,
            admission.shared,
        )

    def advance(
        self,
        lanes: AtomisticLabelLanes,
        active: Array,
        budget: Array,
        consumed: Array,
        /,
    ) -> AtomisticLaneAdvance:
        del budget
        program = self.program

        def evaluate(
            unwrapped: Array,
            positions: Array,
            vectors: Array,
            images: Array,
            resources: AtomisticNeighborhoodBundleState,
        ) -> tuple[Array, Array, Array, Array]:
            kwargs: dict[str, Any] = {"unwrapped_positions": unwrapped}
            if self.periodic:
                kwargs.update(
                    cell=program.system.cell, cell_vectors=vectors, image_counts=images
                )
            evaluation = program.evaluate(
                positions, resources, compute_stress=self.stress, **kwargs
            )
            stress = (
                evaluation.stress
                if self.stress and evaluation.stress is not None
                else jnp.full((3, 3), jnp.nan, dtype=evaluation.energy.dtype)
            )
            finite = jnp.isfinite(evaluation.energy) & jnp.all(
                jnp.isfinite(evaluation.forces)
            )
            if self.stress:
                finite = finite & jnp.all(jnp.isfinite(stress))
            return (
                evaluation.energy,
                evaluation.forces,
                stress,
                evaluation.successful & finite,
            )

        energy, forces, stress, ok = jax.vmap(evaluate)(
            lanes.unwrapped,
            lanes.positions,
            lanes.cell_vectors,
            lanes.images,
            lanes.resources,
        )
        need = active & ~lanes.done
        capacity = need & lanes.resources.capacity_only[:, 0]
        success = need & ok & ~capacity
        failed = need & ~ok & ~capacity
        outcome = jnp.where(
            active & lanes.done,
            int(AtomisticStageOutcome.COMPLETED),
            jnp.where(
                success,
                int(AtomisticStageOutcome.COMPLETED),
                jnp.where(
                    capacity,
                    int(AtomisticStageOutcome.CAPACITY_PENDING),
                    jnp.where(
                        failed,
                        int(AtomisticStageOutcome.FAILED),
                        int(AtomisticStageOutcome.ACTIVE),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        updated = AtomisticLabelLanes(
            lanes.unwrapped,
            lanes.positions,
            lanes.cell_vectors,
            lanes.images,
            lanes.resources,
            jnp.where(success, energy, lanes.energy),
            _lane_where(success, forces, lanes.forces),
            _lane_where(success, stress, lanes.stress),
            lanes.done | success,
            lanes.shared,
        )
        reasons = jnp.where(
            active & lanes.shared, int(AtomisticCampaignReason.SHARED_EVALUATION), 0
        ).astype(jnp.int32)
        return AtomisticLaneAdvance(
            updated, outcome, consumed + (success | failed).astype(jnp.int32), reasons
        )

    def export(
        self, lanes: AtomisticLabelLanes, slot: int, /
    ) -> tuple[AtomisticTaskPacket, ...]:
        lane = jax.device_get(_lane_slice(lanes, int(slot)))
        evaluation = None
        if (
            bool(lane.done)
            and bool(np.isfinite(lane.energy))
            and bool(np.all(np.isfinite(lane.forces)))
        ):
            evaluation = AtomisticEvaluationRecord(
                lane.energy,
                lane.forces,
                np.asarray(lane.stress) if self.stress else None,
                self.evaluation_key(),
            )
        system = self.program.system
        ports = _system_ports(system)
        return (
            AtomisticTaskPacket(
                positions=np.asarray(lane.unwrapped),
                cell_vectors=np.asarray(lane.cell_vectors) if self.periodic else None,
                species=ports["species"],
                atom_type_ids=ports["atom_type_ids"],
                atom_mask=ports["atom_mask"],
                evaluation=evaluation,
            ),
        )

    def grow(
        self, lanes: AtomisticLabelLanes, slots: np.ndarray, /
    ) -> AtomisticCapacityGrowth:
        transfers: list[AtomisticCapacityTransfer] = []
        bundle = self.program.neighborhood_bundle
        for slot in (int(value) for value in slots):
            resources = _lane_slice(lanes.resources, slot)
            kinds = _capacity_kinds(resources)
            try:
                grown = bundle.grow_capacities(resources)
            except (RuntimeError, ValueError):
                transfers.append(
                    AtomisticCapacityTransfer(
                        slot,
                        None,
                        None,
                        int(AtomisticStageOutcome.CAPACITY_REFUSED),
                        kinds | int(AtomisticCampaignReason.LADDER_EXHAUSTED),
                    )
                )
                continue
            lane = jax.device_get(_lane_slice(lanes, slot))
            ports = _system_ports(self.program.system)
            packet = AtomisticTaskPacket(
                positions=np.asarray(lane.unwrapped),
                cell_vectors=np.asarray(lane.cell_vectors) if self.periodic else None,
                species=ports["species"],
                atom_type_ids=ports["atom_type_ids"],
                atom_mask=ports["atom_mask"],
            )
            transfers.append(
                AtomisticCapacityTransfer(
                    slot,
                    self._replace(self.program.with_neighborhood_bundle(grown)),
                    packet,
                    int(AtomisticStageOutcome.ACTIVE),
                    kinds,
                )
            )
        return AtomisticCapacityGrowth(self, lanes, tuple(transfers))

    def rebind(
        self,
        lanes: AtomisticLabelLanes,
        occupied: np.ndarray,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> tuple[AbstractAtomisticStageOperation, Any, dict[int, str]]:
        """Teacher transition of unlabeled lanes (the campaign refuses labeled ones).

        A label lane holds only its configuration: resources are formed
        afresh under the target teacher's bundle and the lane is evaluated by
        the target teacher; an evaluation reused from the incoming packet under
        the previous teacher is discarded rather than relabeled.
        """
        if not isinstance(operation, AtomisticLabelStageOperation):
            raise TypeError("A label bank transitions only to a label operation.")
        if (
            operation.program.system.prepared_id != self.program.system.prepared_id
            or operation.stress != self.stress
            or operation.reuse != self.reuse
        ):
            raise ValueError(
                "A teacher transition keeps the system and requested labels."
            )
        slots = [int(value) for value in np.flatnonzero(occupied)]
        if bool(np.any(np.asarray(lanes.done)[slots])):
            raise ValueError("Labeled lanes belong to the previous teacher.")
        lineage_id = canonical_fingerprint(
            {
                "kind": "atomistic-label-teacher-transition",
                "source": self.binding_id,
                "target": operation.binding_id,
                "source_hamiltonian": self.hamiltonian_id,
                "target_hamiltonian": operation.hamiltonian_id,
            }
        )
        if not slots:
            return operation, None, {}
        bank: AtomisticLabelLanes | None = None
        for slot in slots:
            lane = _lane_slice(lanes, slot)
            wrapped, images, resources = _LABEL_LANE(
                operation, lane.unwrapped, lane.cell_vectors
            )
            single = AtomisticLabelLanes(
                lane.unwrapped,
                wrapped,
                lane.cell_vectors,
                images,
                resources,
                jnp.full_like(lane.energy, jnp.nan),
                jnp.full_like(lane.forces, jnp.nan),
                jnp.full_like(lane.stress, jnp.nan),
                jnp.asarray(False),
                jnp.asarray(False),
            )
            if bank is None:
                bank = _lane_broadcast(single, int(occupied.shape[0]))
            bank = _lane_insert(bank, slot, single)
        return operation, bank, {slot: lineage_id for slot in slots}

    def lane_template(
        self, lane_count: int, packet: AtomisticTaskPacket | None, /
    ) -> Any:
        del packet
        system = self.program.system
        dtype = jnp.dtype(system.plan.coordinate_dtype)

        def single(unwrapped: Array, vectors: Array) -> AtomisticLabelLanes:
            wrapped, images, resources = _label_lane(self, unwrapped, vectors)
            return AtomisticLabelLanes(
                unwrapped,
                wrapped,
                vectors,
                images,
                resources,
                jnp.zeros((), dtype=dtype),
                jnp.zeros((system.capacity, 3), dtype=dtype),
                jnp.zeros((3, 3), dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(False),
            )

        lane = eqx.filter_eval_shape(
            single,
            jax.ShapeDtypeStruct((system.capacity, 3), dtype),
            jax.ShapeDtypeStruct((3, 3), dtype),
        )
        return _abstract_lanes(lane, lane_count)

    def lane_random_words(self, lanes: AtomisticLabelLanes, slot: int, /) -> None:
        return None

    def capacity_levels(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        bundle = self.program.neighborhood_bundle
        return bundle.source_levels, bundle.role_levels

    def with_capacity_levels(
        self, levels: tuple[tuple[int, ...], tuple[int, ...]], /
    ) -> AtomisticLabelStageOperation:
        bundle = _rebuilt_bundle(self.program.neighborhood_bundle, levels)
        if bundle is self.program.neighborhood_bundle:
            return self
        return self._replace(self.program.with_neighborhood_bundle(bundle))


# -- generation -----------------------------------------------------------------------


@final
class AtomisticGenerationStageOperation(AbstractAtomisticStageOperation):
    """Native generation lanes; accepted proposals spawn tasks of system families.

    One task is one stable proposal ordinal of a prepared native sampler
    (`AbstractPreparedAtomisticSampler`).  Proposals are screened exactly as
    the generation stream screens them: sampler status, support admission,
    then the declared filters in order.  An accepted proposal becomes a child
    task (``atomistic_proposal_id``) of the family whose system has exactly
    the proposal's active atomic-number sequence, in the support's length
    unit; positions are placed in that system's active slots.  An unroutable
    proposal records ``UNROUTABLE``; a rejected proposal is ``FAILED`` with
    its status bits.  Sampler budgets and refusals are the sampler's.

    PyTree: the family routing tables are dynamic ``int32`` arrays per family
    (``route_numbers``: active atomic numbers, ``route_slots``: their system
    slots; their lengths are the families' active counts, i.e. layout), with
    the declared family labels and capacities static; ``routes`` rebuilds the
    exact host table.  Numeric revisions and the binding are dynamic
    `IdentityWords`.  ``campaign_id`` is the declared campaign label (never a
    digest, so never hashed into words) and stays static with
    ``steps_per_round``.
    """

    sampler: AbstractPreparedAtomisticSampler
    filters: tuple[AbstractProposalFilter, ...]
    route_numbers: tuple[Array, ...]
    route_slots: tuple[Array, ...]
    revision_identities: IdentityWords
    operation_identity: IdentityWords
    campaign_id: str = eqx.field(static=True)
    steps_per_round: int = eqx.field(static=True)
    route_families: tuple[str, ...] = eqx.field(static=True)
    route_capacities: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        sampler: AbstractPreparedAtomisticSampler,
        families: Mapping[str, PreparedAtomisticSystem],
        /,
        *,
        campaign_id: str,
        filters: Sequence[AbstractProposalFilter] = (),
        steps_per_round: int = 1,
    ) -> None:
        if not isinstance(sampler, AbstractPreparedAtomisticSampler):
            raise TypeError("sampler must be an AbstractPreparedAtomisticSampler.")
        screens = tuple(filters)
        if any(not isinstance(item, AbstractProposalFilter) for item in screens):
            raise TypeError("filters must be AbstractProposalFilter instances.")
        if int(steps_per_round) < 1:
            raise ValueError("steps_per_round must be positive.")
        length = sampler.support.length_unit.unit_id
        routes = []
        for family in sorted(families):
            system = families[family]
            if not isinstance(system, PreparedAtomisticSystem):
                raise TypeError("families must map names to PreparedAtomisticSystem.")
            if system.plan.units.scale.length_unit.unit_id != length:
                raise ValueError(
                    f"Family {family} measures length in another unit than the "
                    "generation support."
                )
            active = np.flatnonzero(np.asarray(system.active_mask, dtype=np.bool_))
            numbers = np.asarray(system.plan.atomic_numbers, dtype=np.int64)[active]
            routes.append(
                (
                    _identifier(family, "family"),
                    tuple(int(value) for value in numbers),
                    tuple(int(value) for value in active),
                    int(system.capacity),
                )
            )
        campaign = _identifier(campaign_id, "campaign_id")
        revisions = (sampler.numeric_revision().revision_id,)
        self.sampler = sampler
        self.filters = screens
        self.campaign_id = campaign
        self.steps_per_round = int(steps_per_round)
        self.route_families = tuple(route[0] for route in routes)
        self.route_numbers = tuple(
            jnp.asarray(np.asarray(route[1], dtype=np.int32)) for route in routes
        )
        self.route_slots = tuple(
            jnp.asarray(np.asarray(route[2], dtype=np.int32)) for route in routes
        )
        self.route_capacities = tuple(route[3] for route in routes)
        self.revision_identities = _identity_rows(revisions)
        self.operation_identity = IdentityWords.of(
            {
                "kind": "atomistic-generation-stage-operation",
                "sampler": _identifier(sampler.sampler_id, "sampler_id"),
                "support": sampler.support.support_id,
                "facts": {
                    str(k): str(v) for k, v in dict(sampler.signature_facts()).items()
                },
                "filters": [item.filter_id for item in screens],
                "campaign": campaign,
                "routes": [[route[0], list(route[1])] for route in routes],
                "steps": self.steps_per_round,
                "revisions": list(revisions),
            }
        )

    @property
    def routes(self) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...], int], ...]:
        """Host family table: label, active atomic numbers, active slots, capacity."""
        return tuple(
            (
                family,
                tuple(int(value) for value in np.asarray(numbers)),
                tuple(int(value) for value in np.asarray(slots)),
                capacity,
            )
            for family, numbers, slots, capacity in zip(
                self.route_families,
                self.route_numbers,
                self.route_slots,
                self.route_capacities,
                strict=True,
            )
        )

    def _screened(self, block: AtomisticProposalBlock, /) -> Array:
        """The generation owner's screening: status, support admission, filters."""
        status, _, _ = screen_atomistic_proposals(
            block, self.sampler.support, self.filters
        )
        return status

    @property
    def kind(self) -> AtomisticStageOperationKind:
        return AtomisticStageOperationKind.GENERATION

    @property
    def capacity_mode(self) -> AtomisticCapacityMode:
        return AtomisticCapacityMode.REBUCKET

    @property
    def supports_rebind(self) -> bool:
        return True

    @property
    def supports_shared_entry(self) -> bool:
        return False

    @property
    def native_lanes(self) -> int | None:
        """Samplers prepared for a fixed lane count (e.g. crystal packers)."""
        lanes = getattr(self.sampler, "lanes", None)
        return None if lanes is None else int(lanes)

    def signature(self, lanes: int, /) -> AtomisticStageSignature:
        facts = {str(k): str(v) for k, v in dict(self.sampler.signature_facts()).items()}
        return AtomisticStageSignature(
            self.kind,
            lanes,
            topology=canonical_fingerprint({"facts": facts}),
            resources=facts.get("layout", "sampler"),
            model=self.sampler.sampler_id,
            derivatives=facts.get("derivatives", "none"),
            precision=facts.get("precision", "sampler"),
            method_state=canonical_fingerprint(
                {"facts": facts, "steps": self.steps_per_round}
            ),
        )

    def evaluation_key(self) -> None:
        return None

    def admit(
        self,
        lanes: Any,
        lane_count: int,
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        keys: Array,
        ordinals: np.ndarray,
        /,
        *,
        sources: tuple[AbstractAtomisticStageOperation, ...] = (),
    ) -> AtomisticLaneAdmission:
        del sources
        count = len(packets)
        state = self.sampler.vacant(int(lane_count)) if lanes is None else lanes
        mask = np.zeros((lane_count,), dtype=np.bool_)
        source = np.zeros((lane_count,), dtype=np.int32)
        lane_ordinals = np.zeros((lane_count,), dtype=np.int64)
        for index in range(count):
            lane = int(slots[index])
            mask[lane] = True
            source[lane] = index
            lane_ordinals[lane] = int(ordinals[index])
        state = self.sampler.admit(
            state,
            jnp.asarray(mask),
            keys[jnp.asarray(source)],
            jnp.asarray(lane_ordinals),
            None,
        )
        return _no_admission(count, state)

    def advance(
        self, lanes: Any, active: Array, budget: Array, consumed: Array, /
    ) -> AtomisticLaneAdvance:
        del budget
        before = self.sampler.terminal(lanes)
        state = self.sampler.advance(lanes, self.steps_per_round)
        terminal = self.sampler.terminal(state)
        status = self._screened(self.sampler.emit(state))
        outcome = jnp.where(
            active & terminal & (status == 0),
            int(AtomisticStageOutcome.COMPLETED),
            jnp.where(
                active & terminal,
                int(AtomisticStageOutcome.FAILED),
                int(AtomisticStageOutcome.ACTIVE),
            ),
        ).astype(jnp.int32)
        steps = jnp.where(active & ~before, self.steps_per_round, 0).astype(jnp.int32)
        return AtomisticLaneAdvance(
            state,
            outcome,
            consumed + steps,
            jnp.where(active, status, 0).astype(jnp.int32),
        )

    def export(self, lanes: Any, slot: int, /) -> tuple[AtomisticTaskPacket, ...]:
        index = int(slot)
        emitted = self.sampler.emit(lanes)
        status = np.asarray(self._screened(emitted))
        terminal = np.asarray(self.sampler.terminal(lanes))
        block = jax.device_get(emitted)
        if not (bool(terminal[index]) and int(status[index]) == 0):
            return ()
        mask = np.asarray(block.atom_mask[index], dtype=np.bool_)
        numbers = tuple(int(value) for value in np.asarray(block.species[index])[mask])
        coordinates = np.asarray(block.positions[index], dtype=np.float64)[mask]
        ordinal = int(block.ordinals[index])
        child = atomistic_proposal_id(
            self.campaign_id,
            self.sampler.support.support_id,
            self.sampler.sampler_id,
            ordinal,
        )
        cell = (
            None
            if block.cell is None
            else np.asarray(block.cell[index], dtype=np.float64)
        )
        for family, sequence, active, capacity in self.routes:
            if sequence != numbers:
                continue
            positions = np.zeros((capacity, 3), dtype=np.float64)
            positions[np.asarray(active, dtype=np.int64)] = coordinates
            atom_mask = np.zeros((capacity,), dtype=np.bool_)
            atom_mask[np.asarray(active, dtype=np.int64)] = True
            species = np.zeros((capacity,), dtype=np.int32)
            species[np.asarray(active, dtype=np.int64)] = numbers
            return (
                AtomisticTaskPacket(
                    positions=positions,
                    cell_vectors=cell,
                    species=species,
                    atom_mask=atom_mask,
                    child_id=child,
                    family=family,
                ),
            )
        return (
            AtomisticTaskPacket(
                positions=coordinates,
                cell_vectors=cell,
                species=np.asarray(numbers, dtype=np.int32),
                child_id=child,
            ),
        )

    def grow(self, lanes: Any, slots: np.ndarray, /) -> AtomisticCapacityGrowth:
        return AtomisticCapacityGrowth(
            self,
            lanes,
            tuple(
                AtomisticCapacityTransfer(
                    int(slot),
                    None,
                    None,
                    int(AtomisticStageOutcome.CAPACITY_REFUSED),
                    int(AtomisticCampaignReason.LADDER_EXHAUSTED),
                )
                for slot in slots
            ),
        )

    def rebind(
        self,
        lanes: Any,
        occupied: np.ndarray,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> tuple[AbstractAtomisticStageOperation, Any, dict[int, str]]:
        """Sampler transition of in-flight attempts (the campaign refuses finished ones).

        Each running attempt is restarted under the target sampler at its own
        stable ordinal and lane key: the partial chain of the previous sampler
        is discarded explicitly (recorded in the lineage), never continued
        under another model.  The support and family routing are unchanged.
        """
        if not isinstance(operation, AtomisticGenerationStageOperation):
            raise TypeError(
                "A generation bank transitions only to a generation operation."
            )
        if (
            operation.sampler.support.support_id != self.sampler.support.support_id
            or operation.routes != self.routes
            or operation.campaign_id != self.campaign_id
        ):
            raise ValueError("A sampler transition keeps the support and family routing.")
        slots = [int(value) for value in np.flatnonzero(occupied)]
        if bool(np.any(np.asarray(self.sampler.terminal(lanes))[slots])):
            raise ValueError("Finished proposals belong to the previous sampler.")
        lineage_id = canonical_fingerprint(
            {
                "kind": "atomistic-generation-sampler-transition",
                "source": self.binding_id,
                "target": operation.binding_id,
                "policy": "restart-attempt-at-ordinal",
            }
        )
        if not slots:
            return operation, None, {}
        mask = np.zeros((int(occupied.shape[0]),), dtype=np.bool_)
        mask[slots] = True
        state = operation.sampler.admit(
            operation.sampler.vacant(int(occupied.shape[0])),
            jnp.asarray(mask),
            lanes.keys,
            lanes.ordinals,
            None,
        )
        return operation, state, {slot: lineage_id for slot in slots}

    def lane_template(
        self, lane_count: int, packet: AtomisticTaskPacket | None, /
    ) -> Any:
        del packet
        return self.sampler.vacant(int(lane_count))

    def lane_random_words(self, lanes: Any, slot: int, /) -> None:
        return None

    def capacity_levels(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (), ()

    def with_capacity_levels(
        self, levels: tuple[tuple[int, ...], tuple[int, ...]], /
    ) -> AtomisticGenerationStageOperation:
        if tuple(levels[0]) or tuple(levels[1]):
            raise ValueError("A generation operation has no capacity ladder.")
        return self


# -- campaign plan --------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class AtomisticStageRoute:
    """Where an outcome goes: the next stage (or terminal) and whether to publish."""

    next_stage: str | None = None
    publish: bool = True

    def __post_init__(self) -> None:
        if self.next_stage is not None:
            _identifier(self.next_stage, "next_stage")
        if not isinstance(self.publish, bool):
            raise TypeError("publish must be a bool.")


@final
@dataclass(frozen=True, slots=True)
class AtomisticStageRouting:
    """Outcome routing of one stage; ``None`` selects the declared default.

    Defaults: ``CONVERGED``/``COMPLETED`` continue to the next stage in plan
    order without publishing (publishing when the stage is last);
    ``EXHAUSTED``, ``FAILED`` and ``CAPACITY_REFUSED`` terminate and publish.
    """

    converged: AtomisticStageRoute | None = None
    completed: AtomisticStageRoute | None = None
    exhausted: AtomisticStageRoute | None = None
    failed: AtomisticStageRoute | None = None
    capacity_refused: AtomisticStageRoute | None = None


@final
@dataclass(frozen=True, eq=False)
class AtomisticStagePlan:
    """Scientific operation per system family, budget, routing and lane policy.

    ``operations`` maps a system family to its stage operation; families with
    equal operation bindings in several stages share one bank.  ``budget`` is
    the per-visit budget of fixed-length operations (accepted dynamics
    steps, sampler steps); relaxation and path operations keep their own
    convergence budget.  ``lane_capacity`` bounds resident lanes per bank
    (natively group-batched operations require exactly their group count);
    ``queue_capacity`` bounds tasks waiting to enter the stage (upstream
    lanes park when it is full).  ``entry`` permits reuse of an incoming
    compatible evaluation; ``revision`` governs in-flight lanes when an
    operation is rebound to another numeric revision; ``continuation``
    decides whether a dynamics stage continues an incoming walker's dynamics
    (``CONTINUE``) or explicitly begins a new run from its phase point
    (``RESTART``, for example a change of integrator or ensemble).
    """

    name: str
    operations: Mapping[str, AbstractAtomisticStageOperation]
    budget: int | None = None
    routing: AtomisticStageRouting = AtomisticStageRouting()
    lane_capacity: int = 8
    queue_capacity: int = 64
    entry: AtomisticEntryPolicy = AtomisticEntryPolicy.EVALUATE
    revision: AtomisticRevisionPolicy = AtomisticRevisionPolicy.DRAIN
    continuation: AtomisticContinuationPolicy = AtomisticContinuationPolicy.CONTINUE

    def __post_init__(self) -> None:
        _identifier(self.name, "stage name")
        if not isinstance(self.operations, Mapping) or not self.operations:
            raise ValueError("A stage needs at least one family operation.")
        for family, operation in self.operations.items():
            _identifier(family, "family")
            if not isinstance(operation, AbstractAtomisticStageOperation):
                raise TypeError(
                    "Stage operations must be AbstractAtomisticStageOperation."
                )
        if self.budget is not None and (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or self.budget < 0
        ):
            raise ValueError("budget must be a non-negative int or None.")
        for name in ("lane_capacity", "queue_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int.")
        if not isinstance(self.routing, AtomisticStageRouting):
            raise TypeError("routing must be AtomisticStageRouting.")
        if not isinstance(self.entry, AtomisticEntryPolicy):
            raise TypeError("entry must be an AtomisticEntryPolicy.")
        if not isinstance(self.revision, AtomisticRevisionPolicy):
            raise TypeError("revision must be an AtomisticRevisionPolicy.")
        if not isinstance(self.continuation, AtomisticContinuationPolicy):
            raise TypeError("continuation must be an AtomisticContinuationPolicy.")
        if self.continuation is AtomisticContinuationPolicy.RESTART and not any(
            operation.kind is AtomisticStageOperationKind.DYNAMICS
            for operation in self.operations.values()
        ):
            raise ValueError("RESTART declares a new dynamics run; the stage has none.")


@final
@dataclass(frozen=True, eq=False)
class AtomisticCampaignTask:
    """Stable task identity, system family and initial packet.

    The initial packet carries no method continuation except a walker's
    `AtomisticDynamicsContinuation` (a dynamics run continued across
    campaigns); the entry stage's operation continues it or refuses it.
    """

    task_id: str
    family: str
    packet: AtomisticTaskPacket
    material_id: str
    run_id: str = "run-0"
    replica_id: str = "replica-0"
    group_id: str | None = None
    entry_stage: str | None = None
    ordinal: int = 0

    def __post_init__(self) -> None:
        for name in ("task_id", "family", "material_id", "run_id", "replica_id"):
            _identifier(getattr(self, name), name)
        if self.group_id is not None:
            _identifier(self.group_id, "group_id")
        if self.entry_stage is not None:
            _identifier(self.entry_stage, "entry_stage")
        if not isinstance(self.packet, AtomisticTaskPacket):
            raise TypeError("packet must be an AtomisticTaskPacket.")
        if self.packet.continuation is not None and not isinstance(
            self.packet.continuation, AtomisticDynamicsContinuation
        ):
            raise ValueError(
                "An initial packet carries no method continuation other than a "
                "dynamics continuation."
            )
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or not 0 <= self.ordinal < _UINT32_LIMIT
        ):
            raise ValueError("ordinal must be a uint32-range int.")

    def record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "material_id": self.material_id,
            "run_id": self.run_id,
            "replica_id": self.replica_id,
            "group_id": self.group_id,
            "entry_stage": self.entry_stage,
            "ordinal": self.ordinal,
            "packet": _packet_digest(self.packet),
        }


@final
@dataclass(frozen=True, eq=False)
class AtomisticCampaignPlan:
    """Stages, tasks and the bounded pending-publication bank of one campaign."""

    campaign_id: str
    stages: tuple[AtomisticStagePlan, ...]
    tasks: tuple[AtomisticCampaignTask, ...]
    max_pending_publications: int = 16

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("stages must be a non-empty tuple.")
        if not isinstance(self.tasks, tuple):
            raise TypeError("tasks must be a tuple.")
        value = self.max_pending_publications
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_pending_publications must be a positive int.")

    def prepare(self) -> PreparedAtomisticCampaign:
        return PreparedAtomisticCampaign(self)


# -- outputs, receipts, sinks -----------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class AtomisticOutputReceipt:
    """Durable identity of one published campaign output."""

    semantic_key: str
    sink_id: str
    payload_id: str
    location: str


@final
@dataclass(frozen=True, slots=True)
class AtomisticCommitReceipt:
    """Durable identity of one committed campaign continuation envelope."""

    round_index: int
    payload_id: str
    location: str


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticCampaignOutput:
    """One stage output of one task visit, idempotent by ``semantic_key``.

    ``semantic_key`` derives only from campaign, task, stage and stage entry;
    ``payload_id`` from the complete published content.  ``ports`` keys are
    ``positions``, ``cell_vectors``, ``momenta``, ``species``,
    ``atom_type_ids``, ``atom_mask``, ``energy``, ``forces``, ``stress``,
    ``path``, ``path_energies`` and ``time`` where present.
    """

    semantic_key: str
    payload_id: str
    campaign_id: str
    task_id: str
    material_id: str
    run_id: str
    replica_id: str
    group_id: str | None
    parent_id: str | None
    stage: str
    stage_entry: int
    outcome: AtomisticStageOutcome
    ports: Mapping[str, np.ndarray]
    evaluation_key: AtomisticEvaluationKey | None
    numeric_revision_ids: tuple[str, ...]
    rng_words: np.ndarray
    counters: Mapping[str, int]
    reasons: int
    campaign_reasons: int
    hamiltonian_lineage: tuple[str, ...] = ()

    @property
    def evaluation_key_id(self) -> str | None:
        return None if self.evaluation_key is None else self.evaluation_key.key_id

    def manifest(self) -> dict[str, Any]:
        return {
            "format": _OUTPUT_FORMAT,
            "semantic_key": self.semantic_key,
            "payload_id": self.payload_id,
            **_output_metadata(
                self.campaign_id,
                self.task_id,
                self.material_id,
                self.run_id,
                self.replica_id,
                self.group_id,
                self.parent_id,
                self.stage,
                self.stage_entry,
                self.outcome,
                self.evaluation_key,
                self.numeric_revision_ids,
                self.rng_words,
                self.counters,
                self.reasons,
                self.campaign_reasons,
                self.hamiltonian_lineage,
            ),
            "ports": sorted(self.ports),
        }


def _output_metadata(
    campaign_id: str,
    task_id: str,
    material_id: str,
    run_id: str,
    replica_id: str,
    group_id: str | None,
    parent_id: str | None,
    stage: str,
    stage_entry: int,
    outcome: AtomisticStageOutcome,
    evaluation_key: AtomisticEvaluationKey | None,
    numeric_revision_ids: tuple[str, ...],
    rng_words: np.ndarray,
    counters: Mapping[str, int],
    reasons: int,
    campaign_reasons: int,
    hamiltonian_lineage: tuple[str, ...],
    /,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "task_id": task_id,
        "material_id": material_id,
        "run_id": run_id,
        "replica_id": replica_id,
        "group_id": group_id,
        "parent_id": parent_id,
        "stage": stage,
        "stage_entry": int(stage_entry),
        "outcome": AtomisticStageOutcome(outcome).name,
        "evaluation_key": None if evaluation_key is None else evaluation_key.to_record(),
        "numeric_revision_ids": list(numeric_revision_ids),
        "rng_words": [int(value) for value in np.asarray(rng_words).reshape(-1)],
        "counters": {str(name): int(value) for name, value in sorted(counters.items())},
        "reasons": int(reasons),
        "campaign_reasons": int(campaign_reasons),
        "hamiltonian_lineage": list(hamiltonian_lineage),
    }


def _semantic_output_key(
    campaign_id: str, task_id: str, stage: str, entry: int, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-campaign-output",
            "campaign": campaign_id,
            "task": task_id,
            "stage": stage,
            "entry": int(entry),
        }
    )


def _make_output(
    campaign_id: str,
    record: AtomisticTaskRecord,
    stage: str,
    entry: int,
    outcome: int,
    packet: AtomisticTaskPacket,
    numeric_revision_ids: tuple[str, ...],
    rng_words: np.ndarray,
    counters: Mapping[str, int],
    reasons: int,
    campaign_reasons: int,
    /,
) -> AtomisticCampaignOutput:
    return _task_output(
        campaign_id,
        record.task,
        record.parent,
        record.lineage,
        stage,
        entry,
        outcome,
        packet,
        numeric_revision_ids,
        rng_words,
        counters,
        reasons,
        campaign_reasons,
    )


def _task_output(
    campaign_id: str,
    task: AtomisticCampaignTask,
    parent: str | None,
    lineage: tuple[str, ...],
    stage: str,
    entry: int,
    outcome: int,
    packet: AtomisticTaskPacket,
    numeric_revision_ids: tuple[str, ...],
    rng_words: np.ndarray,
    counters: Mapping[str, int],
    reasons: int,
    campaign_reasons: int,
    /,
) -> AtomisticCampaignOutput:
    """One task's stage output (also the stage×domain transport sink's)."""
    key = _semantic_output_key(campaign_id, task.task_id, stage, entry)
    ports = packet.ports()
    evaluation_key = None if packet.evaluation is None else packet.evaluation.key
    words = np.asarray(rng_words, dtype=np.uint32).reshape(-1)
    metadata = _output_metadata(
        campaign_id,
        task.task_id,
        task.material_id,
        task.run_id,
        task.replica_id,
        task.group_id,
        parent,
        stage,
        entry,
        AtomisticStageOutcome(outcome),
        evaluation_key,
        numeric_revision_ids,
        words,
        counters,
        reasons,
        campaign_reasons,
        lineage,
    )
    payload_id = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-output-payload",
            "semantic_key": key,
            "metadata": metadata,
            "ports": array_tree_fingerprint(ports),
        }
    )
    return AtomisticCampaignOutput(
        semantic_key=key,
        payload_id=payload_id,
        campaign_id=campaign_id,
        task_id=task.task_id,
        material_id=task.material_id,
        run_id=task.run_id,
        replica_id=task.replica_id,
        group_id=task.group_id,
        parent_id=parent,
        stage=stage,
        stage_entry=int(entry),
        outcome=AtomisticStageOutcome(outcome),
        ports=ports,
        evaluation_key=evaluation_key,
        numeric_revision_ids=tuple(numeric_revision_ids),
        rng_words=words,
        counters=dict(counters),
        reasons=int(reasons),
        campaign_reasons=int(campaign_reasons),
        hamiltonian_lineage=lineage,
    )


def _output_from_manifest(
    manifest: Mapping[str, Any], ports: Mapping[str, np.ndarray], /
) -> AtomisticCampaignOutput:
    evaluation = manifest["evaluation_key"]
    output = AtomisticCampaignOutput(
        semantic_key=manifest["semantic_key"],
        payload_id=manifest["payload_id"],
        campaign_id=manifest["campaign_id"],
        task_id=manifest["task_id"],
        material_id=manifest["material_id"],
        run_id=manifest["run_id"],
        replica_id=manifest["replica_id"],
        group_id=manifest["group_id"],
        parent_id=manifest["parent_id"],
        stage=manifest["stage"],
        stage_entry=int(manifest["stage_entry"]),
        outcome=AtomisticStageOutcome[manifest["outcome"]],
        ports={name: np.asarray(ports[name]) for name in manifest["ports"]},
        evaluation_key=None
        if evaluation is None
        else AtomisticEvaluationKey.from_record(evaluation),
        numeric_revision_ids=tuple(manifest["numeric_revision_ids"]),
        rng_words=np.asarray(manifest["rng_words"], dtype=np.uint32),
        counters=dict(manifest["counters"]),
        reasons=int(manifest["reasons"]),
        campaign_reasons=int(manifest["campaign_reasons"]),
        hamiltonian_lineage=tuple(manifest["hamiltonian_lineage"]),
    )
    expected = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-output-payload",
            "semantic_key": output.semantic_key,
            "metadata": {
                key: value
                for key, value in output.manifest().items()
                if key not in ("format", "semantic_key", "payload_id", "ports")
            },
            "ports": array_tree_fingerprint(dict(output.ports)),
        }
    )
    if expected != output.payload_id:
        raise ValueError("Campaign output payload identity is corrupt.")
    return output


class AbstractAtomisticCampaignSink(abc.ABC):
    """Durable, idempotent destination of campaign outputs.

    ``publish`` returns a receipt only after the record is durable and returns
    ``None`` when nothing became durable (downstream backpressure).  Repeated
    publication of the same ``semantic_key`` must return the existing receipt
    and must refuse a different payload.  ``receipt`` looks up a durable
    record so a replay after a crash never duplicates an output.
    """

    sink_id: str

    @abc.abstractmethod
    def publish(
        self, record: AtomisticCampaignOutput, /, *, round_index: int
    ) -> AtomisticOutputReceipt | None:
        raise NotImplementedError

    @abc.abstractmethod
    def receipt(self, semantic_key: str, /) -> AtomisticOutputReceipt | None:
        raise NotImplementedError


@final
class AtomisticDirectorySink(AbstractAtomisticCampaignSink):
    """One atomically published array archive per output, named by semantic key.

    ``accept_per_round`` bounds how many new records the sink makes durable
    per campaign round (explicit downstream throughput, a backpressure
    source); ``None`` accepts every record.
    """

    __slots__ = ("_accepted", "_limit", "root", "sink_id")

    def __init__(
        self,
        root: str | Path,
        /,
        *,
        sink_id: str = "campaign-directory",
        accept_per_round: int | None = None,
    ) -> None:
        if accept_per_round is not None and (
            isinstance(accept_per_round, bool) or int(accept_per_round) < 0
        ):
            raise ValueError("accept_per_round must be a non-negative int or None.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sink_id = _identifier(sink_id, "sink_id")
        self._limit = None if accept_per_round is None else int(accept_per_round)
        self._accepted: dict[int, int] = {}

    def _path(self, semantic_key: str, /) -> Path:
        return self.root / f"{_identifier(semantic_key, 'semantic_key')}.output"

    def receipt(self, semantic_key: str, /) -> AtomisticOutputReceipt | None:
        path = self._path(semantic_key)
        if not path.exists():
            return None
        manifest, _ = read_array_archive(path)
        if (
            manifest.get("format") != _OUTPUT_FORMAT
            or manifest.get("semantic_key") != semantic_key
        ):
            raise ValueError("The published file is not this campaign output.")
        return AtomisticOutputReceipt(
            semantic_key, self.sink_id, str(manifest["payload_id"]), str(path)
        )

    def publish(
        self, record: AtomisticCampaignOutput, /, *, round_index: int
    ) -> AtomisticOutputReceipt | None:
        existing = self.receipt(record.semantic_key)
        if existing is not None:
            if existing.payload_id != record.payload_id:
                raise ValueError(
                    "A different payload is already published under this semantic key."
                )
            return existing
        accepted = self._accepted.get(int(round_index), 0)
        if self._limit is not None and accepted >= self._limit:
            return None
        path = self._path(record.semantic_key)
        write_array_archive(
            path,
            manifest=record.manifest(),
            arrays={f"port/{name}": value for name, value in record.ports.items()},
        )
        self._accepted[int(round_index)] = accepted + 1
        return AtomisticOutputReceipt(
            record.semantic_key, self.sink_id, record.payload_id, str(path)
        )

    def outputs(self) -> dict[str, AtomisticCampaignOutput]:
        """Every durable output, verified, keyed by semantic key."""
        records: dict[str, AtomisticCampaignOutput] = {}
        for path in sorted(self.root.glob("*.output")):
            manifest, arrays = read_array_archive(path)
            ports = {
                name[len("port/") :]: value
                for name, value in arrays.items()
                if name.startswith("port/")
            }
            output = _output_from_manifest(manifest, ports)
            records[output.semantic_key] = output
        return records


@final
class AtomisticCampaignStore:
    """Atomically replaced continuation envelope: the last durable commit.

    The envelope holds every lane bank leaf as one archive member, so its
    admission limits are explicit and larger than the single-state default.
    """

    __slots__ = ("limits", "path")

    def __init__(
        self, path: str | Path, /, *, limits: ArrayArchiveLimits | None = None
    ) -> None:
        self.path = Path(path)
        self.limits = _CAMPAIGN_ARCHIVE_LIMITS if limits is None else limits
        if not isinstance(self.limits, ArrayArchiveLimits):
            raise TypeError("limits must be ArrayArchiveLimits.")

    def write(self, manifest: Mapping[str, Any], arrays: Mapping[str, Any], /) -> str:
        write_array_archive(
            self.path, manifest=manifest, arrays=arrays, limits=self.limits
        )
        return str(self.path)

    def read(self) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        if not self.path.exists():
            return None
        manifest, arrays = read_array_archive(self.path, limits=self.limits)
        return dict(manifest), dict(arrays)


# -- campaign state ----------------------------------------------------------------------


class _Phase(StrEnum):
    REFILLED = "refilled"
    EVALUATED = "evaluated"
    STAGED = "staged"
    PUBLISHED = "published"
    COMMITTED = "committed"


class AtomisticTaskStatus(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    TERMINAL = "terminal"


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticTaskRecord:
    """Continuation record of one task.

    ``entries[s]`` counts admissions into stage ``s`` (the stage RNG counter);
    ``consumed`` is the stage budget consumed in the current visit (carried
    across capacity transfers); ``packet`` is the input of a queued task.
    ``carried`` is the walker's latest dynamics continuation packet held
    outside a dynamics lane: while the task is admitted to a stage that
    preserves its configuration (labeling) and after the task is terminal
    (`PreparedAtomisticCampaign.terminal_packets`).  ``lineage`` holds the
    owner Hamiltonian transitions of the current visit.
    """

    task: AtomisticCampaignTask
    stage: int
    status: AtomisticTaskStatus
    packet: AtomisticTaskPacket | None
    bucket: str | None
    lane: int
    entries: tuple[int, ...]
    consumed: int
    outcome: int
    reasons: int
    campaign_reasons: int
    parent: str | None
    lineage: tuple[str, ...] = ()
    carried: AtomisticTaskPacket | None = None


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticLaneBank:
    """Resident lanes of one operation binding and their host bookkeeping."""

    bucket_id: str
    stage: str
    family: str
    operation: AbstractAtomisticStageOperation
    lane_count: int
    lanes: Any
    occupant: tuple[str | None, ...]
    lane_stage: np.ndarray
    budget: np.ndarray
    consumed: np.ndarray
    outcome: np.ndarray
    reasons: np.ndarray
    campaign_reasons: np.ndarray
    keys: np.ndarray
    held: tuple[AtomisticTaskPacket | None, ...]
    template: AtomisticTaskPacket | None
    retired: bool

    @property
    def occupied(self) -> np.ndarray:
        return np.asarray([value is not None for value in self.occupant], dtype=np.bool_)


def _vacant_bank(
    bucket_id: str,
    stage: str,
    family: str,
    operation: AbstractAtomisticStageOperation,
    lanes: int,
    template: AtomisticTaskPacket | None,
    /,
) -> AtomisticLaneBank:
    zeros = np.zeros((lanes,), dtype=np.int32)
    return AtomisticLaneBank(
        bucket_id=bucket_id,
        stage=stage,
        family=family,
        operation=operation,
        lane_count=lanes,
        lanes=None,
        occupant=(None,) * lanes,
        lane_stage=np.full((lanes,), -1, dtype=np.int32),
        budget=zeros.copy(),
        consumed=zeros.copy(),
        outcome=zeros.copy(),
        reasons=zeros.copy(),
        campaign_reasons=zeros.copy(),
        keys=np.zeros((lanes, 2), dtype=np.uint32),
        held=(None,) * lanes,
        template=template,
        retired=False,
    )


@final
@dataclass(frozen=True, slots=True)
class AtomisticRoundEvidence:
    """Observation of one round (utilization, refill, backlog, sharing, capacity)."""

    round_index: int = 0
    lane_capacity: int = 0
    occupied_lanes: int = 0
    advanced_lanes: int = 0
    admitted: int = 0
    parked: int = 0
    group_waiting: int = 0
    pending: int = 0
    published: int = 0
    backpressured: bool = False
    shared_evaluations: int = 0
    capacity_transfers: int = 0
    capacity_growths: int = 0
    capacity_refusals: int = 0
    terminal_tasks: int = 0
    spawned: int = 0


@final
@dataclass(frozen=True, slots=True, eq=False)
class AtomisticCampaignState:
    """Complete host continuation of a campaign at one transaction phase."""

    plan_id: str
    phase: str
    round_index: int
    root_key: np.ndarray
    tasks: tuple[AtomisticTaskRecord, ...]
    queues: tuple[tuple[str, ...], ...]
    transfers: tuple[tuple[str, tuple[str, ...]], ...]
    banks: tuple[AtomisticLaneBank, ...]
    routes: tuple[tuple[int, str, str], ...]
    pending: tuple[AtomisticCampaignOutput, ...]
    receipts: tuple[AtomisticOutputReceipt, ...]
    commit: AtomisticCommitReceipt | None
    evidence: AtomisticRoundEvidence

    def task(self, task_id: str, /) -> AtomisticTaskRecord:
        for record in self.tasks:
            if record.task.task_id == task_id:
                return record
        raise KeyError(task_id)

    def bank(self, bucket_id: str, /) -> AtomisticLaneBank:
        for bank in self.banks:
            if bank.bucket_id == bucket_id:
                return bank
        raise KeyError(bucket_id)

    @property
    def complete(self) -> bool:
        return (
            all(record.status is AtomisticTaskStatus.TERMINAL for record in self.tasks)
            and not self.pending
        )


def _replace(value: Any, /, **changes: Any) -> Any:
    fields = {name: getattr(value, name) for name in value.__dataclass_fields__}
    fields.update(changes)
    return type(value)(**fields)


def _advance_bank(
    operation: AbstractAtomisticStageOperation,
    lanes: Any,
    active: Array,
    budget: Array,
    consumed: Array,
) -> AtomisticLaneAdvance:
    return operation.advance(lanes, active, budget, consumed)


_ADVANCE = eqx.filter_jit(_advance_bank)


class _Working:
    """Mutable scratch copy of a state inside one phase (never escapes)."""

    def __init__(self, state: AtomisticCampaignState, /) -> None:
        self.state = state
        self.records = {record.task.task_id: record for record in state.tasks}
        self.order = [record.task.task_id for record in state.tasks]
        self.queues = [list(queue) for queue in state.queues]
        self.transfers = {bucket: list(tasks) for bucket, tasks in state.transfers}
        self.banks = {bank.bucket_id: bank for bank in state.banks}
        self.bank_order = [bank.bucket_id for bank in state.banks]
        self.routes = {(stage, family): bucket for stage, family, bucket in state.routes}
        self.pending = list(state.pending)
        self.receipts = list(state.receipts)

    def add_bank(self, bank: AtomisticLaneBank, /) -> None:
        if bank.bucket_id in self.banks:
            raise ValueError("A lane bank already exists for this binding.")
        self.banks[bank.bucket_id] = bank
        self.bank_order.append(bank.bucket_id)

    def rename_bank(self, old: str, bank: AtomisticLaneBank, /) -> None:
        new = bank.bucket_id
        if new != old and new in self.banks:
            raise ValueError("The grown binding collides with another resident bank.")
        del self.banks[old]
        self.banks[new] = bank
        self.bank_order[self.bank_order.index(old)] = new
        self.routes = {
            key: (new if value == old else value) for key, value in self.routes.items()
        }
        if old in self.transfers:
            self.transfers[new] = self.transfers.pop(old)
        for task_id, record in list(self.records.items()):
            if record.bucket == old:
                self.records[task_id] = _replace(record, bucket=new)

    def freeze(self, **changes: Any) -> AtomisticCampaignState:
        values = {
            "plan_id": self.state.plan_id,
            "phase": self.state.phase,
            "round_index": self.state.round_index,
            "root_key": self.state.root_key,
            "tasks": tuple(self.records[task_id] for task_id in self.order),
            "queues": tuple(tuple(queue) for queue in self.queues),
            "transfers": tuple(
                (bucket, tuple(tasks))
                for bucket, tasks in sorted(self.transfers.items())
                if tasks
            ),
            "banks": tuple(self.banks[bucket] for bucket in self.bank_order),
            "routes": tuple(
                (stage, family, bucket)
                for (stage, family), bucket in sorted(self.routes.items())
            ),
            "pending": tuple(self.pending),
            "receipts": tuple(self.receipts),
            "commit": self.state.commit,
            "evidence": self.state.evidence,
        }
        values.update(changes)
        return AtomisticCampaignState(**values)


def _vacate(bank: AtomisticLaneBank, slot: int, /) -> AtomisticLaneBank:
    occupant = list(bank.occupant)
    held = list(bank.held)
    occupant[slot] = None
    held[slot] = None
    outcome = bank.outcome.copy()
    outcome[slot] = int(AtomisticStageOutcome.ACTIVE)
    campaign = bank.campaign_reasons.copy()
    campaign[slot] = 0
    stage = bank.lane_stage.copy()
    stage[slot] = -1
    return _replace(
        bank,
        occupant=tuple(occupant),
        held=tuple(held),
        outcome=outcome,
        campaign_reasons=campaign,
        lane_stage=stage,
    )


@dataclass(frozen=True, slots=True, eq=False)
class _SpawnedChild:
    """Routable child of a terminal lane: stable identity, family, stripped packet."""

    child_id: str
    family: str
    packet: AtomisticTaskPacket


@dataclass(frozen=True, slots=True, eq=False)
class _StagedMember:
    """Staging decision of one terminal lane.

    ``target`` is the next stage position of the outcome route (``None``:
    terminal); ``continuation`` is the packet of a task that moves on (its
    dynamics continuation kept, any other method state stripped);
    ``final`` is a terminal walker's last dynamics continuation packet.
    """

    task_id: str
    bucket: str
    slot: int
    outcome: int
    campaign: int
    target: int | None
    continuation: AtomisticTaskPacket | None
    children: tuple[_SpawnedChild, ...]
    output: AtomisticCampaignOutput | None
    final: AtomisticTaskPacket | None


def _carries_dynamics(packet: AtomisticTaskPacket | None, /) -> bool:
    return packet is not None and isinstance(
        packet.continuation, AtomisticDynamicsContinuation
    )


def _observed(
    carried: AtomisticTaskPacket, observed: AtomisticTaskPacket, /
) -> AtomisticTaskPacket:
    """An observed configuration with the walker's momenta, time and continuation.

    The observing stage (labeling) evaluated the carried configuration without
    moving it, so the walker's dynamics continuation still describes it.
    """
    return AtomisticTaskPacket(
        positions=observed.positions,
        cell_vectors=observed.cell_vectors,
        momenta=carried.momenta,
        species=observed.species,
        atom_type_ids=observed.atom_type_ids,
        atom_mask=observed.atom_mask,
        path=observed.path,
        path_energies=observed.path_energies,
        time=carried.time,
        evaluation=observed.evaluation,
        continuation=carried.continuation,
    )


@final
class PreparedAtomisticCampaign:
    """Validated campaign plan and its round transaction phases."""

    __slots__ = ("_stage_index", "plan", "plan_id")

    def __init__(self, plan: AtomisticCampaignPlan, /) -> None:
        if not isinstance(plan, AtomisticCampaignPlan):
            raise TypeError("plan must be an AtomisticCampaignPlan.")
        names = [stage.name for stage in plan.stages]
        if len(set(names)) != len(names):
            raise ValueError("Stage names must be unique.")
        index = {name: position for position, name in enumerate(names)}
        self._require_stage_programs(plan, index)
        self._require_campaign_tasks(plan, names, index)
        self.plan = plan
        self._stage_index = index
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-campaign-plan",
                "campaign": plan.campaign_id,
                "stages": [
                    {
                        "name": stage.name,
                        "operations": {
                            family: operation.binding_id
                            for family, operation in sorted(stage.operations.items())
                        },
                        "budget": stage.budget,
                        "routes": [
                            None if route is None else [route.next_stage, route.publish]
                            for route in (
                                stage.routing.converged,
                                stage.routing.completed,
                                stage.routing.exhausted,
                                stage.routing.failed,
                                stage.routing.capacity_refused,
                            )
                        ],
                        "lane_capacity": stage.lane_capacity,
                        "queue_capacity": stage.queue_capacity,
                        "entry": stage.entry.value,
                        "revision": stage.revision.value,
                        "continuation": stage.continuation.value,
                    }
                    for stage in plan.stages
                ],
                "tasks": [task.record() for task in plan.tasks],
                "max_pending": plan.max_pending_publications,
            }
        )

    def _require_stage_programs(
        self, plan: AtomisticCampaignPlan, index: Mapping[str, int], /
    ) -> None:
        """Validate routes and shared operation capacities before binding a plan."""
        bindings: dict[str, int] = {}
        for position, stage in enumerate(plan.stages):
            for route in self._routes_of(plan, position):
                if route.next_stage is not None and route.next_stage not in index:
                    raise ValueError(
                        f"Stage {stage.name} routes to unknown stage {route.next_stage}."
                    )
            for family, operation in stage.operations.items():
                self._require_stage_operation(stage, family, operation, bindings)
                self._require_operation_routes(plan, position, family, operation, index)

    @staticmethod
    def _require_stage_operation(
        stage: AtomisticStagePlan,
        family: str,
        operation: AbstractAtomisticStageOperation,
        bindings: dict[str, int],
        /,
    ) -> None:
        """Admit one operation's lane, entry, revision and finite-work contracts."""
        native = operation.native_lanes
        if native is not None and native != stage.lane_capacity:
            raise ValueError(
                f"Stage {stage.name} family {family}: the native group count "
                "must equal lane_capacity."
            )
        previous = bindings.setdefault(operation.binding_id, stage.lane_capacity)
        if previous != stage.lane_capacity:
            raise ValueError("A shared operation needs one lane capacity in every stage.")
        if (
            stage.entry is AtomisticEntryPolicy.SHARE
            and not operation.supports_shared_entry
        ):
            raise ValueError(
                f"Stage {stage.name}: {operation.kind.value} operations cannot reuse "
                "an incoming evaluation (no shared-entry transition)."
            )
        if (
            stage.revision is AtomisticRevisionPolicy.REPRIME
            and not operation.supports_rebind
        ):
            raise ValueError(
                f"Stage {stage.name}: {operation.kind.value} operations have no "
                "rebind transition; use DRAIN."
            )
        if stage.budget is None and operation.kind in (
            AtomisticStageOperationKind.DYNAMICS,
            AtomisticStageOperationKind.GENERATION,
        ):
            raise ValueError(
                f"Stage {stage.name} needs a budget for {operation.kind.value}."
            )

    def _require_operation_routes(
        self,
        plan: AtomisticCampaignPlan,
        position: int,
        family: str,
        operation: AbstractAtomisticStageOperation,
        index: Mapping[str, int],
        /,
    ) -> None:
        """Every declared continuation or generated family has a receiving owner."""
        if isinstance(operation, AtomisticGenerationStageOperation):
            targets = {
                route.next_stage
                for route in self._routes_of(plan, position)
                if route.next_stage is not None
            }
            for target in targets:
                missing = {route[0] for route in operation.routes} - set(
                    plan.stages[index[target]].operations
                )
                if missing:
                    raise ValueError(
                        f"Generated families {sorted(missing)} have no operation in {target}."
                    )
        else:
            for route in self._routes_of(plan, position):
                if route.next_stage is None:
                    continue
                if family not in plan.stages[index[route.next_stage]].operations:
                    raise ValueError(
                        f"Family {family} has no operation in stage {route.next_stage}."
                    )

    @staticmethod
    def _require_campaign_tasks(
        plan: AtomisticCampaignPlan,
        names: Sequence[str],
        index: Mapping[str, int],
        /,
    ) -> None:
        """Task identities and whole groups fit their declared entry and lane banks."""
        identifiers = [task.task_id for task in plan.tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Task IDs must be unique.")
        groups: dict[str, list[AtomisticCampaignTask]] = {}
        for task in plan.tasks:
            entry = names[0] if task.entry_stage is None else task.entry_stage
            if entry not in index:
                raise ValueError(f"Task {task.task_id} enters unknown stage {entry}.")
            if task.family not in plan.stages[index[entry]].operations:
                raise ValueError(
                    f"Task {task.task_id} family has no operation in {entry}."
                )
            if task.group_id is not None:
                groups.setdefault(task.group_id, []).append(task)
        for group, members in groups.items():
            if len({(task.family, task.entry_stage) for task in members}) != 1:
                raise ValueError(
                    f"Group {group} members need one family and entry stage."
                )
            family = members[0].family
            for stage in plan.stages:
                if family in stage.operations and len(members) > stage.lane_capacity:
                    raise ValueError(
                        f"Group {group} exceeds the lane capacity of {stage.name}."
                    )

    # -- routing ----------------------------------------------------------------------

    @staticmethod
    def _routes_of(
        plan: AtomisticCampaignPlan, position: int, /
    ) -> tuple[AtomisticStageRoute, ...]:
        return tuple(
            PreparedAtomisticCampaign._route_in(plan, position, outcome)
            for outcome in _TERMINAL
        )

    @staticmethod
    def _route_in(
        plan: AtomisticCampaignPlan, position: int, outcome: int, /
    ) -> AtomisticStageRoute:
        stage = plan.stages[position]
        routing = stage.routing
        following = (
            plan.stages[position + 1].name if position + 1 < len(plan.stages) else None
        )
        success = AtomisticStageRoute(following, following is None)
        terminal = AtomisticStageRoute(None, True)
        choice = {
            int(AtomisticStageOutcome.CONVERGED): (routing.converged, success),
            int(AtomisticStageOutcome.COMPLETED): (routing.completed, success),
            int(AtomisticStageOutcome.EXHAUSTED): (routing.exhausted, terminal),
            int(AtomisticStageOutcome.FAILED): (routing.failed, terminal),
            int(AtomisticStageOutcome.CAPACITY_REFUSED): (
                routing.capacity_refused,
                terminal,
            ),
        }[int(outcome)]
        return choice[1] if choice[0] is None else choice[0]

    def route(self, stage: str, outcome: AtomisticStageOutcome, /) -> AtomisticStageRoute:
        return self._route_in(self.plan, self._stage_index[stage], int(outcome))

    def _require(self, state: AtomisticCampaignState, *phases: _Phase) -> None:
        if not isinstance(state, AtomisticCampaignState):
            raise TypeError("state must be an AtomisticCampaignState.")
        if state.plan_id != self.plan_id:
            raise ValueError("The campaign state belongs to another plan.")
        if state.phase not in {phase.value for phase in phases}:
            raise ValueError(
                f"Phase {state.phase!r} cannot enter this transaction; "
                f"expected {[phase.value for phase in phases]}."
            )

    def _sources(self, work: _Working, /) -> tuple[AbstractAtomisticStageOperation, ...]:
        """Resident and plan operations that may own a continuation's runtime."""
        unique: dict[int, AbstractAtomisticStageOperation] = {}
        for bucket in work.bank_order:
            operation = work.banks[bucket].operation
            unique.setdefault(id(operation), operation)
        for stage in self.plan.stages:
            for operation in stage.operations.values():
                unique.setdefault(id(operation), operation)
        return tuple(unique.values())

    def terminal_packets(
        self, state: AtomisticCampaignState, /
    ) -> dict[str, AtomisticTaskPacket]:
        """Final dynamics continuation packet of every terminal walker.

        Keys are task IDs of terminal tasks whose last visit ended with an
        `AtomisticDynamicsContinuation` (a dynamics stage, or labeling of a
        walker), whatever the outcome; ``state.task(task_id).outcome`` tells
        whether the visit completed.  The packet re-enters another campaign as
        an `AtomisticCampaignTask` initial packet (continued, not restarted);
        a changed model is entered through the source operation's
        `AtomisticDynamicsStageOperation.transition_packet` first.
        """
        if not isinstance(state, AtomisticCampaignState):
            raise TypeError("state must be an AtomisticCampaignState.")
        if state.plan_id != self.plan_id:
            raise ValueError("The campaign state belongs to another plan.")
        return {
            record.task.task_id: record.carried
            for record in state.tasks
            if record.status is AtomisticTaskStatus.TERMINAL
            and record.carried is not None
        }

    # -- start --------------------------------------------------------------------------

    def start(self, root_key: Array, /) -> AtomisticCampaignState:
        """Queue every plan task at its entry stage; banks start vacant."""
        key = jnp.asarray(root_key)
        words = np.asarray(jax.random.key_data(key), dtype=np.uint32)
        if words.shape != (2,):
            raise ValueError("root_key must be one JAX PRNG key.")
        plan = self.plan
        stages = len(plan.stages)
        records = []
        queues: list[list[str]] = [[] for _ in range(stages)]
        for task in plan.tasks:
            entry = self._stage_index[
                plan.stages[0].name if task.entry_stage is None else task.entry_stage
            ]
            records.append(
                AtomisticTaskRecord(
                    task=task,
                    stage=entry,
                    status=AtomisticTaskStatus.QUEUED,
                    packet=task.packet,
                    bucket=None,
                    lane=-1,
                    entries=(0,) * stages,
                    consumed=0,
                    outcome=int(AtomisticStageOutcome.ACTIVE),
                    reasons=0,
                    campaign_reasons=0,
                    parent=None,
                )
            )
            queues[entry].append(task.task_id)
        banks: dict[str, AtomisticLaneBank] = {}
        routes = []
        for position, stage in enumerate(plan.stages):
            for family, operation in sorted(stage.operations.items()):
                bucket = operation.binding_id
                if bucket not in banks:
                    banks[bucket] = _vacant_bank(
                        bucket, stage.name, family, operation, stage.lane_capacity, None
                    )
                routes.append((position, family, bucket))
        return AtomisticCampaignState(
            plan_id=self.plan_id,
            phase=_Phase.REFILLED.value,
            round_index=0,
            root_key=words,
            tasks=tuple(records),
            queues=tuple(tuple(queue) for queue in queues),
            transfers=(),
            banks=tuple(banks.values()),
            routes=tuple(sorted(routes)),
            pending=(),
            receipts=(),
            commit=None,
            evidence=AtomisticRoundEvidence(),
        )

    # -- evaluate -------------------------------------------------------------------------

    def evaluate(self, state: AtomisticCampaignState, /) -> AtomisticCampaignState:
        """Advance every admitted active lane by one native segment.

        Capacity-pending lanes then enter their operation's capacity
        lifecycle: rebucketed tasks leave their lane for the transfer queue of
        the grown binding (re-primed, no physical or RNG advancement);
        grow-mode banks are replaced in place.
        """
        self._require(state, _Phase.REFILLED)
        work = _Working(state)
        advanced = transfers_count = growths = refusals = 0
        lane_capacity = occupied_total = 0
        for bucket in list(work.bank_order):
            bank = work.banks[bucket]
            occupied = bank.occupied
            lane_capacity += bank.lane_count
            occupied_total += int(np.sum(occupied))
            held = np.asarray([value is not None for value in bank.held], dtype=np.bool_)
            runnable = (
                occupied & ~held & (bank.outcome == int(AtomisticStageOutcome.ACTIVE))
            )
            if bank.lanes is None or not np.any(runnable):
                continue
            step = _ADVANCE(
                bank.operation,
                bank.lanes,
                jnp.asarray(runnable),
                jnp.asarray(bank.budget),
                jnp.asarray(bank.consumed),
            )
            outcome = bank.outcome.copy()
            consumed = bank.consumed.copy()
            reasons = bank.reasons.copy()
            outcome[runnable] = np.asarray(step.outcome)[runnable]
            consumed[runnable] = np.asarray(step.consumed)[runnable]
            reasons[runnable] = np.asarray(step.reasons)[runnable]
            advanced += int(np.sum(runnable))
            bank = _replace(
                bank,
                lanes=step.lanes,
                outcome=outcome,
                consumed=consumed,
                reasons=reasons,
            )
            pending_slots = np.flatnonzero(
                runnable & (outcome == int(AtomisticStageOutcome.CAPACITY_PENDING))
            )
            if pending_slots.size == 0:
                work.banks[bucket] = bank
                continue
            growth = bank.operation.grow(bank.lanes, pending_slots)
            if bank.operation.capacity_mode is AtomisticCapacityMode.GROW:
                if growth.outcome is None:
                    raise RuntimeError("A grow-mode operation must report lane outcomes.")
                outcome = bank.outcome.copy()
                campaign = bank.campaign_reasons.copy()
                for slot in pending_slots:
                    outcome[slot] = int(growth.outcome[slot])
                    if outcome[slot] == int(AtomisticStageOutcome.CAPACITY_REFUSED):
                        campaign[slot] |= int(AtomisticCampaignReason.LADDER_EXHAUSTED)
                        refusals += 1
                grown = _replace(
                    bank,
                    bucket_id=growth.operation.binding_id,
                    operation=growth.operation,
                    lanes=growth.lanes,
                    outcome=outcome,
                    campaign_reasons=campaign,
                )
                work.rename_bank(bucket, grown)
                growths += 1
                continue
            for transfer in growth.transfers:
                slot = int(transfer.slot)
                task_id = bank.occupant[slot]
                if task_id is None:
                    raise RuntimeError("A capacity transfer names a vacant lane.")
                if transfer.operation is None or transfer.packet is None:
                    outcome = bank.outcome.copy()
                    campaign = bank.campaign_reasons.copy()
                    outcome[slot] = int(transfer.outcome)
                    campaign[slot] |= int(transfer.reasons)
                    bank = _replace(bank, outcome=outcome, campaign_reasons=campaign)
                    refusals += 1
                    continue
                target = transfer.operation.binding_id
                if target not in work.banks:
                    work.add_bank(
                        _vacant_bank(
                            target,
                            self.plan.stages[int(bank.lane_stage[slot])].name,
                            bank.family,
                            transfer.operation,
                            bank.lane_count,
                            _template_packet(transfer.packet),
                        )
                    )
                record = work.records[task_id]
                work.records[task_id] = _replace(
                    record,
                    status=AtomisticTaskStatus.QUEUED,
                    packet=transfer.packet,
                    bucket=target,
                    lane=-1,
                    consumed=int(bank.consumed[slot]),
                    campaign_reasons=int(bank.campaign_reasons[slot])
                    | int(transfer.reasons),
                )
                work.transfers.setdefault(target, []).append(task_id)
                bank = _vacate(bank, slot)
                transfers_count += 1
            work.banks[bucket] = bank
        evidence = AtomisticRoundEvidence(
            round_index=state.round_index,
            lane_capacity=lane_capacity,
            occupied_lanes=occupied_total,
            advanced_lanes=advanced,
            capacity_transfers=transfers_count,
            capacity_growths=growths,
            capacity_refusals=refusals,
        )
        return work.freeze(phase=_Phase.EVALUATED.value, evidence=evidence)

    # -- stage outputs ----------------------------------------------------------------------

    def stage_outputs(self, state: AtomisticCampaignState, /) -> AtomisticCampaignState:
        """Classify terminal lanes and move whole groups or park them.

        A group leaves a stage only when every member still in the stage is
        terminal. A unit proceeds only if its receiving queues and publication
        bank both admit it; otherwise its accepted lanes remain parked.
        """
        self._require(state, _Phase.EVALUATED)
        plan = self.plan
        work = _Working(state)
        located, units, waiting = self._ready_stage_units(work)
        plans: list[list[_StagedMember]] = []
        for stage_position, _, members in units:
            stage = plan.stages[stage_position]
            member_plans: list[_StagedMember] = []
            for task_id in members:
                member_plans.append(
                    self._stage_member(
                        work, stage_position, stage, task_id, located[task_id]
                    )
                )
            plans.append(member_plans)
        accepted = self._stage_allocation(work, plans)
        parked, terminal, spawned = self._commit_stage_moves(work, plans, accepted)
        for bucket in list(work.bank_order):
            bank = work.banks[bucket]
            if (
                bank.retired
                and not np.any(bank.occupied)
                and not work.transfers.get(bucket)
                and bucket not in work.routes.values()
            ):
                del work.banks[bucket]
                work.bank_order.remove(bucket)
        evidence = _replace(
            state.evidence,
            parked=parked,
            group_waiting=waiting,
            terminal_tasks=terminal,
            spawned=spawned,
            pending=len(work.pending),
            backpressured=bool(parked),
        )
        return work.freeze(phase=_Phase.STAGED.value, evidence=evidence)

    @staticmethod
    def _ready_stage_units(
        work: _Working, /
    ) -> tuple[dict[str, tuple[str, int]], list[tuple[int, str, tuple[str, ...]]], int]:
        """Locate terminal lanes and admit only complete, canonically ordered groups."""
        located: dict[str, tuple[str, int]] = {}
        for bucket in work.bank_order:
            bank = work.banks[bucket]
            for slot, task_id in enumerate(bank.occupant):
                if task_id is not None and int(bank.outcome[slot]) in _TERMINAL:
                    located[task_id] = (bucket, slot)
        units: list[tuple[int, str, tuple[str, ...]]] = []
        seen: set[str] = set()
        waiting = 0
        for task_id in sorted(located):
            record = work.records[task_id]
            group = record.task.group_id
            unit = task_id if group is None else group
            if unit in seen:
                continue
            members = (
                (task_id,)
                if group is None
                else tuple(
                    sorted(
                        other_id
                        for other_id, other in work.records.items()
                        if other.task.group_id == group
                        and other.stage == record.stage
                        and other.status is not AtomisticTaskStatus.TERMINAL
                    )
                )
            )
            seen.add(unit)
            if not all(member in located for member in members):
                waiting += len([member for member in members if member in located])
                continue
            units.append((record.stage, unit, members))
        units.sort()
        return located, units, waiting

    def _classify_stage_packets(
        self,
        work: _Working,
        packets: tuple[AtomisticTaskPacket, ...],
        route: AtomisticStageRoute,
        outcome: int,
        campaign: int,
        /,
    ) -> tuple[AtomisticTaskPacket | None, list[_SpawnedChild], int, int]:
        """Separate the resident continuation from uniquely routable generated tasks."""
        plan = self.plan
        children: list[_SpawnedChild] = []
        own: AtomisticTaskPacket | None = None
        for packet in packets:
            child_id = packet.child_id
            if child_id is None:
                own = packet
                continue
            family = packet.family
            next_stage = route.next_stage
            if (
                family is None
                or next_stage is None
                or family not in plan.stages[self._stage_index[next_stage]].operations
                or child_id in work.records
            ):
                campaign |= int(AtomisticCampaignReason.UNROUTABLE)
                if outcome in (
                    int(AtomisticStageOutcome.COMPLETED),
                    int(AtomisticStageOutcome.CONVERGED),
                ):
                    outcome = int(AtomisticStageOutcome.FAILED)
                continue
            children.append(_SpawnedChild(child_id, family, packet.stripped()))
        return own, children, outcome, campaign

    def _stage_member(
        self,
        work: _Working,
        stage_position: int,
        stage: AtomisticStagePlan,
        task_id: str,
        location: tuple[str, int],
        /,
    ) -> _StagedMember:
        """Bind one outcome's publication, continuation and RNG evidence together."""
        plan = self.plan
        bucket, slot = location
        bank = work.banks[bucket]
        record = work.records[task_id]
        outcome = int(bank.outcome[slot])
        campaign = int(bank.campaign_reasons[slot])
        held = bank.held[slot]
        packets = (held,) if held is not None else bank.operation.export(bank.lanes, slot)
        route = self._route_in(plan, stage_position, outcome)
        own, children, outcome, campaign = self._classify_stage_packets(
            work, packets, route, outcome, campaign
        )
        if outcome != int(bank.outcome[slot]):
            route = self._route_in(plan, stage_position, outcome)
            children = []
        if (
            own is not None
            and record.carried is not None
            and bank.operation.preserves_configuration
        ):
            # The observing stage left the walker's phase point intact.
            own = _observed(record.carried, own)
        next_stage = route.next_stage
        target = None if next_stage is None else self._stage_index[next_stage]
        if own is not None and isinstance(
            own.continuation, AtomisticDynamicsContinuation
        ):
            words = np.asarray(own.continuation.state.random_key, dtype=np.uint32)
        else:
            # Refused admission has no live lane; its visit key is evidence
            # only, not an executed method's random stream.
            words = (
                None
                if held is not None
                else bank.operation.lane_random_words(bank.lanes, slot)
            )
        entry = record.entries[stage_position] - 1
        output = None
        if route.publish:
            output = _make_output(
                plan.campaign_id,
                record,
                stage.name,
                entry,
                outcome,
                own
                if own is not None
                else (children[0].packet if children else AtomisticTaskPacket()),
                bank.operation.numeric_revision_ids,
                bank.keys[slot] if words is None else words,
                {"consumed": int(bank.consumed[slot]), "stage_entry": int(entry)},
                int(bank.reasons[slot]),
                campaign,
            )
        continuation = None
        final = None
        if own is not None and target is not None and not children:
            continuation = own if _carries_dynamics(own) else own.stripped()
        elif _carries_dynamics(own):
            final = own
        return _StagedMember(
            task_id,
            bucket,
            slot,
            outcome,
            campaign,
            target,
            continuation,
            tuple(children),
            output,
            final,
        )

    @staticmethod
    def _stage_needs(members: list[_StagedMember], /) -> tuple[dict[int, int], int]:
        """Quota requested by one indivisible unit in queues and publication storage."""
        per_stage: dict[int, int] = {}
        outputs = 0
        for member in members:
            if member.target is not None and (
                member.continuation is not None or member.children
            ):
                per_stage[member.target] = per_stage.get(member.target, 0) + (
                    len(member.children) if member.children else 1
                )
            outputs += member.output is not None
        return per_stage, outputs

    def _stage_allocation(
        self, work: _Working, plans: list[list[_StagedMember]], /
    ) -> np.ndarray:
        """Intersect native prefix admission across every destination and publication."""
        plan = self.plan
        queue_needs: dict[int, list[tuple[int, int]]] = {}
        pending_needs: list[tuple[int, int]] = []
        for position, member_plans in enumerate(plans):
            per_stage, outputs = self._stage_needs(member_plans)
            for target, need in per_stage.items():
                queue_needs.setdefault(target, []).append((position, need))
            if outputs:
                pending_needs.append((position, outputs))
        accepted = np.ones((len(plans),), dtype=np.bool_)
        for target, needs in queue_needs.items():
            allocation = allocate_frontier_slots(
                len(work.queues[target]),
                np.asarray([need for _, need in needs], dtype=np.int32),
                plan.stages[target].queue_capacity,
            )
            for (position, _), ok in zip(
                needs, np.asarray(allocation.accepted), strict=True
            ):
                accepted[position] &= bool(ok)
        if pending_needs:
            allocation = allocate_frontier_slots(
                len(work.pending),
                np.asarray([need for _, need in pending_needs], dtype=np.int32),
                plan.max_pending_publications,
            )
            for (position, _), ok in zip(
                pending_needs, np.asarray(allocation.accepted), strict=True
            ):
                accepted[position] &= bool(ok)
        return accepted

    def _commit_stage_moves(
        self,
        work: _Working,
        plans: list[list[_StagedMember]],
        accepted: np.ndarray,
        /,
    ) -> tuple[int, int, int]:
        """Vacate admitted lanes and publish, queue or terminate their whole units."""
        plan = self.plan
        parked = terminal = spawned = 0
        for position, member_plans in enumerate(plans):
            if not accepted[position]:
                parked += len(member_plans)
                continue
            for member in member_plans:
                task_id = member.task_id
                bank = work.banks[member.bucket]
                consumed = int(bank.consumed[member.slot])
                reasons = int(bank.reasons[member.slot])
                work.banks[member.bucket] = _vacate(bank, member.slot)
                record = work.records[task_id]
                if member.output is not None:
                    work.pending.append(member.output)
                if member.target is not None and member.continuation is not None:
                    work.records[task_id] = _replace(
                        record,
                        stage=member.target,
                        status=AtomisticTaskStatus.QUEUED,
                        packet=member.continuation,
                        bucket=None,
                        lane=-1,
                        consumed=0,
                        outcome=int(AtomisticStageOutcome.ACTIVE),
                        reasons=reasons,
                        campaign_reasons=member.campaign,
                        lineage=(),
                        carried=None,
                    )
                    work.queues[member.target].append(task_id)
                    continue
                work.records[task_id] = _replace(
                    record,
                    status=AtomisticTaskStatus.TERMINAL,
                    packet=None,
                    bucket=None,
                    lane=-1,
                    consumed=consumed,
                    outcome=member.outcome,
                    reasons=reasons,
                    campaign_reasons=member.campaign,
                    carried=member.final,
                )
                terminal += 1
                if member.children and member.target is not None:
                    target = member.target
                    entry_stage = plan.stages[target].name
                    for child in member.children:
                        child_task = AtomisticCampaignTask(
                            child.child_id,
                            child.family,
                            child.packet,
                            material_id=child.child_id,
                            run_id=record.task.run_id,
                            replica_id=record.task.replica_id,
                            entry_stage=entry_stage,
                            ordinal=record.task.ordinal,
                        )
                        work.records[child.child_id] = AtomisticTaskRecord(
                            task=child_task,
                            stage=target,
                            status=AtomisticTaskStatus.QUEUED,
                            packet=child_task.packet,
                            bucket=None,
                            lane=-1,
                            entries=(0,) * len(plan.stages),
                            consumed=0,
                            outcome=int(AtomisticStageOutcome.ACTIVE),
                            reasons=0,
                            campaign_reasons=0,
                            parent=task_id,
                        )
                        work.order.append(child.child_id)
                        work.queues[target].append(child.child_id)
                        spawned += 1
        return parked, terminal, spawned

    # -- publish ----------------------------------------------------------------------------

    def publish(
        self, state: AtomisticCampaignState, sink: AbstractAtomisticCampaignSink, /
    ) -> AtomisticCampaignState:
        """Obtain durable receipts in order; stop at the first backpressure."""
        self._require(state, _Phase.STAGED)
        if not isinstance(sink, AbstractAtomisticCampaignSink):
            raise TypeError("sink must be an AbstractAtomisticCampaignSink.")
        pending = list(state.pending)
        receipts = list(state.receipts)
        published = 0
        backpressured = False
        while pending:
            record = pending[0]
            receipt = sink.receipt(record.semantic_key)
            if receipt is None:
                receipt = sink.publish(record, round_index=state.round_index)
            if receipt is None:
                backpressured = True
                break
            if (
                not isinstance(receipt, AtomisticOutputReceipt)
                or receipt.semantic_key != record.semantic_key
                or receipt.payload_id != record.payload_id
            ):
                raise ValueError(
                    "The sink returned a receipt for another output payload."
                )
            receipts.append(receipt)
            pending.pop(0)
            published += 1
        evidence = _replace(
            state.evidence,
            published=published,
            pending=len(pending),
            backpressured=state.evidence.backpressured or backpressured,
        )
        return _replace(
            state,
            phase=_Phase.PUBLISHED.value,
            pending=tuple(pending),
            receipts=tuple(receipts),
            evidence=evidence,
        )

    # -- commit -----------------------------------------------------------------------------

    def commit(
        self, state: AtomisticCampaignState, store: AtomisticCampaignStore, /
    ) -> AtomisticCampaignState:
        """Write the complete continuation envelope atomically."""
        self._require(state, _Phase.PUBLISHED)
        if not isinstance(store, AtomisticCampaignStore):
            raise TypeError("store must be an AtomisticCampaignStore.")
        committed = _replace(state, phase=_Phase.COMMITTED.value, commit=None)
        manifest, arrays = _encode_state(self, committed)
        location = store.write(manifest, arrays)
        return _replace(
            committed,
            commit=AtomisticCommitReceipt(
                state.round_index, manifest["payload_id"], location
            ),
        )

    # -- refill -----------------------------------------------------------------------------

    def refill(self, state: AtomisticCampaignState, /) -> AtomisticCampaignState:
        """Admit queued groups into free lanes after the durable commit.

        Transfers precede stage queues. Semantic keys depend only on stage,
        task and visit count, never lanes, rounds or completion order. Native
        prefix allocation and refill routing admit each whole group atomically.
        """
        self._require(state, _Phase.COMMITTED)
        plan = self.plan
        work = _Working(state)
        candidates, flat = self._refill_candidates(work)
        admitted_total = shared_total = 0
        if flat:
            keys, key_index = self._refill_keys(work, flat)
            sources: tuple[AbstractAtomisticStageOperation, ...] | None = None
            for bucket in list(work.bank_order):
                entries = candidates.get(bucket, [])
                if not entries or bucket not in work.banks:
                    continue
                bank = work.banks[bucket]
                dynamics_bank = (
                    bank.operation.kind is AtomisticStageOperationKind.DYNAMICS
                )
                allocation = self._refill_slots(work, bank, entries)
                if allocation is None:
                    continue
                chosen, slots = allocation
                names = [
                    f"{plan.stages[work.records[task_id].stage].name}/{task_id}"
                    for task_id, _ in chosen
                ]
                packets, carried, restarted = self._refill_packets(
                    work, bank, chosen, dynamics_bank
                )
                lane_keys = keys[
                    jnp.asarray([key_index[name] for name in names], dtype=jnp.int32)
                ]
                ordinals = np.asarray(
                    [work.records[task_id].task.ordinal for task_id, _ in chosen],
                    dtype=np.int64,
                )
                if sources is None:
                    sources = self._sources(work)
                admission = bank.operation.admit(
                    bank.lanes,
                    bank.lane_count,
                    slots,
                    packets,
                    lane_keys,
                    ordinals,
                    sources=sources,
                )
                admitted, shared = self._apply_refill_admission(
                    work,
                    bank,
                    bucket,
                    chosen,
                    slots,
                    packets,
                    carried,
                    restarted,
                    lane_keys,
                    admission,
                )
                admitted_total += admitted
                shared_total += shared
        evidence = _replace(
            state.evidence, admitted=admitted_total, shared_evaluations=shared_total
        )
        return work.freeze(
            phase=_Phase.REFILLED.value,
            round_index=state.round_index + 1,
            evidence=evidence,
        )

    @staticmethod
    def _refill_candidates(
        work: _Working, /
    ) -> tuple[dict[str, list[tuple[str, bool]]], list[tuple[str, str, bool]]]:
        """Preserve transfer priority, queue arrival order and bank traversal order."""
        candidates: dict[str, list[tuple[str, bool]]] = {}
        for bucket, tasks in work.transfers.items():
            candidates.setdefault(bucket, []).extend((task_id, True) for task_id in tasks)
        for position, queue in enumerate(work.queues):
            for task_id in queue:
                record = work.records[task_id]
                bucket = work.routes[(position, record.task.family)]
                candidates.setdefault(bucket, []).append((task_id, False))
        flat = [
            (bucket, task_id, transfer)
            for bucket in work.bank_order
            for task_id, transfer in candidates.get(bucket, ())
        ]
        return candidates, flat

    def _refill_keys(
        self, work: _Working, flat: list[tuple[str, str, bool]], /
    ) -> tuple[Array, dict[str, int]]:
        """Address workset keys by semantic task visits, reusing transfer counters."""
        plan = self.plan
        semantic = [
            f"{plan.stages[work.records[task_id].stage].name}/{task_id}"
            for _, task_id, _ in flat
        ]
        signatures = tuple(
            work.banks[bucket].operation.signature(work.banks[bucket].lane_count).pool
            for bucket, _, _ in flat
        )
        workset = ExecutionWorksetPlan(
            tuple(semantic),
            signatures,
            bucket_capacity=min(64, max(bank.lane_count for bank in work.banks.values())),
        ).prepare()
        position = {name: index for index, name in enumerate(workset.plan.semantic_ids)}
        counters = np.zeros((workset.item_count,), dtype=np.uint32)
        for name, (_, task_id, transfer) in zip(semantic, flat, strict=True):
            record = work.records[task_id]
            count = record.entries[record.stage] - (1 if transfer else 0)
            counters[position[name]] = count
        keys = workset.semantic_keys(
            jax.random.wrap_key_data(jnp.asarray(work.state.root_key, dtype=jnp.uint32)),
            jnp.asarray(counters),
        ).reshape((-1,))
        item_bucket = np.asarray(workset.item_bucket)
        item_slot = np.asarray(workset.item_slot)
        key_index = {
            name: int(item_bucket[index]) * workset.bucket_capacity
            + int(item_slot[index])
            for name, index in position.items()
        }
        return keys, key_index

    @staticmethod
    def _refill_slots(
        work: _Working,
        bank: AtomisticLaneBank,
        entries: list[tuple[str, bool]],
        /,
    ) -> tuple[list[tuple[str, bool]], np.ndarray] | None:
        """Allocate an indivisible prefix of groups, then route it to vacant slots."""
        units: list[list[tuple[str, bool]]] = []
        grouped: dict[str, list[tuple[str, bool]]] = {}
        for task_id, transfer in entries:
            group = work.records[task_id].task.group_id
            if group is None or transfer:
                units.append([(task_id, transfer)])
            elif group in grouped:
                grouped[group].append((task_id, transfer))
            else:
                grouped[group] = [(task_id, transfer)]
                units.append(grouped[group])
        free = ~bank.occupied
        if not np.any(free):
            return None
        sizes = np.asarray([len(unit) for unit in units], dtype=np.int32)
        allocation = allocate_frontier_slots(int(np.sum(~free)), sizes, bank.lane_count)
        chosen = [
            member
            for unit, ok in zip(units, np.asarray(allocation.accepted), strict=True)
            if ok
            for member in unit
        ]
        # Prefix semantics: a refused unit blocks every later unit.
        if not chosen:
            return None
        routing = refill_completed_tasks(
            jnp.arange(bank.lane_count, dtype=jnp.int32),
            jnp.asarray(free),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            len(chosen),
        )
        refill_mask = np.asarray(routing.refill_mask)
        rank = np.asarray(routing.terminal_rank)
        slots = np.zeros((len(chosen),), dtype=np.int32)
        for lane in np.flatnonzero(refill_mask):
            slots[int(rank[lane])] = int(lane)
        return chosen, slots

    def _refill_packets(
        self,
        work: _Working,
        bank: AtomisticLaneBank,
        chosen: list[tuple[str, bool]],
        dynamics_bank: bool,
        /,
    ) -> tuple[
        tuple[AtomisticTaskPacket, ...], list[AtomisticTaskPacket | None], np.ndarray
    ]:
        """Apply declared observation, continuation or restart policy at admission."""
        plan = self.plan
        queued: list[AtomisticTaskPacket] = []
        carried: list[AtomisticTaskPacket | None] = []
        restarted = np.zeros((len(chosen),), dtype=np.bool_)
        for index, (task_id, transfer) in enumerate(chosen):
            record = work.records[task_id]
            queued_packet = record.packet
            if queued_packet is None:
                raise RuntimeError("A queued task has no input packet.")
            walker = record.carried
            if not transfer:
                walker = None
                if _carries_dynamics(queued_packet):
                    if not dynamics_bank:
                        # Labeling observes the configuration; other methods
                        # move it and end the incoming dynamics run.
                        if bank.operation.preserves_configuration:
                            walker = queued_packet
                        queued_packet = queued_packet.stripped()
                    elif (
                        plan.stages[record.stage].continuation
                        is AtomisticContinuationPolicy.RESTART
                    ):
                        queued_packet = queued_packet.stripped()
                        restarted[index] = True
            queued.append(queued_packet)
            carried.append(walker)
        return tuple(queued), carried, restarted

    @staticmethod
    def _begin_stage_visit(
        work: _Working,
        bucket: str,
        task_id: str,
        record: AtomisticTaskRecord,
        transfer: bool,
        restarted: bool,
        /,
    ) -> tuple[list[int], int]:
        """Spend one new visit counter even if admission immediately rebuckets it."""
        if transfer:
            work.transfers[bucket].remove(task_id)
        else:
            work.queues[record.stage].remove(task_id)
        entries = list(record.entries)
        if not transfer:
            entries[record.stage] += 1
        visit = int(AtomisticCampaignReason.DYNAMICS_RESTARTED) if restarted else 0
        return entries, visit

    def _transfer_refill(
        self,
        work: _Working,
        bank: AtomisticLaneBank,
        task_id: str,
        record: AtomisticTaskRecord,
        operation: AbstractAtomisticStageOperation,
        packet: AtomisticTaskPacket,
        carried: AtomisticTaskPacket | None,
        entries: list[int],
        transfer: bool,
        visit: int,
        /,
    ) -> None:
        """Carry the same visit and accepted packet into its grown resource bank."""
        target = operation.binding_id
        if target not in work.banks:
            work.add_bank(
                _vacant_bank(
                    target,
                    self.plan.stages[record.stage].name,
                    bank.family,
                    operation,
                    bank.lane_count,
                    _template_packet(packet),
                )
            )
        work.transfers.setdefault(target, []).append(task_id)
        work.records[task_id] = _replace(
            record,
            packet=packet,
            bucket=target,
            entries=tuple(entries),
            consumed=record.consumed if transfer else 0,
            campaign_reasons=record.campaign_reasons if transfer else visit,
            carried=carried,
        )

    def _apply_refill_admission(
        self,
        work: _Working,
        bank: AtomisticLaneBank,
        bucket: str,
        chosen: list[tuple[str, bool]],
        slots: np.ndarray,
        packets: tuple[AtomisticTaskPacket, ...],
        carried: list[AtomisticTaskPacket | None],
        restarted: np.ndarray,
        lane_keys: Array,
        admission: AtomisticLaneAdmission,
        /,
    ) -> tuple[int, int]:
        """Commit owner admission, resource transfers and refusal evidence to scratch."""
        plan = self.plan
        transferred = {index: operation for index, operation in admission.transfers}
        transitions = dict(admission.lineage)
        key_words = np.asarray(jax.random.key_data(lane_keys), dtype=np.uint32)
        occupant = list(bank.occupant)
        held = list(bank.held)
        lane_stage = bank.lane_stage.copy()
        budget = bank.budget.copy()
        consumed = bank.consumed.copy()
        outcome = bank.outcome.copy()
        reasons = bank.reasons.copy()
        campaign = bank.campaign_reasons.copy()
        bank_keys = bank.keys.copy()
        template = bank.template
        admitted = shared = 0
        for index, (task_id, transfer) in enumerate(chosen):
            record = work.records[task_id]
            entries, visit = self._begin_stage_visit(
                work, bucket, task_id, record, transfer, bool(restarted[index])
            )
            if index in transferred:
                self._transfer_refill(
                    work,
                    bank,
                    task_id,
                    record,
                    transferred[index],
                    packets[index],
                    carried[index],
                    entries,
                    transfer,
                    visit,
                )
                continue
            slot = int(slots[index])
            stage = plan.stages[record.stage]
            occupant[slot] = task_id
            lane_stage[slot] = record.stage
            budget[slot] = _INT32_MAX if stage.budget is None else stage.budget
            consumed[slot] = record.consumed if transfer else 0
            reasons[slot] = (
                0
                if admission.method_reasons is None
                else int(admission.method_reasons[index])
            )
            bank_keys[slot] = key_words[index]
            if bool(admission.refused[index]):
                outcome[slot] = int(admission.refusal[index])
                # A refused walker keeps its continuation as evidence.
                held[slot] = (
                    packets[index]
                    if _carries_dynamics(packets[index])
                    else packets[index].stripped()
                )
                campaign[slot] = int(admission.reasons[index]) | visit
            else:
                outcome[slot] = int(AtomisticStageOutcome.ACTIVE)
                held[slot] = None
                campaign[slot] = (
                    int(admission.reasons[index])
                    | visit
                    | (
                        int(AtomisticCampaignReason.SHARED_EVALUATION)
                        if bool(admission.shared[index])
                        else 0
                    )
                )
                if template is None:
                    template = _template_packet(packets[index])
            if transfer:
                campaign[slot] |= record.campaign_reasons
            work.records[task_id] = _replace(
                record,
                status=AtomisticTaskStatus.ACTIVE,
                packet=None,
                bucket=bucket,
                lane=slot,
                entries=tuple(entries),
                consumed=int(consumed[slot]),
                lineage=(*record.lineage, *transitions.get(index, ())),
                carried=carried[index],
            )
            admitted += 1
            shared += int(bool(admission.shared[index]))
        work.banks[bucket] = _replace(
            bank,
            lanes=admission.lanes,
            occupant=tuple(occupant),
            held=tuple(held),
            lane_stage=lane_stage,
            budget=budget,
            consumed=consumed,
            outcome=outcome,
            reasons=reasons,
            campaign_reasons=campaign,
            keys=bank_keys,
            template=template,
        )
        return admitted, shared

    # -- composition ---------------------------------------------------------------------------

    def run_round(
        self,
        state: AtomisticCampaignState,
        sink: AbstractAtomisticCampaignSink,
        store: AtomisticCampaignStore,
        /,
        *,
        session: IterationSession | None = None,
    ) -> AtomisticCampaignState:
        """One complete transaction: evaluate, stage, publish, commit, refill."""
        committed = self.commit(
            self.publish(self.stage_outputs(self.evaluate(state)), sink), store
        )
        refilled = self.refill(committed)
        if session is not None:
            session.emit(
                self.scope, _round_record(refilled.evidence, refilled.round_index)
            )
        return refilled

    def run(
        self,
        state: AtomisticCampaignState,
        sink: AbstractAtomisticCampaignSink,
        store: AtomisticCampaignStore,
        /,
        *,
        maximum_rounds: int,
        session: IterationSession | None = None,
    ) -> AtomisticCampaignState:
        """Run rounds until every task is terminal and every output is durable."""
        rounds = int(maximum_rounds)
        if rounds < 1:
            raise ValueError("maximum_rounds must be positive.")
        for _ in range(rounds):
            if state.complete:
                break
            state = self.run_round(state, sink, store, session=session)
        return state

    @property
    def scope(self) -> IterationScope:
        return IterationScope(
            f"atomistic-campaign:{self.plan.campaign_id}",
            parent_scope_id=None,
            role="atomistic-campaign",
            algorithm_id=self.plan_id,
            depth=0,
        )

    def resume(
        self,
        store: AtomisticCampaignStore,
        /,
        *,
        operations: Sequence[AbstractAtomisticStageOperation] = (),
    ) -> AtomisticCampaignState:
        """Restore the last durable commit and refill (fresh-process continuation).

        ``operations`` supplies rebound operations (other numeric revisions)
        that are not in the plan; grown capacity levels are reconstructed from
        the plan's operations.
        """
        payload = store.read()
        if payload is None:
            raise FileNotFoundError("The campaign store holds no durable commit.")
        manifest, arrays = payload
        state = _decode_state(self, manifest, arrays, tuple(operations))
        committed = _replace(
            state,
            commit=AtomisticCommitReceipt(
                state.round_index, manifest["payload_id"], str(store.path)
            ),
        )
        return self.refill(committed)

    # -- revision rebinding ----------------------------------------------------------------------

    def rebind_operation(
        self,
        state: AtomisticCampaignState,
        stage: str,
        family: str,
        operation: AbstractAtomisticStageOperation,
        /,
    ) -> AtomisticCampaignState:
        """Bind a stage family to another Hamiltonian of the same structural signature.

        ``DRAIN``: admitted lanes finish under their own Hamiltonian; new
        admissions use ``operation``.  ``REPRIME``: every admitted lane is moved
        by the owner's explicit Hamiltonian transition (forces/evidence
        re-primed, lineage recorded, physical and RNG continuation preserved by
        the owner).  The transition is atomic: an owner refusal raises and the
        state is unchanged.  REPRIME is refused while the bank holds completed
        or refused lanes awaiting publication (their outputs belong to the old
        Hamiltonian) or queued capacity transfers.  Durable at the next commit.
        """
        self._require(state, _Phase.REFILLED)
        position = self._stage_index[stage]
        plan_stage = self.plan.stages[position]
        work = _Working(state)
        current_id = work.routes[(position, family)]
        bank = work.banks[current_id]
        current = bank.operation
        if operation.kind is not current.kind:
            raise TypeError("A rebind keeps the operation kind.")
        if (
            operation.signature(bank.lane_count).signature_id
            != current.signature(bank.lane_count).signature_id
        ):
            raise ValueError("A rebind keeps the static stage signature.")
        if operation.binding_id == current_id:
            return state
        if plan_stage.revision is AtomisticRevisionPolicy.DRAIN:
            target = operation.binding_id
            if target not in work.banks:
                work.add_bank(
                    _vacant_bank(
                        target, stage, family, operation, bank.lane_count, bank.template
                    )
                )
            work.routes[(position, family)] = target
            if current_id not in work.routes.values():
                work.banks[current_id] = _replace(bank, retired=True)
            return work.freeze()
        if any(
            bucket == current_id and key != (position, family)
            for key, bucket in work.routes.items()
        ):
            raise ValueError(
                "The bank is shared with other stage families; REPRIME would move "
                "their Hamiltonian too. Rebind every sharing route explicitly or DRAIN."
            )
        if work.transfers.get(current_id):
            raise ValueError(
                "Capacity transfers are queued for this bank; rebind after their admission."
            )
        occupied = bank.occupied
        held = np.asarray([value is not None for value in bank.held], dtype=np.bool_)
        if np.any(
            occupied & (held | (bank.outcome != int(AtomisticStageOutcome.ACTIVE)))
        ):
            raise ValueError(
                "Completed or refused lanes await publication under the current "
                "Hamiltonian; REPRIME after they leave the bank."
            )
        if bank.lanes is None or not np.any(occupied):
            rebound = _replace(
                bank, bucket_id=operation.binding_id, operation=operation, lanes=None
            )
            work.rename_bank(current_id, rebound)
            return work.freeze()
        target, lanes, lineage = current.rebind(bank.lanes, occupied, operation)
        rebound = _replace(
            bank, bucket_id=target.binding_id, operation=target, lanes=lanes
        )
        work.rename_bank(current_id, rebound)
        for slot, lineage_id in lineage.items():
            task_id = bank.occupant[slot]
            if task_id is None:
                continue
            record = work.records[task_id]
            work.records[task_id] = _replace(
                record, lineage=(*record.lineage, lineage_id)
            )
        return work.freeze()


def _round_record(evidence: AtomisticRoundEvidence, ordinal: int, /) -> IterationRecord:
    metrics = {
        name: jnp.asarray(int(getattr(evidence, name)), dtype=jnp.int32)
        for name in (
            "lane_capacity",
            "occupied_lanes",
            "advanced_lanes",
            "admitted",
            "parked",
            "group_waiting",
            "pending",
            "published",
            "shared_evaluations",
            "capacity_transfers",
            "capacity_growths",
            "capacity_refusals",
            "terminal_tasks",
            "spawned",
        )
    }
    metrics["backpressured"] = jnp.asarray(evidence.backpressured)
    return IterationRecord(
        IterationCoordinates(
            IterationPhase.COMMIT,
            ordinal,
            accepted=evidence.terminal_tasks,
            active=evidence.occupied_lanes > 0,
            committed=True,
        ),
        0,
        metrics,
    )


# -- continuation envelope --------------------------------------------------------------------


_PACKET_FIELDS = (
    "positions",
    "cell_vectors",
    "momenta",
    "species",
    "atom_type_ids",
    "atom_mask",
    "path",
    "path_energies",
)


def _encode_packet(
    prefix: str, packet: AtomisticTaskPacket, arrays: dict[str, object], /
) -> dict[str, Any]:
    fields = []
    for name in _PACKET_FIELDS:
        value = getattr(packet, name)
        if value is not None:
            arrays[f"{prefix}/{name}"] = value
            fields.append(name)
    arrays[f"{prefix}/time"] = packet.time
    evaluation = None
    if packet.evaluation is not None:
        arrays[f"{prefix}/evaluation/energy"] = packet.evaluation.energy
        arrays[f"{prefix}/evaluation/forces"] = packet.evaluation.forces
        if packet.evaluation.stress is not None:
            arrays[f"{prefix}/evaluation/stress"] = packet.evaluation.stress
        evaluation = packet.evaluation.key.to_record()
    continuation = packet.continuation
    dynamics = None
    generic = None
    if isinstance(continuation, AtomisticDynamicsContinuation):
        dynamics = {
            "source_binding_id": continuation.source_binding_id,
            "layout_id": continuation.layout_id,
            "lineage": list(continuation.lineage),
            "state": pack_array_tree(
                f"{prefix}/continuation", jax.device_get(continuation.state), arrays
            ),
        }
    elif continuation is not None:
        generic = pack_array_tree(
            f"{prefix}/continuation", jax.device_get(continuation), arrays
        )
    return {
        "fields": fields,
        "evaluation": evaluation,
        "child_id": packet.child_id,
        "family": packet.family,
        "continuation": generic,
        "dynamics_continuation": dynamics,
    }


class _DynamicsStateTemplates:
    """State structures of dynamics continuations by ``continuation_layout_id``.

    Every revision of one dynamics/table layout shares one state structure,
    so any plan, resident or supplied operation of that layout restores it;
    the restored continuation keeps its own recorded source binding.
    """

    __slots__ = ("_cache", "_operations")

    def __init__(self, operations: Sequence[AbstractAtomisticStageOperation], /) -> None:
        self._operations = tuple(
            operation
            for operation in operations
            if isinstance(operation, AtomisticDynamicsStageOperation)
        )
        self._cache: dict[str, AtomisticDynamicsState] = {}

    def __call__(self, layout_id: str, /) -> AtomisticDynamicsState:
        if layout_id not in self._cache:
            for operation in self._operations:
                if operation.continuation_layout_id == layout_id:
                    self._cache[layout_id] = operation.state_template()
                    break
            else:
                raise ValueError(
                    "No plan, resident or supplied dynamics operation has the layout "
                    "of a stored walker continuation; supply the operation to resume()."
                )
        return self._cache[layout_id]


def _decode_packet(
    meta: Mapping[str, Any],
    prefix: str,
    arrays: Mapping[str, np.ndarray],
    continuation_template: Any,
    dynamics_templates: _DynamicsStateTemplates,
    /,
) -> AtomisticTaskPacket:
    values = {name: np.asarray(arrays[f"{prefix}/{name}"]) for name in meta["fields"]}
    evaluation = None
    if meta["evaluation"] is not None:
        key = AtomisticEvaluationKey.from_record(meta["evaluation"])
        stress_name = f"{prefix}/evaluation/stress"
        evaluation = AtomisticEvaluationRecord(
            arrays[f"{prefix}/evaluation/energy"],
            arrays[f"{prefix}/evaluation/forces"],
            arrays[stress_name] if stress_name in arrays else None,
            key,
        )
    continuation: Any = None
    dynamics = meta["dynamics_continuation"]
    if dynamics is not None:
        state = unpack_array_tree(
            dynamics["state"], arrays, dynamics_templates(dynamics["layout_id"])
        )
        continuation = AtomisticDynamicsContinuation(
            jax.device_get(state),
            source_binding_id=dynamics["source_binding_id"],
            layout_id=dynamics["layout_id"],
            lineage=tuple(dynamics["lineage"]),
        )
    elif meta["continuation"] is not None:
        if continuation_template is None:
            raise ValueError(
                "A transferred continuation needs its destination bank template."
            )
        continuation = unpack_array_tree(
            meta["continuation"], arrays, continuation_template
        )
    return AtomisticTaskPacket(
        **values,
        time=arrays[f"{prefix}/time"],
        evaluation=evaluation,
        continuation=continuation,
        child_id=meta["child_id"],
        family=meta["family"],
    )


def _encode_state(
    campaign: PreparedAtomisticCampaign, state: AtomisticCampaignState, /
) -> tuple[dict[str, Any], dict[str, object]]:
    arrays: dict[str, object] = {}
    banks = []
    for index, bank in enumerate(state.banks):
        prefix = f"bank/{index:04d}"
        for name in (
            "lane_stage",
            "budget",
            "consumed",
            "outcome",
            "reasons",
            "campaign_reasons",
            "keys",
        ):
            arrays[f"{prefix}/{name}"] = np.asarray(getattr(bank, name))
        sources, roles = bank.operation.capacity_levels()
        banks.append(
            {
                "bucket_id": bank.bucket_id,
                "stage": bank.stage,
                "family": bank.family,
                "lane_count": bank.lane_count,
                "levels": [list(sources), list(roles)],
                "native_lineage": list(bank.operation.hamiltonian_lineage),
                "occupant": list(bank.occupant),
                "retired": bank.retired,
                "held": [
                    None
                    if packet is None
                    else _encode_packet(f"{prefix}/held/{slot}", packet, arrays)
                    for slot, packet in enumerate(bank.held)
                ],
                "template": None
                if bank.template is None
                else _encode_packet(f"{prefix}/template", bank.template, arrays),
                "lanes": None
                if bank.lanes is None
                else pack_array_tree(
                    f"{prefix}/lanes", jax.device_get(bank.lanes), arrays
                ),
            }
        )
    plan_tasks = {task.task_id for task in campaign.plan.tasks}
    tasks = []
    for index, record in enumerate(state.tasks):
        prefix = f"task/{index:06d}"
        task = record.task
        tasks.append(
            {
                "task": task.record(),
                "spawned": None
                if task.task_id in plan_tasks
                else _encode_packet(f"{prefix}/initial", task.packet, arrays),
                "stage": record.stage,
                "status": record.status.value,
                "packet": None
                if record.packet is None
                else _encode_packet(f"{prefix}/packet", record.packet, arrays),
                "carried": None
                if record.carried is None
                else _encode_packet(f"{prefix}/carried", record.carried, arrays),
                "bucket": record.bucket,
                "lane": record.lane,
                "entries": list(record.entries),
                "consumed": record.consumed,
                "outcome": record.outcome,
                "reasons": record.reasons,
                "campaign_reasons": record.campaign_reasons,
                "parent": record.parent,
                "lineage": list(record.lineage),
            }
        )
    pending = []
    for index, output in enumerate(state.pending):
        prefix = f"pending/{index:06d}"
        for name, value in output.ports.items():
            arrays[f"{prefix}/{name}"] = value
        pending.append(output.manifest())
    arrays["root_key"] = state.root_key
    manifest: dict[str, Any] = {
        "format": _CHECKPOINT_FORMAT,
        "plan_id": campaign.plan_id,
        "campaign_id": campaign.plan.campaign_id,
        "round_index": state.round_index,
        "tasks": tasks,
        "queues": [list(queue) for queue in state.queues],
        "transfers": [[bucket, list(tasks_)] for bucket, tasks_ in state.transfers],
        "banks": banks,
        "routes": [list(route) for route in state.routes],
        "pending": pending,
        "receipts": [
            [receipt.semantic_key, receipt.sink_id, receipt.payload_id, receipt.location]
            for receipt in state.receipts
        ],
    }
    manifest["payload_id"] = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-envelope",
            "manifest": manifest,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return manifest, arrays


def _restored_operation(
    campaign: PreparedAtomisticCampaign,
    stage: str,
    family: str,
    bucket_id: str,
    levels: tuple[tuple[int, ...], tuple[int, ...]],
    lineage: tuple[str, ...],
    operations: tuple[AbstractAtomisticStageOperation, ...],
    /,
) -> AbstractAtomisticStageOperation:
    candidates = [
        operation
        for plan_stage in campaign.plan.stages
        for name, operation in plan_stage.operations.items()
        if name == family
    ]
    candidates.extend(operations)
    for operation in candidates:
        if operation.binding_id == bucket_id:
            return operation
    for operation in candidates:
        try:
            grown = operation.with_capacity_levels(levels).with_lineage(lineage)
        except (ValueError, RuntimeError, TypeError, IndexError):
            continue
        if grown.binding_id == bucket_id:
            return grown
    raise ValueError(
        f"No plan or supplied operation reproduces lane bank {bucket_id}; supply the "
        "rebound operation to resume()."
    )


def _decode_state(
    campaign: PreparedAtomisticCampaign,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    operations: tuple[AbstractAtomisticStageOperation, ...],
    /,
) -> AtomisticCampaignState:
    if manifest.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError("The file is not an atomistic campaign checkpoint.")
    if manifest["plan_id"] != campaign.plan_id:
        raise ValueError("The checkpoint belongs to another campaign plan.")
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key not in ("payload_id", "arrays")
    }
    expected = canonical_fingerprint(
        {
            "kind": "atomistic-campaign-envelope",
            "manifest": unsigned,
            "arrays": array_tree_fingerprint(dict(arrays)),
        }
    )
    if expected != manifest["payload_id"]:
        raise ValueError("The campaign checkpoint payload identity is corrupt.")
    restored = [
        _restored_operation(
            campaign,
            entry["stage"],
            entry["family"],
            entry["bucket_id"],
            (tuple(entry["levels"][0]), tuple(entry["levels"][1])),
            tuple(entry["native_lineage"]),
            operations,
        )
        for entry in manifest["banks"]
    ]
    dynamics_templates = _DynamicsStateTemplates(
        (
            *restored,
            *(
                operation
                for stage in campaign.plan.stages
                for operation in stage.operations.values()
            ),
            *operations,
        )
    )
    banks = []
    templates: dict[str, Any] = {}
    for index, (entry, operation) in enumerate(
        zip(manifest["banks"], restored, strict=True)
    ):
        prefix = f"bank/{index:04d}"
        template = (
            None
            if entry["template"] is None
            else _decode_packet(
                entry["template"], f"{prefix}/template", arrays, None, dynamics_templates
            )
        )
        lanes = None
        structure = None
        if entry["lanes"] is not None or template is not None:
            structure = operation.lane_template(int(entry["lane_count"]), template)
        if entry["lanes"] is not None:
            lanes = unpack_array_tree(entry["lanes"], arrays, structure)
        templates[entry["bucket_id"]] = structure
        banks.append(
            AtomisticLaneBank(
                bucket_id=entry["bucket_id"],
                stage=entry["stage"],
                family=entry["family"],
                operation=operation,
                lane_count=int(entry["lane_count"]),
                lanes=lanes,
                occupant=tuple(entry["occupant"]),
                lane_stage=np.asarray(arrays[f"{prefix}/lane_stage"], dtype=np.int32),
                budget=np.asarray(arrays[f"{prefix}/budget"], dtype=np.int32),
                consumed=np.asarray(arrays[f"{prefix}/consumed"], dtype=np.int32),
                outcome=np.asarray(arrays[f"{prefix}/outcome"], dtype=np.int32),
                reasons=np.asarray(arrays[f"{prefix}/reasons"], dtype=np.int32),
                campaign_reasons=np.asarray(
                    arrays[f"{prefix}/campaign_reasons"], dtype=np.int32
                ),
                keys=np.asarray(arrays[f"{prefix}/keys"], dtype=np.uint32),
                held=tuple(
                    None
                    if meta is None
                    else _decode_packet(
                        meta, f"{prefix}/held/{slot}", arrays, None, dynamics_templates
                    )
                    for slot, meta in enumerate(entry["held"])
                ),
                template=template,
                retired=bool(entry["retired"]),
            )
        )
    plan_tasks = {task.task_id: task for task in campaign.plan.tasks}
    records = []
    for index, entry in enumerate(manifest["tasks"]):
        prefix = f"task/{index:06d}"
        identity = entry["task"]
        if entry["spawned"] is None:
            task = plan_tasks[identity["task_id"]]
            if task.record() != identity:
                raise ValueError("A checkpoint task differs from the plan task.")
        else:
            task = AtomisticCampaignTask(
                identity["task_id"],
                identity["family"],
                _decode_packet(
                    entry["spawned"],
                    f"{prefix}/initial",
                    arrays,
                    None,
                    dynamics_templates,
                ),
                material_id=identity["material_id"],
                run_id=identity["run_id"],
                replica_id=identity["replica_id"],
                group_id=identity["group_id"],
                entry_stage=identity["entry_stage"],
                ordinal=int(identity["ordinal"]),
            )
            if task.record() != identity:
                raise ValueError("A spawned checkpoint task identity is corrupt.")
        packet = None
        if entry["packet"] is not None:
            structure = (
                None if entry["bucket"] is None else templates.get(entry["bucket"])
            )
            continuation = None if structure is None else _lane_slice_abstract(structure)
            packet = _decode_packet(
                entry["packet"],
                f"{prefix}/packet",
                arrays,
                continuation,
                dynamics_templates,
            )
        carried = (
            None
            if entry["carried"] is None
            else _decode_packet(
                entry["carried"], f"{prefix}/carried", arrays, None, dynamics_templates
            )
        )
        records.append(
            AtomisticTaskRecord(
                task=task,
                stage=int(entry["stage"]),
                status=AtomisticTaskStatus(entry["status"]),
                packet=packet,
                bucket=entry["bucket"],
                lane=int(entry["lane"]),
                entries=tuple(int(value) for value in entry["entries"]),
                consumed=int(entry["consumed"]),
                outcome=int(entry["outcome"]),
                reasons=int(entry["reasons"]),
                campaign_reasons=int(entry["campaign_reasons"]),
                parent=entry["parent"],
                lineage=tuple(entry["lineage"]),
                carried=carried,
            )
        )
    pending = []
    for index, meta in enumerate(manifest["pending"]):
        prefix = f"pending/{index:06d}"
        ports = {name: np.asarray(arrays[f"{prefix}/{name}"]) for name in meta["ports"]}
        pending.append(_output_from_manifest(meta, ports))
    return AtomisticCampaignState(
        plan_id=campaign.plan_id,
        phase=_Phase.COMMITTED.value,
        round_index=int(manifest["round_index"]),
        root_key=np.asarray(arrays["root_key"], dtype=np.uint32),
        tasks=tuple(records),
        queues=tuple(tuple(queue) for queue in manifest["queues"]),
        transfers=tuple(
            (bucket, tuple(tasks)) for bucket, tasks in manifest["transfers"]
        ),
        banks=tuple(banks),
        routes=tuple(
            (int(stage), family, bucket) for stage, family, bucket in manifest["routes"]
        ),
        pending=tuple(pending),
        receipts=tuple(
            AtomisticOutputReceipt(key, sink, payload, location)
            for key, sink, payload, location in manifest["receipts"]
        ),
        commit=None,
        evidence=AtomisticRoundEvidence(round_index=int(manifest["round_index"])),
    )


def _lane_slice_abstract(structure: Any, /) -> Any:
    """One lane of a bank structure (shape/dtype only), e.g. a continuation template."""
    lanes = getattr(structure, "state", None)
    source = structure if lanes is None else lanes
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(tuple(leaf.shape[1:]), leaf.dtype), source
    )


def write_atomistic_campaign_checkpoint(
    path: str | Path,
    campaign: PreparedAtomisticCampaign,
    state: AtomisticCampaignState,
    /,
) -> AtomisticCommitReceipt:
    """Write a committed campaign state as a standalone continuation envelope."""
    if not isinstance(campaign, PreparedAtomisticCampaign):
        raise TypeError("campaign must be a PreparedAtomisticCampaign.")
    campaign._require(state, _Phase.COMMITTED, _Phase.REFILLED)
    committed = _replace(state, phase=_Phase.COMMITTED.value, commit=None)
    manifest, arrays = _encode_state(campaign, committed)
    location = AtomisticCampaignStore(path).write(manifest, arrays)
    return AtomisticCommitReceipt(state.round_index, manifest["payload_id"], location)


def read_atomistic_campaign_checkpoint(
    path: str | Path,
    campaign: PreparedAtomisticCampaign,
    /,
    *,
    operations: Sequence[AbstractAtomisticStageOperation] = (),
) -> AtomisticCampaignState:
    """Read and verify a campaign continuation envelope (committed phase)."""
    payload = AtomisticCampaignStore(path).read()
    if payload is None:
        raise FileNotFoundError(str(path))
    manifest, arrays = payload
    state = _decode_state(campaign, manifest, arrays, tuple(operations))
    return _replace(
        state,
        commit=AtomisticCommitReceipt(
            state.round_index, manifest["payload_id"], str(path)
        ),
    )


__all__ = [
    "AbstractAtomisticCampaignSink",
    "AbstractAtomisticStageOperation",
    "AtomisticCampaignOutput",
    "AtomisticCampaignPlan",
    "AtomisticCampaignReason",
    "AtomisticCampaignState",
    "AtomisticCampaignStore",
    "AtomisticCampaignTask",
    "AtomisticCapacityGrowth",
    "AtomisticCapacityMode",
    "AtomisticCapacityTransfer",
    "AtomisticCommitReceipt",
    "AtomisticContinuationPolicy",
    "AtomisticDirectorySink",
    "AtomisticDynamicsContinuation",
    "AtomisticDynamicsLanes",
    "AtomisticDynamicsStageOperation",
    "AtomisticEntryPolicy",
    "AtomisticEvaluationKey",
    "AtomisticEvaluationRecord",
    "AtomisticGenerationStageOperation",
    "AtomisticLabelLanes",
    "AtomisticLabelStageOperation",
    "AtomisticLaneAdmission",
    "AtomisticLaneAdvance",
    "AtomisticLaneBank",
    "AtomisticOutputReceipt",
    "AtomisticReactionPathStageOperation",
    "AtomisticRelaxationStageOperation",
    "AtomisticRevisionPolicy",
    "AtomisticRoundEvidence",
    "AtomisticStageOperationKind",
    "AtomisticStageOutcome",
    "AtomisticStagePlan",
    "AtomisticStageRoute",
    "AtomisticStageRouting",
    "AtomisticStageSignature",
    "AtomisticTaskPacket",
    "AtomisticTaskRecord",
    "AtomisticTaskStatus",
    "PreparedAtomisticCampaign",
    "atomistic_hamiltonian_revision",
    "read_atomistic_campaign_checkpoint",
    "write_atomistic_campaign_checkpoint",
]
