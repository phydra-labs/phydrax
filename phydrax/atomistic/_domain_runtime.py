#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared execution of one distributed periodic atomistic domain.

`prepare_atomistic_domain(plan, model)` admits a layered learned potential
for an `AtomisticDomainPlan` and returns the `PreparedAtomisticDomainRuntime`
that owns ingress, evaluation, certificates, transactional rebuilds, model
rebinding, training partials and stage-transport packets.

Execution is partition-local. Every owner evaluates only its own receivers;
before each interaction the model's source payload of owned rows crosses the
strategy's column exchange once (``HALO``: deduplicated physical halo columns;
``GRAPH_PARTITION``: replicated owner-blocked columns) and reverse cotangents
return through the exact transpose once per interaction in reverse order
(`_feature_execution`). Owner energies, strain partials and statuses reduce
in the plan's declared owner order. An optional reciprocal epilogue
(`_reciprocal_distribution`) adds the distributed long-range share of a model
that exposes a global payload (MACE-LES; its latent Ewald definition must be
the distributed one). Domain terms (`_domain_terms`) execute on their own
roles under either strategy (polarization: HALO); Ewald/PME, LJ-PME and
FourierD3 terms add their term-owned reciprocal work distributed by
`_reciprocal_distribution`; terms consuming the
model's predicted fields (AIMNet2 ``CHARGES``) run inside the model readout,
so their cotangents return through the layered reverse sweep exactly once.

A declared nearest-neighbor selection (for example UMA ``max_neighbors``) is
applied owner-locally at the current geometry on the complete candidate set
of each owned receiver, with the single-device canonical order ``(measure,
source stable ID, lexicographic n)``; lexicographic order is translation
invariant, so the stored-coordinate image representation selects exactly the
single-device routes.

Lane-reference ownership (``owner_lanes``) runs the identical program as
named vmap lanes on one device; it is the numerical oracle, not distributed
hardware evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, final, Generic, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._data_plane import make_global_array_from_process_local_data
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import IdentityWords, NumericRevision
from .._strict import StrictModule
from .._trainable import combine_parameters, NonTrainableState, partition_parameters
from ..backends.distributed import JaxCollectiveProvider
from ..discretization.particle._image_neighborhood import ImageNeighborSelectionPolicy
from ..discretization.spatial._distributed_relations import (
    DistributedHaloPlan,
    DistributedMigrationEvidence,
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ..ein import contract
from ..typing import Bool, Float, Int32, Int64, PRNGKey, Scalar
from ._conditioning import (
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticObservable,
)
from ._domain_dynamics import (
    AtomisticDomainConstraintRows,
    export_domain_constraints,
    import_domain_constraints,
    rebuild_domain_constraints,
)
from ._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
from ._domain_state import (
    AtomisticDomainConstraintState,
    AtomisticDomainForceCache,
    AtomisticDomainState,
    AtomisticDomainStrategy,
    AtomisticDomainTopology,
    AtomisticDomainTopologyEvidence,
    prepare_domain_relations,
)
from ._domain_terms import (
    AtomisticDomainEwaldTerm,
    AtomisticDomainFourierD3Term,
    AtomisticDomainLennardJonesPMETerm,
    AtomisticDomainLennardJonesTerm,
    AtomisticDomainPolarizationTerm,
    domain_exception_topology,
    owner_ewald_exception_share,
    owner_ewald_real_share,
    owner_fourier_d3_share,
    owner_lennard_jones_pme_real_share,
    owner_lennard_jones_pme_reciprocal_share,
    owner_lennard_jones_share,
    owner_polarization_share,
    owner_reciprocal_share,
    OwnerExceptionRoute,
    OwnerTermResult,
    OwnerTermRoute,
    PreparedAtomisticDomainTerm,
    replicated_halo,
)
from ._dynamics import (
    AtomisticControllerState,
    AtomisticEnergyLedgerState,
    AtomisticStepRejectionReason,
)
from ._execution_binding import atomistic_case_address_words
from ._feature_execution import (
    AtomisticColumnExchange,
    AtomisticLayeredModel,
    HaloColumnExchange,
    owner_atom_energies,
    owner_layer_gradients,
    OwnerConditioning,
    OwnerLayerTopology,
)
from ._graph_partition import prepare_graph_partition_topology
from ._latent_electrostatics import PreparedLatentEwald
from ._neighborhood_bundle import AtomisticNeighborhoodRole
from ._potential import AbstractAtomisticPotential, atomistic_model_revision
from ._reciprocal_distribution import (
    prepare_owner_polarization,
    prepare_reciprocal_distribution,
    PreparedOwnerPolarization,
    PreparedReciprocalDistribution,
)


_REASON = AtomisticStepRejectionReason
_MODEL_TERM = "model"


def _reason(flag: AtomisticStepRejectionReason, failed: Array, /) -> Array:
    return jnp.where(failed, jnp.int32(int(flag)), jnp.int32(0))


def _select_tree[T](predicate: Array, candidate: T, previous: T, /) -> T:
    """Leafwise transactional selection of two equally structured trees."""

    def select(new: Array, old: Array) -> Array:
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            return jax.random.wrap_key_data(
                jnp.where(predicate, jax.random.key_data(new), jax.random.key_data(old)),
                impl=jax.random.key_impl(old),
            )
        return jnp.where(predicate, new, old)

    return jax.tree.map(select, candidate, previous)


def _typed_key(value: PRNGKey, /) -> Array:
    key = jnp.asarray(value)
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key) or key.shape:
        raise TypeError("rng_key must be one typed key from jax.random.key.")
    return key


def _nonempty(name: str, value: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _real(name: str, value: float, /) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


# ---------------------------------------------------------------------------
# Evidence and results
# ---------------------------------------------------------------------------


@final
class AtomisticDomainCertificate(StrictModule):
    """Topology reuse certificate of every role at the current geometry and cell.

    Per role ``r`` (arrays aligned with ``plan.roles``): ``cell_deformation[r]
    = max_{n in S_r} |n @ (H - H_ref)|`` over the role stencil and
    ``margin[r] = skin_r - (2 max|x - x_ref| + cell_deformation[r])``.
    ``stencil_complete[r]`` certifies that every image with ``|d| < cutoff_r``
    in the CURRENT cell lies in ``S_r`` for the current fractional extent of the
    stored coordinates. A static-cell plan additionally requires the
    reference cell. ``reasons`` is an `AtomisticStepRejectionReason` bitmask.
    """

    __strict_contract__ = True

    certified: Bool[Scalar]
    maximum_displacement: Float[Scalar]
    cell_deformation: Array
    margin: Array
    displacement_certified: Array
    stencil_complete: Array
    topology_successful: Array
    owners_current: Bool[Scalar]
    cell_admissible: Bool[Scalar]
    cell_static: Bool[Scalar]
    reasons: Int32[Scalar]


@final
class AtomisticDomainExecutionStatus(StrictModule):
    """Fail-closed numerical, relation, model, selection and per-term status.

    Per-term arrays follow ``plan.terms``: capacity (packets, patches, halos,
    exception directory and routes) and scientific status (including an
    exception naming an absent atom) are separate; ``term_image_ambiguity``
    marks a minimum-image role with two images (or a self image) inside its
    cutoff. ``identity_bound`` is the traced equality of the state's plan and
    model-revision identity words with the runtime's (a state of another
    material or revision is refused numerically, as ``POTENTIAL``).
    """

    __strict_contract__ = True

    identity_bound: Bool[Scalar]
    relation_successful: Bool[Scalar]
    model_successful: Bool[Scalar]
    finite: Bool[Scalar]
    capped_receivers: Int32[Scalar] | Int64[Scalar]
    maximum_selected_degree: Int32[Scalar]
    term_capacity_successful: Array
    term_scientific_successful: Array
    term_image_ambiguity: Array


@final
class AtomisticDomainEvaluation(StrictModule):
    """Distributed energy, forces, strain derivative and stress.

    ``atom_energies`` and ``forces`` are owner-blocked rows of the state
    layout. ``owner_energies[o]`` is owner ``o``'s local atom-energy sum plus
    its share of the global and domain-term energies; ``energy`` is their sum
    in owner order. ``local_energy``/``global_energy`` are the layered model's
    atom and global parts and ``term_energies`` the plan's domain terms (in
    ``plan.terms`` order); ``atom_energies`` are the model's only.
    ``atom_energy_complete`` is the structural fact that they decompose the
    energy: false when the model declares a global payload or a system
    energy, or when any domain term contributes.
    ``strain_derivative`` is ``dE/d strain`` at fixed fractional coordinates
    (row cell ``H' = H @ F.T``), the owner-ordered sum of every owner's edge,
    global-term and domain-term partials; ``stress = sym(dE/d strain) / V(H)``
    exists only for fully periodic three-dimensional cells. Failed execution
    poisons every numerical output with NaN.
    """

    __strict_contract__ = True

    energy: Float[Scalar]
    local_energy: Float[Scalar]
    global_energy: Float[Scalar]
    term_energies: Array
    owner_energies: Array
    atom_energies: Array
    forces: Array
    strain_derivative: Array
    stress: Array
    system_contexts: PyTree[Array]
    certificate: AtomisticDomainCertificate
    status: AtomisticDomainExecutionStatus
    reasons: Int32[Scalar]
    successful: Bool[Scalar]
    strain_available: bool = eqx.field(static=True)
    atom_energy_complete: Bool[Scalar]

    def force_cache(self, step: ArrayLike, /) -> AtomisticDomainForceCache:
        """This evaluation as the force cache of ``step``."""
        return AtomisticDomainForceCache(
            forces=self.forces,
            energy=self.energy,
            owner_energies=self.owner_energies,
            strain_derivative=self.strain_derivative,
            stress=self.stress,
            strain_available=jnp.asarray(self.strain_available) & self.successful,
            step=jnp.asarray(step, dtype=jnp.int32),
            successful=self.successful,
        )


@final
class AtomisticDomainLossGradient(StrictModule):
    """Energy/force/stress loss and its exact PARAMETER-lane gradient."""

    __strict_contract__ = True

    loss: Float[Scalar]
    parameter_gradient: PyTree[Array]
    evaluation: AtomisticDomainEvaluation
    successful: Bool[Scalar]


@final
class AtomisticDomainTransition(StrictModule):
    """Result of one global wrap, migration and multi-role topology transaction."""

    __strict_contract__ = True

    state: AtomisticDomainState
    committed: Bool[Scalar]
    migration: DistributedMigrationEvidence
    topologies: tuple[AtomisticDomainTopologyEvidence, ...]
    constraints_committed: Bool[Scalar]
    cell_admissible: Bool[Scalar]
    reasons: Int32[Scalar]


class AtomisticDomainOwnedRows(NamedTuple):
    """Owner-blocked model-role inputs of an owner region."""

    positions: Array
    species: Array
    active: Array
    receivers: Array
    columns: Array
    shifts: Array
    valid: Array
    source_ids: Array
    relation_leaves: tuple[Array, ...]
    send_slots: Array | None
    send_valid: Array | None
    route_identity: tuple[Array, ...]


class AtomisticDomainReplicatedInputs(NamedTuple):
    """Replicated inputs of an owner region.

    ``halo`` is the halo plan with its arrays removed (only its static owner
    geometry is used inside a region); ``admitted`` conjoins the role's
    topology evidence, owner epochs, the current certificate and the state's
    identity binding to the runtime (`identity_bound`).
    """

    cell_vectors: Array
    conditioning: AtomisticConditioning | None
    halo: DistributedHaloPlan | None
    admitted: Array
    capacity_successful: Array
    key: Array


_OwnedT = TypeVar("_OwnedT")
_ReplicatedT = TypeVar("_ReplicatedT")
_TermOwnedT = TypeVar("_TermOwnedT")
_TermReplicatedT = TypeVar("_TermReplicatedT")


@final
class AtomisticDomainOwnerView(
    StrictModule, Generic[_OwnedT, _ReplicatedT, _TermOwnedT, _TermReplicatedT]
):
    """Model-role inputs for region-level execution inside a caller's owner map.

    Build with `PreparedAtomisticDomainRuntime.owner_view` (owned
    `AtomisticDomainOwnedRows`, replicated `AtomisticDomainReplicatedInputs`
    and `AtomisticDomainTermView` inputs) and map with `partition_specs`, the
    same tree with ``P(owner_axis)`` for owned rows and ``P()`` for replicated
    leaves. ``terms`` carries the domain-term inputs when the runtime has
    terms, so region-level predictions include them.
    """

    owned: _OwnedT
    replicated: _ReplicatedT
    terms: AtomisticDomainTermView[_TermOwnedT, _TermReplicatedT] | None
    relation_tree: PyTreeDef = eqx.field(static=True)
    column_count: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    role_id: str = eqx.field(static=True)

    def partition_specs(
        self,
    ) -> AtomisticDomainOwnerView[
        PartitionSpec, PartitionSpec, PartitionSpec, PartitionSpec
    ]:
        """This view's owner-map specs (a tree prefix of the view)."""
        return AtomisticDomainOwnerView(
            owned=PartitionSpec(self.axis_name),
            replicated=PartitionSpec(),
            terms=None if self.terms is None else self.terms.partition_specs(),
            relation_tree=self.relation_tree,
            column_count=self.column_count,
            axis_name=self.axis_name,
            role_id=self.role_id,
        )


@final
class AtomisticDomainOwnerPredictions(StrictModule):
    """Region-level predictions of one owner (replicated scalars, owned rows)."""

    energy: Array
    forces: Array | None
    strain_derivative: Array | None
    stress: Array | None
    atom_mask: Array
    successful: Array
    capacity_failure: Array


class AtomisticDomainExceptionRows(NamedTuple):
    """Owner-blocked exception routes of one term (`domain_exception_topology`)."""

    receivers: Array
    columns: Array
    rows: Array
    valid: Array
    send_slots: Array
    send_valid: Array


class AtomisticDomainExceptionInputs(NamedTuple):
    """Replicated exception evidence and array-free partner halo of one term."""

    halo: DistributedHaloPlan
    capacity_successful: Array
    missing: Array


class AtomisticDomainTermRows(NamedTuple):
    """Owner-blocked inputs of the domain terms (every role's routes).

    ``exceptions`` follow ``runtime.terms`` (``None`` for a term without
    listed pairs).
    """

    positions: Array
    active: Array
    stable_ids: Array
    image_counts: Array
    fields: dict[str, Array]
    routes: tuple[OwnerTermRoute, ...]
    send_slots: tuple[Array | None, ...]
    send_valid: tuple[Array | None, ...]
    exceptions: tuple[AtomisticDomainExceptionRows | None, ...]


class AtomisticDomainTermInputs(NamedTuple):
    """Replicated inputs of the domain terms (array-free halo geometry)."""

    cell_vectors: Array
    halos: tuple[DistributedHaloPlan | None, ...]
    exceptions: tuple[AtomisticDomainExceptionInputs | None, ...]


@final
class AtomisticDomainTermView(StrictModule, Generic[_OwnedT, _ReplicatedT]):
    """Domain-term inputs (owned `AtomisticDomainTermRows`, replicated
    `AtomisticDomainTermInputs`) mapped over owners with :meth:`partition_specs`."""

    owned: _OwnedT
    replicated: _ReplicatedT
    axis_name: str = eqx.field(static=True)

    def partition_specs(self) -> AtomisticDomainTermView[PartitionSpec, PartitionSpec]:
        """This view's owner-map specs (a tree prefix of the view)."""
        return AtomisticDomainTermView(
            owned=PartitionSpec(self.axis_name),
            replicated=PartitionSpec(),
            axis_name=self.axis_name,
        )


type _TermInputView = AtomisticDomainTermView[
    AtomisticDomainTermRows, AtomisticDomainTermInputs
]
type _OwnerInputView = AtomisticDomainOwnerView[
    AtomisticDomainOwnedRows,
    AtomisticDomainReplicatedInputs,
    AtomisticDomainTermRows,
    AtomisticDomainTermInputs,
]


class AtomisticDomainForceRecord(NamedTuple):
    """Replicated header of a transported force cache."""

    energy: np.ndarray
    strain_derivative: np.ndarray
    stress: np.ndarray
    strain_available: np.ndarray
    step: np.ndarray
    successful: np.ndarray


@final
class AtomisticDomainPacket(StrictModule, NonTrainableState):
    """Per-process host fragment of a domain state for stage transport.

    ``rows`` are this process's addressable owner blocks (fixed capacity
    ``addressable owners * local_capacity``) of every per-atom continuation
    leaf; ``header`` holds the replicated continuation (cell, payload,
    controllers, ledger, key data, step, time, force record, conditioning,
    constraint rows). Fragments of several source processes are concatenated
    by `PreparedAtomisticDomainRuntime.from_packet`. A packet is a host
    transport record (NumPy leaves), never a compiled argument: its exact
    ``model_revision_id``, ``run_id`` and ``header_id`` digests stay host
    strings and are refused exactly at `from_packet`.
    """

    rows: Mapping[str, np.ndarray]
    atom_payload: Mapping[str, np.ndarray]
    cell_vectors: np.ndarray
    replicated_payload: Mapping[str, np.ndarray]
    controllers: AtomisticControllerState | None
    ledger: AtomisticEnergyLedgerState | None
    conditioning: AtomisticConditioning | None
    constraints: AtomisticDomainConstraintRows | None
    rng_key_data: np.ndarray
    case_identity_words: np.ndarray
    step_index: np.ndarray
    time: np.ndarray
    force_record: AtomisticDomainForceRecord
    key_implementation: str = eqx.field(static=True)
    logical_count: int = eqx.field(static=True)
    role_ids: tuple[str, ...] = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    header_id: str = eqx.field(static=True)

    def without_rows(self) -> AtomisticDomainPacket:
        """Header-only fragment for a destination process that receives no rows.

        Constraint rows keep their statics and replicated residuals in one
        all-inactive row block, so `from_packet` still merges and validates
        them.
        """
        empty = {name: value[:0] for name, value in self.rows.items()}
        payload = {name: value[:0] for name, value in self.atom_payload.items()}
        constraints = self.constraints
        if constraints is not None:
            block = constraints.row_capacity
            constraints = AtomisticDomainConstraintRows(
                row_ids=np.zeros((block,), np.int64),
                atoms=np.full((block, 2), -1, np.int64),
                components=np.full((block,), -1, np.int64),
                targets=np.zeros((block,), constraints.targets.dtype),
                lagrange=np.zeros((block,), constraints.lagrange.dtype),
                active=np.zeros((block,), np.bool_),
                position_residual=constraints.position_residual,
                velocity_residual=constraints.velocity_residual,
                topology_epoch=constraints.topology_epoch,
                row_capacity=block,
                row_count=constraints.row_count,
                halo_capacity=constraints.halo_capacity,
                directory_capacity=constraints.directory_capacity,
                migration_capacity=constraints.migration_capacity,
                plan_id=constraints.plan_id,
            )
        return AtomisticDomainPacket(
            rows=empty,
            atom_payload=payload,
            cell_vectors=self.cell_vectors,
            replicated_payload=self.replicated_payload,
            controllers=self.controllers,
            ledger=self.ledger,
            conditioning=self.conditioning,
            constraints=constraints,
            rng_key_data=self.rng_key_data,
            case_identity_words=self.case_identity_words,
            step_index=self.step_index,
            time=self.time,
            force_record=self.force_record,
            key_implementation=self.key_implementation,
            logical_count=self.logical_count,
            role_ids=self.role_ids,
            model_revision_id=self.model_revision_id,
            run_id=self.run_id,
            header_id=self.header_id,
        )


# ---------------------------------------------------------------------------
# Topology discovery (halo strategy)
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _halo_edges(
    plan: AtomisticDomainPlan,
    role: AtomisticDomainRolePlan,
    positions: Array,
    active: Array,
    cell_vectors: Array,
    inverse: Array,
) -> tuple[Array, ...]:
    """Alias exchange in the current cell plus owner-dense candidate filtering.

    Candidate sources of one owner are its own rows (translation zero) and the
    received periodic image aliases. The filter is an owner-local dense bound
    of ``local_capacity x (local_capacity + owner_count * alias_capacity)``
    candidates; it never forms global pairs. Edges are receiver-major in
    candidate order and keep every candidate with ``|d| <= cutoff + skin``.
    """
    ownership = plan.ownership
    axis = ownership.axis_name
    local = ownership.local_capacity
    edges = role.edge_capacity
    radius = role.search_radius

    def body(
        rows: Array, alive: Array, vectors: Array, inv: Array, shifts: Array
    ) -> tuple[Array, ...]:
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        aliases = plan.partition.local_alias_exchange(
            rows, alive, vectors, inv, shifts, radius, axis, role.alias_capacity
        )
        candidates = jnp.concatenate((rows, aliases.positions), axis=0)
        candidate_valid = jnp.concatenate((alive, aliases.valid), axis=0)
        count = candidates.shape[0]
        separation = rows[:, None, :] - candidates[None, :, :]
        squared = jnp.sum(separation * separation, axis=-1)
        same_row = jnp.arange(local)[:, None] == jnp.arange(count)[None, :]
        selected = (
            alive[:, None]
            & candidate_valid[None, :]
            & ~same_row
            & (squared <= radius * radius)
        )
        total = jnp.sum(selected, dtype=jnp.int32)
        flat = jnp.nonzero(selected.reshape((-1,)), size=edges, fill_value=0)[0]
        valid = jnp.arange(edges, dtype=jnp.int32) < total
        receiver = (flat // count).astype(jnp.int32)
        candidate = (flat % count).astype(jnp.int32)
        owned = candidate < local
        alias = jnp.clip(candidate - local, 0, aliases.slots.shape[0] - 1)
        route_owner = jnp.where(owned, me, aliases.owners[alias])
        route_slot = jnp.where(owned, candidate, aliases.slots[alias])
        translation = jnp.where(owned[:, None], 0, aliases.translations[alias])
        statistics = jnp.stack(
            (
                aliases.maximum_load,
                total,
                jnp.max(jnp.sum(selected, axis=1, dtype=jnp.int32)),
            )
        ).astype(jnp.int32)
        return (
            jnp.where(valid, receiver, 0),
            jnp.where(valid, route_owner, 0),
            jnp.where(valid, route_slot, 0),
            jnp.where(valid[:, None], -translation, 0).astype(jnp.int32),
            valid,
            statistics[None],
        )

    owner = PartitionSpec(axis)
    replicated = PartitionSpec()
    return ownership.map(
        body,
        (owner, owner, replicated, replicated, replicated),
        (owner,) * 6,
    )(positions, active, cell_vectors, inverse, role.stencil.shifts)


@eqx.filter_jit
def _column_ids(halo: DistributedHaloPlan, stable_ids: Array, columns: Array) -> Array:
    """Stable ID of every edge's source column (owner-local columns)."""
    owners = halo.ownership.owner_count
    gathered = halo.gather(stable_ids).reshape((owners, halo.column_count))
    per_owner = columns.reshape((owners, -1))
    return jnp.take_along_axis(gathered, per_owner, axis=1).reshape((-1,))


def _stencil_status(
    plan: AtomisticDomainPlan, role: AtomisticDomainRolePlan, cell_vectors: Array
) -> tuple[Array, Array]:
    """Margin of the role stencil for ``cutoff + skin`` with wrapped endpoints."""
    cell = plan.partition.cell
    margin = cell.image_stencil_margin(
        role.stencil,
        cell_vectors,
        role.search_radius,
        jnp.ones((cell.rank,), dtype=cell_vectors.dtype),
    )
    return margin.astype(cell_vectors.dtype), jnp.all(margin > 0.0)


def _halo_topology(
    plan: AtomisticDomainPlan,
    role: AtomisticDomainRolePlan,
    layout: DistributedPointLayout,
    positions: Array,
    cell_vectors: Array,
    epoch: Array,
) -> AtomisticDomainTopology:
    """Discover, route, and prepare one halo-strategy role epoch."""
    cell = plan.partition.cell
    inverse, inverse_ok = cell.inverse_for_vectors_with_status(cell_vectors)
    dtype = positions.dtype
    receivers, route_owners, route_slots, shifts, valid, statistics = _halo_edges(
        plan,
        role,
        positions,
        layout.active,
        cell_vectors.astype(dtype),
        inverse.astype(dtype),
    )
    halo = DistributedHaloPlan(
        plan.ownership,
        route_owners,
        route_slots,
        valid,
        halo_capacity=role.halo_capacity,
    )
    edge_valid = valid & halo.route_valid
    columns = halo.route_columns
    source_ids = jnp.where(
        edge_valid,
        _column_ids(halo, layout.stable_ids, columns).astype(jnp.int64),
        jnp.int64(0),
    )
    epoch_ = jnp.asarray(epoch, dtype=jnp.int32)
    leaves, tree = prepare_domain_relations(
        plan.ownership,
        role.streaming,
        role.relation_owner_id,
        halo.column_count,
        receivers,
        columns,
        edge_valid,
        layout.active,
        epoch_,
    )
    owner_epochs = layout.owner_epochs.astype(jnp.int32)
    owners_current = jnp.all(owner_epochs == owner_epochs[0])
    finite = jnp.all(jnp.where(layout.active[:, None], jnp.isfinite(positions), True))
    margin, stencil_complete = _stencil_status(plan, role, cell_vectors)
    maximum_alias = jnp.max(statistics[:, 0])
    maximum_edges = jnp.max(statistics[:, 1])
    alias_capacity = jnp.asarray(role.alias_capacity, dtype=jnp.int32)
    edge_capacity = jnp.asarray(role.edge_capacity, dtype=jnp.int32)
    capacity_ok = (
        (maximum_alias <= alias_capacity)
        & (maximum_edges <= edge_capacity)
        & halo.evidence.successful
    )
    cell_ok = jnp.asarray(inverse_ok, dtype=jnp.bool_)
    scientific_ok = owners_current & finite & cell_ok & stencil_complete
    evidence = AtomisticDomainTopologyEvidence(
        successful=capacity_ok & scientific_ok,
        capacity_successful=capacity_ok,
        scientific_successful=scientific_ok,
        owners_current=owners_current,
        positions_finite=finite,
        cell_admissible=cell_ok,
        stencil_complete=stencil_complete,
        stencil_margin=margin,
        maximum_search_load=maximum_alias,
        search_capacity=alias_capacity,
        maximum_edge_count=maximum_edges,
        edge_capacity=edge_capacity,
        maximum_degree=jnp.max(statistics[:, 2]),
        halo_successful=halo.evidence.successful,
        refused_routes=halo.evidence.refused_routes,
        maximum_halo_load=halo.evidence.maximum_halo_load,
        halo_capacity=halo.evidence.halo_capacity,
        replicated_bytes=jnp.zeros((), dtype=jnp.int64),
        replicated_capacity_bytes=jnp.zeros((), dtype=jnp.int64),
    )
    return AtomisticDomainTopology(
        route_owners=route_owners,
        route_slots=route_slots,
        receivers=receivers,
        shifts=shifts,
        valid=edge_valid,
        columns=columns,
        source_ids=source_ids,
        relation_leaves=leaves,
        reference_positions=positions,
        reference_cell_vectors=jnp.asarray(cell_vectors, dtype=dtype),
        owner_epochs=owner_epochs,
        epoch=epoch_,
        evidence=evidence,
        halo=halo,
        column_count=halo.column_count,
        relation_tree=tree,
        streaming_plan_id=role.streaming.plan_id,
        role_id=role.role_id,
        strategy=AtomisticDomainStrategy.HALO,
    )


def _build_topologies(
    plan: AtomisticDomainPlan,
    layout: DistributedPointLayout,
    positions: Array,
    cell_vectors: Array,
    epochs: tuple[Array, ...],
) -> tuple[AtomisticDomainTopology, ...]:
    topologies = []
    for role, epoch in zip(plan.roles, epochs, strict=True):
        if plan.strategy is AtomisticDomainStrategy.HALO:
            topology = _halo_topology(plan, role, layout, positions, cell_vectors, epoch)
        else:
            topology = prepare_graph_partition_topology(
                plan, role, layout, positions, cell_vectors, epoch
            )
        if topology.role_id != role.role_id or topology.strategy is not plan.strategy:
            raise RuntimeError("A strategy produced a topology of another role.")
        topologies.append(topology)
    return tuple(topologies)


def _topology_reasons(evidence: AtomisticDomainTopologyEvidence, /) -> Array:
    source = (evidence.maximum_search_load > evidence.search_capacity) | (
        evidence.replicated_bytes > evidence.replicated_capacity_bytes
    )
    route = ~evidence.capacity_successful & ~source
    return (
        _reason(_REASON.SOURCE_CAPACITY, source)
        | _reason(_REASON.ROUTE_CAPACITY, route)
        | _reason(_REASON.DOMAIN, ~evidence.owners_current | ~evidence.stencil_complete)
        | _reason(_REASON.CELL, ~evidence.cell_admissible)
        | _reason(_REASON.NONFINITE, ~evidence.positions_finite)
        | _reason(
            _REASON.DOMAIN,
            ~evidence.scientific_successful
            & evidence.owners_current
            & evidence.stencil_complete
            & evidence.cell_admissible
            & evidence.positions_finite,
        )
    )


def _case_words(case_identity: str | ArrayLike | None, run_id: str, /) -> Array:
    """Eight uint32 case-address words (`atomistic_case_address_words`).

    A string is the system's declared case identity (for parity with a
    single-device graph, its prepared system identity); already encoded words
    are adopted unchanged; ``None`` addresses the run.
    """
    if case_identity is None or isinstance(case_identity, str):
        identity = (
            run_id if case_identity is None else _nonempty("case_identity", case_identity)
        )
        return atomistic_case_address_words((identity,))[0]
    words = np.asarray(case_identity)
    if words.shape != (8,) or words.dtype != np.uint32:
        raise ValueError("Encoded case identity words must be eight uint32 values.")
    return jnp.asarray(words)


@eqx.filter_jit
def _route_identity(
    plan: AtomisticDomainPlan,
    topology: AtomisticDomainTopology,
    layout: DistributedPointLayout,
    image_counts: Array,
    case_words: Array,
) -> tuple[Array, ...]:
    """Owner-blocked semantic route addresses of one role epoch.

    Exactly ``AtomisticGraphInputs.route_identity_fields``: the case words,
    the source and receiver stable particle IDs, and the material image shift
    ``n + c_source - c_receiver`` (int64), zero on invalid edges. Source image
    counts travel through the strategy's own column table, so the addresses
    are independent of ranks, slots and layouts.
    """
    owners = plan.ownership.owner_count
    local = plan.ownership.local_capacity
    edges = topology.receivers.shape[0] // owners
    owner_index = jnp.repeat(jnp.arange(owners, dtype=jnp.int32), edges)
    receiver_rows = owner_index * local + topology.receivers
    receiver_ids = layout.stable_ids[receiver_rows].astype(jnp.int64)
    receiver_counts = image_counts[receiver_rows]
    rank = image_counts.shape[1]
    if topology.halo is not None:
        table = topology.halo.gather(image_counts).reshape(
            (owners, topology.column_count, rank)
        )
        source_counts = jnp.take_along_axis(
            table, topology.columns.reshape((owners, edges))[..., None], axis=1
        ).reshape((-1, rank))
    else:
        source_counts = image_counts[topology.columns]
    material = (
        topology.shifts.astype(jnp.int64)
        + source_counts.astype(jnp.int64)
        - receiver_counts.astype(jnp.int64)
    )
    valid = topology.valid
    count = topology.receivers.shape[0]
    words = tuple(
        jnp.where(valid, jnp.broadcast_to(case_words[column], (count,)), jnp.uint32(0))
        for column in range(case_words.shape[0])
    )
    return (
        *words,
        jnp.where(valid, topology.source_ids, jnp.int64(0)),
        jnp.where(valid, receiver_ids, jnp.int64(0)),
        *(jnp.where(valid, material[:, axis], jnp.int64(0)) for axis in range(rank)),
    )


# ---------------------------------------------------------------------------
# Owner-region execution
# ---------------------------------------------------------------------------


def _owner_selection(
    policy: ImageNeighborSelectionPolicy,
    receivers: Array,
    displacement: Array,
    valid: Array,
    source_ids: Array,
    shifts: Array,
    receiver_count: int,
) -> tuple[Array, Array, Array]:
    """Declared nearest-neighbor selection of one owner's complete candidates.

    Mirrors `PreparedImageNeighborSelection.select`: eligibility, measure,
    canonical order ``(receiver, measure, source stable ID, lexicographic n)``,
    strict or degeneracy-band retention. Every candidate of an owned receiver
    is present, so the selection is the single-device selection.
    """
    routes = receivers.shape[0]
    squared = jax.lax.stop_gradient(jnp.sum(displacement * displacement, axis=-1))
    bound = policy.radius * policy.radius
    eligible = valid & (
        squared <= bound if policy.radius_boundary == "closed" else squared < bound
    )
    if policy.coincidence_threshold is not None:
        eligible = eligible & (squared > policy.coincidence_threshold)
    measure = jnp.where(
        eligible,
        squared if policy.measure == "squared-distance" else jnp.sqrt(squared),
        0.0,
    )
    masked_shifts = jnp.where(eligible[:, None], shifts, 0)
    keys = tuple(masked_shifts[:, axis] for axis in reversed(range(shifts.shape[1])))
    keys = keys + (
        jnp.where(eligible, source_ids, 0),
        measure,
        jnp.where(eligible, receivers, 0),
        (~eligible).astype(jnp.int32),
    )
    order = jax.lax.stop_gradient(jnp.lexsort(keys)).astype(jnp.int32)
    sorted_receivers = receivers[order]
    sorted_eligible = eligible[order]
    sorted_measure = measure[order]
    index = jnp.arange(routes, dtype=jnp.int32)
    head = sorted_eligible & jnp.concatenate(
        (
            jnp.ones((min(routes, 1),), dtype=jnp.bool_),
            sorted_receivers[1:] != sorted_receivers[:-1],
        )
    )
    start = jax.lax.cummax(jnp.where(head, index, 0))
    eligible_count = (
        jnp.zeros((receiver_count,), dtype=jnp.int32)
        .at[jnp.where(eligible, receivers, 0)]
        .add(eligible.astype(jnp.int32))
    )
    neighbors = policy.maximum_neighbors
    capped = eligible_count[sorted_receivers] > neighbors
    if policy.strict:
        kept = (index - start) < neighbors
    else:
        anchor = sorted_measure[jnp.clip(start + neighbors, 0, routes - 1)]
        kept = sorted_measure <= anchor + policy.degeneracy_tolerance
    selected = (
        jnp.zeros((routes,), dtype=jnp.bool_)
        .at[order]
        .set(sorted_eligible & (~capped | kept))
    )
    selected_count = (
        jnp.zeros((receiver_count,), dtype=jnp.int32)
        .at[jnp.where(selected, receivers, 0)]
        .add(selected.astype(jnp.int32))
    )
    return (
        selected,
        jnp.sum(eligible_count > selected_count, dtype=jnp.int32),
        jnp.max(selected_count),
    )


class _OwnerOutputs(NamedTuple):
    atom_energies: Array
    gradient: Array
    local_energy: Array
    global_share: Array
    strain: Array
    finite: Array
    relation_successful: Array
    model_successful: Array
    capped_receivers: Array
    selected_degree: Array
    system_contexts: PyTree[Array]
    parameter_gradient: PyTree[Array] | None
    global_share_declared: Array
    field_terms: _TermStack | None


def _column_exchange(
    plan: AtomisticDomainPlan,
    halo: DistributedHaloPlan | None,
    send_slots: Array | None,
    send_valid: Array | None,
    /,
) -> AtomisticColumnExchange:
    """One owner's column exchange of a role under the plan's strategy.

    The halo strategy binds the role's array-free halo plan to this owner's
    send routes; the graph-partition strategy replicates owner columns.
    """
    if plan.strategy is AtomisticDomainStrategy.HALO:
        if halo is None or send_slots is None or send_valid is None:
            raise ValueError(
                "A halo-strategy role view needs its halo plan and send routes."
            )
        return HaloColumnExchange(
            halo, send_slots, send_valid, plan.message_capacity_bytes
        )
    graph_partition = plan.graph_partition
    if graph_partition is None:
        raise ValueError(
            "The graph-partition strategy requires an AtomisticGraphPartitionPlan."
        )
    return graph_partition.exchange(plan.ownership)


def _owner_exchange(
    runtime: PreparedAtomisticDomainRuntime, view: _OwnerInputView
) -> AtomisticColumnExchange:
    owned = view.owned
    return _column_exchange(
        runtime.plan, view.replicated.halo, owned.send_slots, owned.send_valid
    )


def _owner_topology(
    runtime: PreparedAtomisticDomainRuntime,
    view: _OwnerInputView,
    exchange: AtomisticColumnExchange,
) -> tuple[OwnerLayerTopology, Array, Array]:
    """One owner's executable model-role topology, with declared selection."""
    owned = view.owned
    relation, receiver_reduction, source_reduction = jax.tree.unflatten(
        view.relation_tree, [leaf[0] for leaf in owned.relation_leaves]
    )
    role = runtime.model_role
    valid = owned.valid
    capped = jnp.zeros((), dtype=jnp.int32)
    degree = jnp.zeros((), dtype=jnp.int32)
    if role.selection is not None:
        coordinates = jax.lax.stop_gradient(owned.positions)
        columns = exchange.gather(coordinates)
        vectors = view.replicated.cell_vectors.astype(coordinates.dtype)
        displacement = (
            coordinates[owned.receivers]
            - columns[owned.columns]
            + contract("ea,ad->ed", owned.shifts.astype(coordinates.dtype), vectors)
        )
        selected, capped, degree = _owner_selection(
            role.selection,
            owned.receivers,
            displacement,
            valid,
            owned.source_ids,
            owned.shifts,
            owned.positions.shape[0],
        )
        valid = valid & selected
    topology = OwnerLayerTopology(
        relation=relation,
        receiver_reduction=receiver_reduction,
        source_reduction=source_reduction,
        receivers=owned.receivers,
        columns=owned.columns,
        shifts=owned.shifts,
        valid=valid,
        route_identity=owned.route_identity,
    )
    return topology, capped, degree


def _region_outputs(
    runtime: PreparedAtomisticDomainRuntime,
    model: AtomisticLayeredModel,
    view: _OwnerInputView,
    *,
    gradients: bool,
    parameters: bool,
) -> _OwnerOutputs:
    """Execute the model role of one owner region (no leading owner axis)."""
    plan = runtime.plan
    axis = plan.ownership.axis_name
    owned = view.owned
    replicated = view.replicated
    exchange = _owner_exchange(runtime, view)
    topology, capped, degree = _owner_topology(runtime, view, exchange)
    dtype = owned.positions.dtype
    vectors = replicated.cell_vectors.astype(dtype)
    context = OwnerConditioning(
        replicated.conditioning,
        jnp.zeros(owned.active.shape, dtype=jnp.int32),
        replicated.key,
    )
    global_term = (
        None if runtime.reciprocal is None else runtime.reciprocal.global_term(axis)
    )
    field_term = None
    if runtime.field_term_positions:
        terms = view.terms
        if terms is None:
            raise ValueError("Field-dependent domain terms need the view's term inputs.")

        def field_term(
            positions: Array,
            fields: Mapping[AtomisticObservable, Array],
            atom_mask: Array,
            cell_vectors: Array,
        ) -> tuple[Array, Array, _TermStack]:
            del atom_mask  # the term view's owned activity is the same mask
            results = _owner_term_results(
                runtime, terms, positions, cell_vectors, fields, field_terms=True
            )
            stack = _stack_terms(results, positions.dtype)
            return jnp.sum(stack.shares), jnp.all(stack.scientific), stack

    cutoff = runtime.model_role.cutoff
    if gradients:
        result = owner_layer_gradients(
            model,
            exchange,
            topology,
            owned.positions,
            owned.species,
            owned.active,
            vectors,
            cutoff,
            context=context,
            reduction_mode=plan.reduction.mode,
            axis_name=axis,
            accumulation=plan.reduction.mode,
            parameters=parameters,
            global_term=global_term,
            field_term=field_term,
        )
        gradient = jnp.where(owned.active[:, None], result.gradient, 0.0)
        strain = result.strain_gradient
        parameter_gradient = result.parameter_gradient
    else:
        result = owner_atom_energies(
            model,
            exchange,
            topology,
            owned.positions,
            owned.species,
            owned.active,
            vectors,
            cutoff,
            context=context,
            reduction_mode=plan.reduction.mode,
            axis_name=axis,
            global_term=global_term,
            field_term=field_term,
        )
        gradient = jnp.zeros_like(owned.positions)
        strain = jnp.zeros((3, 3), dtype=dtype)
        parameter_gradient = None
    atom_energies = jnp.where(owned.active, result.atom_energies, 0.0)
    local_energy = plan.reduction.ordered_sum(atom_energies)
    global_share = jnp.asarray(result.global_energy, dtype=local_energy.dtype)
    finite = (
        jnp.all(jnp.isfinite(atom_energies))
        & jnp.isfinite(global_share)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(strain))
    )
    return _OwnerOutputs(
        atom_energies=atom_energies,
        gradient=gradient,
        local_energy=local_energy,
        global_share=global_share,
        strain=strain,
        finite=finite,
        relation_successful=jnp.asarray(result.relation_successful, dtype=jnp.bool_),
        model_successful=jnp.asarray(result.model_successful, dtype=jnp.bool_),
        capped_receivers=capped,
        selected_degree=degree,
        system_contexts=result.system_contexts,
        parameter_gradient=parameter_gradient,
        global_share_declared=jnp.asarray(result.global_share_declared, dtype=jnp.bool_),
        field_terms=result.field_aux,
    )


def _stress_scale(runtime: PreparedAtomisticDomainRuntime, cell_vectors: Array) -> Array:
    """``1 / V(H)`` of the current cell (stress requires a full periodic cell)."""
    volume, _ = runtime.plan.partition.cell.measure_with_vectors(cell_vectors)
    return 1.0 / volume


def owner_predictions(
    runtime: PreparedAtomisticDomainRuntime,
    parameters: PyTree[Array],
    view: AtomisticDomainOwnerView[
        AtomisticDomainOwnedRows,
        AtomisticDomainReplicatedInputs,
        AtomisticDomainTermRows,
        AtomisticDomainTermInputs,
    ],
    /,
    *,
    forces: bool,
    stress: bool,
    model_state: PyTree[Array] | None = None,
) -> AtomisticDomainOwnerPredictions:
    """Region-level predictions inside a caller-owned owner map.

    ``parameters`` is the PARAMETER lane; it is combined with ``model_state``
    (default: the runtime's MODEL_STATE lane) and the runtime's FIXED lane.
    Energy, strain derivative and stress are replicated owner-ordered sums of
    the model and every domain term (field-dependent terms inside the model's
    readout, fixed-input terms beside it); forces are this owner's rows.
    ``successful`` conjoins every owner's numerical, relation, model and term
    status with the view's admission and capacities; ``capacity_failure``
    marks a topology, packet, halo or exception capacity refusal.
    """
    if stress and not runtime.stress_available:
        raise ValueError("Stress requires a fully periodic three-dimensional cell.")
    model = runtime.bind_parameters(parameters, model_state)
    plan = runtime.plan
    axis = plan.ownership.axis_name
    gradients = bool(forces) or bool(stress)
    outputs = _region_outputs(runtime, model, view, gradients=gradients, parameters=False)
    reduction = plan.reduction
    provider = JaxCollectiveProvider(axis)
    owner_energy = outputs.local_energy + outputs.global_share
    gradient = outputs.gradient
    strain_partial = outputs.strain
    local = outputs.finite & outputs.relation_successful & outputs.model_successful
    capacity = jnp.asarray(view.replicated.capacity_successful, dtype=jnp.bool_)
    scientific = jnp.asarray(True)
    if outputs.field_terms is not None:
        capacity = capacity & jnp.all(outputs.field_terms.capacity)
        scientific = scientific & jnp.all(outputs.field_terms.scientific)
    if view.terms is not None and runtime.fixed_term_positions:
        fixed = _owner_term_outputs(runtime, view.terms, gradients)
        owner_energy = owner_energy + jnp.sum(fixed.stack.shares)
        gradient = gradient + fixed.gradient
        strain_partial = strain_partial + fixed.strain
        local = (
            local
            & jnp.all(jnp.isfinite(fixed.stack.shares))
            & jnp.all(jnp.isfinite(fixed.gradient))
            & jnp.all(jnp.isfinite(fixed.strain))
        )
        capacity = capacity & jnp.all(fixed.stack.capacity)
        scientific = scientific & jnp.all(fixed.stack.scientific)
    energy = reduction.owner_sum(owner_energy, axis)
    successful = (
        (provider.minimum(local.astype(jnp.int32)) == 1)
        & view.replicated.admitted
        & capacity
        & scientific
    )
    strain = None
    stress_value = None
    if gradients:
        strain = reduction.owner_sum(strain_partial, axis)
        if stress:
            symmetric = 0.5 * (strain + strain.T)
            stress_value = symmetric * _stress_scale(
                runtime, view.replicated.cell_vectors
            )
    return AtomisticDomainOwnerPredictions(
        energy=energy,
        forces=-gradient if forces else None,
        strain_derivative=strain,
        stress=stress_value,
        atom_mask=view.owned.active,
        successful=successful,
        capacity_failure=~capacity,
    )


def owner_parameter_partials(
    runtime: PreparedAtomisticDomainRuntime,
    parameters: PyTree[Array],
    view: AtomisticDomainOwnerView[
        AtomisticDomainOwnedRows,
        AtomisticDomainReplicatedInputs,
        AtomisticDomainTermRows,
        AtomisticDomainTermInputs,
    ],
    /,
    *,
    energy_cotangent: ArrayLike,
    force_cotangent: ArrayLike | None = None,
    stress_cotangent: ArrayLike | None = None,
    model_state: PyTree[Array] | None = None,
) -> PyTree[Array]:
    """This owner's exactly-once share of a linearized global functional.

    Each owner differentiates its local surrogate ``e (E_local + E_global
    share) + sum_owned f . F + <sym(s) / V, strain partial>`` through its own
    explicit reverse sweep, including every column gather and exact
    transpose; the owner-axis sum of the shares is the PARAMETER-lane gradient
    of ``e E + sum f . F + s : stress``. No owner-count scaling is applied.
    Field-dependent domain terms (predicted charges) are part of the model
    readout and differentiate with it; fixed-input domain terms do not depend
    on the parameters and contribute no partial.
    """
    owned = view.owned
    dtype = owned.positions.dtype
    energy_scale = jnp.asarray(energy_cotangent, dtype=dtype)
    if energy_scale.shape:
        raise ValueError("energy_cotangent must be a scalar.")
    force_scale = (
        jnp.zeros_like(owned.positions)
        if force_cotangent is None
        else jnp.asarray(force_cotangent, dtype=dtype)
    )
    if force_scale.shape != owned.positions.shape:
        raise ValueError("force_cotangent must match this owner's rows.")
    force_scale = jnp.where(owned.active[:, None], force_scale, 0.0)
    strain_scale = jnp.zeros((3, 3), dtype=dtype)
    if stress_cotangent is not None:
        if not runtime.stress_available:
            raise ValueError("Stress requires a fully periodic three-dimensional cell.")
        target = jnp.asarray(stress_cotangent, dtype=dtype)
        if target.shape != (3, 3):
            raise ValueError("stress_cotangent must have shape (3, 3).")
        strain_scale = (
            0.5
            * (target + target.T)
            * _stress_scale(runtime, view.replicated.cell_vectors)
        )
    state_lane = runtime.model_state_lane(model_state)

    def surrogate(lane: PyTree[Array]) -> Array:
        model = combine_parameters(lane, state_lane, runtime.fixed_lane)
        outputs = _region_outputs(runtime, model, view, gradients=True, parameters=False)
        return (
            energy_scale * (outputs.local_energy + outputs.global_share)
            + jnp.sum(force_scale * -outputs.gradient)
            + jnp.sum(strain_scale * outputs.strain)
        )

    return jax.grad(surrogate)(parameters)


# ---------------------------------------------------------------------------
# Prepared runtime
# ---------------------------------------------------------------------------


def _require_layered(model: AbstractAtomisticPotential, /) -> None:
    for name in (
        "interaction_count",
        "init_features",
        "prepare_sources",
        "update_receivers",
        "atom_energies",
        "layer_context",
        "global_payload",
        "system_energy",
        "owner_fields",
    ):
        if not hasattr(model, name):
            raise TypeError(
                f"Domain execution requires a layered learned potential; the model "
                f"lacks {name!r}."
            )


def _require_conditioning(
    plan: AtomisticDomainPlan, conditioning: AtomisticConditioning | None, /
) -> AtomisticConditioning | None:
    """Admit case-level conditioning of one system bound to owner-local rows."""
    if conditioning is None:
        return None
    if not isinstance(conditioning, AtomisticConditioning):
        raise TypeError("conditioning must be AtomisticConditioning or None.")
    if conditioning.case_count != 1:
        raise ValueError("A domain holds one periodic system (case_count == 1).")
    if conditioning.atom_capacity != plan.ownership.local_capacity:
        raise ValueError(
            "Domain conditioning binds owner-local rows: atom_capacity must equal "
            "the ownership local_capacity."
        )
    if any(
        field.level is AtomisticConditionLevel.ATOM and value is not None
        for field, value in zip(
            conditioning.spec.fields, conditioning.values, strict=True
        )
    ):
        raise ValueError(
            "Per-atom conditioning rows are not owner-blocked; carry them as "
            "atom_payload and admit only case-level conditions."
        )
    return conditioning


def _host_rows(array: Array, /) -> np.ndarray:
    """This process's addressable owner blocks of an owner-blocked array."""
    shards = sorted(array.addressable_shards, key=lambda shard: shard.index[0].start or 0)
    seen: list[np.ndarray] = []
    starts: set[int] = set()
    for shard in shards:
        start = shard.index[0].start or 0
        if start in starts:
            continue
        starts.add(start)
        seen.append(np.asarray(shard.data))
    return np.concatenate(seen, axis=0)


def _host_tree(tree: Any, /) -> Any:
    return jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), tree)


@final
class PreparedAtomisticDomainRuntime(StrictModule):
    """A domain plan bound to an optional layered model and its domain terms.

    The model is held as its three role lanes; ``training_binding`` removes the
    PARAMETER and MODEL_STATE lanes for training kernels that supply them per
    call. The model revision is computed once at preparation (a term-only
    runtime binds the identity of its terms); states carry the revision they
    were primed for and stale revisions are refused, never re-hashed per call.
    ``terms`` are the plan's domain terms with their prepared distributed work.

    PyTree layout: ``model_revision_identity`` and ``runtime_identity`` are the
    exact canonical digests carried as dynamic `IdentityWords`
    (``model_revision_id`` and ``runtime_id`` decode them on the host), and
    every other static is declared layout (model role index, training
    binding, model-state requirement). Runtimes of every material and
    parameter revision of one layout therefore share one treedef, and the
    compiled kernels (`_evaluate_regions`, `_parameter_regions`,
    `_term_regions`) compile once per layout; compiled evaluation compares
    the state's identity words with the runtime's and folds the result into
    its status.

    ``scientific_identity`` (host ``scientific_id``), dynamic and computed
    once at preparation, is the layout-independent digest of the domain
    physics: the layered model revision (``None`` without a model; a
    term-only ``model_revision_id`` digests its terms' placement-bearing
    ``term_id``), every term's ``scientific_id``, each role's neighborhood
    requirement (cutoff and selection), the domain reciprocal definition and
    the reference cell with its dynamic-cell flag. Ownership, partition,
    strategy, reduction order, stencils, skins, streaming, reciprocal
    placement and every capacity only place or order this computation and
    stay bound by ``runtime_identity``. ``hamiltonian_revision_identity``
    (host ``hamiltonian_revision_id``), also computed once, is the numeric
    revision of that Hamiltonian: a `NumericRevision` bound to
    ``atomistic-domain-hamiltonian:<scientific_id>`` over every inexact leaf
    of the canonical model and of the canonical domain-term definitions
    (prepared terms, their systems, exception weights and reciprocal
    definitions); no ownership, topology or placement array enters it.
    """

    plan: AtomisticDomainPlan
    parameter_lane: PyTree[Array]
    state_lane: PyTree[Array]
    fixed_lane: PyTree[Array]
    reciprocal: PreparedReciprocalDistribution | None
    terms: tuple[PreparedAtomisticDomainTerm, ...]
    model_revision_identity: IdentityWords
    runtime_identity: IdentityWords
    scientific_identity: IdentityWords
    hamiltonian_revision_identity: IdentityWords
    model_role_index: int | None = eqx.field(static=True)
    training_bound: bool = eqx.field(static=True)
    model_state_required: bool = eqx.field(static=True)

    @property
    def model_revision_id(self) -> str:
        """Host decoding of the bound model (or term) revision digest."""
        return self.model_revision_identity.fingerprint()

    @property
    def runtime_id(self) -> str:
        """Host decoding of the canonical prepared-runtime digest."""
        return self.runtime_identity.fingerprint()

    @property
    def scientific_id(self) -> str:
        """Host decoding of the layout-independent domain-physics digest."""
        return self.scientific_identity.fingerprint()

    @property
    def hamiltonian_revision_id(self) -> str:
        """Host decoding of the layout-independent Hamiltonian numeric revision."""
        return self.hamiltonian_revision_identity.fingerprint()

    def conditioning_id(self, state: AtomisticDomainState, /) -> str:
        """Layout-independent digest of a state's evaluated conditioning.

        Host boundary (this process's replica of the replicated conditioning).
        Binds the conditioning specification, every admitted case-level value
        and the default-use flags (``None`` without conditioning). The
        owner-local ``atom_capacity`` is layout and is excluded; per-atom
        conditions are refused at admission.
        """
        self._require_plan_host(state)
        conditioning = state.conditioning

        def replica(leaf: Array) -> np.ndarray:
            return np.asarray(leaf.addressable_data(0))

        payload: dict[str, Any] | None = None
        if conditioning is not None:
            payload = {
                "spec": conditioning.spec.spec_id,
                "values": [
                    None if value is None else array_tree_fingerprint(replica(value))
                    for value in conditioning.values
                ],
                "defaults_used": array_tree_fingerprint(
                    replica(conditioning.defaults_used)
                ),
            }
        return canonical_fingerprint(
            {"kind": "atomistic-domain-evaluated-conditioning", "conditioning": payload}
        )

    # -- model binding ------------------------------------------------------

    @property
    def model_role(self) -> AtomisticDomainRolePlan:
        return self.plan.roles[self._require_model()]

    @property
    def stress_available(self) -> bool:
        cell = self.plan.partition.cell
        return cell.fully_periodic and cell.rank == 3

    @property
    def field_term_positions(self) -> tuple[int, ...]:
        """Positions in ``terms`` of terms consuming the model's predicted fields."""
        return tuple(
            position
            for position, prepared in enumerate(self.terms)
            if prepared.term.model_fields
        )

    @property
    def fixed_term_positions(self) -> tuple[int, ...]:
        """Positions in ``terms`` of terms with fixed per-atom inputs."""
        return tuple(
            position
            for position, prepared in enumerate(self.terms)
            if not prepared.term.model_fields
        )

    @property
    def model(self) -> AtomisticLayeredModel:
        if self.training_bound:
            raise ValueError("A training binding holds no PARAMETER/MODEL_STATE lanes.")
        return combine_parameters(self.parameter_lane, self.state_lane, self.fixed_lane)

    def model_state_lane(self, model_state: PyTree[Array] | None, /) -> PyTree[Array]:
        if model_state is not None:
            return model_state
        if self.training_bound and self.model_state_required:
            raise ValueError("This training binding requires an explicit model_state.")
        return self.state_lane

    def bind_parameters(
        self, parameters: PyTree[Array], model_state: PyTree[Array] | None = None, /
    ) -> AtomisticLayeredModel:
        """Combine a PARAMETER lane (and MODEL_STATE lane) with the FIXED lane."""
        return combine_parameters(
            parameters, self.model_state_lane(model_state), self.fixed_lane
        )

    def training_binding(self) -> PreparedAtomisticDomainRuntime:
        """The same runtime with PARAMETER and MODEL_STATE lanes removed."""
        self._require_model()
        return PreparedAtomisticDomainRuntime(
            plan=self.plan,
            parameter_lane=jax.tree.map(lambda _: None, self.parameter_lane),
            state_lane=jax.tree.map(lambda _: None, self.state_lane),
            fixed_lane=self.fixed_lane,
            reciprocal=self.reciprocal,
            terms=self.terms,
            model_revision_identity=self.model_revision_identity,
            runtime_identity=self.runtime_identity,
            scientific_identity=self.scientific_identity,
            hamiltonian_revision_identity=self.hamiltonian_revision_identity,
            model_role_index=self.model_role_index,
            training_bound=True,
            model_state_required=self.model_state_required,
        )

    def _require_model(self) -> int:
        """The model role index; a term-only runtime has no model."""
        index = self.model_role_index
        if index is None:
            raise ValueError("This domain runtime executes terms only; it has no model.")
        return index

    # -- state admission ----------------------------------------------------

    def _require_state(self, state: AtomisticDomainState, /) -> None:
        """Static layout admission (traceable): type and role alignment."""
        if not isinstance(state, AtomisticDomainState):
            raise TypeError("state must be an AtomisticDomainState.")
        if tuple(topology.role_id for topology in state.topologies) != tuple(
            role.role_id for role in self.plan.roles
        ):
            raise ValueError("The state topologies do not align with the plan roles.")

    def _require_plan_host(self, state: AtomisticDomainState, /) -> None:
        """Exact host refusal of a state of another domain plan (transactions)."""
        self._require_state(state)
        if state.plan_id != self.plan.plan_id:
            raise ValueError("The domain state belongs to another plan.")

    def identity_bound(self, state: AtomisticDomainState, /) -> Array:
        """Traced equality of the state's plan and revision words with this runtime's.

        A state of another material, term or model revision is refused
        numerically by evaluation (rebind it with `rebind_model` on a runtime
        prepared for that model).
        """
        self._require_state(state)
        return state.plan_identity.equals(
            self.plan.plan_identity
        ) & state.model_revision_identity.equals(self.model_revision_identity)

    # -- ingress ------------------------------------------------------------

    def _cell(self, cell_vectors: ArrayLike | None, dtype: np.dtype, /) -> np.ndarray:
        reference = np.asarray(self.plan.partition.cell.vectors, dtype=dtype)
        if cell_vectors is None:
            return reference
        matrix = np.asarray(cell_vectors, dtype=dtype)
        if matrix.shape != reference.shape or not np.all(np.isfinite(matrix)):
            raise ValueError(f"cell_vectors must be finite with shape {reference.shape}.")
        if not self.plan.dynamic_cell and not np.array_equal(matrix, reference):
            raise ValueError("A static-cell domain plan admits only its reference cell.")
        return matrix

    def _replicated(
        self,
        replicated_payload: Mapping[str, ArrayLike] | None,
        controllers: AtomisticControllerState | None,
        ledger: AtomisticEnergyLedgerState | None,
        conditioning: AtomisticConditioning | None,
    ) -> tuple[dict[str, Array], AtomisticConditioning | None]:
        if controllers is not None and not isinstance(
            controllers, AtomisticControllerState
        ):
            raise TypeError("controllers must be AtomisticControllerState or None.")
        if ledger is not None and not isinstance(ledger, AtomisticEnergyLedgerState):
            raise TypeError("ledger must be AtomisticEnergyLedgerState or None.")
        payload = {
            str(name): jnp.asarray(value)
            for name, value in sorted((replicated_payload or {}).items())
        }
        return payload, _require_conditioning(self.plan, conditioning)

    def _assemble(
        self,
        layout: DistributedPointLayout,
        rows: Mapping[str, Array],
        atom_payload: Mapping[str, Array],
        cell_vectors: Array,
        *,
        rng_key: PRNGKey,
        replicated_payload: dict[str, Array],
        controllers: AtomisticControllerState | None,
        ledger: AtomisticEnergyLedgerState | None,
        conditioning: AtomisticConditioning | None,
        constraints: AtomisticDomainConstraintRows | None,
        force_record: AtomisticDomainForceRecord | None,
        step_index: int,
        time: float,
        run_id: str | None,
        case_identity: str | ArrayLike | None,
    ) -> AtomisticDomainState:
        """Build every role topology and the state; refuse failed ingress."""
        plan = self.plan
        positions = rows["positions"]
        dtype = positions.dtype
        for prepared_term in self.terms:
            missing = set(prepared_term.term.field_names) - set(atom_payload)
            if missing:
                raise ValueError(
                    f"Domain term {prepared_term.term.term_key!r} needs its per-atom "
                    f"payload {sorted(missing)}."
                )
        zero_epoch = tuple(jnp.zeros((), dtype=jnp.int32) for _ in plan.roles)
        topologies = _build_topologies(plan, layout, positions, cell_vectors, zero_epoch)
        for role, topology in zip(plan.roles, topologies, strict=True):
            if not bool(topology.evidence.successful):
                raise ValueError(
                    f"The initial {role.requirement.label} domain topology was refused: "
                    f"{jax.device_get(topology.evidence)}."
                )
        constraint_state: AtomisticDomainConstraintState | None = None
        if constraints is not None:
            if not isinstance(constraints, AtomisticDomainConstraintRows):
                raise TypeError(
                    "constraints must be AtomisticDomainConstraintRows or None."
                )
            transition = import_domain_constraints(constraints, layout)
            if not bool(transition.committed):
                raise ValueError(
                    "The domain constraint rows were refused: "
                    f"{jax.device_get(transition.evidence)}."
                )
            constraint_state = transition.state
        step = jnp.asarray(step_index, dtype=jnp.int32)
        owners = plan.ownership.owner_count
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        if force_record is None:
            force = AtomisticDomainForceCache(
                forces=rows["forces"],
                energy=nan,
                owner_energies=jnp.full((owners,), jnp.nan, dtype=dtype),
                strain_derivative=jnp.full((3, 3), jnp.nan, dtype=dtype),
                stress=jnp.full((3, 3), jnp.nan, dtype=dtype),
                strain_available=jnp.asarray(False),
                step=step - 1,
                successful=jnp.asarray(False),
            )
        else:
            force = AtomisticDomainForceCache(
                forces=rows["forces"],
                energy=jnp.asarray(force_record.energy, dtype=dtype),
                owner_energies=jnp.full((owners,), jnp.nan, dtype=dtype),
                strain_derivative=jnp.asarray(
                    force_record.strain_derivative, dtype=dtype
                ),
                stress=jnp.asarray(force_record.stress, dtype=dtype),
                strain_available=jnp.asarray(
                    force_record.strain_available, dtype=jnp.bool_
                ),
                step=jnp.asarray(force_record.step, dtype=jnp.int32),
                successful=jnp.asarray(force_record.successful, dtype=jnp.bool_),
            )
        # A continued run names its exact canonical run digest (the source
        # state's ``run_id``); a fresh run derives it from the plan and revision.
        run = (
            canonical_fingerprint(
                {
                    "kind": "atomistic-domain-run",
                    "plan": plan.plan_id,
                    "model": self.model_revision_id,
                }
            )
            if run_id is None
            else _nonempty("run_id", run_id)
        )
        return AtomisticDomainState(
            layout=layout,
            positions=positions,
            momenta=rows["momenta"],
            masses=rows["masses"],
            species=rows["species"],
            image_counts=rows["image_counts"],
            cell_vectors=jnp.asarray(cell_vectors, dtype=dtype),
            atom_payload=dict(atom_payload),
            replicated_payload=replicated_payload,
            controllers=controllers,
            ledger=ledger,
            constraints=constraint_state,
            force=force,
            rng_key=_typed_key(rng_key),
            step_index=step,
            time=jnp.asarray(_real("time", time), dtype=dtype),
            topologies=topologies,
            conditioning=conditioning,
            case_identity_words=_case_words(case_identity, run),
            plan_identity=plan.plan_identity,
            model_revision_identity=self.model_revision_identity,
            run_identity=IdentityWords.from_fingerprint(run),
        )

    def ingress_global(
        self,
        positions: ArrayLike,
        species: ArrayLike,
        /,
        *,
        rng_key: PRNGKey,
        momenta: ArrayLike | None = None,
        masses: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        atom_payload: Mapping[str, ArrayLike] | None = None,
        replicated_payload: Mapping[str, ArrayLike] | None = None,
        conditioning: AtomisticConditioning | None = None,
        controllers: AtomisticControllerState | None = None,
        ledger: AtomisticEnergyLedgerState | None = None,
        constraints: AtomisticDomainConstraintRows | None = None,
        step_index: int = 0,
        time: float = 0.0,
        case_identity: str | ArrayLike | None = None,
        run_id: str | None = None,
    ) -> AtomisticDomainState:
        """Host ingress: wrap in the current cell, own, distribute, build topologies.

        ``species`` are the model's native species indices; every per-atom
        input is in logical atom order. Ingress refuses a failed first
        topology, constraint import or singular cell with its evidence.
        ``run_id`` continues an existing run and must be its exact canonical
        run digest (a state's ``run_id``); by default the run digest derives
        from the plan and the bound revision.
        """
        plan = self.plan
        cell = plan.partition.cell
        coordinates = np.asarray(positions)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("positions must have shape (atoms, 3).")
        if not np.issubdtype(coordinates.dtype, np.floating):
            raise TypeError("positions must have a floating dtype.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("positions must be finite.")
        count = coordinates.shape[0]
        dtype = coordinates.dtype
        species_ = np.asarray(species)
        if species_.shape != (count,) or not np.issubdtype(species_.dtype, np.integer):
            raise ValueError("species must hold one integer species index per atom.")
        momentum = (
            np.zeros_like(coordinates) if momenta is None else np.asarray(momenta, dtype)
        )
        mass = (
            np.ones((count,), dtype=dtype)
            if masses is None
            else np.asarray(masses, dtype)
        )
        if momentum.shape != coordinates.shape or mass.shape != (count,):
            raise ValueError("momenta and masses must match the atom rows.")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("masses must be finite and positive.")
        images_in = (
            np.zeros((count, cell.rank), dtype=np.int32)
            if image_counts is None
            else np.asarray(image_counts)
        )
        if images_in.shape != (count, cell.rank) or not np.issubdtype(
            images_in.dtype, np.integer
        ):
            raise ValueError("image_counts must hold one integer image per lattice axis.")
        payload = {}
        for name, value in sorted((atom_payload or {}).items()):
            array = np.asarray(value)
            if not array.shape or array.shape[0] != count:
                raise ValueError(f"atom_payload[{name!r}] must lead with the atom rows.")
            payload[str(name)] = array
        system = plan.system
        if system is not None:
            # Domain terms address the prepared system: logical row i is its
            # i-th active particle and stable IDs are its particle IDs.
            active_rows = np.flatnonzero(np.asarray(system.active_mask))
            if active_rows.size != count:
                raise ValueError(
                    "Domain terms need one logical row per active system particle."
                )
            system_ids = np.asarray(system.plan.particle_ids)[active_rows].astype(
                np.int64
            )
            if stable_ids is None:
                stable_ids = system_ids
            elif not np.array_equal(np.asarray(stable_ids), system_ids):
                raise ValueError(
                    "Domain terms need the system particle IDs as stable IDs."
                )
            for term in plan.terms:
                for name, value in term.atom_fields().items():
                    if name in payload:
                        raise ValueError(f"atom_payload key {name!r} is reserved.")
                    payload[name] = value
        replicated, conditioning_ = self._replicated(
            replicated_payload, controllers, ledger, conditioning
        )
        vectors = self._cell(cell_vectors, dtype)
        fractional, solved = cell.fractional_with_vectors_with_status(
            jnp.asarray(coordinates), jnp.asarray(vectors)
        )
        if not bool(solved):
            raise ValueError("The current cell is singular.")
        fractional = np.asarray(fractional)
        wraps = np.where(np.asarray(cell.periodic_axes), np.floor(fractional), 0.0)
        wrapped = fractional - wraps
        stored = coordinates - wraps.astype(dtype) @ vectors
        owners = np.asarray(plan.partition.owners(jnp.asarray(wrapped)))
        layout = DistributedPointLayout.from_global(
            plan.ownership, wrapped, owners, stable_ids=stable_ids, epoch=0
        )
        rows = {
            "positions": layout.distribute(stored),
            "momenta": layout.distribute(momentum),
            "masses": layout.distribute(mass),
            "species": layout.distribute(species_.astype(np.int32)),
            "image_counts": layout.distribute(
                (images_in + wraps.astype(np.int64)).astype(np.int32)
            ),
            "forces": layout.distribute(np.zeros_like(coordinates)),
        }
        return self._assemble(
            layout,
            rows,
            {name: layout.distribute(value) for name, value in payload.items()},
            jnp.asarray(vectors),
            rng_key=rng_key,
            replicated_payload=replicated,
            controllers=controllers,
            ledger=ledger,
            conditioning=conditioning_,
            constraints=constraints,
            force_record=None,
            step_index=step_index,
            time=time,
            run_id=run_id,
            case_identity=case_identity,
        )

    def ingress_process_local(
        self,
        positions: ArrayLike,
        species: ArrayLike,
        /,
        *,
        rng_key: PRNGKey,
        active: ArrayLike,
        stable_ids: ArrayLike,
        logical_indices: ArrayLike,
        logical_count: int,
        momenta: ArrayLike | None = None,
        masses: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        forces: ArrayLike | None = None,
        force_record: AtomisticDomainForceRecord | None = None,
        cell_vectors: ArrayLike | None = None,
        atom_payload: Mapping[str, ArrayLike] | None = None,
        replicated_payload: Mapping[str, ArrayLike] | None = None,
        conditioning: AtomisticConditioning | None = None,
        controllers: AtomisticControllerState | None = None,
        ledger: AtomisticEnergyLedgerState | None = None,
        constraints: AtomisticDomainConstraintRows | None = None,
        step_index: int = 0,
        time: float = 0.0,
        case_identity: str | ArrayLike | None = None,
        run_id: str | None = None,
    ) -> AtomisticDomainState:
        """Process-local ingress without global host materialization.

        Per-atom inputs are this process's rows of its addressable owner blocks
        (``addressable owners * local_capacity`` rows, owner-major), in any
        owner assignment; ``active`` marks occupied rows. Rows are assembled
        with ``make_global_array_from_process_local_data``, adopted with
        `DistributedPointLayout.from_blocks`, wrapped in the current cell and
        migrated on device to their fractional owners with every continuation
        leaf (packet capacity ``local_capacity`` per owner pair, so only a
        receive overflow can refuse). Eager audits (the only host
        synchronizations, all scalar): stable-ID and logical-row uniqueness in
        ``from_blocks``, the cell solve status, the migration commit, every
        role's topology evidence and the constraint import.
        """
        plan = self.plan
        ownership = plan.ownership
        cell = plan.partition.cell
        total = ownership.total_capacity
        coordinates = np.asarray(positions)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("positions must have shape (local rows, 3).")
        if not np.issubdtype(coordinates.dtype, np.floating):
            raise TypeError("positions must have a floating dtype.")
        dtype = coordinates.dtype
        rows_here = coordinates.shape[0]
        if rows_here % ownership.local_capacity:
            raise ValueError("Process-local rows must be whole owner blocks.")
        mask = np.asarray(active, dtype=np.bool_)
        if mask.shape != (rows_here,):
            raise ValueError("active must hold one boolean per local row.")
        if not np.all(np.isfinite(coordinates[mask])):
            raise ValueError("Active positions must be finite.")

        def local(name: str, value: ArrayLike | None, default: np.ndarray) -> np.ndarray:
            array = default if value is None else np.asarray(value)
            if array.shape[:1] != (rows_here,):
                raise ValueError(f"{name} must lead with the local rows.")
            return np.where(
                mask.reshape(mask.shape + (1,) * (array.ndim - 1)), array, 0
            ).astype(array.dtype)

        species_ = local("species", species, np.zeros((rows_here,), np.int32))
        if not np.issubdtype(species_.dtype, np.integer):
            raise TypeError("species must be integer model species indices.")
        mass = local("masses", masses, np.ones((rows_here,), dtype))
        if np.any(mass[mask] <= 0.0) or not np.all(np.isfinite(mass)):
            raise ValueError("Active masses must be finite and positive.")
        host_rows = {
            "positions": np.where(mask[:, None], coordinates, 0).astype(dtype),
            "momenta": local("momenta", momenta, np.zeros_like(coordinates)).astype(
                dtype
            ),
            "masses": mass.astype(dtype),
            "species": species_.astype(np.int32),
            "image_counts": local(
                "image_counts",
                image_counts,
                np.zeros((rows_here, cell.rank), np.int32),
            ).astype(np.int32),
            "forces": local("forces", forces, np.zeros_like(coordinates)).astype(dtype),
        }
        if (forces is None) != (force_record is None):
            raise ValueError("forces and force_record are supplied together.")
        host_payload = {
            str(name): local(f"atom_payload[{name!r}]", value, np.zeros(0))
            for name, value in sorted((atom_payload or {}).items())
        }
        ids = np.asarray(stable_ids)
        logical = np.asarray(logical_indices)
        if ids.shape != (rows_here,) or logical.shape != (rows_here,):
            raise ValueError(
                "stable_ids and logical_indices must hold one entry per row."
            )
        replicated, conditioning_ = self._replicated(
            replicated_payload, controllers, ledger, conditioning
        )
        vectors = jnp.asarray(self._cell(cell_vectors, dtype))

        def assemble(array: np.ndarray) -> Array:
            return make_global_array_from_process_local_data(
                ownership.sharding(array.ndim),
                array,
                global_shape=(total,) + array.shape[1:],
            )

        device_rows = {name: assemble(value) for name, value in host_rows.items()}
        device_payload = {name: assemble(value) for name, value in host_payload.items()}
        device_active = assemble(mask)
        wrapped, wraps, solved = _wrap_rows(
            cell, device_rows["positions"], device_active, vectors
        )
        if not bool(solved):
            raise ValueError("The current cell is singular.")
        initial = DistributedPointLayout.from_blocks(
            ownership,
            wrapped,
            active=device_active,
            stable_ids=assemble(ids.astype(np.int64)),
            logical_indices=assemble(logical.astype(np.int32)),
            logical_count=logical_count,
            epoch=0,
        )
        if not bool(initial.stable_ids_unique):
            raise ValueError("Stable global IDs must be unique.")
        destinations = jnp.where(device_active, plan.partition.owners(wrapped), 0)
        migrated = initial.migrate(
            destinations,
            packet_capacity=ownership.local_capacity,
            points=wrapped,
            payload={"rows": device_rows, "wraps": wraps, "atom_payload": device_payload},
        )
        if not bool(migrated.evidence.committed):
            raise ValueError(
                "Process-local ingress migration was refused: "
                f"{jax.device_get(migrated.evidence)}."
            )
        moved = migrated.payload
        layout = migrated.layout
        rows = dict(moved["rows"])
        rows["positions"], rows["image_counts"] = _apply_wraps(
            layout, rows["positions"], rows["image_counts"], moved["wraps"], vectors
        )
        return self._assemble(
            layout,
            rows,
            moved["atom_payload"],
            vectors,
            rng_key=rng_key,
            replicated_payload=replicated,
            controllers=controllers,
            ledger=ledger,
            conditioning=conditioning_,
            constraints=constraints,
            force_record=force_record,
            step_index=step_index,
            time=time,
            run_id=run_id,
            case_identity=case_identity,
        )

    # -- dynamics -----------------------------------------------------------

    def with_dynamics(
        self,
        state: AtomisticDomainState,
        positions: ArrayLike,
        momenta: ArrayLike,
        /,
        *,
        step_index: ArrayLike,
        time: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        force: AtomisticDomainForceCache | None = None,
        rng_key: PRNGKey | None = None,
        controllers: AtomisticControllerState | None = None,
        ledger: AtomisticEnergyLedgerState | None = None,
        constraints: AtomisticDomainConstraintState | None = None,
        atom_payload: Mapping[str, ArrayLike] | None = None,
        replicated_payload: Mapping[str, ArrayLike] | None = None,
    ) -> AtomisticDomainState:
        """Advance dynamical leaves within the current owner and topology epoch.

        Traceable; inactive padding rows are kept exactly zero. Ownership,
        layout and topologies change only through `rebuild`. A static-cell
        plan refuses ``cell_vectors``.
        """
        self._require_state(state)
        layout = state.layout
        position = jnp.asarray(positions, dtype=state.positions.dtype)
        momentum = jnp.asarray(momenta, dtype=state.momenta.dtype)
        if (
            position.shape != state.positions.shape
            or momentum.shape != state.momenta.shape
        ):
            raise ValueError("Dynamics must preserve the owner-blocked row shape.")
        step = jnp.asarray(step_index, dtype=jnp.int32)
        if step.shape:
            raise ValueError("step_index must be a scalar.")
        time_ = state.time if time is None else jnp.asarray(time, dtype=state.time.dtype)
        if time_.shape:
            raise ValueError("time must be a scalar.")
        if cell_vectors is None:
            vectors = state.cell_vectors
        else:
            if not self.plan.dynamic_cell:
                raise ValueError("A static-cell domain plan has a fixed cell.")
            vectors = jnp.asarray(cell_vectors, dtype=state.cell_vectors.dtype)
            if vectors.shape != state.cell_vectors.shape:
                raise ValueError("cell_vectors must keep their shape.")
        padding = ~layout.active

        def rows(value: Array) -> Array:
            mask = padding.reshape(padding.shape + (1,) * (value.ndim - 1))
            return layout.plan.place(jnp.where(mask, jnp.zeros((), value.dtype), value))

        if force is None:
            force_ = state.force
        else:
            if not isinstance(force, AtomisticDomainForceCache):
                raise TypeError("force must be an AtomisticDomainForceCache.")
            if jax.tree.structure(force) != jax.tree.structure(state.force):
                raise ValueError("force must keep the cache structure.")
            force_ = eqx.tree_at(lambda cache: cache.forces, force, rows(force.forces))
        payload = state.atom_payload
        if atom_payload is not None:
            if set(atom_payload) != set(state.atom_payload):
                raise ValueError("atom_payload must keep its declared entries.")
            payload = {}
            for name in sorted(atom_payload):
                array = jnp.asarray(
                    atom_payload[name], dtype=state.atom_payload[name].dtype
                )
                if array.shape != state.atom_payload[name].shape:
                    raise ValueError(f"atom_payload[{name!r}] must keep its shape.")
                payload[name] = rows(array)
        replicated = state.replicated_payload
        if replicated_payload is not None:
            if set(replicated_payload) != set(state.replicated_payload):
                raise ValueError("replicated_payload must keep its declared entries.")
            replicated = {
                name: jnp.asarray(
                    replicated_payload[name], dtype=state.replicated_payload[name].dtype
                )
                for name in sorted(replicated_payload)
            }

        def same(value: Any, previous: Any, name: str) -> Any:
            if value is None:
                return previous
            if jax.tree.structure(value) != jax.tree.structure(previous):
                raise ValueError(f"{name} must keep its declared structure.")
            return value

        return AtomisticDomainState(
            layout=layout,
            positions=rows(position),
            momenta=rows(momentum),
            masses=state.masses,
            species=state.species,
            image_counts=state.image_counts,
            cell_vectors=vectors,
            atom_payload=payload,
            replicated_payload=replicated,
            controllers=same(controllers, state.controllers, "controllers"),
            ledger=same(ledger, state.ledger, "ledger"),
            constraints=same(constraints, state.constraints, "constraints"),
            force=force_,
            rng_key=state.rng_key if rng_key is None else _typed_key(rng_key),
            step_index=step,
            time=time_,
            topologies=state.topologies,
            conditioning=state.conditioning,
            case_identity_words=state.case_identity_words,
            plan_identity=state.plan_identity,
            model_revision_identity=state.model_revision_identity,
            run_identity=state.run_identity,
        )

    # -- certificate --------------------------------------------------------

    def certificate(self, state: AtomisticDomainState, /) -> AtomisticDomainCertificate:
        """Displacement, cell-deformation and current-cell stencil certificate."""
        self._require_state(state)
        plan = self.plan
        cell = plan.partition.cell
        layout = state.layout
        alive = layout.active
        dtype = state.positions.dtype
        vectors = state.cell_vectors.astype(dtype)
        fractional, solved = cell.fractional_with_vectors_with_status(
            state.positions, vectors
        )
        cell_ok = jnp.asarray(solved, dtype=jnp.bool_)
        upper = jnp.max(jnp.where(alive[:, None], fractional, -jnp.inf), axis=0)
        lower = jnp.min(jnp.where(alive[:, None], fractional, jnp.inf), axis=0)
        excursion = jnp.where(
            jnp.any(alive), jnp.maximum(upper - lower, 0.0), jnp.zeros_like(upper)
        )
        static_ok = (
            jnp.asarray(True)
            if plan.dynamic_cell
            else jnp.all(vectors == cell.vectors.astype(dtype))
        )
        owner_epochs = layout.owner_epochs.astype(jnp.int32)
        displacements, deformations, margins, complete, successful = [], [], [], [], []
        owners_current = jnp.all(owner_epochs == owner_epochs[0])
        topology_reasons = jnp.int32(0)
        for role, topology in zip(plan.roles, state.topologies, strict=True):
            moved = state.positions - topology.reference_positions
            distance = jnp.sqrt(jnp.sum(moved * moved, axis=-1))
            displacement = jnp.max(jnp.where(alive, distance, 0.0))
            change = vectors - topology.reference_cell_vectors.astype(dtype)
            translated = contract("sa,ad->sd", role.stencil.shifts.astype(dtype), change)
            deformation = jnp.max(jnp.sqrt(jnp.sum(translated * translated, axis=-1)))
            margin = role.skin - (2.0 * displacement + deformation)
            stencil_margin = cell.image_stencil_margin(
                role.stencil, vectors, role.cutoff, excursion
            )
            displacements.append(displacement)
            deformations.append(deformation)
            margins.append(margin)
            complete.append(jnp.all(stencil_margin > 0.0))
            owners_current = owners_current & jnp.all(
                topology.owner_epochs == owner_epochs
            )
            successful.append(topology.evidence.successful)
            topology_reasons = topology_reasons | _topology_reasons(topology.evidence)
        maximum = jnp.max(jnp.stack(displacements))
        deformation_ = jnp.stack(deformations)
        margin_ = jnp.stack(margins)
        displacement_ok = margin_ >= 0.0
        complete_ = jnp.stack(complete)
        successful_ = jnp.stack(successful)
        certified = (
            jnp.all(displacement_ok)
            & jnp.all(complete_)
            & jnp.all(successful_)
            & owners_current
            & cell_ok
            & static_ok
        )
        reasons = (
            _reason(_REASON.DOMAIN, ~jnp.all(displacement_ok) | ~jnp.all(complete_))
            | _reason(_REASON.DOMAIN, ~owners_current)
            | _reason(_REASON.CELL, ~cell_ok | ~static_ok)
            | jnp.where(jnp.all(successful_), jnp.int32(0), topology_reasons)
        )
        return AtomisticDomainCertificate(
            certified=certified,
            maximum_displacement=maximum,
            cell_deformation=deformation_,
            margin=margin_,
            displacement_certified=displacement_ok,
            stencil_complete=complete_,
            topology_successful=successful_,
            owners_current=owners_current,
            cell_admissible=cell_ok,
            cell_static=static_ok,
            reasons=reasons,
        )

    # -- evaluation ---------------------------------------------------------

    def owner_view(
        self, state: AtomisticDomainState, /, *, key: PRNGKey | None = None
    ) -> AtomisticDomainOwnerView[
        AtomisticDomainOwnedRows,
        AtomisticDomainReplicatedInputs,
        AtomisticDomainTermRows,
        AtomisticDomainTermInputs,
    ]:
        """Model-role inputs of every owner region (owner-blocked + replicated)."""
        self._require_state(state)
        topology = state.topologies[self._require_model()]
        certificate = self.certificate(state)
        halo = topology.halo
        owned = AtomisticDomainOwnedRows(
            positions=state.positions,
            species=state.species,
            active=state.layout.active,
            receivers=topology.receivers,
            columns=topology.columns,
            shifts=topology.shifts,
            valid=topology.valid,
            source_ids=topology.source_ids,
            relation_leaves=topology.relation_leaves,
            send_slots=None if halo is None else halo.send_slots,
            send_valid=None if halo is None else halo.send_valid,
            route_identity=_route_identity(
                self.plan,
                topology,
                state.layout,
                state.image_counts,
                state.case_identity_words,
            ),
        )
        replicated = AtomisticDomainReplicatedInputs(
            cell_vectors=state.cell_vectors,
            conditioning=state.conditioning,
            halo=None if halo is None else eqx.filter(halo, eqx.is_array, inverse=True),
            admitted=certificate.certified & self.identity_bound(state),
            capacity_successful=topology.evidence.capacity_successful,
            key=state.rng_key if key is None else _typed_key(key),
        )
        return AtomisticDomainOwnerView(
            owned=owned,
            replicated=replicated,
            terms=self.term_view(state) if self.terms else None,
            relation_tree=topology.relation_tree,
            column_count=topology.column_count,
            axis_name=self.plan.ownership.axis_name,
            role_id=topology.role_id,
        )

    def evaluate(
        self,
        state: AtomisticDomainState,
        /,
        *,
        forces: bool = True,
        stress: bool = True,
        key: PRNGKey | None = None,
    ) -> AtomisticDomainEvaluation:
        """Partition-local energy, forces, strain derivative and stress.

        A stale owner epoch, an expired certificate (displacement plus cell
        deformation beyond the skin, or an incomplete stencil in the current
        cell), a refused topology, a failed relation/model/global term, a
        non-finite output, or a state bound to another plan or revision
        (``status.identity_bound``) fails the evaluation and poisons every value.
        """
        self._require_state(state)
        if self.training_bound:
            raise ValueError(
                "A training binding evaluates only through owner_predictions with "
                "explicit parameters."
            )
        if not isinstance(forces, bool) or not isinstance(stress, bool):
            raise TypeError("forces and stress must be bools.")
        if stress and not self.stress_available:
            raise ValueError("Stress requires a fully periodic three-dimensional cell.")
        gradients = forces or stress
        reduction = self.plan.reduction
        owners = self.plan.ownership.owner_count
        dtype = state.positions.dtype
        term_count = len(self.terms)
        owner_terms = jnp.zeros((owners, term_count), dtype=dtype)
        term_capacity = term_scientific = jnp.ones((term_count,), dtype=jnp.bool_)
        term_ambiguity = jnp.zeros((term_count,), dtype=jnp.bool_)
        field_positions = np.asarray(self.field_term_positions, dtype=np.int32)
        fixed_positions = np.asarray(self.fixed_term_positions, dtype=np.int32)
        field_sum = jnp.zeros((owners,), dtype=dtype)
        if self.model_role_index is not None:
            view = self.owner_view(state, key=key)
            term_view = view.terms
            outputs = _evaluate_regions(self, view, gradients)
            model_local = outputs.local_energy
            model_global = outputs.global_share
            model_strain = outputs.strain
            model_gradient = outputs.gradient
            atom_energies = outputs.atom_energies
            finite = jnp.all(outputs.finite)
            relation = jnp.all(outputs.relation_successful)
            model = jnp.all(outputs.model_successful)
            capped = jnp.sum(outputs.capped_receivers)
            degree = jnp.max(outputs.selected_degree)
            contexts = jax.tree.map(lambda leaf: leaf[0], outputs.system_contexts)
            declared = jnp.any(outputs.global_share_declared)
            fields = outputs.field_terms
            if fields is not None:
                owner_terms = owner_terms.at[:, field_positions].set(fields.shares)
                term_capacity = term_capacity.at[field_positions].set(
                    jnp.all(fields.capacity, axis=0)
                )
                term_scientific = term_scientific.at[field_positions].set(
                    jnp.all(fields.scientific, axis=0)
                )
                term_ambiguity = term_ambiguity.at[field_positions].set(
                    jnp.any(fields.ambiguity, axis=0)
                )
                field_sum = jnp.sum(fields.shares, axis=1)
        else:
            term_view = self.term_view(state)
            model_local = jnp.zeros((owners,), dtype=dtype)
            model_global = jnp.zeros((owners,), dtype=dtype)
            model_strain = jnp.zeros((owners, 3, 3), dtype=dtype)
            model_gradient = jnp.zeros_like(state.positions)
            atom_energies = jnp.zeros(state.layout.active.shape, dtype=dtype)
            finite = relation = model = jnp.asarray(True)
            capped = degree = jnp.zeros((), dtype=jnp.int32)
            contexts = ()
            declared = jnp.asarray(False)
        # The model's global share already holds the field-dependent terms.
        owner_energies = model_local + model_global
        strain_partials = model_strain
        gradient = model_gradient
        if fixed_positions.size:
            if term_view is None:
                raise RuntimeError("Fixed-input domain terms lack their term view.")
            terms = _term_regions(self, term_view, gradients)
            owner_energies = owner_energies + jnp.sum(terms.shares, axis=1)
            strain_partials = strain_partials + terms.strain
            gradient = gradient + terms.gradient
            owner_terms = owner_terms.at[:, fixed_positions].set(terms.shares)
            term_capacity = term_capacity.at[fixed_positions].set(
                jnp.all(terms.capacity, axis=0)
            )
            term_scientific = term_scientific.at[fixed_positions].set(
                jnp.all(terms.scientific, axis=0)
            )
            term_ambiguity = term_ambiguity.at[fixed_positions].set(
                jnp.any(terms.ambiguity, axis=0)
            )
            finite = finite & jnp.all(jnp.isfinite(terms.shares))
        term_energies = reduction.ordered_sum(owner_terms)
        energy = reduction.ordered_sum(owner_energies)
        local_energy = reduction.ordered_sum(model_local)
        global_energy = reduction.ordered_sum(model_global - field_sum)
        strain = reduction.ordered_sum(strain_partials)
        finite = finite & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(strain))
        if stress:
            symmetric = 0.5 * (strain + strain.T)
            stress_value = symmetric * _stress_scale(self, state.cell_vectors)
        else:
            stress_value = jnp.full((3, 3), jnp.nan, dtype=dtype)
        certificate_record = self.certificate(state)
        certificate = certificate_record.certified
        term_successful = term_capacity & term_scientific
        bound = self.identity_bound(state)
        status = AtomisticDomainExecutionStatus(
            identity_bound=bound,
            relation_successful=relation,
            model_successful=model,
            finite=finite,
            capped_receivers=capped,
            maximum_selected_degree=degree,
            term_capacity_successful=term_capacity,
            term_scientific_successful=term_scientific,
            term_image_ambiguity=term_ambiguity,
        )
        accepted = (
            bound & certificate & finite & relation & model & jnp.all(term_successful)
        )
        reasons = (
            certificate_record.reasons
            | _reason(_REASON.NONFINITE, ~finite)
            | _reason(_REASON.RELATION, ~relation)
            | _reason(_REASON.POTENTIAL, ~bound | ~model | ~jnp.all(term_scientific))
            | _reason(_REASON.ROUTE_CAPACITY, ~jnp.all(term_capacity))
            | _reason(_REASON.IMAGE_AMBIGUITY, jnp.any(term_ambiguity))
        )
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        alive = state.layout.active
        force = jnp.where(alive[:, None], -gradient, 0.0)
        return AtomisticDomainEvaluation(
            energy=jnp.where(accepted, energy, nan),
            local_energy=jnp.where(accepted, local_energy, nan),
            global_energy=jnp.where(accepted, global_energy, nan),
            term_energies=jnp.where(accepted, term_energies, nan),
            owner_energies=jnp.where(accepted, owner_energies, nan),
            atom_energies=jnp.where(accepted, atom_energies, nan),
            forces=jnp.where(accepted & forces, force, nan),
            strain_derivative=jnp.where(accepted & gradients, strain, nan),
            stress=jnp.where(accepted, stress_value, nan),
            system_contexts=contexts,
            certificate=certificate_record,
            status=status,
            reasons=reasons,
            successful=accepted,
            strain_available=bool(stress),
            atom_energy_complete=~declared & jnp.asarray(term_count == 0),
        )

    def term_view(
        self, state: AtomisticDomainState, /
    ) -> AtomisticDomainTermView[AtomisticDomainTermRows, AtomisticDomainTermInputs]:
        """Inputs of the domain terms over every role (owner-blocked + replicated)."""
        self._require_state(state)
        names = sorted({name for term in self.terms for name in term.term.field_names})
        routes, send_slots, send_valid, halos = [], [], [], []
        for topology in state.topologies:
            routes.append(
                OwnerTermRoute(
                    receivers=topology.receivers,
                    columns=topology.columns,
                    shifts=topology.shifts,
                    valid=topology.valid,
                    source_ids=topology.source_ids,
                )
            )
            halo = topology.halo
            send_slots.append(None if halo is None else halo.send_slots)
            send_valid.append(None if halo is None else halo.send_valid)
            halos.append(None if halo is None else replicated_halo(halo))
        exception_rows: list[AtomisticDomainExceptionRows | None] = []
        exception_inputs: list[AtomisticDomainExceptionInputs | None] = []
        for prepared in self.terms:
            table = prepared.term.exceptions
            if table is None:
                exception_rows.append(None)
                exception_inputs.append(None)
                continue
            exceptions = domain_exception_topology(state.layout, table)
            exception_halo = exceptions.halo
            exception_rows.append(
                AtomisticDomainExceptionRows(
                    receivers=exceptions.receivers,
                    columns=exception_halo.route_columns,
                    rows=exceptions.rows,
                    valid=exceptions.valid,
                    send_slots=exception_halo.send_slots,
                    send_valid=exception_halo.send_valid,
                )
            )
            exception_inputs.append(
                AtomisticDomainExceptionInputs(
                    halo=replicated_halo(exception_halo),
                    capacity_successful=exceptions.capacity_successful,
                    missing=exceptions.missing,
                )
            )
        return AtomisticDomainTermView(
            owned=AtomisticDomainTermRows(
                positions=state.positions,
                active=state.layout.active,
                stable_ids=state.layout.stable_ids,
                image_counts=state.image_counts,
                fields={name: state.atom_payload[name] for name in names},
                routes=tuple(routes),
                send_slots=tuple(send_slots),
                send_valid=tuple(send_valid),
                exceptions=tuple(exception_rows),
            ),
            replicated=AtomisticDomainTermInputs(
                cell_vectors=state.cell_vectors,
                halos=tuple(halos),
                exceptions=tuple(exception_inputs),
            ),
            axis_name=self.plan.ownership.axis_name,
        )

    def owner_parameter_partials(
        self,
        state: AtomisticDomainState,
        /,
        *,
        energy_cotangent: ArrayLike,
        force_cotangent: ArrayLike | None = None,
        stress_cotangent: ArrayLike | None = None,
        parameters: PyTree[Array] | None = None,
        model_state: PyTree[Array] | None = None,
        key: PRNGKey | None = None,
    ) -> PyTree[Array]:
        """Owner-stacked exactly-once PARAMETER-lane partials (leading owner axis).

        ``force_cotangent`` is owner-blocked; the ordered owner sum of the
        returned partials is the gradient of ``e E + sum f . F + s : stress``.
        """
        self._require_state(state)
        self._require_model()
        lane = self.parameter_lane if parameters is None else parameters
        if self.training_bound and parameters is None:
            raise ValueError("A training binding requires explicit parameters.")
        dtype = state.positions.dtype
        energy = jnp.asarray(energy_cotangent, dtype=dtype)
        force = (
            jnp.zeros_like(state.positions)
            if force_cotangent is None
            else jnp.asarray(force_cotangent, dtype=dtype)
        )
        if force.shape != state.positions.shape:
            raise ValueError("force_cotangent must be owner-blocked rows.")
        stress_ = (
            None
            if stress_cotangent is None
            else jnp.asarray(stress_cotangent, dtype=dtype)
        )
        view = self.owner_view(state, key=key)
        state_lane = self.model_state_lane(model_state)
        return _parameter_regions(self, view, lane, state_lane, energy, force, stress_)

    def loss_gradient(
        self,
        state: AtomisticDomainState,
        energy_reference: ArrayLike,
        force_reference: ArrayLike,
        /,
        *,
        energy_weight: float,
        force_weight: float,
        stress_reference: ArrayLike | None = None,
        stress_weight: float = 0.0,
        key: PRNGKey | None = None,
    ) -> AtomisticDomainLossGradient:
        """Energy/force/stress loss and its exact PARAMETER-lane gradient.

        ``loss = w_E (E - E_ref)^2 + w_F sum_atoms |F - F_ref|^2 + w_S
        |stress - stress_ref|^2`` with owner-blocked ``force_reference``. The
        gradient differentiates the owner-local forces themselves (mixed
        coordinate/parameter derivatives through every exchange) using the
        linearized cotangents ``2 w (residual)``, reduced in owner order. A
        failed evaluation fails the gradient and poisons it with NaN.
        """
        self._require_model()
        weights = tuple(
            _real(name, value)
            for name, value in (
                ("energy_weight", energy_weight),
                ("force_weight", force_weight),
                ("stress_weight", stress_weight),
            )
        )
        supervise_stress = stress_reference is not None or weights[2] != 0.0
        if supervise_stress and (stress_reference is None or not self.stress_available):
            raise ValueError(
                "Stress supervision requires a stress reference and a fully periodic "
                "three-dimensional cell."
            )
        evaluation = self.evaluate(state, forces=True, stress=supervise_stress, key=key)
        dtype = state.positions.dtype
        reference_energy = jnp.asarray(energy_reference, dtype=dtype)
        if reference_energy.shape:
            raise ValueError("energy_reference must be a scalar.")
        references = jnp.asarray(force_reference, dtype=dtype)
        if references.shape != state.positions.shape:
            raise ValueError("force_reference must be owner-blocked rows (total, 3).")
        alive = state.layout.active[:, None]
        energy_residual = evaluation.energy - reference_energy
        force_residual = jnp.where(alive, evaluation.forces - references, 0.0)
        loss = weights[0] * energy_residual**2 + weights[1] * jnp.sum(force_residual**2)
        stress_cotangent = None
        if supervise_stress:
            target = jnp.asarray(stress_reference, dtype=dtype)
            if target.shape != (3, 3):
                raise ValueError("stress_reference must have shape (3, 3).")
            stress_residual = evaluation.stress - target
            loss = loss + weights[2] * jnp.sum(stress_residual**2)
            stress_cotangent = jax.lax.stop_gradient(2.0 * weights[2] * stress_residual)
        partials = self.owner_parameter_partials(
            state,
            energy_cotangent=jax.lax.stop_gradient(2.0 * weights[0] * energy_residual),
            force_cotangent=jax.lax.stop_gradient(2.0 * weights[1] * force_residual),
            stress_cotangent=stress_cotangent,
            key=key,
        )
        reduction = self.plan.reduction
        gradient = jax.tree.map(reduction.ordered_sum, partials)
        accepted = evaluation.successful
        return AtomisticDomainLossGradient(
            loss=jnp.where(accepted, loss, jnp.nan),
            parameter_gradient=jax.tree.map(
                lambda value: jnp.where(accepted, value, jnp.nan), gradient
            ),
            evaluation=evaluation,
            successful=accepted,
        )

    # -- transactions -------------------------------------------------------

    def rebuild(self, state: AtomisticDomainState, /) -> AtomisticDomainTransition:
        """Wrap, migrate and rebuild every role epoch as one global transaction.

        Every atom moves with its complete continuation (positions, momenta,
        masses, species, image counts, force-cache rows and every
        ``atom_payload`` entry) to the owner of its wrapped fractional
        coordinate in the CURRENT cell; stored positions are wrapped by whole
        current-cell translations that are added to the image counts.
        Constraint rows re-anchor through ``rebuild_domain_constraints`` on the
        migrated layout. The candidate commits only if the migration, every
        role topology on every owner, the cell solve and the constraints admit
        it; otherwise the accepted state is returned bitwise with the
        attempt's evidence. Traceable and pytree-structure preserving.
        """
        self._require_state(state)
        plan = self.plan
        cell = plan.partition.cell
        layout = state.layout
        vectors = state.cell_vectors.astype(state.positions.dtype)
        wrapped, wraps, solved = _wrap_rows(cell, state.positions, layout.active, vectors)
        destinations = jnp.where(layout.active, plan.partition.owners(wrapped), 0)
        payload = {
            "rows": {
                "positions": state.positions,
                "momenta": state.momenta,
                "masses": state.masses,
                "species": state.species,
                "image_counts": state.image_counts,
                "forces": state.force.forces,
            },
            "wraps": wraps,
            "atom_payload": dict(state.atom_payload),
        }
        migrated = layout.migrate(
            destinations,
            packet_capacity=plan.migration_capacity,
            points=wrapped.astype(layout.points.dtype),
            payload=payload,
        )
        moved = migrated.payload
        new_layout = migrated.layout
        positions, images = _apply_wraps(
            new_layout,
            moved["rows"]["positions"],
            moved["rows"]["image_counts"],
            moved["wraps"],
            vectors,
        )
        topologies = _build_topologies(
            plan,
            new_layout,
            positions,
            vectors,
            tuple(topology.epoch + 1 for topology in state.topologies),
        )
        constraints = state.constraints
        constraints_committed = jnp.asarray(True, dtype=jnp.bool_)
        constraint_reasons = jnp.int32(0)
        if constraints is not None:
            constraint_transition = rebuild_domain_constraints(constraints, new_layout)
            constraints = constraint_transition.state
            constraints_committed = jnp.asarray(
                constraint_transition.committed, dtype=jnp.bool_
            )
            # Binding owns the capacity/scientific split. A refused rebuild
            # must retain capacity-only evidence so a larger route can retry.
            constraint_reasons = jnp.where(
                constraints_committed,
                jnp.int32(0),
                constraint_transition.evidence.reasons,
            )
        topologies_ok = jnp.all(
            jnp.stack([topology.evidence.successful for topology in topologies])
        )
        cell_ok = jnp.asarray(solved, dtype=jnp.bool_)
        committed = (
            migrated.evidence.committed & topologies_ok & constraints_committed & cell_ok
        )
        candidate = AtomisticDomainState(
            layout=new_layout,
            positions=positions,
            momenta=moved["rows"]["momenta"],
            masses=moved["rows"]["masses"],
            species=moved["rows"]["species"],
            image_counts=images,
            cell_vectors=state.cell_vectors,
            atom_payload=moved["atom_payload"],
            replicated_payload=state.replicated_payload,
            controllers=state.controllers,
            ledger=state.ledger,
            constraints=constraints,
            force=eqx.tree_at(
                lambda cache: cache.forces, state.force, moved["rows"]["forces"]
            ),
            rng_key=state.rng_key,
            step_index=state.step_index,
            time=state.time,
            topologies=topologies,
            conditioning=state.conditioning,
            case_identity_words=state.case_identity_words,
            plan_identity=state.plan_identity,
            model_revision_identity=state.model_revision_identity,
            run_identity=state.run_identity,
        )
        evidence = migrated.evidence
        migration_reasons = _reason(
            _REASON.DOMAIN,
            (evidence.invalid_destinations > 0) | ~evidence.epochs_consistent,
        ) | _reason(
            _REASON.ROUTE_CAPACITY,
            (evidence.maximum_packet > evidence.packet_capacity)
            | (evidence.maximum_received > evidence.local_capacity),
        )
        reasons = migration_reasons | _reason(_REASON.CELL, ~cell_ok)
        for topology in topologies:
            reasons = reasons | jnp.where(
                topology.evidence.successful,
                jnp.int32(0),
                _topology_reasons(topology.evidence),
            )
        reasons = reasons | constraint_reasons
        return AtomisticDomainTransition(
            state=_select_tree(committed, candidate, state),
            committed=committed,
            migration=evidence,
            topologies=tuple(topology.evidence for topology in topologies),
            constraints_committed=constraints_committed,
            cell_admissible=cell_ok,
            reasons=jnp.where(committed, jnp.int32(0), reasons),
        )

    def rebind_model(
        self, model: AtomisticLayeredModel, state: AtomisticDomainState, /
    ) -> AtomisticDomainState:
        """Bind ``state`` to this runtime's model revision; the force cache is stale.

        ``model`` must be this runtime's model revision (prepare a runtime for
        an updated model first). No physical or random step is consumed. A
        host transaction: plan and revision identities are refused exactly.
        """
        self._require_plan_host(state)
        self._require_model()
        if not isinstance(model, AbstractAtomisticPotential):
            raise TypeError("model must be an AbstractAtomisticPotential.")
        revision = atomistic_model_revision(model).revision_id
        if revision != self.model_revision_id:
            raise ValueError("The model is not this runtime's prepared revision.")
        if revision == state.model_revision_id:
            return state
        return AtomisticDomainState(
            layout=state.layout,
            positions=state.positions,
            momenta=state.momenta,
            masses=state.masses,
            species=state.species,
            image_counts=state.image_counts,
            cell_vectors=state.cell_vectors,
            atom_payload=state.atom_payload,
            replicated_payload=state.replicated_payload,
            controllers=state.controllers,
            ledger=state.ledger,
            constraints=state.constraints,
            force=eqx.tree_at(
                lambda cache: (cache.step, cache.successful),
                state.force,
                (state.step_index - 1, jnp.asarray(False)),
            ),
            rng_key=state.rng_key,
            step_index=state.step_index,
            time=state.time,
            topologies=state.topologies,
            conditioning=state.conditioning,
            case_identity_words=state.case_identity_words,
            plan_identity=state.plan_identity,
            model_revision_identity=self.model_revision_identity,
            run_identity=state.run_identity,
        )

    # -- stage transport ----------------------------------------------------

    def to_packet(self, state: AtomisticDomainState, /) -> AtomisticDomainPacket:
        """This process's host fragment of an accepted state.

        Only addressable shards of owner-blocked leaves are read; replicated
        leaves are read from this process's replica.
        """
        self._require_plan_host(state)
        layout = state.layout
        rows = {
            "positions": _host_rows(state.positions),
            "momenta": _host_rows(state.momenta),
            "masses": _host_rows(state.masses),
            "species": _host_rows(state.species),
            "image_counts": _host_rows(state.image_counts),
            "forces": _host_rows(state.force.forces),
            "stable_ids": _host_rows(layout.stable_ids),
            "logical_indices": _host_rows(layout.logical_indices),
            "active": _host_rows(layout.active),
        }
        payload = {name: _host_rows(value) for name, value in state.atom_payload.items()}
        force = state.force
        record = AtomisticDomainForceRecord(
            *(
                np.asarray(jax.device_get(value))
                for value in (
                    force.energy,
                    force.strain_derivative,
                    force.stress,
                    force.strain_available,
                    force.step,
                    force.successful,
                )
            )
        )
        key_data = np.asarray(jax.device_get(jax.random.key_data(state.rng_key)))
        implementation = str(jax.random.key_impl(state.rng_key))
        header = {
            "cell_vectors": np.asarray(jax.device_get(state.cell_vectors)),
            "replicated_payload": _host_tree(dict(state.replicated_payload)),
            "controllers": _host_tree(state.controllers),
            "ledger": _host_tree(state.ledger),
            "conditioning": _host_tree(state.conditioning),
            "rng_key_data": key_data,
            "case_identity_words": np.asarray(jax.device_get(state.case_identity_words)),
            "step_index": np.asarray(jax.device_get(state.step_index)),
            "time": np.asarray(jax.device_get(state.time)),
            "force_record": record,
        }
        constraints = (
            None
            if state.constraints is None
            else export_domain_constraints(state.constraints)
        )
        role_ids = tuple(topology.role_id for topology in state.topologies)
        header_id = canonical_fingerprint(
            {
                "kind": "atomistic-domain-packet-header",
                "arrays": array_tree_fingerprint(
                    {
                        name: value
                        for name, value in header.items()
                        if name
                        not in (
                            "controllers",
                            "ledger",
                            "conditioning",
                            "force_record",
                            "replicated_payload",
                        )
                    }
                ),
                "replicated": array_tree_fingerprint(
                    jax.tree.leaves(
                        (
                            header["replicated_payload"],
                            header["controllers"],
                            header["ledger"],
                            header["conditioning"],
                            tuple(record),
                        )
                    )
                ),
                "key_implementation": implementation,
                "logical_count": layout.logical_count,
                "role_ids": list(role_ids),
                "model_revision": state.model_revision_id,
                "run": state.run_id,
            }
        )
        return AtomisticDomainPacket(
            rows=rows,
            atom_payload=payload,
            cell_vectors=header["cell_vectors"],
            replicated_payload=header["replicated_payload"],
            controllers=header["controllers"],
            ledger=header["ledger"],
            conditioning=header["conditioning"],
            constraints=constraints,
            rng_key_data=key_data,
            case_identity_words=header["case_identity_words"],
            step_index=header["step_index"],
            time=header["time"],
            force_record=record,
            key_implementation=implementation,
            logical_count=layout.logical_count,
            role_ids=role_ids,
            model_revision_id=state.model_revision_id,
            run_id=state.run_id,
            header_id=header_id,
        )

    def merge_packets(
        self, fragments: Sequence[AtomisticDomainPacket], /
    ) -> AtomisticDomainPacket:
        """This process's single fragment of a packet split over any processes.

        Headers must agree exactly; rows (and constraint rows) of every
        fragment are concatenated and compacted into this process's
        addressable owner rows, without ingress. Refused when the model
        revision or role identities differ, or when the active rows exceed
        this process's local capacity. Stage transport re-fragments
        continuations with it when a destination has another process layout.
        """
        parts = tuple(fragments)
        if not parts or any(
            not isinstance(part, AtomisticDomainPacket) for part in parts
        ):
            raise TypeError("fragments must be a nonempty sequence of packets.")
        head = parts[0]
        if any(part.header_id != head.header_id for part in parts):
            raise ValueError("Packet fragments carry different replicated headers.")
        if head.model_revision_id != self.model_revision_id:
            raise ValueError("The packet belongs to another model revision.")
        if head.role_ids != tuple(role.role_id for role in self.plan.roles):
            raise ValueError("The packet roles differ from this runtime's roles.")
        ownership = self.plan.ownership
        local_rows = (
            len(
                ownership.sharding(1).addressable_devices_indices_map(
                    (ownership.total_capacity,)
                )
            )
            * ownership.local_capacity
            if not ownership.lane_reference
            else ownership.total_capacity
        )
        active = np.concatenate([part.rows["active"] for part in parts], axis=0)
        selected = np.flatnonzero(active)
        if selected.size > local_rows:
            raise ValueError(
                f"{selected.size} transported atoms exceed this process's "
                f"{local_rows} local rows."
            )

        def compact(values: np.ndarray) -> np.ndarray:
            output = np.zeros((local_rows,) + values.shape[1:], dtype=values.dtype)
            output[: selected.size] = values[selected]
            return output

        return AtomisticDomainPacket(
            rows={
                name: compact(np.concatenate([part.rows[name] for part in parts]))
                for name in head.rows
            },
            atom_payload={
                name: compact(np.concatenate([part.atom_payload[name] for part in parts]))
                for name in head.atom_payload
            },
            cell_vectors=head.cell_vectors,
            replicated_payload=head.replicated_payload,
            controllers=head.controllers,
            ledger=head.ledger,
            conditioning=head.conditioning,
            constraints=_merged_constraints(parts, ownership),
            rng_key_data=head.rng_key_data,
            case_identity_words=head.case_identity_words,
            step_index=head.step_index,
            time=head.time,
            force_record=head.force_record,
            key_implementation=head.key_implementation,
            logical_count=head.logical_count,
            role_ids=head.role_ids,
            model_revision_id=head.model_revision_id,
            run_id=head.run_id,
            header_id=head.header_id,
        )

    def from_packet(
        self, fragments: Sequence[AtomisticDomainPacket], /
    ) -> AtomisticDomainState:
        """Assemble this process's fragments and ingress them on this runtime.

        `merge_packets` concatenates and compacts the fragments into this
        process's addressable owner blocks; the rows are then migrated on
        device to their owners in the current cell (`ingress_process_local`).
        """
        merged = self.merge_packets(fragments)
        rows = merged.rows
        key = jax.random.wrap_key_data(
            jnp.asarray(merged.rng_key_data), impl=merged.key_implementation
        )
        return self.ingress_process_local(
            rows["positions"],
            rows["species"],
            rng_key=key,
            active=rows["active"],
            stable_ids=rows["stable_ids"],
            logical_indices=rows["logical_indices"],
            logical_count=merged.logical_count,
            momenta=rows["momenta"],
            masses=np.where(rows["active"], rows["masses"], 1.0).astype(
                rows["masses"].dtype
            ),
            image_counts=rows["image_counts"],
            forces=rows["forces"],
            force_record=merged.force_record,
            case_identity=merged.case_identity_words,
            cell_vectors=merged.cell_vectors,
            atom_payload=merged.atom_payload,
            replicated_payload=merged.replicated_payload,
            conditioning=jax.tree.map(jnp.asarray, merged.conditioning),
            controllers=jax.tree.map(jnp.asarray, merged.controllers),
            ledger=jax.tree.map(jnp.asarray, merged.ledger),
            constraints=merged.constraints,
            step_index=int(merged.step_index),
            time=float(merged.time),
            run_id=merged.run_id,
        )


def _merged_constraints(
    parts: tuple[AtomisticDomainPacket, ...], ownership: DistributedOwnershipPlan, /
) -> AtomisticDomainConstraintRows | None:
    """Constraint rows of every fragment compacted into this process's blocks.

    Active rows of all fragments are kept (each row ID once), statics and the
    replicated residuals must agree, and padding rows receive IDs above
    ``row_count`` unique across processes (their global row slot).
    """
    blocks = [part.constraints for part in parts]
    if all(block is None for block in blocks):
        return None
    if any(block is None for block in blocks):
        raise ValueError("Packet fragments disagree on carrying constraint rows.")
    present = [block for block in blocks if block is not None]
    head = present[0]

    def statics(block: AtomisticDomainConstraintRows) -> tuple[Any, ...]:
        return (
            block.row_capacity,
            block.row_count,
            block.halo_capacity,
            block.directory_capacity,
            block.migration_capacity,
            block.plan_id,
            np.dtype(block.targets.dtype),
        )

    for block in present[1:]:
        if statics(block) != statics(head) or not all(
            np.array_equal(getattr(block, name), getattr(head, name))
            for name in ("position_residual", "velocity_residual", "topology_epoch")
        ):
            raise ValueError("Packet fragments carry different constraint headers.")
    active = np.concatenate([block.active for block in present])
    selected = np.flatnonzero(active)
    ids = np.concatenate([block.row_ids for block in present])[selected]
    if np.unique(ids).size != ids.size:
        raise ValueError("Packet fragments repeat an active constraint row.")
    process = jax.process_index()
    owners = [
        owner
        for owner, owner_process in enumerate(ownership.owner_processes)
        if ownership.lane_reference or owner_process == process
    ]
    capacity = head.row_capacity
    total = len(owners) * capacity
    if selected.size > total:
        raise ValueError(
            f"{selected.size} transported constraint rows exceed this process's "
            f"{total} constraint rows."
        )

    def compact(name: str, fill: Any) -> np.ndarray:
        values = np.concatenate([getattr(block, name) for block in present])[selected]
        output = np.full((total,) + values.shape[1:], fill, dtype=values.dtype)
        output[: selected.size] = values
        return output

    slots = np.concatenate(
        [owner * capacity + np.arange(capacity, dtype=np.int64) for owner in owners]
    )
    row_ids = head.row_count + slots
    row_ids[: selected.size] = ids
    return AtomisticDomainConstraintRows(
        row_ids=row_ids,
        atoms=compact("atoms", -1),
        components=compact("components", -1),
        targets=compact("targets", 0),
        lagrange=compact("lagrange", 0),
        active=compact("active", False),
        position_residual=head.position_residual,
        velocity_residual=head.velocity_residual,
        topology_epoch=head.topology_epoch,
        row_capacity=capacity,
        row_count=head.row_count,
        halo_capacity=head.halo_capacity,
        directory_capacity=head.directory_capacity,
        migration_capacity=head.migration_capacity,
        plan_id=head.plan_id,
    )


# ---------------------------------------------------------------------------
# Compiled device helpers
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _wrap_rows(
    cell: Any, positions: Array, active: Array, vectors: Array
) -> tuple[Array, Array, Array]:
    """Wrapped fractional coordinates, integer wraps and the cell solve status."""
    fractional, solved = cell.fractional_with_vectors_with_status(positions, vectors)
    images = jax.lax.stop_gradient(
        jnp.where(cell.periodic_mask, jnp.floor(fractional), 0.0)
    )
    alive = active[:, None]
    wrapped = jnp.where(alive, fractional - images, 0.0)
    wraps = jnp.where(alive, images, 0.0).astype(jnp.int32)
    finite = jnp.all(jnp.where(alive, jnp.isfinite(fractional), True))
    return wrapped, wraps, jnp.asarray(solved, dtype=jnp.bool_) & finite


def _apply_wraps(
    layout: DistributedPointLayout,
    positions: Array,
    image_counts: Array,
    wraps: Array,
    vectors: Array,
) -> tuple[Array, Array]:
    alive = layout.active[:, None]
    translation = contract(
        "na,ad->nd", wraps.astype(positions.dtype), vectors.astype(positions.dtype)
    )
    return (
        jnp.where(alive, positions - translation, 0.0),
        jnp.where(alive, image_counts + wraps, 0).astype(jnp.int32),
    )


@eqx.filter_jit
def _evaluate_regions(
    runtime: PreparedAtomisticDomainRuntime,
    view: _OwnerInputView,
    gradients: bool,
) -> _OwnerOutputs:
    axis = runtime.plan.ownership.axis_name
    lanes = (runtime.parameter_lane, runtime.state_lane)

    def body(local_view: _OwnerInputView, local_lanes: tuple) -> _OwnerOutputs:
        model = combine_parameters(local_lanes[0], local_lanes[1], runtime.fixed_lane)
        outputs = _region_outputs(
            runtime, model, local_view, gradients=gradients, parameters=False
        )
        return _OwnerOutputs(
            atom_energies=outputs.atom_energies,
            gradient=outputs.gradient,
            local_energy=outputs.local_energy[None],
            global_share=outputs.global_share[None],
            strain=outputs.strain[None],
            finite=outputs.finite[None],
            relation_successful=outputs.relation_successful[None],
            model_successful=outputs.model_successful[None],
            capped_receivers=outputs.capped_receivers[None],
            selected_degree=outputs.selected_degree[None],
            system_contexts=jax.tree.map(
                lambda leaf: leaf[None], outputs.system_contexts
            ),
            parameter_gradient=None,
            global_share_declared=outputs.global_share_declared[None],
            field_terms=jax.tree.map(lambda leaf: leaf[None], outputs.field_terms),
        )

    owner = PartitionSpec(axis)
    return runtime.plan.ownership.map(
        body, (view.partition_specs(), PartitionSpec()), owner
    )(view, lanes)


@eqx.filter_jit
def _parameter_regions(
    runtime: PreparedAtomisticDomainRuntime,
    view: _OwnerInputView,
    parameters: PyTree[Array],
    model_state: PyTree[Array],
    energy_cotangent: Array,
    force_cotangent: Array,
    stress_cotangent: Array | None,
) -> PyTree[Array]:
    axis = runtime.plan.ownership.axis_name

    def body(
        local_view: _OwnerInputView,
        lane: PyTree[Array],
        state_lane: PyTree[Array],
        energy: Array,
        force: Array,
        stress: Array | None,
    ) -> PyTree[Array]:
        partial = owner_parameter_partials(
            runtime,
            lane,
            local_view,
            energy_cotangent=energy,
            force_cotangent=force,
            stress_cotangent=stress,
            model_state=state_lane,
        )
        return jax.tree.map(lambda leaf: leaf[None], partial)

    owner = PartitionSpec(axis)
    replicated = PartitionSpec()
    return runtime.plan.ownership.map(
        body,
        (view.partition_specs(), replicated, replicated, replicated, owner, replicated),
        owner,
    )(view, parameters, model_state, energy_cotangent, force_cotangent, stress_cotangent)


class _TermStack(NamedTuple):
    """One owner's per-term shares and globally reduced statuses."""

    shares: Array
    capacity: Array
    scientific: Array
    ambiguity: Array


class _TermOwnerOutputs(NamedTuple):
    """One owner's fixed-input term shares, exactly-once gradient and strain."""

    stack: _TermStack
    gradient: Array
    strain: Array


class _TermOutputs(NamedTuple):
    shares: Array
    gradient: Array
    strain: Array
    capacity: Array
    scientific: Array
    ambiguity: Array


def _stack_terms(results: Sequence[OwnerTermResult], dtype: Any, /) -> _TermStack:
    def stacked(field: str, kind: Any) -> Array:
        return jnp.stack(
            [jnp.asarray(getattr(result, field), dtype=kind) for result in results]
        )

    return _TermStack(
        shares=stacked("energy_share", dtype),
        capacity=stacked("capacity_successful", jnp.bool_),
        scientific=stacked("scientific_successful", jnp.bool_),
        ambiguity=stacked("ambiguity", jnp.bool_),
    )


def _term_exchange(
    runtime: PreparedAtomisticDomainRuntime, view: _TermInputView, index: int
) -> AtomisticColumnExchange:
    """Column exchange of role ``index`` under the plan's strategy."""
    return _column_exchange(
        runtime.plan,
        view.replicated.halos[index],
        view.owned.send_slots[index],
        view.owned.send_valid[index],
    )


def _global_flag(flag: Array, axis_name: str, /) -> Array:
    value = jax.lax.stop_gradient(jnp.all(flag)).astype(jnp.int32)
    return JaxCollectiveProvider(axis_name).minimum(value) == 1


def _combined(parts: tuple[OwnerTermResult, ...], /) -> OwnerTermResult:
    """One term's owner result from its real-space and reciprocal parts."""
    share = parts[0].energy_share
    capacity = parts[0].capacity_successful
    scientific = parts[0].scientific_successful
    for part in parts[1:]:
        share = share + part.energy_share
        capacity = capacity & part.capacity_successful
        scientific = scientific & part.scientific_successful
    return OwnerTermResult(share, capacity, scientific, parts[0].ambiguity)


def _term_charges(
    term: AtomisticDomainEwaldTerm,
    owned: AtomisticDomainTermRows,
    fields: Mapping[AtomisticObservable, Array] | None,
    dtype: Any,
    /,
) -> tuple[Array, Array]:
    """Declared (fixed) or model-predicted charges of the owned rows."""
    if not term.model_fields:
        (name,) = term.field_names
        return (
            jax.lax.stop_gradient(owned.fields[name].astype(dtype)),
            jnp.asarray(True),
        )
    if fields is None or AtomisticObservable.CHARGES not in fields:
        raise ValueError(
            f"The layered model predicts no CHARGES for domain term {term.term_key}."
        )
    values = jnp.asarray(fields[AtomisticObservable.CHARGES]).astype(dtype)
    if values.shape != owned.active.shape:
        raise ValueError("Predicted charges must hold one value per owned row.")
    usable = owned.active & jnp.isfinite(values)
    return jnp.where(usable, values, 0.0), jnp.all(~owned.active | usable)


# One explicit per-term scientific dispatch: it preserves plan order, field
# dependencies, the collective call sequence and per-term capacity/scientific
# evidence; the numerical branch kernels already live at their owning substrates.
def _owner_term_results(
    runtime: PreparedAtomisticDomainRuntime,
    view: _TermInputView,
    positions: Array,
    cell_vectors: Array,
    fields: Mapping[AtomisticObservable, Array] | None,
    /,
    *,
    field_terms: bool,
) -> tuple[OwnerTermResult, ...]:
    """Owner shares of the fixed-input (or field-dependent) terms in plan order."""
    plan = runtime.plan
    axis = plan.ownership.axis_name
    owned = view.owned
    replicated = view.replicated
    results = []
    for position, prepared in enumerate(runtime.terms):
        term = prepared.term
        if bool(term.model_fields) != field_terms:
            continue
        index = prepared.role_indices[0]
        requirement = plan.roles[index].requirement
        route = owned.routes[index]
        exception_rows = owned.exceptions[position]
        exception_inputs = replicated.exceptions[position]
        exceptions = exception_exchange = None
        if exception_rows is not None and exception_inputs is not None:
            exceptions = OwnerExceptionRoute(
                receivers=exception_rows.receivers,
                columns=exception_rows.columns,
                rows=exception_rows.rows,
                valid=exception_rows.valid,
                send_slots=exception_rows.send_slots,
                send_valid=exception_rows.send_valid,
                capacity_successful=exception_inputs.capacity_successful,
                missing=exception_inputs.missing,
            )
            exception_exchange = HaloColumnExchange(
                exception_inputs.halo,
                exception_rows.send_slots,
                exception_rows.send_valid,
                plan.message_capacity_bytes,
            )
        if isinstance(term, AtomisticDomainPolarizationTerm):
            polarization = prepared.work
            if not isinstance(polarization, PreparedOwnerPolarization):
                raise TypeError(
                    f"Domain polarization term {term.term_key!r} lacks its prepared "
                    "owner polarization."
                )
            halo = replicated.halos[index]
            send_slots = owned.send_slots[index]
            send_valid = owned.send_valid[index]
            if halo is None or send_slots is None or send_valid is None:
                raise ValueError(
                    f"Domain polarization term {term.term_key!r} needs its halo plan "
                    "and send routes (HALO strategy)."
                )
            results.append(
                owner_polarization_share(
                    term,
                    polarization,
                    positions,
                    route,
                    halo,
                    send_slots,
                    send_valid,
                    exceptions,
                    None if exception_inputs is None else exception_inputs.halo,
                    {name: owned.fields[name] for name in term.field_names},
                    owned.active,
                    cell_vectors,
                    axis,
                    plan.message_capacity_bytes,
                )
            )
            continue
        if not isinstance(
            term,
            (
                AtomisticDomainLennardJonesTerm,
                AtomisticDomainEwaldTerm,
                AtomisticDomainLennardJonesPMETerm,
                AtomisticDomainFourierD3Term,
            ),
        ):
            raise TypeError(f"Unsupported domain term {type(term).__name__}.")
        exchange = _term_exchange(runtime, view, index)
        columns = exchange.gather(positions)
        if isinstance(term, AtomisticDomainLennardJonesTerm):
            (name,) = term.field_names
            types = owned.fields[name]
            results.append(
                owner_lennard_jones_share(
                    term,
                    requirement,
                    positions,
                    columns,
                    exchange.gather(types),
                    types,
                    route,
                    cell_vectors,
                    owned.stable_ids,
                    axis,
                )
            )
            continue
        reciprocal = prepared.work
        if not isinstance(reciprocal, PreparedReciprocalDistribution):
            raise TypeError(
                f"Domain term {term.term_key!r} lacks its prepared reciprocal "
                "distribution."
            )
        if isinstance(term, AtomisticDomainLennardJonesPMETerm):
            type_name, anchor_name = term.field_names
            types = owned.fields[type_name]
            results.append(
                _combined(
                    (
                        owner_lennard_jones_pme_real_share(
                            term,
                            requirement,
                            positions,
                            columns,
                            exchange.gather(types),
                            types,
                            route,
                            cell_vectors,
                            owned.stable_ids,
                            axis,
                        ),
                        owner_lennard_jones_pme_reciprocal_share(
                            reciprocal,
                            positions,
                            types,
                            owned.fields[anchor_name],
                            owned.active,
                            cell_vectors,
                            axis,
                        ),
                    )
                )
            )
            continue
        if isinstance(term, AtomisticDomainFourierD3Term):
            table_name, channel_name, active_name = term.field_names
            table_rows = owned.fields[table_name]
            d3_active = owned.fields[active_name]
            results.append(
                owner_fourier_d3_share(
                    term,
                    reciprocal,
                    positions,
                    columns,
                    exchange.gather(table_rows),
                    exchange.gather(d3_active & owned.active),
                    table_rows,
                    owned.fields[channel_name],
                    d3_active,
                    route,
                    owned.image_counts,
                    owned.active,
                    cell_vectors,
                    axis,
                )
            )
            continue
        charges, charges_ok = _term_charges(term, owned, fields, positions.dtype)
        parts = [
            owner_ewald_real_share(
                term,
                requirement,
                positions,
                columns,
                exchange.gather(charges),
                charges,
                route,
                cell_vectors,
                owned.stable_ids,
                axis,
            ),
            owner_reciprocal_share(
                reciprocal,
                positions,
                charges,
                owned.image_counts,
                owned.active,
                cell_vectors,
                axis,
            ),
        ]
        if exceptions is not None and exception_exchange is not None:
            parts.append(
                owner_ewald_exception_share(
                    term,
                    positions,
                    exception_exchange.gather(positions),
                    exception_exchange.gather(charges),
                    charges,
                    exceptions,
                    cell_vectors,
                    axis,
                )
            )
        share = parts[0].energy_share
        for part in parts[1:]:
            share = share + part.energy_share
        capacity = scientific = jnp.asarray(True)
        for part in parts:
            capacity = capacity & part.capacity_successful
            scientific = scientific & part.scientific_successful
        results.append(
            OwnerTermResult(
                share,
                capacity,
                scientific & _global_flag(charges_ok, axis),
                parts[0].ambiguity,
            )
        )
    return tuple(results)


def _owner_term_outputs(
    runtime: PreparedAtomisticDomainRuntime,
    view: _TermInputView,
    gradients: bool,
) -> _TermOwnerOutputs:
    """One owner's fixed-input term shares with exactly-once derivatives.

    Each owner differentiates its own share at ``x (I + e)^T``, ``H (I + e)^T``;
    column gathers transpose through their exchanges, so the owner sums of the
    position and strain partials are the derivatives of the total term energy.
    """
    positions = view.owned.positions
    dtype = positions.dtype
    vectors = view.replicated.cell_vectors.astype(dtype)

    def total(coordinates: Array, strain: Array) -> tuple[Array, _TermStack]:
        deformation = jnp.eye(3, dtype=dtype) + strain
        results = _owner_term_results(
            runtime,
            view,
            contract("nd,ed->ne", coordinates, deformation),
            contract("ad,ed->ae", vectors, deformation),
            None,
            field_terms=False,
        )
        stack = _stack_terms(results, dtype)
        return jnp.sum(stack.shares), stack

    zero = jnp.zeros((3, 3), dtype=dtype)
    if gradients:
        _, pullback, stack = jax.vjp(total, positions, zero, has_aux=True)
        gradient, strain = pullback(jnp.ones((), dtype=dtype))
    else:
        _, stack = total(positions, zero)
        gradient, strain = jnp.zeros_like(positions), zero
    gradient = jnp.where(view.owned.active[:, None], gradient, 0.0)
    return _TermOwnerOutputs(stack, gradient, strain)


@eqx.filter_jit
def _term_regions(
    runtime: PreparedAtomisticDomainRuntime,
    view: _TermInputView,
    gradients: bool,
) -> _TermOutputs:
    """Owner-stacked fixed-input term shares, forces and strain partials."""
    axis = runtime.plan.ownership.axis_name

    def body(local: _TermInputView) -> _TermOutputs:
        outputs = _owner_term_outputs(runtime, local, gradients)
        stack = outputs.stack
        return _TermOutputs(
            shares=stack.shares[None],
            gradient=outputs.gradient,
            strain=outputs.strain[None],
            capacity=stack.capacity[None],
            scientific=stack.scientific[None],
            ambiguity=stack.ambiguity[None],
        )

    return runtime.plan.ownership.map(
        body, (view.partition_specs(),), PartitionSpec(axis)
    )(view)


def _prepare_terms(
    plan: AtomisticDomainPlan, /
) -> tuple[PreparedAtomisticDomainTerm, ...]:
    prepared = []
    for term in plan.terms:
        work: PreparedReciprocalDistribution | PreparedOwnerPolarization | None = None
        if isinstance(
            term,
            (
                AtomisticDomainEwaldTerm,
                AtomisticDomainLennardJonesPMETerm,
                AtomisticDomainFourierD3Term,
            ),
        ):
            work = prepare_reciprocal_distribution(
                term.reciprocal, plan.ownership, plan.partition
            )
        elif isinstance(term, AtomisticDomainPolarizationTerm):
            if plan.strategy is not AtomisticDomainStrategy.HALO:
                raise ValueError(
                    "The owner polarization relation binds a physical halo plan "
                    "(OwnerMultipoleRelation); a polarization term needs the HALO "
                    "strategy."
                )
            work = prepare_owner_polarization(term.plan, plan.ownership, plan.partition)
        elif not isinstance(term, AtomisticDomainLennardJonesTerm):
            raise TypeError(f"Unsupported domain term {type(term).__name__}.")
        prepared.append(
            PreparedAtomisticDomainTerm(
                term=term, work=work, role_indices=plan.term_role_indices(term)
            )
        )
    return tuple(prepared)


def _require_reciprocal_binding(
    plan: AtomisticDomainPlan, model: AtomisticLayeredModel | None, /
) -> None:
    """``plan.reciprocal`` distributes exactly the model's own global epilogue.

    The only layered model with a global payload is MACE-LES, whose latent
    Ewald action must be the distributed definition (same latent definition
    and scale contract); every other model, and a term-only runtime, refuses a
    reciprocal epilogue.
    """
    from ..nn.atomistic._mace_les import MACELESPotential

    reciprocal = plan.reciprocal
    if isinstance(model, MACELESPotential):
        if reciprocal is None:
            raise ValueError(
                "MACE-LES declares a latent Ewald payload; plan.reciprocal must "
                "distribute its latent Ewald action."
            )
        definition = reciprocal.definition
        latent = model.latent_ewald
        if (
            not isinstance(definition, PreparedLatentEwald)
            or definition.definition.definition_id != latent.definition.definition_id
            or definition.scale.scale_id != latent.scale.scale_id
        ):
            raise ValueError(
                "plan.reciprocal distributes another definition than the model's "
                "latent Ewald action."
            )
    elif reciprocal is not None:
        raise ValueError(
            "plan.reciprocal is a layered model's global epilogue; this runtime's "
            "model declares none."
        )


def prepare_atomistic_domain(
    plan: AtomisticDomainPlan, model: AtomisticLayeredModel | None = None, /
) -> PreparedAtomisticDomainRuntime:
    """Admit an optional layered learned potential and the plan's domain terms.

    The model's directed-image-graph requirement
    (``requirements.neighborhood(DIRECTED_IMAGE_GRAPH)``) must be exactly the
    requirement of the plan's ``(DIRECTED_IMAGE_GRAPH, "model")`` role, so the
    domain uses the model's own cutoff and declared selection. A runtime
    without a model executes the plan's terms only.
    """
    if not isinstance(plan, AtomisticDomainPlan):
        raise TypeError("plan must be an AtomisticDomainPlan.")
    terms = _prepare_terms(plan)
    if model is None:
        if not terms:
            raise ValueError("A domain runtime needs a layered model or domain terms.")
        if any(term.model_fields for term in plan.terms):
            raise ValueError(
                "A field-dependent domain term consumes the layered model's predicted "
                "fields; the runtime needs that model."
            )
        revision = canonical_fingerprint(
            {
                "kind": "atomistic-domain-terms",
                "terms": [term.term_id for term in plan.terms],
            }
        )
        index = None
        lanes: tuple[PyTree[Array], PyTree[Array], PyTree[Array]] = (None, None, None)
    else:
        if not isinstance(model, AbstractAtomisticPotential):
            raise TypeError("Domain execution requires an AbstractAtomisticPotential.")
        _require_layered(model)
        requirement = model.requirements.neighborhood(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
        )
        index = plan.role_index(
            AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH, _MODEL_TERM
        )
        if plan.roles[index].requirement.requirement_id != requirement.requirement_id:
            raise ValueError(
                "The plan's model role does not carry the model's directed-graph "
                "requirement (cutoff/selection)."
            )
        revision = atomistic_model_revision(model).revision_id
        lanes = partition_parameters(model)
    _require_reciprocal_binding(plan, model)
    reciprocal = (
        None
        if plan.reciprocal is None
        else prepare_reciprocal_distribution(
            plan.reciprocal, plan.ownership, plan.partition
        )
    )
    scientific_id = canonical_fingerprint(
        {
            "kind": "atomistic-domain-science",
            "model": None if model is None else revision,
            "terms": [term.scientific_id for term in plan.terms],
            "roles": [
                [role.role.value, role.term_key, role.requirement.requirement_id]
                for role in plan.roles
            ],
            "reciprocal": None
            if plan.reciprocal is None
            else plan.reciprocal.scientific_id,
            "cell": plan.partition.cell.cell_id,
            "dynamic_cell": plan.dynamic_cell,
        }
    )
    # Canonical definitions only: the model and the plan's domain terms
    # (their prepared terms, systems, exception weights, reciprocal
    # definitions). Placement parameters of these objects are static.
    numeric = eqx.filter((model, plan.terms, plan.reciprocal), eqx.is_inexact_array)
    hamiltonian_revision = NumericRevision(
        f"atomistic-domain-hamiltonian:{scientific_id}",
        {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(numeric)[0]
        },
    ).revision_id
    return PreparedAtomisticDomainRuntime(
        plan=plan,
        parameter_lane=lanes[0],
        state_lane=lanes[1],
        fixed_lane=lanes[2],
        reciprocal=reciprocal,
        terms=terms,
        model_revision_identity=IdentityWords.from_fingerprint(revision),
        runtime_identity=IdentityWords.of(
            {
                "kind": "prepared-atomistic-domain",
                "plan": plan.plan_id,
                "model": revision,
            }
        ),
        scientific_identity=IdentityWords.from_fingerprint(scientific_id),
        hamiltonian_revision_identity=IdentityWords.from_fingerprint(
            hamiltonian_revision
        ),
        model_role_index=index,
        training_bound=False,
        model_state_required=bool(jax.tree.leaves(lanes[1])),
    )


__all__ = [
    "AtomisticDomainCertificate",
    "AtomisticDomainEvaluation",
    "AtomisticDomainExceptionInputs",
    "AtomisticDomainExceptionRows",
    "AtomisticDomainExecutionStatus",
    "AtomisticDomainForceRecord",
    "AtomisticDomainLossGradient",
    "AtomisticDomainOwnedRows",
    "AtomisticDomainOwnerPredictions",
    "AtomisticDomainOwnerView",
    "AtomisticDomainPacket",
    "AtomisticDomainReplicatedInputs",
    "AtomisticDomainTermInputs",
    "AtomisticDomainTermRows",
    "AtomisticDomainTermView",
    "AtomisticDomainTransition",
    "owner_parameter_partials",
    "owner_predictions",
    "prepare_atomistic_domain",
    "PreparedAtomisticDomainRuntime",
]
