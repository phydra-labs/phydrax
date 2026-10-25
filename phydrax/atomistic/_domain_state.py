#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scientific state and per-role topology of distributed atomistic domains.

This module is the bottom of the domain-execution dependency graph: it holds
the data contracts shared by the domain plan, the halo and graph-partition
strategies, distributed dynamics and durability, and imports none of them.

Per-atom rows are owner-blocked in the canonical `DistributedPointLayout`
order: owner ``o`` holds rows ``[o * local_capacity, (o + 1) * local_capacity)``.
Stored Cartesian ``positions`` were wrapped into the cell that was current at
the last accepted topology epoch; ``positions + image_counts @ cell_vectors`` is
the unwrapped trajectory (row-cell convention). ``cell_vectors`` is a dynamic
replicated leaf: ownership is the declared fractional partition of the CURRENT
cell, never a cell-diagonal guess.

One `AtomisticDomainTopology` exists per declared neighborhood role. Its edges
``(receiver row, source column, n)`` carry the displacement
``d = x_receiver - x_source + n @ H`` relative to stored coordinates, so a
periodic image alias reuses the physical atom's column and repeated images of
one source keep their multiplicity as distinct edges. Each topology retains
every owner's prepared streamed relation and both geometric cotangent
reductions as owner-stacked leaves of ``relation_leaves`` until the next epoch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.tree_util import PyTreeDef

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..discretization.spatial._distributed_relations import (
    _bucket_ranks,
    _pack,
    DistributedHaloPlan,
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ..sparse import EdgeRelation, RelationExecutionPlan, RelationExecutionState
from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationPlan
from ..typing import Bool, Dim, Float, Int32, Int64, PRNGKey, Scalar, UInt32
from ._conditioning import AtomisticConditioning
from ._dynamics import AtomisticControllerState, AtomisticEnergyLedgerState


class _OwnerRowDim(Dim, minimum=1):
    """Owner-blocked atom rows: ``owner_count * local_capacity``."""


class _OwnerEdgeDim(Dim, minimum=1):
    """Owner-blocked edge routes: ``owner_count * edge_capacity``."""


class _OwnerCountDim(Dim, minimum=1):
    """Owners of one ownership plan."""


class _LatticeRankDim(Dim, minimum=1):
    """Lattice rank of the periodic cell."""


class _ConstraintRowDim(Dim, minimum=1):
    """Owner-blocked constraint rows: ``owner_count * row_capacity``."""


class AtomisticDomainStrategy(StrEnum):
    """Spatial execution strategy of one domain plan.

    ``HALO`` partitions geometry and receivers and exchanges per-layer source
    features through role-specific physical halos. ``GRAPH_PARTITION``
    replicates the admitted geometry/topology search and partitions receiver
    work, gathering features through replicated column exchanges.
    """

    HALO = "halo"
    GRAPH_PARTITION = "graph-partition"


@final
class AtomisticDomainTopologyEvidence(StrictModule, NonTrainableState):
    """Capacity, scientific coverage, and identity evidence of one role epoch.

    Capacity failures (``maximum_search_load > search_capacity``, edge or halo
    overflow, refused halo routes, replicated memory) are reported separately
    from scientific failures (non-finite coordinates, stale owners, a singular
    current cell, or an image stencil that does not cover ``cutoff + skin`` in
    the current cell). ``maximum_search_load`` is the alias packet load of the
    halo strategy and the candidate/cell occupancy of the graph partition.
    """

    __strict_contract__ = True

    successful: Bool[Scalar]
    capacity_successful: Bool[Scalar]
    scientific_successful: Bool[Scalar]
    owners_current: Bool[Scalar]
    positions_finite: Bool[Scalar]
    cell_admissible: Bool[Scalar]
    stencil_complete: Bool[Scalar]
    stencil_margin: Float[_LatticeRankDim]
    maximum_search_load: Int32[Scalar]
    search_capacity: Int32[Scalar]
    maximum_edge_count: Int32[Scalar]
    edge_capacity: Int32[Scalar]
    maximum_degree: Int32[Scalar]
    halo_successful: Bool[Scalar]
    refused_routes: Int32[Scalar]
    maximum_halo_load: Int32[Scalar]
    halo_capacity: Int32[Scalar]
    replicated_bytes: Int64[Scalar]
    replicated_capacity_bytes: Int64[Scalar]


@final
class AtomisticDomainTopology(StrictModule, NonTrainableState):
    """One accepted per-role image graph epoch.

    Owner-blocked edge routes name their receiver row and the physical source
    atom by ``(route_owners, route_slots)`` with integer image ``shifts``;
    ``columns`` index the owner-local source column space of ``column_count``
    columns (owned rows plus deduplicated halo columns for ``HALO``; global
    owner-blocked rows for ``GRAPH_PARTITION``). ``valid`` is the executable
    edge validity. ``source_ids`` are the physical sources' stable IDs, used by
    declared nearest-neighbor selection tie-breaks. ``reference_positions`` and
    ``reference_cell_vectors`` are the stored coordinates and cell the
    candidate edges were certified for; ``owner_epochs`` is the layout epoch
    witness the topology was discovered for. Statics are layout: the column
    space size, relation structure, streamed tiling, role configuration and
    strategy.
    """

    __strict_contract__ = True

    route_owners: Int32[_OwnerEdgeDim]
    route_slots: Int32[_OwnerEdgeDim]
    receivers: Int32[_OwnerEdgeDim]
    shifts: Int32[_OwnerEdgeDim, _LatticeRankDim]
    valid: Bool[_OwnerEdgeDim]
    columns: Int32[_OwnerEdgeDim]
    source_ids: Int64[_OwnerEdgeDim]
    relation_leaves: tuple[Array, ...]
    reference_positions: Float[_OwnerRowDim, Literal[3]]
    reference_cell_vectors: Float[_LatticeRankDim, Literal[3]]
    owner_epochs: Int32[_OwnerCountDim]
    epoch: Int32[Scalar]
    evidence: AtomisticDomainTopologyEvidence
    halo: DistributedHaloPlan | None
    column_count: int = eqx.field(static=True)
    relation_tree: PyTreeDef = eqx.field(static=True)
    streaming_plan_id: str = eqx.field(static=True)
    role_id: str = eqx.field(static=True)
    strategy: AtomisticDomainStrategy = eqx.field(static=True)


@final
class AtomisticDomainForceCache(StrictModule):
    """Owner-blocked force cache bound to the step it was evaluated for.

    ``strain_derivative`` is ``dE/d strain`` (row cell ``H' = H @ F.T`` at
    fixed fractional coordinates and integer images) and ``stress`` the
    tension-positive ``sym(dE/d strain) / V``; both are NaN unless
    ``strain_available``. The cache is valid iff ``step == state.step_index``
    and ``successful``.
    """

    __strict_contract__ = True

    forces: Float[_OwnerRowDim, Literal[3]]
    energy: Float[Scalar]
    owner_energies: Float[_OwnerCountDim]
    strain_derivative: Float[Literal[3], Literal[3]]
    stress: Float[Literal[3], Literal[3]]
    strain_available: Bool[Scalar]
    step: Int32[Scalar]
    successful: Bool[Scalar]


@final
class AtomisticDomainConstraintState(StrictModule):
    """Owner-blocked distance-constraint rows (data only).

    Algorithms (projection, migration, directory routing) are owned by
    ``_domain_dynamics``. ``rows`` is the constraint-row ownership on the atom
    group and address plan with ``local_capacity = row_capacity``; its stable
    IDs are global constraint-row IDs. ``atoms`` are endpoint stable atom IDs
    and ``components`` the component anchor (minimum member stable ID; the row
    owner is the anchor atom's owner), ``-1`` on padding. ``halo`` routes rows
    ``2 r`` and ``2 r + 1`` to the left/right endpoints of local row ``r`` over
    the ATOM ownership. Statics are layout: directory and migration
    capacities and ``plan_id``, the configuration fingerprint of the
    constraint plan (method settings, capacities and the constraint count,
    which is the row layout's logical count).
    """

    __strict_contract__ = True

    rows: DistributedPointLayout
    atoms: Int64[_ConstraintRowDim, Literal[2]]
    components: Int64[_ConstraintRowDim]
    targets: Float[_ConstraintRowDim]
    lagrange: Float[_ConstraintRowDim]
    halo: DistributedHaloPlan
    position_residual: Float[Scalar]
    velocity_residual: Float[Scalar]
    topology_epoch: Int32[Scalar]
    successful: Bool[Scalar]
    directory_capacity: int = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class AtomisticDomainState(StrictModule):
    """Complete accepted continuation state of one distributed periodic system.

    Owner-blocked per-atom leaves: ``positions`` (stored, wrapped at the last
    epoch), ``momenta``, ``masses`` (zero on padding), model ``species``
    indices, integer ``image_counts``, the force cache rows and every
    ``atom_payload`` entry (per-atom thermostat/bias/history rows). Replicated
    leaves: ``cell_vectors``, ``replicated_payload``, typed P5 ``controllers``
    and energy ``ledger``, the typed ``rng_key``, ``step_index``, ``time`` and
    case-level ``conditioning``; ``case_identity_words`` are the eight uint32
    words of the system's semantic case address (route identities and keyed
    random addressing are rank- and layout-independent). ``constraints`` carry
    their own row ownership. ``topologies`` align with the plan's roles.
    Migration and topology change happen only through the prepared runtime's
    ``rebuild``.

    ``plan_identity``, ``model_revision_identity`` and ``run_identity`` are
    the exact canonical digests of the domain plan, the bound model revision
    and the run, carried as dynamic `IdentityWords` (``plan_id``,
    ``model_revision_id`` and ``run_id`` decode them on the host), so states of
    every material and parameter revision of one layout share a treedef; the
    runtime compares them with `IdentityWords.equals` inside compiled code.
    """

    __strict_contract__ = True

    layout: DistributedPointLayout
    positions: Float[_OwnerRowDim, Literal[3]]
    momenta: Float[_OwnerRowDim, Literal[3]]
    masses: Float[_OwnerRowDim]
    species: Int32[_OwnerRowDim]
    image_counts: Int32[_OwnerRowDim, _LatticeRankDim]
    cell_vectors: Float[_LatticeRankDim, Literal[3]]
    atom_payload: Mapping[str, Array]
    replicated_payload: Mapping[str, Array]
    controllers: AtomisticControllerState | None
    ledger: AtomisticEnergyLedgerState | None
    constraints: AtomisticDomainConstraintState | None
    force: AtomisticDomainForceCache
    rng_key: PRNGKey
    step_index: Int32[Scalar]
    time: Float[Scalar]
    topologies: tuple[AtomisticDomainTopology, ...]
    conditioning: AtomisticConditioning | None
    case_identity_words: UInt32[Literal[8]]
    plan_identity: IdentityWords
    model_revision_identity: IdentityWords
    run_identity: IdentityWords

    @property
    def plan_id(self) -> str:
        """Host decoding of the domain-plan digest this state belongs to."""
        return self.plan_identity.fingerprint()

    @property
    def model_revision_id(self) -> str:
        """Host decoding of the model revision this state was primed for."""
        return self.model_revision_identity.fingerprint()

    @property
    def run_id(self) -> str:
        """Host decoding of the run digest."""
        return self.run_identity.fingerprint()

    @property
    def force_cache_valid(self) -> Array:
        return (self.force.step == self.step_index) & self.force.successful

    def unwrapped_positions(self) -> Array:
        """Owner-blocked unwrapped coordinates ``x + image_counts @ H``."""
        return self.positions + self.image_counts.astype(
            self.positions.dtype
        ) @ self.cell_vectors.astype(self.positions.dtype)


def _relation_function(
    streaming: StreamedRelationPlan,
    relation_owner_id: str,
    column_count: int,
    receiver_count: int,
) -> Callable[
    [Array, Array, Array, Array, Array],
    tuple[PreparedStreamedRelation, RelationExecutionState, RelationExecutionState],
]:
    def prepare(
        receivers: Array, columns: Array, valid: Array, alive: Array, epoch: Array
    ) -> tuple[PreparedStreamedRelation, RelationExecutionState, RelationExecutionState]:
        relation = EdgeRelation(
            columns,
            receivers,
            source_size=column_count,
            target_size=receiver_count,
            valid=valid,
        )
        reduction = RelationExecutionPlan()
        return (
            streaming.prepare(
                relation,
                owner_id=relation_owner_id,
                epoch=epoch,
                receiver_valid=alive,
            ),
            reduction.prepare(relation),
            reduction.prepare(relation.transpose()),
        )

    return prepare


@eqx.filter_jit
def _owner_relation_leaves(
    ownership: DistributedOwnershipPlan,
    streaming: StreamedRelationPlan,
    relation_owner_id: str,
    column_count: int,
    receivers: Array,
    columns: Array,
    valid: Array,
    active: Array,
    epoch: Array,
) -> tuple[Array, ...]:
    edges = receivers.shape[0] // ownership.owner_count
    prepare = _relation_function(
        streaming, relation_owner_id, column_count, ownership.local_capacity
    )
    axis = ownership.axis_name

    def body(
        receiver: Array, column: Array, edge: Array, alive: Array, step: Array
    ) -> tuple[Array, ...]:
        leaves = jax.tree.leaves(prepare(receiver, column, edge, alive, step))
        return tuple(leaf[None] for leaf in leaves)

    shapes = jax.eval_shape(
        lambda: jax.tree.leaves(
            prepare(
                receivers[:edges],
                columns[:edges],
                valid[:edges],
                active[: ownership.local_capacity],
                epoch,
            )
        )
    )
    owner = PartitionSpec(axis)
    return ownership.map(
        body,
        (owner, owner, owner, owner, PartitionSpec()),
        tuple(owner for _ in shapes),
    )(receivers, columns, valid, active, epoch)


def prepare_domain_relations(
    ownership: DistributedOwnershipPlan,
    streaming: StreamedRelationPlan,
    relation_owner_id: str,
    column_count: int,
    receivers: Array,
    columns: Array,
    valid: Array,
    active: Array,
    epoch: Array,
    /,
) -> tuple[tuple[Array, ...], PyTreeDef]:
    """Prepare every owner's streamed relation and cotangent reductions once.

    ``receivers``/``columns``/``valid`` are owner-blocked edges of
    ``owner_count * edge_capacity`` rows; ``active`` the owner-blocked atom
    rows. Returns the owner-stacked relation leaves and the static structure
    of one owner's ``(PreparedStreamedRelation, receiver reduction, source
    reduction)``; both strategies share this exact preparation.
    """
    if receivers.shape[0] % ownership.owner_count:
        raise ValueError("Edge routes must be owner-blocked.")
    epoch_ = jnp.asarray(epoch, dtype=jnp.int32)
    leaves = _owner_relation_leaves(
        ownership,
        streaming,
        relation_owner_id,
        column_count,
        receivers,
        columns,
        valid,
        active,
        epoch_,
    )
    edges = receivers.shape[0] // ownership.owner_count
    prepare = _relation_function(
        streaming, relation_owner_id, column_count, ownership.local_capacity
    )
    shapes = jax.eval_shape(
        prepare,
        jax.ShapeDtypeStruct((edges,), jnp.int32),
        jax.ShapeDtypeStruct((edges,), jnp.int32),
        jax.ShapeDtypeStruct((edges,), jnp.bool_),
        jax.ShapeDtypeStruct((ownership.local_capacity,), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    return tuple(leaves), jax.tree.structure(shapes)


_INT64_MAX = int(np.iinfo(np.int64).max)


def owner_directory_lookup(
    ids: Array,
    active: Array,
    queries: Array,
    query_valid: Array,
    /,
    *,
    owners: int,
    capacity: int,
    axis: str,
) -> tuple[Array, Array, Array, Array, Array]:
    """Resolve stable IDs to ``(owner, slot)`` through a hashed directory.

    Inside one owner region: every owner registers its active IDs with bucket
    owner ``id mod owners`` and sends its queries there; bucket owners answer
    by an exact sorted search and return answers through the reverse
    all-to-all.  Registration and query packets hold ``capacity`` entries per
    (sender, bucket) pair; overflow is reported, never truncated silently.
    Returns owner, slot, found, the global maximum packet load, and the
    global count of fitted queries whose ID is absent. Absence is proven only
    when every owner registered all of its active IDs: after a registration
    overflow (reported through the load) an unanswered query is unresolved
    capacity, not a missing ID, and the count is zero.
    """
    provider = JaxCollectiveProvider(axis)
    local = ids.shape[0]
    identifiers = ids.astype(jnp.int64)
    bucket = jnp.mod(identifiers, owners).astype(jnp.int32)
    rank, loads = _bucket_ranks(bucket, active, owners)
    fits = active & (rank < capacity)
    registration_complete = provider.minimum(
        jnp.all(~active | fits).astype(jnp.int32)
    ).astype(jnp.bool_)
    registered = provider.all_to_all(
        _pack(identifiers, bucket, rank, fits, owners, capacity, -1),
        split_axis=0,
        concat_axis=0,
    ).reshape((-1,))
    registered_slots = provider.all_to_all(
        _pack(
            jnp.arange(local, dtype=jnp.int32), bucket, rank, fits, owners, capacity, -1
        ),
        split_axis=0,
        concat_axis=0,
    ).reshape((-1,))
    registered_owner = jnp.repeat(jnp.arange(owners, dtype=jnp.int32), capacity)
    key = jnp.where(registered >= 0, registered, _INT64_MAX)
    order = jnp.argsort(key, stable=True)
    sorted_key = key[order]
    sorted_owner = registered_owner[order]
    sorted_slot = registered_slots[order]

    asked = queries.astype(jnp.int64)
    query_bucket = jnp.mod(jnp.where(query_valid, asked, 0), owners).astype(jnp.int32)
    query_rank, query_loads = _bucket_ranks(query_bucket, query_valid, owners)
    query_fits = query_valid & (query_rank < capacity)
    received = provider.all_to_all(
        _pack(asked, query_bucket, query_rank, query_fits, owners, capacity, -1),
        split_axis=0,
        concat_axis=0,
    ).reshape((-1,))
    position = jnp.clip(
        jnp.searchsorted(sorted_key, received, side="left"), 0, sorted_key.shape[0] - 1
    )
    hit = (received >= 0) & (sorted_key[position] == received)
    answer_owner = jnp.where(hit, sorted_owner[position], -1).reshape((owners, capacity))
    answer_slot = jnp.where(hit, sorted_slot[position], -1).reshape((owners, capacity))
    back_owner = provider.all_to_all(answer_owner, split_axis=0, concat_axis=0)
    back_slot = provider.all_to_all(answer_slot, split_axis=0, concat_axis=0)
    safe_rank = jnp.clip(query_rank, 0, capacity - 1)
    owner = jnp.where(query_fits, back_owner[query_bucket, safe_rank], -1)
    slot = jnp.where(query_fits, back_slot[query_bucket, safe_rank], -1)
    found = query_fits & (owner >= 0)
    load = provider.maximum(jnp.maximum(jnp.max(loads), jnp.max(query_loads)))
    unresolved = provider.sum(jnp.sum(query_fits & ~found, dtype=jnp.int32))
    missing = jnp.where(registration_complete, unresolved, 0)
    return owner, slot, found, load.astype(jnp.int32), missing


__all__ = [
    "AtomisticDomainConstraintState",
    "AtomisticDomainForceCache",
    "AtomisticDomainState",
    "AtomisticDomainStrategy",
    "AtomisticDomainTopology",
    "AtomisticDomainTopologyEvidence",
    "owner_directory_lookup",
    "prepare_domain_relations",
]
