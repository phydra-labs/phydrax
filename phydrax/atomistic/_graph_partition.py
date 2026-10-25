#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Replicated-geometry graph partition of owner-local atomistic execution.

The graph-partition strategy replicates the admitted geometry inside every
owner region while partitioning receiver and model work by the canonical
point ownership. Owner ``r`` keeps receivers on its owned rows
``[r * L, (r + 1) * L)`` (``L = local_capacity``) and addresses every source
by its global owner-blocked row ``q = owner * L + slot``, so the source column
space of every owner is the full owner-blocked row space of
``C = owner_count * L`` rows.

Topology preparation gathers the owner-blocked stored positions, activity and
stable IDs once per epoch and runs one bounded native image route search per
role (fractional cell list or Morton cluster tiles) over that replicated
geometry, restricted to this owner's receivers by stable ID. The search
enumerates the complete integer image stencil of ``cutoff + skin`` under the
current cell, keeps repeated images of one source and nonzero self images as
distinct routes, and stores routes in canonical receiver-major
``(receiver id, source id, lexicographic n)`` order with
``d = x_receiver - x_source + n @ H``. Declared nearest-neighbor selection is
applied per evaluation by the domain runtime over these complete candidates.

Per-layer source payloads cross owners through `ReplicatedColumnExchange`: a
tiled all-gather of owned rows into the ``C`` columns and, as its exact linear
transpose, a tiled reduce-scatter that returns every column cotangent to its
owning row exactly once without any owner materializing the reduced
full-length buffer. Global system conditions (charge normalization, composition
or channel-balance reductions) use the strategy-neutral system stages of the
layered executor; nothing here is model specific.

Every replicated buffer is charged against ``replicated_memory_bytes``:
the replicated search geometry of each role topology and each exchanged column
buffer (positions and every layer payload). Exceeding the bound is a typed
`AtomisticGraphPartitionResourceError` refusal at preparation or trace time,
distinct from scientific failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.distributed import JaxCollectiveProvider
from ..discretization import (
    AbstractImageRouteSearch,
    ImageReceiverSubset,
    ParticleImageCapacity,
)
from ..discretization._periodic_cell import lattice_right_inverse_with_status
from ..discretization.particle._distributed import FractionalOwnerPartition
from ..discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ._domain_state import (
    AtomisticDomainStrategy,
    AtomisticDomainTopology,
    AtomisticDomainTopologyEvidence,
    prepare_domain_relations,
)
from ._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticClusterTileImageSearch,
)


if TYPE_CHECKING:
    from ._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
    from ._feature_execution import NodeState


_AMBIENT_DIMENSION = 3
_SEARCH_RANK_BYTES = np.dtype(np.int32).itemsize
_ACTIVE_BYTES = np.dtype(np.bool_).itemsize


def _positive_int(name: str, value: int, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _row_bytes(leaf: Array | jax.ShapeDtypeStruct, rows: int, /) -> int:
    """Bytes of one leading-axis row of a node leaf with ``rows`` leading rows."""
    shape = tuple(leaf.shape)
    if not shape or shape[0] != rows:
        raise ValueError(
            f"Node payload leaves must lead with {rows} owned rows; got shape {shape}."
        )
    return int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(leaf.dtype).itemsize


def _column_bytes(payload: NodeState, local_capacity: int, column_count: int, /) -> int:
    """Bytes of the replicated column buffer of one exchanged node payload."""
    return sum(
        column_count * _row_bytes(leaf, local_capacity)
        for leaf in jax.tree.leaves(payload)
    )


def _geometry_bytes(rows: int, position_dtype: np.dtype, id_dtype: np.dtype, /) -> int:
    """Replicated search geometry of one role topology.

    Gathered stored positions ``(C, 3)``, activity ``(C,)``, stable IDs
    ``(C,)`` and the int32 order-preserving search ranks ``(C,)``.
    """
    return rows * (
        _AMBIENT_DIMENSION * np.dtype(position_dtype).itemsize
        + _ACTIVE_BYTES
        + np.dtype(id_dtype).itemsize
        + _SEARCH_RANK_BYTES
    )


@final
class AtomisticGraphPartitionMemoryEvidence(StrictModule, NonTrainableState):
    """Exact replicated-buffer charges of graph-partition execution.

    ``charges`` names every replicated buffer with its exact byte count:
    ``"search-geometry"`` (gathered positions, activity, stable IDs and int32
    search ranks of one role topology), ``"positions"`` (the gathered
    coordinate columns of one evaluation) and ``"layer-k"`` (the gathered
    source payload columns of interaction ``k``; its reverse cotangent buffer
    has the same size and is charged by the same bound). Each charge is
    admitted individually against ``replicated_memory_bytes``.
    """

    charges: tuple[tuple[str, int], ...] = eqx.field(static=True)
    replicated_rows: int = eqx.field(static=True)
    replicated_memory_bytes: int = eqx.field(static=True)
    maximum_charge_bytes: int = eqx.field(static=True)
    admitted: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        charges: Sequence[tuple[str, int]],
        /,
        *,
        replicated_rows: int,
        replicated_memory_bytes: int,
    ) -> None:
        entries = tuple((str(name), int(size)) for name, size in charges)
        if not entries:
            raise ValueError("Graph-partition memory evidence needs at least one charge.")
        names = [name for name, _ in entries]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Replicated buffer charges need distinct nonempty names.")
        if any(size < 0 for _, size in entries):
            raise ValueError("Replicated buffer charges must be nonnegative.")
        rows = _positive_int("replicated_rows", replicated_rows)
        capacity = _positive_int("replicated_memory_bytes", replicated_memory_bytes)
        maximum = max(size for _, size in entries)
        self.charges = entries
        self.replicated_rows = rows
        self.replicated_memory_bytes = capacity
        self.maximum_charge_bytes = maximum
        self.admitted = maximum <= capacity
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "atomistic-graph-partition-memory-evidence",
                "charges": [[name, size] for name, size in entries],
                "replicated_rows": rows,
                "replicated_memory_bytes": capacity,
            }
        )

    @property
    def refused(self) -> tuple[tuple[str, int], ...]:
        """Charges exceeding ``replicated_memory_bytes``."""
        return tuple(
            (name, size)
            for name, size in self.charges
            if size > self.replicated_memory_bytes
        )


class AtomisticGraphPartitionResourceError(MemoryError):
    """Typed replicated-memory refusal retaining its exact charges."""

    evidence: AtomisticGraphPartitionMemoryEvidence

    def __init__(self, evidence: AtomisticGraphPartitionMemoryEvidence, /) -> None:
        self.evidence = evidence
        refused = ", ".join(f"{name}={size}" for name, size in evidence.refused)
        super().__init__(
            "Graph-partition replicated buffers exceed replicated_memory_bytes="
            f"{evidence.replicated_memory_bytes}: {refused}."
        )


def _require_admitted(evidence: AtomisticGraphPartitionMemoryEvidence, /) -> None:
    if not evidence.admitted:
        raise AtomisticGraphPartitionResourceError(evidence)


@final
class ReplicatedColumnExchange(StrictModule):
    """Owned rows to the full owner-blocked column space, and its exact adjoint.

    Inside one mapped owner region, `gather` all-gathers every owned node leaf
    ``(L, ...)`` into owner-major columns ``(owner_count * L, ...)``; column
    ``q`` is row ``q % L`` of owner ``q // L``. `transpose` reduce-scatters
    column contributions so owner ``r`` receives the owner sum of its own
    column block exactly once. Both are native linear collectives whose
    registered transposes are each other, so `gather` is differentiable in
    reverse and forward mode (HVP and mixed derivatives) with VJP equal to
    `transpose`. Every exchanged column buffer is charged against
    ``replicated_memory_bytes`` at trace time.
    """

    axis_name: str = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    replicated_memory_bytes: int = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)

    def __init__(
        self,
        ownership: DistributedOwnershipPlan,
        /,
        *,
        replicated_memory_bytes: int,
    ) -> None:
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        capacity = _positive_int("replicated_memory_bytes", replicated_memory_bytes)
        self.axis_name = ownership.axis_name
        self.owner_count = ownership.owner_count
        self.local_capacity = ownership.local_capacity
        self.replicated_memory_bytes = capacity
        self.ownership_id = ownership.plan_id

    @property
    def column_count(self) -> int:
        return self.owner_count * self.local_capacity

    def message_bytes(self, owned: NodeState, /) -> int:
        """Static bytes of the replicated column buffer of one exchange."""
        return _column_bytes(owned, self.local_capacity, self.column_count)

    def _charge(self, name: str, owned: NodeState, /) -> None:
        _require_admitted(
            AtomisticGraphPartitionMemoryEvidence(
                ((name, self.message_bytes(owned)),),
                replicated_rows=self.column_count,
                replicated_memory_bytes=self.replicated_memory_bytes,
            )
        )

    def gather(self, owned: NodeState, /) -> NodeState:
        """Owner-major columns of every owned leaf (tiled all-gather)."""
        self._charge("gather", owned)
        provider = JaxCollectiveProvider(self.axis_name)
        return jax.tree.map(
            lambda leaf: provider.all_gather(leaf, axis=0, tiled=True), owned
        )

    def transpose(self, columns: NodeState, owned: NodeState, /) -> NodeState:
        """Return column cotangents to owned rows exactly once (reduce-scatter).

        ``owned`` supplies the primal row structure; integer leaves have no
        cotangent and return ``float0`` zeros.
        """
        self._charge("transpose", owned)
        provider = JaxCollectiveProvider(self.axis_name)
        count = self.column_count

        def scatter(cotangent: Array, primal: Array) -> Array | np.ndarray:
            if not jnp.issubdtype(primal.dtype, jnp.inexact):
                return np.zeros(primal.shape, dtype=jax.dtypes.float0)
            expected = (count,) + tuple(primal.shape[1:])
            if tuple(cotangent.shape) != expected:
                raise ValueError(
                    f"Column cotangents must have shape {expected}; got "
                    f"{tuple(cotangent.shape)}."
                )
            return provider.reduce_scatter(
                jnp.asarray(cotangent, dtype=primal.dtype), scatter_axis=0
            )

        return jax.tree.map(scatter, columns, owned)


type GraphPartitionSearchPolicy = (
    AtomisticCellListImageSearch | AtomisticClusterTileImageSearch
)


@final
class AtomisticGraphPartitionPlan(StrictModule, NonTrainableState):
    """Bounded replicated-geometry receiver partition of a domain plan.

    Binds the domain's ownership, fractional partition and roles to one
    host-prepared bounded native image search per role over the replicated
    owner-blocked rows (``particle_capacity = owner_count * local_capacity``),
    so topology preparation is traceable. ``search`` is the declared scalable
    route (fractional cell list or Morton cluster tiles; the dense reference
    is not a graph-partition route). Search capacities: cell/tile occupancy
    ``maximum_particles_per_cell``, ``role.edge_capacity`` stored routes per
    owner, ``receiver_edge_capacity`` stored routes per receiver atom, and the
    role's complete image stencil count. ``replicated_memory_bytes`` bounds
    every replicated buffer (`AtomisticGraphPartitionMemoryEvidence`).

    Roles are `AtomisticDomainRolePlan` directed roles (every image route, no
    receiver subset); every atom is a receiver of exactly its owner.
    """

    searches: tuple[AbstractImageRouteSearch, ...]
    search: GraphPartitionSearchPolicy
    role_ids: tuple[str, ...] = eqx.field(static=True)
    replicated_memory_bytes: int = eqx.field(static=True)
    receiver_edge_capacity: int = eqx.field(static=True)
    maximum_particles_per_cell: int = eqx.field(static=True)
    ownership_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ownership: DistributedOwnershipPlan,
        partition: FractionalOwnerPartition,
        roles: Sequence[AtomisticDomainRolePlan],
        /,
        *,
        replicated_memory_bytes: int,
        receiver_edge_capacity: int,
        maximum_particles_per_cell: int,
        search: GraphPartitionSearchPolicy,
    ) -> None:
        if not isinstance(ownership, DistributedOwnershipPlan):
            raise TypeError("ownership must be a DistributedOwnershipPlan.")
        if not isinstance(partition, FractionalOwnerPartition):
            raise TypeError("partition must be a FractionalOwnerPartition.")
        if not isinstance(
            search, (AtomisticCellListImageSearch, AtomisticClusterTileImageSearch)
        ):
            raise TypeError(
                "search must be an AtomisticCellListImageSearch or "
                "AtomisticClusterTileImageSearch; the dense reference is not a "
                "bounded graph-partition route."
            )
        if partition.owner_count != ownership.owner_count:
            raise ValueError(
                "The fractional partition must declare one region per owner."
            )
        role_plans = tuple(roles)
        if not role_plans:
            raise ValueError("A graph partition needs at least one role.")
        role_ids = tuple(role.role_id for role in role_plans)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("Graph-partition roles must have distinct role ids.")
        capacity_bytes = _positive_int("replicated_memory_bytes", replicated_memory_bytes)
        degree = _positive_int("receiver_edge_capacity", receiver_edge_capacity)
        occupancy = _positive_int(
            "maximum_particles_per_cell", maximum_particles_per_cell
        )
        cell = partition.cell
        if cell.vectors.shape[-1] != _AMBIENT_DIMENSION:
            raise ValueError(
                "Graph-partition atomistics requires a three-dimensional cell."
            )
        if any(role.stencil.cell_id != cell.cell_id for role in role_plans):
            raise ValueError("A role image stencil belongs to another PeriodicCell.")
        # Host preparation of the immutable reference cell, once per plan.
        reference = np.asarray(jax.device_get(cell.vectors))[None]
        periodic = np.asarray(cell.periodic_axes, dtype=np.bool_)[None]
        particles = ownership.total_capacity
        searches = tuple(
            search.search(
                role.search_radius,
                reference,
                periodic,
                ParticleImageCapacity(
                    maximum_particles_per_cell=occupancy,
                    maximum_edges=role.edge_capacity,
                    maximum_degree=degree,
                    maximum_images=role.stencil.image_count,
                ),
                particle_capacity=particles,
            )
            for role in role_plans
        )
        self.searches = searches
        self.search = search
        self.role_ids = role_ids
        self.replicated_memory_bytes = capacity_bytes
        self.receiver_edge_capacity = degree
        self.maximum_particles_per_cell = occupancy
        self.ownership_id = ownership.plan_id
        self.partition_id = partition.partition_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-graph-partition-plan",
                "ownership": ownership.plan_id,
                "partition": partition.partition_id,
                "roles": list(role_ids),
                "searches": [item.search_id for item in searches],
                "search_policy": search.policy_id,
                "replicated_memory_bytes": capacity_bytes,
                "receiver_edge_capacity": degree,
                "maximum_particles_per_cell": occupancy,
            }
        )

    def require_compatible(
        self,
        ownership: DistributedOwnershipPlan,
        partition: FractionalOwnerPartition,
        roles: Sequence[AtomisticDomainRolePlan],
        /,
    ) -> None:
        """Refuse a domain plan whose ownership, partition or roles differ."""
        if ownership.plan_id != self.ownership_id:
            raise ValueError("The graph partition was prepared for another ownership.")
        if partition.partition_id != self.partition_id:
            raise ValueError("The graph partition was prepared for another partition.")
        if tuple(role.role_id for role in roles) != self.role_ids:
            raise ValueError("The graph partition was prepared for other roles.")

    def role_search(self, role_id: str, /) -> AbstractImageRouteSearch:
        """Prepared bounded search of one role."""
        if role_id not in self.role_ids:
            raise ValueError("The role is not part of this graph partition.")
        return self.searches[self.role_ids.index(role_id)]

    def exchange(
        self, ownership: DistributedOwnershipPlan, /
    ) -> ReplicatedColumnExchange:
        """The replicated column exchange of this plan's ownership."""
        if ownership.plan_id != self.ownership_id:
            raise ValueError("The graph partition was prepared for another ownership.")
        return ReplicatedColumnExchange(
            ownership, replicated_memory_bytes=self.replicated_memory_bytes
        )

    def geometry_evidence(
        self,
        ownership: DistributedOwnershipPlan,
        position_dtype: np.dtype | type,
        id_dtype: np.dtype | type,
        /,
    ) -> AtomisticGraphPartitionMemoryEvidence:
        """Replicated search-geometry charge of one role topology."""
        rows = ownership.total_capacity
        return AtomisticGraphPartitionMemoryEvidence(
            (
                (
                    "search-geometry",
                    _geometry_bytes(rows, np.dtype(position_dtype), np.dtype(id_dtype)),
                ),
            ),
            replicated_rows=rows,
            replicated_memory_bytes=self.replicated_memory_bytes,
        )


def graph_partition_memory_evidence(
    plan: AtomisticDomainPlan,
    /,
    *,
    position_dtype: np.dtype | type,
    id_dtype: np.dtype | type = np.int64,
    layer_payloads: Sequence[NodeState] = (),
) -> AtomisticGraphPartitionMemoryEvidence:
    """Exact replicated-buffer charges of one graph-partition evaluation.

    ``layer_payloads[k]`` is the owned source payload of interaction ``k``
    (arrays or ``jax.ShapeDtypeStruct`` leaves leading with ``local_capacity``
    rows). Charges are the role search geometry, the gathered position columns
    and every layer's gathered payload columns, each exactly as charged by
    topology preparation and `ReplicatedColumnExchange`.
    """
    graph = plan.graph_partition
    if graph is None:
        raise ValueError("The domain plan has no graph partition.")
    ownership = plan.ownership
    local = ownership.local_capacity
    rows = ownership.total_capacity
    position = np.dtype(position_dtype)
    if not np.issubdtype(position, np.floating):
        raise TypeError("position_dtype must be floating.")
    charges = [
        ("search-geometry", _geometry_bytes(rows, position, np.dtype(id_dtype))),
        ("positions", rows * _AMBIENT_DIMENSION * position.itemsize),
    ]
    charges.extend(
        (f"layer-{index}", _column_bytes(payload, local, rows))
        for index, payload in enumerate(layer_payloads)
    )
    return AtomisticGraphPartitionMemoryEvidence(
        charges,
        replicated_rows=rows,
        replicated_memory_bytes=graph.replicated_memory_bytes,
    )


@eqx.filter_jit
def _owner_routes(
    search: AbstractImageRouteSearch,
    ownership: DistributedOwnershipPlan,
    support_id: str,
    positions: Array,
    active: Array,
    stable_ids: Array,
    cell_vectors: Array,
    origin: Array,
    periodic: Array,
) -> tuple[Array, ...]:
    """Replicated bounded search of every owner restricted to its receivers.

    Each owner gathers the owner-blocked geometry, ranks stable IDs in an
    order-preserving int32 key space (active atoms by stable ID, then padding
    rows), and searches its own active rows as receivers over every active
    source. Ranks keep canonical route order identical to stable-ID order
    while padding rows can never alias an atom's identity.
    """
    axis = ownership.axis_name
    local = ownership.local_capacity
    rows = ownership.total_capacity
    search_arrays, search_structure = eqx.partition(search, eqx.is_array)

    def body(
        owned: Array,
        alive: Array,
        identifiers: Array,
        vectors: Array,
        shift_origin: Array,
        lattice_periodic: Array,
        prepared: PyTree[Array],
    ) -> tuple[Array, ...]:
        route_search = eqx.combine(prepared, search_structure)
        provider = JaxCollectiveProvider(axis)
        me = jax.lax.axis_index(axis).astype(jnp.int32)
        geometry = jax.lax.stop_gradient(provider.all_gather(owned, axis=0, tiled=True))
        usable = provider.all_gather(alive, axis=0, tiled=True)
        ids = provider.all_gather(identifiers, axis=0, tiled=True)
        row = jnp.arange(rows, dtype=jnp.int32)
        order = jnp.lexsort((row, ids, (~usable).astype(jnp.int32)))
        ranks = jnp.zeros((rows,), dtype=jnp.int32).at[order].set(row)
        receivers = ImageReceiverSubset(
            ranks.reshape((-1, local))[me][None, :], alive[None, :]
        )
        result = route_search.routes(
            geometry[None],
            usable[None],
            ranks[None],
            vectors[None],
            shift_origin[None],
            lattice_periodic[None],
            support_id=support_id,
            relation_schema_id=route_search.search_id,
            receivers=receivers,
        )
        relation = result.relation
        evidence = result.evidence
        valid = relation.valid
        column = jnp.where(valid, relation.source_indices, 0).astype(jnp.int32)
        receiver = jnp.where(valid, relation.receiver_indices - me * local, 0).astype(
            jnp.int32
        )
        shifts = jnp.where(valid[:, None], relation.image_shifts, 0).astype(jnp.int32)
        source_ids = jnp.where(valid, ids[column], 0).astype(jnp.int64)
        counts = jnp.stack(
            (
                evidence.required_cell_occupancy[0],
                evidence.required_edges[0],
                evidence.stored_edges[0],
                evidence.required_degree[0],
            )
        ).astype(jnp.int32)
        flags = jnp.stack(
            (
                evidence.capacity_failure[0],
                evidence.scientific_failure[0],
                evidence.stencil_overflow[0],
            )
        )
        return (
            receiver,
            column,
            shifts,
            valid,
            source_ids,
            counts[None],
            flags[None],
        )

    owner = PartitionSpec(axis)
    replicated = PartitionSpec()
    return ownership.map(
        body,
        (owner, owner, owner, replicated, replicated, replicated, replicated),
        (owner, owner, owner, owner, owner, owner, owner),
    )(positions, active, stable_ids, cell_vectors, origin, periodic, search_arrays)


def prepare_graph_partition_topology(
    plan: AtomisticDomainPlan,
    role: AtomisticDomainRolePlan,
    layout: DistributedPointLayout,
    positions: Array,
    cell_vectors: Array,
    epoch: Array,
    /,
) -> AtomisticDomainTopology:
    """Discover and prepare one role's graph-partition topology epoch.

    ``positions`` are owner-blocked stored Cartesian coordinates of
    ``layout``; ``cell_vectors`` the current row lattice. Routes are complete
    candidates within ``cutoff + skin`` under the current cell; ``columns``
    index global owner-blocked rows and ``halo`` is ``None``. Capacity
    failures (cell occupancy, route, receiver degree or stencil image
    capacity) and scientific failures (prepared cell-list envelope, domain,
    non-finite or singular geometry, receiver coverage, stale owners,
    duplicate stable IDs) are reported separately; replicated geometry above
    ``replicated_memory_bytes`` is refused before execution.
    """
    graph = plan.graph_partition
    if graph is None or plan.strategy is not AtomisticDomainStrategy.GRAPH_PARTITION:
        raise ValueError("The domain plan does not declare the graph-partition strategy.")
    ownership = plan.ownership
    if layout.plan.plan_id != ownership.plan_id:
        raise ValueError("The layout belongs to another ownership plan.")
    search = graph.role_search(role.role_id)
    rows = ownership.total_capacity
    stored = jnp.asarray(positions)
    if stored.shape != (rows, _AMBIENT_DIMENSION) or not jnp.issubdtype(
        stored.dtype, jnp.floating
    ):
        raise ValueError(
            f"positions must be floating owner-blocked rows of shape "
            f"{(rows, _AMBIENT_DIMENSION)}."
        )
    cell = plan.partition.cell
    vectors = jnp.asarray(cell_vectors, dtype=stored.dtype)
    if vectors.shape != cell.vectors.shape:
        raise ValueError("cell_vectors must match the prepared lattice shape.")
    memory = graph.geometry_evidence(ownership, stored.dtype, layout.stable_ids.dtype)
    _require_admitted(memory)
    epoch_ = jnp.asarray(epoch, dtype=jnp.int32)
    if epoch_.shape:
        raise ValueError("epoch must be a scalar.")
    receivers, columns, shifts, valid, source_ids, counts, flags = _owner_routes(
        search,
        ownership,
        f"atomistic-graph-partition:{role.role_id}",
        stored,
        layout.active,
        layout.stable_ids,
        vectors,
        cell.origin.astype(stored.dtype),
        cell.periodic_mask,
    )
    column_count = rows
    leaves, tree = prepare_domain_relations(
        ownership,
        role.streaming,
        role.relation_owner_id,
        column_count,
        receivers,
        columns,
        valid,
        layout.active,
        epoch_,
    )
    owner_epochs = jnp.asarray(layout.owner_epochs).astype(jnp.int32)
    owners_current = jnp.all(owner_epochs == owner_epochs[0])
    finite = jnp.all(jnp.where(layout.active[:, None], jnp.isfinite(stored), True))
    # Status-carrying lattice solve: a singular or non-finite current cell is a
    # reported scientific failure, never a runtime abort inside a trajectory.
    raw_inverse, solved = lattice_right_inverse_with_status(vectors)
    cell_admissible = jnp.all(solved) & jnp.all(jnp.isfinite(vectors))
    inverse = jnp.where(cell_admissible, raw_inverse, 0.0)
    reach = jnp.sqrt(jnp.sum(inverse * inverse, axis=0))
    # Wrapped-endpoint completeness margin of the role stencil in the current
    # cell: ``extents + 1 - excursion - R * reach`` with excursion one.
    margin = jnp.where(
        cell.periodic_mask,
        jnp.asarray(role.stencil.extents, dtype=stored.dtype)
        - role.search_radius * reach,
        jnp.inf,
    ).astype(stored.dtype)
    stencil_complete = jnp.all(margin > 0.0) & ~jnp.any(flags[:, 2])
    capacity_successful = ~jnp.any(flags[:, 0])
    scientific_successful = (
        ~jnp.any(flags[:, 1])
        & owners_current
        & finite
        & cell_admissible
        & stencil_complete
        & layout.stable_ids_unique
    )
    evidence = AtomisticDomainTopologyEvidence(
        successful=capacity_successful & scientific_successful,
        capacity_successful=capacity_successful,
        scientific_successful=scientific_successful,
        owners_current=owners_current,
        positions_finite=finite,
        cell_admissible=cell_admissible,
        stencil_complete=stencil_complete,
        stencil_margin=margin,
        maximum_search_load=jnp.max(counts[:, 0]),
        search_capacity=jnp.asarray(graph.maximum_particles_per_cell, dtype=jnp.int32),
        maximum_edge_count=jnp.max(counts[:, 1]),
        edge_capacity=jnp.asarray(role.edge_capacity, dtype=jnp.int32),
        maximum_degree=jnp.max(counts[:, 3]),
        halo_successful=jnp.asarray(True),
        refused_routes=jnp.sum(counts[:, 1] - counts[:, 2], dtype=jnp.int32),
        maximum_halo_load=jnp.asarray(0, dtype=jnp.int32),
        halo_capacity=jnp.asarray(0, dtype=jnp.int32),
        replicated_bytes=jnp.asarray(memory.maximum_charge_bytes, dtype=jnp.int64),
        replicated_capacity_bytes=jnp.asarray(
            graph.replicated_memory_bytes, dtype=jnp.int64
        ),
    )
    route_owners = (columns // ownership.local_capacity).astype(jnp.int32)
    route_slots = (columns % ownership.local_capacity).astype(jnp.int32)
    return AtomisticDomainTopology(
        route_owners=route_owners,
        route_slots=route_slots,
        receivers=receivers,
        shifts=shifts,
        valid=valid,
        columns=columns,
        source_ids=source_ids,
        relation_leaves=leaves,
        reference_positions=stored,
        reference_cell_vectors=vectors,
        owner_epochs=owner_epochs,
        epoch=epoch_,
        evidence=evidence,
        halo=None,
        column_count=column_count,
        relation_tree=tree,
        streaming_plan_id=role.streaming.plan_id,
        role_id=role.role_id,
        strategy=AtomisticDomainStrategy.GRAPH_PARTITION,
    )


__all__ = [
    "AtomisticGraphPartitionMemoryEvidence",
    "AtomisticGraphPartitionPlan",
    "AtomisticGraphPartitionResourceError",
    "ReplicatedColumnExchange",
    "graph_partition_memory_evidence",
    "prepare_graph_partition_topology",
]
