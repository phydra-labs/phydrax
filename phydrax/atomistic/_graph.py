#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState, PeriodicCell
from ..discretization.particle._image_neighborhood import (
    AbstractImageRouteSearch,
    CellListImageRouteSearch,
    DenseImageRouteSearch,
    image_certificate,
    ParticleImageCapacity,
    ParticleImageNeighborhoodState,
)
from ..discretization.particle._image_relation import ParticleImageRelationEvidence
from ..graph import GraphIR
from ..sparse import EdgeRelation
from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationPlan
from ..typing import parse
from ._system import PreparedAtomisticSystem
from ._types import AtomisticBatch


AtomisticGraphBackend: TypeAlias = Literal["dense", "particle"]


class AtomisticGraphExecutionPlan(StrictModule, NonTrainableState):
    """Resource and realization policy independent of learned model parameters.

    ``backend="dense"`` is the bounded named dense reference (finite pairs or,
    for periodic batches, pairs times a complete image stencil).
    ``backend="particle"`` uses scalable cell-list image search for prepared
    batch topologies and the particle neighborhood for runtime systems.
    ``image_capacity`` charges image-aware routes; ``streamed`` is the
    receiver-major streamed relation schedule plan prepared once per topology.
    """

    maximum_neighbors: int = eqx.field(static=True)
    maximum_dense_atoms: int | None = eqx.field(static=True)
    backend: AtomisticGraphBackend = eqx.field(static=True)
    image_capacity: ParticleImageCapacity | None
    streamed: StreamedRelationPlan
    maximum_candidate_slots: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_neighbors: int,
        /,
        *,
        backend: AtomisticGraphBackend = "dense",
        maximum_dense_atoms: int | None = None,
        image_capacity: ParticleImageCapacity | None = None,
        streamed: StreamedRelationPlan | None = None,
        maximum_candidate_slots: int = 10_000_000,
        plan_id: str | None = None,
    ) -> None:
        neighbors = int(maximum_neighbors)
        dense = None if maximum_dense_atoms is None else int(maximum_dense_atoms)
        slots = int(maximum_candidate_slots)
        if neighbors < 0:
            raise ValueError("maximum_neighbors must be non-negative.")
        if slots <= 0:
            raise ValueError("maximum_candidate_slots must be positive.")
        backend = parse(backend, AtomisticGraphBackend, "backend")
        if backend == "dense" and (dense is None or dense <= 0):
            raise ValueError(
                "Dense graph execution requires positive maximum_dense_atoms."
            )
        if backend == "particle" and dense is not None:
            raise ValueError(
                "maximum_dense_atoms is valid only for dense graph execution."
            )
        if image_capacity is not None and not isinstance(
            image_capacity, ParticleImageCapacity
        ):
            raise TypeError("image_capacity must be a ParticleImageCapacity or None.")
        schedule = StreamedRelationPlan() if streamed is None else streamed
        if not isinstance(schedule, StreamedRelationPlan):
            raise TypeError("streamed must be a StreamedRelationPlan or None.")
        generated = canonical_fingerprint(
            {
                "kind": "atomistic-graph-execution-plan",
                "maximum_neighbors": neighbors,
                "maximum_dense_atoms": dense,
                "backend": backend,
                "image_capacity": (
                    None if image_capacity is None else image_capacity.capacity_id
                ),
                "streamed": schedule.plan_id,
                "maximum_candidate_slots": slots,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.maximum_neighbors = neighbors
        self.maximum_dense_atoms = dense
        self.backend = backend
        self.image_capacity = image_capacity
        self.streamed = schedule
        self.maximum_candidate_slots = slots
        self.plan_id = identifier


class AtomisticGraphTopology(StrictModule, NonTrainableState):
    """Prepared directed route topology, separate from numerical geometry.

    Integer routes, image shifts ``n``, candidate membership, stable route
    order and the prepared streamed schedule are fixed for one
    topology epoch.  Binding positions and cell vectors only computes
    geometry, cutoff masks and certificates.  Atom arrays are flat case-major.
    ``stable_route_ids`` is the canonical receiver-major rank of every route.
    Batch topologies and runtime image-neighborhood topologies retain their
    reference build frame, so binding with moved positions or a deformed cell
    evaluates the stored-and-absent image certificate.  Runtime topologies
    (``execution_id=None``) carry the relation schema as ``topology_id``; their
    content is bound by array equality (``admit_supplied_topology``),
    never by identifier, and their streamed schedule plan must be the binding
    execution plan's.
    """

    senders: Array
    receivers: Array
    edge_cases: Array
    candidate_mask: Array
    image_shifts: Array
    stable_route_ids: Array
    atomic_numbers: Array
    atom_type_ids: Array
    masses: Array
    atom_mask: Array
    atom_cases: Array
    periodic_mask: Array
    epoch: Array
    overflow: Array
    evidence: ParticleImageRelationEvidence | None
    minimum_image_cell: PeriodicCell | None
    reference_cell_vectors: Array | None
    reference_positions: Array | None
    reference_lattice: Array | None
    lattice_origin: Array | None
    wrap_counts: Array | None
    stencil_extents: Array | None
    streamed: PreparedStreamedRelation
    case_count: int = eqx.field(static=True)
    atom_capacity: int = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    search_radius: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    atom_topology_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    execution_id: str | None = eqx.field(static=True)

    @property
    def edge_capacity(self) -> int:
        return int(self.senders.shape[0])

    @property
    def relation(self) -> EdgeRelation:
        size = self.case_count * self.atom_capacity
        return EdgeRelation(
            self.senders,
            self.receivers,
            source_size=size,
            target_size=size,
            valid=self.candidate_mask,
        )

    def prepare_streamed(self, plan: StreamedRelationPlan, /) -> PreparedStreamedRelation:
        """Return the epoch-bound streamed schedule for ``plan``.

        The schedule prepared with the topology is returned when ``plan``
        matches it; another plan prepares once from this topology's binding.
        """
        if self.streamed.plan.plan_id == plan.plan_id:
            return self.streamed
        prepared, _ = _bound_schedule(
            plan,
            self.relation,
            owner_id=self.owner_id,
            epoch=self.epoch,
            stable_route_ids=self.stable_route_ids,
            receiver_valid=self.atom_mask,
        )
        return prepared


def _bound_schedule(
    streamed: StreamedRelationPlan | PreparedStreamedRelation,
    relation: EdgeRelation,
    /,
    *,
    owner_id: str,
    epoch: Array,
    stable_route_ids: Array,
    receiver_valid: Array,
) -> tuple[PreparedStreamedRelation, Array]:
    """Prepare a schedule for one topology epoch, or admit one already prepared.

    An admitted schedule must name the same owner, endpoint extents and route
    capacity.  Its binding to *this* topology's content is checked numerically
    (no callback): epoch, receiver keys, active-route mask, stable route IDs and
    every scheduled lane's source must equal the topology's.  The returned
    status is false on any mismatch, so a schedule prepared for other routes of
    equal shape fails the topology instead of silently driving evaluation.
    """
    if isinstance(streamed, StreamedRelationPlan):
        prepared = streamed.prepare(
            relation,
            owner_id=owner_id,
            epoch=epoch,
            stable_route_ids=stable_route_ids,
            receiver_valid=receiver_valid,
        )
        return prepared, jnp.asarray(True)
    if not isinstance(streamed, PreparedStreamedRelation):
        raise TypeError(
            "streamed must be a StreamedRelationPlan or PreparedStreamedRelation."
        )
    if streamed.binding.owner_id != owner_id:
        raise ValueError("Streamed schedule belongs to another topology owner.")
    if (
        streamed.direction != "target"
        or streamed.route_shape != relation.route_shape
        or streamed.input_shape != relation.input_shape
        or streamed.output_shape != relation.output_shape
    ):
        raise ValueError(
            "Streamed schedule direction, route capacity or endpoint extents differ "
            "from the topology."
        )
    schedule = streamed.schedule
    groups = schedule.route_groups
    valid = relation.valid
    safe_receivers = jnp.where(valid, relation.target_indices, 0)
    safe_senders = jnp.where(valid, relation.source_indices, 0)
    active = valid & receiver_valid[safe_receivers]
    lane_valid = schedule.lane_valid
    lane_sources = jnp.where(lane_valid, safe_senders[schedule.lane_routes], 0)
    matches = (
        (streamed.binding.epoch == epoch)
        & jnp.all(groups.item_keys == safe_receivers)
        & jnp.all(groups.item_valid == active)
        & jnp.all(groups.stable_ids == stable_route_ids)
        & jnp.all(schedule.lane_sources == lane_sources)
    )
    return streamed, matches


def _build_topology(
    streamed: StreamedRelationPlan | PreparedStreamedRelation,
    /,
    **fields: Any,
) -> AtomisticGraphTopology:
    size = int(fields["case_count"]) * int(fields["atom_capacity"])
    relation = EdgeRelation(
        fields["senders"],
        fields["receivers"],
        source_size=size,
        target_size=size,
        valid=fields["candidate_mask"],
    )
    prepared, epoch_matches = _bound_schedule(
        streamed,
        relation,
        owner_id=fields["owner_id"],
        epoch=fields["epoch"],
        stable_route_ids=fields["stable_route_ids"],
        receiver_valid=fields["atom_mask"],
    )
    fields["overflow"] = fields["overflow"] | ~epoch_matches
    return AtomisticGraphTopology(streamed=prepared, **fields)


_STATIC_TOPOLOGY_FIELDS = (
    "case_count",
    "atom_capacity",
    "lattice_rank",
    "search_radius",
    "skin",
    "owner_id",
    "atom_topology_id",
    "topology_id",
    "execution_id",
)
_CONTENT_TOPOLOGY_FIELDS = (
    "senders",
    "receivers",
    "edge_cases",
    "candidate_mask",
    "image_shifts",
    "stable_route_ids",
    "atomic_numbers",
    "atom_type_ids",
    "masses",
    "atom_mask",
    "atom_cases",
    "periodic_mask",
    "reference_cell_vectors",
    "reference_positions",
    "reference_lattice",
    "lattice_origin",
    "wrap_counts",
    "stencil_extents",
)


def admit_supplied_topology(
    supplied: AtomisticGraphTopology, expected: AtomisticGraphTopology, /
) -> AtomisticGraphTopology:
    """Bind a supplied runtime topology to the one its neighborhood prepares now.

    ``expected`` is prepared from the current neighborhood, system and cell with
    ``supplied``'s epoch and streamed schedule, whose content binding is already
    folded into ``expected.overflow``.  Identity and capacities must agree on the
    host; route indices, masks, integer images, node metadata and the reference
    build frame are compared numerically (no callback), so routes of another
    search or epoch with coincident schema and shape fail the case instead of
    evaluating.  The returned topology is ``expected``.
    """
    if not isinstance(supplied, AtomisticGraphTopology):
        raise TypeError("topology must be an AtomisticGraphTopology.")
    supplied_cell = supplied.minimum_image_cell
    expected_cell = expected.minimum_image_cell
    if (
        any(
            getattr(supplied, name) != getattr(expected, name)
            for name in _STATIC_TOPOLOGY_FIELDS
        )
        or (supplied_cell is None) != (expected_cell is None)
        or (
            supplied_cell is not None
            and expected_cell is not None
            and supplied_cell.cell_id != expected_cell.cell_id
        )
    ):
        raise ValueError(
            "Supplied graph topology was not prepared from this neighborhood, system "
            "and cell."
        )
    matches = jnp.asarray(True)
    for name in _CONTENT_TOPOLOGY_FIELDS:
        left = getattr(supplied, name)
        right = getattr(expected, name)
        if left is None and right is None:
            continue
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Supplied graph topology {name} has another layout.")
        matches = matches & jnp.all(left == right)
    return eqx.tree_at(
        lambda value: value.overflow, expected, expected.overflow | ~matches
    )


class AtomisticGraph(StrictModule, NonTrainableState):
    """Fixed-capacity directed atomistic graph and complete failure evidence.

    ``overflow`` reports per-case capacity, topology and certificate failures;
    ``nonfinite`` reports per-case nonfinite active coordinates, runtime cell
    vectors or candidate geometry, independent of whether the case has edges.
    """

    graph: GraphIR
    neighbor_counts: Array
    edge_slots: Array
    maximum_neighbor_count: Array
    overflow: Array
    nonfinite: Array
    topology: AtomisticGraphTopology
    cell_vectors: Array
    candidate_mask: Array
    maximum_neighbors: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return ~(self.overflow | self.nonfinite)

    def require_success(self, value: ArrayLike, /) -> Array:
        return eqx.error_if(
            jnp.asarray(value),
            ~jnp.all(self.valid),
            "Atomistic graph overflow or nonfinite geometry; prediction is not valid.",
        )


def _receiver_major_rank(senders: Array, receivers: Array, /) -> Array:
    order = jnp.lexsort((senders, receivers))
    return (
        jnp.zeros(order.shape, dtype=jnp.int32)
        .at[order]
        .set(jnp.arange(order.shape[0], dtype=jnp.int32))
    )


def _active_edge_slots(
    receivers: Array, stable_route_ids: Array, active: Array, /
) -> Array:
    """Rank active edges within each receiver in the prepared route order.

    Uses the topology's receiver-major rank (a scatter, no sort): the result
    equals the historical lexsort-by-(receiver, sender) slot for every
    active edge and zero for inactive edges.
    """
    count = receivers.shape[0]
    if count == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    index = jnp.arange(count, dtype=jnp.int32)
    order = jnp.zeros((count,), dtype=jnp.int32).at[stable_route_ids].set(index)
    sorted_receivers = receivers[order]
    sorted_active = active[order].astype(jnp.int32)
    starts = jnp.where(
        (index == 0) | (sorted_receivers != jnp.roll(sorted_receivers, 1)), index, 0
    )
    group_start = jax.lax.associative_scan(jnp.maximum, starts)
    inclusive = jnp.cumsum(sorted_active)
    before_group = jnp.where(group_start > 0, inclusive[group_start - 1], 0)
    rank = inclusive - before_group - 1
    return (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[order]
        .set(jnp.where(sorted_active > 0, rank, 0))
    )


def _resolved_cell_vectors(
    topology: AtomisticGraphTopology,
    cell_vectors: ArrayLike | None,
    dtype: jnp.dtype,
    /,
) -> Array:
    if cell_vectors is None:
        if topology.lattice_rank == 0:
            return jnp.zeros((topology.case_count, 0, 3), dtype=dtype)
        if topology.reference_cell_vectors is None:
            raise ValueError("Image graph binding requires explicit cell_vectors.")
        return topology.reference_cell_vectors.astype(dtype)
    vectors = jnp.asarray(cell_vectors, dtype=dtype)
    if vectors.ndim == 2:
        vectors = vectors[None]
    if topology.lattice_rank > 0 and vectors.shape != (
        topology.case_count,
        topology.lattice_rank,
        3,
    ):
        raise ValueError(
            "cell_vectors must have shape (lattice rank, 3) or (case, lattice rank, 3)."
        )
    if topology.lattice_rank == 0 and (
        vectors.ndim != 3 or vectors.shape[0] != topology.case_count
    ):
        raise ValueError("cell_vectors must have one lattice per graph case.")
    return vectors


def _edge_displacement(
    topology: AtomisticGraphTopology, positions: Array, vectors: Array, /
) -> Array:
    raw = positions[topology.receivers] - positions[topology.senders]
    if topology.lattice_rank > 0:
        raw = raw + contract(
            "ei,eid->ed",
            topology.image_shifts.astype(positions.dtype),
            vectors[topology.edge_cases],
            backend="jax",
        )
    elif topology.minimum_image_cell is not None:
        cell = topology.minimum_image_cell
        raw = (
            cell.minimum_image(raw)
            if vectors.shape[1] == 0
            else cell.minimum_image_with_vectors(raw, vectors[0])
        )
    return jnp.where(topology.candidate_mask[:, None], raw, 0.0)


def _certificate_failure(
    topology: AtomisticGraphTopology,
    positions: Array,
    vectors: Array,
    cutoff: float,
    /,
) -> Array:
    if (
        topology.reference_positions is None
        or topology.reference_lattice is None
        or topology.lattice_origin is None
        or topology.wrap_counts is None
        or topology.stencil_extents is None
    ):
        return jnp.zeros((topology.case_count,), dtype=jnp.bool_)
    shape = (topology.case_count, topology.atom_capacity)
    rank = topology.reference_lattice.shape[1]
    reference_lattice = topology.reference_lattice.astype(positions.dtype)
    current_lattice = (
        jnp.where(topology.periodic_mask[:, :, None], vectors, reference_lattice)
        if topology.lattice_rank == rank
        else reference_lattice
    )
    certificate = image_certificate(
        topology.reference_positions.reshape(shape + (3,)),
        positions.reshape(shape + (3,)),
        reference_lattice,
        current_lattice,
        topology.lattice_origin,
        topology.wrap_counts.reshape(shape + (rank,)),
        topology.stencil_extents,
        topology.atom_mask.reshape(shape),
        topology.periodic_mask,
        cutoff,
    )
    return ~certificate.valid(topology.search_radius - cutoff)


def _node_metadata(
    stored: Array, current: ArrayLike | None, atom_count: int, name: str, /
) -> Array:
    if current is None:
        return stored
    value = jnp.asarray(current).reshape((-1,))
    if value.shape != (atom_count,):
        raise ValueError(f"{name} must have one flat case-major value per atom.")
    return value.astype(stored.dtype)


def _periodicity_mismatch(
    topology: AtomisticGraphTopology, periodic_axes: ArrayLike | None, /
) -> Array:
    """Per-case flag for runtime periodicity differing from the topology's.

    Finite and classical pair topologies (lattice rank zero) admit only fully
    nonperiodic runtime cases.  Concrete mismatches are refused on the host.
    """
    if periodic_axes is None:
        return jnp.zeros((topology.case_count,), dtype=jnp.bool_)
    declared = jnp.asarray(periodic_axes, dtype=jnp.bool_)
    if declared.ndim == 1:
        declared = declared[None]
    if declared.shape[0] != topology.case_count:
        raise ValueError("periodic_axes must have one row per graph case.")
    prepared = (
        topology.periodic_mask
        if topology.lattice_rank > 0
        else jnp.zeros(declared.shape, dtype=jnp.bool_)
    )
    if prepared.shape != declared.shape:
        raise ValueError("periodic_axes rank differs from the topology lattice rank.")
    mismatch = jnp.any(prepared != declared, axis=-1)
    if not isinstance(mismatch, jax.core.Tracer) and bool(np.any(mismatch)):
        raise ValueError(
            "Runtime periodic axes differ from the prepared topology; re-prepare the "
            "topology for the changed periodicity."
        )
    return mismatch


def bind_atomistic_graph(
    topology: AtomisticGraphTopology,
    execution: AtomisticGraphExecutionPlan,
    positions: ArrayLike,
    /,
    *,
    cutoff: float,
    cell_vectors: ArrayLike | None = None,
    periodic_axes: ArrayLike | None = None,
    atomic_numbers: ArrayLike | None = None,
    atom_type_ids: ArrayLike | None = None,
    masses: ArrayLike | None = None,
) -> AtomisticGraph:
    """Bind numerical geometry to a prepared topology.

    ``d_e = x[receiver] - x[sender] + n_e @ H[case_e]`` for image topologies
    (``H`` = explicit ``cell_vectors``, differentiable); the historical
    selected minimum image for classical periodic pair topologies; plain
    receiver-minus-sender otherwise.  No route discovery or sort occurs.

    ``periodic_axes`` (``(case, 3)``) declares the runtime periodicity; a case
    whose periodicity differs from the topology's is refused on concrete host
    input and reported in ``overflow`` when traced (cell deformation alone is
    admitted through the certificate).  ``atomic_numbers``, ``atom_type_ids``
    and ``masses`` (flat case-major) bind the current declared node metadata;
    omitted values publish the metadata stored at topology preparation.
    """
    if not isinstance(topology, AtomisticGraphTopology):
        raise TypeError("topology must be an AtomisticGraphTopology.")
    if not isinstance(execution, AtomisticGraphExecutionPlan):
        raise TypeError("execution must be an AtomisticGraphExecutionPlan.")
    if topology.execution_id is not None and topology.execution_id != execution.plan_id:
        raise ValueError("Topology was prepared under another graph execution plan.")
    if topology.streamed.plan.plan_id != execution.streamed.plan_id:
        raise ValueError(
            "Topology streamed schedule was prepared under another streamed plan than "
            "the graph execution plan."
        )
    cutoff_value = float(cutoff)
    if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if cutoff_value > topology.search_radius:
        raise ValueError(
            f"cutoff={cutoff_value} exceeds the topology candidate radius "
            f"{topology.search_radius}."
        )
    coordinate = jnp.asarray(positions)
    expected = (topology.case_count * topology.atom_capacity, 3)
    coordinate = (
        coordinate.reshape(expected) if coordinate.size == expected[0] * 3 else coordinate
    )
    atom_count = expected[0]
    node_numbers = _node_metadata(
        topology.atomic_numbers, atomic_numbers, atom_count, "atomic_numbers"
    )
    node_types = _node_metadata(
        topology.atom_type_ids, atom_type_ids, atom_count, "atom_type_ids"
    )
    node_masses = _node_metadata(topology.masses, masses, atom_count, "masses")
    periodicity_mismatch = _periodicity_mismatch(topology, periodic_axes)
    if coordinate.shape != expected:
        raise ValueError(f"positions must have {expected[0]} atoms in three dimensions.")
    vectors = _resolved_cell_vectors(topology, cell_vectors, coordinate.dtype)
    displacement = _edge_displacement(topology, coordinate, vectors)
    squared = jnp.sum(displacement * displacement, axis=-1)
    tiny = jnp.asarray(jnp.finfo(coordinate.dtype).tiny, dtype=coordinate.dtype)
    positive = jnp.sqrt(jnp.maximum(squared, tiny))
    distance = jnp.where(squared > 0.0, positive, 0.0)
    safe = jnp.where(distance > 0.0, distance, 1.0)
    direction = displacement / safe[:, None]
    edge_mask = topology.candidate_mask & (
        distance < jnp.asarray(cutoff_value, coordinate.dtype)
    )
    case_count = topology.case_count
    atom_capacity = topology.atom_capacity
    neighbor_counts = (
        jnp.zeros((case_count * atom_capacity,), dtype=jnp.int32)
        .at[topology.receivers]
        .add(edge_mask.astype(jnp.int32))
        .reshape((case_count, atom_capacity))
    )
    maximum_neighbor_count = jnp.max(neighbor_counts, axis=1)
    overflow = (
        (maximum_neighbor_count > execution.maximum_neighbors)
        | topology.overflow
        | _certificate_failure(topology, coordinate, vectors, cutoff_value)
        | periodicity_mismatch
    )
    atom_nonfinite = topology.atom_mask & ~jnp.all(jnp.isfinite(coordinate), axis=-1)
    edge_nonfinite = topology.candidate_mask & ~jnp.all(
        jnp.isfinite(displacement), axis=-1
    )
    nonfinite = (
        (
            jnp.zeros((case_count,), dtype=jnp.int32)
            .at[topology.atom_cases]
            .add(atom_nonfinite.astype(jnp.int32))
            > 0
        )
        | (
            jnp.zeros((case_count,), dtype=jnp.int32)
            .at[topology.edge_cases]
            .add(edge_nonfinite.astype(jnp.int32))
            > 0
        )
        | ~jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
    )
    active_edges = (
        jnp.zeros((case_count,), dtype=jnp.int32)
        .at[topology.edge_cases]
        .add(edge_mask.astype(jnp.int32))
    )
    edge_blocks = (
        jnp.zeros((case_count,), dtype=jnp.int32)
        .at[topology.edge_cases]
        .add(jnp.ones_like(topology.edge_cases))
    )
    graph = GraphIR(
        nodes={
            "atomic_numbers": node_numbers,
            "atom_type_ids": node_types,
            "masses": node_masses,
            "case_index": topology.atom_cases,
        },
        edges={
            "displacement": displacement,
            "distance": distance[:, None],
            "direction": direction,
            "case_index": topology.edge_cases,
        },
        senders=topology.senders,
        receivers=topology.receivers,
        globals={
            "active_edge_count": active_edges,
            "maximum_neighbor_count": maximum_neighbor_count,
            "overflow": overflow,
            "nonfinite": nonfinite,
        },
        n_node=jnp.full((case_count,), atom_capacity, dtype=jnp.int32),
        n_edge=edge_blocks,
        node_mask=topology.atom_mask,
        edge_mask=edge_mask,
        graph_mask=jnp.ones((case_count,), dtype=jnp.bool_),
    )
    edge_slots = _active_edge_slots(
        topology.receivers, topology.stable_route_ids, edge_mask
    )
    graph_id = canonical_fingerprint(
        {
            "kind": "atomistic-graph-realization",
            "topology": topology.topology_id,
            "execution": execution.plan_id,
            "cutoff": cutoff_value,
        }
    )
    return AtomisticGraph(
        graph=graph,
        neighbor_counts=neighbor_counts,
        edge_slots=edge_slots,
        maximum_neighbor_count=maximum_neighbor_count,
        overflow=overflow,
        nonfinite=nonfinite,
        topology=topology,
        cell_vectors=vectors,
        candidate_mask=topology.candidate_mask,
        maximum_neighbors=execution.maximum_neighbors,
        cutoff=cutoff_value,
        topology_id=topology.topology_id,
        execution_id=execution.plan_id,
        graph_id=graph_id,
    )


def _finite_dense_topology(
    batch: AtomisticBatch, execution: AtomisticGraphExecutionPlan, /
) -> AtomisticGraphTopology:
    dense_limit = execution.maximum_dense_atoms
    if execution.backend != "dense" or dense_limit is None:
        raise TypeError("Finite dense topology requires a dense execution plan.")
    if batch.atom_capacity > dense_limit:
        raise ValueError(
            f"Dense atomistic graph capacity {batch.atom_capacity} exceeds the explicit "
            f"maximum_dense_atoms={dense_limit} resource guard."
        )
    atom_capacity = batch.atom_capacity
    case_count = batch.case_count
    edge_capacity = atom_capacity * (atom_capacity - 1)
    local_senders = np.repeat(np.arange(atom_capacity, dtype=np.int32), atom_capacity - 1)
    local_offsets = np.tile(np.arange(atom_capacity - 1, dtype=np.int32), atom_capacity)
    local_receivers = local_offsets + (local_offsets >= local_senders)
    case_offsets = np.repeat(
        np.arange(case_count, dtype=np.int32) * atom_capacity, edge_capacity
    )
    senders = np.tile(local_senders, case_count) + case_offsets
    receivers = np.tile(local_receivers, case_count) + case_offsets
    order = np.lexsort((senders, receivers))
    rank = np.empty_like(order, dtype=np.int32)
    rank[order] = np.arange(order.shape[0], dtype=np.int32)
    flat_mask = batch.atom_mask.reshape((-1,))
    return _build_topology(
        execution.streamed,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        edge_cases=jnp.asarray(
            np.repeat(np.arange(case_count, dtype=np.int32), edge_capacity)
        ),
        candidate_mask=flat_mask[senders] & flat_mask[receivers],
        image_shifts=jnp.zeros((senders.shape[0], 0), dtype=jnp.int32),
        stable_route_ids=jnp.asarray(rank),
        atomic_numbers=batch.atomic_numbers.reshape((-1,)),
        atom_type_ids=batch.atom_type_ids.reshape((-1,)),
        masses=batch.masses.reshape((-1,)),
        atom_mask=flat_mask,
        atom_cases=batch.atom_cases,
        periodic_mask=jnp.zeros((case_count, 0), dtype=jnp.bool_),
        epoch=jnp.zeros((), dtype=jnp.int32),
        overflow=jnp.zeros((case_count,), dtype=jnp.bool_),
        evidence=None,
        minimum_image_cell=None,
        reference_cell_vectors=None,
        reference_positions=None,
        reference_lattice=None,
        lattice_origin=None,
        wrap_counts=None,
        stencil_extents=None,
        case_count=case_count,
        atom_capacity=atom_capacity,
        lattice_rank=0,
        search_radius=float("inf"),
        skin=0.0,
        atom_topology_id=batch.atom_topology_id,
        owner_id=canonical_fingerprint(
            {
                "kind": "atomistic-finite-dense-topology",
                "atoms": batch.atom_topology_id,
                "execution": execution.plan_id,
            }
        ),
        topology_id=batch.atom_topology_id,
        execution_id=execution.plan_id,
    )


def _case_binning_lattice(
    cell: np.ndarray,
    periodic: np.ndarray,
    positions: np.ndarray,
    active: np.ndarray,
    radius: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-rank search lattice whose periodic rows are the physical cell.

    Nonperiodic rows only bin candidates (their image shifts are zero); when
    the physical rows cannot bin the case (finite molecule, singular vacuum
    row, atoms outside the cell) they are replaced by an orthogonal bounding
    complement that covers every active atom plus ``radius``.
    """
    points = positions[active]
    periodic_rows = np.flatnonzero(periodic)
    if periodic_rows.size:
        lattice = cell[periodic_rows]
        singular_values = np.linalg.svd(lattice, compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps * 16 * singular_values[0]:
            raise ValueError("Periodic cell rows of a batch case are rank deficient.")
    if periodic_rows.size == 3:
        return cell, np.zeros((3,), dtype=np.float64)
    full = np.linalg.svd(cell, compute_uv=False)
    if periodic_rows.size and full[-1] > np.finfo(np.float64).eps * 16 * full[0]:
        fractional = points @ np.linalg.inv(cell)
        nonperiodic = fractional[:, ~periodic]
        if np.all((nonperiodic >= 0.0) & (nonperiodic < 1.0)):
            return cell, np.zeros((3,), dtype=np.float64)
    if periodic_rows.size:
        _, _, basis = np.linalg.svd(cell[periodic_rows])
        complement = basis[periodic_rows.size :]
    else:
        complement = np.eye(3)
    projection = points @ complement.T
    lower = projection.min(axis=0) - radius
    upper = projection.max(axis=0) + radius
    lattice = np.array(cell, dtype=np.float64, copy=True)
    for row, axis in enumerate(np.flatnonzero(~periodic)):
        lattice[axis] = complement[row] * (upper[row] - lower[row])
    return lattice, lower @ complement


def prepare_atomistic_graph_topology(
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    cutoff: float,
    skin: float = 0.0,
) -> AtomisticGraphTopology:
    """Prepare one batch topology on the host, outside differentiation.

    Finite batches under the dense backend keep the historical all-pairs
    layout.  Periodic batches (and every batch under the particle backend)
    search directed image routes within ``cutoff + skin`` with explicit
    integer shifts, case isolation and separately charged capacities: the
    particle backend uses the case-batched fractional cell list, the dense
    backend the bounded named all-pairs-times-stencil reference.  Positions
    and cells must be concrete.
    """
    if not isinstance(batch, AtomisticBatch):
        raise TypeError("batch must be an AtomisticBatch.")
    if not isinstance(execution, AtomisticGraphExecutionPlan):
        raise TypeError("execution must be an AtomisticGraphExecutionPlan.")
    cutoff_value = float(cutoff)
    skin_value = float(skin)
    if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if not np.isfinite(skin_value) or skin_value < 0.0:
        raise ValueError("skin must be finite and non-negative.")
    case_count = batch.case_count
    atom_capacity = batch.atom_capacity
    periodic = (
        np.zeros((case_count, 3), dtype=np.bool_)
        if batch.periodic_axes is None
        else np.asarray(batch.periodic_axes, dtype=np.bool_)
    )
    if execution.backend == "dense" and not np.any(periodic):
        return _finite_dense_topology(batch, execution)
    capacity = execution.image_capacity
    if capacity is None:
        raise ValueError(
            "Periodic or particle-backend graph topology requires image_capacity."
        )
    radius = cutoff_value + skin_value
    positions = np.asarray(batch.positions, dtype=np.float64)
    active = np.asarray(batch.atom_mask, dtype=np.bool_)
    cells = (
        np.zeros((case_count, 3, 3), dtype=np.float64)
        if batch.cells is None
        else np.asarray(batch.cells, dtype=np.float64)
    )
    lattices = np.empty((case_count, 3, 3), dtype=np.float64)
    origins = np.empty((case_count, 3), dtype=np.float64)
    for case in range(case_count):
        lattices[case], origins[case] = _case_binning_lattice(
            cells[case], periodic[case], positions[case], active[case], radius
        )
    search: AbstractImageRouteSearch
    if execution.backend == "dense":
        dense_limit = execution.maximum_dense_atoms
        if dense_limit is None or atom_capacity > dense_limit:
            raise ValueError(
                f"Dense image graph capacity {atom_capacity} exceeds the explicit "
                f"maximum_dense_atoms={dense_limit} resource guard."
            )
        search = DenseImageRouteSearch(
            radius,
            lattices,
            periodic,
            capacity,
            particle_capacity=atom_capacity,
            maximum_dense_routes=case_count
            * dense_limit
            * dense_limit
            * capacity.maximum_images,
        )
    else:
        search = CellListImageRouteSearch(
            radius,
            lattices,
            capacity,
            particle_capacity=atom_capacity,
            maximum_candidate_slots=execution.maximum_candidate_slots,
        )
    owner_id = canonical_fingerprint(
        {
            "kind": "atomistic-image-graph-topology",
            "atoms": batch.atom_topology_id,
            "execution": execution.plan_id,
            "search": search.search_id,
            "radius": radius,
        }
    )
    dtype = batch.positions.dtype
    result = eqx.filter_jit(search.routes)(
        batch.positions,
        batch.atom_mask,
        batch.particle_ids,
        jnp.asarray(lattices, dtype=dtype),
        jnp.asarray(origins, dtype=dtype),
        jnp.asarray(periodic),
        support_id=batch.atom_topology_id,
        relation_schema_id=owner_id,
    )
    relation = result.relation
    content = array_tree_fingerprint(
        {
            "senders": np.asarray(relation.source_indices),
            "receivers": np.asarray(relation.receiver_indices),
            "shifts": np.asarray(relation.image_shifts),
            "valid": np.asarray(relation.valid),
        }
    )
    return _build_topology(
        execution.streamed,
        senders=relation.source_indices,
        receivers=relation.receiver_indices,
        edge_cases=relation.route_cases,
        candidate_mask=relation.valid,
        image_shifts=relation.image_shifts,
        stable_route_ids=jnp.arange(relation.capacity, dtype=jnp.int32),
        atomic_numbers=batch.atomic_numbers.reshape((-1,)),
        atom_type_ids=batch.atom_type_ids.reshape((-1,)),
        masses=batch.masses.reshape((-1,)),
        atom_mask=batch.atom_mask.reshape((-1,)),
        atom_cases=batch.atom_cases,
        periodic_mask=jnp.asarray(periodic),
        epoch=jnp.zeros((), dtype=jnp.int32),
        overflow=~result.evidence.successful,
        evidence=result.evidence,
        minimum_image_cell=None,
        reference_cell_vectors=jnp.asarray(cells, dtype=dtype),
        reference_positions=batch.positions.reshape((-1, 3)),
        reference_lattice=jnp.asarray(lattices, dtype=dtype),
        lattice_origin=jnp.asarray(origins, dtype=dtype),
        wrap_counts=result.wrap_counts,
        stencil_extents=result.stencil_extents,
        case_count=case_count,
        atom_capacity=atom_capacity,
        lattice_rank=3,
        search_radius=radius,
        skin=skin_value,
        atom_topology_id=batch.atom_topology_id,
        owner_id=owner_id,
        topology_id=canonical_fingerprint({"owner": owner_id, "content": content}),
        execution_id=execution.plan_id,
    )


def particle_atomistic_graph_topology(
    system: PreparedAtomisticSystem,
    neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
    /,
    *,
    epoch: ArrayLike = 0,
    cell: PeriodicCell | None = None,
    streamed: StreamedRelationPlan | PreparedStreamedRelation,
) -> AtomisticGraphTopology:
    """Prepare one system's directed topology from a runtime neighborhood.

    An image neighborhood contributes its already canonical directed routes
    and explicit shifts with no sort, plus its search frame (reference
    positions, lattice, wrap counts and stencil extents), so binding moved
    positions or another runtime cell evaluates the stored-and-absent image
    certificate with skin ``search_radius - cutoff`` and fails the case outside
    the epoch.  A classical pair-once neighborhood is expanded into both
    directions and keeps the historical minimum-image geometry under ``cell``.
    ``streamed`` attaches a schedule prepared for this topology epoch, or
    prepares one from a plan.
    """
    if not isinstance(system, PreparedAtomisticSystem):
        raise TypeError("system must be a PreparedAtomisticSystem.")
    epoch_array = jnp.asarray(epoch, dtype=jnp.int32)
    if epoch_array.shape != ():
        raise ValueError("epoch must be a scalar.")
    capacity = system.capacity
    common = {
        "atomic_numbers": system.plan.atomic_numbers,
        "atom_type_ids": system.plan.atom_type_ids,
        "masses": system.plan.masses,
        "atom_mask": system.active_mask,
        "atom_cases": jnp.zeros((capacity,), dtype=jnp.int32),
        "epoch": epoch_array,
        "case_count": 1,
        "atom_capacity": capacity,
        "skin": 0.0,
        "execution_id": None,
        "atom_topology_id": system.prepared_id,
    }
    if isinstance(neighborhood, ParticleImageNeighborhoodState):
        relation = neighborhood.relation
        if cell is not None and (
            system.cell is None or cell.cell_id != system.cell.cell_id
        ):
            raise ValueError(
                "Image routes carry explicit shifts; only the system cell is admitted."
            )
        if relation.case_count != 1 or relation.particle_capacity != capacity:
            raise ValueError("Image neighborhood does not index this atomistic system.")
        if relation.support_id != system.particles.support.support_id:
            raise ValueError("Image neighborhood belongs to another particle support.")
        if system.cell is None or neighborhood.cell_vectors.shape[1:] != tuple(
            system.cell.vectors.shape
        ):
            raise ValueError("Image neighborhood lattice does not match the system cell.")
        vectors = neighborhood.cell_vectors
        # The search frame is the stored-and-absent certificate witness, so a
        # moved position or runtime cell outside the epoch fails the binding.
        return _build_topology(
            streamed,
            senders=relation.source_indices,
            receivers=relation.receiver_indices,
            edge_cases=relation.route_cases,
            candidate_mask=relation.valid,
            image_shifts=relation.image_shifts,
            stable_route_ids=jnp.arange(relation.capacity, dtype=jnp.int32),
            periodic_mask=system.cell.periodic_mask[None],
            overflow=~neighborhood.evidence.successful,
            evidence=neighborhood.evidence,
            minimum_image_cell=None,
            reference_cell_vectors=vectors,
            reference_positions=neighborhood.reference_positions,
            reference_lattice=vectors,
            lattice_origin=system.cell.origin.astype(vectors.dtype)[None],
            wrap_counts=neighborhood.wrap_counts,
            stencil_extents=neighborhood.stencil_extents,
            lattice_rank=relation.lattice_rank,
            search_radius=neighborhood.search_radius,
            owner_id=neighborhood.relation_schema_id,
            topology_id=neighborhood.relation_schema_id,
            **common,
        )
    elif isinstance(neighborhood, ParticleNeighborhoodState):
        pairs = neighborhood.pair_relation
        if isinstance(pairs.left_indices, jax.core.Tracer) or isinstance(
            pairs.right_indices, jax.core.Tracer
        ):
            senders = jnp.concatenate((pairs.left_indices, pairs.right_indices))
            receivers = jnp.concatenate((pairs.right_indices, pairs.left_indices))
            stable_ids = _receiver_major_rank(senders, receivers)
        else:
            # Frozen route content is immutable preparation; no runtime sort.
            with jax.ensure_compile_time_eval():
                senders = jnp.concatenate((pairs.left_indices, pairs.right_indices))
                receivers = jnp.concatenate((pairs.right_indices, pairs.left_indices))
                stable_ids = _receiver_major_rank(senders, receivers)
        return _build_topology(
            streamed,
            senders=senders,
            receivers=receivers,
            edge_cases=jnp.zeros((senders.shape[0],), dtype=jnp.int32),
            candidate_mask=jnp.concatenate((pairs.valid, pairs.valid)),
            image_shifts=jnp.zeros((senders.shape[0], 0), dtype=jnp.int32),
            stable_route_ids=stable_ids,
            periodic_mask=jnp.zeros((1, 0), dtype=jnp.bool_),
            overflow=jnp.zeros((1,), dtype=jnp.bool_),
            evidence=None,
            minimum_image_cell=cell,
            reference_cell_vectors=None,
            reference_positions=None,
            reference_lattice=None,
            lattice_origin=None,
            wrap_counts=None,
            stencil_extents=None,
            lattice_rank=0,
            search_radius=float("inf"),
            owner_id=pairs.relation_schema_id,
            topology_id=pairs.relation_schema_id,
            **common,
        )
    else:
        raise TypeError(
            "neighborhood must be a ParticleNeighborhoodState or "
            "ParticleImageNeighborhoodState."
        )


def realize_atomistic_graph(
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    /,
    *,
    cutoff: float,
    positions: ArrayLike | None = None,
    topology: AtomisticGraphTopology | None = None,
    cell_vectors: ArrayLike | None = None,
) -> AtomisticGraph:
    """Bind batch geometry to a prepared topology.

    Without ``topology`` this is the historical finite dense realization (cell
    metadata is not used); periodic batches pass a topology from
    ``prepare_atomistic_graph_topology``.  ``cell_vectors`` default to the
    batch cells for image topologies.
    """
    if not isinstance(batch, AtomisticBatch):
        raise TypeError("batch must be an AtomisticBatch.")
    if topology is None:
        axes = batch.periodic_axes
        if axes is not None and (
            isinstance(axes, jax.core.Tracer) or bool(np.any(np.asarray(axes)))
        ):
            raise ValueError(
                "Periodic batches require a topology from "
                "prepare_atomistic_graph_topology."
            )
        if execution.backend != "dense":
            raise TypeError(
                "Dense realization requires a dense AtomisticGraphExecutionPlan."
            )
        topology = _finite_dense_topology(batch, execution)
    elif topology.atom_topology_id != batch.atom_topology_id:
        raise ValueError(
            "Topology was prepared for another atom identity; equal shapes are not "
            "the same batch."
        )
    coordinate = batch.positions if positions is None else jnp.asarray(positions)
    if coordinate.shape != batch.positions.shape:
        raise ValueError("positions must have the batch position shape.")
    vectors = (
        batch.cells
        if cell_vectors is None and topology.lattice_rank > 0
        else cell_vectors
    )
    return bind_atomistic_graph(
        topology,
        execution,
        coordinate.reshape((-1, 3)),
        cutoff=cutoff,
        cell_vectors=vectors,
        periodic_axes=(
            jnp.zeros((batch.case_count, 3), dtype=jnp.bool_)
            if batch.periodic_axes is None
            else batch.periodic_axes
        ),
        atomic_numbers=batch.atomic_numbers,
        atom_type_ids=batch.atom_type_ids,
        masses=batch.masses,
    )


def realize_particle_atomistic_graph(
    system: PreparedAtomisticSystem,
    neighborhood: ParticleNeighborhoodState | ParticleImageNeighborhoodState,
    execution: AtomisticGraphExecutionPlan,
    positions: ArrayLike,
    /,
    *,
    cutoff: float,
    cell: PeriodicCell | None = None,
    cell_vectors: ArrayLike | None = None,
) -> AtomisticGraph:
    """Prepare a runtime neighborhood topology and bind one system's geometry."""
    if not isinstance(execution, AtomisticGraphExecutionPlan) or (
        execution.backend != "particle"
    ):
        raise TypeError("Particle realization requires a particle graph execution plan.")
    coordinate = jnp.asarray(positions)
    expected = (system.capacity, 3)
    if coordinate.shape != expected:
        raise ValueError(f"positions must have shape {expected}.")
    topology = particle_atomistic_graph_topology(
        system, neighborhood, cell=cell, streamed=execution.streamed
    )
    return bind_atomistic_graph(
        topology, execution, coordinate, cutoff=cutoff, cell_vectors=cell_vectors
    )


__all__ = [
    "AtomisticGraph",
    "AtomisticGraphExecutionPlan",
    "AtomisticGraphTopology",
    "bind_atomistic_graph",
    "particle_atomistic_graph_topology",
    "prepare_atomistic_graph_topology",
    "realize_atomistic_graph",
    "realize_particle_atomistic_graph",
]
