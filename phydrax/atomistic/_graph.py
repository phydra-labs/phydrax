#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, final, Literal, TypeAlias, TypedDict

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._dtype_names import real_precision_dtype_name
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState, PeriodicCell
from ..discretization.particle._image_neighborhood import (
    AbstractImageRouteSearch,
    CellListImageRouteSearch,
    DenseImageRouteSearch,
    image_certificate,
    ImageNeighborSelectionPolicy,
    ImageSelectionCapacity,
    ImageSelectionEvidence,
    ParticleImageCapacity,
    ParticleImageNeighborhoodState,
    PreparedImageNeighborSelection,
)
from ..discretization.particle._image_relation import (
    ParticleImageRelation,
    ParticleImageRelationEvidence,
)
from ..graph import GraphIR
from ..sparse import EdgeRelation, gather_routes, RelationAccumulation
from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationPlan
from ..typing import parse
from ._execution_binding import atomistic_case_address_words
from ._neighborhood_bundle import (
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticRoleEvidence,
    PreparedAtomisticNeighborhoodBundle,
)
from ._sites import AtomisticSiteDomain
from ._system import PreparedAtomisticSystem
from ._types import AtomisticBatch, AtomisticScaleContract


AtomisticGraphBackend: TypeAlias = Literal["dense", "particle"]


class _SystemTopologyFields(TypedDict):
    particle_ids: Array
    case_identity_words: Array
    atomic_numbers: Array
    atom_type_ids: Array
    masses: Array
    atom_mask: Array
    atom_cases: Array
    epoch: Array
    case_count: int
    atom_capacity: int
    skin: float
    execution_id: None
    support_id: str
    search_failure: Array


class AtomisticGraphExecutionPlan(StrictModule, NonTrainableState):
    """Resource and realization policy independent of learned model parameters.

    ``backend="dense"`` is the bounded named dense reference (finite pairs or,
    for periodic batches, pairs times a complete image stencil).
    ``backend="particle"`` uses scalable cell-list image search for graph
    layouts and the particle neighborhood for runtime systems.
    ``image_capacity`` charges image-aware routes; ``streamed`` is the
    receiver-major streamed relation schedule plan prepared once per topology.
    Its accumulation policy also governs the geometric route gathers of
    ``bind_atomistic_graph``: ordered policies prepare validated geometry
    relations once per topology, so position and cell derivatives accumulate
    in route storage order.

    ``search_cell`` (a full-rank ``(3, 3)`` row lattice in scale length units)
    declares the route-search envelope of every reusable ``AtomisticGraphLayout``:
    it fixes the cell-list grid or dense image stencil (widened by the relative
    ``search_margin``) and the minimum binning length of nonperiodic rows. A
    payload exceeding the envelope's storage (cell occupancy, images, routes,
    degree) is a capacity failure; one needing a wider stencil than declared is
    the search's scientific stencil failure. Neither recompiles.
    """

    maximum_neighbors: int = eqx.field(static=True)
    maximum_dense_atoms: int | None = eqx.field(static=True)
    backend: AtomisticGraphBackend = eqx.field(static=True)
    image_capacity: ParticleImageCapacity | None
    streamed: StreamedRelationPlan
    maximum_candidate_slots: int = eqx.field(static=True)
    search_cell: tuple[tuple[float, float, float], ...] | None = eqx.field(static=True)
    search_margin: float = eqx.field(static=True)
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
        search_cell: ArrayLike | None = None,
        search_margin: float = 0.0,
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
        envelope = None if search_cell is None else _search_cell(search_cell)
        margin = float(search_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("search_margin must be finite and non-negative.")
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
                "search_cell": None
                if envelope is None
                else [list(row) for row in envelope],
                "search_margin": margin,
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
        self.search_cell = envelope
        self.search_margin = margin
        self.plan_id = identifier


def _search_cell(value: ArrayLike, /) -> tuple[tuple[float, float, float], ...]:
    """Validate one declared full-rank ``(3, 3)`` search envelope lattice."""
    cell = np.asarray(value, dtype=np.float64)
    if cell.shape != (3, 3) or np.any(~np.isfinite(cell)):
        raise ValueError("search_cell must be a finite (3, 3) row lattice.")
    singular = np.linalg.svd(cell, compute_uv=False)
    if singular[-1] <= np.finfo(np.float64).eps * 16 * singular[0]:
        raise ValueError("search_cell must be a full-rank lattice.")
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in cell)


@final
class AtomisticGeometryRelations(StrictModule, NonTrainableState):
    """Validated relations of ordered geometric route gathers in one topology epoch.

    Derived once per topology build from its canonical routes when the
    streamed plan orders accumulation; never serialized. ``senders`` and
    ``receivers`` read flat atom rows and ``lattices`` reads the case lattice
    of every image route.
    """

    senders: EdgeRelation
    receivers: EdgeRelation
    lattices: EdgeRelation | None


class AtomisticGraphTopology(StrictModule, NonTrainableState):
    """Prepared directed route topology, separate from numerical geometry.

    Integer routes, image shifts ``n``, candidate membership, stable route
    order and the prepared streamed schedule are fixed for one
    topology epoch.  Binding positions and cell vectors only computes
    geometry, cutoff masks and certificates.  Atom arrays are flat case-major.
    ``stable_route_ids`` is the canonical receiver-major rank of every route.
    Layout and runtime image-neighborhood topologies retain their reference
    build frame, so binding with moved positions or a deformed cell evaluates
    the stored-and-absent image certificate.  Static identity is schema only:
    ``topology_id`` is the content-free relation schema (also the streamed
    schedule owner) and ``support_id`` the particle support slot schema.
    Material identity (case address words, stable particle IDs, node metadata,
    periodicity) and route content are dynamic leaves, bound by array
    equality (``admit_supplied_topology``, ``AtomisticGraphPayload.admit``),
    never by identifier; ``content_id()`` is their host identity.
    ``overflow`` is per-case capacity failure; ``search_failure`` is per-case
    scientific route-search or material-binding failure of a layout topology.
    ``geometry_relations`` holds the derived relations of ordered geometric
    gathers and is ``None`` under fast accumulation.
    """

    senders: Array
    receivers: Array
    edge_cases: Array
    candidate_mask: Array
    image_shifts: Array
    stable_route_ids: Array
    particle_ids: Array
    case_identity_words: Array
    atomic_numbers: Array
    atom_type_ids: Array
    masses: Array
    atom_mask: Array
    atom_cases: Array
    periodic_mask: Array
    epoch: Array
    overflow: Array
    search_failure: Array
    evidence: ParticleImageRelationEvidence | None
    minimum_image_cell: PeriodicCell | None
    reference_cell_vectors: Array | None
    reference_positions: Array | None
    reference_lattice: Array | None
    lattice_origin: Array | None
    wrap_counts: Array | None
    stencil_extents: Array | None
    streamed: PreparedStreamedRelation
    geometry_relations: AtomisticGeometryRelations | None
    case_count: int = eqx.field(static=True)
    atom_capacity: int = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    search_radius: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    execution_id: str | None = eqx.field(static=True)
    role_evidence: AtomisticRoleEvidence | None = None
    role_selection: ImageNeighborSelectionPolicy | None = None

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
        # Capacity-bounded (content-free) schedules stay capacity bounded, so a
        # second plan never reintroduces content into the static binding.
        prepared, _ = _bound_schedule(
            plan,
            self.relation,
            owner_id=self.topology_id,
            epoch=self.epoch,
            stable_route_ids=self.stable_route_ids,
            receiver_valid=self.atom_mask,
            capacity_bounded=self.streamed.binding.content_id is None,
        )
        return prepared

    def content_id(self) -> str:
        """Host boundary: scientific graph identity of this topology epoch.

        Binds the schema and support slot together with the material identity
        (case address words, stable particle IDs, atomic numbers, type IDs,
        masses, atom mask, periodicity) and the route content (endpoints,
        cases, candidate mask, integer images, stable order, epoch). Distinct
        materials of one layout have distinct content identities. Never call
        inside ``jit``; compiled code binds content numerically.
        """
        return canonical_fingerprint(
            {
                "kind": "atomistic-graph-topology-content",
                "schema": self.topology_id,
                "support": self.support_id,
                "material": array_tree_fingerprint(
                    {
                        "case_identity_words": np.asarray(self.case_identity_words),
                        "particle_ids": np.asarray(self.particle_ids),
                        "atomic_numbers": np.asarray(self.atomic_numbers),
                        "atom_type_ids": np.asarray(self.atom_type_ids),
                        "masses": np.asarray(self.masses),
                        "atom_mask": np.asarray(self.atom_mask),
                        "periodic_mask": np.asarray(self.periodic_mask),
                    }
                ),
                "routes": array_tree_fingerprint(
                    {
                        "senders": np.asarray(self.senders),
                        "receivers": np.asarray(self.receivers),
                        "edge_cases": np.asarray(self.edge_cases),
                        "candidate_mask": np.asarray(self.candidate_mask),
                        "image_shifts": np.asarray(self.image_shifts),
                        "stable_route_ids": np.asarray(self.stable_route_ids),
                        "epoch": np.asarray(self.epoch),
                    }
                ),
            }
        )


def _bound_schedule(
    streamed: StreamedRelationPlan | PreparedStreamedRelation,
    relation: EdgeRelation,
    /,
    *,
    owner_id: str,
    epoch: Array,
    stable_route_ids: Array,
    receiver_valid: Array,
    capacity_bounded: bool,
) -> tuple[PreparedStreamedRelation, Array]:
    """Prepare a schedule for one topology epoch, or admit one already prepared.

    An admitted schedule must name the same owner, endpoint extents and route
    capacity.  Its binding to *this* topology's content is checked numerically
    (no callback): epoch, receiver keys, active-route mask, stable route IDs and
    every scheduled lane's source must equal the topology's.  The returned
    status is false on any mismatch, so a schedule prepared for other routes of
    equal shape fails the topology instead of silently driving evaluation.
    ``capacity_bounded`` prepares a content-free schedule whose static tile
    bound depends only on extents (layout topologies).
    """
    if isinstance(streamed, StreamedRelationPlan):
        prepared = streamed.prepare(
            relation,
            owner_id=owner_id,
            epoch=epoch,
            stable_route_ids=stable_route_ids,
            receiver_valid=receiver_valid,
            capacity_bounded=capacity_bounded,
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
    *,
    capacity_bounded: bool = False,
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
        owner_id=fields["topology_id"],
        epoch=fields["epoch"],
        stable_route_ids=fields["stable_route_ids"],
        receiver_valid=fields["atom_mask"],
        capacity_bounded=capacity_bounded,
    )
    fields["overflow"] = fields["overflow"] | ~epoch_matches
    geometry = None
    if prepared.plan.accumulation != "fast":
        geometry = AtomisticGeometryRelations(
            senders=relation,
            receivers=relation.transpose(),
            lattices=(
                EdgeRelation(
                    fields["edge_cases"],
                    fields["receivers"],
                    source_size=int(fields["case_count"]),
                    target_size=size,
                    valid=fields["candidate_mask"],
                )
                if int(fields["lattice_rank"]) > 0
                else None
            ),
        )
    return AtomisticGraphTopology(
        streamed=prepared, geometry_relations=geometry, **fields
    )


_STATIC_TOPOLOGY_FIELDS = (
    "case_count",
    "atom_capacity",
    "lattice_rank",
    "search_radius",
    "skin",
    "topology_id",
    "support_id",
    "execution_id",
)
_CONTENT_TOPOLOGY_FIELDS = (
    "senders",
    "receivers",
    "edge_cases",
    "candidate_mask",
    "image_shifts",
    "stable_route_ids",
    "particle_ids",
    "case_identity_words",
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
    vectors or candidate geometry, independent of whether the case has edges;
    ``topology.search_failure`` reports scientific route-search failure. All
    three invalidate the case; only ``overflow`` is a capacity failure.
    """

    graph: GraphIR
    neighbor_counts: Array
    edge_slots: Array
    maximum_neighbor_count: Array
    overflow: Array
    nonfinite: Array
    topology: AtomisticGraphTopology
    positions: Array
    cell_vectors: Array
    image_counts: Array | None
    selection_evidence: ImageSelectionEvidence | None
    role_evidence: AtomisticRoleEvidence | None
    selection_id: str | None = eqx.field(static=True)
    candidate_mask: Array
    maximum_neighbors: int = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        admitted = ~(self.overflow | self.nonfinite | self.topology.search_failure)
        if self.selection_evidence is not None:
            admitted = admitted & self.selection_evidence.successful
        if self.role_evidence is not None:
            admitted = admitted & self.role_evidence.successful
        return admitted

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
    geometry = topology.geometry_relations
    if geometry is None:
        raw = positions[topology.receivers] - positions[topology.senders]
        case_vectors = vectors[topology.edge_cases] if topology.lattice_rank else None
    else:
        # Ordered policies gather through the topology's validated relations so
        # position and cell cotangents accumulate in route storage order.
        accumulation = topology.streamed.plan.accumulation
        raw = gather_routes(
            geometry.receivers, positions, accumulation=accumulation
        ) - gather_routes(geometry.senders, positions, accumulation=accumulation)
        case_vectors = (
            None
            if geometry.lattices is None
            else gather_routes(geometry.lattices, vectors, accumulation=accumulation)
        )
    if case_vectors is not None:
        raw = raw + contract(
            "ei,eid->ed",
            topology.image_shifts.astype(positions.dtype),
            case_vectors,
            backend="jax",
        )
    elif topology.minimum_image_cell is not None:
        cell = topology.minimum_image_cell
        raw = (
            cell.minimum_image(raw)
            if vectors.shape[1] == 0
            else cell.minimum_image_with_vectors(raw, vectors[0])
        )
    return jnp.where(
        topology.candidate_mask[:, None], raw, jnp.zeros((), dtype=raw.dtype)
    )


def case_energy_totals(
    atom_energy: Array,
    atom_cases: Array,
    case_count: int,
    accumulation: RelationAccumulation,
    /,
) -> Array:
    """Per-case sums of flat case-major atom energies.

    Fast accumulation adds by ``atom_cases``. Ordered policies reduce each
    case's contiguous atom row natively, never by unordered duplicate-index
    additions, so repeated evaluations agree bitwise on one backend.
    """
    if accumulation == "fast":
        return (
            jnp.zeros((case_count,), dtype=atom_energy.dtype)
            .at[atom_cases.reshape((-1,))]
            .add(atom_energy)
        )
    return jnp.sum(atom_energy.reshape((case_count, -1)), axis=1)


def _certificate_failure(
    topology: AtomisticGraphTopology,
    positions: Array,
    vectors: Array,
    cutoff: float,
    /,
) -> Array:
    if topology.role_evidence is not None:
        if (
            topology.reference_positions is None
            or topology.reference_cell_vectors is None
        ):
            raise RuntimeError("A bundle graph requires its certified geometry.")
        same_position = jnp.all(
            jnp.where(
                topology.atom_mask[:, None],
                positions == topology.reference_positions,
                True,
            ).reshape((topology.case_count, -1)),
            axis=-1,
        )
        same_cell = jnp.all(vectors == topology.reference_cell_vectors, axis=(-2, -1))
        return ~(same_position & same_cell)
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


def _select_graph_routes(
    topology: AtomisticGraphTopology,
    execution: AtomisticGraphExecutionPlan,
    positions: Array,
    vectors: Array,
    policy: ImageNeighborSelectionPolicy,
    certified: Array,
    /,
) -> tuple[Array, ImageSelectionEvidence]:
    if topology.minimum_image_cell is not None:
        raise ValueError(
            "Scientific graph selection requires explicit image or finite routes."
        )
    relation = ParticleImageRelation(
        topology.relation,
        topology.particle_ids[topology.senders],
        topology.particle_ids[topology.receivers],
        topology.image_shifts,
        topology.edge_cases,
        case_count=topology.case_count,
        particle_capacity=topology.atom_capacity,
        support_id=topology.support_id,
        relation_schema_id=topology.topology_id,
    )
    selected = PreparedImageNeighborSelection(
        policy,
        ImageSelectionCapacity(
            maximum_routes=topology.edge_capacity,
            maximum_degree=execution.maximum_neighbors,
        ),
        source_radius=topology.search_radius,
        case_count=topology.case_count,
    ).select(
        relation, positions, vectors, evidence=topology.evidence, certified=certified
    )
    mask = (
        jnp.zeros(topology.candidate_mask.shape, dtype=jnp.bool_)
        .at[selected.parent_routes]
        .max(selected.relation.valid)
    )
    return mask, selected.evidence


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
    image_counts: ArrayLike | None = None,
    selection: ImageNeighborSelectionPolicy | None = None,
) -> AtomisticGraph:
    """Bind numerical geometry to a prepared topology.

    ``d_e = x[receiver] - x[sender] + n_e @ H[case_e]`` for image topologies
    (``H`` = explicit ``cell_vectors``, differentiable); the historical
    selected minimum image for classical periodic pair topologies; plain
    receiver-minus-sender otherwise.  No route discovery or sort occurs.
    Route gathers follow the topology's streamed accumulation; values are equal
    under every policy, and ordered policies sequence position and cell
    cotangents in route storage order through the topology's prepared
    ``geometry_relations`` instead of unordered scatter additions.

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
    if topology.role_evidence is not None:
        bound_selection = topology.role_selection
        if selection is not None and (
            bound_selection is None or selection.policy_id != bound_selection.policy_id
        ):
            raise ValueError("A bundle graph cannot change its scientific selection.")
        selection = bound_selection
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
    counts = None if image_counts is None else jnp.asarray(image_counts)
    if counts is not None:
        if not jnp.issubdtype(counts.dtype, jnp.integer):
            raise TypeError("image_counts must be integer representation counts.")
        if counts.shape == (
            topology.case_count,
            topology.atom_capacity,
            topology.lattice_rank,
        ):
            counts = counts.reshape((atom_count, topology.lattice_rank))
        elif counts.shape != (atom_count, topology.lattice_rank):
            raise ValueError("image_counts must match the graph's case-major image axes.")
    vectors = _resolved_cell_vectors(topology, cell_vectors, coordinate.dtype)
    displacement, distance, direction = _graph_edge_geometry(
        topology, coordinate, vectors
    )
    certificate_failure = _certificate_failure(
        topology, coordinate, vectors, cutoff_value
    )
    selected_evidence = None
    edge_mask = topology.candidate_mask & (
        distance < jnp.asarray(cutoff_value, coordinate.dtype)
    )
    if selection is not None:
        if selection.radius > cutoff_value:
            raise ValueError("The graph cutoff does not cover its scientific selection.")
        if topology.role_evidence is None:
            edge_mask, selected_evidence = _select_graph_routes(
                topology,
                execution,
                coordinate,
                vectors,
                selection,
                ~(
                    certificate_failure
                    | topology.overflow
                    | topology.search_failure
                    | periodicity_mismatch
                ),
            )
        else:
            selected_evidence = topology.role_evidence.selection
            if selected_evidence is None:
                raise RuntimeError("A selected bundle graph lacks selection evidence.")
            edge_mask = topology.candidate_mask
    return _assemble_atomistic_graph(
        topology,
        execution,
        coordinate,
        vectors,
        (node_numbers, node_types, node_masses),
        (displacement, distance, direction),
        edge_mask,
        certificate_failure,
        periodicity_mismatch,
        counts,
        cutoff_value,
        selection,
        selected_evidence,
    )


def _graph_edge_geometry(
    topology: AtomisticGraphTopology, coordinate: Array, vectors: Array, /
) -> tuple[Array, Array, Array]:
    """Preserve the graph's finite zero-distance value and derivative convention."""
    displacement = _edge_displacement(topology, coordinate, vectors)
    squared = jnp.sum(displacement * displacement, axis=-1)
    tiny = jnp.asarray(jnp.finfo(coordinate.dtype).tiny, dtype=coordinate.dtype)
    positive = jnp.sqrt(jnp.maximum(squared, tiny))
    zero = jnp.zeros((), dtype=coordinate.dtype)
    distance = jnp.where(squared > zero, positive, zero)
    safe = jnp.where(distance > zero, distance, jnp.ones((), dtype=coordinate.dtype))
    return displacement, distance, displacement / safe[:, None]


def _assemble_atomistic_graph(
    topology: AtomisticGraphTopology,
    execution: AtomisticGraphExecutionPlan,
    coordinate: Array,
    vectors: Array,
    node_metadata: tuple[Array, Array, Array],
    edge_geometry: tuple[Array, Array, Array],
    edge_mask: Array,
    certificate_failure: Array,
    periodicity_mismatch: Array,
    counts: Array | None,
    cutoff_value: float,
    selection: ImageNeighborSelectionPolicy | None,
    selected_evidence: ImageSelectionEvidence | None,
    /,
) -> AtomisticGraph:
    """Assemble one already-admitted numerical graph without rediscovering routes."""
    node_numbers, node_types, node_masses = node_metadata
    displacement, distance, direction = edge_geometry
    role_evidence = topology.role_evidence
    if role_evidence is not None:
        role_evidence = replace(
            role_evidence,
            stale=role_evidence.stale | certificate_failure | periodicity_mismatch,
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
        | (
            certificate_failure | periodicity_mismatch
            if role_evidence is None
            else jnp.zeros_like(certificate_failure)
        )
    )
    if selected_evidence is not None:
        overflow = overflow | selected_evidence.capacity_failure
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
            "selection": None if selection is None else selection.policy_id,
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
        positions=coordinate,
        cell_vectors=vectors,
        image_counts=counts,
        selection_id=None if selection is None else selection.policy_id,
        selection_evidence=selected_evidence,
        role_evidence=role_evidence,
        candidate_mask=topology.candidate_mask,
        maximum_neighbors=execution.maximum_neighbors,
        cutoff=cutoff_value,
        topology_id=topology.topology_id,
        execution_id=execution.plan_id,
        graph_id=graph_id,
    )


def _neighborhood_graph_topology(
    system: PreparedAtomisticSystem,
    bundle: PreparedAtomisticNeighborhoodBundle,
    state: AtomisticNeighborhoodBundleState,
    requirement: AtomisticNeighborhoodRequirement,
    execution: AtomisticGraphExecutionPlan,
    /,
) -> AtomisticGraphTopology:
    """Bind a certified role epoch; never rediscover, reselect, or sort routes."""
    if bundle.system.support_schema_id != system.support_schema_id:
        raise ValueError("The graph resource bundle belongs to another support layout.")
    if requirement.role is not AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH:
        raise ValueError("A learned graph requires a directed-image role.")
    view = bundle.view(state, requirement)
    if view.entity is not AtomisticSiteDomain.PHYSICAL_ATOMS:
        raise ValueError("Learned graph nodes must be physical atoms.")
    relation = view.relation
    if relation.case_count != 1 or relation.particle_capacity != system.capacity:
        raise ValueError("A program graph must index its single prepared system.")
    if relation.support_id != system.support_schema_id:
        raise ValueError("Graph routes belong to another indexed particle support.")
    schedule = bundle.graph_schedule(state, requirement)
    if schedule.plan.plan_id != execution.streamed.plan_id:
        raise ValueError("The role schedule and graph execution plans differ.")
    periodic = (
        jnp.zeros((1, relation.lattice_rank), dtype=jnp.bool_)
        if system.cell is None
        else system.cell.periodic_mask[None]
    )
    return _build_topology(
        schedule,
        senders=relation.source_indices,
        receivers=relation.receiver_indices,
        edge_cases=relation.route_cases,
        candidate_mask=relation.valid,
        image_shifts=relation.image_shifts,
        stable_route_ids=jnp.arange(relation.capacity, dtype=jnp.int32),
        particle_ids=system.plan.particle_ids,
        case_identity_words=system.case_address_words,
        atomic_numbers=system.plan.atomic_numbers,
        atom_type_ids=system.plan.atom_type_ids,
        masses=system.plan.masses,
        atom_mask=system.active_mask,
        atom_cases=jnp.zeros((system.capacity,), dtype=jnp.int32),
        periodic_mask=periodic,
        epoch=bundle.graph_epoch(state, requirement),
        overflow=view.evidence.capacity_failure,
        search_failure=jnp.broadcast_to(
            ~bundle.system.prepared_identity.equals(system.prepared_identity), (1,)
        ),
        evidence=None,
        minimum_image_cell=None,
        reference_cell_vectors=state.geometry.cell_vectors,
        reference_positions=state.geometry.positions[0],
        reference_lattice=None,
        lattice_origin=None,
        wrap_counts=None,
        stencil_extents=None,
        case_count=1,
        atom_capacity=system.capacity,
        lattice_rank=relation.lattice_rank,
        search_radius=requirement.bounded_cutoff(),
        skin=0.0,
        topology_id=schedule.binding.owner_id,
        support_id=system.support_schema_id,
        execution_id=execution.plan_id,
        role_evidence=view.evidence,
        role_selection=requirement.selection,
    )


def _dense_routes(
    case_count: int, atom_capacity: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite all-pairs routes of a layout; they depend only on capacities."""
    edge_capacity = atom_capacity * (atom_capacity - 1)
    local_senders = np.repeat(np.arange(atom_capacity, dtype=np.int32), atom_capacity - 1)
    local_offsets = np.tile(np.arange(atom_capacity - 1, dtype=np.int32), atom_capacity)
    local_receivers = local_offsets + (local_offsets >= local_senders)
    case_offsets = np.repeat(
        np.arange(case_count, dtype=np.int32) * atom_capacity, edge_capacity
    )
    senders = (np.tile(local_senders, case_count) + case_offsets).astype(np.int32)
    receivers = (np.tile(local_receivers, case_count) + case_offsets).astype(np.int32)
    order = np.lexsort((senders, receivers))
    rank = np.empty(order.shape, dtype=np.int32)
    rank[order] = np.arange(order.shape[0], dtype=np.int32)
    cases = np.repeat(np.arange(case_count, dtype=np.int32), edge_capacity)
    return senders, receivers, cases, rank


def _case_binning_lattice(
    cell: np.ndarray,
    periodic: np.ndarray,
    positions: np.ndarray,
    active: np.ndarray,
    radius: float,
    lengths: tuple[float, float, float],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-rank search lattice whose periodic rows are the physical cell.

    Nonperiodic rows only bin candidates (their image shifts are zero). They
    are the orthonormal complement of the periodic rows, spanning every active
    atom plus ``radius`` and padded to the layout's minimum binning
    ``lengths``, so a nonperiodic row never needs a wider fractional stencil
    than the layout's declared search envelope. Route content and order are
    independent of the binning lattice.
    """
    periodic_rows = np.flatnonzero(periodic)
    if periodic_rows.size:
        singular_values = np.linalg.svd(cell[periodic_rows], compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps * 16 * singular_values[0]:
            raise ValueError("Periodic cell rows of a batch case are rank deficient.")
    if periodic_rows.size == 3:
        return cell, np.zeros((3,), dtype=np.float64)
    if periodic_rows.size:
        _, _, basis = np.linalg.svd(cell[periodic_rows])
        complement = basis[periodic_rows.size :]
    else:
        complement = np.eye(3)
    projection = positions[active] @ complement.T
    lower = projection.min(axis=0) - radius
    upper = projection.max(axis=0) + radius
    lattice = np.array(cell, dtype=np.float64, copy=True)
    for row, axis in enumerate(np.flatnonzero(~periodic)):
        lattice[axis] = complement[row] * max(upper[row] - lower[row], lengths[axis])
    return lattice, lower @ complement


def _binning_lattices(
    batch: AtomisticBatch,
    radius: float,
    lengths: tuple[float, float, float],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Host binning lattices and origins of every case of a concrete batch."""
    cases = batch.case_count
    periodic = (
        np.zeros((cases, 3), dtype=np.bool_)
        if batch.periodic_axes is None
        else np.asarray(batch.periodic_axes, dtype=np.bool_)
    )
    cells = (
        np.zeros((cases, 3, 3), dtype=np.float64)
        if batch.cells is None
        else np.asarray(batch.cells, dtype=np.float64)
    )
    positions = np.asarray(batch.positions, dtype=np.float64)
    active = np.asarray(batch.atom_mask, dtype=np.bool_)
    lattices = np.empty((cases, 3, 3), dtype=np.float64)
    origins = np.empty((cases, 3), dtype=np.float64)
    for case in range(cases):
        lattices[case], origins[case] = _case_binning_lattice(
            cells[case], periodic[case], positions[case], active[case], radius, lengths
        )
    return lattices, origins


def _search_radius(cutoff: float, skin: float, /) -> tuple[float, float]:
    cutoff_value = float(cutoff)
    skin_value = float(skin)
    if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if not np.isfinite(skin_value) or skin_value < 0.0:
        raise ValueError("skin must be finite and non-negative.")
    return cutoff_value, skin_value


def _candidate_radius(cutoff: float, skin: float, dtype: str, /) -> float:
    """Cover the closed scientific radius with the search's open ball."""
    precision = np.dtype(dtype)
    bound = cutoff + skin
    if not np.isfinite(bound) or bound >= np.finfo(precision).max:
        raise ValueError("cutoff plus skin must have finite candidate-search headroom.")
    rounded = precision.type(bound)
    if rounded >= np.finfo(precision).max:
        raise ValueError("cutoff plus skin must have finite candidate-search headroom.")
    radius = np.nextafter(rounded, precision.type(np.inf))
    return float(radius)


def _reference_reach(lattices: np.ndarray, /) -> np.ndarray:
    """Per-case reciprocal row norms ``|b_i|`` of full-rank reference lattices."""
    reach = np.empty((lattices.shape[0], 3), dtype=np.float64)
    for case, matrix in enumerate(lattices):
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] <= np.finfo(np.float64).eps * 16 * singular[0]:
            raise ValueError("The search envelope lattice is rank deficient.")
        reach[case] = np.linalg.norm(np.linalg.inv(matrix), axis=0)
    return reach


@final
class AtomisticGraphLayout(StrictModule, NonTrainableState):
    """One reusable executable graph layout of a declared workload bucket.

    The layout owns every static choice of graph preparation and evaluation:
    scale, coordinate dtype, case and atom capacities, execution plan, search
    ``cutoff + skin`` (outward-rounded in the coordinate dtype), the periodicity
    class (``periodic_axes`` cases may declare; ``cells`` whether payloads carry
    cell vectors and periodic axes)
    and the fixed route search. ``layout_id`` (also the support-slot schema of
    its topologies) and ``topology_id`` (relation schema) fingerprint that
    configuration only, never material content, so every payload of one
    layout shares one static tree definition and one compiled search,
    evaluation and training step. Material identity stays in dynamic leaves
    and in the host ``AtomisticGraphTopology.content_id()``.

    ``lattice_rank`` is ``0`` (finite all-pairs routes, dense backend without
    periodic axes) or ``3`` (image search). The image search is built once
    from the declared envelope: ``search_cell`` (``(3, 3)`` or
    ``(case, 3, 3)`` row lattices; defaults to ``execution.search_cell``) and
    ``execution.search_margin`` fix its grid or stencil, and
    ``binning_lengths`` are the minimum nonperiodic binning row lengths. A
    payload never recompiles: storage exceedance (cell occupancy, images,
    routes, degree) is capacity failure and stencil, domain or representation
    exceedance is the search's scientific failure, both per case.
    """

    scale: AtomisticScaleContract
    execution: AtomisticGraphExecutionPlan
    search: AbstractImageRouteSearch | None
    case_capacity: int = eqx.field(static=True)
    atom_capacity: int = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    periodic_axes: tuple[bool, bool, bool] = eqx.field(static=True)
    cells: bool = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    binning_lengths: tuple[float, float, float] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        case_capacity: int,
        atom_capacity: int,
        coordinate_dtype: Any,
        cutoff: float,
        skin: float = 0.0,
        periodic_axes: Sequence[bool] = (False, False, False),
        cells: bool | None = None,
        search_cell: ArrayLike | None = None,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        if not isinstance(execution, AtomisticGraphExecutionPlan):
            raise TypeError("execution must be an AtomisticGraphExecutionPlan.")
        cases = int(case_capacity)
        atoms = int(atom_capacity)
        if cases <= 0 or atoms <= 0:
            raise ValueError("case_capacity and atom_capacity must be positive.")
        dtype = real_precision_dtype_name(coordinate_dtype)
        cutoff_value, skin_value = _search_radius(cutoff, skin)
        axes = tuple(bool(value) for value in periodic_axes)
        if len(axes) != 3:
            raise ValueError("periodic_axes must name three Cartesian lattice rows.")
        carries = any(axes) if cells is None else bool(cells)
        if any(axes) and not carries:
            raise ValueError("A layout admitting periodic axes must carry cells.")
        image = execution.backend == "particle" or any(axes)
        dense_limit = execution.maximum_dense_atoms
        if execution.backend == "dense" and (dense_limit is None or atoms > dense_limit):
            raise ValueError(
                f"Dense atomistic graph capacity {atoms} exceeds the explicit "
                f"maximum_dense_atoms={dense_limit} resource guard."
            )
        search: AbstractImageRouteSearch | None = None
        lengths = (0.0, 0.0, 0.0)
        if image:
            radius = _candidate_radius(cutoff_value, skin_value, dtype)
            capacity = execution.image_capacity
            if capacity is None:
                raise ValueError(
                    "Periodic or particle-backend graph layouts require image_capacity."
                )
            reference = execution.search_cell if search_cell is None else search_cell
            if reference is None:
                raise ValueError(
                    "Periodic or particle-backend graph layouts require a declared "
                    "route-search envelope: AtomisticGraphExecutionPlan(search_cell=...) "
                    "or AtomisticGraphLayout(search_cell=...)."
                )
            lattices = np.asarray(reference, dtype=np.float64)
            if lattices.shape == (3, 3):
                lattices = np.broadcast_to(lattices, (cases, 3, 3)).copy()
            if lattices.shape != (cases, 3, 3) or np.any(~np.isfinite(lattices)):
                raise ValueError(
                    "search_cell must be finite (3, 3) or (case, 3, 3) row lattices."
                )
            reach = _reference_reach(lattices)
            lengths = (
                float(1.0 / np.max(reach[:, 0])),
                float(1.0 / np.max(reach[:, 1])),
                float(1.0 / np.max(reach[:, 2])),
            )
            if execution.backend == "dense" and dense_limit is not None:
                search = DenseImageRouteSearch(
                    radius,
                    lattices,
                    np.broadcast_to(np.asarray(axes, dtype=np.bool_), (cases, 3)),
                    capacity,
                    particle_capacity=atoms,
                    maximum_dense_routes=cases
                    * dense_limit
                    * dense_limit
                    * capacity.maximum_images,
                    deformation_margin=execution.search_margin,
                )
            else:
                search = CellListImageRouteSearch(
                    radius,
                    lattices,
                    capacity,
                    particle_capacity=atoms,
                    maximum_candidate_slots=execution.maximum_candidate_slots,
                    deformation_margin=execution.search_margin,
                )
        layout_id = canonical_fingerprint(
            {
                "kind": "atomistic-graph-layout",
                "scale": scale.scale_id,
                "execution": execution.plan_id,
                "case_capacity": cases,
                "atom_capacity": atoms,
                "coordinate_dtype": dtype,
                "cutoff": cutoff_value,
                "skin": skin_value,
                "periodic_axes": list(axes),
                "cells": carries,
                "search": None if search is None else search.search_id,
                "binning_lengths": list(lengths),
            }
        )
        self.scale = scale
        self.execution = execution
        self.search = search
        self.case_capacity = cases
        self.atom_capacity = atoms
        self.coordinate_dtype = dtype
        self.cutoff = cutoff_value
        self.skin = skin_value
        self.periodic_axes = (axes[0], axes[1], axes[2])
        self.cells = carries
        self.lattice_rank = 3 if image else 0
        self.binning_lengths = lengths
        self.topology_id = canonical_fingerprint(
            {"kind": "atomistic-graph-layout-routes", "layout": layout_id}
        )
        self.layout_id = layout_id

    @classmethod
    def for_batch(
        cls,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        cutoff: float,
        skin: float = 0.0,
    ) -> AtomisticGraphLayout:
        """Host boundary: the one-batch layout of ``batch``'s capacities and class.

        The search envelope is ``execution.search_cell`` when declared, else
        the batch's own binning lattices (the per-batch search configuration).
        Periodic metadata must be concrete; finite all-pairs layouts admit a
        traced batch without periodic metadata.
        """
        if not isinstance(batch, AtomisticBatch):
            raise TypeError("batch must be an AtomisticBatch.")
        if not isinstance(execution, AtomisticGraphExecutionPlan):
            raise TypeError("execution must be an AtomisticGraphExecutionPlan.")
        cutoff_value, skin_value = _search_radius(cutoff, skin)
        declared = (
            np.zeros((3,), dtype=np.bool_)
            if batch.periodic_axes is None
            else np.any(np.asarray(batch.periodic_axes, dtype=np.bool_), axis=0)
        )
        axes = (bool(declared[0]), bool(declared[1]), bool(declared[2]))
        reference = None
        if (
            execution.backend == "particle" or any(axes)
        ) and execution.search_cell is None:
            reference, _ = _binning_lattices(
                batch,
                _candidate_radius(
                    cutoff_value,
                    skin_value,
                    real_precision_dtype_name(batch.positions.dtype),
                ),
                (0.0, 0.0, 0.0),
            )
        return cls(
            batch.scale,
            execution,
            case_capacity=batch.case_count,
            atom_capacity=batch.atom_capacity,
            coordinate_dtype=batch.positions.dtype,
            cutoff=cutoff_value,
            skin=skin_value,
            periodic_axes=axes,
            cells=batch.cells is not None,
            search_cell=reference,
        )

    @property
    def search_radius(self) -> float:
        """Actual open candidate radius, covering the closed cutoff plus skin."""
        return float("inf") if self.search is None else self.search.radius

    def require_batch(self, batch: AtomisticBatch, /) -> None:
        """Host refusal of a batch outside this layout's static class."""
        if not isinstance(batch, AtomisticBatch):
            raise TypeError("batch must be an AtomisticBatch.")
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError("The batch and graph layout scale contracts differ.")
        if (batch.case_count, batch.atom_capacity) != (
            self.case_capacity,
            self.atom_capacity,
        ):
            raise ValueError(
                f"The batch shape {(batch.case_count, batch.atom_capacity)} differs "
                f"from the layout capacities {(self.case_capacity, self.atom_capacity)}."
            )
        if batch.positions.dtype != jnp.dtype(self.coordinate_dtype):
            raise ValueError("The batch coordinate dtype differs from the layout's.")
        if (batch.cells is not None) != self.cells:
            raise ValueError(
                "The batch cell metadata differs from the layout's cell class."
            )
        axes = batch.periodic_axes
        if axes is not None and not isinstance(axes, jax.core.Tracer):
            outside = np.asarray(axes, dtype=np.bool_) & ~np.asarray(
                self.periodic_axes, dtype=np.bool_
            )
            if np.any(outside):
                raise ValueError(
                    "The batch declares periodic axes outside the layout's "
                    "periodicity class."
                )

    def prepare(self, batch: AtomisticBatch, /) -> AtomisticGraphPayload:
        """Host boundary: prepare the layout-bound graph payload of ``batch``.

        Material case identities (case address words of ``structure_ids``) and
        stable particle IDs become dynamic leaves; nothing derived from content
        enters a static field. Image layouts need concrete positions and cells
        (host binning); route search and schedule preparation run in one
        compiled executable per layout (stable module-level entrypoints).
        """
        self.require_batch(batch)
        cases = self.case_capacity
        words = atomistic_case_address_words(batch.structure_ids)
        periodic = (
            jnp.zeros((cases, 3), dtype=jnp.bool_)
            if batch.periodic_axes is None
            else jnp.asarray(batch.periodic_axes, dtype=jnp.bool_)
        )
        if self.lattice_rank == 0:
            topology = _layout_finite_topology(
                self,
                batch.atom_mask.reshape((-1,)),
                batch.particle_ids.reshape((-1,)),
                words,
                batch.atomic_numbers.reshape((-1,)),
                batch.atom_type_ids.reshape((-1,)),
                batch.masses.reshape((-1,)),
            )
        else:
            lattices, origins = _binning_lattices(
                batch, self.search_radius, self.binning_lengths
            )
            dtype = batch.positions.dtype
            topology = _layout_image_topology(
                self,
                batch.positions,
                batch.atom_mask,
                batch.particle_ids,
                jnp.asarray(lattices, dtype=dtype),
                jnp.asarray(origins, dtype=dtype),
                periodic,
                (
                    jnp.zeros((cases, 3, 3), dtype=dtype)
                    if batch.cells is None
                    else batch.cells
                ),
                words,
                batch.atomic_numbers.reshape((-1,)),
                batch.atom_type_ids.reshape((-1,)),
                batch.masses.reshape((-1,)),
            )
        return AtomisticGraphPayload(
            layout=self,
            topology=topology,
            positions=batch.positions,
            element_mask=batch.element_mask,
            cells=batch.cells if self.cells else None,
            periodic_axes=periodic if self.cells else None,
        )


@eqx.filter_jit
def _layout_finite_topology(
    layout: AtomisticGraphLayout,
    atom_mask: Array,
    particle_ids: Array,
    words: Array,
    atomic_numbers: Array,
    atom_type_ids: Array,
    masses: Array,
    /,
) -> AtomisticGraphTopology:
    """Finite all-pairs topology; routes depend only on the layout capacities."""
    cases, atoms = layout.case_capacity, layout.atom_capacity
    senders, receivers, edge_cases, rank = _dense_routes(cases, atoms)
    return _build_topology(
        layout.execution.streamed,
        capacity_bounded=True,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        edge_cases=jnp.asarray(edge_cases),
        candidate_mask=atom_mask[senders] & atom_mask[receivers],
        image_shifts=jnp.zeros((senders.shape[0], 0), dtype=jnp.int32),
        stable_route_ids=jnp.asarray(rank),
        particle_ids=particle_ids,
        case_identity_words=words,
        atomic_numbers=atomic_numbers,
        atom_type_ids=atom_type_ids,
        masses=masses,
        atom_mask=atom_mask,
        atom_cases=jnp.repeat(jnp.arange(cases, dtype=jnp.int32), atoms),
        periodic_mask=jnp.zeros((cases, 0), dtype=jnp.bool_),
        epoch=jnp.zeros((), dtype=jnp.int32),
        overflow=jnp.zeros((cases,), dtype=jnp.bool_),
        search_failure=jnp.zeros((cases,), dtype=jnp.bool_),
        evidence=None,
        minimum_image_cell=None,
        reference_cell_vectors=None,
        reference_positions=None,
        reference_lattice=None,
        lattice_origin=None,
        wrap_counts=None,
        stencil_extents=None,
        case_count=cases,
        atom_capacity=atoms,
        lattice_rank=0,
        search_radius=float("inf"),
        skin=0.0,
        topology_id=layout.topology_id,
        support_id=layout.layout_id,
        execution_id=layout.execution.plan_id,
    )


@eqx.filter_jit
def _layout_image_topology(
    layout: AtomisticGraphLayout,
    positions: Array,
    atom_mask: Array,
    particle_ids: Array,
    lattices: Array,
    origins: Array,
    periodic: Array,
    cells: Array,
    words: Array,
    atomic_numbers: Array,
    atom_type_ids: Array,
    masses: Array,
    /,
) -> AtomisticGraphTopology:
    """Image topology from the layout's fixed search; one executable per layout.

    The search evidence keeps its canonical classification: storage
    exceedance (``capacity_failure``) is ``overflow``; stencil, domain,
    nonfinite-lattice, representation and receiver failures
    (``scientific_failure``) are ``search_failure``.
    """
    search = layout.search
    if search is None:
        raise RuntimeError("An image layout lacks its route search.")
    result = search.routes(
        positions,
        atom_mask,
        particle_ids,
        lattices,
        origins,
        periodic,
        support_id=layout.layout_id,
        relation_schema_id=layout.topology_id,
    )
    relation = result.relation
    evidence = result.evidence
    cases, atoms = layout.case_capacity, layout.atom_capacity
    return _build_topology(
        layout.execution.streamed,
        capacity_bounded=True,
        senders=relation.source_indices,
        receivers=relation.receiver_indices,
        edge_cases=relation.route_cases,
        candidate_mask=relation.valid,
        image_shifts=relation.image_shifts,
        stable_route_ids=jnp.arange(relation.capacity, dtype=jnp.int32),
        particle_ids=particle_ids.reshape((-1,)),
        case_identity_words=words,
        atomic_numbers=atomic_numbers,
        atom_type_ids=atom_type_ids,
        masses=masses,
        atom_mask=atom_mask.reshape((-1,)),
        atom_cases=jnp.repeat(jnp.arange(cases, dtype=jnp.int32), atoms),
        periodic_mask=periodic,
        epoch=jnp.zeros((), dtype=jnp.int32),
        overflow=evidence.capacity_failure,
        search_failure=evidence.scientific_failure,
        evidence=evidence,
        minimum_image_cell=None,
        reference_cell_vectors=cells,
        reference_positions=positions.reshape((-1, 3)),
        reference_lattice=lattices,
        lattice_origin=origins,
        wrap_counts=result.wrap_counts,
        stencil_extents=result.stencil_extents,
        case_count=cases,
        atom_capacity=atoms,
        lattice_rank=3,
        search_radius=layout.search_radius,
        skin=layout.skin,
        topology_id=layout.topology_id,
        support_id=layout.layout_id,
        execution_id=layout.execution.plan_id,
    )


@final
class AtomisticGraphPayload(StrictModule, NonTrainableState):
    """Layout-bound numerical graph input of one host batch.

    Static structure is the layout's alone. Node metadata, stable particle
    IDs and case address words are dynamic topology leaves; ``positions``
    ``(case, atom, 3)`` is the evaluation geometry and ``cells``/``periodic_axes``
    are present exactly when ``layout.cells``. Host provenance (batch and
    structure IDs, topology content) stays with the ``AtomisticBatch`` and
    ``topology.content_id()``.
    """

    layout: AtomisticGraphLayout
    topology: AtomisticGraphTopology
    positions: Array
    element_mask: Array
    cells: Array | None
    periodic_axes: Array | None

    @property
    def layout_id(self) -> str:
        return self.layout.layout_id

    @property
    def scale(self) -> AtomisticScaleContract:
        return self.layout.scale

    @property
    def execution(self) -> AtomisticGraphExecutionPlan:
        return self.layout.execution

    @property
    def case_count(self) -> int:
        return self.layout.case_capacity

    @property
    def atom_capacity(self) -> int:
        return self.layout.atom_capacity

    def _cases(self, value: Array, /) -> Array:
        return value.reshape((self.case_count, self.atom_capacity))

    @property
    def atom_mask(self) -> Array:
        """Active atoms ``(case, atom)``."""
        return self._cases(self.topology.atom_mask)

    @property
    def atom_cases(self) -> Array:
        """Flat case-major owning case of every atom row."""
        return self.topology.atom_cases

    @property
    def atomic_numbers(self) -> Array:
        return self._cases(self.topology.atomic_numbers)

    @property
    def atom_type_ids(self) -> Array:
        return self._cases(self.topology.atom_type_ids)

    @property
    def masses(self) -> Array:
        return self._cases(self.topology.masses)

    def graph(
        self,
        *,
        cutoff: float,
        positions: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
        image_counts: ArrayLike | None = None,
        selection: ImageNeighborSelectionPolicy | None = None,
    ) -> AtomisticGraph:
        """Bind geometry to this payload's routes; pure and traceable.

        ``positions`` default to the payload geometry (``(case, atom, 3)`` or
        flat); ``cell_vectors`` default to the payload cells for image layouts.
        Runtime periodicity is the payload's.
        """
        coordinate = self.positions if positions is None else jnp.asarray(positions)
        size = self.case_count * self.atom_capacity
        if coordinate.shape not in ((self.case_count, self.atom_capacity, 3), (size, 3)):
            raise ValueError("positions must have the payload shape (case, atom, 3).")
        vectors = (
            self.cells
            if cell_vectors is None and self.topology.lattice_rank > 0
            else cell_vectors
        )
        return bind_atomistic_graph(
            self.topology,
            self.layout.execution,
            coordinate.reshape((size, 3)),
            cutoff=cutoff,
            cell_vectors=vectors,
            periodic_axes=(
                jnp.zeros((self.case_count, 3), dtype=jnp.bool_)
                if self.periodic_axes is None
                else self.periodic_axes
            ),
            image_counts=image_counts,
            selection=selection,
        )

    def admit(self, batch: AtomisticBatch, /) -> AtomisticGraphPayload:
        """Host boundary: bind these routes to ``batch`` at its geometry.

        The layout class is refused on the host. Node metadata, stable
        particle IDs, case address words, element mask and periodic axes are
        compared numerically per case (no callback); a case of other material
        content fails as a binding (scientific) failure in
        ``topology.search_failure``, never as capacity, so growing a resource
        cannot admit foreign material. Moved positions or deformed cells are
        admitted through the stored image certificate.
        """
        self.layout.require_batch(batch)
        cases, atoms = self.case_count, self.atom_capacity
        periodic = (
            jnp.zeros((cases, 3), dtype=jnp.bool_)
            if batch.periodic_axes is None
            else jnp.asarray(batch.periodic_axes, dtype=jnp.bool_)
        )
        topology = self.topology
        atom_pairs = (
            (topology.atomic_numbers, batch.atomic_numbers),
            (topology.atom_type_ids, batch.atom_type_ids),
            (topology.masses, batch.masses),
            (topology.atom_mask, batch.atom_mask),
            (topology.particle_ids, batch.particle_ids),
            (self.element_mask, batch.element_mask),
        )
        foreign = jnp.any(
            topology.case_identity_words
            != atomistic_case_address_words(batch.structure_ids),
            axis=1,
        )
        for stored, current in atom_pairs:
            foreign = foreign | jnp.any(
                stored.reshape((cases, atoms)) != current.reshape((cases, atoms)),
                axis=1,
            )
        if topology.lattice_rank > 0:
            foreign = foreign | jnp.any(topology.periodic_mask != periodic, axis=1)
        return AtomisticGraphPayload(
            layout=self.layout,
            topology=eqx.tree_at(
                lambda value: value.search_failure,
                topology,
                topology.search_failure | foreign,
            ),
            positions=batch.positions,
            element_mask=batch.element_mask,
            cells=batch.cells if self.layout.cells else None,
            periodic_axes=periodic if self.layout.cells else None,
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
    common: _SystemTopologyFields = {
        "particle_ids": system.plan.particle_ids,
        "case_identity_words": atomistic_case_address_words((system.prepared_id,)),
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
        "support_id": system.support_schema_id,
        "search_failure": jnp.zeros((1,), dtype=jnp.bool_),
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
        if relation.support_id != system.particle_discretization().support.support_id:
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
            topology_id=pairs.relation_schema_id,
            **common,
        )
    else:
        raise TypeError(
            "neighborhood must be a ParticleNeighborhoodState or "
            "ParticleImageNeighborhoodState."
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
    "AtomisticGraphLayout",
    "AtomisticGraphPayload",
    "AtomisticGraphTopology",
    "bind_atomistic_graph",
    "particle_atomistic_graph_topology",
    "realize_particle_atomistic_graph",
]
