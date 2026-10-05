#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation, KeyGroupPlan
from ...typing import checked
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from .._periodic_cell import lattice_right_inverse_with_status, PeriodicCell
from ._core import ParticleDiscretization
from ._image_relation import (
    checked_int32_add,
    checked_int32_shift,
    ParticleImageRelation,
    ParticleImageRelationEvidence,
    representation_shifts,
    symmetric_int32,
)
from ._precision import ParticleRealization


_INT32_LIMIT = 2**31 - 1


class ParticleImageCapacity(StrictModule, NonTrainableState):
    """Separately charged fixed capacities of one image-aware route search.

    Cell occupancy, stored directed routes, receiver degree and the complete
    integer image stencil are distinct budgets.  Candidate slots follow from
    these and the search stencil and are guarded by the owning plan.
    """

    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_images: int = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_particles_per_cell: int,
        maximum_edges: int,
        maximum_degree: int,
        maximum_images: int,
    ) -> None:
        values = {
            "maximum_particles_per_cell": int(maximum_particles_per_cell),
            "maximum_edges": int(maximum_edges),
            "maximum_degree": int(maximum_degree),
            "maximum_images": int(maximum_images),
        }
        for name, value in values.items():
            if value <= 0 or value > _INT32_LIMIT:
                raise ValueError(f"{name} must be a positive int32 capacity.")
        self.maximum_particles_per_cell = values["maximum_particles_per_cell"]
        self.maximum_edges = values["maximum_edges"]
        self.maximum_degree = values["maximum_degree"]
        self.maximum_images = values["maximum_images"]
        self.capacity_id = canonical_fingerprint(
            {"kind": "particle-image-capacity", **values}
        )

    def dominates(self, other: ParticleImageCapacity, /) -> bool:
        return (
            self.maximum_particles_per_cell >= other.maximum_particles_per_cell
            and self.maximum_edges >= other.maximum_edges
            and self.maximum_degree >= other.maximum_degree
            and self.maximum_images >= other.maximum_images
        )

    def covers(self, evidence: ParticleImageRelationEvidence, /) -> bool:
        """Host decision whether concrete evidence requirements fit this capacity."""
        return (
            int(np.max(np.asarray(evidence.required_cell_occupancy)))
            <= self.maximum_particles_per_cell
            and int(np.sum(np.asarray(evidence.required_edges))) <= self.maximum_edges
            and int(np.max(np.asarray(evidence.required_degree))) <= self.maximum_degree
            and int(np.max(np.asarray(evidence.required_images))) <= self.maximum_images
        )


class ParticleImageCapacityLadder(StrictModule, NonTrainableState):
    """Declared finite, monotone sequence of image capacities.

    Capacity growth is a host lifecycle transaction: callers keep the accepted
    physical/RNG state, select the next entry that covers the observed
    requirements, rebuild, and retry the same attempt.  Scientific or
    geometric failures are refused, and the ladder never grows past its last
    declared entry.
    """

    entries: tuple[ParticleImageCapacity, ...]
    ladder_id: str = eqx.field(static=True)

    def __init__(self, entries: tuple[ParticleImageCapacity, ...], /) -> None:
        values = tuple(entries)
        if not values:
            raise ValueError("A capacity ladder requires at least one entry.")
        for entry in values:
            if not isinstance(entry, ParticleImageCapacity):
                raise TypeError("Capacity ladder entries must be ParticleImageCapacity.")
        for previous, entry in zip(values[:-1], values[1:], strict=True):
            if not entry.dominates(previous) or entry.capacity_id == previous.capacity_id:
                raise ValueError("Capacity ladder entries must strictly grow.")
        self.entries = values
        self.ladder_id = canonical_fingerprint(
            {
                "kind": "particle-image-capacity-ladder",
                "entries": [entry.capacity_id for entry in values],
            }
        )

    def select(
        self,
        evidence: ParticleImageRelationEvidence,
        current: ParticleImageCapacity,
        /,
    ) -> ParticleImageCapacity:
        """Choose the next declared capacity for a capacity-only failure.

        Requires concrete host evidence; calling this inside a traced loop fails
        instead of synchronizing.
        """
        identifiers = [entry.capacity_id for entry in self.entries]
        if current.capacity_id not in identifiers:
            raise ValueError("Current capacity is not an entry of this ladder.")
        if bool(np.any(np.asarray(evidence.scientific_failure))):
            raise ValueError(
                "Image neighborhood failure is geometric or scientific, not "
                "capacity-only; refusing a capacity retry."
            )
        if not bool(np.any(np.asarray(evidence.capacity_failure))):
            raise ValueError("Image neighborhood evidence has no capacity failure.")
        for entry in self.entries[identifiers.index(current.capacity_id) + 1 :]:
            if entry.covers(evidence):
                return entry
        raise RuntimeError(
            "Image capacity ladder is exhausted for the observed requirements."
        )


class _LatticeFrame(StrictModule):
    wrapped: Array
    fractional: Array
    images: Array
    usable: Array
    domain_violation: Array
    nonfinite: Array
    unrepresentable: Array
    solved: Array
    reach: Array


def _lattice_frame(
    positions: Array,
    active: Array,
    vectors: Array,
    origin: Array,
    periodic: Array,
    /,
) -> _LatticeFrame:
    raw_inverse, solved = lattice_right_inverse_with_status(vectors)
    inverse = jnp.where(solved[:, None, None], raw_inverse, 0.0)
    finite = jnp.all(jnp.isfinite(positions), axis=-1) & solved[:, None]
    safe = jnp.where(finite[..., None], positions, origin[:, None, :])
    fractional = contract(
        "cnd,cdr->cnr", safe - origin[:, None, :], inverse, backend="jax"
    )
    floors = jax.lax.stop_gradient(
        jnp.where(periodic[:, None, :], jnp.floor(fractional), 0.0)
    )
    # Wrap counts are stored as int32 image translations: range-check the
    # floating floor before the cast instead of letting it saturate.  The
    # symmetric bound ``|n| < 2**31`` is exact in every float dtype.
    representable = jnp.all(jnp.abs(floors) < 2.0**31, axis=-1)
    images = jnp.where(representable[..., None], floors, 0.0).astype(jnp.int32)
    wrapped_fractional = fractional - images.astype(fractional.dtype)
    inside = jnp.all(
        periodic[:, None, :] | ((wrapped_fractional >= 0.0) & (wrapped_fractional < 1.0)),
        axis=-1,
    )
    wrapped = safe - contract(
        "cnr,crd->cnd", images.astype(safe.dtype), vectors, backend="jax"
    )
    return _LatticeFrame(
        wrapped=wrapped,
        fractional=jnp.clip(wrapped_fractional, 0.0, 1.0),
        images=images,
        usable=active & finite & representable & inside,
        domain_violation=active & finite & representable & ~inside,
        nonfinite=active & ~finite,
        unrepresentable=active & finite & ~representable,
        solved=solved,
        reach=jnp.sqrt(jnp.sum(inverse * inverse, axis=1)),
    )


def _required_images(reach: Array, radius: float, periodic: Array, /) -> Array:
    extents = jnp.floor(1.0 + radius * reach)
    factors = jnp.where(periodic, 2.0 * extents + 1.0, 1.0)
    return jnp.minimum(jnp.prod(factors, axis=-1), float(_INT32_LIMIT)).astype(jnp.int32)


def _stencil_extents(reach: Array, radius: float, periodic: Array, /) -> Array:
    """Complete wrapped-frame image extents ``floor(1 + R * reach)`` per axis."""
    return jnp.where(
        periodic,
        jnp.minimum(jnp.floor(1.0 + radius * reach), float(_INT32_LIMIT)),
        0.0,
    ).astype(jnp.int32)


class _PackedRoutes(StrictModule):
    relation: ParticleImageRelation
    degree: Array
    required_edges: Array
    stored_edges: Array
    shift_overflow: Array


class ImageCertificate(StrictModule, NonTrainableState):
    """Per-case evidence that a cached image relation remains complete.

    For a relation searched at radius ``R_s`` in a wrapped build frame with
    complete extents ``K``, the routes within ``radius`` are all stored while
    ``coverage_margin > 0`` (every such image still has ``|n_i| <= K_i``) and
    ``2 * maximum_displacement + cell_deformation <= R_s - radius`` (no stored
    or absent route in that stencil moved further than the skin).
    """

    maximum_displacement: Array
    cell_deformation: Array
    coverage_margin: Array

    def valid(self, skin: ArrayLike, /) -> Array:
        spent = 2.0 * self.maximum_displacement + self.cell_deformation
        return (jnp.isfinite(spent) & (spent <= jnp.asarray(skin, spent.dtype))) & (
            self.coverage_margin > 0.0
        )


def image_certificate(
    reference_positions: ArrayLike,
    positions: ArrayLike,
    reference_vectors: ArrayLike,
    vectors: ArrayLike,
    origin: ArrayLike,
    wrap_counts: ArrayLike,
    stencil_extents: ArrayLike,
    active: ArrayLike,
    periodic: ArrayLike,
    radius: float,
    /,
) -> ImageCertificate:
    """Evaluate the stored-and-absent image certificate for case-batched inputs.

    ``positions`` must be in the same lattice representation as
    ``reference_positions`` (callers first undo whole-cell rewrapping).
    Shapes: positions ``(case, N, d)``, vectors ``(case, r, d)``, origin
    ``(case, d)``, wrap counts ``(case, N, r)``, extents/periodic
    ``(case, r)``, active ``(case, N)``.
    """
    current = jnp.asarray(positions)
    reference = jnp.asarray(reference_positions, dtype=current.dtype)
    lattice = jnp.asarray(vectors, dtype=current.dtype)
    reference_lattice = jnp.asarray(reference_vectors, dtype=current.dtype)
    wraps = jnp.asarray(wrap_counts).astype(current.dtype)
    mask = jnp.asarray(active, dtype=jnp.bool_)
    periodic_mask = jnp.asarray(periodic, dtype=jnp.bool_)
    extents = jnp.asarray(stencil_extents).astype(current.dtype)
    cell_delta = lattice - reference_lattice
    motion = (
        current - reference - contract("cnr,crd->cnd", wraps, cell_delta, backend="jax")
    )
    distance = jnp.sqrt(jnp.sum(motion * motion, axis=-1))
    maximum_displacement = jnp.max(jnp.where(mask, distance, 0.0), axis=-1)
    row_change = jnp.sqrt(jnp.sum(cell_delta * cell_delta, axis=-1))
    cell_deformation = jnp.sum(extents * row_change, axis=-1)
    raw_inverse, solved = lattice_right_inverse_with_status(lattice)
    inverse = jnp.where(solved[:, None, None], raw_inverse, 0.0)
    frame = contract(
        "cnd,cdr->cnr",
        current
        - contract("cnr,crd->cnd", wraps, lattice, backend="jax")
        - jnp.asarray(origin, dtype=current.dtype)[:, None, :],
        inverse,
        backend="jax",
    )
    upper = jnp.max(jnp.where(mask[..., None], frame, -jnp.inf), axis=1)
    lower = jnp.min(jnp.where(mask[..., None], frame, jnp.inf), axis=1)
    spread = jnp.where(jnp.any(mask, axis=-1)[:, None], upper - lower, 0.0)
    reach = jnp.sqrt(jnp.sum(inverse * inverse, axis=1))
    margin = extents + 1.0 - spread - radius * reach
    coverage = jnp.where(
        solved,
        jnp.min(jnp.where(periodic_mask, margin, jnp.inf), axis=-1),
        -jnp.inf,
    )
    return ImageCertificate(maximum_displacement, cell_deformation, coverage)


def _pack_routes(
    candidate: Array,
    sources: Array,
    wrapped_shifts: Array,
    images: Array,
    stable_ids: Array,
    *,
    edge_capacity: int,
    support_id: str,
    relation_schema_id: str,
) -> _PackedRoutes:
    """Compact candidate routes into canonical receiver-major stable order.

    Raw shifts re-express wrapped-frame translations for the caller's
    coordinate representation: ``n = n_wrapped - w[receiver] + w[source]``.
    A candidate whose ``n`` leaves the symmetric int32 range is reported per
    case in ``shift_overflow`` instead of being stored wrapped around.
    Valid routes are ordered by ``(case, receiver id, source id, n)``; padding
    routes follow with zero indices and shifts.
    """
    case_count, particle_capacity, slots = candidate.shape
    rank = wrapped_shifts.shape[-1]
    source_images = jnp.take_along_axis(
        images, sources.reshape((case_count, particle_capacity * slots, 1)), axis=1
    ).reshape((case_count, particle_capacity, slots, rank))
    raw_shifts, overflow = checked_int32_shift(
        wrapped_shifts.astype(jnp.int32),
        source_images,
        jnp.broadcast_to(images[:, :, None, :], source_images.shape),
    )
    shift_overflow = jnp.any(candidate & jnp.any(overflow, axis=-1), axis=(1, 2))
    degree = jnp.sum(candidate, axis=-1, dtype=jnp.int32)
    required_edges = jnp.sum(degree, axis=-1, dtype=jnp.int32)
    flat_valid = candidate.reshape((-1,))
    total = jnp.sum(flat_valid, dtype=jnp.int32)
    selected = jax.lax.stop_gradient(
        jnp.nonzero(flat_valid, size=edge_capacity, fill_value=0)[0]
    )
    stored = jnp.minimum(total, edge_capacity)
    route_valid = jnp.arange(edge_capacity, dtype=jnp.int32) < stored
    per_case = particle_capacity * slots
    route_case = (selected // per_case).astype(jnp.int32)
    receiver_local = ((selected // slots) % particle_capacity).astype(jnp.int32)
    source_local = sources.reshape((-1,))[selected].astype(jnp.int32)
    shifts = raw_shifts.reshape((-1, rank))[selected]
    receiver_ids = stable_ids[route_case, receiver_local]
    source_ids = stable_ids[route_case, source_local]
    keys = tuple(shifts[:, axis] for axis in reversed(range(rank))) + (
        source_ids,
        receiver_ids,
        route_case,
        (~route_valid).astype(jnp.int32),
    )
    order = jax.lax.stop_gradient(jnp.lexsort(keys))
    route_valid = route_valid[order]
    route_case = jnp.where(route_valid, route_case[order], case_count - 1)
    padding_endpoint = (case_count - 1) * particle_capacity
    receivers = jnp.where(
        route_valid,
        route_case * particle_capacity + receiver_local[order],
        padding_endpoint,
    )
    sources_global = jnp.where(
        route_valid,
        route_case * particle_capacity + source_local[order],
        padding_endpoint,
    )
    shifts = jnp.where(route_valid[:, None], shifts[order], 0)
    receiver_ids = jnp.where(route_valid, receiver_ids[order], 0)
    source_ids = jnp.where(route_valid, source_ids[order], 0)
    stored_edges = (
        jnp.zeros((case_count,), dtype=jnp.int32)
        .at[route_case]
        .add(route_valid.astype(jnp.int32))
    )
    size = case_count * particle_capacity
    relation = ParticleImageRelation(
        EdgeRelation(
            sources_global,
            receivers,
            source_size=size,
            target_size=size,
            valid=route_valid,
        ),
        source_ids,
        receiver_ids,
        shifts,
        route_case,
        case_count=case_count,
        particle_capacity=particle_capacity,
        support_id=support_id,
        relation_schema_id=relation_schema_id,
    )
    return _PackedRoutes(relation, degree, required_edges, stored_edges, shift_overflow)


class ImageRouteSearchResult(StrictModule, NonTrainableState):
    """Routes plus the build frame needed by lifecycle certificates."""

    relation: ParticleImageRelation
    evidence: ParticleImageRelationEvidence
    wrap_counts: Array
    stencil_extents: Array


class AbstractImageRouteSearch(StrictModule, NonTrainableState):
    """Case-batched fixed-capacity image route search over row lattices.

    Inputs are ``positions[case, particle, d]``, ``vectors[case, r, d]``,
    ``origin[case, d]`` and ``periodic[case, r]``.  Cases never exchange routes.
    """

    radius: eqx.AbstractVar[float]
    case_count: eqx.AbstractVar[int]
    particle_capacity: eqx.AbstractVar[int]
    lattice_rank: eqx.AbstractVar[int]
    capacity: eqx.AbstractVar[ParticleImageCapacity]
    candidate_slot_count: eqx.AbstractVar[int]
    search_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def _candidates(
        self, frame: _LatticeFrame, vectors: Array, periodic: Array, stable_ids: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """Return candidate mask, sources, wrapped shifts, occupancy, cell overflow,
        required offset extents, and stencil overflow."""
        raise NotImplementedError

    def routes(
        self,
        positions: ArrayLike,
        active: ArrayLike,
        stable_ids: ArrayLike,
        vectors: ArrayLike,
        origin: ArrayLike,
        periodic: ArrayLike,
        /,
        *,
        support_id: str,
        relation_schema_id: str,
    ) -> ImageRouteSearchResult:
        value = jnp.asarray(positions)
        expected = (self.case_count, self.particle_capacity)
        if value.ndim != 3 or value.shape[:2] != expected:
            raise ValueError(f"positions must have shape {expected} + (d,).")
        dimension = value.shape[-1]
        lattice = jnp.asarray(vectors, dtype=value.dtype)
        if lattice.shape != (self.case_count, self.lattice_rank, dimension):
            raise ValueError("vectors must have shape (case, lattice rank, d).")
        shift_origin = jnp.asarray(origin, dtype=value.dtype)
        if shift_origin.shape != (self.case_count, dimension):
            raise ValueError("origin must have shape (case, d).")
        periodic_mask = jnp.asarray(periodic, dtype=jnp.bool_)
        if periodic_mask.shape != (self.case_count, self.lattice_rank):
            raise ValueError("periodic must have shape (case, lattice rank).")
        active_mask = jnp.asarray(active, dtype=jnp.bool_)
        ids = jnp.asarray(stable_ids)
        if active_mask.shape != expected or ids.shape != expected:
            raise ValueError("active and stable_ids must have shape (case, particle).")
        frame = _lattice_frame(value, active_mask, lattice, shift_origin, periodic_mask)
        (
            candidate,
            sources,
            wrapped_shifts,
            occupancy,
            cell_overflow,
            required_offsets,
            stencil_overflow,
        ) = self._candidates(frame, lattice, periodic_mask, ids)
        packed = _pack_routes(
            candidate,
            sources,
            wrapped_shifts,
            frame.images,
            ids,
            edge_capacity=self.capacity.maximum_edges,
            support_id=support_id,
            relation_schema_id=relation_schema_id,
        )
        required_degree = jnp.max(packed.degree, axis=-1)
        required_images = _required_images(frame.reach, self.radius, periodic_mask)
        total_required = jnp.sum(packed.required_edges, dtype=jnp.int32)
        evidence = ParticleImageRelationEvidence(
            active_particles=jnp.sum(active_mask, axis=-1, dtype=jnp.int32),
            required_cell_occupancy=occupancy,
            required_edges=packed.required_edges,
            stored_edges=packed.stored_edges,
            required_degree=required_degree,
            required_images=required_images,
            required_offset_extents=required_offsets,
            cell_overflow=cell_overflow,
            stencil_overflow=stencil_overflow,
            image_overflow=required_images > self.capacity.maximum_images,
            edge_overflow=(packed.stored_edges < packed.required_edges)
            | (total_required > self.capacity.maximum_edges),
            degree_overflow=required_degree > self.capacity.maximum_degree,
            domain_violation=jnp.any(frame.domain_violation, axis=-1),
            # Lattice validity is case-level: an empty or all-inactive case
            # under a singular or non-finite cell is still refused.
            nonfinite=jnp.any(frame.nonfinite, axis=-1)
            | ~jnp.all(jnp.isfinite(lattice), axis=(-2, -1))
            | ~frame.solved,
            representation_overflow=jnp.any(frame.unrepresentable, axis=-1)
            | packed.shift_overflow,
        )
        return ImageRouteSearchResult(
            packed.relation,
            evidence,
            frame.images.reshape((-1, self.lattice_rank)),
            _stencil_extents(frame.reach, self.radius, periodic_mask),
        )


def _host_reach(vectors: np.ndarray, /) -> np.ndarray:
    matrices = np.asarray(vectors, dtype=np.float64)
    if matrices.ndim != 3 or np.any(~np.isfinite(matrices)):
        raise ValueError("Reference lattices must be finite (case, rank, d) arrays.")
    reach = []
    for matrix in matrices:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values[-1] <= np.finfo(np.float64).eps * max(singular_values[0], 1.0):
            raise ValueError("Reference lattice is rank deficient.")
        inverse = matrix.T @ np.linalg.inv(matrix @ matrix.T)
        reach.append(np.sqrt(np.sum(inverse * inverse, axis=0)))
    return np.stack(reach)


class CellListImageRouteSearch(AbstractImageRouteSearch):
    """Fractional-lattice cell list with explicit image-shifted neighbor cells.

    Particles are wrapped on periodic axes and binned on a static fractional
    grid whose cell perpendicular height is at least ``radius`` when the cell
    admits it.  Neighbor cells span ``ceil(radius * grid * reach)`` cells per
    axis, so short periodic axes enumerate several translated copies of the
    same cell without duplicates: each unwrapped cell offset maps to one
    ``(cell, image)`` pair.  Grouping reuses the native ``KeyGroupPlan``.
    """

    offsets: Array
    key_groups: KeyGroupPlan
    radius: float = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    cell_strides: tuple[int, ...] = eqx.field(static=True)
    offset_extents: tuple[int, ...] = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    capacity: ParticleImageCapacity
    candidate_slot_count: int = eqx.field(static=True)
    search_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        reference_vectors: ArrayLike,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
        maximum_candidate_slots: int,
        deformation_margin: float = 0.0,
    ) -> None:
        value = float(radius)
        margin = float(deformation_margin)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Image search radius must be finite and positive.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("deformation_margin must be finite and nonnegative.")
        reach = _host_reach(np.asarray(reference_vectors))
        cases, rank = map(int, reach.shape)
        particles = int(particle_capacity)
        if particles <= 0:
            raise ValueError("particle_capacity must be positive.")
        heights = 1.0 / np.max(reach, axis=0)
        grid = tuple(max(1, int(np.floor(height / value))) for height in heights)
        cell_count = prod(grid)
        if cell_count > _INT32_LIMIT:
            raise ValueError("Image cell grid exceeds int32 cell keys.")
        envelope = value * (1.0 + margin)
        extents = tuple(
            max(1, int(np.ceil(envelope * grid[axis] * np.max(reach[:, axis]))))
            for axis in range(rank)
        )
        # Admit the candidate workspace from the scalar stencil size before any
        # radius-sized offset enumeration is materialized.
        offset_count = prod(2 * extent + 1 for extent in extents)
        slots = cases * particles * offset_count * capacity.maximum_particles_per_cell
        if slots > int(maximum_candidate_slots):
            raise ValueError(
                f"Image cell list requires {slots} candidate slots, exceeding "
                f"maximum_candidate_slots={int(maximum_candidate_slots)}."
            )
        axes = np.meshgrid(
            *(np.arange(-extent, extent + 1, dtype=np.int32) for extent in extents),
            indexing="ij",
        )
        offsets = np.stack([axis.reshape((-1,)) for axis in axes], axis=-1)
        strides = tuple(prod(grid[axis + 1 :]) for axis in range(rank))
        key_groups = KeyGroupPlan(
            particles,
            max(min(particles, cell_count), 1),
            cell_count - 1,
            maximum_group_size=capacity.maximum_particles_per_cell,
            case_shape=(cases,),
        )
        self.offsets = jnp.asarray(offsets)
        self.key_groups = key_groups
        self.radius = value
        self.grid_shape = grid
        self.cell_strides = strides
        self.offset_extents = extents
        self.case_count = cases
        self.particle_capacity = particles
        self.lattice_rank = rank
        self.capacity = capacity
        self.candidate_slot_count = slots
        self.search_id = canonical_fingerprint(
            {
                "kind": "cell-list-image-route-search",
                "radius": value,
                "grid": list(grid),
                "offset_extents": list(extents),
                "case_count": cases,
                "particle_capacity": particles,
                "capacity": capacity.capacity_id,
                "key_groups": key_groups.plan_id,
            }
        )

    def _candidates(
        self, frame: _LatticeFrame, vectors: Array, periodic: Array, stable_ids: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        axis_grid = jnp.asarray(self.grid_shape, dtype=jnp.int32)
        axis_strides = jnp.asarray(self.cell_strides, dtype=jnp.int32)
        grid = axis_grid[None, None, :]
        strides = axis_strides[None, None, :]
        coordinates = jnp.clip(
            jax.lax.stop_gradient(
                jnp.floor(frame.fractional * grid.astype(frame.fractional.dtype))
            ).astype(jnp.int32),
            0,
            grid - 1,
        )
        # Keys stay in the prepared int32 key space: coordinates are clipped to
        # the grid and the grid has fewer than 2**31 cells (checked at prepare),
        # so the reductions are exact in int32 and never default-promote.
        cell_keys = jnp.where(
            frame.usable, jnp.sum(coordinates * strides, axis=-1, dtype=jnp.int32), -1
        ).astype(jnp.int32)
        groups = self.key_groups.build(cell_keys, frame.usable, stable_ids=stable_ids)
        unwrapped = coordinates[:, :, None, :] + self.offsets[None, None, :, :]
        periodic_axis = periodic[:, None, None, :]
        offset_grid = axis_grid[None, None, None, :]
        image = jnp.floor_divide(unwrapped, offset_grid)
        in_range = (unwrapped >= 0) & (unwrapped < offset_grid)
        cell_valid = jnp.all(periodic_axis | in_range, axis=-1) & frame.usable[:, :, None]
        neighbor = jnp.where(
            periodic_axis,
            jnp.mod(unwrapped, offset_grid),
            jnp.clip(unwrapped, 0, offset_grid - 1),
        )
        image = jnp.where(periodic_axis, image, 0)
        neighbor_keys = jnp.sum(
            neighbor * axis_strides[None, None, None, :], axis=-1, dtype=jnp.int32
        )
        offset_count = self.offsets.shape[0]
        lookup = groups.lookup(
            neighbor_keys.reshape((self.case_count, -1)),
            valid=cell_valid.reshape((self.case_count, -1)),
        )
        group_slots = lookup.group_slots
        starts = jnp.take_along_axis(groups.group_starts, group_slots, axis=1)
        counts = jnp.take_along_axis(groups.group_counts, group_slots, axis=1)
        members = self.capacity.maximum_particles_per_cell
        rank = jnp.arange(members, dtype=jnp.int32)
        sorted_positions = jnp.clip(
            starts[..., None] + rank[None, None, :], 0, self.particle_capacity - 1
        )
        sources = jnp.take_along_axis(
            groups.storage_to_logical,
            sorted_positions.reshape((self.case_count, -1)),
            axis=1,
        ).reshape((self.case_count, self.particle_capacity, offset_count * members))
        source_valid = (
            lookup.supported[..., None] & (rank[None, None, :] < counts[..., None])
        ).reshape(sources.shape)
        wrapped_shifts = jnp.broadcast_to(
            -image[:, :, :, None, :],
            (
                self.case_count,
                self.particle_capacity,
                offset_count,
                members,
                self.lattice_rank,
            ),
        ).reshape(sources.shape + (self.lattice_rank,))
        candidate = _within_radius(
            frame, vectors, sources, wrapped_shifts, source_valid, self.radius
        )
        occupancy = groups.evidence.maximum_group_size.astype(jnp.int32)
        cell_overflow = (
            groups.evidence.group_overflow
            | groups.evidence.member_overflow
            | (groups.evidence.duplicate_stable_ids > 0)
        )
        required_offsets = jnp.ceil(
            self.radius * axis_grid[None, :].astype(frame.reach.dtype) * frame.reach
        ).astype(jnp.int32)
        stencil_overflow = jnp.any(
            required_offsets > jnp.asarray(self.offset_extents, dtype=jnp.int32)[None, :],
            axis=-1,
        )
        return (
            candidate,
            sources,
            wrapped_shifts,
            occupancy,
            cell_overflow,
            required_offsets,
            stencil_overflow,
        )


def _within_radius(
    frame: _LatticeFrame,
    vectors: Array,
    sources: Array,
    wrapped_shifts: Array,
    source_valid: Array,
    radius: float,
    /,
) -> Array:
    """Mask candidates inside ``radius`` excluding the zero self translation."""
    case_count, particle_capacity, slots = sources.shape
    flat_sources = sources.reshape((case_count, -1))
    source_positions = jnp.take_along_axis(
        frame.wrapped, flat_sources[..., None], axis=1
    ).reshape((case_count, particle_capacity, slots, -1))
    source_usable = jnp.take_along_axis(frame.usable, flat_sources, axis=1).reshape(
        sources.shape
    )
    translation = contract(
        "cnmr,crd->cnmd",
        wrapped_shifts.astype(vectors.dtype),
        vectors,
        backend="jax",
    )
    displacement = frame.wrapped[:, :, None, :] - source_positions + translation
    squared = jnp.sum(displacement * displacement, axis=-1)
    receivers = jnp.arange(particle_capacity, dtype=sources.dtype)[None, :, None]
    self_zero = (sources == receivers) & jnp.all(wrapped_shifts == 0, axis=-1)
    return (
        source_valid
        & source_usable
        & frame.usable[:, :, None]
        & ~self_zero
        & jax.lax.stop_gradient(squared < radius * radius)
    )


class DenseImageRouteSearch(AbstractImageRouteSearch):
    """Bounded named dense image reference: every particle pair times a stencil.

    Candidate work is ``case * N * N * images`` and is admitted only under an
    explicit ``maximum_dense_routes`` guard.  It exists for validation and small
    systems, not as a hidden default for scalable periodic execution.
    """

    stencil: Array
    radius: float = eqx.field(static=True)
    stencil_extents_bound: tuple[int, ...] = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    capacity: ParticleImageCapacity
    candidate_slot_count: int = eqx.field(static=True)
    search_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        reference_vectors: ArrayLike,
        reference_periodic: ArrayLike,
        capacity: ParticleImageCapacity,
        /,
        *,
        particle_capacity: int,
        maximum_dense_routes: int,
        deformation_margin: float = 0.0,
    ) -> None:
        value = float(radius)
        margin = float(deformation_margin)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Image search radius must be finite and positive.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("deformation_margin must be finite and nonnegative.")
        reach = _host_reach(np.asarray(reference_vectors))
        periodic = np.asarray(reference_periodic, dtype=np.bool_)
        if periodic.shape != reach.shape:
            raise ValueError("reference_periodic must have shape (case, lattice rank).")
        cases, rank = map(int, reach.shape)
        particles = int(particle_capacity)
        if particles <= 0:
            raise ValueError("particle_capacity must be positive.")
        envelope = value * (1.0 + margin)
        extents = tuple(
            int(
                np.max(
                    np.where(
                        periodic[:, axis], np.floor(1.0 + envelope * reach[:, axis]), 0.0
                    )
                )
            )
            for axis in range(rank)
        )
        image_count = prod(2 * extent + 1 for extent in extents)
        if image_count > capacity.maximum_images:
            raise ValueError(
                f"Dense image stencil requires {image_count} images, exceeding "
                f"maximum_images={capacity.maximum_images}."
            )
        routes = cases * particles * particles * image_count
        if routes > int(maximum_dense_routes):
            raise ValueError(
                f"Dense image reference requires {routes} candidate routes, "
                f"exceeding maximum_dense_routes={int(maximum_dense_routes)}."
            )
        grids = np.meshgrid(
            *(np.arange(-extent, extent + 1, dtype=np.int32) for extent in extents),
            indexing="ij",
        )
        stencil = np.stack([grid.reshape((-1,)) for grid in grids], axis=-1)
        self.stencil = jnp.asarray(stencil.reshape((-1, rank)))
        self.radius = value
        self.stencil_extents_bound = extents
        self.case_count = cases
        self.particle_capacity = particles
        self.lattice_rank = rank
        self.capacity = capacity
        self.candidate_slot_count = routes
        self.search_id = canonical_fingerprint(
            {
                "kind": "dense-image-route-search",
                "radius": value,
                "extents": list(extents),
                "case_count": cases,
                "particle_capacity": particles,
                "capacity": capacity.capacity_id,
                "maximum_dense_routes": int(maximum_dense_routes),
            }
        )

    def _candidates(
        self, frame: _LatticeFrame, vectors: Array, periodic: Array, stable_ids: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        images = self.stencil.shape[0]
        sources = jnp.broadcast_to(
            jnp.arange(self.particle_capacity, dtype=jnp.int32)[None, None, :, None],
            (self.case_count, self.particle_capacity, self.particle_capacity, images),
        ).reshape((self.case_count, self.particle_capacity, -1))
        shifts = jnp.broadcast_to(
            self.stencil[None, None, None, :, :],
            (
                self.case_count,
                self.particle_capacity,
                self.particle_capacity,
                images,
                self.lattice_rank,
            ),
        ).reshape(sources.shape + (self.lattice_rank,))
        lawful = jnp.all(periodic[:, None, None, :] | (shifts == 0), axis=-1)
        candidate = _within_radius(frame, vectors, sources, shifts, lawful, self.radius)
        required = _stencil_extents(frame.reach, self.radius, periodic)
        stencil_overflow = jnp.any(
            required > jnp.asarray(self.stencil_extents_bound, dtype=jnp.int32)[None, :],
            axis=-1,
        )
        zeros = jnp.zeros((self.case_count,), dtype=jnp.int32)
        return (
            candidate,
            sources,
            shifts,
            zeros,
            jnp.zeros((self.case_count,), dtype=jnp.bool_),
            required,
            stencil_overflow,
        )


class ParticleImageNeighborhoodState(StrictModule, NonTrainableState):
    """Fixed-capacity image-aware relation plus its build frame and evidence.

    ``relation`` shifts are expressed for ``reference_positions`` (flat
    case-major ``(case * N, d)``, the positions passed to ``build``).
    ``wrap_counts`` and ``stencil_extents`` describe the wrapped build frame
    used by lifecycle certificates for stored and absent images.
    """

    relation: ParticleImageRelation
    evidence: ParticleImageRelationEvidence
    cell_vectors: Array
    reference_positions: Array
    wrap_counts: Array
    stencil_extents: Array
    search_radius: float = eqx.field(static=True)
    prepared_neighborhood_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.evidence.successful)

    @property
    def capacity_failure(self) -> Array:
        return jnp.any(self.evidence.capacity_failure)

    @property
    def scientific_failure(self) -> Array:
        return jnp.any(self.evidence.scientific_failure)

    def require_success(self, value: ArrayLike, /) -> Array:
        return eqx.error_if(
            jnp.asarray(value),
            ~self.successful,
            "Image neighborhood construction failed capacity, stencil, or domain checks.",
        )

    def with_representation_offsets(
        self, offsets: ArrayLike, /
    ) -> ParticleImageNeighborhoodState:
        """Re-express the state for positions ``x' = x - offsets @ H``.

        Route shifts become ``n + offsets[receiver] - offsets[source]``, the
        reference positions move by the same whole translations and wrap counts
        become ``wrap_counts - offsets``, so the build frame and its
        certificates are unchanged.  Offsets, shifts or wrap counts outside
        ``|n| <= 2**31 - 1`` set ``evidence.representation_overflow`` for their
        case instead of wrapping around.
        """
        relation = self.relation
        shifts, route_overflow = representation_shifts(relation, offsets)
        delta, representable = symmetric_int32(jnp.asarray(offsets))
        wraps, wrap_overflow = checked_int32_add(self.wrap_counts, -delta)
        wraps = jnp.where(wrap_overflow, self.wrap_counts, wraps)
        cases, particles = relation.case_count, relation.particle_capacity
        particle_overflow = jnp.any(wrap_overflow | ~representable, axis=-1)
        route_failure = (
            jnp.zeros((cases,), dtype=jnp.int32)
            .at[relation.route_cases]
            .add(route_overflow.astype(jnp.int32))
        ) > 0
        overflow = route_failure | jnp.any(
            particle_overflow.reshape((cases, particles)), axis=-1
        )
        translation = contract(
            "pr,prd->pd",
            delta.astype(self.reference_positions.dtype),
            jnp.repeat(
                self.cell_vectors.astype(self.reference_positions.dtype),
                particles,
                axis=0,
            ),
            backend="jax",
        )
        return ParticleImageNeighborhoodState(
            eqx.tree_at(lambda value: value.image_shifts, relation, shifts),
            eqx.tree_at(
                lambda evidence: evidence.representation_overflow,
                self.evidence,
                self.evidence.representation_overflow | overflow,
            ),
            self.cell_vectors,
            self.reference_positions - translation,
            wraps,
            self.stencil_extents,
            self.search_radius,
            self.prepared_neighborhood_id,
            self.relation_schema_id,
        )


class AbstractParticleImageNeighborhoodPlan(StrictModule, NonTrainableState):
    """Structural plan for an image-aware directed particle relation."""

    key: eqx.AbstractVar[DiscretizationKey]
    cell: eqx.AbstractVar[PeriodicCell]
    backend: eqx.AbstractVar[ParticleRealization]
    search_radius: eqx.AbstractVar[float]
    capacity: eqx.AbstractVar[ParticleImageCapacity]
    plan_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> AbstractPreparedParticleImageNeighborhood:
        raise NotImplementedError

    @abc.abstractmethod
    def with_capacity(
        self, capacity: ParticleImageCapacity, /
    ) -> AbstractParticleImageNeighborhoodPlan:
        raise NotImplementedError


class AbstractPreparedParticleImageNeighborhood(StrictModule, NonTrainableState):
    """Prepared image-aware search bound to one particle support and cell."""

    plan: eqx.AbstractVar[AbstractParticleImageNeighborhoodPlan]
    search: eqx.AbstractVar[AbstractImageRouteSearch]
    particle_ids: eqx.AbstractVar[Array]
    default_active_mask: eqx.AbstractVar[Array]
    key: eqx.AbstractVar[DiscretizationKey]
    cell: eqx.AbstractVar[PeriodicCell]
    backend: eqx.AbstractVar[ParticleRealization]
    particle_capacity: eqx.AbstractVar[int]
    source_support_id: eqx.AbstractVar[str]
    relation_schema_id: eqx.AbstractVar[str]
    particle_discretization_id: eqx.AbstractVar[str]
    numeric_version: eqx.AbstractVar[str]
    preparation: eqx.AbstractVar[PreparationReport]
    prepared_id: eqx.AbstractVar[str]
    artifact_kind: eqx.AbstractVar[str]

    @property
    def box(self) -> PeriodicCell:
        return self.cell

    @property
    def search_radius(self) -> float:
        return self.plan.search_radius

    @property
    def capacity(self) -> ParticleImageCapacity:
        return self.plan.capacity

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def resolved_cell_vectors(
        self, dtype: jnp.dtype, cell_vectors: ArrayLike | None, /
    ) -> Array:
        vectors = (
            self.cell.vectors.astype(dtype)
            if cell_vectors is None
            else jnp.asarray(cell_vectors, dtype=dtype)
        )
        if vectors.shape != self.cell.vectors.shape:
            raise ValueError("cell_vectors must match the prepared PeriodicCell shape.")
        return vectors

    def resolved_active_mask(self, active_mask: ArrayLike | None, /) -> Array:
        if active_mask is None:
            return self.default_active_mask
        requested = jnp.asarray(active_mask, dtype=jnp.bool_)
        if requested.shape != (self.particle_capacity,):
            raise ValueError("active_mask must have particle-capacity shape.")
        return self.default_active_mask & requested

    def build(
        self,
        positions: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        cell_vectors: ArrayLike | None = None,
    ) -> ParticleImageNeighborhoodState:
        """Search directed image routes within ``search_radius`` of ``positions``.

        ``cell_vectors`` defaults to the prepared cell; a deformed cell is
        searched exactly and fails closed when it leaves the prepared stencil
        envelope.
        """
        value = jnp.asarray(positions)
        expected = (self.particle_capacity, self.cell.ambient_dimension)
        if value.shape != expected:
            raise ValueError(f"Particle positions must have shape {expected}.")
        vectors = self.resolved_cell_vectors(value.dtype, cell_vectors)
        active = self.resolved_active_mask(active_mask)
        result = self.search.routes(
            value[None],
            active[None],
            self.particle_ids[None],
            vectors[None],
            self.cell.origin.astype(value.dtype)[None],
            self.cell.periodic_mask[None],
            support_id=self.source_support_id,
            relation_schema_id=self.relation_schema_id,
        )
        return ParticleImageNeighborhoodState(
            result.relation,
            result.evidence,
            vectors[None],
            value,
            result.wrap_counts,
            result.stencil_extents,
            self.search_radius,
            self.prepared_id,
            self.relation_schema_id,
        )


def _image_key(name: str, backend: str, /) -> DiscretizationKey:
    return DiscretizationKey(
        name,
        DiscretizationRole.AUXILIARY,
        domain_labels=("material_point", "image_relation", backend),
    )


def _validated_radius(search_radius: float, /) -> float:
    radius = float(search_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("search_radius must be finite and positive.")
    return radius


class CellListParticleImageNeighborhoodPlan(AbstractParticleImageNeighborhoodPlan):
    """Scalable image-aware cell-list search for triclinic and partial lattices.

    Not limited by the unique-image radius: cutoffs larger than half a lattice
    height enumerate every admitted translation, including nonzero self images.
    Atoms must lie inside ``[0, 1)`` fractional extent on nonperiodic lattice
    axes; violations fail closed.
    """

    search_radius: float = eqx.field(static=True)
    cell: PeriodicCell
    capacity: ParticleImageCapacity
    maximum_candidate_slots: int = eqx.field(static=True)
    deformation_margin: float = eqx.field(static=True)
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        search_radius: float,
        cell: PeriodicCell,
        capacity: ParticleImageCapacity,
        /,
        *,
        maximum_candidate_slots: int = 10_000_000,
        deformation_margin: float = 0.0,
        name: str = "cell-list-particle-image-neighborhood",
        plan_id: str | None = None,
    ) -> None:
        radius = _validated_radius(search_radius)
        slots = int(maximum_candidate_slots)
        margin = float(deformation_margin)
        if slots <= 0:
            raise ValueError("maximum_candidate_slots must be positive.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("deformation_margin must be finite and nonnegative.")
        key = _image_key(name, "cell_list")
        self.search_radius = radius
        self.cell = cell
        self.capacity = capacity
        self.maximum_candidate_slots = slots
        self.deformation_margin = margin
        self.backend = "cell_edge_list"
        self.key = key
        self.name = str(name)
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "cell-list-particle-image-neighborhood-plan",
                "search_radius": radius,
                "cell": cell.cell_id,
                "capacity": capacity.capacity_id,
                "maximum_candidate_slots": slots,
                "deformation_margin": margin,
                "key": key.key_id,
            },
        )

    def with_capacity(
        self, capacity: ParticleImageCapacity, /
    ) -> CellListParticleImageNeighborhoodPlan:
        return CellListParticleImageNeighborhoodPlan(
            self.search_radius,
            self.cell,
            capacity,
            maximum_candidate_slots=self.maximum_candidate_slots,
            deformation_margin=self.deformation_margin,
            name=self.name,
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedCellListParticleImageNeighborhood:
        return PreparedCellListParticleImageNeighborhood(self, particles)


class DenseParticleImageNeighborhoodPlan(AbstractParticleImageNeighborhoodPlan):
    """Bounded named dense image reference for validation and small systems."""

    search_radius: float = eqx.field(static=True)
    cell: PeriodicCell
    capacity: ParticleImageCapacity
    maximum_dense_routes: int = eqx.field(static=True)
    deformation_margin: float = eqx.field(static=True)
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        search_radius: float,
        cell: PeriodicCell,
        capacity: ParticleImageCapacity,
        /,
        *,
        maximum_dense_routes: int,
        deformation_margin: float = 0.0,
        name: str = "dense-particle-image-neighborhood",
        plan_id: str | None = None,
    ) -> None:
        radius = _validated_radius(search_radius)
        routes = int(maximum_dense_routes)
        margin = float(deformation_margin)
        if routes <= 0:
            raise ValueError("maximum_dense_routes must be positive.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("deformation_margin must be finite and nonnegative.")
        key = _image_key(name, "dense_pairs")
        self.search_radius = radius
        self.cell = cell
        self.capacity = capacity
        self.maximum_dense_routes = routes
        self.deformation_margin = margin
        self.backend = "dense_pairs"
        self.key = key
        self.name = str(name)
        self.plan_id = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "dense-particle-image-neighborhood-plan",
                "search_radius": radius,
                "cell": cell.cell_id,
                "capacity": capacity.capacity_id,
                "maximum_dense_routes": routes,
                "deformation_margin": margin,
                "key": key.key_id,
            },
        )

    def with_capacity(
        self, capacity: ParticleImageCapacity, /
    ) -> DenseParticleImageNeighborhoodPlan:
        return DenseParticleImageNeighborhoodPlan(
            self.search_radius,
            self.cell,
            capacity,
            maximum_dense_routes=self.maximum_dense_routes,
            deformation_margin=self.deformation_margin,
            name=self.name,
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedDenseParticleImageNeighborhood:
        return PreparedDenseParticleImageNeighborhood(self, particles)


def _image_preparation(
    search: AbstractImageRouteSearch, particles: ParticleDiscretization, backend: str, /
) -> PreparationReport:
    return PreparationReport(
        capabilities=(
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
            DiscretizationCapability.GEOMETRY_REFRESH,
            DiscretizationCapability.MATRIX_FREE,
        ),
        diagnostics=(
            f"{backend} image search with explicit integer translations",
            "routes are directed and ordered by stable receiver, source and image",
            "nonzero self images are retained; the zero self translation is excluded",
            "cell, image, edge and degree capacities are charged separately",
            "overflow, stencil-envelope and domain failures fail closed",
        ),
        resource_counts={
            "particle_capacity": particles.capacity,
            "candidate_slot_count": search.candidate_slot_count,
            "maximum_particles_per_cell": search.capacity.maximum_particles_per_cell,
            "edge_capacity": search.capacity.maximum_edges,
            "degree_capacity": search.capacity.maximum_degree,
            "image_capacity": search.capacity.maximum_images,
        },
    )


def _require_cell_support(
    cell: PeriodicCell, particles: ParticleDiscretization, /
) -> None:
    if cell.ambient_dimension != particles.ambient_dimension:
        raise ValueError("PeriodicCell dimension does not match particle support.")


class PreparedCellListParticleImageNeighborhood(
    AbstractPreparedParticleImageNeighborhood
):
    plan: CellListParticleImageNeighborhoodPlan
    search: CellListImageRouteSearch
    particle_ids: Array
    default_active_mask: Array
    key: DiscretizationKey
    cell: PeriodicCell
    preparation: PreparationReport
    backend: ParticleRealization = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: CellListParticleImageNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ) -> None:
        _require_cell_support(plan.cell, particles)
        search = CellListImageRouteSearch(
            plan.search_radius,
            np.asarray(plan.cell.vectors)[None],
            plan.capacity,
            particle_capacity=particles.capacity,
            maximum_candidate_slots=plan.maximum_candidate_slots,
            deformation_margin=plan.deformation_margin,
        )
        self.plan = plan
        self.search = search
        _bind_prepared(self, plan, particles, search, "cell-list")


class PreparedDenseParticleImageNeighborhood(AbstractPreparedParticleImageNeighborhood):
    plan: DenseParticleImageNeighborhoodPlan
    search: DenseImageRouteSearch
    particle_ids: Array
    default_active_mask: Array
    key: DiscretizationKey
    cell: PeriodicCell
    preparation: PreparationReport
    backend: ParticleRealization = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    relation_schema_id: str = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: DenseParticleImageNeighborhoodPlan,
        particles: ParticleDiscretization,
        /,
    ) -> None:
        _require_cell_support(plan.cell, particles)
        search = DenseImageRouteSearch(
            plan.search_radius,
            np.asarray(plan.cell.vectors)[None],
            np.asarray(plan.cell.periodic_axes, dtype=np.bool_)[None],
            plan.capacity,
            particle_capacity=particles.capacity,
            maximum_dense_routes=plan.maximum_dense_routes,
            deformation_margin=plan.deformation_margin,
        )
        self.plan = plan
        self.search = search
        _bind_prepared(self, plan, particles, search, "dense")


def _bind_prepared(
    prepared: PreparedCellListParticleImageNeighborhood
    | PreparedDenseParticleImageNeighborhood,
    plan: CellListParticleImageNeighborhoodPlan | DenseParticleImageNeighborhoodPlan,
    particles: ParticleDiscretization,
    search: CellListImageRouteSearch | DenseImageRouteSearch,
    backend: str,
    /,
) -> None:
    preparation = _image_preparation(search, particles, backend)
    relation_schema_id = canonical_fingerprint(
        {
            "kind": "particle-image-relation-schema",
            "plan": plan.plan_id,
            "particles": particles.prepared_id,
            "source_support": particles.support.support_id,
            "search": search.search_id,
        }
    )
    prepared.particle_ids = particles.particle_ids
    prepared.default_active_mask = particles.active_mask
    prepared.key = plan.key
    prepared.cell = plan.cell
    prepared.preparation = preparation
    prepared.backend = plan.backend
    prepared.particle_capacity = particles.capacity
    prepared.source_support_id = particles.support.support_id
    prepared.relation_schema_id = relation_schema_id
    prepared.particle_discretization_id = particles.prepared_id
    prepared.numeric_version = particles.numeric_version
    prepared.artifact_kind = f"{backend}-particle-image-neighborhood"
    prepared.prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-particle-image-neighborhood",
            "plan": plan.plan_id,
            "particles": particles.prepared_id,
            "search": search.search_id,
            "relation_schema": relation_schema_id,
            "preparation": preparation.report_id,
            "numeric_version": particles.numeric_version,
        }
    )


__all__ = [
    "AbstractImageRouteSearch",
    "AbstractParticleImageNeighborhoodPlan",
    "AbstractPreparedParticleImageNeighborhood",
    "CellListImageRouteSearch",
    "CellListParticleImageNeighborhoodPlan",
    "DenseImageRouteSearch",
    "DenseParticleImageNeighborhoodPlan",
    "ImageCertificate",
    "ImageRouteSearchResult",
    "image_certificate",
    "ParticleImageCapacity",
    "ParticleImageCapacityLadder",
    "ParticleImageNeighborhoodState",
    "PreparedCellListParticleImageNeighborhood",
    "PreparedDenseParticleImageNeighborhood",
]
