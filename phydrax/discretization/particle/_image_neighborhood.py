#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import prod
from typing import Literal, NamedTuple

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
from ...typing import checked, parse
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
    resolved_identifier,
)
from .._periodic_cell import (
    _complete_image_shifts,
    _padded_reference_reach,
    lattice_right_inverse_with_status,
    PeriodicCell,
)
from ._core import ParticleDiscretization
from ._image_relation import (
    checked_int32_add,
    checked_int32_shift,
    compact_image_routes,
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

    ``maximum_case_edges`` selects the per-case segment layout: case ``c``
    owns route slots ``[c * B, (c + 1) * B)`` with ``B = maximum_case_edges``,
    overflow is charged per case, and one case's routes can be replaced
    without moving another's.  ``case_count * B`` must not exceed
    ``maximum_edges``.  Without it all cases share one compacted buffer.
    """

    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_images: int = eqx.field(static=True)
    maximum_case_edges: int | None = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_particles_per_cell: int,
        maximum_edges: int,
        maximum_degree: int,
        maximum_images: int,
        maximum_case_edges: int | None = None,
    ) -> None:
        values: dict[str, int] = {
            "maximum_particles_per_cell": int(maximum_particles_per_cell),
            "maximum_edges": int(maximum_edges),
            "maximum_degree": int(maximum_degree),
            "maximum_images": int(maximum_images),
        }
        for name, value in values.items():
            if value <= 0 or value > _INT32_LIMIT:
                raise ValueError(f"{name} must be a positive int32 capacity.")
        identity: dict[str, object] = {"kind": "particle-image-capacity", **values}
        case_edges = None if maximum_case_edges is None else int(maximum_case_edges)
        if case_edges is not None:
            if case_edges <= 0 or case_edges > values["maximum_edges"]:
                raise ValueError(
                    "maximum_case_edges must be a positive segment width no larger "
                    "than maximum_edges."
                )
            identity["maximum_case_edges"] = case_edges
        self.maximum_particles_per_cell = values["maximum_particles_per_cell"]
        self.maximum_edges = values["maximum_edges"]
        self.maximum_degree = values["maximum_degree"]
        self.maximum_images = values["maximum_images"]
        self.maximum_case_edges = case_edges
        self.capacity_id = canonical_fingerprint(identity)

    def route_buffer(self, case_count: int, /) -> int:
        """Return the stored route capacity for ``case_count`` cases."""
        if self.maximum_case_edges is None:
            return self.maximum_edges
        size = int(case_count) * self.maximum_case_edges
        if size > self.maximum_edges:
            raise ValueError(
                f"{int(case_count)} cases of maximum_case_edges="
                f"{self.maximum_case_edges} exceed maximum_edges={self.maximum_edges}."
            )
        return size

    def dominates(self, other: ParticleImageCapacity, /) -> bool:
        same_layout = (self.maximum_case_edges is None) == (
            other.maximum_case_edges is None
        )
        return (
            same_layout
            and self.maximum_particles_per_cell >= other.maximum_particles_per_cell
            and self.maximum_edges >= other.maximum_edges
            and self.maximum_degree >= other.maximum_degree
            and self.maximum_images >= other.maximum_images
            and (
                self.maximum_case_edges is None
                or other.maximum_case_edges is None
                or self.maximum_case_edges >= other.maximum_case_edges
            )
        )

    def covers(self, evidence: ParticleImageRelationEvidence, /) -> bool:
        """Host decision whether concrete evidence requirements fit this capacity."""
        required_edges = np.asarray(evidence.required_edges)
        return (
            int(np.max(np.asarray(evidence.required_cell_occupancy)))
            <= self.maximum_particles_per_cell
            and int(np.sum(required_edges)) <= self.maximum_edges
            and (
                self.maximum_case_edges is None
                or int(np.max(required_edges)) <= self.maximum_case_edges
            )
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
        if bool(np.any(np.asarray(evidence.tile_overflow))):
            raise ValueError(
                "Cluster tile overflow is charged to the tile search capacity, not "
                "to this ladder; re-prepare the cluster search with a larger tile "
                "capacity."
            )
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
    origin: Array


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
        origin=origin,
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
    """Packed routes and per-case counts.

    ``count_overflow`` marks cases whose exact required route count exceeds
    the int32 count range; their ``required_edges`` and degrees saturate at
    ``2**31 - 1``.
    """

    relation: ParticleImageRelation
    degree: Array
    required_edges: Array
    stored_edges: Array
    shift_overflow: Array
    count_overflow: Array


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


class _ResolvedReceivers(StrictModule):
    slots: Array
    valid: Array
    violation: Array


def _full_receivers(stable_ids: Array, /) -> _ResolvedReceivers:
    cases, particles = stable_ids.shape
    return _ResolvedReceivers(
        jnp.broadcast_to(
            jnp.arange(particles, dtype=jnp.int32)[None, :], (cases, particles)
        ),
        jnp.ones((cases, particles), dtype=jnp.bool_),
        jnp.zeros((cases,), dtype=jnp.bool_),
    )


class ImageReceiverSubset(StrictModule, NonTrainableState):
    """Declared receiver rows of an image search by stable material ID.

    Row ``r`` of case ``c`` requests every route received by the particle whose
    stable ID is ``receiver_ids[c, r]`` where ``receiver_valid[c, r]``.  IDs are
    dynamic numerical leaves; only the row capacity is static.  The mapping to
    particle slots is resolved from the search's current stable IDs, so it is
    independent of storage order.  A valid row whose ID is absent, inactive,
    ambiguous or repeated is a ``receiver_violation`` of its case, never
    silently dropped.
    """

    receiver_ids: Array
    receiver_valid: Array
    row_capacity: int = eqx.field(static=True)

    def __init__(
        self, receiver_ids: ArrayLike, receiver_valid: ArrayLike | None = None, /
    ) -> None:
        ids = jnp.asarray(receiver_ids)
        if (
            ids.ndim != 2
            or ids.shape[1] <= 0
            or not jnp.issubdtype(ids.dtype, jnp.integer)
        ):
            raise ValueError(
                "receiver_ids must be integers with shape (case, rows) and rows > 0."
            )
        valid = (
            jnp.ones(ids.shape, dtype=jnp.bool_)
            if receiver_valid is None
            else jnp.asarray(receiver_valid, dtype=jnp.bool_)
        )
        if valid.shape != ids.shape:
            raise ValueError("receiver_valid must match receiver_ids.")
        self.receiver_ids = ids
        self.receiver_valid = valid
        self.row_capacity = int(ids.shape[1])

    def resolve(self, stable_ids: ArrayLike, active: ArrayLike, /) -> _ResolvedReceivers:
        """Map requested IDs to particle slots of ``stable_ids[case, particle]``."""
        ids = jnp.asarray(stable_ids)
        mask = jnp.asarray(active, dtype=jnp.bool_)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("stable_ids and active must have shape (case, particle).")
        cases, particles = ids.shape
        if self.receiver_ids.shape[0] != cases:
            raise ValueError("Receiver subset case count differs from the search.")
        requested = self.receiver_ids.astype(ids.dtype)
        order = jnp.argsort(ids, axis=1, stable=True).astype(jnp.int32)
        ordered = jnp.take_along_axis(ids, order, axis=1)
        left = jax.vmap(lambda row, query: jnp.searchsorted(row, query, side="left"))(
            ordered, requested
        ).astype(jnp.int32)
        right = jax.vmap(lambda row, query: jnp.searchsorted(row, query, side="right"))(
            ordered, requested
        ).astype(jnp.int32)
        slots = jnp.take_along_axis(order, jnp.clip(left, 0, particles - 1), axis=1)
        resolved = (
            self.receiver_valid
            & ((right - left) == 1)
            & jnp.take_along_axis(mask, slots, axis=1)
        )
        request_order = jax.vmap(
            lambda key, valid: jnp.lexsort((key, (~valid).astype(jnp.int32)))
        )(requested, self.receiver_valid)
        sorted_ids = jnp.take_along_axis(requested, request_order, axis=1)
        sorted_valid = jnp.take_along_axis(self.receiver_valid, request_order, axis=1)
        repeated = (
            sorted_valid[:, 1:]
            & sorted_valid[:, :-1]
            & (sorted_ids[:, 1:] == sorted_ids[:, :-1])
        )
        violation = jnp.any(self.receiver_valid & ~resolved, axis=1) | jnp.any(
            repeated, axis=1
        )
        return _ResolvedReceivers(
            jnp.where(resolved, slots, 0).astype(jnp.int32),
            resolved,
            jax.lax.stop_gradient(violation),
        )


class ImageReceiverRows(StrictModule, NonTrainableState):
    """Compact receiver-row mapping of a receiver-subset search result.

    ``particle_indices[c * R + r]`` is the flat case-major particle of row ``r``
    of case ``c`` (the case's first particle on invalid rows) and
    ``route_rows[e]`` is the compact row receiving route ``e`` (the case's
    first row on padding routes).  Per-row reductions therefore use
    ``case_count * row_capacity`` segments instead of every particle.
    """

    particle_indices: Array
    valid: Array
    route_rows: Array
    row_capacity: int = eqx.field(static=True)
    case_count: int = eqx.field(static=True)


def _receiver_rows(
    resolved: _ResolvedReceivers, relation: ParticleImageRelation, /
) -> ImageReceiverRows:
    cases, rows = resolved.slots.shape
    particles = relation.particle_capacity
    case_index = jnp.arange(cases, dtype=jnp.int32)[:, None]
    row_ids = case_index * rows + jnp.arange(rows, dtype=jnp.int32)[None, :]
    row_of_slot = (
        jnp.zeros((cases, particles), dtype=jnp.int32)
        .at[case_index, jnp.where(resolved.valid, resolved.slots, particles)]
        .set(row_ids, mode="drop")
    )
    route_rows = jnp.where(
        relation.valid,
        row_of_slot.reshape((-1,))[relation.receiver_indices],
        relation.route_cases * rows,
    )
    particle_indices = jnp.where(
        resolved.valid, case_index * particles + resolved.slots, case_index * particles
    )
    return ImageReceiverRows(
        particle_indices.reshape((-1,)),
        resolved.valid.reshape((-1,)),
        route_rows,
        rows,
        cases,
    )


class _Candidates(StrictModule):
    """Fixed-shape candidate routes ``(case, row, slot)`` of one search.

    Row ``(c, i)`` belongs to receiver particle ``receivers[c, i]``; a receiver
    may own several rows (cluster tiles).  Tile fields are zero for searches
    without intermediate tiles.
    """

    candidate: Array
    receivers: Array
    sources: Array
    wrapped_shifts: Array
    occupancy: Array
    cell_overflow: Array
    required_offsets: Array
    stencil_overflow: Array
    required_tile_pairs: Array
    tile_overflow: Array
    cluster_extent_overflow: Array


def _within_radius(
    frame: _LatticeFrame,
    vectors: Array,
    receivers: Array,
    sources: Array,
    wrapped_shifts: Array,
    source_valid: Array,
    radius: float,
    /,
) -> Array:
    """Mask row candidates inside ``radius`` excluding the zero self translation.

    ``receivers[case, row]`` and ``sources[case, row, slot]`` are local slots
    of the wrapped frame; ``wrapped_shifts`` are wrapped-frame translations.
    """
    case_count, rows, slots = sources.shape
    flat_sources = sources.reshape((case_count, -1))
    source_positions = jnp.take_along_axis(
        frame.wrapped, flat_sources[..., None], axis=1
    ).reshape((case_count, rows, slots, -1))
    source_usable = jnp.take_along_axis(frame.usable, flat_sources, axis=1).reshape(
        sources.shape
    )
    receiver_positions = jnp.take_along_axis(frame.wrapped, receivers[..., None], axis=1)
    receiver_usable = jnp.take_along_axis(frame.usable, receivers, axis=1)
    translation = contract(
        "cnmr,crd->cnmd",
        wrapped_shifts.astype(vectors.dtype),
        vectors,
        backend="jax",
    )
    displacement = receiver_positions[:, :, None, :] - source_positions + translation
    squared = jnp.sum(displacement * displacement, axis=-1)
    self_zero = (sources == receivers[:, :, None]) & jnp.all(wrapped_shifts == 0, axis=-1)
    return (
        source_valid
        & source_usable
        & receiver_usable[:, :, None]
        & ~self_zero
        & jax.lax.stop_gradient(squared < radius * radius)
    )


def _no_tiles(case_count: int, /) -> tuple[Array, Array, Array]:
    return (
        jnp.zeros((case_count,), dtype=jnp.int32),
        jnp.zeros((case_count,), dtype=jnp.bool_),
        jnp.zeros((case_count,), dtype=jnp.bool_),
    )


def _assemble_routes(
    route_valid: Array,
    route_case: Array,
    receiver_local: Array,
    source_local: Array,
    shifts: Array,
    stable_ids: Array,
    /,
    *,
    segmented: bool,
    support_id: str,
    relation_schema_id: str,
) -> tuple[ParticleImageRelation, Array]:
    """Canonical relation of selected route slots and its stored count per case.

    Slot ``e`` holds a route of case ``route_case[e]`` from local particle
    ``source_local[e]`` to ``receiver_local[e]`` with raw integer shift
    ``shifts[e]`` where ``route_valid[e]``; other slots are padding whose
    endpoint fields are ignored.  Valid routes are ordered by ``(case,
    receiver id, source id, n)``.  Shared layout (``segmented=False``) puts
    every padding slot after every valid route on the last case; the per-case
    layout keeps each slot's ``route_case`` segment and orders its valid routes
    before its padding.
    """
    case_count, particle_capacity = stable_ids.shape
    rank = shifts.shape[-1]
    receiver_ids = stable_ids[route_case, receiver_local]
    source_ids = stable_ids[route_case, source_local]
    invalid = (~route_valid).astype(jnp.int32)
    shift_keys = tuple(shifts[:, axis] for axis in reversed(range(rank)))
    keys = (
        shift_keys + (source_ids, receiver_ids, invalid, route_case)
        if segmented
        else shift_keys + (source_ids, receiver_ids, route_case, invalid)
    )
    order = jax.lax.stop_gradient(jnp.lexsort(keys))
    route_valid = route_valid[order]
    route_case = (
        route_case[order]
        if segmented
        else jnp.where(route_valid, route_case[order], case_count - 1)
    )
    padding_endpoint = route_case * particle_capacity
    receivers_global = jnp.where(
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
            receivers_global,
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
    return relation, stored_edges


def _pack_routes(
    candidates: _Candidates,
    images: Array,
    stable_ids: Array,
    *,
    edge_capacity: int,
    case_capacity: int | None,
    support_id: str,
    relation_schema_id: str,
) -> _PackedRoutes:
    """Compact candidate routes into canonical receiver-major stable order.

    Raw shifts re-express wrapped-frame translations for the caller's
    coordinate representation: ``n = n_wrapped - w[receiver] + w[source]``.
    A candidate whose ``n`` leaves the symmetric int32 range is reported per
    case in ``shift_overflow`` instead of being stored wrapped around.  Valid
    routes are ordered by ``(case, receiver id, source id, n)``.  In the
    shared layout padding routes follow every valid route; in the per-case
    segment layout (``case_capacity``) segment ``c`` holds case ``c``'s routes
    followed by its own padding.
    """
    candidate = candidates.candidate
    case_count, rows, slots = candidate.shape
    particle_capacity = images.shape[1]
    rank = images.shape[-1]
    sources = candidates.sources
    receivers = candidates.receivers
    source_images = jnp.take_along_axis(
        images, sources.reshape((case_count, rows * slots, 1)), axis=1
    ).reshape((case_count, rows, slots, rank))
    receiver_images = jnp.take_along_axis(images, receivers[..., None], axis=1)
    raw_shifts, overflow = checked_int32_shift(
        candidates.wrapped_shifts.astype(jnp.int32),
        source_images,
        jnp.broadcast_to(receiver_images[:, :, None, :], source_images.shape),
    )
    shift_overflow = jnp.any(candidate & jnp.any(overflow, axis=-1), axis=(1, 2))
    row_degree = jnp.sum(candidate, axis=-1, dtype=jnp.int32)
    case_index = jnp.arange(case_count, dtype=jnp.int32)
    degree = (
        jnp.zeros((case_count, particle_capacity), dtype=jnp.int32)
        .at[case_index[:, None], receivers]
        .add(row_degree)
    )
    required_edges = jnp.sum(row_degree, axis=-1, dtype=jnp.int32)
    per_case = rows * slots
    flat_valid = candidate.reshape((case_count, per_case))
    if case_capacity is None:
        selected = jnp.nonzero(
            flat_valid.reshape((-1,)), size=edge_capacity, fill_value=0
        )[0].astype(jnp.int32)
        total = jnp.sum(flat_valid, dtype=jnp.int32)
        route_valid = jnp.arange(edge_capacity, dtype=jnp.int32) < jnp.minimum(
            total, edge_capacity
        )
        route_case = selected // per_case
    else:
        width = int(case_capacity)
        rank_in_case = jnp.cumsum(flat_valid, axis=1, dtype=jnp.int32) - 1
        target = jnp.where(flat_valid & (rank_in_case < width), rank_in_case, width)
        local = (
            jnp.zeros((case_count, width + 1), dtype=jnp.int32)
            .at[case_index[:, None], target]
            .set(
                jnp.broadcast_to(
                    jnp.arange(per_case, dtype=jnp.int32)[None, :],
                    (case_count, per_case),
                )
            )[:, :width]
        )
        route_valid = (
            jnp.arange(width, dtype=jnp.int32)[None, :]
            < jnp.minimum(required_edges, width)[:, None]
        ).reshape((-1,))
        route_case = jnp.repeat(case_index, width)
        selected = route_case * per_case + local.reshape((-1,))
    selected = jax.lax.stop_gradient(selected)
    route_case = route_case.astype(jnp.int32)
    row = (selected % per_case) // slots
    receiver_local = receivers.reshape((-1,))[route_case * rows + row]
    source_local = sources.reshape((-1,))[selected].astype(jnp.int32)
    shifts = raw_shifts.reshape((-1, rank))[selected]
    relation, stored_edges = _assemble_routes(
        route_valid,
        route_case,
        receiver_local,
        source_local,
        shifts,
        stable_ids,
        segmented=case_capacity is not None,
        support_id=support_id,
        relation_schema_id=relation_schema_id,
    )
    # The admitted candidate tensor has at most ``2**31 - 1`` slots, so every
    # count above is exact.
    return _PackedRoutes(
        relation,
        degree,
        required_edges,
        stored_edges,
        shift_overflow,
        jnp.zeros((case_count,), dtype=jnp.bool_),
    )


class ImageRouteSearchResult(StrictModule, NonTrainableState):
    """Routes plus the build frame needed by lifecycle certificates.

    ``receiver_rows`` is the compact row mapping of a receiver-subset search
    and ``None`` for a complete search.
    """

    relation: ParticleImageRelation
    evidence: ParticleImageRelationEvidence
    wrap_counts: Array
    stencil_extents: Array
    receiver_rows: ImageReceiverRows | None = None


class _RouteSearchOutcome(StrictModule):
    """Canonically packed routes plus the evidence owned by one search route.

    Fields mirror the search-specific members of
    ``ParticleImageRelationEvidence``; searches without cells or tiles report
    zero occupancy and tile pairs and no overflow.
    """

    packed: _PackedRoutes
    occupancy: Array
    cell_overflow: Array
    required_offsets: Array
    stencil_overflow: Array
    required_tile_pairs: Array
    tile_overflow: Array
    cluster_extent_overflow: Array


class AbstractImageRouteSearch(StrictModule, NonTrainableState):
    """Case-batched fixed-capacity image route search over row lattices.

    Inputs are ``positions[case, particle, d]``, ``vectors[case, r, d]``,
    ``origin[case, d]`` and ``periodic[case, r]``.  Cases never exchange routes.
    An optional ``ImageReceiverSubset`` restricts receivers to declared stable
    IDs; sources always range over every active particle.
    """

    radius: eqx.AbstractVar[float]
    case_count: eqx.AbstractVar[int]
    particle_capacity: eqx.AbstractVar[int]
    lattice_rank: eqx.AbstractVar[int]
    capacity: eqx.AbstractVar[ParticleImageCapacity]
    candidate_slot_count: eqx.AbstractVar[int]
    search_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def _search(
        self,
        frame: _LatticeFrame,
        vectors: Array,
        periodic: Array,
        stable_ids: Array,
        receivers: _ResolvedReceivers,
        /,
        *,
        support_id: str,
        relation_schema_id: str,
    ) -> _RouteSearchOutcome:
        """Return canonically packed routes and the search-owned evidence."""
        raise NotImplementedError

    @property
    def route_capacity(self) -> int:
        """Stored route slots: ``maximum_edges`` or ``case_count * maximum_case_edges``."""
        return self.capacity.route_buffer(self.case_count)

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
        receivers: ImageReceiverSubset | None = None,
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
        if receivers is not None and not isinstance(receivers, ImageReceiverSubset):
            raise TypeError("receivers must be an ImageReceiverSubset or None.")
        frame = _lattice_frame(value, active_mask, lattice, shift_origin, periodic_mask)
        resolved = (
            _full_receivers(ids)
            if receivers is None
            else receivers.resolve(ids, active_mask)
        )
        outcome = self._search(
            frame,
            lattice,
            periodic_mask,
            ids,
            resolved,
            support_id=support_id,
            relation_schema_id=relation_schema_id,
        )
        packed = outcome.packed
        required_degree = jnp.max(packed.degree, axis=-1)
        required_images = _required_images(frame.reach, self.radius, periodic_mask)
        # A count beyond int32 cannot be stored by any int32 capacity: it is
        # edge overflow and an unrepresentable requirement, never a ladder retry.
        edge_overflow = (packed.stored_edges < packed.required_edges) | (
            packed.count_overflow
        )
        if self.capacity.maximum_case_edges is None:
            # Per-case counts are exact or flagged, so a wrapped sum only arises
            # when the exact total already exceeds every int32 capacity; the
            # shared buffer then also stores fewer routes than some case needs.
            total_required = jnp.sum(packed.required_edges, dtype=jnp.int32)
            edge_overflow = edge_overflow | (total_required > self.capacity.maximum_edges)
        evidence = ParticleImageRelationEvidence(
            active_particles=jnp.sum(active_mask, axis=-1, dtype=jnp.int32),
            required_cell_occupancy=outcome.occupancy,
            required_edges=packed.required_edges,
            stored_edges=packed.stored_edges,
            required_degree=required_degree,
            required_images=required_images,
            required_offset_extents=outcome.required_offsets,
            cell_overflow=outcome.cell_overflow,
            stencil_overflow=outcome.stencil_overflow,
            image_overflow=required_images > self.capacity.maximum_images,
            edge_overflow=edge_overflow,
            degree_overflow=required_degree > self.capacity.maximum_degree,
            domain_violation=jnp.any(frame.domain_violation, axis=-1),
            # Lattice validity is case-level: an empty or all-inactive case
            # under a singular or non-finite cell is still refused.
            nonfinite=jnp.any(frame.nonfinite, axis=-1)
            | ~jnp.all(jnp.isfinite(lattice), axis=(-2, -1))
            | ~frame.solved,
            representation_overflow=jnp.any(frame.unrepresentable, axis=-1)
            | packed.shift_overflow
            | packed.count_overflow,
            required_tile_pairs=outcome.required_tile_pairs,
            tile_overflow=outcome.tile_overflow,
            cluster_extent_overflow=outcome.cluster_extent_overflow,
            receiver_violation=resolved.violation,
        )
        return ImageRouteSearchResult(
            packed.relation,
            evidence,
            frame.images.reshape((-1, self.lattice_rank)),
            _stencil_extents(frame.reach, self.radius, periodic_mask),
            None if receivers is None else _receiver_rows(resolved, packed.relation),
        )


class _AbstractCandidateImageRouteSearch(AbstractImageRouteSearch):
    """Image search that materializes all candidate rows before packing them."""

    @abc.abstractmethod
    def _candidates(
        self,
        frame: _LatticeFrame,
        vectors: Array,
        periodic: Array,
        stable_ids: Array,
        receivers: _ResolvedReceivers,
        /,
    ) -> _Candidates:
        """Return fixed-shape candidate rows for the resolved receivers."""
        raise NotImplementedError

    def _search(
        self,
        frame: _LatticeFrame,
        vectors: Array,
        periodic: Array,
        stable_ids: Array,
        receivers: _ResolvedReceivers,
        /,
        *,
        support_id: str,
        relation_schema_id: str,
    ) -> _RouteSearchOutcome:
        candidates = self._candidates(frame, vectors, periodic, stable_ids, receivers)
        packed = _pack_routes(
            candidates,
            frame.images,
            stable_ids,
            edge_capacity=self.capacity.maximum_edges,
            case_capacity=self.capacity.maximum_case_edges,
            support_id=support_id,
            relation_schema_id=relation_schema_id,
        )
        return _RouteSearchOutcome(
            packed,
            candidates.occupancy,
            candidates.cell_overflow,
            candidates.required_offsets,
            candidates.stencil_overflow,
            candidates.required_tile_pairs,
            candidates.tile_overflow,
            candidates.cluster_extent_overflow,
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
        # Immutable reference-cell preparation, not runtime particle algebra.
        # Keep this bounded host solve off the selected JAX device.
        reciprocal = np.linalg.solve(matrix @ matrix.T, matrix).T
        if not np.all(np.isfinite(reciprocal)):
            raise ValueError("Reference lattice coordinate solve is non-finite.")
        reach.append(np.sqrt(np.sum(reciprocal * reciprocal, axis=0)))
    return np.stack(reach)


class CellListImageRouteSearch(_AbstractCandidateImageRouteSearch):
    """Fractional-lattice cell list with explicit image-shifted neighbor cells.

    Particles are wrapped on periodic axes and binned on a static fractional
    grid whose cell perpendicular height is at least ``radius`` when the cell
    admits it.  Neighbor cells span ``ceil(radius * grid * reach)`` cells per
    axis, sized from the padded host reference reach (`_padded_reference_reach`), so
    short periodic axes enumerate several translated copies of the same cell
    without duplicates: each unwrapped cell offset maps to one
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
        bound = _padded_reference_reach(_host_reach(np.asarray(reference_vectors)))
        extents = tuple(
            max(1, int(np.ceil(envelope * grid[axis] * np.max(bound[:, axis]))))
            for axis in range(rank)
        )
        # Admit the candidate workspace from the scalar stencil size before any
        # radius-sized offset enumeration is materialized.
        offset_count = prod(2 * extent + 1 for extent in extents)
        slots = cases * particles * offset_count * capacity.maximum_particles_per_cell
        if slots > min(int(maximum_candidate_slots), _INT32_LIMIT):
            raise ValueError(
                f"Image cell list requires {slots} candidate slots, exceeding "
                f"maximum_candidate_slots={int(maximum_candidate_slots)} or the "
                "int32 candidate index space."
            )
        capacity.route_buffer(cases)
        offsets = _complete_image_shifts(extents)
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
        self,
        frame: _LatticeFrame,
        vectors: Array,
        periodic: Array,
        stable_ids: Array,
        receivers: _ResolvedReceivers,
        /,
    ) -> _Candidates:
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
        rows = receivers.slots.shape[1]
        row_coordinates = jnp.take_along_axis(
            coordinates, receivers.slots[..., None], axis=1
        )
        row_usable = (
            jnp.take_along_axis(frame.usable, receivers.slots, axis=1) & receivers.valid
        )
        unwrapped = row_coordinates[:, :, None, :] + self.offsets[None, None, :, :]
        periodic_axis = periodic[:, None, None, :]
        offset_grid = axis_grid[None, None, None, :]
        image = jnp.floor_divide(unwrapped, offset_grid)
        in_range = (unwrapped >= 0) & (unwrapped < offset_grid)
        cell_valid = jnp.all(periodic_axis | in_range, axis=-1) & row_usable[:, :, None]
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
        ).reshape((self.case_count, rows, offset_count * members))
        source_valid = (
            lookup.supported[..., None] & (rank[None, None, :] < counts[..., None])
        ).reshape(sources.shape)
        wrapped_shifts = jnp.broadcast_to(
            -image[:, :, :, None, :],
            (self.case_count, rows, offset_count, members, self.lattice_rank),
        ).reshape(sources.shape + (self.lattice_rank,))
        candidate = _within_radius(
            frame,
            vectors,
            receivers.slots,
            sources,
            wrapped_shifts,
            source_valid,
            self.radius,
        )
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
        return _Candidates(
            candidate,
            receivers.slots,
            sources,
            wrapped_shifts,
            groups.evidence.maximum_group_size.astype(jnp.int32),
            cell_overflow,
            required_offsets,
            stencil_overflow,
            *_no_tiles(self.case_count),
        )


class _ReferenceStencil(NamedTuple):
    """Bounded scalar metadata of a prepared reference stencil (no shifts)."""

    radius: float
    case_count: int
    lattice_rank: int
    particle_capacity: int
    extents: tuple[int, ...]
    image_count: int


def _reference_stencil(
    radius: float,
    reference_vectors: ArrayLike,
    reference_periodic: ArrayLike,
    capacity: ParticleImageCapacity,
    /,
    *,
    particle_capacity: int,
    deformation_margin: float,
    owner: str,
) -> _ReferenceStencil:
    """Prepared complete image stencil metadata of an all-pairs reference search.

    Per periodic axis the stencil spans ``floor(1 + R (1 + margin) reach)``
    images over every reference case, sized from the padded host reference
    reach (`_padded_reference_reach`), and is charged to ``maximum_images``.
    Only scalar metadata is produced: callers enforce their candidate or tile
    budgets before `_complete_image_shifts` materializes the translations.
    """
    value = float(radius)
    margin = float(deformation_margin)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Image search radius must be finite and positive.")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("deformation_margin must be finite and nonnegative.")
    reach = _padded_reference_reach(_host_reach(np.asarray(reference_vectors)))
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
            f"{owner} image stencil requires {image_count} images, exceeding "
            f"maximum_images={capacity.maximum_images}."
        )
    return _ReferenceStencil(value, cases, rank, particles, extents, image_count)


def _stencil_evidence(
    reach: Array, radius: float, periodic: Array, bound: tuple[int, ...], /
) -> tuple[Array, Array]:
    """Required complete extents per case and whether they exceed ``bound``."""
    required = _stencil_extents(reach, radius, periodic)
    overflow = jnp.any(required > jnp.asarray(bound, dtype=jnp.int32)[None, :], axis=-1)
    return required, overflow


class DenseImageRouteSearch(_AbstractCandidateImageRouteSearch):
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
        reference = _reference_stencil(
            radius,
            reference_vectors,
            reference_periodic,
            capacity,
            particle_capacity=particle_capacity,
            deformation_margin=deformation_margin,
            owner="Dense",
        )
        value, cases = reference.radius, reference.case_count
        rank, particles = reference.lattice_rank, reference.particle_capacity
        extents = reference.extents
        routes = cases * particles * particles * reference.image_count
        if routes > min(int(maximum_dense_routes), _INT32_LIMIT):
            raise ValueError(
                f"Dense image reference requires {routes} candidate routes, "
                f"exceeding maximum_dense_routes={int(maximum_dense_routes)} or the "
                "int32 candidate index space."
            )
        capacity.route_buffer(cases)
        self.stencil = jnp.asarray(_complete_image_shifts(extents))
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
        self,
        frame: _LatticeFrame,
        vectors: Array,
        periodic: Array,
        stable_ids: Array,
        receivers: _ResolvedReceivers,
        /,
    ) -> _Candidates:
        images = self.stencil.shape[0]
        rows = receivers.slots.shape[1]
        sources = jnp.broadcast_to(
            jnp.arange(self.particle_capacity, dtype=jnp.int32)[None, None, :, None],
            (self.case_count, rows, self.particle_capacity, images),
        ).reshape((self.case_count, rows, -1))
        shifts = jnp.broadcast_to(
            self.stencil[None, None, None, :, :],
            (self.case_count, rows, self.particle_capacity, images, self.lattice_rank),
        ).reshape(sources.shape + (self.lattice_rank,))
        lawful = (
            jnp.all(periodic[:, None, None, :] | (shifts == 0), axis=-1)
            & receivers.valid[:, :, None]
        )
        candidate = _within_radius(
            frame, vectors, receivers.slots, sources, shifts, lawful, self.radius
        )
        required, stencil_overflow = _stencil_evidence(
            frame.reach, self.radius, periodic, self.stencil_extents_bound
        )
        return _Candidates(
            candidate,
            receivers.slots,
            sources,
            shifts,
            jnp.zeros((self.case_count,), dtype=jnp.int32),
            jnp.zeros((self.case_count,), dtype=jnp.bool_),
            required,
            stencil_overflow,
            *_no_tiles(self.case_count),
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
    resource_counts = {
        "particle_capacity": particles.capacity,
        "candidate_slot_count": search.candidate_slot_count,
        "maximum_particles_per_cell": search.capacity.maximum_particles_per_cell,
        "edge_capacity": search.capacity.maximum_edges,
        "degree_capacity": search.capacity.maximum_degree,
        "image_capacity": search.capacity.maximum_images,
    }
    if search.capacity.maximum_case_edges is not None:
        resource_counts["case_edge_capacity"] = search.capacity.maximum_case_edges
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
        resource_counts=resource_counts,
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


SelectionMeasure = Literal["squared-distance", "distance"]
RadiusBoundary = Literal["open", "closed"]


class ImageNeighborSelectionPolicy(StrictModule, NonTrainableState):
    """Scientifically declared nearest-neighbor selection of a directed image graph.

    The policy belongs to a model definition; it is not a buffer limit and is
    identified separately from any capacity.  Per receiver, a route is
    eligible when it passes the radius test (``d < radius`` for ``"open"``,
    ``d <= radius`` for ``"closed"``) and, when ``coincidence_threshold`` is
    set, ``d**2 > coincidence_threshold``.  Let ``m`` be the declared
    ``measure`` (``d**2`` or ``d``) and ``m_(0) <= m_(1) <= ...`` the
    receiver's eligible values in the canonical order ``(m, source stable ID,
    lexicographic n)``.  A receiver with at most ``k = maximum_neighbors``
    eligible routes keeps all of them.  Otherwise ``strict`` keeps the first
    ``k`` canonical routes, and the degeneracy band keeps every eligible route
    with ``m <= m_(k) + degeneracy_tolerance``, where ``m_(k)`` is the
    0-based ``k``-th value (the ``k + 1``-th nearest).  The band therefore
    keeps at least ``k + 1`` routes; this is the rule of FAIRChem's
    ``get_max_neighbors_mask`` (which applies it to squared distances).  A
    selected graph is directed and generally not reversal closed.
    """

    radius: float = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    degeneracy_tolerance: float = eqx.field(static=True)
    measure: SelectionMeasure = eqx.field(static=True)
    strict: bool = eqx.field(static=True)
    radius_boundary: RadiusBoundary = eqx.field(static=True)
    coincidence_threshold: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        maximum_neighbors: int,
        /,
        *,
        degeneracy_tolerance: float,
        measure: SelectionMeasure,
        strict: bool,
        radius_boundary: RadiusBoundary,
        coincidence_threshold: float | None = None,
    ) -> None:
        value = float(radius)
        neighbors = int(maximum_neighbors)
        tolerance = float(degeneracy_tolerance)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Selection radius must be finite and positive.")
        if neighbors <= 0 or neighbors > _INT32_LIMIT:
            raise ValueError("maximum_neighbors must be a positive int32 count.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("degeneracy_tolerance must be finite and nonnegative.")
        measure = parse(measure, SelectionMeasure, "measure")
        radius_boundary = parse(radius_boundary, RadiusBoundary, "radius_boundary")
        threshold = (
            None if coincidence_threshold is None else float(coincidence_threshold)
        )
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0.0):
            raise ValueError("coincidence_threshold must be finite and nonnegative.")
        self.radius = value
        self.maximum_neighbors = neighbors
        self.degeneracy_tolerance = tolerance
        self.measure = measure
        self.strict = bool(strict)
        self.radius_boundary = radius_boundary
        self.coincidence_threshold = threshold
        self.policy_id = canonical_fingerprint(
            {
                "kind": "image-neighbor-selection-policy",
                "radius": value,
                "maximum_neighbors": neighbors,
                "degeneracy_tolerance": tolerance,
                "measure": measure,
                "strict": bool(strict),
                "radius_boundary": radius_boundary,
                "coincidence_threshold": threshold,
            }
        )


class ImageSelectionCapacity(StrictModule, NonTrainableState):
    """Storage for a selected graph, distinct from the scientific selection.

    ``maximum_routes`` is one shared buffer, or the total bound of the
    per-case segments of width ``maximum_case_routes``.  ``maximum_degree``
    bounds selected routes per receiver for row-layout consumers.
    """

    maximum_routes: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_case_routes: int | None = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_routes: int,
        maximum_degree: int,
        maximum_case_routes: int | None = None,
    ) -> None:
        routes = int(maximum_routes)
        degree = int(maximum_degree)
        if routes < 0 or routes > _INT32_LIMIT or degree <= 0 or degree > _INT32_LIMIT:
            raise ValueError(
                "Selection routes must be nonnegative int32 and degree positive int32."
            )
        case_routes = None if maximum_case_routes is None else int(maximum_case_routes)
        if case_routes is not None and (case_routes <= 0 or case_routes > routes):
            raise ValueError(
                "maximum_case_routes must be positive and no larger than maximum_routes."
            )
        self.maximum_routes = routes
        self.maximum_degree = degree
        self.maximum_case_routes = case_routes
        self.capacity_id = canonical_fingerprint(
            {
                "kind": "image-selection-capacity",
                "maximum_routes": routes,
                "maximum_degree": degree,
                "maximum_case_routes": case_routes,
            }
        )


class ImageSelectionEvidence(StrictModule, NonTrainableState):
    """Per-case evidence of one declared selection.

    ``capped_receivers`` counts receivers whose eligible routes exceeded the
    selection: that is the declared scientific rule, never a failure.
    ``route_overflow`` and ``degree_overflow`` mean the selected graph did not
    fit its storage.  Source failures are carried from the candidate search:
    an incomplete or uncertified candidate set cannot support a selection.
    """

    eligible_routes: Array
    selected_routes: Array
    stored_routes: Array
    candidate_degree: Array
    selected_degree: Array
    capped_receivers: Array
    route_overflow: Array
    degree_overflow: Array
    source_capacity_failure: Array
    source_scientific_failure: Array
    source_uncertified: Array

    @property
    def capacity_failure(self) -> Array:
        return self.route_overflow | self.degree_overflow | self.source_capacity_failure

    @property
    def scientific_failure(self) -> Array:
        return self.source_scientific_failure | self.source_uncertified

    @property
    def successful(self) -> Array:
        return ~(self.capacity_failure | self.scientific_failure)


class ImageNeighborSelection(StrictModule, NonTrainableState):
    """A declared selection of a candidate image relation.

    ``relation`` keeps the selected routes in candidate (receiver-major) order;
    ``parent_routes`` maps each stored route to its candidate route.
    """

    relation: ParticleImageRelation
    parent_routes: Array
    evidence: ImageSelectionEvidence
    policy_id: str = eqx.field(static=True)
    capacity_id: str = eqx.field(static=True)


class PreparedImageNeighborSelection(StrictModule, NonTrainableState):
    """Prepared declared selection over a candidate relation of ``source_radius``.

    The candidate search must cover the selection radius: ``source_radius >=
    radius`` for an open boundary and ``source_radius > radius`` for a closed
    one (the image searches test ``d < source_radius``).  Selection is a pure
    function of the candidate relation and current geometry; masks carry no
    derivative.
    """

    policy: ImageNeighborSelectionPolicy
    capacity: ImageSelectionCapacity
    source_radius: float = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: ImageNeighborSelectionPolicy,
        capacity: ImageSelectionCapacity,
        /,
        *,
        source_radius: float,
        case_count: int,
    ) -> None:
        if not isinstance(policy, ImageNeighborSelectionPolicy):
            raise TypeError("policy must be an ImageNeighborSelectionPolicy.")
        if not isinstance(capacity, ImageSelectionCapacity):
            raise TypeError("capacity must be an ImageSelectionCapacity.")
        radius = float(source_radius)
        cases = int(case_count)
        if np.isnan(radius) or radius <= 0.0 or cases <= 0:
            raise ValueError("source_radius must be positive and case_count positive.")
        covered = (
            radius > policy.radius
            if policy.radius_boundary == "closed"
            else radius >= policy.radius
        )
        if not covered:
            raise ValueError(
                "The candidate search radius does not cover the selection radius "
                "and boundary."
            )
        if (
            capacity.maximum_case_routes is not None
            and cases * capacity.maximum_case_routes > capacity.maximum_routes
        ):
            raise ValueError("Per-case selection segments exceed maximum_routes.")
        self.policy = policy
        self.capacity = capacity
        self.source_radius = radius
        self.case_count = cases
        self.selection_id = canonical_fingerprint(
            {
                "kind": "prepared-image-neighbor-selection",
                "policy": policy.policy_id,
                "capacity": capacity.capacity_id,
                "source_radius": None if np.isposinf(radius) else radius,
                "case_count": cases,
            }
        )

    def select(
        self,
        relation: ParticleImageRelation,
        positions: ArrayLike,
        cell_vectors: ArrayLike,
        /,
        *,
        evidence: ParticleImageRelationEvidence | None = None,
        certified: ArrayLike | None = None,
    ) -> ImageNeighborSelection:
        """Select routes of ``relation`` under current ``positions``/``cell_vectors``.

        ``evidence`` (the candidate search evidence) and ``certified`` (a
        per-case lifecycle certificate of the cached candidate set) are
        carried into the selection status.
        """
        if not isinstance(relation, ParticleImageRelation):
            raise TypeError("relation must be a ParticleImageRelation.")
        if relation.case_count != self.case_count:
            raise ValueError("Relation case count differs from the prepared selection.")
        policy = self.policy
        cases = relation.case_count
        size = cases * relation.particle_capacity
        displacement = relation.displacement(positions, cell_vectors)
        squared = jax.lax.stop_gradient(jnp.sum(displacement * displacement, axis=-1))
        bound = policy.radius * policy.radius
        eligible = relation.valid & (
            squared <= bound if policy.radius_boundary == "closed" else squared < bound
        )
        if policy.coincidence_threshold is not None:
            eligible = eligible & (squared > policy.coincidence_threshold)
        measure = jnp.where(
            eligible,
            squared if policy.measure == "squared-distance" else jnp.sqrt(squared),
            0.0,
        )
        routes = relation.capacity
        shifts = jnp.where(eligible[:, None], relation.image_shifts, 0)
        keys = tuple(shifts[:, axis] for axis in reversed(range(relation.lattice_rank)))
        keys = keys + (
            jnp.where(eligible, relation.source_particle_ids, 0),
            measure,
            jnp.where(eligible, relation.receiver_indices, 0),
            (~eligible).astype(jnp.int32),
        )
        order = jax.lax.stop_gradient(jnp.lexsort(keys)).astype(jnp.int32)
        sorted_receivers = relation.receiver_indices[order]
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
            jnp.zeros((size,), dtype=jnp.int32)
            .at[jnp.where(eligible, relation.receiver_indices, 0)]
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
            jnp.zeros((size,), dtype=jnp.int32)
            .at[jnp.where(selected, relation.receiver_indices, 0)]
            .add(selected.astype(jnp.int32))
        )
        compaction = compact_image_routes(
            selected,
            relation.route_cases,
            case_count=cases,
            capacity=(
                self.capacity.maximum_routes
                if self.capacity.maximum_case_routes is None
                else None
            ),
            case_capacity=self.capacity.maximum_case_routes,
        )
        chosen = relation.take_routes(
            compaction,
            relation_schema_id=canonical_fingerprint(
                {
                    "kind": "selected-image-relation",
                    "parent": relation.relation_schema_id,
                    "selection": self.selection_id,
                }
            ),
        )
        eligible_routes = (
            jnp.zeros((cases,), dtype=jnp.int32)
            .at[jnp.where(eligible, relation.route_cases, 0)]
            .add(eligible.astype(jnp.int32))
        )
        eligible_rows = eligible_count.reshape((cases, -1))
        selected_rows = selected_count.reshape((cases, -1))
        selected_degree = jnp.max(selected_rows, axis=-1)
        false = jnp.zeros((cases,), dtype=jnp.bool_)
        uncertified = (
            false
            if certified is None
            else ~jnp.broadcast_to(jnp.asarray(certified, dtype=jnp.bool_), (cases,))
        )
        return ImageNeighborSelection(
            chosen,
            jnp.where(chosen.valid, compaction.indices, 0),
            ImageSelectionEvidence(
                eligible_routes,
                compaction.required,
                compaction.stored,
                jnp.max(eligible_rows, axis=-1),
                selected_degree,
                jnp.sum(eligible_rows > selected_rows, axis=-1, dtype=jnp.int32),
                compaction.overflow,
                selected_degree > self.capacity.maximum_degree,
                false if evidence is None else evidence.capacity_failure,
                false if evidence is None else evidence.scientific_failure,
                uncertified,
            ),
            policy.policy_id,
            self.capacity.capacity_id,
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
    "ImageNeighborSelection",
    "ImageNeighborSelectionPolicy",
    "ImageReceiverRows",
    "ImageReceiverSubset",
    "ImageRouteSearchResult",
    "ImageSelectionCapacity",
    "ImageSelectionEvidence",
    "image_certificate",
    "ParticleImageCapacity",
    "ParticleImageCapacityLadder",
    "ParticleImageNeighborhoodState",
    "PreparedCellListParticleImageNeighborhood",
    "PreparedDenseParticleImageNeighborhood",
    "PreparedImageNeighborSelection",
]
