#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._periodic_cell import _complete_image_shifts
from ._image_neighborhood import (
    _assemble_routes,
    _INT32_LIMIT,
    _LatticeFrame,
    _no_tiles,
    _PackedRoutes,
    _reference_stencil,
    _ResolvedReceivers,
    _RouteSearchOutcome,
    _stencil_evidence,
    _within_radius,
    AbstractImageRouteSearch,
    ParticleImageCapacity,
)
from ._image_relation import checked_int32_shift


class ImageTilePolicy(StrictModule, NonTrainableState):
    """Explicit tile widths and workspace bound of a tiled image reference.

    One tile step tests ``receiver_width`` receiver rows against
    ``source_width`` source particles times the complete prepared image
    stencil for every case, so the materialized candidate workspace is
    ``case * receiver_width * source_width * images`` and does not grow with
    the all-pairs work ``case * rows * particles * images``.  Width one on both
    axes is the scalar reference: one receiver/source pair per step.  The
    workspace is admitted against ``maximum_tile_candidates`` at preparation.
    """

    receiver_width: int = eqx.field(static=True)
    source_width: int = eqx.field(static=True)
    maximum_tile_candidates: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, *, receiver_width: int, source_width: int, maximum_tile_candidates: int
    ) -> None:
        values = {
            "receiver_width": int(receiver_width),
            "source_width": int(source_width),
            "maximum_tile_candidates": int(maximum_tile_candidates),
        }
        for name, value in values.items():
            if value <= 0 or value > _INT32_LIMIT:
                raise ValueError(f"{name} must be a positive int32 value.")
        self.receiver_width = values["receiver_width"]
        self.source_width = values["source_width"]
        self.maximum_tile_candidates = values["maximum_tile_candidates"]
        self.policy_id = canonical_fingerprint({"kind": "image-tile-policy", **values})

    @property
    def scalar(self) -> bool:
        """Whether every tile step tests exactly one receiver/source pair."""
        return self.receiver_width == 1 and self.source_width == 1


class ImageTileSchedule(StrictModule, NonTrainableState):
    """Logical traversal of one tiled search call for a static row count.

    ``tile_steps`` counts both sweeps (count, then emit) and
    ``traversed_candidates`` the padded pair-image tests over both sweeps;
    ``workspace_candidates`` is the materialized candidate tile of one step.
    """

    row_count: int = eqx.field(static=True)
    receiver_width: int = eqx.field(static=True)
    source_width: int = eqx.field(static=True)
    tile_steps: int = eqx.field(static=True)
    workspace_candidates: int = eqx.field(static=True)
    traversed_candidates: int = eqx.field(static=True)


def _saturating_add(left: Array, right: Array, /) -> Array:
    """``min(left + right, 2**31 - 1)`` for int32 ``0 <= left``, ``right >= -1``.

    The headroom ``2**31 - 1 - left`` never overflows, so no intermediate
    wraps.
    """
    return left + jnp.minimum(right, _INT32_LIMIT - left)


def _saturating_scan(
    values: Array, starts: Array, /, *, axis: int
) -> tuple[Array, Array]:
    """Inclusive sums of nonnegative int32 ``values`` restarting where ``starts``.

    Returns ``min(S, 2**31 - 1)`` for the exact running segment sum ``S`` and
    whether ``S`` exceeds ``2**31 - 1``.  Both are functions of the exact sum,
    so the combination is associative and no intermediate wraps.
    """

    def combine(
        left: tuple[Array, Array, Array], right: tuple[Array, Array, Array], /
    ) -> tuple[Array, Array, Array]:
        left_start, left_value, left_over = left
        right_start, right_value, right_over = right
        headroom = _INT32_LIMIT - left_value
        value = jnp.where(
            right_start, right_value, left_value + jnp.minimum(right_value, headroom)
        )
        over = jnp.where(
            right_start, right_over, left_over | right_over | (right_value > headroom)
        )
        return left_start | right_start, value, over

    _, value, over = jax.lax.associative_scan(
        combine, (starts, values, jnp.zeros(values.shape, dtype=jnp.bool_)), axis=axis
    )
    return value, over


def _shift_exclusive(inclusive: Array, /, *, axis: int) -> Array:
    """Exclusive prefix from an inclusive one: zero followed by all but the last."""
    size = inclusive.shape[axis]
    leading = jnp.zeros_like(jax.lax.slice_in_dim(inclusive, 0, 1, axis=axis))
    return jnp.concatenate(
        (leading, jax.lax.slice_in_dim(inclusive, 0, size - 1, axis=axis)), axis=axis
    )


def _row_prefix(row_degree: Array, /) -> tuple[Array, Array, Array]:
    """Saturated first route of each row, case totals and exact total overflow.

    ``row_degree[case, row]`` holds exact per-row counts.  Saturated values are
    ``min(exact, 2**31 - 1)``; the third result marks cases whose exact total
    exceeds the int32 count range.
    """
    inclusive, over = _saturating_scan(
        row_degree, jnp.zeros(row_degree.shape, dtype=jnp.bool_), axis=1
    )
    return _shift_exclusive(inclusive, axis=1), inclusive[:, -1], over[:, -1]


def _receiver_degree(row_slots: Array, row_degree: Array, particles: int, /) -> Array:
    """Saturated per-particle degree summed over every row of the same slot.

    Several rows reach one slot only for repeated (violating) receiver
    requests; the sums saturate instead of wrapping.  A particle degree is a
    partial sum of its case total, so it saturates only when the case total
    overflows, which ``_row_prefix`` reports.
    """
    cases = row_slots.shape[0]
    order = jnp.argsort(row_slots, axis=1, stable=True)
    slots = jnp.take_along_axis(row_slots, order, axis=1)
    counts = jnp.take_along_axis(row_degree, order, axis=1)
    starts = jnp.concatenate(
        (
            jnp.ones((cases, 1), dtype=jnp.bool_),
            slots[:, 1:] != slots[:, :-1],
        ),
        axis=1,
    )
    sums, _ = _saturating_scan(counts, starts, axis=1)
    case_index = jnp.arange(cases, dtype=jnp.int32)[:, None]
    # Running sums are nondecreasing within a slot, so the maximum is its total.
    return jnp.zeros((cases, particles), dtype=jnp.int32).at[case_index, slots].max(sums)


def _route_targets(
    candidate: Array,
    prior: Array,
    base: Array,
    /,
    *,
    buffer: int,
    segment: int | None,
) -> tuple[Array, Array]:
    """Route-buffer slot of every tile candidate and the advanced row counters.

    ``prior[case, row]`` counts the row's routes emitted by earlier source
    tiles (exact, at most ``particles * images``) and ``base[case, row]`` is
    the saturated slot of the row's first route.  A candidate's exact slot is
    ``base + prior + rank``; its saturated value is below ``buffer`` (or
    ``segment``) exactly when the exact one is, so saturation never stores a
    route in a wrong slot.  Dropped candidates target ``buffer``.
    """
    rank = prior[:, :, None] + jnp.cumsum(candidate, axis=-1, dtype=jnp.int32) - 1
    position = _saturating_add(jnp.broadcast_to(base[:, :, None], rank.shape), rank)
    if segment is None:
        target = jnp.where(candidate & (position < buffer), position, buffer)
    else:
        keep = candidate & (position < segment)
        case_index = jnp.arange(candidate.shape[0], dtype=jnp.int32)[:, None, None]
        # ``case * segment + position < case_count * segment <= buffer``.
        target = jnp.where(
            keep, case_index * segment + jnp.where(keep, position, 0), buffer
        )
    return target.reshape((-1,)), prior + jnp.sum(candidate, axis=-1, dtype=jnp.int32)


class TiledImageRouteSearch(AbstractImageRouteSearch):
    """Prepared scalar/tiled all-pairs image reference with bounded tiles.

    The search tests the same candidates as ``DenseImageRouteSearch`` (every
    receiver row, every source particle, the complete prepared image stencil)
    but traverses them with a ``lax.fori_loop`` over receiver-row tiles (outer)
    and source tiles (inner) declared by an ``ImageTilePolicy``.  The first
    sweep counts each row's routes and shift overflow; the second recomputes
    each tile and scatters its routes directly to their slots.  A route's slot
    is its rank in the dense ``(case, row, source, image)`` candidate order, so
    stored routes, overflow subsets and every evidence count equal the dense
    reference, and the canonical relation assembly orders them by ``(case,
    receiver id, source id, n)``.

    Materialized intermediates are one candidate tile (``candidate_slot_count``),
    per-row counters and the fixed route buffer; none scales with the all-pairs
    work.  Required counts are accumulated for every route independently of
    the stored capacity.  Each row count is exact (``particles * images`` is
    admitted into int32 at preparation); row prefixes, case totals and degrees
    saturate at ``2**31 - 1`` with an exact overflow bit, and saturated slots
    are below the route buffer exactly when the exact slots are.  A case whose
    exact required count exceeds the int32 count range is charged as both edge
    overflow and ``representation_overflow``: storage cannot hold it and no
    declared int32 capacity can, so it fails and is never a ladder retry.
    Edge overflow is otherwise charged exactly and never turns a truncated
    relation into a success.  Width one is the scalar reference.
    Positions keep their floating dtype; triclinic and partially periodic row
    lattices are supported.  The topology is piecewise constant: the search is
    evaluated on stopped-gradient inputs, and derivatives flow through the
    returned relation's displacements, not through route selection.
    """

    stencil: Array
    tiles: ImageTilePolicy
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
        tiles: ImageTilePolicy,
        deformation_margin: float = 0.0,
    ) -> None:
        if not isinstance(capacity, ParticleImageCapacity):
            raise TypeError("capacity must be a ParticleImageCapacity.")
        if not isinstance(tiles, ImageTilePolicy):
            raise TypeError("tiles must be an ImageTilePolicy.")
        reference = _reference_stencil(
            radius,
            reference_vectors,
            reference_periodic,
            capacity,
            particle_capacity=particle_capacity,
            deformation_margin=deformation_margin,
            owner="Tiled",
        )
        cases, particles = reference.case_count, reference.particle_capacity
        images = reference.image_count
        if particles * images > _INT32_LIMIT:
            raise ValueError(
                f"Tiled image reference rows hold {particles * images} candidates, "
                "exceeding the int32 per-row count space."
            )
        source_width = min(tiles.source_width, particles)
        workspace = cases * tiles.receiver_width * source_width * images
        if workspace > min(tiles.maximum_tile_candidates, _INT32_LIMIT):
            raise ValueError(
                f"Tiled image reference requires {workspace} candidate slots per "
                f"tile, exceeding maximum_tile_candidates="
                f"{tiles.maximum_tile_candidates} or the int32 candidate index space."
            )
        capacity.route_buffer(cases)
        self.tiles = tiles
        self.radius = reference.radius
        self.stencil_extents_bound = reference.extents
        self.case_count = cases
        self.particle_capacity = particles
        self.lattice_rank = reference.lattice_rank
        self.capacity = capacity
        self.candidate_slot_count = workspace
        self.search_id = canonical_fingerprint(
            {
                "kind": "tiled-image-route-search",
                "radius": reference.radius,
                "extents": list(reference.extents),
                "case_count": cases,
                "particle_capacity": particles,
                "capacity": capacity.capacity_id,
                "tiles": tiles.policy_id,
            }
        )
        # Every tile and index budget is enforced before the shifts exist.
        self.tile_schedule()
        self.stencil = jnp.asarray(_complete_image_shifts(reference.extents))

    def tile_schedule(self, row_count: int | None = None, /) -> ImageTileSchedule:
        """Logical schedule for ``row_count`` receiver rows (all particles by default).

        Refuses schedules whose padded rows, padded sources or per-sweep tile
        steps leave the int32 index range used by the traversal loop.
        """
        rows = self.particle_capacity if row_count is None else int(row_count)
        if rows <= 0:
            raise ValueError("row_count must be positive.")
        row_width = min(self.tiles.receiver_width, rows)
        source_width = min(self.tiles.source_width, self.particle_capacity)
        row_tiles = -(-rows // row_width)
        source_tiles = -(-self.particle_capacity // source_width)
        if (
            max(
                row_tiles * row_width,
                source_tiles * source_width,
                row_tiles * source_tiles,
            )
            > _INT32_LIMIT
        ):
            raise ValueError(
                "Tiled image reference padded rows, sources or tile steps exceed "
                "the int32 traversal index range."
            )
        images = prod(2 * extent + 1 for extent in self.stencil_extents_bound)
        workspace = self.case_count * row_width * source_width * images
        return ImageTileSchedule(
            row_count=rows,
            receiver_width=row_width,
            source_width=source_width,
            tile_steps=2 * row_tiles * source_tiles,
            workspace_candidates=workspace,
            traversed_candidates=2 * row_tiles * source_tiles * workspace,
        )

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
        # Route selection is piecewise constant in the coordinates.
        frame = jax.lax.stop_gradient(frame)
        vectors = jax.lax.stop_gradient(vectors)
        cases, particles, rank = (
            self.case_count,
            self.particle_capacity,
            self.lattice_rank,
        )
        schedule = self.tile_schedule(receivers.slots.shape[1])
        row_width, source_width = schedule.receiver_width, schedule.source_width
        row_tiles = -(-schedule.row_count // row_width)
        source_tiles = -(-particles // source_width)
        padded_rows = row_tiles * row_width
        padding = ((0, 0), (0, padded_rows - schedule.row_count))
        row_slots = jnp.pad(receivers.slots, padding)
        row_valid = jnp.pad(receivers.valid, padding)
        source_index = jnp.arange(source_tiles * source_width, dtype=jnp.int32)
        source_real = source_index < particles
        source_slots = jnp.minimum(source_index, particles - 1)
        stencil = self.stencil
        images = stencil.shape[0]
        lawful_image = jnp.all(periodic[:, None, :] | (stencil[None, :, :] == 0), axis=-1)
        slots_per_row = source_width * images
        tile_shifts = jnp.broadcast_to(
            stencil[None, None, None, :, :],
            (cases, row_width, source_width, images, rank),
        ).reshape((cases, row_width, slots_per_row, rank))

        def tile(step: Array, /) -> tuple[Array, Array, Array, Array, Array, Array]:
            first_row = (step // source_tiles) * row_width
            first_source = (step % source_tiles) * source_width
            slots = jax.lax.dynamic_slice_in_dim(row_slots, first_row, row_width, axis=1)
            valid = jax.lax.dynamic_slice_in_dim(row_valid, first_row, row_width, axis=1)
            members = jax.lax.dynamic_slice_in_dim(
                source_slots, first_source, source_width
            )
            real = jax.lax.dynamic_slice_in_dim(source_real, first_source, source_width)
            sources = jnp.broadcast_to(
                members[None, None, :, None], (cases, row_width, source_width, images)
            ).reshape((cases, row_width, slots_per_row))
            lawful = (
                valid[:, :, None, None]
                & real[None, None, :, None]
                & lawful_image[:, None, None, :]
            ).reshape((cases, row_width, slots_per_row))
            candidate = _within_radius(
                frame, vectors, slots, sources, tile_shifts, lawful, self.radius
            )
            source_images = jnp.broadcast_to(
                frame.images[:, members][:, None, :, None, :],
                (cases, row_width, source_width, images, rank),
            ).reshape(tile_shifts.shape)
            receiver_images = jnp.take_along_axis(frame.images, slots[..., None], axis=1)
            raw_shifts, overflow = checked_int32_shift(
                tile_shifts,
                source_images,
                jnp.broadcast_to(receiver_images[:, :, None, :], tile_shifts.shape),
            )
            return first_row, slots, sources, candidate, raw_shifts, overflow

        steps = row_tiles * source_tiles

        def count(step: Array, carry: tuple[Array, Array], /) -> tuple[Array, Array]:
            row_degree, shift_overflow = carry
            first_row, _, _, candidate, _, overflow = tile(step)
            prior = jax.lax.dynamic_slice_in_dim(row_degree, first_row, row_width, axis=1)
            row_degree = jax.lax.dynamic_update_slice_in_dim(
                row_degree,
                prior + jnp.sum(candidate, axis=-1, dtype=jnp.int32),
                first_row,
                axis=1,
            )
            shift_overflow = shift_overflow | jnp.any(
                candidate & jnp.any(overflow, axis=-1), axis=(1, 2)
            )
            return row_degree, shift_overflow

        row_degree, shift_overflow = jax.lax.fori_loop(
            0,
            steps,
            count,
            (
                jnp.zeros((cases, padded_rows), dtype=jnp.int32),
                jnp.zeros((cases,), dtype=jnp.bool_),
            ),
        )
        row_first, required_edges, count_overflow = _row_prefix(row_degree)
        segment = self.capacity.maximum_case_edges
        buffer = self.route_capacity
        case_index = jnp.arange(cases, dtype=jnp.int32)
        if segment is None:
            case_end, _ = _saturating_scan(
                required_edges, jnp.zeros((cases,), dtype=jnp.bool_), axis=0
            )
            # Later cases' first slots saturate past the buffer exactly when
            # their exact first slots do; no case-level flag is needed.
            row_base = _saturating_add(
                _shift_exclusive(case_end, axis=0)[:, None], row_first
            )
            route_case = jnp.zeros((buffer,), dtype=jnp.int32)
        else:
            row_base = row_first
            route_case = jnp.repeat(case_index, segment)
        tile_cases = jnp.broadcast_to(
            case_index[:, None, None], (cases, row_width, slots_per_row)
        ).reshape((-1,))

        def emit(
            step: Array,
            carry: tuple[Array, Array, Array, Array, Array, Array],
            /,
        ) -> tuple[Array, Array, Array, Array, Array, Array]:
            written, route_valid, route_case, receiver_local, source_local, shifts = carry
            first_row, slots, sources, candidate, raw_shifts, _ = tile(step)
            prior = jax.lax.dynamic_slice_in_dim(written, first_row, row_width, axis=1)
            base = jax.lax.dynamic_slice_in_dim(row_base, first_row, row_width, axis=1)
            target, advanced = _route_targets(
                candidate, prior, base, buffer=buffer, segment=segment
            )
            receiver_rows = jnp.broadcast_to(slots[:, :, None], candidate.shape)
            route_valid = route_valid.at[target].set(True, mode="drop")
            route_case = route_case.at[target].set(tile_cases, mode="drop")
            receiver_local = receiver_local.at[target].set(
                receiver_rows.reshape((-1,)), mode="drop"
            )
            source_local = source_local.at[target].set(
                sources.reshape((-1,)), mode="drop"
            )
            shifts = shifts.at[target].set(raw_shifts.reshape((-1, rank)), mode="drop")
            written = jax.lax.dynamic_update_slice_in_dim(
                written, advanced, first_row, axis=1
            )
            return written, route_valid, route_case, receiver_local, source_local, shifts

        _, route_valid, route_case, receiver_local, source_local, shifts = (
            jax.lax.fori_loop(
                0,
                steps,
                emit,
                (
                    jnp.zeros((cases, padded_rows), dtype=jnp.int32),
                    jnp.zeros((buffer,), dtype=jnp.bool_),
                    route_case,
                    jnp.zeros((buffer,), dtype=jnp.int32),
                    jnp.zeros((buffer,), dtype=jnp.int32),
                    jnp.zeros((buffer, rank), dtype=jnp.int32),
                ),
            )
        )
        relation, stored_edges = _assemble_routes(
            route_valid,
            route_case,
            receiver_local,
            source_local,
            shifts,
            stable_ids,
            segmented=segment is not None,
            support_id=support_id,
            relation_schema_id=relation_schema_id,
        )
        degree = _receiver_degree(row_slots, row_degree, particles)
        required, stencil_overflow = _stencil_evidence(
            frame.reach, self.radius, periodic, self.stencil_extents_bound
        )
        return _RouteSearchOutcome(
            _PackedRoutes(
                relation,
                degree,
                required_edges,
                stored_edges,
                shift_overflow,
                count_overflow,
            ),
            jnp.zeros((cases,), dtype=jnp.int32),
            jnp.zeros((cases,), dtype=jnp.bool_),
            required,
            stencil_overflow,
            *_no_tiles(cases),
        )


__all__ = ["ImageTilePolicy", "ImageTileSchedule", "TiledImageRouteSearch"]
