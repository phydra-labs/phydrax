#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._triangle_ray import (
    intersect_triangle_rays,
    prepare_triangle_ray_query,
    PreparedTriangleRayQuery,
    TriangleRayIntersectionStatus,
    TriangleRayQueryPlan,
)
from ._interface import evaluate_refractive_interface, OpticalRayState


class NonSequentialSurfaceKind(IntEnum):
    """Concrete interaction attached to an oriented triangle."""

    DIELECTRIC = 0
    MIRROR = 1
    ABSORBER = 2
    DETECTOR = 3


class NonSequentialBranchMode(IntEnum):
    """Requested dielectric candidates for one physical interface."""

    BOTH = 0
    REFLECTION_ONLY = 1
    TRANSMISSION_ONLY = 2


class NonSequentialOpticsStatus(IntEnum):
    """Terminal status of one fixed-capacity non-sequential ray tree."""

    SUCCESS = 0
    INVALID_INPUT = 1
    AMBIGUOUS_INTERSECTION = 2
    TRAVERSAL_CAPACITY_EXHAUSTED = 3
    MEDIUM_MISMATCH = 4
    INTERFACE_FAILURE = 5
    BRANCH_CAPACITY_EXHAUSTED = 6
    INTERACTION_CAPACITY_EXHAUSTED = 7
    NONFINITE_RESULT = 8


class NonSequentialSurfaceTable(StrictModule, NonTrainableState):
    """Immutable oriented triangle interfaces and their concrete optical roles.

    Triangle normals point from ``negative_medium_indices`` to
    ``positive_medium_indices``. Triangles sharing a physical interface must
    share a ``surface_id``; otherwise a nearest-distance edge tie is reported as
    ambiguous. Medium zero has no special hidden meaning.
    """

    vertices: Array
    triangles: Array
    surface_ids: Array
    negative_medium_indices: Array
    positive_medium_indices: Array
    refractive_indices: Array
    surface_kinds: Array
    branch_modes: Array
    detector_indices: Array
    detector_acceptance_cosines: Array
    surface_count: int = eqx.field(static=True)
    detector_count: int = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        negative_medium_indices: ArrayLike,
        positive_medium_indices: ArrayLike,
        refractive_indices: ArrayLike,
        /,
        *,
        surface_ids: ArrayLike | None = None,
        surface_kinds: ArrayLike | None = None,
        branch_modes: ArrayLike | None = None,
        detector_indices: ArrayLike | None = None,
        detector_acceptance_cosines: ArrayLike | None = None,
    ):
        vertices_host = np.asarray(vertices)
        triangles_host = np.asarray(triangles)
        if vertices_host.ndim != 2 or vertices_host.shape[1:] != (3,):
            raise ValueError("vertices must have shape (n_vertices, 3).")
        if triangles_host.ndim != 2 or triangles_host.shape[1:] != (3,):
            raise ValueError("triangles must have shape (n_triangles, 3).")
        if triangles_host.shape[0] < 1:
            raise ValueError("At least one oriented triangle surface is required.")
        triangle_count = triangles_host.shape[0]
        negative = np.asarray(negative_medium_indices)
        positive = np.asarray(positive_medium_indices)
        media = np.asarray(refractive_indices)
        if negative.shape != (triangle_count,) or positive.shape != (triangle_count,):
            raise ValueError("Medium-index arrays must have shape (n_triangles,).")
        if not np.issubdtype(negative.dtype, np.integer) or not np.issubdtype(
            positive.dtype, np.integer
        ):
            raise TypeError("Medium indices must be integers.")
        if media.ndim != 1 or media.size < 1 or np.iscomplexobj(media):
            raise ValueError(
                "refractive_indices must be a non-empty real rank-one array."
            )
        if not np.all(np.isfinite(media)) or np.any(media <= 0.0):
            raise ValueError("refractive_indices must be finite and positive.")
        if (
            np.any(negative < 0)
            or np.any(positive < 0)
            or np.any(np.maximum(negative, positive) >= media.size)
        ):
            raise ValueError("A surface contains an out-of-range medium index.")

        if surface_ids is None:
            ids = np.arange(triangle_count, dtype=np.int32)
        else:
            ids = np.asarray(surface_ids)
        if ids.shape != (triangle_count,) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("surface_ids must be an integer (n_triangles,) array.")
        if np.any(ids < 0):
            raise ValueError("surface_ids must be non-negative.")
        surface_count = int(np.max(ids)) + 1
        for surface_id in range(surface_count):
            members = ids == surface_id
            if not np.any(members):
                continue
            if np.any(negative[members] != negative[members][0]) or np.any(
                positive[members] != positive[members][0]
            ):
                raise ValueError(
                    "Triangles sharing a surface_id must share medium sides."
                )

        kinds = (
            np.full(
                (triangle_count,),
                int(NonSequentialSurfaceKind.DIELECTRIC),
                dtype=np.int32,
            )
            if surface_kinds is None
            else np.asarray(surface_kinds)
        )
        modes = (
            np.full((triangle_count,), int(NonSequentialBranchMode.BOTH), dtype=np.int32)
            if branch_modes is None
            else np.asarray(branch_modes)
        )
        detectors = (
            np.full((triangle_count,), -1, dtype=np.int32)
            if detector_indices is None
            else np.asarray(detector_indices)
        )
        acceptances = (
            np.zeros((triangle_count,), dtype=float)
            if detector_acceptance_cosines is None
            else np.asarray(detector_acceptance_cosines)
        )
        arrays = (kinds, modes, detectors, acceptances)
        if any(value.shape != (triangle_count,) for value in arrays):
            raise ValueError(
                "Per-triangle optical arrays must have shape (n_triangles,)."
            )
        if not np.issubdtype(kinds.dtype, np.integer) or np.any(
            (kinds < int(NonSequentialSurfaceKind.DIELECTRIC))
            | (kinds > int(NonSequentialSurfaceKind.DETECTOR))
        ):
            raise ValueError("surface_kinds contains an unsupported tag.")
        if not np.issubdtype(modes.dtype, np.integer) or np.any(
            (modes < int(NonSequentialBranchMode.BOTH))
            | (modes > int(NonSequentialBranchMode.TRANSMISSION_ONLY))
        ):
            raise ValueError("branch_modes contains an unsupported tag.")
        if not np.issubdtype(detectors.dtype, np.integer) or np.any(detectors < -1):
            raise ValueError(
                "detector_indices must be integers greater than or equal to -1."
            )
        detector_surfaces = kinds == int(NonSequentialSurfaceKind.DETECTOR)
        if np.any(detector_surfaces & (detectors < 0)) or np.any(
            ~detector_surfaces & (detectors >= 0)
        ):
            raise ValueError("Exactly detector surfaces must have a detector index.")
        if np.any(~np.isfinite(acceptances)) or np.any(
            (acceptances < 0.0) | (acceptances > 1.0)
        ):
            raise ValueError("detector_acceptance_cosines must lie in [0, 1].")
        for surface_id in range(surface_count):
            members = ids == surface_id
            representative = np.flatnonzero(members)[0]
            if (
                np.any(kinds[members] != kinds[representative])
                or np.any(modes[members] != modes[representative])
                or np.any(detectors[members] != detectors[representative])
                or np.any(acceptances[members] != acceptances[representative])
            ):
                raise ValueError(
                    "Triangles sharing a surface_id must share every optical role."
                )
        detector_count = 0 if np.all(detectors < 0) else int(np.max(detectors)) + 1

        dtype = jnp.result_type(vertices_host, media, 0.0)
        self.vertices = jnp.asarray(vertices_host, dtype=dtype)
        self.triangles = jnp.asarray(triangles_host, dtype=jnp.int32)
        self.surface_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.negative_medium_indices = jnp.asarray(negative, dtype=jnp.int32)
        self.positive_medium_indices = jnp.asarray(positive, dtype=jnp.int32)
        self.refractive_indices = jnp.asarray(media, dtype=dtype)
        self.surface_kinds = jnp.asarray(kinds, dtype=jnp.int32)
        self.branch_modes = jnp.asarray(modes, dtype=jnp.int32)
        self.detector_indices = jnp.asarray(detectors, dtype=jnp.int32)
        self.detector_acceptance_cosines = jnp.asarray(acceptances, dtype=dtype)
        self.surface_count = surface_count
        self.detector_count = detector_count


class NonSequentialOpticsPlan(StrictModule, NonTrainableState):
    """Host plan for a bounded exact non-sequential scalar-power trace."""

    surfaces: NonSequentialSurfaceTable
    maximum_interactions: int = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    triangle_leaf_size: int = eqx.field(static=True)
    record_history: bool = eqx.field(static=True)
    ray_tolerance: float = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        surfaces: NonSequentialSurfaceTable,
        /,
        *,
        maximum_interactions: int,
        branch_capacity: int,
        traversal_stack_capacity: int = 64,
        triangle_leaf_size: int = 8,
        record_history: bool = False,
        ray_tolerance: float = 1e-9,
        tie_tolerance: float = 1e-9,
        power_tolerance: float = 0.0,
    ):
        maximum = int(maximum_interactions)
        branches = int(branch_capacity)
        stack = int(traversal_stack_capacity)
        leaf = int(triangle_leaf_size)
        if maximum < 1 or branches < 1 or stack < 1 or leaf < 1:
            raise ValueError("All non-sequential capacities must be positive.")
        tolerances = (ray_tolerance, tie_tolerance, power_tolerance)
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0 for value in tolerances
        ):
            raise ValueError("Non-sequential tolerances must be finite and non-negative.")
        self.surfaces = surfaces
        self.maximum_interactions = maximum
        self.branch_capacity = branches
        self.traversal_stack_capacity = stack
        self.triangle_leaf_size = leaf
        self.record_history = bool(record_history)
        self.ray_tolerance = float(ray_tolerance)
        self.tie_tolerance = float(tie_tolerance)
        self.power_tolerance = float(power_tolerance)


class PreparedNonSequentialOptics(StrictModule, NonTrainableState):
    """Prepared immutable scene with fixed execution/resource bounds."""

    surfaces: NonSequentialSurfaceTable
    triangle_query: PreparedTriangleRayQuery
    maximum_interactions: int = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    record_history: bool = eqx.field(static=True)
    ray_tolerance: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    maximum_triangle_tests: int = eqx.field(static=True)
    maximum_branch_candidates: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    required_bytes_per_ray: int = eqx.field(static=True)
    exact_visibility: bool = eqx.field(static=True)
    pathwise_differentiability_claimed: bool = eqx.field(static=True)


class NonSequentialOpticsResult(StrictModule, NonTrainableState):
    """Terminal fixed branch arrays, optional history, and complete power ledger."""

    rays: OpticalRayState
    powers: Array
    medium_indices: Array
    live: Array
    launched_power: Array
    absorbed_power: Array
    detected_power: Array
    escaped_power: Array
    discarded_power: Array
    ambiguous_power: Array
    truncated_power: Array
    live_power: Array
    power_ledger_residual: Array
    interaction_counts: Array
    status: Array
    successful: Array
    finite: Array
    history_origins: Array
    history_directions: Array
    history_powers: Array
    history_live: Array
    history_triangle_indices: Array
    history_distances: Array


def prepare_nonsequential_optics(
    plan: NonSequentialOpticsPlan,
    /,
) -> PreparedNonSequentialOptics:
    """Build the conservative triangle hierarchy and fixed resource evidence."""

    triangle_plan = TriangleRayQueryPlan(
        plan.surfaces.vertices,
        plan.surfaces.triangles,
        entity_ids=plan.surfaces.surface_ids,
        leaf_size=plan.triangle_leaf_size,
        traversal_stack_capacity=plan.traversal_stack_capacity,
        acceleration="bvh",
        forward_tolerance=plan.ray_tolerance,
        tie_tolerance=plan.tie_tolerance,
    )
    triangle_query = prepare_triangle_ray_query(triangle_plan)
    scalar_bytes = int(plan.surfaces.vertices.dtype.itemsize)
    history_capacity = plan.maximum_interactions + 1 if plan.record_history else 0
    state_scalars = 3 + 3 + 1 + 1 + 1 + 1
    required_bytes = plan.branch_capacity * scalar_bytes * state_scalars
    required_bytes += history_capacity * plan.branch_capacity * scalar_bytes * 8
    return PreparedNonSequentialOptics(
        plan.surfaces,
        triangle_query,
        plan.maximum_interactions,
        plan.branch_capacity,
        plan.record_history,
        plan.ray_tolerance,
        plan.power_tolerance,
        plan.maximum_interactions * plan.branch_capacity * triangle_query.triangle_count,
        2 * plan.branch_capacity,
        history_capacity,
        int(required_bytes),
        True,
        False,
    )


def _select_candidates(
    capacity: int,
    positions: Array,
    directions: Array,
    refractive_indices: Array,
    geometric_lengths: Array,
    optical_lengths: Array,
    powers: Array,
    medium_indices: Array,
    live: Array,
) -> tuple[Array, ...]:
    count = live.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    sentinel = jnp.asarray(count, dtype=jnp.int32)
    ordering = jnp.sort(jnp.where(live, indices, sentinel))[:capacity]
    selected = ordering < count
    safe = jnp.minimum(ordering, count - 1)
    accepted_rank = jnp.cumsum(live.astype(jnp.int32)) - 1
    accepted = live & (accepted_rank < capacity)
    dropped = jnp.sum(jnp.where(live & ~accepted, powers, 0.0))
    return (
        jnp.where(selected[:, None], positions[safe], 0.0),
        jnp.where(selected[:, None], directions[safe], 0.0),
        jnp.where(selected, refractive_indices[safe], 1.0),
        jnp.where(selected, geometric_lengths[safe], 0.0),
        jnp.where(selected, optical_lengths[safe], 0.0),
        jnp.where(selected, powers[safe], 0.0),
        jnp.where(selected, medium_indices[safe], 0).astype(jnp.int32),
        selected,
        dropped,
    )


def _trace_one(
    prepared: PreparedNonSequentialOptics,
    origin: Array,
    direction: Array,
    refractive_index: Array,
    geometric_length: Array,
    optical_length: Array,
    launched_power: Array,
    initial_medium: Array,
) -> tuple[Array, ...]:
    capacity = prepared.branch_capacity
    dtype = origin.dtype
    positions = jnp.zeros((capacity, 3), dtype=dtype).at[0].set(origin)
    directions = jnp.zeros((capacity, 3), dtype=dtype).at[0].set(direction)
    indices = jnp.ones((capacity,), dtype=dtype).at[0].set(refractive_index)
    geometric = jnp.zeros((capacity,), dtype=dtype).at[0].set(geometric_length)
    optical = jnp.zeros((capacity,), dtype=dtype).at[0].set(optical_length)
    powers = jnp.zeros((capacity,), dtype=dtype).at[0].set(launched_power)
    media = jnp.zeros((capacity,), dtype=jnp.int32).at[0].set(initial_medium)
    direction_norm = jnp.sqrt(jnp.sum(direction * direction))
    direction_ok = jnp.isfinite(direction_norm) & (direction_norm > 0.0)
    safe_initial_medium = jnp.clip(
        initial_medium, 0, prepared.surfaces.refractive_indices.shape[0] - 1
    )
    expected_index = prepared.surfaces.refractive_indices[safe_initial_medium]
    index_tolerance = 32.0 * jnp.finfo(dtype).eps * jnp.maximum(1.0, expected_index)
    input_finite = (
        jnp.all(jnp.isfinite(origin))
        & jnp.all(jnp.isfinite(direction))
        & direction_ok
        & jnp.isfinite(refractive_index)
        & (refractive_index > 0.0)
        & (jnp.abs(refractive_index - expected_index) <= index_tolerance)
        & jnp.isfinite(geometric_length)
        & jnp.isfinite(optical_length)
        & jnp.isfinite(launched_power)
        & (launched_power >= 0.0)
        & (initial_medium >= 0)
        & (initial_medium < prepared.surfaces.refractive_indices.shape[0])
    )
    live = (
        jnp.zeros((capacity,), dtype=bool)
        .at[0]
        .set(input_finite & (launched_power > 0.0))
    )
    detector = jnp.zeros((prepared.surfaces.detector_count,), dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    history_origins = jnp.zeros((prepared.history_capacity, capacity, 3), dtype=dtype)
    history_directions = jnp.zeros((prepared.history_capacity, capacity, 3), dtype=dtype)
    history_powers = jnp.zeros((prepared.history_capacity, capacity), dtype=dtype)
    history_live = jnp.zeros((prepared.history_capacity, capacity), dtype=bool)
    history_triangles = jnp.full(
        (max(prepared.history_capacity - 1, 0), capacity), -1, dtype=jnp.int32
    )
    history_distances = jnp.zeros(
        (max(prepared.history_capacity - 1, 0), capacity), dtype=dtype
    )
    if prepared.record_history:
        history_origins = history_origins.at[0].set(positions)
        history_directions = history_directions.at[0].set(directions)
        history_powers = history_powers.at[0].set(powers)
        history_live = history_live.at[0].set(live)

    initial_state = (
        positions,
        directions,
        indices,
        geometric,
        optical,
        powers,
        media,
        live,
        zero,
        detector,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        history_origins,
        history_directions,
        history_powers,
        history_live,
        history_triangles,
        history_distances,
    )

    def step(interaction, state):
        (
            positions_,
            directions_,
            indices_,
            geometric_,
            optical_,
            powers_,
            media_,
            live_,
            absorbed_,
            detector_,
            escaped_,
            discarded_,
            ambiguous_,
            truncated_,
            interactions_,
            saw_ambiguity,
            saw_traversal,
            saw_medium,
            saw_interface,
            saw_branch_capacity,
            history_origins_,
            history_directions_,
            history_powers_,
            history_live_,
            history_triangles_,
            history_distances_,
        ) = state
        hit = intersect_triangle_rays(prepared.triangle_query, positions_, directions_)
        hit_status = hit.status
        success = live_ & (hit_status == int(TriangleRayIntersectionStatus.SUCCESS))
        miss = live_ & (hit_status == int(TriangleRayIntersectionStatus.MISS))
        ambiguous_hit = live_ & (
            hit_status == int(TriangleRayIntersectionStatus.AMBIGUOUS_HIT)
        )
        traversal_failure = live_ & (
            hit_status == int(TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED)
        )
        other_query_failure = live_ & ~(
            success | miss | ambiguous_hit | traversal_failure
        )
        escaped_ = escaped_ + jnp.sum(jnp.where(miss, powers_, 0.0))
        ambiguous_ = ambiguous_ + jnp.sum(jnp.where(ambiguous_hit, powers_, 0.0))
        truncated_ = truncated_ + jnp.sum(
            jnp.where(traversal_failure | other_query_failure, powers_, 0.0)
        )
        saw_ambiguity = saw_ambiguity | jnp.any(ambiguous_hit)
        saw_traversal = saw_traversal | jnp.any(traversal_failure | other_query_failure)

        triangle = jnp.maximum(hit.triangle_indices, 0)
        surface_kind = prepared.surfaces.surface_kinds[triangle]
        branch_mode = prepared.surfaces.branch_modes[triangle]
        normal = hit.oriented_normals
        direction_dot_normal = jnp.sum(directions_ * normal, axis=-1)
        positive_orientation = direction_dot_normal > 0.0
        incident_medium = jnp.where(
            positive_orientation,
            prepared.surfaces.negative_medium_indices[triangle],
            prepared.surfaces.positive_medium_indices[triangle],
        )
        transmitted_medium = jnp.where(
            positive_orientation,
            prepared.surfaces.positive_medium_indices[triangle],
            prepared.surfaces.negative_medium_indices[triangle],
        )
        interface_normal = jnp.where(positive_orientation[:, None], normal, -normal)
        medium_match = success & (media_ == incident_medium)
        mismatch = success & ~medium_match
        truncated_ = truncated_ + jnp.sum(jnp.where(mismatch, powers_, 0.0))
        saw_medium = saw_medium | jnp.any(mismatch)
        valid_hit = medium_match
        segment_distance = jnp.where(valid_hit, hit.intersection.distances, 0.0)
        hit_position = jnp.where(valid_hit[:, None], hit.intersection.points, positions_)
        next_geometric = geometric_ + segment_distance
        next_optical = optical_ + segment_distance * indices_
        interactions_ = (interactions_ + jnp.sum(valid_hit, dtype=jnp.int32)).astype(
            jnp.int32
        )

        detector_surface = valid_hit & (
            surface_kind == int(NonSequentialSurfaceKind.DETECTOR)
        )
        incidence_cosine = jnp.abs(direction_dot_normal)
        detector_accepted = detector_surface & (
            incidence_cosine >= prepared.surfaces.detector_acceptance_cosines[triangle]
        )
        detector_rejected = detector_surface & ~detector_accepted
        detector_index = prepared.surfaces.detector_indices[triangle]
        if prepared.surfaces.detector_count > 0:
            detector_ = detector_.at[jnp.maximum(detector_index, 0)].add(
                jnp.where(detector_accepted, powers_, 0.0)
            )
        absorber = valid_hit & (surface_kind == int(NonSequentialSurfaceKind.ABSORBER))
        absorbed_ = absorbed_ + jnp.sum(jnp.where(absorber, powers_, 0.0))

        dielectric = valid_hit & (
            surface_kind == int(NonSequentialSurfaceKind.DIELECTRIC)
        )
        mirror = valid_hit & (surface_kind == int(NonSequentialSurfaceKind.MIRROR))
        interface_active = dielectric | mirror
        incident_index = prepared.surfaces.refractive_indices[
            jnp.maximum(incident_medium, 0)
        ]
        transmitted_index = prepared.surfaces.refractive_indices[
            jnp.maximum(transmitted_medium, 0)
        ]
        interface = evaluate_refractive_interface(
            directions_, interface_normal, incident_index, transmitted_index
        )
        interface_ok = interface_active & interface.reflection_valid
        interface_failure = interface_active & ~interface.reflection_valid
        truncated_ = truncated_ + jnp.sum(jnp.where(interface_failure, powers_, 0.0))
        saw_interface = saw_interface | jnp.any(interface_failure)
        reflectance = jnp.mean(interface.reflectance, axis=-1)
        transmittance = jnp.mean(interface.transmittance, axis=-1)
        reflection_power = jnp.where(mirror, powers_, powers_ * reflectance)
        transmission_power = powers_ * transmittance
        reflection_requested = branch_mode != int(
            NonSequentialBranchMode.TRANSMISSION_ONLY
        )
        transmission_requested = branch_mode != int(
            NonSequentialBranchMode.REFLECTION_ONLY
        )
        reflection_live = (
            interface_ok
            & reflection_requested
            & (reflection_power > prepared.power_tolerance)
        )
        transmission_live = (
            dielectric
            & interface.transmission_valid
            & transmission_requested
            & (transmission_power > prepared.power_tolerance)
        )
        discarded_ = discarded_ + jnp.sum(
            jnp.where(interface_ok & ~reflection_requested, reflection_power, 0.0)
            + jnp.where(
                dielectric & interface.transmission_valid & ~transmission_requested,
                transmission_power,
                0.0,
            )
        )
        tiny_power = jnp.where(
            interface_ok & reflection_requested & ~reflection_live,
            reflection_power,
            0.0,
        ) + jnp.where(
            dielectric
            & interface.transmission_valid
            & transmission_requested
            & ~transmission_live,
            transmission_power,
            0.0,
        )
        discarded_ = discarded_ + jnp.sum(tiny_power)

        passthrough_live = detector_rejected
        candidate_positions = jnp.stack((hit_position, hit_position), axis=1).reshape(
            (2 * capacity, 3)
        )
        candidate_directions = jnp.stack(
            (
                jnp.where(
                    passthrough_live[:, None],
                    directions_,
                    interface.reflected_directions,
                ),
                jnp.where(
                    passthrough_live[:, None],
                    directions_,
                    interface.transmitted_directions,
                ),
            ),
            axis=1,
        ).reshape((2 * capacity, 3))
        candidate_indices = jnp.stack(
            (
                indices_,
                prepared.surfaces.refractive_indices[jnp.maximum(transmitted_medium, 0)],
            ),
            axis=1,
        ).reshape((2 * capacity,))
        candidate_geometric = jnp.stack((next_geometric, next_geometric), axis=1).reshape(
            (2 * capacity,)
        )
        candidate_optical = jnp.stack((next_optical, next_optical), axis=1).reshape(
            (2 * capacity,)
        )
        candidate_powers = jnp.stack(
            (
                jnp.where(passthrough_live, powers_, reflection_power),
                transmission_power,
            ),
            axis=1,
        ).reshape((2 * capacity,))
        candidate_media = jnp.stack((media_, transmitted_medium), axis=1).reshape(
            (2 * capacity,)
        )
        candidate_live = jnp.stack(
            (reflection_live | passthrough_live, transmission_live), axis=1
        ).reshape((2 * capacity,))
        (
            positions_,
            directions_,
            indices_,
            geometric_,
            optical_,
            powers_,
            media_,
            live_,
            dropped,
        ) = _select_candidates(
            capacity,
            candidate_positions,
            candidate_directions,
            candidate_indices,
            candidate_geometric,
            candidate_optical,
            candidate_powers,
            candidate_media,
            candidate_live,
        )
        truncated_ = truncated_ + dropped
        saw_branch_capacity = saw_branch_capacity | (dropped > prepared.power_tolerance)
        if prepared.record_history:
            history_origins_ = history_origins_.at[interaction + 1].set(positions_)
            history_directions_ = history_directions_.at[interaction + 1].set(directions_)
            history_powers_ = history_powers_.at[interaction + 1].set(powers_)
            history_live_ = history_live_.at[interaction + 1].set(live_)
            history_triangles_ = history_triangles_.at[interaction].set(
                hit.triangle_indices
            )
            history_distances_ = history_distances_.at[interaction].set(
                hit.intersection.distances
            )
        return (
            positions_,
            directions_,
            indices_,
            geometric_,
            optical_,
            powers_,
            media_,
            live_,
            absorbed_,
            detector_,
            escaped_,
            discarded_,
            ambiguous_,
            truncated_,
            interactions_,
            saw_ambiguity,
            saw_traversal,
            saw_medium,
            saw_interface,
            saw_branch_capacity,
            history_origins_,
            history_directions_,
            history_powers_,
            history_live_,
            history_triangles_,
            history_distances_,
        )

    final = jax.lax.fori_loop(0, prepared.maximum_interactions, step, initial_state)
    (
        positions,
        directions,
        indices,
        geometric,
        optical,
        powers,
        media,
        live,
        absorbed,
        detector,
        escaped,
        discarded,
        ambiguous,
        truncated,
        interactions,
        saw_ambiguity,
        saw_traversal,
        saw_medium,
        saw_interface,
        saw_branch_capacity,
        history_origins,
        history_directions,
        history_powers,
        history_live,
        history_triangles,
        history_distances,
    ) = final
    live_power = jnp.sum(jnp.where(live, powers, 0.0))
    detected = jnp.sum(detector)
    ledger_residual = launched_power - (
        absorbed + detected + escaped + discarded + ambiguous + truncated + live_power
    )
    finite = (
        jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(directions))
        & jnp.all(jnp.isfinite(indices))
        & jnp.all(jnp.isfinite(powers))
        & jnp.isfinite(ledger_residual)
    )
    status = jnp.where(
        ~input_finite,
        int(NonSequentialOpticsStatus.INVALID_INPUT),
        jnp.where(
            saw_ambiguity,
            int(NonSequentialOpticsStatus.AMBIGUOUS_INTERSECTION),
            jnp.where(
                saw_traversal,
                int(NonSequentialOpticsStatus.TRAVERSAL_CAPACITY_EXHAUSTED),
                jnp.where(
                    saw_medium,
                    int(NonSequentialOpticsStatus.MEDIUM_MISMATCH),
                    jnp.where(
                        saw_interface,
                        int(NonSequentialOpticsStatus.INTERFACE_FAILURE),
                        jnp.where(
                            saw_branch_capacity,
                            int(NonSequentialOpticsStatus.BRANCH_CAPACITY_EXHAUSTED),
                            jnp.where(
                                live_power > prepared.power_tolerance,
                                int(
                                    NonSequentialOpticsStatus.INTERACTION_CAPACITY_EXHAUSTED
                                ),
                                jnp.where(
                                    ~finite,
                                    int(NonSequentialOpticsStatus.NONFINITE_RESULT),
                                    int(NonSequentialOpticsStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(NonSequentialOpticsStatus.SUCCESS)
    return (
        positions,
        directions,
        indices,
        geometric,
        optical,
        powers,
        media,
        live,
        launched_power,
        absorbed,
        detector,
        escaped,
        discarded,
        ambiguous,
        truncated,
        live_power,
        ledger_residual,
        interactions,
        status,
        successful,
        finite,
        history_origins,
        history_directions,
        history_powers,
        history_live,
        history_triangles,
        history_distances,
    )


def trace_nonsequential_optics(
    prepared: PreparedNonSequentialOptics,
    rays: OpticalRayState,
    powers: ArrayLike,
    medium_indices: ArrayLike,
    /,
) -> NonSequentialOpticsResult:
    """Trace exact fixed-interaction, fixed-branch scalar-power ray trees.

    Fresnel powers are the unpolarized mean of the documented ``(s, p)``
    interface values. Every omitted, failed, ambiguous, capacity-limited, live,
    absorbed, detected, or escaped contribution remains visible in the ledger.
    """

    batch_shape = rays.origins.shape[:-1]
    powers_ = jnp.asarray(powers, dtype=rays.origins.dtype)
    media_ = jnp.asarray(medium_indices, dtype=jnp.int32)
    powers_ = jnp.broadcast_to(powers_, batch_shape)
    media_ = jnp.broadcast_to(media_, batch_shape)
    flat_count = math.prod(batch_shape) if batch_shape else 1
    values = jax.vmap(
        lambda origin, direction, index, geometric, optical, power, medium: _trace_one(
            prepared, origin, direction, index, geometric, optical, power, medium
        )
    )(
        rays.origins.reshape((flat_count, 3)),
        rays.directions.reshape((flat_count, 3)),
        rays.refractive_indices.reshape((flat_count,)),
        rays.geometric_path_lengths.reshape((flat_count,)),
        rays.optical_path_lengths.reshape((flat_count,)),
        powers_.reshape((flat_count,)),
        media_.reshape((flat_count,)),
    )
    (
        origins,
        directions,
        indices,
        geometric,
        optical,
        terminal_powers,
        terminal_media,
        live,
        launched,
        absorbed,
        detector,
        escaped,
        discarded,
        ambiguous,
        truncated,
        live_power,
        residual,
        interactions,
        status,
        successful,
        finite,
        history_origins,
        history_directions,
        history_powers,
        history_live,
        history_triangles,
        history_distances,
    ) = values
    branch_shape = batch_shape + (prepared.branch_capacity,)
    terminal_rays = OpticalRayState(
        origins.reshape(branch_shape + (3,)),
        directions.reshape(branch_shape + (3,)),
        indices.reshape(branch_shape),
        geometric.reshape(branch_shape),
        optical.reshape(branch_shape),
    )
    return NonSequentialOpticsResult(
        terminal_rays,
        terminal_powers.reshape(branch_shape),
        terminal_media.reshape(branch_shape),
        live.reshape(branch_shape),
        launched.reshape(batch_shape),
        absorbed.reshape(batch_shape),
        detector.reshape(batch_shape + (prepared.surfaces.detector_count,)),
        escaped.reshape(batch_shape),
        discarded.reshape(batch_shape),
        ambiguous.reshape(batch_shape),
        truncated.reshape(batch_shape),
        live_power.reshape(batch_shape),
        residual.reshape(batch_shape),
        interactions.reshape(batch_shape),
        status.reshape(batch_shape),
        successful.reshape(batch_shape),
        finite.reshape(batch_shape),
        history_origins.reshape(
            batch_shape + (prepared.history_capacity, prepared.branch_capacity, 3)
        ),
        history_directions.reshape(
            batch_shape + (prepared.history_capacity, prepared.branch_capacity, 3)
        ),
        history_powers.reshape(
            batch_shape + (prepared.history_capacity, prepared.branch_capacity)
        ),
        history_live.reshape(
            batch_shape + (prepared.history_capacity, prepared.branch_capacity)
        ),
        history_triangles.reshape(
            batch_shape
            + (max(prepared.history_capacity - 1, 0), prepared.branch_capacity)
        ),
        history_distances.reshape(
            batch_shape
            + (max(prepared.history_capacity - 1, 0), prepared.branch_capacity)
        ),
    )


__all__ = [
    "NonSequentialBranchMode",
    "NonSequentialOpticsPlan",
    "NonSequentialOpticsResult",
    "NonSequentialOpticsStatus",
    "NonSequentialSurfaceKind",
    "NonSequentialSurfaceTable",
    "PreparedNonSequentialOptics",
    "prepare_nonsequential_optics",
    "trace_nonsequential_optics",
]
