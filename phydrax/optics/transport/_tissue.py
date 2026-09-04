#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
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
from ..geometric._interface import evaluate_refractive_interface
from ..geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)


class TissueTransportStatus(IntEnum):
    """Terminal status of one bounded photon-packet history."""

    SUCCESS = 0
    INVALID_INPUT = 1
    AMBIGUOUS_INTERSECTION = 2
    TRAVERSAL_CAPACITY_EXHAUSTED = 3
    MEDIUM_MISMATCH = 4
    INTERFACE_FAILURE = 5
    BRANCH_CAPACITY_EXHAUSTED = 6
    INTERACTION_CAPACITY_EXHAUSTED = 7
    NONFINITE_RESULT = 8


class TissueTransportCoefficients(StrictModule, NonTrainableState):
    """Piecewise-homogeneous scalar tissue coefficients.

    ``mu_a`` and ``mu_s`` are inverse-length coefficients, ``g`` is the
    Henyey--Greenstein first moment, and ``n`` is the positive real refractive
    index. All arrays have one entry per medium.
    """

    mu_a: Array
    mu_s: Array
    g: Array
    n: Array
    medium_count: int = eqx.field(static=True)

    def __init__(
        self,
        mu_a: ArrayLike,
        mu_s: ArrayLike,
        g: ArrayLike,
        n: ArrayLike,
        /,
    ):
        absorption = np.asarray(mu_a)
        scattering = np.asarray(mu_s)
        anisotropy = np.asarray(g)
        refractive = np.asarray(n)
        if any(
            value.ndim != 1 for value in (absorption, scattering, anisotropy, refractive)
        ):
            raise ValueError("Tissue coefficient arrays must be rank one.")
        if (
            not (
                absorption.shape
                == scattering.shape
                == anisotropy.shape
                == refractive.shape
            )
            or absorption.size < 1
        ):
            raise ValueError(
                "Tissue coefficient arrays must have one matching non-empty shape."
            )
        if any(
            np.iscomplexobj(value)
            for value in (absorption, scattering, anisotropy, refractive)
        ):
            raise TypeError("Tissue transport coefficients must be real-valued.")
        if not all(
            np.all(np.isfinite(value))
            for value in (absorption, scattering, anisotropy, refractive)
        ):
            raise ValueError("Tissue transport coefficients must be finite.")
        if np.any(absorption < 0.0) or np.any(scattering < 0.0):
            raise ValueError("mu_a and mu_s must be non-negative.")
        if np.any(np.abs(anisotropy) >= 1.0):
            raise ValueError("Henyey--Greenstein g must lie strictly between -1 and 1.")
        if np.any(refractive <= 0.0):
            raise ValueError("Tissue refractive indices must be positive.")
        dtype = jnp.result_type(absorption, scattering, anisotropy, refractive, 0.0)
        self.mu_a = jnp.asarray(absorption, dtype=dtype)
        self.mu_s = jnp.asarray(scattering, dtype=dtype)
        self.g = jnp.asarray(anisotropy, dtype=dtype)
        self.n = jnp.asarray(refractive, dtype=dtype)
        self.medium_count = int(absorption.size)


class TissueTransportPlan(StrictModule, NonTrainableState):
    """Host plan for fixed-capacity Monte Carlo transport through triangle media."""

    surfaces: NonSequentialSurfaceTable
    coefficients: TissueTransportCoefficients
    maximum_interactions: int = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    fresnel_branching: Literal["stochastic", "expected-split"] = eqx.field(static=True)
    roulette_threshold: float = eqx.field(static=True)
    roulette_survival_probability: float = eqx.field(static=True)
    traversal_stack_capacity: int = eqx.field(static=True)
    triangle_leaf_size: int = eqx.field(static=True)
    ray_tolerance: float = eqx.field(static=True)
    tie_tolerance: float = eqx.field(static=True)
    weight_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        surfaces: NonSequentialSurfaceTable,
        coefficients: TissueTransportCoefficients,
        /,
        *,
        maximum_interactions: int,
        branch_capacity: int = 1,
        fresnel_branching: Literal["stochastic", "expected-split"] = "stochastic",
        roulette_threshold: float = 0.0,
        roulette_survival_probability: float = 0.1,
        traversal_stack_capacity: int = 64,
        triangle_leaf_size: int = 8,
        ray_tolerance: float = 1e-9,
        tie_tolerance: float = 1e-9,
        weight_tolerance: float = 0.0,
    ):
        maximum = int(maximum_interactions)
        branches = int(branch_capacity)
        stack = int(traversal_stack_capacity)
        leaf = int(triangle_leaf_size)
        if maximum < 1 or branches < 1 or stack < 1 or leaf < 1:
            raise ValueError("All tissue transport capacities must be positive.")
        if fresnel_branching not in ("stochastic", "expected-split"):
            raise ValueError(
                "fresnel_branching must be 'stochastic' or 'expected-split'."
            )
        roulette_threshold_ = float(roulette_threshold)
        roulette_probability = float(roulette_survival_probability)
        if not math.isfinite(roulette_threshold_) or roulette_threshold_ < 0.0:
            raise ValueError("roulette_threshold must be finite and non-negative.")
        if (
            not math.isfinite(roulette_probability)
            or not 0.0 < roulette_probability <= 1.0
        ):
            raise ValueError("roulette_survival_probability must lie in (0, 1].")
        tolerances = (ray_tolerance, tie_tolerance, weight_tolerance)
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0 for value in tolerances
        ):
            raise ValueError("Transport tolerances must be finite and non-negative.")
        if coefficients.medium_count != int(surfaces.refractive_indices.shape[0]):
            raise ValueError(
                "Surface and transport medium tables must have matching length."
            )
        if not np.array_equal(
            np.asarray(coefficients.n), np.asarray(surfaces.refractive_indices)
        ):
            raise ValueError(
                "Surface and transport refractive-index arrays must match exactly."
            )
        self.surfaces = surfaces
        self.coefficients = coefficients
        self.maximum_interactions = maximum
        self.branch_capacity = branches
        self.fresnel_branching = fresnel_branching
        self.roulette_threshold = roulette_threshold_
        self.roulette_survival_probability = roulette_probability
        self.traversal_stack_capacity = stack
        self.triangle_leaf_size = leaf
        self.ray_tolerance = float(ray_tolerance)
        self.tie_tolerance = float(tie_tolerance)
        self.weight_tolerance = float(weight_tolerance)


class PreparedTissueTransport(StrictModule, NonTrainableState):
    """Prepared exact geometry and explicit fixed-work transport evidence."""

    surfaces: NonSequentialSurfaceTable
    coefficients: TissueTransportCoefficients
    triangle_query: PreparedTriangleRayQuery
    maximum_interactions: int = eqx.field(static=True)
    branch_capacity: int = eqx.field(static=True)
    fresnel_branching: Literal["stochastic", "expected-split"] = eqx.field(static=True)
    roulette_threshold: float = eqx.field(static=True)
    roulette_survival_probability: float = eqx.field(static=True)
    weight_tolerance: float = eqx.field(static=True)
    maximum_triangle_tests: int = eqx.field(static=True)
    maximum_random_draws: int = eqx.field(static=True)
    required_bytes_per_photon: int = eqx.field(static=True)
    exact_visibility: bool = eqx.field(static=True)
    remaining_optical_depth: bool = eqx.field(static=True)
    implicit_capture: bool = eqx.field(static=True)
    keyed_sampling: bool = eqx.field(static=True)
    pathwise_differentiability_claimed: bool = eqx.field(static=True)


class TissueTransportTallies(StrictModule, NonTrainableState):
    """Fixed absorption, flux, detector, escape, roulette, and residual tallies."""

    absorption: Array
    surface_flux: Array
    detector: Array
    escape: Array
    roulette: Array
    live: Array
    truncated: Array
    launched: Array
    ledger_residual: Array


class TissueTransportResult(StrictModule, NonTrainableState):
    """Per-photon and estimator-level transport output with uncertainty evidence."""

    per_photon_tallies: TissueTransportTallies
    tallies: TissueTransportTallies
    standard_errors: TissueTransportTallies
    terminal_positions: Array
    terminal_directions: Array
    terminal_weights: Array
    terminal_medium_indices: Array
    terminal_optical_depths: Array
    terminal_live: Array
    interaction_counts: Array
    status: Array
    successful: Array
    sample_count: Array
    maximum_absolute_ledger_residual: Array
    finite: Array
    all_successful: Array


def prepare_tissue_transport(
    plan: TissueTransportPlan,
    /,
) -> PreparedTissueTransport:
    """Prepare exact triangle traversal and fixed work/resource evidence."""

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
    scalar_bytes = int(plan.coefficients.mu_a.dtype.itemsize)
    state_scalars = 3 + 3 + 1 + 1 + 1 + 1
    tally_scalars = (
        plan.coefficients.medium_count
        + plan.surfaces.surface_count
        + plan.surfaces.detector_count
        + 5
    )
    required_bytes = plan.branch_capacity * scalar_bytes * state_scalars
    required_bytes += tally_scalars * scalar_bytes
    maximum_random_draws = plan.maximum_interactions * plan.branch_capacity * 5
    return PreparedTissueTransport(
        plan.surfaces,
        plan.coefficients,
        triangle_query,
        plan.maximum_interactions,
        plan.branch_capacity,
        plan.fresnel_branching,
        plan.roulette_threshold,
        plan.roulette_survival_probability,
        plan.weight_tolerance,
        plan.maximum_interactions * plan.branch_capacity * triangle_query.triangle_count,
        maximum_random_draws,
        int(required_bytes),
        True,
        True,
        True,
        True,
        False,
    )


def _uniform_for_lanes(base_key: Array, semantic_ids: Array, stream: int) -> Array:
    stream_key = jr.fold_in(base_key, stream)
    keys = jax.vmap(lambda semantic_id: jr.fold_in(stream_key, semantic_id))(
        semantic_ids.astype(jnp.uint32)
    )
    return jax.vmap(lambda key: jr.uniform(key))(keys)


def _sample_optical_depth(uniform: Array, dtype: jnp.dtype) -> Array:
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    return -jnp.log(jnp.maximum(uniform.astype(dtype), tiny))


def _sample_henyey_greenstein(
    directions: Array, anisotropy: Array, cosine_uniform: Array, azimuth_uniform: Array
) -> Array:
    small = jnp.abs(anisotropy) < 1e-6
    denominator = 1.0 - anisotropy + 2.0 * anisotropy * cosine_uniform
    ratio = (1.0 - anisotropy * anisotropy) / jnp.where(small, 1.0, denominator)
    hg_cosine = (1.0 + anisotropy * anisotropy - ratio * ratio) / jnp.where(
        small, 1.0, 2.0 * anisotropy
    )
    cosine = jnp.where(small, 2.0 * cosine_uniform - 1.0, hg_cosine)
    cosine = jnp.clip(cosine, -1.0, 1.0)
    sine = jnp.sqrt(jnp.maximum(0.0, 1.0 - cosine * cosine))
    azimuth = 2.0 * jnp.pi * azimuth_uniform
    z_reference = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, 1.0), dtype=directions.dtype), directions.shape
    )
    x_reference = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0), dtype=directions.dtype), directions.shape
    )
    reference = jnp.where(
        (jnp.abs(directions[:, 2]) < 0.9)[:, None], z_reference, x_reference
    )
    transverse_one = jnp.cross(reference, directions)
    transverse_one = (
        transverse_one
        / jnp.sqrt(jnp.sum(transverse_one * transverse_one, axis=-1))[:, None]
    )
    transverse_two = jnp.cross(directions, transverse_one)
    scattered = cosine[:, None] * directions + sine[:, None] * (
        jnp.cos(azimuth)[:, None] * transverse_one
        + jnp.sin(azimuth)[:, None] * transverse_two
    )
    return scattered / jnp.sqrt(jnp.sum(scattered * scattered, axis=-1))[:, None]


def _compact_candidates(
    capacity: int,
    positions: Array,
    directions: Array,
    weights: Array,
    media: Array,
    optical_depths: Array,
    semantic_ids: Array,
    live: Array,
) -> tuple[Array, ...]:
    count = live.shape[0]
    source_indices = jnp.arange(count, dtype=jnp.int32)
    sentinel = jnp.asarray(count, dtype=jnp.int32)
    ordering = jnp.sort(jnp.where(live, source_indices, sentinel))[:capacity]
    selected = ordering < count
    safe = jnp.minimum(ordering, count - 1)
    rank = jnp.cumsum(live.astype(jnp.int32)) - 1
    accepted = live & (rank < capacity)
    dropped = jnp.sum(jnp.where(live & ~accepted, weights, 0.0))
    return (
        jnp.where(selected[:, None], positions[safe], 0.0),
        jnp.where(selected[:, None], directions[safe], 0.0),
        jnp.where(selected, weights[safe], 0.0),
        jnp.where(selected, media[safe], 0).astype(jnp.int32),
        jnp.where(selected, optical_depths[safe], 0.0),
        jnp.where(selected, semantic_ids[safe], 0).astype(jnp.uint32),
        selected,
        dropped,
    )


def _transport_one(
    prepared: PreparedTissueTransport,
    root_key: Array,
    photon_id: Array,
    origin: Array,
    direction: Array,
    initial_medium: Array,
    initial_weight: Array,
) -> tuple[Array, ...]:
    capacity = prepared.branch_capacity
    dtype = origin.dtype
    direction_norm = jnp.sqrt(jnp.sum(direction * direction))
    direction_ok = jnp.isfinite(direction_norm) & (direction_norm > 0.0)
    unit_direction = direction / jnp.where(direction_ok, direction_norm, 1.0)
    input_valid = (
        jnp.all(jnp.isfinite(origin))
        & jnp.all(jnp.isfinite(direction))
        & direction_ok
        & (initial_medium >= 0)
        & (initial_medium < prepared.coefficients.medium_count)
        & jnp.isfinite(initial_weight)
        & (initial_weight >= 0.0)
    )
    photon_key = jr.fold_in(root_key, photon_id.astype(jnp.uint32))
    initial_tau = _sample_optical_depth(jr.uniform(jr.fold_in(photon_key, 0)), dtype)
    positions = jnp.zeros((capacity, 3), dtype=dtype).at[0].set(origin)
    directions = jnp.zeros((capacity, 3), dtype=dtype).at[0].set(unit_direction)
    weights = jnp.zeros((capacity,), dtype=dtype).at[0].set(initial_weight)
    media = (
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[0]
        .set(jnp.clip(initial_medium, 0, prepared.coefficients.medium_count - 1))
    )
    optical_depths = jnp.zeros((capacity,), dtype=dtype).at[0].set(initial_tau)
    semantic_ids = jnp.zeros((capacity,), dtype=jnp.uint32)
    live = (
        jnp.zeros((capacity,), dtype=bool).at[0].set(input_valid & (initial_weight > 0.0))
    )
    absorption = jnp.zeros((prepared.coefficients.medium_count,), dtype=dtype)
    surface_flux = jnp.zeros((prepared.surfaces.surface_count,), dtype=dtype)
    detector = jnp.zeros((prepared.surfaces.detector_count,), dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    initial_state = (
        positions,
        directions,
        weights,
        media,
        optical_depths,
        semantic_ids,
        live,
        absorption,
        surface_flux,
        detector,
        zero,
        zero,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
    )

    def step(interaction, state):
        (
            positions_,
            directions_,
            weights_,
            media_,
            optical_depths_,
            semantic_ids_,
            live_,
            absorption_,
            surface_flux_,
            detector_,
            escape_,
            roulette_,
            interaction_count_,
            saw_ambiguity,
            saw_traversal,
            saw_medium,
            saw_interface,
            saw_branch_capacity,
        ) = state
        event_key = jr.fold_in(photon_key, interaction + 1)
        hit = intersect_triangle_rays(prepared.triangle_query, positions_, directions_)
        query_success = live_ & (hit.status == int(TriangleRayIntersectionStatus.SUCCESS))
        query_miss = live_ & (hit.status == int(TriangleRayIntersectionStatus.MISS))
        ambiguous_hit = live_ & (
            hit.status == int(TriangleRayIntersectionStatus.AMBIGUOUS_HIT)
        )
        traversal_failure = live_ & (
            hit.status == int(TriangleRayIntersectionStatus.TRAVERSAL_CAPACITY_EXHAUSTED)
        )
        other_query_failure = live_ & ~(
            query_success | query_miss | ambiguous_hit | traversal_failure
        )
        failure_weight = jnp.sum(
            jnp.where(
                ambiguous_hit | traversal_failure | other_query_failure, weights_, 0.0
            )
        )
        saw_ambiguity = saw_ambiguity | jnp.any(ambiguous_hit)
        saw_traversal = saw_traversal | jnp.any(traversal_failure | other_query_failure)

        safe_triangle = jnp.maximum(hit.triangle_indices, 0)
        normal = hit.oriented_normals
        positive_orientation = jnp.sum(directions_ * normal, axis=-1) > 0.0
        incident_medium = jnp.where(
            positive_orientation,
            prepared.surfaces.negative_medium_indices[safe_triangle],
            prepared.surfaces.positive_medium_indices[safe_triangle],
        )
        transmitted_medium = jnp.where(
            positive_orientation,
            prepared.surfaces.positive_medium_indices[safe_triangle],
            prepared.surfaces.negative_medium_indices[safe_triangle],
        )
        interface_normal = jnp.where(positive_orientation[:, None], normal, -normal)
        medium_match = query_success & (media_ == incident_medium)
        mismatch = query_success & ~medium_match
        failure_weight = failure_weight + jnp.sum(jnp.where(mismatch, weights_, 0.0))
        saw_medium = saw_medium | jnp.any(mismatch)
        valid_boundary = medium_match

        extinction = (
            prepared.coefficients.mu_a[media_] + prepared.coefficients.mu_s[media_]
        )
        boundary_distance = jnp.where(valid_boundary, hit.intersection.distances, jnp.inf)
        boundary_optical_depth = extinction * boundary_distance
        transportable = query_miss | valid_boundary
        volume_event = (
            live_
            & transportable
            & (extinction > 0.0)
            & (optical_depths_ < boundary_optical_depth)
        )
        surface_event = valid_boundary & ~volume_event
        unbounded_escape = query_miss & (extinction == 0.0)
        volume_distance = optical_depths_ / jnp.where(extinction > 0.0, extinction, 1.0)
        event_distance = jnp.where(volume_event, volume_distance, boundary_distance)
        event_positions = (
            positions_
            + jnp.where(
                (volume_event | surface_event)[:, None], event_distance[:, None], 0.0
            )
            * directions_
        )
        remaining_depth = jnp.where(
            surface_event,
            jnp.maximum(optical_depths_ - boundary_optical_depth, 0.0),
            optical_depths_,
        )
        interaction_count_ = (
            interaction_count_ + jnp.sum(volume_event | surface_event, dtype=jnp.int32)
        ).astype(jnp.int32)

        albedo = prepared.coefficients.mu_s[media_] / jnp.where(
            extinction > 0.0, extinction, 1.0
        )
        absorbed_weight = jnp.where(volume_event, weights_ * (1.0 - albedo), 0.0)
        absorption_ = absorption_.at[media_].add(absorbed_weight)
        scattered_weight = weights_ * albedo
        cosine_uniform = _uniform_for_lanes(event_key, semantic_ids_, 1)
        azimuth_uniform = _uniform_for_lanes(event_key, semantic_ids_, 2)
        scattered_direction = _sample_henyey_greenstein(
            directions_,
            prepared.coefficients.g[media_],
            cosine_uniform,
            azimuth_uniform,
        )
        next_depth = _sample_optical_depth(
            _uniform_for_lanes(event_key, semantic_ids_, 3), dtype
        )
        volume_live = volume_event & (scattered_weight > prepared.weight_tolerance)
        failure_weight = failure_weight + jnp.sum(
            jnp.where(
                volume_event & ~volume_live & (scattered_weight > 0.0),
                scattered_weight,
                0.0,
            )
        )

        surface_kind = prepared.surfaces.surface_kinds[safe_triangle]
        detector_surface = surface_event & (
            surface_kind == int(NonSequentialSurfaceKind.DETECTOR)
        )
        incidence_cosine = jnp.abs(jnp.sum(directions_ * normal, axis=-1))
        detector_accepted = detector_surface & (
            incidence_cosine
            >= prepared.surfaces.detector_acceptance_cosines[safe_triangle]
        )
        detector_rejected = detector_surface & ~detector_accepted
        detector_index = prepared.surfaces.detector_indices[safe_triangle]
        if prepared.surfaces.detector_count > 0:
            detector_ = detector_.at[jnp.maximum(detector_index, 0)].add(
                jnp.where(detector_accepted, weights_, 0.0)
            )
        absorber = surface_event & (
            surface_kind == int(NonSequentialSurfaceKind.ABSORBER)
        )
        absorption_ = absorption_.at[media_].add(jnp.where(absorber, weights_, 0.0))
        mirror = surface_event & (surface_kind == int(NonSequentialSurfaceKind.MIRROR))
        dielectric = surface_event & (
            surface_kind == int(NonSequentialSurfaceKind.DIELECTRIC)
        )
        incident_index = prepared.coefficients.n[jnp.maximum(incident_medium, 0)]
        transmitted_index = prepared.coefficients.n[jnp.maximum(transmitted_medium, 0)]
        interface = evaluate_refractive_interface(
            directions_, interface_normal, incident_index, transmitted_index
        )
        interface_active = mirror | dielectric
        interface_failure = interface_active & ~interface.reflection_valid
        failure_weight = failure_weight + jnp.sum(
            jnp.where(interface_failure, weights_, 0.0)
        )
        saw_interface = saw_interface | jnp.any(interface_failure)
        reflectance = jnp.mean(interface.reflectance, axis=-1)
        transmittance = jnp.mean(interface.transmittance, axis=-1)
        reflectance = jnp.where(mirror, 1.0, reflectance)
        transmittance = jnp.where(mirror, 0.0, transmittance)

        fresnel_uniform = _uniform_for_lanes(event_key, semantic_ids_, 4)
        if prepared.fresnel_branching == "stochastic":
            choose_reflection = (
                interface_active
                & interface.reflection_valid
                & (fresnel_uniform < reflectance)
            )
            choose_transmission = (
                dielectric & interface.transmission_valid & ~choose_reflection
            )
            reflection_weight = weights_
            transmission_weight = weights_
        else:
            choose_reflection = (
                interface_active & interface.reflection_valid & (reflectance > 0.0)
            )
            choose_transmission = (
                dielectric & interface.transmission_valid & (transmittance > 0.0)
            )
            reflection_weight = weights_ * reflectance
            transmission_weight = weights_ * transmittance

        crossing_sign = jnp.where(positive_orientation, 1.0, -1.0)
        surface_id = prepared.surfaces.surface_ids[safe_triangle]
        surface_flux_ = surface_flux_.at[surface_id].add(
            jnp.where(choose_transmission, crossing_sign * transmission_weight, 0.0)
        )
        passthrough = detector_rejected
        escape_ = escape_ + jnp.sum(jnp.where(unbounded_escape, weights_, 0.0))

        first_position = jnp.where(
            volume_event[:, None], event_positions, hit.intersection.points
        )
        first_direction = jnp.where(
            volume_event[:, None], scattered_direction, interface.reflected_directions
        )
        first_weight = jnp.where(volume_event, scattered_weight, reflection_weight)
        first_medium = media_
        first_depth = jnp.where(volume_event, next_depth, remaining_depth)
        first_live = volume_live | choose_reflection | passthrough
        first_direction = jnp.where(passthrough[:, None], directions_, first_direction)
        first_weight = jnp.where(passthrough, weights_, first_weight)
        first_depth = jnp.where(passthrough, remaining_depth, first_depth)
        second_position = hit.intersection.points
        second_direction = interface.transmitted_directions
        second_weight = transmission_weight
        second_medium = transmitted_medium
        second_depth = remaining_depth
        second_live = choose_transmission

        candidate_positions = jnp.stack(
            (first_position, second_position), axis=1
        ).reshape((2 * capacity, 3))
        candidate_directions = jnp.stack(
            (first_direction, second_direction), axis=1
        ).reshape((2 * capacity, 3))
        candidate_weights = jnp.stack((first_weight, second_weight), axis=1).reshape(
            (2 * capacity,)
        )
        candidate_media = jnp.stack((first_medium, second_medium), axis=1).reshape(
            (2 * capacity,)
        )
        candidate_depths = jnp.stack((first_depth, second_depth), axis=1).reshape(
            (2 * capacity,)
        )
        candidate_semantic_ids = jnp.stack(
            (2 * semantic_ids_ + 1, 2 * semantic_ids_ + 2), axis=1
        ).reshape((2 * capacity,))
        candidate_live = jnp.stack((first_live, second_live), axis=1).reshape(
            (2 * capacity,)
        )
        (
            positions_,
            directions_,
            weights_,
            media_,
            optical_depths_,
            semantic_ids_,
            live_,
            capacity_dropped,
        ) = _compact_candidates(
            capacity,
            candidate_positions,
            candidate_directions,
            candidate_weights,
            candidate_media,
            candidate_depths,
            candidate_semantic_ids,
            candidate_live,
        )
        failure_weight = failure_weight + capacity_dropped
        saw_branch_capacity = saw_branch_capacity | (
            capacity_dropped > prepared.weight_tolerance
        )

        roulette_uniform = _uniform_for_lanes(event_key, semantic_ids_, 5)
        roulette_candidate = (
            live_
            & (prepared.roulette_threshold > 0.0)
            & (weights_ < prepared.roulette_threshold)
        )
        survive = roulette_candidate & (
            roulette_uniform < prepared.roulette_survival_probability
        )
        killed = roulette_candidate & ~survive
        boosted = weights_ / prepared.roulette_survival_probability
        roulette_ = roulette_ + jnp.sum(
            jnp.where(killed, weights_, 0.0) + jnp.where(survive, weights_ - boosted, 0.0)
        )
        weights_ = jnp.where(survive, boosted, weights_)
        live_ = live_ & ~killed
        return (
            positions_,
            directions_,
            weights_,
            media_,
            optical_depths_,
            semantic_ids_,
            live_,
            absorption_,
            surface_flux_,
            detector_,
            escape_,
            roulette_,
            interaction_count_,
            saw_ambiguity,
            saw_traversal,
            saw_medium,
            saw_interface,
            saw_branch_capacity,
        ), failure_weight

    def scan_step(state, interaction):
        next_state, truncated_increment = step(interaction, state)
        return next_state, truncated_increment

    final_state, truncated_increments = jax.lax.scan(
        scan_step,
        initial_state,
        jnp.arange(prepared.maximum_interactions, dtype=jnp.int32),
    )
    (
        positions,
        directions,
        weights,
        media,
        optical_depths,
        semantic_ids,
        live,
        absorption,
        surface_flux,
        detector,
        escape,
        roulette,
        interaction_count,
        saw_ambiguity,
        saw_traversal,
        saw_medium,
        saw_interface,
        saw_branch_capacity,
    ) = final_state
    del semantic_ids
    truncated = jnp.sum(truncated_increments)
    live_weight = jnp.sum(jnp.where(live, weights, 0.0))
    detected = jnp.sum(detector)
    absorbed = jnp.sum(absorption)
    ledger_residual = initial_weight - (
        absorbed + detected + escape + roulette + live_weight + truncated
    )
    finite = (
        jnp.all(jnp.isfinite(absorption))
        & jnp.all(jnp.isfinite(surface_flux))
        & jnp.all(jnp.isfinite(detector))
        & jnp.isfinite(escape)
        & jnp.isfinite(roulette)
        & jnp.isfinite(live_weight)
        & jnp.isfinite(truncated)
        & jnp.isfinite(ledger_residual)
        & jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(directions))
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(optical_depths))
    )
    status = jnp.where(
        ~input_valid,
        int(TissueTransportStatus.INVALID_INPUT),
        jnp.where(
            saw_ambiguity,
            int(TissueTransportStatus.AMBIGUOUS_INTERSECTION),
            jnp.where(
                saw_traversal,
                int(TissueTransportStatus.TRAVERSAL_CAPACITY_EXHAUSTED),
                jnp.where(
                    saw_medium,
                    int(TissueTransportStatus.MEDIUM_MISMATCH),
                    jnp.where(
                        saw_interface,
                        int(TissueTransportStatus.INTERFACE_FAILURE),
                        jnp.where(
                            saw_branch_capacity,
                            int(TissueTransportStatus.BRANCH_CAPACITY_EXHAUSTED),
                            jnp.where(
                                live_weight > prepared.weight_tolerance,
                                int(TissueTransportStatus.INTERACTION_CAPACITY_EXHAUSTED),
                                jnp.where(
                                    ~finite,
                                    int(TissueTransportStatus.NONFINITE_RESULT),
                                    int(TissueTransportStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(TissueTransportStatus.SUCCESS)
    return (
        absorption,
        surface_flux,
        detector,
        escape,
        roulette,
        live_weight,
        truncated,
        initial_weight,
        ledger_residual,
        positions,
        directions,
        weights,
        media,
        optical_depths,
        live,
        interaction_count,
        status,
        successful,
        finite,
    )


def _standard_error(values: Array) -> Array:
    count = values.shape[0]
    mean = jnp.mean(values, axis=0)
    squared = jnp.sum((values - mean) ** 2, axis=0)
    return jnp.where(count > 1, jnp.sqrt(squared / (count * (count - 1))), 0.0)


def _tallies_from_values(
    values: tuple[Array, ...], reduction: str
) -> TissueTransportTallies:
    fields = values[:9]
    if reduction == "none":
        reduced = fields
    elif reduction == "mean":
        reduced = tuple(jnp.mean(value, axis=0) for value in fields)
    else:
        reduced = tuple(_standard_error(value) for value in fields)
    return TissueTransportTallies(*reduced)


def simulate_tissue_transport(
    prepared: PreparedTissueTransport,
    origins: ArrayLike,
    directions: ArrayLike,
    medium_indices: ArrayLike,
    key: Array,
    /,
    *,
    photon_ids: ArrayLike | None = None,
    initial_weights: ArrayLike = 1.0,
) -> TissueTransportResult:
    """Simulate keyed photon packets with fixed work and explicit uncertainty.

    Randomness is folded by caller-visible photon ID, interaction index, semantic
    branch ID, and draw purpose. Therefore the same photon IDs are invariant to
    batching and lane placement. This stochastic solver makes no pathwise
    differentiability claim.
    """

    origins_ = jnp.asarray(origins, dtype=prepared.coefficients.mu_a.dtype)
    directions_ = jnp.asarray(directions, dtype=prepared.coefficients.mu_a.dtype)
    if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
        raise ValueError("origins and directions must have matching shape B + (3,).")
    batch_shape = origins_.shape[:-1]
    sample_count = math.prod(batch_shape) if batch_shape else 1
    media_ = jnp.broadcast_to(jnp.asarray(medium_indices, dtype=jnp.int32), batch_shape)
    weights_ = jnp.broadcast_to(
        jnp.asarray(initial_weights, dtype=origins_.dtype), batch_shape
    )
    if photon_ids is None:
        photon_ids_ = jnp.arange(sample_count, dtype=jnp.uint32).reshape(batch_shape)
    else:
        photon_ids_ = jnp.broadcast_to(
            jnp.asarray(photon_ids, dtype=jnp.uint32), batch_shape
        )
    key_ = jnp.asarray(key, dtype=jnp.uint32)
    if key_.shape != (2,):
        raise ValueError("key must be one JAX PRNG key with shape (2,).")

    def simulate_one(inputs):
        photon_id, origin, direction, medium, weight = inputs
        return _transport_one(
            prepared, key_, photon_id, origin, direction, medium, weight
        )

    values = jax.lax.map(
        simulate_one,
        (
            photon_ids_.reshape((sample_count,)),
            origins_.reshape((sample_count, 3)),
            directions_.reshape((sample_count, 3)),
            media_.reshape((sample_count,)),
            weights_.reshape((sample_count,)),
        ),
    )
    per_photon_flat = _tallies_from_values(values, "none")
    per_photon = TissueTransportTallies(
        per_photon_flat.absorption.reshape(
            batch_shape + (prepared.coefficients.medium_count,)
        ),
        per_photon_flat.surface_flux.reshape(
            batch_shape + (prepared.surfaces.surface_count,)
        ),
        per_photon_flat.detector.reshape(
            batch_shape + (prepared.surfaces.detector_count,)
        ),
        per_photon_flat.escape.reshape(batch_shape),
        per_photon_flat.roulette.reshape(batch_shape),
        per_photon_flat.live.reshape(batch_shape),
        per_photon_flat.truncated.reshape(batch_shape),
        per_photon_flat.launched.reshape(batch_shape),
        per_photon_flat.ledger_residual.reshape(batch_shape),
    )
    tallies = _tallies_from_values(values, "mean")
    standard_errors = _tallies_from_values(values, "standard-error")
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        ledger,
        positions,
        terminal_directions,
        terminal_weights,
        terminal_media,
        terminal_depths,
        terminal_live,
        interactions,
        status,
        successful,
        finite,
    ) = values
    branch_shape = batch_shape + (prepared.branch_capacity,)
    return TissueTransportResult(
        per_photon,
        tallies,
        standard_errors,
        positions.reshape(branch_shape + (3,)),
        terminal_directions.reshape(branch_shape + (3,)),
        terminal_weights.reshape(branch_shape),
        terminal_media.reshape(branch_shape),
        terminal_depths.reshape(branch_shape),
        terminal_live.reshape(branch_shape),
        interactions.reshape(batch_shape),
        status.reshape(batch_shape),
        successful.reshape(batch_shape),
        jnp.asarray(sample_count, dtype=jnp.int32),
        jnp.max(jnp.abs(ledger)),
        jnp.all(finite),
        jnp.all(successful),
    )


__all__ = [
    "PreparedTissueTransport",
    "TissueTransportCoefficients",
    "TissueTransportPlan",
    "TissueTransportResult",
    "TissueTransportStatus",
    "TissueTransportTallies",
    "prepare_tissue_transport",
    "simulate_tissue_transport",
]
