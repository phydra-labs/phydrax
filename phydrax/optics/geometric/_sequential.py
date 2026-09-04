#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from numbers import Integral, Real
from typing import Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.analytic import RigidFrame
from ._interface import evaluate_refractive_interface, OpticalRayState


SurfaceKind = Literal["plane", "sphere", "conic", "even-asphere"]
SurfaceInteraction = Literal["transmit", "reflect"]

_KIND_TAGS: dict[str, int] = {
    "plane": 0,
    "sphere": 1,
    "conic": 2,
    "even-asphere": 3,
}
_INTERACTION_TAGS: dict[str, int] = {"transmit": 0, "reflect": 1}


class SequentialOpticsStatus(IntEnum):
    """Terminal status for one lane of a fixed sequential prescription."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DIRECTION = 2
    INVALID_NORMAL = 3
    COPLANAR = 4
    PARALLEL = 5
    BEHIND_RAY = 6
    WRONG_SIDE_INCIDENCE = 7
    MISSED_SURFACE = 8
    APERTURE_CLIPPED = 9
    ROOT_NONCONVERGENCE = 10
    TOTAL_INTERNAL_REFLECTION = 11
    INVALID_SAG_DOMAIN = 12
    TANGENT_SURFACE = 13
    NUMERICAL_FAILURE = 14


class SequentialOpticsResult(StrictModule):
    """Compact final rays and explicit trace evidence without path history."""

    rays: OpticalRayState
    valid: Array
    status: Array
    traversed_surfaces: Array
    minimum_snell_discriminant: Array
    minimum_aperture_margin: Array
    maximum_intersection_residual: Array
    finite: Array
    successful: Array
    producer_id: str = eqx.field(static=True)


class _SurfaceIntersection(StrictModule):
    distance: Array
    local_point: Array
    local_normal: Array
    residual: Array
    aperture_margin: Array
    grazing_margin: Array
    root_switch_margin: Array
    valid: Array
    status: Array


def _host_real_vector(values: ArrayLike, size: int, name: str, /) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.dtype == np.dtype(bool)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must be real numeric data.")
    result = raw.astype(float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real array with shape ({size},).")
    return result


def _frame_fingerprint(frame: RigidFrame, /) -> dict[str, object]:
    return array_tree_fingerprint(
        (np.asarray(frame.rotation), np.asarray(frame.translation))
    )


def _sag_and_radial_derivative(
    radial_squared: Array,
    curvature: Array,
    conic_constant: Array,
    coefficients: Array,
    coefficient_active: Array,
    /,
) -> tuple[Array, Array, Array]:
    radial_squared = jnp.maximum(radial_squared, 0.0)
    radicand = 1.0 - (1.0 + conic_constant) * curvature**2 * radial_squared
    domain_valid = radicand >= 0.0
    root = jnp.sqrt(jnp.maximum(radicand, 0.0))
    denominator = 1.0 + root
    base = curvature * radial_squared / denominator
    safe_root = jnp.maximum(root, jnp.sqrt(jnp.finfo(radial_squared.dtype).eps))
    base_derivative = curvature / (2.0 * safe_root)

    coefficient_count = coefficients.shape[0]
    powers = jnp.arange(coefficient_count, dtype=radial_squared.dtype) + 2.0
    active_coefficients = jnp.where(coefficient_active, coefficients, 0.0)
    polynomial = jnp.sum(
        active_coefficients * radial_squared[..., None] ** powers,
        axis=-1,
    )
    polynomial_derivative = jnp.sum(
        active_coefficients * powers * radial_squared[..., None] ** (powers - 1.0),
        axis=-1,
    )
    sag = base + polynomial
    derivative = base_derivative + polynomial_derivative
    finite = jnp.isfinite(sag) & jnp.isfinite(derivative)
    return sag, derivative, domain_valid & finite


def _surface_values(
    local_origin: Array,
    local_direction: Array,
    distance: Array,
    curvature: Array,
    conic_constant: Array,
    coefficients: Array,
    coefficient_active: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    point = local_origin + distance[..., None] * local_direction
    radial_squared = jnp.sum(point[..., :2] ** 2, axis=-1)
    sag, radial_derivative, domain_valid = _sag_and_radial_derivative(
        radial_squared,
        curvature,
        conic_constant,
        coefficients,
        coefficient_active,
    )
    residual = point[..., 2] - sag
    directional_derivative = local_direction[..., 2] - 2.0 * radial_derivative * jnp.sum(
        point[..., :2] * local_direction[..., :2], axis=-1
    )
    normal = jnp.concatenate(
        (
            -2.0 * radial_derivative[..., None] * point[..., :2],
            jnp.ones_like(point[..., 2:3]),
        ),
        axis=-1,
    )
    normal_norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    normal = normal / jnp.where(normal_norm > 0.0, normal_norm, 1.0)[..., None]
    return residual, directional_derivative, point, normal, domain_valid


def _plane_intersection(
    local_origin: Array,
    local_direction: Array,
    maximum_distance: Array,
    aperture_radius: Array,
    aperture_active: Array,
    *,
    forward_tolerance: float,
    incidence_tolerance: float,
    intersection_tolerance: float,
) -> _SurfaceIntersection:
    denominator = local_direction[..., 2]
    height = local_origin[..., 2]
    parallel = jnp.abs(denominator) <= incidence_tolerance
    coplanar = parallel & (jnp.abs(height) <= intersection_tolerance)
    safe_denominator = jnp.where(parallel, 1.0, denominator)
    distance = -height / safe_denominator
    forward = (distance > forward_tolerance) & (distance <= maximum_distance)
    point = local_origin + distance[..., None] * local_direction
    radial_distance = jnp.sqrt(jnp.sum(point[..., :2] ** 2, axis=-1))
    aperture_margin = jnp.where(
        aperture_active, aperture_radius - radial_distance, jnp.inf
    )
    aperture_valid = (~aperture_active) | (aperture_margin >= 0.0)
    finite = (
        jnp.all(jnp.isfinite(point), axis=-1)
        & jnp.isfinite(distance)
        & jnp.isfinite(radial_distance)
    )
    hit_valid = (~parallel) & forward & aperture_valid & finite
    status = jnp.where(
        ~finite,
        int(SequentialOpticsStatus.NUMERICAL_FAILURE),
        jnp.where(
            coplanar,
            int(SequentialOpticsStatus.COPLANAR),
            jnp.where(
                parallel,
                int(SequentialOpticsStatus.PARALLEL),
                jnp.where(
                    distance <= forward_tolerance,
                    int(SequentialOpticsStatus.BEHIND_RAY),
                    jnp.where(
                        distance > maximum_distance,
                        int(SequentialOpticsStatus.MISSED_SURFACE),
                        jnp.where(
                            ~aperture_valid,
                            int(SequentialOpticsStatus.APERTURE_CLIPPED),
                            int(SequentialOpticsStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    normal = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, 1.0), dtype=local_origin.dtype),
        local_origin.shape,
    )
    return _SurfaceIntersection(
        distance=distance,
        local_point=point,
        local_normal=normal,
        residual=jnp.abs(point[..., 2]),
        aperture_margin=aperture_margin,
        grazing_margin=jnp.abs(denominator) - incidence_tolerance,
        root_switch_margin=jnp.full_like(distance, jnp.inf),
        valid=hit_valid,
        status=status,
    )


def _sphere_intersection(
    local_origin: Array,
    local_direction: Array,
    curvature: Array,
    maximum_distance: Array,
    aperture_radius: Array,
    aperture_active: Array,
    *,
    forward_tolerance: float,
    incidence_tolerance: float,
    intersection_tolerance: float,
) -> _SurfaceIntersection:
    inverse_curvature = 1.0 / curvature
    shifted = local_origin.at[..., 2].add(-inverse_curvature)
    quadratic = jnp.sum(local_direction * local_direction, axis=-1)
    linear = 2.0 * jnp.sum(shifted * local_direction, axis=-1)
    constant = jnp.sum(shifted * shifted, axis=-1) - inverse_curvature**2
    discriminant = linear**2 - 4.0 * quadratic * constant
    tangent = jnp.abs(discriminant) <= intersection_tolerance
    real_roots = discriminant >= 0.0
    root_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = 2.0 * quadratic
    first_distance = (-linear - root_discriminant) / denominator
    second_distance = (-linear + root_discriminant) / denominator

    def candidate(distance: Array) -> tuple[Array, Array, Array, Array, Array]:
        point = local_origin + distance[..., None] * local_direction
        radial_squared = jnp.sum(point[..., :2] ** 2, axis=-1)
        radicand = 1.0 - curvature**2 * radial_squared
        domain = radicand >= 0.0
        root = jnp.sqrt(jnp.maximum(radicand, 0.0))
        sag = curvature * radial_squared / (1.0 + root)
        residual = jnp.abs(point[..., 2] - sag)
        branch = residual <= 16.0 * intersection_tolerance
        forward = (distance > forward_tolerance) & (distance <= maximum_distance)
        return point, residual, domain, branch, forward

    first_point, first_residual, first_domain, first_branch, first_forward = candidate(
        first_distance
    )
    second_point, second_residual, second_domain, second_branch, second_forward = (
        candidate(second_distance)
    )
    first_valid = real_roots & first_domain & first_branch & first_forward
    second_valid = real_roots & second_domain & second_branch & second_forward
    use_first = first_valid & (~second_valid | (first_distance <= second_distance))
    use_second = second_valid & ~use_first
    any_root = first_valid | second_valid
    distance = jnp.where(use_first, first_distance, second_distance)
    point = jnp.where(use_first[..., None], first_point, second_point)
    residual = jnp.where(use_first, first_residual, second_residual)
    radial_squared = jnp.sum(point[..., :2] ** 2, axis=-1)
    root = jnp.sqrt(jnp.maximum(1.0 - curvature**2 * radial_squared, 0.0))
    raw_normal = jnp.concatenate(
        (
            -curvature * point[..., :2],
            root[..., None],
        ),
        axis=-1,
    )
    normal_norm = jnp.sqrt(jnp.sum(raw_normal * raw_normal, axis=-1))
    normal = raw_normal / jnp.where(normal_norm > 0.0, normal_norm, 1.0)[..., None]
    incidence = jnp.abs(jnp.sum(local_direction * normal, axis=-1))
    radial_distance = jnp.sqrt(radial_squared)
    aperture_margin = jnp.where(
        aperture_active, aperture_radius - radial_distance, jnp.inf
    )
    aperture_valid = (~aperture_active) | (aperture_margin >= 0.0)
    finite = (
        jnp.isfinite(distance)
        & jnp.all(jnp.isfinite(point), axis=-1)
        & jnp.all(jnp.isfinite(normal), axis=-1)
        & jnp.isfinite(residual)
    )
    valid = any_root & (~tangent) & aperture_valid & finite
    all_behind = (first_distance <= forward_tolerance) & (
        second_distance <= forward_tolerance
    )
    any_domain = first_domain | second_domain
    status = jnp.where(
        tangent & real_roots,
        int(SequentialOpticsStatus.TANGENT_SURFACE),
        jnp.where(
            ~finite,
            int(SequentialOpticsStatus.NUMERICAL_FAILURE),
            jnp.where(
                ~real_roots,
                int(SequentialOpticsStatus.MISSED_SURFACE),
                jnp.where(
                    all_behind,
                    int(SequentialOpticsStatus.BEHIND_RAY),
                    jnp.where(
                        ~any_domain,
                        int(SequentialOpticsStatus.INVALID_SAG_DOMAIN),
                        jnp.where(
                            ~any_root,
                            int(SequentialOpticsStatus.MISSED_SURFACE),
                            jnp.where(
                                ~aperture_valid,
                                int(SequentialOpticsStatus.APERTURE_CLIPPED),
                                int(SequentialOpticsStatus.SUCCESS),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    second_forward_root = jnp.where(
        use_first & second_valid,
        second_distance,
        jnp.where(use_second & first_valid, first_distance, jnp.inf),
    )
    root_switch_margin = jnp.abs(second_forward_root - distance)
    return _SurfaceIntersection(
        distance=distance,
        local_point=point,
        local_normal=normal,
        residual=residual,
        aperture_margin=aperture_margin,
        grazing_margin=incidence - incidence_tolerance,
        root_switch_margin=root_switch_margin,
        valid=valid,
        status=status,
    )


def _bounded_sag_intersection(
    local_origin: Array,
    local_direction: Array,
    curvature: Array,
    conic_constant: Array,
    coefficients: Array,
    coefficient_active: Array,
    maximum_distance: Array,
    aperture_radius: Array,
    aperture_active: Array,
    *,
    bracket_sample_count: int,
    root_iteration_count: int,
    forward_tolerance: float,
    incidence_tolerance: float,
    intersection_tolerance: float,
) -> _SurfaceIntersection:
    dtype = local_origin.dtype
    unit_samples = jnp.linspace(0.0, 1.0, bracket_sample_count + 1, dtype=dtype)
    first_distance = jnp.asarray(forward_tolerance, dtype=dtype)
    distances = first_distance + unit_samples * (maximum_distance - first_distance)
    sample_shape = (bracket_sample_count + 1,) + (1,) * (local_origin.ndim - 1)
    sampled_distance = distances.reshape(sample_shape)
    sampled_origin = local_origin[None, ...]
    sampled_direction = local_direction[None, ...]
    residuals, derivatives, _, _, domains = _surface_values(
        sampled_origin,
        sampled_direction,
        sampled_distance,
        curvature,
        conic_constant,
        coefficients,
        coefficient_active,
    )
    left_residual = residuals[:-1]
    right_residual = residuals[1:]
    bracketed = (
        domains[:-1]
        & domains[1:]
        & (
            (left_residual * right_residual <= 0.0)
            | (jnp.abs(left_residual) <= intersection_tolerance)
            | (jnp.abs(right_residual) <= intersection_tolerance)
        )
    )
    has_bracket = jnp.any(bracketed, axis=0)
    first_bracket = jnp.argmax(bracketed.astype(jnp.int32), axis=0)
    gather_index = first_bracket[None, ...]
    distance_grid = jnp.broadcast_to(distances.reshape(sample_shape), residuals.shape)
    lower = jnp.take_along_axis(distance_grid[:-1], gather_index, axis=0)[0]
    upper = jnp.take_along_axis(distance_grid[1:], gather_index, axis=0)[0]
    lower_residual = jnp.take_along_axis(left_residual, gather_index, axis=0)[0]
    upper_residual = jnp.take_along_axis(right_residual, gather_index, axis=0)[0]
    stationary_bracketed = (
        domains[:-1] & domains[1:] & (derivatives[:-1] * derivatives[1:] <= 0.0)
    )
    has_stationary = jnp.any(stationary_bracketed, axis=0)
    first_stationary = jnp.argmax(stationary_bracketed.astype(jnp.int32), axis=0)
    stationary_index = first_stationary[None, ...]
    stationary_lower = jnp.take_along_axis(distance_grid[:-1], stationary_index, axis=0)[
        0
    ]
    stationary_upper = jnp.take_along_axis(distance_grid[1:], stationary_index, axis=0)[0]
    stationary_lower_derivative = jnp.take_along_axis(
        derivatives[:-1], stationary_index, axis=0
    )[0]
    stationary_upper_derivative = jnp.take_along_axis(
        derivatives[1:], stationary_index, axis=0
    )[0]

    def refine(carry, _):
        low, high, flow, fhigh = carry
        midpoint = 0.5 * (low + high)
        fmid, dfmid, _, _, midpoint_domain = _surface_values(
            local_origin,
            local_direction,
            midpoint,
            curvature,
            conic_constant,
            coefficients,
            coefficient_active,
        )
        safe_derivative = jnp.where(jnp.abs(dfmid) > incidence_tolerance, dfmid, 1.0)
        newton = midpoint - fmid / safe_derivative
        use_newton = (
            midpoint_domain
            & jnp.isfinite(newton)
            & (newton > low)
            & (newton < high)
            & (jnp.abs(dfmid) > incidence_tolerance)
        )
        candidate = jnp.where(use_newton, newton, midpoint)
        fcandidate, _, _, _, candidate_domain = _surface_values(
            local_origin,
            local_direction,
            candidate,
            curvature,
            conic_constant,
            coefficients,
            coefficient_active,
        )
        candidate = jnp.where(candidate_domain, candidate, midpoint)
        fcandidate = jnp.where(candidate_domain, fcandidate, fmid)
        same_side = jnp.signbit(flow) == jnp.signbit(fcandidate)
        exact_low = jnp.abs(flow) <= intersection_tolerance
        exact_high = jnp.abs(fhigh) <= intersection_tolerance
        frozen = exact_low | exact_high
        next_low = jnp.where(frozen, low, jnp.where(same_side, candidate, low))
        next_high = jnp.where(frozen, high, jnp.where(same_side, high, candidate))
        next_flow = jnp.where(frozen, flow, jnp.where(same_side, fcandidate, flow))
        next_fhigh = jnp.where(frozen, fhigh, jnp.where(same_side, fhigh, fcandidate))
        return (next_low, next_high, next_flow, next_fhigh), None

    (lower, upper, lower_residual, upper_residual), _ = jax.lax.scan(
        refine,
        (lower, upper, lower_residual, upper_residual),
        xs=None,
        length=root_iteration_count,
    )

    def refine_stationary(carry, _):
        low, high, dlow, dhigh = carry
        midpoint = 0.5 * (low + high)
        _, derivative_midpoint, _, _, midpoint_domain = _surface_values(
            local_origin,
            local_direction,
            midpoint,
            curvature,
            conic_constant,
            coefficients,
            coefficient_active,
        )
        same_side = jnp.signbit(dlow) == jnp.signbit(derivative_midpoint)
        frozen = (
            (jnp.abs(dlow) <= incidence_tolerance)
            | (jnp.abs(dhigh) <= incidence_tolerance)
            | ~midpoint_domain
        )
        next_low = jnp.where(frozen, low, jnp.where(same_side, midpoint, low))
        next_high = jnp.where(frozen, high, jnp.where(same_side, high, midpoint))
        next_dlow = jnp.where(
            frozen, dlow, jnp.where(same_side, derivative_midpoint, dlow)
        )
        next_dhigh = jnp.where(
            frozen, dhigh, jnp.where(same_side, dhigh, derivative_midpoint)
        )
        return (next_low, next_high, next_dlow, next_dhigh), None

    (
        (
            stationary_lower,
            stationary_upper,
            stationary_lower_derivative,
            stationary_upper_derivative,
        ),
        _,
    ) = jax.lax.scan(
        refine_stationary,
        (
            stationary_lower,
            stationary_upper,
            stationary_lower_derivative,
            stationary_upper_derivative,
        ),
        xs=None,
        length=root_iteration_count,
    )
    stationary_distance = jnp.where(
        jnp.abs(stationary_lower_derivative) <= jnp.abs(stationary_upper_derivative),
        stationary_lower,
        stationary_upper,
    )
    (
        stationary_residual,
        stationary_derivative,
        _,
        _,
        stationary_domain,
    ) = _surface_values(
        local_origin,
        local_direction,
        stationary_distance,
        curvature,
        conic_constant,
        coefficients,
        coefficient_active,
    )
    stationary_tangent = (
        ~has_bracket
        & has_stationary
        & (stationary_distance > forward_tolerance)
        & stationary_domain
        & (jnp.abs(stationary_residual) <= intersection_tolerance)
        & (jnp.abs(stationary_derivative) <= incidence_tolerance)
    )
    choose_lower = jnp.abs(lower_residual) <= jnp.abs(upper_residual)
    distance = jnp.where(choose_lower, lower, upper)
    residual_value, derivative, point, normal, domain_valid = _surface_values(
        local_origin,
        local_direction,
        distance,
        curvature,
        conic_constant,
        coefficients,
        coefficient_active,
    )
    residual = jnp.abs(residual_value)
    forward_hit = distance > forward_tolerance
    converged = (
        has_bracket & domain_valid & forward_hit & (residual <= intersection_tolerance)
    )
    tangent = converged & (jnp.abs(derivative) <= incidence_tolerance)
    radial_distance = jnp.sqrt(jnp.sum(point[..., :2] ** 2, axis=-1))
    aperture_margin = jnp.where(
        aperture_active, aperture_radius - radial_distance, jnp.inf
    )
    aperture_valid = (~aperture_active) | (aperture_margin >= 0.0)
    finite = (
        jnp.isfinite(distance)
        & jnp.isfinite(residual)
        & jnp.all(jnp.isfinite(point), axis=-1)
        & jnp.all(jnp.isfinite(normal), axis=-1)
    )
    valid = converged & (~tangent) & aperture_valid & finite
    all_domain = jnp.all(domains, axis=0)
    closest_index = jnp.argmin(jnp.where(domains, jnp.abs(residuals), jnp.inf), axis=0)
    closest_derivative = jnp.take_along_axis(
        derivatives, closest_index[None, ...], axis=0
    )[0]
    closest_distance = jnp.take_along_axis(
        distance_grid, closest_index[None, ...], axis=0
    )[0]
    sampled_tangent = (
        ~has_bracket
        & jnp.any(domains, axis=0)
        & (closest_distance > forward_tolerance)
        & (
            jnp.min(jnp.where(domains, jnp.abs(residuals), jnp.inf), axis=0)
            <= intersection_tolerance
        )
        & (jnp.abs(closest_derivative) <= incidence_tolerance)
    )
    status = jnp.where(
        tangent | sampled_tangent | stationary_tangent,
        int(SequentialOpticsStatus.TANGENT_SURFACE),
        jnp.where(
            ~finite,
            int(SequentialOpticsStatus.NUMERICAL_FAILURE),
            jnp.where(
                has_bracket & ~forward_hit,
                int(SequentialOpticsStatus.BEHIND_RAY),
                jnp.where(
                    ~has_bracket & ~all_domain,
                    int(SequentialOpticsStatus.INVALID_SAG_DOMAIN),
                    jnp.where(
                        ~has_bracket,
                        int(SequentialOpticsStatus.MISSED_SURFACE),
                        jnp.where(
                            ~converged,
                            int(SequentialOpticsStatus.ROOT_NONCONVERGENCE),
                            jnp.where(
                                ~aperture_valid,
                                int(SequentialOpticsStatus.APERTURE_CLIPPED),
                                int(SequentialOpticsStatus.SUCCESS),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    bracket_start = distances[:-1].reshape(
        (bracket_sample_count,) + (1,) * (local_origin.ndim - 1)
    )
    later_brackets = bracketed & (
        jnp.arange(bracket_sample_count).reshape(
            (bracket_sample_count,) + (1,) * (local_origin.ndim - 1)
        )
        > first_bracket[None, ...] + 1
    )
    has_second = jnp.any(later_brackets, axis=0)
    second_index = jnp.argmax(later_brackets.astype(jnp.int32), axis=0)
    second_start = jnp.take_along_axis(
        jnp.broadcast_to(bracket_start, bracketed.shape),
        second_index[None, ...],
        axis=0,
    )[0]
    root_switch_margin = jnp.where(
        has_second, jnp.maximum(second_start - distance, 0.0), jnp.inf
    )
    return _SurfaceIntersection(
        distance=distance,
        local_point=point,
        local_normal=normal,
        residual=residual,
        aperture_margin=aperture_margin,
        grazing_margin=jnp.abs(derivative) - incidence_tolerance,
        root_switch_margin=root_switch_margin,
        valid=valid,
        status=status,
    )


class SequentialOpticsPlan(StrictModule, NonTrainableState):
    """Immutable fixed-layout prescription for one declared optical route.

    In each local frame the vertex-connected sag is
    ``w = c*r**2/(1 + sqrt(1 - (1 + k)*c**2*r**2)) + sum(a_j*r**(4+2*j))``.

    Even-asphere coefficient column ``j`` multiplies ``r**(4 + 2*j)``. For a
    caller-chosen length unit ``L``, curvature has units ``L**-1`` and column
    ``j`` has units ``L**(-3 - 2*j)``; every frame and length uses that same unit.
    """

    frames: tuple[RigidFrame, ...]
    surface_kinds: tuple[SurfaceKind, ...] = eqx.field(static=True)
    interactions: tuple[SurfaceInteraction, ...] = eqx.field(static=True)
    curvatures: Array
    conic_constants: Array
    even_asphere_coefficients: Array
    coefficient_active: Array
    clear_semi_diameters: Array
    aperture_active: Array
    maximum_intersection_distances: Array
    refractive_indices: Array
    surface_count: int = eqx.field(static=True)
    coefficient_capacity: int = eqx.field(static=True)
    bracket_sample_count: int = eqx.field(static=True)
    root_iteration_count: int = eqx.field(static=True)
    intersection_tolerance: float = eqx.field(static=True)
    forward_tolerance: float = eqx.field(static=True)
    incidence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        frames: Sequence[RigidFrame],
        surface_kinds: Sequence[SurfaceKind],
        interactions: Sequence[SurfaceInteraction],
        curvatures: ArrayLike,
        conic_constants: ArrayLike,
        even_asphere_coefficients: ArrayLike,
        coefficient_active: ArrayLike,
        clear_semi_diameters: ArrayLike,
        aperture_active: ArrayLike,
        maximum_intersection_distances: ArrayLike,
        refractive_indices: ArrayLike,
        *,
        bracket_sample_count: int = 32,
        root_iteration_count: int = 32,
        intersection_tolerance: float = 1.0e-9,
        forward_tolerance: float = 1.0e-10,
        incidence_tolerance: float = 1.0e-10,
    ):
        frame_tuple = tuple(frames)
        kinds = tuple(surface_kinds)
        routes = tuple(interactions)
        surface_count = len(frame_tuple)
        if surface_count == 0:
            raise ValueError(
                "A sequential prescription must contain at least one surface."
            )
        if len(kinds) != surface_count or len(routes) != surface_count:
            raise ValueError(
                "frames, surface_kinds, and interactions must have equal length."
            )
        if any(not isinstance(frame, RigidFrame) for frame in frame_tuple):
            raise TypeError("Every surface frame must be a RigidFrame.")
        if any(frame.dimension != 3 for frame in frame_tuple):
            raise ValueError(
                "Sequential optical surface frames must be three-dimensional."
            )
        unknown_kinds = set(kinds).difference(_KIND_TAGS)
        unknown_routes = set(routes).difference(_INTERACTION_TAGS)
        if unknown_kinds or unknown_routes:
            raise ValueError(
                "Every surface kind and interaction must be supported explicitly."
            )

        curvature = _host_real_vector(curvatures, surface_count, "curvatures")
        conic = _host_real_vector(conic_constants, surface_count, "conic_constants")
        coefficient_values = np.asarray(even_asphere_coefficients)
        if (
            coefficient_values.dtype == np.dtype(bool)
            or not np.issubdtype(coefficient_values.dtype, np.number)
            or np.issubdtype(coefficient_values.dtype, np.complexfloating)
        ):
            raise TypeError("even_asphere_coefficients must be real numeric data.")
        coefficients = coefficient_values.astype(float)
        active_coefficients = np.asarray(coefficient_active)
        if active_coefficients.dtype != np.dtype(bool):
            raise TypeError("coefficient_active must contain booleans.")
        if coefficients.ndim != 2 or coefficients.shape[0] != surface_count:
            raise ValueError(
                "even_asphere_coefficients must have shape (surface_count, A)."
            )
        if active_coefficients.shape != coefficients.shape:
            raise ValueError("coefficient_active must match even_asphere_coefficients.")
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("even-asphere coefficients must be finite.")
        aperture_radius = _host_real_vector(
            clear_semi_diameters, surface_count, "clear_semi_diameters"
        )
        active_aperture = np.asarray(aperture_active)
        if active_aperture.dtype != np.dtype(bool):
            raise TypeError("aperture_active must contain booleans.")
        if active_aperture.shape != (surface_count,):
            raise ValueError(f"aperture_active must have shape ({surface_count},).")
        maximum_distance = _host_real_vector(
            maximum_intersection_distances,
            surface_count,
            "maximum_intersection_distances",
        )
        indices = _host_real_vector(
            refractive_indices, surface_count + 1, "refractive_indices"
        )
        if (
            not isinstance(bracket_sample_count, Integral)
            or isinstance(bracket_sample_count, (bool, np.bool_))
            or not isinstance(root_iteration_count, Integral)
            or isinstance(root_iteration_count, (bool, np.bool_))
        ):
            raise TypeError("Root work counts must be integers.")
        if any(
            not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            for value in (
                intersection_tolerance,
                forward_tolerance,
                incidence_tolerance,
            )
        ):
            raise TypeError("Intersection tolerances must be real scalars.")
        brackets = int(bracket_sample_count)
        iterations = int(root_iteration_count)
        root_tolerance = float(intersection_tolerance)
        forward = float(forward_tolerance)
        incidence = float(incidence_tolerance)
        if (
            brackets < 2
            or iterations <= 0
            or not np.isfinite(root_tolerance)
            or not np.isfinite(forward)
            or not np.isfinite(incidence)
            or root_tolerance <= 0.0
            or forward < 0.0
            or incidence <= 0.0
        ):
            raise ValueError(
                "Root and intersection controls must be finite and positive."
            )
        if np.any(maximum_distance <= forward):
            raise ValueError(
                "Each maximum intersection distance must exceed forward_tolerance."
            )
        if np.any(indices <= 0.0):
            raise ValueError("Refractive indices must be positive real values.")
        if np.any(active_aperture & (aperture_radius <= 0.0)) or np.any(
            (~active_aperture) & (aperture_radius != 0.0)
        ):
            raise ValueError(
                "Only active circular apertures may have positive semi-diameter."
            )
        if np.any((~active_coefficients) & (coefficients != 0.0)):
            raise ValueError("Inactive even-asphere coefficients must be exactly zero.")

        for index, (kind, route) in enumerate(zip(kinds, routes, strict=True)):
            if kind == "plane":
                if (
                    curvature[index] != 0.0
                    or conic[index] != 0.0
                    or np.any(coefficients[index] != 0.0)
                    or np.any(active_coefficients[index])
                ):
                    raise ValueError("Plane rows must have exactly neutral sag data.")
            elif kind == "sphere":
                if curvature[index] == 0.0:
                    raise ValueError("Sphere curvature must be nonzero.")
                if (
                    conic[index] != 0.0
                    or np.any(coefficients[index] != 0.0)
                    or np.any(active_coefficients[index])
                ):
                    raise ValueError(
                        "Sphere rows must have exactly neutral conic/asphere data."
                    )
            elif kind == "conic":
                if np.any(coefficients[index] != 0.0) or np.any(
                    active_coefficients[index]
                ):
                    raise ValueError("Conic rows must have exactly neutral asphere data.")
            else:
                if not np.any(active_coefficients[index]):
                    raise ValueError(
                        "Even-asphere rows must declare at least one active coefficient."
                    )
            if active_aperture[index] and kind != "plane":
                radial_domain = (
                    1.0
                    - (1.0 + conic[index])
                    * curvature[index] ** 2
                    * aperture_radius[index] ** 2
                )
                if radial_domain < 0.0:
                    raise ValueError(
                        "The clear aperture leaves the vertex-connected sag domain."
                    )
            if route == "reflect" and indices[index + 1] != indices[index]:
                raise ValueError(
                    "A reflective interaction must retain the incident refractive index."
                )

        self.frames = frame_tuple
        self.surface_kinds = kinds
        self.interactions = routes
        self.curvatures = jnp.asarray(curvature)
        self.conic_constants = jnp.asarray(conic)
        self.even_asphere_coefficients = jnp.asarray(coefficients)
        self.coefficient_active = jnp.asarray(active_coefficients)
        self.clear_semi_diameters = jnp.asarray(aperture_radius)
        self.aperture_active = jnp.asarray(active_aperture)
        self.maximum_intersection_distances = jnp.asarray(maximum_distance)
        self.refractive_indices = jnp.asarray(indices)
        self.surface_count = surface_count
        self.coefficient_capacity = coefficients.shape[1]
        self.bracket_sample_count = brackets
        self.root_iteration_count = iterations
        self.intersection_tolerance = root_tolerance
        self.forward_tolerance = forward
        self.incidence_tolerance = incidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sequential-optics-plan",
                "frames": [_frame_fingerprint(frame) for frame in frame_tuple],
                "surface_kinds": kinds,
                "interactions": routes,
                "arrays": array_tree_fingerprint(
                    (
                        curvature,
                        conic,
                        coefficients,
                        active_coefficients,
                        aperture_radius,
                        active_aperture,
                        maximum_distance,
                        indices,
                    )
                ),
                "bracket_sample_count": brackets,
                "root_iteration_count": iterations,
                "intersection_tolerance": root_tolerance,
                "forward_tolerance": forward,
                "incidence_tolerance": incidence,
            }
        )

    def prepare(self, /) -> "PreparedSequentialOptics":
        """Validate and lower the prescription into a fixed-shape executable."""
        return PreparedSequentialOptics(self)


class PreparedSequentialOptics(StrictModule, NonTrainableState):
    """Prepared fixed-route executable with bounded intersection work."""

    __hash__ = object.__hash__

    rotations: Array
    translations: Array
    kind_tags: tuple[int, ...] = eqx.field(static=True)
    interaction_tags: tuple[int, ...] = eqx.field(static=True)
    curvatures: Array
    conic_constants: Array
    even_asphere_coefficients: Array
    coefficient_active: Array
    clear_semi_diameters: Array
    aperture_active: Array
    maximum_intersection_distances: Array
    refractive_indices: Array
    sag_domain_margins: Array
    surface_count: int = eqx.field(static=True)
    coefficient_capacity: int = eqx.field(static=True)
    bracket_sample_count: int = eqx.field(static=True)
    root_iteration_count: int = eqx.field(static=True)
    intersection_tolerance: float = eqx.field(static=True)
    forward_tolerance: float = eqx.field(static=True)
    incidence_tolerance: float = eqx.field(static=True)
    worst_case_surface_evaluations: int = eqx.field(static=True)
    worst_case_root_evaluations: int = eqx.field(static=True)
    validation_margin: float = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SequentialOpticsPlan, /):
        if not isinstance(plan, SequentialOpticsPlan):
            raise TypeError("plan must be a SequentialOpticsPlan.")
        rotations = np.stack([np.asarray(frame.rotation) for frame in plan.frames])
        translations = np.stack([np.asarray(frame.translation) for frame in plan.frames])
        kind_tags = tuple(_KIND_TAGS[kind] for kind in plan.surface_kinds)
        interaction_tags = tuple(
            _INTERACTION_TAGS[interaction] for interaction in plan.interactions
        )
        if any(tag not in _KIND_TAGS.values() for tag in kind_tags) or any(
            tag not in _INTERACTION_TAGS.values() for tag in interaction_tags
        ):
            raise ValueError("Preparation encountered an unknown surface or route tag.")
        curvature = np.asarray(plan.curvatures)
        conic = np.asarray(plan.conic_constants)
        aperture = np.asarray(plan.clear_semi_diameters)
        aperture_active = np.asarray(plan.aperture_active)
        radicands = 1.0 - (1.0 + conic) * curvature**2 * aperture**2
        sag_margins = np.where(aperture_active, radicands, np.inf)
        validation_margin = float(np.min(sag_margins))
        nonlinear_count = sum(tag >= _KIND_TAGS["conic"] for tag in kind_tags)
        root_evaluations = nonlinear_count * (
            plan.bracket_sample_count + 1 + 3 * plan.root_iteration_count + 2
        )

        self.rotations = jnp.asarray(rotations)
        self.translations = jnp.asarray(translations)
        self.kind_tags = kind_tags
        self.interaction_tags = interaction_tags
        self.curvatures = plan.curvatures
        self.conic_constants = plan.conic_constants
        self.even_asphere_coefficients = plan.even_asphere_coefficients
        self.coefficient_active = plan.coefficient_active
        self.clear_semi_diameters = plan.clear_semi_diameters
        self.aperture_active = plan.aperture_active
        self.maximum_intersection_distances = plan.maximum_intersection_distances
        self.refractive_indices = plan.refractive_indices
        self.sag_domain_margins = jnp.asarray(sag_margins)
        self.surface_count = plan.surface_count
        self.coefficient_capacity = plan.coefficient_capacity
        self.bracket_sample_count = plan.bracket_sample_count
        self.root_iteration_count = plan.root_iteration_count
        self.intersection_tolerance = plan.intersection_tolerance
        self.forward_tolerance = plan.forward_tolerance
        self.incidence_tolerance = plan.incidence_tolerance
        self.worst_case_surface_evaluations = plan.surface_count
        self.worst_case_root_evaluations = root_evaluations
        self.validation_margin = validation_margin
        self.source_plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sequential-optics",
                "source_plan_id": plan.plan_id,
                "kind_tags": kind_tags,
                "interaction_tags": interaction_tags,
                "arrays": array_tree_fingerprint((rotations, translations, sag_margins)),
                "worst_case_surface_evaluations": plan.surface_count,
                "worst_case_root_evaluations": root_evaluations,
            }
        )

    def _trace(
        self, origins: ArrayLike, directions: ArrayLike, /
    ) -> tuple[SequentialOpticsResult, Array]:
        origin_input = jnp.asarray(origins)
        direction_input = jnp.asarray(directions)
        if (
            origin_input.shape != direction_input.shape
            or origin_input.ndim < 1
            or origin_input.shape[-1] != 3
        ):
            raise ValueError("origins and directions must have matching shape B + (3,).")
        if (
            jnp.issubdtype(origin_input.dtype, jnp.complexfloating)
            or jnp.issubdtype(direction_input.dtype, jnp.complexfloating)
            or origin_input.dtype == jnp.dtype(bool)
            or direction_input.dtype == jnp.dtype(bool)
        ):
            raise TypeError("Sequential geometric rays must be real numeric arrays.")
        dtype = jnp.result_type(origin_input.dtype, direction_input.dtype, 0.0)
        raw_origin = origin_input.astype(dtype)
        raw_direction = direction_input.astype(dtype)
        input_finite = jnp.all(jnp.isfinite(raw_origin), axis=-1) & jnp.all(
            jnp.isfinite(raw_direction), axis=-1
        )
        direction_norm = jnp.sqrt(jnp.sum(raw_direction * raw_direction, axis=-1))
        direction_valid = jnp.isfinite(direction_norm) & (
            direction_norm > self.incidence_tolerance
        )
        safe_origin = jnp.where(jnp.isfinite(raw_origin), raw_origin, 0.0)
        safe_direction_placeholder = jnp.broadcast_to(
            jnp.asarray((0.0, 0.0, 1.0), dtype=dtype), raw_direction.shape
        )
        normalized_direction = (
            raw_direction / jnp.where(direction_valid, direction_norm, 1.0)[..., None]
        )
        direction = jnp.where(
            (input_finite & direction_valid)[..., None],
            normalized_direction,
            safe_direction_placeholder,
        )
        active = input_finite & direction_valid
        status = jnp.where(
            ~input_finite,
            int(SequentialOpticsStatus.NONFINITE_INPUT),
            jnp.where(
                ~direction_valid,
                int(SequentialOpticsStatus.INVALID_DIRECTION),
                int(SequentialOpticsStatus.SUCCESS),
            ),
        ).astype(jnp.int32)
        batch_shape = raw_origin.shape[:-1]
        refractive_index = jnp.full(batch_shape, self.refractive_indices[0], dtype=dtype)
        geometric_path = jnp.zeros(batch_shape, dtype=dtype)
        optical_path = jnp.zeros(batch_shape, dtype=dtype)
        traversed = jnp.zeros(batch_shape, dtype=jnp.int32)
        minimum_snell = jnp.full(batch_shape, jnp.inf, dtype=dtype)
        minimum_aperture = jnp.full(batch_shape, jnp.inf, dtype=dtype)
        maximum_residual = jnp.zeros(batch_shape, dtype=dtype)
        branch_margin = jnp.full(batch_shape, jnp.inf, dtype=dtype)
        finite = input_finite
        origin = safe_origin

        for index, (kind_tag, interaction_tag) in enumerate(
            zip(self.kind_tags, self.interaction_tags, strict=True)
        ):
            rotation = self.rotations[index].astype(dtype)
            translation = self.translations[index].astype(dtype)
            local_origin = (origin - translation) @ rotation
            local_direction = direction @ rotation
            maximum_distance = self.maximum_intersection_distances[index].astype(dtype)
            aperture_radius = self.clear_semi_diameters[index].astype(dtype)
            aperture_active = self.aperture_active[index]
            if kind_tag == _KIND_TAGS["plane"]:
                intersection = _plane_intersection(
                    local_origin,
                    local_direction,
                    maximum_distance,
                    aperture_radius,
                    aperture_active,
                    forward_tolerance=self.forward_tolerance,
                    incidence_tolerance=self.incidence_tolerance,
                    intersection_tolerance=self.intersection_tolerance,
                )
            elif kind_tag == _KIND_TAGS["sphere"]:
                intersection = _sphere_intersection(
                    local_origin,
                    local_direction,
                    self.curvatures[index].astype(dtype),
                    maximum_distance,
                    aperture_radius,
                    aperture_active,
                    forward_tolerance=self.forward_tolerance,
                    incidence_tolerance=self.incidence_tolerance,
                    intersection_tolerance=self.intersection_tolerance,
                )
            elif kind_tag in (_KIND_TAGS["conic"], _KIND_TAGS["even-asphere"]):
                intersection = _bounded_sag_intersection(
                    local_origin,
                    local_direction,
                    self.curvatures[index].astype(dtype),
                    self.conic_constants[index].astype(dtype),
                    self.even_asphere_coefficients[index].astype(dtype),
                    self.coefficient_active[index],
                    maximum_distance,
                    aperture_radius,
                    aperture_active,
                    bracket_sample_count=self.bracket_sample_count,
                    root_iteration_count=self.root_iteration_count,
                    forward_tolerance=self.forward_tolerance,
                    incidence_tolerance=self.incidence_tolerance,
                    intersection_tolerance=self.intersection_tolerance,
                )
            else:
                raise RuntimeError("Prepared surface tag is not exhaustive.")

            attempted = active
            hit = attempted & intersection.valid
            status = jnp.where(
                attempted & ~intersection.valid, intersection.status, status
            )
            minimum_aperture = jnp.where(
                attempted,
                jnp.minimum(minimum_aperture, intersection.aperture_margin),
                minimum_aperture,
            )
            maximum_residual = jnp.where(
                attempted,
                jnp.maximum(maximum_residual, intersection.residual),
                maximum_residual,
            )
            world_point = intersection.local_point @ rotation.T + translation
            world_normal = intersection.local_normal @ rotation.T
            interface = evaluate_refractive_interface(
                direction,
                world_normal,
                refractive_index,
                self.refractive_indices[index + 1].astype(dtype),
                incidence_tolerance=self.incidence_tolerance,
            )
            minimum_snell = jnp.where(
                hit,
                jnp.minimum(minimum_snell, interface.snell_discriminant),
                minimum_snell,
            )
            if interaction_tag == _INTERACTION_TAGS["transmit"]:
                selected_direction = interface.transmitted_directions
                interface_valid = interface.transmission_valid
                interface_status = jnp.where(
                    interface.incident_cosine <= self.incidence_tolerance,
                    int(SequentialOpticsStatus.WRONG_SIDE_INCIDENCE),
                    jnp.where(
                        interface.snell_discriminant < 0.0,
                        int(SequentialOpticsStatus.TOTAL_INTERNAL_REFLECTION),
                        int(SequentialOpticsStatus.NUMERICAL_FAILURE),
                    ),
                ).astype(jnp.int32)
            elif interaction_tag == _INTERACTION_TAGS["reflect"]:
                selected_direction = interface.reflected_directions
                interface_valid = interface.reflection_valid
                interface_status = jnp.where(
                    interface.incident_cosine <= self.incidence_tolerance,
                    int(SequentialOpticsStatus.WRONG_SIDE_INCIDENCE),
                    int(SequentialOpticsStatus.NUMERICAL_FAILURE),
                ).astype(jnp.int32)
            else:
                raise RuntimeError("Prepared interaction tag is not exhaustive.")
            selected_finite = jnp.all(jnp.isfinite(selected_direction), axis=-1)
            route_success = hit & interface_valid & selected_finite
            status = jnp.where(hit & ~interface_valid, interface_status, status)
            status = jnp.where(
                hit & interface_valid & ~selected_finite,
                int(SequentialOpticsStatus.NUMERICAL_FAILURE),
                status,
            )
            distance = intersection.distance
            origin = jnp.where(route_success[..., None], world_point, origin)
            direction = jnp.where(route_success[..., None], selected_direction, direction)
            geometric_path = jnp.where(
                route_success, geometric_path + distance, geometric_path
            )
            optical_path = jnp.where(
                route_success,
                optical_path + refractive_index * distance,
                optical_path,
            )
            refractive_index = jnp.where(
                route_success,
                self.refractive_indices[index + 1].astype(dtype),
                refractive_index,
            )
            traversed = traversed + route_success.astype(jnp.int32)
            surface_margin = jnp.minimum(
                intersection.grazing_margin,
                jnp.minimum(
                    intersection.root_switch_margin,
                    intersection.distance - self.forward_tolerance,
                ),
            )
            surface_margin = jnp.minimum(surface_margin, intersection.aperture_margin)
            surface_margin = jnp.minimum(surface_margin, interface.snell_discriminant)
            branch_margin = jnp.where(
                attempted, jnp.minimum(branch_margin, surface_margin), branch_margin
            )
            finite = finite & jnp.where(
                attempted,
                jnp.isfinite(intersection.distance)
                & jnp.isfinite(intersection.residual)
                & selected_finite,
                True,
            )
            active = route_success

        successful = active & finite & (status == int(SequentialOpticsStatus.SUCCESS))
        status = jnp.where(
            active & ~finite,
            int(SequentialOpticsStatus.NUMERICAL_FAILURE),
            status,
        ).astype(jnp.int32)
        rays = OpticalRayState(
            origins=origin,
            directions=direction,
            refractive_indices=refractive_index,
            geometric_path_lengths=geometric_path,
            optical_path_lengths=optical_path,
        )
        result = SequentialOpticsResult(
            rays=rays,
            valid=successful,
            status=status,
            traversed_surfaces=traversed,
            minimum_snell_discriminant=minimum_snell,
            minimum_aperture_margin=minimum_aperture,
            maximum_intersection_residual=maximum_residual,
            finite=finite,
            successful=successful,
            producer_id=self.prepared_id,
        )
        return result, branch_margin

    def execute(
        self, origins: ArrayLike, directions: ArrayLike, /
    ) -> SequentialOpticsResult:
        """Trace one declared route with fixed work and no splitting or history."""
        result, _ = self._trace(origins, directions)
        return result


__all__ = [
    "PreparedSequentialOptics",
    "SequentialOpticsPlan",
    "SequentialOpticsResult",
    "SequentialOpticsStatus",
    "SurfaceInteraction",
    "SurfaceKind",
]
