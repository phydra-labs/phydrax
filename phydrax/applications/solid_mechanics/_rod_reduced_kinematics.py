#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ...linalg import AbstractLinearOperator, FunctionLinearOperator
from ._rod_dynamics import RodState


if TYPE_CHECKING:
    from ._rod_reduction import PreparedReducedRod, ReducedRodState


def _quaternion_conjugate(quaternion: Array, /) -> Array:
    return jnp.concatenate((quaternion[..., :1], -quaternion[..., 1:]), axis=-1)


def _quaternion_multiply(left: Array, right: Array, /) -> Array:
    left_scalar = left[..., :1]
    right_scalar = right[..., :1]
    left_vector = left[..., 1:]
    right_vector = right[..., 1:]
    scalar = left_scalar * right_scalar - jnp.sum(
        left_vector * right_vector, axis=-1, keepdims=True
    )
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + jnp.cross(left_vector, right_vector)
    )
    return jnp.concatenate((scalar, vector), axis=-1)


def _unit_quaternion(quaternion: Array, /) -> Array:
    return quaternion / jnp.sqrt(jnp.sum(quaternion * quaternion, axis=-1, keepdims=True))


def _rotation_vector_quaternion(rotation_vector: Array, /) -> Array:
    angle_squared = jnp.sum(rotation_vector * rotation_vector, axis=-1)
    threshold = jnp.sqrt(jnp.finfo(rotation_vector.dtype).eps)
    regular = angle_squared > threshold * threshold
    safe_angle = jnp.sqrt(jnp.maximum(angle_squared, threshold * threshold))
    regular_scalar = jnp.cos(0.5 * safe_angle)
    regular_scale = jnp.sin(0.5 * safe_angle) / safe_angle
    squared_squared = angle_squared * angle_squared
    limiting_scalar = 1.0 - angle_squared / 8.0 + squared_squared / 384.0
    limiting_scale = 0.5 - angle_squared / 48.0 + squared_squared / 3840.0
    scalar = jnp.where(regular, regular_scalar, limiting_scalar)
    scale = jnp.where(regular, regular_scale, limiting_scale)
    return _unit_quaternion(
        jnp.concatenate(
            (scalar[..., None], scale[..., None] * rotation_vector),
            axis=-1,
        )
    )


def _quaternion_rotation_matrix(quaternion: Array, /) -> Array:
    normalized = _unit_quaternion(quaternion)
    scalar = normalized[..., 0]
    x = normalized[..., 1]
    y = normalized[..., 2]
    z = normalized[..., 3]
    return jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - scalar * z),
            2.0 * (x * z + scalar * y),
            2.0 * (x * y + scalar * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - scalar * x),
            2.0 * (x * z - scalar * y),
            2.0 * (y * z + scalar * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(normalized.shape[:-1] + (3, 3))


def _planar_rotation_matrix(angle: Array, /) -> Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(angle.shape + (2, 2))


def _validate_coefficients(
    prepared: "PreparedReducedRod", coefficients: ArrayLike, /
) -> Array:
    return prepared.coefficient_space.validate(jnp.asarray(coefficients))


def target_native_strains(
    prepared: "PreparedReducedRod",
    coefficients: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Return requested native discrete stretch/shear and bend/twist increments."""
    values = _validate_coefficients(prepared, coefficients)
    stretch_shear = ein.contract("sdk,k->sd", prepared.basis.stretch_shear_basis, values)
    bend_twist = ein.contract("sdk,k->sd", prepared.basis.bend_twist_basis, values)
    return stretch_shear, bend_twist


def _orientations_from_strain(
    prepared: "PreparedReducedRod", bend_twist: Array, /
) -> Array:
    rod = prepared.rod
    segment_count = rod.plan.segment_count
    if rod.plan.dimension == 2:
        orientations = [prepared.base_orientation]
        for junction in range(segment_count - 1):
            increment = (
                rod.rest_relative_orientations[junction]
                + rod.dual_lengths[junction] * bend_twist[junction, 0]
            )
            orientations.append(orientations[-1] + increment)
        return jnp.stack(orientations)

    orientations = [_unit_quaternion(prepared.base_orientation)]
    for junction in range(segment_count - 1):
        strain_increment = _rotation_vector_quaternion(
            rod.dual_lengths[junction] * bend_twist[junction]
        )
        current_relative = _quaternion_multiply(
            rod.rest_relative_orientations[junction], strain_increment
        )
        orientations.append(
            _unit_quaternion(_quaternion_multiply(orientations[-1], current_relative))
        )
    return jnp.stack(orientations)


def _positions_from_strain(
    prepared: "PreparedReducedRod",
    orientations: Array,
    stretch_shear: Array,
    /,
) -> Array:
    rod = prepared.rod
    positions = jnp.zeros_like(rod.plan.rest_positions)
    first_node = rod.plan.segment_node_ids[0, 0]
    positions = positions.at[first_node].set(prepared.base_position)
    frames = (
        _planar_rotation_matrix(orientations)
        if rod.plan.dimension == 2
        else _quaternion_rotation_matrix(orientations)
    )
    material_tangents = rod.rest_stretch_shear + stretch_shear
    spatial_edges = rod.plan.rest_lengths[:, None] * ein.contract(
        "sij,sj->si", frames, material_tangents
    )
    for segment in range(rod.plan.segment_count):
        start = rod.plan.segment_node_ids[segment, 0]
        end = rod.plan.segment_node_ids[segment, 1]
        positions = positions.at[end].set(positions[start] + spatial_edges[segment])
    return positions


def lift_configuration(
    prepared: "PreparedReducedRod",
    coefficients: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Lift dimensionless coordinates to native positions and orientation points."""
    values = _validate_coefficients(prepared, coefficients)
    stretch_shear, bend_twist = target_native_strains(prepared, values)
    orientations = _orientations_from_strain(prepared, bend_twist)
    positions = _positions_from_strain(prepared, orientations, stretch_shear)
    configuration = (positions, orientations)
    prepared.rod.configuration_schema.validate(configuration)
    return configuration


def _configuration_velocity(
    prepared: "PreparedReducedRod",
    point: Array,
    tangent: Array,
    /,
) -> tuple[Array, Array]:
    def configuration(values):
        return lift_configuration(prepared, values)

    (positions, orientations), (position_velocity, orientation_velocity) = jax.jvp(
        configuration, (point,), (tangent,)
    )
    del positions
    if prepared.rod.plan.dimension == 2:
        angular_velocity = orientation_velocity
    else:
        material_rate = _quaternion_multiply(
            _quaternion_conjugate(orientations), orientation_velocity
        )
        angular_velocity = 2.0 * material_rate[..., 1:]
    return position_velocity, angular_velocity


def lift_velocity_operator(
    prepared: "PreparedReducedRod",
    coefficients: ArrayLike,
    /,
) -> AbstractLinearOperator:
    """Return the matrix-free reduced-tangent to native-velocity JVP action."""
    point = _validate_coefficients(prepared, coefficients)

    def pushforward(tangent):
        return _configuration_velocity(prepared, point, tangent)

    return FunctionLinearOperator(
        pushforward,
        source=prepared.coefficient_space,
        target=prepared.native_velocity_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "reduced-rod-native-velocity-action",
                "reduction": prepared.prepared_id,
            }
        ),
    )


def lift_effort_pullback_operator(
    prepared: "PreparedReducedRod",
    coefficients: ArrayLike,
    /,
) -> AbstractLinearOperator:
    """Return the algebraic native-effort to reduced-dual VJP action."""
    velocity_operator = lift_velocity_operator(prepared, coefficients)

    def pullback(effort):
        return velocity_operator.transpose_mv(effort)

    return FunctionLinearOperator(
        pullback,
        source=prepared.native_effort_space,
        target=prepared.reduced_effort_space,
        transpose_action=velocity_operator.mv,
        operator_id=canonical_fingerprint(
            {
                "kind": "reduced-rod-native-effort-pullback",
                "reduction": prepared.prepared_id,
            }
        ),
    )


def lift_reduced_rod_velocity(
    prepared: "PreparedReducedRod",
    coefficients: ArrayLike,
    coefficient_velocities: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Push a reduced tangent to native translational and angular velocity."""
    operator = lift_velocity_operator(prepared, coefficients)
    rates = _validate_coefficients(prepared, coefficient_velocities)
    velocities, angular_velocities = operator.mv(rates)
    return velocities, angular_velocities


def lift_reduced_rod_state(
    prepared: "PreparedReducedRod",
    state: "ReducedRodState",
    /,
) -> RodState:
    """Lift one packed reduced phase state to the native rod state contract."""
    prepared.validate_state(state)
    configuration = lift_configuration(prepared, state.coefficients)
    velocity = lift_reduced_rod_velocity(
        prepared, state.coefficients, state.coefficient_velocities
    )
    native_state = prepared.rod.state_from_configuration(configuration)
    return prepared.rod.state_with_velocity(native_state, velocity)


def pullback_reduced_rod_loads(
    prepared: "PreparedReducedRod",
    coefficients: ArrayLike,
    native_forces: ArrayLike,
    native_moments: ArrayLike,
    /,
) -> Array:
    """Pull native force/moment covectors to the reduced coefficient dual."""
    efforts = prepared.rod.effort_from_load(native_forces, native_moments)
    return lift_effort_pullback_operator(prepared, coefficients).mv(efforts)


__all__ = [
    "lift_configuration",
    "lift_effort_pullback_operator",
    "lift_reduced_rod_state",
    "lift_reduced_rod_velocity",
    "lift_velocity_operator",
    "pullback_reduced_rod_loads",
    "target_native_strains",
]
