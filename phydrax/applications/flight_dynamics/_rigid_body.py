#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Quaternion six-degree-of-freedom rigid-body flight dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class FlightDynamicsState:
    position_inertial_m: Array
    velocity_body_m_s: Array
    attitude_body_to_inertial: Array
    angular_velocity_body_rad_s: Array
    time_s: Array


@dataclass(frozen=True, slots=True)
class FlightDynamicsStep:
    state: FlightDynamicsState
    translational_acceleration_body_m_s2: Array
    angular_acceleration_body_rad_s2: Array
    quaternion_norm_error: Array
    kinetic_energy_j: Array
    successful: Array


def _rotation_matrix(quaternion: Array) -> Array:
    w, x, y, z = quaternion
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _quaternion_rate(quaternion: Array, angular_velocity: Array) -> Array:
    w, x, y, z = quaternion
    p, q, r = angular_velocity
    return 0.5 * jnp.asarray(
        (
            -x * p - y * q - z * r,
            w * p + y * r - z * q,
            w * q + z * p - x * r,
            w * r + x * q - y * p,
        )
    )


@dataclass(frozen=True, slots=True)
class RigidBodyFlightSystem:
    mass_kg: float
    inertia_body_kg_m2: Array
    gravity_inertial_m_s2: Array

    @classmethod
    def create(
        cls,
        mass_kg: float,
        inertia_body_kg_m2: ArrayLike,
        /,
        *,
        gravity_inertial_m_s2: ArrayLike = (0.0, 0.0, -9.80665),
        tolerance: float = 1e-10,
    ) -> RigidBodyFlightSystem:
        inertia = np.asarray(inertia_body_kg_m2, dtype=np.float64)
        gravity = np.asarray(gravity_inertial_m_s2, dtype=np.float64)
        if mass_kg <= 0 or inertia.shape != (3, 3) or gravity.shape != (3,):
            raise ValueError("Flight mass, inertia, or gravity has incompatible shape.")
        if not np.allclose(inertia, inertia.T, atol=tolerance, rtol=0):
            raise ValueError("Flight inertia tensor must be symmetric.")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0:
            raise ValueError("Flight inertia tensor must be positive definite.")
        return cls(float(mass_kg), jnp.asarray(inertia), jnp.asarray(gravity))

    def _derivatives(
        self,
        state: FlightDynamicsState,
        force_body_n: Array,
        moment_body_n_m: Array,
    ) -> tuple[Array, Array, Array, Array]:
        quaternion = state.attitude_body_to_inertial
        rotation = _rotation_matrix(quaternion)
        position_rate = rotation @ state.velocity_body_m_s
        gravity_body = rotation.T @ self.gravity_inertial_m_s2
        velocity_rate = (
            force_body_n / self.mass_kg
            + gravity_body
            - jnp.cross(state.angular_velocity_body_rad_s, state.velocity_body_m_s)
        )
        angular_momentum = self.inertia_body_kg_m2 @ state.angular_velocity_body_rad_s
        angular_right = moment_body_n_m - jnp.cross(
            state.angular_velocity_body_rad_s, angular_momentum
        )
        angular_rate = solve(
            LinearSystem(DenseLinearOperator(self.inertia_body_kg_m2)),
            angular_right,
            policy=LinearSolvePolicy(DenseLU()),
        ).value
        quaternion_rate = _quaternion_rate(quaternion, state.angular_velocity_body_rad_s)
        return position_rate, velocity_rate, quaternion_rate, angular_rate

    def advance(
        self,
        state: FlightDynamicsState,
        force_body_n: ArrayLike,
        moment_body_n_m: ArrayLike,
        step_size_s: float,
        /,
    ) -> FlightDynamicsStep:
        force = jnp.asarray(force_body_n)
        moment = jnp.asarray(moment_body_n_m)
        arrays = (
            state.position_inertial_m,
            state.velocity_body_m_s,
            state.angular_velocity_body_rad_s,
            force,
            moment,
        )
        if any(jnp.shape(value) != (3,) for value in arrays) or jnp.shape(
            state.attitude_body_to_inertial
        ) != (4,):
            raise ValueError("Flight state, force, and moment have incompatible shapes.")
        if step_size_s <= 0:
            raise ValueError("Flight integration step must be positive.")
        quaternion = state.attitude_body_to_inertial / jnp.sqrt(
            contract(
                "i,i->",
                state.attitude_body_to_inertial,
                state.attitude_body_to_inertial,
            )
        )
        normalized = FlightDynamicsState(
            state.position_inertial_m,
            state.velocity_body_m_s,
            quaternion,
            state.angular_velocity_body_rad_s,
            state.time_s,
        )
        first = self._derivatives(normalized, force, moment)
        dt = float(step_size_s)
        midpoint_quaternion = quaternion + 0.5 * dt * first[2]
        midpoint_quaternion = midpoint_quaternion / jnp.sqrt(
            contract("i,i->", midpoint_quaternion, midpoint_quaternion)
        )
        midpoint = FlightDynamicsState(
            normalized.position_inertial_m + 0.5 * dt * first[0],
            normalized.velocity_body_m_s + 0.5 * dt * first[1],
            midpoint_quaternion,
            normalized.angular_velocity_body_rad_s + 0.5 * dt * first[3],
            normalized.time_s + 0.5 * dt,
        )
        second = self._derivatives(midpoint, force, moment)
        next_quaternion = quaternion + dt * second[2]
        next_quaternion = next_quaternion / jnp.sqrt(
            contract("i,i->", next_quaternion, next_quaternion)
        )
        next_state = FlightDynamicsState(
            normalized.position_inertial_m + dt * second[0],
            normalized.velocity_body_m_s + dt * second[1],
            next_quaternion,
            normalized.angular_velocity_body_rad_s + dt * second[3],
            normalized.time_s + dt,
        )
        quaternion_error = jnp.abs(
            contract("i,i->", next_quaternion, next_quaternion) - 1
        )
        kinetic = 0.5 * (
            self.mass_kg
            * contract(
                "i,i->", next_state.velocity_body_m_s, next_state.velocity_body_m_s
            )
            + contract(
                "i,ij,j->",
                next_state.angular_velocity_body_rad_s,
                self.inertia_body_kg_m2,
                next_state.angular_velocity_body_rad_s,
            )
        )
        successful = (
            jnp.all(jnp.isfinite(next_state.position_inertial_m))
            & jnp.all(jnp.isfinite(next_state.velocity_body_m_s))
            & jnp.all(jnp.isfinite(next_state.angular_velocity_body_rad_s))
            & (quaternion_error < 1e-12)
        )
        return FlightDynamicsStep(
            next_state, second[1], second[3], quaternion_error, kinetic, successful
        )


__all__ = ["FlightDynamicsState", "FlightDynamicsStep", "RigidBodyFlightSystem"]
