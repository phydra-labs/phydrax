#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DensePropertyVerificationPolicy,
    inverse_small_linear,
    SmallLinearSolvePlan,
    verify_dense_properties,
)
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


def _cross_matrix(value: Array, /) -> Array:
    x, y, z = value
    return jnp.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


_INERTIA_SOLVE = SmallLinearSolvePlan(3)

_INERTIA_POLICY = DensePropertyVerificationPolicy(
    require_hermitian=True,
    require_positive_definite=True,
)


def _quaternion_derivative(quaternion: Array, omega: Array, /) -> Array:
    scalar = quaternion[0]
    vector = quaternion[1:]
    return 0.5 * jnp.concatenate(
        (
            jnp.asarray((-jnp.sum(vector * omega),)),
            scalar * omega + jnp.cross(vector, omega),
        )
    )


class VehicleConfiguration(StrictModule, NonTrainableState):
    dry_mass: Array
    dry_inertia: Array
    tank_locations: Array
    tank_capacities: Array
    wheel_axes: Array
    wheel_inertias: Array
    context: AstrodynamicsContext
    configuration_id: str = eqx.field(static=True)

    def __init__(
        self,
        dry_mass,
        dry_inertia,
        tank_locations,
        tank_capacities,
        wheel_axes,
        wheel_inertias,
        context,
        /,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        mass = np.asarray(dry_mass, dtype=np.float64)
        inertia = np.asarray(dry_inertia, dtype=np.float64)
        tanks = np.asarray(tank_locations, dtype=np.float64)
        capacities = np.asarray(tank_capacities, dtype=np.float64)
        axes = np.asarray(wheel_axes, dtype=np.float64)
        wheel_values = np.asarray(wheel_inertias, dtype=np.float64)
        if (
            inertia.shape != (3, 3)
            or tanks.ndim != 2
            or tanks.shape[1] != 3
            or capacities.shape != (tanks.shape[0],)
            or axes.ndim != 2
            or axes.shape[1] != 3
            or wheel_values.shape != (axes.shape[0],)
        ):
            raise ValueError("Vehicle configuration arrays are inconsistent.")
        norms = np.sqrt(np.sum(axes * axes, axis=1)) if axes.size else np.empty((0,))
        inertia_evidence = verify_dense_properties(
            jnp.asarray(inertia), policy=_INERTIA_POLICY
        )
        if (
            mass.shape != ()
            or not np.isfinite(mass)
            or mass <= 0.0
            or np.any(~np.isfinite(inertia))
            or not bool(np.asarray(inertia_evidence.successful))
            or np.any(~np.isfinite(tanks))
            or np.any(~np.isfinite(capacities))
            or np.any(~np.isfinite(axes))
            or np.any(~np.isfinite(wheel_values))
            or np.any(capacities < 0.0)
            or np.any(wheel_values <= 0.0)
            or np.any(norms <= 0.0)
        ):
            raise ValueError(
                "Vehicle masses, inertias, and geometry must be finite and physical."
            )
        self.dry_mass = jnp.asarray(mass)
        self.dry_inertia = jnp.asarray(inertia)
        self.tank_locations = jnp.asarray(tanks)
        self.tank_capacities = jnp.asarray(capacities)
        self.wheel_axes = (
            jnp.asarray(axes / norms[:, None]) if axes.size else jnp.empty((0, 3))
        )
        self.wheel_inertias = jnp.asarray(wheel_values)
        self.context = context
        self.configuration_id = canonical_fingerprint(
            {
                "kind": "vehicle-configuration",
                "context": context.context_id,
                "values": array_tree_fingerprint(
                    (mass, inertia, tanks, capacities, axes, wheel_values)
                ),
            }
        )


class VehicleState(StrictModule):
    position: Array
    velocity: Array
    quaternion: Array
    angular_velocity: Array
    tank_masses: Array
    wheel_momentum: Array


class VehicleEffectorEvaluation(StrictModule):
    force: Array
    torque: Array
    tank_mass_flow: Array
    wheel_torque: Array
    valid: Array


class VehicleResult(StrictModule):
    times: Array
    states: VehicleState
    valid: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class CoupledVehiclePlan(StrictModule, NonTrainableState):
    configuration: VehicleConfiguration
    effectors: tuple[Callable, ...]
    times: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, configuration, effectors, times, /, *, effector_ids):
        if not isinstance(configuration, VehicleConfiguration):
            raise TypeError("configuration must be a VehicleConfiguration.")
        items = tuple(effectors)
        identifiers = tuple(str(value).strip() for value in effector_ids)
        if (
            not items
            or len(items) != len(identifiers)
            or any(not callable(value) for value in items)
            or any(not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("Vehicle effectors and IDs are inconsistent.")
        times_host = np.asarray(times, dtype=np.float64)
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
        ):
            raise ValueError("Vehicle times must be finite and strictly increasing.")
        self.configuration = configuration
        self.effectors = items
        self.times = jnp.asarray(times_host)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-vehicle-plan",
                "configuration": configuration.configuration_id,
                "effectors": list(identifiers),
                "times": array_tree_fingerprint(times_host),
            }
        )

    def _mass_properties(self, state: VehicleState, /) -> tuple[Array, Array, Array]:
        total_mass = self.configuration.dry_mass + jnp.sum(state.tank_masses)
        center = (
            jnp.sum(
                state.tank_masses[:, None] * self.configuration.tank_locations, axis=0
            )
            / total_mass
        )
        offsets = self.configuration.tank_locations - center
        point_inertia = jnp.sum(
            state.tank_masses[:, None, None]
            * (
                jnp.sum(offsets * offsets, axis=1)[:, None, None] * jnp.eye(3)
                - offsets[:, :, None] * offsets[:, None, :]
            ),
            axis=0,
        )
        return total_mass, center, self.configuration.dry_inertia + point_inertia

    def _derivative(
        self, time: Array, state: VehicleState, command: Any, /
    ) -> tuple[VehicleState, Array]:
        evaluations = tuple(effector(time, state, command) for effector in self.effectors)
        force = jnp.sum(jnp.stack(tuple(value.force for value in evaluations)), axis=0)
        torque = jnp.sum(jnp.stack(tuple(value.torque for value in evaluations)), axis=0)
        mass_flow = jnp.sum(
            jnp.stack(tuple(value.tank_mass_flow for value in evaluations)), axis=0
        )
        wheel_torque = jnp.sum(
            jnp.stack(tuple(value.wheel_torque for value in evaluations)), axis=0
        )
        effectors_valid = jnp.all(jnp.stack(tuple(value.valid for value in evaluations)))
        total_mass, _, inertia = self._mass_properties(state)
        inverse_result = inverse_small_linear(_INERTIA_SOLVE, inertia)
        inverse = inverse_result.value
        determinant = inverse_result.determinant
        wheel_angular_momentum = jnp.sum(
            self.configuration.wheel_axes * state.wheel_momentum[:, None], axis=0
        )
        body_momentum = inertia @ state.angular_velocity + wheel_angular_momentum
        angular_acceleration = inverse @ (
            torque - jnp.cross(state.angular_velocity, body_momentum)
        )
        derivative = VehicleState(
            state.velocity,
            force / total_mass,
            _quaternion_derivative(state.quaternion, state.angular_velocity),
            angular_acceleration,
            -mass_flow,
            wheel_torque,
        )
        valid = (
            effectors_valid
            & (total_mass > 0.0)
            & inverse_result.successful
            & (jnp.abs(determinant) > 1.0e-18)
        )
        return derivative, valid

    def rollout(
        self, initial: VehicleState, command_schedule: Callable, /
    ) -> VehicleResult:
        if not isinstance(initial, VehicleState):
            raise TypeError("initial must be a VehicleState.")
        if (
            initial.position.shape != (3,)
            or initial.velocity.shape != (3,)
            or initial.quaternion.shape != (4,)
            or initial.angular_velocity.shape != (3,)
            or initial.tank_masses.shape != self.configuration.tank_capacities.shape
            or initial.wheel_momentum.shape != self.configuration.wheel_inertias.shape
        ):
            raise ValueError("Vehicle state shapes do not match configuration.")
        initial_valid = (
            jnp.all(jnp.isfinite(initial.position))
            & jnp.all(jnp.isfinite(initial.velocity))
            & jnp.all(jnp.isfinite(initial.quaternion))
            & (jnp.sum(initial.quaternion**2) > 0.0)
            & jnp.all(jnp.isfinite(initial.angular_velocity))
            & jnp.all(jnp.isfinite(initial.tank_masses))
            & jnp.all(initial.tank_masses >= 0.0)
            & jnp.all(initial.tank_masses <= self.configuration.tank_capacities)
            & jnp.all(jnp.isfinite(initial.wheel_momentum))
        )

        def add(state, derivative, factor):
            return jax.tree.map(
                lambda value, delta: value + factor * delta, state, derivative
            )

        def step(carry, interval):
            state, active = carry
            start, end = interval
            dt = end - start
            command = command_schedule(start)
            k1, valid1 = self._derivative(start, state, command)
            k2, valid2 = self._derivative(
                start + 0.5 * dt, add(state, k1, 0.5 * dt), command
            )
            k3, valid3 = self._derivative(
                start + 0.5 * dt, add(state, k2, 0.5 * dt), command
            )
            k4, valid4 = self._derivative(end, add(state, k3, dt), command)
            next_state = jax.tree.map(
                lambda value, a, b, c, d: value + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d),
                state,
                k1,
                k2,
                k3,
                k4,
            )
            quaternion = next_state.quaternion / jnp.sqrt(
                jnp.sum(next_state.quaternion**2)
            )
            next_state = eqx.tree_at(
                lambda value: value.quaternion, next_state, quaternion
            )
            finite = jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value))
                        for value in jax.tree.leaves(next_state)
                    )
                )
            )
            valid = (
                active
                & valid1
                & valid2
                & valid3
                & valid4
                & finite
                & jnp.all(next_state.tank_masses >= 0.0)
                & jnp.all(next_state.tank_masses <= self.configuration.tank_capacities)
            )
            accepted = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old), next_state, state
            )
            return (accepted, valid), (accepted, valid)

        intervals = jnp.stack((self.times[:-1], self.times[1:]), axis=-1)
        (_, completed), outputs = jax.lax.scan(step, (initial, initial_valid), intervals)
        states = jax.tree.map(
            lambda value, tail: jnp.concatenate((value[None], tail), axis=0),
            initial,
            outputs[0],
        )
        valid = jnp.concatenate((initial_valid[None], outputs[1]))
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return VehicleResult(self.times, states, valid, status, completed, self.plan_id)


class FswSchedule(StrictModule, NonTrainableState):
    breakpoints: Array
    commands: Array
    modes: Array

    def __init__(self, breakpoints, commands, modes, /):
        points = np.asarray(breakpoints, dtype=np.float64)
        command_values = np.asarray(commands, dtype=np.float64)
        mode_values = np.asarray(modes, dtype=np.int32)
        if (
            points.ndim != 1
            or command_values.shape[0] != points.size
            or mode_values.shape != (points.size,)
            or np.any(np.diff(points) <= 0.0)
        ):
            raise ValueError("FSW schedule arrays are inconsistent.")
        self.breakpoints = jnp.asarray(points)
        self.commands = jnp.asarray(command_values)
        self.modes = jnp.asarray(mode_values)

    def __call__(self, time: ArrayLike, /) -> Array:
        index = jnp.clip(
            jnp.searchsorted(self.breakpoints, time, side="right") - 1,
            0,
            self.breakpoints.size - 1,
        )
        return self.commands[index]


__all__ = [
    "CoupledVehiclePlan",
    "FswSchedule",
    "VehicleConfiguration",
    "VehicleEffectorEvaluation",
    "VehicleResult",
    "VehicleState",
]
