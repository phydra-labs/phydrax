#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import quaternion_rotation_matrix
from ._vehicle import VehicleEffectorEvaluation, VehicleState


class ThrusterEffector(StrictModule, NonTrainableState):
    directions_body: Array
    locations_body: Array
    maximum_thrust: Array
    mass_flow_per_force: Array
    tank_assignment: Array

    def __init__(
        self,
        directions_body,
        locations_body,
        maximum_thrust,
        mass_flow_per_force,
        tank_assignment,
        /,
    ):
        directions = jnp.asarray(directions_body)
        locations = jnp.asarray(locations_body)
        thrust = jnp.asarray(maximum_thrust)
        flow = jnp.asarray(mass_flow_per_force)
        assignment = jnp.asarray(tank_assignment)
        if (
            directions.ndim != 2
            or directions.shape[1] != 3
            or locations.shape != directions.shape
            or thrust.shape != (directions.shape[0],)
            or flow.shape != thrust.shape
            or assignment.ndim != 2
            or assignment.shape[0] != directions.shape[0]
        ):
            raise ValueError("Thruster arrays are inconsistent.")
        norms = jnp.sqrt(jnp.sum(directions * directions, axis=1))
        self.directions_body = directions / norms[:, None]
        self.locations_body = locations
        self.maximum_thrust = thrust
        self.mass_flow_per_force = flow
        self.tank_assignment = assignment

    def __call__(self, time, state: VehicleState, command, /):
        del time
        throttle = jnp.clip(jnp.asarray(command)[: self.maximum_thrust.size], 0.0, 1.0)
        rotation = quaternion_rotation_matrix(state.quaternion)
        force_body = (
            throttle[:, None] * self.maximum_thrust[:, None] * self.directions_body
        )
        total_force_body = jnp.sum(force_body, axis=0)
        torque_body = jnp.sum(jnp.cross(self.locations_body, force_body), axis=0)
        mass_flow_thruster = throttle * self.maximum_thrust * self.mass_flow_per_force
        tank_flow = contract("kt,k->t", self.tank_assignment, mass_flow_thruster)
        return VehicleEffectorEvaluation(
            rotation @ total_force_body,
            torque_body,
            tank_flow,
            jnp.zeros_like(state.wheel_momentum),
            jnp.all(jnp.isfinite(throttle)),
        )


class ReactionWheelEffector(StrictModule, NonTrainableState):
    axes_body: Array
    maximum_torque: Array
    command_offset: int = eqx.field(static=True)

    def __init__(
        self, axes_body: ArrayLike, maximum_torque: ArrayLike, /, *, command_offset=0
    ):
        axes = jnp.asarray(axes_body)
        torque = jnp.asarray(maximum_torque)
        if axes.ndim != 2 or axes.shape[1] != 3 or torque.shape != (axes.shape[0],):
            raise ValueError("Reaction-wheel arrays are inconsistent.")
        self.axes_body = axes / jnp.sqrt(jnp.sum(axes * axes, axis=1))[:, None]
        self.maximum_torque = torque
        self.command_offset = int(command_offset)

    def __call__(self, time, state: VehicleState, command, /):
        del time
        wheel_count = int(self.maximum_torque.size)
        requested = jnp.asarray(command)[
            self.command_offset : self.command_offset + wheel_count
        ]
        applied = jnp.clip(requested, -self.maximum_torque, self.maximum_torque)
        body_torque = -contract("wi,w->i", self.axes_body, applied)
        return VehicleEffectorEvaluation(
            jnp.zeros((3,)),
            body_torque,
            jnp.zeros_like(state.tank_masses),
            applied,
            jnp.all(jnp.isfinite(applied)),
        )


class SensorEvaluation(StrictModule):
    measurement: Array
    valid: Array
    saturated: Array


class LinearSensorPlan(StrictModule, NonTrainableState):
    matrix: Array
    bias: Array
    lower: Array
    upper: Array

    def __init__(self, matrix, bias, lower, upper, /):
        matrix_ = jnp.asarray(matrix)
        bias_ = jnp.asarray(bias)
        if matrix_.ndim != 2 or bias_.shape != (matrix_.shape[0],):
            raise ValueError("Sensor matrix and bias are inconsistent.")
        self.matrix = matrix_
        self.bias = bias_
        self.lower = jnp.broadcast_to(jnp.asarray(lower), bias_.shape)
        self.upper = jnp.broadcast_to(jnp.asarray(upper), bias_.shape)

    def evaluate(self, state_vector: ArrayLike, /) -> SensorEvaluation:
        raw = self.matrix @ jnp.asarray(state_vector) + self.bias
        clipped = jnp.clip(raw, self.lower, self.upper)
        saturated = clipped != raw
        return SensorEvaluation(clipped, jnp.all(jnp.isfinite(raw)), saturated)


__all__ = [
    "LinearSensorPlan",
    "ReactionWheelEffector",
    "SensorEvaluation",
    "ThrusterEffector",
]
