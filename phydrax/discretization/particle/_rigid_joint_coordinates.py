#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from opt_einsum import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ._rigid_body import (
    _principal_angle,
    _quaternion_conjugate,
    _quaternion_multiply,
    _rigid_body_relative_rotation,
    _rigid_body_rotation_matrix,
    RigidBodyKinematics,
)
from ._rigid_joints import _RigidMobileIncrement, PreparedRigidJointGraph


class RigidJointCoordinates(StrictModule):
    fixed_translation: Array
    fixed_rotation: Array
    ball_rotation: Array
    hinge_angle: Array
    hinge_rate: Array
    prismatic_position: Array
    prismatic_rate: Array
    distance_length: Array
    distance_rate: Array
    chart_margin: Array
    finite: Array


class RigidJointCoordinateState(StrictModule):
    previous_hinge_angle: Array
    unwrapped_hinge_angle: Array


class RigidJointCoordinateUpdate(StrictModule):
    candidate_state: RigidJointCoordinateState
    accepted_state: RigidJointCoordinateState
    coordinates: RigidJointCoordinates
    successful: Array


class PreparedRigidJointCoordinates(StrictModule, NonTrainableState):
    graph: PreparedRigidJointGraph
    prepared_id: str = eqx.field(static=True)

    def __init__(self, graph: PreparedRigidJointGraph, /):
        if not isinstance(graph, PreparedRigidJointGraph):
            raise TypeError("graph must be a PreparedRigidJointGraph.")
        self.graph = graph
        self.prepared_id = f"rigid-joint-coordinates/{graph.prepared_id}"

    def initialize_state(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointCoordinateState:
        coordinates = self.evaluate(kinematics)
        return RigidJointCoordinateState(
            coordinates.hinge_angle,
            coordinates.hinge_angle,
        )

    def _configuration_coordinates(
        self, kinematics: RigidBodyKinematics, /
    ) -> tuple[Array, Array, Array]:
        graph = self.graph
        rotation = _rigid_body_rotation_matrix(graph.bodies, kinematics.orientation)
        dimension = graph.bodies.ambient_dimension

        if dimension == 2:
            ball_current = _principal_angle(
                kinematics.orientation[graph.ball_right]
                - kinematics.orientation[graph.ball_left]
            )
        else:
            ball_current = _quaternion_multiply(
                _quaternion_conjugate(kinematics.orientation[graph.ball_left]),
                kinematics.orientation[graph.ball_right],
            )
        ball_rotation = _rigid_body_relative_rotation(
            graph.bodies, graph.ball_rest_orientation, ball_current
        )

        if dimension == 3:
            left_axis = contract(
                "...ij,...j->...i",
                rotation[graph.hinge_left],
                graph.hinge_axis_left,
            )
            left_transverse = contract(
                "...ij,...j->...i",
                rotation[graph.hinge_left],
                graph.hinge_transverse_left_1,
            )
            right_transverse = contract(
                "...ij,...j->...i",
                rotation[graph.hinge_right],
                graph.hinge_transverse_right_1,
            )
            sine = jnp.sum(
                left_axis * jnp.cross(left_transverse, right_transverse), axis=-1
            )
            cosine = jnp.sum(left_transverse * right_transverse, axis=-1)
            hinge_angle = jnp.arctan2(sine, cosine)
        else:
            hinge_angle = jnp.zeros((0,), dtype=kinematics.position.dtype)

        left_offset = contract(
            "...ij,...j->...i",
            rotation[graph.prismatic_left],
            graph.prismatic_anchor_left,
        )
        right_offset = contract(
            "...ij,...j->...i",
            rotation[graph.prismatic_right],
            graph.prismatic_anchor_right,
        )
        separation = (
            kinematics.position[graph.prismatic_right]
            + right_offset
            - kinematics.position[graph.prismatic_left]
            - left_offset
        )
        axis = contract(
            "...ij,...j->...i",
            rotation[graph.prismatic_left],
            graph.prismatic_axis_left,
        )
        prismatic_position = jnp.sum(axis * separation, axis=-1)

        distance_left = kinematics.position[graph.distance_left] + contract(
            "...ij,...j->...i",
            rotation[graph.distance_left],
            graph.distance_anchor_left,
        )
        distance_right = kinematics.position[graph.distance_right] + contract(
            "...ij,...j->...i",
            rotation[graph.distance_right],
            graph.distance_anchor_right,
        )
        distance_length = jnp.linalg.norm(distance_right - distance_left, axis=-1)
        return hinge_angle, prismatic_position, distance_length

    def evaluate(self, kinematics: RigidBodyKinematics, /) -> RigidJointCoordinates:
        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        graph = self.graph
        residuals = graph.residuals(kinematics)
        rotation = _rigid_body_rotation_matrix(graph.bodies, kinematics.orientation)
        if graph.bodies.ambient_dimension == 2:
            ball_current = _principal_angle(
                kinematics.orientation[graph.ball_right]
                - kinematics.orientation[graph.ball_left]
            )
        else:
            ball_current = _quaternion_multiply(
                _quaternion_conjugate(kinematics.orientation[graph.ball_left]),
                kinematics.orientation[graph.ball_right],
            )
        ball_rotation = _rigid_body_relative_rotation(
            graph.bodies, graph.ball_rest_orientation, ball_current
        )
        hinge_angle, prismatic_position, distance_length = (
            self._configuration_coordinates(kinematics)
        )

        zero = graph.empty_increment(kinematics.position.dtype)
        tangent = _RigidMobileIncrement(
            kinematics.velocity[graph.mobile_indices],
            kinematics.angular_velocity[graph.mobile_indices],
        )
        rate_function = lambda increment: self._configuration_coordinates(
            graph.retract(kinematics, increment)
        )
        _, rates = jax.jvp(rate_function, (zero,), (tangent,))
        hinge_rate, prismatic_rate, distance_rate = rates
        angle_leaves = (hinge_angle,)
        if ball_rotation.shape[-1:] == (1,):
            angle_leaves = angle_leaves + (ball_rotation[..., 0],)
        maximum_angle = jnp.max(
            jnp.concatenate(tuple(jnp.ravel(value) for value in angle_leaves), axis=0),
            initial=0.0,
        )
        chart_margin = jnp.pi - jnp.abs(maximum_angle)
        finite = (
            tree_allfinite(residuals)
            & tree_allfinite(ball_rotation)
            & tree_allfinite(rates)
            & jnp.isfinite(chart_margin)
        )
        return RigidJointCoordinates(
            residuals.fixed_translation,
            residuals.fixed_rotation,
            ball_rotation,
            hinge_angle,
            hinge_rate,
            prismatic_position,
            prismatic_rate,
            distance_length,
            distance_rate,
            chart_margin,
            finite,
        )

    def update(
        self,
        state: RigidJointCoordinateState,
        kinematics: RigidBodyKinematics,
        successful: Array = jnp.asarray(True),
        /,
    ) -> RigidJointCoordinateUpdate:
        coordinates = self.evaluate(kinematics)
        delta = _principal_angle(coordinates.hinge_angle - state.previous_hinge_angle)
        candidate = RigidJointCoordinateState(
            coordinates.hinge_angle,
            state.unwrapped_hinge_angle + delta,
        )
        valid = jnp.asarray(successful) & coordinates.finite
        accepted = tree_where(valid, candidate, state)
        return RigidJointCoordinateUpdate(candidate, accepted, coordinates, valid)


def prepare_rigid_joint_coordinates(
    graph: PreparedRigidJointGraph, /
) -> PreparedRigidJointCoordinates:
    return PreparedRigidJointCoordinates(graph)


__all__ = [
    "PreparedRigidJointCoordinates",
    "RigidJointCoordinateState",
    "RigidJointCoordinateUpdate",
    "RigidJointCoordinates",
    "prepare_rigid_joint_coordinates",
]
