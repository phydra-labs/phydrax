#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar joint coordinates, rates, and multi-turn winding semantics.

Maximal-coordinate graphs measure hinge angles from body poses, so their
unwrapping necessarily compares consecutive principal endpoints and is valid
only while each accepted increment stays inside the reported chart margin.
Reduced articulations instead integrate the generalized coordinate directly:
their configuration stores principal hinge angles for forward kinematics,
and `PreparedLiftedJointCoordinates` carries the integer winding history that
lifts it. A lifted update follows the actual increment ``q0 + h v`` of the
integrator, never the subtraction of two principal endpoints, so a full turn
with identical endpoints is recorded exactly.
"""

from __future__ import annotations

import math
from numbers import Integral

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
from ..._tree_math import tree_allfinite, tree_where
from ...typing import checked
from ._reduced_articulation import (
    _configuration_increment,
    PreparedReducedArticulation,
)
from ._rigid_body import (
    _principal_angle,
    _quaternion_conjugate,
    _quaternion_multiply,
    _rigid_body_relative_rotation,
    _rigid_body_rotation_matrix,
    RigidBodyKinematics,
)
from ._rigid_joints import _RigidMobileIncrement, PreparedRigidJointGraph


_MAXIMUM_SUPPORTED_TURNS = 2**29
_TWO_PI = 2.0 * math.pi


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

    @checked
    def __init__(self, graph: PreparedRigidJointGraph, /) -> None:
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
        _rigid_body_relative_rotation(
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

    @checked
    def evaluate(self, kinematics: RigidBodyKinematics, /) -> RigidJointCoordinates:
        graph = self.graph
        residuals = graph.residuals(kinematics)
        _rigid_body_rotation_matrix(graph.bodies, kinematics.orientation)
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


def reduced_scalar_joint_dofs(
    articulation: PreparedReducedArticulation,
    joint_ids: ArrayLike,
    owner: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve scientific joint IDs to reduced velocity DOFs and hinge flags.

    The mapping is by joint ID through the articulation's topological edge
    order, never by input position. Fixed edges have no scalar coordinate and
    are refused, as are IDs absent from the articulation.
    """

    requested = np.asarray(joint_ids)
    if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
        raise TypeError(f"{owner} joint IDs must be a rank-1 integer array.")
    hinges = frozenset(articulation.hinge_dof_indices)
    edges = {
        int(identifier): edge
        for edge, identifier in enumerate(
            np.asarray(articulation.joint_ids, dtype=np.int64)
        )
    }
    dofs = np.empty(requested.shape, dtype=np.int32)
    hinge = np.empty(requested.shape, dtype=np.bool_)
    for row, identifier in enumerate(requested.astype(np.int64)):
        edge = edges.get(int(identifier))
        if edge is None:
            raise ValueError(
                f"{owner} joint ID {int(identifier)} is not an articulation edge."
            )
        velocity = articulation.joint_velocity_slices[edge]
        if velocity.stop - velocity.start != 1:
            raise ValueError(
                f"{owner} joint ID {int(identifier)} is a fixed edge without a "
                "scalar coordinate."
            )
        dofs[row] = velocity.start
        hinge[row] = velocity.start in hinges
    return dofs, hinge


class LiftedJointCoordinateState(StrictModule):
    """Integer turn history lifting a principal reduced configuration.

    ``winding[i]`` counts completed turns of hinge DOF ``i``; prismatic DOFs
    carry zero. The lifted coordinate is ``configuration + 2 pi winding``.
    """

    winding: Array


class LiftedJointCoordinateUpdate(StrictModule):
    """One lifted coordinate advance along the actual increment ``q0 + h v``.

    ``configuration`` is the principal endpoint produced by the articulation
    integrator, ``lifted_configuration`` its lift by the candidate winding, and
    ``lift_residual`` the measured disagreement between that lift and the
    lifted source plus ``h v``. Failed updates keep the source winding.
    """

    configuration: Array
    lifted_configuration: Array
    winding_increment: Array
    candidate_state: LiftedJointCoordinateState
    accepted_state: LiftedJointCoordinateState
    lift_residual: Array
    lift_tolerance: Array
    within_range: Array
    finite: Array
    successful: Array


class PreparedLiftedJointCoordinates(StrictModule, NonTrainableState):
    """Winding owner for one prepared fixed-base scalar-joint articulation.

    Forward kinematics keep evaluating principal hinge angles; this owner
    carries the integer turn counts that define lifted coordinates for joint
    stops, servo errors, and the continuous trajectory ``q(s) = q0 + s h v``.
    The counter range is explicit: a candidate whose winding magnitude would
    exceed ``maximum_turns`` is refused, never wrapped. ``lift_resolution`` is
    the floating spacing of the largest admitted lifted hinge magnitude in the
    articulation dtype, the coordinate resolution available to consumers.
    """

    hinge_mask: Array
    nq: int = eqx.field(static=True)
    hinge_dof_indices: tuple[int, ...] = eqx.field(static=True)
    maximum_turns: int = eqx.field(static=True)
    lift_resolution: float = eqx.field(static=True)
    articulation_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        articulation: PreparedReducedArticulation,
        /,
        *,
        maximum_turns: int = 1024,
    ) -> None:
        if isinstance(maximum_turns, bool) or not isinstance(maximum_turns, Integral):
            raise TypeError("maximum_turns must be an integer.")
        turns = int(maximum_turns)
        if turns < 1 or turns > _MAXIMUM_SUPPORTED_TURNS:
            raise ValueError(
                f"maximum_turns must lie in [1, {_MAXIMUM_SUPPORTED_TURNS}]."
            )
        hinge_indices = articulation.hinge_dof_indices
        mask = np.zeros((articulation.nq,), dtype=np.bool_)
        mask[list(hinge_indices)] = True
        dtype = np.dtype(articulation.reference_position.dtype)
        resolution = float(
            np.spacing(np.asarray(_TWO_PI * turns + math.pi, dtype=dtype))
        )
        self.hinge_mask = jnp.asarray(mask)
        self.nq = articulation.nq
        self.hinge_dof_indices = hinge_indices
        self.maximum_turns = turns
        self.lift_resolution = resolution
        self.articulation_id = articulation.prepared_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lifted-joint-coordinates",
                "articulation": articulation.prepared_id,
                "hinge_dof_indices": list(hinge_indices),
                "maximum_turns": turns,
            }
        )

    def _configuration(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != (self.nq,) or not jnp.issubdtype(array.dtype, jnp.floating):
            raise ValueError(f"{name} must be a floating vector of shape {(self.nq,)}.")
        return array

    def _state(self, state: LiftedJointCoordinateState, /) -> Array:
        if not isinstance(state, LiftedJointCoordinateState):
            raise TypeError("state must be LiftedJointCoordinateState.")
        if state.winding.shape != (self.nq,) or not jnp.issubdtype(
            state.winding.dtype, jnp.integer
        ):
            raise ValueError(f"winding must be an integer vector of shape {(self.nq,)}.")
        return state.winding.astype(jnp.int32)

    def initial_state(self, /) -> LiftedJointCoordinateState:
        return LiftedJointCoordinateState(jnp.zeros((self.nq,), dtype=jnp.int32))

    def lift(
        self,
        configuration: ArrayLike,
        state: LiftedJointCoordinateState,
        /,
    ) -> Array:
        """Return ``configuration + 2 pi winding`` on hinge DOFs."""

        point = self._configuration(configuration, "configuration")
        winding = self._state(state)
        return point + _TWO_PI * winding.astype(point.dtype)

    def split(
        self, lifted_configuration: ArrayLike, /
    ) -> tuple[Array, LiftedJointCoordinateState, Array]:
        """Decompose a lifted coordinate into principal angles and winding.

        The third result is false when the winding leaves the declared range or
        the input is nonfinite; the returned winding is then zero and must not
        be accepted as history.
        """

        lifted = self._configuration(lifted_configuration, "lifted_configuration")
        principal = jnp.where(self.hinge_mask, _principal_angle(lifted), lifted)
        turns = jnp.where(self.hinge_mask, jnp.round((lifted - principal) / _TWO_PI), 0.0)
        valid = jnp.all(jnp.isfinite(lifted)) & jnp.all(
            jnp.abs(turns) <= self.maximum_turns
        )
        winding = jnp.where(valid, turns, 0.0).astype(jnp.int32)
        return principal, LiftedJointCoordinateState(winding), valid

    def advance(
        self,
        configuration: ArrayLike,
        state: LiftedJointCoordinateState,
        velocity: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> LiftedJointCoordinateUpdate:
        """Advance winding by the turns crossed by the actual ``q0 + h v``.

        The principal endpoint is the articulation integrator's exact result,
        and the winding increment is the integer number of turns removed by
        its principal-angle wrap, so identical endpoints of a full turn still
        advance the winding by one.
        """

        source = self._configuration(configuration, "configuration")
        rate = self._configuration(velocity, "velocity")
        if rate.dtype != source.dtype:
            raise TypeError("velocity must share the configuration dtype.")
        step = jnp.asarray(step_size, dtype=source.dtype)
        if step.shape != ():
            raise ValueError("step_size must be scalar.")
        previous = self._state(state)
        raw = source + step * rate
        endpoint = _configuration_increment(source, rate, step, self.hinge_dof_indices)
        removed = jnp.where(self.hinge_mask, jnp.round((raw - endpoint) / _TWO_PI), 0.0)
        bound = self.maximum_turns
        increment = jnp.clip(removed, -(bound + 1), bound + 1).astype(jnp.int32)
        winding = previous + increment
        residual = jnp.max(jnp.abs(raw - endpoint - _TWO_PI * removed), initial=0.0)
        magnitude = jnp.max(jnp.abs(raw) + math.pi, initial=math.pi)
        tolerance = 8.0 * jnp.finfo(source.dtype).eps * magnitude
        finite = (
            jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(rate))
            & jnp.isfinite(step)
            & jnp.all(jnp.isfinite(endpoint))
        )
        prismatic_clean = jnp.all(jnp.where(self.hinge_mask, True, previous == 0))
        within_range = (
            prismatic_clean
            & jnp.all(jnp.abs(previous) <= bound)
            & jnp.all(jnp.abs(removed) <= bound)
            & jnp.all(jnp.abs(winding) <= bound)
        )
        successful = finite & within_range & (residual <= tolerance)
        candidate = LiftedJointCoordinateState(winding)
        accepted = tree_where(successful, candidate, LiftedJointCoordinateState(previous))
        lifted = endpoint + _TWO_PI * winding.astype(source.dtype)
        return LiftedJointCoordinateUpdate(
            endpoint,
            lifted,
            increment,
            candidate,
            accepted,
            residual,
            tolerance,
            within_range,
            finite,
            successful,
        )


__all__ = [
    "LiftedJointCoordinateState",
    "LiftedJointCoordinateUpdate",
    "PreparedLiftedJointCoordinates",
    "PreparedRigidJointCoordinates",
    "RigidJointCoordinateState",
    "RigidJointCoordinateUpdate",
    "RigidJointCoordinates",
    "prepare_rigid_joint_coordinates",
]
