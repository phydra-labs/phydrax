#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from ._reduced_articulation import PreparedReducedArticulation
from ._rigid_body import (
    _quaternion_conjugate,
    _quaternion_multiply,
    _quaternion_relative_rotation_vector,
    _quaternion_retract,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._rigid_joint_coordinates import reduced_scalar_joint_dofs
from ._rigid_joints import PreparedRigidJointGraph, RigidJointKind


class RigidJointCoordinate(StrEnum):
    FIXED_TRANSLATION = "fixed-translation"
    FIXED_ROTATION = "fixed-rotation"
    BALL_ANCHOR = "ball-anchor"
    BALL_ORIENTATION = "ball-orientation"
    HINGE_ANCHOR = "hinge-anchor"
    HINGE_AXIAL = "hinge-axial"
    PRISMATIC_AXIAL = "prismatic-axial"


_COORDINATE_DIMENSIONS = {
    RigidJointCoordinate.FIXED_TRANSLATION: 3,
    RigidJointCoordinate.FIXED_ROTATION: 3,
    RigidJointCoordinate.BALL_ANCHOR: 3,
    RigidJointCoordinate.BALL_ORIENTATION: 3,
    RigidJointCoordinate.HINGE_ANCHOR: 3,
    RigidJointCoordinate.HINGE_AXIAL: 1,
    RigidJointCoordinate.PRISMATIC_AXIAL: 1,
}
_COORDINATE_JOINT_KINDS = {
    RigidJointCoordinate.FIXED_TRANSLATION: RigidJointKind.FIXED,
    RigidJointCoordinate.FIXED_ROTATION: RigidJointKind.FIXED,
    RigidJointCoordinate.BALL_ANCHOR: RigidJointKind.BALL,
    RigidJointCoordinate.BALL_ORIENTATION: RigidJointKind.BALL,
    RigidJointCoordinate.HINGE_ANCHOR: RigidJointKind.HINGE,
    RigidJointCoordinate.HINGE_AXIAL: RigidJointKind.HINGE,
    RigidJointCoordinate.PRISMATIC_AXIAL: RigidJointKind.PRISMATIC,
}
_FREE_COORDINATES = frozenset(
    (
        RigidJointCoordinate.BALL_ORIENTATION,
        RigidJointCoordinate.HINGE_AXIAL,
        RigidJointCoordinate.PRISMATIC_AXIAL,
    )
)
_SCALAR_AXIAL_COORDINATES = frozenset(
    (RigidJointCoordinate.HINGE_AXIAL, RigidJointCoordinate.PRISMATIC_AXIAL)
)


def _joint_identifiers(value: ArrayLike, /) -> np.ndarray:
    identifiers = np.asarray(value)
    if identifiers.ndim != 1:
        raise ValueError("joint_ids must be a vector.")
    if not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError("joint_ids must contain integers.")
    identifiers = identifiers.astype(np.int64, copy=False)
    if np.unique(identifiers).size != identifiers.size:
        raise ValueError("joint_ids must be unique within a law plan.")
    return identifiers


def _coordinate_dimension(coordinate: RigidJointCoordinate, /) -> int:
    if not isinstance(coordinate, RigidJointCoordinate):
        raise TypeError("coordinate must be a RigidJointCoordinate.")
    return _COORDINATE_DIMENSIONS[coordinate]


def _coordinate_values(
    name: str,
    value: ArrayLike,
    count: int,
    dimension: int,
    /,
    *,
    positive: bool = False,
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    shape = (count, dimension)
    if values.ndim == 0:
        values = np.full(shape, values, dtype=values.dtype)
    elif dimension == 1 and values.shape == (count,):
        values = values[:, None]
    elif values.shape == (dimension,):
        values = np.broadcast_to(values, shape).copy()
    elif values.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return values


def _scalar_values(
    name: str,
    value: ArrayLike,
    count: int,
    /,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full((count,), values, dtype=values.dtype)
    elif values.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return values


def _positive_semidefinite_matrices(
    name: str,
    value: ArrayLike,
    count: int,
    dimension: int,
    /,
) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    shape = (count, dimension, dimension)
    if matrices.ndim == 0:
        matrices = np.broadcast_to(
            matrices * np.eye(dimension, dtype=matrices.dtype), shape
        ).copy()
    elif matrices.shape == (dimension, dimension):
        matrices = np.broadcast_to(matrices, shape).copy()
    elif matrices.shape != shape:
        raise ValueError(
            f"{name} must be scalar, ({dimension}, {dimension}), or {shape}."
        )
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{name} must be finite.")
    if count == 0:
        return matrices
    scale = np.maximum(np.max(np.abs(matrices), axis=(-2, -1)), 1.0)
    tolerance = 64.0 * np.finfo(matrices.dtype).eps * scale
    skew = np.max(np.abs(matrices - np.swapaxes(matrices, -1, -2)), axis=(-2, -1))
    if np.any(skew > tolerance):
        raise ValueError(f"{name} must be symmetric.")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.any(eigenvalues < -tolerance[:, None]):
        raise ValueError(f"{name} must be positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return contract("...ik,...k,...jk->...ij", eigenvectors, eigenvalues, eigenvectors)


def _chart_tolerance(value: float, /) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("chart_tolerance must be finite and lie in [0, 1).")
    return tolerance


def _scalar_axial_coordinate(value: RigidJointCoordinate, /) -> RigidJointCoordinate:
    if not isinstance(value, RigidJointCoordinate):
        raise TypeError("coordinate must be a RigidJointCoordinate.")
    if value not in _SCALAR_AXIAL_COORDINATES:
        raise ValueError(
            "Scalar actuators require the hinge-axial or prismatic-axial coordinate."
        )
    return value


def _plan_identifier(
    kind: str,
    coordinate: RigidJointCoordinate,
    values: dict[str, np.ndarray],
    plan_id: str | None,
    /,
) -> str:
    generated = canonical_fingerprint(
        {
            "kind": kind,
            "coordinate": coordinate.value,
            "values": array_tree_fingerprint(values),
        }
    )
    identifier = generated if plan_id is None else str(plan_id)
    if not identifier:
        raise ValueError("plan_id must be nonempty.")
    return identifier


class CompliantRigidJointLawPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    stiffness: Array
    rest_coordinate: Array
    coordinate_scale: Array
    coordinate: RigidJointCoordinate = eqx.field(static=True)
    chart_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        coordinate: RigidJointCoordinate,
        stiffness: ArrayLike,
        /,
        *,
        rest_coordinate: ArrayLike = 0.0,
        coordinate_scale: ArrayLike = 1.0,
        chart_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ) -> None:
        identifiers = _joint_identifiers(joint_ids)
        dimension = _coordinate_dimension(coordinate)
        stiffness_ = _positive_semidefinite_matrices(
            "stiffness", stiffness, identifiers.size, dimension
        )
        rest = _coordinate_values(
            "rest_coordinate", rest_coordinate, identifiers.size, dimension
        )
        scale = _coordinate_values(
            "coordinate_scale",
            coordinate_scale,
            identifiers.size,
            dimension,
            positive=True,
        )
        tolerance = _chart_tolerance(chart_tolerance)
        self.joint_ids = jnp.asarray(identifiers)
        self.stiffness = jnp.asarray(stiffness_)
        self.rest_coordinate = jnp.asarray(rest)
        self.coordinate_scale = jnp.asarray(scale)
        self.coordinate = coordinate
        self.chart_tolerance = tolerance
        self.plan_id = _plan_identifier(
            "compliant-rigid-joint-law-plan",
            coordinate,
            {
                "joint_ids": identifiers,
                "stiffness": stiffness_,
                "rest_coordinate": rest,
                "coordinate_scale": scale,
                "chart_tolerance": np.asarray(tolerance),
            },
            plan_id,
        )

    @property
    def count(self) -> int:
        return self.joint_ids.shape[0]

    def prepare(
        self, graph: PreparedRigidJointGraph, /
    ) -> PreparedCompliantRigidJointLaw:
        return PreparedCompliantRigidJointLaw(self, graph)


class DissipativeRigidJointLawPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    damping: Array
    coordinate_scale: Array
    coordinate: RigidJointCoordinate = eqx.field(static=True)
    chart_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        coordinate: RigidJointCoordinate,
        damping: ArrayLike,
        /,
        *,
        coordinate_scale: ArrayLike = 1.0,
        chart_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ) -> None:
        identifiers = _joint_identifiers(joint_ids)
        dimension = _coordinate_dimension(coordinate)
        damping_ = _positive_semidefinite_matrices(
            "damping", damping, identifiers.size, dimension
        )
        scale = _coordinate_values(
            "coordinate_scale",
            coordinate_scale,
            identifiers.size,
            dimension,
            positive=True,
        )
        tolerance = _chart_tolerance(chart_tolerance)
        self.joint_ids = jnp.asarray(identifiers)
        self.damping = jnp.asarray(damping_)
        self.coordinate_scale = jnp.asarray(scale)
        self.coordinate = coordinate
        self.chart_tolerance = tolerance
        self.plan_id = _plan_identifier(
            "dissipative-rigid-joint-law-plan",
            coordinate,
            {
                "joint_ids": identifiers,
                "damping": damping_,
                "coordinate_scale": scale,
                "chart_tolerance": np.asarray(tolerance),
            },
            plan_id,
        )

    @property
    def count(self) -> int:
        return self.joint_ids.shape[0]

    def prepare(
        self, graph: PreparedRigidJointGraph, /
    ) -> PreparedDissipativeRigidJointLaw:
        return PreparedDissipativeRigidJointLaw(self, graph)


class RigidJointEffortMotorPlan(StrictModule, NonTrainableState):
    """Stateless bounded effort motors on free scalar joint coordinates.

    The commanded effort is a runtime input of `PreparedRigidJointEffortMotor`
    and is never stored in the plan.
    """

    joint_ids: Array
    effort_limit: Array
    coordinate: RigidJointCoordinate = eqx.field(static=True)
    chart_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        /,
        *,
        effort_limit: ArrayLike,
        coordinate: RigidJointCoordinate = RigidJointCoordinate.HINGE_AXIAL,
        chart_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ) -> None:
        identifiers = _joint_identifiers(joint_ids)
        axial = _scalar_axial_coordinate(coordinate)
        limit = _scalar_values(
            "effort_limit", effort_limit, identifiers.size, positive=True
        )
        tolerance = _chart_tolerance(chart_tolerance)
        self.joint_ids = jnp.asarray(identifiers)
        self.effort_limit = jnp.asarray(limit)[:, None]
        self.coordinate = axial
        self.chart_tolerance = tolerance
        self.plan_id = _plan_identifier(
            "rigid-joint-effort-motor-plan",
            axial,
            {
                "joint_ids": identifiers,
                "effort_limit": limit,
                "chart_tolerance": np.asarray(tolerance),
            },
            plan_id,
        )

    @property
    def count(self) -> int:
        return self.joint_ids.shape[0]

    def prepare(self, graph: PreparedRigidJointGraph, /) -> PreparedRigidJointEffortMotor:
        return PreparedRigidJointEffortMotor(self, graph)


class RigidJointPDServoPlan(StrictModule, NonTrainableState):
    """Stateless bounded proportional-derivative servos on scalar coordinates.

    Target coordinates and rates are runtime command inputs of
    `PreparedRigidJointPDServo`; the plan owns only gains, limits, and scales.
    Hinge targets are compared with the unwrapped coordinate.
    """

    joint_ids: Array
    proportional_gain: Array
    derivative_gain: Array
    effort_limit: Array
    coordinate_scale: Array
    coordinate: RigidJointCoordinate = eqx.field(static=True)
    chart_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        /,
        *,
        proportional_gain: ArrayLike,
        derivative_gain: ArrayLike,
        effort_limit: ArrayLike,
        coordinate: RigidJointCoordinate = RigidJointCoordinate.HINGE_AXIAL,
        coordinate_scale: ArrayLike = 1.0,
        chart_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ) -> None:
        identifiers = _joint_identifiers(joint_ids)
        axial = _scalar_axial_coordinate(coordinate)
        proportional = _scalar_values(
            "proportional_gain",
            proportional_gain,
            identifiers.size,
            nonnegative=True,
        )
        derivative = _scalar_values(
            "derivative_gain", derivative_gain, identifiers.size, nonnegative=True
        )
        limit = _scalar_values(
            "effort_limit", effort_limit, identifiers.size, positive=True
        )
        scale = _scalar_values(
            "coordinate_scale", coordinate_scale, identifiers.size, positive=True
        )
        tolerance = _chart_tolerance(chart_tolerance)
        self.joint_ids = jnp.asarray(identifiers)
        self.proportional_gain = jnp.asarray(proportional)[:, None]
        self.derivative_gain = jnp.asarray(derivative)[:, None]
        self.effort_limit = jnp.asarray(limit)[:, None]
        self.coordinate_scale = jnp.asarray(scale)[:, None]
        self.coordinate = axial
        self.chart_tolerance = tolerance
        self.plan_id = _plan_identifier(
            "rigid-joint-pd-servo-plan",
            axial,
            {
                "joint_ids": identifiers,
                "proportional_gain": proportional,
                "derivative_gain": derivative,
                "effort_limit": limit,
                "coordinate_scale": scale,
                "chart_tolerance": np.asarray(tolerance),
            },
            plan_id,
        )

    @property
    def count(self) -> int:
        return self.joint_ids.shape[0]

    def prepare(self, graph: PreparedRigidJointGraph, /) -> PreparedRigidJointPDServo:
        return PreparedRigidJointPDServo(self, graph)


class RigidJointLawCompatibility(StrictModule, NonTrainableState):
    joint_found: Array
    joint_kind_matches: Array
    coordinate_is_free: Array
    dimension_supported: Array
    compatible: Array
    valid: Array


def _plan_coordinate_and_ids(
    plan: (
        CompliantRigidJointLawPlan
        | DissipativeRigidJointLawPlan
        | RigidJointEffortMotorPlan
        | RigidJointPDServoPlan
    ),
    /,
) -> tuple[RigidJointCoordinate, np.ndarray]:
    if isinstance(
        plan,
        (
            CompliantRigidJointLawPlan,
            DissipativeRigidJointLawPlan,
            RigidJointEffortMotorPlan,
            RigidJointPDServoPlan,
        ),
    ):
        return plan.coordinate, np.asarray(plan.joint_ids)
    raise TypeError("plan must be a rigid-joint law or actuator plan.")


def evaluate_rigid_joint_law_compatibility(
    plan: (
        CompliantRigidJointLawPlan
        | DissipativeRigidJointLawPlan
        | RigidJointEffortMotorPlan
        | RigidJointPDServoPlan
    ),
    graph: PreparedRigidJointGraph,
    /,
) -> RigidJointLawCompatibility:
    if not isinstance(graph, PreparedRigidJointGraph):
        raise TypeError("graph must be a PreparedRigidJointGraph.")
    coordinate, requested = _plan_coordinate_and_ids(plan)
    identifiers = np.asarray(graph.row_layout.joint_ids)
    kinds = np.asarray(graph.row_layout.joint_kinds)
    if identifiers.size == 0:
        found = np.zeros(requested.shape, dtype=np.bool_)
        selected_kinds = np.zeros(requested.shape, dtype=np.int32)
    else:
        order = np.argsort(identifiers)
        sorted_identifiers = identifiers[order]
        ranks = np.searchsorted(sorted_identifiers, requested)
        safe_ranks = np.minimum(ranks, sorted_identifiers.size - 1)
        found = (ranks < sorted_identifiers.size) & (
            sorted_identifiers[safe_ranks] == requested
        )
        selected_kinds = kinds[order[safe_ranks]]
    kind_matches = found & (selected_kinds == int(_COORDINATE_JOINT_KINDS[coordinate]))
    coordinate_is_free = np.full(
        requested.shape, coordinate in _FREE_COORDINATES, dtype=np.bool_
    )
    dimension_supported = np.full(
        requested.shape, graph.bodies.ambient_dimension == 3, dtype=np.bool_
    )
    compatible = found & kind_matches & coordinate_is_free & dimension_supported
    return RigidJointLawCompatibility(
        jnp.asarray(found),
        jnp.asarray(kind_matches),
        jnp.asarray(coordinate_is_free),
        jnp.asarray(dimension_supported),
        jnp.asarray(compatible),
        jnp.asarray(np.all(compatible) and graph.bodies.ambient_dimension == 3),
    )


class RigidJointHingeCoordinateState(StrictModule):
    wrapped_coordinate: Array
    unwrapped_coordinate: Array
    chart_valid: Array


class RigidJointHingeCoordinateUpdate(StrictModule):
    candidate_state: RigidJointHingeCoordinateState
    accepted_state: RigidJointHingeCoordinateState
    successful: Array


def candidate_rigid_joint_hinge_coordinate(
    state: RigidJointHingeCoordinateState,
    wrapped_coordinate: Array,
    chart_margin: Array,
    /,
    *,
    chart_tolerance: float = 1.0e-8,
) -> RigidJointHingeCoordinateState:
    if not isinstance(state, RigidJointHingeCoordinateState):
        raise TypeError("state must be a RigidJointHingeCoordinateState.")
    wrapped = jnp.asarray(wrapped_coordinate)
    margin = jnp.asarray(chart_margin)
    if wrapped.shape != state.wrapped_coordinate.shape:
        raise ValueError("wrapped_coordinate must match the hinge state shape.")
    if margin.shape != wrapped.shape[:-1]:
        raise ValueError("chart_margin must match the hinge joint axis.")
    tolerance = _chart_tolerance(chart_tolerance)
    difference = wrapped - state.wrapped_coordinate
    increment = jnp.arctan2(jnp.sin(difference), jnp.cos(difference))
    finite = (
        jnp.all(jnp.isfinite(wrapped), axis=-1)
        & jnp.isfinite(margin)
        & jnp.all(jnp.isfinite(increment), axis=-1)
    )
    valid = finite & (margin > tolerance)
    return RigidJointHingeCoordinateState(
        wrapped,
        state.unwrapped_coordinate + increment,
        valid,
    )


def accept_rigid_joint_hinge_coordinate(
    state: RigidJointHingeCoordinateState,
    candidate: RigidJointHingeCoordinateState,
    accepted: ArrayLike,
    /,
) -> RigidJointHingeCoordinateState:
    if not isinstance(state, RigidJointHingeCoordinateState) or not isinstance(
        candidate, RigidJointHingeCoordinateState
    ):
        raise TypeError("state and candidate must be RigidJointHingeCoordinateState.")
    if (
        state.wrapped_coordinate.shape != candidate.wrapped_coordinate.shape
        or state.unwrapped_coordinate.shape != candidate.unwrapped_coordinate.shape
        or state.chart_valid.shape != candidate.chart_valid.shape
    ):
        raise ValueError("state and candidate hinge-coordinate shapes must match.")
    predicate = jnp.asarray(accepted, dtype=jnp.bool_) & jnp.all(candidate.chart_valid)
    return jax.tree.map(lambda new, old: jnp.where(predicate, new, old), candidate, state)


def update_rigid_joint_hinge_coordinate(
    state: RigidJointHingeCoordinateState,
    wrapped_coordinate: Array,
    chart_margin: Array,
    accepted: ArrayLike,
    /,
    *,
    chart_tolerance: float = 1.0e-8,
) -> RigidJointHingeCoordinateUpdate:
    candidate = candidate_rigid_joint_hinge_coordinate(
        state,
        wrapped_coordinate,
        chart_margin,
        chart_tolerance=chart_tolerance,
    )
    successful = jnp.all(candidate.chart_valid)
    accepted_state = accept_rigid_joint_hinge_coordinate(
        state, candidate, jnp.asarray(accepted, dtype=jnp.bool_) & successful
    )
    return RigidJointHingeCoordinateUpdate(candidate, accepted_state, successful)


class RigidJointLawEvidence(StrictModule):
    finite: Array
    chart_valid: Array
    compatible: Array
    valid: Array


class RigidJointLawEvaluation(StrictModule):
    coordinate: Array
    rate: Array
    effort: Array
    load: RigidBodyLoad
    stored_energy: Array
    dissipation_rate: Array
    actuator_source_power: Array
    saturation_margin: Array
    chart_margin: Array
    saturated: Array
    candidate_state: RigidJointHingeCoordinateState | None
    accepted_state: RigidJointHingeCoordinateState | None
    evidence: RigidJointLawEvidence
    successful: Array


class _PreparedRigidJointCoordinate(StrictModule, NonTrainableState):
    graph: PreparedRigidJointGraph
    indices: Array
    coordinate: RigidJointCoordinate = eqx.field(static=True)
    chart_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        graph: PreparedRigidJointGraph,
        joint_ids: Array,
        coordinate: RigidJointCoordinate,
        chart_tolerance: float,
        /,
    ) -> None:
        match _COORDINATE_JOINT_KINDS[coordinate]:
            case RigidJointKind.FIXED:
                joint_plan = graph.plan.fixed
            case RigidJointKind.BALL:
                joint_plan = graph.plan.ball
            case RigidJointKind.HINGE:
                joint_plan = graph.plan.hinge
            case RigidJointKind.PRISMATIC:
                joint_plan = graph.plan.prismatic
            case kind:
                raise ValueError(f"No scalar joint law coordinate for kind {kind!r}.")
        if joint_plan is None:
            indices = np.empty((0,), dtype=np.int32)
        else:
            available = np.asarray(joint_plan.joint_ids)
            order = np.argsort(available)
            ranks = np.searchsorted(available[order], np.asarray(joint_ids))
            indices = order[ranks].astype(np.int32)
        self.graph = graph
        self.indices = jnp.asarray(indices)
        self.coordinate = coordinate
        self.chart_tolerance = chart_tolerance

    @property
    def count(self) -> int:
        return self.indices.shape[0]

    def _relative_quaternion(
        self, kinematics: RigidBodyKinematics, left: Array, right: Array, /
    ) -> Array:
        return _quaternion_multiply(
            _quaternion_conjugate(kinematics.orientation[left]),
            kinematics.orientation[right],
        )

    def wrapped_coordinate(
        self, kinematics: RigidBodyKinematics, /
    ) -> tuple[Array, Array]:
        if self.coordinate is RigidJointCoordinate.FIXED_TRANSLATION:
            rotation = quaternion_rotation_matrix(kinematics.orientation)
            left = self.graph.fixed_left[self.indices]
            right = self.graph.fixed_right[self.indices]
            relative = kinematics.position[right] - kinematics.position[left]
            value = (
                contract("...ji,...j->...i", rotation[left], relative)
                - self.graph.fixed_rest_offset[self.indices]
            )
            margin = jnp.ones((self.count,), dtype=value.dtype)
        elif self.coordinate is RigidJointCoordinate.FIXED_ROTATION:
            left = self.graph.fixed_left[self.indices]
            right = self.graph.fixed_right[self.indices]
            current = self._relative_quaternion(kinematics, left, right)
            reference = self.graph.fixed_rest_orientation[self.indices]
            value = _quaternion_relative_rotation_vector(reference, current)
            delta = _quaternion_multiply(_quaternion_conjugate(reference), current)
            norm = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
            margin = jnp.abs(delta[..., 0]) / norm
        elif self.coordinate is RigidJointCoordinate.BALL_ANCHOR:
            rotation = quaternion_rotation_matrix(kinematics.orientation)
            left = self.graph.ball_left[self.indices]
            right = self.graph.ball_right[self.indices]
            left_offset = contract(
                "...ij,...j->...i",
                rotation[left],
                self.graph.ball_anchor_left[self.indices],
            )
            right_offset = contract(
                "...ij,...j->...i",
                rotation[right],
                self.graph.ball_anchor_right[self.indices],
            )
            value = (
                kinematics.position[left]
                + left_offset
                - kinematics.position[right]
                - right_offset
            )
            margin = jnp.ones((self.count,), dtype=value.dtype)
        elif self.coordinate is RigidJointCoordinate.BALL_ORIENTATION:
            left = self.graph.ball_left[self.indices]
            right = self.graph.ball_right[self.indices]
            relative = self._relative_quaternion(kinematics, left, right)
            reference = self.graph.ball_rest_orientation[self.indices]
            value = _quaternion_relative_rotation_vector(reference, relative)
            delta = _quaternion_multiply(_quaternion_conjugate(reference), relative)
            norm = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
            margin = jnp.abs(delta[..., 0]) / norm
        elif self.coordinate is RigidJointCoordinate.HINGE_ANCHOR:
            rotation = quaternion_rotation_matrix(kinematics.orientation)
            left = self.graph.hinge_left[self.indices]
            right = self.graph.hinge_right[self.indices]
            left_offset = contract(
                "...ij,...j->...i",
                rotation[left],
                self.graph.hinge_anchor_left[self.indices],
            )
            right_offset = contract(
                "...ij,...j->...i",
                rotation[right],
                self.graph.hinge_anchor_right[self.indices],
            )
            value = (
                kinematics.position[left]
                + left_offset
                - kinematics.position[right]
                - right_offset
            )
            margin = jnp.ones((self.count,), dtype=value.dtype)
        elif self.coordinate is RigidJointCoordinate.PRISMATIC_AXIAL:
            # The signed slide along the left-frame axis; it is globally
            # defined, so the chart margin is identically one.
            rotation = quaternion_rotation_matrix(kinematics.orientation)
            left = self.graph.prismatic_left[self.indices]
            right = self.graph.prismatic_right[self.indices]
            left_offset = contract(
                "...ij,...j->...i",
                rotation[left],
                self.graph.prismatic_anchor_left[self.indices],
            )
            right_offset = contract(
                "...ij,...j->...i",
                rotation[right],
                self.graph.prismatic_anchor_right[self.indices],
            )
            axis = contract(
                "...ij,...j->...i",
                rotation[left],
                self.graph.prismatic_axis_left[self.indices],
            )
            separation = (
                kinematics.position[right]
                + right_offset
                - kinematics.position[left]
                - left_offset
            )
            value = jnp.sum(axis * separation, axis=-1)[:, None]
            margin = jnp.ones((self.count,), dtype=value.dtype)
        else:
            rotation = quaternion_rotation_matrix(kinematics.orientation)
            left = self.graph.hinge_left[self.indices]
            right = self.graph.hinge_right[self.indices]
            axis = contract(
                "...ij,...j->...i",
                rotation[left],
                self.graph.hinge_axis_left[self.indices],
            )
            left_transverse = contract(
                "...ij,...j->...i",
                rotation[left],
                self.graph.hinge_transverse_left_1[self.indices],
            )
            right_transverse = contract(
                "...ij,...j->...i",
                rotation[right],
                self.graph.hinge_transverse_right_1[self.indices],
            )
            sine = jnp.sum(axis * jnp.cross(left_transverse, right_transverse), axis=-1)
            cosine = jnp.sum(left_transverse * right_transverse, axis=-1)
            value = jnp.arctan2(sine, cosine)[:, None]
            right_axis = contract(
                "...ij,...j->...i",
                rotation[right],
                self.graph.hinge_axis_right[self.indices],
            )
            alignment = jnp.sum(axis * right_axis, axis=-1)
            branch_margin = jnp.sqrt(jnp.maximum(0.5 * (1.0 + cosine), 0.0))
            margin = jnp.minimum(alignment, branch_margin)
        return value, margin

    def _coordinate_from_tangent(
        self,
        kinematics: RigidBodyKinematics,
        state: RigidJointHingeCoordinateState | None,
        translation: Array,
        rotation: Array,
        /,
    ) -> Array:
        moved = RigidBodyKinematics(
            kinematics.position + translation,
            kinematics.velocity,
            _quaternion_retract(kinematics.orientation, rotation),
            kinematics.angular_velocity,
        )
        wrapped, _ = self.wrapped_coordinate(moved)
        if self.coordinate is RigidJointCoordinate.HINGE_AXIAL and self.count > 0:
            if not isinstance(state, RigidJointHingeCoordinateState):
                raise TypeError(
                    "Hinge-axial evaluation requires RigidJointHingeCoordinateState."
                )
            difference = wrapped - state.wrapped_coordinate
            increment = jnp.arctan2(jnp.sin(difference), jnp.cos(difference))
            return state.unwrapped_coordinate + increment
        return wrapped

    def initialize_state(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointHingeCoordinateState | None:
        _validate_kinematics(self.graph, kinematics)
        if self.coordinate is not RigidJointCoordinate.HINGE_AXIAL or self.count == 0:
            return None
        wrapped, margin = self.wrapped_coordinate(kinematics)
        finite = jnp.all(jnp.isfinite(wrapped), axis=-1) & jnp.isfinite(margin)
        return RigidJointHingeCoordinateState(
            wrapped,
            wrapped,
            finite & (margin > self.chart_tolerance),
        )

    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        state: RigidJointHingeCoordinateState | None,
        /,
    ) -> tuple[
        Array,
        Array,
        Array,
        RigidJointHingeCoordinateState | None,
        RigidJointHingeCoordinateState | None,
    ]:
        _validate_kinematics(self.graph, kinematics)
        if self.coordinate is RigidJointCoordinate.HINGE_AXIAL and self.count > 0:
            if not isinstance(state, RigidJointHingeCoordinateState):
                raise TypeError(
                    "Hinge-axial evaluation requires RigidJointHingeCoordinateState."
                )
            if state.wrapped_coordinate.shape != (self.count, 1):
                raise ValueError("Hinge-coordinate state has an incompatible shape.")
        elif state is not None:
            raise TypeError("Only hinge-axial laws accept runtime coordinate state.")
        zero_translation = jnp.zeros_like(kinematics.position)
        zero_rotation = jnp.zeros_like(kinematics.angular_velocity)
        tangent_function = lambda translation, rotation: self._coordinate_from_tangent(
            kinematics, state, translation, rotation
        )
        coordinate, rate = jax.jvp(
            tangent_function,
            (zero_translation, zero_rotation),
            (kinematics.velocity, kinematics.angular_velocity),
        )
        wrapped, chart_margin = self.wrapped_coordinate(kinematics)
        if state is None:
            return coordinate, rate, chart_margin, None, None
        candidate = candidate_rigid_joint_hinge_coordinate(
            state,
            wrapped,
            chart_margin,
            chart_tolerance=self.chart_tolerance,
        )
        return coordinate, rate, chart_margin, candidate, state

    def load(
        self,
        kinematics: RigidBodyKinematics,
        state: RigidJointHingeCoordinateState | None,
        effort: Array,
        /,
    ) -> RigidBodyLoad:
        zero_translation = jnp.zeros_like(kinematics.position)
        zero_rotation = jnp.zeros_like(kinematics.angular_velocity)
        function = lambda translation, rotation: self._coordinate_from_tangent(
            kinematics, state, translation, rotation
        )
        _, pullback = jax.vjp(function, zero_translation, zero_rotation)
        force, torque = pullback(effort)
        return RigidBodyLoad(force, torque)


def _validate_kinematics(
    graph: PreparedRigidJointGraph, kinematics: RigidBodyKinematics, /
) -> None:
    if not isinstance(kinematics, RigidBodyKinematics):
        raise TypeError("kinematics must be RigidBodyKinematics.")
    capacity = graph.bodies.capacity
    if (
        kinematics.position.shape != (capacity, 3)
        or kinematics.velocity.shape != (capacity, 3)
        or kinematics.orientation.shape != (capacity, 4)
        or kinematics.angular_velocity.shape != (capacity, 3)
    ):
        raise ValueError("Rigid-body kinematics have incompatible joint-law shapes.")


def _require_compatible(
    plan: (
        CompliantRigidJointLawPlan
        | DissipativeRigidJointLawPlan
        | RigidJointEffortMotorPlan
        | RigidJointPDServoPlan
    ),
    graph: PreparedRigidJointGraph,
    /,
) -> RigidJointLawCompatibility:
    compatibility = evaluate_rigid_joint_law_compatibility(plan, graph)
    if not bool(np.asarray(compatibility.valid)):
        requested = np.asarray(plan.joint_ids)
        rejected = requested[~np.asarray(compatibility.compatible)]
        raise ValueError(
            "Rigid-joint laws require existing free three-dimensional coordinates; "
            f"incompatible joint IDs: {rejected.tolist()}."
        )
    return compatibility


def _prepared_identifier(kind: str, plan_id: str, graph_id: str, /) -> str:
    return canonical_fingerprint({"kind": kind, "plan": plan_id, "graph": graph_id})


def _evidence(
    compatibility: RigidJointLawCompatibility,
    coordinate: Array,
    rate: Array,
    effort: Array,
    load: RigidBodyLoad,
    stored_energy: Array,
    dissipation_rate: Array,
    source_power: Array,
    saturation_margin: Array,
    chart_margin: Array,
    chart_tolerance: float,
    /,
) -> RigidJointLawEvidence:
    leaves = (
        coordinate,
        rate,
        effort,
        load.force,
        load.torque,
        stored_energy,
        dissipation_rate,
        source_power,
        saturation_margin,
        chart_margin,
    )
    finite = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in leaves)))
    chart_valid = jnp.all(chart_margin > chart_tolerance)
    compatible = compatibility.valid
    valid = (
        finite
        & chart_valid
        & compatible
        & (stored_energy >= 0.0)
        & (dissipation_rate >= 0.0)
    )
    return RigidJointLawEvidence(finite, chart_valid, compatible, valid)


def _evaluation(
    coordinate: Array,
    rate: Array,
    effort: Array,
    load: RigidBodyLoad,
    stored_energy: Array,
    dissipation_rate: Array,
    source_power: Array,
    saturation_margin: Array,
    chart_margin: Array,
    saturated: Array,
    candidate_state: RigidJointHingeCoordinateState | None,
    current_state: RigidJointHingeCoordinateState | None,
    compatibility: RigidJointLawCompatibility,
    chart_tolerance: float,
    /,
) -> RigidJointLawEvaluation:
    evidence = _evidence(
        compatibility,
        coordinate,
        rate,
        effort,
        load,
        stored_energy,
        dissipation_rate,
        source_power,
        saturation_margin,
        chart_margin,
        chart_tolerance,
    )
    accepted_state = (
        None
        if current_state is None or candidate_state is None
        else accept_rigid_joint_hinge_coordinate(
            current_state, candidate_state, evidence.valid
        )
    )
    return RigidJointLawEvaluation(
        coordinate,
        rate,
        effort,
        load,
        stored_energy,
        dissipation_rate,
        source_power,
        saturation_margin,
        chart_margin,
        saturated,
        candidate_state,
        accepted_state,
        evidence,
        evidence.valid,
    )


class PreparedCompliantRigidJointLaw(StrictModule, NonTrainableState):
    plan: CompliantRigidJointLawPlan
    graph: PreparedRigidJointGraph
    compatibility: RigidJointLawCompatibility
    coordinate: _PreparedRigidJointCoordinate
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: CompliantRigidJointLawPlan, graph: PreparedRigidJointGraph, /
    ) -> None:
        compatibility = _require_compatible(plan, graph)
        self.plan = plan
        self.graph = graph
        self.compatibility = compatibility
        self.coordinate = _PreparedRigidJointCoordinate(
            graph, plan.joint_ids, plan.coordinate, plan.chart_tolerance
        )
        self.prepared_id = _prepared_identifier(
            "prepared-compliant-rigid-joint-law", plan.plan_id, graph.prepared_id
        )

    def initialize_state(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointHingeCoordinateState | None:
        return self.coordinate.initialize_state(kinematics)

    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        state: RigidJointHingeCoordinateState | None = None,
        /,
    ) -> RigidJointLawEvaluation:
        coordinate, rate, margin, candidate, current = self.coordinate.evaluate(
            kinematics, state
        )
        normalized = (coordinate - self.plan.rest_coordinate) / self.plan.coordinate_scale
        response = contract("nij,nj->ni", self.plan.stiffness, normalized)
        effort = -response / self.plan.coordinate_scale
        stored_energy = jnp.maximum(
            0.5
            * jnp.sum(
                contract("ni,nij,nj->n", normalized, self.plan.stiffness, normalized)
            ),
            0.0,
        )
        load = self.coordinate.load(kinematics, state, effort)
        zero = jnp.zeros((), dtype=coordinate.dtype)
        return _evaluation(
            coordinate,
            rate,
            effort,
            load,
            stored_energy,
            zero,
            zero,
            jnp.zeros_like(effort),
            margin,
            jnp.zeros_like(effort, dtype=jnp.bool_),
            candidate,
            current,
            self.compatibility,
            self.plan.chart_tolerance,
        )


class PreparedDissipativeRigidJointLaw(StrictModule, NonTrainableState):
    plan: DissipativeRigidJointLawPlan
    graph: PreparedRigidJointGraph
    compatibility: RigidJointLawCompatibility
    coordinate: _PreparedRigidJointCoordinate
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: DissipativeRigidJointLawPlan, graph: PreparedRigidJointGraph, /
    ) -> None:
        compatibility = _require_compatible(plan, graph)
        self.plan = plan
        self.graph = graph
        self.compatibility = compatibility
        self.coordinate = _PreparedRigidJointCoordinate(
            graph, plan.joint_ids, plan.coordinate, plan.chart_tolerance
        )
        self.prepared_id = _prepared_identifier(
            "prepared-dissipative-rigid-joint-law", plan.plan_id, graph.prepared_id
        )

    def initialize_state(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointHingeCoordinateState | None:
        return self.coordinate.initialize_state(kinematics)

    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        state: RigidJointHingeCoordinateState | None = None,
        /,
    ) -> RigidJointLawEvaluation:
        coordinate, rate, margin, candidate, current = self.coordinate.evaluate(
            kinematics, state
        )
        normalized_rate = rate / self.plan.coordinate_scale
        response = contract("nij,nj->ni", self.plan.damping, normalized_rate)
        effort = -response / self.plan.coordinate_scale
        raw_dissipation = jnp.sum(
            contract(
                "ni,nij,nj->n",
                normalized_rate,
                self.plan.damping,
                normalized_rate,
            )
        )
        dissipation = jnp.maximum(raw_dissipation, 0.0)
        load = self.coordinate.load(kinematics, state, effort)
        zero = jnp.zeros((), dtype=coordinate.dtype)
        return _evaluation(
            coordinate,
            rate,
            effort,
            load,
            zero,
            dissipation,
            zero,
            jnp.zeros_like(effort),
            margin,
            jnp.zeros_like(effort, dtype=jnp.bool_),
            candidate,
            current,
            self.compatibility,
            self.plan.chart_tolerance,
        )


class PreparedRigidJointEffortMotor(StrictModule, NonTrainableState):
    plan: RigidJointEffortMotorPlan
    graph: PreparedRigidJointGraph
    compatibility: RigidJointLawCompatibility
    coordinate: _PreparedRigidJointCoordinate
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: RigidJointEffortMotorPlan, graph: PreparedRigidJointGraph, /
    ) -> None:
        compatibility = _require_compatible(plan, graph)
        self.plan = plan
        self.graph = graph
        self.compatibility = compatibility
        self.coordinate = _PreparedRigidJointCoordinate(
            graph,
            plan.joint_ids,
            plan.coordinate,
            plan.chart_tolerance,
        )
        self.prepared_id = _prepared_identifier(
            "prepared-rigid-joint-effort-motor", plan.plan_id, graph.prepared_id
        )

    def initialize_state(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointHingeCoordinateState | None:
        return self.coordinate.initialize_state(kinematics)

    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        command: ArrayLike,
        state: RigidJointHingeCoordinateState | None = None,
        /,
    ) -> RigidJointLawEvaluation:
        """Apply the runtime commanded effort, clipped to the plan limit."""

        coordinate, rate, margin, candidate, current = self.coordinate.evaluate(
            kinematics, state
        )
        requested = _scalar_command(command, self.plan.count, coordinate.dtype)
        limit = self.plan.effort_limit.astype(coordinate.dtype)
        effort = jnp.clip(requested, -limit, limit)
        saturation_margin = limit - jnp.abs(requested)
        saturated = jnp.abs(requested) > limit
        source_power = jnp.sum(effort * rate)
        load = self.coordinate.load(kinematics, state, effort)
        zero = jnp.zeros((), dtype=coordinate.dtype)
        return _evaluation(
            coordinate,
            rate,
            effort,
            load,
            zero,
            zero,
            source_power,
            saturation_margin,
            margin,
            saturated,
            candidate,
            current,
            self.compatibility,
            self.plan.chart_tolerance,
        )


class PreparedRigidJointPDServo(StrictModule, NonTrainableState):
    plan: RigidJointPDServoPlan
    graph: PreparedRigidJointGraph
    compatibility: RigidJointLawCompatibility
    coordinate: _PreparedRigidJointCoordinate
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: RigidJointPDServoPlan, graph: PreparedRigidJointGraph, /
    ) -> None:
        compatibility = _require_compatible(plan, graph)
        self.plan = plan
        self.graph = graph
        self.compatibility = compatibility
        self.coordinate = _PreparedRigidJointCoordinate(
            graph,
            plan.joint_ids,
            plan.coordinate,
            plan.chart_tolerance,
        )
        self.prepared_id = _prepared_identifier(
            "prepared-rigid-joint-pd-servo", plan.plan_id, graph.prepared_id
        )

    def initialize_state(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointHingeCoordinateState | None:
        return self.coordinate.initialize_state(kinematics)

    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        target_coordinate: ArrayLike,
        target_rate: ArrayLike,
        state: RigidJointHingeCoordinateState | None = None,
        /,
    ) -> RigidJointLawEvaluation:
        """Track runtime targets; hinge targets use the unwrapped coordinate."""

        coordinate, rate, margin, candidate, current = self.coordinate.evaluate(
            kinematics, state
        )
        target = _scalar_command(target_coordinate, self.plan.count, coordinate.dtype)
        target_velocity = _scalar_command(target_rate, self.plan.count, coordinate.dtype)
        scale = self.plan.coordinate_scale.astype(coordinate.dtype)
        coordinate_error = (target - coordinate) / scale
        rate_error = (target_velocity - rate) / scale
        requested = (
            self.plan.proportional_gain * coordinate_error
            + self.plan.derivative_gain * rate_error
        )
        limit = self.plan.effort_limit.astype(coordinate.dtype)
        effort = jnp.clip(requested, -limit, limit)
        saturation_margin = limit - jnp.abs(requested)
        saturated = jnp.abs(requested) > limit
        source_power = jnp.sum(effort * rate)
        load = self.coordinate.load(kinematics, state, effort)
        zero = jnp.zeros((), dtype=coordinate.dtype)
        return _evaluation(
            coordinate,
            rate,
            effort,
            load,
            zero,
            zero,
            source_power,
            saturation_margin,
            margin,
            saturated,
            candidate,
            current,
            self.compatibility,
            self.plan.chart_tolerance,
        )


def _scalar_command(value: ArrayLike, count: int, dtype: DTypeLike, /) -> Array:
    command = jnp.asarray(value, dtype=dtype)
    if command.shape != (count,):
        raise ValueError(f"Scalar joint commands must have shape {(count,)}.")
    return command[:, None]


def _vector_values(
    name: str,
    value: ArrayLike,
    count: int,
    /,
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full((count,), values, dtype=np.float64)
    elif values.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} must not contain NaN.")
    return values


class ScalarJointMotorPlan(StrictModule, NonTrainableState):
    """Stateless motors mapping a command space to reduced generalized effort.

    Motor ``i`` acts on the scalar coordinate of ``joint_ids[i]``. The
    transmission order is fixed: the command is first clipped to
    ``[command_lower, command_upper]``, multiplied by the signed ``gear``,
    and the result is clipped to ``[-effort_limit, effort_limit]``. The
    effort is then added to the joint's generalized-effort component, so a
    positive effort does positive work on a positive coordinate rate. An
    ``effort_limit`` of exactly zero is an explicitly disabled motor that
    produces zero effort for every command; infinite bounds declare an
    unbounded side. Effort units are N m on hinges and N on prismatic joints;
    command units are those implied by ``gear``.
    """

    joint_ids: Array
    gear: Array
    command_lower: Array
    command_upper: Array
    effort_limit: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        gear: ArrayLike,
        /,
        *,
        effort_limit: ArrayLike,
        command_lower: ArrayLike = -np.inf,
        command_upper: ArrayLike = np.inf,
        plan_id: str | None = None,
    ) -> None:
        identifiers = np.asarray(joint_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("joint_ids must be a rank-1 integer array.")
        identifiers = identifiers.astype(np.int64, copy=False)
        count = identifiers.size
        gear_ = _scalar_values("gear", gear, count)
        if np.any(gear_ == 0.0):
            raise ValueError("gear must be nonzero; disable a motor by effort_limit=0.")
        lower = _vector_values("command_lower", command_lower, count)
        upper = _vector_values("command_upper", command_upper, count)
        if np.any(lower >= upper) or np.any(lower == np.inf) or np.any(upper == -np.inf):
            raise ValueError("command_lower must lie below command_upper.")
        limit = _vector_values("effort_limit", effort_limit, count)
        if np.any(limit < 0.0):
            raise ValueError("effort_limit must be nonnegative.")
        generated = canonical_fingerprint(
            {
                "kind": "scalar-joint-motor-plan",
                "values": array_tree_fingerprint(
                    {
                        "joint_ids": identifiers,
                        "gear": gear_,
                        "command_lower": lower,
                        "command_upper": upper,
                        "effort_limit": limit,
                    }
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.joint_ids = jnp.asarray(identifiers)
        self.gear = jnp.asarray(gear_)
        self.command_lower = jnp.asarray(lower)
        self.command_upper = jnp.asarray(upper)
        self.effort_limit = jnp.asarray(limit)
        self.plan_id = identifier

    @property
    def count(self) -> int:
        return self.joint_ids.shape[0]

    def prepare(
        self, articulation: PreparedReducedArticulation, /
    ) -> PreparedScalarJointMotors:
        return PreparedScalarJointMotors(self, articulation)


class ScalarJointMotorEvaluation(StrictModule):
    """Motor efforts, saturation margins, and actuator source power.

    Margins are positive inside the linear transmission range; a negative
    margin identifies the saturated side. ``source_power`` is the exact
    generalized power ``tau . v`` delivered by the motors.
    """

    generalized_effort: Array
    actuator_effort: Array
    clamped_command: Array
    command_margin: Array
    effort_margin: Array
    command_saturated: Array
    effort_saturated: Array
    disabled: Array
    joint_rate: Array
    source_power: Array
    finite: Array
    successful: Array


class PreparedScalarJointMotors(StrictModule, NonTrainableState):
    """Scalar joint motors bound to reduced velocity DOFs by joint ID."""

    plan: ScalarJointMotorPlan
    dof_indices: Array
    hinge_mask: Array
    nv: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: ScalarJointMotorPlan,
        articulation: PreparedReducedArticulation,
        /,
    ) -> None:
        dofs, hinge = reduced_scalar_joint_dofs(
            articulation, np.asarray(plan.joint_ids), "Motor"
        )
        self.plan = plan
        self.dof_indices = jnp.asarray(dofs)
        self.hinge_mask = jnp.asarray(hinge)
        self.nv = articulation.nv
        self.prepared_id = _prepared_identifier(
            "prepared-scalar-joint-motors", plan.plan_id, articulation.prepared_id
        )

    def evaluate(self, command: ArrayLike, velocity: ArrayLike, /) -> ScalarJointMotorEvaluation:
        rate = jnp.asarray(velocity)
        if rate.shape != (self.nv,) or not jnp.issubdtype(rate.dtype, jnp.floating):
            raise ValueError(f"velocity must be a floating vector of shape {(self.nv,)}.")
        requested = jnp.asarray(command, dtype=rate.dtype)
        if requested.shape != (self.plan.count,):
            raise ValueError(f"command must have shape {(self.plan.count,)}.")
        lower = self.plan.command_lower.astype(rate.dtype)
        upper = self.plan.command_upper.astype(rate.dtype)
        limit = self.plan.effort_limit.astype(rate.dtype)
        clamped = jnp.clip(requested, lower, upper)
        transmitted = self.plan.gear.astype(rate.dtype) * clamped
        effort = jnp.clip(transmitted, -limit, limit)
        command_margin = jnp.minimum(requested - lower, upper - requested)
        effort_margin = limit - jnp.abs(transmitted)
        joint_rate = rate[self.dof_indices]
        generalized = jnp.zeros_like(rate).at[self.dof_indices].add(effort)
        source_power = jnp.sum(effort * joint_rate)
        finite = (
            jnp.all(jnp.isfinite(requested))
            & jnp.all(jnp.isfinite(rate))
            & jnp.all(jnp.isfinite(generalized))
            & jnp.isfinite(source_power)
        )
        return ScalarJointMotorEvaluation(
            generalized,
            effort,
            clamped,
            command_margin,
            effort_margin,
            (requested < lower) | (requested > upper),
            jnp.abs(transmitted) > limit,
            limit == 0.0,
            joint_rate,
            source_power,
            finite,
            finite,
        )


class ScalarJointDampingPlan(StrictModule, NonTrainableState):
    """Executed linear viscous damping on reduced scalar joint coordinates.

    The generalized effort is ``-c v`` on each listed joint and the
    dissipation rate ``c v^2`` is nonnegative. Units of ``c`` are N m s/rad on
    hinges and N s/m on prismatic joints.
    """

    joint_ids: Array
    damping: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        damping: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ) -> None:
        identifiers = _joint_identifiers(joint_ids)
        coefficients = _scalar_values(
            "damping", damping, identifiers.size, nonnegative=True
        )
        generated = canonical_fingerprint(
            {
                "kind": "scalar-joint-damping-plan",
                "values": array_tree_fingerprint(
                    {"joint_ids": identifiers, "damping": coefficients}
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.joint_ids = jnp.asarray(identifiers)
        self.damping = jnp.asarray(coefficients)
        self.plan_id = identifier

    @property
    def count(self) -> int:
        return self.joint_ids.shape[0]

    def prepare(
        self, articulation: PreparedReducedArticulation, /
    ) -> PreparedScalarJointDamping:
        return PreparedScalarJointDamping(self, articulation)


class ScalarJointDampingEvaluation(StrictModule):
    generalized_effort: Array
    joint_rate: Array
    dissipation_rate: Array
    finite: Array
    successful: Array


class PreparedScalarJointDamping(StrictModule, NonTrainableState):
    """Viscous joint damping bound to reduced velocity DOFs by joint ID."""

    plan: ScalarJointDampingPlan
    dof_indices: Array
    nv: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: ScalarJointDampingPlan,
        articulation: PreparedReducedArticulation,
        /,
    ) -> None:
        dofs, _ = reduced_scalar_joint_dofs(
            articulation, np.asarray(plan.joint_ids), "Damping"
        )
        self.plan = plan
        self.dof_indices = jnp.asarray(dofs)
        self.nv = articulation.nv
        self.prepared_id = _prepared_identifier(
            "prepared-scalar-joint-damping", plan.plan_id, articulation.prepared_id
        )

    def evaluate(self, velocity: ArrayLike, /) -> ScalarJointDampingEvaluation:
        rate = jnp.asarray(velocity)
        if rate.shape != (self.nv,) or not jnp.issubdtype(rate.dtype, jnp.floating):
            raise ValueError(f"velocity must be a floating vector of shape {(self.nv,)}.")
        coefficients = self.plan.damping.astype(rate.dtype)
        joint_rate = rate[self.dof_indices]
        effort = -coefficients * joint_rate
        generalized = jnp.zeros_like(rate).at[self.dof_indices].add(effort)
        dissipation = jnp.sum(coefficients * joint_rate * joint_rate)
        finite = jnp.all(jnp.isfinite(rate)) & jnp.all(jnp.isfinite(generalized))
        return ScalarJointDampingEvaluation(
            generalized, joint_rate, dissipation, finite, finite
        )


__all__ = [
    "CompliantRigidJointLawPlan",
    "DissipativeRigidJointLawPlan",
    "PreparedCompliantRigidJointLaw",
    "PreparedDissipativeRigidJointLaw",
    "PreparedRigidJointEffortMotor",
    "PreparedRigidJointPDServo",
    "PreparedScalarJointDamping",
    "PreparedScalarJointMotors",
    "RigidJointCoordinate",
    "RigidJointEffortMotorPlan",
    "RigidJointHingeCoordinateState",
    "RigidJointHingeCoordinateUpdate",
    "RigidJointLawCompatibility",
    "RigidJointLawEvaluation",
    "RigidJointLawEvidence",
    "RigidJointPDServoPlan",
    "ScalarJointDampingEvaluation",
    "ScalarJointDampingPlan",
    "ScalarJointMotorEvaluation",
    "ScalarJointMotorPlan",
    "accept_rigid_joint_hinge_coordinate",
    "candidate_rigid_joint_hinge_coordinate",
    "evaluate_rigid_joint_law_compatibility",
    "update_rigid_joint_hinge_coordinate",
]
