#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite
from numbers import Integral
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._reduced_articulation import (
    ArticulationKinematics,
    PreparedReducedArticulation,
)
from ...discretization.particle._rigid_body import quaternion_rotation_matrix
from ...metrix import SpecialOrthogonalGroup
from ...optim import (
    AbstractBoundedLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    Bounds,
    implicit_least_squares,
    least_squares,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)


_SO3 = SpecialOrthogonalGroup(3)


def _nonempty_identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _body_identifier(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("body_id must be an integer.")
    return int(value)




def _floating_vector(value: ArrayLike, size: int, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.asarray(0.0).dtype)
    return array


def _positive_weight(value: ArrayLike, size: int, name: str, /) -> Array:
    array = np.asarray(value)
    if array.shape == ():
        array = np.full((size,), array)
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or have shape ({size},).")
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be numeric.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return jnp.asarray(array)


def _task_bounds(bounds: Bounds | None, tolerance: float, size: int, /) -> Bounds:
    if bounds is None:
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        return Bounds(-tolerance_, tolerance_)
    if not isinstance(bounds, Bounds):
        raise TypeError("bounds must be Bounds or None.")
    lower = np.asarray(bounds.lower)
    upper = np.asarray(bounds.upper)
    if lower.shape not in ((), (size,)) or upper.shape not in ((), (size,)):
        raise ValueError(f"Task bounds must be scalar or have shape ({size},).")
    if not all(
        np.issubdtype(value.dtype, np.floating)
        or np.issubdtype(value.dtype, np.integer)
        for value in (lower, upper)
    ):
        raise TypeError("Task bounds must be real.")
    lower = np.broadcast_to(lower, (size,))
    upper = np.broadcast_to(upper, (size,))
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError("Task bounds must be ordered and cannot contain NaN.")
    return bounds


def _determinant_3x3(matrix: np.ndarray, /) -> float:
    return float(
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def _local_transform(value: ArrayLike | None, /) -> Array:
    transform = np.eye(4) if value is None else np.asarray(value)
    if transform.shape != (4, 4):
        raise ValueError("local_transform must have shape (4, 4).")
    if not (
        np.issubdtype(transform.dtype, np.floating)
        or np.issubdtype(transform.dtype, np.integer)
    ) or not np.all(np.isfinite(transform)):
        raise ValueError("local_transform must be a finite real matrix.")
    rotation = transform[:3, :3]
    orthogonality_error = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    if (
        orthogonality_error > 1.0e-6
        or abs(_determinant_3x3(rotation) - 1.0) > 1.0e-6
        or not np.allclose(transform[3], np.asarray((0.0, 0.0, 0.0, 1.0)))
    ):
        raise ValueError("local_transform must be a homogeneous rigid transform.")
    return jnp.asarray(transform)


def frame_pose_transform(position: ArrayLike, orientation: ArrayLike, /) -> Array:
    """Build a homogeneous transform from position and scalar-first quaternion."""

    position_ = _floating_vector(position, 3, "position")
    orientation_ = _floating_vector(orientation, 4, "orientation")
    dtype = jnp.result_type(position_.dtype, orientation_.dtype)
    transform = jnp.eye(4, dtype=dtype)
    transform = transform.at[:3, :3].set(
        quaternion_rotation_matrix(orientation_.astype(dtype))
    )
    return transform.at[:3, 3].set(position_.astype(dtype))


def _orientation_valid(orientation: Array, /) -> Array:
    norm_squared = jnp.sum(orientation * orientation)
    return jnp.all(jnp.isfinite(orientation)) & (
        norm_squared > jnp.finfo(orientation.dtype).eps
    )


class FramePositionTask(StrictModule, NonTrainableState):
    """Weighted position target for one immutable body-attached frame."""

    body_id: int = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    local_transform: Array
    target_transform: Array
    weight: Array
    bounds: Bounds
    target_valid: Array
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_id: int,
        frame_id: str,
        target_position: ArrayLike,
        /,
        *,
        local_transform: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
        task_id: str,
    ):
        target = _floating_vector(target_position, 3, "target_position")
        transform = jnp.eye(4, dtype=target.dtype).at[:3, 3].set(target)
        self.body_id = _body_identifier(body_id)
        self.frame_id = _nonempty_identifier(frame_id, "frame_id")
        self.local_transform = _local_transform(local_transform)
        self.target_transform = transform
        self.weight = _positive_weight(weight, 3, "weight")
        self.bounds = _task_bounds(bounds, tolerance, 3)
        self.target_valid = jnp.all(jnp.isfinite(target))
        self.task_id = _nonempty_identifier(task_id, "task_id")


class FrameOrientationTask(StrictModule, NonTrainableState):
    """Weighted SO(3) target for one immutable body-attached frame."""

    body_id: int = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    local_transform: Array
    target_transform: Array
    weight: Array
    bounds: Bounds
    target_valid: Array
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_id: int,
        frame_id: str,
        target_orientation: ArrayLike,
        /,
        *,
        local_transform: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
        task_id: str,
    ):
        orientation = _floating_vector(target_orientation, 4, "target_orientation")
        position = jnp.zeros((3,), dtype=orientation.dtype)
        self.body_id = _body_identifier(body_id)
        self.frame_id = _nonempty_identifier(frame_id, "frame_id")
        self.local_transform = _local_transform(local_transform)
        self.target_transform = frame_pose_transform(position, orientation)
        self.weight = _positive_weight(weight, 3, "weight")
        self.bounds = _task_bounds(bounds, tolerance, 3)
        self.target_valid = _orientation_valid(orientation)
        self.task_id = _nonempty_identifier(task_id, "task_id")


class FramePoseTask(StrictModule, NonTrainableState):
    """Weighted position and SO(3) target for a body-attached frame."""

    body_id: int = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    local_transform: Array
    target_transform: Array
    weight: Array
    bounds: Bounds
    target_valid: Array
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        body_id: int,
        frame_id: str,
        target_position: ArrayLike,
        target_orientation: ArrayLike,
        /,
        *,
        local_transform: ArrayLike | None = None,
        position_weight: ArrayLike = 1.0,
        orientation_weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
        task_id: str,
    ):
        position = _floating_vector(target_position, 3, "target_position")
        orientation = _floating_vector(target_orientation, 4, "target_orientation")
        self.body_id = _body_identifier(body_id)
        self.frame_id = _nonempty_identifier(frame_id, "frame_id")
        self.local_transform = _local_transform(local_transform)
        self.target_transform = frame_pose_transform(position, orientation)
        self.weight = jnp.concatenate(
            (
                _positive_weight(position_weight, 3, "position_weight"),
                _positive_weight(orientation_weight, 3, "orientation_weight"),
            )
        )
        self.bounds = _task_bounds(bounds, tolerance, 6)
        self.target_valid = jnp.all(jnp.isfinite(position)) & _orientation_valid(
            orientation
        )
        self.task_id = _nonempty_identifier(task_id, "task_id")


FrameTask: TypeAlias = FramePositionTask | FrameOrientationTask | FramePoseTask


@jax.custom_jvp
def _so3_log_coordinates(rotation: Array, /) -> Array:
    """Principal native SO(3) logarithm with a stable tangent rule at identity."""

    return _SO3.vee(_SO3.log(rotation))


@_so3_log_coordinates.defjvp
def _so3_log_coordinates_jvp(primals, tangents):
    (rotation,), (rotation_tangent,) = primals, tangents
    coordinates = _so3_log_coordinates(rotation)
    body_tangent_matrix = jnp.swapaxes(rotation, -1, -2) @ rotation_tangent
    body_tangent = jnp.stack(
        (
            0.5 * (body_tangent_matrix[2, 1] - body_tangent_matrix[1, 2]),
            0.5 * (body_tangent_matrix[0, 2] - body_tangent_matrix[2, 0]),
            0.5 * (body_tangent_matrix[1, 0] - body_tangent_matrix[0, 1]),
        )
    )
    squared_angle = jnp.sum(coordinates * coordinates)
    angle = jnp.sqrt(jnp.maximum(squared_angle, jnp.finfo(rotation.dtype).eps))
    near_zero = squared_angle < 1.0e-8
    safe_squared_angle = jnp.where(near_zero, 1.0, squared_angle)
    safe_sine = jnp.where(near_zero, 1.0, jnp.sin(angle))
    coefficient = jnp.where(
        near_zero,
        1.0 / 12.0 + squared_angle / 720.0,
        1.0 / safe_squared_angle
        - (1.0 + jnp.cos(angle)) / (2.0 * angle * safe_sine),
    )
    first_cross = jnp.cross(coordinates, body_tangent)
    tangent = (
        body_tangent
        + 0.5 * first_cross
        + coefficient * jnp.cross(coordinates, first_cross)
    )
    return coordinates, tangent


class FrameTaskResidual(StrictModule):
    """Unweighted and weighted residual evidence for one frame task."""

    residual: Array
    weighted_residual: Array
    bound_violation: Array
    rotation_angle: Array
    finite: Array
    chart_valid: Array
    feasible: Array
    body_id: int = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)
    task_kind: str = eqx.field(static=True)


class IKFeasibilityEvidence(StrictModule):
    task_bounds_satisfied: Array
    joint_bounds_satisfied: Array
    maximum_task_violation: Array
    maximum_joint_violation: Array
    feasible: Array


class IKChartEvidence(StrictModule):
    configuration_valid: Array
    task_charts_valid: Array
    configuration_roundtrip_error: Array
    minimum_rotation_margin: Array
    valid: Array


class IKFiniteEvidence(StrictModule):
    initial_configuration: Array
    optimizer_parameters: Array
    optimizer_residual: Array
    final_configuration: Array
    final_kinematics: Array
    task_residuals: Array
    valid: Array


class InverseKinematicsStatus(IntEnum):
    SUCCESS = 0
    OPTIMIZER_FAILED = 1
    INFEASIBLE = 2
    CHART_INVALID = 3
    NONFINITE = 4


class FrameInverseKinematicsResult(StrictModule):
    optimizer: LeastSquaresResult
    configuration: Array
    kinematics: ArticulationKinematics
    task_residuals: tuple[FrameTaskResidual, ...]
    posture_residual: Array
    feasibility: IKFeasibilityEvidence
    chart: IKChartEvidence
    finite: IKFiniteEvidence
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class _FrameIKEvaluation(StrictModule):
    kinematics: ArticulationKinematics
    task_residuals: tuple[FrameTaskResidual, ...]
    posture_residual: Array
    residual: Array


class _LocalFrameIKResidual(StrictModule):
    plan: "FrameInverseKinematicsPlan"
    reference_configuration: Array

    def __call__(self, candidate: Array, args=None, /) -> Array:
        del args
        configuration = self.plan.canonical_configuration(
            self.reference_configuration, candidate
        )
        return self.plan.residual(configuration)


class FrameInverseKinematicsPlan(StrictModule, NonTrainableState):
    """Fixed frame-task residual plan lowered to native nonlinear least squares."""

    articulation: PreparedReducedArticulation
    tasks: tuple[FrameTask, ...]
    posture_configuration: Array | None
    posture_weight: Array | None
    body_slots: tuple[int, ...] = eqx.field(static=True)
    rotation_chart_tolerance: float = eqx.field(static=True)
    configuration_chart_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        articulation: PreparedReducedArticulation,
        tasks: Sequence[FrameTask],
        /,
        *,
        posture_configuration: ArrayLike | None = None,
        posture_weight: ArrayLike = 1.0,
        rotation_chart_tolerance: float = 1.0e-6,
        configuration_chart_tolerance: float = 1.0e-6,
        plan_id: str | None = None,
    ):
        if not isinstance(articulation, PreparedReducedArticulation):
            raise TypeError("articulation must be PreparedReducedArticulation.")
        tasks_ = tuple(tasks)
        if not tasks_ or any(
            not isinstance(
                task, (FramePositionTask, FrameOrientationTask, FramePoseTask)
            )
            for task in tasks_
        ):
            raise TypeError("tasks must be a nonempty sequence of frame tasks.")
        task_ids = tuple(task.task_id for task in tasks_)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Frame IK task IDs must be unique.")
        body_ids = np.asarray(articulation.body_ids)
        body_indices = np.asarray(articulation.body_indices)
        slots = []
        frame_definitions: dict[str, tuple[int, np.ndarray]] = {}
        for task in tasks_:
            matches = np.flatnonzero(body_ids == task.body_id)
            if matches.size != 1:
                raise ValueError(f"Unknown articulation body ID {task.body_id}.")
            slots.append(int(body_indices[int(matches[0])]))
            local = np.asarray(task.local_transform)
            if task.frame_id in frame_definitions:
                previous_body, previous_local = frame_definitions[task.frame_id]
                if previous_body != task.body_id or not np.array_equal(
                    previous_local, local
                ):
                    raise ValueError(
                        "A frame_id must identify one immutable body-local transform."
                    )
            else:
                frame_definitions[task.frame_id] = (task.body_id, local)
        rotation_tolerance = float(rotation_chart_tolerance)
        configuration_tolerance = float(configuration_chart_tolerance)
        if (
            not isfinite(rotation_tolerance)
            or rotation_tolerance <= 0.0
            or rotation_tolerance >= float(np.pi)
            or not isfinite(configuration_tolerance)
            or configuration_tolerance < 0.0
        ):
            raise ValueError("IK chart tolerances are invalid.")
        if posture_configuration is None:
            posture = None
            posture_weight_ = None
        else:
            posture = _floating_vector(
                posture_configuration, articulation.nq, "posture_configuration"
            )
            posture_weight_ = _positive_weight(
                posture_weight, articulation.nv, "posture_weight"
            )
        generated_id = canonical_fingerprint(
            {
                "kind": "frame-inverse-kinematics",
                "articulation": articulation.prepared_id,
                "tasks": task_ids,
                "posture": posture is not None,
            }
        )
        self.articulation = articulation
        self.tasks = tasks_
        self.posture_configuration = posture
        self.posture_weight = posture_weight_
        self.body_slots = tuple(slots)
        self.rotation_chart_tolerance = rotation_tolerance
        self.configuration_chart_tolerance = configuration_tolerance
        self.plan_id = (
            generated_id
            if plan_id is None
            else _nonempty_identifier(plan_id, "plan_id")
        )

    def _require_configuration(self, configuration: ArrayLike, /) -> Array:
        return _floating_vector(
            configuration, self.articulation.nq, "configuration"
        )

    def canonical_configuration(
        self,
        reference_configuration: ArrayLike,
        candidate_configuration: ArrayLike,
        /,
    ) -> Array:
        """Map a candidate through the articulation's local configuration chart."""

        reference = self._require_configuration(reference_configuration)
        candidate = self._require_configuration(candidate_configuration)
        increment = self.articulation.configuration_difference(reference, candidate)
        return self.articulation.integrate_configuration(reference, increment)

    def _rotation_residual(
        self, target: Array, current: Array, target_valid: Array, /
    ) -> tuple[Array, Array, Array]:
        relative = jnp.swapaxes(target, -1, -2) @ current
        rotation_valid = (
            target_valid
            & jnp.all(jnp.isfinite(relative))
            & _SO3.contains(relative)
        )
        cosine = jnp.clip((jnp.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        angle = jnp.arccos(cosine)
        margin = jnp.pi - angle
        chart_valid = rotation_valid & (margin > self.rotation_chart_tolerance)
        safe_rotation = jnp.where(chart_valid, relative, jnp.eye(3, dtype=relative.dtype))
        return _so3_log_coordinates(safe_rotation), chart_valid, angle

    def _task_residual(
        self, task: FrameTask, body_slot: int, body_transforms: Array, /
    ) -> FrameTaskResidual:
        dtype = body_transforms.dtype
        current = body_transforms[body_slot] @ task.local_transform.astype(dtype)
        target = task.target_transform.astype(dtype)
        position_residual = current[:3, 3] - target[:3, 3]
        if isinstance(task, FramePositionTask):
            residual = position_residual
            chart_valid = jnp.asarray(True)
            angle = jnp.asarray(0.0, dtype=dtype)
            task_kind = "position"
        else:
            rotation_residual, chart_valid, angle = self._rotation_residual(
                target[:3, :3], current[:3, :3], task.target_valid
            )
            if isinstance(task, FrameOrientationTask):
                residual = rotation_residual
                task_kind = "orientation"
            else:
                residual = jnp.concatenate((position_residual, rotation_residual))
                task_kind = "pose"
        weight = task.weight.astype(dtype)
        weighted = weight * residual
        finite = (
            task.target_valid
            & jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(weighted))
        )
        violation = task.bounds.violation(residual)
        feasible = finite & chart_valid & task.bounds.contains(residual)
        return FrameTaskResidual(
            residual,
            weighted,
            violation,
            angle,
            finite,
            chart_valid,
            feasible,
            task.body_id,
            task.frame_id,
            task.task_id,
            task_kind,
        )

    def evaluate(self, configuration: ArrayLike, /) -> _FrameIKEvaluation:
        configuration_ = self._require_configuration(configuration)
        kinematics = self.articulation.forward_kinematics(configuration_)
        task_residuals = tuple(
            self._task_residual(task, slot, kinematics.body_transforms)
            for task, slot in zip(self.tasks, self.body_slots, strict=True)
        )
        residual_parts = tuple(value.weighted_residual for value in task_residuals)
        if self.posture_configuration is None:
            posture_residual = jnp.zeros((0,), dtype=configuration_.dtype)
        else:
            posture_weight = self.posture_weight
            assert posture_weight is not None
            difference = self.articulation.configuration_difference(
                self.posture_configuration.astype(configuration_.dtype), configuration_
            )
            posture_residual = posture_weight.astype(configuration_.dtype) * difference
            residual_parts = residual_parts + (posture_residual,)
        return _FrameIKEvaluation(
            kinematics,
            task_residuals,
            posture_residual,
            jnp.concatenate(residual_parts),
        )

    def residual(self, configuration: ArrayLike, args=None, /) -> Array:
        """Evaluate the fixed-capacity weighted IK residual; compatible with JIT/AD."""

        del args
        return self.evaluate(configuration).residual

    def least_squares_problem(
        self,
        reference_configuration: ArrayLike,
        /,
        *,
        joint_bounds: Bounds | None = None,
    ) -> NonlinearLeastSquaresProblem:
        reference = self._require_configuration(reference_configuration)
        if joint_bounds is not None and not isinstance(joint_bounds, Bounds):
            raise TypeError("joint_bounds must be Bounds or None.")
        if joint_bounds is not None:
            joint_bounds.materialize(reference)
        return NonlinearLeastSquaresProblem(
            _LocalFrameIKResidual(self, reference),
            bounds=joint_bounds,
            problem_id=self.plan_id,
        )

    def implicit_solution(
        self,
        initial_configuration: ArrayLike,
        /,
        *,
        method: AbstractLeastSquaresMethod,
        termination: OptimizationTermination,
    ) -> Array:
        """Return the regular local IK root with native implicit sensitivity."""

        if not isinstance(method, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod.")
        if isinstance(method, AbstractBoundedLeastSquaresMethod):
            raise ValueError("Implicit IK sensitivity does not support joint bounds.")
        if not method.capabilities.implicit_differentiation:
            raise ValueError("method does not support implicit differentiation.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination.")
        initial = self._require_configuration(initial_configuration)
        problem = self.least_squares_problem(initial)
        candidate = implicit_least_squares(
            problem,
            initial,
            method=method,
            termination=termination,
        )
        return self.canonical_configuration(initial, candidate)

    def solve(
        self,
        initial_configuration: ArrayLike,
        /,
        *,
        method: AbstractLeastSquaresMethod,
        termination: OptimizationTermination,
        joint_bounds: Bounds | None = None,
    ) -> FrameInverseKinematicsResult:
        """Solve one explicitly configured local least-squares IK problem."""

        if not isinstance(method, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination.")
        bounded_method = isinstance(method, AbstractBoundedLeastSquaresMethod)
        if bounded_method != (joint_bounds is not None):
            raise ValueError(
                "Bounded methods require joint_bounds; unbounded methods require None."
            )
        initial = self._require_configuration(initial_configuration)
        problem = self.least_squares_problem(initial, joint_bounds=joint_bounds)
        optimizer = least_squares(
            problem,
            initial,
            method=method,
            termination=termination,
        )
        configuration = self.canonical_configuration(initial, optimizer.parameters)
        evaluation = self.evaluate(configuration)
        task_feasible = jnp.all(
            jnp.stack(tuple(value.feasible for value in evaluation.task_residuals))
        )
        maximum_task_violation = jnp.max(
            jnp.stack(tuple(value.bound_violation for value in evaluation.task_residuals))
        )
        if joint_bounds is None:
            joint_feasible = jnp.asarray(True)
            joint_violation = jnp.asarray(0.0, dtype=configuration.dtype)
        else:
            joint_feasible = joint_bounds.contains(configuration)
            joint_violation = joint_bounds.violation(configuration)
        feasibility = IKFeasibilityEvidence(
            task_feasible,
            joint_feasible,
            maximum_task_violation,
            joint_violation,
            task_feasible & joint_feasible,
        )
        local_roundtrip_error = jnp.max(
            jnp.abs(configuration - optimizer.parameters)
        )
        if self.posture_configuration is None:
            posture_roundtrip_error = jnp.asarray(
                0.0, dtype=configuration.dtype
            )
        else:
            posture_roundtrip = self.canonical_configuration(
                self.posture_configuration, configuration
            )
            posture_roundtrip_error = jnp.max(
                jnp.abs(posture_roundtrip - configuration)
            )
        roundtrip_error = jnp.maximum(
            local_roundtrip_error, posture_roundtrip_error
        )
        configuration_chart_valid = (
            jnp.isfinite(roundtrip_error)
            & (roundtrip_error <= self.configuration_chart_tolerance)
        )
        task_charts_valid = jnp.all(
            jnp.stack(tuple(value.chart_valid for value in evaluation.task_residuals))
        )
        rotation_margins = tuple(
            jnp.pi - value.rotation_angle
            for value in evaluation.task_residuals
            if value.task_kind != "position"
        )
        minimum_rotation_margin = (
            jnp.asarray(jnp.pi, dtype=configuration.dtype)
            if not rotation_margins
            else jnp.min(jnp.stack(rotation_margins))
        )
        chart = IKChartEvidence(
            configuration_chart_valid,
            task_charts_valid,
            roundtrip_error,
            minimum_rotation_margin,
            configuration_chart_valid & task_charts_valid,
        )
        finite_initial = jnp.all(jnp.isfinite(initial))
        finite_optimizer_parameters = jnp.all(jnp.isfinite(optimizer.parameters))
        finite_optimizer_residual = jnp.all(jnp.isfinite(optimizer.residual)) & jnp.isfinite(
            optimizer.objective
        )
        finite_configuration = jnp.all(jnp.isfinite(configuration))
        finite_kinematics = evaluation.kinematics.finite
        finite_tasks = jnp.all(
            jnp.stack(tuple(value.finite for value in evaluation.task_residuals))
        ) & jnp.all(jnp.isfinite(evaluation.residual))
        finite_valid = (
            finite_initial
            & finite_optimizer_parameters
            & finite_optimizer_residual
            & finite_configuration
            & finite_kinematics
            & finite_tasks
        )
        finite = IKFiniteEvidence(
            finite_initial,
            finite_optimizer_parameters,
            finite_optimizer_residual,
            finite_configuration,
            finite_kinematics,
            finite_tasks,
            finite_valid,
        )
        successful = (
            optimizer.successful & feasibility.feasible & chart.valid & finite.valid
        )
        status = jnp.where(
            ~finite.valid,
            int(InverseKinematicsStatus.NONFINITE),
            jnp.where(
                ~chart.valid,
                int(InverseKinematicsStatus.CHART_INVALID),
                jnp.where(
                    ~optimizer.successful,
                    int(InverseKinematicsStatus.OPTIMIZER_FAILED),
                    jnp.where(
                        ~feasibility.feasible,
                        int(InverseKinematicsStatus.INFEASIBLE),
                        int(InverseKinematicsStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return FrameInverseKinematicsResult(
            optimizer,
            configuration,
            evaluation.kinematics,
            evaluation.task_residuals,
            evaluation.posture_residual,
            feasibility,
            chart,
            finite,
            status,
            successful,
            self.plan_id,
        )


__all__ = [
    "FrameInverseKinematicsPlan",
    "FrameInverseKinematicsResult",
    "FrameOrientationTask",
    "FramePoseTask",
    "FramePositionTask",
    "FrameTask",
    "FrameTaskResidual",
    "IKChartEvidence",
    "IKFeasibilityEvidence",
    "IKFiniteEvidence",
    "InverseKinematicsStatus",
    "frame_pose_transform",
]
