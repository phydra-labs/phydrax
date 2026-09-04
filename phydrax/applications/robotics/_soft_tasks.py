#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...control import (
    AbstractControlParameterization,
    BSplineControlBoundCertificate,
    BSplineControlParameterization,
    ControlProblem,
    ControlResult,
    DiscreteControlDynamics,
)
from ...discretization.particle._rigid_body import quaternion_rotation_matrix
from ...dynamics import (
    AbstractDiscretePlant,
    ControlVectorCodec,
    DiscreteStepContext,
    DiscreteSystem,
    DiscreteTransitionResult,
    EncodedControl,
    EncodedPlantState,
    InputLayout,
    PlantParameters,
    PlantReplayResult,
    PlantRuntimeState,
    PlantStateVectorCodec,
    PlantStepContext,
    TimeGrid,
)
from ...metrix import SpecialOrthogonalGroup, StateChartEvidence
from ...optim import (
    AbstractBoundedLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    Bounds,
    ConvexProgramResult,
    ConvexSolvePolicy,
    implicit_least_squares,
    least_squares,
    LeastSquaresResult,
    MinimizationProblem,
    MinimizationResult,
    minimize,
    NonlinearConstraint,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
    QuadraticProgram,
    solve_quadratic_program,
    SQP,
)
from ..solid_mechanics._rod_reconstruction import (
    _integrate,
    _total_strain,
    PreparedRodReconstruction,
    RodReconstructionEvaluation,
)
from ..solid_mechanics._rod_reduction import ReducedRodState
from ._ik import _so3_log_coordinates


_SO3 = SpecialOrthogonalGroup(3)


def _finite_real_array(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> Array:
    host = np.asarray(value)
    if host.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {host.shape}.")
    if not (
        np.issubdtype(host.dtype, np.floating) or np.issubdtype(host.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be real-valued.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    result = jnp.asarray(host)
    return result if jnp.issubdtype(result.dtype, jnp.inexact) else result.astype(float)


def _positive_weight(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> Array:
    host = np.asarray(value)
    if host.shape == ():
        host = np.broadcast_to(host, shape)
    elif host.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}.")
    if not (
        np.issubdtype(host.dtype, np.floating) or np.issubdtype(host.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be real-valued.")
    if not np.all(np.isfinite(host)) or np.any(host <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return jnp.asarray(host)


def _positive_scalar(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _nonnegative_scalar(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _task_bounds(
    bounds: Bounds | None,
    tolerance: float,
    size: int,
    dtype: np.dtype,
    /,
) -> Bounds:
    if bounds is None:
        tolerance_ = _nonnegative_scalar(tolerance, "tolerance")
        return Bounds(-tolerance_, tolerance_)
    if not isinstance(bounds, Bounds):
        raise TypeError("bounds must be a Bounds or None.")
    probe = jnp.zeros((size,), dtype=dtype)
    lower, upper = bounds.materialize(probe)
    lower_host = np.asarray(lower)
    upper_host = np.asarray(upper)
    if np.any(np.isnan(lower_host)) or np.any(np.isnan(upper_host)):
        raise ValueError("Task bounds cannot contain NaN.")
    if np.any(lower_host > upper_host):
        raise ValueError("Task bounds must be ordered.")
    return bounds


def _canonical_quaternion(value: ArrayLike, name: str, /) -> Array:
    quaternion = np.asarray(value)
    if quaternion.shape != (4,):
        raise ValueError(f"{name} must have shape (4,).")
    if not np.issubdtype(quaternion.dtype, np.number) or np.iscomplexobj(quaternion):
        raise TypeError(f"{name} must be real-valued.")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} must be finite.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.result_type(quaternion.dtype, float)).eps:
        raise ValueError(f"{name} must have nonzero norm.")
    canonical = quaternion / norm
    nonzero = np.flatnonzero(canonical)
    if nonzero.size and canonical[int(nonzero[0])] < 0.0:
        canonical = -canonical
    return jnp.asarray(canonical)


def _query_index(
    reconstruction: PreparedRodReconstruction, arc_length: float, /
) -> tuple[int, float]:
    if not isinstance(reconstruction, PreparedRodReconstruction):
        raise TypeError("reconstruction must be a PreparedRodReconstruction.")
    station = float(arc_length)
    if not isfinite(station):
        raise ValueError("arc_length must be finite.")
    queries = np.asarray(reconstruction.plan.queries.arc_lengths)
    station_value = np.asarray(station, dtype=queries.dtype)
    matches = np.flatnonzero(queries == station_value)
    if matches.size != 1:
        raise ValueError(
            "arc_length must exactly identify one prepared physical frame query."
        )
    index = int(matches[0])
    return index, float(queries[index])


def _bounds_content(bounds: Bounds, size: int, dtype: np.dtype, /) -> dict[str, Any]:
    lower, upper = bounds.materialize(jnp.zeros((size,), dtype=dtype))
    return array_tree_fingerprint(
        {"lower": np.asarray(lower), "upper": np.asarray(upper)}
    )


def _task_identifier(
    kind: str,
    reconstruction: PreparedRodReconstruction,
    payload: dict[str, Any],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": f"continuum-{kind}-task",
            "reconstruction": reconstruction.reconstruction_id,
            "query_plan": reconstruction.plan.queries.plan_id,
            **payload,
        }
    )


class ContinuumPositionTask(StrictModule, NonTrainableState):
    """Position target at one prepared physical material-frame query."""

    target_position: Array
    weight: Array
    bounds: Bounds
    query_index: int = eqx.field(static=True)
    arc_length: float = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedRodReconstruction,
        arc_length: float,
        target_position: ArrayLike,
        /,
        *,
        weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
    ):
        index, station = _query_index(reconstruction, arc_length)
        dtype = reconstruction.reduced.coefficient_space.dtype
        target = _finite_real_array(target_position, (3,), "target_position")
        weight_ = _positive_weight(weight, (3,), "weight")
        bounds_ = _task_bounds(bounds, tolerance, 3, dtype)
        self.target_position = target
        self.weight = weight_
        self.bounds = bounds_
        self.query_index = index
        self.arc_length = station
        self.reconstruction_id = reconstruction.reconstruction_id
        self.task_id = _task_identifier(
            "position",
            reconstruction,
            {
                "query_index": index,
                "arc_length": station,
                "target": array_tree_fingerprint(np.asarray(target)),
                "weight": array_tree_fingerprint(np.asarray(weight_)),
                "bounds": _bounds_content(bounds_, 3, dtype),
            },
        )


class ContinuumOrientationTask(StrictModule, NonTrainableState):
    """Scalar-first quaternion orientation target at one material frame."""

    target_orientation: Array
    weight: Array
    bounds: Bounds
    query_index: int = eqx.field(static=True)
    arc_length: float = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedRodReconstruction,
        arc_length: float,
        target_orientation: ArrayLike,
        /,
        *,
        weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
    ):
        index, station = _query_index(reconstruction, arc_length)
        dtype = reconstruction.reduced.coefficient_space.dtype
        target = _canonical_quaternion(target_orientation, "target_orientation")
        weight_ = _positive_weight(weight, (3,), "weight")
        bounds_ = _task_bounds(bounds, tolerance, 3, dtype)
        self.target_orientation = target
        self.weight = weight_
        self.bounds = bounds_
        self.query_index = index
        self.arc_length = station
        self.reconstruction_id = reconstruction.reconstruction_id
        self.task_id = _task_identifier(
            "orientation",
            reconstruction,
            {
                "query_index": index,
                "arc_length": station,
                "target": array_tree_fingerprint(np.asarray(target)),
                "weight": array_tree_fingerprint(np.asarray(weight_)),
                "bounds": _bounds_content(bounds_, 3, dtype),
            },
        )


class ContinuumPoseTask(StrictModule, NonTrainableState):
    """Position and orientation target at one material frame."""

    target_position: Array
    target_orientation: Array
    weight: Array
    bounds: Bounds
    query_index: int = eqx.field(static=True)
    arc_length: float = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedRodReconstruction,
        arc_length: float,
        target_position: ArrayLike,
        target_orientation: ArrayLike,
        /,
        *,
        position_weight: ArrayLike = 1.0,
        orientation_weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
    ):
        index, station = _query_index(reconstruction, arc_length)
        dtype = reconstruction.reduced.coefficient_space.dtype
        position = _finite_real_array(target_position, (3,), "target_position")
        orientation = _canonical_quaternion(target_orientation, "target_orientation")
        weight_ = jnp.concatenate(
            (
                _positive_weight(position_weight, (3,), "position_weight"),
                _positive_weight(orientation_weight, (3,), "orientation_weight"),
            )
        )
        bounds_ = _task_bounds(bounds, tolerance, 6, dtype)
        self.target_position = position
        self.target_orientation = orientation
        self.weight = weight_
        self.bounds = bounds_
        self.query_index = index
        self.arc_length = station
        self.reconstruction_id = reconstruction.reconstruction_id
        self.task_id = _task_identifier(
            "pose",
            reconstruction,
            {
                "query_index": index,
                "arc_length": station,
                "position": array_tree_fingerprint(np.asarray(position)),
                "orientation": array_tree_fingerprint(np.asarray(orientation)),
                "weight": array_tree_fingerprint(np.asarray(weight_)),
                "bounds": _bounds_content(bounds_, 6, dtype),
            },
        )


class ContinuumShapeTask(StrictModule, NonTrainableState):
    """Centerline-position target at every prepared material-frame query."""

    target_positions: Array
    weight: Array
    bounds: Bounds
    query_count: int = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedRodReconstruction,
        target_positions: ArrayLike,
        /,
        *,
        weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
    ):
        if not isinstance(reconstruction, PreparedRodReconstruction):
            raise TypeError("reconstruction must be a PreparedRodReconstruction.")
        count = reconstruction.plan.queries.query_count
        dtype = reconstruction.reduced.coefficient_space.dtype
        target = _finite_real_array(target_positions, (count, 3), "target_positions")
        weight_ = _positive_weight(weight, (count, 3), "weight")
        size = 3 * count
        bounds_ = _task_bounds(bounds, tolerance, size, dtype)
        self.target_positions = target
        self.weight = weight_
        self.bounds = bounds_
        self.query_count = count
        self.reconstruction_id = reconstruction.reconstruction_id
        self.task_id = _task_identifier(
            "shape",
            reconstruction,
            {
                "target": array_tree_fingerprint(np.asarray(target)),
                "weight": array_tree_fingerprint(np.asarray(weight_)),
                "bounds": _bounds_content(bounds_, size, dtype),
            },
        )


class ContinuumPostureTask(StrictModule, NonTrainableState):
    """Reduced-coordinate posture target in the prepared coefficient space."""

    target_coefficients: Array
    weight: Array
    bounds: Bounds
    coordinate_count: int = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedRodReconstruction,
        target_coefficients: ArrayLike,
        /,
        *,
        weight: ArrayLike = 1.0,
        bounds: Bounds | None = None,
        tolerance: float = 1.0e-6,
    ):
        if not isinstance(reconstruction, PreparedRodReconstruction):
            raise TypeError("reconstruction must be a PreparedRodReconstruction.")
        space = reconstruction.reduced.coefficient_space
        count = space.size
        target = _finite_real_array(target_coefficients, (count,), "target_coefficients")
        weight_ = _positive_weight(weight, (count,), "weight")
        bounds_ = _task_bounds(bounds, tolerance, count, space.dtype)
        self.target_coefficients = target
        self.weight = weight_
        self.bounds = bounds_
        self.coordinate_count = count
        self.reconstruction_id = reconstruction.reconstruction_id
        self.task_id = _task_identifier(
            "posture",
            reconstruction,
            {
                "target": array_tree_fingerprint(np.asarray(target)),
                "weight": array_tree_fingerprint(np.asarray(weight_)),
                "bounds": _bounds_content(bounds_, count, space.dtype),
            },
        )


ContinuumTask: TypeAlias = (
    ContinuumPositionTask
    | ContinuumOrientationTask
    | ContinuumPoseTask
    | ContinuumShapeTask
    | ContinuumPostureTask
)


def _task_size(task: ContinuumTask, /) -> int:
    if isinstance(task, (ContinuumPositionTask, ContinuumOrientationTask)):
        return 3
    if isinstance(task, ContinuumPoseTask):
        return 6
    if isinstance(task, ContinuumShapeTask):
        return 3 * task.query_count
    return task.coordinate_count


class ContinuumTaskResidual(StrictModule):
    """One task's geometric residual and explicit feasibility evidence."""

    residual: Array
    weighted_residual: Array
    bound_violation: Array
    rotation_angle: Array
    finite: Array
    chart_valid: Array
    feasible: Array
    task_id: str = eqx.field(static=True)
    task_kind: str = eqx.field(static=True)


class ContinuumTaskEvaluation(StrictModule):
    """Fixed-query kinematics and every task residual at one coefficient point."""

    coefficients: Array
    poses: Array
    positions: Array
    orientations: Array
    strains: Array
    task_residuals: tuple[ContinuumTaskResidual, ...]
    residual: Array
    unweighted_residual: Array
    maximum_task_violation: Array
    minimum_rotation_margin: Array
    maximum_scaled_local_error: Array
    maximum_increment_angle: Array
    finite: Array
    reconstruction_valid: Array
    chart_valid: Array
    feasible: Array
    reconstruction_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)


class ContinuumIKFeasibilityEvidence(StrictModule):
    task_bounds_satisfied: Array
    coefficient_bounds_satisfied: Array
    native_mechanics_satisfied: Array
    maximum_task_violation: Array
    maximum_coefficient_violation: Array
    feasible: Array


class ContinuumIKChartEvidence(StrictModule):
    reconstruction_valid: Array
    task_charts_valid: Array
    minimum_rotation_margin: Array
    valid: Array
    reconstruction_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)


class ContinuumIKFiniteEvidence(StrictModule):
    source_coefficients_finite: Array
    optimizer_parameters_finite: Array
    optimizer_objective_finite: Array
    candidate_coefficients_finite: Array
    candidate_residual_finite: Array
    accepted_coefficients_finite: Array
    accepted_reconstruction_finite: Array
    valid: Array


class ContinuumIKStatus(IntEnum):
    SUCCESS = 0
    OPTIMIZER_FAILED = 1
    INFEASIBLE = 2
    CHART_INVALID = 3
    NONFINITE = 4


class ContinuumInverseKinematicsResult(StrictModule):
    """Optimizer evidence plus fail-closed source/candidate/accepted rod states."""

    optimizer: LeastSquaresResult | MinimizationResult
    source_state: ReducedRodState
    candidate_state: ReducedRodState
    accepted_state: ReducedRodState
    candidate_evaluation: ContinuumTaskEvaluation
    accepted_evaluation: ContinuumTaskEvaluation
    accepted_reconstruction: RodReconstructionEvaluation
    feasibility: ContinuumIKFeasibilityEvidence
    chart: ContinuumIKChartEvidence
    finite: ContinuumIKFiniteEvidence
    status: Array
    successful: Array
    solver_kind: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _ContinuumIKResidual(StrictModule):
    plan: "ContinuumInverseKinematicsPlan"

    def __call__(self, coefficients: Array, args: Any = None, /) -> Array:
        del args
        return self.plan.residual(coefficients)


class _ContinuumIKObjective(StrictModule):
    plan: "ContinuumInverseKinematicsPlan"

    def __call__(self, coefficients: Array, args: Any = None, /) -> Array:
        del args
        residual = self.plan.residual(coefficients)
        return 0.5 * jnp.vdot(residual, residual).real


class _ContinuumIKConstraint(StrictModule):
    plan: "ContinuumInverseKinematicsPlan"

    def __call__(self, coefficients: Array, args: Any = None, /) -> Array:
        del args
        return self.plan.evaluate(coefficients).unweighted_residual


class ContinuumInverseKinematicsPlan(StrictModule, NonTrainableState):
    """Fixed material-query continuum IK lowered to native NLS or SQP."""

    reconstruction: PreparedRodReconstruction
    tasks: tuple[ContinuumTask, ...]
    rotation_chart_tolerance: float = eqx.field(static=True)
    residual_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedRodReconstruction,
        tasks: Sequence[ContinuumTask],
        /,
        *,
        rotation_chart_tolerance: float = 1.0e-6,
    ):
        if not isinstance(reconstruction, PreparedRodReconstruction):
            raise TypeError("reconstruction must be a PreparedRodReconstruction.")
        tasks_ = tuple(tasks)
        task_types = (
            ContinuumPositionTask,
            ContinuumOrientationTask,
            ContinuumPoseTask,
            ContinuumShapeTask,
            ContinuumPostureTask,
        )
        if not tasks_ or any(not isinstance(task, task_types) for task in tasks_):
            raise TypeError("tasks must be a nonempty sequence of continuum tasks.")
        if any(
            task.reconstruction_id != reconstruction.reconstruction_id for task in tasks_
        ):
            raise ValueError("Every continuum task must bind this reconstruction.")
        task_ids = tuple(task.task_id for task in tasks_)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(
                "Duplicate content-identical continuum tasks are not allowed."
            )
        tolerance = _positive_scalar(rotation_chart_tolerance, "rotation_chart_tolerance")
        if tolerance >= np.pi:
            raise ValueError("rotation_chart_tolerance must be smaller than pi.")
        self.reconstruction = reconstruction
        self.tasks = tasks_
        self.rotation_chart_tolerance = tolerance
        self.residual_size = sum(_task_size(task) for task in tasks_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "continuum-inverse-kinematics-plan",
                "reconstruction": reconstruction.reconstruction_id,
                "route": reconstruction.route_id,
                "tasks": list(task_ids),
                "rotation_chart_tolerance": tolerance,
            }
        )

    @property
    def coordinate_count(self) -> int:
        return self.reconstruction.reduced.coefficient_space.size

    def _coefficients(self, value: ArrayLike, /) -> Array:
        return self.reconstruction.reduced.coefficient_space.validate(jnp.asarray(value))

    def _orientation_residual(
        self, target: Array, current: Array, /
    ) -> tuple[Array, Array, Array]:
        target_rotation = quaternion_rotation_matrix(target)
        current_rotation = quaternion_rotation_matrix(current)
        relative = jnp.swapaxes(target_rotation, -1, -2) @ current_rotation
        finite = jnp.all(jnp.isfinite(relative))
        contained = _SO3.contains(relative)
        cosine = jnp.clip((jnp.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        angle = jnp.arccos(cosine)
        chart_valid = (
            finite & contained & (jnp.pi - angle > self.rotation_chart_tolerance)
        )
        return _so3_log_coordinates(relative), chart_valid, angle

    def _task_residual(
        self,
        task: ContinuumTask,
        coefficients: Array,
        positions: Array,
        orientations: Array,
        /,
    ) -> ContinuumTaskResidual:
        dtype = coefficients.dtype
        rotation_angle = jnp.asarray(0.0, dtype=dtype)
        chart_valid = jnp.asarray(True)
        if isinstance(task, ContinuumPositionTask):
            residual = positions[task.query_index] - task.target_position.astype(dtype)
            weight = task.weight.astype(dtype)
            task_kind = "position"
        elif isinstance(task, ContinuumOrientationTask):
            residual, chart_valid, rotation_angle = self._orientation_residual(
                task.target_orientation.astype(dtype),
                orientations[task.query_index],
            )
            weight = task.weight.astype(dtype)
            task_kind = "orientation"
        elif isinstance(task, ContinuumPoseTask):
            position = positions[task.query_index] - task.target_position.astype(dtype)
            orientation, chart_valid, rotation_angle = self._orientation_residual(
                task.target_orientation.astype(dtype),
                orientations[task.query_index],
            )
            residual = jnp.concatenate((position, orientation))
            weight = task.weight.astype(dtype)
            task_kind = "pose"
        elif isinstance(task, ContinuumShapeTask):
            residual = (positions - task.target_positions.astype(dtype)).reshape((-1,))
            weight = task.weight.astype(dtype).reshape((-1,))
            task_kind = "shape"
        else:
            residual = coefficients - task.target_coefficients.astype(dtype)
            weight = task.weight.astype(dtype)
            task_kind = "posture"
        weighted = weight * residual
        finite = jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.isfinite(weighted))
        violation = task.bounds.violation(residual)
        feasible = finite & chart_valid & task.bounds.contains(residual)
        return ContinuumTaskResidual(
            residual,
            weighted,
            violation,
            rotation_angle,
            finite,
            chart_valid,
            feasible,
            task.task_id,
            task_kind,
        )

    def evaluate(self, coefficients: ArrayLike, /) -> ContinuumTaskEvaluation:
        """Evaluate fixed-work reconstruction and task evidence without changing routes."""

        values = self._coefficients(coefficients)
        all_poses, quadrature_error, maximum_angle, reconstruction_valid = _integrate(
            self.reconstruction, values
        )
        poses = all_poses[self.reconstruction.query_union_indices]
        positions = poses[:, 4:]
        orientations = poses[:, :4]
        strains = _total_strain(
            self.reconstruction,
            values,
            self.reconstruction.plan.queries.arc_lengths,
        )
        task_residuals = tuple(
            self._task_residual(task, values, positions, orientations)
            for task in self.tasks
        )
        residual = jnp.concatenate(
            tuple(result.weighted_residual for result in task_residuals)
        )
        unweighted = jnp.concatenate(tuple(result.residual for result in task_residuals))
        maximum_violation = jnp.max(
            jnp.stack(tuple(result.bound_violation for result in task_residuals))
        )
        rotation_margins = tuple(
            jnp.pi - result.rotation_angle
            for result in task_residuals
            if result.task_kind in ("orientation", "pose")
        )
        minimum_margin = (
            jnp.asarray(jnp.pi, dtype=values.dtype)
            if not rotation_margins
            else jnp.min(jnp.stack(rotation_margins))
        )
        task_finite = jnp.all(
            jnp.stack(tuple(result.finite for result in task_residuals))
        )
        task_charts = jnp.all(
            jnp.stack(tuple(result.chart_valid for result in task_residuals))
        )
        task_feasible = jnp.all(
            jnp.stack(tuple(result.feasible for result in task_residuals))
        )
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(poses))
            & jnp.all(jnp.isfinite(strains))
            & jnp.isfinite(quadrature_error)
            & jnp.isfinite(maximum_angle)
            & task_finite
            & jnp.all(jnp.isfinite(residual))
        )
        chart_valid = reconstruction_valid & task_charts
        return ContinuumTaskEvaluation(
            values,
            poses,
            positions,
            orientations,
            strains,
            task_residuals,
            residual,
            unweighted,
            maximum_violation,
            minimum_margin,
            quadrature_error,
            maximum_angle,
            finite,
            reconstruction_valid,
            chart_valid,
            finite & chart_valid & task_feasible,
            self.reconstruction.reconstruction_id,
            self.reconstruction.route_id,
        )

    def residual(self, coefficients: ArrayLike, args: Any = None, /) -> Array:
        del args
        return self.evaluate(coefficients).residual

    def task_bound_vectors(self, /) -> tuple[Array, Array]:
        dtype = self.reconstruction.reduced.coefficient_space.dtype
        lowers: list[Array] = []
        uppers: list[Array] = []
        for task in self.tasks:
            lower, upper = task.bounds.materialize(
                jnp.zeros((_task_size(task),), dtype=dtype)
            )
            lowers.append(jnp.asarray(lower, dtype=dtype))
            uppers.append(jnp.asarray(upper, dtype=dtype))
        return jnp.concatenate(tuple(lowers)), jnp.concatenate(tuple(uppers))

    def least_squares_problem(
        self, /, *, coefficient_bounds: Bounds | None = None
    ) -> NonlinearLeastSquaresProblem:
        if coefficient_bounds is not None and not isinstance(coefficient_bounds, Bounds):
            raise TypeError("coefficient_bounds must be a Bounds or None.")
        if coefficient_bounds is not None:
            coefficient_bounds.materialize(
                jnp.zeros(
                    (self.coordinate_count,),
                    dtype=self.reconstruction.reduced.coefficient_space.dtype,
                )
            )
        return NonlinearLeastSquaresProblem(
            _ContinuumIKResidual(self),
            bounds=coefficient_bounds,
            problem_id=f"continuum-ik:nls:{self.plan_id}",
        )

    def sqp_problem(
        self, /, *, coefficient_bounds: Bounds | None = None
    ) -> MinimizationProblem:
        if coefficient_bounds is not None and not isinstance(coefficient_bounds, Bounds):
            raise TypeError("coefficient_bounds must be a Bounds or None.")
        lower, upper = self.task_bound_vectors()
        constraint = NonlinearConstraint(
            _ContinuumIKConstraint(self),
            lower=lower,
            upper=upper,
            constraint_id=f"continuum-ik:task-bounds:{self.plan_id}",
        )
        return MinimizationProblem(
            _ContinuumIKObjective(self),
            bounds=coefficient_bounds,
            constraints=(constraint,),
            problem_id=f"continuum-ik:sqp:{self.plan_id}",
        )

    def implicit_solution(
        self,
        initial_coefficients: ArrayLike,
        /,
        *,
        method: AbstractLeastSquaresMethod,
        termination: OptimizationTermination,
    ) -> Array:
        """Differentiate only an unbounded regular local NLS solution."""

        if not isinstance(method, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod.")
        if isinstance(method, AbstractBoundedLeastSquaresMethod):
            raise ValueError("Implicit continuum IK does not support coefficient bounds.")
        if not method.capabilities.implicit_differentiation:
            raise ValueError("method does not support implicit differentiation.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination.")
        initial = self._coefficients(initial_coefficients)
        return implicit_least_squares(
            self.least_squares_problem(),
            initial,
            method=method,
            termination=termination,
        )

    def _result(
        self,
        source: Array,
        optimizer: LeastSquaresResult | MinimizationResult,
        coefficient_bounds: Bounds | None,
        solver_kind: str,
        /,
    ) -> ContinuumInverseKinematicsResult:
        candidate = self._coefficients(optimizer.parameters)
        candidate_evaluation = self.evaluate(candidate)
        task_feasible = jnp.all(
            jnp.stack(
                tuple(value.feasible for value in candidate_evaluation.task_residuals)
            )
        )
        if coefficient_bounds is None:
            coefficient_feasible = jnp.asarray(True)
            coefficient_violation = jnp.asarray(0.0, dtype=candidate.dtype)
        else:
            coefficient_feasible = coefficient_bounds.contains(candidate)
            coefficient_violation = coefficient_bounds.violation(candidate)
        candidate_native = self.reconstruction.reduced.evaluate(
            ReducedRodState(candidate, jnp.zeros_like(candidate))
        )
        feasibility = ContinuumIKFeasibilityEvidence(
            task_feasible,
            coefficient_feasible,
            candidate_native.valid,
            candidate_evaluation.maximum_task_violation,
            coefficient_violation,
            task_feasible & coefficient_feasible & candidate_native.valid,
        )
        task_charts = jnp.all(
            jnp.stack(
                tuple(value.chart_valid for value in candidate_evaluation.task_residuals)
            )
        )
        chart = ContinuumIKChartEvidence(
            candidate_evaluation.reconstruction_valid,
            task_charts,
            candidate_evaluation.minimum_rotation_margin,
            candidate_evaluation.chart_valid,
            self.reconstruction.reconstruction_id,
            self.reconstruction.route_id,
        )
        finite_source = jnp.all(jnp.isfinite(source))
        finite_optimizer = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(
            optimizer.objective
        )
        finite_candidate = candidate_evaluation.finite & candidate_native.finite
        preliminary_finite = finite_source & finite_optimizer & finite_candidate
        preliminary_success = (
            optimizer.successful & feasibility.feasible & chart.valid & preliminary_finite
        )
        accepted = jnp.where(preliminary_success, candidate, source)
        accepted_evaluation = self.evaluate(accepted)
        accepted_state = ReducedRodState(accepted, jnp.zeros_like(accepted))
        accepted_reconstruction = self.reconstruction.evaluate(accepted_state)
        finite_accepted = (
            accepted_evaluation.finite
            & accepted_reconstruction.finite
            & jnp.all(jnp.isfinite(accepted))
        )
        finite = ContinuumIKFiniteEvidence(
            finite_source,
            finite_optimizer,
            jnp.isfinite(optimizer.objective),
            finite_candidate,
            jnp.all(jnp.isfinite(candidate_evaluation.residual)),
            jnp.all(jnp.isfinite(accepted)),
            finite_accepted,
            preliminary_finite & finite_accepted,
        )
        successful = preliminary_success & finite.valid
        status = jnp.where(
            ~finite.valid,
            int(ContinuumIKStatus.NONFINITE),
            jnp.where(
                ~chart.valid,
                int(ContinuumIKStatus.CHART_INVALID),
                jnp.where(
                    ~optimizer.successful,
                    int(ContinuumIKStatus.OPTIMIZER_FAILED),
                    jnp.where(
                        ~feasibility.feasible,
                        int(ContinuumIKStatus.INFEASIBLE),
                        int(ContinuumIKStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return ContinuumInverseKinematicsResult(
            optimizer,
            ReducedRodState(source, jnp.zeros_like(source)),
            ReducedRodState(candidate, jnp.zeros_like(candidate)),
            accepted_state,
            candidate_evaluation,
            accepted_evaluation,
            accepted_reconstruction,
            feasibility,
            chart,
            finite,
            status,
            successful,
            solver_kind,
            self.plan_id,
        )

    def solve_least_squares(
        self,
        initial_coefficients: ArrayLike,
        /,
        *,
        method: AbstractLeastSquaresMethod,
        termination: OptimizationTermination,
        coefficient_bounds: Bounds | None = None,
    ) -> ContinuumInverseKinematicsResult:
        if not isinstance(method, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination.")
        bounded = isinstance(method, AbstractBoundedLeastSquaresMethod)
        if bounded != (coefficient_bounds is not None):
            raise ValueError(
                "Bounded methods require coefficient_bounds; unbounded methods require None."
            )
        source = self._coefficients(initial_coefficients)
        self.reconstruction.evaluate(ReducedRodState(source, jnp.zeros_like(source)))
        optimizer = least_squares(
            self.least_squares_problem(coefficient_bounds=coefficient_bounds),
            source,
            method=method,
            termination=termination,
        )
        return self._result(source, optimizer, coefficient_bounds, "nls")

    def solve_sqp(
        self,
        initial_coefficients: ArrayLike,
        /,
        *,
        method: SQP,
        termination: OptimizationTermination,
        coefficient_bounds: Bounds | None = None,
    ) -> ContinuumInverseKinematicsResult:
        if not isinstance(method, SQP):
            raise TypeError("method must be SQP.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination.")
        source = self._coefficients(initial_coefficients)
        self.reconstruction.evaluate(ReducedRodState(source, jnp.zeros_like(source)))
        optimizer = minimize(
            self.sqp_problem(coefficient_bounds=coefficient_bounds),
            source,
            method=method,
            termination=termination,
        )
        return self._result(source, optimizer, coefficient_bounds, "sqp")


class DifferentialIKCompilation(StrictModule):
    """Materialized task linearization and its native convex QP."""

    program: QuadraticProgram
    coefficients: Array
    weighted_residual: Array
    task_jacobian: Array
    desired_task_rate: Array
    velocity_reference: Array
    one_step_lower_velocity: Array
    one_step_upper_velocity: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class DifferentialIKBoundsEvidence(StrictModule):
    velocity_bounds_satisfied: Array
    coefficient_bounds_satisfied: Array
    maximum_velocity_violation: Array
    maximum_coefficient_violation: Array
    feasible: Array


class ContinuumDifferentialIKResult(StrictModule):
    """QP result with source/candidate/accepted reduced velocities and states."""

    optimizer: ConvexProgramResult
    compilation: DifferentialIKCompilation
    source_state: ReducedRodState
    candidate_state: ReducedRodState
    accepted_state: ReducedRodState
    candidate_velocity: Array
    accepted_velocity: Array
    achieved_task_rate: Array
    desired_task_rate: Array
    reduced_task_effort: Array
    candidate_evaluation: ContinuumTaskEvaluation
    accepted_reconstruction: RodReconstructionEvaluation
    bounds: DifferentialIKBoundsEvidence
    chart_valid: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ContinuumDifferentialIKPlan(StrictModule, NonTrainableState):
    """Local continuum resolved-rate IK compiled to the native convex QP."""

    inverse_kinematics: ContinuumInverseKinematicsPlan
    correction_gain: float = eqx.field(static=True)
    velocity_regularization: Array
    time_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        inverse_kinematics: ContinuumInverseKinematicsPlan,
        /,
        *,
        correction_gain: float = 1.0,
        velocity_regularization: ArrayLike = 1.0e-8,
        time_step: float = 1.0,
    ):
        if not isinstance(inverse_kinematics, ContinuumInverseKinematicsPlan):
            raise TypeError(
                "inverse_kinematics must be a ContinuumInverseKinematicsPlan."
            )
        gain = _nonnegative_scalar(correction_gain, "correction_gain")
        step = _positive_scalar(time_step, "time_step")
        count = inverse_kinematics.coordinate_count
        regularization = _positive_weight(
            velocity_regularization,
            (count,),
            "velocity_regularization",
        )
        self.inverse_kinematics = inverse_kinematics
        self.correction_gain = gain
        self.velocity_regularization = regularization
        self.time_step = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "continuum-differential-ik-plan",
                "inverse_kinematics": inverse_kinematics.plan_id,
                "correction_gain": gain,
                "velocity_regularization": array_tree_fingerprint(
                    np.asarray(regularization)
                ),
                "time_step": step,
            }
        )

    def compile(
        self,
        coefficients: ArrayLike,
        /,
        *,
        velocity_reference: ArrayLike | None = None,
        velocity_bounds: Bounds | None = None,
        coefficient_bounds: Bounds | None = None,
    ) -> DifferentialIKCompilation:
        ik = self.inverse_kinematics
        values = ik._coefficients(coefficients)
        count = ik.coordinate_count
        reference = (
            jnp.zeros_like(values)
            if velocity_reference is None
            else ik._coefficients(velocity_reference)
        )
        evaluation = ik.evaluate(values)
        jacobian = jax.jacrev(ik.residual)(values)
        desired = -self.correction_gain * evaluation.residual
        regularization = self.velocity_regularization.astype(values.dtype)
        quadratic = jacobian.T @ jacobian + jnp.diag(regularization)
        linear = -(jacobian.T @ desired) - regularization * reference
        if velocity_bounds is not None and not isinstance(velocity_bounds, Bounds):
            raise TypeError("velocity_bounds must be a Bounds or None.")
        native_bounds = Bounds() if velocity_bounds is None else velocity_bounds
        native_bounds.materialize(values)
        if coefficient_bounds is None:
            one_step_lower = jnp.full_like(values, -jnp.inf)
            one_step_upper = jnp.full_like(values, jnp.inf)
            inequality_matrix = None
            inequality_rhs = None
        else:
            if not isinstance(coefficient_bounds, Bounds):
                raise TypeError("coefficient_bounds must be a Bounds or None.")
            lower, upper = coefficient_bounds.materialize(values)
            one_step_lower = (lower - values) / self.time_step
            one_step_upper = (upper - values) / self.time_step
            identity = jnp.eye(count, dtype=values.dtype)
            lower_indices = np.flatnonzero(np.isfinite(np.asarray(lower)))
            upper_indices = np.flatnonzero(np.isfinite(np.asarray(upper)))
            lower_rows = -identity[jnp.asarray(lower_indices, dtype=jnp.int32)]
            upper_rows = identity[jnp.asarray(upper_indices, dtype=jnp.int32)]
            inequality_matrix = jnp.concatenate((lower_rows, upper_rows), axis=0)
            inequality_rhs = jnp.concatenate(
                (
                    -one_step_lower[jnp.asarray(lower_indices, dtype=jnp.int32)],
                    one_step_upper[jnp.asarray(upper_indices, dtype=jnp.int32)],
                ),
                axis=0,
            )
        program = QuadraticProgram(
            quadratic,
            linear,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            bounds=native_bounds,
            problem_id=f"continuum-differential-ik:{self.plan_id}",
            convexity_evidence=(
                "J.T@J plus strictly-positive diagonal velocity regularization"
            ),
        )
        finite = (
            evaluation.finite
            & jnp.all(jnp.isfinite(jacobian))
            & jnp.all(jnp.isfinite(quadratic))
            & jnp.all(jnp.isfinite(linear))
            & jnp.all(jnp.isfinite(reference))
        )
        return DifferentialIKCompilation(
            program,
            values,
            evaluation.residual,
            jacobian,
            desired,
            reference,
            one_step_lower,
            one_step_upper,
            finite,
            self.plan_id,
        )

    def solve(
        self,
        coefficients: ArrayLike,
        /,
        *,
        velocity_reference: ArrayLike | None = None,
        velocity_bounds: Bounds | None = None,
        coefficient_bounds: Bounds | None = None,
        policy: ConvexSolvePolicy | None = None,
    ) -> ContinuumDifferentialIKResult:
        compilation = self.compile(
            coefficients,
            velocity_reference=velocity_reference,
            velocity_bounds=velocity_bounds,
            coefficient_bounds=coefficient_bounds,
        )
        optimizer = solve_quadratic_program(compilation.program, policy=policy)
        candidate_velocity = optimizer.primal
        source_coefficients = compilation.coefficients
        candidate_coefficients = source_coefficients + self.time_step * candidate_velocity
        candidate_evaluation = self.inverse_kinematics.evaluate(candidate_coefficients)
        candidate_native = self.inverse_kinematics.reconstruction.reduced.evaluate(
            ReducedRodState(candidate_coefficients, candidate_velocity)
        )
        if velocity_bounds is None:
            velocity_feasible = jnp.asarray(True)
            velocity_violation = jnp.asarray(0.0, dtype=candidate_velocity.dtype)
        else:
            velocity_feasible = velocity_bounds.contains(candidate_velocity)
            velocity_violation = velocity_bounds.violation(candidate_velocity)
        if coefficient_bounds is None:
            coefficient_feasible = jnp.asarray(True)
            coefficient_violation = jnp.asarray(0.0, dtype=candidate_velocity.dtype)
        else:
            coefficient_feasible = coefficient_bounds.contains(candidate_coefficients)
            coefficient_violation = coefficient_bounds.violation(candidate_coefficients)
        bounds = DifferentialIKBoundsEvidence(
            velocity_feasible,
            coefficient_feasible,
            velocity_violation,
            coefficient_violation,
            velocity_feasible & coefficient_feasible,
        )
        achieved_rate = compilation.task_jacobian @ candidate_velocity
        tracking_error = achieved_rate - compilation.desired_task_rate
        reduced_effort = (
            self.inverse_kinematics.reconstruction.reduced.reduced_effort_space.validate(
                compilation.task_jacobian.T @ tracking_error
            )
        )
        finite = (
            compilation.finite
            & jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.all(jnp.isfinite(candidate_coefficients))
            & jnp.all(jnp.isfinite(achieved_rate))
            & jnp.all(jnp.isfinite(reduced_effort))
            & candidate_evaluation.finite
            & candidate_native.finite
        )
        successful = (
            optimizer.successful
            & bounds.feasible
            & candidate_evaluation.chart_valid
            & candidate_native.valid
            & finite
        )
        accepted_velocity = jnp.where(
            successful, candidate_velocity, jnp.zeros_like(candidate_velocity)
        )
        accepted_coefficients = jnp.where(
            successful, candidate_coefficients, source_coefficients
        )
        accepted_state = ReducedRodState(accepted_coefficients, accepted_velocity)
        accepted_reconstruction = self.inverse_kinematics.reconstruction.evaluate(
            accepted_state
        )
        return ContinuumDifferentialIKResult(
            optimizer,
            compilation,
            ReducedRodState(source_coefficients, jnp.zeros_like(source_coefficients)),
            ReducedRodState(candidate_coefficients, candidate_velocity),
            accepted_state,
            candidate_velocity,
            accepted_velocity,
            achieved_rate,
            compilation.desired_task_rate,
            reduced_effort,
            candidate_evaluation,
            accepted_reconstruction,
            bounds,
            candidate_evaluation.chart_valid,
            finite & accepted_reconstruction.finite,
            successful & accepted_reconstruction.valid,
            self.plan_id,
        )


def _encoded_state(
    codec: PlantStateVectorCodec,
    anchor: EncodedPlantState,
    vector: Array,
    /,
) -> EncodedPlantState:
    return codec.replace_point_vector(anchor, vector)


def _encoded_control(codec: ControlVectorCodec, vector: Array, /) -> EncodedControl:
    return EncodedControl(
        vector,
        semantic_id=codec.semantic_id,
        numeric_revision_id=codec.numeric_revision_id,
        schema_id=codec.schema_id,
        executable_signature_id=codec.executable_signature_id,
        codec_id=codec.codec_id,
    )


def _advance_key(key: Array, step_count: Array, /) -> Array:
    typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
    current = key if typed else jax.random.wrap_key_data(key)

    def advance(_index, value):
        return jax.random.split(value, 2)[0]

    advanced = jax.lax.fori_loop(0, step_count, advance, current)
    return advanced if typed else jax.random.key_data(advanced)


class _ControlledPlantTransition(StrictModule):
    plant: AbstractDiscretePlant
    state_codec: PlantStateVectorCodec
    control_codec: ControlVectorCodec
    parameters: PlantParameters
    fixed_mode_anchor: EncodedPlantState
    initial_key: Array
    initial_step_index: Array

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        control: Array,
        args: Any = None,
        /,
    ) -> DiscreteTransitionResult:
        del args
        payload = self.state_codec.decode_point(
            _encoded_state(self.state_codec, self.fixed_mode_anchor, state)
        )
        step_index = self.initial_step_index + context.step_index
        source = PlantRuntimeState(
            payload,
            context.source,
            step_index,
            _advance_key(self.initial_key, context.step_index),
            self.plant.semantic_provenance.semantic_id,
            self.plant.numeric_revision.revision_id,
            self.plant.state_schema.schema_id,
            self.plant.execution_signature.signature_id,
        )
        commands = self.control_codec.decode_command(
            _encoded_control(self.control_codec, control)
        )
        result = self.plant.step(
            PlantStepContext(context.source, context.target, step_index),
            source,
            commands,
            self.parameters,
        )
        candidate = self.state_codec.encode_point(result.candidate_state.payload).vector
        accepted = self.state_codec.encode_point(result.accepted_state.payload).vector
        return DiscreteTransitionResult(
            candidate, accepted, result.successful, result.status
        )


class _ContinuumTerminalCost(StrictModule):
    trajectory: "SmoothReducedRodTrajectoryPlan"

    def __call__(self, time: Array, state: Array, args: Any = None, /) -> Array:
        del time, args
        residual = self.trajectory.inverse_kinematics.residual(
            self.trajectory.coefficients_from_vector(state)
        )
        return 0.5 * jnp.vdot(residual, residual).real


class _ContinuumTerminalViolation(StrictModule):
    trajectory: "SmoothReducedRodTrajectoryPlan"

    def __call__(self, time: Array, state: Array, args: Any = None, /) -> Array:
        del time, args
        evaluation = self.trajectory.inverse_kinematics.evaluate(
            self.trajectory.coefficients_from_vector(state)
        )
        return evaluation.maximum_task_violation


class _ControlEffortCost(StrictModule):
    weight: Array

    def __call__(
        self,
        time: Array,
        state: Array,
        control: Array,
        args: Any = None,
        /,
    ) -> Array:
        del time, state, args
        return 0.5 * self.weight.astype(control.dtype) * jnp.vdot(control, control).real


class _TrajectorySQPObjective(StrictModule):
    problem: ControlProblem
    parameterization: AbstractControlParameterization

    def __call__(self, coefficients: Array, args: Any = None, /) -> Array:
        del args
        return self.problem.evaluate(
            self.parameterization, coefficients
        ).sampled_loss.total


class _TrajectorySQPConstraint(StrictModule):
    trajectory: "SmoothReducedRodTrajectoryPlan"
    problem: ControlProblem
    parameterization: AbstractControlParameterization

    def __call__(self, coefficients: Array, args: Any = None, /) -> Array:
        del args
        rollout = self.problem.rollout(self.parameterization, coefficients)
        return self.trajectory.inverse_kinematics.evaluate(
            self.trajectory.coefficients_from_vector(rollout.final_state)
        ).unweighted_residual


class SmoothReducedRodReplay(StrictModule):
    """Exact accepted plant replay and codec/chart/task evidence at every node."""

    replay: PlantReplayResult
    encoded_states: Array
    task_evaluations: tuple[ContinuumTaskEvaluation, ...]
    chart_evidence: tuple[StateChartEvidence, ...]
    chart_valid: Array
    final_task_feasible: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)


class SmoothReducedRodTrajectoryStatus(IntEnum):
    SUCCESS = 0
    OPTIMIZER_FAILED = 1
    CANDIDATE_FAILED = 2
    REPLAY_FAILED = 3
    REPLAY_MISMATCH = 4
    INFEASIBLE = 5
    CHART_INVALID = 6
    NONFINITE = 7


class SmoothReducedRodTrajectoryResult(StrictModule):
    """SQP candidate and authoritative accepted plant replay."""

    optimizer: MinimizationResult
    candidate: ControlResult
    replay: SmoothReducedRodReplay
    replay_matches_candidate: Array
    command_vectors: Array
    control_bounds: BSplineControlBoundCertificate | None
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SmoothReducedRodTrajectoryPlan(StrictModule, NonTrainableState):
    """Fixed-base/contact-free plant trajectory lowering with accepted replay.

    Controlled plants lower smooth B-spline commands through ``ControlProblem``
    and SQP. Autonomous passive plants intentionally expose accepted replay only;
    they have no fabricated zero-width optimization decision.
    """

    plant: AbstractDiscretePlant
    state_codec: PlantStateVectorCodec
    control_codec: ControlVectorCodec | None
    parameters: PlantParameters
    inverse_kinematics: ContinuumInverseKinematicsPlan
    time_grid: TimeGrid
    running_control_weight: Array
    coefficient_offset: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    profile: Literal["passive", "tendon"] = eqx.field(static=True)
    controlled: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plant: AbstractDiscretePlant,
        state_codec: PlantStateVectorCodec,
        parameters: PlantParameters,
        inverse_kinematics: ContinuumInverseKinematicsPlan,
        time_grid: TimeGrid,
        /,
        *,
        control_codec: ControlVectorCodec | None = None,
        profile: Literal["passive", "tendon"],
        running_control_weight: float = 1.0e-6,
    ):
        if not isinstance(plant, AbstractDiscretePlant):
            raise TypeError("plant must be an AbstractDiscretePlant.")
        if not isinstance(state_codec, PlantStateVectorCodec):
            raise TypeError("state_codec must be a PlantStateVectorCodec.")
        if not isinstance(parameters, PlantParameters):
            raise TypeError("parameters must be PlantParameters.")
        if not isinstance(inverse_kinematics, ContinuumInverseKinematicsPlan):
            raise TypeError(
                "inverse_kinematics must be a ContinuumInverseKinematicsPlan."
            )
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if parameters.schema_id != plant.parameter_schema.schema_id:
            raise ValueError("parameters must bind the plant parameter schema.")
        if parameters.numeric_revision.revision_id != plant.numeric_revision.revision_id:
            raise ValueError("parameters must bind the plant numeric revision.")
        plant.parameter_schema.validate(parameters.values)
        if plant.state_schema.case_ndim != 0:
            raise ValueError(
                "Smooth reduced-rod trajectories require an unbatched plant."
            )
        if state_codec.schema_id != plant.state_schema.schema_id:
            raise ValueError("state_codec must bind the plant state schema.")
        if state_codec.semantic_id != plant.semantic_provenance.semantic_id:
            raise ValueError("state_codec must bind the plant semantic provenance.")
        if state_codec.numeric_revision_id != plant.numeric_revision.revision_id:
            raise ValueError("state_codec must bind the plant numeric revision.")
        if state_codec.executable_signature_id != plant.execution_signature.signature_id:
            raise ValueError("state_codec must bind the plant executable signature.")
        topology_ids = dict(plant.execution_signature.topology_ids)
        if topology_ids.get("reduction") != (
            inverse_kinematics.reconstruction.reduced.prepared_id
        ):
            raise ValueError(
                "Plant executable reduction does not match the task reconstruction."
            )
        controlled = plant.control_schema is not None
        if profile not in ("passive", "tendon"):
            raise ValueError("profile must be 'passive' or 'tendon'.")
        if (profile == "passive") != (not controlled):
            raise ValueError(
                "Passive profiles require an autonomous plant; tendon profiles "
                "require a controlled plant."
            )
        if controlled:
            if not isinstance(control_codec, ControlVectorCodec):
                raise TypeError("Controlled plants require a ControlVectorCodec.")
            if control_codec.schema.case_ndim != 0 or control_codec.size < 1:
                raise ValueError(
                    "Controlled trajectories require a nonempty unbatched control schema."
                )
            if control_codec.schema_id != plant.control_schema.schema_id:
                raise ValueError("control_codec must bind the plant control schema.")
            if control_codec.semantic_id != state_codec.semantic_id:
                raise ValueError("State and control codecs must bind one semantics.")
            if control_codec.numeric_revision_id != state_codec.numeric_revision_id:
                raise ValueError(
                    "State and control codecs must bind one numeric revision."
                )
            if (
                control_codec.executable_signature_id
                != state_codec.executable_signature_id
            ):
                raise ValueError("State and control codecs must bind one executable.")
        elif control_codec is not None:
            raise ValueError("Passive plants do not accept a ControlVectorCodec.")
        contact_leaves = tuple(
            leaf
            for leaf in state_codec.schema.leaves
            if leaf.path.endswith(".contact_state.values")
        )
        if len(contact_leaves) != 1 or contact_leaves[0].shape != (0,):
            raise ValueError(
                "Smooth reduced-rod trajectories require explicit zero-width contact state."
            )
        coordinate_count = inverse_kinematics.coordinate_count
        reduced_matches = tuple(
            index
            for index in state_codec.dynamic_leaf_indices
            if state_codec.schema.leaves[index].path.endswith(".reduced_state.values")
        )
        if len(reduced_matches) != 1:
            raise ValueError(
                "Plant codec must expose one dynamic reduced_state.values leaf."
            )
        reduced_index = reduced_matches[0]
        reduced_leaf = state_codec.schema.leaves[reduced_index]
        if reduced_leaf.shape != (2 * coordinate_count,):
            raise ValueError(
                "Plant reduced-state leaf does not match the reconstruction coefficient space."
            )
        offset = 0
        for index in state_codec.dynamic_leaf_indices:
            if index == reduced_index:
                break
            offset += prod(state_codec.schema.leaves[index].shape)
        weight = _positive_scalar(running_control_weight, "running_control_weight")
        self.plant = plant
        self.state_codec = state_codec
        self.control_codec = control_codec
        self.parameters = parameters
        self.inverse_kinematics = inverse_kinematics
        self.time_grid = time_grid
        self.running_control_weight = jnp.asarray(weight)
        self.profile = profile
        self.coefficient_offset = offset
        self.coefficient_count = coordinate_count
        self.controlled = controlled
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-fixed-base-contact-free-reduced-rod-trajectory",
                "plant_semantics": plant.semantic_provenance.semantic_id,
                "plant_revision": plant.numeric_revision.revision_id,
                "plant_executable": plant.execution_signature.signature_id,
                "profile": profile,
                "state_codec": state_codec.codec_id,
                "control_codec": None
                if control_codec is None
                else control_codec.codec_id,
                "inverse_kinematics": inverse_kinematics.plan_id,
                "time_grid": time_grid.time_id,
                "running_control_weight": weight,
            }
        )

    def coefficients_from_vector(self, state_vector: ArrayLike, /) -> Array:
        vector = jnp.asarray(state_vector)
        if vector.shape != self.state_codec.layout.shape:
            raise ValueError(
                f"state_vector must have shape {self.state_codec.layout.shape}."
            )
        flat = vector.reshape((-1,))
        start = self.coefficient_offset
        return flat[start : start + self.coefficient_count]

    def _initial_point(self, initial_state: PlantRuntimeState, /) -> EncodedPlantState:
        if not isinstance(initial_state, PlantRuntimeState):
            raise TypeError("initial_state must be a PlantRuntimeState.")
        self.plant.checkpoint(initial_state)
        if initial_state.time.shape != () or initial_state.step_index.shape != ():
            raise ValueError("Trajectory initial runtime metadata must be scalar.")
        if not np.array_equal(
            np.asarray(initial_state.time), np.asarray(self.time_grid.times[0])
        ):
            raise ValueError("initial_state time must equal the trajectory initial time.")
        return self.state_codec.encode_point(initial_state.payload)

    def control_problem(self, initial_state: PlantRuntimeState, /) -> ControlProblem:
        """Build the existing discrete geometry-aware control problem."""

        initial_point = self._initial_point(initial_state)
        initial = initial_point.vector
        if not self.controlled:
            raise ValueError(
                "A passive plant has no optimization control; use accepted_replay."
            )
        control_codec = self.control_codec
        assert control_codec is not None
        transition = _ControlledPlantTransition(
            self.plant,
            self.state_codec,
            control_codec,
            self.parameters,
            initial_point,
            initial_state.key,
            initial_state.step_index,
        )
        system = DiscreteSystem(
            transition,
            state_layout=self.state_codec.layout,
            input_layout=InputLayout(
                (control_codec.size,),
                layout_id=f"input-layout:reduced-rod-trajectory:{control_codec.codec_id}",
            ),
            system_id=f"controlled-plant-vector:{self.plan_id}",
        )
        dynamics = DiscreteControlDynamics(
            system,
            method_id="accepted-discrete-plant-through-state-control-codecs",
        )
        return ControlProblem(
            dynamics,
            self.time_grid,
            initial,
            running_cost=_ControlEffortCost(self.running_control_weight),
            terminal_cost=_ContinuumTerminalCost(self),
            terminal_constraints=(_ContinuumTerminalViolation(self),),
            problem_id=f"smooth-reduced-rod-control:{self.plan_id}",
        )

    def sqp_problem(
        self,
        initial_state: PlantRuntimeState,
        parameterization: BSplineControlParameterization,
        /,
        *,
        control_bounds: Bounds | None = None,
    ) -> MinimizationProblem:
        if not isinstance(parameterization, BSplineControlParameterization):
            raise TypeError(
                "Smooth trajectory SQP requires BSplineControlParameterization."
            )
        problem = self.control_problem(initial_state)
        if parameterization.control_shape != problem.control_shape:
            raise ValueError("parameterization control shape does not match the plant.")
        if control_bounds is not None and not isinstance(control_bounds, Bounds):
            raise TypeError("control_bounds must be a Bounds or None.")
        lower, upper = self.inverse_kinematics.task_bound_vectors()
        terminal_constraint = NonlinearConstraint(
            _TrajectorySQPConstraint(self, problem, parameterization),
            lower=lower,
            upper=upper,
            constraint_id=f"smooth-reduced-rod-terminal-task:{self.plan_id}",
        )
        return MinimizationProblem(
            _TrajectorySQPObjective(problem, parameterization),
            bounds=control_bounds,
            constraints=(terminal_constraint,),
            problem_id=f"smooth-reduced-rod-trajectory-sqp:{self.plan_id}",
        )

    def accepted_replay(
        self,
        initial_state: PlantRuntimeState,
        command_vectors: ArrayLike | None = None,
        /,
    ) -> SmoothReducedRodReplay:
        """Replay only plant-accepted payloads and audit each codec chart."""

        self._initial_point(initial_state)
        step_count = self.time_grid.num_steps
        if self.controlled:
            control_codec = self.control_codec
            assert control_codec is not None
            if command_vectors is None:
                raise ValueError("Controlled plant replay requires command_vectors.")
            vectors = jnp.asarray(command_vectors)
            expected = (step_count, control_codec.size)
            if vectors.shape != expected:
                raise ValueError(f"command_vectors must have shape {expected}.")
            commands = tuple(
                control_codec.decode_command(_encoded_control(control_codec, vector))
                for vector in vectors
            )
        else:
            if command_vectors is not None:
                raise ValueError("Passive plant replay does not accept command_vectors.")
            commands = (None,) * step_count
        contexts = tuple(
            PlantStepContext(
                self.time_grid.times[index],
                self.time_grid.times[index + 1],
                initial_state.step_index + index,
            )
            for index in range(step_count)
        )
        replay = self.plant.replay(
            self.plant.checkpoint(initial_state),
            contexts,
            commands,
            self.parameters,
        )
        encoded = tuple(
            self.state_codec.encode_point(state.payload).vector
            for state in replay.accepted_states
        )
        encoded_states = jnp.stack(encoded)
        evaluations = tuple(
            self.inverse_kinematics.evaluate(self.coefficients_from_vector(vector))
            for vector in encoded
        )
        geometry = self.state_codec.layout.geometry
        charts = []
        for source, target in zip(encoded[:-1], encoded[1:], strict=True):
            local = jnp.asarray(geometry.inverse_retract(source, target))
            charts.append(
                geometry.chart_evidence(
                    source,
                    local,
                    local,
                    jnp.zeros_like(target),
                )
            )
        chart_evidence = tuple(charts)
        chart_valid = (
            jnp.asarray(True)
            if not chart_evidence
            else jnp.all(jnp.stack(tuple(evidence.valid for evidence in chart_evidence)))
        )
        final_feasible = evaluations[-1].feasible
        finite = jnp.all(jnp.isfinite(encoded_states)) & jnp.all(
            jnp.stack(tuple(value.finite for value in evaluations))
        )
        successful = replay.successful & chart_valid & final_feasible & finite
        return SmoothReducedRodReplay(
            replay,
            encoded_states,
            evaluations,
            chart_evidence,
            chart_valid,
            final_feasible,
            finite,
            successful,
            self.plan_id,
            self.state_codec.codec_id,
        )

    def solve_sqp(
        self,
        initial_state: PlantRuntimeState,
        parameterization: BSplineControlParameterization,
        initial_coefficients: ArrayLike,
        /,
        *,
        method: SQP,
        termination: OptimizationTermination,
        control_bounds: Bounds | None = None,
    ) -> SmoothReducedRodTrajectoryResult:
        if not isinstance(method, SQP):
            raise TypeError("method must be SQP.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination.")
        initial = jnp.asarray(initial_coefficients)
        if initial.shape != parameterization.parameter_shape:
            raise ValueError(
                "initial_coefficients must have the parameterization parameter shape."
            )
        optimizer = minimize(
            self.sqp_problem(
                initial_state,
                parameterization,
                control_bounds=control_bounds,
            ),
            initial,
            method=method,
            termination=termination,
        )
        problem = self.control_problem(initial_state)
        candidate = problem.evaluate(parameterization, optimizer.parameters)
        commands = parameterization.sample(
            optimizer.parameters, self.time_grid.times[:-1]
        )
        replay = self.accepted_replay(initial_state, commands)
        replay_matches_candidate = jnp.array_equal(
            candidate.trajectory.states, replay.encoded_states
        )
        if control_bounds is None:
            bound_certificate = None
            control_bounds_valid = jnp.asarray(True)
        else:
            control_codec = self.control_codec
            assert control_codec is not None
            lower, upper = control_bounds.materialize(
                jnp.zeros((control_codec.size,), dtype=commands.dtype)
            )
            bound_certificate = parameterization.bound_certificate(
                optimizer.parameters, lower, upper
            )
            control_bounds_valid = bound_certificate.certified
        finite = (
            jnp.isfinite(optimizer.objective)
            & jnp.all(jnp.isfinite(commands))
            & replay.finite
        )
        successful = (
            optimizer.successful
            & candidate.successful
            & replay.successful
            & replay_matches_candidate
            & control_bounds_valid
            & finite
        )
        status = jnp.where(
            ~finite,
            int(SmoothReducedRodTrajectoryStatus.NONFINITE),
            jnp.where(
                ~replay.chart_valid,
                int(SmoothReducedRodTrajectoryStatus.CHART_INVALID),
                jnp.where(
                    ~optimizer.successful,
                    int(SmoothReducedRodTrajectoryStatus.OPTIMIZER_FAILED),
                    jnp.where(
                        ~candidate.successful,
                        int(SmoothReducedRodTrajectoryStatus.CANDIDATE_FAILED),
                        jnp.where(
                            ~replay.replay.successful,
                            int(SmoothReducedRodTrajectoryStatus.REPLAY_FAILED),
                            jnp.where(
                                ~replay_matches_candidate,
                                int(SmoothReducedRodTrajectoryStatus.REPLAY_MISMATCH),
                                jnp.where(
                                    ~(replay.final_task_feasible & control_bounds_valid),
                                    int(SmoothReducedRodTrajectoryStatus.INFEASIBLE),
                                    int(SmoothReducedRodTrajectoryStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return SmoothReducedRodTrajectoryResult(
            optimizer,
            candidate,
            replay,
            replay_matches_candidate,
            commands,
            bound_certificate,
            status,
            successful,
            self.plan_id,
        )


__all__ = [
    "ContinuumDifferentialIKPlan",
    "ContinuumDifferentialIKResult",
    "ContinuumIKChartEvidence",
    "ContinuumIKFeasibilityEvidence",
    "ContinuumIKFiniteEvidence",
    "ContinuumIKStatus",
    "ContinuumInverseKinematicsPlan",
    "ContinuumInverseKinematicsResult",
    "ContinuumOrientationTask",
    "ContinuumPoseTask",
    "ContinuumPositionTask",
    "ContinuumPostureTask",
    "ContinuumShapeTask",
    "ContinuumTask",
    "ContinuumTaskEvaluation",
    "ContinuumTaskResidual",
    "DifferentialIKBoundsEvidence",
    "DifferentialIKCompilation",
    "SmoothReducedRodReplay",
    "SmoothReducedRodTrajectoryPlan",
    "SmoothReducedRodTrajectoryResult",
    "SmoothReducedRodTrajectoryStatus",
]
