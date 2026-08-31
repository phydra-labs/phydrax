#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    RankPolicy,
    solve,
)
from ...optim import (
    AbstractRobustLoss,
    HuberLoss,
    least_squares,
    LeastSquaresResult,
    OptimizationStatus,
    OptimizationTermination,
)
from ._model import CameraModel, project_points
from ._rig import CameraRig


CAMERA_PARAMETER_COUNT = 16
CalibrationGauge = Literal["world-points", "reference-camera"]


class CameraCalibrationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    INSUFFICIENT_OBSERVATIONS = 2
    UNOBSERVABLE = 3
    OPTIMIZATION_FAILED = 4
    NONFINITE_RESULT = 5


class CameraCalibrationProblem(StrictModule, NonTrainableState):
    """Fixed-capacity world-point/pixel correspondences for one camera rig."""

    initial_rig: CameraRig
    world_points: Array
    observed_pixels: Array
    observation_valid: Array
    observation_weights: Array
    holdout: Array
    observation_capacity: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_rig: CameraRig,
        world_points: ArrayLike,
        observed_pixels: ArrayLike,
        observation_valid: ArrayLike,
        *,
        observation_weights: ArrayLike | None = None,
        holdout: ArrayLike | None = None,
    ):
        if not isinstance(initial_rig, CameraRig):
            raise TypeError("initial_rig must be a CameraRig.")
        points_host = np.asarray(world_points, dtype=float)
        pixels_host = np.asarray(observed_pixels, dtype=float)
        valid_host = np.asarray(observation_valid, dtype=bool)
        if points_host.ndim != 2 or points_host.shape[1:] != (3,):
            raise ValueError("world_points must have shape (observations, 3).")
        observations = int(points_host.shape[0])
        if observations < 1:
            raise ValueError("Calibration observation capacity must be positive.")
        expected_pixels = (initial_rig.capacity, observations, 2)
        expected_mask = (initial_rig.capacity, observations)
        if pixels_host.shape != expected_pixels:
            raise ValueError(f"observed_pixels must have shape {expected_pixels}.")
        if valid_host.shape != expected_mask:
            raise ValueError(f"observation_valid must have shape {expected_mask}.")
        if observation_weights is None:
            weights_host = np.ones(expected_mask, dtype=float)
        else:
            weights_host = np.asarray(observation_weights, dtype=float)
            if weights_host.shape != expected_mask:
                raise ValueError(f"observation_weights must have shape {expected_mask}.")
        if holdout is None:
            holdout_host = np.zeros(expected_mask, dtype=bool)
        else:
            holdout_host = np.asarray(holdout, dtype=bool)
            if holdout_host.shape != expected_mask:
                raise ValueError(f"holdout must have shape {expected_mask}.")
        rig_valid = np.asarray(initial_rig.camera_valid, dtype=bool)[:, None]
        if np.any(valid_host & ~rig_valid):
            raise ValueError("Inactive cameras cannot own valid observations.")
        if np.any(holdout_host & ~valid_host):
            raise ValueError("Holdout entries must be valid observations.")
        active_points = np.any(valid_host, axis=0)
        if np.any(~np.isfinite(points_host[active_points])):
            raise ValueError("Active world points must be finite.")
        if np.any(~np.isfinite(pixels_host[valid_host])):
            raise ValueError("Valid observed pixels must be finite.")
        if np.any(~np.isfinite(weights_host[valid_host])) or np.any(
            weights_host[valid_host] <= 0.0
        ):
            raise ValueError("Valid observation weights must be finite and positive.")
        safe_points = np.where(active_points[:, None], points_host, 0.0)
        safe_pixels = np.where(valid_host[..., None], pixels_host, 0.0)
        safe_weights = np.where(valid_host, weights_host, 0.0)
        self.initial_rig = initial_rig
        self.world_points = jnp.asarray(safe_points)
        self.observed_pixels = jnp.asarray(safe_pixels)
        self.observation_valid = jnp.asarray(valid_host)
        self.observation_weights = jnp.asarray(safe_weights)
        self.holdout = jnp.asarray(holdout_host)
        self.observation_capacity = observations
        self.problem_id = canonical_fingerprint(
            {
                "kind": "camera-calibration-problem",
                "rig": initial_rig.rig_id,
                "observations": observations,
                "data": array_tree_fingerprint(
                    (safe_points, safe_pixels, valid_host, safe_weights, holdout_host)
                ),
            }
        )


class CameraCalibrationPlan(StrictModule, NonTrainableState):
    """Gauge, free-parameter, robust-loss, and observability policy."""

    free_parameter_mask: Array
    robust_loss: AbstractRobustLoss
    camera_capacity: int = eqx.field(static=True)
    free_parameter_indices: tuple[int, ...] = eqx.field(static=True)
    gauge: CalibrationGauge = eqx.field(static=True)
    reference_camera: int | None = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        free_parameter_mask: ArrayLike,
        *,
        gauge: CalibrationGauge = "world-points",
        reference_camera: int | None = None,
        robust_loss: AbstractRobustLoss | None = None,
        maximum_steps: int = 64,
        rank_tolerance: float = 1e-8,
        maximum_condition: float = 1e10,
    ):
        mask_host = np.asarray(free_parameter_mask, dtype=bool)
        if mask_host.ndim != 2 or mask_host.shape[1:] != (CAMERA_PARAMETER_COUNT,):
            raise ValueError("free_parameter_mask must have shape (camera_capacity, 16).")
        camera_capacity = int(mask_host.shape[0])
        if camera_capacity < 1 or not np.any(mask_host):
            raise ValueError("At least one calibration parameter must be free.")
        if gauge not in ("world-points", "reference-camera"):
            raise ValueError("gauge must be 'world-points' or 'reference-camera'.")
        if gauge == "reference-camera":
            if reference_camera is None:
                raise ValueError("reference-camera gauge requires reference_camera.")
            reference = int(reference_camera)
            if reference < 0 or reference >= camera_capacity:
                raise ValueError("reference_camera is outside camera capacity.")
            if np.any(mask_host[reference, 10:16]):
                raise ValueError("The reference camera pose must be fixed by the gauge.")
        else:
            if reference_camera is not None:
                raise ValueError(
                    "reference_camera is only valid for reference-camera gauge."
                )
            reference = None
        loss = HuberLoss(1.0) if robust_loss is None else robust_loss
        if not isinstance(loss, AbstractRobustLoss):
            raise TypeError("robust_loss must be an AbstractRobustLoss or None.")
        maximum_steps_ = int(maximum_steps)
        if maximum_steps_ < 1:
            raise ValueError("maximum_steps must be positive.")
        if not math.isfinite(rank_tolerance) or rank_tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive.")
        if not math.isfinite(maximum_condition) or maximum_condition <= 1.0:
            raise ValueError("maximum_condition must be finite and greater than one.")
        flat_mask = mask_host.reshape(-1)
        indices = tuple(int(index) for index in np.flatnonzero(flat_mask))
        self.free_parameter_mask = jnp.asarray(mask_host)
        self.robust_loss = loss
        self.camera_capacity = camera_capacity
        self.free_parameter_indices = indices
        self.gauge = gauge
        self.reference_camera = reference
        self.maximum_steps = maximum_steps_
        self.rank_tolerance = float(rank_tolerance)
        self.maximum_condition = float(maximum_condition)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "camera-calibration-plan",
                "free_parameter_mask": mask_host.tolist(),
                "gauge": gauge,
                "reference_camera": reference,
                "robust_loss": loss.loss_id,
                "maximum_steps": maximum_steps_,
                "rank_tolerance": rank_tolerance,
                "maximum_condition": maximum_condition,
            }
        )


class CameraCalibrationDiagnostics(StrictModule, NonTrainableState):
    training_rms: Array
    holdout_rms: Array
    per_camera_training_rms: Array
    per_camera_holdout_rms: Array
    training_count: Array
    holdout_count: Array
    rank: Array
    condition_number: Array
    singular_values: Array
    observable: Array


class CameraCalibrationResult(StrictModule, NonTrainableState):
    rig: CameraRig
    parameter_delta: Array
    free_parameter_mask: Array
    covariance: Array
    valid: Array
    status: Array
    diagnostics: CameraCalibrationDiagnostics
    optimization: LeastSquaresResult | None


def _skew(vector: Array) -> Array:
    zero = jnp.zeros((), dtype=vector.dtype)
    x, y, z = vector[0], vector[1], vector[2]
    return jnp.stack(
        (
            jnp.stack((zero, -z, y)),
            jnp.stack((z, zero, -x)),
            jnp.stack((-y, x, zero)),
        )
    )


def _rotation_increment(rotation_vector: Array) -> Array:
    angle_squared = jnp.sum(rotation_vector * rotation_vector)
    safe_angle_squared = jnp.maximum(
        angle_squared,
        jnp.finfo(rotation_vector.dtype).tiny,
    )
    safe_angle = jnp.sqrt(safe_angle_squared)
    first_exact = jnp.sin(safe_angle) / safe_angle
    second_exact = (1.0 - jnp.cos(safe_angle)) / safe_angle_squared
    first_series = 1.0 - angle_squared / 6.0 + angle_squared**2 / 120.0
    second_series = 0.5 - angle_squared / 24.0 + angle_squared**2 / 720.0
    use_series = angle_squared < 1e-8
    first = jnp.where(use_series, first_series, first_exact)
    second = jnp.where(use_series, second_series, second_exact)
    skew = _skew(rotation_vector)
    return (
        jnp.eye(3, dtype=rotation_vector.dtype)
        + first * skew
        + second * contract("ij,jk->ik", skew, skew)
    )


def _camera_from_delta(camera: CameraModel, delta: Array) -> CameraModel:
    intrinsics = eqx.tree_at(
        lambda value: (
            value.focal_length,
            value.principal_point,
            value.skew,
        ),
        camera.intrinsics,
        (
            camera.intrinsics.focal_length * jnp.exp(delta[0:2]),
            camera.intrinsics.principal_point + delta[2:4],
            camera.intrinsics.skew + delta[4],
        ),
    )
    distortion = eqx.tree_at(
        lambda value: (value.radial, value.tangential),
        camera.distortion,
        (
            camera.distortion.radial + delta[5:8],
            camera.distortion.tangential + delta[8:10],
        ),
    )
    frame = camera.pose.frame
    rotation = contract(
        "ij,jk->ik",
        _rotation_increment(delta[10:13]),
        frame.rotation,
    )
    transformed_frame = eqx.tree_at(
        lambda value: (value.rotation, value.translation),
        frame,
        (rotation, frame.translation + delta[13:16]),
    )
    pose = eqx.tree_at(lambda value: value.frame, camera.pose, transformed_frame)
    return eqx.tree_at(
        lambda value: (value.intrinsics, value.distortion, value.pose),
        camera,
        (intrinsics, distortion, pose),
    )


def _rig_from_delta(
    rig: CameraRig,
    parameter_delta: Array,
    free_parameter_mask: Array,
) -> CameraRig:
    delta = parameter_delta.reshape((rig.capacity, CAMERA_PARAMETER_COUNT))
    delta = jnp.where(free_parameter_mask, delta, 0.0)
    cameras = tuple(
        _camera_from_delta(camera, delta[index])
        for index, camera in enumerate(rig.cameras)
    )
    return eqx.tree_at(lambda value: value.cameras, rig, cameras)


def _pixel_residuals(
    problem: CameraCalibrationProblem,
    plan: CameraCalibrationPlan,
    parameter_delta: Array,
) -> tuple[Array, CameraRig]:
    rig = _rig_from_delta(
        problem.initial_rig,
        parameter_delta,
        plan.free_parameter_mask,
    )
    predictions = jnp.stack(
        tuple(
            project_points(camera, problem.world_points).pixels for camera in rig.cameras
        ),
        axis=0,
    )
    return predictions - problem.observed_pixels, rig


def _training_residual(
    problem: CameraCalibrationProblem,
    plan: CameraCalibrationPlan,
    parameter_delta: Array,
) -> Array:
    pixel_residual, _ = _pixel_residuals(problem, plan, parameter_delta)
    training = problem.observation_valid & ~problem.holdout
    squared = jnp.sum(pixel_residual * pixel_residual, axis=-1)
    evaluation = plan.robust_loss.evaluate(squared)
    denominator = jnp.maximum(squared, jnp.finfo(pixel_residual.dtype).tiny)
    factor = jnp.sqrt(jnp.maximum(evaluation.rho, 0.0) / denominator)
    factor = jnp.where(
        squared > 0.0,
        factor,
        jnp.sqrt(jnp.maximum(evaluation.first, 0.0)),
    )
    weighted = (
        jnp.sqrt(problem.observation_weights)[..., None]
        * factor[..., None]
        * pixel_residual
    )
    return jnp.where(training[..., None], weighted, 0.0).reshape(-1)


def _masked_rms(residual: Array, mask: Array) -> Array:
    count = jnp.sum(mask).astype(jnp.int32)
    squared = jnp.sum(residual * residual, axis=-1)
    value = jnp.sqrt(jnp.sum(jnp.where(mask, squared, 0.0)) / jnp.maximum(count, 1))
    return jnp.where(count > 0, value, jnp.nan)


def _calibration_evidence(
    problem: CameraCalibrationProblem,
    plan: CameraCalibrationPlan,
    parameter_delta: Array,
) -> tuple[CameraCalibrationDiagnostics, Array]:
    residual, _ = _pixel_residuals(problem, plan, parameter_delta)
    training = problem.observation_valid & ~problem.holdout
    holdout = problem.observation_valid & problem.holdout
    training_count = jnp.sum(training).astype(jnp.int32)
    holdout_count = jnp.sum(holdout).astype(jnp.int32)
    per_camera_training = jnp.stack(
        tuple(
            _masked_rms(residual[index], training[index])
            for index in range(problem.initial_rig.capacity)
        )
    )
    per_camera_holdout = jnp.stack(
        tuple(
            _masked_rms(residual[index], holdout[index])
            for index in range(problem.initial_rig.capacity)
        )
    )
    training_rms = _masked_rms(residual, training)
    holdout_rms = _masked_rms(residual, holdout)

    def raw_training(candidate):
        raw, _ = _pixel_residuals(problem, plan, candidate)
        return jnp.where(training[..., None], raw, 0.0).reshape(-1)

    jacobian = jax.jacrev(raw_training)(parameter_delta)
    flat_weights = jnp.repeat(
        jnp.where(training, problem.observation_weights, 0.0).reshape(-1),
        2,
    )
    free_jacobian = jacobian[:, plan.free_parameter_indices]
    weighted_jacobian = jnp.sqrt(flat_weights)[:, None] * free_jacobian
    row_count = int(weighted_jacobian.shape[0])
    linear = solve(
        LeastSquaresProblem(DenseLinearOperator(weighted_jacobian)),
        jnp.eye(row_count, dtype=weighted_jacobian.dtype),
        policy=LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(relative_cutoff=plan.rank_tolerance),
        ),
    )
    singular_values = linear.diagnostics.singular_values
    assert singular_values is not None
    rank = jnp.asarray(linear.diagnostics.rank).reshape(-1)[0]
    condition = jnp.asarray(linear.diagnostics.condition_estimate).reshape(-1)[0]
    free_count = len(plan.free_parameter_indices)
    equation_count = 2 * training_count
    observable = (
        (equation_count >= free_count)
        & (rank == free_count)
        & jnp.isfinite(condition)
        & (condition <= plan.maximum_condition)
    )
    pseudoinverse = linear.value
    weighted_residual = _training_residual(problem, plan, parameter_delta)
    degrees_of_freedom = jnp.maximum(equation_count - free_count, 1)
    residual_variance = (
        jnp.sum(weighted_residual * weighted_residual) / degrees_of_freedom
    )
    free_covariance = residual_variance * contract(
        "ik,jk->ij",
        pseudoinverse,
        pseudoinverse,
    )
    total_parameters = problem.initial_rig.capacity * CAMERA_PARAMETER_COUNT
    covariance = jnp.zeros(
        (total_parameters, total_parameters),
        dtype=free_covariance.dtype,
    )
    indices = jnp.asarray(plan.free_parameter_indices, dtype=jnp.int32)
    covariance = covariance.at[indices[:, None], indices[None, :]].set(free_covariance)
    covariance = jnp.where(observable, covariance, jnp.nan)
    diagnostics = CameraCalibrationDiagnostics(
        training_rms,
        holdout_rms,
        per_camera_training,
        per_camera_holdout,
        training_count,
        holdout_count,
        rank,
        condition,
        singular_values,
        observable,
    )
    return diagnostics, covariance


def calibrate_camera_rig(
    problem: CameraCalibrationProblem,
    plan: CameraCalibrationPlan,
    /,
) -> CameraCalibrationResult:
    """Calibrate all selected rig parameters with native robust least squares."""

    if not isinstance(problem, CameraCalibrationProblem):
        raise TypeError("problem must be a CameraCalibrationProblem.")
    if not isinstance(plan, CameraCalibrationPlan):
        raise TypeError("plan must be a CameraCalibrationPlan.")
    if problem.initial_rig.capacity != plan.camera_capacity:
        raise ValueError("Calibration plan and rig capacities differ.")
    inactive = ~np.asarray(problem.initial_rig.camera_valid, dtype=bool)
    if np.any(np.asarray(plan.free_parameter_mask)[inactive]):
        raise ValueError("Inactive cameras cannot have free parameters.")
    initial = jnp.zeros(
        (problem.initial_rig.capacity * CAMERA_PARAMETER_COUNT,),
        dtype=problem.world_points.dtype,
    )
    initial_diagnostics, initial_covariance = _calibration_evidence(
        problem,
        plan,
        initial,
    )
    free_count = len(plan.free_parameter_indices)
    if int(np.asarray(initial_diagnostics.training_count)) * 2 < free_count:
        return CameraCalibrationResult(
            problem.initial_rig,
            initial.reshape((problem.initial_rig.capacity, CAMERA_PARAMETER_COUNT)),
            plan.free_parameter_mask,
            initial_covariance,
            jnp.asarray(False),
            jnp.asarray(
                int(CameraCalibrationStatus.INSUFFICIENT_OBSERVATIONS),
                dtype=jnp.int32,
            ),
            initial_diagnostics,
            None,
        )
    if not bool(np.asarray(initial_diagnostics.observable)):
        return CameraCalibrationResult(
            problem.initial_rig,
            initial.reshape((problem.initial_rig.capacity, CAMERA_PARAMETER_COUNT)),
            plan.free_parameter_mask,
            initial_covariance,
            jnp.asarray(False),
            jnp.asarray(int(CameraCalibrationStatus.UNOBSERVABLE), dtype=jnp.int32),
            initial_diagnostics,
            None,
        )

    def residual_function(candidate, _):
        return _training_residual(problem, plan, candidate)

    optimization = least_squares(
        residual_function,
        initial,
        termination=OptimizationTermination(maximum_steps=plan.maximum_steps),
    )
    parameter_delta = jnp.where(
        plan.free_parameter_mask.reshape(-1),
        optimization.parameters,
        0.0,
    )
    diagnostics, covariance = _calibration_evidence(
        problem,
        plan,
        parameter_delta,
    )
    rig = _rig_from_delta(
        problem.initial_rig,
        parameter_delta,
        plan.free_parameter_mask,
    )
    finite = jnp.all(jnp.isfinite(parameter_delta)) & jnp.isfinite(
        diagnostics.training_rms
    )
    optimization_success = optimization.status == int(OptimizationStatus.SUCCESS)
    valid = optimization_success & diagnostics.observable & finite
    status = jnp.where(
        ~finite,
        int(CameraCalibrationStatus.NONFINITE_RESULT),
        jnp.where(
            ~diagnostics.observable,
            int(CameraCalibrationStatus.UNOBSERVABLE),
            jnp.where(
                ~optimization_success,
                int(CameraCalibrationStatus.OPTIMIZATION_FAILED),
                int(CameraCalibrationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return CameraCalibrationResult(
        rig,
        parameter_delta.reshape((problem.initial_rig.capacity, CAMERA_PARAMETER_COUNT)),
        plan.free_parameter_mask,
        covariance,
        valid,
        status,
        diagnostics,
        optimization,
    )


__all__ = [
    "CAMERA_PARAMETER_COUNT",
    "CalibrationGauge",
    "CameraCalibrationDiagnostics",
    "CameraCalibrationPlan",
    "CameraCalibrationProblem",
    "CameraCalibrationResult",
    "CameraCalibrationStatus",
    "calibrate_camera_rig",
]
