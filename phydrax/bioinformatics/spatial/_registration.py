#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import coordax as cx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from opt_einsum import contract

from ... import linalg as la
from ..._strict import StrictModule
from ...integration import discrete
from ...transport import discrete_problem, Sinkhorn, SquaredEuclideanCost
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._frame import AffineSpatialTransform, SpatialFrame


class RegistrationStatus(IntEnum):
    OK = 0
    TRANSPORT_NOT_CONVERGED = 1
    OUTER_NOT_CONVERGED = 2
    DEGENERATE_GEOMETRY = 3


@dataclass(frozen=True, slots=True)
class SpatialRegistrationPlan:
    """Host convergence and regularization plan for rigid OT registration."""

    epsilon: float = 0.05
    outer_iterations: int = 20
    sinkhorn_iterations: int = 500
    transform_tolerance: float = 1.0e-5
    sinkhorn_tolerance: float = 1.0e-7
    rank_tolerance: float = 1.0e-6

    def __post_init__(self):
        for name in (
            "epsilon",
            "transform_tolerance",
            "sinkhorn_tolerance",
            "rank_tolerance",
        ):
            value = float(object.__getattribute__(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        for name in ("outer_iterations", "sinkhorn_iterations"):
            value = int(object.__getattribute__(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)


class RegistrationUncertainty(StrictModule):
    residual_variance: Array
    translation_standard_error: Array
    coupling_effective_size: Array
    rotation_identifiability: Array
    valid: Array


class RegistrationEvidence(StrictModule):
    iterations: Array
    transform_residual: Array
    transport_marginal_residual: Array
    singular_values: Array
    transport_converged: Array


class SpatialRegistrationResult(StrictModule):
    transform: AffineSpatialTransform
    aligned_coordinates: Array
    coupling: Array
    uncertainty: RegistrationUncertainty
    valid: Array
    status: Array
    evidence: RegistrationEvidence
    method_contract: BioinformaticsMethodContract


def _registration_contract(
    plan: SpatialRegistrationPlan, /
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "entropic_ot_rigid_spatial_registration",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.ITERATIVE_TOLERANCE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "The rigid alignment is conditional on normalized source and target masses "
            "and the chosen entropic regularization strength."
        ),
        truncation_statement=(
            "The full dense finite coupling is retained; iteration limits are reported "
            "as non-convergence rather than as a converged alignment."
        ),
        capacity_semantics=(
            "Source-target coupling capacity is the complete Cartesian product of the "
            "supplied point sets."
        ),
        assumptions=(
            "Point clouds are expressed in explicit frames with equal dimension.",
            "A rigid transform is scientifically appropriate for the sections.",
        ),
        nondifferentiable_outputs=(
            "transform",
            "aligned_coordinates",
            "coupling",
            "uncertainty",
            "status",
            "evidence",
            "valid",
        ),
        absolute_tolerance=plan.transform_tolerance,
        relative_tolerance=plan.sinkhorn_tolerance,
    )


def _normalized_weights(name: str, value: Any | None, count: int, dtype, /) -> Array:
    weights = (
        jnp.ones((count,), dtype=dtype)
        if value is None
        else jnp.asarray(value, dtype=dtype)
    )
    if weights.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}.")
    host = np.asarray(weights)
    if np.any(~np.isfinite(host)) or np.any(host < 0.0) or float(np.sum(host)) <= 0.0:
        raise ValueError(f"{name} must be finite, non-negative, and have positive mass.")
    return weights / jnp.sum(weights)


def _determinant_2_or_3(matrix: Array, /) -> Array:
    if int(matrix.shape[0]) == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    return jnp.sum(matrix[0] * jnp.cross(matrix[1], matrix[2]))


def _rigid_update(
    moving: Array,
    fixed: Array,
    coupling: Array,
    source_weights: Array,
    target_weights: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    source_center = jnp.sum(source_weights[:, None] * moving, axis=0)
    target_center = jnp.sum(target_weights[:, None] * fixed, axis=0)
    source_centered = moving - source_center
    target_centered = fixed - target_center
    covariance = contract(
        "ij,ik,jl->kl",
        coupling,
        source_centered,
        target_centered,
    )
    dimension = int(moving.shape[1])
    decomposition = la.svd.svd(
        la.svd.SVDProblem(la.DenseLinearOperator(covariance)),
        policy=la.svd.SVDSolvePolicy(count=dimension),
    )
    left = jnp.asarray(decomposition.left_vectors)
    right = jnp.asarray(decomposition.right_vectors)
    provisional = left @ right.T
    orientation = _determinant_2_or_3(provisional)
    correction = (
        jnp.eye(dimension, dtype=moving.dtype)
        .at[-1, -1]
        .set(jnp.where(orientation >= 0.0, 1.0, -1.0))
    )
    row_rotation = left @ correction @ right.T
    matrix = row_rotation.T
    offset = target_center - source_center @ row_rotation
    return (
        matrix,
        offset,
        jnp.asarray(decomposition.singular_values),
        decomposition.successful,
    )


def register_spatial_points(
    moving: Any,
    fixed: Any,
    source_frame: SpatialFrame,
    target_frame: SpatialFrame,
    /,
    *,
    source_weights: Any | None = None,
    target_weights: Any | None = None,
    plan: SpatialRegistrationPlan | None = None,
) -> SpatialRegistrationResult:
    """Register two 2D/3D point clouds by alternating native OT and rigid geometry."""
    if not isinstance(source_frame, SpatialFrame) or not isinstance(
        target_frame, SpatialFrame
    ):
        raise TypeError("source_frame and target_frame must be SpatialFrame instances.")
    configured = SpatialRegistrationPlan() if plan is None else plan
    if not isinstance(configured, SpatialRegistrationPlan):
        raise TypeError("plan must be a SpatialRegistrationPlan or None.")
    source = jnp.asarray(moving, dtype=float)
    target = jnp.asarray(fixed, dtype=source.dtype)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("moving and fixed must have shape (point, coordinate).")
    dimension = int(source.shape[1])
    if dimension not in (2, 3) or int(target.shape[1]) != dimension:
        raise ValueError("Rigid spatial registration supports matching 2D or 3D frames.")
    if source_frame.dimension != dimension or target_frame.dimension != dimension:
        raise ValueError("Point dimensions must match their explicit spatial frames.")
    if int(source.shape[0]) < dimension or int(target.shape[0]) < dimension:
        raise ValueError("Rigid registration needs at least coordinate-dimension points.")
    if np.any(~np.isfinite(np.asarray(source))) or np.any(
        ~np.isfinite(np.asarray(target))
    ):
        raise ValueError("Registration point coordinates must be finite.")
    source_mass = _normalized_weights(
        "source_weights", source_weights, int(source.shape[0]), source.dtype
    )
    target_mass = _normalized_weights(
        "target_weights", target_weights, int(target.shape[0]), source.dtype
    )

    solver = Sinkhorn(
        configured.epsilon,
        max_iterations=configured.sinkhorn_iterations,
        min_iterations=1,
        tolerance=configured.sinkhorn_tolerance,
        check_every=5,
        early_stop=False,
    )
    matrix = jnp.eye(dimension, dtype=source.dtype)
    offset = jnp.sum(target_mass[:, None] * target, axis=0) - jnp.sum(
        source_mass[:, None] * source, axis=0
    )
    converged = jnp.asarray(False)
    all_transport_converged = jnp.asarray(True)
    first_converged = jnp.asarray(configured.outer_iterations, dtype=jnp.int32)
    transform_residual = jnp.asarray(jnp.inf, dtype=source.dtype)
    singular_values = jnp.zeros((dimension,), dtype=source.dtype)
    svd_success = jnp.asarray(True)
    coupling = jnp.zeros((int(source.shape[0]), int(target.shape[0])), dtype=source.dtype)
    marginal_residual = jnp.asarray(jnp.inf, dtype=source.dtype)

    for iteration in range(configured.outer_iterations):
        aligned = source @ matrix.T + offset
        source_measure = discrete(
            aligned,
            cx.Field(source_mass, dims=("point",)),
            axes="point",
            normalized=True,
            provenance="spatial-registration-source",
        )
        target_measure = discrete(
            target,
            cx.Field(target_mass, dims=("point",)),
            axes="point",
            normalized=True,
            provenance="spatial-registration-target",
        )
        problem = discrete_problem(
            source_measure,
            target_measure,
            cost=SquaredEuclideanCost(),
        )
        transport = solver(problem)
        candidate_coupling = transport.dense_plan()
        candidate_matrix, candidate_offset, candidate_singular, candidate_svd = (
            _rigid_update(
                source,
                target,
                candidate_coupling,
                source_mass,
                target_mass,
            )
        )
        delta = jnp.sqrt(
            jnp.sum((candidate_matrix - matrix) ** 2)
            + jnp.sum((candidate_offset - offset) ** 2)
        )
        step_converged = (
            (delta <= configured.transform_tolerance)
            & transport.converged
            & candidate_svd
        )
        first_converged = jnp.where(
            (~converged) & step_converged,
            iteration + 1,
            first_converged,
        )
        matrix = jnp.where(converged, matrix, candidate_matrix)
        offset = jnp.where(converged, offset, candidate_offset)
        coupling = jnp.where(converged, coupling, candidate_coupling)
        singular_values = jnp.where(converged, singular_values, candidate_singular)
        svd_success = jnp.where(converged, svd_success, candidate_svd)
        transform_residual = jnp.where(converged, transform_residual, delta)
        marginal_residual = jnp.where(
            converged,
            marginal_residual,
            transport.diagnostics.normalized_marginal_residual,
        )
        all_transport_converged = all_transport_converged & transport.converged
        converged = converged | step_converged

    aligned = source @ matrix.T + offset
    pair_residual = aligned[:, None, :] - target[None, :, :]
    residual_variance = jnp.sum(
        coupling * jnp.sum(pair_residual * pair_residual, axis=-1)
    )
    effective_size = 1.0 / jnp.maximum(jnp.sum(coupling * coupling), 1.0e-30)
    rotation_identifiability = jnp.min(singular_values) / jnp.maximum(
        jnp.max(singular_values), 1.0e-30
    )
    geometry_valid = svd_success & (rotation_identifiability >= configured.rank_tolerance)
    uncertainty_valid = converged & all_transport_converged & geometry_valid
    translation_standard_error = jnp.sqrt(
        residual_variance / jnp.maximum(effective_size - dimension, 1.0)
    )
    uncertainty = RegistrationUncertainty(
        residual_variance=residual_variance,
        translation_standard_error=translation_standard_error,
        coupling_effective_size=effective_size,
        rotation_identifiability=rotation_identifiability,
        valid=uncertainty_valid,
    )
    valid = converged & all_transport_converged & geometry_valid
    status = jnp.where(
        ~all_transport_converged,
        int(RegistrationStatus.TRANSPORT_NOT_CONVERGED),
        jnp.where(
            ~converged,
            int(RegistrationStatus.OUTER_NOT_CONVERGED),
            jnp.where(
                ~geometry_valid,
                int(RegistrationStatus.DEGENERATE_GEOMETRY),
                int(RegistrationStatus.OK),
            ),
        ),
    ).astype(jnp.int32)
    transform = AffineSpatialTransform._from_numeric(
        matrix,
        offset,
        jnp.asarray(source_frame.code, dtype=jnp.uint8),
        jnp.asarray(target_frame.code, dtype=jnp.uint8),
        0.0
        if source_frame.unit.micrometre_scale is None
        else source_frame.unit.micrometre_scale,
        0.0
        if target_frame.unit.micrometre_scale is None
        else target_frame.unit.micrometre_scale,
    )
    evidence = RegistrationEvidence(
        iterations=first_converged,
        transform_residual=transform_residual,
        transport_marginal_residual=marginal_residual,
        singular_values=singular_values,
        transport_converged=all_transport_converged,
    )
    return SpatialRegistrationResult(
        transform=transform,
        aligned_coordinates=aligned,
        coupling=coupling,
        uncertainty=uncertainty,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_registration_contract(configured),
    )


__all__ = [
    "RegistrationEvidence",
    "RegistrationStatus",
    "RegistrationUncertainty",
    "SpatialRegistrationPlan",
    "SpatialRegistrationResult",
    "register_spatial_points",
]
