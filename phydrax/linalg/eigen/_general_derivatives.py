#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._materialization import materialize
from .._operators import AbstractLinearOperator
from ._general import (
    _operator_coordinate_action,
    GeneralEigenSolveResult,
    GeneralEigenSolveStatus,
    PreparedGeneralEigenSolve,
)


class GeneralEigenvalueDerivativeStatus(IntEnum):
    """Portable status for a simple-mode or repeated-cluster derivative."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    INVALID_CLUSTER = 2
    NONFINITE_MODE = 3
    DEFECTIVE_CLUSTER = 4
    NONFINITE = 5


class GeneralInvariantProjectorDerivativeStatus(IntEnum):
    """Portable status for an isolated invariant-projector derivative."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    INCOMPLETE_SPECTRUM = 2
    NONFINITE_MODE = 3
    SINGULAR_MASS = 4
    DEFECTIVE_BASIS = 5
    EXTERNAL_GAP_TOO_SMALL = 6
    NONFINITE = 7
    RESIDUAL_TOLERANCE_NOT_MET = 8


class GeneralEigenvalueDerivativeDiagnostics(StrictModule):
    """Cluster spread, duality, conditioning, and finiteness evidence."""

    cluster_size: int = eqx.field(static=True)
    repeated_cluster: Array
    cluster_spread: Array
    cluster_scale: Array
    duality_error: Array
    overlap_condition: Array
    perturbation_norm: Array
    projected_derivative_norm: Array
    finite: Array


class GeneralEigenvalueDerivativeProvenance(StrictModule):
    """Source solve identities and analytical projected-perturbation convention."""

    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    method: str = eqx.field(static=True)
    cluster_indices: tuple[int, ...] = eqx.field(static=True)
    within_cluster_denominators: bool = eqx.field(static=True)
    numeric_version: Array


class GeneralEigenvalueDerivativeResult(StrictModule):
    """First-order simple derivative or basis-invariant repeated-cluster data."""

    projected_derivative: Array
    projected_eigenvalue_derivatives: Array
    scalar_derivative: Array | None
    trace_derivative: Array
    cluster_eigenvalue: Array
    status: Array
    diagnostics: GeneralEigenvalueDerivativeDiagnostics
    provenance: GeneralEigenvalueDerivativeProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(GeneralEigenvalueDerivativeStatus.SUCCESS)


class GeneralInvariantProjectorDerivativeDiagnostics(StrictModule):
    """External gaps and differentiated projector-identity residual evidence."""

    cluster_size: int = eqx.field(static=True)
    complement_size: int = eqx.field(static=True)
    minimum_external_gap: Array
    basis_condition: Array
    commutator_residual_norm: Array
    tangent_residual_norm: Array
    relative_residual: Array
    perturbation_norm: Array
    derivative_norm: Array
    finite: Array
    converged: Array


class GeneralInvariantProjectorDerivativeProvenance(StrictModule):
    """Source identities and cluster-to-complement derivative convention."""

    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    method: str = eqx.field(static=True)
    cluster_indices: tuple[int, ...] = eqx.field(static=True)
    complement_indices: tuple[int, ...] = eqx.field(static=True)
    denominator_scope: str = eqx.field(static=True)
    used_internal_denominators: bool = eqx.field(static=True)
    numeric_version: Array


class GeneralInvariantProjectorDerivativeResult(StrictModule):
    """Analytical first derivative of one isolated right invariant projector."""

    projector: Array
    value: Array
    external_denominators: Array
    status: Array
    diagnostics: GeneralInvariantProjectorDerivativeDiagnostics
    provenance: GeneralInvariantProjectorDerivativeProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(GeneralInvariantProjectorDerivativeStatus.SUCCESS)


def general_eigenvalue_derivative(
    prepared: PreparedGeneralEigenSolve,
    result: GeneralEigenSolveResult,
    operator_perturbation: AbstractLinearOperator | ArrayLike,
    mode_indices: int | Sequence[int],
    /,
    *,
    mass_perturbation: AbstractLinearOperator | ArrayLike | None = None,
) -> GeneralEigenvalueDerivativeResult:
    """Return analytical first-order data for a simple mode or repeated cluster.

    A repeated cluster returns the eigenvalues and trace of the projected
    perturbation ``Lᴴ (dA - lambda dB) R``.  These data are invariant under a
    simultaneous change of right basis and dual left basis; no spectral gap is
    formed within the cluster.
    """
    _require_source_identity(prepared, result)
    indices = _mode_indices(mode_indices, result.eigenvalues.size)
    if result.eigenvalues.size == 1 and len(indices) == 1:
        return _general_simple_eigenvalue_derivative(
            prepared,
            result,
            operator_perturbation,
            mass_perturbation,
        )
    if result.eigenvalues.size != prepared.problem.dimension:
        raise ValueError(
            "Cluster eigenvalue derivatives require the complete dense pencil spectrum."
        )
    _require_source_pair(prepared, result)
    matrix_perturbation = _perturbation_matrix(
        prepared,
        operator_perturbation,
        "operator_perturbation",
    )
    mass_matrix_perturbation = (
        np.zeros_like(np.asarray(prepared.mass_matrix))
        if mass_perturbation is None
        else _perturbation_matrix(prepared, mass_perturbation, "mass_perturbation")
    )
    all_values = np.asarray(result.eigenvalues)
    values = all_values[list(indices)]
    finite_modes = np.asarray(result.finite_mask)[list(indices)]
    scale = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    cluster_value = np.mean(values)
    spread = float(np.max(np.abs(values - cluster_value), initial=0.0))
    tolerance = prepared.plan.policy.tolerance
    repeated = len(indices) > 1
    complement = tuple(index for index in range(all_values.size) if index not in indices)
    external_collision = bool(
        complement
        and np.any(
            np.abs(all_values[list(complement)] - cluster_value)
            <= tolerance.cluster_relative * scale
        )
    )
    valid_cluster = (
        spread <= tolerance.cluster_relative * scale and not external_collision
    )
    finite_modes_only = bool(np.all(finite_modes) and np.all(np.isfinite(values)))
    right = np.asarray(result.right_eigenvector_coordinates)[:, list(indices)]
    left = np.asarray(result.left_eigenvector_coordinates)[:, list(indices)]
    mass = np.asarray(prepared.mass_matrix)
    overlap = np.conj(left.T) @ mass @ right
    overlap_rank = int(np.linalg.matrix_rank(overlap))
    overlap_condition = float(np.linalg.cond(overlap))
    overlap_pseudoinverse = scipy_linalg.lstsq(
        overlap,
        np.eye(overlap.shape[0], dtype=overlap.dtype),
        cond=1.0e-15,
        lapack_driver="gelsd",
    )[0]
    dual_left = left @ np.conj(overlap_pseudoinverse.T)
    duality_error = float(
        np.linalg.norm(
            np.conj(dual_left.T) @ mass @ right - np.eye(len(indices), dtype=right.dtype)
        )
    )
    projected = np.conj(dual_left.T) @ (
        matrix_perturbation @ right - mass_matrix_perturbation @ right @ np.diag(values)
    )
    projected_derivatives = np.linalg.eigvals(projected)
    trace_derivative = np.trace(projected)
    finite_output = bool(
        np.all(np.isfinite(projected))
        and np.all(np.isfinite(projected_derivatives))
        and np.isfinite(trace_derivative)
        and np.isfinite(duality_error)
    )
    source_success = bool(
        int(np.asarray(result.status)) == int(GeneralEigenSolveStatus.SUCCESS)
    )
    if not source_success:
        status = GeneralEigenvalueDerivativeStatus.SOURCE_FAILURE
    elif not finite_modes_only:
        status = GeneralEigenvalueDerivativeStatus.NONFINITE_MODE
    elif not valid_cluster:
        status = GeneralEigenvalueDerivativeStatus.INVALID_CLUSTER
    elif overlap_rank < len(indices):
        status = GeneralEigenvalueDerivativeStatus.DEFECTIVE_CLUSTER
    elif not finite_output:
        status = GeneralEigenvalueDerivativeStatus.NONFINITE
    else:
        status = GeneralEigenvalueDerivativeStatus.SUCCESS
    scalar = (
        projected[0, 0]
        if len(indices) == 1 and status == GeneralEigenvalueDerivativeStatus.SUCCESS
        else None
    )
    if (
        prepared.plan.policy.failure.mode == "error"
        and status != GeneralEigenvalueDerivativeStatus.SUCCESS
    ):
        raise RuntimeError(
            f"General eigenvalue derivative failed its contract: {status.name}."
        )
    dtype = result.eigenvalues.dtype
    return GeneralEigenvalueDerivativeResult(
        projected_derivative=jnp.asarray(projected, dtype=dtype),
        projected_eigenvalue_derivatives=jnp.asarray(
            projected_derivatives,
            dtype=dtype,
        ),
        scalar_derivative=(None if scalar is None else jnp.asarray(scalar, dtype=dtype)),
        trace_derivative=jnp.asarray(trace_derivative, dtype=dtype),
        cluster_eigenvalue=jnp.asarray(cluster_value, dtype=dtype),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        diagnostics=GeneralEigenvalueDerivativeDiagnostics(
            cluster_size=len(indices),
            repeated_cluster=jnp.asarray(repeated),
            cluster_spread=jnp.asarray(spread),
            cluster_scale=jnp.asarray(scale),
            duality_error=jnp.asarray(duality_error),
            overlap_condition=jnp.asarray(overlap_condition),
            perturbation_norm=jnp.asarray(
                np.linalg.norm(matrix_perturbation)
                + abs(cluster_value) * np.linalg.norm(mass_matrix_perturbation)
            ),
            projected_derivative_norm=jnp.asarray(np.linalg.norm(projected)),
            finite=jnp.asarray(finite_output),
        ),
        provenance=GeneralEigenvalueDerivativeProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            mass_operator_id=(
                None
                if prepared.problem.mass_operator is None
                else prepared.problem.mass_operator.operator_id
            ),
            method="dual projected first-order pencil perturbation",
            cluster_indices=indices,
            within_cluster_denominators=False,
            numeric_version=prepared.numeric_version,
        ),
    )


def _general_simple_eigenvalue_derivative(
    prepared: PreparedGeneralEigenSolve,
    result: GeneralEigenSolveResult,
    operator_perturbation: AbstractLinearOperator | ArrayLike,
    mass_perturbation: AbstractLinearOperator | ArrayLike | None,
    /,
) -> GeneralEigenvalueDerivativeResult:
    """Stage a targeted paired-eigenvector quotient without materialization."""
    value = result.eigenvalues[0]
    right = result.right_eigenvector_coordinates[:, 0]
    left = result.left_eigenvector_coordinates[:, 0]
    matrix_tangent = _perturbation_action(
        prepared,
        operator_perturbation,
        right,
        "operator_perturbation",
    )
    mass_tangent = (
        jnp.zeros_like(matrix_tangent)
        if mass_perturbation is None
        else _perturbation_action(
            prepared,
            mass_perturbation,
            right,
            "mass_perturbation",
        )
    )
    mass_action = (
        right
        if prepared.problem.mass_operator is None
        else _operator_coordinate_action(
            prepared.problem.mass_operator,
            right,
            adjoint_action=False,
        )
    )
    denominator = jnp.vdot(left, mass_action)
    numerator_action = matrix_tangent - value * mass_tangent
    numerator = jnp.vdot(left, numerator_action)
    real_dtype = jnp.real(value).dtype
    tiny = jnp.finfo(real_dtype).tiny
    nonsingular = jnp.abs(denominator) > tiny
    derivative = numerator / jnp.where(
        nonsingular,
        denominator,
        jnp.asarray(1, dtype=denominator.dtype),
    )
    source_success = (
        result.status == int(GeneralEigenSolveStatus.SUCCESS)
    ) & jnp.array_equal(
        result.provenance.numeric_version,
        prepared.numeric_version,
    )
    finite_mode = (
        result.finite_mask[0]
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(right))
        & jnp.all(jnp.isfinite(left))
    )
    finite_output = (
        jnp.isfinite(denominator) & jnp.isfinite(numerator) & jnp.isfinite(derivative)
    )
    status = jnp.asarray(
        int(GeneralEigenvalueDerivativeStatus.SUCCESS),
        dtype=jnp.int32,
    )
    status = jnp.where(
        ~nonsingular,
        int(GeneralEigenvalueDerivativeStatus.DEFECTIVE_CLUSTER),
        status,
    )
    status = jnp.where(
        ~finite_output,
        int(GeneralEigenvalueDerivativeStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        ~finite_mode,
        int(GeneralEigenvalueDerivativeStatus.NONFINITE_MODE),
        status,
    )
    status = jnp.where(
        ~source_success,
        int(GeneralEigenvalueDerivativeStatus.SOURCE_FAILURE),
        status,
    ).astype(jnp.int32)
    if prepared.plan.policy.failure.mode == "error":
        derivative = eqx.error_if(
            derivative,
            status != int(GeneralEigenvalueDerivativeStatus.SUCCESS),
            "General eigenvalue derivative failed its numerical contract.",
        )
    projected = derivative.reshape((1, 1))
    overlap_condition = jnp.where(nonsingular, 1.0, jnp.inf)
    perturbation_norm = jnp.linalg.norm(matrix_tangent) + jnp.abs(
        value
    ) * jnp.linalg.norm(mass_tangent)
    return GeneralEigenvalueDerivativeResult(
        projected_derivative=projected,
        projected_eigenvalue_derivatives=derivative.reshape((1,)),
        scalar_derivative=derivative,
        trace_derivative=derivative,
        cluster_eigenvalue=value,
        status=status,
        diagnostics=GeneralEigenvalueDerivativeDiagnostics(
            cluster_size=1,
            repeated_cluster=jnp.asarray(False),
            cluster_spread=jnp.asarray(0, dtype=real_dtype),
            cluster_scale=jnp.maximum(jnp.abs(value), 1),
            duality_error=jnp.where(
                nonsingular,
                jnp.abs(denominator / denominator - 1),
                jnp.inf,
            ),
            overlap_condition=overlap_condition,
            perturbation_norm=perturbation_norm,
            projected_derivative_norm=jnp.abs(derivative),
            finite=finite_output,
        ),
        provenance=GeneralEigenvalueDerivativeProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            mass_operator_id=(
                None
                if prepared.problem.mass_operator is None
                else prepared.problem.mass_operator.operator_id
            ),
            method="matrix-free paired simple-mode pencil quotient",
            cluster_indices=(0,),
            within_cluster_denominators=False,
            numeric_version=prepared.numeric_version,
        ),
    )


def _perturbation_action(
    prepared: PreparedGeneralEigenSolve,
    perturbation: AbstractLinearOperator | ArrayLike,
    vector: Array,
    name: str,
    /,
) -> Array:
    dimension = prepared.problem.dimension
    if isinstance(perturbation, AbstractLinearOperator):
        if (
            perturbation.batch_shape
            or not perturbation.source.compatible(prepared.problem.operator.source)
            or not perturbation.target.compatible(prepared.problem.operator.target)
        ):
            raise ValueError(f"{name} must be an unbatched operator on the pencil space.")
        return _operator_coordinate_action(
            perturbation,
            vector,
            adjoint_action=False,
        )
    value = jnp.asarray(perturbation)
    if value.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        raise TypeError(f"{name} must use an inexact dtype.")
    return value @ vector


def general_invariant_projector_derivative(
    prepared: PreparedGeneralEigenSolve,
    result: GeneralEigenSolveResult,
    operator_perturbation: AbstractLinearOperator | ArrayLike,
    mode_indices: int | Sequence[int],
    /,
    *,
    mass_perturbation: AbstractLinearOperator | ArrayLike | None = None,
) -> GeneralInvariantProjectorDerivativeResult:
    """Differentiate an isolated right invariant projector using external gaps only.

    For a generalized regular pencil, the projector is that of ``B^{-1} A``.
    Every denominator couples the requested cluster to its complement.  Repeated
    eigenvalues inside the cluster are never divided by one another.
    """
    _require_source_pair(prepared, result)
    dimension = prepared.problem.dimension
    indices = _mode_indices(mode_indices, result.eigenvalues.size)
    complement = tuple(index for index in range(dimension) if index not in indices)
    matrix_perturbation = _perturbation_matrix(
        prepared,
        operator_perturbation,
        "operator_perturbation",
    )
    mass_matrix_perturbation = (
        np.zeros_like(np.asarray(prepared.mass_matrix))
        if mass_perturbation is None
        else _perturbation_matrix(prepared, mass_perturbation, "mass_perturbation")
    )
    values = np.asarray(result.eigenvalues)
    right = np.asarray(result.right_eigenvector_coordinates)
    matrix = np.asarray(prepared.matrix)
    mass = np.asarray(prepared.mass_matrix)
    complete = values.size == dimension and right.shape == (dimension, dimension)
    finite_modes = bool(
        complete
        and np.all(np.asarray(result.finite_mask))
        and np.all(np.isfinite(values))
    )
    mass_rank = int(np.asarray(prepared.mass_rank))
    source_success = bool(
        int(np.asarray(result.status)) == int(GeneralEigenSolveStatus.SUCCESS)
    )
    if complete:
        basis_rank = int(np.linalg.matrix_rank(right))
        basis_condition = float(np.linalg.cond(right))
        dual = scipy_linalg.lstsq(
            right,
            np.eye(right.shape[0], dtype=right.dtype),
            cond=1.0e-15,
            lapack_driver="gelsd",
        )[0]
    else:
        basis_rank = 0
        basis_condition = np.inf
        dual = np.zeros(
            (dimension, dimension), dtype=np.result_type(right.dtype, np.complex64)
        )
    cluster_list = list(indices)
    complement_list = list(complement)
    right_cluster = right[:, cluster_list] if complete else dual[:, : len(indices)]
    right_complement = (
        right[:, complement_list] if complete else dual[:, : len(complement)]
    )
    dual_cluster = dual[cluster_list, :]
    dual_complement = dual[complement_list, :]
    projector = right_cluster @ dual_cluster
    if complement:
        denominators = (
            values[cluster_list, None] - values[None, complement_list]
            if complete
            else np.full((len(indices), len(complement)), np.nan)
        )
        minimum_gap = float(np.min(np.abs(denominators), initial=np.inf))
    else:
        denominators = np.zeros(
            (len(indices), 0), dtype=np.result_type(values.dtype, np.complex64)
        )
        minimum_gap = np.inf
    scale = max(
        float(np.max(np.abs(values[cluster_list]), initial=0.0)) if complete else 0.0,
        float(np.max(np.abs(values[complement_list]), initial=0.0))
        if complete and complement
        else 0.0,
        1.0,
    )
    gap_ok = minimum_gap > prepared.plan.policy.tolerance.cluster_relative * scale
    if (
        complete
        and finite_modes
        and mass_rank == dimension
        and basis_rank == dimension
        and gap_ok
    ):
        pencil_operator = np.linalg.solve(mass, matrix)
        effective_perturbation = np.linalg.solve(
            mass,
            matrix_perturbation - mass_matrix_perturbation @ pencil_operator,
        )
        lower_numerator = dual_complement @ effective_perturbation @ right_cluster
        upper_numerator = dual_cluster @ effective_perturbation @ right_complement
        lower_coefficients = lower_numerator / denominators.T
        upper_coefficients = upper_numerator / denominators
        derivative = (
            right_complement @ lower_coefficients @ dual_cluster
            + right_cluster @ upper_coefficients @ dual_complement
        )
        commutator_target = (
            projector @ effective_perturbation - effective_perturbation @ projector
        )
        commutator_residual = np.linalg.norm(
            pencil_operator @ derivative
            - derivative @ pencil_operator
            - commutator_target
        )
        tangent_residual = np.linalg.norm(
            projector @ derivative + derivative @ projector - derivative
        )
        perturbation_norm = np.linalg.norm(effective_perturbation)
    else:
        derivative = np.full_like(projector, np.nan)
        commutator_residual = np.nan
        tangent_residual = np.nan
        perturbation_norm = np.linalg.norm(matrix_perturbation) + np.linalg.norm(
            mass_matrix_perturbation
        )
    derivative_norm = np.linalg.norm(derivative)
    residual_scale = (
        np.linalg.norm(matrix) * derivative_norm
        + np.linalg.norm(projector) * perturbation_norm
    )
    tiny = np.finfo(np.asarray(matrix).real.dtype).tiny
    relative_residual = (commutator_residual + tangent_residual) / max(
        residual_scale,
        tiny,
    )
    residual_tolerance = (
        prepared.plan.policy.tolerance.absolute
        + prepared.plan.policy.tolerance.relative * residual_scale
    )
    residual_ok = bool(
        np.isfinite(commutator_residual)
        and np.isfinite(tangent_residual)
        and commutator_residual + tangent_residual <= residual_tolerance
    )
    finite_output = bool(
        np.all(np.isfinite(projector))
        and np.all(np.isfinite(derivative))
        and np.all(np.isfinite(denominators))
        and np.isfinite(relative_residual)
    )
    if not source_success:
        status = GeneralInvariantProjectorDerivativeStatus.SOURCE_FAILURE
    elif not complete:
        status = GeneralInvariantProjectorDerivativeStatus.INCOMPLETE_SPECTRUM
    elif not finite_modes:
        status = GeneralInvariantProjectorDerivativeStatus.NONFINITE_MODE
    elif mass_rank < dimension:
        status = GeneralInvariantProjectorDerivativeStatus.SINGULAR_MASS
    elif basis_rank < dimension:
        status = GeneralInvariantProjectorDerivativeStatus.DEFECTIVE_BASIS
    elif not gap_ok:
        status = GeneralInvariantProjectorDerivativeStatus.EXTERNAL_GAP_TOO_SMALL
    elif not finite_output:
        status = GeneralInvariantProjectorDerivativeStatus.NONFINITE
    elif not residual_ok:
        status = GeneralInvariantProjectorDerivativeStatus.RESIDUAL_TOLERANCE_NOT_MET
    else:
        status = GeneralInvariantProjectorDerivativeStatus.SUCCESS
    if (
        prepared.plan.policy.failure.mode == "error"
        and status != GeneralInvariantProjectorDerivativeStatus.SUCCESS
    ):
        raise RuntimeError(
            f"General invariant-projector derivative failed its contract: {status.name}."
        )
    dtype = result.eigenvalues.dtype
    return GeneralInvariantProjectorDerivativeResult(
        projector=jnp.asarray(projector, dtype=dtype),
        value=jnp.asarray(derivative, dtype=dtype),
        external_denominators=jnp.asarray(denominators, dtype=dtype),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        diagnostics=GeneralInvariantProjectorDerivativeDiagnostics(
            cluster_size=len(indices),
            complement_size=len(complement),
            minimum_external_gap=jnp.asarray(minimum_gap),
            basis_condition=jnp.asarray(basis_condition),
            commutator_residual_norm=jnp.asarray(commutator_residual),
            tangent_residual_norm=jnp.asarray(tangent_residual),
            relative_residual=jnp.asarray(relative_residual),
            perturbation_norm=jnp.asarray(perturbation_norm),
            derivative_norm=jnp.asarray(derivative_norm),
            finite=jnp.asarray(finite_output),
            converged=jnp.asarray(
                status == GeneralInvariantProjectorDerivativeStatus.SUCCESS
            ),
        ),
        provenance=GeneralInvariantProjectorDerivativeProvenance(
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            mass_operator_id=(
                None
                if prepared.problem.mass_operator is None
                else prepared.problem.mass_operator.operator_id
            ),
            method="biorthogonal cluster-to-complement eigenprojector derivative",
            cluster_indices=indices,
            complement_indices=complement,
            denominator_scope="selected cluster to complement only",
            used_internal_denominators=False,
            numeric_version=prepared.numeric_version,
        ),
    )


def _require_source_pair(
    prepared: PreparedGeneralEigenSolve,
    result: GeneralEigenSolveResult,
    /,
) -> None:
    _require_source_identity(prepared, result)
    if prepared.matrix.shape != (
        prepared.problem.dimension,
        prepared.problem.dimension,
    ):
        raise ValueError(
            "Cluster and projector derivatives require DenseSchurQZ preparation."
        )
    if int(np.asarray(result.provenance.numeric_version)) != int(
        np.asarray(prepared.numeric_version)
    ):
        raise ValueError("result does not belong to the prepared numerical pencil.")


def _require_source_identity(
    prepared: PreparedGeneralEigenSolve,
    result: GeneralEigenSolveResult,
    /,
) -> None:
    if not isinstance(prepared, PreparedGeneralEigenSolve):
        raise TypeError("prepared must be a PreparedGeneralEigenSolve.")
    if not isinstance(result, GeneralEigenSolveResult):
        raise TypeError("result must be a GeneralEigenSolveResult.")
    provenance = result.provenance
    if (
        provenance.problem_id != prepared.problem.problem_id
        or provenance.plan_id != prepared.plan.plan_id
        or provenance.prepared_id != prepared.prepared_id
    ):
        raise ValueError("result does not belong to the prepared numerical pencil.")


def _mode_indices(value: int | Sequence[int], size: int, /) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        indices = (int(value),)
    else:
        indices = tuple(int(index) for index in value)
    if not indices:
        raise ValueError("mode_indices must be nonempty.")
    if len(set(indices)) != len(indices):
        raise ValueError("mode_indices must not contain duplicates.")
    if any(index < 0 or index >= size for index in indices):
        raise IndexError("mode_indices contain an out-of-range mode.")
    return tuple(sorted(indices))


def _perturbation_matrix(
    prepared: PreparedGeneralEigenSolve,
    perturbation: AbstractLinearOperator | ArrayLike,
    name: str,
    /,
) -> np.ndarray:
    dimension = prepared.problem.dimension
    if isinstance(perturbation, AbstractLinearOperator):
        if (
            perturbation.batch_shape
            or not perturbation.source.compatible(prepared.problem.operator.source)
            or not perturbation.target.compatible(prepared.problem.operator.target)
        ):
            raise ValueError(f"{name} must be an unbatched operator on the pencil space.")
        value = np.asarray(
            materialize(perturbation, prepared.plan.policy.materialization)
        )
    else:
        value = np.asarray(perturbation)
    if value.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}.")
    if not np.issubdtype(value.dtype, np.inexact):
        raise TypeError(f"{name} must use an inexact dtype.")
    return value


__all__ = [
    "GeneralEigenvalueDerivativeDiagnostics",
    "GeneralEigenvalueDerivativeProvenance",
    "GeneralEigenvalueDerivativeResult",
    "GeneralEigenvalueDerivativeStatus",
    "GeneralInvariantProjectorDerivativeDiagnostics",
    "GeneralInvariantProjectorDerivativeProvenance",
    "GeneralInvariantProjectorDerivativeResult",
    "GeneralInvariantProjectorDerivativeStatus",
    "general_eigenvalue_derivative",
    "general_invariant_projector_derivative",
]
