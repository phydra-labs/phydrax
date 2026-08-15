#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._materialization import materialize
from .._operators import AbstractLinearOperator
from .._policies import FailurePolicy
from .._spaces import _coordinate_pairing_matrix
from ._problems import Eigenproblem
from ._self_adjoint_spectrum import (
    prepare_self_adjoint_spectrum,
    PreparedSelfAdjointSpectrum,
    SelfAdjointSpectrumPlan,
    SelfAdjointSpectrumPolicy,
    SelfAdjointSpectrumStatus,
)
from ._spectral_derivatives import (
    attach_density_derivative,
    attach_projector_derivative,
    density_from_projector,
    projector_from_selection,
)
from ._subspaces import SpectralSelection


SelfAdjointSubspaceDifferentiation: TypeAlias = Literal["none", "projector"]


class SelfAdjointSpectralSubspaceStatus(IntEnum):
    """Status of one fixed-shape self-adjoint spectral cluster."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2
    SELECTION_DIMENSION_MISMATCH = 3
    BOUNDARY_UNRESOLVED = 4
    CLUSTER_NOT_ISOLATED = 5
    PROJECTOR_RESIDUAL_TOO_LARGE = 6
    DIFFERENTIATION_REJECTED = 7


class SelfAdjointSpectralDerivativeStatus(IntEnum):
    """Status of one explicit projector and density-kernel derivative."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2
    RESIDUAL_TOO_LARGE = 3


class SelfAdjointSpectralSubspacePolicy(StrictModule):
    """Selection, isolation, differentiation, and failure requirements."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    minimum_external_gap: float = eqx.field(static=True)
    differentiation: SelfAdjointSubspaceDifferentiation = eqx.field(static=True)
    failure: FailurePolicy

    def __init__(
        self,
        *,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        minimum_external_gap: float = 0.0,
        differentiation: SelfAdjointSubspaceDifferentiation = "none",
        failure: FailurePolicy | None = None,
    ):
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        gap = float(minimum_external_gap)
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (relative, absolute, gap)
        ):
            raise ValueError("Subspace tolerances and gaps must be finite and non-negative.")
        if differentiation not in ("none", "projector"):
            raise ValueError("differentiation must be 'none' or 'projector'.")
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.minimum_external_gap = gap
        self.differentiation = differentiation
        self.failure = failure_


class SelfAdjointSpectralSubspaceDiagnostics(StrictModule):
    """Selection, isolation, projector, and metric-orthogonality evidence."""

    selected_count: Array
    expected_count: int = eqx.field(static=True)
    boundary_distance: Array
    external_gap: Array
    certified_external_gap: Array
    differentiation_cutoff: Array
    idempotence_error: Array
    commutator_error: Array
    metric_self_adjointness_error: Array
    basis_orthogonality_error: Array
    finite: Array
    converged: Array


class SelfAdjointSpectralSubspaceProvenance(StrictModule):
    """Source spectrum, selection, and differentiation identity."""

    problem_id: str = eqx.field(static=True)
    spectrum_plan_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    differentiation: SelfAdjointSubspaceDifferentiation = eqx.field(static=True)
    method: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class SelfAdjointSpectralSubspace(StrictModule):
    """A fixed-shape metric spectral projector and contravariant density kernel."""

    selected_eigenvalues: Array
    complement_eigenvalues: Array
    basis: Array
    dual_basis: Array
    projector: Array
    density_kernel: Array
    status: Array
    diagnostics: SelfAdjointSpectralSubspaceDiagnostics
    provenance: SelfAdjointSpectralSubspaceProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SelfAdjointSpectralSubspaceStatus.SUCCESS)

    def project_coordinates(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != (self.projector.shape[1],):
            raise ValueError("vector must match the projector coordinate dimension.")
        return self.projector @ value


class SelfAdjointSpectralDerivativeDiagnostics(StrictModule):
    """Cross-gap, commutator, tangent, and density-identity residuals."""

    selected_to_complement_residual_norm: Array
    complement_to_selected_residual_norm: Array
    commutator_residual_norm: Array
    tangent_residual_norm: Array
    density_identity_residual_norm: Array
    perturbation_norm: Array
    projector_derivative_norm: Array
    density_derivative_norm: Array
    relative_residual: Array
    finite: Array
    converged: Array


class SelfAdjointSpectralDerivativeProvenance(StrictModule):
    """Source spectrum and derivative-method identity."""

    problem_id: str = eqx.field(static=True)
    spectrum_plan_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class SelfAdjointSpectralDerivativeResult(StrictModule):
    """Directional derivatives of an isolated metric projector and density kernel."""

    projector: Array
    density_kernel: Array
    status: Array
    diagnostics: SelfAdjointSpectralDerivativeDiagnostics
    provenance: SelfAdjointSpectralDerivativeProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SelfAdjointSpectralDerivativeStatus.SUCCESS)


def self_adjoint_spectral_subspace(
    spectrum_or_problem,
    selection: SpectralSelection,
    /,
    *,
    policy: SelfAdjointSpectralSubspacePolicy | None = None,
    spectrum_policy: SelfAdjointSpectrumPolicy | SelfAdjointSpectrumPlan | None = None,
) -> SelfAdjointSpectralSubspace:
    """Construct an isolated self-adjoint projector from one reusable spectrum."""
    if not isinstance(selection, SpectralSelection):
        raise TypeError("selection must be a SpectralSelection.")
    selected_policy = (
        SelfAdjointSpectralSubspacePolicy() if policy is None else policy
    )
    if not isinstance(selected_policy, SelfAdjointSpectralSubspacePolicy):
        raise TypeError("policy must be a SelfAdjointSpectralSubspacePolicy or None.")
    if isinstance(spectrum_or_problem, PreparedSelfAdjointSpectrum):
        if spectrum_policy is not None:
            raise ValueError("spectrum_policy must be omitted for prepared spectrum state.")
        spectrum = spectrum_or_problem
    else:
        spectrum = prepare_self_adjoint_spectrum(spectrum_or_problem, spectrum_policy)
    expected = selection.expected_dimension
    if expected is None:
        raise ValueError("Self-adjoint spectral selections require expected_dimension.")
    if expected > spectrum.problem.dimension:
        raise ValueError("expected_dimension cannot exceed the spectrum dimension.")

    values = spectrum.eigenvalues
    observed_mask = selection.mask(values)
    observed_count = jnp.sum(observed_mask, axis=-1, dtype=jnp.int32)
    order = jnp.argsort(~observed_mask, axis=-1, stable=True)
    ordered_values = jnp.take_along_axis(values, order, axis=-1)
    ordered_vectors = jnp.take_along_axis(
        spectrum.eigenvectors,
        order[..., None, :],
        axis=-1,
    )
    ordered_inverse = jnp.take_along_axis(
        spectrum.inverse_basis,
        order[..., :, None],
        axis=-2,
    )
    n = spectrum.problem.dimension
    selected_mask = jnp.arange(n) < expected
    projector = projector_from_selection(
        ordered_vectors,
        ordered_inverse,
        selected_mask,
    )
    density = density_from_projector(projector, spectrum.paired_metric)
    diagnostics, status, differentiation_valid = _subspace_evidence(
        spectrum,
        selection,
        selected_policy,
        ordered_values,
        ordered_vectors,
        ordered_inverse,
        selected_mask,
        observed_count,
        projector,
    )
    if selected_policy.differentiation == "projector":
        projector = jax.lax.cond(
            jnp.all(differentiation_valid),
            lambda value: attach_projector_derivative(
                spectrum.problem,
                value,
                ordered_values,
                ordered_vectors,
                ordered_inverse,
                selected_mask,
            ),
            jax.lax.stop_gradient,
            projector,
        )
        density = jax.lax.cond(
            jnp.all(differentiation_valid),
            lambda value: attach_density_derivative(
                spectrum.problem,
                value,
                jax.lax.stop_gradient(projector),
                spectrum.paired_metric,
                ordered_values,
                ordered_vectors,
                ordered_inverse,
                selected_mask,
            ),
            jax.lax.stop_gradient,
            density,
        )
    else:
        projector = jax.lax.stop_gradient(projector)
        density = jax.lax.stop_gradient(density)
    if selected_policy.failure.mode == "error":
        projector = eqx.error_if(
            projector,
            jnp.any(status != int(SelfAdjointSpectralSubspaceStatus.SUCCESS)),
            "Self-adjoint spectral subspace did not satisfy its numerical contract.",
        )
    return SelfAdjointSpectralSubspace(
        selected_eigenvalues=ordered_values[..., :expected],
        complement_eigenvalues=ordered_values[..., expected:],
        basis=ordered_vectors[..., :, :expected],
        dual_basis=jnp.conj(
            jnp.swapaxes(ordered_inverse[..., :expected, :], -1, -2)
        ),
        projector=projector,
        density_kernel=density,
        status=status,
        diagnostics=diagnostics,
        provenance=SelfAdjointSpectralSubspaceProvenance(
            problem_id=spectrum.problem.problem_id,
            spectrum_plan_id=spectrum.plan.plan_id,
            selection_id=selection.selection_id,
            differentiation=selected_policy.differentiation,
            method="full-dense metric spectral projector",
            numeric_version=spectrum.eigen_prepared.numeric_version,
        ),
    )


def self_adjoint_spectral_projector_derivative(
    spectrum: PreparedSelfAdjointSpectrum,
    selection: SpectralSelection,
    operator_perturbation: AbstractLinearOperator | ArrayLike,
    metric_perturbation: AbstractLinearOperator | ArrayLike | None = None,
    /,
    *,
    policy: SelfAdjointSpectralSubspacePolicy | None = None,
) -> SelfAdjointSpectralDerivativeResult:
    """Evaluate exact directional derivatives for one isolated spectral cluster."""
    if not isinstance(spectrum, PreparedSelfAdjointSpectrum):
        raise TypeError("spectrum must be a PreparedSelfAdjointSpectrum.")
    selected_policy = (
        SelfAdjointSpectralSubspacePolicy() if policy is None else policy
    )
    subspace = self_adjoint_spectral_subspace(
        spectrum,
        selection,
        policy=SelfAdjointSpectralSubspacePolicy(
            relative_tolerance=selected_policy.relative_tolerance,
            absolute_tolerance=selected_policy.absolute_tolerance,
            minimum_external_gap=selected_policy.minimum_external_gap,
            differentiation="none",
            failure=selected_policy.failure,
        ),
    )
    expected = selection.expected_dimension
    if expected is None:
        raise ValueError("Self-adjoint spectral selections require expected_dimension.")
    values = spectrum.eigenvalues
    observed_mask = selection.mask(values)
    order = jnp.argsort(~observed_mask, stable=True)
    ordered_values = jnp.take_along_axis(values, order, axis=-1)
    vectors = jnp.take_along_axis(
        spectrum.eigenvectors,
        order[..., None, :],
        axis=-1,
    )
    inverse_basis = jnp.take_along_axis(
        spectrum.inverse_basis,
        order[..., :, None],
        axis=-2,
    )
    n = spectrum.problem.dimension
    selected_mask = jnp.arange(n) < expected
    operator_matrix = _perturbation_matrix(
        spectrum,
        operator_perturbation,
        "operator_perturbation",
    )
    if metric_perturbation is None:
        metric_matrix = jnp.zeros_like(operator_matrix)
    else:
        if isinstance(spectrum.problem, Eigenproblem):
            raise ValueError("metric_perturbation requires a generalized eigenproblem.")
        metric_matrix = _perturbation_matrix(
            spectrum,
            metric_perturbation,
            "metric_perturbation",
        )
    pairing = _coordinate_pairing_matrix(spectrum.problem.operator.source)
    residual_images = (
        operator_matrix @ vectors
        - (metric_matrix @ vectors) * ordered_values[..., None, :]
    )
    perturbation = (
        jnp.conj(jnp.swapaxes(vectors, -1, -2))
        @ pairing
        @ residual_images
    )
    paired_metric_tangent = pairing @ metric_matrix
    selected = selected_mask.astype(vectors.dtype)
    membership_difference = selected[:, None] - selected[None, :]
    gaps = (
        ordered_values[..., :, None].astype(vectors.dtype)
        - ordered_values[..., None, :].astype(vectors.dtype)
    )
    cross = membership_difference != 0
    safe_gaps = jnp.where(cross, gaps, 1)
    derivative_in_basis = jnp.where(
        cross,
        membership_difference * perturbation / safe_gaps,
        0,
    )
    projector_derivative = vectors @ derivative_in_basis @ inverse_basis
    density_derivative = jnp.swapaxes(
        jnp.linalg.solve(
            jnp.swapaxes(spectrum.paired_metric, -1, -2),
            jnp.swapaxes(
                projector_derivative
                - subspace.density_kernel @ paired_metric_tangent,
                -1,
                -2,
            ),
        ),
        -1,
        -2,
    )
    diagnostics, status = _derivative_evidence(
        spectrum,
        subspace,
        operator_matrix,
        metric_matrix,
        ordered_values,
        vectors,
        inverse_basis,
        selected_mask,
        perturbation,
        derivative_in_basis,
        projector_derivative,
        density_derivative,
        paired_metric_tangent,
        expected,
        selected_policy,
    )
    if selected_policy.failure.mode == "error":
        projector_derivative = eqx.error_if(
            projector_derivative,
            jnp.any(status != int(SelfAdjointSpectralDerivativeStatus.SUCCESS)),
            "Self-adjoint projector derivative did not satisfy its contract.",
        )
    return SelfAdjointSpectralDerivativeResult(
        projector=projector_derivative,
        density_kernel=density_derivative,
        status=status,
        diagnostics=diagnostics,
        provenance=SelfAdjointSpectralDerivativeProvenance(
            problem_id=spectrum.problem.problem_id,
            spectrum_plan_id=spectrum.plan.plan_id,
            selection_id=selection.selection_id,
            method="exact cross-cluster Sylvester derivative",
            numeric_version=spectrum.eigen_prepared.numeric_version,
        ),
    )


def _subspace_evidence(
    spectrum,
    selection,
    policy,
    values,
    vectors,
    inverse_basis,
    selected_mask,
    observed_count,
    projector,
):
    n = spectrum.problem.dimension
    expected = selection.expected_dimension
    if expected is None:
        raise ValueError("expected_dimension is required.")
    batch_shape = values.shape[:-1]
    selected_values = values[..., :expected]
    complement_values = values[..., expected:]
    if expected == n:
        external_gap = jnp.full(batch_shape, jnp.inf, dtype=values.dtype)
        certified_gap = external_gap
    else:
        distances = jnp.abs(
            selected_values[..., :, None]
            - complement_values[..., None, :]
        )
        order = jnp.argsort(
            ~selection.mask(spectrum.eigenvalues),
            axis=-1,
            stable=True,
        )
        ordered_residuals = jnp.take_along_axis(
            spectrum.source_diagnostics.residual_norms,
            order,
            axis=-1,
        )
        ordered_relative = jnp.take_along_axis(
            spectrum.source_diagnostics.relative_residuals,
            order,
            axis=-1,
        )
        scales = jnp.maximum(jnp.abs(values), 1)
        uncertainty = 4 * jnp.maximum(
            ordered_residuals,
            ordered_relative * scales,
        )
        certified_distances = (
            distances
            - uncertainty[..., :expected, None]
            - uncertainty[..., None, expected:]
        )
        external_gap = jnp.min(distances, axis=(-2, -1))
        certified_gap = jnp.min(certified_distances, axis=(-2, -1))
    boundary_distance = jnp.min(
        selection.boundary_distance(values),
        axis=-1,
    )
    scale = jnp.maximum(jnp.max(jnp.abs(values), axis=-1), 1)
    cutoff = jnp.sqrt(jnp.finfo(values.dtype).eps) * max(n, 1) * scale
    idempotence = jnp.linalg.norm(
        projector @ projector - projector,
        axis=(-2, -1),
    )
    spectral_operator = (
        vectors * values.astype(vectors.dtype)[..., None, :]
    ) @ inverse_basis
    commutator = jnp.linalg.norm(
        spectral_operator @ projector - projector @ spectral_operator,
        axis=(-2, -1),
    )
    metric_adjointness = jnp.linalg.norm(
        jnp.conj(jnp.swapaxes(projector, -1, -2)) @ spectrum.paired_metric
        - spectrum.paired_metric @ projector,
        axis=(-2, -1),
    )
    basis = vectors[..., :, :expected]
    basis_orthogonality = jnp.linalg.norm(
        jnp.conj(jnp.swapaxes(basis, -1, -2))
        @ spectrum.paired_metric
        @ basis
        - jnp.eye(expected, dtype=vectors.dtype),
        axis=(-2, -1),
    )
    residual_scale = (
        jnp.linalg.norm(projector, axis=(-2, -1))
        + jnp.linalg.norm(spectral_operator, axis=(-2, -1))
        + 1
    )
    residual_tolerance = (
        policy.absolute_tolerance + policy.relative_tolerance * residual_scale
    )
    residual_ok = (
        idempotence + commutator + metric_adjointness + basis_orthogonality
        <= residual_tolerance
    )
    finite = (
        jnp.all(jnp.isfinite(projector), axis=(-2, -1))
        & jnp.isfinite(idempotence)
        & jnp.isfinite(commutator)
        & jnp.isfinite(metric_adjointness)
        & jnp.isfinite(basis_orthogonality)
    )
    source_success = spectrum.status == int(SelfAdjointSpectrumStatus.SUCCESS)
    dimension_matches = observed_count == expected
    boundary_resolved = boundary_distance > cutoff
    isolated = certified_gap > jnp.maximum(cutoff, policy.minimum_external_gap)
    differentiation_valid = (
        source_success
        & dimension_matches
        & boundary_resolved
        & isolated
        & residual_ok
        & finite
    )
    status = jnp.where(
        ~finite,
        int(SelfAdjointSpectralSubspaceStatus.NONFINITE),
        jnp.where(
            ~source_success,
            int(SelfAdjointSpectralSubspaceStatus.SOURCE_FAILURE),
            jnp.where(
                ~dimension_matches,
                int(SelfAdjointSpectralSubspaceStatus.SELECTION_DIMENSION_MISMATCH),
                jnp.where(
                    ~boundary_resolved,
                    int(SelfAdjointSpectralSubspaceStatus.BOUNDARY_UNRESOLVED),
                    jnp.where(
                        ~isolated,
                        int(SelfAdjointSpectralSubspaceStatus.CLUSTER_NOT_ISOLATED),
                        jnp.where(
                            ~residual_ok,
                            int(
                                SelfAdjointSpectralSubspaceStatus.PROJECTOR_RESIDUAL_TOO_LARGE
                            ),
                            int(SelfAdjointSpectralSubspaceStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = SelfAdjointSpectralSubspaceDiagnostics(
        selected_count=observed_count,
        expected_count=expected,
        boundary_distance=boundary_distance,
        external_gap=external_gap,
        certified_external_gap=certified_gap,
        differentiation_cutoff=cutoff,
        idempotence_error=idempotence,
        commutator_error=commutator,
        metric_self_adjointness_error=metric_adjointness,
        basis_orthogonality_error=basis_orthogonality,
        finite=finite,
        converged=status == int(SelfAdjointSpectralSubspaceStatus.SUCCESS),
    )
    return diagnostics, status, differentiation_valid

def _derivative_evidence(
    spectrum,
    subspace,
    operator_matrix,
    metric_matrix,
    values,
    vectors,
    inverse_basis,
    selected_mask,
    perturbation,
    derivative_in_basis,
    projector_derivative,
    density_derivative,
    paired_metric_tangent,
    expected,
    policy,
):
    expected = int(expected)
    membership = selected_mask.astype(vectors.dtype)
    membership_difference = membership[:, None] - membership[None, :]
    gaps = (
        values[..., :, None].astype(vectors.dtype)
        - values[..., None, :].astype(vectors.dtype)
    )
    sylvester_residual = (
        gaps * derivative_in_basis
        - membership_difference * perturbation
    )
    selected_to_complement = jnp.linalg.norm(
        sylvester_residual[..., :expected, expected:],
        axis=(-2, -1),
    )
    complement_to_selected = jnp.linalg.norm(
        sylvester_residual[..., expected:, :expected],
        axis=(-2, -1),
    )
    spectral_operator = (
        vectors * values.astype(vectors.dtype)[..., None, :]
    ) @ inverse_basis
    operator_perturbation = vectors @ perturbation @ inverse_basis
    commutator = jnp.linalg.norm(
        spectral_operator @ projector_derivative
        - projector_derivative @ spectral_operator
        - subspace.projector @ operator_perturbation
        + operator_perturbation @ subspace.projector,
        axis=(-2, -1),
    )
    tangent = jnp.linalg.norm(
        subspace.projector @ projector_derivative
        + projector_derivative @ subspace.projector
        - projector_derivative,
        axis=(-2, -1),
    )
    density_identity = jnp.linalg.norm(
        projector_derivative
        - density_derivative @ spectrum.paired_metric
        - subspace.density_kernel @ paired_metric_tangent,
        axis=(-2, -1),
    )
    perturbation_norm = (
        jnp.linalg.norm(operator_matrix, axis=(-2, -1))
        + jnp.linalg.norm(metric_matrix, axis=(-2, -1))
    )
    projector_norm = jnp.linalg.norm(
        projector_derivative,
        axis=(-2, -1),
    )
    density_norm = jnp.linalg.norm(
        density_derivative,
        axis=(-2, -1),
    )
    residual = (
        selected_to_complement
        + complement_to_selected
        + commutator
        + tangent
        + density_identity
    )
    scale = (
        perturbation_norm
        + projector_norm
        + density_norm
        + jnp.linalg.norm(spectral_operator, axis=(-2, -1)) * projector_norm
        + 1
    )
    relative = residual / scale
    tolerance = policy.absolute_tolerance + policy.relative_tolerance * scale
    finite = (
        jnp.all(jnp.isfinite(projector_derivative), axis=(-2, -1))
        & jnp.all(jnp.isfinite(density_derivative), axis=(-2, -1))
        & jnp.isfinite(residual)
    )
    source_success = (
        subspace.status == int(SelfAdjointSpectralSubspaceStatus.SUCCESS)
    )
    residual_ok = residual <= tolerance
    status = jnp.where(
        ~finite,
        int(SelfAdjointSpectralDerivativeStatus.NONFINITE),
        jnp.where(
            ~source_success,
            int(SelfAdjointSpectralDerivativeStatus.SOURCE_FAILURE),
            jnp.where(
                ~residual_ok,
                int(SelfAdjointSpectralDerivativeStatus.RESIDUAL_TOO_LARGE),
                int(SelfAdjointSpectralDerivativeStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = SelfAdjointSpectralDerivativeDiagnostics(
        selected_to_complement_residual_norm=selected_to_complement,
        complement_to_selected_residual_norm=complement_to_selected,
        commutator_residual_norm=commutator,
        tangent_residual_norm=tangent,
        density_identity_residual_norm=density_identity,
        perturbation_norm=perturbation_norm,
        projector_derivative_norm=projector_norm,
        density_derivative_norm=density_norm,
        relative_residual=relative,
        finite=finite,
        converged=status == int(SelfAdjointSpectralDerivativeStatus.SUCCESS),
    )
    return diagnostics, status


def _perturbation_matrix(
    spectrum,
    perturbation,
    name,
):
    n = spectrum.problem.dimension
    if isinstance(perturbation, AbstractLinearOperator):
        if not perturbation.source.compatible(spectrum.problem.operator.source):
            raise ValueError(f"{name} source space must match the spectrum.")
        if not perturbation.target.compatible(spectrum.problem.operator.target):
            raise ValueError(f"{name} target space must match the spectrum.")
        matrix = materialize(perturbation, spectrum.plan.policy.materialization)
    else:
        matrix = jnp.asarray(perturbation)
    expected = spectrum.problem.batch_shape + (n, n)
    if matrix.shape == (n, n) and spectrum.problem.batch_shape:
        matrix = jnp.broadcast_to(matrix, expected)
    elif matrix.shape != expected:
        raise ValueError(f"{name} must have shape {(n, n)} or {expected}.")
    if matrix.dtype != spectrum.eigenvectors.dtype:
        raise TypeError(f"{name} dtype must match the spectrum coordinates.")
    return matrix


__all__ = [
    "SelfAdjointSpectralDerivativeDiagnostics",
    "SelfAdjointSpectralDerivativeProvenance",
    "SelfAdjointSpectralDerivativeResult",
    "SelfAdjointSpectralDerivativeStatus",
    "SelfAdjointSpectralSubspace",
    "SelfAdjointSpectralSubspaceDiagnostics",
    "SelfAdjointSpectralSubspacePolicy",
    "SelfAdjointSpectralSubspaceProvenance",
    "SelfAdjointSpectralSubspaceStatus",
    "SelfAdjointSubspaceDifferentiation",
    "self_adjoint_spectral_projector_derivative",
    "self_adjoint_spectral_subspace",
]
